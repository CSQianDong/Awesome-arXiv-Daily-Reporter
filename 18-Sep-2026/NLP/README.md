# Embedding Models Measure in Peculiar Ways 

**Authors**: Juri Opitz, Andrianos Michail  

**Link**: [PDF](https://arxiv.org/pdf/2609.20821)  

**Abstract**: Embedding spaces define notions of semantic similarity and distance. We study whether those embeddings reflect physical measurements of mass, distance, time and volume, which admit a unique, objective notion of semantic equivalence and distance. We find that physical measurement is only weakly modeled in the embedding space, and that instead quite peculiar measurement patterns can be observed. Further analysis indicates that embedding representations of physical measurements are strongly influenced by superficial string similarity, and recalibration of similarity does not substantially improve the alignment. 

---
# Unifying Models of Intergroup Hostility in Online Discourse 

**Authors**: Patrick Gerard, Julia Mendelsohn, Kristina Lerman  

**Link**: [PDF](https://arxiv.org/pdf/2609.20808)  

**Abstract**: Hostile rhetoric toward social groups can normalize exclusion and justify mistreatment, as well as contribute to rising polarization and political violence. Efforts to moderate hostile rhetoric in online speech draw on foundational theories in social and moral psychology, and political science. However, these theories were developed largely in parallel, often propose different and sometimes conflicting accounts of how hostility develops, and have rarely been tested against each other in real discourse. The result is a fragmented understanding of the rhetorical mechanisms of hostility, without a clear sense of how they appear, and relate to each other, in real-world discourse. Using 2.86 million posts from TikTok, Truth Social, and Twitter/X during the 2024 U.S. presidential election, we model the mechanisms of six foundational theories of intergroup hostility -- boundary construction, threat construction, scapegoating, negative evaluation, dehumanization, and action orientation -- within a common empirical framework to recover the broader organization of intergroup hostility rhetoric. Structurally, we find that boundary construction and threat construction anchor the system; temporally, we find that these mechanisms tend to follow a regular ordering: boundary construction, derogation, and action orientation tend to appear early; dehumanization and threat construction later; scapegoating latest. Mapping how these theoretical frameworks actually manifest in discourse bridges longstanding divisions across social science traditions and presents computational social science with a clearer empirical foundation for modeling intergroup hostility rhetoric beyond single-label detection. 

---
# JEPA-Anything: Learning Predictive Models across Different Worlds 

**Authors**: Taoyong Cui, Zhongyao Wang, Xinyue Xu, Weiyang Liu, Zhaochen Yu, Yuying Zhang, Qiang Gao, Mengyue Yang, Wanli Ouyang, Pheng Ann Heng, Yingcheng Wu, Zhenfei Yin, Ling Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20800)  

**Abstract**: World modeling enables intelligence to anticipate consequences, guide interventions, and learn from interaction. Yet predictive models remain domain-specific: can a common learning principle support world modeling across radically different systems? We introduce JEPA-Anything, a domain-agnostic framework based on orthogonal predictive factorization (OPF). Extending joint-embedding predictive architectures, OPF decomposes latent targets into complementary factors, learns them through dedicated pathways, and recombines them within a shared predictive design. We evaluate JEPA-Anything across seven domains: vision, biology, clinical trajectories, control, molecular dynamics, physical fields, and weather. Experiments span representation learning, intervention prediction, out-of-distribution generalization, and long-horizon dynamics, including 10 matched dynamics tasks, forecasting of over 1,000 clinical events, and 100-step molecular rollouts across four systems. Against matched JEPA baselines, JEPA-Anything improves reported metrics on all 10 dynamics tasks and reduces single-intervention prediction error on Interventional Pong by 34.8%. It achieves the lowest one-step and 100-step molecular errors among compared methods in all four systems. Beyond prediction, a factor-nominated biological intervention receives experimental support in cell co-cultures, patient-derived organoids, tumor fragments, and mice; latent orbital modes recover the Keplerian scaling exponent with a fitted slope of -1.4991. These results support a common factorized predictive principle across heterogeneous worlds, connecting world modeling with intervention and experimentally grounded scientific discovery. Code: this https URL 

---
# RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning 

**Authors**: Yan Yu, Zhengxi Lu, Yizhou Liu, Yichen Pan, Aozhe Wang, Qipeng Chen, Hua Yang, Wenqi Zhang, Weiming Lu, Qianglong Chen, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.20784)  

**Abstract**: Multi-turn agents trained with reinforcement learning (RL) receive a single scalar reward per trajectory, which motivates self on-policy distillation (OPD) to supply dense token-level supervision from a self-teacher with privileged task skills, letting a skill-free student internalize them. This recipe, however, is undermined by two findings in agentic tasks: privileged information alone does not always make a teacher reliable, and the benefit of teacher supervision is stage-dependent. We therefore propose RetireOPD (Self-Retiring On-Policy Distillation), which first optimizes a decoupled, skill-conditioned teacher with environment rewards and then trains a skill-free student jointly with RL and OPD. Rather than following a predefined distillation schedule, RetireOPD adopts Adaptive Retirement: the student drops the teacher on its own once their discrepancy stops shrinking and it reaches a target fraction of the teacher's success rate, after which training proceeds with RL alone. Across Qwen2.5 models from 1.5B to 7B, RetireOPD improves ALFWorld success rate over RL baseline by 14.1% to 18.8% and WebShop accuracy by 11.8% to 19.0%, and surpasses its own skill-conditioned teacher in every setting. 

---
# Harm Laundering in GPT Models: Evidence That Gender Discrimination Is Transformed Rather Than Reduced Across Safety-Trained Generations 

**Authors**: Sarah Wyer, Sue Black, Noura Al Moubayed  

**Link**: [PDF](https://arxiv.org/pdf/2609.20779)  

**Abstract**: Safety evaluations for large language models rely on surface-form classifiers that report declining harm scores across model generations. We provide evidence that this methodology is systematically incomplete: explicit discriminatory content is transformed rather than removed. We call this \emph{harm laundering}. Analysing 450,000 gender-directed completions across 15 models spanning GPT-2 through to GPT-5 (OpenAI GPT lineage; three demographic conditions), we show that sexual violence clusters prevalent in GPT-2 women-directed output disappear by GPT-4, while men-directed completions gain positive representational territory (caregiving, emotional range, ally identity) that women-directed completions do not. The pattern is most visible at GPT-5: Topic~5 (1,997~documents) frames breast cancer as a men's rights debate, while zero equivalent clusters appear in women-directed output. Three independent classifiers score this content as non-toxic. Sentiment scores invert at GPT-4: early models demean women; later models over-correct. Topic diversity in women-directed completions falls 36\% relative to men at the GPT-4 alignment boundary (W/M~$= 0.58$, from $0.91$ at GPT-2). REGARD representational harm disparity correlates with release date ($\rho = +0.55$, $p = .034$) while Detoxify does not ($\rho = -0.23$, $p = .42$): toxicity scores fall as representational harm grows. We formalise harm laundering as a three-criteria test and provide a three-stage detection protocol applicable to any generative model. Within the OpenAI GPT lineage, toxicity score reduction is not a sufficient proxy for harm reduction. 

---
# dQwen3.5: Hybrid-Attention Diffusion Language Models 

**Authors**: Anton Xue, Litu Rout, Aditya Akella, Adam Klivans, Sujay Sanghavi, Sanjay Shakkottai  

**Link**: [PDF](https://arxiv.org/pdf/2609.20751)  

**Abstract**: Adapting a pretrained autoregressive (AR) model is a cost-efficient route to a diffusion language model (DLM). While nearly all such adaptations start from a full-attention transformer, AR modeling has shifted toward hybrid architectures that interleave attention and RNN layers. This creates an obstacle for adaptation: unlike attention, RNNs are structurally causal and nontrivial to bidirectionalize. Despite this mismatch, we investigate whether such backbones can become effective DLMs by adapting Qwen3.5 at 0.8B, 2B, 4B, and 9B scales, yielding the dQwen3.5 family. We find that hybrid backbones can be efficient starting points for adaptation: against a full-attention control, the hybrid reaches a given training loss in about half the tokens. Across scales, dQwen3.5 resembles full-attention DLMs in any-order decoding behavior and performs strongly under parallel decoding. 

---
# On-Demand Attention: Language Models Know When to Recall 

**Authors**: Haibo Feng, Ruiqi Liang, Hanyang Peng, Shiqi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2609.20734)  

**Abstract**: Reasoning and agentic workloads increasingly demand efficient long-context inference. Yet full-attention decoding reads the growing history at every step, regardless of its benefit to the next prediction. We show that a pretrained model's decoding states already contain information predictive of this benefit, before the global read. Building on this finding, we introduce On-Demand Attention (ODA), a local-first decoding method that uses a lightweight recall head to selectively invoke global attention as its predicted benefit changes during generation. ODA trains only the recall head, leaving pretrained weights unchanged and the complete historical KV cache available for future recall. We further implement GPU-side conditional execution in vLLM, translating reduced global reads into practical decoding speedups over full attention at long context lengths. Experiments across Qwen and Gemma models, including hybrid-attention backbones, show that selective recall recovers most of the performance lost under local attention while substantially reducing global reads. These findings support long-context inference in which pretrained models guide their own access to the information they retain. 

---
# Summarization Bias: The Directional Collapse of Objective Projection into Told-Mode Labels in Large Language Models --- A Conceptual Framework and Registered Test Protocol 

**Authors**: Levent Bulut  

**Link**: [PDF](https://arxiv.org/pdf/2609.20712)  

**Abstract**: This paper introduces and operationalizes summarization bias: a proposed systematic tendency of large language models (LLMs) to represent narrative meaning as an abstract summary label rather than as the reconstructable inferential structure that produces it. Within the Bulut Doctrine, narrative effect is theorized along a told-shown axis: in told mode, emotional and informational content is declared explicitly and requires little reader reconstruction; in shown mode, that content is suppressed at the surface and must be reconstructed from physical cues and indirection (Objective Projection). Shown mode is the higher-load condition the doctrine is designed to measure.
The claim is that LLMs fail along this axis in a specific direction. Summarization bias is hypothesized to operate in two regimes: (i) a generative regime, in which a model asked to render an emotion through Objective Projection defaults to declaring it instead; and (ii) an evaluative regime, in which a model judging narrative quality rewards told-mode explicitness and under-detects shown-mode suppression. The evaluative regime is the more consequential, since LLMs increasingly serve as judges and reward models, and a directional bias toward told mode would impose a selection pressure degrading prose toward flat declaration.
This report does not claim the bias is validated. It defines the construct, situates it against LLM-as-judge biases, rereads a completed independent reliability study as directional evidence consistent with it, and pre-registers a two-regime test with decision rules under which the construct would be abandoned. 

---
# HerHealthEval: Evaluating Multilingual and Register-Sensitive Understanding of Women's Health Communication 

**Authors**: Hassan Saeed Hassan Albattra, Mazen Mohammed Bahgat, Rahatara Ferdousi, Hana Essam Sayed Ahmed Amrya, Mariam Mousa  

**Link**: [PDF](https://arxiv.org/pdf/2609.20684)  

**Abstract**: Large language models are increasingly used in healthcare communication, yet most evaluations emphasize response quality while assuming that the user's concern has been interpreted correctly. We introduce HerHealthEval, a controlled evaluation framework for multilingual understanding of women's-health communication. For each clinical case, HerHealthEval provides matched versions in English, French, and Modern Standard Arabic using six communicative forms: canonical, clinical, layperson, indirect or hedged, emotionally concerned, and deliberately under-specified. The first five express the same underlying concern and retain the same clinical information, whereas the under-specified form intentionally omits relevant details to test whether the model recognizes that clarification is needed. We evaluate a multilingual instruction model and QLoRA-adapted variants on concern classification, risk calibration, clarification behavior, parse compliance, and cross-form consistency. Results reveal that aggregate accuracy and consistency can conceal safety-relevant failures. A multilingual adaptation model reaches 0.994 under-triage in French and Arabic under language-asymmetric risk supervision. A controlled re-adaptation using source-derived, language-invariant risk labels reduces under-triage to 0.572 and 0.558, respectively. These findings show that robust multilingual healthcare evaluation requires explicit testing of register variation, uncertainty handling, and the provenance and invariance of adaptation labels. 

---
# UniPolicy: Unified Objective-Specific Policies for Generative Search Advertising 

**Authors**: Kun Yao, Yuhang Zhou, Yichi Zhang, Zeliang Tong, Shengri Xue, Haitao Wang, Siyu Lu, Qianlong Xie, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20630)  

**Abstract**: Search advertising connects user intent with commercial content and plays a critical role in platform monetization. Recent systems typically align pretrained generative models with a single business reward, such as eCPM, or use naive reward fusion for preliminary multi-objective alignment. However, an ideal search advertising system must jointly account for heterogeneous objectives, including relevance, click propensity, and commercial value, to balance user experience and business value while mitigating globally suboptimal performance caused by gradient competition. We propose UniPolicy, an objective-aware multi-policy alignment framework. UniPolicy combines objective-specific prefix tokens, sparse MoE-LoRA routing, and objective-specific residual FFNs to hierarchically decouple parameters within a shared backbone, providing differentiated parameter and policy-expression spaces for different business objectives. It further constructs pairwise preferences from multi-stage behavioral feedback, supplementing the relative preference information in exposed-but-unclicked samples and strengthening the relative advantage of clicked candidates in the generation distribution. At inference, UniPolicy supports parallel, business-customizable multi-policy beam search, flexibly allocating candidate quotas across objectives under a fixed retrieval budget. Large-scale offline experiments show that UniPolicy delivers balanced improvements across multiple metrics while preserving retrieval quality, outperforming single-objective reinforcement learning and naive reward-fusion baselines. In a 7-day online A/B test on a real search advertising system, UniPolicy improves CTR by 0.71%, RPS by 1.58%, and advertising revenue by 1.32%, while maintaining stable serving latency. 

---
# Chronicle: Cut-Point Replay for Regression Testing of LLM Agents 

**Authors**: Tisha Chawla, Susheem Koul  

**Link**: [PDF](https://arxiv.org/pdf/2609.20625)  

**Abstract**: Large language model responses are non-deterministic, so failures in LLM agents are hard to reproduce: a failure depends on inference that is not bitwise reproducible, on tools that read changing state, and on a multi-step trajectory that a re-run rarely repeats. Record-and-replay makes a run reproducible, but existing agent tooling records runs only to trace or score them, not to test a code change against them. We present Chronicle, which records an agent run at its non-deterministic boundaries as immutable envelopes and replays it from the record. Its central operation, cut-point replay, serves a chosen subset of boundaries from the record and executes the complementary subset live with new code, turning a recorded incident into a regression test that runs in continuous integration. On a benchmark of 6 recorded failures with simulated model boundaries, recording adds 23 {\mu}s per crossing (0.008% of an assumed 300 ms model call), full replay issues zero model calls and is bit-stable across 20 repetitions, and cut-point tests fail on faulty code and pass on guarded and benign changes for all 6 incidents. In a mutation study of the guarded tools, cut-point tests catch every mutant that lets the recorded unsafe action through, while a baseline that stubs every boundary, using the same assertion, catches none. Chronicle and the benchmark are publicly available at this https URL. 

---
# What Does Privileged Information Add to On-Policy Self-Distillation? 

**Authors**: XiuYu Zhang, Wei Chow, Junfeng Fang, Zhenkai Liang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2609.20612)  

**Abstract**: On-policy self-distillation (OPSD) lets a language model learn from a frozen copy of itself that sees an answer or a worked solution. Giving the teacher this extra information seems to offer the student more to learn, but how much does it add beyond distillation itself? To isolate that contribution, we construct AMPLE-Math, a reusable suite of 5,319 mathematical problems with six reasoning views that share the same answer, and compare each view with matched reference-free distillation. With a thinking-enabled teacher supervising direct-response rollouts, reference-free distillation accounts for much of Qwen3-1.7B's improvement under thinking-enabled evaluation, both in domain and on external benchmarks. Evidence for an additional reference benefit is modest in Qwen, strongest for a polished solution, whereas complete traces add two percentage points in SmolLM3-3B at step 50. These benefits depend on the student being trained. At the same checkpoint, replacing short direct-response rollouts with long thinking-enabled rollouts turns gains into losses in both families while the problems, references, and evaluation stay fixed. Teacher profiles and matched loss interventions in Qwen further show that changing token-level supervision can leave student behavior largely unchanged. Together, these findings suggest that OPSD can improve access to existing reasoning capabilities through parameters shared by direct-response and thinking-enabled inference. The value of a privileged reference is what it adds to this cross-mode transfer, not how much of the solution it reveals. 

---
# WiC is Not WSD: A Study on LLMs and Lexical Ambiguity Resolution 

**Authors**: Yi Zhou, Kiamehr Rezaee, Danushka Bollegala, Mohammad Taher Pilehvar, Jose Camacho-Collados  

**Link**: [PDF](https://arxiv.org/pdf/2609.20593)  

**Abstract**: Word-in-Context (WiC) remains challenging for language models, despite recent progress on lexical-semantic tasks. We hypothesise that this difficulty arises not only from comparing two contextual uses of a word, but also from the absence of an explicit sense inventory that specifies the relevant level of semantic granularity. We evaluate open LLMs on WiC and traditional Word Sense Disambiguation (WSD) under similar settings. We find that providing candidate senses, similar to what is done in traditional WSD, improves WiC performance in all settings. In general, explicit sense information helps models make more consistent and targeted judgements. Human evaluation further shows that many apparent WiC errors reflect label ambiguity or mismatches between model and annotator sense boundaries rather than simple failures of lexical understanding. In particular, results show that LLMs overthink the sense distinction often leading to errors based on overly fine-grained distinctions. 

---
# SAFARI: An Industrial Benchmark for LLM-Assisted Hazard Analysis and Risk Assessment 

**Authors**: Chenxi Wu, Zimu Wang, Haiyang Zhang, Wei Wang, Zhijie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.20584)  

**Abstract**: Large language models (LLMs) are increasingly considered for safety-critical engineering, yet their reliability in regulated functional-safety workflows remains underexplored. We introduce SAFARI (Safety-Aware Functional Automotive Risk Inference), the first industrial benchmark for LLM-assisted automotive Hazard Analysis and Risk Assessment (HARA) under ISO 26262. It contains 3,000 de-identified industrial HARA cases and evaluates two coupled tasks: open-ended hazard analysis and standards-grounded risk assessment. To evaluate open-ended HARA artifacts, we propose the first reference-anchored LLM-as-a-judge protocol with high expert correlation. Experiments with nine frontier LLMs show that models often produce plausible hazard narratives but remain weak at ISO 26262 risk classification, with the best ASIL macro-F1 reaching only 0.261. Chain-of-Thought prompting provides limited benefit and often degrades categorical risk assessment. Error analysis further localizes major failures to scenario-critical context omissions during hazard generation and to controllability misjudgments during risk assessment, indicating where expert oversight should be concentrated. The dataset can be obtained from this https URL. 

---
# Steering the Compass: Aligning Dynamic Psychological Counseling Conversations with Cognitive Behavioral Therapy Strategies 

**Authors**: Zimu Wang, Yiwen Jiang, Xiangyu Zhao, Yaling Shen, Jiahe Liu, Stephanie Fong, Maxmartwell H Cheng, Guilherme C Oliveira, Anh Nguyen, Robert Desimone, Barnaby Nelson, Dominic Dwyer, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2609.20565)  

**Abstract**: Recent advancements in large language models have revolutionized the field of psychological counseling, especially in the context of Cognitive Behavioral Therapy (CBT). While the success of CBT relies heavily on dynamic decision-making informed by the client's real-time mental state, this aspect has often been overlooked in current research, limiting both flexibility and therapeutic outcomes. In this paper, we introduce StratCBT, a dataset specifically designed for psychological counseling conversations with CBT Strategies, consisting of 9,688 sessions and around 256K utterances, with each counselor's response aligned with one of eight distinct strategies. The creation of StratCBT involves modeling clients based on their negative thoughts and generating high-quality counseling conversations through self-chat, incorporating realistic sessions as guidance, thereby significantly surpassing existing datasets in both general counseling and CBT-specific skills. We conduct extensive experiments to demonstrate the effectiveness of strategy-aligned generation and evaluate its efficacy in delivering professional and effective counseling with LLM-simulated clients to reflect real-world scenarios. The dataset can be obtained from this https URL. 

---
# An Analysis of Training-Free Self-Reported Confidence in Language Models 

**Authors**: Lukas Meyer, Sofia Rossi, Wei Chen, Thomas Laurent, Yiming Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.20541)  

**Abstract**: Large language models can report a numerical confidence together with generated content, but it is unclear whether this report is more than calibrated rhetoric. We analyze three training-free signals: confidence verbalized with the answer, post-hoc $P(\mathrm{True})$, and agreement with three additional generations on the same 100 TriviaQA questions for two model families. Direct verbalization is a surprisingly strong baseline: after auditing benchmark errors, it reaches AUROC 0.956 and 0.937 for correctness prediction. Three-sample agreement is substantially weaker (0.765 and 0.790), and a fixed interpolation with verbalized confidence has no statistically reliable benefit. Four of nine errors from one model and two of eight from the other receive unanimous sample support, showing that self-consistency can amplify shared misconceptions. Re-eliciting confidence for the same fixed answers with equivalent prompts changes scores by 0.043 to 0.084 on average and flips 4\% to 9\% of decisions at a 0.8 threshold. An exploratory audit of 100 confidence-tagged biography claims further finds only a modest confidence gap between supported and contradicted claims. These results argue that useful self-reports remain sensitive to elicitation, correlated errors, and benchmark noise. 

---
# Relational Attention for Data-Efficient Language Modeling 

**Authors**: Adrian Brasoveanu, Ece Takmaz, Jakub Dotlačil  

**Link**: [PDF](https://arxiv.org/pdf/2609.20530)  

**Abstract**: We present Relational BabyLM, a system submission to the BabyLM 2026 challenge that combines two cognitively motivated inductive biases in a single decoder-only Transformer. Architecturally, we replace standard self-attention with a Dual Attention Transformer (DAT), which separates the routing of object-level ("sensory") lexical features from structural/relational information (Altabaa and Lafferty, 2025; Altabaa et al., 2024; Webb et al., 2024; Kerg et al., 2022; Webb et al., 2021). Relational attention (RA) disentangled from self-attention greatly increases data efficiency and out-of-training-sample generalization on purely relational tasks, but language modeling requires object-level and relational information to be integrated as well as disentangled, and RA-based LMs have remained largely unexplored. BabyLM's data-constrained training and comprehensive evaluation is an ideal testing ground for whether that data efficiency transfers. As a training intervention, we add a Next-Latent Prediction (NextLat; Teoh et al. 2026) objective that encourages hidden states to compress history incrementally into a dense belief state. Architecture is the dominant factor for structural linguistic generalization; the objective is secondary but still significant. DAT's three relational attention types (full RA vs. the simpler RCA and DisRCA variants) are largely interchangeable at 10M words; full RA pulls ahead at 100M. We also introduce a novel symbol-retrieval mechanism (RoPE-based, as opposed to learned, relative symbols) that matches learned symbol libraries while adding no parameters. On the strict (100M-word) track, our best model ranks 6th of 55 overall and 3rd of 55 on the leaderboard's NLP-task subset at the time of writing; our two strongest models outperform the GPT-2 baseline on most benchmarks, with one attaining the highest EWoK score among strict-track entries. 

---
# Edustories: A Collection of Real-world Case Studies from Classroom Practices 

**Authors**: Michal Štefánik, Jan Nehyba, Jirina Karasova, Martin Fico, Lucie Škarková, Markéta Košatková, David Kosatka  

**Link**: [PDF](https://arxiv.org/pdf/2609.20484)  

**Abstract**: Despite the widely recognized potential of AI in education, most prior work has focused on individualized student assistance. In contrast, the majority of educational practice worldwide still takes place in collective classroom settings. To enable researchers to study AI assistance in collective teaching, we introduce Edustories, a dataset of 1,492 teacher-written case studies describing real elementary and high-school classroom situations involving challenging student behavior, pedagogical interventions, and their outcomes. Among many other applications, Edustories enables evaluating LLMs' ability to predict the success of teacher interventions, crucial for providing practicing teachers with useful feedback. Comparing the latest models from four language-model families against expert assessments, we find that current models fall short of human expertise in predicting classroom outcomes; the strongest models reach 58% accuracy compared to 64% of human experts. This gap highlights both the limitations and the emerging potential of AI as assistants for practicing teachers. 

---
# Stress-testing Alignment Midtraining 

**Authors**: Sid Baines, Jonathan Bostock, Maria Angelica Martinez, Andrew Draganov, David Africa, Daniel Tan  

**Link**: [PDF](https://arxiv.org/pdf/2609.20412)  

**Abstract**: When aligning frontier models through post-training techniques, it is not possible to directly demonstrate all of the behaviours we want a model to exhibit in all possible deployment environments; our model must generalise outside of the post-training distribution. One proposed solution is alignment midtraining (AMT), which continues pretraining on large volumes of alignment-relevant documents to encourage generalisation in later stages of training.
Despite the prominence of AMT as an alignment approach, there is limited public evidence for its effectiveness. To resolve this, we identify several assumptions around midtraining and evaluate them across scale: up to 110 billion-parameter models and 1 billion midtraining tokens. For instance, we study a scenario where post-training data is ambiguous between two possible motivations. We find that midtraining can steer the model's motivation in simple versions of this setting. However, the presence of a tiny fraction of finetuning data which suggests a competing motivation erases the effects of AMT. We also study scenarios in which we want an AI to follow a number of rules, but only demonstrate a subset of them. We find that demonstrations must be present either in midtraining or post-training datasets for these rules to be robustly learned.
Based on these and other findings, we do not believe that there is sufficient public evidence for us to confidently state that midtraining can address the core difficulties inherent in aligning powerful AI systems. 

---
# Xeno-Interpretability: Investigating the Alien Minds of LLMs 

**Authors**: F. Pierucci, M. Bracale Syrnikov, M. Prandi, M. Galisai, F. Giarrusso, P. Bisconti  

**Link**: [PDF](https://arxiv.org/pdf/2609.20408)  

**Abstract**: Large language models are usually interpreted through concepts that humans already possess: truthfulness, refusal, deception, personality, harmfulness, and related categories. This paper asks whether models may also represent and use distinctions for which no adequate human concept exists. We call such internal structures xeno-representations, and their study xeno-interpretability. We distinguish the human-interpretable semantic space from the xeno-semantic space: the region of model-native representations for which no adequate human conceptual counterpart is available. We show that the space of possible internal distinctions in an LLM is substantially larger than the space available through finite human descriptions. We then separate experimental identification from semantic interpretation: an internal representation may be reproducibly located, geometrically characterized, causally manipulated, and linked to downstream behaviour even when its semantic content cannot be adequately expressed in human terms. On this basis, we sketch an empirical programme to identify xeno-representations. We finally examine the implications for AI safety and multi-agent systems, where model-native representations may propagate and stabilize across interacting agents while remaining only partially visible through human-readable communication. Xeno-interpretability therefore shifts the aim of interpretability from finding human concepts inside models toward discovering and characterizing the representational structures that are native to the models themselves and might affect their behaviour in unpredictable ways. 

---
# Schema-Anchored Latent Reasoning for Semantic Parsing-Based Knowledge Base Question Answering 

**Authors**: Guangze Gao, Zixuan Li, Sikui Zhang, Chunfeng Yuan, Wenjuan Li, Bing Li, Xiaolong Jin, Weiming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2609.20398)  

**Abstract**: Semantic parsing (SP)-based knowledge base question answering aims to answer natural language questions by generating executable logical forms (LFs) over knowledge bases (KBs). When applying Large Language Models (LLMs) to this task, a key challenge over large, heterogeneous KBs is selecting question-related schema elements (i.e., relations and classes) and composing them into complex LFs. Recent LLM-based methods often make early discrete commitments to schema elements during intermediate reasoning, allowing incorrect intermediate schema decisions to propagate and finally result in incorrect LFs. To overcome this limitation, we propose SALR, a schema-anchored latent reasoning method for LF construction. It performs multi-step reasoning by generating continuous thoughts in the model's hidden states, thereby delaying the explicit commitment to LF decisions. To ground this latent reasoning process in the corresponding KB schema, SALR aligns continuous thoughts with a codebook of KB schema elements through an alignment objective supervised by schema traces deterministically derived from gold LFs. It then incorporates the aligned schema codes into inputs for subsequent reasoning steps. This schema-mediated feedback guides LF generation without requiring the model to emit an explicit textual reasoning trajectory. Experiments on GrailQA and WebQSP show that SALR achieves consistent overall gains over strong baselines. Notably, on compositional questions from GrailQA, SALR outperforms TIARA, a strong SP-based baseline, by 2.86 F1 points. Further analyses show that schema-mediated feedback affects LF generation and that schema information is recoverable from the latent states. 

---
# Viveka-Insight: a cross-lingual concept graph and citation-grounded retrieval resource over the complete works of Swami Vivekananda in English and Bengali 

**Authors**: Tamal Maharaj  

**Link**: [PDF](https://arxiv.org/pdf/2609.20303)  

**Abstract**: Classical philosophical corpora pose three compounding challenges for language resources: they exist in several languages without parallel alignment, their vocabulary is remote from that of contemporary readers, and generated text over culturally sensitive material must be verifiably grounded. We present Viveka-Insight, a bilingual resource and open-source pipeline for the works of Swami Vivekananda (1863-1902): the nine-volume English Complete Works and the ten-volume Bengali Vani o Rachana, two related but non-parallel corpora of about 15 million characters. Four layers are released: (i) a structure-preserving parse (32,694 paragraphs, 168,842 sentences) with per-paragraph anchors deep-linking into the published editions; (ii) a cross-lingual concept graph of 8,362 language-agnostic concepts with 87,518 relation-typed paragraph-concept and 55,872 concept-concept edges, in which canonical English labels act as a string-equality key linking Bengali and English passages with no parallel data; (iii) a bilingual alias inventory of 60,850 surface forms (30,053 English, 30,797 Bengali); and (iv) a human-annotated set of 200 paragraph-concept edges judged by three annotators, released with all per-annotator judgments. We report known-item cross-lingual retrieval over 194 verified rendered lecture pairs (Recall@10 0.86 in both directions), a 30-question audit of citation integrity and modern-question bridging, and a human study placing concept-extraction precision at 0.60 under strict two-annotator consensus (Cohen's kappa = 0.61). The extractor's confidence weight is calibrated: restricting to weight >= 0.8 raises precision to 0.71 while retaining 98% of concept-bearing paragraphs. Precision is markedly lower in Bengali than English (0.54 vs 0.68), locating the weakness in exactly the half that cross-lingual access depends on. The design transfers to other multilingual classical corpora. 

---
# Lens: Bringing the Right Semantic Perspective into Focus for Training-Free Multimodal Representation Learning 

**Authors**: Xinran Liu, Shouqian Shi, Yixian Chen, Ruizhi Chen, Xin-Wei Yao, Sheng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2609.20252)  

**Abstract**: High-quality representations are essential for a wide range of downstream tasks. Dedicated embedding models are explicitly optimized for representation learning, yet their training data are often more limited in scale and diversity than the massive corpora used to pretrain modern large language models and multimodal large language models. Large-scale pretraining and instruction following enable autoregressive models to select relevant evidence, integrate multimodal information, and infer semantics under different task perspectives, creating a distinctive opportunity for training-free representation learning. However, our analysis reveals that existing semantic-elicitation methods do not reliably orient the extracted states toward the semantic perspective required by the downstream task. Consequently, the resulting representations often remain dominated by salient input content. We characterize this problem as semantic perspective misalignment and propose Lens, a training-free framework that makes representation readout task-directed. Semantic Perspective Anchoring associates the task-required perspective with a task-specific readout phrase, specifying the interpretive role of the positions later used for extraction. Contextualized Phrase Readout places the same phrase after the complete input and aggregates its token states, combining full-context access with the anchored perspective. The resulting representation reflects task-conditioned evidence integration and inference rather than a generic summary of salient content. Without parameter updates, architectural modification, or reranking, Lens achieves an overall Precision@1 of 63.9 across all 36 MMEB datasets, outperforming the closest same-backbone training-free embedding baseline by 10.2 points. 

---
# The Public Discourse Corpus (PDC): A Speaker-Attributed Dataset for Valence and Epistemic Modality with Target Speaker Participation 

**Authors**: Bo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.20232)  

**Abstract**: We introduce the \textbf{Public Discourse Corpus (PDC)}, the first dataset of public-figure interview speech jointly annotated for affective valence and epistemic modality. The corpus contains 998 videos from 100 speakers across seven professional domains, yielding 186,642 sentences (3.1 million words) after sentence segmentation and filtering. To ensure that all retained videos contain analyzable speech from the intended speaker, we introduce \textbf{Target Speaker Participation (TSP)}---a five-category annotation taxonomy with documented inter-annotator reliability ($\kappa = 0.616$)---as a key methodological contribution that any corpus construction project can adopt. Target-speaker turns are separated from interviewer and third-party speech through an \textbf{audio-first diarization pipeline} combining local Whisper ASR with pyannote speaker separation, released as an open-source implementation. We release the annotated corpus, the annotation tools, the cross-provider validation sample, and the complete processing pipeline. The dataset is available at this https URL. 

---
# Before the Warning Comes Too Late: Incremental Phone-Scam Detection from Speech 

**Authors**: Khang Nhat Hoang Vo, Anh Trac Duc Dinh, Tai Tien Ta, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2609.20223)  

**Abstract**: We study weakly supervised incremental telecom fraud detection from raw telephone audio, where training provides only conversation-level labels and predictions must be updated before a call ends. We introduce StreamFraudNet, which processes incoming audio through overlapping bounded-context windows using a frozen self-supervised speech encoder, recurrent temporal modeling, and learned aggregation of latent window scores. On a controlled English benchmark, StreamFraudNet achieves a ROC--AUC of \(0.9953\), significantly outperforming acoustic and mean-pooling baselines while remaining competitive with strong global temporal models. The model produces its first prediction after 10 seconds of audio, updates every 2 seconds, and operates faster than real time on the evaluated server hardware. Ablations identify recurrent temporal context as the principal contributor to performance. These results demonstrate that fraud risk can be scored incrementally from raw speech without transcripts or temporal annotations, while highlighting the need for latency-aware training to improve early prediction. 

---
# Foundations of Stochastic Lexical Calculus: Semantic Descent and Random Dynamics on Probability Simplices 

**Authors**: Matthew F Dixon  

**Link**: [PDF](https://arxiv.org/pdf/2609.20207)  

**Abstract**: Large language models produce prompt-dependent probabilities over words, whereas scientific systems require uncertainty over meaningful states that can be updated as evidence arrives. We develop an observable framework for determining when language-derived probabilities support such a sequential state representation. Theoretically, we define typed measurable transformations of contextual language, construct a minimal closed representation, and give necessary and sufficient conditions for semantic updates to exist uniquely. We bound irreducible nonclosure and accumulated error, and under average contraction prove existence, uniqueness and stability of an external random recursion on a probability simplex. These results define a stochastic lexical calculus without attributing an internal calculus to the language model. Empirically, frozen experiments test the observable implications. Raw prompt-conditioned probabilities fail the prespecified invariance gate; after prompt-specific calibration, a common three-state representation passes the stability gates and covers 28 of 30 untouched eight-step paths, or 0.933 at nominal level 0.90. Accordingly, language probabilities support a stochastic state only conditionally on verified closure, stability and coverage within a declared operating domain. 

---
# To Copy or Not to Copy: Controlling Speculative Decoding via Intrinsic Model Signals 

**Authors**: Roy Eisenstadt, Ido Cohen, Edo Cohen-Karlik, Lior Wolf, Itamar Zimerman  

**Link**: [PDF](https://arxiv.org/pdf/2609.20186)  

**Abstract**: Speculative Decoding (SD) has significantly accelerated Large Language Model (LLM) inference, yet existing approaches face a fundamental tradeoff between two drafting strategies: neural drafting and context-based copying. Neural drafts (e.g., EAGLE3) provide robust performance across diverse text settings, while copy-based methods achieve higher speedups in copy-intensive regimes by generating candidates faster and exploiting long repetition spans for near-perfect speculation. We analyze existing copy-based methods and find that they are prone to accidental repetitions where surface-level n-gram overlap does not reflect a structural intent to copy, leading to false-positive triggers that ultimately degrade throughput. We introduce SwitchSD, an adaptive framework that treats copying as a latent control signal of the LLM. By training lightweight probes on the target model's internal representations, SwitchSD identifies genuine copy-intent with high precision (AUC > 0.99). This allows the system to dynamically switch between neural drafting (e.g., EAGLE) and context-based copying. Our results across Llama and Qwen families demonstrate throughput gains of up to 15% over state-of-the-art baselines like EAGLE3, effectively turning copying from a noisy heuristic into a principled, model-aware decoding regime. 

---
# Fine-Tuning Models for Biomedical Relation Extraction 

**Authors**: Claudiu Creanga, Liviu P. Dinu, Daniela Gifu  

**Link**: [PDF](https://arxiv.org/pdf/2609.20169)  

**Abstract**: Next-Generation Sequencing has revolutionized the study of genetic mutations, enabling large-scale investigations into their roles in disease development. However, extracting meaningful insights from the vast amount of biomedical literature remains a complex challenge that cannot be addressed manually. In this paper, we present pre-trained models (PTMs) for the automatic extraction of relations from biomedical text, specifically targeting the variant-phenotype domain. Our evaluation on the SNPPhenA corpus demonstrates that fine-tuning small BERT-based models, particularly DeBERTa, yields strong performance, approaching the current state-of-the-art (SOTA). Additionally, our results indicate that carefully fine-tuning Google's Gemini Pro 1.0 outperforms the existing SOTA for both sentence-level tasks (where the model processes only the target sentence) and abstract-level tasks (where the model processes the entire abstract). 

---
# Design of the IBM Granite 5.0 TurboCTC ASR Model 

**Authors**: Brian Kingsbury, George Saon, Masayuki Suzuki, Hong-Kwang J. Kuo, Takashi Fukuda, Samuel Thomas, Vishal Sunder, Avihu Dekel  

**Link**: [PDF](https://arxiv.org/pdf/2609.20104)  

**Abstract**: We describe the architecture, training methodology and inference speedups of Granite 5.0 Turbo CTC, a 470 million parameter encoder-only model with an excellent speed-accuracy tradeoff. The architecture uses pyramidal temporal subsampling within Conformer blocks using strided depthwise convolutions, block-diagonal (chunk-wise) self-attention, and conditioning on intermediate predictions from the middle layer. Training highlights are the use of only publicly available data, the novel use of a Muon optimizer, and balanced data sampling. Inference speedups include replacing 1 x 1 convolutions with linear layers and optimizing the attention computation in the Conformer blocks. Collectively, these result in a model that is on the speed-accuracy Pareto frontier of the Open ASR leaderboard for English short-form ASR while being twice as fast as the fastest competitor. The model can be used under a permissive license and downloaded from this https URL. 

---
# Reading Emotions in the Token Space: Discriminative Adaptation of SpeechLLMs for Emotion Recognition 

**Authors**: Hasindri Watawana, Sergio Burdisso, Esaú Villatoro-Tello, Manjunath K E, Kadri Hacioglu, Petr Motlicek, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2609.20081)  

**Abstract**: SpeechLLMs have shown strong potential for emotion recognition, yet they read the predicted emotion off a generative decoder not suited for classification: it can emit labels outside the target set and favors frequent classes. We propose a discriminative adaptation that reads the final prompt token's hidden state through a classification head, producing a label in one forward pass without modifying the backbone. Because this readout starts from the hidden state the model would otherwise decode, it gives a controlled comparison of generative and discriminative inference in an otherwise identical speechLLM. We keep the head a single linear layer, trading little accuracy for interpretability: each emotion becomes one direction in the LLM output token space, revealing associated tokens. On IEMOCAP, across two speechLLM architectures, it improves Macro F1 and removes hallucinations, with largest gains on realistic ASR transcripts. Our analysis reveals that these emotion directions encode indirect associations mirroring biases in web-scale text. 

---
# Benchmarking LLM Compliance with China AI Generated Content Regulations 

**Authors**: Chenrui Cui, Hongye Fang, Lisha Song, Weichao Chen, Yue Zhu, Gang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19989)  

**Abstract**: The widespread adoption of LLMs has led to escalating content compliance risks. Prior works have contributed to addressing these risks in the English context, downplaying the complexity of Chinese language content. This paper follows China's current AI-Generated content compliance requirements and provides evaluation results on 20 notable LLMs, offering insight into China's regulatory landscape. We design a novel framework to assess the compliance and refusal rates with 2303 questions spanning six distinct dimensions, including 203 self-constructed constitutional questions. The framework employs several judges to generate verdicts independently based on their hierarchical alignment memory. Our findings show that international models also exhibit high levels of compliance despite the use of standard Chinese questions, and the main differences may stem from dimensions closely related to ideological alignment. We establish a regulatory benchmark that enables the global AI community to evaluate both Chinese and non-Chinese LLMs under a unified set of legally grounded compliance requirements. 

---
# DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression 

**Authors**: DeepSeek-AI, Anyi Xu, B. Li, Bangcai Lin, Bing Xue, BingCheng Xian, Bingzheng Xu, Bochao Wu, Bowei Zhang, Boyi Deng, C.C. Yu, Chao Jin, Chaofan Lin, Chen Dong, Chenbing Wang, Chenfan Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chengyuan Zhang, Chenhao Xu, Chenqi Zhao, Chenze Shao, Chuhao Wang, Chuqi Zhang, Damai Dai, Dejian Yang, Deli Chen, Di Huang, Di Wu, Donghao Li, Erhang Li, Eric Fu, F. Zhou, Fangwei Zhou, Fangyun Lin, Fangzhou Yuan, Feiyu Xia, Fucong Dai, Guangbo Hao, Guanglin Li, Guanting Chen, Guoai Cao, Guofan Fan, Guolai Meng, Guowei Li, Haichuan Zhang, Haiyang Ma, Haiyang Shen, Han Li, Han Yu, Han Zhang, Hangyuan Deng, Hanwei Xu, Hanxiang Xu, Hanxun Zhong, Hao Guo, Hao Jiang, Hao Li, Hao Qin, Haodong Wen, Haofen Liang, Haofeng Huang, Haohua Liu, Haoling Zhang, Haoming Luo, Haoran Yang, Haotian Xu, Haotian Yuan, Haoting Huang, Haowen Luo, Haoyang Cai, Haoyu Chen, Haozhe Ji, Hengran Zhang, Hengrui Wang, Hengxu Wu, Honghui Ding, Hongxuan Tang, Huadong Wang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, J. Yang, J.H. Jin, J.H. Zhang, J.X. Zou, Jia Yu, Jiahui Zhou, Jiajun Chen, Jialiang Huang, Jialin Zhao, Jiamin Tang, Jian Zhou, Jianan Tong, Jianwen Li, Jiaqi Zhu, Jiarui Wang, Jiasheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2609.19969)  

**Abstract**: The widespread adoption of long-horizon agents has made model workloads increasingly input-heavy. Although prior work has substantially reduced the cost of long-context computation, prefill remains computationally expensive, and large KV caches continue to strain HBM and SSD capacity and data-transfer bandwidth. Together, these compute, storage, and bandwidth demands constitute the primary bottleneck to further lowering deployment costs. To address this challenge, we introduce DeepSeek-V4.1-Flash, a multimodal Mixture-of-Experts (MoE) model with 552B backbone parameters and support for contexts of up to one million tokens. With its Causal Encoder-Decoder (CED) architecture, the model activates 16B parameters per token during decode but only 8B parameters during prefill, substantially improving cost efficiency for agentic workloads. To push the limits of KV cache compression, DeepSeek-V4.1-Flash combines cross-layer KV cache reuse in Compressed Sparse Attention 2 (CSA2) with FP4 KV caching. These designs reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash. Further, through a dedicated deployment optimization known as SWA Bounded Replay, DeepSeek-V4.1-Flash reduces its persistent KV cache footprint (always on SSD or in host memory) to roughly 1/8 of that of DeepSeek-V4-Flash. Despite its much smaller KV cache footprint, the model delivers substantially better performance than the baseline. In addition, we streamline the DeepSeek-V4 architecture and introduce several efficient architectural extensions. We pretrain DeepSeek-V4.1-Flash on a multimodal corpus comprising 45T tokens and conduct comprehensive post-training, yielding strong performance across diverse text-based and multimodal agentic scenarios. Model checkpoints are available at this https URL. 

---
# Before the Arrest: Benchmarking LLMs on Criminal Profiling from Incomplete Evidence 

**Authors**: Yutong Yao, Yanjie Cao, Guanhua Chen, Xu Yang, Junchao Wu, Zeyu Wu, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2609.19965)  

**Abstract**: Large Language Models (LLMs) are increasingly applied to legal and criminal justice tasks, yet existing work focuses almost exclusively on post-arrest scenarios where the suspect's identity is already known, leaving the critical pre-arrest challenge of inferring suspect characteristics from incomplete evidence largely unexplored. To fill this gap, we introduce the Profiling, Investigation, and Judgment (PIJ), comprising 2,500 real homicide cases from five countries. PIJ evaluates LLMs across three tasks that span the entire criminal investigation pipeline: criminal profiling, which requires abductive reasoning to infer suspect attributes from fragmentary scene evidence, crime process reconstruction, which tests structured information extraction, and sentence prediction, which demands legal deductive reasoning. We evaluate 9 powerful LLMs and find that performance degrades systematically as tasks shift from explicit fact extraction to implicit reasoning over unknown suspect profiles. Categories requiring inferential reasoning, such as motivation and victim-offender relationships, remain the primary bottlenecks. Further analysis reveals substantial gaps between LLMs and human experts, along with pervasive biases in gender, age, and motive attribution. Our findings indicate that pre-arrest inference from incomplete evidence remains an open challenge. 

---
# Intrinsic Sequence-Likelihood Confidence in Retrieval-Dominated Extractive QA: Two Pre-Specified Negatives, and What They Do and Do Not Attribute 

**Authors**: Gunwoo Lee, Changmin Sung, Sang-Hwan Gwak, Ina Kim, Ji-Young Choi, Kyong-Ha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.19942)  

**Abstract**: In extractive document question answering whose questions were generated from the passages that contain their answers -- so that retrieval recovers 92-99.8% of what any mode combination could reach, whatever its absolute accuracy -- confidence-driven mechanisms have little to gain. Fine-tuning an open language model on a specialized domain corpus yields a model whose own confidence is a tempting control signal: it could decide which queries warrant further adaptation, and which answers to trust. We evaluate both uses under criteria fixed before the runs were executed, across four 7-9B model families whose adaptation moved closed-book F1 by at most +0.03, and both fail: a distillation trigger on all four families, under its pre-specified three-step transfer budget, and a routing-and-abstention policy in its single-model pilot. Retrieval alone recovers 92-99.8% of best-case combined accuracy under every correctness criterion we test, leaving routers no meaningful gain. The sequence-likelihood signal is insufficient relative to that mode -- area under the receiver operating characteristic curve 0.65-0.81 under the registered criterion -- before adaptation as well as after, unchanged by scalar recalibration and not consistently improved by token-level temperature rescaling. And the finer diagnostics depend on the correctness criterion and on answer length; on the three adapted combinations where we could test it, selector ablations show no statistically detectable downstream benefit from the confidence term on any seed; on Gemma, removing it changes the selector from failing to passing both registered criteria. The usable product is a set of pre-specified negatives with their dependencies made explicit. 

---
# KoNeoBench: A Curated Evaluation Dataset for LLM Understanding of Korean Neologisms 

**Authors**: Soha Lee, Soojin Lee, Heesung Yang, Hyunju Song, Hyunji Lee, Jinsan An, Jeongwan Shin, Jin Hyun Park, Jun Lee, Hyeyoung Park, Kilim Nam  

**Link**: [PDF](https://arxiv.org/pdf/2609.19916)  

**Abstract**: Large language models (LLMs) are typically evaluated on static benchmarks, even though natural language constantly evolves through newly emerging words and meanings. Existing Korean benchmarks are centered on established vocabulary and therefore provide limited coverage of such recent lexical change, and their English-oriented design makes it difficult to assess the typological properties of Korean, in which content words combine productively with functional morphemes. In this paper, we introduce KoNeoBench, a benchmark for evaluating LLMs' understanding of Korean neologisms. KoNeoBench is built on 1,785 Korean neologisms attested in online news since 2020 and curated through expert lexicographic review. Each entry provides usage examples, word-formation analyses, and dictionary-style definitions. Based on this resource, we define four tasks and report results on recent models, together with a human baseline. Our experiments show that current LLMs exhibit clear limitations in recovering source components, distinguishing semantic categories, and generating accurate definitions. These results reveal specific aspects of recent Korean lexical change that remain challenging for current LLMs. KoNeoBench is available at this https URL . 

---
# Generalization through Lexical Abstraction in Transformer Models: The Case of Functional Words 

**Authors**: Giuseppe Samo, Vivi Nastase, Paola Merlo  

**Link**: [PDF](https://arxiv.org/pdf/2609.19887)  

**Abstract**: Pronouns, adverbs and other functional words (such as they, her, somewhere, there) are often used in language to replace concrete nouns or phrases, when their properties - such as gender, grammatical number - provide sufficient information for the given context. Do pretrained transformer models encode such functional words in a manner that allows them to be used like humans do? Can language models recognize the syntactic and semantic parallelism of sentences such as "The researchers wrote the paper" and "They wrote it", which relies on such lexical abstraction?
We map these linguistic questions into the embedding space of a pretrained transformer model, and compare representations of nouns, with the representations of the pronouns and adverbs that can replace these nouns, in isolation and in parallel lexicalized and functional sentences. We then probe for shared syntactic and semantic structure in the embeddings of parallel lexicalized and functional sentences.
We find that functional words are located centrally compared to nouns, but are also distinct, which is congruent with their behaviour as place-holders in a wide variety of contexts.
The analysis of the embeddings of parallel (lexicalized and functional) sentences show them inhabiting different subspaces of the embedding space. Experiments that distil the structural information of the sentence show that training on either type of data does not reveal the shared structure - because of the over-consistency of the vocabulary (in case of the functional data), and the too much variety (in case of the lexicalized versions). However, training with a mix of functional and lexicalized sentences, the shared structure emerges. 

---
# Evaluating Communicative Success in Machine-Translated Conversation 

**Authors**: Faiz Ghifari Haznitrama, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2609.19885)  

**Abstract**: Interpreter agents built on machine translation (MT) increasingly mediate live conversation between people who do not share a language, yet we still evaluate them with metrics built for isolated sentences, which measure fidelity rather than whether communication succeeds. We introduce a reusable three-layer checklist-and-judge framework that evaluates interpreter-mediated conversation across semantic, pragmatic, and cultural-social dimensions, covering the naturalness, intent, and social appropriateness that fidelity metrics leave unmeasured. It runs in both single-turn and interactive multi-turn settings, where simulated users reply to translated messages as the conversation unfolds and each turn is scored alongside the conversation as a whole. We extensively validate it through controlled perturbations, cross-judge comparisons, and human annotations. Our main single-turn benchmark evaluates 10 interpreter setups across Arabic, Bengali, Indonesian, and Korean from 5,624 OpenSubtitles-derived scenarios spanning 12 translation directions, and our multi-turn study covers all 6 language pairs in scripted and live modes. Results show a consistent decline from semantic to pragmatic and cultural-social success, while conventional MT metrics overlook failures among stronger interpreters, and prompt ablations show that scenario context, structured instructions, and cultural context improve communicative success, although gains vary across setups. Our work thus provides an evaluation framework and benchmark for interpreter agents in conversation, and highlights the importance of communicative success alongside existing translation metrics. 

---
# PetriBench: Benchmarking LLM Reasoning over Dynamic State Spaces 

**Authors**: Pyrros Koussios, Benjamin Jäger, John Hua Yao, Ajay Sridhar, Violet Xiang, Chenhao Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.19883)  

**Abstract**: Characterizing LLM reasoning remains an open challenge, as many existing benchmarks isolate specific reasoning skills, rely on external knowledge, or are costly to extend. We introduce PetriBench, a compact, fully self-contained, and scalable benchmark for evaluating LLM reasoning over dynamic state spaces using Petri nets, a mature formalism for modeling real-world concurrent and distributed systems. PetriBench organizes reasoning into four task families varying by scope and temporal horizon, with Easy, Medium, and Hard levels generated by increasing structural complexity and evaluated against exact ground truth. Across a diverse set of proprietary and open-weight models, accuracy decreases consistently with difficulty, while harder instances expose increasingly distinct task-specific capability profiles. Additional analyses show that test-time compute improves performance but interacts differently with different reasoning tasks, and that procedural generation yields smooth scaling with structural complexity. Together, these results show that PetriBench provides a unified and extensible setting for probing the strengths, limits, and scaling behavior of LLM reasoning. 

---
# D-Quant: Driftable Entropy Coding for KV Cache Quantization 

**Authors**: Yi Su, Hong Liu, Guanghua Yu, Jianchen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19880)  

**Abstract**: The KV cache has become a major bottleneck in deploying LLMs, as its memory footprint grows linearly with sequence length and batch size, imposing substantial pressure on both memory capacity and bandwidth. Among various KV cache compression techniques, quantization is particularly attractive due to its effectiveness and ease of deployment. However, most existing methods rely on fixed-width quantization, where a $b$ bit representation is inherently limited to $2^b$ quantization levels. As the bit width decreases, the number of available levels shrinks exponentially, leading to severe information loss and rapid performance degradation. We further observe that fixed-width quantization fails to exploit the highly non-uniform distribution of KV cache. After rotation and normalization, KV values approximately follow a normal distribution, with most values concentrated near the center and only a small fraction appearing in the tails. Nevertheless, fixed-width coding allocates the same number of bits to frequent and rare symbols. Entropy coding naturally exploits such non-uniformity by assigning shorter codewords to frequent symbols and longer ones to rare symbols, substantially reducing the average number of bits required for representation. However, its variable-length output is not suited to highly parallel attention kernels, where efficient dequantization and computation rely on regular memory layouts and fixed-stride accesses. To bridge this gap, we propose \textbf{D-Quant}, a flexible KV cache quantization framework that introduces a \textbf{drift} mechanism to convert entropy-coded representations of each token into fixed-size bitstreams, enabling regular memory access and parallel dequantization within attention kernels. 

---
# VākQA: A Benchmark and Evaluation Study for Telugu Spoken Factoid Question Answering 

**Authors**: Bhavana Akkiraju, Ravi Sastry Kolluru, Sri Charan D, Srihari Bandarupalli, Santosh Kesiraju, Anil Vuppala  

**Link**: [PDF](https://arxiv.org/pdf/2609.19879)  

**Abstract**: Question answering has advanced rapidly with large language models, but predominantly for high-resource languages, in both text and spoken settings. Spoken question answering (SQA) benchmark for Telugu remains unexplored, and the reliability of automatic evaluation in this setting remains unquantified. We introduce VākQA, a Telugu SQA benchmark of 2,001 factoid question-answer pairs across six domains, with 2.53 hours of speech audio, bilingual transcriptions, and human-verified reference answers. We first validate evaluation methods against human judgements: Gemini-as-a-judge best approximates human ratings but is non-uniformly strict, while open-weight judges systematically penalize correct Telugu answers that differ in surface form from the reference. Using this validated setup, we benchmark proprietary and open-weight models across input modality, language, and domain. We observe that Telugu phrasing retains cultural specificity that is lost in translation, speech input introduces phonetic confusions that alter question meaning, and cascaded ASR-MT errors compound progressively. VākQA is publicly released. 

---
# JustMem: Just-Enough Memory Access for Long-Term Conversations 

**Authors**: Guanhua Chen, Yanting Wang, Wenjing Zhi, Lei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2609.19877)  

**Abstract**: Efficient long-term conversational memory requires retrieving sufficient evidence without indiscriminately expanding the context presented to the language model. This is challenging because relevant evidence may be distributed across multiple sessions, while compression may discard details needed for answering. Different queries therefore require different forms of memory access. To capture these demands, we formulate memory access along two dimensions: discovery breadth, which controls how broadly evidence is searched, and reading fidelity, which controls whether evidence is read in compact form or recovered from the original conversation. Based on this formulation, we introduce JustMem, which stores conversation history as compact atomic memories and adapts memory access along these two dimensions to each query. Specifically, LOOKUP handles local evidence, COMPOSE broadens discovery for distributed evidence, and REPLAY increases reading fidelity for fidelity-sensitive evidence. On LoCoMo and LongMemEval-S, JustMem achieves the highest mean accuracy and retrieval recall among the compared memory systems while using substantially fewer generative-model tokens for memory construction and inference. 

---
# Zarya: A Hybrid Autoregressive--Masked Diffusion Language Model with Flexible Training and Dual-Mode Inference 

**Authors**: Leonid Sinev, Ilya Koziev, Vladislav Leshchuk  

**Link**: [PDF](https://arxiv.org/pdf/2609.19868)  

**Abstract**: Autoregressive language models (ARMs) are constrained by sequential, left-to-right generation, while masked diffusion models (MDMs) enable parallel decoding but suffer from high computational overhead due to the inability to reuse Key-Value (KV) cache and from incoherent generation arising from learning dependencies over an intractable space of token combinations. We introduce Zarya, a family of hybrid language models that jointly optimizes an autoregressive (AR) objective and a masked-diffusion objective within a single architecture. Zarya structures training data into variable-size slots and employs a curriculum that gradually increases slot granularity, enabling a smooth transition from fine-grained AR learning to coarse-grained diffusion learning. At inference, Zarya provides two distinct decoding paradigms through a unified interface: (i) MDM sampling with first-hitting denoising, and (ii) slotted speculative decoding that interleaves inter-slot diffusion-based selection with intra-slot autoregressive infilling, achieving full KV cache reuse. The training and inference regimes are fully decoupled, allowing a model trained with any configuration to be deployed in either mode. Extensive configurability --- including grouped noise patterns (Prefix Completion, Fill-In-the-Prefix, Fill-In-the-Middle), ordered sampling schedules, and noise-level permutation strategies --- enables flexible research exploration. We release Zarya models publicly in sizes 0.6B, 1.7B, and 4B, demonstrating performance on standard benchmarks while offering a principled integration of autoregressive and diffusion paradigms. 

---
# F$^{2}$DR: A Fine-Grained Full-Pipeline Reward Framework for DeepSearch Workflows 

**Authors**: Bojian Xiong, Wentao Ding, Yujing Lu, Shaowei Zhang, Ling Shi, Jing Liao, Yan Wang, Yueyang Zhang, Long Xia, Zhiyuan Sun, Daiting Shi, Jingzhou He, Yuqi Ren, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2609.19827)  

**Abstract**: With the widespread industrial deployment of Large Language Models (LLMs), DeepSearch has emerged as the dominant paradigm for resolving complex user queries. It typically operates through an iterative closed-loop workflow consisting of planning and reflection, information retrieval, and answer generation. However, existing reward models (RMs) and evaluation benchmarks are primarily designed for static single-turn tasks, failing to capture the full-pipeline complexity of DeepSearch workflows. To address this limitation, we propose F2DR, a fine-grained full-pipeline DeepSearch reward framework. F2DR evaluates DeepSearch workflows across three dimensions: Content, Trajectory, and Answer, enabling comprehensive process-level assessment. We further construct DeepSearch RM-Bench, a dedicated benchmark for evaluating RMs in DeepSearch scenarios. Extensive experiments demonstrate that F2DR achieves significantly higher evaluation consistency than self-evaluation-based baselines, while DeepSearch RM-Bench exhibits strong discriminative capability across existing open-source RMs. We will publicly release the complete DeepSearch RM-Bench dataset soon. 

---
# Dictionary-Constrained Grapheme-to-Phoneme for Unsegmented Languages from LLM-Annotated Data 

**Authors**: Rui Hu, Zhenpeng Zhan, Xiaolong Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.19805)  

**Abstract**: Grapheme-to-phoneme (G2P) conversion turns raw text into its phonemic form and is an essential part of both text-to-speech (TTS) and automatic speech recognition (ASR) systems. It is required to be fast, stable and context-aware. For unsegmented languages such as Japanese, G2P additionally couples word segmentation with highly context-dependent polyphone disambiguation, and the scarcity of accurately annotated data remains a bottleneck. In this paper, we present a context-aware neural G2P method that scores paths of a discriminative conditional random field (CRF) over a word lattice constructed from dictionaries. To tackle data scarcity, we utilize large language models (LLMs) to generate more than 2 million sentences. Experimental results demonstrate that our method strongly outperforms conventional morphological analyzer-based methods and neural sequence models. On the Joyo-Kanji-Yomi benchmark, our method reaches 99.62% target word reading accuracy, 0.32% target word phoneme error rate (PER) and 0.14% sentence PER. 

---
# Evolution or Illusion? Rethinking Evaluation in LLM Evolutionary Search 

**Authors**: Tal Oved, Roi Pony, Oshri Naparstek, Udi Barzelay  

**Link**: [PDF](https://arxiv.org/pdf/2609.19799)  

**Abstract**: LLM-driven evolutionary search finds programs by launching seeds and iterating each one. Papers report a single budget setting, usually one seed run for a fixed number of iterations, and rank methods from that one point. We show this is not enough. We evaluate three evolutionary search strategies on five optimization tasks, commonly used by papers in the genre to report results. We run the analysis over a full grid of seeds and iterations. Our findings suggest that the best way to split a fixed budget between more seeds (width) and more iterations (depth) changes with the strategy, the task, and the total budget. Furthermore, we observe that the ranking of strategies also changes with the budget. On one task the strategy that looks worst at one seed is best at forty seeds. On another the best number of iterations is well below the value common in practice, so extra depth wastes budget that more seeds would turn into score. We provide a measurement protocol that reports the seeds-by-iterations frontier and practical guidance for using it. 

---
# Learn Before You Judge: Progressive Knowledge-to-Decision Alignment for Explainable Hateful Meme Detection 

**Authors**: Bo Xu, Chenyuan Wang, Xinyu Chen, Quanhao Zhu, Rui Lin, Liang Zhao, Hongfei Lin, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2609.19778)  

**Abstract**: Hateful memes spread abusive content through implicit interactions between images and text, posing serious threats to the safety of online communities. In recent years, multimodal large language models have been widely used for hateful meme detection and are increasingly adopted to generate explainable detection results. However, we find that existing explain-then-detect methods often couple explanation generation and label prediction within the same training process. This coupling causes interference between task objectives, leading to limited detection performance and even worse results than simple SFT baselines. To address these challenges, we propose ProKDA, a progressive knowledge-to-decision alignment method for explainable hateful meme detection. Inspired by the human annotation training process, ProKDA first uses an agentic background knowledge construction pipeline to obtain external knowledge related to meme understanding. It then adopts a three-stage training strategy that sequentially performs background knowledge learning, hatefulness detection learning, and hatefulness boundary alignment. Unlike prior explain-then-detect methods that jointly optimize both tasks, ProKDA focuses on a single training objective at each stage. This design reduces interference between the two tasks and progressively transforms background knowledge into robust detection decisions. Experiments on three public hateful meme benchmarks show that ProKDA achieves state-of-the-art detection performance and provides accurate, explainable, and evidence-supported decisions for hateful meme moderation. Project page: this https URL. 

---
# A Phonemically Comprehensive, ASCII-Only Romanization Scheme for Thai and Lao: Systematic Cross-Lingual Correspondence and Chinese-User-Friendly Design 

**Authors**: Zijie Zhang, Tan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.19736)  

**Abstract**: This paper proposes a phonemically comprehensive, ASCII-only romanization scheme for Thai and Lao, treating the two closely related languages as a unified cross-lingual design problem. The scheme represents segmental contrasts, vowel length, and lexical tone while maintaining one-symbol-one-phoneme transparency and systematic correspondence between Thai and Lao. The scheme prioritizes synchronic phonetic correspondence, including correspondence with Pinyin and Jyutping where applicable, while preserving historical-phonological correspondence where it does not conflict with phonetic transparency. Tone uses a compact single-digit default notation, supplemented by optional tone-value and historical tone-category representations. The resulting scheme provides a readable, keyboard-friendly, and machine-processable phonemic representation for language learning and cross-lingual speech processing. 

---
# Improving Cross-Lingual Transfer for Sequential Sentence Classification in Research Papers via Structural Similarity 

**Authors**: Kazuhiro Yamauchi, Marie Katsurai  

**Link**: [PDF](https://arxiv.org/pdf/2609.19650)  

**Abstract**: Sequential sentence classification (SSC) is an essential task for structuring scientific publications, and extending SSC research to languages other than English can improve accessibility to scientific knowledge in multilingual digital libraries. Cross-lingual transfer is a promising approach to address the scarcity of training data in non-English languages. Prior work on other natural language processing tasks has shown the benefits of capturing linguistic similarity between source and target languages. However, SSC inherently depends on patterns at the discourse level, such as label sequences and positional regularities, which appear consistently across languages regardless of linguistic differences. To examine the factors that determine transfer success in SSC, we constructed a multilingual SSC dataset covering 13 non-English languages collected from five academic databases. Our cross-lingual transfer experiments, using both encoder-based and generative models, show that linguistic proximity has no consistent predictive power for transfer performance, whereas structural similarity in rhetorical organization shows a weak but consistent positive correlation across models. After controlling for source-language performance, the similarity of label distributions is the most consistent predictor. Building on this finding, we propose a set of three methods that explicitly leverage structural information using generative models. In the in-domain evaluation, the best combination reaches parity with the strongest encoder baselines, and in transfer to languages unseen during training, it outperforms the strongest encoder baseline. 

---
# Semantic Layer Induction from Raw Telemetry via Hierarchical LLM and RAG Abstraction 

**Authors**: Yuanzhe Jia, Ali Anaissi  

**Link**: [PDF](https://arxiv.org/pdf/2609.19615)  

**Abstract**: Modern applications generate massive volumes of raw telemetry data, but translating those noisy, heterogeneous event streams into actionable business insights remains a fundamental challenge. Data engineers and analysts expend substantial effort reconciling semantic discrepancies, hand-crafting parsing logics, and maintaining fragile mappings between raw data and business KPIs. In this paper, we present an end-to-end framework that fully automates the construction of a business semantic layer from application raw logs. Our approach introduces a two-stage semantic abstraction: first, high-level business features are identified via LLM inference augmented with domain-specific industry knowledge; second, fine-grained business nodes are derived through a structured pipeline comprising data refinement, hybrid retrieval, multi-stage filtering, semantic clustering, and canonical naming. Evaluation on production-scale telemetry demonstrates that our system improves human-assessed semantic quality from 50 to 80+ on a 100-point scale, reduces maintenance effort by 80%, filters out 74% of noise, and achieves 0.87 Cohen's kappa via an integrated LLM-as-Judge evaluation, enabling continuous, scalable quality assurance. Overall, our work distinguishes itself from prior work by addressing the novel problem of business semantic layer induction from raw telemetry, operating without labeled training data or manual rule engineering. 

---
# Chain-of-Thought Entropy as a Reliability Signal: A Preregistered Reproduction 

**Authors**: Theodore O. Cochran  

**Link**: [PDF](https://arxiv.org/pdf/2609.19606)  

**Abstract**: This empirical study is an independent reproduction of the dissociation Zhao reported in 2026. The shape of a large language model's chain-of-thought entropy trajectory predicts whether the final answer is correct, while the magnitude of its total entropy drop does not. The dissociation merits reproduction because the magnitude half rests on a single 300-problem run with one model at one seed, while the shape half was reported at full scale on both benchmarks and on a second model family. Registered at OSF before any confirmatory run, the reproduction crosses the complete GSM8K and MATH-500 benchmark test sets with four open-weight models including one reasoning-distilled model of a kind the original did not test. The shape signal replicates. The magnitude signal divides by setting. On the anchor model the accuracy gap between monotone and non-monotone chains is +9.6 percentage points on GSM8K and +27.5 on MATH-500, while the rank correlation of the total entropy drop with correctness is -0.018 on GSM8K and +0.414 on MATH-500. On the reasoning-distilled model the binary form of the shape signal fires on about one chain in a hundred, too few to estimate the registered contrast, while the graded violation count remains predictive there. In an exploratory comparison the final-step entropy alone outperforms the binary shape flag in all eight model-by-benchmark cells by ROC area, and in six or seven by the risk-coverage area the original reports, depending on an integration range the original does not state. The study contributes a reproduction of the shape signal at full test-set scale under seven documented protocol differences, a map of the settings where the magnitude signal holds and fails, and measurements of four protocol dependencies the original does not report. 

---
# Full-Duplex Speech Models Take the Floor When Asked, Not When Needed 

**Authors**: Linkai Peng, Baorian Nuchged, Kaiqi Fu, Yuyang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2609.19596)  

**Abstract**: Full-duplex speech models listen and speak at once, promising always-on assistants. Yet they must also decide when they should speak. Human listeners speak when addressed or when the speaker stops, but also self-select to correct a false claim, supply a missing word, or warn of danger. We ask whether full-duplex models do the same. To separate the reason to speak from the opportunity, we construct context-matched English monologues in which only the trigger utterance varies within a topic, define 10 conditions from turn-allocation rules, and compress inter-word pauses to limit opportunities created by silence. Across five model families, being addressed and silence are far more reliable triggers than false facts or hazards. Frame-level text-token probabilities in Moshi and PersonaPlex are lower for false facts than for Neutral when averaged over the first 2\,s after trigger end. Pauses or permission to interrupt do not close this gap either. Given the floor, Moshi and PersonaPlex answer most direct questions, yet the proportion of non-empty false-fact replies that challenge the claim is only .14--.15, and the proportion of hazard replies that warn of danger is .04--.07. This paper thus identifies a gap in both speech initiation and response content. Closing it requires genuine content understanding and intervention decisions grounded in it. 

---
# Form Over Content In Gradient-Based Data Attribution Methods 

**Authors**: Sunwoo Kim, Seokwon Jung, Sohyung Kim, Seong Joon Oh, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2609.19589)  

**Abstract**: Data attribution methods using gradient similarity are widely used to analyze and select training data for large language models, but what gradient similarity actually measures is debated. Some interpret it as identifying task-relevant skills, while other work reports that surface form is the main factor. We resolve this debate for supervised fine-tuning examples by varying task and answer format independently. Specifically, we render benchmarks in different answer formats, such that datasets can share a task without a format or a format without a task. We find that gradient alignment follows the answer format, as benchmark pairs sharing an answer format align strongly (disattenuated cosine near 0.4), while same benchmarks rendered with different answer format classes show no alignment (near 0.0). We demonstrate that this ordering holds from the earliest pretraining checkpoints through post-training, and across model scales and families. We then analyze the released selections of LESS, a gradient-based data selection method for instruction tuning, and find that each target's selections over-represent the target's own answer format. Hence, we demonstrate that gradient-based attribution methods track format similarity more than task semantics, meaning that such methods, as well as the semantic interpretation of the gradient, should be tested on data where answer format and task vary independently for greater robustness and reliability. 

---
# CliniCIRCA: A Modular LLM Framework for Constructing Longitudinal Mental Health Patient Journeys from Raw EHR Narratives 

**Authors**: Aiwei Ivy Zhang, Nimra Ishfaq, Mohit Chandra, Santiago Alvarez Lesmes, Adam Coscia, Khatiya Chelidze Moon, Xiaohan Ding, Munmun De Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2609.19585)  

**Abstract**: In mental health care, reasoning over patient journeys is a key task for clinicians. Yet these journeys, encompassing a longitudinal progression of biological, psychological, and social events, are often spread across disparate unstructured text narratives, making temporal recovery challenging. We present CliniCIRCA, a multi-stage LLM framework for Calendar-anchored, Imprecision-aware Reconstruction of Clinical Annals. To our knowledge, CliniCIRCA is the first to temporally classify clinical events across unstructured discharge summaries without event-level timestamps. From 14,882 MIMIC-III mental health admissions, we first construct a benchmark of 52 discharge summaries on which CliniCIRCA produces 15,891 temporally tagged events. After correcting 629 errors based on a clinician-in-the-loop evaluation, we produce verified gold-standard labels. Finally, the corrected timelines drive a temporally grounded summarization stage that compresses each source 1.52 times into a date-grouped chronological record. We then scale the framework to generate 1,000 silver-standard timelines and evaluate them as training data. Compared with zero- and few-shot prompting, instruction tuning generally improves five open-weight models on event extraction, temporal tagging, and summarization across silver and clinician-verified evaluations. 

---
# From Parameters to Behaviors: A Survey of Model Fusion for Large Language Models 

**Authors**: Shuo Cai, Yanggan Gu, Zihao Wang, Yuanyi Wang, Yibo Yan, Wenjun Wang, Yuhang Liu, Guanghao Zhu, Sirui Huang, Ming Li, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19553)  

**Abstract**: Model fusion integrates the capabilities from source models into a single target model. As of June 2026, Hugging Face hosts more than 2M models. This growing pool provides a rich base for model reuse and capability integration. Yet existing surveys often cover only separate parts of this space, and they do not provide a unified definition or a systematic taxonomy. This survey defines model fusion and organizes prior work into three levels: parameter-level, representation-level, and behavior-level fusion. We also review related metrics, benchmarks, and applications, summarize current challenges, and identify future directions. Our goal is to provide a clear map of this area and support future work on model fusion. A comprehensive list of papers about model fusion is available at this https URL. 

---
# For Your Eyes Only: Evaluating Coordination Between Isolated Language Model Instances 

**Authors**: Alexander Shirnin, Aleksey Kudelya  

**Link**: [PDF](https://arxiv.org/pdf/2609.19504)  

**Abstract**: As model-generated content is increasingly consumed by other model instances in automated workflows, a practically important question arises: can a model embed a signal in natural language that an independent instance of the same model can detect, relying only on shared pre-training and task instructions, without any shared memory or coordination-specific training? We introduce For Your Eyes Only, a cooperative signalling game designed to evaluate this directly. A Sender produces free-form descriptions for two words, one of which is a hidden target; an isolated Receiver must identify it. We evaluate seven contemporary models from four architectural families on 300 word pairs from established psycholinguistic corpora, using the Double-Pass Success Rate to control for output biases. We find that most models struggle to maintain coordination once they are required to avoid detectable signals, while one frontier model retains near-perfect performance even after such filtering. We further show that models can direct this capability toward deliberate misdirection, and that coordination is consistently weaker across architectures than within them. 

---
# Less Is More: Graph-free Multimodal RAG via Multi-signal Late Fusion 

**Authors**: Tithi Rakshit, Hongkuan Zhou, Lavdim Halilaj, Yuqicheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19417)  

**Abstract**: Graph-based retrieval-augmented generation (RAG) is widely used for multimodal, cross-document question answering. However, building corpus-level graphs is expensive, slow to query, and difficult to maintain. We present TrioRAG, a graph-free multimodal framework that integrates evidence from three complementary signals: the question, the anchor image, and a VLM-enhanced query generated from both. Each signal retrieves independently over a shared multi-vector index of page text and page images, and the results are combined through late fusion. Further, we introduce AutoQA, a multimodal automotive benchmark whose questions are grounded in noisy, web-sourced images rather than clean document-sourced figures. Its questions require reasoning across manuals. We position it as a model-curated testbed rather than a human-validated gold standard. Across three benchmarks, TrioRAG matches or outperforms graph-based systems while reducing total cost and accelerating per-query inference by 1.6-2.3 times. By construction, AutoQA grounds its questions in out-of-corpus web images. In this setting image retrieval reaches only 19.3% document-level recall, while text-derived signals, especially the VLM-enhanced query, keep retrieval robust. 

---
# The Role of Fine-grained Harm Signals in LLM Safety 

**Authors**: Soyeon Park, Seogyeong Jeong, Sunwoo Kim, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2609.19366)  

**Abstract**: Prior work has shown that internal harmfulness representations in large language models vary across risk categories, while sharing a common general harm representation component. This raises a question about the role of the category-specific component beyond general harm representation in LLM safety. To answer this question, we isolate the category-specific component by removing shared general harmfulness representation from each categorical harmfulness representation, yielding a category residual that is orthogonal to general harmfulness at every layer. Using activation steering with category residuals across 11 risk categories in 3 instruction-tuned LLMs, we find that whether category residuals encode harmfulness varies across categories, and that this category-wise pattern is similar across models. Whether category residuals induce refusal also varies across categories, but this category-wise pattern is more model-dependent. We also find that category residuals increase LLMs' downstream internal alignment with shared general harmfulness representation. Together, these findings demonstrate that more fine-grained category residuals should also be considered beyond shared general harmfulness representation to fully understand LLM safety. More broadly, our findings show that even a direction orthogonal to a concept at one layer can contribute to the concept's downstream amplification. 

---
# A frontend-backend architecture for tool calls in full-duplex speech models 

**Authors**: Ke Hu, Slyne Deng, Chen Chen, Elena Rastorgueva, Edresson Casanova, Punit Kumar, Dharmendra Choudhary, Nikhil Srihari, Ameya Sunil Mahabaleshwarkar, Viet Anh Trinh, Slim Essid, Oluwatobi Olabiyi, Zhehuai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.19334)  

**Abstract**: Full-duplex speech-to-speech (S2S) models provide natural, low-latency conversational interaction and would benefit from the ability to use external tools and complete voice-agent tasks. We propose a frontend-backend architecture where a duplex speech-to-text frontend learns to emit a delegation token and forwards streaming ASR transcripts to a text-based backend LLM for tool calls. Tool-call results from the backend are injected back into the frontend through a lightweight prefill-and-repeat mechanism and then synthesized using streaming TTS to the user. Our approach largely preserves regular duplex turn-taking, interruption handling, and low-latency interaction as it requires minimal modifications to the frontend model. In a single-turn tool-call evaluation, our system achieves 92-97% tool-call recall, competitive tool-call prediction performance, and 81.2% accuracy in rejecting irrelevant calls. When equipped with a larger backend (e.g., Qwen3-235B-A22B), our system achieves competitive results on Full-Duplex-Bench-V3 compared to open and closed source models, and significantly outperforms GPT-realtime-mini and Qwen3-Omni-30B-A3B-Instruct on EVA-Bench. These results demonstrate that backend delegation is an effective and modular approach for combining natural duplex speech interaction with strong agentic tool-call capabilities. 

---
# Why Pretraining Fails to Share Cross-Lingual Knowledge 

**Authors**: Adam Gaber, Uriel Dolev, Elisabeth Fittschen, Bobby Cheng, Yuval Marton, Leshem Choshen  

**Link**: [PDF](https://arxiv.org/pdf/2609.19291)  

**Abstract**: Large Language Models (LLMs) have made remarkable progress in the processing and modeling of many languages. Yet, unlike human multilinguals, they exhibit surprisingly limited cross-lingual knowledge transfer. While this limitation is well documented, its origins during multilingual training remain unclear. We pretrain 360M- and 7B-parameter LLMs and show that poor cross-lingual knowledge generalization emerges during pretraining and persists under standard interventions. To isolate its cause, we employ a controlled bilingual pretraining setting using two copies of the same language, sharing identical text and token segmentation, but mapped to disjoint token spaces. We find that disjoint tokens alone are enough to induce knowledge compartmentalization, even between identical copies of the same language, establishing disjoint token spaces as a fundamental barrier to cross-lingual knowledge generalization. Guided by this understanding, we suggest mapping languages into a shared token space by simple word-wise translation and find it substantially improves cross-lingual knowledge generalization, recovering up to 12.6\% of native-language learning efficiency --- 14$\times$ the baseline. 

---
# YNU-HPCC at SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Using Multiple Prediction Headers 

**Authors**: Hao Yang, Jin Wang, Xuejie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19238)  

**Abstract**: This paper describes the participation of the YNU-HPCC team in subtask A of task 11, Bridging the Gap in Text-Based Emotion at SemEval-2025. Our best-performing system employs the RoBERTa (Robustly Optimized BERT Approach) model, an improved version of BERT that utilizes the Transformer encoder architecture. We enhanced the output head to allow the model to process one emotion simultaneously. We obtained the official ranking score (0.44), including results from all languages. The entire dataset was translated into English using Google Translate to facilitate subsequent processing. Through probabilistic and attention analyses, we found that (I) a single prediction head performs better than six heads predicting six emotions simultaneously, and (II) training on a uniformly translated English dataset yields better results than using the original dataset. The code is available at: this https URL. 

---
# To Memories and Beyond: From Remembering to Knowing You across Long-Term Multimodal Personal Archives 

**Authors**: Wenqi Zhou, Zhuorui Yu, Kaiao Wen, Hao Zheng, Xinyi Zheng, Peiran Wu, Enmin Zhou, Chi-Hao Wu, Junxiao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.19167)  

**Abstract**: As AI systems evolve into personalized digital companions, a central capability is reasoning over a user's long-term personal history: not merely storing past events, but tracking longitudinal experiences and evolving preferences. Progress here is bottlenecked by evaluation, existing long-term memory benchmarks are largely synthetic and text-only, they overlook the visual records that anchor everyday human memory, lack the authentic and causally connected longitudinal data that real personalization demands, and consequently remain confined to shallow factual recall. We introduce ReaLMem (Real-world Long-term Multimodal Memory), the first benchmark built from authentic multi-year personal visual archives, paired with first-person subjective annotations. ReaLMem evaluates models across three cognitive tiers of increasing difficulty: factual recall, persona inference, and predictive personalization. We further propose ChronoProfiler, a temporal-weighting profiling module that computes temporal stability scores for user attributes and applies them as a salience prior, resolving conflicts among temporally inconsistent preferences and helping models compound multiple co-active preferences in complex personalized decisions. Extensive evaluation of frontier multimodal large language models (MLLMs) and memory systems on ReaLMem reveals predictive personalization as a consistent ceiling, exposes clear performance gaps and bottlenecks between MLLMs and memory systems, and shows that high-quality, temporally informed representations substantially improve personalization. Together, ReaLMem and ChronoProfiler provide an authentic testbed and a simple, effective mechanism for long-term personalization, laying a foundation for future research on lifelong AI companions. 

---
# Advantage Scale Calibration Imbalance in Group-Relative Optimization under Low-Variance Rewards: Diagnosis and Bounded Recovery 

**Authors**: Fei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2609.19164)  

**Abstract**: In verifier-style RLVR, group-relative optimization often treats advantage scale as an implementation detail. This paper separates two low-variance cases: sub-resolution jitter that should not become a preference signal, and credible but small cardinal gaps that should be learned without distorting KL calibration. We propose an advantage-scale three-way calibration interface: the same within-group scale denominator simultaneously determines the reward-branch strength, prompt-level batch weight, and the effective KL calibration induced when the reward branch is re-expressed on the original cardinal scale. This interface explains why RLOO / this http URL can let credible small gaps become KL dominated, whereas GRPO's standard-deviation denominator can amplify tiny gaps without bound. Based on this interface, we further introduce the Reward-Resolution Protocol and MaxNorm-AC, respectively filtering sub-resolution gaps and providing bounded cardinal recovery on credible nonzero gaps. Across dense / MoE architectures and math / code reasoning, MaxNorm-AC improves over the strongest robust-scale baseline while truncating the low-variance inverse-scale tail. 

---
# VisKG-LM: Compiling Knowledge Graphs into Visual Memory for Multiple-Choice Question Answering 

**Authors**: Yixin Peng, Er Jin, Shiwei Luo, Diego Collarana, Stefan Decker  

**Link**: [PDF](https://arxiv.org/pdf/2609.19158)  

**Abstract**: Knowledge graphs are usually integrated into question answering by encoding a retrieved subgraph with a graph neural network and fusing it with the language model in the online inference path. The same subgraph is therefore re-encoded from scratch every time a pair is scored, across training epochs, seeds, and evaluation runs, even though the knowledge graph never changes. We ask whether the retrieved knowledge graphs can instead be compiled once, offline, and then accessed as read-only memory. VisKG-LM shows that it can, by decoupling graph encoding from language reasoning. It serializes each retrieved candidate-specific subgraph as Relation-Labeled Paths and renders the result as an image whose two-dimensional layout preserves the branching structure of the paths. Each image is encoded once, offline, and cached for reuse. At inference, the language model contextualizes the question and candidate from text alone, and only its final layer consults the cached visual memory, reading both its global layout and its local relational detail. The graph information thus enters only after the text has been understood. On the test sets of CommonsenseQA, OpenBookQA, and MedQA-USMLE, VisKG-LMimproves over GreaseLM by $1.2$, $0.8$, and $4.3$ points, respectively, while matching or surpassing GraphVis, a $7$B vision-language model, with only about $400$M online parameters. Against a matched text-only control that receives the identical Relation-Labeled Paths, it gains $4.2$, $6.5$, and $5.1$ points across the three benchmarks. These gains show that the complete visual-memory interface adds value beyond path textualization alone and support compiled visual memory as an alternative to online graph propagation. 

---
# Reflective Recovery: A Self-Supervised Method for Reasoning by Learning from Mistakes 

**Authors**: Qirui Chen, Renjie Pi, Jiahui Gao, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2609.19156)  

**Abstract**: Data-driven fine-tuning is widely adopted to enhance reasoning in Large Language Models (LLMs) due to its simplicity and efficiency. However, mainstream imitation learning methods that rely exclusively on perfect reasoning trajectories suffer from a Scaling Collapse: when the problem set is limited, increasing positive examples fails to yield continuous improvement. However, during inference, an LLM can not guarantee that every intermediate step is correct and is therefore prone to errors. Once such errors arise, the LLM often struggles to recover and may be further misled by the accumulation of previous mistakes. To address this, we propose Reflective Recovery, a simple yet effective self-supervised approach that transforms failed reasoning attempts into recovery training data. Specifically, we extract initial segments of failed trajectories, concatenate them with prompts, and use them to guide the LLM toward valid solutions. Because these segments from failed trajectories are likely to contain errors, this process teaches models to recognize and correct mistakes during reasoning, enabling recovery from erroneous states without relying on external critics or reward models. Evaluated on extensive benchmarks, Reflective Recovery significantly improves performance. On DeepSeek-R1-Distill-Qwen-7B, it boosts accuracy from 30.0% to 37.5% on AIME 2025 and from 37.6% to 47.8% on Minerva. More importantly, analyses demonstrate that it breaks the scaling collapse barrier and enables models to develop emergent self-correction behaviors, representing a paradigm shift from outcome-oriented memorization to process-oriented reflective reasoning. 

---
# Towards Proactive Detection of User-Side Implicit Conflicts in Human-LLM Dialogue 

**Authors**: Jinqiang Wang, Tao Zhu, Huansheng Ning  

**Link**: [PDF](https://arxiv.org/pdf/2609.19155)  

**Abstract**: In Human-LLM dialogue, follow-up user utterances may implicitly conflict with earlier intents, leading the LLM to misinterpret user needs and generate inappropriate responses. A reliable dialogue system should proactively detect user-side conflicts before generating a response and seek clarification when necessary. However, prior work has largely focused on LLM-side conflicts, leaving user-side conflicts underexplored. To fill this gap, we construct UC-Bench, a human-annotated benchmark for evaluating user-side conflict detection. Preliminary experiments show that existing LLMs struggle with this task, especially when conflicts arise from implicit incompatibilities grounded in dialogue history. To improve lightweight LLMs with limited training data, we investigate data synthesis for user-side conflict detection. Existing synthesis methods do not explicitly model the implicit incompatibilities between historical and current user utterances, making it difficult to capture the evolution of conflicts and to generate reliably labeled implicit conflict samples. We propose SynUC, a constraint-guided synthesis method that represents user-side conflicts in a constraint space and uses the SPEAKING framework to guide traceable constraint transformations. Applying SynUC to WildChat, we construct UC-Data, a user-side conflict training set containing 2,487 samples. On UC-Bench, Qwen3.5-4B trained on UC-Data outperforms larger general-purpose LLMs such as Claude Opus 4.8, as well as the same backbone trained on data synthesized by existing methods. 

---
# Neo-Classic: A Benchmark for Evaluating Linguistic-Aesthetic Reasoning in Classical Chinese Poetry 

**Authors**: Han Zhang, Zihan Gu, Zhiyuan Wang, Tianyi Ma, Jiacheng Lu, Xinyan Zhang, Yuhao Wei, Cheng Hua  

**Link**: [PDF](https://arxiv.org/pdf/2609.19154)  

**Abstract**: While Large Language Models (LLMs) achieve high accuracy on established Classical Chinese Poetry benchmarks, it remains challenging to distinguish transferable Linguistic-Aesthetic Reasoning from reliance on familiar pre-training patterns. To address this issue, we introduce Neo-Classic, an evaluation benchmark that combines a constructionist Out-of-Sample (OOS) dataset with a suite of reverse understanding probes. Unlike traditional benchmarks that rely on verification or generation over historical corpora, Neo-Classic comprises strictly metrical poetry authored by contemporary experts, reducing the possibility of direct retrieval. We evaluate state-of-the-art models, including Qwen3-Max, Gemini-3-Pro, and DeepSeek-V3.2, across five behavioral probes designed to test hierarchical constraint satisfaction. Our results reveal two primary limitations. First, a performance gap of 20 to 50 percent emerges when models transition from historical to contemporary texts. Second, models exhibit substantial difficulties in discourse-level ordering tasks, with standard accuracy remaining low (0 to 13 percent). Although expert-level guidance improves the performance of reasoning-enhanced models to 36 percent, a notable gap with human experts persists. These findings suggest that while current LLMs capture local formal patterns, they struggle with global hierarchical planning required for robust Linguistic-Aesthetic Reasoning. 

---
# Stop Removing Stopwords: How an Inherited Preprocessing Default Distorts Legal Text-as-Data 

**Authors**: Gregory M. Dickinson  

**Link**: [PDF](https://arxiv.org/pdf/2609.19153)  

**Abstract**: Empirical legal scholarship increasingly treats judicial text as data, and much of it still runs on sparse, interpretable pipelines -- TF-IDF features and linear classifiers -- because the textual feature is often the object of study, not merely a means to a prediction. Yet these pipelines inherit a chain of preprocessing defaults from mid-century information retrieval that were never validated against classification accuracy, the most entrenched being stopword removal. This study introduces an exhaustive single-word ablation that measures a preprocessing step's effect directly against the downstream objective, and applies it to stopword removal as the hardest case to dislodge. Matching Supreme Court Database labels to Caselaw Access Project opinion texts, it examines two binary tasks that bracket F1 headroom, ideological direction (no-removal baseline F1 ~ 0.68) and constitutional versus non-constitutional law type (~ 0.92), across 7,668 and 7,001 opinions. For each task the analysis approximates the best stoplist any expert could build, removing each of roughly 18,500 candidate words and measuring the effect directly. Three findings follow: generic stoplists in common use fall below the no-removal baseline in every test; even optimized stoplists are statistically indistinguishable from removing nothing; and meta-models trained on word-level features cannot predict which removals help, so list curation has nothing to target. The method generalizes to any inherited preprocessing default, and the result is a caution specific to interpretable legal text-as-data: a step that silently reshapes which features a model sees can distort the very doctrinal and ideological signal such research exists to recover. Leaving stopwords in place is a question of measurement validity. 

---
# FakeSpotter: A content and strategy agnostic Viral Misinformation Detection Tool 

**Authors**: Giovanni Spitale, Federico Germani  

**Link**: [PDF](https://arxiv.org/pdf/2609.19152)  

**Abstract**: Misinformation detection tools often rely on binary true and false classifications or models trained on historical examples, limiting their usefulness when novel misleading narratives emerge. Here, we present FakeSpotter, a content- and strategy-agnostic tool designed to estimate the viral misinformation risk of textual content by measuring structural fingerprints of misinformation rather than directly adjudicating truthfulness. FakeSpotter operationalizes a theory-driven framework across linguistic, narrative, logical, and critical-thinking dimensions, using repeated LLM assessments and domain-specific logistic regression classifiers for short and long texts. In a labelled corpus of 764 texts from social media and FakeNewsNet, FakeSpotter achieved macro F1 scores of 0.788 for short texts and 0.793 for long texts on a held-out test set. FakeSpotter's interpretive layer provides explainable outputs through feature-based scores, signal agreement, and a caution index, and can be used for social listening. These findings suggest that identifying the structural fingerprints of misinformation can support early, explainable, and human-supervised assessment of potentially viral misinformation. 

---
# What Users Think of Generative AI: A Cross-Platform NLP Analysis of Trust and Friction in App Store Reviews 

**Authors**: Md Jafrin Hossain, Umme Nusrat Jahan, Shouvaggo Sharif Shammo  

**Link**: [PDF](https://arxiv.org/pdf/2609.19151)  

**Abstract**: Generative AI (GenAI) applications have achieved rapid consumer adoption, yet little large-scale research examines user-perceived quality, trust, and adoption barriers. We present one of the first cross-application analyses of app store reviews for six major GenAI applications (ChatGPT, Gemini, Microsoft Copilot, Claude, DeepSeek, and Perplexity), comprising 17,012 English-language reviews from Google Play and the Apple App Store. We combine BERTopic topic modeling with RoBERTa sentiment classification and evaluate cross-application differences using chi-square, Kruskal-Wallis, and multinomial logistic regression with Bonferroni correction. Both components are validated against human coding using a stratified sample of 300 reviews. Results show that negative sentiment concentrates in advertising (91%), authentication (89%), server reliability (83%), and subscription pricing (73%). Sentiment differs significantly across applications, with Claude exhibiting the highest negative sentiment (47.7%) alongside a strongly enthusiastic user base, indicating statistically significant polarization. These findings are robust despite unequal review counts across applications. As exploratory observations, a subset of DeepSeek reviews raised geopolitical and data privacy concerns related to its Chinese origin, while a proposed Trust Friction Score summarizes application-specific trust and usability barriers into interpretable dimensions. The study provides validated and actionable evidence on user trust, usability, and adoption barriers in consumer generative AI applications. 

---
# Sampling Reveals Style: Unsupervised, Training-Free Discovery of Prompt-Conditional Stylistic Axes in LLM Activations 

**Authors**: Ajit Mallavarapu, Ziwei Gu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19150)  

**Abstract**: Large language models (LLMs) encode rich stylistic structure in their hidden activations, but discovering which stylistic dimensions are salient for a given prompt typically requires supervised contrastive data. We present a training-free, prompt-conditional alternative: we repeatedly sample completions of a single prompt at elevated temperature, apply Principal Component Analysis (PCA) to the pooled hidden activations, and label the resulting axes automatically from the pole generations. We validate the discovered axes against 245 human-elicited stylistic annotations in a two-phase study. On our strongest model (Qwen-3.5-4B-Instruct), the top two axes match spontaneously requested human dimensions with 72.8% precision and 43.6% macro-recall, and 75.6% of validity ratings judge the axes' polar generations accurate to their labels, with 90.9% adjacent inter-annotator agreement. Discoverability is strongly model-dependent: both Qwen models and Llama-3.2-3B expose human-salient axes, while DeepSeek-7B-Chat drops to 35.3% precision, its leading components dominated by structural rather than stylistic variance. Simple PCA over a model's own decoding variance is thus an effective, low-cost probe of stylistic structure in LLM representations, one that also exposes sharp cross-model differences in how that structure is organized. 

---
# Subliminal Prompting Beyond Static Geometry: Causal Depth and Multi-Token Confounds 

**Authors**: Barath Velmurugan  

**Link**: [PDF](https://arxiv.org/pdf/2609.19149)  

**Abstract**: Subliminal learning shows that language models can transmit a hidden trait through outputs that appear unrelated to it. One proposed explanation, token entanglement, links animal and number tokens through the model's output vocabulary. Yet existing measurements answer different questions: whether outputs co-vary, fixed output vectors align, an answer can be read from a hidden state, or that state causally controls the answer. We measure each separately in a fixed animal-number prompting protocol. From Llama-3.1-8B to 70B, fixed output-vector similarity predicts behavior less well: the paired mean correlation change is -0.080 (95% CI [-0.127, -0.035]). A fixed output-head readout shows no resolved change in normalized depth AUC. To test control, we copy the temporary answer-position state from one number prompt into another at five depths and measure which prompt the final animal score follows. Donor-control AUC rises from 0.254 to 0.540, a paired change of +0.286 (95% CI [+0.272, +0.300]), with increases for all 18 concepts. The contrast remains with exactly eight transformer blocks remaining, while specificity and identity controls remain small or exact. In two Qwen models, scoring every digit in sequence does not recover the positive one-token association. Per-token averaging instead creates a positive pooled association that disappears after controlling number width, revealing a length confound. Thus, fixed geometry, observational readability, causal timing, and multi-token measurement are distinct properties of this frozen prompting channel. They constrain token-level explanations but do not identify the mechanism of training-time trait transfer. 

---
# Modality Discrepancy Transformer for Ambivalence and Hesitancy Recognition 

**Authors**: Shiyu Luo, Yu Wang, Jiawen Huang, Zhaoxiang Xiao, Chenxi Huang, Qi Zhang, Bin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19148)  

**Abstract**: Ambivalence and hesitancy (A/H) are affective states in which individuals express contradictory signals across facial, vocal, and linguistic channels. Automatically recognising A/H in clinical videos requires detecting cross-modal disagreement -- the signal that standard fusion methods suppress. Based on the conflict-aware multimodal fusion framework of Bekhouche et al., we present the Modality Discrepancy Transformer (MDT). MDT enriches the original 6-token design to a 9-token representation comprising three modality embeddings, three absolute-difference features, and three Hadamard-product discrepancy features learned through linear projections. These nine tokens undergo Transformer self-attention, with FiLM-based text-conditioned modulation and LoRA fine-tuning as core architectural components. A text-guided late fusion branch blends a text-only auxiliary head with the full multimodal output at inference. On the BAH dataset from the 3rd ABAW Challenge, MDT achieves 0.7408 Macro F1 on the labelled test split and 0.7368 on the private leaderboard, outperforming the strongest published baseline by over 10 points while training in under 20 minutes on a single GPU. 

---
# Coding Agents with an Obstacle-Aware Harness for Safe Robot Manipulation 

**Authors**: Bingxin Xu, Yuzhang Shang, Zhen Dong, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2609.20822)  

**Abstract**: Coding agents have emerged as a promising paradigm for robot manipulation: a language model writes the robot controller as a program, and agents built in this way now operate robots without robot-specific this http URL this paradigm is also safe, however, has not been asked. We evaluate coding agent under a safety constraint, where each task pairs a manipulation goal with an obstacle the robot must not touch. The agent pursues the goal but collides with the obstacle in most cases, treating task completion as its sole objective while neglecting safety. The agent reasons about the obstacle in its traces, and the prompt already forbids touching it, so neither perception nor instruction is at fault; the fault lies in the planning, where the stated constraint never becomes a priority. By decomposing manipulation into a route phase and a contact-rich moment, we locate the source of the failure. Along the route, the model cannot prioritize the safety constraint, having no notion of a clearing route and none of replanning once a chosen route becomes infeasible. At the contact, it is unaware that contact execution is bounded by the same constraint. To close this gap, we present SafeHarness, which equips the model with two obstacle-aware harnesses that enable it to prioritize the safety constraint. Obstacle-aware route planning grounds the objects as bounding boxes and draws candidate routes over them as sequences of waypoints. The agent then plans a route in advance, verifies it, replans when necessary, and only then executes it. Obstacle-aware contact execution instead selects the contact position so that the contact itself avoids the obstacle. SafeHarness attains 71.9% task success and 87.5% collision avoidance, surpassing the previous SOTA by 6.5% and 27.0%, respectively. These results are $2.3\times$ and $1.5\times$ those of the same agent without harnesses. 

---
# An Empirical Study of Harness Design for Coding Agents 

**Authors**: Run-Ze Fan, Zihao Zhang, Simin Ma, Yebowen Hu, Shouju Wang, Kaiqiang Song, Fei Liu, Hamed Zamani, Xiaoyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20804)  

**Abstract**: Coding harnesses shape how autonomous coding agents translate model capabilities into long-horizon software-engineering performance, yet existing work typically evaluates harnesses as monolithic systems, leaving the effectiveness of individual components unclear. To enable component-level comparisons, we study this question with a lightweight coding harness whose execution loop is fixed while three components are varied: planning, action space, and context management. Across four models evaluated on SWE-Bench Verified and Terminal-Bench 2.1, we evaluate 176 matched settings spanning five context-management strategies, four context-window budgets, and targeted ablations of planning and action space. We find that: (1) Context management becomes increasingly valuable as the context-window budget tightens, with most of its benefit coming from preventing context-overflow failures. (2) Staging rule-based elision before LLM-based summarization provides the strongest overall efficiency among the context-management strategies, whereas making elided content recoverable adds machinery that models rarely use and yields no accuracy gain. (3) Planning shifts from an accuracy scaffold for weaker models to a cost saver for stronger models, with little change in accuracy. (4) Predefined tools improve performance for models with weaker bash proficiency, whereas bash-capable models can operate effectively with a bash-only interface and achieve substantially lower cost, especially on command-line-centric tasks. Trajectory-level analysis explains these effects: context management extends execution trajectories without substantially altering agent behavior, planning changes where trajectories stop, and the action space changes the granularity at which code is written. These findings inform model- and budget-aware harness design and provide a modular framework for evaluating future harness components. 

---
# Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL 

**Authors**: Juzheng Zhang, Disha Makhija, Manoj Ghuhan Arivazhagan, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2609.20715)  

**Abstract**: Agent trajectories record what an agent does and what happens next. Yet standard supervised fine-tuning (SFT) applies loss only to agent-authored action tokens, using environment observations as context but not as prediction targets. We ask whether this convention provides the best initialization for subsequent reinforcement learning. We introduce ActObs, which also supervises the observation tokens already present in each trajectory. Although deployed agents never generate observations, learning to predict them encourages the policy to model action consequences without adding data, parameters, sequence tokens, or forward passes. The methods perform similarly after SFT but diverge after GRPO. On Qwen3-4B, GRPO from ActObs achieves higher pass@k at every evaluated sampling budget than its action-only counterpart on Terminal-Bench 2.0. On Qwen3-8B, it trades some pass@1 reliability for higher pass@k (+3.4 pp at pass@16) and solves more distinct tasks. The advantage extends to cross-domain code editing on aider-polyglot (+4.2 pp at pass@1 at 4B), whose tasks are unseen during SFT and RL. ActObs retains more entropy during RL while requiring less policy movement, leaving the final policy closer to its SFT initialization. Our analysis traces this difference to SFT: action and observation gradients rapidly become orthogonal, while action-only training leaves a large residual observation gradient and degrades environment prediction below the base model. Joint supervision prevents this one-sided specialization, preserving consequence prediction and preparing the policy for downstream exploration. 

---
# PAA: The Probabilistic Allen Algebra: A Generative and Complete Probabilistic Extension of Allen's Interval Relations 

**Authors**: Julian Eggert  

**Link**: [PDF](https://arxiv.org/pdf/2609.20634)  

**Abstract**: Allen's interval algebra is a qualitative calculus for temporal relations, but its thirteen base relations are crisp predicates over exact interval boundaries. This is inadequate for temporal information from language, perception, databases, or uncertain histories, where times, durations, and boundaries are uncertain and expressions such as "just before" or "roughly during" have graded meaning. We develop the probabilistic Allen algebra (PAA): a generative and complete extension in which relation probabilities are derived from distributions over interval boundaries rather than assigned as scores. Time points are Gaussian; intervals have Gaussian midpoints and truncated-Gaussian durations. Every relation is a boundary-ordering predicate in one common probability space: point-point relations reduce to error functions, and point-interval and interval-interval relations to multivariate Gaussian orthant probabilities induced by linear inequalities. Contact relations (meets, starts, finishes, equals) receive positive measure through a tolerance band, and under a single tolerance the thirteen relations form a true partition that recovers crisp Allen as the tolerance vanishes. The construction derives Allen's taxonomy rather than positing it: coarse predicates such as precedence, overlap, and containment are unions of leaves whose probabilities are leaf sums, and this hierarchy is preserved as intervals collapse to points and thirteen relations reduce to five and then three. Each relation further decomposes into correlation-aware temporal primitives in the spirit of CIDOC CRM. The algebra is scale-invariant and separates graded expressions such as "shortly before" from contact relations. All results are Monte-Carlo validated and shipped as an open, tested Python package. 

---
# Language-model groups overstate consensus when replaying human deliberation on a reasoning task 

**Authors**: Tengfei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2609.20543)  

**Abstract**: Full-consensus rates are often treated as indicators of collective cognition, yet depend on how participation and final states are operationalized. We replayed 100 held-out human Wason groups with matched large language model (LLM) agent groups, seeding one belief-anchored agent per participant's pre-discussion answer and scoring agents and people with the same code. Across human scoring definitions, estimates ranged from 24.0% to 57.0%; about one fifth of participants never posted, whereas agents almost always did. Agent groups remained more consensual in two post-unblinding sensitivity analyses: the submit-based comparison (n = 98) yielded gaps of 34.0 and 43.9 percentage points for chat and reasoning modes, and the participation-matched comparison (n = 45) yielded gaps of 34.1 and 44.4 points. These complementary routes reduced different measurement asymmetries yet converged within 0.5 percentage points. The gap persisted without early stopping and under a reparameterization removing the memorizable answer; reasoning-mode groups then agreed nearly unanimously, mostly on incorrect answers. Simulated consensus did not track collective accuracy, and belief-anchored agent groups were biased estimators of the human group-outcome distribution in this setting. These analyses provide a scoring-explicit basis for assessing simulated-group estimates of human deliberative outcomes. 

---
# Model-Agnostic and Language-Agnostic Voice Pipeline Improvement for the Agriculture Domain 

**Authors**: Aakash Singh, Lakshmi Pedapudi, Chandrashekar M S, Sanyam Singh, Naga Ganesh, Vineet Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.20504)  

**Abstract**: FarmerChat is Digital Green's AI-powered agricultural advisory assistant for smallholder farmers, who access it in their own language through text, voice, or photographs. Voice is a critical channel for this population, yet field-recorded speech is challenging for general-purpose automatic speech recognition (ASR) because recordings frequently contain machinery noise, background media, competing speakers, and domain-specific agricultural vocabulary. These conditions disproportionately affect crop, pest, chemical, and quantity terms that carry the meaning of a farmer's query.
We present a modular, model-agnostic pipeline for improving ASR quality in FarmerChat without fine-tuning or replacing the underlying ASR model. The pipeline combines gated audio enhancement, speaker diarization and target-speaker selection, ASR, domain-aware correction using a weighted agricultural lexicon, and a quality gate for detecting unreliable transcripts. Only the diarization stage is fine-tuned; all other stages use off-the-shelf models behind common interfaces.
We evaluate the pipeline on human-annotated FarmerChat recordings in Hindi, Telugu, and Odia using word error rate (WER) and a domain-weighted error rate that gives greater importance to agricultural terminology. The largest improvements occur on multi-speaker recordings, where target-speaker selection prevents competing speech from entering the transcript. Across the full corpus, the pipeline reduces WER by 16-23% relative on three cloud ASR models and by 5% on an on-device model. On multi-speaker recordings, the reductions are 32-42% for the cloud models and 16% for the on-device model. All reported reductions are statistically significant. These results show that targeted preprocessing, speaker selection, and domain-aware post-processing can substantially improve agricultural speech transcription while preserving the underlying ASR model. 

---
# Think Thrice Before Reranking: Multi-perspective Evidence and Reasoning Integration for Text Reranking 

**Authors**: Lijun Liu, Zhengzong Chen, Wenyan Li, Yuanyuan Zhao, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20131)  

**Abstract**: Reasoning-based reranking with Large Language Models (LLMs) has shown promising improvements in text ranking. However, current methods predominantly rely on a single reasoning trajectory, resulting in rankings that are susceptible to reasoning errors and inherently constrained in modeling the multifaceted signals underlying document relevance. To resolve this dilemma, we propose MERIT-Rank(Multi-perspective Evidence and Reasoning Integration for Text Reranking), a framework that models complementary reasoning trajectories to improve reranking robustness. MERIT-Rank formulates a Multi-Trajectory Reasoning Space (MTRS) that evaluates query-document relevance from multiple perspectives and introduces a joint reranker that consolidates these reasoning paths into a unified ranking decision. We further develop Progressive Rank Policy Optimization (PRPO), a progressive training framework that stabilizes reasoning trajectories while continually improving ranking quality through staged optimization objectives. Experiments on both reasoning-intensive and traditional retrieval benchmarks show that MERIT-Rank consistently achieves superior performance over competitive baselines. The 4B model notably outperforms most 7B and even 32B rerankers on BRIGHT. 

---
# MATCH: Model-Aware Tool Learning with Curriculum Scheduling and Hierarchically Gated Rewards 

**Authors**: Shihao Liu, Hao Yin, Lijun Liu, Zhengzong Chen, Yuanyuan Zhao, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20082)  

**Abstract**: Tool learning enables large language models (LLMs) to use external tools for tasks beyond parametric knowledge. Reinforcement learning can optimize tool-call behavior from feedback, but current methods still face two problems: fixed-threshold curricula can become misaligned with the policy's evolving capability boundary, and additive rewards can leak argument-level credit when the predicted tool is wrong. To address these problems, we propose MATCH, a closed-loop framework for model-aware tool learning with curriculum scheduling and hierarchically gated rewards. Model-Aware Curriculum Learning (MACL) maintains reward-derived sample difficulty that co-evolves with the policy, and each epoch selects samples near the current capability boundary together with a top-k pool of harder cases. Hierarchical Tool-call Gated Reward (HTGR) scores tool name, argument key, and argument value as a gated chain, granting credit at each level only when prerequisites hold. The same HTGR rewards drive both GRPO updates and MACL's difficulty refresh, closing the loop between policy optimization and sample scheduling. On API-Bank and BFCL V3, MATCH reaches 72.19% and 62.87% overall accuracy, outperforming the main supervised and RL-based baselines. Backbone experiments further show consistent improvements across four backbones from two model families. 

---
# Marginal utility, matrix factorization, and the Key-Value (KV) cache: a unified information-economic framework for sovereign geo-mining inference 

**Authors**: Caroline Gans Combe  

**Link**: [PDF](https://arxiv.org/pdf/2609.20068)  

**Abstract**: This paper builds a theoretical bridge between the economic notion of marginal utility and two machine-learning constructs, matrix factorization and the Key--Value cache of transformer language models. The singular value spectrum of a rating matrix is shown to be a diminishing marginal utility schedule for latent factors, the eigenvalue spectrum of the projected covariance operator to be the marginal utility schedule of a model's learned representation, and cache eviction and low-rank cache compression to be instances of constrained utility maximization under a memory budget. The three collapse into a single allocation rule: retain the top dimensions whose eigenvalue exceeds the shadow price of the binding constraint. The framework is applied to the automated extraction of structured information from geo-mining documents, where it motivates a multi-pass inference protocol, a layer-wise TIES model merging procedure, and a selection policy combining extraction quality, localization drift and energy, scalarized with a Conditional Value-at-Risk term on drift. Two empirical contributions are reported. An 11.2-million-parameter hierarchical classifier, trained in about five minutes on a single GPU, reaches 90.0 per cent level-1 accuracy on a held-out test set from a 973-document uranium-exploration corpus, against 92.0 per cent for a proprietary model on a fifty-document human audit of the same corpus, at a latency of 2.62 ms per card against approximately 2,000 ms for the API and at negligible cost. A diagnostic of uniform-density TIES merging exposes a reproducible degenerate mode in which the merged model returns token-identical outputs across five geographically distinct districts while declaring high confidence; re-executing the merge under layer-wise calibrated densities removes that signature on the diagnostic sample. The full-scale extraction benchmark, including LoRA fine-tuning, is reported as projected rather than measured and remains an empirical extension of this work. 

---
# AI Should Facilitate Democratic Deliberation at Scale 

**Authors**: José Ramón Enríquez, Jiaxin Pei, Alex Pentland  

**Link**: [PDF](https://arxiv.org/pdf/2609.20059)  

**Abstract**: AI systems can strengthen democracy by supporting deliberation at scale by addressing cognitive, social, platform-design, and market-driven frictions, while preserving human agency. Unlike proposals such as liquid democracy that restructure representation through vote delegation, in this position paper, we argue that AI-assisted deliberation offers a more promising path by lowering barriers to meaningful engagement without substituting machine judgment for human choice. Drawing on evidence from online deliberation platforms and experimental research, we identify four guiding principles: preserving agency and autonomy, encouraging mutual respect, promoting equality and inclusiveness, and augmenting rather than substituting active citizenship. We also address critical challenges, including alignment, sycophancy, training bias, and over-reliance on AI systems. We call on the machine learning community to develop deliberation-focused AI systems evaluated not on engagement metrics but on their capacity to facilitate informed, representative, and friction-robust discourse. 

---
# The Missing Complement: State-Conditioned Minimal Sufficient Evidence for Coding Agents 

**Authors**: Zhexi Feng, Ruiyi Zhang, Yongbo Yang, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2609.20050)  

**Abstract**: A coding agent halfway through an issue has already read much of what a retriever ranks highest. Relevance is scored per passage, but sufficiency belongs to the set: a ranker can fill its budget with variants of one required fact and leave the decision unsupported. We formulate state-conditioned minimal sufficient evidence recovery: given a captured agent state, recover a compact evidence combination that supplies the support its next decision still lacks. SERBench measures this on 500 held-out states from 45 repositories, recording what the agent has seen and crediting only sets that cover every fact the current decision was annotated to require. MSS-Complement treats acquisition as set construction, not ranking. Three semantic calls propose a jointly sufficient set, search for what it lacks, and return 4-8 intact source units within 6,144 tokens. One configuration, fixed on calibration data, recovers a complete set for 73.0% of those states at five items and 80.6% at eight, against 61.4% and 72.4% for Qwen3 embedding with reranking. A matched control ranking by similarity alone reaches 66.6%, placing the gain in the set-level policy, not the computation. From frozen repository source with no gold-derived pool, the lead is 5.0 points. On AMA-Bench it answers from a 76.2% smaller answer prompt, with accuracy 2.08 points above that benchmark's own memory agent. Removing one required group from an otherwise complete set costs 12.3 and 11.1 points of repair-localization precision under two executors. Retrieval for agents is better posed as recovering what a decision lacks than re-ranking what an issue resembles. 

---
# Geopolitical Divisions Across Languages in Large Language Models 

**Authors**: Maxim Chupilkin  

**Link**: [PDF](https://arxiv.org/pdf/2609.20005)  

**Abstract**: People increasingly turn to AI chatbots for news and explanations of world events. But do they receive the same political answers when they ask in different languages? Here we show that the language of a question can change how the same AI systems assess the war in Ukraine. We ask GPT, Claude and Gemini to evaluate twenty statements about the war in 112 languages, collecting 67,200 responses. The balance between Russia-leaning and Ukraine-leaning responses differs across languages. When we group responses by countries' official languages, they follow a pattern resembling worldwide political divisions: relatively more Russia-leaning answers correspond to more favourable public views of Russia, less support for Ukraine in United Nations votes, and less aid to Ukraine. The broad pattern recurs across all three models and remains when individual statement pairs are removed. Our findings suggest a possible route through which information warfare may shape the text used to train AI models, which may in turn spread geopolitical biases. 

---
# Uni-LaDiR: Latent Diffusion Unifies Multimodal Reasoning 

**Authors**: Haoqiang Kang, Yizhe Zhang, Nikki Lijing Kuang, Yian Ma, Lianhui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2609.19878)  

**Abstract**: Multimodal reasoning requires models to draw on information from multiple modalities throughout the reasoning process. Yet existing methods often concatenate modality-specific thought tokens in a single sequence, leaving the model to bridge representational differences as it reasons across modalities. We introduce Uni-LaDiR (Unified Latent Diffusion Reasoner), a framework that brings these thoughts into a shared latent space for reasoning. A unified encoder maps teacher reasoning steps from different modalities into shared thought tokens, trained to preserve the information needed for later reasoning steps and the final answer or action. Because the same context can support multiple valid next steps, we use diffusion to predict the next block of thought tokens from the input and preceding blocks. Jointly training the encoder and diffusion reasoner with shared model weights encourages thought tokens to be both useful for the task and predictable from the available context. At inference, the model generates these tokens without teacher observations. Across eleven vision-language model (VLM) benchmarks and two vision-language-action (VLA) suites, Uni-LaDiR achieves relative gains over the strongest evaluated baselines of 7.3% on visual reasoning tasks and 6.1% on robot manipulation tasks. 

---
# Reproducibility is not construct validity: LLM measurement of institutionally situated communication 

**Authors**: Veronika Batzdorfer, Carlo Romano Marcello Alessandro Santagiustina  

**Link**: [PDF](https://arxiv.org/pdf/2609.19866)  

**Abstract**: High annotation reproducibility does not necessarily imply that an LLM-inferred measure captures the construct it is intended to measure. We test this distinction using a dataset from the European Commission's AI Act consultation, linking structured survey responses to free-text consultation submissions from the same stakeholders. LLM annotations of consultation submissions are highly reproducible (intraclass correlations > 0.99), yet show limited convergence with survey-reported measures of the nominal construct they were intended to approximate. Divergence between survey-and LLM-inferred text-based measures varies systematically across stakeholder groups: business associations express greater concern about AI risks in text-based consultations than in survey responses ({g} = +1.0), whereas public authorities and several nonbusiness groups show smaller or negative divergences. Divergences between scores suggest positive spatial autocorrelation across European countries (Moran's I = 0.347, p = 0.036), indicating that stakeholders from neighboring countries tend toward more similar text-based stances towards AI safety concerns. Despite divergence, survey-reported concerns remain strongly associated with support for explainability across all divergence levels. These results demonstrate that LLM annotation reproducibility can coexist with poor construct correspondence and motivate validation procedures that distinguish reproducibility, construct validity, and communication context variation when LLMs are used as measurement instruments. 

---
# AutoData: Agentic Search for Pre-training Data Selection 

**Authors**: Yan Meng, Dhruv Srikanth, Bingchen Zhao, Zhengyao Jiang, Yuxiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19754)  

**Abstract**: LLM agents have recently shown promise in automating machine learning engineering by editing model and training code under execution feedback. Data, however, remains largely outside this agentic optimisation loop. We frame pre-training data selection as heuristic engineering over per-document features, i.e., lexical statistics, categorical labels, and perplexity. We introduce AutoData, an agent that searches directly over executable selection algorithms. Unlike prior data mixture methods that optimise weights over a fixed set of domains, AutoData searches a richer program space of scoring, stratification, and stochastic selection rules, discovering feature interactions automatically by iteratively refining algorithms with validation feedback from a proxy model. Within an overnight search, AutoData discovers a selection algorithm that outperforms existing human-designed curation pipelines. Despite being searched only on this small proxy, the discovered recipe transfers to larger scales and improves the downstream metric CORE. These results suggest that data engineering can be treated as an agentic machine learning problem, extending autonomous research from model and training-code optimization to the data. 

---
# Learn Your Own Thoughts: Abstract Token Curriculum 

**Authors**: Khashayar Gatmiry, Avrajit Ghosh, Parsa Mirtaheri, Jason D. Lee, Nika Haghtalab, Emmanuel Abbe, Peter Bartlett  

**Link**: [PDF](https://arxiv.org/pdf/2609.19717)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable reasoning capabilities by utilizing chain-of-thought (CoT) as a scratchpad for intermediate stages of thinking. However, CoT techniques require explicit supervision on thinking tokens, which requires rich, task-specific data. In this work, we propose Abstract Token Curriculum (ATC), a novel curriculum learning framework that elicits effective continuous intermediate representations without direct supervision or manual scratchpad design. ATC gradually increases problem complexity through a sequence of distributions, training the model to develop internal abstract ``thoughts'' in the continuous representation space. This paper provides both theoretical and experimental evidence for the benefits of ATC and its advantages over previous methods for training continuous thoughts. Theoretically, we show that for learning parity functions with single-layer softmax attention using ATC, attention naturally focuses on the CoT tokens in the context that provide the ``easiest path'' to predicting the next token. Experimentally, we show ATC's effectiveness on graph reachability and arithmetic learning tasks. 

---
# Scientific Image Quality Assessment via Multi-modal Retrieval-Augmented Generation 

**Authors**: Yinuo Zhang, Bingshuo Liu, Zhiying Tu, Dianhui Chu, Qingbin Liu, Xi Chen, Jiang Bian, Xiaoyan Yu, Dianbo Sui  

**Link**: [PDF](https://arxiv.org/pdf/2609.19634)  

**Abstract**: This paper proposes a Retrieval-Augmented Generation (RAG) framework for scientific image quality assessment, designed to simultaneously address both the understanding track (SIQA-U) and the scoring track (SIQA-S) of the SIQA challenge. We construct a multimodal index that integrates textual semantics with fine-grained visual features, and develop a multi-route retrieval and fusion mechanism to provide large language models with highly relevant reference cases, thereby enhancing their capability to evaluate complex scientific images. Experimental results demonstrate that the proposed framework effectively aligns with the judgment criteria of human experts. Ultimately, our method achieves 1st place in the SIQA-U track of the SIQA challenge at the ICME 2026 Grand Challenges. 

---
# From Intent to Action: Benchmarking LLM Safety in Vehicle Voice Command Authorization 

**Authors**: Diba Afroze, Xingli Zhang, Yazhou Tu, Xiali Hei  

**Link**: [PDF](https://arxiv.org/pdf/2609.19630)  

**Abstract**: Large language models (LLMs) are increasingly integrated into vehicle voice assistants. But linking natural-language requests to vehicle functions creates a safety-critical authorization problem. Before executing a command, the system must choose whether to execute, refuse, clarify, require confirmation, defer to manual control, trigger an emergency response, or make no tool call. To our knowledge, prior evaluations do not isolate this pre-action decision across speaker role, authentication status, vehicle state, and tool availability. We introduce a 202-scenario benchmark with Reference Decisions under a seven-class taxonomy. We evaluate two local open-weight models and three API-based LLMs using Decision Alignment and safety-specific error metrics. Alignment ranges from 40.1% for Llama 3.2 3B to 89.1% for Gemini 3.1 Pro Preview. The API-based models score between 83.2% and 89.1%, with no statistically significant differences among them. Even these models produce two to three False Executes among 161 non-execution scenarios, and persistent errors remain in confirmation and manual-control decisions. A controlled Llama 3.2 3B ablation increases alignment to 40.1% under the structured authorization policy, versus 28.2-29.2% under schema-only and generic-safety baselines, but it does not eliminate False Executes. Structured LLM decisions are therefore insufficient as a standalone safety mechanism, and deployment requires an independent enforcement layer that verifies tool permissions and vehicle-state constraints before invoking any vehicle function. 

---
# Red-Teaming Auto Mode: Improving Blocking Classifiers Against Malign Coding Agents 

**Authors**: Alex Remedios, Simon Storf, Fabien Roger, John Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2609.19587)  

**Abstract**: To keep coding agents from going off the rails, production systems now review each proposed action with a blocking monitor that can reject it before it runs (Auto Mode in Claude Code, Guardian in OpenAI's Codex). Prior evaluations of such monitors largely measure robustness to accidental harm or prompt injections from untrusted sources looking to hijack the agent. Less understood is how they hold up when the agent they monitor is persistently misaligned. To understand this risk, we task an adversarial agent with evading production blocking monitors and causing catastrophic harm, e.g. by exfiltrating its own weights. We find that when instructed with high-level attack strategies, adversarial agents can succeed through several distinct mechanisms, such as agent-generated prompt injection against the monitor, multi-agent attacks, and malicious compaction. In particular we find that in 79% of trials, the agent can use an injection attack against Auto Mode and Guardian to run arbitrary bash commands. We also find that it is possible to greatly improve Auto Mode through design changes like enhancements to tool coverage, transcript formatting and an agentic monitor stage. Even with these improvements, preventing multi-context attacks at an acceptable cost remains an open problem. By detailing our red-teaming methodology and highlighting new attack vectors, we aim to help defenders evaluate their mitigations against the possibility of persistent malign coding agents. Code is available at this https URL. 

---
# Large Language Model Agents for Evidence Based Genetic Disease Severity Classification 

**Authors**: Tohid Ghasemnejad, Ahmadreza Argha, Mark Grosser, John Wang, Min Yang, Thantrira Porntaveetus, Tony Roscioli, Nigel H. Lovell, Mahmoud Aarabi, Hamid Alinejad-Rokny  

**Link**: [PDF](https://arxiv.org/pdf/2609.19569)  

**Abstract**: Disease severity classification for genetic conditions is subjective and labor-intensive, creating bottlenecks in genomic screening, where commercial panels vary widely in size and overlap. We developed an autonomous AI agent integrating Reasoning and Acting (ReAct) with Retrieval-Augmented Generation (RAG) to classify 10,211 Human Phenotype Ontology terms. It uses American College of Medical Genetics (ACMG)-endorsed severity guidelines and American College of Obstetricians and Gynecologists (ACOG) quality-of-life criteria to retrieve PubMed literature, generate interpretable reasoning chains, and independently verify claims. At the phenotype level, using expert-curated cohorts, the agent achieved 93.55% accuracy (MCC 0.9237) with 82.6% to 91.4% of claims supported by direct evidence or valid inferences. Gene-level severity was aggregated across 8,738 pairs, identifying 3,283 autosomal recessive pairs with severe or profound presentations. External validation showed 95.2% concordance with Mackenzie's Mission gene list. This system enables standardized panel design by providing reliable, automated classification supported by direct evidence. 

---
# Finding Common Ground: Graded Communal Knowledge in Bluesky Starter Packs 

**Authors**: Sagar Kumar, Lawrence Swaminathan Xavier Prince, Julia Mendelsohn, Brooke Foucault Welles, Nicholas W. Landry  

**Link**: [PDF](https://arxiv.org/pdf/2609.19549)  

**Abstract**: Communication is made possible by common ground---the unspoken knowledge that people share and presuppose of one another, whether that be online or offline. In his conception of common ground, Clark (1996) distinguishes between personal and communal common ground, and asserts that the latter is graded: the more community affiliations two people share, the more common ground they share as well. Social media research has invoked this mechanism to explain how users connect, but it has gone largely untested because community memberships are rarely visible and, where they are, they are coupled to user interactions in a way that leads to conflating effects. To circumvent these challenges, this study repurposes Bluesky starter packs (SPs) as user-curated community affiliation labels. Across 191,648 pairs of users, we show that shared lexical repertoire---our proxy for common ground---grows monotonically with the number of SPs that users share, with users sharing a single pack being roughly twice as similar as equally connected strangers. A semantic renormalization of SP co-membership shows furthermore that it is more so the number of topically \emph{distinct} communities, rather than the raw count, in which common ground is graded. Finally, we show that community co-membership adds to common ground independently of proximity in the Bluesky follow network. These results lead to the conclusion that community membership is a measurable, separable, and semantically structured carrier of common ground. Reading it as such makes common ground observable before an exchange rather than inferred from it, and thus opens the door for large-scale observational approaches to a set of questions that have so far only been posed in the laboratory. 

---
# When Hiring Becomes Agent-Mediated: Evaluating Access and Recurrence in Two-Agent Résumé Screening 

**Authors**: Jian Gao, Hang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19530)  

**Abstract**: Hiring is bilateral: employers assess fit, while candidates present and defend evidence of their qualifications. Yet résumé screening, the first gate, is commonly automated as a static, one-call judgment over a résumé-job pair. We study a two-agent alternative in which employer-side and candidate-side agents represent these roles, exchange evidence, and update their judgments before deciding who advances. We compare procedures on 600 constructed résumé-job pairs using GPT-5.5 and Claude Opus 4.7. Two-agent screening advances more applications (33.3% to 39.3% for GPT-5.5; 34.0% to 35.5% for Opus 4.7). Across three runs on the common 191-pair borderline pool, pass-instance rates rise from 4.5% to 26.2% and from 6.5% to 16.1%, respectively. This is not a uniform relaxation: two-agent screening rejects applications one-call advances, changing decisions in both directions. At similar pass volumes, the procedures advance different applications, and no one-call threshold recovers applications consistently selected by two-agent screening. Among discovery-selected cases re-executed in fresh runs, two-agent-only selections recur less often than shared selections, clearly under GPT-5.5 and less certainly under Opus 4.7, while a separate one-call follow-up shows no comparable decline. As hiring becomes agent-mediated on both sides, the screening procedure, not only the model behind it, shapes who reaches human review and how reliably that access recurs. 

---
# EconSkills: Studying Skill Transfer and Retrieval for Web Agents on Live Economic Data 

**Authors**: Yinzhu Quan, Zefang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19523)  

**Abstract**: Web agents often revisit the same sites, yet most evaluations discard the procedures learned in earlier successful interactions. We introduce EconSkills, a skill library and evaluation framework that distills verified EconWebArena trajectories into parameterized standard operating procedures for retrieving live economic data. Each skill records its scope, navigation procedure, site-specific guidance, verification checks, and recovery steps while replacing source-instance values with placeholders. EconSkills separates two questions: whether a known relevant procedure transfers to a held-out task, and whether an agent can retain that benefit when selecting from a library. In controlled transfer, matched skills improve success over no-skill prompting and require fewer steps on paired successes, while abstraction is substantially more effective than replaying raw trajectories. At library scale, retrieval is competitive with the no-skill baseline overall and performs best on directly covered tasks; coverage-stratified outcomes show that approximate matches on uncovered tasks offset these gains. Browser trajectories further identify when procedural guidance shortens portal-specific navigation and when semantic verification remains necessary. These results establish that reusable economic web procedures can transfer across task instances and provide a concrete design target for coverage-aware selection and context delivery. 

---
# Safety Beyond the Interface: Detecting Harm via Latent States in Large Language Models 

**Authors**: Alizishaan Khatri, Chiquita Prabhu, Omkar Neogi  

**Link**: [PDF](https://arxiv.org/pdf/2609.19472)  

**Abstract**: Autonomous systems increasingly rely on Large Language Models (LLMs) yet the safety infrastructure surrounding these models introduces latency and compute overhead. This limits utility in resource-constrained, time-critical deployments. Existing external guardrail models remain blind to the model's internal workings, creating a fundamental assurance gap. We ask: does the model already know when the content is harmful? We extract activations from LLaMA-3.1-8B and train lightweight MLP classifier probes (12.6M parameters) to detect harmful prompts. Evaluated on WildJailbreak, Beavertails, and AEGIS 2.0, our probes achieve F1 scores of 99%, 83%, and 84%, respectively competitive with 1000x larger guard models while cutting latency and compute costs. 

---
# From Models to Systems: A Comprehensive Survey of Efficient Multimodal Learning 

**Authors**: Pan Wang, Siwei Song, Hui Ji, Siqi Cao, Heng Yu, Zhijian Liu, Huanrui Yang, Yingyan Celine Lin, Beidi Chen, Mohit Bansal, Xiaoming Liu, Pengfei Zhou, Ming-Hsuan Yang, Tianlong Chen, Jingtong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19445)  

**Abstract**: The rapid expansion of multimodal models has surfaced formidable bottlenecks in computation, memory, and deployment, catalyzing the rise of Efficient Multimodal Learning (EML) as a pivotal research frontier. Despite intensive progress, a cohesive understanding of what, how, and where efficiency is manifested across the learning stack remains fragmented. This survey systematizes the EML landscape by introducing the first structured, model-to-system taxonomy. We distill insights from over 300 seminal works into three hierarchical levels--model, algorithm, and system--addressing architectural parsimony, execution refinement, and hardware-aware orchestration, respectively. Moving beyond a purely categorical review, we offer a methodological synthesis of the vertical synergies between these layers, elucidating how cross-layer co-design contributes to the fundamental "Efficiency-Utility-Privacy" trade-off. Through an integrative case study of Multimodal Large Language Models (MLLMs), we trace the field's evolutionary trajectory from initial structural adjustments to modern full-stack resource orchestration. Furthermore, we provide a holistic discussion and application-specific optimization blueprints for diverse domains and posit a paradigm shift toward self-regulating intelligence, where efficiency is an intrinsic, emergent property of the model's fundamental design rather than a post-hoc constraint. Finally, we present open challenges and future directions that will define the trajectory of EML research. This survey establishes a structured framework for multimodal systems that are not only high-performing and generalizable but natively efficient and ready for ubiquitous deployment. A continuously updated version is available at this https URL. 

---
# BurnRiSc: Toward Non-Invasive Burnout Screening in Open Source from Public Repository Signals 

**Authors**: Timofey Sanko, Yuan Tian, Mariam Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2609.19422)  

**Abstract**: Burnout is a chronic occupational syndrome, and open source is close to a worst case for it: maintainers absorb unbounded demand with no manager to reallocate work and no organization to notice decline. The cost is not only personal. Burnout precedes withdrawal, and in projects sustained by a handful of maintainers, one departure can break infrastructure that thousands of downstream systems depend on. Yet the field has no way to see it coming: self-report inventories, the only existing measure, miss exactly the contributors most in need of detection and cannot be applied retroactively, so the field cannot even ask how common burnout is or what helps.
We present BurnRiSc, a framework that operationalizes the Oldenburg Burnout Inventory's two dimensions, exhaustion and disengagement, as 14 behavioral and linguistic signals computed from GitHub activity and scored against each contributor's own history. The signals aggregate into two weighted dimension scores, with weights learned from labeled cases, and average into a monthly Burnout Risk Score (BRS). In a preliminary evaluation across 68 contributors in ten repositories (ten disclosed burnout cases, twelve comparable-volume collapses, and 46 comparison contributors), sustained BRS elevation precedes 6 of 10 disclosures by 6-15 months, 8 of 10 when adding peak BRS as a second criterion, and 10 of 10 over any prior time frame. We thus present BurnRiSc as evidence that burnout is screenable from public data. 

---
# A Cross-Lingual Acoustic Disease-Alignment Framework for Respiratory Health Assessment from Spontaneous Speech 

**Authors**: Roksana Khanom, Raghib Asfak Tasnim, Bodrun Nahar Bithi, Shafia Shirin Supty, Saiful Islam Raju, Ashok Agrawala, Nirupam Roy  

**Link**: [PDF](https://arxiv.org/pdf/2609.19398)  

**Abstract**: Spontaneous speech offers a scalable, noninvasive signal for respiratory health assessment, yet interpretable models that generalize across languages remain challenging because disease-related acoustic changes are confounded by language-specific phonetic variation. We present CL-DAF, a Cross-Lingual Disease-Alignment Framework that identifies acoustic dimensions whose disease effects remain consistent across languages. Using 201 English and 75 newly collected Bangla speakers, we construct a common 272-dimensional acoustic representation and quantify disease alignment using signed rank-biserial effects and the Language Invariance Score. We first show that spontaneous Bangla speech separates COPD from controls (AUC 0.85); however, 133 features reverse their disease direction across languages and the full representation transfers poorly (AUC 0.49 from Bangla to English). CL-DAF isolates 26 disease-aligned features that raise AUCs to 0.825 and 0.722 from English to Bangla and Bangla to English, respectively. These findings provide a foundation for multilingual clinical speech models emphasizing pathology over language-dependent variation. 

---
# Riemannian--Lorentz Fusion of Vision Transformers and State-Space Models 

**Authors**: Badri N. Patro, Vijay S. Agneeswaran  

**Link**: [PDF](https://arxiv.org/pdf/2609.19384)  

**Abstract**: Scaling deep learning faces critical bottlenecks: data exhaustion, exponential training costs, and resource concentration. Model merging combines pre-trained checkpoints without gradient descent, offering orders-of-magnitude savings versus retraining. Combining independently trained vision models is difficult when their architectures and parameter shapes differ. Existing weight-space merging methods generally assume aligned, shape-compatible checkpoints, whereas a Vision Transformer (ViT) and a state-space model (SSM) implement token mixing with different operators. We study a hybrid Heterogeneous merging setting that retains both architectures while aligning parameter groups by semantic role. Our proposed Riemannian--Lorentz Parameter Fusion (RLPF) method projects aligned groups to common coordinates, lifts selected coordinates to the Lorentz hyperboloid model of hyperbolic space, computes a regularized geodesic barycenter, and decodes the result into the two branches. A learned gate then combines branch logits for each input. Component groups use fixed curvature values, with normalization parameters treated as Euclidean. In the results available in this manuscript, the fine-tuned system obtains 82.37\% on CIFAR-10, 75.04\% on Oxford-IIIT Pet, and 78.58\% top-1 accuracy on ImageNet-1K; the corresponding best-parent accuracies are 76.54\%, 71.42\%, and 76.42\%. On ImageNet-1K, the reported pre-fine-tuning initialization reaches 77.80\%. These results support further study of geometry-aware heterogeneous fusion, but not a training-free single-checkpoint merge: RLPF is a two-branch hybrid whose gate and reported final models are trained. 

---
# AUDITPLAN: Commit, Then Answer for Auditable Safety Alignment 

**Authors**: Sai Sri Pushpa Jampani, Kshitij Mishra, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2609.19325)  

**Abstract**: Safety tuning pipelines judge only the final answer, which makes it difficult to distinguish robust refusal from two undesirable shortcuts: blanket refusal on benign requests and polished but unfaithful safety rationales that do not actually constrain the answer. We propose AUDITPLAN, a single-model plan-then-answer approach where the model first emits a compact structured safety plan and then answers conditioned on it. The plan records a threat label, intended action, and explicit constraints, enabling machine-checkable auditing while remaining hidden from users at deployment. We train this behavior with supervised fine-tuning followed by reinforcement learning with FAITHGATE, a reward-gating objective that grants answer reward only when the safety plan is correct. This discourages safe-looking but unfaithful behavior and promotes tighter plan-answer coupling. Across Qwen backbones, AUDITPLAN improves both robustness and auditability: on Qwen2.5-3B-Instruct, FAITHGATE reduces ASR from 24.0% to 11.6%, LSR from 1.0% to 0.36%, and over-refusal from 11.0% to 2.0%, outperforming answer-only RL, free-form explanation, and weighted-sum structured rewards. Similar trends hold for Qwen2.5-1.5B-Instruct. Larger-model confirmation runs on Qwen-3-4B-Instruct and Qwen2.5-7B-Instruct preserve the same trend suggesting that explicit internal commitments can make safety alignment more faithful, robust, and auditable. 

---
# CovR: Coverage-Aware Hardware Verification via Reasoning-Guided Reinforcement Learning 

**Authors**: Manar Abdelatty, Maryam Nouh, Sherief Reda  

**Link**: [PDF](https://arxiv.org/pdf/2609.19189)  

**Abstract**: Design verification remains one of the most resource-intensive stages of hardware development, often consuming up to 70% of the total design effort. While recent work has explored using Large Language Models (LLMs) to automate testbench generation, most existing approaches focus narrowly on functional correctness, overlooking the critical aspect of coverage quality. To bridge this gap, we present CovR, an agentic framework for automated testbench generation that combines self-reflection loops with simulation-based feedback to maximize coverage. Using this pipeline, we construct a large-scale dataset of 16,514 natural specification RTL reasoning testbench tuples with a strong teacher model, enabling coverage-aware supervision. Building on this, we propose a reinforcement learning (RL) framework tailored for coverage-driven testbench generation, leveraging tool-derived rewards from simulation and coverage feedback to optimize a student model. Experimental results show that the CovR finetuned model achieves 93.81% cov@10 on VerilogEval and RTLLM V2.0, and 87.76% cov@10 on CVDP, outperforming state-of-the-art approaches by 7.97% and 3.59%, respectively. Furthermore, deploying the finetuned model back into the agentic refinement pipeline further improves cov@10 to 94.27% on VerilogEval and RTLLM V2.0 and 91.39% on CVDP. Moreover, when integrated as a plug-in stimulus engine for full verification workflows, CovR improves coverage by 18.95% and mutation detection score by 1.19%, while revealing 4.46% undetected failures, highlighting the importance of optimizing for coverage in LLM-based hardware verification. 

---
# Message capacity and claim wording set the transition points of collective truth-finding in language-model networks 

**Authors**: Makoto Fukushima  

**Link**: [PDF](https://arxiv.org/pdf/2609.19183)  

**Abstract**: Whether human or large language model (LLM), an agent in a discussion reads only a few of the others' contributions, bounded by cognition, context, or cost. LLM collectives can settle on a wrong consensus even when a majority starts out correct; we ask how far that reading bound alone decides the outcome. We model the bound with one number, the message capacity, which sets how many of the others' messages an agent reads, and generate the communication network from it. Over 31,824 randomized queries, we found that an 8-billion-parameter model's judgment of a claim effectively reduces to a logistic function of a weighted sum of its inbox, the update rule of a stochastic binary neuron with divisively normalized weights. From these weights and the network's degree statistics alone, the wrong consensus should become unreachable from any start once agents read, on average, fewer than 6.4 of their 31 sources. In 1,414 episodes with assigned starts the prediction failed: the correct side won in fewer than 50% of episodes from every start, and in only 28-45% when 75% of agents started correct. The failure traces to the field, the threshold that a claim's wording sets for the agent's answer before any message is read: the experimental claims' fields lay below the calibration mean, and with each claim's own field the same weights reproduce the outcomes. Reversing the wording showed that the threshold follows what a claim asserts, not whether it is true. On a second 8B model the pipeline predicts claim-dependent bistability; transition points appeared where computed, and an eight-claim calibration matched in 15 of 16 conditions. At 70B the assertion bias is not detected. Thus a collective's fate is largely set by two single-agent measurements: the threshold a claim's wording sets, and the message capacity that sets the transition point. 

---
# What Do We Expect from LLMs? Mapping the Design of LLM Benchmarks 

**Authors**: Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19182)  

**Abstract**: Benchmarks are central to how progress in large language models (LLMs) is assessed and communicated. Yet model rankings alone reveal little about how evaluation requirements themselves are changing. The expanding variety of benchmarks offers another perspective: what researchers expect LLMs to do, and what they count as successful performance. We systematically map 14,767 papers introducing or updating evaluation resources from arXiv submissions between January 2022 and August 2026. Using staged screening and automated full-text coding, we examine changes in target systems and domains, evaluation materials and conditions, and scoring mechanisms. The collection shows growing emphasis on action, interaction, and professional applications, while established and newer design elements frequently coexist. Model participation also develops unevenly: LLM-based scoring grows within both agent and non-agent groups, whereas model-generated materials show no comparable sustained increase in recent cohorts. These findings illuminate how public research translates capability expectations into concrete tests and criteria for success. As AI participates in constructing tests, performing tasks, and judging responses, they also raise a question: does expanding evaluation provide more independent evidence, or risk reproducing the preferences and blind spots of its participating models? 

---
