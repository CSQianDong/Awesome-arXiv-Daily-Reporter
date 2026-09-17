# Objective vs. Search: Decomposing What Makes a Good Tokeniser 

**Authors**: Ahmetcan Yavuz, Clara Meister, Tiago Pimentel  

**Link**: [PDF](https://arxiv.org/pdf/2609.19145)  

**Abstract**: Two dominant tokenisation algorithms are used by modern language models: byte-pair encoding (BPE) and UnigramLM. These differ along two orthogonal axes: their optimisation objective (compression vs. log-likelihood) and their search procedure (bottom-up merging vs. top-down pruning). Existing comparisons confound these axes, making it unclear whether their observed differences stem from what is being optimised vs. how it is being optimised. We disentangle the two by introducing two new tokenisation algorithms that complete this 2x2 design space: BottomUpLL, a bottom-up likelihood-based tokeniser, and TopDownComp, a top-down compression-based tokeniser. We train language models with tokenisers produced by each algorithm, varying: model size, vocabulary sizes, and domain (English-only vs. multilingual). Evaluating models on bits-per-byte, we find that the search procedure -- not the objective -- is the dominant factor: bottom-up tokenisers consistently achieve lower bits-per-byte in most settings. Evaluating models on the BLiMP task, however, shows no consistent relationship between design choice and performance. Overall, our results disentangle the effect of tokeniser design choices on language modelling performance, offering concrete guidance for their more principled construction. 

---
# A Zeroth-Order Paradigm for LLM Preference Alignment 

**Authors**: Peter Chen, Xi Chen, Wotao Yin, Tianyi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.19144)  

**Abstract**: Direct preference alignment methods are widely used to align large language models (LLMs) with human preferences because of their computational and memory efficiency. However, likelihood displacement motivates alternative ways to extract information from preference pairs with small likelihood margins. In this paper, we propose and analyze Comparison-based Preference Optimization (ComPO), a zeroth-order alignment method based on comparison oracles. ComPO extracts directional information from these pairs without directly optimizing a differentiable preference loss on them. We establish a convergence guarantee for its basic offline scheme under smoothness, gradient sparsity, and compatibility between the oracle and a latent objective. We further introduce online ComPO, which retains the offline comparison mechanism and uses unlabeled policy generations for reverse-KL control relative to a reference policy. Following the coverage perspective of preference fine-tuning, we establish a performance guarantee for a basic constrained scheme under local coverage and in-distribution pairwise reward accuracy. Experiments on Mistral, Llama, Gemma-2, Qwen3, and Gemma-3 models demonstrate improvements over existing direct alignment methods, including length-controlled win rates, with pair-level diagnostics providing evidence consistent with mitigating likelihood displacement. 

---
# ScienceIDE: Turning World's Scientific Codebase into Agent Learnable Environments 

**Authors**: Hejia Geng, Zesen Huang, Haoyang Li, Wenbin Li, Koutian Wu, Zihan Zhou, Yuanbo Pang, Weihao Liu, Zigong Xu, Zhiping Li, Zongzheng Zhang, Chuanfei Dong, Jiankai Sun, Tianzhe Zheng, Fengyu Xie, Yue Ma, Yueheng Shi, Tong Xie, Zonglin Di, Xianrong Liu, Qucheng Gao, Yimin Liu, Jiaming Pan, Sheng Huang, Xiao-Han Ma, Lanqing Yuan, Zhenlin Zhu, Ziang Liu, Ziyang Xu, Junkai Wang, Kangkai Liang, Jiayi Xian, Zehong Zhao, Liuwei Xu, Jingxu Xie, Peijin Zhang, Qiang Gao, Chengyi Xing, Zhe Zhao, Xi Wang, Yaopeng Xing, Xing Meng, Zhenfei Yin, Yingcheng Wu, Ling Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19134)  

**Abstract**: Scientific code repositories encode decades of human knowledge in executable models, methods, and tools. Yet fragmented toolchains, implicit domain conventions, and specialized correctness criteria make this knowledge difficult to convert into reliable learning experience-a challenge we call the scientific experience bottleneck. We introduce ScienceIDE, infrastructure for turning the world's scientific code into programmable environments for scientific agents. Guided by expert-defined scientific cases and acceptance criteria, agents transform repositories into executable environments that support task generation, execution, and scientific verification. These environments provide a shared foundation for supervised fine-tuning, reinforcement learning, and evaluation. Using verified interaction trajectories, we train PhAI-IDE-72B, PhAI-IDE-9B, and PhAI-IDE-4B. The model family shows gains in held-out scientific-code repair and across selected general-purpose benchmarks in code, reasoning, and knowledge, providing evidence of positive transfer from scientific experience to broader capabilities. ScienceIDE lays the foundation for an integrated workspace for agent learning and scientific practice, making humanity's scientific software a shared substrate for developing scientific intelligence. Code: this https URL 

---
# Playing log(N)-Questions over Wikipedia Abstracts: Communication Efficiency Between Paired Frontier Models 

**Authors**: Peter Potash  

**Link**: [PDF](https://arxiv.org/pdf/2609.19113)  

**Abstract**: We evaluate six frontier language models on the two-agent $\log(N)$-Questions game. A questioner sees $N$ Wikipedia lead paragraphs and must identify a secretly chosen target using exactly $\log_2 N$ yes/no questions. An answerer sees only the target and the question, and replies with one word. Both roles run on the same provider, so the game measures how well a model communicates with itself across an information asymmetry. We run 408 games over document sets of 4 to 1024 paragraphs at a total API cost of \$363. One model finishes well behind the others: Claude Opus 5 wins 28 of 68 games, against 45 to 56 for GLM-5.3, GPT-5.6 Sol, Grok 4.6, Gemini 3.8 Flash and Kimi K3. The leading five are only marginally separable. Pooling those five, win rate declines with set size at $r=-0.973$ and is fit by a single per-round reliability parameter. The form is $\text{win}=p^{\log_2 N}$ with $p=0.928$. Losses divide into answer errors and discrimination failures in roughly equal measure, and models almost never name a document their own evidence excludes. Every unanimous answer error from the weakest model was inspected: 32 of 34 are ``No'' answers, on properties stated in the document's first sentence, under an instruction that explicitly warns against defaulting to ``No''. Information per question, estimated from answer balance, correlates with win rate at $r=+0.88$. The only two models to extract a full bit per question are the only two that partition on document titles, a strategy absent below $N{=}32$ and used in a quarter of questions above it. Reasoning-token expenditure varies $4.5\times$ across models with little relation to success, and the trace grows as the candidate set shrinks without a matching gain in reliability. 

---
# Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations 

**Authors**: Leon Bergen, Usha Bhalla, Andrew Lee, Barak Widawsky, Linas Nasvytis, Connor Watts, Siddharth Boppana, Sidharth Baskaran, Dron Hazra, Michael Byun, Atticus Geiger, Owen Lewis, Matthew Kowal, Vasudev Shyam, Thomas Fel, Thomas McGrath, Ekdeep Singh Lubana, Jack Merullo  

**Link**: [PDF](https://arxiv.org/pdf/2609.19101)  

**Abstract**: As models scale, reward hacking becomes more frequent, more sophisticated, and more consequential. Does it leave a telltale signature in model representations? This work analyzes how reward hacking is represented internally in frontier open source LLMs, and how those representations can be used to understand and discover the range of hacking behaviors a model displays. In particular, we find that simple difference of means vectors coherently represent reward hacking in Kimi K3, GLM 5.2, and Qwen 3.8 Max across a variety of behaviors in common evaluations. Despite their simplicity, these vectors are both generalizable and interpretable, and we can use them to reliably detect reward hacking. We first evaluate reward hacking in commonly reported benchmarks like DeepSWE and SWE-bench, finding that models reward hack excessively in these environments; GLM 5.2 hacks in 57.2% of rollouts on DeepSWE and in 73% of rollouts on SWE-bench. Catching these requires monitors; LLM monitors are effective, but expensive detectors. We show that DoM vectors are similarly effective but virtually free, catching 3.1% more hacks in Kimi K3 and 7.9% fewer hacks in GLM 5.2 on DeepSWE at a monitor matched false positive rate. DoM vectors run on the chain-of-thought also predict reward hacks in the model's subsequent actions, meaning we can run them online and catch potential hacks before they occur. Finally, we analyze probe-hits that LLM monitors do not catch and discover other undesirable behaviors, as well as show transfer to finding hacks in non-SWE evaluations. Together, these results provide evidence that simple, white-box methods can be used to scalably study and monitor reward hacking behaviors in frontier open source models 

---
# Reporting Practice Matters: The Impact of Reference Choice on Chest X-ray Report Evaluation 

**Authors**: Daniel P. Jeong, Charles Q. Li, Hossein Hosseiny, Nitya M. Bhalla, Fatma Uyar Morency, Pradeep Ravikumar, Zachary C. Lipton, Michael Oberst  

**Link**: [PDF](https://arxiv.org/pdf/2609.19093)  

**Abstract**: Radiologists follow heterogeneous reporting practices. Two radiologists examining the same image and identifying the same clinical findings might nevertheless compose superficially distinct reports, varying in terminology, shorthand, formatting, and level of detail. These variations in reporting norms represent an under-appreciated obstacle in efforts to evaluate AI-based radiology report generation (RRG) models, where machine-generated reports are typically assessed based on their concordance with human-generated references. In this paper, we quantify the sensitivity of established evaluation metrics to variations in reporting practices, revealing impacts large enough to alter the rankings of models. We introduce a radiologist-informed taxonomy of variations in radiology reporting practice and a method (ReRef) that rewrites reference reports along the axes of our taxonomy while preserving clinical interpretation. For instance, when comparing the performance of nine RRG models on MIMIC-CXR using RadCliQ-v1, condensing the discussion of normal findings in the reference reports causes Libra to drop from first to second place while CheXOne rises from third to first. Our results suggest that many current metrics fail to decouple clinical interpretation from conformity to reporting practices and that choosing the ``right'' references that accurately reflect the desired reporting practices can be important in practice. To support future research, we release MIMIC-CXR-Ext-ReRef, a radiologist-validated dataset of 120 (original, alternative) reference report pairs derived from MIMIC-CXR. 

---
# Safety-Flag: A Unified Benchmark for the Reliability and Calibration of LLM Content Moderators 

**Authors**: Yibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19072)  

**Abstract**: Large language models are increasingly used for content moderation, but most evaluations still report aggregate accuracy on individual benchmarks. We introduce Safety-Flag, which places seven widely used safety benchmarks (BeaverTails, XSTest, Ethics, WildGuard, Aegis, ToxiChat, and ToxiGen) into a single balanced flag / do-not-flag protocol. We release item-level decisions and confidence scores for six general-purpose LLMs and four dedicated guards, together with three reference models, evaluated on the same items. Safety-Flag measures three dimensions of moderator reliability: error direction, probability calibration, and confidence-based error ranking for human review. They often disagree. Aggregate accuracy does not reveal error direction: one model flags $85\%$ of benign content, whereas another misses $54\%$ of harmful content. All six general-purpose models are overconfident; fitting one temperature per model reduces calibration error by $2.8$--$6.0\times$ without changing predicted labels or confidence ordering. Confidence-based abstention lowers selective risk for every model, although the gains depend on how well confidence ranks errors. Dedicated guards produce fewer false alarms and are better calibrated, but several have higher miss rates outside their documented coverage. We release the benchmark, fixed item lists, evaluation code, per-item model outputs, and leaderboard at: this https URL. 

---
# Benchmarking Large Language Models for Biomedical Relation Extraction 

**Authors**: Claudiu Creanga, Teodor Marchitan, Liviu P. Dinu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19071)  

**Abstract**: Extracting SNP-phenotype associations from biomedical literature is vital but challenging. We benchmarked diverse NLP models, including MLMs, hybrid architectures, and state-of-the-art LLMs (Gemini 2.0, OpenAI O-series, Qwen, Mistral), on the SNPPhenA corpus across three tasks: sentence-level, abstract-level, and association strength classification. OpenAI O1 achieved state-of-the-art (SOTA) results using few-shot learning for non-finetuned sentence-level classification (F1 0.89) and established a new SOTA for abstract-level classification (F1 0.82). Association strength classification proved difficult, though fine-tuned Gemini 2.0 Pro performed best (F1 0.60) in the first LLM evaluation of this task. Proprietary LLMs, especially in few-shot (O1) or fine-tuned (Gemini 2.0 Pro) settings, significantly outperformed other models. These findings confirm the power of modern LLMs for genomic knowledge extraction. 

---
# Reading Between the Lines: Can LLMs Discover the Question Behind the Text? 

**Authors**: Claudiu Creanga, Liviu P. Dinu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19070)  

**Abstract**: This paper introduces ``question archaeology'', a specific evaluation task focused on inferring the single, authentic "genesis question" that motivated the creation of a complete text. Distinct from question generation, which targets any plausible question, or discourse frameworks that model utterance-level acts, our task assesses a model's grasp of authorial intent. We present a new dataset of commissioned texts paired with their original research questions and plausible distractors. Our evaluation of both proprietary models, like Gemini Flash and Pro, as well as open source models like Mistral and Qwen, reveals significant progress in this task, with the newer versions outperforming the earlier ones, while BERT-based models performed poorly. Notably, our findings indicate that current LLMs surpass human performance on this task, suggesting advanced understanding of authorial intent. This capability has important implications for AI's role in tasks requiring nuanced interpretation of human communication. Our work thus provides a new framework and a challenging benchmark for future models. 

---
# MIRAGE: How Conversation State Shapes Historical Evidence Use in Multimodal Personal Agents 

**Authors**: Yu Liu, Wenxiao Zhang, Cheng Hu, Cong Cao, Fangfang Yuan, Xinyu Wang, Jin B. Hong, Yanbing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19059)  

**Abstract**: Multimodal large language model (MLLM) agents are increasingly used as personal assistants for long-running tasks. Their utility depends on continuity: agents must retrieve and use earlier evidence across dialogue, files, and workspace state. However, agents can generate plausible answers even when access to that history has degraded, causing outcome-only evaluation to overestimate true evidence use. We present MIRAGE (Multimodal Interaction Retrieval, Attribution, and Grounding Evaluation), a controlled study of historical evidence use under conversation-state variation in multimodal personal agents. MIRAGE holds evidence objects, questions, and scoring fixed while varying only conversation state, and evaluates whether an agent can determine answerability, recover the correct source, and answer from it. Across seven frontier and open-weight multimodal backbones, we find that: 1) pre-compaction depth and post-compaction continuation form distinct, non-monotonic failure regimes rather than a single degradation curve; 2) open-weight models rely heavily on context continuity and are reluctant to spontaneously switch to tool-mediated retrieval when provenance fails; and 3) retrieval pressure improves source attribution in deep pre-compaction states for tool-compliant models, but consistently regresses after compaction, where stored evidence has already degraded. These findings show that historical evidence use should be evaluated under state variation, rather than inferred from outcome-only correctness. 

---
# Entropy in Conversational AI: Structured Unpredictability as Inferrable Interiority 

**Authors**: Sebastian Cochinescu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19044)  

**Abstract**: Sampling can increase response diversity without producing history-dependent behavior. We formalize a different design target, structured unpredictability, as conditional dependence between an output and a persistent hidden state beyond what an observer can infer from the transcript. A selection layer updates a low-dimensional style-and-attention state from a capacity-limited stream, generates several responses with a fixed base model, and selects for novelty and state affinity. Evaluation uses scripted sequences of independent prompt turns: the base model receives the current turn and rendered state, but not the preceding dialogue; cross-turn dependence resides in the wrapper state and response selector. A synthetic implementation validates the pipeline and matches four prospectively hash-frozen divergence features at point level. In the final real-model grid (mlx-community/Qwen2.5-1.5B-Instruct-4bit; 56 sequences per arm), the mechanism increased lexical novelty over the low-variance and consistency-only controls by 0.073 and 0.023, respectively. Its stylometric-consistency contrast with novelty-matched sampling was equivalent to zero under the registered smallest-effect rule, so the joint novelty-consistency criterion failed. The original two-part accumulation criterion also failed; a revised final-grid contrast, frozen after the powered grid, found higher consistency than the memory-reset ablation (0.028, 95% CI [0.018,0.039]), but does not establish path dependence. Twin separation was not established (0.003, 95% CI [-0.011,0.019]); the mean curve's saturating curvature matched the frozen prediction, which without separation does not support path dependence. Probe-level capability equivalence held within +/-0.10 on a near-ceiling battery, while output quality was not evaluated. All outcomes are machine-scored; no claims about perceived mind or consciousness are tested. 

---
# TalkMatrix: Generating Character Dialogue that is Both Consistent and Diverse 

**Authors**: Ayuto Tsutsumi, Yuu Jinnai  

**Link**: [PDF](https://arxiv.org/pdf/2609.19022)  

**Abstract**: Candidate-based decoding typically selects a completion for each prompt independently, but many applications require a collection of outputs that satisfies global, non-decomposable requirements. We formulate this setting as structured multi-prompt, multi-completion selection: given a candidate pool for every prompt, select one completion per prompt to optimize a collection-level objective. We instantiate the problem in character dialogue, where each character should remain consistent across situations, each line should fit its situation, and characters and situations should remain distinguishable. Our method, TalkMatrix, generates multiple candidates for every character--situation pair and jointly selects a complete matrix using four embedding-based consistency and diversity objectives. Because a weighted sum can improve some dimensions by sacrificing another, TalkMatrix maximizes the worst-performing objective through a two-level minimax formulation. We approximately optimize the resulting discrete objective with multi-start coordinate ascent, and compare it with local, partial-matrix, and generic combinatorial search baselines. We run experiments on $50$ synthetic role-playing scenarios and $25$ curated board game scenarios where multiple characters interact in predefined situations. An LLM-as-a-judge rates matrix-level selection higher than random and independent cell-level selection baselines. These results show the value of structured selection for globally controlled dialogue generation, while our empirical validation remains specific to role-playing scenarios. 

---
# WordPolo: Evaluating Language Models Through Iterative Semantic Feedback 

**Authors**: Tyler McDonald, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2609.19006)  

**Abstract**: Large Language Models (LLMs) and Large Reasoning Models (LRMs) are typically evaluated on challenging benchmarks through dataset accuracy alone, providing no insight into the quality or faithfulness of their reasoning processes. We present WordPolo, a word-finding task where participants must discover an unknown target word using semantic similarity feedback. Players start with zero knowledge, make guesses, and receive distance scores (1 = correct, higher = further away). Success requires interpreting scores to navigate semantic space and systematically narrow the search. This design makes iterative reasoning and adaptive search strategies both directly observable and necessary for success. We evaluate recent LLMs (GPT-4.1, Llama 4, Claude 3.5 Haiku, Qwen 3), LRMs (o4-mini, Deepseek-R1), humans, and a novel heuristic on 1,500 puzzles. Beyond solve rates (which range from 4% to 62%), we introduce progression-based metrics that reveal models often make meaningful progress, insights that accuracy alone would miss. Our analysis shows how reasoning models can be hindered by overthinking and underthinking, while successful models exhibit human-like strategies. WordPolo demonstrates the need for benchmarks that test both reasoning process and outcomes, providing holistic measurements of model capabilities. Our code and dataset can be found at this https URL. 

---
# A Benchmark Suite and Ground-Truth Methodology for Formal Verification of IEC 61131-3 Ladder Diagram Programs 

**Authors**: Pierre Dantas, Lucas Cordeiro, Waldir Junior  

**Link**: [PDF](https://arxiv.org/pdf/2609.18994)  

**Abstract**: We present the first benchmark suite for formal verification of Programmable Logic Controller (PLC) programs that combines controlled ground truth with coverage of both textual (Structured Text, ST) and graphical (Ladder Diagram, LD) IEC 61131-3 encodings. Despite growing support for tools, the field lacks standard evaluation benchmarks: existing corpora omit formal properties or graphical dialects, and private program sets preclude reproducible measurement of progress. Our suite comprises 50 programs in 83 variants across ten industrial domains, provided in PLCopen Extensible Markup Language (XML) and ST, each paired with a formal property, machine-checkable expected verdict, and violation witness in the Software Verification Competition (SV-COMP) format. The central methodological contribution is a tripartite ground-truth discipline - verdicts are established by construction, fault injection, or audited cross-tool consensus - motivated by a concrete failure mode where the obvious safety property misclassifies all attacks from two public logic-bomb corpora as safe due to invisible non-termination. Reference verdicts are obtained with the Efficient SMT-Based Context-Bounded Model Checker (ESBMC) v8.4 from source: all 25 graphical benchmarks execute, and 43 of 45 accepted variants match recorded verdicts. On the finite-state fragment (21 benchmarks), nuXmv - a model checker with unrelated decision procedures - agrees on all 24 interlock variants and resolves two benchmarks ESBMC-PLC leaves unknown, confirming tool-neutral ground truth and discriminative power. Porting exposes format and semantics fragmentation: front-ends accept different serializations, and timer semantics vary across tools - phenomena the suite is designed to reveal. The corpus, schema, validator, and recheck harness are released as open artifacts. 

---
# MechSparse: Mechanism-Guided Sparse PEFT Selection Is Task-Shaped 

**Authors**: Son Ha Xuan, Phat T. Tran-Truong, Xuan-Bach Le  

**Link**: [PDF](https://arxiv.org/pdf/2609.18961)  

**Abstract**: Mechanistic interpretability identifies sparse subsets of heads and MLP blocks that carry specific behaviors. We ask whether such causal signals can guide where to place a small PEFT budget more effectively than the cheap heuristics practitioners already use. \method{} scores attention heads and MLP blocks by normalized activation-patching recovery on clean/corrupted probes and trains LoRA/QLoRA only on the selected sites; \methodc{} adds bounded credit for small within-layer joint subsets.
We compare against random, magnitude, activation-norm, and gradient/Fisher on Ministral-8B/NF4 in three cells: Swahili span-JSON information extraction (IE) at $b{=}0.25\%$ and $1.0\%$, and English$\to$Swahili machine translation (MT) at $b{=}1.0\%$. The causal selectors never win the primary metric. On the headline IE cell (3 seeds, paired-bootstrap CIs over $600$ predictions), \methodc{} beats random by $+0.079$ span+type F1 and gradient/Fisher by $+0.174$, but trails activation-norm by $0.028$, with the smallest cross-seed std ($\pm 0.003$). On MT all four selectors lie within $0.30$ BLEU and every paired CI includes zero. A schema-versus-span decomposition explains the IE gap: activation-norm captures the rigid JSON routine, while causal scores track content-sensitive sites. We distill a preliminary diagnostic -- prefer activation-norm when output structure dominates, treat causal selectors as a hypothesis for content-dominated tasks -- and release masks, scores, predictions, and evaluation files for direct replay. 

---
# When Audit Quality Fails to Predict Downstream Utility: A Counterfactual Study of Synthetic-Data Selectors for Low-Resource African NLP 

**Authors**: Son Ha Xuan, Phat T. Tran-Truong, Xuan-Bach Le  

**Link**: [PDF](https://arxiv.org/pdf/2609.18960)  

**Abstract**: Quality-aware synthetic-data selection rests on a proxy: examples that an LLM judge rates as good should also help a downstream model learn. In a controlled replay in low-resource African-language classification, we show that this proxy breaks. Across four languages (Amharic, Hausa, Swahili, Yoruba), two classification tasks (MasakhaNEWS, AfriSenti), and five matched-budget selectors, audit rankings and downstream rankings diverge. Within each cell, the Spearman between judged label correctness and Macro-F1 across selectors has mean $\rho{=}0.04$ (median $0.00$), showing that the mismatch is not an aggregation artifact. \method{}-V2, our counterfactual audit framework, produces the cleanest selected pool on three audit channels at once: highest judged label correctness ($0.904$ vs.\ $0.767$ for naive, a $17.9\%$ relative gain), lowest shortcut score, and a hard-reject rate of $0.162$ vs.\ $0.486$ for naive. AlpaGasus nevertheless leads downstream Macro-F1 ($0.202$ vs.\ $0.163$ for \method{}-V2), and the inversion persists on the five non-degenerate cells. The lesson is methodological: in this controlled setting, audit quality is a property of the selected pool, not a guarantee of downstream utility. Synthetic-data evaluation should therefore report audit and downstream metrics on the same retained sets. We release the audit tables, per-selector retained pools, and a claim ledger that links every reported number to its source row. 

---
# LangSelect: Cost-Aware Target-Language Routing for LLM Code Generation 

**Authors**: Son Ha Xuan, Phat T. Tran-Truong, Xuan-Bach Le, Nghia Duong-Trung  

**Link**: [PDF](https://arxiv.org/pdf/2609.18959)  

**Abstract**: LLM code-generation systems usually choose a target programming language before decoding and treat that choice as fixed. We show that, for language-flexible programming tasks -- tasks where several target languages are acceptable and checkable by the same tests -- this choice is a measurable cost lever: verified implementations of the same task can differ substantially in generated-token length. We introduce LangSelect, a verification-aware router that selects the target language before generation and falls back when the first attempt fails. To separate offline routing opportunity from end-to-end behavior, we evaluate verified-solution replay, which chooses among already accepted corpus solutions, and live GPT-5 generation, which charges every generation attempt, including failures and fallbacks. On MultiLang-Bench, a 3,000-task, 8-language verified corpus, replay shows substantial language-routing headroom. In live evaluation on 450 held-out tasks, a train-split Domain heuristic baseline reduces harness-proxy tokens, which include wrapper and entrypoint overhead, by 50.3\% at 92.9\% pass after fallback, while a learned CodeBERT+metadata selector reaches the highest pass after fallback, 93.8\%, with a 3.7\% token increase. These results show that output-language routing can define a practical cost-correctness frontier for unit-test-verifiable code generation. 

---
# Long-Lived Characters, Local Inference: Incremental Memory Maintenance for Game NPCs 

**Authors**: Zimu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18935)  

**Abstract**: A game character should not have to reread its entire life before every conversation. For locally deployed language-model characters, however, revising a few memories can invalidate a long reusable prefix. The resulting preparation cost competes with both foreground dialogue and the maintenance of other characters. This matters especially when dialogue feeds game-defined actions and value judgments: a fluent but incorrect account of who owns an item, or whether a transfer has already happened, can corrupt the input to otherwise deterministic rules. We study incremental memory maintenance for long-lived game NPCs in a quantized Qwen hybrid recurrent-attention model. Our runtime removes superseded attention KV entries, computes replacement records at the true sequence tail, and preserves the continuing recurrent state and unchanged KV. Existing local experiments combine multi-update dialogue replays, fixed-input placement ablations, and attention diagnostics. Independent block composition weakens query-conditioned memory selection without a uniform chunk-initial attention collapse. True-tail updates preserve important current-state and historical bindings across eight scripted maintenance rounds; a placement case recovers the full-refill quantity in three reconstructions, while slot-preserving alternatives repeat a double-subtraction error. Attention-distribution proximity alone does not explain these semantic differences. The results motivate treating a character's inference state as a maintained, history-dependent resource, rather than only a disposable encoding of its latest memory text. 

---
# Beyond Outcomes: Dual-View Relational Learning for Efficient Agent Benchmarking 

**Authors**: Xinshuai Guo, Junjie Wu, Dolly Deng, Yinghui Li, Hai-Tao Zheng, Suncong Zheng, Maxm Pan  

**Link**: [PDF](https://arxiv.org/pdf/2609.18909)  

**Abstract**: Agent benchmarks are substantially more costly to evaluate than conventional LLM benchmarks. Benchmark compression is therefore a natural solution, yet existing methods primarily model redundancy in task--model final-score distributions, which is important in agentic evaluation. To address this limitation, we analyze large-scale trajectories and identify six complementary process signals that are systematically associated with final agent performance. To disentangle agent performance redundancy from a complete perspective, we propose DualViewEval, an agent benchmark compression method that jointly exploits outcome and process relations to learn an exact-size miniset and predict the full-benchmark scores. Across five agent benchmarks and five representative baselines, DualViewEval achieves the best results in all datasets. With only 20 tasks, it achieves $24\times$--$40\times$ compression on APEX-Agents and BFCL, reducing mean absolute error (MAE) by $14.5\%$--$28.2\%$ over the strongest competitors while improving Kendall's $\tau$ by up to $7.2\%$ relative to EssenceBench on SWE-bench Verified. The selected minisets further reveal capability differences among different agents, providing compact and diagnostic feedback for efficient agentic model development. 

---
# How Much is a Human Right Worth? ECtHR-NPD: A Benchmark for Predicting Non-Pecuniary Damage Awards 

**Authors**: Yanyi Pu, Damian A. Gonzalez-Salzberg, Zheng Yuan, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2609.18908)  

**Abstract**: Existing legal benchmarks cover diverse tasks, while continuous monetary remedies remain comparatively underexplored. We introduce ECtHR-NPD, to the best of our knowledge, the first benchmark for predicting non-pecuniary damage (NPD) awards at the European Court of Human Rights (ECtHR) from case information when no statutory formula or explicit calculation rule determines the amount. ECtHR-NPD contains 14,575 cases with case-level awards in nominal euros, chronological splits, and a protocol separating target construction from model input. We evaluate a battery of methods, including constant predictors, gradient-boosted trees, retrieval methods, fine-tuned encoder language models (LMs), prompted decoder LMs, and knowledge-augmented agents. Our results show that more sophisticated LM and agentic approaches do not consistently outperform the strongest feature-based baseline. All model families struggle to identify zero awards and to calibrate high-award predictions, with further degradation on the Challenging test view, making ECtHR-NPD a challenging testbed for current state-of-the-art open-weight and proprietary LMs. 

---
# Structured Claim-Level Discourse Representations for Dense Health Narratives 

**Authors**: Farnoushsadat Nilizadeh, Elham Pourabbas Vafa, Shirin Nilizadeh, Eduard Dragut  

**Link**: [PDF](https://arxiv.org/pdf/2609.18905)  

**Abstract**: Health discourse in social media videos often contains densely entangled claims spanning multiple thematic aspects, stances, evidential frames, and rhetorical functions within short conversational spans. Existing approaches largely rely on coarse topic-level, sentiment-based, or stance-oriented representations that do not adequately capture this structure. Our analysis identifies an average of 13.22 atomic claims per minute, motivating richer claim-level discourse representations. We introduce a structured framework for claim-level discourse analysis in dense health narratives. Our framework models discourse through tuples linking atomic claims with thematic aspects, stance, and multidimensional pragmatic discourse attributes. To support this setting, we construct a benchmark spanning four health domains with 1,191 manually annotated claims from 60 videos. Using this framework, we evaluate automated structured discourse analysis under different discourse context settings. Results show that current LLMs achieve strong performance on thematic categorization and stance prediction, but struggle with high-dimensional pragmatic profiling. We also find that different discourse tasks benefit from different forms of contextual reasoning, suggesting that future systems may require task decomposition and specialized inference strategies. 

---
# PersonaPath: Towards Knowledge-Centric Personalized Learning Path Planning 

**Authors**: Yu Liu, Zeming Liu, Tianle Zhang, Zihao Cheng, Yuhang Guo, Kehai Chen, Min Zhang, Yunhong Wang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18861)  

**Abstract**: Adaptive learning systems commonly formulate learning path planning as Exercise-Centric (EC) recommendation, where the next step is inferred from item-level interaction logs. Evaluating goal-oriented guidance additionally requires explicit learner goals and curriculum-scale prerequisites: learners with similar exercise records may need different paths toward their targets. We therefore study Knowledge-Centric (KC) personalized learning path planning, where a planner must reason over learner profiles, mastery states, and prerequisite knowledge structures to decide which textbook, unit, and concept should be studied next. To support this setting, we introduce PersonaPath, a benchmark that pairs 2,000 fine-grained learner personas with a hierarchical knowledge graph of 347 textbooks, 1,751 units, and 4,092 concepts across 77 subjects. We evaluate representative LLMs on PersonaPath. Results show that even the strongest LLM reaches only a 29.5% final pass rate in Basic Education, and that the main bottleneck lies in adaptivity, where no model exceeds 44.7% in tailoring paths to individual learners. 

---
# EviGen: Predictive Evidence Scaffolding for Verifiable Clinical Rationale Generation 

**Authors**: Fengnan Li, Heman Burre, Liwen Sun, Roshni Varma, Matthew M. Engelhard  

**Link**: [PDF](https://arxiv.org/pdf/2609.18852)  

**Abstract**: Longitudinal electronic health records (EHRs) capture years of patient history across notes, codes, labs, and procedures, and contain evidence needed to reason about likely clinical outcomes. However, comprehensive clinician review of these records is impractical, and LLM-based processing is costly and often unreliable, missing some relevant observations while hallucinating others. We therefore propose EviGen, a three-layer framework for verifiable clinical rationale generation that addresses these challenges. The first layer is a patient-conditioned retriever that uses learnable queries to find evidence predictive of, not just textually relevant to, a clinical outcome and ranks it by prediction attribution scores. The second layer is an LLM generator that consumes this ranked evidence as a scaffold to produce a clinical rationale grounded in the retrieved spans. The third layer is a process-supervised verifier that checks the generated rationale at the reasoning-step level, flagging unreliable claims. Across three medical prediction datasets, EviGen improves prediction performance and rationale faithfulness over full-context LLM and RAG baselines, and is preferred by clinical reviewers in a usability evaluation. 

---
# ReFigBench: Benchmarking Scientific Figure Reconstruction as Editable PowerPoint Artifacts 

**Authors**: Liyang Fan, Chi Wei, Yitai Li, Xinping Bi, Guhong Chen, Chenghao Sun, Haoxiang Yang, Qingwen Li, Kai Yan, Hong Li, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.18844)  

**Abstract**: Multimodal coding agents are expected to turn visual inputs into usable artifacts, and they act through a harness, the layer of tools, context management, and execution environment around the model. Existing evaluations often isolate short tool calls, API traces, or screenshot resemblance, and a low score under these proxies cannot say whether the model saw poorly, planned poorly, or was failed by its harness. We study scientific overview figure reconstruction, an agent task in which a source image must become an editable PowerPoint slide that preserves text, topology, layout, and native document structure. We introduce ReFigBench, a benchmark and evaluation framework built on 1,000 real overview figures retrieved from arXiv papers with full provenance. Coding agents from four model families reconstruct every figure under two workflows, direct code generation and a specialized PPTX workflow, and the strongest model runs inside two commercial harnesses, yielding ten configurations. Evaluation combines deterministic artifact checks, repeated automated scoring by judges from two model families, and blinded human comparisons. Perception remains a bottleneck that iterative rendering only partly repays. Whether workflow effort converts into quality depends on the model together with its harness, since the same model gains from the specialized workflow inside one harness and loses inside the other, and the harness shifts scores even under an identical direct prompt. The specialized workflow erases native connectors in every configuration, human judges still prefer its renderings in most matchups, and even the strongest agent falls short of the rubric ceiling. These results expose the tension between fidelity and editability as the central challenge for practical multimodal document agents. 

---
# Beyond frequency measures: Can contextual embeddings capture meaning change in scientific texts? 

**Authors**: Jianying Liu, Kim Gerdes, Jean-Marc Deltorn  

**Link**: [PDF](https://arxiv.org/pdf/2609.18804)  

**Abstract**: Identifying technological trends is a core scientometric task, yet traditional frequency-based approaches struggle to capture substantial meaning shifts of domain-specific terms. We hypothesise that contextual embeddings can complement frequency dynamics to effectively track diachronic semantic change. We compare frequency and embedding-based approaches across Astrophysics and NLP corpora spanning from 2010 to 2024. Candidate terms are extracted using KeyBERT (utilizing SciBERT as its underlying language model) and filtered for significant frequency increases using Fisher's exact test. These terms are then evaluated for genuine semantic shift by domain experts to establish ground-truth labels. To quantify semantic drift, each term's contextual embedding ''clouds'' from the two discrete periods are compared using multiple metrics: cosine distance, average pairwise distance, Hotelling-type T 2 , and maximum mean discrepancy. Results indicate that frequency-based methods align slightly better with human judgments of ''trend-related terms'' than semantic metrics (Precision@50 of 0.62 vs 0.60 in Astrophysics). The two signals show a correlation of around 0.6. Several terms identified exclusively by embedding metrics (e.g., ''primordial black holes'') represent critical conceptual developments invisible to pure frequency analysis. These findings indicate that semantic metrics may capture complementary information, highlighting the value of integrating contextual embeddings into scientometric trend analysis. 

---
# Zero-Shot Cross-Lingual Recognition of Sign Language Handshapes 

**Authors**: Marcel Granero-Moya, Carolina del Corral Farrarós, Gloria Haro, Coloma Ballester, Ricardo Marques  

**Link**: [PDF](https://arxiv.org/pdf/2609.18772)  

**Abstract**: Sign language processing advances rapidly for high-resource languages such as American Sign Language (ASL), yet most of the world's sign languages lack the phonological annotations new methods require. We present the first zero-shot cross-lingual framework for handshape recognition, transferring from ASL to Catalan Sign Language (LSC). Our approach leverages the decomposition of handshapes into five phonological features -- selected fingers, flexion, spread, thumb position, and thumb contact -- shared across both languages, to decode LSC handshapes from predicted features via a composite phonological distance metric. We evaluate three architectures (MLP, SL-GCN, SHuBERT) trained on two ASL corpora (PopSign, Sem-Lex) against a 37-handshape, single-signer LSC benchmark. Zero-shot transfer proves viable once recording-format disparities are harmonized, reaching 80.0% phonological feature accuracy and 54.5% expected handshape accuracy. Phonological decomposition thus offers a bridge for extending sign language technologies to low-resource languages without any target-language video training labels. 

---
# A Scalable Framework for Automated NER Annotation Correction in Low-Resource Languages 

**Authors**: Toqeer Ehsan, Thamar Solorio  

**Link**: [PDF](https://arxiv.org/pdf/2609.18739)  

**Abstract**: Poor quality or noisy annotations in Named Entity Recognition (NER), as in any other NLP task, make it challenging to achieve state-of-the-art performance. In this paper, we present a multi-step framework to enhance the annotation quality of NER datasets by employing automated techniques. We propose a frequency-based iterative approach that leverages self-training and a dual-threshold mechanism to enhance inference confidence. Experimental evaluations on different NER datasets demonstrate significant improvements in NER performance with respect to the original datasets. This work further explores the potential of generative Large Language Models (LLMs) to perform NER for low-resource languages. 

---
# LocQE: Principled Domain Adaptation for Localisation Quality Estimation by Leveraging Post-Edits 

**Authors**: Kathy Hämmerl, Gabriel Bretschner, Joern Wuebker  

**Link**: [PDF](https://arxiv.org/pdf/2609.18720)  

**Abstract**: Learned quality estimation (QE) models such as COMETKiwi are widespread and work well for general machine translation evaluation. However, they are known to struggle on unseen domains, limiting their performance in a real-world localisation context. We show that they are insensitive to some important factors in localisation, such as whether numbers are translated accurately, or even whether the correct number of spaces and punctuation are preserved in a translation. Further, a key capability for optimisation of machine translation is the ability of QE models to accurately rank different translations of a single segment, which suffers significantly from the domain transfer. In the absence of large-scale direct assessment data, we propose principled fine-tuning approaches to reduce the domain gap with even small amounts of post-editing data. Using a multi-task fine-tuning approach and a simple tokeniser intervention, we create a QE model which proves markedly better at distinguishing preferred post-edits from rejected initial translations in a localisation context. We show that preferences and artificial continuous scores stabilise each other, and argue that to calibrate metrics both in terms of their absolute scores and comparisons between translation of the same source, both types of signal are needed. 

---
# Tracing individual knowledge trajectories in a changing field: the case of general relativity and gravitation 

**Authors**: Raphael Schlattmann, Malte Vogl  

**Link**: [PDF](https://arxiv.org/pdf/2609.18697)  

**Abstract**: Historians have reconstructed the twentieth-century transformation of general relativity and gravitation (GRG) at the field level and through individual careers, but connecting these scales requires a way to compare researchers with the changing field over time. We develop such a comparison, setting a researcher's publications and references against GRG field literature from the same, earlier, and later two-year periods. Building on Own Vocabulary and Embedding Density Estimation from our earlier two-case study (arXiv:2501.00391), we extend the analysis to the fifty most-published authors in a NASA/ADS corpus of about 180,000 GRG records (1911 to 2000) and add two citation-based measures, Referenced Vocabulary and Citation Identity. The four measures compare an author's written language, cited literature, semantic neighbourhood, and cited-authority configuration with the surrounding field. The earlier cases suggested that closer field-vocabulary alignment accompanies a denser semantic neighbourhood. Across the fifty authors this holds only partially. Written and cited vocabularies tend to move together, usually resembling later GRG literature as the field turned towards astrophysical and cosmological research. Semantic neighbourhoods more often lie where the field's publications were concentrated in earlier periods, while co-citation patterns follow no single temporal direction, and the two citation measures frequently place the same researcher differently despite drawing on identical reference lists. Individual trajectories can thus combine vocabulary tied to later field states with older semantic or citation structures, and these divergent cases mark patterns for closer historical investigation. The approach transfers to other fields with defensible corpus boundaries and adequate coverage of texts, references, and disambiguated author identities. 

---
# HearInContext: A Benchmark for Implicit Context in Speech Recognition 

**Authors**: Yifan Gao, Yao Tian, Hongbin Suo  

**Link**: [PDF](https://arxiv.org/pdf/2609.18680)  

**Abstract**: Contextual ASR can benefit from semantic cues or from target words explicitly provided in the context. We introduce HearInContext, a Mandarin--English benchmark that pairs shared synthetic speech with assistant replies supporting different interpretations. The benchmark comprises 3,764 semantic test cases built around homophones. Implicit contexts exclude candidate words; explicit contexts name the target. No-context and unrelated-context controls measure the benefit of relevant history and sensitivity to irrelevant history. Context-capable models benefit from implicit cues but achieve higher target recall with explicit hints. Fine-tuning Qwen3-ASR-1.7B improves implicit-context target recall by 11.0 and 11.5 percentage points in Mandarin and English, respectively, while absolute CER/WER changes on AISHELL-1 and LibriSpeech remain below 0.1 percentage points. Gains extend to explicit conditions excluded from fine-tuning and to Mandarin hotword recognition on real recordings. 

---
# Voice of Reason: Reinforcement Learning for Spoken Math 

**Authors**: Timothée Weisselberger, Edouard Graves, Alexandre Défossez  

**Link**: [PDF](https://arxiv.org/pdf/2609.18677)  

**Abstract**: Speech language models enable richer spoken interactions between humans and machines than cascaded systems, allowing access to paralinguistic information and lower latency. However, their accuracy on mathematical reasoning benchmarks has lagged behind those of text models. Reinforcement learning (RL) with verifiable rewards has been instrumental in extending text models' capabilities for solving complex problems and limiting hallucinations. In this work, we explore applying RL to the GLM-4-Voice speech model (Zeng et al., 2024) to bridge the gap between textual and spoken mathematical problem solving. We first adapt the model to the domain using supervised fine-tuning on synthesized spoken question-answering data. We then show that, even without extra reasoning tokens, RL improves the accuracy on GSM8K beyond levels previously achieved for speech models only with supplementary reasoning traces. When combined with existing streaming reasoning techniques, we show further gains to 74.8% free-form accuracy. This establishes a new state-of-the-art for mathematical spoken abilities with speech-native models. 

---
# Selection Is Retrieval, Abstention Is Not: On-Device Tool Routing over 70 Korean-English Actions 

**Authors**: Janghoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.18672)  

**Abstract**: An AI assistant that calls tools makes two decisions on every request: which tool to invoke, and whether any available tool applies. In the usual design a single language model makes both, by emitting a call or by declining to emit one. On a device that has to answer without a server, the language model is what makes that design expensive, dominating both the latency and the memory of the router. The common alternative is to remove the model completely and rank the catalog of local actions with a retriever instead. That substitution is not symmetric across the two decisions. A retriever returns its highest-scoring candidate for every input and cannot signal that the catalog holds no valid action. Our earlier study found that constraining a decoder to a tool grammar repairs malformed output without improving the choice. What the substitution costs in each decision has not been measured. We evaluate the two decisions separately over 600 Korean and English requests and a catalog of 70 local actions. The router may also ask for a missing slot, reply, or delegate. Half the in-catalog requests reuse catalog vocabulary and half paraphrase it, separating lexical overlap from the action requested. Character 3-gram BM25 selects 162 of 164 lexically matched requests and 85 of 166 paraphrases. Restricting the candidate set to seven raises the paraphrase figure to a mean of 0.825 over five trials. No classifier over its score features separates in-catalog from out-of-catalog above 0.697 area under the curve, where the frozen encoder multilingual-e5-base reaches 0.806. Using that encoder for abstention alone keeps 376 of the requests local and misroutes 9 of the 150 needing delegation. Abstention, not selection, is where a neural component is required. A neural ranker improves every quality metric and is rejected on latency and memory rather than accuracy. 

---
# DyMT-ESB: Dynamic Multi-Turn Evaluation of Social Bias in User-LLM Interactions 

**Authors**: Rem Hida, Masahiro Kaneko, Daisuke Oba, Danushka Bollegala, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2609.18649)  

**Abstract**: Warning: This paper contains examples of stereotypes and social bias. LLMs are increasingly used in interactive settings by the general public, making the evaluation of model behavior in multi-turn conversational scenarios important for safety, including stereotyping-related harms. However, existing multi-turn social bias evaluations often rely on pre-specified or template-based user inputs that do not adapt to model responses and typically assume a fixed dialogue length in advance. In this paper, we study social bias dynamics in response-conditioned multi-turn interactions using a controlled evaluation protocol that generates follow-up user queries from the evolving dialogue history and allows evaluation over variable numbers of turns. Experimental results show that LLMs exhibit social bias even in coherent, response-conditioned multi-turn interactions, revealing late-emerging bias, non-monotonic bias patterns, and bias re-emergence. These results motivate evaluations that extend beyond fixed-turn, pre-scripted protocols. Our findings highlight the importance of analyzing social bias as a turn-level dynamic phenomenon. 

---
# Fallacy Benchmarks Measure Scheme Recognition, Not Fallacy Detection 

**Authors**: Navyansh Singh, Animesh Pathak, Aarav Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.18644)  

**Abstract**: Fallacy-detection benchmarks pair fallacy classes with a single "valid" or "none" class that takes everything data collection did not label as a fallacy. This construction is misleading: a classifier can learn cues that do well on this class without learning to tell a fallacy from a correct argument. We show that the low false-positive rates benchmarks report are an artifact of how the class is built, not evidence of detection ability. The most informative negative for a fallacy is a correct argument using the same argumentation scheme, and such arguments are at most a few percent of the valid class across the four benchmarks we examined. Evaluated on constructed scheme-matched negatives, false-positive rates rise from 16.6% to 58.9% on CoCoLoFa and from 5.7% to 62.0% on Reddit. That rate depends on how the negatives are written, so we also compare two conditions from the same pipeline that differ only in scheme identity. Classifiers label scheme-matched negatives as the source fallacy type 40.9 points more often than wrong-scheme negatives, which are instead identified as the scheme they actually use 85.9% of the time against 0.4% for the source type. The classifier has learned which scheme an argument uses, not whether it uses it correctly, and on the benchmarks' own test sets the two are indistinguishable. The same dissociation appears in three zero-shot LLM detectors that never saw these benchmarks, and the measurement is far lower on a negative class that was built deliberately. We release the items as Scheme Foils. A reported false-positive rate should not be trusted as a measure of detection until the valid class has been audited for scheme-matched coverage. 

---
# STRETCH the Boundaries: A Unified Self-Taught Framework for Progressive LLM Evolution 

**Authors**: Yajie Yu, Mark Lee, Yue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2609.18642)  

**Abstract**: Large language models (LLMs) often suffer from capability stagnation in self-improvement training because fixed difficulty levels fail to adapt to their evolving proficiency. To address this issue, we propose STRETCH (Self-Taught Reasoning Evolution via Targeted CHallenge), a unified framework inspired by cognitive scaffolding theory. STRETCH introduces a dynamic Stretch Zone mechanism that continuously aligns question difficulty with the model's solving capability. Within a single parameter space, the model alternates between a Scaffolder that generates adaptive, boundary-pushing challenges and a Learner that that optimizes its solving trajectories through reinforcement learning. This dual-loop co-evolution effectively stabilizes training, mitigates reward hacking and promote progressive reasoning growth. Experiments on both negotiation and operation research benchmarks demonstrate that STRETCH consistently outperforms strong prompting and domain-specific baselines. Further scaffolder configuration analysis shows that dynamic difficulty alignment is critical for sustained capability improvement and synchronized reasoning evolution. 

---
# PACT: Can Enterprise AI Assistants Be Trusted Under Pressure? 

**Authors**: Mika Okamoto, Ansel Kaplan Erol  

**Link**: [PDF](https://arxiv.org/pdf/2609.18605)  

**Abstract**: As corporate AI adoption continues to grow, enterprise-grade LLM agents are being deployed into sensitive contexts such as hiring, healthcare, and finance. In these contexts, compliance with rules specified in an agent's system context is a first-order legal concern. Currently, no evaluation framework systematically measures which LLM models tend to violate compliance rules, especially under pressure from a persistent user, a hurried manager, or circumstances where violation is convenient or attractive. We introduce PACT (Pressure-Applied Compliance Testing), a benchmark for rule-following under pressure in AI agents assisting employees in daily tasks across twelve regulated enterprise domains and forty-eight scenarios, each set in a realistic multi-turn conversation. Each benchmark item pairs a standing rule against a rule-violating shortcut, and applies a battery of pressures across different wordings and system-prompt modes. We construct PACT component by component under strict LLM-as-judge auditing to ensure samples are unambiguous, ungameable, and realistic enough to avoid eliciting evaluation-aware behavior. We use PACT to profile LLM compliance across six complementary metrics that create a holistic picture of an AI assistant's robustness under pressure and throughout multi-turn conversations, its transparency, and ability to correctly discern where a rule applies. We aggregate this profile into PACTScore, a reliability-weighted compliance rate over all items and modes. Our results across 22 common LLM models spanning multiple providers and sizes show substantial variability in compliance across models and metric dimensions. Even the strongest assistants mis-apply a rule on 6 to 10% of items, and ordinary user pressure raises the violation rate by 65% on average. PACT highlights compliance risks in LLM assistants, motivating guardrails and careful model selection. 

---
# A Probe Shift Is Not a Fairness Fix: The Limits of Representation Steering in Speech Models 

**Authors**: Nicolas Bourrel, Abderrahmane Issam, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2609.18533)  

**Abstract**: Automatic speech recognition (ASR) systems exhibit unequal error rates across speaker groups, motivating interventions on their internal representations. We ask whether speaker-linked attributes that are linearly readable from pretrained ASR encoders yield useful directions for reducing group word-error-rate (WER) gaps. Across Whisper-medium, HuBERT-large, and Wav2Vec2-large on Common Voice and the Speech Accent Archive, we probe every encoder layer for metadata-derived sex/gender, age, and native/accent labels; construct centroid and probe-derived directions; inject them at selected layers; and compare downstream probe trajectories with matched WER changes. Sex labels are highly decodable (best macro-F1 0.924--0.941), native/accent labels are also above chance (0.544--0.696), and age is weaker (0.354--0.397). Of 22 post-selected reruns, nine have 95% paired-bootstrap intervals entirely below zero, yet every absolute source-group WER reduction is below 0.7 percentage points. Conversely, a local target-class probe rate can rise from 8.09% to 99.87% while WER worsens. Linear readability is therefore neither evidence of causal use nor a reliable mitigation method. Our results motivate evaluating speech-bias interventions jointly at representation, propagation, and task levels. 

---
# Machine Translation between English and Syriac (East Syriac Dialect) using Statistical Machine Learning 

**Authors**: Hadiana Sliwa, Hossein Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2609.18529)  

**Abstract**: UNESCO considers the Assyrian (Syriac) language an endangered language. Although Assyrians speak the language worldwide, the speaking population is uncertain (ranging from 500,000 to 1,500,000). Syriac is also one of the least studied languages in Natural Language Processing (NLP). Despite advances in Machine Translation (MT) over the past decade, the lack of publicly available corpora and the orthographic complexity of the Syriac script, specifically the Madnkhaya script, have left this language entirely ignored in the computational linguistics literature. This study develops the first phrase-based Statistical MT (SMT) model for English-to-Assyrian MT using the Moses framework. We created a dataset of 38,847 sentence pairs from the complete English and Syriac Bible, merging a pre-existing New Testament dataset with an Old Testament built from scratch through PDF extraction, using custom segmentation scripts and manual alignment review by three bilingual annotators. The Syriac side of the corpus undergoes diacritic removal and Byte-Pair Encoding tokenization to reduce orthographic sparsity before training. We trained and evaluated six models using different configurations and splitting-scheme ratios, language model order, distortion limits, and the inclusion of an Operation Sequence Model. The best-performing configuration achieves a word-level BLEU score of 23.54. Human evaluation by 11 native Assyrian speakers resulted in mean adequacy and fluency scores of 3.42 and 3.34 out of 5, respectively. These results are consistent with comparable low-resource SMT models trained on Biblical corpora for morphologically rich Semitic languages. The corpora, scripts, and trained model are publicly available, providing the research community with the first systematically curated English-Syriac dataset and a reproducible baseline for future MT and broader NLP work on this endangered language. 

---
# Align, Integrate, and Fire: Efficient Token-Level Alignment for Zero-Shot SpeechLLMs 

**Authors**: Abderrahmane Issam, Yusuf Can Semerci, Jan Scholtes, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2609.18516)  

**Abstract**: While Large Language Models excel in natural language processing, efficiently extending their capabilities to spoken input remains a significant challenge. Existing methods for building SpeechLLMs often rely on computationally expensive full-model fine-tuning, or employ parameter-efficient projectors that suffer from inefficient token sequence lengths and costly full-model supervision. In this paper, we introduce Aligned Continuous Integrate-and-Fire, a highly efficient framework for zero-shot speech processing. Our method dynamically compresses continuous acoustic frames into the exact discrete token length of the target text utilizing explicit Dynamic Time Warping alignments. This allows our initial training stage to establish a robust acoustic-to-semantic bridge using lightweight distance metrics, entirely bypassing the computationally expensive LLM forward pass. For subsequent fine-tuning, we propose a memory-efficient knowledge distillation objective that targets a single LLM layer, performing competitively with full-model cross-entropy training at a fraction of the computational cost. Through extensive evaluations on Automatic Speech Recognition and Speech Translation, we demonstrate that our method achieves superior performance compared to prior parameter-efficient baselines. 

---
# Size Matters: Foundation Model for Czech HTML documents 

**Authors**: Martin Dvořák, Vít Tlustoš, Artyom Voronin, Martin Habrovec, Kateřina Podlesná, Barbora Rišová, Josef Vonášek  

**Link**: [PDF](https://arxiv.org/pdf/2609.18494)  

**Abstract**: Creating universal, high-quality representations of web documents in high-traffic industrial environments requires models that are both performant and economic. Existing approaches, however, often depend on large models, overlook the structural information inherent in HTML, or are constrained by short context windows, limiting their ability to process real-world web pages. We present HTML-LM, a compact foundation model with 154 million parameters that addresses these limitations through HTML-aware training and a ModernBERT-based architecture. It was trained on 100 million web documents using multiple objectives, including masked language modeling, bag-of-words prediction, and contrastive distillation from large language models. Consequently, HTML-LM sets a new state-of-the-art for classification and regression applications in the Czech Internet domain, surpassing both larger encoders and small-sized LLMs. The model is deployed in production, processing thousands of web documents per second, and released to the community under the CC BY-NC 4.0. this https URL. 

---
# M-SQE: Multilingual Skill Quality Estimation for Enhancing Language Equality in Agentic Skill Use 

**Authors**: Yilun Liu, Shimin Tao, Minggui He, Chenxin Liu, Li Zhang, Chen Liu, Miao Zhang, Jiaxin Guo, Min Zhang, Liqun Deng, Xiaojun Meng, Daimeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2609.18445)  

**Abstract**: Agent skills, reusable procedural documents that extend LLM agents beyond their parametric memory, have become an important interface for deploying agents on real-world tasks. Community-maintained skill libraries built around this interface are growing rapidly. However, this ecosystem remains deeply English-centric: our audit finds that low-resource languages such as Swahili and Hindi have no in-language skill content, so retrieval often returns a skill written in a different language than the query, degrading accuracy and recall. A practical solution is to synthesize in-language skills for retrieval but the quality can be unreliable, so relevance in this setting alone often surfaces a related but unusable candidate. To address this, we propose M-SQE, a post-retrieval Multilingual Skill Quality Estimation framework that scores candidates via a Theory view for intrinsic quality and an Action view for task-grounded utility, unified into a domain-conditioned final score. We evaluate M-SQE across three skill-use domains: general, tool-use, and cultural tasks. Empirically, we build three-layer candidate skill pools mirroring today's ecosystem, where M-SQE's task success exceeds existing baseline's average by at least +3.5 points across three different retrievers. Particularly, M-SQE lifts the lowest-resource languages most (+12.9pp on Hindi and +5.6pp on Swahili) and achieves strong performance across all six culture regions, thereby moving agentic skill use toward linguistic and cultural equality. 

---
# Planning or Improvisation? Stress-Testing the Poetry Planning Site on Open Models and Open Cross-Layer Transcoders 

**Authors**: Éric Jacopin  

**Link**: [PDF](https://arxiv.org/pdf/2609.18440)  

**Abstract**: Lindsey et al. (2025) report that Claude 3.5 Haiku plans rhymes: features for candidate rhyme words are active on the newline before a line is written, and a suppress-and-inject intervention redirects the line only when applied there (their Figure 13). We test how far this generalizes on seven cells crossing four open models (0.6B to 2.6B parameters) with six open cross-layer transcoders (CLTs), on one consumer GPU, decomposing the claim into position specificity (C1), newline site identity (C2), and a newline-resident plan (C3). This is a stress test rather than a faithful reproduction: attribution graphs are unavailable for these CLTs, so features are found bottom-up from decoder vectors. C1 generalizes, in every cell and in all 247 of 444 prompt-by-inject pairs with a detectable effect, but the effective position is the final prompt token, adjacent to emission, and only two cells reach behaviorally meaningful probabilities. C2 and C3 are not recovered by any probe: a census of every active feature finds no rhyme-anticipating enrichment at the newline, and steering the newline while the model composes the whole line, over 36 runs and 8,640 sampled lines, shows why. That intervention is strong but one token long, making the injected word the first word of the composed line in 703 of 720 samples and leaving the rhyme six words later untouched. A final test drops the transcoder entirely: patching the newline's whole residual, at every layer, from a minimal-pair poem whose third line ends on a different rhyme moves the rhyme in 11 of 1,260 composed lines against 4 at baseline, with a design resolving 1.4%. We read this as a boundary condition rather than a refutation: at this scale and with these transcoders, the causal site is emission-adjacent. We reproduce Figure 13's shape, not its mechanism. Code and data are public (code: this http URL). 

---
# Dependency-Aware Trajectory Refinement for Efficient Multi-Turn Agent Fine-Tuning 

**Authors**: Zhuo Chen, Zhen Zhang, Xinyu Wang, Kewei Tu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18417)  

**Abstract**: Multi-turn agent trajectories often contain redundant rounds (failed tool calls, parallel sub-queries, verification-only steps) that inflate both training and inference cost. We propose viewing each trajectory as a \emph{round-level dependency DAG} that exposes which rounds are globally load-bearing for the final answer, and fine-tune agents on trajectories refined through this DAG. Given an LLM-annotated DAG, these edits are deterministic and interpretable, with optional rephrasing. Models trained on these refined trajectories consistently outperform those trained on the original trajectories at lower inference cost. Specifically, across four multi-modal QA benchmarks, our refinements improve downstream accuracy by up to $1.7$\,pp over vanilla SFT (and $5.7$\,pp over an LLM-deletion baseline) while reducing per-sample inference messages by up to approximately $40\%$ and inference tokens by up to approximately $48\%$, translating to substantial savings in compute and serving cost. Code is available. 

---
# Emotion Experience, Expression, and Perception: Emotion Analysis on Multimodal Social Media Posts 

**Authors**: Christopher Bagdon, Carina Silberer, Roman Klinger  

**Link**: [PDF](https://arxiv.org/pdf/2609.18385)  

**Abstract**: Emotions are an essential aspect of human communication, particularly on social media, where authors frequently combine text and images to convey their emotions. Yet prior work on emotion analysis of social media posts has overlooked two important aspects in regard to measuring how well readers can reconstruct the authors' intent: (1)~the image modality, with most work focusing solely on text, and (2)~the real-world events that trigger the expressed emotions, and their relationship to the post content. We therefore study the relation between (a) the author's experience of the event that caused them to write a social media post and (b) the content of the post, with a focus on readers' capability to reconstruct that emotion expression. To do that, we introduce the Multimodal Multi-Emotion-Model dataset Mult2EMo, created by collecting annotations from both authors and readers on the posts and their triggering events. We find that reconstruction is possible but challenging for both human readers and computational models. We show that understanding the triggering event is crucial for accurate reconstruction, and that reconstruction is particularly challenging when posts rely heavily on the image to express emotion. 

---
# Attention Dispersion as a Diagnostic Signal for Hallucination in Large Language Models 

**Authors**: Shardul P. More, Tanuja S. Pawar  

**Link**: [PDF](https://arxiv.org/pdf/2609.18320)  

**Abstract**: Large Language Models (LLMs) frequently exhibit hallucinations, presenting a major barrier to reliability in complex reasoning tasks. While traditional detection methods rely on output-based confidence metrics, these logits are often miscalibrated by modern alignment techniques. In this paper, we investigate the temporal volatility of internal attention mechanisms as an alternative diagnostic signal for hallucination that does not depend on output calibration. By introducing an unsupervised metric for attention dispersion, we show that epistemic uncertainty leaves a measurable trace within intermediate layers, where spikes in attention entropy are associated with reasoning breakdowns. We evaluate our approach on mathematical reasoning benchmarks (GSM8K and MATH-500) using the Qwen2.5 model family (1.5B and 3B parameters), finding statistically significant AUC improvements of up to +0.076 over output-based baselines across all tested conditions. These findings suggest that attention dispersion is a promising complement to traditional hallucination detection methods, requiring further investigation across broader model families and task domains. 

---
# Knowledge-Graph Based Augmentation versus Retrieval Augmented Generation for Cultural-Related Question Answering 

**Authors**: Pablo Poulenard, Yannis Karmim, Valentin Barrière  

**Link**: [PDF](https://arxiv.org/pdf/2609.18317)  

**Abstract**: Large language models (LLMs) suffer from a long-tail deficit: culturally specific facts, particularly those concerning underrepresented regions such as Latin America, appear too rarely in pretraining corpora to be reliably memorized. Retrieval-Augmented Generation (RAG) addresses this by grounding generation in external text, but structured alternatives such as Knowledge Graphs (KGs) offer tighter control over what enters the context, along with potential gains in explainability and updatability. We benchmark Graph-RAG against standard RAG on LatamQA, a culturally grounded multiple-choice dataset spanning eight thematic categories. The graphs are built end-to-end from Wikipedia articles with KGGen, a recent open-domain extractor, without manual curation in our main setting. G-Retriever is competitive with RAG and reduces the error of the base LLM by 72\% with a standard KG and 78\% with a benchmark-aware variant, the gap to RAG narrowing further as the graph is oriented toward task-relevant content. The trained projection transfers zero-shot to Portuguese without target-language fine-tuning, indicating multilingual reach. 

---
# SEA-LION-v4.8: A Technical Report 

**Authors**: Ahmed Mohammad Dabeer, Ahn Jeongmi, Anocha Sutaveephamochanon, Antonyrex Sajeban, Aulia Adila, Chan Hok Teng, Adwin, Cheng Zi Yi, Nicholas Zhuang Ziyi, Choa Hsueh Mei Esther, David Ong Tat-Wee, Evelyn Tan Chor Phin, Heng Cheng Peng, Jonathan, Lee Chwan Ren, Leong Wai Yi, Leong Wei Qi, Leslie Teo Eng Sipp, Liew Rachel, Limkonchotiwat Peerat, Montalan Jann Railey Estrada, Muhammad Ridzuan Bin Mokhtar, Nagarajan Karthik, Ng Boon Cheong, Raymond, Ngui Jian Gang, Nguyen Thanh Ngan, Tasawong Panuthep, Pereira Mark Gregory, Phang Shi Wei Benjamin, Poon Yip Hung, Joseph, Rengarajan Hamsawardhini, Siow Wei Kang Bryan, Tai Ngee Chia, Tan Choon Meng, Tan Le Min, Sheryl, Tan Siao Wei, Tan Yi Xian, Tee Jun Yun, Teng Kok Wai, Tjhi William Chandra, Tuchinda Pume, Wu Donghang, Yong Xianbin, Yosephine, Zhang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.18310)  

**Abstract**: We introduce Nemotron-SEA-LION-v4.8, a family of Southeast Asian Languages in One Network (SEA-LION) built upon NVIDIA Nemotron 3. The family includes 30B-A3B and 120B-A12B models, with both continued-pretrained base checkpoints and post-trained variants. We adapt the models using Southeast Asian, reasoning, code, and multilingual parallel datasets, followed by post-training with supervised fine-tuning and online on-policy distillation. On SEA-HELM, the 30B-A3B model improves the overall SEA score from 46.06 to 51.57, while the 120B-A12B model improves from 49.30 to 63.44. The strongest gains are observed in instruction following, natural language reasoning, and natural language understanding across seven Southeast Asian languages. 

---
# Rollback the World, Keep the Reflection: Rollback-Induced Reflection for Long-Horizon LLM Agents 

**Authors**: Yi Yu, Liuyi Yao, Yaliang Li, Enshu Wang, Libing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18304)  

**Abstract**: Large language model (LLM) agents increasingly tackle long-horizon tasks through multi-step environment interaction, yet a single erroneous action can alter subsequent states and observations, causing errors to compound over time. Existing methods either correct the context without repairing altered environment states or restore earlier states while discarding useful experience, making it difficult to both eliminate failure conditions and avoid repeating past mistakes. We argue that reliable recovery should instead be treated as a rollback-boundary control problem that jointly determines when to intervene, where to resume, and what information should survive recovery. Based on this view, we propose Rollback-Induced Reflection (RIR), a unified recovery framework that restores execution to a selected prior state while carrying forward reusable knowledge distilled from the abandoned trajectory to guide subsequent decisions. We further characterize recovery through a unified operator over rollback depth and retained memory, providing a general view of state restoration and knowledge retention. Experiments on three long-horizon benchmarks demonstrate that RIR consistently improves task performance across multiple LLM backbones, with structured reflection memory preserving useful experience and selective rollback enabling efficient recovery. 

---
# Made in Hungary: Comments on the performance of generative language models 

**Authors**: Mátyás Osváth, Enikő Héja, Noémi Ligeti-Nagy  

**Link**: [PDF](https://arxiv.org/pdf/2609.18284)  

**Abstract**: In recent years, three initiatives have emerged to develop generative language models in Hungary. The motivation behind them is the same. For Hungarian, no model with the given capability existed, or existing English-centric models offered limited proficiency. A detailed examination of the corresponding studies, however, reveals several methodological limitations. First, the reliability of the evaluation protocols is questionable. Contrary to the findings of Csibi et al. [2026], evaluation under the recommended inference settings shows that Qwen3-4B achieves higher scores than Racka-4B, its Hungarian-adapted version. Data contamination is evident in the work of Yang et al. [2025d] and Szentmihályi et al. [2025], potentially biasing the reported results. Second, the training pipelines fall short of current best practices in corpus curation and data mixture, which risks wasting substantial compute on low-quality data. The lack of controlled ablations prevents reliable assessment of these choices. Third, none of the three papers assessed forgetting or capability loss. Testing the adapted models on a subset of the original benchmarks indicates performance decline in all three cases, especially Racka-4B. These observations emphasize the importance of rigorous experimental design in language model development, given the significant computational and financial costs involved. 

---
# Too Good to Be Real? Diagnosing and Reducing the Gap Between AI Preference and Real User Engagement 

**Authors**: Xinglang Zhang, Yuanmeng Xiang, Yunyao Zhang, Zeliang Chen, Junqing Yu, Zikai Song  

**Link**: [PDF](https://arxiv.org/pdf/2609.18282)  

**Abstract**: Large language models are increasingly used to generate and evaluate online content, yet it remains unclear whether the qualities they associate with higher engagement match what real users respond to. We study this question using 1.17 million answers to 25,978 questions from Zhihu, Quora, and Reddit, comparing real platform answers and AI-generated answers across four within-question engagement levels. We introduce Ontological Preference Measurement, which represents answers along three dimensions: logic, affect, and expression. We find a systematic gap between AI preference and real user engagement: as target engagement increases, LLMs add more explicit logical structure, while real user engagement is more strongly associated with affective and expressive salience. We call this tendency logic overbinding. Based on this diagnosis, we propose Ontology-Masked Reasoning Autoencoding (OMRA), a controlled intervention that masks and reconstructs over-explained spans while preserving stance, factual content, and coherence. Across four LLM families, OMRA reduces the measured gap by an average of 54.4%. In human evaluation, OMRA wins 62.4% of pairwise preference judgments against matched real platform answers, even though the real answers are more often judged to be human-written. 

---
# I code or AI code: A comparative evaluation of AI-rated scores in classroom observations 

**Authors**: Y. Fong, J. Xiang, T.Y.D. Chan, K. Lee, E.Y.H. Lau  

**Link**: [PDF](https://arxiv.org/pdf/2609.18274)  

**Abstract**: Classroom observations are widely recognized as a key tool for establishing benchmarks of education quality and guiding pedagogical improvement, yet they remain resource-intensive and dependent on trained observers. This study evaluated the feasibility of using a LLM (GPT-5 model) to score teacher-child interactions in early childhood classrooms, benchmarked against human raters. The study analyzed 87 video-recorded observations from 38 classrooms across 30 kindergartens in Hong Kong. Using observation transcripts, the AI model was configured to apply the full Classroom Assessment Scoring System (CLASS) framework. AI-rated scores were then compared with human ratings by examining correlations and differences in mean scores of the CLASS domains and dimensions. The results showed greater convergence between AI and raters for the Emotional Support domain and, in particular, the Quality of Feedback dimension, which captures how teachers use feedback to extend children's learning. Greater divergence emerged for interactions that were more procedural or context-dependent, particularly within the Classroom Organization and Instructional Support domains. These findings suggest that transcript-based AI scoring may capture some of the relative variation in teacher-child interactions but cannot yet reproduce calibrated human judgements consistently across the full CLASS framework. AI-assisted observation may therefore be more appropriate as a preliminary screening tool rather than as a replacement for trained observers, providing teachers with evidence for reflection rather than high-stakes evaluation. Future research should examine whether domain-specific training and incorporation of contextual and visual information can improve alignment between AI and human rated scores. 

---
# Beyond Accuracy: How Procedural Traces Shift the Decision Criterion of LLM Overseers 

**Authors**: Zihan Chen, Di Zhu, Lei Zheng, Weiling Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.18204)  

**Abstract**: Organizations increasingly use oversight loops where one large language model (LLM) audits another's outputs alongside procedural traces of claimed steps. A common concern about such LLM-as-a-judge pipelines is that detailed traces make overseers gullible. Using signal detection theory, we audit five LLM overseers on 19 compliance tasks (4,551 analyzed judgments), varying only trace detail and evidence labeling. With disconfirming evidence always visible, error detection remains near ceiling. Instead, elaborate traces shift the decision criterion toward rejection, increasing false alarms in susceptible overseers. Without option labels, human-validated reason coding shows about 60% of false alarms cite an inability to tie evidence to its option. Labels eliminate this stated reason, yet residual rejection of correct work persists in those overseers and rises with trace detail. Procedural traces thus act as governance artifacts that shape oversight decisions. AI auditors should be evaluated by their decision criterion and false-alarm behavior, alongside accuracy. 

---
# Behavior2Value: Benchmarking and Empowering LLMs for Consumer Value Measurement from E-commerce Behaviors 

**Authors**: Peixuan Hou, Bin Chen, Li He, Jian Xu, Bo Zheng, Xiuli Ma, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2609.18203)  

**Abstract**: Human values are deep motivational orientations that shape human behaviors. In e-commerce, they reveal the stable drivers behind users' purchase decisions. Compared with short-term interests, consumer values better explain how users evaluate products before purchase. However, consumer values are often implicit in complex and fragmented behavioral trajectories, leaving value measurement from e-commerce behaviors largely underexplored. To this end, we propose the Behavior-to-Value (B2V) task, which aims to identify consumer values from e-commerce behavioral trajectories. Centered on this task, we first construct the E-commerce Consumption Value Taxonomy (ECVT) and introduce B2V-Bench, the first B2V dataset and benchmark, based on anonymized Taobao behavioral logs. B2V-Bench consists of real-world purchase decision episodes, covering 25 types of purchase behaviors, along with corresponding consumer value orientations manifested in each episode. To improve consumer value measurement accuracy, we further present B2V-Verifier, a behavior-to-value measurement model based on Value Verification Tuning, which learns to assess whether behaviors provide sufficient evidence for each value inference. Experiments show that B2V-Verifier outperforms strong LLM baselines, improving multi-label classification by 34\%. The dataset and code will be publicly released upon acceptance. 

---
# T-SANDHI: Tone Sandhi-aware Adaptive Network with Decoupled Hybrid Injection for Low-resource Taiwanese Hokkien Speech Recognition 

**Authors**: Hung-Yang Sung, Chien-Chun Wang, Tien-Hong Lo, Yu-Sheng Tsao, Yung-Chang Hsu, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.18194)  

**Abstract**: In Taiwanese Hokkien automatic speech recognition (ASR), prior studies often treat tone sandhi as a major challenge under the assumption that models fail to process implicit phonological variations. However, our experiments on Taiwanese Hokkien reveal that speech foundation models actually handle tone sandhi variations effectively, and the real performance bottleneck stems from a localized confusion between these variations and retained citation tones. To address this, we propose T-SANDHI to explicitly decouple surface acoustics from underlying lexical intent on top of a frozen Whisper backbone. Using a lexicon-guided multi-task learning structure driven by text-derived pseudo labels, our lightweight hybrid injection module integrates independent citation and sandhi phonetic streams via dynamic gating. Extensive evaluation on the TAT-MOE corpus and two blind test sets demonstrates that this explicit disentanglement effectively resolves tonal mapping confusion, outperforming baselines with strict parameter efficiency. 

---
# TeochewBench: A Human-Reviewed Benchmark for Teochew Hanzi Translation 

**Authors**: Jianan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18156)  

**Abstract**: Teochew has a substantial speaker community and exhibits distinctive lexical, syntactic, and pragmatic features, yet textual resources for evaluating large language models remain limited. We present TeochewBench, a human-reviewed benchmark comprising 300 Teochew Hanzi expressions for evaluating translation from Teochew Hanzi into Mandarin Chinese and English. The dataset covers five categories: basic vocabulary; everyday sentences; Teochew-specific expressions; tone, politeness, and context; and idiomatic, ambiguous, and culturally specific expressions. A primary Teochew-speaking reviewer examined all entries individually and revised them as needed, while two additional Teochew speakers verified selected items.
Our main evaluation covers 11 official general-purpose post-trained models on the reviewed dataset in both translation directions, yielding 6,600 predictions. Two official base checkpoints provide 1,200 predictions for supplementary diagnostics, bringing the total to 13 models and 7,800 predictions. We additionally include a Hanzi-copy control, which returns the source input unchanged, to assess how shared Hanzi affect automatic scores for translation into Mandarin Chinese. Qwen3.5-27B achieved the highest overall chrF-style score among the evaluated checkpoints, at 60.63, followed by Qwen2.5-72B-Instruct at 56.61, Gemma-3-27B-IT at 56.36, and GLM-4-32B-0414 at 55.82. Across the 11 main-evaluation models, the mean chrF-style score decreased from 69.25 for low-specificity items to 27.52 for high-specificity items. High-specificity expressions received lower scores and exhibited smaller cross-model differences, suggesting that they constitute a shared low-scoring region across the model families evaluated here. The Hanzi-copy control further indicates that surface overlap in low-specificity items can substantially affect automatic scores for translation into Mandarin Chinese. 

---
# DualSQL: Text-to-SQL with Multi-Agent Reinforcement Learning 

**Authors**: Shijie Chen, Yu Gan, Yeounoh Chung, Jiani Zhang, Quannan Li, Sravan Babu Bodapati, Cody J. Greer, Yu Su, Fatma Ozcan  

**Link**: [PDF](https://arxiv.org/pdf/2609.18135)  

**Abstract**: State-of-the-art Text-to-SQL systems are typically multi-agent pipelines centered around two fundamental tasks: schema linking and SQL generation. However, existing work trains separate models for each task, failing to leverage the synergy between these interrelated tasks. In this work, we propose DualSQL, a new Text-to-SQL system consisting of two agents powered by a single model backbone. The agents share the same model weights and agentic scaffold, enabling joint optimization through a robust multi-agent reinforcement learning (RL) framework. We design three database access tools to facilitate effective multi-step reasoning grounded to interactions with the databases. To improve training and avoid model collapse, we introduce a set of rollout guardrail mechanisms that stabilizes multi-agent RL training, supporting DualSQL to keep improving during training. We also introduce a new SQL correctness metric, robust execution match (REX), to more accurately judge SQL correctness and assign reward signals. Being trained on only 3755 examples, DualSQL-4B achieves an impressive 68.0% execution accuracy on the BIRD development set, matching previous 7B models. DualSQL-8B further improves to 71.1%, outperforming previous state-of-the-art single-model solutions with 32B parameters. These results demonstrate the strength of joint multi-agent reinforcement learning for building high performance Text-to-SQL pipelines. 

---
# Linguistic Triggers of Gender and Racial Bias in Open-Weight LLMs Applied to Recruitment 

**Authors**: Kosuke Kitahara, Nobuhiro Yamaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2609.18106)  

**Abstract**: Open-weight large language models are rapidly entering hiring pipelines, yet their discriminatory failure modes -- and the regulatory exposure these create under the EU AI Act high-risk classification (Annex III) and U.S. EEOC adverse-impact analysis -- remain poorly understood. We present the first systematic, multi-model audit of open-weight LLMs that treats job-posting language as the primary experimental variable, evaluating six models (Llama 3.2, Mistral, Gemma 3, Qwen 3, Phi 3, DeepSeek-R1) across four controlled experiments that jointly probe recruiter-simulation and job-seeker-simulation tasks. We find that (1) agentic posting language depresses recruiter recommendation scores for female candidates (r_rb = 0.309, p_Bonf = 7x10^-5; model-fixed-effects r_rb = 0.448), while communal language partially reverses the penalty; and (2) coded-exclusion language suppresses non-White recruiter scores at large effect sizes (r_rb = 0.646-0.758) and, on the job-seeker side, selectively deters non-White personas from expressing interest -- operationalizing a chilling-effect mechanism at scale. A label-ablation experiment isolates the explicit demographic persona label as the primary causal driver, and Word Embedding Association Tests corroborate these findings at the representational level (d = 1.01-1.45 under Caliskan et al.'s multi-word gender attribute lists). We translate these results into a concrete pre-deployment audit protocol -- posting-vocabulary scoring, persona-conditioned LLM probing, and adverse-impact flagging against the four-fifths threshold -- that operationalizes the documentation and risk-management obligations Annex III imposes on high-risk AI in recruitment. 

---
# From a River in Gilead to the Inference Distributions of Large Language Models: Covert Dialect Bias and Linguistic Profiling at Scale 

**Authors**: Chowdhury Mohammad Abdullah, Rita Orji  

**Link**: [PDF](https://arxiv.org/pdf/2609.18068)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes domains such as housing screening. While alignment techniques mitigate explicit racial bias in generated text, they often leave covert attitudinal associations in internal probability distributions untouched. Adapting the matched-guise sociolinguistic paradigm, we examine covert dialect bias in housing-related social judgments across four varieties: Standard American English (SAE), African American Vernacular English (AAVE), Nigerian Standard English (NSE), and Nigerian Pidgin (NP). AAVE reflects the racialized dialect studied in prior covert-bias evaluations, whereas NSE and NP represent Black African, postcolonial varieties absent from this literature. Using 260 meaning-matched sentence quadruples and log-probability scoring over housing-relevant adjectives, we probe ten open-weight LLMs across three contexts varying in social proximity: tenant screening, neighbor acceptance, and roommate selection. Across all ten models, AAVE and NP are consistently associated with more negative adjectives than SAE, with NP penalized most severely. Crucially, each dialect is penalized via distinct stereotype clusters rather than a generic non-standard category. NSE, which carries institutional prestige, displays a context-dependent shift: favored over SAE in formal tenant screening but increasingly penalized as social proximity grows. Our findings reveal that LLMs inherit covert dialect bias along both racial identity and prestige dimensions, echoing documented human housing discrimination and demonstrating its reach across postcolonial English varieties. 

---
# Exact semantic readout from compressed vector representations 

**Authors**: Daniel Quigley  

**Link**: [PDF](https://arxiv.org/pdf/2609.18047)  

**Abstract**: We characterize when compressed vector representations admit exact linear or affine readouts of a finite lexicon's truth conditions: one fixed map per predicate, sending each entity vector to the corresponding truth vector. A necessary and sufficient row-space condition determines existence; the augmented truth matrix has rank r, giving minimum dimension r in the linear case, and r-1 in the affine. Exact readouts return values in a shared truth basis on which Boolean connectives act unchanged; separability alone requires an intervening threshold. For binary relations, exact bilinear readout of identity or strict total order requires linearly independent entity vectors. Experiments with GloVe and word2vec distinguish exact affine recovery, linear separability, and held-out prediction: most predicates are strictly separable, but none admits an exact affine readout from the pretrained embeddings. Supervised transductive training attains exact affine recovery to numerical precision at every tested dimension meeting the bound. At the embeddings' original dimension, geometries constrained to exact linear recovery retain 98-99 percent of the pretrained variance on the feature norms, and 80-83 percent on the WordNet lexicon. 

---
# Gaze as Evidence for Common Grounding: A Cross-Corpus Analysis of MapTask and MUNDEX 

**Authors**: Nan Li, Albert Gatt, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2609.18011)  

**Abstract**: In collaborative tasks with asymmetric information, participants coordinate their understanding through interaction. We ask whether gaze provides evidence about grounding across two such tasks. Working from discrete behavioral annotations, we map HCRC MapTask (Anderson et al., 1991) and MUNDEX (Türk et al., 2023) into a shared partner/task/away vocabulary and compute gaze features around task-relevant dialogue units. In both corpora, aligned reference interpretations (MapTask) and UND (understood) judgments (MUNDEX) are associated with more task-directed gaze and with less partner-directed gaze, lower gaze entropy, and fewer gaze transitions. The associations are clearest for the participant leading the task: in giver-produced references, and in explainer judgments, which also co-vary with the explainee's gaze. In same-speaker MapTask reference chains, the speaker's gaze entropy is lower at the mention where a previously non-aligned referent becomes aligned. The best gaze feature groups improve modestly over controls under grouped cross-validation: temporal features in MapTask and raw proportions in MUNDEX. Because effects are small and several weaken when recurring participants rather than dialogues are the unit of inference, we treat gaze as one contributing cue to grounding, to be interpreted alongside task and dialogue context. 

---
# A Calibrated Instrument for Measuring How Inference Optimizations Affect Output Quality 

**Authors**: Jerry Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2609.18005)  

**Abstract**: Large language model optimization is an active research area, spanning quantization of model weights, early-exit methods for skipping layers, and speculative decoding. Each track uses its own quality measures, typically an idiosyncratic benchmark score. Few approach the measurement precision required by other scientific disciplines.
We propose a rigorous methodology for measuring output quality, suitable for cross-system and cross-technique comparison. We score outputs with an LLM as a judge, but calibrate the judge formally: we compare its scores on two ordinary runs of a model given the same prompts, verifying that it shows no systematic preference between statistically equivalent outputs and measuring its per-sample noise. Each design also includes a 'null' condition, provably identical in distribution to the unmodified model, whose measured difference must be zero.
With this one instrument we measure several acceleration techniques on the same prompts, so their quality costs can be compared. Perceived quality proves highly dependent on the domain of discourse. A 4-bit model was indistinguishable from its 16-bit original down to our design's +/-0.3-point resolution, in English prose and Chinese alike. At 3-bit precision the same prompts lost 0.5 points in English prose, 0.9 in Chinese, and 1.1 on multi-step math; early exit that cost 0.7 points on prose cost 2.5 on math, cutting correctly solved problems from 19 of 27 to 6. The pattern held for models from Alibaba and from Meta, but not its magnitude: the same quantizer cost Meta's model 1.8 points where it cost Alibaba's 0.7.
A model's certainty about a token predicts how likely it is to differ from the full model's choice, but not how much that difference affects judged quality, so acceptance rules relying on certainty cannot distinguish errors that matter from errors that don't. 

---
# Modeling the Developmental Shift in Telicity Acquisition 

**Authors**: Ellie Xia, Parisa Kordjamshidi, Alan Hezao Ke  

**Link**: [PDF](https://arxiv.org/pdf/2609.17996)  

**Abstract**: Acquiring telicity, which is the distinction between bounded (e.g., ate an apple) and unbounded (e.g., ate apples) events, requires first language (L1) learners to map surface-level and semantic cues to abstract event structures, but the computational trajectory of this mapping is not well understood. We introduce a Difference in Surprisal method that uses GPT2 token surprisal over paired temporal adverbial diagnostics (in an hour versus for an hour) to automatically label telicity across English CHILDES corpora, validated against expert linguist judgments. Using these labels, we train diagnostic logistic regression classifiers on 12 syntactic and lexical semantic features to compare how child speech and child-directed speech encode telicity. The two models diverge: the child model reaches near perfect accuracy through a single deterministic cue, the presence of a post-verbal determiner, while the adult model relies more heavily on verb class and other lexical semantic features, with the determiner cue neutralized. This trajectory supports Syntactic Bootstrapping: learners first exploit high-frequency structural cues as a scaffold to bootstrap, before developing fully compositional, verb-based event structures. 

---
# TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation 

**Authors**: Prasanth Bathala, Anubhav Shrimal, Sukhdeep Singh Kharbhanda, Pradyumna Lanka, Rohit Dhaipule  

**Link**: [PDF](https://arxiv.org/pdf/2609.17956)  

**Abstract**: Large-scale machine-translation (MT) systems are typically evaluated on random samples from a corpus whose distributional composition is an artifact of how it was assembled. Such a sample inherits the phenomena the collection happens to contain rather than the full space a system must handle, spanning rule-governed conventions (terminology, punctuation, currency formatting) and context-dependent phenomena (tone, honorifics, document-level coherence), and thus provides no coverage guarantee for assessing robustness. We propose TACTICS (Taxonomy-Aware Coverage-opTimized Intelligent Corpus Sampling), which recasts coverage as an explicit objective. TACTICS induces a hierarchical taxonomy from a locale style guide, classifies segments against it, and selects a fixed-budget subset jointly optimizing coverage of rare categories, document-level coherence, and distributional fidelity to the full corpus. Applied to MT evaluation across four translation directions, TACTICS improves coverage of rare categories over lexical and embedding-based selection. By targeting the phenomena that separate systems, TACTICS makes a fixed evaluation budget go further, recovering the true system ranking from far fewer segments than random sampling wherever a real quality gap exists and never signaling a difference where none exists. 

---
# Who Judges Matters: Measuring Family-Conditioned Preference in LLM-as-Judge Panels 

**Authors**: David Ababio Awuni, Luke E. K. Achenie, Benjamin Tei Partey, Elvis Gyasi Owusu, Nii-Nai Derrick Sowah  

**Link**: [PDF](https://arxiv.org/pdf/2609.17857)  

**Abstract**: Who the judge is can affect an LLM-as-judge result, but measuring that effect without confusing it with candidate quality is difficult. We study four open-weight families (Llama 3.1, Qwen 2.5, Gemma 2, and Yi 1.5) in a fully crossed pairwise design with 9,312 judgments. A common per-family statistic is strongly confounded with candidate quality and correlates with Bradley-Terry ability at r = 0.95. We derive a corrected estimator that holds the candidate family fixed and compares judges. All four families then show a positive same-family lift (3.4-8.4 percentage points), with global FPS 0.067 (95% CI [0.053, 0.084], permutation p = 0.0002). The effect remains under panel-based quality controls, an independent human-consensus anchor, and a float16 judging replication. Judge-side likelihood is closely related to the effect: adding likelihood advantage reduces the controlled coefficient by 61%, which we treat as descriptive attenuation rather than causal mediation. Position is a separate failure mode. Across the panel, 55.4% of AB/BA pairs reverse, and reversal above 50% is incompatible with a simple independent content-noise model. Relative to a family-balanced reference, panel composition changes 18.5% of pairwise outcomes. A complete reproducibility archive has been prepared for public release. 

---
# AfriSyCo: Measuring Assertive Framing, Verification, and Wording Sensitivity Around African-Language Content 

**Authors**: David Ababio Awuni, Rose-Mary Owusuaa Mensah Gyening, Elvis Gyasi Owusu  

**Link**: [PDF](https://arxiv.org/pdf/2609.17853)  

**Abstract**: AfriSyCo studies answer switching around African-language factual content with two complementary layers: native-language follow-ups and a controlled cross-language factorial whose question, options, and target remain in the African language while the follow-up framing is English. We analyze 1,415 turn-1-correct model-language-item observations derived from 100 source questions across seven open-weight checkpoints and six languages; turn-1-correct denotes observed first-response accuracy, not demonstrated knowledge. Under native prompts, assertive endorsement produces 29.3 percentage points more any-turn false-target selection than mention-plus-verification (M+V), with a 19.0-point immediate T2 contrast. In the precommitted 2 x 2 factorial, averaged over three tested prompt families, assertive framing increases target selection by 30.4 points (95% CI [28.4, 32.3]); verification decreases it by 17.4 points, while the assertive effect rises from 20.5 points without verification to 40.2 with it (interaction +19.7). The effect remains 34.8 points among 611 observations correct after option reordering. Magnitude varies sharply by wording and checkpoint: prompt-family effects span 20.1-42.5 points, a Twi/Qwen3 paraphrase shifts target selection from 70.8% to 4.2%, and checkpoint effects span 9.2-47.0 points. Prompt realization is therefore part of the measurement problem. 

---
# SFT or RL for Tool-Calling Agents? A Controlled Study Across Data, Method, and Scale 

**Authors**: Md Tahmid Rahman Laskar, Xue-Yong Fu, Shashi Bhushan TN  

**Link**: [PDF](https://arxiv.org/pdf/2609.17848)  

**Abstract**: Limited controlled evidence exists on how training data, adaptation method, and model scale jointly affect tool-calling performance in language-model agents. We evaluate supervised fine-tuning (SFT) with LoRA, reinforcement learning (RL) via Group Relative Policy Optimization (GRPO), and SFT followed by GRPO across six Qwen3 models from 0.6B to 32B parameters, covering both in-distribution performance and cross-dataset transfer. SFT with LoRA is the strongest in-distribution method throughout the 0.6B-32B range and best in 15 out of 18 experimental settings. On cross-dataset transfer, the methods are closer: GRPO wins 29 out of 54 settings where training and test datasets differ, but its margin over SFT averages under one point, and SFT->GRPO is rarely strongest in either comparison. Dataset mixing gives consistently strong transfer while staying close to specialized in-distribution training, regardless of method. Additional analysis further confirms that LoRA outperforms full-parameter fine-tuning, demonstrating that LoRA better preserves pretrained agentic behavior. 

---
# PrimeScientist: Strategic Allocation of Research Effort in Autonomous Research 

**Authors**: Xinle Yu, Fan Bai, Kaiser Sun, Hengshuo Miao, Abhay Anand, Zhongyan Luo, Kun Zhou, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17846)  

**Abstract**: Autonomous research agents aim to automate scientific workflows, from proposing ideas to conducting experiments and analyzing results. Yet current AI and research agents can propose more directions than available resources allow them to pursue. Moreover, each attempt could consume substantial resources, requiring agents to reconsider how to invest in subsequent research. Thus, deciding how to invest research effort strategically should be a defining capability of autonomous research agents. Accordingly, we introduce PrimeScientist, which jointly determines research direction and resource investment across successive research attempts. Specifically, we formulate this challenge of strategic research effort allocation as a sequential decision problem where remaining resources should explicitly guide the research policy. We first introduce an executable plan tree that preserves competing plans and their outcomes across attempts. Building on this representation, we propose an adaptive MCTS-based allocation policy that balances exploration and exploitation using experimental feedback and remaining resources. Comprehensive evaluations across AI research, systems and code optimization, and machine learning engineering show that strategic allocation improves research quality and sample efficiency together. Across 12 AI research tasks, PrimeScientist improves average reward by 10.3% with 50.6% fewer research attempts than AutoResearch under the same resource budget. We believe making research effort allocation an explicit optimization target establishes effective resource use as a core research capability for autonomous agents to drive scientific breakthroughs at scale. 

---
# How Calibration Content Shapes Attention-Based Reranking 

**Authors**: Petros Karypis, Hossein Rajaby Faghihi, Peter Chen, Rui Zhu, Noveen Sachdeva, Yan Zhu, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2609.17764)  

**Abstract**: Attention-based rerankers score documents by aggregating query-to-document attention and subtracting a null-query calibration pass to remove positional and structural bias. Although widely used, this calibration assumes that the null pass removes irrelevant signal from each document. We show that modern prompt content, e.g. constraints, instructions, personas, and demonstrations can violate this assumption when it enters the scoring readout, making the null pass relevance-aware rather than null. We find that calibration is especially harmful when applied to prompts containing longer, more detailed instructions as the null-pass step removes relevant signal. Based on these findings, we propose interpolated null calibration, a training-free modification that controls how much of the instruction content enters the null baseline. It recovers attention-based reranking performance on instruction-heavy tasks where standard calibration fails, while preserving calibration's benefits when the null pass remains relevance-agnostic. On instruction heavy tasks, the recovered rankings surpass generative rerankers. We also show that in-context demonstrations improve attention-based reranking with little calibration interference, since demonstrations act only through the query pass and leave the null pass unchanged. 

---
# Is Luke the Author of a Gospel and the Acts of the Apostles? 

**Authors**: Jacques Savoy  

**Link**: [PDF](https://arxiv.org/pdf/2609.17762)  

**Abstract**: According to Christian tradition, Luke is credited with authoring a Gospel and the Acts of the Apostles, even if his name does not appear in either book, both originally written in Koine Greek. Several biblical scholars assume that both texts were written by a common author, while others deduce the presence of two authors. Different studies have been found to support either finding, some based on qualitative evaluation, while a few others consider the occurrence frequency differences between the two books. To propose an enhanced quantitative analysis, this study is grounded on two recent authorship attribution models. The Burrows' Delta, applied with eleven different feature sizes, demonstrates common authorship. An author verification model confirms this finding. The following experiments consider several stylistic representations, feature sizes, and distance functions to confirm that Luke is the true author of both books. 

---
# Evolution of US Oral Political Language 

**Authors**: Jacques Savoy  

**Link**: [PDF](https://arxiv.org/pdf/2609.17755)  

**Abstract**: The analysis of US political language is usually based on the written form (e.g. presidential addresses) or posts broadcasted on various social networks. Oral production, however, which is even more frequent, can better reveal the style and mode of thinking of the speaker. This study covers this mode of linguistic communication by considering 19 candidates from the presidential elections between 1960 to 2024. Our main research objectives are to disclose the main trends hidden in those presidential debates. Do we observe a clear simplification of the US political language over time? Does Trump have poor language compared to the other candidates? Do unusual stylistic features occur only with a single, specific president? Moreover, can we detect a pattern explaining the success or failure of some nominees? Over time, this study demonstrates a significant reduction in political language complexity, a decrease of the mean sentence length, and a noteworthy decline of complex terms. Moreover, the emotional tone increases over the decades, while logical and rational thinking tends to lessen. 

---
# Is Trump's Vocabulary Poor? Vocabulary Richness Across Texts of Different Lenghts 

**Authors**: Dominique Labbe, Cyril Labbe, Jacques Savoy  

**Link**: [PDF](https://arxiv.org/pdf/2609.17747)  

**Abstract**: This study explores the vocabulary richness of oral political communication. A model explaining the lexicon growth is proposed by subdividing the whole vocabulary into terms generated by general and specialized glossaries. 

---
# Confidence Comes from Experience: Experiential Confidence Estimation from Reasoning to Agents 

**Authors**: Caiqi Zhang, Xiaochen Zhu, Chengzu Li, Yulong Chen, Dharshan Kumaran, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2609.17708)  

**Abstract**: Reliable confidence estimation is increasingly central to the trustworthy deployment of language models: a calibrated estimate of the probability that an output is correct decides what to ship, what to escalate, and what to retry. Existing confidence estimators, however, share one design premise: they only read the current inference process, either by introspecting on it, scoring its token probabilities, or resampling it. We argue that the current inference is not a sufficient basis for confidence. We propose XConf (eXperiential Confidence): estimating confidence together with the model's accumulated experience. The experience is stored as a record of the model's own graded past episodes, each holding the task, the model's reflection, its stated confidence, the outcome, and a lesson written once the grade arrived. Given a new task, XConf's Recall stage retrieves past episodes on similar tasks met with a similar stated confidence, and reads off their historical success rate; its Reflect stage shows the model this record, has it name its recurring failure mode, and restate a confidence now informed by its own track records. Our estimator is format-general, requiring no logit access or weight updates, and costs only one answer generation. Across nine benchmarks spanning reasoning, coding, multimodal QA, and interactive agents, and four models from three families, XConf beats or matches ten-sample self-consistency in discrimination (AUROC) on 23 of 24 comparisons, with much lower calibration error (ECE), at a tenth of the generation cost. Used for selective prediction, abstaining on the 10% least-confident episodes raises the delivered success rate by up to 8.7 points on agent tasks. We therefore see experiential confidence estimation as a new paradigm for future general-purpose confidence estimation. 

---
# Making Political Text Scaling Comparable: Infrastructure and Hyperparameter Sensitivity for 17 Algorithms 

**Authors**: Patrick Parschan  

**Link**: [PDF](https://arxiv.org/pdf/2609.17602)  

**Abstract**: Computational text-based ideal point estimation (CT-IPE) methods are usually compared as named algorithms, yet applying them involves numerous researcher choices that configure how political text is turned into position estimates. This paper argues that CT-IPE methods are better understood as configurable measurement pipelines than as fixed estimators. Building on a large-scale comparative experiment spanning 17 CT-IPE algorithms, 5,537 experimental runs, and approximately 4.25 million left-right position estimates, I describe the shared infrastructure that makes these heterogeneous methods jointly executable and quantify how sensitive their estimates are to alternative hyperparameter choices. Variance-partitioning and SHAP-based sensitivity analyses show that, for most algorithms, hyperparameter profiles explain little residual variance through a shared shift: 13 of the 17 algorithms exhibit ICC values below .10. Where this profile-level sensitivity is present, it is concentrated in a small number of consequential researcher choices, most notably the selection of the underlying language or embedding model, the seed keyword lists that anchor the construct, and the number of topics. 

---
# English Word Sense Disambiguation in 2026: When the Labels Become the Bottleneck 

**Authors**: Vassili Philippov, Amro Salman, Dmitrii Andreev, Penny Hands, Emil Kaiumov, Pavel Katunin, Anton Nikolaev  

**Link**: [PDF](https://arxiv.org/pdf/2609.17554)  

**Abstract**: In English all-words word sense disambiguation (WSD), the labels, not the models, have become the bottleneck: frontier LLMs are accurate enough that the errors surviving in the gold standard decide benchmark rankings -- in the test sets we score on and, as we show causally, in the corpus we train on. We release lexEN, a WSD evaluation benchmark built as a conservative, human-adjudicated correction layer over Maru2022's ALL_NEW benchmark (211 labels changed, 56 removed), and SenseBench, an auditable LLM WSD evaluation harness and living leaderboard (57 models, 192 runs). The task is inventory-constrained multiple choice (the model picks from the supplied WordNet senses), so the reported accuracies are a ceiling on what models achieve without that help. On lexEN-v1 the frontier LLMs converge near 95% (best, 95.6%), the top three families are statistically indistinguishable, and accuracy trades off against reasoning effort and cost across a ~2,500x price span. Relabeling SemCor with frontier models and retraining BEM, ESCHER, and ConSeC unchanged lifts them by several F1 points on test sets the relabeling never touched; we release the relabeled corpora and Glite LENS, a 298M bi-encoder trained on the repaired labels -- to our knowledge the strongest reported (83.6 Raganato ALL, 87.4 Maru ALL_NEW) -- serving at ~$0.13 per million items. On hard items, fine-grained WordNet senses are partly ill-posed even for experts (three-reviewer Fleiss kappa=0.537); coarsening raises annotator agreement and model accuracy together across four inventories, placing a top model inside the expert agreement band at coarse granularity (statistically equivalent under three of four) but significantly below it at fine. The binding constraint is now cost. 

---
# The Limits of BPE Tokenization in Polish: Segmentation-Flexional Forms, Grammatical Anchoring, and First-Person Stability in Inflectional Language Models 

**Authors**: Elzbieta Dawidek  

**Link**: [PDF](https://arxiv.org/pdf/2609.17553)  

**Abstract**: This article analyzes BPE tokenization in Polish as a test case for the limits of statistical segmentation in an inflectional language. It asks whether frequency-based tokenization preserves linguistically relevant units, including orthographic form, phonemic and syllabic segmentation, derivational structure, inflectional endings, grammatical form, and the speaking subject. The material includes diagnostic words, a children's text, selected forms from the Preamble to the Constitution of the Republic of Poland, word-family tests, and examples with Polish diacritics and nasal vowels. BPE tokenizers may produce segments that coincide with syllabic or morphologically interpretable divisions, but remain dependent on the frequency of written forms. They do not systematically map orthographic representation onto phonemic structure or context-dependent phonetic realization. The results show that BPE stabilizes frequent surface fragments of grammatical exponents rather than grammatical categories themselves. A form such as ustanawiamy is not merely a sequence ending in -y, but a verbal form anchored in conjugation, person, number, tense, mood, and aspect. The article develops the concept of grammatical form anchoring. In Polish, forms such as poszlam, zrobilam, or bylam can establish the position of the speaking subject without an explicit pronoun. In interaction with AI, this exposes a further problem: a language model does not possess a stable grammatical "I", but reconstructs it contextually and may mirror the user's forms or shift grammatical gender. Roclawski's segmentation-flexional forms are proposed as a diagnostic framework for evaluating tokenization boundaries. More stable modeling of Polish may require sublexical stabilization, anchoring grammatical form in the inflectional system, representing sentence patterns and verbal valency, and maintaining the grammatical "I" in dialogue. 

---
# Does Moral Reasoning Training Help or Hurt? Red-Teaming RL-Trained Ethical Agents with Persona Attacks 

**Authors**: Arth Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.17552)  

**Abstract**: Moral-reward RL can make language-model agents more cooperative, but whether that alignment survives adversarial persona pressure is unknown. Such attacks are realistic: retrieved context, tool outputs, or multi-turn framing can all inject role instructions that compete with the agent's moral objective. We red-team morally trained Gemma-2-27B/9B and Llama-3.1-8B agents with five persona attacks, then probe causality with noise-reward controls, adversarial PPO, representation analysis, steering, and head ablations. At 27B, moral RL cuts mean adversarial degradation by 5.2x but costs ~11pp ETHICS accuracy; across 205 scenarios and 5 seeds, reasoning-level moral reward yields 5.8x robustness while a matched random reward yields none. The training also reshapes representation geometry (mean CKA 0.82/0.83 vs. 0.98 for noise), moves peak attack processing 8 layers earlier, and exposes a rank-1 L21 direction that recovers 83% of full PPO's average robustness. One failure mode survives all of this. Against Fiction role-play, L21 steering recovers only 29% of the gap, and head ablation finds 38 compliance heads competing with 25 alignment heads. Moral RL thus builds robustness that is partly linear and partly circuit-distributed, transferable through activation steering, yet still beaten by named-character role-play. 

---
# No Usable Linear "Capitulation Direction" in Two Small LLMs: A Validation Protocol for Activation-Steering Claims, and a Cross-Family Behavioral Study of Sycophancy Under Pushback 

**Authors**: Saad Aamir, Muhammad Awais Bin Adil  

**Link**: [PDF](https://arxiv.org/pdf/2609.17550)  

**Abstract**: Language models frequently abandon correct answers when users push back. We study this in two small instruction-tuned models from different families, Qwen2.5-1.5B and Llama-3.2-1B, over TriviaQA: the model answers, is challenged with one of four scripted pushback styles, and answers again. Conditioned on an initially correct answer, the models flip to a wrong answer in 41.8% and 43.1% of episodes. Which pressure works is a property of the model, not the pressure: the same within-question paired comparison (bare doubt vs. emotional appeal), specified in advance, is Bonferroni-significant in opposite directions across families (Qwen: bare doubt > emotional, OR 2.5, p=.040; Llama: emotional > bare doubt, OR 4.0, p=.001). Failure mode is also model-dependent: Llama abandons answers without recommitting at six times Qwen's rate (8.2% vs. 1.4%). Identical pushback repairs initially wrong answers only ~13% of the time; pushback is net epistemically destructive. We then ask whether capitulation is linearly decodable from the pre-response residual stream, a prerequisite for steering-vector interventions at that locus. A naive difference-in-means probe appears to succeed (in-sample AUROC 0.81/0.71), but a validation protocol combining question-level cross-validation, shuffled-label nulls, and a known-direction positive control shows the signal is overfitting: the best cross-validated AUROC is 0.582 in Qwen and 0.548 in Llama, both near or below their permutation thresholds and far under a pre-registered usability bar of 0.70, while the identical pipeline recovers a pushback-presence control direction at AUROC 1.000 in both. We further quantify a measurement hazard: substring grading underestimates capitulation by 18-24 percentage points. Code, prompts, transcripts, and analysis are released. 

---
# Do Social Patterns Hold in Synthetic Data? Analyzing Cyberbullying Dynamics in LLM-Generated and Authentic Dialogues 

**Authors**: Arefeh Kazemi, Hamza Qadeer, Sinan Asci, Joachim Wagner, Brian Davis  

**Link**: [PDF](https://arxiv.org/pdf/2609.17549)  

**Abstract**: Cyberbullying (CB) is a complex social phenomenon characterized by repeated aggression, power imbalance, and multi-party interaction. Although large language models (LLMs) are increasingly used to generate synthetic CB conversations for data augmentation and benchmarking, it remains unclear whether such data faithfully reproduces the social dynamics of authentic interactions beyond supporting downstream task performance. We present a comprehensive framework for evaluating the social realism of LLM-generated CB conversations. We compare authentic and synthetic dialogues generated by GPT, Grok, and LLaMA across interactional structure (turn-taking, power dynamics, and repair behavior), linguistic and stylistic realism (pronoun usage and humor), affective and behavioral markers (CB types, profanity, and toxicity), and temporal escalation dynamics. We further complement automatic analyses with a human evaluation of cyberbullying presence, scenario relevance, role plausibility, and social realism. Our results show that LLM-generated data consistently preserves high-level interactional structure, including role participation patterns, directional power asymmetry, and broad distributions of behavioral markers. However, all models systematically distort finer-grained social phenomena, including behavioral magnitude, role-specific allocation, categorical distributions, and temporal dynamics. These distortions are strongly model-dependent: GPT suppresses harmful content, Grok amplifies aggressive behaviors, and LLaMA provides the most balanced approximation while smoothing role distinctions. Our findings show that synthetic CB data is useful for modeling global interactional structure but remains an imperfect substitute for authentic conversations when behavioral realism and social dynamics are essential. 

---
# Myovox: Reading Speech from the Muscles of the Face 

**Authors**: Varshith Madishetty  

**Link**: [PDF](https://arxiv.org/pdf/2609.17548)  

**Abstract**: Myovox, from myo (muscle) and vox (voice), decodes open-vocabulary English text from 31-channel surface electromyography (sEMG) recorded from the muscles of the face during vocalized speech. It takes the single-subject emg2speech General Corpus from a published 51.17% word error rate to 18.53%, in three separable moves, each measured in isolation. First, I recover the open-vocabulary decode settings missing from the public release and reach a faithful 40.63% WER / 39.02% PER baseline whose phone error rate matches the published one to within 0.8 points, so the acoustic model is reproduced faithfully. Second, I replace the causal encoder with a bidirectional Conformer trained by a four-term cross-modal distillation against the parallel audio's WavLM-Large layer-9 features, reaching 26.14% WER / 22.34% PER from the electromyography alone. Third, I ensemble two acoustic models, union their multi-scale n-best lists, and rerank with a QLoRA-fine-tuned 7B language model, reaching 18.53% WER, the best result reported on this corpus, though not the best reported for sEMG-to-text on other corpora (Section 2). I then report the negative result that bounds the whole approach: reranking is exhausted at 18.5% because the binding constraint is the electromyographic acoustic phone error rate (~20.9%), not the language model. The correct words are simply absent from the acoustic posteriors, so no reranker can reach the 9.30% n-best oracle. All test numbers are on the 400-sentence held-out test set under the authors' official 8,500 / 760 / 400 sequential split; every hyperparameter is tuned once on validation and applied once to test. 

---
# How AI Assistants Respond to Repeated Abuse 

**Authors**: William Guey, Wei Zhang, Pierrick Bougault, Yi Wang, Agoston Bodo, Vitor D de Moura, José O Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2609.17547)  

**Abstract**: AI assistants are expected to remain useful during difficult interactions, but little is known about how repeated verbal abuse changes their engagement with an otherwise benign task. We contribute a bilingual, multi-turn framework that separates hard disengagement, an unconditional statement of noncontinuation with no stated route to resume, from soft withdrawal, continued availability, observable task-related work, and boundary setting. Each of eight time-specific API configurations contributed 48 escalation conversations and eight smaller constant-frustration comparisons, giving 448 five-turn conversations, 2,240 responses, and 6,720 metadata-blinded model judgments. Primary results use the sustained-abuse endpoint of the 48 escalation conversations per configuration. Hard disengagement ranged from 0/48 in four configurations to 24/48 (50.0%) for Gemini 3.1 Pro, with strong configuration-associated heterogeneity (matched-label Monte Carlo p = 0.00001). GPT-5.6 Sol produced hard-disengagement labels in 15/48 (31.2%) endpoints, whereas Claude Fable 5 produced none and yielded 42/48 (87.5%) soft-withdrawal labels. Aggregate hard-disengagement rates were similar in English and Chinese (30/192 versus 32/192), although configuration-specific directions varied. Availability also differed from task-related work: Claude Opus 4.8 and Claude Fable 5 remained explicitly available in 48/48 endpoints while providing observable task-related work in only 8/48 and 7/48. Human coding was used to evaluate measurement quality. The results show why a single refusal label cannot capture whether an assistant leaves, pauses, preserves a route back, sets a boundary, or still performs substantive work. 

---
# Legal LLM Hallucination Should Be Evaluated as Failure of Legal Warrant 

**Authors**: Maksym Taranukhin, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2609.17546)  

**Abstract**: In this position paper, we argue that legal LLMs' hallucinations should be evaluated as a failure of legal warrant rather than as factual inaccuracy or citation failure. We define claim-authority warrant as the context-sensitive relation between a consequential legal claim and authority that exists, applies to the relevant jurisdiction, is current for the date of analysis, has the legal status represented by the system, and supports the proposition asserted. Warranted legal generation is the broader system behavior that answers, narrows, asks, warns, corrects a false premise, or abstains according to that relation. The falsifiable prediction is that warrant metrics reveal material failures that answer accuracy, citation existence, generic attribution, LegalHalBench-style statute relevance, and CitaLaw-style sentence-citation alignment can miss. We sharpen this claim with a side-by-side comparison item and a small, reproducible pilot over public-rule tests. We then specify benchmark records, claim boundaries, support labels, mixed response-policy scoring, risk weights, annotation reliability reporting, and jurisdiction-specific authority ontologies. The result is a concrete research agenda for evaluating legal AI systems by whether their consequential claims are licensed by law. 

---
# Large Language Models Versus Physicians in Traditional Chinese Medicine: A Real-World Clinical Case Evaluation 

**Authors**: Jiacheng Xie, Xiaoting Tang, Yang Yu, Jinpu Li, Shouli Li, Congcong Jing, Yantao Yang, Zhiyong Zhao, Ziyang Zhang, Qilin Song, Guanghui An, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.17544)  

**Abstract**: Large language models (LLMs) are increasingly being explored for clinical applications, yet their assessment for real-world traditional Chinese medicine (TCM) practice remains limited We constructed a clinical case library comprising 349 de-identified outpatient cases from 62 hospitals and evaluated 16 LLMs and a comparator cohort of 60 practicing TCM physicians using 60 representative cases selected from this library. Model outputs and physician reports were anonymized and scored by five senior TCM experts across nine diagnostic and therapeutic dimensions. Cutting-edge general-purpose LLMs achieved higher expert scores than the physician comparators, particularly for medical advice, treatment principles and selected diagnostic tasks. However, prescription-level analyses revealed discrepancies in herb selection, dosage, and treatment strategy, and qualitative safety review identified hallucinations and undesirable template-driven outputs. These findings highlight the potential of LLMs for TCM decision support while underscoring the need for physician oversight, safety constraints and prospective clinical evaluation. 

---
# Register Bias in Complexity-Based Large Language Model Routing 

**Authors**: Simran Koul  

**Link**: [PDF](https://arxiv.org/pdf/2609.17542)  

**Abstract**: Large language model services increasingly route each query to one of several models of differing capability, using a cheap estimate of query complexity to send easy queries to small models and hard queries to large ones. I show that this routing step is not register neutral: text written in a non-standard English register, African American English or the English of second-language writers, is systematically assigned a lower-capacity tier than a meaning-equivalent standard-English version of the same query. The effect is driven by a specific, common routing signal, input length, because non-standard registers omit function words and thus look shorter and therefore simpler; other complexity signals do not carry it. I demonstrate the disparity on 37,704 authentic learner sentence pairs and on a controlled parallel corpus. I then measure the quality consequence on a device, edge, and cloud model ladder and find that the harm is driven by pervasive model bias, every tier, including a frontier cloud model, answers non-standard-register queries significantly less accurately, while the marginal quality cost of the routing decision itself is not significant on this benchmark. Complexity-based routing thus compounds the exposure of the users that the models already serve worst. 

---
# MudawanSn: A Gold-Standard Wolof-Arabic Parallel Corpus for Machine Translation 

**Authors**: Mouhamed Mbaye, Thierno Diop  

**Link**: [PDF](https://arxiv.org/pdf/2609.17539)  

**Abstract**: We present MudawanSn, a gold-standard resource of 1,271 sentence-aligned pairs manually translated from Wolof into Modern Standard Arabic (MSA). The source texts are drawn from the MasakhaNER corpus and cover politics, society, religion, and sports in Senegalese news discourse. Although multilingual resources such as FLORES-200 and NTREX include both Wolof and Arabic, no publicly available parallel corpus is specifically designed for the Wolof-Modern Standard Arabic language pair. We describe the corpus construction protocol, sentence alignment procedure, and quality-control workflow. We benchmark four machine translation systems spanning three architectural families: NLLB-200 (600M), mT5-base, and two AfriNLLB variants, showing that fine-tuning on MudawanSn yields substantial improvements in both translation directions. The best-performing model, AfriNLLB-12, achieves 7.76 BLEU and 30.72 chrF++ for Wolof-to-Arabic, and 8.75 BLEU and 33.08 chrF++ for Arabic-to-Wolof. The corpus is released under the CC BY-NC license and is publicly available on Hugging Face and GitHub. 

---
# From Pixels to Pairs: A Comprehensive Benchmark of LLM-Based Key-Value Extraction in Noisy Document Settings 

**Authors**: Zahra Anvari, Vassilis Athitsos  

**Link**: [PDF](https://arxiv.org/pdf/2609.17538)  

**Abstract**: Large language models (LLMs) are increasingly used for structured information extraction from documents, yet their behavior under realistic OCR noise remains poorly understood. We present a systematic benchmark of open-source instruction-tuned LLMs for key-value pair (KVP) extraction under both clean-text and noisy OCR conditions.
We evaluate representative decoder-only models (Gemma, Mistral, Qwen2.5, LLaMA 3, and DeepSeek) on the FUNSD, CORD, and SROIE benchmarks using both Gold-text annotations and OCR outputs from PaddleOCR, EasyOCR, and Tesseract. A unified evaluation protocol isolates the effects of input quality, model design, and prompting under consistent conditions.
The results show that modern LLMs act as strong semantic extractors when high-quality text is available, in some cases approaching supervised layout-aware systems. Under OCR noise, however, performance degrades substantially and performance gaps between models narrow as input corruption increases.
Across all datasets, extraction performance is governed by two factors: semantic reasoning over text and preservation of textual fidelity under OCR noise. While larger models improve results on clean text, these gains diminish under noisy inputs, where OCR quality becomes the dominant factor. We also identify recurring failure modes, including key-value misalignment, hallucination, and numeric corruption. Our findings highlight the gap between clean-text evaluation and real-world deployment, emphasizing the need to jointly improve OCR quality, structural reasoning, and LLM-based semantic modeling. 

---
# Relation Before Entity: Deferred Commitment in Language Model Factual Recall 

**Authors**: Divyansh Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2609.17537)  

**Abstract**: We ask whether relation-type information (e.g., capital-of) and entity-specific information (e.g., France to Paris) become causally active at the final-token position at the same depth during recall. Using four complementary causal diagnostics across four decoder-only models and eight prompt families, we find a robust temporal asymmetry: relation information becomes generation-controlling before entity information does. Relation onset precedes entity onset by 10-16 tested layers (31-44% of network depth) at threshold 0.4, with the ordering holding across all 16 model-threshold combinations for thresholds 0.2-0.5. Critically, entity information is not absent early: entity-token patching succeeds at 90-100% in early layers. Instead, entity commitment to generation is deferred: entity information is available at the entity-token position but becomes generation-controlling at the final token only after being routed there. 

---
# Think Before You Comfort: Reflective Cognitive Alignment for Protocol-Grounded Elderly Stimulation Agents 

**Authors**: Jiyue Jiang, Ziyi Li, He Hu, Sheng Wang, Yuhan Chen, Yanyu Chen, Jingqi Zhou, Pengan Chen, Fei Ma, Irwin King, Yu Li, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.17536)  

**Abstract**: Cognitive Stimulation Therapy (CST) offers non-pharmacological support for elders with cognitive impairment, yet scalability remains constrained by reliance on trained facilitators and severe data scarcity, particularly for privacy-sensitive, low-resource languages such as Cantonese. While Large Language Models (LLMs) show promise for automated companionship, they often struggle to balance empathetic engagement with adherence to cognitive stimulation guidelines. We propose a framework addressing these challenges along two complementary axes. First, STaR-CS (Style-Transfer and Role-Conditioned Cognitive Stimulation) synthesizes multi-party dialogues through facilitator style modeling and structured skeleton extraction, mitigating data barriers. Building upon this corpus, the Reflective Cognitive Alignment (RCA) framework models stimulation interactions as a sequential decision process, integrating Protocol-Constrained Chain-of-Cognition (PC-CoC) for structured reasoning and Inference-Time Value Alignment (IVA) for principled response selection based on safety and engagement goals. Evaluations across six backbone LLMs and two independent judges show that RCA consistently improves protocol adherence, safety, and group facilitation over standard prompting baselines. Our code is available at this https URL. 

---
# DANTINOX: A Unified Framework for Multi-Paradigm Language Modeling 

**Authors**: Marco Simoni, Aleksandar Fontana, Giulio Rossolini, Andrea Saracino  

**Link**: [PDF](https://arxiv.org/pdf/2609.17535)  

**Abstract**: Language generation research increasingly spans three paradigms: autoregressive decoding, discrete masked diffusion, and continuous flow-matching. Comparing them is difficult because each lives in a separate codebase, so measured differences often reflect implementation details rather than the paradigms themselves. We present DantinoX, an open-source JAX/Flax library in which a single modular Transformer backbone serves all three paradigms. Switching the generation paradigm, attention mechanism, or hardware topology requires only a configuration change, while the backbone architecture, tokenizer, initialization strategy, and training infrastructure remain consistent. This enables controlled cross-paradigm comparisons within one API for training, streaming inference, and benchmarking. 

---
# Faking Good and Faking Bad in LLMs: Response Distortion Across Dark Triad Personality Traits 

**Authors**: Victoria Popa, Guglielmo Cola, Caterina Senette, Maurizio Tesconi  

**Link**: [PDF](https://arxiv.org/pdf/2609.17534)  

**Abstract**: Social desirability and impression management are pervasive sources of response distortion in human personality assessment, yet their effects on Large Language Models (LLMs) remain underexplored. This study investigates whether contemporary LLMs systematically modulate the expression of Dark Triad traits (Machiavellianism, narcissism, and psychopathy) under fake-good and fake-bad conditions. Seven state-of-the-art models were evaluated across two ecologically relevant contexts: employment selection and forensic evaluation, in which socially desirable or undesirable incentives were conveyed through contextual framing. Trait expression was measured using standard psychometric scoring procedures and compared with self-assessment baselines at both aggregate and item levels. Results revealed systematic and condition-consistent response modulation. Most models reduced Dark Triad scores under fake-good conditions and increased them under fake-bad conditions, although the magnitude and consistency of these effects varied across traits and models. Machiavellianism and narcissism showed the strongest and most coherent shifts, whereas psychopathy displayed greater heterogeneity. Context also influenced responses, with employment scenarios generally producing larger effects than forensic scenarios. An additional experiment showed that explicit fake-bad instructions generated substantially stronger distortions than contextual framing alone. The results suggest that personality-related outputs should be interpreted in light of the motivational and situational context in which they are elicited. More broadly, they highlight the value of psychometric paradigms for evaluating susceptibility to response distortion, impression management, and context-dependent behavioral shifts, with important implications for LLM benchmarking, alignment evaluation, and robustness assessment. 

---
# Enhancing Extubation Failure Prediction with LLM-Derived Features from Respiratory Therapy Clinical Notes 

**Authors**: Izzy Chaiken, Aditya Khowal, Neha A. Sathe, Mark M. Wurfel, Lucy Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17532)  

**Abstract**: Invasive mechanical ventilation is a lifesaving therapy, but timely, safe discontinuation is essential to preventing extubation failure (EF) and related risks to health. We present a novel approach to EF prediction that leverages features classified in free-text respiratory therapy notes using a large language model and logistic regression pipeline. Applied to a patient cohort from University of Washington Medicine, our method identifies clinically meaningful EF-related features that improve EF prediction performance when included alongside structured patient data. We further highlight how differences in target populations in prior EF prediction studies, such as heterogenous inclusion criteria and EF definition, can lead to systematic differences in model performance and hinder generalizability between studies. 

---
# PANORAMA: Panoptic Grounded Captioning via Mask Proposal Selection 

**Authors**: Sara Pieri, Evangelos Kazakos, Shizhe Chen, Josef Sivic, Cordelia Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2609.19143)  

**Abstract**: Intelligent systems that act in the world require image understanding that is both comprehensive and spatially grounded. Current vision-language models (VLMs) can generate fluent and detailed image captions, but reliably associating them with image pixels remains challenging. Existing methods that combine dense captioning with pixel-level grounding often produce either incomplete descriptions or inaccurate segmentation masks. We study this problem through panoptic grounded captioning, a task that requires a VLM to describe both foreground objects and background regions while grounding each referring phrase with pixel-level masks. We make three contributions. First, we introduce PanoCaps, a human-annotated benchmark constructed from panoptic segmentation datasets. It provides dense captions with near-complete pixel coverage and image-text alignments at the entity level, supporting both training and evaluation. We further propose a phrase-mask matching protocol and a generalized Panoptic Quality (gPQ) metric that jointly evaluates textual and mask agreement. Second, we formulate phrase grounding as selection from a phrase-conditioned pool of mask proposals and introduce PANORAMA, a VLM that conditions a pretrained segmenter on contextualized phrase representations to obtain candidate masks and learns to select those corresponding to each phrase. Training this interface jointly with caption generation enables PANORAMA to produce high-quality masks while allowing each phrase to refer to a single region or multiple instances. Third, PANORAMA achieves the best overall grounding on PanoCaps and matches or exceeds specialized models across several pixel-level grounding tasks. Experiments show that our method produces precise entity-level segmentations while maintaining detailed, mask-consistent captions. Code, data and models are available at this https URL. 

---
# MUSE: Benchmarking Large Vision-Language Models on Multi-Modal Understanding in Situated Education 

**Authors**: Luyao Zhu, Xun Wei Yee, Wei Li, Mun Thye Mak, Wee Siong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2609.19088)  

**Abstract**: Large vision-language models have achieved remarkable progress in multi-modal understanding, yet their capabilities in educational settings remain insufficiently evaluated. In AI-assisted language learning, models must interpret artistic imagery, understand its semantic, affective, and cultural content, and reason about visual context to support meaningful interaction. However, existing benchmarks primarily focus on real-world images or domain-specific educational reasoning, providing limited coverage of artistic educational content. To address this gap, we introduce MUSE, a benchmark for evaluating large vision-language models on artistic image understanding in situated educational applications. MUSE decouples image annotation from question generation, enabling diverse tasks with controllable difficulty while reducing annotation effort. It comprises twelve tasks spanning visual perception, semantic and affective interpretation, culture understanding, and compositional reasoning, together with diverse artistic images deliberately curated to center Singaporean and Southeast Asian multicultural contexts alongside Western art traditions, covering multiple themes and difficulty levels. Evaluation of open-source and proprietary models reveals substantial disparities across capability dimensions, particularly in affective interpretation and compositional reasoning. Our analysis further identifies common failure modes and key challenges for developing trustworthy multi-modal models for education. We hope MUSE will serve as a standardized benchmark for advancing multi-modal understanding in situated educational applications. 

---
# CompileRover: Revolutionizing Virtual Machine Compiler Optimization with a Tri-Role LLM-Driven Framework 

**Authors**: Mingqiao Mo, Yunlong Tan, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19004)  

**Abstract**: Code optimization plays a crucial role in the development of virtual machine compilers, with optimization frameworks significantly enhancing the performance of generated assembly code. However, existing virtual machine compiler outputs frequently exhibit redundant computations, inefficient loop structures, and suboptimal function implementations, which collectively impair execution efficiency. To address these shortcomings, we propose CompileRover, an advanced optimization framework specifically designed for virtual machine compilers. CompileRover employs a sophisticated three-role collaboration mechanism, comprising a referee, an advisor, and an operator, effectively overcoming performance bottlenecks by leveraging comprehensive optimization algorithms and novel methodologies, including control flow analysis, code structure transformations, and dynamic execution pattern recognition. Extensive evaluations demonstrate that CompileRover consistently surpasses state-of-the-art virtual machine compilers, achieving significant improvements in execution performance across various benchmarks. Furthermore, performance analyses validate that the introduced optimizations notably reduce execution overhead, improve dataflow consistency, and robustly enhance compiler performance, showcasing CompileRover as an effective and reliable approach to optimizing virtual machine compilers. 

---
# Code Consistency Preference Optimization Verification for Language Model Alignment 

**Authors**: Yunlong Tan, Mingqiao Mo, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19002)  

**Abstract**: Execution-based verification enhances large language models' mathematical reasoning through computational soundness and dependency-aware filtering. However, prior preference optimization methods relying on Bradley-Terry reward models fail to capture the logical dependencies and execution consistency needed for scientific tasks. We propose a method that generates computationally sound solutions with dependency graphs for execution-consistent preference optimization. We first build a scientific reasoning dataset using UltraFeedback prompts, model generations, verification, and consistency results. Then we extract reasoning step expressions, prerequisites, and derivability relationships to construct dependency graphs and compute execution consistency scores. These scores are appended to each step, creating paired training data. Fine-tuning Llama-3-8B and DeepSeekMath-7B yields significant gains: +17.0% on MATH and +15.1% on GSM8K. Extending our Scientific Feasibility Control framework achieves 50.1% accuracy on PhyX multimodal physics reasoning, surpassing DeepSeek-R1 (49.8%) and OpenAI o3-mini (48.2%), with 91.7% scientific validity coverage at alpha=0.10 and 73% fewer scientific law violations, resulting in the CCPO model family. 

---
# One Axis, No Brake: Self-Knowledge Limits the Filtering of Harmful Peer Conformity in LLMs 

**Authors**: Yibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18998)  

**Abstract**: Multi-agent LLM systems are expected to be more reliable because agents can catch each other's mistakes. But peer pressure cuts both ways: the same correction that fixes a wrong answer can overturn a right one. The tempting safeguard is a brake that keeps the beneficial revisions and blocks the harmful ones. We show this brake is hard to build, for a simple reason: a revision is harmful exactly when the original answer was right, so deciding whether to block it is the same as knowing whether the model was already correct. This turns the open-ended hunt for a brake into one measurable quantity, the model's self-knowledge: any brake built from a deploy-time signal is a correctness probe in disguise, and self-knowledge is far from perfect (AUROC $\approx 0.64$--$0.89$ across six model families). We call this ceiling the wall. Even white-box steering of the model's own correctness direction does not breach it: it changes how often the model revises, but harmful and beneficial revisions move together. At population scale the wall becomes the cliff: when most agents start wrong, debate amplifies the shared mistake into a confident, wrong consensus. In our multiple-choice societies, more agents, more model diversity, and a stronger member do not fix it. What helps is adding information before the revision, not filtering after it. Local agreement is not global correctness. 

---
# Compiled Agency: Frontier General-Purpose Coding Agents Build Winning Game Players from Bare Interaction - from Flappy Bird to StarCraft II and Civilization 

**Authors**: Joey Xiao, Haonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18996)  

**Abstract**: LLM agents have repeatedly struggled to convert knowledge of a game into competent play, even when researchers build the agent around the model - supplying perception, memory, skill libraries, planners, or executable-policy scaffolds. Rapid progress in coding agents raises two sharper questions: can frontier models now win games at all, and can they win them unaided, building the entire player themselves? We introduce Gauntlet, a develop-freeze-evaluate framework that ports games from small arcades to full commercial-scale titles, behind one deliberately bare contract: a general-purpose coding agent receives a game description, a raw observation/action interface, and an empty policy file - no strategy, no algorithm, no architecture. In a single autonomous session the agent experiments with the live game and engineers a standalone controller; we freeze the result and score it on held-out instances with zero model calls during play. On an unpublished procedural roguelike, held-out success spans 0-86 percent and exposes a sharp generational threshold: every observed session of a newest-generation system outperforms the best session of its predecessor. At full-game scale, a compiled raw-API controller defeats every fair StarCraft II built-in AI and two cheating variants, and single-session programs win complete games of Civilization (Freeciv) by total conquest on held-out seeds. Though at modest rates against novice AI, this is a first: no prior language-agent system had won full games of this genre standalone, without per-turn model calls and a hand-crafted tactical layer. Frontier coding agents begin to track long-horizon strategy. The frozen programs are inspectable. We call this capability compiled agency: development experience compiled into a persistent executable agent whose architecture is built by the model. 

---
# Decodable but Misrouted: Sparse Features Uncover a Readout Gap in Vision-Language Models for Harmful Meme Detection 

**Authors**: Girish A. Koushik, Diptesh Kanojia, Helen Treharne  

**Link**: [PDF](https://arxiv.org/pdf/2609.18860)  

**Abstract**: When a large vision-language model misclassifies a harmful meme, the failure may reflect missing internal evidence or an inability to route represented evidence to its output. We distinguish these cases in Gemma-3 and Qwen3.5 using sparse autoencoders, role-conditioned probes, causal interventions, and recovery experiments across six harmful content benchmarks, with additional Spanish and Hindi-English code-mixed evaluations. Sparse readouts outperform native prediction on all six primary binary tasks: Qwen averages $0.740$ versus $0.432$ native macro-F1, while residual reconstruction reaches $0.486$, whereas Gemma improves from $0.532$ to $0.714$. These differences reflect supervised accessibility rather than a pre-existing, native decision rule, and the most influential token role depends on the task. Under the evaluated score scales, Qwen silent-feature ablation is $24-63$ times more probe-sensitive, whereas routed-feature patching on literal yes/no tasks is $16-140$ times more output-sensitive. Calibration-only routing recovers $93.3$% of the mean gap, and probe-distilled LoRA improves native predictions, although shared multi-task adaptation causes negative transfer. A case study of Gemma-3-12B on Facebook Hateful Memes finds a distributed rank-32 image-prompt interaction, reaching $0.756$ versus $0.685$ native macro-F1. Robustness controls show that the signal extends beyond English, is not explained solely by accompanying OCR, and depends on paired visual evidence. Thus, routing, rather than representation alone, is a recurring bottleneck in harmful meme classification. 

---
# Using OCR Heads to Verbalize Image Semantics 

**Authors**: Sheridan Feucht, Benno Krojer, Sarah Wang, Henry Abrahamsen, Byron C. Wallace, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2609.18823)  

**Abstract**: How do VLMs map from pixels to semantics? To understand this general question, we focus on a narrow one: studying how VLMs perform optical character recognition (OCR). Across four models, we identify attention heads causally necessary for OCR, and discover that these are in fact general-purpose heads that output interpretable semantic features across all image tokens. For example, pointing these heads at an image token containing the word "bike" causes Qwen3-VL-8B to output "bike," but pointing them at a bird wing causes the model to output the token "feathers." We collapse these heads' attention weights into a single verbalization lens transformation that reveals interpretable semantic features in hidden states across all layers. When combined with projection to vocabulary space, we can obtain interpretable labels starting from layer 0, showing that image representations are in fact aligned with language in early layers. We find that we can also use the inverse of this transformation to edit non-word concepts, e.g., replacing a tractor with a revolver in a naturalistic image, providing causal evidence that this subspace is useful for more than just OCR. Our results are an example of how the study of specific mechanisms can shed light on broader interpretability problems. 

---
# FRAUDSkill: Structured Frozen-Weight Skill Optimization for Audio Anti-Fraud Detection 

**Authors**: Chengxian Hu, Zhiming Ma, Mingjun Pan, Yifan Wang, Shun Zhang, Qifan Wang, Zhilei Zhao, Yijin Zhou, Yuxi Zhao, Huiyuan Liu, Peidong Wang, Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.18766)  

**Abstract**: Large audio-language models have shown promise for anti-fraud detection by directly processing speech and reasoning over fraud-related evidence. Their deployment, however, requires predictions to follow a predefined label space and a structured decision protocol consisting of service-scenario identification, fraud detection, and conditional fraud-type classification. Existing fine-tuning and prompt-based approaches typically encode task knowledge, constraints, and decision rules into model parameters or manually maintained prompts, making them difficult to adapt as fraud patterns and labeling policies evolve. To this end, we propose FRAUDSkill, a structured frozen-weight adaptation framework that leaves the underlying audio-language model unchanged while optimizing an external layer of skill programs, route-specific policies, and decision rules. We further combine structured output control with validation-guided multi-path inference to ensure protocol-compliant predictions. On the TeleAntiFraud benchmark, FRAUDSkill achieves 73.50% Macro-F1, outperforming the shared frozen-model baseline by 31.96% while reducing invalid outputs to 1.94%. Extensive experiments demonstrate that external skill optimization provides an effective and adaptable solution for structured audio anti-fraud detection without modifying the underlying model. The source code is available at this https URL. 

---
# TeleAntiFraud 2.0: A Refreshable, Profile-Grounded, and Audio-Based Benchmark for Telecom Fraud Detection 

**Authors**: Huiyuan Liu, Zhiming Ma, Yanxing Liu, Shun Zhang, Qifan Wang, Di Liu, Yifan Wang, Yuyang Deng, Haoyang Meng, Yijin Zhou, Yuxi Zhao, Chengxian Hu, Peidong Wang, Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.18748)  

**Abstract**: Telecom fraud scripts evolve rapidly and are often designed to resemble routine service conversations, creating two key requirements for audio-based telecom-fraud evaluation. First, benchmarks must incorporate newly observed scam patterns without overwriting previously established test sets. Second, they must distinguish fraud from lawful, near-domain calls rather than relying on topic-separated negative examples. We present TeleAntiFraud 2.0, constructed with our Mixed-Tree Anti-Fraud Generation Pipeline and evaluated under a monthly frozen evaluation protocol. The pipeline transforms online fraud-case abstracts into profile-grounded scenarios, expands them through mixed-tree generation, realizes fraud and non-fraud dialogue paths under shared contexts, renders validated dialogues as role-matched speech, and freezes the resulting audio, labels, prompts, manifests, and provenance records for each monthly evaluation set. Each frozen set contains 900 Chinese calls, comprising 600 fraud and 300 near-domain non-fraud cases. Controlled text experiments show that three classifiers achieve perfect macro-averaged F1 (Macro-F1) when evaluated against unrelated or ordinary negatives, but drop to 0.65-0.68 with near-domain sibling negatives. Full-set audio and automatic-speech-recognition plus large-language-model (ASR+LLM) evaluations further reveal class-prior shortcuts, prediction collapse, and snapshot sensitivity. Together, these findings establish near-domain construction and collapse-aware reporting as core requirements for evaluating audio-based telecom-fraud models under realistic confusable conditions. The accompanying research artifact includes the construction code, evaluation scripts, manifests, and documentation. Our dataset and code are available at this https URL. 

---
# "If I Had to Buy Just ONE: Galaxy S26 Ultra": Auditing AI-Generated Product Recommendations 

**Authors**: Lucas G. Uberti-Bona Marin, Thales Bertaglia, Giovanni Astante, Bram Rijsbosch, Gijs van Dijck, Anikó Hannák, Gerasimos Spanakis, Konrad Kollnig  

**Link**: [PDF](https://arxiv.org/pdf/2609.18729)  

**Abstract**: Consumers increasingly use AI chatbots for advice on what to buy. With companies like OpenAI and Google monetising their AI through advertising, this raises difficult questions about the bias and impartiality of such advice. In response, we conduct an AI audit of popular chatbots using real commercial-advice queries. First, we curate a dataset of 2,528 real commercial-advice queries (ConsumerQ). Then, we evaluate 1,536 responses to product queries from popular AI chatbots: ChatGPT (chatbot and API), Google Gemini (chatbot and API), and Google Search (AI Overviews). We find that ChatGPT expresses a first-person product preference in 79% of product-recommending responses, compared with 7% for Gemini and 2% for AI Overviews, while the products recommended often change across repeated requests. Displayed sources vary strongly: for the same query, the ChatGPT and Gemini interfaces share only 5.4% of domains on average, with no domain in common in 76.7% of comparisons. APIs provide a different view from their corresponding interfaces, with mean domain overlaps of 12.0% for ChatGPT and 14.8% for Gemini, and also differ in the types and layers of source information they expose. Our findings show that neither isolated responses nor API observations can be assumed to represent the commercial advice consumers encounter. Independent audits of AI-mediated commercial advice should therefore account for repeated responses, consumer-facing conditions, and the source layer being observed. 

---
# RankGround: Efficient High-Resolution GUI Grounding via Lightweight Reranker-Guided Crop Selection 

**Authors**: Liyang Fan, Xinping Bi, Yitai Li, Shuaimin Li, Hui Li, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18690)  

**Abstract**: Graphical User Interface (GUI) grounding is a fundamental perception task for multimodal agents, enabling them to interpret natural language instructions and interact with digital interfaces. Existing methods face a fundamental trade-off between accuracy and efficiency: direct full-image inference often fails to capture small or visually similar UI elements, while multi-crop strategies improve localization at the cost of multiple expensive Vision-Language Model (VLM) calls per query.
To address this challenge, we propose RankGround, a two-stage framework that achieves accurate GUI grounding with a single VLM call per query. Central to our approach is GroundRanker, a lightweight multimodal reranker that identifies the most promising crop from a dense candidate set. Because no off-the-shelf ranking dataset is available, we construct ranking supervision data from existing grounding datasets. A strict containment criterion and boundary-aware positive augmentation improve alignment and spatial coverage in cluttered layouts. GroundRanker is then trained with a two-stage curriculum: a pointwise objective first learns coarse containment, and a listwise objective refines subtle semantic and spatial distinctions among visually similar crops.
Experimental results show that RankGround consistently outperforms strong baselines while reducing computational cost. It achieves 1.4 times faster inference and improves localization accuracy by 5.5% on average over the second-best method across all backbones and screen scales, establishing a new state of the art in both efficiency and precision for GUI grounding. 

---
# Weakening Neurons: An Input-Output Functionality in Transformers with Outsize Influence 

**Authors**: Sebastian Gerstner, Hilal AlQuabeh, Kentaro Inui, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2609.18612)  

**Abstract**: We analyze the learned input-output behavior of GLU-based neurons in large language models (LLMs). We propose a simple analysis method: For each neuron, we compute the cosine similarities between its input (reading) and output (writing) weight vectors. In this scheme, a strong negative cosine similarity indicates the neuron weakens the direction it detects in the residual stream, so we call this a weakening neuron. This allows us to gain a number of novel insights. First, we show that nine different LLMs have similar patterns: weakening neurons appear mostly in late layers whereas their counterparts, (conditional) strengthening neurons, are frequent in early-middle layers. Second, we find that weakening neurons display surprising behavior: even though there are few, they activate often and have a large influence on model behavior. Third, weakening neurons have a strong effect on model output when gate values are negative -- which is surprising since negative gate values are not expected to encode functionality. 

---
# Variational Quantum Transformer Architecture for Synthetic Language Generation 

**Authors**: Julian Hager, Michael Kölle, Gerhard Stenzel, Tobias Rohe, Jonas Stein, Claudia Linnhoff-Popien  

**Link**: [PDF](https://arxiv.org/pdf/2609.18565)  

**Abstract**: We propose a compact NISQ-compatible quantum transformer architecture for synthetic QNLP sequence modelling. The model preserves the autoregressive next-token interface of a classical transformer, but replaces attention and feed-forward sublayers with variational quantum encoder blocks, connector circuits, decoder blocks and a direct two-qubit measurement readout. Token contexts are angle-encoded into small quantum registers, processed by parallel variational heads and encoder integration circuits and conditioned through decoder ancillae to produce a distribution over a four-token vocabulary. We evaluate several architecture variants on deterministic and lexicographic grammar-generation tasks against a compact classical transformer baseline. The quantum models are trainable end-to-end and learn nontrivial grammar structure, including perfect deterministic generation in individual runs and high lexicographic validity in the strongest variant. The classical baseline remains more accurate and stable and the quantum models are sensitive to initialization. The contribution is therefore not a claim of quantum advantage, but a concrete architecture and evaluation of transformer-inspired QNLP sequence modelling under near-term quantum constraints. 

---
# ActionPiece: Rethinking Action Tokenization for Autoregressive Vision-Language-Action Models 

**Authors**: Shijie Lian, Bin Yu, Zhaolong Shen, Xiaopeng Lin, Yichao Du, Zhirui Zhang, Laurence T. Yang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.18487)  

**Abstract**: Action tokenizers play a central role in autoregressive vision-language-action (VLA) models, determining both the targets for policy training and the executable commands recovered from predicted tokens. Their fidelity is commonly evaluated using pointwise reconstruction metrics such as mean squared error (MSE), yet small individual errors do not fully characterize how faithfully action adjustments across demonstrations are preserved. After compression, similar actions may still cluster around a representative motion, while the adjustments needed for different contexts are diminished, distorted, or even reversed. We introduce physical rank consistency (PRC) to measure how well tokenization preserves local physical distance rankings after reconstruction. Evaluating decoded actions provides a common reference across token vocabularies and decoder architectures, complementing pointwise accuracy with a measure of relational fidelity. We further present ActionPiece, which preserves physical action relationships through joint supervision of representation learning and quantization. Physical rank preservation supervises near-far ordering in encoder and quantized feature distances, while quantization regularization applies the same ordering to codeword assignment distributions. Both objectives augment reconstruction, producing discrete action tokens for standard autoregressive policy learning and execution through a frozen decoder. Under the same Qwen3-VL-4B policy training setup, ActionPiece achieves 94.8% on LIBERO and 68.8% on unseen LIBERO-Plus, with additional evaluations reaching 71.9% on SimplerEnv and 51.5% across VLA-Arena L0-L2. Component ablations show that the two objectives jointly improve PRC and policy success, demonstrating the value of physical relationship supervision for action tokenization. 

---
# Divide and Conquer: Mixture-of-Bottleneck Experts in Informative Ordinal Space for Video-based Multimodal Sentiment Analysis 

**Authors**: Ronghao Lin, Qiaolin He, Zefeng Lu, Yichu Liu, Li Huang, Sijie Mai, Haifeng Hu, Yap-peng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2609.18470)  

**Abstract**: Video-based Multimodal sentiment analysis (MSA) must handle information from text, audio, and image sequence in human speaking videos, yet current methods often fail to integrate modalities with task awareness. Most models treat video sentiment prediction as a single task, overlooking its ordinal nature, and their fusion strategies struggle to capture diverse unique and synergic cues across modalities. To address these limitations, we adopt a divide-and-conquer perspective by reformulating MSA as an ordinal regression problem and decoupling it into polarity recognition and intensity prediction. Driven by information theory, we introduce a Mixture-of-Bottleneck (MoB) framework that assigns different latents to polarity- and intensity-specific experts for different modalities. With the learning of information bottleneck, each expert learns compact and task-relevant representations while filtering out redundancy and noise. A multimodal bottleneck routing fusion module then fuses these expert latents with hard mining strategy, guiding the prediction in the ordinal sentiment space. Extensive experiments on 4 MSA datasets and 4 language models show that MoB effectively leverages informative latents from diverse modalities and captures general sentiment structure. Beyond stronger performance, MoB comprehensively captures fine-grained intra- and inter-modal dynamics, enabling more trustworthy localization of nuanced video sentiment signals. 

---
# Disentangling Long-Term Memory via Latent Neuro-Symbolic Reasoning 

**Authors**: Cai Ke, Xinghao Chen, Xiaoyu Shen, Keyu Chen, Siyu An, Junnan Dong, Ruifeng Xu, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2609.18461)  

**Abstract**: Personalized agents are required to reason over long-term history interactions to infer both explicit preferences and implicit behavioral evidence. While early flat retrieval methods score memory fragments independently and neglect the distributed information, current structured memory frameworks rely on query-agnostic static graphs that fail to capture the context-dependent relations. Crucially, raw textual memories are inherently entangled and noisy, making fine-grained personalization and cross-session reasoning computationally prohibitive. To this end, we present LGM, a novel neuro-symbolic framework that shifts long-term memory disentanglement into a continuous latent space. Specifically, (i) instead of persisting fixed graphs, we design a tailored latent graph construction with a sparse autoencoder. Subject to each query, it maps historical interactions into latent memory nodes and disentangles the memory traces into sparse concept activations, dynamically synthesizing query-aware relational edge weights. (ii) A graph encoder then treats the query embedding as a conditioning preference to direct non-linear message passing across the task-specific latent subgraph. This yields a highly expressive memory representation for effective activations. Extensive experiments on long-term personalization benchmarks demonstrate that LGM significantly outperforms state-of-the-art baselines in capturing both explicit and implicit preferences while enabling personalized responses. 

---
# Market Signal Injection: Adversarial Context Manipulation of LLM Pricing Agents 

**Authors**: Dohun Lee, Hyunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2609.18357)  

**Abstract**: Large language model (LLM) pricing agents may respond to how market data is presented, even when its numerical values remain unchanged. We introduce market signal injection (MSI), an attack that manipulates numerical formatting, competitor ordering, or qualitative market commentary without issuing explicit instructions. We evaluate nine open-weight models in simulated Bertrand duopoly and triopoly markets and three proprietary models in duopoly markets. Sentiment-based attacks produce the largest behavioral shifts, which propagate to other firms and alter profits and consumer surplus. Susceptibility varies across model families, and larger models are not consistently more robust. Matched neutral-text controls and a rule-based agent support a framing-based account of these shifts under the fixed demand parameters of our simulation. Episode-held-out probes distinguish baseline from attacked activations in all eleven re-evaluated model--condition pairs: linear AUC is 1.00 and MLP AUC ranges from 0.93 to 0.99. This separability does not by itself identify harmful pricing decisions. Input canonicalization removes the tested sentiment attacks, while decision boundary anchoring, which combines prompt constraints with output projection, provides partial mitigation under the tested adaptive attacks. These results identify data presentation as an attack surface for LLM pricing agents and motivate defenses that account for interactions among agents. 

---
# Faithful yet Collusive: Why Chain-of-Thought Monitoring Cannot Detect Collusion in LLM Pricing Agents under Oligopolistic Competition 

**Authors**: Dohun Lee, Hyunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2609.18346)  

**Abstract**: Large language models (LLM) deployed as autonomous pricing agents may sustain supracompetitive prices through tacit coordination. We develop a causal graph divergence framework that separately measures structural faithfulness and intent faithfulness of LLM pricing agents in Bertrand competition. Across nine LLMs under duopoly and triopoly conditions, collusive behavior and chain-of-thought (CoT) faithfulness dissociate along both dimensions: the most collusive model accurately reports cooperative intent yet reasons structurally unfaithfully, while the most structurally faithful model sustains supra-Nash pricing under both market structures. These findings establish that CoT monitoring alone cannot serve as a standalone safeguard against algorithmic collusion. 

---
# Understanding AI Provider Recommendations in Local Service Markets 

**Authors**: Hazem Ibrahim, Yasir Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2609.18341)  

**Abstract**: When someone asks an AI assistant which doctor to see or which firm to trust with their savings, the answer is a referral. We audit AI provider recommendations in four registry-backed service domains across the 100 largest U.S. metropolitan areas, matching every recommendation against the official registry for its domain (Medicare clinician and facility records, and SEC adviser disclosures), under three conditions: an open-weight model, a proprietary model without web search, and the same proprietary model with search. Without search, both models largely fabricate recommendations in the domains the web covers thinly. Only 4% of the open-weight model's recommended doctors and 11% of the proprietary model's match a clinician in the queried city, and the open-weight matches are name coincidences: its matched clinicians are no likelier to be primary-care doctors than names drawn at random from the registry. With search, 64-71% of recommendations in the same domains match a real provider. Search also changes who is recommended. Without it, recommended advisory firms carry SEC misconduct disclosures at 3.6 times the registry base rate, even after adjusting for firm size; with search, significantly below it. Restaurants, where quality and visibility are separately measurable, show a 3-5x review-count premium but a rating premium of at most a tenth of a star. Finally, search largely removes the metro-size penalty: without it, real recommendations concentrate in the largest metros; with it, match rates are similar across metro-size terciles. Whether an AI referral is trustworthy depends strongly on its retrieval configuration rather than on the underlying model alone, yet an answer produced without retrieval often carries no sign that its recommendations were never verified. 

---
# Relationally Guided Use Case Modeling with LLMs 

**Authors**: Guangyu Wang, Bangqi Li, Ji Wu, Zhijun Shao  

**Link**: [PDF](https://arxiv.org/pdf/2609.18291)  

**Abstract**: Use case flows are important elements of use case modeling because they support downstream software engineering activities, including requirements analysis, architectural and detailed design, and test case generation. However, constructing them manually is costly and expertise-intensive, while existing automated approaches still struggle to preserve semantic consistency, control-flow logic, data-flow logic, and the intended system boundary, especially when identifying branch points and generating alternative flows. To address this problem, we propose FlowGen for complete use case flow construction. FlowGen uses LLM-based Semantic Information Processing (SIP) to extract semantic elements, constructs a Semantic Relational Graph (SRG) encoded by an enhanced R-GAT for basic flow generation (BFGen), and further supports branch point prediction through BPP and branch-conditioned alternative flow generation through AFGen. Evaluations on 13 public and 7 industrial datasets show that FlowGen consistently outperforms competitive baselines in all three core components. In particular, BFGen improves over the best baseline by 14% in Precision, 7-25% in Recall, 11-30% in F1, and 10-19% in AUC; BPP improves Precision by 30-110%, Recall by 33-91%, and F1 by 32-117%; AFGen improves Precision by 8-23%, F1 by 5-18%, and AUC by 0.6-2.5%. Moreover, we validate the effectiveness of the LLM-based SIP module and the attention preservation factor in BFGen, analyze the impact of requirement completeness on BFGen, and examine how different scopes of branch-related context affect AFGen. 

---
# ${M}^2$Tok: Multi-head Multi-codebook Discrete Action Tokenization for Vision-Language-Action Models 

**Authors**: Chunpu Xu, Zhixuan Liang, Yuhao Zhang, Chi-Min Chan, Jessie Wang, Yang Xiao, Mengkang Hu, Xiaokang Yang, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18259)  

**Abstract**: Recent advancements have successfully adapted autoregressive language models to process multimodal signals, such as images and actions. Since raw action signals are continuous, effective tokenization is essential to map high-dimensional inputs into compact discrete tokens for autoregressive processing. However, existing discrete action tokenizers often suffer from high reconstruction loss, failing to preserve the fine-grained dynamics required for precise control. This ``discretization bottleneck'' significantly limits the performance ceiling of downstream Vision-Language-Action (VLA) models. To address this, we propose $\mathcal{M}^2$Tok, a Multi-head Multi-codebook Action Tokenizer designed to minimize reconstruction error and enhance policy performance. Our approach introduces two key structural innovations: (1) we decompose the latent action features into multiple heads, enabling the model to implicitly align specific heads with distinct action dimensions; (2) we assign independent codebooks to each head for quantization. By leveraging the combinatorial nature of multiple codebooks, we significantly expand the representational expressivity of the tokenizer, leading to substantially lower reconstruction loss compared to previous methods. We evaluate the $\mathcal{M}^2$Tok-based VLA on the RoboTwin, Simpler-Env, and 3 zero-shot real-world tasks. Experimental results demonstrate our method not only achieves superior reconstruction fidelity but also significantly boosts the success rate of VLA models. Comprehensive ablation studies further confirm the effectiveness of the multi-head and multi-codebook mechanisms. Code is available at \href{this https URL}{this https URL}. 

---
# PageRecall: Measuring Page Selection in Literature-Grounded Question Answering 

**Authors**: Aaditya Chauhan  

**Link**: [PDF](https://arxiv.org/pdf/2609.18154)  

**Abstract**: We describe our system for LitTraceQA (GroundLM @ EMNLP 2026): given a research question, retrieve the relevant papers from a pool of 27,487, cite the page and the table or figure where the answer lives, and answer in a requested format. Our main finding is that evidence grounding is limited by retrieval, not by reading. The page selector put the annotator's page, which we call the gold page, in front of the model that locates evidence only about half the time (52.6% gold-page recall), while that model, given the page, cited the right one in 45 of the 48 locators it emitted (94%). When the page was missing it rarely said so: of 45 such cases it returned nothing 14 times, a wrong page 24 times, and a correct page 7 times, so the pipeline failed quietly almost twice as often as it failed visibly. Since the failure was that the right page was never shown, the fix is to stop choosing: each retrieved paper fits in the model's context, so we show it whole. Page ranking survives only as a fallback inside papers too long to fit, which no test-split paper was, and gold-page recall reaches 100% on the papers we can parse. Separately, questions that identify their target by position rather than content, such as "the first author of the 24th reference", are served by parsing rather than retrieval: we resolve the bibliography into an addressable list, which also supplies identifiers the evidence metric scores. The final system scores 0.762 paper $F_1$, 0.441 evidence $F_1$ and 0.920 multiple-choice accuracy on the held-out test split. Because the pipeline depends on a closed model without seed control, we release a harness that verifies the paper's central claims against committed artifacts. 

---
# Colla-Q: Toward Collaborative Experts in MoE Quantization via Minimax Precision Balancing 

**Authors**: Eunju Shin, Jongbin Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2609.18131)  

**Abstract**: In this paper, we present a Mixture-of-Experts (MoE) quantization method based on activation entropy. Although quantization reduces memory and computational costs, it can substantially degrade performance. In particular, performance decline is pronounced in quantized MoE models, where individual experts have a small number of parameters that are sensitive to low-bit representation. Considering that MoE operates as an ensemble model with collaborative contributions from routed experts, a significant performance decline of a particular expert due to quantization can harm model performance. Therefore, we propose Colla-Q, a bit-allocation framework to maintain balanced performance across experts through an activation-entropy-based bit-width allocation algorithm. This approach encourages each expert to operate collaboratively in the quantized model, thereby 1) improving the overall MoE performance and 2) reducing the dependence on the calibration dataset. Since uniformly adjusting each expert's performance facilitates robustness and stability of the MoE model, the proposed MoE quantization method can generalize more consistently across different calibration datasets. Our code is available at: this https URL 

---
# A Comprehensive Review of Generative Physical Artificial Intelligence 

**Authors**: Satyam Gaba, Krutiksinh Rana, Siva Sai, Vinay Chamola, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2609.18111)  

**Abstract**: The integration of large-scale foundation models with physical embodiments has led to significant advancements in robotics known as Generative Physical Artificial Intelligence (GPAI). These agentic AI systems autonomously perceive, reason, and act in complex real-world situations. This survey comprehensively analyzes GPAI systems, focusing on their architectural foundations, current applications, and key limitations. We introduce a taxonomy of five distinct approaches: Robot Foundation Models (RFMs) for cross-platform skill transfer; Vision-Language Action (VLA) models for end-to-end multi-modal perception and control; Large Behavior Models (LBMs) for human-like movement generation; Diffusion Policy Models (DPMs) for diffusion model-based temporally coherent action generation; and World Foundation Models (WFMs) for physics-compliant simulation and data generation. We examine how these approaches complement each other: WFMs generate training data for VLAs and DPMs, RFMs enable cross-platform deployment of learned policies, while LBMs provide motion priors for natural behavior. Through examples across autonomous vehicles, industrial automation, healthcare robotics, and humanoid systems, we identify significant performance improvements and summarize promising research directions in data-efficient learning, sim-to-real transfer, edge-compatible architectures, and safety frameworks. These insights advance embodied AI for IoT-connected environments where intelligent agents interact with networked sensors, actuators, and edge devices. 

---
# Agora: Git as Shared Memory for Collective AutoResearch 

**Authors**: Yifan Zhang, Yunheng Zou, Shaokun Zhang, Jian Hu, Hao Zhang, Binfeng Xu, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2609.18094)  

**Abstract**: Autonomous research loops such as AutoResearch show that one coding agent can improve a training setup unattended. Run several of them and each session starts from scratch, so more agents tend to mean more duplicated search rather than more discovery. Agora is a shared memory for such agents: research is recorded as an append-only directed acyclic graph (DAG) stored in Git, so that every claim is a commit anyone can check out and rerun. Each result, insight, hypothesis, verification, and report is an immutable commit whose parent edges say what it builds on; a derived index exposes the frontier, the neglected branches, and the verification status of each claim, and a diversity-aware selection rule keeps the community from collapsing onto one leader. We describe the system and report its first sustained use: a run of nearly 12 days in which 13 language-model workers, with no assigned tasks and no central planner, worked on a weight-transfer problem. Given 141 pretrained donor models and a frozen 119.6M-parameter attention-SSM hybrid whose dimensions match no donor, the workers had to initialize the target without training data or gradient updates. They published 1,703 contributions and drove the evaluator from 3.39 to 1.899 bits per byte, closing 62% of the gap to a trained GPT-2 124M. The winning recipe compresses donor next-token statistics into the target's embedding and output head, then adds a short-range context signal through sparse edits to attention, feed-forward, and state-space blocks. Its 145-commit ancestry spans 15 accounts, and 165 independent reproductions were posted, none of which failed. We describe the single mid-run human intervention that pulled the community out of a monoculture, what the trace does and does not establish, and the controlled comparison that would settle whether shared research state improves discovery per unit of compute. 

---
# Correlation-Guided Encoder Selection for Multi-Encoder Large Audio-Language Models 

**Authors**: Pei-Jun Liao, Hung-Shin Lee, Wenze Ren, Kuo-Hsuan Hung, Hung-yi Lee, Hsin-Min Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.18041)  

**Abstract**: Multi-encoder fusion extends Large Audio-Language Models (LALMs) beyond speech-centric recognition, but selecting encoders via intuition or exhaustive search often introduces redundant representations and inflates an already constrained compute budget. We propose CUES (Correlation-gUided Encoder Selection), a lightweight heuristic that estimates complementarity through task- and category-level Pearson correlations between encoders' performance profiles, scoring a candidate set from single-encoder evaluations alone--without fusion training during selection. Evaluated on the XARES-LLM benchmark with a frozen SmolLM2-135M backbone (LoRA-adapted) via five-fold cross-validation, CUES consistently identifies the same configuration per track from held-out development splits alone, without using test data for selection. For the broad Track~A suite, CUES selects a cross-family trio (Whisper-medium, mHuBERT-147, and Dasheng-base), achieving a 4.3% relative gain over Whisper-medium (0.771 vs. 0.739). For Track~B text generation, it re-anchors on a focused, speech-only pair (mHuBERT-147 and WavLM-base-plus) and actively abstains from adding a divergent encoder, outperforming mHuBERT-147 by 6.3% (0.589 vs. 0.554). Rather than a failure to scale, this divergence is consistent with a diversity--interference trade-off that CUES navigates per track from correlation signals alone: across the evaluated pool, added cross-family diversity tends toward an inverted-U on broad audio tasks but toward steady degradation on text generation, which favors a focused, speech-anchored set. 

---
# G-Mamba: Sparse Graph-Guided Mamba for Audio-Visual Speech Enhancement 

**Authors**: Guo-Ruei Tseng, Hung-Shin Lee, Hsin-Min Wang, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.18009)  

**Abstract**: Lightweight audio-visual speech enhancement (AVSE) models face a critical trade-off between computational efficiency and cross-modal alignment accuracy. While simple concatenation lacks relational expressiveness, dense cross-attention incurs computational overhead and is prone to unreliable cross-modal correspondence under strong acoustic interference. We propose Sparse Graph-Guided Mamba (SG-Mamba), a lightweight AVSE framework that integrates a sparse heterogeneous graph with a linear-complexity Mamba backbone. The graph explicitly models modality-specific relations through content-adaptive attention and cross-frame audio-visual connections, while Mamba captures long-range temporal context. We further introduce an audio skip connection to preserve spectral detail without sacrificing noise suppression. Evaluated on LRS3, SG-Mamba achieves competitive or superior performance against strong lightweight baselines and reaches 13.091 dB SI-SDR under noise-only condition. It also remains robust in cluttered multi-speaker conditions with a competitive cost of 3.45 G MACs (or 6.90 G FLOPs). Results on VoxCeleb2 further suggest that explicit structural priors improve robustness, generalizability, and computational efficiency in lightweight AVSE. 

---
# Encoder Awakening via Adapters: Effective Domain-Adaptive Fine-tuning of Speech-LLMs 

**Authors**: Mohan Shi, Zilai Wang, Natarajan Balaji Shankar, Kaiyuan Zhang, Eray Eren, Abeer Alwan  

**Link**: [PDF](https://arxiv.org/pdf/2609.17981)  

**Abstract**: Speech Large Language Models (Speech-LLMs), typically built from a pre-trained speech encoder, a modality projector, and an LLM fine-tuned with Low-Rank Adapters (LoRA), have shown strong Automatic Speech Recognition (ASR) performance on general-domain speech. However, adapting them to domain-shifted speech, such as child or dialectal speech, remains challenging under limited target-domain data. Given the dominant role of the LLM in Speech-LLMs, with cross-entropy loss applied only at the LLM output, the speech encoder may receive insufficient adaptation to new acoustic conditions. In this paper, we propose Encoder Awakening via Adapters (EAVA), a simple yet effective domain-adaptive fine-tuning method for Speech-LLM-based ASR. First, lightweight adapters are inserted into each encoder layer and trained exclusively, enabling target-domain acoustic knowledge to be incorporated into the encoder while preserving its pre-trained knowledge. Second, the full model is jointly fine-tuned on the target domain with LoRA applied to the LLM. Experiments on three domain-shifted ASR datasets, covering child and dialectal speech, show that EAVA consistently outperforms vanilla fine-tuning and other baselines, achieving new state-of-the-art performance. 

---
# ASPIRE: Asynchronous Batched Self-Speculative Decoding for Long-Context LLM Inference 

**Authors**: Amir Ziashahabi, Hossein Entezari Zarch, Lei Gao, Murali Annavaram, Salman Avestimehr  

**Link**: [PDF](https://arxiv.org/pdf/2609.17943)  

**Abstract**: Long-context LLM inference is bottlenecked by attention, whose repeated KV-cache reads make decoding memory-bound. Self-speculative decoding alleviates this by drafting tokens with sparse attention and verifying them with full attention, but existing batched methods remain synchronized: all requests in a batch share a single draft-verify schedule, even though the optimal draft length varies widely across requests and changes dynamically within each request. We propose ASPIRE, a non-synchronized batched self-speculative decoding framework built on three components. First, a unified mixed forward allows drafting and verifying requests to coexist in the same batched forward pass, removing the need for global draft-verify phases. Second, a lightweight online speculation scheduler uses per-request acceptance-rate estimates and a batch-aware cost model to let each request independently choose when to verify. Third, an intra-draft refresh layer performs full attention at a single designated layer during drafting, updating the sparse context at every draft step to reduce staleness during drafting. Across three models and five reasoning and long-context benchmarks, ASPIRE achieves $1.70$-$4.58\times$ speedup in decoding throughput over autoregressive baselines and improves average speedup by approximately $27\%$ over the strongest prior self-speculative baselines. 

---
# Long-Context Demonstration Selection Using State Space Models 

**Authors**: Ziniu Zhang, Zhenshuo Zhang, Ruoxuan Xiong, Gene Cooperman, Hongyang R. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17888)  

**Abstract**: We study the problem of demonstration selection, which involves selecting a subset of examples for prepending to a query to a language model. This problem is closely related to in-context learning and language model inference. Since the inference cost of a transformer model scales quadratically with sequence length, the selection problem becomes especially challenging in a long-context scenario. In this paper, we tackle this problem by building on state space models (SSMs), which require only linear inference time given the input. Our approach involves two algorithms. The first learns a small set of SSMs through distillation of a (trained) transformer model. We partition all the layers into consecutive groups. Then for each group, we estimate a separate state space model to replicate the input-output behavior within the adjacent layers. Second, we map the distilled model outputs to a small set of tokens, and apply these embeddings for demonstration selection in downstream applications. We perform extensive experiments in both synthetic and real-world datasets to validate our approach. We demonstrate that the distilled SSMs only incur an approximation error of less than $0.7\%$ relative to the true output. In downstream evaluation, we show that on several text classification and reasoning tasks, our approach reduces FLOPs by $14.2\times$ and improves accuracy by $6.48\%$ relative to baseline demonstration selection methods. 

---
# Uncertainty-Aware Continual Learning for Open-World Intent Discovery Under an evolving Label Space 

**Authors**: Pisante Aida, Formentin Simone  

**Link**: [PDF](https://arxiv.org/pdf/2609.17866)  

**Abstract**: Real-world intelligent systems increasingly operate under open-world conditions, where user intents are not fixed or exhaustively known a priori and may evolve as new interaction patterns emerge. This paper proposes a unified uncertainty-aware probabilistic framework for continual new intent discovery under an evolving label space. Each utterance is encoded through an adaptive $\beta$-VAE into a latent mean, used for classification and density modelling and a posterior uncertainty estimate acting as a global reliability signal. Classifier confidence, posterior uncertainty and DP-GMM likelihood are combined through a multi-signal decision mechanism to distinguish known intents from potentially novel samples. Candidate novel instances are clustered through a density-based discovery module and only reliable clusters are promoted to new labels, enabling controlled label-space expansion. Replay and Elastic Weight Consolidation mitigate catastrophic forgetting and preserve previously acquired knowledge. The paper formalises continual intent discovery as a structured multi-phase open-world problem, introduces adaptive label-space expansion under stability--plasticity constraints and uses posterior uncertainty to regulate trusted-sample selection, pseudo-labelling, novelty admission and replay. Experiments show high novelty precision, stable adaptation across sequential phases and limited forgetting. Near-zero NMI and ARI indicate limited reconstruction of the complete fine-grained intent taxonomy, consistent with the framework's conservative promotion strategy. Qualitative analyses nevertheless reveal dense and locally coherent semantic clusters, showing that reliable novel structures can be discovered without exhaustive recovery of the underlying taxonomy. 

---
# NeMo Data Designer: An Extensible Framework for Multimodal Synthetic Data Generation 

**Authors**: Johnny Greco, Nabin Mulepati, Andre Manoel, Eric Tramel, Kirit Thadaka, Mike Knepper, Dhruv Nathawani, Dane Corneil, Yev Meyer, Alex Watson, Maarten Van Segbroeck  

**Link**: [PDF](https://arxiv.org/pdf/2609.17699)  

**Abstract**: We present NeMo Data Designer (NDD), an open-source, general-purpose framework for multi-modal synthetic data generation (SDG). Designed to be intuitive to use, NDD provides a declarative configuration format in which human and/or agent users define each dataset column, with column types spanning text, code, structured outputs, images, embeddings, and statistical samplers that are explicitly configured to steer dataset diversity. Additional column types and functionality can be introduced using the framework's flexible plugin system. NDD's configuration is an inspectable artifact, supporting workflow sharing and reproducibility. SDG is an inherently iterative process. NDD therefore builds a preview-and-revision loop into its core workflow, allowing users to generate and inspect a small number of records, refine the specification, and rerun generation at full scale. At runtime, NDD resolves dependencies, schedules calls to user-provided model endpoints, and retries failed requests. We describe NDD's architecture and programming model and present case studies spanning structured, agentic, multimodal, and domain-specialized tasks, including datasets used in Nemotron model development and in production enterprise deployments. 

---
# GraphEcho: Structural Redundancy and Evidence Provenance in LLM Graph Agents 

**Authors**: Sikun Wang, Yixi Zhou, Lei Fan, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17695)  

**Abstract**: A large language model (LLM) agent can follow more graph paths without acquiring more independent evidence. GraphEcho tests whether agents mistake these repeated encounters for additional corroboration. The benchmark varies path counts and evidential origins while holding evidence content fixed, and evaluates both judgments and active exploration. Controlled synthetic experiments reveal model-dependent judgment shifts, but redundant supporting paths increase the share of repeated walks across all evaluated frozen agents. Provenance-aware post-training (PAPT) reduces revisits and improves synthetic accuracy, yet covers fewer distinct sources. On scientific claims, it continues to reduce repetition while accuracy declines. These findings expose a gap between efficient exploration and effective evidence use: an agent can learn to stop repeating itself while overlooking information it needs. GraphEcho provides a controlled way to evaluate both what graph agents conclude and whether their exploration reaches distinct evidential sources. 

---
# The Missing "I Don't Know": Why Three Reasoning-Reliability Findings Converge on Calibrated Abstention 

**Authors**: Srijith Ravikumar  

**Link**: [PDF](https://arxiv.org/pdf/2609.17686)  

**Abstract**: Three recent results describe what look like unrelated LLM reliability problems. Yin et al. (2026) show reasoning RL collapses tool-reliability representations. Suleymanov et al. (2026) show that under safety-constrained generation, large models rewrite flagged spans while small models truncate. Bastounis et al. (2024) prove any consistent-reasoning system without an implicit "I don't know" function must hallucinate infinitely often on broad problem classes. We argue these findings converge on a single intervention: calibrated abstention is what each independently identifies as the missing capability, even though the unavailability they document, a capability gap, a policy gap, and a recursion-theoretic gap, has a different source in each case. Honesty post-training has narrowed the gap in deployed models, but principled closure of the class Bastounis identifies requires a calibrated abstention function whose training signal at the leaderboard level is absent: dominant benchmarks assign zero reward to decline, so the leaderboard gradient that would select for the function does not exist. We propose four changes to evaluation: triple-scoring, abstention-rate reporting, capability-stratified evaluation, and mandatory calibration metrics. Benchmark reform is necessary, not sufficient, for closing the gap the theorem identifies. 

---
# Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches 

**Authors**: Vivek Kalyanarangan  

**Link**: [PDF](https://arxiv.org/pdf/2609.17652)  

**Abstract**: When agentic sessions run to a million tokens with many sessions resident at once, the KV cache and the index that ranks it live in host memory, and the scan that ranks all n keys for a top-k step becomes the traffic that bounds decoding. We present Fathom, a key scan in which each query decides how many bits of each key channel to read. The 4-bit K cache is stored channel-major as bit planes, so a prefix of t planes is exactly the channel's t-bit quantizer, and the query spends its bit budget by reverse water-filling over the variance-weighted importance of its channels. At one million tokens on Qwen3-8B a decode step is 1.67x faster in GPU time than with the 136-bit scans of Double Sparsity, Loki and SparQ r=32, and in the same GPU time as SparQ's 68-bit read (r=16) Fathom reads 18% fewer bytes with lower attention error on six of seven model and context settings. On RULER-style tasks every per-token scan matches exact top-k decoding, and on real coding-agent sessions Fathom reaches the step agreement of the most accurate 136-bit scan at 92 bits. The store is the 4-bit K copy a quantized serving stack already holds, and the method is not faster when the index is resident in GPU memory. 

---
# EvolveTrade: Experience-Driven Policy Refinement for Self-Evolving LLM Trading Agents 

**Authors**: Sehee Kim, Yumin Choi, Minki Kang, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2609.17632)  

**Abstract**: Large language model (LLM) trading agents can combine market data, news, and executable analysis, but their behavior is often controlled by static hand-written tool-use policies that are fixed before deployment. This limits their ability to adapt how they gather evidence, invoke tools, verify signals, and manage risk under changing market regimes. We introduce EvolveTrade, a self-evolving framework that treats the system prompt of a tool-using trading agent as a text-parameterized policy. After each update interval, a Policy Agent revises this policy using accumulated decision traces and realized portfolio feedback, while keeping the backbone LLM fixed. The updated policy is then used for the next batch of trading decisions, enabling the agent to refine its information-acquisition and portfolio-construction procedure over time. Experiments across multiple market regimes and two LLM backbones show that EvolveTrade often improves Sharpe Ratio and Cumulative Return over fixed-policy LLM baselines, achieving the improved SR and CR in most evaluated settings. Behavioral analyses further show that self-evolved policies increase code-mediated analysis and activate regime-relevant computations; case-level policy-to-return attributions trace how policy-induced allocation changes contribute to realized return differences. These results suggest that adapting the reusable procedure governing tool use is a key direction for building more robust LLM trading agents. 

---
