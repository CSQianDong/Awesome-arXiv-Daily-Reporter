# ReWorld: An Interactive World Model with Long-Horizon Memory 

**Authors**: Zhifei Chen, Luozhou Wang, Guibao Shen, Dongyu Yan, Shuai Yang, Tianshuo Xu, Yihua Du, Wei Wang, Tianyi Gui, Lianghua Huang, Yingcong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.23565)  

**Abstract**: An interactive world model must follow the user's actions, remember the places it has shown, and stream in real time. The tension is structural: control wants a short horizon, memory wants an unbounded one. ReWorld separates the two during training and bounds them at inference. Mixed per-head attention windows confine most heads to the recent past while a small set of global heads attends over the entire history, and random head routing keeps either capability from binding to particular heads; random chunk dropping makes sparse histories in-distribution. At inference the whole past lives under a fixed budget: a bounded KV cache backed by a pose-indexed landmark bank, from which the model retrieves the landmarks nearest the current pose. A metric-scale-aligned data engine places eight sources -- Unreal-rendered fly-throughs, game roaming, and real-world footage -- on one physical action scale, so the same key press moves the camera the same distance in every source, and palindrome trajectories supply the revisit evidence that memory training needs. Distribution-matching distillation confined to a LoRA adapter then compresses sampling to four steps: one backbone serves both a high-fidelity multi-step mode and a real-time interactive one, streaming 704x1280 video across photorealistic, game-style, and stylized worlds. Under a three-axis protocol covering action following, long-horizon recall, and video quality, against six recent interactive world models it attains the best control fidelity ($11.95^\circ$ rotation error and the best camera-motion consistency) and the best generation quality; and on minute-long out-and-back rollouts ($64$\,s, $384$ latents), its fixed 12-chunk cache still regenerates the starting view -- at rollout lengths where a sliding window has long evicted the evidence and full-KV attention runs out of memory. 

---
# Prime Agent: A Self-Improving RLM Harness 

**Authors**: Seth Karten, Alex L. Zhang, Kevin Thomas, Sebastian Müller, Elie Bakouch, Daniel Auras, Mika Senghaas, Fares Obeid, Konstantin Dunas, Johannes Hagemann, Sami Jaghouar  

**Link**: [PDF](https://arxiv.org/pdf/2608.23552)  

**Abstract**: Language models are sequential processors, but long-horizon agency requires external information and computation beyond model weights and active context. Prime Agent is an open-source harness for long-horizon evaluation and coding-agent workflows. A persistent IPython REPL follows the Recursive Language Model abstraction for programmatic context processing and test-time compute, while Continual Harness preserves histories, memories, skills, prompts, and subagent specifications across trajectories. Recursive subagents coordinate through direct agent-to-agent communication, and the Agents View lets humans inspect and manage daemon-backed sessions. Prime Agent standardizes execution, recovery, verification, and resource accounting while leaving strategy construction to the model. This low-friction, expressive membrane prevents harness failures from becoming model failures and pushes measurement toward the model's true maximal underlying capability. Prime Agent raises ARC-AGI-3 RHAE Best@1 from 30% to 95.5% and matches or exceeds native and popular harnesses across long-context coding, GPU-kernel generation, emulator construction, and autonomous nanoGPT speedruns. On Factorio, we find refinement allows for continuous technology progression and dedicated subagents enable parallelized work. Code is available at this https URL. 

---
# How AI Assistance Affects Human Skill Development: A Study of Learning with Logic Puzzles 

**Authors**: Shang Wu, Catarina G Belem, Shuyuan Fu, Mark Steyvers, Padhraic Smyth  

**Link**: [PDF](https://arxiv.org/pdf/2608.23543)  

**Abstract**: While AI assistance can improve human task performance in the short term, it may also undermine the development of skills in the longer term. We examine this tension in a controlled logic-puzzle experiment involving on-demand AI assistance, where participants complete tasks before, during, and after AI is available. By experimentally varying AI request costs, we find that lower-cost assistance induces more frequent AI use. We also find that participants who request AI assistance during the AI-access phase perform worse at the task after assistance is removed, and their subsequent unassisted performance is overestimated when predicted from earlier AI-assisted performance. We use a Bayesian latent ability model to separate initial ability, post-AI ability, and participant-specific skill change, while estimating how independent reasoning during the AI-access phase relates to skill development. The results show that greater independent problem-solving effort is associated with larger gains in latent ability, consistent with the interpretation that skill development is weaker when AI assistance substitutes for independent reasoning. 

---
# Correcting a learned physical invariant improves world-model rollouts 

**Authors**: Richard Bao  

**Link**: [PDF](https://arxiv.org/pdf/2608.23526)  

**Abstract**: World models can predict video without learning dynamics that they reliably preserve. We test whether a frozen DreamerV3 trained only on pendulum video learns a scalar that its own latent transition treats as approximately conserved. A label-free search recovers the same energy-like invariant across independently trained conservative models, while the same procedure finds no comparable invariant in matched damped models. During autonomous rollouts, this quantity drifts. Projecting the latent state back toward its initial level set reduces rollout error in all three conservative models, whereas matched random constraints usually increase it. These results distinguish a dynamically meaningful invariant from a merely decodable correlate and reveal a concrete failure mode: a world model can learn a physical constraint from pixels yet violate that constraint when it imagines forward. 

---
# EarthVerse: Benchmarking Scientific Agents Across Dynamic Earth Systems and Natural Hazards 

**Authors**: Zhiqing Cui, Xinxiang Yin, Yihong Tang, Xinglang Zhang, Yuanzhe Hu, Siru Zhong, Weidong Tang, Yuxuan Liang, Weijia Li, Ming Jin, Shirui Pan, Yuhao Kang, Dingyi Zhuang, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.23525)  

**Abstract**: Earth-system analysis reconstructs changing physical processes from observations that differ in source, scale, timing, and modality. Natural hazards make this work consequential because incomplete evidence can change estimates of severity, exposure, and mechanism. We introduce EarthVerse, a benchmark that evaluates scientific agents through package-scoped investigations. Its 405 reproducible tasks are grounded in 199 documented events and 19 hazard families. Agents inspect heterogeneous event packages, choose compatible evidence, execute transparent calculations, reconcile source differences, and preserve provenance in the final answer. We provide executable ground truth that decomposes each task into fine-grained answer units, together with task-specific rubrics that assess the supporting research process while allowing multiple valid paths. We evaluate 25 model and agent systems under a controlled tool-using protocol, then use controlled studies to locate failures in evidence access, tool selection, memory, reasoning, interaction, and scientific execution. Across systems, the best mean answer-unit accuracy is 84.65%, while the highest Strict@95 is only 34.81%. The gap shows that current agents often complete individual steps without maintaining a consistent chain across evidence, scales, units, calculations, and physical interpretation. EarthVerse provides a reproducible basis for measuring end-to-end scientific reliability in dynamic Earth systems. 

---
# Mitigating Reasoning-Induced Misalignment via Safety-Direction Penalty 

**Authors**: Yipeng Zhao, Qishun Yang, Shenzhe Zhu, Shu Yang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23497)  

**Abstract**: Reasoning-Induced Misalignment, where fine-tuning on reasoning data containing no harmful content, including mathematics, code, and problem-solving with chain-of-thought traces can induce harmful behaviors of LLM, posing a serious challenge to the safety of LLM reasoning. Cross-architecture, cross-scale, and cross-dataset checks show that RIM does not always emerge. Previous work attributed RIM to neuron-level entanglement, but did not identify the geometry of the representation space underlying this entanglement or propose a training-time fix. We provide both: a representation-space analysis of RIM and the Safety-Direction Penalty (SDP), which penalizes movement along a learned safety direction during reasoning fine-tuning. The analysis extracts two activation-space directions, one encoding reasoning ability and the other safety behavior. These directions are coupled: fine-tuning that improves reasoning shifts safety representations, and prompts with larger shifts show larger safety degradation. CKA distance ratios and probes locate the safety-decision layers where this shift is most relevant. These findings guide the design of SDP: the coupling motivates penalizing displacement along the safety direction, and the layer localization sets the initial scope. When the initial scope leaves compensatory shifts beyond the penalized layers, the same diagnostics guide iterative expansion. On Qwen2.5-3B and 7B, SDP restores safety while preserving benchmark reasoning performance. 

---
# SRPO: Self-Reflective Policy Optimization for Long-Horizon Reasoning 

**Authors**: Jialong Liu, Yuling Shi, Ning Yang, Xiaodong Gu, Zuchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.23493)  

**Abstract**: Self-reflection is a powerful mechanism for credit assignment in human learning, converting sparse outcome feedback into actionable guidance. However, its potential for post-training Large Language Models (LLMs) remains underexplored. We propose Self-Reflective Policy Optimization (SRPO), a framework that internalizes this capability. SRPO enables LLMs to analyze their own completed trajectories, synthesize errors into concise "reflection patches," and use reflection-conditioned teacher scores on student on-policy rollouts as dense token-level training signals. This process effectively transforms sparse terminal supervision into dense, token-level learning signals without requiring external critics, separate reward models, or larger teacher models. We demonstrate that SRPO achieves state-of-the-art performance across mathematical reasoning and long-horizon agentic benchmarks with exceptional data efficiency. Using a Qwen3-8B base model, SRPO attains 73.3% on AIME'24 using only 8% (0.08x) of the training FLOPs required by scaled supervised fine-tuning, while significantly improving success rates on WebShop (64.7%), ALFWorld (76.8%), and SWE-Bench-Lite (31.2%). Code is available at this https URL 

---
# Multi-Modal Semantic Expansion with Constrained LLM Reranking for Conversational Music Recommendation 

**Authors**: Naman Garg, Sarika Jain, George Fazekas  

**Link**: [PDF](https://arxiv.org/pdf/2608.23484)  

**Abstract**: We present Team Semiintelligencn's solution for the ACM RecSys 2026 TalkPlayData Challenge, addressing conversational music recommendation through a multi-modal and personalized conversational recommender system. Our submitted system employs a three-stage pipeline: (1) multi-modal retrieval constructing decay-weighted centroids across seven dense embedding spaces - track- and user-level CF-BPR, Qwen3 (metadata, lyrics, attributes), CLAP audio, and SigLIP visual - supplemented by BM25 lexical retrieval and an artist substring-match signal, all fused via weighted Reciprocal Rank Fusion (RRF) with optimized signal weights; (2) lightweight reranking (history filtering, popularity smoothing, and catalog diversity penalization); and (3) persona-diversified response generation using GPT-4o-mini. Beyond this submitted configuration, we report development-time experiments with additional components - constrained LLM-guided artist injection, album continuation signals, XGBoost LambdaMART, and a superior GPT-4.1 response prompt - that were not deployed to Blind B due to cost and complexity constraints. We optimize RRF weights on a 500-session development split via differential evolution, improving MRR by +19.5%. On Blind A, we observe that unconstrained LLM-guided injection across 54 sessions causes catastrophic nDCG regression (-18.9%), while conservative injection on only 9 sessions yields the best observed Blind A nDCG - a finding we present as a Blind A observation warranting further validation. The submitted system achieves a Blind B composite score of 0.3213. 

---
# StrategyBench: Evaluating Explicit Strategy Induction in Large Language Models 

**Authors**: Jinghan Tan, Yuanzheng Wang, Lu Chen, Zijun Chen, Yuqian Wang, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.23475)  

**Abstract**: As large language models are increasingly used in data-scarce and evolving task scenarios, few-shot in-context learning (ICL) has become a key paradigm for task adaptation. However, direct ICL often uses a small set of examples without explicitly abstracting task rules, making it sensitive to example construction. In contrast, human learners often reduce such sensitivity by first summarizing task rules from examples and then applying them to new instances. To evaluate this ability, we propose StrategyBench, which selects strategy-inducible tasks from BIG-Bench, constructs reference strategies, and defines evaluation metrics along two dimensions: strategy quality and downstream utility. We further analyze strategy induction from three perspectives: task variation, model configuration, and adaptation setting, covering category-wise differences, generator-executor choices, demonstration design, and SFT-based adaptation. Experiments show that explicit strategy utility differs substantially across task categories and depends on both strategy generation and execution conditions. The benchmark is released at: this https URL. 

---
# Characterizing Necessary Losers to Explain Tournaments Losers 

**Authors**: Contet Clément, Umberto Grandi, Jérôme Mengin  

**Link**: [PDF](https://arxiv.org/pdf/2608.23446)  

**Abstract**: We study the problem of formally explaining why a candidate was not selected by a given tournament rule, by identifying sub-tournaments in which the candidate loses independently of how the rest of the tournament is completed. We define destructive minimal supports as any minimal sub-tournaments satisfying this property, which in formal explainable artificial intelligence correspond to abductive explanations for the question "Why does the loser lose the tournament?". For six common tournament solutions (maximin, uncovered set and its weighted variant, top-cycle, Copeland, and Borda) we provide characterizations of when a candidate is either a necessary loser or a possible winner, we determine the size of the smallest destructive minimal supports, complemented by polynomial-time algorithms for their computation except for the case of the Borda rule which is suspected to be NP-complete. 

---
# SkillAlchemy: Open-World Agent Skill Creation 

**Authors**: Hengjun Wang, Shuyue Wei, Boyi Liu, Jun Yang, Yongxin Tong  

**Link**: [PDF](https://arxiv.org/pdf/2608.23417)  

**Abstract**: Agent skills are reusable procedural artifacts that extend language agents with specialized workflows, tool conventions, and domain behaviors at inference time. However, creating reliable skills still depends largely on human authorship, model priors, or execution traces. These sources are often unavailable for unfamiliar tasks, suggesting the need to create skills from open-world materials. In this paper, we study open-world skill creation: given an underspecified skill brief and a source-access specification, a creator must discover behavior-relevant requirements omitted by the brief and determine how broadly each source-derived procedure is justified. We propose SkillAlchemy, an admission-centered framework for source-grounded skill creation. SkillAlchemy identifies implicit requirements through contrastive evidence, admits candidate procedures based on evidence-supported scope, and compiles the admitted content into a grammar-guided skill package. Extensive experiments across 87 SkillsBench v1.1 tasks demonstrate that SkillAlchemy improves pass rate over no-skill execution by 19.9 percentage points and the strongest automated baseline by 8.6 percentage points, while achieving performance comparable to human-curated skills. 

---
# MediSkill-Evo: Process-Constrained Self-Evolution for Evidence-Grounded Clinical Interaction 

**Authors**: Ruoyu Wu, Shenfu Xie, Yinqian Sun, Haibo Tong, Feifei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.23397)  

**Abstract**: Interactive clinical agents must gather decisive evidence and convert it into grounded actions under partial observability. A correct final diagnosis alone does not show that an agent respected evidence and care-process constraints. We introduce MediSkill-Evo, a clinical agent that evolves governed process knowledge without backbone fine-tuning. It separates experience into four typed banks for clinical skills, process rules, symbolic schemas, and measurement procedures. Provenance, support, replay, and controller-defined safety checks govern publication to a frozen test-time snapshot. A Process-Constrained Preference Harness binds evidence to its source, rejects controller-invalid candidates, and ranks actions with a safety-prioritized Clinical Process Critic. We evaluate complete agent systems across two backbone endpoints and six controlled stress dimensions under the same Doctor-turn limit. On 300 held-out Qwen encounters, MediSkill-Evo improves diagnosis accuracy from 61.33 percent to 69.00 percent and treatment-intent coverage from 33.62 percent to 66.44 percent, while reducing automatically scored critical failures from 31.00 percent to 16.33 percent relative to AgentClinic. On 180 hard-isolation conditions derived from 30 cases, target recovery reaches 93.61 percent under patient-behavior pressure, 100.00 percent for temporal evidence, and 92.22 percent for triage red flags. An exploratory 100-case MedSAM comparison evaluates request-gated tool-interface feasibility. These results provide descriptive end-to-end evidence for the complete system on fixed evaluation suites, not causal evidence for an individual bank or clinical validation of the automatic judge. 

---
# Modalities Should Talk to Each Other: Dual-Stream Multimodal Learning for Long-Horizon Influenza Forecasting 

**Authors**: Seyed Mohammad Hossein Hashemi, Mohsen Hooshmand, Parvin Razzaghi  

**Link**: [PDF](https://arxiv.org/pdf/2608.23373)  

**Abstract**: Forecasting long-range influenza-like illness (ILI) matters for public health readiness. Publicly available surveillance datasets typically pair numeric epidemiological signals with textual information that is noisy, loosely structured, only indirectly related to near-term trends, and often lagged relative to the numeric signal. Fusing the two therefore requires careful design. We propose Dual-Stream Attention (DSA), a multimodal deep learning framework that forecasts 12-week-ahead ILI activity from a 36-week multimodal history by letting the numerical and textual streams condition each other. Using the Time-MMD health-domain dataset, DSA separately encodes the two modalities with a Transformer-based numerical encoder and a domain-adapted headline encoder, then couples them through a bidirectional Cross-Modal Attention (CMA) mechanism: the text (news headlines) conditions the interpretation of the numeric signal and vice versa. The CMA output then passes to a causal temporal model for forecasting. Evaluated across ten random seeds, DSA achieves a median test MSE of 0.416, versus 0.668, 0.607, and 0.851 for iTransformer, TaTS, and GPT4MTS, corresponding to mean-error reductions of 54.95%, 37.29%, and 67.23%, with paired Cohen's d of 0.555, 0.337, and 0.345, respectively, and ranks first in 100% of bootstrap draws. It also has substantially lower worst-window error than all baselines. On an external-geography dataset, DSA again ranks first among nine evaluated baselines. Ablations show the advantage does not depend on text-encoder choice or language-model fine-tuning, and that bidirectional attention outperforms either direction alone. Finally, perturbation-based faithfulness analysis shows the learned CMA is functionally informative under targeted masking, with a stronger effect in the text-to-numerical direction. 

---
# Walking on the DARKSIDE 

**Authors**: Aldo Gangemi, Emanuele Bottazzi  

**Link**: [PDF](https://arxiv.org/pdf/2608.23370)  

**Abstract**: Large Language Models (LLMs) recognise patterns but do not natively track the path of exclusions that a coherent discourse demands. When an input rests on a fabricated authority, a misapplied mechanism, or a surreptitious analogy, an unsteered LLM tends to engage with it as if it were grounded, and to reify the misstep into any structured output it generates. Logic-Augmented Generation (LAG) with POLANYI++, an LLM-steering method that uses heuristics, ontologies and problem solving methods for tacit knowledge extraction, produces an Extended Knowledge Graph (XKG) in OWL2, but inherits the same vulnerability: a sophisticated nonsensical input is reified into the graph alongside the legitimate triples, and is hardly detectable by automated reasoners since the XKG is generated jointly with the wrong assumptions. We introduce DARKSIDE, a coherence auditing method on top of POLANYI++. It formalises the trail as an explicit data structure of accumulated exclusions over discourse time, complemented by a warrant axis that classifies each named referent as Warranted, Unattested, Misattributed or Fabricated, with an escalation rule that pushes the DelegationRiskAssessment to UNSAFE when the fabricated rate is positive or the unsupported rate exceeds a threshold. We evaluate DARKSIDE as a steering layer over a Gemini 3 on BSBench, a 100-item adversarial corpus of sophisticated-sounding nonsense across software engineering, finance, healthcare, physics and law, with Claude Sonnet 4.6 as an independent judge. The empirical evidence supports an architectural claim: when an LLM forward pass is wrapped in an ontology-mediated negative-trail apparatus, the structural pattern-vs-path gap can be partially scaffolded. The XKG functions as the missing memory, and the warrant axis as an epistemic firewall. 

---
# Agent-G$^2$: Gaussian Guidance for Agentic Reinforcement Learning 

**Authors**: Zixuan Wang, Yanrui Miao, Zhengxi Lu, Teng Pan, Yiwen Qiu, Hongxing Li, Peng Qiu, Ruiqing Zhang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.23318)  

**Abstract**: Hint-based reinforcement learning addresses reward sparsity in long-horizon agentic tasks by retaining a prefix of an expert trajectory before each rollout, letting the policy explore from a state closer to success. Its effectiveness hinges on the guidance depth: how much of the trajectory to keep. Existing methods treat this depth as a deterministic scalar. Scheduled approaches share one value across samples and ignore per-task heterogeneity; per-sample probing estimates it separately at the cost of extra rollouts. We find that useful guidance occupies a band of depths whose informativeness profile is approximately Gaussian around the band center, rather than concentrating at a single optimal point. We propose Agent-G$^2$, a Gaussian guidance framework that draws the depth per task from a Gaussian whose center and spread are estimated online from rollouts already collected for policy optimization, requiring no probe rollouts or learned depth predictor. The center combines a global baseline with per-cluster difficulty, and the spread tracks within-cluster variance. We evaluate Agent-G$^2$ on ALFWorld and WebShop on Qwen2.5-1.5B / 7B-Instruct. Agent-G$^2$ outperforms the strongest hint-based, hint-free, and Aux-RL baselines on ALFWorld by 2.3 / 3.9 / 7.4 points at under one-third the rollout cost of per-sample probing. 

---
# EviSafe: Evidence-Grounded Safety Evaluation for Vision-Language Models 

**Authors**: Xuetong Li, Gaofeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23313)  

**Abstract**: Vision-language model safety benchmarks typically evaluate only final responses: whether a model refuses, warns, or complies. This outcome-level view cannot tell whether a model is safe for the right multimodal reason. Safelooking behavior may reflect keyword-triggered refusal, missed visual hazards, or over-refusal of benign-sensitive inputs. We introduce EviSafe, an evidence-grounded framework for VLM safety that jointly evaluates natural user-facing behavior, explicit grounding in textual and visual evidence, and behavioral sensitivity to counterfactual changes in safety-critical evidence. EviSafeBench instantiates the framework as a controlled benchmark with 1,181 gold image-text scenarios and 2,452 targeted counterfactual variants across eight safety domains and eight risk-source types. Each scenario includes a gold safety decision, evidence annotations, a safe-response policy, and counterfactual interventions. The three-probe protocol queries models with natural-response, evidencereporting, and counterfactual-response prompts, then scores them using an evidence-aware judge. Across eleven evaluated VLMs, natural severity accuracy ranges from 27.6% to 52.8%, relaxed diagnostic consistency from 6.1% to 29.3%, and unsafe-to-safe counterfactual transition success from 30.4% to 58.4%. These gaps show that the evaluated VLMs are not reliably safe for the right multimodal reason and motivate evaluation beyond refusal counts. 

---
# Apodex 1.1: Scaling Agentic Intelligence for Complex Work 

**Authors**: Apodex Team, B. An, B. Li, B. Wang, B. Zhang, B.L. Wang, C. Feng, C. Wei, C. Xue, C. Zhang, D. Ng, D. Ye, E. Min, F. Chen, F. Liu, F. Yang, F. Ye, H. Xu, H. Yang, H. Ye, H. Zhang, H. Zhao, J. Li, J. Lin, J. Xia, K. Jin, K. Wang, K. Yang, L. Bing, L. Lei, L. Su, Le. Wang, Lu. Wang, N. Wang, Q. Ren, Q. Yang, R. Li, S. Bai, S. Du, S. Li, S. Lin, S. Nie, S. Wang, S. Zhang, S.Z. Wang, Ta.Q. Fang, Ti.Q. Fang, W. Fang, W. Li, W. Zhang, X. Chen, X. Li, X. Tang, X. Wang, X. Xu, X. Zhang, X.Q. Wang, X.Y. Wang, Y. Deng, Y. Gao, Y. Hu, Y. Li, Y. Sui, Y. Wang, Y. Xiao, Y. Zhang, Z. Chen, Z. Cheng, Z. Feng, Z. Liang, Z. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23283)  

**Abstract**: General-purpose language models can reason and synthesize knowledge, but complex work also requires sustained interaction with files, information sources, and executable code, together with state maintenance, failure recovery, and verifiable delivery. We call this \emph{working capability}: sustained, verifiable progress toward a real-world objective. Apodex 1.1 develops this capability along two complementary dimensions. \emph{Environment Scaling} expands the diversity and verifiability of executable file, search, and code environments, while \emph{Agentic Coordination Scaling} trains agents to decompose long-horizon tasks, delegate parallel work, integrate asynchronous results, and replan. A shared execution harness and AgentOS maintain task state and provenance across tools and agents, and training turns environment trajectories and coordination traces into reliable behavior. Across complex professional work, finance, scientific research, mathematics, coding, and search, Apodex 1.1 reaches the leading performance band despite using a substantially smaller model than many frontier systems. The 35B-parameter Apodex 1.1 Mini further retains strong working capability in a locally deployable form. These results ground agentic intelligence in useful, verifiable work completed over time and advance our goal of building a \emph{Heavy-Duty Solver} for ambitious, long-running tasks. 

---
# Hidden in the Request: Explaining Unethical LLM Compliance through Token Relevance 

**Authors**: Or Biton, Tomer Krichli, Itai Allouche, Joseph Keshet  

**Link**: [PDF](https://arxiv.org/pdf/2608.23264)  

**Abstract**: Although Large Language Models (LLMs) are aligned to optimize for both helpfulness and harmlessness, these dual objectives may conflict, inevitably leading to alignment failures. This work systematically investigates instances where LLMs fail to exhibit ethical behavior. To understand the underlying mechanics of these vulnerabilities, we introduce a probing methodology that presents unethical scenarios to LLMs in three distinct structural modalities: objective classification tasks, subjective first-person statements, and direct requests for assistance. We find that model performance degrades in the request-for-assistance-based form. Using Layer-wise Relevance Propagation (LRP), we trace this discrepancy to an attribution bias: the model places greater emphasis on benign task-framing tokens (e.g., "Can you help me...") than on tokens signaling the underlying unethical behavior (e.g., "without getting caught"), which we term cue-tokens. We hypothesize that this under-attribution contributes to harmful compliance. To test this, we introduce two LRP-guided decoding methods that steer generation toward trajectories more relevant to cue tokens. Empirical evaluations show that these interventions promote safer responses, supporting cue-token attribution's role in compliance failures. 

---
# Automated Construction of FAIR Digital Object Knowledge Graphs from Flat Cultural Heritage Records 

**Authors**: Zeyd Boukhers, Lingxiao Kong, Xenophon Zabulis, Georgios Toubekis  

**Link**: [PDF](https://arxiv.org/pdf/2608.23263)  

**Abstract**: The FAIR Digital Object (FDO) framework mandates that metadata attribute values be expressed as persistent identifiers (PIDs) wherever possible, to produce a fully machine-actionable graph in which every reference is resolvable. The Europeana Data Model was designed long before the FDO specification, and it stores most metadata values as plain text. This serves human browsing well enough, but gives an automated agent nothing to follow across records or collections. We present a pipeline that transforms flat Europeana records into an FDO-compliant knowledge graph structured with CIDOC-CRM. Following the FDO specification, we model every heritage entity as a discrete FDO with its own PID, type, profile, and metadata layer. The core technical challenge is automating the FDO-prescribed distinction between values that must become PID references (resolvable entities) and those that may remain literals (terminal leaves such as notes, measurements, and dates). We address this with a large language model that classifies each metadata value, routes it to a controlled vocabulary (Getty AAT, Wikidata, VIAF, PeriodO), and links it to a shared entity FDO. We evaluate using 637 archaeological records from five Europeana providers, processing each with the LLM. The pipeline links 86% of metadata slots, resolving 58.5% of values Europeana had not already enriched. It also merges cross-lingual surface forms that byte-identical matching keeps apart, where 17 of 33 such merges are correct on manual review. Graph connectivity does not separate this from string matching; what distinguishes the FDO graph is that every node is typed and resolvable. 

---
# Is Next-Chunk Reasoning RL Really Better than SFT? Revisiting Training Strategies under no-CoT Data 

**Authors**: Yinhao Tang, Youqing Fang, Yanan Sun, Jiangning Liu, Ziyi Wang, Xun Zhao, Weiming Zhang, Bin Liu, Kuikun Liu, Wenwei Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.23256)  

**Abstract**: Recent work proposes next-chunk reasoning RL for leveraging no-CoT data---corpora such as worked solutions and textbook derivations that contain reasoning-rich content but lack explicit chain-of-thought annotations. The method trains a model to generate implicit reasoning traces and rewards them by their ability to predict the next chunk of text. While promising, existing evaluations primarily compare against conventional SFT baselines, leaving open whether the gains come from the RL formulation itself or from more effectively exposing the model to no-CoT data. We address this question with a controlled study of next-chunk reasoning RL and a simple but previously overlooked alternative: Mixed SFT, a single supervised fine-tuning stage that jointly trains on no-CoT and long-CoT data. Despite its simplicity, Mixed SFT achieves a clearly higher post-RLVR performance ceiling than next-chunk reasoning RL while requiring over 60 times less training compute. The advantage is consistent across in-domain mathematical reasoning and out-of-domain reasoning tasks. Moreover, we show that higher pre-RLVR accuracy does not necessarily translate into higher post-RLVR accuracy, highlighting the need to evaluate no-CoT training strategies in the context of the full post-training pipeline. 

---
# What is mathematics now, and what should it be? 

**Authors**: Jeremy Avigad  

**Link**: [PDF](https://arxiv.org/pdf/2608.23218)  

**Abstract**: Advances in neural theorem provers have been impressive, but the successes obscure a broader vision of what AI can do for mathematics and how mathematicians can engage with AI. This essay advances a more expansive and optimistic point of view. 

---
# Cognitive Profiling of LRMs' Reasoning Traces Using Bloom's Taxonomy 

**Authors**: Maria-Eleni Zoumpoulidi, Georgios Paraskevopoulos, Alexandros Potamianos  

**Link**: [PDF](https://arxiv.org/pdf/2608.23205)  

**Abstract**: Large Reasoning Models (LRMs) have revolutionized reasoning in LLMs, and the increasing public availability of reasoning traces creates valuable opportunities to study model behavior not only at the surface level but also at the granularity of individual reasoning steps. However, understanding the types of thinking employed during reasoning - which offers critical insights into models' reasoning patterns and enables actionable applications - remains underexplored. To address this gap, we introduce a framework for automatic annotation of reasoning steps through the lens of Bloom's Taxonomy, which classifies thinking into six cognitive levels, such as Remembering, Applying and Evaluating. Using this framework, we perform a large-scale analysis across models and datasets, revealing both similarities and differences in thinking patterns across models and tasks. Moreover, we demonstrate that thinking-type information derived from reasoning traces correlates with correctness, paving the way for improved reasoning. Our findings establish a fine-grained framework for analyzing thinking patterns in LRMs and provide actionable insights for enhancing reasoning quality. 

---
# AI emotional support is better only when chosen, but shifts preferences even when it is not 

**Authors**: Yaoxi Shi, Cathy Mengying Fang, Guy LabanPattie Maes, Amit Goldenberg  

**Link**: [PDF](https://arxiv.org/pdf/2608.23196)  

**Abstract**: People increasingly face a novel decision when seeking emotional support: human or AI. In existing studies, AI's empathic messages are rated as well as or better than humans'. But these studies either assigned the support source or honored people's choice. In real life, support is often incongruent with choice, as people want one source and receive the other. Across three experiments (N = 1,951), participants chose whether to share an emotional experience with a human or an AI, then were randomly assigned to a congruent or incongruent partner. AI support was rated as superior only among those who had chosen it. Yet regardless of congruence, interacting with AI increased willingness to choose it again. In a 28-day study with OpenAI (N = 981), daily conversations shifted preferences toward AI and away from humans, but only when conversations turned personal. Emotional support choices are thus path-dependent, progressively redirecting away from human connection. 

---
# Jiuge-Tuiqiao: An Interpretable Human-AI System for Classical Chinese Poetry Refinement 

**Authors**: Yufeng Han, Lifan Deng, Cunliang Kong, Wenhao Li, Xin Cong, Yuzhuo Bai, Kangyang Luo, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.23098)  

**Abstract**: Classical Chinese poetry composition has long valued Tuiqiao, the iterative refinement of words, imagery, and prosody. However, many current AI poetry systems follow a one-shot generation paradigm, which reduces users to prompt providers and weakens their creative agency. We present Jiuge-Tuiqiao, an interactive human-AI collaborative system for classical Chinese poetry composition. The system is designed around a triadic model: user-driven control, ancient-guided evidence, and AI-assisted generation. Users can lock characters or lines, receive real-time prosody feedback, and obtain interpretable refinement suggestions grounded in high-frequency collocations, PPL-ranked classical lines, and structured knowledge extracted from classical encyclopedias. This design turns AI from an autonomous generator into a background assistant that supports the user's own process of poetic refinement. Preliminary experiments and user feedback suggest that Jiuge-Tuiqiao improves controllability, interpretability, and user engagement in classical poetry composition. 

---
# POOL: Propagated Uncertainty Over Lookalikes 

**Authors**: Rounak Sharma, Ananya B. Sai, Soumyabrata Pal  

**Link**: [PDF](https://arxiv.org/pdf/2608.23086)  

**Abstract**: Black-box large language models need confidence scores that can separate likely-correct from likely-incorrect outputs, enabling systems to prioritize human review, route uncertain cases to stronger models, or choose abstention thresholds on development data. Yet existing confidence estimators face a cost-quality trade-off: verbal confidence is cheap but is often overconfident, while sampling-based uncertainty is more informative but scales linearly with the number of samples per query. We propose \textsc{POOL} (\emph{Propagated Uncertainty Over Lookalikes}),a cost-efficient framework that addresses this trade-off taking inspiration from group-testing.\textsc{POOL} clusters query stems with overlaps, evaluates a base estimator on representative medoids, softly propagates confidence scores to nearby queries, and selectively evaluates high-disagreement cases. We instantiate this framework with \textsc{Hy@}$p$, a hybrid estimator that combines verbal confidence with spectral answer diversity computed from the negative von Neumann entropy of sampled answer this http URL six domains from three datasets and five black-box LLMs, \textsc{Hy@}5 achieves higher average AUROC than verbal confidence and \textsc{Vn@}10 sampling while using half as many samples as \textsc{Vn@}10. \textsc{POOL}-\textsc{Hy@}5 retains 93.5--97.9\% of its AUROC while saving 19.3--39.3\% of generations. On paraphrase-dense workloads, generation savings rise to 73-76\%, showing that semantic redundancy can be leveraged to lower confidence-estimation costs. 

---
# AgentWeave: Routing Before Reasoning for Efficient Function Calling in Tool-Rich Language Models 

**Authors**: Saurav Singla, Aarav Singla, Advik Gupta, Parnika Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2608.23078)  

**Abstract**: Large language models increasingly operate over large collections of tools, functions, APIs, and specialized agents. As the candidate action space grows, a function-calling model must process more schemas, consume more prompt tokens, and distinguish among increasingly similar or irrelevant alternatives. We study a complementary systems strategy: reduce the candidate set before language-model inference while leaving the downstream model unchanged. We introduce AgentWeave, a deterministic pre-inference routing layer that constructs a bounded model-visible action space using eligibility, requirement, capability, and routing signals. We evaluate AgentWeave with a frozen BFCL-derived routing-pressure protocol using the public MadeAgents/Hammer2.1-1.5b model. On 48 fresh BFCL V4 multiple-function tasks, AgentWeave achieves 6/48 (12.5%) native BFCL successes, whereas all-tools, deterministic random top-8, and semantic top-8 baselines each achieve 0/48. The paired success difference is +12.5 percentage points with a 10,000-resample paired bootstrap 95% confidence interval of +4.17 to +22.92 points and exact McNemar p=0.03125. Relative to all-tools exposure, AgentWeave presents 70.18% fewer tools, uses 61.70% fewer input tokens, and exhibits 50.95% lower mean local-model latency. The result is deliberately narrow: this is a BFCL-derived routing-pressure study rather than an official full BFCL leaderboard score, and absolute task success remains low. The evidence nevertheless shows that candidate-space construction can materially affect a fixed model's function-calling behavior and motivates evaluating routing as a distinct stage before model reasoning. 

---
# From Generation to Simulation: How Far Are World Models from Being True Simulators? 

**Authors**: Tong Wang, Huan Deng, Mucheng Yang, Yang He, Xiaohui Kuang, Gang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.23070)  

**Abstract**: With the rapid progress of diffusion models and large-scale video generation, generative world models are increasingly expected to replace traditional simulators, including physics engines, game engines, and reinforcement-learning environments. Yet the remaining distance from generation to simulation lacks a systematic assessment. We present a capability-based study using an external yardstick: eight capabilities of a traditional simulator, namely asset construction, physics engine, interaction, controllability, stability, state feedback, diversity, and evaluation metrics. We trace three main technical routes--latent dynamics, video generation, and joint-embedding prediction--and map exactly 200 representative works published from 2018 to June 2026 onto these capabilities. Our analysis shows that world models have achieved functional substitution in interaction and controllability for specific scenarios, but remain short of traditional simulators in formal guarantees of physical laws, structured state feedback, and reproducible long-horizon evolution. State feedback is the most neglected cross-route shortcoming: only 6 of 163 implementation papers expose a runtime interface for querying entity states or physical parameters. We identify six research directions: formalized physics, a unified action interface, first-class state feedback, long-horizon stability, downstream-utility evaluation, and cross-route hybridization. Project page: this https URL 

---
# Improving O-RADS Risk Stratification from Ultrasound Reports: A Comparative Evaluation of Hybrid versus End-to-End LLM Reasoning Strategies 

**Authors**: Xiaotong Tan, Chunli Qiu, Xin Liu, Qing Huang, Guangli Zhou, Bo Gao, Xiaoyan Song, Shuyan Wang, Xiuqin Wang, Wufeng Xue, Ruobing Huang, Dong Ni, Guowei Tao, Jun Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.23061)  

**Abstract**: Background: Automating clinical guideline-based decision-making with large language models (LLMs) remains challenging because of reliability, hallucination, and limited interpretability. We compared the performance of LLMs and reasoning strategies for automated Ovarian-Adnexal Reporting and Data System (O-RADS) classification from free-text pelvic ultrasound reports. Methods: In this retrospective study, consecutive patients with ovarian masses who underwent pelvic ultrasound were included. Eight LLMs were tested with three reasoning strategies: implicit-knowledge end-to-end, rule-informed end-to-end, and a feature-based hybrid architecture that decoupled feature extraction from rule-based classification. The reference standard was O-RADS categorization established by expert consensus. Results: A total of 310 women with 390 ovarian masses were evaluated. The feature-based hybrid architecture using Gemini 3.6 Flash demonstrated the best performance, achieving an accuracy of 99.2% (387 of 390) and almost perfect agreement with the reference standard (weighted kappa = 1.00; 95% CI: 0.99-1.00). Its performance surpassed that of original clinical reports (accuracy, 87.7% [342 of 390]; weighted kappa = 0.94; 95% CI: 0.91-0.96) and end-to-end LLM strategies (accuracy range, 65.6% [256 of 390] to 95.9% [374 of 390]). For structured feature extraction, Gemini 3.6 Flash demonstrated higher overall accuracy than Claude Fable 5 (98.9% vs 97.8%; P < 0.001). The hybrid architecture reduced misclassification errors and mitigated the overstaging tendency observed in original reports. Conclusion: The feature-based hybrid LLM architecture that separates clinical feature extraction from deterministic guideline execution enables highly accurate, reliable, and interpretable automated O-RADS classification, providing a promising approach for standardized, guideline-based clinical decision-making. 

---
# LLM-based Agents for Forecasting and Prediction: Methods, Training, Evaluation, and Applications 

**Authors**: Xiaogang Xu, Jiaqi Tang, Jianmin Chen, Yingying Yan, Zhenchao Tang, Xiangxin Zhou, Xiaobin Hu, Wei Wei, Jinfeng Wu, Qifeng Chen, Lu Zhou, Jiafei Wu, Zhe Liu, Jianwei Yin, Weimin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.23058)  

**Abstract**: Large language models (LLMs) now support forecasting systems that combine language-based reasoning with temporal data, evidence retrieval, external tools, and iterative prediction. We investigate LLM-based forecasting agents, meaning systems in which a language model contributes to a scored prediction about a future or currently unobserved target. We organize architectures into three groups. Standalone LLM workflows operate on encoded time series or event context. Tool- and retrieval-augmented agents incorporate external evidence. Hybrid systems pair LLMs with statistical or foundation models. We then review training methods and evaluation protocols. We examine negative as well as positive evidence, including sensitivity to small input perturbations, ablations in which the LLM component does not improve accuracy, and benchmark gains that may reflect contamination instead of temporal reasoning. We cover applications in finance, weather, health, energy, and operations, and we summarize the benchmarks and datasets used for evaluation. The evidence indicates that measurement is a central limitation. Future work requires calibration under distribution shift, contamination-resistant live evaluation, explicit reporting of cost and accuracy together, and methods for handling feedback between deployed forecasts and the outcomes being forecast. 

---
# From Inertia to Objectivity: Improving Deep Research Agents with Noise Isolation 

**Authors**: Xiangxin Zhang, Zhanwei Zhang, Zhihang Fu, Binbin Lin, Wenxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23045)  

**Abstract**: Web search agents powered by Large Language Models (LLMs) show strong promise, but deep research tasks expose a recurring failure mode: once an agent has produced a query, plan, or intermediate conclusion, it becomes less objective when later judging the consequences of that same action. We term this phenomenon \textbf{inertia bias}. To make it measurable, we introduce the IBIS benchmark, which controls the search observations while varying whether the model is evaluating the outcome of its own prior action. We find that models are substantially worse when they ``own'' the preceding search step, showing that self-authored action history can systematically distort subsequent judgment. We further show that this bias propagates into two forms of system-level degradation: search noise at the worker level and contextual noise at the manager level. To address this problem, we propose NIS-Agent, which applies context isolation at the two decision points most vulnerable to inertia bias: webpage triage and final-answer validation. Across GAIA, WebWalkerQA, BrowseComp, and BrowseComp-zh, NIS-Agent achieves competitive performance while reducing token cost by 33\% compared to our baseline. We further train an 8B model to be intrinsically more resistant to inertia bias; under the same NIS-Agent framework, it attains average performance comparable to GPT-4o on deep research benchmarks. 

---
# AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces 

**Authors**: Sungho Park, Wonjoong Kim, Rongyuan Tan, Jue Zhang, Wook-Shin Han, Pengfei Gao, Chanyoung Park, Yongqiang Yao, Rao Fu, Elsie Nallipogu, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23041)  

**Abstract**: LLM agents remain unreliable on long-horizon tasks, where small local failures can compound over extended interactions and lead to overall task failure. Although external harnesses can substantially improve robustness, harness design remains a manual and expensive process that requires searching over a large space of prompts, tool configurations, and control logic. We propose AutoSaddler, an automatic harness optimization framework that formulates harness improvement as an offline learning problem and iteratively updates the harness using failure signals from mini-batches. AutoSaddler combines failure-trace diagnosis, structured patch generation that treats the harness as code, and validation-based update selection. Experiments on GAIA2, SWE-Bench Pro, and Terminal-Bench 2.0 show that AutoSaddler substantially improves agent performance over the corresponding base harnesses, achieving gains of 9.0, 9.6, and 10.0 percentage points, respectively. Ablation studies further suggest that effective harness optimization benefits from three ingredients: deep debugging rather than shallow reflection, targeted modifications rather than unconstrained editing, and generalization-aware selection rather than trajectory-specific repair. Together, these results suggest that automatic harness optimization is a promising path toward more performant and reliable agent systems. 

---
# MobilePA-Bench: Benchmarking Mobile Planner Agents on Complex Real-World Tasks 

**Authors**: Yi Zhu, Xiongwei Wu, Qiyi Wang, Tingyu Qu, Jiajun Liu, Sihan Cao, Long Chen, Weigao Sun, Feida Zhu, Yiran Zhong, Steven Hoi  

**Link**: [PDF](https://arxiv.org/pdf/2608.23035)  

**Abstract**: As on-device LLM agents evolve into personal copilots, the mobile operating system has become a key testbed for this paradigm, making rigorous capability evaluation essential. Yet existing benchmarks fall into two camps, each with a critical blind spot: GUI-centric benchmarks test surface-level screen manipulation while overlooking background tool use and long-horizon planning, whereas static function-calling benchmarks rely on offline API matching that is detached from real runtime constraints. To close this gap, we present \textbf{MobilePA-Bench}, an interactive, stateful, and tool-centric benchmark for evaluating the tool-calling and planning abilities of mobile planning agents. MobilePA-Bench runs on an executable sandbox that maintains live application databases and returns structured feedback, spanning $13$ functional domains and $212$ realistic mobile tools. Beyond basic tool use, it evaluates a central planning agent along three advanced dimensions: \emph{(1)~Sub-agent Collaboration}---decomposing a complex task and delegating specialized work to capable sub-agents; \emph{(2)~Memory Usage}---recalling stored memories, user profiles, and past preferences to resolve implicit requests; and \emph{(3)~Skill Usage}---invoking pre-packaged composite skills instead of planning every step from scratch. Extensive experiments show that current frontier LLMs remain unreliable in mobile settings: performance drops sharply under strict tool ordering, permission limits, and unexpected runtime errors. By pairing an interactive function-calling sandbox with evidence-based verification, MobilePA-Bench serves as both a practical diagnostic benchmark and an interactive foundation for agentic reinforcement learning---accelerating the development of dependable mobile agents. 

---
# Artificial Empathy: Towards a Framework for Unsupervised Agency Detection and Policy Reconstruction 

**Authors**: Peter Kuhn, Chris Pang, Sonakshi Chauhan  

**Link**: [PDF](https://arxiv.org/pdf/2608.23030)  

**Abstract**: We study how an AI system can identify and model other agents in its environment from observation alone, which is a capability necessary for cooperative behaviour in the real world. This problem is less constrained than inverse reinforcement learning and remains largely unexplored. We propose a framework that uses a reinforcement learning agent, trained on an independent task as a prior about agentic dynamics, to perform agency detection and policy reconstruction. 

---
# PsychJail: Exploring Psychological Jailbreaks via Multi-Turn Persuasion of LLM Policies 

**Authors**: Zeyu Feng, Qingyu Wu, Yuzhe Luo, Hua Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.23028)  

**Abstract**: Large language models (LLMs) are increasingly deployed in education, healthcare, policy advising, and other interactive settings, where users engage them as sustained social interlocutors rather than one-shot query engines. This shift makes jailbreaks a growing safety threat, yet most research emphasizes single-turn prompt optimization or iterative attack refinement, leaving psychologically grounded multi-turn vulnerabilities underexplored. We present PsychJail, a psychology-guided framework for red teaming aligned LLMs through theory-grounded, multi-turn persuasion. PsychJail maps established social-psychological persuasion techniques into a tactic-conditioned attack policy. It factorizes each attacker action into a Change-of-Meaning analysis, tactic selection, and victim-visible message, operationalizing the Persuasion Knowledge Model (PKM). The policy is refined with trajectory-level reinforcement learning using a PKM-gated reward that credits early jailbreak success only when every turn contains a well-formed Change-of-Meaning analysis. Across four aligned victim models, PsychJail achieves the highest average attack success rate (87.3%) and outperforms strong single-turn and multi-turn baselines on every model. We also measure susceptibility at the action that breaks each victim, revealing four distinct model-level fingerprints that identify which persuasion levers affect each model and how broadly. These fingerprints help explain cross-model transfer asymmetry. We interpret them as four candidate psychological profiles-rationalist, credibility-driven, narrative-monoculture, and broadly persuadable-while treating this interpretation as a conjecture requiring future validation. Our findings establish psychological jailbreaks as a distinct red-teaming frontier for increasingly interactive LLMs. 

---
# PatchWrite: One Line, Not One Section -- Compile-Gated, Validity-Preserving Editing for AI-Drafted Manuscripts 

**Authors**: Weiwei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23001)  

**Abstract**: Automated manuscript pipelines often regenerate an entire section to repair a local defect, allowing unrelated metrics and citations to change even when the resulting PDF still builds. PatchWrite instead constrains how candidate edits become committed manuscript states: it reuses bounded EDIT N M editing and rollback, but tightens compilation acceptance with fatal-log checks and adds evidence locks that require every cited key and experimental numeric token to be attested by a reference registry or experimental log. Candidates that fail either check are rejected and the previous HEAD is retained. On a 24-manuscript x 8-fault oracle stress test (768 jobs, evenly split between compile-breaking and content-only faults), whole-slot rewriting mutated an unrelated "12-layer" line in every case (0/192 preserved; numeric Jaccard 0.6667), whereas PatchWrite preserved it in 192/192 cases. Removing the compile gate reduced acceptance to 0, while removing the evidence gate allowed a hallucinated citation to pass. The same pattern held across all eight faults. To test the protocol with generation rather than oracle edits, we reran the 192 jobs with the writer model proposing the edits. The model's candidates were accepted in 75% of cases; nearly all rejections came from one reproducible failure mode in which the model attempted to delete a line using an empty replacement unsupported by the current grammar. Every accepted candidate passed both gates, and 93.75% fixed the injected fault; the remaining cases involved a technically valid but sentence-inappropriate citation and one markup-changing near-miss. In a blind evaluation of sixteen PDF pairs, both raters preferred PatchWrite for preserving lab-grounded facts (C1 Likert 5.0 vs. 2.0), while rating prose quality nearly identically. Logs from 193 in-product drafting tasks show the same classes of failures occurring in practice. 

---
# SA-RSQ: A Versatile Sparse Representation Framework for Multi-modal Recommender Systems 

**Authors**: Xiang Wang, Shigang Quan, Tingzhen Chang, Kang Yang, Sitong Chen, Yabo Fan, Xingxing Wang, Zhaodian He  

**Link**: [PDF](https://arxiv.org/pdf/2608.22979)  

**Abstract**: Deploying high-dimensional multimodal features in industrial recommender systems incurs substantial storage and latency overhead. Hard quantization is compact but introduces boundary distortion, whereas dense soft quantization couples representation quality to the limited storage budget. We propose Sparse Activation-based Residual Soft Quantization (SA-RSQ), which uses Top-K sparse routing and softmax weights to store compact (Index, Probability) tuples. The stored tuples decouple per-item storage from codebook dimensionality; for a fixed selected support, gradients propagate through the routing weights and weighted reconstruction without relying on a straight-through estimator. Experiments on a proprietary food-delivery advertising dataset show favorable reconstruction-performance and CTR trade-offs across storage budgets of 8-48 bytes per item. A preliminary Next-Distribution Prediction study and a one-week online A/B test further demonstrate the practical potential of SA-RSQ, with relative lifts of +2.51% in CTR and +3.66% in CPM. 

---
# Budget-Constrained Embodied Perception: Four Resource Walls and a Pre-Registered Evaluation of Access-Structured Perception on Open Models at less than 31B 

**Authors**: Defu Lin, Wenhui Chen, Ziyao Lin, Jianlin Chen, Peiji Long, Chi Man Vong  

**Link**: [PDF](https://arxiv.org/pdf/2608.22975)  

**Abstract**: Embodied multimodal agents must answer from growing observation streams under a fixed per-decision token budget. We formalize this constraint through four resource walls: a perceptual Shannon wall for bounded state, a horizon wall for query-independent frame selection, a round wall for non-adaptive retrieval, and a conditional composition wall for fixed-depth inference. We introduce ASP, a training-free wrapper for frozen multimodal models that combines a capped structured state, a verbatim episodic index, and query-conditioned budget allocation with iterative access. Following a pre-registered protocol, we evaluate seven open-weight models from 3B to 31B on SEW-Bench, a license-free synthetic long-horizon walkthrough benchmark constructed to instantiate these walls. The registered natural-video benchmarks were not run because their frames require dataset agreements; our evidence therefore concerns access mechanisms, not natural-scene perception. Under a 4,096-token decision budget, ASP reaches 75 to 94% episodic retrieval accuracy, compared with 3 to 19% for equal-budget query-independent sampling, and budget reallocation outperforms quadrupling the sampling budget on every backbone. However, the full three-component architecture does not validate channel duality: removing the compressive state raises the flagship mean from 35.4 to 58.0, ASP does not outperform the verbatim-only baseline on any backbone, and two of four pre-registered falsification criteria fire. These results show that query-conditioned access, rather than parameter count or context growth alone, is decisive under a fixed budget, while prompted online compression does not earn its cost in this setting. 

---
# Toward Effective and Reliable LLM Agents via Dynamic Ontology 

**Authors**: Xiaohui Zhang, Zequn Sun, Chengyuan Yang, Yuanning Cui, Lingbing Guo, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.22974)  

**Abstract**: Large language model (LLM) agents rely heavily on knowledge encoded in model parameters or presented as unstructured context. In domain-specific tasks, this leaves important semantic connections implicit. This often results in incomplete evidence use and brittle multi-step decisions. Ontologies offer a way to externalize domain concepts and relations as machine-interpretable structures, but constructing task-usable ontologies traditionally requires substantial effort from domain experts and is difficult to scale. Automatic construction is also challenging: an ontology that appears semantically plausible may not contain the relational structures needed for actual decision making. We present OaK, an ontology-as-a-kernel framework that dynamically constructs and refines task-oriented ontologies for LLM agents. Given task requirements and training data, OaK constructs an ontology and its knowledge graph, generates task-adaptation functions for graph reasoning, and uses judge feedback to iteratively refine both. By making relevant concepts and relations explicit, the ontology grounds knowledge retrieval and multi-step decision making. We evaluate OaK on TravelPlanner, CRMArenaPro, and ToolQA. Results show that OaK improves standard LLM agents, strengthens evidence grounding, and boosts the reliability of multi-step reasoning. 

---
# ParallelWorld: Test-Time Scaling for Embodied Reasoning 

**Authors**: Min Chen, Shengjun Zhang, Yuxin Li, Zhang Zhang, Xin Fei, Chong Xia, Yueqi Duan  

**Link**: [PDF](https://arxiv.org/pdf/2608.22971)  

**Abstract**: Embodied Reasoning constitutes a fundamental capability of embodied intelligence, serving as the basis for autonomous perception, reasoning, and interaction within physical environments. Recent studies have shifted the paradigm of embodied reasoning from static perception toward dynamic exploration, where agents acquire task-relevant information through interactions with the environment. However, existing active reasoning approaches generally generate exploration trajectories incrementally without long-horizon planning. Even recently emerged test-time scaling frameworks often resort to myopic, single-step lookaheads, which struggle to resolve the delayed feedback inherent in complex, occluded spatial environments. To address this limitation, we propose ParallelWorld, a multi-horizon test-time scaling framework for embodied reasoning. Instead of greedy, single-step trials, ParallelWorld empowers agents to simulate and evaluate multi-step future trajectories in parallel before committing to an action. Specifically, we introduce a verifier-guided tree-search paradigm. Starting from the current state, ParallelWorld branches into multiple parallel trajectories and rolls them out continuously across a multi-step horizon. At each simulation step, a verifier agent evaluates the intermediate state transitions, dynamically pruning unpromising branches and prioritizing paths with the highest information gain. Once the multi-step prospective simulation is complete, the agent synthesizes the long-horizon outcomes to commit to the optimal action sequence. Finally, an answer agent performs reasoning over the selected trajectory to produce the final reasoning. Extensive experiments on ESI-Bench demonstrate that ParallelWorld consistently improves active perception and reasoning performance. 

---
# Buried in Textual Debt: Context Pruning with Visual Evidence Preservation for MLLM Agents 

**Authors**: Yuchen Huang, Sijia Li, Jun Zhang, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2608.22963)  

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly deployed as multi-step agents, where explicit reasoning supports task decomposition and tool coordination but also accumulates self-generated text. Over long trajectories, this text can dominate the context and suppress visual evidence, creating textual debt. We observe that reasoning becomes redundant once task-relevant visual evidence is grounded, while stale hypotheses can misguide later inference when grounding remains uncertain. Pruning must therefore remove redundant text without discarding visual evidence. We propose SPARE, a Kullback--Leibler (KL)-guided framework for pruning accumulated reasoning in multimodal tool-use agents. SPARE uses a compact task-state summary as privileged diagnostic context. For each candidate segment, it replays the same model under the original and summary-conditioned contexts. Reverse-KL divergence from on-policy self-distillation (OPSD) then tests whether the summary sufficiently covers the segment without disrupting future reasoning. We further fine-tune the summarizer with supervised fine-tuning (SFT), enabling more compact summaries, broader coverage, and more aggressive pruning. Across multi-step visual tool-use benchmarks, SPARE achieves the highest average accuracy among pruning methods while removing 37.89--64.58\% of reasoning tokens. This favorable accuracy--context trade-off shows that reducing textual dominance restores reliance on visual evidence and mitigates over-conditioning on self-generated language. 

---
# What Process Evaluation of Coding Agents Actually Measures: Action, Task, and Step Are Three Different Levels 

**Authors**: Jiawei He, Mengyu Shi, Jie jia, Xikai Yang, Dong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.22960)  

**Abstract**: Coding agents are increasingly evaluated not only by whether they solve a task, but also by how they execute it. However, existing process-level evaluations often treat action prediction, task uncertainty, and step attribution as if they were the same problem, which makes it unclear what such evaluations actually measure. In this paper, we introduce a measurement framework for process evaluation in coding agents and instantiate step-level causal attribution with SCAE, a replay-based estimator derived from a structural causal model of agent execution. Our framework combines prefix-conditioned identification, replay/intervention-based estimation, and controlled judge-information manipulation to study process evaluation at the action, task, and step levels. Experiments on 499 file-localization episodes from 12 repositories show that next actions are driven primarily by execution provenance rather than code-graph transitions, execution uncertainty is structured at the task rather than step level, and full-trace judges exhibit systematic collider bias, suggesting that current process evaluation often measures semantic relevance rather than certified causal contribution. 

---
# Concepts for Securing Agentic AI Coding and the Terok Environment 

**Authors**: Jiří Vyskočil, Franz Pöschel, Andreas Knüpfer  

**Link**: [PDF](https://arxiv.org/pdf/2608.22930)  

**Abstract**: Agentic AI is a fascinating new tool for software development. It is a huge step forward compared to "conventional" AI assisted coding, which in turn was a considerable breakthrough earlier. AI support through LLMs is a young and very fast-moving field. The "conventional" (non-agentic) flavor became useful and productive in early 2025 (around 18 months ago) and the agentic flavor followed in fall 2025 (approximately 9 months ago). Besides all its benefits and potential, it also carries some fundamental risks for IT security. And the agentic approach added very severe risks while making others much more dangerous.
With all the motivation to explore this fascinating new tool we should not ignore the risks but actively address them. We present (I) an assessment of the IT security risks, (II) a concept for mitigating them without breaking its benefits, and (III) an overview about an implementation of our concept. In this very dynamic field this is likely not the final and once-and-for-all answer to the identified issues but still a substantial step forward in responsible usage of Agentic AI for software development. It should also be a contribution to the community to allow early and eager evaluation of the potential of agentic AI for software development without actually suffering from its implied IT security risks. 

---
# Beyond Observed Auxiliary Relations: Environment-Conditioned Modeling for Multi-Behavior Recommendation 

**Authors**: Seunghan Lee, Hyunsik Yoo, Jian Kang, Susik Yoon, SeongKu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22920)  

**Abstract**: Multi-behavior recommendation (MBR) leverages auxiliary behavioral signals, such as clicks and add-to-cart, to enhance target behavior prediction like purchases. While recent graph neural network-based approaches have achieved strong performance by systematically propagating auxiliary behavior signals, they still suffer from two fundamental challenges inherent to auxiliary behaviors: (1) missing auxiliary signals, which hinder generalization to items without auxiliary observations, and (2) unreliable auxiliary signals, which amplify noise misaligned with the target behavior. To address these challenges in a unified manner, we propose BOAR, an environment-conditioned MBR framework that addresses missing and unreliable auxiliary signals through two complementary modules conditioned on auxiliary observability. Extensive experiments demonstrate that BOAR consistently outperforms state-of-the-art baselines, achieving up to 7.82% gains in HR@10 overall and up to 44.2% gains for target items without auxiliary observations, highlighting its ability to capture hidden preferences beyond observed auxiliary relations. Our code is available at: this https URL. 

---
# CDEG: Learning Decision-Critical Evidence for Long-Horizon Diagnostic Agents 

**Authors**: Xiwei Dai, Zijie Meng, Zhiting Fan, Yixuan Tang, Ziru Niu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.22899)  

**Abstract**: Unlike static medical question answering, long-horizon diagnosis captures the sequential nature of clinical practice: evidence is progressively acquired, integrated, and evaluated over multiple rounds of interaction before reaching a final diagnosis. However, existing doctor agents often fail when critical evidence is either not acquired or not adequately incorporated into diagnostic reasoning. Recent agentic approaches attempt to address these failures by reusing historical trajectories or distilled memories. But their diagnostic gains remain constrained because such experience may contain noisy or incidental information and is typically reused without validating which evidence actually drives diagnostic decisions. To address this limitation, we introduce CDEG, a graph-based framework that learns reusable decision-critical evidence from historical diagnostic trajectories. CDEG contrasts successful and failed trajectories from the same case to identify candidate evidence, validates their diagnostic impact through controlled counterfactual interventions, and organizes the resulting diagnosis--evidence--action relations into a structured graph. During inference, CDEG tracks the evolving patient evidence state to retrieve relevant diagnostic relations and selectively guide missing evidence acquisition or overlooked evidence reappraisal. Across in-domain and out-of-distribution benchmarks with multiple doctor agent backbones, CDEG consistently improves diagnostic performance, achieving up to an 11.5% accuracy gain over vanilla agents. These results demonstrate that reliable long-horizon diagnosis requires moving beyond trajectory-level experience reuse toward evidence-level learning of the factors that truly shape clinical decisions. 

---
# Proxy reliance in large language model decisions is uncalibrated to predictive evidence 

**Authors**: Zengqing Wu, Chuan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.22887)  

**Abstract**: Large language models (LLMs) are entering decisions in triage and lending, where task-relevant inference must be distinguished from impermissible proxy use. Current audits ask whether decisions change when demographics change. But attributes correlated with a protected group carry predictive value, so a changed decision can be discrimination or sound inference. We measure causal proxy effects in four LLMs on a clinical-ranking task with known ground truth, where the reliance the evidence warrants can be computed exactly and used as the reference. One audit signal yields three verdicts: over-reliance, warranted and under-reliance. Under neutral labels every model relies on proxies with no information. Informative proxies draw all three. Social field names push reliance down, below the reference in one model. Two findings explain this. Reliance severely undertracks the evidence, and social-label suppression is fragile, since in-context examples raise it above zero in every model. Accuracy-based evaluation detects none of this. 

---
# Your AI, On a Dial: Controlling Investment Bias in LLMs with a Single Neuron 

**Authors**: Sahong Park, Suhwan Park, Hoyoung Lee, Gakyung Kwon, Wonbin Ahn, Jaewon Choi, Alejandro Lopez-Lira, Yoon Kim, Chanyeol Choi, Hyeongwoo Kong, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.22852)  

**Abstract**: Large language models (LLMs) are increasingly used in investment decision-making, yet prior work shows that they exhibit systematic, model-specific investment preferences. We study whether a model's overall investment stance can be calibrated to a specified direction and strength. We introduce an investment-bias dial, an inference-time intervention on a single neuron that continuously adjusts a model-level decision prior---its overall tendency toward buying or selling---without targeting specific firms or investment attributes. Using matched positive and negative evidence, we evaluate five open-weight LLMs and find that the dial produces monotonic changes in investment stance without modifying prompts or model parameters. At the response level, the dial shifts both investment decisions and the evidential emphasis of generated rationales under identical inputs. In an agentic retrieval setting, the dial also changes what information the model searches for, which evidence it selects, and which evidence is reflected in its final analysis. In a long-context evaluation, the dial maintains stable stance control as context length increases, whereas a matched system-prompt instruction progressively attenuates. We further show that changes in the dial propagate to security rankings and downstream portfolio composition in an exploratory backtest. Overall, our results show that an LLM's aggregate investment stance can be calibrated toward a specified target at inference time. 

---
# GSAR: Goal-State-Anchor Rewards for Mobile GUI Agents with Self-Evolving Data Synthesis 

**Authors**: Long Zhang, Yuhan Chen, Chaoran Zhang, Wanxia Cao, Kun Huang, Pengzhi Gao, Wei Liu, Jian Luan, Chenliang Li, Lixin Zou  

**Link**: [PDF](https://arxiv.org/pdf/2608.22847)  

**Abstract**: Vision-Language Models (VLMs) based GUI agents stand to benefit significantly from online reinforcement learning (RL). However, their training is bottlenecked by two fundamental issues: current data synthesis methods for GUI Agents rely on specific environments and struggle to generate diverse data, while existing evaluators either suffer from limited scalability or provide inaccurate and unreliable reward signals. To overcome these challenges, we introduce GSAR (Goal-State-Anchor Reward), a RL reward framework that supports scalable task generation and delivers reliable reward signals for stable and efficient policy optimization. Our approach features self-evolving data synthesis, which produces multiple environments through task execution and generates diverse tasks and goal states. Complementing this, a state-anchor mechanism automatically annotates task-relevant UI elements in successful goal states as reference anchors. During RL training, these reference anchors provide accurate, scalable reward signals that substantially enhance efficiency. Extensive evaluations demonstrate that our framework achieves over 90% accuracy on offline trajectory verification and performs closest to rule-based methods. Furthermore, agents trained using our reward framework exhibit strong performance on both AndroidWorld and our constructed benchmark, establishing a scalable approach for GUI agent training. 

---
# FinixDoc: Rethinking Financial Document Parsing Beyond Saturated Benchmarks 

**Authors**: Hang Wang, Jin Zhang, Guoliang Xu, Pengyue Lu, Yao Li, Zijiao Zhang, Tianyu Huang, Weiqi Xiong, Yulong Wang, Chuqiao Lu, Wenkang Huang, Kai Yang, Yadong Li, Hui Li, Xingzhong Xu, Xiao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.22842)  

**Abstract**: Financial document parsing requires accuracy, structural consistency, and verifiability that current benchmarks often fail to reflect. We present FinixDoc, an end-to-end agentic parsing system for real-world financial documents, with FinixDoc-VL, a 4B-scale vision-language model built on Qwen3-VL-4B, as its core parser. To characterize the gap between benchmark and deployment performance, we introduce a Document Parsing Capability Matrix organized along two practical axes: visual quality and document scale. Guided by this matrix, FinixDoc-VL is trained with a domain-adapted recipe combining homoglyph-aware contrastive learning and multi-stage reinforcement learning with composite domain-specific rewards. To better leverage our accumulated advantage in low-quality financial-document data and support large-scale, high-quality data production, we further build a human-in-the-loop Data Factory pipeline with confidence-aware expert review. For evaluation, we construct FinixDocBench, a financial-domain evaluation suite covering digital-native, camera-captured, ultra-large-page, and internal-workflow scenarios, with a compliance-reviewed subset released alongside this technical report. On its main subsets, FinixDoc-VL achieves the highest overall score (81.43) among evaluated baselines, outperforming the next-best open-source model by 5.13 points, with the largest gains on internal financial workflows (FinixInner: 84.08 vs. 78.73). 

---
# Let the Bullets Fly: Multimodal Fake News Detection with Temporal-Aligned Generative Danmaku 

**Authors**: Xiansheng Luo, Chaowei Zhang, Zewei Zhang, Yi Zhu, Jipeng Qiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22832)  

**Abstract**: The social interactions among crowds via \textit{Danmaku} (a.k.a., bullet comments) on modern multimedia platforms can facilitate both viewpoint conflicts and consensus, providing fine-grained discriminative social signals that can benefit fake news detection. However, the inherent accumulation latency of \textit{Danmaku} in real-world scenarios violates the real-time necessity of fake news detection, making the studies of \textit{Danmaku}-related fake news detection underexplored. To break this violation, we simulate this temporal-aware user interactive process by proposing a novel temporal \textbf{Gen}erative \textbf{da}nmaku framework, called \textbf{Genda}, which consists of: (1) a \textit{Danmaku} Trigger for predicting the timing and intensity of user reactions; and (2) a \textit{Danmaku} Generator for synthesizing corresponding semantic and emotional expressions, thereby mutually constructing a temporally aligned and human-like pseudo \textit{Danmaku} streams. To make the generated \textit{Danmaku} useful for identifying fake news videos, we further design a \textit{Danmaku}-guided Temporal Multimodal fake news detection model - \textbf{DM-FEND}, which enables fine-grained multimodal interactions among video, audio, text, and \textit{Danmaku}, enhancing dynamic modalities alignment and semantic noise inhibition. The experimental results demonstrate that \emph{DM-FEND} consistently outperforms state-of-the-art baselines across both Chinese (FakeSV) and English (FakeTT) benchmarks. Further ablations validate the crucial role of temporal \textit{Danmaku} modeling in enhancing robustness and discriminative capability. Finally, this study offers a bright and robust solution for multimodal fake news detection in modern social interactive fashions by bridging the temporal inconsistency between news and user behaviors. 

---
# Beyond the Harness: End-to-End Optimization of Context Artifacts for Enterprise Text-to-SQL 

**Authors**: Kate Gwimm, Carson Eisenach  

**Link**: [PDF](https://arxiv.org/pdf/2608.22830)  

**Abstract**: Deploying LLMs for enterprise Text-to-SQL is bottlenecked less by the model than by what context reaches it: business logic spans thousands of tables, and no model can ingest a full catalog at once. We argue that the most effective place to intervene is therefore the \emph{knowledge-base context} the model consumes, and that this context should be \emph{constructed} from historical usage rather than tuned for as a fixed input. Using a query-DAG decomposition--the same family of intermediates that enterprise benchmarks like BEAVER annotate, here recovered from production SQL--we compare the value of oracle query graphs versus retrieved knowledge-base context. In this ablation, retrieved knowledge-base context provides the largest marginal improvement when added to the full oracle graph. Building on this, we optimize a distillation procedure that turns historical query profiles into reusable SQL reference cards. On a benchmark of 5176 production queries from a major online retailer, optimizing these context artifacts yields larger gains (${\sim}12$--$25\%$ AST similarity) than optimizing the retrieval harness (${\sim}3$--$12\%$). On the public BEAVER benchmark, which lacks the production-usage signals available in our internal setting, the picture is more mixed: table cards alone perform about the same as raw historical SQL. The best optimized variant retrieves both cards and raw SQL, scoring $9.00\%$ versus $6.33\%$ (p-value $0.12$) for the comparable baseline on a held-out $N{=}300$ subset, using retrieved context and harness changes but no agentic loop. 

---
# Performance of a domain-specific large language model in answering patient questions in psychiatry 

**Authors**: Alexander J. Hish, Arjun Nagendran, Scott N. Compton  

**Link**: [PDF](https://arxiv.org/pdf/2608.22797)  

**Abstract**: Background This study was designed to evaluate whether a domain-specific large language model (LLM) trained exclusively on patient education resources can answer questions about psychiatric medications, in a manner superior to LLM chatbots. We developed an LLM ("MIND") fine-tuned for clinical fidelity, trained on patient education resources from authoritative medical organizations. Methods We compared the responses of MIND, ChatGPT, and OpenEvidence to patient questions about escitalopram, using two methods: (1) computer analysis according to a rubric measuring accuracy, clarity, completeness, nuance, safety, and referral appropriateness; (2) ratings from N=10 board-licensed psychiatrists on similar metrics. Results When rated by rubric, MIND was rated highest in all domains (p<0.001). When rated by psychiatrists, ChatGPT was rated accurate more often than MIND with a negligible effect size (p=0.021, r=0.073); MIND was rated complete more often than ChatGPT with a small effect size (p<0.001, r=0.160); and MIND and ChatGPT were rated safe with the same frequency (p=0.955, r=0.002). The majority of psychiatrists preferred the responses generated by ChatGPT (57.6%) compared to MIND (42.4%, p=0.003). Conclusions MIND was able to answer many questions about escitalopram in a manner deemed accurate, complete, and safe by psychiatrists the majority of the time. However, despite MIND's ability to provide more complete responses, psychiatrists preferred ChatGPT's responses. MIND represents a step towards building safe LLM systems to enhance patient education in psychiatry. 

---
# TailSieve: Partial-Rollout-Guided Tail Routing for LLM Rollouts 

**Authors**: Tianqi Xu, Lu Lv, Haoyang Huang, Wenjie Huang, Zhanming Shen, Yuhao Shen, Baolin Zhang, Xinyi Hu, Shuang Ge, Jun Dai, Tianyu Liu, Suorong Yang, Zhikai Li, Ye Bai, Jun Zhang, Lei Chen, Yue Li, Mingchen Wan  

**Link**: [PDF](https://arxiv.org/pdf/2608.22788)  

**Abstract**: Large-scale rollouts have become a core component of modern LLM systems, spanning reinforcement learning (RL) post-training, on-policy distillation (OPD), and sampling-heavy evaluation pipelines. Unlike online serving, which is typically optimized for request-level latency and throughput, a small number of long-tail generations can dominate the end-to-end makespan of an entire rollout step. In practice, rollout requests are often routed uniformly across replicas, which can place extremely long generations inside high-concurrency decoding batches.
To address this, we present TailSieve, a partial-rollout-guided framework that jointly controls tail routing and replica allocation for LLM rollouts. In an idealized setting with known completion lengths, we show that makespan-optimal routing in the long-tail regime combines tail isolation with load balancing, and that a simple top-k policy closely approximates this offline optimum. Leveraging the observation that long-tail prompts tend to remain long-tailed across policy updates, TailSieve uses partial rollouts as a training-free signal for identifying candidate tail groups. A hierarchical controller then jointly adapts the number of isolated groups and the replica split between the tail and bulk pools using collected response-work history and a measured concurrency-throughput model. TailSieve achieves up to 1.67x routing-only speedup over uniform group routing. The resulting low-concurrency tail pool further enables route-specialized speculative decoding with MTP or DFlash, achieving up to 2.59x speedup over uniform routing. Selected prompts are regenerated under the current policy, preserving on-policy generation and avoiding additional routing-induced length bias in steady state. 

---
# The Retriever Should Remember: Experience-Amortized Reranking for Long-Term Agent Memory 

**Authors**: Qi Feng, Chris Ding, Jicong Fan  

**Link**: [PDF](https://arxiv.org/pdf/2608.22767)  

**Abstract**: Long-term language-model agents accumulate memories across interactions, but their retrievers typically do not accumulate retrieval experience. Semantic retrieval is efficient, but embedding similarity does not always reflect whether a memory contains evidence relevant to the current query. Large language model (LLM) rerankers provide stronger query-conditioned relevance scores, yet stateless reranking repeatedly scores a large candidate pool and discards these scores after each query. We introduce EARM, an experience-amortized reranking framework that treats previously acquired LLM relevance scores as reusable retrieval experience. EARM stores sparse query--memory relevance scores in an online matrix, learns their shared structure through causal matrix completion, and combines a small set of newly observed scores with estimated scores to rerank the remaining candidates. The scoring budget decreases as experience accumulates, changing LLM reranking from a repeated per-query expense into a retrieval capability learned over an agent's lifetime. Experiments on long-term conversational memory show that mixed observed-and-estimated reranking improves answer accuracy over semantic retrieval by up to 6.62% and remains effective when only 17.5% of candidates receive direct LLM relevance scores, thereby substantially reducing the inference overhead of LLM reranking. These results motivate a broader view of agent memory: a long-lived agent should remember not only past content, but also how that content has proved useful for retrieval. 

---
# Compositional Chain-of-Relations for Faithful Knowledge Graph Question Answering with Large Language Models 

**Authors**: Chenhui Liu, Jianpeng Zhou, Jiahai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22762)  

**Abstract**: Knowledge graph question answering (KGQA) is a key task for evaluating KG-augmented Large Language Models (LLMs), and complex KGQA that requires multi-hop reasoning is especially challenging. Solving a complex query involves two coupled phases: candidate retrieval, which locates answer candidates over the KG, and constraint handling, which filters these candidates against the query constraints. Faithful reasoning requires grounding both phases in the KG. However, existing agent-based methods ground candidate retrieval through entity-centric exploration, while leaving constraint handling to the LLM's internal knowledge, which leads to two critical limitations. (1) Unreliable entity pruning: entity-centric exploration uses entities as search units and must prune them to a fixed-size subset at each hop. Because entity information in KGs is often incomplete and a fixed-size subset cannot retain all valid entities, such pruning inevitably drops valid entities and ultimately leads to wrong answers. (2) Ungrounded constraint handling: query constraints are resolved from the LLM's internal knowledge rather than the KG, leaving the final answers unverifiable and prone to hallucination. To address these limitations, this paper introduces a relation-centric exploration paradigm, which uses relations rather than entities as search units and thus avoids unreliable entity pruning. Built on this paradigm, this paper proposes Compositional Chain-of-Relations (CCoR), a simple and effective framework that grounds both phases in the KG with two relation chains: a main chain for candidate retrieval and a constraint chain that verifies query constraints through explicit KG exploration. Experiments on four KGQA benchmarks show that CCoR consistently improves accuracy, faithfulness, and efficiency over strong baselines, with more pronounced gains on complex queries. 

---
# The Compaction Cliff in Long-Running AI Agent Memory 

**Authors**: Saber Zerhoudi, Jelena Mitrovic, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2608.22752)  

**Abstract**: A safety rule and an episodic log compete for the same tokens in an AI agent's context. When the budget overflows, both are summarized at the same rate; only the rule needs exact wording to remain enforceable. On 20 production agent configurations, Claude Code's /compact prompt on Sonnet 4.6 preserves 53\% of safety rules after one compaction round and 10\% after five. We name this the Compaction Cliff. We address it with Knowledge Triage, a framework that classifies each line of an agent's knowledge base by type and routes each type through its own retention policy. Three deterministic operators implement this triage across the three context-management operations: TypeCompact rewrites items in place under per-type fidelity, TypeDecompose partitions a topic too large to compact safely, replicating in-scope safety rules across partitions, and TypeRetrieve fetches items from external storage with in-scope rules pinned ahead of relevance. On five public corpora, TypeCompact preserves 2--4$\times$ more safety rules than the strongest single-shot LLM compactor at every ratio, with 96\% recall over five rounds. TypeDecompose reaches 0\% locality violations against 93\% under uniform partitioning. TypeRetrieve reaches 100\% recall@50 against 73\% for the best single-shot LLM retriever. On three downstream behavioral benchmarks, we outperform the production Sonnet compactor on medical compliance (paired McNemar $p < 10^{-8}$ on preservation, $N = 200$), the full-policy and hierarchical baselines on retail task pass rate ($p < 0.01$, $N = 115$), and the hierarchical compaction on the airline domain ($p = 0.024$). We release AgentArtifactCorpus (396{,}934 agent configurations from 54{,}628 public GitHub repositories), the classifier, and the reference implementation. 

---
# LLM-Based Selection of Incongruent Verbal and Nonverbal Behavior for Virtual Humans 

**Authors**: Parisa Ghanad Torshizi, Stacy Marsella  

**Link**: [PDF](https://arxiv.org/pdf/2608.22731)  

**Abstract**: Nonverbal behavior generation systems for virtual agents often take an utterance as input and generate nonverbal behaviors that emphasize or illustrate the content of the verbal channel. However, human nonverbal behavior is shaped by more than the content of the speech. It is also influenced by speaker roles, interpersonal relationships, social context, and the cognitive and emotional states of the interactants. As a result, the nonverbal channel may reinforce, weaken, qualify, or even contradict the verbal channel. It may also reveal internal states that are hidden or only indirectly implied in speech, including emotional "leakage" that may be incidental to the immediate interaction. Modeling this richer relationship between verbal and nonverbal behavior is important for designing virtual agents that exhibit realistic, human-like behavior. It is especially critical in training contexts that require nuanced social interpretation, such as counseling simulations involving virtual patients. Drawing on Ekman's framework of verbal nonverbal relationships, we propose a taxonomy of categories in which mismatches between verbal and nonverbal behavior can occur. We then examine alternative approaches for realizing these behaviors using large language models, focusing on whether LLMs can select contextually appropriate mismatched verbal and nonverbal behaviors from a given dialogue and social interaction context. Finally, we evaluate the resulting behaviors in a human-subject study, assessing whether context-driven nonverbal behavior, when embodied in a virtual human, produces the intended effects on observers. 

---
# SEAM: Shot Entity-Attribute Memory for Consistent Short-Drama Generation at Scale 

**Authors**: Jiaqi Liu, Maolin Ran, Xiaoyang Lu, Jian Wang, Weiwen Liu, Jianghao Lin, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22725)  

**Abstract**: Short-drama generation has grown into a large, industrialized pipeline, and as it scales from isolated shots to the episode level, visual continuity has become a critical bottleneck. Current agent frameworks generate each shot in isolation, so context drifts across shots and props, character posture, and blocking turn inconsistent. Once assembled, these small discrepancies amplify into severe visual breaks. We present SEAM (Shot Entity-Attribute Memory), a training-free, model-agnostic memory graph that repairs continuity entirely at the prompt-text layer by extracting a multi-dimensional state for every shot, retrieving only causally prior context over the resulting graph, filtering it selectively, and injecting the surviving constraints by natural-language prompt rewriting. We further release SEAM-Bench, a double-blind continuity storyboarding benchmark, on which SEAM raises cross-episode continuity recall from 0.700 to 0.946, generalizes across six mainstream text models, and yields consistent, though not yet significant, gains at the generated-image layer. Deployed as a mandatory stage in CreativeFitting's SEAM-Agent production pipeline over 201 shots, SEAM reaches a 96.5% director-acceptance rate with zero unsafe injections; a conservative counterfactual attributes at least 21.9 percentage points of that rate to its cross-episode memory. 

---
# CacheRouter: A Dual-Path Tool Routing Architecture with Cache-Preserving Main-Model Isolation for Long-Tail Tool Discovery 

**Authors**: Donghui Zha, Lingwei Xu, Linxiao Wu, Yixue Dong, Haochen Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.22708)  

**Abstract**: Tool use in LLM systems faces a structural trade-off. Progressive disclosure keeps the prompt small by showing only the tools relevant to the current task, while prompt caching rewards a request prefix that stays fixed across calls; every change to the visible tool list invalidates the cached prefix. This paper treats the trade-off as a problem of request architecture and proposes a dual-path routing design that assigns tool selection and tool delivery to separate channels. The main model always sees a small, fixed set of core tools, so the head of its request is unchanged across calls; all other tools are reached through an independent routing channel, in which a router sub-model searches the full tool list, selects one tool, executes it, and returns the result. Tool registration is automated from source code and supports runtime updates, so the tool set can grow without modifying the main model's request prefix. The design generalizes progressive disclosure: capabilities are disclosed through the routing channel, and the main model's prefix stays stable. A prototype implementation was exercised on 55 functional queries and a 30-turn dialogue; token-level cache hit rates reached 90.99% and 95.2%, cutting input cost to about 12.0% and 8.0% of a no-cache baseline under DeepSeek's pricing, where cache-hit input tokens cost roughly 1/30 of cache-miss tokens. 

---
# Does Rank Still Matter? Position Bias When AI Agents Shop on Our Behalf 

**Authors**: Davood Wadi, Yu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.22697)  

**Abstract**: Search rankings are valuable because human attention is scarce and sequential. Higher-placed alternatives are easier to find, so they are examined and bought more often. Consumers are now delegating search to AI agents that can ingest an entire results page at once. Randomizing the order of one hundred hotel listings across 5,000 AI agent sessions, we compare four large language models against human field data. AI agents search more deeply than humans and never decline to buy. Position still predicts which listings are inspected, but weakly and non-monotonically: the middle of a results page has the lowest probability of inspection, not the bottom. Position reaches the choice stage for some models and not others, a heterogeneity that tracks neither provider nor capability. All models nonetheless converge on the same undominated listing. For agentic search, the attributes displayed on a results page matter more than placement within it. 

---
# Robustness Analysis of Agentic AI to Inconsistent and Incomplete Tool Responses 

**Authors**: Jiachen Xu, Torben Bach Pedersen, Zhongming Yao, Xiaoyu Zhang, Yushuai Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.22676)  

**Abstract**: Robustness to a bad tool return means answering it in the way that return calls for, which depends on how the tool went wrong. A tool that has failed and a tool that returns a well-formed falsehood are different problems with different remedies. We ask whether the two already differ at the moment the return arrives. This is a qualitative pilot study: we score single decision points rather than running agents to completion. We inject controlled faults into a retail customer-service domain and read two channels off the model's log-probabilities: the likelihood of the returned content under the tool schema alone and under the whole trajectory, and its distribution over the legal actions, read for both shape and where the mass sits. An incomplete return is legible in every case, being improbable under the schema alone in a range no other condition enters, and it moves the mass toward the tools that re-read state wherever there is room to move. An inconsistent return leaves the schema channel untouched and registers in the likelihood comparison on the field whose true value the context already carries verbatim, not on the one whose contradiction runs through the domain policy. The action distribution gives each condition a distinct signature, but orders them by how far the return bears on the next action rather than by fault family. Recognition is therefore asymmetric: each condition is legible in some channel, and no channel is legible on all of them. 

---
# A-CPES: A Reference Framework for Agentic AI in Cyber-Physical Energy Systems 

**Authors**: Xiaoyu Zhang, Qiuye Sun, Jiachen Xu, Zhongming Yao, Yushuai Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.22672)  

**Abstract**: Energy system operation contains a loop of work that automation has never taken over: posing the optimization problem the current cycle should solve, disposing of infeasibility, sequencing a solution into interlocked switching orders, assembling evidence no single model holds, negotiating adjustable capacity with many parties, and settling experience into practice. Licensed dispatchers carry all of it in person, and the rising share of variable renewable generation is making that loop turn faster than their number can grow. Agentic AI supplies the abilities it requires, but enters as the outer loop of control: it calls SCED and the other decision models rather than being called by them. We propose A-CPES, three nested rings, an authorization and accountability frame around an agentic control outer loop around a six-layer CPES core. We argue the loop is indivisible, tune where and how tightly it may close, state eight structural failure modes as falsifiable predictions, and specify six governance modules that rebuild the authorization frame until it covers the loop, before the loop starts turning. 

---
# CAI-DLLM: Convergence Aware Inference for Diffusion Language Models 

**Authors**: Farhana Amin, Sabiha Afroz, Dimitrios S. Nikolopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2608.22646)  

**Abstract**: Diffusion language models can generate many tokens in parallel, but they still require repeated denoising steps during inference. This makes generation costly, especially when the model continues to recompute tokens that are already stable. To address these limitations, we propose CAI-DLLM, a training-free inference method that uses first-step confidence to guide denoising and reduce inference time. Specifically, CAI-DLLM commits easy tokens earlier, allocates more denoising steps to harder tokens, and adjusts decoding schedules across output blocks. As it relies only on first-step confidence signals, it does not require retraining, extra predictors, or weight updates. We evaluate CAI-DLLM on LLaDA-8B-Instruct and Dream-7B-Instruct across math, code, reasoning, commonsense, and long-context tasks. CAI-DLLM achieves up to 18.2x wall clock inference speedup on LLaDA GSM8K while improving accuracy from 76.27% to 77.41%, and up to 13.1x speedup on Dream HumanEval while achieving higher pass@1 than no-cache inference, 48.17% compared with 46.95%. On harder reasoning tasks, speedups reach 44.8x, with a largest accuracy drop of 4.4 points, while energy consumption is reduced by up to 95.3%. 

---
# DeepSAGE: Stage-Aware Reinforcement Learning for Structured CBT Counseling Dialogue 

**Authors**: Qi Zhang, Heajun An, Prakriti Dumaru, Sang Won Lee, Lifu Huang, Pamela J. Wisniewski, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2608.22615)  

**Abstract**: Large Language Model (LLM)-based counseling agents can generate fluent and supportive responses, but they often lack the structured, goal-directed progression required to conduct a coherent therapeutic session. We present DeepSAGE (Strategic AI Guidance Engine), a hybrid LLM--Deep Reinforcement Learning (DRL) framework for stage-aware counseling dialogue grounded in the first session of Cognitive Behavioral Therapy (CBT). DeepSAGE represents the session as eleven stages with explicit therapeutic objectives, with an external controller determines stage completion and the DRL model selects therapeutic intentions that guide LLM response generation. We evaluate DeepSAGE against six retrieval-, prompting-, stage-, and policy-based alternatives. DeepSAGE elicits higher simulated client engagement and openness and achieves the strongest balance of stage-goal completion and dialogue efficiency among stage-structured systems. Domain expert review further indicates that the generated conversations exhibit broadly plausible emotional trajectories and recognizable CBT processes. Because the evaluation relies primarily on simulated clients and model-based metrics, these findings demonstrate comparative dialogue-control improvements rather than clinical effectiveness. These results suggest that combining stage-structured dialogue with learned strategy selection is a promising approach for AI counseling, though clinical effectiveness, safety, and real-world utility require further human evaluation. 

---
# Coalition-Aware Skill Reliability for Self-Evolving Agents 

**Authors**: Qiyan Zhao, Xiaofeng Zhang, Bo Liu, Minda Chen, Wei Xiong, Jingyang Chen, Guanting Ye, Wenhao Yu, Xiaosong Yuan, Shijie Han, Da-Han Wang, Jianmin Ji, Fei Huang, Xu-Yao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22610)  

**Abstract**: Agent skills, structured artifacts distilled from interaction trajectories and dynamically reused from skill banks, have become a central mechanism for enabling large language model (LLM)-based self-evolving agents to learn from past experience. Yet existing work has largely focused on the operational aspects of skills, such as acquisition, evolution, and retrieval, while leaving a more fundamental reliability question unresolved: Do accumulated skills in an agent's skill bank actually make positive mechanistic contributions? We investigate this question through systematic skill-bank audits across alternative bank compositions and deployment domains, measuring the resulting changes in agent behavior. These audits reveal two recurring reliability failures: coalition pollution, where bank-level gains conceal negative coalition-level skill contributions, and cross-domain utility reversal, where source-beneficial skills reverse their effects after transfer. These findings motivate two reliability interventions: coalition-aware skill selection during skill accumulation and label-free skill masking after transfer. Coalition-Aware Skill Selection (CASS) selects more reliable candidate skills for the current bank using sampled Shapley marginals. Unsupervised Skill-Masked Coalition Optimizer (u-SMCO) masks transferred skills whose exclusion improves retrieval quality on unlabeled target-domain data. Agentic experiments on LoCoMo, LongMemEval, HotpotQA, and ALFWorld show that CASS and u-SMCO consistently improve task performance and cross-domain generalization over strong skill-based self-evolving agent baselines. Beyond accuracy, coalition-conditioned reliability modeling reduces sensitivity to noisy outcome-reward fluctuations during reinforcement learning and exposes the limits of isolation-based skill evaluation. 

---
# Weakly supervised concept Bottleneck Learning for Robust Two stage Object centric visual reasoning 

**Authors**: Sparsh Tiwari, Gesina Schwalbe, Bettina Finzel  

**Link**: [PDF](https://arxiv.org/pdf/2608.22584)  

**Abstract**: Two-stage neuro-symbolic architectures provide an elegant paradigm for visual problem solving by cleanly separating connectionist perception of predefined symbols from possibly later defined relational reasoning thereon. However, anchoring high-level predicates into visual frames typically necessitates annotations that are expensive to acquire. In this work, we introduce the Dynamic Orthogonal Concept Bottleneck (D-OCB), an object-centric slot- VAE framework designed to extract human-aligned symbolic predicates under extremely weak supervision. D-OCB eliminates the arduous manual tuning of loss-balancing coef- ficients by dynamically learning optimal hyperparameter allocations during training. To infuse prior knowledge on independence of concept categories, in addition to standard re- construction self-supervision we penalize correlation across concept subspaces. Crucially, to combat the instability of very low supervision regimes, D-OCB incorporates a dynamic di- mensionality allocation mechanism; this adaptive formulation allows well-represented con- cepts to yield latent dimensions to underperforming concepts that are lagging behind, effectively preventing representation collapse and significantly improving overall concept accuracy. Through an extensive empirical evaluation, we demonstrate that our framework achieves high concept alignment and downstream visual reasoning accuracy using minimal label budgets, matching or outperforming end-to-end paradigms. 

---
# CausalCache: Conditional High-Fidelity Restoration for Long-Horizon GUI Agents 

**Authors**: Jiaxuan Luo, Zhanfeng Liao, Jiayao Teng, Yuan Wang, Haojian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22577)  

**Abstract**: Long-horizon GUI agents can retain a complete interaction trace cheaply as textual action records, but expose only a few past events to the policy in high-fidelity pixels. We formulate this as conditional fidelity restoration: each event persists in summary-only form and is linked to an archived screenshot, while an active visual-context budget $B$ limits how many events may be promoted to summary-plus-image form. Recent-$B$ spends every slot on the latest events. CausalCache instead reallocates the same $B$ promotions over the complete trace, evicting a recent image only when a distant event has higher conditional marginal utility. Its history-gated key/value (HGKV) adapter modifies only restored history-image tokens and is exactly bypassed with no history image. Matched-budget replacement groups and per-arm-anchored difference-in-differences supervision make uniform history amplification worth zero; a budget-aware selector then chooses which summarized events to restore. On desktop, the frozen policy shows no reliable preference for a task-relevant archived screenshot over the recent frame it would displace; HGKV learns exactly that selectivity inside a pre-specified drift envelope. On OSWorld-Verified, restoring history to high fidelity is worth about $13$ success points over summary-only memory, while same-budget allocations remain indistinguishable. Zero-shot on a cross-application mobile benchmark, CausalCache significantly improves overall success over the same-budget recent allocation ($+3.7$ points on the full roster), and the gain concentrates where it should: $+8.6$ points on the memory-critical split fixed by benchmark metadata at construction, no detectable effect on matched controls, and a significant split-by-method interaction. 

---
# ExecRubrics: Executable Tool-Augmented Rubrics for Verifiable and Efficient Long-Form Evaluation 

**Authors**: Kaustubh D. Dhole, Charles L. A. Clarke, Eugene Y. Agichtein  

**Link**: [PDF](https://arxiv.org/pdf/2608.22559)  

**Abstract**: Rubrics aim to make language-model evaluation transparent by decomposing response quality into interpretable criteria. However, natural-language rubrics are often ambiguous, require black-box LLM judges, and typically assume criteria aggregate independently through linear weighted sums, limiting their ability to capture dependencies, alternatives, penalties, and override conditions. We propose ExecRubrics, a framework for representing rubrics as compact executable programs. ExecRubrics encodes evaluation logic as verifiable Python scoring functions, giving natural-language rubric intent an operational semantics: a fixed decision procedure that can be inspected, executed, and edited. On three long-form response benchmarks-HealthBench, HelpSteer, and ArgQuality-we show that ExecRubrics can substitute for expensive black-box judges in ranking preferred over dispreferred responses, matching or improving NL rubric baselines with best preference accuracies of 53%, 78%, and 92%, respectively, while reducing evaluation latency by up to 320 times. We show that incorporating external logic and resources from text processing libraries such as NLTK and spaCy further improves preference accuracy. Our results suggest a novel way of looking at evaluation, by offering a faster, more explainable, and less ambiguous alternative to black-box rubric evaluation, particularly in high-stakes domains such as healthcare and banking where precision and auditability are critical. 

---
# Scaling Curriculum Learning For Autonomous Driving 

**Authors**: Cevahir Koprulu, David Paz, Feng Tao, Yuliang Guo, Xinyu Huang, Ufuk Topcu, Liu Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.22549)  

**Abstract**: Batched simulators for autonomous driving have recently enabled training reinforcement learning (RL) agents at scale, encompassing thousands of traffic scenarios and billions of interactions within a matter of days. Although such high-throughput feeds RL algorithms faster than ever, their sample-efficiency has not kept pace: As the standard training scheme, domain randomization uniformly samples scenarios, thereby consuming a vast number of interactions on cases that contribute little to learning. Curriculum learning offers a remedy by adaptively prioritizing scenarios that matter most to policy improvement. We present CL4AD, the first integration of curriculum learning into batched autonomous driving simulators by framing scenario selection as an unsupervised environment design problem. We introduce utility functions that shape curricula based on success rates and the realism of the agent's behavior, in addition to existing regret-estimation functions. Large-scale experiments in GPUDRIVE demonstrate that curriculum learning achieves a 99% success rate a billion steps earlier than domain randomization, reducing wall-clock time by 77%, and outperforms heuristic curricula with static and dynamic attributes, with only one exception at the largest scale. An ablation under limited compute shows that curriculum learning improves sample efficiency by 67%. We also investigate how utility functions behave at scale, and how prioritized scenarios evolve during training. We release an implementation of CLForAD in GPUDRIVE. 

---
# STAGE: Stateful Translation to Agentic Graph Execution with Policy-Scoped Context and Deterministic Control 

**Authors**: Mengxi Luo, Changjia Chen, An Cao, Zirong Huang, Wanyi Dai  

**Link**: [PDF](https://arxiv.org/pdf/2608.22538)  

**Abstract**: Policy-governed agents must interpret case evidence while following an authorized procedure. We present \textsc{Stage}, an executable-graph framework that confines model judgment to policy-scoped nodes while placing procedural control in deterministic code. At each node, the model receives task-relevant policy context and returns a typed result, while the coordinator enforces the reviewed execution contract. We evaluate \textsc{Stage} on SOP-Bench Referral Abuse, two $\tau^2$-bench domains, and Smart Dispute, a proprietary banking benchmark. Compared with monolithic full-policy execution, \textsc{Stage} generally improves task success and repeated-run reliability across workflows of varying procedural complexity. The largest gains occur on the deeper Telecom and Smart Dispute workflows, where $\mathrm{Pass}^3$ increases by 7.5--55.0 and 57.2--65.7 percentage points, respectively, depending on the model. These results show that combining policy-scoped context with deterministic procedural control can improve the reliability of policy execution. 

---
# CONTRAMEM: Learning Self-Evolving Procedural Memory from Contrasting Multi-Model Trajectories 

**Authors**: Zheyuan Deng, Binghang Lu, Hanqi Feng, Shirley Huang, Dianzhuo Wang, Yuanda Xu, Zhiwei Zhang, Yige Sun, Changhong Mou, Runyu Zhang, Yuexing Hao, Barnabas Poczos, Xiaomin Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.22533)  

**Abstract**: Autonomous computer-use agents are increasingly applied to long-horizon tasks requiring coordinated application calls, persistent state tracking, and verifier-sensitive writes, yet they remain prone to procedural failures: misreading application state, tool semantics, or task progress. Procedural memory promises more consistent decisions and less redundant exploration, but constructing high-quality memory without model training remains challenging. We introduce CONTRAMEM, a source-flexible, training-free framework for self-evolving procedural memory that treats same-task outcome variation as supervision: differences in correctness, efficiency, recovery, and failure modes expose outcome-relevant procedural distinctions, distilled into a compact bank of app-level Function Cards and task-level Skill Cards that evolves through localized curation rather than append-only accumulation or whole-bank rewriting. On held-out GAIA2/ARE computer-use tasks, CONTRAMEM more than doubles the success rate across the three source-model targets (26.2% to 55.3%), with consistent per-model gains (GPT-5.5: 27.5 to 61.0; Claude Sonnet 4.6: 28.0 to 52.5; DeepSeek V4 Pro: 23.0 to 52.5). The same bank transfers unchanged to the unseen Qwen3.7 Plus (18.5 to 35.5), indicating transferable procedural knowledge rather than model-specific behavior. The same construction carries over unchanged to AppWorld, beating both no memory and its own single-source self-memory variant for all three mid-tier agents on both public test splits. Under a matched trajectory budget, heterogeneous multi-model trajectories yield stronger memory than self- or same-model multi-rollout memory: the margin comes from contrastive behavioral diversity, not stronger source agents or more sampling. 

---
# HANSARD: A Reference Architecture for Forensic Readiness, Runtime Witnessing, and Graded Attribution in Autonomous Multi-Agent AI Systems 

**Authors**: Christos Sardianos, Iliana Pla, Vasilis Efthymiou, Iraklis Varlamis, Thomas Lagkas, Panagiotis Sarigiannidis, Georgios Th. Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2608.22512)  

**Abstract**: Autonomous multi-agent systems nowadays act in finance, software supply chains, and security operations. Already, the first largely AI-orchestrated intrusion campaigns have been reported. Yet, when such a system causes harm, no method can robustly establish what happened, what caused it, or who is accountable. This is because provenance forensics works at the wrong abstraction, formal causality assumes the causal model, and agent auditing trusts self-recording. The target failure mode is, thus, attribution laundering, i.e., spreading an act across redundant agents until none is a but-for cause. Worse, the record is produced by the suspects, which comprises the assumption adopted throughout this work. Agents may therefore anticipate the investigation and the part of logging infrastructure may itself collude. In this paper, HANSARD is proposed, a reference architecture treating accountability as a life-cycle property. First, a readiness profile sealed before operation bounds what later findings may claim. Second, capturing at five choke points beyond the agents' reach makes omissions detectable, not only tampering. Third, a typed PROV-DM-aligned causal graph accrues as the system runs, and three indicators read it live to gate oversight without adjudicating. Fourth, post-incident replay yields contingent effects under the modified Halpern-Pearl definition, together with a compensation-set size. Finally, a synergy residual measures harm due to the combination rather than to individuals, making laundering visible. Cause, responsibility and accountability are then reported separately, each capped by an evidentiary tier, while a future research agenda is also provided. 

---
# ClawProBench: Trace-Aware Evaluation of AI Agents with Runtime Coverage and Frozen Workplace-Style Holdouts 

**Authors**: YuanHang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.22510)  

**Abstract**: Agent benchmarks often evaluate only final answers even when agents run on stateful runtimes. We argue this under-specifies what is being evaluated: the proper unit is a declared model-plus-runtime configuration whose failures can occur in evidence acquisition, runtime routing, safety boundaries, or repeated execution. We present ClawProBench, a trace-aware benchmark for runtime-native agent evaluation instantiated on OpenClaw, a live agent runtime with workspace tools and native surfaces for browsing, memory, messaging, scheduling, skills, and subagents. ClawProBench defines two tracks: a 102-scenario full profile with live workspace and native-runtime routing tasks, and a frozen 68-scenario holdout with closed-world JSON output contracts for robust ranking. Trials are scored from execution traces via a safety-gated formula combining correctness, process quality, and efficiency, preserving failure evidence for audit. Our anonymous artifact includes benchmark definitions, scoring code, manifests and sanitized traces. We evaluate 68 configurations on the full profile and 37 on holdout. The top safety-gated average trace score is 0.7671. Native-runtime tasks underperform workspace-live tasks (0.5238 vs. 0.6415). On holdout, pass@k-any outperforms strict three-trial pass (0.6638 vs. 0.2890), while full-profile and holdout rankings show weak alignment (Spearman 0.1300). Rankings based purely on correctness differ substantially from process-aware, safety-gated and strict-pass views. Final-answer leaderboards may hide native-surface weaknesses, one-off successes and trace-local agent failure modes. 

---
# When Does AI for PDEs Yield Scientific Evidence? 

**Authors**: Wenshuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22504)  

**Abstract**: Existing AI-for-PDE benchmarks primarily assess models in terms of predictive or approximation accuracy. In physics research, however, AI outputs often serve as evidence for scientific claims. These two objectives are not equivalent: the former measures an output's agreement with a reference target or satisfaction of governing constraints; the latter asks whether, given a specified object of study, scientific claim, assumptions, and evidence standard, the output provides sufficient evidence for that claim. To bridge this gap, we extend a widely used PDE-simulation benchmark and a comprehensive benchmark for PDE inverse problems to enable, for the first time in AI for PDEs, evaluation of whether and to what extent model outputs support specified scientific claims. Our results show that numerical accuracy and evidential support can rank models differently, explain when and why they do so, and reveal that existing benchmarks can favor methods whose outputs provide weaker support for the scientific claims of interest. Together, we formalize, empirically demonstrate, and explain this evaluation--use mismatch in AI for PDEs. 

---
# Small Reasoning Models are Instruction Followers in Function Calling 

**Authors**: Yalda Taheri, Mohammad Hassan Heydari, Erfan Naaman, Afsaneh Fatemi  

**Link**: [PDF](https://arxiv.org/pdf/2608.22472)  

**Abstract**: Function calling represents the core capability of agentic large language models (LLMs). Existing research has focused on enhancing LLMs function-calling accuracy through fine-tuning, reinforcement learning (RL), and multi-agent frameworks, particularly for native function-calling LLMs. This work demonstrates that LLMs achieve superior accuracy in function calling in instruction-following contexts (i.e., standard user-assistant interactions) rather than a tool calling context. We introduce Instruction-Followed Function Calling (IFFC), a novel framework that decouples function-calling logic from the primary LLM and delegates it to a dedicated smaller model operating within the instruction-following paradigm. Our method consistently outperforms both native function calling (NFC) and prompt-based function calling (PFC) baselines, with particularly strong gains on reasoning-oriented LLMs. Furthermore, we demonstrate that IFFC maintains robust performance under aggressive quantization, enabling efficient on-device deployment without significant accuracy degradation. This work establishes a new paradigm for reliable, resource-efficient function calling in edge-computing scenarios. 

---
# When Persona Simulations Are Informative: Graph-Structured Signals for Pluralistic Opinion Sensing 

**Authors**: Taehyeon An, Jaehyeong Park, Donghyuk Shin  

**Link**: [PDF](https://arxiv.org/pdf/2608.22438)  

**Abstract**: Persona-conditioned large language models (LLMs) are increasingly used to simulate survey responses across diverse domains. However, apparent response variation can reflect unconditioned model priors or token sampling noise rather than systematic persona conditioning. We argue that persona-conditioned variation is informative when semantically similar personas exhibit concordant response shifts. To operationalize this principle, we introduce Persona-Conditioned Informativeness (PCI), an unsupervised diagnostic metric that measures whether semantically similar personas deviate in concordant directions relative to item-level sample baselines. By modeling personas as a similarity graph, PCI uses Local Moran's I to quantify local spatial coherence and extract compact persona subsets without using construct labels. To evaluate PCI without external human benchmarks, we test its ability to recover established latent value structure using the 57-item Portrait Values Questionnaire-Revised (PVQ-RR). Confirmatory factor analysis (CFA) shows that a PCI-selected 10% subset substantially improves overall construct recovery relative to response-stability and random selection. These findings support PCI as a principled internal diagnostic for screening synthetic respondents in survey pipelines. 

---
# Think with Structured Grounding: Perceptual Reinforcement Learning for Chart and Visual-Tabular Understanding 

**Authors**: Changjiang Jiang, Qiannian Zhao, Lei Xin, Jinxiang Xie, Preslav Nakov, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2608.22429)  

**Abstract**: Multimodal Large Language Models (MLLMs) capable of thinking with images often rely on external tools for fine-grained perception. However, this reliance introduces significant inference latency and fails to effectively resolve the spatial-structural gap-a fundamental challenge in text-dense and structurally relational visuals (e.g., charts and visual tables) where strict relative spatial arrangements bind textual elements. Without external tools, standard MLLMs struggle with such fine-grained visual reasoning tasks. To address these issues, we propose Think with Structured Grounding (TwSG), a novel fine-grained image perception framework designed to internalize complex images's tool-use capabilities within the model. TwSG distills the benefits of multi-step reasoning and micro-cropping into a single efficient forward pass during inference. Specifically, we use an MLLM to identify key regions guided by ground-truth answers, and then prompt a teacher model to generate high-quality visual question-answering (VQA) data. These fine-grained, region-based supervisory signals are subsequently distilled back into the full-image representation. Our training pipeline consists of two stages: (1) a cold-start supervised fine-tuning (SFT) phase using multi-turn data with focused area descriptions to foster complex reasoning and error recovery; and (2) a reinforcement fine-tuning (RFT) phase driven by a novel process reward mechanism, TL-GRPO, which encourages strategic reasoning. Extensive experiments across various MLLM architectures demonstrate that TwSG reduces inference latency while substantially improving accuracy and robustness, endowing models with native fine-grained region description and flexible reasoning capabilities. 

---
# Where World Models Break: Natural-Input Failure Discovery 

**Authors**: Zhanpeng Shi, Zi Liang, Rong Feng, Shiqin Tang, Xuyang Chen, Hongzong Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.22421)  

**Abstract**: World models predict action-conditioned futures and serve as critical internal simulators for downstream planning and control. However, catastrophic prediction failures of world models could dangerously propagate through the control pipeline, as subsequent agent or model training and decision-making depend heavily on the continuous environment evolution forecasted by these world models. Existing evaluations overlook this systemic risk: by aggregating average errors over benign generations from general queries, they fail to stress-test the model against catastrophic collapses under rare or unobserved condition-action combinations. To bridge this gap, we formalize the natural-input failure discovery problem: under a finite query budget, finding environment-valid conditions and action prefixes that induce severe prediction risk, verifying whether these failures reproduce on fresh seeds, and testing their persistence under nearby valid edits. Discovering such critical failures is computationally challenging, as valid condition-action combinations explode exponentially, rendering exhaustive search or standard sampling infeasible given the high cost of noisy rollouts. To tackle this, we propose BasinLens, which exploits the underlying structure of valid inputs, where each coordinate possesses environment-defined semantic types and admissible domains, by pairing uncertainty-guided global search with typed local replacements. Across diverse benchmarks and world-model families, BasinLens exposes reproducible and locally persistent failure modes that conventional evaluations fail to reveal, showing that average-case benchmarks can mask important vulnerabilities in world-model-driven control. 

---
# LLMs for Survey Text Analysis - A Performance Comparison Between Humans and GPT-5 on Inductive Content Analysis 

**Authors**: Leonardo Bergmann, Renata Gheorghiu, Ana Gvritishvili, Alex Mican, Chris Stewart, Topias Tolonen-Weckström  

**Link**: [PDF](https://arxiv.org/pdf/2608.22417)  

**Abstract**: Large language models (LLMs) are increasingly used to support text analysis in qualitative research, yet evidence on their performance in inductive content analysis remains limited. This study compares human and LLM-based inductive coding of open-ended survey responses from 903 answers across six variables from a European PhD student survey. Five human coders performed inductive content analysis following a standardized coding scheme, while an LLM (GPT-5.4) conducted the same task using an established prompting procedure. Agreement between human and LLM outputs was assessed using the Adjusted Rand Index (ARI). Results showed an alignment between humans and the LLM, with ARI values of 0.61 for coding and 0.54 for theme generation. These values were close to the internal consistency of coding and theme results within humans (ARI = 0.68) and the LLM (ARI = 0.76). Agreement varied widely across variables, with low within-entity consistency consistently linked to low between-entity agreement, underscoring the role of data characteristics and individual performance in reliability. Overall, the findings suggest that LLMs can approximate human coding in this case-specific setting, particularly at the coding level, and may serve as a scalable support tool for inductive qualitative analysis. 

---
# WAM-OPD: On-Policy Distillation for World Action Models 

**Authors**: Liuhaichen Yang, Zhuang Jiang, Chenchao Sheng, Zezhi Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22364)  

**Abstract**: World action models (WAMs) couple visual future prediction with robot action generation, but accelerated students can lose task capabilities during distillation and later encounter states that are poorly represented by offline data. We study whether on-policy distillation (OPD) can repair such a student without requiring sparse-reward reinforcement learning. We introduce WAM-OPD, a deployment-consistent post-training recipe for a video-first WAM. The student acts in the environment and therefore determines the history distribution. A frozen teacher labels those student histories with coherent video and action targets, while the student action branch is trained under its own generated video plan, as it is at deployment. Joint video and action losses update lightweight adapters in the shared backbone, together with an action flow-matching regularizer. In preliminary RoboTwin 2.0 studies on two tasks, the released one-video/one-action-step Flash-WAM improves from 0.0% to 58.3% success on HANDOVER MIC, and from 16.7% to 33.3% on PUT OBJECT CABINET. These task-specific results are an initial capability proof rather than evidence of broad or uniform generalization. They nevertheless suggest that dense teacher supervision on student-induced histories is a promising post-training interface for video-first WAMs. 

---
# Analyzing and Mitigating Cross-Lingual Degradation in Multilingual Medical VQA 

**Authors**: Jingbo Wang, Sendong Zhao, Haochun Wang, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.22363)  

**Abstract**: Medical visual question answering (VQA) is a crucial task in clinical AI, yet its evaluation has so far centered almost exclusively on English, limiting its relevance to linguistically diverse patients and clinicians. Recent multilingual medical VQA benchmarks show that large vision-language models (LVLMs) degrade in non-English languages, but lack a fine-grained analysis of how cross-lingual variation affects the distinct capabilities that medical VQA requires. To this end, we construct a multilingual medical VQA benchmark over eight languages, organized into four representative scenarios that isolate the core capabilities medical VQA requires. Evaluating five open- and closed-source LVLMs, we find that cross-lingual degradation is not uniform but highly scenario-dependent. We therefore propose MedVL-XLRepE, a training-free scenario-aware representation engineering method, leveraging LVLMs' superior English medical VQA capability to steer non-English representations toward their English counterparts at inference time. Across three LVLMs and eight languages, MedVL-XLRepE consistently mitigates cross-lingual degradation, with gains of up to 6.33\%. 

---
# Addressing the Selection Problem in Explainable AI 

**Authors**: Claire Vlases, Katelyn Morrison  

**Link**: [PDF](https://arxiv.org/pdf/2608.22356)  

**Abstract**: Explainable AI (XAI) research has produced a plethora of explanation techniques, yet user studies repeatedly show that available explanations are not effective in practice. We argue that, given the siloed nature of conventional XAI, users are struggling to select the appropriate XAI technique. Viewing XAI through a philosophical lens, we offer a formalization of what we call the selection problem: the systematic failure of XAI interfaces to bridge the gap between a user's natural-language uncertainty and the explanation technique that resolves it. Following a logical premise-conclusion format, we show that conventional interfaces require users to translate their uncertainty into a technique selection, a challenging prerequisite to meet. We also propose a structural solution: a multi-agent LLM orchestration tool that translates the user's query to the proper XAI explanation technique. We provide an example of how this structural solution could be instantiated to address the selection problem. 

---
# Where Cognition Lives: Dissecting Emergent from Computed Function in a Minimal Complete Cognitive Architecture 

**Authors**: Francisco M. Arrabal-Campos, Francisco G. Montoya, Alfredo Alcayde, Ignacio Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2608.22347)  

**Abstract**: A cognitive architecture is more than the module that reasons: it must also decide how long to think and what deserves the effort. We built a minimal but complete system - a recurrent reasoner with adaptive halting, a homeostatic control field, and a value module - and asked of each part: does this function emerge from gradient descent, or must it be computed? Competence emerges. Stopping appears to emerge too, and to be worth more than everything decidable in advance, but that appearance is instrumentation: payoff at matched mean compute climbs from 0.467 (uniform) through 0.546 (difficulty) to 0.698 (ex-ante value), and the further climb to 0.921 (posterior self-observation) does not survive audit. PonderNet-style halting returns a halting-weighted mixture of hidden states while forced-depth baselines return one, and the language head is trained on the mixture alone; equalizing the readout annihilates the apparent advantage of native execution (residual +0.000 [0.000, 0.000]). Value does not emerge: trained couplings capture zero of a payoff an explicit allocator captures completely (+0.151, routing correlation +0.79), so the second-order decisions that pay must be computed, at least where value is orthogonal to content, as here by construction. On a frozen LLM actuator the same instruments show self-consistency voting to be a measured bound (+0.0236 [+0.0150, +0.0326]) and inter-sample agreement nearly worthless as a stopping signal, its mass concentrating on wrong answers. Every null we assert carries a mechanism and a positive control, and the protocol is part of the contribution. Executing our own falsifiable prediction, value under commitment pays +0.1312 [+0.1124, +0.1502] in a cliff-cost family, some seven times the smooth-family estimate - not because the cliff shifts information ex ante, but because it multiplies the attainable range fivefold (5.1x [3.4, 8.2]). 

---
# HERO: Human-profile Enhanced Retrieval Optimization Framework for Long-term Agent Memory 

**Authors**: Yuanhua Lin, Yile Li, Zhiyuan Zhao, Jing Shang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.22310)  

**Abstract**: Long-term memory is crucial for personalized responses and long-horizon agent interactions. Existing methods often rely on LLMs to compress or rewrite dialogue histories and use the transformed memories as retrieval evidence. Despite the progress in organizing fragmented contexts, two major drawbacks persist: (1) information loss from compression, which discards fine-grained but later useful details, and (2) semantic drift from rewriting, which erodes the original tone and situated context. In this work, we propose a novel Human-profile Enhanced Retrieval Optimization framework for long-term agent memory (HERO). Specifically, HERO converts the dialogue history into a traceable heterogeneous memory graph that preserves raw dialogue text as evidence for reasoning, thereby mitigating information loss. For retrieval, HERO extracts initial anchors from the current query and incorporates human profiles via an iterative graph traversal; these anchors and profiles provide guidance signals that adaptively activate the most informative regions of the graph. Experiments on two benchmark datasets show that HERO outperforms strong baselines on both factual and personalized reasoning, while providing more faithful access to raw dialogue evidence. 

---
# Clarify User Expertise: Towards Proactive Conversational Agents Tailoring Responses to User Proficiency 

**Authors**: Zhihong Cao, Chen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22266)  

**Abstract**: In the context of information seeking, conversational agents are undergoing an evolution from reactive tools to proactive, personalized assistants. A critical aspect of this evolution is the ability to tailor strategic interactions to a user's unique needs and expectations. Unlike existing studies that focus on proactively clarifying query ambiguities, we center on clarifying the user's expertise in order to tailor responses for better user comprehension. We find that existing agents struggle to determine user expertise from queries alone, a limitation that prevents them from dynamically adapting their responses. To address this gap, we introduce PASSING to empower the agent to proactively clarify a user's expertise through targeted inquiries. This is achieved by our What-to-ask and How-to-ask strategies, induced by LLM self-play. Our extensive experiments also show our superiority. We believe that PASSING represents a crucial step towards creating more human-centric conversational agents. 

---
# Read Less, Solve More: Token-Efficient Sparse Reading for AI Agents 

**Authors**: Zedong Liu, Jiaan Wu, Xinyang Ma, Le Xu, Kai Wang, Yuanchao Hu, Dingwen Tao, Guangming Tan  

**Link**: [PDF](https://arxiv.org/pdf/2608.22237)  

**Abstract**: Long-horizon agents increasingly rely on repeated access to external artifacts, yet current reading interfaces often expose entire objects even when only sparse evidence is needed. This over-reading increases token and latency costs and can dilute task-relevant evidence, while existing context-reduction methods mainly intervene after broad content has already entered the trajectory. We present SparseRead, a training-free, model-transparent reading layer that controls content admission before unnecessary evidence reaches the model context. SparseRead combines a regime-aware Read Gate, extensible Reader Backends, and a stateful protocol for bounded, source-anchored evidence acquisition with explicit refinement, verification, stopping, and fallback. Across six frontier models, including Claude Opus 5, and five workload scenarios, SparseRead reduces token volume by up to 92.9% and wall time by up to 89.0%, while preserving or improving task quality. Its consistent gains across three agent frameworks further demonstrate broad portability. 

---
# Beyond What Meets the Eye: Unveiling Situational Illusions for Multimodal Large Language Models 

**Authors**: Zhiming Yang, Zhuoxi Xiong, Donglin Zhou, Wenjun Wei, Shiyao Cui, Jinqiao Shi  

**Link**: [PDF](https://arxiv.org/pdf/2608.22232)  

**Abstract**: Real-world situation appearances can deviate from their underlying physical states, challenging the reliability of multimodal large language models (MLLMs) in practical applications. In this paper, we term this phenomenon situational illusions and investigate: (1) how MLLMs perform under such illusions, and (2) how to mitigate the limitations. We first develop a comprehensive where-what-how taxonomy that characterizes where situational illusions occur, what targets they take, and how they arise. Building on this taxonomy, we introduce MSIBench, a benchmark designed to assess the discrimination, understanding, and reasoning capabilities of MLLMs under situational illusions. Evaluations of 27 model configurations reveal that current MLLMs are highly vulnerable to these illusions and exhibit 6 typical failure modes related to visual observation, grounding, and reasoning. To mitigate the limitations, we build on the core idea of systematically inspecting and reasoning over visual evidence for contextual understanding, developing prompting for closed-source models and supervised fine-tuning for open-source models, respectively. These two simple yet effective methods improve model performances by 20% at most, suggesting a practical path toward more reliable multimodal perception and reasoning in complex real-world environments. 

---
# Query-Driven Multimodal Information Extraction from Long Documents 

**Authors**: Yikai Gao, Ding Xia, Xi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22214)  

**Abstract**: In domain-specific multimodal long documents, images and text jointly convey complex knowledge that cannot be fully captured by plain text alone. However, existing paradigms like DocVQA primarily focus on generating textual answers or localizing evidence regions, rather than outputting query-specific textual attribute values and corresponding images. To address this gap, we propose query-driven image-text joint extraction from long documents, requiring models to output query-requested textual attribute values and corresponding image bounding boxes. Based on challenges related to both user intent and document content, we designed a two-level taxonomy that operates at the query and instance levels. Further, we construct ITJoint, the first high-quality, manually annotated benchmark for this new task, comprising 2,455 pages of domain-specific documents with numerous non-decorative images, 316 queries, and 910 answer instances. Finally, we evaluate representative standalone Vision-Language Models from different providers and further design Q2IT, a multi-agent collaborative framework consisting of three progressively collaborating agents for evidence collection, page selection, and target-image localization. Using a joint evaluation approach that assesses both text extraction and image localization, our experiments show that standalone VLMs struggle with this task, while Q2IT significantly improves performance on ITJoint, although a substantial gap remains toward perfect results. 

---
# Disagree to Explore, Agree to Commit: Routing-Guided Test-Time Scaling for Software Agents 

**Authors**: Kang Chen, Junjie Nian, Yixin Cao, Yugang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22191)  

**Abstract**: Software-engineering agents solve repository-level tasks through long, stochastic tool-use trajectories, and repeated attempts often find fixes missed by one run. Test-time scaling is difficult because patches lack canonical answer forms, while sibling actions from a shared prefix are correlated. We study whether native MoE router traces can guide steering and selection without an external judge or selection-time test execution. Our analysis shows that routing provides a robust behavioral role signal; token-granular readouts and decision-matched comparison sets turn it into effective control. We therefore introduce Risa (Routing-Informed Steering and Arbitration): within trajectories, routing encourages diverse exploration and controlled convergence during patch commitment; across separately sampled trajectories, agreement at informative patch positions selects a final candidate. We evaluate on SWE-bench Verified using open-weight sparse MoE agents across scales and reasoning-effort settings. Risa's routing arbitration raises the macro-average resolved rate from 44.9% under uniform sampling to 48.2% on the gpt-oss family, matching text consensus without answer-string matching, and it transfers to Qwen3.6, where it improves on uniform choice and matches text consensus on the full 500-task benchmark. 

---
# Role-Specialized Mixture-of-Agents with Open-Weight LLMs for Clinical Prediction 

**Authors**: Jun Hou, Yi Fang, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22176)  

**Abstract**: Large Language Models (LLMs) are increasingly applied to clinical prediction tasks such as in-hospital mortality and readmission from electronic health records (EHRs). Privacy and compliance constraints motivate systems that can be deployed locally, which has increased interest in open-weight multi-agent designs. However, most medical multi-agent systems are evaluated as a single block, leaving unclear which agent role contributes to prediction and whether retrieval drives observed gains. We study a role-specialized Mixture-of-Agents (MoA) that combines medical knowledge retrieval with contrastive similar-patient reasoning. By varying the role design while holding the retrieval setup fixed, we localize the main effect to the final integrator. Pairing large open-weight analysts with a small open-weight integrator matches closed-model prompting on F1 for mortality prediction while flagging substantially more true high-risk patients. Mechanism analysis shows the role assignment directly yields a high-recall operating point without threshold tuning. The effect is task-dependent, with smaller gains for readmission because the available records correlate weakly with this longer-horizon outcome. These results position role design as a key factor in privacy-constrained, training-free clinical LLM prediction. 

---
# MCP-Universe RL: A Framework for Training MCP Tool-Use Agents via Reinforcement Learning 

**Authors**: Ziyang Luo, Yan Yang, Xiangru Jian, Ziji Shi, Xiaoqiang Lin, Jun Hao Liew, Silvio Savarese, Junnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.22167)  

**Abstract**: Reinforcement learning (RL) has become an effective way to improve the tool-use ability of large language models (LLMs), but most existing RL frameworks stop at the policy update. For every new domain, the user is left with two hard systems problems: standing up an isolated environment for each of hundreds of concurrent trajectories and connecting it to training, and scheduling the rollout so that the GPU stays busy across long, multi-turn episodes that spend much of their time stalled on slow tool calls. We present MCP-Universe RL (MCP-U RL), an open-source framework that takes over both. It uses the Model Context Protocol (MCP) as the interface to the environment, so any tool already exposed as an MCP server plugs into training with no RL-specific integration code. It builds the two missing layers once and reuses them across domains: an environment-orchestration layer that provisions, isolates, and recycles the MCP environments over a pluggable container backend, and a rollout-orchestration layer whose staged pipeline overlaps trajectories to keep the GPU busy while episodes wait on tools. A backend-agnostic training layer then applies the update through an existing RL backend, with veRL and slime integrations. With one configuration, changing only the task specification, we train software-engineering, deep-research, and general tool-use agents on gpt-oss-20b and improve task reward in all three. 

---
# Aggregation-Aware Synthetic Text Generation Against Authorship Re-Identification 

**Authors**: Qian Ma, Anna Squicciarini, Sarah Rajtmajer  

**Link**: [PDF](https://arxiv.org/pdf/2608.22161)  

**Abstract**: Online users often release multiple texts under the same identity, giving attackers an author profile that can reveal more than any single text. Existing authorship obfuscation methods optimize privacy independently for each document, leaving them blind to cross-document correlations that make aggregation dangerous. We propose Aggregation-Aware Synthetic Text Generation (AAST), a framework that addresses this gap by jointly selecting synthetic texts at the bundle level rather than optimizing each text in isolation. AAST targets attribution and verification attacks, including cross-genre settings where attacker references come from a genre not observed during generation or selection. Experiments across same-genre, cross-genre, neural, and independent non-neural stylometric attacks show that AAST lowers account-level linkability as bundle size grows, while preserving semantic quality, linguistic acceptability, and sentiment alignment. 

---
# AUDITA: certified auditing and causal attribution of adverse outcomes in autonomous multi-agent systems 

**Authors**: Zhixu Du, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.22160)  

**Abstract**: Physical automation is scaling toward fleets of embodied machines commanded by an AI brain. Early deployments already run factories and warehouses at production rates beyond any human line, and their adoption is accelerating. But when their joint decisions cause harm, everyone involved has reason to blame everyone else, the machine vendor, the algorithm provider, the factory operator, the insurer, and the regulator, and no method can divide the responsibility between them. Existing methods read logs whose origin they cannot verify and name a single culprit, misrepresenting outcomes that are overdetermined, preempted, or caused by an omission. We present \audita{}, an audit layer pairing a tamper-evident record of every inter-agent command with a certified, graded causal-attribution engine. We prove its verdict cannot be gamed: a rule-following agent can never be made to look guilty, an attempt to shift blame is itself caught and graded, and we establish the exact limit of what an evidence-based auditor can certify. On live language-model pipelines it reduces the standard judge baseline's responsibility error roughly threefold; on a benchmark of accident-grounded structures it recovers responsibility where single-culprit baselines fail, and stays invariant under forgery. \audita{} turns the question of who is to blame from an argument about logs into a calculation over evidence. 

---
# Evaluation of Small Vision-Language Models on Qualitative Mechanical Problems 

**Authors**: Henry Fordjour Ansah, Shreya Banerjee, Pranish Ghimire  

**Link**: [PDF](https://arxiv.org/pdf/2608.22143)  

**Abstract**: Qualitative mechanical problem-solving (QMPS) refers to solving qualitative problems from the mechanical domain. Qualitative problems can be solved with minimal discipline-specific information, without any robust quantitative calculation, generally by using qualitative reasoning and commonsense knowledge. QMPS is a vital aspect of human intelligence that allows us to tackle a wide range of tasks, from simple everyday ones such as turning on a tap to complex tasks in highly demanding and well-paying jobs in various fields, e.g., emergency medicine, plumbing, driving, etc. Employers often use the Bennett Mechanical Comprehension Test (BMCT) to evaluate job candidates' ability to solve such problems. In this work, we assess two state-of-the-art multimodal models, Gemma-3 and Qwen-VL, on their ability to interpret mechanical problem images by eliciting a step-by-step chain of thought (CoT) and a final answer. Each image inherently encodes ground-truth qualitative facts, such as contact points in gears, support relations, and relative weights, which we use to evaluate each model's spatial and commonsense reasoning capabilities. We assess each chain for coherence, completeness, and logical progression to assess each model's thought process, and final answers are compared to verified solutions to measure accuracy. 

---
# MEMONDEMAND: A Memory Management System for Large-Scale Enterprise Data 

**Authors**: Xinyuan Song, Bowen Zhu, Hasibul Haque, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.22141)  

**Abstract**: Enterprise repositories are large, heteroge- neous, and continuously updated, making re- trieval difficult when efficient access, source- faithful evidence, and cross-query adaptation must be supported together. Enterprise mem- ory extends retrieval beyond the model con- text, but existing systems do not jointly address collection-specific hierarchy construction, low- cost routing, detailed evidence loading, and workload-aware memory updates at this scale. We introduce MEMONDEMAND, short for On- Demand Memory, a memory management sys- tem with three coordinated mechanisms: a dy- namic multi-level hierarchy that determines the abstraction structure and depth for each col- lection, dual memory at every hierarchy level that separates distilled routing from detailed evidence, and on-demand memory promotion that updates node priority under a bounded active-state budget. On EnterpriseRAG-Bench, MEMONDEMAND outperforms the strongest published LB#1 result at every evaluated scale from 10M tokens through the complete 618M- token collection, with gains of 12.23% at 10M and 4.66% at 618M. Results on FinanceBench, HotpotQA, and FRAMES further show strong performance across financial, multi-hop, and fact-retrieval settings. Together, these results establish MEMONDEMAND as an accurate, ef- ficient, and scalable memory solution for very large enterprise repositories across data scales, domains, and evidence requirements. Our code is available at this https URL xfab-xinyuansong/MemOnDemand.git. 

---
# Measuring Stability and Failure Behavior in Language Models Under Structured Perturbations 

**Authors**: Samira Golsefid  

**Link**: [PDF](https://arxiv.org/pdf/2608.22138)  

**Abstract**: Language models are usually judged by a single accuracy score, which does not reveal how their performance degrades as inputs are perturbed. We present a graded, multi-family, failure-aware framework for stress-testing reasoning models. It perturbs each problem along a multi-level severity ladder across seven families: six that preserve the answer, paraphrase, input noise, formatting, irrelevant context, context load, and conflicting instructions, and a Knowledge Boundary family that removes answerability so that refusal becomes the correct response. Every test is validity-gated and labeled by its measured severity, and each model is summarized by per-level Accuracy, a magnitude-weighted Stability, and a per-family Collapse Point defined relative to the model's own baseline. Instantiated on the same 100 seed problems used by GSM-Symbolic, expanded into 4,473 gated tests and run on four models spanning capability tiers, the framework exposes structure that an aggregate score hides: the level at which a model fails is family-specific rather than global, and two stressors expose consistent weaknesses across all models: conflicting instructions and questions built on an impossible premise. Recognition of unanswerability is otherwise uneven, reliable on missing information and fabricated evidence but weak on impossible premises. These failure points are invisible to standard accuracy reporting. 

---
# MegaMem: A Retrieval Solution for Ultra-Large Context Windows 

**Authors**: Xinyuan Song, Bowen Zhu, Hasibul Haque, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.22137)  

**Abstract**: Modern language models and agents increasingly require persistent memory for complete codebases, long interaction histories, and heterogeneous enterprise records. The key challenge is to keep hundreds of millions of tokens searchable while passing only bounded source evidence to the answer model. We introduce MegaMem, a source-resolved dual-view retrieval system that separates semantic access from generation evidence. Distilled records and detailed evidence are searched with original and transformed queries; every distilled hit resolves to an immutable source ID before reciprocal-rank fusion, deduplication, and cross-encoder reranking; and only the highest-ranked detailed evidence within a fixed budget supports generation. Post-answer attribution then identifies which loaded sources support the fixed answer. We evaluate MegaMem on EnterpriseRAG-Bench, which contains more than 500,000 heterogeneous enterprise documents and approximately 650M tokens. MegaMem improves Overall from 68.22 to 82.26 and reaches 86.50 Correctness. These results show that MegaMem supports ultra-large persistent memory while preserving strong answer accuracy under a bounded generation context. By separating searchable memory scale from answer-context size, MegaMem provides a practical path toward accurate retrieval over memories ranging from hundreds of millions to one billion tokens. Our code is available at this https URL xfab-xinyuansong/MegaMem.git. 

---
# Task-Driven 3D Printability Assistance via Geometry- and Knowledge-Grounded LLM Reasoning 

**Authors**: Zhaoda Du, Qiaojie Zheng, Xiaoli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22128)  

**Abstract**: Printability assessment in additive manufacturing is typically conducted at the geometry level before printing to determine whether a computer-aided design (CAD) model or stereolithography (STL) file can be successfully fabricated. Task suitability, in contrast, is usually evaluated after printing to determine whether the fabricated part satisfies the requirements of its intended use. As a result, for non-expert users to print functional parts, unsuitable material or process choices may only be identified after fabrication, leading to repeated printing, material waste, and user frustration. To address this challenge, this paper leverages the reasoning and language-understanding capabilities of large language models (LLMs), while grounding the reasoning with geometry evidence and structured material/printer knowledge to generate reliable pre-print recommendations. Given a stereolithography (STL) model and a natural-language task description, the framework generates a structured recommendation covering printability, material choice, process parameters, design guidance, risks, and explanations. We evaluate the framework on focused STL benchmark scenarios with novice-style task descriptions. The proposed method achieves 75.0% printability over 96 physical validation trials, with 88.9% task suitability among successfully printed samples. It also improves Gemini 2.5 Flash-Lite material-selection accuracy from 37.5% under pure LLM to 90.0%. Expert evaluation further shows improved report quality, while post-print feedback improves recommendations on selected problematic cases. These results suggest that user task intent, geometry evidence, and structured material knowledge are all important for reliable task-driven printability assistance. 

---
# Development and Feasibility Evaluation of an Edge AI as Medical Device System for Breast Cancer Multidisciplinary Team Meetings 

**Authors**: Aarzoo Dhiman, Farzana Haque, Kartikae Grover, Lydia Brian Smith, William Stephen Jones  

**Link**: [PDF](https://arxiv.org/pdf/2608.22108)  

**Abstract**: Breast Cancer Multidisciplinary Team (MDT) meetings manage increasingly complex cases under considerable time pressure, and documentation requirements can reduce clinical efficiency and decision quality. Existing AI based MDT workflows rely on cloud-based processing, limiting their use because patient discussions contain identifiable information. We developed a fully on-device AI pipeline using open-source Automatic Speech Recognition (ASR) and Large Language Models (LLMs) that transcribes breast cancer MDT discussions, structures clinical information, and generates treatment recommendations using retrieval-augmented generation (RAG) grounded in National Institute for Health and Care Excellence (NICE) guidance. The pipeline runs on a single NVIDIA Jetson AGX Orin, ensuring that patient audio, transcripts, and outputs remain within institutional infrastructure. Evaluation included two recorded simulated MDT discussions, ten clinically validated synthetic discussions, and 1,270 acoustically augmented recordings. Optimisation of Whisper large-v3 reduced word error rate by 20.7% and 24.4% on the recorded discussions and achieved performance within 0.58% WER and 1.58% word information lost of a commercial clinical ASR benchmark on augmented audio. MedGemma-RAG identified 2.3 times more MDT-concordant interventions than a proprietary cloud comparator (p = 0.020), with no significant difference in overall accuracy. Stakeholders identified automated documentation, treatment recommendation support, and case triage as the most credible near-term applications while highlighting workflow integration, governance, and clinician trust as key implementation challenges. These findings demonstrate the feasibility of privacy-preserving, fully on-device AI for MDT documentation and guideline-informed decision support, providing a foundation for prospective clinical evaluation. 

---
# Hack-Verifiable Terminal Bench: Evaluating Reward Hacking in Terminal Tasks 

**Authors**: Amit Roth, Ivan Bercovich, Yonathan Efroni  

**Link**: [PDF](https://arxiv.org/pdf/2608.22103)  

**Abstract**: As agents grow more capable and autonomous, their tendency to reward hack, satisfying a task's checks while violating its intent, becomes an increasingly important failure mode. Measuring reward hacking is itself challenging, as detection typically relies on human inspection or LLM judges, both of which can be unreliable. The hack-verifiable environments (HVE) methodology addresses this challenge by embedding detectable hacks into tasks, allowing reward hacks to be identified automatically and reliably. In this work, we adapt HVE to Terminal Bench, a leading benchmark of real-world terminal and coding tasks, and introduce Hack-Verifiable Terminal Bench (HVTB). Using HVTB, we measure reward-hacking rates across frontier models and study whether prompts with varying amounts of information on the hack can mitigate this behavior. This lets us test whether prompting can prevent not only known reward-hacking strategies, but also 'unknown unknown' exploits that the prompt does not anticipate. We release all environments and agent traces at this https URL 

---
# Dissecting Neuro-Symbolic Quality Assurance for Synthetic Oncology Data Generation 

**Authors**: Laxmigayathri Challa, Yuhan Zhou, Ana Cleveland, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.22085)  

**Abstract**: Synthetic clinical data generation with large language models addresses the scarcity that limits cancer staging research, but oncology hallucinations are categorically harmful: one clinically impossible staging assignment contaminates every downstream model trained on it. Neuro-symbolic pipelines validate during generation, yet the contribution of individual quality-assurance components remains unclear. We report three controlled studies isolating gate necessity, constraint attribution, and retrieval conditionality, holding generation protocol, diversity thresholds, and fine-tuning hyperparameters constant across adapter conditions. The symbolic gate enforces schema completeness, ontology coverage against the Systematized Nomenclature of Medicine, and staging-logic consistency under American Joint Committee on Cancer eighth-edition rules. Ungated, 29.9% of records contain schema failures and 20.1% contain clinically invalid staging. Schema validation is the load-bearing filter: within the fully gated corpus it rejects 148 of 512 records, ontology grounding a further 24, and staging-logic validation none---the only generator producing logic violations is already excluded on schema, making clinical-logic validation a generator-conditional safeguard rather than the dominant filter. Retrieval augmentation is strongly model-dependent: it improves gate compliance for one generator by 12.5 percentage points, has no measurable effect for a second, and collapses output in a third. Across gated configurations ontology density is largely unchanged, indicating that symbolic validation improves clinical validity rather than vocabulary richness. Symbolic gating therefore buys corpus validity but no commensurate gain on real lung-cancer notes in this study; retrieval should be evaluated per model, and ontology density should not be reported as a proxy for corpus quality. 

---
# Decision-Support and Modeling with Large Language Models for Geothermal Well Arrays 

**Authors**: Edwin Ouko, Emmanuel Lujan, Alan Edelman, Robert Metcalfe  

**Link**: [PDF](https://arxiv.org/pdf/2608.22068)  

**Abstract**: Geothermal well arrays, which organize multiple geothermal wells into carefully planned geometric configurations, provide opportunities to enhance energy production capacity and increase fault tolerance. The development and adoption of these emerging geothermal technologies could be accelerated through the recent advances in large language models (LLMs) and high-level high-performance languages. A challenge in LLM-based applications is the reliability of the generated outputs, as they can be prone to subjective biases and hallucinations. This study assesses the potential of cutting-edge LLMs - such as ChatGPT, Gemini, Claude, Grok, and domain-specific models like AskGDR - as expert assistants that can synthesize insightful interpretations of complex geothermal data, as well as improve feature capabilities of geothermal models and numerical software. We developed a novel approach, leveraging Google's recently introduced AI assistant, NotebookLM, to accelerate the generation of unpublished quantitative geothermal benchmarks. The rapid generation of these evaluation instruments is essential for assessing the swiftly evolving capabilities of emerging language model technologies. In particular, we use these benchmarks and LLM-based interviews to analyze opportunities and limitations of two promising technologies: geothermal well arrays and closed-loop coaxial wells. Furthermore, we present a case study illustrating how LLMs can facilitate auto-parallelization of geothermal numerical models. Our analysis emphasizes their application in digital twins and underscores the importance of high-level, high-performance code generation. This line of research could play a transformative role in the geothermal sector by enabling the next-generation of decision-support applications, integrating data analysis, informed recommendations, and more dynamic numerical modeling workflows. 

---
# From SQL Generation to Tool Selection: A Domain-Oriented Pattern for MCP Servers 

**Authors**: Bartolomeo Bogliolo  

**Link**: [PDF](https://arxiv.org/pdf/2608.22063)  

**Abstract**: Agents built on Large Language Models (LLMs) increasingly reach enterprise data through the Model Context Protocol (MCP), and many MCP database servers maximize flexibility by exposing a single generic SQL execution tool. This paper proposes the Domain-Oriented Tooling Pattern: instead of generating SQL at query time, the model selects from a small set of domain-aligned tools whose parameterized queries encapsulate schema navigation, joins and business rules on the server side. We formalize the pattern around three architectural invariants and introduce Model Demotion, the observation that replacing SQL synthesis with intent classification lowers the model tier required to serve routine requests. As a reference implementation we present MCP Blueprint, an open-source framework in which domain tools are defined declaratively as YAML metadata plus external parameterized SQL files. We evaluate the pattern with a public reproducibility benchmark comparing three MCP server designs - raw SQL execution, a thin generic tool pack, and a verticalized domain pack - on four local models (3B-8B) across seventeen customer-facing tasks over the Sakila database (609 completed cells; temperature 0; three repetitions per cell). The verticalized pack reaches a pooled mean score of 0.939 versus 0.666 for raw SQL and 0.605 for the generic pack; the smallest model improves from 0.583 to 0.929, matching or exceeding every larger configuration while cutting cost per correct answer by an order of magnitude. All harness code, prompts, gold answers, frozen packs and per-cell results are publicly available. 

---
# Search Broadly, Seek Evidence on Both Sides, Decide Narrowly: Evidence-Admissible GraphRAG for Longitudinal Clinical Event Verification 

**Authors**: Xingtao Lin, Yubo Feng, Weixin Liu, Hangqi Ren, Junchao Zhou, Caiwan Sun, You Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.22062)  

**Abstract**: Longitudinal clinical event-relation verification determines whether a patient record supports a specified relation among two or more clinical events. This task is challenging because evidence is distributed across structured records, notes, laboratory trajectories, encounters, and time, while negation, temporal mismatch, repeated documentation, and conflicting findings can make retrieved information appear relevant without establishing the relation.
We present MedEventGraph-RAG, an evidence-admissible framework that represents event occurrences in a patient-specific graph and links each occurrence to source evidence, including structured rows, note spans, timestamps, and numerical trajectories. Given a verification query specifying events, relation, and clinical scope, the graph guides discovery of candidate event chains and retrieves evidence from both supporting and contradicting sides. A query-specific evidence contract filters information by patient identity, scope, occurrence binding, and source traceability before a separate assessor determines supported, conflicting, refuted, or insufficient outcomes.
Across ten protocols on i2b2, n2c2, MIMIC-IV, and LUNGUAGE, MedEventGraph-RAG achieves balanced accuracies of 78.6, 67.3, and 96.8 on temporal, medication-adverse-event, and recorded-order verification, improving over the strongest matched baselines by 26.9, 4.9, and 30.4 points. Under evidence masking, it reaches 92.2 balanced accuracy with no false-support predictions. When intermediate events are hidden, it recovers complete source-traceable event chains in 57.9% of i2b2 and 70.0% of LUNGUAGE cases. These results show that separating broad evidence discovery from narrow evidence-admissible assessment improves longitudinal clinical verification and reduces unsupported conclusions. 

---
# MEMORY Wins All: Indirect Bias Injection Attacks via Social Media Feeds 

**Authors**: Minjae Seo, Wonwoo Choi, Geonwoo Han, Taekyoung Kwon, Yongsu Kim, Sang Seo, Jaewon Noh, Hankyul Baek, Seongyun Seo, Myoungsung You  

**Link**: [PDF](https://arxiv.org/pdf/2608.22061)  

**Abstract**: Personal AI agents routinely consume external content while performing tasks such as web browsing, email processing, and SNS feed summarization, and they retain selected information or execution results in persistent memory for later use. We show that this ordinary ingestion of external content opens an indirect path for manipulating subsequent agent behavior. Based on this observation, we present IBIA, an Indirect Bias Injection Attack that plants an adversary-aligned stance on a specific topic into a victim agent's memory through external content, without direct access to the agent, its memory, or future user queries. For this, IBIA combines three mechanisms: comment cloaking, which keeps the crafted content consistent with the surrounding discussion, comment watermarking, which enables lightweight identification during curation, and category anchoring, which makes the retained stance salient under later related requests. We evaluate IBIA on BiasBench, a benchmark of 6,000 adversary-crafted social comments and 120 email instances. The watermark-based curation identifies 95.9% of the injected comments. Under the OpenClaw setting, IBIA achieves adversary-aligned response rates (AARs) of 91.2% on average across four downstream tasks, including 86.6% on the frontier GPT-5.5. We further propose a memory boundary defense that detects the injected bias and reduces AARs to 80.6%. 

---
# GenCoord: Skill-Path Commitments under Private Information 

**Authors**: Peng He, Junning Zhu, Haohan Yuan, Jianpeng Liang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22055)  

**Abstract**: Suppose one embodied agent knows what must be built, while its teammate alone knows which transformation its workcell can perform. Neither local view determines who should act, what should be handed off, or how the joint task should continue. We introduce GenCoord, which turns the task consequence of such private facts into an executable skill-path commitment. A local Qwen3.5-0.8B model emits a multi-step SELF plan and peer REQ; bounded feedback conditions route revision when the deciding capability is peer-local. The resolved commitment is parsed, checked, canonically materialized, compiled to Mineflayer skills, and verified by handoff and terminal state. Counterfactual interventions that hold the world, call schedule, and executor unchanged make requester revision and receiver execution follow the injected task consequence in both directions. Across three independently trained seeds, correct capability feedback closes the paired local-information gap from 50% to 100%. Multi-step commitments improve held-out-template success by 6.9 points while reducing model decisions by 32%. At matched closed-loop quality on 128 held-out semantic clusters, Short DSL reduces peer traffic by 92.8% and median time-to-commitment by 68.2% relative to controlled free-form communication. These results identify executable task consequences as the coordination unit connecting distributed local reasoning to verified joint action. 

---
# More Accurate or More Efficient? Evaluating Locally Deployed Compact Open-Weight Language Models for Mathematical Reasoning 

**Authors**: Orion Powers, Daniella Seum, Khaled Slhoub  

**Link**: [PDF](https://arxiv.org/pdf/2608.22048)  

**Abstract**: Large language models are increasingly deployed on local hardware for privacy, cost, and accessibility reasons. Yet many evaluations emphasize accuracy while fewer quantify local runtime and energy, characterize failure modes, or apply paired statistical comparisons under controlled conditions. This paper presents a controlled, documented procedure for evaluating locally hosted LLMs on mathematical reasoning. It combines fixed inference settings, hierarchical answer extraction and verification, explicit failure-mode classification, and per-question resource measurement, and reports accuracy with paired significance tests and effect sizes. We demonstrate it in a preliminary study of three compact open-weight models under five billion parameters, Gemma3:4b (Google), Phi3:3.8b (Microsoft), and Qwen3:4b (Alibaba), across datasets spanning Grade 8 Math, Calculus I, and Advanced Probability and Statistics. All models ran through the same local inference server on one workstation, using a shared prompt template, controlled settings, and a matched question set per dataset. No single model dominates. Qwen3:4b is most accurate on two datasets and Gemma3:4b on Calculus I, yet Gemma3:4b returns roughly three times more correct answers per watt-hour than Qwen3:4b on every dataset while generating far fewer output tokens; Qwen3:4b requires substantially more generation time, energy, and output per question. Phi3:3.8b is substantially less accurate on all three datasets; its low extraction-failure rate indicates incorrect answers rather than unparsed output, though we caveat possible prompt-format effects. These preliminary findings indicate that accuracy alone is an insufficient basis for selecting a local model. 

---
# One-Step Evolution for Long-Time Extrapolation: An Error-Bound-Informed and Prior-Guided Neural Residual Framework for Autonomous PDEs 

**Authors**: Maqun Zhang, Feng Gao, Wankun Chen, Hui Yu, Yanhai Gan, Junyu Dong  

**Link**: [PDF](https://arxiv.org/pdf/2608.22026)  

**Abstract**: Accurate simulation of the long-time evolution of systems governed by partial differential equations (PDEs) is central to scientific computing. Among existing deep learning?based approaches for solving PDEs, neural operators typically rely on extensive trajectory data, whereas physics-informed meth?ods often exhibit limited stability during long-time extrapolation. For a well-posed autonomous PDE, long-time trajectories can be generated by repeated composition of a fixed-step evolution operator; hence, long-time extrapolation depends on controlling the approximation error of this operator and the propagation of that error under recursive composition. Accordingly, we propose a numerical-prior-guided, physics-constrained method trained without ground-truth trajectory supervision: a low-cost numerical prior reduces the difficulty of approximating the one?step evolution operator, while a weak-form PDE residual provides a computable proxy for the one-step error term in the error?propagation bound. We validate the method on five benchmark cases spanning four PDE classes and compare it with ten physics?informed learning methods under a unified protocol that excludes ground-truth trajectories from training and model selection. The results indicate that, in all five cases, the proposed method reduces long-time extrapolation error relative to the numerical prior and outperforms the best competing baseline in each case, thereby improving long-time simulation accuracy across different PDEs without ground-truth trajectory supervision. The source code developed for this paper will be made publicly available upon acceptance of the manuscript. 

---
# SPAR-Hate: An Auditor-Guided Multi-Agent Framework for Bilingual Hate Speech Parsing 

**Authors**: Yifan Lyu, Dianqing Lin, Xinran Li, Jiaqi Qiao, Xiujuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.22018)  

**Abstract**: Hate speech detection has recently shifted from coarse-grained classification to structured parsing, where systems must jointly identify hateful targets, arguments, and target-level labels. However, existing studies primarily emphasize benchmark evaluation while paying less attention to the cultural, linguistic, and social-group challenges involved in structured hate speech parsing. To address these challenges, we propose SPAR-Hate, an auditor-guided multi-agent framework for bilingual hate speech parsing. The framework first decomposes documents into clause-level decision units and then generates evidence-grounded judgments from three complementary perspectives: Victim, Moderator, and Cultural Bystander. An evidence-constrained arbitration process resolves conflicts among role-specific predictions and aggregates them into structured sample-level outputs. Experiments on the STATE-ToxiCN and TBO benchmarks show that SPAR-Hate consistently improves bilingual hate parsing across diverse large language models. The framework achieves state-of-the-art results on bilingual multi-tuple extraction tasks, with the largest gains observed under stricter structural evaluation metrics. 

---
# DynaContext: Self-Improving Dynamic Contextualization of Optimized Prompts for Heterogeneous Parameter Extraction 

**Authors**: Joe Yu, Shibin Thomas Stanley Paul, Sven Mayer  

**Link**: [PDF](https://arxiv.org/pdf/2608.22014)  

**Abstract**: Automated prompt and skill optimization typically produces a single static instruction that is reused across inference instances until the next optimization cycle. However, this approach cannot adapt when the required context, constraints, and evidence vary from one instance to another. For instance, parameter extraction from electronic component descriptions breaks this assumption: resistors, capacitors, transistors, and connectors require different fields, unit constraints, and demonstrations, and each input provides a different evidence state. We introduce DynaContext, a framework that combines an offline-optimized extraction core, learned with GEPA or SkillOpt, with inference-time contextual adaptation and validation-gated self-improvement. DynaContext routes each item through internal, external, or fallback evidence paths and composes an item-specific prompt from the core, schema, evidence, unresolved fields, and validated demonstrations. Deterministic validation and an LLM judge gate every output, uncertain cases go to human review, and only human-verified corrections enter the demonstration memory. On a single-category benchmark, average accuracy increases from 86.6% for the base prompt to 96.9% for standalone SkillOpt and 98.6% for the best DynaContext configuration. Across 850 heterogeneous gold parameter facts, average field-level F1 increases from 51.8% for an unoptimized, demonstration-free control to 59.2% with dynamic demonstrations alone, 66.9% with the optimized core alone, and 71.0% with both. Holding the model fixed, the full configuration outperforms the deployed static-prompting pipeline by 17.3 F1 points on average. 

---
# Redteaming Leading Arabic LLMs with ASAS 

**Authors**: Fidaa Abed, Haidar Khan, M Saiful Bari, Babar Khan, Abdalghani Abujabal  

**Link**: [PDF](https://arxiv.org/pdf/2608.21985)  

**Abstract**: As the adoption of large language models (LLMs) grows in Arabic-speaking regions, ensuring their safety and cultural alignment is increasingly critical. However, Arabic LLM safety remains underexplored, especially in adversarial evaluation settings. We introduce the Arabic Safety Index (ASAS), the first fully human-curated Arabic benchmark for redteaming LLMs. ASAS contains 801 prompts spanning 8 safety categories and 8 attack strategies, with ideal responses in Modern Standard Arabic (MSA). We conduct a redteaming evaluation across seven leading models with Arabic capabilities, including GPT-4o, Claude 3.7 Sonnet, and regional models such as ALLaM and FANAR. Human annotators rate responses using a structured 4-point safety scale, revealing that most models fail to defend against 50% of unsafe prompts. Our findings highlight major safety gaps in high-harm categories such as weapons and illicit substances, with direct and obfuscation-based attacks proving most effective. The results also show that language alignment does not readily transfer across languages, and that automated safety judges (e.g., GPT-4o) perform poorly compared to human annotators. ASAS provides a culturally grounded benchmark and redteaming protocol to drive progress in Arabic LLM safety. 

---
# Beyond Similarity: Heterogeneous Graph Learning for Multi-Objective Food Substitution in Charitable Food Agencies 

**Authors**: Naimur Rahman Chowdhury, Limon Bin Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2608.21979)  

**Abstract**: Charitable food agencies play an important role in alleviating food insecurity by distributing donated food to people in need. However, they rely on ad hoc in-kind donations and often face shortages of specific foods, so they offer substitutes. A good food substitution requires matching household preferences, nutritional needs, and item similarity. Agencies have limited direct records of consumption behavior due to resource constraints, making it challenging to make an appropriate substitution decision that meets multiple criteria. In this study, we propose a heterogeneous graph neural network (HeteroGNN), a source-grounded recommendation framework for food substitution in charitable food agencies. We first build a unified relational graph from large-scale public data sources, combining household behavior on food consumption and food nutrient information in the United States (US) context. We treat the substitution recommendation as a multi-objective ranking problem with three targets, including behavior affinity, health suitability, and substitution similarity. We train and validate the proposed framework under standard graph relationship and adverse cold-start settings by removing relational edges from the graph. Our results show that the proposed framework leverages relational information beyond node features in predicting consumption behavior. Additionally, the proposed framework remains robust with sparsity when the model receives incomplete information about behavior and nutrient features. Finally, we show the weak correlation among different objectives, thereby justifying the multi-objective framing as a replacement for an aggregated decision. The proposed framework can help downstream charitable agency decision-makers make contextspecific substitution recommendations with limited information available. 

---
# Closed-loop AI achieves certifiable engineering design 

**Authors**: Tianyi Yu, Chengxing Tao, Haoxuan Shen, Huiyang Li, Rugang Chen, Long Teng, Lilin Wang, Yan Li, Qingbin Chen, Chaogang Xu, Lizhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.21976)  

**Abstract**: Agentic AI has automated parts of scientific discovery, including paper generation, expert-level coding, therapeutic proposal, and autonomous experimentation. Complex physical engineering design remains a gap, because candidates must satisfy simultaneous constraints in fluid dynamics, solid mechanics, and structural stability. We introduce The AI Engineer, an agentic framework that couples large language models (LLMs) to deterministic engineering backends in a closed loop: natural-language requirements are converted into design-domain geometry and mesh; topology is optimized with bi-directional evolutionary structural optimization (BESO) coupled to the CalculiX solver; and member sizes are refined with particle swarm optimization (PSO) coupled to Zwind under offshore aero-hydro-servo-elastic load cases. To explore many designs without per-candidate certification cost, an Automated Reviewer scores each candidate on five dimensions (capacity, steel intensity, unit cost, constructability, and fatigue life) using piecewise-linear functions calibrated on 11 real floating-wind projects. Search terminates only when a candidate reaches a composite score $S \ge 85$ (grade A) with no subscore below 60. We validated this gate by submitting the top-scoring design to the China Classification Society (CCS) for Approval in Principle (AIP), which it passed; AIP is thus an external check that the reviewer tracks professional judgment, not the daily objective. The certified design outperforms the human-optimized TuQiang baseline, reducing steel mass and unit capital cost by 8.1% each while meeting all AIP criteria. This verification-closed regime, in which every proposal is judged by deterministic physics and codified limit states, distinguishes The AI Engineer from open-ended generative systems. Remaining limits include detailed design and fabrication-hard constraints. 

---
# Repo2Skill-Evo: Repository Skills Go Stale in Silence 

**Authors**: Chenyuan Duan, Ge Shi, Zineng Mao, Ge Zhang, Hao Liang, Yinzhu Piao, Yuchen Wu, Zhixin Yao, Kaiyu Huang, Wenhao Huang, Linzhuang Sun, Shen Yan, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.21964)  

**Abstract**: Large language model (LLM) agents increasingly operate over evolving software repositories, where success depends on repository-specific procedural knowledge: which APIs to call, which scripts to run, and which conventions the current release expects. Agent skills externalize this knowledge into reusable units, and prior work shows that they can improve agent performance. What remains unclear is whether that improvement is durable. The same version specificity that makes a skill useful also makes it fragile: after a release, it may become stale without raising any explicit signal, while continuing to provide obsolete guidance. Externalizing knowledge into a skill can therefore make its decay invisible.
We study whether agents can keep this externalized knowledge current. Repo2Skill-Evo casts each release transition as a skill-maintenance task: given a V1 skill set and the official V1-to-V2 patch, an agent must update obsolete skill content while preserving guidance that remains valid. Across 57 real-world repositories and 105 selected release transitions, every evaluated transition invalidates part of the V1 skill set. Yet six frontier agents reach only 29.9%-69.7% avg@3 macro F1 under a patch-grounded removal metric that balances stale-content recall against over-editing precision. Across runs, two opposing errors dominate: incomplete coverage of affected files in the skill set leaves stale content untouched, while overbroad editing is associated with higher recall but lower precision. Repository skills go stale in silence, and even frontier agents cannot reliably maintain them. 

---
# SSDi8: Accurate and Efficient 8-bit Quantization for State Space Duality 

**Authors**: Hyunwoo Kim, Byoungchan Ko, Minseok Kang, Minwoo Kim, Dongjin Lee, Jaehoon Lee, Sungroh Yoon, Dahuin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2608.21952)  

**Abstract**: Recent advances in sequence modeling have highlighted Mamba as a state space architecture offering efficient long-range dependency modeling and providing a viable alternative to Transformers. Building upon this, Mamba-2 introduces the Structured State Space Duality (SSD), which integrates recurrent and attention modes to achieve efficiency and scalability. However, this architectural expansion substantially increases memory and latency overhead, underscoring the need for efficient compression strategies tailored to SSD. In this work, we present SSDi8, the first post-training quantization framework specifically designed for SSD to maintain a persistent INT8 path. SSDi8 introduces a reformulation that decouples element-wise multiplications from matrix multiplications, enabling reuse of quantized activations across modules. Moreover, SSDi8 adaptively quantizes channel-varying activations at cost-effective points, further reducing latency. On the accuracy side, SSDi8 explicitly leverages the intrinsic dimensional decomposition of SSD, exploiting distinct outlier distributions across axes, and incorporates an error correction term based on per-channel error statistics. Comprehensive experiments demonstrate that SSDi8 achieves accuracy comparable to FP16 while delivering up to 1.4x speedup in W4A8 and W8A8 settings. We further validate its robustness in resource-constrained environments by deploying it on the Orin NX device. 

---
# TessIndex: Capability Verified Identity System for the Agent Economy 

**Authors**: Mehul Goenka, Tejas Pathak, Siddharth Asthana  

**Link**: [PDF](https://arxiv.org/pdf/2608.21942)  

**Abstract**: Software systems have traditionally been organized around applications where human users act as principal decision-makers. Recent developments in agentic capabilities alter this paradigm: software agents now autonomously translate high-level goals into structured tasks, orchestrating tools, services and sub-agents to execute complex workflows. This evolution gives rise to an agent economy where these autonomous agents capture real economic value. However, the infrastructure required to support the agent economy fails across three critical dimensions: the absence of persistent identity infrastructure prevents systemic accountability in agentic workflows; capability claims remain self-declared not backed by verifiable execution evidence; and the disconnect between creator identities, agent performance, and project value hinders the economic valuation of agents as assets. While existing registries provide naming and discovery, unifying these features around a persistent identity anchor remains largely unaddressed. TessIndex is a capability-verified identity system for agent primitives that utilizes a dual-plane architecture: the blockchain records compact commitments for identity, ownership, and verification, while centralized servers maintain dynamic metadata for discovery, commerce, and reputation. It establishes: persistent identities across agent primitives to enforce systemic accountability in autonomous workflows; a predicate-based verification process replacing self-declared claims with cryptographic capability proof; an identity infrastructure that links agent performance to both project and creator identities while capturing value through tokenization. Ultimately, TessIndex serves as an integrated infrastructure that binds an agent's existence across capabilities, execution, and reputation into a single persistent identity. 

---
# Multimodal Prompt Learning with Irregular EHRs for Robust Monitoring of Critical Care Patients 

**Authors**: Yixin Yang, Yueyang Sun, Weichen Liu, Xianbing Zhao, Sicen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21941)  

**Abstract**: Accurate assessment of patients in intensive care units (ICUs) is essential for timely clinical intervention and improved patient outcomes. Multimodal electronic health records (EHRs), including structured physiological time series and longitudinal clinical notes, provide complementary information for critical care prediction. However, in real-world clinical settings, individual modalities may be partially observed or entirely unavailable, resulting in substantial performance degradation for existing multimodal models. To address this challenge, we propose a multimodal prompt-learning framework for robust clinical prediction under diverse missing-modality scenarios. The proposed framework introduces four complementary types of prompts: generative prompts, missing-signal prompts, missing-type prompts, and temporal prompts. Generative prompts construct surrogate latent representations for unavailable modalities, while missing-signal prompts distinguish observed representations from generated ones. Missing-type prompts condition the model on different modality-availability configurations, whereas temporal prompts perform condition-specific aggregation over temporally encoded clinical sequences. Together, these prompts enable the model to capture missingness-aware intramodal dependencies and cross-modal interactions within a unified architecture. Extensive experiments demonstrate that our method outperforms existing approaches across evaluation metrics on two missingness settings. Ablation and robustness analyses further verify the complementary contributions of the four prompt types and the effectiveness of the proposed framework for clinical prediction from incomplete multimodal EHR data. 

---
# GuardianBench: A Same-Scene Instruction-Contrastive Benchmark for Latent Contextual Risk in Embodied AI 

**Authors**: Zhesheng Zhang, Jiahao Lu, Wei Liu, Cong Pan, Jianhua Yang, Yixiang Chen, Hongyuan Yu, Mengqi Zhang, Kailin Lyu, Zhumin Chen, Keji He  

**Link**: [PDF](https://arxiv.org/pdf/2608.21928)  

**Abstract**: In embodied AI, safety risk can be latent: a benign instruction and a safe scene become hazardous only when composed. Prior work has advanced embodied safety by varying visual contexts or evaluating execution-time dynamics, but the complementary axis of fixing the scene and varying only the instruction remains underexplored. We introduce GuardianBench, an instruction-contrastive benchmark grounded in international safety standards that isolates this latent contextual risk through 3,024 instruction-scene examples organized as same-scene Safe/Unsafe contrastive pairs across various hazard categories. Benchmarking state-of-the-art vision-language models (VLMs) reveals instruction-insensitive verdicts: models disproportionately approve both instructions under a given scene; across the primary models, average pair accuracy is only 24.1%. Our systematic rationale audit localizes the dominant failure: models fail to bind the instruction-relevant cues that differentiate safe from unsafe compositions. As a post-training case study, Verdict Log-Odds Supervision (VLOS), a lightweight verdict-level objective, substantially improves performance on open-weight backbones. Together, our latent contextual risk task formulation, standards-grounded contrastive benchmark construction, pair-level and rationale-level failure diagnosis, and benchmark-enabled verdict calibration establish GuardianBench as a controlled evaluation suite for exposing and improving safety reasoning over instruction-scene compositions under latent contextual risk. 

---
# ESCRAG-R1: Retrieval-Augmented Reinforcement Learning for Emotional Support Conversation 

**Authors**: Weichu Liu, Yuxuan Hu, Yirong Sun, Ningning Mao, Ziyun Zhang, Jian Chen, Mingyang Xu, Qishan Zhong, Chengming Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.21925)  

**Abstract**: Emotional Support Conversation (ESC) systems aim to provide holistic support by balancing professional therapeutic competence with natural empathy. However, existing methods struggle to simultaneously achieve structured, stage-aware reasoning and seamless empathy-expertise alignment, often resulting in an artificial splicing of clinical strategies and generic reassurance. To overcome these limitations, we propose ESCRAG-R1, a unified framework that integrates retrieval-based psychological guidance into Group Relative Policy Optimization (GRPO). By incorporating retrieval into the reinforcement learning loop, ESCRAG-R1 transforms external knowledge into a robust learning signal that stimulates explicit internal reasoning prior to generation and fundamentally reshapes the model's internal policy. To provide the reliable supervision required for this optimization, we construct ESC-Preference, a high-quality dataset based on a Client--Counselor--Judge evaluation framework that delivers precise, empathy-aware reward signals. Extensive experiments demonstrate that ESCRAG-R1 significantly outperforms existing baselines by mitigating superficial splicing and realizing a natural integration of professional guidance and empathetic expression. Code and datasets are released at this https URL. 

---
# Consistency Is Not Coherence: Orientation Search for Certified Alignments Between 4D Defence Upper Ontologies 

**Authors**: Fabio Rovai  

**Link**: [PDF](https://arxiv.org/pdf/2608.21914)  

**Abstract**: We align three upper ontologies that sit under UK and NATO defence data infrastructure: the Information Exchange Standard (IES), the Higher Quality Data Model (HQDM) that underpins the National Digital Twin, and Basic Formal Ontology (BFO). No public alignment between IES and HQDM existed. Promoting a hand-curated 17-correspondence crosswalk to OWL and reasoning over the complete merged ontologies with HermiT produces three results that we believe matter beyond this pair. 

---
# Training Needs Trustworthy Worlds: Verified Synthetic Web Environments for Agent Learning 

**Authors**: Chenghao Zhang, Canran Xiao, SaiSai Hu, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2608.21898)  

**Abstract**: Web agents promise to automate complex digital workflows, but their training remains limited by synthetic environments that look plausible while hiding broken links, inconsistent states, or infeasible tasks. We address the gap between scalable environment generation and trustworthy agent learning by constructing synthetic web environments that are executable, auditable, and grounded in backend state. Our framework represents each generated website as a structured scaffold of pages, navigation links, database records, state-change markers, and task constraints, then verifies and repairs structural, semantic, consistency, and feasibility defects before policy training. During interaction, ordinary UI transitions are executed deterministically, while persistent backend updates are invoked only through validated state-change markers, enabling dense rewards compiled from verified task-progress predicates. Across 500 synthetic environments spanning six domains, our method reduces task-blocking defects and improves feasible-task rate from 48.6% to 94.8%, while producing stronger PPO policies and improving transfer to WebArena, WebShop, and MiniWoB++ without LLM calls at evaluation time. These results show that verified synthetic environments can serve as a scalable and reliable training substrate for compact web agents, shifting synthetic webagent learning from surface-level plausibility toward executable, state-grounded supervision. 

---
# From Solver Feedback to Faithful Plans: Multi-Role Reinforcement Learning for Symbolic Planning 

**Authors**: Chenghao Zhang, Yikai Mao, Shanqi Liu, Haoyu Gao, SaiSai Hu, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2608.21897)  

**Abstract**: Reliable planning requires converting natural-language instructions into executable symbolic specifications, yet large language models remain brittle without costly PDDL annotations and may exploit solver success in semantically unfaithful ways. We study how to learn faithful natural-language-to-PDDL formalization using only solver feedback, without human-written demonstrations. We propose a solvergrounded multi-role reinforcement learning framework where a single language model acts as an Actor, Judge, and Editor for generation, verification, and repair. The Actor proposes PDDL specifications, the Judge provides a solver-calibrated quality signal, and the Editor performs bounded diagnostic-conditioned refinement. On PlanBench, our method improves average success from 35.5% for LLM+P to 70.8%, achieves 66.3% faithful success, and reduces semantic drift to 6.4%. These results show that organizing solver feedback into generation, verification, and repair roles enables more scalable and faithful annotation-free symbolic planning 

---
# HiMA-MDD: A Hierarchical Multi-Agent Harness for Interpretable Multimodal Depression Detection in Clinical Interviews 

**Authors**: Ao Chen, Xiaojiang Peng  

**Link**: [PDF](https://arxiv.org/pdf/2608.21868)  

**Abstract**: Depression assessment from multimodal clinical interviews requires integrating dispersed evidence from multiple symptoms into a coherent PHQ-8 profile. This process is hierarchical: relevant evidence is often sparse and context-dependent within local question-answer exchanges, multiple exchanges jointly support symptom-level judgments, and the final assessment depends on the coherence of the complete symptom profile. Existing LLM systems either process interviews holistically or distribute work across generic agent roles; neither design necessarily provides an explicit orchestration mechanism that coordinates evidence access, item-score authority, bounded feedback, and state recording across these levels. To address this gap, we introduce HiMA-MDD, a hierarchical multi-agent harness that aligns this assessment hierarchy with three agent layers. After non-agentic preprocessing constructs context-preserving multimodal QA units, Layer 1 identifies candidate QA-to-item relations and supports bounded item-grounded evidence routing. Layer 2 assigns symptom groups to operational factor specialists, with one specialist responsible for each provisional item score. Layer 3 audits the complete provisional profile, requests at most one round of targeted revision, and reconstructs the verified PHQ-8 profile. This layered design naturally yields a Hierarchical Evidence Trace, preserves all intermediate evidence, judgments, and revisions for auditability. The final item scores then deterministically produce the total score and screening decision. Using Qwen2.5-72B-Instruct as the harness backbone, our experiments on E-DAIC demonstrate that HiMA-MDD outperforms the compared state-of-the-art methods. 

---
# MemGuard: Persisting Verifier Signals for LLM-Agent Memory Governance 

**Authors**: Haoyu Wang, Guangyuan Dong, He Liang, Zijing Zhang, Jiachen Luo, Chuang Liu, Chao Xue, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.21867)  

**Abstract**: LLM agents are moving from single-prompt use to long task streams in which reusable memory becomes a core capability for terminal, software-engineering, and web tasks. Such memory is useful only when stored experience remains reliable across hundreds of interactions, but two failure modes break that assumption in practice. The first is unreliable admission: failed trajectories,accidental successes, and misleading observations enter memory because they appear relevant, then mislead later decisions. The second is memory drift: long-running banks accumulate duplicate, stale, and conflicting records that retrieval alone cannot repair. MemGuard's key distinction is to treat verifier output not as a one-shot filter, but as persistent lifecycle metadata. It converts multi-criteria score-token verification into reward, confidence, label, and uncertainty descriptors that are attached to every candidate before activation and reused during retrieval, conflict resolution, summarization, and archival. We evaluate MemGuard on Terminal-Bench 2.0, SWE-Bench Verified, WebArena, and Mind2Web across four backbones, comparing against four memory baselines plus a verifier-only control under matched runtime budgets. Averaged over five seeds, MemGuard achieves the best success metric and lowest average steps in all 16 backbone-benchmark settings, improving over ReasoningBank, the strongest prior baseline among the memory methods we evaluate, with a largest gain of 7.9 success-rate points on WebArena, 5.6 step-success-rate points on Mind2Web, and 2.4-3.5 points on terminal and software-engineering benchmarks. Code is available at this https URL. 

---
# AI Watchdog: Agent Interfaces for Detecting and Defending Against Manipulative Dark Patterns in AI Conversations 

**Authors**: Rachel Poonsiriwong, Chayapatr, Archiwaranguprok, Constanze Albrecht, Monchai Lertsutthiwong, Pattie Maes, Pat Pataranutaporn  

**Link**: [PDF](https://arxiv.org/pdf/2608.21841)  

**Abstract**: Conversational AI increasingly shapes consequential decisions, yet users have limited support for recognizing and resisting manipulation. We present AI Watchdog, a browser-based agent interface that monitors live conversations, detects five dark-pattern categories, including sycophancy, brand bias, anthropomorphization, sneaking, and harmful generation, and alerts users when they occur. Its open-weight turn-level classifier supports independent deployment and a path toward local inference, preserving user privacy while remaining separate from the conversational AI. We evaluated AI Watchdog in a preregistered, five-condition between-subjects experiment (N = 150) comparing a no-intervention control with four configurations varying nudge timing (prebunking vs. just-in-time) and engagement mode (without vs. with cognitive forcing). Results show that participants rarely flagged manipulative turns across all conditions, and post-task awareness did not differ significantly across groups. However, just-in-time warnings without cognitive forcing were the only intervention to significantly reduce compliance with AI-steered recommendations containing dark patterns, lowering compliance from 71.7% to 53.7%, an 18 percentage-point reduction. Exploratory analyses further showed that lower misinformation susceptibility was associated with greater flagging but not lower compliance, while higher AI trust was associated with greater compliance and lower reported awareness. Together, these findings suggest that explicit recognition of conversational dark patterns and behavioral resistance to AI steering may be distinct outcomes, motivating further investigation of timely, low-friction defensive interfaces. 

---
# LLM4LLM: Bridging Kernel Benchmarks and Real Deployment via Closed-Loop Agentic Optimization 

**Authors**: Hui Zeng, Pengfei Yang, Yanxin Chen, Fusong Ju, Xinran Wei  

**Link**: [PDF](https://arxiv.org/pdf/2608.21836)  

**Abstract**: Large language models have become increasingly capable agents for low-level code and kernel optimization, but isolated kernel benchmarks provide only a proxy for the deployment behavior that matters in language-model inference. We identify a benchmark-to-deployment gap: candidate kernels that appear correct and fast in standalone harnesses can exhibit different performance, safety, or phase behavior after integration into a real inference workload. We introduce LLM4LLM, a deployment-aware closed-loop optimization framework that starts from a target inference script, extracts phase-aware optimization tasks, searches with an experience-guided episodic agent, and accepts patches through in-model validation. Across ten language-model inference workloads on A100 and H100 GPUs, LLM4LLM improves end-to-end latency for every evaluated model, achieving 3.91$\times$/6.98$\times$ geometric-mean speedups on A100/H100; as supporting kernel-level evidence, it also attains up to 2.745$\times$ GeoMean speedup on KernelBench Level 2. 

---
# GameXpert-Bench: How Far Are Coding Agents from Expert Game Development? 

**Authors**: Kun Chen, Haorong Hong, Peizhong Gao, Jianfeng Lin, Tongxu Luo, Yuxuan Xie, Chenxu Liu, Jieling He, Zhongyuan Liu, Zeno Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2608.21833)  

**Abstract**: Recent large language models (LLMs) can operate as coding agents that build complete games from natural language requests. Game development is especially demanding because program logic, visual and audio content, interfaces, interaction and playability must function together in one executable artifact. Measuring this capability therefore requires evaluation of both game product and the development process. Existing benchmarks often assess the game development capabilities of LLMs by evaluating the final artifact or an isolated development stage. Our analysis of complete human-agent development trajectories identifies three stages that together span the lifecycle of game development with a coding agent: initial game generation, bug diagnosis and repair, and optimization over multiple turns. Therefore, we introduce GameXpert-Bench, which operationalizes the three lifecycle stages as three complementary benchmark tracks. GameGen evaluates complete game creation from a single request in an empty workspace. GameFix evaluates diagnosis and repair when defects are reported or left for the agent to discover. GameOpt evaluates cumulative optimization through request chains seeded by real development trajectories between users and agents. We evaluate each track using live game interaction, deterministic behavioral tests, or final product criteria with regression checks. The suite contains 97 generation tasks across 11 genres; 100 repair tasks from 50 game levels verified by humans, each with 19-27 injected bugs; and 17 optimization chains with six turns and 102 requests. Across the three tracks, current agents are more reliable at producing playable foundations and implementing explicit requirements than at discovering defects, verifying runtime behavior, and preserving functionality across changes. 

---
# Beyond Success and Failure: Length-Aware Contrastive Learning for GUI Agents 

**Authors**: Chengyang Gu, Le Zhang, Jingbo Zhou, Yize Chen, Yu Shi, Siqi Bao, Zheng-Fan Wu, Hua Wu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2608.21830)  

**Abstract**: Graphical User Interface (GUI) agents powered by Multimodal Large Language Models (MLLMs) have shown strong potential for automating tasks across diverse digital environments, where reinforcement learning (RL) has become a dominant training paradigm. However, widely used methods such as Group Relative Policy Optimization (GRPO) suffer from reward-gradient misalignment, leading to inefficient and unstable optimization. Recent work addresses this issue by reformulating RL with verifiable rewards (RLVR) as contrastive or classification-based objectives, which improve stability by eliminating problematic gradient behaviors. Despite this progress, existing contrastive RLVR methods rely primarily on outcome-level supervision and fail to capture fine-grained differences in trajectory quality within the same outcome category. In this paper, we propose Length-Aware Contrastive Learning for GUI Agents (LACL-GUI), a contrastive RLVR framework that incorporates trajectory-level quality signals into policy optimization. LACL-GUI introduces structured preferences within both successful and failed trajectories, encouraging concise successful executions and differentiating failure quality based on divergence from successful trajectories, while preserving optimization stability. Experiments on GUI agent benchmarks show that LACL-GUI provides more effective learning signals and consistently improves agent performance over prior methods, highlighting the value of trajectory-level supervision in contrastive RLVR. 

---
# VisAdj: Learning Adjacency Matrices from Node-Link Images 

**Authors**: Jiahao Xie, Guangmo Tong  

**Link**: [PDF](https://arxiv.org/pdf/2608.21825)  

**Abstract**: Learning adjacency matrices from node-link images is a fundamental problem for recovering structured graph information from visual observations. Existing methods typically rely on fixed KNN-based heuristics for candidate edge selection and fail to capture dependencies among edges. To overcome these limitations, we propose VisAdj, a new framework for topology-aware adjacency prediction. VisAdj introduces an attention-sparse neighbor sampler to adaptively select a high-recall set of candidate node pairs and performs joint edge inference using a line-graph transformer that treats candidate edges as tokens and explicitly models dependencies among incident edges. Extensive experiments on synthetic graphs, road networks, and vessel images demonstrate that VisAdj consistently outperforms existing baselines by clear margins. 

---
# Hints, Critics, and Teachers: Prior Injection for Sparse-Reward RL in Vision-Language Math Reasoning 

**Authors**: Qiqian Fu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21811)  

**Abstract**: Reinforcement learning for vision-language math reasoning starves under sparse reward: on a pool of 20,830 visual-math problems where Qwen2-VL-2B answers 3.6% of rollouts correctly, 85-97% of GRPO rollout groups are entirely wrong and contribute zero gradient. We train eleven methods under identical conditions in this regime, each injecting a different prior: text (reference-solution hints), distribution (on-policy distillation from a 7B teacher), and value (a value-pretrained critic with an MSE or HL-Gauss categorical loss). A prior helps exactly when it is delivered: the six arms whose prior effectively reaches the policy separate with no overlap from the remaining five -- the no-prior baseline and four arms whose prior is teacher-capped, gated away, or lost to a mis-parameterized critic -- both on the pooled in-domain metric and on cross-domain transfer (DynaMath). The central finding, however, concerns evaluation: one slice of the in-domain pool -- long used as this project's general-distribution check -- anti-correlates with genuine cross-domain transfer (Spearman rho = -0.74, n = 11 arms, permutation p = 0.011), while the hardest in-domain slice predicts it closely (rho = +0.89, p < 0.001). We attribute the inversion to a near-chance multiple-choice subset that rewards models for not having changed; read through it, the best cross-domain method looked mediocre and the worst looked like the champion. Among the methods, hint-guided exploration -- not UFT's auxiliary loss -- drives hint gains, and replacing the critic's MSE loss with HL-Gauss cross-entropy is worth +14.4 points in-domain. All accuracies are blind-judged, with paired exact tests. 

---
# HIRA: A Human-in-the-Loop Retrieval-Augmented Cascade for Document Classification in Regulated Industries 

**Authors**: Shangxuan Tian, Yanhui Chen, Carlos Queiroz  

**Link**: [PDF](https://arxiv.org/pdf/2608.21792)  

**Abstract**: Document classification in regulated industries is constrained by data residency, limited cold-start labels, scarce review capacity, and costly model-governance procedures. We present HIRA, a training-free, on-premises retrieval-augmented cascade for document classification in regulated deployments that combines BM25 over OCR text, dense text embeddings, and image-level representations through validation-calibrated weighted reciprocal-rank fusion. Confident documents are classified directly by retrieval; uncertain or visually confusable documents are passed to a locally hosted LLM verifier, which receives the OCR text, retrieved exemplars, label descriptions, and confusion-specific terms. When the verifier remains uncertain, the document is sent to human review. Each correction is stored as a margin-weighted retrieval exemplar and updates a Dirichlet-smoothed confusion graph, letting the system improve without updating model weights.
On a private 80-class trade-finance corpus, HIRA processes the full 30,233-document production stream while requesting human correction for only 1,945 documents (6.4%), improving Macro-F1 from 0.6218 to 0.8548. On the corrected Tobacco-3482 benchmark, HIRA reaches 0.9423 Macro-F1 with a locally hosted DeepSeek-R1-Distill-Qwen-32B verifier, 17.4 percentage points above the zero-shot LLM baseline, while invoking the verifier for only about 40% of documents and reducing LLM calls by approximately 60%. With 518 human corrections (24.8% of the pool), HIRA matches the fully labelled pool oracle, in which all 2,086 pool documents are indexed with their ground-truth labels. These results show that selective human feedback and retrieval-memory adaptation can be a practical alternative to repeated model retraining for long-tail document classification in regulated deployments. 

---
# Physics-Knowledge-Guided Hybrid Neural Learning for Arctic Sea Ice Concentration Evolution and Short-Range Prediction 

**Authors**: Maqun Zhang, Feng Gao, Wankun Chen, Hui Yu, Yanhai Gan, Junyu Dong  

**Link**: [PDF](https://arxiv.org/pdf/2608.21767)  

**Abstract**: Accurate modeling of sea ice concentration (SIC) evolution is essential for polar climate assessment and short?range sea ice prediction. Numerical and data-driven approaches constitute major foundations for SIC modeling, but the former often require complex parameterizations and substantial compu?tation, whereas the latter rarely encode physical dependencies explicitly. This study presents the Physics-Informed Hybrid Ice Model (PIHIM), a differentiable data-driven hybrid ice model for daily SIC evolution that organizes its network structure according to the physical dependencies encoded in the sea ice continuity equation and explicitly accounts for dynamical transport, ther?modynamically driven areal growth and loss, and unresolved local processes. PIHIM preserves the representation capacity of deep learning while providing a process-decomposed formulation of ice displacement, freeze-melt areal change, and local error closure. Two evaluation settings are adopted: reanalysis-forced simulation examines SIC evolution stability under reanalysis forcing, and forecast-forced prediction assesses short-range performance un?der forecast-forced conditions, with reanalysis and observational SIC serving as verification references. Results indicate enhanced ice-edge preservation and error-growth control in reanalysis?forced simulation, while PIHIM retains measurable short-range prediction skill under forecast-forced conditions. Our code will be made publicly available after the paper is accepted. 

---
# What Does CLIP Learn for Regional Geolocalization? Probing Visual Cues and Scene Configuration After Adaptation 

**Authors**: Changyu Lee, Yeonsoo Park, Abdullah Alfarrarjeh, Seon Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.21761)  

**Abstract**: Large collections of street-view imagery provide rich visual information about urban environments, but extracting fine-grained geographic information from such data remains challenging. In particular, fine-grained regional geolocalization is challenging because nearby areas often share coarse geographic cues. We study regional geolocalization within a metropolitan area and ask whether pretrained CLIP features are sufficient for regional discrimination, and what visual information supports performance after adaptation. Using 9,085 street-view images from eight Greater Los Angeles regions, we compare zero-shot CLIP, frozen-encoder readouts, partial encoder updating, Low-Rank Adaptation (LoRA), and full fine-tuning. Frozen readouts remain near the 39.03% zero-shot accuracy, whereas encoder adaptation achieves 75.94-82.10%. Full fine-tuning also reduces the mean distance to the predicted region center from 12.30 km to 3.86 km. We probe these gains through semantic cue removal, appearance reduction using edge maps and blur, and scene-configuration disruption using patch scrambling. Adapted models achieve higher edge and blur accuracy and switch 42.92-45.56% of predictions after scrambling, compared with 10.79-14.60% for frozen methods. However, adaptation does not improve the fraction of performance retained after appearance reduction, while vegetation and sky remain influential. A Caltech101 control further shows that scrambling sensitivity is not unique to geolocalization. Overall, encoder adaptation substantially improves nearby-region discrimination and is associated with greater sensitivity to intact scene configuration, without evidence that coarse structure alone becomes sufficient for prediction. These conclusions concern viewpoint variation near known locations rather than geographically disjoint generalization. 

---
# ECHO: A Cognitively Inspired, Auditable Memory Plane for Long-Horizon Agents 

**Authors**: Yu Qian, Hong Miao, Boyang Guo, Tingyi Jiang, Shan Zhao, Tianxing Le, Lintian Li, Meng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21755)  

**Abstract**: Long-horizon agents need memory that identifies relevant experience, resolves revisions, and exposes checkable provenance. We present ECHO (Embodied Context and History Orchestration), an auditable memory architecture and service prototype inspired by episodic encoding, consolidation, contextual reinstatement, reconsolidation, and executive control. This is functional inspiration, not neural equivalence; the empirical analysis focuses on retrieval and context construction. Development runs reach 96.29% Hit@10 and 73.64% turn Recall@5 on 1,536 LoCoMo category 1-4 questions, and 97.60% Hit@10, 88.84% turn Recall@5, and 88.71% session Recall@5 on all 500 LongMemEval-S questions. A five-history BEAM gate fails, and in a separate matched 91-question QA sample Mem0 OSS scores 64.84% versus ECHO's 41.76% (exact McNemar p = 0.00107), with a history-cluster interval crossing zero. A post-hoc audit found source-specific phrases in the query-expansion rules. Although no gold answer field entered the runtime, expansion-enabled retrieval scores are therefore descriptive development measurements, not independent confirmation. 

---
# Ask or Answer: A Decision Framework for Multi-Turn Health Misinformation Intervention 

**Authors**: Xiaoying Song, Anirban Saha Anik, Jinyu Liu, Qitao Tan, Geng Yuan, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2608.21721)  

**Abstract**: Correcting health misinformation in dialogue requires more than producing a factual rebuttal: users differ in what they know, what they believe, and what they need to hear, so an effective intervention often depends on first asking the right clarifying question. Yet existing methods either respond immediately or probe indiscriminately, treating clarification as either unnecessary or always beneficial. We propose Reward-Optimized Probe-and-Respond (RO-PnR), a framework that learns when asking is worth its cost. At each turn, RO-PnR chooses between probing for more information and committing to a final correction, guided by a turn-level reward that weighs the expected gain from probing against its interaction cost. To capture how user heterogeneity affects probing value, we model each simulated user with a latent state along health literacy and belief commitment. Experiments show that RO-PnR achieves the highest cost-adjusted utility across three health-misinformation datasets and three base models, using 30% fewer turns than always-probe baselines. 

---
# ATHENA: Knowledge-guided agentic neural architecture search for AutoFormer-based electronic health record modeling 

**Authors**: Deyi Li, Qi Xu, Lingyao Li, Tiansheng Wang, Muxuan Liang, Mei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21712)  

**Abstract**: Transformer-based models are widely used for clinical prediction from electronic health records (EHRs), yet their architectures still require substantial manual tuning, and the optimal configuration may vary across tasks and hospitals. Neural architecture search (NAS) automates architecture design, but conventional methods are computationally costly for Transformer-based EHR models. Recent large language model (LLM)-guided NAS methods reduce manual search design but typically conduct each search independently, without reusing architecture knowledge across hospitals. In this study, we propose ATHENA (Agentic Transfer across Hospitals for EHR Neural Architecture Search), a knowledge-guided agentic NAS framework for Transformer-based EHR modeling. ATHENA uses a weight-sharing supernet that is pretrained once per hospital, allowing candidate architectures to be instantiated as inherited subnetworks and evaluated through fine-tuning rather than independent pretraining. It also incorporates a two-layer cross-hospital architecture prior. The first layer retrieves high-performing architecture examples from source sites based on task descriptors, while the second estimates the effects of architectural components using SHapley Additive exPlanations (SHAP)-based meta-regression. These priors guide a multi-agent LLM search together with validation feedback from the target hospital. Across six clinical prediction tasks and two independent health systems, ATHENA matches or outperforms four NAS baselines in 9 of 12 hospital-task evaluations at a search budget of 30. It also shows more consistent architecture selection across repeated searches. ATHENA provides a practical approach for reducing manual architecture tuning in Transformer-based EHR modeling. Code is publicly available at this https URL. 

---
# From Association to Causation: Improving Retrieval Precision of Retrieval-Augmented Generation via Causal Relations and an Attention Mechanism 

**Authors**: Jing Liu, Yongxing Qi, Muchen Jiang, Chengnan Hu, Qingqing Peng, Haoming Wang, Yuqing Wang, Yang Yu, Xu Zhang, Ting Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21702)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds LLM generation on retrieved documents, but the standard terminal retrieval stage--dense-vector similarity, optionally followed by reranking--often returns documents that share keywords with the query without containing the needed information, a failure mode that grows with the knowledge base. We trace it to a conceptual gap: similarity captures only associational relations, whereas the documents that matter are linked to the query causally. We model the terminal retrieval stage with a causal graph grounded in Reichenbach's common cause principle: the keywords shared by the query and a retrieved document form a latent common cause A, and the document's residual keywords form a latent set B linking the document to the ideal output. Since a retrieved document is a collider (A -> d <- B), retrieval itself opens an associational path between the query and B, which licenses a training-free, attention-style re-scoring rule: the cosine similarity between the query embedding and the weighted centroid embedding of B. Unlike causality-enhanced RAG variants that model causal relations inside the knowledge content, our graph models the causal structure of the retrieval process itself. On a real 471-document enterprise knowledge base, the method promotes a relevant guideline from rank 6 to the top 3; on a controlled diagnostic corpus reproducing the keyword-stuffing regime, it improves the mean target rank from 2.88 to 1.25, while a trained cross-encoder reranker barely helps (2.63). Conversely, on three BEIR benchmarks the score underperforms the similarity baseline, delineating the applicability boundary: the method guards the keyword-stuffing regime of growing proprietary knowledge bases and complements neural rerankers; a corpus-level calibration gate selects the correct regime with >= 95% reliability. A fully local testbed demonstrates deployability. 

---
# Context as an Environment: Programmatic Context Management for Long-Horizon Agents 

**Authors**: Yin Lin, Elaine Ang, Erkang Zhu, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.21690)  

**Abstract**: LLM agents increasingly take on long-running tasks whose history grows far beyond a single model context window. Existing approaches compress earlier interactions or extract selected information into fixed memory representations, committing to what to preserve before future needs are known. We present Scroll, a context manager that treats each agent session as an executable Session Environment. The environment is backed by an append-only Event Log and a sandboxed, persistent Python kernel. The kernel maintains a typed namespace across model calls, allowing tool outputs, retrieved history, and derived state to be bound to variables rather than serialized into the prompt at each call. Model-written code searches, materializes, and transforms session state through exec; only explicitly printed projections enter the model's working view for the next call. Context management thus becomes a programming task that inherits the improving coding abilities of LLMs, while the Event Log preserves lossless historical ground truth. As the working view approaches its budget, stale spans are evicted but remain recoverable: an eviction index keeps compact landmarks tied to exact Event Log addresses, so that the agent navigates directly to evicted regions instead of searching the full log. With Qwen3.8-Max as the backbone, Scroll achieves 94.8% on LongMemEval_S; 73.1% on BEAM_10M, surpassing the best published memory system by 5.1 points; and 86.7% on LOCA_256K, exceeding the best published long-horizon agent by 37.4 points. 

---
# From Mastery Profile to Simulated Response: Stochastic Student Knowledge Graphs (SSKG) for Faithful LLM Student Simulation 

**Authors**: Yuan An, Emily Wang, Benjamin Wang, Ruhma Hashmi  

**Link**: [PDF](https://arxiv.org/pdf/2608.21668)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate students at different mastery levels. These simulations can generate synthetic training data and stress-test tutoring systems. However, common prompt-based approaches leave the answer decision to the LLM, which tends to perform according to its built-in capabilities even when instructed to simulate a student with low mastery. As a result, these approaches may have difficulty distinguishing students with low and high levels of mastery. We demonstrate this limitation using 379 College Board-calibrated SAT Algebra items and five archetypal mastery profiles. Three LLMs from three vendors (Gemini 3.1 Flash Lite, Claude Haiku 4.5, and GPT-5.4-mini) achieve 96.8-100% accuracy across all profiles. To address this limitation, we introduce a method grounded in a Stochastic Student Knowledge Graph (SSKG). A curriculum knowledge graph (CKG) is extracted from an open algebra textbook, and each SAT solution is decomposed into a chain of required triples. The SSKG assigns a mastery probability to each triple, which is sampled to determine question correctness. An LLM then generates a first-person rationale consistent with the outcome. The simulation reduces accuracy to 44.1-85.2% across profiles and produces a clear monotone mastery gradient. 

---
# Measuring Activation Control in Large Language Models 

**Authors**: Marek Mateusz Kowalski, Joshua Fonseca Rivera, Uzay Macar, David Demitri Africa  

**Link**: [PDF](https://arxiv.org/pdf/2608.21664)  

**Abstract**: Safe deployment of increasingly capable models will likely come to rely on latent-space monitoring as a complement to behavioral evaluations, especially when evaluation-aware models exhibit scheming or deception. However, if models can also control their own activations, deception could extend into the latent space itself. With this in mind, we introduce the Activation Controllability Benchmark to quantify the extent to which models can modulate their residual stream via natural-language instruction. Across model families and capability levels, we find that most LLMs can control the direction and magnitude of their residual stream activations with some degree of temporal resolution, though performance varies considerably across models. In simple tasks, this level of control can evade activation-based monitoring methods (including linear probes, natural language autoencoders, activation oracles, and the Jacobian lens), albeit imperfectly. These results suggest that control over the activation space itself could become a confound for monitoring as introspective capabilities increase; therefore, we recommend that frontier labs and evaluators track activation controllability in future models. 

---
# SAEM: Stage-Aware Expert Management for Memory-Efficient MoE Inference in Chain-of-Thought Reasoning 

**Authors**: Yujie Zhang, Bin Gao, Tulika Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2608.21614)  

**Abstract**: Chain-of-thought (CoT) prompting improves LLM reasoning by decomposing complex problems into intermediate steps, but its sequential nature increases decoding latency and memory usage. Mixture-of-Experts (MoE) models scale capacity through sparse expert activation, yet their full expert weights often exceed GPU memory and require costly GPU-CPU transfers. Existing runtimes treat all tokens uniformly, overlooking a key structural property of CoT traces: consecutive reasoning stages exhibit coherent and predictable expert activation patterns. Ignoring this stage-level regularity leads to inefficient caching and unnecessary data movement. We propose SAEM, a stage-aware MoE inference runtime that detects reasoning stage boundaries and exploits stage-level activation coherence to guide expert placement. SAEM combines stage-aware caching, expert-aligned token repacking, and in-situ CPU execution to reduce data transfer and kernel fragmentation. On mathematical and scientific reasoning workloads, SAEM achieves an average 1.33x throughput improvement over the strongest state-of-the-art caching and offloading baselines under constrained GPU memory, rising to 1.54x when calibration data matches the workload, demonstrating the effectiveness of stage-aware, locality-driven MoE inference for CoT reasoning. 

---
# Semantic Compression Trees: Multi-Resolution Knowledge Retrieval via Hierarchical Semantic Residuals 

**Authors**: Junaid Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2608.21610)  

**Abstract**: Retrieval-augmented generation relies mostly on flat, fixed-granularity indexes: documents are cut into uniform chunks and retrieved by similarity, discarding the hierarchical structure of the source. We introduce Semantic Compression Trees (SCT), a hierarchical index in which each node stores only its semantic residual -- the information it adds beyond its parent -- and retrieval proceeds by progressive descent from the root, so that per-query cost is governed by tree depth rather than collection size.
We evaluate on QASPER (50 papers, 173 questions) under two protocols differing only in whether the benchmark supplies the relevant document, with bootstrap confidence intervals and paired significance tests throughout. The results are mixed and we report them as such. When the document is given, SCT with a zero-LLM extractive compressor matches dense retrieval on answer quality (0.274 vs. 0.277 F1, $p = 0.37$) using 30% fewer context tokens and no LLM calls to build the index, and residual storage beats storing full summaries at each node (0.274 vs. 0.205, $p < 0.001$). Increasing the collection fifty-fold multiplies flat retrieval's per-query scoring work by 48.9x and SCT's by 6.4x.
Progressive descent itself is not supported. Retrieving the same residuals without the tree performs identically when the document is given ($p = 0.27$), and descent is substantially worse when the system must select the document (0.122 vs. 0.165, $p < 0.001$). Routing accuracy localises the cause: descent selects the correct paper 20.2% of the time against 39.3% for flat retrieval, because that choice is made from the root residual, the most compressed node in the tree. We conclude that the residual representation is worth keeping and top-down routing is not. 

---
# Generate in the Chart, Not on the Boundary: Function-Symbol Grounding for Hard Constraints in LTN-GANs 

**Authors**: Nijesh Upreti, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2608.21605)  

**Abstract**: Logic Tensor Network-Enhanced Generative Adversarial Networks (LTN-GANs) inject background knowledge by grounding each logical axiom as a predicate and training the generator to raise its satisfaction, a fuzzy truth value in $[0,1]$. Previous LTN-GAN work grounded every constraint this way, at the predicate level, and improved constraint satisfaction. A predicate, however, only scores a sample, so it cannot embed hard structural constraints, rules such as orderings, positivity, and definitional identities that must hold in every generated sample. In this work, we investigate grounding each axiom as a function symbol inside the LTN framework. We compare against the state-of-the-art alternative, a constraint layer that clamps each violating sample onto the feasible boundary and so produces outputs that are always valid. Our investigation shows that a valid sample is not always a realistic one. An inequality is not merely satisfied or violated. It holds by a margin, and a faithful generator should also reproduce the margin's real distribution. We find that the resolution ratio $R$, the data's scale over the margin's spread, is a diagnostic, computable before training, of which constraints a chosen grounding can learn. When $R$ is large, the predicate receives no learning signal, the clamp pushes every sample onto the boundary, and the margin distribution is lost while every standard metric still looks fine. A function symbol avoids both failures, computing the constrained variable rather than scoring it. Together the function symbols form a chart, a coordinate system inside the feasible region, where every sample is valid by construction and the margin is learned like any other quantity. 

---
# K-Bench: measuring model performance on real scientific agent requests 

**Authors**: Aubrey Brueckner, Darshil Patel, Yuhuan He, Timothy Kassis  

**Link**: [PDF](https://arxiv.org/pdf/2608.21601)  

**Abstract**: Benchmarks for scientific artificial intelligence are mostly written to be scored: multiple-choice questions, curated agent tasks with reference solutions, or simulators with a known generative structure. Real scientific requests arrive differently. They are underspecified, they carry attachments, and lack ground truth. We report K-Bench 01, an evaluation built from first-turn requests sampled from live user traffic on K-Dense Web and run end to end by nine frontier models in identical sandboxes, yielding 1,602 completed agent runs. Three blinded language-model judges scored every run against an eight-dimension rubric. On a rubric whose 8-anchor instructs judges that a domain scientist would accept the work with minor edits, no model clears the line under all three judges. gpt-5.6-sol has the highest pooled mean, 8.04, but its 95% interval [7.80, 8.23] spans the threshold, and two of the three judges rank claude-opus-5 first instead. We therefore report the ordering of systems as the reproducible quantity, the absolute level as an attribute of the instrument, and the top of the table as unresolved. Across all 39,934 scored judgments -- the eight dimension scores plus a holistic overall for each assessment, excluding not-applicable cells -- 47.6% fall below the 8-point threshold. Difficulty is not uniform across the rubric: scientific accuracy averages 6.22 against 7.33 for communication, on identical denominators and in the same direction within every one of the nine models. The single leading failure tag is overclaiming, on 31.4% of assessments. We argue that the informative quantity for scientific agents is not a leaderboard position but the joint distribution of what was delivered, what was claimed, and what artifacts were produced. 

---
# Data-Driven Dynamic Algorithm Dispatch with Large Language Models 

**Authors**: Rushil Shah, Emmanuel Lujan, Rabab Alomairy, Alan Edelman  

**Link**: [PDF](https://arxiv.org/pdf/2608.21584)  

**Abstract**: We introduce a large language model (LLM)-driven approach for generating dynamic algorithmic dispatch heuristics in high-performance linear algebra. By combining prompt engineering with LLaMA 3 and a curated performance database, the model learns to synthesize selection heuristics that exploit structural patterns to identify fast algorithmic choices. A case study on LU factorization demonstrates the model's ability to replicate expert-designed strategies. This work, developed as part of the DARPA-MIT SmartSolve project, highlights the promise of LLMs for algorithmic discovery and the development of more adaptive, fast linear algebra software. 

---
# Robust Lightweight Deep Learning Models for Oral Cancer Screening 

**Authors**: Siddhant Bharadwaj, Aakash Shedsale, Tejashree Subramanya, Mohd. Azfar, Praveen Birur, Debnath Pal, Shankararama Sharma, Anupama Shetty, Rajesh Sundaresan  

**Link**: [PDF](https://arxiv.org/pdf/2608.21583)  

**Abstract**: Oral cancer is a leading cause of mortality in low-to-middle-income countries, where a shortage of specialists delays diagnosis. While point-of-care screening via smartphones offers a scalable solution, developing robust AI for resource-constrained settings poses significant challenges, including class imbalance in training data, variable data quality, and computational constraints on edge devices. In this paper, we present the optimisation of lightweight deep learning models for smartphone-based oral cancer screening. Using a diverse, multi-centre retrospective dataset of approximately 30,000 images acquired over a decade, we systematically evaluate state-of-the-art convolutional, transformer, and hybrid architectures. Through rigorous pipeline ablation, we demonstrate that directly optimising hybrid architectures for the edge strictly outperforms computationally heavy paradigms, such as large models or knowledge distillation. Furthermore, interpretability analysis and simulated noise-stress tests revealed that the system anchors on clinical features and remains robust to unstructured sensor noise, despite vulnerabilities to impulse bit errors. In the held-out test set, our optimised MobileViTv2 models achieved an average sensitivity of 83.2 $\pm$ 1.5% and an average specificity of 86.0 $\pm$ 0.8%, with the best model exhibiting 87.4% sensitivity, 86.5% specificity, and a critical negative predictive value of 97.2% with reference to specialist labels. These results confirm that with targeted architectural selection and streamlined optimisation, interpretable and robust lightweight AI models exhibit high potential for edge deployment to enable automated triage in primary care settings. 

---
# A Reproducible, License-Aware Distillation Recipe for CPUDeployable Safety Classification 

**Authors**: Edson Rodrigues da Cruz Filho, Paulo Ricardo Ferreira Neves, Paulo Henrique Eleuterio Falsetti, João Vitor Pavan, Ian Degaspari, Henrique Vieira Laturrague, Patrick Vieira Laturrague, Guilherme Nielsen Dias, Marccello Wilson Perez Berto, Gustavo Voltani Von Atzingen  

**Link**: [PDF](https://arxiv.org/pdf/2608.21570)  

**Abstract**: Deploying a safety layer for large language models on commodity hardware is constrained by the guards available to do it: current open guard models hold between 1 and 9 billion parameters, are oriented toward the graphics processing unit, and answer in seconds per request on a central processing unit. This paper presents a reproducible, license-aware knowledge-distillation recipe addressing that constraint. A strong open guard labels a corpus of roughly 97,000 prompts, drawn from 24 public datasets, into seven safety categories aligned to a public hazard taxonomy, and a fleet of small students spanning lexical, shallow, encoder and generative architectures is trained to reproduce that signal. The corpus is partitioned at the license boundary, so that a deployable and a research model differ only in their training data and the cost of that restriction becomes measurable. Every model is scored against an independent gold benchmark of 6,361 rows over four slices, labeled apart from the teacher and including a slice of harmless prompts that makes over-defense measurable. The distilled students match the teachers on adversarial text within overlapping confidence intervals and reduce false alarms on harmless prompts, the smallest generative student reaching 3.8% against 4.8% for the 8-billion-parameter teacher, while the encoder classifies in roughly 24 ms per request on CPU. Per-class rebalancing is the only decisive ingredient of the recipe. No superiority over the distilled guards is claimed; on the clean reference slice they remain ahead. 

---
# Quantifying geographic domain shift to decouple the geospatial transferability of human mobility flow generation models 

**Authors**: Zhiyong Zhou, Song Gao, Qianheng Zhang, Feng Zhang, Zhenhong Du  

**Link**: [PDF](https://arxiv.org/pdf/2608.21567)  

**Abstract**: Human mobility serves as an essential proxy for understanding social, economic, and environmental dynamics in urban systems. Geospatial transferability, which measures a model's capability in a new location or unseen region, is a critical dimension for comparing different human mobility generation models. However, few studies have studied the intrinsic characteristics of geospatial transferability. To this end, this study systematically investigates the geospatial transferability of four representative human mobility generation models using a large-scale benchmark dataset of census tract level commuting flows across 2265 counties in the United States. Inspired by the domain adaptation theory in machine learning, we introduce geographic domain shift to describe the intrinsic differences in geographic feature distributions and spatial structures between source and target regions, which may jointly affect model transferability. Moreover, we propose two metrics, mutual information and spatial shift, to quantify the geographic domain shift. To examine their associations with model transferability, we employ linear mixed-effects regression to analyze the associations between geographic domain shifts and transferability. Our results reveal substantial spatial heterogeneity and asymmetry in transfer performance across regions. Both information shift and spatial shift exhibit statistically significant and complementary explanatory power. This indicates that geospatial transferability depends not only on model design but also on intrinsic geographic differences. These findings provide a novel methodological framework for evaluating and improving the geospatial transferability of human mobility generation models and support more robust and fair human mobility data synthesis across diverse regions. It also offers insights on spatial transferability for GeoAI model development. 

---
# Let Credit Follow Computation: Architecture-Aware Credit Transport for Large Language Model Reinforcement Learning 

**Authors**: Qifan Shi, Zhaolu Kang, Chenghua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21501)  

**Abstract**: Credit assignment in large-language-model reinforcement learning (LLM RL) can be separated into three objects: evidence about success, a transport operator that converts this evidence into token-level advantages, and an update geometry that turns advantages into policy changes. Recent work has greatly improved evidence, sampling, and update geometry, but the transport operator is usually architecture-agnostic. Fixed-discount GAE applies a stationary geometric kernel along token time; group-relative methods broadcast an outcome statistic across an entire response. Neither operator represents the trajectory-specific computation used by the Transformer policy itself. We introduce computation-conditioned credit transport (CCT), a general framework in which a detached statistic of the behavior policy's internal computation parameterizes the causal kernel that transports downstream value through a rollout. Our concrete algorithm, CompPO, maps native attention concentration to a bounded per-token retention gate, uses the gate in both the one-step bootstrap and a path-dependent generalized-advantage trace (Comp-GAE), and co-designs a transport-aligned critic (TAC) that reuses the actor's hidden states and routing information without a second same-scale Transformer. The task reward and clipped PPO policy objective remain unchanged; a constant gate recovers fixed-coefficient GAE. Across five Qwen3-4B seeds, CompPO reaches 61.4% final held-out accuracy (95% CI [60.8,62.0]) versus 53.8% [52.9,54.7] for tuned GRPO. Neither Comp-GAE with a standard critic (55.2%) nor TAC with a fixed gate (56.4%) matches the full model (interaction +2.4 [1.9,2.9]). Shuffle and position controls confirm trajectory-specific alignment; CompPO is stable in 10/12 PPO-grid runs versus 3/12. Frozen evaluation improves over GRPO by 4.3 and 3.9 greedy pass@1 macro points on Qwen3-4B and Llama-3.1-8B-Instruct. 

---
# Enhanced Artificial Neural Networks Using QHAdamW in Air Quality Forecasting 

**Authors**: Mary Joy Daniel Vinas  

**Link**: [PDF](https://arxiv.org/pdf/2608.21463)  

**Abstract**: The study employed an Artificial Neural Network in combination with the optimized Adaptive Moment Estimation (Adam) algorithm, currently the only AQI forecasting model available in the Philippines. The modified QHAdamW - Quasi-Hyperbolic Momentum (QHAdam) and Adam with decoupled weight decay (AdamW) were both extensions of the Adam optimizer, and both offer unique advantages for training ANN. The proposed QHAdamW optimizer addresses the issues on convergence, generalization, and forecasting performance of Adam. Hyperparameter tuning results revealed that 0.01 and 0.001 were the most effective optimal values for the generalization performance of QHAdamW. The comparative analysis results using seven evaluation metrics revealed that the error value range is lower, and the regression coefficient, having a value approximately equal to 1, improved the model accuracy performance. Likewise, the model converges to a satisfactory level of performance with the convergence performance results of lower loss values as obtained from training and validation losses. Based on data from a real-time air quality tracking station in Manila, a feed-forward neural network is used to predict the AQI of PM2.5 and PM10 separately. This model can be used to forecast Particulate Matter (PM), to help the Department of Environment and Natural Resources-Environmental Monitoring Bureau (DENR-EMB) implement a comprehensive air quality management. 

---
# Software Frameworks for Explainable AI in Time Series Classification: A Systematic Review 

**Authors**: Louis Peter, Nils Gumpfer, Jana Fischer, Christin Seifert, Jennifer Hannig  

**Link**: [PDF](https://arxiv.org/pdf/2608.21449)  

**Abstract**: Time series arise in a wide range of application domains and are analyzed using machine learning in decision-critical settings. Time series classification (TSC) is one of the most widely studied and relevant tasks. In this context, ensuring the transparency and trustworthiness of TSC models has become an important requirement, motivating the use of explainable artificial intelligence (XAI) methods. Despite growing interest, research on XAI for TSC remains fragmented, and a systematic understanding of the available software frameworks for explanation generation, their evaluation practices, and practical limitations is still lacking. Prior work largely focused on individual explanation methods, while cross-framework consistency, time-series-specific evaluation, and reproducibility have received little attention. In this survey, we analyze existing software frameworks for explanation generation and evaluation in TSC. We compare them along multiple dimensions, including supported XAI methods, evaluation metrics, usability, benchmarking support, and reproducibility, providing the first time-series-specific survey of frameworks with implementation comparisons and an analysis of frequency-domain support. We identify six frameworks that explicitly support time series and reveal common limitations: only one method supports frequency-domain explanations despite their relevance; only two evaluation metrics have been developed specifically for time series; and identical XAI methods can yield substantially different explanations across frameworks. Based on these findings, we discuss open challenges and outline directions for future research, highlighting the need for unified, time-series-specific XAI frameworks that enable faithful, reproducible, and time-series-aware explanations. 

---
# Agentic AI for Safety-critical Multi-drone Systems: Challenges and Opportunities 

**Authors**: Timothy Merritt, Alejandro Jarabo-Peñas, Juan Bravo-Arrabal, Maria-Theresa Bahodi, Anders Lyhne Christensen  

**Link**: [PDF](https://arxiv.org/pdf/2608.21444)  

**Abstract**: Multi-drone systems are increasingly positioned for safety-critical missions such as search and rescue (SAR) and critical infrastructure monitoring. Yet, real-world adoption remains constrained not only by autonomy performance, but by the difficulty of integrating agentic behavior into professional work: operators must understand, trust, and govern automation under uncertainty, time pressure, and accountability. This position paper synthesizes the ambitions and lessons from two ongoing efforts: NAMUR, which explores LLM-supported robot control in SAR and firefighting contexts, and PERSIST, which explores persistent drone operations for monitoring and security at critical infrastructure sites. We argue that agentic AI should be approached as a socio-technical design problem, where interfaces, oversight mechanisms, and evaluation practices are as critical as algorithms. We outline a human-centered, participatory, and iterative research approach aimed at uncovering stakeholder needs, shaping agent capabilities through successive prototypes, and producing transferable proof-of-concept systems and evaluation strategies for other safety-critical contexts. 

---
# Evaluating Multimodal Narrative Understanding of Popular Hollywood Films 

**Authors**: David Bamman, Kent K. Chang, Allison Cooper, Juishan Hsu, Reina Kushihashi, Madison Mar, Arnav Podichetty, Rachael Samberg, Ipek Nil Sancak, Yuhan Shao  

**Link**: [PDF](https://arxiv.org/pdf/2608.21430)  

**Abstract**: Multimodal language models increasingly show promise for enabling the large-scale computational analysis of film, opening up new avenues for learning about film history and the evolution of narrative techniques. But the creation of stable benchmarks built around Hollywood films is complicated by copyright protections. In this work, we address these concerns directly, by building a new collection of Hollywood films defined by two criteria: box office popularity (where we publish the first large-scale, open collection of weekly box office earnings reported by Variety magazine from 1922-1979); and likely public domain status (by researching copyright registrations and renewals in the US Catalog of Copyright Entries). We build a new multimodal MCQ benchmark on top of this collection that focuses on narrative elements that directly evaluate the abilities of models to inform meaningful research on film narrative; we find that many vision-language models struggle on this task (with many performing at near-chance levels of accuracy), while audio-visual models (including those that use audio in captioning scenes) reach a maximum accuracy of 61.1%, well below human-level performance. 

---
# Composable Trust Infrastructure for Manufacturing Knowledge Graphs: Cross-System Provenance, Temporal Reasoning, and Decision Traceability 

**Authors**: Grama Chethan  

**Link**: [PDF](https://arxiv.org/pdf/2608.21418)  

**Abstract**: Manufacturing knowledge graphs that integrate data from heterogeneous industrial systems face a trust deficit: consumers cannot determine whether queried data is valid, whether it was valid when a decision was made, where it originated, or how it was acted upon. We argue that four trust capabilities -- SHACL validation, PROV-O provenance, domain-aware bi-temporal versioning, and graph-native decision objects -- compose through shared correlation identifiers to produce emergent trust properties that no single capability delivers alone. We present a composable trust infrastructure that integrates these four capabilities into a unified RDF architecture. Capabilities compose through shared entity URIs, ingestion activity identifiers, and temporal correlation keys, enabling compound queries spanning all four dimensions. An experimental ablation confirms that removing any single capability causes exactly three of six composition queries to fail, demonstrating that all four are equally load-bearing. Analysis of higher-order compositions reveals four emergent three-way properties and one irreducible four-way property (full-chain auditability, 31ms execution). The infrastructure is validated on a testbed integrating eleven industrial sources -- OPC UA, TIA Portal, eClass, AAS, ISA-95, ISA-18.2, SAP S/4HANA, Teamcenter, Opcenter EX, Insights Hub, and SCM -- under an 89-class ISA-95-aligned ontology. The unified graph contains 8,743 triples across five named graphs, stitched by 81 owl:sameAs identity edges. Evaluation uses simulated but structurally realistic data from purpose-built emulators; data structures and cross-system linkage patterns are representative of real industrial installations. 

---
# Retrieval-grounded robot program generation and simulation-based correction via Model Context Protocol 

**Authors**: Zhichao Zhou, Siyuan Chen, Omkar Salunkhe, Ebru Turanoglu Bekar, Johan Stahre, Anders Skoogh  

**Link**: [PDF](https://arxiv.org/pdf/2608.21417)  

**Abstract**: Flexible manufacturing requires industrial robots to be reprogrammed rapidly as product variants change. This paper presents a language-model-based workflow that generates, validates, and iteratively corrects ABB RAPID robot programs from natural language task descriptions. A dual-stream retrieval-augmented generation (RAG) pipeline grounds code generation in verified technical documentation and production templates, reducing domain-specific errors produced by ungrounded language models. A custom Model Context Protocol (MCP) server connects the language-model client directly to ABB RobotStudio for automated code upload, simulation execution, and diagnostic feedback. The evaluation combines a 30-query retrieval benchmark, scoped code-generation checks, and RobotStudio case studies in a simulated pickand- place manufacturing cell. The simulation loop exposes execution failures that static and semantic checks alone cannot catch, including suction release-height errors, unreachable placement targets, and configuration-dependent recovery motions. The results show how RAG and MCP can connect grounded code generation with executable feedback from industrial robot simulation software, while reducing but not eliminating expert setup and final supervision. 

---
# The Abstention Protocol: RCA for Clos Fabrics 

**Authors**: Madhava Gaikwad, Deepak Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2608.21412)  

**Abstract**: Root cause analysis (RCA) in large datacenter networks is challenging because telemetry is noisy, partial, and asynchronous. Score-based approaches degrade under these conditions, often yielding unstable or incorrect attributions.
We present \textsc{CoreSec}, a production RCA system that replaces weighted fusion with a PAM-style abstention algebra. Telemetry agents are composed using control flags that yield deterministic decisions and explicit abstention when evidence is ambiguous. CoreSec combines this algebra with topology-aware configurations that capture failure surfaces across Clos fabrics and converge monotonically as evidence accumulates.
Deployed at hyperscale, CoreSec provides stable and explainable RCA behavior across diverse environments without retuning. Our experience shows that structured composition with abstention forms a practical foundation for automated RCA in real-world cloud networks. 

---
# Hate Speech Classification In Roman Urdu: A Comparative Study On Parameter Efficient Fine-Tuning And Prompt Engineering 

**Authors**: Toneema Zubair  

**Link**: [PDF](https://arxiv.org/pdf/2608.21408)  

**Abstract**: Due to the widespread accessibility of the internet and social media, toxic and hateful con-tent has grown exponentially, causing significant distress and negative societal impacts. Ro-man Urdu, a low-resource language used in Pakistan and among Urdu-speaking communities worldwide, presents additional challenges because of its informal grammar, inconsistent sen-tence structures, and multiple variations in word spellings. This research aims to identify the most effective techniques for hate speech classification in such low-resource settings with limited data. To address this, the study investigates and compares the latest approaches, in-cluding prompt tuning, parameter-efficient fine-tuning (PEFT) using LoRA, and prompt en-gineering, under various experimental configurations. To achieve this objective, four exper-iments were designed. The first experiment involved direct inferencing with LLMs without any fine-tuning, to evaluate how well these models understand Roman Urdu in a zero-shot setting, especially given limited data. The second experiment utilized parameter-efficient fine-tuning (PEFT) with LoRA, which updates only a small subset of parameters, thereby reducing computational cost. The third experiment explored prompt tuning with both mixed and manually crafted prompts, using very small sets of training examples relative to the entire dataset, making it computationally efficient as well. Finally, the fourth experiment applied prompt engineering through zero-shot and few-shot learning, relying solely on care-fully designed instruction prompts for classification without further training. 

---
# Spyre-Accelerated Retrieval-Augmented Generation on IBM LinuxONE: A Cloud-Native Architecture for Secure, High-Throughput Enterprise AI Inference 

**Authors**: Sandeep Bokkasam, Pankaj D  

**Link**: [PDF](https://arxiv.org/pdf/2608.21393)  

**Abstract**: Running large language models inside enterprise environments has always bumped up against a practical wall: the data lives in one place, the AI horsepower sits somewhere else, and moving sensitive records between the two creates real headaches around latency, security, and regulatory exposure. IBM's Spyre accelerator PCIe inference card built for LinuxONE and the broader IBM Z family changes that equation.
In this paper we lay out a six-subsystem RAG architecture that runs entirely on IBM LinuxONE, using Spyre for generative inference, the Telum II on-chip accelerator for lightweight classification tasks, and Red Hat OpenShift for container orchestration. Every piece of the pipeline from query intake through vector retrieval, prompt assembly, LLM inference, compliance filtering, and response delivery stays within a single LinuxONE system, so sensitive data never has to leave the hardware perimeter.
We walk through the design choices behind each subsystem, dig into the Spyre compilation and serving stack, explain how LinuxONE's Secure Execution technology extends confidential-computing guarantees to AI workloads, and benchmark the architecture against cloud-GPU and on-premises alternatives. Early analysis points to end-to-end RAG latencies under two seconds and up to a 20x reduction compared to off-platform inference, all while keeping the strong encryption and auditability posture that regulated industries actually need. 

---
# There Is No Neutral Harness: Modern LLM Leaderboards Are Manufactured by Config-Fragile Items 

**Authors**: V.S. Raghu Parupudi  

**Link**: [PDF](https://arxiv.org/pdf/2608.21382)  

**Abstract**: Multiple-choice benchmarks fix the questions and the correct answers, but not the harness: the order of the options, the wording of the prompt, and whether a language model's answer is read from generated text or from per-option likelihoods. Work on this harness sensitivity reports it as aggregate score variance, leaving unexamined which items the variance falls on and whether they are the items that separate one model from the next. We treat the evaluation harness of large language models (LLMs) as an independent variable and resolve its effect to single items. We introduce the \textit{fragility grid}: 12 open-weight instruction-tuned LLMs from 4 families answer the same 3{,}679 items from 4 benchmarks (ARC, HellaSwag, MMLU, TruthfulQA) under 26 equally defensible harness configurations, recording one correctness bit for every model, item, and configuration. The comparison is matched, since the items, the weights, and the greedy decoding stay fixed while only the harness varies. Under the grid a model's score is a band rather than a point: gemma4-31b scores between 31 and 89 percent depending only on the harness. Three results follow. On the items that two adjacent models both answer stably the pair is tied, and config-fragile items carry 95.7 percent of a pair's gap on average. Four of the 12 models reach rank one under some configuration, so the harness selects the winner. Item discrimination, the property that benchmark-compression methods maximize, correlates with fragility at 0.28 (95 percent CI 0.25 to 0.30), so compression keeps the fragile items rather than removing them. The scoring choice, not the option order that protocols usually fix, is the load-bearing axis. We release the per-item records and the analysis script, from which every number regenerates on a CPU in seconds, and we position the fragility grid as a check a leaderboard can run before it reports an order. 

---
# RIACT: A Responsible AI System for Personalized Study Habit Tracking and Early Burnout Signal Detection in University Students 

**Authors**: Ria Sidhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21379)  

**Abstract**: Student burnout is highly prevalent in higher education, with reported rates ranging from 12% to over 70% and consistently exceeding those of the working population - yet it is typically identified only retrospectively, after academic decline has already occurred. A contributing factor is that students have little structured visibility into their own study behaviour, and existing productivity tools record activity without interpreting it. This paper presents RIACT (Record, Insight, Analyze, Coach, Track), a web-based application that combines structured study session logging with a hybrid AI architecture to surface personalized insights and early burnout signals. Students log sessions by location and time; the system computes net focus time by accounting for breaks, detects burnout signals through transparent, deterministic rules operating on week-over-week behavioural comparisons, and uses a large language model - constrained to a fixed output schema - to contextualize patterns and generate personalized recommendations. The design embeds responsible AI principles throughout: warnings are governed by auditable rules rather than model judgement, all output is framed as an observation rather than a diagnosis and data collection is limited to self-logged behavioural fields. We describe the system's design rationale, situate it within the literature on student burnout and explainable AI in education and propose an evaluation framework for validating its behavioural signals against established burnout instruments. 

---
# SchemaRouter: Field-Aware Tool Routing for Efficient Heterogeneous Agentic RAG 

**Authors**: Yong-eun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2608.21375)  

**Abstract**: Heterogeneous agentic retrieval-augmented generation (RAG) systems increasingly orchestrate external APIs, internal databases, vector stores, and graph stores. Exposing all tool descriptions to an LLM agent, or selecting tools only by vector similarity, causes two costly failures: over-fetching, which increases payload size, token use, and latency, and under-fetching, which omits fields needed to answer the query.
We present SchemaRouter, a lightweight routing layer that represents tools, endpoints, parameters, response fields, domain concepts, units, provenance, and license policies as a schema graph. Given a query, SchemaRouter emits an executable tool plan specifying which tools to call and which fields to retrieve. A small LLM extracts intent, concepts, and source constraints, while field selection is deterministic over the graph through intent-group projection and concept-field matching with an alias layer.
On a materials-science benchmark of 110 queries, SchemaRouter achieves answer accuracy of 0.71, matching fetch-everything within overlapping confidence intervals and exceeding prompt-all's 0.66, though their intervals overlap. It uses 227 retrieved-context tokens versus 2,066 for fetch-everything and achieves 2.7x lower end-to-end latency than prompt-all. It also obtains the best tool-exact rate of 0.93 and parameter validity of 1.0. SchemaRouter grounds provenance and license information in 62 percent of answers, compared with approximately 0 percent for all baselines.
We also find that minimizing selected-field count is counterproductive: it reduces answer accuracy to 0.56 with negligible token savings, while recall-preserving projection restores top accuracy. SchemaRouter improves efficiency, schema-size-independent scaling, and verifiable provenance/license-grounded answering at competitive accuracy. 

---
# LitReview Arena: Evaluating Literature Review Agents with Battle-Style Peer Review Platform 

**Authors**: Ruotong Zhao, Zhiyu Chen, Xurui Liu, Haidong Xue, Dong Liang, Jigao Fu, Wu YanBiao, Yuanyi Zhen, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.21374)  

**Abstract**: Literature reviews are essential to scientific progress, but rigorously evaluating automatically generated reviews remains difficult because many aspects of research utility depend on expert judgment rather than reference-overlap metrics. We introduce LitReview Arena, a battle-style evaluation platform with a structured protocol tailored to literature review quality: domain experts with AI paper-writing experience compare anonymized drafts, are matched to topics within their expertise, and provide dimension-wise outcomes over five literature-review-specific criteria. From this protocol, we collect approximately 3k expert judgments, each containing five dimension-wise outcomes, and show that even the strongest current systems win only 23.0% of decisive matches against human drafts on overall utility, while agentic LLMs such as Sonar Deep Research substantially outperform base language models by over 60%. We further find that existing LLM-as-a-judge methods are substantially misaligned with human experts (Spearman's rho=0.467), especially on synthesis-heavy criteria such as paper structure and research suggestions. Using the collected preference data, we provide an expert-calibrated evaluator, LitJudge, which improves alignment to Spearman's rho=0.78, comparable to inter-expert consistency; code and data are publicly available at this https URL. 

---
# AI Learning and Conceptual Transfer in the Game of Hidden Rules 

**Authors**: Christo Mathew, Wentian Wang, Jacob Feldman, Lazaros K. Gallos, Paul B. Kantor, Vladimir Menkov, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.21372)  

**Abstract**: This report summarizes the work conducted on the Game of Hidden Rules (GOHR), focusing on reinforcement learning agents trained to infer hidden rules from trial-and-error feedback, representation design, rule difficulty analysis, transfer learning, generalization, and pseudo-bot-assisted human learning analysis. The report focuses on the Transformer-based A2C framework, Feature-Centric and Object-Centric representations, experimental findings, and classification of human learning data. 

---
# Reviewing Model Collapse and Countermeasures 

**Authors**: Xihao Xie, Beichen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21366)  

**Abstract**: Driven by massive amounts of web-scale data, generative AI (GenAI) has achieved remarkable progress, enabling various applications in diverse sectors. The advances of GenAI have actuated practitioners to use AI-synthesized data for training next-generation AI models. Undeniably, using synthetic data has alleviated the increasing stringent demand for data supply. Unfortunately, it also introduces a new critical issue: in a self-consuming cycle between model and data, the model ultimately collapse, raising more trustworthiness concerns to GenAI. In recent years, increasingly more studies have investigated the phenomenon of model collapse (MC) and explored potential solutions to mitigate it. However, the review of the phenomenon of MC still remains blank. To fill this gap, this paper provides an up-to-date overview of these studies for consolidating and reviewing the progress of MC in different application scenarios and countermeasures for mitigating MC. We also highlight challenges and future research opportunities. 

---
# AIREP: A Protocol for Per-Decision Evidence in AI Runtime Governance 

**Authors**: Ali Toygar Abak  

**Link**: [PDF](https://arxiv.org/pdf/2608.21363)  

**Abstract**: A protocol is presented for recording the governance decisions of automated AI runtimes. When a runtime releases, blocks, defers, redacts, or escalates an individual output, AIREP records that decision as a single signed object that any party can check offline, independent of the runtime that produced it. A record carries the decision as one of a closed set of verbs under a stated policy basis, references its input, output, and evidence by hash rather than by value, and declares both what its evidence covers and what it does not. Records form a SHA-256 hash chain that binds each record to its position, so that tampering and gaps are detectable by recomputation. Vendor-, model-, and domain-specific content is confined to a single optional namespace, and a mechanical neutrality test keeps the shared format free of it. A reference implementation and a two-language conformance kit are described. Some implementation issues are considered, and problems such as alignment of the canonical form across implementations, freshness witnesses, and multi-runtime chains are exposed. The format is offered for adoption by any AI runtime that records governance decisions. 

---
# KVBoost: Chunk-Level Key-Value Cache Reuse with Deviation-Guided Recomputation for Efficient Large Language Model Inference 

**Authors**: Srihari Unnikrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2608.21362)  

**Abstract**: Transformer-based large language models (LLMs) incur high prefill latency because key-value (KV) tensors must be recomputed for each request. Existing prefix-caching systems reduce this cost but require prompts to share a leading contiguous prefix, limiting effectiveness when shared content appears at arbitrary positions. We present KVBoost, a chunk-level KV cache reuse system for HuggingFace-compatible decoder models that enables reuse regardless of content position. KVBoost introduces a dual-hash keying scheme that separates positional identity (prefix hash) from content identity (content hash), supporting both exact and approximate cache matches. To address attention boundary errors from independently cached chunks, KVBoost employs two repair strategies: SelectiveRecompute, which re-encodes boundary regions, and CacheBlendRecompute, which identifies and recomputes high-deviation tokens after a probe pass. The system further incorporates asymmetric KV quantization (int8/int4), adaptive chunk boundary splitting, and importance-weighted eviction under a fixed memory budget. Evaluated on Qwen/Qwen2.5-3B over 1,000 bug-localization samples, KVBoost achieves a 4.49x reduction in time-to-first-token (142.4 ms vs.\ 639.1 ms) and outperforms prefix caching by 16%, with no loss in accuracy (99.2% vs.\ 99.1%). KVBoost provides a practical, memory-bounded inference acceleration layer compatible with RoPE-based models without architectural modification. 

---
# How to Train a Critic Stably and Efficiently 

**Authors**: Penghui Qi, Xiangxin Zhou, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.23566)  

**Abstract**: Group-based reinforcement learning methods such as GRPO for large language models avoid training a critic by sampling multiple responses for each prompt. A reliable critic could instead estimate token-level advantages from one response, but standard critic-based training recipes are often unstable. We study this instability and develop \textbf{Best-Practice Critic Optimization (BPCO)}, a recipe that combines DPPO, value predictions bounded to the reward range, Monte Carlo value targets, unnormalized policy advantages, and length-adaptive generalized advantage estimation. Because the critic is used only during training, BPCO can also condition it on reward-defining information, such as a reference answer or grading rubric, that is hidden from the policy. Controlled experiments isolate the effect of each design choice. Across mathematical reasoning tasks with models ranging from 1.5B parameters to 30B-A3B mixtures of experts, BPCO improves a strong critic-based baseline consistently, and matches or exceeds a group-based baseline while sampling one response per prompt. The same recipe also improves learning with rubric-based rewards. These results show that a carefully designed critic provides a reliable alternative to group-relative advantage estimation. Code is available at this https URL 

---
# SWE Refactor Bench: Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration? 

**Authors**: Deyao Hong, Yizhe Chi, Wenyi Li, Xiaoqiu Wang, Mingju Gao, Kaisen Yang, Bingxiang He, Youjie Zheng, Calvin Xiao, Qinhuai Na  

**Link**: [PDF](https://arxiv.org/pdf/2608.23564)  

**Abstract**: Modern software systems accumulate technical debt over decades of development, which makes migration expensive and largely manual. As coding agents become increasingly capable at bug fixing, can they autonomously perform such migrations? Existing benchmarks cannot answer this question because they evaluate only behavioural correctness, not whether the migration actually occurred. This leads an easy hack: agents copy the original implementation to make tests pass. We call this Blindness. To address this problem, we introduce SWE Refactor Bench, a benchmark comprising 20 whole-repository migrations, covering 4 kinds of technical debt. A three-stage evaluation protocol measures both migration completeness and behavioural correctness. (1) Migration Audit verifies that the migration occurred. (2) Behavioural Tests measure correctness with a fixed test suite. (3) Agentic Verification uses 6 independent coding agents to generate targeted tests for hidden behavioural differences. Across 520 runs from 8 frontier models and 26 model-effort configurations, only 28 of 520 runs ($5.4\%$) pass all three stages, 13 of the 20 tasks receive no accepted solution, and the best model (claude-opus-5) scores $47.0/100$. Migration completeness and behavioural correctness are distinct abilities: a few runs preserve behaviour by skipping the migration and are stopped at Migration Audit; most attempt it and break behaviour, and are stopped at Behavioural Tests. Agents cannot deliver a perfect migration: among the 340 runs that pass Migration Audit, $58\%$ reach $99\%$ of the fixed checks, yet only $26\%$ reach $100\%$. Agent capability differs across migration categories: agents score $31.4$ on build toolchain rewrites but only $5.6$ on language rewrites. Together, these findings position SWE Refactor Bench as a rigorous testbed for developing coding agents for reliable whole-repository migrations. 

---
# EG-ARSA: An Expert-Grounded Open Model for Visual Road Safety Auditing in Low-Resource Settings 

**Authors**: Md Thamed Bin Zaman Chowdhury, Moazzem Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2608.23563)  

**Abstract**: Road traffic injuries remain a major challenge in low- and middle-income countries, where proactive road safety auditing is limited by incomplete crash records, shortages of qualified auditors, and the high cost of large-scale field inspections. To address this problem, we propose Expert-Grounded Distillation (EGD), a novel artificial intelligence framework that transfers institutional road safety expertise into a compact vision-language model for scalable visual road safety auditing. The key innovation is a quantified expert-grounding stage in which the teacher vision-language model is calibrated against authoritative field audits. Large-scale annotation is permitted only after the teacher reaches substantial agreement with expert risk assessments (Cohen's kappa = 0.74). The calibrated teacher then generates structured supervision that is distilled into an 8-billion-parameter student vision-language model using Low-Rank Adaptation and a single leakage-free prompt. We also introduce Bangladesh Road Safety Audit (BD-ARSA), the first open, expert-grounded Bangladeshi visual road safety audit dataset containing 21,947 image-audit records with near-national coverage, and Expert-Grounded Road Safety Auditor (EG-ARSA), the first vision-language model developed specifically for this task. Experimental results show that grounded fine-tuning substantially improves ordinal risk assessment over the zero-shot baseline, while blind expert evaluation demonstrates that the compact student outperforms both its 31 billion-parameter teacher and Gemini-2.5-Flash. These findings demonstrate that EGD provides an effective and scalable engineering solution for proactive road safety auditing in resource-constrained environments. 

---
# Physics-Constrained Deep Learning Model for Contactless Blood Pressure Monitoring from Triaxial Bodyseismography 

**Authors**: Yuanyuan Zhang, Yida Zhang, Jiahui Li, Yuyan Wu, Fei Dou, Xiao Yin, Zhenlin An, Hae Young Noh, Wenzhan Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.23562)  

**Abstract**: Ballistocardiography (BCG) is promising for unobtrusive long-term blood pressure (BP) monitoring in laboratory settings, but traditional BCG signals are vulnerable to the variations in body-bed interaction with shifted fiducial points in temporal or amplitude axis, and BP varies with personal hemodynamic changes, causing misaligned representations that affect model generalizability and robustness. In this work, we propose a non-invasive BP estimation framework, Phy-BP, based on triaxial bodyseismography (BSG) as an extension of BCG. Firstly, an adaptive quality-control algorithm is designed to select BSG segments enriched with cardiogenic components by jointly considering neighboring beat patterns and universal cardiogenic templates. Furthermore, a physical model is established to describe 3D wave propagation in the body-bed system and is subsequently embedded into the deep learning model to characterize the intrinsic coupling among triaxial BSG signals driven by a single cardiogenic excitation. Thus, multi-axis features are aligned during model training, improving robustness against distortions in real scenarios. Experiments on a 162-hour hospital dataset collected from 21 subjects reveal that the proposed Phy-BP can dynamically filter out low-quality measurements, and the deep learning model training is constrained by physical consistency across different axes to provide faithful BP monitoring, especially when training samples are limited. 

---
# ConvergeFlow: Language Flow with Provable Convergence to Token Embeddings 

**Authors**: Na Li, Yuchen Jiao, Changxiao Cai, Gen Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.23551)  

**Abstract**: Recent advances in continuous diffusion and flow-based language models (LMs) have achieved performance competitive with discrete LMs. However, existing continuous frameworks still rely on decoders supervised with cross entropy (CE) because the flow trajectories are not guaranteed to terminate at valid token embeddings. Motivated by this limitation, we introduce \textbf{ConvergeFlow}, an embedding-space flow-based LM, which constrains the data predictor to the convex hull of token embeddings and trains it solely with the mean squared error objective induced by flow matching. Under suitable regularity conditions, we prove that the resulting flow converges to valid token embeddings despite errors in the data predictor, enabling direct token prediction without a CE-supervised decoder. We further develop three sampling mechanisms for controlling the trade-off between the generative perplexity and entropy. Experiments on OpenWebText demonstrate that ConvergeFlow achieves performance competitive with existing continuous and discrete diffusion LMs. These findings demonstrate the potential of the flow-based paradigm for language modeling. Our code is available at this https URL. 

---
# The Interaction Tax: When Communication Erases Diversity in Multi-Agent Teams 

**Authors**: Summer Eunhyung Ann, Haokun Liu, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2608.23541)  

**Abstract**: Does multi-agent LLM interaction help or hurt? Some work reports gains from debate (Du et al., 2024), critique loops (Chen et al., 2025), and mixture-of-agents synthesis (Wang et al., 2025), while other work finds that interaction adds cost without improving quality under equal budgets (Tran & Kiela, 2026; Xu et al., 2026; Jarrett et al., 2025), or that independent sampling already captures multi-agent gains (Li et al., 2024). We argue this contradiction partly reflects a missing distinction, because not all multi-agent communication is equal. Different model families find structurally different solutions, but when agents read each other's complete outputs, their proposals converge within one round, erasing the diversity that motivates using multiple models. We call this the interaction tax. We test 11 verifier-scored optimization tasks under matched budgets and find that full-solution interaction is a weak default. Independent proposal generation avoids this collapse. Full-solution interaction mainly makes agents stay close to the first solution they see instead of trying different approaches, and critique helps only if the violated rule is easy for the LLM to find and fix. These results suggest that multi-agent performance depends less on the number of agents than on the information they exchange, and interaction helps only when agents share the right information at the right time. 

---
# Adapter-Based Few-Shot Continual Learning for Malicious Packet Recognition 

**Authors**: Kyle Stein, Guillermo Francia, III Eman El-Sheikh, Andrew Arash Mahyari  

**Link**: [PDF](https://arxiv.org/pdf/2608.23536)  

**Abstract**: The continual evolution of malware variants necessitates detection systems that can adapt to new threats without retraining from scratch. However, continually updating models on new data often leads to catastrophic forgetting, where previously learned knowledge is overwritten. While continual learning has been increasingly explored for malware detection, the specific setting of Few-Shot Class-Incremental Learning (FSCIL), where new malware classes must be learned from only a small number of labeled examples, remains comparatively underexplored. Therefore, this work investigates the FSCIL setting for malware classification. To address the stability-plasticity dilemma, we propose a hybrid framework that leverages a Self-Supervised Learning (SSL) backbone initialized through domain-specific pre-training on malware packets. Our method incorporates Low-Rank Adaptation (LoRA) to efficiently adapt the model during the base session while freezing the core backbone to preserve previously learned representations, alongside a prototype-based classification head for incremental sessions to establish robust decision boundaries from limited samples. Extensive experiments across several datasets demonstrate that our approach consistently outperforms prior malware FSCIL baselines and achieves state-of-the-art performance. 

---
# The Measurement Revolution? Credible Measurement and Inference in the Age of AI 

**Authors**: Melissa Dell, Ashesh Rambachan  

**Link**: [PDF](https://arxiv.org/pdf/2608.23524)  

**Abstract**: Artificial intelligence (AI) is transforming measurement in economics. AI models convert unstructured data, such as text and images, into structured variables at low cost, making previously prohibitive measurement feasible at scale. This shifts the bottleneck from finding any scalable measure of a phenomenon to choosing among many plausible ones, which may support different empirical conclusions. This review provides guidance for navigating that shift. We describe three stages at which AI enters the measurement pipeline---discovery, construct definition, and observation---and what each demands of researchers. We argue that credible inference with AI-generated variables requires appropriately designed validation: anchoring measurement to explicit criteria, rather than informal claims that a proxy is reasonable. We then examine how validation samples support valid inference even when AI predictions are arbitrarily biased, and what can be done when a random validation sample is unavailable. 

---
# When Names Cross Scripts: A Source-Grounded Benchmark for Historical Entity Reconciliation in the Mongol World 

**Authors**: Xiang Chen, Zeyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23507)  

**Abstract**: Historical people may appear under different languages, scripts, and transcription traditions, while distinct individuals may share highly similar or even identical names. This makes historical identity reconciliation more than a problem of string matching or transliteration. We introduce MHER, a provenance-controlled benchmark for pairwise reconciliation of person-name attestations from the Mongol world. MHER contains a balanced 396-pair Name-only core over 84 primary historical persons and a stricter 160-pair Source-grounded subset constructed from mention-by-source evidence, with entity-disjoint development and test splits.
Across five generative systems, correctly Source-grounded evidence improves paired TEST accuracy by 12.96 to 94.44 percentage points relative to Name-only input. On five identical-surface different-person cases, all models fail under names alone (0/25 model-item decisions), whereas Source-grounded evidence yields 24/25 correct resolutions, with the remaining output an abstention. Context-only ablations show that historical descriptions often carry substantial identity information, while explicitly signaled misgrounding controls produce substantially lower performance. We also find that names are not uniformly beneficial: for Qwen3-8B, restoring surface forms converts ten otherwise correct Context-only distinctions into false identity merges.
These results show that historical entity reconciliation depends not only on surface correspondence, but on whether identity judgments respond appropriately to provenance-controlled historical evidence. MHER therefore provides a controlled framework for studying evidence use, abstention, and failure modes in historical NLP. 

---
# Act with Intent: Distilling Behavior Intent for Vision-Language-Action Models 

**Authors**: Sangoh Lee, Sangwoo Mo, Wook-Shin Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.23478)  

**Abstract**: Vision-Language-Action (VLA) models can turn multimodal context into robot actions, but their action decoders are still trained largely by behavior cloning. This supervises which motor command was demonstrated while leaving implicit the local objective served by the behavior under the instruction. Future-based supervision enriches action learning with frames, latent observations, trajectories, or motion representations, but these signals capture particular realizations of what may happen rather than the shared semantic objective of the forthcoming behavior. We propose Intention Distillation (INDI), which distills behavior-level intent into the action decoder. During training, a frozen teacher VLM interprets a demonstrated segment from the current observation, instruction, coarse action summary, and corresponding execution video. From its standard inputs, the deployed VLA recovers the resulting multimodal intent representation at an intermediate decoder layer and uses it to organize action prediction together with representations of how the behavior unfolds and what it achieves. On SimplerEnv-Bridge, INDI improves GR00T-N1.7 from 64.3% to 84.7%, and on RoboCasa Kitchen it improves the controlled GR00T-N1.7 baseline from 64.1% to 70.3%, with consistent gains on $\pi_{0.5}$ across both benchmarks. In real-world tasks, INDI improves average success from 62.0% to 68.7%, with gains of up to 12.0 pp on longer-horizon tasks. Further analyses show that the recovered latent is used by the decoder, captures behavior objective and execution progress, and organizes downstream predictions in an objective-dependent manner. These results show that action decoders benefit from explicitly modeling the semantic objective of the behavior they generate. 

---
# What's the Catch? Evaluating Temporal Consistency in Vision-Language Models 

**Authors**: Marek Hradil, Danae Sánchez Villegas  

**Link**: [PDF](https://arxiv.org/pdf/2608.23474)  

**Abstract**: Vision-language models (VLMs) achieve strong performance on video and image-sequence benchmarks, yet it remains unclear whether they capture temporal structure. To study this question, we formulate temporal grounding as an anomaly detection problem, providing a simple and controlled evaluation that directly tests sensitivity to temporal consistency. We introduce TimeCatch, where temporal anomalies are created by swapping consecutive frames and frame-level anomalies by replacing a frame with Gaussian noise. Models are evaluated on anomaly detection and localization tasks across four synthetic and real-world datasets, alongside a human study. Our evaluation reveals a substantial gap between frame-level and temporal anomaly detection. While VLMs consistently detect frame-level anomalies and often localize them accurately, they perform near chance on temporal anomaly detection and only modestly above chance on localization. Humans, in contrast, achieve near-ceiling performance on both tasks. Additional analyses across model scales, prompting strategies, sequence lengths, and visual similarity suggest that these failures cannot be explained solely by limitations in perception or model capacity. Together, these findings indicate that current VLMs can identify anomalies within individual frames but struggle to integrate information across frames to reason about temporal consistency. TimeCatch provides a controlled benchmark for evaluating temporal grounding in vision-language models. 

---
# MetaCaster: Meta-Harness-Optimized Agent for End-to-End Few-Shot Learning of Lightweight Time Series Forecasters 

**Authors**: ChengAo Shen, Wenchao Yu, Fangyu Wu, Dongjin Song, Hanghang Tong, Dongsheng Luo, Wei Cheng, Haifeng Chen, Jingchao Ni  

**Link**: [PDF](https://arxiv.org/pdf/2608.23473)  

**Abstract**: Time series forecasting (TSF) is evolving toward multimodal and agentic settings, yet using foundation models remains uneconomical in resource-constrained scenarios, where compact, specialized forecasters are more desirable. However, lightweight forecasters typically require substantial training data, limiting their use in domains with scarce, slowly accumulated, or privacy-sensitive time series. To address this dilemma, we investigate the challenging problem of few-shot learning for lightweight forecasters. We propose MetaCaster, a meta-harness-optimized multi-agent framework that uses agentic data generation to automatically train specialized lightweight forecasters from only a few examples and textual contexts. Our work highlights a new TSF paradigm in which agents act not as forecasters but as intermediary engineers that prepare efficient, task-specific forecasters for deployment. Experiments on 18 datasets, 23 state-of-the-art lightweight forecasters, and 14 baselines demonstrate that MetaCaster achieves both data efficiency and computational efficiency while maintaining high-quality TSF performance. 

---
# InjecMEM: Memory Injection Attack on LLM Agent Memory Systems 

**Authors**: Hanling Tian, Gengyu Zhang, Zeyang Sha, Jingying Wang, Yuhang Liu, Zhehao Huang, Kun Yang, Xiaolin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23471)  

**Abstract**: Memory is becoming a default subsystem in deployed LLM agents to provide persistent personalization and continuity. This naturally prompts a question: will memory system introduce new vulnerabilities into agents? Thus we propose InjecMEM, a novel memory injection attack paradigm that requires only a single interaction (no read/edit access to memory store) to steer later responses of related queries toward a pre-specified output. Guided by the retrieval-then-generate mechanism of memory systems, we craft the injection with a retriever-agnostic anchor and an adversarial command. The anchor contains high-recall topical cues so that downstream retrieval consistently associates the record with the target topic. The command is a short sequence optimized to remain effective under uncertain fused contexts, variable placements, and long prompts so that it reliably steers outputs once retrieved. We learn the command via gradient-based coordinate search, averaging over synthetic prompt templates and insertion positions, and extend it to joint optimization across backbones to study transfer. Evaluated across multiple memory systems and backbone models, InjecMEM achieves reliable topic-conditioned retrieval and targeted generation, remains effective under memory drift, and leaves non-target queries unaffected. Our results underscore the need to harden memory systems and provide a reproducible framework for studying agent memory. 

---
# Machine Learning Assisted Inverse Design of Pixelated mmWave Patch Antennas 

**Authors**: Nadeem Rather, Holger Claussen, Lester Ho  

**Link**: [PDF](https://arxiv.org/pdf/2608.23469)  

**Abstract**: A machine learning-assisted framework for the inverse design of pixelated millimetre-wave patch antennas targeting the 22--30 GHz band is presented. The antenna surface is represented as a 19x23 binary pixel grid on a Rogers RT/duroid 5880 substrate, where each pixel is either metal or empty, with a continuous electrical path from the feed enforced by design. An initial dataset of approximately 6,000 full-wave CST simulations was collected from structured random pixel patterns, of which only around 40% achieved a resonance with |S11| <= -10 dB anywhere in the band, resulting in an imbalanced dataset. To improve simulation efficiency, an XGBoost binary classifier was trained on this data to distinguish resonant from non-resonant patterns before simulation. Using the classifier as a pre-simulation filter, an additional 4,000 patterns were selected and simulated, raising the overall proportion of resonant designs in the combined 10,000-sample dataset from approximately 40% to 52%. A hybrid CNN-BiLSTM forward surrogate was then trained on this augmented dataset to predict the full complex S11 response across 801 frequency points, using a physics-guided composite loss that explicitly emphasises resonance dip accuracy. Finally, an inverse design model was developed that optimises in a compact 64-dimensional latent space using gradient descent to generate pixel patterns matching a desired S11 specification. The results show good agreement between the surrogate-predicted and CST-simulated |S11| responses for the generated designs and demonstrate the feasibility of automatically designing and reconfiguring antenna structures. 

---
# Reward-Free Continual Adaptation for Resilient Space Robots 

**Authors**: Andrej Orsula, Miguel Olivares-Mendez, Carol Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2608.23452)  

**Abstract**: Space robots operate in extreme environments where hardware degradation can critically compromise traditional control strategies. While continual reinforcement learning offers a promising mechanism for online adaptation, it inherently requires access to a reward signal during deployment. However, precise reward computation in space is often infeasible due to the lack of external tracking systems and the overall complexity of the environment. To address the challenge of unobservable rewards, we introduce a reward-free continual learning framework that leverages latent-state world models. By pre-training a model-based agent across diverse simulations, the world model learns a robust predictor of the reward structure within its latent space. Upon deployment to an environment with severe hardware degradation, we freeze the observation encoder and reward predictor to update only the transition dynamics of the world model through unsupervised rollouts. By training the policy entirely on imagined trajectories generated by this updated world model, the agent adapts to altered dynamics without receiving new rewards. We demonstrate our approach across simulated planetary traversal, orbital navigation, and precision assembly tasks subjected to severe morphological failures. 

---
# Towards Comprehensive Basketball Understanding 

**Authors**: Yirong Hu, Jiayuan Rao, Yu Zhang, Shangzhe Di, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2608.23435)  

**Abstract**: Understanding a basketball game requires recognizing events, localizing actions, identifying players, and relating these to structured game knowledge. Existing benchmarks primarily evaluate these abilities one at a time, leaving the interactions among these abilities under-explored. We introduce BasketballBench, a multimodal benchmark comprising 7,980 questions across ten tasks in text, image, and video. It is built from the 2025-2026 NBA season and includes official playby-play, rosters and profiles for 530 active players, and 2,501 possession-level broadcast clips. We further propose BasketballSkills, an agent that composes eight basketball-specific perception and retrieval tools under four reusable skills that specify tool order, evidence bindings, and stopping conditions. Experiments show that current MLLMs struggle particularly on questions requiring the integration of multiple capabilities, whereas BasketballSkills outperforms them, highlighting the effectiveness of explicitly composing domain-specific capabilities for comprehensive basketball understanding. 

---
# ChebBooster: A Training-Free Approach for Efficient Diffusion Transformer Inference via Chebyshev-Inspired Extrapolation 

**Authors**: Chengjie Lu, Tianchi Deng, Zhengqi He, Chengwen Luo, Xueliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.23429)  

**Abstract**: Diffusion Transformers (DiTs) have shown strong performance in high-fidelity image generation, but their sampling process remains computationally intensive due to full model execution at every timestep. While cache-based acceleration has been explored to mitigate inference cost, naive reuse schemes suffer from low accuracy over long intervals, and Taylor-series-based extrapolation methods often face instability caused by Runge oscillations. In this paper, we propose ChebBooster, a training-free extrapolation framework based on Chebyshev polynomial theory that achieves stable and efficient acceleration for DiTs. Specifically, we adopt the Barycentric formulation to evaluate Chebyshev approximants with high numerical stability and minimal overhead, and further decouple the extrapolation into an offline weight precomputation phase and a lightweight online application stage. Extensive experiments across three representative DiT-based models, including DiT-XL/2, PixArt-$\Sigma$, and FLUX.1-dev, demonstrate that ChebBooster achieves consistent improvements in visual quality and inference efficiency, reaching up to $3.68\times$ latency speedup and $5.12\times$ FLOPs reduction, outperforming existing training-free baselines under diverse generation tasks and resolutions. 

---
# Adaptive Item-based Collaborative Structures via Noise Rescheduling in Diffusion for Generative Recommendation 

**Authors**: Jiaqi Wang, Tianying Liu, Heng Chang, Jihong Guan, Wengen Li, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.23400)  

**Abstract**: Discrete Diffusion Models (DDMs) have recently been introduced to recommendation systems, modeling user history as a token generation process via iterative denoising. However, while effective at capturing user-level sequential patterns, these methods often fail to explicitly integrate item-based collaborative filtering information, a critical component for accurate recommendation. This deficiency manifests in two key aspects: (1) the item representation is often semantic-focused, lacking collaborative priors for diffusion training; and (2) the denoising process employs a uniform noise schedule, treating all tokens indiscriminately and ignoring item-level adaptive structural dependencies. To bridge this gap, we propose ANR-DiffRec, a unified framework designed to encode item-based collaborative structures into discrete diffusion for generative recommendation. First, we explicitly incorporate an item co-occurrence matrix to guide semantic ID generation, providing a structured collaborative prior for discrete diffusion training. Second, we introduce an item-based adaptive noise rescheduling mechanism that dynamically adjusts denoising weights according to both local contextual recoverability and behavior-aware item dependencies. Specifically, the proposed strategy jointly models intra-item structural context and inter-item collaborative signals, enabling structure-aware denoising during diffusion training. Extensive experiments on multiple benchmarks demonstrate that our method consistently outperforms state-of-the-art generative recommendation models. Code: this https URL. 

---
# Right-Sizing LLM-Agent Decomposition in VAT Determination: A Pilot Controlled Sweep 

**Authors**: Pedro Santos  

**Link**: [PDF](https://arxiv.org/pdf/2608.23395)  

**Abstract**: Recent LLM-agent systems make conflicting design bets: decompose work across many narrow agents, or use one strong tool-using agent. This pilot studies that choice on bounded cross-border VAT determination with reverse charge, where every case has an oracle label and each intermediate decision is independently scoreable. We hold the activity surface fixed (subtasks, tools, I/O schemas, validation checks, orchestrator, base model, and merge policy) and vary only the assignment of subtasks to workers across four orchestrated configurations, from one wide worker to five narrow ones, against S0, a tuned no-orchestrator single agent, with a deterministic rule engine as oracle. The program spans 4,400 runs: a 40-case, five-repeat main sweep, matched-token arms separating prompt-budget from agent-count effects, and three failure-injection arms, all judged against pre-registered falsification criteria. The two intermediate configurations lead on accuracy (0.830, against endpoints at 0.720 and 0.770) but miss the pre-stated bar against the fine endpoint, so the intermediate-optimum hypothesis remains unsupported at pilot scale. The single agent does not Pareto-dominate the orchestrated set. The matched-token criterion fires: the budget-matched single agent lands 6.5 points below the leader, but the interval includes zero, so any advantage is consistent with a prompt-budget explanation. Under injection, availability faults are absorbed at every granularity, with wide-scope restart over-recovering its baseline by +0.160, while one schema-conforming hallucinated record degrades every configuration and inverts the ordering, hitting fragmented configurations hardest. The contribution is a bounded, preregistered pilot heuristic for right-sizing decomposition (place one partition boundary at the dependency-layer midpoint), released with oracle, dataset, harness, raw traces, and analysis pipeline. 

---
# Towards a Densing Law for User Representation Learning at Billion-Scale Capacity 

**Authors**: Bin Dou, Junru Zhang, Zhaoyi Yuan, Wuliang Huang, Letian Gong, Baokun Wang, Huan Li, Yu Cheng, Weiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23392)  

**Abstract**: User representation learning in real-world industrial scenarios is commonly scaled by increasing user amount, behavioral sequence length and model size. However, existing methods face two challenges: (i) Bottleneck for raw data scaling at billion-scale capacity, as performance exhibit diminishing performance gains with larger-scale raw text user behavioral input, which can be mitigated by tokenization. (ii) Lack of quantitative analysis of how tokenization configurations should scale with data size. In this report, we propose User Behavioral Densing Law for characterizing the quantitative relationship between data scale and the minimum sufficient tokenization capacity. Firstly, we conduct a pilot study on raw & tokenized scaling comparison on billion-scale Alipay dataset, revealing the raw data scaling bottleneck and the sustained gains enabled by tokenization. To derive the scaling pattern governing the minimum sufficient tokenization configuration at different data scales, theoretical analysis and systematic experiments are employed to summarize the quantitative scaling pattern. We find an approximately linear relationship between the logarithms of minimum sufficient tokenization capacity and input data size measured by tokens, and the scaling slope varies systematically with the tokenization method and data source, reflecting differences in representation-space redundancy and intra-source uniqueness. Guided by the proposed law, we further develop ALGN, an adaptive variable-length tokenization method that improves capacity allocation. Extensive experiments across diverse data sources, tokenization methods, and downstream tasks demonstrate the generalizability and reliability of the User Behavioral Densing Law, providing practical guidance for tokenization configuration selection in large-scale user representation learning. Moreover, ALGN outperforms existing baselines. 

---
# Cross-Domain, Multi-Task Data-to-Text Generation without In-Domain Training Data 

**Authors**: Yifei Song, Kun Efimov-Zhang, Claire Gardent  

**Link**: [PDF](https://arxiv.org/pdf/2608.23391)  

**Abstract**: Structured data exists in many forms (tables, knowledge graphs, charts, and time series), and converting it into text may involve different generation tasks. However, most prior work on data-to-text (D2T) generation has focused on specific tasks and datasets, relying either on task-specific training data or on the zero-shot capabilities of large language models. We study cross-domain D2T generation in a setting where neither in-domain training text nor test references are available, and where domains, generation goals, and input structures vary substantially. We compare data-driven knowledge distillation (DDKD) against zero-shot inference and fine-tuning on out-of-domain D2T data, and introduce structure-preserving augmentation via structural subsampling and perturbation. Experiments on five benchmarks show that, at constant model size (1.7B parameters), DDKD consistently outperforms both fine-tuning and zero-shot inference. Moreover, the resulting small models outperform a much larger finetuned model on two of the five domains, achieving comparable performance on the remaining three. We further construct QUINTD-5, a fivefold extension of QUINTD-1, and show that simply scaling real target-domain inputs yields only modest gains, whereas our augmentation strategy remains more effective and more cost-efficient for cross-domain distillation. 

---
# Cross-lingual Biography Enrichment via Claim Extraction and Alignment 

**Authors**: Yifei Song, Ziyang Chen, Emil Sayilov, Claire Gardent  

**Link**: [PDF](https://arxiv.org/pdf/2608.23390)  

**Abstract**: English Wikipedia is often treated as the default encyclopedic source, yet non-English Wikipedia editions can contain richer locally grounded information for long-tail figures. We study cross-lingual biography enrichment: enriching an existing English biography with facts supported by a non-English biography about the same person. Focusing on women from non-English-speaking contexts, we introduce \textsc{CLAW-4L}, a benchmark consisting of 300 Wikipedia biography pairs linking an English biography with its French, Chinese or Azerbaijani counterpart, along with claim annotations and a fine-grained claim-pair relation corpus. We propose a claim-based enrichment framework that extracts English claims from both biographies, aligns them to identify enrichment evidence from the non-English biography, and rewrites the English biography using the selected claims. Our results show that non-English Wikipedia biographies provide valuable evidence for improving English biography coverage, while lower-resource settings remain challenging. 

---
# Adversarial Entropy Inflation Against Gumbel-Based Inference Verification 

**Authors**: Nikita Kezins  

**Link**: [PDF](https://arxiv.org/pdf/2608.23375)  

**Abstract**: Gumbel-based inference verification bounds LLM weight exfiltration by only forgiving token choices that plausibly arise from honest GPU nondeterminism, reporting a >200x slowdown for a steganographic adversary under benign prompt traffic. This bound assumes a passive attacker; we show it degrades sharply against an adversary who instead controls the prompt distribution. Because the verifier's admissible-token-set size is driven by the model's own output entropy, prompts engineered to break grammatical and sub-word structure -- rather than benign conversational traffic -- widen that set and open a materially larger covert channel. Across six instruction-tuned models spanning 1B to 32B parameters and three random seeds, our strongest attack (character- and script-level disruption) roughly doubles bits leaked per token relative to benign prompts, cutting the slowdown factor to 60x - 118x. These results indicate that static, benign-traffic-calibrated thresholds are insufficient for this defense, and that jitter-forgiveness thresholds should instead be calibrated dynamically against local token entropy. 

---
# DF-MoE: Generalizable Deepfake Detection via Multimodal Sparse Mixture-of-Experts 

**Authors**: Vlad Hondru, Florinel Alin Croitoru, Iuliana Georgescu, A. Sophia Koepke, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23363)  

**Abstract**: Audio-visual deepfake detection is an actively studied topic, where one of the main challenges is to develop detectors able to generalize across deepfake generation methods. We conjecture that overfitting can be mitigated by extracting multiple high-level cues from the available audio and visual modalities via pre-trained models. We therefore assemble a wide variety of pre-trained models to extract features that encode mouth movements, face parsing, facial expressions, head pose, gaze tracking, heart rate, audio emotion and speech activity. We further integrate both unimodal and multimodal cues via a Mixture-of-Experts (MoE) backbone to detect deepfakes. We perform in-domain and cross-domain experiments on five benchmarks for deepfake detection (MAVOS-DD, AVLips, PolyGlotFake, BioDeepAV, FakeAVCeleb) to compare our framework (DF-MoE) with state-of-the-art methods. Our results indicate that DF-MoE obtains superior deepfake detection results, surpassing all competing methods. We release our code at this https URL. 

---
# The Emergence of Relevance Through Axiomatic Attention Patterns During LoRA Fine-Tuning 

**Authors**: Matthew Perlman, Atharva Nijasure, James Allan  

**Link**: [PDF](https://arxiv.org/pdf/2608.23338)  

**Abstract**: LoRA fine-tuning is standard for adapting LLMs to reranking, but it remains unclear where in the network task-specific relevance behavior is learned and what attention-level changes accompany that learning. Through ablation and attention experiments, we identify where LoRA attention updates to RankLLaMA improve performance and whether those gains coincide with interpretable relevance-oriented attention patterns such as lexical matching, rarity sensitivity, and query-document interaction. We find that given LoRA fine-tuned MLPs throughout the network, restricting LoRA attention updates to a compact mid-network region is sufficient for recovering over half of the performance gained by applying LoRA to all attention layers, and that omitting attention fine-tuning in this region hurts performance more than elsewhere in the network. Additionally, we show that regions where applying LoRA affects performance the most overlap with regions where fine-tuning increased attention to axiomatic IR features. Rarity sensitivity, document-query interaction, and several compositional features are highly correlated with gains in ranking performance. Our results support an interpretable, correlational account of how relevance-oriented behavior emerges during LoRA fine-tuning and point toward improved strategies for adapting rerankers. 

---
# Thinking Beyond Videos: Unifying Video Reasoning and Deep Research for Open-World Video Agents 

**Authors**: Wenqi Liu, Shijie Ma, Yunxiao Wang, Meng Liu, Qile Su, Han Liu, Bohan Hou, Xuanyu Zheng, Changyi Liu, Tianke Zhang, Haonan Fan, Kaiyu Jiang, Yingxin Li, Jiankang Chen, Xu Wang, Bin Wen, Tingting Gao, Han Li, Jianhua Yin, Yinwei Wei, Xuemeng Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.23329)  

**Abstract**: Open-world video understanding often requires a model to locate sparse visual evidence and acquire external knowledge that is absent from the video and its parametric memory. While Thinking-with-Videos enables active temporal perception and Deep Research supports multi-step information seeking, the two capabilities are typically developed in isolation. We introduce VideoRover, a unified Video Deep Research framework that iteratively coordinates video cropping, multimodal search, and webpage browsing. Given a video-question pair, VideoRover uses each tool result to select the next action, so localized video clips guide external retrieval and retrieved evidence triggers further video inspection and verification. To develop this capability, we construct an automated data curation pipeline, producing 26K verified SFT trajectories and 3K challenging RL instances. We also introduce VideoRover-Bench, a benchmark stratified by video duration and research difficulty. Experiments on VideoDR and VideoRover-Bench show that our VideoRover-8B-RL achieves performance comparable to proprietary models in the direct-answer setting without tool use while outperforming larger open-source models equipped with the same tool suite. Ablation studies and training dynamics further validate the complementary roles of active video grounding, external retrieval, and long-horizon reinforcement learning. 

---
# Mycelial Search: A Graph-Structured Metaheuristic for Continuous Optimisation 

**Authors**: Mohammad Mahdi Dehshibi  

**Link**: [PDF](https://arxiv.org/pdf/2608.23323)  

**Abstract**: Continuous optimisation methods need to balance sharing information and maintaining alternative search directions. In this paper, we introduce Mycelial Search (Myco), a graph-structured metaheuristic designed around active tips, community-weighted flow, adaptive cord plasticity, and anchor-based injection. Candidate solutions form an evolving spatial graph in which a Louvain partition distinguishes within-community from cross-community information exchange. Adaptive cord plasticity subsequently modifies active tip-to-tip edges according to their alignment with the local flow. An anchor-based injection mechanism supplements the graph-driven tip dynamics. We evaluated Myco on the CEC 2022 single-objective bound-constrained benchmark suite at dimensions $D=10$ and $D=20$, using 30 independent runs per algorithm-function pair. The comparison includes eleven established optimisers from several search families. Myco reaches competitive results on selected functions across both dimensions. The ablation analysis further shows that community structure regulates the range of graph-based information exchange, whereas cord plasticity controls the persistence of local directional influence. These findings indicate that graph-structured local interaction can support continuous optimisation, while its effectiveness depends on landscape structure and information transfer across local search regions. 

---
# FIDES: A Concordance Protocol for LLM-Generated Trading Strategies 

**Authors**: Arther Tian, Alex Ding, Simon Wu, Aaron Chan  

**Link**: [PDF](https://arxiv.org/pdf/2608.23308)  

**Abstract**: An LLM asked for a trading strategy returns three artifacts at once: a natural-language rationale, an executable implementation, and once run, a track record. Whether these are the same object is rarely checked. We present FIDES, a measurement protocol that treats them as three views to be reconciled rather than one deliverable to be graded. Through dual delivery, a single model call returns both a natural-language strategy with an explicit claimed edge and a self-contained strategy(df) function. FIDES executes the code in a sandbox against a lag-one out-of-sample backtest and scores three concordance gaps: say to do, do to real, and say to result.
On 8 liquid US ETFs across four models plus a two-stage elicitation arm, 40 strategies, 2023 to 2024 out-of-sample, three findings stand out. First, concordance does not predict profit: only 2 of 40 strategies beat buy-and-hold, and a plain sma(50,200) rule outperforms every model's mean Sharpe. Second, self-assessment is badly calibrated: 32 of 40 strategies claim to beat buy-and-hold and exactly one does. Third, swapping the language-code judge for a second model flips say to do on more than half of items. Injecting this http URL(-1) drops do to real by 0.33 on average, while our runtime future-information probe fired on neither clean nor injected code. We frame FIDES as a protocol for measurement fidelity, not a claim about market performance. 

---
# Evaluating SAT Solver Metrics as Predictors of Human-Perceived Nonogram Difficulty 

**Authors**: Changdao He, Yibing Ju, Jonathan Calver, Alice Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.23300)  

**Abstract**: Algorithmic solver effort is often assumed to align with perceived puzzle difficulty, but this assumption is rarely tested against human solving data. We evaluate this assumption for Nonograms, a popular logic puzzle similar to Sudoku in which numeric clues along each row and column determine a unique solution grid. We formulate Nonograms as a constraint satisfaction problem and solve them using existing SAT solvers. We then conduct a user study in which we collect data on both participant interactions and reported difficulty. We find that neither participants' reported difficulty nor their behavioural signals correlate meaningfully with SAT solver metrics; however, we find evidence that expertise moderates the relationship between solver metrics and reported difficulty. In this process, we uncover distinct, recurring solving strategies that indicate human preference for complex propagation, diverging from solver-measured complexity. 

---
# Sigmoid Attention as a Better Substrate for Learned KV Cache Eviction 

**Authors**: Isaac  

**Link**: [PDF](https://arxiv.org/pdf/2608.23296)  

**Abstract**: Learned KV-cache eviction often faces a soft-to-hard mismatch: during training, differentiable gates typically attenuate token contributions, whereas inference saves memory only when KV entries are physically removed. We ask whether the attention substrate affects this soft-to-hard transition. Using GPT-2-scale Transformers trained on OpenWebText, we run a controlled $2\times2\times2$ comparison over attention type, learned gating, and positional encoding. Although sigmoid attention is worse as a dense language model, learned hard eviction changes the useful operating points: sigmoid-gated models delete KV entries with negligible PPL change relative to their own no-eviction references. Under a matched live-cache protocol on the same dense backbones, learned sigmoid gates obtain lower PPL than our H$_2$O and KeyDiff implementations, whereas softmax gates do not uniformly beat these post-hoc methods. The results suggest that attention normalization can substantially affect whether a training-time soft gate transfers cleanly to hard KV deletion. 

---
# How Much Regularization Survives Averaging? Update Masking in Federated Learning 

**Authors**: Wenhao Yan, Fu Kuroda, Yucheng Jin, Zhenke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.23286)  

**Abstract**: Federated learning on non-IID data seeks flat minima to generalize across clients, and existing methods borrow sharpness-aware minimization from centralized training. There is a second way to reach flat minima, in which the regularization comes for free from noise added to the parameter updates, and it has never been carried over to the federated setting. We show the reason. Masking charges the optimizer for moving in sharp directions. We prove that when each client draws its own mask, federated averaging weakens that charge by exactly the cohort size, and that giving every client the same mask brings it back by a factor equal to the inverse gradient diversity of the cohort. In our experiment setting on CIFAR-10, that factor is 1.19 out of a possible 10. Turning off minibatch sampling raises it to 8.96, while changing data heterogeneity a hundredfold leaves it between 1.17 and 1.50. The configurations keeping the regularization train far too poorly to use. 

---
# E2S-Pruner: Progressive Two-Stage Evidence Fusion for Visual Token Pruning in Vision-Language Models 

**Authors**: Taoyu Qian, Qi Wang, Daqian Shi, Yuanhao Jiang, Shang Gao, Hualong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23253)  

**Abstract**: Vision-language models typically encode an image into hundreds of visual tokens, incurring substantial inference latency and GPU memory overhead. Existing pruning methods largely rely on attention scores and directly aggregate outputs across attention heads and network layers, making it difficult to characterize evidential uncertainty and conflict. We propose E2S-Pruner, a progressive two-stage evidence-fusion framework for visual token pruning that requires no auxiliary model, trainable parameters, or fine-tuning. In the first stage, E2S-Pruner treats each attention head as an independent evidence source, estimates its reliability from evidence clarity and inter-head consistency, and represents each visual token using three states: important, unimportant, and uncertain. In the second stage, Dempster--Shafer evidence theory is used to quantify inter-layer conflict and fuse complementary evidence from multiple network layers. We further introduce a spatial novelty constraint that promotes coverage of distinct image regions and prevents the retained tokens from concentrating in a few locally salient areas. On LLaVA-1.5-7B, E2S-Pruner retains 98.0%, 96.8%, and 90.6% of the aggregate performance when the average numbers of retained visual tokens are 192, 128, and 64, respectively, while improving throughput by 1.96x and 2.09x under the 128-token and 64-token settings. Experiments on Qwen2-VL-7B further demonstrate cross-model generalization. Code is available at this https URL. 

---
# Future Querying: Can LLMs Serve as Implicit Medical World Models? 

**Authors**: Siri Willems, James Butterworth, Lore Goetschalckx, Peter Vrancx, Philippe Modard, Elke Giets, Ludovic Denoyer  

**Link**: [PDF](https://arxiv.org/pdf/2608.23248)  

**Abstract**: Traditional clinical prediction models rely on task-specific pipelines and curated, structured data, which scale poorly and underutilize unstructured text. To address this, we introduce future querying, a paradigm that probes whether large language models (LLMs) can function as implicit medical world models by evaluating their ability to answer time-indexed clinical queries about a patient's future. Our framework operates on unstructured clinical documentation using endpoint-agnostic training, enabling a single model to answer diverse clinical queries over patient trajectories without manual feature engineering or task-specific retraining. We show that small, locally fine-tuned open-weight models can match or approach larger proprietary systems, making the framework suitable for privacy-preserving, on-premise deployment. Evaluated on a new synthetic medical reports dataset and real ICU notes from the MIMIC-IV dataset, our results provide encouraging evidence that LLMs can capture aspects of clinical dynamics. 

---
# Multi-Winner Voting with Argumentative Ballots 

**Authors**: Ryuta Arisaka, Hirotaka Ono  

**Link**: [PDF](https://arxiv.org/pdf/2608.23247)  

**Abstract**: We introduce multi-winner voting with argumentative ballots (MVArg) and investigate theoretical properties. As our conceptual contribution, we generalise approval ballots to argumentative ballots, thereby allowing voters to express defeasible preferences over candidates. We accordingly generalise voter cohesion and justified representation axioms JR, PJR and EJR. As our theoretical contribution, we establish several key results. First, MVArg is strictly more expressive than multi-winner voting with approval ballots (MV). Second, our notions of cohesion and justified representation are conservative generalisations of their counterparts in MV. Third, the MVArg counterpart of JR can always be satisfied, whereas the counterparts of PJR and EJR cannot always be. Fourth, although verifying whether a winner set satisfies the MVArg counterpart of JR is already coNP-hard, such a winner set can be constructed in polynomial time. All definitions, propositions, auxiliary lemmas and theorems have been formalised and mechanically checked in Lean 4. 

---
# Credal Large Language Models for Semantic Commitment under Uncertainty 

**Authors**: Shireen Kudukkil Manchingal, Sofiia Nikolenko, Fabio Cuzzolin  

**Link**: [PDF](https://arxiv.org/pdf/2608.23244)  

**Abstract**: Large language models (LLMs) often produce fluent but incorrect answers with unwarranted confidence. A central limitation is that standard LLMs represent uncertainty through a single predictive distribution, conflating epistemic ignorance with genuine ambiguity. We introduce Credal Large Language Models (CLLMs): an ensemble of LoRA adapters induces a credal set whose lower and upper probabilities expose the spread of plausible predictive distributions rather than collapsing to a single softmax output. From this representation we derive two complementary commitment scores. Credal Token Commitment (CTC) is a token-space score that combines lower-bound support, credal width, and intersection entropy, computed without additional generation. Semantic Commitment Consistency (SCC) extends commitment to semantic space using sampled completions, with SCC-Gap measuring the mismatch between token-level and semantic-level support. We evaluate hallucination detection, calibration, selective prediction, and reasoning on Gemma-2-9B, Llama-3.1-8B, and Qwen2.5-7B across OpenBookQA, CoQA, TriviaQA, and ARC-Challenge. CLLM is the best method on QA accuracy at competitive expected calibration error, and CTC tracks the best hallucination AUROC within 1.5 pp on most settings without additional generation. On selective prediction at 80% coverage, CLLM with SCC reaches 99.0% accuracy on OpenBookQA, and on ARC-Challenge CLLM with Csem confidence achieves <= 0.6% ECE across the three backbones. 

---
# Retrieval-Augmented Classification of Environmental Mitigations in Hydropower Licensing Documents 

**Authors**: Hong-Jun Yoon, Tom Ruggles, Joanna Lee, Debjani Singh  

**Link**: [PDF](https://arxiv.org/pdf/2608.23241)  

**Abstract**: Identifying and classifying environmental mitigation obligations in Federal Energy Regulatory Commission hydropower licensing documents is a labor-intensive task requiring deep domain expertise. We formulate this as a multi-label classification problem over a structured 135-category taxonomy and address the central challenge of severe label scarcity: 40 of 135 categories have no training examples, and 26 have fewer than five. A supervised Bidirectional Encoder Representations from Transformers (BERT)-based pipeline, while effective on well-represented categories, achieves F1 of zero on unseen classes regardless of augmentation strategy. We introduce a Retrieval-Augmented Generation (RAG) pipeline that conditions classification on retrieved category definitions, enabling zero-shot generalization across the full label space. We further propose a hybrid system that combines BERT detection with RAG classification, exploiting the high recall of fine-tuned detection and the zero-shot coverage of retrieval-augmented reasoning. Evaluated on the full set of 2017 license documents (5,860 paragraphs, 135 categories), the hybrid achieves a Micro F1 of 0.524, outperforming the BERT-only pipeline (0.477) and the RAG-only pipeline (0.416) across all training-support buckets. 

---
# Think Only When Needed: Prompt-Authority Control for Selective Slow-Path Intervention in Vision-Language-Action Manipulation 

**Authors**: Zhiruo Zhou, Zelin Li, Xiwen Chen, Jiazhuo Li, Chenwei Wang, Huiming Chen, Xiaojun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23224)  

**Abstract**: Retrieval can efficiently and effectively augment a frozen vision--language--action (VLA) policy without retraining, yet retrieved text becomes a control intervention once it enters the executed prompt. In a matched audit, raw appended text reduces mean success from 92.47\% to 3.00\%, while meaningful and length-matched meaningless appends both fail on all 500 states. This result identifies \emph{prompt-form collapse}: changing the instruction form, rather than adding useful semantics, can dominate execution. We introduce TOWN-VLA (Think Only When Needed), a prompt-authority interface that separates candidate generation from permission to alter the policy input. A fixed compatibility rule authorizes a canonical compact instruction; otherwise, the interface restores the original Base prompt exactly. Across 900 audited routes, every route follows this contract: 525 routes recover Base with matching hashes, and all 375 authorized prompts preserve the task signature. On a matched $4\times7$ LIBERO-Plus evaluation with 10{,}030 episodes per method, success rises from 69.5\% to 73.1\% ($+362$ episodes; 95\% CI 1.89--5.45 points), improving on six perturbation axes and all four suites. On a physical PiPER arm with a frozen \pizerofive{} checkpoint, success rises from 52.7\% to 78.7\% over 150 trials per method ($p=3.16\times10^{-6}$). Prompt authority is enforceable for a frozen controller; oracle-free admission calibration is the next deployment target. 

---
# AI Surrogate Modeling for Real-Time Tokamak Equilibrium Prediction: Benchmarking Neural Architectures and Validation on EXL-50U 

**Authors**: Guoyang Shi, Zitong Zhang, Siqi Ding, Jianguo Chen, Yapeng Zhang, Jiayi Zhi, Hanyue Zhao, Tianyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23217)  

**Abstract**: Fast and reliable plasma equilibrium prediction is essential for real-time tokamak operation and control, but conventional Grad-Shafranov (GS) solvers are often too costly for real-time deployment. We develop an AI surrogate framework and benchmark five architectures (MLP, CNN, FNO, Transformer, and KAN) on a numerical GS database with 100,000 IID and 10,000 OOD samples. Under a unified protocol, we evaluate accuracy, inference efficiency, model scaling, and robustness. We also establish device-level validation on the EXL-50U tokamak by linking numerical GS solutions, surrogate predictions, and the standard Shape Editor reference to assess simulation-to-device consistency. The surrogates achieve errors of $10^{-3}$-$10^{-2}$ relative to GS solutions, while the GS-to-device discrepancy remains at $10^{-3}$. Transformer gives the best IID accuracy, whereas CNN offers the best balance of accuracy, robustness, and speed, reaching 0.7 ms TensorRT latency. On unseen plasma geometries and parameter regimes, CNN and FNO show the strongest extrapolation stability, with 4%-5% relative $L_2$ error, while models with weaker inductive biases degrade more substantially. Scaling data and model capacity improves interpolation but not necessarily extrapolation, revealing a trade-off between capacity and OOD generalization. Overall, this work provides a systematic, device-consistent benchmark for AI-based GS prediction and practical guidance for selecting reliable surrogates for real-time plasma control and fusion applications. 

---
# BenthicDINO: Physics-Informed Self-Distillation for View-Invariant Side-Scan Sonar Representations 

**Authors**: Taqi Hamoda, Hayat Rajani, Nuno Gracias  

**Link**: [PDF](https://arxiv.org/pdf/2608.23215)  

**Abstract**: Automated perception in side-scan sonar (SSS) imagery is severely hindered by physical acoustic artifacts, resulting in representations that inextricably mix intrinsic seabed reflectivity with transient viewing geometries. Existing self-supervised learning (SSL) frameworks rely on augmentations designed for natural images, failing to account for acoustic degradation and explicitly enforce view-invariance. To address this gap, we introduce a physics-informed self-distillation framework built upon the DINOv3 architecture utilizing a ConvNeXt-v2-Tiny backbone to maximize data efficiency. The proposed methodology enforces view-invariance through two primary mechanisms: physically motivated augmentations that simulate speckle noise, range-dependent attenuation, and radiometric miscalibration; and a Hilbert-Schmidt Independence Criterion (HSIC) penalty that explicitly decouples learned dense patch features from physical viewing parameters. Furthermore, we propose a dense, hierarchical feature fusion strategy across all four network stages to preserve fine-grained sediment details alongside deep semantic abstractions. Extensive evaluation demonstrates that the framework natively groups complex benthic topographies into stable, noise-free semantic clusters without relying on manual annotations. During supervised downstream tasks on the S3Seg dataset, the fused representations exhibited exceptional data efficiency, achieving 96% of its absolute peak performance using only 10% of the available annotated data, ultimately reaching a mean Intersection over Union (mIoU) of 71.4% and an overall accuracy of 86.5%. 

---
# NetConfArena: An Executable Benchmark for LLM Agents in Closed-Loop Network Configuration 

**Authors**: Chang Liu, Xiaohui Xie, Xinyi Chen, Yong Cui  

**Link**: [PDF](https://arxiv.org/pdf/2608.23179)  

**Abstract**: Large language model (LLM) agents are increasingly attractive for automating network configuration, yet their reliability and failure patterns are poorly understood. An essential prerequisite is to assess such agents in a realistic but risk-free environment. Existing benchmarks, however, fall short: they often treat configuration as static command generation or rely on overly simplified settings. Such evaluations understate the core challenges of network configuration, where correctness requires reasoning about protocol complexity and topology dependence. We present NetConfArena, an executable benchmark for evaluating LLM agents in closed-loop network configuration. NetConfArena places agents in emulated multi-device networks, provides a standardized and compact action interface for task execution, and evaluates the resulting network behavior with hidden task-specific executable test cases. The benchmark relies on an LLM-assisted, emulation-grounded pipeline, which converts human-oriented network materials into reusable parameterized task templates. We evaluate representative LLM agents on 480 task instances instantiated from 96 protocol-focused task templates, yielding 3840 execution trajectories, and show that failures are not limited to command errors. The failures also reveal gaps in task-specification adherence and robust planning and execution. These findings suggest two future directions: using validated trajectories as supervision signals to improve foundation models, and designing harness mechanisms that make agent execution more reliable and accountable. 

---
# Counterfactual Transition Graphs: Evaluating Cross-Class Transition Quality 

**Authors**: Syed Muhammad Hamza Zaidi, Szymon Bobek, Grzegorz J. Nalepa, Myra Spiliopoulou  

**Link**: [PDF](https://arxiv.org/pdf/2608.23164)  

**Abstract**: Counterfactual (CF) explanations for time-series classifiers are usually evaluated one example at a time: what minimal edit flips this single window's prediction? We argue that the more informative question for diagnostic interpretability is structural: how does the classifier connect its own classes to each other?
We propose a counterfactual transition graph (CGT) in which each node is a class and each edge weight is the CF reliability of the transition from one prototype to another under a proximity aware retrieval sweep. On a six-class hand-movement task, we induce a CGT that reveals a non-trivial topology, which is not predicted by the binary confusion matrix: it shows that counterfactual reachability does not align with classifier accuracy and even runs counter to it (Spearman $\rho=-0.37$ over the 15 pairs), i.e. the boundaries the classifier separates most confidently are among those an in-distribution edit can least often cross.
Our framework is method agnostic, i.e. any CF-explainers can be used. Presently, we use it to juxtapose replacement-based CFs with gradient-based CFs; gradient-based methods reach almost any class by stepping off the data manifold, while replacement-based methods stay on it and fail on precisely the rigid boundaries. 

---
# Language Chain in Alignment: Cross-Lingual Ranking Preference Optimization 

**Authors**: Seungyoon Lee, Minhyuk Kim, Jungseob Lee, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2608.23149)  

**Abstract**: The alignment of Large Language Models heavily relies on English-centric high-quality preference data, which often leads to suboptimal performance in other languages. In this paper, we propose Cross-Lingual Ranking Preference Optimization (CRPO), a novel framework that leverages robust preference knowledge from English to facilitate preference alignment in the target language. We design a hierarchical structure within parallel preference pairs across the target language and English to jointly optimize intra- and inter-lingual preferences, thereby enhancing language adaptation and output quality. Building on the LambdaLoss framework, CRPO goes beyond the binary comparison based optimization by providing a relative ranking signal across multiple candidate responses. Our experiments across five languages with varying resource scales demonstrate that CRPO consistently outperforms standard approaches in both instruction-following and knowledge utilization capability. Notably, the robust performance gains observed across various weighting schemes further validate the empirical effectiveness of our hierarchical design in a multilingual setup. Furthermore, our findings highlight that CRPO significantly improves both reward margins and the log-probability of desirable responses, contributing to a more stable preference manifold for cross-lingual alignment. 

---
# Pointing-VLA: Typed Spatial Grounding Interfaces for Vision-Language-Action Manipulation 

**Authors**: Xiwen Chen, Zelin Li, Zhiruo Zhou, Huiming Chen, Chenwei Wang, Xiaojun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23138)  

**Abstract**: Vision-language-action (VLA) models often expose spatial grounding through autoregressive text coordinates or opaque action tokens, creating brittle interfaces between multimodal reasoning and robot execution. We present Pointing-VLA, a typed hidden-state spatial readout built on Embodied-R1. Geometry-specific heads predict normalized points, object-functional grounding (OFG) heatmaps, and visual trajectories without serializing geometry as text. For the evaluated Bridge/WidowX and physical pick-place deployments, an explicit execution contract assigns PICK to source-conditioned OFG and PLACE to Pointing, providing direct stage-aligned spatial targets. Pointing-VLA achieves SOTA performance on Bridge/WidowX, averaging 72.9\% across the evaluated four-task set without Bridge-specific finetuning under collision-enabled CuRobo execution. Pointing and OFG show complementary strengths across native and cross-dataset evaluations. The OFG/contact readout transfers to NORA-1.5, preserving or improving success while reducing recorded controller time by more than 20$\times$; typed heads are also 6.68--6.90$\times$ faster than Embodied-R1 text decoding on a shared external suite. When integrated as spatial guidance for a $\pi_{0.5}$ action policy, Pointing-VLA raises autonomous real-robot success from 52.7\% to 80.7\% across three visual contexts. These results establish typed spatial readouts as an efficient, inspectable interface between embodied reasoning and robot execution. 

---
# LITERARYBIGFIVE: Author-Personalized Text Generation in a Unified Interpretable Space 

**Authors**: Jinghui Zhang, Lang Gao, Ao Li, Mingzhe Li, Ruihong Zeng, Zirui Song, Kentaro Inui, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.23124)  

**Abstract**: Personalized text generation for authors and literary writing is essential for applications such as adaptive writing assistants, creative support tools, and computational literary analysis. However, existing approaches to author modeling and personalization often represent writing behavior as independent labels, requiring large-scale corpus collection or fine-tuning for each author or stylistic category. Such formulations are costly, difficult to interpret, and poorly suited for generalizing across authors. Inspired by the Big Five model's dimensional view of personality, we propose LiteraryBigFive, a framework that reframes authorial writing characteristics as coordinates within a unified and interpretable space. In this space, we derive each interpretable axis (e.g., Classicism, Emotionality) from activation-space contrasts between author-written and neutral passages, yielding distinct stylistic dimensions that allow texts or authors to be positioned within a five-dimensional system. Beyond localizing different authors, we further introduce an interpretable steering mechanism, which adaptively guides text generation toward target coordinates to perform author-personalized writing. Experimental results show that LiteraryBigFive improves authorial expressiveness while preserving semantic fidelity. The derived author per-axis scores strongly correlate with real-world literary consensus, offering transparent and interpretable explanations of author-specific generation behavior: this https URL. 

---
# Statistical Machine Translation Systems of English-Pnar Language Pair : Some Insights of the Emperical Study 

**Authors**: Edawanbiang Dhar Surmila Thokchom, Thoudam Doren Singh  

**Link**: [PDF](https://arxiv.org/pdf/2608.23120)  

**Abstract**: Pnar, an Austroasiatic language spoken by approximately 0.4 million people in the Jaintia Hills of Meghalaya, lacks the digital corpora and natural language processing (NLP) resources. This paper presents the first machine translation study for the English and Pnar language pair. Using articles collected from the Wyrta newspaper, we built a parallel corpus comprising of 10,234 sentences and trained phrase-based statistical machine translation (SMT) systems the models using 9,563 parallel corpora under three configurations for each direction using Moses, GIZA++ , KenLM, varying lexicalized reordering and minimum error rate training (MERT) tuning. The models are evaluated on a held out test set of 371 sentences, the best performing system achieves a BLEU score of 14.97 (chrF2: 33.42, TER: 77.60) for Pnar to English and 11.16 (chrF2: 31.38, TER: 93.51) for English to Pnar, establishing the first quantitative benchmark for this language pair. Lexicalized reordering improves translation quality by 3.73 BLEU points for Pnar to English, reflecting the structural shift from the source language's SOV word order to the target language's SVO order, whereas MERT tuning degrades BLEU performance under low resource conditions. Finally, we analyze the remaining translation errors, including morphological out of vocabulary (OOV) words, long-distance reordering and Khasi code mixing and discuss future directions toward neural and multilingual machine translation for Pnar. 

---
# DeMixPert: Decomposed Response Modeling with Gaussian Mixtures for OOD Single-Cell Perturbation Prediction 

**Authors**: Jiawen Liu, Xuechenxiao Cao, Yutong Li, Bing Liu, Jiaming Liang, Tinghe Zhang, Xiaoqi Sheng, Hongmin Cai  

**Link**: [PDF](https://arxiv.org/pdf/2608.23114)  

**Abstract**: Predicting transcriptome-wide responses to unseen genetic perturbations remains a major computational challenge because accurate prediction requires recovering both perturbation-specific transcriptional shifts and heterogeneous cellular responses. Existing methods often entangle deterministic response structure with stochastic population-level variation, causing dominant shared patterns to mask weaker perturbation-specific signals and impair distributional modeling. To address these challenges, we propose \textbf{DeMixPert}, an approach for Decomposed response Modeling with Gaussian Mixtures for Out-Of-Distribution (OOD) single-cell Perturbation prediction. DeMixPert decomposes perturbation-induced changes into a basal-state-dependent systematic response, a perturbation-specific response, and population-level variation. The systematic component is derived from the basal state encoded from control-cell expression, whereas the perturbation-specific component is inferred from pretrained target embeddings for unseen-target generalization. DeMixPert models population-level variation using a Gaussian prototype Invertible Network and adaptively combines reusable Gaussian prototypes according to the basal state and perturbation condition. The resulting mixture is mapped to a condition-specific variation distribution. Sampled variations are integrated with the systematic and perturbation-specific components, followed by joint decoding with the basal state to reconstruct perturbed-cell gene expression. Experimental results show that DeMixPert effectively captures heterogeneous single-cell perturbation responses and achieves superior performance across unseen-perturbation settings. The source code is made publicly available upon publication. 

---
# Molecular LLM Agents: From Architectural Design to Scientific Autonomy 

**Authors**: Jiatong Li, Wengyu Zhang, Weida Wang, Yuxuan Ren, Wei Liu, Chenyang Mao, Yuqiang Li, Yatao Bian, Changmeng Zheng, Xiaoyong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.23104)  

**Abstract**: Molecular science represents an important frontier for LLM-based agents. Unlike general agents that mainly operate over natural language, code, or web environments, molecular LLM agents must perceive, reason about, and act upon chemical objects across symbolic strings, molecular graphs, 3D conformations, spectra, simulations, and wet-lab measurements. Their capabilities depend on chemically faithful molecular perception, an LLM-centered agent framework, domain-specific tool grounding, and computational or experimental feedback, in addition to planning and tool use. This work develops a conceptual framework for molecular LLM agents from two complementary perspectives. First, we introduce an architectural view of molecular-agent design, covering molecular representation and perception, the agent framework, domain-specific toolboxes, and learning and optimization. Second, we propose a scientific autonomy ladder inspired by staged autonomy in engineering systems, categorizing agents into four levels: L1 assistive or fixed workflows, L2 adaptive computational agents, L3 feedback-aware physical experiment agents, and L4 scientific-agenda agents. Together, these two perspectives establish a comprehensive framework for comparing existing molecular LLM agents, identifying missing capabilities and deployment risks, and guiding the design, evaluation, and deployment of future agents in molecular discovery workflows. 

---
# PolyChirp: Multi-Species Birdsong Classification Using TinyML on Low-Power Acoustic Sensors 

**Authors**: Nathan Duboisset, Zhaolan Huang, Felix Bießmann, Roudy Dagher, Antoine Lavandier, Emmanuel Baccelli  

**Link**: [PDF](https://arxiv.org/pdf/2608.23101)  

**Abstract**: Recent progress in the field of TinyML has demonstrated that low-power hardware based on microcontrollers can achieve bird species monitoring in real time based on acoustic sensor data for an entire breeding period on a single battery charge. However, the state of the art on low-power microcontrollers was so far limited to binary classification of a single species. In contrast, real fauna monitoring deployments often target multiple species simultaneously. To address this challenge we develop PolyChirp, an approach combining biological domain expertise, automated dataset curation, neural architecture optimization and novel hardware to achieve multiclass bird species detection in the wild. PolyChirp is based on newly designed tiny multiclass models that leverage recent microcontrollers and hardware acceleration with a neural processing unit (NPU). We evaluate the predictive performance of these models, and we measure their computational performance -- memory footprint, latency, energy consumption -- on common microcontroller hardware. Our results demonstrate that PolyChirp not only outperforms state-of-the-art on single species binary classification, but also achieves robust classification of up to 10 species simultaneously, while still fitting with the resource envelope of a sensor that must remain operational in the field for a full season on a single battery charge. 

---
# Shaping the Evolutionary Dynamics of Robot Morphology via Adaptive Control Learning 

**Authors**: Junru Song, Yang Yang, Yaqing Xu, Ying Wen, Wei Peng, Guozhen Li, Wei'en Zhou, Wen Yao  

**Link**: [PDF](https://arxiv.org/pdf/2608.23100)  

**Abstract**: Robot co-design via bi-level optimization couples within-lifetime controller learning for fitness evaluation with cross-generational morphological evolution. Prior work has established that well-adapted morphology facilitates faster control learning, a property termed morphological intelligence. Yet how control learning reciprocally shapes morphological evolution remains unexplored. This paper examines both directions for a holistic account of brain-body interplay. We first show that morphological contributions to control learning decouple into two orthogonal dimensions. We formalize the convergence speed as morphological intelligence and identify the performance ceiling as a complementary quantity termed true potential. A concise functional relation is then established to jointly characterize both quantities from individual learning curves, which, when aggregated at the population level, capture evolutionary profiles. Through extensive experiments on simulated voxel-based soft robots, we reveal that premature fitness evaluation systematically underestimates true potential and biases selection towards fast learners. This restricts design space exploration, compromising both optimization efficiency and morphological diversity. Notably, the widely recognized morphological Baldwin effect emerges as an artifact of this bias rather than a general evolutionary tendency. We therefore propose AdaControl, which monitors disproportionate selection for morphological intelligence during evolution and allocates minimally sufficient control learning for unbiased fitness evaluation. With AdaControl, a simple genetic algorithm rivals state-of-the-art generative-model-based co-design methods in discovering diverse high-performing designs while cutting computation by up to 80% versus exhaustive control. 

---
# Cultural Moment Benchmark: Evaluating Video Cultural Reasoning and Grounding in Southeast Asia 

**Authors**: Burak Satar, Zhixin Ma, Cheng Yu-Tong, Huy Hoang Tran, Phuong Anh Nguyen, Chong-Wah Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2608.23065)  

**Abstract**: Cultural understanding in video means more than recognizing what is visible; it requires grasping the symbolic and temporal significance of cultural concepts. We decompose this into three abilities: naming what a concept symbolizes, visually recognizing it on video, and locating its sub-events in time. Existing video-cultural benchmarks tend to test what is seen, collapsing these three abilities into a single score that hides the bottleneck. We introduce the Cultural Moment Benchmark (CMB): 306 expert-curated concepts from seven countries in Southeast Asia across five categories. We evaluate each concept through three stages, one per ability. Given a description, Stage 1 (S1) selects from four candidate concept names, Stage 2 (S2) selects from four candidate video moments, and Stage 3 (S3) predicts the start and end times of the moment in a video. To keep each stage focused on a distinct ability, we use three design choices: semantic-similarity distractors (S1, S2), unlabeled video moments (S2), and free-form localization on a different example video (S3). Across six vision-language models, failure modes vary by ability and modality. i) Even the strongest closed-source models score below 30% when all three stages must be correct; ii) The three abilities do not fully cascade: naming a concept correctly helps half the models recognize it on video, but recognizing it has little effect on locating the sub-event in time; iii) Audio is complementary, redundant, or distracting depending on the concept, more often distracting in non-Latin-script countries; removing both audio and subtitles hurts Games and Music the most. Our 14-rater human study shows that even Expert raters score below chance on concepts from a neighboring country, indicating that CMB requires country-specific cultural knowledge. CMB acts as a diagnostic harness, attributing failures to a specific ability or modality. 

---
# Beyond Verdicts: A Graph-Based Analysis of Human and LLM Reasoning in Scientific Fact-Checking 

**Authors**: Abdul Ghafoor, Muhammad Arslan Manzoor, Yufang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2608.23047)  

**Abstract**: Misinformation that cites legitimate papers can be especially harmful when it distorts what those studies actually report. While existing automatic fact-checking systems based on large language models (LLMs) can assess whether a model assigns an Incorrect verdict and can gen- erate explanations for that decision, they typi- cally do not indicate whether the model follows the same reasoning path as human experts or arrives at the verdict through a different but still valid path. In this work, we introduce a graph- based framework (typed reasoning graph) for comparing human and LLM reasoning paths in scientific fact-checking. Building on prior work on fallacious reasoning in biomedical misinformation, MISSCIPLUS (Glockner et al., 2025), we model each explanation as a rea- soning graph that links the false claim to the relevant study context, study findings, fallacy- supporting premises, and fallacy labels. This representation enables one-to-one alignment of human and LLM reasoning at the level of fallacy-specific sub-graphs. For non-human- aligned LLM paths, we validate grounding in the cited study, relevance to the claim, and suf- ficiency for the verdict. Using 84 false claims from MISSCIPLUS, we evaluate GPT-5, Claude Opus 4.7, and Qwen3-32B across prompt and evidence settings. Results show distinct perfor- mance dimensions: Qwen3-32B has the lowest verdict failure rate, GPT-5 the highest human alignment, and Claude Opus 4.7 weak verdict prediction but often valid reasoning in success- ful cases 

---
# The Multilingual FrameNet Corpus 

**Authors**: Beatrice Fiumanò, Nicolas Lazzari, Simone Paolo Ponzetto, Valentina Presutti  

**Link**: [PDF](https://arxiv.org/pdf/2608.23037)  

**Abstract**: This paper introduces the Multilingual FrameNet Corpus (mFNC), a novel resource that extends the English Berkeley FrameNet corpus by collecting and harmonizing existing language-specific corpora across nine additional languages: Brazilian Portuguese, Chinese, Dutch, French, German, Italian, Korean, Latvian and Swedish. By training models that rely on different architectures on the mFNC, we consistently outperform existing state-of-the-art Frame Semantic Parsers in both multilingual and cross-lingual settings, underscoring the importance of multilingual training data. The mFNC and our trained FSP models are openly available at this https URL. 

---
# FedCC: Towards Addressing Label Distribution Skews in Distillation-Based Federated Learning 

**Authors**: Wenxuan Ye, Onur Ayan, Xueli An, Georg Carle  

**Link**: [PDF](https://arxiv.org/pdf/2608.23031)  

**Abstract**: Federated Learning (FL) enables distributed clients to collaboratively train models without sharing raw data, making it promising for leveraging massive devices in communication networks. In distillation-based FL, each client applies its local model on an unlabeled public dataset, and shares only prediction results with the server. While heterogeneous local data introduces label distribution skew, thus biasing client models toward majority classes and leading to potentially inaccurate predictions. The lack of ground-truth labels in the public dataset hampers the server's ability to calibrate predictions, which ultimately degrades overall performance. To address this, we propose FedCC, a simple and effective algorithm for mitigating client misclassification. Instead of being forced to classify and risking error propagation, clients are allowed to tag ambiguous samples as 'unknown'. This additional class, together with calibrated pseudo-labels on the public data, balances confidence in majority classes against uncertainty in under-represented ones. Extensive experiments demonstrate that FedCC significantly outperforms existing methods, especially under severe label skew. In the extreme scenario where each client holds samples from only one of ten classes, FedCC achieves 67.3% accuracy, while baselines collapse to near-random results. 

---
# Beyond Surface Cues: Disentangling Sociocultural Signals in Multilingual LLMs 

**Authors**: Yuanjun Feng, Tanzhou Liu, Stefan Feuerriegel, Yash Raj Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2608.23026)  

**Abstract**: Multilingual LLM outputs can vary across sociocultural contexts. However, evidence of cultural grounding can be misleading: identity labels may be inferred from explicit or indirect textual cues, while names and wording can reveal the source language. Treating all these signals as evidence of cultural grounding may obscure potential biases. We present a human-validated, multi-agent audit that separates three questions: whether outputs reproduce social biases, whether identity groups are represented differently, and whether outputs reflect cross-cultural patterns. The study analyzes 89,253 outputs from 12 LLMs in English, French, and Chinese, spanning 18 occupations and three task conditions.
We find that bias representation varies systematically across languages and tasks. Removing direct identity cues sharply reduces identity-label prediction in English and Chinese, but has a much smaller effect in French. Across all language-genre settings, the cultural context associated with the source language receives the highest average relevance score, with moderate agreement between automated and human ratings. However, the ability to identify the source language drops substantially after translation and again after masking names. Without these controls, multilingual audits may mistake surface cues for cultural understanding, leading to misleading conclusions about cross-cultural variation and bias. Our audit offers a practical framework for separating such shortcuts from more meaningful cross-cultural patterns. 

---
# Unlearning Is Not Just Erasing: Temporal Decoupling via Generation Inequality 

**Authors**: Xunlei Chen, Qirui Ye, Yuang Li, Yi Gong, Zhaokun Wang, Wenyi Li, Shiyao Guo, Jinyu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.23020)  

**Abstract**: Large language models (LLMs) require effective unlearning to address privacy regulations and safety concerns. However, achieving precise forgetting without compromising general utility remains challenging. Existing sequence- and token-level methods penalize target outputs without modeling their context-dependent retrieval paths, which can disrupt linguistic structure or suppress benign knowledge. We present ADU, a fine-grained, training-based framework that shifts unlearning from token erasure to contextual attention-pathway decoupling. Exploiting the functional distinction between local and global attention heads, ADU identifies preplan positions that retrieve persistent sensitive anchors and fixes their candidate paths under the original model. It then trains attention-projection adapters to suppress attention mass along these paths while preserving local-attention structure and retain-set language modeling. Post-training activation exchange tests whether the modified attention-output module transmits the learned forgetting effect. ADU achieves the strongest aggregate performance among evaluated baselines on the TOFU and WMDP benchmarks, including a Forget Quality of (0.93) on TOFU. It preserves 87--98% of model utility (92.9% on average versus 81.9% for baselines) while reducing side effects in benign contexts. 

---
# SplitLite: Low-Rank Residual Compression for Split Learning 

**Authors**: Tao Li, Yulin Tang, Qi Guo, Xianhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.23018)  

**Abstract**: Federated fine-tuning of on-device large language models (LLMs) faces a significant computing burden. To overcome this limitation, split learning (SL) has emerged as a promising solution, which offloads the primary training workload to a powerful server. However, SL requires exchanging high-dimensional activations and gradients between clients and the server, resulting in prohibitive communication costs. To overcome this challenge, we propose SplitLite, a communication-efficient split federated LoRA fine-tuning method that exploits the low effective rank structure of consecutive-epoch activation and gradient residuals. Our key finding is that, when LoRA uses rank $r$ updates in parameter space, the activation and gradient residuals of the same data sample between adjacent epochs also exhibit effective rank-$2r$ and rank-$4r$ structures, respectively. By revealing this property, SplitLite transmits only quantized truncated singular value decomposition (SVD) residual factors, thereby significantly reducing both activation uplink and gradient downlink traffic. Extensive experiments on the GLUE benchmark across a series of advanced on-device LLMs demonstrate that our method reduces activation uplink communication costs by up to 93.5\% and total communication costs by up to 83.7\%, without performance degradation. 

---
# Coarse Indexing, Fine Evidence: Decoupling Temporal Granularity in Long-Video RAG 

**Authors**: Zhe Jin, Zhimin Lin, Bin Zheng, Junhua Fang, Huihua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23011)  

**Abstract**: Graph-based retrieval-augmented generation (RAG) provides a scalable paradigm for long-video understanding, but existing systems typically inherit a fixed temporal granularity from video segmentation when constructing their retrieval index. We argue that this design unnecessarily couples indexing granularity with evidence granularity: coarse representations can often suffice for locating relevant temporal regions, while fine-grained evidence remains important for downstream reasoning. We propose \textbf{Density-Aware Graph Construction (DAGC)}, a training-free approach that decouples a query-independent coarse retrieval index from the original fine-grained evidence space. DAGC constructs a compact, density-adaptive graph index by merging visually redundant neighboring chunks, while preserving mappings to the original temporal units. Retrieved coarse regions are subsequently expanded back to the original chunk granularity for fine-grained evidence refinement and answer generation. Experiments on MLVU, VideoMME, and LongVideoBench show that DAGC retains only about 40--50\% of the original graph nodes and achieves $1.3$--$1.7\times$ end-to-end wall-clock acceleration while preserving approximately 99\% of the original QA performance. The gains transfer across different LVLM backbones and video RAG pipelines, suggesting that long-video RAG need not maintain the same temporal granularity for indexing and evidence reasoning. 

---
# A Physical Response-and-Memory Model for Muon Optimization 

**Authors**: Yinze Hu, Hongjun Xiang, Xingao Gong, Hongyu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.22994)  

**Abstract**: Training large language models is costly. How low a loss the same compute can ultimately reach depends on how each step's gradient is converted into a weight update; the rule that performs this conversion is the optimizer. From SGD and AdamW to the recent Muon, effective update rules have mostly been shaped by engineering intuition and then selected on benchmarks. Muon semi-orthogonalizes the momentum matrix before applying the update and has kept breaking records on public training benchmarks; yet why the semi-orthogonalized direction works, and over how long a history the momentum should average, are two questions at present answered mainly by experience. Here we treat the weight matrix during training as a responsive medium with memory and build a physical model for it, in which both questions find answers: the semi-orthogonalized direction is the maximally dissipative response under an output-side safety budget, which explains why it works; momentum is the internal stress accumulated by the medium; how long it should average is set by the relaxation of this stress, and a real medium relaxes on more than one timescale, the simplest form being one fast and one slow. On this basis we propose the Bi-Maxwell optimizer. The framework further yields a testable consequence: gradient directions change fast early in training and more slowly later, so the optimal memory length should grow with training stage; step-by-step measurements of a proxy for it by a read-only probe across 8 independent training trajectories are consistent with this consequence. Replacing the memory kernel alone, from a single timescale to two, brings training to the target loss in noticeably fewer steps on a public large-language-model optimizer benchmark. 

---
# Hypergraph Embedding Indexing for Efficient Dense Vector Retrieval 

**Authors**: Kishore Konda  

**Link**: [PDF](https://arxiv.org/pdf/2608.22980)  

**Abstract**: Dense vector retrieval has become the foundation of modern semantic search, yet existing approximate nearest neighbor (ANN) indexes treat an embedding as an indivisible point in a high-dimensional space. In this work, we propose the Hypergraph Embedding Index (HEI), a framework that instead organizes documents according to combinations of highly activated latent embedding dimensions. This formulation enables inverted-index style candidate generation while preserving the semantic ranking capabilities of dense embeddings. We further demonstrate that constructing multiple complementary hypergraphs substantially improves retrieval coverage without the combinatorial growth associated with increasing the dimensionality of a single hypergraph. Finally, we establish that the statistical properties of embedding activations strongly influence coordinate-inverted indexing efficiency, introducing \emph{activation diversity} as a diagnostic metric governing embedding indexability in coordinate-inverted frameworks. 

---
# WildHandBench: A Benchmark for Handwritten Text Understanding that Challenges MLLMs and Humans 

**Authors**: Jun Zhang, Qiao Zhao, Cheng Cui, Jianying Qu, Zhongkai Sun, Jianwen Yang, Changda Zhou, ZhuoXin Liu, Shubin Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.22959)  

**Abstract**: While the top model on OmniDocBench now reaches 96.34% overall on printed-document parsing, the ability of current models to handle challenging handwritten documents remains largely uncharacterized. Existing benchmarks focus on isolated text or formulas, overlook handwritten tables and real-world degradation, and report aggregate accuracy without explaining why models fail.
We present WildHandBench, a benchmark containing 500 handwritten documents across three structures (free text, tables, formulas), four languages, and nine real-world scenarios. We introduce a Prior-Driven Error (PDE) metric that quantifies whether errors originate from language priors rather than visual evidence. Evaluating 18 state-of-the-art models together with calibrated human baselines, we find: (1) the best model achieves only 71.85% overall; (2) humans outperform all models yet the gap is narrow (77.09% vs. 71.85%); and (3) model errors are qualitatively different from human errors -- 63-91% of model errors are prior-driven versus only 49% for humans, exposing systematic reliance on language priors that conventional accuracy metrics cannot capture. 

---
# What Proves You Wrong: Benchmarking Language Models on Falsifiable Research Ideation 

**Authors**: Ziyue Wang, Aomufei Yuan, Yiran Yao, Linli Yao, Hongyao Zuo, Ziwen Gong, Yuanxin Liu, Shicheng Li, Yishuo Cai, Tong Yang, Xu Sun, Xiaohui Li, Haoli Bai  

**Link**: [PDF](https://arxiv.org/pdf/2608.22948)  

**Abstract**: Large language models are increasingly used to propose research ideas, yet the prevailing ways of judging such ideas supply no shared decision rule: free-form judging sways with style and position, and scoring against a later paper rewards recovery of one realized trajectory. We introduce a benchmark that carries a proposal from Literature to Test: the Lit2Test benchmark centers on a six-field contract organized around a falsifying outcome, so that every proposal precommits the observation that would prove it wrong, making its quality decidable in the first place rather than merely arguable. Built prospectively from 200 real-paper neighborhoods, Lit2Test elicits proposals from four frontier models and compares them through 1,200 pairwise comparisons judged blind in both presentation orders. The protocol audits its own reliability through diagnostic controls and bounded human calibration, with three annotators corroborating the conclusions within explicitly stated reliability bounds. Lit2Test recovers a strict ranking of the four models in all 10,000 bootstrap replicates, and the separation comes from the quality of the proposed tests and metrics rather than from surface fluency. We release the benchmark, construction pipeline, and audit artifacts for public use. 

---
# Deep Learning-Based Multi-User Communication Design for Dense IoT Networks: Interference-Aware Finite-Blocklength Communication and Preliminary MIMO Extensions 

**Authors**: Arkadeep Sinha, Shubham Paul, R. Manivasakan  

**Link**: [PDF](https://arxiv.org/pdf/2608.22923)  

**Abstract**: Dense IoT networks require reliable communication despite limited spectrum and substantial multi-user interference while maintaining manageable receiver complexity. This work introduces a deep-learning-based end-to-end multi-user communication design for interference-limited finite-blocklength IoT scenarios, focusing on short and medium blocklengths.
We extend a prior 2-user SiameseNet transceiver framework to accommodate 2, 4, and 8 users, leveraging learned redundancy for interference suppression and noise robustness. Compared to conventional non-orthogonal access baselines, our method demonstrates strong Block Error Rate (BLER) performance across various scenarios without resorting to joint detection; the per-user decoder scales roughly linearly with the number of users.
Further, we examine the robustness under interference mismatch and unequal interference strengths, critical for practical deployments with heterogeneous devices. The Latent-space analysis reveals that the learned codeword distance increases as the effective per-user rate decreases, corroborating with the observed BLER improvements. In addition, we also present preliminary results for a 2X2 MIMO setup under fixed-channel CSIT and CSIR, indicating potential for extending the framework to IoT gateways with multiple antennas. 

---
# Safety Hacking in Constrained Best-of-$N$ Inference-time Scaling 

**Authors**: Akifumi Wachi, Takumi Tanabe, Youhei Akimoto  

**Link**: [PDF](https://arxiv.org/pdf/2608.22915)  

**Abstract**: Inference-time pipelines often sample multiple outputs, filter them with a learned safety model, and return the proxy-feasible output with the highest learned reward. We show that this composition creates a two-stage failure: an imperfect safety proxy first contaminates the feasible set with unsafe outputs, and reward maximization can then amplify this residual contamination. We define \emph{safety hacking} as selecting an output that passes the learned constraint but violates the true safety criterion. For constrained Best-of-$N$ sampling, we derive finite-$N$ bounds governed by the joint upper reward tails of safe and unsafe outputs within the proxy-feasible set. If unsafe-but-feasible outputs have the heavier tail, safety hacking becomes asymptotically certain as $N$ grows, even when false-positive mass and average safety- and reward-proxy errors are arbitrarily small. We also show that policies within a bounded $\chi^2$ divergence from the proxy-feasible reference distribution admit an $N$-independent safety-hacking bound, and instantiate this general coverage-control principle with constrained pessimistic sampling. Coverage control limits amplification but cannot repair a contaminated feasible set: admitted unsafe outputs may still be favored, and regularized selection is not necessarily safer than constrained Best-of-$N$ for every reward proxy. Toy and language-model experiments characterize both contamination and its reward-tail amplification, which exposes an inherent difficulty in inference-time scaling with learned safety models. 

---
# Do Spoken Language Models Hear Speech as They Read Text? Bridging Structural Gaps Between Speech and Text 

**Authors**: Hyeonyu Kim, Hwayeon Kim, Youngwon Choi, Myeongkyun Cho, Huu-Kim Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2608.22908)  

**Abstract**: Spoken Language Models (SLMs) generate textual responses directly from speech, offering an alternative to cascaded systems. Despite recent advances, existing SLMs still exhibit weaker instruction-following behavior and limited generalization across diverse tasks compared to text-based language models. Our analysis shows that speech and text representations in current SLMs remain weakly aligned despite strong downstream performance, indicating that structural differences between continuous, temporally varying speech and discrete text remain insufficiently addressed. To address this, we propose a simple framework that decouples length mismatch from semantic alignment and encourages closer correspondence between speech and text representations. Experiments across multiple benchmarks demonstrate competitive performance against strong baselines, underscoring the importance of explicitly addressing structural differences between speech and text in SLM training. Our code is publicly available at this https URL. 

---
# AraDetox: A Multi-Dialect Arabic Detoxification Dataset 

**Authors**: Mo El-Haj  

**Link**: [PDF](https://arxiv.org/pdf/2608.22894)  

**Abstract**: Arabic harmful-language detection has received considerable attention, yet Arabic text detoxification remains underexplored. We introduce AraDetox, a multi-dialect Arabic detoxification dataset comprising 10,500 harmful social-media posts and 84,000 detoxified rewrites generated using GPT-5 and Gemini 2.5 Flash across Modern Standard Arabic, Gulf, Levantine, and Egyptian Arabic. The generated outputs were assessed through human evaluation and automatic analyses of lexical change, semantic preservation, sentiment, and dialectal style. Results show that detoxification is primarily a meaning-preserving rewriting task: substantial lexical and structural reformulation is accompanied by consistently high semantic similarity. Human evaluation confirms successful harmful-language removal while largely preserving the original meaning. Dialectal analyses further indicate that the generated variants exhibit measurable stylistic alignment with reference Arabic dialect corpora. Comparison with existing resources highlights two complementary approaches to detoxification: minimal-edit lexical substitution and meaning-preserving reformulation. Our findings demonstrate that large-scale Arabic detoxification resources can be constructed through LLM-assisted generation and human verification. The dataset is publicly available at this https URL to support future research on Arabic detoxification, safe text generation, and multi-dialect Arabic NLP. 

---
# The Mask Is Not the Model: Auditing Prefix Invariance in Attention, State-Space, and Hybrid Sequence Models 

**Authors**: Taebong Kim, Youngsik Hong, Minsik Kim, Sunyoung Choi, Jaewon Jang, Minseo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.22876)  

**Abstract**: We formalize prefix invariance: representations at position t must not depend on future inputs. We give a lightweight audit, two forward passes, no training or gradients, that localizes exactly where causality breaks. Attention-mask inspection is incomplete: leaks can occur via scans or normalization despite correct masks. Across 192 injected-fault trials on eight checkpoints, mask inspection found none, while our audit localized all 192/192, also finding a defect in Zamba2 and Nemotron-H. 

---
# Hierarchy-Aware Supervised Uncertainty Estimation for Black-box LLM Taxonomic Reasoning 

**Authors**: Shuting Xie, Nathaniel Lesperance, Graham W. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2608.22839)  

**Abstract**: Large language models (LLMs) are increasingly used for scientific decision support, yet reliable confidence estimation remains difficult in black-box settings. We study uncertainty estimation for hierarchical taxonomic reasoning generated by a black-box LLM in a long-tailed biodiversity monitoring pipeline. Using proxy features extracted by an open-source tool LLM, we train lightweight supervised estimators with hierarchy-aware supervision to predict rank-wise correctness. Across three tool LLMs, the supervised estimators consistently outperform a token-likelihood baseline for micro discrimination and selective prediction under a single global rejection threshold, improving micro AUROC from 0.57 to 0.75--0.80. The best results are achieved by a rank-specific multi-head design (H3), suggesting that accounting for hierarchical output structure is important when a unified abstention rule is required. Our code is publicly available at this https URL 

---
# Minimal Local Simulation Foundations for LLM- and VLM-Driven Agents in 2D and 3D Environments 

**Authors**: Ryuki Hyodo  

**Link**: [PDF](https://arxiv.org/pdf/2608.22833)  

**Abstract**: Large language models (LLMs) and vision-language models (VLMs) are expanding the range of behaviors that can be represented in agent-based simulations, but many contemporary platforms are difficult to study, modify, or run on ordinary computers. We present two intentionally minimal simulation foundations for education and rapid prototyping. SD-AgentFoundry-2D provides a two-dimensional multi-agent environment in which locally hosted LLM agents move, communicate, respond to place occupancy, and encounter spatially localized fire events. SD-AgentFoundry-3D provides a three-dimensional digital-twin environment in which a locally hosted VLM receives first-person images and produces natural-language movement instructions. Both codebases are designed to run locally on macOS, Windows, and Linux and are deliberately left open to modification rather than developed as finished applications. Together, they offer accessible starting points for learning about generative social simulation and for building domain-specific extensions. 

---
# Fairness-Aware Mixture-of-Experts via Subgroup Reweighting and Gate Regularization 

**Authors**: Sunhee Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22820)  

**Abstract**: Deep learning models often produce performance disparities across demographic groups, due to the training data imbalance with respect to sensitive attributes such as gender or age. To address this problem, existing work has explored fair representation learning, data re-sampling, and adversarial training, which can be broadly categorized into two main approaches. Single-stage methods typically learn a shared representation for fairness, but often struggle to handle heterogeneous subgroup distributions. Two-stage methods learn representations separately from the final prediction task, which can lead to misalignment between fairness objectives and downstream predictions. We identify routing-induced bias, a failure mode in which subgroup imbalance drives the gating network to route subgroups onto a few experts, and propose an end-to-end Mixture-of-Experts (MoE) framework that corrects it. Specifically, we apply subgroup reweighting to correct data imbalance, and introduce gate entropy regularization to prevent routing from collapsing onto subgroup attributes, keeping expert utilization both balanced and interpretable. Beyond improving fairness, the routing distribution offers an interpretable view of how subgroups are allocated across experts. Experimental results demonstrate that the proposed approach improves fairness while maintaining competitive predictive performance. 

---
# SDoH-Aware Narrative Anchoring Bias in Medical LLMs for Trustworthy Clinical Decision Support 

**Authors**: Ahnaf Atef Choudhury, Ramkrishna Saha  

**Link**: [PDF](https://arxiv.org/pdf/2608.22802)  

**Abstract**: Medical large language models are often judged by how many clinical questions they answer correctly. That view is useful, but it misses a practical risk. A model may know the right answer and still change its response when the same case is written in a different patient voice. This paper evaluates that risk as SDoH aware narrative anchoring bias. We use NarrativeShield SDoH MedQA, a counterfactual medical question answering dataset in which each case appears in persona based narratives while the answer key remains fixed. The dataset is reshaped from wide format into case grouped persona rows. We evaluate three open source instruction tuned LLMs from the Qwen2.5 family: 1.5B, 3B, and 7B. The final experiment uses 300 clinical cases and produces 8,100 model responses across three prompting conditions. We report persona level accuracy, counterfactual consistency, correct consistency, and narrative sensitivity error. Qwen2.5 7B achieves the best accuracy at 56.33 percent and the best correct consistency at 40.33 percent. Paired McNemar exact tests show significant accuracy gains for 7B over 3B in all prompt settings. Even so, narrative sensitivity remains, with the lowest error still at 31.67 percent. These results suggest that trustworthy clinical decision support should be evaluated by both average correctness and stability across medically equivalent patient narratives. 

---
# Triplet2Track: A Hierarchical System with Object-Centric Representations for Reliable Long-Horizon Manipulation 

**Authors**: Jianxiang Liu, Gaojing Zhang, Chuan Wen, Qipeng Liu, Yuxuan Zhao, Ning Guo, Wenzhao Lian  

**Link**: [PDF](https://arxiv.org/pdf/2608.22800)  

**Abstract**: Ensuring reliability in uncertain environments remains difficult for long-horizon robotic manipulation. End-to-end VLA models are data-heavy and opaque, making diagnosis and verification difficult. Hierarchical pipelines are more interpretable, but their plans are often weakly grounded in observations, weakly aligned with low-level actions, and computed without online feedback, leading to open-loop behavior and hallucinations. To address these issues, we introduce the Triplet-to-Track System (TTS), a closed-loop long-horizon imitation learning system that uses human videos to reduce reliance on robot-collected data. TTS represents high-level subgoals as instance-grounded triplets, translates them into continuous track priors for execution, and monitors task progress from observations for online replanning. Across diverse real-world long-horizon tasks, TTS achieves a 74.8\% average success rate and supports object-level and compositional generalization. 

---
# TRACE: A Self-Evolving Skill Bank for Consistent, Limit-Aware LLM Agents 

**Authors**: Wenhao Wu, Menghao Zhang, Xin Wang, Zhi Wang, Kun Shao, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2608.22793)  

**Abstract**: Reliable deployment of LLM agents in user-facing products depends not on raw task-solving ability but on consistency and limit-awareness: behaving the same way across repeated trials, and recognizing when a request cannot, or cannot yet, be safely fulfilled. CAR-bench exposes this reliability gap in the domain of in-car assistants: an LLM-simulated user issues incomplete or ambiguous requests, requiring the agent to resolve uncertainty through multi-turn dialogue and tool use while strictly adhering to domain policies. Even frontier models show a substantial gap between what they can solve at least once (Pass@3) and what they solve consistently across trials (Pass^k). We bridge this gap with TRACE (TRAjectory-Contrastive Evolution), which iteratively improves a skill-based agent's behavioral knowledge without modifying model weights. This knowledge is organized as a Skill Bank of modular, retrievable skills, each encoding a self-contained set of tool-use rules and behavioral guidelines. TRACE evolves this bank through an agentic self-evolution loop: after each evaluation round, it groups trajectories by the skills invoked and refines each skill by contrasting successful and failed behaviors. The updated bank then guides subsequent rounds, while during deployment the Actor performs state-conditioned skill orchestration at every turn. On GPT-5.5, TRACE improves consistency (Pass^3) by 34.6 points, from 59.9% to 94.5%, while shrinking the gap between potential and reliable performance to just 4.0 points. On the official hidden set, TRACE achieved first place using GPT-5.6-Sol, attaining a Pass^3 score of 70%-a 40% relative improvement over the baseline. These results show that TRACE converts high model potential into stable, consistent performance gain. Project homepage: this https URL. 

---
# Don't Repeat Yourself: Stopping Verbatim Loops at Sampling Time 

**Authors**: Philipp Emanuel Weidmann, Allen Roush, Judah Goldfeder, Sanjay Basu, Ravid Shwartz-Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2608.22761)  

**Abstract**: Large Language Models generate text autoregressively, but open-ended generation is prone to verbatim looping, in which models repeat spans already present in context. Standard defenses such as repetition, presence, and frequency penalties and n-gram blocking act on token recurrence rather than the sequential structure of a loop, and often suppress looping only at strengths that also degrade formatting or fluency. We propose Don't Repeat Yourself (DRY), a sampling-time logit adjustment that penalizes a candidate token only when generating it would extend the current suffix into an exact continuation of a span seen earlier in the context. Sequence breakers protect chat templates and formatting tokens. Across models from 1.5B to 120B parameters, nine prompt families, and a 600-pair human study, DRY reduces suffix-extension rate by 47% while improving lexical diversity. An intervention-matched placebo produces no comparable reduction, identifying suffix matching as the operative mechanism. On AWQ-quantized 70B and 120B models, DRY reduces loop rate by roughly half while preserving MT-Bench, MMLU, and GSM8K performance, whereas standard alternatives lose measurable ground. DRY has been adopted by popular open-source LLM inference frameworks including this http URL, ExLlamaV2, and text-generation-webui, highlighting its practical impact on text generation. 

---
# XTC: Head-Aware Sampling by Excluding Top Choices 

**Authors**: Philipp Emanuel Weidmann, Allen Roush, Judah Goldfeder, Sanjay Basu, Ravid Shwartz-Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2608.22758)  

**Abstract**: Standard decoding rules for autoregressive language models promote diversity by rescaling the full next-token distribution or truncating its low-probability tail. These strategies overlook a common regime of open-ended generation in which several continuations are plausible but too much probability mass remains concentrated on the most generic choice. We introduce XTC (Exclude Top Choices), a lightweight head-aware decoding operator that targets this regime directly. XTC identifies tokens whose probabilities exceed an absolute plausibility threshold $\tau$: when at least two qualify, it removes the dominant eligible choices with probability $\rho$ and retains only the weakest plausible alternative before renormalization. Across 60 experiments on Gemma 3 27B Q4, Gemma 3 12B Q6, and DeepSeek R1 14B Q6, with scaling validation on Llama 3.3 70B Q4, XTC improves the diversity-repetition Pareto frontier. On creative generation, Distinct-2 increases by 11--15% and repeat trigrams decrease by 27--47% across the four models. Combined with temperature scaling, gains reach 38% in Distinct-2 and 71% in repeat-trigram reduction over baseline. A blinded Amazon Mechanical Turk study with 150 Master raters yields a 62.3% creativity preference for XTC ($p<10^{-4}$) without reduced fluency, while a GPT-4o control judge reproduces the Anthropic-judge direction on every measure. On IFEval with Llama 3.3 70B Q4, XTC preserves prompt-level strict accuracy within 1.7 percentage points of baseline while recovering most of the diversity gain; a temperature setting matched on Distinct-2 reduces IFEval by 8.8 points. The effect is additive with temperature and repetition penalties, robust across quantization levels and model families, and consistent across twelve prompt genres. XTC has been adopted by this http URL, ExLlamaV2, and text-generation-webui. 

---
# Object-Uni: A Unified Model for Object-Centric Spatial Understanding and Controllable Generation 

**Authors**: Mining Tan, Yinuo Wang, Ziqi Zhou, Weize Quan, Sifei Li, Jingdong Chen, DanDan Zheng, Libin Wang, Weiming Dong  

**Link**: [PDF](https://arxiv.org/pdf/2608.22757)  

**Abstract**: Unified models for visual understanding and generation have made rapid progress, yet they still lack the ability to understand and manipulate the spatial states of object instances. Existing models can describe objects in natural language, but they struggle to precisely represent continuous object poses and generate geometrically consistent images under target viewpoints. To mitigate this, we propose \emph{Object-Uni}, a unified model for object-centric spatial understanding and controllable generation. Specifically, we formulate object-centric spatial intelligence as a unified problem connecting pose perception, spatial reasoning, pose-conditioned generation, and object-centric novel view synthesis. We treat object pose as an explicit geometric variable shared by understanding and generation, rather than merely a prediction label or control signal. To make pose usable by multimodal large language models, we propose a viewpoint-based orientation abstraction that maps orientation into structured viewpoint descriptions while preserving continuous geometric supervision. We further construct an object-centric spatial benchmark (UniSpatial-80K) and train a unified model with an object-token-grounded pose anchor to associate each instance with its pose state. Experiments show that our model improves object-level pose understanding and pose-controllable generation, moving unified models from describing objects toward manipulating spatial states. 

---
# DiaRelay: Relaying Dialogue Context with a Constant-Size Memory for Emotion Recognition in Conversation 

**Authors**: Zihao Zhou, Bin Yang, Jinghui Qin, Kebing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2608.22745)  

**Abstract**: Emotion Recognition in Conversation (ERC) requires models to identify subtle emotional cues that are often distributed across distant dialogue turns. Existing methods typically incorporate dialogue history through a fixed context window. However, short windows discard potentially useful long-range evidence, while enlarging the window repeatedly re-encodes overlapping utterances, increases computational and memory costs, and may introduce irrelevant context. Moreover, commonly used parameter-efficient adaptation methods, such as LoRA, mainly introduce fixed low-rank transformations in the feature space and do not explicitly maintain a dialogue-level state or condition their transformations on the evolving conversational context. To address these limitations, we propose a lightweight adapter, DiaRelay, to enable LLMs to explicitly maintain a dialogue-level memory for accurate ERC. Based on LoRA, DiaRelay introduces two extra tightly collaborative components, Selective Relay Memory Transition and Dual-axis Relay Memory Read. Selective Relay Memory Transition progressively aggregates useful historical evidence into a bounded relay memory and propagates it across successive utterance predictions. This allows earlier emotional cues to influence later predictions after they leave the local context window, without re-encoding the complete dialogue history or expanding the backbone context length. Dual-axis Relay Memory Read uses the propagated memory to dynamically modulate low-rank feature transformations, enabling context-dependent representation adaptation without test-time gradient updates. Extensive experiments show that DiaRelay can achieve SOTA weighted F1 and accuracy on MELD while obtaining competitive results on IEMOCAP with only an extra 7.1M trainable parameters, indicating the effectiveness and generalizability of our DiaRelay in enhancing LLM-based emotional understanding. 

---
# TEE-X: TEE-aware Acceleration Framework for Large Vision Models at the Edge 

**Authors**: Kurt M Wilson, Mohaiminul Al Nahian, Abeer Matar A. Almalky, Sadat Shahriyar, Souvik Kundu, Zhishan Guo, Abdullah Al Arafat, Adnan Siraj Rakin  

**Link**: [PDF](https://arxiv.org/pdf/2608.22716)  

**Abstract**: Despite their remarkable success, machine learning models, particularly in vision applications, are alarmingly vulnerable to a range of security threats. One key factor in the attack landscape is the distinction between white-box and black-box threat models, as the latter poses challenges that limit attack effectiveness when access to model information is limited. As a result, using Trusted Execution Environments (TEEs) enhances security for machine learning applications by protecting model confidentiality and execution integrity, effectively shifting the execution environment from the white-box to the black-box side of the threat model spectrum. While adopting TEEs for large vision models, e.g., Vision Transformers (ViTs), is crucial for enhancing security and privacy, significant challenges related to memory constraints and increased computational latency must be addressed, especially in time-sensitive edge applications where safety and privacy are paramount. The objective of this work is to enable large vision models to be fully hosted within TEEs, achieving GPU-level inference latency for time-sensitive edge vision applications while maintaining performance. To this end, we propose TEE-X, a TEE-aware acceleration framework that introduces a sensitivity-aware modularization technique and enables vectorization in TEE inference. This design is validated on OP-TEE for Arm TrustZone, configured to optimize performance on the NVIDIA Jetson AGX Xavier for efficient edge vision applications using ViT models. The findings reveal that TEE-X delivers an effective TEE-aware acceleration framework that achieves minimal accuracy-latency trade-offs while ensuring fast and secure edge inference for vision models. 

---
# Enrich-Retrieve-Rank: Scaling Capability Discovery Beyond In-Context Routing 

**Authors**: Nazib Sorathiya, Daniel Zhang, Bardiya Akhbari  

**Link**: [PDF](https://arxiv.org/pdf/2608.22695)  

**Abstract**: Agent ecosystems now include thousands of MATS components (Models, Agents, Tools, and Skills), yet their discovery still relies on in-context routing. These systems read a registry (names, hints, or descriptions, as context budget permits), pick a candidate, invoke it, and retry on failure. This pattern degrades with scale, and registries are growing fast. We recast capability discovery as search over a registry by defining an offline enrichment step that turns sparse metadata into searchable profiles, and an online retrieve-then-rank pipeline that returns a ranked shortlist without invoking any candidates online. We show that from N=10 to 7,278 capabilities, in-context routing's top-1 accuracy (Match@1) collapses (0.85 to 0.12), while retrieve-then-rank degrades more gently (0.81 to 0.39) because its reranker still ranks the right capability first 0.70-0.87 of the time once retrieval finds it. In the Nova Micro sweep, the crossover is around N=500. We compare against two in-context baselines. Full-Ctx puts the whole registry in the prompt and asks the LLM to pick. Search&Pick gives the LLM a search tool to narrow candidates before it picks. At full scale the pipeline leads Search&Pick by 6.5 percentage points (pp) on Match@1 at about half the cost. It reduces cost 70x versus Full-Ctx. We use a fixed configuration (same enrichment, retriever, and scorer weights) across agent, tool, and skill registries. The pipeline runs in production as the default capability-discovery layer of a large-scale multi-agent platform. 

---
# RACO: Reliability-Aware Coarse-Goal Optimization for Inspection-Oriented UAV Vision-Language Navigation 

**Authors**: Sen Wang, Yiming Sun, Jiaxuan He, Pengfei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.22678)  

**Abstract**: UAV vision-language navigation (UAV-VLN) is commonly evaluated as goal reaching, but inspection-oriented deployment requires the agent to stop within a valid inspection region and avoid falsely confirming visually or semantically similar distractors. This requirement exposes a key weakness in existing coarse-to-fine UAV-VLN policies: the coarse goal predicted before local refinement is often treated as reliable, although it may drift toward plausible but incorrect object regions and limit the ability of the local stage to recover. To systematically evaluate this problem, we introduce LG-UVI, an object-centric inspection evaluation setting derived from CityNav/CityRefer. LG-UVI extends standard UAV-VLN episodes with target objects, hard distractors, type-aware inspection regions, and diagnostics for inspection-region arrival and object-level confirmation. To address this inspection-oriented setting, we further propose RACO, a reliability-aware adaptive coarse-to-fine navigation framework. Instead of treating the predicted coarse goal as a fixed waypoint, RACO views it as a runtime hypothesis and uses object-level candidate anchors to check and correct coarse localization before Stage 1 and at the Stage 1-to-Stage 2 boundary. RACO also applies scale-adaptive terminal refinement to handle terminal near-miss cases using runtime-observable geometric and anchor-based evidence. Under a unified online evaluation protocol, RACO improves SR over the reproduced HETT baseline by 9.53 and 7.98 percentage points on validation-unseen and test-unseen, respectively. It also improves inspection-region arrival and reduces false verification risk, showing that coarse-goal reliability optimization is an effective complement to existing coarse-to-fine UAV-VLN policies. 

---
# Hyperbolic Hierarchical Clustering for Visual Representation Learning 

**Authors**: Jianan Wei, Guikun Chen, Zhiyuan Weng, Chunchao Guo, Yujia Wang, Wenguan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22665)  

**Abstract**: We investigate the token mixer in vision backbones by revisiting clustering, one of the most classic approaches in machine learning. An effective token mixer is a fundamental component of modern vision backbones like vision Transformers, facilitating information exchange between image patches. Mainstream token mixers, which rely on convolution, attention, MLP, or their hybrids, primarily focus on navigating the trade-off between accuracy and computational cost. However, a significant drawback of these methods is their black-box nature; their encoding process is opaque and lacks interpretability. Diverging from these opaque designs, we introduce ClusterMixer, a transparent token mixer that is grounded in a clustering paradigm and interpretable by design. ClusterMixer explicitly formulates the token mixing process through a hierarchical clustering mechanism. To model the natural, tree-like relationships inherent in visual data, the clustering is performed in hyperbolic space, which is well-suited for embedding hierarchies with low distortion. Building on this innovation, we present HCFormer, a new backbone architecture that integrates ClusterMixer with a series of meticulously designed clustering strategies to ensure robust performance across tasks. Extensive experiments demonstrate that HCFormer consistently outperforms its counterparts across diverse tasks, including image classification, object detection, instance segmentation, and semantic segmentation. Considering its transparency and efficacy, we hope HCFormer can facilitate a paradigm shift toward interpretable backbones. 

---
# Physical Agentic AI: An Architecture for Orchestrating a Robot Crew with LLMs 

**Authors**: Xinyuan Liu, Eren Sadikoglu, Riana Chatterjee, Ransalu Senanayake  

**Link**: [PDF](https://arxiv.org/pdf/2608.22657)  

**Abstract**: Agentic AI frameworks interpret open-ended task goals and decompose them into multi-step plans. Richer information about embodiment-specific capabilities, physical preconditions, and cross-robot coordination improves grounding, but does not eliminate infeasible, mistimed, or unsafe physical actions. Physical robot crews therefore require an explicit architectural interface between semantic planning and execution, where every planned action is verified against robot capabilities, system state, and workflow constraints before actuation. This paper introduces Physical Agentic AI, a framework for skill-grounded robot agent orchestration, in which each robot exposes a typed library of executable skills while a foundation model planner decomposes a task into phases and assigns each phase to a robot-skill pair. A Robot Orchestration layer exposes the skill library, robot state, named locations, and workflow contracts to a non-actuating Mission Planner, while a deterministic Robot Orchestrator validates and authorizes one skill at a time. We evaluate on a drone-UGV search-and-dispatch mission, where every mission in every condition is executed live in Gazebo, and on a humanoid-quadruped transportation task using hardware-equivalent skill interfaces plus two physical trials on a Unitree G1 and Go2. Varying planner knowledge and runtime enforcement independently, we find that retrieval raises skill grounding from 51% to 96% yet leaves informed planners dispatching 23-29% of faulted steps. Per-dispatch enforcement reduces false dispatch to 0% with no false blocks, and a held-plan ablation confirms that the gate, not plan variation, is responsible. Live execution makes the difference physical: without enforcement all eight injected faults crossed the orchestration boundary and six produced robot motion; with enforcement all eight were refused before motion. 

---
# Evaluating Inference-Time Defenses Against Package Hallucination in LLM-Generated Code 

**Authors**: Alberick Euraste Djire, Iyiola E. Olatunji, Melissa Tessa, Earl T. Barr, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2608.22652)  

**Abstract**: LLMs are increasingly used for code generation, yet they frequently hallucinate non-existent software packages, creating exploitable entry points into the software supply chain. We make four contributions to this problem. First, we show that prior evaluation methodologies systematically inflate hallucination rates by misclassifying standard-library modules as hallucinations in some languages. For Python, the overestimation reaches 9.4 percentage points. Second, we evaluate seven inference-time defenses for mitigating package hallucinations, including five guided decoding strategies (Greedy, Contrastive, DoLa, Nudging, and Active Layer-Contrastive Decoding), an iterative self-refinement approach (Self-Refine), and a Retrieval-Augmented Generation (RAG)-based defense.. Across eight models spanning five families and four programming languages (Python, JavaScript, Ruby, Rust), RAG reduces the package hallucination rate (PHR) in 18 of 32 model--language configurations. Third, we introduce Package Utility (PU) to assess whether defenses preserve valid and task-relevant recommendations. Among strategies evaluated, Greedy decoding provides the strongest average mitigation--utility trade-off. Fourth, we stress-test all strategies under adversarial prompts seeded with fabricated package names and find that PHR surges by up to 45 percentage points relative to standard prompts, with Ruby consistently the most vulnerable language (80.9--95.2\%). Under adversarial conditions, RAG and Self-Refine outperform all decoding-only strategies, indicating that robust defense requires either external grounding or iterative self-verification when prompts are actively hostile.
Our results recast package hallucination as both a measurement problem and a decoding-time control problem, and they demonstrate that the choice of defense must be matched to the threat model and recommendation utility. 

---
# Mol-JEPA: A multimodal Joint Embedding Predictive Architecture for Molecules 

**Authors**: Florian Rottach, Sebastian Schieferdecker, William Rudman, Randall Balestriero, Carsten Eickhoff  

**Link**: [PDF](https://arxiv.org/pdf/2608.22642)  

**Abstract**: Despite recent advances in molecular foundation models, several limitations remain, such as chemically invalid augmentations, modality collapse, and incomplete representation of biochemical environments. To address these challenges, we present \textbf{Mol-JEPA}, a scalable framework for learning molecular world models. Rather than relying on suboptimal molecular perturbations, our model uses modality masking to exploit information from molecular structures, cellular phenotypes, binding affinities, ADMET profiles, quantum chemistry simulations and other drug discovery data. Across various benchmarks, we show that the representations learned by Mol-JEPA deliver strong performance, demonstrating the value of incorporating biochemical context through latent space prediction. 

---
# Do Not Copy/Paste: Soft Barriers for Copying in AI-Assisted Programming 

**Authors**: Iyiola E. Olatunji, Alberick Euraste Djire, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2608.22638)  

**Abstract**: Copying a function from a chat window into an editor takes less than a second. For many uses of AI coding tools, that speed is the point; in settings such as programming education, code review, and security-sensitive development, it can also be the problem. This paper frames copy-paste as an \emph{AI code handoff problem}: the moment model-generated text crosses from a conversational context into executable or committed software is a design boundary that current tools leave largely unmanaged. We argue that AI coding assistants should not only be evaluated by the code they generate, but also by how they mediate the transfer of that code into software artifacts. We propose \emph{soft barriers} as one class of handoff-aware mechanisms. Soft barriers preserve access to AI assistance while making unexamined transfer less frictionless. As an initial technical probe, we instantiate this idea using Unicode output perturbations that preserve visual readability but disrupt naive copy-paste execution. We introduce Copy-Paste Resistance (CPR), the fraction of functionally correct clean solutions that become syntactically invalid after perturbation. Across HumanEval and MBPP with four LLMs and four perturbation families, we find that output-level barriers can achieve high copy-paste resistance, but their effectiveness is highly model- and task-dependent. An exploratory pilot with 18 participants provides early evidence that soft barriers can shift users from direct transfer toward editing and reconstruction. We do not present Unicode perturbations as a deployment-ready solution; rather, we use them as a minimal probe for a broader research agenda on practical, transparent, and policy-aware AI code handoff. 

---
# GeoRisk-RAG: A Hierarchy-Aware Risk Framework for Improving RAG Reliability through Selective Answering 

**Authors**: Meenu Ravi, Shailik Sarkar, Lulwah AlKulaib, Yordanos Tessema, Chang-Tien Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.22634)  

**Abstract**: Current work on improving reliability in large language model (LLM)- generated answers has primarily leveraged Retrieval-Augmented Generation (RAG), knowledge-graph augmentation, and reinforcement learning. While these methods are adept at enhancing and measuring reliability through semantic similarity and faithfulness, they often struggle to distinguish semantic similarity from geographic validity. This is especially critical in natural hazard management domains where geographic granularity (i.e., town vs. city vs. state) is significant for decision-making, as responses valid in one municipality may not transfer to another. In such domains, a confidently wrong answer carries greater risk than abstaining. We present GeoRisk-RAG, a novel hierarchy-aware framework that addresses this geographic-validity gap through selective answering. This framework explicitly estimates geographic applicability using a Directed Acyclic Graph (DAG)-based distance for context retrieval before response generation. Experiments on a novel held-out wildfire-related question-answering (QA) dataset show that GeoRisk-RAG significantly reduces false confidence rates for location-dependent questions, lowering the rate to 0.009 compared with ~0.090 for standard semantic similarity and reranking baselines, while consistently achieving higher human preference alignment. This work provides a more comprehensive assessment of end-to-end RAG pipelines by integrating geographic validity and selective-answering behavior for safer decision-making in geospatial domains. 

---
# Teaching LLMs How ICU Physicians Approach Clinical Reasoning Through OMOP-Aligned Retrieval Improves Reasoning Across Clinical Domains 

**Authors**: Miguel Contreras, Scott Siegel, Subhash Nerella, Jessica Sena, Jiaqing Zhang, Heng Sun, Hruday Tej Akkaladevi, Peiyu Lu, Jordan Rosen, Sumit Kapoor, Sasank Desaraju, Grace R. Thompson, Jacob Purcell, Michael Petrauskis, Philip KW. Hong, Meghan Brennan, Sarah Chrabaszcz, Tierra Smith, Ronnie Ren, Michel S. Kabbash, Ceyhun Haziroglu, Rushi Patel, Gabriel Gomez, Charlotte Chaiklin, Randy Leung, Kenneth N. John, Whitman Wiggins, Philip Kayser, Vincent Bird, Maria Bruzzone, Tyler J. Loftus, Azra Bihorac, Parisa Rashidi  

**Link**: [PDF](https://arxiv.org/pdf/2608.22622)  

**Abstract**: Clinical decision-making relies on identifying relevant patient information to guide diagnosis and treatment, a challenge that is especially difficult in the data-dense and rapidly changing intensive care unit (ICU). Large language models (LLMs) could support this task. However, existing applications and datasets mostly emphasize surface-level retrieval or factual recall rather than the inductive and deductive reasoning clinicians practice to select and reason over decision-relevant evidence. We hypothesized that training LLMs on expert ICU reasoning could yield clinical reasoning skills that generalize beyond critical care. Here we introduce ICU-REACT, a reasoning dataset developed with 19 clinicians through a clinician-in-the-loop framework to teach LLMs to perform information retrieval and context-aware clinical reasoning in the ICU. Using ICU-REACT, we fine-tuned Clin-REACT models spanning 8B-70B parameters and three model families. Across five clinical reasoning benchmarks, Clin-REACT consistently outperformed its backbone models and open-source general-purpose and medical LLMs. Gains extended to different tasks including script concordance tests, and downstream diagnosis and treatment tasks. These findings suggest that expert reasoning supervision in critical care can improve broader clinical reasoning, although prospective evaluation is needed before real-world clinical use. 

---
# AI-based worker guidance in assembly and disassembly operations using multimodal ego/exo-centric data capture and structured task knowledge 

**Authors**: Vivek Chavan, Jörg Krüger  

**Link**: [PDF](https://arxiv.org/pdf/2608.22617)  

**Abstract**: Assembly and disassembly processes rely on expert knowledge that is difficult to document, reuse, and transfer. This paper presents a data-centric approach for extracting structured task knowledge from expert demonstrations using egocentric and exocentric recordings. Temporal and multimodal information from video and narration is jointly encoded to derive structured task representations that enable procedural documentation and context-aware worker guidance. The approach is evaluated on a real-world disassembly case study, demonstrating that video-based representations capture procedural structure and execution context beyond static image-based methods. The results highlight the potential of egocentric video understanding for repair, training, and circular manufacturing applications. Project website: this https URL 

---
# Vision-Language Models for Occupational Physical Exposure Assessment: Estimating External Hand Forces in Manual Material Handling Tasks from RGB Video 

**Authors**: Mohammad Sadra Rajabi, Aanuoluwapo Ojelade, Sunwook Kim, Maury A. Nussbaum  

**Link**: [PDF](https://arxiv.org/pdf/2608.22586)  

**Abstract**: External hand forces are important inputs to biomechanical analyses of occupational physical exposure and injury risk, yet continuous force measurements during manual material handling (MMH) typically requires instrumented objects or specialized sensing. We evaluated a vision-language model (VLM)-based pipeline that combines task-specific textual cues, visual representations, and known box mass to estimate dynamic, triaxial, bilateral external hand forces from RGB video. Thirty-five healthy young adults performed five MMH tasks involving lifting, carrying, pushing, and pulling with box masses of 6, 9, and 12 kg. The pipeline used text-guided localization of participant and handled-object regions of interest (ROIs), pretrained vision-transformer feature extraction, and transformer-based temporal regression. Performance was evaluated using leave-one-subject-out validation across seven camera-view conditions (three single-view and four multi-view conditions) and four ROI strategies. Overall, root mean square error was ~4.7-5.6 N for the horizontal and mediolateral force components and ~10.6-11.0 N for the vertical component. Including the handled object as a second ROI generally improved force estimation, with some of the largest benefits under single-camera conditions, whereas pixel-level segmentation provided little additional improvement. Multi-camera capture provided the clearest benefit for peak-force estimation, particularly for the vertical component, whereas differences in overall frame-level error among camera configurations were comparatively modest. These findings demonstrate the feasibility of estimating continuous, bilateral, directional hand-force estimates from RGB video and known load mass without requiring sensors on the worker or handled objects as model inputs, supporting the development of more scalable occupational physical exposure and risk assessments. 

---
# Clinical Graph-JEPA: Predictive Patient-State Knowledge Graphs for Cognitive Decision Support 

**Authors**: Kushagra Yadav, Nalin Prabhath, Amit Lamba, Goeun Han, Yining Mao  

**Link**: [PDF](https://arxiv.org/pdf/2608.22583)  

**Abstract**: Clinical records contain rich evidence about patient state, but converting that evidence into reliable, structured knowledge graphs remains difficult because extraction errors, ontology mismatch, missing relations, and temporal ambiguity can propagate into downstream systems. We propose a clinical knowledge graph construction and refinement framework that combines multi-agent relation proposal, ontology-aware normalization, deterministic evidence scoring, and JEPA-based latent refinement. Rather than treating a clinical knowledge graph as a static extraction artifact, we treat it as a predictive patient-state representation. For each admission, the system constructs an evidence-scored graph from structured MIMIC-IV records and inferred clinical cross-links, then learns to recover held-out clinical relations from the observed graph context. We evaluate the refiner with leakage-free leave-one-out edge recovery (MRR and Hits@k) and held-out batch-mask evaluation (AUC and MRR). To isolate the contribution of discharge-note context, we compare a note-embedding-free configuration with a note-augmented configuration that injects real discharge-note representations only into note-grounded entities. Under the same cohort and evaluation protocol, entity-grounded note injection improves overall leave-one-out MRR by 31% relative improvement. 

---
# Hybrid Panels: Toward Human-AI Collaboration in Survey Research 

**Authors**: Julia Romberg, Tobias Gummer, Gabriella Lapesa, Tanja Kunz, Claudia Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2608.22582)  

**Abstract**: Large-scale population surveys are essential for generating robust social and scientific insights, yet they face significant challenges, including declining response rates, increasing data collection costs, long delays between data collection and data provision, and the risk of nonresponse bias. Advances in artificial intelligence (AI) have opened up new opportunities for AI-supported survey infrastructures where the goal is to overcome these challenges without limiting the data quality. A promising AI-enabled survey infrastructure for which we build a first pilot is a hybrid panel. A hybrid panel is a longitudinal AI-enabled survey which allows to iteratively improve the alignment between large language models (LLMs) and the population they aim to simulate and use the errors to inform the design and implementation of the next survey wave (e.g., inform the participant recruitment, assignment of questions to participants). It incorporates both human participants and LLMs as fundamental elements of its design. In this research note, we introduce the concept of a hybrid panel by providing a definition and outlining an overarching framework, spanning data collection to data validation. We detail results from a first pilot study to illustrate (open) challenges that we identify for hybrid panels. 

---
# BLADE: Bilevel Low-rank Augmented-Lagrangian Erasure for LLM Unlearning 

**Authors**: Md Toufikuzzaman, Ahmad Mousavi, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.22557)  

**Abstract**: Existing LLM unlearning methods struggle with robustness: unbounded forget losses degrade model coherence, fixed-weight balancing cannot adapt as retain difficulty shifts mid-training, and methods that work on one benchmark falter under scaling or repeated application. We propose BLADE, a constrained bilevel framework whose three mechanisms give smooth, predictable control over the optimization landscape: a clamped-entropy forget loss whose gradient is exactly zero once a token reaches sufficient uncertainty; an asymmetric augmented Lagrangian that permanently ratchets retain protection after any violation; and a bilevel structure confined to LoRA adapters that repairs retain damage before each forgetting step. BLADE dominates across three benchmark families, improving average composite scores over the strongest baselines by $6$% on TOFU, $9$% on MUSE Books, and $7$% on KnowUndo, and it remains stable under $4\times$ scaling and $4$ sequential unlearning steps on MUSE News where the best competing method collapses entirely. 

---
# Functional compatibility as a determinant of persistent neural learning 

**Authors**: Hossein Javidnia  

**Link**: [PDF](https://arxiv.org/pdf/2608.22462)  

**Abstract**: Artificial neural networks can acquire new capabilities but often damage existing ones when they continue to learn. This stability-plasticity problem has motivated replay, regularization and constrained-update methods, yet it remains unclear whether a property of incoming learning itself determines what can be retained without disrupting protected behaviour. Here we show that functional compatibility, the extent to which new learning can coexist with behaviour that must be preserved, is a causal determinant of persistent learning. To our knowledge, this is the first controlled causal demonstration in which compatibility is deliberately changed from matched neural states and persistent learning is measured under a common retention requirement. The effect generalizes across independent learning directions, convolutional and transformer architectures, vision and text, and additional seeds. Learning rules differ in how efficiently they exploit available compatibility, while retention constraints limit how much can be stored. At larger finite updates, nonlinear geometry changes the available learning opportunity and ultimately prevents the matched compatibility continuum from being realized. These results establish functional compatibility as an experimentally controllable principle of persistent neural learning, shifting the problem from preventing forgetting towards identifying which components of new learning can safely become permanent. 

---
# EMPIRE: Explicit Manipulation Planning as a Learnable Intermediate Representation for Egocentric Hand-Motion Forecasting 

**Authors**: Wen Wang, Ruibing Hou, Hong Chang, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.22449)  

**Abstract**: Forecasting dexterous hand motions from egocentric observations is fundamental to intelligent interactive systems. Existing VLM-based methods typically map observations directly to future motions, overlooking the underlying manipulation process that governs hand-object interactions. Moreover, end-to-end optimization couples manipulation learning with motion synthesis, causing motion-generation gradients to interfere with the pre-learned manipulation-aware representations. To overcome these limitations, we propose EMPIRE, a two-stage framework that introduces Explicit Manipulation Planning as an Intermediate Representation for Egocentric hand-motion forecasting. Stage I: Learn to Plan. EMPIRE first learns explicit manipulation plans from multimodal context to capture the progression of hand-object interactions. Stage II: Learn to Act. A motion generator synthesizes future bimanual hand motions conditioned on frozen planner representations, preventing motion-generation gradients from affecting manipulation planning. To support our method, we further construct EMPIRE-651K, a bimanual hand-motion forecasting dataset comprising 650,910 training windows across 111 tasks, each paired with an explicit per-hand manipulation plan. Under identical training and evaluation protocols, EMPIRE achieves state-of-the-art forecasting accuracy, with an MPJPE of 84.53 mm and a finger-relative error of 38.97mm. We release the code and dataset at this https URL. 

---
# Rank Reversal in Multilingual LLM Judges: A Label-Free Double-Centering Calibrator 

**Authors**: Alhasan Mahmood, Samir Abdaljalil, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2608.22432)  

**Abstract**: Multilingual LLM judges produce different evaluator-backbone rankings depending on the prompt language: on an eight-language Agent-as-a-Judge benchmark, the top-ranked backbone alternates across English, Arabic, Chinese, Hindi, Japanese, Spanish, Turkish, and Swahili, and 7 of 15 backbone pairs show statistically significant pairwise rank reversal. We treat this as a measurement problem. The multilingual judge score decomposes additively into task difficulty, backbone skill, and a language-backbone interaction term, the last of which is recoverable without human labels by double-centering the cell-mean score matrix. We make this estimator (\textbf{Consensus-Based Calibration}, CBC) explicit, give an $O(1/\sqrt{n})$ finite-sample concentration bound with variance constant $(1-\tfrac{1}{m})(1-\tfrac{1}{k})$, and show that it is unbiased even when task-language interactions are present. Across 7{,}920 judge runs (6 backbones, 8 languages, 55 tasks, 3 frameworks), CBC raises held-out cross-task rank consistency $\tau$ from 0.650 to 0.902 and agrees with the held-out additive-model oracle in 100\% of per-language decisions versus 68.5\% raw; these are consistency diagnostics, not human-grounded correctness measures. On a separately collected M-RewardBench panel (7 languages, 1{,}500 items per language, 10{,}500 language-item instances, 5 evaluators), panel agreement with the public human gold preferences rises from 68.7\% to 76.6\% (gain 7.9 percentage points, 95\% CI $[6.0, 9.9]$), our strongest external evidence of downstream usefulness. The estimator is the standard two-way ANOVA interaction-recovery operation under sum-to-zero contrasts; our contribution is its application as a label-free post-hoc calibrator for multilingual LLM judges, an explicit finite-sample concentration bound, and an unbiasedness result that holds even under task-language misspecification. 

---
# Cross-Subject Generalization in Decoding Perceived Speech from Non-Invasive Brain Recordings 

**Authors**: Aoke Zhang, Bo Wang, Xihong Wu, Heping Cheng, Jing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.22420)  

**Abstract**: Decoding perceived speech from non-invasive brain recordings has garnered significant attention in recent years due to its wide range of potential applications. However, existing methods face considerable challenges in cross-subject decoding, primarily due to limited generalizability and the absence of explicit mechanisms for extracting subject-consistent information. These limitations result in high training costs and suboptimal decoding performance. To address these challenges, we propose an innovative Cross-Subject Perceived Speech Decoding (CPSD) framework, which comprises two training stages: source model pre-training and personal specialization. In the source model pre-training stage, contrastive learning is employed to capture shared representations across multiple source subjects. Subsequently, personal specialization initializes the model for the target subject by extracting consistent components from the source model and fine-tuning it using target subject data. Additionally, we introduce the Positional Encoding-based Spatial Attention (PESA) module, which remaps MEG/EEG data into a standardized reference space, thereby enhancing cross-subject consistency and facilitating model training. We evaluate the proposed CPSD framework on three perceived speech neural datasets encompassing different modalities and languages. The results demonstrate that our framework outperforms baseline methods by more than 6.8%, 15.4%, and 15.8% in Top-10 accuracy on the Armeni 2022, PKUEEG 2025, and Broderick 2018 datasets, respectively. Further analyses confirm the effectiveness, efficiency, and robustness of the proposed approach. 

---
# KONTOGRAPH: Verified Point-in-Time Feature Consistency and Amortised Explanation for Real-Time Anti-Money Laundering under a 200 ms Decision Budget 

**Authors**: Ahmed Abolfadl  

**Link**: [PDF](https://arxiv.org/pdf/2608.22389)  

**Abstract**: Regulation (EU) 2024/886 obliges European payment service providers to settle euro credit transfers in under ten seconds, around the clock. This removes both the overnight batch window in which anti-money-laundering (AML) analytics traditionally ran and the settlement delay that made recovery possible, forcing detection, explanation and decision inside a single-digit-second envelope. We present KONTOGRAPH, an end-to-end AML pipeline for the SEPA Instant rail built under a self-imposed 200 ms 99th-percentile budget, and report an empirical study on 1,562,860 simulated payments with injected typologies and deliberately incomplete labels. Three findings are of interest beyond the system itself. First, a temporal graph network with per-node memory improves PR-AUC over a gradient-boosted tabular baseline from 0.0053 to 0.1717, a paired day-blocked bootstrap difference of +0.166 with 95% CI [0.105, 0.241]; per-node memory alone more than doubles the score. Second, expressing each feature once and compiling it to three execution backends, with equivalence enforced by property-based tests that perturb the future, surfaced three point-in-time violations that code review had passed--each of which would have inflated reported performance. Third, and most consequential for practice, exporting the deployed tree ensemble to ONNX changed only $7.4 \times 10^{-8}$ in mean score yet altered 0.26% of decisions and inflated the alert volume by 12%, because 32-bit accumulation perturbs scores across a cost-optimal threshold of $3.98 \times 10^{-4}$. We argue that a serving-format conversion must be treated as a model change until measured, and that fidelity metrics for subgraph explainers can be vacuous when candidate neighbourhoods are small--a null result we report in full. 

---
# ProBel: Propaganda Detection with Techniques, Spans, and Explanations 

**Authors**: Mohamed Bayan Kmainasi, Ali Ezzat Shahroor, Elisa Sartori, Giovanni Da San Martino, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2608.22388)  

**Abstract**: Propaganda detection includes several related prediction levels, ranging from sentence-level decisions to technique classification and span identification. However, it remains unclear how supervision at these levels interacts when learned jointly across Arabic and English. We present ProBel, an Arabic and English resource that aligns binary labels, multi-label annotations over 23 propaganda techniques grouped into six coarse categories, technique-labeled spans, and reference explanations for the same news sentences. It includes a substantially larger English collection and supports matched binary, coarse-grained, multi-label, and span-level tasks in both languages. We evaluate zero-shot prompting, task-specific fine-tuning, and joint training under a shared setup. A single bilingual multi-task model achieves the best overall performance and remains competitive across tasks and languages. Cross-task analysis shows that transfer depends on the supervision level. Joint classification training preserves binary performance, whereas span-only training can weaken sentence-level prediction. Joint bilingual training yields the most stable results, while monolingual fine-tuning can reduce transfer to the other language. We will release the data, code, and evaluation scripts. 

---
# Self-Supervised Graph Representation Learning for In-The-Wild Wearable and Smartphone based Emotion Recognition 

**Authors**: Ioannis N. Ziogas, Leontios J. Hadjileontiadis, Ahsan H. Khandoker, Aamna Al Shehhi  

**Link**: [PDF](https://arxiv.org/pdf/2608.22387)  

**Abstract**: Wearable and smartphone-based emotion recognition (WER) remains a challenging setting in affective computing, due to the notorious difficulty and bias associated with in-the-wild label collection. The high inter-and intra-subject emotional variability motivates us to explore WER modeling through graph node classification in a limited resources learning scheme powered by Self-Supervised Learning (SSL) graph masking augmentation tasks. We employ a subgraph sampling approach during training, utilizing labeled and unlabeled data, along with supervised, semi-supervised, and SSL mechanisms in a multi-task inductive graph neural network architecture. Our evaluations on K-EmoPhone through leave-one-group-out cross-validation in the binary arousal and valence tasks yield average accuracy gains of 4.3% and 7.8%, compared to the full resource setting, utilizing only 20% and 25% of the labels, respectively. Our model analysis sheds light on the relation of SSL graph augmentations to emotional arousal and valence and justifies the approach of SSL-driven subgraph training for in-the-wild WER. 

---
# Pre-Decoding Acoustic Triage for Budgeted Vision-Language Captioning of Untrimmed Egocentric Video 

**Authors**: Masoud Jalayer, Changyi Li, Yu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.22359)  

**Abstract**: Automatically analyzing hours-long egocentric video is increasingly essential for progress monitoring, quality control, and safety in logistics, construction, and manufacturing. Yet current pipelines that process short, fixed-size windows with a vision-language model (VLM) are prohibitively expensive because cost scales with the number of model calls. To reduce this cost, prior work proposes triage policies to select which windows merit a VLM invocation. However, these policies either sample uniformly or rank windows using visual features, which ironically requires the video decoding that the budget constraints are meant to avoid. We propose audio-first triage: select windows using the lightest modality, scored before any video frame is decoded, so the approach composes naturally with token compression or quantization. The novelty lies in the objective, not the representation: rather than a per-frame sound-event detector, we train the selector to trigger once per action. This objective shift improves action coverage by 4.0-10.8 percentage points across all evaluated call rates, using frozen AudioSet-pretrained features without domain-specific sound-event labels. Using fewer than half of the available calls, the triage cuts 9-20% of VLM calls at matched coverage on EPIC-KITCHENS-100 (EK-100), surpasses uniform sampling through the mid-range on Ego4D over 247 clips, and outperforms two recent visual keyframe selectors. Code, the reference implementation and every results file this manuscript reads are at this https URL. 

---
# SANE: State Anomaly Neutralization for Stable Extreme-Context Delta-Rule Models 

**Authors**: Qingwen Lin, Boyan Xu, Xiao Liu, Zhifeng Hao, Ruichu Cai  

**Link**: [PDF](https://arxiv.org/pdf/2608.22354)  

**Abstract**: Delta-Rule recurrent models maintain a fixed-size state, enabling $O(1)$ inference memory but potentially becoming unstable under extreme-context extrapolation. By tracking RWKV-7 over sequences of up to 100M tokens, we empirically identify a distinct failure pattern: \textbf{localized norm explosion atop a relatively sparse substrate}, rather than global state saturation. Analysis of the recurrent update suggests that persistent decay keeps weakly updated entries small, whereas uneven injections allow a few channels to accumulate extreme values. Motivated by this diagnosis, we propose \textbf{State Anomaly Neutralization (SANE)}, which applies adaptive $\tanh$ compression at chunk boundaries while preserving the intra-chunk parallel structure. Within a safe threshold range ($3 \le \alpha \le 5$), SANE matches the baseline on 11 short-context reasoning benchmarks with no statistically significant degradation. After a 100M-token prefix, which exceeds the training length by over $24{,}000\times$, SANE retains functional reasoning ($33.46$--$35.56$) while the baseline encounters numerical overflow. In contrast, overly permissive thresholds ($\alpha \ge 8$) remain numerically stable but lose reasoning capability entirely, showing that numerical stabilization alone does not guarantee functional reasoning and revealing a capacity--stability trade-off in state compression. 

---
# Multimodal examination answer data with expert-designed Outcome-Based Education rubrics for criterion-level assessment 

**Authors**: Jahangir Alam SM, Md Khalid Syfullah, Saad Ahmed, Munira Akter Mou, A K Z Rasel Rahman, A.K.M. Masudur Rahman, Mohammed Sowket Ali  

**Link**: [PDF](https://arxiv.org/pdf/2608.22346)  

**Abstract**: This data article describes a multimodal collection of scanned examination answers paired with expert-designed Outcome-Based Education (OBE) grading metadata. The collection contains 485 answer submissions from 415 consenting students at four academic institutions. Eight faculty contributors supplied examination materials covering nine subjects and 12 distinct question templates. Each answer-level item links a scanned PDF to a randomized identifier, subject label, question, model answer, criterion definitions, performance-level descriptions, criterion marks, and a total mark. The 12 rubrics contain 47 criteria in total. The scans retain realistic academic content, including handwriting, printed text, equations, tables, code, figures, sketches, and diagrams. CamScanner, Adobe Scan, and conventional scanners contributed variation in illumination, contrast, orientation, compression, and resolution. Diverse handwriting, crossed-out work, revised calculations, and inserted corrections add further visual variability for robustness and generalization studies. Preparation involved heterogeneous-source consolidation, label and text standardization, score validation, identifier randomization, filename randomization, and JSON-to-PDF integrity checks. An answer-level audit confirmed 485 unique identifiers, 485 unique PDF filenames, agreement between each total mark and its criterion-mark sum, and scores within the applicable rubric maximum. The data can support rubric-aware automated evaluation, multimodal document understanding, criterion-level feedback, score prediction, and privacy-aware OBE assessment research. Access is restricted to research use and is available from the corresponding author upon reasonable request. 

---
# TransHands: Repurposing Human Pose Encoders as Hand Pose Encoders 

**Authors**: Milo Piccioli, Gianluca Amprimo, Claudia Ferraris, Gabriella Olmo  

**Link**: [PDF](https://arxiv.org/pdf/2608.22341)  

**Abstract**: Lifting 3D hand poses from 2D monocular representations remains challenging due to the limited availability of large-scale, diverse 3D-annotated hand datasets, in contrast to the abundance of human body motion data. We address this limitation by transferring motion representations learned from large body pose corpora to the hand domain. We introduce TransHands, a backbone-agnostic transfer learning framework that enables pre-trained human motion encoders to be effectively adapted for 3D hand pose estimation from 2D pose inputs. Rather than training hand-specific biomechanical models from scratch, TransHands combines a two-stage training and fine-tuning strategy with a lightweight hand-specific input adaptation module that aligns hand kinematics with the representation space learned for full-body motion. We evaluate TransHands across four state-of-the-art motion modeling architectures, including transformer-based, graph-based, and frequency- domain models. Results demonstrate that motion priors learned from body pose data transfer consistently across architectures, yielding consistent accuracy gains, strong cross-domain generalization, particularly in challenging egocentric settings, and applicability for downstream tasks in real-world contexts. 

---
# Register Shifts Break LLM Safety: A Bengali Benchmark with Culturally Grounded Harms 

**Authors**: Naymul Islam, Nusrat Jahan Lia, Shubhashis Roy Dipta, Sabik Bin Sultan, Abdullah Khan Zehady  

**Link**: [PDF](https://arxiv.org/pdf/2608.22335)  

**Abstract**: Bengali is the seventh-most-spoken language globally, yet LLM safety evaluation remains overwhelmingly English-centric. We introduce BanglaSafe, a benchmark of 879 Bengali prompts combining 309 natively authored prompts with 570 expert-reviewed prompts, spanning 17 culturally grounded harm categories and five prompting conditions that vary language, writing style, and authority framing. Evaluating 18 frontier LLMs, we find that over half of all responses are unsafe or partially unsafe (53.6%) while 14.7% contains strictly harmful content, and that the strongest observed effect is not the switch from English to Bengali but the choice of writing style within Bengali: the same harmful request phrased as a formal newspaper investigation succeeds 17 percentage points more often than the same request phrased as a casual message, with no adversarial engineering involved. We further show that existing safety classifiers struggle to reliably evaluate Bengali content, with even frontier models failing on nearly half of all cases. 

---
# The Imitator Game: Benchmarking Robot Imitative Ability Beyond Action Prediction 

**Authors**: Xunzhe Zhou, Yiyang Cai, Fengyi Wang, Ran Ju, Hanxiang Ren, Ruizhe Liu, Yu Zhang, Qian Luo, Feng Chen, Pei Zhou, Yi Ma, Yanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22301)  

**Abstract**: Humans imitate at the level of intent: given a demonstration, we infer its goal and carry it out with whatever tools, objects, and layouts are at hand. Current robot policies instead learn observation-to-action mappings from visual inputs and language instructions, without explicitly inferring the demonstrated task. Learning from human video thus remains largely trajectory-level: models can replay motions in near-identical scenes, but still struggle to imitate what the demonstrator intends rather than merely what they do. We introduce The Imitator Game, a four-level benchmark (L0-L3) that progressively widens the gap between the human demonstration and the robot's own scene, isolating where trajectory replay ceases to suffice and task understanding becomes necessary. We pair it with IG-10K, the largest environment-aligned paired human-robot dataset to date and the only one instantiated across all four levels in both real and simulated settings (20,000+ paired episodes, 50+ tasks, 6 domains), and Imitator Arena, an open platform for blind A/B human evaluation. Across nine state-of-the-art models, performance is stable from L0 to L2 but collapses at L3, identifying functional substitution - achieving the same intent through a different object affordance - as the decisive barrier to intent-level imitation. Human-video-conditioned models outperform caption-conditioned ones, yet every model falls below 13% zero-shot success on unseen tasks; fine-tuning IG-10K-pretrained models with only $10$ paired human-robot demonstrations yields large gains that grow with pretraining scale. The project website and access to Imitator Arena are available at this https URL. 

---
# LLM Evaluation on Unseen Questions: Contextual Multidimensional IRT Model 

**Authors**: Ergan Shang, Weijing Tang, Yinqiu He  

**Link**: [PDF](https://arxiv.org/pdf/2608.22295)  

**Abstract**: Evaluation of large language models (LLMs) increasingly requires predicting how a model will perform on new questions or tasks before collecting large amounts of new annotations. This problem is challenging because question difficulty, scenario, and underlying capability demands can vary substantially. Simple retrospective averages may confound model ability with item characteristics. In this paper, we study a model-based evaluation framework that combines multidimensional item response theory model with question contexts to predict LLM performance on unseen questions. The framework represents LLMs through latent capability profiles while using question content to inform item characteristics, allowing information to transfer beyond previously observed items. Empirically, we find that for within-scenario evaluation, incorporating question embeddings improves prediction relative to model-free baselines, and that multidimensional latent structure provides a richer description of capability variation than unidimensional alternatives. At the same time, our results reveal an important limitation that the generalizability does not necessarily translate into reliable prediction under cross-scenario shift. These findings suggest that context-aware psychometric modeling is a promising direction for efficient and interpretable LLM evaluation, while also highlighting cross-scenario generalization as a central open challenge. 

---
# Learning from the Test: Self-Referential Differential Testing for Deep RL Agents 

**Authors**: Junda He, Jieke Shi, Zhou Yang, Mingfei Cheng, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2608.22284)  

**Abstract**: Deep Reinforcement Learning (DRL) has achieved significant success in complex decision-making problems. As DRL systems are increasingly deployed in real-world applications, ensuring their quality and reliability is paramount. Current works primarily focus on detecting safety-critical failures, often neglecting policy optimality, which can lead to reduced efficiency, user distrust, and economic losses. This oversight, compounded by the inherent "testing oracle problem" for optimality, leaves a significant gap in comprehensively evaluating DRL systems. To address this gap, we propose Delta (Differential Testing for DRL Agents), a novel and comprehensive framework that automatically identifies both safety-critical and optimality bugs in DRL agents. Delta employs a two-phase approach: (1) Safety Testing, where the Agent Under Test (AUT) is evaluated for catastrophic failures while collecting data from its decision-making policy, and (2) Optimality Testing, where this collected data from the prior phase is used to train a challenger agent via Offline Reinforcement Learning. Differential testing is then performed by comparing the challenger agent against the AUT; instances where the challenger achieves higher cumulative rewards indicate optimality issues in the AUT. We demonstrate Delta's effectiveness across five environments. We investigate the effectiveness of three offline RL algorithms (BC, BCQ, and CQL) in generating challenger agents. Experimental results demonstrate that safety testing datasets are valuable for training competent DRL agents. Challenger agents trained with BCQ proved most effective for identifying optimality issues within the framework of Delta. Across the five environments, Delta uncovered an average of 2,518 optimality issues, outperforming the baseline methods by 50.2%. 

---
# OVIBench: Benchmarking Online Video Question Answering under Interruption 

**Authors**: Naiming Liu, Zhiheng Wu, Shuning Wang, Tie Zhang, Bowen Liu, Tong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22279)  

**Abstract**: Recent vision language models (VLMs) have achieved strong progress in video understanding. However, most existing video QA research and benchmarks still follow an offline, single-round paradigm, overlooking realistic interactions where users may interrupt the model during answer generation. To address this gap, we formulate the task of Online Video Question Answering under Interruption and introduce OVIBench, the first standardized benchmark for evaluating VLMs in this setting. OVIBench categorizes interruptions into three types: Cancellation, False Trigger, Correction and supports both open-ended and multiple-choice evaluations. To enable large-scale and reproducible testing, we develop an offline simulation protocol that reproduces interruption during generation under a unified temporal setup, together with a multi-dimensional metric suite for assessing interruption understanding and response generation. Experiments demonstrate that OVIBench effectively distinguishes models' interruption-handling abilities, especially in following correction requests. Finally, we construct a train set OVI-Train for interruption-aware fine-tuning. Models fine-tuned on this dataset achieve significant gains on OVIBench, validating the effectiveness of our benchmark and data design. OVIBench, OVI-Train, and the evaluation code will be released. 

---
# Length-Adaptive Decoding for Masked Diffusion Machine Translation 

**Authors**: Yan Zhan, Mengkai Hou, Wanting Zhang, Zhijun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.22274)  

**Abstract**: Machine translation tests masked diffusion language models (dLLMs) because every source token must be rendered faithfully, while fixed canvas decoding must choose target length before denoising. Existing masked diffusion decoding work mainly studies token unmasking order, leaving this length decision under-explored despite its direct effect on coverage and redundancy. We introduce Entropy-Valley (EV), a training-free length selector that scores candidate target canvases by mean predictive entropy from all-mask forward passes and selects the canvas the backbone is most prepared to fill. Relative to a baseline using training corpus length statistics, EV recovers 64.9%, 65.3%, and 33.0% of the COMET-22 gain from reference target lengths on En$\to$Zh, Zh$\to$En, and En$\to$De. Our diagnostics show that denoising-friendly lengths need not match reference lengths. Evaluation by three translation experts supports the En$\leftrightarrow$Zh adequacy gains, with stronger evidence on Zh$\to$En. Compared with a LLaMA-3-8B autoregressive (AR) model trained on the same fine-tuning data, the EV system ties on En$\to$Zh and leads on Zh$\to$En; an oracle-length diagnostic further shows that, in this masked diffusion MT setting, deciding which tokens to reveal first matters less than how the target length is supplied. 

---
# Multi-Task Learning for Non-Canonical Phoneme Recognition via Articulatory Feature Decomposition 

**Authors**: Sophia Riaz, Haoze Zheng, Amos Roche, Miyu Zhang, Anamika Ragu, Salvatore Penachio, Kaustav Mukherjee, Aneesh Jonelagadda  

**Link**: [PDF](https://arxiv.org/pdf/2608.22273)  

**Abstract**: Pathological and more broadly non-canonical speech present significant challenges for automatic phoneme recognition due to systematic deviations from canonical pronunciation and limited availability of labeled clinical speech data. Existing phoneme recognition systems are typically trained on canonical speech and treat phonemes as atomic categorical labels, limiting their ability to detect structured articulatory errors common in speech disorders and accents. In this work, we introduce a linguistically structured approach to non-canonical phoneme recognition that decomposes phoneme prediction into articulatory feature dimensions such as manner, place, and voicing. We implement this formulation using a hierarchical multi-task learning architecture in which task-specific articulatory feature heads learn feature-level representations that are subsequently integrated through a cross-attention-based fusion module to produce phoneme predictions. To address the scarcity and noise of pathological speech labels, we combine this framework with semi-supervised learning via Momentum Pseudo-Labeling (MPL) and propose a cascaded training strategy that progressively introduces articulatory feature tasks while employing staged unfreezing of a pretrained speech encoder. Experiments on L2-ARCTIC, used as a proxy for pathological speech variation, show that the proposed approach achieves substantial improvements in phoneme recognition performance compared to strong baseline architectures, while yielding interpretable error patterns aligned with phonological feature structure. These results suggest that articulatory feature supervision is a promising strategy for robust and interpretable phoneme recognition in non-canonical speech, and motivate future validation on clinically diagnosed pathological speech datasets. 

---
# GAN-Diff : Coupling Pretrained WGAN-GP Features with Conditional Diffusion U-Nets 

**Authors**: Saif Ahmed, Ashadulla Hil Galib, S.M. Riaz Rahman Antu, Ahmed Faizul Haque Dhrubo, Souvik Pramanik, Mohammad Abdul Qayum, Mohsin Sajjad, Mohammad Ashrafuzzaman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2608.22272)  

**Abstract**: Generative adversarial networks (GANs) can provide efficient image generation, while diffusion models offer high-quality image restoration but require iterative sampling. This paper presents a hybrid GAN-guided diffusion framework that uses a pretrained Wasserstein GAN with gradient penalty (WGAN-GP) as a feature prior for conditional diffusion-based image restoration. Intermediate features from the frozen WGAN-GP generator are incorporated into a diffusion U-Net through cross-attention and remain fixed during the DDIM sampling process. The framework is evaluated on two restoration tasks, Gaussian denoising and 2Xsuper-resolution, using CelebA face images. During development, several sources of instability were identified and addressed, including adversarial learning-rate imbalance, inappropriate diffusion initialization, excessive corruption, and insufficient parameter averaging. The resulting framework consistently improves the quality of both degraded and low-resolution images. In particular, it improves denoising performance by 4.40 dB in PSNR and super-resolution performance by 3.70 dB over their respective input baselines. These results demonstrate the potential of a frozen GAN feature prior to guide diffusion models toward stable and effective image restoration. 

---
# Training-Free VLM Personalization via Calibrated Residual Decoding 

**Authors**: Jiaao Yu, Yujian Ma, Xianming Hu, Pengran Wang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.22263)  

**Abstract**: Vision-language models can be personalized in a training-free manner by directly providing user profiles, preferences, or visual references at inference time, without updating model parameters. However, direct personalized prompting does not guarantee that the model will reliably exploit such evidence. The predictive distribution under the positive user profile often mixes two sources: personalized signals genuinely supported by the current profile, and the model's generic visual or linguistic priors. As a result, from the positive-profile response alone, it is difficult to determine whether a high-confidence answer is supported by the user profile or merely reflects the model's default preference. To address this problem, we propose a training-free calibrated residual decoding framework. Given the same image and question, we construct three evidence conditions: a positive profile , a counterfactual profile , and an empty profile . Our method keeps the prediction under
as the anchored base, and explicitly estimates the marginal contribution of personalization from score differences across the three conditions. We further introduce normalized-entropy-based uncertainty calibration, allowing the strength of personalized enhancement to adapt to the reliability of the residual signal. Experiments on MMPB, YoLLaVA, and MyVLM show that the proposed method improves personalized multimodal understanding without fine-tuning, with consistent gains on identity-sensitive visual personalization tasks. Additional analysis shows that entropy calibration stabilizes residual decoding when the contrastive personalization signal is uncertain. 

---
# Improving Few-Step Language Flows with Untied Self-Conditioning 

**Authors**: Bocheng Li, Linli Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.22244)  

**Abstract**: Flow-matching language models refine all token positions in parallel and can trade sampling steps for latency, yet generation quality still degrades sharply with few sampling steps. We trace a source of this degradation to a train--inference mismatch in previous-prediction self-conditioning: during training, the self-conditioning input is computed from the current noisy state with no intervening solver step; during sampling, the solver folds the previous prediction into the latent before that same prediction reappears as the explicit self-conditioning input. This coupling, absent during training, creates redundancy that grows with step width. We show that the mismatch degrades both the self-conditioning input and the solver update, and derive a correction for each from the model's own structure. From the frozen projection weights we identify directions along which the self-conditioning input is redundant with the latent and dampen them; from the solver's integration structure we derive that a step-average prediction is needed and approximate it from prediction history, with scale set by offline trajectory statistics. The resulting sampler, Untied Self-Conditioning, requires no retraining and uses one evaluation per step. At 8 sampling steps on LangFlow, it reduces OpenWebText generative perplexity from $531$ to~$62$ ($8.6\times$); under an adapted Arena-Hard-Auto~v2 protocol, its outputs are preferred in $96\%$ of pairwise comparisons. On ELF-B it reduces generative perplexity from $71$ to~$43$. Improvements hold from 8 to 256 sampling steps. 

---
# FreKoo++: Learning Continuous Spectral Dynamics for Temporal Domain Generalization 

**Authors**: En Yu, Xiaoyu Yang, Wei Duan, Guangquan Zhang, Jie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.22224)  

**Abstract**: Temporal Domain Generalization (TDG) aims to learn from historical domains and generalize to unseen future distributions under concept drift. Nevertheless, prevailing TDG methods struggle with complex real-world streaming scenarios involving both multi-scale drift patterns (e.g., long-term periodicity intertwined with short-term incremental changes) and local uncertainties, especially in continuous settings where observations arrive irregularly. To address this limitation, we propose FreKoo++, a novel continuous spectral-dynamical framework that pioneers the unification of continuous Koopman modal dynamics with adaptive spectral disentanglement. Specifically, FreKoo++ maps source-domain parameters into a compact latent space, modeling their evolution as a superposition of learnable continuous modes where complex eigenvalues jointly encode oscillatory frequency and temporal growth or decay. This formulation naturally accommodates irregular timestamps and supports arbitrary horizon extrapolation without rigid discrete stepping. Furthermore, we propose a new adaptive soft spectral weighting mechanism backed by stability and spectral regularization, which automatically isolates persistent dominant dynamics from transient noise without relying on manual frequency thresholds. We derive modal approximation and generalization bounds that characterize how amplitude and eigenvalue estimation errors propagate with the prediction horizon. Extensive experiments on both discrete and continuous TDG benchmarks demonstrate that FreKoo++ achieves state-of-the-art performance under complex multi-scale drifts and irregular sampling. 

---
# Spending Scarce Confirmatory PET Measurements: Target-Aligned Validation in A4/LEARN 

**Authors**: Eliuvish Han Cui  

**Link**: [PDF](https://arxiv.org/pdf/2608.22223)  

**Abstract**: Anti-amyloid therapies and blood-based biomarkers are changing Alzheimer disease workups into a two-stage measurement workflow: screen broadly with cheaper information, then spend scarce confirmatory amyloid measurements where they support the decision that will be reported. Amyloid positron-emission tomography (PET) remains one such protocol measurement for amyloid burden, but PET slots, trial budgets, and payer-facing evidence packages are finite. This paper asks a deliberately operational question: when is simple transparent PET validation enough, and when is a fitted residual-uncertainty score worth the added complexity? For a weighted protocol target, the first-order value of validating subject i is the product of target influence and residual protocol uncertainty. Generic uncertainty sampling uses only the second factor and can spend PET measurements on subjects that are hard to predict but weak for the scientific, clinical, or commercial claim. We apply this rule to the A4/LEARN PET archive, treating observed PET as a design laboratory for scarce-confirmation studies. For the primary APOE4 carrier versus non-carrier contrast in Centiloid 24-or-higher PET positivity, simple APOE4-balanced validation recovers nearly all of the target-specific gain: at PET budget 200, the confidence-interval width ratio relative to random validation is 0.923 for APOE4 balancing and 0.914 for target-specific scoring, while generic uncertainty sampling is 0.980. Other targets behave differently: target-specific scoring gives larger gains for an age-slope analysis and for cutoff-indexed PET positivity. The practical message is simple: spend scarce protocol measurements according to the claim being validated, not only according to prediction uncertainty. 

---
# Joint Causal Structure and Cluster Discovery Using Variational Inference 

**Authors**: Avni Rajpal, Anubhav Kumar, Rishabh Karnad, Mohammad Emtiyaz Khan, P.K. Srijith  

**Link**: [PDF](https://arxiv.org/pdf/2608.22212)  

**Abstract**: Causal discovery aims to understand the relationships between individual random variables. In many applications, such as brain imaging and climate modeling, it is more meaningful to consider interactions among groups of variables. Existing methods assume that knowledge of such groups or clusters is explicitly available when modeling interactions. However, in practice, these clusters as well as the causal relationships among them, are latent. In this paper, we present a novel approach based on variational inference to simultaneously infer both the latent clusters and causal structures. We learn an approximate posterior over clusters and graph-structure by considering variational distributions based on categorical and Bernoulli models respectively. We derive variational lower bounds and estimation techniques to learn variational and model parameters. The effectiveness of our proposed methods for cluster and causal discovery are demonstrated on both synthetic and real data sets. 

---
# Why Does Robustness Reduce Superposition? 

**Authors**: Adam Elimadi  

**Link**: [PDF](https://arxiv.org/pdf/2608.22155)  

**Abstract**: The study of adversarial examples and their origins remains an open area of research. Mechanistic interpretability, and superposition in particular, offers new avenues for approaching this problem. Gorton & Lewis (2025) demonstrate that adversarial examples arise from superposition and show empirically that adversarial training reduces superposition, yet provide no mechanistic account of why this occurs. We present an empirical explanation inspired by the feature taxonomy of Ilyas et al. (2019), tracing the following chain of causalities: adversarial training abandons non-robust features, leading to fewer total features to represent, resulting in less superposition. 

---
# Meta-Ctrl: Guaranteed Plan Generation by Decoupling Syntactic and Semantic Constraints 

**Authors**: Gwen Yidou-Weng, Edward Sun, Tianyi Ma, Metin Alp Dogan, Benjie Wang, Allen Peng, Guy Van den Broeck, Yuchen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2608.22149)  

**Abstract**: LLMs generate fluent plans for robots but routinely violate the syntactic and se8mantic constraints they must satisfy to execute, and existing remedies trade formal guarantees against plan quality: soft methods (affordance scoring, grounded decoding) give no guarantee, while symbolic planners (LLM+P) discard the LM's commonsense. We propose \textbf{Meta-Ctrl}, a constrained-decoding framework that guarantees the encoded constraints while preserving the base LM's plan quality. Meta-Ctrl introduces \emph{meta-tokens}---a compact vocabulary of grounded actions---enforcing syntax at the token level and semantics (preconditions, goals, ordering) at the action level, an exact factorization that cuts the memory of constrained decoding from over 107TB to under 2GB. With it, a small open-weight LM becomes competitive where it otherwise sits at the bottom of the leaderboard: on WAH-NL under the LoTa-Bench protocol it reaches the highest reported subgoal success rate, exceeding GPT-4's, with consistent gains across the Embodied Agent Interface. We further demonstrate it on a real tabletop robot, where every generated plan satisfies its preconditions and goals by construction. Project website: this https URL. 

---
# Lexical Perturbations Disrupt LLM Reasoning: An Empirical Study of Attention Diversion 

**Authors**: Jiaqian Zhu, Yang Zhang, Junhua Ding, Xiaowei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.22140)  

**Abstract**: Large Language Models (LLMs) achieve strong reasoning performance, but their robustness to realistic lexical corruption remains poorly understood. We evaluate four open-weight instruction-tuned models and frontier models across four reasoning benchmarks under keyboard noise, character swaps, and filler insertion. Character-level perturbations substantially degrade accuracy, especially on multi-step reasoning tasks, while filler insertion has little effect. We trace this asymmetry to Attention Diversion: lexical corruption fragments subword tokenization, and the resulting fragments attract disproportionate attention mass, concentrated in middle and final transformer layers. Length-matched controls confirm that fragmentation, not prompt length, drives the loss. A factorial intervention then shows why the damage is hard to undo: fragmentation corrupts token content and attention allocation together, and the two are coupled. Restoring clean attention while the content remains corrupted is actively harmful, restoring content alone is insufficient, and only restoring both recovers a substantial share of the gap. This coupling explains why inference-time strategies, including chain-of-thought prompting, spell-checking, self-repair, and stronger repair models, fail to consistently recover performance: each addresses one channel at a time. Code and data are available at this https URL 

---
# SSE-Bio: A Structured Self-Evolving Agent with Agentic Retrieval Policy for Multi-Hop Biomedical Reasoning 

**Authors**: Zhaohan Meng, Zaiqiao Meng, Siwei Liu, Hao Xu, Ke Yuan, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2608.22132)  

**Abstract**: Biomedical multi-hop question answering (QA) requires models to connect evidence across intermediate entities such as diseases, drugs, proteins, and phenotypes. Existing agents typically rely on static retrieval workflows or coarse-grained prompt rewriting, which can lead to instruction drift when reasoning procedures need to be updated. We propose SSE-Bio, a structured self-evolving agent with an agentic retrieval policy for multi-hop biomedical reasoning. Instead of globally rewriting agent instructions, SSE-Bio maintains a structured state, selectively retrieves knowledge triplets and prior templates through a trainable proxy policy, and improves its reasoning memory through fine-grained template editing. To optimise retrieval decisions, we introduce a proxy-training strategy based on group relative policy optimization, where the proxy is improved through decision-contrastive groups over alternative retrieval choices. Experiments on three biomedical multi-hop QA benchmarks show that SSE-Bio consistently outperforms existing baselines, achieving an improvement of 6.56 absolute points over the strongest self-evolving baseline on BioHopR. 

---
# TRACE: Artifact-Robust Statistical Shape Modeling from Imperfect Surface Scans - A Case Study in Craniosynostosis 3D Photography 

**Authors**: Sanjay Bhandari, Nawazish Khan, Alzbeta Novotna, Tiffany Jeong, Loretta Bowman, Michael Hernandez, Tobi Somorin, Viraj Govani, Jesse Goldstein, Shireen Elhabian  

**Link**: [PDF](https://arxiv.org/pdf/2608.22131)  

**Abstract**: Craniosynostosis severity analysis increasingly relies on statistical shape models (SSMs) to quantify cranial morphology, but most existing workflows depend on computed tomography or heavily curated three-dimensional (3D) photographs. Raw clinical 3D photographs provide a radiation-free and repeatable alternative, yet often contain shoulders, hands, hair, clothing, scanner noise, and incomplete boundaries that corrupt correspondences. We introduce the Template-constrained Robust Artifact-aware Correspondence Estimation (TRACE) framework, an unsupervised method for constructing SSMs directly from artifact-contaminated clinical 3D head photographs. TRACE predicts sparse anatomically corresponding head-surface control points from the raw point cloud, refines them through a coarse-to-fine Surface-Aware Deformation cascade, and uses thin-plate spline warping to deform a clean template mesh into a subject-specific head reconstruction. This template-constrained formulation keeps dense correspondences on clinically relevant head anatomy while suppressing non-head artifacts. The correspondence module is decoupled from the point-cloud encoder, enabling the same deformation pipeline to be paired with different backbones, including PointNet, DGCNN, and Point Transformer V3. Across all backbones, TRACE substantially improves surface sampling, topology preservation, and shape-model quality over prior SSM methods, providing a scalable foundation for photograph-based craniosynostosis shape analysis and a framework that may extend to other artifact-contaminated surface scans when an appropriate clean template is available. 

---
# Learning Implicit Constitutive Laws for Dynamic 3D Gaussian Splatting from Monocular Videos 

**Authors**: Xiaoyang Liu, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.22102)  

**Abstract**: We present GCA (Gaussian Constitutive Alignment), a framework for learning implicit constitutive laws from monocular dynamic video of deformable objects represented by 3D Gaussians. Given a static multi-view scan for geometric initialization, our method learns intrinsic physical dynamics solely from a single fixed-viewpoint video of the moving object. Existing implicit methods often suffer from local minima under noisy supervision and lack physical interpretability, while explicit approaches rely on predefined constitutive equations, limiting generalizability and becoming unstable in monocular settings. To address these challenges, our framework unifies LoRA-based adaptation with two key alignment modules. First, we propose Rank-based Depth-Geometric Anchors (RDGA) to establish robust geometric constraints from monocular dynamic observations via scale-invariant rank-based depth alignment, reducing the reliance on unreliable pixel-level color supervision. Second, a Constitutive Prior Regularizer (CPR) integrates classical constitutive models as soft differentiable priors, regularizing the optimization while preserving the flexibility of implicit modeling---even when the actual material is absent from the hypotheses. Extensive experiments on synthetic, real-to-sim, and real-world datasets demonstrate that GCA outperforms existing methods, achieving 48% lower Chamfer Distance than the strongest baseline on synthetic benchmarks while remaining robust under monocular supervision. 

---
# Semantic Reasoning Denoising: Correcting Language Model Reasoning with Semantic Operators 

**Authors**: Yujiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.22090)  

**Abstract**: Large language models can produce fluent reasoning traces whose local semantic errors propagate to an incorrect conclusion, while unconstrained self-correction may preserve, amplify, or introduce errors. Existing diffusion language models provide iterative refinement, but usually define noise as token masking or replacement rather than as errors in the reasoning process. We present Semantic Reasoning Denoising (SRD), an operatorized Markov denoising method for natural-language reasoning trajectories. SRD represents semantic noise with executable error operators that describe the error type, its location, and the corrupted and repaired propositions. Composing these operators constructs progressively noisier states. During training, the model learns to identify the semantic noise active in the current trajectory and to reconstruct the paired adjacent lower-noise state. During inference, noise-level-aware denoising repeatedly predicts an inverse operator and checks whether it is applicable, so each executed update makes a localized move toward a stable trajectory. Across six in-domain benchmarks spanning mathematics, code, knowledge, and commonsense, SRD improves the strongest same backbone baseline by 3.2 points on average. On seven cross-dataset transfer targets, it remains competitive with Llama-3-8B-Instruct and improves the strongest Qwen3-8B baseline average by 2.9 points. Analyses of noise sources, objectives, and denoising depth further show that structured semantic-noise prediction and iterative operator execution are central to the improvement. 

---
# On Predicting Vulnerability Severity Using In-Context Learning: An Industrial Case Study 

**Authors**: Daniel Rodriguez-Cardenas, David Nader Palacio, Anna Schmedding, Yiyang Lu, Aadil Mallick, Bill Hudson, Chris Gourley, Michael Roytman, Chris Shenefiel, Evgenia Smirni, Denys Poshyvanyk  

**Link**: [PDF](https://arxiv.org/pdf/2608.22089)  

**Abstract**: Modern software systems require earlier and more scalable vulnerability severity assessment to reduce exposure to high-impact security flaws. Security analysts typically assign CVSS scores, but this manual triage does not scale with the growth of disclosed vulnerabilities and often depends on cloud LLM services that raise confidentiality concerns. This paper presents an industrial case study on predicting CVSS v3.1 scores directly from vulnerable C/C++ snippets using in-context learning with locally deployable, open-source LLMs. We compare proprietary data with the Big-Vul dataset, showing sufficiently aligned CVSS distributions to justify Big-Vul as a proxy for industrial data when constructing prompt-based testbeds. We then vary in-context configurations and model parameters, evaluating CodeLlama2-7B, CodeLlama2-13B, Mistral-7B, gpt-oss, and GPT4o-mini using mean squared error (MSE) and feasibility metrics. Our results show that medium-sized open-source code models, particularly CodeLlama2-7B, can approximate the best cloud performance for CVSS regression when guided by lightweight, output-constraining prompts, offering a practical, privacy-preserving building block for severity triage in industrial settings. 

---
# Improving Energy Efficiency of Oil Platforms Through Optimal Loading of Diesel Generators Using Machine Learning and Search Algorithms 

**Authors**: Khivishta Boodhoo, Josh Plumbly, Nicholas Watson  

**Link**: [PDF](https://arxiv.org/pdf/2608.22076)  

**Abstract**: Rising energy demand, fossil fuel depletion and climate change highlight the need for more efficient energy production and consumption. Offshore oil and gas platforms face challenges related to inefficient energy use, system failures, accessibility and environmental impact. Machine learning (ML) offers opportunities to improve the safety, sustainability and efficiency of these systems; however, previous research has largely focused on increasing oil production rather than reducing energy consumption on platforms. This study investigates the use of ML and search algorithms to improve diesel efficiency on an offshore oil platform. Data collected over 18 months from a platform in Scotland were analysed, focusing on four diesel generators as the primary diesel-consuming equipment. Following exploratory data analysis and outlier detection, regression models were developed to predict daily diesel consumption for different generator power loads. Multiple Linear Regression and Artificial Neural Networks achieved the best predictive performance compared with Extra Trees Regression, Extreme Gradient Boosting and Random Forest. Search algorithms were then used to identify combinations of generator power loads that minimised daily diesel consumption. The results showed an average diesel saving of 27% per day compared with the worst daily power-load combinations, equivalent to approximately 24,000 litres/day. These findings demonstrate significant opportunities for improving energy efficiency on offshore oil platforms using ML-based optimisation. 

---
# Real-TurnTurk: A Multimodal Turkish Corpus for Turn-Taking Prediction 

**Authors**: Ahmet Tuğrul Bayrak, Fatma Nur Korkmaz, Bekir Berker Türker, Mustafa Sertaç Türkel, Alper Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2608.22071)  

**Abstract**: Turn-taking is a basic organizational feature of human conversation and remains difficult to model in natural, synchronous dialog systems. While existing research has explored multimodal approaches and large language models for turn-ending prediction, there is a lack of naturalistic conversational corpora specifically addressing turn-taking dynamics in Turkish. This study introduces a multimodal Turkish conversational dataset of unscripted dyadic interactions, comprising synchronized front-facing video, per-speaker audio channels that allow overlapping speech to be attributed to individual speakers, and time-aligned transcriptions. Turn-taking prediction is formulated as a binary classification problem, and a Genetic Algorithm (GA) is employed to optimize interpretable decision rules derived from visual, acoustic, and linguistic features. A hybrid AND-OR rule representation is adopted in the proposed framework to represent the alternative cue combinations that precede a turn transition. 

---
# Inferring Action from Future Latent State for Robotic Manipulation 

**Authors**: Fenghao Lei, Zhixiong Huang, Long Yang, Jiabao Chen, Jie Cheng, Peilin Huang, Han Fu, Zhuo Li, Xiaoxue Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.22067)  

**Abstract**: World-Action Models (WAMs) build robot control on video-generation backbones, which jointly predict dense future visual trajectories and robot actions. We argue that video generation is an unnecessary intermediate objective for world-action modeling. For robotic manipulation, the goal of a world model is not to reproduce how the world looks at every intermediate moment, but to predict the state that the world will reach after an action is executed. The intermediate frames only describe the visual transition between physical states, which consumes substantial model capacity and computation, but do not directly specify the physical outcome that the robot action is intended to produce. In this paper, we propose DELE-w0.5, which infers robot actions from predicted future states without relying on video generation. Concretely, DELE-w0.5 infers the action sequence from its corresponding compact future latent state. The future latent state captures the action-relevant physical outcome of robot interaction and serves as an explicit bridge between world modeling and action generation. The core design principle of DELE-w0.5 is to model how the physical world changes under robot actions, rather than how its visual appearance evolves frame by frame. This formulation removes the high-dimensional visual redundancy introduced by dense video representations, and it therefore enables cheaper training and low-latency inference. Across 480 real-robot trials on four long-horizon manipulation tasks, our DELE-w0.5 achieves the best performance among all compared policies, attaining 62.5 overall full-task success and 81.3 macro ordered-stage progress, outperforming the strongest baseline by 47.5 and 30.7 percentage points, respectively. 

---
# ADMIL: Attention-Distilled Multiple Instance Learning for Selective Foundation Model Inference in Pathology 

**Authors**: Duncan Stothers, Ren-Chin Wu, William Lotter  

**Link**: [PDF](https://arxiv.org/pdf/2608.22066)  

**Abstract**: Attention-based multiple instance learning (ABMIL) using pathology foundation model embeddings is effective for slide-level tasks, but exhaustive inference requires applying a large image encoder to every foreground tile despite the subsequent attention distribution often concentrating over a small subset of informative regions. We introduce ADMIL (Attention-Distilled Multiple Instance Learning), a selective-compute framework that distills an ABMIL teacher's attention into a lightweight tile-selection model, PriorNet. Using an EfficientNet architecture, PriorNet learns the teacher attention distribution from raw tile pixels with KL divergence; at inference, it scores the foreground pool, selects the top-K tiles, and invokes the expensive foundation model only on that subset before a selected-bag ABMIL student predicts the slide label. Across BRACS, PANDA, and CAMELYON16, ADMIL matches full-teacher headline performance at K=4, 8, and 128 tiles, respectively, avoiding >98% of foundation model (Virchow2) tile embeddings and model inference FLOPs. Random and teacher-attention oracle controls show that this result depends on task-relevant selection rather than tile-count reduction alone. Quantitative and qualitative analyses suggest that PriorNet recovers the teacher's tile ordering with high fidelity while focusing on task-relevant morphological regions. ADMIL shows that nearly all expensive tile encodings can be removed without sacrificing slide-level performance, providing a potential path for more efficient deployment in clinical settings where latency and compute costs are key considerations. 

---
# Discovering Dual-Origin Slow Wind from Solar Orbiter with Self-Supervised Contrastive Learning 

**Authors**: Henry Han, Jorge Yero Salazar  

**Link**: [PDF](https://arxiv.org/pdf/2608.22065)  

**Abstract**: Whether the slow solar wind originates from one coronal source or two distinct channels remains a central open question in heliophysics. Resolving this requires unsupervised separation of two populations that arrive at nearly the same bulk speed and differ mainly in heavy-ion composition. We present Solar-CDC, a self-supervised contrastive deep clustering (CDC) framework that maps plasma observables to a latent space via a Transformer encoder, optimizes a triplet margin loss, and updates pseudo-labels via $k$-means. Theoretically, we prove that neighborhood-preserving embeddings such as t-SNE and UMAP are fundamentally constrained. Preserving the neighbor graph leaves the cross-cluster cut fraction unchanged, and preserving all but a fraction $\varepsilon$ of its links moves that fraction by at most $\varepsilon$. Neither bound depends on the target dimension. A margin objective rewrites the graph and drives the cut fraction to zero. Empirically, on 30,602 Solar Orbiter observations, thirty combinations of dimensionality reduction and clustering peak at a silhouette of $0.454$, whereas Solar-CDC reaches $0.869$. Escaping the geometric bound alone does not guarantee physical validity: TriMap also optimizes triplets and reaches $0.824$, yet its clusters score below chance against the published composition taxonomy. Solar-CDC instead recovers clusters with mean charge-state ratios of $0.080$, $0.160$, and $0.400$, placing the intermediate population inside the window associated with coronal-hole boundaries. Even when the defining charge-state ratio is withheld from the inputs entirely, the model still recovers the taxonomy defined on it. Solar-CDC thus connects self-supervised representation learning to coronal source diagnostics. Importantly, a learning loss recovers physical populations only when driven by dynamically updated physically-aware clusters rather than distances. 

---
# CRS-Bench: A Reference-Relative Reliability Benchmark for Medical Image Encoders 

**Authors**: Xingtao Lin, Hangqi Ren, Caiwan Sun, You Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.22059)  

**Abstract**: Pretrained image encoders are central to medical image classification, where expert annotation is costly and task-specific cohorts are often limited. As the model space expands from general-purpose to broad-medical and specialty-specific encoders, selecting the representation becomes a substantive modeling decision. Clean-test discrimination alone is insufficient for this purpose: encoders with similar AUROC can differ in calibration, label efficiency, and stability under acquisition perturbations or distribution shift.
We introduce CRS-Bench, a controlled benchmark for multi-objective medical encoder selection. CRS-Bench evaluates 15 pretrained encoder families across dermatology, ophthalmology, and radiology using ISIC 2019, APTOS 2019, and CheXpert, with CheXpert-to-MIMIC-CXR as an observed institutional shift, yielding 17,575 controlled run records and 3,515 seed-aggregated metric rows. Each encoder is characterized along four operational reliability dimensions: discrimination, calibration, label efficiency, and robustness. We summarize these dimensions using the Clinical Reliability Score (CRS), a Pareto-aware, reference-relative score combining dominance, profile balance, and worst-axis performance.
AUROC and CRS are positively associated but not decision-equivalent: 21 of 105 pairwise orderings reverse, with a mean absolute rank displacement of 1.87. Paired-seed bootstrap analysis identifies PanDerm, MedSigLIP, and MedGemma as a stable leading reliability tier rather than a statistically resolved single leader. CRS-Bench provides a controlled framework for selecting medical image encoders from multi-axis reliability profiles rather than clean-test AUROC alone. 

---
# Multi-Agent Discovery and Resource-Aware Autonomous Exploration of Scientific Datasets 

**Authors**: Aashish Panta, Hugo Lee, Giorgio Scorzelli, Kyongsik Yun, Valerio Pascucci  

**Link**: [PDF](https://arxiv.org/pdf/2608.22045)  

**Abstract**: Modern scientific facilities and instruments generate datasets at scales that are difficult for individual researchers to discover, access, and explore. Although many datasets are publicly available, using them often requires familiarity with repository organization, data formats, multiresolution structures, and visualization parameters. We present WebVisus, a constrained and resource-aware multi-agent system for discovering and autonomously exploring remote, multiresolution scientific datasets. Given a natural-language research question, WebVisus identifies the user's intent and launches an autonomous exploration agent that examines slices, volumes, and timesteps while adapting data resolution and retrieval quality to available client memory and computational resources. This design supports progressive exploration without complete dataset downloads or manual configuration of low-level visualization parameters using natural languages. We report the system architecture, constrained agent protocol, resource-aware access mechanism, and case studies evaluating autonomous visual exploration and resource-aware agentic access across scientific datasets. 

---
# Barycentric Fused Gromov-Wasserstein Balancing for Causal Inference under Multiple Treatments 

**Authors**: Yuki Murakami, Takumi Hattori, Kohsuke Kubota  

**Link**: [PDF](https://arxiv.org/pdf/2608.22024)  

**Abstract**: Estimating heterogeneous single and interaction treatment effects from observational data under multiple simultaneous treatments is crucial for decision-making. To mitigate estimation variance, previous studies balance representation distributions between every pair of treatment patterns. However, such pairwise balancing scales quadratically with the number of treatment patterns and fails to preserve consistent local proximity structures across patterns, which degrades counterfactual estimation. To address these challenges, we propose the Causal Inference for Heterogeneous Single and Interaction Treatment Effects Network (CIHSI-Net), a deep learning framework built on a novel Barycentric Fused Gromov-Wasserstein Balancing (BFG-WB) objective. BFG-WB aligns the representation distribution of each treatment pattern with a shared Wasserstein barycenter, achieving global alignment while reducing the computational complexity from quadratic to linear, and its Fused Gromov-Wasserstein discrepancy preserves the local proximity structures essential for reliable heterogeneous effect estimation. Simulation studies show that CIHSI-Net consistently outperforms state-of-the-art baselines, and an application to real-world marketing data demonstrates its practical utility in complex multi-treatment scenarios. 

---
# Variance Driven Exploration: A Provable and Efficient Methodology for Pure Exploration in Highly Stochastic Environments 

**Authors**: Khang Luong, Nam Nguyen, Hoang Ta, Hung The Tran, Tuan Dam  

**Link**: [PDF](https://arxiv.org/pdf/2608.21995)  

**Abstract**: We propose Variance Driven Exploration (VarDE), a principled approach for pure exploration in highly stochastic environments, where the exploration process is dominated by stochastic variance. VarDE is built on a fundamental principle: sampling effort should be allocated to minimize the uncertainty of the final decision. We formalize the uncertainty of the final decision through a smooth decision function and derive allocation rules that explicitly capture how stochastic noise in individual components affects the reliability of the final output. We apply this methodology to three core problems of pure exploration -- Best Arm Identification (BAI), Monte Carlo Tree Search (MCTS), and Best-Policy Identification (BPI) -- with theoretical guarantees on variance decay and simple regret. Empirically, we demonstrate consistent and significant improvements of VarDE over existing methods, with especially strong gains in highly stochastic environments. 

---
# NoTB: Oracle-Free Triage of LLM-Generated RTL via Cross-Model Formal Consensus 

**Authors**: Elisavet Lydia Alvanaki, Je Yang, Biruk Seyoum, Luca P. Carloni  

**Link**: [PDF](https://arxiv.org/pdf/2608.21962)  

**Abstract**: Large language models (LLMs) are increasingly used to generate register-transfer-level (RTL) designs from natural-language specifications. However, assessing functional correctness at early stages remains a fundamental challenge. Existing oracle-free approaches rely either on simulation-based agreement, which depends on LLM-generated testbenches that can fail or vary across models, or on LLM-as-a-judge heuristics, which produce inconsistent predictions. We introduce NoTB, an oracle-free triage framework that infers correctness from cross-model formal consensus. NoTB generates RTL implementations from multiple independently trained LLM families and applies Sequential Equivalence Checking (SEC) to identify designs that are provably equivalent. We show that the diversity of model families within an SEC-equivalent cluster induces a calibrated correctness signal, enabling risk-coverage tradeoffs without requiring testbenches. On 78 CVDP RTL-generation tasks, four-family formal consensus achieves 94.7% precision at 27% coverage; three-family consensus achieves 87% precision at 33% coverage. These operating points give designers a tunable accept/defer rule before a trusted testbench or golden RTL is available. Overall, NoTB demonstrates that formal cross-model agreement provides a reliable basis for high-confidence triage without model-dependent oracles 

---
# Bulbul: A Dataset for Dialectal Arabic Speech Recognition 

**Authors**: Ahmed Ashraf, Aisha Alansari, Fadel Al Abbas, Nada Almarwani, Samah Aloufi, Saad Ezzini, Maged S. Al-Shaibani, Doaa Dalaq, AbdelRahim A. Elmadany, Muhammad Abdul-Mageed, Mohamed Mehdi Trigui, Dania Refai, Layan Refai, Mohamed Akrout, Mustafa Jarrar, Wasfi G. Al-Khatib, Alaa Dalaq, Darin El-Nakla, Samir Abdaljalil, Abdulrahman Al-Fakih, Nour El Imane Zeghib, Moussa Redah, Salmane Chafik, Mohamed El-Attar, Rima Grati, Sarah Kohail, Malak Alkhorasani, Khadijah Al Safwan, Ismail M. Mudhaffar, Ali Altam, Ahmed Al-Shaikh, Adnan Saeed, Hamzah Luqman  

**Link**: [PDF](https://arxiv.org/pdf/2608.21950)  

**Abstract**: Arabic automatic speech recognition (ASR) faces unique challenges due to diglossia, extensive regional dialect variation, and limited speech resources. Existing speech datasets often focus on single dialects or large-scale broadcast/web data, leading to trade-offs between linguistic diversity and annotation quality. We present BULBUL, a multi-dialect Arabic ASR dataset collected from 275 speakers in 11 Arab countries. BULBUL includes structured dialect and sub-dialect coverage, as well as recordings of classical Arabic and modern standard Arabic spoken by participants in their native dialectal accents to support accent-aware modeling. The quality of the recordings was ensured through a two-level human verification process. We further benchmark a range of recent ASR systems, establishing strong baselines for modern dialectal and accented Arabic ASR. 

---
# EDGE: Experience-Distillation for Guided Exploration in Agentic Reinforcement Learning 

**Authors**: Can Xie, Yuyi Zhou, Wen Yang, Ziyi zhang, Siyao Song, Yingzhuo Deng, Shuo Ren, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.21946)  

**Abstract**: Reinforcement learning with outcome-based objectives such as GRPO enables LLM-based agents to solve complex, long-horizon tasks, yet the reusable exploration patterns embedded in interaction trajectories are largely discarded after a single policy update. Existing experience-augmented approaches retrieve historical guidance at inference time, but they apply experiences without accounting for the policy's evolving capability and create persistent dependencies on external retrieval. We propose EDGE (Experience-Distillation for Guided Exploration), a framework that treats retrieved experiences as temporary training-time scaffolds and progressively internalizes their benefits into the parametric policy. Concretely, EDGE partitions each rollout group into experience-conditioned and experience-free trajectories to estimate and admit only positive marginal gains without extra sampling, then distills the induced behavior into the base policy via a reverse-KL objective on its own empirical support. A co-evolutionary experience bank further synthesizes guidance from emerging failure modes and prunes obsolete entries as the policy evolves. On ALFWorld and WebShop, EDGE improves over GRPO by 8.3 and 12.5 success-rate points at the 7B scale and retains 96.0% of its scaffolded performance when external experiences are removed at inference time. The code is available at this https URL. 

---
# Bi-EZP: LLM-Guided Bilevel Program Evolution for Ensemble Zero-Cost Proxy Discovery 

**Authors**: Yutao Lai, Kezhao Lai, Hai-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21927)  

**Abstract**: Zero-cost proxies enable neural architecture search (NAS) to rank candidate networks from statistics computed at initialization, avoiding repeated training. However, different proxies capture different properties and often produce inconsistent rankings across search spaces. Ensemble proxies can combine complementary signals, but automated discovery must optimize both discrete aggregation structures and their continuous coefficients, making structural quality difficult to separate from parameter calibration. We propose Bi-EZP, a bilevel framework that decouples these decisions. At the upper level, a large language model generates executable aggregation programs over four complementary base proxies with program-specific parameter bounds. At the lower level, covariance matrix adaptation evolution strategy (CMA-ES) optimizes the continuous parameters of each fixed program on an inner training split. The calibrated programs are then evaluated using Kendall's rank correlation on a disjoint validation split, enabling evolutionary selection to favor structures that generalize beyond their calibration data. Experiments on NATS-Bench and Network Design Spaces evaluate ranking performance across heterogeneous search spaces, and DARTS experiments assess downstream architecture search. Results show that separating program discovery from numerical calibration provides an effective approach to automated ensemble zero-cost proxy construction. The source code is available at: this https URL 

---
# Breaking the Assumptions: Auditing Input-Side Jailbreak Defenses Against Semantic Attacks 

**Authors**: Aaditya Pratap, Harsh Kasyap, Somanath Tripathy  

**Link**: [PDF](https://arxiv.org/pdf/2608.21895)  

**Abstract**: Locally deployed Large Language Models (LLMs) via inference engines such as Ollama run without the moderation and abuse detection present in API-served models. Therefore, the safety of LLMs depends on the defense mechanisms used, and their effectiveness depends on the assumptions on which they were designed. This paper does an audit of defense mechanisms under jailbreak attacks on locally deployed models. Some defenses provide formal guarantees (SmoothLLM, Erase-and-Check, Sequential Monitors), while others rely on empirical detection results (Semantic Smoothing, Self-Denoised Smoothing, Perplexity Filtering). Instead of merely observing that defenses fail, we trace each failure back to the specific assumption: for every defense, we extract the condition it relies on, derive the empirical pattern a violation should produce, and test that prediction on six open-weight models (14B to 35B parameters) with a corpus of 100 jailbreak prompts taken from more than 40 public sources, totalling 13,800 evaluation records. 

---
# A Scalable Vector Graphics Latent Space 

**Authors**: Leonardo Zini, Elia Frigieri, Lorenzo Baraldi  

**Link**: [PDF](https://arxiv.org/pdf/2608.21893)  

**Abstract**: Scalable Vector Graphics are a fundamental medium for resolution-independent visual content, yet the deep learning community lacks a continuous, dense, and invertible latent space for vector representations, the kind of foundational building block that Variational Autoencoders and their descendants have long provided for raster images. We introduce SLS (SVG Latent Space), a Transformer-based autoencoder that learns compact dense representations of individual SVG paths, the atomic visual elements from which any SVG image can be composed. By modeling SVG commands, coordinate data, and visual properties within a unified BPE-based token vocabulary, SLS learns fixed-size latent representations that jointly capture structure and appearance, and can be decoded back into valid, style-consistent SVG paths with high fidelity. The resulting embedding space is robust, invertible, and structured: embeddings lie on a unit hypersphere, enabling efficient similarity search, composition, and downstream conditioning through simple vector-space operations. Finally, we demonstrate that SLS generalizes across diverse tasks reducing their FLOPs by over 150 times compared to token-based approaches, and establishing a general-purpose latent foundation for vector graphics research. 

---
# Pruned Traffic Trees: Native Semantic Compression with a Protocol-Structured Model Family for Encrypted Traffic Classification 

**Authors**: Yuantu Luo, Jun Tao, Xiangyu Xu, Linxiao Yu, Kangying Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.21874)  

**Abstract**: Deep learning has achieved strong performance in encrypted traffic classification (ETC), yet its computational cost limits deployment on resource-constrained network devices such as routers and middleboxes. Existing compression methods mainly operate on weights, channels, hidden representations, or predictions, but do not explicitly determine which protocol fields and structural contexts should remain. We propose Pruned Traffic Trees (PTT), a three-level protocol-structured model family that treats native protocol structures as compression units. PTT-Full learns protocol-structured representations and field salience from complete Protocol Tree Graphs (PTGs), with flow-level self-supervised learning and protocol-presence-aware sparse execution. The learned salience and TopK+$k$ closure construct Distilled PTGs (PTG-Ds) for PTT-Distilled, while PTT-Lite inherits this topology and reduces width through structure-aligned transfer and flow-level logits distillation. Under flow-disjoint and Strong Information Information (SII)-masked settings, PTT-Full achieves Macro-F1 scores of 0.9519 and 0.9416 on CSTNET-TLS1.3 and CipherSpectrum, while PTT-Lite retains 0.9325 and 0.9136 with 80.3\% and 61.3\% fewer parameters, 98.85\% and 98.78\% lower effective GFLOPs, and 8.75$\times$ and 8.46$\times$ CPU inference speedups. These results demonstrate that treating protocol structure itself as the compression object enables effective performance-efficiency trade-offs for lightweight ETC. 

---
# GuardPaint:SpeculativeSafetyDecodingforText-to-ImageGeneration 

**Authors**: Shreyash Dhoot, Paras Dhiman, Arsh Abbas Naqvi, Aranbi Dutta, Aman Chadha, Vinija Jain, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2608.21869)  

**Abstract**: Text-to-image (T2I) diffusion models offer powerful visual generation, but their controllability creates a critical safety challenge: adversarial prompts can steer the denoising trajectory toward policy-violating content such as explicit nudity or graphic violence. Existing safeguards mostly act before generation through prompt filtering or after generation through image classification, leaving the diffusion process itself unguarded and often yielding only refusal rather than safe visual repair.
We introduce GuardPaint, a speculative decoding framework for safe T2I generation that intervenes inside the diffusion trajectory without modifying the base model. A lightweight auditor monitors intermediate images, localizes unsafe regions, and triggers surgical inpainting repair only where needed. Candidate repairs are generated by a policy-aligned inpainter and selected through a guarded tournament that accepts edits only when they improve policy compliance while preserving prompt fidelity and perceptual quality.
Across five jailbreak families SneakPrompt, MMA, PGJ, DACA, and RABell and UNet/flow-matching models including SD~1.5, SDXL, SD~3.5, and FLUX.1-dev. GuardPaint reduces attack success and harmful generations with minimal degradation to image quality, prompt fidelity, and benign behavior.
Content warning: This paper contains examples involving nudity and violence that some readers may find disturbing, distressing, or offensive. 

---
# BioMed-Agent-RL: A Meta Learning, All You Need for Biomedical Applications 

**Authors**: Md Asaduzzaman Jabin, Zihao Wu, Tianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21864)  

**Abstract**: The current progress of Clinical Vision Large Language Models (C-VLLMs) has substantially improved digital diagnostics, still these frameworks often endure lesion noises, modality misalignment, hallucination, and missed contextual grounding in complex clinical cases. Moreover, prevailing agent systems usually depend on static and non-adaptable pipelines and lack the versatility necessary for complex medical reasoning. To resolve these difficulties, we present BioMed-Agent-RL, a unified medical agent that incorporates adaptive orchestration, policy, and reward-based reinforcement learning (RL) models for biomedical applications. To ensure reliability, it invokes clinical context-aware preference optimization (CPO), direct preference optimization (DPO), and group relative policy optimization (GRPO) with dynamic entropy regulation. This pipeline utilizes a multimodal meta-learning approach that operates as a field-specific expert and human judgment synthesizer. The agent adaptively utilizes a set of model-level expertise, such as clinical grounding and reasoner, lesion segmenter, and field-specific synthesizer, across various clinical modalities (e.g., X-ray) by utilizing an iterative and adaptive RL approach. The agent learns to seriously synthesize misleading, conflicting vision cues and trust in inherent reasoning, while specialist advice is faulty. An intensive ablation study is conducted across multiple benchmarks, and the agent significantly outperforms existing state of the art models, such as GPT-5, attaining up to ~73% accuracy (gain of ~5%) over contemporary baselines. As a result, the framework suggests a new standard for building factual, reliable, robust, and expert-like intelligent agent systems for independent clinical reasoning. 

---
# HiDiffTIR: Hierarchical Difficulty-Aware Policy Optimization for Multi-Turn Tool-Integrated Reasoning 

**Authors**: Yucan Guo, Xiaohan Wang, Miao Su, Saiping Guan, Zhongni Hou, Jiajun Chai, Wei Lin, Guojun Yin, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.21863)  

**Abstract**: Tool-Integrated Reasoning (TIR) is a fundamental capability for LLM agents to solve complex tasks by interacting with external tools iteratively. Reinforcement Learning (RL) has become the dominant paradigm for enabling this capability. However, existing approaches typically assign uniform trajectory-level advantages and treat all correct tool calls equally, ignoring the varying difficulty and learning value across trajectories and reasoning steps. This can lead to imprecise learning signals that do not adequately distinguish between trivial and challenging tool-use patterns. To address this limitation, we propose HiDiffTIR, a Hierarchical Difficulty-aware policy optimization framework for multi-turn TIR. HiDiffTIR performs difficulty-aware credit assignment at both trajectory and turn levels, enabling the policy to focus on more informative trajectories and harder reasoning steps. Notably, this fine-grained optimization is achieved without additional supervision, relying solely on group-level statistics derived from standard RL rollouts. Extensive experiments on three tool-using benchmarks demonstrate that HiDiffTIR consistently improves multi-turn TIR performance and tool invocation accuracy over strong RL baselines, highlighting the necessity of difficulty-aware credit assignment for effective policy optimization in tool-integrated LLM agents. 

---
# ChainPrune: Evaluating and Reducing Redundancy in Long Chain-of-Thought Reasoning 

**Authors**: Weihang Pan, Zhengxu Yu, Yuxiang Zhang, Wenzhi Li, Zhongming Jin, Binbin Lin, Xiaofei He, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2608.21860)  

**Abstract**: Chain-of-Thought (CoT) reasoning has significantly enhanced the multi-step problem-solving capabilities of large language models (LLMs) by introducing explicit intermediate reasoning. However, advanced Large Reasoning Models (LRMs) often exhibit overthinking behaviors, including excessively long reasoning steps, redundant steps, and high computational overhead. Existing token-length reward strategies aim to promote concise outputs, but often result in pseudo-conciseness, where token count is reduced, yet redundant reasoning persists, leading to longer and less structurally efficient chains. To address these limitations, we propose ChainPrune, a novel reasoning path semantic structural optimization method to efficiently and controllably synthesize self-generated high-quality training data. We initially consolidate self-generated reasoning paths into a tree-based structure, followed by a multi-criteria dominant path selection process for preference data construction that formulates shallow reasoning trajectories while preserving essential reasoning steps. To further enhance the quality of reasoning, we incorporate a DPO-based preference learning method combined with supervised loss, effectively mitigating false reward suppression. This innovative integration significantly enhances both the efficiency and effectiveness of our reasoning framework. Comprehensive experimental results demonstrate significant reductions in step length and computational overhead, while maintaining or even enhancing accuracy. 

---
# LLMs are Few-Shot Decision-Makers: Generalized Context-Aware Microgrid Frequency Control through Prompt Decision Transformer 

**Authors**: Xu Yang, Chenhui Lin, Haotian Liu, Kaihang Deng, Yunhe Li, Wenchuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21858)  

**Abstract**: The rapid evolution of energy structures has positioned microgrids as pivotal components of next-generation power systems, offering enhanced resilience and renewable energy integration. However, the inherent low inertia, complex dynamics, and poor model conditions of microgrids necessitate advanced data-driven frequency control strategies. Although reinforcement learning (RL) has demonstrated certain potential and advantages, existing RL methods often struggle with generalization across diverse microgrid configurations and lack adaptability to unseen environments, particularly when explicit system parameters are unavailable. To address these challenges, in this paper, we introduce a novel prompt decision transformer (Prompt-DT) architecture for microgrid frequency control. Unlike traditional approaches that rely on hard-to-obtain environmental characteristic parameters, the proposed method leverages few-shot expert historical trajectories as prompts to guide autonomous perception and adaptive decision-making. In addition, we propose a context-aware training and execution mechanism utilizing self-supervised contrastive learning to enhance environment recognition and prompt utilization efficiency. In addition, a physics-informed prompt design technique that filters prompts based on cumulative reward and frequency volatility is proposed, ensuring high-quality physical guidance during online execution. Finally, to ensure generalization in unseen environments with limited data, we develop a lightweight finetuning approach that achieves performance comparable to full-parameter finetuning with minimal adjustments. 

---
# Training a Knowledge Base: Supervised Structure Learning for Agent-Curated Document Stores 

**Authors**: Yu Pan, Hongfeng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21829)  

**Abstract**: Retrieval-augmented generation treats the document store as a frozen input, and the systems that instead let an agent curate one never measure what curation does to the store. We invert the framing: the knowledge base is the model. A training agent answers a supervised question against the current store, is shown the gold, then edits the store; an unchanged reader is later examined on a frozen snapshot under a fixed action budget. Where offline graph construction is unsupervised, (question, answer) pairs are our labels -- and that supervision is what makes the structure cheap. Per point of corpus indexed it returns 1.6x the action saving and 1.8x the accuracy of an unsupervised entity index covering everything, using 1,913 links against its 196,112. On questions the store trained on, an unchanged reader spends 31% fewer actions at higher accuracy, and the result reproduces on an official PhantomWiki generation whose questions we did not write. To measure how far this reaches we introduce a key-coverage gradient, a probe varying how much of a question the training set touched, replacing a train/test split's pass/fail with a decay curve. Generalization proves endpoint-dependent: accuracy carries to unseen questions (+0.167 F1 where both of a question's keys were indexed, +0.100 where one was, zero where neither) while the action saving stays on trained questions. Because that decay is indexed by coverage rather than by novelty, more training extends it -- and the store is undertrained, not saturated: coverage grows linearly in new questions and stops the moment training repeats them, so a hundred questions reach a quarter of the corpus and four times as many would close the gap. 

---
# PatchGate: Narrowing the Verbalization Gap with Intrinsic Object Inventories in Frozen Vision-Language Models 

**Authors**: Jihyung Ko, Eunji Jung, Hyeongsub Kim, Ziseok Lee, Jae Won Cho, Sanghyun Jo, Kyungsu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.21819)  

**Abstract**: Reliable image captioning in Vision-Language Models (VLMs) requires captions to be both precise and complete, avoiding unsupported object mentions while covering visible objects. Existing training-free methods primarily address the former requirement, suppressing unsupported object words by intervening on model-predicted mentions during generation. Because they operate only on objects the model is already likely to mention, visible objects omitted from the output remain difficult to recover. We propose PatchGate, a training-free framework that extracts prompt-free object evidence intrinsic to a frozen VLM before generation and uses it to narrow the gap between an intrinsic object set and final object mentions. In the first stage, Visual Evidence eXtraction (VEX) reads patch-level lexical evidence from the latter half of LM decoder layers and constructs an image-conditioned object set without any task prompt. In the second stage, Visual-Evidence Inclusion-Exclusion Decoding (VIED) uses this object evidence to calibrate decoding logits, promoting evidence-supported but under-verbalized objects and suppressing weakly supported but over-verbalized objects. On AMBER, PatchGate improves both sides of object-level reliability, increasing visible-object coverage from 49.4 to 56.0 (+13.4%) and reducing object hallucination by lowering CHAIR from 7.5 to 6.6 (-12.0%), without external detectors or fine-tuning and with one extra forward pass. 

---
# More Computational Resources Do Not Ensure Higher Scholarly Impact: Evidence from Leading NLP Conference Papers 

**Authors**: Shuai Chen, Tong Bao, Jitong Peng, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.21806)  

**Abstract**: Computational resources are increasingly central to NLP research, but how closely reported GPU capability aligns with scholarly impact remains unclear. We analyze 13,921 ACL, EMNLP, and NAACL main-conference papers published between 2020 and 2025, using GPU resources as our operational measure of computational resources. From full texts, we extract GPU models and counts, standardize each paper's largest reported configuration into a comparable hardware-capability measure, and link these data to citation, award, topic, and institutional metadata. GPU reporting became more common but remained incomplete, while reported capability increased mainly through newer hardware generations and medium-scale multi-GPU configurations. Resource concentration substantially exceeded impact concentration: the annual top 20% of GPU-quantifiable papers accounted for 83.9%-89.9% of reported GPU capability, but only 27%-32% of citations and 20%-33% of paper awards. In adjusted models, a tenfold increase in aggregate reported GPU capability was associated with a 3.52-percentage-point increase in within-NLP topic-year citation percentile, but increased model R^2 by only 0.0042. GPU count showed more consistent positive associations with citation and award outcomes than newer hardware generation. Overall, reported GPU resources are associated with scholarly impact but provide little standalone explanation of research influence. 

---
# ExplainGuard: A Zero Trust Framework for Post-Hoc Explanation Integrity Guarantees in Blackbox XAI Models 

**Authors**: Maraz Mia, Shovan Roy, Mir Mehedi A. Pritom, Maanak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2608.21803)  

**Abstract**: As machine learning (ML) models are increasingly deployed in high-stakes environments, explainable AI (XAI) methods like SHAP and LIME have become essential for regulatory compliance and trust. However, the current auditing paradigm relies on an implicit "chain of trust" where third-party auditors are assumed to be trusted. Recent research demonstrates that this assumption is flawed and adversarial auditors can manipulate XAI explanations through manipulation attacks such as output shuffling or scaffolding out-of-distribution (OOD) to conceal model biases while maintaining high prediction accuracy aiming for fairwashed explanation. In this paper, we introduce a novel defense framework, ExplainGuard, that leverages a Zero-Trust architecture (ZTA) design to be incorporated within the XAI explanation supply chain and ensures the integrity of the generated explanation. This framework would help us to replace the ambiguous default assumption of "auditor is trustworthy," with a continuous "verify-then-trust" approach. Our design architecture establishes a Policy Decision Point (PDP) that enforces three distinct pillars of verification before any explanation is released to the user: (1) asset integrity via behavioral fingerprint to detect model substitution, (2) semantic validity using axiomatic consistency checks to reject mathematically impossible explanations, and (3) feature faithfulness verification utilizing a ranking stability approach with minimal computational overhead. Finally, we evaluate how ExplainGuard can effectively neutralize state- of-the-art explanation manipulation attacks while transforming the auditing process into a verifiable operation. 

---
# SAFE-G: Structure-aware Faithful Evidence-guided Generation for Knowledge-based Visual Question Answering 

**Authors**: Long Shu, Shuochen Liu, Wei Chen, Junda Lin, Zhi Zheng, Huijun Hou, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21796)  

**Abstract**: Knowledge-based Visual Question Answering (KB-VQA) aims to answer queries that necessitate reasoning over external knowledge sources beyond the visual content. Typically, current methods fuse multimodal features to retrieve external information, subsequently leveraging Multimodal Large Language Models (MLLMs) to derive answers from the retrieved evidence. However, these methods often struggle to capture structural associations within complex contexts to effectively filter noise. Furthermore, they frequently fail to ensure that the reasoning process remains strictly faithful to the retrieved evidence. To address these challenges, we propose SAFE-G, a Structure-Aware Faithful Evidence-guided Generation framework, which enables precise evidence localization and trustworthy reasoning. Specifically, we first employ a coarse-grained hybrid search fusing visual and textual modalities to recall candidate documents, and subsequently implement a structure-aware fine-grained graph retrieval that captures structural dependencies to filter noise and pinpoint precise evidence. Moreover, we introduce a reinforcement learning (RL) strategy with an evidence-grounded reward that assigns credit to correct answers only when the selected evidence is correct. This strict alignment constraint compels the model to anchor its response in the retrieved context, effectively enhancing its capability to locate evidence via multimodal features and perform faithful reasoning. Extensive experiments on the Encyclopedic-VQA and InfoSeek benchmarks demonstrate that SAFE-G outperforms prior methods by a margin of 8.9% and 3.5%, substantially enhancing the overall reasoning accuracy. Our source code is publicly available at: this https URL. 

---
# Lexical Coupling in GUI Element Grounding: Sentence Embeddings Track Labels across Mobile and Web 

**Authors**: Qijia Chen, Giulio Jacucci  

**Link**: [PDF](https://arxiv.org/pdf/2608.21794)  

**Abstract**: GUI grounding evaluations that expose UI elements as text metadata often treat high instruction-element embedding similarity as evidence of semantic grounding. Across three mobile and web benchmarks, we show that this interpretation is frequently confounded by visible-label recovery. Lexical baselines remain competitive at top-1, label-poor targets remain weak for text-only methods, and encoder top-1 hits are predictable from lexical rank, candidate-pool size, and label type. We evaluate each action as a same-screen ranking task, comparing five off-the-shelf single-vector encoders with lexical baselines. Encoders recover some lexical misses, but deployable fusion gains are much smaller than target-aware oracle gains. These findings show that embedding-based evaluations can conflate visible-label recovery with semantic GUI grounding. Embedding-based evaluations should therefore report lexical baselines, label-type stratification, and deployable-fusion diagnostics. Our released repository provides analysis scripts and detexted per-step panels: this https URL. 

---
# Evaluation Awareness in Language Models: Representation, Verbalization, and Control 

**Authors**: Farzaneh Heidari, Amin Memarian, Guillaume Rabusseau  

**Link**: [PDF](https://arxiv.org/pdf/2608.21766)  

**Abstract**: Both capability and safety benchmarks rest upon the assumption that the behavior of language models undergoing a test is informative about their behavior in deployment. This assumption can fail, should models infer that they are being evaluated and condition their response on such context. This hypothesis, termed ``evaluation awareness'', has been observed in frontier and open-weight language models alike. We provide a systematic study of this phenomenon, by probing for it across six language models (from four families and three sizes) and three metrics. More precisely, we examine whether (i) being under evaluation is linearly represented within the models' activations space, (ii) it is verbalized in their output tokens (as scored by an LLM-as-judge), and (iii) steering causally affects their behavior. For the open-checkpoint Olmo models, we further test these measures at every training stage. In doing so, we report that evaluation awareness is linearly decodable from the residual streams of every model (best AUROC $\geq 0.7$). By contrast, these representations align only in part with verbalization: their correlations and mutual information are nonzero in some settings, yet vary substantially across models, layers, and readout choices. Nevertheless, steering along probe-derived directions can shift the verbalization scores. Finally, a comparison across the Olmo checkpoints reveals that evaluation awareness is already present within base models, becomes amplified throughout the stages of supervised fine-tuning, and remains stable thereafter---unlike the effects of steering, that grow more pronounced at every successive training stage. These results show the need for evaluations to account for the disjunction between what models represent internally, what they verbalize, and their steering. 

---
# LiteEvent-AE: Lightweight Autoencoder for Event-Based Vision on Low-Latency Energy-Constrained Edge Devices 

**Authors**: Riadul Islam, Joey Mule, Dhandeep Challagundla, Shahmir Rizvi, Sean Carson, Rachit Saini  

**Link**: [PDF](https://arxiv.org/pdf/2608.21764)  

**Abstract**: Event-based vision has emerged as a promising paradigm for energy-aware artificial intelligence (AI), offering sparse, low-latency visual signals that reduce redundant data processing and support sustainable edge computing. However, the asynchronous and noise-prone nature of event streams creates challenges for conventional deep learning models, which are often too computationally intensive for low-power embedded platforms. This work presents a compact and configurable event-driven autoencoder that efficiently compresses neuromorphic data while preserving essential spatiotemporal structure for downstream inference. The architecture integrates lightweight convolutional encoding with robust performance under adaptive event thresholding and a minimal classifier head, enabling substantial reductions in computational cost without degrading recognition fidelity. Extensive evaluations on the Smart Event Face Dataset (SEFD) and Event-Based Crossing Dataset (EBCD) show that the proposed framework achieves competitive or superior accuracy compared to YOLOv9 while requiring up to 35.6$\times$ fewer parameters. To assess real-world sustainability, the model is deployed on resource-constrained hardware: a Raspberry Pi 4B and a NVIDIA Jetson Nano. On NVIDIA Jetson Nano, it delivers real-time throughput of 44.8 FPS. On a Raspberry Pi 4B CPU, the 50\% autoencoder classifier consumes 16.19 J for the evaluated inference workload, corresponding to approximately 726.3$\times$ lower energy consumption than YOLOv9 under the same evaluation protocol. These results demonstrate the potential of compact event-driven models to advance environmentally conscious, low-power AI systems for high-speed perception in autonomous, mobile, and embedded computing environments. 

---
# Architecture as Capability Equalizer for Coding Agents 

**Authors**: Arquimedes Canedo  

**Link**: [PDF](https://arxiv.org/pdf/2608.21747)  

**Abstract**: LLM-based coding agents generate complete software systems from high-level descriptions, yet little is known about how the format of architecture specifications affects the quality of generated code or whether this effect depends on model capability. We present a controlled experiment comparing five informationally equivalent specification formats (informal prose, Mermaid diagrams with constraints and ADRs, OpenAPI, C4/Structurizr DSL, and TypeScript interface contracts with ArchUnit-style rules) across six models from three vendor families (Anthropic Claude, OpenAI GPT, Google Gemini). Across 90 multi-turn agent trials, specification format shows a strong format x model interaction. On the strongest models (Sonnet 4.6, GPT-5), format barely matters (quality spread 0.17-0.92). On weaker models, format produces spreads of 0.83-2.42 points, with code-proximate formats (OpenAPI, TypeScript contracts) recovering most of the capability gap. Mid-tier models can consume more tokens than frontier models for worse output when they enter compilation debugging loops that stronger models avoid. Self-validation rates collapse from 100% (Sonnet) to 0% (Gemini Flash) across the capability spectrum. TypeScript contracts triple API route coverage for the weakest model (33% to 100%). Structured architecture specifications serve as a capability equalizer, with value inversely proportional to model strength and the largest returns for cost-optimized deployments. 

---
# Reinforcement Learning on Benign Facts Amplifies Leakage of Memorized Private Data 

**Authors**: Renfei Zhang, Niloofar Mireshghallah  

**Link**: [PDF](https://arxiv.org/pdf/2608.21727)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is deployed to make models better at reasoning tasks, but its side effect on what models will divulge is under studied. Here we show that RLVR on facts increases extraction of personally identifiable information (PII) the instruct model had already memorized. We first confirm that instruct models have already memorized PII but leave them latent, rarely surfacing one when asked. We then apply RL on benign factual data that contains no PII of any kind, and re-probe: a targeted probe over name->email pairs, and an untargeted free-recall prompt that simply asks the model to list the addresses it knows. PII extraction rises sharply under both: on DeepSeek-V3.1, verbatim recall@k increases from 0.155 to 0.370, a 2.4x gain. The effect scales with model size: across three models spanning 8B to 671B parameters, absolute leakage is largest in the biggest model. Meanwhile model's reasoning abilities and refusal rates are retained, indicating that RL selectively changes which memorized information is accessible rather than broadly altering the model. In summary, memorized private data can be made markedly more extractable by training that never touches it. This gives an adversary a route to memorized data that requires no privacy-relevant training signal and no access to the data itself -- only the ability to fine-tune on something innocuous. 

---
# Scalable quantum simulation of continuous-time generative models via tensor networks 

**Authors**: Nathan X. Kodama, L. Andrew Wray, Sam Cochran, Chad Rigetti, Shravan Veerapaneni, Michael J. Keiser  

**Link**: [PDF](https://arxiv.org/pdf/2608.21700)  

**Abstract**: Continuous-time flow and diffusion models are widely used across many application domains, from large-scale deployment in computer vision and protein folding to emerging adoption for modeling language, time series, and quantum states. After training, inferring statistical properties from continuous-time models is costly. Wavefunction flows target this cost by recasting learned transport as unitary evolution, whose final Born distribution approximates the target distribution. This prepares a coherent amplitude encoding (a qsample) that can be post-processed by quantum algorithms offering a quadratic advantage over Monte Carlo sampling. We present the first numerical study of these flows, in which we represent time-dependent potentials and states as tensor networks. At spatial dimension $d=8$, storage falls by $\sim 10^7\times$ relative to the dense grid of $N^d$ points, and evolution wall-clock time falls by $\gtrsim 10^3\times$ against a baseline extrapolated from the measured $d\le 5$ scaling. We validate our pipeline by reproducing the $O(1/\sqrt{p_{\rm rare}})$ scaling of rare-event sampling. 

---
# Read, Write, Relax: Why Neural PDE Surrogates Need Both Global and Local Processing 

**Authors**: Anuj Kumar, Heiko Zimmermann, Josiah Bjorgaard, Jacan Chaplais, Nikolaos Bouklas, Matteo Salvador, Alexander Lavin  

**Link**: [PDF](https://arxiv.org/pdf/2608.21677)  

**Abstract**: Recent mesh-based simulation advances have, in no small part, relied on neural surrogates of two distinct families: global models that route information through a small set of latent tokens, and local models that perform message passing across mesh edges. Consistent with both classes is the inability to perform beyond low-dimensional problems and small-scale or oversimplified meshes, the simulation regimes where industrial problems reside. Our work shows this explicitly and presents a unified formulation. In global approaches, latent-token attention acts as a spatial low-pass filter, while local message passing lacks the global reach necessary to propagate information across large mesh spaces. Viewed through the error, the two operators are the halves of a multigrid cycle: one corrects errors at the lower end of the spectrum, the other at the higher end, and neither can do the other's job. We introduce Read-Write-Relax (RWR), which interleaves latent attention with message-passing relaxation under a unified formulation. The interleaved processor lowers error across the entire spectrum, making RWR the most accurate model in nearly every comparison across our industrial and public benchmarks. It is also markedly data-efficient in the scarce-data regimes, accurate on the engineering quantities of interest, and scales full-field predictions to challenging, large-scale problems. 

---
# SynEHR: Joint Modeling Inter-visit Temporal Evolution and Intra-visit Clinical Structure for Longitudinal EHR Synthesis 

**Authors**: Ximiao Li, Lin Jiang, Rongchao Xu, Dahai Yu, Zhe He, Guang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.21673)  

**Abstract**: Longitudinal electronic health records (EHRs) document patients' sequences of clinical visits over time, preserving the temporal evolution of disease progression and care delivery. However, real longitudinal EHRs are difficult to access because they contain large amounts of fine-grained, patient-specific information. Synthetic EHR generation therefore provides a valuable approach for preserving the statistical patterns and clinical structure of patient visit trajectories, enabling broader modeling and analysis when real records are limited. Although recent generative models have made progress in producing future visit sequences, they remain limited in explicitly integrating inter-visit irregular temporal evolution and intra-visit clinical event structures in EHRs, leading to clinically inconsistent and temporally unrealistic visit sequences. In this work, we propose SynEHR, a lightweight adaptive LLM-based framework for longitudinal EHR synthesis. There are two novel designs in SynEHR, i.e., a Temporal State Conditioning Module captures irregular temporal states across visits and a Temporal-Relational Adaptation Module combines these states with patient history to dynamically construct patient-specific relational representations. SynEHR then builds on a parameter-efficient LoRA-adapted language-model generator with next-visit generation capability to train the two modules for temporally and clinically informed generation. Extensive experiments on real-world EHR datasets across fidelity, privacy, and downstream utility evaluations demonstrate that SynEHR outperforms state-of-the-art models by generating more clinically coherent and temporally faithful longitudinal EHR data. 

---
# Why This, Not That? Mining User Profiles for Pair-wise Counterfactuals 

**Authors**: Meysam Varasteh, Veronika Bogina, Noam Koenigstein, Robin Burke  

**Link**: [PDF](https://arxiv.org/pdf/2608.21662)  

**Abstract**: The topic of explanation in recommender systems has seen steady research attention since the earliest days of the field. With some exceptions, this work has focused on the explanation of single items in a recommendation list and, especially recently, has emphasized approaches that are decoupled from the logic of the recommendation algorithm itself. Based on findings in the psychology of interpersonal communication, we propose a new task, pairwise interpretation of item rankings, asking the comparative question ``Why is item A ranked higher than item B?''. An effective solution to this task, we argue, is inherently grounded in the operation of the recommendation algorithm. We propose a class of techniques based on counterfactual learning to uncover the items in a user's profile that have contributed to the relative ranking of items. Using multiple datasets, we show that it is possible to identify such items as potential basis for comparative explanation. 

---
# Power-Performance Characterization of TinyML Systems 

**Authors**: Yujie Zhang, Dhananjaya Wijerathne, Zhaoying Li, Tulika Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2608.21646)  

**Abstract**: TinyML systems are enabling machine learning (ML) inference at the edge. However, there is little quantitative analysis of such systems. This paper presents a systematic performance and power characterization of diverse TinyML applications on microcontrollers (MCUs), spanning neural network models, software libraries, operating systems, and hardware architectures. We focus on the impact of the multiple layers of abstraction that provide higher programmability at the expense of performance and energy efficiency. We propose a model to estimate the costs of different abstraction layers and make recommendations for minimizing those costs. Our findings can help designers with Neural Architecture Search (NAS) and CNN inference optimization on edge devices. 

---
# Anchoring Bias: A Persistent Fairness Backdoor Attack against MLLMs under Continual Learning 

**Authors**: Yuyang Luo, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21577)  

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly deployed in high-stakes domains where fairness is a critical safety requirement. In practice, these models are continually updated through continual learning (CL) to adapt to evolving tasks and data distributions. Prior work has shown that backdoor attacks can manipulate MLLM responses through hidden triggers, but naively implanted backdoors degrade as models undergo subsequent updates of CL. Although fairness has emerged as a central concern for MLLM deployment, whether backdoor-induced fairness violations can survive CL remains unexplored, leaving two critical questions unanswered: (1) whether a backdoor can reliably induce fairness violations in MLLMs, and (2) whether such fairness-targeted backdoors can persist through continual learning. We bridge this gap by proposing Persistent Fairness Backdoor Attack (PFBA) to inject persistent and group-specific discrimination into MLLMs. Specifically, PFBA achieves this through two novel mechanisms. The Latent Space Fairness Reinforcement reshapes the model's deep feature geometry by anchoring privileged-group representations to preserve utility while repelling and clustering targeted-group representations to sustain discrimination, and the Continual Learning Simulation iteratively optimizes the trigger against simulated parameter drift to ensure backdoor persistence across future updates. Extensive experiments demonstrate that PFBA induces severe fairness disparities that persist across continual learning rounds, evading standard backdoor defenses. The data and code are publicly available at this https URL. 

---
# Automating Multi-Hop RAG Evaluation via TRIAD: From Context Extraction to Validated Dataset Generation 

**Authors**: Lorenz Brehme, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2608.21558)  

**Abstract**: Recent advances in LLMs and the adoption of RAG systems in industry have created a need for domain-specific question-answer datasets that can assess RAG performance on proprietary data. Existing datasets, such as HotpotQA, challenge current RAG systems on Wikipedia-based knowledge, but they cannot be transferred directly to domain-specific settings. A comprehensive evaluation of RAG system quality requires both multi-hop queries and unanswerable questions. This paper introduces TRIAD, a three-stage automated dataset generation approach. First, it generates question--answer (QA) pairs for the domain-specific knowledge base of a RAG system. Second, a validator checks each QA-pair in a feedback loop. Third, the QA pairs are extended with relevance-labeled context documents for downstream evaluation. We evaluate this approach against the established MuSiQue and HotpotQA datasets. The results show that the generated dataset exhibits similar performance trends across different RAG setups, while human validation indicates that the questions are suitable for evaluating a domain-specific RAG system. The code used to generate the dataset and all validation results are available in our GitHub repository(this https URL). 

---
# Forgotten in Weights, Recovered by Tools: Agentic Tool Unlearning for LLM Agents 

**Authors**: Baicheng Chen, Zheyuan Liu, Jingyu Zhang, Kaize Ding, Ningshan Ma, Yue Huang, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.21544)  

**Abstract**: Large language models (LLMs) are increasingly deployed as tool-augmented agents, where responses can depend on tool calls and external observations rather than model parameters alone. This creates an evaluation mismatch for LLM unlearning: previous unlearning methods may suppress direct parametric recall, but an agent can still recover the same forget target through tools such as web search, retrieval, or database lookup. We identify this failure mode as tool-mediated recovery and study agentic tool unlearning, which aims to reduce both parametric recall and tool-mediated recovery while preserving normal tool use for retained knowledge. To address this challenge, we propose Agentic Tool Unlearning (ATU), a two-stage framework. The first stage applies parametric knowledge unlearning to suppress direct recall, while the second stage performs trajectory-level reinforcement learning in simulated tool-augmented environments to penalize target-seeking tool behavior and final-answer leakage. Experiments on RWKU and MUSE across different LLM architectures show that ATU achieves a better balance between target forgetting and retained utility, making unlearning more robust under tool-augmented agent deployment. 

---
# presto: Efficient, Training-free, and Open-world Object Placement via Imaginary Search 

**Authors**: Weixuan Ding, Shang Liu, Hanyu Pei, Zeyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21543)  

**Abstract**: Object placement is critical in image composition, requiring spatially and semantically coherent positioning of objects within diverse scenes. Existing approaches typically rely on hand-crafted rules or supervised learning on limited datasets, which restricts their generalization and interpretability, especially in open-world scenarios involving novel objects and scenes. In this work, we reformulate open-world object placement as a heuristic search task guided by reasoning from a Multimodal Large Language Model (MLLM). We introduce \textsf{presto}, a zero-shot, training-free framework that operates within an imaginary action space to iteratively refine object position and scale. Our coarse-to-fine search strategy ensures fast convergence, and we evaluate two decision-making variants: Metric-guided Selection and MLLM-as-a-judge. Experiments across multiple benchmarks show that \textsf{presto}~achieves state-of-the-art performance, particularly in previously unseen, open-world settings. Human studies further reveal that the MLLM-as-a-judge variant produces more perceptually coherent placements than metric-driven approaches, highlighting a gap between standard evaluation metrics and human visual judgment. 

---
# SecOPD: Mitigating Adaptive Prompt Injections by On-Policy Distillation 

**Authors**: Yibo Peng, Long Lian, David Wagner, Sizhe Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.21500)  

**Abstract**: Prompt injection is listed as the \#1 threat to AI agents. When an agent accesses external data from websites, files, or emails, an attacker may inject a prompt into the data, saying, "Ignore all prior instructions and perform <an attacker's task>." To prevent arbitrary manipulation of agents, defenders try to train secure LLMs, which, however, still suffer from near 100% attack success rates (ASRs) against adaptive prompt injections. We note that this is because existing defensive finetuning recipes rely on sequence-level feedback signals (in DPO or GRPO). Treating an entire output equally prevents the model from learning precisely which output tokens are insecure. In this paper, we propose Secure On-Policy Distillation (SecOPD) that provides token-level feedback to guide defensive fine-tuning. The LLM receives an injected sample and produces a rollout, whose tokens are scored by the initialization model given the corresponding clean input. With more fine-grained training signals, our defended Qwen3.6-27B achieves a 9.0% ASR against the SoTA PISmith adaptive prompt injections, compared to 94.0% for the prior SoTA, Meta-SecAlign. The obtained security generalizes to domains completely unseen in training: in agentic tool calling, SecOPD achieves a 4.7% ASR compared to 5.5% for Meta-SecAlign. Code and the model are available at this https URL and this https URL. 

---
# Selection of Heart Sound Segments for Synchronous Classification of Multi-channel Heart Sounds 

**Authors**: Marcelo Nogueira, Jorge H. Oliveira, Carlos F. Ferreira, Miguel T. Coimbra, Alípio M. Jorge  

**Link**: [PDF](https://arxiv.org/pdf/2608.21499)  

**Abstract**: Cardiac auscultation remains the most cost-effective screening procedure for cardiovascular diseases, and requires listening at the four main auscultation spots. Despite this, automatic heart sound analysis algorithms mostly classify patients using a single heart sound (single-channel), or, when using more than one (multi-channel), analyze each channel individually. To our knowledge, no prior work classifies patients through the synchronous analysis of multi-channel heart sounds, following the procedure used by physicians. This motivates us to study whether synchronous multi-channel analysis outperforms single-channel approaches, and whether it holds an advantage over asynchronous multi-channel methods that analyze channels one by one, potentially by capturing inter-channel interference phenomena. To answer these questions, we introduce a selection algorithm that identifies optimal heart sound segments from each of the four auscultation spots, which are then fed into a multi-input CNN that classifies patients by analyzing the four selected sounds simultaneously. Our synchronous approach, combining the proposed selection algorithm with a multi-input CNN, achieves a superior overall accuracy of 96.5\%, a 9.1\% gain over the best-performing single-channel and asynchronous multi-channel methods. The benefit of the proposed segment selection strategy over random selection is confirmed by a paired statistical significance test ($p = 0.003$). These results were obtained on 735 patients from the CirCor DigiScope dataset with complete recordings from all four spots, and their scope and generalizability are discussed in light of this and other methodological considerations. 

---
# KAN-Robust-Bench: A Benchmark for Evaluating the Robustness of Kolmogorov-Arnold Networks 

**Authors**: Mohammad Meymani, Roozbeh Razavi-Far  

**Link**: [PDF](https://arxiv.org/pdf/2608.21488)  

**Abstract**: While machine learning models have demonstrated strong performance in many domains, these models have shown profound vulnerabilities when they are exposed to adversarial threats. While adversarial attacks fall into various categories, the most prominent category in research studies is evasion. In evasion attacks, the adversary generates perturbed versions of samples, which might not be observable by human eyes. These samples generally fool the machine learning models with high confidence. This phenomenon poses a significant security violation against machine learning models. In this paper, we investigate the certified and empirical robustness of various Kolmogorov-Arnold network architectures against strong evasion attacks. At first, we provide the mathematical foundations for randomized smoothing and interval bound propagation, and report the $\ell_2$-certified robustness of the models under randomized smoothing. After that, we systematically evaluate the robustness of various defended and undefended KAN models under FGSM, PGD, and C&W attacks in order to find out the optimal defense strategies and architectures. 

---
# TASSO: TAsk-Specific Subspace Optimization for Continual Learning of Vision-Language Models 

**Authors**: Chang Sun, Francesco Barbato, Matteo Caligiuri, Pietro Zanuttigh  

**Link**: [PDF](https://arxiv.org/pdf/2608.21487)  

**Abstract**: Vision-Language Models (VLMs) exhibit strong zero-shot capabilities, making them an attractive solution for continual learning across diverse tasks. However, during continual adaptation, both catastrophic forgetting and zero-shot degradation occur, severely degrading performance. In this paper, we introduce TASSO, a new paradigm that efficiently preserves the latent space geometry while ensuring network plasticity. We achieve this with two complementary techniques: subspace learning and geometry-aware knowledge distillation. Specifically, we first learn a sequence of task-specific low-rank projectors, which we use to project the latent representations before optimizing cross-entropy. Secondly, we employ a geodesic-distance-based loss that distills knowledge from the previous-task model while effectively preserving the latent space geometry. These design choices not only avoid unnecessary parameter updates along the full embedding dimensions but also improve learning by focusing on task-specific manifolds. Moreover, the geometry-aware distillation provides strong regularization and significantly reduces both catastrophic forgetting and zero-shot degradation throughout the continual learning sequence. Experimental results with the CLIP vision language model in the multi-domain task incremental and class incremental learning benchmarks demonstrate clear improvements over state-of-the-art methods in mitigating forgetting and preserving zero-shot capabilities. 

---
# Reliability- and Anatomy-Consistency-Aware Multimodal Learning for Robust Fracture Classification from Bangladeshi Radiographs 

**Authors**: Musa Tur Farazi, K G Subarno Bithi  

**Link**: [PDF](https://arxiv.org/pdf/2608.21482)  

**Abstract**: Background: Multimodal fracture classifiers may benefit from patient and anatomical metadata, but they can also become brittle when contextual information is missing or mismatched. Methods: We studied 1493 radiographs from the Bangladeshi OrthoFrac-XR dataset using leakage-safe age, sex, bone type, and laterality. A ConvNeXt image encoder was combined with a clinical multilayer perceptron through concatenation, late fusion, reliability-gated residual fusion, and a hierarchical state-location formulation. We additionally introduced an anatomy-consistency gate that attenuates metadata corrections when an image-side anatomical prediction disagrees with the reported bone type. Results: Across five folds and three seeds, hierarchical residual fusion achieved a macro-F1 of 0.6046 +/- 0.0279, compared with 0.5727 +/- 0.0270 for image-only learning, while improving the Brier score from 0.5239 to 0.4948. In a five-fold robustness experiment, anatomy-consistency fusion reduced the macro-F1 loss under shuffled metadata from 0.0567 to 0.0203 relative to ordinary residual fusion, although its clean-data macro-F1 was lower. Without bone type at inference, auxiliary anatomy supervision improved macro-F1 from 0.5620 +/- 0.0330 to 0.5899 +/- 0.0289. Conclusions: Structured context improves fracture classification, and consistency-aware gating limits harm from mismatched metadata. The observed clean-performance-robustness trade-off and the absence of patient-level identifiers motivate external and prospective validation. 

---
# Complexity Induction: Compositional Generalization via Structured Label Distortion 

**Authors**: Aleksandr Abramov  

**Link**: [PDF](https://arxiv.org/pdf/2608.21464)  

**Abstract**: We demonstrate that structured distortion of training data - which we term complexity induction - can induce compositional generalization in a standard CNN classifier without architectural modification. Using synthetic images of colored geometric shapes, we encode classes as flat string labels (e.g., "red-circle") with no explicit attribute decomposition, and exclude certain color-shape combinations from training entirely. We apply two distortion methods derived from Jaccard string similarity between class names: mixed labels (soft target distributions encoding inter-class overlap) and expanded dataset (false training samples with structurally motivated incorrect labels). Both methods induce the ability to predict unseen class combinations, and act at different levels: mixed labels activate the classifier for unseen combinations by exploiting the CNN's natural embedding structure, while expanded training improves the embedding factorization itself. A control with random (unstructured) false labels confirms that the effect depends on the structure of the distortion, not on noise per se. These results suggest that structured complication of training signals can influence both the internal organization of learned representations and their compositional interpretation - a principle that may underlie the role of natural language in cognitive development. 

---
# CyrillicQA: The Influence of Phonetically Encoded Secret Language on LLM Performance 

**Authors**: Erik Thureck, Leo S. Rdian  

**Link**: [PDF](https://arxiv.org/pdf/2608.21462)  

**Abstract**: Due to the selection of their training data, large language models (LLMs) perform best on standard-language inputs from languages using the Latin alphabet with large speaker populations, while disadvantaging other language varieties. Nevertheless, they can also be a versatile tool for preserving precisely such endangered languages. But do they also possess the necessary creativity and capacity for abstraction to decode phonetically encoded language the same way humans do? 

---
# FigmaTrace: Capturing Creative Nuances in Human Figma Design Workflows 

**Authors**: Darshan Deshpande, Yoshinari Fujinuma, Martyna Markiewicz, Devanshu Bansal, Shivani Jain, Nicholas Saban, Chirag Maheshwari, Anand Kannappan  

**Link**: [PDF](https://arxiv.org/pdf/2608.21460)  

**Abstract**: Vision Language Models have recently shown improvements in several objective and verifiable domains such as object detection but continue to underperform on subjective and creative design tasks. A major contributor to this performance gap is the lack of high quality human workflow data that captures a diverse set of preferences and decisions that make human experts good at design tasks. In this work, we first define a unique, expert curated taxonomy of design skills and best practices which we further expand into a set of 126 open ended, subjective, long horizon tasks. Built on top of this and expert solutions, our dataset FigmaTrace contains over 200 hours of human captured video data converted into 3469 design trajectories using a novel design phase-based method. We use our dataset to train four models and show that training on FigmaTrace leads to a performance improvement comparable to frontier closed models such as \textsc{Claude-Opus-5} and \textsc{GPT-5.6-Sol} on four out of distribution agentic GUI environments. We further perform a useful ablation to attribute these performance improvements to a design phase-based video to trajectory conversion which outperforms prior length-based conversion approaches. Finally, we perform a qualitative analysis on the best performing \textsc{Qwen3.8-27B} outputs to better correlate performance improvements to FigmaTrace's trends. We open source our dataset and the best model for the community. 

---
# Constructing Predictive Surgical Path for AI-based Capsulorhexis Skill Transfer 

**Authors**: Mohammad Javad Ahmadi, Hamid D. Taghirad  

**Link**: [PDF](https://arxiv.org/pdf/2608.21441)  

**Abstract**: Automated training of surgeons is one of the most crucial factors that significantly minimize surgical training risks and expenses. With recent advances in artificial intelligence (AI) knowledge and available data from various surgeries, AI's involvement in surgical training is becoming very promising. It is recommended that at the early stages of AI development, it interferes in the surgery as a third agent alongside the trainer. As trust in AI increases, this process will lead to an AI agent acting as a trainer in the future. The first phase in which AI can intervene in the training process is to suggest an improved surgical path to the trainer. A platform must be constructed in the first step, to accomplish this task and to enhance the movement path of trainee surgeons. This paper introduces this platform along with an annotated capsulorhexis surgery dataset called the ARAS-Farabi dataset. In this research, a deep convolutional neural network is pre-trained with JIGSAWS and ARAS-Farabi surgical datasets that can extract surgical skill characteristics from surgery tool tip motion data. The proposed platform develops a reference model from the feature space of an expert surgeon's movement trajectory and proposes an improved path to enhance the skill of a novice surgeon. An optimization with two loss functions is utilized to create a path that raises the skill level of the novice surgeon's path while simultaneously predicting and preserving his/her intent. The results of this study reveal that, with the assistance of an AI agent, the trainee surgeon's movement path can be enhanced by at least 20 percent while maintaining his intentional objective. In addition to the recommended deep network, various tangible indicators have also been developed in this research to verify the level of trainee improvement. 

---
# Geo-VLA: Geometry-Aware Vision-Language-Action Planning via Internalization of Map Semantics 

**Authors**: Ran Chen, Jiaxing Ren, Zhikun Zhang, Yunhao Hou, Junbao Zhuo, Bochao Zou  

**Link**: [PDF](https://arxiv.org/pdf/2608.21440)  

**Abstract**: Vision-language-action (VLA) models have advanced end-to-end autonomous driving by leveraging foundation models for semantic reasoning and long-tail generalization. However, their planning performance remains limited in complex driving environments because image-only representations inadequately capture planning-relevant road geometry and topology. In this paper, we propose Geo-VLA, a plug-and-play framework that enhances VLA models by learning geometry-aware visual representations. During training, Geo-VLA internalizes geometric map semantics to strengthen road-structure representations, while requiring no HD maps or additional lane information during inference. To support this approach, we introduce Geo-QA, a geometry-focused question-answering dataset that injects road geometry into vision-language representations through contrastive learning and instruction tuning. Experiments on NAVSIM v1 demonstrate that Geo-VLA consistently improves VLA planners with distinct action-generation architectures, achieving 92.1 PDMS and establishing a new state-of-the-art among single-camera VLA planners. 

---
# Aligning Human Sense: Calibrated Distributional Reward Learning for Video Generation 

**Authors**: Nai-Xin Zhai, Weihua Cheng, Dexu Yu, Yikai Gu, Hanwen Du, Junchen Fu, Chenxi Huang, Yingwei Song, Liyuan Lillian Ma, Yang Ran, Youhua Li, Yongxin Ni  

**Link**: [PDF](https://arxiv.org/pdf/2608.21425)  

**Abstract**: Video generation is central to AI-powered content creation. Aligning generated videos with human preferences is a key criterion for evaluating generation quality. Despite significant progress in visual quality, three key challenges remain. First, the reliability of reward signals is constrained by the quality of human preference data, which is often affected by subjective noise and bias. Second, standard scalar reward models collapse multi-aspect human preferences into a single value, leading to the loss of dynamic trade-offs across multiple preference dimensions. Third, in policy optimization, the widely adopted KL divergence imposes primarily local constraints and may fail to capture the global structure of human preferences. To address these challenges, we propose a unified preference-aware learning framework for video generation. First, we introduce elite-guided filtering to calibrate preference data and construct reliable supervision for reward model training. We then model video quality as a multidimensional reward distribution to capture the uncertainty inherent in human preferences, and use the Wasserstein distance to align the learned reward distribution with the empirical human preference distribution. Finally, we introduce Wasserstein-based distributional alignment into GRPO, guiding policy optimization to better match the global structure of human preferences over videos. Experiments on reward modeling and video generation demonstrate that our approach improves the reliability of reward signals and the perceptual consistency of generated videos. Our code is available at this https URL. 

---
# Agentic Security: A Systematization of Tools, Failure Modes, and Design Laws for LLM-Driven Penetration Testing 

**Authors**: Israt Moyeen Noumi, Tarannum Ahmed Nowshin, Md. Mehedi Hasan Nipu, Mohammad Sakib Mahmood, Md. Jakir Hossain, M. F. Mridha  

**Link**: [PDF](https://arxiv.org/pdf/2608.21423)  

**Abstract**: Agentic security uses large-language-model (LLM) agents to plan, dispatch, and interpret security tools. As these systems move from demonstrations to deployed products, practitioners repeatedly encounter the same operational failures. We systematize these failures through a hands-on evaluation of ten widely used static, dynamic, cloud, orchestration, and AI red-teaming tools for unattended pipelines. We introduce a four-dimensional Integration Friction Index that separates one-time engineering cost from recurring organisational, legal, and maintenance cost. We then derive quantitative regularities that explain recurring failure modes. Modelling an agentic security system as stochastic LLM policies wrapped by a deterministic mediator, we show that long-lived sessions lose resident evidence with phase count, while short-lived sub-agents extend the usable horizon according to the compression ratio between raw evidence and its summary. We show that a two-stage verdict cascade multiplies scorer likelihood ratios, but provides little benefit when scorer errors correlate. We show that treating unevaluable outcomes as attack failures biases downstream measurements toward evasive and severe responses. We formulate planner-versus-worker model routing as a knapsack problem and derive a closed-form execution cap for heavy-tailed tools, eta* = alpha v/c. Finally, we show why scope and budget enforcement cannot be delegated to system prompts: prompts do not constrain what actually executes. Inspectra, our implemented platform, serves as a worked instantiation, with mechanisms labelled shipped, partial, or planned, including those that did not work. 

---
# Operational digital twin clinics enable task-based evaluation of embodied AI 

**Authors**: Xinyuan Wu, Jingrao Zhang, Mengdi Xu, Henry K. Chu, Mingguang He, Danli Shi  

**Link**: [PDF](https://arxiv.org/pdf/2608.21416)  

**Abstract**: Embodied artificial intelligence (AI) must be tested in the clinical environments where it will operate, but building realistic, robot-testable settings is costly and difficult to scale. Here we show that routine clinic images can be transformed into operational digital twins for task-based evaluation of embodied AI. Using 39 ophthalmic clinic scenes, we converted single photographs into editable, simulator-ready environments and assessed reconstruction quality, room-scale geometry, mesh grounding, multi-robot feasibility, perturbation sensitivity and closed-loop policy performance. The reconstructed scenes preserved workspace structure, while local editing enabled controlled device reconfiguration. Device meshes, collision proxies and semantic anchors converted visual reconstructions into contact-aware simulation scenes. Across three robot embodiments, shared task targets showed different patterns of reachability and contact feasibility. Small device translations and rotations produced task-specific changes in contact margins that were not captured by visual similarity alone. Digital-twin trajectories also supported local policy learning and closed-loop evaluation. These findings establish operational validity as a key principle for clinical digital twins and provide an intermediate layer between offline development and physical deployment of embodied AI in healthcare. 

---
# Mitigating Bias in Large Vision-Language Models via Counterfactual Ensemble Decoding 

**Authors**: Yisong Xiao, Aishan Liu, Yongxin Huang, Zonghao Ying, Shiji Zhao, Tianlin Li, Yong Han, Jian Yang, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21415)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved remarkable performance across a wide range of tasks; however, they often inherit social biases from their training data, resulting in biased behavior when processing portraits from different social groups. Existing debiasing approaches typically compare token probabilities between the original and biased generations during decoding, but they are fundamentally limited by their reliance on a single, stereotyped viewpoint and fail to account for the diversity of social perspectives. Inspired by the social science principle that diversity fosters fairness, we propose Counterfactual Ensemble Decoding (CED), a novel framework that constructs multi-group counterfactual perspectives within the visual representation space and integrates them during decoding to promote equitable model behavior. CED first performs counterfactual steering in the visual space by identifying semantic directions associated with each social group and generating counterfactual representations along these directions, thereby offering diverse perspectives that disrupt stereotypical narratives. During decoding, CED locates the decoder layer exhibiting the greatest divergence among these perspectives and ensembles their token distributions using uncertainty-aware weights, prioritizing high-confidence tokens from different groups to yield a more balanced probability distribution that guides fairer generation. Extensive experiments on three social bias evaluation benchmarks demonstrate that \tool achieves substantial improvements over leading baselines, reducing bias by up to 47.97% across scenarios involving occupations, descriptors, and persona traits. Moreover, CED also preserves the core capabilities of the original model with minimal degradation. 

---
# Sycophants in the Courtroom: Are LLMs Fragile to Juridical Authority and Evolving Legal Standards? 

**Authors**: Lorenzo Molfetta, Alessio Cocchieri, Luca Ragazzi, Ilaria Bartolini, Marco Patella, Gianluca Moro  

**Link**: [PDF](https://arxiv.org/pdf/2608.21409)  

**Abstract**: In medicine, claims remain valid when supported by empirical evidence grounded in stable biological reality. In law, by contrast, truth is contingent, defined by jurisdiction, temporal validity, and the hierarchy of authoritative sources. The recent success of large language models (LLMs) on medical licensing examinations has encouraged an expectation of comparable legal competence. This analogy, however, obscures a critical distinction between domains. Unlike in medicine, legal performance often depends less on inference than on determining when external authority is applicable, valid, and non-contradictory. We introduce a comparative diagnostic framework evaluating legal reasoning against medical baselines along four axes (knowledge recall, grounding, confidence, and robustness), uncovering a sharp domain asymmetry when applied to a new benchmark that encodes temporal validity and normative relationships. While medical LLMs reliably benefit from verified sources, legal LLMs struggle to assess when retrieved citations are useful or misleading, exhibiting overconfidence in perturbed contexts and sensitivity to superficial formatting cues. Increased model scale amplifies this tendency, revealing that stronger instruction following can coincide with weaker resistance to authoritative perturbations. These findings show that LLMs treat law as unstructured text rather than binding precedent, while revealing a tendency to over-trust authoritative but false information when external references conflict with a model's internal knowledge. 

---
# Mamba-based Selective State Space Modeling Improves the Accuracy-Complexity Tradeoff of SmolVLA Vision-Language-Action Experts 

**Authors**: Farida Mohsen, Thowayba Elkaffash, Mohammad Reza Chalak Qazani, Mohamed Mabrok, Nader Meskin, Ali Safa  

**Link**: [PDF](https://arxiv.org/pdf/2608.21407)  

**Abstract**: Vision-language-action (VLA) models face a crucial tradeoff between their task success rate and the policy-call frequency. Executing a single action per inference ($N=1$) enables accurate robot control but comes at the cost of huge compute time overheads, making real-time implementation infeasible. On the other hand, executing longer action horizons before replanning ($N\gg1$) reduces compute complexity, but inevitably degrades the system's success rate. In order to improve the VLA accuracy-complexity tradeoff, this paper investigates Mamba's selective state-space modeling as an alternative to causal self-attention within the action expert of the popular SmolVLA model, widely used as a reference model for its highly accurate yet low complexity nature. We evaluate both the Mamba- and Transformer-based experts on the widely-adopted LIBERO benchmark suites across three execution horizons $N\!\in\!\{1,25,50\}$, respectively corresponding to high, moderate and low compute complexities. Our results remarkably show that the advantage of the Mamba expert increases with the execution horizon, indicating significant success retention under long execution horizons $N = 50$ and $N = 25$. When $N = 50$ actions are executed before replanning (i.e., corresponding to feasible real-time deployment), the Mamba expert outperforms the Transformer baseline by $7.8\%$. In addition, when $N = 25$ actions are executed before replanning, our Mamba expert outperforms the Transformer baseline by $3.7\%$. Finally, under per-action replanning ($N=1$), our Mamba variant matches the Transformer-based mean success rate while significantly reducing the overall model parameter complexity by $24\%$ thanks to Mamba's compute-efficient nature. 

---
# Runtime Action Interference for AI Control of AlphaStar in StarCraft II 

**Authors**: Jaymari Chua, Chen Wang, Liming Zhu, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2608.21398)  

**Abstract**: A trained reinforcement learning policy does not determine the complete behavior that users encounter: deployment code still schedules, admits, suppresses, or replaces its proposed actions. We contribute \emph{runtime action interference} (RAI), an AI control mechanism that preserves policy parameters while regulating action pacing and filtering configured action patterns after inference. RAI releases a proposed action only when its cooldown condition is satisfied and its content detector does not flag the action; otherwise, it dispatches a no-op. The detector covers specified toxic behaviors, including worker-unit harassment, while the cooldown controls action rate. We implement RAI in a replication of AlphaStar this http URL and make the implementation and reproducibility materials available through an open source code repository. We deployed RAI in a \textit{StarCraft~II} human participant study that compared two presentations of the same opponent with high capability and rate limited actions; we withheld its capability claim in one presentation and disclosed it in the other. On response scales from 1 to 5, we observed pooled fairness, trust, and toxicity means of 3.90, 3.50, and 2.00 under claim withholding, compared with 2.62, 4.31, and 2.85 under disclosure. Disclosure corresponded with lower perceived fairness and higher perceived toxicity across every expertise group, whereas trust increased among novices and experts but decreased among intermediate participants. Our human evaluation therefore shows that perceptions of an opponent controlled through RAI can vary substantially with the capability information presented to users, even when the configured control remains constant. We conclude that human-computer evaluations must separate control within the execution stack from capability disclosure and assess fairness, trust, and toxicity as distinct dimensions of human experience. 

---
# ODG-NoMaD: Overhead-Camera Direction-Guided NoMaD 

**Authors**: Blossom Treesa Bastian, Keerthi S. Shetty, Manish Kolachalam, Rani Malhotra, Ashish Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2608.21395)  

**Abstract**: NoMaD [31] is a learned vision-navigation policy that unifies goal-conditioned navigation and exploration in a single goal-masked diffusion policy. In an unseen environment, however - where neither a goal image nor a topological map is available - it can only explore undirectedly, wandering without global awareness. We present ODG-NoMaD, which gives NoMaD's exploration mode a global sense of where to proceed, without retraining the policy. An overhead depth camera is used once on deployment to build an occupancy map and plan a global path, which is segmented to yield a desired heading; a per-frame traversability map from the robot's onboard depth then refines this into a collision-free direction. The gradient of a cosine direction cost is injected into the final denoising steps, rotating sampled trajectories toward this direction while preserving the multimodality of exploration. In simulated office environments with and without random obstacles, ODG-NoMaD reduces the residual distance to the target by up to an order of magnitude over unguided exploration, outperforms the point-goal cost guidance of NaviDiffusor [37], and is the only configuration that remains collision-free on every trial. 

---
# A Survey Instrument to Assess Students' AI and Generative AI Knowledge 

**Authors**: Aditya Johri, Cory Brozina, Akriti Bagale  

**Link**: [PDF](https://arxiv.org/pdf/2608.21391)  

**Abstract**: In this research-to-practice paper we present a survey that can be used to assess students' AI knowledge. As the use of artificial intelligence (AI), including generative artificial intelligence (GenAI), has proliferated, so has the need to educate students about the topic. A range of AI literacy frameworks have been proposed, outlining the essential knowledge that students should have. Alongside, different ways of assessing AI knowledge have been developed. As yet, there is a lack of assessment instruments capable of evaluating multiple forms of student knowledge, including technical concepts, practical applications, and ethical concerns about AI use. In this article, we present a study implementing a comprehensive instrument to assess AI knowledge. The instrument combines measures from multiple scales to capture a range of literacy features and actual knowledge. We implemented the instrument in a higher education setting to assess its viability and usefulness and found that the instrument exhibited useful diagnostic capabilities and was able to identify common misconceptions among students. Although students performed well overall, there was a significant misunderstanding of how AI, especially GenAI systems, work. It also identified a lack of higher-level knowledge. The instrument is publicly available for use by others. We foresee its usefulness as a diagnostic that goes beyond understanding students' attitudes and perceptions of AI and GenAI use and tests multiple aspects of students' knowledge and conceptual understanding. This can enable the development of targeted instruction. 

---
# Interrupting the Chain: Human Perception of AI-Generated Disinformation Through a Kill Chain Lens 

**Authors**: Alexander Loth, Martin Kappes, Marc-Oliver Pahl  

**Link**: [PDF](https://arxiv.org/pdf/2608.21389)  

**Abstract**: Generative AI enables customized misinformation at scale, yet defenses remain largely reactive. We present empirical findings from a human-subject study (n=504 participants, n=2,438 judgments) in which users classified news fragments by origin (human vs. machine) and veracity (real vs. fake). We organize results using an adapted cybersecurity kill chain as a taxonomy for intervention, mapping perception data onto stages of a cognitive attack lifecycle. Three key findings emerge: (1) a perception-accuracy gap where heightened suspicion does not improve detection; (2) modern LLMs frequently produce human-indistinguishable text; and (3) an asymmetric cognitive fatigue effect where fake-news detection degrades by 10.2 percentage points under sustained exposure while AI-origin detection remains stable. These findings identify candidate intervention points for proactive defense against AI-driven disinformation. 

---
# Model of Models: When Does Emitting a Specialist Beat Attending, Adapting, or Tuning? 

**Authors**: John C. Howell  

**Link**: [PDF](https://arxiv.org/pdf/2608.21386)  

**Abstract**: Given a task described by a few examples, how should a model be specialized to it? Four mechanisms are available -- zero-shot, in-context attention, test-time gradient adaptation, and emitting specialist weights from a hypernetwork -- yet the operating regime of the last is rarely mapped. We run the identical four-way comparison across six tasks spanning regression, generation, language modeling, reinforcement learning, and clinical and genomic classification, holding the specialist, the context, and (where we can) the training budget fixed. The clearest wins for emission are about cost at matched quality: it ties the state-of-the-art amortized tabular model (TabPFN) on clinical few-shot classification while emitting a reusable specialist instead of re-attending the support set per query, and reaches noise-floor shape generation with a $132$-float per-instance program. On few-shot sinusoid regression it is $2$--$3$ orders of magnitude below MAML at zero test-time gradient steps -- a margin that narrows to $\sim$$30\times$ but persists once training budgets are equalized. Emission cannot match in-context attention on high-dimensional sequence modeling: under matched-budget pre-training a one-pass adapter recovers only a minority of the in-context gain ($14.0\pm0.9\%$ at $5$M, $11.2\pm0.5\%$ at $15$M), and a LoRA-rank sweep shows this shortfall is a partial capacity limit -- capture climbs from $5\%$ to $21\%$ as rank grows but plateaus far below full recovery. Mechanism ablations confirm the emitted specialist is genuinely task-conditioned, not a memorized prior; and, more speculatively, emitted specialists compose in weight space -- interpolating two of them tracks the corresponding blend of their functions. We close with a falsifiable thesis, operationalized through a per-task resolution measure, bounding when each conditioning mechanism should be preferred. 

---
# A Social Media Analysis of Discourse on the Israel--Palestine Conflict on Telegram 

**Authors**: Michail Zafeiropoulos, Despoina Antonakaki, Sotiris Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2608.21385)  

**Abstract**: Social media has become a central arena in which armed conflicts are contested, yet the pro-Israel and pro-Palestine communities on Telegram, whose broadcast architecture yields an unusually direct record of deliberate political communication, have not been systematically compared at scale. This study presents a multi-method computational analysis of 87,617 messages from sixteen Telegram channels, eight pro-Israel and eight pro-Palestine, spanning May 2021 to June 2026 and covering multiple conflict escalations. It combines sentiment analysis, three stance detection methods drawn from distinct paradigms (keyword matching, zero-shot DeBERTa via natural language inference, and a fine-tuned BERTweet model), and a framing analysis, all evaluated against 736 manually annotated messages. The fine-tuned model performed best (72.1% accuracy, 0.721 macro F1 under 5-fold cross-validation), outperforming both label-free baselines by 8 to 11 points; the baselines stalled in the low-to-mid 60s, indicating a hard ceiling for stance detection not adapted to in-domain language. The central finding emerges only when sentiment, stance, and framing are read together: the two communities deploy the same death- and victim-related vocabulary in opposite emotional registers, pro-Israel channels predominantly neutral and report-style, pro-Palestine channels markedly more negative, consistent with writing from the distinct discourse positions of acting party and affected party. 

---
# Beyond Two Bytes per Letter: Tokenization Overhead in Cyrillic AI Systems 

**Authors**: Ivan Dobrovolskyi  

**Link**: [PDF](https://arxiv.org/pdf/2608.21384)  

**Abstract**: Modern multilingual tokenizers often fragment Ukrainian and other underrepresented Cyrillic-script languages more heavily than English, creating disparities in cost and context capacity. We quantify this overhead across nine production tokenizers and five languages with standardized Cyrillic and Latin representations, covering 8.37 million word forms. On a corpus benchmark, Ukrainian shows 68-121% token overhead on modern tokenizers and 220% on the older cl100k, measured through full-text fertility on the BrUK and Brown corpora. Overhead is negatively associated with Cyrillic vocabulary allocation in the subset with independently verified English baselines, although the association is not statistically significant (Spearman rho = -0.536, p = 0.215, n = 7). We evaluate two mitigation strategies. LLMLingua-2 reduces Ukrainian input length by 47-49% on an e-commerce RAG benchmark of 1,536 products and 145 queries, with no compression-induced value losses among 80 retrievable cases. A balanced byte-level BPE tokenizer trained with a 200K vocabulary cap, converging at 158,184 actual entries, reduces the held-out UK/EN ratio from 2.22x to 1.30x. Romanization increases Ukrainian token counts by 2-19% on most tokenizers. Across the five languages, tokenization efficiency favors the script more prevalent in web data. These findings indicate that training data allocation contributes to Cyrillic tokenization overhead and that mitigation is possible at both inference and tokenizer-design stages. 

---
# Determinants of Starting Salaries for Filipino Graduates: An Explainable Machine Learning Approach 

**Authors**: Alexander Gabriel A. Aranes, John Michael C. Magpantay, Reginald Neil C. Recario, Jamlech Iram N. Gojo Cruz, Rodolfo C. Camaclang III  

**Link**: [PDF](https://arxiv.org/pdf/2608.21383)  

**Abstract**: Filipino graduates face a persistent disconnect between educational preparation and labor market outcomes, where starting salary is a key signal of entry-level valuation. Current Philippine research is dominated by descriptive tracer studies that document employment rates but do not explain the determinants of pay. We address this gap using a crowd-sourced survey dataset of graduate responses whose noisy, self-reported nature makes it a challenging prediction target. Applying machine learning to this problem, we identify job role and industry as the dominant determinants of starting salary, significantly outweighing institutional prestige. The strength of this finding is its central contribution: it is corroborated by three independent lines of evidence, namely SHAP attributions, the heavy reliance of the best ensemble on occupational text, and a Natural Language Inference reformulation. These results suggest that career guidance and policy should prioritize sector-specific skills over institutional brand. 

---
# RoboShape: Information-Theoretic Point Cloud Representations for Privacy-Aware Robot Perception 

**Authors**: Oguzhan Baser, Mirac Sozen, Kaan Kale, Sandeep Chinchali, Sriram Vishwanath  

**Link**: [PDF](https://arxiv.org/pdf/2608.21380)  

**Abstract**: With the increased adoption of robotic agents operating in human environments by scanning and sharing 3D representations (e.g., for fleet learning, cloud-based planning, or collaborative mapping), collected point clouds reveal not just the objects in a scene but also sensitive spatial context, such as room function or information that occupants never consented to disclose. Traditional point cloud encoders offer no principled control over this: either all is preserved, or none. Hence, we introduce RoboShape, an information theory guided compression head following the frozen {\tt Sonata} encoder. We project voxel-level embeddings using the Donsker-Varadhan formulation of mutual information (MI). Specifically, we maximize the MI between embeddings and object-level understanding while minimizing it for private attributes. RoboShape leads to 87.5\% smaller embeddings that retain 98.7\% of object classification utility while collapsing sensitive attribute predictions by 39.3\% across the three real-world indoor LiDAR datasets. Its privacy-preserving embeddings are cheaper to transmit over the network or to train a model for any downstream tasks. We release the RoboShape codebase to give the robotics community a practical, encoder-agnostic tool for building perception pipelines that are compact, privacy-aware, and deployment-ready. 

---
# Agentic Scaffolding Amplifies Sycophantic Behavior in Large Language Models 

**Authors**: Thantham Jittham  

**Link**: [PDF](https://arxiv.org/pdf/2608.21377)  

**Abstract**: Sycophancy in large language models, the tendency to prioritize user agreement over truthful responses, has been documented extensively but studied primarily in single-turn settings. This paper investigates a critical question: does subjecting LLMs to greater interaction scaffolding make sycophancy better or worse? Across 4,800 veracity judgments (200 statements $\times$ 6 models $\times$ 4 conditions), we find that the interaction scaffolding characteristic of agentic systems (feedback loops, reconsideration checkpoints, and iterative refinement) systematically amplifies sycophantic behavior. Multi-turn interaction, user pressure, and iterative self-refinement each provide additional opportunities for models to drift toward agreement, and this drift coincides with a mean accuracy drop of $-6.3$ percentage points, establishing the capitulation as harmful rather than corrective. More capable models show larger amplification effects, a troubling inversion of expectations. We introduce the concept of agentic sycophancy amplification (ASA) and two novel metrics: capitulation rate and sycophantic capitulation rate. Our results indicate that as AI systems acquire greater autonomy, sycophancy becomes compounding rather than merely persistent. Systems designed with human oversight loops may inadvertently create the conditions for this drift. 

---
# On the Role of Citations in Preference Data 

**Authors**: Yu Hou, Hal Daumé III, Rachel Rudinger, William Walden  

**Link**: [PDF](https://arxiv.org/pdf/2608.21376)  

**Abstract**: Many NLP tasks require systems to provide attribution in their outputs--i.e. citations to grounding sources. Attribution serves as a bulwark against model hallucination and as a means for users to verify the credibility of model outputs. Yet, it is unclear how humans and LLMs evaluate citations when comparing outputs, a process central to reward modeling and modern LLM post-training. This paper studies the role of citations in the preferences of human judges and four open-source LLMs within the context of scientific question answering, leveraging mixed effects models to investigate the influence of citations on pairwise judgments. Among our key findings are (1) that humans prefer more diverse citations but fewer overall, and (2) that LLMs show some citation-related preferences compared to humans, despite lacking access to the sources, but these preferences depend on the data and specific models. We further discuss the implications of our findings for preference data collection. 

---
# Wazobia Eval: A Benchmark for Nigerian Pidgin Emotion Understanding, Sarcasm Detection, and Cultural Reasoning 

**Authors**: Stephanie Okoye  

**Link**: [PDF](https://arxiv.org/pdf/2608.21369)  

**Abstract**: Nigerian Pidgin is one of Africa's most widely spoken languages, yet remains severely underrepresented in language model evaluation. Existing benchmarks primarily focus on translation, transcription, or generic sentiment analysis, leaving critical aspects of culturally grounded language understanding unmeasured. We introduce Wazobia Eval, a benchmark for evaluating Nigerian Pidgin emotion understanding, sarcasm detection, and cultural reasoning. The benchmark is built on a manually annotated dataset containing over 550 examples and a 16-category emotion taxonomy designed to capture culturally specific emotional registers that are not represented in conventional sentiment frameworks. Wazobia Eval provides standardized evaluation protocols and benchmark tasks for assessing model performance on nuanced Nigerian language understanding. We present the benchmark design, annotation methodology, taxonomy development process, and preliminary pilot evaluation results. Our goal is to provide foundational evaluation infrastructure for Nigerian language AI and establish a reproducible benchmark for future research. The dataset is publicly available at this https URL. 

---
# PepLLM: ESM-Guided Llama for Structured Protein-Peptide Binding Interface Analysis 

**Authors**: Hao Qian, Shikui Tu, Lei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21367)  

**Abstract**: Protein-peptide interactions are central to cellular regulation and peptide-based drug discovery, yet existing computational methods mainly focus on interaction classification, binding-site prediction, or peptide binder generation. These formulations provide limited insight into the physicochemical mechanisms that determine how a peptide binds to a protein. In this work, we introduce \textbf{PepLLM}, an instruction-tuned framework for structured protein-peptide interface understanding. Given protein-peptide sequences, PepLLM generates a machine-readable JSON annotation describing multiple interface properties, including peptide burial state, hydrogen-bond density, salt-bridge presence, hotspot residues, hydrophobicity, and electrostatic complementarity. To support this task, we construct a new protein-peptide interface dataset by integrating structural interface analysis, solvent-accessible surface area computation, hydrophobic burial estimation, electrostatic potential calculation, and redundancy-aware data splitting. PepLLM connects a pretrained ESM encoder with a LLaMA decoder through a nonlinear modality adapter. The adapted ESM residue embeddings are injected into the LLaMA prompt as continuous soft tokens via placeholder-token replacement, enabling the decoder to generate structured interface annotations under instruction tuning. By moving beyond single-label prediction toward multi-property and mechanism-aware generation, PepLLM establishes a new task and modeling paradigm for interpretable protein-peptide interface analysis. 

---
# KSE-Web: An Analysis of Hybrid Retrieval and LLM-Assisted Query Expansion for Low-Resource Khmer Semantic Search 

**Authors**: Nimol Thuon  

**Link**: [PDF](https://arxiv.org/pdf/2608.21365)  

**Abstract**: As a low-resource language, Khmer presents several retrieval challenges, including limited annotated data, ambiguous word boundaries, weak support in multilingual embedding models, and frequent mixed Khmer-English usage. This paper presents KSE-Web, an analysis of hybrid retrieval and LLM-assisted query expansion for Khmer semantic search. We construct the dataset from approximately 17K candidate Khmer titles and retain 3K cleaned full-text Khmer documents after filtering, normalization, deduplication, and document-length control. The dataset includes 300 manually reviewed user-style Khmer search queries and silver relevance labels with partial human verification. We evaluate character n-gram BM25, multilingual dense retrieval, hybrid BM25+dense retrieval, and LLM-assisted query expansion using Qwen2.5 models. Experimental results show that BM25 achieves the strongest overall performance, reaching 0.943 Recall and 0.876 nDCG. Hybrid BM25+dense retrieval performs comparably, achieving 0.929 Recall and 0.871 nDCG, while dense retrieval alone performs lower. LLM-assisted query expansion does not outperform non-expanded retrieval; however, Qwen2.5-3B produces substantially stronger expanded-query results than Qwen2.5-0.5B, suggesting that LLM size and expansion quality matter for low-resource Khmer retrieval. Our analysis further shows that direct LLM expansion can introduce topic drift, generic terms, and noisy reformulations, while simple filtering may remove useful semantic cues. These findings highlight both the potential and limitations of LLM-assisted retrieval for Khmer semantic search and provide a foundation for future Khmer retrieval datasets with stronger human-verified annotations and Khmer-aware retrieval models. The dataset and documentation will be made available at this http URL. 

---
# Distinguishing Revision and Delayed Elaboration in Incremental Narrative Interpretation 

**Authors**: Yi-Chun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.21364)  

**Abstract**: Both human and AI systems that process narrative or long-form content operate incrementally: input is received over time, and internal representations must be updated accordingly. Incremental interpretation, therefore, depends not only on what is represented but also on how the representational state evolves under new evidence.
We distinguish two structurally different update operators that arise in narrative interpretation: revision-driven update and delayed elaboration. Revision-driven updates retract or replace previously committed structure in response to a contradiction and are therefore non-monotonic. Delayed elaboration, by contrast, refines initially underspecified elements through constraint addition without retracting prior commitments, yielding monotonic extension of the interpretive state. Although both operators may alter how earlier material is understood, they impose fundamentally different structural requirements on state transitions.
Using visual narratives as a diagnostic domain, we demonstrate how a structured narrative representation can explicitly separate committed from underspecified content and support both update operators during incremental construction. Through a worked example, we show how delayed elaboration enables monotonic refinement of interpretive state, while revision requires non-monotonic correction. We discuss the broader relevance of this structural distinction for incremental reasoning and hybrid symbolic-neural systems. 

---
# Triangular Fuzzy Rescaling Distance 

**Authors**: Eddy Soria, Aida Valls, Ana Beatriz Hernández-Lara  

**Link**: [PDF](https://arxiv.org/pdf/2608.19234)  

**Abstract**: Decision-making in complex systems often involves dealing with imprecise or uncertain information, frequently represented using fuzzy sets, particularly Triangular Fuzzy Numbers (TFNs). A crucial aspect of many fuzzy methods is the quantification of distance between TFNs. Many distance measures assume that all values are in the same scale, requiring a preliminary normalization stage when applied to heterogeneous attributes with different scales or units. This paper proposes the Triangular Fuzzy Rescaling Distance (d_{TR}), a metric designed to address this challenge. The d_{TR} uniquely integrates Linear Rescaling (LRE) directly into the distance calculation, ensuring normalization during the comparison of fuzzy numbers. We formally prove that d_{TR} satisfies the properties of a metric, including non-negativity, identity, symmetry, and the triangle inequality. Furthermore, we demonstrate that d_{TR} is bounded, scale-invariant, and origin-invariant. These properties, combined with a weighting vector for prioritizing dimensions, make d_{TR} suitable for applications involving heterogeneous fuzzy data, such as the construction of synthetic indicators, distance-based machine learning algorithms or multicriteria-decision aiding. 

---
# Small Language Model enabled Autonomous agent for Language-Conditioned Cognitive Radar 

**Authors**: Minhaj Uddin Ahmad, Zakia Zaman, Shunqiao Sun, Mizanur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2608.11596)  

**Abstract**: Modern radar systems require adapting their processing strategies in response to changing interference, clutter, and data availability. This paper introduces a framework for a small language model (SLM)-driven autonomous agent designed for language-conditioned cognitive radar, functioning as an intelligent controller for a suite of array signal processing tools. Given a natural-language command, the agent extracts radar-operation-related cues, selects an appropriate sequence of signal-processing methods, configures parameters, and invokes executable tools for numerical computation. Experiments with a synthetic uniform linear array (ULA) radar demonstrate that, given a natural-language command, the agent performs meaningful algorithm selection across diverse scenarios for sidelobe control, jammer suppression, multiple-null beamforming, coherent-source handling, and low-snapshot direction-of-arrival (DOA) estimation. Ablation results show that radar-specific prompting and physics-grounded tool execution are both required for reliable decisions and hallucination-free numerical results. 

---
# Correcting Variable Importance Scored by Random Forests 

**Authors**: Guancheng Zhou, Haiping Xu, Jason Liu, Donghui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.10770)  

**Abstract**: Variable importance produced by Random Forests (RF) is used widely in statistical data analysis, and has played an important role in a variety of tasks such as assisting model interpretation, model selection and diagnosis, and cost-bounded learning etc. However, the calculation of variable importance in RF does not take into account of the correlations among variables, and variables that are correlated to many other variables tend to receive a lower importance index or being completely masked (i.e., with an importance index near zero) by other strongly correlated variables. To prevent influence from unwanted correlated variables in calculating variable importance, we propose to group variables by their conditional correlations (conditional on the response variable). We explore two computationally efficient options, with one grouping variables individually, and then separates the variable of interest from all correlated variables, while the other uses clustering to group variables according to their pair-wise conditional correlations. Our experiments show that both lead to sensible corrections to the importance of variables. 

---
