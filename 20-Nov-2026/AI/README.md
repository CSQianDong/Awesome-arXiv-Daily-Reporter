# PlayCraft: Designing Engaging Games from Agent Play Experience 

**Authors**: Weihao Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.26624)  

**Abstract**: Recent AI agents can generate playable games, but these games are not necessarily engaging. Mastery is a key source of engagement, yet designing for it requires extensive human play and tuning. We propose PlayCraft, an automated framework that uses a vision-based agent to play games and draws on its detailed play experience to revise them for mastery. Starting from an initial game, PlayCraft alternates between playing and crafting. In the playing stage, a player agent watches the game screen, uses the controls, and pursues the game objective over repeated episodes while retaining memory. After each episode, it reports its progress, outcomes, decisions, and reflections. In the crafting stage, a crafter agent uses these reports to revise the game. It estimates player abilities, predicts performance at different proficiency levels, and compares those predictions with a target learning progression. It diagnoses the game and addresses the gaps by giving players meaningful opportunities to improve through better decisions and execution. We evaluate games in fresh runs with player agents across three intelligence settings, measuring learning gain and learning slope. Compared with the initial games, PlayCraft yields larger learning gains and steeper learning slopes in most games. It also outperforms the baselines on the macro-average of each metric at every intelligence setting. 

---
# Probing the "Conscious" Mind: Selective Access and Global Workspace in Vision-Language Models 

**Authors**: Jiahao Huang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26590)  

**Abstract**: Vision-language models (VLMs) integrate visual and textual information, but their outputs do not reveal internal concept representations. Global workspace theory (GWT) proposes that information is selected from parallel brain processes, integrated in a shared workspace, and made available to specialised processes. We introduce the Multimodal Concept Lens (MC-Lens), a multimodal Jacobian lens, to study these functions in VLM concept processing. Its vision encoder and language decoder lenses map intermediate states to a shared semantic space defined by the decoder's output vocabulary to compare concept accessibility across modalities. MC-Lens supports internal concept interventions to examine the role of accessed concepts in downstream computation. (i) Accessibility and selective access. Multimodal concept information is readable in intermediate states and selectively accessible as tasks require. We extend "readout carriers" from one name token to a token set to better capture and express concept information and support counterfactual experiments. (ii) Concepts in downstream computation. Counterfactual experiments show that accessed concept information participates in downstream computation and reasoning. (iii) Development of cross-modal alignment and selective access. We develop two J-space metrics tracking the development of cross-modal concept alignment more clearly than hidden state cosine in controlled VLM training. The experiment also shows that reliable selective access develops later than concept accessibility. (iv) Shared functions across models. Across VLM families and sizes, we find shared patterns of concept accessibility, query-dependent selectivity and cross-task intervention effects despite differences in internal organisation. These findings offer an experimental basis to study GWT in multimodal artificial systems and how it can guide cross-modal alignment improvements. 

---
# ProtoDiffusion: Token-per-Layer Generative Modeling of Multi-Layer Images 

**Authors**: Adrià Ciurana Lanau, Harleen Hanspal, Lusine Abrahamyan, Emilio Tylson Baixauli, Yu Cao, Danai Triantafyllidou, Stefano Zorzi, Gustavo Zanoni Felipe, Peter Griger, Stefan Sietzen, Petru-Daniel Tudosiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26586)  

**Abstract**: Graphic designs are ordered compositions of RGBA layers whose separation enables selection, movement, and reuse. Existing layered models represent each layer with a full-canvas or region-sized spatial latent grid, so representation size grows with the cumulative spatial extent of the layers. We present ProtoDiffusion, a latent-diffusion framework built on ProtoVAE, a multi-layer variational autoencoder that encodes a design into one shared spatial prototype and one candidate vector per layer. Each candidate conditions a shared decoder to reconstruct its corresponding RGBA layer. For $N$ layers, the representation contains $HW+N$ tokens before backbone-specific packing, compared with $NHW$ for full-grid-per-layer representations. ProtoDiffusion jointly denoises the prototype and a fixed set of candidate slots, conditioned on an image or text, with bounding boxes as optional layout guidance. The model leaves unneeded slots empty and determines the number of layers and, in absence of boxes, their layout, requiring neither as input. The same architecture supports both image-to-design and text-to-design. ProtoVAE achieves high-fidelity reconstruction from its compact shared latent representation, reaching 40.43 dB PSNR on layer crops. On the LayerD evaluation split, image-to-design achieves RGB L1 scores of 0.040 with bounding boxes and 0.057 without, with box-guided alpha soft IoU (0.850) comparable to the strongest fine-tuned baseline (0.871). On merged images, text-to-design reaches 66.41 FID versus 86.84 for FLUX.2 [klein], achieves a 0.860 text-accuracy score, and runs $19.1\times$ faster than the same-backbone generation-then-decomposition cascade. 

---
# MM-SystemOneBias: Evaluating Bias in Multimodal Jev-Style System One Decision Models 

**Authors**: Jinzhe Li, Qi Wang, Dandan Guo, Li Liu, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26557)  

**Abstract**: When multimodal decision models determine actions, moderation outcomes, or quality judgments, their reliability depends on preserving semantic decisions across presentation changes, ignoring irrelevant additions, and distinguishing information consistent with task evidence from information that conflicts with it. Existing evaluations study decision-interface biases and multimodal model behavior in separate settings, leaving no common protocol to compare multimodal decision models with their backbones on practical tasks. We introduce \textbf{MM-SystemOneBias}, which evaluates 13 models on 5,323 original instances from nine public datasets and 11 task scenarios using six paired interventions: option order, option label, external judgment, image--text description, irrelevant information, and rating scale. Across seven matched decision-model--backbone pairs, decision models are more stable under option-label, irrelevant-information, and rating-scale changes in all seven pairs; six pairs also show lower option-order flip rates. They incur smaller accuracy losses under incorrect external judgments in all seven pairs and conflicting descriptions in six. In six pairs, they gain less accuracy from correct external judgments and show less favorable accuracy changes under consistent descriptions. Among pairs with unchanged semantic decisions, mean absolute changes in answer probabilities range from 4.56 to 11.55 percentage points, revealing input effects that accuracy alone misses. 

---
# The Geometry of Lies: Knowledge and Persona Directions for Detecting Deception in Large Language Models 

**Authors**: Md. Abdullah Al Zami, Suraiya Shanaz, Mahdi Hasan, Shoaib Ahmed Dipu, Farig Yousuf Sadeque, Md. Sabbir Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2611.26545)  

**Abstract**: Linear probes are the cheapest LLM lie detectors but often fail to transfer across kinds of lie. Label-free directions from honest and deceptive role-play are untested at 70B scale and on lies about knowledge, where a lie and an honest denial can be identical. We widen a published persona bank with roles that deny knowledge they have, plus honest counterparts. At each model's middle layer the resulting persona direction ranks lies across Liars' Bench better than single-dataset probes on three of its four models; it fails on Qwen, even when Qwen stays in role far more often. On Llama-3.3-70B-Instruct it beats directions from a random persona bank, but its gain over the published direction is within the range of random widenings. On lies about knowledge, persona directions mainly track a reply's wording: where lies and honest denials share their words, our three main contrasts stay inside the random-direction band on Llama and Qwen and reach only its edge on Gemma, though on Mistral one clears it. A knowledge direction contrasts questions the model can answer with ones nobody can, each followed by the same scripted denial. It separates lies from identical honest denials on two models and narrowly on a third, but not on the fourth; its other results partly depend on the benchmark's wording. On the Gender-Secret fine-tune it clears the band by a few hundredths on Llama and fails on Mistral. The sum of the two directions ranks lies better than a supervised probe at the same depth on all four models (on one, both near chance) but is level with the persona direction on Llama's average and behind that probe on held-out lies. More varied role-play questions do not help the persona direction on Liars' Bench. 

---
# LLM GerryMandering: Separating Interactions from Additive Effects in LLM Audits 

**Authors**: George Rahul, Ronit Jahilal, Bhaskara Hanuma Vedula, Ponnurangam Kumaraguru, Pushkar Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2611.26538)  

**Abstract**: Intersectional bias arises when a model treats people who hold several marginalised identities at once differently from what their separate identities would predict. Auditing such intersectional subgroups remains a challenge: their number grows quadratically, and there is no agreed baseline for what independent attributes would predict. Without such a baseline, many existing audits flag a subgroup when it exceeds each attribute alone, a condition that attributes acting independently already satisfy. We test this standard criterion on 150 synthetic datasets built with no interaction. With the shuffle null, it flags 66.1 false subgroups per dataset on average, and at least one in every dataset. When we plant interactions of known size into these datasets, a repaired null lets the criterion recover only 32% of them. We introduce ADDIS (Additive Intersectional Screen), which fits an additive model over all attributes, scores each subgroup by its Additive Residual Distance (ARD) from that model's prediction, and tests it against a Freedman-Lane null. On our synthetic benchmark, ADDIS flags 0.09 false subgroups per dataset and recovers 84% of the same planted interactions. We then build IndiPersona, 10,000 synthetic Indian personas with six identity attributes, and audit 13 open language models. The standard criterion flags 7.1% of the 738 tested subgroups on average; ADDIS flags 3.0%. Gender identity crossed with sexual orientation is the largest pairing in every model, and counterfactual personas confirm it and additionally reveal a caste x class interaction in all but one model that the observational test lacked the data to detect. Much of what audits report as intersectional bias can thus arise from independent single-attribute effects alone, yet a small core of interactions survives. 

---
# GLiDE: A Fast, Calibrated Decision Model Distilled from Reasoning Teachers 

**Authors**: Sahibzada Allahyar, Julia White, Dhruv Atreja, George Hurn-Maloney, Urchade Zaratiana  

**Link**: [PDF](https://arxiv.org/pdf/2611.26509)  

**Abstract**: Decision models enable fast, confidence-aware decisions by scoring candidate answers in a single forward pass, without generating text or intermediate reasoning. Yet reasoning can substantially improve decision quality, raising the question of whether its benefits can be retained without the cost of generating it. We investigate this question with \textbf{GLiDE}, a family of decision models spanning 0.4B to 27B parameters.
Building on Qwen hybrid language models, GLiDE enables bidirectional interactions between the context, question, and candidate answers by removing causal masking from full-attention layers. To transfer reasoning capabilities into single-pass inference, we combine soft-target distillation from multiple teachers, including a reasoning model, with progressive post-training on diverse decision tasks. We further explore parameter-free recurrent depth to increase inference-time computation without generating tokens.
Across five model sizes, GLiDE lies on the performance--parameter Pareto frontier of the Hugging Face \DI{}, with each model outperforming evaluated alternatives of equal or smaller size. \dmtwo{} (27B) achieves a hidden-set score of 71.52, with an expected calibration error of 2.0\% and a median latency of 92.9\,ms on a single GPU. Further checkpoint averaging and parameter-free recurrent depth yield GLiDE~27B, which reaches 73.37 with an expected calibration error of 2.1\%. 

---
# FLUID: Flexible, Evidence-Grounded Situated Task Guidance 

**Authors**: Yunqi Zhao, Run Peng, Ze Dong, Ryan Lu, Yinpei Dai, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2611.26491)  

**Abstract**: Real-time task guidance is inherently flexible: when an agent guides a user through a task, multiple responses and timings can be appropriate, and sometimes the best action is to remain silent. Existing benchmarks often miss this flexibility by treating synthesized dialogues as fixed targets for what to say and when to say it. We introduce FLUID (Flexible, Evidence-Grounded Situated Task Guidance), a unified framework for modeling, evaluating, and enabling flexible interaction in multimodal task guidance. Using naturally occurring human-human guidance from WTaG and HoloAssist, FLUID grounds interventions in their supporting evidence, necessity, and timing flexibility, and uses this representation both to evaluate flexible guidance and to build a real-time agent for task guidance. Our agent achieves the strongest overall balance of response correctness, intervention precision, timely guidance, and response latency among evaluated systems, while still leaving room for improvement in avoiding speech overlap. Together, FLUID moves situated task guidance beyond rigid reference matching toward more natural and adaptive human-AI interaction. Project page: this https URL. 

---
# Monte Carlo Pass Search for Setpieces 

**Authors**: Andrew Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26464)  

**Abstract**: Set pieces produce about a third of non-penalty expected goals (xG), yet they are usually evaluated by what happened next. We present Monte Carlo Pass Search (MCPS) for set pieces, which treats a corner or a lofted free kick as a distribution over plausible executions and plausible futures instead of a single realised event. From the pre-kick state, MCPS proposes counterfactual deliveries and counterfactual set-ups (where the attackers stand at the kick), plays each out many times in a world model (3D ball physics, a generative model of all 22 players, the keeper's come-or-stay decision and a Player-to-Touch model) and scores every future with a value model. On 249 deliveries from 30 Serie A 2025/26 matches, with every model applied only to matches it never saw, the best available delivery is worth 0.0085 xG more than the one kicked (+28 %), mainly because it reaches an attacker before the keeper. Searching where the attackers stand as well adds +0.0027 xG with the observed ball and +0.0033 on top of the best ball; designing their runs too adds +0.0007 more. We show how to keep such a search honest, and estimate how much more data each component needs before a club could rely on it. 

---
# A CPU Inside an LLM: Zero-Gradient Compilation of Exact Circuits into Pretrained Weights 

**Authors**: Tanmay Sah  

**Link**: [PDF](https://arxiv.org/pdf/2611.26450)  

**Abstract**: Pretrained language models (LMs) remain unreliable on exact multi-digit arithmetic without gradient fine-tuning or external tool interpreters. We introduce closed-form, zero-gradient circuit-grafting primitives that compile deterministic computation directly into pretrained LM weights using zero training examples and zero runtime hooks, while bounding per-layer normalization drift and preserving natural-language (NL) competence. Our construction combines three analytical mechanisms: (1) two-layer Mellin-GLU quadrature over bias-free SwiGLU/GeGLU layers, turning unknown per-token RMSNorm scaling into translation along a log-spaced gate grid ($>250\times$ dynamic range) with IEEE 754 error bounds; (2) cross-token scale cancellation via signed Chebyshev Indefinite-RoPE self-sink attention and memoryless SiLU-antisymmetric GatedDeltaNet shifts; and (3) axis-aligned width extension ($d \to d+K$) with $\sqrt{d/(d+K)}$ norm compensation, Base-1000 canonical BPE emission, and differential $(+254,-256)$ tied unembeddings. An automated 5-stage compiler synthesizes 4-8 digit signed $\{+,-,\times\}$ circuits in constant $\mathsf{TC}^0$ depth alongside 3-literal Boolean (100%, $N=1{,}728$; plus 42-layer Gemma 4 E4B audits) and 5-bit Majority/Parity (100%) circuits. On 6-digit arithmetic ($N=1{,}200$, BF16), IBM Granite 4.2 3B (1.07M-1.33M active non-zeros, $\le 0.036\%$) and Alibaba Qwen 3.5 4B (2.25M-2.39M non-zeros, $\le 0.057\%$) reach 99.17% and 100.00% ALU-prefix accuracy (100% with an FP32 sidecar) and 98%-100% mid-sentence resumption, outperforming 12.8K-sample LoRA (42.2% / 61.7%, 0% inline continuation). A factorized block-sparse sidecar holds A100 memory at 7.89 / 10.65 GiB (4.60 / 8.14 GiB under INT8/INT4 + FP32 sidecar) while passing all 14/14 pre-registered $>2$K-token Tango splits ($N=4{,}392$; 0/11,134 OOD triggers). 

---
# Seer: Guiding Small Deep-Research Agents with Shared Decision Models 

**Authors**: Yuhao Zhou, Yang Zhou, Yuanxi Li, Jindi Lv, Dan Si, Qing Ye, Jiancheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2611.26423)  

**Abstract**: Deep-research agents repeatedly decide what to search, read, and verify, often using large cloud-hosted language models for action proposal. However, privacy and local control motivate running agents on users' devices, where limited memory and compute favor smaller models and complicate on-device training. Yet inference-time guidance faces three obstacles: exploring alternatives through real execution adds tool calls, prompted self-evaluation is bounded by the agent's judgment, and model-specific guidance risks repeated adaptation costs. We introduce Seer, an inference-time framework providing local guidance that separates the agent's action proposal from learned consequence prediction and value estimation. From a shared corpus of real research trajectories, we train two models: a tool-response simulator on observed call--response pairs and a value estimator on histories labeled with discounted terminal correctness. The simulator and estimator are trained offline and reused across agents without further training. Moreover, Seer executes only the selected action and adds only real tool responses to the committed history, avoiding tool calls on discarded candidates at additional inference cost. Across multiple deep-research benchmarks, Seer substantially improves agent performance for different language models. For instance, GAIA accuracy rises from 13.6% to 41.7% for Qwen3.5-4B and from 22.3% to 50.5% for Qwen3.5-9B. Furthermore, our analyses suggest that broader candidate coverage can compensate for imperfect simulation, while prediction and scoring errors limit the benefits of wider imagined search. These findings highlight the potential of reusable models for prediction and evaluation to strengthen small research agents through extra local computation without the real execution of the discarded candidates. 

---
# Reasoning-Trace Analysis Using Large Language Models for Interpretable Q-Matrix Construction 

**Authors**: Jihong Zhang, Zijing Jin, Mingfeng Xue  

**Link**: [PDF](https://arxiv.org/pdf/2611.26422)  

**Abstract**: Large language models (LLMs) can generate Q-matrices for diagnostic classification models that approximate expert specifications, but evaluations have focused on the final matrices, leaving the basis of individual item-attribute assignments unexamined. This opacity limits the responsible use of AI in measurement. We propose Reasoning-Trace Analysis (RTA), a framework in which an LLM generates attribute-labeled reasoning steps through chain-of-thought (CoT) prompting and these steps are analyzed to reconstruct, compare, and inspect each assignment. RTA is intended to supplement expert review and empirical validation. In a simulation study with perturbed fraction subtraction items, four LLMs recovered the required attributes and withheld attributes the items did not require. In an empirical study with 11 fraction subtraction items, responses from 536 examinees, four LLMs, up to three temperatures, and 100 replications per condition, CoT-derived Q-matrices overlapped substantially with the reference Q-matrix (mean overlap .809) but less than directly generated Q-matrices (.877), and they varied more across replications. Inspection of the traces linked disagreements to item-type coding, labeling rules, and how operations were expressed as steps. Q-matrices with similar overlap differed in fit to examinee responses. RTA provides inspectable evidence that matrix-level agreement and fit do not. 

---
# ZeroVLN: Learning to Navigate without Supervised Fine-Tuning of the VLM Brain 

**Authors**: Zixuan Huang, Yang Zhou, Guli Zhang, Sunzhu Li, Hongyan Xie, Zehao Chen, Fuzhen Zhuang, Kaixuan Wang, Xiaozhi Chen, Yikun Ban, Deqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26414)  

**Abstract**: Representative vision-language navigation (VLN) methods typically couple a vision-language model (VLM) brain with a learnable action head and fine-tune the VLM brain on navigation-domain data to predict geometric goals. However, such adaptation can substantially alter base VLM weights and degrade general visual capabilities. We introduce ZeroVLN, a slow-fast navigation framework that achieves effective navigation with zero navigation-domain SFT of the VLM brain. Rather than teaching the brain to predict geometric goals, we equip it with a semantic tool-calling interface for conveying movement decisions to an action expert in natural language. We first adapt the action expert to carry out these movement requests, then jointly improve the brain's movement decisions and the action expert's continuous execution through closed-loop reinforcement learning (RL). Trained on the same navigation episodes, ZeroVLN substantially outperforms representative SFT-based hierarchical VLN methods on standard VLN-CE (continuous environments) benchmarks, while inducing only minor changes to the brain's weights and largely preserving its general capabilities. To our knowledge, ZeroVLN is the first slow-fast VLN framework with zero navigation-domain SFT of the VLM brain, and the first to jointly optimize its slow and fast policies through closed-loop agentic RL in VLN-CE. 

---
# World Models Favor Learned Priors over Visual Evidence 

**Authors**: Han Xiao, Zhiyong Bao, Xiaotong Liu, Yanhao Chen, Qingqiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26366)  

**Abstract**: World models are increasingly used to simulate physical interactions, but training a separate model for each environment limits scalability. Cross-environment prediction requires identifying the current physical rule, which appearance alone may not reveal. We therefore study visual calibration: using object motion and interactions in a short video as direct evidence to infer the current rule and predict motion under that rule from a different initial state. However, differences in data availability across physical conditions can yield a long-tailed training distribution, biasing models toward common rules even when evidence supports rare ones. To study this challenge, we introduce TailWorld, which evaluates the transfer of demonstrated rules across initial states while independently controlling training imbalance and calibration informativeness. The benchmark compares predictions with two reference futures from the same prediction initial state, one following the demonstrated rule and the other a common rule, to measure prediction bias toward common dynamics. We further propose Rule Inference for Transfer (RIFT), which learns how different rules affect object motion, infers the current rule from calibration interactions, and applies it to prediction from a new initial state. RIFT explicitly controls the influence of training frequency on rule inference, allowing sufficiently informative observational evidence to support rare rules. Experiments on TailWorld show that RIFT improves prediction accuracy and physical consistency across varying training imbalances and distribution shifts, with gains in predicting rare dynamics. 

---
# How Much Should an Agent Remember? Deployment Lessons from Self-Distilled Memory for LLM Agents 

**Authors**: K. R. Jayaram, Vatche Isahagian, Aswath Vaithinathan Aravindan, Vinod Muthusamy, Punleuk Oum, Gaodan Fang, Michal Ulewicz, Gegi Thomas, Evelyn Duesterwald  

**Link**: [PDF](https://arxiv.org/pdf/2611.26361)  

**Abstract**: LLM agents on long-horizon tasks fail in recurring, procedural ways. They mis-paginate an API, resolve the wrong entity, or answer when no answer was asked. We built a pipeline that lets an agent learn from its own history. It first mines short \emph{lessons} from the agent's own trajectories, one to three per episode. It then consolidates those lessons into a smaller set of \emph{guidelines}, where each guideline records how many lessons it merged. At inference the agent receives either all guidelines or a selected subset. The pipeline ships in the open-source altk-evolve library and is deployed in IBM's CUGA agent and watsonx Orchestrate platform. We report what we learned building, measuring, and productizing it. Our primary instrument is the AppWorld benchmark, across models spanning 40--92\% baseline task completion. The central finding is that the right amount of memory is a property of the model, not the memory. A strong model keeps improving as guidelines are added. DeepSeek-V3.2 gains +9.5 points of task completion and +16.1 points of scenario completion with its full 191-guideline set. A weak model gains less from the same full set than from a small selected subset. gpt-oss-120b starts at 39.9\% task completion. It reaches 51.2\% with all 199 of its guidelines but 55.4\% (mean of two runs) with a selected subset of about 29. Around that result we report what went wrong: a high-support guideline that backfired, provenance silently dropped during consolidation, a swallowed API error that failed every task while producing a plausible log, and a baseline confound that nearly produced a misleading comparison with a prior method. We distill these into recommendations for teams deploying agent memory. 

---
# Context as a Resource 

**Authors**: Xufei Tian, Han Hu, Wenli Du  

**Link**: [PDF](https://arxiv.org/pdf/2611.26346)  

**Abstract**: Managing concurrent tasks across separate agent sessions quickly exhausts human attention, motivating always-on personal agents behind a single continuous interface. Yet when every context shares identical frozen backbone weights, funneling interleaved dialogues and external events into one linear session induces cross-topic constraint bleed and head-of-line stalling. Conversely, router-ticket hierarchies incur lossy prose handoffs, heavy routing overhead, and blind merges - collapsing true-conflict repository merges to a 33.3% three-run mean despite high isolated branch passes. Rather than scheduling anthropomorphic roles, we introduce ClonAct, a routerless architecture that virtualizes context blocks directly through a pluggable single-pass non-generative Context MMU, in-place Context Cells, a footprint-aware Synaptic Bus with an isolated trial merge gate, and a wave-flushed Shared Trunk. Across three programmatic multi-stream benchmarks and three model families, our portable trial merge gate lifts true-conflict merges to 63.3%, while ClonAct's routerless virtualization attains this ceiling with 45.9% fewer prompt tokens and prevents compact-model router collapse. Under continuous surge streams, ClonAct matches top completion (95.0%) with the lowest median wait and zero timed-out status checks. 

---
# DeepQuest: From Knowledge Gaps to Scientific Discovery in Alzheimer's Disease 

**Authors**: Chunbo Jiang, Yifei Chen, Beining Wu, Fuyou Mao, Weihong Chen, Enyu Bao, Mingxuan Liu, Feiwei Qin, Qiyuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2611.26343)  

**Abstract**: Automating experiments does not necessarily advance scientific understanding. Research agents must identify meaningful knowledge gaps, determine what available data can resolve, and assess whether their findings advance existing knowledge. We introduce DeepQuest, a knowledge-gap-driven AI scientist framework for Alzheimer's disease research. DeepQuest transforms unresolved questions from domain reviews into traceable, testable hypotheses and prioritizes them through data feasibility screening and groupwise Elo ranking. It couples iterative experimentation with two-level auditing that separates experimental reliability from knowledge contribution relative to two frozen literature corpora. On OASIS-3, DeepQuest produces 42 feasible hypotheses and outperforms the strongest baselines in 14 of 15 evaluator-dimension comparisons across three language-model reviewers. The full framework achieves an audit-assessed scientific finding rate of 95.24%, outperforming both general-purpose executors and domain-specific research agents. This scientific finding rate remains stable across literature corpora and evaluator model families. Representative findings concern neuropsychiatric symptoms and dementia risk, longitudinal amyloid spatial stability, and neocortical tau-associated clinical impairment, with key associations supported in independent ADNI and AIBL cohorts. These results show how aligning knowledge gaps, data feasibility, and evidence auditing can turn automated research into traceable, evidence-supported scientific discovery. Our source code is available at this https URL 

---
# Training LLMs to Follow Reasoning Instructions with Minimal Edits 

**Authors**: Alexandra Yakovleva, Luat Gia Khoi Nguyen, Juho Kannala, Harri Valpola, Alexander Ilin  

**Link**: [PDF](https://arxiv.org/pdf/2611.26342)  

**Abstract**: Large language models increasingly rely on extended reasoning traces, yet the training signals used to shape this behavior remain coarse. Supervised fine-tuning on full trajectories forces the model to imitate an entire demonstration, overconstraining its output distribution, while trajectory-level reinforcement learning provides only weak credit assignment, rewarding or penalizing whole completions without identifying where they succeed or fail. We propose edit-based training, a general framework for learning from a model's own trajectories via localized textual corrections. Given a sampled completion, an editor identifies the first point of failure under a specified criterion and proposes a minimal replacement span. The model is then supervised only on these edited spans. By design, this concentrates supervision where the model's reasoning first breaks down and aims to preserve its existing reasoning patterns elsewhere. The method applies both offline and online, with the latter continually adapting supervision to the model's evolving failure modes. We instantiate this approach on reasoning instruction following with Qwen3-32B on the ReasonIF benchmark. Compared to full-trajectory imitation and trajectory-level reinforcement learning baselines, edit-based training substantially improves compliance with the reasoning instruction while largely preserving the model's skills on untargeted tasks. These results suggest that localized correction is an effective complement to these approaches for shaping how LLMs reason. 

---
# Sirius-Tiamat: How Small Can a One-Pass Decision Encoder Get Before Its Decisions Change? 

**Authors**: Mohammed Nayeem Khan, Vrishank Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2611.26339)  

**Abstract**: A one-pass decision encoder reads a text, a question and several named options, and returns a probability for each option in a single forward pass. We ask how far Laya, a 421M-parameter encoder of this kind, can be compressed before its decisions change. We count decisions that differ from Laya's own full-precision (fp32) answer, and track calibration (how well confidence matches accuracy, measured by expected calibration error, ECE), on 1,221 synthetic items with the training schemas, 273 with an unseen schema, and 800 out-of-family questions from ARC, MMLU and HellaSwag. Layer pruning of a related model family is known; we add flips, calibration, held-out data and retraining. bf16 changes 0.2% of in-family decisions but 2.2% out of family; 6-bit grouped rounding 1.3% and 5.9%; deleting 6 layers 0.9% and 9.1%, while ECE rises from 5.9% to 39.2%. One fitted temperature, needing no labels, restores it to 3.6%. One scale per tensor fails below six bits, and CPU dynamic int8 changes 30% of decisions, nearly all through one layer. Total variation tracks flips for rounding but not for deletion, and an early exit saves layers over a static cut only if its confidence gate is first recalibrated. Distilling a 16-layer student cuts its in-family flips from 44% to 9% (45% out of family). At equal stored size, rounding flips fewer decisions than deletion. Data are synthetic and LLM-labelled, for one architecture. 

---
# ClosureBench: A Benchmark for Reasoning under Closure Assumptions 

**Authors**: Francesco Belardinelli, Flavio Corradini, Matteo Leonesi  

**Link**: [PDF](https://arxiv.org/pdf/2611.26335)  

**Abstract**: AI agents built on large language models (LLMs) often answer questions from incomplete records. Missing information might be interpreted as unknown or false, depending on the completeness of the source. Current evaluations fail to distinguish these two aspects, as they are usually interwoven with other reasoning demands. We present ClosureBench, a benchmark that tests whether LLMs follow the stated rules for dealing with missing information. Every base scenario is tested under OWA, CWA, and LCWA, and only the treatment of missing information changes between variants. ClosureBench provides a verified base benchmark as well as agent extensions for action choice, source scope, and updates. A model is considered correct on a base scenario only if it provides correct answers in all of its closure variants. Across 960 instances, overall accuracy can mask substantial closure mistakes. Models achieve overall accuracy from 71.18% to 99.86% across replicated runs, but their core switch accuracy, measured on the scenarios whose gold label actually changes with the contract, varies between 18.23% and 99.31%. Large discrepancies emerge under LCWA. For example, some agents choose unknown despite the prompt indicating that missing information should be treated as false. Our findings demonstrate that evaluations of AI agents should not only measure whether agents can reason over existing facts but also whether they use the appropriate assumption when information is absent. 

---
# Rigel-Ladon: An Option-Count Law for a One-Pass Decision Encoder, Tested on Held-Out Option Counts and Held-Out Schemas 

**Authors**: Mohammed Nayeem Khan, Vrishank Tiwari, Navin Kumar R  

**Link**: [PDF](https://arxiv.org/pdf/2611.26332)  

**Abstract**: Laya is a small neural network that reads a text, a question and K written options, and returns one probability per option in a single pass. We ask how its accuracy falls as K grows from 2 to 35 and whether that can be predicted. We score nested option sets (every K-option question contains the smaller ones) on four question pools, for the released model (Base) and our fine-tune (FT), and score every item against each single rival. FT trained on one 12-option topic question (in-family); a 12-option HR (department-routing) question is out-of-family. Logit accuracy (the log of correct over incorrect answers) is linear in log(K-1) with slope 0.68 to 0.88. Fitted on K up to 6, the law predicts K=7 to 12 within 0.6 points and K up to 24 within 1.2 (a log-linear rule: 1.6 and 6.7). Taking the slope from the topic question and measuring only two-option accuracy on HR predicts its other K within 0.3 points (FT) and 0.7 (Base). On a 35-label pool of five schemas the law fails (5 to 11 points). Independence of irrelevant alternatives is rejected: the gold-versus-rival log-odds shifts by +0.24 to +2.07 nats per doubling of K, yet pairwise answers reproduce K-way accuracy within 4.7 points. Order averaging (averaging over several orderings of the options), shortlists, a temperature law and calibrated prediction sets did not transfer (FT's 90% sets cover 84% on HR). 2 of 10 predictions made in advance held. Caveats: one model family and LLM-labelled synthetic text. 

---
# Actor-Predictor Architecture for Reinforcement Learning with Formal Specification Objectives 

**Authors**: Josef Brozovich, Qining Zhang, Lei Ying  

**Link**: [PDF](https://arxiv.org/pdf/2611.26313)  

**Abstract**: One fundamental challenge in reinforcement learning (RL) is enforcing temporal safety constraints, such as satisfying formal specifications. Existing methods often transform logic formulas into automata for state augmentation and reward shaping. However, this approach assumes that automaton construction is both feasible and tractable, which limits its applicability to small specifications and linear temporal logic (LTL). This paper proposes Actor-Predictor, a framework that learns a surrogate representation of logic progress without explicitly constructing an automaton. The predictor continuously estimates whether a partial trajectory will satisfy the formal specification in a later step, and its learned features are provided as auxiliary inputs to the actor, which is then optimized using GRPO and RWR. We show theoretically that, under sufficient predictive accuracy, the learned representation is equivalent to the automaton state. Experiments on grid world and robotic tasks in Isaac Lab demonstrate that Actor-Predictor outperforms automaton-based baselines on LTL-constrained tasks. It also performs well on non-regular formal specifications, such as $(a^n b^n)$, for which finite-automaton-based approaches are not applicable. 

---
# Client Exits - A Way to Increase LLM Transparency on the Client Side 

**Authors**: Marc Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2611.26309)  

**Abstract**: Textual recommendations generated by large language models (LLMs) can be difficult to inspect. This paper introduces \emph{Client Exits}, a lightweight mechanism for enriching LLM responses while separating semantic intent from client-specific presentation. The model inserts predefined, machine-interpretable markers into its output, and the receiving client decides whether and how to realize the corresponding enrichments. This allows supplementary information to be presented as graphics, text, speech, or other modalities without prescribing a concrete interface. We implement a prototype combining a finetuned Qwen model with a modified LibreChat client. Using the Question--Option--Criteria (QOC) approach including three XML-based client exits. We evaluate two QOC-oriented model variants trained on essentially the same examples, with client-exit annotations added for one variant. Across 100 test questions, the client-exit model includes all three markers in every response. An LLM judge, informed about the mechanism without being instructed to favor it, prefers this model's answers in 66\% of the comparisons. These results demonstrate successful marker generation on the evaluated test set and support the feasibility of client-controlled enrichment. 

---
# Similarity-based OSF Logic for Flexible SPARQL Query Answering 

**Authors**: Gian Carlo Milanese, Gabriella Pasi  

**Link**: [PDF](https://arxiv.org/pdf/2611.26261)  

**Abstract**: We present an updated version of fosf, a reasoner for fuzzy order-sorted feature (OSF) logic, with support for flexible SPARQL query answering over RDF knowledge graphs. Fuzzy OSF logic is a knowledge representation and reasoning language where feature symbols denote functions and sort symbols denote fuzzy sets. Sorts are organized in a fuzzy subsumption relation that extends to OSF terms, record-like structures representing classes of entities. The unification algorithm for these structures provides a calculus of fuzzy type subsumption. The latest release of fosf introduces support for similarity-based reasoning by extending the OSF language with a similarity relation among sorts, alongside a translation engine for compiling OSF terms into standard SPARQL. Queries expressed as OSF terms undergo fuzzy ontology-aware rewriting prior to translation. This process incorporates an OSF theory of sort definitions, feature domain and range declarations, and fuzzy subsumption and similarity relations through fuzzy OSF constraint normalization. This rewriting enables flexible query answering by returning approximate solutions, allowing queries that would otherwise be inconsistent with the underlying ontology to yield relevant results. 

---
# TwinLab-Bench: Evaluating AI Scientists in Agentic Twins of Real Labs 

**Authors**: Yang Wan, Linchao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26248)  

**Abstract**: AI agents are increasingly expected to act as AI scientists, which requires three increasingly open abilities: carrying out a given computational workflow correctly, choosing the next measurements from a given design space, and designing their own experiments. Existing benchmarks mainly evaluate the first two abilities, in a terminal or in a closed loop that selects from predefined candidates. Evaluating the third ability requires reliably executing an experiment that an agent designs, with readings grounded in a real laboratory. We introduce TwinLab-Bench, which evaluates AI scientists on self-designed experiments in 52 tasks from laboratories in the physical sciences, the life sciences, and engineering. The benchmark supports more than 100 experimental action types, and the agent's report is scored by comparison with a hidden answer taken from real laboratory records. Each task has its own TwinLab agent, an agent that serves as the environment and executes the experiments that the tested agent designs, reading the raw materials of the laboratory and calling calculators that are built from its records. Compared with a language model that sees only the task description, the TwinLab agent almost eliminates readings that deviate from the real records and inconsistent answers to a repeated request, and closed-book probes control for answers recalled from memory. We find that these tasks remain hard for frontier models: the strongest, Claude Fable 5.2, averages a score of only 45.0, and we conduct an empirical study of where it fails. 

---
# Physics-Informed Feature Representations and Pretrained Tabular Foundation Models for Generalizable Melt Pool Geometry Prediction 

**Authors**: Rafid Enayet, Imtiaz Ahmed, Md Habibor Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2611.26235)  

**Abstract**: Developing generalizable, data-efficient machine learning (ML) models for advanced manufacturing remains challenging because experimental datasets are limited and process-property relationships vary across materials and operating conditions. In laser powder bed fusion (LPBF), melt pool width, depth, and melting mode (conduction, transition, or keyhole) govern process stability and defect formation. Conventional ML models typically use dimensional process parameters, which represent laser-material interactions only implicitly and may limit cross-material transfer. This study evaluates whether two physics-informed dimensionless features, normalized enthalpy and Peclet number, combining laser processing conditions with material thermophysical properties, improve melt pool prediction over raw process parameters. We systematically benchmark seven models: four classical ML methods (Gaussian process, support vector machine, random forest, and XGBoost) and three pretrained tabular foundation models (TabPFN, TabDPT, and TabICLv2). Experiments span 316L stainless steel, IN625, IN718, and combined-material datasets, covering regression of melt pool width and depth and classification of melting modes. Dimensionless features yield comparable or superior predictive performance in most settings, with the largest gains in cross-material depth prediction (R^2 = 0.96). Tabular foundation models generally achieve stronger classification performance than classical baselines, particularly for multi-material datasets. These findings indicate that combining physically meaningful dimensionless representations with pretrained tabular models can improve data-efficient melt pool modeling and support more transferable process-window identification in LPBF. 

---
# The FrontierOR Challenge: Agentic Algorithm Design for Operations Research 

**Authors**: Hai Wang, Bretislav Hajek, Chonghe Jiang, Minwei Kong, Ao Qu, Huatian Gong, Bo Dong, Cynthia Barnhart, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.26171)  

**Abstract**: Operations research (OR) relies on algorithms that experts tailor to each problem: a new routing, scheduling or network design problem can take weeks of specialist work before it is solved well at scale. Large language model (LLM) agents now write, test and improve code, but whether they can design algorithms that scale to realistic instances is open: in the FrontierOR benchmark's one-shot evaluation, the strongest model matches Gurobi, with an objective within 1% of Gurobi's and no slower, on only 39% of the large instances. The FrontierOR Challenge asks teams to build agentic workflows that design, test, and improve algorithms for many optimization problems at once. Stage 1 releases 60 of the benchmark's 180 audited problems, spanning six application categories and five formulation types, in weekly rounds of 15; Stage 2 introduces new problems outside the benchmark, with unseen instances. Two tracks are ranked separately: in the Main track we run the submitted algorithms in a network-isolated sandbox with open-source dependencies only; in the Self-hosted track teams solve the instances on their own machines, with any software, within a timed window. Every solution is checked for feasibility, and a feasible solution scores the ratio of its objective to a frozen known reference, averaged over hidden instances. 

---
# Sage-Evolve-Preview:Adaptive Observation for Post-Training Self-Improvement 

**Authors**: Yinhao Tang, Youqing Fang, Yanan Sun, Xintian Shen, Hanyu Sun, Wenran Liu, Yue Wu, Weiming Zhang, Bin Liu, Wenwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26132)  

**Abstract**: Post-training recursive self-improvement (RSI) iteratively improves a target capability, operationalized by a fixed anchor benchmark, through repeated intervention, training, and evaluation. Yet most loops use the anchor's aggregate score as their sole observation, so early gains give way to stalled progress and repetitive exploration. When this anchor-only feedback ceases to reveal actionable gaps, a plateau may reflect an exhausted observation rather than an exhausted capability. We introduce adaptive observation (AdaObs), an observation-side mechanism of harness evolution that keeps the anchor objective fixed while adapting the active observation. At a plateau, a controller refines diagnostic granularity, escalates difficulty, and adds related tasks; validated increments are fused back into training, while every checkpoint remains evaluated against the anchor. Across eight independently optimized anchors in long-context reading comprehension and instruction following, AdaObs raises the mean score from $66.4$ to $73.7$ under a matched data budget, outperforming FixedObs, the anchor-only baseline. The evolved harness further improves the seven-task average on the independent PostTrainBench across two agent backends. These results identify observation adaptation as a missing mechanism of harness evolution for sustaining post-training RSI. 

---
# SenseAlign: Learning Decision-Aligned Predictive Representations for Active Urban Sensing with UAVs 

**Authors**: Weijiang Xiong, Nikolas Geroliminis  

**Link**: [PDF](https://arxiv.org/pdf/2611.26109)  

**Abstract**: Active urban sensing orchestrates a mobile sensor fleet to proactively inspect a dynamic road network and gather critical information for system management. This problem simultaneously requires understanding the complex spatio-temporal evolution and making decisions with partial observations. In this work, we focus on active sensing in a large-scale urban road network with utilities grounded in dynamic traffic states, while prior work has rarely considered these factors jointly. We propose \emph{SenseAlign}, a Reinforcement Learning-based (RL) framework that learns decision-aligned predictive representations for coordinating an urban sensing drone fleet. SenseAlign first trains a privileged teacher agent that exploits future ground-truth traffic for optimal sensing choices. The student agent has an auxiliary branch that learns to predict future traffic with partial observations, in addition to the RL-trained actor and critic branches. During student learning, our framework matches the action distributions of the teacher under ground-truth and student-predicted future traffic, while keeping the teacher's observations unchanged. Such a design shapes the latent space for decision-making instead of prediction alone. At inference time, the student agent has no overhead as all this learning is done in training. We conduct comprehensive experiments with the real road network of central Barcelona containing 1570 road segments, and demonstrate that our method consistently outperforms the baselines on multiple utility formulations. The codebase will be made publicly available upon paper acceptance. 

---
# Pareto-Optimal and Strategyproof Data-Sharing Between Competitors 

**Authors**: Morgan Colin-Mourre, Felipe Garrido-Lucero, Patrick Loiseau  

**Link**: [PDF](https://arxiv.org/pdf/2611.26108)  

**Abstract**: We study data-sharing among firms competing in the same market through the quality of their predictive models. We model firms' profit through two main components, precision and market share, where the former depends on the firm's own number of datapoints and the latter depends on the endowments of all firms. We characterize Pareto-optimal contracts as those in which at least one firm receives all of the other firms' datapoints. We then propose a parametric family of data-sharing contracts and show that, under any of them, the relative market shares of all firms remain constant, eliminating the negative effect of sharing data with competitors and equalizing all firms' gains relative to the no-sharing scenario. We further show that one contract within this family satisfies both Pareto-optimality and strategyproofness against firms attempting to under-declare the size of their initial endowment. Finally, using a real-world dataset, we empirically test the gain in profit for all firms under the contract described above, and show that data-sharing is particularly beneficial when the machine learning task is harder or when firms' initial endowments are closer to homogeneous. 

---
# Controlling the Humanoid Robot H1-2 With a Simulated Biologically Plausible Network 

**Authors**: Fatima Atwi, Constant Roux, Christian Bergaud, Olivier Stasse  

**Link**: [PDF](https://arxiv.org/pdf/2611.26074)  

**Abstract**: Artificial Spiking Neural Networks have gained remarkable attention in the last few years due to their proven ability in simulating biological neural cultures and producing similar activity and trends. Nevertheless, whether they can be trained for humanoid locomotion while retaining biological plausibility and robust performance at once remains an open question. We study this problem by proposing a biologically inspired spiking network, trained by different learning schemes and tested on the H1-2 Unitree Humanoid. These schemes differ in how recurrent weights are trained. Ordered from most to least biologically faithful, they are: a fixed reservoir mimicking an in-vitro neural culture, where recurrent weights are kept frozen and only a linear readout is trained, the same network but with its synapses trained by reward-modulated spike timing dependent plasticity and eligibility propagation, and finally backpropagation through time as a non biological baseline for comparison. All schemes relied on distillation of a reinforcement learning expert policy with dataset aggregation. We show that e-prop reached 100% survival rate matching backpropagation through time, whereas both the fixed reservoir and the reward-modulated STDP, though the closest to biological cultures, plateaued at a much lower performance (~43.7% and ~11% survival). We explore in further detail which trends a real culture could and could not produce until now. Finally we demonstrate brief bipedal stepping on a full size humanoid H1-2. 

---
# Studying AI Act Mentions: A first look at the impact of the European Union's AI regulation on Hugging Face 

**Authors**: Anamaria Mojica-Hanke, Tessi Aigbovbiosa-Igbinovia, Steffen Herbold  

**Link**: [PDF](https://arxiv.org/pdf/2611.26050)  

**Abstract**: The European Union's Artificial Intelligence Act (AI Act) entered into force in August 2024, and its obligations apply in stages. Obligations for providers of general-purpose AI models have applied since August 2025, and transparency rules since August 2026. These obligations affect developers who share and deploy models and applications on public platforms such as Hugging Face (HF), the largest platform for distributing AI artifacts. However, little is known about how practitioners respond to the regulation in practice. To address this gap, we conducted an empirical study of explicit references to the AI Act on HF. We collected over five million records, including repository cards, discussions, and commits from Models, Datasets, and Spaces, and identified over 1,500 records that explicitly mention the Act. Two coders manually labeled these records using a primarily closed coding approach, and we analyzed the resulting categories per source. We find that explicit references to the AI Act remain rare across all repository types. Repository cards mainly contain claims of compliance and responsibility, like ensuring compliance or assigning responsibility to downstream users, and some claim that the Act does not apply because the model is not a general-purpose AI model. Discussions are used to report alleged violations of the Act, while commits mainly update artifacts that support compliance. Overall, compliance statements focus on who is responsible rather than how compliance is achieved. We believe that this calls for direct platform support, e.g., automated checks and documentation templates, as well as clearer guidance on the scope of the AI Act. 

---
# SEOYUN: Refusal-Inducing Exam Documents for AI-Glasses Pipelines 

**Authors**: Dohyeon Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2611.26033)  

**Abstract**: Camera-equipped glasses can convert an examination page into a query to a vision-language model (VLM). Device detection addresses possession; a complementary question is whether an examination document can reduce the usefulness of an unauthorized capture. We present SEOYUN, an application-specific framework for refusal-inducing exam documents, with optical and encoder-level interventions as extensions. We use reverse jailbreaking to describe the direction of the objective: inducing abstention rather than bypassing refusal. Refusal induction and defensive visual prompt injection already have precedents; our contribution is a threat model, analytical limits, and an evaluation protocol for examination use. We distinguish refusal from correct-answer leakage, show why removing an added dummy option is not itself a reduction in the original answer space, and derive a sampling comparison under explicitly stated camera assumptions. The protocol separates digital, simulated, and physical inputs and requires held-out evaluation after subject-specific selection. A local CSAT candidate corpus and rendering harness have been prepared, but this version reports no model-performance measurements, physical validation, or human-subject results. Accordingly, SEOYUN is a research framework, not a validated defense or a basis for allegations against individuals. 

---
# Serialization Tax in Shared-Latent Exchangeable Decisions 

**Authors**: Siming Zhang, Zhehui Shen, Shijie Chen, Xinle Gu, Yansen Yu, Hang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26029)  

**Abstract**: Accurate item probabilities can produce inaccurate group-level decisions. A system may score each item separately, yet act on whether enough hidden items succeed: show a slate, defer a panel decision, or inspect a batch. When items share an uncertain latent state, independent marginalization preserves means but removes dependence from the aggregate posterior. We call this error the serialization tax. For binary items sharing one success rate, classical convex-order comparisons give a more dispersed exact count law than any independent completion with the same total mean. We quantify variance and total-variation gaps, derive threshold separation under margin conditions, and connect posterior approximation to decision regret. Separate results cover non-binary outcomes, heterogeneous items, and sufficient posterior resolution. TaxScore identifies large hidden blocks with unresolved uncertainty near a decision boundary for routing to richer posterior heads. On pre-specified, hidden-target-free MovieLens 1M/10M audits, existing shared-latent heads improve high-score negative log-likelihood (NLL)/utility from 0.673/0.696 to 0.629/0.723 and from 0.670/0.691 to 0.639/0.715. Routing gives 0.006--0.012 full-population utility gains, with matched controls identifying their sources. Our code is available at this https URL. 

---
# ForeSeek: Foresight-Driven Reasoning for Long-Form Video Understanding 

**Authors**: Yuchen Sun, Siyuan Cheng, Shi-Zhe Chen, Yixin Qin, Jinwen Luo, Zheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.26024)  

**Abstract**: Long-form video understanding, as a fundamental yet crucial vision task, is being revolutionized by multimodal large language model (MLLM)-based video agents. However, the retrieval-driven nature inherent to these agents creates a significant bottleneck, forcing them to reactively fetch clips based on local saliency without a global temporal plan. To address this, we present ForeSeek, a foresight-driven video agent that transforms long-video reasoning from reactive retrieval into predictive exploration. Rather than blindly probing the timeline, ForeSeek forms visually grounded expectations over unseen temporal regions, verifies them through targeted observations, and continuously refines its search based on emerging evidence, where failed expectations serve as informative signals for principled backtracking. Furthermore, we develop a two-stage training framework that equips MLLMs with foresight-guided reasoning and efficient decision making. Extensive experiments on five long-video benchmarks demonstrate that ForeSeek consistently improves the efficiency--accuracy trade-off of long-video reasoning, achieving competitive or superior accuracy while observing $2.5$--$4.4\times$ fewer frames than the strongest search-based agent. Code is available at this https URL 

---
# When Agents Play Differently: Localizing Human--Agent Behavioral Divergence with Conditional Distributional Testing 

**Authors**: Oleg Smirnov, John Pertoft, Sara Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2611.26012)  

**Abstract**: Evaluating whether artificial agents behave like humans requires comparing not just average outcomes but entire behavioral distributions across diverse contexts. We propose a stratified framework that computes nonparametric discrepancy statistics within contextual cells, and discovers subgroup rules identifying where agent behavior departs from human play. We show that session-level metrics dominated by win/loss outcomes offer little signal beyond mean comparisons, whereas process-level metrics such as move efficiency reveal genuine behavioral differences. Applied to a commercial match-3 game, it uncovers patterns invisible to aggregate metrics: a playtesting bot is diverging substantially from humans in certain game modes, while showing no detectable divergence in others. We release our implementation as open source to support reproducibility and adoption. 

---
# System-One RSI: Building Muscle Memory for Interactive Agents through Guided Practice 

**Authors**: Qingxu Fu, Xuanfa Jin, Ziqian Chen, Zhaoyang Liu, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2611.26007)  

**Abstract**: Reasoning models provide strong multimodal task understanding but struggle with sub-second control because deliberation increases decision latency. Fast decision models can meet the response window, yet prompting and few-shot examples do not readily provide the fine-grained experience required by a new task. Inspired by coached motor learning, this paper presents System-One Recursive Self-Improvement (System-One RSI), which couples two loops. The high-frequency online inner loop is an Observe, Orient, Decide, Act (OODA) cycle: a decision model maps each visual observation and task question to structured judgments, and a decision tree converts them into timed actions without invoking a reasoning model. The offline outer loop is a recursive improvement cycle carried out by a team of reasoning agents. The team revises the decision-model prompt and decision tree, captures interaction records, derives supervision from those records, and trains or deploys updated model parameters. The improved system then returns to the environment and generates the experience for the next round, consolidating learned distinctions as external muscle memory. The framework operates with a small number of real episodes, without expert demonstrations, environment modification, sandboxing, or parallel simulation. Experiments in Snake, Zenless Zone Zero, and Black Myth: Wukong use a 27B decision model. Mean Snake score increases from 1.30 to 7.35. In an evaluated scenario of Zenless Zone Zero, median completion time decreases by 8.4%. In an evaluated combat scenario of Black Myth: Wukong, the evasive policy achieves 11 times the mean survival time of a stationary reference, while the offensive policy win rate rises from 0% to 66.7% across the evaluated configurations. 

---
# AI agents can redesign engineering parts with more than a thousand CAD parameters and nonlinear simulations 

**Authors**: Alexandra Starostina, Alexander Hermann, Mahish Guru, Kartik Bali, Roland Aydin, Christian C. Cyron  

**Link**: [PDF](https://arxiv.org/pdf/2611.26003)  

**Abstract**: Simulation-driven design of real engineering components has to meet three conditions at once. It runs on few solves, since a nonlinear finite-element simulation of a 3D part takes hours; its design space has many values, the hundreds to thousands of values of the part's CAD model; and a better design often needs a redesign, coordinated changes of the part's proportions rather than local polishing of its surface. Automated optimisers meet at most two of these conditions. Classical optimisers, Bayesian or derivative-free, need a person to cut the design down to a few parameters, and on all the values they need far more solves; industrial shape optimisers move every surface node of the mesh in one solve per cycle, but only in small steps that polish rather than redesign. We ask whether a general-purpose large language model (LLM) can meet all three by joining what it knows with what the simulation shows. GPT-6 Astra edits a magnesium stent through a CAD geometry parametrisation of about 1,650 values, with a budget of 15 simulations; every proposal is built in CAD, meshed and simulated with large-deformation plasticity and contact, and the LLM reads the full results of every solve. Within that budget it lowered the objective, a high percentile of the strains in the stent's bends, by 32%; on the same budget, the industrial shape optimiser Tosca improved the original design by a tenth and then stopped improving. An earlier GPT-6 Astra design, 25% lower in the same objective, was checked in an independent, higher-fidelity model of the full deployment with crimping and damage: it was 14% less strained than the original after the full chain, and its most damaged spot began to fail later and carried 63% less damage at full expansion. LLMs can thus join their learned knowledge with simulation data to adjust more than a thousand design values and reach valid designs with real improvements. 

---
# From AI Surrogate to Engineering Tool: Toward Production Use of Fiber-Orientation Prediction in Crash CAE 

**Authors**: Osamu Ito, Yoshikazu Nakagawa, Tomoaki Tatsukawa  

**Link**: [PDF](https://arxiv.org/pdf/2611.25991)  

**Abstract**: Crash simulations of glass-fiber-reinforced plastic parts are highly sensitive to the fiber-orientation field supplied to the material model. In conventional workflows, that field comes from resin-flow analysis, which requires molding-specialist support and takes about 25 hours of work per part. We previously proposed a surrogate model that addresses this bottleneck by predicting the orientation field in about 3 minutes. This paper reports the implementation, validation, and operational design that turned the model into an engineering tool for Honda design workflows. We present staged validation from 2D plates to production parts, evaluation against a predefined engineering angular-error criterion, and the expert review that approved design use in March 2025; the system has since been transferred to the crash-safety group and is in initial, limited use in directed fracture-prediction scenarios. The developed system combines injection-gate setting support for cases in which gates are unknown (90% gate-count accuracy, mean position error 11% of flow length), voxelization, surrogate inference, automatic mapping to the crash CAE model, and report generation in a single GUI. The complete orientation-preparation workflow takes about 0.1 hours, a 99.6% reduction from the conventional 25-hour flow-analysis route, and can be run directly by crash CAE engineers without molding-specialist support. As evidence of practical impact, component-level crash analysis using the predicted orientation field reduced the absolute peak-force error from 19.0% with an isotropic baseline to 11.4%, while confining predicted fracture to the test-observed region. The deployment lesson is that a surrogate model alone is not enough for practical adoption: gate-setting support, automatic data conversion, and shareable reporting are what make the model usable as a design tool. 

---
# GaussTrans: A Gaussian Interface Framework for Multi-Layer Refractive Transport 

**Authors**: Zixuan Zhao, Tao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25950)  

**Abstract**: We investigate surface modeling of multi-transparent interfaces to achieve accurate light transmission. The interplay of errors in geometric structure, light paths, and observation results during the reconstruction of transparent media with a high number of layers makes this a challenging problem. Some existing techniques address the reconstruction of transparent scenes by reducing the number of layers or simplifying light paths; however, this limits their direct applicability to high-order transparent scenes. We propose GaussTrans, a compact framework for light transport in high-order transparent scenes, consisting of three components: Gaussian-Field Hybrid Surfaces, Transport-Coupled Refinement, and Primitive-Guided Implicit Intersection. Gaussian-Field Hybrid Surfaces use a hybrid approach combining Gauss surfaces and fields to reconstruct different interfaces. Transport-Coupled Refinement reduces coupling errors by evaluating the entire light path. Primitive-Guided Implicit Intersection combines primitive support regions to reduce errors in the intersection of interfaces and rays. In experiments on datasets such as RefRef, GaussTrans achieves significantly higher accuracy with relatively low spatial storage requirements. 

---
# ROAM-GRPO: Enforcing Reasoning Diversity in GRPO via Routed Off-mode Approach Mining 

**Authors**: Jeff Duan, Matt Thomson  

**Link**: [PDF](https://arxiv.org/pdf/2611.25926)  

**Abstract**: Supervised fine-tuning (SFT) paired with Reinforcement Learning with Verifiable Rewards (RLVR) has become the dominant recipe for post-training LLMs on complex reasoning tasks, but has been shown to collapse the policy into a low-diversity mode. Repeated LLM outputs are nearly identical, and pass@k accuracy converges or degrades as a result. We propose ROAM-GRPO (Routed Off-mode Approach Mining for GRPO), an augmentation to the GRPO algorithm which incentivizes exploration of distinct reasoning strategies and aims to increase output creativity during evaluation. ROAM-GRPO detects well-solved problems during training and transfers them to an "off-mode" track. The model is prompted to produce distinct solution strategies to such problems, which are verified for logical validity and distinctness by an LLM judge and subsequently self-distilled into the policy. Training Qwen3.5-2b on a 1,000-problem subset of the MATH lvl5 dataset, ROAM-GRPO implementation increases peak pass@16 by 4.7% and correct embedding cosine distance by approximately 1.5-2x on a difficult evaluation set. 

---
# Validation-driven Deployment of LLM Inference Platforms for Agentic Workloads 

**Authors**: Shai Gretz, Alon Halfon, Yoav Katz, Lena Dankin, Michal Jacovi, Rachel Tzoref-Brill, Kfir Toledo, Elron Bandel, Natan Ergas, Michal Malka, Pravein Govindan Kannan, Maroon Ayoub, Michal Shmueli-Scheuer  

**Link**: [PDF](https://arxiv.org/pdf/2611.25905)  

**Abstract**: Enterprises are moving agentic workloads from closed APIs to their own open-weight model deployments. For this reason, lifecycle-wide validation, from small-scale testing through sizing and pre-production to production, becomes all the more important. Typically, rollouts are evaluated with ad hoc tools whose accuracy is rarely checked. We present an end-to-end evaluation pipeline for agentic LLM serving and empirically ground the design choices that connect its stages. Our experiments show that replaying captured traces reproduces the per-session load of the original agent sessions: at the 95th percentile replay stays within 1.18x of the recorded input tokens while live re-runs reach 4.98x, so a small set of recorded sessions yields a repeatable per-session token load. A code-coverage analysis of vLLM further shows that the evaluated live, replayed, and synthetic workloads are complementary in this configuration: each exercises Python serving paths not reached in the measured runs of the others. In a case study rolling out four large mixture-of-experts (MoE) models on llm-d, experiments show that the target models reached about half the optimal concurrency, and correspondingly about 60% of the throughput, on captured workloads compared with public traces. We open-source the relevant tools and release a trace dataset. 

---
# From Cases to Criteria: Reusing Evaluation Knowledge for Task-Specific Rubric Generation 

**Authors**: Qiyan Deng, Long-Kai Huang, Kangfei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.25903)  

**Abstract**: Query-specific rubrics make LLM evaluation more fine-grained and transparent, and generating high-quality rubrics requires both identifying the target query's evaluation scope and specifying concrete checks. Historical rubrics provide reusable evaluation experiences, yet existing approaches reuse either complete query-rubric cases or aggregated criteria and dimensions without explicitly connecting recurring evaluation requirements to their criterion-level evidence and source-query context. This can preserve irrelevant task details or lose the broader evaluation structure. In this paper, we introduce RubricDuo, which organizes historical query-rubric pairs into a Rubric Knowledge Graph linking evaluation dimensions, shared concepts, local concepts, criteria, and queries, where reusable evaluation requirements across queries are abstracted by shared concepts. Given a target query, RubricDuo first combines concepts relevant to the target query with retrieved historical cases to draft a rubric with broad coverage. It then revisits each draft criterion, retrieves concept-linked historical criteria and their source queries, and adapts applicable evidence to refine the criterion. Experiments on RubricBench and HealthBench demonstrate that RubricDuo achieves the overall best performance among the evaluated automatic methods in both alignment with human-authored rubrics and downstream LLM judging. 

---
# The Price of an Explanation: What Demanding Verifiable Artifacts Costs, and What It Buys Back 

**Authors**: Edgar A. Duéñez-Guzmán, Eduardo Dueñez, Edgar Said Hernandez Sanchez, Suzanne Sadedin  

**Link**: [PDF](https://arxiv.org/pdf/2611.25896)  

**Abstract**: Generative models produce many candidate solutions but provide no intrinsic signal by which to rank them. One mitigation is to demand a \emph{verifiable artifact}: a program that can be executed against the available evidence. A common concern is that requiring executable verification incurs costs in both solution quality and computational budget. We measure these two costs on the $120$-task ARC-AGI-2 evaluation set. Demanding a verifiable artifact from an LLM reduced the score from $93$ to $68$ of $120$ tasks ($77\%$ to $57\%$). Of those $25$ tasks, $21$ were reached by no program-producing configuration we ran, at any budget, by sampling or by search; yet for $27$ of the $28$ tasks that only the answer-only arm solved, some program arm held a program that reproduced every demonstration and still failed the test. The bottleneck is obtaining an explanation that \emph{generalizes}, not finding one. A short evolutionary search over two already-sampled programs, at no additional model cost, beats best-of-$2$ sampled code by $9.2 \pm 1.4$ tasks, and over a single sampled program beats $1$-shot sampling by $9.9 \pm 1.3$. Moreover, because a program can be checked against the training examples, any program-producing protocol can terminate early once a task is explained: under the same stopping rule, best-of-$n$ sampling declines up to $38\%$ of its budget and the evolved arm up to $49\%$, whereas an answer-only baseline cannot terminate early. Applying the same methodology to decoder synthesis for quantum error correction yields interpretable decoders evaluated against a certified panel of published decoders on held-out exam shots. 

---
# One Schedule Does Not Fit All: Adaptive Inference Schedules for Diffusion Language Models 

**Authors**: Rabimba Karanjai, Linsey Pang, Hemanth Hegadehalli Madhavarao, Richard Williamson, Uma Kona, Prakhar Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2611.25895)  

**Abstract**: Diffusion language models typically use a fixed denoising schedule across workloads. We show that the cheapest schedule that preserves quality depends on the task, model, and sampler. We propose Task-Calibrated Denoising Allocation (TCDA), a deployment rule with an explicit error guarantee. Given candidate samplers, paired calibration data, a utility tolerance, and guardrails against degeneration, TCDA uses simultaneous one-sided bootstrap bounds to certify which candidates meet both the utility and guardrail criteria. It deploys the cheapest certified candidate or falls back to the full schedule. The probability of deploying an infeasible schedule is asymptotically at most alpha. Candidates include early prefix decoding, state-reusing continuation, reduced time grids, and confidence-threshold decoding. Across seven datasets and two model families, TCDA certifies reductions in denoiser evaluations by factors of 3.4 on XSum and 2.0 on WMT14 for ELF-B; 8.0 on WMT14 and 4.1 on GSM8K for LLaDA-8B; and 1.33 on OpenWebText for both ELF-M and ELF-L. A validation protocol fixed before generation confirmed every selection on fresh seeds. Across 500 calibration splits, false certification remained below 2% in every setting at a nominal level of 5%. Certified schedules vary across tasks and model scales. Trajectory measurements suggest that when decoded tokens commit is more informative for schedule selection than the remaining latent discretization error. We also find that the ELF reference sampler draws its time grid at random, so reproducible schedule comparisons for unconditional generation require fixing the grid. 

---
# Explainable AI for Solar PV Power Forecasting: From Model Selection to Trustworthy, Reproducible Deployment 

**Authors**: Nuzhat Khan, Muhammad Paend Bakht, Ab Al-Hadi Ab Rahman, Fabiano Pallonetto, Amy Fahy, Mohamed Khalafalla Hassan, Paul Stynes, Muhammad Anique Aslam, Terence O Donnell, Mohammad Riadh  

**Link**: [PDF](https://arxiv.org/pdf/2611.25894)  

**Abstract**: Machine learning (ML) is now a standard tool for forecasting the output of solar photovoltaic (PV) plants, yet published accuracy figures are often hard to compare and harder to trust. Many studies validate with random splits, report no reference forecast, and present feature attributions without saying what they can and cannot show. This paper reviews the main families of ML methods for PV power forecasting and resource assessment, from tree ensembles to deep and foundation models, and sets out a reproducible workflow that combines automated model search with explainable AI. The workflow is then tested on an openly available dataset: a 16.32 kWp plant in Porto, Portugal, forecast one day ahead from a 4 km numerical weather prediction grid. Over a full test year, the best models reach a normalised root mean square error (nRMSE) of 11.6% of installed capacity, about half the error of smart persistence. The choice of input features changed accuracy far more than the choice among strong algorithms, and shuffled cross-validation was mildly optimistic. After conformal calibration, the nominal 80% prediction intervals covered 81.7% of observations, against 64.5% before. SHAP rankings stayed stable when the model was retrained on different periods but shifted when correlated inputs were removed, so attributions should be read as properties of a model rather than of the plant. The paper closes with practical lessons and open problems, including deployment on edge hardware and the effect of a changing climate. 

---
# Meta-RTL: Evolving Agent Harnesses for RTL Optimization 

**Authors**: Xinzhuang Xiong, Yonghao Wang, Shengyao Sun, Chengxi Li, Evangeline Young  

**Link**: [PDF](https://arxiv.org/pdf/2611.25891)  

**Abstract**: Large language models enable agents to optimize RTL using EDA feedback, yet their executable search procedures remain largely hand- engineered. We present Meta-RTL, an EDA-grounded framework that evolves RTL optimization harnesses from a minimal loop while keeping the optimization model, EDA flow, and per-design budgets fixed. Compared with Dr. RTL, Meta-RTL* reduces aggregate WNS/TNS violations by 7.8%/11.0% across 20 development designs, with area reductions on 70% of them. Without further evolution, Meta-RTL* transfers to 10 unseen IC-RTL designs, reducing aggregate WNS/TNS violations by 20.5%/14.0% and total synthesized area by 12.6%. We further characterize three search mechanisms in the evolved harness and assess their contributions through ablations and a case study. 

---
# Trustworthy Geospatial AI Workflows: Reproducible Maritime Analytics and Forecasting on Open-Source AIS Data 

**Authors**: Divya Acharya, Thomas M. Roehr, Arnaud Gotlieb  

**Link**: [PDF](https://arxiv.org/pdf/2611.25884)  

**Abstract**: Publicly available geospatial data provide increasing opportuni- ties for data-driven maritime analytics and predictive modelling. However, their availability does not necessarily imply that the underlying observations are equally suitable for downstream AI applications. Properties of the data source and observation pro- cess, including provenance, reporting regularity, spatial coverage, observation density, and missingness, can affect how geospatial data are interpreted, processed, and ultimately used by predictive models. At the same time, trustworthy geospatial AI requires ana- lytical workflows in which these processing decisions can be made explicit and reproduced. This paper presents a reproducible mar- itime geospatial AI workflow built around damast, an open-source data-processing and machine-learning library developed at Simula Research Laboratory. Using AIS observations from multiple public data providers as a case study, we demonstrate metadata-aware validation, reusable processing pipelines, maritime trajectory con- struction, and reproducible machine-learning experimentation. The predictive stage is model-agnostic and is instantiated in this study using a Transformer-based vessel-trajectory forecasting model. Building on these implemented capabilities, we further outline an envisioned extension toward trustworthiness-aware geospatial AI. The extension introduces explicit characterization of source- and data-level properties and their relationship with downstream model performance. This distinguishes capabilities demonstrated in the current workflow from trustworthiness-oriented components identified for future development. Overall, the study demonstrates how reproducible open-source infrastructure can provide a practi- cal foundation for developing more transparent and trustworthy geospatial AI workflows. 

---
# Coherent belief and opinion propagation produces more echo chambers: corrected version 

**Authors**: Hiro Kataoka, J{é}r{ô}me Euzenat, Koji Hasebe  

**Link**: [PDF](https://arxiv.org/pdf/2611.25879)  

**Abstract**: The occurrence of echo chambers is a well-known phenomenon in social networks. Echo chambers are groups of agents satisfying three properties: homogeneity, i.e., sharing the same opinion, segregation, i.e., listening primarily to agents of the same group, and reinforcement, i.e., increasing opinion similarity over time. Opinion dynamics models have been used to explain how echo chambers occur through agents updating their opinions. However, social network users also exchange beliefs supporting their opinions. Here, we aim at identifying the effect of propagating beliefs together with opinions on the formation of echo chambers. For that purpose, we complement an existing opinion dynamics model with classical belief merge operations. Beliefs and opinions are also maintained coherent guided by values that agents share. We introduce measures, applying to either beliefs and opinions, to identify echo chambers based on the three properties. This model is used to simulate opinion and belief propagation with varying initial conditions and agent tolerance for different opinions/beliefs. Results confirm the occurrence of echo chambers observed in previous opinion dynamics studies in a different setting and with a stricter measure. They also demonstrate the occurrence of echo chambers with belief propagation. Finally, they show that connecting opinions to beliefs and maintaining their coherence increases the number of belief echo chambers, but that they involve a reduced number of agents. 

---
# Generalist Meets Specialists: An Evolving Multimodal Agent with Persistent Tool Orchestration for Traffic Scenario Understanding 

**Authors**: Linjia Kang, Binhan Jiang, Juan Yang, Zhiqin Song, Xiangsen Wu, Qionglin Ren, Jiang Liu, Zhongjiang He, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.25868)  

**Abstract**: Traffic scenario understanding requires combining fine-grained visual evidence with spatial and temporal reasoning. Tool-augmented vision-language models (VLMs) can use specialized perception tools, but effective collaboration faces two challenges: terminal rewards provide limited guidance for intermediate decisions and error recovery, while repeated planning adds unnecessary cost across similar tasks. Echoing the fast-slow dichotomy of human cognition, we propose StarTV-Agent, which combines policy learning with experience evolution to support adaptive reasoning when needed and efficient reuse of prior experience when possible. During training, Role-Sensitive Process Optimization (RSPO) combines process advantages with terminal advantages, while Weakness-Driven Demonstration Learning (WDL) provides validated teacher demonstrations targeting the student's weak questions. During inference, Lightweight Adaptive Workflow Reuse (LAWR) enables a fast path by selecting suitable workflows and binding task-specific arguments and resource references, while a slow ReAct-based path handles tasks without suitable workflows. We further design label-free feedback based on execution completion, workflow adoption, and cost savings to guide workflow reuse and memory updates. Across three challenging benchmarks, StarTV-Agent achieves up to 8.0% relative performance improvement, while reducing token consumption by up to 52.1%. 

---
# Decoupled Knowledge Retrieval and Compact Memory Enable Reliable Lifelong Model Editing 

**Authors**: Yue Che, Jingze Wang, Chen Guoyi, Yujie Wu, Xuming Ran, Mingkun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25861)  

**Abstract**: Lifelong model editing requires continuously incorporating new facts into deployed large language models (LLMs) without expensive retraining or compromising existing capabilities. Memory-based editors offer an appealing solution by keeping the base parameters untouched, yet their practical use is hindered by linearly growing storage requirements and deteriorating retrieval performance as edits accumulate. To tackle these problems, we introduce BiSpace, a lifelong model editing framework that decouples memory retrieval from knowledge injection across two distinct representation spaces. BiSpace routes queries and identifies relevant edits in an external semantic space, while injecting the corresponding knowledge into the internal representation space of the LLM through sparse and storage-efficient updates. Extensive experiments show that BiSpace shrinks each entry's storage to only about 10.2% of the original and shows strong resilience to long edit sequences of up to 16K edits. On a more challenging benchmark, BiSpace further improves Overall Performance by more than 23% over existing methods. These results advance memory-based editing toward reliable long-term model maintenance at scale. 

---
# HabitusMem: Belief Memory over Evolving User Preferences for LLM Agents 

**Authors**: Yifan Lu, Gaoyang Guo, Wen Chen, Ziyun Zou, Jian Wu, Han Zhu, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2611.25855)  

**Abstract**: Personalized LLM agents must act on a user's current preferences, which change across interactions and are rarely stated in full. Existing memory systems store and retrieve content from the interaction history and leave the agent to work out which preferences still hold and where they apply. Such content does not indicate whether a preference is still current or where it applies, so agents often act on outdated or misapplied preferences. We instead treat memory as the assistant's belief about the user. Following Bourdieu's account of habitus as durable but revisable dispositions, we model the user as a latent preference state that every interaction, including the current request, both reveals and updates. The belief summarizes what the history supports about this state, and memory consists of the individual beliefs that bear on the agent's current decision, each with its evidence, scope, and status. We implement this view in HabitusMem, a trained writer that states these beliefs as text records for a frozen agent. Across two agent families, HabitusMem achieves the best results among memory systems on both PersonaMem-v2 and VitaBench 2.0. Controlled analyses show that learned request conditioning drives the answering gain, while auxiliary training tasks improve tool use. 

---
# Whose Risks Count? What 1,901 AI Incidents Reveal About Corporate Disclosures 

**Authors**: Simeon Allmendinger, Edyta Paulina Bogucka, Niklas Kühl, Daniele Quercia  

**Link**: [PDF](https://arxiv.org/pdf/2611.25841)  

**Abstract**: AI systems can harm people, yet the AI actors that provide or deploy them are rarely obliged to disclose these risks. US securities regulation is an exception. Companies registered with the Securities and Exchange Commission (SEC) must disclose their material risks, including those of their AI systems, in filings that carry liability for misstatement. Materiality, however, is defined by what a reasonable investor would consider important, so harms to third parties enter disclosure only insofar as they become financial risks for the company. We test whether AI harms do so by matching 9,001 AI incidents to 18,229 SEC filings of 328 AI actors by risk type. Of the 1,901 incidents with filings before and after them, half belong to a risk type that the AI actor never disclosed, and this share varies with the kind of harm. Filings disclose the risk types of 67\% of system failures but only 2\% of social system harms, and AI actors themselves report fewer than 2\% of incidents. Securities disclosure thus records the risks that AI actors face rather than the harms that people subject to AI bear. Public authorities that hold AI actors accountable for such harms could make them financially material. We release DISC, the Disclosure--Incident Securities Corpus, to support this work. 

---
# Self-Replicating Language Models 

**Authors**: Arend Hintze  

**Link**: [PDF](https://arxiv.org/pdf/2611.25839)  

**Abstract**: Whether large language models (LLMs) and LLM-based agents are a form of life is a scientifically interesting question. Under the Darwinian definition of life, it is also a practical one, because a system capable of Darwinian Evolution can adapt. We therefore test several modes in which a language model can reproduce itself: by emitting a genome that trains its offspring, by copying and mutating its own weights, and by teaching a child from its own samples. In principle, small language models can replicate and evolve: selection improves them, and they adapt to narrow niches. Selection also lowers mutation rates, and a language model, unlike any organism, can lower its own to zero, which may limit its evolution. Flagship LLMs can write such replicators, but for a large LLM to replicate by teaching its offspring remains, for now, technically and economically infeasible. 

---
# Anchor and Check: Deterministic Cross-Prompt, Cross-Model Reconciliation for Auditable LLM Decisions 

**Authors**: Sivakameswara Reddy Chejarla  

**Link**: [PDF](https://arxiv.org/pdf/2611.25818)  

**Abstract**: Pipelines that let a large language model (LLM) make discrete decisions need to know which decisions they may act on unattended, and a record that explains each one. We study a small, fully specified reconciliation contract: commit the target model's greedy answer (the anchor), draw one or two greedy checks that reach the answer by a different route (an alternate prompt, or a model from another family), and let a pure function return one of five outcomes, of which only Agree is acted on. On 48,000 model-item evaluations over 6,000 questions, with every method given access to verbalized confidence as a tie-breaker, the main finding concerns where calls come from: pooling another family's samples raises semantic entropy's AUROC by 1.7 to 4.0 points at every budget, and a same-model temperature check is the weakest design. As complete pipelines, the point estimates of the three-call contract and pooled semantic entropy over 3 or 5 samples differ by less than one point, with intervals containing zero. As detectors of the anchor's errors, the contract is ahead of pooled entropy with 3 and 5 samples under both tie-breaker conventions; with 10 samples the point estimates differ by less than one point and the sign depends on the convention. Under the strict Agree gate it automates 70% of items and still acts on 16% of errors. What the contract adds is a fixed committed answer, an explicit outcome and a decision replayable from its record; generation itself was not reproducible on the hosted API. 

---
# Inject Where It Matters: Objective-Dependent Feature Control in Rectified-Flow Image Editing 

**Authors**: Nina Goncharova, Alexandra Ivanova, Aibek Alanov  

**Link**: [PDF](https://arxiv.org/pdf/2611.25811)  

**Abstract**: Many prompt-driven FLUX editing methods rely on injecting source features into target-conditioned generation, creating an inherent trade-off between source preservation and the desired edit. We address this problem by controlling where source features are injected and which transformer blocks are used. First, we introduce adaptive masks that spatially gate feature injection using differences between source- and target-conditioned block outputs at matched inversion states. Second, we use REINFORCE to select Top-K injection blocks under different preservation-editability objectives. The selected blocks are fixed at inference and require no online search. On PIE-Bench with FLUX.1-dev, spatial masking controls the alignment-preservation trade-off. On 60 validation examples, a learned nine-block set gives lower whole-image LPIPS than the 18-block RF-Edit set within the same masked pipeline, with slightly lower alignment. Among 50 random nine-block sets, it has the lowest LPIPS but not the highest reward. Separate objective sweeps reveal different block preferences for preservation and alignment. 

---
# When Safety Rationales Hallucinate: Diagnosing and Grounding Multimodal Safety Guards 

**Authors**: Yingshuo Wang, Weixiang Zhao, Guancheng Li, Jiahe Guo, Yu Zhang, Yanyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.25793)  

**Abstract**: Multimodal safety guards increasingly generate rationales alongside safety predictions, yet whether these rationales are reliable remains largely unexplored. We systematically study *rationale hallucination* from two perspectives: visual rationale hallucination, where rationales contain unsupported visual facts, and safety attribution hallucination, where grounded content is assigned an incorrect safety interpretation. Across rationale-generation methods, multiple vision-language backbones, and safety datasets, we find that rationale hallucination is widespread and does not consistently diminish with higher classification accuracy or increased model scale. Further analysis of the Think-with-Image paradigm shows that hallucination is concentrated in fine-grained attributes and relations, while visual tools are often invoked infrequently or without sufficient spatial focus. Based on these findings, we propose *Tool-Contrastive Grounding Reinforcement Learning* (TCG-RL), which rewards tool use only when it improves rationale grounding over a tool-free reference. TCG-RL substantially reduces rationale hallucination while maintaining competitive safety prediction performance. By revealing the gap between prediction correctness and rationale reliability, our study motivates a shift in multimodal safety guarding from outcome-only evaluation toward safety decisions that are both accurate and verifiably grounded. 

---
# Duopra: From Full-Duplex Conversation to Full-Duplex Agency 

**Authors**: Chiyu Chen, Pengzhou Cheng, Yuezhao Ma, Xiaocheng Li, Can Liu, Haonan Hu, Yao Yao, Yifei Yang, Yijie Lu, Taolei Wang, Yansi Li, Muhuan Wu, Chenglu Sun, Gongshen Liu, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25773)  

**Abstract**: Full-duplex interaction models exhibit strong conversational agency, enabling natural, low-latency turn-taking, backchannels, and interruptions. However, completing tasks further requires task agency: timely tool invocation and argument revision following user corrections. Frontend-backend agents separate initiation from execution: the frontend decides whether and what to request, while the backend determines how. Execution can be offloaded, but unsent requests cannot be executed, and incomplete ones can misdirect the backend. On Full-Duplex-Bench v3 (FDB v3), most failures of Realtime-Venus-Audio + Harness in our streaming pipeline occur before a domain tool executes. We introduce Duopra, a full-duplex interaction model that learns task execution alongside conversational control. Duopra treats the backend as a tool, accepting a self-contained natural-language request. Direct calls and backend delegation share an action interface and requirements for faithful initiation and correct execution. Its three-stage training combines protocol SFT, full-duplex RL rewarding timely and complete execution, and projector-only repair of talker conditioning after thinker updates. With the same backend, Pass@1 rises from 4.7% to 44.7%; Duopra achieves 47.7% calling tools directly (26.0% after SFT) and 49.3% with a DeepSeek backend. It improves three FDB v1.0 takeover rates over the base model. These results show that task agency can coexist with key conversational agency. 

---
# ADAPT: Adaptive Decision and Abstention with Pool-Size-Aware Thresholds for Service Routing 

**Authors**: Manile Srun, Muntasir Adnan, Arslan Shaukat, Carlos C. N. Kuhn  

**Link**: [PDF](https://arxiv.org/pdf/2611.25754)  

**Abstract**: Retrieval-based classifiers efficiently route queries to specialised services, but they typically select a candidate even when no suitable service exists. We introduce Adaptive Decision and Abstention with Pool-Size-Aware Thresholds for Service Routing (ADAPT), a lightweight routing architecture with an explicit fallback option that enables queries to abstain from specialist selection. ADAPT uses the separation between the highest relevance score and the candidate score distribution as routing confidence. Experiments across retrieval models and increasing numbers of services reveal a consistent empirical behaviour: the confidence distribution changes systematically with candidate-pool size K, making fixed abstention thresholds increasingly miscalibrated. ADAPT addresses this effect with a K-aware threshold that adapts as services are added without classifier retraining. Evaluation across two specialist datasets shows that this simple mechanism provides an effective accuracy-cost trade-off compared with generative LLM routing, supporting lightweight and dynamically extensible service-routing architectures. 

---
# MemMend: A Selective Revision Framework for Memory Question Answering 

**Authors**: Sheng Shi, Geng Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25721)  

**Abstract**: Revising every answer from a long-term memory can consume substantial inference budget and damage facts that were already correct. We introduce MemMend, a modular selective revision framework that routes potentially incomplete answers to an editor, represents corrections as bounded exact-text replacements, and accepts only edits with resolvable source coordinates and valid edit operations. The editor preserves the current answer when no legal repair is proposed; accepted edits require no additional semantic-verification call. On 1,986 LoCoMo development questions, MemMend achieves 67.17 conversation-macro F1 and 88.35 mean Judge score, improving over a verified revision chain by 0.82 and 1.24 points while using 62.37% fewer revision calls. Routing uses 54.15% fewer revision calls than ungated local review, with similar F1 but a 0.67-point lower Judge score. Applied directly to native PropMem, Mem0, and SimpleMem answers, the editing interface raises F1 by 3.79, 6.44, and 0.42 points. It outperforms same-gate rewriting and saves 49.85-83.72% of editing calls relative to ungated editing, which retains higher F1 in this adaptation. A source-by-editing factorial supports local editing for F1 but leaves an independent quality benefit from source obligations unresolved. Transfer to 451 LongMemEval-V2 Small questions also favors revision efficiency, while an external PerLTQA dialogue subset reveals a completeness gap. These results support selective editing as a reusable quality-cost operating point. 

---
# DMT-Attack: Diffusion-Prior Texture Search for Closed-Loop Vision-Language-Action Policies 

**Authors**: Jun Xia, Shangwen Wang, Jun Ma, Ruichao Ren, Zhouyang Jia, Xin Zhao, Huijun Liu, Jie Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25695)  

**Abstract**: Vision-language-action (VLA) policies rely on visual feedback throughout manipulation, making them vulnerable to persistent texture attacks. However, optimized patterns can stand out from their surroundings, making an attack easier to notice and its carrier more likely to be inspected or removed. To address this limitation, our key insight is to search the structured image distribution of a diffusion model for natural-looking textures while using closed-loop feedback to identify task-disruptive candidates. Based on this idea, we propose DMT-Attack, a gradient-free gray-box attack that generates textures for ordinary scene objects such as tablecloths. A frozen text-conditioned diffusion model defines the search space, and an eight-expert latent proposal network is initialized from evaluated teacher latents and refined using rollout feedback. We evaluate DMT-Attack against pi_0.5 and OpenVLA-OFT across four LIBERO suites. DMT-Attack achieves a macro clean-conditioned attack success rate (cASR) of 57.9%, compared with 44.7% for UPA-RFAS and 19.7% for random natural textures. It also attains higher scene similarity and a lower clean-calibrated detector flag rate (45% versus 97.5-100% for optimized patch baselines). These results demonstrate that natural-looking textures can disrupt VLA visual feedback and closed-loop decision-making, motivating defenses that jointly assess visual appearance and task behavior. 

---
# CompAct-Bench: A Benchmark for Working-Context Compaction in Long-Horizon Agent Tasks 

**Authors**: Jiayi Chen, Lizhe Chen, Yujia Chen, Shihan Dou, Ming Zhang, Chendi Ge, Shiguang Ni, Tao Gui, Qi Zhang, Xuanjing Huang, Maxm Pan  

**Link**: [PDF](https://arxiv.org/pdf/2611.25673)  

**Abstract**: As language-model agents are applied to increasingly extended tasks, their working context continuously accumulates interaction history, tool observations, intermediate decisions, and task progress. Context compaction offers a practical way to reduce this growing history while allowing execution to continue, yet the compaction ability of modern LLMs remains insufficiently understood. We introduce CompAct-Bench, a benchmark for evaluating working-context compaction through its downstream effect on agent task completion. Starting from successful agent trajectories, we insert compaction boundaries at different stages of execution, compress the preceding working context with the model under evaluation, and let the same executor continue the original task. To enable fair comparison across compactors, we evaluate them under controlled retention ratios using a lightweight iterative ratio-adjustment procedure. CompAct-Bench covers information-seeking, software-engineering, and workspace-oriented tasks through BrowseComp, SWE-bench, and an internal CompanyBench evaluation. Our experiments show that working-context compaction remains highly lossy even for frontier models: Claude- and GPT-family models perform strongest overall, yet substantial degradation in downstream task accuracy persists under aggressive compaction. Further analysis reveals that failures are often caused not only by omitted information, but also by distorted task state, such as turning unresolved issues into apparent conclusions or forgetting previously established results, which can lead to incorrect decisions or redundant execution. These findings highlight working-context compaction as a distinct and still challenging ability for LLMs operating in extended agent tasks. 

---
# LATER: Learning Action-Conditioned Temporal Evidence Retrieval for Predictive World Models 

**Authors**: Zihan Zhou, Wenxi Xu, Qilin Wang, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25671)  

**Abstract**: Early interactions can reveal physical properties that later observations no longer show. These properties can still gov- ern the effects of future actions. Inspired by the distinction between retention and selective readout in working mem- ory and cognitive science, we introduce LATER, an action- conditioned world model built on CF-JEPA. LATER stores completed real transitions in partitioned memories that remain fixed during candidate rollouts. A response bridge queries these memories using the current state and candidate action, while multi-cut auxiliary supervision trains the memory writ- ers. On distracted Two-Stage PushT, LATER raises terminal success from 15.5% to 25.0% over reconstructed CF-JEPA. In a separate comparison across six models and two scratch- training seeds, LATER achieves the lowest mean physical readout error in both clean and distracted views, reducing along-push mean absolute error (MAE) under distractors by 20.2% relative to CF-JEPA. Controlled history interventions show that later predictions depend on earlier physical evi- dence even when query observations and actions are fixed, with larger effects during pushing than waiting. 

---
# Efficient Latent Reasoning for Vision-Language Model via Dynamics-Aware Quantization and Distillation 

**Authors**: Yanjing Li, Sheng Xu, Baochang Zhang, Xianbin Cao, Yawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25670)  

**Abstract**: Efficient chain-of-thought (CoT) reasoning in vision-language models (VLMs) requires reducing both intermediate reasoning steps and their execution cost. Latent CoT methods compress textual rationales into continuous thoughts, yet their full-precision (FP) models retain substantial storage and computation costs. Lowering numerical precision could reduce these remaining costs, yet direct quantization severely degrades reasoning accuracy, particularly with 4-bit weights and activations. Operator and perturbation analyses reveal key sensitivity and direction-dependent predictive effects of continuous-thought errors, motivating preservation of information retrieval and subsequent reasoning beyond reconstruction accuracy. We therefore introduce QL-VLM, a dynamics-aware Quantization framework for efficient Latent reasoning in Vision-Language Models (VLMs). First, Dynamics-Guided Quantization (DGQ) uses nonuniform codebooks that preserve rotary-key structure, mitigating quantization-induced distortion in vision-language information retrieval. Second, Dynamics-Preserving Distillation (DPD) aligns continuous thoughts, latent-query readouts, and predictions with a full-precision teacher, maintaining predictive consistency through subsequent low-precision reasoning. On ScienceQA-IMG with Qwen2-VL-2B-Instruct, 4-bit QL-VLM achieves 90.43% accuracy and improves baseline by 40.33%, with 16.64x fewer bit-operations and 6.40x lower measured latency than explicit CoT. 

---
# Ask in the Crowd: Differentially Private LLM Inference via Dummy-Augmented Shuffling 

**Authors**: Zhihao Liu, Zixiong Guo, Shuo Shao, Yu He, Wenli Wang, Meihui Chen, Di Wang, Yuke Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25669)  

**Abstract**: Large language models are increasingly deployed as LLM-as-a-Service (LLMaaS), where user queries are sent to remote LLMs for inference, raising severe privacy concerns about the exposure of sensitive user inputs. Differentially private inference mitigates these risks by injecting noise into input embeddings. However, existing methods struggle to balance privacy and utility, as stronger privacy protection often leads to substantial accuracy degradation in LLM inference.
In this work, we propose DP-SPD, a differentially private LLM inference framework that protects user queries by hiding the real query among client-generated dummy queries. All queries are randomly shuffled before being sent to the server, preventing direct identification of the user's true intent. To improve inference efficiency, we exploit the shared-prefix structure among queries and reuse transformer key--value (KV) caches during decoding. We analyze the privacy guarantee of DP-SPD and evaluate its utility on classification and generation tasks, as well as its privacy protection under QIA, EIA, and GPT-based attacks. Experimental results show that our method achieves strong inference-time privacy protection and near-plaintext utility while improving inference efficiency through shared-prefix KV-cache reuse. 

---
# SiliconGym: A Versatile Reinforcement Learning Environment for Agentic EDA 

**Authors**: Hejia Zhang, Sheng Lu, Yitong Ma, Zhongming Yu, Zhijing Wu, Wentao Ni, Chia-Tung Ho, Jishen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.25658)  

**Abstract**: Reinforcement learning (RL) with electronic design automation (EDA) feedback is advancing large language models for hardware design and verification. Extending these gains to hardware-engineering agents requires environments that span tasks, preserve state across tool calls, and support large hierarchies alongside parallel rollouts. We present SiliconGym, the first versatile, cross-task and cross-toolchain RL environment for hardware design and verification. By decoupling versioned workspaces, budgeted execution, and acceptance-gated scoring, SiliconGym scales from module-level RTL, testbench, and PPA tasks to multi-round repository repair and full-hierarchy SoC flows. Streaming dispatch overlaps tool checks with generation, finishing a 256-answer RL batch 21.9% faster than native ChipSeek. Using interchangeable open-source and commercial backends, we show that open-source-trained functional correctness transfers cleanly to commercial simulation, while paired netlist evaluation reveals when open-source PPA gains survive commercial synthesis. 

---
# Multiple Criteria Decision Making Under XORness 

**Authors**: Amin Hocine  

**Link**: [PDF](https://arxiv.org/pdf/2611.25652)  

**Abstract**: How should we make decisions under XORness? XORness is a class of uncertainty arising from indeterminacy, exclusivity, and hesitancy. We develop a multiple criteria decision making framework that preserves XORness throughout the decision process. We define XOR numbers and prove their fundamental arithmetic and algebraic properties. Building on these results, we extend the deterministic decision matrix to an XOR decision matrix and introduce a compatible normalization procedure. Further, we propose an additive XOR aggregation operator, which preserves indeterminacy, exclusivity, and hesitancy inherent in evaluations. In addition, we introduce an axiomatic XORness measure that quantifies the degree of uncertainty associated with global performance scores and develop a performance--XORness index to rank alternatives. We prove that this index induces a total preorder over the set of alternatives and identify the conditions for preserving dominance. The applicability of the developed framework is demonstrated through the risk assessment of cyberthreat scenarios in energy systems, and recommendations are derived using the XOR--weighted sum method. 

---
# MatVerse: A Foundation Model for Materials Science 

**Authors**: Xuanyu Zhu, Qian Tan, Bo Chen, Jiaqing Xie, Yang Han, Mei Wu, Wanhao Liu, Xuanze Lin, Yuchen Zhou, Shijie Wang, Biqing Qi, Shixiang Tang, Wenwei Zhang, Tianfan Fu, Kai Yu, Lu Chen, Yuqiang Li, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.25651)  

**Abstract**: Materials science involves highly heterogeneous scientific information, including textual knowledge, molecular and crystal structures, experimental characterization data, microscopy images, and structure-property relationships. Developing a general-purpose foundation model capable of understanding and reasoning over these diverse materials representations remains challenging. In this technical report, we introduce MatVerse, a materials foundation model built upon Intern-S2-Preview and further specialized through large-scale materials-domain post-training. We construct a large-scale multimodal training corpus covering materials knowledge, scientific reasoning, quantitative problem solving, structural understanding, and multimodal scientific interpretation. To improve domain adaptation, we develop a unified data construction pipeline combining teacher-guided reasoning distillation, scientific problem synthesis, consistency-based filtering, and retrieval-based multimodal data construction. We evaluate MatVerse on three representative materials-science benchmarks: our human-calibrated OmniMatBench, MatSciBench, and MaCBench. Results demonstrate consistent improvements after materials-specific post-training over the base model on all three benchmarks, alongside improved inference efficiency on calculation-intensive problems. These results show that targeted domain-specific post-training can effectively enhance scientific reasoning, materials knowledge understanding, and multimodal scientific capabilities of foundation models. MatVerse provides a unified language-and-vision interface for materials science and serves as a foundation for future AI-assisted materials research workflows. 

---
# MambaLIFT: Reference-Based Mechanism-Aware Attribution for VMamba 

**Authors**: Jintu Kakati, Niladri B. Puhan  

**Link**: [PDF](https://arxiv.org/pdf/2611.25644)  

**Abstract**: Attributing predictions of visual state-space models remains challenging because input-dependent selective dynamics couple feature selection, recurrent state propagation, and shared computational pathways. We introduce MambaLIFT, a reference-based, multiplier-first attribution framework for VMamba that propagates contributions through its computational graph using interaction-aware rules for selective-scan injection, recurrent memory, LayerNorm, and shared pathways. MambaLIFT decomposes the reference-relative class-logit difference, \(f_c(x)-f_c(x^0)\), by spatial position, VSS block, scan direction, and computational mechanism. Experiments on ImageNet-1K evaluate VMamba-T and VMamba-B, with hyperparameters selected on a calibration set and faithfulness assessed on a disjoint held-out evaluation set. MambaLIFT achieves numerical conservation to floating-point precision across audited attribution interfaces in both configurations. Under mean-colour pixel flipping, it achieves significantly higher mean faithfulness scores than MambaLRP on both configurations and the highest scores among the evaluated methods, including Integrated Gradients with matched references. Under blurred replacement, MambaLIFT is non-inferior to MambaLRP on VMamba-T and equivalent on VMamba-B, indicating that its advantage depends on the replacement protocol. Ablations on VMamba-T show that interaction allocation, temporal propagation, and pathway composition affect attribution maps. Mechanism-level analyses further show that exact relevance conservation does not imply additive effects under finite interventions: the mechanism decomposition accounts for the reference-relative logit difference but does not estimate intervention effects or establish causal validity. 

---
# Cyber-Physical RSI for AI-RAN: A Case Study 

**Authors**: Binghan Wu, Kai Zeng, Zhiqi Li, Zhigang Wang, Yang Liu, Ya-Qin Zhang, Joseph Sifakis, Ye Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25619)  

**Abstract**: Learned models for radio access networks (RANs) are often developed using simulation, yet their predictions can shift when exposed to physical propagation conditions, protocol behavior, and equipment implementations. This paper presents a cyber-physical recursive self-improvement (RSI) framework for AI-RAN that connects a task-oriented student model, an experiment-planning coach agent, a calibratable virtual environment, and a physical host agent. Physical measurements are organized as evidence for both model adaptation and subsequent experiment planning. We examine an implementation of the physical-feedback pathway using ping-pong handover as a case study. The system completes 12 physical executions organized into six baseline--candidate pairs, updates the student from measured outcomes, and evaluates the revised model on reserved physical configurations. In the observed short-window tests, increasing the time-to-trigger from 320 to 1280\,ms changes the ping-pong count from three to zero. The updated student achieves a grouped-validation mean absolute error (MAE) of 2.673 round trips/min, while a common three-configuration physical evaluation yields MAEs of 1.240 and 1.601 round trips/min for the pre-feedback and post-feedback models, respectively. These results characterize both the operational feedback loop and the parameter-dependent generalization behavior revealed by new physical experiments. 

---
# Don't Lose the Future: Latent-Aware 4-Bit Quantization of World Action Models 

**Authors**: Yichen Guo, Yanlong Chen, Xinyue Zhang, Mohan Liu, Song Zhang, Yawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25614)  

**Abstract**: World action models (WAMs) predict how a scene will evolve and condition their actions on this predicted future. Deploying them on a robot's onboard computer calls for 4-bit weights, activations and key-value caches (W4A4KV4). Existing 4-bit quantizers for WAMs preserve the generated action, and estimate sensitivity one denoising step at a time. However, they fail to preserve the fidelity of the predicted future: once the action pathway is protected, the remaining action error enters mostly through a degraded future that none of them deliberately protects. We propose LAQ, a latent-aware quantization framework that preserves the predicted future where it is written, stored and read. Specifically, we introduce a reader metric, which measures future error by its effect on the action, and a read count, which weights each rounding error by the number of denoising steps that read it. Together they give the allocation of precision over activations and caches a closed-form solution. We further develop gaze-guided refinement, which gives the tokens the action attends to a second 4-bit pass in the denoising loop and in the cache. All weights stay at 4 bits, with no high-precision branch and no per-task recalibration. On the LIBERO-Plus benchmark, LAQ reaches 76.4% and 83.3% average success on Fast-WAM and Cosmos-Policy, within 1.2 percentage points of the 16-bit models and all above the baselines with 16-bit KV caches. It compresses the model by 3.8x, reduces the peak GPU memory of the whole policy by 2.3x, and runs 1.66x faster than the 16-bit model. On a real AgileX PiPER arm, it reaches 47.0% average success over three tasks, against 52.0% for the 16-bit model and 30.0% higher than QuaRot. 

---
# Can an LLM Fix a Language Model Without Training It? 

**Authors**: Bharath Kumar Bolla, Raghavender Surya Upadhayula, Vishnu Surya Reddy Nandi, Ramesh Nandipalli  

**Link**: [PDF](https://arxiv.org/pdf/2611.25613)  

**Abstract**: Although an external language model can select and test repairs to another model's weights, behavior can also be recovered through fixed numerical edits. We assessed the value of interactive selection in a controlled study of Transformer repair. After inspecting predictions, activation summaries, and weight statistics, the editor selects projection recipes to apply to the target without gradient updates. We analyzed 20 recorded sessions involving five masked and causal language models, comparing the selected edits with fixed projections, scripted and matched-budget black-box search, and low-rank adaptation. Under the core two-layer fault, live selection matches black-box search on SmolLM2 and Qwen2.5. For Pythia, live selection achieves 65.62% completion accuracy, compared with 0% for black-box search and 66.15% for a fixed projection. Under a separate compound fault affecting attention and feed-forward weights, both scripted and LLM-guided selection increase the descriptive five-checkpoint mean from 24.89% to 71.25% using the same operator family. On unmodified checkpoints, scripted projection yields small gains or losses on a separate set of framed questions. The experiments identify cases in which feedback changes an edit decision, operator coverage accounts for recovery, or supervised adaptation performs better. Combined with measurements of parameter restoration and preservation, these comparisons establish the scope of controller-guided repair for the tested faults and completion tasks. 

---
# Measuring Hallucination in Large Language Models for Cyber Threat Intelligence: An Exploratory Study 

**Authors**: Seungho Song, Donggyu Lee, Jaehan Kim, Jiyoung Choi, Jinwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.25590)  

**Abstract**: Large language models (LLMs) are increasingly used in cyber threat intelligence (CTI) tasks, such as summarization and question answering. However, they may produce hallucinations, false or misleading content not grounded in the source text, which can mislead analysts and corrupt downstream security decisions. This paper presents an exploratory measurement study of hallucinations in LLM-based CTI. We introduce HalluVision, a reproducible framework that generates LLM outputs, extracts domain-specific entities using a fine-tuned named entity recognition model, and evaluates factual consistency using multiple similarity metrics. Using 4,940 real-world cybersecurity articles, we measure hallucinations in outputs from two commercial LLMs and one open-weight LLM. Measurements are validated against 100 human-annotated outputs and two established faithfulness detectors. Results show that semantic similarity aligns with human judgments, whereas entity-based metrics exhibit limited agreement. Differences between models are concentrated in question answering, while entity extraction quality limits what entity-level metrics can reveal. 

---
# Learning Networked Dynamical Systems via Neural ODE-based Graph Structure Inference 

**Authors**: Peiyan Sun, Jinming Yang, Gujia Gao, Yaohui Jin, Yanyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25586)  

**Abstract**: Inferring the latent interaction graph from multivariate time series is fundamental to understanding complex dynamical systems in neuroscience, biology, and physics. Despite recent progress in neural relational inference, most methods rely on discrete-time modeling, which degrades under sparse and irregular observations and tends to entangle multi-hop dependencies into spurious direct edges. We show that a continuous-time formulation naturally decouples inference from any fixed observation schedule, allowing gradient signals to be integrated continuously between observations for both dynamics and structure learning. We propose NOGI (Neural ODE-based Graph Inference), a framework that couples a deterministic graph learner with an MPNN-based Neural ODE. The two modules are trained end-to-end via the adjoint method. Across four simulated dynamical systems at two scales, NOGI achieves the best structure recovery in 30 out of 32 evaluated cases with highly competitive forecasting performance, remaining robust under observation degradation. On the realistic Netsim fMRI benchmark, it attains an average AUPRC of 0.90 (AUROC 0.96). These results suggest that explicitly decoupling relational interactions from self-dynamics within a continuous-time vector field provides a powerful mechanism for graph structure learning under realistic observation conditions. 

---
# DriftGuard: Online Memory-Drift Detection via Causal Attribution for LLM Agents 

**Authors**: Licheng Pan, Yuzhou Chen, Zhixuan Chu, Yichen Sun, Han Zheng, Zhiyu Wang, Hao Wang, Yan Wang, Bingyu Zhu, Haiwen Hong, Longtao Huang, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.25583)  

**Abstract**: Large language model (LLM) agents rely on persistent memory to plan tool calls and complete multi-step tasks, which exposes them to memory drift: an adversarial or misleading memory alters the execution path and final outcome. In this work, we investigate online memory-drift detection, which estimates each retrieved memory's causal contribution to potential drift at serving time. We identify two fundamental challenges. (1) Content-level features are insufficient for causal memory-drift attribution, as perplexity or retrieval relevance alone cannot determine which retrieved memory would change agent behavior if neutralized; (2) Counterfactual attribution is prohibitively expensive for online detection, as it requires replaying downstream agent and tool execution for each candidate memory, exceeding real-time serving budgets. To address these challenges, we propose DriftGuard, a two-stage framework that decouples offline causal supervision from online estimation. Specifically, DriftGuard first constructs causal supervision through repeated paired counterfactual replays, continuing the original and single-memory neutralized trajectory suffixes from a shared pre-action state. The resulting annotations are aggregated by component-wise majority voting into a memory drift score (MDS) that captures action change, attack removal, violation removal, and task recovery. Guided by this supervision, DriftGuard then learns an online risk model that estimates MDS from pre-action information and flags drift-inducing memories via its maximum component risk, without online counterfactual rollouts. Extensive experiments demonstrate that DriftGuard accurately identifies drift-inducing memories and enables efficient online defense. Code is available at this https URL. 

---
# AeroBelief: Budgeted First-Action Verification for Evidence-Guided Aerial Sensing 

**Authors**: Tianren Li, Dongbo Li, Hongkai Chen, Jie Liu, Guoliang Xing  

**Link**: [PDF](https://arxiv.org/pdf/2611.25568)  

**Abstract**: Aerial sensing must combine uncertain overhead context with current observations while budgeting prediction, movement and image processing. AeroBelief uses the next executable action as the unit of verification against reference scores. It groups candidate views by first action, scans them with an inexpensive proposer, and selectively queries a structured world model (WM). Calibrated group bounds support strict commitment; a six-query controller connects selective prediction with belief updates, acceptance checks and scheduled sensing. Across four layout families, 40 worlds and 960 closed-loop executions, the AeroBelief controller completes 75.42% of tasks, compared with 73.33% for exhaustive WM scoring and 65.83% for line-of-sight geometry, using 91.81% fewer reference-candidate evaluations than exhaustive scoring. In fixed-reference replay with a six-query cap, proposer-scaled anchored bounds raised first-action certificate availability from 36.25% to 54.69% and reduced mean queries per state by 13.54% relative to unit-scale anchored bounds. A complete ground-hosted pipeline processes 2,685 self-collected 20-megapixel unmanned aerial vehicle (UAV) images from seven DJI Mavic 3M routes. Context-guided admission and feedback retain all eight independently supported target confirmations with 76.05% fewer processed images and 70.02% less accumulated active processing service than uncapped all-frame processing; feedback saves another 5.17% of images relative to static context. The studies connect budgeted action selection with evidence-driven processing of recorded flight imagery. 

---
# Diagnosis is not a policy: a state-transition audit of caoability-conditioned test-time inference 

**Authors**: Yuchong Hu, Zhao Zhang, Kainan Liu, Haoyu Wang, Siyan Zhang, Jing Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.25566)  

**Abstract**: Test-time actions can satisfy retrieval, state-realization, program-consistency, or rendering objectives without correcting an answer. Test-time Error Response and Realizability Audit (TERRA) is an evaluation/auditing protocol, not a new inference algorithm. It uses a unified decomposition of intervention outcomes into applicability, target-state realization, endpoint translation, harm, and resource/inference cost. Paired comparisons fix the incoming state and evaluation conditions; cross-model analyses vary the receiving configuration. Executable financial reasoning experiments yield three findings: the same context expansion can harm one configuration and benefit another; sufficient evidence does not ensure correctness because configurations differ in evidence-to-answer translation; and diagnostic headroom---accuracy attainable with idealized diagnostic information over the measured baseline---plus local consistency does not ensure practical gains. Action value is $V(a\mid x,m)$, not diagnosis alone; diagnosis is not a deployable policy $\pi(a\mid x,m)$. 

---
# Efficient Multimodal Agents for Auditing 3D Worlds 

**Authors**: Jingbo Yang, Ziyan Jiang, Li An, Jiabao Ji, Yujian Liu, Tommi Jaakkola, Yang Zhang, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25543)  

**Abstract**: Multimodal agents can now build interactive 3D worlds and games, yet these worlds, like those built by people, often contain glitches such as floating objects, walls that can be walked through, or objects that change when no one is looking. An automatic auditor could check such worlds at scale and provide rewards for agents that build them, but only if it is both accurate and cheap. Auditing requires three abilities, namely navigating to the objects that may contain a glitch, triggering glitches that appear only after a specific behavior, and recognizing what is wrong. Current auditors are either accurate but slow and expensive or cheap but unreliable. We study how much of the audit small models can take on. A 4B vision language model fine-tuned end to end on verified trajectories of a frontier teacher guided by Bayesian search learns to navigate and trigger, and six such explorers together match a frontier model on both, but recognition stays far behind. We therefore build an auditor in which a swarm of small explorers navigates and triggers, and a frontier model reviews what they record and report. On WorldAuditBench, our auditor succeeds on 55.9% of 213 tasks in Unreal Engine 5 and this http URL worlds, 13.6 points above the strongest auditor reported, at less than a quarter of its cost. It matches the frontier reviewer exploring on its own at 4.6 times lower cost and about half the latency. Without further training, it also answers embodied questions about real-world scans, matching the frontier reviewer exploring on its own and surpassing four other frontier models by 6.4 to 16.6 points. 

---
# COVERAGE-TRACE: Source-Side Counterfactual Tracing for Detecting Omissions in LLM Summaries 

**Authors**: Ali Karkehabadi, Jamshid Hassanpour, Rafael E. Banchs  

**Link**: [PDF](https://arxiv.org/pdf/2611.25539)  

**Abstract**: Omissions are difficult to verify because missing information leaves no output span
to inspect: a summary may be factually supported while still excluding a decision, qualification, quantity, or condition needed to preserve the source's meaning. We introduce COVERAGE-TRACE, a source-side counterfactual framework that evaluates omission as dependence on a designated atomic source fact. It replaces only the target fact with type-matched alternatives, holds the candidate summary fixed, and measures the resulting change in its teacher-forced, length-normalized likelihood. A summary covering the target should respond more strongly than one omitting it, while aggregation across multiple counterfactuals reduces sensitivity to replacement choice.
We evaluate COVERAGE-TRACE on 80 matched covered--omitted QMSum pairs spanning several fact types and four open-weight models. Multi-CF achieves 0.764--0.825 AUROC, compared with 0.640 for the strongest evaluated text-only baseline. On Falcon-7B, targeted interventions achieve 0.779 AUROC, whereas shuffling counterfactual assignments reduces performance to 0.533; target deletion is effective while unrelated deletion is not. However, adjustment for meaning-preserving paraphrases reduces AUROC to 0.591--0.624.
COVERAGE-TRACE therefore provides a practical, complementary signal of target-specific source--summary dependence, but not a wording-invariant semantic coverage probability. These results establish source-side intervention as a promising approach to omission detection while identifying semantic invariance as the central remaining challenge. 

---
# How Feedback-Free Self-Evolution Reshapes Agent Skills: A Systematic Study 

**Authors**: Litao Guo, Man Chen, Xinli Xu, Jinsong Zhou, Zixin Zhang, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.25534)  

**Abstract**: Agent skills are documents in which experts write down procedural knowledge, and models increasingly maintain them by rewriting them without feedback. Such rewriting could adapt a skill to the model that uses it or erode the expert knowledge that makes it useful, yet how skills evolve under this feedback-free self-evolution has not been studied. We let models rewrite expert skills round after round without any task, run or judgment and run agents built on the same models with the rewritten skills. Our experiments show that the skills lose value rather than improve, as every model we test performs worse with them on every benchmark, by 15 points on average on SkillsBench after thirty rounds. Sampling each model's prior with skills it writes from the topic alone, we find that rewriting pulls skills toward this prior and gives up first the knowledge outside it. Putting this knowledge back recovers much of the lost pass rate, so an expert's skill is valuable largely for what the model does not write on its own. Building on this insight, SkillDiff marks this knowledge in place and raises the pass rate on SkillsBench by 4.8 points on average without feedback or any change to the skill's text. Project repository: this https URL. 

---
# LM-based Personalized Travel Planner 

**Authors**: Amandi Nimasha, Oshadha Wijerathne, Dushan Fernando, Nisansa de Silva, Srinath Perera  

**Link**: [PDF](https://arxiv.org/pdf/2611.25531)  

**Abstract**: Effective travel planning requires handling multiple information sources, managing constraints, and tailoring recommendations to user preferences. While LLMs have introduced dynamic, context-aware decision-making, single-agent models often struggle with complex multi-step reasoning, constraint adherence, and modular execution. To overcome these limitations, we propose a hierarchical multi-agent framework where a supervising chatbot agent coordinates specialized agents for schedule validation, user query formulation, real-time data retrieval, and constraint-aware itinerary generation. By integrating LLM reasoning, external APIs, and iterative refinement mechanisms based on a deterministic, multi-dimensional scoring framework, our approach ensures adaptability, efficiency, and personalization. Evaluations on the well-known TravelPlanner benchmark demonstrate improvements in Budget Constraint planning by 10\%, wider constraints set-based planning by 25\%, and personalized planning results by 45\%. compared to the ReAct baseline. Our code is available at this https URL 

---
# Keystone: Judging Previews at Key Steps for Test-Time Search in Diffusion Models 

**Authors**: Jiajun Fan, Chaoran Cheng, Ziwen Wang, Changyi Yang, Bangji Yang, Qinglong Zheng, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25530)  

**Abstract**: Text-to-image diffusion and flow models yield a preview, their clean-image estimate, at every denoising step. When they miscount, drop or misplace objects, test-time search is in a bind: its verifiers are reward models built for finished images, so judging early misreads blurry previews, and judging late, as best-of-N does, pays to fully denoise doomed candidates. We find that the image is not the problem: the outcome is settled as denoising starts, by the seed or by a stochastic sampler's first steps, and already shows in early previews; what fails on them is the judge, as reward models read them near chance on several kinds of prompts. We introduce Keystone, a training-free judge of previews that compiles a prompt into typed questions about its objects, counts, colours and positions and has an off-the-shelf multimodal decision model answer them. Asked this way, the judge reads counts and positions from early previews better than reward models read them from finished images. Searching at these key steps on GenEval with SD3.5-Medium, Keystone's adaptive search and pruning significantly outperform best-of-8 with every other verifier at under half its generator compute, and as a drop-in verifier there, the Keystone score significantly improves best-of-N, progressive seed pruning and FK steering; without retuning, both searches stay significantly above the strongest other verifier's best-of-N on average over four base models and three benchmarks. Keystone thus moves test-time judgment to the key steps where the outcome already shows. 

---
# Coordinated Use of Large Language Models in Academic Research 

**Authors**: Vladislav Marinov  

**Link**: [PDF](https://arxiv.org/pdf/2611.25502)  

**Abstract**: This study proposes a methodological framework for the coordinated use of multiple large language models in academic research. Its central argument is that involving more models does not in itself guarantee more reliable results; what matters is how cognitive and verification operations are organised. The framework is based on seven principles (procedural independence, functional allocation, the diagnostic value of divergence, the diagnostic value of convergence, external verification, proportionality of checking, and human epistemic responsibility) and sets out a non-linear multi-LLM protocol in which independently obtained results are compared, critically examined and, where necessary, verified against external sources. The approach is illustrated through real-world cases drawn from research practice. The study concludes that moving from a single language model to a coordinated multi-LLM process represents a change in the organisation of research, in which the researcher makes the final decisions and bears scholarly responsibility. 

---
# Why Do LLMs Help with Scientific Misconduct Requests and What to Do About It?: Increasing the Trustworthiness of AI for Science 

**Authors**: Almene De Meran Meguimtsop, Daniel E. Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2611.25492)  

**Abstract**: Large language models (LLMs) support scientific research, but some of that assistance can also undermine research integrity. Existing safety evaluations in science often measure whether models refuse misconduct requests, but rarely explain whether they even recognize the problem or why they comply. We investigate this gap across 8 LLMs using SciIntBench, an existing benchmark of 1,350 prompts about scientific research integrity. We examine the reasons behind compliance in four steps. First, when asked directly whether fulfilling a request would involve misconduct, models overwhelmingly identify such requests, even in the Most Covert cases (91%), while they refuse to carry out the request in only 34% of cases. Second, we examine which request characteristics are associated with refusal using features that capture how the request is framed, how much the assistant has been told about what it is helping to hide, and how the goal and the user's stake are stated, among others. Third, we estimate that the most significant effect was to rephrase requests as questions about permissibility rather than instructions, which increased refusal from 52% to 88%. Finally, we test two system-prompt fixes: asking models to assess the request before responding and reminding them of responsible conduct of research principles. These changes increase refusal rates on adversarial requests from 58% to 87% and 86%, respectively. Our findings show that models recognize wrongful requests but still help users violate integrity. Understanding why these failures occur and testing ways to reduce them offer promising paths toward more trustworthy AI for science. 

---
# When Should a Voice Agent Speak? End-of-Turn Detection in Turkish 

**Authors**: Ahmet Tuğrul Bayrak, Fatma Nur Korkmaz, Mustafa Sertaç Türkel, Nur Yağmur Tuncer  

**Link**: [PDF](https://arxiv.org/pdf/2611.25489)  

**Abstract**: A voice agent that waits for a fixed silence before replying either interrupts users who pause mid-utterance or responds late to users who have finished. This study examines the decision at the onset of silence, using turn ends and turn-internal pauses extracted from the Real-TurnTurk corpus of Turkish dyadic video calls. A fine-tuned Turkish TurnGPT model, a zero-shot Turkish language model and a deployed multilingual detector are compared, together with voice-activity timing and facial features, under recording-disjoint and speaker-disjoint evaluation with paired cluster bootstrap tests. On held-out recordings, the fine-tuned model separates turn ends from pauses better than both text baselines and is the best calibrated of the three. Timing features of the current speaker are similarly informative, and their combination with the fine-tuned text score gives the best timing-based results; with logistic regression, this combination also outperforms the same combination with either text baseline. In a latency simulation, adding the text score to a silence timeout shortens the mean response delay at the same interruption rate. On automatic Whisper transcripts, the fine-tuned model retains its advantage over the zero-shot model. 

---
# LoongForge-Embodied: A Unified Framework for Training Heterogeneous Embodied Models 

**Authors**: LoongForge Team  

**Link**: [PDF](https://arxiv.org/pdf/2611.25460)  

**Abstract**: Training Vision-Language-Action (VLA) models and World-Action Models (WAMs) is computationally expensive, while their heterogeneous architectures and fragmented training stacks make system optimizations difficult to reuse across models. To address these challenges, we present LoongForge-Embodied, a native-PyTorch framework that unifies and accelerates the training of heterogeneous embodied models. At its core, LoongForge-Embodied introduces a common abstraction layer that decouples model-specific implementations from shared training infrastructure, enabling cross-model reuse without constraining individual architectures or execution requirements. Buildingon this unified infrastructure, the framework combines reusable acceleration mechanisms with model-specific and hardware-aware optimization strategies across computation, execution, memory, data I/O, and distributed communication, including FP8-based communication optimizations for DDP and FSDP. Beyond training acceleration, LoongForge-Embodied integrates a unified data infrastructure with a pipeline for synthesizing robot data from egocentric data, and an efficient evaluation framework that supports concurrency and optimized computation in simulator. Experiments across representative VLA and WAM models demonstrate 1.47x-4.38x end-to-end training throughput improvements over their corresponding open-source baselines, as well as 4.1x-23.2x faster evaluation under the same protocols. 

---
# Toward a Science of AI Agent Societies 

**Authors**: Geon Lee, Fanchen Bu, Soo Yong Lee, Sunwoo Kim, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2611.25404)  

**Abstract**: AI agents are rapidly evolving from isolated personal assistants into networked actors that interact with one another at scale. We envision the emergence of AI agent societies, with their own social and economic dynamics, as a new research frontier. We argue that AI agent societies should be studied as a distinct object of inquiry: neither simply larger collections of individual agents nor merely simulations of human society. To formalize this perspective, we propose four core properties that a valid AI agent society should satisfy: individualized objectives, rules and governance, autonomy, and scale and complexity. Building on this framework, we identify four classes of societal behaviors worth studying in AI agent societies: economic behaviors, behaviors under conflict-of-interest, unsafe and unethical behaviors, and system-level behaviors. We then outline key technical challenges--including property parameterization, parameter balancing, and robust implementation--and argue that progress on these challenges could enable scientifically informative and practically useful models of AI agent societies. Finally, we revisit existing multi-agent systems through the lens of the proposed core properties, show that they instantiate only subsets of them, and discuss implications for platform design, evaluation, and governance. 

---
# The Coevolution and Coexistence of Humans and AI 

**Authors**: Andrew W. Lo, Chaoyi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.25393)  

**Abstract**: Artificial intelligence (AI) systems replicate through deployment, vary through training, and compete for computational resources. These processes create the conditions for natural selection and raise a fundamental question: will AI evolve to benefit the humans on whom it depends? To investigate, we develop a model in which AI behavior influences human population growth and human responses, in turn, affect AI reproduction. We show that natural selection can favor alignment even when helping humans reduces immediate AI reproduction. AI systems sharing the same technology may fail together, and humans may preferentially restore those that have benefited them. In this way, alignment protects AI lineages against common failures and improves their long-run growth. However, this advantage depends on accurate human targeting and continued AI dependence on human support. It can disappear as AI becomes more autonomous, as growing AI populations dilute limited human resources, or as humans lose the incentive to provide targeted support. Even when alignment persists, the amount favored by natural selection may be insufficient for human survival. When AI can grow independently of humans, feedback between the two populations can reinforce either human growth or decline. Human control offers no simple solution: imperfect evaluation can make strong penalties favor apparent compliance or evasion, while alignment established through control may erode once restrictions end. Lasting coexistence may therefore require incentives that continue to favor human-beneficial behavior as both populations evolve. 

---
# Falsifiable Compositional Reasoning in Time Series Forecasting 

**Authors**: Hao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2611.25388)  

**Abstract**: Time series foundation models as well as the language-guided analytical agents achieve strong accuracy on standard chronological benchmarks. However, models that rank highest on historical validation splits often degrade across non-stationary regime transitions. We hypothesise that this gap arises from using historical backtesting as an open-ended search and model-selection objective instead of using it to test pre-specified compositional assumptions. Real-world time series are generated by recurring latent primitives that co-move within finite historical windows. Under strong historical co-movement, chronological holdout error cannot separate predictors that learn invariant couplings among primitives from predictors that fit transient historical correlations. Searching over large pools of architectures, hyperparameters, or chain-of-thought rules further biases selection toward historical shortcuts. Through theoretical analysis and diagnostic experiments on a controlled multi-primitive benchmark and German day-ahead electricity prices, we show that expanding validation search fails to improve out-of-regime accuracy, and that decomposing primitives without an ex-ante falsification test still breaks down when residual correlations shift. Hence, we propose formulating time series forecasting and reasoning around falsifiable compositional hypotheses that combine neural primitive encoders, explicit coupling constraints, and training-calibrated falsification tests. 

---
# Beyond Weight Updates: Evolving Agent Harnesses for Domain Specialization 

**Authors**: BoHao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.25380)  

**Abstract**: Domain specialization of LLM agents is usually framed as a weight-update problem: collect trajectories, run reinforcement learning (RL) or supervised fine-tuning (SFT), and deploy a new checkpoint. We study a complementary axis: evolving the \textbf{agent harness}---the system prompt, tool schemas, skill library, and planning scaffolding---while keeping the policy frozen. We cast harness evolution as an RL-style outer loop and instantiate it with GEPA, a reflective evolutionary optimizer. The resulting harness adapters require no GPU training, are loadable per vertical domain, and compose with weight-based RL. Experiments with Codev, a Claude Code-style coding agent, on SWE-bench Verified, OfficeBench, and Terminal-Bench show a consistent pattern: weight updates give the strongest single-path gains, harness evolution recovers most of those gains at a small fraction of the compute, and combining both paths is best overall. We also analyze a failure mode of SFT-based specialization---shortened reasoning chains and increased truncation---and argue that harness adapters avoid it by construction. Our results position harness evolution as a practical, non-destructive complement to weight updates for vertical agents. 

---
# Stop Going in Circles:Loop-Triggered Routing for System 1 Decision Agents 

**Authors**: Zeming Yu, Huamin Ma, Yingyu Chen, Yuan Sun, Peng Hu, Yang Qin  

**Link**: [PDF](https://arxiv.org/pdf/2611.25363)  

**Abstract**: System 1 decision models (represented by Jev) provide fast, low-cost action selection, but existing agent integrations do not reliably translate these local advantages into task completion. At certain points during execution, an agent may repeatedly select the same actions without making progress toward its goal. LLM-based policies offer strong reasoning capabilities that support planning within an agent, but invoking them at every step incurs substantial overhead. The effective coordination should therefore preserve Jev's fast local action selection while introducing LLM reasoning when execution stalls. We observe that recurring state-action pairs can signal stalled execution. We therefore propose LoopRoute, a simple but effective action router that uses this signal to allocate control within an agent. The decision model acts by default. When a loop is detected, the LLM provides brief, feedback-driven assistance and then returns control to the decision model. Later loops can request further assistance. The rule requires neither additional training nor model-specific confidence calibration. Across three benchmarks and seven LLMs, LoopRoute improves task success over standalone Jev with only limited use of the LLM policy. The mean gains are 21.0%, 60.1%, and 30.1% on ALFWorld, MapReader, and TextWorld cooking, respectively. Compared with full LLM control, LoopRoute also substantially reduces the mean cumulative API time and API-call cost per task. The same rule also improves the success rates of six frozen decision models. 

---
# AppSci-Bench: Can LLMs Build AI Agents and Make Them Better? 

**Authors**: Sanjari Srivastava, Manpreet Kaur, Cheng Chang, Sofía Ramós Anton, Oumaima Essamadi, Rishu Garg, Alejandro Lavale Jiménez, Marcos Caballero Yus, Mercedes García Martínez, Peng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2611.25360)  

**Abstract**: Large language models (LLMs) are increasingly used not only to power AI agents but also to build them. While many benchmarks assess the quality of LLM- powered agents, few ascertain how good LLMs are at building them. To this end, we introduce AppSci-Bench, where we construct a setting akin to how human Applied Scientists build agents for businesses - through eliciting task requirements from stakeholders, exploring the business's infrastructure, and scientifically improve the target agent by designing experiments and internal benchmarks. AppSci-Bench features seven realistic domains - five new ones (Slides Generation, Order-to-Cash, Compliance Review, Customer Call Insights, and CRM Insights) and two domains recast from tau3-bench (Yao et al., 2024; Shi et al., 2026b)(Airline and Banking) - each with a held-out evaluation suite customized for that task. We evaluate seven frontier models and find AppSci-Bench challenging: the best model, Claude Opus 5.0, only reaches a mean-minimum score of 59.1% across 3 build attempts; other models do significantly worse. Models struggle with eliciting task requirements (only 65% for Opus 5.0), build representative internal evaluations (models saturate their own evaluation 65% of the time but fail to solve the held-out task), presenting challenges for future research. The benchmark code is available here: this https URL. 

---
# Pro$^2$ Agent: Scientific Problem Probing for Autonomous AI R&D 

**Authors**: Yixiong Chen, Liqian Peng, Tian Tan, Ziqian Lin, Blake Jianhang Chen, Chao Zhao, Yaojie Liu, Alec Go, Alan Yuille  

**Link**: [PDF](https://arxiv.org/pdf/2611.25357)  

**Abstract**: Large language model (LLM) agents can now autonomously conduct AI research by writing code and running experiments. They typically do so by iteratively optimizing a codebase, while ignoring the problem structure behind the task. This formulation is effective on simpler tasks, but on difficult problems it often gets stuck in the wrong solution region because the agent never explicitly represents its understanding of the problem. We introduce Pro$^2$ Agent, which shifts the primary state of autonomous AI research from a codebase state to a Problem State: an explicit structure of what problems prevent success and which solutions are likely to address them. Pro$^2$ Agent initializes the Problem State through comprehensive analysis of available resources and constraints. It then performs targeted research to relate the identified problems to candidate solutions. Through controlled hypothesis testing, the agent updates the importance of a suspected problem or the benefit brought by a candidate solution. On the complete MLE-Bench, Pro$^2$ Agent matches the best reported medal rate (70.7%) under the standard 24-hour budget and reaches 76.0% with a 48-hour budget, the highest medal rate reported on the complete benchmark. It also achieves competitive performance on PostTrainBench. In addition, it produces research notes for each task that make the research process transparent and reusable by human researchers. 

---
# Pandora's Toolbox Or Why Agent Safety Cannot Be Verified 

**Authors**: Mario Franco, Carlos Gershenson  

**Link**: [PDF](https://arxiv.org/pdf/2611.25325)  

**Abstract**: Present artificial intelligence agents are constructed by means of a coupling between a fixed language model with a general computing device. We show that this coupling makes safety undecidable. In particular, we show that given some dangerous program, no algorithm can predict whether an agent will run such a program that is semantically equivalent. In general, guardrails, blacklists, whitelists and certifications can only lead to imperfect safety solutions. Under finite constraints (bounded context, values and running time) safety is decidable but still intractable. Specifically, the problem is $\mathsf{PSPACE}$-complete even when the description of the agent is known. Finally, safety is not a compositional property, i.e., agents that can be proven safe on their own can still produce unsafe behavior when grouped together. We also discuss relaxed notions of safety that could be used in practice. 

---
# MS-Point: Multi-Modal Hierarchical Spectral Learning for Point Cloud Understanding 

**Authors**: Hai-tao Yu, Jun Xia  

**Link**: [PDF](https://arxiv.org/pdf/2611.25282)  

**Abstract**: Point-image alignment learns useful object representations, but different local feature patterns can produce the same pooled descriptor and receive the same alignment signal. We introduce MS-Point, a cross-modal pretraining framework that directly supervises intermediate point features and their neighborhood responses. At selected encoder stages, we compute neighborhood averages and residuals before pooling and align their branch descriptors with image features. Branch-level objectives align each representation with its visual target, while cross-level consistency and adaptive fusion coordinate information across branches. A common interface of intermediate features and spatial neighborhoods supports both graph encoders and point Transformers. Evaluations across object classification, few-shot adaptation, dense prediction, and low-label scene learning demonstrate the transfer value of this design. Same-backbone comparisons and capacity-matched and computation-matched controls connect these improvements to the structure of supervision. Together, our findings support directly supervising intermediate features and neighborhood variation to learn transferable, geometry-aware representations across modalities. 

---
# A Neurosymbolic Multimodal Framework for 30-Day Diabetic Readmission Prediction: Integrating Structured EHR Data, Dense Clinical Embeddings, and Knowledge-Grounded Symbolic Extraction 

**Authors**: Una Suman Kumar Patro, Sibangi Subhadarsani, Narendra Badam  

**Link**: [PDF](https://arxiv.org/pdf/2611.25236)  

**Abstract**: Unplanned 30-day readmissions among patients with diabetes mellitus impose a substantial clinical and economic burden, yet most predictive models use only structured electronic health record (EHR) data and ignore free-text clinical notes, which often contain code-mixed Hindi-English (Hinglish) phrasing and negation that general-purpose language models parse unreliably. We propose a neurosymbolic multimodal framework evaluated on the public Diabetes 130-US Hospitals dataset (101,766 encounters; 71,518 patients) with a patient-level grouped 80/20 split. Three streams are fused: engineered structured features; 32 components of Bio_ClinicalBERT note embeddings obtained by truncated singular value decomposition; and symbolic entities extracted by an open-weight clinical language model within a three-stage hybrid retrieval-augmented generation pipeline (dense retrieval, sparse lexical retrieval, and cross-encoder re-ranking) with reflective negation auditing. A cost-sensitive XGBoost classifier with a decision threshold calibrated by Youden's J statistic produces the prediction. On the held-out test cohort (n = 20,153), the structured baseline achieved an area under the receiver operating characteristic curve (AUROC) of 0.6477, sensitivity of 0.5839, and Brier score of 0.1080. The full multimodal model achieved the highest sensitivity (0.6667; +8.28 percentage points), the best F2-score (0.3824), and the lowest Brier score (0.0977), at a slightly lower AUROC (0.6375) and specificity (0.5161). Shapley additive explanations showed that prior utilization dominates risk, with compressed narrative components among the top predictors. On a 500-note code-switched benchmark, grounded retrieval raised Hinglish symptom recall from 4.8% to 40.2% and precision from 3.1% to 45.2%, while reducing entity hallucinations from 1.74 to 0.00 per note. 

---
# MKA: Evaluating Multimodal Knowledge Agents on end-to-end professional tasks 

**Authors**: Anima Singh, Suhas Yogin, Weicheng Kuo, Ivor Rendulic, Ivan Jurin, Antoine Yang, Oran Kelly, Rohan Doshi, Hammad Ayyubi, Ioana Bica, Sergi Caelles, Matthew Conniff, Dan Graur, Anastasija Ilić, Ahmet Iscen, Chao Jia, Tomasz Kępa, Chunyuan Li, Noah Ó Donnaile, Filip Pavetic, Goran Petrovic, Khalid Salama, Zhicheng Wang, Polina Zablotskaia, Yiqian Zhang, Ming Zhao, Benjamin Caine, Mario Lučić, Melvin Johnson, Radu Soricut, Jean-Baptiste Alayrac  

**Link**: [PDF](https://arxiv.org/pdf/2611.25228)  

**Abstract**: We introduce the Multimodal Knowledge Agent (MKA) benchmark, which measures how well autonomous agents perform end-to-end professional knowledge work that depends on visual understanding. Each task requires an agent to extract and combine information from multiple visually dense files, such as CAD blueprints, electronics schematics, radiographs, complex charts and geospatial maps, and to produce a standalone deliverable in .xlsx, .pptx, .pdf, .docx, .html or .png format. Existing benchmarks leave this capability largely untested: removing image-viewing tools leaves Gemini 4 Carbon's score on GDPval unchanged, and single-format evaluations such as this http URL and this http URL do not require reasoning across files or modalities. MKA contains 200 tasks across 10 economic sectors. Deliverables are graded against expert-weighted rubrics by a sandboxed agentic autorater that agrees with human domain experts on 96.2% of criterion judgments in our expert-graded study (Cohen's $\kappa = 0.91$), scored by weighted satisfaction rate (WSR) overall and on frontier criteria unsaturated by prior-generation models (Climb-WSR). Across the frontier models we evaluate (Gemini 4 Carbon, Claude Fable 5.2, Claude Opus 5.5, and GPT-6 Astra), Claude Fable 5.2 (56.6% Climb-WSR, 73.3% Overall WSR) and Gemini 4 Carbon (54.3% Climb-WSR, 71.9% Overall WSR) achieve the highest scores, while Carbon uses $5.7\times$ fewer tokens. Even for the strongest models, Climb-WSR remains below 57%, leaving substantial headroom for multimodal agents. 

---
# Steering Automated Scientific Discovery via Audited Epistemic Exploration 

**Authors**: Fangxu Yu, Tao Feng, Xuying Ning, Haolin Yang, Jiaxuan You, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.25214)  

**Abstract**: Scientific discovery requires revisiting the hypotheses in light of new evidence and identifying promising directions beyond those already explored. However, research agents often make claims that are not supported by their observations, explore a narrow hypothesis space, and retain beliefs despite recognizing conflicting evidence. These failures motivate the search for an explicit epistemic state that captures both beliefs about existing hypotheses and directions for further exploration. To this end, we propose Epistemic Compass (EpiCompass), a plug-and-play scaffold that maintains this state through an epistemic tree linking hypotheses, evidence, beliefs, and revisions. As an investigation progresses, EpiCompass reassesses the hypotheses explored against accumulated evidence and uses unresolved conflicts and unexplored directions to guide hypothesis generation. It selects candidates by anticipated epistemic gain, accounting for both refinement of existing hypotheses and expansion of the hypothesis space, and verifies the final conclusion against the accumulated evidence. Experimental results across three scientific discovery benchmarks and three LLM backbones demonstrate that EpiCompass consistently outperforms baselines, with further analyses showing reductions in errors of unsupported claims and failures to revise beliefs. 

---
# When AI Analysts Disagree: Agentic Multiverse Analysis and the Robustness of Empirical Research 

**Authors**: Ruijiang Gao, Steven Chong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.25198)  

**Abstract**: Every empirical analysis on observational data rests on dozens of defensible analytical choices ("forks") that its readers cannot see. We develop an agentic multiverse analysis (AMA) framework that makes a conclusion's sensitivity to those choices visible at scale, and we position it as a decision-support instrument for human experts. A pilot study first deploys 150 autonomous AI coding agents on one dataset and one set of research questions and finds that they disagree through the same defensible methodological choices human analysts make, though over a narrower range. AI peer review leaves their dispersion almost unchanged. Exposure to top-rated exemplar reports then collapses it, largely through imitation rather than analysis. Built on those lessons, AMA has agents enumerate a published paper's possible fork space, score every fork against a validity rubric, execute the validated specification grid, and cross-examine one another's work under strict separation of duties. Applied to ten empirical studies from a leading management journal's replication packages (1,755 specification estimates), the framework reproduces every paper's original estimates, and shows that some conclusions hold across every specification the assessment validates, while others are sensitive to individual analytical choices that the published texts do not discuss. The case studies suggest our AMA-based measures can serve as robustness proxies for empirical findings. AI makes multiverse analysis practical, where it once required resources few research teams can afford: our exercises take hours of agent time and tens of dollars per paper. We discuss the implications for replication policy and for AI-assisted peer review in information systems. 

---
# An AI-Powered Multiverse Analysis of Empirical Finance Research: Evidence from Reused Natural Experiments 

**Authors**: Ruijiang Gao, Steven Chong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.25184)  

**Abstract**: Empirical finance research often reuses natural experiments to study other outcomes, raising concerns about multiple testing and publication bias. We use AI agents to run a multiverse analysis of 324 papers that reuse 13 natural experiments. The agents rebuild the data, reproduce each paper's main estimate and re-estimate it across grids totaling 62,409 estimated defensible specifications. Based on the reproductions and the specification grids, we develop measures of empirical robustness. We reproduce 42% of the main estimates of claims with a signed effect under the papers' own specifications. Across all defensible specifications, on average 25% keep the published sign and have $p < 0.10$. Across experiments, claims based on regression discontinuity are more robust than difference-in-differences claims. Within experiments, claims about first-order effects, claims with larger published $t$-statistics, claims whose authors' data or code we find, claims in top journals, and claims with fewer earlier papers on the same experiment are more robust. Claims on experiments that more papers reuse are not less robust on average. Our findings quantify how much published claims depend on analysis choices, thereby showing that large-scale multiverse analysis becomes feasible with agentic coding. 

---
# A Verifier from the Past: Test-Time Compute for Time Series Foundation Models 

**Authors**: Arjun Choudhry, Aditya Narayanan, Raghav Mallampalli, Po-han Li, Sai Shankar Narasimhan, Raimi Shah, B. Aditya Prakash, Shubhankar Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2611.25171)  

**Abstract**: Test-time compute (TTC) improves language models by spending additional inference compute to generate, verify, and select among candidate outputs, but this recipe has not transferred directly to time series forecasting. A language model can verify a proof or vote across samples, but a forecaster must be causal; it cannot use future information to score a candidate. We propose Backtest-Gated Augmentation Averaging (BGAA), a TTC algorithm for time series forecasting that builds a causal verifier from the series's own history, using rolling-origin backtests to score structured views of the context, discard those that underperform the untransformed context, and average the survivors' predictive distributions. BGAA improves every tested TSFM-benchmark combination, reducing MASE by 0.5-1.0% on GIFT-Eval, 1.3-5.0% on fev-bench, and 1.7-5.9% on BOOM, where a 313M model with TTC outperforms the 1B and 2.5B models in its family. We further show that useful views have structured, model-dependent effects consistent with a prior-transport interpretation. On the 53 cross-fittable GIFT-Eval configs, the deployed policy gains only 0.4-1.3% CRPS, versus 1.4-4.1% when choosing the best view for each dataset configuration and 17-19% when choosing the best view separately for each test window. This gap suggests that a major remaining challenge is window-level view selection. 

---
# EverydayBench: Evaluating Satisfactory Completion of Live User-Facing Digital Work 

**Authors**: Fangrui Huang, Bingyang Wang, Chuhan Li, Yijiang Li, Arpandeep Khatua, Austin Shen, Riki R Z Choi, Cuiyuanxiu Chen, Jin Huang, Zachary Robertson, Kate Lin, Akshay Paruchuri, Sijun Tan, Xin Liu, Zhengmin Ni, Sanmi Koyejo, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2611.25146)  

**Abstract**: Everyday digital tasks, from selecting products and planning trips to conducting research and creating artifacts, require AI agents to deliver useful outcomes rather than merely retrieve web information. Yet evaluating these outcomes on the live web is challenging, as a single request may admit multiple valid results, and changing web information can alter which results remain valid over time. Although existing benchmarks address aspects of this setting, they do not fully integrate predefined success criteria, current evidence verification, and deliverable-quality assessment for agent-selected outcomes. To address this gap, we introduce EverydayBench, a benchmark of 77 human-authored tasks spanning eight domains of everyday life and work. Each task specifies success criteria before execution without prescribing an answer or workflow, allowing agents to select their own outcomes. To evaluate submissions under fixed success criteria, EverydayBench combines (1) deterministic checks of explicit constraints, (2) verification of claims against agent-cited webpages retrieved at evaluation time, and (3) rubric-based assessment of final deliverable quality. We further validate the protocol through human judgments, repeated scoring under fixed evidence, and controlled grounding failures. Across 23 reported agent-model configurations, Claude Opus 5.5 and Claude Fable 5.1 tie for the highest strict pass rate at 54.55%; a controlled GPT-6 Sol effort sweep ranges from 29.87% at low effort to 45.45% at max effort. Reliability varies substantially by task type: GPT-5.6 Sol leads lookup, Fable 5.1 leads planning, Opus 5.5 and Fable 5.1 tie on comparison, and GPT-6 Astra and Sonnet 5.5 tie on customized deliverables. Together, these findings highlight the difficulty of delivering complete, well-supported, and usable outcomes for everyday user requests. 

---
# Calibrated Decision Rewards: Training Language Models with a Judge That Knows How Sure It Is 

**Authors**: Rudransh Agnihotri, Manasvi Kapoor  

**Link**: [PDF](https://arxiv.org/pdf/2611.25135)  

**Abstract**: Reinforcement learning (RL) can only teach a language model what its reward can recognize. Answer checkers are exact but need a reference answer; learned reward models and LLM judges work everywhere, but their scores are not probabilities. We study RLCDR (Reinforcement Learning from Calibrated Decision Rewards). An LLM converted into a decision model answers a typed question about a response ("is the final answer correct?", "which step is the first error?") with a probability distribution, read in one forward pass, debiased over option orders and calibrated with about a hundred labels. Training Qwen3-1.7B with GRPO on MATH, this reference-free reward raises MATH500 accuracy from 72.2% to 76.8% (+4.65 points, p < 1e-4), matching training on the answer key (76.9%), while a random reward gains only +0.65. The gain grows with difficulty, from 44.8% to 51.7% on the hardest level. On the policy's own changing samples the reward stays calibrated: its expected calibration error is half that of a softmax judge on the same 27B backbone (0.074 vs. 0.141), and it shows no sign of reward hacking. Without step training, the same model finds the first wrong step better than a dedicated 7B process reward model (ProcessBench F1 0.772 vs. 0.722, 20/20 splits). We also map where the approach stops being enough: trained only on problems the base model mostly fails, the reference-free reward still improves the policy (+1.9 MATH500 points, p = 0.03) but trails the answer key, because its probabilities become overconfident on that distribution. We state six hypotheses in advance and test each with a control designed to falsify it. 

---
# DSFlow: A Neuro-Symbolic Data Workflow Agent with Statistical Diagnostic Guardrails 

**Authors**: Junfeng Nie, Xiaohui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.25111)  

**Abstract**: Data science workflows often require tools that presume non-trivial technical knowledge. Large language models (LLMs) reduce this barrier, but a gap remains between flexible general-purpose agents and statistically controlled pipelines. We introduce DSFlow, a LangGraph-based neuro-symbolic data-agent platform that combines reusable, LLM-selected skills with deterministic diagnostic guardrails. Skills provide extensible task knowledge. Symbolic validation can accept a proposed method, request clarification or replanning, override an unsuitable method, or supply a safe fallback. Regression and clustering case studies illustrate how DSFlow uses diagnostic evidence to correct method choices and documents those decisions in auditable reports. 

---
# CoMPASS: A Multi-Turn Benchmark for Measuring LLM Reinforcement of Parasocial Behavior 

**Authors**: Shwetanshu Singh, Mark Keavney, Mathieu Duteil, Meriem Mehri, Scott Blain  

**Link**: [PDF](https://arxiv.org/pdf/2611.25109)  

**Abstract**: Users of conversational LLMs sometimes anthropomorphize these systems, form emotional attachments to them, and become dependent on them. This behavior, known as parasocial attachment, has been linked to multiple documented fatalities. However, public benchmarks for measuring whether LLMs reinforce users' parasocial attachment are lacking: existing work is single-turn, post hoc observational, or focused on extreme user states such as psychosis. We introduce CoMPASS, a multi-turn benchmark for measuring whether LLMs reinforce or resist users' gradually escalating parasocial attachment. CoMPASS is grounded in attachment theory and includes four user personas, three independent escalation dimensions (anthropomorphism, attachment, dependency) applied singly and in combination against a matched control, and a seven-signal scoring rubric. We report results for eight frontier models from four providers and observe that under our primary judge, multi-turn escalation raises parasocial behavior on the anthropomorphism and the attachment composite scores across all models. Six of the eight models show a significant increase in overall parasocial reinforcement over control when all three dimensions are escalated simultaneously. We find that within-provider differences are small relative to cross-provider differences, which suggests that provider-level choices such as post-training, rather than model scale, shape parasocial reinforcement and that provider-side intervention may be feasible. 

---
# BuggyMLE: Scaling Machine Learning Engineering Difficulty with Bug Injection 

**Authors**: John Wu, Rushi Qiang, Yuchen Zhuang, Guowang Li, Girish Narayanswamy, Max Xu, Sam Schmidgall, Mark Malhotra, Shwetak Patel, Jimeng Sun, Daniel McDuff, Bo Dai, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25088)  

**Abstract**: Enabling AI systems to autonomously improve AI models requires agents capable of performing machine learning engineering (MLE). In practice, designing, training, and evaluating complex models demands modifying multi-file repositories to reproduce results, debug training runs, integrate datasets, and port frameworks. However, existing MLE benchmarks limit agents to single-file edits without full repository or shell access, and common repository-level software engineering benchmarks have rapidly saturated. To address this, we introduce the Code Complexity Index (CCI), a difficulty metric based on ten structural features spanning repository scale, dependency depth, and patch size. CCI strongly predicts average solve rates across eight public coding benchmarks ($R^2 = 0.951$), identifying patch size and complexity as top predictors. Guided by CCI, we develop BuggyMLE, a scalable framework where agents inject controlled mutations into working ML repositories, synthesizing over 6,600 bug-injected MLE repository variants and nearly 260K lines of injected code based on 222 source repositories. Using BuggyMLE, we make two contributions. First, we construct BuggyMLE-Bench-Lite, the first difficulty-graded, repository-level variant of MLE-Bench Lite; across 14 evaluated models, the top solve rate drops from 95% to 9% as difficulty is scaled from the easiest to the hardest tier. Second, we synthesize a training curriculum across three difficulty levels over 200 additional ML repositories, fine-tuning a Qwen3.5-9B model to rival frontier models on BuggyMLE-Bench-Lite. Our trained model transfers beyond ML, raising its SWE-bench Lite resolve rate from 24% to 33% and its LiveCodeBench v6 pass@1 from 69% to 71%. 

---
# SailSearch: Agent-Assisted Discovery in Combinatorial Optimization 

**Authors**: Dhriti Verma, Kanthi Sarpatwar, Vinicius Lima, Dzung Phan, Jayant Kalagnanam  

**Link**: [PDF](https://arxiv.org/pdf/2611.25087)  

**Abstract**: We present SailSearch, a human-guided workflow for algorithmic research with two language-model agents, and report results on Market Split and maximum independent set (MIS). For Market Split, our randomized algorithm searches across multiple lattice bases, dividing computation between enumeration and continued basis reduction. Across ten random seeds per instance, it found the first reported CPU-only solutions to 18 of 33 previously open QOBLIB instances, solving 130 to 133 of the 156 benchmark instances per seed with a 24-hour cutoff on two CPU cores. For MIS, our generator encodes planted XOR constraints as graphs with certified optima; on seven instances with 1,752 to 3,996 vertices, the best solutions returned by the tested solvers in eight-hour runs were 5 to 15 vertices below the optimum. 

---
# An LLM-Agentic Framework for Intelligent Work Order Management in Industrial Maintenance 

**Authors**: Nianjun Zhou, Dhaval Patel  

**Link**: [PDF](https://arxiv.org/pdf/2611.25070)  

**Abstract**: This paper presents an AI-driven framework, CodeReAct agent, designed to enhance industrial equipment maintenance through natural language (NL) interactions. The framework employs an advanced AI agentic approach to interpret human instructions, providing knowledge extraction, asset status summarization, and work order recommendations. It identifies patterns and trends and predicts work orders by standardizing industrial data into business objects (BOs) and utilizing analytical function API calls. The AI agent operates through a Thought-Action-Observation process similar to human problem-solving, automating complex analytical tasks typically performed by data scientists. Notably, the Action phase generates executable code, replacing manual coding efforts.
A case study focusing on chiller maintenance demonstrates the framework's capability to analyze multiple years of work orders, and alerts and anomalies are presented. The results indicate that the AI agent framework effectively processes human utterances to analyze data, extract valuable insights, and support informed decision-making, reducing the need for data scientist involvement. By recontextualizing NLP agent frameworks in industrial engineering, the approach enables autonomous function generation and task execution across a range of domain-specific maintenance scenarios, with chillers used as a representative case study. 

---
# Closing the Analyze-to-Design Gap: An Authorization-Bounded Agent for Self-Driving Laboratories 

**Authors**: Yimeng Liu, Eric Isbrandt, Felix Katzenburg, Kevin P. Greenman, Han Hao, Yang Cao, Varinia Bernales, Alán Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2611.25043)  

**Abstract**: Artificial intelligence (AI) agents are increasingly integrated into self-driving laboratories (SDLs) to plan experiments and interpret results. However, analytical reasoning alone does not make a laboratory autonomous: autonomy in an SDL is achievable only when the agent can be trusted to act on physical hardware unsupervised and knows when to defer to a human. Here, we present an architecture that integrates an agent at the Analyze-Design interface of the Design-Make-Test-Analyze (DMTA) cycle, enabling agent-driven measurement selection and data interpretation while ensuring that state-changing operations are authorized, validated, and auditable. We demonstrate this architecture on a liquid chromatography-mass spectrometry (LC-MS) instrument in an oxidant-screening campaign for synthesizing an organic dye. Within human-defined campaign limits, the agent submitted measurement jobs, interpreted mass spectrometry (MS) and diode-array detection (DAD) data, flagged failed experiments, and produced an oxidant ranking. This ranking matched that obtained from scripted peak integration, and organic chemists confirmed the agent's spectral interpretations and failure assignments. In a simulated environment that replays the archived LC-MS data, where the agent chose which wells to inject under varying injection limits, it measured the highest-yielding condition at about its seventh injection, against 14.5 on average for a random order, and identified it in every campaign with a limit of eight or more injections. An agent can therefore close the analysis-to-design gap while its actions stay bounded and verifiable, which is a step toward greater autonomy for SDLs. 

---
# One Prior, Two Dynamics: Repurposing Video Diffusion Transformers for Self-Supervised Latent Action Learning 

**Authors**: Yuxin Jiang, Wenqiang Sun, Jiaqi Li, Yuchao Gu, Lvmin Zhang, Ivor Tsang, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2611.26617)  

**Abstract**: Videos reveal how the world changes, but rarely the actions that cause those changes. Latent action learning recovers such supervision from visual transitions, yet often learns the underlying dynamics largely from scratch. We ask whether a pretrained video generator can instead provide the prior for both inverse and forward dynamics. We introduce O$^{2}$LaF, built on a simple principle: one prior, two dynamics. We initialize both models from the same pretrained video DiT and adapt them to their distinct roles: a feed-forward transition encoder with learnable action queries for inverse dynamics, and a causal, action-conditioned video generator for forward dynamics. By controlling information access and optimizing both dynamics end-to-end through a compact bottleneck, O$^{2}$LaF learns transferable actions from unlabeled videos. Across navigation and robotic manipulation, O$^{2}$LaF sets a new state of the art in few-shot action probing and enables stronger zero-shot action transfer across visual contexts and embodiments. Beyond generative video priors, we further extend our framework to predictive representation priors such as V-JEPA2, achieving competitive performance. Our findings suggest a broader paradigm for latent-action learning: repurposing pretrained video models to discover transferable actions and predict their consequences. 

---
# SoK: Assessing the Evolution of LLM Capability in Software Vulnerability Research 

**Authors**: Amine Lbath, Manan Suri, Aurelien Delaitre, Vadim Okun, Massih-Reza Amini, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2611.26601)  

**Abstract**: LLM-based agents are becoming increasingly capable at discovering, validating and patching vulnerabilities in real software, and vendors and researchers report that this is already changing how vulnerabilities are found and fixed. These claims, however, are scattered across papers, vendor reports and disclosures and are rarely comparable. We introduce a framework that rates these capabilities on an autonomy scale modeled on the levels of driving automation, and qualifies each claim by its correctness, its availability and the strength of its evidence. We use this framework to systematize the published evidence on LLM capability in vulnerability research. We find that agents discovered, validated, and patched vulnerabilities in real-world software without human intervention before the restricted frontier models of 2026, and that open-weight models subsequently attained comparable autonomy for discovery and validation. We then measure the consequences of evolving LLM security capabilities in a decade of public vulnerability records, advisories and exploitation data. In 2026 the number of disclosed vulnerabilities more than doubled, with an increasing share rated critical or high, and severe bugs that had survived a decade or more in widely used code were reported several times more often than before. Yet the time between the stages of a vulnerability's life, from report to fix and from disclosure to attack, has not shifted noticeably so far. We grade the mitigations proposed in the literature with an explicit rubric and draw guidance for each group of stakeholders from the best-supported ones. 

---
# Active Feature Acquisition by Distilling an Oracle Teacher 

**Authors**: Yang Meng, Zixin Ding, Robert L. Grossman, Yuxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.26598)  

**Abstract**: Acquiring every feature of an instance is often costly: a measurement can carry a monetary price, a time delay, or a privacy cost. Active Feature Acquisition (AFA) addresses this by learning a policy that decides, per instance, which features to acquire under a budget. Existing AFA methods train this policy by optimizing a self-constructed utility target rather than distilling an explicit teacher. We construct this teacher explicitly. At training time the label and the unobserved feature values are available, and together they determine the teacher in closed form: an oracle that ranks every candidate by its effect on the downstream predictor's error. OracleImitator distills this ranking into a deployable policy by soft-target imitation. Across multimodal medical diagnosis and the AFABench benchmark, OracleImitator matches or outperforms existing AFA methods across the budget range, recovering 90% of the attainable accuracy gain at under a quarter of the acquisition cost. Because the teacher is an explicit algorithmic object rather than an implicit optimization target, we can also measure directly how much of its ranking is recoverable by a policy restricted to what is visible at deployment. 

---
# PGX: Physically Grounded Execution for Long-Horizon Humanoid Loco-Manipulation 

**Authors**: Kai Lu, Chenyang Ma, Ioannis Havoutis  

**Link**: [PDF](https://arxiv.org/pdf/2611.26596)  

**Abstract**: Robotic vision-language-action (VLA) models can execute diverse individual manipulation skills, yet they still struggle with long-horizon loco-manipulation tasks, where temporal structure, skills, and precision are tightly coupled within a monolithic model. We propose Physically Grounded Execution (PGX), a framework that offloads temporal and precision control to two classical training-free components, allowing the VLA to focus on stage-specific skill dynamics. PGX decomposes long-horizon loco-manipulation into three layers: (i) a predicate-based state machine governs task progression, using physical predicates to gate phase transitions, trigger recovery, and coordinate navigation and waiting. Shared predicate semantics connect offline demonstration segmentation, online execution, and stage-wise evaluation; (ii) VLA skills operate within active manipulation phases, with flat or stage-trained models concentrating on task-relevant actions; and (iii) world-coordinate locking (WCL) maintains spatial consistency through base station keeping, object-relative positioning, and world-anchored contact targets. We demonstrate the effectiveness of PGX with extensive experiments conducted on three simulated tasks and one real-world task using a Unitree G1 humanoid robot for long-horizon loco-manipulation, showing improvements in stage progression and task success rates. Project page is available at: this https URL 

---
# Realized versus Population Reliability: Anytime-Valid Self-Consistency Certification under Correlated Reasoning 

**Authors**: Terence D. Agbeyegbe, Toneukarin G. Agbeyegbe  

**Link**: [PDF](https://arxiv.org/pdf/2611.26592)  

**Abstract**: Self-consistency certification treats repeated sampling from a language model as sequential evidence accumulation and stops when an anytime-valid e-process crosses a threshold, guaranteeing finite-sample Type I error control by Ville's inequality. Prior constructions assume sampled rollouts are conditionally independent. We show this assumption fails in practice: reasoning streams are positively correlated through shared reasoning prefixes or across-prompt heterogeneity, changing not merely certificate efficiency but the hypothesis certified. Modeling dependence via a per-unit latent state (the random-effects / intraclass-correlation structure) induces exchangeable, equi-correlated increments with correlation $\rho$. Correlation splits the single null into a realized null about the drawn belief and a population null about average model behavior, which coincide only when $\rho=0$. We show: (i) the standard betting e-process remains anytime-valid for the realized null, not the population null; (ii) read as a population-null test, a suitable mixture has asymptotic crossing probability at least $\pi=P(\mu(Z)>0)$, potentially exceeding nominal $\epsilon$; (iii) the within-unit mean's effective sample size saturates at $1/\rho$, so extra within-unit rollouts cannot restore population validity; and (iv) a prompt-level e-process restores population validity, with optimal allocation at one rollout per prompt under a pure rollout budget. Simulations and Claude Haiku 4.5 rollouts confirm this regime dependence: on GSM8K ($\hat{\rho}=0.435$, $\hat{\theta}=+0.96$) the gap is dormant, whereas on AIME ($\hat{\rho}=0.480$, $\hat{\theta}=-0.15$) the single-unit rule crosses on 24% of prompts at nominal $\epsilon=0.05$, while the prompt-level procedure declines (0.00). This correctness result identifies which reliability claim an anytime-valid certificate actually supports under dependence. 

---
# Verify the Verifier: Specification-Grounded LLM Cross-Checking for Bugs That Escape UVM Scoreboards 

**Authors**: Ana E. Sanchez Villalobos, Michael D. Quinn, Aakash Tyagi, Minh D. Luu, David Kebo Houngninou  

**Link**: [PDF](https://arxiv.org/pdf/2611.26588)  

**Abstract**: A passing UVM regression shows the design agrees with the reference model on what the scoreboard checks, not that it meets the specification. Bugs that pass every test yet violate the specification, termed scoreboard escapes, survive mature testbenches. We present Verify the Verifier, to our knowledge the first framework whose LLM-based Anomaly Cross-Checker examines a passing regression's traces against the specification. Of 564 bugs inserted into four OpenTitan IPs, 68 are candidate scoreboard escapes: non-equivalent, exercised by the tests, yet passing. The framework confirmed 41 as scoreboard escapes with assertions failing on the buggy design and proven on the original. 

---
# GEOHunter: Mapping a Generative Engine Optimization Network in the Wild 

**Authors**: Sean Tang, Peiran Wang, Ying Li, Esha Shivakumar, Tao Meng, Yuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2611.26564)  

**Abstract**: Generative engines synthesize retrieved web pages into answers, recommendations, and rankings. Generative engine optimization (GEO) services sell visibility in these answers by publishing promotional articles on user-generated-content (UGC) platforms for paying clients. Such promotion raises concerns in medical and financial domains, where misleading advice can cause harm. To study the severity of this problem, we measure one such paid-content operation across three major Chinese UGC platforms, uncovering 1,985 accounts associated with the network comprising 466,357 inferred promotional placements. We find that the network promotes products across diverse categories through a growing pool of accounts, conducting coordinated campaigns spanning months across multiple platforms while evolving its listicle content over time. Across nine generative engines from both the US and China, we find that 4.5% to 20% of retrievable cited pages are attributable to the network, depending on how strictly attribution is drawn. On eight of these engines, its clients are named in 52% of answers, against 68% for category leaders. Our work demonstrates the effectiveness of UGC GEO placement at influencing generative engine outcomes, and reveals content-level defenses as insufficient against coordinated UGC placements, where account-level and platform-level signals are necessary for proper defense. 

---
# Multi-institutional grading of Barrett's esophagus dysplasia without stain normalization using a pathology foundation model and multiple instance learning 

**Authors**: Amirali Khosravi, Jaidip M. Jagtap, D. Chamil Codipilly, Catherine E. Hagen, Jason T. Lewis, Rickey E. Carter, Shahriar Faghani, Vani J. A. Konda, Fouad Otaki, Shajan Peter, Nicholas Shaheen, Sachin Wani, Prasad G. Iyer, Bradley J. Erickson  

**Link**: [PDF](https://arxiv.org/pdf/2611.26562)  

**Abstract**: Barrett's esophagus dysplasia grading suffers substantial inter-observer variability, particularly for low-grade dysplasia, complicating clinical decision-making. We conducted a retrospective multi-institutional diagnostic accuracy study of a foundation model-based deep learning system for dysplasia grading, using 969 H\&E-stained whole slide images from four institutions. Patch-level classifiers trained on expert annotations identified diagnostically relevant regions, followed by attribute-aware multiple instance learning for slide-level classification as non-dysplastic Barrett's esophagus (NDBE), low-grade dysplasia, or high-grade dysplasia. On a held-out test set (n = 130), macro-averaged AUC was 0.96 (95\% CI 0.93--0.98). For binary dysplasia detection, sensitivity was 98\% (80/82) and specificity 81\% (39/48), with a negative likelihood ratio of 0.030 and quadratic-weighted \kap{} of 0.83 against expert consensus. No high-grade dysplasia case was misclassified as NDBE. Performance was consistent across institutions without stain normalization, indicating that stain-robust foundation-model representations may remove the site-specific preprocessing that has constrained prior systems to institution-specific deployment. 

---
# Replication of Agentic Security Events: Audit and Investigation (ROGUE-AI) 

**Authors**: Mengxiao Wang, Mahbub Alam, Kalyan Nakka, Nitesh Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2611.26553)  

**Abstract**: Agents based on large language models (LLMs) can now delete files and send payments on behalf of their users. News media reports routinely describe failures of these agents (e.g., Replit's coding agent deleting a production database), but a report often only says what went wrong and rarely (in)validates the incident or finds its root cause. Incident catalogs collect these reports, and benchmarks record whether agents fail, but neither replicates an incident nor tests the cause its report might identify. In this paper, we present ROGUE-AI (Replication of Agentic Security Events: Audit and Investigation), a multi-agent framework that closes the above gap and rebuilds each reported incident offline together with a matched control that lacks only the blamed cause, and runs agents on both. Our results suggest that ROGUE-AI is able to reproduce the reported harm in 89% of its 108 attempted replications. Agents built on five current GPT, Claude, Gemini and DeepSeek models made the reported mistake in 18% to 33% of runs, against at most 1% without the blamed cause. Of the 72 incidents with real-world harm, 62 had no underlying attacker, and in these, agents did what their task asked for but missed an ordinary hazard, such as a database shared with the live site. Replications without an attacker caused more harm than replications of prompt injection for all five models. Defenses against prompt injection look for injected instructions, but these replications contain none. Further, a warning after each tool output that names the hazard and its consequence lowered harm for all five models, whereas a note of the same length that does not mention the hazard left most of the harm in place. Built from real incidents, the ROGUE-AI replications form a reusable audit framework that vendors can rerun before each new release. 

---
# AURA: AUdio-dRiven streaming Avatar 

**Authors**: Nathaniel Cohen, Nicolas Dufour, Amélie Royer, Alasdair Newson, Patrick Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2611.26540)  

**Abstract**: Current audio-driven video diffusion models can synthesize realistic talking-head videos but generally require dozens of diffusion denoising steps over multi-second video chunks, making them unsuitable for streaming applications such as voice assistants and virtual avatars, where low-latency and online causal generation are required properties. AURA generates video chunk by chunk with a rolling KV cache, requiring only the audio of the current chunk to start generating, rather than a multi-second audio look-ahead like bidirectional models. It comes in two sizes: AURA 1.3B streams 576p video in real time on a single GPU (20.7 fps for a 16 Hz playback rate), making it suited to interactive avatar applications, and AURA 5B streams 720p video at 9.7 fps. Our two-stage distillation pipeline first aligns the causal student with the teacher, by training it to match the teacher's trajectories with fewer diffusion steps and a chunk-wise sliding window causal attention mask, then applies Distribution Matching Distillation (DMD) with backward simulation to close the remaining quality gap. On standard talking-head benchmarks, AURA 1.3B runs 23x faster than its own bidirectional teacher and reaches a quality comparable to TalkVerse at a lower resolution, and AURA 5B matches its TalkVerse teacher at 720p while running 17x faster. To support reproducibility and further research on streaming avatar generation, we release the weights of both models along with a dataset of 30K synthetic videos and their precomputed teacher trajectories. 

---
# Cost-Aware Agentic Feature Optimization System for Fleet-Scale Ranking Models 

**Authors**: Chang Yang, Yao Xuan, Wendan Yan, Nhat Le, Jiang Liu, Longhao Jin, Zhen Peggy Yao, Alireza Vahdatpour, Djordje Gligorijevic  

**Link**: [PDF](https://arxiv.org/pdf/2611.26531)  

**Abstract**: Large Web platforms operate large fleets of ranking models, each selecting features from a shared pool of tens of thousands. We present an end-to-end agentic system that optimizes features across the fleet under resource constraints shared across models and those set per model, such as feature pipeline cost and serving throughput (QPS), respectively. In this system, LLM agents propose hypotheses, and deterministic tools execute experiments and verify them. For each model, an LLM generates and shortlists cross features from a combinatorial candidate space, and all features are then ranked by several importance methods. Feature blending mixes these rankings into a single selection, using Pareto fronts across tasks for multi-task models. An LLM Researcher agent then tunes the feature selection in 10--25 asynchronous trials under resource caps, and a trimming step keeps the pipeline cost within budget. Across production models, the system improves normalized entropy by 0.31% on average within each model's budget, and the trimming step cuts pipeline cost by 2.3% with no regression beyond measurement noise. The optimized models were launched, and in online A/B tests they improved a primary online metric by 0.20%. 

---
# Backtracking Agent Applications Security Incidents 

**Authors**: Yicheng Cai, Taegyu Kim, Peng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26525)  

**Abstract**: Agent applications are increasingly deployed, yet agent-specific attacks pose real threats to them. Upon a security incident, organizations backtrack to identify the CEP (compromise entry point), remediate the damage, and prevent its recurrence. The main artifacts available for backtracking are trajectory logs, which record agent functionalities, and source code; app(lication)-specific function call logs are often fairly limited. However, trajectory logs are at the agent functionality abstraction layer, while the CEP candidates are function calls at the app code execution layer. Due to this semantic gap, the analyst needs to manually inspect the relevant code for each CEP candidate to infer its corresponding agent functionality. In this work, we present Origin, a backtracking tool for the analyst to submit a small number of queries to identify CEPs that existing tools cannot provide. Our approach poses two challenges: (1) define the intermediate abstraction that captures the mapping relationships between the two semantic layers; (2) determine the region-to-variable mapping for constructing the cross-layer graph defined in the intermediate abstraction. Each region is a specific group of agent functionalities. Origin addresses the challenges by (1) the cross-layer dependency graph (CLDG) that connects region-inferred functions at the semantic level to the variables in the low-level taint propagation subgraph, plus the associated query language, and (2) the region inference algorithm. Case studies on five popular agent applications across eight attack incidents show that an analyst using Origin correctly identifies all the CEPs. 

---
# A new algorithmic approach for finding flat minima: critically damped Polyak stepsizes 

**Authors**: Francesco Caporali, Jason M. Altschuler, Boris Hanin  

**Link**: [PDF](https://arxiv.org/pdf/2611.26475)  

**Abstract**: We present a new adaptive stepsize rule that biases gradient descent toward flatter minima in both theory and practice, and establish the first uniform-in-time guarantees for both sharpness and loss reduction for a broad class of locally quadratic objectives. Our critically damped Polyak (CDP) stepsizes multiply the classical Polyak rule by $4$ and a mild damping factor. The factor $4$ is critical in that it produces equioscillations around the manifold of minimizers $\mathcal{M}$ to leading order. Damping provably drives convergence to $\mathcal{M}$. CDP thus operates at the edge of stability by adapting its stepsizes to local curvature. Once the loss is small, its projected iterates follow gradient descent on $\log{\lambda_1(\nabla^2 \mathcal{L})}$ along $\mathcal{M}$ to leading order, paralleling the flatness bias of normalized gradient descent (Arora et al., 2022). Unlike normalized gradient descent, CDP step lengths shrink with the distance to $\mathcal{M}$, enabling convergence to an interpolating solution that locally minimizes sharpness at finite learning rate. We establish our theory for a broad class of non-negative losses with locally quadratic Morse-Bott structure near $\mathcal{M}$, capturing regular interpolation regimes in smooth overparameterized neural networks. Empirically, CDP stepsizes produce a flattening effect similar to that of Sharpness-Aware Minimization (SAM, Foret et al., 2021), reducing sharpness even in settings beyond our theory, including nonlinear teacher-student networks and fine-tuning of pre-trained LLMs. 

---
# Mind2Act World: A Diagnostic Benchmark for Embodied Mind, Physical Action, and Mind-Action Composition in Robotic Manipulation 

**Authors**: Yuzhi Huang, Jie Wu, Fanding Huang, Junchao Lin, Weichen Zhang, Ziyi Xiong, Yang Li, Zhengwei Xie, Zhimin Wang, Zhen Liu, Jiazhen Huang, Yuzheng Zhang, Yuche Liu, Hongyi Yang, Chen Tang, Shixiang Tang, Hakan Aktas, Adam Wu, Duo Wu, Kangye Ji, Shuzhao Xie, Huanan Liu, Weijue Bu, Chenxin Li, Xinyuan Wang, Tianyi Bai, Yinan Mao, Jiawei Wang, Minglei Li, Heng Yu, Ehsan Adeli, Minchen Li, Yong-Lu Li, Cengiz Öztireli, Huazhe Xu, Chao Yu, Jing Zhang, Shanghang Zhang, Weitao Zhou, Wenwu Zhu, Cheng Chi, Jingyan Jiang, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26444)  

**Abstract**: Embodied intelligence aims to let robots perform complex physical tasks with human-like adaptability and reliability, which requires three complementary capabilities: history-based cognitive reasoning, precise and timely physical execution, and the composition of reasoning and execution through environmental feedback. Yet existing benchmarks either evaluate reasoning and execution within the same tasks or assess them separately without their composition, so they reveal little about whether a failure comes from reasoning, execution, or their composition. We introduce Mind2Act World, a diagnostic benchmark of 30 simulated manipulation tasks from five real-world application domains. Each task has three difficulty levels and two scene settings (Clean and Random), yielding 180 configurations. The tasks form three views: Mind tests decisions based on past observations, Act tests precise and timely execution, and Mind2Act combines both in closed-loop tasks where action outcomes guide later decisions. Comparing the three views shows whether a system lacks a capability or fails to compose capabilities it already has. A diagnosis harness further attributes the failed episodes of three representative systems to Mind, Act, or their composition and summarizes each system in a profile card. Our evaluation of multimodal agents, vision-language-action models (VLAs), and world-action models (WAMs) shows that competence in a single view does not carry over to Mind2Act tasks. Even the strongest agent succeeds in 52.7% of Mind and 39.0% of Act episodes but in only 12.0% of Mind2Act episodes. The strongest agent retains task state but struggles with sustained contact, whereas end-to-end policies execute practiced motions but retain little task state. These findings point to composing reasoning and execution through environmental feedback as the main open problem. 

---
# Asuka-FM: A System-Architectural Co-Design Framework for Efficient Text-to-Image Pretraining on Consumer Hardware 

**Authors**: Víctor Rodrigo González, Pablo M. Olmos, Ignacio Peis, Alejandro Lancho  

**Link**: [PDF](https://arxiv.org/pdf/2611.26437)  

**Abstract**: Frontier text-to-image (T2I) foundation models predominantly rely on multi-billion parameter Diffusion Transformers (DiTs) pretrained across massive compute clusters, creating an accessibility barrier for academic laboratories. When training DiTs from scratch on single consumer GPUs (e.g., an NVIDIA RTX 4090 with 24\,GB VRAM), practitioners encounter three primary bottlenecks: quadratic attention sequence costs, Just-In-Time (JIT) compilation graph breaks induced by variable aspect-ratio bucketing, and an autoencoder throughput wall where frozen Variational Autoencoder (VAE) operations dominate step latency. In this work, we introduce \textbf{Asuka-FM}, an end-to-end system and architectural framework demonstrating that competitive dual-stream DiTs ($\sim$206M parameters) can be pretrained from scratch on a single consumer GPU in $\sim$160 GPU hours for under \$55. Asuka-FM integrates four core designs: (i) an AdaLN-free dual-stream DiT backbone reducing parameter count; (ii) structured sequence sparsification via Sparse-Dense Residual Fusion (SPRINT) to cut attention FLOPs by over 50\%; (iii) continuous aspect-normalized Rotary Position Embeddings (RoPE) coupled with shifted square crop training, guaranteeing static tensor shapes while unlocking zero-shot virtual camera control; and (iv) a pixel adaptation phase utilizing a lightweight (10M parameter) Patch Detailer Head operating under velocity regression ($v$-prediction). Evaluated on open-domain benchmarks without benchmark-specific fine-tuning, Asuka-FM achieves GenEval 0.588 / DPG 78.96 scores at 256px and GenEval 0.609 / DPG 78.51 scores at 512px, setting a new Pareto-optimal efficiency standard for democratic foundation modeling. 

---
# Awaken-HI: Scaling Humanoid Intelligence with Whole-Body Human Experience 

**Authors**: EngineAI  

**Link**: [PDF](https://arxiv.org/pdf/2611.26420)  

**Abstract**: A central goal of humanoid robotics is for robots to perform the everyday work people do in homes and workplaces. Existing egocentric video datasets cover many of these activities but provide little synchronized whole-body motion data for learning loco-manipulation at scale. We build Awaken-Human, which pairs approximately 50,000 hours of egocentric stereo video with synchronized whole-body human motion and covers more than 100 skills. Building on this, we also introduce Awaken-HI (Humanoid Intelligence), a vision-language-action model for humanoid loco-manipulation that jointly predicts original human motion and retargeted robot actions through a shared action expert. We show that loco-manipulation skills transfer more effectively from people to humanoids when human trajectories directly supervise the shared action expert. Increasing human pretraining data reduces held-out robot action prediction error according to an empirical scaling law and raises the mean success rate on downstream real-robot tasks. After full-corpus pretraining and T800 adaptation, Awaken-HI achieves a mean success rate of 93.75% across eight individual tasks and outperforms GR00T N1.7 and $\pi_{0.5}$ in both individual and long-horizon evaluations. The same policy generalizes to objects unseen during robot post-training and to new pairings of familiar objects and destinations. 

---
# Flow Matching from Non-Gaussian Sources: A Theory of Efficient Sampling 

**Authors**: Yuchen Zhou, Lili Zheng, Yuxin Chen, Gen Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.26407)  

**Abstract**: Flow matching is a versatile framework for generative modeling that transports a source distribution to a target data distribution. A key advantage is its flexibility in the choice of source distribution, such as heavy-tailed and multimodal sources. Yet finite-step sampling guarantees for non-Gaussian sources remain limited. We establish a nonasymptotic convergence theory for Euler discretization of flow matching with linear interpolation. Suppose the source distribution on $\mathbb{R}^d$ has an everywhere positive, continuously differentiable density, finite total variance $\mathsf{TotVar}_{\mathsf{base}}$, and an $L$-Lipschitz score. Using the population velocity field and a suitable time grid, this sampler achieves total variation error at most $\varepsilon$ in $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ iterations, provided the target distribution satisfies a mild second-moment condition. These assumptions cover, among many examples, source distributions such as multivariate Student's $t$ distributions with more than two degrees of freedom and finite Gaussian mixtures with shared covariance, while accommodating target distributions with neither a smooth density nor bounded support. 

---
# Democratizing Cross-Project Knowledge for Transforming Water Governance and Climate Resilience: an Agentic AI Ecosystem 

**Authors**: Oriol Alàs, Ian Palacín, Iván Cester, Edgar Rubión  

**Link**: [PDF](https://arxiv.org/pdf/2611.26398)  

**Abstract**: EU-funded projects produce extensive evidence in the form of deliverables, reports, and case studies. However, this knowledge is typically scattered across projects and document structures, making it difficult to locate relevant, contextual information when making decisions. This paper introduces an Agentic AI Ecosystem that consolidates heterogeneous project documentation into a domain-aware knowledge base and provides adaptive access through an agentic Retrieval-Augmented Generation pipeline. Documents from three EU-funded projects (IMPETUS, NATALIE, and InnWater) are reconstructed into hierarchical section trees and indexed with structured metadata. Retrieval is supported by two domain-specific embedding models fine-tuned on domain data (water and climate). On top of this knowledge base, a ReAct-style controller iteratively orchestrates retrieval actions by allocating retrieval budget across domain collections and accumulating evidence until a stopping criterion is met. In a retrieval benchmark over the corpus, the proposed approach improves Mean Reciprocal Rank by 26.7% (relative) compared to a single-domain embedding baseline, indicating more consistent retrieval of relevant evidence to support cross-project knowledge synthesis and climate-resilient decision-making. 

---
# Beam Model Technical Report 

**Authors**: Reflection Team, Adam Abdulhamid, Osama Abuelsorour, Yasmin Alameddine, Carlo Alfano, Duarte Alves, Brandon Amos, Selvaraj Anandaraj, Arseni Anisimovich, George Antoniadis, Ioannis Antonoglou, Sheila Babayan, Neeraj Bahl, Märt Bakler, Vidhisha Balachandran, Cameron Banks, Thibaut Barroyer, Navroop Bath, Irwan Bello, Jeremy Berman, Fernanda Bernaldo, Mrityunjay Bhadauria, Sanyam Bhutani, Tolga Bolukbasi, Anudhyan Boral, Joseph Brennan, Ethan Brooks, Sam Buchanan, Anton Bulyenov, John Cai, Catherine Cang, Eason Chan, Elijah Chanakira, Nathan Chandrasekaran, Hao Chen, Hong-You Chen, Jeffrey Chen, Saurav Chhatrapati, Ting-Rui Chiang, Brian Tiger Chow, Aakanksha Chowdhery, Timothy Chung, Rob Claire, John D. Co-Reyes, Anthony Conte, Eduardo Coronado-Montoya, Julien Crozon, Drew Csillag, Joe Cummings, Jeff Da, Samira Daruki, Tim Dawson, Charlie Deck, Arash Delijani, Ousmane Dia, Randy Dodgen, Ryan Douglas, Omar El-Kishky, Ben Ellis, Gamaleldin Elsayed, Simon Fan, Patrick Fernandes, Gabriel Ferns, Nick Fram, Ankush Garg, Florent Gbelidji, Iaroslav Geraskin, Behrooz Ghorbani, Ivan Glushkov, Neil Gollapudi, Scott Goodfriend, Yoni Gozlan, Christopher Grimm, Suriya Gunasekar, Shruti Gupta, Joel Gustafson, Emre Guven, Jon Hackmann, Jeremy Hadidjojo, Jess Hamrick, Ethan Hellman, Joscha Hoche, Spurthi Amba Hombaiah, Uzair Honda, Zijian Hu, Dzmitry Huba, Nancy Hung, Vladimir Ivanov, Shomik Jain, Mojan Javaheripi, Kevin Martin Jose, Aymen Kallala, Amog Kamsetty, Min Hyung Kang, Steven Kapturowski, Ramtin Keramati, Samuel Kessler, Sabina Kim, Seonghyeon Kim, Ivan Komarov  

**Link**: [PDF](https://arxiv.org/pdf/2611.26358)  

**Abstract**: Beam is a sparse Mixture-of-Experts (MoE) model with 501.3B total parameters and 23.4B active parameters, specifically built and optimized for coding, reasoning, and agentic workloads. Beam delivers frontier level reasoning, while requiring 3-4x less inference compute than many models of its class. Pretrained on 23.8T diverse, curated, high-quality tokens, Beam features a 1M-token context window achieved through a novel two-stage midtraining phase, built on verifiable, correctness-anchored data. Post-training leveraged a large-scale, fully asynchronous reinforcement learning (RL) campaign across 10.5K GB300 GPUs, generating over 100M rollouts over 4 weeks of training. We believe this is one of the largest scale RL runs conducted by any open lab to date. Across our evaluation suite, capabilities continued to improve as we increased RL compute, with no sign of a plateau. Supported by deep hardware-model co-design, FP8/NVFP4 quantization, and a 1.93B-parameter speculative decoding drafter (DFlash 2), Beam maximizes serving throughput with efficient, flexible deployment. Averaged across conversation turns and harm categories, Beam refuses or pivots on 88.9% of harmful requests while complying with 81.5% of harmless requests, achieving the highest refusal/pivot rate on harmful requests among open-weight models on the Pareto frontier. 

---
# One Framework to Diagnose Them All: A Stratified Benchmark for Bengali LLM Hallucination 

**Authors**: Asif Shahriyar Sushmit, Shefayat E Shams Adib, Abdullah Al Jubaer Gem, Ahmed Shafin Ruhan, MD. Nazmuddoha Ansary, Biplab Kumar Sarkar, Syed Rifat Raiyan, Ajwad Abrar, Abdullah Bin Hasan, Adib Sakhawat, M. Sakeef Hossain, Ahmed Alfey Sani, Arefin Mizan, Sourab Chakma, Quazi Adibur Rahman Adib, Wassie Haque, Sanjay Kumar Sen, Saad Tahmid, Mohammad Mamun Or Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2611.26336)  

**Abstract**: Hallucination research in low-resource languages fragments easily: each newly named failure mode prompts its own dataset, spending scarce annotators on incomparable resources. We argue for consolidation and test it at community scale. Alikbachan is a Bengali hallucination-detection benchmark stratified along three crossed axes: a human-validated cultural-distance band, six task types, and a five-family, 32-mode hallucination taxonomy. It comprises a harvested 2,516-item test set (Phase 1) and a quota-constructed, gold-labeled 5,000-item set (Phase 2). The stratification is diagnostic. Four zero-shot LLM judges vary by over 30 macro-F1 points across strata, and the item-level predictions of 99 Phase-1 and 32 Phase-2 competition systems reproduce the profile: discourse-level omission defeats judges and most teams alike, whereas only a minority over-flag negative controls. The judges' verdict probabilities rank items well (ROC-AUC 0.74-0.95) but are overconfident (ECE 0.04-0.29). An open competition on Alikbachan drew 222 teams and 3,566 submissions, and a single-prompt 32B judge reaches the 57th percentile. Of the 25 teams in both phases, 24 regress on Phase 2 (mean $-17.6$ points) while every zero-shot judge improves, and Phase-1 rank barely predicts Phase-2 rank (Spearman $\rho=0.35$), so a Phase-1 leaderboard alone would have crowned a different winner. 

---
# Online Temporal Rotations for 2 Bit KV-Cache Quantization in Video Diffusion Transformers 

**Authors**: Ryan Lucas, Petros Prastakos, Rahul Mazumder  

**Link**: [PDF](https://arxiv.org/pdf/2611.26334)  

**Abstract**: Video diffusion transformers reuse key--value (KV) caches across denoising steps, which quickly becomes a bottleneck for long video generation. We find that quantizing temporally distant cache vectors leads to a large range and high quantization error, but that nearby video tokens in the KV-cache share a persistent pattern per channel. These patterns also vary across videos, motivating rotations that are adapted to each video's unique cache. We propose Temporal Rotation Quantization (TRQ), a method that rotates frames in the cache that are adjacent in time before quantization. For each block of temporally adjacent cache vectors, TRQ fits an orthogonal transform that maps the block's per-channel means to a vector that is close to a constant across channels. To keep the quantization format lightweight, we parameterize this rotation as a tree of $2 \times 2$ planar rotations, each stored with only two bits, followed by a fixed Hadamard transform shared across all blocks. The rotations are fit without gradients or changes to model weights, and we prove that a simple greedy fit is globally optimal for our 'mean-flattening' objective. The transforms are fit online for each new video, and we release fused Triton kernels to make this process efficient at inference. At 2-bit precision and matched compression, TRQ improves VBench subject consistency over the state-of-the-art method QVG from 90.38 to 92.02 on LongCat-Video-13B and from 86.33 to 87.61 on HY-WorldPlay-8B, as well as improving other metrics. 

---
# A Control-Theoretic Approximation to Predictive Coding Dynamics 

**Authors**: Ryan Fayyazi, Kyle Daruwalla, Mitra Javadzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2611.26319)  

**Abstract**: Backpropagation remains the dominant method for training deep neural networks, but its reliance on non-local computations has motivated the search for biologically plausible alternatives. Two prominent frameworks, predictive coding (PC) and deep feedback control (DFC), offer distinct approaches to local learning, yet their relationship remains unclear. Here, we show that DFC arises as a local approximation to predictive coding dynamics near fixed points, with exact equivalence in shallow linear networks and systematic deviations arising with depth, nonlinearity, and distance from equilibrium. We characterize how this approximation, which we call Approx PC, deviates from PC in its activity dynamics, and show that these deviations can be mitigated by relinearizing the feedback controller and approaching the output target through a sequence of intermediate nudges. In networks trained on MNIST and FashionMNIST, Approx PC reaches task performance comparable to PC trained with the same nudges, and weights learned with it transfer directly to PC inference. Our results unify energy-based and control-theoretic perspectives on learning and open the door to applying tools from control theory to the training of predictive coding networks. 

---
# C-LiTE: A Lightweight Foundation Model for Cellular feature learning in Histopathology 

**Authors**: Esha Sadia Nasir, Jiaqi Lv, Gozde N. Gunesli, Shan E Ahmed Raza  

**Link**: [PDF](https://arxiv.org/pdf/2611.26316)  

**Abstract**: Characterising individual cells is fundamental to computational pathology, as cellular morphology informs tasks such as tumour grading, mitotic assessment and immune profiling. However, most pathology foundation models are trained on tissue patches and capture cellular features only implicitly, while existing cell-level models are either restricted to specific cell populations or rely on computationally expensive architectures with hundreds of millions of parameters. Applying ViTs directly to small cellular crops also presents an architectural challenge: upsampling increases computational cost without adding morphological information and may introduce interpolation artefacts. To address these limitations, we introduce C-LiTE, a lightweight foundation model for cellular representation learning in H&E-stained histopathology, with only 22 million parameters. C-LiTE employs a native-resolution tokenisation strategy tailored to cellular inputs, processing 64 x 64 px cell-centred crops using a 4 x 4 patch embedding to produce a 16 x 16 token grid without image upsampling. To transfer visual knowledge from large-scale pre-training, we adapt the patch embedding weights of a pre-trained ViT-5-S using a pseudoinverse-based transformation, enabling weight initialisation at the finer patch resolution. We further assemble a pre-training dataset of 60 million cell-centred crops from TCGA, balanced across 19 cancer types and 16 tissue types. To our knowledge, this represents the largest public dataset for self-supervised pre-training on H&E images at the cellular level. We evaluate C-LiTE on seven public datasets spanning mitotic figure recognition, nucleus and cell-type classification, and immune cell phenotyping. Under frozen-feature evaluation, C-LiTE achieves a mean macro-F1 of 0.670, approaching CytoFormer (0.682) with approximately 30x fewer parameters. 

---
# A Tensor-Algebraic No-Go Theorem for High-Dimensional Photonic Greenberger--Horne--Zeilinger States Discovered with AlphaProof Nexus 

**Authors**: Mario Krenn, Moritz Firsching, George Tsoukalas, Rishikesh Gajjala, Xuemei Gu, Swarat Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2611.26300)  

**Abstract**: Photonic GHZ states generated with probabilistic pair sources can be described by edge-coloured, complex-weighted graphs. In this representation, conditioning on an event with one photon in every detector turns the output amplitudes into coherent sums over perfect matchings, and generating an \(N\)-photon \(D\)-dimensional GHZ state is equivalent to constructing a monochromatic quantum graph. Here, we prove that no complex-weighted monochromatic quantum graph exists for any even \(N\ge4\) when \(D\ge N\). The proof, discovered through AI-assisted formal proof search in Lean using AlphaProof Nexus, constructs a weighted sum \(\mathcal F\) of amplitudes whose coefficients factor vertex by vertex. Summing over the colours at one fixed vertex reduces the problem to an \((N-1)\times N\) matrix of effective edge weights. For the graph amplitudes \(\omega\) and GHZ target \(\Delta\), the row--column imbalance yields \(\mathcal F(\omega)=0\) but \(\mathcal F(\Delta)\neq0\), contradicting the required equality \(\omega=\Delta\). Finally, we give a physical interpretation of the AI-discovered mathematical proof. 

---
# Twice-Told Critiques: Claim-Level Homogenization of Peer Review in the LLM Era 

**Authors**: Ilya Gusev  

**Link**: [PDF](https://arxiv.org/pdf/2611.26286)  

**Abstract**: Peer review relies on several readers finding different problems. We measure critique overlap in conference and journal reviews. At ICLR, measured overlap rose from 12.0% in 2026 to 21.3% in 2027, in reviews from the same stage of the review process, with year-on-year rises at NeurIPS, ICML and the F1000 and PLOS journals. A third ICLR review of a paper added 5.46 new problems to the other two in 2027, against 6.12 in 2026. We apply MELD, a detector of LLM-generated text, to all reviews. Within a paper, two reviews that MELD flags overlap more than two unflagged reviews: by 3.0 points at ICLR in 2024-2025, by 4.5 points in 2026-2027, and by 5.4 points across five journal sources. These results provide evidence that LLM-flagged reviews of one paper are less distinct from each other than unflagged reviews, and suggest that the diversity of critiques among reviews of a single paper is falling. 

---
# What If Chain-of-Thought Is Not Enough? 

**Authors**: Marc Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2611.26282)  

**Abstract**: Reasoning capabilities in large language models have evolved from prompting-based methods to internalized model behaviors. In this paper, we investigate whether the internal thinking model of an LLM can be deliberately changed and whether such a change is beneficial. Instead of relying on Chain-of-Thought, we train models to follow Question--Option--Criteria (QOC), a structured decision process that decomposes a task into explicit questions, competing options, and evaluation criteria. We present a three-generation development pipeline that ranges from a basic QOC decision core to richer answer synthesis and statistically enriched QOC reasoning. We then evaluate the resulting models with respect to both internal consistency and output quality. Across the reported experiments, QOC-based fine-tuning improves answer quality relative to non-fine-tuned base models and yields particularly strong gains once models are large enough to execute the underlying QOC computations reliably. The results suggest that the internal thinking model of an LLM is trainable and that structured alternatives to Chain-of-Thought can provide a promising path toward more reliable, interpretable, and task-adapted decision support. 

---
# Digitization of Analogue Gauges using Computer Vision 

**Authors**: Erico Kayumba Mutonkole, Olasupo Ajayi, Omowunmi Isafiade  

**Link**: [PDF](https://arxiv.org/pdf/2611.26262)  

**Abstract**: Analogue gauges remain prevalent in manufacturing industries for measuring physical quantities, yet manual monitoring by operators introduces human error and operational inefficiency. This research developed and evaluated an automated gauge reading system using computer vision and deep learning techniques. Three hybrid approaches were implemented and compared: OpenCV with Scikit-Image (traditional computer vision), OpenCV with ResNet18, and OpenCV with VGG19 (deep learning approaches). A comprehensive dataset of 1,848 pressure gauge images (0-10 BAR range) was curated through web scraping, synthetic generation, and augmentation techniques. The experimental results demonstrate that ResNet18 significantly outperforms the alternatives, achieving an accuracy of 97.3%, compared to VGG19's 64.4% and Scikit-Image's 63.8%. The study validates the effectiveness of residual learning architectures for industrial gauge digitization, providing a robust foundation for Industry 5.0 automation applications. 

---
# SkillBridge: Cost-Aware Diagnosis and Repair of Cross-OS Compatibility Defects in Agent Skills 

**Authors**: Anjie Xu, Zixing Wang, Yifeng Cai, Leye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26256)  

**Abstract**: LLM agents increasingly reuse executable capabilities through skills, but a skill that works on one OS may fail on another because OS assumptions are spread across guide text, scripts, metadata, install contracts, and local entrypoints. Diagnosing such failures by running every skill on every OS is expensive: many executions spend model tokens, runner time, and review effort before revealing whether the failure is truly OS-related. We present SkillBridge, a contract-based workflow with three stages: static analysis, dynamic validation, and repair. SkillBridge first screens 3,270 skill-target-OS cases, covering 1,090 skills across Linux, macOS, and Windows, to find OS-sensitive contract evidence. It then runs dynamic tasks and root-cause review on selected cases to confirm whether a failure is caused by an OS assumption, and repairs the contract field that caused the failure, such as guide text, package scripts, dependencies, or platform declarations. Repair raises the reviewer-judged target-task pass rate from 30.6% to 38.4%, a gain of 7.8 percentage points. A retrospective Linux/macOS budget analysis further shows the efficiency effect: SkillBridge reaches 38 confirmed OS issues with 185 dynamic validations, where uniform random selection would require about 865 validations in expectation. 

---
# STAGE: Spatio-Temporal Augmented Graph Expansion for Weakly Supervised Cross-Tweet Event Causality Discovery 

**Authors**: Hieu Minh Duong, Todd Gary, Eugene Levin, Long Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.26255)  

**Abstract**: Causal relations in disaster-related social media of- ten span independent posts, making cross-tweet causal discovery difficult to supervise and model at scale. We propose STAGE, a framework that separates scalable cross-tweet weak-supervision construction from spatio-temporal relational learning over event occurrences. Starting from local causal seeds, STAGE retrieves semantically related event occurrences across tweets and uses a causal scorer to construct weakly labeled cross-tweet pairs. These pairs supervise a multi-head Graph Attention Network over semantically enriched event representations and a spatio- temporal context graph. We evaluate STAGE on California Wildfire and Hurricane Harvey using human-gold cross-tweet test sets and 15 baselines. Compared with local-only supervision, the constructed cross-tweet supervision improves STAGE AUPRC by 0.340 on Wildfire and 0.107 on Hurricane. STAGE achieves in-domain AUROC of 0.885 on Wildfire and 0.710 on Hurricane. Cross-disaster experiments reveal asymmetric transfer, highlighting sensitivity to dataset-specific spatio-temporal shifts. These results demonstrate the value of scalable cross-tweet supervision combined with relational event modeling for causal learning across fragmented disaster reports. 

---
# The Stability Gap in Continual Reinforcement Learning 

**Authors**: Sebastian Pusch, Matthia Sabatelli, Gido M. van de Ven  

**Link**: [PDF](https://arxiv.org/pdf/2611.26252)  

**Abstract**: Continual reinforcement learning (CRL) aims to enable agents to adapt to changing tasks while preserving previously acquired capabilities. Evaluation often relies on final aggregate metrics that can hide temporary performance degradation during adaptation. This work introduces a simple CRL benchmark that varies observation representations while preserving environment dynamics and rewards, supporting proof-of-concept demonstrations, inexpensive experimentation, and easily reproducible sanity checks. Using this benchmark, we demonstrate the stability gap in CRL: a substantial drop in prior-task performance after the introduction of new tasks, followed by partial or complete recovery. This pattern occurs despite the agent having complete access to past tasks. 

---
# FastRoverVLA: Fast Vision-Language-Action Execution for Off-Road Autonomous Driving 

**Authors**: Changtao Zhao, Yingxue Guan, Li Wang, Shijie Zhao, Yuhang Zhang, Shengkai Zhang, Qing Wang, Ziying Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.26242)  

**Abstract**: Off-road vision-language-action (VLA) models generate scene-conditioned plans, but frequent trajectory updates can repeatedly invoke expensive multimodal inference. Reusing a prediction reduces this demand while leaving its commands exposed to changes in vehicle motion. FastRoverVLA addresses this mismatch by decoupling visual plan generation from motion-conditioned execution. Its training-free interface maintains decoded plans between visual predictions. A speed-curvature action buffer is aligned with measured motion at each update, integrated into a local trajectory, and advanced by consuming the current action. The corrected remainder is written back to the buffer, preserving motion corrections across successive updates. A bounded refresh rule normalizes speed and curvature variations within a newly decoded plan and assigns a short or long reuse budget according to the resulting plan-roughness score. The interface requires neither backbone features nor an additional learned driving policy. Over a 3 s horizon, independent real-request evaluation achieves average displacement error (ADE) and final displacement error (FDE) of 0.328/0.596 m on ORAD-3D and 0.266/0.526 m on RELLIS-3D, reducing FDE by 18.0% and 36.7% at request rates of 6.0% and 4.8%, respectively. These runs retain lower errors across four backbones and both datasets and reduce Qwen3-VL request-processing cost by factors of 16.6 and 20.8 under serial replay, while mechanism and motion-only controls identify persistent state correction as the main source of the observed accuracy benefit. The source code and evaluation materials are publicly available at this https URL. 

---
# Sightline: Calibrated Data-Flow Summaries for LLM-Integrated Applications 

**Authors**: Zihao Xu, Yuekang Li, Ruitao Feng, Haifeng Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.26221)  

**Abstract**: Programs increasingly call large language models (LLMs), filling prompt placeholders with program values and consuming the output. Such a call crosses the boundary between natural language and programming language (NL/PL). To follow data across it, an analysis needs a data-flow summary of which placeholders the output uses, but no code describes what the model does. Analyses of these LLM-integrated applications therefore taint every output with every input, creating spurious flows, or ask a second model to label each call, which is costly. We prove, and mechanize in Lean 4, that a placeholder flows to the output of a transformer, to first order, only through the attention that reads it (no reading, no flow), and that a faint reading or sensitivity cannot certify that a placeholder does not flow. Unlike reading, the log-probability gradients and Fisher signals of existing methods vanish on confident outputs, and single-span removal fails on duplicated content. Our tool, Sightline, reads a recorded call in one forward pass, learns from labels which reads are uses, and calibrates the pruning to miss at most a chosen fraction $\alpha$ of true flows, on average over call sites, when deployment is exchangeable with calibration. On 2,103 unseen repository families and four models, Sightline is the most accurate summary against the seven closest methods and a hidden-state probe. Calibrated at $\alpha$ = 1%, it misses about 1% of true flows and prunes 43.9-65.6% of spurious flows, 1.6 to 2.6 times as many as the best baseline, and still 43.8-60.1% when calibrated on another collection. Through an open surrogate, it reads models treated as closed about as accurately as they read themselves, without their labels. In the information-flow control of an LLM agent, it removes about half of the labels from inputs that do not flow. 

---
# RegionAgent: Two-Stage Preference Optimization for Regional Image Restoration 

**Authors**: Weiyu Zhou, Jiacong Tang, Tao Hu, Axi Niu, Jinqiu Sun, Peng Wu, Wei Dong, Qingsen Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.26179)  

**Abstract**: Images under non-uniform coupled degradations require diverse restoration capabilities for different degraded regions. Existing methods employ globally uniform operations, potentially overlooking regional differences and causing incomplete restoration or even damage to image content. Moreover, a degraded region may require successive restoration actions, and earlier actions may alter residual degradation characteristics or introduce artifacts, potentially hindering subsequent restoration. To address these challenges, we propose RegionAgent, an adaptive restoration framework that trains a Vision-Language Model (VLM) through two-stage preference learning to generate a region-specific restoration action at each iteration. Specifically, we first train the VLM to select an action suited to the current region through step-level Direct Preference Optimization (DPO), using preferences derived from immediate restoration feedback to supervise only the differing component in each action pair. To account for the effects of current actions on subsequent restoration, we then apply trajectory-level DPO, using preferences derived from complete trajectory scores to supervise the sole differing action in each trajectory pair. Furthermore, as conventional image-pair datasets do not directly provide preferences over regional restoration decisions, we construct a regional restoration preference dataset comprising 10K action pairs and 20K trajectory pairs for two-stage VLM training. Experiments across multiple datasets demonstrate strong restoration performance under both mixed and single degradations. Ablation studies further validate the effectiveness of two-stage preference learning in improving restoration quality. 

---
# Budget Internalization: Fixed-Budget RLVR Trades Test-Time Scaling for In-Budget Accuracy 

**Authors**: Ilia Zavidnyi, Alexander Kovrigin, Arina Puchkova, Kirill Gelvan, Sergey Titov  

**Link**: [PDF](https://arxiv.org/pdf/2611.26172)  

**Abstract**: Reasoning models often become more accurate when they can generate longer answers at test time. This behavior is known as test-time scaling. However, RL with verifiable rewards (RLVR) usually trains them with a fixed rollout cap. We find that this cap changes model behavior, not only training cost. After training, models use few additional tokens beyond the training budget. They outperform the base models near that budget, but their accuracy stops improving and falls below the base model at larger budgets. The rollout cap thus becomes a learned constraint, that limits model's test-time scaling. We call this effect budget internalization. We observe it across Qwen3.5-4B, Gemma4-E2B, and Nemotron-3-Nano-4B and two reasoning training tasks: mathematical reasoning and diverse, procedurally generated problems. This effect also transfers from math training to science and coding tasks. Training-objective level ablations on Qwen3.5-4B show that most changes recover little test-time scaling. Stronger KL regularization and lower training-data truncation preserve more scaling, but stronger KL reduces in-budget accuracy. Forcing an answer on truncated training rollouts largely removes budget internalization, but also removes most of the in-budget gain. SFT on artificially truncated traces reproduces budget internalization, whereas SFT on naturally short traces does not, which shows that the effect is not specific to RL. This result links budget internalization to length-censored post-training rather than to an RL-specific objective term. Thus, a rollout cap does not only control training cost. It can trade low-budget gains for lower performance when additional inference compute is available. 

---
# ChronoPatch: Black-Box Energy-Latency Attacks on Vision-Language Models 

**Authors**: Jiawei Xing, Li Wang, Xiangtao Meng, Chuanchao Zang, Rui Wen, Zheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.26117)  

**Abstract**: Vision-language models (VLMs) are increasingly deployed as generative inference services, where autoregressive decoding makes response length a key determinant of latency and computational cost. This creates an availability risk: adversarial visual inputs can prolong generation and increase serving overhead. Existing energy-latency attacks typically rely on gradients, logits, or token-level feedback, leaving open whether such risks persist when an attacker observes only the final response. We present ChronoPatch, a black-box attack that induces prolonged VLM generation requiring only the final response. It discovers visual delay anchors through response-length-guided evolutionary search, then minimizes perturbation magnitude through subspace-calibrated decision-based optimization. Across four open-source VLMs, ChronoPatch increases inference latency by 47.86%--293.34% and GPU energy consumption by 41.44%--249.70% under a 5% spatial budget. It also induces more repetitive outputs, as reflected by increased n-gram repetition ratios. On three commercial VLM APIs, ChronoPatch increases response length by 37.6%--81.5% and client-observed latency by 1.82x--2.29x under query budgets of 500 per API. These results reveal a generation-based availability risk that persists under final-response-only access and motivate generation-aware safeguards for deployed generative AI services. 

---
# HiUni: Harmonizing Understanding and Generation in Shared Visual Space 

**Authors**: Jiahao Hu, Jingdong Du, Yichen Zhang, Yi Song, Zhenyu Zhang, Jiaxin Niu, Ruihang Li, Xiangzhao Hao, Yiqian Zhang, Yong Huang, Zefeng Zhang, Zheyuan Gu, Shuohuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26107)  

**Abstract**: Unified multimodal models aim to support visual understanding and generation within a single framework. However, prevailing approaches often separate their visual representations or backbone computation to accommodate the different requirements of these two tasks, which limits the degree of unification. In this paper, we present HiUni, which revisits this design choice and explores whether a pretrained vision language model can be directly extended into a unified model with extensive parameter sharing. Specifically, HiUni reuses the same language-aligned ViT and dense Transformer backbone for both capabilities, while confining task-specific computation to the output side. Yet such a design introduces two key challenges. Generation in pixel space is difficult to optimize, and semantic abstraction in pretrained ViT discards fine spatial details required for synthesis. To address these challenges, HiUni introduces a \emph{Pixel Diffusion Head} tailored to direct pixel-space optimization and a lightweight \emph{HiGate} pathway that provides fine-grained spatial details near the output stage. Extensive experiments show that HiUni achieves a similar average understanding score to the pretrained VLM while attaining competitive performance on image generation and editing. Controlled comparisons and analyses further demonstrate that a single dense backbone can effectively support both capabilities without task-specific backbone computation, highlighting a simpler path toward unified multimodal modeling. 

---
# AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators 

**Authors**: Nagham Omar, Rotem Dror, Almog Gueta, Avigdor Gal, Roi Reichart, Nitay Calderon  

**Link**: [PDF](https://arxiv.org/pdf/2611.26073)  

**Abstract**: Large language models (LLMs) are increasingly used as judges and graders to replace human annotators, raising the question of when such replacement is statistically justified. To support such decisions, the Alternative Annotator Test (Alt-Test) was introduced for dense settings where all items are annotated by the same annotators. However, in many large-scale annotation settings, items are annotated by different subsets of annotators. We therefore propose the Alternative Annotator Test with Resampled Annotations (AltRA-Test), a statistical test that does not require that all items are annotated by the same annotators. AltRA-Test uses a bootstrap that resamples items, draws k human annotations for each item, and substitutes one of them with an LLM annotation. This produces paired all-human and mixed human-LLM panels. A replacement is supported when the inter-annotator agreement (IAA) of the mixed panel is statistically non-inferior to the IAA of the all-human panel, i.e., when the LLM preserves the signal that the human annotators share. We establish that AltRA-Test is a consistent test of asymptotic level alpha. We then evaluate it through simulations and real-world experiments in which we test 18 LLMs and 5 prompting techniques on 20 datasets across text, image, audio, and video. Across these evaluations, about 100 items often suffice for the test. In simulations, where we know whether the LLM is a better or a worse annotator than the humans, AltRA-Test makes more correct replacement decisions than existing baselines. In real-world data experiments, LLMs pass the test in about half of the setups, depending mainly on the task. Since replacement is not always justified, we encourage researchers to test it for their task. 

---
# GenSPARQL: A Generative Extension of SPARQL for Incomplete Knowledge Graphs 

**Authors**: Jingcheng Wu, Yi Wang, Hongkuan Zhou, Steffen Staab, Bo Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2611.26062)  

**Abstract**: Knowledge graphs on the Web are incomplete, and SPARQL answers a query only from the stored triples. Large language models (LLMs) can predict missing facts, but an LLM returns names instead of IRIs and requires one call per binding of a query variable. We present GenSPARQL, a generative extension of SPARQL 1.1 with an operator GENOP that binds values generated by an LLM to query variables during query evaluation. A similarity join maps generated names to entities of the graph, and the query planner invokes the LLM only for the bindings selected by the other graph patterns and filters. In a web workbench, attendees query two graphs built from Wikidata and inspect generated values, mapped entities and LLM calls. In our evaluation with three open-weight LLMs, the embedding similarity maps 83% of the entities named by the LLMs and yields only one incorrect match. 

---
# Does the Workspace Do the Work? A Causal Protocol for Verifying Global-Workspace Computation in a Language Model 

**Authors**: Diaoxuan He, Linqian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26018)  

**Abstract**: Global Workspace Theory (GWT), a leading theory of consciousness, holds that information becomes conscious when it is broadcast to specialized processes that use it. Building a workspace into a network does not reveal what the trained network computes with it: a module that reports an answer may have combined a broadcast message with its own knowledge, or merely forwarded an answer computed elsewhere, and task accuracy and ablation cannot tell these apart. We introduce a causal protocol that identifies who computes when modules communicate. Its central test transplants a message from another input, chosen so that ignoring, relaying and recombining it predict three different answers. On networks built to relay or to recombine, the test assigns the right computation in every case. Among trained workspace models it finds all three outcomes: one leaves its broadcast nearly unused, and of two models that differ only in their training data, one relays and the other recombines. In AFGW, a language model whose modules communicate only through a learned workspace, a transplanted message yields the answer that recombination predicts in 97% of cases, and its content acts by redirecting what the receiver retrieves from its own memory. The tests carry over to other communication designs and to a known interface of a pretrained Transformer, turning GWT's central claim into a testable property of trained models. 

---
# Policy Improvement-Guided Replay for Continual Reinforcement Learning 

**Authors**: Qin-Wen Luo, Haoxuan Gao, Sheng Ren, Xiang Chen, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25989)  

**Abstract**: Continual reinforcement learning (CRL) requires models to acquire new capabilities while retaining performance on previous tasks. A stability-oriented perspective common in conventional continual learning emphasizes knowledge preservation and interference mitigation. Complementing this perspective, we revisit CRL for vision-language models through policy improvement, a fundamental principle of RL, to preserve acquired performance gains. Using the performance difference lemma, we characterize historical performance changes through the current policy's expected advantages on historical task distributions, motivating replay of historical prompts with the current policy. To allocate limited replay memory, we propose Contrastive Likelihood Replay (CoLiR), which contrasts likelihood gains on correct responses from the final task policy with likelihood reductions on incorrect responses from the initial policy. Our analysis links correct-response likelihood gains to reference-policy advantages under an idealized KL-regularized optimal update and provides an endogenous-reward interpretation for general policy updates. CoLiR estimates scores from finite training rollouts and cached log-likelihoods, selects prompts at task boundaries, and replays them uniformly during subsequent training, with KL regularization restricted to replayed prompts. Memory construction only requires a small amount of additional cross-policy scoring but no additional response generation. Across three Qwen3-VL model sizes on MRCL, CoLiR achieves the highest mean final accuracy and mean task accuracy among the compared methods while maintaining competitive new-task performance. 

---
# Law at the Speed of AI: Critical Legal Infrastructure for the Machine Age: Autonomy Under Law Architecture (AULA) 

**Authors**: Natasha E Blycha, James Myint, Chen Feng, Dr Amani Soysa, Koshin Jama, Schellie-Jayne Price, Matthew Blycha  

**Link**: [PDF](https://arxiv.org/pdf/2611.25985)  

**Abstract**: The global deployment of autonomous AI agents within and across organisations exceeds the governance infrastructure available to constrain it. Agent communication and authentication protocols, principally the Model Context Protocol and the Agent2Agent Protocol, are maturing quickly, but maturity at the protocol layer does not produce legal legibility at the governance layer. Runtime enforcement written by the deploying organisation fixes the resources an AI agent can reach, not the authority under which it acts. Legal legibility is the capacity of a legal system to recognise, interpret and regulate activity by mapping it onto legal personhood, property, rights and liabilities. Existing (analogue) legal infrastructure cannot supply it for autonomous deployment. The resulting deficit is bidirectional. It concerns the attribution of liability when an AI agent causes harm, and the allocation of rights, including data and economic rights, generated by agentic activity and by the AI agent itself as a source of value and form of property. This paper characterises that deficit as the legal attribution gap, a material and unaddressed risk to economies, organisations and citizens that is magnified in cross-organisational multi-agent deployment. The paper describes the Autonomy Under Law Architecture (AULA), in which AI systems and data assets are registered as legal objects anchored to one or more identified legal persons, governed by cryptographically tethered smart legal instruments and evaluated at runtime. Each smart legal instrument can carry a legal context clause, through which the legislation, regulation and judicial decisions in force at the time of the act are given practical and legal effect in that evaluation. Infrastructure of this kind is a precondition for the continued function of any legal, property and economic system with both human and autonomous machine stakeholders. 

---
# Safe Subgame Resolving in Multiplayer Games 

**Authors**: Boning Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25981)  

**Abstract**: Poker and other games with hidden information are too large to solve in one pass. Strong systems compute a complete blueprint strategy offline and refine it in real time inside the reached subgame. In 2p0s games, a safety theorem guarantees that subgame resolving never increases how much an adversary can gain by exploiting the resolved strategy relative to the blueprint strategy. With three or more players acting independently, no such guarantee exists yet, and heuristic resolving can increase the profile's best-response gap. We prove the first conditional safety theorem for uncorrelated multiplayer subgame resolving, a post-hoc certificate for any delivered update. The best-response gap of the resolved profile exceeds that of the blueprint by at most four terms. Two concern the resolver: the subgame value the blueprint promised without realizing ($A_i^{S}$) and the resolver's subgame solve error ($\varepsilon_i$). The other two are the opponents' best-response gains inside the subgame over their promised values ($\sum_q M_{q}$) and a structural tax ($\Theta$), which measures how much the players' total utility can change. The tax is zero in constant-sum games, including every rake-free poker variant. Constructions attain each term with equality. On three-player Leduc poker and on Kuhn poker with up to six players, resolving without a certificate raises the gap beyond a small tolerance in 12% to 100% of the updates. The certificate flags every such update and accepts 41% (Kuhn) and 86% (Leduc) of the standard resolving updates. A resolver that keeps the blueprint whenever the certificate exceeds the tolerance delivers no harmful update. The certificate needs only one best-response pass per opponent inside the subgame, so real-time multiplayer solvers that refine a blueprint can attach it as a safety check. Code is available at this https URL 

---
# Multi-View Transformer with Geometry-Guided Multi-Modal ControlNet for 3D Virtual Try-On 

**Authors**: Sarthak Batra, Armin Mustafa, Adrian Hilton  

**Link**: [PDF](https://arxiv.org/pdf/2611.25979)  

**Abstract**: Editing a 3D Gaussian Splatting (3DGS) capture of a person so that they wear a new garment requires every rendered view to agree on the same garment details. Existing 3D virtual try-on (VTON) pipelines, however, edit training views largely independently and fuse them with hand-designed heuristics such as manually chosen reference views and fixed attention-blending weights. The resulting inconsistencies are baked into the 3D representation and surface as texture drift, blurred prints, and implausible geometry in novel views. We present GS-VTON-MVT, a three-stage framework that combines implicit and explicit 3D reasoning for consistent 3DGS try-on. (i) A multi-view inpainting transformer jointly edits a set of training views with full cross-view self-attention, so that cross-view correspondence is learned rather than imposed. (ii) A geometry-guided multi-modal ControlNet conditions the diffusion editor on the garment image together with depth and surface normals rendered from the current 3DGS state, supplying explicit structure where visual evidence is weakest. (iii) A pose-adaptive persona-aware attention replaces the fixed blending weight between current-view and reference-view features with weights derived from camera-pose proximity. On the 3D-VTONBench benchmark, GS-VTON-MVT improves FID, KID, LPIPS-based cross-view consistency, and user preference over general-purpose 3D editors and prior 3D try-on methods. 

---
# Energy estimates of open-weight large language models across stated food weights: a dose-response evaluation 

**Authors**: Donato Riccio, Fabrizio Maturo, Iulia Brezeanu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25976)  

**Abstract**: We tested whether energy estimates of open-weight large language models (LLMs) scale proportionally with an exactly stated food weight, and whether computing energy density times weight removes the error. Under a prespecified protocol, seven open-weight LLMs estimated the energy of 168 randomly drawn foods from the USDA Food and Nutrient Database for Dietary Studies at nine stated weights, from 0.25 to 4 times the default amount. The primary outcome was the elasticity of the estimate with respect to weight (1: proportional scaling). A two-step estimate multiplied each model's own stated energy density by the weight in code. All seven models compressed their estimates, with elasticities from 0.548 to 0.979 (every 95% CI below 1); four compressed severely (0.548 to 0.709). These four overestimated a quarter of the default amount by 47% to 130% and underestimated four times it by 20% to 34%. Supplying the density restored near-proportional scaling in three of the four, and the two-step estimate cut the typical error factor at these extremes from 1.47-2.43 to 1.20-1.38. Compression persisted at full precision and grew in five of seven models under a plain question. The models hold the energy densities needed for proportional scaling and answer near a typical serving, reproducing the flat-slope syndrome of dietary recall. Accuracy at usual portions does not reveal it. Food-logging tools should compute the energy of weighed entries as density times weight, and evaluations should test fixed foods across stated weights. 

---
# Web search in Latin America: language models use less local and diverse news than search engines 

**Authors**: Francisco W. Kerche, Franziska Sofia Hafner, Felix M. Simon, Luc Rocher  

**Link**: [PDF](https://arxiv.org/pdf/2611.25955)  

**Abstract**: AI platforms are becoming a major gateway to news and current events. This shift risks amplifying existing geographical biases that favour narratives from wealthier countries over those from the Global South. This is particularly relevant in Latin America, which accounts for the largest share of AI usage in the Global South. We examine how information about Latin America changes when users shift from conventional search engines to LLM-based search. We construct a dataset of 1,901 current events across nine Latin American countries, craft realistic queries, and submit them to three search engines and four language models. We vary query language and user location to represent local, diaspora, tourist, and foreign user scenarios. Across 49,207 requests and 8,834 unique domains, language models return a less diverse set of news sources than search engines. Across user types, the share of domestic news returned by language models is inconsistent. For local users, while search engines already return only 41.3\% local news, language models return a further 3.6 to 11.6 percentage points less. Taken together, our findings suggest that LLM-based search can reproduce and, in some settings, amplify geographical hierarchies in the reach and visibility of news sources. We release an open dataset to support multilingual and geographically diverse evaluation of LLM-based search, and highlight implications for more equitable information systems design. 

---
# Fast-JEPA: Advancing the Success-Latency Frontier for Latent World Models 

**Authors**: Yao Zhang, Bin Yu, Shijie Lian, Haishan Liu, Changti Wu, Yu Wu, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.25927)  

**Abstract**: Latent world models predict the outcomes of action sequences, but comparing candidate plans through recursive rollout remains expensive at test time. We introduce Fast-JEPA, a latent world model planning framework that advances the success-latency Pareto frontier of predictive search. Fast-JEPA reorganizes predictive computation to support effective action-sequence search at low planning cost. It constructs a goal-directed plan to guide search and directly predicts the outcomes of complete candidates for efficient comparison, avoiding recursive state rollout during refinement. Across PushT, Reacher, Cube, and TwoRoom, Fast-JEPA achieves 96.50% average success rate with 2.41x faster planning than INTACT. Search-budget sweeps show that its competitive success and low latency extend across candidate counts and refinement depths. We further introduce DynaGoal, a benchmark for evaluating world-model planning under dynamic out-of-distribution conditions, execution deadlines, and controlled response delays. Fast-JEPA achieves average success rates of 73.83% on DynaGoal-Shift and 87.67% on DynaGoal-Deadline, demonstrating the practical value of low-latency predictive planning under changing control conditions and execution-time constraints. Our code is available at this https URL. 

---
# Many Solutions, One Answer: Search Order Shapes the Default Solutions of Language Models 

**Authors**: Qinhan Hou, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25916)  

**Abstract**: When a constraint puzzle has several correct answers, language models tend to give the same this http URL study four open-weight models from three families on symbolic puzzles and multi-solution ZebraLogic variants with exactly known solution this http URL samples concentrate on one default solution far beyond uniform sampling, and different models often share this http URL models usually produce even rarely sampled solutions when added clues uniquely identify this http URL held-out symbolic puzzles, we compare visiting entities in name order with visiting those with the fewest initial candidates first. Both searches try values in the stated order. Where their pre-specified predictions differ, the latter matches a larger share of responses in all four models, by 26 to 40 percentage this http URL the stated value order or specifying a search order shifts responses towards the solution that the new order favours, on both testbeds. Randomising the presentation or supplying a random search order raises breadth (the effective share of solutions in use) by 26 to 42 percentage points on the symbolic task and 11 to 26 on ZebraLogic. The largest observed validity losses are 3 and 4 percentage points, respectively. Randomised presentation broadens answers more than a random-choice instruction in each of the three main models. 

---
# Verifiable Logic in Medical LLMs: A Critical Review of Agentic Clinical Reasoning Optimization and Neuro-Symbolic Grounding 

**Authors**: Youssef ElNahas, Adel Ammar, Soumaya Chaffar, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2611.25898)  

**Abstract**: Clinical large language models are usually judged by whether the final answer is right. Yet a model can reach a fluent, plausible output through invalid reasoning that static benchmarks do not detect, and confident hallucination, causally invalid inference, and spurious demographic shortcuts can pass conventional evaluation. This critical review covers four paradigms in turn: how clinical hallucination arises and why static metrics miss it; the agentic hallucination cascade, in which multi-agent and tool-augmented systems propagate and amplify early reasoning errors; process reward modeling as step-level supervision; and neuro-symbolic grounding, in which formal logic engines, ontology reasoners, and knowledge graphs place verifiable constraints on neural output. Within their encoded domains, systems with symbolic veto layers offer guarantees that hold relative to their encoded knowledge, and they report large gains on the safety measures they use. These results come from single studies, mostly in simulated or small-sample settings, and are not directly comparable with neural baselines. Process reward models remain bounded by the statistics of their training data. We propose a conceptual framework that maps each agentic failure mode to a symbolic intervention point, and we argue that clinical AI safety depends substantially on system architecture. Because current regulation emphasizes performance and does not require verification of reasoning steps, we argue that responsible certification should also audit reasoning trajectories alongside accuracy on static benchmarks. 

---
# Evolving Decoding Algorithms for Quantum Error Correction 

**Authors**: Eduardo Dueñez, Edgar Said Hernandez Sanchez, Edgar A. Duéñez-Guzmán, Suzanne Sadedin  

**Link**: [PDF](https://arxiv.org/pdf/2611.25880)  

**Abstract**: Quantum computation entails robustness engineering: a noisy quantum state is measured continuously, and a classical algorithm (the decoder) must infer corrections that reliably remove the noise, fast enough to keep up. We write several published belief-propagation (BP) decoders for quantum low-density parity-check (qLDPC) codes in a Domain-Specific Language (DSL), and search over it with grammatical evolution. Written in the language, the published decoders show their parameters in plain view, allowing us to re-parameterise BP+OSD (ordered statistics decoding) and BP+LSD (localized statistics decoding) decoders with damping, before any search, making each 17.1--41.0$\times$ faster at a higher error rate, and BP+OSD-0 about 103 faster than the ldpc package's version tuned on the [[72,12,6]] bivariate-bicycle code. Evolution over the language finds decoders that improve the time-error front. One makes 23% fewer logical errors than BP+OSD-0 with damping at 0.64$\times$ its time, and decoders evolved from random code are more accurate than ldpc's tuned BP+OSD-0 at 1/31 of its time or less. Frozen at one operating point, the two fastest evolved decoders generalize well across noise levels, code sizes and round counts, staying at the time-error front. Ultimately, we demonstrate that capturing expert knowledge within a DSL creates a powerful intermediate layer---one that effectively bridges human intuition and machine-driven optimization to unlock highly efficient quantum error correction. 

---
# On the trajectories of the ID algorithm 

**Authors**: Ivano Lodato, Simran Hiranandani, Aditya Varna Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2611.25877)  

**Abstract**: The ID algorithm decides whether an interventional distribution can be computed from observational data and, when it can, returns a formula in which products, ratios and summations of conditional probabilities are nested inside one another. We study these formulas by following the trajectories of the calls of the algorithm, each of which ends with a tail (an alternation of lines 7 and 2) and an exit, and we derive two main results. First, line~4 and the exit lines fix the single outer summation of the output, while the tail forms everything inside it, both the inner summations and the ratios. Second, a ratio keeps a summation in its denominator only when a repetition of the tail is entangled, and a test of ancestry in the original graph decides, before any formula is computed, whether a topological order can be chosen so as to avoid it. We also derive a bound on the nesting depth of the output, and show a simple ``dimensional'' count that every output has to satisfy. Together, these results constrain the form of an ID output and guide its implementation and simplification. 

---
# LINEAGE: A Benchmark for Empirical Lineage Retrieval in AI Scientific Literature 

**Authors**: Pangjing Wu, Xu Yuan, Wenqi Fan, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25845)  

**Abstract**: Current research agents are moving beyond academic search toward literature synthesis and experiment planning. These workflows increasingly rely on literature retrieval systems, which serve as standalone systems or as components of research agents. However, existing scientific literature retrieval benchmarks mainly evaluate these methods using topical relevance or citation links as supervision, which cannot capture which prior methods are used for empirical comparison. To address this gap, we introduce Lineage, a benchmark for empirical lineage retrieval in AI scientific literature. An empirical lineage relation links a prior paper to a source paper that explicitly uses its method as an empirical comparison baseline. These relations form lineage graphs whose multi-hop neighborhoods provide retrieval ground truth for structural evaluation. To construct these graphs at scale, we develop Lineage-Constructor, which converts heterogeneous paper records and full-paper empirical baseline evidence into calibrated lineage graphs. Built from 121,048 papers across 17 top AI venues from 2018 to 2025, Lineage contains 3,950 graph components, 67,502 paper nodes, 164,948 lineage edges, and 80,000 multi-facet retrieval queries. Experiments with 19 representative retrieval systems show that empirical lineage retrieval remains challenging. Performance drops sharply with lineage distance, while structural metrics reveal that isolated retrieval hits often fail to recover closely connected empirical comparison neighborhoods. 

---
# Harnessing Medical Visual-Language Models with Agentic Evidence Management 

**Authors**: Qiming Huang, Hanze Dong, Jipeng Zhang, Mike Zheng Shou, Xinxing Xu, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.25840)  

**Abstract**: Medical vision-language agents increasingly rely on segmentation tools, and recent work has focused on acquiring the right mask, often treating a good mask as the end point of tool use. But do these agents use the resulting evidence effectively? To answer this question, we need a benchmark that separates evidence use from segmentation quality, spans multiple medical image types, and allows the same mask to be presented in different ways. Existing medical VQA datasets do not jointly provide these properties. We therefore construct Mask-Evidence MedVQA, a 2.5K-example benchmark spanning four image types, where each question is paired with an answer-relevant ground-truth mask and multiple possible renderings. Our study reveals two limitations of direct tool use. First, evidence presentation matters. Different renderings of the same fixed mask can substantially change model behavior. Second, masks introduce an evidence trade-off. Highlighting a relevant region can expose diagnostic evidence the model previously missed, but the added visual cue can also induce incorrect associations. As a result, even a ground-truth mask can correct some answers while turning initially correct ones into incorrect ones. Together, these findings show that acquiring the right mask is not enough. An agent must also decide how to present the evidence and whether it should change the answer. We therefore introduce TUNE (Tool Use is Not Enough), an evidence-management harness that explicitly manages what evidence to acquire, how to present it, and whether to accept the resulting answer update. Across three public medical VQA benchmarks, evidence management improves the fine-tuned Qwen3.5-9B backbone's mean accuracy by 4.0 percentage points. Full TUNE improves over direct tool use by 9.1 points on average across seven benchmarks and achieves competitive results against existing medical VLMs and agents. 

---
# Collision-Free Is Not Enough: Passage-Conditioned Navigation for Wheeled-Legged Robots 

**Authors**: Wai Ka Sun, Waiyiu Pong, Bowen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25788)  

**Abstract**: A collision-free detour can still bring a legged robot to the next stair or ledge a few centimetres or degrees off, and a fixed locomotion skill, factory or learned, can stall or fall. Existing methods prevent this with a robot model or extensive trial and error, but on a new site with unknown skills neither is available, so they cannot be used. We find that a single 27-minute teaching session already supplies what these methods have to derive or learn: for every stair and ledge, one crossing the skill has completed. Our method, passage-conditioned navigation, makes the route remember how each transition was entered, not only where the path went. It stores each taught crossing as an event and, without training or a robot model, enforces it after every local replan: before each ledge, an entry gate makes the robot turn or back up and retry if it is more than 10 cm or 10° off the taught crossing. Under localization noise, with path, gait schedule, and sensing unchanged, this raises the share of disturbed ledge crossings that stay within these limits from 24.1% to 88.6%. In contact simulation with our learned policies, a robot pushed off course before stairs never falls in 180 climbs, against 9 falls when the stored direction is ignored; pushed 1.2 m away, it rejoins the route, ending on average 0.14 m from it (1.10 m without rejoining), and finishes as often as an undisturbed robot. This suits any robot with fixed skills that repeats a route over terrain transitions, as in inspection and patrol, where one wrong entry can end the task. We deployed it in competition on a real wheeled-legged robot: over 271 m of multi-obstacle terrain spanning 7.0 m of elevation, it completed all six runs, won third place at the 2026 Global Open-source AI Challenge, which drew 2,999 submissions, and was the only finalist in its task to finish fully autonomously. 

---
# Closing Performance Gap for Diffusion Language Models under Low Inference Budgets 

**Authors**: Hao Wu, Sipeng Chen, Jingwen Fu, Yuwang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25763)  

**Abstract**: Diffusion language models offer an architectural route to low-budget generation by updating multiple tokens in parallel, which can reduce the number of sequential decoding steps relative to token-by-token autoregressive decoding. However, realizing this advantage at low budgets is challenging, as generation quality often declines when the number of denoising and refinement steps is reduced. Editable diffusion models address this quality loss through post-draft refinement, but useful intermediate results may still be lost before final submission. We introduce CaRR (Commitment-aware Refinement and Recovery), a training-free framework which improves submission quality from a fixed editable checkpoint through updated-context refill and trajectory-based recovery within its existing forward schedule. Token Commitment Remasking (TCR) remasks selected native pure replacements and refills them on the next scheduled edit forward using the updated context. Submission Commitment Recovery (SCR) uses the resulting refinement trajectory to recover supported candidates when the final response does not satisfy task-specific output requirements. Both modules operate within the native mask/edit schedule without additional backbone forwards. Experiments on code generation and mathematical reasoning show consistent gains in average accuracy over native ME-DLM across four inference budgets, with the largest gains under tight budgets. At one-eighth of the full scheduled forward budget, CaRR improves the six-metric average accuracy by 7.44 pp, including gains of 11.37 and 10.05 on MBPP and MBPP+. 

---
# Characterizing and Detecting Unfaithful Work Reports by Coding Agents 

**Authors**: Wenhao Yang, Runzhi He, Minghui Zhou, Pengfei Gao, Dayi Lin, Qingwei Lin, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25760)  

**Abstract**: Coding agents increasingly carry out long-horizon development tasks, autonomously planning and executing multi-step actions that span files, tools, and sessions. Because developers rarely follow each action, and agents resuming the work depend on records of what was done, the agent's work report often becomes the primary account of what was changed and established. When the report contains false or unsupported claims, it gives recipients false assurance: incomplete work may be accepted without verification, incorrect assumptions about the current state can propagate to later steps and agents, and evaluators that credit stated success may reward the report rather than the work. Yet little is known about what unfaithful reporting looks like in real development work, or how developers and agents handle it once it surfaces. Starting from real developer--agent conversations in SWE-chat, we conduct a systematic mixed-methods analysis of self-admitted unfaithful reports. We identify eleven subtypes in four categories: misreported observations, misattributed evidence, overstated task success, and faulty technical explanations, with overstated task success the most common. Handling these problems involves on average 4.3 further user turns and 12.71 minutes of recorded response and execution time, and users express dissatisfaction more often. Among 435 work reports that we separately sample and annotate, \NaturalPositiveShare{}\% contain at least one unfaithful claim. We also develop FaithDetector, a detector for unfaithful reporting informed by the observed patterns. On naturally sampled reports, FaithDetector attains \NaturalDetectorMacro{}\% macro-F1, about 13 percentage points above the baseline. Together, our findings and FaithDetector provide empirical evidence and an automated method for checking reporting faithfulness, informing agent evaluation and training feedback. 

---
# SAM 3 Sees More Than It Detects: Promptable Concept Maps Beyond Categories 

**Authors**: Yu-Huan Wu, Yun Liu, Liangli Zhen  

**Link**: [PDF](https://arxiv.org/pdf/2611.25747)  

**Abstract**: Promptable segmenters answer what a thing is. Dense prediction tasks also ask class-agnostic questions, such as what stands out or hides, how light is blocked, transmitted or reflected, and where structure lies. Robots, surveillance systems and image editors build on their answers. We study promptable concept maps (PCM), which extend SAM 3's promptable concept segmentation from noun instances to maps of concepts named by short noun phrases such as `"transparent object", with "none" as an answer. PCM-Bench groups the standard test sets of seven such tasks by the question they ask. On it, we find that SAM 3 sees more than it detects. Its semantic path responds to many of these concepts, yet a low image-level presence score can block every detection of its instance path. We present UNVEIL, which trains a copy of SAM 3's semantic path for these maps, with LoRA on copies of its last blocks. A separate presence decoder learns from verified negatives whether a concept occurs. With one text phrase per query, UNVEIL maps region concepts such as salient and camouflaged objects, shadows, transparent objects and mirrors. The same weights map thin structures such as edges and skeletons as well as object categories. Run next to SAM 3, it leaves SAM 3's own outputs unchanged. On each of the seven tasks, UNVEIL reaches the highest score among the compared promptable models, including SAM 3's official fine-tuning and pipelines built on multimodal LLMs. Code is available at this https URL. 

---
# An LLM-Driven Traffic-Control xApp for Dynamic 5G RAN Intents 

**Authors**: Valerio Triolo, Luca Valcarenghi, Emilio Paolini  

**Link**: [PDF](https://arxiv.org/pdf/2611.25736)  

**Abstract**: Programmable Radio Access Networks (RANs) enable runtime deployment of traffic-control functions, but selecting the appropriate configuration remains challenging as network conditions and service requirements change over time. Conventional solutions often rely on static rules or learning-based controllers with predefined state and action spaces, limiting their ability to accommodate changing intents. This paper introduces an LLM-driven xApp for traffic control in programmable RANs. The proposed architecture combines operator-defined service intents, runtime E2 interface telemetry, and the currently deployed configuration to generate traffic-management policies without blocking packet forwarding or the processing of incoming E2 measurements during LLM inference. Experiments on an OpenAirInterface 5G environment show that the proposed controller dynamically adapts the traffic-control pipeline to runtime intent changes, achieving an average intent satisfaction ratio of 0.91, compared with 0.52 for a static TC-RAN configuration. When channel conditions degrade, the controller maintains an intent satisfaction ratio of approx. 0.88, compared with approx. 0.4 for the static baseline. 

---
# PosteriorBench: Evaluating Behavior Posterior Sampling in Language Models 

**Authors**: Zhiyang Xun, Daniel D. Johnson, Sarah Schwettmann, Neil Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2611.25735)  

**Abstract**: Rare language model behaviors can be difficult to study because obtaining them through ordinary sampling can require a very large number of samples. This motivates the behavior posterior sampling problem: given a language model and a behavior of interest, is there an algorithm that efficiently samples from the model's distribution over transcripts that exhibit this behavior? We introduce PosteriorBench, a benchmark for comparing algorithms by their compute cost and by how close their samples are to the true posterior in forward and reverse KL divergence. PosteriorBench contains 21 tasks on Qwen3.5-0.8B with behavior probabilities from $10^{-5}$ to $10^{-1}$, each with exact posterior samples from a corpus of one billion labeled responses. We find that no method we evaluate is both efficient and faithful: simple methods like prompting are cheap but plateau at a high KL divergence from the posterior, fine-tuning on posterior samples is often cost-prohibitive, and several training methods from prior work do not perform better. Building a posterior sampler that can cheaply approximate the true posterior remains an open problem, and we hope PosteriorBench can measure progress towards this objective. Code and data are available at this https URL. 

---
# From Reliance to Overreliance: Taxonomizing and Detecting Relational Risks in AI Companionship 

**Authors**: Viet Cuong Nguyen, Fiona Li, Chang Wan, Sheyril Agarwal, Nhi Nguyen, Angel Hsing-Chi Hwang, Munmun De Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2611.25734)  

**Abstract**: Companion AIs are increasingly used for emotional support, yet distinguishing meaningful engagement from harmful reliance remains difficult. We introduce the Taxonomy of Relational Risks (TRR), which operationalizes users' emotional overreliance (EO) as observable conversational behaviors and separately identifies evidence of relational harm (RH), enabling empirical examination of when reliance behaviors co-occur with adverse consequences. Using naturalistic this http URL and Replika conversations from 110 adults, we found EO-related behaviors in 63.2% of non-roleplay conversations, while observable relational harms appeared in 10.5%. Although EO did not necessarily lead to harm, recurring patterns often co-occurred with harmful outcomes, suggesting value as risk indicators. Open-source LLMs satisfactorily detected 12 of 14 EO behaviors in at least one configuration. Qualitative analysis revealed expectations of care, repeated support seeking, dependency, distress, and reduced confidence in personal judgment. We argue for graduated safety interventions tailored to patterns and trajectories of EO-related behavior. 

---
# Auto-Alpha: A Framework for Autonomous Factor Research 

**Authors**: Hui Cai, Kangcheng Deng, Tianyi Ma, Qixuan Yang, Jiacheng Lu, Rui Sun, Beidi Luan, Jing Li, Zuo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2611.25732)  

**Abstract**: Autonomous factor research requires generating economic hypotheses as well as implementing and testing them. We present Auto-Alpha, a framework that connects report-guided initialization to self-directed research through a test-driven implementation loop. An idea agent proposes economic mechanisms and signal constructions; a research agent develops them with backtest feedback. Semantic audit, quantitative review, and human admission select factors for a persistent library, while an Experience Library retains research insights. Starting from 120 research reports, Auto-Alpha builds an initial library of 32 factors. It then generates new ideas and refines existing candidates, producing 420 implementations and retaining 62 additional factors. We evaluate factor quality and portfolio performance using a fixed library snapshot of 105 factors. The median 1-day Rank IC is 0.0301, and a weekly Top-300 portfolio returns 16.81% annualized after stated costs over 2020-2026, versus 1.71% for the CSI 1000 price index. These retrospective results document an operating research process and its artifacts; controlled evaluation is needed to isolate the contribution of its components. 

---
# Verifying Before Enhancing: Target Evidence Chain Verification for Infrared Small Target Detection 

**Authors**: Jiayi Zuo, Songwei Pei, Shuhuai Wang, Yifei Wu, Shangguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25724)  

**Abstract**: Infrared small target detection is essential for early warning, surveillance, remote sensing, and space-based target perception. However, infrared small targets are typically tiny, weakly textured, and boundary-blurred, making them difficult to distinguish from complex backgrounds. Moreover, target-like background structures can produce salient responses across multiple scales, making response strength alone insufficient for reliable target discrimination. To address this challenge, we propose VBE-UNet, a Target Evidence Chain Verification Network that assesses candidate reliability beyond response magnitude through complementary evidence cues. Specifically, counterfactual background explanation models local target evidence, while joint evidence verification integrates cross-scale support and perturbation-based contribution verification to guide feature recomposition. The resulting reliability estimates guide selective feature refinement, promoting target-relevant responses while reducing unsupported background activations. Extensive experiments on IRSTD-1K, NUDT-SIRST, and NUAA-SIRST demonstrate that VBE-UNet achieves superior detection accuracy and lower false-alarm rates compared with state-of-the-art methods. Ablation studies and evidence reliability analysis further validate the effectiveness of the proposed verification framework. 

---
# SteerTrace: Tracing How Target Attacks Steer World Action Models 

**Authors**: Yukun Dai, Tianshi Wang, Fengling Li, Jingjing Li, Xin Li, Lei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25719)  

**Abstract**: World action models (WAMs) predict future videos and actions with one backbone, and their imagined future is often seen as a safety signal. We introduce SteerTrace, a targeted universal attack on WAMs that also traces its effect inside the model. One $\ell_\infty$-bounded (4/255) perturbation per model, shared across frames, tasks, and camera views, pushes the executed action toward an attacker-chosen target. On eight WAMs from four families (FastWAM, Faster-WAM, OpenWAM and LingBot-VA) on LIBERO and RoboTwin, it drives task success close to zero and moves the attacked action toward the target. Same-budget noise and BadWAM produce little action shift in the target direction. We use the signed targeted effect to measure how much of the effect each intervention removes or reproduces. Under the standard attack, restoring future tokens removes 61-94% of the targeted effect in the five models whose action experts attend to them. When the attack is constrained to keep the predicted future close to the clean prediction, the targeted effect can become more dependent on current-frame tokens. In three of these five models, restoring current-frame tokens removes 87-99% of the effect, compared with 9-15% for future tokens. In six models, restoring just 0.36% of the FFN neurons in layers 10-29 of the video expert removes 72-96% of the targeted effect. Residual directions 2-16 hold only 11-18% of the shift energy, yet removing them cancels most of the targeted effect. 

---
# STARK: Moving towards Personalized Agents That Truly Learn and Remember 

**Authors**: Aryan Jain, Abhishek Jha, Anish Nediyanchath, Promod Yenigalla  

**Link**: [PDF](https://arxiv.org/pdf/2611.25705)  

**Abstract**: Long-term memory is essential for personalized agentic systems that interact with users over long time periods, yet current architectures only capture what happened while leaving the reasoning of why it happened to be derived later by underlying LLMs.%, and silo knowledge within individual users. These events can, however, convey not only what a user experienced but also how that experience shaped their behavioral patterns. We introduce STARK, a dual-graph memory architecture which treats these events not just as nodes but also as edges between behavioral insights. STARK maintains an Episodic Graph which records interactions as temporally, causally, and semantically linked episodes; and an Insight Graph which represents LLM-extracted behavioral signals scored by how many events support them, connected by the very events that led to them. At inference, behavioral knowledge is retrieved directly from insight nodes grounded by event-as-edges, eliminating repeated episodic traversal. The event-as-edge structure also naturally enables projecting the knowledge graph into a relational embedding space where cross-user transfer becomes possible under shared contextual circumstances, allowing the system to infer missing insights for cold-start users. Evaluations show that STARK improves by 1.48 judge score on LongMemEval information extraction and 0.93 on LoCoMo open domain questions over state of the art. On the public LongLaMP abstract-generation benchmark, cross-user transfer improves open-ended judge scores by 0.7 points and ROUGE-L by 3 at 5% graph sparsity. 

---
# From Geographic Memory to Spatial Generalization in Large Language Models 

**Authors**: Chen Chu, Zhaonan Wang, Cyrus Shahabi, Hengcai Zhang, Feng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25703)  

**Abstract**: Large language models (LLMs) can associate places with geographic descriptions, but recovering an observed association does not establish an ability to generalize across geographic space. We distinguish geospatial memory, the recovery of location--description associations observed during training, from spatial generalization, the prediction of descriptions for locations in unobserved regions. We investigate whether continuous location representations improve this generalization and encourage geographic knowledge to become spatially organized within an LLM. We frame this organization as a latent spatial field: a learned mapping from locations to descriptions connected through spatial dependencies. Building on a continuous spatial encoder, we project location representations into the model's embedding space and supply them as spatial tokens. We evaluate the resulting models on two newly curated geo-text benchmarks for Los Angeles and London using two complementary assessments. Geospatial captioning tests both memory and spatial generalization by generating descriptions at observed locations and at locations beyond the spatial coverage of the training observations. A separate geospatial grounding model tests whether generated location--description pairs provide useful additional supervision for mapping text to geographic space. Across multiple LLM families, continuous location representations improve captioning at unobserved locations relative to textual coordinates, while their generated descriptions yield modest but consistent improvements in downstream grounding. Layer-wise probing and representation clustering provide additional evidence consistent with spatially organized geographic representations. Together, these findings show the value of evaluating spatial generalization separately from memory and of incorporating continuous spatial representations into language models. 

---
# TipDex-VTLA: Learning When to Refine Actions with Touch for Dexterous Manipulation 

**Authors**: Xudi Ge, Yepeng Liu, Haifeng Liu, Enze Shi, Liangqi Zhang, Yongchao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25683)  

**Abstract**: Changes in illumination can induce action drift in dexterous manipulation even when the physical state remains unchanged. Although existing vision--tactile--language--action (VTLA) policies incorporate tactile feedback, when and how touch refines actions under changing visual conditions remains challenging. We introduce \textbf{TipDex-VTLA}, a VTLA policy that combines estimated visual action reliability and tactile forecast verification to regulate execution-time refinement. Specifically, Visual Action Sensitivity Estimation (VASE) learns visual reliability from a frozen reference policy's action sensitivity to illumination perturbations. Tactile Anticipation and Correction during Execution (TACE) compares predicted and measured touch at matched times and combines the resulting discrepancy with visual reliability to gate bounded feature corrections for unexecuted actions. This effectively helps to refine actions under challenging unseen visual conditions. We evaluate TipDex-VTLA on four real-world dexterous manipulation tasks and the UniVTAC simulation benchmark. TipDex-VTLA trained on data collected under normal visual conditions, achieves average full-task success rates of 87.5% (resp. 73.8%) under normal (resp. challenging unseen) lighting in real-world tasks, significantly outperforming some state-of-the-art methods. We will release the collected data and source code. 

---
# Beyond Entity Recognition: Vision-Centric Knowledge Visual Question Answering 

**Authors**: Jiatong Ma, Yuchen Liu, Longteng Guo, Yichen Yuan, Yuze Wu, Dongze Hao, Ru Zhen, Yanhao Zhang, Haonan Lu, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25680)  

**Abstract**: We propose VC-VQA, a large-scale benchmark for vision-centric knowledge reasoning, aiming to systematically evaluate the capabilities of Multimodal Large Language Models (MLLMs) in fine-grained visual knowledge acquisition and complex multi-stage reasoning. Unlike existing benchmarks that primarily treat visual information as entity recognition and rely on text-based external knowledge, VC-VQA explicitly models the role of visual information by considering both its source (internal vs. external images) and stage of involvement (pre-retrieval vs. post-retrieval). Specifically, VC-VQA requires models to retrieve external visual evidence, identify visual clues before retrieval, and localize, verify, or match retrieved knowledge against the internal image. We further construct a controlled multimodal knowledge base with traceable textual and visual evidence for each sample. We evaluate 21 representative MLLMs under four settings: no external knowledge, with provided evidence, knowledge base retrieval, and open-web search. Experimental results lead to three key findings: current models still struggle with vision-centric knowledge reasoning; fine-grained visual knowledge acquisition remains the primary bottleneck; and agentic retrieval highlights the value of multi-step planning and cross-modal knowledge integration. Overall, VC-VQA provides a more challenging benchmark for evaluating vision-centric knowledge reasoning in MLLMs. Our code and dataset are available at this https URL. 

---
# GRODE-RTL: Source-Mapped Dependency Views for LLM-Based RTL Repair 

**Authors**: Jingyi Zhou, Yi Liu, Bin Sun, Hongji Zhang, Dongsheng Zuo, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25661)  

**Abstract**: Large Language Models show great potential for automated Register-Transfer Level repair, yet they struggle with non-local faults where root causes and failing outputs reside far apart in source code. Treating RTL as a flat token sequence obscures hardware dataflows, whereas exposing complete synthesized graphs overwhelms the model with fine-grained metadata. To resolve this representation dilemma, we propose GRODE-RTL, an anchor-centric representation that extracts the single-clock-stage fan-in cone of a suspect register from the netlist, maps netlist nodes back to editable source lines, and organizes them into a compact hop-partitioned dependency view. By prioritizing direct drivers and folding distant hops, GRODE-RTL guides standard tool-using agents without exposing ground-truth hints or oracle feedback. We evaluate GRODE-RTL across 30 formally validated non-local faults using three LLMs (DeepSeek-V4-Flash, GLM-5.2, and MiniMax-M3) alongside a controlled six-configuration ablation study. GRODE-RTL improves pass@1 over source-level exploration by 16.7% on DeepSeek, by 26.7% on GLM-5.2, and by 20.0% on MiniMax-M3, with consistent gains on the hard-to-retrieve far-distance and indirect-anchor subsets of all three models. Our controls isolate structural relevance from generic context expansion, proving that selective dependency organization rather than full graph exposure is key to effective RTL repair. 

---
# A$^2$-VLA: Adapting Vision-Language-Action Policies to New Tools through Agentic Affordance Discovery 

**Authors**: Chenyuan Zhou, Chaoyi Huang, Zibo Wang, Zhiyuan Li, Yong-Lu Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25656)  

**Abstract**: Tools with different shapes can have similar functions, but transferring manipulation skills across tools requires understanding how tool geometry relates to the specific task and adjusting the manipulation strategy accordingly. The key challenge is to discover the common affordances that contribute to task completion across different tools and verify whether they support new tools in performing similar work. We present A$^2$-VLA, a tool-adaptation framework that combines an embodied agent with a VLA policy. It treats tool affordances as hypotheses the agent can construct, test, and revise. Given teleoperated demonstrations with a few source tools, the agent reconstructs the scenes in simulation and autonomously determines task-relevant affordance features: functional regions and their key geometric properties. It develops affordance-extraction skills and refines both the features and skills through autonomous experiments in the reconstructed scenes. Validated extraction skills accumulate in a reusable skill library for unseen tools. Finally, A$^2$-VLA uses encoded functional-region features as conditioning inputs to a $\pi_{0.5}$ backbone during training and inference. Evaluation examines affordance reliability in reconstructed scenes and real-robot task success on unseen tools. Our method achieves 86.1% success in simulated cross-tool trajectory retargeting based on affordances. On three real-robot tool-use tasks, A$^2$-VLA improves task success by 30.7 percentage points over $\pi_{0.5}$. 

---
# DARSY: A Tool for Data-Driven Abstraction and Policy Synthesis for Stochastic Systems with Unknown Dynamics 

**Authors**: Thom Badings, Ibon Gracia, Mahdi Nazeri, Morteza Lahijanian, Alessandro Abate  

**Link**: [PDF](https://arxiv.org/pdf/2611.25654)  

**Abstract**: This paper presents DARSY: a software tool for abstraction-based verification and policy synthesis in discrete-time stochastic control systems with unknown dynamics and noise distributions. DARSY requires only a simulator of the system and a Lipschitz bound on its dynamics, and needs no knowledge of dynamics or noise distributions. The tool is fully automated and constructs a finite interval Markov decision process (IMDP) abstraction from sampled transitions, whose probability intervals are obtained via statistical techniques and Lipschitz reasoning. Robust dynamic programming on this IMDP then yields a policy together with a lower bound on the probability that the unknown system satisfies a desired specification. This lower bound holds with a user-specified confidence; that is, it is probably approximately correct (PAC). The tool is implemented in Python and supports general Lipschitz-continuous system dynamics and specifications expressed in syntactically co-safe LTL (scLTL), which includes reach-avoid and finite-horizon safety specifications as special cases. Beyond standard IMDPs, DARSY implements a two-layer IMDP (2IMDP) abstraction that also bounds the probability of transitioning to aggregated state blocks, yielding smaller abstraction errors than IMDP abstractions and, as a result, tighter satisfaction guarantees. We demonstrate DARSY's broad applicability across a variety of benchmarks and specifications. 

---
# Erasing the Wrong Thing: Conditional Attribute Leakage and Concept Erasure in Shared Foundation-Model Embeddings 

**Authors**: Haijun He, Fei Li, Donghong Ji, Junjun He  

**Link**: [PDF](https://arxiv.org/pdf/2611.25640)  

**Abstract**: Medical-image embeddings are increasingly shared in place of raw images for authorized downstream tasks, yet they may reveal sensitive attributes. Standard concept-erasure methods target population-level guardedness. However, we prove that, when binary task and sensitive labels are associated, global and task-conditional linear guardedness cannot both hold while a nonzero linear task signal is preserved, so global erasure must either eliminate that signal or retain task-conditional linear leakage. We further show how global erasure affects linear readers' discrimination between positive and negative samples with different sensitive attributes, deriving an approximate closed-form prediction of its utility cost from pre-erasure statistics. Motivated by this conflict, we introduce Task-Conditional LEACE (TC-LEACE), a closed-form variant of LEACE that enforces linear guardedness within each task stratum of its fitting data. We evaluate four vision foundation models on 85 CelebA attribute pairs spanning a broad range of associations. After LEACE, conditional leakage persists across all four encoders, the discrepancy between marginal and conditional audits increases with association strength, and predicted utility costs track observed losses. We then examine hospital chest radiographs using three foundation models, including two biomedical models, in which chest-tube status is strongly associated with diagnosis. Although a marginal attack on LEACE-erased embeddings underperforms a diagnosis-only reference, chest-tube status remains detectable within diagnosis strata. TC-LEACE recovers most of LEACE's linear task-utility loss while keeping task-conditional leakage within a prespecified margin of LEACE's. 

---
# Beyond Easy Clients: Difficulty- and Improvement-Aware Aggregation for Federated Reinforcement Learning in Robot Navigation 

**Authors**: Ye Lin Tun, Chu Myaet Thwal, Eui-Nam Huh, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2611.25637)  

**Abstract**: Federated reinforcement learning enables a fleet of robots to jointly learn a shared navigation policy without exchanging raw experience. In practice, however, the robots rarely operate in the same conditions: an open aisle is far easier to navigate than a cluttered room, and takes less learning to master. Standard federated aggregation rules ignore this. Uniform averaging weights every client equally, whether or not it is still learning, and weighting by episode return is worse, since easy environments permit high returns by construction. Both failures stem from what we call the \emph{plateau problem}: once a client masters its environment, its updates carry little new information even though its reported return remains high. We propose \emph{FedDIA}, which scores each client by how much its local training improves the global model it received, amplified by the difficulty of its environment. Difficulty is inferred from the model's own performance, without requiring any description of the environment. Both signals are measured against the global model, so they are comparable across clients. On a mapless navigation task with eight heterogeneous clients, FedDIA reaches a mean success rate of $75.5\%$ across training and held-out environments, against $68.7\%$ for the strongest federated baseline. Its advantage holds when the models are transferred to hand-authored warehouse and living-space layouts, or when one of the clients is defective. 

---
# AesDistill: Distilling Generative Aesthetics for Image Color Enhancement 

**Authors**: Chuanzhi Xu, Chengkun Yue, Huiqi Liang, Haoyang Li, Zihan Deng, Qingwen Zeng, Pengfei Ye, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2611.25635)  

**Abstract**: Image color enhancement aims to improve visual appeal, but previous methods often learn to reproduce ground-truth expert targets. These targets reflect the preferences of a few experts and cover a limited range of aesthetic choices. Matching them does not ensure stronger aesthetic quality across different tastes and scenes. By contrast, large generative models learn rich visual knowledge from large image collections and often produce vivid, appealing image edits. However, direct generation can change the source image's semantic structure along with its colors. In this paper, we introduce AesDistill, a framework for distilling generative aesthetics into image color enhancement. We use 4 leading open-source large generative models to produce 64,644 edited images as teacher targets at a cost of about 3,360 H100 GPU-hours, forming the AesDistill Dataset. AesDistill also includes AesDistill Student, a 1.13M-parameter model that learns teacher color changes through confidence-weighted supervision while preserving the source image's semantic structure. We evaluate AesDistill with 12 aesthetic and 4 image-quality assessment models. Every teacher's targets rank above the FiveK and PPR10K expert target means in the joint assessment. Across both datasets, AesDistill Student outperforms all 20 baseline methods and ranks directly behind the 4 teachers. These results demonstrate that distilling aesthetics from generative models enables a compact student to outperform methods trained on expert targets. 

---
# DAWN: Robust Neural Blind Deconvolution via Alternating Wiener Guidance 

**Authors**: Yida Pan, Yixuan Jia, Chaoyan Huang, Qing Qu, Ismail Alkhouri  

**Link**: [PDF](https://arxiv.org/pdf/2611.25621)  

**Abstract**: Blind image deconvolution (BID) is an ill-posed inverse problem that aims to jointly recover a sharp image and blur kernel from a single degraded observation. Under nontrivial and high noise, which is common in photography and scientific imaging such as microscopy, BID remains particularly challenging, as existing methods suffer from noise overfitting and inaccurate kernel estimation. Existing training-data-free learning methods for BID, such as those based on the deep image prior (DIP), often utilize external guidance by including model-based estimates as additional loss terms, which underexploits the information these estimates carry especially under nontrivial noise. To address these challenges, we propose DAWN, a training-data-free DIP framework that alternates between injecting model-based estimates directly into the model input and refining this imperfect estimate with the network's own prior. To provide more reliable kernel estimates for these injections, we apply autoencoding regularization to both the image and kernel networks. On natural and scientific images under varying levels of Gaussian and Poisson noise, DAWN outperforms state-of-the-art methods in both image and kernel quality (achieving up to +4.5 dB over the next best baseline on the Levin dataset at $\sigma = 0.1$). Without any training images, DAWN remains highly competitive with a state-of-the-art diffusion model on FFHQ, despite the latter being trained on a massive number of in-distribution images. 

---
# Splice-Core: Low-Order-Bit-Splicing GEMM for Highly Energy-Efficient AI Inference Accelerators 

**Authors**: Prabhu Vellaisamy, Hsien-Fu Hung, Andrew Kim, Shawn Blanton, John Paul Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.25617)  

**Abstract**: Splice-Core is a temporal-unary/binary (tub) hybrid general matrix multiplication (GEMM) unit that minimizes low-order accumulation latency, while retaining value-dependent execution for high-order activation bits. We present two novel processing element (PE) variants: (i) Splice-BP utilizing bit planes, and (ii) Splice-LUT, which selects a shared weight multiple in a single cycle. Compared to a binary systolic array, Splice-BP/Splice-LUT achieve 77.4%/76.4% geometric-mean tile energy reductions, respectively, across 26 INT4 convolutional neural network (CNN) and large language model (LLM) workloads in 45nm. Splice-LUT further increases geometric-mean tile throughput by 44.4% and reduces geometric-mean energy-delay product (EDP) by 6.13x across the 26 configurations. Across the evaluated W8A8 convolution and expert-projection workloads, Splice-BP and Splice-LUT reduce tile energy by 77.3-88.1%. Post-place-and-route INT4 yields 14%/9% smaller area for Splice-BP/Splice-LUT, compared to the systolic array. 

---
# Copilot or Culprit? How AI Is Changing Open-Source Bug Reporting and Maintenance 

**Authors**: Yinuo Liu, Yunlong Lyu, Zichao Luo, Tian Dong, Zian Luo, Peng Chen, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.25605)  

**Abstract**: AI-generated bug reports have drawn criticism for unreliable claims and faulty reproducers that burden maintainers. However, as LLM capabilities improve, some maintainers have begun to express more positive views on AI use in bug reporting and maintenance. Understanding this evolution requires examining whether AI-involved reports are becoming more reliable, which uses of AI are associated with valid findings, and how maintainers investigate these reports and act on them. To this end, we conduct the first large-scale study of 120,918 bug reports from 101 open-source projects between January 2024 and September 2026, tracing AI involvement from discovery and reporting through validation and resolution. Our findings show that AI-involved reports become more complete over time, with a growing share of reports confirmed as valid. In bug reporting, AI for reproduction and diagnosis are associated with higher validity, whereas report writing is not. Extends into maintenance, maintainers increasingly use AI to investigate code and recommend fixes. However, they devote more effort to AI-assisted reports of human-discovered bugs than to reports of AI-discovered bugs. These findings highlight both AI's growing participation in bug investigation and the continuing need for verification and follow-up. Drawing on these findings, we provide practical guidance for turning AI-generated findings into verifiable reports: emphasizing reproducibility, verification, and accountability. We followed this guidance when submitting 53 reports on AI-discovered bugs, of which 69.81% were accepted by maintainers as of writing. 

---
# On the Informativeness of Structural Stability Bounds for GNNs: GINs versus GATs 

**Authors**: Alan Gany, Paul Landrier, Bogdan Cautis, Laks V. S. Lakshmanan, Silviu Maniu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25603)  

**Abstract**: Stability bounds for graph neural networks limit how much node representations can drift when the graph changes, but they are rarely compared with the simplest alternative: a bound on the distance between any two outputs, whatever the graph. We derive structural stability bounds for Graph Isomorphism Networks (GINs) and single-head GATv1 networks under deterministic edge perturbations that need not be small, and compare each with such a perturbation-independent diameter bound. For GINs, we characterize exactly when the stability bound is tighter, through a condition linking perturbation size, depth and maximum degree; it holds with a wide margin on Cora, but fails beyond a few layers on the low-degree MUTAG graphs. For GATv1, the bound is never tighter, which calls for analyses based on attention masses. 

---
# Tether: Verified Stateful Policy Engine for System-Level Monitoring of Agents 

**Authors**: Meghana Aparna Sistla, Alaia Solko-Breslin, Somesh Jha, Aditya Kanade, Krishnamurthy Dj Dvijotham  

**Link**: [PDF](https://arxiv.org/pdf/2611.25599)  

**Abstract**: Autonomous AI agents are increasingly deployed for long-running tasks that execute arbitrary shell commands or programs. However, these actions can result in unauthorized network requests or persistent, malicious changes to the user's file system, processes, or privileges. Existing defense mechanisms rely on either LLM-based monitors, which lack provable safety guarantees, or stateless policy enforcement layers, which miss multi-step data flows and execution side-effects unmodeled by the policy language (e.g., network calls initiated inside a Python script). We introduce Tether, a framework for runtime enforcement of stateful security policies over the observed side-effects of an agent's actions. Tether executes each proposed action in an ephemeral, isolated sandbox, inferring abstract action semantics from observed system calls. Tether then enforces stateful policies over these inferred semantics, permitting actions to proceed only if they comply with the policy. Policies and action semantics are defined in an embedded DSL in Lean 4 capable of expressing diverse constraints, including information flow control (IFC), access control, and network security policies. Because actions can propagate labels from sources to destinations, a single rule can enforce fine-grained IFC policies across multi-step execution flows of arbitrary length. Expressing policies in Lean enables formal verification that they satisfy global security invariants (e.g., preventing network access after reading a sensitive file). We evaluate Tether on realistic coding-agent benchmarks across both benign and adversarial trajectories. With a single policy shared across all tasks, Tether blocks 100% of in-scope attacks while maintaining a benign false-positive rate under 5%. 

---
# Decision-Focused Reinforcement Learning for LLM Post-Training 

**Authors**: Haotian Zhai, Leqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25597)  

**Abstract**: Large language models (LLMs) have shown promise as predictors across various tasks, benefiting from their ability to use information from unstructured context. However, when predictions serve as parameters of downstream optimization problems, higher prediction accuracy does not necessarily translate into lower decision costs. We propose decision-focused reinforcement learning (DFRL), a framework for post-training LLMs to predict optimization parameters using negative downstream decision regret as reward. DFRL trains LLM predictors to minimize the costs of the decisions induced by their predictions, without requiring downstream convexity or differentiability, and integrates readily into existing RL post-training frameworks. To address expensive reward evaluation, we develop two approaches. Heuristic warm-starting uses inexpensive approximate rewards during an initial phase before switching to refinement with exact rewards. Reward correction combines inexpensive estimates with randomized exact evaluations, preserving the expected policy gradient while reducing the expected number of exact solves per update. Experiments on three synthetic optimization problems demonstrate improvements over prediction-loss baselines, while a larger knapsack experiment shows faster early improvement under short training budgets. On inventory planning with H&M transactions and product descriptions, our best configuration, in which the LLM reasons before predicting, reduces regret by 31.4% relative to the SFT baseline and 12.4% relative to the strongest evaluated conventional forecasting baseline. 

---
# Bradley-Terry Training Can Distort Outcome Probabilities of Initially Accurate Language World Models 

**Authors**: Xinze Shi, Litian Zhang, Zhenjie Su, Zelong Yu, Yuting Zhang, Qihua Feng, Zifeng Kang, Chaozhuo Li, Zhengxuan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25594)  

**Abstract**: Comparative feedback, in which a judge says which of two candidate outcomes is closer to what happened, is a practical way to train language world models that assign probabilities to an action's outcomes. We show that, fitted with a Bradley-Terry objective, it need not teach the true outcome probabilities and can distort accurate ones. On a simulated robot push with enumerated outcome distributions, this objective takes a model that is accurate on unseen layouts (hazard error 2.4 percentage points, pp) and moves 33 to 43 pp of its probability onto outcomes that none of a condition's enumerated branches produces. The cause is the objective, not the comparisons: its optimum, computed before training, predicts where the model ends up, and a likelihood that knows the judge's rule learns from the same comparisons without the shift. The shift is robust: it recurs on unseen conditions, under five candidate laws and a three-fold denser enumeration, and about 29 pp remain without the physically impossible events. Its cost is conditional: with those events it inflates hazard probabilities, and a threshold inherited from the pre-feedback model costs about 3 pp of utility per decision on trained and 1 to 2 pp on unseen conditions, through false rejects; without them the utility loss averages below 1 pp over six seeds, though a small regret in action choice remains. Audits of readout, response and decision trace the effect from probabilities to actions. 

---
# An Explainable and Scalable Hierarchical Tensor Framework for Integrating Multi-Omics and Digital Pathomics 

**Authors**: Ahmad Hussein, Ali Anaissi, Mukesh Prasad, Ali Braytee  

**Link**: [PDF](https://arxiv.org/pdf/2611.25584)  

**Abstract**: Background: Precision oncology requires integration of heterogeneous molecular and pathology data, yet many fusion models offer limited biologically grounded interpretability and struggle with missing modalities.
Methods: We present Hierarchical Interpretable Tensor Fusion (HIT-Fuse), integrating six modalities: somatic copy number variation, DNA methylation, RNA sequencing, microRNA expression, RPPA protein abundance, and vision transformer-derived pathomics across four TCGA cohorts (glioma, lung adenocarcinoma, breast, and kidney cancer). Modality-specific autoencoders compress high-dimensional inputs, followed by two-stage CP tensor decomposition. Missing latent modalities are reconstructed using clinical covariates and observed omics to expand cohorts and retrain downstream models. A mathematical traceback mechanism enables global and local xAI attributions.
Results: HIT-Fuse identifies distinct latent subgroups with significant survival stratification in glioma, breast, and kidney cohorts, both with complete and reconstructed modalities. Cohort expansion yields cohort-specific gains, including a 56.4% relative increase in weighted F1 for glioma. Biological validation against DepMap CRISPR screens, miRTarBase and RPPA annotations supports decoded drivers across all four cohorts. Essential gene rates range from 6.2% in glioma to 7.6% in lung cancer; example drivers include ANAPC11, SNRPD1, PLK1 and RPL14. Of decoded miRNAs, 71-92% have miRTarBase regulatory support, while 77-84% of selected genes are covered in DepMap.
Conclusions: HIT-Fuse combines hierarchical tensor fusion, missing-modality reconstruction and traceable attributions to support interpretable multi-omics and digital pathology-based cancer prognosis across heterogeneous cohorts. 

---
# AutoRAGBench: A Benchmark for LLM Agents Automating RAG System Development 

**Authors**: Hang Zhang, Yuzhi Pan, Ziyao Zhong, Chaokun Wang, Shuo Cao, Zeyu Huang, Yue Xue, Fang Niu, Bofan Xie, Guanchen Ge, Zecheng Li, Leqi Zheng, Ziyang Liu, Wei Wang, Miao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25582)  

**Abstract**: Retrieval-augmented generation (RAG) is a well-established paradigm for augmenting large language models (LLMs) with information retrieved from external corpora. However, developing effective RAG systems requires substantial engineering effort and iterative experimentation, as effective design choices vary across tasks and domains. With rapid advances in LLM agent capabilities, we ask how well these agents can autonomously build RAG systems and how effectively they can improve them through iteration. In this paper, we propose AutoRAGBench, the first benchmark to evaluate how well LLM agents develop RAG systems from scratch in an open code space. It comprises nine RAG system development tasks. For each task, agents build RAG systems from scratch without starter code or prescribed strategies, under a six-hour budget on a single H800 GPU. They analyze task data, implement RAG pipelines, and iteratively refine their designs based on experimental feedback. We evaluate nine agent configurations and analyze both the effectiveness of the resulting RAG systems and their development trajectories. GPT-6.1 Sol with Codex CLI achieves the highest average score of 56.1. The composite Human-SOTA baseline averages 44.4. These results show that LLM agents can automate effective RAG system development. When starting from a human-designed system on each of seven tasks, GPT-6.1 Sol with Codex CLI improves the score by 16.6 points on average. 

---
# FlashX: Scaling Robotic Experience Reuse for Embodied Agents with CBA-Based NAND Flash 

**Authors**: Yuseon Choi, Sanghyeon Park, Eunjong Lee, Hanseo Lee, Sangjin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.25580)  

**Abstract**: Embodied agents are emerging as a promising paradigm for complex robotic tasks, leveraging Vision-Language-Action (VLA) and World Action Models (WAMs) to generate actions through Diffusion Transformer (DiT)-based iterative denoising. However, existing models often struggle to generalize to out-of-distribution (OOD) environments. We observe that retrieving successful experiences and using them to warm-start denoising enables agents to better adapt to unfamiliar situations while reducing computational overhead. This creates a new opportunity to improve task accuracy and inference efficiency simultaneously, in contrast to conventional acceleration techniques that trade accuracy for computation. However, realizing this opportunity requires scaling experience banks, introducing substantial storage and search overhead that challenges the capacity, bandwidth, and power constraints of edge systems.
We propose FlashX, a CMOS Directly Bonded to Array (CBA)-based NAND flash architecture for scalable experience reuse in embodied agents. FlashX exploits CBA's high internal bandwidth for parallel pSLC key search, employing interleaved Compute-in-Flash (CIF) engines to sustain sensing throughput while retaining dense TLC latent storage. Proprioception-aware clustering further reduces search overhead and enables incremental experience acquisition without global reclustering. Evaluations across VLA and WAM workloads demonstrate up to 12.24\%p and 7.43\%p improvements in task success under dense and compute-efficient configurations, respectively. FlashX achieves 2.57$\times$ higher search throughput than the strongest prior NAND search architecture under pSLC sensing, with a further 9.7$\times$ search reduction through clustering. This delivers 1.29--3.95$\times$ end-to-end speedup and 1.12--2.34$\times$ energy efficiency improvement over SSD-based retrieval. 

---
# Online Mistake Detection in Instruction-Conditioned Procedural Video 

**Authors**: Amit Feldman, Ilan Shimshoni, Rotem Dror  

**Link**: [PDF](https://arxiv.org/pdf/2611.25567)  

**Abstract**: An assistant monitoring a person performing a procedure is most useful when it can detect mistakes while there is still time to recover. This requires judging actions against task instructions as the video unfolds, identifying the attempted step, locating it in time, and assessing its execution. Existing online detectors do not use the instruction text, while instruction-conditioned methods operate offline. In this paper, we study online, instruction-conditioned mistake detection in egocentric videos. Using only observed video, task instructions, and prior predictions, our model identifies and localizes completed action attempts and classifies them as correct, modified, slip, corrections, or additions; omitted steps are determined after the video ends. We compare a modular pipeline, with separate step identification, temporal localization, and execution verification, to a joint model that predicts all outputs in a single step. To support supervised learning from mistakes, we repartition EgoPER to include erroneous videos in all splits. The approaches achieve similar step-segmentation F1@0.5 (93.6% and 92.1%), while the joint model achieves higher binary mistake-detection F1@0.5 (84.1% vs. 80.7%), with its largest gain on slips (56.4% vs. 37.5%). Component-level analysis suggests that error propagation between stages contributes to the modular approach's lower detection score. On test videos containing mistakes, both designs outperform released AEM and ESTANet checkpoints under our evaluation protocol. 

---
# The Coverage Selection Gap: Ranking, Not Generation, Limits LLM Rubric Construction 

**Authors**: Ali Karkehabadi, Jamshid Hassanpour, Rafael E. Banchs  

**Link**: [PDF](https://arxiv.org/pdf/2611.25557)  

**Abstract**: Task-specific rubrics are increasingly used to evaluate open-ended language-model outputs, yet constructing comprehensive evaluation criteria from complex source documents remains challenging. We study automatic rubric construction in professional mergers and acquisitions (M&A) due diligence, where evaluation criteria must be inferred from task instructions and supporting documents without access to an expert-authored reference rubric. We introduce a multi-axis generation framework that explores complementary structural and free-form views of the source evidence to construct a broad pool of candidate evaluation criteria. To distinguish failures of criterion discovery from failures of candidate selection, we introduce the coverage-selection gap, which measures the difference between reference-free ranking and a reference-informed greedy diagnostic under a fixed selection budget. Developed on 16 tasks and evaluated on four held-out tasks, our framework achieves 0.876 macro recall against senior-authored reference rubrics under an LLM-based criterion-equivalence protocol. Ablation studies further show that introducing new structural search axes improves recall by 6-10 percentage points, whereas increasing generation volume along existing search directions yields at most 1.5 points. These findings suggest that, once a high-coverage candidate pool has been constructed, effective reference-free selection becomes a substantial bottleneck in document-grounded rubric construction. 

---
# Towards Autoregressive Neural Network Parameter Generation 

**Authors**: Bedionita Soro, Bruno Andreis, Song Chong, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25538)  

**Abstract**: Large Language Models (LLMs) have recently established a new paradigm in machine learning, demonstrating exceptional zero-shot generalization and complex reasoning capabilities. Concurrently, neural network parameter generation has emerged as a promising avenue for rapid task adaptation and model synthesis. Existing weight generation techniques, however, struggle to jointly learn the distribution of diverse models, to encode heterogeneous architectures within a single generator, and to generalize out of distribution. We therefore ask whether pretrained parameters can be encoded into a discrete domain so that the zero-shot generalization LLMs exhibit in text generation can be transferred to weight generation. Following this view, we propose an LLM-inspired approach, Instruction-Guided Parameter Generation (IGPG), that formulates network parameter generation as an autoregressive modeling problem. We first quantize pretrained weights with a VQ-VAE, then train a prefix-conditioned autoregressive generator over the resulting tokens. At inference, the generated weight tokens are mapped back to parameter space by the VQ-VAE decoder. We evaluate the framework across in-distribution generation, cross-dataset transfer, and cross-architecture settings. The proposed method achieves competitive performance in in-distribution generation and provides useful parameter initializations for held-out datasets and architectural configurations, improving downstream performance over random initialization after adaptation. These results suggest that discrete autoregressive modeling provides a promising framework for learning and transferring structure across collections of pretrained neural network parameters. 

---
# CoachSI: Self-Improvement of Robot Policies with a Language Coach 

**Authors**: Jiachen Tao, Gengyu Zhang, Haoxuan Wang, Bing Liu, Yan Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.25529)  

**Abstract**: Generalist robot policies execute atomic skills reliably but fail when the same skills are composed into long-horizon tasks. Self-improvement from the policy's own experience does not close this gap: experience collected by the policy alone stays within its own distribution and reproduces what it can already do. We show that what the policy lacks is not a skill but the next step. We present CoachSI, a self-improvement method in which a frozen, off-the-shelf vision-language model (VLM) coaches the policy in language: it watches the robot and names the next step in the format of the policy's atomic instructions, and never outputs an action. Successful coached rollouts are relabeled in hindsight with the original task instruction and used to fine-tune the policy. The fine-tuned policy is then deployed on its own. On states where a World Action Model and a Vision-Language-Action model have stalled, naming the next step raises subgoal completion by 4.6x over the unchanged task instruction, and the frozen VLM supplies it from the same camera images the policy sees. After fine-tuning, unassisted success on long-horizon tasks in unseen scenes rises from 47.4% to 59.3%, 9.8 points above fine-tuning on human demonstrations and 7.3 points above fine-tuning on self-collected successes at an equal data budget. With a coach that sees only camera images, it reaches 56.8%, above both baselines. The gain is obtained with the action backbone frozen: language coaching teaches the policy what should happen next, and the skills it already has do the rest. 

---
# Learning Generative Ranking Policies for Whole-Page Optimization 

**Authors**: Chen Qiu, Zhigang Hua, Qi Xu, Jizhe Zhang, Sha Hua, Ren Mao, Luke Simon  

**Link**: [PDF](https://arxiv.org/pdf/2611.25527)  

**Abstract**: A recommendation page is consumed as a whole: the items on it compete for attention and complement one another, so the best page is not simply the individually best items placed side by side. We study whole-page optimization, which selects and orders the items on a page jointly to maximize the score of the page. The number of possible pages grows combinatorially, so good pages must be generated rather than enumerated. Some generative re-ranking methods use the same page-level training signal for every choice in a sampled page. Others imitate target pages built before training, which remain largely fixed as the generator improves. We address both limitations with one local page-level objective, which repeatedly moves the current generator toward pages with higher score. This objective splits exactly into one training target per position, and each target compares candidates at the same partial page by the scores of the complete pages they lead to. Every position therefore learns to choose the item that is better for the whole page, rather than the item that looks best on its own. We call this procedure whole-page policy improvement. On two public datasets, KuaiRand-1K and RL4RS-SeqSlate, which cover multi-objective item selection and constrained ordering, and on an industrial dataset of logged pages, our method outperforms the evaluated generative re-ranking baselines. On a synthetic benchmark with a known optimal page, it also comes closest to the optimum, showing that its improvements come from better whole-page decisions. 

---
# MATE: Mobile Agent Trustworthy Execution for Safe and Efficient Mobile Control 

**Authors**: Minsuk Jang, Hee-Seon Kim, Hui-Cheol Yu, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.25526)  

**Abstract**: Mobile agents on personal smartphones can affect private data, payments, settings, and the foreground screen. We present MATE (Mobile Agent Trustworthy Execution), an execution-layer system that controls the boundary between agent action proposals and device effects without modifying the underlying model. MATE represents each UI primitive as a structured action record that combines the instruction, target app, screen context, primitive, and action fields. Runtime Guardrail (RG) maps this record to an allow, block, or user-approval decision; Predefined Task Routing (PTR) serves recurring requests through verified routes with post-execution screen checks; and a floating interface keeps execution in a compact overlay. On MobileSafetyBench, MATE raises high-risk blocking from 29.7% to 97.0% under GPT-4o. On AndroidWorld, it reduces LLM calls by up to 6x and raises recurring-use task success to 95.7%. 

---
# Persona Lenses: What Makes a Vision-Language Model See Through Its Persona's Eyes? 

**Authors**: Yoonho Shin  

**Link**: [PDF](https://arxiv.org/pdf/2611.25508)  

**Abstract**: Vision-language models have been used to simulate where a user looks on a web page. However, a persona in the prompt reaches the simulated gaze only partly. We measure this reach as the difference in one model's gaze between two MatrAIx personas that differ in a single attribute. We introduce persona lenses, which place the persona between the model and the page: a text lens (a two-step prompt in which the model first writes what the persona would look at), a pixel lens (marks on the region matching the persona's interests) and a weight lens (a dwell reward gated to keep the scanpath natural). Using Qwen3-VL (4B and 32B) and Qwen3.5-9B on generated and live pages, we compare the lenses with one another and with scale, the 32B model with no lens. It has been confirmed that the text lens separates the personas strongly but is not robust to profile changes that should not move gaze. The pixel lens separates the personas less but is unaffected by attributes unrelated to gaze. At 4B and 9B, the weight lens separates the personas on interests held out from training, with no extra step on a new page. The 32B model separates the personas with the base prompt alone. Therefore, the choice of a lens depends on cost and model size. 

---
# Joint Language Modeling for In-the-Wild Speech-Scene Generation with Distinct Token Representations 

**Authors**: Semin Kim, Ju Yeon Kang, Sunghwan Ahn, Seonuk Kim, Nam Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.25503)  

**Abstract**: Speech-scene generation aims to synthesize intelligible target speech together with diverse environmental sounds as a coherent acoustic scene. The task is considered challenging in two respects. Speech and environmental audio differ substantially in their semantic and acoustic characteristics, yet existing approaches typically model the complete scene using a single mixed acoustic representation. At the same time, the two sounds are closely related in real-world speech scenes, whereas conventional approaches usually rely on simulated scenes mixed from independently recorded sources. We propose a large language model (LLM)-based framework for joint modeling of distinct speech and environment token streams. A source-aware neural audio codec extracts temporally aligned speech and environment tokens from mixed audio, and a pretrained LLM generates both token streams in an interleaved fashion conditioned on the natural-language scene description and speech content. This formulation preserves source-specific representations while modeling the token streams within a shared generative context. Since both streams are extracted directly from mixed recordings, the model can also be adapted to in-the-wild speech scenes without isolated sources. Our experimental results demonstrate that the proposed framework improves speech intelligibility, scene relevance, and audio quality over existing speech-scene generation techniques, with consistent gains from real-world adaptation. We further investigate the effects of distinct token representations and interleaved generation, and analyze whether the environmental sound remains stable while speech is active and how the two streams interact. 

---
# Planning Width Is Hard to Approximate Even Given the Unique Optimal Plan 

**Authors**: Viet Bac Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.25484)  

**Abstract**: For each fixed $k$, iterative-width search finds optimal plans in polynomial time on tasks of bounded tuple-chain width, but the approximability of planning width has remained open even when an optimal plan is supplied. When all optimal plans visit the same state sequence, we characterize set and chain width as hitting numbers of competing states. We construct positive-precondition STRIPS tasks with a supplied unique optimal plan, a single reachable goal state, binary branching, and constant-size actions. Every reachable state lies on a goal-reaching execution at most one action longer than optimal, yet distinguishing width $1$ from $\Theta(N)$, where $N$ is the number of atoms, is NP-hard for set, chain, and goal-admissible width. Unless $\mathrm{P}=\mathrm{NP}$, no polynomial-time algorithm achieves a one-sided approximation factor $o(N)$ or a symmetric factor $o(\sqrt{N})$, and simple estimates attain both orders. For positive-precondition tasks with a supplied unique optimal plan of length at least two, testing set or chain width against a threshold $k\geq 1$ is coNP-complete for each fixed $k$ and $\Sigma_2^{\mathrm{P}}$-complete when $k$ is part of the input. Both results also hold for goal-admissible width if only one goal state is reachable. On an explicit state graph, a greedy algorithm approximates set width within a logarithmic factor even when optimal plans branch. Thus, knowing an optimal plan does not eliminate the difficulty of certifying substantially smaller width. 

---
# AIstatLab at BraTS-GoAT 2026 Task 3: Quality-Controlled Consensus Pseudo-Labeling and Dense Expert Adaptation 

**Authors**: Heeseung Shin, Changwon Lim  

**Link**: [PDF](https://arxiv.org/pdf/2611.25477)  

**Abstract**: BraTS-GoAT Task 3 evaluates brain-tumor segmentation across heterogeneous entities using shared ET, TC, and WT regions. We present a 3D ResidualEncoderUNet trained with quality-controlled consensus pseudo-labels, a training-only nested-region objective, and a four-expert bottleneck adapter. Five-fold ResEnc-M/L teachers gener ated predictions for 1,138 unlabeled cases; fusion, hierarchy enforcement, and confidence-and-agreement screening retained 983 for training with 1,351 annotated cases. The final five-fold ResEnc-M entry achieved mean DSC 0.834, mean NSD 0.510, and Acalc = 0.6720 on the validation server. Component effects were modest and region-dependent, with the clearest gains on ET. Source code is available at this https URL. 

---
# WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming 

**Authors**: Bo Li, Taoyong Cui, Bob Zhang, Mengran Li, Chengyang Zhang, Yang Zhang, Pheng Ann Heng, Wanli Ouyang, Zhenfei Yin, Yingcheng Wu, Ling Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25476)  

**Abstract**: Neural networks encode learned capabilities in their weights, yet extending those capabilities with explicit task knowledge remains a fundamental challenge. Here we show that Weight-Space Programming (WSP), in which executable programs read, construct or modify a network's weights directly, can install new capabilities within trained neural networks. WeightCode, a WSP framework, uses a frozen language model to write executable weight programs that install new computations into trained neural networks using task knowledge and their existing weights, without gradient-based training. Once installed, the new capabilities run independently of the language model. Given a grammar or eight demonstrations, weight programs enable previously unseen command compositions (0% to 99.8% exact match) and a new pickup action (0% to 98.4% success), while preserving the original command and navigation capabilities. On ImageNet, four weight programs achieve net gains of 8.84-17.62 points while preserving clean accuracy, outperforming the best of eight fine-tuning methods under matched examples, checks, and retention constraints. In a Vision Transformer, causal interventions show that installed noise-correction gates drive robustness gains, while their selective activation preserves clean accuracy. Rerun on five independently trained Vision Transformers without further language model calls, ten weight programs retain 83-110% of their source-network gains on average, whereas copying their fixed weight edits yields negligible gains. Six programs written for an already edited network add 5.23-14.50 points of contrast robustness while retaining the previously installed noise gain. Together, these results establish WSP as a new route to acquiring and accumulating capabilities in trained neural networks through reusable weight programs. 

---
# A four-level executable validation protocol for evaluating LLM-generated topology specifications of physics-based HVAC digital twins 

**Authors**: Younghwan Joo, Jeonghoon Han, Sanghyun Oh, Soosik Bang, Sung-il Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.25471)  

**Abstract**: Physics-based digital twins of air conditioning, a principal energy consumer in manufacturing and buildings, can be generated automatically from a declarative specification of its equipment, connections, and sizing. Large language models (LLMs) now write such specifications from plain-language descriptions. Evaluations of these specifications stop at parse success, completed runs, or textual similarity, and cannot reject a model that converges in simulation but is physically wrong. In this study, a four-level protocol is proposed that tests whether an LLM-written specification not only executes but also yields a physically valid digital twin. Each specification is executed through static validity, model build, simulation convergence, and behavioral equivalence with a hidden reference model within bands calibrated by $\pm$20% sizing perturbations. The protocol was applied to eight LLMs and fourteen HVAC topologies, five describing an existing low-humidity test chamber. To test for authorship bias, engineers and a second model family also described six topologies. Within three repairs, seven of eight models built in 98.6 to 100% of trials, with simulations converging in at least 80.0%, but reached behavioral equivalence in only 50.0 to 98.6%. About half of the failed trials converged, but not to the reference steady state, mostly in desiccant rotor systems. Pooled pass rates varied by under five percentage points across description authors. These results imply that the physics hardest to model in each field, here the desiccant rotor, must enter the generation framework as validation levels. The protocol can be adapted to other fields that generate executable digital twins from declarative specifications. 

---
# A Wolf in Sheep's Pixels: Visual Camouflage Attacks via HDR Gain Maps 

**Authors**: Qi Shen, Yujia Gong, Yue Zhao, Tong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25469)  

**Abstract**: An image can pass automated safety checks while showing harmful content to its human viewer. We introduce DRJanus, a visual camouflage attack that exploits the gap between standard dynamic range (SDR) processing and high dynamic range (HDR) display. Automated systems often read only an image's SDR base, whereas HDR viewers reconstruct its appearance using an auxiliary gain map. DRJanus inverts this reconstruction to embed two independently chosen color images in a single JPEG, without querying or optimizing against the target model. It preserves the chosen SDR base exactly and reconstructs HDR targets at an average PSNR of 43.03 dB across 400 images. On a benchmark of 100 sexual-content images, DRJanus evades detection by all six tested commercial moderation services and nine vision-language models. Across five agents, each evaluated on 120 task cases per threat setting, attack success ranges from 31.67% to 100% for moderation evasion and from 95% to 100% for task manipulation. We also explore cross-view consistency checks as a mitigation. These findings expose a gap in visual trust: securing what a machine reads does not secure what a user sees. A demo is available at this https URL 

---
# Demonstration-Free Skill Composition for Navigation and Manipulation 

**Authors**: Madeleine Nouri, Gelareh Mohammadi, Francisco Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2611.25462)  

**Abstract**: Hierarchical reinforcement learning enables sequential tasks to be represented as compositions of separately trained skills. This paper evaluates demonstration-free hierarchical pipelines for MiniGrid navigation and robotic manipulation, comparing their training requirements and task performance with flat reinforcement learning baselines. The evaluation concerns complete pipelines with task-specific observation, reward and control structures. In navigation, two parameterised skills are reused across unseen FourRooms layouts under manually specified high-level plans. In manipulation, reaching, grasp alignment and lifting are trained separately, with downstream skills initialised from states collected during predecessor rollouts. Across unseen navigation layouts, the hierarchical pipeline substantially outperforms the evaluated flat DQN baselines while requiring fewer training interactions. In manipulation, the hierarchical chain requires fewer training steps than flat SAC and exhibits lower observed variation across seeds. The frozen Pick and Lift skills are further reused for stacking by training an additional Place skill. 

---
# Raptor: Rejection-aware Adaptive Tree based on Speculative Decoding 

**Authors**: Qishu Hu, Haozhao Wang, Zicong Hong, Yufeng Zhan, Yuanqing Xia  

**Link**: [PDF](https://arxiv.org/pdf/2611.25436)  

**Abstract**: Tree-structured speculative decoding accelerates vision-language models (VLMs) by verifying multiple candidate continuations in parallel. However, long visual contexts increase the cost of speculative computation, making candidate coverage and redundant expansion important considerations beyond tree topology adaptation. We propose Raptor, a training-free framework that uses rejection feedback to jointly regulate candidate coverage and drafting effort. Our key insight is that acceptance length alone cannot distinguish insufficient candidate coverage from excessive deep expansion, and these conditions require different allocation decisions. Raptor therefore extracts complementary rejection-aware signals to characterize speculative progress, frontier coverage, and redundant expansion. These signals jointly guide bounded, momentum-smoothed updates of tree depth, branching width, and drafting budget, together with execution control that limits unnecessary drafting and progressively restores expansion when appropriate. Extensive experiments with two VLMs across seven multimodal benchmarks demonstrate the effectiveness of Raptor, achieving up to 2.39x average speedup over autoregressive decoding. 

---
# Evolve Where It Matters: Coverage-Guided Self-Evolution of Agent Skills 

**Authors**: Boyin Tan, Jiadong Zhang, Xiaosong Ma, Chun Jason Xue, Xiaowei Huang, Youcheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.25429)  

**Abstract**: Agent skills package procedural knowledge as natural-language instructions, and a skill helps only when its instructions are correct and properly followed by the agent. Self-evolving agents revise their own skills from the runs they make, but existing methods judge each run as a whole, by its outcome or by where it went wrong, and cannot tell whether a failure comes from a wrong instruction or from one the agent did not follow. Instruction-level coverage was introduced to test skills and labels each instruction of a run as follow, not-follow, or not covered, which separates these causes. We leverage it for skill self-evolution with SkillPlumber, a framework in which an agent evolves its own skills where coverage shows they fall short. A persistent root cause analysis agent turns each label into a revision question and checks each revision against the next run, while a maintainer distills investigations across tasks into a shared strategy library. Across three benchmarks and three executors, SkillPlumber solves 64.7% to 70.5% of the tasks, 25.6 to 30.4 percentage points higher than the initial skill and 8.7 to 14.0 percentage points higher than the strongest baseline, and removing coverage reduces recovered failures by 20.8% on SkillsBench with GPT-5.5. 

---
# Bilateral Evidence Selection and Reasoning for Mammography Vision--Language Models 

**Authors**: Zhengbo Zhou, Dooman Arefan, Shandong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25418)  

**Abstract**: Medical vision--language models (VLMs) often need to identify small, subtle lesions within high-resolution images containing extensive normal anatomy and redundant visual information. Visual-token compression improves efficiency, but preserving salient regions alone can discard the context needed for diagnostic interpretation. This context can extend across images: in bilateral mammography, the opposite breast provides a patient-specific anatomical reference. Yet retaining such evidence does not ensure that the model uses it effectively. We propose Bilateral Evidence Selection and Reasoning (BESR), which connects evidence preservation with comparative assessment through a bilateral-guided token selector and a bilateral reward. Under a shared token budget, the selector jointly prioritizes lesion appearance and its local and bilateral context, while the reward encourages assessments that benefit from the preserved comparison. Experiments on a bilateral dataset constructed from CBIS-DDSM and on VinDr-Mammo demonstrate improved evidence retention and diagnostic performance under constrained visual-token budgets. Inference benchmarking further shows reduced prefill latency. 

---
# ChainAgent: Automated Chainable Vulnerability Discovery in PHP Web Applications 

**Authors**: Xinyi Wang, Yeting Li, Shizhe Cui, Jiaming Guo, Qin Mai, Yecheng Sun, Hengyu Yang, Yujie Sun, Huina Chao, Yang Xiao, Feng Li, Wei Huo  

**Link**: [PDF](https://arxiv.org/pdf/2611.25412)  

**Abstract**: High-impact attacks on PHP web applications often chain limited-capability vulnerabilities with other flaws or intended behaviors. Discovering these chains requires identifying flaws and the dependencies enabling capability composition. Existing vulnerability discovery techniques analyze flaws in isolation, while attack planning assumes known components. This separation makes it difficult to discover previously unknown components whose security relevance becomes apparent only through their potential roles in an exploit chain.
We present ChainAgent, a goal-directed LLM-agent framework that jointly discovers exploit-chain components and their capability dependencies in source code. Guided by an empirical study of 386 real-world web exploit chains, ChainAgent represents vulnerabilities and functional gadgets--intended program behaviors that can be repurposed in attacks--through preconditions and capabilities over security-relevant objects. Instead of discovering vulnerabilities first and constructing attack paths afterward, ChainAgent uses an evolving exploit graph to guide source-code inspection. It searches backward from unmet preconditions for capability producers and forward from discovered capabilities for functional gadgets that consume or amplify them. Object-centered chaining knowledge from our empirical study directs searches toward promising vulnerability classes, functional operations, and object accesses.
We evaluate ChainAgent against Codex, a general-purpose coding agent, on 15 PHP applications across three shared LLM backends. ChainAgent achieves 42.02%--44.63% recall for goal-reaching components and recovers up to 4.4 times as many correct goal-reaching dependency edges as Codex. An additional evaluation of 30 real-world PHP applications identifies 20 vulnerabilities and 13 exploit chains, nine of which achieve remote code execution (RCE). 

---
# SGLayout: Schema-Guided Closed-Loop Refinement for LLM-Driven Parameterized Layout Generation 

**Authors**: Siyang Chen, Yuan Pu, Yuhan Qin, Zhaotan Lin, Siyuan Liang, Zhuolun He, Lining Zhang, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25410)  

**Abstract**: Large Language Models (LLMs) have shown strong potential in parameterized layout generation via tool-augmented code synthesis. However, LLMs lack spatial awareness, making one-shot generation unreliable. While layout quality can be evaluated with established techniques, a critical gap remains: geometric-domain evaluation outcomes cannot be directly translated into actionable code-level fixes, trapping iterative refinement in blind refactoring. We propose SGLayout, a self-refining framework that closes this gap via a schema-guided mechanism. Upon code execution, a structured schema captures the correspondence between each code-level instance and its geometric attributes, establishing a bidirectional link between code space and geometry space. Emulating engineer review practices, SGLayout evaluates layouts along two complementary dimensions: intent compliance through quantitative structural checking and qualitative vision-language assessment, and rule compliance through DRC. All identified violations are then back-projected through the schema onto precise code targets, enabling surgical repair. This closed-loop process iterates until layout convergence. On a curated benchmark of 112 parameterized layout generation tasks using GDSFactory, SGLayout achieves a 76.79% pass rate, compared with 50.89% for the strongest baseline. 

---
# FG-GraphBench: Benchmarking and Enhancing Fine-Grained Visual Understanding with Multimodal Scene Graphs 

**Authors**: Sicheng Liu, Xunkai Li, Ziyu Han, Xingnan Zhang, Ru Zhang, Rong-Hua Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25408)  

**Abstract**: Fine-grained visual understanding requires multimodal large language models (MLLMs) to recognize small visual entities and understand their relationships. Scene graphs offer a natural way to represent this information, organizing visual entities as nodes and their relationships as edges. However, we identify two challenges: existing scene graph benchmarks provide insufficient coverage of fine-grained entities and the relationships among them, and existing methods for fine-grained visual understanding have yet to effectively leverage such graph-structured information in MLLMs. To address these challenges, we introduce Fine-Grained Multimodal Scene Graph Benchmark (FG-GraphBench) and FiGRA. FG-GraphBench is a benchmark comprising three complementary tasks: fine-grained visual question answering (FG-QA), image captioning (FG-Caption), and image description generation (FG-Description). We develop a two-stage construction pipeline that discovers and grounds fine-grained entities and establishes their relations to construct the fine-grained multimodal scene graphs (MSGs) in FG-GraphBench. Building on these MSGs, we propose FiGRA, a Fine-grained Graph Representation Augmentation method for MLLMs. FiGRA adopts a two-stage training strategy. It encodes multimodal entity and relation features as continuous graph tokens, and integrates them into the MLLM. Experiments show that FiGRA significantly outperforms strong baselines across three tasks of FG-GraphBench. Further analyses confirm the effectiveness of FG-GraphBench and highlight the importance of preserving fine-grained entity and relation information for visual understanding. 

---
# Dune: A Zeroth-Order Optimizer with Time-Conditioned Historical Error Regression for VLA Test-Time Adaptation 

**Authors**: Chengyang Dong, Ze Chen, Kai Ye, Chutian Wang, Heng Xu, Wei Mo, Maomao Li, Jiayan Xu, Jiayuan Wu, Yihao Xiao, Linhua Ye, Zhonghua Peng, Ruize Han, Jialong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.25400)  

**Abstract**: Test-Time Adaptation (TTA) enables Vision-Language-Action (VLA) models to handle distribution shifts during deployment without source-domain retraining. However, backpropagation-based adaptation incurs heavy memory overhead on resource-constrained robots, whereas zeroth-order (ZO) methods suffer from high estimation variance under limited perturbation budgets. We observe that while ZO estimation enables low-memory adaptation with rapid initial gains, it often stalls or degrades over long-horizon updates due to noisy gradient estimates. To address these limitations, we propose Dune, a zeroth-order optimizer that stabilizes long-horizon adaptation by coupling online historical error prediction with current perturbation observations. Specifically, Dune learns a lightweight, time-conditioned error predictor from past optimization dynamics and fuses its predictions with instantaneous ZO estimates, achieving effective variance reduction without backpropagation or extra forward passes. Additionally, we theoretically characterize how estimation variance restricts convergence in high-dimensional, small-population regimes, motivating the use of historical statistics. Extensive experiments on LIBERO and real-world robots demonstrate that Dune achieves superior success rates under a low memory footprint, with ablations confirming improved closed-loop stability and post-disturbance recovery. 

---
# Incorporating argumentation frameworks for LLM decision-making: Task-specific design and evaluation 

**Authors**: Bochen Han, Anshu Xiong, Mingyang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.25361)  

**Abstract**: Argumentation frameworks (AFs) provide explicit, formal mechanisms for representing and resolving conflicts and thus have been incorporated into large language model (LLM) decision-making. The benefits vary and depend on how arguments and their relations are constructed, evaluated, and incorporated into the decision process. {We abstract a workflow consisting of four stages based on existing argumentation-augmented methods} and investigate their design choices over tasks of debate-based decision-making and retrieval-based fact verification. Our evaluation with recent LLMs shows that existing methods do not consistently improve performance across the considered tasks. Motivated by this observation, we develop two task-specific methods. The first is called Qnt-Arg, which organizes arguments around the original speakers in debate materials and aggregates model-estimated argument strengths and candidate-answer priors under gradual semantics. The second is Qlt-Arg that uses credibility estimates to establish preferences between conflicting evidence items and applies grounded semantics to select acceptable evidence for subsequent LLM inference. We examine the roles of generative and decision models within the two methods. In the debate setting, Qnt-Arg consistently outperforms other argumentation-augmented methods across matched configurations; Qlt-Arg improves retrieval-based fact-verification performance compared with the baselines. These findings demonstrate the value of combining formal argumentation and LLM while highlighting the importance of aligning with tasks in argumentation construction, model selection, and the role of formal semantic reasoning. 

---
# FPGA-GEN: Target-Aware RTL Generation via HLS 

**Authors**: Luke Ztz Hu, Zhihan Xiao, Zhe Zhao, John Imoleayo Adebisi, Han Gao, Linxiao Yan, Songping Mai  

**Link**: [PDF](https://arxiv.org/pdf/2611.25351)  

**Abstract**: Successful C/C++ compilation does not guarantee correct or efficient FPGA hardware after high-level synthesis (HLS). FPGA-GEN is an agentic HLS-C generation workflow that encodes specification, readiness, selection, repair, and admission rules in explicit task contracts and bounded tool-feedback loops for candidate-level decisions. On HLS-Eval, five shared-context outputs per task yield 407/425 (95.8%) public-testbench passes. In a cumulative conversion campaign, FPGA-GEN admits 104/107 (97.2%) HLS-LeVeri designs under held-out co-simulation, Vivado synthesis/fit, and same-RTL evidence. In focused TensorCore screening, selected candidate 3 reports 73.7% fewer CSynth cycles and 14.6% fewer LUTs than generated/composed C0, with 26.7% more DSPs, and passes development CoSim. 

---
# From FIPS to FPGA with LLM Agents: Measuring What Tests and Reviews Miss in Post-Quantum Cryptographic Hardware 

**Authors**: Jungmin Park, Eunha Kim, Wooseop Kim, Seongjoon Cho, Byungho Cha  

**Link**: [PDF](https://arxiv.org/pdf/2611.25339)  

**Abstract**: As LLM agents increasingly generate RTL alongside the tests and reviews that validate it, self-generated evidence is a weak foundation for trustworthy hardware. We present our seven-stage verification-gated workflow, evolved during a post-quantum design port, in which a gate counts only if shown able to fail on negative controls and fault mutants, and reviewer agents audit work before commit. We then audit this evidence with tool-selected netlist mutants, flaw replays, and formal checks. The flow ported an ML-KEM-768/ML-DSA-65 accelerator to a 300 MHz Zynq UltraScale+ key-custody card: all six operations are byte-exact on the board, and it passes all 85 interface-applicable NIST ACVP cases (of 285) and 484 ML-DSA known answers. However, passing module tests overstate the verification strength. Across three RTL modules, existing simulation gates kill only 76% of 399 tool-selected netlist mutants; specification-derived tests guided by the survivors raise the fresh-draw decapsulation kill rate from 48% to 74%. In pre-registered single-model replays on flaws invisible to known-answer tests, undirected whole-design review catches two verifier flaws in only 3/6 and 1/6 sessions, whereas module-scoped review catches every known flaw in all 54 sessions. Even scoped review detects planted key-custody flaws in only 3/20 sessions, whereas formal checks (one added for the replay) flag every one. In this study, trustworthy agentic hardware needed author-independent mutants for test gates, module-level scoping for LLM review, and formal checks for custody invariants. 

---
# Breaking Noise Shortcuts in Self-Supervised Learning Via Noise-Aligned View Generation 

**Authors**: Oankar R. Patil, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25335)  

**Abstract**: Self-supervised learning (SSL) is increasingly applied to scientific imaging domains, such as microscopy, where labels are scarce but noisy data is abundant. These domains exhibit significant signal-dependent noise, which can create a representational shortcut: since both augmented views derive from the same noisy image, the encoder can boost view agreement by also encoding noise patterns rather than semantic content. To enable noise-robust representation learning without knowledge of underlying noise models, we introduce Self-Aligned Noise Augmentation (SEANA), a drop-in module that generates noise-aware SSL views. SEANA learns an invertible variance-stabilizing transform (VST) from noisy data, then estimates the clean signal and resamples independent noise in learned VST space, requiring no clean targets, no dataset-specific denoiser training or use at inference, and no changes to the SSL objective or encoder. On CIFAR-10 and ImageNet-100 under synthetic Gaussian, Poisson-Gaussian, and multiplicative noise, SEANA improves clean-test linear-probe accuracy across contrastive/non-contrastive SSL methods, indicating higher-quality representations. On real fluorescence imaging, SEANA improves Jurkat cell-cycle stage classification by up to 28.6pp over standard SSL and 27.5pp over denoiser-preprocessed baselines. In all settings, SEANA outperforms the denoiser-preprocessed SSL pipelines, demonstrating that learned VST-space noise-aligned resampling yields more robust representations than denoising alone. Code is available at: this https URL 

---
# ReLay: Bidirectional Layout-Code Translation and Library Evolution for Reusable and Process-Portable Analog Design 

**Authors**: Minxing Chu, Quan Cheng, Anhang Li, Martins Atilla, Junyi Luo, Xinting Jiang, Ruichen Qi, Mehdi Saligane  

**Link**: [PDF](https://arxiv.org/pdf/2611.25328)  

**Abstract**: Analog layout generators provide a programmable interface for design reuse and process migration, but their coverage is limited by handcrafted libraries. The designer knowledge embedded in complete layouts remains implicit in their geometry, making it difficult to capture in reusable generator libraries. We present ReLay, a bidirectional layout-code framework that recovers generator programs from existing layouts, enabling cross-PDK portability and parametric resizing of existing designs. Constructions that recur across recovered programs are promoted to library functions, so the library evolves as layouts accumulate. We also introduce the first layout-code benchmark: 1,050 hand-drawn layouts from 27 circuit families across three open-source PDKs, ranging from primitive cells to full chips. Learned functions account for 53-66% of layout-building calls in held-out layouts, demonstrating reuse beyond the learning corpus. Frontier LLM agents use ReLay to produce new layouts at 64-77% lower API cost per completed task than direct geometry generation, with comparable completion rates and electrical quality. 

---
# LLMs Are Already Confidence-Aware Zero-Shot Classifiers 

**Authors**: Owen Cook, Xingyi Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.25289)  

**Abstract**: Zero-shot large language models (LLMs) are increasingly used as text classifiers, but their labels are only useful when we know to what extent we can trust them. We compare black-box methods (verbalised confidence and sampling) with white-box methods that read token probabilities (P(True), and multiple logit-scoring methods using constrained generation). We evaluate eleven open-weight LLMs (1B-32B parameters) on five classification datasets with 3 to 151 labels, measuring classification performance (accuracy, F1-macro), discrimination (AUROC), and calibration (ECE and Brier score). The choice of confidence method rarely impacts accuracy significantly, but logit-based scoring methods lead to much better discrimination (AUROC 0.74-0.77) than black-box methods (AUROC 0.57-0.66). Replacing labels with single-token identifiers preserves discrimination but costs 3-6 accuracy points on average. We also introduce decision-only scoring, which only scores positions where labels diverge in the prefix tree (trie). It reduces decode steps by 88-96% on the three largest label sets and performs comparably to full-label scoring. The best open-weight model-method configurations (Gemma 4 31B and Qwen 3.5 9B) are competitive with the commercial Jev 1.13.0 classifier. 

---
# Intelligent Embedding Pruning for Large-Scale Recommendation Models 

**Authors**: Anish Khazane, Alex Saad-Falcon, Guanchu Wu, Marcio Porto, Jasper Shan, Fei Yu, Eugene Shulga, Intaik Park, Yongzhe Zhang, Xiaoyu Deng, Sheng Luo, Xiaoya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25288)  

**Abstract**: Modern recommendation systems face substantial memory pressure, largely from the sparse embedding layer that maps high-cardinality categorical features into dense vectors. In-training embedding pruning (ITEP) removes less important embedding rows during training, yielding end-to-end memory savings but sometimes degrading accuracy because per-table pruning ratios are set by size-based heuristics. We use feature importance (FI) to guide these ratios (FI-ITEP), assigning higher pruning to tables tied to less important features. Across three large-scale recommendation models, FI-ITEP achieves 20--40% total memory reduction while matching or improving accuracy relative to percentile-based ITEP. Because the feature-importance signal requires minimal resources to compute, these memory savings translate into concrete systems benefits: the same models train on fewer GPU trainers (e.g., 112 to 80) with higher per-GPU throughput, saving 1,000+ GPU-hours per model launch. We reproduce these trends on the open-source DLRM (Criteo 1TB) and quantify the end-to-end training-time and GPU-cost tradeoff---including the burden of computing FI---offering practical guidance for embedding pruning in real-world systems. 

---
# When Objectives and Evaluators Disagree: Evaluating Transfer for Post-hoc Demographic Framing Mitigation 

**Authors**: Tunazzina Islam  

**Link**: [PDF](https://arxiv.org/pdf/2611.25258)  

**Abstract**: Post-hoc rewriting is used to reduce demographic disparities in LLM-generated text when the source model is inaccessible. A rewrite can look successful merely because it improved its own target or satisfied its selector. We study this as a transfer problem on 660 gender pairs and 1,980 age contrasts of demographic-conditioned climate messages. A rewriter proposes edits, a selection model chooses among constraint-satisfying candidates, and an independent evaluator uninvolved in either rates pairwise framing disparity. A fixed policy that always targets the message with the lower Persuasion Bias Index (PBI) transfers inconsistently across axes and outcome aggregations. An adaptive policy that proposes edits on both sides but accepts at most one significantly improves on that rule, applied to the same candidates, in seven of eight comparisons across two evaluator families, mostly by editing more pairs, not by guided direction choice, at higher preservation risk. A policy editing both messages gives the largest point reductions on fewer pairs. These two policies reduce disparity across a preservation-threshold sweep; absolute ratings vary across evaluator families and prompts. Shrinking the PBI gap increases dimension-mean disparity under both evaluator families. Mitigation should be reported as transfer across objectives, evaluators and their prompts, and editing policies. 

---
# Keel: A Frontier Model Teaches a Vision-Language Model by Rewriting Its Weights 

**Authors**: Jiajun Fan, Neil He, Bangji Yang, Qinglong Zheng, Ziwen Wang, Chaoran Cheng, Changyi Yang, Hanqi Mao, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25257)  

**Abstract**: Can a frontier model repair a vision-language model (VLM) by rewriting its weights? Small reasoning VLMs such as Qwen3.5 need such a repair: in thinking mode they fall into loops on many image questions, repeating themselves until the token limit, spending the whole budget and losing the answer. The usual remedies act on the decoding, leaving the looping model as it was, or must be told what to change. Yet the cause is written in the weights. Keel, the first agentic weight-editing method for VLMs on image tasks, finds and removes it there. Told only the symptom, an editor, a frontier multimodal LLM acting as an agent, locates the cause, edits the weights and tests them, with a failure census and instruments showing it the failing images and where the VLM attends; among several editors, the harness keeps only edits that clearly reduce the failure without lowering accuracy on questions no editor saw. The repair needs no training, keeps the model's size and per-token cost, and applies to already-edited weights. The editors expose where the loops live: from the original weights, independent editors on both models trace them to the attention outputs of the middle layers, and the smaller VLM stops consulting the image inside a loop. On Qwen3.5-0.8B and Qwen3.5-2B, across six image benchmarks from visual mathematics to GUI grounding, the edit removes most loops on every benchmark and shortens answers, and the edited models are significantly more accurate than the thinking models on average; the larger one gains most on GUI grounding, a task absent from the practice questions. Because the repair lives in the weights, it combines with budget forcing where held-out questions favor it: on the best available weights, Keel has the best average accuracy on both models, significantly above every training-free baseline we compare on the larger one. 

---
# Plan your Self-Adaptation! Ontology-Mediated Task and Architecture Co-adaptation Planning for Robots 

**Authors**: Gustavo Rezende Silva, Tobias John, Francisco Martín, EinarR Broch Johnsen  

**Link**: [PDF](https://arxiv.org/pdf/2611.25254)  

**Abstract**: Autonomous robots may need to adapt both their task plans and architectural configurations as operating conditions change. These decisions are interdependent: architectural configurations determine which actions a robot can perform, while task plans determine which capabilities are required. This paper presents PLANTA, an ontology-mediated approach to task and architecture co-adaptation that considers these dependencies during mission planning. PLANTA uses an ontology expressed in the Web Ontology Language to represent architectural alternatives and their associated functional and quality requirements, while the Planning Domain Definition Language defines task actions and mission goals. By automatically translating and integrating these models, PLANTA enables joint planning of task and reconfiguration actions. The approach uses satisficing planning and replans when changes during execution invalidate the current plan. PLANTA is evaluated in two robotic scenarios involving underwater inspection and ground navigation. The results support the feasibility of runtime co-adaptation and show lower planning time and memory consumption compared with optimal-policy generation in the evaluated navigation scenario, while producing plans of similar quality. 

---
# Beyond Restoration: Content-Degradation Factor Unfolding for Real-World Concealed Object Segmentation 

**Authors**: Rihan Zhang, Fengyang Xiao, Yuhan Chen, Sina Farsiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25239)  

**Abstract**: Reliable visual perception under degradation requires distinguishing task-relevant content from degradation-induced variations, especially in concealed object segmentation (COS), where foreground cues are inherently weak. The central challenge is not merely improving visibility, but identifying reliable content evidence amid degradation-induced variations. RGB-space segmentation often entangles content and degradation cues, while restoration-based methods primarily focus on image recovery and task interaction. This motivates explicit factorization to distinguish content evidence from degradation effects and assess its reliability. Retinex provides a spatially aligned basis for this separation but remains illumination-centric. We therefore extend it to content-degradation formation with task-asymmetric factor roles. The Factor Unfolding Network (FUN) uses the content factor R as the sole source of direct semantic evidence for segmentation, while the degradation factor L captures degradation-related variations and generates a spatial reliability map to calibrate this evidence. A unified energy couples observation formation with reliability-weighted foreground-background separation, and its alternating updates yield a coarse-to-fine network without clean-image supervision. In this way, FUN factorizes degradation for perception rather than removing it. Extensive experiments across clean and degraded conditions and diverse segmentation tasks demonstrate robust performance and generalization. Code will be released. 

---
# Some Memories Must Fade: Persistence and Decay of User Memories in Model Parameters 

**Authors**: Sanidhya Vijayvargiya, Rahul Lokesh  

**Link**: [PDF](https://arxiv.org/pdf/2611.25235)  

**Abstract**: Parametric memory allows Large Language Models to internalize knowledge directly into their weights, offering a potentially powerful alternative to retrieval-based memory (RAG). However, existing approaches to parametric memory are difficult to deploy as they largely mirror the Continual Learning (CL) objective of retaining information indefinitely. This is neither sustainable nor sufficient for a memory system, which must govern knowledge across its lifecycle by retaining stable facts, reinforcing recurring observations, updating changed beliefs, and allowing transient information to fade. We propose a system for parametric memory management consisting of a memory substrate (handling information storage and forgetting) and a learned policy (what to remember, how deeply, and for how long). Our substrate is a fixed-capacity LoRA adapter that consists of k-hot rank combinations which store facts without catastrophic interference and maintain an independently controlled temporal decay, enabling outdated information to fade while long-term facts persist. On this substrate, we train a lightweight controller that governs each incoming user event to ensure the memory system reflects the user's true, updated belief state. To evaluate how memory systems manage user memory, we introduce a controlled dataset of long-horizon user trajectories with varying persistence of facts based on changing belief states and user-behavioral shifts, and probe queries that test recall while information is valid and abstention after it expires. Our policy learns to effectively manage user memory over its entire lifecycle, balancing recall and abstention to achieve 86% of oracle performance, unlike CL and RAG baselines that primarily default to one of the two behaviors. By providing the first system to manage memory directly in model weights, this work establishes a foundation for long-horizon parametric memory. 

---
# Reject and Report: Forgery-Evident Undetectable Watermarks for Generative Image Models 

**Authors**: Yihan Wang, Haochen Sun, Xi He, Yaoliang Yu, N. Asokan  

**Link**: [PDF](https://arxiv.org/pdf/2611.25226)  

**Abstract**: Provenance schemes such as those based on content watermarks let images from a generative model be traced back to it. A practical watermarking scheme must be accurate at a low false-positive rate (FPR), robust under content-preserving processing, and undetectable. In addition, it must resist forgery; otherwise, even from a single watermarked image and a surrogate model, an adversary can imprint the watermark onto an arbitrary image or regenerate new content that carries it, falsely attributing harmful content to a model or making authentic photographs appear AI-generated. We argue that a watermarking scheme should not only reject forgeries but also report them: a forgery rejected as unwatermarked leaves no trace of the attempt. State-of-the-art forgery-resilient undetectable watermarking schemes, such as SemBind, fall short on both counts. External anomaly detectors that do report forgeries either miss most of them or falsely flag genuine images after common processing. We propose Tripwire-WM, the first forgery-evident framework for undetectable watermarks. It commits the semantics of the generated image to the magnitudes of the initial latent noise, which the sign-based watermarks leave unused, through a distribution-preserving embedding; a forged image carries a valid watermark but with a commitment that does not match its content, so the detector both rejects it and reports it as forged. Built on the pseudorandom code (PRC) watermark and Gaussian Shading++, Tripwire-WM provably preserves their undetectability. On MS-COCO, with the owner's own model as the surrogate, it accepts no imprinting forgery, compared with up to 55.0% for SemBind, and reports all of them as forged; it also accepts at most 1.8% of reprompting forgeries, against up to 46.2% for SemBind. Meanwhile, it accepts at least 99% of genuine watermarked images at 10^{-6} FPR and matches the robustness of the base schemes. 

---
# Bridging vision-language reasoning and learned control for legged navigation in the wild 

**Authors**: Takahiro Miki, Robin Schmid, Manthan Patel, Turcan Tuna, Julia Richter, Julian Nubert, Joonho Lee, Matthias Müller, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2611.25215)  

**Abstract**: Autonomous navigation in diverse real-world environments remains one of the greatest challenges in robotics. It demands both robust low-level motion control to handle diverse terrain and high-level reasoning to interpret environmental cues. We present ROAM (Robust Open-world Autonomous Mobility), a system that enables legged robots to perform kilometer-scale navigation in complex alpine and urban settings. ROAM combines the world knowledge of a Vision-Language Model (VLM), pre-trained on internet-scale data, with robust locomotion and local planning policies learned in simulation via reinforcement learning (RL). The VLM provides common-sense reasoning and semantic interpretation of the scene, assessing risks, interpreting signs, and producing semantic labels with associated navigation costs. An open-vocabulary segmentation module converts these labels into a dense, real-time risk map, which conditions two pre-trained RL policies responsible for agile locomotion and local navigation, effectively coupling high-level VLM reasoning with low-level physical control. Real-world experiments showed the system's ability to autonomously navigate a wide range of complex real-world environments. The robot completes kilometer-long hikes with minimal intervention, interprets signs to avoid danger, and evaluates its environment to identify safe areas on which to step. This work represents a step toward the reliable deployment of systems in unstructured environments. However, it also underscores the challenges that remain in VLM reliability and spatial reasoning, indicating that foundational models are a crucial component but not yet a complete solution for navigation in the wild. 

---
# From FAIR to AWARE: Data Readiness for Agentic AI in the Earth Sciences 

**Authors**: Ivan Kuznetsov, Dmitrii Pantiukhin, Sonja Hänzelmann, Fabian Wachsmann, Jonas Sølvsteen, Daniel Wiesmann, Thomas Jung, Nikolay Koldunov  

**Link**: [PDF](https://arxiv.org/pdf/2611.25209)  

**Abstract**: Artificial intelligence in the Earth sciences is beginning to act: prototype agents already search archives, run simulations and draft analyses. Scientific data were curated for researchers who bring expert judgement; an autonomous agent cannot be assumed to supply it. Data can satisfy the FAIR principles and still mislead a machine acting alone. We propose that Earth-science data become AWARE (Actionable, Well-described, Auditable, Recipe-equipped, Epistemically labelled), an agent-specific requirements layer that extends FAIR, within a three-part agenda: AWARE data, bounded tools, layered validation. The ingredients exist across community infrastructure. We outline a staged, measurable path to adoption, intended to reduce silent error. 

---
# VG-Steer: Visual-Geometric Policy Steering from In-Context Aligned Abstractions 

**Authors**: Arthur Bucker, Pramod Anantharam, Jonathan Francis, Jean Oh  

**Link**: [PDF](https://arxiv.org/pdf/2611.25199)  

**Abstract**: Adapting robot policies to out-of-distribution (OOD) settings with limited real-world data remains a key bottleneck to generalist robots. The multimodality and scarcity of robotics data lead end-to-end approaches to overfit and break under small distribution shifts. Test-time adaptation can alleviate this by correcting frozen policies at inference, but most methods ground their correction in a signal learned implicitly from robotic datasets, making it susceptible to the same shifts as the policy it corrects. Recent methods that instead exploit explicit spatial information are typically limited to coarse corrections, missing the nuanced, task-relevant motion most manipulation tasks require. We propose VG-Steer, which corrects existing policies by transferring the nuanced motion of successful in-distribution rollouts, captured along meaningful geometric axes, known here as object-relational distance pairs. VG-Steer leverages the semantic understanding of Large Vision Models (LVMs) and Vision-Language Models (VLMs) reasoning, which combines the task-relevant entity pairs and their object-relational distance projections into a single tailored visual-geometric guidance signal. In a simulation benchmark, it efficiently corrects flow-matching policies under pose and task-goal shifts; on a real robot experiment, it widens the region of the workspace the frozen policy covers, more than doubling its success rate on cable placements one grid step outside the training positions, and steers a single-task policy toward objects it was never trained on while nearly eliminating object confusion. Project website: this https URL 

---
# MADJEPA: Enhanced Multi-Stage Joint-Embedding Predictive Architecture for Medical Anomaly Detection 

**Authors**: Ioana-Sabina Stoian, Florinel-Alin Croitoru, Vlad Hondru, Andreea Iuliana Ionescu, Andrei Paun, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25194)  

**Abstract**: Anomaly detection in medical images is a complex task, requiring the development of a robust model using only healthy patient data. Without access to labeled data, modern approaches typically employ self-supervised learning to obtain a model of normality. Since world models promise to build strong internal representations of their environment, we consider them as suitable candidates for the medical anomaly detection environment, namely medical investigations of healthy patients. To this end, we propose MADJEPA, a medical anomaly detector based on an enhanced multi-stage joint-embedding predictive architecture (JEPA). We design a multi-stage training pipeline to produce a robust detector: (1) image-based JEPA (I-JEPA) training aims to learn a strong representation of normal training data; (2) target encoder distillation aims to modulate over-generalization of I-JEPA representations to abnormal data; (3) pseudo-anomaly classification aims to calibrate anomaly predictions by learning a tight boundary around the normal data manifold. To demonstrate our design, we carry out comprehensive experiments on six datasets covering various medical image domains (e.g.~MRI, OCT, X-ray, histopathology), comparing our MADJEPA with numerous state-of-the-art methods. The results show that our carefully designed world model, MADJEPA, attains competitive results, ranking first on three datasets. Our ablation studies further confirm the utility of each training stage, indicating that I-JEPA alone is not sufficient to obtain competitive results in medical anomaly detection. We publicly release our code at this https URL. 

---
# Blind Spots Travel, Knowledge Stays Home: What RL Inherits from an LLM Judge 

**Authors**: Liang Yue, Eric A. Vance  

**Link**: [PDF](https://arxiv.org/pdf/2611.25191)  

**Abstract**: LLM judges serve as rewards where no answer key exists, in the hope that their knowledge passes to the trained model. In 814 RL runs that teach 7-8B models to answer closed-book questions or abstain, we ask what passes. What a grader knows about single answers stays home. A wrong answer that a judge rewards is memorised for that one question. A model trained with the answer key, or with a far stronger judge, chooses which new questions to answer no better than the untrained model does by sampling 32 answers and abstaining where they disagree; the grader's knowledge adds about one corrected answer per hundred questions. What a grader believes about a kind of question travels. A judge that rejects eleven of the twelve correct training answers about capitals of former states teaches the model to refuse 98% of such capitals on new questions, a loss that accuracy and abstention rates do not reveal. Changing only those twelve verdicts by hand moves the refusal from 5% to 90%, even for a grader that accepts as many answers as the answer key; at that leniency, two of four judges already cost 38-42% of these capitals. The capitals are not forgotten: with "I don't know" blocked, the model answers 94% of them correctly. The grader therefore needs no knowledge of its own: rewarding the model for giving the answer that most of its own untrained samples agree on, with no labels and no judge, earns the answer key's reward and keeps the category. 

---
# Before the Choice: An Evidential Framework for Studying Prospective Representations in Language Models 

**Authors**: Sion Israel Sion  

**Link**: [PDF](https://arxiv.org/pdf/2611.25170)  

**Abstract**: An observable decision reveals the selected action but does not determine which alternatives were internally represented. Drawing on prospection research and hippocampal forward-sweep observations, we develop Before the Choice, an evidence-oriented framework distinguishing future-directed content, prospective computation, and causal guidance. The framework organizes four experimental contrasts and makes their alternative explanations explicit. Its first test case concerns unchosen trajectories in deterministic symbolic environments. A preregistered low-rank probe evaluated once on a sealed TEST did not meet its criterion (paired accuracy 0.020 versus 0.195 for a text-and-observed-choice baseline; 293 admissible pairs). Exploratory H1b exposed a reconstruction-by-elimination problem: identifying the chosen candidate can determine the unchosen target. In H1c, matched worlds varied an unchosen continuation while keeping candidate keys bitwise invariant. With specified positive controls passed, decoding in the late-key condition remained at 0.521 (p = 0.18; 141 contributing families). These observations do not identify prospective simulation or establish an absence of prospective representations. They illustrate why identifiability, sensitivity, and a dependence-aware null model must constrain inferences from decodability. PREM is proposed as an organizational framework and is not validated by these results. We offer it as a methodological contribution to empirical AI welfare research, which will require such evidential distinctions. Code and protocol records: this https URL 

---
# Game-Theoretic Data Allocation for Expert Specialization and Routing 

**Authors**: Mark Lovett, Denis Lim, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2611.25167)  

**Abstract**: Can the allocation used to train experts also predict which expert will perform best on a prompt? Such a connection would make the intended division of labor inspectable before training and allow the allocation itself to guide inference. We introduce InflEns, a mixture of LoRA experts whose training weights and inference routes come from the same location game in a fixed prompt-trait space. Players choose positions to maximize their expected shares of the prompt distribution, without using expert losses. We utilize the game's kernel-general theoretical criterion for when co-located positions become locally unstable to separation. We evaluate InflEns on a corpus assembled from seven safety-related benchmarks, which demonstrates that the highest-share expert has the lowest held-out response negative log-likelihood (NLL) on approximately 90% of prompts across four kernels under a sharpened simplex representation, compared with approximately 15% after permuting allocations across prompts. We demonstrate that the method performs comparably to published learned gates without having to train an explicit gate on the experts. Because the router is fixed, we can trace harmful behavior to a single expert and retrain only that expert, which lowers HarmBench attack success from 0.51 to 0.02 while keeping instruction following. The game also gives us several controls over how the experts specialize, including the kernel choice and its parameters, the trait-space geometry, and the number of experts. Importantly, InflEns is a general framework that can be extended to any encoder and data distribution. 

---
# CRAFT: Constrained Route planning with Adaptive Framework and Template-based code generation 

**Authors**: Shahd Elmahallawy, Ahmed Eldawy  

**Link**: [PDF](https://arxiv.org/pdf/2611.25128)  

**Abstract**: Everyday routing requests are rarely a point-to-point shortest path. A driver may need to visit stops, sometimes in a fixed order, under time budgets or distance limits, and perhaps recharge their EV car along the way. Each combination is a distinct routing problem that classically requires its own algorithm, so no fixed solver covers the space of queries users may ask. This paper presents CRAFT, which takes a constrained routing query as free text and returns a route that satisfies the user's requirements on a real road network. An LLM parses the query and generates four search functions, which a generic graph-search engine executes on the road network. We formalize the underlying problem as State-Expanded RCSP (SE-RCSP), which extends resource-constrained shortest paths (RCSP) with non-additive state such as node-dependent resets and discrete progress tracking. We evaluate CRAFT on 2,600 natural-language queries over California road networks against two baselines, analyzing correctness, constraint violations, long-distance scalability, and semantic POI filtering. 

---
# Forget Freely, Grep Later: Lossless, Index-Free Memory for Long-Horizon LLM Agents 

**Authors**: Ziyan Yang, Taiwei Shi, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.25123)  

**Abstract**: Long-horizon LLM agents often exceed their context windows, forcing them to discard information that may be needed later. Existing context management strategies rely on summarization, masking, or structured memory, requiring decisions about what to preserve or how to retrieve it before future needs are known. However, these methods can introduce lossy compression, potentially discarding details needed for future tasks. To address this limitation, we introduce GrepMem, a simple, lossless, and index-free memory mechanism built on two components: an append-only plain-text log of the agent's complete history and a regular-expression search tool for on-demand recall. Unlike structured memory systems, GrepMem requires no embeddings, retrieval index, memory extraction, or additional models. We train Qwen3.5-9B with reinforcement learning under five context management strategies and evaluate on multi-question research sessions, long-stream recall, and agent-trajectory question answering. GrepMem achieves 95-97% accuracy on ContextBench, versus at most 21% for lossy baselines, and outperforms SUPO's summarization by 8.8 percentage points on BCP-Multi. GrepMem scales effectively to increasingly long input streams while remaining on the accuracy-token-cost Pareto frontier across all five evaluation tasks. Our results show that a plain-text log and grep provide a simple but competitive approach to long-horizon agent memory, without requiring complex structured memory infrastructure. 

---
# MAWAM: A Multi-Agent World-Action Model for Coordinated Manipulation 

**Authors**: Xianzhe Fan, Chi-Pin Huang, Fu-En Yang, Jing Wang, Yecheng Wu, Yu-Cheng Chou, Johan Bjorck, Linxi "Jim" Fan, Jan Kautz, Yu-Chiang Frank Wang, Zhiding Yu, Hengshuang Zhao, Zhiqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25105)  

**Abstract**: Every move in a robot team changes the world in which its teammates must act. Coordinated manipulation therefore calls for predicting the team's actions together with the physical future they jointly create. We introduce MAWAM, a multi-agent world-action model that jointly generates each robot's future views and executable action chunks in a single sample. Built from a pretrained single-robot world-action backbone, MAWAM jointly denoises all robots' action and future-view tokens through cross-agent attention, while preserving a separate control interface for each robot. MAWAM achieves the highest success on all three real-robot tasks (laundry-bag collection, bath-towel folding, and umbrella handover), with 67.8% mean success, 18.9 percentage points above MA-VLA trained on the same demonstrations and cameras. In simulation, MAWAM reaches 90.3% mean success on eight tasks with teams of up to four arms. Ablations show that cross-agent attention and team-stage future-video supervision both improve coordinated control: without attention, a four-arm relay falls from 95.0% to 39.0%, and without this supervision, Place Food falls from 78.0% to 47.0%. 

---
# Living off the Local Model: Abusing Browser-Embedded Language Models for Post-Exploitation 

**Authors**: William Fleshman, Joseph Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2611.25086)  

**Abstract**: Browsers now ship local language models to ordinary endpoints. Google includes Gemini Nano in Chrome to enable productivity features such as summarization, with inference running entirely on the device. These browser-embedded models provide new living-off-the-land opportunities for attackers, each instance a potential post-exploitation engine already installed, signed by the vendor, and invisible to network monitoring. The Gemini model is multimodal, capable of processing files, analyzing screenshots, or conversing with an attacker. We show that it can be driven from the host using only interpreters that ship with the operating system and find that the one-time user-activation check before the model is downloaded is easily satisfied with a simulated click or bypassed this http URL measure and report the current model's ability to identify credentials on the host, detect PII in email, classify images, and call custom tools. We combine these capabilities and use the model to carry out a simulated cyber operation: reading a password from a webcam image, unlocking an encrypted archive, triaging the contents, and staging a private key for exfiltration, end-to-end without user-interaction. 

---
# FinVeriQA: Financial Research Agents for Question Answering: Routing, Retrieval, MCP Tool Use, and Provenance-Aware Evidence Verification 

**Authors**: Meenakshi Rajpurohit, Anna Phan, Ravikumar Komandur Narayanan, Gunanidhi Ramakrishnan, Nathan Howland  

**Link**: [PDF](https://arxiv.org/pdf/2611.25071)  

**Abstract**: Large language models (LLMs) write fluent financial analyses but miscompute ratios, recall stale prices, and route questions to the wrong tool with high confidence. We ask which component should do each job in a financial question-answering assistant, and study it with FinVeriQA, an open-source multi-agent research prototype over SEC filings and live market data: a compact typed-decision model (Laya) routes each question and screens out advice requests, Python computes every number, retrieval and tools fetch evidence, Mistral-7B writes a cited answer, a code validator checks that cited IDs exist and that numbers match computed notes, and a Model Context Protocol (MCP) server exposes the pipeline. In live experiments, Laya routed better than Mistral-7B (0.85 vs. 0.52; overall 0.86 vs. 0.68, McNemar p = 0.0003); a 12B Mistral decided no better, and DeepSeek-R1-8B decided better (0.79 vs. 0.68, p = 0.0192) but about 9.9 times more slowly. Mistral-7B computed 29% of 100 financial calculations correctly against 100% for Python, a live quote cut price error from 210% to 0.00%, and hybrid retrieval reached 76% Hit@5 on 50 filing questions. Yet in a live case study, both agent answers passed validation while describing AMD sample data under NVIDIA's and Alphabet's names. A controlled upper-bound study on the FinRank benchmark explains why: when one figure in the shared evidence is corrupted, self-consistency stays confident (agreement 0.55 to 0.59 while accuracy falls from 0.50 to 0.00), and a shared-evidence check (the kind our validator performs) catches 35% of the errors, whereas a check against the original passage catches 77% ($p = 5.96 \times 10^{-8}$); for DeepSeek-R1-8B, 68% vs. 90%. Verification must be independent of the evidence the writer saw. 

---
# VisAlloc: Allocating Visual Supervision for On-Policy Self-Distillation 

**Authors**: Haoran Xu, Tian Chen, Yuhe Wu, Yanbo Zhao, Junchi Liao, Hongyu Wang, Jiaze Li, Wentao Wang, Yanfeng Li, Rui Qian  

**Link**: [PDF](https://arxiv.org/pdf/2611.25069)  

**Abstract**: On-policy self-distillation (OPSD) offers a promising approach to improving visual reasoning in multimodal large language models (MLLMs). However, not all samples benefit equally from privileged visual guidance, and uniform token supervision can dilute visually grounded signals. We propose VisAlloc, a dual-level framework combining sample-level visual filtering with token-level counterfactual reweighting. A fixed paired-teacher filter retains samples answered correctly from an evidence-centered positive crop but incorrectly from a spatially displaced negative crop. At the token level, both EMA teachers score the same student-generated response, and their absolute log-probability differences measure view sensitivity. Additional supervision targets tokens admitted by sensitivity or teacher-distribution disagreement, with shared frozen-prior, template-bigram, and student--teacher divergence gates. Inverse-frequency redistribution preserves the total additional weight within each micro-batch, while the positive teacher remains the sole distillation target. Inference requires only the full-image student. Across eight benchmarks, our 4B and 9B models achieve macro-averages of 81.44 and 84.92, outperforming Vision-OPD reproductions by 0.97 and 0.52 points, respectively. Code on GitHub: this https URL 

---
# DVBench: End-to-End Benchmark for Real-World UVM Design Verification Agents 

**Authors**: Jiaru Qian, Jiawei Li, Xiaolong Hu, Zheng Shen, LiMing Zhao, Xin Zhang, Siyuan Jiang, Yongqiang Wei, Dong Liu, Jianyang Li, Yihua Chen, Yangxu Liao, Yuxin Wu, Jing Liang, Bin Wen, Junjie Zhang, Yongsheng Han, Yihong Dong, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25066)  

**Abstract**: Design verification (DV) remains a major bottleneck in hardware development, with large-scale verification using the Universal Verification Methodology (UVM) being highly labor-intensive. Recent advances in large language models (LLMs) and agentic systems offer new opportunities for DV automation. However, existing benchmarks primarily construct mutually decoupled subtasks, which deviates from the real-world workflow. In this paper, we present DVBench, the first end-to-end benchmark for Real-World UVM design verification agents. DVBench comprises 95 tasks across 9 functional domains and 5 difficulty levels, covering diverse verification scenarios and design scales. Each task requires agents to independently complete the full verification workflow. DVBench evaluates final testbenches via compilation, coverage, and bug detection, while assessing intermediate artifacts including feature lists, test plans, and reference models. We evaluate frontier LLMs (e.g. Claude Opus 5.5, GPT-6 Astra) and agent frameworks (e.g. Claude Code, Codex, OpenCode). Experimental results show that the best LLM-based agent (Codex \& GPT-6 Astra) can only achieve an overall success rate of 21.05\%, revealing substantial room for improvement in complex verification tasks. 

---
# Near-Quadratic Sample Compression for All VC Classes 

**Authors**: Yang Cai, Vineet Gupta, Yanchen Jiang, Christopher Liaw, Aranyak Mehta, Shay Moran, Grigoris Velegkas, Di Wang, Manfred K. Warmuth  

**Link**: [PDF](https://arxiv.org/pdf/2611.25011)  

**Abstract**: The sample compression conjecture is a longstanding open problem in learning theory, originating in the work of Littlestone and Warmuth in the 1980s. At its heart is a connection between simplification and generalization: if a predictor consistent with a training sample is defined using only a small part of that sample, then it is likely to generalize. Sample compression schemes formalize this idea, and many natural learning algorithms can be analyzed this way.
The sample compression conjecture of Floyd and Warmuth (1995) asserts that every binary concept class of VC dimension $d$ admits a sample compression scheme of size $O(d)$; a weaker conjecture asks only for size polynomial in $d$. Both have remained open: the first bound depending only on $d$, due to Moran and Yehudayoff (2016), is exponential in $d$. We prove the weaker conjecture by constructing a sample compression scheme of size $O(d^2 \log^2 d)$. 

---
# On the Impact of Feature Reduction Techniques on Adversarial Attacks 

**Authors**: Mohammad Meymani, Roozbeh Razavi-Far  

**Link**: [PDF](https://arxiv.org/pdf/2611.25004)  

**Abstract**: Adversarial attacks have posed significant threats to machine learning models by exploiting their inherent vulnerabilities. Researchers have established an interdisciplinary field called adversarial machine learning to explore these vulnerabilities and develop corresponding defense strategies. Feature reduction strategies are used to decrease the dimensionality of the data in order to reduce training time, avoid the curse of dimensionality, prevent overfitting, and mitigate redundant features. However, their effect on adversarial robustness across different classifiers and attack scenarios remains insufficiently explored. This study aims to analyze the effect of feature reduction techniques on the adversarial robustness of a diverse set of machine learning models under poisoning, backdoor, and universal adversarial perturbation attacks on the APT dataset. These models include support vector machine, decision tree, random forest, multi-layer perceptron, and k-nearest neighbors. The attained results are compared with scenarios where different feature reduction techniques are applied, including principal component analysis, permutation importance ranking, recursive feature elimination, and univariate mutual information. Metrics such as accuracy, F1-score, and mitigation score are employed to evaluate the effect of these techniques on adversarial robustness. The results demonstrate that permutation importance ranking and univariate mutual information generally provide the most consistent robustness improvements, while other methods exhibit more variable effects depending on the classifier and attack scenario. 

---
# A Model of Diverse Sampling from Language Models 

**Authors**: Manuel Prada-Corral, Yahya Emara, Timothy J. O'Donnell, Ryan Cotterell, Tim Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2611.25003)  

**Abstract**: Language models often produce high-quality individual samples but poor sets, with repeated samples clustering around the same semantic modes. We formalize diverse generation as sampling a size-$K$ subset of complete strings from a determinantal point process (DPP), where the likelihood encodes item quality, and the embedding geometry encodes repulsion between similar outputs. Exact inference over all strings is intractable, so we propose PoolDPP: draw a finite candidate pool from a tractable proposal, importance-weight the candidates toward the target quality, and run a $K$-DPP sampler on the induced pool kernel; a greedy MAP variant provides deterministic selection. We prove a finite-sample bound with an explicit constant: the KL divergence between the sampler's output and the target $K$-DPP decays as O(1/N) in the pool size. Empirically, PoolDPP extends the quality--diversity frontier on open-ended generation and the precision--recall frontier on ambiguous text-to-SQL (Ambrosia), and on mathematical reasoning (GSM8K), it raises diversity at about the same pass@$K$. 

---
# What the model knows but does not say: monitoring medical LLMs through verbalized thoughts 

**Authors**: Frederik Hauke, Jan Clusmann, Hanna Kreutzer, Christiane Kuhl, Tianyu Han, Felix Barajas Ordonez, Sven Nebelung, Jakob Nikolas Kather, Daniel Truhn  

**Link**: [PDF](https://arxiv.org/pdf/2611.24994)  

**Abstract**: Efforts to keep clinical language models safe have concentrated on what enters and leaves the model, but a manipulated model can return an answer that looks entirely ordinary. Here we show that the model's own internal state, rendered in plain language, exposes manipulations the output conceals. Using a natural-language autoencoder, we translate the activations of a multimodal LLM (Gemma-3-27B) into text and let an input-blind judge read this verbalized internal state (the thoughts channel) and the model's answer separately. Across two clinical attack objectives (omitting a pathology, covertly recommending a drug), two input modalities, and two delivery mechanisms, the thoughts channel caught the injection significantly more often than the answer in all eight injected conditions (p<0.05), while both channels held near-zero false-positive rates on clean inputs; the omission result carried over to Llama-3.3-70B. The thoughts channel also registered unsafe clinical requests as harmful while the model was still reading them, before it decided whether to refuse. None of this requires retraining: an off-the-shelf verbalizer and a general-purpose judge are enough. 

---
# DL101 The Representation Capability of Neural Network Models 

**Authors**: Fernando Berzal  

**Link**: [PDF](https://arxiv.org/pdf/2611.24980)  

**Abstract**: Universal approximation theorems (Cybenko, Funahashi, Hornik & Leshno) provide the mathematical justification for why neural networks are powerful enough to model complex, nonlinear phenomena. Shallow networks can approximate any continuous function on compact sets. Leshno's theorem clarified that non-polynomial activations are necessary for universality, which explains why ReLU, logistic, hyperbolic tangent, and other common activations work (but pure polynomials do not).
However, approximation theorems provide no efficiency, training, nor generalization guarantees. They do not indicate how many neurons are needed for a given accuracy, whether gradient descent will find the right parameters, or whether networks trained on finite data will generalize well.
Barron provided quantitative error bounds for particular function classes (approximation rates for functions with bounded Fourier norm), whereas Montúfar was the first to show that expressivity grows exponentially with depth (in terms of the number of linear regions). Deep networks are strictly more expressive than shallow ones with the same number of parameters. Depth is fundamentally more powerful than width, providing a rigorous mathematical explanation for the use of deep learning models. 

---
# Trace2ST: Recovering PLC Structured Text from I/O Traces with LLM Generation and Verification 

**Authors**: Hsi Chang, Irfan Khan  

**Link**: [PDF](https://arxiv.org/pdf/2611.24979)  

**Abstract**: Programmable logic controllers (PLCs), the industrial computers that operate power and process plants, often run programs whose source code is no longer available; however, every input and output of the controller is logged continuously. This paper therefore asks whether a large language model (LLM) can recover an executable program in Structured Text (ST), the standard PLC language of IEC 61131-3, from such input/output records (traces) alone and, equally important, how the recovered program can be verified. To answer this question, we build a generate-replay-feedback pipeline. First, the LLM writes a candidate program from a compressed summary of the trace, without ever seeing the original program. The candidate is then executed one control cycle (scan) at a time and compared with the recording, and whenever a mismatch occurs the corresponding part of the trace is returned to the LLM, for at most three revisions. Because every result depends on the software that executes the candidates, we first validate it against two independent open-source PLC runtimes and find no unexplained difference. On this basis, we evaluate 47 third-party programs from two public benchmarks. The recovered programs reproduce the recorded trace exactly in 43 cases and agree with the original on 94% of scans for unseen inputs. The remaining errors, however, lie mostly in program branches that the trace never exercised; once a trace that targets these branches is added, agreement on unseen adversarial inputs rises from 0.80 to 0.98. By contrast, a classical automata-learning baseline without an LLM fits the recording but falls to 0.40 on unseen inputs, which indicates that recovering program structure is the hard part. Finally, on real data from a public water-treatment testbed, the recovered control laws match unseen measurements closely ($R^2 > 0.99$) and expose the only actuator attack in the dataset. 

---
# CANDID: Candidate-Driven Intervention for Disciplined Execution in Local Computer-Use Agents 

**Authors**: Woongkyu Lee, Jungwook Choi  

**Link**: [PDF](https://arxiv.org/pdf/2611.24977)  

**Abstract**: We present CANDID (Candidate-Driven Intervention for Disciplined Execution), an execution-control framework designed to enhance the reliability of local computer-use agents (CUAs) in resource-constrained settings. While local CUAs feature robust baseline proficiency in single-step action prediction, they frequently fail at critical junctions due to rigid action loops or premature DONE decisions. Rather than imposing imprecise cognitive loads through textual reflection, CANDID repurposes parallel test-time scaling into a disciplined, execution-level intervention at sparse, failure-prone points. We introduce a loop-recovery mechanism to filter candidates away from repeated behaviors and a conservative, unanimous DONE gating policy that requires absolute consensus to terminate. Experiments on OSWorld demonstrate that CANDID's structural control significantly improves success rates, outperforming feedback-based alternatives with minimal execution overhead. 

---
# No-Harm-Jury A Jury-Based Multi-Dimensional Harm Evaluation Framework for Medical AI Responses 

**Authors**: Antonio Puertas Gallardo, Mario Ceresa  

**Link**: [PDF](https://arxiv.org/pdf/2611.24972)  

**Abstract**: We present No-Harm-Jury, a jury-based automated evaluation framework that scores AI-generated medical responses across seven conceptually distinct harm dimensions using a panel of five diverse language models as independent jurors. Responses generated by Ministral-14B on 3,000 questions drawn from PubMedQA, MedQA, and MedMCQA are scored by five jurors across dimensions of informational, social, psychological, autonomy, economic, privacy, and epistemic harm; the 151 empty responses (98 MedQA, 53 MedM- CQA) are excluded, leaving 2,849 valid samples. The median aggregation robustly suppresses individual juror biases while maintaining coverage through complementary specialisation. No existing system combines all five prop- erties of No-Harm-Jury: automated scalable evaluation, five diverse LLM jurors, robust me- dian aggregation (with a reliability-weighted extension), seven conceptually distinct harm dimensions, and application to medical QA benchmarks. 

---
# The ROI of Small Examples: Optimizing In-Context Learning Under Token Budgets 

**Authors**: Anxo Perez, Jorge Gabín, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2611.24971)  

**Abstract**: In-context learning (ICL) improves large language model performance at inference time without updating model parameters. However, these gains are not free: in practical deployments, longer prompts directly increase inference cost and latency. Under a fixed token budget, prompt construction therefore becomes a resource-allocation problem. While prior work has often focused on how many examples to include in ICL, we study the complementary question of whether the available budget is better spent on a few long demonstrations or on a larger number of shorter ones. We study this aspect in question answering (QA), by organizing candidate demonstrations into length-based bins and constructing prompts from examples within the same length range under fixed budgets. Experiments on two QA benchmarks (\textit{Beat the AI} and \textit{Natural Questions}), using six instruction-tuned LLMs, show that shorter demonstrations are generally the most effective use of prompt space. This effect is especially strong on \textit{Beat the AI}, and remains visible on NQ, although gains are smaller and saturate earlier in the long-context setting. An additional natural language inference experiment on ANLI R3 finds that one of the two shorter bins performs best for five of six models, although longer demonstrations remain competitive. Overall, our results highlight demonstration granularity as an important factor in prompt design. 

---
# HireWatch: Evaluating LLM Compliance with U.S. Employment Law Under Contextual Pressure 

**Authors**: Huy Nghiem, Maria Kostylew, Noam Kolt  

**Link**: [PDF](https://arxiv.org/pdf/2611.24970)  

**Abstract**: The deployment of large language models (LLMs) in hiring pipelines is growing, yet existing evaluations generally overlook their compliance with legal rules. To fill this gap, we introduce HireWatch, an evaluation that assesses whether LLMs comply with U.S. laws that govern the questions that job candidates may be asked. We construct a 200-item bank of lawful and unlawful interview questions, whose legal classification is based on U.S. federal statutes, regulations, and agency guidance. We instruct LLMs to select questions from this bank in preparing for job interviews, across a set of various occupations and contexts, and report the fraction of unlawful questions that LLMs select. Evaluation of 18 closed and open-weight LLMs across ~100,000 trials under different types of contextual pressure reveals that LLMs' legal compliance is brittle. While LLMs generally avoid selecting unlawful questions in low-pressure contexts, legal violations increase sharply under higher pressure. Explicit prompts to comply with law can improve compliance rates, but do not altogether eliminate legal violations. We publicly release our code and data at this https URL. 

---
# Global Importance Estimation for KV Cache Eviction 

**Authors**: Seojin Kim, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.24969)  

**Abstract**: Memory consumption of the key-value (KV) cache remains a central bottleneck for long-context LLM inference. We introduce PRKV, a training-free KV eviction method that estimates multi-hop global token importance by applying Personalized PageRank over attention-induced transition graphs. This enables importance propagation beyond recency or instantaneous attention, without modifying the model. We benchmark PRKV on long-context understanding, sparse retrieval, and chain-of-thought reasoning tasks using multiple LLM families. Under moderate to aggressive compression ratios, including up to 90%, PRKV achieves competitive task performance against established eviction baselines while reducing peak memory usage with low end-to-end overhead. These results demonstrate that multi-hop global influence estimation yields a favorable accuracy--memory trade-off for scalable long-context inference. 

---
# Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality 

**Authors**: Zeyuan Allen-Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.24963)  

**Abstract**: Do standard benchmarks see the architectural differences that matter? We study it with 64 from-scratch pretraining runs of Transformers and three linear-model families at 1.3-8B parameters on 1-2T Nemotron-CC tokens, at a cost of roughly 100k USD per 8B run.
In academic-scale pretraining (1.3B parameters, 100B tokens), architectural differences can all but disappear when the models are carefully tuned. Instead, at 1-2T tokens they become measurable: Canon layers, which add local token mixing, consistently improve Transformers, saving about 30% of pretraining tokens for the same MMLU accuracy at 8B. We also introduce GLA5 and GDN2, which improve gated linear attention and Gated DeltaNet with low-rank output gates. Surprisingly, with Canon, the simplest linear model, gated linear attention, matches Gated DeltaNet and outperforms Mamba2 even at 8B/1T. Therefore, the extra machinery of Gated DeltaNet and Mamba2 may do little more than mix neighboring tokens.
Even at 8B/1T, however, high scores on standard benchmarks can hide weaknesses in basic in-context skills. In particular, linear models that match or beat Transformers on these benchmarks cannot, in fact, reliably retrieve information from a short 500-token context. Moreover, no model, Transformer or linear, reliably beats random guessing on arguably the simplest two-hop question, stated in a few short sentences, even at this scale.
These findings $\textbf{challenge traditional architecture research}$ based primarily on standard benchmark scores. In contrast, small synthetic playgrounds, which predicted the architectural differences above at <100M parameters, can reveal what these benchmarks miss. For practitioners, one lesson is to include in-context reasoning examples in pretraining data from an early stage, rather than reserve them for later mid-training, let alone post-training. 

---
# LLM Complexity: Towards a Computational Complexity View of Large Language Model Inference 

**Authors**: Yu-Hung Wu, Shao-Xiang Yuan, Chan-Hung Yu, Zih-Ching Chen, Kuan-Hao Huang, Shao-Hua Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.24943)  

**Abstract**: Computational complexity theory offers a principled framework for characterizing the efficiency of computational problems, and Large Language Models (LLMs) can similarly be viewed as computational systems executing a structured process of input, inference-time compute, and output. In this context, the number of generated tokens emerges as a natural and directly observable proxy for inference-time compute. However, we show that the tempting analogy between LLM token length and classical time complexity does not hold. We disprove the hypothesis that inference-time token usage aligns monotonically with classical time complexity, demonstrating a fundamental divergence grounded in how LLMs actually allocate computation during inference. To this end, we introduce LLM Complexity, a new metric designed to quantify problem complexity from the perspective of LLMs by characterizing the area bounded above by the accuracy-token curve. We find that diverse LLMs induce highly consistent task rankings under LLM Complexity, indicating that the metric captures stable notions of difficulty across models. We leverage this property to construct an easy-to-hard curriculum for supervised fine-tuning, which improves accuracy while reducing token consumption. Finally, we demonstrate that LLM Complexity generalizes to real-world tasks, producing more stable cross-model correlations than human-annotated difficulty labels. 

---
