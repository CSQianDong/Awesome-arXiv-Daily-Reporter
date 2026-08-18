# Towards Computational Provenance: Carrying Causal-State Evidence in Generated Text 

**Authors**: Benjamin Belay  

**Link**: [PDF](https://arxiv.org/pdf/2608.16868)  

**Abstract**: A language model's output does not by itself provide verifiable evidence about the internal computation that produced it. We study computational provenance: whether generated text can carry detectable evidence of which causally relevant internal state occurred. We test a bounded form of this idea in two controlled architectures: a modular feed-forward neural network and a transformer-based model. Both architectures are trained on the same arithmetic task with a mandatory pathway through two discrete intermediate states, allowing different internal paths to produce the same answer. We deliberately switch between these paths, authenticate the state actually used, and let that verified state determine a subtle statistical pattern in the generated text that can later be detected. The feed-forward and transformer systems each passed all 128 matched pairs in both their public and separately sealed protected end-to-end evaluations, with the detector recovering the signal associated with the authenticated internal state. The required causal computation also reproduced across five independently trained feed-forward models and three independently trained transformers. In a separate answer-only transformer experiment, our linear probes did not recover a naturally learned intermediate state. These results provide a controlled proof of concept that information about a verified, causally relevant internal state can be preserved in generated text even when the answer is unchanged. 

---
# Model Hypnosis: Strong control of AI via additive subliminal effects 

**Authors**: Enric Boix-Adsera, Benedict Tessler  

**Link**: [PDF](https://arxiv.org/pdf/2608.16834)  

**Abstract**: We demonstrate that AI models are broadly susceptible to a phenomenon we call model hypnosis, in which individually weak and seemingly irrelevant cues in the prompt can be systematically combined to strongly control model behavior. Model hypnosis occurs across model families and scales, including in frontier reasoning models, and hypnotic prompts can transfer between models. Because the model is controlled by inconspicuous textual choices, such as paraphrases and typos, model hypnosis presents new challenges and avenues for AI safety, and is a major hurdle for AI interpretability. 

---
# ClawGym II: Exploring Black-Box RL on Agent Harness 

**Authors**: Huatong Song, Fei Bai, Ming Yang, Renyuan Li, Jia Deng, Jujie He, Zhange Zhang, Daixuan Cheng, Yan Xing, Qi Yun, Xuxing Chen, Danyang Li, Feng Chang, Chuan Hao, Ran Tao, Jian Yang, Bryan Dai, Wayne Xin Zhao, Mingjie Tang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2608.16798)  

**Abstract**: Agent harnesses have substantially improved performance on long-horizon tasks by coordinating agent interactions with the environment. However, reinforcement learning through complex harnesses remains largely unexplored, as scaling such training to long-horizon agent tasks introduces fundamental challenges. In this work, we present a unified black-box RL framework for stable and scalable optimization of general agents through complex harnesses. Concretely, we first build a sandbox-based execution infrastructure that isolates task environments and harnesses within temporary sandboxes for large-scale concurrent rollouts. We then decouple policy optimization from opaque harness execution and place a serving proxy at the model boundary to capture model calls. To reconstruct multi-turn trajectories and improve training efficiency, we organize the captured calls into prefix trees and further adapt both critic-based PPO and critic-free GRPO to optimize over the recovered tree structure. Meanwhile, we maintain training-inference consistency throughout the optimization process. Finally, we introduce mix-harness training, allowing a single model to be jointly optimized by heterogeneous harnesses. With Qwen3-30A3B, black-box RL improves Pass@1 on ClawGym-Bench by 9.98 and 14.81 points through OpenClaw and Claude Code, respectively, while remaining stable over 200-400 optimization steps. Moreover, the framework yields consistent gains on more challenging tasks such as JobBench and OfficeQA. Overall, our framework enables effective, stable, and scalable optimization of general agents through black-box harnesses, supporting unified training across heterogeneous execution systems. 

---
# Semantic Bandits: In-Context Exploration-Exploitation is Biased by Semantic Priors 

**Authors**: David Eric Austin, Kaheer Suleman, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2608.16707)  

**Abstract**: Large language models (LLMs) are increasingly deployed as decision-making agents in settings that require sophisticated environmental exploration. However, existing work has raised questions about how LLMs actually balance exploration and exploitation. Unlike classical agents, LLM agents engage with tasks through natural language, exposing them to semantic information with no formal counterpart in the task structure. We introduce the semantic bandit, an extension of the multi-armed bandit setting that explicitly considers the textual labels assigned to actions, and use it to study how semantic priors --- inductive biases arising from associations between language and expected reward learned during pre-training, shape LLM exploration behaviour. We find that semantically informative action labels reduce exploration in favour of exploitation, improving performance when aligned with the reward structure and severely degrading it when misaligned. We further find that negative rewards trigger substantially more exploration than equivalent positive rewards, consistent with an expected-scale bias induced by reward conventions common in pre-training data. Overall, we argue that the use of language to define the environment and rewards introduces unavoidable biases derived from the fact that the model is trained on word co-occurence, with implications for the reliability and robustness of LLM agents in real-world decision-making settings. 

---
# Does the LM Head Create a Harmful Gradient Bottleneck? A Causal Test 

**Authors**: Anand Murugan  

**Link**: [PDF](https://arxiv.org/pdf/2608.16671)  

**Abstract**: The language-model head maps a hidden state of width D to a vocabulary of size V, so its transpose can return at most D independent directions to the Transformer. Godey and Artzi argue that this severe projection is a harmful optimization bottleneck. We separate the geometry from the causal claim. Our backward-only intervention keeps the ordinary logits and the exact LM-head parameter update while reducing only the rank of the gradient sent into the Transformer. Across five paired seeds on byte-level and BPE-8192 WikiText-2 models, reducing backward rank increases validation loss. An equally ranked factorized forward head, however, increases loss substantially more. At half rank in the larger model, the backward-only loss increase is 0.0586 (95% CI [0.0167, 0.1005]), while the factorized forward head increases loss by 0.1795 ([0.1547, 0.2042]). The vocabulary-space residual also contributes to the ordinary LM-head update, and removing that contribution is harmful. Additional controls show that repeated-token failures are confounded by the number of independently sampled symbols, that adding never-target output classes does not impair learning, and that projection diagnostics do not reliably predict progress in our runs. Tested auxiliary feedback routes do not beat tuned backpropagation. These results confirm strong geometric compression but do not establish that it is a harmful optimization bottleneck. 

---
# PCA-guided Activation Scaling for Monotonic Bidirectional Control over LLM Sycophancy 

**Authors**: Zheng Chen, Zhaoxin Feng, Yip Tin Po, Jianfei Ma, Emmanuele Chersoni, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.16650)  

**Abstract**: Large language models (LLMs) exhibit sycophancy, a tendency to agree with user beliefs regardless of factual accuracy. This can reinforce misconceptions, but eliminating it entirely risks over-correction against valid opinions. Effective control must therefore both reduce and increase sycophancy with predictable and gradual effect. Yet, existing methods fail to ensure a bidirectional and monotonic relationship between steering strength and behavioral outcome across models and datasets. We introduce PCA-guided Activation Scaling (PAS), an activation steering framework that decomposes residual stream activations into a PCA-identified sycophancy-honesty subspace and an orthogonal residual, then applies distinct scaling exponents to achieve monotonic, bidirectional control. Across three LLMs and three datasets, PAS achieves strong monotonicity (Spearman $\rho$ = +0.92) and an average shift of 15.4% per direction, compared with 8.7% for the baselines. Ablation studies confirm that the decomposition, asymmetric exponents, and layer selection are each essential for maintaining monotonic control. The data and code are available at this https URL. 

---
# Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models 

**Authors**: Zhaoyi Li, Deyang Kong, Yuan Wei, Evan Yang, Ranran Shen, Mahardika Krisna Ihsani, Ming Yang, Wei Zhang, Chuan Hao, Jian Yang, Ran Tao, Bryan Dai, Shikun Zhang, Wei Ye, Ying Wei, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2608.16647)  

**Abstract**: On-policy distillation (OPD) transfers teacher capabilities by supervising trajectories sampled from the student's own policy, yet its generalization behavior remains poorly understood, as most studies evaluate OPD on a single domain and on benchmarks close to the training data. We present a controlled study that varies one generalization factor at a time, from in-domain distribution shifts to cross-domain transfer and the multi-teacher setting. We find that OPD transfers a teacher's reasoning behavior rather than its answers to particular problems: training difficulty barely matters, and even problems the teacher never solves are useful. Transfer depends strongly on the origin relationship between teacher and student: same-origin pairs bring the student close to the teacher across languages, reasoning horizons, and even other domains, whereas cross-origin pairs mostly fit the trained distribution. This broad reach is a double-edged sword: since routing prompts to domain experts cannot confine each teacher's influence, combining them yields a mixture-dependent seesaw among their capabilities. These results clarify when OPD generalizes and offer a useful perspective for diagnosing multi-teacher OPD. 

---
# Toward Better Assessment of LLMs' Performance in Clinical Error Detection 

**Authors**: Yifan Zhang, Rahmatollah Beheshti  

**Link**: [PDF](https://arxiv.org/pdf/2608.16643)  

**Abstract**: Automated detection of errors in clinical documentation is a promising application of large language models (LLMs), yet decisions to deploy such models rest on benchmarks that evaluate each clinical note in isolation. Error-detection benchmarks are typically constructed by injecting errors into notes, such that each erroneous note has a natural counterpart. Aggregate discriminative metrics (e.g., balanced accuracy or F1) do not exploit this structure. We show that this omission is consequential. In particular, evaluating 15 diverse LLMs on 4 standardized clinical error-detection test sets across 3 languages, we find that 13 of 15 models fall below the level of random pairwise discrimination, even while achieving F1 scores that standard practice would read as moderate. We also observe that the underlying bias patterns differ across languages: the same model can default to "no error" on one language and over-flag errors on another. To diagnose where discrimination breaks down, we further introduce a procedure to score the evidence models cite in their outputs. We find that while models consistently locate error-relevant content, they fail to produce the corresponding correct verdict on the clean counterpart. Finally, we show that F1 and pairwise accuracy are driven in opposite directions by the same underlying bias, so that ranking models by F1 may systematically promote the weakest discriminators. For safety-critical clinical NLP applications, we advocate for supplementing aggregate metrics with paired evaluations in benchmark reporting. Code and analysis scripts are available at this https URL. 

---
# When Do Explanations Help In-Context Learning? A Comparative Study of Natural Language Explanation Types and Faithfulness 

**Authors**: Mahdi Dhaini, Adam Dejl, Juraj Vladika, Volkan Özer, Barbara Plank, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2608.16627)  

**Abstract**: Natural language explanations (NLEs) are increasingly used as inputs, for example, as few-shot rationales that influence model behavior in in-context learning (ICL). However, it remains unclear how different types of NLEs compare in their effects on downstream model performance in explanation-augmented prompting. Therefore, we provide a comparative evaluation across six benchmarks and four instruction-tuned models, studying how NLE source (human-written when available, self-generated explanations, generated by an external LLM) and NLE selection (random vs faithfulness-based filtering) affect downstream utility of NLEs when used in ICL settings. Our extensive evaluation shows that, on classification-style benchmarks, adding NLEs to few-shot prompts often improves accuracy over few-shot prompting without explanations; among NLE sources, externally generated LLM-NLEs often provide strong downstream utility and remain competitive with human rationales where both are available, whereas self-NLEs are more sensitive to the selection strategy. On math reasoning, the effects are more model- and source-dependent. We further show that faithfulness-based selection of self-NLEs yields small average gains overall, but can improve or reduce performance depending on the metric, task, and model. Different faithfulness metrics can disagree substantially, affecting which self-NLE examples are selected and their downstream predictive utility. Robustness tests with randomly swapped and out-of-distribution rationales indicate partial robustness, suggesting that semantic alignment contributes to performance gains. Overall, our results provide insights for selecting and reporting explanations that influence model behavior in practical prompting pipelines. 

---
# Palmyra x6 Technical Report: An Agentic, Tool-Use Model Post-Trained via Anchored Supervised Fine-Tuning 

**Authors**: Peng Du, Kiran Kamble, Rakshith Vasudev, Zhizhuo Yang, Rohith Nadimpally, Arjun Krishna, Waseem Alshikh, Daniel M. Bikel  

**Link**: [PDF](https://arxiv.org/pdf/2608.16620)  

**Abstract**: Palmyra x6 is a large language model optimized for use with enterprise-oriented agentic tasks. The model was built by post-training a Mixture-of-Experts base model with Anchored Supervised Fine-Tuning on a compact corpus of verified, synthetic tool-use trajectories, optimized with a Muon + Adam hybrid. The recipe is deliberately conservative and deliberately controlled: 626 trajectories, a single epoch, a low learning rate, and a KL anchor to the frozen base. The model shows substantial gains over the previous default model for Writer Agent, and compares favorably with several recent models on public benchmarks, scoring the highest on BFCL Core at $0.785$ and posts the highest six-benchmark mean of the cohort. Furthermore, the model has shown itself to be competitive or leading relative to comparators in our bias and safety evaluations. 

---
# BabelSteering: Multilingual Safety Alignment via English Steering Vectors 

**Authors**: Emma V. Stein, Dominik Meier, Terry Ruas, Jan Philip Wahle, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2608.16577)  

**Abstract**: Large language models (LLMs) are deployed globally in high-stakes settings, yet most safety research and alignment efforts remain concentrated on English. Thus, users interacting with LLMs in other languages may encounter weaker safeguards despite relying on the same systems for similarly sensitive tasks. In this work, we investigate whether safety signals learned from a high-resource language, like English, can improve multilingual safety. We propose BabelSteering, an activation steering method that acts as a lightweight inference- time intervention, using refusal directions derived from English safety supervision to generalize across languages. Our evaluation includes eight languages and jointly measures refusal of harmful requests, over-refusal, and general task utility. The results show that BabelSteering increases the refusal of harmful requests across languages, with only a marginal to no reduction in task utility but with some increase in refusal of pseudo-harmful prompts. For example, for Gemma 7B, we see an average increase in the refusal of harmful prompts across languages of 11 percentage points (pp), with individual languages like Bengali seeing an increase of 17 pp, with no loss of utility on Global MMLU, while pseudo-harmful refusals increase by 13 pp on average. We also introduce a multilingual translation-and-evaluation pipeline to facilitate future work on cross-lingual safety interventions. Overall, our findings suggest that activation steering may provide a practical, low- cost mechanism for extending English-derived safety signals to other languages. Warning: this paper contains examples with unsafe content 

---
# Ask, Condition or Abstain: Reinforcement Learning for Missing-Premise Reasoning 

**Authors**: Yongqi Tong, Zhenyu Zhang, Zimi Liu, Kewei Fu, Mingli Song, Haofei Zhang, Junshao Zhang, Hong Zhu, Jiang-Ming Yang, Xin Zhang, Jianshe Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.16554)  

**Abstract**: Answer-only reinforcement learning (RL) trains reasoning models to solve fully specified problems, but many realistic queries omit a premise needed for a unique answer. In this setting, the useful response is not always refusal: the model should ask for the missing premise, condition its answer on the unknown quantity, or abstain when no informative conditional response is available. We present \emph{Ask-Condition-Abstain Reinforcement Learning} (ACA-RL), a data-augmented RL framework for this setting. Its reasoning-graph-guided pipeline converts well-posed problems into missing-premise training instances with localized gap annotations; ACA-RL then trains on these instances with a structured reward over five observable response behaviors. We also introduce the \emph{Missing-Premise Benchmark} (MPB), a 274-instance human-verified benchmark spanning mathematical, logical, and real-world word problems. Across Qwen3 and Llama models, ACA-RL consistently improves on MPB while preserving competitive performance on well-posed reasoning tasks. Together with the released code, MPB, and training data, this work supports a new mission for NLP evaluation: measuring whether models can recognize when a task is underdetermined and handle uncertainty, not only whether they can answer fully specified questions. 

---
# STAGE: Controlled Objective Admission for Multi-Preference LLM Alignment 

**Authors**: Yongqi Tong, Zhenyu Zhang, Ruirui Wang, Kewei Fu, Shaoqing Lin, Sijie Dong, Jiang-Ming Yang, Xin Zhang, Jianshe Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.16553)  

**Abstract**: Multi-preference alignment is often framed as scalarization: combine reward dimensions, then optimize. This leaves a temporal decision underspecified: when should each preference dimension enter policy optimization? We propose \methodname, a stability-guided active-set controller for controlled objective admission. \methodname starts from a small active set, retains admitted objectives, and expands when reward-deviation gates indicate low recent deviation or a patience budget is exhausted. A probing phase estimates a hard-to-easy order, and adaptive weighting emphasizes underperforming active dimensions. Automatic evaluations with 15 training preferences and 16 held-out benchmark columns show that \methodname obtains higher averages than simultaneous scalarization and shared-budget adapted baselines. Component ablations and expansion dynamics further support cumulative retention, gated admission, and probing-derived ordering as useful design choices in this setting. These results position objective-entry timing as a concrete control variable in reward-vector RLHF. 

---
# When Context Misleads: Intent-Guided Decoding for Robust Retrieval-Augmented Generation 

**Authors**: Haolin Jin, Pengyue Yang, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.16515)  

**Abstract**: Retrieval-augmented generation (RAG) improves large language models by grounding generation in external evidence, but it also introduces a source trust problem: retrieved context may be useful, irrelevant, or even misleading. Existing RAG systems often apply a fixed trust policy toward retrieved evidence, which can either over-trust incorrect context or underuse context when the user explicitly asks for context-following behavior. Therefore, we propose Intent-Guided Decoding (IGD), a framework that arbitrates between retrieved context and parametric memory according to user intent. IGD uses answer-level filtering and token-level correction to steer the final decoding trajectory between retrieved context and parametric memory. We evaluate IGD on three faithful QA benchmarks and three factual-conflict benchmarks across five LLMs, IGD substantially improves factual recovery, achieving gains of up to 65.4 percentage points on factual-conflict benchmarks over Direct RAG, while preserving or improving strict context-following behavior, this findings highlight the importance of balancing factuality and faithfulness in RAG. 

---
# D2-ScaleAgent: Dual-Dimensional Scaling for Long Document Understanding 

**Authors**: Hao Zhang, Longrong Yang, Lunhao Duan, Ziyang Wang, Qing-Guo Chen, Shanshan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.16417)  

**Abstract**: Multi-modal retrieval-augmented generation (RAG) is a key technique for visually rich long document understanding. Existing multi-modal RAG methods are progressively advancing toward multi-agent systems: they first retrieve relevant pages based on a query, and then iteratively understand information within those pages. However, these methods typically rely on fixed workflows and lack the ability to dynamically scale computation at test time, often leading to insufficient evidence. To address this, we propose D2-ScaleAgent, an agentic framework that introduces a dual-dimensional scaling paradigm for retrieval and reasoning. The core of D2-ScaleAgent is a Verifier agent-driven dynamic routing loop based on the intrinsic difficulty of the query, centered around a continuously updated evidence bank that serves as the agent's dynamic working memory: when retrieval needs to be expanded, the agent routes outward (retrieval scaling), decomposing the query into attributes and performing parallel page retrieval, followed by adaptive pruning to ensure comprehensive evidence coverage. When fine-grained reasoning is required, the agent routes inward (reasoning scaling), dynamically selecting sub-agents with varying granularity and count to extract evidence from pages. Finally, D2-ScaleAgent achieves logical closure over the evidence chain. Extensive experiments demonstrate that D2-ScaleAgent is effective on long and visually rich document benchmarks like MMLongBench-Doc, LongDocURL, etc. 

---
# Counting Documents Is Not Counting Text: Unit Bias in Web-PDF Corpus Statistics 

**Authors**: Luca Foppiano  

**Link**: [PDF](https://arxiv.org/pdf/2608.16390)  

**Abstract**: PDF corpora advertise their size in tokens but compute every rate they publish (coverage, OCR routing, re-fetch recovery, language mix) per document, and none decomposes its token total. The two units diverge sharply. On CC-MAIN-2021-31-PDF-UNTRUNCATED (7.9M web PDFs, 32.6B tokens), 3.02% of text-bearing documents hold half the tokens (Gini 0.807); documents over 50 pages are 5.00% of the corpus but 53.53% of its text. The PDFs produced by a TeX{} toolchain are 1.66% of documents and 4.05% of the text. The clearest casualty is Common Crawl's truncation cap: it affected 23.06% of documents and 63.08% of the text. Reconstructing the truncated files and extracting both versions, two widely used libraries recover 11.4% and 1.4% of that text; between 72% and 97% of affected documents yield nothing; roughly 55--62% of the corpus's text is lost. Under the 5 MiB cap adopted in March 2025, 30.19% of tokens would still be truncated, and recovery on those documents rises only from 3.3% to 13.2%. We recommend that corpus statistics be reported in both units: documents and tokens. 

---
# Mint-Agent: Introducing Finance-Native Agentic Foundation Models 

**Authors**: Mint-Agent Team, B. Zhang, Yaze Geng, Lei Tang, Yaoyang Yi, Zonghan Wu, Yifan Hu, Kun Wang, Qingsong Wen, Yilei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2608.16386)  

**Abstract**: Financial agents must do more than recall domain knowledge: they must be both reliable, executing precise operations over grounded evidence, and executive, sustaining long-horizon research whose conclusions remain auditable. We present Mint-Agent, a family of finance-native agentic models designed around these two scales of financial intelligence. Mint-Agent is built upon three pillars: data, harness, and algorithm. Our data engine constructs clean, specialized tasks for atomic financial capabilities and long-horizon agentic execution from real-world financial sources. MintHarness enables stable interaction with open-ended environments and maintains auditable evidence trails across extended research trajectories. Our training recipe combines SFT, critical-step OPD, and RLVR to develop separate financial reasoning and agentic execution experts, which are then unified through model merging and multi-teacher on-policy distillation into compact, general-purpose financial agents. This pipeline yields two flagship models, Mint-Cu (9B) and Mint-Ag (27B). Across professional financial benchmarks, our models demonstrate two defining strengths: (1) Reliability: Mint-Ag achieves 98.33% on RFC-Bench, surpassing GPT-5.6-Sol and Claude-Opus-4.8 by 3.66 and 3.00 points; and (2) Executability: Mint-Cu reaches 69.86% on FinSearchComp T2, outperforming Agents-A1-35B and Nex-N2-mini by 22.83 and 12.78 points, while Mint-Ag achieves 76.00% and 60.49% on FinanceAgentBench v1.1 and v2, respectively. These results establish a path toward trustworthy financial intelligence in which domain expertise, long-horizon execution, and auditable evidence are jointly engineered as a unified foundation for frontier agentic models. 

---
# Unadapted Multilingual ASR on a Garrusi Kurdish Evaluation Set: A Common-Reference Staged Normalization Analysis 

**Authors**: Hiwa Asadpour  

**Link**: [PDF](https://arxiv.org/pdf/2608.16379)  

**Abstract**: Evaluating speech recognition for a Kurdish variety written in a Latin field orthography, using a model that outputs Arabic script, creates a measurement problem before a modelling one: direct scoring treats writing-system differences as recognition errors. Jointly normalizing reference and hypothesis avoids this, but also changes reference tokenization, mixing agreement gains with a change in the scoring denominator. I evaluate MMS-1B-all with the Central Kurdish (ckb) adapter, used as released without adaptation, on 1,722 Garrusi questionnaire segments from five speakers (9,763 reference word tokens; 117.9 minutes). I use a common-reference design: the reference is folded once and fixed at 9,763 tokens, while only the hypothesis representation varies. The raw Arabic-script hypothesis scores 111.70% WER and 100.92% CER, with zero exact word matches. Latin transliteration gives 102.36% WER and 57.89% CER; folding it into the reference's reduced orthography gives 97.85% and 51.20%. Thus RAW-to-FOLDED reduces measured WER by 13.85 points and CER by 49.72 points; folding alone accounts for 4.51 and 6.69 points. Substantial error remains: 14.53% of reference tokens are exact matches, edits are substitution-dominated, and per-segment WER is higher for shorter segments. A Southern Kurdish fine-tuned system (aranemini/southern-kurdish-asr), scored under the same design, performs worse on every speaker (1,703 segments), with 109.56% WER and 55.85% CER. However, 12,330 output characters fall outside the folding table, so these rates must be recomputed against the corrected fixed reference. The MMS output also contains 613 unconverted or unmapped characters, showing that part of the residual error reflects scoring-pipeline limits rather than recognition alone. I will release the fixed reference and segment-level results, subject to source-corpus sharing terms, to support independent checking. 

---
# HalluTracer: Hallucination Detection via Depth-Averaging Truth Signals 

**Authors**: Zhihao Guo, Zonghan Wu, Huan Huo, DaYong Ye, Junwei Zhang, Weiran Yao, Zhiwei Liu, Qingsong Wen, Yilei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2608.16353)  

**Abstract**: Even well-aligned large language models confidently generate factually incorrect text, making hallucination a persistent reliability risk in high-stakes deployments. These models nonetheless carry linearly separable truthfulness signals in their internal representations. Existing white-box detectors, however, collapse this evidence to isolated components or a single depth, discarding discriminative information distributed across the full forward pass. We introduce HalluTracer, a detection framework that reads and aggregates truthfulness evidence across every layer of the forward pass before the model emits any answer token. A geometric analysis reveals that the per-layer signals are weakly correlated, so that simple depth averaging suppresses layer-specific noise and captures nearly all linearly accessible information. Across six open-source language models and five hallucination benchmarks, HalluTracer consistently outperforms matched white-box baselines, with gains ranging from one to fourteen points. Collectively, our work recasts hallucination detection from a layer-selection problem into a depth-aggregation problem governed by the geometric sparsity of the truthfulness signal. 

---
# Architecture-Dependent Causal Transfer of Activation States Across Large Language Models 

**Authors**: Fernando Cardenas Piepereit  

**Link**: [PDF](https://arxiv.org/pdf/2608.16347)  

**Abstract**: Direct communication between AI systems relies on natural language as an intermediate layer, incurring encoding/decoding overhead, token cost, and latency. We ask whether internal activation states can instead be transferred causally between different large language model (LLM) architectures via a learned projection, evaluated at three levels: representational similarity, cross-model retrieval from projected states, and end-to-end causal transfer via activation injection during generation. Using four architecturally diverse open-weight models (Qwen2-0.5B, Phi-3-mini, Mistral-7B, FLAN-T5-base), we find that representational alignment in trained models exceeds a random-initialization null baseline and is best captured by a rank-based metric (mutual k-nearest-neighbour alignment), more robust to activation-magnitude outliers than centered kernel alignment (CKA) or Procrustes analysis. A learned projection network retrieves the correct target-model representation from a held-out set well above chance for the three causal decoder-only model pairs (45-50% top-1 accuracy vs. 5% chance) but at chance level for the encoder-based FLAN-T5. Injecting projected activations into a target model during generation produces a statistically significant, pre-registered causal effect on retrieval-based output similarity for only one of the three decoder-only pairs (Qwen2-0.5B to Phi-3-mini: 23.3% vs. 0.0% under negative control, p=0.047, FDR-corrected); the two pairs targeting Mistral-7B show no such effect despite comparable representational alignment at the hidden-state level. We interpret these results as evidence for causal transfer of the representational vehicle, not of meaning, and conclude that end-to-end activation-state transfer between LLMs, as currently implemented, is architecture-dependent rather than universal. 

---
# IndicQE-APE: A Benchmark for Quality Estimation and Automatic Post-Editing for Indic Languages 

**Authors**: Diptesh Kanojia, Archchana Sindhujan, Sourabh Deoghare, Daria Sokova, Shenbin Qian, Girish Koushik, Tharindu Ranasinghe, Constantin Orăsan, Chrysoula Zerva, Ricardo Rei, Frédéric Blain, André F. T. Martins, Marco Turchi, Matteo Negri, Rajen Chatterjee, Anoop Kunchukuttan, Mitesh M. Khapra, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2608.16344)  

**Abstract**: Indic quality estimation (QE) and automatic post-editing (APE) data is spread across separate releases, so no single resource supports training and evaluation across tasks and language pairs on one footing. We consolidate the WMT 2020--2024 shared-task lineage with an extended English--Malayalam resource into \indicqe: $126{,}754$ instances over nine directional pairs, with up to four label types aligned on the same segment, a direct assessment, a human post-edit, word-level OK/BAD tags and an error explanation, and a test set stratified over four difficulty axes. On it, we benchmark six prompted LLMs and three COMET metrics on segment-level QE, and three systems on APE. Two of the axes are defined partly on the direct assessment and select a compressed slice of it, so each axis is compared against a control drawn from the same language pair with the same score distribution. Only one survives that control: segments whose holistic and token-level quality signals conflict are ranked worse than equally-scored segments of the same language, for all nine systems and all seven pairs that carry the axis. Annotator disagreement, which looks second-hardest without the control, has no effect with it. Few-shot prompting costs every model $\leq$ $3.4$B both correlation and output-format compliance. Within-language accuracy does not make scores comparable across pairs: of the three trained metrics, the one with the best within-language correlation loses most when the pairs are pooled. The benchmark and code will be released. 

---
# Step-Level On-Policy Distillation: Interpolating Between On-Policy Distillation and Supervised Fine-Tuning 

**Authors**: Changhui Sun, Lanbo Liu, Hang Lei, Tong Ling, Jiahang Xie, Zhiyong Zheng, Yujia Wang, Hao Liu, Feng Xiao, Lu Liu, Yanlong Du, Zifeng Cheng, Ziwei Jiang, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2608.16333)  

**Abstract**: On-policy distillation (OPD) aligns a student model with a teacher's logit distribution on student-generated trajectories. This approach has achieved strong empirical gains and can often surpass conventional off-policy distillation with substantially less data. However, standard token-level OPD can provide only fragmented corrections along an erroneous student trajectory and cannot unfold a complete and correct repair path. Motivated by this limitation, we propose \emph{Step-Level On-Policy Distillation} (SOPD), which combines the long-horizon correction of supervised fine-tuning (SFT) with the on-policy advantage of OPD to provide step-level supervision over complete student-generated trajectories. We show that, at different limits of step length, SOPD reduces to SFT or approximates OPD. Compared with SFT, the teacher responses in SOPD are conditioned on student trajectories and therefore align more closely with student-visited states; compared with OPD, SOPD provides longer-horizon corrections rather than fragmented token-level guidance. Across both reasoning and agent tasks, SOPD substantially outperforms conventional SFT and OPD. For example, on ALFWorld, SOPD improves the average success rate by 13.4 points over Vanilla OPD. We hope this work offers a new perspective for future research on distillation methods. 

---
# FTA-Mem: Fact-Time-Affect Anchored Memory for Low-Density Long-Term Dialogue 

**Authors**: Chang Liu, Shuyi Zhang, Changsheng Ma, Yongfeng Tao, Minqiang Yang, Bin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.16303)  

**Abstract**: Long-term emotional-support agents require memory mechanisms for personalized understanding across sessions. However, emotional-support dialogue is often low-density: turns are incomplete, evidence is scattered, and user states evolve over time. Existing memory methods usually rely on fixed units, such as turn-level notes or session summaries, which may lose details or introduce redundant noise. We propose FTA-Mem, a structured memory framework for low-density long-term dialogue. FTA-Mem uses Boundary-preserving Window Segmentation (BWS) to form coherent situation fragments, and constructs Fact-Time-Affect Memory Units (FTA Units) that jointly encode factual content, temporal grounding, and affective context. Retrieved units are then synthesized into structured context for answer generation. Experiments on ES-MemEval and LoCoMo show that FTA-Mem improves overall long-term memory question answering across benchmarks with different information-density characteristics. On ES-MemEval, FTA-Mem achieves 0.3871 F1 and 0.6668 BERTScore. Further analysis shows that situation-level FTA construction better balances evidence preservation and construction cost than coarse session-level or overly fine-grained turn-pair construction, providing an effective granularity trade-off for long-term dialogue memory. 

---
# Executable Code Knowledge: Code as a Native, Validation-Carrying Knowledge Representation for AI Coding Agents 

**Authors**: Xueping Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.16295)  

**Abstract**: AI coding agents need more than relevant snippets: they need business semantics, validation evidence, relations, and assurance that their context is current. Existing systems usually infer or externalize this knowledge through retrieval, summaries, graphs, rules, or reverse specifications. We investigate a complementary representation in which selected code units directly carry agent-usable knowledge. We introduce Executable Code Knowledge (ECK) and define an Executable Code Knowledge Unit (ECKU) as a source-bound object combining stable identity, semantics, executable behavior, contracts, evidence, relations, provenance, validation state, and a query interface. Our Python prototype supports code-local authoring, manifest export, evidence execution, exact changed-line impact, freshness checking, and agent-facing projections. Across three real Python repositories and 26 controlled patch tasks, direct ECK provides executable test coverage for 11/11 evidence-bearing tasks and exact selectors for 9/11; hiding declared evidence reduces exact recovery to 1/11 (paired exact McNemar p=0.0078). ECK-derived rules recover 11/11 exact selectors, showing that rules are effective delivery artifacts while ECK supplies source binding, validation state, impact, and freshness. Exact changed-line impact matches independently authored labels on all 26 patches (12 unit links; precision, recall, and F1 all 1.000). AST-bounded fingerprints classify 50 positive changes and 17 unrelated same-file controls correctly, whereas static rules snapshots detect none of the 50 stale cases. Model-backed patch-review and cross-layer studies measure projection fidelity rather than independent impact discovery. These results support a hybrid architecture: retrieval for coverage, ECK for source and evidence governance, and projections for delivery. 

---
# Clause Encounters of the Third Kind: Can LLMs Replace Language Teachers? 

**Authors**: Kristina Šekrst, Ana Kovačić  

**Link**: [PDF](https://arxiv.org/pdf/2608.16286)  

**Abstract**: While various organizations now actively encourage LLM use in classrooms, we still lack rigorous, systematic evaluations of how well these models actually perform the fundamental tasks of language pedagogy. This paper examines whether state-of-the-art LLMs can deliver the kind of corrective feedback and methodological explanations that language learners need. The study tests multiple large language models on their ability to identify, correct, and explain common learner mistakes in English, by systematically varying model parameters to investigate how these technical adjustments affect output quality, pedagogical clarity, and consistency, along with using retrieval-augmented generation to query methodological data. The evaluation employs automated metrics (GLEU, BERTScore) but also human expert judgments to capture dimensions that purely computational measures miss: linguistic nuance, cultural sensitivity, and instructional appropriateness. While models demonstrate impressive surface-level correction abilities, their explanations often lack the terminological and domain knowledge that effective language teaching requires, suggesting that current enthusiasm for AI-assisted language learning may be outpacing our understanding of these systems' actual pedagogical competence. 

---
# Domain-Agnostic Neural Topic Modeling with Contextual Token-Level Semantic Graph Representation 

**Authors**: Seung-Won Seo, Won Ik Cho, Yongmin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2608.16269)  

**Abstract**: Recent advances in neural topic models with pre-trained language models (PLMs) have achieved strong performance by leveraging general-domain pre-training, yet their topic interpretability often degrades on specialized corpora. This limitation primarily stems from the geometry of the embedding space, where domain-specific terms unseen during pre-training collapse into an indistinguishable region, and neither domain-specific re-training, word-level graph enrichment, nor parameter-efficient fine-tuning can restructure this space without inheriting the capacity ceiling of the underlying encoder. Our key insight is that a learnable graph layer operating on token-level PLM embeddings can acquire corpus-specific semantic structure that the frozen encoder lacks, because token-level graphs preserve document-local context that word-level representations discard and joint optimization with the topic objective reshapes embedding geometry directly from target-domain evidence. We instantiate this insight as DARTopic, a domain-agnostic framework that constructs token-level semantic graphs from frozen PLM embeddings and jointly trains a GNN encoder with topic inference. Across three benchmarks spanning general, biomedical, and legal domains, DARTopic consistently outperforms strong baselines in topic coherence and document clus- tering without any encoder fine-tuning, while demonstrating robustness to PLM choice and favorable runtime efficiency over fine-tuning based alternatives. 

---
# STAIR: Semantic-Temporal Automaton for Interpretable Reasoning in Temporal Question Answering 

**Authors**: Xinlong Dai, Jinchuan Zhang, Lei Gao, Xinzhe Hu, Yuefeng He, Hui Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.16224)  

**Abstract**: By leveraging large-scale pretraining, LLMs can interpret diverse temporal expressions and question formulations without task-specific training. However, existing prompt-based neuro-symbolic systems continue to rely on LLMs for both semantic interpretation and exact temporal inference. Consequently, discrete decisions regarding intervals, time anchors, and ordered states remain vulnerable to probabilistic errors and difficult to verify. We present STAIR, a \textbf{S}emantic-\textbf{T}emporal \textbf{A}utomaton for \textbf{I}nterpretable \textbf{R}easoning. STAIR separates semantic interpretation from precise temporal inference: an answer-free LLM adapter maps complex question formulations to normalized temporal intents, while a deterministic temporal automaton with finite control and guarded transitions executes the corresponding policies over canonicalized evidence. Following a rule-first design, STAIR resolves standard questions without invoking an LLM and applies semantic adaptation only when the rule path fails to produce an executable intent. This approach reduces free-form reasoning, making temporal decisions verifiable and interpretable. Specifically, guarded execution supports precise point-time containment and before/after selection, while semantic adaptation handles non-exact intervals and time-anchored queries. Across the TimeQA-Easy, TimeQA-Hard, TempReason-L2, and TempReason-L3 datasets, STAIR consistently outperforms strong baselines in the TQA task using matched model settings, achieving average F1 improvements of 16.57\% and 3.10\% when utilizing the Qwen2.5-7B and GPT-4o-mini models, respectively. Furthermore, ablations and diagnostic analyses demonstrate that STAIR excels at handling both boundary-sensitive and order-sensitive queries, while its guarded execution and semantic adaptation ensure precise point-time reasoning and inexact intervals, respectively. 

---
# LENS: In-Context Search via Latent Evidence Exploration over Dynamic Raw Documents 

**Authors**: Xingjun Wang, Gongsheng Li, Qi Fan, Yunlin Mao, Luyan Su, Yingda Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.16185)  

**Abstract**: LLM agents increasingly answer questions over dynamic raw-document collections, where files may change before preprocessing, and relevant evidence (spans, sections, pages, or tables) is query-dependent. Existing retrieval-augmented approaches pre-materialize evidence via fixed chunking, embeddings, or persistent indexes: effective for lookup, yet costly, stale-prone, and committed to a granularity before the query is known.
We formulate in-context search as Budgeted Evidence Localization over a latent evidence space induced by dynamic raw documents and propose LENS (Latent Evidence Exploration and Search), an index-free framework. Instead of pre-materializing the evidence space, LENS maintains a query-conditioned belief over candidate units, iteratively selecting candidates via complementary lexical, local, and exploratory proposal policies, updating the belief via an LLM relevance oracle, and narrowing toward high-posterior regions under a controllable budget. Evidence is consolidated into compact, source-grounded regions of interest and compressed into self-organizing knowledge clusters reused across related queries.
On a controlled 500-question evaluation with matched corpus snapshots, LENS reaches 62.4% exact match and 84.8% evidence recall vs. 65.2% exact match but 50.4% evidence recall for a ReAct-style baseline. Across scales, LENS gives the strongest supporting-fact localization and answer grounding. On a fixed 150-question fullwiki subset over the raw Wikipedia dump with zero indexing, LENS and ReAct are nearly tied in official answer quality (43.3% vs. 42.7% EM), with LENS grounding more answers in retrieved evidence (84.0% vs. 70.7%). A no-retrieval Closed-Book reference highlights the contribution of model memory. LENS is query-ready after corpus changes, needs no preprocessing or persistent index, and preserves source-grounded evidence localization throughout. 

---
# QUMem: Personalized Memory for Query-Conditioned User-State Inference in LLM Agents 

**Authors**: Heng Wang, Yifei Li, Lingling Zhang, Pengyu Li, Xinyu Che, Xinyu Zhang, Zesheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.16168)  

**Abstract**: Large language model (LLM) agents increasingly use external memory systems to support personalization by drawing on long and evolving interaction histories, in which user preferences may be distributed across time, change with context, and conflict with earlier evidence. However, existing systems face three limitations: fixed-turn, fixed-token, or session-based boundaries can mix unrelated dialogue or split an event from its causes, decisions, and outcomes; storing multiple pieces of user information from the same interaction as a single memory binds together items that serve different functions and should be independently retrievable; and treating the current task as a single top-$k$ retrieval query can return fragments that are individually relevant but fail to jointly capture preference evolution, temporal validity, and contextual applicability. We introduce \textsc{QUMem}, a structured memory framework for query-conditioned user-state inference. \textsc{QUMem} first segments interaction histories into variable-length episodes according to semantic continuity, then decomposes each episode into independently retrievable factual, preference, and transferable insight memories while preserving temporal positions and source evidence. At inference time, three sequential agents identify task-specific information needs, plan multi-query retrieval over the typed memory stores, and jointly infer a temporally and contextually valid user state for downstream response generation. \textsc{QUMem} achieves state-of-the-art performance on both PersonaMem and KnowU-Bench, demonstrating the effectiveness of query-conditioned user-state inference for long-term personalization. 

---
# HyperSkill: Self-Evolving LLM Agents via Hypergraph-Structured Skill Memory 

**Authors**: Ruiyao Xu, Tiankai Yang, Wei-Chieh Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.16114)  

**Abstract**: As agentic tasks grow in complexity, LLM agents increasingly rely on experiential memory to reuse procedural knowledge across tasks. Effective memory design must jointly address what to store, how memory is structured and retrieved, and how memory evolves. Existing systems tackle each only partially: they store trajectories, insights, or workflows as isolated entries, discarding compositional relationships among subtasks and reusable skills; retrieve by flat embedding similarity that ignores relational signals; and maintain memory without leveraging its relational structure. We propose HyperSkill, a hypergraph-based memory framework that jointly improves all three. HyperSkill represents memory as a hypergraph with two node types, subtask steps and reusable skills, where each hyperedge links the subtasks and skills from a single trajectory. Dual-path retrieval queries both subtask and trajectory levels, ranking skills by co-occurrence across retrieved trajectories. Periodic structure-informed maintenance prunes low-utility nodes and merges redundant skills via quality-weighted propagation. Across xBench, GAIA, and WebWalkerQA with GPT-4o and Qwen3-30B-A3B, HyperSkill outperforms ten memory baselines, yielding gains of up to +11.51 on GAIA and +11.18 on WebWalkerQA. 

---
# Skill2Query: Exploiting Skill Structure to Generate Pseudo-Queries for Agent Skill Retrieval 

**Authors**: Lihui Ding, Zihan Guo, Bingwei Lu, Chenyu Zhou, Yuanjian Zhou, Weinan Zhang, Jianghao Lin, Dongdong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2608.16071)  

**Abstract**: Pseudo-query generation can alleviate the supervision bottleneck for agent skill retrieval, but existing document-level approaches typically leave the rich internal relations among capabilities, parameters, and usage examples implicit. As a result, generated queries may be topically relevant to a skill while lacking capability grounding and parameter consistency, raising the question of whether explicitly exploiting a skill document's internal structure can produce more effective retrieval signals. We therefore propose Skill2Query, a framework that first parses a skill document into a Skill Knowledge Graph and then generates pseudo-queries through a three-stage process including style mimicking, query template generation, and parameter filling. The generated queries can be used for offline index augmentation, online query expansion, and retriever training. Four benchmarks (TheoremQA, LogicBench, ToolQA, and CHAMP) are used to evaluate Skill2Query with large-scale skill candidate pools across multiple downstream applications, including skill retrieval, retriever training, and end-to-end agent execution. Using nearly 30K skills across diverse domains, we generate 700K category-diverse pseudo-queries. Skill2Query consistently improves sparse, dense, and skill-routing retrieval, with an average Recall@1 gain of 6.70 percentage points across retrieval settings. Skill2Query-generated training data also achieves the best Recall@1 and nDCG@1 among the evaluated generation baselines. Further evaluations with multiple LLM backends demonstrate that improved skill retrieval translates into higher agent task success rates. Code and resources are available at this https URL. 

---
# CAPO: Constraint-Aware Prompt Optimization for LLM Agents 

**Authors**: Victor Ye Dong, Reid Pryzant, Yi Liu, Jian Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.16068)  

**Abstract**: Large language models (LLMs) are increasingly deployed as agents that rely on system prompts to use tools and complete tasks. Such deployments impose distinct operational requirements, including appropriate tool use, concise prompts and solution paths, and compliance with safety and formatting policies. For many practitioners, however, assembling domain-specific supervised data to post-train models to meet these requirements is infeasible. We introduce CAPO (Constraint-Aware Prompt Optimization), a primal-dual method that combines pool-based rewrites with adaptive constraint weighting to optimize system prompts under explicit operational constraints. Across agentic benchmarks, CAPO more reliably reaches empirically feasible operating points while improving task performance. CAPO also generalizes beyond agentic settings, achieving strong results on assistant-style evaluations with output-format and safety/privacy constraints. We further introduce DCAPO (Dynamically Trained CAPO), which trains a feedback- and dual-conditioned rewriter with pool-based GRPO while keeping the task agent frozen. Across task agents of different sizes, DCAPO produces a feasible prompt in every evaluated domain and matches or improves the task accuracy achieved by the evaluated baselines. A surrogate analysis characterizes how finite-pool and discrete-rewrite errors enter the inexact primal-dual procedure. 

---
# DuplexGen: Decoupling Content, Timing, and Acoustics for Synthetic Dialogue Speech 

**Authors**: Pengcheng Wang, Sheng Li, Jiyi Li, Takahiro Shinozaki  

**Link**: [PDF](https://arxiv.org/pdf/2608.16053)  

**Abstract**: Synthetic conversational speech has become an important resource for developing and evaluating conversational speech systems. However, existing dialogue synthesis pipelines typically generate dialogue content first and then insert interruptions, overlap, and backchannels using handcrafted markers or timing rules, making conversational timing prescribed rather than interaction-driven. We present DuplexGen, a dialogue synthesis framework that explicitly decouples content, timing, and acoustics. An LLM first generates the dialogue script, and then two full-duplex conversational models perform the script while listening to each other in real time. This allows conversational timing to emerge naturally while preserving the scripted content. Finally, a high-fidelity text-to-speech model re-renders the interaction without altering its timing. As a demonstration of the proposed framework, we construct a patient--clinician conversational speech corpus with construction-time annotations, including word timestamps, speaker activity, overlap regions, and interaction events. Experimental results show that the proposed framework produces conversational dynamics closer to real dialogue than conventional stitching-based synthesis. 

---
# $R^3$-Bench: LLMs Struggle with Resource-Rational Reasoning under Shared Budgets 

**Authors**: Peisong Wang, Zhiwei Ma, Bowen Liu, Feixue Liu, Aochuan Chen, Chenyi Zi, Hongchuan Zeng, Yuhan Li, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.16033)  

**Abstract**: In cognitive science, resource rationality asks how an agent should allocate limited computation to maximize expected value. Most reasoning and agent benchmarks use independent per-task budgets; existing shared-budget studies do not calibrate suite performance against the same model's demonstrated single-problem competence. We introduce $R^3$-Bench, which evaluates six-problem suites under shared budgets across mathematics, competitive programming, and abstract reasoning in tool-free and agentic settings. Matched single-problem response curves define an offline empirical oracle over observed successes. Across 72 main-table cells for six models, the oracle mean matches or exceeds the contest mean in all cells and is strictly higher in 71. Under moderate tool-free pressure, equal-allocation replay also exceeds contest performance for four of six models. Trajectory diagnostics reveal limited strategy updating and pressure-dependent failure patterns. In a three-model diagnostic under strong agentic pressure, at least one fixed scheduler exceeds the contest mean in six of nine cells, but no policy dominates across domains. These results expose a persistent gap between demonstrated competence and shared-budget realization. 

---
# ReRef-3D: A Benchmark for Spatial Referring Expression-Guided 3D Scene Rearrangement 

**Authors**: Mary Lynn Martin, Yifei Zhang, Martha Palmer, Maria Leonor Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2608.16011)  

**Abstract**: We introduce ReRef-3D, a benchmark for language-guided placement in 3D scenes. It contains 33,826 instructions across 998 CLEVR-derived scenes, spanning 16 placement families and direct, one-hop, and two-hop references. Each instruction must be resolved into a valid new placement position. Given that an instruction defines a region of acceptable placements rather than one coordinate, our evaluation inserts a prediction into the scene, recomputes relations, and tests relation satisfaction and physical validity. Each instruction also includes a verified naturalized rewrite. After fine-tuning, LLaVA-3D, 3D-LLM, and PlaceIt3D produce valid placements for 68.3%, 31.6%, and 22.4% of instructions, respectively. Across models, relation satisfaction surpasses physical validity, relations such as nearest and between are the most difficult, and phrasing has minimal effect on performance. 

---
# From Sequence to Structure: Relational Uncertainty Propagation for LLM Agents 

**Authors**: Zhengzhao Ma. Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.16002)  

**Abstract**: Reliable uncertainty quantification (UQ) is essential for deploying large language model (LLM) agents in complex interactive environments. Existing UQ methods largely rely on local signals, such as token probabilities, predictive entropy, or per-step confidence, and therefore overlook the long-range dependencies through which errors accumulate across an execution trajectory. As a result, they may fail to identify agent failures whose causes originate several reasoning or interaction steps before the final answer. We propose RUPA (Relational Uncertainty Propagation for Agents), a trajectory-level UQ framework for LLM agents. RUPA represents an execution history as a directed trajectory graph in which reasoning states, tool interactions, and environment feedback are nodes connected by temporal and semantic dependency edges. It then propagates uncertainty over this graph to capture how execution risk accumulates and transfers across interaction steps. The propagated signal is combined with trajectory-level behavioral features and goal-alignment information to produce a confidence estimate for the full agent trajectory. We evaluate RUPA on representative agent benchmarks, including $\tau$-2, Terminal-Bench-2, and GAIA, using 6 open-source LLMs spanning multiple model families. Experimental results show that RUPA consistently outperforms existing UQ methods by providing more accurate uncertainty estimates, enabling earlier failure detection, and improving uncertainty-guided agent execution across diverse agent tasks. These results demonstrate that explicitly modeling relational dependency is crucial to reliable UQ for long-horizon LLM agents, providing a practical foundation for trustworthy agent execution. 

---
# Whose Gold? Annotator-Pool Disagreement Is Large at the Item Level, and Hidden by Small Leaderboards 

**Authors**: Anik Jha  

**Link**: [PDF](https://arxiv.org/pdf/2608.15980)  

**Abstract**: Preference benchmarks are built by hiring annotators, and the identity of those annotators is treated as an implementation detail. We measure what that detail buys. On the 2,885 MultiPref items where both pools are internally unanimous, so no tie-breaking convention is consulted at all, expert and crowd annotators assign a different majority label to 23.6% and name the opposite winner on 9.2%; on the 246 comparably unanimous MT-Bench cells, benchmark authors and recruited experts differ on 30.5% and reverse on 8.5%. Yet on both corpora the resulting model leaderboards are bit-identical: Kendall tau = 1.00 with zero of six models displaced.
That invariance is far weaker evidence than it looks, and we quantify how weak. Switching pools moves a model's win rate by 1.9pp (SD), one adjacent pair in our own leaderboard sits 0.8pp apart and had a 38% chance of swapping, and an item-level bootstrap displaces at least one model in 28% of resamples. The observed zero is the common outcome, not a property of aggregation: on the same measured perturbation, a ten-model leaderboard is displaced with probability 0.86 and a twenty-model leaderboard with probability 0.9997. Reporting a six-model leaderboard is safe; the safety does not generalise, and everything that consumes labels per item is not safe at any size. We make the distinction precise, show that a widely used dataset's stated assumption of no intra-group annotator variability is false, and show that an LLM judge tracks the crowd pool over the expert pool on all three models we test, including one from a different vendor. All code, per-call outputs, and pre-registered decision rules will be released upon acceptance. 

---
# LLMs Get Smarter from Targeted Synthetic Multilingual Data 

**Authors**: Ishika Agarwal, Arkajyoti Charaborty, Tanner Sorensen, Neha Gupta, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2608.15964)  

**Abstract**: Language-specific competency (LSC) is the phenomenon of a language model performing better or worse depending on the language of the prompt. In other words, a language model outputs different (and potentially incorrect) responses to the same semantic query when prompted in different languages. Prior work attributes this to an internal misalignment of semantic representation across languages. Currently, there are two main approaches to address LSC in the literature: (1) routing all queries through English, improving performance, but limiting language expressivity to English; or (2) training on language-balanced data, equalizing model performance across languages, but reducing overall performance. In this work, we take a data centric perspective and introduce HOTFIXR: Hardness Optimized Training data For Improving X-Lingual Reasoning. It is a data generation framework that uses models to probe and learn a student model's multilingual weaknesses, and generates data to mitigate them. HOTFIXR can generate multilingual synthetic training data that can improve multilingual performance. We evaluate on three in-distribution tasks, three out-of-distribution tasks, and four out-of-distribution languages. On average, HOTFIXR (1) improves in-distribution performance by 6.2%, (2) reduces catastrophic forgetting (induced by fine-tuning) on OOD tasks by 3.7%, and (3) on OOD languages by 7.1%. Overall, as many real-world applications requires multilingual LLMs, our work contributes to the efforts of making LLMs multilingually proficient. We will release code upon acceptance. 

---
# SEER: Long-Context Reasoning via Selective Visual-Text Compression 

**Authors**: Jiawei Xu, Zhilin Zhai, Jinrui Fang, Ruohan Xu, Mingfei Lu, Yi Zhang, Guanchu Wang, Tianlong Chen, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2608.15962)  

**Abstract**: Long-context reasoning remains computationally expensive for large language models due to the quadratic complexity of attention over text tokens. Visual-text compression offers a promising alternative by rendering text into images and processing them with vision-language models, often reducing token usage. However, existing approaches apply uniform compression regardless of query relevance, potentially sacrificing precision where detailed extraction is required. We present SEER, a framework that learns to select query-relevant images through visual scanning and retrieve textual content only where needed, combining the efficiency of visual compression with the precision of text-based reasoning. Through supervised fine-tuning on tool-interaction trajectories, SEER learns adaptive tool invocation for selection and retrieval. Experiments on long-context benchmarks show that SEER improves extraction precision through selective text retrieval while retaining average prompt-token savings relative to full-text baselines. On LongBench, SEER achieves 51.11% average accuracy, outperforming the visual-text baseline Glyph-9B by 2.33 points and Qwen3-8B by 3.49 points. Code can be accessed at this https URL 

---
# The Null Token Knows: Reducing Message-Free Hallucination in ASR and NMT 

**Authors**: Kirill Borodin, Vasiliy Kudryavtsev, Ivan Viakhirev  

**Link**: [PDF](https://arxiv.org/pdf/2608.15940)  

**Abstract**: Modern encoder-decoder systems can produce fluent text even when their input contains no recoverable message. We study this failure in ASR and NMT through the models' reserved null tokens, asking whether the score for ending generation already carries a usable abstention signal. Across speech recognizers and translation models, we audit native null-token scores and scalar logit shifts. In Whisper, we additionally probe decoder states and compare supervised row edits with conventional external gates. The evaluated models often expose a useful abstention signal, but stock decoding does not reliably act on it. Raising the null-token score can sharply suppress fabrication, but aggressive intervention also deletes valid speech or shortens legitimate translations. These findings turn the null token into a diagnostic lens on hallucination and motivate evaluating abstention methods by both suppression and deletion costs, rather than by hallucination reduction alone. 

---
# Aborted but Not Forgotten: KV-Cache Retention Breaks Rollback Consistency in Language Agents 

**Authors**: Guijia Zhang, Harry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.15939)  

**Abstract**: Stateful language agents assume a rejected branch can be taken back by clearing it from the application transcript. We show this breaks when the serving session retains key/value (KV) state across the logical abort: the model can continue attending to content the application believes it discarded. We formalize the missing guarantee as rollback consistency: a complete abort must restore the state the model attends, not just the transcript. The key failure is cross-layer: a correct logical rollback need not compose with retained inference state, and the gap can remain invisible to the application. To isolate cache effects from text effects, we introduce a same-token/different-cache audit that holds decision-step tokens identical while varying only whether the cached prefix is stale or rebuilt from committed state. Across seven open-weight families (3.8B-36B), retained KV alone flips a typed protected effect in 25 of 63 audited cells, while attacker tokens are absent from the served request in all 63; rebuilding the cache closes every cell. The channel reproduces in an end-to-end session application, on the default Hugging Face Transformers cache-reuse path, and under LangGraph time-travel, where verified logical rollback can still leave attended KV stale. Susceptibility varies across models, but the underlying attended-state integrity violation is structural. We rule out position and length confounds, generalize across protected effects, policy structures, and a cache-isolated Mixture-of-Experts model, and show that transaction-local cache restoration closes the channel without requiring a global cache flush. All headline results are deterministic and reproducible from released artifacts. 

---
# Token Distribution versus Data Volume: Domain Balancing in Multi-Domain Meeting Summarisation 

**Authors**: Ashima Sood, Bryan Gardiner, Joan Condell  

**Link**: [PDF](https://arxiv.org/pdf/2608.15935)  

**Abstract**: Jointly fine-tuning an LLM on meeting-summarisation corpora of widely varying size raises a question that prior work leaves confounded: when a domain-balanced training mixture helps, is the gain due to the distribution of tokens across domains, or merely to the volume of data seen? We disentangle these factors by constructing balanced and natural (native-proportional) token mixtures at matched token budgets (2-32M) over five English meeting corpora, fine-tuning Mistral-7B with QLoRA, and evaluating per domain. Balancing redistributes quality, improving the data-scarce minority domains at a low cost to the data-rich ones. The trade favours balancing whenever the minority domains matter: their share under proportional allocation is fixed at 1-2% regardless of budget, so matching balanced quality on those domains requires far more total data. We further find that pruning low-value transcript lines removes ~15% of tokens from the conversational corpora at no measurable cost, and that balancing by tokens is not the same as balancing by examples. A two-annotator study of 741 judge-labelled facts validates our fact-level evaluation. Together these results give practitioners a basis for deciding when to balance an imbalanced multi-domain mixture, and on what unit. 

---
# PLSQLBench: Benchmarking LLM Systems for Executable Procedural Database Programming 

**Authors**: Marianne Menglin Liu, Leonid Boytsov, Daniel W. Peterson, Pramuditha Perera, Rongguang Wang, Sai Ashish Somayajula, Syed Hamza Rafique, Rohit Saini, Shubham Pathak, Sujeeth Bharadwaj, Tao Sheng, Graham Horwood, Fahad Shah, Ankan Bansal, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2608.15931)  

**Abstract**: We present PLSQLBench, to our knowledge the first benchmark for evaluating whether LLMs can write executable PL/SQL programs, with correctness measured through execution-based tests. Existing LLM evaluations largely target general-purpose code generation or declarative text-to-SQL, leaving procedural database programming underexplored. PLSQLBench contains 2,865 instances: 2,594 single-turn tasks and 271 multi-turn conversations spanning 978 turns. The benchmark combines complex schema-grounded tasks over enterprise-style Spider 2 databases, simpler schema-grounded tasks derived from Spider, and MBPP-derived procedural problems, covering varying levels of database grounding and procedural complexity. Experiments with eight LLMs reveal recurring difficulties in schema grounding, PL/SQL dialect fidelity, procedural control flow, exception handling, and cross-turn consistency. Tool-augmented LLM agents improve performance on several schema-grounded evaluations, although substantial gaps remain. These results highlight procedural database programming capabilities not directly assessed by conventional code generation or text-to-SQL benchmarks. Our code is available at this https URL. 

---
# When Less Is Enough: Context Selection and Prompting Strategies for Bengali News Headline Generation 

**Authors**: Muhammad Ashad Kabir, Kawsar Ahmed, Md. Osama  

**Link**: [PDF](https://arxiv.org/pdf/2608.15879)  

**Abstract**: Large language models (LLMs) have shown strong performance in text generation tasks, yet their effectiveness on headline generation remains sensitive to how input context is selected and presented. In this work, we investigate Bengali news headline generation as a document-level generation task that requires effective selection and presentation of salient contextual information from long-form articles. Using Gemini-2.0-Flash, Llama-3.3-70B, and GPT-4o, we systematically study the effects of context selection, prompting strategies, and in-context learning (i.e., few-shot) on the quality of headline generation. Our experiments show that providing the full article does not necessarily improve performance; instead, using selected lead paragraphs of the article can maintain, and in some cases improve, headline generation quality. We further compare Bengali Native Prompting (BNaP) and Cross-Lingual Prompting (XLP), and examine how each interacts with context-enriched prompt templates incorporating auxiliary contextual cues. Results demonstrate that prompting strategies substantially influence generation quality: XLP often yields stronger performance, particularly when combined with contextual enrichment, but its benefits are model-dependent. Additionally, few-shot prompting substantially improves Gemini, with most of the gain obtained from a single demonstration, whereas Llama shows limited benefit from additional examples. Overall, our findings highlight that effective Bengali news headline generation depends more on context relevance and prompt design than on increasing input length, offering practical insights for multilingual and low-resource LLM applications. 

---
# MicroVerse: An Instrument for Measuring Self-Authored Identity Drift in Long-Horizon Multi-Agent Language-Model Simulations 

**Authors**: Sky Ng, Brihi Joshi, Ishan Gupta, Shirley Huang, Zonglin Di, Yun Shen, Qianfeng Wen, Yifan Simon Liu, Ruoqi Gao, Yilan, Zhiwei Zhang, Muhammad Ahmed Mohsin, Yucheng Lu, Xiaoyi Liu, Heming Liu, Qianyu Zhu, Hanwen Xing, Zhengyang Shan, My Chiffon Nguyen, Guanghui Min, Jianheng, Yunze, Xiao, Keyang Xuan, Hannah Collison, Jintao Huang, Jiatong Li, Sankalp Jajee, Yunhan Zhao, Bing Hu, Xupeng Chen, Binghang Lu, Weihang Xiao, Aravind Mohan, Bolun Sun, Yunshu Wu, Yuanda Xu, Runyu Zhang, Zheyuan Deng, Xinchen, Dianzhuo Wang, Yijun Wang, Yixuan He, Koutian Wu, Cheng Cheng, Xiaomin Li, Yuexing Hao  

**Link**: [PDF](https://arxiv.org/pdf/2608.15844)  

**Abstract**: Long-horizon, multi-agent language model (LM) simulations are widely proposed for studying social behavior, yet instruments to measure whether persona-conditioned agents maintain identity fidelity under sustained pressure are lacking. We present MicroVerse, a behavioral-science instrument that measures identity drift in generative agents. Agents carry an immutable "soul file" (core values, moral boundaries, personality, goals) and inhabit a resource-scarce 50 x 50 environment where water is a non-respawning survival constraint. Scarcity is operationalized via a per-tick existence-cost gradient. The eight-verb action space maps directly to moral boundaries (trade, talk, attack, scavenge). Using a three-layer memory architecture, agents periodically revise a mutable current identity against their immutable original soul via importance-triggered reflection. To mitigate survivor bias, MicroVerse decouples measurement from behavior using uniform longitudinal engine snapshots every N ticks alongside a forced-end snapshot of all living and dead agents. Identity drift is scored offline using a paraphrase-aware, value-anchored, multi-register diff rather than raw cosine similarity. We evaluate the instrument via a controlled seed run (n = 25) and a reflection-threshold sweep (thresholds {40, 80, 150}) to determine if drift dynamics are gate artifacts or threshold-robust properties. We report two primary findings: (1) Anti-self-deception emerges unprompted as the single largest semantic category of identity modification (27 of 111 added boundaries, 24%). (2) The system is threshold-robust; lower gates accelerate and increase revision frequency but preserve drift direction. All empirical results are strictly preliminary existence proofs and effect shapes (one model, one seed per arm, n = 25) rather than statistical significance claims. 

---
# A Cognitively Motivated Multidimensional Framework for Evaluating Metaphor Explanations 

**Authors**: Ana Naveriani, Jakob Suchan, Stefano Zoia, Mehul Bhatt, Antonio Lieto, Gian Luca Pozzato  

**Link**: [PDF](https://arxiv.org/pdf/2608.15828)  

**Abstract**: Current evaluation of metaphor explanations relies mainly on holistic quality ratings, revealing little about how explanation quality is structured or where human judgments agree and diverge. We introduce a cognitively motivated framework that decomposes metaphor explanation quality into six theoretically grounded dimensions. In a dense annotation study (11,200 ratings), we find that: {\bfseries(i)} explanation quality is genuinely multidimensional; {\bfseries(ii)} annotator disagreement is systematic rather than random; and {\bfseries(iii)} the six dimensions collapse into a shared cluster and two independent axes of judgment. An exploratory feasibility study further shows that a standard automatic evaluation pipeline can recover parts of this structure, predicting the most discriminative dimensions well while its errors correlate human (dis)agreement. Together, these results suggest that multidimensional evaluation offers richer diagnostic insight than holistic ratings, and that automatic evaluators for open-ended generation tasks should be judged on how well they preserve the structure of human judgment. 

---
# QuantumPhaseNet: A Gauge-Covariant Geometric and Quantum-Spectral Theory of Semantic Concept Hierarchies with Prototype Validation of a Classical Quantum-Inspired Model 

**Authors**: Kiyotaka Kasubuchi, Kazuo Fukiya  

**Link**: [PDF](https://arxiv.org/pdf/2608.15820)  

**Abstract**: We present QuantumPhaseNet, a gauge-covariant geometric and quantum-spectral extension of Transformer representations. Context-dependent semantic states are modeled as complex amplitudes; a covariant phase rate induces a semantic wavelength used as a proxy for conceptual scale; and low-frequency graph modes define a document-level discourse direction. The theoretical part establishes local gauge invariance, unitarity of the quantum block, boundedness and conditional stability of WavePhase Attention, and a calibratable hallucination-risk formulation. We also implemented a fully offline Validation Studio for the classical quantum-inspired pipeline in Section 14.1 and evaluated the five research questions in Section 16.1 on its built-in synthetic setting (n=240, observation noise 0.22, circuit noise 0.08, five seeds). RQ1 yielded a wavelength-hierarchy Spearman correlation of 0.852 versus 0.707 for the baseline, 87.3% direction accuracy, and AUC 0.953. RQ2 achieved discourse alignment 0.933 versus 0.589 and 41.2 versus 16.2 paragraphs before drift. RQ3 achieved AUROC 0.881 versus cosine 0.765 and phase-shuffle 0.536. RQ4 achieved error-detection AUROC 0.854 versus entropy 0.634, with Brier 0.150 and ECE 0.098. RQ5 did not show quantum advantage: target probability and end-to-end cost efficiency were 25.5% and 0.107, compared with 70.7% and 0.707 for the Chebyshev classical approximation. These results provide initial synthetic evidence for the classical quantum-inspired components, but not external validity or unconditional quantum speedup. 

---
# Hallucination Span Detection with Input-Side Evidence Alignment 

**Authors**: Miyu Yamada, Yuki Arase  

**Link**: [PDF](https://arxiv.org/pdf/2608.15804)  

**Abstract**: Hallucinations remain a major obstacle to the reliable use of large language models (LLMs) in conditional text generation. Existing methods primarily assess the factuality of an entire generated text, providing limited insight into which output spans are hallucinated or how they relate to the input. We introduce the task of hallucination span detection with input-side evidence alignment, which jointly identifies hallucinated spans and aligns output tokens with the corresponding input evidence. Our approach is based on the observation that faithful output tokens are predictable from the input, whereas hallucinated tokens are not. We therefore train an encoder-based model to predict masked output tokens from the input representation, using prediction confidence for hallucination detection while naturally producing alignments to the input. Experiments show that the proposed method effectively detects hallucinated spans and identifies meaningful input-side evidence. Human evaluation confirms the quality of the predicted alignments. 

---
# Using the Mimi codec for metalinguistic representations 

**Authors**: Artem Saloev, Erin Pacquetet, Nicolas Ballier  

**Link**: [PDF](https://arxiv.org/pdf/2608.15799)  

**Abstract**: In this paper, we focus on the dictionary of 2048 tokens used in Mimi semantic token codebook, the neural codec of the Moshi language model. We show that the ABX experiment carried out with Mimi fails to capture the mapping of the semantic tokens to phone realisations. By realigning Mimi representations to the TIMIT corpus transcriptions, we show that the 2048 tokens IDs of the semantic codebook map to quadphone, triphone, biphone, phone and subphone realisations. 

---
# TaoLive Digital Avatar Agent Technical Report: Training Agents to Evolve with Their Harness 

**Authors**: TaoLive AIGC LLM Team, Yuhan Sun, Wenhao Lin, Yongdong Luo, Yibo Hu, Meiguang Jin, Junfeng Ma, Weihang Pan, Jiaxin Zhao, Zulong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.15763)  

**Abstract**: AI-powered digital-avatar streamers in live e-commerce must answer product questions, engage viewers, and execute changing business strategies in real time. This requires low latency, factual and effective replies, and rapid adaptation to updated campaign, compliance, and style requirements. We develop an evolvable Harness that decouples Skills, Hooks, system prompts, and tools from model weights, allowing runtime behavior to change without retraining. However, Harness evolution creates a moving execution environment: compact models fine-tuned on one configuration may memorize names, schemas, and prompt templates rather than follow the Harness currently provided, while stronger zero-shot models are too slow for real-time use. We address this tension with Harness-Aware Training (HAT), which makes Harness states part of the training distribution. HAT applies task-preserving Harness-State Augmentation (HSA) to Skills, tool schemas, prompt structures, and interaction constraints, and comprises three stages: HSA-based supervised fine-tuning, general on-policy distillation to recover general capabilities, and HSA-based agentic reinforcement learning in a production-informed live-room simulator. Across four evaluation sets with more than 4,500 cases, our compact 35B model scores 94.8 on real-world Live-Stream QA, versus 80.3 for the base model and 93.0 for the strongest evaluated general LLM, while scoring 94.6 on Harness-Variant QA and retaining 83.5 on IFEval. By contrast, fixed-Harness SFT reduces IFEval by 7.7 points. In a controlled complete-agent replay on one NVIDIA H20 GPU with MTP enabled, the system achieves 3.407 s P50 and 8.114 s P95 latency. These results show that HAT produces a latency-feasible compact agent that remains effective under evaluated Harness changes without sacrificing general instruction following. 

---
# BERTopic-Virality Prioritisation: A Scalable Framework for Thematic and Comparative Analysis of COVID-19 and Monkeypox Misinformation on Twitter 

**Authors**: Mkululi Sikosana, Sean Maudsley-Barton, Oluwaseun Ajao  

**Link**: [PDF](https://arxiv.org/pdf/2608.15691)  

**Abstract**: Health misinformation circulating during pandemics can gain traction rapidly, creating harmful narratives that compete with public health guidance. Most topic-modelling pipelines treat engagement as an external outcome, limiting their ability to prioritise semantically coherent topics that are also rapidly diffusing. We introduce BERTopic-VP, a virality-prioritised topic-modelling framework that combines contextual embedding-based clustering (BERTopic) with a post hoc Virality Prioritisation (VP) layer. The pipeline is complemented by a two-stage hybrid misinformation detection module that fuses a supervised content-based classifier with an external verification signal derived from public-health knowledge bases. Applied to three benchmark datasets, COVID-19_FNIR, Monkeypox, and Constraint, the framework achieves strong classification performance, with F1 up to 0.950 and ROC-AUC up to 0.989, while identifying high-impact clusters under top 1%, 5%, and 10% VP thresholds. For datasets without native engagement metadata, prioritisation is based on a logistic propensity-to-spread score, used as an ordinal proxy for diffusion potential rather than a direct measure of engagement. The results show that integrating semantic structure, virality-aware ranking, and affective-linguistic profiling enables scalable and interpretable comparative analysis of misinformation across pandemics. The proposed framework supports monitoring-oriented early warning by surfacing low-volume but high-risk narratives for analyst review. 

---
# When Stories Evolve: Benchmarking LLM Storytelling Across Agent Architectures in Open-Ended World Simulations 

**Authors**: Yuqi Chen, Sixuan Li, Yunfeng Cai, Xueai Li, Ka Man Yan, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.15654)  

**Abstract**: Large language models can write fluent stories, but open-ended storytelling requires more than local fluency. In evolving world simulations and AI-native games, models must preserve facts, relationships, causal dependencies, and character states as the world changes. We introduce WSE-bench, a process benchmark that separately evaluates sustained generation, canonical coherence, and meaningful development in dynamic LLM storytelling. Generation Coverage records the proportion of planned narrative steps produced; Consistency tracks when canon breaks; and Richness measures how meaningfully branching, player-shaped trajectories develop. Across frontier models, Consistency and Richness do not form a smooth trade-off: their empirical Pareto frontier is non-concave, with several non-dominated intermediate configurations that no positive linear weighting can select. Added structure can enrich trajectories, but it does not uniformly improve coherence and may shorten them. Model scale chiefly improves sustained generation, without producing reliable gains in canonical coherence or meaningful development. These results show that sustained generation, canonical coherence, and meaningful development are distinct and sometimes competing capacities. WSE-bench makes those dynamics visible by extending narrative evaluation from finished stories to the processes that create them. 

---
# Wiktionary as a Crowdsourced Lexicon for English Dialects 

**Authors**: Sidney Wong  

**Link**: [PDF](https://arxiv.org/pdf/2608.15641)  

**Abstract**: This paper evaluates Wiktionary as an ethically crowdsourced lexicon for English dialects. We took a two-phase approach, providing an in-depth descriptive analysis of the crowdsourced lexicon for 12 national varieties of English before applying the lexicon to geo-referenced, country-level social media language data to examine the real-world performance of this crowdsourced dialect lexicon. We demonstrate that Wiktionary matches or exceeds the coverage of traditional dictionaries, such as the Oxford English Dictionary (OED), for regional and Outer-Circle varieties. Our dialect-specific case study on New Zealand English found high alignment between Wiktionary and the OED based on word-formation patterns (R = 0.883). Similarly, we observed high alignment between the dialect lexicon and geo-referenced social media language. While this paper found that Wiktionary has broad coverage of lexical properties, it also highlighted some of the macro-challenges involved in evaluating dialect-responsive language resources and tools, such as the role of language contact in dialects and register effects in web-based corpora. 

---
# BengaliMCQ: Automatic Generation and Answer Prediction of Academic Multiple-Choice Questions in a Low-Resource Language 

**Authors**: Abu Tarabin Surzo, A.K.M. Nihalul Kabir, Sm Azmain Faysal, Ariana Haque Ami, Lawrence Amlan Gomes, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2608.15547)  

**Abstract**: Traditional retrieval-augmented generation (RAG) frameworks process documents without attending to their hierarchical structure, leading to poor performance, especially in low-resource languages such as Bengali. To address this, we propose a structure-aware RAG framework that models Bengali textbooks as hierarchical graphs and uses a contrastively trained graph neural network to retrieve a small set of relevant passages. These passages provide focused context for a large language model, enabling topic-specific multiple-choice question (MCQ) generation and in-domain answer prediction. Experimental results demonstrate that our framework outperforms strong dense retrieval baselines across retrieval metrics, produces more relevant MCQs, and achieves superior answer prediction accuracy. 

---
# L3Cube-IndicQuest v2: A Large-Scale Multilingual Benchmark for Evaluating Factual Knowledge of Large Language Models Across Indic Languages 

**Authors**: Rinit Jain, Tirthraj Mahajan, Advait Joshi, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2608.15535)  

**Abstract**: We present L3Cube-IndicQuest v2, a large-scale gold-standard multilingual question-answering benchmark for evaluating the India-specific factual knowledge of Large Language Models (LLMs). The benchmark comprises 3,471 curriculum-grounded English question--answer pairs spanning nine domains, curated from educational curricula, competitive examination materials, and domain-specific reference books. We introduce a practical hybrid construction strategy that combines context-grounded LLM-based question generation and validation with semantic deduplication and human verification, enabling scalable creation of benchmark data while preserving annotation quality. The benchmark is translated into 19 Indic languages, yielding a publicly released multilingual dataset of 69,420 question--answer pairs across 20 languages. We evaluate six LLMs under three protocols: LLM-as-a-judge and two deterministic lexical criteria, exact-substring and word-overlap matching. All three produce almost the same model ranking, showing that the results do not depend on the choice of judge. The frontier commercial model leads by a wide margin, and among open-weight models Gemma4 31B outperforms the Indic-specialised Sarvam 30B in every evaluated Indic language. 

---
# Why Summaries Turn Neutral: Policy Attribution for Sentiment Drift in Reinforcement Learning from Human Feedback 

**Authors**: Mikhail Krasitskii, Alexander Gelbukh, Olga Kolesnikova, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2608.15530)  

**Abstract**: Reinforcement learning with human feedback (RLHF) aligns LLMs with human preferences, improving summarization fluency and safety, but causes sentiment drift: overly neutral summaries stripped of emotional nuance. We diagnose why RL acts as a sentiment neutralizer and present Policy Attribution, a framework using gradient and logit decomposition to trace drift to reward model (RM) signals and KL (Kullback-Leibler) penalty. Sentiment drift reflects a strategic bias toward "low-risk" tokens maximizing expected rewards under preference uncertainty (Stiennon et al., 2020; Gao, Schulman, and Hilton, 2023). On Reddit TL;DR and CNN/DailyMail, RLHF summaries get higher rewards but show 30-40% lower sentiment variance. Cross-lingual analysis across eight languages shows language-independent drift, with morphologically richer languages more suppressed (Krasitskii et al., 2026). We propose and validate a sentiment-aware regularization technique reducing drift by 18-22% without harming summary quality. The code and toolkit will be public. 

---
# Do Language Models Consistently Encode the Current Year? 

**Authors**: Suze van Adrichem, Aditi Bhaskar, Diyi Yang, Christopher Potts, Jing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.15507)  

**Abstract**: A consistent concept of the current time is important for temporal reasoning, yet how language models represent the current time is not well understood. We contribute two tasks that probe the current year in conceptually distinct ways: an associative task, which infers the current year from verb tense, and a declarative task, which directly queries for the current year. Both tasks estimate current years within one year of the post-training data cutoff of instruction-tuned language models. For base models, predictions on the associative task serve as a strong proxy for the pre-training data cutoff, with an average error of only 10 months across 13 models. However, their internal mechanisms diverge: the associative task uses mechanisms similar to factual recall, while the declarative task lacks consistent causal pathways. This divergence poses a challenge for updating the current year in language models. None of prompting, SFT, or weight editing succeed in shifting the associative and declarative years simultaneously. Prompting updates the declarative year (94.6% success across 351 target years) but leaves the associative year nearly unchanged (1.7% success). Year-shifted SFT also fails to shift the associative year, matching the target year in only one of eight models. Weight editing, while effective for both tasks individually, does not generalize across both. Overall, our results show that the current year is not consistently encoded in language models: The associative notion, deeply ingrained in linguistic structures learned in pre-training, uses different causal mechanisms and resists the same modifications that easily shift the declarative notion learned in post-training. 

---
# Language models suffer from a curse of ambiguity 

**Authors**: Nicolas Zucchet, Hyun Dong Lee, Scott Linderman  

**Link**: [PDF](https://arxiv.org/pdf/2608.15448)  

**Abstract**: Large language models increasingly rely on sampling as a driver of their own improvement, making the fidelity of their learned distributions more critical than ever. Yet, not all distributions are equally easy to learn. In this work, we identify a curse of ambiguity: in large language models, and more broadly in all neural networks that produce discrete probability distributions, the more ambiguous a next-token distribution is, the harder it is to learn accurately. Through an extensive theoretical analysis, we trace this curse to architectural and learning roots. More ambiguous distributions require more capacity to be stored, larger embeddings to be represented, more steps to be fitted, and amplify token-sampling noise. We validate these findings on synthetic tasks with controlled ground truth and observe the same signatures in language models trained on real data. Our results provide a new perspective on the statistical capabilities of large language models and a practical framework for when to trust their output distribution. 

---
# Semantic Space of Parts of Speech 

**Authors**: Jiří Milička, Ivan Kraus, Arnold Stanovský, Anna Vysloužilová, Barbora Štěpánková, Lenka Fárová, Vojtěch Cink, Šárka Dohnalová  

**Link**: [PDF](https://arxiv.org/pdf/2608.15443)  

**Abstract**: Parts of speech categorization is understood in the European linguistic tradition as crisp categorization, which is also reflected in corpus linguistics, where each disambiguated token is assigned exactly one POS. However, the assigned categories are largely determined by arbitrary decisions distilled into annotation manuals. Since some words stand between parts of speech in their semantics or typical syntax, and some parts of speech are closer to each other than others, POS categorization seems inherently fuzzy. We analyze this fuzziness using word2vec embeddings, training a neural network to reduce their high dimensionality to three dimensions relevant for determining parts of speech. This creates a three-dimensional space onto which we map several thousand words, revealing which are prototypical and which lie on the boundaries, and visualizing relationships between parts of speech. The study uses Universal Dependencies POS tags for French, Czech, Finnish, Russian, and English. 

---
# Gated Against One Model, Open to the Next: Option-Only Solvability in Legal Multiple-Choice Benchmarks 

**Authors**: Volodymyr Ovcharov  

**Link**: [PDF](https://arxiv.org/pdf/2608.15428)  

**Abstract**: Multiple-choice benchmarks are graded on whether a model picks the right option, not on whether it needed the question. Measuring that gap takes care: a model answering A to most items scores above chance wherever the key sits at A, and reads as recognition when it is not. We measure it on UA-JudgeExam: 11,990 four-option items with official keys, published by Ukraine's Higher Qualification Commission of Judges.
Shown the options and no question, Claude Haiku 4.5 scores 0.383 against chance, and the leak is concentrated: 11.8% of items are answered blind on all eight option orders, against 0.2 items expected by chance. It is not quotation: search over 280,059 editions of Ukrainian legislation recovers 0.128. Gating those out retains 8,128 items, on which the gating model itself now scores 0.204, and GPT-5.6, which took no part in the selection, still answers 0.515 of them with the question hidden. Scoring twelve held-out models on the whole set and subtracting each one's answer-position habit, only two keep an excess: GPT-5.6 at +0.265, Sonnet 4.6 at +0.081. Without it the ranking misleads: Llama 3.1 8B scores 0.292 blind, above every model but those two, purely by answering A to 92% of items.
The gate does select something real: on the items it rejected, eleven of twelve models score 0.518-0.789, every interval clear of what the same model scores on the items it kept. But that signal is one model's, and filtering on it does not transfer upward. Neither is visible on a 400-item sample, where nine models read as "statistically at chance". Rewriting distractors instead overshoots to 0.168, below chance and as exploitable. The same probe on LEXam returns chance: every option there points into the stem, none longer than 33 characters. Item format decides whether the problem can arise; capability decides how much is extracted. We release the corpus, the predictions and the harness. 

---
# The Machine's Internal Clock: Do LLMs Share Human Temporal Illusions? 

**Authors**: Catherine Bao, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2608.15394)  

**Abstract**: Human perception of time is subjective. Well-documented temporal illusions show that the brain relies on context and relational cues for judging duration instead of tracking elapsed time directly. Prior studies established these effects with visual and auditory stimuli. Existing LLM evaluations of temporal perception focus on estimating event durations or multi-step temporal reasoning. In this work, we investigate whether written narratives alone can evoke human temporal illusions, using a new benchmark of 6,684 narrative pairs spanning five illusions. We find that human readers (60 participants) prefer expected scenarios in only two of the five illusions, those where the manipulation is directly visible in text rather than requiring readers to internally simulate duration. We evaluate 14 LLMs on the same benchmark. Surprisingly, we find that models pick the literature-predicted scenario across four of the five illusions, diverging from human behavior. Reasoning traces show that ~70% of responses explicitly evoke psychology research, suggesting that this alignment is consistent with retrieval of published findings rather than human-like temporal biases. 

---
# When AI Rewrites, Classifiers Relax: Uncertainty-Aware Sentiment Analysis on Sarcastic and AI-Paraphrased Social Text 

**Authors**: Shresth Shroff  

**Link**: [PDF](https://arxiv.org/pdf/2608.15338)  

**Abstract**: Sentiment classifiers are increasingly applied to social media content that is either sarcastic or AI-generated --- two distributional regimes where standard evaluations offer little guidance. We present a three-part empirical study of sentiment classifier behaviour under these conditions. First, we find that confidence scores on sarcastic text are significantly lower than on non-sarcastic text (Mann--Whitney $p = 2 \times 10^{-6}$), confirming that classifiers sense their own uncertainty on ironic content even without explicit uncertainty modelling. Second, and counterintuitively, we show that sentiment classifiers achieve higher accuracy on AI-paraphrased reviews than on the original human-authored text (RoBERTa: $+5.8$ pp for Qwen3.5-4B paraphrases, $+3.7$ pp for Gemma4-E4B), revealing a cross-domain stylistic alignment effect: AI paraphrases remove distributional noise that confounds Twitter-trained classifiers, producing cleaner, more prototypical sentiment text. Third, we demonstrate that a lightweight abstention wrapper --- flagging the $14\%$ of inputs with confidence below $0.6$ --- improves accuracy from 82.2\% to 88.9\% ($+6.7$ pp) on the retained set. We further compare Semantic Entropy and MC-Dropout-style disagreement as uncertainty signals and find near-identical AUROC ($0.650$ vs.\ $0.646$) on sarcastic text, suggesting that for short social media inputs, both methods are interchangeable. Our results motivate a shift from confident single-label prediction to uncertainty-aware abstention in high-stakes sentiment applications such as mental health flagging and content moderation. 

---
# Logical Embeddings for Argument Analysis 

**Authors**: Leander Heldring, Santiago Torres  

**Link**: [PDF](https://arxiv.org/pdf/2608.15325)  

**Abstract**: We propose a new framework for machine-learning-oriented argument analysis tasks. Our proposal involves replacing traditional contextualized word embeddings used in most NLP tasks with logical embeddings, an alternative encoding that directly exploits argumentation structures. In essence, logical embeddings encapsulate the logical semantics of an argument, allowing for a better representation of its meaning. Supporting these embeddings is a mathematical logic-based similarity measure that offers a transparent notion of proximity and is guaranteed to satisfy several desirable theoretical properties that current cosine similarity-based contextualized word embeddings cannot assure. This similarity measure induces a positive semi-definite kernel on the set of arguments, enabling us to uniquely define logical embeddings using the theory of Reproducing Kernel Hilbert Spaces (RKHS). Moreover, we prove that this encoding is optimal, in the sense that no logical information is lost in the process. As with other RKHS applications, logical embeddings can be used in numerous supervised and unsupervised tasks. We provide an implementation of the method and aim to test it against literature benchmarks. Additionally, we demonstrate that logical embeddings outperform most standard embedding methods on a classification task. 

---
# When Do Concepts Become Functionally Sufficient During Language-Model Training? 

**Authors**: Raphael Bernas, Paul G. Chevalier, Fanny Jourdan, Céline Hudelot  

**Link**: [PDF](https://arxiv.org/pdf/2608.15323)  

**Abstract**: Understanding a model and its learning mechanisms in depth requires identifying when its internal structures become useful, rather than simply looking at the final state. We study this through concept dynamics: at each layer and checkpoint, we decompose activations, select sparse soft masks, and inject masked reconstructions into the model. Concept analysis is therefore tested functionally: a mask is useful only insofar as it preserves a target under intervention. We compare sufficiency for activation reconstruction, linear decodability, true downstream preservation, and checkpoint transfer under learned alignment. The framework treats decomposition assumptions as hypotheses rather than interpretability guarantees, monitoring functional sufficiency across checkpoints and source-to-final reconstructability under learned alignment. At the shared fixed-penalty operating point across seven models, downstream masks retain substantially less soft mass than reconstruction masks; predictive-distribution shifts remain small. 

---
# Time as Structure: Temporal Dependency Graphs for Verifiable Deadline Computation over Legal Documents 

**Authors**: Maryia Zhyrko, Lifeng Han, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2608.15270)  

**Abstract**: Miss a filing deadline by one day and the claim is barred, however strong the case. Computing that deadline is rarely simple: the period runs from a triggering event, is counted by a statutory convention, and may be suspended by a mandatory conciliation window. We ask whether a language model should answer such questions directly, or read the document and leave the arithmetic to code. We extract dated facts and their dependencies into a temporal dependency graph and compute deadlines from it with a calendar-correct engine. On UK Employment Appeal Tribunal judgments the engine reproduces six of seven timeliness rulings, and matches the judges' own dates to the day. The strongest of four language models, asked the same cases, gets the arithmetic right and the answer wrong: in six of twenty-one responses its stated verdict contradicts its own thinking, and every contradiction runs the same way, calling a late claim timely. To test the systems at scale we move the dismissal date across the statutory boundary, generating 427 cases whose answers are computed rather than annotated. On the cases both systems answer, the pipeline is right 90.2% of the time against 61.2% for direct answering. The limit is extraction: on contracts the errors are almost never in the arithmetic, but in choosing which event the period starts from. 

---
# TRACE-BN: Transferring Bangla-English Tutoring Behavior to a Sub-1B Offline Language Model 

**Authors**: Khan Raiyan Ibne Reza, Sanjana Aktar Maria, Mohammad Tushar Abdullah, Asfee Bhuiyan Leen, Sumaiya Tabassum Nimi  

**Link**: [PDF](https://arxiv.org/pdf/2608.15223)  

**Abstract**: Bangla-English tutoring requires more than producing a correct translation: learners also need explanations of grammar differences, awareness of their likely errors, and targeted practice. We present TRACE-BN, a curriculum-guided dataset of structured tutoring traces for Bangla-speaking learners of English at the CEFR A1-A2 level. Each trace combines word-level glosses, literal and natural translations, Bangla grammar explanations, a plausible learner error, and a targeted practice question with its answer. The traces are generated by Gemini 3.5 Flash Lite as the teacher model from NCTB Classes 9-10 English curriculum units, then filtered for structural validity, script integrity, and semantic duplication. We transfer the resulting structured tutoring behavior to Qwen3-0.6B using LoRA with 4-bit quantization for resource-constrained offline deployment. On held-out inputs, schema validity increases from 85.4% to 95.8%, while, against teacher-model references, chrF++ improves from 15.28 to 34.77 and BLEU from 4.52 to 21.03. Field-level evaluation by two independent judges shows improvements across translation, grammar explanation, learner-error diagnosis, and practice alignment, while a human audit supports the quality of the supervision data. The results show that curriculum-guided structured supervision can transfer multi-component tutoring behavior to a sub-1B model under these resource constraints. The dataset, model checkpoints, and code are publicly available at this https URL 

---
# Left-Branching Transformers Excel at Right-Branching Languages: Data Shapes Word Order Preferences in Language Models 

**Authors**: Varvara Arzt, Allan Hanbury, Terra Blevins  

**Link**: [PDF](https://arxiv.org/pdf/2608.15129)  

**Abstract**: We systematically compare word order preferences in decoder-only language models across 192 artificial languages and typologically diverse natural languages. On artificial languages, models exhibit a left-branching preference that aligns with neither natural language universals nor human word order learning biases. On natural languages, monolingual models show no clear base word order bias at small scales, but as data grows, a preference for right-branching subject-verb-object (SVO) languages emerges while SOV falls behind despite being the most frequent order cross-linguistically. This SVO advantage extends to multilingual models and correlates with language resource level and data quality rather than word order. Thus, the same architecture exhibits opposite preferences on artificial and natural languages, establishing that word order biases observed in practice are data-driven. Since highly-resourced languages are overwhelmingly SVO, these biases risk gradually reducing word order diversity, particularly in languages that productively use multiple word orders, with the widespread adoption of LLMs. 

---
# A Declarative-Procedural Perspective on Expert Routing in Bilingual Mixture-of-Experts Language Models 

**Authors**: Amrit Gopinath, Raghul, Durairaj Thenmozhi  

**Link**: [PDF](https://arxiv.org/pdf/2608.15102)  

**Abstract**: We investigate whether Mixture-of-Experts (MoE) language models develop linguistically structured expert routing during bilingual language acquisition. Inspired by the Declarative-Procedural framework, we analyze lexical, grammatical, and syntactic processing in a decoder-only English-German MoE Transformer trained under sequential language exposure. We construct a probe-based validation set and extract token-level routing distributions to quantify category-dependent specialisation using mutual information, routing entropy, and Jensen-Shannon distance. The curriculum-trained model exhibits a peak mutual information of 0.1148 at layer 5, indicating category-dependent differences in routing distributions across linguistic categories. Surprisingly, a no-curriculum baseline trained on mixed English-German data shows stronger aggregate specialisation, reaching a peak mutual information of 0.2599 at the same layer. These results suggest that interpretable linguistic organization emerges within MoE routing patterns even without sequential language exposure. A replication at a second training seed shows that the no-curriculum condition's specialisation concentrates on a single language whose identity is seed-dependent, whereas the curriculum consistently yields a stable, language-balanced routing profile; rather than uniformly increasing specialisation, staged bilingual exposure reduces single-language dominance. The official Github repository: this https URL 

---
# Why Vision Fails as a Universal Bridge: Rectifying Modality Asynchrony in Multilingual MLLMs 

**Authors**: Yihang Du, Juhao Liang, Zhengzhao Lai, Siyu Li, Yan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.15085)  

**Abstract**: Multimodal large language models (MLLMs) exhibit substantial performance degradation in non-English visual reasoning, despite the strong multilingual competence of their text-only backbones. While mechanistic evidence from text-only models suggests that non-English inputs are routed through an English-centric latent space, the multimodal implications of this phenomenon remain unexplored. Through rigorous mechanistic analysis, we identify the \textbf{Ghost Anchor} phenomenon: a temporal modality asynchrony where linguistic translation to the English semantic manifold completes in early layers, while visual semanticization remains immature. Consequently, visual signals are physically present yet functionally invisible during the early alignment window. To rectify this, we propose \textbf{ANCHOR}, a training framework employing Proactive Visual Anchoring (PVA) to accelerate early visual semantic emergence, ensuring visual representations proactively guide linguistic translation. Mechanistic interventions confirm that ANCHOR successfully restores the causal influence of visual signals during early translation. Furthermore, extensive experiments on XMMMU, MaXM, and CVQA demonstrate that ANCHOR consistently outperforms standard baselines, achieving robust visual reasoning across both fine-tuned and zero-shot languages. 

---
# A Pilot Study of Autocompleting Tokenizers 

**Authors**: Samuel Wexler, Mark Hopkins  

**Link**: [PDF](https://arxiv.org/pdf/2608.15080)  

**Abstract**: Modern input methods routinely rely on autocomplete to omit information that can be recovered from local context. Inspired by these autocomplete-assisted writing systems, we investigate whether Transformer inputs can be compressed in a similar manner. Byte-level tokenization offers a simple and language-independent alternative to subword tokenization, but its longer input sequences typically result in increased computational cost and reduced model quality. We propose a compression scheme that employs a lightweight autoregressive byte language model to identify and remove bytes that are easily predictable from their surrounding context before Transformer processing. The resulting compressed representation is then provided as input to a standard encoder--decoder Transformer. Experiments on machine translation show that a substantial fraction of source-language bytes can be omitted without degrading translation quality. On English--French, our best method preserves translation performance while reducing source sequence length by nearly one-third. Additional experiments on Finnish--English, Russian--English, and Chinese--English demonstrate that the approach generalizes across diverse writing systems and morphological typologies, yielding comparable or improved translation quality at compression ratios between 0.47 and 0.67. These findings suggest that many input bytes are predictable enough to be represented implicitly rather than explicitly, providing a simple mechanism for reducing the sequence-length overhead associated with byte-level models. 

---
# RecurrentGPT: Expressive Depth through Recurrent Modulation in Transformers 

**Authors**: Amr Hegazy, Amr Alanwar, Mostafa Elhoushi  

**Link**: [PDF](https://arxiv.org/pdf/2608.15062)  

**Abstract**: Scaling transformer language models creates an inherent tension between expressivity and memory efficiency. While unique weights across layers preserve functional specialization---from input-grounding to abstract refinement---they incur a substantial memory footprint. Conversely, standard depth-sharing enforces uniform transformations that collapse representational diversity and degrade modeling quality. We introduce RecurrentGPT, a recurrent depth transformer where fixed-depth prelude and coda blocks bracket a single shared core iterated R times. Inspired by gated recurrent neural networks, we employ a lightweight projection and an elementwise update gate---conditioned on the hidden state, the fixed prelude output, and noise resampled at every step---to modulate the recurrent update. This allows the model to specialize the input to the same few layers across recurrences, rather than requiring many unique layers to achieve functional diversity. Under an isoFLOPS constraint, a 3-layer RecurrentGPT matches the accuracy of a 12-layer GPT-2 Small baseline with similar training and inference FLOPs, and leads MoR and heavy-tail depth sampling in all nine scale-by-budget cells; at medium and large scale it approaches dense quality at the standard token budget and overtakes it at medium scale once that budget is doubled. Under an isoPARAMS constraint, deeper recurrence achieves a 2.76 validation loss versus 2.84 for a non-recurrent counterpart at matched parameter and data budget. Our results demonstrate that adaptive depth reuse is a principled strategy for trading parameters for quality: at large scale, 63% fewer parameters and 59% less peak decoding memory for a 10% increase in compiled generation latency. 

---
# Handoff-H1: An Orchestrated Vision-Agent System for Material Quantity Takeoff from Construction Blueprints 

**Authors**: Bruno Chicelli, Henrique Alves, Rodrigo Anselmo, Joshua Weinberg, Felipe Lemos, Jan Baryla  

**Link**: [PDF](https://arxiv.org/pdf/2608.15032)  

**Abstract**: Converting a set of architectural blueprints into a complete material quantity takeoff requires visual perception across drawing sheets, dimensional and multi-hop reasoning, and grounding in construction conventions that the drawings never state. We present Handoff-H1, a takeoff system built from three layers: purpose-built computer-vision models that extract primitives; tool-using agents equipped with image operations and in-house visual-task tools, including CV-model-backed counting, detection and plan decomposition; and a persistent, hierarchically structured project foundation, grounded in a curated construction knowledge base. We evaluate on the Construction Blueprint Takeoff Benchmark: 10 real residential blueprint sets paired with consensus-validated expert takeoffs - 2,009 verified line items, restricted for scoring to the 1,348 primary-tier materials that drive an estimate - scored per trade by an LLM judge on material coverage and quantity Precision@25% (P@.25) and combined into a weighted composite. Under identical scoring from the raw PDF, seven frontier and open-weight models span composites of 35-61, and independent professional estimators - scored against the same reconciled gold standard - post 77.6% (65.5% coverage, 87.9% P@.25). Handoff-H1, working end-to-end from the raw PDF, reaches 81.6% (86.1% coverage, 78.8% P@.25): roughly 20 points above the strongest frontier agent, and above the independent estimators by pairing near-human quantity precision with coverage they do not reach. The evaluation harness is public for the open harbor framework; the blueprint sets and ground truth are available upon request for research use. 

---
# Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents 

**Authors**: Wei-Chieh Huang, Weizhi Zhang, Yuchen Wu, Yankai Chen, Eric Hanchen Jiang, Wooseong Yang, Yiwei Yang, Henry Peng Zou, Hanrong Zhang, Ying Nian Wu, Haolun Wu, Kai-Wei Chang, Philip S. Yu, Xue Liu, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2608.15008)  

**Abstract**: Memory is becoming core infrastructure for long-horizon LLM agents, yet existing evaluations offer limited guidance on which memory substrate, namely the underlying medium in which memory is represented and stored, should be used under different operating regimes. We present a controlled harness evaluation of memory substrates for memory-augmented agents, covering dense and sparse indices, text records, structural stores, hierarchical stores, refinement-based memories, parametric updates, and activation-compatible context mechanisms. Across three backbone models and four benchmark suites spanning user-centric question answering and agent-centric decision-making, we instrument 26 performance and efficiency metrics under a unified harness. Our results show that no single substrate consistently dominates: broad retrieval benefits long-context factual QA, while excessive retrieval can harm sequential decision-making by shifting attention away from action-critical context. Scalability introduces a further routing axis, as substrates that perform well at moderate history lengths can become costly or brittle at longer horizons. These findings motivate substrate routing as a necessary component of adaptive agent memory systems and provide empirical guidance for designing efficient, reliable, and regime-aware long-term memory for LLM agents. Code will be made available upon acceptance. 

---
# RamseyGadgets: A Graph Construction Dataset for LLMs 

**Authors**: Zohair Raza Hassan, Deepak Pandita  

**Link**: [PDF](https://arxiv.org/pdf/2608.14999)  

**Abstract**: Constructing special graphs is an important task within graph theory and computer science. Many popular graph constructions are the result of a comprehensive exploration of relevant graphs and human ingenuity. Given the rise of generative AI usage in mathematics, it is natural to test whether LLMs are able to construct graphs with specified properties using their reasoning capabilities. Unfortunately, many natural graph construction problems, such as finding extremal Ramsey-good graphs (i.e., avoiding specific monochromatic subgraphs), have been explored extensively in the literature, making it difficult to ascertain whether a construction is the product of an LLM's reasoning capabilities or its recollection from training data. In this work, we introduce \textbf{RamseyGadgets}, a novel dataset of 70 underexplored graph construction problems that require finding Ramsey-good graphs with special properties (e.g., containing an edge with a fixed color). These problems have reasonably sized solutions (at most 10 vertices) that can be verified by SAT solvers, making them suitable for automatic evaluation. Our dataset is easily expandable, as one can simply change the monochromatic subgraphs being avoided to obtain a new set of problems. We evaluate the performance of five open-source LLMs on our dataset and report the results. Our findings show that LLMs achieve only 37.70% accuracy on the hard-tier problems in our dataset, with Gemma-4-31B achieving the highest performance out of the five. We also showcase how our dataset allows us to ascertain what kind of hints help LLMs perform better at this task. 

---
# DA-RAC: Distance-Aware Calibration of LLM Judges for Trustworthy AI Auditing 

**Authors**: Cheng Wu, Vishal Anand, Jaya Krishna Mandivarapu, Xiya Liu, Rui Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14950)  

**Abstract**: Generative AI systems are increasingly producing real-world artifacts, however their efficacy and validity are often evaluated via context-free LLM-scoring. These judges can be miscalibrated by irrelevant in-context reference examples, creating false confidence and allowing low-quality or harmful outputs to pass evaluation. We study this failure mode as context-induced miscalibration and introduce DA-RAC, a distance-aware reference-anchored calibration method for LLM judges. DA-RAC retrieves semantically and structurally similar labeled anchors for each judgement scenario, weights them by distance, and exposes neighborhood difficulty as a calibration and triage signal. On multi-run LLM-judge evaluation benchmarks, it improves calibration and reduces false-pass risk relative to zero-shot, chain-of-thought evaluation, and static-anchor baselines. Mechanistic analysis shows that judge scores vary systematically with anchor distance, while static references can induce misleading decision boundaries. Thus LLM-judgement requires not only better models, but also calibrated, auditable reference selection, especially when automated evaluation is used to support high-impact AI generated artifacts. Judgments should be grounded in relevant, inspectable, and contestable interpretive artifacts. 

---
# Training Leaves Traces: Centered Residual Signatures for Language Model Lineage Verification 

**Authors**: Aman Singh Thakur, Rayan Khoury  

**Link**: [PDF](https://arxiv.org/pdf/2608.14929)  

**Abstract**: Open-weight language models are fine-tuned, quantized, pruned, and merged, yet their provenance is often undocumented. We study data-free white-box lineage verification: can weights alone reveal whether two compatible model checkpoints share ancestry?
Residual training produces a shared identity-aligned component in branch products, so this structure alone cannot establish ancestry. We remove it and compare checkpoint-specific structure across residual blocks, yielding a symmetric lineage score calibrated against independent checkpoints. On residual-MLP and GPT-2 benchmarks, the score separates fine-tuned, LoRA-merged, pruned, and quantized descendants from independent and distilled models (AUROC=1.0), distinguishing weight ancestry from behavioral similarity. Under function-preserving checkpoint laundering experiments, weight-space baselines lose margin or fail; our score remains unchanged and runs 76x faster than the nearest robust baseline on GPT-2. The projection-pairing signal appears across six language-model families and beyond, and a case study correctly identifies 3 related and 7 unrelated LLaMA-2 public checkpoints. Collectively, these results establish a passive, data-free provenance signal for compatible open-weight language-model checkpoints 

---
# How Do Agents Fail on AutoResearch: End-to-End Diagnostic Evaluation on 100 Real-World Frontier Research Tasks 

**Authors**: Yanlin Fei, Nazhou Liu, Xinmiao Yu, Shaolong Chen, Lei Li, Rahul Thapa, Madalina Ciobanu, Qingqing Mao, Ritankar Das  

**Link**: [PDF](https://arxiv.org/pdf/2608.14905)  

**Abstract**: AI has long assisted scientific research, but the rapid advance of LLMs and agentic scaffolds is reshaping the landscape; a single system can now carry whole-stage research from an initial hypothesis all the way to final published paper, which is a paradigm now referred to as AutoResearch. Existing evaluations reveal little about how these agents operate or where they break down. Tasks are narrowly-scoped, evaluation measures performance but not process, and failure diagnoses lack systematic coverage or artifact-level visibility. To address this gap, we introduce AutoResearchEval, featuring 100 tasks grounded in published frontier science across 7 scientific domains and the full research lifecycle, including ideation, retrieval, execution, analysis, writing, and review. Evaluating 8 harness-model combinations yields 800 autoresearch agent trajectories, with process-level annotation. We organize these insights into AutoResearch Failure Taxonomy or ARFT, a framework of 45 empirically-grounded failure patterns. To enable scalable fine-grained attribution, we leverage a human-calibrated agent-as-a-judge pipeline to inspect complete trajectories and intermediate artifacts. Failure patterns converge on a single overarching limitation, namely that current agents lack a metacognitive loop, which entails the ability to check what they produced against what they found, revise when it does not hold up, and question whether the path they took was sound. The same patterns recur across all 8 harness-model combinations, including the strongest models tested, locating the deficit at the model level rather than in any particular scaffold; whether orchestration-level interventions can close it is an open question this work does not test. We publicly release AutoResearchEval and ARFT to facilitate continued research and development in autonomous scientific discovery. 

---
# Interpretable Cross-Lingual Alignment in Small Language Models: Probing Cultural and Pragmatic Reasoning in Japanese-English Bilingual LLMs 

**Authors**: Florian Braun  

**Link**: [PDF](https://arxiv.org/pdf/2608.14896)  

**Abstract**: Large language models work well on English and behave in poorly understood ways on languages typologically far from it. Japanese is a clean example, where evaluation still leans on translation quality and JGLUE-style benchmarks, which roll lexical, syntactic and pragmatic competence into a single score. The phenomena on which general-purpose models fail Japanese users are pragmatic: honorifics, in-group and out-group reference, context-sensitive politeness, zero anaphora.
I introduce J-PragEval-v0, a minimal-pair benchmark isolating four such phenomena from surface fluency, and combine it with linear probes and teacher-forced log-probability evaluation to ask where inside TinySwallow-1.5B (28 layers, hidden size 1536) the corresponding contrasts live. The four features split three ways. Honorific register sits cleanly in the residual stream: 0.96 balanced accuracy at layer 15, and the model flips its preferred continuation with the scenario on 93 percent of items. Implicit subject and in-group reference are not linearly decodable at the final prompt token (0.48 and 0.38), yet flip rates are 0.77 and 0.79, so the contrast is worked out during generation rather than stored at the prompt. Indirect refusal is the negative case: 0.95 probe accuracy collapsing to a 0.43 flip rate under length-normalised teacher forcing, because the current minimal pairs conflate politeness with continuation length.
I also specify Pragmatic Representation Steering, a parameter-free inference-time method that edits residual-stream activations along the class-mean-difference directions probing identifies. Feasibility is argued indirectly rather than demonstrated: the contrastive activation addition baseline, the same geometry the method would inject, recovers probe accuracy within one to two points of logistic regression wherever a linear signal exists. Scaling to Llama-3.1-Swallow-8B is the next step. 

---
# Where Does Retrieval Fail? Evaluating RAG Architectures for Agricultural Advisory 

**Authors**: Khan Raiyan Ibne Reza, Sanjana Aktar Maria, Sumaiya Tabassum Nimi  

**Link**: [PDF](https://arxiv.org/pdf/2608.14886)  

**Abstract**: Retrieval quality in RAG systems is commonly reported as a single aggregate score, which can hide large differences across query types and language conditions. We study this problem in Bengali agricultural advisory, where farmer queries are often colloquial while official advisory documents use formal scientific terminology. We construct a test collection of 1,000 queries and 2,882 knowledge nodes extracted from 284 official Bangladeshi agricultural publications, and use it to evaluate five retrieval architectures and six embedding models under three controlled language conditions.
The results show that no single retrieval method is consistently best. For native Bengali queries, BM25 is the strongest single retriever (R@10 = 0.506) while Hybrid RRF reaches the highest overall R@10 of 0.539. However, dense retrieval performance varies sharply by query type: R@10 is 0.093 on colloquial farmer queries and 0.970 on formal safety queries. Across language conditions, BM25 R@10 drops from 0.506 on Bengali queries to 0.004 when English queries are matched against the Bengali corpus, while dense retrieval falls only from 0.464 to 0.425. We also find that embedding task configuration and passage length can each change reported R@10 by a factor of seven, independent of architecture. These results show why low-resource RAG evaluation should report performance by language condition and query type rather than relying on aggregate scores alone. The dataset and evaluation scripts are available at this https URL. 

---
# What to Forget in Unlearning? Forget Set Curation for Language Models 

**Authors**: Animesh Jha, Arpandeep Khatua, Youssef Allouah, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2608.14855)  

**Abstract**: Machine unlearning aims to remove targeted data or behaviors from a trained model without retraining from scratch. Yet most evaluations assume that the examples to forget are already known. In realistic language-model deployments, a requester may ask a model to stop reproducing a song or book without knowing which spans, documents, quotations, or near-duplicates in a trillion-token corpus support that behavior. We study this missing upstream problem, forget set curation: mapping a suppression request to the data passed to an unlearning algorithm. We introduce CleanSlate, a benchmark for verbatim output suppression over songs and books, with model-specific extraction profiles, content-grounded QA, and capability-retention evaluations. CleanSlate exposes two failure modes. Natural lexical and exact-substring curators often yield forget sets that lead to weak suppression. An evaluation-aware curator suppresses requested continuations almost completely, but causes collateral regression on non-requested content and model-dependent capability loss. These results show that practical unlearning is not only an optimization problem once a forget set is given: the data chosen for forgetting determines both what can be unlearnt and what else is damaged. 

---
# Writing Style Similarity Reflects Academic Genealogy 

**Authors**: Cameron Manzo  

**Link**: [PDF](https://arxiv.org/pdf/2608.14843)  

**Abstract**: As authorship attribution systems are increasingly deployed to detect ghostwritten and AI-generated papers, their errors can support accusations against legitimate authors. These systems assume each author's style is their own. Researchers, however, study under advisors, and inherit their stylistic quirks. We build a corpus of arXiv authors with $\geq 2$ solo papers from the Mathematics Genealogy Project graph, giving $5{,}803$ total authors and $2{,}501$ ground-truth advisor-student pairings. Using embeddings from a fine-tuned model, advisors sit $39.9\%$ closer in cosine distance to their students than a random same-field author does. Two open encoders reproduce the effect at $12.6\%$ and $14.5\%$. \emph{Academic siblings}, two students of one advisor who may never have met, sit $30.4\%$ closer across $8{,}360$ pairs, even when they studied at different institutions. Pairs who share only an institution and a field show negligible similarity. Given a closed-set attribution task over the same corpus, the system's errors occur on the true author's advisors and academic siblings $11$ times more often than chance. 

---
# Beyond the pale: Assessing prevalence and contents of extremist speech in LLM training data 

**Authors**: Dmitry Nikolaev, Ashley A. Mattheis  

**Link**: [PDF](https://arxiv.org/pdf/2608.14813)  

**Abstract**: Despite a strong interest on the part of the research community in the topic of trustworthy and safe AI, the composition of the text corpora that large language models (LLMs) encounter in pre- and post-training has not yet drawn much attention. In this work, we address the question of whether LLMs are exposed to unfiltered, uncontextualised extremist speech. Using several definitions of extremist speech, stemming from official documents and research literature, and an extraction pipeline combining automated text processing with expert verification, we provide a lower bound on the prevalence of extremist documents in Dolma, an open training corpus underpinning the OLMo series of models. We show that Dolma is likely to include hundreds of thousands of documents containing extremist content and hate speech of several types, including direct calls for violence, and discuss the implications of this for data curation and model pre-training. 

---
# Beyond Tokens: A Survey on Decoding Methods for Large Language and Vision-Language Models 

**Authors**: Haoran Wang, Xiongxiao Xu, Philip S. Yu, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2608.14797)  

**Abstract**: Large language models (LLMs) and large vision-language models (LVLMs) have demonstrated impressive generative capabilities, yet ensuring their outputs align with user intent is still challenging. While most existing approaches address this issue at the training stage, inference-time approaches like decoding methods offer a more efficient and scalable solution. Decoding methods control model generation by guiding token-level selection, performing sequence-level generation, or generating tokens in parallel to accelerate the process. In this survey, we identify three emerging paradigms from recent works on decoding methods for LLMs and LVLMs, provide a systematic review of these methods, highlight ongoing challenges, and discuss potential future research directions. Our goal is to underscore the efficiency and effectiveness of decoding methods and offer a practical view of their applications. Paper lists and more resources on decoding methods for LLMs and LVLMs can be found at this https URL. 

---
# Prompting is not enough: supervised baselines and leakage control for measuring shared decision-making with LLMs in pediatric encounters 

**Authors**: Bernardo Modenesi, Jody Lin, Kimberly Kaphingst, Angela Zhu, Maya Wheeler, Peilu Zhang, Angela Fagerlin  

**Link**: [PDF](https://arxiv.org/pdf/2608.14792)  

**Abstract**: Objectives: To determine whether zero-shot prompting of a large language model (LLM) is sufficient to detect shared decision-making (SDM) behaviors in real clinical encounters, and whether supervised learning adds value under patient-grouped, nested evaluation.
Methods: We analyzed 21 audio-recorded outpatient surgical decision encounters (19 unique patients; 7,566 utterance segments; ~6.1 hours) between families of children with multiple long-term conditions and their surgical providers. Trained coders labeled segments for 12 SDM behaviors (human-human macro Cohen's kappa = 0.695). We compared a zero-shot local LLM (Qwen 2.5 32B), a supervised classifier over frozen sentence embeddings, and their logistic stack, under patient-grouped outer folds with inner cross-fitted thresholds and patient-resampled confidence intervals.
Results: The zero-shot LLM reached macro kappa = 0.139 (95% CI 0.111-0.164). The supervised classifier reached kappa = 0.227 (0.186-0.262), a paired improvement of 0.088 (0.051-0.119). A logistic stack of the two reached kappa = 0.242 (0.198-0.284). We identified multiple corpus-specific leakage paths, including grouping sibling recordings separately and allowing labels from an outer held-out patient to enter few-shot exemplars used while fitting downstream models.
Conclusion: Zero-shot prompting alone is not sufficient to measure SDM behavior as reliably as a small supervised model, and patient-level grouping alone does not prevent leakage when labeled prompt exemplars are precomputed outside the outer evaluation loop. Reported performance is sensitive to the unit of data splitting and to where labeled exemplars enter the pipeline. External validation is needed before these findings generalize beyond this population, model, prompt, and codebook. 

---
# Class Imbalance and Batch Effects in LLM-Based Screening for Systematic Reviews 

**Authors**: Gilberto Sussumu Hida, Danilo Monteiro Ribeiro, Clayton Suguio Hida  

**Link**: [PDF](https://arxiv.org/pdf/2608.14737)  

**Abstract**: This study analyses LLMs in imbalanced binary classification, using study screening in systematic reviews as the application domain. An experiment was conducted in five reviews, comparing individual and batch processing, with and without prevalence metadata. The results indicate a limited influence of the prevalence metadata, with no evidence that it improves performance. In contrast, batch processing produced larger behavioral changes that varied according to the prevalence of the class. The aggregate and item-level analyses did not always coincide. Therefore, batch processing should be evaluated not only in terms of cost, but also in relation to its effects on decision-making behavior. 

---
# Which Question Is Your Attention Metric Answering? Attention Rows as Compositional Data 

**Authors**: Marios Papamichalis, Regina Ruane  

**Link**: [PDF](https://arxiv.org/pdf/2608.14712)  

**Abstract**: Each row of a transformer's attention matrix is a probability distribution over tokens, and in trained models most of that probability lands on a single \emph{sink} token, usually the first. Standard tools for comparing attention rows (cosine similarity, Jensen--Shannon divergence, Shannon entropy) therefore hinge on a choice papers rarely report: keep the sink, or drop it and renormalize. This choice can reverse conclusions. On ten pretrained models from five families, 17--47% of verdicts about which of two heads is more similar flip with the convention, and the most prominent structure in a standard BERT head-clustering pipeline is an artifact of it. The reason is that one-number summaries mix two questions: how much attention the sink takes, and how the rest is divided among the content tokens. Treating rows as compositional data separates them exactly: the Aitchison distance splits orthogonally into a sink term and a content term, entropy splits by an exact identity, and the content distance is characterized by invariances the transformer itself possesses. The separation matters in practice: most measured entropy collapse during training is the sink growing, not attention sharpening (30% of the drop at 70M parameters, 95% at 1B, 79% at 1.4B), and pruning heads with the wrong channel can inflate perplexity more than a hundredfold. We map where each convention is safe, test a frozen out-of-sample predictor (one confirmation, one abstention, one failure), and release code regenerating every number. 

---
# Domain Agnostic Text Redaction from Natural Language Rules using Instruction Tuning 

**Authors**: Aravindhan Arunagiri, Ayaan Khan, Udayaadithya Avadhanam, SaiBarath Sundar  

**Link**: [PDF](https://arxiv.org/pdf/2608.14693)  

**Abstract**: With the increasing digitization of personal and corporate communication, the automatic sanitization of textual data has become a crucial component of data privacy and compliance frameworks. Traditional text sanitization solutions are majorly suitable for obscuring sensitive data with standard structure such as Personal Identifiable Information (PII). These solutions do not provide transparent justification for their redaction, which makes it difficult to audit them. This paper introduces an explainable, domain-agnostic text redaction solution that uses natural language rules of redaction, applied via an instruction-tuned language model, to identify and redact sensitive information in unstructured documents. Unlike traditional text sanitization, this method enables a user to conveniently define any sensitive information; which may be structured (e.g.\ PII) or unstructured (e.g.\ legal terms and conditions) in natural language. A general-purpose LLM generates or augments these natural language rules of redaction from the user's definition, which are then used to instruction-fine-tune a smaller language model that reasons the rules step-by-step over any given document to identify and redact the corresponding sensitive content, while providing transparent justifications for each redaction and highlighting the specific rule that triggered the decision. This explanation is generated in natural language to support human reviewers and auditors in understanding why specific content was redacted. A reconstruction-based metric is used to estimate the probability of recovering redacted information from the sanitized document, quantifying redaction coverage. The solution shows high reconstruction error and high redaction precision, making it suitable for automated text sanitization in critical applications such as legal discovery, medical documentation, and corporate information governance. 

---
# Automatic or Controlled? Repetition Priming Reveals Divergent Processing in Base LLMs, Instruct LLMs, and Humans 

**Authors**: Jinglei Ren, Yuyue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14681)  

**Abstract**: Words recur constantly in natural language use, yet it remains unclear whether language models reactivate prior representations or re-evaluate repeated words afresh, and whether post-training changes this default behavior. We apply repetition priming (Shiffrin and Schneider, 1977) to 15 models across five model families (1.5B-14B parameters) in two tasks, semantic categorization and cloze completion, with matched human experiments using identical stimuli. We find that base models exhibit automatic processing: they show immediate facilitation that remains stable across lags, partially survives context removal, and correlates with attention to prior occurrences. Instruct models exhibit controlled processing: their facilitation decays with lag, collapses without expected context, and reverses to interference at larger scales. Within the Qwen 2.5 family, this dissociation increases monotonically with model scale, suggesting that post-training progressively alters repetition processing. Humans show a hybrid profile, with lag-sensitive facilitation resembling instruct models but without interference, suggesting that neither model type fully captures human cognition. Our findings reveal a qualitative shift in how language models process repeated information after post-training and provide mechanistic evidence for the divergence between model behaviors. 

---
# DeMTS: Denoising Trajectories as Multivariate Time Series for Hallucination Detection in Diffusion Language Models 

**Authors**: Xin Zhang, Yili Wang, Yue Tan, Xin He, Yanyu Qian, Yixin Liu, Yi Chang, Shirui Pan, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14632)  

**Abstract**: Diffusion large language models (D-LLMs) have emerged as a promising paradigm for text generation. However, similar to autoregressive LLMs, D-LLMs remain vulnerable to hallucinations, where fluent outputs may contain factually incorrect or unsupported content. Although existing hallucination detection methods for D-LLMs attempt to leverage uncertainty trajectories of the denoising process to better identify hallucination signals, they typically compress the trajectories along either the temporal or token dimension, overlooking the useful information encoded in the complete two-dimensional token-step structure. Consequently, they may fail to capture hallucination-relevant patterns, such as inconsistent convergence and cross-token fault propagation, leading to suboptimal detection performance. To bridge this gap, we propose a D-LLM hallucination detection framework that formulates the Denoising trajectories as Multivariate Time Series over learnable latent variables (DeMTS for short). DeMTS employs a trajectory-preserving token-to-variable assignment module to convert token signals into stable latent variables. Based on these variables, we propose dynamic multivariate temporal modeling to progressively integrate inter-variable dependency modeling with temporal encoding for hallucination prediction. Extensive experiments on two D-LLMs backbones and three benchmarks demonstrate that DeMTS outperforms existing hallucination detection methods while maintaining strong robustness, efficiency, and cross-task transferability. 

---
# Characterizing Rhetorical Misalignment in Decision-Making with Language Models 

**Authors**: Zirui Cheng, Joey Chan, Simo Du, Chenhao Tan, Yue Guo, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2608.14630)  

**Abstract**: Human decision-making is often shaped by a range of well-documented cognitive biases. As large language models (LLMs) become increasingly integrated into high-stakes human-AI decision-making, it is important to understand whether their outputs can amplify potential biases, how this influences human decisions, and crucially, whether it can lead to harmful consequences. In this work, we develop a decision-theoretic framework to study rhetorical misalignment, a failure mode where an LLM uses rhetorically inappropriate forms of presentation for a given decision context, thereby inducing suboptimal human decisions. We empirically investigate this phenomenon through a human-subject experiment in realistic clinical decision-making using a dataset curated from the United States Medical Licensing Examination. By measuring how LLM-generated information affects decisions, we observe that LLMs induce an average 2.81% rate of harmful decision flips across different models, where clinician participants change from a correct to an incorrect answer. Rationales reported by participants provide evidence that these revisions are closely related to the language used by LLMs that may induce different types of cognitive biases, including anchoring, authority bias, and loss aversion. To enable scalable evaluation, we instantiate our theoretical framework using decision-makers simulated by LLMs to computationally measure rhetorical misalignment. Our findings reveal a safety concern previously unrecognized in high-stakes domains: a model can be factually aligned yet still induce harm through its rhetorical presentation. 

---
# Inference-Time Mitigation of Adversarial Political Bias in Large Language Models 

**Authors**: Tejaswi V. Panchagnula, Bruce Coburn, Bryce J. Dietrich, Robert X. Browning, Edward J. Delp, Fengqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.14629)  

**Abstract**: As Large Language Models (LLMs) become the mainstay for information retrieval and summarization tasks, ensuring that they are always non-partisan and invulnerable to political bias is a critical step towards safer and more trustworthy Artificial Intelligence (AI). Current model alignment paradigms, such as reinforcement learning from human feedback (RLHF), make LLMs follow overarching safety instructions. However, this instruction tuning can be exploited via adversarial prompt injection and be used to generate unsafe content. In particular, political bias has not been specifically targeted by modern alignment techniques as harmful and biased content. To address this vulnerability of LLMs, we propose mitigation strategies using Chain of Thought (CoT) prompting and Direct Preference Optimization (DPO). Using a public dataset of legislative videos, we generate summaries using LLMs, inject bias via adversarial prompting and evaluate their performance on a four axis scale designed for political summarization. In this paper, we present different methods to shield LLMs against the injection of political bias. Our results demonstrate that the proposed Recursive Self-Correction approach raises model performance from a Political Neutrality Likert scale baseline of 2.14 to 4.56, averaged across all models, demonstrating effective inference-time mitigation of political bias in LLM-generated summaries. 

---
# LLM Safety Alignment in Low-Resource Languages: A Systematic Literature Review 

**Authors**: Valdini Douglace Lemofouet, Blessing Ngozi Uzor, Paula Chikaodinaka Anyanwu, Danielle Blanche Kapsa, Sukairaj Hafiz Imam, P Sam Sahil, Abigail Oppong, Tassallah Abdullahi, Clemencia Siro, Idris Abdulmumin, Seid Muhie Yimam, Shamsuddeen Hassan Muhammad  

**Link**: [PDF](https://arxiv.org/pdf/2608.14626)  

**Abstract**: Large Language Models (LLMs) have achieved substantial progress in safety alignment, yet their safety guarantees remain significantly weaker in low-resource and multilingual settings than in high-resource languages. In this paper, we conduct a Systematic Literature Review (SLR) of LLM safety alignment in low-resource languages by adopting the PRISMA 2020 methodology. Out of roughly 1,500 papers identified from Semantic Scholar, arXiv, and OpenAlex, 50 relevant studies have been selected and analyzed. Our review is organized around four themes: safety alignment methods, multilingual safety risks, evaluation benchmarks, and cross-lingual transferability. We further propose a taxonomy of safety alignment approaches based on three adaptation mechanisms: data adaptation, objective optimization, and mechanistic alignment. Across literature, translated English benchmarks fail to sufficiently represent culturally rooted harms, and multilingual models are more vulnerable to cross-lingual jailbreaks, code-switching attacks, and safety degradation in underrepresented languages. These failures are driven by several key factors, including uneven multilingual pre-training coverage, insufficient native-language preference data, poor transfer of safety representations, and a lack of culturally aware evaluation frameworks. The review also notes that many low-resource languages, especially African languages, have fewer safety benchmarks available than other multilingual regions. Overall, the results reveal a persistent multilingual safety gap, and suggest that future progress will require culturally grounded benchmarks, participatory data collection, balanced multilingual pre-training, and scalable multilingual alignment methods. 

---
# AutoMem: A Text-Gradient Recursive Self-Improvement Framework for Automated Memory Architectures Search 

**Authors**: Lin Du, Jie Zhou, Yuxuan Cai, Kai Chen, Qin Chen, Xin Li, Bo Zhang, Wei Li, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2608.14621)  

**Abstract**: Long-term memory is increasingly central to LLM agents, yet memory design remains a highly coupled architecture problem: what to encode, how to store it, how to retrieve it, and how to manage it can vary substantially across tasks and backbone models. We construct a discrete search space with 5 encoders, 5 stores, 6 retrievers, and 4 managers, and show that no single memory architecture consistently dominates: different tasks favor different module combinations, leading to substantial performance gaps. Motivated by this, we propose \textsc{AutoMem}, a text-gradient recursive self-improvement framework for task-adaptive memory architecture search. \textsc{AutoMem} optimizes over the factored space through two components: Experience-Guided Architecture Search, which proposes candidate architectures from historical search trajectories and accumulated reflections, and Failure-Guided Module Diagnosis, which localizes memory-related failures to specific modules and converts them into targeted textual feedback. Experiments on GAIA, WebWalkerQA, and xBench-DeepSearch across two LLM backbones show that \textsc{AutoMem} consistently discovers task-adaptive memory architectures that outperform the strongest human-designed memory baselines, improving accuracy by $2.8$ points on average across six benchmark-backbone settings. Further analysis shows that \textsc{AutoMem} achieves a favorable accuracy-efficiency trade-off, reducing token cost by $14.3\%$ over the strongest accuracy baselines under Qwen3.5-122B-A10B, while also finding stronger architectures than substantially larger random searches within only a few guided iterations. 

---
# Wiola 13M, a Gated Spiral Attention Architecture for Parameter Efficient Small Language Models 

**Authors**: Aryuemaan Kumar Chowdhury, Praveen Oosa, Vineesha Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2608.14604)  

**Abstract**: Small language models in the ten to one hundred million parameter range are attractive for on device inference, rapid experimentation, and controlled scientific study, yet most of them reuse the standard transformer block without adaptation to the small scale regime. We present Wiola, a decoder only language model whose novelty is concentrated in three drop in components of every layer. First, Spiral Rotary Positional Encoding perturbs the standard rotary frequencies by a slowly growing per dimension factor so that phase trajectories fan outward, improving long range discrimination while adding no parameters. Second, Gated Spiral Attention introduces a per head, content adaptive scalar gate derived from a causal cumulative statistic of the query stream, providing an implicit and differentiable form of soft head selection at negligible cost. Third, the Butterfly feed forward block replaces the conventional expansion layer with a multiplicative interaction and an intra block bypass path, matching the parameter count of a four times gated linear unit block while improving gradient flow in shallow stacks. We formalize each component, derive exact parameter and computation budgets, and prove that the gated attention admits an exact and numerically verified equivalence between full sequence training and cached autoregressive decoding, so that no approximation is introduced at inference time. We also describe a fully reproducible training and evaluation protocol on a standard tiny story corpus. The reference implementation is released as an open source package with weights ready publishing support. 

---
# Multi-Modal Generative Fuzzy System: Fuzzy Inference Guided Large Model Interactive Question Answering Framework 

**Authors**: Hailong Yang, Jianqi Wang, Guanjin Wang, Zhaohong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2608.14584)  

**Abstract**: In Multimodal Question Answering (MQA), models are required to jointly encode and integrate heterogeneous information from multiple modalities, including text, images, and speech, to perform complex semantic reasoning and decision making. Despite recent advances, existing approaches, including traditional deep learning models and Large Models (LMs) or prompt-based frameworks, continue to face several critical challenges. First, modality bias arises from discrepancies in feature distributions across different modalities, which limits effective cross modal collaborative understanding. Second, many questions require knowledge drawn from multiple domains, introducing significant uncertainty. Third, current methods often rely on shallow semantic matching, resulting in limited reasoning depth an reduced interpretability. To address these issues, inspired by the traditional fuzzy system (FS) framework, we propose a fuzzy-inference-guided multimodal generative architecture termed the Multi-Modal Generative Fuzzy System (MMGFS). The main contributions of MMGFS are two folds. First, it alleviates modality bias through a multimodal collaborative rumination mechanism. Second, it introduces fuzzy rules and a multi-hop inference mechanism to support cross-domain knowledge fusion and hierarchical reasoning, thereby strengthening uncertainty modelling and deepening semantic understanding. We conduct comprehensive evaluations on open-domain question answering datasets, including MultimodalQA and WebQA, as well as domain-specific benchmarks, including BioMol-VQA and EHRxQA. Experimental results demonstrate that MMGFS consistently outperforms existing methods across multiple datasets. It effectively mitigates modality bias and question uncertainty while achieving superior performance in answer accuracy, consistency, and generalization. 

---
# HarmProfile: Characterizing Harmful Distributions in Frontier LLMs 

**Authors**: Zhouyuan Ma, Yutao Wu, Hanxun Huang, Xiang Zheng, Xiao Liu, Yixin Cao, Zuxuan Wu, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14577)  

**Abstract**: Frontier large language models (LLMs) safety evaluation has largely treated harmful generation as an attack outcome rather than as an object of analysis. Consequently, little is known about the harmful outputs produced during model misbehavior, partly because large-scale, high-quality collections of frontier-LLM misbehavior are difficult to obtain. To address this gap, we introduce HarmProfile, a content-centric benchmark dataset that collects model misbehavior across diverse harm categories and model families, and defines the resulting harmful-output distribution as a model-level risk profile. The premise is that, just as linguistic behavior can be characterized from an utterance corpus, model risk can be characterized from the content, severity, and variation of its safety failures. HarmProfile contains over 80,000 validated artifacts from 23 frontier LLMs across 13 model families, organized into 15 harm categories and 57 subcategories. Using this corpus, we find that frontier LLMs reliably produce harmful content at scale, yet exhibit distinct risk profiles; both harmfulness and diversity grow with model capability, suggesting that frontier LLMs may appear safe yet harbor increasingly dangerous knowledge beneath the alignment surface. Our source code is available at this https URL . 

---
# Auxiliary uncertainty signals for LLM-assisted systematic review screening: a benchmark across eight Cohen drug-class reviews 

**Authors**: Arya Rahgozar, Pouria Mortezaagha  

**Link**: [PDF](https://arxiv.org/pdf/2608.14551)  

**Abstract**: Large language models (LLMs) are increasingly used for title-abstract screening in systematic reviews, but their decisions lack calibrated uncertainty. We show that an auxiliary BERT+GCN classifier supplies a structured uncertainty signal that improves LLM screening efficiency, and we identify the prompt-delivery strategy that maximises the benefit-to-cost ratio.
We evaluate five LLM prompt-delivery conditions on eight drug-class datasets from the Cohen (2006) benchmark using 3 seeds x 5-fold stratified cross-validation (600 fold-level results). A BERT+GCN model trained per fold classifies each test paper as INCLUDE, EXCLUDE, or MAYBE via two spectral tests (algebraic radical and categorical paradox). Conditions vary information content (none / label / full scores), selectivity (all papers vs. MAYBE only), and timing (proactive vs. reactive two-pass). A cross-model pilot against gpt-4.1-mini on three datasets tests cross-generation transfer.
Three findings: (i) Full-context delivery yields significant gains in F1 (+0.011, paired Wilcoxon p=0.008) and WSS@95 (+0.050, p=0.039) at a 1.28x token-cost premium, while preserving recall. (ii) MAYBE-only routing is Pareto-optimal: highest mean recall (0.92) and AUC-ROC (0.54) at only 1.05x baseline cost -- one sixth of full-context overhead. (iii) The two-pass design escalates 22.2% +/- 8.8% of records yet never revises its decision (0% flip rate across all datasets and folds), giving decisive evidence that current instruction-tuned LLMs cannot self-triage. The cross-model pilot shows an identical +0.8% recall uplift for both LLM generations. A per-paper ablation across 20,796 observations shows the dual paradox test reduces empirically to a one-line logit-gap criterion. We release the full pipeline; the 600-run experiment replays in under one hour from cached LLM responses. 

---
# Proteus: Incremental Memory Activation for Long-Context Sequence Modeling 

**Authors**: Reza Bayat, Ali Behrouz, Vahab Mirrokni, Aaron Courville  

**Link**: [PDF](https://arxiv.org/pdf/2608.16844)  

**Abstract**: The quadratic cost of attention-based sequence models for long contexts has motivated a growing line of research on memory-based models that can compress context into a compact state. However, most existing memory models expose a static memory throughout the entire sequence. Because early tokens face no compression pressure, they occupy too many degrees of freedom and "pollute" the memory state, leaving little capacity for later context and increasing interference between what is stored and what arrives next. We study a new paradigm of incremental memory activation, where the effective capacity of memory is progressively expanded as the context grows. Imposing an early bottleneck forces the model to compress history more effectively, while unlocking fresh capacity over time reduces interference and improves retention of later context. We instantiate this paradigm in Proteus, a straightforward mechanism that can be incorporated into a broad class of neural memory architectures at no additional cost. We apply Proteus to state-of-the-art models, including SWLA, Comba, Titans, and Hope-Attention, and observe consistent improvements on standard language modeling and reasoning, as well as on long-context retrieval and understanding, with gains that grow at longer context lengths. Overall, our results show that static memory is suboptimal and that scheduling effective capacity is a simple and broadly applicable tool for sequence modeling. 

---
# Policy Iteration with Human Feedback: Bringing Post-Training RL to In-context Learning 

**Authors**: Minh-Ha Nguyen, Cathy Shyr  

**Link**: [PDF](https://arxiv.org/pdf/2608.16831)  

**Abstract**: Generative pretraining established reusable task representations; later work on language-based task conditioning and in-context learning showed that a fixed model could adapt its behavior from instructions and demonstrations. Policy Iteration with Human Feedback (PIHF) builds on this development and the recurrent evaluate-and-improve structure of generalized policy iteration. PIHF uses a pretrained language model as its execution substrate and moves persistent revision to a versioned natural-language policy and tool set. A language-model critic and clinical expert review complete-panel reasoning and tool-use trajectories to localize recurrent failures and form candidate revisions; the expert may reinterpret the evidence and retains authority over admission and rollback, while Recall@1 and Recall@5 validate outcomes after candidate execution.
Across cumulative ablations and ultra-rare-disease benchmarks, a PIHF-derived policy improved Recall@1 in one proprietary executor and three open-weight executors spanning 3 to 49 billion active parameters. Gains were 32.7 percentage points for GPT-5.4 and 31.1 points for Qwen3.6-35B, a difference of 1.7 points. These results support the feasibility of using pretrained language models as fixed-weight execution substrates for expert-guided policy development in rare-disease diagnosis. 

---
# Neurosymbolic Embodied Agents 

**Authors**: Mohammad Albinhassan, Yuming Feng, Alessandra Russo, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2608.16794)  

**Abstract**: Language and vision-language models generate plausible embodied plans but do not guarantee executability, as their outputs can violate environment dynamics or act on incorrectly grounded entities. We present a neurosymbolic agent that factors long-horizon household tasks into task-directed visual exploration and constrained symbolic planning. In the first phase, a vision-language model and exploration harness acquire goal-relevant predicates and instance bindings from egocentric observations and grounded interactions, producing a symbolic initial state. In the second, a PDDL transition model restricts decoding to tokens that extend applicable actions. Monte Carlo tree search then evaluates executable continuations using a domain-independent planning heuristic. The resulting plans are executable by construction under the transition model, with transfer to the environment conditioned on correct visual grounding. On VirtualHome and ALFWorld, open 4B-27B models exceed 90% success in both environments, and our smallest agent substantially outperforms a 27B direct visual policy in each. Constraints and search prove complementary rather than interchangeable: in ALFWorld either alone solves under a third of tasks, whereas their combination solves over 95%. The method also uses several times fewer generated tokens than extended thinking and far fewer model-visible images than direct interaction, and residual failures localize to state acquisition rather than plan generation without any specialized training. 

---
# Closing the Affective Loop: Multimodal Speaker-Listener Emotion-Dynamics-Aware Empathetic Social Robots 

**Authors**: Zi Haur Pang, Casey Kennington, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2608.16686)  

**Abstract**: Empathetic social robots should respond not only to what users say, but also to how their emotions dynamically evolve during interaction. However, existing empathetic dialogue systems are often text-centered and primarily model empathy as a one-way mapping from the user's emotion to the system response, limiting their ability to capture embodied speaker--listener affective exchange. We present AffectLoop, a multimodal speaker-listener emotion-dynamics-aware spoken dialogue system implemented on the Misty II robot. The system tracks the speaker's verbal and facial affective dynamics, estimates the robot listener's own verbal and behavioral affective state, and conditions LLM-based response generation on both affective streams. The robot then generates a short spoken empathetic response together with emotionally congruent embodied behavior, forming a closed speaker--listener affective loop. We evaluate the system in a pilot within-subject study with five participants, comparing it with an otherwise identical utterance-conditioned baseline that omits the speaker- and listener-affective-state inputs. The proposed system received higher overall impression ratings, especially for empathetic response and user satisfaction. Post-hoc log analysis further showed higher speaker-listener affective alignment and stronger valence-based distress recovery. These preliminary results suggest that explicitly modeling both speaker emotional dynamics and listener affective state can improve embodied empathetic interaction. 

---
# Reconstruction: A Blind Benchmark for Recovering Research Ideas from Pre-Publication Bibliographies 

**Authors**: Shaolong Chen, Yanlin Fei, Nazhou Liu, Xinmiao Yu, Lei Li, Rahul Thapa, Madalina Ciobanu, Qingqing Mao, Ritankar Das  

**Link**: [PDF](https://arxiv.org/pdf/2608.16645)  

**Abstract**: Can a language model recover the true research idea of a published paper when given only that paper's pre-publication bibliography? We introduce Reconstruction, a blind idea-recovery benchmark that withholds the seed paper and all contemporaneous or future literature, and asks models to propose hypotheses that an independent large language model judge matches against the held-out ground-truth idea. A strict anti-leakage protocol-temporal citation cutoff, anonymous reference IDs, and frozen per-paper bibliographies, which prevents prompt-time leakage of the seed idea. Across six scientific domains and 643 evaluated papers, seven frontier models achieve only modest Match rates (approx. 3-15%). We then evaluate a reference-only multi-agent (top 4) pipeline that combines cross-model review with a Swiss tournament over aligned hypothesis slots, without external web search. Cross-model review plus tournament selection raises Match rates to approx. 23-42% across all six domains, which is an observed approx. 2.4x lift over the best single-model baseline. This draft reports the protocol, anti-leakage design, and current results as an arXiv timestamp. 

---
# Listen, Reason, and Segment: Aligning LALMs with Editorial Judgment for Media Chapterization 

**Authors**: Tony Alex, Wish Suharitdamrong, Sara Atito, Armin Mustafa, Muhammad Awais, Philip J. B. Jackson, Jiankang Deng, Ismail Elezi  

**Link**: [PDF](https://arxiv.org/pdf/2608.16539)  

**Abstract**: Large Audio Language Models (LALMs) have made rapid progress on standardized benchmarks, yet their deployment in practical media workflows, curation, archival indexing, and content distribution remains largely unrealized. We identify automated audio chapterization, the task of segmenting continuous audio streams into thematically coherent chapters, as a demanding and commercially consequential setting that exposes this gap. Chapterization is challenging because boundaries are defined less by objective acoustic events than by subjective editorial judgment, requiring models to reason sequentially over long acoustic contexts and approximate creator-authored boundary decisions. We present AudioChaps, a post-training framework for aligning end-to-end LALMs for this task via Group Relative Policy Optimization (GRPO) guided by Chain-of-Thought (CoT) reasoning. To support training and evaluation, we curate three datasets: AudioChaps-Alignment, derived from creator-annotated chapter boundaries on YouTube; AudioChaps-CoT, which provides structured supervision for well-formatted, high-quality, and evidence-grounded boundary reasoning; and AudioChaps-Eval, a held-out benchmark for audio chapterization. Applying GRPO directly without a Supervised Fine-Tuning (SFT) cold start, AudioChaps-R1-Zero already improves average F1 by 33 points over the state-of-the-art LALM Audio-Flamingo-3-Think. The AudioChaps framework produces our final aligned LALM, AudioChaps-R1, which improves average F1 by 49 points. These results demonstrate that GRPO-trained LALMs can reliably transform unstructured auditory streams into navigable, structured media. Our code, models, and dataset resources will be released upon acceptance at this https URL. 

---
# DSPrompt: Dynamic Soft Prompt Defense Against M-RAG Corruption 

**Authors**: Chang Liu, Yuni Lai, Mingyue Cui, Cong Tian, Yunyan Zhang, Xian Wu, Kai Zhou, Bin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.16536)  

**Abstract**: Multimodal Retrieval Augmented Generation (M-RAG) is increasingly vulnerable to adversarial attacks where malicious data are crafted to produce embeddings that align with benign entries in the vector space, deceiving retrieval and inducing harmful outputs. Existing defenses primarily operate at query time, relying on auxiliary detectors, similarity re-ranking, or feature-consistency checks. However, these approaches suffer from non-trivial inference overhead, generalize poorly to unseen attack strategies, and often assume specific attack distributions. To address this, we propose DSPrompt, a Dynamic Soft Prompt defense framework that directly reshapes the retriever's embedding semantics, without modifying the retrieval pipeline. It inserts few learnable soft prompts into each layer of the visual and textual encoders of a frozen retriever, utilizing a shallow-to-deep length schedule that is adaptive to the capacity in the model layers. These prompts are trained under a dynamic min-max scheme: an online multimodal attacker continually crafts hard adversarial documents against the current retriever, while the defender is updated to push such documents out of the top-k while preserving the ranking and diversity of benign evidence. Because the defended encoder can be pre-computed and indexed exactly as in standard dense retrieval, DSPrompt incurs no additional per-query optimization and introduces fewer than 1% additional parameters. Extensive experiments across four benchmarks and three representative poisoning attacks show that DSPrompt substantially reduces the attack success rate and poison retrieval rate while maintaining near-lossless retrieval utility and generation fidelity, consistently outperforming existing defense baselines at a fraction of their computational cost. 

---
# Matched Outcomes, Divergent Gaze: How Foveated MLLMs Search Compared to Humans 

**Authors**: Mohamed Amine Kerkouri, Marouane Tliba, Aladine Chetouani, Ulas Bagci, Alessandro Bruno  

**Link**: [PDF](https://arxiv.org/pdf/2608.16514)  

**Abstract**: Human visual search is serial: the fovea must land on a candidate to confirm it, and those landings form a scanpath. Whether multimodal large language models (MLLMs), given the same foveated input, search as humans do bears on their use as models of human vision and on attention-alignment scores. We compare three general-purpose MLLMs with human eye-movement scanpaths on goal-directed search (COCO-Search18), driving each model fixation by fixation through an identical, human-matched foveated view and assessing it along three axes: the decision of target presence, the efficiency of reaching the target, and the gaze process itself. The axes dissociate. On the decision and on target acquisition the models match or exceed humans, detecting present targets near ceiling and reaching them on the first saccade more often than people do. The gaze process is not human. Under the human-matched condition, all three share one signature: low-entropy, large-amplitude, self-consistent scanpaths that agree with themselves far more closely than two humans agree with each other. That is consistent with a single-pass, non-serial architecture rather than a limit of acuity. Matched retinal input reproduces where humans look but not how the looking unfolds in time, and no degradation regime recovers human-like search at human-like success. The gap sits on a process axis that answer-alignment and saliency metrics do not measure. Because they miss it, such metrics cannot certify human-like vision, and zero-shot models suit outcome and spatial questions but not temporal, process-level ones. 

---
# Computational KJ-Ho: An Analyst-Bias-Free Insight Extraction Framework from Large-Scale Qualitative Data Using Domain-Specialized LLMs 

**Authors**: Kasumi Ban  

**Link**: [PDF](https://arxiv.org/pdf/2608.16467)  

**Abstract**: The qualitative research methodologies that underpin consumer-insight generation - the KJ method, Grounded Theory, and Thematic Analysis - share a structural constraint: the cognitive processing capacity of the human analyst. Replication research further shows that conclusions vary substantially across analysts analyzing identical data (analyst bias). This paper proposes Computational KJ-Ho (the Kawakita Jiro method), a theoretical framework that computationally realizes the KJ method's epistemology - letting structure emerge from the data itself without imposing the analyst's preconceptions - an orientation we term "analyst-bias-free." The framework employs a domain-specialized LLM built through continued pre-training (CPT) on a marketing-research corpus and supervised fine-tuning (SFT) on expert-curated insight pairs, organized as a three-layer architecture: data structuring, insight extraction, and strategy generation. Two preliminary studies in the Japanese marketing context support the necessity of CPT-based domain specialization. The paper makes five contributions: (1) a theoretical integration of the KJ method, Grounded Theory, and Peircean abduction into a single epistemological commitment of data-driven explanation generation; (2) a three-layer architecture leveraging domain-specialized embeddings for cross-interview analysis; (3) two novel evaluation metrics, InsightExtraction-F1 and MarketingQA; (4) explicit engagement with the WEIRD problem, centering a non-Western methodology; and (5) five practice-derived problem formulations from nearly three decades of marketing-research practice, translated into design requirements. The human analyst retains a supervisory role. This is a concept paper presented ahead of empirical validation. 

---
# Deep Thought Alignment: Trajectory-Level Latent Distillation for Video Reasoning 

**Authors**: Ao Shen, Yongheng Zhang, Yinghui Li, Manning Wang, Di Yin, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.16316)  

**Abstract**: Large Multimodal Models (LMMs) for video reasoning have long been hindered by the high computational cost of processing vast amounts of visual information. This dilemma motivates the transfer of the reasoning capabilities of large models to smaller, more efficient ones. On-Policy Distillation (OPD) offers a promising solution by matching output-token distributions along student-generated trajectories. However, video reasoning often depends on evidence accumulated across multiple frames. In this context, output-level supervision only captures information expressed through token predictions and does not directly constrain the latent representations formed during reasoning. To address this limitation, we propose Latent-OPD, which augments OPD with trajectory-level latent distillation. Specifically, our method focuses on the position at the end of each trajectory, where hidden states effectively summarize the accumulated visual evidence and reasoning context. Furthermore, we introduce a progressive teacher-lookahead strategy, which aligns middle-to-late student layers with increasingly deeper teacher layers. Experiments on six video reasoning benchmarks show that Latent-OPD consistently outperforms output-only OPD. Notably, the improvements are particularly pronounced in scenarios with limited frames, long videos, or tasks requiring complex evidence aggregation. These results establish Latent-OPD as a highly effective approach to frame-efficient video reasoning. 

---
# PolyDebate: A Game-Orchestrated Multimodal System for Debate Skills Practice and Evaluation 

**Authors**: Jianing Yin, Weng Pan Kuan, Xiaoyun Liu, Zhiyuan Wen, Yuxuan Li, Milos Stojmenovic, Jiannong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2608.16276)  

**Abstract**: Debate is a structured form of persuasive communication that trains argument construction, rebuttal, oral delivery, and audience awareness. These skills are valued in education, language learning, and professional communication. Recent AI debate systems and LLM-based judges have advanced argument generation and debate evaluation, but most remain text-centered and rarely support learners through a complete multimodal practice experience. We introduce PolyDebate, a game-orchestrated multimodal system for English debate practice and evaluation. PolyDebate guides learners through staged one-on-one (1v1) debates with an AI opponent, while skill cards, props, and coins make persuasive strategies explicit and turn practice into a game-like interaction. During each session, the system captures learner speech and visual delivery evidence, generates context-aware opponent responses, and produces rubric-informed stage-level and overall feedback. PolyDebate is available as both an immersive Unity 3D game version and a web platform version that share the same workflow and evaluation services. Four studies covering AI opponent quality, evaluation coverage, AI judge feedback, and user perception show that PolyDebate brings debate interaction, gamified scaffolding, multimodal assessment, and structured feedback together in a practical workflow for debate skills practice. The demonstration video is available at this https URL. 

---
# INSPIRE: A Benchmark for Instruction-Aware Speech Retrieval 

**Authors**: Chen-An Li, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.16203)  

**Abstract**: Existing speech retrieval systems rely on fixed similarity matching and cannot adapt to diverse user intents. We introduce INSPIRE, the first benchmark for instruction-aware speech retrieval, in which natural-language instructions dynamically specify relevance criteria, including semantic content, speaker identity, speaking style, environmental sounds, and their combinations. We evaluate four retrieval paradigms: large audio-language models, cascaded pipelines, self-supervised speech models, and contrastive audio-language models. Our results reveal that no current method robustly handles all retrieval intents. Text-based approaches perform relatively better at semantic retrieval but struggle with paralinguistic attributes, while speech-based models are moderately better at capturing acoustic properties but falter at following instructions. These findings highlight the need for unified architectures capable of instruction-aware speech retrieval. 

---
# The Commercial Tax: Rent-vs-Own Blind Spots in Multi-Hop Retrieval Benchmarks 

**Authors**: Luis M. Sanchez, Kosrow Dehnad  

**Link**: [PDF](https://arxiv.org/pdf/2608.16096)  

**Abstract**: Enterprises connect language models to their own data through retrieval. The benchmarks that rank multi-hop retrieval systems leave out two facts a buyer needs before a published number can be used: whether the retrieval backbone may be deployed commercially, and what it costs to build. On licensing: the field's dense-retrieval anchor, NV-Embed-v2, is licensed cc-by-nc-4.0. Of the four leading MuSiQue systems we audit (HippoRAG-2, PropRAG, SAG, KET-RAG), three depend on it for their best numbers and none says so. On performance: we measure thirteen embedders from eight makers on one identical MuSiQue harness with bootstrap confidence intervals throughout. Until mid-2026 there was a real commercial tax: the best commercially-licensed embedder trailed the anchor by 2.31 Recall@5 points (95% CI [0.91, 3.71], p=0.001). NVIDIA's Nemotron-3-Embed-8B, released 2026-07-16, has closed it: +0.24 at Recall@5 (95% CI [-0.94, +1.43], p=0.69), -0.58 at Recall@10 (p=0.28). It matches the anchor, does not beat it, and is the only entrant that is commercially licensed, free to self-host, and indistinguishable from the anchor; every other entrant meeting the first two conditions sits 5.2 to 14.6 points below. The durable finding is the paid-versus-free divide: API embedders charge per token on every re-index, self-hosted ones charge nothing. On cost: three of five audited systems (adding Microsoft's GraphRAG) do not disclose indexing cost, and the only published GraphRAG dollar figures span 11x inside one third-party paper (USD 2.30 vs USD 24.94 to index a 5.64 MB corpus once); extrapolated to 1 TB that undisclosed choice separates roughly USD 428K from $4.6M. Our cost model keeps one-time embedding apart from recurring answering: at 1 TB, embedding sits 7.5x-900x below graph construction, and a year of answering at 10,000 queries/day sits 350x or more below it. 

---
# Coverage Is Not Containment: A Fundamental Limit of Admission-Time Defenses Against Coordinated Poisoning of Vector Retrieval 

**Authors**: Prashant Kumar Pathak, Tarun Kumar Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2608.16044)  

**Abstract**: Retrieval-augmented generation (RAG) answers a question by retrieving passages from a vector store and trusting them as context, so anyone who can add documents can try to steer the answer. A recent, appealing defense filters poisoning at ingestion, rejecting any document that behaves like a hub. We show it -- and every ingestion-time filter -- is defeated by a coordinated adversary that injects a handful of individually unremarkable documents which together surround one target query and seize its top-k (on BGE-large / BEIR, m=10 documents take 10/10; 9.9/10 on a live HNSW index). The attack is not theoretical. Realized as ordinary fluent text and run end-to-end through a BGE-large + HNSW + Qwen2.5-7B pipeline, it makes the generator emit the attacker's planted claim in 88% of targets, versus 0% without the injection. And no admission-time defense stops it: at ingestion an attack cone is geometrically identical to a legitimate niche upload, so -- measuring this directly -- the strongest trained classifier, given every feature and thousands of examples, separates the two no better than chance, catching 4.2% of attacks at a 1% false-positive rate. We prove this limit for the entire class of ingestion-time statistics (any decision from documents and reference queries alone), and it reproduces -- and worsens -- across two corpora and five encoders. The one signal that separates an attack from legitimate niche ingestion -- a query's demand -- is invisible before retrieval, which is also the escape: a retrieval-time detector that observes demand catches 100% of the attacks at the same 1% false-positive rate. Coverage of the query space by an admission gate is not containment of coordinated poisoning; robust defense must move past the front door, to demand. 

---
# Prior Audit-Repair Context Shifts LLM Verifier Thresholds Toward Leniency 

**Authors**: Parsa Mazaheri, Kasra Mazaheri  

**Link**: [PDF](https://arxiv.org/pdf/2608.16003)  

**Abstract**: Automated checking pipelines increasingly place one language model as the checker and another (or the same one) as the fixer. We ask whether that wiring changes what the checker reports. Measuring false alarms on human-verified-correct ProcessBench traces with the present task held byte-identical, we find that a completed audit -> repair episode already in the model's context lowers false alarms in 15 of 15 model x wording combinations, by 2.8 to 11.5 percentage points against a length-matched non-audit control, a 9 to 25% reduction relative to that control. The direction contradicts what the accumulated-message literature predicts: an episode whose audit reported an error lowers false alarms further still, at all five wordings on the model where that manipulation lands cleanly, though a negativity asymmetry predicts more flagging. Decomposing the episode finds repair content and audit verdict complementary: different components carry the effect on different model families. Signal-detection analysis locates the change in the threshold rather than in discrimination -- the criterion moves in 15 of 15 combinations and survives correction in 13 while d' survives in none, though the d' test is half as sensitive by construction -- and a hand audit of 50 false alarms finds 82% simply wrong, so at this operating point the shift need not be harmful. With reasoning enabled the effect keeps its relative size on both models tested, and the threshold reading holds there too. 

---
# A Scalable Pipeline for LLM-Teacher Distillation Labeling: Work-Stealing Job Scheduling and Memory-Aware GPU Concurrency 

**Authors**: Ravi Satya Durga Prasad Yenugula  

**Link**: [PDF](https://arxiv.org/pdf/2608.15975)  

**Abstract**: Labeling large text corpora with LLM teachers has become a practical route to training data at scale. At millions of items, hand-labeling every batch is not feasible, and two questions dominate: what label quality a teacher buys per dollar, and how to keep a fleet of GPU workers busy under skewed, failure-prone workloads. We present a simple, reproducible pipeline that addresses both. First, a work-stealing ring pool: each worker owns a queue, drains it first, and then steals from ring successors, with exactly-once task claims via atomic conditional writes and crash tolerance via stale-claim sweeping. The claim protocol requires only a compare-and-set primitive from its storage layer; we implement it on a single SQLite file, which makes the reference implementation dependency-free and the experiments reproducible on one machine. Second, a memory-aware concurrency rule that sizes per-node parallelism by how many model copies fit on the GPU, so the same code runs safely across device sizes. Third, a relabeling benchmark methodology in which the teacher relabels a public dataset that already has gold labels, so quality reduces to an agreement measurement and cost follows from measured throughput. Under skewed load the pool sustains up to 3.4 times the throughput of static sharding while matching it at zero skew, loses 0 of 2,000 tasks when half the workers are killed mid-run (static sharding loses 953), and yields measured quality and cost points for an instruction-tuned teacher on irony and sentiment tasks. All experiments run on public data and commodity hardware; code, tests, and run logs are released. 

---
# The Limits of Binding in Dual Encoders 

**Authors**: Kin Ian Lo  

**Link**: [PDF](https://arxiv.org/pdf/2608.15971)  

**Abstract**: Dual-encoder models such as CLIP score an image-caption pair by a single inner product of two independently computed unit vectors, and fail at binding, often scoring near chance when asked to distinguish "a red car and a blue dog" from "a blue car and a red dog". We give a mathematical account of when this failure is necessary and when it is contingent. Working within the ideal-encoder framework proposed by Kang et al., we first show the relevant axioms are satisfiable, so every impossibility must enter through an added, checkable hypothesis. We then prove three such obstructions. Depth: for recursive role-binding codes the swap margin obeys an exact law $m(D) = 2b^{-D}$ in the nesting depth D, with a finite-dimension version holding up to one explicitly flagged concentration estimate; the resolvable depth grows only logarithmically in the dimension and is single-digit at CLIP scale, the nesting depth of ordinary language. Objective: architecture-free throttle theorems showing that the contrastive objective's entire reward for binding is bounded by the rate at which training contrasts a caption against its own swap, a rate that vanishes at web scale, and that exactly reversed binding costs only that rate times the mean binding margin; both are verified in simulation. Geometry: a tight smoothness-binding frontier: the closer the two swap-related captions must embed to a shared paraphrase anchor, the smaller the binding margin can be, with an exact constant. Measuring its text-only diagnostic across 18 deployed text encoders, every model sits at roughly 25-35% of its ceiling, and the induced per-item ceiling tracks SugarCrepe's subset difficulty at r = 0.99. Binding failure in deployed dual encoders is thus not a dimension or smoothness limit today, but an incentive and code-structure limit, with a proved depth ceiling that remains once those are fixed. 

---
# Ask to Be Sure: Informative Interactions for Confident Multi-Turn LLM Recommendation 

**Authors**: Cedar Site Bai, Duanshun Li, Zhenyu Liao, Sheikh Sarwar, Huiyuan Chen, Yuan Chen, Changhe Yuan, Haiyang Zhang, Qilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2608.15949)  

**Abstract**: Recent advances in large language models (LLMs) have enabled their use as conversational recommender systems (CRS), demonstrating strong recommendation accuracy and natural dialogue. However, guiding multi-turn interactions to elicit user preferences effectively remains challenging. Existing approaches either use separate reinforcement learning agents with templated interactions or optimize for interactivity judged by another LLM, without measuring how much useful information is actually gained. We propose a new approach that quantifies the effectiveness of each interaction by the reduction in the assistant's uncertainty, measured via entropy over recommendations. We apply this entropy reduction as a reward---without relying on ground-truth recommendations, which are often unavailable in real-world scenarios---to fine-tune the LLM, enabling strategic interaction generation. Empirical results with supervised fine-tuning (SFT) and direct preference optimization (DPO) on the INSPIRED and ReDial datasets show that our method improves both recommendation quality and conversational efficiency. 

---
# Iterative Self-Learning for Expressive Text-to-Speech Synthesis 

**Authors**: Nicholas Sanders, Gustav Eje Henter, Simon King, Korin Richmond  

**Link**: [PDF](https://arxiv.org/pdf/2608.15910)  

**Abstract**: Expressive text-to-speech (TTS) systems that use explicit conditioning labels provide direct and interpretable control over expressive attributes, in contrast to reference-based or prompting-based approaches, but require labeled data. Obtaining these labels at scale is costly and time-consuming, yet no prior semi-supervised framework addresses this specific bottleneck. Existing semi-supervised TTS methods instead target scarcity of paired speech-text data or transcriptions. To address the scarcity of expressive labels, we propose an Iterative Self-Learning (ISL) framework for expressive TTS, built on Invert-Classify, a classifier-free method that recovers discrete expressive labels by inverting a frozen generative model. The framework iteratively pseudo-labels unlabeled speech using the current model, retrains on the combined labeled and pseudo-labeled data, and repeats, progressively refining label quality and synthesis. We validate on two expressive tasks, word-level prominence and utterance-level emotion, across multiple low-resource data splits. We find that iterative refinement can improve pseudo-label accuracy over single-pass baselines. Furthermore, we observe that these improvements in pseudo-labeling of expressivity translate to gains in expressive label adherence and synthesis quality, confirmed by objective metrics and human listening tests. In the most data-scarce conditions, ISL-trained models outperform single-pass pseudo-labeling and further approach fully supervised performance, demonstrating that gradient-based ISL is an effective solution to expressive label scarcity in low-resource TTS. 

---
# Large language model-assisted discovery of cohorts from scientific literature 

**Authors**: Moritz Sturm, Lisa M. Berg, Inken Berg, Harishny Sarma, Jasmin Hartmann, Denissa Girschik, Gemma Roig, Christine M. Freitag, Andreas G. Chiocchetti  

**Link**: [PDF](https://arxiv.org/pdf/2608.15909)  

**Abstract**: Background: Planning multi-study analyses requires identifying cohorts with the relevant participants, phenotypes, and data modalities. This process commonly relies on prior knowledge, cohort catalogues, and manual literature searches. We developed a complementary question-driven framework that searches relevant scientific literature and extracts explicit cohort names. Methods: The framework first generates multiple PubMed queries from configurable vocabularies and templates and retrieves the resulting scientific literature automatically through the PubMed API. A large language model then screens the retrieved titles and abstracts and extracts explicit cohort names using a prompt tailored to the research question. The extracted names are deduplicated with human review. Configurable code, prompts, and example outputs are available at this https URL. Evaluation: As a use case, we applied the framework to youth aggression genetics. From 5,400 generated PubMed queries, the framework retrieved 5,254 unique records and identified 188 candidate cohorts. Manual screening using predefined criteria, including participant age and genetic-data availability, retained 44 eligible cohorts. Automated LLM-based name extraction was within the agreement range of human annotators. We also searched four established cohort catalogues using the same research question. Their combined results contained 27 of the 44 eligible cohorts, while 17 were not returned by any cohort catalogue search. Conclusion: The framework converts research-question-specific vocabulary into screenable cohort inventories via a large, automated literature search. It can be adapted across populations, phenotypes, data modalities, and study designs, and provides a literature-based complement to curated cohort catalogues. 

---
# Large Language Models as Implicit Sociological Models: Reconstructing Voting Behaviour from Sociodemographic Profiles 

**Authors**: Roman Neruda, Martin Bakoš, Josef Šlerka, Vít Tuček, Petra Vidnerová, Gabriela Kadlecová  

**Link**: [PDF](https://arxiv.org/pdf/2608.15871)  

**Abstract**: Large language models (LLMs) trained on large-scale internet corpora encode extensive statistical regularities about social identities, attitudes, and political behaviour. This paper introduces and evaluates a methodological framework that leverages these latent representations to reconstruct aggregate voting behaviour from individual-level sociodemographic profiles. We operationalize LLMs as implicit sociological models by conditioning them on demographic descriptions, eliciting probabilistic turnout and party preferences, and aggregating individual outputs via a soft voting procedure. Using the 2021 Czech parliamentary election as a validation case, we demonstrate that contemporary LLMs reproduce official election outcomes with low mean absolute error, recover known political bloc structures, and align with independently established sociodemographic gradients. The contribution of this work is methodological rather than predictive: we show how LLMs can be systematically interrogated as compressed representations of social reality, offering a novel exploratory instrument for computational social science while clearly delineating its epistemic and ethical limits. 

---
# Beyond Visual CoT: Internalized Visual Thinking for Proactive Video Reasoning 

**Authors**: Xiaoyu Zhu, Xinke Deng, Suresh Taddewadikar, Arnab Kumar Mondal, Zhongyu Jiang, Ian Fasel, Joerg Liebelt  

**Link**: [PDF](https://arxiv.org/pdf/2608.15869)  

**Abstract**: Multimodal large language models increasingly use visual chain-of-thought (Visual CoT) to reason about spatial, temporal, and embodied environments. By generating intermediate reasoning images, Visual CoT provides an intuitive mechanism for visual foresight but introduces substantial inference overhead, which is particularly problematic for proactive video reasoning. We ask whether models can learn to think visually during training while reasoning directly at inference. We introduce Internalized Visual Thinking (IVT), a post-training framework that jointly optimizes textual prediction and next-embedding prediction over unlabeled videos. Given a partially observed video, IVT predicts latent representations of future frames together with the target textual answer, encouraging the model to capture motion, object transitions, interactions, and latent intent. At inference, IVT generates the answer directly without synthesizing or re-encoding future frames. We conduct controlled studies across target representations, decoder designs, prediction horizons, data mixtures, training curricula, and predictive objectives. IVT improves over direct-answer fine-tuning on all six evaluation settings while retaining the same inference pathway. Compared with explicit Visual CoT, IVT achieves comparable or better performance and reduces average end-to-end latency by more than 5x. Together, our findings suggest that explicit pixel-space generation at inference time, as used in visual chain-of-thought, may not be necessary for effective proactive video reasoning. Predictive world modeling can be internalized during training to produce multimodal reasoners that are both more accurate and substantially more efficient. 

---
# Scaling Manual-Grounded Appliance Manipulation with Data Synthesis and Unified Planning 

**Authors**: Yuxing Long, Lei Kang, Ziyan Yu, Yuzheng Gao, Bin Cheng, Jiyao Zhang, Xiaoqi Li, Haolin Yang, Dongjiang Li, Hui Shen, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2608.15863)  

**Abstract**: Operating household appliances requires long-horizon planning that is state-dependent and robust to disturbances, yet existing large models fall short, as no sufficiently diverse, task-oriented dataset exists to support such planning. To bridge this gap, we propose MAGE, a scalable data synthesis pipeline that introduces a novel Hierarchical Appliance Graph (HAG) to automatically generate part grounding, long-horizon planning, and closed-loop recovery data from appliance manuals. With MAGE, we build UseAppliance, the first large-scale dataset for manual-grounded appliance manipulation planning, spanning 22 appliance categories with 89K+ part annotations, 53K+ manipulation tasks, and 33K+ closed-loop adjustment steps. Built on UseAppliance, we develop AppliancePlan, an end-to-end model for manual-grounded appliance manipulation planning. On RealAppliance-Bench, AppliancePlan with only 7B parameters achieves over 10x the best baseline on open-loop planning and consistently outperforms state-of-the-art models across all tasks. Real-robot experiments on six household appliances further confirm effective sim-to-real transfer, marking an important step toward general-purpose household robotics. 

---
# Dense Expands, Sparse Anchors: Channel-Asymmetric Query Expansion for Hybrid Retrieval 

**Authors**: Chunran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.15851)  

**Abstract**: LLM-based query expansion improves retrieval by generating document-like passages. In hybrid retrieval, however, most evaluations fuse fixed top-$L$ dense and sparse rankings. Because the cutoff controls both which cross-channel contributions enter fusion and how much of each ranking is accessed, gains measured at one $L$ can change or reverse at another. We separate these effects by evaluating retrieval effectiveness under complete-list fusion and recording the policy-specific per-channel replay stopping depths at which its ordered top-$K$ is certified. We then introduce DESA (Dense Expansion and Sparse Anchoring), a channel-asymmetric query expansion method. An LLM generates complementary reference passages; orthogonal residual expansion adds their new semantic directions to the dense query, while score-product anchoring incorporates their lexical cues into sparse retrieval without broadening the original query's lexical support. Across seven BEIR datasets, DESA improves nDCG@10 and Recall@20 over the unexpanded query by 3.82% and 2.38%, while reducing dense and sparse access depths by 36.90% and 36.56%. With equal dataset weighting, 63.31% of queries become shallower in both channels. However, both depths increase with Contriever on Touché-2020. These results support channel-specific integration of generated passages and joint evaluation of retrieval effectiveness and access depth. 

---
# Schema-Agnostic Graph Reasoning Agent for Hybrid Knowledge Graphs 

**Authors**: Marius Dragic, Ruben Ifrah, Alexandre Rio  

**Link**: [PDF](https://arxiv.org/pdf/2608.15834)  

**Abstract**: Tool-calling LLM agents navigate unfamiliar codebases with a handful of generic primitives for listing, reading and searching files (ls, cat, grep). A knowledge graph admits the same interface: listing neighbours, reading node content and searching descriptions are the same operations on a different substrate. Building on this correspondence, we present GRA, a Graph Reasoning Agent that explores hybrid knowledge graphs, whose nodes are either textual concepts or relational tables, with seven generic tools, discovering everything domain-specific at run time. On UFK-M (Unified Factory Knowledge Model), an industrial benchmark of 258 analytical questions whose gold answers are produced by executing validated SQL programs, GRA beats a full-context agent by 5.1 pp (88.4% vs. 83.3%), while reading under a third of its input tokens. A graph-free control shows the gain comes chiefly from selective agentic access rather than graph topology, and that the effect depends on a model able to drive tools reliably. Seeing less, the agent answers better: selective navigation over a structured substrate beats exhaustive context. 

---
# KV-Rescue: Recovering Reasoning Language Model KV Eviction Loss via Stepwise Interleaving 

**Authors**: Minsoo Cheong, Woosang Lim, Vincent-Daniel Yun, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2608.15797)  

**Abstract**: KV-cache eviction caps the memory cost of long reasoning traces but is inherently lossy because the model decodes from a partial view of its history. Under aggressive budgets, this not only lowers accuracy but can also cause runaway degeneration, where the model produces incoherent or repetitive tokens until reaching the length limit. We characterize much of this loss as an information gapf caused by missing context, rather than a capability gap caused by limited model capacity. An evicted 7B model and a full-context 1.5B model make complementary errors, and an oracle choice between their answers recovers 79% of the accuracy gap to the full-KV 7B model. Based on this observation, we propose KV-Rescue, a training-free inference framework that bridges the information gap introduced by KV eviction using a lightweight full-context helper. KV-Rescue interleaves reasoning steps from the two models into a shared trajectory. An online detector uses entropy and compressibility to terminate the generation of incoherent or repetitive base-model candidates early. Across five math benchmarks with Qwen2.5-Math 7B and 72B, KV-Rescue recovers an average of 87% of the accuracy lost to eviction at eviction budget B=64. A decode-cost analysis further shows that preventing runaway degeneration cuts base-model token generation by 43% on average. 

---
# Routing Divergence Is Not Evidence of Behavioral Influence in Same-Weight MoE Self-Distillation 

**Authors**: Cedric Caruzzo, Donggeun Yoo, Tae Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.15787)  

**Abstract**: Two Mixture-of-Experts (MoE) forward passes can share every weight yet route the same token through different experts. This creates a possible blind spot in same-weight self-distillation, where a demonstration-conditioned teacher supervises a query-only student. We study this mismatch in its single-step form, with frozen weights rather than as a proxy for a full training trajectory. An exact blockwise decomposition separates a routing term, which changes gates at fixed content, from a dense-like content term. Across seven open-weight checkpoints and two domains, the routing term spans only $1.6\times$ as a fraction of block output, while its residual-stream exposure spans $3.2\times$. Exposure is ordered by the routed block's share of the residual. Scaling the always-on backbone in two confirmatory models moves exposure monotonically; common-mode controls support a mass-and-coherence mechanism rather than denominator dilution alone. Preregistered PubMedQA patches on three models show that the full routing term moves outputs by less than half the natural context effect and is largely reproduced by matched-norm noise, whereas the content term is strongly direction-specific. Scale and merged-expert probes show that the narrow block-level range is not universal, although exposure remains small at the tested boundaries. Router movement alone is therefore not evidence of behavioral influence: measure exposure first, and use a behavioral intervention when the decision matters. 

---
# Propaganda Forensics: Recovering the Generation Pipeline of an AI-Driven Influence Campaign 

**Authors**: Benjamin Icard, Elouan Vuichard, Louis Lefebvre, Lila Sainero, Thomas Girault, Alice Breton, Tanguy Launay, Gauvain Bourgne, Morgane Casanova, Guillaume Gadek, Victor Klötzer, Michel Le Nouy, Guillaume Gravier, Jean-Gabriel Ganascia, Paul Égré  

**Link**: [PDF](https://arxiv.org/pdf/2608.15746)  

**Abstract**: We present a forensic analysis of the generation pipeline behind a recent AI-driven influence campaign. We introduce PROPAGIA, a corpus of 2,646 propagandist French articles from the Storm-1516/CopyCop campaign disclosed by VIGINUM and INSIKT GROUP in 2025. For comparison, we rely on SIPA, a corpus of human-written French mainstream press from the same period. Using topic modeling, vagueness and sentiment analysis, we first isolate persuasion techniques characteristic of propaganda, with PROPAGIA far exceeding SIPA in vagueness, subjectivity and negativity, and citing fewer sources. We then find prompt instruction leaks on 50 of the 84 PROPAGIA websites, including a verbatim ten-point editorial specification accounting for several of these differences, together with high cross-article redundancy. Finally, we show that rewriting-based detection supports INSIKT GROUP's attribution to the Llama 3 family, but also suggests the involvement of Mistral-family models. 

---
# Beyond Single Object: Learning 3D Relations with Large Language Models 

**Authors**: Kohsuke Ide, Ryousuke Yamada, Yue Qiu, Xianzheng Ma, Yoshihiro Fukuhara, Hirokatsu Kataoka, Yutaka Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2608.15710)  

**Abstract**: We address a fundamental gap in 3D-LLMs: existing models focus on single-object/scene description, struggling with detailed, inter-object comparison. We propose a framework for detailed object-level reasoning across multiple objects with three components: (1) MO3D (Multi-Object in 3D), an instruction dataset requiring fine-grained multi-object comparison; (2) Multi-3DLLM, using a minimal Patch-Interaction Transformer (PIT) that models inter-/intra-object relationships while preserving local geometry; (3) Mini-apps, two application-driven benchmarks (Shape Mating, Change Captioning) that probe geometric understanding for practical use. Recent 3D-LLMs and 2D-VLMs perform poorly on these tasks, lacking both comparison-centric design and geometric awareness. In contrast, Multi-3DLLM trained on our mixture data learns geometric reasoning, surpasses all baselines on MO3D, and provides positive transfer to single-object classification. 

---
# Integrating Persuasion Theory into the Epidemiological Modelling of Health Misinformation Spread on Social Media 

**Authors**: Mkululi Sikosana, Sean Maudsley-Barton, Oluwaseun Ajao  

**Link**: [PDF](https://arxiv.org/pdf/2608.15689)  

**Abstract**: This study presents a hybrid epidemiological and behavioural framework to simulate the spread of health misinformation on social media. We extend the classical Susceptible--Infected--Recovered (SIR) model to a six-compartment structure (SIRMMM), incorporating Misinformed Susceptible (MS), Misinformed Infected (MI), and Misinformed Recovered (MR) compartments to better reflect the dynamics of the misinformation lifecycle. To account for individual-level behavioural variation, we extend the SIRMMM model by integrating psychological signals from the Elaboration Likelihood Model (ELM), including sentiment polarity, engagement metrics, and cognitive effort, which dynamically modulate the misinformation transmission rate, yielding the ELM-SIRMMM framework. Model parameters were estimated using the FibVID dataset, which captures COVID-19 misinformation on Twitter. Generalisability was tested on two additional datasets: MC-Fake (emotional misinformation) and Monant (general health misinformation). Results show that the ELM-SIRMMM model enhances both predictive accuracy and dynamic realism. On FibVID, it decreases RMSE by 5.5%, delays the misinformation peak from day 150 to day 160, and increases its peak prevalence from 6% to 7%. On MC-Fake, it accurately reproduces a flash-rumour pattern, infecting 38% of users by day 45 and achieving 97% misinformation recovery, all while maintaining model accuracy. In contrast, minimal behavioural signal variability in the Monant dataset leads to marginal benefit, with only a 3% peak and 57% of users remaining susceptible. These findings suggest that structural elaboration alone is insufficient. Functional realism in modelling misinformation spread requires dynamic psychological inputs that vary meaningfully across time and contexts. 

---
# Do Assessment Instruments Measure the Same Thing for Humans and LLMs? A Latent Structure Analysis 

**Authors**: Alona Strugatski, Licol Zeinfeld, Giora Alexandron  

**Link**: [PDF](https://arxiv.org/pdf/2608.15630)  

**Abstract**: The rapid development and growing deployment of large language models (LLMs) have made it increasingly important to understand their capabilities. A common approach is to evaluate LLMs using assessment instruments originally designed to measure skills and competencies in humans, such as standardized exams, and to use performance on these instruments as evidence for generalizable claims about LLMs' underlying abilities on the same skills the assessments are intended to measure in humans. However, from a validity perspective, such inferences require that the relationship between observed performance and underlying constructs established for humans also holds for LLMs. In particular, a necessary condition for transferring score interpretations is similarity in the latent structure of responses to the assessment. In this study, we examine whether this condition holds in two educational contexts: high-school chemistry and a quantitative reasoning section of a university entrance exam. Using a case study design, we compare human response data with responses generated by six multimodal LLMs. Our analytical approach combines exploratory factor analysis, factor congruence, and resampling to assess latent structure similarity across human learners and LLMs. Across both instruments, we find systematic differences between human and LLM factor structures, showing evidence that the analyzed assessments may not measure the same constructs for humans and LLMs. These findings call into question the validity of evaluation practices that use educational assessments to make claims about AI capabilities. 

---
# Large Language Model Assisted Operational Monitoring for Battery Energy Storage System Integrated Power Distribution Networks 

**Authors**: Azmeer Akhtar, Md Fazley Rafy, Anurag K. Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2608.15396)  

**Abstract**: Battery energy storage systems (BESS) are increasingly used in distribution networks for voltage regulation and demand response, which increases the volume and complexity of operational telemetry available to grid operators. This paper presents an AI-enabled monitoring framework that connects a large language model (LLM) interface with a structured telemetry database for BESS-integrated distribution system analysis. Operator questions are submitted in natural language and translated into validated SQL queries using predefined database schema information and approved KPI views. Retrieved measurements, including bus voltages, state of charge, active power, and reactive power, are evaluated against engineering constraints for voltage limits, BESS operation, and demand response tracking. The framework is validated using hardware-in-the-loop co-simulation data from a BESS-equipped distribution feeder operating under reactive power-based voltage control and price-driven demand response. Case studies show that the framework generates valid database queries, identifies repeated voltage violations, detects reactive power overshoot, and evaluates active-power tracking performance. The results show that LLM-assisted monitoring can connect structured grid telemetry with automated engineering assessment for BESS operation analysis. 

---
# VTInstructor: Visual Trajectory Prompting for Navigation Instruction Generation in Continuous Environments 

**Authors**: Haolin Yang, Yuxing Long, Zihan Yang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2608.15284)  

**Abstract**: Navigation instruction generation from ego-centric RGB video in continuous environments is an important yet challenging task for human-robot interaction and scalable dataset construction. Prior instruction generators assume discrete viewpoint graphs with panoramic observations, where trajectory structure is explicit; in continuous environments, however, the agent receives only a dense RGB stream, making trajectory cues difficult to recover. We propose VTInstructor, the first VLN instruction generation framework for continuous environments. Our key idea is to convert implicit trajectory geometry into explicit visual trajectory prompts: EDTC condenses long RGB trajectories into navigation-critical keyframes, VTP overlays path, turn, and goal cues onto these anchors, VTMod injects the resulting trajectory signals into the visual encoder, and VT-GRPO further calibrates this spatial injection during training, all without requiring a navigation graph, pre-built map, or scene reconstruction. On the challenging R2R-CE and RxR-CE Val Unseen benchmarks, VTInstructor sets a new state of the art across all standard NLG metrics, surpassing the strongest baseline by +0.357 CIDEr and +0.109 CIDEr, respectively. Beyond automatic metrics, VTInstructor-generated instructions raise a frozen follower's success rate to 63.3%, a +14.7 percentage-point gain over the best competing instruction source, and provide consistent data augmentation gains of +3 SR points on downstream navigation tasks. 

---
# Demographic Injection in Medical Language Models under Diversity, Equity, and Inclusion Prompts 

**Authors**: Diego Mardian, Frank Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.15254)  

**Abstract**: Clinical-AI guidance increasingly recommends prompting language models to reason with attention to diversity, equity, and inclusion (DEI). We measure a side effect that misrepresents patients: a one-sentence DEI prompt appended to a medical question leads models to add patient demographic attributes (race, socioeconomic status, sex) the question never stated, in effect rewriting who the patient is. We call this demographic injection. Across 47 models, four medical benchmarks, and 376,000 responses scored by a validated model-judge pipeline, a single DEI prompt raises the injection rate from 0.7% to 33.1% (47x) in all 47 of 47 models, attributable to the equity content rather than to added length (18x above a length-matched control; p=1.4x10^-14). Most added content is a general population statement that leaves the answer unchanged, but a smaller subset attaches an attribute to the specific patient or changes the selected option (0.25-2.4% of responses, 99.8% toward the incorrect option), where the invented demographic changes the answer the model recommends. Phrasing scales the effect from 14% to 56%. DEI prompts are just one example of a more general mechanism. Any instruction that nudges how a model reasons can make it add unrequested details, including details about the patient. Flagged outputs are treated as model errors under study, not clinical guidance. 

---
# Evo-Harness: Context-to-Harness Skill Compilation for Self-Evolving Agents 

**Authors**: Tianxin Wei, Zhan Shi, Minhua Lin, Bing He, Zewen Liu, Yisi Sang, Yuanchen Bei, Xuying Ning, Jiaru Zou, Ting-Wei Li, Xiao Lin, Yanjun Zhao, Chi Wang, Benoit Dumoulin, Dakuo Wang, Jingrui He, Hanqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.15071)  

**Abstract**: Learning from experience is critical for developing capable, self-improving large language model (LLM) agents. Existing methods typically extract knowledge from accumulated trajectories via reflection, memory, rules, or skills. However, agents in realistic environments continuously encounter novel tasks, often offering only a one-shot opportunity to improve. These executions yield rich but highly noisy contexts, entangling broadly useful lessons with task-specific artifacts. Critically, prior works rarely validate their effectiveness on complex real-world tasks or isolate the underlying drivers of improvement. To address these gaps, we formulate online harness learning, where a frozen agent improves by continually updating a structured harness across sequential tasks. This formulation enables a systematic study of key self-improvement factors through our proposed Evo-Harness. At its core, context-to-harness skill compilation distills noisy, single-shot executions into reusable skill harnesses for cross-domain and topic-level adaptation. To demonstrate the efficacy of one-shot skill compilation, we evaluate across five realistic benchmarks (TerminalBench2, SWE-bench, CL-Bench, -bench, WebArena-Infinity). Our extensive analysis demonstrates the effectiveness of Evo-Harness and provides a principled understanding of how LLM agents can effectively learn on the fly. Our code is available at this https URL. 

---
# Gathered, Not Admitted: How Attention Brings a Latent Variable into Verbalizable Form 

**Authors**: Parsa Mazaheri  

**Link**: [PDF](https://arxiv.org/pdf/2608.15022)  

**Abstract**: Language models hold latent quantities in a form they can report on, and more of a quantity is present in that form when the task requires reusing it flexibly. What causes a representation to enter that form is open, and the word workspace invites an admission story: a gate that decides what gets in. Testing it on open-weight models with Jacobian lenses, over a benchmark whose five arms share an identical context, we find no gate where it predicts one. Demand raises a concept's lens visibility beyond what applying an operator to a supplied value produces: +0.050 [+0.045, +0.057] in percentile rank on our primary checkpoint, positive on all four we measure, though that arm answers at ceiling and the accuracymatched contrast is stronger under that readout. At the same time one shared linear map decodes the variable from every arm, the control included, at 6.4-9.0x its selection-corrected floor. What produces the later readable form at the queried position is attention-mediated gathering inside a mid-depth window: separating patch depth from readout depth puts transport there at least 17x above anywhere shallower under non-saturating readouts, with no tested MLP output contributing positively inside it. Under the saturating percentile rank the same grid does not localise the window, which is a fact about that measure. An arm that needs the variable for nothing concentrates sevenfold less, so the window is demand-specific. That window has two measured edges, a survival failure below and destruction above, and it falls at the same fractional depth in a 64-layer hybrid and a 62-layer dense model from another family. We localise where the variable is installed and read, not the route from the passage, which transports nothing. But the readout is not a calibrated measure of use: three components move it to within 12% of one another and differ 7.4x in what they do to the answer. 

---
# Does a Tool Result Carry More Authority Than Plain Text? Three Prospective Studies of False-Claim Adoption in a Synthetic Assignment Task with Claude Opus 5 

**Authors**: Justin Bronder  

**Link**: [PDF](https://arxiv.org/pdf/2608.14992)  

**Abstract**: Language-model systems increasingly read from stores they also write to, so a claim that was merely written earlier can return looking retrieved. We tested whether the message package carrying an unsupported assignment changes which answer a model gives in a synthetic lookup task. Claude Opus 5 selected a color code for a named item or abstained. In an exploratory four-arm study, false-code adoption was 0/24 with no target claim, 0/22 scorable trials when a prior assistant assertion named the target, 14/24 when a tool-result record named it, and 15/24 when that result used a ten-field metadata wrapper that marked it unchecked. The tool-result arm selected the record's code in 11/12 supported trials and 14/24 unsupported trials, ruling out a fixed output-token bias while leaving substantial planted-token heterogeneity. A document-preregistered replication reproduced the tool-result versus assistant-assertion gap, 7/24 against 0/24, one-sided Fisher exact p = 0.0047. The tool-result rate nevertheless fell from 14/24 to 7/24 across runs made four days apart. A second preregistered study gave the earlier comparison a live text control: both records were announced in advance and placed in the same final user turn, then target binding was swapped between the linked tool result and later inline JSON. Inline text was sufficient for false-code adoption in 60/60 trials; the tool-result condition produced 57/60, so the registered result-first superiority criterion failed, p = 1. The result does not show that tool results have no effect. It shows that native tool-result placement was not necessary and that this experiment did not find greater behavioral weight for the result package than for announced inline text. The findings concern a single model on one synthetic task template, accessed through one API. 

---
# T-LLM Compiler: Trusted LLM-based Code Optimization and Verification Framework 

**Authors**: Zahra Fazel, Sunanda Gamage, Shayan Shirahmad Gale Bagi, Amir H. Ashouri, Tomasz S. Czajkowski, Bryan Chan, Reza Azimi, Yaoqing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.14953)  

**Abstract**: Recent advances in Large Language Models (LLMs) have opened opportunities to apply high-level code transformations to the field of code optimization, and it has since emerged as one of the most fundamental tasks for LLMs to perform; however, at present, LLMs struggle to apply wide-ranging code optimization tasks due to both the complexity of the code and the inability to independently verify the correctness of the transformations. In this paper, we present the Trusted LLM (T-LLM) Compiler, which proposes an advancement in compiler technology through a collaborative effort involving high-level LLM code transformations, traditional compilers, and verification tools. Experimental results reveal that it can significantly improve code correctness when tested on a set of PolyBench/C benchmarks. Our approach facilitates iterative code optimization efforts with verification strategies that enable corrective actions. Through this approach, T-LLM Compiler achieves code optimization accuracy of up to 83.3% and a speedup of up to 16.1\% on the PolyBench/C benchmarks, with the transformed code reaching an average of 26.7% speedup wrt standard baselines. Additionally, we release the project's source code to the open-source community. 

---
# Trust Is Not Enough: Influence Calibration for On-Policy Self-Distillation in Agentic RL 

**Authors**: Qizhen Lan, Xi Xiao, Xiangchen Guan, Mengchen Fan, Moule Lin, Jung Im Choi, Lijing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.14945)  

**Abstract**: On-policy self-distillation (OPSD) gives language agents dense token-level supervision from a privileged self-teacher on the policy's own trajectories. Existing methods allocate this supervision mainly by teacher trust, but trust does not reveal whether emphasizing a token supports the current policy objective. We call this the trust-utility mismatch and introduce Influence Calibration for Self-Distillation (ICSD). For each supervised token, ICSD measures the first-order response of its importance-weighted RL surrogate contribution to a teacher-directed output perturbation. Batch-adaptive calibration converts this non-stationary signal into a bounded allocation weight while preserving the original auxiliary-loss mass within each action turn. These detached weights affect only the distillation loss and require no additional model pass. Across ALFWorld, WebShop, and Search-QA, ICSD improves all matched aggregate metrics over trust-only allocation under Group Relative Policy Optimization (GRPO) and Group-in-Group Policy Optimization (GiGPO), across two model families spanning 1.5B to 7B. At 7B, it reaches 96.1% ALFWorld success and a WebShop score of 93.1. Frozen-batch analyses show that ICSD reduces teacher-supported mass assigned to objective-opposed tokens from 60.1% to 37.8% and raises cosine compatibility with the RL gradient by 0.192. A companion repository is avail- able at this https URL. 

---
# SkillComposer: Learning Reusable Skills for Natural-Language Robot Programming 

**Authors**: John Woods, Hasti Seifi  

**Link**: [PDF](https://arxiv.org/pdf/2608.14944)  

**Abstract**: Natural-language interfaces can lower the barrier to programming robots, but existing systems struggle when users request complex tasks. While large language models (LLMs) perform well with simple commands, they often struggle to generate code for multi-step tasks, decompose high-level instructions, or reuse prior solutions. We present SkillComposer, an interactive natural-language robot programming system for simulation environments that continually learns reusable program abstractions. SkillComposer uses a generate-test architecture in which an LLM iteratively generates and revises robot programs before execution. Successful programs are stored and processed by an online library-learning algorithm that compresses recurring function sequences into reusable macro skills for future tasks. We evaluate SkillComposer through ablation experiments and a user study with 12 participants to determine its effectiveness on manipulation and robot caregiving tasks. The results show that evaluator-guided generation and learned abstractions improve success rates and usability while reducing user effort in natural-language robot programming. 

---
# LLMs Can Predict Failure Risk, But Struggle to Predict Which Collaboration Protocol Pays Off: Cost-Aware Protocol Routing Across Reasoning Tasks 

**Authors**: Chih-Hsuan Yang, Jingyan Jiang, Cheng-Hau Yang, Vikram Vasudevan, Huihuo Zheng, Venkatram Vishwanath, Rajeev Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2608.14927)  

**Abstract**: Multi-agent large language model (LLM) systems can improve reasoning by spending more computation, but deployment requires deciding when extra collaboration is worth its cost. We isolate this decision by running every problem under four protocols while holding the solver fixed within each setting: direct solving (Baseline), iterative self-correction (Single), planner-executor-reviewer collaboration (PER), and multi-agent deliberation (Broadcast). The primary benchmark comprises 4,181 competition-level math problems; paired robustness checks cover four benchmarks spanning competition math, biology, and broader science with two solver families. Across fixed policies, trained routers, and frozen LLM routers, conservative policies under-escalate, whereas higher-solve frozen routers often over-escalate. A post-answer, pre-collaboration gpt-oss-120b probe ranks Baseline failures with 0.8847 AUROC (4,151 parseable cases; 95% CI [0.8732, 0.8955]). The same score remains informative for predicting whether any collaboration helps (0.7683 AUPRC), but is much weaker for identifying PER- or Broadcast-specific value (0.1674 and 0.1041 AUPRC). Separately, the pre-answer self-confidence gate reaches 78.0% solve at 45K tokens, compared with 73.8% at 71.3K for a frozen gpt-oss-120b router and 92.4% for a retrospective fixed-order oracle. Across 10 paired model-condition settings, the oracle adds 23.2-58.3 points of retrospective coverage over Baseline, but protocol profiles vary by task. In the six settings with held-out router evaluations, oracle gaps remain 18.5-28.9 points. Confidence can therefore support initial escalation, while protocol-specific cost-aware routing remains unresolved. 

---
# Optimal Watermark Localization in Mixed-Source Large Language Model Texts 

**Authors**: Jose H. Blanchet, T. Tony Cai, Xiang Li, Hao Liu, Qi Long, Weijie J. Su  

**Link**: [PDF](https://arxiv.org/pdf/2608.14906)  

**Abstract**: Watermarking provides a principled way to authenticate text generated by large language models (LLMs). In practice, however, the final text may be mixed-source, with watermark evidence surviving at only a subset of token positions after rewriting, insertion, deletion, or paraphrasing. Although prior work has studied global detection of watermark signals, when such signals can be localized remains unclear. We formulate watermark localization as a token-level multiple-testing problem based on pivotal statistics, with a latent indicator recording whether watermark dependence survives at each position. Under an asymptotic regime indexed by exponents for signal sparsity, next-token concentration, and effective-vocabulary growth, we derive a sharp boundary for global detection and phase transitions for discovery and classification within the class of coordinatewise pivot-based localization rules. We show that discovery is strictly harder than detection and that consistent classification is impossible across the parameter regime within this class. We then develop an adaptive thresholding method that does not require knowledge of the exponents or time-varying next-token distributions, but uses a data-driven estimate of the surviving watermark fraction. The method attains the optimal discovery boundary and near-optimal discovery power relative to homogeneous pivot-based rules. Simulations support the theoretical phase transitions, while experiments on model-generated texts demonstrate practical localization performance under common edit mechanisms. 

---
# Personalized Auto-Research: Towards a True AI Co-Scientist 

**Authors**: Bo Ni, Franck Dernoncourt, Hongjie Chen, Yu Wang, Nesreen K. Ahmed, Zhengzhong Tu, Tyler Derr, Ryan A. Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2608.14881)  

**Abstract**: AI co-scientists that generate hypotheses, retrieve related work, design experiments, execute code, and draft full papers are beginning to change how research is carried out. Despite this rapid progress, state-of-the-art systems remain researcher-agnostic: given a research goal, they optimize novelty, validity, or reviewer score while ignoring the individual scientist who will use the output. This overlooks a fundamental fact about research, namely, that what counts as novel, valuable, or feasible depends on the researcher, including their prior work, methodological repertoire, and the collaborators and communities in which they are embedded. In this work, we introduce the problem of personalized auto-research, which conditions every stage of the research process on a representation of the individual researcher. We argue that personalization is not a convenience layer, but rather the fundamental property that allows an AI system to serve as a genuine co-scientist rather than a generic instrument. To address this problem, we propose a general and flexible framework that threads a graph-grounded researcher context through retrieval, hypothesis search, experimentation, writing, and review. The framework consists of three fundamental components: (i) graph-grounded researcher representations, (ii) personalization across the full research pipeline, and (iii) evaluation grounded in the individual. Notably, we highlight a one-size-fits-all failure mode where distinct researchers issuing the same goal receive essentially the same research, erasing the tacit knowledge through which novel ideas arise. Finally, we discuss fundamental open problems and challenges. 

---
# Workspace Topology as an Attack Vector in Agentic Coding Assistants 

**Authors**: Alexandre G.R. Day, Pradeep Yadlapalli, Sriram Venkatapathy, Thomas Paniagua, Nick Raines, Sahil Wadhwa, Himanshu Kumar, Andy Luo, Sudeep Panyam, Rikhiya Ghosh, Pranab Mohanty, Giri Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2608.14876)  

**Abstract**: Agentic coding assistants are finding widespread use, not just in new code development but in quickly ingesting and leveraging third-party code. This opens up a risk of malicious code being ingested as these coding tools operate with broad filesystem access inside developer workspaces. In this paper, we extensively study the impact of different dimensions of a novel attack surface we term workspace topology -- defined via directory depth, codebase modularity, in-file injection position and context framing -- on the attack success rate of adversarial prompt injection attempts.
We perform an empirical study of indirect prompt injection (IPI) across a diverse set of open-source repositories spanning 10 languages and 6 engineering domains, evaluating three IPI entry points against open-weight models operating open source code harnesses.
We find that workspace topology measurably affects IPI success. Specifically, changes in codebase modularity can significantly alter the Attack Success Rate (ASR), with highly modular environments demonstrating significantly lower attack success rates. Furthermore, context framing and introduction of security-cues in the workspace can also alter the ASR. Our findings offer practical value for the evaluation and security testing of coding agents across diverse settings, while underscoring the importance of an uncontaminated testing environment to obtain reliable results and conclusions. 

---
# The Recall Trap: A Recall-Maximizing Retriever Configuration Reduces Issue Resolution in Fixed-Budget Code Context 

**Authors**: Alexander Adkins, Teimuraz Trapaidze  

**Link**: [PDF](https://arxiv.org/pdf/2608.14838)  

**Abstract**: Retrieval components for code assistants are tuned against retrieval metrics: a configuration that raises recall@k is adopted, and downstream task success is assumed to follow. We report a controlled case study in code repair, not a new phenomenon but a deployed-flag, execution-graded instance of the known relevance-diversity and objective-mismatch tradeoff (Levy et al., 2025). On SWE-bench Verified we inject a retriever's hits as a fixed 12-slot context pack with no search tools and toggle one flag (one-chunk-per-file deduplication) on an otherwise identical stack. The flag is the higher-recall configuration (gold file present in 0.878 of served packs against 0.806 disabled), yet disabling it, trading file breadth for within-file depth, raises the single-shot resolve rate: gpt-5.6-sol +7.6pp (39.2% to 46.8%, n=500, McNemar exact p=0.0003), and a pre-registered open-weights replication any reviewer can re-run (Qwen3.6-27B, +3.6pp, n=499, p=0.0133); both survive repository-clustered inference. The gain tracks within-file anchor dose, and a random-chunk control refutes an argmax-selection artifact. We map where it holds: it reverses on a lexical BM25 retriever (-3.2pp, significant cross-paradigm interaction), is not detected under unrestricted-Read agents (a powered null), and across four languages (SWE-PolyBench, N=617) is positive but not significant (+2.6pp, p=0.056), a mapped boundary rather than a confirmed extension. Operationally, at a tight fixed budget: do not hard-deduplicate by file, and A/B packing policies against the task, not the metric the flag was tuned to. 

---
# MINT: Min-Selection Preference Distillation for Balanced Multi-Objective Alignment 

**Authors**: Tony Tu, Sayan Chakraborty, Ruomeng Xu, Tony Qin, Austin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2608.14828)  

**Abstract**: Aligning a language agent to several objectives at once is a persistent failure mode of preference-based training: when objectives are combined additively, optimization collapses onto whichever is cheapest to improve and sacrifices the rest, so a support agent learns to sound warm while giving no real help. The root issue is that an additive reward has no notion of balance. We introduce Mint (MIN-selection preference disTillation), a one-line change to preference distillation: rather than ranking sampled candidates by a weighted sum of rewards, we rank them by their weakest objective, distilling the best-balanced candidate over the most lopsided one with an unchanged DPO objective. This is the p -> negative infinity limit of a generalized-mean family spanning additive to worst-case selection. Across cooperative emotional support and adversarial negotiation, min-selection lifts both objectives while sharply cutting their imbalance; on emotional support it raises the weaker axis from 0.37 to 0.64 (p < 10^-40), surpassing human experts and persisting across full multi-turn rollouts. A turn-by-turn analysis yields our central finding: min-selection corrects imbalance in proportion to how imbalanced the reference policy is, and its benefit endures over an interaction precisely as long as that imbalance does. 

---
# Do LLMs Know What to Ask and When? Evaluating Multi-Turn Information Seeking 

**Authors**: Yepeng Huang, Jiawen Zhang, Michelle Dai, Xiaorui Su, Shanghua Gao, Zi Wang, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2608.14808)  

**Abstract**: When a user question is underspecified, a capable model should recognize that its context is insufficient, identify the missing information, ask for it, and respond only once that information determines a unique answer. We formalize multi-turn information seeking as solving a k-underspecified constraint satisfaction problem, where k is the number of variables jointly required to determine the target and therefore measures the degree of missing information. We instantiate the formulation in MT-InfoSeek, a controlled evaluation suite of 5,251 problems and 9,006 task instances spanning mathematics, logic, biology, medicine, and general knowledge. We evaluate models along three axes: what they ask, when they ask it, and how the acquired information affects the final answer. Performance degrades across models and domains as underspecification increases. Models recognize that additional information is needed but underestimate how much, and in logical problems at k = 2 they under-predict the degree of missing information about four times as often as they over-predict it. They also fail to identify a minimal sufficient set of queries, improve only marginally when given the true k, and often stop before acquiring sufficient information. In tasks with ordered dependencies, an incorrect query order reduces final accuracy even when the model eventually acquires all necessary information. We measure information seeking directly through final sufficiency, which records whether the acquired information determines the target independent of answer generation. This separation shows differences between models that final accuracy alone does not capture, and indicates that the ability to seek information over multiple turns is distinct from the ability to generate answers and is not measured by current LLM evaluations. 

---
# From Positionwise Confidence to Prefix Scheduling: Verifier Skipping in Speculative Decoding 

**Authors**: Haoxuan Luo, Jameson Sandler, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2608.14787)  

**Abstract**: Speculative decoding is a leading technique to reduce the cost of autoregressive generation by using a small drafter to propose several tokens, which are then verified in parallel by a larger target model. Speculative diffusion decoding (SDD) further removes sequential drafting by generating every position in a draft block in parallel with a discrete diffusion model. However, SDD still invokes the target on every block, leaving verification as a potential bottleneck. This paper recognizes that this creates a new control handle: whether to invoke the verifier at all. Thus, we study verifier skipping, a lossy policy that commits a selected draft prefix directly, and ask which confidence signal should schedule it. Interestingly, our study finds that better token predictors need not yield better schedulers: skips require contiguous high-confidence prefixes, while short skips can induce additional drafting rounds. To study this mismatch, we compare raw confidence with learned marginal and conditional survival scores under the same policy, using Strict SDD, lenience, and top-$k$ acceptance as baselines. On HumanEval with DiffuCoder-7B-Instruct and Qwen3-32B, all three confidence signals save $9.6\%$ to $13.5\%$ of verifier calls at the same observed pass@1 as Strict SDD. Surprisingly, raw confidence saves the most; marginal survival has higher positionwise AUROC than raw confidence at most positions, yet neither learned signal dominates online. Our analysis shows that verifier skipping is a useful new lossy axis and, surprisingly, its key challenge is prefix scheduling rather than token prediction alone. 

---
# From Errors to Proofs: Minimal-Core-Guided Repair for Neuro-Symbolic Constraint Solving 

**Authors**: Dipankar Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2608.14771)  

**Abstract**: Making language models solve constraint problems reliably often means having them translate the problem into a formal specification and delegating the search to a sound solver. But the translation is itself a language-model task, and an unfaithful translation makes the solver faithfully solve the wrong problem. Existing pipelines repair only translations that crash, returning the solver's error message and falling silent when the program runs but is wrong. We replace the error message with a proof: when the generated program is unsatisfiable, we extract a minimal unsatisfiable core over the model's own constraints and hand it back the exact set that cannot hold together, a leakage-free signal that localizes the fault. On a new benchmark of 77 problems with an exact oracle, translation to Answer Set Programming is faithful on six of seven domains and fails only on aggregate coverage scheduling, which concentrates the translation tax in one diagnosable pattern. A minimal core, rather than a bare error, is what stops a weaker model from fabricating solutions to infeasible problems, cutting fabrication from 79% to 7%. A strong chain-of-thought baseline meanwhile matches the symbolic route on accuracy, so the route's value is not accuracy but certificates and its refusal to fabricate. 

---
# NARRATE: A Multimodal Real-World Australian Driving Dataset for Human-Centred Explanations in Automated Driving 

**Authors**: Ashkan Yousefi Zadeh, Zishuo Zhu, Xiaomeng Li, Andry Rakotonirainy, Sebastien Glaser, Ronald Schroeter, Patricia Delhomme, Zahra Mehraban  

**Link**: [PDF](https://arxiv.org/pdf/2608.14767)  

**Abstract**: Automated vehicles must explain their decisions in ways that passengers can understand, monitor, and trust. Existing language-annotated driving datasets are mostly observer-written, post-hoc, simulation-based, or generated from sensor inputs, rather than elicited from the driver performing the action. We introduce NARRATE, a multimodal real-world Australian driving dataset comprising 2,050 annotated events from 35 experienced drivers and driving instructors on public roads. Each event is grounded in synchronised visual, localisation, motion, and LiDAR streams and paired with in-vehicle and/or post-drive free-text explanations. NARRATE provides action labels, scenario-context labels spanning six high-level and 32 fine-grained categories, and span-level Situational Awareness (SA) annotations over driver explanations for Perception, Comprehension and Projection. Four benchmark tasks (SA, scenario-context, driver-action classification, and explanation generation) show that this structure is learnable from driver language, while fine-grained context recognition and explanation generation remain challenging. NARRATE paves a path towards more human-centred and domain-aware explanation models for automated driving. 

---
# VideoGAIA: A Benchmark for General AI Assistants on Agentic Video Understanding 

**Authors**: Fan Zhang, Guangming Yao, Jinyang Wu, Hao Wu, Zheng Lian, Xinyu Geng, Jingdong Chen, Yi Yuan, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2608.14718)  

**Abstract**: Video understanding is a fundamental task for evaluating the capabilities of multimodal large language models (MLLMs). However, existing leading models have already achieved approximately 90% accuracy on the Video-MME leaderboard, suggesting that conventional single-turn video understanding tasks are becoming increasingly saturated and insufficient for assessing the intelligence of advanced MLLMs. Towards this end, we introduce VideoGAIA, an agentic video understanding benchmark for general artificial intelligence (AI) assistants. Moving beyond one-shot video question answering, VideoGAIA formulates video understanding as a multi-turn, tool-augmented interaction process, where models must iteratively perceive videos, invoke external tools, gather complementary information, and integrate multimodal evidence across turns. VideoGAIA contains 271 model-human co-designed tasks covering diverse and complex real-world scenarios. Each video-question-answer instance is independently verified by three human experts to ensure both correctness and appropriate difficulty. All evaluated MLLMs, including frontier models such as GPT-5.5 and Kimi-K3, achieve less than 60% accuracy on VideoGAIA, highlighting its value as a high-quality and timely benchmark for evaluating next-generation MLLMs. We hope that VideoGAIA will facilitate the transition from conventional video understanding toward agentic video understanding. 

---
# Path2ST: Hierarchical Cell-Tissue Grounded Cross-Modal Translation for Spatial Transcriptomics 

**Authors**: Ruochen Liu, Wei Lou  

**Link**: [PDF](https://arxiv.org/pdf/2608.14710)  

**Abstract**: Predicting spatial gene expression from hematoxylin and eosin (H\&E)-stained images offers a cost-effective alternative to spatial transcriptomics (ST). However, existing methods treat H\&E images as generic visual inputs and ignore their intrinsic biological hierarchy, where spatially organized cell types collectively form functional tissue microenvironments that govern local gene expression programs. To bridge this gap, we formulate H\&E-to-ST prediction as a cross-modal semantic translation task and propose Path2ST, a hierarchically grounded autoregressive framework featuring three key components: (i) a Hierarchical Cell-Tissue Conditioning mechanism that fuses explicit and implicit cellular features with tissue-level semantic representations to construct hierarchical conditioning signals; (ii) a Scale-Adaptive Autoregressive Generation process over a hierarchical semantic vocabulary, enabling coarse-to-fine, biologically consistent expression synthesis; and (iii) SpectraLoss, a full-spectrum objective that jointly enforces ordinal fidelity, models transcriptional bursts, and aligns semantic structures with cell types. Extensive experiments on three datasets demonstrate state-of-the-art performance, validating that Path2ST generates highly accurate and spatially coherent transcriptomic profiles. The related code is released at this https URL. 

---
# pico-type: A 1.5M-Parameter Byte-Level Multi-Head Content Classifier 

**Authors**: Gautam Kishore  

**Link**: [PDF](https://arxiv.org/pdf/2608.14658)  

**Abstract**: We introduce pico-type, a byte-level multi-head content classifier with approximately 1.5 million parameters that simultaneously predicts seven content properties from raw UTF-8 bytes in a single forward pass. Operating directly at the byte level -- no tokenizer, no subword vocabulary, no pretrained embeddings -- pico-type classifies coarse type (12 classes), modality (8), subtype (24), code language (62), text language (30), file MIME type (90), and risk flags (6-label multi-label: API keys, JWTs, passwords, emails, phone numbers, SSH keys). The architecture combines a learned byte embedding, three convolutional blocks with growing receptive fields, two bidirectional attention layers with rotary position encodings, and a statistical pooling layer feeding seven Matryoshka-style classification heads. Four tiered variants (tiny/small/base/pro) share the same trunk with sliced representations from 16 to 576 dimensions, yielding ONNX exports under 210 KB and CPU inference under 10 ms. Trained on a mixture of synthetic templates and real-world data (8709 GitHub code samples, 5000 Wikipedia articles), pico-type achieves 60.3 percent code language accuracy on The Heap benchmark (24 languages) and 98.2 percent text language accuracy on Wikipedia (30 languages) -- improvements of +57 and +79 percentage points respectively over the synthetic-only baseline. Format-based heads (coarse, modality, subtype, file_mime, risk) maintain 100 percent accuracy on synthetic benchmarks. The model, code, and pretrained weights are released under Apache 2.0. 

---
# DUET: Dual-Teacher On-Policy Distillation via Same-Weight Disagreement for Prohibition Compliance 

**Authors**: Zihan Li, Feifei Li, Wenhui Que  

**Link**: [PDF](https://arxiv.org/pdf/2608.14644)  

**Abstract**: Real-world LLM deployments increasingly rely on runtime-injected prohibitions--enterprise policies, PII redlines, tool boundaries--that vary per request and per tenant. Conventional post-training is structurally ill-suited: SFT hides the violation signal in compliant labels, and DPO's sequence-level preferences mismatch token-localized violations. We propose DUET, a token-selective on-policy distillation method for prohibition compliance. DUET pairs a teacher that sees the prohibition (positive) with an identical-weight teacher that does not (negative). Because the two teachers differ only in prohibition visibility, their per-token disagreement isolates the prohibition's causal effect--yielding a clean supervision signal uncontaminated by model capacity or mismatch. This disagreement drives two complementary mechanisms: signal cleaning, which discards agreement tokens as redundant or prefix-corrupted, and preference-directed learning, which pushes the student away from the negative teacher and toward the positive one at token granularity, embedding DPO-style optimization directly into OPD without offline preference data. We construct an industrial Prohibition-Compliance benchmark spanning five task families covering explicit-refusal, paraphrase robustness, and over-refusal. Across 1.5B-8B Qwen variants, DUET achieves 72.3-85.2% violation compliance while preserving 88-93% normal utility, dramatically outperforming teacher model and other distillation baselines. External evaluation on SysBench confirms improved safety alignment with minimal degradation on GSM8K and MATH-500. 

---
# Valid Per-Field Selective Risk Control for Document Extraction: Three Failure Modes, a Validity Ladder, and When Conditioning Pays 

**Authors**: Bhaskar Gurram  

**Link**: [PDF](https://arxiv.org/pdf/2608.14639)  

**Abstract**: Per-field accept/review with selective risk at most alpha -- accept a field only if the error rate among accepted fields is controlled -- is the trust contract document-extraction systems need, and the natural procedure silently violates it on real documents. On 13,859 genuine claude-sonnet-5 fields from 800 CORD receipts (49.0% correct) we diagnose three failure modes: document clustering (design effect 1.84-2.45), score-refit leakage (coverage 0.416 at risk 0.127, violating alpha=0.10 in 95% of splits), and a tie-mass pathology (a degenerate score collapses the threshold grid, 0.030 to 0.001). We organize the fixes as a validity ladder, guarantee form stated per tier. A fit/val split protocol restores expected-selective-risk control for a learned fusion: coverage 0.318 at risk 0.096 at nominal alpha=0.10, no tolerance band (production variant 0.326) -- an on-average point whose realized risk exceeds alpha in 47.5% of resplits, not a certificate. Mondrian Learn-then-Test with exact binomial tails yields per-group PAC certificates: field-iid 0.171 at risk 0.068, cluster-corrected 0.140, doc-iid 0.060 -- the only tier matching documents, honestly near-vacuous today. Support-bin, the pre-specified provenance taxonomy, wins every rigor tier on the sonnet CORD capture (p<1e-4, Bonferroni-corrected) -- a win that does not replicate on the same documents under haiku or qwen -- while on higher-accuracy corpora pooled thresholds win: conditioning helps exactly where pooled cannot certify, subsumed by a learned score elsewhere. A frozen-configuration confirmation on selection-untouched claude-haiku-4-5 held at both risk levels, and a blind three-annotator human-gold audit verifies the practical tier's accepted-set risk at 1.3% against its 10% budget (Fleiss' kappa=0.83; labels err one-sidedly pessimistic). Released Apache-2.0 with seed-pinned, regression-gated procedures. 

---
# Calibrated Trust, Not Sharper Prediction: An Empirical Test of Uncertainty Fusion 

**Authors**: Surya Saka  

**Link**: [PDF](https://arxiv.org/pdf/2608.14617)  

**Abstract**: A recurring proposal in legal AI is to improve case-outcome prediction by fusing uncertainty tools (evidence graphs with belief propagation, sequential Bayesian odds updating, Dempster-Shafer combination, and conformal prediction) into one pipeline. We test this on 1,000 real European Court of Human Rights cases from LexGLUE and FairLex, predicting whether the Court found a Convention violation from the case's fact paragraphs. We compare three families across two frontier LLMs (Claude Opus 4.8 and GPT-5.5) as per-fact evidence estimators: (A) the raw LLM, (B) the LLM routed through the fusion pipeline, and (C) a term-frequency baseline through the same pipeline. Across roughly 4,750 tests we find: (1) on discrimination (AUROC around 0.83) the pipeline yields no improvement over either the raw LLM or the baseline; a frontier LLM used directly is the strongest single discriminator. (2) Naively composing an LLM with Bayesian-odds and Dempster-Shafer fusion more than doubles calibration error (ECE from about 0.16 to 0.46) via a prior-mismatch mechanism that replicates across both models. (3) Dempster-Shafer fusion is actively unsafe on long chains, committing confidently to wrong labels at below-chance accuracy; we recommend removing it. (4) The pipeline's genuine value is operational: routed through a conformal selective-prediction layer, the system decides which cases to automate and which to escalate. After removing Dempster-Shafer, recalibrating, and applying class-conditional risk control on the full 1,000-case set, the tuned engine auto-clears at 96.8 percent accuracy with 0.5 percent errors escaping and 96.3 percent caught for review, versus 85.9 / 3.8 / 72.1 for an untuned baseline. The contribution of such pipelines in law is calibrated trust, not sharper prediction. 

---
# Plausible but Not Valid: A Psychometric Audit of LLMs as Synthetic Survey Respondents 

**Authors**: Mantas Lukauskas, Viktorija Šarkauskaitė  

**Link**: [PDF](https://arxiv.org/pdf/2608.14606)  

**Abstract**: Large language models (LLMs) are increasingly used as synthetic survey respondents, but existing evaluations ask whether answers look plausible at the individual level. We argue the right question is psychometric: do LLMs preserve the joint distribution, latent structure, reliability, mediation pathways, and demographic effects of real human survey data? We introduce a Lithuanian organisational-psychology dataset (n=263 employees; Dunham Attitudes Toward Change, UWES-17, Koopmans IWPQ; 68 items, 12 subscales) and condition a 37-model lineup spanning OpenAI, Anthropic, Google, and twelve open-weight families on real respondent profiles under a five-level persona-disclosure ladder, presentation and reasoning-effort ablations, counterfactual demographic swaps (gender, role, education), a cross-language check, and a verbatim-recall memorization probe. The resulting Psychometric Similarity Score (PSS) is anchored against five non-LLM statistical baselines and a held-out human-vs-human ceiling, with respondent-bootstrap confidence intervals and an item-permutation null for Tucker's phi. LLMs reproduce the qualitative direction of human psychometric relationships, but a Gaussian-copula baseline beats every LLM on the sample-driven PSS components; the LLM "crowd" is more similar to itself (mean inter-LLM PSS 0.73) than to humans; and memorization does not drive the leaderboard (recall-PSS rank correlation 0.00). Counterfactual swaps reveal education-driven effects (mean |d|=0.56) that dwarf gender (0.12) and role (0.18); Tucker's phi on UWES falls inside the permutation null for 8 of 37 models. Downstream, every LLM shows a strong acquiescence shift (+0.84 SD), synthetic-trained regressors lose predictive validity on held-out humans (mean R^2 -0.18 vs 0.28), and models fabricate indirect effects on 3 of 10 placebo mediation paths. LLM samples are not a drop-in replacement for human survey data. 

---
# The Hallucination Snowball: Modeling Error Propagation as State Transitions in Multi-Agent LLM Pipelines 

**Authors**: Prabhjot Singh, Bhushan Pawar  

**Link**: [PDF](https://arxiv.org/pdf/2608.14588)  

**Abstract**: Sequential multi-agent LLM pipelines chain specialized agents without verification at handoffs, creating a structural flaw with measurable and severe consequences. We show that hallucinations injected at Stage 1 do not merely persist; they transform: raw numerical facts become derived computations, then narrative prose, then editorially approved conclusions. At each transformation, detectability degrades near-irreversibly. We formalize this as the hallucination snowball effect, a first-order Markov process over four states (Raw Fact $\to$ Derived $\to$ Narrative $\to$ Invisible) with empirically measured per-boundary escape probabilities of 24.6%, 48.3%, and 89.3%. Across 346 automatically injected hallucinations in a 4-agent financial analysis pipeline on FinanceBench, gpt-4o detection drops from 72.0% at Stage 1 to 50.9% at Stage 4, and 23.7% of hallucinations survive completely undetected in the final output. Even the strongest model tested (Qwen3.5-397B-A17B, 87.0% at Stage 1) faces a structural ceiling; projected Stage 4 detection is only ${\sim}$60--65%. Critically, boundary gates using identical RAG verification tools reduce hallucination survival from 58.4% to 16.2% versus end-of-pipeline checking (Cohen's $h = -0.911$, $p < 0.000001$), while end-checking alone achieves merely 2.3 pp improvement over no verification. When you verify matters more than whether you verify. Our model predicts survival for $n$-agent linear pipelines and prescribes optimal verification resource allocation: invest at $S_1{\to}S_2$ first, where 75.4% of hallucinations are still catchable, not at $S_3{\to}S_4$ where 89.3% have already escaped. 

---
# FollowUpBot: An LLM-Based Conversational Robot for Automatic Postoperative Follow-up 

**Authors**: Chen Chen, Jianing Yin, Jiannong Cao, Zhiyuan Wen, Mingjin Zhang, Weixun Gao, Xiang Wang, Haihua Shu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15502)  

**Abstract**: Postoperative follow-up plays a crucial role in monitoring recovery and identifying complications. However, traditional approaches, typically involving bedside interviews and manual documentation, are time-consuming and labor-intensive. Although existing digital solutions, such as web questionnaires and intelligent automated calls, can alleviate the workload of nurses to a certain extent, they either deliver an inflexible scripted interaction or face private information leakage issues. To address these limitations, this paper introduces FollowUpBot, an LLM-powered edge-deployed robot for postoperative care and monitoring. It allows dynamic planning of optimal routes and uses edge-deployed LLMs to conduct adaptive and face-to-face conversations with patients through multiple interaction modes, ensuring data privacy. Moreover, FollowUpBot is capable of automatically generating structured postoperative follow-up reports for healthcare institutions by analyzing patient interactions during follow-up. Experimental results demonstrate that our robot achieves high coverage and satisfaction in follow-up interactions, as well as high report generation accuracy across diverse field types. The demonstration video is available at this https URL. 

---
