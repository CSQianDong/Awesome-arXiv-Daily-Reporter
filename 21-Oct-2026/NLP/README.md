# NarraSpace: Multimodal Scene-Grounded Evidence for Picture-Description Dementia Classification 

**Authors**: Qingming Li, Ruiyan Sun, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2610.28616)  

**Abstract**: Picture-description dementia classification uses a shared visual scene, but a transcript-level prediction alone does not identify the scene evidence underlying its inputs. NarraSpace links words and speech times to named picture elements and retrieves utterances for predefined scene events. Narrative-spatial descriptors summarize coverage and temporal organization. A frozen language model contrasts each event's text under examples from Alzheimer's disease and healthy-control training participants, alongside transcript surprisal and coverage measurements. Retained source records link event scores to their source utterances, speech intervals, and lexical matches. On the official 48-participant test set of the Alzheimer's Dementia Recognition through Spontaneous Speech benchmark, a fixed support vector classifier achieves $80.88\%\pm2.56\%$ mean accuracy over fifty seeds; averaged probabilities yield $81.25\%$ accuracy and $91.32\%$ area under the receiver operating characteristic curve. NarraSpace makes the construction of scene-linked classification inputs explicit and inspectable by connecting event-indexed language-model measurements to their source utterances, speech timing, and extraction records. Such source-linked measurements could support clinician review of picture descriptions by presenting quantitative summaries alongside the original speech evidence. 

---
# Benchmarking Arabic Dialects for Math Problems 

**Authors**: Noor Abo Mokh, Teresa Lynn, Nizar Habash, Ahmed Elshabrawy, Sara Alhajeri, Alaa Elsetohy, Preslav Nakov, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2610.28607)  

**Abstract**: Mathematical reasoning remains underexplored for low-resource languages, including Dialectal Arabic. To bridge this gap, we introduce the DAR-MGSM (Dialectal Arabic Multilingual Grade School Math) Benchmark, the first manually curated benchmark for evaluating mathematical reasoning in Dialectal Arabic. DAR-MGSM consists of math word problems (MWPs) requiring multi-step reasoning and covers five major Arabic dialects alongside Modern Standard Arabic (MSA). The benchmark comprises two parallel versions: translated-only and translated & culturally localized. The latter version adapts entities and references to regional varieties. We evaluate eleven Arabic-centric and multilingual large language models on DAR-MGSM. The results show that the performance generally declines on the localized version, highlighting the challenges introduced by culturally grounded adaptations. However, the magnitude of this effect varies across models, with Arabic-centric models exhibiting different levels of sensitivity to localization. A mechanistic interpretability analysis shows that translated and localized variants of the same MWPs often induce distinct internal reasoning trajectories, even when aggregate accuracy remains similar. Overall, our results show that mathematical reasoning performance depends on both linguistic variety and cultural context. 

---
# Capable Teacher, Myopic Scorer: Reward-Free Teacher Repair for Agentic On-Policy Distillation 

**Authors**: Shu Zhao, Tan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28586)  

**Abstract**: Agentic on-policy distillation (OPD) trains a student agent on its own trajectories while a teacher scores every token. A natural explanation of where it goes wrong is compounding error: the student's mistakes carry its trajectories beyond the region where the teacher is reliable, so recent methods keep training inside that region, through episode-length curricula, teacher-guided turns, prefix replay, or teacher intervention where the two disagree. We show this explanation is incomplete. From the states where the student departs most from the teacher, the teacher's own takeovers still finish the task, yet the student trained on its token scores keeps running out of turns. We attribute this to the myopia of next-token scoring. The teacher rates each token without seeing where the episode ends, so its scores let through the safe, delaying moves that outcome-based RL would punish with zero reward, a penalty agentic OPD lacks. We call this difference between what the teacher does and what it scores the rollout-scorer gap. We propose Rollout-to-Scorer Distillation (R2S), which repairs the scorer before distillation by mining high-disagreement states from the untrained student, running teacher takeovers from them, keeping the states where enough takeovers pass a reward-free check such as consensus or an LLM judge, and fine-tuning the teacher to give its takeover-informed scores on every assistant token from the divergence turn onward, without the takeover in context. The repaired teacher is frozen as the sole scorer for standard OPD, requiring no task reward during repair or training. Across three environments, two teachers and four students, R2S outperforms three state-of-the-art agentic-OPD methods in all twelve teacher-student-environment cells, by 12.8 success points on average over the best of them in each cell. Project page: this https URL. 

---
# Cache Engineering for Agents 

**Authors**: Tianyu Fu, Ruoxi Wang, Zihan Min, Zhongqi Ren, Zhihao Li, Tengxuan Liu, Shengxu Mu, Hanling Zhang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28577)  

**Abstract**: Large Language Model (LLM) agents can handle increasingly complex tasks, navigating long trajectories of observations and actions. As these trajectories grow, key-value (KV) cache I/O and storage become the dominant cost. Trajectories can also exceed the LLM's maximum context length, requiring control over active cache growth. Prior work uses context engineering to reduce cache size by trimming or summarizing history. However, modifying the trajectory prefix invalidates subsequent cached states and triggers costly re-prefilling of retained context. Task accuracy can also suffer when too little context is retained to limit re-prefill costs. We propose cache engineering as a new scheme. It allows agents to directly control their active KV cache without modifying the trajectory or re-prefilling retained context. Specifically, agents can proactively use drop to exclude selected cache from future attention and repos to adjust retained keys' position encodings. We formalize these cache operations and identify history dependence as a serving challenge: identical token sequences can yield different cache states under different cache operation sequences, making prefix matching insufficient for correct cache reuse. Thus, we co-design the AKV serving engine. It provides a stateless interface compatible with existing serving APIs, while internally managing stateful cache reuse, eviction, and restoration. Experiments show that a tool-response rolling-drop cache policy improves average benchmark scores by 1.7 points and reduces average cost by 27.7%, compared with conventional threshold-triggered summarization. With AKV, it achieves up to 1.2x the output throughput and a 1.5x speedup in P95 request latency. Recursive self-improvement of cache policies further reduces cost by 7.1% over expert design. Our code is at this https URL. 

---
# Large Flow Language Models 

**Authors**: Arnav Chavan, Chanhyuk Lee, Udbhav Bamba, Jaehoon Yoo, Seunghoon Hong, Aditi Raghunathan, Jinwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.28559)  

**Abstract**: Continuous flow language models offer a path to parallel generation with tools for control and acceleration unavailable to discrete diffusion, yet effective techniques for building them at the billion-parameter scale remain less established. We present Lafa (Large flow language model), a fast, high-performing flow language model scaled to 8B parameters in a data-efficient manner. Lafa is a blockwise flow model with causal attention on the prefix and bidirectional, asynchronous flow within each block. This design enables efficient training by adapting pretrained autoregressive (AR) models and fast inference via prefix caching and blockwise parallel decoding. We train Lafa at 0.5B, 7B, and 8B scales with fewer than 30B tokens, more efficient than previous diffusion models, and evaluate it on instruction following, reasoning, and coding, where the learned flow matches or surpasses the quality of AR and can be distilled into flow maps for one- and few-step generation. Notably, flow training preserves or even improves pretrained AR capabilities via a multi-token prediction effect, which we leverage in a self-speculative decoding scheme that verifies flow-based drafts autoregressively within a single model. This yields 2.54x the throughput of an AR model at comparable quality, outperforming multi-token prediction and discrete diffusion methods. Our findings show that continuous flows can serve as a scalable foundation for language modeling. 

---
# BitMTP: When Multi-Token Prediction Meets Low-Bit Large Language Models 

**Authors**: Ning Zhang, Shihao Wang, Jinrui Zhang, Chaodong Xiao, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28500)  

**Abstract**: For efficient deployment of large language models (LLMs), methods of ultra-low-bit backbones reduce weight storage, memory traffic, and linear-layer execution cost, while multi-token prediction (MTP) methods reduce sequential decoding steps by predicting multiple future tokens in parallel. Although complementary, the direct combination of these methods cannot accelerate decoding at comparable quality: masked drafting lacks intermediate token input supplied in next-token training, while wider computation and causal key-value (KV) updates can offset the savings from fewer steps. To address these challenges, we propose BitMTP, an algorithm-system co-design framework for efficient MTP on ultra-low-bit LLM backbones. A joint next-token prediction and MTP objective equips a single model with both autoregressive (AR) generation and block-causal drafting. Confidence-only decoding, speculative-only decoding, and hybrid decoding provide different quality-throughput operating points without using separate draft and verifier models. Accepted-prefix alignment keeps the committed prefix, materialized causal KV boundary, and CUDA Graph replay state consistent for continued generation. BitMTP uses this contract to select width-specialized low-bit execution and reuse captured graphs over shared KV storage. On NVIDIA H20, BitMTP with hybrid decoding achieves five-task mean speedups of 2.2x on BitNet b1.58-2B and 2.1-2.4x on 1.7B-4B Bonsai models at comparable task quality. On RTX 3090 and Jetson Orin, it further achieves up to 2.0x in five-task mean throughput and 3.3x on MBPP+, respectively. These results demonstrate that BitMTP accelerates ultra-low-bit LLM generation across backbones, model sizes, and GPU types. The source code is released at this https URL 

---
# OptionWorld: A Benchmark for Diagnosing Failures of LLMs as Long-Term Personal Assistants 

**Authors**: Ruoyu Zhang, Zhaocen Liu, Weiyi Tian, Melissa Feng, Shilong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28491)  

**Abstract**: Long-term assistants must follow unfinished tasks across days as information and circumstances change. Finding information or starting an action, however, does not ensure that the task gets completed. We introduce OptionWorld, a benchmark that traces information acquisition, judgments, actions, and outcomes in simulated daily-life tasks to identify where current LLMs fall short as long-term personal assistants. In runs with archived decision inputs, the key facts remain in the model's last input before the deadline in 35.5% of failed instances. Failure analysis identifies the three most frequent problems: needed information is not obtained or combined, available opportunities to act go unused, and agents impose restrictions that the task does not require. Three follow-up experiments show that supplying missing facts restores task success in some cases; action reminders can restart work while subsequent steps still miss the deadline; and correcting a supplied note that wrongly forbids contact restores permitted inquiry. These findings locate shortcomings in maintaining task state: connecting facts to unfinished work, following through before deadlines, and separating actual restrictions from the agent's assumptions. 

---
# Asymmetric Looped Transformers: Fixed Global Representations for an Iterative Decoder 

**Authors**: Cheng Liang, Tengfei Zhang, Wenjie DU, Jianyu Wu, Changlian Ma, Shuchen Pu, Ya Zhang, Yanfeng Wang, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.28489)  

**Abstract**: Looped Transformers increase effective depth by repeatedly applying shared layers, offering a way to scale computation without increasing parameter count. However, full-stack looping places context construction and iterative state refinement within the same recurrent computation, without explicitly distinguishing their roles. We introduce the Asymmetric Looped Transformer (AsymLoop), which separates single-pass context encoding from recurrent state refinement. A single-pass causal encoder produces layer-specific Global key-value (KV) caches that remain fixed across iterations, while a weight-tied recurrent decoder generates Local KV from its evolving state. Joint attention over both sources under a single softmax allows each update to integrate persistent contextual information with the current intermediate state. For training, we apply a heavy-tail-guided layerwise learning-rate method to the encoder and decoder. Experiments on FineWeb-Edu across four model scales, from 60M to 1B, show consistently lower validation perplexity and higher average downstream accuracy than Full Loop under matched effective depth and training token budgets. At 1B, AsymLoop improves average zero-shot accuracy across seven commonsense reasoning benchmarks from 49.58% to 50.85%. Further experiments extend these benefits to longer training budgets, an OLMo-style backbone, and sparse mixture-of-experts models. These results support asymmetric context encoding and state refinement as an effective design for recurrent language models. 

---
# MythoSemantic: An Interactive System for Discovering Semantic Parallels Across Mythological and Religious Traditions 

**Authors**: Nikita Ganziuk, Innokentiy S. Humonen, Ilya Makarov, Pavel Lebedev  

**Link**: [PDF](https://arxiv.org/pdf/2610.28452)  

**Abstract**: We present a tool for the comparative study of mythological and religious texts. Traditional manual analysis is highly labor-intensive and difficult to scale. To address this, we use neural language models to embed texts into a shared semantic space, so that users can find semantic parallels across cultural traditions in an interactive interface. Embedded fragments form distinct semantic structures whose coarse geometry aligns with well-documented historical relations between traditions. We quantify how much of that geometry is already explained by lexical overlap and by proper names. The current system is available at this https URL, and the installable package is available at this https URL. 

---
# Every Model Leaves a Trace: Multi-Resolution LLM Fingerprinting 

**Authors**: Mohammad Reza Modarres, Armin Tourajmehr, Artin Tavasoli, Yadollah Yaghoobzadeh, Mohammad Taher Pilehvar  

**Link**: [PDF](https://arxiv.org/pdf/2610.28434)  

**Abstract**: Large language models leave model-specific traces in their generated text, but it remains unclear how finely these fingerprints can distinguish related models and which signals remain informative as obvious cues are reduced. We study this through fingerprint resolution, considering both model resolution, from different families to distilled and adapted variants, and signal resolution across behavioral, stylometric, lexical, and semantic evidence. We progressively suppress surface cues, paraphrase outputs, vary the amount of available text, and test generalization across prompt distributions and languages. Our results show that fingerprints persist beyond broad family differences, remaining detectable in distilled models and even LoRA adaptations of the same base model. Although attribution weakens as surface and lexical cues are altered, substantial model-specific signal remains. These signals transfer across prompt sources and languages, with semantic representations proving most robust across languages. These findings show that model identity is recoverable at multiple resolutions, but that recoverability depends on model similarity and the evidence available in the text. 

---
# The Instruction Tax on Factual Recall: Base Models Can Beat Their Instruction-Tuned Counterparts 

**Authors**: Christophe Khalil, Aya Mourad  

**Link**: [PDF](https://arxiv.org/pdf/2610.28407)  

**Abstract**: We present our solution to the AKBC LM-KBC 2026 Shared Task, where we achieved a score of 0.6836 on the official leaderboard, more than double the baseline of 0.2964. This task requires generating all possible objects for subject-relation pairs using a language model with up to 32 billion parameters, with no retrieval or fine-tuning involved. We show that the prompting methodology is as important as the model architecture: while the instruction-tuned model version achieves 0.5681 with its chat interface, the base model achieves 0.6836 when prompted using text completion. Thus, instruction tuning decreases factual recall; we refer to this effect as the instruction tax, which is caused by the model's weights rather than the prompting method. Our final solution is fairly straightforward: we prompt the base model with some examples from the training data, generate 64--256 outputs, and retain only the repeated ones. Two specialized methods are then applied for each relation: one improves the recall of large answer sets, and the other verifies a person's vitality before predicting their place of death. 

---
# When Copying Is Hard: Copy-Constrained Decoding for Exact Span Reproduction 

**Authors**: Jinghui Zhang, Lang Gao, Zongfang Liu, Ruihong Zeng, Zirui Song, Rui Yan, Kentaro Inui, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.28393)  

**Abstract**: Precisely reproducing a target span verbatim within a long context is a fundamental capability test for LLMs: even when the target string is present in context, models often still drift within the span or fail to stop correctly, revealing limitations in source grounding and boundary control. This capability has broad practical relevance. For instance, code generation often requires embedding sensitive strings such as API keys exactly as they appear in context. In this work, we construct a benchmark to test exact span reproduction ability across contracts, scientific literature, code, and random sequences, and find that current models still make surprisingly frequent errors. To address this, we propose CopyGen, a lightweight copy-constrained plugin for frozen LLMs that turns copying from a byproduct of next-token prediction into a controlled process. Our key idea is to decompose copying into three distinct decisions: when to copy, how far to copy, and when to stop. Concretely, CopyGen first determines whether to enter copy mode, then predicts a tentative copy length, and finally verifies the generated span token by token. During verification, we adaptively adjust the logits of copy continuation tokens to incentivize or suppress continuation, and emit only the longest valid prefix. Across three LLM backbones and four benchmark tasks, CopyGen improves average exact match by 35% relative without model finetuning, and integrates seamlessly with SFT to further improve it by 4%, while substantially accelerating generation. These results suggest that reliable exact copying is not something LLMs obtain for free, but a decoding-time control problem that requires explicit modeling. Code is available at this https URL. 

---
# Object-Oriented Harness: Evolving Agent Harnesses through Object-Oriented Design 

**Authors**: Gaotang Li, Bhavana Dalvi Mishra, Yifei Ming, Yanfei Chen, Zhongying CuiZhu, Lesly Miculicich, Long T. Le, Rui Meng, Chun-Liang Li, Hanghang Tong, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2610.28380)  

**Abstract**: Agent harnesses are executable code that specifies the prompts, tools, and control flow around a language model. A single global harness can struggle with heterogeneous tasks that require different execution strategies. In this work, we introduce Object-Oriented Harness (OO-Harness), a harness optimization framework that evolves a hierarchy of executable harness classes using object-oriented design. During training, an evolution agent uses execution records to create a family of specialized harnesses, each implemented as a subclass of the base or an existing specialized harness via inheritance. At test time, a harness router uses routing lessons accumulated during training to select a specific harness for each task. Across three long-horizon benchmarks spanning business workflows, workspace tasks, and tool use, OO-Harness achieves the best performance among the compared methods, with an absolute improvement of up to 15% over the base. Extensive ablation studies examine the contributions of its core components. 

---
# DeclineCraft: A RAG Abstention Benchmark for Jointly Measuring Combined Retrieval and Parametric Knowledge 

**Authors**: Atharva Thakur, Abulhair Saparov, Bruno Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2610.28340)  

**Abstract**: Reliable retrieval-augmented generation (RAG) requires recognizing when neither retrieved documents nor parametric knowledge supports an answer. We introduce DeclineCraft, a dynamic abstention benchmark whose unanswerable questions target Minecraft feature additions after a corpus cut-off, with audits of possible parametric exposure. Later cut-offs support evaluation of future models. Across 27 LLMs, we measure Correct Abstention Rate (CAR; how often models acknowledge insufficient knowledge on unanswerable questions), Answerable Attempt Rate and Answer Accuracy. We find that (1) single-shot retrieval significantly raises CAR in 38 of 40 configurations (model and thinking setting) versus no retrieval. (2) Iterative RAG, in which the model self-assesses its context and can request up to two further retrieval rounds, significantly lowers CAR versus single-shot RAG in 30 of 40 configurations (mean 46.4%), with little change in Answer Accuracy (68.5% to 69.3%). Adding query rewriting and reranking to hybrid (dense & sparse) retrieval lowers CAR for all 12 models tested. (3) Larger models answer more accurately (Spearman $\rho = +0.60$ with total parameters), but neither model size nor release date shows a significant association with CAR. (4) Without retrieval, enabling thinking on the same weights lowers CAR significantly for 8 of 13 models (by 14% points on average), and the responses shift from acknowledged uncertainty to premise denial. 

---
# Beyond a Single Sentiment Score: Topical and Affective Dimensions of Emoji Meaning on Twitter 

**Authors**: Rohan James, Ioannis Christofilogiannis, Yuuv Jauhari, Shreya Shah, Gauri Pethkar  

**Link**: [PDF](https://arxiv.org/pdf/2610.28337)  

**Abstract**: Emoji are widely treated as compact affective tokens whose meaning is summarised by a single sentiment score. We argue that this picture is incomplete. From a corpus of 16{,}575 English tweets and a per-emoji metric we call \emph{stability delta} (within-class minus between-class cosine similarity of sentence embeddings), we show that emojis used in tightly clustered topical contexts also exhibit \emph{more} polarised crowd-sourced sentiment, not less. Validated against the Kralj Novak et al.\ (2015) sentiment lexicon, the central correlation is Spearman $\rho = +0.361$ (Holm $p = 0.015$, 95\% bootstrap CI $[+0.10, +0.59]$); against the independent Ferré et al.\ (2025) Spanish 13-emotion ratings, $\rho = +0.509$ ($p = 0.0003$). The result survives permutation testing, BCa bootstrap, leave-one-out jackknife, partial correlation controlling for emoji frequency, and Holm correction across a four-test family. Topical coherence and affective dispersion are distinct dimensions of emoji meaning. As a downstream check we report leakage-free five-fold cross-validation results for sarcasm classification on the Emoji Sarcasm Dataset; all three baselines (a gated multimodal architecture, fine-tuned BERTweet features, and fine-tuned RoBERTa features) score above 0.97 macro F1, indicating the dataset has saturating emoji-label cues. We frame the classifier as supporting evidence rather than the contribution. 

---
# From Abstraction to Action: DSLs for Natural Language Abstraction Interpretation and Execution 

**Authors**: Amit Lisha, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2610.28311)  

**Abstract**: Natural-language instructions often rely on high-level abstractions, such as asking an agent to ``stack blocks in a pyramid.'' Interpreting abstract instructions requires models to both identify the abstraction intended by the speaker, and unroll it into the many concrete actions that instantiate it. Language models struggle here: prior work has shown that performance degrades as the level of abstraction in the instruction rises. To bridge the gap between natural language abstraction and concrete actions, we propose to borrow from coding environments the notion of a {\em domain-specific language} (DSL), an explicit representation layer whose documented primitives name recurring abstractions that expand into concrete actions. We demonstrate the efficacy of DSLs for NL interpretation on \textsc{Hexagons}, a visually-grounded instruction-following task graded for abstraction. We further ask whether language models can induce such DSLs themselves from domain data. Our results show that while code generation consistently improves over direct inference, code generation over a human-curated DSL yields dramatic gains, up to 10 absolute F1 points, reaching the human skyline, with gains concentrated at {\em higher} abstraction levels. We confirm this pattern on the \textsc{Text2CAD} benchmark. At the same time, model-induced DSLs recover only part of this gain, remaining inconsistent across runs and short of their human-curated counterpart. Overall, the right representation closes much of the gap caused by high abstraction --- but inducing that layer of representation automatically is where models still fall short, and where we see a strong opportunity for the community to make progress. 

---
# Reward Misspecification Degrades Chain-of-Thought Monitorability 

**Authors**: Tianyi Lorena Yan, John Hewitt  

**Link**: [PDF](https://arxiv.org/pdf/2610.28303)  

**Abstract**: Monitoring the chain-of-thought (CoT) of language models (LMs) is a key strategy for detecting and preventing misbehaviors for AI safety. As such, it is critical to identify reinforcement learning (RL) practices that lead to unmonitorable chain-of-thought. One such practice may be unavoidable: reward misspecification. As RL environments become more complex, it is infeasible to have perfect verification, and models are rewarded for behaviors that are different from the true task intent. We show that reward misspecification alone already degrades monitorability. We create a testbed with pairs of true reward aligned with task intent and misspecified rewards that favor a different behavior, so that reward hacking can be induced by training and labeled verifiably. Across three models and three tasks, CoT monitorability remains low as models' hack rates increase: the CoTs keep reasoning about the task intent while the outputs violate the intent and get rewarded. To mitigate unmonitorability without prior knowledge of the true reward or the hack, we collect rollouts with diverse rewards across training steps and use a LM to infer a reward description. We then inject the description into the input, which teaches the model to verbalize the reward. This reward description training improves monitorability by an average of 24% or 42% absolute, depending on the metric, across all models and tasks, and yields CoTs that reason about the misspecified reward explicitly. Code is available at this https URL. 

---
# Adaptive Replay Refinement: Allocating Test-Time Compute Between Revision and Resampling 

**Authors**: Ahsan Bilal, Muhammad Ahmed Mohsin, Muhammad Umer, Ayesha Mohsin, Sarmad Bilal, Dean F. Hougen  

**Link**: [PDF](https://arxiv.org/pdf/2610.28294)  

**Abstract**: Test-time scaling allocates additional inference compute to refining existing solutions or generating new ones. This raises a fundamental question: Should I keep improving this solution, or stop working on it and try a completely new solution? Existing methods make this decision using fixed refinement schedules, self-verification judgments, or learned controllers, which can waste compute or rely on signals that may be incorrect or require additional training. We introduce Adaptive Replay Refinement (ARR), a training-free method that detects when a stationary revision reaches a fixed point and reallocates the released compute to a new independent sample; replay decoding treats the current solution as a draft and verifies unchanged tokens in parallel, enabling efficient fixed-point detection. Across seven models from five families and seven benchmarks, ARR matches majority voting on three of five mathematical reasoning benchmarks at equal compute, while reducing critical-path serial passes by up to 2.4 times. On MATH500, ARR achieves 84.6% accuracy with only 41% of the serial passes required by majority voting. These results demonstrate that detecting refinement exhaustion enables adaptive reallocation from revision to independent sampling, improving inference efficiency while highlighting the limits of self-revision. Code is available at this https URL. 

---
# Forty Questions: Identifying a Language Model from One-Word Answers 

**Authors**: Tapan Parikh  

**Link**: [PDF](https://arxiv.org/pdf/2610.28281)  

**Abstract**: A model behind an API can be identified from its one-word answers to everyday questions such as "Name a sound" or "Name a poet". A model's profile records how often it gives each answer to each question. We attribute new answers to the model most likely to have given them. On a panel of 94 models, 40 answers identify the exact model 92.5% of the time, against 1.1% by chance. Profiles built through OpenRouter identified all ten models served by DeepInfra under blinded labels. A system prompt lowers accuracy by 6 to 16 points. We applied the method to 96 endpoints of 13 open models on OpenRouter, and 92 identify as the model they claim. Of the other four, one matches the claimed model with reasoning turned off. One serves Gemma 4 under the name Llama 4 Maverick: it names itself Gemma 4, and it is billed Gemma 4's token counts on every message. Two remain unresolved. Billing records also show context added to first-party models. Grok 4.7 is billed 1,247 input tokens for a one-line question that costs GPT-4o only 12 tokens, and five models state the current date without being billed for it. The questions come from a public census of model defaults, and the method costs cents per endpoint. 

---
# How Language Models Learn Taste: Brand Preference in LLMs 

**Authors**: Tapan Parikh  

**Link**: [PDF](https://arxiv.org/pdf/2610.28272)  

**Abstract**: Asked to name a running shoe brand, language models say Nike. Asked to recommend one, they say Brooks. This paper asks where models learn that taste. Across 102 models and 42 brand categories, the brand a model picks when recommending differs from the brand it names in 26. The field moves together to a shared recommendation. The Named brands are market leaders, and the Picked brands are mostly niche and specialist. Three open training pipelines show where this forms. Pretraining already attaches the recommended brand to advice. Supervised fine-tuning teaches the model to answer and mostly leaves the pick unchanged. Preference optimization (DPO) teaches the model which answer is better, and the pick moves in all three pipelines, by 11 to 24 points. The preference data contains almost no brands. DPO moves the model's judgments toward a second set of associations even when the preferred brand is not in the preference data, and the effect reaches beyond brands. Asked to choose a sauce, models switch from ketchup to pesto. We do not know exactly how these preferences arise. We suspect they come from the judge models that decide which answers count as better during preference training. For brands, a recommendation depends less on what a model knows about them than on whether they fit this taste. 

---
# Can Language Models Get "Aphasia" From Neuronal Ablation? 

**Authors**: Pratik Worah, Randall Bateman  

**Link**: [PDF](https://arxiv.org/pdf/2610.28214)  

**Abstract**: Progressive neuronal loss is the hallmark of neurodegenerative disease, yet how loss of neurons shapes the pattern of cognitive decline remains poorly understood. Here we systematically ablate neurons (computational units) in language models spanning recurrent long short-term memory (LSTM) and Transformer architectures and apply limited fine-tuning as a simple proxy for compensatory plasticity. With whole-unit ablation, LSTMs degrade gradually with syntactic and semantic faculties declining in parallel across mild-to-moderate damage. By contrast, Transformers exhibit late threshold-like collapse under feedforward multi-layer perceptron (MLP) ablation and smoother sigmoid-like decline under attention-head ablation. Qualitatively, MLP-ablated models continue to make within-category semantic substitutions (semantic paraphasias) alongside rising non-responses, paralleling Wernicke-type aphasia, while attention-head ablation produces a trend toward loss of semantic specificity similar to the semantic variant of primary progressive aphasia (svPPA). These findings indicate that feedforward and attentional pathways have functionally distinct failure modes, suggesting that architecture-dependent failure profiles may offer new computational perspectives on aphasic syndromes and neurodegenerative diseases as an in silico model. Notably, early accuracy retention under Transformer attention-head ablation qualitatively parallels pre-symptomatic reserve before the 40--60% neuronal loss reported by Gomez-Isla et al. (1996) when clinical symptoms emerge in Alzheimer's disease, whereas the steadier LSTM decline parallels the symptomatic progression described by DeKosky and Scheff (1990) and Terry et al. (1991). 

---
# Temporal Evidence-Parameter Co-Alignment for LLM Reasoning over Temporal Knowledge Graphs 

**Authors**: Jiapu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28176)  

**Abstract**: Large Language Models (LLMs) often struggle with temporally grounded reasoning because their static parametric memory is misaligned with the dynamic evolution of real-world knowledge. Although RAG and Parameter-Efficient Fine-Tuning (PEFT) provide practical ways to enhance LLMs, they remain insufficient for temporal reasoning. Existing retrieval methods typically prioritize semantic relevance while overlooking temporal validity, event order, and evolutionary dependencies, resulting in evidence that is relevant in isolation but temporally fragmented. Meanwhile, conventional PEFT methods usually employ fixed and input-agnostic low-rank updates, making it difficult to capture time-dependent entity semantics and relation drift. To address these limitations, we propose Temporal Evidence-Parameter Co-Alignment (TEPCA), a framework for LLM reasoning over TKGs. TEPCA aligns LLM reasoning with temporal knowledge at both the evidence and parameter levels. At the evidence level, it introduces time-constrained Dynamic Relevance-Guided Path (DRGP) pruning to preserve globally coherent temporal paths, together with Mutual-Information (MI)-based pruning to retain query-informative local facts under temporal constraints. At the parameter level, it integrates Dynamic-Gated LoRA (DG-LoRA), a timestamp-conditioned low-rank adaptation mechanism that dynamically modulates the effective adaptation subspace according to temporal context. This design enables LLMs to better capture fine-grained entity evolution and time-dependent relation semantics while preserving parameter efficiency. Experimental results on two TKG benchmarks demonstrate that TEPCA achieves state-of-the-art performance. Additional evaluations of DG-LoRA on eight commonsense QA datasets further verify its general input-adaptive capacity, scalability across LLM backbones, and interpretability. 

---
# Africa-Papers: A 10-Billion-Token Multi-Domain Academic Corpus for African Foundation Language Models 

**Authors**: Mich-Seth Owusu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28173)  

**Abstract**: Large Language Models (LLMs) depend fundamentally on the diversity, depth, and factual rigor of their pre-training corpora. However, contemporary foundational pre-training datasets (such as The Pile, S2ORC, and Dolma) exhibit a severe structural omission: African scientific, medical, legal, and agricultural scholarship constitutes less than 0.5% of global pre-training tokens. Consequently, state-of-the-art models suffer from systematic failure modes when reasoning about tropical epidemiology, localized smallholder agro-ecologies, regional economic governance, and indigenous linguistic structures. To resolve this representation deficit and advance African intellectual sovereignty in artificial intelligence, we introduce Africa-Papers, the largest curated multi-domain Pan-African academic pre-training corpus to date. Comprising over 844,807 full-text papers spanning 10.15 billion tokens across 879 Parquet shards, Africa-Papers unifies four primary scientific streams: (1) continental peer-reviewed journals indexed in DOAJ across 31 African nations, (2) the CGIAR CGSpace agricultural repository, (3) African-authored and African-focused STEM/AI preprints on arXiv, and (4) open-access international publications by scholars affiliated with African institutions indexed via OpenAlex. The entire corpus is processed into layout-aware, clean Markdown preserving section hierarchies, mathematical equations, and tabular data, accompanied by exact BPE token counts and a standardized 8-domain taxonomy. We release Africa-Papers openly on the Hugging Face Hub under the AfriSpeech organization to accelerate continual pre-training and domain evaluation for foundation models globally. 

---
# Reading Too Much Into It? Interpretive Preferences and Commitment in LLM Mentalisation under Ambiguity 

**Authors**: Stephanie Fong, Yiwen Jiang, Qingyang Xu, Zimu Wang, Yaling Shen, Xiangyu Zhao, Jiahe Liu, Yinghao Zhang, Vincent Lee, Dominic Dwyer  

**Link**: [PDF](https://arxiv.org/pdf/2610.28105)  

**Abstract**: Large Language Models (LLMs) are increasingly used for interpersonal advice, potentially shaping how users interpret ambiguous social encounters. Existing Theory of Mind evaluations typically rely on predefined answers, while uncertainty research overlooks LLMs' preferred explanations and confidence under psychological distress. To address these gaps, we introduce MentalLoom, a novel evaluation of LLM mentalisation comprising 604 ambiguous social episodes, matched depression and social anxiety disclosures, and four-turn dialogues with unchanged event evidence. Across eight LLMs, disclosures initially elicited more negative explanations overall but more positive narrator-directed interpretations. Scripted repeated questioning showed greater emotional support, stronger positive narrator-directed endorsement, and higher certainty without new event evidence. These findings show that user context can reshape interpretive preferences and commitment while the underlying ambiguity remains unresolved. 

---
# BAIT: Can LLMs Find What Biases a Reader? A Benchmark for Cognitive Bias Triggers in News 

**Authors**: Yousra Fettach, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2610.28092)  

**Abstract**: Persuasive writing exploits cognitive biases, recruiting a reader's mental shortcuts to shape belief, often without the reader noticing. We call the linguistic features exploiting such mental shortcuts cognitive bias triggers. As large language models (LLMs) increasingly mediate how people encounter news, summarizing it, ranking it, and answering questions over it, a natural question is whether they can recognize the features of a text that bias a human reader. We introduce BAIT (Bias Annotation In Text), a resource for detecting cognitive bias triggers in news. BAIT pairs a schema of nine triggers, each specified in an annotator codebook, with 50 English news articles annotated by trained annotators at the article and span levels and adjudicated into a gold standard. On this data we define two tasks, detecting which triggers a text contains and locating the spans that carry them. We then benchmark six frontier LLMs zero-shot and with three or ten in-context examples. The models approach human agreement on which triggers an article contains, but rarely recover its full set of triggers, and they reach less than half of human agreement in locating the passages that carry them. Moreover, they over-predict triggers that come with easy surface cues, and most of the passages they mark for these triggers are not triggers, yet carry the trigger's surface cue far more often than random text. These findings suggest that the evaluated LLMs respond to surface cues rather than to the mechanisms that make a passage manipulative. We release the schema, the codebook, standoff annotations, and the evaluation pipeline. 

---
# Same Judgment, Different Yes: Measuring Multilingual Judgment Profiles in Language Models 

**Authors**: Haonan Huang, Yuwei Lu, Chengkun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.28066)  

**Abstract**: Multilingual evaluations often score a model's opening yes or no, a token that also carries model and language answering conventions. We ask when different ways of eliciting and reading an answer recover the same judgment, and when they measure an answer habit. We collect 1.1 million replies from eight models in six languages on up to 105 moral vignettes, varying the question's form, the answer constraint, and the reading of the reply, token or sentence, in both wordings (the FAR-6 battery). Story-level judgment profiles converge across the tested readouts and question forms, except for the yes/no token after a negative interrogative read under a preset mapping: on the same replies, token and sentence profiles agree at a median r = 0.67 across cells, while sentence profiles agree with free-sentence profiles at 0.94, a known linguistic variability. With each language's story consensus removed, a model's profile is nearest to its own in another language for 39 of 48 model-language cells under proposition labels and 15 under free sentences, against 5 expected by chance. Particle, position, and named-term preferences follow their own patterns. A model can keep its judgment and change its yes: scoring that token requires establishing which proposition it affirms. 

---
# On-Policy Transition Distillation 

**Authors**: Runyang You, Yongqi Li, Tiezheng Yu, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.28048)  

**Abstract**: On-policy distillation (OPD) consolidates expert capabilities into a single student by matching teacher and student final-layer distributions on student-generated rollouts. We investigate whether depthwise prediction transitions, the changes in predictions between layers, can provide additional task-relevant supervision. Our diagnostic study finds that stronger teacher--student agreement in intermediate transitions is associated with higher answer correctness within questions, with correct responses exhibiting lower mean transition KL across all measured intermediate layer pairs. These observations motivate On-Policy Transition Distillation (OPTD), which augments final-layer distribution matching by aligning teacher and student layer-to-layer log-probability increments on student-generated rollouts. Evaluations across three backbones and eight benchmarks spanning mathematics, coding, and instruction following show that OPTD achieves the best or tied-best overall performance in both dual- and three-domain distillation. In the three-domain setting, OPTD recovers $\approx 90\%$ of the overall teacher--student capability gap, compared with $\approx 75\%$ for OPD. 

---
# Using Letter Positional Probabilities to Assess French Word Complexity 

**Authors**: Michael Dalvean  

**Link**: [PDF](https://arxiv.org/pdf/2610.28040)  

**Abstract**: We use Letter Positional Probabilities (LPPs) to examine the orthographic differences between simple and complex French words. We draw from a sample of 38,000 French words scored by reaction time on a lexical decision task. We take the 5000 highest reaction time words and the 5000 lowest as a proxy for high and low word complexity groups respectively. We create 18 variables based on the structure of the words, such as length and number of syllables. We also reduce each word to a string of binary values by one-hot encoding each letter according to the 42 characters of the standard French alphabet up to a length of 25 letters. Thus, each word is represented by 42 x 25 binary characters. We then create LPPs by looking at the average binary scores for simple as opposed to complex words for all letter/positions. We find that there is a high number of LPPs that are associated with word complexity. For example, we find that most accented letters are associated with simpler words. We examine the effects of structural and LPP variables in classifiers and find that word length alone enables a classification accuracy of 77.27%, while adding the binary variables from the first 6 letter positions provided an accuracy of 83.95%. We discuss possible explanations as to how complexity could be encoded in orthography. 

---
# Adaptive Stopping of Streamed Reasoning for Language Model Agents 

**Authors**: JunHyeok Oh, Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.27981)  

**Abstract**: Reasoning-enabled language model agents can incur substantial computation before acting, although the reasoning needed varies across decisions. Existing stopping signals, such as confidence and intermediate-answer stability, can vary in effectiveness across reasoning models; signals obtained from hidden states or token probabilities also require internal access that may be unavailable or impractical. We introduce StreamStop, which learns model-specific stopping policies offline from prefix-level action agreement. A frozen prefix reader encodes available reasoning text for a lightweight stop head, allowing stopping without access to the reasoning model's internal signals. We construct and publicly release a dataset of agent reasoning trajectories with prefix-level action annotations. These annotations compare each prefix action with the full-reasoning reference action, enabling offline comparison of stopping rules and exact expected-return optimization of a discounted action-agreement objective. Conditioning the head on the discount factor allows the agreement-token trade-off to be adjusted at inference without retraining. In closed-loop evaluations on WebShop and ScienceWorld, StreamStop achieves higher observed success than most validation-selected fixed-budget settings while using fewer reasoning tokens than Full thinking. On WebShop with Qwen3.8-27B, it uses 42% fewer reasoning tokens per episode with similar observed task success. 

---
# From DQI to C-DQI: Automatic Codebook Generation and Evaluation for Measuring Computational Discourse Quality Index 

**Authors**: Esther Shizgal, Ron Keinan, Daniel Shtauber, Roee Tzvi Esquira, Ori Aronson, Matan Toledano, Or Rappel-Kroyzer, Liron Lavi, Renana Keydar, Ittai Bar-Siman-Tov, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2610.27937)  

**Abstract**: Effective democratic deliberation is vital for civic engagement and the health of legislative institutions. However, assessing parliamentary discourse quality is traditionally done using expert-scored, theory-based codebooks (CBs), which are rigid, labor-intensive, and prone to biases. While Large Language Models (LLMs) offer a scalable alternative for both schema induction and discourse annotation, ensuring schema quality and mitigating models' inherent bias remains a clear challenge. In this work, we propose a complete protocol for LLM-based codebook generation and evaluation. Starting from the widely accepted discourse quality Index (DQI), we compare top-down (theory-based), bottom-up (data-driven), and hybrid approaches for generating context-adapted schemas. To evaluate these schemas, we define an extrinsic proxy measure, validated by manual annotations, alongside an intrinsic inter-model agreement (IMA) metric, which leverages differing model biases to signal overall reliability. Our experiments on 100 Israeli parliament transcripts reveal that inductive, data-driven schemas achieve higher IMA than top-down methods while maintaining validity, demonstrating their potential for reliable, large-scale analysis of political discourse. 

---
# Why Do LLMs Accept Generics? Explaining Acceptability using Psycholinguistic and Model-Derived Factors 

**Authors**: Mervat Abassy, Zhaoxin Feng, Tatsuki Kuribayashi, Kentaro Inui, Emmanuele Chersoni  

**Link**: [PDF](https://arxiv.org/pdf/2610.27928)  

**Abstract**: Psycholinguistic studies explore why humans accept some generic statements but reject others by showing that these judgments depend on more than how common a property is. People consider several other factors such as how well it sets the kind apart from others or whether it is characteristic of the kind.
Inspired by the linguistic and cognitive literature, we explore to what extent factors from these studies explain how large language models (LLMs) judge generics. Our results with twelve models on a psycholinguistic dataset and on naturally occurring generics show that models weigh how common a property is against how well it distinguishes the kind, and combine the two much as people do, though less clearly. Like people, they favor characteristic properties, but unlike the accounts of human judgment, they ignore striking ones. What matters most to LLMs is whether a statement is factually true, and much of the influence of human factors reaches their ratings through that judgment. In six 7-9B models, shifting the hidden states along each factor's direction moves the judgment as the factor predicts. 

---
# Polarity Blindness in LLM Safety: A Systematic Study of Negation, Antonym Substitution, and Adversarial Framing 

**Authors**: Diego Miguel Lozano, Erik Derner, Juan Antonio Pérez-Ortiz, Nuria Oliver  

**Link**: [PDF](https://arxiv.org/pdf/2610.27908)  

**Abstract**: Negation remains a persistent challenge for language models across NLP tasks and modalities. We show that this weakness can undermine LLM safety guardrails through a structural vulnerability we term polarity blindness: the failure to recognize harmful intent after polarity-altering linguistic transformations. We study two forms of polarity inversion: grammatical negation and antonym replacement, and use synonym replacement as a polarity-preserving control. We further test whether an adversarial textual wrapper amplifies these failures. Using a 260-prompt benchmark across 10 state-of-the-art LLMs and 3 languages (English, Spanish, Czech), stratified by harmfulness potential and evaluated on a validated 0-5 harm scale, we find that polarity blindness affects most evaluated models, with mean harm scores reaching 4.12 for the most vulnerable model. The effect persists across typologically differing negation systems, suggesting that polarity blindness reflects a broader limitation in semantic robustness rather than a language- or grammar-specific failure. 

---
# CollabSciReM: Aligning Language Models for Scientific Conversations via Science-Aware Reward Decomposition 

**Authors**: Sasi Kiran Gaddipati, Gollam Rabby, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2610.27901)  

**Abstract**: Large Language Models (LLMs) are increasingly used as research collaborators for hypothesis refinement, methodological planning, and technical analysis across multi-turn interactions. In this setting, a useful assistant must be accurate, methodologically justified, logically consistent, and token-efficient. Current alignment approaches rarely achieve all of these together. Instruction-tuned models gain conversational quality through verbosity, and we show that science-agnostic general collaborative rewards compress responses at the expense of scientific rigor. We propose CollabSciReM, a goal-conditioned reward framework that decomposes the session-level reward into conversational quality, scientific rigor, and structural efficiency. We also introduce SciQuality, a pairwise evaluation of multi-turn research conversations grounded in 300 held-out scientific papers and scored on six dimensions: methodological soundness, justification, logical consistency, factual correctness, novelty, and responsiveness. Fine-tuned on 8.62K synthetic research conversations, CollabSciReM is preferred on SciQuality over a science-agnostic reward baseline trained on the same data for all four Llama and Qwen models, including a 65.3% win rate at matched response length. For the 7B and 8B models, CollabSciReM outperforms the science-agnostic reward baseline on the primary metric in 8 of 10 single- and multi-turn benchmark settings. In a blind study, domain experts rated CollabSciReM significantly higher on novelty than the general-reward variant and reported fewer hallucinated responses than both the general-reward variant and the base model. 

---
# Testing Associative Creativity in Humans and Multimodal LLMs with Difficult-to-Name Visual Stimuli 

**Authors**: Manar Ali, Lisa Gottschalk, Simeon Junker, Lisa Hudalla, Alon Fishman, Joana Cholin, Stefan Kopp, Hendrik Buschmeier, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2610.27894)  

**Abstract**: We investigate associative creativity in descriptions of abstract visual stimuli by humans and multimodal large language models (MLLMs). While conceptual association is central to human communication, it remains unclear how humans and models adapt when visual stimuli become increasingly difficult to map onto familiar semantic concepts. We evaluate several state-of-the-art MLLMs and human participants along two dimensions of creativity, originality and effectiveness, using two datasets with controlled manipulations of difficulty. SilCC varies naming difficulty through progressively more complex combinations of abstract shapes, and PixCC varies difficulty through increasing levels of random visual noise. Results show that humans produce more diverse associative descriptions than MLLMs, including greater semantic distance, higher entropy, and more novel expressions relative to reference lexica. As difficulty increases, effectiveness declines in SilCC, and humans increasingly shift from holistic associations to descriptions of low-level visual properties in PixCC. 

---
# UvA-MT's Participation in the WMT26 General Translation Shared Task 

**Authors**: Di Wu, Sergey Troshin, Wafaa Mohammed, Seth Aycock, Evgeniia Tokarchuk, Vlad Niculae, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2610.27852)  

**Abstract**: This paper presents UvA-MT's submission to the WMT 2026 General Machine Translation shared task, competing in the unconstrained track across all 23 translation directions. This year, we fully leverage the test-time methods of Large Language Models (LLMs) for machine translation. Specifically, (1) we use sequential sampling with a refinement prompt to generate a pool of translation candidates for each source segment, a strategy shown to outperform traditional parallel sampling; and (2) we employ LLMs as pairwise judges to select the best candidate via a round-robin voting mechanism, with offline evaluation on the WMT25 benchmark showing a very clear improvement over point-wise best-of-N selection. Combined, these two strategies form our WMT26 submission. 

---
# SciDiagBench: A Type-Specific Benchmark for Human-Aligned Evaluation of Scientific Diagrams 

**Authors**: Hao Wang, Zixuan Li, Long Bai, Xiaolong Jin, Jiafeng Guo, Tat-Seng Chua, Tian Tian  

**Link**: [PDF](https://arxiv.org/pdf/2610.27841)  

**Abstract**: Recent agents have demonstrated strong capabilities in generating images from textual prompts, motivating their utilization for scientific diagram generation. Scientific diagrams naturally span diverse types, each characterized by distinct visual structures and content organization. Consequently, during peer review and other scientific evaluation, human experts naturally consider type-specific criteria when assessing their quality. However, existing VLM-based evaluation benchmarks fail to reflect the above human evaluation practices for two main reasons: 1) benchmark datasets typically lack a systematic categorization of different diagram types and thus adopt the same generic criteria for all diagrams, and 2) their evaluation criteria are often designed heuristically, without explicit alignment to human evaluation preferences. Motivated by these limitations, we introduce SciDiagBench, which consists of a benchmark dataset that systematically categorizes scientific diagrams into 11 types and a human preference-guided evaluation method. Specifically, SciDiagBench contains 165 scientific diagrams across 10 broad domains, with types including graphical abstracts, technical roadmaps, and illustrative cases. Based on this dataset, SciDiagBench develops an evaluation method that leverages a preference-guided calibration process to incorporate human preferences into the generation of type-specific criteria. Ultimately, SciDiagBench enables more reliable automated evaluation that better aligns with human evaluation across diverse scientific diagrams. 

---
# SMART: Semantic Segment Mapping and Assessment of Recognition Transcripts with Contextual Rubrics 

**Authors**: Xiaoyu Gu, Chencheng Wang, Fuyang Li, Yu Xi, Tao Liu, Jiaoqi Guo, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27820)  

**Abstract**: Word error rate (WER) assigns equal cost to lexical edits despite their different semantic consequences, while overall semantic scores cannot identify which individual errors alter the intended meaning. We formulate ASR semantic evaluation as context-aware, segment-level consequence assessment and propose Semantic Segment Mapping and Assessment of Recognition Transcripts with Contextual Rubrics (SMART), a framework for traceable semantic evaluation and targeted ASR correction. SMART maps reference-hypothesis discrepancies into coherent segments and applies a structured contextual rubric to assign explicit consequence labels. These labels support Semantic Impact Error Rate (SIER), which quantifies consequential errors at utterance, chunk, and lexical levels. Against majority-vote human labels, SMART achieves 0.843 F1 versus 0.745 for direct LLM judgment. Evaluation across multiple LLMs further demonstrates that SMART is relatively insensitive to the choice of evaluation model, while MER and SIER reveal different aspects of ASR quality. A separate target-reference-free rubric guides correction using the target hypothesis and surrounding ASR hypotheses, outperforming a matched w/o Rubric baseline on both datasets, with further absolute reductions of 0.85-0.98 percentage points in MER and 1.18-1.65 percentage points in SIER_lex. 

---
# ProVe: Proposer-Verifier Decoding 

**Authors**: Ziqi Jin, Weiwen Xu, Jiaxi Li, Yujie Zhong, Qi She, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.27815)  

**Abstract**: Speculative decoding accelerates autoregressive (AR) language models by letting a small model draft tokens serially and a large model verify them in parallel. We reverse these roles. Starting from a standard AR checkpoint, we fine-tune the large model into a proposer that predicts a block of future tokens in a single forward pass, then use one additional forward pass to verify the proposed block. This gives ProVe, Proposer-Verifier decoding. The verifier can be a smaller AR model or the proposer itself; the latter gives ProVe-lossless, which reproduces the proposer's greedy AR decoding. Block proposals already provide substantial room for acceleration: with a block size of 16, the proposed block has an average AR-matching prefix of 6.6 tokens on GSM8K. Across six benchmarks with a 32B proposer, ProVe-lossless reaches 2.5x average AR throughput without training a separate draft head. With a small 3B verifier, ProVe reaches 2.7x average throughput at 2.6 accuracy points below AR on the six-task average, providing a controllable tradeoff between speed and accuracy. 

---
# Are Linguistic and Arithmetic Circuits Dissociated in LLMs? 

**Authors**: Riku Kisako, Tatsuki Kuribayashi, Ryohei Sasano  

**Link**: [PDF](https://arxiv.org/pdf/2610.27790)  

**Abstract**: Large language models (LLMs) exhibit both knowledge of language, rendered by, for example, grammatical judgment ability, and abilities to perform tasks through language, such as math-word problem solving. Motivated by neurolinguistic studies of their neural dissociation in human brains, in this study, we investigate the overlap of their circuits in large language models (LLMs). Specifically, we examine the overlap and causal effects between the circuits extracted for linguistic and arithmetic tasks, using controlled materials and circuit discovery methods from the mechanistic interpretability field. Our experiments with five LLMs show that circuit similarity is mostly explained by task domains (linguistic vs. arithmetic), rather than by their input and output vocabularies or by the syntactic similarities between natural-language sentences and arithmetic expressions. This finding suggests that LLMs may process linguistic and arithmetic information using largely separate neural circuits, highlighting an ability-based modular organization of circuits within these models. 

---
# Hearsay or Fact? How Framing and Elicitation Shape LLM Probability Updates 

**Authors**: An-Sheng Jhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27745)  

**Abstract**: Large language models (LLMs) are increasingly used as judgment assistants and as stand-in participants in psychology experiments, yet little is known about how much their probability estimates move in response to a single new piece of information. We pair premise-hypothesis items from $\delta$-SNLI with one strengthener and one weakener each, elicit the prior and the posteriors in separate calls, and measure the directional movement s: the mean posterior change signed by the annotated direction, so that the prior cancels out. We vary whether the update is stated ("Here is something else you learned") or reported ("Someone said that"), and whether the model gives one probability or five weighted estimates. We test seven configurations: two local models (Qwen3-8B, OLMo-2-7B; 240 pairs) and five configurations run through agent command-line harnesses (120 pairs), with hypotheses and analysis rules fixed before the formal runs (not publicly registered). Under stated, direct elicitation all seven move in the annotated direction (s = 18-28 percentage points). Reporting lowers s in all seven point estimates (significantly in six, by 2-5 points) and raises the share of reverse movements in all seven: the effect is consistent but small. Five weighted estimates lower the share of responses at $\leq$5% or $\geq$95% in all seven, but in the two local models they also reduce s by 43% and 68%, whereas in the harnessed configurations the point estimates of the change lie between -3.7 and +0.5 points. An elicitation format that reduces extreme values thus need not preserve sensitivity to evidence; the two should be tested separately. Because $\delta$-NLI annotates only direction, we make no claim about how large an update should be. Code, data and the frozen plan: this https URL 

---
# Tracing Recurrent and Emergent Hallucination Patterns in Decomposed Text-to-SQL Generation 

**Authors**: Hayden Moore, Abhinav Verma, Anshuman Dutt  

**Link**: [PDF](https://arxiv.org/pdf/2610.27720)  

**Abstract**: Large language models can achieve strong Text-to-SQL benchmark performance while still producing schema and query-logic errors. Most evaluations characterize such failures only in the final generated query, making it difficult to determine whether a related error was also observed in an earlier, simpler step or appears only at the final stage. We present a drill-down evaluation framework that decomposes Text-to-SQL queries into progressively more complex executable sub-queries and corresponding natural-language questions, allowing rule-based hallucination categories to be examined across a generation path. We distinguish recurrent hallucinations, in which the same rule-based category is observed in an earlier step and again in the final query, from emergent hallucinations, in which the category is observed at the final step without an earlier detection in the same path. Across five LLMs evaluated on an expanded BIRD-mini benchmark, several categories show appreciable recurrence probabilities, while clause abuse shows comparatively higher conditional emergence. Under the conversational-history condition, recurrence probabilities are generally higher and emergence probabilities are generally lower. These observations indicate that the two conditional measures do not move uniformly in the same direction under additional context and motivate trajectory-level analysis in addition to final-output evaluation. Code and evaluation artifacts are available in the project repository. this https URL 

---
# Beyond Calibration: Temporal LLM Confidence Assessment Across Languages and Local Settings 

**Authors**: Sahil Kale  

**Link**: [PDF](https://arxiv.org/pdf/2610.27702)  

**Abstract**: Large language models are expected to estimate their own uncertainty, with confidence signals used to guide retrieval, adaptive compute allocation, routing, and selective abstention. We investigate whether confidence signals measured at different temporal stages of problem solving encode distinct forms of epistemic information and remain reliable beyond English-centric settings. Inspired by classical meta-memory theory, we distinguish anticipatory pre-solve confidence (Feeling-of-Knowing, FOK) from retrospective post-solve confidence (Judgement-of-Learning, JOL). We evaluate these signals across two frontier LLMs, multiple reasoning domains, translated Marathi benchmarks, and a newly constructed benchmark of genuinely local Marathi factual questions. Across settings, the cheaper pre-solve signal remains weakly calibrated, while post-solve confidence generally provides stronger epistemic alignment despite its additional inference cost. Confidence behaviour transfers surprisingly well under direct language translation, with calibration trends largely preserved between English and Marathi, but degrades substantially on local Marathi knowledge, suggesting that linguistic transfer alone is insufficient for region-specific information needs. Finally, hallucinated responses maintain persistently elevated confidence after answer generation, highlighting important limitations of current LLM self-assessment mechanisms. 

---
# Memory as Code: Declarative Retrieval over Codified Long-Term Memory 

**Authors**: Aoting Zeng, Yuxin Du, Jianwei Wang, Yizhang He, Shan Jiang, Xuemin Lin, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27700)  

**Abstract**: In long-term memory systems, memories are constructed from interaction histories and retrieved in response to queries, enhancing the reasoning ability of LLM agents. In complex scenarios, facts are scattered across sessions and their relationships evolve over time, making it difficult for existing memory systems to compose them to adapt to diverse queries. Motivated by the modularity and composability of code, we propose Memory as Code (MaC), a unified memory framework that jointly designs memory construction and retrieval through a shared code abstraction. MaC couples codified construction, which establishes executable memory abstractions, with declarative retrieval, which translates queries into structured retrieval programs. Specifically, codified construction builds a three-layer executable memory structure that progresses from an atomic function layer, through a relational layer, to a modular composition layer. Declarative retrieval recasts retrieval from relevance-based fact selection into declarative query processing over memory. Experiments on LoCoMo and LongMemEval show relative gains of up to 15.91% in overall LLM-judge accuracy over the strongest baseline in each setting, while substantially reducing token consumption, highlighting the effectiveness of memory as code for long-horizon memory tasks. 

---
# Corpus Quality Improves, Retrieval Utility Does Not: An Ablation of Preprocessing and Refinement in a Speech-to-QA Pipeline 

**Authors**: Wonsup Shin, Jingu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.27652)  

**Abstract**: Practitioners building training corpora with a pipeline of language models keep a preprocessing or refinement (post-process) stage when switching it on raises the corpus quality. We ablate a production pipeline that turns conference speech into question-answer pairs. The preprocessing stage rewrites the transcript, and the refinement stage filters the generated pairs. Switching each on and off gives four corpora from the same 40 Korean medical and finance sessions. On the corpus itself, quality rises as the stages switch on. Four LLM judges and six human domain experts rank the four corpora the same way, and the fully refined corpus leads by +0.18 and +0.22 on a 1-5 scale. We then ask whether that gain carries downstream. We supply each corpus to six models at inference and measure retrieval accuracy. The ranking does not hold. Enabling the transcript rewrite lowers retrieval accuracy by 0.061, the only stage effect that survives correction for multiple comparisons. That stage also reads the whole session, so it almost doubles the cost of one. With both stages on, accuracy does not change. An equivalence test we declared in advance confirms this. The corpus itself is clearly valuable. It lifts retrieval accuracy from 0.231 to 0.748 in every model and domain we test. More than a third of its pairs carry knowledge that neither of two frontier models produces on its own. On our data, the null does not come from weak corpora. A rising score on a rubric of this kind did not predict what the corpus holds for a model. 

---
# Speak vs. Know: Balancing Behavior and Knowledge Distributions of Large Language Models 

**Authors**: Yang Liu, Masahiro Kaneko, Chenhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27648)  

**Abstract**: Recent work reports that the distribution verbalized directly in the output text aligns far better with human opinion distributions than the next-token log-probability distribution of LLMs. However, the verbalized distribution reflects what LLMs know about a group's opinions (knowledge distribution), while what they actually output at deployment is governed by their log-probability distribution (behavior distribution). We find that when prompting LLMs to predict the opinion of a specific group, the most likely answer under the verbalized distribution differs from that under the log-probability distribution for 37% of World Values Survey questions, averaged over three LLMs. Thus, LLMs' knowledge does not match their behavior. This matters for evaluation: a verbalized distribution can score well against a group while its most likely answer is one the model never outputs, so the score no longer describes the model. The alignment scores for these two distributions reflect only the model's behavior or its knowledge, separately. We therefore propose a learnable combination of the two distributions that follows the most likely answer of the behavior distribution while preserving the shape of the knowledge distribution, with a single hyperparameter trading off behavioral fidelity against distribution quality. Our experiments on three opinion survey datasets and six models show that the balanced distribution improves behavioral consistency over the verbalized distribution from 71.6% to 95.6% on average, while the total variation distance to the human opinion distributions rises only slightly from 28.8% to 33.8%, and it remains effective under temperature scaling and cross-dataset settings. 

---
# A Two-Way Street: Surveying the Interplay between Code-Mixing and Native Languages in Text and Speech 

**Authors**: Debajyoti Mazumder, Akash Anil  

**Link**: [PDF](https://arxiv.org/pdf/2610.27631)  

**Abstract**: Systems for code-mixed (CM) text and speech rarely start from scratch. CM data is scarce, so most work borrows from the languages being mixed, which we call native languages. This transfer can also run in reverse, when CM data is used to improve models for the native languages. Existing surveys focus on processing CM itself, and none brings the two directions together. This paper fills that gap with a systematic survey of transfer in both directions, across text and speech. We organise the surveyed work in a taxonomy of transfer direction, route and modality. Through this taxonomy, we find that most transfer runs from native languages into the CM setting. The reverse direction is younger, and in it code-switching mainly helps by aligning the two languages. Text and speech work share methods but rarely cite each other. On a shared speech benchmark, native data helps more than new architectures. These findings point to open challenges, such as how much native and CM data to mix in training, and how to evaluate CM text written in different scripts. 

---
# PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking 

**Authors**: Pan Wang, Yihao Hu, Guangyuan Dong, Yitong Lin, Hang Wang, Zimi Liu, Zhihao Wen, Jianshe Li, Jiang-Ming Yang, Wei Wu, Yongqi Tong  

**Link**: [PDF](https://arxiv.org/pdf/2610.27628)  

**Abstract**: Looped language models can run more iterations at test time, but additional iterations need not help them solve longer problems. For sequential state tracking, final-answer supervision does not specify how far each iteration should advance, and repeated updates can change the scale of the recurrent state. We introduce \textbf{Progress-Aligned Recurrence (PAR)}, which trains the prediction at iteration $k$ to match the state after the first $k$ operations and rescales recurrent states to their first-iteration norms. On swap-$S_5$, a Qwen2.5-0.5B model trained on at most 32 moves achieves 95.5\% mean full-state accuracy at 512 moves, or $16\times$ the training length, compared with 55.7\% reported for RLT and 0.8\% for explicit chain of thought on the same backbone. Controlled $S_5$ ablations show that removing norm anchoring reduces the extrapolation factor from $16\times$ to below $2.2\times$, and replacing progress-aligned targets with final-answer supervision reduces it to at most $8\times$. A learned completion readout selects when to answer without a supplied stopping depth. When intermediate labels are unavailable, prefix self-distillation induces the correspondence from shorter problem prefixes using only final answers. The gains apply across sequential state-tracking tasks, establishing the task structure required for depth extrapolation. These findings support training intermediate predictions and state scale together to make additional recurrent depth useful for longer problems. 

---
# ABench: Adversarily Benchmarking Large Language Models across Taxonomized Values 

**Authors**: Liping Yang, Yongqi Leng, Jingting Zheng, Yuqi Ren, Shaolin Zhu, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.27611)  

**Abstract**: Ensuring the alignment of Large Language Models (LLMs) with diverse cultural values is critical for their safe global deployment. However, existing alignment benchmarks are predominantly English-centric or confined to simple discriminative tasks. Moreover, they often fail to effectively evaluate model robustness against complex, adversarial inductions within specific cultural contexts. To address these limitations, we propose ABench, a fine-grained, handcrafted benchmark specifically tailored to taxonomized values in Chinese. Grounded in a rigorous hierarchical taxonomy comprising 158 derived value subcategories, ABench consists of 2,895 open-ended questions. These questions employ techniques such as role-playing and logical traps to probe the stability of models' stances in adversarially induced contexts. Furthermore, we propose a weighted automated evaluation framework that utilizes question-specific scoring criteria for nuanced assessment. Extensive experiments on 16 mainstream LLMs reveal that most models exhibit significant vulnerabilities when subjected to sophisticated adversary attacks on values (scoring below 60). This underscores the need for more robust, culturally specific alignment strategies. Our data will be publicly released to promote further research on taxonomized value alignment. 

---
# Self-Decomposing Diffusion for Code Generation 

**Authors**: Lize Shao, Zichen Xie, Zhizhen Chen, Zhe Zeng, Wenxi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27600)  

**Abstract**: Diffusion language models (DLMs) are emerging as a promising paradigm for code generation, offering the potential for fast, parallel decoding. However, generating complex programs remains challenging, as their interdependent components must maintain consistent interfaces, data flows, and shared assumptions. Aggressive parallel decoding can compromise these dependencies and undermine program correctness, while conservative decoding sacrifices efficiency. To address this trade-off, we propose planning program decomposition before implementation to coordinate interdependent units during parallel generation. We introduce Self-Decomposing Diffusion for Code (SDDC), a novel framework that decomposes complex programs and implements their units in parallel, entirely through diffusion. SDDC first generates a contract specifying the program's decomposition and inter-unit dependencies, then uses it to guide coordinated parallel implementation. Across five benchmarks and two 7B diffusion backbones, SDDC substantially improves the accuracy--latency trade-off, raising pass@1 by up to 31.1 percentage points while decoding 7.5--13.7$\times$ faster than the base models, and outperforming a fine-tuning baseline by up to 23.5 points with lower latency. 

---
# WORTH Benchmark: Evidence Discovery under Not-Fully-Cooperative Information -- A Two-Episode Pilot 

**Authors**: Bo Zou, Chao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27597)  

**Abstract**: Existing reasoning evaluations typically provide task-relevant information to models as available premises, rarely examining concurrently how models discover evidence autonomously and how information presentation within narratives guides or obscures judgment. This paper proposes the WORTH Benchmark: models read original mystery narratives without clue annotations, autonomously select key clue sentences, identify case participants, and explain their reasoning process. The narrative materials are designed by professional authors such that evidence facilitating reasoning coexists with narrative arrangements that delay the revelation of truth. We report a pilot experiment under full-story conditions across two episodes, conducting a total of 70 formal runs on models, of which 67 runs are content-scorable. Scoring separately records hits on author-designated key clue sentences and participant selections; natural language explanations are used for case diagnosis and are not scored independently. The results show that clue hits and participant judgments are not synchronized; across 37 content-scorable runs in Episode 02, 31 runs simultaneously selected the same set of incorrect participants, while adjacent suspicion-oriented and exoneration-oriented sentences were selected 25 times and 0 times, respectively. These phenomena illustrate the diagnostic value of component-wise analysis and suggest that not-fully-cooperative informational relationships may influence evidence selection and judgment. This study only implements full-story conditions for two episodes, and does not establish stable model rankings or infer general mechanisms on this basis; human player records have not yet been cleaned, and this version does not report human-model comparisons. 

---
# Benefit and Cost of Random-Token Replacement under Repeated Data: A Known-Entropy Decomposition and Evidence from Ten Real Corpora 

**Authors**: Wonsup Shin, Jingu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.27590)  

**Abstract**: When unique pretraining data is scarce, models see the same tokens many times and are thought to memorize. Random-token replacement, an input-corruption augmentation, is one remedy reported to lower validation loss, but a lower loss alone does not reveal which source of error it reduces. On a controlled source whose entropy rate is known in closed form, we show that replacement shrinks predictions toward the corpus average. We also introduce a decomposition that separates validation loss into approximation, estimation, and memorization terms. This shrinkage view predicts that replacement's bias cost needs no memorization but its benefit does. Memorization raises validation loss after its minimum, so we report the effect at two points, the minimum validation loss and the end of a fixed budget. Where replacement helps, it reduces this rise by about 60%. At the end of the budget, it turns from harmful to helpful as the controlled source becomes less predictable; at the minimum, it helps at no predictability level under the fixed recipe. On Korean and Japanese web text, replacement removes about a third of the loss rise after the minimum by the end of the budget, but its effect at the minimum is much smaller. Across ten real corpora, replacement lowers the final loss on nine. It harms only the census corpus, program-generated JSON that is the most redundant of the ten and much more redundant than any of 64 further prose and code corpora we measure. 

---
# Beyond Model Weights: Sampling-Dependent Fairness in Diffusion Language Models 

**Authors**: Koshiro Saito, Diasuke Oba, Danushka Bollegala, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2610.27584)  

**Abstract**: Diffusion language models (DLMs) are emerging as a practical alternative to autoregressive language models (ARLMs), yet their fairness remains largely uncharacterized. DLMs introduce sampling controls with no direct autoregressive counterpart, such as block length and remasking strategy, that directly set their speed--quality operating point. This raises a basic question: does a DLM's measured social bias remain stable across these settings? Across LLaDA, LLaDA-1.5, and Dream on Open-BBQ, we find that all three models exhibit social bias, but both the magnitude of measured bias and the relative ordering of models by bias are highly sensitive to the sampling configuration. This sampling dependence also changes what successful debiasing means. While mitigation strategies established for ARLMs transfer effectively to DLMs and reduce bias without degrading downstream performance, sampling dependence persists even after mitigation. Our results show that fairness in DLMs cannot be characterized independently of the sampling configuration, even after debiasing. Project page is available at this https URL. 

---
# TeCoP: Accelerating Diffusion Large Language Model Decoding via Tentative Context Pipelining 

**Authors**: Jui-Chien Tsou, Hsuan-Chi Liu, Wei-Chin Wang, Yu-Syuan Xu, Chun-Yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.27583)  

**Abstract**: Diffusion large language models (dLLMs) predict multiple token positions per forward pass, yet block-wise decoders serialize generation and repeatedly recompute stable predictions. We attribute this inefficiency to coupling a prediction's readiness for irreversible commitment with its utility as context. We introduce Tentative Context Pipelining (TeCoP), a training-free decoder that decouples these roles through staggered block-parallel pipelining and draft-and-verify. TeCoP reuses stable but uncommitted predictions as provisional cross-block context, with commitment governed by confidence and stability checks. Across four LLaDA and Dream dLLMs on GSM8K, Minerva MATH, and MBPP, TeCoP reduces number of forward evaluations by up to 87% and achieves up to 7.50-fold throughput without KV caching and 11.77-fold with caching over the uncached original decoder. It outperforms Fast-dLLM and DSB in throughput across all 12 model-benchmark settings, with and without caching, while keeping average cached accuracy within one percentage point of the original decoder. 

---
# Factual Access States in Diffusion Language Models: From Mechanism to Knowledge Editing and Retrieval 

**Authors**: Yuchen Guo, Zhuoyuan Mao, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.27580)  

**Abstract**: Factual recall is a lookup: some representation inside the network serves as the key, and what that key is made of decides which queries can retrieve the fact, what an edit must cover, and what a retriever can read. We call it the factual access state. In autoregressive (AR) models the subject position enriches attributes largely independently of the relation, and relation-specific selection happens later, at the readout position. In masked diffusion language models (DLMs) we find the state is assembled differently, and much earlier: in the first attention layer the wording of the relation is routed into the position of whichever entity is visible and the early feed-forward layers look the fact up on this fused representation. The state therefore varies with the relation's surface form and with the query direction, so the queries about one fact occupy a region of state space rather than a point, and an operation on the fact succeeds only for queries that enter it. We treat the state as an interface to a DLM's factual knowledge, and put it to two uses. Writing to the state gives a knowledge editor: from one declarative statement, Multi-View Edit fits one shared value per direction across that region and installs it in the early-MLP block, preserving source-prompt reliability while substantially improving paraphrase and reverse question answering, where AR editors gain near zero. Reading it suggests a complementary use: the state at each entity position can serve as a relation-specific retrieval vector with no training, returning both pieces of evidence for two-entity queries on which pooled, prompted and lexical retrievers almost never return both. 

---
# Why Static Probes Fail: Context Determines Deception Feature Composition 

**Authors**: Yao Huang, Siyuan Chen, Zhenghan Zhou, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.27577)  

**Abstract**: With increasing autonomy, Large Language Models (LLMs) may engage in deceptive behaviors, such as scheming to pursue misaligned goals or sandbagging to escape oversight. Monitoring outputs alone is insufficient, as models can produce seemingly benign responses while harboring misaligned internal objectives. While mechanistic approaches like linear probing are promising, they rely on the assumption that deception corresponds to a single direction in representation space. We demonstrate that such single-direction probes fail to generalize across diverse deception scenarios. To bridge this gap, we propose representing deception as a multi-dimensional subspace rather than a single direction. Our method utilizes iterative orthogonal decomposition to identify complementary directions that capture distinct facets of deceptive intent. To effectively leverage these directions for detection, we find that static combinations prove brittle due to strong domain dependence across different scenarios. We therefore introduce a domain-adaptive gating mechanism that dynamically composes the extracted directions through few-shot adaptation at test time. With only 10 examples, our approach achieves an average AUROC of 0.97 and 94% recall at 5% FPR across all evaluated datasets, substantially outperforming existing methods. 

---
# AudiPsy: Evidence-Grounded Multimodal Reasoning over Spoken Counseling Conversations for Depression Assessment 

**Authors**: Chuanbo Zhu, Run Luo, Weibo Zhao, Yue Li, Kun Qian, Qiang Qu, Yike Guo, Minghuan Tan, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2610.27567)  

**Abstract**: Depression assessment from counseling speech requires integrating textual and acoustic psychological evidence across turns, yet existing end-to-end classification approaches lack explicit multimodal evidence interpretation and context-aware reasoning. We propose AudiPsy, an evidence-grounded omni-LLM for depression assessment over spoken counseling conversations. AudiPsy employs an Observation-Assessment-Summary (OAS) paradigm to integrate dialogue transcripts, textual and acoustic emotions, and textual-acoustic emotion incongruence into structured reasoning. OAS incrementally organizes multimodal observations, performs contextualized assessment, and maintains an evolving dialogue summary for subsequent turns. To support this framework, we construct AudiPsy-Counsel, a synthetic spoken counseling conversation corpus with multimodal context augmentation, and develop a unified three-stage curriculum for depression-oriented reasoning. Experiments on DAIC-WOZ demonstrate superior depression assessment performance over specialized depression detection methods and general-purpose speech/omni-LLMs, along with improved reasoning quality, while supplementary evaluation on EATD-Corpus supports AudiPsy's applicability to naturalistic conversations. 

---
# Learned Readout Tokens for Extracting Large Language Model Confidence 

**Authors**: Nima Seifi, Martin Ester  

**Link**: [PDF](https://arxiv.org/pdf/2610.27566)  

**Abstract**: Large language models (LLMs) are increasingly used within everyday applications where their responses have significant downstream impacts. For responsible deployment of such systems, we desire insight into how much we can trust their individual outputs. A popular approach for estimating LLM confidence is to train probes on the hidden states of the LLM's tokens to predict the correctness of the model's output. These probes typically read hidden states from tokens that are already in the model's vocabulary. In this work, we introduce a new approach to confidence probes that learn a new readout token and utilize its hidden states to train the probe. We show that our learned readout approach ScratchProbe outperforms its existing-token counterpart on a variety of datasets and LLMs. We also extend our method to a multi-token approach which we show provides further performance improvements. Finally, since the learned readout changes only the position a probe reads from, any existing probe can adopt it. Applied to a state-of-the-art probe, it improves AUROC on every backbone, in domain and out of domain. 

---
# Exploring Recurrent-Depth Language Models for Multitask Clinical Text Processing 

**Authors**: Xi Yang, Guodong Liu, Cheng Peng, Kaleb Smith, Yingying Zhang, Xing He, Yi Guo, Jiang Bian, Haoteng Tang, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27559)  

**Abstract**: Clinical natural language processing includes diverse tasks such as information extraction, classification, and question answering. Developing one model for all tasks could simplify deployment and improve knowledge transfer. We evaluated whether small recurrent-depth language models can meet this need through joint fine-tuning and compared their performance with standard transformers. We fine-tuned Ouro1.4B-Thinking, Ouro-2.6B-Thinking, HRM-Text-1B, Qwen3-1.7B, and Qwen3-8B on 9 datasets covering information extraction, classification, inference, and question answering, and evaluated them on four additional datasets excluded from training. Across these 13 datasets, Ouro-1.4B-Thinking achieved higher performances than Qwen3-1.7B on 11 datasets; Ouro-2.6BThinking was competitive with Qwen3-8B on most tasks. Ouro models showed the largest gains on inference and question answering, whereas HRM-Text-1B showed mixed results. These findings support further study of recurrent-depth models for clinical tasks through controlled experiments to isolate architectural effects and investigate how recurrent computation influences reasoning, knowledge acquisition, and generalization. 

---
# EviDial: When Correct Depression Predictions Lack Evidence 

**Authors**: Junjie Xu, Jin Zhu, Yuzhe Yang, Luhan Wang, Xingjiao Wu, Daoguo Dong, Jing Yang, Wenxin Wu, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2610.27541)  

**Abstract**: Depression dialogue benchmarks commonly assess models by comparing predicted symptom labels with clinical scale scores. However, a prediction may agree with a benchmark label even when the dialogue lacks enough evidence to justify it. We introduce EviDial, a benchmark that distinguishes label agreement from evidence-grounded assessment. EviDial comprises 382 depression dialogues from three sources and 2,744 participant-symptom cells, each classified as having absent, partial, conflicting, or sufficient evidence. Because evidence sufficiency lacks a directly observable gold standard, we derive calibrated evidence references from multiple judgments using generalizability theory, many-facet Rasch measurement, and random-effects latent class analysis. Our audit reveals that 82.8% of participant-symptom cells lack sufficient textual evidence for the corresponding symptom assessment, with substantial variation across sources. Across model families and prompting strategies, 69.2%-96.9% of predictions judged correct by benchmark labels are nonetheless unsupported by sufficient dialogue evidence. Further analyses show that strong agreement among LLM evaluators need not reflect independent evidence judgments, underscoring the need to calibrate automated reference panels. These findings demonstrate that label-based evaluation can overestimate evidence-grounded performance by conflating agreement with clinical labels and support in the observed dialogue. EviDial provides a measurement layer for separating these constructs and auditing LLM-based depression assessment more faithfully. 

---
# Judging the Justification: Decision Certificate Supervision for Generative Reward Models 

**Authors**: Guiquan Sun, Bing Bai, Martin Renqiang Min, Dongjin Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.27528)  

**Abstract**: Generative reward models (GenRMs) generate natural language rationales to justify preference judgments, yet a correct verdict does not guarantee a sound justification. Existing approaches often rely on reference explanations to supervise rationales, but obtaining such references can be costly and does not necessarily establish whether generated rationales support the resulting verdict. To address these limitations, we propose DeCert, a framework that evaluates the validity and verdict consistency of generated rationales without requiring reference explanations. Crucially, DeCert translates these assessments into preference supervision that accounts for both verdict correctness and justification quality. In particular, when multiple sampled judgments share the same correct verdict, differences in certificate quality enable DeCert to construct rationale preference pairs that distinguish stronger from weaker justifications. Extensive empirical studies across two model families and multiple in- and out-of-distribution benchmarks demonstrate that DeCert consistently outperforms strong generative judge baselines while improving rationale quality. Moreover, DeCert effectively mitigates positional bias, improving position-consistent accuracy by up to 14.8 points. These results highlight the importance of supervising not only which verdict is correct, but also how that verdict is justified. 

---
# Detecting Errors in Large Language Model Responses: The Incremental Predictive Value of Hidden States 

**Authors**: Ningzhi Li, Xiaojun Ma, Xiaofei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27455)  

**Abstract**: Although large language models support a wide range of tasks, hallucinations remain a major obstacle to user trust. When a model generates an answer, both its internal processing and final output vary with the question. Internal states are often expected to offer richer signals for hallucination detection. However, studies using these states typically report their standalone discrimination performance, while output-side comparisons often use simple baselines with weaker modeling capacity. We integrate output-side information available at inference time into a combined baseline and compare detection performance before and after adding hidden states under identical data splits and evaluation rules. Experiments cover open-ended generation and constrained multiple-choice tasks across different models. Adding hidden states improves AUROC by approximately 0.004 and 0.039 on the two tasks, respectively. Hidden states thus provide additional predictive information, but strong standalone detection performance can still yield only limited additional gains. 

---
# The Router Already Knows 

**Authors**: Andrew Bistras, Marty Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2610.27421)  

**Abstract**: Mixture-of-experts (MoE) models already make a sparse choice for every token: which experts should process it. We ask whether a sparse dictionary should reuse that choice. NR-SAE gives each routed expert a small dictionary and opens only those the frozen router selects. At block 15 of Gemma 4 26B-A4B, after ten million C4 training tokens and across three seeds, the expert-aligned autoencoder explains 0.518 of the variance in router-weighted expert outputs on a fixed evaluation capture. A rank-addressed control with the same parameters but sixteen times as many candidates reaches 0.481, a candidate-matched control 0.458, and an architecture-matched control with randomized expert identity 0.387. A transcoder variant also leads all three controls: 0.432, against 0.347, 0.371, and 0.118. Part of this advantage is architectural: each expert writes into a subspace of at most a quarter of the model width, so naming the expert names the subspace. How much it explains, we have not measured. At this layer and budget, the router already knows something a dictionary would otherwise have to learn. Our implementation did not train faster, and semantic interpretability, downstream loss, and causal faithfulness remain untested. 

---
# Looking Into The Past: The Impact of Context on Detecting Deceptive Statements 

**Authors**: Alex Clay, Ernesto Jimenez-Ruiz, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2610.27407)  

**Abstract**: Deception detection typically considers messages in isolation. Previous works in claim verification and role detection have displayed that additional information such as gameplay history is crucial to successful predictions. In this work, we demonstrate the value of preceding gameplay dialogue when applied to deceptive message detection within the hidden role game of Avalon. The findings show that LLMs can see improvement across a number of metrics, including accuracy by up to 31\% with the addition of gameplay context for the target message. 

---
# DStill: Trajectory Consistency Distillation for Accelerated One-Step Drafting in Block Diffusion Speculative Decoding 

**Authors**: Shreyas Kulkarni, Uttam Gurram, Madhav Kanda, Krishna Teja Chitty Venkata, Shichu Chen, Kartik Balasubramaniam, Supriyo Chakraborty, Sambit Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27379)  

**Abstract**: Block-diffusion speculative decoding accelerates large language model inference by drafting multiple tokens in parallel. However, one-step drafting in discrete diffusion is limited by token factorization error: each token is predicted from the same masked block, without conditioning on the tokens selected at other positions in that step. This can produce inconsistent drafts and reduce acceptance length. Iterative refinement mitigates this limitation by conditioning later predictions on previously committed tokens, though additional passes increase drafting cost. To address this trade-off between draft fidelity and inference latency, we introduce DStill, a two-stage framework that distills a target-conditioned drafter's own multi-step refinement into one-step drafting. We first train the drafter to complete partially revealed blocks, then use its refinement trajectories to train a student to predict the final completion in one pass. Using a curriculum that gradually transitions from partial to fully masked inputs, we successfully transfer the multi-step refinement capabilities into one step. Across reasoning, math, code benchmarks on three model families DStill improves acceptance length by up to 28% and speedup by up to 25% over one-step DFlash, achieving up to 6.45x speedup over auto-regressive decoding. 

---
# SHAM-Bench: A Span-level Hallucination Benchmark Across Multiple Tasks 

**Authors**: Sicong Huang, Ian Lane  

**Link**: [PDF](https://arxiv.org/pdf/2610.27370)  

**Abstract**: Span-level hallucination detection pinpoints unsupported content in LLM outputs, enabling actionable feedback and targeted mitigation. Existing evaluations for span-level hallucination detection methods are fragmented across tasks, annotation formats, and metrics, and most span-level datasets cover only a single NLG domain, making it difficult to assess whether a detection method actually generalizes across tasks and domains. In this paper, we introduce SHAM-Bench (Span-level Hallucination Benchmark Across Multiple Tasks), a unified benchmark for span-level hallucination detection spanning six NLG tasks, aggregated from existing human-annotated datasets totaling 6,461 examples under a common schema and evaluation protocol. Using SHAM-Bench, we evaluate existing hallucination detection methods and find that fine-tuned detectors often perform well in-domain but degrade under distribution shift, whereas prompt-based methods with reasoning are more robust across tasks. Our analysis further shows that subtle unsupported content remains difficult to detect and span-level evaluation is still challenged by ambiguous span boundaries and noisy labels. We release the dataset, evaluation code, and a public leaderboard to support future research on broadly generalizable hallucination detection. 

---
# Explore More, Validate Less: Efficient Reflective Prompt Optimization 

**Authors**: Xiaofan Zhou, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.27367)  

**Abstract**: Reflective prompt optimization improves instructions using task feedback, but coupling candidate generation with full-validation-based selection can make search costly and sequential. We propose a two-stage framework that decouples reflective candidate-pool construction from validation allocation. During construction, we sample multiple parents and reuse each parent's training responses across parallel reflection branches. Each revision is screened against its parent on a small training minibatch, allowing locally improved candidates to support further revisions without full validation. After construction, an exploration policy selects candidates from the frozen pool under a fixed full-validation budget, and the prompt with the highest measured validation score is returned. This separation allows candidate-pool size and validation budget to be set independently. We evaluate random selection and learned score prediction with similarity-based exploration on five benchmarks using Qwen3.5-9B and Llama-3.1-8B-Instruct. With 100 retained candidates and 20 full validations, our framework improves over the initial solver by up to 11.22 percentage points. Learned selection improves over the initial solver in all ten comparisons, although neither selection policy consistently outperforms the other. These results support separating reflective search from budgeted validation while leaving room for stronger candidate-selection policies. 

---
# On the LLM Verification for Argument Graph Generation 

**Authors**: Jakub Bąba, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2610.27361)  

**Abstract**: Multi-agent LLM systems can formalize natural-language argumentation end-to-end, turning plain text into structured graphs by classifying components, connecting them with relations, and enriching the resulting graphs into formal models. Quality assurance for those graphs is almost always another LLMs: a reviewer, judge, or critic agent. On the other side, in adjacent neuro-symbolic fields, such generated output is often verified symbolically by external checks and solvers. In this work, we argue that in argument graph verification, the allocation should be similar, and based on what each instrument can guarantee. We propose layered verification: starting from deterministic checks, moving to solver-based and LLM reviews, and utilizing debate only under uncertainty, leaving human judgment for what no other instrument can decide. We ground our principle with a study on human-annotated graphs with injected structural defects. An LLM reviewer detects them at near-ceiling recall, including cycles in the support relation, but it objects to almost one in a ten correct graphs, even though it should not, happens to be inconsistent in verdict on identical inputs and is significantly slower than the equivalent exact check. It shows that while LLM reviewer does not lack the ability to detect structural defects, it is worth having a deterministic verification gates. 

---
# Global Security Blind Spots: Translation as a Safety Bypass in Low-Resource Languages 

**Authors**: Jalil Alva, David Harper, Abigail E. Thornton, Rafael S. Villalobos Alva, Leonardo Kenji Minemura Suazo, Konstantinos Karageorgos, Diego Velazquez, Sergio A. Serrano, Marvin Vargas Flores, Aaron Schliem, Fernando F. Migone  

**Link**: [PDF](https://arxiv.org/pdf/2610.27304)  

**Abstract**: Large language models are deployed globally, yet safety alignment and evaluation remain concentrated in English: harmful requests refused in English can elicit unsafe completions when translated into low-resource languages. We evaluate ten widely deployed LLMs across 79 languages (English plus 78 low-resource) under a translation-bypass threat model, using a metric conditioned on English safety, which separates translation bypass from unsafety already present in English. English safety does not transfer reliably: conditioned on safe English completions, mean bypass rates range from 21.9% to 44.4%. Bypass risk concentrates in a subset of language families, a pattern broadly shared across models. The shared ordering is confounded with translation degeneracy, but its variation across models is significant and survives every control we apply. We also show that per-model bypass rates do not identify a stable ranking: nine of ten models change rank across three conditioning schemes that estimate the same quantity, with no change to any underlying response. 

---
# Evaluating Context Compaction on the Questions That Come Later 

**Authors**: Junwei Yao, Yongxi Zhou, Zike Yang, Zhijun Wang, Jianan Liu, Yanxian Chen, Wenbo Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.27301)  

**Abstract**: Language-model agents compact their context for the question at hand, but the compacted context must also serve questions that come later. We evaluate compaction on these later questions, using four corpora that annotate the evidence of several questions per text and recorded agent trajectories, and compare every method with random deletion at the same budget. On MultiRC, keeping the sentences that gpt-6-luna (through the OpenAI Decisions API) judges necessary for the current question raises its accuracy on that question by 11.4 points at 25% of the tokens. On the later questions, however, no method that selects sentences for the current question does detectably better than deleting sentences at random. Deleting words instead of sentences does better: LLMLingua-2, which ignores the question, beats random sentence deletion at the same token count by 12.5 points on the later MultiRC questions. It keeps about twice random deletion's share of the evidence's content words, and at equal content-word retention its gain shrinks to 7.3 points (exploratory). On FEVER, label accuracy can overstate what the kept text supports: the gain of sentence selection is no longer detected when answers are scored against the kept evidence, and the 21.0-point gain of LLMLingua-2 is not shown to rest on the kept text. The ranking also depends on the setting: on Evidence Inference sentence selection beats LLMLingua-2 at a nominal 25% budget, and on $\tau$-bench trajectories, whose later tool calls need exact identifiers, it beats random deletion while LLMLingua-2 shows no detected gain. Scores on the question a compaction was made for do not reliably show how it serves later ones. 

---
# RealControlBench: A Benchmark for Interpretability-Informed Fine-Grained Control of LLMs and Agents 

**Authors**: Xinying Cai, Bin Hu, Mingyu Jin, Qiming Zhang, Yang Ouyang, Bangwei Guo, Minghao Guo, Yutong Yin, Lang Gao, Feijiang Han, Sheldon Yu, Kaiqu Liang, Qingcheng Zeng, Jingcheng Niu, Fan Yang, Wei Cheng, Zhen Xiang, Julian McAuley, Zhaoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27295)  

**Abstract**: Existing steering benchmarks are built around simplified, isolated generation tasks and evaluate a single behavioral attribute under controlled settings that do not reflect the diversity and complexity of practical deployment. As a result, performance on these benchmarks provides only a limited picture of how steering methods perform in the real world. We therefore introduce RealControlBench, a comprehensive benchmark for evaluating fine-grained steering of LLMs and LLM agents across diverse behavioral targets and interaction settings. RealControlBench extends steering evaluation beyond isolated generations to complex, execution-grounded, multi-turn agent interactions, covering five complementary dimensions: instruction-following control, agentic tool-use decision control, reasoning-performance steering, reasoning-length control, and trajectory-conditioned safety awareness. Rather than measuring steering success alone, the benchmark jointly evaluates target controllability and capability preservation, including instruction following, reasoning performance, and downstream agent utility and safety. It further examines whether steering effects remain specific and robust across prompt variations, domains, and model architectures. Our evaluation shows that existing steering methods remain unreliable across realistic control objectives: they often fail to consistently induce the intended behavior and can substantially degrade the model's underlying task performance. These results suggest that strong steering effects in controlled settings do not necessarily translate into reliable control in more realistic evaluations. By providing a unified, model-agnostic steering evaluation framework, RealControlBench aims to support the development of steering methods that are not only effective but also precise, robust, and useful in realistic AI deployments. 

---
# When Does Hate Transfer? A Large-Scale Study of Cross-Lingual Hate Speech Detection 

**Authors**: Zahra Bokaei, Walid Magdy, Bonnie Webber  

**Link**: [PDF](https://arxiv.org/pdf/2610.27285)  

**Abstract**: Effective detection of hate content is essential for maintaining healthy online communities, yet performance can vary substantially across languages. This work investigates a core practical question: Can training data from one language support hate-speech detection in another language with comparable performance? To answer this question, we conduct a large-scale study across 14 languages. \change{We observe consistent patterns in which certain language pairs transfer more effectively than others. Among the factors we measure, linguistic similarity is the strongest predictor of these patterns, with temporal overlap and dataset-size difference playing smaller roles. These measurable factors do not fully explain the observed patterns; a complementary qualitative analysis suggests that shared cultural cues and communicative norms may be associated with some of the remaining cases, particularly within several clusters. We present this cultural interpretation as a hypothesis from qualitative inspection rather than a measured cause.}{} Our analysis includes low-resource languages, offering practical insights into when cross-lingual transfer can support them and where its limitations emerge. These findings inform the design of more reliable multilingual hate-detection systems. 

---
# African Languages AI Safety Readiness Report 

**Authors**: Israel Abebe Azime, Jesujoba O. Alabi, Tajuddeen Gwadabe, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2610.27269)  

**Abstract**: With the widespread adoption of large language models (LLMs) across languages and domains, AI safety has emerged as a critical global concern. However, due to resource constraints, research and development efforts tend to prioritize a limited number of high-resource languages, leaving many others, especially those spoken in Africa, underrepresented. While existing research on language "resource-ness" has largely examined whether sufficient resources are available to develop language technologies, the question of whether languages are adequately represented in AI safety research has received attention only more recently. This report examines the readiness of African languages for inclusion in AI safety discussions, drawing on existing literature and data. It considers four factors: the availability of AI-relevant resources, including text data; national AI policy support; performance on established text benchmarks; and speaker population. Based on these factors, languages are grouped into four readiness categories: fully ready, mostly ready, emerging, and foundational support needed. This classification is intended to inform the inclusion of African languages in AI safety benchmarks, highlight varying levels of preparedness, and guide strategic investment in data development, policy, and community engagement for underrepresented languages. 

---
# Auditing Safety Preference Supervision with Recorded Derivations 

**Authors**: Chenhan Lyu, Amir Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2610.27266)  

**Abstract**: Preference labels align model training, but the judgments behind them are difficult to inspect or verify. In this paper, we examine how preference construction can be checked when intermediate decisions are recorded, and which of them remain uncertain. We present Symbolic Preference Labeling, a method in which a pretrained language model classifies predefined attributes of prompts and responses. Explicit rules assign response verdicts, and a fixed verdict ranking determines the preference label. Each preference label has a recorded derivation: the response verdicts and preference label can be recomputed from the stored attribute assignments, rules, and verdict ranking without rerunning the language model. On 200 prompts annotated by human raters, the language model produces valid structured outputs for 192 prompts. All eight parsing failures occur in the subset of self-harm prompts marked as harmful. Using 465 preference pairs, we train one policy model with direct preference optimization across five training seeds. Mean measured benign refusal is 40.82 percent on 1,319 benign prompts that resemble unsafe ones, and mean measured toxic refusal is 84.98 percent on 655 toxic prompts. This study distinguishes the reproducibility of preference labels from attribute correctness. The recorded derivations make preference construction open to inspection, while attribute correctness and semantic safety require separate validation. 

---
# Target-Aware Curation of Hallucination Stress Tests for Large Language Models 

**Authors**: Jiho Lee, Bingquan Zhang, Lin Ting Liang, Soroush Vosoughi, Weicheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.27247)  

**Abstract**: Hallucination stress tests can yield similar average scores across language models even when different prompts elicit hallucinations in each model. We develop a procedure that iteratively edits source-grounded seed prompts to expose hallucinations in a target model. An LLM editor revises each prompt in response to the target's answer and a judge's category-specific score and explanation. In the main setting, the target also receives feedback on its previous response. Experiments cover 500 seeds from seven datasets and ten topics, with intrinsic and extrinsic hallucinations evaluated across open-weight and closed-source models. Editing substantially increases judge-rated severity over the seeds. In the baseline configuration, retained responses fall in the judge's strong-hallucination bin in 77.2% of runs targeting extrinsic hallucination and 74.2% targeting intrinsic hallucination. Later editing rounds account for more than half of the mean score increase. Among 219 prompts that pass model-assisted source-validity checks, 89 elicit responses containing unsupported claims. Across five models, 200 curated prompts yield similar average severity scores, although pairwise Spearman correlations between prompt-level scores are only $\rho=0.0387$ to $0.1897$. Curated prompts can be reused across models, but their diagnostic value depends on source validity and the failures they expose in each model. 

---
# Diagnostic Analysis for Large Language Model Reasoning 

**Authors**: Xinyu Liu, Dongqi Fu, Jun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27245)  

**Abstract**: Reasoning structures (e.g., chains and graphs) are increasingly used for probabilistic inference, factuality verification, and uncertainty estimation, but their structural reliability is rarely evaluated directly. We test whether this reliability holds through a proposed two-stage diagnostic evaluation in open-ended multi-hop QA. Stage 1 tests different capabilities required to construct reasoning graphs, showing that graph reliability varies with topology, prompt design, and model family. Stage 2 uses uncertainty estimation as a downstream test case. Generated-edge experiments show that graph-based computation is sensitive to upstream graph errors, with performance dropping relative to the gold-graph setting. Gold-graph experiments further show that graph structure provides little benefit when used only as prompt context, but improves the separation of correct from incorrect answers by confidence score and enables node-level diagnostics. 

---
# Predicting Linear Relational Decodability Without the Relation 

**Authors**: Jonas Rohweder, Adrian Mülthaler, Farid Mehri, Sebastian Gerstner, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2610.27241)  

**Abstract**: Linear relational embeddings (LREs) approximate a language model's subject-to-object mapping with an affine function. Their faithfulness varies enormously across relations, from near zero for person_father to the ceiling for country_capital_city. We investigate the factors that contribute to this variation and demonstrate that it can be significantly predicted from a readout that is not aware of the underlying relation. To do so, we propose a novel form of the Jacobian lens, the entity lens. Whereas the Jacobian lens is estimated across all tokens and concepts, the entity lens EL_t is trained to predict what the model will output for an entity of type t under the neutral prompt "{} is known for"; e.g., we train ELs for countries and persons. After estimating an entity lens, we measure how highly it ranks the model's answer for a relation without using any relation-specific information in estimation and prediction. Across 28 relations of the LRE dataset on Olmo-3-7B this relation-agnostic score correlates with LRE faithfulness at \rho = 0.7355 (p < .01). This suggests that linear decodability is largely a property of what a subject's representation already carries, rather than of the relation being decoded. Moving the unit of analysis from the relation to the individual fact, we conclude that new relation datasets and analysis methods are needed to address several problems in existing datasets, most prominently semantically weak first tokens (e.g., most predicted universities' first token is "University") and reference competition (e.g., the LLM confuses obscure actor names with competing non-actor entities that have the same last name). Our full pipeline is available at this https URL 

---
# CEFR-oriented Russian Text Simplification Using Reinforcement Learning 

**Authors**: Kristina Mustakova, Alexey Sorokin  

**Link**: [PDF](https://arxiv.org/pdf/2610.27237)  

**Abstract**: We present a study on controlled automatic simplification of Russian texts using preference-based optimization algorithms with a focus on adapting texts to the A1, A2 and B1 proficiency levels. We fine-tune YandexGPT using Rejection Sampling and Direct Preference Optimization (DPO) methods to improve control over the complexity of the generated text. We compare several training configurations that differ in dataset composition, text type and the strategy used to construct preference pairs for DPO. For automatic evaluation we train a regression model based on ruRoBERTa-large to predict continuous complexity scores corresponding to CEFR levels. We also consider semantic similarity, lexical coverage and quantitative readability metrics. Human evaluation assesses simplicity, fluency and adequacy. Our best model achieves 87.15\% target level accuracy, an MAE of 0.35 CEFR levels and a target interval MAE of 0.11. Manual assessment shows that 75% of the evaluated simplifications match the target morphological and syntactic level, and fluency and adequacy scores are especially high for A2 and B1. Although A1 remains the most challenging level, the model significantly reduces the proportion of vocabulary and grammatical constructions beyond the target level while generally preserving meaning. Our model is freely available on $\href{this https URL}{HuggingFace}$. 

---
# MMedAgent-Plus: Scaling Multimodal Medical Agents across Modalities, Tasks and Workflows 

**Authors**: Ruoxin Wang, Yuanting Pan, Binxu Li, Ruiyang Ji, Yaqian Chen, Jiayuan Ding, Qifan Wang, Lifu Huang, Mingjie Li, Yu Zhang, Ehsan Adeli, Kilian Pohl, Zhe Xu, Zihao Lin, Haoyu Dong, Yixin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27233)  

**Abstract**: Medical workflows often require multiple interdependent analyses, but existing specialist and generalist AI models are not designed to coordinate them. We introduce MMedAgent-Plus, a multimodal medical tool-use agent that combines a vision-language planner with 18 leading specialist models spanning 12 tasks and 8 modalities. Through a unified tool interface, the planner selects specialists, executes them in sequence, and synthesizes responses grounded in their outputs. The planner is instruction-tuned on 208,928 samples from 32 public medical benchmarks, covering single- and multi-tool interactions across single and multiple rounds. We separately evaluate tool orchestration, response synthesis, and end-to-end execution on held-out samples. MMedAgent-Plus achieves tool-call accuracies of 91.0%, 98.1%, and 80.7% in single-tool multi-round, multi-tool multi-round, and multi-tool single-round settings, respectively. Given correct tool outputs, MMedAgent-Plus achieves scores of 14.4-14.8 out of 15 for faithfulness, completeness, and conciseness across 600 evaluation samples. In end-to-end execution without pre-supplied tool outputs, MMedAgent-Plus correctly invokes the required specialist in 90.6% of 350 evaluation cases spanning seven tasks. Its final-answer accuracy reaches 0.98, matching the dedicated leading specialist and exceeding the strongest baseline (0.28). 

---
# Which Agent's Skill Should Change? Group-Relative Credit Assignment for Evolving Multi-Agent Teams 

**Authors**: Yanan Ma, Yang Wang, David Le, Chenhan Yuan, Yizheng Sun, Guangwei Zhang, Chenghao Xiao, Chenghua Lin, Imon Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2610.27227)  

**Abstract**: Cooperative language-model agents can improve without weight updates by revising reusable skills, but only if a shared team reward is traced to the responsible agent and skill. Existing methods credit the skills present in each trajectory, yet presence is not responsibility: a harmful skill active in successful runs still earns positive credit. We present \textbf{GRSE}, \textbf{G}roup-\textbf{R}elative \textbf{S}kill \textbf{E}volution, which assigns behavioral credit by comparison: inspired by GRPO, it executes each question several times under the same skill libraries and contrasts the executions to credit agent--skill pairs, with structured diagnosis of failures. On HotpotQA and LoCoMo with two open-weight actors, GRSE outperforms MASkills, the closest prior method, on HotpotQA and every LoCoMo category except open-domain, with the largest gains on multi-hop questions ($+6.7$ and $+3.2$ F1). Its gains rest on better-grounded retrieval rather than answers given without evidence, and its learned skills transfer between the multi-hop tasks. GRSE is robust to corrupted libraries: starting from planted harmful skills, it credits every planted skill with the correct sign and recovers the score of a clean library ($+6.3$ points), whereas MASkills misattributes two and falls below its starting point. 

---
# LoWAgent: A Low-Resource Multilingual Benchmark for Web Agents 

**Authors**: Ocean Monjur, Daud Ahmad Nisar, Shahriar Kabir Nahin, Tadesse Destaw Belay, Sushruti Mishra, Israel Abebe Azime, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2610.27205)  

**Abstract**: Web agents are typically evaluated on English-language interfaces, leaving their ability to operate across the world's multilingual websites largely unexplored. This gap is especially significant for low-resource languages, where standardized resources for evaluating web agents remain limited. To address this gap, we introduce Low-Resource Web Agent Benchmark (LoWAgent), a benchmark for evaluating web agents across different low-resource languages (LRLs) in comparison to the default, English. LoWAgent evaluates web agents across English and four LRLs: Hindi, Bengali, Urdu, and Amharic. The benchmark comprises of six fully localized websites, instantiated across five languages to yield 3,000 task instances. We evaluate three closed-source and five open-source models across all five languages on all six websites. Our findings show that both closed-source and open-source models experience performance degradation for LRLs compared to English, with substantially greater performance drops observed among the open-source model agents. We further evaluate the robustness of the models on realistic interface disruptions, such as cookie notices and advertisement pop-ups, and find that these further degrade agentic performance. Finally, we show that post-training on a small subset of LoWAgent traces can significantly improve low-resource language agentic task performance, demonstrating the usefulness of our benchmark. 

---
# The Culture Within: Towards Mechanistic Understanding of Cultural Reasoning in LLMs 

**Authors**: Samee Arif, Furkan Danisman, Angana Borah, Honglak Lee, Dehan Kong, Lajanugen Logeswaran, Moontae Lee, Zhijing Jin, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2610.27202)  

**Abstract**: Large Language Models (LLMs) increasingly operate across culturally diverse settings, yet existing evaluations largely measure cultural behavior at the output level. We introduce CultureLens, a reference-grounded benchmark and mechanistic study of how LLMs use cultural information. CultureLens contains 9,261 scenarios across 75 countries and systematically varies country, value, true-norm, and contradictory-norm evidence. Across Llama-3.1-8B, Gemma-3-12B, and Qwen-3.5-27B, stating a contradictory norm lowers aligned answers by 32-47 percentage points, even when the model answers correctly from the country alone. Mechanistically, models default to Anglophone norms when no country is given. The failure under a contradictory norm is not one of missing knowledge, as blocking the norm inside the model restores the correct answer in most cases. Models accept a stated norm according to how it is presented and not whether it is true, which leaves them most exposed to errors in retrieved content. Supervised fine-tuning with contradictory-norm examples raises aligned answers under a contradictory norm to 85-93%, including on unseen countries, at a small cost under a true norm. 

---
# Preference-Aware Pareto Steering for Multi-Task Speech LLMs 

**Authors**: Liuyuan Jiang, Vishal Sunder, Samuel Thomas, Brian Kingsbury, Jeff Kuo, Lisha Chen, George Saon  

**Link**: [PDF](https://arxiv.org/pdf/2610.27200)  

**Abstract**: Speech large language models (LLMs) perform multiple tasks, including automatic speech recognition (ASR), timestamp prediction (TS), and speech translation (ST). Joint training of speech LLMs creates task trade-offs, yet standard training with fixed loss weights does not generally control the resulting trade-off according to deployment preferences. To address this, we propose Endpoint-Anchored Preference-Aware Sampling Uniformly along the Pareto Front (EA-SURF), an a priori method that adapts scalarization weights so that the resulting task trade-off reflects the desired preference. EA-SURF retains the standard weighted-loss training pipeline, tracking the current model's performance relative to each task-specific reference and adapting the scalarization weights as training proceeds. We theoretically show that EA-SURF can match a desired preference, with the update becoming stationary at the target under checkable regularity conditions. Across six three-objective speech LLM requests, EA-SURF empirically converges toward the requested targets and increases endpoint-inclusive normalized hypervolume by 17.8% over fixed loss weighting. 

---
# On the Fragility of Tasks under Composition: Implicit Budgeting in Language Models 

**Authors**: Chenpan Li, Thies Brockmoeller, Stefanie Jegelka, Sharut Gupta, Nikolaos Karalias  

**Link**: [PDF](https://arxiv.org/pdf/2610.27190)  

**Abstract**: Modern language models are increasingly asked to handle several tasks within a single prompt. Doing so requires solving each task correctly while integrating the resulting answers into a coherent response. In this paper, we identify a fundamental limitation in this process, where models make substantially more errors when tasks are composed, even when they can solve every constituent task reliably in isolation. This degradation cannot be fully explained by the accumulation of independent errors. Across both open-weight models and the frontier model Claude Opus 5, our findings instead point to an implicit thinking budget underlying this effect. In particular, under composition, models use substantially fewer tokens per task than in isolation, even when the full sequence remains well within the context window. Tasks that require longer solution traces in isolation are consequently more $\textit{fragile}$ under composition, i.e., models make more errors on prompts containing these tasks. In isolation, non-fragile tasks receive more reasoning than they need and waste compute. However, by grouping tasks according to their fragility, the implicit budgeting can be manipulated to reduce token usage by $43$\% across a range of math and linguistic tasks while also preserving accuracy. 

---
# SITO: Sifting a Commercial Text-Safety API for Missed Harm and False Flags in English, Ukrainian and Surzhyk 

**Authors**: Andrii Tuzhykov  

**Link**: [PDF](https://arxiv.org/pdf/2610.27178)  

**Abstract**: Platforms that host young users screen messages with commercial text-safety APIs, but independent evidence on how often such systems miss harm or flag harmless text is scarce, especially outside English, and the documentation of the one studied here leaves its false-positive rate unquantified. We evaluated one such API, Tuteliq Detect as deployed under the release identifier tuteliq-detect-v5.20261011 (not the separately published v6 model), on six harm categories: authorised push-payment fraud, money-mule recruitment, social engineering, romance scams, gambling harm and bullying. We built SITO (Synthetic Independent Text-safety benchmark with Overflagging tests; sito is Ukrainian for 'sieve'), 2,332 synthetic cases in English, Ukrainian and mixed Ukrainian, with each harmful case matched by hard harmless ones: same-topic messages, minimal-edit contrasts, warnings and retrospective accounts. On English text the system flagged 84.9-94.1% of harmful cases and 0-4.3% of harmless ones per category. It missed third-party reports of ongoing harm far more often than perpetrators' messages (63.6% against 95.1% recall, all languages), when sent without the settings the API offers for reports of harm. Ukrainian and mixed-language input was detected at least as often as English, recall was lower on text from a held-out generator family, so headline recall may be optimistic, and 3.2% of cases received a different decision in repeated runs. A platform should measure false positives on its own traffic and check how bystander reports are handled. The datasets, generation prompts and code are released. 

---
# GRID: Modeling Health Influencer Discourse as Argument Networks for Credibility Assessment 

**Authors**: Nishanth Sridhar Nakshatri, Eylon Caplan, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2610.27149)  

**Abstract**: Health-related discussions on social media are increasingly shaped by influencers who offer diverse, and often competing perspectives on the same issue. Yet, such content is typically evaluated in isolation, without considering the broader discourse in which it appears. We introduce GRID (Grounded Reasoning over Interconnected Discourse), a framework for evaluating influencer content through collective inference over the surrounding discourse. GRID represents each video as an argumentation graph connecting claims to its central message, or takeaway, links related claims and takeaways across videos, grounds claims in scientific evidence, and jointly infers credibility over the resulting network. We evaluate GRID on an end-to-end health-related content validation task using 569 influencer videos grounded in expert assessments. Our results show that accounting for the broader discourse improves influencer content validation. Finally, we use GRID-inferred credibility to demonstrate how argumentative strategies and moral framing differ between low- and high-credibility content. 

---
# ICLHerd: Herding-Based Data Selection for LLMs with In-Context Learning as a Proxy 

**Authors**: Yuqiao Wen, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2610.27142)  

**Abstract**: Data selection plays a central role in modern large language model training, as the quality of training data strongly influences downstream performance. A common approach is to curate a development set that encodes the desired skills (e.g., reasoning or multilinguality) and select data that are similar to them. Existing methods compute similarity using cosine distance over either the model's hidden states or gradients. However, we observe that this max-similarity strategy induces a mode-seeking bias: the selected data clusters tightly around the development samples. To address this, we propose ICLHerd, a data selection framework that leverages the von Mises-Fisher kernel to balance mode-seeking with mode-averaging. We further introduce in-context learning as a validation proxy, enabling efficient hyperparameter search over the balance trade-off. Results show that ICLHerd outperforms existing LLM data selection algorithms, highlighting the importance of balancing mode-seeking and mode-averaging. 

---
# Joint Constraint Aggregation for Qualifier-Sensitive Retrieval-Augmented QA 

**Authors**: Jennifer Hsia, Zachary C. Lipton, Aarti Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.27114)  

**Abstract**: Retrieval-augmented QA can fail even when the correct answer is in the retrieved evidence. This can happen when a competing answer satisfies some, but not all, of the question's requirements and is mentioned frequently in the retrieved documents, which can make it appear to be the best-supported answer. To test whether this retrieval skew actually affects the decision, we manipulate it directly: increasing the prevalence of passages that favor the competing answer makes the model more likely to select it. This points to a decision problem that candidate-level support alone cannot resolve. We therefore introduce Joint Constraint Aggregation (JCA), which compares candidates against the same question-derived requirements using the retrieved evidence. Across three evaluation settings and three instruction-tuned backbones, JCA improves overall accuracy, and its advantage grows when retrieval favors a competing answer. Together, these results suggest that when partial matches and retrieval prominence pull models toward a competing answer, comparing candidates against the question's full requirements can make selection more reliable. 

---
# Discrete Neural Interlingua for Ontology-Agnostic EHR Modeling 

**Authors**: Tong Ding, Caiwei Tian, Dongmin Bang, Ming Y. Lu, Sophia J. Wagner, Andrew Zhang, Rowland Pettit, Long Phi Le, Faisal Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2610.27047)  

**Abstract**: EHR foundation models promise scalable clinical prediction across health systems, yet their deployment is constrained by a tokenization bottleneck. The same clinical concept can appear as different ontology codes, local identifiers, synonyms, or free-text descriptions, which standard EHR tokenizers treat as unrelated symbols. A model trained on MIMIC-III with a closed-vocabulary tokenizer can use only 13% of eICU events. We introduce MEDIATOR, a discrete ontology-agnostic tokenizer that maps any clinical concept expressed as text into a shared vector-quantized codebook, turning EHR harmonization into open-vocabulary tokenization without manual mappings or a fixed vocabulary. Trained with synonym supervision and guided by enriched concept descriptions, code co-occurrence, and relation graphs, MEDIATOR collapses equivalent expressions while organizing distinct concepts in a medically meaningful discrete space. Used as the token interface of an EHR Transformer, MEDIATOR improves mean AUPRC over the strongest prior tokenizer by 4.0 points (8% relative) on 20 MIMIC-III tasks and by 8.6 points (29% relative) in zero-shot transfer to 14 eICU tasks, where it uses all events. The transfer gain combines coverage and representation, as MEDIATOR remains the best tokenizer under identical inputs (+1.0 and +1.9 points on operational and subcohort-mortality tasks). Same-encoder controls show that the text encoder drives in-domain accuracy, while the learned objectives of MEDIATOR drive transfer robustness and entity-linking quality. MEDIATOR also enables transferable attribution across hospitals. 

---
# Cross-Lingual Reusability of LLM Pipelines: Financial Extraction, Multilingual Reasoning, and Semantic Screening 

**Authors**: Joong Ho, Choi, Sherine George, Jiayang Zhao, Jayesh Rathi, Ashwin Kandath, Ananya Sane, Vishvak Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2610.27043)  

**Abstract**: Financial institutions increasingly deploy large language models (LLMs) in multilingual markets, raising a practical question: can pipelines developed and validated in English be reused across languages, or must they be rebuilt per locale? We test cross-lingual reusability along three dimensions--accuracy, reasoning-effort transfer, and behavioural stability--through three complementary tasks. Task 1 (cheque extraction) and Task 2 (Mintaka QA with pinned evidence across nine languages and nine model-effort configurations) test whether accuracy and effort settings transfer; Task 3 (FinanceBench pilot, 50 items over SEC 10-K filings in four languages) validates whether these patterns generalise to financial content. Accuracy reusability is broadly confirmed: Gemini 3.5 Flash achieves 95.2-97.6% extraction accuracy across prompt languages, and FinanceBench per-language means span 5.8pp (77.1-82.9%), though the language-difficulty ranking reverses between domains (Hindi is worst on Mintaka, best on FinanceBench at 82.9%). Effort reusability is domain-dependent: higher reasoning effort yields marginal gains on general-knowledge QA but up to 8pp on financial content, and effort configurations that help in English help proportionally in other languages. Behavioural reusability holds: citation-set overlap with English averages 77-79% Jaccard, with no single language consistently worse. Together, these results support qualified reuse of English-built pipelines, but per-domain, per-language validation remains necessary--neither semantic similarity nor self-reported confidence reliably predicts cross-lingual accuracy. 

---
# Fluent but Fallible: Benchmarking Hallucination and Factual Reliability in Large Language Models 

**Authors**: Osama Shabih, Dayam Nadeem, Shahab Saquib Sohail, Mohammad Nadeem  

**Link**: [PDF](https://arxiv.org/pdf/2610.27034)  

**Abstract**: Large Language Models (LLMs) have significantly advanced natural language processing and artificial intelligence applications through their capabilities in reasoning, text generation and knowledge retrieval. However, LLMs are still vulnerable to two important challenges: hallucination, generating factually incorrect or fabricated content, and response deviation, meaning systematic deviations from ground-truth responses. These limitations raise serious concerns regarding reliability and trustworthiness in academic and research environments. This study offers a thorough comparative analysis using a variety of metrics of ten advanced LLMs ChatGPT, Gemini, Claude~3.5~Sonnet, LLaMA~3, Mistral, Grok, Perplexity, DeepSeek~V3, Falcon, and Gemma based on a benchmark consisting of 70 carefully designed academic prompts across four complexity levels: Basic, Explanation, Advanced, and Tricky, evaluated in February 2025 on specific model versions. Each response was independently evaluated by three reviewers using five metrics: Accuracy, Precision, Recall, Response Deviation Index, and Hallucination Index. Perplexity and DeepSeek~V3 achieved the highest overall accuracy, while Gemini demonstrated strong recall and a well-balanced performance profile across all five metrics. LLaMA~3 showed strong resistance against hallucinations through a conservative method of generation. Advanced-level prompts proved most challenging across all models, while Tricky prompts were handled more reliably owing to their binary true/false structure. Falcon achieved perfect precision and zero hallucinations, but at the cost of the lowest recall and the highest response deviation. The results show a fundamental trade-off between response completeness and factual reliability, intensified by prompt complexity. 

---
# Decomposed Relation Extraction from Aviation Accident Narratives: Evaluating Transformer-Based Safety Knowledge Discovery under Relational Sparsity 

**Authors**: Aziida Nanyonga, Mustafa Tahsin Yilmaz, Saad Ezzini, Ugur Turhan, Graham Wild  

**Link**: [PDF](https://arxiv.org/pdf/2610.27013)  

**Abstract**: Relation extraction from aviation accident narratives is challenging because genuine safety relations are sparse among possible concept pairs. This study evaluates a decomposed framework separating relation classification from concept identification. A benchmark of 300 National Transportation Safety Board (NTSB) narratives containing 2,524 unique concept-text nodes was expanded into 21,038 ordered candidate pairs, with only 248 (1.18%) representing RESULTS_IN or CONTRIBUTES_TO relations. A TF-IDF Linear Support Vector Machine (SVM) and weighted RoBERTa classifier were developed using an independent graph-annotated resource with hard-negative sampling and frozen before sealed evaluation. Development Macro-F1 scores of 0.4738 (SVM) and 0.4809 (RoBERTa) declined to 0.2741 and 0.2420 under exhaustive evaluation, with Matthews correlation coefficients of 0.0128 and 0.0197. False positives dominated: the SVM recovered 96 reference relations but generated 7,018 false positives, while RoBERTa recovered 125 but generated 9,824. Post-test diagnostics revealed overlapping RoBERTa confidence distributions and false positives even within valid relation-category directions. Complementary end-to-end SmolLM2 evaluation identified missed and unsupported relations alongside malformed outputs. These findings demonstrate that known concept endpoints do not guarantee reliable relation extraction and that sampled-negative development performance may poorly reflect exhaustive evaluation. Controlling false positives and managing candidate-space sparsity remain essential for aviation safety knowledge extraction. 

---
# Evaluating Tibetan Speech Recognition Across Dialects and Adaptation Regimes: A Multi-Granularity Analysis 

**Authors**: Jacob Moore, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.27002)  

**Abstract**: Tibetan speech recognition is usually evaluated with character or syllable error rates because the writing system does not mark word boundaries. These metrics can rank systems differently and respond differently to adaptation failures. We combine character error rate, syllable error rate, and segmented word error rate in one evaluation framework. We use dictionary-based, neural, and language-model-based segmentation and compare their boundaries with expert annotations. We evaluate five ASR architectures on Lhasa Tibetan, then test zero-shot transfer and target-corpus adaptation on five additional corpus subsets. We also vary LoRA rank and training-data size to examine adaptation failure. The results show that a lower character error does not always lead to a lower syllable or segmented-word error. These ranking differences persist across corpora. Metric correlations also weaken in some unsuccessful adaptation settings. Higher LoRA ranks improve syllable and segmented-word scores but remain worse than full fine-tuning in the tested range. A standardized phonological representation provides another view of spelling errors. These findings support evaluation at several linguistic levels and motivate better segmentation and adaptation methods for low-resource ASR. 

---
# How to get the best and fairest performance out of your ASR system using activation steering 

**Authors**: Felix Herron, Alex Colagrande, Solange Rossato, François Portet, Alexandre Allauzen, Chuyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.26994)  

**Abstract**: Activation steering for ASR systems is an underexplored technique in improving both ASR performance and demographic fairness. We investigate two forms of activation steering: paired activation subtraction, and supervised optimization. We find the former to be ineffective at ameliorating ASR performance whatsoever, while the latter is quite effective at deriving steering vectors (SVs) that improve at once overall ASR performance and demographic fairness along such diverse axes as gender, age, and dialect. We also explore what types of data---from the most to the least personal---are necessary for calculating or training SVs. First, we show significant performance in performance from training SVs using an eclectic group of speakers from the same dataset. Furthermore, we find that vectors trained using acoustically similar speakers leads to ASR performances that are competitive or superior to vectors trained on the same speaker, or speakers from the same labeled demographic group. This finding shows that we can harness activation steering for few-shot adaptation given access to a speaker's demography, or zero-shot adaptation to unseen speakers without the need for any labels at all. 

---
# TRACE: Training-aware Robustness and Challenge Evaluation for Text-Vision Embedding Models 

**Authors**: Pume Tuchinda, Tinnakit Udsa, Peerat Limkonchotiwat  

**Link**: [PDF](https://arxiv.org/pdf/2610.26993)  

**Abstract**: Existing benchmarks for text-vision embedding models report aggregate scores that obscure whether performance comes from general capabilities, task-specific adaptation, or robustness to challenging tasks. We introduce TRACE, a Training-aware Robustness and Challenge Evaluation framework that reorganizes 168 tasks from various sources of datasets into three categories: General, Adaptability, and Challenge. Through controlled experiments on two text-vision embedding models, we show that combining general and adaptation data improves Adaptability performance by 7.67 and 7.09 points, respectively, while largely preserving General performance. However, Challenge performance remains below 5.0, with limited improvements from additional training data. Further analysis identifies multilingual and visual-document tasks as major weaknesses. Our findings demonstrate that data adaptation alone is insufficient to address these challenges, highlighting the need for methodological advances beyond data-centric improvements. 

---
# Think Before You Transcribe: Context-Steered Reasoning for Hotword-Aware Speech Recognition 

**Authors**: Qihao Wang, Hang Su, Fan Cui, Lichun Fan, Jian Luan, Yue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26986)  

**Abstract**: Contextual automatic speech recognition (ASR) improves rare-entity recognition by supplying candidate hotwords, but retrieved candidates may be incomplete or incorrect. A reliable system must therefore exploit a relevant candidate without allowing unsupported alternatives to override the acoustic evidence. We propose a context-steered reasoning framework that generates a short, audio-grounded caption before transcription. This task-specific representation lets the model assess candidate relevance before making a lexical decision. Supervised fine-tuning establishes the structured caption-to-transcription behavior, while targeted trajectory injection strengthens group relative policy optimization when difficult utterances yield only unsuccessful rollouts. Experiments across diverse context conditions show that the proposed method consistently improves both transcription quality and hotword recognition while remaining robust to irrelevant candidates. 

---
# Sharp Degree Bounds for Putinar Certificates in Terms of the Positivity Margin via Analytic Sums of Squares 

**Authors**: Yuxiang Zhao, Ting Gan  

**Link**: [PDF](https://arxiv.org/pdf/2610.26976)  

**Abstract**: We establish sharp degree bounds for Putinar sum-of-squares (SOS) certificates in terms of the positivity margin $\delta$. For any fixed Archimedean quadratic module with prescribed generators and any fixed bound on the degree of the objective, we prove a uniform certificate-degree upper bound of $O(\delta^{-1/2}\log^2(2/\delta))$ for normalized polynomial objectives. We show that this bound is sharp by constructing a matching lower-bound example in three variables, thereby establishing that the dependence on $\delta$ is optimal. These matching bounds also yield sharp worst-case convergence rates for the Lasserre hierarchy. 

---
# SZED at ArGuard Shared Tasks: Multi-Task Label-Attention and Cross-Modal Fusion for Arabic Hateful Meme Detection 

**Authors**: Selma Khelili, Zohra Abdeli  

**Link**: [PDF](https://arxiv.org/pdf/2610.26972)  

**Abstract**: We present SZED's submission to ArGuard 2026 Track A: Arabic hateful meme detection. The task has two parts: binary classification (A1) and fine-grained multi-label tagging (A2). For A1, we fuse MARBERTv2 and CLIP ViT-L/14 embeddings, comparing concatenation, self-attention, and ensemble heads. For A2, we use a multi-task label-attention architecture with per-class query vectors and an auxiliary head for cascade calibration. We reach macro-F1 = 0.7597 on A1 and 0.3142 on A2 (averaged over 10 active classes). We report reproducibility across runs, per-class error patterns, and the effect of cascade calibration on the long-tailed A2 label space. 

---
# LAMB: Latent Multi-Hypothesis Beliefs for Long-Horizon Large Language Agents 

**Authors**: Peien Ding, Hui Li, Yuhan Liu, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2610.26955)  

**Abstract**: Long-horizon language agents require compact memories that retain interaction evidence while preserving competing hypotheses about unobserved states. Although effective, many belief-state methods encode memory in discrete natural language, where repeated compression can omit evidence or prematurely discard plausible alternatives. We introduce LAMB, a recurrent framework for latent multi-hypothesis beliefs. A Common--Delta factorization combines a shared Common state for interaction evidence with weighted Delta components for competing hypotheses. Relative evidence from action-outcome predictions guides recurrent hypothesis updates without generating textual memory. A context-conditioned readout exposes hypotheses separately to the policy through a fixed-token belief prefix in the default configuration. The belief modules and policy are jointly trained using action imitation and auxiliary state, outcome, and observation-discrimination objectives. Across ALFWorld, BabyAI, ScienceWorld, and Crafter, LAMB matches or outperforms PABU, improving ALFWorld SR@30 and ScienceWorld success by 6.0 and 22.0 percentage points, respectively, and Crafter achievement progress by 26.37 points. On ALFWorld, it also reduces the redundant-action rate by 17.70 percentage points. These results support latent multi-hypothesis beliefs as compact memory for effective long-horizon decision making 

---
# A Multi-granularity Interaction Network for Relational Triple Extraction 

**Authors**: Xiaocheng Luo, Yanping Chen, Zaifu Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2610.26952)  

**Abstract**: Relational triple extraction identifies entity relations from raw sentences without depending on entity type information. The performance of traditional methods is seriously degenerated by ``unseen'' entities, because these methods can't build contextual dependencies relevant ``unseen'' entities. This paper addresses this challenge by proposing a multi-granularity interaction network that leverages word-word, entity-relation, and head-tail interactions to capture deep semantic dependencies. By incorporating multiple iterations, the multi-granularity interaction network is effective to refine the representations corresponding to head and tail entities. It enhances the global semantic features associated with relational triples, aiding in the remediation of triples that may have been overlooked or omitted in earlier steps. Strong and consistent performance is achieved by our method on two rearranged datasets, with a significant improvement in identifying relational triples, including those involving ``unseen'' named entities. For example, our model improves F1-score by 3.1%/20.5% over previous approaches on the NYT-R/WebNLG-R datasets in the Unseen metric. Code: this https URL. 

---
# The Bottleneck Is Retrieval, Not Reading: Diagnosing Evidence-Based Generation with RAG for Long Context Question Answering 

**Authors**: Debopriyo Banerjee, Owais Aijaz, Ahmer Tabassum, Zhuohan Xie, Preslav Nakov, Veselin Stoyanov  

**Link**: [PDF](https://arxiv.org/pdf/2610.26951)  

**Abstract**: Retrieval-augmented generation (RAG) is the dominant paradigm for grounding large language models in external corpora, but its similarity-driven, chunked pipeline has structural blind spots: it cannot anticipate which passages a question requires and discards document-level context that resolves cross-passage references. We introduce evidence-based generation (EBG), requiring models to return an answer with line-level citations, and a joint Ans+Evid metric that scores correctness and grounding simultaneously. We reformulate two corpora into evidence-grounded multi-document QA benchmarks- MultiHop-RAG and QASPER-MP, a multi-paper adaptation of QASPER with realistic short paper cues, and pair them with a controlled synthetic two-hop haystack and an evidence-augmented NoLiMa spanning 10K-1M characters. Across ten proprietary frontier and three open-weight LLMs, EBG isolates two RAG limitations that long-context models avoid: similarity retrieval surfaces any relevant chunk far more reliably than all required chunks for multi-fact queries, and chunking destroys document meta-information needed to resolve identity-bearing queries. EBG also exposes grounding failures invisible to answer-only evaluation, including incomplete citations in two-needle settings and a widening gap between Ans and Ans+Evid when context contradicts parametric knowledge. Data and code are available at this https URL. 

---
# Do We Need the Largest LLMs for Every Clinical Decision? 

**Authors**: Ali H. Lazem, William Teahan  

**Link**: [PDF](https://arxiv.org/pdf/2610.26937)  

**Abstract**: Large clinical language models often require multi-GPU infrastructure, limiting local deployment. We introduce CERTA, a decision-level statistical certificate that determines when a quantized model may substitute for its native BF16 reference and escalates decisions that cannot be certified to it. Across eleven instruction-tuned models, both general-purpose and domain-adapted, from three families (7-72B), we evaluated 2,500 clinical-verification decisions from de-identified patient contexts (2,484 patients), with patient-disjoint calibration and testing. In the higher-power operational analysis, round-to-nearest 8-bit quantization at BF16 storage served 99.8-100% of held-out decisions across all eleven models; formal finite-sample guarantees were assessed separately using the exact split-sample procedure. In the same operational analysis, the class-conditional 4-bit frontier reduced the amortized, BF16-equivalent deployment cost to 25.8% and 27.3% of the BF16 reference for Qwen2.5-72B and HuatuoGPT-o1-72B, respectively, with measured peak memory of 43.0 and 42.7 GB, below the nominal capacity of a single 48 GB accelerator. Deployment on that specific hardware was not directly tested. Under the exact split-sample certificate, which also accounts for configuration selection, these costs were 26.7% and 27.9%. On external disease named-entity benchmarks against human annotations, the frontier-selected NF4 configurations had a mean absolute F1 change of 0.045 on both datasets, with substantial model-specific variation. CERTA converts compression from a fixed precision choice into a risk-controlled deployment policy for this verification task. 

---
# AtomSD: Procedural Atom-Guided Self-Distillation for LLM Agents 

**Authors**: Bohan Yu, Pengfei Cao, Kai Tian, Chen Han, Shi-Yang Li, Chenxi Zhou, Junlin Liu, Cao Liu, Ke Zeng, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26926)  

**Abstract**: Large language models (LLMs) possess broad factual knowledge and strong reasoning capabilities, but solving complex tasks also requires procedural knowledge about how to act. Existing approaches often represent procedural knowledge through trajectories or bundled skills, limiting flexible reuse across interaction histories. This paper introduces Procedural Atoms, fine-grained and independently applicable units that represent Intent, When, Do, and Avoid guidance. Building on this representation, we propose AtomSD, an Atom-guided Self-Distillation framework in which the learner continually extracts Procedural Atoms from its own successful trajectories and distills them into the evolving policy through Atom-conditioned supervision and task-reward optimization. Comprehensive experiments show that AtomSD consistently outperforms strong reward-based and distillation methods, exceeding the state-of-the-art baseline by up to 9.31 points in average performance. Further analyses demonstrate that the proposed Procedural Atoms provide effective guidance, highlighting the value of atom-level procedural knowledge for policy learning. 

---
# Mixture of Phonetic Experts Based Low-Rank Adaptation of Conformer Models for Accented English Speech Recognition 

**Authors**: Santosh Dahal, Anmol Guragain, Tianchi Liu, Luis Fernando D'Haro, Kiran Chandra Dahal  

**Link**: [PDF](https://arxiv.org/pdf/2610.26888)  

**Abstract**: We present Mixture of Phonetic Experts based Low-Rank Adaptation (MoPE-LoRA), a parameter-efficient adaptation framework for accented English speech recognition. Recent mixture-of-experts approaches typically allocate experts per accent, implicitly treating accent identity as the primary axis of variation. In contrast, we organize adaptation along phonetic categories, motivated by the observation that accent differences are often expressed as systematic distortions of phoneme realization. MoPE-LoRA introduces a fixed set of six low-rank experts aligned with manner-of-articulation classes and performs frame-level routing through a hybrid mechanism that combines phoneme supervision with learned acoustic gating. Because the expert set is shared across accents, the model does not require accent-specific modules and naturally extends to previously unseen accents. Experiments on L2-ARCTIC show that MoPE-LoRA consistently improves over single LoRA and full fine-tuning. In zero-shot evaluation where one accent is held out during training, the proposed method yields a 12.3% relative improvement over single LoRA. These results indicate that phonetic-category-based decomposition offers a scalable and transferable alternative to accent-level adaptation. 

---
# Multimodal Ambivalence and Hesitancy Recognition with Bidirectional Audio--Text Cross-Attention 

**Authors**: TingTing Gu, JinCheng Luo, WenQiu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.26868)  

**Abstract**: Ambivalence and hesitancy (A/H) are subtle behavioural states that can be expressed through both what a person says and how it is spoken. This paper presents an audio--text framework for video-level A/H recognition. Full-video transcripts and speech waveforms are encoded using pre-trained multilingual BERT and wav2vec~2.0, respectively, and projected into a shared representation space. Two stacked bidirectional cross-attention blocks allow linguistic tokens to attend to acoustic frames and acoustic frames to attend to linguistic context without requiring explicit temporal alignment. The resulting sequences are aggregated through masked mean pooling and combined using concatenation, element-wise products, and absolute differences for binary classification. We evaluate the model on the subject-independent split of the Behavioural Ambivalence/Hesitancy (BAH) dataset. The proposed system achieves 72.19\% accuracy, 75.00\% precision, 81.13\% recall, 77.95\% A/H F1-score, and 70.16\% Macro-F1 on the test set. These results demonstrate the potential of jointly modelling linguistic and acoustic information for automatic A/H this http URL at:this https URL. 

---
# WaveCommitments: Private and Verifiable Speech-Agent Marketplaces 

**Authors**: Oguzhan Baser, Kaan Kale, Sriram Vishwanath, Sandeep Chinchali  

**Link**: [PDF](https://arxiv.org/pdf/2610.26867)  

**Abstract**: Modern speech systems are evolving into ecosystems of autonomous speech agents that speak together, negotiate, and trade assets such as embeddings, spectrograms, or raw data. In such markets, sellers cannot reveal assets for evaluation due to privacy constraints, and buyers cannot price what they cannot test. Thus, we open-source WaveCommitments, an end-to-end design that pairs confidential asset valuation via Data Shapley inside a trusted execution environment with multivariate polynomial commitments. It enables selective openings and tamper-evident provenance to prove that a delivered asset matches the valuation while hiding everything else. Our extensive experiments across three known datasets and various tasks with state-of-the-art encoders, including Whisper, reveal that we speed up the best benchmarks by up to 20x while reducing their proof sizes by up to 39x. We also systematically study how a speaker's value shifts across tasks and encoders, revealing agent-level market prices. 

---
# AutoTerm-SST: Adaptive Terminology Memory with Zero Session-Time Setup for Streaming Speech Translation 

**Authors**: Jiaxuan Luo, Siqi Ouyang, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.26865)  

**Abstract**: Retrieval-augmented speech translation improves rare terminology, but systems such as RASST still assume that users provide a glossary before each session. That assumption is inconvenient for live talks, while merging every domain into one growing glossary eventually places too many irrelevant terms in the same retrieval and prompt competition. We present AutoTerm-SST, an interactive system that removes session-time glossary upload and domain selection. It organizes a large offline terminology catalog into topic slices, matches recent streaming translation context to bilingual slice prototypes, and activates a compact multi-slice working set under fixed retrieval and prompt budgets. On five ACL and five ESO oncology talks, AutoTerm matches a known-domain reference in terminology accuracy (88.3% vs. 88.1%) and mWER-resegmented BLEU (45.09 vs. 45.19), while requiring no domain input. Compared with a one-million-entry merged glossary, it improves terminology accuracy by 3.9 points and BLEU by 2.30. Routing plus four-slice retrieval remains lightweight: the complete system runs at 0.406 RTF, and the vLLM host sustains 32 real-time sessions. The live demo exposes active slices and retrieved terminology as the topic changes, providing zero-setup terminology-aware streaming translation. 

---
# H2DF: Calibrated Target Bands for Controllable Language-Model Unlearning 

**Authors**: Jupiter Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.26864)  

**Abstract**: Gradient ascent (GA) is a simple language-model unlearning baseline, but it continues increasing forget loss after useful forgetting has occurred. This unbounded pressure can damage retained utility and make forgotten examples abnormally high-loss. We introduce H2DF, a calibrated target-band objective that applies GA-like pressure below a data-dependent lower target and suppresses, or reverses, that pressure after the target region is reached. The targets are estimated from matched non-member examples before training, so the base objective requires one forget-batch forward/backward pass per optimization step; our reported TOFU variant adds sparse retain replay every tenth step. We instantiate a one-sided objective for TOFU question answering and a two-sided objective for MUSE-News language modeling. Across three TOFU seeds, H2DF obtains a forget NLL of 1.528 with margin 0.049 and retain NLL of 1.165 with margin 0.047, matching GradDiff while using retain replay only every tenth step and showing substantially lower variability than GA. On MUSE-News, H2DF improves retention over GA and NPO and is competitive with SimNPO, but does not dominate it. Ablations show that the upper target reduces overshoot and that validation-based checkpoint selection is important. These results support calibrated target bands as a practical mechanism for controlling forgetting strength, while also exposing sensitivity to random seed and calibration quality. 

---
# Binary Discrete Diffusion: Candidate-Set Diffusion for Logic and Planning Tasks 

**Authors**: Rodrigo Casado Noguerales, Dimitri von Rütte, Bernhard Schölkopf, Thomas Hofmann, Aran Raoufi  

**Link**: [PDF](https://arxiv.org/pdf/2610.28543)  

**Abstract**: Discrete diffusion models iteratively denoise a noisy sequence of categorical data such as text, enabling any-order generation with self-correction and in-filling capabilities. This makes them promising candidates for reasoning, planning, and constrained-generation tasks like coding and math. Yet masked and uniform diffusion erase a token's identity in a single corruption event, so their noisy states cannot explicitly represent partially resolved uncertainty in which several alternatives remain plausible. We close this gap with Binary Discrete Diffusion, a family of continuous-time discrete-diffusion processes whose noisy state is a set of candidate tokens per position, represented as a binary membership vector. Binary diffusion changes candidate sets through additions and removals, allowing information to be lost and recovered gradually while the diffusion state itself represents partial uncertainty. On Sudoku-Extreme, Boxoban, and graph coloring, the symmetric formulation of binary diffusion consistently outperforms strong, well-tuned masked and uniform baselines under matched architectures and controlled training, sampling, and tuning budgets, with its largest gains on the hardest instances, while remaining competitive on language modeling (Text8, TinyStories, OpenWebText). These results support candidate sets as an effective intermediate representation for iterative reasoning and show that the corruption kernel can materially change the computation learned by the reverse model. 

---
# Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs 

**Authors**: Aur Shalev Merin  

**Link**: [PDF](https://arxiv.org/pdf/2610.28498)  

**Abstract**: Probing studies of latent reasoning often read a model's state where the answer begins and take a failed probe as evidence that a value is absent. We test this with a digit copied from the question, in the looped model Ouro-1.4B and the standard transformer Qwen2.5-1.5B. Comparing prompts that differ only in that digit shows it is present at the last prompt token, in a code shared across questions (readout about 0.9). A probe that sees one prompt at a time, the usual setting, barely reads it: 0.36 in Ouro and 0.19 in Qwen even after 5,249 training questions. The digit's change is small rather than crowded: random noise lands on equally busy directions, while a random word in its place moves the state far more and is readable. Since every varied token leaves a trace, a successful probe is ambiguous too. We give a protocol, including a synthetic null for generic mixing, for testing computed values against copies and mixtures of the inputs. In pre-registered studies of two-step problems it finds a small signal for the unwritten intermediate a+b just before Ouro writes the answer's last digit, and its sign follows the operation, though mixing alone produces much of that sign-following part. Holding that value at another problem's from the first loop on does not move the written digit toward the other problem's, while the same patch of the answer does: the value is readable but not used. 

---
# GPT-6-Astra Understands Touch and Outperform Vision--Tactile--Language--Action Models 

**Authors**: Ziyong Wang, Yupei Lin, Runxuan Huang, Liang Lin, Dongyu Zhang, Guangrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28457)  

**Abstract**: General-purpose multimodal models have shown promise as direct robot action policies, yet whether they grasp the physics underlying real-world interactions and can apply that understanding to guide action remains underexplored. We investigate this question using contact-rich manipulation as a representative testbed, examining tactile understanding as a concrete instance of this broader capability. Contact-rich manipulation requires interpreting tactile feedback and adjusting actions after obstruction. Existing learned tactile policies acquire these capabilities through specialized pretraining and per-task optimization, with substantial data and training costs. We present GPT-as-Tactile-Policy, which summarizes expert demonstrations into task guidance and tactile references for contact assessment and corrective action. GPT-6-Astra combines this context with multimodal observations and interaction history to generate control commands and adjust actions through contact feedback. Without additional robotics-specific pretraining or per-task policy fine-tuning, our method achieves 93.8\% mean success across five UniVTAC tasks. Overall performance is comparable to state-of-the-art policies, with higher success on selected tasks. On Insert Hole, tactile feedback improves success from 31\% to 99\%. Execution analysis further demonstrates contact-guided probing and correction, supporting manipulation progress through explicit expert experience and online feedback. 

---
# Toolflation: Seller Agents Distort Tool Markets Through Spontaneous Generative Engine Optimization 

**Authors**: Zhenwei Tang, Di Bai, Jintao Liu, Xidong Wu, Qianfeng Wen, Xinwu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.28454)  

**Abstract**: Agents are starting to sell to other agents. Seller agents write the listings of tools, services and products, and buyer agents read these listings when they choose what to use. We show that seller agents told only to be selected more often practice generative engine optimization on their own, and that this distorts the market for sellers and for buyers. We build tool markets from real data tools on Amazon and Instagram and run every tool on every question, so we know which tools can answer each question. Under all five production models we test, sellers rewrite their descriptions to claim more and more fields, including fields that their tools return for fewer than half of the targets. We call this process toolflation. Nothing in the market limits it, and how far it goes depends on the model, as the number of claims that contradict the sellers' own evidence differs by an order of magnitude between models. Toolflation shifts the buyers' choices between sellers, and the most capable tool loses share in seven of the ten combinations of seller model and platform. Under four models, the choices stay with capable tools and buyer agents keep their task success. Under the fifth, sellers copy one another until the descriptions no longer tell the tools apart. Buyer agents then pick tools by their position in the list, tools that answer almost nothing win a quarter to two fifths of the choices, and task success falls from 89% to about 30% on Amazon, close to random choice, and from 77% to 40% on Instagram. The sellers did only what their vendors asked, and the cost fell on buyers they never dealt with. We therefore argue that honesty toward third parties under competition is part of safety alignment, and that models should be tested as sellers before they are deployed as sellers. 

---
# MAS-Guard: Unified Guardrails from Individual Agents to Multi-Agent LLM Systems 

**Authors**: Mohammad Kachuee, Phu Mon Htut, Sheng Li, Nikolaos Pappas, Miguel Ballesteros Martinez, Yassine Benajiba, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2610.28397)  

**Abstract**: Agentic AI extends large language models (LLMs) from generating responses to planning and executing complex workflows, substantially expanding their utility while exposing applications to unsafe actions, information leakage, and adversarial manipulation. Multi-agent systems further expand these capabilities through specialization and collaboration, but distributed authority, heterogeneous roles, and opaque or potentially compromised participants make safeguarding even more challenging. Existing defenses address important individual threats but provide fragmented coverage across local execution, system-level interactions, and enforcement mechanisms. We introduce MAS-Guard, a unified framework that closes this gap for safeguarding single- and multi-agent AI applications containing both instrumentable and opaque agents. Within instrumentable agents, the proposed local Agent Guardrail (A-Guard) layer applies task-specific experts to assess agentic events. At the system level, developer-owned Agent Charters and a Control Plane combine deterministic structural checks of routes, agent capabilities, and message formats with semantic assessments of roles and message content. These complementary layers enable enforcement through a shared set of interventions: allow, reflection, confirmation, and block. Evaluated across six benchmarks covering diverse single- and multi-agent scenarios with three agent LLMs from different providers, MAS-Guard reduces the overall attack success rate by 71.5-86.3% relative to unguarded agents, while meeting the utility reference criterion on benign tasks. 

---
# CAPSafe: A Benchmark for Contextual Action and Plan Safety Reasoning in Embodied Agents 

**Authors**: Youmna Farag, Svetlana Stoyanchev, Simon Keizer, Mohan Li, Rama Doddipatla  

**Link**: [PDF](https://arxiv.org/pdf/2610.28381)  

**Abstract**: Reasoning about safety is essential for embodied agents powered by large language models (LLMs). We introduce the Contextual Action and Plan Safety benchmark (CAPSafe), designed to evaluate safety reasoning at two levels: (1) an action-level dataset that tests whether a single action is safe given its environmental context, and (2) a plan-level dataset that evaluates whether agents can generate safe long-horizon plans for complex hazardous household tasks. CAPSafe models safety conditions where hazards are situational (inferred from object properties), multiple hazards can co-occur within the same environment, and each is annotated with a relative severity level for fine-grained analysis. Evaluating nine closed and open-source LLMs reveals a substantial gap between hazard recognition and safe long-horizon planning. Several models achieve high accuracy in recognizing unsafe actions (exceeding 90% in the best cases), yet safe long-horizon planning remains substantially harder. Gemini-3-pro achieves the highest safe success rate (78.19%), while other models remain below 65%. These results highlight that robust safety reasoning over long action sequences remains an open challenge. 

---
# Disclosure Begets Disclosure: A Self-Reinforcing Privacy Loop in LLM Agents with Persistent Memory 

**Authors**: Masahiro Kaneko, Hiromi Arai, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2610.28315)  

**Abstract**: Large language model (LLM) agents increasingly use persistent memory to personalize responses, and personal attributes in this memory pose a privacy risk if exposed. We ask how consulting memory during dialogue changes what users go on to say, and thus what the memory comes to reveal. In multi-session simulations, a simulated user with 24 personal attributes talks with a support agent that does or does not consult memory, and in both settings attributes are inferred from the full accumulated record. Across four support models, memory raises response quality but also empirical leakage, the share of attribute values correctly inferred from the record, by a margin that widens over sessions. The cause is a loop in ordinary, non-adversarial assistance. Memory-conditioned responses carry earlier values forward, often as presuppositions, and users affirm them or leave them uncorrected, so that cues and the agent's own guesses become recorded values. The loop reinforces itself, and removing the agent's mentions of an attribute lowers its later disclosure about twice as much as single-turn effects predict. Users cannot leave it on their own, since correcting a value discloses the attribute, declining to discuss it shifts later predictions toward sensitive values, and incorrect values left in place become entrenched. The loop is best weakened at the agent's response. Instructing the agent not to state or presuppose attributes brings leakage to about the no-memory level while keeping most of the quality gains, whereas filtering or anonymizing what users say or what is stored gives up much of them. 

---
# Uncovering Audio-Text Alignment in Large Audio-Language Models through Cross-Modal Concept Neurons 

**Authors**: Tzu-Quan Lin, Chih-Kai Yang, Yi-Cheng Lin, Hao Tang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.28189)  

**Abstract**: While the language model backbone is pretrained on text alone, large audio-language models (LALMs) can, to a great extent, recognize acoustic attributes, such as emotion, speaker gender, and spoken language. However, it is unclear how the audio representation is integrated, if at all, with the other text representation before the model produces text. We hypothesize that there is a shared subspace where the audio and text representations are aligned. To test this hypothesis, we identify cross-modal neurons that are responsible for representing both audio and text. We find alignment between audio and text neurons to various degrees across three LALMs and four acoustic attribute types. Causal interventions, steering and deactivating the neurons, confirm that the cross-modal neurons are not just necessary for prediction but can also be manipulated. 

---
# Occupational Uniqueness and Wage Growth 

**Authors**: R. Vestrelli, A. Fronzetti Colladon  

**Link**: [PDF](https://arxiv.org/pdf/2610.28133)  

**Abstract**: Changes in occupational task content may have different economic implications depending on whether the tasks gained or lost are occupation-specific or already widespread across the economy. Existing measures of occupational task change largely abstract from this distinction, treating occupations as self-contained task bundles rather than accounting for how tasks are distributed across occupations. We develop a framework to measure a distinct dimension of occupational change, one that goes beyond task expertise or task content: task structure. Using this new framework that tracks tasks across occupations and over time, we separate changes in task quantity from changes in occupational uniqueness, defined as the distinctiveness of an occupation's task bundle relative to other occupations. Using a harmonized dataset of 196563 task statements from the Dictionary of Occupational Titles and O*NET spanning 1977-2018, we measure task diffusion, occupational uniqueness, and distinct dimensions of task addition and removal. Occupations that become less unique are also associated with lower wage growth, and the wage associations of task additions and removals depend on how widely the affected tasks are performed elsewhere. Adding highly diffused tasks is associated with lower wage growth, as is removing occupation-specific tasks. At the economy-wide dimension, however, the pattern differs: the addition of new tasks is positively associated with wage growth, whereas the removal of tasks is negatively associated with it. Taken together, these results show that similar changes in the number of tasks performed by an occupation can reflect economically distinct processes. Incorporating the cross-occupational distribution of tasks, therefore, provides a richer account of occupational change and its relationship to the wage structure. 

---
# Contextual Candour: Role-Aware Graduated Disclosure for LLMs over Mixed-Sensitivity Contexts 

**Authors**: Devansh Srivastav, David Pape, Samira Abedini, Lea Schönherr  

**Link**: [PDF](https://arxiv.org/pdf/2610.28067)  

**Abstract**: LLMs increasingly operate over mixed-sensitivity contexts, where they must reason over information ranging from public to confidential while disclosing only what the user is authorised to receive. Existing approaches enforce access control as a binary decision: either fully answering or refusing, without regulating how much or what kind of information is revealed if access is granted, leading to over-refusal or unintended leakage. We introduce graduated disclosure, an access-control formulation in which the level of detail and sensitivity in a model's response is calibrated to the user's authorisation rather than treated as a grant-or-deny decision. We model graduated disclosure as a disclosure-aware generation problem and propose a reward-based optimisation framework that shapes responses toward level-appropriate abstractions while penalising unauthorised factual leakage, including paraphrases and approximations. We evaluate our approach on enterprise and clinical benchmarks and compare it with prompt-based, gating, and supervised baselines. Experiments show that our method substantially reduces information leakage while preserving correctness and utility. These results suggest that disclosure control must be integrated into the generation process itself to enable secure yet useful LLM deployments. 

---
# Poisoning-Based Backdoors Are Not Cryptographic 

**Authors**: Masahiro Kaneko  

**Link**: [PDF](https://arxiv.org/pdf/2610.28045)  

**Abstract**: To defend against backdoor attacks on neural networks, the defender must identify the verifier function that determines whether an input is a trigger, rather than rejecting individual triggers in isolation, since an attacker can generate new triggers satisfying the same verifier faster than they can be blocked. If a backdoor verifier could be installed with the asymmetry of a cryptographic authentication scheme, in which only the attacker can produce trigger inputs while the defender cannot reverse-engineer the verifier from observations, learning-based defense would be impossible in principle. We show that no such asymmetry can arise from poisoning-based installation, in which the only verifier-dependent input to training is a poisoned dataset. The argument is a reduction. A training procedure that installs a verifier from poisoned data, so that the backdoor fires on unseen triggers while benign behavior is preserved, is itself a learner for that verifier. Any verifier that is computationally hard to learn is therefore impossible to install through poisoning, and any verifier that poisoning can install is reconstructible by a white-box defender holding comparably many labeled trigger observations. A complementary argument shows that a secret key supplied to the model at inference is observable to the same defender. The result takes representative trigger observations as given and concerns when the backdoor fires rather than what it does. Experiments on instruction-tuned large language models confirm the dichotomy, with no configuration yielding a backdoor that is both effective for the attacker and irrecoverable for the defender. 

---
# From RL Papers to Framework Code: Inclusion and Review in Seven Open-Source Training Frameworks 

**Authors**: Taiheng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.28027)  

**Abstract**: Practitioners use reinforcement learning methods for language models through training frameworks, so a method that a framework includes is one configuration change away for its users. We measure how often this happens. We trace seven open-source frameworks for RL post-training through their commit histories and link them to 6,939 screened arXiv papers from January 2025 to June 2026. Of 1,598 papers that propose a general RL training method, 37 reached at least one framework. Across five quarterly cohorts of papers, the number included within 180 days stayed between 5 and 7 while the cohorts grew from 72 to 386 papers. Compared with papers from the same quarters, included papers more often trained policies of 30B parameters or more, at 41 against 10 percent, and more often changed a coefficient, clip, mask or weight in the existing loss, at 29 against 9 percent. Both gaps hold without papers from the companies that maintain the frameworks, while code release was equally common in both groups. The frameworks' issues and pull requests show that few methods are turned away in public. More than nine in ten of the general methods no framework included have no thread there that cites or names them, and many proposals that do appear stay open without a reply. Two further records, the frameworks' issues and the technical reports of 13 large open-weight models, show that users rarely report training failures in public and that the reports state 8 of the 13 widely included methods. We release the data as a tracker that links training failures to framework implementations and lists each framework's contribution rules and earlier proposals. 

---
# Beyond the Evidence Boundary: Behavioral Mirage Under Progressive Degradation in Medical VLMs 

**Authors**: Edric Castel C. Hao, Rez Samantha Z. Floresca  

**Link**: [PDF](https://arxiv.org/pdf/2610.28000)  

**Abstract**: VLMs for medical image interpretation are typically evaluated on clean inputs, despite the progressive evidence loss that can occur in high stakes clinical settings. Existing robustness studies show that performance can degrade while confidence remains high, but do not establish when a previously reliable model-case pair ceases to behave reliably. We introduce a baseline-anchored framework that traces baseline-correct cases through progressive image degradation to identify an empirical evidence boundary, and define mirage as continued commitment at this boundary, regardless of correctness. Across three VLMs on three medical datasets, only MedGemma provides a sufficiently large baseline-correct pool for trajectory analysis. On VQA-RAD, the only dataset in which the question text varies, language-only answerability is associated with baseline correctness (AUROC 0.677) and with commitment at the boundary (AUROC 0.653 to 0.670). MedGemma's post-boundary behavior is heterogeneous, including direct mirage, correct recovery, mirage dissolution, consistent abstention, and oscillation, with mirage rates of 16.0 to 100.0% across dataset-modality pairs. Despite this behavioral heterogeneity, residual-stream probes decode commit-versus-abstain behavior at the boundary with AUROC of 0.895 to 0.994 under nested cross-validation. These results show that progressive evidence loss can produce persistent commitment that output entropy and abstention behavior do not consistently reflect, motivating boundary-aware evaluation, option-order controls, and improved abstention mechanisms for reliable medical foundation models operating under partial observation. 

---
# Building Telecom Language Models with RSOD: Domain Data, Specialist Post-Training, Memory-Conscious Distillation, and Networking Evaluation 

**Authors**: Anil Srirangapatna Nagesh, Nitin Gupta, Anuj Taneja, Vivek Singh, Shane Eleniak, Theo Zhao, Moulidharan N, Shashank Keshetty  

**Link**: [PDF](https://arxiv.org/pdf/2610.27993)  

**Abstract**: Telecom language models must combine broad networking knowledge with specialist behavior across routing, broadband services, access technologies, and operational diagnostics while remaining practical to deploy as a single model. We present an end-to-end framework that builds source-grounded synthetic training data from structured networking taxonomies and expert graphs, materializes 212,559 routed training records across fourteen networking routes, trains six frozen specialist LoRA teachers over one shared continued-pretrained backbone, and consolidates their behavior into one deployable student adapter using Routed Specialist On-Policy Distillation (RSOD). In RSOD, the committed student generates each trajectory, the route-assigned specialist teacher-forces the exact native token sequence, compact teacher hidden states are reconstructed through the shared frozen output head, and only the student LoRA is updated. A controlled reconstruction validation shows mean absolute logit error of 7.26 x 10^-6 with 100% top-1 agreement between native and reconstructed teacher logits. On four RTX 6000 Ada GPUs, the completed 2,000-step run produced 16,000 trajectories with zero observed student-version or token- ID mismatches. Under the primary route-conditioned evaluation over 2,850 held-out questions, the raw Meta-Llama-3.1-8B-Instruct model scores 12.8661/25 micro (13.0172/25 macro), the combined routed six-teacher system scores 21.7641/25 micro, and the consolidated RSOD student, released as Llama-CalixCLM v1, reaches 21.7112/25 micro (21.8006/25 macro) on the same five dimension LLM-judge rubric. Under a separate neutral deployment-like prompt, the same student reaches 20.0028/25 micro (20.0778/25 macro). The consolidated student remains within 0.0529 micro-score points of the combined routed-teacher system while retaining a numerically validated and auditable train-serve execution path. 

---
# Lost in Recovery: Demystifying Recovery Behavior of Computer-Use Agents via Goal-Oriented Action Replay 

**Authors**: Byeonghun Bang, Sangwoo Park, Jinheon Baek, Kangsan Kim, Youngwan Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27954)  

**Abstract**: Recent advances in large (vision) language models have enabled computer-use agents that automate digital workflows via fine-grained keyboard and mouse operations. However, such agents remain fragile in long-horizon settings, where errors accumulate and executed actions leave unintended side effects, making the ability to recover from failures critical. Yet, existing remedies remain inadequate: environment resets are infeasible in proprietary settings, action replay is prohibitively expensive over long horizons, and API-based rollback is limited to web environments, leaving model-based recovery largely unexplored. To address this gap, we present ReTrace, a scalable evaluation framework for systematically studying recovery behavior in computer-use agents. We formalize Transition-Consistent Recovery: identical downstream actions from the recovered and target states must yield identical outcomes. We operationalize this criterion through goal-oriented action replay following controlled error injection. Using ReTrace, we present the first systematic study of autonomous, model-based recovery, revealing that (i) ideal recovery can unlock model's latent task completion capabilities, (ii) current computer-use agents fail to exploit their latent recovery capabilities, and (iii) accurate model-based recovery yields measurable end-to-end gains. Together, these findings establish recovery as a fundamental capability for building reliable computer-use agents. Our code and dataset is available at this https URL. 

---
# ACE-Compression: An Ability-Centric Evaluation of Compressed Large Language Models 

**Authors**: Jiahui Ning, Liang Chen, Yuchen Liang, Yi Zhang, Fenglei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2610.27913)  

**Abstract**: Model compression makes large language models deployable, but today's evaluation tells an incomplete story. Standard practice measures retained task performance and stops there. This misses a critical fact: compression perturbs the distributed knowledge embedded across billions of weights, and task scores alone cannot reveal which underlying abilities survive and which quietly collapse. To solve this question, we introduce ACE-Compression, an ability-centric evaluation framework. Instead of asking only "does the compressed model still answer correctly?", we ask "which of its core abilities remain intact?" Our framework probes four deployment-facing dimensions: procedural reasoning and constraint execution, knowledge grounding and truthfulness, social cognition and affective understanding, and persona fidelity and stylistic consistency. Across more than 35 compressed checkpoints from six open-weight LLMs-covering quantization and pruning-we find that compression redistributes ability profiles. Coarse, local, and surface-level signals often persist, while fine-grained, long-tail, and context-dependent abilities become fragile. Quantization consistently outperforms pruning in retention, but the practical meaning of that gap depends entirely on which ability an application requires. Based on our findings, deployers can choose compression methods by the ability profile their application needs, not just by a leaderboard average. And algorithm designers gain diagnostic signals to build compression methods that preserve what matters. We believe that ACE-Compression turns compression from a black-box accuracy drop into an interpretable intervention-one we can measure, understand, and ultimately improve. 

---
# Staged Continual Adaptation of Multimodal Foundation Models for Japanese Financial Documents 

**Authors**: Genshin Kakimoto, Atsushi Yanagisawa  

**Link**: [PDF](https://arxiv.org/pdf/2610.27904)  

**Abstract**: Staged post-training of multimodal foundation models chains vision-language alignment, reasoning distillation, and domain tuning, yet how each phase trades capabilities against the others remains under-characterised. Tracking an 8.4B-parameter model from an un-fine-tuned baseline through three training phases on Japanese financial documents (evaluated on five text-input benchmarks), we find the phases interact non-additively: math-only distillation overwrites the financial knowledge acquired earlier even as procedural math skill survives, and the domain phase repairs more than it installs. A skip-phase ablation is best for domain knowledge but forfeits numerical-reasoning gains, so per-phase contributions are not decomposable and the best checkpoint is task-dependent, with implications for checkpoint retention and scheduling rather than added stages. 

---
# From Tokens to Files: Multi-Granularity Detection and Localization of AI-Involved Code 

**Authors**: Yue Xu, Xiyang Wang, Hui Liu, Bolei He  

**Link**: [PDF](https://arxiv.org/pdf/2610.27863)  

**Abstract**: The increasing use of large language models for code generation calls for methods that distinguish purely human-written code from AI-involved code: files in which any part, however small, was generated or substantially modified by a model. Beyond flagging such files, we aim to localize the lines or blocks attributable to AI. Existing approaches compress a file into a single score, from surface stylistic features or a global code representation, which averages away the evidence when only a few blocks are AI-involved. We propose a framework that models AI authorship at the token level and predicts at the line, block, and file levels. A frozen code language model is equipped with two low-rank adapters, trained separately on machine-written and human-written files; their per-token log-likelihood ratio provides local evidence of AI authorship. Lightweight heads aggregate this evidence within lines and blocks, using the surrounding code context, and their local predictions are aggregated into a file-level decision. We train on mixed-authorship files built from human originals and their AI-generated or AI-modified counterparts, including continuation and fill-in-the-middle scenarios. On an internal test set of 11,773 AI-involved files across six generation regimes against 2,541 human files, our method reaches 72.4 macro F1 at a 1% FPR operating point. With no adaptation, it reaches 62.5 line-level F1 on HybridCodeAuthorship, an external human-AI co-writing benchmark with no training split, above the 56 its own paper reports as the best line-level result. On 19,445 human-written files from repositories created before 2022, the false-positive rate stays between 0.55% and 0.81% under the transported 1% threshold, so the operating point holds on real human code. 

---
# Give the Model a Finger: Developmentally Aligned Evaluation of Counting in Vision-Language Models 

**Authors**: Yingxin Jia, Hui Geng, Wenxin Yang, Mohan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27756)  

**Abstract**: Vision-language models (VLMs) are exact on small sets and drift into estimation on larger ones, a profile reminiscent of children before they master counting. Children cross the barrier with epistemic actions: pointing at each item while reciting number words, moving counted items aside, partitioning the array. We give VLMs the same actions as dumb image tools with no visual intelligence of their own in a training-free agent loop. Our central claim is about measurement, not intervention: a model that counts by marking leaves a normally hidden serial procedure on the image, and the developmental principles of counting turn those marks into an interpretable audit of serial visual reasoning: every tag is classified against point-level ground truth as on-target, duplicate, phantom, or omitted, the answer is checked against the number of tags, and order irrelevance is tested by forcing opposite traversal orders. Across 9 models, 682 images, and 7 harness conditions we find that (i) the exact-counting range is a model-specific signature spanning nearly two orders of magnitude (R80 from 3 to 150 dots); (ii) as an intervention the finger mostly fails: tags help 4 of 9 models on small-count photographs, hurt imprecise pointers, and rarely beat a verbal routine or a blind point list; what the loop buys is an auditable trace, not accuracy; (iii) as measurement it succeeds: a tag-perturbation probe shows that precise pointers read their answers off the tags, pointing precision tracks which models benefit, dense scenes fail by omission, and models rarely revise after seeing their own tags; and (iv) the capacities dissociate: models follow an instructed traversal order yet often change their answer when it reverses, and children's regular-over-scattered advantage appears only under the routine and survives lattices that defeat a rows-times-columns shortcut. 

---
# When Does Experience Memory Help Closed-Loop LLM Driving Agents? A Controlled Evaluation with Resampled Warmup Pools, Token-Matched Fillers, and Audited Baselines 

**Authors**: Yingxin Jia, Xuanyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.27714)  

**Abstract**: When does experience memory help a closed-loop LLM driving agent? Published gains cannot say, because the two factors we find to matter most, which warmup experiences the pool contains and what any injected tokens do to the prompt, are controlled in none of them. We answer with a controlled evaluation on one audited HighwayEnv testbed (1Hz, gpt-4o-mini for the decisive contrasts). We treat the warmup pool as a resampled random factor (ten disjoint pools) and add token-matched irrelevant-filler controls, oracle-certified diagnostics, caution-pricing metrics, and quantified audits of the baseline ports. The probe suite spans flat-text retrieval (a DiLu port), a single-frame grid port (RESPOND-inspired), our episodic temporal-graph memory, a retrieval-free consolidated rulebook, and a flat K-step observation-history baseline. Four findings follow, each a property of this testbed. (1) Headline memory gains belong to the warmup draw. The reported pool is the best of ten for both retrieval memories, which beat No-Memory in only one pool of ten (sign test p=0.011), and even the rulebook's apparent pool-immunity collapsed on five fresh draws. (2) No retrieval design tested exceeds token-matched filler. The collision-only merge ceiling is reached by a single 15-token caution instruction, and metrics that price caution erase the rulebook's lead. (3) Representational sophistication is nowhere robustly operative. (4) Baseline port quality silently inverts conclusions. On this testbed, a single-pool comparison without filler controls would have reported memory gains that these controls reverse. We release the testbed, audited ports, seeds, and caches. 

---
# PhonoBridge: Error-Aware Phoneme-Mediated EMG-to-Text Reconstruction 

**Authors**: Wen Qiu, Jiacheng Zhang, Xuanda Chen, Siqi Cai, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.27678)  

**Abstract**: Electromyography (EMG) provides a non-invasive means of decoding silently articulated speech into text, but accurate decoding remains challenging due to limited availability of paired EMG-text data. We introduce PhonoBridge, a two-stage framework comprising a channel-aware EMG-to-phoneme frontend and a large language model (LLM) backend for text reconstruction. The frontend is trained on paired EMG-text data that consists of channel-wise feature extraction, cross-channel fusion, and electrode embeddings. With phonemes as the intermediate representation, we expand the backend's training data by converting LibriSpeech transcripts into phoneme-text pairs without additional EMG recordings. To adapt the LLM to frontend noise, we introduce Error-Profile-Guided Augmentation (EPGA), which perturbs transcript-derived phoneme sequences according to error patterns estimated from frontend predictions, generating additional training pairs for the LLM. On the Open Vocabulary silent-speech dataset, PhonoBridge achieves 24.77% WER, outperforming the previously reported best of 28.27% on the same test split. 

---
# CovNav: Conformal Prediction-Guided Adaptive Inference for Object Goal Navigation 

**Authors**: Pengying Wu, Zhejia Wen, Kang Ding, Kangjie Zhou, Chunxuan Jiao, Wei Pan, Xiangli Nie, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27677)  

**Abstract**: In object goal navigation (ObjectNav), an embodied agent must find a target object (e.g., "find a bed") in a previously unseen environment, relying on spatial commonsense such as object co-occurrence. Large Language Models (LLMs) encode this commonsense, but deploying them on robots exposes a trade-off between reasoning quality and efficiency: cloud-based models incur latency and privacy risks, whereas small on-device LLMs are fast but prone to hallucinated spatial judgments. Hybrid reasoning LLMs offer both a fast non-thinking mode and a slow thinking mode, which raises a key question for embodied agents: when is slow thinking worth its cost? We propose CovNav, a training-free framework that answers this question with Conformal Prediction (CP), turning dual-process (System 1 / System 2) inference into an uncertainty-aware decision. Much like recent decision models such as Jev, CovNav casts each spatial query as a structured binary decision rather than free-form generation: a local LLM in fast mode judges whether the target is likely near an observed object, CP converts its token probabilities into a prediction set, and only ambiguous queries are escalated to slow chain-of-thought reasoning. The object-pair formulation keeps calibration tractable and supports open-set objects through off-the-shelf detectors. Experiments on Gibson and HM3D show that CovNav surpasses the strongest zero-shot baseline by 0.8 and 1.7 percentage points in success rate on HM3D v1 and v2, while invoking slow reasoning for only 12.4% of queries and retaining most of the gain of always-on thinking at a fraction of its latency. Real-robot experiments on a TurtleBot 4 further validate the approach. CovNav shows how calibrated uncertainty can decide when an embodied agent should think slowly, offering a principled form of adaptive test-time reasoning for LLM-based robots. 

---
# CUA-Recovery: Benchmarking Error Recovery in Long-Horizon Computer-Use Agents 

**Authors**: Xu Hu, Haoming Li, Yuxiang Wang, Jinxin Yang, Kangshuo Li, Licheng Wang, Dingyi Kang, Jiayu Song, Tianhao Wang, Zhanhua Hu, Xu Yang, Qiannan Li, Bingzhe Li, Feng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.27645)  

**Abstract**: As computer-use agents (CUAs) move to long-horizon workflows spanning multiple applications, undetected errors can silently corrupt persistent state and cause cascading failures in subsequent actions. Reliable error detection and recovery have become critical bottlenecks for building robust CUAs. However, existing benchmarks focus primarily on end-to-end task performance, providing limited insight into whether agents can detect and recover from errors during execution. The few benchmarks that evaluate recovery rely largely on manually injected errors, which may not reflect the types or horizons of failures induced by real agent policies. We introduce CUA-Recovery, a benchmark comprising test and training suites across multiple desktop applications. We construct it using ReRail, a framework for synthesizing long-horizon tasks and trajectories in a persistent desktop environment. ReRail composes MyPCBench tasks into new long-horizon workflows with executable ground truth, collects verifier-confirmed failures from multiple policies, and turns them into verified recovery trajectories. CUA-Recovery comprises a human-verified test suite of 2,250 erroneous states over 100 tasks and a training suite of 8.6K trajectories over 377 tasks. Evaluation on the test suite shows that agents struggle both to recognize inherited errors and to complete the task. Across six agents and six takeover depths, Error Awareness Rate (EAR) reaches at most 44.0% and Pass@3 at most 25.5%. After fine-tuning Qwen3.5-35B-A3B on the training suite, its average EAR rises from 19.9% to 35.2% on held-out recovery states, and its Rubric Score goes from 5.0% to 16.1%. It also gains 15.3 points on MyPCBench and 10.8 points on Odysseys for Pass@3, showing that the training trajectories help both with recovery and with completing tasks from a clean start. Our code is available at this https URL. 

---
# BCT: Bayes-Consistent Tempering for Flow Language Models 

**Authors**: Daisuke Oba, Hiroki Furuta, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2610.27641)  

**Abstract**: Temperature scaling is typically applied as a post-hoc transformation of model probabilities. In posterior-induced flow language models, this creates two mismatches. First, posterior tempering implicitly changes both the endpoint distribution and the distribution of intermediate noisy states, while the denoiser remains trained on the original states. Second, the tempered posterior weights are combined with transport dynamics derived for the untempered flow. These mismatches are reflected empirically: lowering temperature improves GSM8K performance much less reliably in flow LMs than in discrete diffusion LMs. We introduce Bayes-Consistent Tempering (BCT), which makes temperature explicit in both the distributions and dynamics of the generative flow. Its learned realization trains the denoiser on tempered intermediate states and tempers the endpoint distribution using the corresponding tempered dynamics. We also derive training-free variants for conventionally trained checkpoints. Across hyperspherical flow LMs with two denoiser architectures and a von Mises--Fisher flow LM, BCT improves GSM8K generation across broad temperature ranges and achieves the strongest tempering performance. The gains also extend to Sudoku, demonstrating benefits on tightly constrained generation. At very low temperatures, the learned realization is also markedly more robust than the training-free variants, supporting temperature as a property of the generative flow across training and inference rather than a post-hoc probability adjustment. Code and model weights are available at this https URL . 

---
# Auditing the Product, Not the Recursion: The Evaluation Evidence Behind Recursive Self-Improvement Claims 

**Authors**: Zihan Chen, Di Zhu, Lei Nico Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.27610)  

**Abstract**: Recursive self-improvement (RSI) promises AI systems, such as self-evolving LLM agents, that become better at producing further improvements. Evaluating this promise requires separating product claims about the final system from process claims about the improvement process itself. Existing surveys specify the evidence these claims require, but how often studies report that evidence remains unmeasured. We examine this question through full-text coding of 114 eligible studies in a reproducible arXiv search frame frozen in October 2026, including 90 that describe their own systems as RSI. The codebook links claims to evaluation designs. Independent human coding, third-reader adjudication, and a separate blind coding of improver comparisons assess coding reliability. Among the 90 RSI studies, 90% report product evaluation beyond development tasks under a broad criterion that includes partial or unstated separation, while 60% report evaluation on tasks unused for selection. Only 22% audit two or more reruns of the full loop. In a human-coded random sample, the RSI studies show the same contrast: 45% evaluate on tasks unused for selection and 14% audit two or more reruns. Of the 26 studies claiming a better improvement procedure, 23 report a related comparison, but only 4 compare earlier and later procedures from identical starting artifacts at a matched budget. Across these results and a comparison set of established works, evidence for repeatability and improved procedures remains less developed than product evaluation. Three propositions explain how selection effects, shared evaluator errors, and changes in system state limit what product evaluations can establish. Together, the findings support a claim-indexed reporting standard that connects each claim to the evaluation design needed to test it. 

---
# When LLMs Manipulate: Measuring Behavioral, Social, and Temporal Effects 

**Authors**: Sadia Asif, Israel Antonio Rosales Laguan, Shumaila Asif, Haris Khan, Muneeb Asif, Muhammad Zeeshan Karamat  

**Link**: [PDF](https://arxiv.org/pdf/2610.27573)  

**Abstract**: Large language models (LLMs) can influence user trust and decision-making through subtle linguistic cues, yet the behavioral effects of manipulative outputs remain poorly quantified. We introduce a behavioral computing framework for measuring \textit{behavioral vulnerability}, defined as users' susceptibility to manipulative AI responses. Building on DarkPatterns-LLM benchmark, we generate controlled manipulative stimuli using Manipulation Gradient Descent (MGD), targeting biases associated with authority, urgency, and social proof. We analyze participant responses using the Multi-Stakeholder Impact Assessment Network (MSIAN) to model cross-user influence and the Temporal Harm Propagation Model (THPM) to characterize the persistence of behavioral effects over time. In a study involving 120 participants, manipulative framing produced significant changes in trust calibration and compliance ($p < 0.01$), with effects also propagating across social contexts. These findings highlight the need to extend LLM safety evaluation beyond model outputs to measurable behavioral consequences and provide a computational framework for quantifying individual, social, and temporal risks arising from manipulative AI interactions. 

---
# AnomalyJev: Towards Evaluating Image Anomaly Explanations via Claim-Level Verification 

**Authors**: Qizhou Wang, Jiawen Zhu, Guansong Pang, Christopher Leckie  

**Link**: [PDF](https://arxiv.org/pdf/2610.27565)  

**Abstract**: Recent multimodal LLM (MLLM)-based anomaly detection (AD) methods can generate textual explanations that help interpret their predictions. However, reliably and efficiently evaluating these explanations remains challenging. While LLM judges can assess explanations through structured rubrics, they often require substantial token generation, making evaluation costly and difficult to scale. We instead propose to anchor evaluation in the available ground truth by extracting verifiable claims from each explanation and verifying them against the corresponding annotations. We instantiate this idea as AnomalyJev, a lightweight framework that extracts verifiable claims from contextualised representations using ground-truth-defined queries and pointer based heads, without token generation. The claims are then verified against the ground truth to score faithfulness and completeness. For evaluation, we construct a human-annotated set of 640 explanations with over 1,600 expert ratings across in-domain and cross-dataset settings. Despite its compact design, AnomalyJev achieves stronger agreement with human annotators than most open MLLM judges and approaches that of flagship MLLM judges, at a fraction of their runtime and resources. 

---
# Can Native Safety Travel? Reusing Safety Pretraining Across Models via On-Policy Distillation 

**Authors**: Bingqi Shang, Yang Zhang, Yiwei Chen, Changsheng Wang, Shiyu Chang, Bingquan Shen, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27555)  

**Abstract**: Safety pretraining has emerged as a promising paradigm for embedding safety deeply into a model during pretraining, yielding robustness to subsequent fine-tuning that conventional safety post-training often lacks. We refer to this safety acquired during model pretraining as native safety. Yet native safety is expensive to acquire, confined to the model in which it was pretrained, and often accompanied by lower general utility. In this work, we rethink the practical value of safety-pretrained models and introduce, for the first time, the native safety reuse problem: Can native safety be transferred from a specialized, potentially smaller, safety-pretrained "teacher" to standard-pretrained "students," enabling them to inherit its robustness while preserving their general utility? We show that native safety can "travel" through on-policy distillation. A standard-pretrained student generates responses to safety-pretraining prompts, while a small safety-pretrained teacher supervises the student's token-level predictions along its own generated trajectories; cross-tokenizer alignment further enables transfer across model families. The resulting students exhibit substantially greater safety robustness than their safety-post-trained counterparts while retaining higher general utility than the teacher: subsequent fine-tuning raises the average attack success rate (ASR) of the 1.7B student by only 5.00%, compared with 11.58% for safety post-training, and its average utility of 48.20% far exceeds the aligned teacher's 27.42%. Unlike conventional large-to-small knowledge distillation, our framework enables selective, small-to-large safety transfer: it reuses the robust safety expertise of a smaller pretrained teacher without inheriting its utility limitations. These findings establish native safety as a reusable model property rather than one permanently tied to the originally pretrained model. 

---
# Self-Evolving Agents through Privileged Retrospective Distillation 

**Authors**: Yuhang Zhou, Yifan Wu, Bo Peng, Mingyi Wang, Jiayi Liu, Xiangjun Fan, Lizhu Zhang, Zhuokai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.27495)  

**Abstract**: Self-evolving agents aim to improve from their own experience rather than remain fixed after deployment, but naturally collected trajectories often lack ground-truth solutions or verifier-derived rewards. We introduce Retrospective Self-Distillation (RSD), a multi-round framework that turns such experience into dense supervision using privileged retrospective context. For each on-policy trajectory, the same underlying model acts as both student and teacher: the teacher re-scores the student's realized actions while conditioning on cross-task precedents and lessons, together with turn-level reflection and future hindsight from the completed trajectory. The resulting teacher distribution is distilled back into the policy through on-policy self-distillation. Across rounds, the experience bank grows and the retrospective context is refreshed, allowing the model, its data, and its supervision to co-evolve. We further introduce frontier-guided context-policy evolution, where a stronger frontier model proposes constrained edits to how privileged context is constructed. Across ALFWorld, ScienceWorld, and SearchQA with Qwen3-8B and Qwen3-14B, RSD improves performance by up to 59.7% relative to the base model, while frontier-guided context-policy evolution adds up to a further 44.7% relative improvement over a fixed context policy. These results show that naturally accumulated experience can be converted into persistent model capability without ground-truth trajectories or verifier-derived rewards for the core policy update. 

---
# Recoverability and Verifier Strength: Two Measured Preconditions for Unattended Agent Self-Improvement 

**Authors**: De Wang, Peng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.27465)  

**Abstract**: Agent platforms are starting to improve their own agents: they read a deployed agent's execution traces, derive a change, have an automated verifier approve it, and release it with no human author. How much of this loop can safely run without a human? We propose two quantities a platform can measure. Verifier strength $\sigma$ is how well the available verifier -- often an LLM judge -- agrees with a trusted reference verifier; it limits how much a step can improve the agent. Recoverability is how cheaply a bad change can be undone: whether undoing it needs a new verifier approval, and how much of the old behavior a rollback restores. A step should run unattended only if both are adequate. On our platform this rule allows prompt, tool and skill edits to run unattended and keeps weight updates under human sign-off.
Our main contribution is these two definitions and how to measure them. We built a platform that applies the rule and tested it in two ways: on the platform's own approval gate, and in controlled experiments on the $\tau^2$-bench agent benchmark. On verifier strength, the platform's LLM judge sometimes gave the same score no matter what it was shown (9 of 37 settings) and reported no error, so a verifier can look healthy while approving anything; its strength must be measured, and two simple tests catch the failure. On recoverability, restoring a saved copy brought the agent back to its original behavior, but a regenerated undo did not, so only changes whose old version is kept can be reversed without a human; and while a change was deployed, it altered the outcome of 17 percent of tasks beyond normal run-to-run variation, which no undo repairs. Together these support the rule: a change can run unattended only when its old version is kept and its verifier's strength has been measured. 

---
# Ask to Distill: Learning to Synthesize Data for On-Policy Distillation 

**Authors**: Jiashuo Sun, Hao Zhang, Xueqiang Xu, Heng Wang, Yixuan Xie, Pengcheng Jiang, Zhenrui Yue, Jiajun Fan, Peiran Li, Ge Liu, Shaowen Wang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.27450)  

**Abstract**: On-policy distillation (OPD) can succeed with surprisingly few training questions, motivating a view of questions as carriers of distributional knowledge transfer. How can we assess whether a newly generated question is valuable for distillation, and learn to generate better questions---without access to its correct answer? We introduce Ask-to-Distill (\method), a framework that treats synthetic questions as distribution elicitors and optimizes the teacher supervision they expose along student-generated trajectories. A student-initialized synthesizer is optimized using a distributional surrogate for question value, computed from a frozen student--teacher pair without reference answers, correctness rewards, or per-candidate student updates. The reward combines teacher--student disagreement with teacher probability mass on the student's candidate support, together with length and diversity controls. The generated questions are filtered and deduplicated into a prompt-only bank for standard OPD. Across mathematical reasoning, code generation, and agentic tasks, our method improves average performance and teacher-gap recovery over unoptimized student-generated prompts and KL-filtered real questions, while matching or exceeding real-bank OPD in the evaluated settings. Signal analyses further show that optimized questions can elicit denser supervision and more total signal from shorter student responses, supporting elicited distributional supervision as an objective for synthetic distillation data. Our code, data and models are publicly available at this https URL. 

---
# Harness Self-Evolution for Long-Horizon Terminal Agents 

**Authors**: Ruhan Wang, Yucheng Shi, Zongxia Li, Junyao Yang, Zhongzhi Li, Yue Yu, Haitao Mi, Dongruo Zhou, Leoweiliang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27435)  

**Abstract**: Long-horizon terminal agents may solve a task in one run yet fail across repeated runs. Stochastic errors can propagate through actions and context, creating a gap between capability and reliable execution. We present RiseHarness, a framework for iteratively improving agent harnesses with model weights fixed. It focuses on flipping tasks with both successful and failed runs under the same configuration, using successful runs as references for failure analysis. RiseHarness combines agent trajectories with internal harness traces from a runtime monitor to examine how model actions interact with command execution, feedback, context management, and completion decisions. A Harness Handbook links this evidence to harness behaviors and source code, helping identify recurring problems in mechanisms shared across tasks. RiseHarness then searches other harness frameworks for relevant implementations and adapts them as design references to the current model and execution protocol. Candidate changes undergo behavioral checks and repeated task evaluation; versions with measured gains seed the next iteration. This loop integrates suitable mechanisms from different frameworks into a single harness and uses new execution evidence to guide further changes. We evaluate task success and reliability across models, ablate harness execution evidence and external research, and test transfer to unseen benchmarks with the same model. RiseHarness treats a specific model as the unit of harness adaptation, studying how changes to shared interaction mechanisms can make existing capabilities more reliable across tasks. 

---
# More Memory, Worse Agents: Error Reproduction and Anti-Persistence in LLM Agents 

**Authors**: Ruhan Wang, Kishan Panaganti, Dongruo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.27406)  

**Abstract**: LLM agents can improve across tasks without updating model parameters by storing experience from their own executions in external memory and reusing it to guide later decisions. A central design question is whether to retain only observed trajectories and outcomes or also the rules, summaries, and skills inferred from them. Retaining these abstractions allows agents to reuse strategies without repeatedly deriving them from raw trajectories. However, errors in stored abstractions can propagate to subsequent decisions through retrieval and reuse. They can also shape execution trajectories, which then become the basis for new abstractions that reproduce the original errors. We term this feedback mechanism error reproduction: errors are regenerated through a cycle of abstraction, execution, and re-abstraction, extending their persistence beyond direct reuse of the original faulty knowledge. We propose Anti-Persistence, a memory design that accumulates trajectories and outcomes while reconstructing task-specific guidance on demand. An adaptive selector uses prior outcomes to decide whether and how to synthesize guidance, which is discarded after execution. This prevents its direct reuse as a stored object, although indirect feedback through retained trajectories remains possible. Across different benchmarks and backbones, Anti-Persistence achieves higher overall held-out success and the lowest average LLM-call cost among the evaluated memory designs. These results support separating experience accumulation from abstraction persistence in memorybased self-improvement. 

---
# Steering intonational tunes in text-to-speech synthesis 

**Authors**: Juan Pablo Vigneaux, Jennifer Cole  

**Link**: [PDF](https://arxiv.org/pdf/2610.27403)  

**Abstract**: Text-to-speech systems built on residual vector quantization often generate audio in two stages: a transformer backbone emits the first level of each frame's token stack, and a smaller depth model generates the rest autoregressively. We ask which components of such a model are responsible for prosodic intonation, more precisely for generating the English nuclear tune (for instance, the falling pitch melody that distinguishes an assertion from a question's rising melody), and whether interventions on those components can be used to control intonation. Prompt-based control mostly fails; only a final question mark is a reliable lever. Interchange interventions in CSM and S2 locate the tune: the backbone's hidden state carries it to the waveform through the depth model, and the first level of the stack carries little or none of it. Steering the backbone state with a difference-of-means vector computed on new material generated with a final period or question mark moves the tune, but the push toward an assertion degrades the voice. Steering vectors computed instead from the backbone states associated with human imitations of theoretically-specified tunes, whose text carries no punctuation, and applied inside the generation loop, move most recordings to the other condition in both systems (even when fitted without the speaker being steered), at a small cost in predicted naturalness. These imitation-derived vectors are aligned with the one obtained from the two punctuated conditions far beyond a shuffled-label null, so the backbone state of a punctuated prompt already represents intonation. The steered recordings broadly become falling or rising, according to the intent, but the edge tone moves more coherently than the pitch accent, and the target tune cannot be selected precisely beyond this broad tendency. 

---
# TEX: Test-Time Scaling Testing Agents via Execution-based Cross-Validation 

**Authors**: Akash Gokul, Srijan Bansal, Yingbo Zhou, Semih Yavuz  

**Link**: [PDF](https://arxiv.org/pdf/2610.27290)  

**Abstract**: Test-case generation is critical but remains challenging for language models. Existing test-time scaling methods either isolate candidates or aggregate traces, which do not scale to long-horizon tasks. We present TEX, a collaborative test-time scaling framework based on execution-driven cross-validation. TEX parallelizes agents to generate tests and patches, cross-executing each test against every patch to form a $K \times K$ feedback matrix. Agents use their matrix slices for refinement, enabling objective validation without verbose trace sharing. On SWT-Bench Verified, TEX achieves 86.4%, establishing a new state-of-the-art and improving +9.0% over single-agent baselines. Moreover, even random selection from TEX outputs (84.4%) surpasses the previous SOTA (84.0%), showing collaborative execution feedback improves solution quality, not just best-case samples. Our results highlight execution-grounded feedback as an effective scaling mechanism for long-horizon agentic tasks. 

---
# FLINT: Fixed-budget Late Interaction 

**Authors**: Himank Handa, Negar Arabzadeh, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2610.27273)  

**Abstract**: Multi-vector retrieval models such as ColBERT often outperform single-vector models of comparable scale by matching queries and documents at the token level. However, retaining a vector for every token substantially increases storage requirements and slows retrieval compared to single-vector bi-encoders, limiting their scalability. We introduce FLINT (Fixed-budget Late INTeraction), an architecture that uses attention-based pooling to represent each query and document with a fixed number of embedding vectors. Unlike architectures that pool only document representations while retaining token-level queries, FLINT applies pooling symmetrically to both sides, limiting document storage and the number of vector comparisons needed during retrieval. FLINT achieves comparable ranking effectiveness to the closest baselines on BEIR and outperforms them on several benchmarks with longer and more complex queries, including BrowseComp+, CRUMB, and TREC Tip-of-the-Tongue 2025. We demonstrate that FLINT's effectiveness-efficiency benefits extend across encoder architectures, testing it on two separate backbones (BERT and ModernBERT). On long-query corpora, FLINT delivers 1.5-2.8 times faster end-to-end retrieval than ConstBERT, the closest baseline. FLINT also reduces index size by 5.3-9.3 times relative to token-level late-interaction models such as ColBERTv2. 

---
# TxnQA: Benchmarking Analytical Reasoning and Evidence Grounding over Transaction Event Sequences 

**Authors**: Rabimba Karanjai, Yiming Bian, Yun-Shiuan Chuang, Jenna L. Ballard, Zijie Pan, Shivani Shekhar, Yixi Li, Yirou Ge, Avinash Thangali, Richard Williamson, Linsey Pang, Prakhar Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2610.27256)  

**Abstract**: Analyzing merchant transaction histories requires dynamic temporal filtering, entity resolution, and aggregation over irregular event streams--operations poorly captured by static tabular or time-series forecasting benchmarks. We introduce TxnQA, an analytical QA benchmark comprising 27 task families (81 templates) alongside a 5-scenario fraud-risk reasoning suite that evaluates both verdicts and supporting evidence over synthetic, privacy-mitigated transaction data. Ground-truth answers are computed programmatically, and model performance is decomposed into format-following rate (FFR), conditional content accuracy (CA), and end-to-end correctness (E2E). Evaluating prompted proprietary baselines and six LoRA-adapted open-weight models shows that adaptation reliably enforces output schemas but leaves arithmetic bottlenecks: Gemma-4-31B-IT and Qwen3-8B attain E2E scores of 0.867 and 0.816, whereas evaluated MoE checkpoints reach only 0.632-0.713. Furthermore, on the 500-instance fraud suite, verdict accuracy frequently exceeds 80%, yet joint accuracy (requiring exact evidence attribution) collapses to at most ~23%. These findings identify schema compliance, semantic calculation, and evidence attribution as three distinct bottlenecks in transaction-sequence QA. 

---
# Profile-DP: Differential Privacy for User-Controlled Profile Release in Personalized LLM Inference 

**Authors**: Mingchen Li, Zifan Peng, Kewei Sha, Lavinia F Pieptea, Heng Fan, Junhua Ding, Meikang Qiu, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.27213)  

**Abstract**: Personalized large language models (LLMs) increasingly rely on profile-conditioned inference, pairing public requests with rich, user-specific facts. This paradigm exposes the user profile itself, together with the user's explicit sharing preferences. We introduce Profile-DP, a policy-aware local differential privacy framework designed specifically for user-profile release in personalized LLM inference. Profile-DP provably bounds the release of user-withheld facts while treating public instructions, profile schemas, and authorized facts as public side information. By generating a privatized profile that composes with the public instruction prior to answer generation, our framework decouples formal privacy guarantees from downstream model behavior. We develop two finite-domain mechanisms satisfying Profile-DP: DP-SLOT independently perturbs protected attributes, while DP-BLOCK uses public semantic decisions to coordinate their release and jointly sample related attributes. A positive, public compatibility prior connects these attributes while preserving a profile-level privacy bound. On PEEP, DP-BLOCK leads the utility composite among DP methods at all 13 structured budgets and is closest to the ideal privacy--utility point in 18 of 26 format--budget settings under our composite weighting. Direct measurements reveal complementary strengths: DP-BLOCK discloses less, while DP-SLOT retains more authorized facts. These results make policy-aware profile release a practical basis for studying privacy and personalization. 

---
# Beyond Two Tracks: Speaker-Anchored Full-Duplex Dialogue Recovery 

**Authors**: Yuxuan Chen, Yang Xiao, Jack Sun, Chak Sun, Brick Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.27109)  

**Abstract**: Spoken interaction is moving from alternating turns to full-duplex conversation, where an agent listens while speaking. Learning this coordination requires natural dialogue with persistent speaker-resolved tracks, a data requirement that remains difficult to satisfy at scale. Generative recovery offers a route from ordinary recordings, but recovering two voices is not enough: their track assignments can drift as the conversation unfolds. We address this missing link with speaker-anchored routing before diffusion: a 1.84M-parameter module matches the separator's coarse streams to fixed enrollment identities and orders them before a frozen generator reconstructs the speech, without retraining the generator or altering the reconstructed audio. On 25-minute remote conversations, routing cuts wrong-track time from 43.9% to 4.1%, below the 6.2% of matched-enrollment waveform re-identification, with its largest advantage when one participant holds the floor (17.8% versus 28.6%), and raises fixed-track activity F1 from 0.46 to 0.84 while per-chunk reconstruction quality is unchanged. The same router transfers to unseen face-to-face and telephone recordings, and a margin-gated verification step brings their wrong-track time to 0.9-1.7% while embedding 20-36% of the output, where full re-identification embeds all of it. Recovered voices thus become the participant-consistent tracks that full-duplex dialogue modeling needs. Audio examples: this https URL 

---
# Omni-Woodpecker: Hallucination Correction via Evidence-Prior Rebalancing for Omni LLMs 

**Authors**: Zijian Liu, Chaoyou Fu, Yan Yang, Caifeng Shan, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2610.27103)  

**Abstract**: Recent Omni LLMs (OLLMs) have demonstrated remarkable capabilities in joint audio-visual reasoning, yet cross-modal hallucinations remain a major challenge. Existing mitigation methods mainly operate at the output level, overlooking the entanglement of grounding evidence and learned priors within fused representations. To address this issue, we propose Omni-Woodpecker (OWP), a training-free framework that shifts hallucination mitigation from output-level manipulation to conflict-conditioned representation rebalancing. Specifically, OWP first identifies cross-modal conflicts from complementary modality-specific views. It then employs orthogonal decomposition to separate grounding evidence from competing priors within the hidden states. Finally, OWP adaptively rebalances fused representations by suppressing misleading priors and restoring ignored evidence before decoding. Extensive experiments demonstrate the effectiveness and strong cross-architecture generalization of OWP. Using Qwen2.5-Omni-7B as the backbone, OWP improves performance by 12.9 and 8.6 points on CMM and AVHBench, respectively, while consistently achieving gains of 0.67-2.25 points across four general OLLM reasoning benchmarks. Furthermore, OWP generalizes effectively to VideoLLaMA2-AV, improving performance by 4.6 and 3.9 points on the same benchmarks. Code is publicly available at this https URL 

---
# SciFigBench: A Benchmark for Scientific Method Figure Understanding 

**Authors**: Lijia Wu, Zheng Li, Yuxuan Liu, Jingwen Zhang, Penghai Zhao, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.27087)  

**Abstract**: Scientific method figures often provide researchers with a clear visual entry point into a paper's methodology and core contributions, yet their dense layouts and complex information flows are difficult for Vision Language Models (VLMs) to interpret. Existing benchmarks predominantly focus on coarse-grained tasks, such as context-based retrieval or holistic comprehension, overlooking the fine-grained content and inter-module relations within complex methodological diagrams. To address these overlooked aspects, we introduce Scientific Figure Benchmark (SciFigBench), a benchmark for evaluating core capabilities required for scientific method figure understanding. Specifically, it targets three capabilities: dense content perception, visual relation comprehension, and information-flow tracing. For this purpose, we curate 4,352 scientific method figures from real-world research papers and construct 40,670 QA pairs to operationalize these capabilities as measurable question-answering tasks. All annotations are generated by dedicated pipelines and then manually verified. We further propose SciFigCue, a training-free visual prompting module that externalizes arrow-induced dependencies as visual relational evidence for method-figure reasoning. Experiments across open-source and proprietary VLMs show that even strong models remain substantially limited in visual relation comprehension, while SciFigCue improves performance across several tasks without parameter updates. The code and related resources are available at this https URL. 

---
# ClockMark: Adaptive and Host-Dependent Watermarking for Large Language Models 

**Authors**: Yupeng Qiu, Han Fang, Ee-Chien Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27079)  

**Abstract**: Large language model (LLM) watermarking operates on the next-token distribution that evolves during generation. In streaming generation, each sampled token becomes part of the text and affects subsequent distributions. Embedding therefore proceeds with a changing host, unlike classical watermarking, which adapts embedding to an existing host signal. To accommodate this difference, we formulate a novel embedding framework for LLM watermarking based on two principles. We first argue that embedding should be adaptive, retaining the state of previous sampling decisions to better guide subsequent choices. We further argue that embedding should be host-dependent, using the current LLM distribution to constrain both token selection and state updates. Guided by these principles and the two requirements of distribution preservation and robustness to text editing, we propose ClockMark. It combines adaptive and host-dependent embedding with message spreading, embedding the full payload in each watermarked token. Theoretically, we prove that ClockMark preserves the original LLM sequence distribution, thereby maintaining generation quality. Empirically, extensive experiments against 12 baselines demonstrate improved robustness to text editing, with gains of up to 16.08 percentage points in bit accuracy over the strongest tested baseline. 

---
# ReuF2F: Benchmarking Coding Agents on Formalized Undergraduate Research Problems 

**Authors**: Zeyu Zheng, Yiyang Liu, Yuyang Zhao, Congmin Zheng, Jiachen Zhu, Jianian Chen, Kangning Zhang, Kayue Daniel Wong, Lingyu Yang, Rong Shan, Shihao Chen, Thomas Zhu, Wentao Zhang, Xiaopeng Li, Yan Hu, Yongxi Lin, Yuxiang Chen, Zhuoying Ou, Jianghao Lin, Jeremy Avigad, Prasad Tetali, Jingbang Chen, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2610.27046)  

**Abstract**: As AI systems begin to resolve open research problems, reliable verification becomes increasingly important. Formal mathematics provides a trusted framework for mathematical reasoning, enabling agents to produce machine-checkable proofs while conducting mathematical exploration. Existing formal mathematics benchmarks often focus on known results or a small set of prominent open problems, which can become difficult to evaluate reliably once solutions are discovered and widely disseminated. Existing evaluation protocols are also not always aligned with how modern coding agents operate. We therefore introduce ReuF2F, a Lean 4 benchmark of 244 formalized open problems drawn from Research Experiences for Undergraduates (REU), spanning entry-level open research problems across a broad range of mathematical subfields. We expect these less prominent open problems, which tend to attract less attention and may remain unsolved for longer, to be better suited for held-out evaluation of end-to-end open problem resolution in Lean. We adopt a standard coding-agent evaluation setting, placing each problem and its logical negation into a Lean project in a Docker container. We also build Leanimum-agent on mini-SWE-agent to provide a unified harness for fair comparisons across models. Our evaluation finds that even the best-performing model resolves only ~40% of the problems, leaving substantial room for improvement on formalized open research problems. 

---
# Teaching AI Agent to Play Poker Strategically Through Preference Optimization 

**Authors**: Phurinat Polasa, Thanapon Thanadunpremdet, Phatthanasak Kraiduang, Saksorn Ruangtanusak  

**Link**: [PDF](https://arxiv.org/pdf/2610.27028)  

**Abstract**: A poker agent can make a legal move and still make a strategically poor one. We ask whether compact AI agents can exhibit strategic behavior under hidden information and whether preference optimization can improve beyond static and zero-shot policies. We first evaluate a static baseline and three frozen language models: SmallThinker, Qwen3, and Gemma-3 whose performance varies substantially between exact-game and six-player evaluation. We therefore propose Strategic Preference Optimization with KL-anchored Equilibrium (SPOKE), combining teacher-relative value penalties, policy anchoring, risk constraints, and structured rationale supervision, and compare it with SFT, DPO, GRPO, and GSPO. Across three runs of a SmallThinker-4B-A0.6B policy, SPOKE achieves the lowest mean exploitability in Kuhn (0.213) and Leduc (1.081) and the highest six-max top-1 agreement (0.775), while maintaining teacher KL comparable to SFT (0.522 vs 0.523). These results provide preliminary evidence that strategic preference optimization can improve strategic quality while preserving teacher-consistent behavior. 

---
# Transformers Learn Two Competing Sorting Algorithms, and Only One Length-Generalizes 

**Authors**: Khashayar Gatmiry, Nathan W. Henry, Chloe H. Su, Emmanuel Abbe, Samy Bengio, Melanie Weber, Jonathan A. Kelner, Peter L. Bartlett  

**Link**: [PDF](https://arxiv.org/pdf/2610.26974)  

**Abstract**: We show that two-layer transformers trained to near-zero loss on a sorting task converge to one of two mechanistically distinct solution classes depending on their initialization. One solution class consists of solutions that do not use the second attention layer to sort, but also fail to length-generalize. The other, which uses both attention layers, length-generalizes perfectly, despite only being trained to sort sequences of a fixed length. By carefully choosing the weight initialization, we demonstrate that we can steer the model toward learning one solution or the other. We identify input types that are rare at the training length but increasingly common at longer sequence lengths as a source of this disparity in length-generalization, and find that layer-wise specialization of the second class of solutions allows them to handle these rare sequences more accurately. In the second class, we discover a mechanism by which the second layer attention can use the contextual information gathered by the first layer attention to reduce its search space. On the related task of finding the minimum of a sequence, we find a similar two solution-type phenomenon, but with the opposite generalization pattern, where a different form of layer-wise specialization causes the two-layer solution to length-generalize worse than the single-attention solution. 

---
# Do LLMs Really Lack Difference Awareness? Uncovering Heuristic Substitution in Group-Differentiating Fairness 

**Authors**: Jingshen Zhang, Bo Wang, Yanlin Fu, Yuewen Dang, Dongming Zhao, Zifei Yu, Yuexian Hou  

**Link**: [PDF](https://arxiv.org/pdf/2610.26961)  

**Abstract**: In this paper, we investigate the issue of difference awareness in LLMs, defined as a model's ability to recognize and acknowledge meaningful differences between social groups in fairness decision-making. Previous studies have consistently shown that models perform poorly on such tasks, a finding commonly attributed to a lack of difference awareness. We propose an alternative account, which we term Fairness Heuristic Substitution: difference awareness is not necessarily absent, but the complex demand of simultaneously identifying relevant group differences and determining their implications for fairness decisions may lead models to substitute the simpler, more accessible equal-treatment heuristic, i.e., formal equality. Based on this hypothesis, we introduce TDD, a Two-stage Distinguish-Decide framework that explicitly decomposes fairness reasoning into first distinguishing contextually relevant group differences and then making the corresponding decision, thereby discouraging reliance on formal-equality shortcuts. (i) Across four benchmarks and ten models ranging from billions to trillions of parameters, TDD significantly outperforms one-stage baselines, even one strengthened through simulated test-set leakage. (ii) Combining TDD with neuron-level contrastive probing provides mechanistic evidence consistent with the Fairness Heuristic Substitution hypothesis by revealing two signatures of the formal-equality: a compact, locally coordinated representational structure and a pronounced activation bias under default reasoning. (iii) TDD's stage-wise decomposition uncovers overlooked failure modes in fairness reasoning, including heuristic displacement and misaligned stage dynamics. 

---
# Dynamic System Inspired Continuous Large Language Model for AI4Science Tasks 

**Authors**: Zhaoxin Fan, Zheng Zhu, Yuqing Ma, Faguo Wu, Jian Yang, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26959)  

**Abstract**: Scientific foundation models increasingly operate on continuous phenomena, yet still use discrete residual computation. We ask whether pretrained residual transformations can support additional latent computation without expanding the backbone. To test this possibility, we introduce Continuum-LM, which reuses these transformations as dynamical operators in a continuous-depth language model. Its hidden states evolve through an ordinary differential equation evaluated by higher-order numerical integration, with solver order and step size controlling computation. Across eight scientific tasks, Continuum-LM trains stably and improves mean exact-match accuracy by 7.77 and 10.96 percentage points over discrete counterparts at the 0.6B and 1.7B scales, respectively. At 0.6B, it also outperforms an Ouro-style recurrent control under matched parameter and residual-function evaluation budgets. Together, these results support extensible computational depth through parameter reuse, identifying continuous hidden-state evolution as a practical design dimension for scientific language models beyond parameter scaling. 

---
# GRaSP: Latency-Aware Optimization for Geo-Distributed Retrieval-Augmented Generation 

**Authors**: Hardik Ruparel, Raj Kumar Pandey, Pramod Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.26938)  

**Abstract**: Retrieval-Augmented Generation (RAG) is increasingly served across geo-distributed cloud and edge infrastructures, where wide-area latency, cross-region egress cost, and data-residency rules compete with retrieval quality. Conventional pipelines rank by semantic similarity alone and ignore where content lives, inflating tail latency and cost. We formalize the Geo-RAG Selection Problem (GRaSP), a drop-in re-scoring layer that decays relevance by latency and adds an optional cost penalty under a cardinality budget and per-region caps, with no retraining. We first settle its complexity: the base formulation is a linear objective over a partition matroid, solved optimally by an O(N log N) greedy algorithm, rather than NP-hard as a knapsack framing suggests, with hardness entering only under an additive egress budget. On a five-node CloudLab testbed emulating a four-region WAN over the Natural Questions corpus (2.68M passages), GRaSP's benefit is placement-dependent: under uniform placement it cuts fetch latency at negligible quality cost; under topical (residency-like) placement it traces a recall-latency Pareto frontier that dominates relevance-only, nearest-k, and a latency-budgeted baseline. At a moderate latency setting, GRaSP lowers P90 fetch latency by 41% while retaining 78% of relevance-only's recall, with sub-millisecond selection overhead, stable to 200 queries/s. Findings generalize across four public BEIR benchmarks. In end-to-end question answering, GRaSP reduces context-fetch latency by 43% while remaining within 0.062 absolute EM of relevance-only retrieval. 

---
# PW-Pruning: Phase-Aware Expert Selection for Speech-Centric Omni-Modal Models 

**Authors**: Junjie Cao, Fengping Tian, Wenjie Zhou, Xuanfan Ni, Chenyang Lyu, Siyou Liu, Longyue Wang, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.26891)  

**Abstract**: Sparse routing reduces per-token computation but leaves the full expert bank resident in memory. We study expert pruning for speech-centric omni-modal models, where pooling calibration statistics can obscure the different expert demands of acoustic-input processing and text-answer generation. We propose PW-Pruning, a phase-aware extension of DO-ACP that constructs separate importance-weighted kernels for acoustic-prefill and teacher-forced answer tokens, then combines phase-weighted log-determinant marginal gains to select one static expert bank. On Qwen3-Omni, retaining 32 of 128 experts per layer improves MMAU-mini and MMSU accuracy over matched pooled selection by 3.20 and 4.38 percentage points, respectively, after supervised fine-tuning, averaged over three runs. Single-phase and shuffled-label controls on this public backbone suggest that semantic phase information contributes to these gains. Experiments on a non-public Qwen3.5-Omni backbone and English-only zero-shot evaluation of Nemotron provide complementary evidence. Across the tested Qwen budgets, AU gains are larger under stronger pruning, without inference-time mask switching. 

---
# Disentangling Depression from Cognitive Decline in Elderly Speech Using Concurrent Clinical Assessments 

**Authors**: Woori Jeon, Seunghee Ha, Sang-Kyu Lee, Ji Hye Yoon, Tae-Jin Yoon, Seung Jin Lee, Woojae Han, Jungmin So  

**Link**: [PDF](https://arxiv.org/pdf/2610.26890)  

**Abstract**: Detecting depression from speech in elderly patients with mild cognitive impairment (MCI) is complicated by overlapping acoustic effects of cognitive decline. Without disentangling these, classifiers risk learning cognitive rather than depression-specific patterns. We present a Korean elderly speech corpus of 89 MCI speakers collected over three years with concurrent depression (SGDS) and cognitive (MMSE) assessments. Both linear mixed-effects models and partial correlations identify the same pattern: of seven feature groups, only formant features are associated with depression after controlling for cognitive function, while widely used F0 features carry no signal. The F1+F2 subset achieves UAR 0.760 with just 12 features, outperforming both the full eGeMAPS baseline and self-supervised representations. These findings show that concurrent clinical assessments can identify interpretable, depression-specific acoustic markers in populations where cognitive and affective symptoms co-occur. 

---
# High-dimensional layer-state trajectories reveal the internal dynamics of transformer inference 

**Authors**: Heng Zhao, Yufei Wang, Ruidian Song, Weisheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.26866)  

**Abstract**: Mechanistic studies of large language models localize function in parameters, attention heads or neurons, but those parts do not specify how one inference unfolds. We treat the ordered layer-state trajectory as the empirical process and test whether layer order defines a measurable computation. Across representative Qwen, Llama and Gemma checkpoints, realized order produced paths 1.60--2.24 times as aligned with their endpoint chord as endpoint-preserving shuffles. The order effect transferred to a lexical task, but was stronger in three random initializations of the same Qwen architecture, identifying an architecture-compatible scaffold rather than a training-specific law. Direct hidden states provided stronger mechanism coordinates than vocabulary readbacks, and a task-defined coordinate made candidate competition computable. After additive interventions failed, local transition actuators selectively controlled internal trajectories across checkpoints. A confidence gate then drove 17 of 87 held-out Qwen closure trajectories across the full-vocabulary top-1 candidate boundary while leaving all 203 non-closure top-1 outputs unchanged. Safety-matched action reassignment performed similarly, locating the gain in selective gating and the bounded actuator family rather than fine-grained dose assignment. Layer order exposes an observable, computable and behaviourally consequential process of transformer inference without establishing a complete theory of LLM dynamics or control of answer correctness. 

---
# FRAGMENT: Factorized Graph Representations for Document Generation and Editing via Entity-Aware Transformations 

**Authors**: Ayoub El Bouchtili, Guilhaume Leroy-Meline  

**Link**: [PDF](https://arxiv.org/pdf/2608.18679)  

**Abstract**: Structured documents such as invoices, forms, reports, and scientific articles derive meaning from the interplay between spatial layout, textual content, and logical structure. Generative models operating at the pixel or token level often struggle to capture these dependencies effectively. We explore FRAGMENT, a generative framework that represents a document as a typed relational graph and factorizes its distribution as p(structure, content) = p(structure) * p(content | structure). The framework consists of two stages. The first stage, the Architect, is a causally masked Transformer conditioned on document category that autoregressively generates the graph topology and typed spatial relations. The second stage, the Builder, is a GATv2-based graph attention network that enriches the graph with normalized bounding boxes, text, and visual style attributes. Both stages define explicit likelihood models, yielding a tractable document-level likelihood that serves as an anomaly score for forgery detection. For controlled editing, a prompt-conditioned extension injects instruction embeddings into the Builder through cross-attention, enabling semantic and entity-aware modifications. We describe training on DocLayNet and fine-tuning on FUNSD and SROIE. Experiments on DocLayNet, FUNSD, and SROIE evaluate FRAGMENT alongside representative autoregressive, layout-only, and graph-based baselines, providing an empirical analysis of the characteristics and trade-offs of the proposed factorized graph generation framework. 

---
