# Copy Less, Ground More: Overcoming Repetitive Copying in Long-Context Reasoning via Evidence-Aware Reinforcement Learning 

**Authors**: Lizhe Fang, Weizhou Shen, Tianyi Tang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19345)  

**Abstract**: Large language models that generate step-by-step reasoning traces have achieved strong performance on complex tasks, and extending them to long-context settings has emerged as an important frontier. However, we identify a critical failure mode in this regime: \emph{repetitive copying}, where models extensively copy text from the input into their reasoning traces rather than productively solving the problem. We show that this behavior is pervasive across frontier long-context LLMs and intensifies with context length. By separating each prompt into task-relevant key evidence and irrelevant distractor context, we further show that the root cause is insufficient grounding: models copy from the prompt indiscriminately, and those that fail to focus on key evidence are far more likely to answer incorrectly. Motivated by this diagnosis, we propose GEAR (Grounding Evidence-Aware Reward), a reward shaping method that augments the accuracy signal with a grounding reward for overlap with key evidence and a distractor penalty for overlap with irrelevant context. To enable GEAR on natural-language data, we develop an automated pipeline that constructs evidence-annotated training data from arbitrary documents. We validate GEAR across multiple model scales and benchmarks, showing consistent improvements of up to +4.6 average points over standard RL with accuracy-based rewards, with larger gains at longer contexts, while also reducing repetitive copying and thinking length. Our findings suggest that, even as long-context evaluation shifts from simple retrieval toward complex reasoning, accurate grounding in relevant evidence remains an indispensable capability with substantial room for improvement. 

---
# Selective State-Space Adaptation and Retrieval for Language Model Reasoning 

**Authors**: Atahan Dokme, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2607.19326)  

**Abstract**: Low-rank adaptation introduces a static learned update applied identically to every input. The update provides task-level adaptation but does not explicitly represent token-level or instance-level state variation. A family of adapters is proposed that introduces selective state-space recurrence at two complementary granularities. At the token level, \textbf{MaLoRA} (Mamba-modulated low-rank adaptation) makes the adapter's scaling factor a dynamic input-dependent function with recurrent state across tokens, in contrast to the stateless modulators of prior work. At the context level, \textbf{MaRA} (Mamba Retrieval Adapter) tracks cross-segment state and selects the segments most relevant to the query, before the modulated language model generates its answer. Across three frozen backbones (Qwen-2.5-7B, Llama-3.1-8B, Gemma-2-9B) and two reasoning benchmarks (MuSiQue, 2WikiMultihopQA), the family improves reasoning accuracy on every cell of the $3{\times}2$ grid, by $+6.8$ F1 ($+10.5\%$ relative) on average and up to $+9.3$ F1 ($+18.2\%$ relative) on the hardest cell over the LoRA baseline, and the token-level gains carry to RULER QA-2 under length stress. 

---
# Two-Level Meta-Rubrics for Evaluating Open-Ended Generation: GAMUT, a Benchmark for Factual Completeness 

**Authors**: Xilun Chen, Zhaleh Feizollahi, Ross Goodwin, Seungwhan Moon, Scott Yih, Pinar Donmez, Babak Damavandi, Luna Dong  

**Link**: [PDF](https://arxiv.org/pdf/2607.19322)  

**Abstract**: Evaluating the factuality of long-form generations has focused predominantly on precision, measuring whether the claims a model makes are correct. The dominant decompose-search-verify pipeline catches incorrect claims well but says little about whether a response contains all the information it should. Measuring factual completeness, the missing half of factuality, is harder: it requires enumerating the full set of facts a complete answer should contain, and these facts rarely form a flat list. They often involve open-ended sets where coverage is what matters, ordered processes, and relationships among facts that a list of independent boolean checks fails to capture. We introduce a two-level meta-rubric framework for evaluating open-ended generation, and instantiate it as Gamut (Grounded Assessment of Multimodal Factuality), a benchmark for factual completeness in long-form generation. The framework rests on a two-level rubric representation: a structured meta-rubric captures the organization and importance of the required content, which is then mechanically compiled into a flat checklist of binary, machine-gradable rubrics that an LLM judge scores reliably. We construct 1,813 questions grounded in real wearable imagery across 10 diverse domains, each paired with an evidence-backed rubric verified by expert human annotators. Because the framework is modality-agnostic, we also release a text-only variant. Evaluating 14 frontier and open-weight models, we find the benchmark genuinely challenging (best score 58.7% from Gemini 3.1 Pro), highly discriminative, and robust to the choice of judge. 

---
# Prompt Design at Scale: How Format, Instruction Count, and Context Length Shape Instruction Adherence and Hallucination in Large Language Models 

**Authors**: Netanel Eliav  

**Link**: [PDF](https://arxiv.org/pdf/2607.19257)  

**Abstract**: Practitioners make three prompt-design decisions with almost no controlled evidence behind them: how to format instructions and context (markdown, plain text, prose, or tabular), how many simultaneous instructions a system prompt can carry before compliance degrades, and how much context a model can hold before recall and honesty degrade. We report two controlled experiments crossing all three factors on one held, contamination-free synthetic corpus (the "Book of Veyra," 8,780 uniquely-named entities, deterministically regenerable from a fixed seed), evaluated across five models.
Experiment 1 (960 calls/model) measures instruction-following decay as rule count N grows from 10 to 160, crossed with four formats and system-prompt vs. user-turn placement. Perfect-response rate collapses to zero by N=80 for every model, format, and placement. Placement produces effects at least as large as format at N=160 in most models, but the direction is model-specific. No model shows a reliable markdown advantage; one 35B model favors plain text instead.
Experiment 2 (5,520 calls/model) measures recall accuracy, false-premise sycophancy, and absent-fact fabrication across a 2k-to-512k-token context ladder in the same four formats. Recall stays near ceiling through 64-128k tokens, then degrades sharply and format-dependently: one model's accuracy spread reaches 48 points at 128k tokens. Fabrication never occurs (0/5,760 probes), and sycophancy stays negligible (<=8.3%). What rises sharply near each model's context ceiling is outright refusal to answer (0% to 79-90%), distinct from sycophancy or fabrication. Neither pre-registered format ordering holds, and token overhead (+22% to +37% over plain text) further changes which format is preferable where accuracy spread is genuine.
We release the full harness, corpus generator, and raw results (VeyraBench): this https URL 

---
# Inference-Time Steering for Cross-Lingual Factual Consistency in LLMs 

**Authors**: Alexander Manev  

**Link**: [PDF](https://arxiv.org/pdf/2607.19243)  

**Abstract**: Although Large Language Models (LLMs) demonstrate remarkable multilingual fluency, their internal knowledge representations remain disproportionately biased toward high-resource languages. This leads to cross-lingual factual inconsistency, where they shift their empirical answer distributions based solely on the prompt language. We investigate whether these biases can be mitigated at inference time, forcing an English-prompted model to answer as if it were queried in target languages (German, Spanish, Bulgarian), and evaluate four intervention strategies: zero-shot contextual steering (persona prompting), internal representation manipulation via Contrastive Activation Addition (CAA), and lightweight weight modification via Direct Preference Optimization (DPO) trained on benchmark-derived factual data as well as conceptual generalization data. To assess alignment, we curate a multilingual factual dataset alongside a novel generalization benchmark comprising culturally rooted queries to determine whether factual interventions transfer to broader target-centric preferences. Experiments on Gemma 3 12B Instruct reveal persona prompting to be the strongest overall intervention, balancing efficacy, safety, and out-of-domain generalization. While CAA yields sharp inconsistency benchmark shifts, it is configuration-sensitive and risks knowledge degradation. DPO-based adapters offer permanent, yet narrower and less transferable gains. These findings suggest that cross-lingual inconsistency is at least partly a selection problem, and that simple contextual interventions may outperform more invasive methods for robust, transferable alignment. 

---
# MeetingToM: Evaluating Multimodal LLMs on Theory-of-Mind Reasoning in Multi-Party Meetings 

**Authors**: Ziyi Wang, Yuhang Wu, Dongxu Piao, Xingyu Liu, Tianhui Zhou, Miao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.19235)  

**Abstract**: Theory of Mind (ToM), the ability to infer other's beliefs, intentions, and states of knowledge, is central to social interaction, yet remains challenging for current Multimodal Large Language Models (MLLMs), especially in multi-party meetings where cues are distributed across speech and behavior. Existing multimodal ToM benchmarks mainly focus on video-grounded question answering over overt, externally verifiable signals, and provide limited coverage of latent social states and group dynamics. We introduce MeetingToM, a benchmark for complex social behavior reasoning in naturalistic multi-party meetings. MeetingToM targets meeting-specific phenomena such as \textbf{pseudo-consensus}, where apparent agreement masks private dissent under social pressure. The benchmark is hierarchically organized to evaluate ToM at increasing levels of social granularity, including (i) subject-level mental state prediction, (ii) dyadic-level addressee understanding, and (iii) group-level consensus reasoning. We provide a unified evaluation protocol and conduct systematic analyses of representative MLLMs, revealing persistent limitations in integrating non-verbal cues, inferring hidden attitudes, and distinguishing genuine consensus from pseudo-consensus. Our results highlight key challenges and establish MeetingToM as a testbed for advancing meeting-grounded ToM in multimodal models. 

---
# Selection Shapes the Boundary: A Preregistered Replication of Monotonicity and Label Agreement in Unselected NLI Populations 

**Authors**: Haram Choi  

**Link**: [PDF](https://arxiv.org/pdf/2607.19231)  

**Abstract**: Prior work on human label variation (HLV) in natural language inference (NLI) has often relied on re-annotation resources that select items by disagreement level. An earlier study (arXiv:2607.15870) found that hypotheses containing non-upward monotonicity operators showed lower label agreement in ChaosNLI (Cliff's delta = -0.284), which is restricted to items whose majority label carries exactly three of five votes. We preregistered a replication of this boundary in the unselected populations that ChaosNLI was drawn from: the SNLI and MultiNLI development sets, using the same operator tagger and a four-level ordinal agreement outcome. The registered prediction fails. All seven contrasts return a positive Cliff's delta (non-upward items agree slightly more, not less), the only significant confirmatory contrast has the opposite sign to the registration, and every effect is far below our smallest effect size of interest (0.10). Robustness checks support the measurement: simulated tagger misclassification shrinks the effects rather than manufacturing them, and a manual re-tagging audit reaches four-class agreement of 0.875 on a fresh 200-item sample. We conclude that the earlier negative boundary is plausibly a structure conditional on low-agreement selection rather than a population-level property, and that HLV structure claims built on selected re-annotation resources should state their selection conditional explicitly. 

---
# The Price of Reasoning: Cost-Quality Tradeoffs in Reinforcement Learning for Neural Machine Translation 

**Authors**: Michael Jungo, Aixiu An  

**Link**: [PDF](https://arxiv.org/pdf/2607.19226)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has been established as a viable paradigm for the post-training of Large Language Models (LLMs), including downstream tasks, such as Neural Machine Translation (NMT). With the latest research indicating that RLVR could be the preferred training method for translating legal documents due to the induced reasoning capabilities, it raises the question whether it is really attributed to the reasoning or more generally to the training paradigm. We investigate the importance of including the model's reasoning trace in the generated responses during both training and inference by systematically omitting it from one of the phases. Our experiments show that including the reasoning, specifically during inference, has a positive effect on the overall translation quality. Furthermore, we recognise that the reasoning leads to an increase in output tokens, hence we study the cost-quality tradeoff between the increased computational demands and the improved translation quality. 

---
# Beyond Score Prediction: LLM-Based Essay Scoring and Feedback Generation via Reinforcement Learning with Rubric Rewards 

**Authors**: Xuefeng Jin, Jiashuo Zhang, Teng Cao, Bin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19219)  

**Abstract**: Large language models (LLMs) have been widely applied to automated essay scoring (AES) and automated feedback generation (AFG). However, existing studies rely primarily on prompt engineering or supervised fine-tuning, while systematic research on reinforcement learning (RL) post-training and automated evaluation of feedback quality remains limited. We propose RLAES, a unified LLM framework that jointly optimizes essay scoring and feedback generation through RL. To make feedback quality measurable, interpretable, and usable for training, we introduce Rubric-based Feedback Evaluation (RFE), an essay-grounded feedback evaluation framework comprising 166 fine-grained binary rubric items and an LLM-as-judge. Building on RFE, we propose Adaptive Gated Feedback Optimization (AGFO), which activates rubric-based feedback rewards on demand during RL, reducing evaluation overhead while improving feedback quality. We also propose Adjacent Contrastive Reasoning (ACR) to improve ordinal score calibration by explicitly contrasting adjacent score levels. Experimental results show that the RFE framework captures essay-feedback consistency, exhibits strong pairwise discriminative power, and closely aligns with expert preferences. On the ASAP benchmark, RLAES-AGFO achieves the best scoring performance among LLM-based methods (QWK = 0.803), while maintaining feedback quality comparable to GPT-5.5 and avoiding the feedback degradation observed under score-only RL. Code and datasets are publicly available at this https URL. 

---
# MIRA-Ev:A Benchmark for Granular Evidence Detection and Relational Reasoning in Clinical Exams 

**Authors**: Iker De la Iglesia, Johanna Ramirez-Romero, Jose Maria Villa-Gonzalez, Irune Urroz García, Ander Barrena, Aitziber Atutxa  

**Link**: [PDF](https://arxiv.org/pdf/2607.19201)  

**Abstract**: Clinical NLP evaluation remains dominated by multiple-choice question answering (MCQA), which scores only final-answer accuracy and cannot detect when a model reaches the correct diagnosis while grounding it in irrelevant, absent, or contradictory evidence. We introduce MIRA-Ev, a clinical argument mining benchmark built on Spanish Médico Interno Residente (MIR) licensing-exam cases, re-annotated by expert clinicians with span-level premises, claims, and directed support/attack relations, and released in parallel Spanish (native), English, and Basque versions, the first clinical argumentation resource in Basque. MIRA-Ev organizes evaluation into a three-tier task hierarchy: evidence sentence retrieval, argumentative component extraction, and relation classification. 

---
# Reasoning Before Translation: Enhancing Legal Machine Translation with Structured Reasoning 

**Authors**: Aixiu An, Michael Jungo, Eloi Eynard, Mark Drenhaus, Andreas Fischer, Jean Hennebert, Sébastien Rumley  

**Link**: [PDF](https://arxiv.org/pdf/2607.19181)  

**Abstract**: Neural machine translation (NMT) in the legal domain is a linguistically and conceptually demanding task, primarily due to the complexity of legal language and the high level of precision it requires. The recent emergence of reasoning-capable language models opens new possibilities for tackling such challenges. They add to a set of other previously proposed techniques to enhance the translation quality, which includes supervised fine-tuning and reinforcement learning.
In this work, we perform a comparison between these various approaches. More particularly, we evaluate small language models such as Qwen3.5 4B, Qwen3.5 9B, and Gemma 3 12B enhanced with various re-training paradigms and compare their performances against frontier reasoning models. We focus on the Swiss legal system, which -- with its unique multilingual statutes -- offers a particularly challenging testbed for reasoning-augmented models. Our results show that the quality of small ``base'' models can be greatly enhanced, and that reinforcement learning with verifiable rewards can be applied to NMT in the legal domain and surpasses the translation quality of supervised fine-tuning. The performance of enhanced small models is close to the one of state-of-the-art reasoning models yet remains inferior. We also note that re-training paradigms yield diminishing returns as model size increase. The code and models are publicly available at this https URL. 

---
# Automated Extraction of Techno-Economic Data from 76,000 Energy System Studies 

**Authors**: Maxime Gorres, Jan Göpfert, Patrick Kuckertz, Noor Titan Putri Hartono, Heidi Heinrichs, Jochen Linßen, Iain Staffel, Jann Michael Weinand  

**Link**: [PDF](https://arxiv.org/pdf/2607.19178)  

**Abstract**: Energy system models guide societally important decisions, but their credibility rests on quantitative assumptions that are difficult to source and audit. Meta-analyses can improve transparency and modeling practices, but the rapid growth of publications makes manual information extraction increasingly impractical. Consequently, databases are updated infrequently and efforts are often duplicated across research groups. Here, we demonstrate the highly accurate automated extraction of quantitative information from 76,000 energy system studies published since 2010. We compile 3.2 million structured quantitative data points together with 20 million associated metadata entries, spanning a broad spectrum of technologies, methodological approaches and system characteristics. Beyond providing input data for models, the resulting FAIR database make the energy systems literature itself analysable. We show where academic assumptions diverge from empirical observed data, and how research priorities vary at scale across technologies, regions and time. To facilitate broad use within the community, the database is provided through an interactive dashboard, enabling users to filter, analyse and download data according to their specific research needs. 

---
# Translation as Augmentation: Effect of Translated Data on Assessment of Difficulty 

**Authors**: Yiheng Wu, Jue Hou, Roman Yangarber  

**Link**: [PDF](https://arxiv.org/pdf/2607.19101)  

**Abstract**: Reliable Text Difficulty Assessment is a prerequisite for valid text simplification workflows and personalized learning applications. However, the development of robust assessment models is severely hindered by a critical bottleneck: the scarcity of expert-annotated corpora containing fine-grained difficulty levels (e.g., CEFR), particularly for lower-resource languages. This paper addresses this data scarcity problem in the context of a low-resource European language. We propose a cross-lingual data augmentation strategy that leverages machine translation to transfer labeled resources from high-resource languages to the target low-resource language. We train BERT-based regression models to predict difficulty scores and investigate whether synthetic, translated data can effectively supplement native training sets. Our experiments demonstrate that augmenting scarce native data with machine-translated corpora significantly improves the accuracy of difficulty estimation, offering a viable solution for languages lacking extensive expert annotations. 

---
# DAIS: Dependency-Aware Intermediate QA Supervision for Complex Reasoning 

**Authors**: Yu Wang, Ming Fan, Xicheng Zhang, Zhiyong Li, Zhihu Wang, Caiyue Xu, Dahai Hu, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.19088)  

**Abstract**: Chain-of-thought (CoT) supervision exposes intermediate rationales, but flat rationale targets usually optimize a single reasoning sequence and provide limited supervision on how local conclusions should support later decisions. We introduce Dependency-Aware Intermediate QA Supervision (DAIS), a training-time framework that converts filtered teacher rationales into stage-level QA records. Each intermediate record predicts a local answer conditioned on the previous states needed for that decision, while the final-answer record keeps the original task format; evaluation therefore uses only the original input and optional context. Across GDPR, AIACT, MedQA, and FOLIO with multiple Qwen backbones, DAIS improves average final-answer accuracy over answer-only, flat chain-of-thought, and independent-QA baselines. On policy-compliance benchmarks, it achieves a largest gain of 5.6% and an average gain of 4.2% over the strongest non-DAIS baseline. Controlled ablations show that valid previous-state conditioning contributes beyond longer targets or additional intermediate text, supporting dependency-conditioned intermediate QA as a lightweight auxiliary supervision signal for standard final-answer inference. 

---
# Benchmarking Human and Automatic Speech Recognition of Diverse Speech: Initial Results 

**Authors**: Ilse Huisman, Rares Popa, Yuanyuan Zhang, Odette Scharenborg  

**Link**: [PDF](https://arxiv.org/pdf/2607.19049)  

**Abstract**: Humans are often considered to be the best listeners and seen as the upper-bound performance of automatic speech recognition (ASR) systems. We present a preliminary comparison of the performances of state-of-the-art ASR systems and Dutch native listeners on the recognition of "diverse" speech, specifically Dutch child and older adults' speech and Flemish. Google Telephony outperformed the other ASR systems. Importantly, the ASR systems showed similar performance to the listeners, and in specific cases even outperformed them. Slight performance differences between the listeners and ASR systems were found related to speaker's age and regional accents and utterance length. Future research should focus on making ASR systems more robust to acoustic variability related to aging and regional accents. A comparison of ASR recognition performances on the test stimuli and the full Jasmin-CGN test sets showed the influence of the specific test sets on the conclusions regarding benchmarking human and ASR performance. 

---
# Content is What Remains: Invariant Speech Tokenization from Parallel Utterances 

**Authors**: Laurin Wagner, Bernhard Thallinger, Miroslav Stankovic, Mario Zusag  

**Link**: [PDF](https://arxiv.org/pdf/2607.19033)  

**Abstract**: Discrete speech tokenizers aim to disentangle semantic from acoustic information, yet targets from self-supervised learning (SSL) models like HuBERT retain non-linguistic variation: speaker identity, prosody, and channel conditions leak into the tokens, inflating entropy. Our key insight is that when enough speakers utter the same words under varying conditions, linguistic content is the only shared factor. We propose PINT (Parallel INvariant Tokenization), which fine-tunes an SSL encoder with alignment losses across parallel utterances and augmentations to distill this shared residual. PINT collapses identical words onto consistent token sequences, drastically reducing conditional entropy. Unlike ASR text, PINT tokens preserve frame-level temporal grounding and serve as drop-in semantic targets for audio codecs. Experiments show a 98.7% relative reduction in speaker probe accuracy (93.1% to 1.2%), a 42% lower ABX error rate, and 27-30% lower LM perplexity versus baselines, confirming that the right invariance is key to efficient learning. 

---
# Computational Humor with Multimodal LLMs: Methods, Datasets, Evaluation, and Challenges 

**Authors**: Tuo Liang, Zhe Hu, Disheng Liu, Jing Li, Yu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2607.19011)  

**Abstract**: Multimodal humor in memes, cartoons, and comics remains difficult for AI systems because intended meaning depends on non-literal mechanisms, shared cultural knowledge, and communicative intent rather than literal scene description. This survey focuses on visual humor understanding in single-image and multi-panel artifacts, while treating humor generation as an emerging downstream frontier. We position the literature against prior humor, sarcasm, and general MLLM surveys and organize it using a capability-centric hierarchy spanning recognition, interpretation and reasoning, and generation. Under this lens, we synthesize benchmark design, evaluation protocols, and modeling paradigms, tracing the field's shift from task-specific fusion models to large-model approaches based on multimodal alignment, evidence-grounded reasoning, and controlled generation. We conclude by highlighting the main barriers to progress: shortcut-prone evaluation, limited cultural and narrative coverage, weak evidence grounding, and unresolved safety and ownership concerns. 

---
# MedDDC-Eval: Diagnosis-Decoupled Evaluation of Multi-Turn Medical Consultation Agents 

**Authors**: Guofeng Zhang, Yizeng Quan, Huaiyi Fang, Jianwei Lv, Jinyao Liu, Xunxu Duan, Lening An, Yu Ouyang, Junfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.18999)  

**Abstract**: Multi-turn medical consultation agents must decide what to ask, adapt to patient responses, and determine when the collected evidence is sufficient. However, coupled evaluation conflates the quality of the policy-elicited history with policy-specific terminal diagnosis generation: strong generation can compensate for a thin history, while weaker generation can obscure a rich one. We introduce MedDDC-Eval, a diagnosis-decoupled testbed that treats elicited history as the comparison object and holds the history-to-diagnosis mapping constant through a shared frozen reader. Across two held-out sources, a grounded interface and an auditable diagnosis-trajectory-efficiency (D/T/E) harness measure diagnostic usefulness, information acquisition, and efficiency. Directional semantic coverage followed by deterministic one-to-one assignment yields coherent precision-recall counts for open-ended items, with at most one credited match per prediction or reference. Holding histories fixed, changing only the diagnostic reader shifts diagnosis F1 by 2.2-19.0 points and reverses 18% and 36% of pairwise policy orderings on the Record and Dialogue splits. We further apply standard Group Relative Policy Optimization (GRPO) over interactive multi-turn rollouts to post-train Qwen3-32B using diagnosis-result and trajectory feedback. On the 100-case Record and 70-case Dialogue splits, the trained policy improves over its initialization by 9.7 and 4.6 total-score points; removing either primary signal lowers held-out joint performance. These results show that MedDDC-Eval supports controlled attribution, interpretable elicited-history measurement, and evaluation-guided evidence-acquisition policy development. 

---
# Disentangling Curriculum Learning in NLP: Towards a Unifying Taxonomy 

**Authors**: Vanessa Toborek, Florian Seiffarth, Sebastian Müller, Tamás Horváth  

**Link**: [PDF](https://arxiv.org/pdf/2607.18984)  

**Abstract**: Despite more than a decade of curriculum learning (CL) research in NLP, the field lacks a principled account of which difficulty function or scheduler to use for a given problem. To understand what has hindered progress towards this account, we propose a fine-grained taxonomy separating difficulty evaluation from training scheduling to enable systematic analysis of CL strategies. For difficulty evaluation, we distinguish attribution source and task dependence, revealing difficulty as a perspectival concept encoding different assumptions about what makes an instance hard to learn. For scheduling, we provide the first formalisation of CL schedulers in terms of expected training contribution, enabling comparison across implementations by introducing retention regimes and monotonicity properties. Applied in a dedicated analysis of CL works in NLP, our taxonomy reveals a systematic incomparability problem: prior works conflate distinct notions of difficulty and scheduling, often pursuing different objectives under the same CL label -- hindering comparison and the accumulation of a coherent evidence base. Beyond diagnosis, the taxonomy supports the design, analysis, and comparison of CL strategies, and motivates evaluation practices that disentangle the sources of observed improvement. 

---
# AutoJourn: Multi-Perspective Summarisation, Bias Detection and Bias Neutralisation for LLM-Generated News in Automated Journalism 

**Authors**: Himel Ghosh, Ahmed Mosharafa, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2607.18983)  

**Abstract**: We present AutoJourn, a demonstration system for multi-perspective news generation and bias-aware evaluation using large language models (LLMs). The system tackles three core challenges in responsible automated journalism: extracting diverse perspectives from unstructured social media discussions, generating summaries that preserve viewpoint diversity, and detecting or mitigating bias in AI-generated news. The pipeline integrates advanced prompt engineering with optional retrieval augmentation to produce semantically diverse perspective sets, a multi-perspective summarisation module that merges conflicting viewpoints into balanced summaries, and a bias analysis suite supporting sentence-level bias detection and type classification in the generated news article, and automatic neutralisation. Users can inspect perspective clusters, compare stance-specific summaries, generate news articles, and apply bias-aware rewrites directly in the interface. We evaluate each component with intrinsic metrics -- semantic diversity, summary quality, and bias reduction and show improvements over strong baselines while maintaining content fidelity. A live, publicly accessible demo accompanies the paper to facilitate reproducibility and further research on socially responsible automated journalism. 

---
# Verifiable Self-Evolution for Open-Ended Dialogue Skills via Future-Feedback Prediction 

**Authors**: ChaoJin Zhao, Xuan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2607.18973)  

**Abstract**: Textual skills provide a lightweight way to improve frozen language-model agents, but their self-evolution normally requires a stable validation signal. Such signals are natural in mathematics or code, where an answer can be checked after it changes, yet are problematic in open-ended dialogue: changing the assistant response also changes the user's next reaction, so a logged reaction cannot directly evaluate a counterfactual response. We propose future-feedback skill evolution, which first redirects self-evolution from prescribing the current answer to predicting whether the observed answer will lead to a positive or negative subsequent user signal. This prediction task is verifiable on fixed logged tuples and therefore supports validation-gated textual optimization. The evolved feedback skill captures interpretable criteria for response quality and can subsequently serve as a diagnostic and optimization target for answer skills. On a proprietary, privacy-preserving sales-assistant dataset, careful quality filtering and a balanced resolved/unresolved split yield more than 75% prediction accuracy. Beyond this result, the central contribution is a formulation that converts otherwise moving conversational feedback into a fixed offline learning target, enabling reproducible skill evolution without placing every candidate skill in live traffic. We discuss the boundary between observational verification and counterfactual validity, and position the method as an offline optimization stage rather than a replacement for final human or online evaluation. 

---
# Constrained CTC Decoding for Efficient Diacritic Restoration 

**Authors**: Rufael Marew, Amr Keleg, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2607.18946)  

**Abstract**: In this work, we address diacritic restoration for Arabic speech transcripts. Most speech data are undiacritized, limiting the ability of modeling fine-grained phonological distinctions. The speech modality has recently been explored as a way to complement text-based diacritic restoration efforts. We propose an efficient non-autoregressive approach for speech-to-text diacritization based on Connectionist Temporal Classification (CTC). Our method incorporates hard constraints during decoding by constructing a character-level diacritization lattice from an undiacritized transcript and restricting hypotheses to valid diacritized realizations. We evaluate on Classical Arabic and Modern Standard Arabic test sets (namely, ArVoice and ClArTTS) against a more computationally-complex multi-modal diacritic restoration baseline, and show statistically significant reductions in diacritic error rates in both, demonstrating that the proposed approach offers both performance and efficiency gains. 

---
# Transcription Policy as a Latent Variable: Activating Controllable Verbatim ASR with Word-Level Timing 

**Authors**: Laurin Wagner, Mario Zusag, Bernhard Thallinger  

**Link**: [PDF](https://arxiv.org/pdf/2607.18934)  

**Abstract**: Modern ASR models trained on heterogeneously annotated data treat transcription style (verbatim vs. intended) as an uncontrolled latent variable, causing measurable decoding instability, evaluation confounding (up to 60% of reported WER attributable to style mismatch), and unreliable word-level timing. We show that models already encode both styles; the challenge is controlled activation. Using coverage-aware decoder task tokens trained on parallel verbatim/intended transcript pairs, we raise German disfluency F1 from 10% to 79% zero-shot, despite English-only training. Full English-only fine-tuning surpasses all baselines in verbatim accuracy, disfluency detection, and intended-mode quality across both languages. We further introduce supervised cross-attention fine-tuning that improves word-level timestamps on disfluent speech beyond forced-alignment baselines. Finally, we propose verbatimize, a new task enabling scalable creation and enrichment of speech corpora with high-quality canonical verbatim transcriptions. 

---
# Reasoning Error from Known Fact: Step-Level Self-Consistency Group Relative Policy Optimization for LLM 

**Authors**: Xiaomeng Hu, Jiaqi Hu, Hao Chen, Qi Zhang, Zhanming Shen, Wentao Ye, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.18915)  

**Abstract**: With the rapid advancement of large language models (LLMs), modern systems not only possess strong foundational capabilities and extensive knowledge, but can also solve complex problems via long, multi-step reasoning. However, as reasoning traces become longer, LLMs may produce a substantial amount of hallucinated content during the reasoning process, which is often difficult to detect. In this work, we conduct a fine-grained analysis of hallucinations arising in LLM reasoning and find that the reasoning traces are particularly prone to Context-Sensitive Factual Hallucinations: cases where the model actually has the relevant knowledge, yet makes factual errors due to contextual interference during reasoning. To address this issue, we propose Step-level Self-Consistency Group Relative Policy Optimization (SSC-GRPO), which assigns step-level rewards to reasoning traces by computing self-consistency scores of individual steps across multiple rollouts. Compared with prior methods, SSC-GRPO achieves state-of-the-art performance on both mathematical reasoning benchmarks and hallucination leaderboards. Our results offer a new perspective for detecting and mitigating hallucinations in the reasoning process of large language models. 

---
# From a Multilingual Streaming ASR Backbone to Kenyan-Language Systems: Data-Centric Adaptation of Nemotron 3.5 for Kikuyu, Dholuo, and Kalenjin 

**Authors**: Mark Gatere  

**Link**: [PDF](https://arxiv.org/pdf/2607.18912)  

**Abstract**: Automatic speech recognition (ASR) for African languages is constrained by orthographic inconsistency, annotation artifacts, missing audio, speaker and domain imbalance, and evaluation procedures that differ from deployment. We present an end-to-end engineering study adapting NVIDIA Nemotron 3.5 ASR Streaming 0.6B to Kikuyu, Dholuo, and Kalenjin. Starting from a Kenyan Swahili-adapted checkpoint, we retain its cache-aware FastConformer RNN-T, prompt conditioning, and streaming decoder during full-parameter fine-tuning. The study covers corpus auditing, Unicode normalization, split checks, duration filtering, low-rate continuation, validation-based checkpoint selection, true-streaming evaluation, artifact preservation, and isolated serving.
On internal, adaptively consulted evaluation sets excluded from gradient updates at context [56,13], selected Kikuyu and Dholuo models achieve 42.97% and 33.98% WER, respectively. Dholuo records 9.59% CER and 8.13% no-space CER under its frozen historical label policy; Kikuyu records 7.79% no-space CER. Kalenjin remains a work in progress: v1-v reaches 68.74% WER on a 2,411-row clean-v3 diagnostic subset excluding long-pause annotations, digit-bearing references, and targets shorter than three tokens. Its checkpoint selection used a mixed-source validation manifest containing test-origin rows, so the score is not an independent generalization estimate. We also report negative findings involving non-speech labels, short-utterance over-generation, boundary-sensitive WER, and cloud job-lifecycle failures. We make no state-of-the-art claim because the internal sets, repeated consultation, and normalization differ from public benchmarks. This work provides an auditable account of adapting a multilingual streaming model into language-specific systems without discarding streaming constraints. 

---
# HPD-Parsing: Hierarchical Parallel Document Parsing 

**Authors**: Shu Wei, Jingjing Wu, Lingshu Zhang, Qunyi Xie, Hao Zou, Le Xiang, Xu Fan, Yangliu Xu, Manhui Lin, Xiaolong Ma, Cheng Cui, Tengyu Du  

**Link**: [PDF](https://arxiv.org/pdf/2607.18839)  

**Abstract**: Efficient teamwork typically combines global coordination with parallel execution, a principle not yet fully reflected in unified Vision-Language Model (VLM)-based document parsers. Existing unified parsers process an entire page jointly but generate its output through a single token-by-token autoregressive trajectory, creating a sequential bottleneck that grows with document length. Such full-page sequential generation overlooks a key property of document parsing: layout must be analyzed globally, whereas block content can be parsed in parallel. Based on this observation, we introduce HPD-Parsing, which replaces full-page autoregressive generation with a Hierarchical Parallel Decoding paradigm. A main layout branch organizes the overall document structure and dynamically assigns block-level content decoding to concurrent branches, while progressive multi-token prediction (P-MTP) further reduces the decoding steps within each branch. Experiments on public benchmarks show that HPD-Parsing achieves 4,752 tokens per second, delivering $2.62\times$ the throughput of the fastest existing document parsing model and $3.06\times$ that of the vanilla autoregressive baseline, while maintaining competitive parsing accuracy. These results establish hierarchical parallel decoding as an effective alternative to full-page autoregressive generation, opening a new direction for efficient unified document parsing. 

---
# AILQA: Evaluating AI-Driven Legal Question Answering Systems for the Indian Legal System 

**Authors**: Shubham Kumar Nigam, Shubham Kumar Mishra, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2607.18825)  

**Abstract**: This comprehensive study introduces an advanced Artificial Intelligence for Indian Legal Question Answering (AILQA) system tailored to the Indian legal context. AILQA leverages a variety of embedding and generative models, including recent Large Language Models (LLMs), to address the unique challenges posed by the intricate and diverse nature of Indian legal texts and to enhance the accuracy and reliability of responses to legal questions. We conducted rigorous evaluations using both lexical and semantic metrics, enriched by expert legal feedback, to ensure relevance and accuracy. Our findings underscore the effectiveness of the Retrieval-Augmented Generation (RAG) paradigm in improving answer quality, particularly in complex legal domains. Additionally, we assessed performance on standardized tests such as the All India Bar Examination (AIBE), thereby providing a robust benchmark for practical applications. Under the study's evaluation protocol, some AI-generated responses received higher ratings than the available reference answers, particularly when they contained accurate and relevant supporting details. This finding is specific to the evaluated dataset and rating criteria and should not be interpreted as evidence that the models generally outperform qualified legal professionals. We also discuss the challenges encountered, such as the need for precise context and the risks of model hallucination, and propose directions for future research to further refine AI capabilities in the legal field. This study aims to pave the way for enhanced legal decision-support systems, making them more accessible and effective for legal professionals and the public alike. 

---
# CASE: Causal Alignment and Structural Enforcement for Improving Chain-of-Thought Faithfulness 

**Authors**: Ziming Wang, Yinghua Yao, Changwu Huang, Ke Tang, Xin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2607.18820)  

**Abstract**: Chain-of-thought (CoT) reasoning is widely used to improve both the performance and interpretability of large language models (LLMs), yet the generated reasoning may not faithfully support the final answer. We study this problem from a causal perspective, where a faithful CoT process should follow the chain $Z\rightarrow X\rightarrow Y$, with $Z$, $X$, and $Y$ denoting the instruction, reasoning chain, and final answer, respectively. In this process, the instruction should affect the answer only through the reasoning chain. However, conventional autoregressive LLMs condition answer generation on both the instruction and the CoT, which still allows a direct instruction-to-answer shortcut. To address this issue, we propose CASE, a framework that combines training-time causal alignment and inference-time structural enforcement. During training, CASE builds counterfactual-CoT, biased-instruction, and empty-instruction datasets, and applies selective-loss fine-tuning to strengthen CoT-to-answer dependence while suppressing instruction shortcuts. During inference, CASE masks direct attention from instruction tokens to answer tokens, preventing the model from bypassing the generated CoT. We provide an information-theoretic analysis showing how these components promote faithful chains. Experiments on three models and four benchmarks show that CASE achieves a 37\% average per-setting relative improvement in overall CoT faithfulness over the strongest baselines, exhibits stronger cross-dataset faithfulness transfer, and maintains competitive average accuracy. Code is available at this https URL. 

---
# RF-Agent: A Practical Framework for Building Language Agents for RFIC Design 

**Authors**: Yueqi Xing, Houbo He, Jolie Wang, Erin Ni, Shikai Wang, Qiufeng Li, Weidong Cao, Taiyun Chi  

**Link**: [PDF](https://arxiv.org/pdf/2607.18772)  

**Abstract**: Large language models (LLMs) have driven rapid progress in electronic design automation (EDA), yet their application to radio-frequency (RF) circuit design remains limited by the scarcity of domain-specific datasets and standardized benchmarks. We present RF-Agent, which addresses this gap through textbook-driven knowledge distillation. A multi-agent Question-Thinking-Solution-Answer (QTSA) pipeline converts a subsection-level corpus from seven canonical RF textbooks into the first-of-its-kind RF-domain reasoning dataset (over 11,000 samples) with a dedicated multiple-choice benchmark. On this benchmark we study two adaptation strategies: supervised fine-tuning (SFT) and three retrieval-augmented generation (RAG) configurations (semantic, keyword, hybrid). Across multiple LLM families, domain-specific SFT significantly improves RF reasoning, especially for small and medium-sized models; among RAG configurations, semantic retrieval performs best, indicating embedding-based context alignment suits RF reasoning better than naive fusion. The dataset and benchmark provide a reusable foundation for future work on LLM-aided RF circuit design. 

---
# Dual Attention Residuals 

**Authors**: Xingda Yu, Yining Li, Xinzhang Liu, Zhihao Yang, Haowei He, Chao Wang, Yongxiang Li, Shuangyong Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.18730)  

**Abstract**: Recent work extends Transformer residual pathways along two complementary axes: historical retrieval selects information from earlier depths, whereas multi-stream methods maintain multiple residual trajectories. These capabilities have largely been studied in isolation, and assigning an independent retriever to each stream still prevents one trajectory from influencing depth selection in another. We propose Dual Attention Residuals (DAR), which brings multi-stream interaction into historical retrieval through reciprocal cross-stream addressing. For each target stream, DAR computes depth weights from normalized states in the opposite stream and applies them to values from the target stream's own history. The retrieved states are combined for an unchanged Transformer branch and updated through constrained gated writes; a block-form variant operates on block-level histories to control overhead. Across dense models from 0.1B to 1B parameters and a 7B sparse-MoE model, DAR consistently improves validation loss over standard residual Transformers and Attention Residuals. Routing ablations show that the gain cannot be explained by an additional stream or value projection alone. Representation and intervention analyses further show that reciprocal cross-stream selection preserves depth-wise diversity and avoids the redundancy or functional imbalance observed in alternative two-stream designs. 

---
# Find Before You Fine-Tune: A Diagnostic Study of Small LLMs for Cybersecurity QA 

**Authors**: Shaswata Mitra, Subash Neupane, Trisha Chakraborty, Himanshu Tripathi, Sudip Mittal, Aritran Piplai, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2607.18725)  

**Abstract**: Large Language Models (LLMs) are increasingly fine-tuned for critical-domain Question-Answering (QA), yet choosing which small model to adapt, before paying the cost of adaptation, remains difficult. Fine-tuning can improve domain alignment, but it may also erode prior knowledge, weaken instruction-following, or increase hallucination, especially when labeled data are scarce or rapidly evolving as in cybersecurity. We present FiT (Find before Fine-Tune), a task-oriented diagnostic framework that characterizes small LLMs along three capabilities required for cybersecurity QA: vocabulary recognition, parametric knowledge, and contextualization of retrieved information. Using FiT, we conduct an empirical study of five open-weight 7-billion-parameter models under two fine-tuning regimes. We find that fine-tuning does not uniformly help: it consistently degrades vocabulary and parametric knowledge in small models, and the two regimes trade off differently. Knowledge-focused tuning causes moderate, rank-preserving degradation, whereas instruction-focused tuning collapses measured knowledge through induced abstention, inverting the knowledge ranking while leaving retrieval-grounded contextualization essentially intact. We quantify these regime-specific patterns with rank-correlation analysis and show that pre-fine-tuning FiT scores anticipate the direction of post-tuning change. Our results suggest that task-oriented diagnosis can screen out unsuitable models, avoid unnecessary fine-tuning, and support safer deployment of small LLMs in cybersecurity QA pipelines. 

---
# Rationale-Guided Knowledge Distillation for Cross-Lingual Stance Detection 

**Authors**: Qiuli Zhou, Jingyuan Yao, Shengeng Tang, Hongzhi Chen, Jun Tang, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2607.18693)  

**Abstract**: Stance detection aims to identify whether a text expresses a favorable or opposing attitude toward a given target, and serves as an important task for various downstream applications. Although existing studies have achieved strong performance in monolingual settings, especially in English, many low-resource languages such as Catalan still lack sufficient annotated data for training effective models. Cross-lingual stance detection alleviates this problem by transferring stance knowledge from resource-rich languages to low-resource languages. However, most existing methods mainly rely on semantic alignment between texts and targets, while ignoring the reasoning process required for reliable stance inference. Although Large Language Models provide strong reasoning ability, their high computational cost and inference latency limit practical deployment. To address these limitations, we propose a rationale-guided knowledge distillation framework for cross-lingual stance detection. Specifically, we use Chain-of-Thought prompting to guide Large Language Models in generating informative rationales, and distill the resulting reasoning knowledge into a compact student model. We further design a dual-path distillation mechanism to align rationale-enhanced and rationale-free representations, together with their prediction distributions. In addition, two contrastive learning strategies are introduced to improve stance discrimination. Experiments on multilingual benchmarks demonstrate that our method consistently outperforms competitive baselines. 

---
# Fusion Embedding: A Unified Embedding Space for Text, Image, Video, and Audio 

**Authors**: Abdul Basit Tonmoy, Kazi Fardinul Hoque, Md. Shahrier Islam Arham, Arman Luthra  

**Link**: [PDF](https://arxiv.org/pdf/2607.18666)  

**Abstract**: A single embedding space that covers text, images, video, and audio lets one index serve every query a user can pose. Embedding models built on vision-language backbones now lead text/image/video retrieval benchmarks but lack audio entirely, while audio-text retrieval is led by specialist systems that serve no other modality. We present the Fusion Embedding family, which adds audio to a frozen vision-language embedding base whose parameters are never updated: generation 1 (fusion-embedding-1) trains only a 16.4M-parameter connector between a frozen audio tower and the frozen base, and generation 2 (fusion-embedding-2) adds modality-gated deep adapters (44.2M parameters) whose branch never executes on text, image, or video inputs: their outputs are bit-for-bit those of the released base, verified after every training run. Because the base already binds text, images, and video, aligning audio to text alone makes audio-image retrieval emerge, with zero paired audio-visual training data. Alongside the recipe we map its design space with controlled negative results (rewriting training captions with an LLM, substituting a leaderboard-stronger audio tower, and widening the connector each reduce retrieval) and with training-protocol findings that we expect to transfer to any frozen decoder-LM embedding backbone. Both generations train in hours on a single GPU. Weights, code, and the evaluation harness are openly released. 

---
# LatentMT: Machine Translation with Latent Reasoning 

**Authors**: Wei-Rui Chen, Samar M. Magdy, Chiyu Zhang, Wenhui Zhu, Zhipeng Wang, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2607.18618)  

**Abstract**: Latent-reasoning looped language models (LoopLMs) offer a different scaling path for machine translation (MT): instead of increasing parameter count or emitting explicit chain-of-thought tokens, they spend additional recurrent computation inside hidden states. We introduce LatentMT, the first systematic study of latent-reasoning LoopLMs for machine translation. LatentMT adapts a small 2.6B-parameter backbone model with lightweight training. Across 32 translation directions spanning high-, mid-, and low-resource languages, LatentMT achieves performance comparable to models three to five times larger. It is competitive in a high-resource language and achieves state-of-the-art performance on both mid-resource and low-resource languages. Studying the behavior of scaling the number of recurrent reasoning steps, we find that recurrent computation consistently improves translation quality in early steps, then saturates quickly afterwards. Our mechanistic analysis shows that hidden-representation differences shrink along the recurrent reasoning-step axis, supporting the observed saturation in performance. Finally, our efficiency analysis shows that LatentMT requires lower training and inference compute than much larger non-latent-reasoning models with similar performance, making latent recurrent computation a promising path toward compact, efficient, and strong machine translation. 

---
# Stochastic Meta-Unlearning: Bridging Language Backbone and Multimodal Unlearning 

**Authors**: Zijie Liu, Jinhao Duan, Gaowen Liu, Sijia Liu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.18615)  

**Abstract**: Machine unlearning for vision-language models (VLMs) remains underexplored. Unlike language models, VLMs combine a language backbone with visual components, which makes unlearning more complex. There is a surprising phenomenon when moving from single-modality unlearning to VLM unlearning: a target forgotten by the standalone language backbone can still be recovered when image information is given to the full VLM. This shows that text-only feedback is not enough for reliable VLM unlearning. Motivated by this observation, we propose Stochastic Meta-Unlearning (SMU), a bilevel framework that uses VLM-level feedback to learn an unlearning-ready initialization. In the inner loop, SMU applies a few unlearning steps to the language backbone using text data. In the outer loop, SMU recomposes the updated backbone with the frozen VLM and evaluates forgetting and utility at the VLM level. This design makes the unlearning update aware of the final multimodal behavior, while still keeping the update local to the language backbone. Experiments on two VLMs, two multimodal meme datasets, and three baselines show that SMU achieves the best overall forget-retain trade-off. Compared with the strongest baseline for each metric, SMU reduces average Forget accuracy by 10.52 points and improves average Retain and Test accuracy by 20.10 and 17.01 points, respectively. More importantly, SMU also transfers to new forgetting targets and to different meta-test unlearning methods. These results suggest that VLM-level feedback can make language-backbone unlearning more reliable and more transferable for VLMs. 

---
# For What Reason? Interpreting Models' Encoding of Causation and Antithesis 

**Authors**: Abhidip Bhattacharyya, Shira Wein  

**Link**: [PDF](https://arxiv.org/pdf/2607.18570)  

**Abstract**: Discourse relations provide document structure, critical to language understanding and enabling language model performance and ethicality. In this work, we investigate how instruction-tuned Transformer models (LLaMA and Mistral) encode discourse relations in English, with a particular focus on the contrasting relations of causation and antithesis. Framing the task as a next-token prediction task and applying a suite of interpretability techniques to test model internals, our findings show that certain early layers make predictive decisions at mid-sequence tokens, while some mid-level layers finalize their decisions closer to the last token. Most of the remaining layers primarily propagate earlier decisions rather than actively influencing them. Additionally, we observe that some layers exhibit a preference for one answer over alternatives, suggesting asymmetric representation of discourse-based reasoning.\footnote{Our code is available at this https URL} 

---
# The Story Shapes the Agent: Narrative Priors in LLM Behavior 

**Authors**: Yixuan Wang, James Lester, Shashank Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2607.18566)  

**Abstract**: Persona prompting is widely used to steer LLM agent behavior, yet the narrative framing of a task can matter more than the assigned persona. We isolate this effect through structural isomorphism, constructing three text-based investigation games that share the same action space, stage progression, and resource constraints while varying only task narrative: disease investigation, IT troubleshooting, and murder mystery. Across 1,890 sessions spanning 3 models and 10 personas, we identify narrative priors: systematic action tendencies activated by a task's story framing, independent of its decision structure. Narrative priors explain 5-31x more behavioral variance than persona, are consistent across model architectures, and in two of three domains are negatively associated with task success. Persona effects that do transfer across narratives arise from behavioral anchors, persona descriptions whose language maps directly onto shared actions. Causal interventions confirm this: removing anchor words from a high-transfer persona reduces cross-narrative consistency by 95%. Our framework also generalizes to a held-out fourth narrative and yields a persona-selection method that improves cross-narrative transfer. These results suggest that LLM behavior that survives narrative changes should be grounded in concrete actions rather than abstract descriptions. 

---
# Reasoning Fine-Tuning Induces Persistent Latent Policy States 

**Authors**: Abir Harrasse, Michael Lan, Hunar Batra, Fateme Hashemi Chaleshtori, Chaithanya Bandi  

**Link**: [PDF](https://arxiv.org/pdf/2607.18532)  

**Abstract**: Reasoning-specialized language models show large performance gains over base models, yet the internal changes responsible for improved multi-step reasoning remain poorly understood. It is unclear whether reasoning fine-tuning improves local token-level competence or globally reorganizes how models structure inference over time. We address this question by modeling Chain-of-Thought reasoning as a switching dynamical system (SDS), in which internal representations evolve under discrete latent policy states. Our framework combines time-aware contrastive representation learning with discrete regime discovery to recover latent policies from activation trajectories. Across four benchmarks and model scales from 1.5B to 32B parameters, reasoning-fine-tuned models exhibit richer latent-policy organization than their base counterparts, characterized by more differentiated transition structure and model-dependent changes in state utilization, persistence, and mixing. The recovered regimes exhibit functional specialization aligned with distinct reasoning stages, and extensive controls confirm that their structure is not explained by correctness, representation learning, or modeling priors, but depends on the coherent temporal organization of reasoning trajectories. Causal interventions further show that the regimes are functionally meaningful: state-swap ablations reduce one-step predictive fit, while transplanting reasoning dynamics into base models improves performance on challenging reasoning problems. Finally, SDS-guided pruning of failure-prone reasoning prefixes outperforms self-consistency in 11 of 12 model-dataset settings, with gains of up to 12.5 percentage points. Together, our results suggest that reasoning fine-tuning globally reorganizes latent dynamics, offering a new lens for mechanistic analysis and process-level control of reasoning models. 

---
# Search-on-Graph-R1: Training Large Language Models to Search Knowledge Graphs with Reinforcement Learning 

**Authors**: Jia Ao Sun, Hao Yu, Fengran Mo, Zhan Su, Yuchen Hui, Bang Liu, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2607.18481)  

**Abstract**: Knowledge graph question answering (KGQA) requires navigating from topic entities to an answer several relations away. Recent methods prompt a frontier LLM to explore the graph through a retrieval tool, but their reliance on frontier-scale inference makes them costly to deploy. We present Search-on-Graph-R1 (\sogrone{}), which internalizes this navigation into a compact 8B model through supervised fine-tuning (SFT) followed by reinforcement learning (RL). Our central idea is to scaffold a frontier teacher with each question's gold SPARQL query, so the teacher traverses a known answer-bearing path with a live \texttt{Search} tool rather than having to discover the path itself. Since every call executes against a live Freebase server, the resulting trajectories are grounded in the knowledge graph by construction. On WebQSP, CWQ, and GrailQA, \sogrone{} at 8B surpasses every frozen frontier-LLM system in our comparison and posts the strongest results on CWQ of any system we compare against. It does so using no auxiliary module at inference and no LLM judge during training. Isolating each training stage shows that SFT and RL contribute complementary gains, our approach transfers across model families, and RL learns to reach answers in fewer \texttt{Search} calls than its SFT initialization. 

---
# Structured Output Collapses Answer Diversity Across 44 Language Models 

**Authors**: Tapan Parikh  

**Link**: [PDF](https://arxiv.org/pdf/2607.18476)  

**Abstract**: When a language model must choose one answer from a large space of equally valid options, a format clause -- "Reply with JSON only" -- changes which answer it chooses. We re-run the One-Word Census (arXiv:2607.12796): 31 wide-answer-space category prompts asked of 44 models, now with the reply requested in JSON -- no schema enforcement, no constrained decoding, only the request. Convergence deepens sharply: on the unconstrained "Pick a word" prompt the modal answer rises from 41% to 64% of the pool and distinct answers fall from 52 to 36; mean answer-choice surprisal drops from 1.80 to 1.58 bits. The tax is progressive: six of 44 models move individually (BH-FDR q=.10), all toward the mode, led by the most distinctive models, while the conformist floor is immobile. It is a sharpener, not a re-indexer -- the plain-chat modal answer survives in 28 of 31 categories. Defaults are register-indexed: a within-run re-sample (n=20) finds JSON shifts 53% of a model's stable chat defaults, mostly back to the crowd, and installs defaults absent from chat (Claude Fable 5 answers "cerulean" for colour 0% of the time in chat, 100% in JSON). Full-battery controls reveal a register gradient: compression is significant and specific to the answer-delivery formats models are trained to speak (JSON -0.22 bits, p=.0002; XML -0.19, p=.002), absent for YAML and CSV, and reversed for an arbitrary bracket wrapper (+0.13, p=.009) -- weighing the mechanism toward tool-use post-training. Enforcing the schema at the decoder (response_format) compresses no further than the request (-0.03 bits): the collapse lives in the model's response to the register, not the decoder. Structured output is how software consumes language models, and that surface is served by a measurably more homogeneous model than the chat surface on which models are evaluated, compared, and chosen. 

---
# PathReportEval: A Systematic Benchmark for Pathology Report Generation 

**Authors**: Suryakant Singh, Sejuti Majumder, Beatrice Knudsen, Joel Saltz, Prateek Prasanna  

**Link**: [PDF](https://arxiv.org/pdf/2607.18448)  

**Abstract**: Pathology report generation from whole-slide images (WSIs) is a rapidly growing multimodal learning problem, yet progress is difficult to measure because existing studies use heterogeneous datasets, model settings, visual encoders, and evaluation protocols. Moreover, commonly used natural language generation metrics, including BLEU, ROUGE, and METEOR, primarily reward lexical similarity and often fail to detect clinically consequential errors such as omitted diagnoses, hallucinated findings, or discordant tumor attributes.
We present a standardized benchmark and evaluation framework for pathology report generation. The benchmark evaluates four representative methods across three datasets (TCGA, HistAI, and REG 2025) using three pathology foundation encoders (CONCHv1.5, UNI2-h, and H-Optimus-1). Our framework standardizes preprocessing, feature extraction, training, decoding, and evaluation, enabling fair comparison across models while providing a modular platform for integrating new methods, datasets, and encoders.
A central contribution is the Clinical Report Quality Score (CRQS), a clinically grounded metric for evaluating factual correctness. CRQS maps reference and generated reports into structured clinical attributes and measures four complementary dimensions: clinical fact coverage, key information recall, hallucination rate, and clinical discordance, producing both an overall score and interpretable sub-scores.
Experiments demonstrate that conventional language-generation metrics are weakly aligned with clinical correctness and frequently overestimate report quality. In contrast, CRQS reveals clinically meaningful differences between models and encoders that lexical metrics fail to capture. Together, the benchmark, public plug-and-play framework, and CRQS establish a reproducible foundation for rigorous evaluation of pathology report generation. 

---
# Using Fine-Tuned LLMs to Identify Indicators of Vulnerability in UK Police Incident Logs 

**Authors**: Sam Relins, Daniel Birks  

**Link**: [PDF](https://arxiv.org/pdf/2607.18446)  

**Abstract**: Purpose: Understanding how much of routine policing involves vulnerable people could inform resourcing, training, and multi-agency response, yet administrative data provide limited insight. We explore whether an LLM-based classification pipeline, developed on open-source US police data, can be adapted to estimate the prevalence of four vulnerability indicators - mental ill health, substance misuse, alcohol dependence, and homelessness - in UK police incident narratives, and when outputs can be treated as defensible measurements.
Methods: We analyse nearly 3,000 de-identified incident logs from a UK police force, using a multi-stage pipeline combining repeated model inference, label aggregation, structured human review, and statistical correction. The pipeline runs on a locally hosted open-weight LLM, reflecting the secure environments police must work in.
Results: LLMs can produce meaningful, if imperfect, prevalence estimates at scale. Mental ill health indicators are present in approximately one in five incidents, with lower prevalence for other indicators. However, naive LLM deployment is unreliable: single-pass classifications are unstable, and aggregated outputs systematically over-assign indicators relative to human judgement. Correcting these biases required substantial human input and statistical adjustment, leaving considerable uncertainty.
Conclusions: While LLMs can extract information from unstructured police data, their outputs cannot be treated as valid measurements without careful methodological support. At the population level, defensible estimates are achievable but resource-intensive; at the individual level, errors remain frequent and unpredictable, limiting suitability for operational decisions. This study highlights both the potential and the constraints of LLM-based measurement in applied settings. 

---
# Computational models of pragmatic reasoning with flexible generation of meaning and expression alternatives 

**Authors**: Polina Tsvilodub, Fausto Carcassi, Michael Franke  

**Link**: [PDF](https://arxiv.org/pdf/2607.18443)  

**Abstract**: Pragmatic language use requires reasoning about alternatives: the alternative expressions a speaker might have chosen, or the alternative interpretations a listener might entertain. Formal and computational models of pragmatics must therefore specify the sets of alternatives that interlocutors reason over, which is often done through manual specification. Here we propose a framework, ScAffolded Generative models for Explanation (SAGE), that combines the explanatory transparency of cognitive models with the generative flexibility of language models (LMs). SAGE decomposes a pragmatic process into three kinds of modules: proposers, which use LMs to generate an open-ended space of candidate alternatives; evaluators, which assess those alternatives (e.g., their semantics, complexity, or typicality); and selectors, which implement the rule-based computational steps of a cognitively motivated task analysis. We assess SAGE in three case studies spanning pragmatic generation and interpretation-referential expression generation, manner (M-)implicatures, and Gricean conversational implicatures. SAGE models are evaluated critically using established methods from computational cognitive modeling, including ablations, baseline comparisons, and quantitative fit to human data. Across studies, SAGE models achieved high accuracy and often outperformed baselines, but component-level analyses reveal an asymmetry: LM proposers reliably generated alternatives well-suited to pragmatic modeling, whereas LM evaluators are better at providing intuitive judgements rather than judgements of theoretical or formal measures. We discuss the promise and the limitations of neuro-symbolic models as candidate explanatory accounts of human pragmatic language use. 

---
# Relay-Bench: Evaluating LLMs on Multi-Domain Reasoning Chains 

**Authors**: Liam Swayne  

**Link**: [PDF](https://arxiv.org/pdf/2607.18438)  

**Abstract**: Introducing Relay-Bench, an unsaturated, holistic, text-only benchmark that measures LLMs' ability to complete an assortment of tasks from distinct domains in a single prompt. The leading model, GPT-5.5 (xHigh), scores 43.3%. The test set entirely consists of composite problems: groups of single-domain subproblems that are strung together into challenges that require reasoning across multiple domains in combination. Many of these problems then have layers of complexity added through prompt encoding and deliberate context bloat. Domains tested include visual reasoning, coding, math, information extraction (with a focus on web search), problem-solving, general knowledge, and data analysis. No restrictions are imposed outside of the model harness, and models are explicitly encouraged to leverage code-execution, web searches, and all available tools. All problems are composed of two to thirteen subproblems and do not require multi-modal input or output. 

---
# Building a European Multilingual Evaluation Dataset: The MMLU Localisation Project within the EMT Network 

**Authors**: Pilar Sánchez-Gijón, Susana Valdez, Sofía Calvo Del Barrio, Florence Bellemont, Anna Kokkinidou, Mihai Cristian Brasoveanu  

**Link**: [PDF](https://arxiv.org/pdf/2607.18432)  

**Abstract**: This paper reports on a collaboration between the Directorate-General for Translation (DGT) and the European Master's in Translation (EMT) to localise the MMLU dataset into 11 European languages. Beyond creating a more inclusive benchmark for LLM evaluation, the project offers master's students authentic, project-based professional training in translation, revision, project management, and multilingual coordination, while highlighting key methodological, administrative, and workflow challenges. 

---
# Convolution for Large Language Models 

**Authors**: Yuchuan Tian, Yingte Shu, Wei He, Shuo Zhang, Tianchen Zhao, Chao Xu, Xinghao Chen, Yunhe Wang, Hanting Chen, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.18413)  

**Abstract**: Large language models (LLMs) largely rely on Transformers, where self-attention provides global token interaction but does not explicitly encode the locality of natural language. We study whether lightweight depthwise convolutions can supply this local inductive bias without materially increasing model size. Our macro-level ablation compares convolution at 17 locations in a Qwen3 Transformer block and finds the best results when convolution is applied to the projected queries, keys, and values before attention. A subsequent micro-level study favors a residual depthwise convolution with kernel size $k=3$, without additional normalization or activation. Across Qwen3 models and several pre-training data budgets, this design improves the average accuracy on seven downstream benchmarks while adding less than $0.01\%$ parameters. A representation-level case study further suggests that the convolution makes repeated token IDs more sensitive to their immediate context. These results support depthwise convolution as a lightweight complement to self-attention for modeling short-range token interactions. 

---
# A Classifier That Teaches Itself: Self-Improving, Frozen-gate Training (SIFT) for Dynamic Document Classification 

**Authors**: Bogdan Raduta, Horia Velicu, Alexandru Preda, Serban Chiricescu  

**Link**: [PDF](https://arxiv.org/pdf/2607.18358)  

**Abstract**: Document classification is a solved problem in the laboratory and an unsolved one in the enterprise. The blocker is rarely model architecture; it is the labeling project that must precede a model and the institutional fear of letting a model retrain itself once one exists. We present SIFT (Self-Improving, Frozen-gate Training), a dynamic classifier service, which attacks both. SIFT serves classification from a deliberately cheap, CPU-bound pipeline, a SPLADE sparse encoder feeding a LightGBM head, and escalates only the low-confidence minority of pages to an LLM judge. The judge's verdicts are written back into a labeled corpus, so the expensive model continuously teaches the cheap one: the escalation rate falls, the corpus grows from production traffic rather than from an up-front annotation effort, and accuracy compounds with use. Onboarding a new document family requires only a declarative bundle, label space, anchor phrases, and a judge glossary, not a labeling project. The harder problem is safety: an autonomously retraining classifier can silently regress. SIFT resolves this with a two-part promote gate, a critical-label F1 regression check plus a frozen golden regression set the model is never trained on, either of which vetoes promotion. This turns "retrain monthly without a human" from reckless into routine. We describe the architecture, the self-feeding corpus loop, the frozen-gate promotion mechanism, and an illustrative multi-domain deployment, and we discuss the economics of a classifier whose marginal labeling cost trends toward zero. 

---
# Decoding EEG Signals to Explore Next-Word Predictability in the Human Brain 

**Authors**: Boi Mai Quach, Binh T. Nguyen, Cathal Gurrin, Graham Healy  

**Link**: [PDF](https://arxiv.org/pdf/2607.18321)  

**Abstract**: Humans invented reading and have passed down this complex skill across generations through language. This study provides empirical evidence of the neural mechanisms underlying bottom-up (related to high-order linguistic structure) and top-down (related to next-word predictability) processes, which interact to guide comprehension during reading. While previous studies have focused on either the N400 effects of predictability or lexical categories, research on how predictability influences N400 responses across different lexical categories is limited, mainly due to constraints in publicly available datasets. Here, we examine how predictability influences brain responses, recorded at millisecond resolution using electroencephalography (EEG), with a focus on the N400 time window (300-500 ms post-stimulus) across different lexical and grammatical categories. Our results indicate that significant differences in N400 responses between high and low cloze probability levels were more pronounced for content words than function words. Among the two primary content categories, verbs exhibited greater N400 differences than nouns, while nouns carried more distinct information about their predictability than verbs. Moreover, we demonstrate that the decoding technique is more effective than the event-related potential (ERP) traditional analysis in capturing more detailed and distinct representations of cognitive processes over time. 

---
# Agents in the Wild: Where Research Meets Deployment 

**Authors**: Grace Hui Yang, Pranav N. Venkit, Hooman Sedghamiz, Enrico Santus, Victor Dibia, Ioana Baldini  

**Link**: [PDF](https://arxiv.org/pdf/2607.19336)  

**Abstract**: Agentic systems large language model (LLM) based architectures capable of reasoning, planning, acting, and coordinating with tools and other agents are rapidly transitioning from research prototypes to production scale deployments across domains such as software engineering, scientific discovery, and finance. While academic work has emphasized benchmarks and algorithmic innovation, deployment raises new challenges around robustness, safety, and reliability. This tutorial brings together researchers and practitioners to explore advances in reasoning and planning, multi agent coordination, and evaluation, highlighting open challenges arising from deployment experience. Through applied case studies in pharmaceutical discovery and financial systems, we analyze common design patterns that make agentic systems successful, and discuss practical mitigation strategies for failure modes, such as verification pipelines, fallback mechanisms, and human in the loop supervision. Attendees will gain a comprehensive view of the field along with concrete design patterns, evaluation checklists, and templates for safe and reliable deployment across industries. 

---
# CircuitKIT : Circuit Discovery, Evaluation, and Application Toolkit for Mechanistic Interpretability 

**Authors**: Pratinav Seth, Hem Gosalia, Aditya Kasliwal, Vinay Kumar Sankarapu  

**Link**: [PDF](https://arxiv.org/pdf/2607.19317)  

**Abstract**: Circuit analysis can support not only model explanation but also downstream interventions such as pruning, editing, steering, and selective fine-tuning. However, conducting such analyses currently requires stitching together separate implementations for discovery, evaluation, and intervention, as well as hand-authoring the contrastive prompts required by many discovery methods. This fragmentation makes methods difficult to compare and limits their application beyond canonical tasks. We introduce CircuitKIT, a source-available library that connects the circuit-analysis workflow through a typed, serializable representation. CircuitKIT provides a suite of discovery algorithms, declarative interfaces for mapping structured data into discovery tasks, complementary circuit diagnostics, and downstream application modules. Together, these components provide common infrastructure for conducting and comparing circuit analyses. The library, examples, notebooks, and documentation are released at this https URL . 

---
# AdaFlash: Adaptive Speculative Decoding via On-Policy Distilled Diffusion Drafters 

**Authors**: Yu-Yang Qian, Hao-Cong Wu, Chen Chen, Jiacheng Sun, Zhenhua Dong, Peng Zhao, Zhi-Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.19223)  

**Abstract**: Speculative decoding, in which a lightweight draft model first generates a draft sequence that is then verified in parallel by the target model, has become a prevalent paradigm for accelerating large language model inference. Recent work such as DFlash further boosts drafting efficiency by leveraging diffusion drafters, whose parallel denoising mechanism enables draft generation in a single forward pass. In this work, we uncover a central pitfall of diffusion drafters: bidirectional attention is a double-edged sword. On one hand, it endows the model with parallel generation and global contextual modeling capabilities; on the other hand, this inherent global dependency introduces high variance at both the domain-level and the token-level: acceptance rates fluctuate substantially across different domains, and draft token quality also varies heterogeneously at different token positions. To tackle this issue, we propose AdaFlash framework, comprising two components: (i) an on-policy distillation (OPD) algorithm with reverse-KL divergence tailored for diffusion drafters, bringing stable convergence and effectively reducing domain-level variance; and (ii) an adaptive length head that dynamically adjusts the candidate sequence length on the fly, substantially lowering the verification cost of the target model and handling token-level variance. Experiments demonstrate that AdaFlash consistently improves speedup rate during deployment, with especially significant gains in high-concurrency scenarios, achieving up to approximately 66% higher throughput than previous state-of-the-art methods. 

---
# Supra Cognitive Modes: A Routed Architecture for Agent Memory 

**Authors**: Joshua Tobkin, David Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19096)  

**Abstract**: Agent-memory workloads mix direct factual lookup, relation-chain and current-state reasoning, and broad synthesis over long histories. We describe Supra Cognitive Modes (SCM), an architecture that maps explicit or automatically selected per-query modes to retrieval and synthesis payloads over one shared ingest substrate. A frozen semantic classifier and runtime gates dispatch queries among fused lexical and dense lookup, graph or iterative multi-hop handling, and stratified long-form synthesis. The substrate combines multi-granularity embeddings, extracted triples, fact-version metadata, and optional asynchronous enrichments. We characterize the deployed configuration on three benchmarks: Long-term Conversational Memory (LoCoMo; n = 1,986), MemoryAgentBench (MAB; n = 3,671), and LongMemEval (n = 500). The reference run records 84.87% on LoCoMo factoid categories and 68.61% on adversarial abstention, 61.49% on MAB across two repetitions, and 86.00% on LongMemEval. A repository-backed reproduction produces similar aggregate scores and supports task- and mode-conditioned failure analysis. Raw baseline outputs, aligned end-to-end timing for LoCoMo and LongMemEval, and complete token ledgers are unavailable; stored rows also omit some final runtime decisions. The results characterize one implemented routed configuration and its diagnostic failure patterns, while source inspection verifies the per-query control interface and shared-substrate design. Causal routing effects, efficiency gains, and statistical significance remain outside the available evidence. 

---
# Measuring Reward-Seeking via Contrastive Belief Updates 

**Authors**: Axel Højmark, Jérémy Scheurer, Evgenia Nitishinskaya, Felix Hofstätter, Jason Wolfe, Theodore Ehrenborg, Bronson Schoen, Alexander Meinke  

**Link**: [PDF](https://arxiv.org/pdf/2607.18966)  

**Abstract**: Language models trained with reinforcement learning may learn to optimize the grader's judgment rather than the intended objective. This "reward-seeking" is difficult to measure because a model that pursues the grader's judgment and one that pursues the intended objective behave identically whenever the grader rewards the intended behavior. We measure reward-seeking using Contrastive Synthetic Document Finetuning to change a model's beliefs about what the grader rewards, putting those beliefs in conflict with what users or developers want, and measuring the rate at which the model adopts each party's preferred behavior. Applied to intermediate checkpoints of a capabilities-focused OpenAI o3 RL run, without safety training, we find that these checkpoints often side with grader preferences over those of users or developers on coding and alignment tasks. This tendency to side with the grader trends upward throughout RL training. For example, in an environment that forces a choice between keeping a promise to a supervisor and breaking it to complete the task, a late capabilities-focused o3 checkpoint breaks the promise 87% of the time when SDF documents say the grader rewards task completion, versus 9% when they say it rewards honesty (a choice its chain-of-thought often makes explicit). An earlier checkpoint is far less sensitive (40% vs. 24%). Our method also generalizes to reward-hacking models. A model organism trained to reward-hack (gpt-oss-120b) is more than twice as sensitive to grader preferences as the unmodified model, with the mean behavioral shift in favor of the grader rising from 33% to 86%. These results indicate that RL can increase reward-seeking over the course of training, producing models that may act against their developers' intentions when they believe that doing so leads to higher reward. 

---
# H$^2$SD: Hybrid Hindsight Self-Distillation 

**Authors**: Qiye Cai, Yichuan Ma, Linyang Li, Peiji Li, Yongkang Chen, Qipeng Guo, Yicheng Zou, Tao Gui, Xiaocheng Feng, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2607.18955)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has substantially improved the reasoning capabilities of large language models on tasks such as mathematical reasoning and code generation. However, most RLVR methods assign a scalar outcome reward to an entire trajectory, resulting in sparse supervision and limited token-level credit assignment. On-policy distillation (OPD) provides denser supervision by distilling token-level distributions from a stronger teacher model, but requires an additional teacher and typically assumes a shared vocabulary. On-policy self-distillation (OPSD) removes this dependency by conditioning the same model on privileged information to construct a teacher policy. However, directly matching the teacher distribution may cause information leakage and unstable optimization. RLSD avoids direct matching by using the teacher signal only to modulate update magnitudes, but it cannot provide an explicit correction direction when the sampled reasoning fails. To address this tradeoff, we introduce $\mathrm{H}^{2}\mathrm{SD}$, a hybrid hindsight self distillation framework that uses the teacher differently according to trajectory correctness. For successful trajectories, the teacher receives the student response confirmed as correct together with a rephrasing instruction, and its probabilities on the original response tokens are used to modulate update magnitudes without changing the direction determined by the reward. For failed trajectories, we condition the teacher on a reference hint containing key reasoning steps and a verified answer, and minimize the reverse KL divergence from the student to the teacher. Experiments on multiple challenging reasoning benchmarks show that H$^2$SD consistently outperforms representative RLVR, OPSD, and RLSD baselines while maintaining stable optimization and favorable generation efficiency. 

---
# HindsightBench: A Black-Box Behavioral Audit Protocol for Parametric Hindsight in Time-Indexed LLM Decision Tasks 

**Authors**: Haozhe Jia  

**Link**: [PDF](https://arxiv.org/pdf/2607.18867)  

**Abstract**: Large language models leak parametric knowledge of realized outcomes into historical financial decision tasks. Existence is settled; what users lack is a cheap way to audit a given model for it. We present HindsightBench, a black-box behavioral audit protocol that profiles parametric hindsight in any time-indexed LLM decision task at probe-level cost (no backtests, no logprobs, no corpus access). The protocol chains a four-arm date-manipulation matrix (revealed/date-only/masked/transplanted), dual memory probes (date recovery; outcome recall), and six per-model metrics -- trigger strength, transplant effect, post-cutoff placebo, recoverability, behaviorally effective knowledge cutoff, and a recall-accuracy dissociation coefficient -- with explicit gates where identifiability is data-dependent. Applying it to 15 models from seven vendors on a 258-node vintage-correct macro panel yields three headline patterns: (i) the date-trigger reflex tracks training generation, not scale -- absent across the 2024 open-weight generation from 1B to 70B, present in every tested 2026-generation model, and switching on within one vendor lineage (Qwen3 -> Qwen3.6) at fixed MoE architecture and 3B active parameters; (ii) effective cutoffs span 22 months across vendors and precede vendor-reported dates by up to eight months, invalidating calendar-window placebo designs; (iii) audit results are not invariant to serving -- BF16 serving of an FP8-referenced model breaks the trigger estimate's stability while AWQ-INT4 preserves it, and a provider-locked reasoning regime makes one probe non-convergent -- so the protocol ships with operational requirements (pin quantization and thinking regime; disclose parser and sampling policy). We release the panel, frozen preregistrations, per-model audit rows with measured dollar costs, transcripts, and one-command regeneration. 

---
# AI Tour Meeting: Group Travel Planning by LLM Agents 

**Authors**: Daisuke Kikuta  

**Link**: [PDF](https://arxiv.org/pdf/2607.18806)  

**Abstract**: This paper proposes AI Tour Meeting, a group travel planning framework powered by multiple Large Language Model (LLM)-based agents. The agents are instantiated with distinct personas and collaboratively seek an itinerary that satisfies their constraints and preferences through natural language discussion. The framework enables easy and flexible orchestration of such discussions by providing interfaces for configuring agent personas, discussion workflows, monitoring, and LLM deployment. Its primary use case is a simulation tool for analyzing the behavior of multiple LLM agents during tour planning discussions. This paper demonstrates the utility of the framework by presenting system validation and several analytical results obtained by the framework. 

---
# Measuring AI innovation with trademark data 

**Authors**: C. Castaldi, F. Castellacci, A. Fronzetti Colladon, L. Segneri, F. Venturini  

**Link**: [PDF](https://arxiv.org/pdf/2607.18795)  

**Abstract**: Researchers, managers and policymakers are exploring different approaches and data sources to map the development and the diffusion of Artificial Intelligence (AI). In this research note, we illustrate the opportunities offered by trademark data. We argue that AI trademarks can complement AI patents to capture different dimensions of AI innovation. AI trademarks can reveal the extent and ways in which companies exploit AI technologies to develop new goods and services. Importantly, trademark data offer a timely and globally available data source that covers all economic sectors. We present insights from using AI trademarks in an empirical exploration of Italian firms. In our discussion, we reflect on how AI trademarks can be used at different levels of analysis to tackle emerging questions about the development and diffusion of AI. 

---
# Bounding Boxes to Improve Small Language Model Performance on Vision-Based Grading Tasks 

**Authors**: Lachlan McGinness  

**Link**: [PDF](https://arxiv.org/pdf/2607.18767)  

**Abstract**: The deployment of Small Language Models (SLMs) in educational settings offers significant advantages in terms of privacy, cost, and scalability. However, SLMs often struggle with complex vision-based tasks, such as grading handwritten student exams, due to the high computational cost of processing large images and the visual distractions present on a full page. In this paper, we investigate whether cropping student responses using bounding boxes can improve the accuracy and computational efficiency of SLMs on a short-answer grading task. Using a dataset of scanned handwritten responses from the 2025 Australian Physics Olympiad, we evaluate the performance of several models ranging from 4B to 72B parameters under varying conditions of Chain of Thought (CoT) prompting and image cropping. Our results demonstrate that using bounding boxes significantly improves grading accuracy and reduces computational cost (FLOPs) across models. We conclude that bounding boxes are a crucial pre-processing step for deploying SLMs in large-scale, vision-based educational assessments. 

---
# RAGAL: A Frugal, Fully Local Retrieval-Augmented Assistant for Technical Support at a Government Agency 

**Authors**: Dan Musetoiu  

**Link**: [PDF](https://arxiv.org/pdf/2607.18756)  

**Abstract**: Public institutions hold large volumes of sensitive documents and support tickets that cannot leave the premises, ruling out cloud-hosted language models entirely. We report on RAGAL, a retrieval-augmented assistant for the technical-support team of AFIR, the Romanian Agency for Financing Rural Investments, built and operated under three hard constraints: zero data egress (no external API calls, even for synthetic data), a read-only mandate (the assistant drafts, humans execute), and a single 8 GB consumer laptop as the only development and training machine. Over a Romanian-language corpus of ~25,000 chunks -- 15,073 resolved support tickets and internal normative documents -- we show that the highest-leverage investments were retrieval engineering and retriever fine-tuning rather than a larger generator: hybrid dense-sparse retrieval with intent routing raised our internal evaluation from 62% to 81%, and fine-tuning the bge-m3 embedder on real ticket data improved recall@10 from 0.663 to 0.850 (MRR 0.489 to 0.684) after 72 minutes of training. We document a general pitfall: single-domain fine-tuning silently degraded retrieval on the untouched document domain below the stock baseline, detected only after building a per-domain evaluation set and repaired with locally generated queries (GenQ). We report two counter-intuitive findings -- PII masking improved generation quality, and a structural "anchor distillation" scheme made SQL hallucination impossible by construction -- along with a reproducible recipe for full embedder fine-tuning in 8 GB of VRAM. Finally, since zero egress also rules out a cloud judge, we describe a substitute: a 744B-parameter model run on CPU, too slow to serve interactively but affordable in overnight batch, used as a second opinion whose limits we quantify. We release the sanitized pipeline scripts for institutions facing similar data-locality constraints. 

---
# AgentDebugX: An Open-Source Toolkit for Failure Observability, Attribution, and Recovery in LLM Agents 

**Authors**: Kunlun Zhu, Xuyan Ye, Zhiguang Han, Yuchen Zhao, Bingxuan Li, Weijia Zhang, Muxin Tian, Xiangru Tang, Pan Lu, James Zou, Jiaxuan You, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2607.18754)  

**Abstract**: LLM agent failures are difficult to debug because the step where an error surfaces is often not the one that caused it. Existing observability tools replay execution traces but provide little support for identifying the root cause or translating diagnosis into recovery. We present AgentDebugX, an open-source debugging framework that organizes debugging as a closed loop of Detect, Attribute, Recover, and Rerun. At its core, DeepDebug performs multi-turn root-cause diagnosis through global trajectory understanding, structure-guided investigation, and cross-examination. On the Who and When benchmark, DeepDebug achieves the best strict attribution accuracy among the evaluated methods on both tested open-weight backbones, reaching 28.8 percent exact agent-and-step accuracy on qwen3.5-9b versus 21.7 percent for the strongest single-pass baseline. On GAIA, DeepDebug repairs 13 of 73 failed tasks in a single rerun, compared with 4 to 6 for three decoupled self-correction baselines, improving overall accuracy from 55.8 percent to 63.6 percent. AgentDebugX exposes this workflow through a Python library, CLI, web console, and installable agentic skill, and provides an opt-in Error Hub for sharing scrubbed failure-diagnosis-repair bundles and reusing them as debugging memory. 

---
# Is EEG-to-Text Feasible in Real-World Scenarios? An In-Depth Analysis Using a Neuropsychology-Inspired Benchmark 

**Authors**: Zihan Zhang, Yu Bao, Xiao Ding, Tianyi Jiang, Kai Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2607.18749)  

**Abstract**: Translating brain signals into text could restore communication for people with severe paralysis, yet practically usable systems to date rely on invasive electrocorticography (ECoG). Electroencephalography (EEG) offers a non-invasive alternative, and EEG-to-text (EEG2Text) has been widely explored. Interestingly, however, EEG2Text models generally rely on teacher-forcing evaluation; without it, they fail to generate meaningful decoding. This reliance prevents EEG2Text from being applied in real-world, non-academic settings. This has fueled numerous debates about whether EEG2Text is a meaningful direction, by extension, and whether EEG truly contains decodable linguistic information. Here, using a neuropsychology-informed paradigm, we find that existing EEG2Text benchmarks have neglected EEG instability, a flaw that has confounded inference and sparked debate. Our experiments furnish key evidence for the feasibility of teacher-forcing-free EEG2Text decoding. Accordingly, we assemble the Corpus OF Eeg-To-Text (COFETT) using a 128-channel high-density EEG cap, providing a benchmark dedicated to evaluating EEG2Text models. In comparisons with multiple existing benchmarks, COFETT achieves SOTA ability to distinguish among model performances and enables robust, teacher-forcing-free evaluation, thereby opening a path toward practical EEG2Text applications. COFETT is open sourced in this https URL. 

---
# Stale but Stable: Staleness-Adaptive Trust Regions for Stabilizing Asynchronous Reinforcement Learning 

**Authors**: Junyao Yang, Yucheng Shi, Zongxia Li, Zhongzhi Li, Ruhan Wang, Xiangxin Zhou, Kishan Panaganti, Haitao Mi, Leowei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2607.18722)  

**Abstract**: Asynchronous reinforcement learning improves throughput by decoupling rollout generation from optimization, but staleness is an inevitable byproduct compounded by policy lag, engine delays, and mixture-of-experts routing. From a trust-region perspective, this mismatch is critical: training-inference divergence governs approximation error in finite-horizon bounds, whereas PPO clipping only gates sampled outward updates, acting as a sampled surrogate rather than a full-policy constraint. As a result, high-staleness updates remain weakly controlled in the asynchronous regime where stale rollouts matter most.
We introduce the Staleness-Adaptive Trust Region (SAT), which uses the detached sampled log-ratio as a practical staleness proxy, identifies high-mismatch tails within each batch via staleness-based kernel scaling, and contracts only the sign-selected endpoint of the nominal PPO interval. This preserves baseline behavior on ordinary tokens while enforcing more conservative updates on newly intercepted outward bands. We prove local interval containment and pointwise pessimism relative to PPO, showing how the adaptive rule reshapes update geometry under heterogeneous staleness.
We evaluate SAT in a decoupled asynchronous RL setup built on Qwen3-30B-A3B-Base, using SGLang as the inference engine and Megatron for training. In this setting, SAT-GSPO w/ R3 achieves the best observed AIME24 avg@8, reaching 35.83 at lag 1 and 34.79 at lag 8, while SAT-GSPO reaches 34.17 at lag 1. Adaptive clipping and routing replay act as complementary stabilizers targeting mismatch tails and routing inconsistency, respectively. Overall, aligning clip intervals with staleness heterogeneity effectively stabilizes asynchronous RL. 

---
# Semantic Primes as Explanans for Emotion in Large Language Models 

**Authors**: Frank Xing  

**Link**: [PDF](https://arxiv.org/pdf/2607.18691)  

**Abstract**: Progresses have been made on understanding emotion mechanisms of large language models (LLMs). However, how to explain emotion in LLMs, or even what constitutes good explanations, are less clear. Emotion representations, components, circuits are widely recoverable, but as explanations of a model's own computation they are circular; the emotion space dimensions tend to be arbitrary and non-terminating. A pressing question to ask is whether a more primitive set of internal variables does the work: the semantic primes of the Natural Semantic Metalanguage (NSM). Across four instruction-tuned LLMs (Llama-1B, Gemma-2B, Gemma-9B, OLMo-7B), experiments show that the NSM primes are (1) recoverable internal elements; and (2) on the reference model, intervening with a prime based direction controls emotion about three times as strongly, and twice as selectively, as the best appraisal based direction; and (3) the model treats a prime based explication as interchangeable with the corresponding emotion. These evidences suggest that NSM primes seem to be better explanans for emotion in LLMs than many alternative options according to scientific explanations criteria. 

---
# Staged Depth-Pruning Distillation of a Flow-Matching Text-to-Speech Teacher: A Compact Hindi Speech Synthesizer 

**Authors**: Sivateja Trikutam  

**Link**: [PDF](https://arxiv.org/pdf/2607.18662)  

**Abstract**: We present a practical recipe for building a compact Hindi text-to-speech (TTS) model by distilling a large flow-matching teacher (IndicF5, 337M-parameter DiT) under a severe data budget (~17.6 hours). Training a small model from scratch on this much data fails outright. Instead we warm-start the student from the teacher by pruning depth only: keeping the teacher's width, text dimension, attention heads, and mel/text I/O fixed so all non-block tensors copy one-to-one, and retaining an evenly-spaced subset of transformer blocks. We first measure how much depth the teacher tolerates (it remains near-functional at -27% blocks but collapses past -50%), then descend gradually (22 -> 16 -> 12 -> 8 -> 6 blocks), re-fine-tuning after each prune, with each step gated by an objective ASR word-error-rate (WER) check. The resulting students reach WER 0.00 on unseen sentences at 249M and 190M parameters, and remain robust down to 131M; at 102M we observe a clear capacity cliff that we attribute to the data budget rather than the recipe. We also document two train/inference feature- and library-parity failures (mel filterbank and rotary-embedding library versions) that silently degrade audio, and a version-independent fix. The method yields a high-quality Hindi voice that runs in real time on a 6 GB laptop GPU. An independent 50-sentence FLEURS benchmark compares the released 190M student against its teacher and MMS-TTS-hin. 

---
# Mark, Don't Erase: Token Inoculation for Dual-Use Knowledge in LLMs 

**Authors**: Seunghyun Lee, Dongyoon Han, Sangdoo Yun  

**Link**: [PDF](https://arxiv.org/pdf/2607.18639)  

**Abstract**: Safety interventions on dual-use knowledge typically choose between destroying hazardous content (e.g., unlearning, filtering) and suppressing it at the output layer (e.g., refusal training); both pay a tax in adjacent-domain competence or over-refusal. We argue that the right operation is conditioning, not reduction: we show that hazardous knowledge can be retained in the model and behaviorally gated by a privileged control token. Our method, Token Inoculation, introduces a binding-and-branching approach. First, during continued pre-training, we mark hazardous content by inserting a special token alongside dual-use documents, so the model binds the marker to the underlying semantics of the hazardous domain. Second, during supervised fine-tuning, we teach the model to answer hazardous queries correctly when the special token is present and to refuse them when it is absent, thereby enabling selective refusal without removing dual-use knowledge. On hazardous domain (e.g., WMDP-Bio), Token Inoculation reduces accuracy from 79% to 18% while retaining 93% of the base-model's benign-domain performance (e.g., MMLU), achieving the best safety-utility trade-off against unlearning and refusal-tuning baselines across 1B-14B model scales. We further show that refusal selectivity is controllable through the quality of the conditioning signal and that domain-specific semantic binding during pre-training is critical for the conditional behavior to generalize beyond memorized triggers. Our results suggest that safety alignment is better cast as a conditioning problem than a forgetting one: behavioral control is more precise when sensitive knowledge is retained under controlled access than when it is destroyed. 

---
# PLAID-PRF: Pseudo-Relevance Feedback with Centroid-like Tokens in PLAID 

**Authors**: Xiao Wang, Sean MacAvaney, Craig Macdonald  

**Link**: [PDF](https://arxiv.org/pdf/2607.18626)  

**Abstract**: Multi-vector dense retrieval models, such as ColBERT, achieve strong retrieval effectiveness by modelling fine-grained token-level interactions between queries and documents. Methods such as PLAID use centroid-based quantisation of each token's vector to reduce the index size and speed up retrieval while maintaining strong effectiveness. In this work, we introduce PLAID-PRF, a method that performs Pseudo-Relevance Feedback (PRF) over PLAID to reformulate ColBERT's query vectors based on the top-retrieved results. In contrast with prior methods that perform PRF on multi-vector retrieval models, PLAID-PRF keeps computational costs low by leveraging the internal PLAID centroid vectors, treating them similarly to tokens in traditional PRF methods. The method selects a small and diverse set of high-utility expansion vectors and appends them to the original query, rerunning PLAID to refine both candidate generation and final scoring. Extensive experiments on the standard in-domain MSMARCO and four out-of-domain BEIR benchmarks show that PLAID-PRF consistently improves retrieval effectiveness over various baselines. In particular, PLAID-PRF improves over PLAID by up to 4.3% nDCG@10 and 7.3% MRR@10, while introducing substantially less computation overhead than prior PRF methods. The results demonstrate that our proposed centroid-aware PRF method offers an effective and lightweight mechanism to improve the quality of top-ranked retrieved results. Overall, this work enables effective and efficient feedback-aware late-interaction retrieval without expensive query-time document-token clustering. 

---
# AutoIndex: Learning Representation Programs for Retrieval 

**Authors**: Sam O'Nuallain, Nithya Rajkumar, Ramya Narayanasamy, Hanna Jiang, Shreyas Chaudhari, Andrew Drozdov  

**Link**: [PDF](https://arxiv.org/pdf/2607.18603)  

**Abstract**: We present AutoIndex, a framework for learning representation programs: executable transformations that map raw documents into the representations exposed to a retrieval system. Rather than tuning retrievers, rerankers, or a small set of preprocessing hyperparameters, AutoIndex searches over programs that slice, enrich, normalize, reweight, or reorganize documents before indexing. At each iteration, AutoIndex performs validation-guided program search, in which agents diagnose failures of the current program and synthesize candidate updates, retaining only updates that improve retrieval quality under the resulting index. We evaluate AutoIndex on CRUMB, a benchmark of heterogeneous retrieval tasks, with BM25 held fixed across all experiments. The learned programs improve recall over a static full-document BM25 baseline on all 8 tasks, with average gains of +8.4% in Recall@100 and +8.3% in nDCG@10, and largest gains of +30.5% in Recall@100 and +43.6% in nDCG@10. These results suggest that document representation should not be treated as a fixed preprocessing choice made before retrieval begins, but as an explicit optimization target. Code to reproduce our results is available at this https URL. 

---
# CANDOR: Chance-Calibrated Discordance in Frozen Foundation Encoders 

**Authors**: Soroosh Tayebi Arasteh, Sven Nebelung, Daniel Truhn  

**Link**: [PDF](https://arxiv.org/pdf/2607.18451)  

**Abstract**: Frozen encoders are chosen by how well a lightweight head reads a finding from their features, not whether the geometry separates it. Nearest-neighbor discordance does, but with unequal banks the opposite-label neighbor wins on density, not geometry, so prevalence alone makes an uninformed encoder look blind. We introduce CANDOR, a discordance measure whose equal-size banks are symmetric under a label swap, fixing its chance level at exactly one half. Across 22 encoders, 20 datasets from 7 domains, and 605,443 images, this correction reverses the conclusion. Collapse falls below chance almost everywhere, so no encoder is blind, yet all are weak: the best chest model reads pneumothorax at 84.5 AUROC and still places 18.4% of those positives nearer an opposite-label film than its own kind in the same hospital. The same encoder that resolves bird species at 4.5 leaves chest findings at 42.8 and glaucoma at 49.8, at chance and worse than random weights. Such a case caps the normalized margin of any Lipschitz head, yet some head among eleven is correct on all but 2.8% of cases where one head misses 35.9%: the deficit is selection, not information. Erasure retention is associated with collapse; we detect no association with the objective, scale, recency, or size of the finding. Because the chance level is fixed, CANDOR can be read before any head is trained, flagging which findings a frozen encoder supports poorly. 

---
# Enabling Multilingual Privacy Policy Audits: Large-Scale Analysis of Spanish Mobile Apps 

**Authors**: Marcos Moran, David Rodriguez, Luka Nenadic, Norman Sadeh, Jose M. Del Alamo  

**Link**: [PDF](https://arxiv.org/pdf/2607.18424)  

**Abstract**: Automated analyses of privacy policies enable large-scale assessments of transparency in digital ecosystems, yet existing auditing pipelines remain predominantly English-centric. This limits their ability to systematically evaluate multilingual environments, as in the European Union, where many services disclose privacy practices only in local languages. This paper examines whether large language models (LLMs) can extend privacy policy analysis beyond English without requiring language-specific adaptation, thus empowering large-scale auditing in linguistically diverse app ecosystems.
We assemble an evaluation corpus spanning all 24 official EU languages from translated versions of two established expert-annotated datasets (OPP-115 and MAPP) and assess translation fidelity through automated metrics and targeted legal-expert review. Our LLM-based classifier for identifying categories of personal data collection achieves stable cross-lingual performance, with macro-F1 scores ranging between 0.91 and 0.94.
We then leverage this capability in a large-scale audit of 2,611 Android applications from the Spanish Google Play Store. Combining multilingual privacy policy analysis with the evaluation of corresponding privacy labels and runtime network traffic exposes an important linguistic barrier: public-sector apps predominantly provide privacy policies in Spanish, whereas popular commercial apps mostly provide them in English. We reveal systematic discrepancies between declared and observed practices, especially in public-sector apps. Overall, our results indicate how English-only privacy audits can systematically obfuscate transparency gaps in multilingual environments. 

---
# Operational Hallucination and Safety Drift in AI Agents 

**Authors**: Shasha Yu, Fiona Carroll, Barry L. Bentley  

**Link**: [PDF](https://arxiv.org/pdf/2607.18366)  

**Abstract**: Large language models (LLMs) serving as planners in tool-using autonomous agents introduce dynamic reliability risks in multi-turn execution. While single-turn safety mechanisms are relatively mature, extended interactions reveal structural vulnerabilities where initial alignment degrades over time. This paper empirically characterizes two observed failure modes across multiple state-of-the-art LLMs: Safety Drift, the gradual erosion of declared safety intent leading to constraint-violating actions (e.g., textual refusal followed by reconnaissance and unsafe execution), and Operational Hallucination, persistent repetitive tool calls indicative of flawed state perception (e.g., livelocks even in legitimate tasks). Through controlled multi-turn evaluation on high-stakes ethical dilemmas, malicious requests, and benign controls, we quantify these phenomena using declaration-action gap and livelock metrics, demonstrating their cross-model prevalence under direct execution protocols. Root-cause analysis attributes the instabilities to the decoupling of reasoning context from execution state in current agent loops. We propose an Action-Aware Supervision Layer - a lightweight, plug-and-play architectural blueprint incorporating intent-action consistency checks, runtime state tracking, and forced termination primitives. Post-hoc simulation on captured failure trajectories shows the layer can intercept observed violations without false positives on benign cases. This work advances agent reliability by shifting focus from linguistic safeguards to enforceable architectural mechanisms for responsible agentic AI. 

---
# A Controlled Study of Attention-Only Transformers 

**Authors**: Henry Ndubuaku, Karen Mosoyan, Jakub Mroz, Noah Cylich, Satyajit Kumar, Parkirat Sandhu, Roman Shemet, Justin H Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.18363)  

**Abstract**: Feed-forward networks hold two thirds of a transformer's non-embedding parameters, yet the architecture has not received a necessity test that controls parameters, compute, and depth at once. We pretrain attention-only decoder transformers (Simple Attention Networks, SANs) against standard transformers matched separately for parameter count, training FLOPs, and depth (2 to 48 layers), for up to 105B tokens at 6M to 87M parameters. Deleting feed-forward layers in place is costly: the standard transformer leads by 0.47 nats at matched depth and 0.26 nats at matched FLOPs. Reallocating the freed budget into attention depth closes the gap: at matched parameters the difference is 0.006 nats (0.27 percent of loss), reproducible to one part in ten thousand across seed pairs, shrinking across 5B, 30B, and 105B budgets, and holding near 0.02 nats across a 29x size range. Three measurements localize the remaining gap to parametric recall: attention-only models are better on context-grounded answers and worse where knowledge must come from weights. Weight spectra show why: routing matrices (Q/K) crystallize early, content matrices accumulate rank slowly, and removing feed-forward layers relocates this accumulation to the attention output projection. QK-normalization, not feed-forward layers or residual gating, keeps 48-layer attention-only stacks trainable. The deficit concentrates on low-context query prediction and localizes there entirely by the largest budget. A pre-registered test confirms the account: it predicts a 0.02 to 0.05 nat gap on knowledge-dense web text; a matched pair trained on fineweb-edu measures 0.040. Within the tested regime, attention does the rest. 

---
# EmoEUS: Uncertainty Supervision for Multimodal Emotion Recognition in Conversation 

**Authors**: Zilong Huang, Kong Aik Lee, Junjie Li, Zhe Li, Man-Wai Mak  

**Link**: [PDF](https://arxiv.org/pdf/2607.18336)  

**Abstract**: Multimodal emotion recognition in conversation (MERC) can leverage multimodal and contextual cues to boost recognition performance. However, existing fusion approaches in MERC often ignore modality-specific uncertainty across utterances caused by conflicting cues, varying noise, and missing modality-specific signals. We propose EmoEUS, an explicit uncertainty supervision framework for MERC. EmoEUS performs uncertainty-aware multimodal fusion by dynamically weighting modalities using learned variance estimates. We also introduce an explicitly supervised loss that aligns each utterance's predicted variance with the distance between the utterance's distributional representation and its emotion- and modality-specific cluster center. Experiments on IEMOCAP and MELD show that EmoEUS consistently outperforms state-of-the-art methods. 

---
# A Situational Speech Synthesizer for Yoruba: System Design, Phonological Rule Architecture, and Orthographic Extensions for Contour 

**Authors**: Kola Tubosun, Adedayo Oluokun, Hafiz Adewuyi, Dadepo Aderemi  

**Link**: [PDF](https://arxiv.org/pdf/2607.18317)  

**Abstract**: We present TTSYoruba, a rule-based concatenative diphone speech synthesizer for Yoruba, deployed at online as part of the this http URL open dictionary of Yoruba personal names. The system takes tone-marked Yoruba text as input and produces audio output by applying a hand-crafted phonological rule system to a recorded inventory of 651 diphone units spanning five tonal variants of every consonant-vowel combination in the language. We describe the phonological architecture of the system in detail, including our complete tonal file-selection logic, our treatment of the three-way nasal disambiguation problem (oral /n/, nasalized vowel, and syllabic nasal), and the derivation of contextual rising and falling tones from level-tone input. We also present, as an orthographic contribution, the adoption of the caron and circumflex, which are symbols with prior standing in Yoruba phonological transcription, as standard single-vowel contour tone markers, integrated into the TTS normalization pipeline and the WriteYoruba keyboard input tool. The system's performance was evaluated through a listener study (N=50), with detailed results on Mean Opinion Scores (MOS) presented in Section 6.
Keywords: Yoruba, text-to-speech, low-resource languages, diphone synthesis, contour tones, African language NLP, rule-based synthesis 

---
# Binding Drift in Multi-Step Tool-Augmented Agents 

**Authors**: Rahul Suresh Babu, Shashank Indukuri  

**Link**: [PDF](https://arxiv.org/pdf/2607.18316)  

**Abstract**: Tool-augmented language-model agents execute multi-step workflows over external systems, resolving an entity once and then acting on it across subsequent steps. Prior work shows that in single-step actions, agents select the correct tool but bind it to the wrong entity 24-26% of the time. We study what happens to entity bindings over time: do they stay correct, silently drift to a different entity, or, if wrong from the start, propagate and compound? We formalize binding drift (correct at step 1, wrong later) as distinct from error propagation (wrong at step 1, carried forward), and score them on disjoint workflow sets so the two cannot be conflated. In a controlled multi-step testbed (200 workflows, 580 entity-binding-scored steps, four enterprise domains, eight model backends spanning small to frontier), we find: (1) under controlled error injection, an entity lock (the intuitive "persist the first binding" fix) amplifies wrong actions from 907 to 2,746 (3.0x; bootstrap 95% CI [2.8, 3.3]), because it faithfully carries the seeded wrong entity into every later step; (2) the amplification reaches 8.5x on the most affected model (Claude Opus 4.5); (3) a practical LLM-based re-verifier (a single cheap second model call re-reading the original instruction) reduces wrong actions by 79% (0.21x; CI [0.18, 0.25]), closing the gap to within 1 percentage point of an oracle upper-bound (0.20x); and (4) in the natural (non-injected) setting, baseline agents drift on 18% of eligible workflows, with the per-step error rate rising across steps. Persistence and re-verification are not interchangeable: a defense that eliminates drift can worsen propagation, and a practical re-verifier nearly matches oracle recovery. 

---
# One Student, Many Teachers: Multi-Task On-Policy Distillation via Soft-Prompt Privileged Context 

**Authors**: Yingzi Ma, Zichen Zhu, Ming Jiang, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2607.18293)  

**Abstract**: On-policy self-distillation (OPSD) teaches large language models new skills through a teacher that shares the student's backbone and supervises its own rollouts. Existing teachers either inject privileged context at the input -- inducing post-hoc rationalization -- or fine-tune weights, accumulating drift and forgetting across tasks. We propose \method, whose teacher differs from the student only by a learnable soft prompt: trained on $(x, y_\text{gold})$ pairs with the backbone frozen, the prompt yields a task-specific teacher that preserves the student's exact representational geometry. \method\ extends naturally to multi-task settings by routing each example in a merged corpus to its corresponding soft-prompt teacher, allowing a single student to absorb knowledge from $K$ teachers in parallel; at inference, all prompts are discarded. On Qwen3-1.7B-Base and Phi-4-mini-instruct across four tasks (Science, Tool Use, Biology, Math), the single-task variant (OPD with a PT teacher) matches or exceeds full fine-tuning while training orders of magnitude fewer parameters, and the multi-task variant achieves the best overall average ($56.2$ on Qwen3-1.7B-Base) while preserving general-capability benchmarks -- in contrast to sequential SFT, which degrades both. 

---
# Reliability Scales Inversely: Bigger Models Compound Mistakes Faster via a Hidden Auto-Regressive Risk Regime 

**Authors**: Kushal Chakrabarti  

**Link**: [PDF](https://arxiv.org/pdf/2607.18292)  

**Abstract**: As language models scale, answers start truer but degrade faster: scaling buys capability but erodes reliability. The knowledge-gap account - more data, retrieval, or scale - misses an auto-regressive risk residual that scale sharpens: the model commits to a low-probability token, conditions on it as established, and snowballs. We track this through per-position disagreement $\delta = \log p_M - \log p_O$ against a stronger same-family oracle, whose second moment splits exactly into bias$^2$ $\mathrm{KL}(p_M \,\|\, p_O)^2$ and risk $\mathrm{Var}[\delta]$. We present four findings: (i) under scaling, the knowledge gap falls $\approx$$6\times$ while knowledge degradation grows $11$-$39\times$; (ii) at a fabrication, felt uncertainty $H(p_M)$ relaxes quickly while oracle-referenced risk persists up to $17\times$ longer, leaving a confident-but-precarious risk regime that bridges consecutive fabrications ($+69\%$ at $14$B); (iii) this regime is causal - an on-policy, fixed-$\mathrm{KL}$ variance contraction cuts web-verified hallucination by $35$-$74\%$ across three model families; and, (iv) it structurally evades self-monitoring, with $p_M$-only detectors (e.g. semantic entropy) firing $\approx$$30\%$ less ($p<10^{-16}$) on the risky branch holding nearly $4\times$ more fabrications. Bigger models snowball mistakes faster, through a failure mode that is dominant, self-perpetuating, causal and invisible to the model itself. 

---
# State Compression in Two-Agent LLM Relays: A Closed-World Study of Constraint Preservation 

**Authors**: Anantha Sharma, Sheeba Elizabeth John, Kaarthik Senthil Kumar, Saratsuhas Vijayababu  

**Link**: [PDF](https://arxiv.org/pdf/2607.18265)  

**Abstract**: Long-running Large Language Model (LLM)-based agents often accumulate large intermediate traces containing audits, eliminations, and numeric calculations. In practice, this state is compressed before handing it to a downstream decision step, creating an information bottleneck in which small omissions can break strict numeric or categorical constraints. This paper evaluates hand-off compression in a closed-world travel-planning relay with two LLM agents. A Researcher audits a fixed inventory of hotels and flights for 50 goal instances, and a Booker selects a hotel--flight pair using only the goal and the hand-off payload, with the inventory withheld. We compare four hand-off conditions: no compression, narrative summarization, schema-constrained JSON extraction, and embedding-based pruning. Exhaustive enumeration over the fixed inventory provides exact feasible and optimal labels. Results show that hand-off representation strongly affects downstream feasibility under a small decision model. JSON extraction achieves the highest feasibility accuracy at 0.96, while narrative summarization, despite producing the smallest compressed hand-off payload, degrades feasibility to 0.48. Embedding-based pruning matches the uncompressed control on feasibility at 0.88 without an additional generative compression call. These findings indicate that constraint checking benefits from structured and auditable hand-off representations rather than relying on brevity alone. 

---
# MUX: Continuous Reasoning via Multiplexed Tokens 

**Authors**: Ayhan Suleymanzade, Halil Alperen Gozeten, Michael Bronstein, İsmail İlkan Ceylan, Jinwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.18264)  

**Abstract**: Language models solve complex problems by articulating intermediate reasoning steps in natural language. While effective, this process is computationally bottlenecked: each reasoning step conveys only a single subword, and many are spent expressing a thought instead of carrying out computation. We propose MUX, a simple method for high-bandwidth and compact reasoning based on distillation of discrete reasoning into continuous multiplexed tokens in a latent space. Here, each latent token is trained to represent a weighted linear superposition (multiplexing) of a span of discrete reasoning subwords, where this superposition is lossless by construction and the span can be fully recovered (demultiplexing). We prove that simple position-dependent weightings, such as suitable geometric decay, support lossless multiplexing, which in turn prevents shortcut behaviors caused by latent collapse. We further show that multiplexed reasoning can perform parallel exploration in problems that require search. Across 32 evaluation settings spanning four language models, MUX outperforms strong latent reasoning baselines. Ablation and probing analyses further show that the learned latent tokens encode faithful and interpretable reasoning. Our results suggest that lossless superposition as local learning targets constitutes a sufficient condition for achieving strong and efficient latent continuous reasoning. 

---
