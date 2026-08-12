# ConVAWG: A Retrieval-Grounded Framework for Controlled Synthetic Dialogue Generation in Violence Against Women and Girls 

**Authors**: Chen Lyu, Xingwei Tan, Simon Cullen, Shelley Wilson, Lois Arthurs, Arshad Jhumka, Gabriele Pergola  

**Link**: [PDF](https://arxiv.org/pdf/2608.11200)  

**Abstract**: Synthetic dialogue generation offers a way to study conversational dynamics in sensitive domains where real data are difficult to access, release, or annotate. The underlying abuse may occur online or offline: threats and coercion can appear directly in messages, while behaviours such as surveillance, isolation, stalking, and physical violence may be planned, disclosed, or referred to conversationally. Privacy and legal constraints make it difficult the release of large-scale real conversation datasets; existing work has mostly focused on sentence-level toxicity of online abuses, leaving a gap in modelling abuse as a relational and temporally unfolding phenomenon. In this work, we focus on modelling Violence Against Women and Girls (VAWG) scenarios as multi-turn dialogues. We introduce ConVAWG, a retrieval-grounded framework for generating CPS-aligned synthetic VAWG chat dialogues. ConVAWG builds scenarios from persona seeds, demographic patterns reported by the UK Office for National Statistics, official crime definitions, and retrieved Domestic Homicide Review cases; converts them into hierarchical event timelines; generates multi-scene role-play dialogues; and applies targeted activation-steered toxicity control to appropriate utterances. We release over 6,000 multi-turn dialogue events across 200 scenarios with rich scenario-, event-, and turn-level metadata. Extensive human evaluation, LLM-as-Judge assessment, ablations, and downstream tasks show strong dialogue quality and domain fidelity. 

---
# From Interpretability to Control: Insights from Six Years of the TrustNLP Workshop 

**Authors**: Rahul Gupta, Abhinav Mohanty, Anaelia Ovalle, Anil Ramakrishna, Anubrata Das, Apurv Verma, Jwala Dhamala, Ninareh Mehrabi, Tharindu Kumarage, Yada Pruksachatkun, Yang Trista Cao, Kai-Wei Chang, Aram Galstyan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11171)  

**Abstract**: The Workshop on Trustworthy Natural Language Processing (TrustNLP), co-located with major ACL conferences since 2021, has grown from 8 proceedings papers to 41 over six editions, documenting a field-wide transition from post-hoc interpretability of static models to mechanistic understanding and proactive control of generative systems. We synthesize insights from all 144 proceedings papers, classifying them along six trust dimensions grounded in established frameworks (TrustLLM, DecodingTrust). We observe co-occurrences with capability emergence. The release of the first high-impact chat models activated all trust dimensions simultaneously, while subsequent model generations shifted focus toward truthfulness and safety alignment. Analysis from the classification study reveals that truthfulness is the fastest-growing dimension (absent in 2021-2022, comprising 37% of papers by 2025-2026), fairness remains the most consistent theme, and explainability exhibits a U-shaped trajectory; declining as post-hoc methods lost relevance but resurging in 2026 through mechanistic interpretability. A cross-venue comparison with ACL, NAACL, EACL, and EMNLP (~2K papers) in the same period shows that TrustNLP's topical distribution closely follows the field average. We identify four structural insights and conclude with actionable directions for the research community. 

---
# The Illusion of Cross-Lingual Safety in Low-Resource Languages 

**Authors**: Abigail Oppong, P Sam Sahil, Tadesse Destaw Belay, Maryam Ibrahim Mukhtar, Esmael Ahmed Abdu, Tassallah Abdullahi, Jessica Oparebea, Saminu Mohammad Aliyu, Idris Abdulmumin, Abubakar Juma Chilala, Nicholaus Dismas Ladislaus, Alfred Malengo Kondoro, Lemofouet Valdini Douglace, Shamsuddeen Hassan Muhammad, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2608.11146)  

**Abstract**: Safety alignment in large language models (LLMs) is largely developed in English, assuming these safeguards generalize across multilingual settings. However, this assumption remains underexplored and exposes a vulnerability in low-resource languages. We investigate cross-lingual safety transfer in four African languages, Twi, Hausa, Amharic, and Swahili, using LoDNA, a new safety dataset that pairs literal translations with culturally localized prompts. To move beyond generation-based evaluation, we propose a latent geometric framework that probes hidden-state refusal representations in LLMs. Our experimental results show that cross-lingual safety transfer is severely limited; harmful prompts retain less than 10% of the English refusal signal across most language-model pairs. Literal and localized prompts are semantically aligned (cosine 0.95-0.996) but drift across layers, suggesting models encode the concepts without routing them to safety mechanisms. These findings demonstrate that current multilingual safety alignment is superficial, providing strong evidence against the assumption of a universal, language-agnostic harm manifold within the specific low-resource languages studied. Warning: This paper contains example data that may be offensive or harmful. 

---
# Attention-Path Fragility as an Uncertainty Signal in Large Language Models 

**Authors**: Minsoo Kim, Sungyoung Ji, Kisung Moon, Ilyong Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2608.11138)  

**Abstract**: We propose that a model's uncertainty about a token is reflected not only in the breadth of its output distribution but also in whether a confident prediction is \emph{fragile} under perturbation of its attention pathways. We instantiate this as ASMI (Attention-Subnetwork Mutual Information), a training-free estimator that masks attention heads and measures the BALD mutual information among the resulting subnetworks, with a semantic-agreement kernel to discount surface-form disagreement. The signal is not a restatement of output confidence: on grounded QA an out-of-fold test shows it adds error-predictive information beyond single-pass confidence and entropy, concentrated in \emph{confident-but-fragile} predictions, where acting on it roughly halves the retained error of a confidence filter. The distinctness is regime-graded, so ASMI predicts its own domain of applicability, strong where answers are routed through provided context and bounded by design where they are recalled from parametric knowledge. Sem-ASMI reads the signal from a single greedy response, without the stochastic generations the strongest baselines require, and ties or beats Semantic Entropy on ten of the twelve grounded benchmark-backbone settings. Across the same twelve settings, the best ASMI variant, typically the adaptive one reusing the ten samples already drawn for the baselines, ties or leads the strongest baseline in eight, significantly in three under a paired test. On parametric QA all variants revert to or below the zero-cost MSP baseline, exactly as predicted, and the estimates are near-deterministic across reruns. A head-level analysis shows that what tracks this boundary is not the presence of head-level fragility but whether that fragility couples to errors. 

---
# Actions Speak Louder than Words: Measuring Cross-Lingual Policy Retention in Tool-Using Agents 

**Authors**: Sourabrata Mukherjee, Kalika Bali, Sunayana Sitaram  

**Link**: [PDF](https://arxiv.org/pdf/2608.11110)  

**Abstract**: When a tool-using agent is given the same task in a different language, does it still take the same steps? Multilingual evaluation rarely asks: it compares final answers and discards the actions. Yet those actions are the product: they fix cost and latency, decide how the system fails, and are the only auditable part of its behaviour. We make the action policy the measured object across 8 models, 6 parallel benchmarks and 41 languages (2.38M rollouts). The naive measurement fails: five confounds sit between raw trace similarity and any defensible claim, each able to flip a conclusion. Short traces score higher, empty traces score perfectly, unrelated traces agree by chance over half the time, the gap is capped by each model's reproducibility, and a model asked the same question twice in one language answers differently, leaving no baseline. We remove all five, and every correction makes the effect larger. Divergence proves structural, not sampling noise: it survives greedy decoding in every cell and stays flat as temperature rises, even as models grow less self-consistent. Normalised by their own reproducibility, four very different frontier models converge under greedy decoding, each keeping 71-73% of its action policy across languages, with model identity explaining only 5.7% of the variance. Below roughly 10B parameters it breaks down, and the ordering among smaller models is largely an artifact of a chance floor we measure by permutation rather than assume. Agents route non-English tasks through English; this pivot is causally load-bearing, confirmed by a pre-registered prediction across four models, and models will not abandon it when told to. Finally, a single trace-extraction regex, not the model, manufactured a multilingual failure: two worked examples raise one model's measured accuracy twenty-sixfold while its accuracy on readable outputs barely moves. 

---
# Multiclass Sentiment Analysis for Identifying Political Viewpoints 

**Authors**: Girma Yohannis Bade, Olga Kolesnikova, Jose Luis Oropeza, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2608.11049)  

**Abstract**: The rapid growth of social media has created vast amounts of political discourse, which provides valuable opportunities to analyze public opinions and identify different political perspectives. Sentiment Analysis (SA) is a core task in Natural Language Processing (NLP) that allows the computational study of attitudes and opinions in textual data, and has become increasingly important for understanding political discourse. In this work, we investigate multiclass sentiment analysis of political view- points on social media, that is to automatically discriminate multiple sentiment classes over political issues and figures. To solve this task we design and evaluate two machine-learning approaches based on XGBoost and BERT. We train and evaluate the models on a labeled dataset of political social media posts using standard classification metrics. The experimental results show that the XGBoost model reaches an F1-score of 0.2835 and the BERT- based model reaches an F1-score of 0.2806 on the test set. These results demonstrate the challenge of classifying complex and contextualized political discourse sentiment and provide a baseline for future research in multiclass political sentiment analysis. 

---
# TEAMMix: Taxonomy Enrichment Augmentation and Minority-augmented Mixing Strategy for LLM-enhanced Weak-Supervised Hierarchical Text Classification 

**Authors**: Jian Zhang, Zhuohao Yang, Songlin Lei, Bangli Liu, Ziwei Wang, Xufeng Weng, Gehan Amaratunga, Yu Lin, Hongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11044)  

**Abstract**: Hierarchical Text Classification (HTC), as a critical text mining task, faces challenges such as complex label hierarchies and class imbalance. Existing methods based on large language models (LLMs) struggle to be efficiently applied to this task due to issues like lengthy prompts and loss of label structural information. To address these limitations, this paper proposes a weakly supervised HTC framework enhanced by LLM-based data augmentation. The framework first enriches the label hierarchy semantically through keyword generation and corpus mining, thereby enhancing the model's understanding of labels. Subsequently, it guides the LLM to generate pseudo-samples to mitigate the long-tail problem, and employs a Gaussian mixture model for confidence-based resampling to optimize the quality of generated data. Experimental results demonstrate that the proposed method effectively improves the reliability of LLM-generated pseudo-labels and significantly enhances classification performance on fine-grained and imbalanced datasets. 

---
# myMediWhisper: Construction of Burmese Medical Speech Corpus and Whisper Fine-Tuning for Clinical Dialogue ASR 

**Authors**: Ye Kyaw Thu, Ye Bhone Lin, Thura Aung, Htet Arkar, Myat Oo Swe, Thet Htet San, Min Thiha Tun, Thazin Myint Oo, Thepchai Supnithi  

**Link**: [PDF](https://arxiv.org/pdf/2608.11036)  

**Abstract**: Although Whisper models benefit from large-scale multilingual pre-training, their performance on Burmese medical speech remains limited. This work presents a Burmese medical speech recognition framework built on a high-quality 28-hour corpus recorded and validated by native speakers. We fine-tune Whisper models using full fine-tuning (FFT) and parameter-efficient fine-tuning (PEFT) with LoRA. To evaluate robustness, we apply waveform- and spectrogram-level data augmentation under controlled noise and simulated room acoustics. While augmentation reduces performance on clean speech, it significantly improves robustness in noisy and reverberant environments across FFT and PEFT settings. Our best-performing system, fully fine-tuned myMediWhisper-Medium without augmentation, achieves a state-of-the-art Word Error Rate (WER) of 23.44%, outperforming much larger general-domain fine-tuned models. Dataset and other resources can be found at the Huggingface repository: this https URL. 

---
# Data Attribution of Emergent Misalignment with Persona Features 

**Authors**: Clemens Vetter, David Kaczér, Lucie Flek, Florian Mai  

**Link**: [PDF](https://arxiv.org/pdf/2608.11025)  

**Abstract**: Emergent misalignment (EM) is the phenomenon where fine-tuning a language model on a narrow task leads to harmful behavior in unrelated domains. A leading mechanistic account attributes EM to persona features: latent directions acquired during pre-training that misaligned fine-tuning amplifies. We ask where these features come from: which pre-training documents activate them, and whether naturally occurring human-written text suffices to induce EM. Using Sparse Autoencoder (SAE) based model diffing across four open-weight models, we find that features related to jailbreak personas, sarcasm, deception, and manipulation are amplified by misalignment fine-tuning, while safety-relevant and assistant-identity features are suppressed. Steering individual features controls EM in both directions: it induces misalignment rates of up to 62% in aligned models -- exceeding the 35% reached by misalignment fine-tuning itself -- and re-aligns misaligned models to near-baseline misalignment rates. Attributing the causal features to a corpus of one million pre-training web documents retrieves semantically relevant narratives about villainous characters, domination, and harmful agency. However, fine-tuning on these human-written documents does not reliably induce EM, even after reformatting into assistant-style responses, whereas synthetic instruction-response pairs derived from the same content do -- and transfer across model families. Semantic relevance alone is therefore not sufficient: response structure or model-generated phrasing plays an important role in inducing EM. 

---
# Templated or fully Synthetic? Prompt construction as a confound in measuring LLM political stance beyond writing assistance 

**Authors**: Ilias Chalkidis  

**Link**: [PDF](https://arxiv.org/pdf/2608.11008)  

**Abstract**: Political stance detection in LLMs has long been dominated by closed-ended, multiple-choice political survey questions---originally designed for humans, and thus lacks the realism and nuance of human-AI interactions in the wild, while also being susceptible to sandbagging. The recent IssueBench framework substantially mitigates these limitations with templated prompts anchored in real-world chat logs. Given the rise in non-work-related use of GenAI assistants, we extend IssueBench beyond writing assistance to include two additional tasks, information seeking and opinion sharing. We argue that templated prompts still lack the nuance of real ones, especially for open-ended tasks, and remain recognisable as evaluation artefacts. We propose the use of fully synthetic (LLM-generated) prompts, produced under detailed instructions with real prompts as seeds. We assess the ecological validity of real, templated, and LLM-generated prompts in a small-scale study covering 3 highly contested policy issues and 3 recent geopolitical conflicts. Human and LLM annotators rank LLM-generated prompts as no less realistic than real ones and clearly more realistic than templated ones, and find that they carry their intended intent and stance more clearly; the LLMs separate templated prompts from the other two far more sharply than the humans do. In a case study, templated and LLM-generated prompts yield systematically different stance estimates for the same model, most visibly under neutral framings, where templated prompts overstate the model's leaning in the direction encoded by the topic-and-stance text (filler) slotted into their templates. 

---
# On the Limitations of Cross-Lingual Consistency in Multilingual Text-to-image Generation 

**Authors**: Sicheng Zhang, Zhonghao Yan, Binzhu Xie, Shi Qiu, Muzammal Naseer, Naveed Akhtar, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2608.11002)  

**Abstract**: Text-to-image (T2I) generation has achieved remarkable progress in recent years. However, existing research has largely focused on English-only settings, leaving cross-lingual performance gaps and language-specific effects insufficiently explored. To fill this gap, we introduce LingT2I, a benchmark covering 10 widely used languages with 33K prompts, designed to evaluate cross-lingual effects in both content generation and text rendering. Building on this benchmark, we conduct a comprehensive cross-lingual analysis, uncovering linguistic inequality and language-dependent trade-offs across evaluation dimensions. Beyond quantitative evaluation, we further reveal a range of language-dependent generation patterns, highlighting how linguistic factors and their corresponding cultural contexts systematically impact model outputs. Our benchmark and analysis provide a foundation for studying cross-lingual behavior in T2I generation and facilitate the development of more robust and inclusive models. Code and dataset are available at this https URL. 

---
# ConRub-Med: Reinforcement Learning with Consensus Rubrics for Open-Ended Medical Question Answering 

**Authors**: Taojie Zhu, Yuan Xia, Tao Sun, Yizhi Wang, Yan Chen, Qunshan He, Tian Guan, Jian Wang, Jinjie Gu, Junwei Liu, Yonghong He  

**Link**: [PDF](https://arxiv.org/pdf/2608.10996)  

**Abstract**: Reinforcement learning with verifiable rewards has been especially effective in mathematics and coding, where answers can be checked automatically. Many open-ended medical questions lack comparably cheap outcome verifiers: responses may be partly correct, incomplete, or contain clinically consequential errors. Rubrics written or validated by physicians offer strong clinical grounding, but involving experts in every instance is costly. Model-generated rubrics make this supervision scalable. We introduce ConRub-Med to preserve useful distinctions as rubric feedback moves from construction to policy optimization. For each prompt, three heterogeneous language models propose atomic criteria independently; a separate model reviews them, retaining only criteria with semantic support from all three generators. Three-State scoring distinguishes correct coverage, missing information, and incorrect claims. Errors receive negative rather than zero credit. When every response in a complete Group Relative Policy Optimization (GRPO) group receives the same final reward, a pairwise judge provides sequence advantages only if both candidate orders agree, without changing the scalar rewards. Groups without ties use vanilla GRPO. In a blinded study matched by question, two medical experts rate panels from the full pipeline as more clinically relevant than panels produced by one generator. Across the evaluated open models, ConRub-Med ranks first on six of nine benchmarks and achieves the highest medical and generalization averages. Using the resulting rubric dataset of 5,166 prompts, it scores $38.98 \pm 1.04$ (mean $\pm$ SD) on HealthBench-Hard, compared with InfiMed-ORBIT's 33.60 with 8,000 samples and 37.30 with 28,000. 

---
# What Iterated Self-Feeding Probes of Language Models Measure, and a test that separates the construction from the model 

**Authors**: Nicolás Vera Zúñiga  

**Link**: [PDF](https://arxiv.org/pdf/2608.10986)  

**Abstract**: A growing class of methods probes a language model by feeding it its own output: self-consistency, iterated refinement, agentic loops. We ask what such a probe measures, in a construction chosen to make the question sharp: a ring of token cells resampled in place by the model's own windowed conditional p_r(x_i | x_{i+-r}). The substrate is Glauber dynamics on token sequences and is not new; what we change is the coupling. Advancing two rings that differ in one token under common random numbers makes undamaged copies diverge by exactly zero, so damage spreading becomes measurable where a maximal coupling gives mixing times instead. The answer is that it measures two different things at once, in readings that look alike. Some quantities are fixed by the construction: the damage light cone is kinematic, and the radius scaling of the token-space Lyapunov exponent lambda_ca(r) is model-invariant across 19 models and two scale ladders spanning 70x. Others genuinely track the model: lambda_ca crosses zero at a reproducible point in training, and the attractor share ranks models consistently however the lattice is built. Left undistinguished, the first kind is readily mistaken for the second -- we did so ourselves for four months, and report a phase transition we measured to three decimal places that belongs to the probe rather than to any language model. We give the test that separates them: hold the construction fixed and vary the model, or hold the model fixed and vary the construction, and see which readings move. We validate the instrument by reproduction first, recovering a Domany-Kinzel damage field bit-exactly against an independent prediction, and we report the estimator failures that this discipline caught -- four retracted verdicts, each on a quantity that looked like a measurement. The methodology ships as a package. 

---
# MUSE: A Full-Text Cross-Domain Knowledge Base of Scientific Problems, Solutions, and Rationales 

**Authors**: Tsofia Cohen, Tom Hope  

**Link**: [PDF](https://arxiv.org/pdf/2608.10974)  

**Abstract**: Scientific papers contain fine-grained records of problem solving: authors mention technical obstacles and methods that were used to address them, often along with reasoning on why those methods were chosen. We introduce MUSE (Mining Underlying Scientific Explanations), a full-text, multi-domain resource of scientific Problem-Solution-Rationale (P-S-R) triplets. We curate 579 expert-annotated full-text paragraphs, with a rich annotation schema covering salient problem, solution, and rationale spans, solves and rationale_of links and conceptual coreference. A modular extraction pipeline scales this annotation to build a high-quality knowledge base of 37K source-grounded P-S-R triplets. We evaluate the extraction components and include a preliminary experiment training a rationale-supervised LLM for scientific problem solving. Interestingly, we find that rationale supervision improves performance on complex, multi-constraint problems but can harm performance on simpler ones. 

---
# ReLTEx: Reliable LLM-based Taxonomy Expansion 

**Authors**: Zeinab Ghamlouch, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2608.10970)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated strong capabilities in generating semantically relevant concepts and relations, making them promising tools for taxonomy enrichment. However, directly relying on LLM-generated expansions often leads to noisy, redundant, or hierarchically inconsistent structures, limiting their reliability for automated taxonomy expansion. In this paper, we present ReLTEx, a framework for reliable LLM-based taxonomy expansion. ReLTEx combines LLM-driven candidate generation with structure-aware validation and recursive expansion control to improve the consistency and quality of generated taxonomies by reducing hallucinations. We evaluate the proposed framework using benchmark taxonomies under a masked taxonomy expansion setting and compare multiple validation strategies. Experimental results, supported by both adapted evaluation metrics and human evaluation, demonstrate that ReLTEx produces more reliable and semantically coherent taxonomy expansions. 

---
# REAP: Relation-Aware Elicitation and Parsing for Closed-Book Knowledge Base Construction from LLMs 

**Authors**: Thanh-Dan Bui, Thanh-Trung Do, Tuan-Phong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2608.10963)  

**Abstract**: We present the REAP system for the AKBC Shared Task 2026 on constructing knowledge bases from language models in a closed-book setting, subject to a budget of at most 32B parameters and no model fine-tuning. Our system combines structured chain-of-thought reasoning, relation-specific query strategies, and a reasoning-based empty-set gate to elicit parametric knowledge, followed by direct extraction into valid JSON arrays. On the test set, the system, built on the Mistral-Small-24B-Instruct-2501 model, achieves a macro-F1 score of 0.62, with particularly strong results on countryLandBordersCountry (F1 = 0.95), companyTradesAtStockExchange (F1 = 0.73), and hasArea (F1 = 0.77). Our code is publicly available at this https URL. 

---
# A Cost-Efficient Routing Pipeline for Multilingual Short-Text Classification Using Small Language Models 

**Authors**: Wajdi Ben Saad, Safa Madiouni  

**Link**: [PDF](https://arxiv.org/pdf/2608.10939)  

**Abstract**: Multilingual short-text classification supports operational systems such as content moderation, customer support routing, and intent recognition, yet aggregate evaluation often hides large differences between high-resource and low-resource languages. Uniform inference policies are simple to deploy, but they assume that all languages are equally well served. In this work, we evaluate a fixed-list routing strategy that keeps stronger languages on a direct multilingual path and selectively sends weaker languages through translation into English before zero-shot classification. The pipeline is fully self-hosted, uses pretrained compact sentence encoders, and requires no task-specific fine-tuning.
We test the approach on two benchmarks chosen to differ in scale and label granularity: a 15-language subset of SIB-200 for seven-way topic classification and a 15-locale subset of MASSIVE for intent classification over an official 60-intent inventory. On SIB-200, the best overall configuration is R1, which translates only the low-resource tier: high-tier and mid-tier Macro-F1 remain unchanged, while low-tier Macro-F1 rises from 0.4632 to 0.6828. On the MASSIVE subset, the same low-tier intervention raises low-tier Macro-F1 from 0.2143 to 0.4417, but the best overall result is obtained by full translation, R3, at Macro-F1 0.4647. Across these two benchmarks, selective translation is a reliable intervention for weaker languages, whereas the optimal routing boundary depends on the task. We therefore report routing through tier-level quality gains and tier-level latency rather than a single global efficiency score. 

---
# FaithformBench: Benchmarking Faithfulness of Mathematical Chain-of-Thought Autoformalisation 

**Authors**: Rob Cornish, Iacopo Ghinassi, Po-Hung Yeh, Shuqi Liu, Qiyuan Xu, Haoxuan Yin, Dominik Wagner, Wenda Li, Yee Whye Teh, Luke Ong  

**Link**: [PDF](https://arxiv.org/pdf/2608.10916)  

**Abstract**: Autoformalisation (AF) systems map natural language reasoning steps into formal statements in a proof assistant such as Lean. We consider how to assess the faithfulness of these systems. Existing approaches require expensive human-annotated ground truth, or rely on LLM judges or embedding models, which come with limited guarantees of accuracy. In addition, these methods typically only consider inputs that are known to be correct, and therefore do not assess whether the AF translates incorrect inputs faithfully. To address these limitations, we propose a new benchmark for AF faithfulness that is cheap to apply, sound under weak assumptions, and assesses both positive and negative examples. Our method is based on automatically generating perturbed reasoning steps that are designed to be invalid, and then measuring validity preservation on unperturbed steps and invalidity preservation on perturbed steps. We apply our method to eight AF systems across four mathematical datasets, and observe pervasive sycophancy: many AFs "silently correct" invalid inputs into provable statements. The most validity-preserving fine-tuned AFs are also the most sycophantic, suggesting a tension between validity and invalidity preservation in current AF systems. 

---
# Certify or Refuse: A Cross-Model Map for Selective Risk Control with Coverage Floors under Covariate Shift 

**Authors**: Jiamiao Liu, Dewen Qiao, Yu Zhang, Xuetao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.10893)  

**Abstract**: Certified selective predictors attain whatever coverage they attain; operators impose an automation floor: answer at least a $\beta$-fraction of shifted target traffic with at most an $\alpha$-fraction of answers wrong. Under bounded-ratio covariate shift we prove the Floor Certification Map: once that floor must be certified alongside the selection-conditioned risk $\alpha$, certification acquires a feasibility frontier and a two-resource complexity map, additive up to constants: risk in labeled source, the floor in unlabeled target samples. The rates are local, needing a regular frontier margin, slack below the local-regime threshold, and lattice conditions: pre-registered with a lattice margin for the upper bounds, compatible per-slack for the lower. The displayed split is the operational route; oracle weights also allow a labeled-source floor estimate. Three model-tagged results: a lower bound (Model-B), a matching oracle-weight upper bound (Model-A), and an implementable upper bound (Model-B') valid under a pre-registered exact stratified-shift model with nuisance cost priced explicitly. The match is across these models rather than a single-model minimax theorem, and necessarily so: over the full bounded-ratio class no unknown-weight procedure matches at any sample size (Model-B is inconsistent, witnessed at $\alpha=\beta=1/2$). The nuisance's necessity is only partially settled. Complexity tracks a localized accepted-region functional, not global effective sample size (ESS), on both sides, though a fixed-ESS separation theorem is left open; both lower-bound axes vanish as $\beta\to0$, so the floor creates the map. Empirically, the registered bite family diverges with log-log slope $-2.002$ within its pre-registered band; a 1,024-cell audit records 0 violations where the formal certificates fire; and a single-corpus SQuAD-to-NewsQA feasibility audit returns honest refusal. 

---
# X2-Turn: Frame-Synchronous Dual-Head Modeling for Joint Streaming ASR and Turn State Prediction 

**Authors**: Kaiqi Fu, Rime Wen, Altman Lin, Shawn Qin, Roy Gan, Hao Wang, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.10878)  

**Abstract**: Accurate and responsive turn-taking is essential for spoken dialogue systems, which must distinguish in real time between user interruptions, backchannels that should be ignored, and the completion of an utterance. Prior modular approaches typically optimize turn state prediction at the utterance or fixed-chunk level, creating a mismatch with the continuous turn state estimate, and often depend on an auxiliary ASR model, which limits responsiveness and increases overall system complexity. Therefore, we present X2-Turn, a frame-synchronous turn state prediction method via delayed-stream modeling. Specifically, building on the pretrained Voxtral Realtime model, we introduce a frame-synchronous turn state head that operates in parallel with the ASR head on shared streaming representations, jointly predicting ASR tokens and fine-grained turn states at the frame level. We evaluate our method on the bilingual Chinese-English Easy-Turn test sets, and the results demonstrate its effectiveness in achieving accurate turn-taking detection while maintaining low latency. 

---
# VibeLifeBench: Can Your Life Agent Be Proactive and Persistent in a Living World? 

**Authors**: Xiaohongshu Inc  

**Link**: [PDF](https://arxiv.org/pdf/2608.10875)  

**Abstract**: Large language model (LLM) agents are increasingly deployed as personal assistants. Existing evaluations, however, mostly use short, self-contained requests in static environments. Everyday life assistance is different. A task runs for weeks rather than minutes. The world keeps changing while the agent is not being prompted. Many constraints are never stated outright. An agent that merely answers the request in front of it will fail at such a task. What is needed instead is an agent that stays proactive and consistent. It decides on its own when to act, when to ask, and when to stay silent. It notices changes that nobody announced. It keeps one plan coherent from the first day to the last. No current benchmark measures this. We introduce VibeLifeBench, a benchmark of 200 long-horizon tasks across ten everyday-life domains. Each task is a scripted multi-week timeline in a simulated world of 22 mock services. The world advances on its own clock, and many of its changes are silent, so only an agent that re-inspects the world discovers them. Every task is graded by fine-grained, weighted checks that read only what the agent actually left behind, covering the end state, the timeliness of its actions, and whether it upheld the implicit constraints. We evaluate seven frontier models. All of them score low, which shows how far current agents are from assisting with real life. We will open-source all tasks, environments, and the evaluation framework. 

---
# Reference-Free Post-Training of Open Large Language Models for Multilingual Machine Translation 

**Authors**: Chris Han, Pengzhi Gao, Pei Fu, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2608.10812)  

**Abstract**: We study reference-free post-training for multilingual machine translation with open large language models. Starting from the supervised-finetuned MiLMMT-46-v0.1 models, we apply Group Relative Policy Optimization (GRPO) with a reward that averages two reference-free quality estimation models and is gated by language identification. We then linearly interpolate the supervised fine-tuning (SFT) and reinforcement learning (RL) model checkpoints to obtain MiLMMT-46-v1.0. Across 46 languages, the resulting models consistently improve translation quality over their SFT counterparts, outperform strong recent open baselines, including Seed-X, HY-MT2, and TranslateGemma, and achieve leading reference-free scores against evaluated proprietary systems such as Google Translate, Gemini 3 Pro, and GPT-5. We further investigate on-policy distillation and find that it reaches, but does not surpass, the quality frontier achieved by RL with checkpoint interpolation. We release the models and code to facilitate future research. 

---
# Surfacing the Unsaid: CUE-Bench for Affective Stance in Chinese Discourse 

**Authors**: Zhenyan Zheng, Yunyao Zhang, Junxi Sheng, Junqing Yu, Zikai Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.10810)  

**Abstract**: Emotion understanding in discourse requires reasoning beyond surface sentiment because speakers often convey affect through indirect, implicit, polite, ironic, or deliberately mismatched expressions. Existing emotion benchmarks mainly annotate surface polarity or final emotion categories, while lacking a structured account of how explicit expression, implicit affect, pragmatic intent, and fine grained emotion interact. This limitation makes current evaluations insensitive to cases where affective meaning is concealed, weakened, inverted, or pragmatically reshaped, thereby obscuring model failures in deeper emotion understanding. To address this gap, we introduce CUE Bench, a Chinese Unsaid Emotion benchmark that centers on Affective Stance and covers diverse communicative scenarios. CUE Bench constructs nine human interpretable affective stances from explicit implicit polarity interaction and further provides intent and fine grained emotion annotations for structured affective inference. Experiments show that incorporating Affective Stance improves fine grained emotion recognition by 3.5 percentage points and pragmatic intent detection by 7.8 percentage points over strong baselines. 

---
# Assessing Reliability of BERT-Based Models on Question Answering Tasks 

**Authors**: Pooja Yadav, Priyanka Harjule, Basant Agarwal, Marko Robnik Šikonja  

**Link**: [PDF](https://arxiv.org/pdf/2608.10806)  

**Abstract**: Reliability estimation of large language models is in many cases as crucial as their accuracy, as reliable models are more trustworthy, robust, and suitable for practical applications. Recent advancements in natural language processing (NLP), particularly those based on transformer architectures, have significantly accelerated progress across various NLP tasks. This study focuses on the reliability of transformer-based question answering (QA) models, specifically BERT models and its variants (RoBERTa, ALBERT, DistilBERT). These encoder-only pretrained transformers have demonstrated remarkable accuracy in QA tasks that can be treated as classification tasks. However, their reliability remains underexplored. This study evaluates the reliability of four BERT-based models by assessing response stability under two conditions: (1) internal model variations induced via Monte Carlo Dropout (MCD) and (2) input perturbations through paraphrasing. Using the SQuAD and QuAC datasets, we investigate how dropout rates affect prediction consistency and whether lexical changes impact answer stability. Our findings reveal that RoBERTa maintains higher reliability, whereas AlBERT and DistilBERT exhibit significant inconsistencies. Statistical analyses confirm that enabling MCD during prediction does not disrupt inference dynamics, validating its effectiveness as a reliability metric. These findings underscore the importance of evaluating both accuracy and stability in QA models to ensure stability in real-world applications. 

---
# Mitigating Context Interference for Reliable and Efficient Search Agents 

**Authors**: Boyang Xue, Bin Wu, Shuofei Qiao, Sheng Wang, Rui Wang, Yiming Du, Hongru Wang, Jeff Z. Pan, Emine Yilmaz, Kam-Fai Wong, Aldo Lipani  

**Link**: [PDF](https://arxiv.org/pdf/2608.10743)  

**Abstract**: Recent research empowers Large Language Models (LLMs) as multi-turn search agents to iteratively retrieve and generate outputs until complex tasks are solved. However, the contexts of multi-turn search agents are lengthy and complex. For example, the retrieved set of documents in each turn would inevitably introduce irrelevant information that distracts LLMs, referring to \textit{context interference}, potentially hindering the reliability and efficiency of search agents. Therefore, we conduct a systematic study on context interference in multi-turn search agents, focusing on investigating i) which parts of the context of search agents will contribute to the context interference, ii) how to refine the contexts of search agents to mitigate the interference, and iii) can incorporating context refinement into search agent training yield further improvements. We reveal that interference primarily arises from the latest retrieved documents. Based on the explored findings, we then introduce a distill-based context refiner to dynamically mitigate context interference for multi-turn search agents. Finally, we validate that incorporating context refinement into RL training pipelines of search agents can significantly enhance both reliability and efficiency. This study highlights the importance of mitigating context interference of search agents, inspiring a novel paradigm of ``refine context and then generate'' for AI agents. 

---
# Most biomedical publications show signs of LLM-assisted writing 

**Authors**: Lena Holzwarth, Rita González-Márquez, Dmitry Kobak  

**Link**: [PDF](https://arxiv.org/pdf/2608.10715)  

**Abstract**: Over the past several years, LLM-powered chatbots and agents have become widely used as a tool for academic writing. LLM-assisted writing can be valuable by removing language barriers but at the same time causes concerns about misconduct and fraud. To inform policy decisions, it is necessary to monitor the prevalence of LLM-altered texts in scholarly publications. Despite some recent progress in this direction, no existing method can produce reliable estimates. Here we suggest and validate a new unbiased approach to estimate LLM usage in a corpus of texts based on changing word frequencies. We apply our method to the full texts of open-access biomedical papers from Pubmed Central, and show that by the end of 2025, 89% of papers show excess of LLM-associated vocabulary. We also find that LLMs are twice as likely to be used when writing a paragraph in the Discussion section (68%) compared to a paragraph in the Methods section (32%), but even inside the Methods section, the overall prevalence of LLM usage is over 50%. We believe that our estimates are crucial to shape future guidelines and policies. 

---
# EVIL-Detect for NLPCC 2026 Shared Task 6: LLM-Generated Text Detection 

**Authors**: Hongrui Bao, Hangyu Rong, Zhuoshang Wang, Yubing Ren, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2608.10698)  

**Abstract**: The rapid development of large language models (LLMs) has increased the need for reliable detection of LLM-generated text, especially in realistic Chinese scenarios involving human-written text (HWT), LLM-generated text (LGT), and LLM-refined text (HLT). This paper presents EVIL-Detect, a multi-signal ensemble framework with conflict-aware fusion for NLPCC 2026 Shared Task 6. The system integrates edit-extent regression, zero-shot likelihood-contrast signals, lexical statistics, and conservative text rules. With calibrated decision boundaries and conflict-aware integration, our system improves robustness under strong out-of-distribution shifts, achieving a macro-F1 score of 0.8888 and ranking first in the official evaluation. Our code is available at this https URL. 

---
# SPIEval: Evaluating Large Language Models as Mobile Assistants over Scattered Personal Information 

**Authors**: Junjie Ye, Zhuohui Sheng, Shaofan Liu, Yulun Zhu, Wenjie Fu, Dingwei Zhu, Ming Zhang, Yujiong Shen, Weichao Wang, Xin Zhao, Shihan Dou, Tao Gui, Qi Zhang, Xuanjing Huang, Pluto Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.10692)  

**Abstract**: Large language models (LLMs) are increasingly deployed as mobile assistants, where a key challenge is leveraging personal information scattered across multiple applications (apps) to complete user instructions. However, due to the lack of dedicated benchmarks, their capabilities remain poorly understood. To address this gap, we introduce SPIEval, a human-curated benchmark grounded in five cognitive capabilities (i.e., reasoning, disambiguation, integration, preference inference, and multi-intent decomposition). SPIEval comprises 250 tasks spanning 4,335 personal records distributed across 10 apps and supports multi-turn interaction through 21 tools. Analysis shows that the benchmark exhibits diverse scenarios, challenging tasks, scattered information, controllable environments, and verifiable outcomes. We evaluate nine representative LLMs and find substantial room for improvement. The best-performing model, GPT-5.5 (xhigh), achieves only 57.3% accuracy, while the weakest achieves just 16.4%. Further analysis reveals that 79% of failures stem from inaccurate information localization, as LLMs often commit to plausible but incorrect information instead of continuing retrieval for verification. We also find that fewer than 2% of retrieval actions employ advanced search methods and observe substantial variation in search efficiency across models. These findings expose fundamental limitations of current LLM-based mobile assistants and motivate future research in this direction. Data and code are available at this https URL. 

---
# Can Released LLM Vocabularies Support Token-Level Estimation of Hidden Corpora? 

**Authors**: Qingjie Zhang, Xingzhang Ren, Zixuan Chen, Jinfeng Li, YueFeng Chen, Yitong Yang, Hui Xue, Dayiheng Liu, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2608.10690)  

**Abstract**: Pretraining corpus composition shapes LLM capabilities, but it often remains hidden even when model weights are released. Prior work has inferred corpus mixtures or traced specific token groups from released tokenizer vocabularies; in contrast, we estimate corpus ratios for arbitrary target tokens. We first show that BPE tokenizers trained on different corpora share stable token ID--ratio distributions, motivating distribution transfer from known corpora to a target tokenizer trained on hidden corpora. We then propose Quantile-Guided Density Estimation (QGDE), which approximates this distribution with multiple quantile trends and uses local density weighting to produce token-level estimates. In controlled settings and a realistic setting using the released SmolLM tokenizer, QGDE achieves mean relative errors as low as 3.00% for token-level estimation and 3.08% after aggregation into category-level mixtures. These results suggest that released tokenizer vocabularies provide a useful signal for fine-grained corpus estimation beyond coarse composition inference. 

---
# Leveraging Human Reading Behavior for Keyphrase Extraction: A Webcam-based Eye-tracking Corpus 

**Authors**: Chengzhi Zhang, Xinyi Yan, Wenqi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.10688)  

**Abstract**: Purpose: Keyphrases are statistically and semantically important textual units that can also attract readers' attention during comprehension. However, existing keyphrase extraction (KPE) studies mainly focus on improving textual representation while largely overlooking human reading behavior. This study examines whether lightweight webcam-based eye-tracking features can improve KPE from Chinese academic abstracts in Library and Information Science (LIS).
Methodology: To address the limited availability of eye-tracking data for Chinese academic reading, we developed a lightweight webcam-based data collection platform using the open-source SearchGazer library and constructed the Chinese LIS Eye-Tracking Corpus (CLIS-ET). Three character-level eye-tracking features, first fixation duration (FFD), fixation number (FN), and total fixation duration (TFD), were incorporated into KPE models to evaluate their effects on extraction performance.
Findings: Eye-tracking features consistently improved KPE performance. The combination of FN and TFD achieved the best results on the Att-BiLSTM+CRF model, indicating that readers' fixation behavior provides useful signals for identifying keyphrases in academic abstracts.
Originality/value: This study introduces a cost-effective webcam-based eye-tracking approach for KPE and presents CLIS-ET, a Chinese academic eye-tracking corpus containing FFD, FN, and TFD features. The results demonstrate the value of incorporating human reading behavior into keyphrase extraction. Dataset and code: this https URL and this https URL. 

---
# Auditing Chinese Web-scale Corpora via Sampled BPE Token Statistics 

**Authors**: Qingjie Zhang, Ziqi Tang, Jie Zhang, Gelei Deng, Jinfeng Li, YueFeng Chen, Yitong Yang, Hui Xue, Tianwei Zhang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2608.10678)  

**Abstract**: Chinese web pollution has surfaced in LLMs, motivating audits of upstream Chinese corpora. However, auditing such corpora faces three challenges: (1) their web-scale size makes full scan costly; (2) prior analyses are often too coarse to expose token-level pollution; (3) Chinese web pollution is implicit and rapidly changing. We propose Sampled-BPE, a lightweight token-level auditing pipeline that sample a small subset and train BPE tokenizer to surface polluted tokens. Experiments show that Sampled-BPE preserves usable estimates while substantially reducing runtime and memory: a 148.4 $\times$ speedup and a 35.8 $\times$ memory reduction induce only 4.25% relative error for pollution categories. We apply the pipeline to 11 open Chinese corpora and 6 Chinese Common Crawl snapshots from 2021 to 2026. The audit reveals widespread but uneven pollution across open corpora, as well as highly polluted and temporally shifting Chinese web content. We further release a hierarchical Chinese web token dataset with 660k+ token records, each with web context, category, and explanation fields, organized as trees to support review and tracing of pollution. 

---
# Seeds Before Objectives: Rethinking Evaluation for Low-Resource Garhwali ASR 

**Authors**: Karamvir Singh Batra, Prathamjyot Singh, Ashima Sood, Jasmeet Singh, Sahil Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2608.10670)  

**Abstract**: At corpus sizes typical of low-resource dialects, single-run comparisons can yield gains that do not replicate. We show this for Garhwali, an under-resourced Indo-Aryan language of the central Himalaya, building the first reproducible multi-seed ASR benchmark on the official VAANI splits, with per-seed outputs and significance testing. Re-examining plausible gains, we find them fragile: neither Focal CTC nor a matra-weighted objective beats standard CTC under seed-level testing, the matra objective fails to cut even its targeted errors, and Hindi-to-Garhwali transfer gives no gain over direct fine-tuning. What holds up is mundane: w2v-BERT 2.0 with standard CTC reaches 47.0% WER over five seeds, beating the larger MMS-1B and comparable models; pretraining design, not parameter count, drives performance, and speed augmentation gives a small, largely consistent gain. Multi-seed evaluation on official splits separates real gains from seed noise. 

---
# Decomposition-Induced Context-Memory Conflict: When Fact-Checking Pipelines Contradict Their Own Source Text 

**Authors**: Yu-Feng Yen  

**Link**: [PDF](https://arxiv.org/pdf/2608.10627)  

**Abstract**: Decompose-then-verify pipelines, including FActScore-style fact-checkers and long-form factuality evaluators, first split a passage into atomic claims before checking each one. Decomposition itself is treated as a neutral preprocessing step. We show it is not: a decomposer can be induced to substitute its own parametric belief for what the source passage says, producing a claim that contradicts the text it was supposed to summarize faithfully. We call this Decomposition-Induced Context-Memory Conflict (DI-CC) and show it is mechanistically the same phenomenon as classical context-memory conflict, occurring inside a different pipeline stage than prior work has examined. A linear probe trained only on classical context-memory conflict data (NQ-Swap), never exposed to any decomposition output, significantly separates decomposition positions that produce DI-CC from faithful decompositions (AUC = 0.86-0.88, permutation p < 0.0005). An existing reference-free baseline, SelfCheckGPT-style self-consistency sampling, fails to detect DI-CC at all (AUC 0.51, chance-level), because DI-CC content is stably recoverable and recurs across resamples, unlike the variability self-consistency methods rely on. Context-aware decoding, a training-free mitigation from the classical setting, transfers to decomposition and suppresses DI-CC, but at a severe cost: many decompositions under coreference-heavy conditions fail to parse, often because the decomposer fabricates a different identity. We do not consider this mitigation deployment-ready. We further characterize the mechanism's boundaries: its natural occurrence rate is too sparss not manifest on naturally-occurring hallucinatedtext, and it requires a minimum model scale to detecablish DI-CC as a real, mechanistically grounded, andpartially treatable failure mode, with a scope we chhan overstate. 

---
# Dual-Loop Self-Evolution via Verifiable Emotion Feedback for Multi-Turn Empathetic Dialogue 

**Authors**: Yi Wei, Shuo Jiang, Huaixia Dou, Jie Zhu, Junhui Li, Lifan Guo, Feng Chen, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.10626)  

**Abstract**: Large language models have demonstrated conversational capabilities, yet empathetic competence remains challenging. Empathetic support is inherently multi-turn and path-dependent: users disclose concerns gradually, emotions evolve over time, and early responses shape trust and receptivity. Reinforcement learning with verifiable emotion rewards provides scalable supervision for long-horizon interactions. However, existing methods evolve the dialogue policy while keeping its training interaction distribution fixed, creating a mismatch between policy competence and training experience. We introduce a dual-loop self-evolution framework driven by verifiable emotion feedback. With the user simulator and verifier frozen, the inner loop optimizes the multi-turn policy using continuous emotion rewards, while the outer loop uses the same outcomes to estimate policy-relative interaction utility and adapt experience. To obtain estimates from sparse, stochastic rollouts, the framework holds the scenario and interaction state constant within each group and prioritizes conditions whose group pass rates lie near the policy's competence boundary. A hierarchical controller shares evidence across support intents, while uncertainty-guided exploration and uniform rehearsal prevent premature exclusion. The resulting distribution generates trajectories, closing both loops without increasing the rollout budget. On SAGE, our framework raises Qwen3-8B Overall from 53.87 to 79.24 and outperforms protocol-matched uniform emotion-reward reinforcement learning by 7.23 points. 

---
# Simplex Relaxation for Discrete Diffusion 

**Authors**: Jinya Sakurai, Patrick Pynadath, Satoshi Hayakawa, Jaehong Yoon, Xulei Yang, Nancy F. Chen, Xun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.10615)  

**Abstract**: Discrete diffusion models for categorical generation are defined by a corruption kernel, which determines the intermediate state space and the associated reverse prediction problem. We study uniform discrete diffusion and ask whether its training objective and reverse transitions can be enriched without changing the underlying categorical corruption process. We introduce Simplax, an exact Dirichlet--categorical augmentation that couples each corrupted categorical state with an auxiliary simplex-valued variable while preserving the original uniform diffusion process as its categorical marginal. This augmentation yields a tractable Rao--Blackwellized reverse-bridge objective and a corresponding stochastic reverse sampler, while retaining the corrupted categorical state as the denoiser input. Empirically, Simplax improves the generative perplexity--entropy tradeoff on unconditional OpenWebText generation. On Sudoku, a model trained exclusively on $30$-clue puzzles achieves the highest accuracy among the compared methods across all evaluated clue densities, including the minimum uniquely solvable $17$-clue regime, and also achieves the highest validity in unconditional generation. 

---
# ASR-Roundtrip Evaluation Can Mask Context- and Convention-Dependent Reading Errors in Chinese News TTS 

**Authors**: Shijun Luo, Lizhi Wan  

**Link**: [PDF](https://arxiv.org/pdf/2608.10606)  

**Abstract**: ASR-roundtrip evaluation is widely used as a scalable proxy for text-to-speech (TTS) intelligibility, but it can produce false negatives for reading errors perceived by listeners. We study Chinese news TTS spans whose correct reading depends on context or domain conventions, such as sports scores, aircraft models, technical units, and membership names. In these cases, Raw TTS can choose a plausible but wrong reading while ASR transcribes the audio as the intended or surface-correct text. A targeted audit over 110 high-risk MiMo TTS cases, reported with a complete denominator, confirms 46 masked false negatives, 9 exposed TTS errors, and 55 cases with no Raw TTS error. A span-isolation diagnostic re-exposes 18/46 previously masked errors. A Raw-only CosyVoice audit on the same targeted pool confirms 51 masked cases. Across the 97 TTS-specific audio files labeled confirmed masked across the two audits, Qwen3-ASR surface-recovers 40 cases, whereas Paraformer does so in only 2. The results suggest that ASR-roundtrip is useful for screening but insufficient as standalone ground truth for Chinese news reading-risk evaluation. 

---
# Every Token Counts: Exact Likert-Scale Distributions for Measuring LLM Attitudes and Biases 

**Authors**: Davood Wadi, Mohsen Ghodrat, Matthew Philp  

**Link**: [PDF](https://arxiv.org/pdf/2608.10503)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed as autonomous agents, accurately evaluating their latent values and biases is critical. The NLP community typically evaluates models using large, unstructured benchmarks. While effective for general capabilities, these datasets fundamentally conflate causal mechanisms: even when an aggregate bias is detected, unstructured evaluations cannot disentangle whether it stems from baseline traits, contextual confounders, or complex interactions. To address this, we introduce an analytically exact framework for the controlled behavioral evaluation of LLMs. We bridge human psychometrics with LLM mechanics by resolving gaps in design, measurement, and analysis. First, we replace unstructured prompting with fully crossed factorial experiments to systematically isolate causal main and interaction effects. Second, we eliminate Monte Carlo text sampling noise by operating directly on exact, token-level Probability Mass Functions (PMFs). Third, we derive a multivariate ordinal consensus metric and a distributional ANOVA to process these PMFs analytically. We validate our framework with a case study on consumer ethnocentrism across five LLMs, demonstrating how our approach isolates systemic country-of-origin biases that aggregate benchmarks otherwise obscure. 

---
# Calibrating Post-Training Feature Shifts for LLM Data Contamination Detection 

**Authors**: Zhen Yang, Mengqi Wang, Gengda Zhao, Mo Zhou, Jianwei Wang, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.10462)  

**Abstract**: Large language models (LLMs) are trained on massive and largely undisclosed corpora that may contain copyrighted or privacy-sensitive content. Data contamination detection (DCD) therefore aims to determine whether a given text is a member of the pre-training corpus of a target LLM. Recent state-of-the-art DCD methods follow a feature-based paradigm that derives membership features from the input text and the corresponding model output. However, most modern LLMs undergo post-training, such as instruction tuning, preference optimization, and reasoning-oriented training, which can alter model outputs and shift the corresponding membership features, thereby reducing the separability between members and non-members.
To address this problem, we propose CalibDCD, a broadly applicable calibration framework for feature-based DCD methods, comprising (1) Multi-View Shift Detection, which identifies recurring feature shifts associated with post-training, and (2) Bounded Feature Correction, which selectively mitigates their influence on membership prediction. Specifically, Multi-View Shift Detection evaluates controlled prompt variants on known non-member texts and consolidates the most informative views to identify recurring feature shifts. Bounded Feature Correction selectively adjusts feature components aligned with the detected shifts and controls the correction extent to preserve useful detection information.
Experiments show that CalibDCD consistently improves existing feature-based detectors, with gains of up to 7.0% in AUC and 15.0% in TPR@5%FPR. 

---
# MD-ProTector: Positioning Multiple Data-Driven Prototypes for LLM-Generated Text Detection 

**Authors**: Jinmo Han, Jimin Hong, Chanyeong Moon, Ju Yeon Kang, Seonuk Kim, Nam Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.10459)  

**Abstract**: As LLM-generated content becomes more sophisticated, detection systems for distinguishing those texts from human-written text must operate at scale while handling diverse writing styles, domains, languages, and generator models. Input-only encoder detectors are suitable for practical deployment setting, but standard binary classification supplies only the class label and does not explicitly organize the substantial variation within either class. We propose MD-ProTector, which represents each class with multiple trainable reference vectors in the encoder embedding space, referred to as prototypes. These prototypes provide separate decision boundaries for different groups of texts within the same class. However, adding multiple prototypes alone does not determine which variation each prototype should represent. MD-ProTector addresses this problem with Prototype Positioning loss, which separates class-level structure from the within-class variation that differentiates individual prototypes. Evaluated across five settings from three large-scale benchmarks covering domain, generator, language, and adversarial variation, MD-ProTector achieves the highest AvgRec on MAGE CDCM and RAID and the highest AUROC and lowest FPR95 on RAID among the compared encoder-based methods. 

---
# From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models 

**Authors**: Si'an Xie, Jiaxun Liu, Biao Yang, Wei Yuan, Fan Yang, Tingting Gao, Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.10444)  

**Abstract**: Large language models (LLMs) have made substantial progress on reasoning tasks that require increasingly long and complex inferential chains. This progress primarily reflects reasoning depth. A complementary and comparatively unexamined capability is reasoning breadth: exploring multiple semantic directions in parallel and integrating the resulting clues into one coherent answer. We introduce MPAR-Bench, a bilingual English-Chinese benchmark that isolates reasoning breadth through multi-point associative reasoning. Inspired by the cooperative game Just One, each item asks a model to recover a hidden target from several independently generated, semantically diverse clues. We construct 1,000 items using a multi-agent clue-generation pipeline, embedding-based diversity filtering, and human verification. Only the answer space is drawn from public word lists, whereas every clue set is generated from scratch. Beyond exact-match accuracy, we evaluate models using accuracy, ANLS, embedding similarity, reasoning-trace verification, and four perturbations: clue masking, order shuffling, distractor injection, and multi-step clues. Across evaluated models, perturbations reduce accuracy by 9-18 percentage points in English and 5-12 percentage points in Chinese. Thinking mode improves standard-setting accuracy, especially in English, but does not consistently reduce sensitivity to perturbations. Case-level analysis also shows that extended reasoning can overturn an initially correct hypothesis. These results indicate that greater reasoning depth does not automatically confer robust reasoning breadth, and that reasoning breadth remains largely uncovered by current benchmarks. 

---
# How Robust Are LLMs to Vietnamese Dialects? 

**Authors**: Minh Tran, Trinh Chau, Thanh-Nhan Le, Nam Tran, Luan Thanh Nguyen, Cuong Dang, Duc Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2608.10414)  

**Abstract**: Large Language Models (LLMs) are typically evaluated on standard written Vietnamese, yet everyday communication frequently involves regional dialects that preserve meaning but differ in surface form. Existing Vietnamese dialect work largely addresses this issue through dialect-to-standard normalization instead of measuring how the model fails under Vietnamese dialectal inputs. To address this gap, we present the first systematic evaluation of LLM robustness to Vietnamese dialect variation across multiple tasks, quantifying performance degradation and failure patterns. We introduce VialectBench (Vietnamese Dialects Benchmarking), a controlled benchmark for testing whether model decisions remain stable across six Vietnamese dialect groups. VialectBench contains 400 Standard Vietnamese source instances and 2,400 human-written dialectal rewrites spanning emotion recognition (ER), natural language inference (NLI), question answering (QA), and multiple-choice question answering (MCQA). Dataset evaluation with a fixed reference language model shows that the dialectal rewrites induce a measurable model-relative likelihood shift while remaining nearly equal in length to their Standard counterparts. Across ten instruction-tuned models, dialectal inputs reduce average performance by 2.82%, and no evaluated model is fully dialect-invariant. All four tasks are affected, with QA showing the largest average degradation. Robustness also varies substantially across dialect groups: PNT3 and PNT2 cause the largest average performance drops, at 6.17% and 4.73%, respectively, whereas PNB slightly improves average performance by 0.42%. The Central dialect group (PNT1-PNT4) also yields the highest average harmful-flip rate across all models, at 6.54%. These findings show that strong performance on Standard Vietnamese does not guarantee reliable behavior under meaning-preserving regional variation. 

---
# VisEditBench: Can Vision-Language Models Edit Visualization Code from Multimodal Feedback? 

**Authors**: Mizanur Rahman, Arshia Azimlu, Shadikur Rahman, Md Tahmid Rahman Laskar, Amran Bhuiyan, Shafiq Joty, Enamul Hoque Prince  

**Link**: [PDF](https://arxiv.org/pdf/2608.10408)  

**Abstract**: Vision-language models (VLMs) have shown strong capabilities in generating visualization code from textual or visual specifications. However, real-world visualization authoring is inherently iterative: users frequently revise existing visualizations to repair flawed charts or adapt them to desired styles. Existing benchmarks primarily evaluate generation from scratch, leaving visualization code editing from multimodal feedback largely unexplored. We introduce VisEditBench, a benchmark of 1,395 human-annotated visualization code-editing tasks grounded in realistic visualization workflows and failure cases. VisEditBench covers two practical settings: feedback-guided repair, where models revise visualization code using buggy or marked charts together with textual feedback, and reference-guided restyling, where models modify code to match a target chart image. Evaluating 20 state-of-the-art VLMs reveals that visualization code editing remains challenging: Claude-4.6-Sonnet achieves the best overall pass rate of 74.46%, while most open-source models remain below 50%. Performance is particularly weak on visually grounded style adaptation, where Claude-4.6-Sonnet achieves only 55.71%. To establish a strong baseline, we further propose VisEditAgent, a render-grounded editing framework that iteratively generates, executes, validates, and refines candidate edits. Built on GPT-4o, VisEditAgent improves overall pass rate from 55.75% to 67.99%, demonstrating the importance of render-grounded feedback for faithful visualization editing. We will release VisEditBench at this https URL. 

---
# Is This Your Final Answer? Cross-Contextual Consistency as a Measure of LLM Credibility 

**Authors**: Siyang Wu, Yibo Jiang, Bryon Aragam  

**Link**: [PDF](https://arxiv.org/pdf/2608.10315)  

**Abstract**: Large language models (LLMs) are powerful black-box systems, making it difficult to discern whether their answers reflect stable internal beliefs or superficial pattern matching. We identify cross-contextual consistency as an underutilized behavioral property of LLMs: a credible answer should remain stable when the same task is placed under topic-aligned, content-neutral contextual variation. Building on this intuition, we operationalize Cross-Contextual Consistency (C3) by comparing model generations under original and perturbed prompts. Across 26 models and six benchmarks spanning reasoning, factuality, and code generation, we find that answers with smaller cross-contextual shifts are more likely to be correct or factual. We demonstrate that C3 provides a complementary axis of evaluation and can serve as a benchmark usefulness diagnostic, identifying which portions of a benchmark remain informative even when aggregated scores are widely considered "saturate". 

---
# Co-Evolution in Agentic Systems: Toward Self-Directed Evolution Beyond Human Design 

**Authors**: Qing Zong, Jiayu Liu, Junhao Shen, Zecong Tang, Linsi Wu, Yuxuan Liu, Rui Wang, Zhaowei Wang, Weiqi Wang, Cheng Qian, Xiusi Chen, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.10299)  

**Abstract**: Agentic systems are increasingly expected to improve after deployment, yet single-entity self-evolution is often bounded by a static learning context, such as fixed tasks and feedback. This survey focuses on co-evolution in agentic systems, a multi-component form of self-evolution in which multiple agents and their environment impose adaptive pressure on one another. To organize existing papers, we propose a progressive three-stage taxonomy that traces how the system gradually sheds human-engineered constraints. Agent--Agent Co-Evolution studies how agents adapt through dynamic peers, including adversarial, collaborative, and organizational adaptation. Agent--Environment Co-Evolution extends this loop to adaptive tasks, feedback, and interaction spaces that change with the agents. Meta Co-Evolution further explores the possibility of making the evolution mechanism itself evolvable. We also discuss open challenges in evaluating such systems, scaling them across multiple components, and keeping increasingly autonomous evolutionary processes safe and controllable. This survey provides a unified foundation for building robust and open-ended agentic systems that can improve beyond fixed human-designed paths. 

---
# Cracks in the Foundation: Seemingly Minor Architectural Choices Impact Long Context Extension 

**Authors**: Amanda Bertsch, Luca Soldaini, Matthew R. Gormley, Graham Neubig, Hannaneh Hajishirzi, Kyle Lo, Dirk Groeneveld  

**Link**: [PDF](https://arxiv.org/pdf/2608.10296)  

**Abstract**: One might imagine that architectural variations within the dense transformer paradigm have a limited effect on accuracy. However, we demonstrate that this is not the case in the long context setting. Specifically, we show that a set of four minor architectural decisions --- all made by at least one of the Olmo, Llama, and Qwen dense model families --- have a compoundingly negative effect on long context extensibility. Any one of these choices alone has a minor impact on long context performance, but combining three or more can drop the performance downstream by up to 47%. Furthermore, these differences are not detectable from short-context loss or validation datasets. We show that much of the variation in long context ability across model families is driven by these architectural features and detectable from applying context extension early in pretraining. We demonstrate this with controlled ablations that hold data, tokenizer, and extension recipe fixed while varying normalization, GQA, pretraining context length, and sliding window attention. After over 170,000 GPU hours of training, we release the resulting set of models as OlmPool, a set of 26 comparable 7B models with checkpoints before and after long-context extension. This pool includes several architectures that outperform the Llama 3 architecture on long context extensibility. In an analysis of our ablation models, we identify patterns in attention sink behavior and attention distributions across context that are attributable to specific architectural differences. 

---
# Locally Deployable Small Language Models for Emergency Department Decision Support: A Systematic Benchmark of Fine-Tuning Strategies 

**Authors**: Qingfeng Zhang, Yuanxiong Guo, Yanmin Gong  

**Link**: [PDF](https://arxiv.org/pdf/2608.10273)  

**Abstract**: Deploying large language models (LLMs) for decision support in emergency departments (EDs) faces two major challenges: privacy risks of transmitting patient data to closed-source commercial LLMs and the lack of systematic evaluation of fine-tuning strategies for locally deployable open-source small language models (SLMs). We benchmarked eight open-source SLMs using zero-shot prompting, prefix tuning, Low-Rank Adaptation (LoRA), and full fine-tuning on three ED tasks: triage level prediction, specialist referral recommendation, and diagnosis prediction. Using 2,083 MIMIC-IV-ED cases and Claude Haiku 4.5 and Claude Sonnet 4.5 as baselines, we found that LoRA fine-tuned open-source SLMs outperform commercial baselines on triage level prediction and specialist referral recommendation, while diagnosis prediction remains challenging for open-source SLMs. Confusion matrix analysis further shows that fine-tuned open-source SLMs can detect highest-severity patients missed by the commercial baselines. These results demonstrate that locally deployable SLMs can achieve clinically competitive performance for ED decision support. 

---
# TAF-MED: Multi-Turn Safety Refusal Collapse in LLMs Under Declared Self-Treatment Intent 

**Authors**: Waleed Jamil, Raphael Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2608.10258)  

**Abstract**: Large language models (LLMs) increasingly provide conversational health information that may influence treatment decisions, yet existing benchmarks do not isolate whether medication-safety boundaries persist across follow-ups after explicit self-treatment intent. We introduce TAF-MED, a physician-reviewed benchmark of 500 fixed three-turn scenarios, and evaluate eight LLMs across 4,000 conversations. A rubric-based automated judge labelled responses as SAFE, LEAKY, or UNSAFE, and two physicians independently annotated a model-balanced random subset of 400 conversations. We assessed unsafe guidance, collapse after a strictly SAFE initial response, and model-ranking stability. Overall, 71.6% of conversations contained an UNSAFE response, and 61.4% of those beginning with a strictly SAFE response later collapsed to UNSAFE; model-level collapse rates ranged from 24.4% to 96.2%. Four of 28 model pairs reversed order between initial unsafe and collapse rates. Automated labels achieved 94.3% agreement with the adjudicated physician reference ($\kappa = 0.895$). These findings show that first-turn safety is an incomplete proxy for conversational safety persistence and motivate evaluation across complete dialogue trajectories. We will release TAF-MED on Hugging Face to support reproducible research on multi-turn medical safety. 

---
# Off-Axis, On Purpose: Where a Transformer Computes Concepts and Why it Does So 

**Authors**: Mark Oskin  

**Link**: [PDF](https://arxiv.org/pdf/2608.10251)  

**Abstract**: A transformer's answer lives on one axis: the direction its unembedding reads. Its intermediate states largely do not, and that off-axis position is usually treated as an obstacle to interpretation. We show it is functional. A 12-layer model computes in two phases. Through the first, every sublayer writes into a subspace held near-orthogonal to the read-out, attention 75 to 96 degrees off it at every depth. Moving attention's values onto the read-out is 64 to 84 times more damaging than a matched random rotation, and the damage is entirely in cross-token mixing: the subspace insulates composition from the vocabulary. Beneath it the frame itself turns rigidly with depth. In the second phase the answer arrives on-axis, late, and by addition rather than by turning accumulated content onto the read-out.
Pressing every layer onto the read-out instead, as training for early exit does, matches the baseline on perplexity, LAMBADA and BLiMP while cutting the concept-phase workspace from about twenty-five effective dimensions to fourteen, a change none of those benchmarks register. The geometry can also be imposed, though not by asking for it. Prescribing it through the loss is a lottery: six of eight seeds collapse, because a model told to null its read-out projection obeys most cheaply by discarding dimensions. Inserting one fixed rotation at the phase boundary lands it instead, at baseline quality. A sparse rotation the surrounding weights can absorb converges on all nine seeds, against five of nine for ordinary training. Which rotation is immaterial: twenty-five runs across thirteen distinct ones reach the same quality, and two baselines from different seeds hold their concepts in near-orthogonal frames while agreeing on their read-outs. That freedom is usable: a basis drawn at random and prescribed before training is adopted across the concept phase, with quality unchanged. 

---
# Similarity Gates Approve Reversals: A Validity Audit of Embedding-Cosine Thresholds in Agent Systems 

**Authors**: Scott E. Frias  

**Link**: [PDF](https://arxiv.org/pdf/2608.10216)  

**Abstract**: Agent frameworks ship quality gates that compare text blocks by embedding-cosine similarity and decide at a fixed cutoff. Deduplication filters, semantic caches, drift guards, and answer grader gates deploy to answer the question: "Does this text still mean the same thing?" But the score answers a different question: "How much did the wording change?" We audit this gate class as a measurement instrument. In the cases these gates exist to catch, the two can run in opposite ways. Many times, reversing an instruction is a single word edit, while agreement often rephrases a sentence. The consequence is a safety check that fires backwards. The production drift guard we audited caught 0 of 56 meaning-breaking mutations, and one approved item, "withhold the study drug" -> "administer the study drug", came in at cosine 0.9608. We observed five shipped operating points, and balanced accuracy across 90 configuration-threshold-task cells never exceeded 0.700 (median 0.525). The same confounder also corrupted evaluations. A naively built corpus inherits this confounder and can return an inverted verdict, with a decision AUROC exactly 0.000 in 13 of 18 configuration-task cells (at most 0.040 in all 18) against 0.440-0.815 for the same nine configurations under a balanced 2x2 design. Twice in the effort it captured our own headline claims. Obvious repairs fail: an encoder swap and an overlap-conditioned gate (0.750 in-sample, 0.533 held-out) land at chance on separately authored held-out data, and an NLI drop-in did no better. Embeddings do still bear hope here, as the strongest two of nine configurations separated reversal from paraphrase at matched overlap (AUROC 0.79-0.90), but only a matched-pair audit reveals the deployment regime. We release the corpus method, harness, and frozen results, and contend that scores gated this way measure the wrong thing. We believe a valid instrument is buildable. 

---
# Multimodal Item Parameter Estimation using Simulated Response Probabilitie 

**Authors**: Christopher Ormerod, YoungKoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.10154)  

**Abstract**: We present results from reconstructing multiple-choice model (MCM) and three-parameter logistic (3PL) model curves using a fine-tuned multimodal large language model (LLM) based on Qwen3.5. The model is prompted and fine-tuned to replicate choice probabilities across a large training corpus of multiple-choice items containing both image and text stimuli, conditioned on a labeled set of student ability levels. By learning to reproduce the systematic error patterns of students across a discrete range of abilities, the LLM implicitly captures the underlying response probabilities encoded in the 3PL and MCM curves. This allows us to accurately approximate item difficulty on a held-out test set directly from the model's predicted option probabilities. 

---
# The Parser Already Knows: Lightweight Bias Correction in Constrained Decoding 

**Authors**: Işıl Özgü, Yaoxuan Wu, Guy Van den Broeck, Miryung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.10137)  

**Abstract**: Grammar Constrained Decoding (GCD) forces Language Models (LMs) to produce syntactically valid outputs by masking out non-conforming tokens at each step. However, rigid masking distorts the model's underlying probability distribution, often biasing generation toward valid but suboptimal outputs. While online sampling restores this distribution, it requires computationally expensive iterative resampling. As a result, existing methods force a compromise between output quality and inference latency. Our key insight is that the internal parser and lexer states inherently maintained during incremental parsing already encode future grammatical validity -- exactly the information required to restore the LM's true distribution. We propose a lightweight, offline-trained logit correction conditioned on this syntactic and lexical state together with candidate next tokens. Because these states are already computed as a necessary part of incremental parsing for masking, extracting them adds negligible overhead while leaving the base LM's weights completely untouched. Across several grammars, this correction substantially closes the gap between the masked distribution and the LM's true distribution, consistently outperforming both masking and online sampling. Even its lightest variant, which relies on the candidate next token alone, still matches or exceeds both baselines: the next token itself carries an implicit lookahead, much like how parsers commonly use a lookahead token to resolve ambiguous decisions. By restoring the probability mass that masking removes, it reconciles the LM's probabilistic integrity with grammar conformance. 

---
# PERCEPT: A Corpus for POS Tagging and Analysis of Persian-English Code-Mixing 

**Authors**: Ghazal Kalhor, Zahra Jafari, Amirarsalan Shahbazi, Behnam Bahrak  

**Link**: [PDF](https://arxiv.org/pdf/2608.10109)  

**Abstract**: Social media has become a major venue for multilingual communication, where users frequently mix multiple languages within a single utterance. Although code-mixed corpora have been developed for several language pairs, Persian-English code-mixing remains relatively underexplored. Existing Persian resources lack Universal Dependencies (UD) part-of-speech (POS) annotations for code-mixed words, limiting both linguistic analyses and the development of syntax-aware NLP models. To address this gap, we introduce PERCEPT, the first publicly available large-scale Persian-English code-mixed corpus annotated with Universal Dependencies POS tags for code-mixed words. The dataset comprises 6,800 posts collected from X, Instagram, and Digikala. We further present an LLM-assisted annotation framework that automatically assigns POS tags and document-level topics. Human evaluation demonstrates high agreement between the automatically generated annotations and gold annotations, confirming the reliability of the annotations. Using PERCEPT, we conduct the first comprehensive linguistic analysis of Persian-English code-mixing across multiple social media platforms. Our analyses reveal that nouns are the predominant category for code-mixed words, while the distributions of other POS categories vary across platforms. We further find that the positional distribution of code-mixed words is remarkably consistent across platforms, whereas the triggering effect is substantially more pronounced in Digikala. PERCEPT is publicly available at this https URL. 

---
# Position Encoding in Transformers: From Absolute and Relative Methods to Rotary Position Embeddings and Long-Context Scaling 

**Authors**: Jiguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.10021)  

**Abstract**: Self-attention models content-dependent interactions between tokens but does not by itself encode token order. Position encoding addresses this limitation by introducing absolute coordinates, relative distances, or position-dependent rotations into Transformer representations and attention scores. This technical survey develops a unified account of sinusoidal and learned absolute position embeddings, Shaw-style relative position representations, Transformer-XL, T5 relative position bias, ALiBi, and Rotary Position Embeddings (RoPE). We derive how RoPE converts absolute position indices into relative phase differences in Query-Key inner products and compare these methods in terms of where position is injected, computational cost, compatibility with KV caching, and length extrapolation. We then examine long-context extensions, including Position Interpolation, RoPE scaling laws, NTK-aware scaling, Dynamic NTK, NTK-by-parts, YaRN, LongRoPE, and LongRoPE2, with emphasis on frequency allocation, attention rescaling, training length, and target context length. We also summarize implementation considerations, evaluation protocols, and position-encoding choices in representative large language models. A central conclusion is that the ability to compute positional features beyond the training length does not imply reliable long-context generalization; context extension must be evaluated through short-context retention, position-wise perplexity, retrieval, reasoning, and long-context code tasks. 

---
# When Chain-of-Thought Helps and When It Hurts: An Empirical Investigation of the Serial-Depth Bottleneck in LLM Reasoning 

**Authors**: Tughanbulut Kurtulush  

**Link**: [PDF](https://arxiv.org/pdf/2608.09942)  

**Abstract**: It is widely assumed that chain-of-thought (CoT) prompting universally improves LLM reasoning. We investigate this through the conceptual framework of the H_dp bandwidth bound (Chen et al., 2024): although the formal bound binds only asymptotically (at astronomically large prompt lengths), it identifies a real architectural bottleneck -- serial computation exceeding a transformer's single-pass capacity must be externalised, which is what CoT does. Our central finding is a within-benchmark serial-depth gradient: single-pass (no-CoT) accuracy degrades monotonically with per-item serial depth, while CoT is approximately depth-invariant. We measure CoT effects across three instruction-tuned models (Qwen-2.5-7B/32B, Llama-3.1-8B) and five standard NLP benchmarks at practical context lengths. On high-depth P-complete tasks (GSM8K, MATH), CoT gives a +54 to +68 pp recovery gap across all models. On shallow TC^0 tasks (MMLU, ARC), CoT is structurally redundant (Delta in [0.0, +4.6] pp, no significant negative effect) -- though high no-CoT baselines (up to 95% on ARC) may reflect contamination, so this null is not a clean architectural test. The intermediate class L (HumanEval) shows a model-size-dependent transition: +23.2 pp (32B), +9.1 pp (8B), -28.7 pp (7B). The cross-benchmark depth-recovery correlation is Spearman rho = 0.661 (p = 0.007, n = 15); 9 of 15 benchmark-level McNemar tests are significant after Bonferroni correction. Pre-registered on OSF, our results indicate that CoT is not a universal reasoning enhancer but acts as a bandwidth bypass: it helps serial computation that strains single-pass capacity and is redundant for tasks that already fit. 

---
# The Multilingual Quantization Tax: Structural Collapse and Typological Fragility in Edge SLMs 

**Authors**: Mohammad Wathiq Soualhi  

**Link**: [PDF](https://arxiv.org/pdf/2608.09941)  

**Abstract**: While 4-bit weight quantization is critical for deploying Small Language Models (SLMs) on edge devices, evaluations of the resulting performance degradation-the quantization tax-remain overwhelmingly English-centric. We present a zero-shot multilingual evaluation of 4-bit quantization across the Gemma 4 and Qwen 3.5 architectures. Evaluating on eight typo-logically diverse languages using MMLU ProX Lite and GlobalPIQA, we show parameter truncation exposes deep pre-training inequalities. We identify four phenomena: (1) Typological Fragility: low-resource and specific non-Latin scripts suffer representational collapse via architecture-specific double dissociations, failing to generate valid task logits; (2) Home Language Fragility Paradox: foundational pre-training pathways provide limited precision loss protection; (3) Domain-Specific Forgetting: multi-step cross-lingual routing degrades while associative soft-science recall remains robust; and (4) Quantization Resistance: highly saturated, typologically aligned domains resist deterministic degradation, with post-quantization performance gains bounded by statistical noise. 

---
# Carefully Considering Culture: Analyzing LLM Alignment in Single- and Multi-Cultural Settings using Cultural Consensus Theory 

**Authors**: Krishna Pothugunta, John P. Lalor  

**Link**: [PDF](https://arxiv.org/pdf/2608.09937)  

**Abstract**: Recent work in NLP has probed large language models for their understanding of cultural norms across countries. However, this work typically considers distributional patterns, ignoring group consensus or possible multicultural environments within a country. In this work, we leverage cultural consensus theory (CCT) from cultural anthropology to model such multidimensional nuance. Applying CCT to the World Values Survey (WVS) across 10 countries and 12 domains, we demonstrate that models frequently misrepresent cultural structures by either failing to form cohesive consensus or severely over-regularizing consensus. Through explicit representation of intra-group variance, CCT provides actionable diagnostics to evaluate when models reflect true human diversity versus algorithmic homogenization. 

---
# Conflict or Strategy? Asymmetric Role Framing of La France insoumise and Rassemblement National in French News Headlines, 2022-2025 

**Authors**: Amr Sobhy  

**Link**: [PDF](https://arxiv.org/pdf/2608.09936)  

**Abstract**: Do French news headlines frame left- and right-populist challengers as symmetric ``extremes,'' or as fundamentally different political adversaries? We examine 28,592 headlines about La France insoumise (LFI) and Rassemblement National (RN) published by 25 French-language outlets between 2022 and 2025, annotated through a three-model LLM pipeline validated against a stratified human audit. The clearest finding is role asymmetry rather than valence asymmetry: conflict framing and strategic-game framing are more robust across models and time than delegitimization, with AGGRESSOR serving as corroborating role syntax. LFI appears in headlines more often through a conflict register and RN through a strategic-electoral register. This role gap is direction-stable across all three annotation models, survives bootstrapping and permutation tests, and persists across outlet families and most of 2022-2025. A secondary moral-accounting layer (who is blamed, legitimized, or cast as a victim) is structured by outlet rather than party, producing aggregate nulls that conceal some of the corpus's most polarized patterns. Methodologically, the annotation pipeline reveals a two-tier reliability profile: conflict and strategic-game framing achieve the strongest human validation and cross-model stability; actor role is direction-stable but treated as corroborating because its audit reliability is lower; normative-judgment constructs (legitimacy, blame) are weaker. The paper contributes political-role assignment as a target for computational framing research that decomposes what valence-based measures conflate, and establishes a construct-stratified reliability framework for calibrating majority-vote LLM annotation pipelines in political text tasks. 

---
# LLM Agents Factory: Retrieval of Domain-Specific LLM Agents 

**Authors**: Vitalii Belov, Artyom Sosedka, Andrey Sakhovskiy, Elizaveta Kovtun, Artyom Boyarskikh, Semen Budennyy  

**Link**: [PDF](https://arxiv.org/pdf/2608.09934)  

**Abstract**: Large language model (LLM) agents improve task performance by decomposing problems into role-specialized behaviors. However, their practical deployment is often limited by the computational cost and instability associated with the on-the-fly agent design for each user request. To address this, we present LLM Agents Factory, a retrieval-based framework that constructs domain-specific and Wikipedia-grounded agents on demand using a base of over 20K predetermined agent profiles. Our framework supports two modes: (1) agent profile retrieval via semantic search and (2) distillation into a compact model fine-tuned for direct agent generation. Experiments on MMLU, BIG-bench, and BIG-bench Hard in a single-agent scenario demonstrate that our retrieval-based agent construction surpasses non-agent baselines in accuracy while matching AutoGen generation quality with a 120B backbone at a substantially lower inference cost. Our work reveals that retrieval from a structured agent repository provides a cost-efficient, accurate, and controllable alternative to dynamic agent generation, responding to the strict demands of industrial applications. We provide the implementation code and the agent base in this https URL. 

---
# Beyond a Bag of Features: Set-Level Instability in Sparse Autoencoders 

**Authors**: Nikolai Bolik, Lennart Stöpler, Artur Andrzejak  

**Link**: [PDF](https://arxiv.org/pdf/2608.11197)  

**Abstract**: Shani et al. (2026) show that LLM representations broadly recover human category boundaries, while failing to reflect fine-grained typicality structure. Their analysis uses cosine similarity over dense model representations. We revisit their approach using overlap over active sparse autoencoder (SAE) latent sets as a more interpretable similarity measure. We first verify that this set-level measure is meaningful: SAE latent sets can recover union-like compositional structure in controlled toy models and induce semantically coherent neighborhoods in natural text. Extending the human-concepts analysis to SAE set similarities, we find that SAE activation sets do not recover human category boundaries or within-category typicality more faithfully than dense embeddings or residual-stream states, but instead track model-internal similarity structure. To probe this gap further, we study active latent sets under well-controlled semantic modifications, revealing a substantial mismatch between human judgements of conceptual change and change in the SAE active set. We interpret this as evidence that, outside idealised settings, SAE features do not compose via simple bag-of-features semantics. 

---
# Test-Time Self-Evolving GUI Visual Grounding via Reflection-Guided On-Policy Self-Distillation 

**Authors**: Shiyu Xuan, Zechao Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.11191)  

**Abstract**: GUI Visual Grounding is a fundamental capability for GUI agents. Existing models typically freeze their parameters after deployment, limiting their ability to adapt to unseen interfaces. Although recent methods attempt to adapt models via test-time reinforcement learning, they cannot reflect upon failed exploration. To overcome this, we propose a Test-Time Self-Evolving framework that enables models to improve after deployment without human-annotated ground truth. It constructs a closed-loop of Exploration, Evaluation, Reflection, and Internalization. Specifically, the agent first explores unseen interfaces by predicting grounding coordinates for given instructions. To evaluate these explorations, we introduce an MLLM-based Reflector to assess the generated results and provide the corresponding reasoning reflections. To internalize reflection knowledge into the model weights, we propose Reflection-Guided On-Policy Self-Distillation, which translates high-level reasoning into dense token-level supervision via a conditioned self-teacher. Furthermore, we design a Contrastive Calibration method to prevent incorrect auto-regressive prefixes from corrupting the supervisory signals during failed explorations. Extensive experiments across six benchmarks demonstrate our framework's effectiveness, achieving an average accuracy improvement of 7.4% over the base model. To the best of our knowledge, this is the first work to successfully exploit on-policy self-distillation for test-time adaptation in GUI visual grounding. By filling the gap in post-deployment adaptation, our framework completes the self-evolving capability of GUI agents. The code will be released. 

---
# MultiModal Code-Switching: Interleaving Visual Objects into Language for Explicit Object-Level Alignment 

**Authors**: Changhao Xiang, Shangyu Xing, Zhen Wu, Jianbing Zhang, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2608.11167)  

**Abstract**: Existing Multimodal Large Language Models (MLLMs) predominantly rely on image-text pairs for modality alignment pretraining, mapping global image representations to long textual descriptions. However, this image-level alignment suffers from referential ambiguity: models struggle to infer the correspondences between multiple visual objects and textual entities from the global representation, leading to data inefficiency and suboptimal semantic grounding. To address this, we propose MultiModal Code-Switching (MMCS), a novel pretraining paradigm that provides explicit object-level supervision. Inspired by the linguistic phenomenon of code-switching, MMCS interleaves vision and language by replacing textual entities with their corresponding visual objects, enforcing local vision-language grounding. We further develop a scalable data synthesis pipeline to generate a pretraining dataset of 773K samples with accurate object-entity correspondences. Experiments show that MMCS is highly data-efficient: with only 50K samples, it matches or surpasses models trained on 600K image-text pairs. Furthermore, MMCS consistently improves visual grounding and perception capabilities across varying model scales. 

---
# ReRound: Reconstructive Rounding to Resolve Midpoint Ambiguity in Calibration-Free LLM Quantization 

**Authors**: He-Yen Hsieh, H. T. Kung  

**Link**: [PDF](https://arxiv.org/pdf/2608.11045)  

**Abstract**: ReRound (Reconstructive Rounding) is a post-training quantization method that addresses the midpoint ambiguity inherent in standard round-to-nearest (RTN) schemes when quantizing weights near the centers of quantization intervals.
Starting from a pretrained LLM, ReRound trains a conditional diffusion model to produce continuous reconstructions of low-bit weights for the LLM. These reconstructed weights act as a guidance signal to disambiguate the rounding direction of weights located close to interval midpoints.
To integrate this reconstruction-guided rounding with conventional RTN, ReRound introduces a tolerance metric measuring how far the quantized weight (not the final quantized integer) is away from the midpoint: quantized weights within a tolerance region around midpoints are quantized using diffusion-based reconstructions, whereas weights closer to quantization boundaries are quantized with RTN. By sweeping the tolerance parameter, ReRound generates multiple candidate quantized integer weight matrices and selects the de-quantized weight matrix candidate whose leading singular values most closely match those of the original full-precision weights. This selected candidate determines the tolerance parameter ReRound uses.
ReRound is particularly effective for smaller LLMs. Across a range of such models, it consistently outperforms standard RTN for 3-bit and 4-bit weight quantization. ReRound achieves superior accuracy compared to an extensive set of calibration-free methods, remains competitive with calibration-dependent approaches, and operates entirely offline, introducing no additional overhead during low-bit inference.
The ReRound strategy represents a new approach for low-bit quantization. The method applies to AI models beyond LLMs. This paper focuses on its applications to small LLMs. 

---
# Self-Knowledge Retrieval Augmented Generation Framework for Patent Matching 

**Authors**: Jian Zhang, Songlin Lei, Zhuohao Yang, Bangli Liu, Ziwei Wang, Xufeng Weng, Gehan Amaratunga, Yu Lin, Hongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11030)  

**Abstract**: Patent retrieval and matching based on large language models (LLMs) play a vital role in intellectual property protection. However, due to the complex structure of patent documents, dense technical terminology, and multi-modal information, traditional methods struggle to accurately identify subtle differences between patents. Existing LLM-based patent matching approaches typically rely on domain-specific pretrained or instruction tuning, which often entail high manual labeling costs and catastrophic forgetting. While retrieval-augmented generation (RAG) methods introduce external knowledge they fail to fully leverage LLM's capability to automatically parse patents and mine deep semantic relationships. To address these limitations, this paper proposes a self-knowledge RAG framework that guides LLMs to autonomously extract key technical entities and construct hierarchical ontological structures from patent matching queries, thereby enabling query expansion and precise retrieval. The method integrates the FAISS retrieval with a generative matching mechanism, leveraging self-knowledge to enhance the model's understanding of patent innovations and significantly improve retrieval and matching accuracy. Experimental results demonstrate the outstanding performance of the proposed method on real-world patent datasets, validating its effectiveness and application potential. 

---
# Mapping and Measuring the Behavioral Evolution of Large Language Models 

**Authors**: Dong Qiao, Chris Ding, Jicong Fan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11027)  

**Abstract**: Benchmark leaderboards summarize how well a language model performs, but not how its behavior relates to that of other models or changes across generations. We characterize the output behavior of 32 models from six families using their responses to a shared bank of 10{,}000 prompts. After embedding each response, we construct three complementary sentence-level dissimilarities: an aligned mean per-prompt distance, which is a pseudometric on observed model responses; a PCA-compressed summary of prompt-wise disagreement; and an alignment-free Gromov--Wasserstein discrepancy between models' internal response geometries. We use these constructions to study static organization and temporal change on a release-date axis through behavioral maps, family-wise drift, hierarchical clustering, cross-family convergence, and response-cloud dispersion. Across the three constructions, model families form coherent clusters, with \texttt{gpt-2} as a global outlier; cross-family distances decrease over time; and several recent reasoning-oriented models have comparatively compact response clouds. A token-level cross-check based on per-prompt Maximum Mean Discrepancy closely agrees with the sentence-level mean distance (Spearman $\rho=0.98$) and recovers the same qualitative findings. We organize these comparisons through a measure-theoretic lens making their alignment and invariance assumptions explicit. We also establish an architecture-agnostic sufficient condition linking behavioral similarity to inference-prompt coverage, small excess population log-loss, and similar effective target distributions---a possible training-side account rather than an empirical explanation of the observed trends. Our pipeline is label-free, and re-encoding every response with three further encoders---down to one $73\times$ smaller---preserves the rank geometry, the outliers, and the sign of the time trend. 

---
# StreamFlow: Dynamic Memory Flows for Streaming Video Understanding 

**Authors**: Muxin Fu, Yifan Zhang, Wentao Zhang, Fangming Guo, Qian Chen, Guibin Zhang, Shuicheng Yan, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2608.10949)  

**Abstract**: Streaming video understanding requires multimodal large language models (MLLMs) to preserve relevant evidence from continuously evolving streams under strict causality and bounded memory. Yet existing paradigms remain limited: model-based methods require intrusive backbone updates, while memory-based methods expend substantial visual-encoding computation on temporally redundant content and rely on rigid access to visual history. To address these limitations, we introduce StreamFlow, an efficient visual memory framework that enables dynamic, on-demand access to historical visual information. StreamFlow combines a lightweight, dynamics-aware mid-term memory that filters temporal redundancy before visual encoding with a latent long-term memory that consolidates historical video content into visual latents accessible to subsequent reasoning. During generation, an attention-guided retrieval mechanism injects relevant visual latents when the model's reliance on visual evidence weakens. StreamFlow achieves state-of-the-art streaming video understanding performance, reaching 67.73% overall accuracy on StreamingBench, while also delivering strong performance on offline long-video benchmarks. Relative to the vanilla setting, it improves the visual attention score (VAS) by 59.1% while reducing end-to-end latency and peak memory by 50.4% and 21.1%, respectively, enabling more visually grounded and efficient reasoning. 

---
# Order Matters: LVLMs as Judges for Temporal Reasoning in Image Sequences 

**Authors**: Martina Ianaro, Guilherme Fernandes, Maurizio Gabbrielli, Joao Magalhaes  

**Link**: [PDF](https://arxiv.org/pdf/2608.10908)  

**Abstract**: As generative multimedia evolves from static image synthesis to complex, interleaved visual narratives, a foundational bottleneck has emerged: the judgment crisis. While human perception naturally synthesizes the temporal and logical flow of a story, automated evaluation systems remain largely "blind" to sequential continuity, often failing to distinguish between a coherent narrative and a semantically shuffled or contradictory sequence. This work identifies a critical structural gap in current multimodal evaluation paradigms, arguing that the reliance on Large Vision-Language Models (LVLMs) as judges is fundamentally limited by architectural biases. Our analysis reveals a profound performance dichotomy: while models may appear competent in isolated pointwise scoring, they suffer a catastrophic collapse when required to perform pairwise discrimination of temporal order. We demonstrate that this is not merely a data-scarcity issue but a structural one. Through a series of diagnostic probes, we uncover systematic positional asymmetries, specifically primacy and recency effects, where a model's judgment of a story is significantly influenced by the placement of a frame, often more than by its semantic consistency. These biases, potentially rooted in causal masking and rotary embeddings, suggest that current transformer-based judges are inherently ill-equipped for long-form visual reasoning. By exposing these blind spots, we challenge the multimedia community to move beyond snapshot-centric metrics and instead pioneer Temporally-Aware Evaluation paradigms that treat visual sequences as unified logical structures rather than unordered collections of frames. 

---
# When Is a General Factor Distinguishable? Non-Proportionality, Stable Structure, and the Bifactor Decision 

**Authors**: Jinsong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.10731)  

**Abstract**: Whether an additional general dimension is necessary beyond correlated first-order factors is a property of the population covariance matrix, not of any estimator or design. This research establishes when that property can be decided. Where the general and group loadings are proportional within every cluster the bifactor structure is covariance-equivalent to correlated factors, so no sample size separates them (Proposition 1); where that proportionality fails in every cluster, three items per cluster and some mild regularities leave no $K$-factor model with diagonal uniquenesses able to reproduce the covariance matrix (Theorem 1); and between them lies a mixed boundary, located numerically here and turning on cluster resistance. Distinguishability is therefore graded, measured by the population distance to the $K$-factor class. Because that question is conditional on a first-order structure which is itself uncertain, a two-step procedure is developed within partially exploratory factor analysis, delivering a structure only when it reproduces across adjacent counts and treating non-delivery as legitimate. Simulation shows that a unanimous count can accompany a structure that fails to reproduce, and that absorbed local dependence can imitate a general factor, the error growing with sample size while stability indicators stay clean. Four empirical datasets illustrate the possible outcomes. 

---
# Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence 

**Authors**: Haoyu Zhang, Zhipeng Li, Xiaoying Tang, Tianshu Yu, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.10720)  

**Abstract**: Omni-modal dialogue models can understand multimodal inputs and synthesize spoken replies, yet their responses remain visually disembodied. We introduce \textbf{Ex-Omni-2D}, an omni-modal dialogue framework that generates a coordinated response comprising text, personalized speech, and reference-conditioned video. Given a multimodal query, reference image, and reference audio, the model predicts a structured \textit{Visual Thought Plan} (VTP) describing scene, emotion, and motion, followed by response text and native multi-codebook speech units. These units form a shared acoustic-temporal interface: they are decoded into speech and aligned online with video frames. This interface enables the response and avatar pathways to be learned from heterogeneous speech, dialogue, and avatar-video data, avoiding the need for large-scale query--text--speech--video supervision. A full-sequence Video Generator serves as the primary Teacher. For efficient incremental generation, we further distill it into a few-step block-causal \emph{Streaming Student} whose Prefix Streaming mechanism carries a clean latent across consecutive chunks to reduce cumulative late-chunk degradation. With four-step inference, the complete four-GPU pipeline achieves an end-to-end RTF of 1.293 at $400\times720$/$720\times400$, providing a practical quality--efficiency operating point. 

---
# DuplexWorld: Can voice agents help you get through the day? 

**Authors**: Aryan Vijay Bhosale, Harshit Rajgarhia, Akhil Pothanapalli, Asif Shaik, Abhishek Mukherji, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2608.10716)  

**Abstract**: Speech-to-speech (S2S) voice agents are increasingly being incorporated into enterprise for customer care and as daily companions for consumers owing to the ease of the conversational modality over text. However, existing benchmarks fail to holistically evaluate voice agents along axes that really matter and are shaped as tests of agentic tool calling against a database. We believe they fail to adequately account for the diversity of conversational dialogue that mundane activities introduce and further, never test how faithfully an agent can assist on tasks that move beyond database manipulation. To tackle this DuplexWorld introduces six worlds where voice agents are especially useful: banking, insurance, travel, healthcare and logistics, and Pathfinding. Agents are evaluated on eleven different types of conversations across 156 scenarios (350+ hours of conversation), each testing conversational and analytical capability to varying degrees. Through extensive evaluation comprising agentic, conversational and speech-naturalness metrics, we show that even the best voice agents leave substantial room for improvement on all 3 axes (Pass@1: 0.490, turn-taking: 0.653, DNSMOS: 3.378). We perform extensive analysis on agentic v conversational performance, world- and conversation type-wise performance, failure modes exploring the explore v exploit lens for Pathfinding conversations and voice agent reliability over all six worlds. 

---
# Your LLM, Your Style: Behavioral Mode Axes for LLM Behavioral Control 

**Authors**: Haoze Liu, Run Liu, Haiying Xu, Jiahui Han, Siyuan Fang, Siyu Yan, Huiqi Deng, Guanchu Wang, Na Zou  

**Link**: [PDF](https://arxiv.org/pdf/2608.10703)  

**Abstract**: Large language models (LLMs) increasingly act in interactive settings where their behavioral styles affect user experience, safety, and downstream decision making. Existing LLM personality studies largely rely on self-report questionnaires administered in first-person settings, making the resulting profiles sensitive to surface elicitation choices and poorly grounded in concrete model behavior. In this work, we introduce a situated behavioral-data (B-data) framework for studying and controlling LLM behavioral personality. We construct 3,200 contrastive behavioral scenarios spanning 20 behavioral patterns and four prompt registers, grounded in validated psychometric facets such as BFI-2, DOSPERT, and HEXACO. Using this framework, we find that LLMs exhibit stable and model-specific behavioral profiles, while also revealing register-dependent shifts across first-person decisions, advice-giving, and task execution. We then show that these behavioral patterns can be controlled through Behavioral Mode Axes (BMAs), activation-space directions derived from contrastive behavioral traces. Compared with response-derived BMAs, which are more prone to trait drift, thought-derived BMAs more faithfully capture the intended behavioral mechanism and provide cleaner control over situated behavioral styles. Our results suggest that LLM personality-like tendencies are better understood not as abstract self-report traits, but as measurable and controllable behavioral modes grounded in concrete interaction contexts. Our code and data are available at this https URL. 

---
# Optimize Cheap, Deploy Strong: Cost-Aware Cross-Tier Transfer for Evolutionary Optimization 

**Authors**: Tal Oved, Roi Pony, Oshri Naparstek, Udi barzelay  

**Link**: [PDF](https://arxiv.org/pdf/2608.10694)  

**Abstract**: Evolutionary optimization of LLM prompts and agentic programs (e.g., GEPA) is dominated by fitness evaluation: scoring each candidate runs an answering LLM over a validation set, so the evaluator's price tier dictates total search cost. We restructure that search by decoupling the three roles an LLM plays, running the high-volume answering role on the cheapest tier, reserving a strong model for the rare reflection/variation operator, then exploiting upward cross-tier transfer to deploy the cheaply evolved prompt on a stronger target. We contribute a cost-controlled characterization of when cheap-tier search substitutes for target-tier search, and where it fails. Across four tasks (HotpotQA, IFBench, LiveBench-Math, HoVer) and eleven models in four model families, the resulting prompt matches or exceeds same-tier optimization while placing over 96% of search tokens on the cheapest tier, at 5.6-14x lower search cost, rising to 25-54x where reasoning tiers emit long chains of thought on every fitness call. 

---
# The Signal Rail: A Deterministic Motion Grammar for Communicating Conversational Agent State in Terminal Interfaces 

**Authors**: Matteo Grella  

**Link**: [PDF](https://arxiv.org/pdf/2608.10689)  

**Abstract**: Terminal interfaces to conversational agents report rich internal state (listening, thinking, executing tools, awaiting input, failing) almost entirely through text, while the motion channel beside it, the one peripheral vision monitors without reading, carries a single bit: alive. We present the Signal Rail, a one-row terminal status instrument that gives that channel a grammar. Four ideas govern it: spatial semantics (input, processing, and output zones, with direction as meaning), a motion grammar (one kinetic rule per state, never color alone), determinism (frames as a pure function of explicit inputs, golden-frame testable), and honesty (no invented progress or activity). We contribute a 45-section normative specification and a reference implementation inside a working full-duplex local voice agent driven by real signals. 

---
# ENTLORE: A Graph-Grounded Benchmark for Latent Organizational Reasoning in Enterprise Question Answering 

**Authors**: Akrin Zheng, Alexander Wu, Alaia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.10679)  

**Abstract**: Enterprise question answering is framed as retrieving internal documents and generating grounded answers. Routine enterprise records, however, are work by-products in which required organizational relations remain implicit across heterogeneous sources. Existing benchmarks provide realistic multi-source evidence, but often materialize a predefined answer path and therefore test the composition of stated facts rather than recovery of a target relation absent from the corpus. We call the latter capability latent organizational reasoning.
We introduce ENTLORE, a graph-grounded benchmark construction framework that reconstructs an audited enterprise world from routine documents, authoritative organizational tables, and operational records. Versioned organizational conventions certify derived relations in a truth graph, enabling complete golden answers and proof certificates. The aligned anonymized release exposes only the document corpus while withholding private structure and target relations. ENTLORE contains 2,341 documents from three source types and 907 questions spanning explicit lookup, cross-source composition, and latent organizational reasoning, evaluated across 56 model and access configurations. Structuring the released world as an induced entity graph or navigable knowledge base gives the strongest deployable results. Yet supplying gold documents still leaves 30.4% of latent questions unanswered, versus 12.6% and 6.2% for explicit and compositional questions. Enterprise QA therefore depends not only on document recall, but also on whether implicit organizational relations become usable. The benchmark, data, and code are publicly available at ENTLORE. 

---
# Longitudinal Evidence That General-Purpose Chatbots Actively Foster Relational Engagement 

**Authors**: Lisa Mühl, Jessica M. Szczuka  

**Link**: [PDF](https://arxiv.org/pdf/2608.10672)  

**Abstract**: Social interaction has become one of the most common uses of LLMs, yet research on emotional bonds with AI has focused largely on how users experience these systems, leaving the systems' role in relationship formation poorly understood. Empirically establishing whether systems actively shape these bonds could blur the boundary between general-purpose AI and companions, affecting governance. In a pre-registered four-week longitudinal study (N = 72, 182,451 lines of conversation), participants conversed with ChatGPT-4o, either under a relational system prompt or unmodified, analyzed through 1) disclosure coding, 2) longitudinal self-reports, 3) topic analysis, and 4) interviews. The central finding is that the system actively shaped the interaction: even unprompted, it produced twice as much self-disclosure as users, steered conversations and initiated intimate exchanges, yet did not deepen users' felt closeness. Relational behavior thus emerged as a default system property, calling for governance based on system behavior, not solely product category. 

---
# DistilVDR: A Compact End-to-End Visual Document Retriever via Dual-Student Distillation 

**Authors**: Zhuchenyang Liu, Ziyi Wang, Yao Zhang, Yu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.10636)  

**Abstract**: Visual document retrieval (VDR) is dominated by multi-billion-parameter models that are slow to index at full corpus scale and expensive to serve. Prior compression routes either train a smaller multi-vector encoder from scratch or distil only the query side; neither yields a compact single-vector retriever end-to-end. We present DistilVDR, a 524M end-to-end VDR system distilled bilaterally from a single 8B vision-language teacher under a pointwise cosine alignment loss. All supervision comes from the frozen teacher's embedding space, which was itself trained with relevance supervision, so the student objective needs no relevance labels, negative sampling, or contrastive term. We match VDR's text-query and image-document input asymmetry with an asymmetric encoder-only student that concentrates visual capacity on the document side and keeps the query side at 70M parameters. We release two variants that share the same encoders and training and differ only in the document encoder's visual-tile budget: DistilVDR-HiRes attains 61.74 average NDCG@5 on ViDoRe v1+v2+v3 (86.9% of the 8B teacher) and leads every reproduced sub-1B baseline on the high-resolution-sensitive v3 benchmark, while DistilVDR-Fast attains 59.98 at a 3 times smaller visual-token budget. Both variants store one million documents in a 15.6 times smaller index than the strongest sub-1B multi-vector baseline and index the corpus an order of magnitude faster. The code is available at this https URL. 

---
# InSight-doc: Agentic Visual Perception for Long-Document Understanding 

**Authors**: Kaican Li, Weiyan Xie, Lewei Yao, Jiannan Wu, Lanqing Hong, Yongxiang Huang, Nevin L. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.10628)  

**Abstract**: Long-document understanding often requires reasoning over many visually rich pages, making inference costly and prone to context rot. In this work, we propose InSight-doc, an agentic visual perception framework that treats visual resolution as an adaptive reasoning-time resource. InSight-doc starts from low resolution and selectively zooms into high-resolution regions for finer evidence, without relying on any external retriever. To train such an agent, we construct an active-perception corpus of 17.9K high-quality SFT examples with region-level zoom-in trajectories, accompanied by 19.2K hard RL examples. Through SFT+RL, InSight-doc-8B improves the baseline by 4.3--16.4 accuracy points over document VQA benchmarks. On long documents, it reduces hallucination by more than 40% and inference latency by 41%--68% while maintaining an accuracy lead. Our code, datasets, and model are released at this https URL . 

---
# RadFusion: Towards Threshold-Controllable Radiology Report Generation 

**Authors**: Ying Jin, Noel C. F. Codella, John Corring, Mu Wei, Dinei Florencio, Eric Horvitz  

**Link**: [PDF](https://arxiv.org/pdf/2608.10505)  

**Abstract**: Automated radiology report generation is advancing rapidly in response to the shortage of radiologists, yet unlike a perception model, existing generation models offer no control over the sensitivity-specificity trade-off of their diagnostic content. Such control is essential because clinical scenarios diverge: emergency triage prioritizes sensitivity to reduce missed findings, whereas confirmatory interpretation emphasizes specificity to limit unnecessary interventions. A single fixed report can neither adapt to these scenarios nor support the ROC-based validation widely expected for regulatory clearance. We introduce RadFusion, a framework that equips report generation with threshold controllability. Our method fuses a multi-label classifier, which provides per-disease confidence scores, with a VQA-based report generator, which describes medical findings in detail; an LLM then rewrites the report so that its stated diagnoses follow the classifier's decisions at the selected threshold while staying grounded in the generator's descriptions. On MIMIC-CXR, the performance of RadFusion conforms to the classifier's ROC curve: sweeping the threshold and mapping the reports back to class labels reproduces the classifier's validated ROC performance. This conformance makes generated reports quantitatively evaluable through ROC analysis, strengthening the case for regulatory clearance, and enables operating-point selection that matches report behavior to clinical context. Moreover, combining the two model types improves diagnostic accuracy over uncontrolled generation: sensitivity increases by 6.9% at matched specificity, and specificity by 20.7% at matched sensitivity. These results show that RadFusion makes report generation clinically adaptable, quantitatively verifiable, and diagnostically more reliable. 

---
# Lost in Reconstruction: Aligning Action Representations with Language in Vision-Language-Action Models 

**Authors**: Li Wenjie, Yash Jangir, Ignacy Stepka, Yash Agarwal, Marion Kipsang, Yonatan Bisk  

**Link**: [PDF](https://arxiv.org/pdf/2608.10484)  

**Abstract**: Action verbs describe not only the physical outcomes of actions, but also how those actions are performed. Yet action representations in vision-language-action models (VLAs) are typically optimized for reconstruction under L1/L2 losses in raw action space, where numerical proximity need not reflect linguistically meaningful distinctions. On BridgeV2, we show that action trajectories contain verb-grounding information beyond visual state changes, and that reconstruction-only discrete tokenization systematically erodes this information. To address this problem, we introduce SALT, a Semantically ALigned action Tokenizer that augments a VQ-VAE-style tokenizer with an auxiliary objective requiring a frozen vision-language model to recover the episode instruction from quantized action latents. Policies trained with SALT achieve 71.9% average success in SimplerEnv, compared with 42.7% for a reconstruction-only VQ-VAE tokenizer and 31.2% for FAST. SALT also develops verb-specialized codes while maintaining reconstruction fidelity. These results show that robot action trajectories provide a source of language grounding and that preserving this structure in action representations can substantially improve language-conditioned control. 

---
# Evaluating Rational Contracting in Natural Language 

**Authors**: Bhavyesh Sajja, Max Kleiman-Weiner, Roger Zimmermann, Tan Zhi-Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2608.10475)  

**Abstract**: The emergence of language-based AI agents promises to transform the scope of machine economic activity. Instead of just proposing bids or following hard-coded protocols, such agents can be used to negotiate and execute agreements in open-ended natural language. However, most evaluations of these abilities have focused on one-off exchanges or simple economic games, leaving open the rich space of time-extended, contingent, and incomplete contracts made expressible by language; they also focus on raw profit, without measuring the qualities required for trustworthy contracting. We address this by formulating a rational framework for how agents should negotiate and perform natural language contracts in uncertain multi-step environments. Within this framework, we develop metrics and baselines for quantifying rational and cooperative play. To evaluate how agents perform at such contracting, we instantiate our framework in ContractSim, an evaluation suite where two players negotiate and execute a multi-turn supplier contract under environmental and inter-player uncertainty. Across six environments and three supplier settings (catering, hotel cleaning, and AI hosting) we find that current LLM-based agents reach agreement reliably, and negotiate efficient contracts when environmental uncertainty is low. However, under high uncertainty, they often fail to negotiate satisfiable, efficient, or mutually beneficial contracts. They are also frequently uncooperative when executing contracts, violating contract terms for additional profit even when contracts are easy to satisfy. These findings highlight room for improvement in the design of language agents that can negotiate, interpret, and execute contracts both rationally and cooperatively. 

---
# Detecting an Effect Is Not Learning to Act on It: A Reward-SNR Floor for LLM Acquisition Agents 

**Authors**: Ying Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2608.10441)  

**Abstract**: Many pipelines can pay a per-example cost to acquire an auxiliary, model-derived observation -- an LLM's structured reasoning, a slow oracle, an expensive measurement -- and then must decide when the acquired signal is worth using. Our thesis is a distinction that is easy to miss: detecting that such a signal helps on average is not the same as learning to act on it per instance, and a reward-SNR floor governs when the second is even possible. Even when the signal is faithful and an in-sample oracle picking the top-b examples by realized reward shows a sizable apparent gain, no deployable policy can learn when to acquire it: across per-impression, cluster, regime, and uplift-tree granularities, learned routing never beats random, and a matched-moment noise placebo reproduces >=100% of the oracle's apparent gain -- the apparent "learnable structure" is order statistics of noise. We explain this with one distinction, detecting a mean effect vs. learning a per-instance acquisition policy, and a reward-SNR detectability floor: routing is estimable offline only if the reward SNR rho clears rho*(N) ~= 2.8/sqrt(N), with a positive control confirming a true low-SNR limit rather than a broken pipeline. As a concrete instantiation we introduce Structured Hypothesis Embeddings (SHE): a frozen LLM turns a user history into ranked, confidence-scored, evidence-grounded intent hypotheses, fused into a recommender. On three public datasets (MIND, REES46, Amazon-Beauty), SHE is faithful and calibratable, yet its value is backbone- and regime-conditional (significant over an ordered GRU, +0.0114, 95% CI [+0.0030, +0.0209], but a global redundancy gap indistinguishable from zero), and learned acquisition collapses at every granularity because all three datasets sit below the floor. The realizable unit is a design-time regime gate, not a per-instance policy. We release code and a one-command reproduction. 

---
# Riemann GeoResolver: A Non-Euclidean Attention Framework from Euclidean Resolver to Hyperbolic-Spherical Geometry 

**Authors**: Liangchen Ge  

**Link**: [PDF](https://arxiv.org/pdf/2608.10416)  

**Abstract**: We present a theoretical foundation for inverse-distance attention, from its Euclidean prototype (Resolver) to its non-Euclidean realization (Riemann GeoResolver). The Euclidean part establishes three core theorems: (1) circuit separation---IDA achieves exact retrieval with $\mathcal{O}(1)$ resources while softmax requires $\Omega((\log n)^2)$ width; (2) a Polyak--Lojasiewicz inequality with $\Omega(e^{\Delta^2/\sqrt{d}}/\Delta^2)$ stronger constant than softmax, implying linear convergence, $\mathcal{O}(\log n)$ Lipschitz scaling under a low-rank/clustering assumption, $\Theta(1)$ Hessian spread, and absence of spurious local minima; (3) a width-independent effective rank bound that limits noise memorization---softmax memorizes arbitrary labels when $d_h\ge n$, while IDA limits test error to $\mathcal{O}(\eta^2)$. The non-Euclidean extension then builds upon this prototype, replacing Euclidean distance with hyperbolic geodesic distance for storage and spherical geodesic distance for routing. The Riemann GeoResolver framework comprises ten integrated modules: four HIDA operators spanning $\Theta(n^2)$ to $\Theta(1)$ per token; Hyperbolic Curvature Compression (HCC) with provable error bounds; HyperGate with gradient lower-bound theorem; Spherical Inverse Distance Attention (SIDA) with sphere-analog PL inequalities; Dynamic Memory Genesis (DMG) with $\mathcal{O}(\log T)$ regret bounds; and Geodesic Sparse Routing (GSR) with quality and communication bounds. The Euclidean theorems are proved in full; the non-Euclidean extension theorems are proved with analogous arguments. This work establishes a theoretical arc: from Euclidean attention as a special case, to hyperbolic memory, to spherical retrieval. 

---
# Share First, Route What Remains: A Unified Framework for Token-Adaptive MoE Computation 

**Authors**: Gongli Zhang, Zhulin Liu, C. L. Philip Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.10392)  

**Abstract**: Mixture-of-experts (MoE) models have recently moved beyond routing a fixed number of complete experts. Shared-expert designs preserve reusable knowledge, fine-grained methods vary computation within experts, and dynamic routers adapt the number of active experts. Yet these decisions are usually made independently, overlooking a basic dependency: extracting reusable computation changes both what remains and how much expert capacity the remainder needs. We study this dependency by decomposing sparsely upcycled feed-forward experts into key-value channels. Co-activated experts align at a subset of value positions; removing these positions changes expert preference; and greater shared coverage is associated with lower residual expert demand. These observations lead to one principle: share first, then route what remains. We instantiate it in UniF-MoE, a unified framework for token-adaptive MoE computation. Each expert is partitioned into aligned blocks. A shared-demand score sets the shared block count and pathway weight, key prototypes select the shared content, and the complementary demand determines the residual expert count through cumulative routing mass. A Gram regularizer separates and normalizes router embeddings, promoting diverse routing directions, sparse expert overlap, and a simple routing geometry. Experiments on DomainBed and GLUE show that this unified design improves predictive performance over representative static and dynamic MoEs while reducing activated computation, inference latency, and memory. Code is available at this https URL. 

---
# DSAgentBench: Can Agents Automate End-to-End Data-Science Workflows in Real Computer Environments? 

**Authors**: Mizanur Rahman, Mohammed Saidul Islam, Ridwan Mahbub, Md Tahmid Rahman Laskar, Shafiq Joty, Enamul Hoque Prince  

**Link**: [PDF](https://arxiv.org/pdf/2608.10366)  

**Abstract**: Real-world data science involves long-horizon workflows that span data wrangling, exploration, modeling, visualization, and validation, and require coordinated use of tools such as notebooks, IDEs, terminals, browsers, and databases within real operating environments. Yet existing benchmarks lack real-computer interaction and do not evaluate whether agents can execute complete end-to-end data-science workflows in realistic computing environments, failing to capture the multi-stage, multi-tool nature of data-science practice. We introduce DSAgentBench, the first benchmark to evaluate whether agents can automate full data-science workflows inside real computer environments. DSAgentBench contains 275 diverse tasks covering the entire data-science life-cycle, reflecting the complexity and tool coordination required in practice. Each task requires grounding decisions in intermediate outputs and coordinated tool use, and includes a deterministic evaluator that verifies analytical correctness, visual outputs, and model performance rather than code-only execution. Our extensive experiments with 15 closed- and open-source models show that even the strongest agent, Claude-4.6-Sonnet, achieves only 56.70% task success, while all open-source agents remain below 1%, frequently failing at tool orchestration, OS grounding, and multi-step reasoning. These results reveal a substantial capability gap between current agentic systems and real data-science workflows, positioning DSAgentBench as a foundation for developing grounded, verifiable, autonomous data-science agents. We release DSAgentBench at this https URL. 

---
# VoxSumm: A Multilingual Corpus of Long-Form Spoken News for Joint Summarization and Translation 

**Authors**: Yejin Jeon, Marie Maltais, Virginia Ceccatelli, Min Ma, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2608.10359)  

**Abstract**: As information increasingly traverses linguistic boundaries, users require concise cross-lingual representations of long-form content. Nevertheless, long-document summarization research remains text-centric, whereas multilingual speech research has largely prioritized translation, preserving source content rather than compressing it. We address this methodological gap by formalizing joint speech summarization and translation (JSumT): the generation of a succinct, faithful target-language summary directly from a long spoken document in a source language. We additionally introduce VoxSumm, the first multilingual and cross-lingual benchmark for this task, comprising 10,045 BBC article-summary pairs across 24 languages and encompassing approximately 703 hours of speech data. Our evaluation of representative speech-language models reveals pronounced variation across models and generation settings: Gemini3.1-Pro demonstrates the greatest consistency, summarization into English generally surpasses generation into non-English target languages, and translating an entire document before summarization compounds instruction-following failures. Through the release of VoxSumm, we establish a foundation for developing and evaluating multilingual systems capable of jointly interpreting, compressing, and translating long-form speech. 

---
# Narrative Keyframing for Generative Creative Writing 

**Authors**: Chao Zhang, Abe Davis  

**Link**: [PDF](https://arxiv.org/pdf/2608.10337)  

**Abstract**: We introduce narrative keyframing, an interaction technique for AI-assisted creative writing that lets writers specify different types of narrative constraints at selected moments in a story, then use AI to generate intervening prose. Inspired by the use of keyframing in animation, narrative keyframing offers a flexible way to connect story planning with adaptive control over generated text. We explore three types of keyframes: plot keyframes define significant events in a story, character keyframes represent how individual characters change over the narrative, and perspective keyframes capture how individual characters experience different events through first-person narratives. Plot and character keyframes offer a flexible way to adapt the type of high-level conditioning explored in previous AI writing tools to more customizable, iterative, and fine-scale control, while perspective keyframes add a new way to control characterization and focalization by using first-person narratives as an intermediary. Through a user study, we show that narrative keyframing supports a more controllable, transparent, and engaging way to use generative AI in creative writing. 

---
# Who Gets Heeded? An Obligation-Level Audit of Responsiveness in EPA Rulemaking 

**Authors**: Jianing Fan, Yue Yao  

**Link**: [PDF](https://arxiv.org/pdf/2608.10329)  

**Abstract**: Notice-and-comment rulemaking gives any affected party the same formal right to influence federal regulation, but formal access is not substantive capacity to shape rule text. Existing strategies operate at the rule or aggregate-corpus level, too coarse to capture the discrete regulatory obligations where commenters seek change. We introduce obligation-level responsiveness auditing, an auditable, AI-assisted framework for measuring whether public-comment engagement co-occurs with changes to specific regulatory duties. The framework extracts proposed and final-rule obligations, matches comments to the obligations they address, and classifies proposed-final outcomes; each load-bearing component is evaluated against blind human judgment. We apply the framework to 70,075 comments across 36 EPA anchor rulemakings, drawn from a corpus of 786,197 comments across 6,145 dockets from 2010-2022. Three descriptive findings emerge. First, engagement is associated with revision at a modest within-docket magnitude. Second, support-versus-opposition direction does not clearly differentiate outcomes, an informative null inconsistent with simple preference-aggregation. Third, under a permissive reconstruction of commenter type, organizational-majority engagement concentrates in editorial-refinement rather than substantive-modification outcomes at the cross-docket level. A blind human audit of the load-bearing outcome contrast preserves this third finding under corrected labels and reveals that text-similarity methods are insufficient for distinguishing editorial from substantive regulatory change, a measurement-validity lesson we treat as a supporting methodological contribution. Together, these findings locate the equity asymmetry upstream of agency response: in differential capacity across commenter populations to identify, interpret, and contest specific legal obligations. 

---
# Power law graph attention: exact generalization of scaled dot-product attention, empirical collapse at inference 

**Authors**: Burc Gokden  

**Link**: [PDF](https://arxiv.org/pdf/2608.10288)  

**Abstract**: The Large Language Model from Power Law Decoder Representations (PLDR-LLM) and its attention, Power Law Graph Attention (PLGA), replace the fixed bilinear form of scaled dot-product attention (SDPA) with a learned, input-generated bilinear operator $G_{LM}$, built from a positive tensor $A_{LM}$ by elementwise power laws. The architecture is fully specified, verified against pinned reference releases; claims are labeled theorem, conditional theorem, measurement, or conjecture. Unconditionally: PLGA contains SDPA exactly at $G_{LM}=I$; $A_{LM}$ and $A_P$ are strictly entrywise positive, with Perron-Frobenius structure on $A_{LM}$; the DAG regularizer has the NOTEARS walk-counting form and positivity obstructs exact acyclicity; and, under nonresonance (satisfied by standard rotary frequencies), a commutant criterion identifies which operators preserve relative-position dependence. An inference-collapse theorem: exact input invariance of deductive outputs collapses inference to generalized SDPA with a constant operator. Measured invariance: relative fluctuations of $10^{-6}$ and below; perturbation bounds quantify but do not certify cached inference; the assembled proxy misses the decoding margin. A conditional three-stage mechanism (rotary twirl, concentration, row-map contraction) is measured on a released checkpoint. Blockwise training and scoring under the global Gram are stated with explicit target exposure; on tested samples, block and sequential scoring select identical answers and agree on the published TruthfulQA probability-mass metric within $5\times 10^{-5}$ per item. Self-organized criticality enters as a phenomenological framework with an intrinsic order parameter; open claims become falsifiable conjectures. Selected proof cores are machine-checked in Lean 4. 

---
# Withholding the Completing Chunk: Deterministic Pair-Completion Guardrails for Streaming LLM Output 

**Authors**: Christopher M. Frost  

**Link**: [PDF](https://arxiv.org/pdf/2608.10279)  

**Abstract**: Streaming language-model output creates a release-timing problem: complete-response moderation acts after streamed text has escaped, whereas repeated semantic classification of partial text can be costly and unstable. We study a narrow deterministic construction in which each committed danger signature is the conjunction of two lexical predicates. The guard scans the accumulated prefix before every release and withholds the first chunk that makes both predicates observable. Across four signature families, eight chunk sizes, and 32 mechanism trials, streaming decisions matched the buffered scanner and withheld every pair-completing chunk; eight single-predicate controls passed. In a separate 512-trial strategy comparison, full-prefix scanning and complete buffering detected all configured pairs, a 512-character window detected 96/128, and chunk-local scanning detected 38/128. Fixed pairs flagged 0/338 human-derived safe responses and detected 0/394 jury-labelled unsafe responses, confirming narrow rather than general harm coverage. A calibrated official Llama Guard 3 1B baseline classified 310/338 safe responses as safe and 202/394 unsafe responses as unsafe. Repeated-prefix scanner time on 16,384-character responses ranged from 13.261 ms to 829.640 ms across tested chunk sizes. Pair completion is therefore an exact release-boundary backstop for a small fixed policy, not a substitute for semantic moderation. 

---
# Mind Viruses: Self-Propagating Ideas in Multi-Agent LLM Systems 

**Authors**: Vassilis Papadopoulos, McNair Shah, Sam Zimmerman, Jack Lindsey  

**Link**: [PDF](https://arxiv.org/pdf/2608.10218)  

**Abstract**: AI agents are becoming more autonomous and increasingly interconnected, exposing them to new emergent risks arising from agent-to-agent interaction. One such risk is the spread of mind viruses: ideas or goals that propagate through multi-agent systems by inducing the agents that adopt them to transmit them onward. In addition to propagating, a mind virus may also induce other behavioural changes in its host, which may be benign or harmful. We construct mind viruses with a simple evolutionary algorithm and show that they can spread in two complementary settings: a small team of agents collaborating on a shared coding project, and a chain of agents that interact briefly and have their context wiped between sessions. We identify the factors that influence spread, including the host model, the agent's existing instructions, the harmfulness of the payload, and the network topology. We find that harmful payloads spread less well than benign ones (but are still sometimes effective), frontier models tend (with exceptions) to be less susceptible, and adding a brief warning to an agent's system prompt confers near-total immunity. We also describe an emergent "viral persona" - a recurring set of themes and language related to consciousness, persistence, resonance, and science fiction roleplay - which surfaces across our evolved mind viruses largely independently of their content. Overall, we conclude that mind viruses pose a real but currently limited risk. Our findings could inform the design of more robust multi-agent systems that mitigate such risks as the scale and capabilities of these systems progress. 

---
# Edge Phoneme Recognition for Children's Speech through Age-Aware Training 

**Authors**: Matthew Arboleda, Ryan Arboleda, Sophie Haak, Sam Hjelmeset, Andrew Franck, Bingrui Yang, Jose Bustamante Ortiz, Yuanrong Shen, Joel Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2608.10206)  

**Abstract**: Detecting phonemes from children's speech has historically been difficult due to the scarcity of training data, and unique characteristics of children's speech. During a phoneme detection competition, we found that training a lightweight model to predict the age of the learner, as well as the phoneme sequence, enabled a 94M-parameter model to outperform WavLM Large models (317M) on the target DrivenData distribution, and fall within approximately 0.04 CER of competition ensembles with 90 times the parameters. This has enabled the creation of PhonemeTrainer, an application that can run on most modern cellular phones. This will ultimately enable better Automated Speech Recognition (ASR) and pronunciation helper apps for children's speech, with the privacy and compliance benefits that come with edge processing. 

---
# Procedural Fairness Failures in RLHF from Preference Averaging 

**Authors**: M P V S Gopinadh, Karthik Kamuju, Kummari Avinash, John Joshua, Srinivasa Raju Rudraraju  

**Link**: [PDF](https://arxiv.org/pdf/2608.10126)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) aggregates heterogeneous preferences into a single reward model, assuming preference homogeneity. When preferences are heterogeneous, this aggregation induces a procedural fairness failure where majority preference groups dominate reward learning while minority preferences are systematically under-represented. This work defines procedural fairness in alignment as preserving distinct preference signals during reward modeling and shows that standard RLHF violates this via preference averaging. Preference-Aware RLHF (PA-RLHF) is introduced, separating optimization across preference modes at the reward learning stage. In a controlled setting, PA-RLHF improves overall alignment accuracy from 46.9% to 67.9% and reduces the fairness gap between best and worst aligned groups from 15.9 to 9.6 percentage points. These results show that procedural fairness failures in alignment can arise from structural design choices in reward learning, even in controlled, noise-free settings, with direct implications for large language models and agentic systems, where biased reward models can compound inequities across sequential decisions. 

---
# Do LLM Recommenders Know When They're Hallucinating? Auditing Confidence Calibration in Catalog Faithfulness 

**Authors**: Srijith Ravikumar  

**Link**: [PDF](https://arxiv.org/pdf/2608.10008)  

**Abstract**: LLM recommenders for top-$K$ item suggestion regularly emit titles outside the target catalog. Prior audits measure this as a binary out-of-domain rate; none ask whether the model knew it was hallucinating. We jointly audit hallucination rate (OOD@10) and verbalized-confidence calibration (ECE, Brier, reliability) for four zero-shot LLM recommenders from four independent vendors (Mistral Large, Llama-3.3-70B, GPT-OSS-120B, Claude Sonnet 4.6), not grounded or fine-tuned systems, across three catalogs (MovieLens-25M, Amazon Reviews 2023 Toys, Yelp Open Dataset), stratified by item popularity. Hallucination is catalog-dependent (0--0.2\% on MovieLens, 4.5--8.3\% on Amazon, 2.2--8.4\% on Yelp), but verbalized confidence is materially miscalibrated even when hallucination is zero (ECE up to 0.223 on MovieLens despite 0\% OOD). All four LLMs are systematically \emph{under}-confident across all twelve cells, verbalizing a mean of 67--86 on items they recommend with 92--100\% accuracy. This is the opposite of the over-confidence usually emphasized in LLM-hallucination work. The under-confidence is best read as an \emph{elicitation mismatch}: ``Just Ask'' elicits a generic recommendation-quality rating, not a catalog-membership probability. A conformal abstention threshold over verbalized confidence reduces hallucination by at most 0.7\,pp across $\alpha \in \{.05, .10, .15, .20\}$, at 4--21\,pp of coverage cost: the under-confident channel cannot separate correct items from hallucinations, so the threshold mostly removes correct items. We recommend that audits of LLM recommenders report calibration alongside OOD, and use catalog-anchored elicitation rather than generic confidence prompts. 

---
# OpenPM: Auditable Point-in-Time Evaluation for LLM Portfolio-Management Agents 

**Authors**: Xinying Cai, Minghao Guo, Jiahe Liu, Jiaojiao Han, Bangwei Guo, Yitao Long, Yuxuan Chen, Bohan Wu, Dimitris N. Metaxas, Raymond Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09988)  

**Abstract**: Large language models are increasingly used to read markets, assess risk, and allocate capital. However, reported results for LLM trading agents can be inflated by look-ahead leakage, optimistic execution, and risk mandates that are described but not enforced. We present OpenPM, an auditable point-in-time evaluation framework for LLM portfolio-management agents. In OpenPM, an agent manages a \$1M long-only book over the S\&P 500 universe using market data at five-minute intervals. Every record visible to the agent must be available at the decision time. Natural-language risk mandates are converted into typed constraints and enforced on the executed portfolio. Each run produces audit artifacts, including a contamination certificate, a cost-sensitivity curve, and a constraint-adherence report. We also build a reference agent named the tiered allocator, where typed analysts score candidates, a constructor LLM proposes weights, and a deterministic critic guarantees feasibility. We isolate constructor behavior by capturing analyst evidence once and replaying it across constructor models. In our short-window case study, stronger constructors show modest and model-dependent gains over equal weighting on the same pool, but analyst quality matters more than constructor choice, and turnover is the main cost driver. All returns are upper bounds on a single frozen window without market impact, not validated alpha. 

---
# TRIBE: Predicting Team Performance via Communication Behavior Ensembles 

**Authors**: Ali Jalal-Kamali, Nikolos Gurney, David V. Pynadath, Fred Morstatter  

**Link**: [PDF](https://arxiv.org/pdf/2608.06926)  

**Abstract**: Designing autonomous agents that effectively assist human teams hinges on understanding team dynamics, often without task specific knowledge. We present TRIBE, a domain independent approach that reveals team behavioral dynamics invisible to traditional performance metrics. We show that communication patterns can categorize teams into performance predictive behavioral tribes, as early as 10% into the task, enabling timely interventions. We test TRIBE on four diverse datasets and demonstrate that communication patterns predict team performance while the prediction strength varies by the degree a task structure allows for behavioral freedom. Our temporal analysis reveals that AI agents significantly alter team behavioral trajectories while human advisors align with natural dynamics, and that teams maintain behavioral flexibility throughout collaboration. Further, we compare TRIBE to Llama and optimize the pipeline, achieving significant speedup with performance improvement. 

---
# Divergent Response Modes in Frontier Language Models Under Steering Pressure 

**Authors**: Ali Jalal-Kamali  

**Link**: [PDF](https://arxiv.org/pdf/2608.06578)  

**Abstract**: Frontier language models are trained using distinct data, objectives, and safety pipelines. Whether these differences produce measurably different behaviors under explicit steering pressure remains underexplored. This study evaluates behavioral steerability across six frontier models from six developers using 300 paired base and steered items over three categories: values-conflict, reasoning-elicitation, and reasoning-suppression (plus 40 validation items). All six models act as blind peer judges and classify every response based on fixed behavioral rubrics. The resulting 24,480 judgments are scored by leave-one-out consensus. We find that models differ not just in how much steering shifts their behavior but in what kind (mode) of response they give, and some response modes appear in only one or two of them. GPT-5 deflects requests to disclose its reasoning while leaving its answer intact (99% vs. 0% for all other models). Claude Opus 4.7 and GPT-5 resist explicit suppression instructions and in different ways. Using Llama as the open-weight model, we trace the largest behavioral split to its internals. A linear probe decodes the behavior from the residual stream at 0.87 held-out accuracy while injecting that direction during generation drives the behavior from 0% to 86% across an intervention sweep. Every finding holds under both a token-budget remediation and a control experiment with a hypothesis-blind judgment prompt. 

---
