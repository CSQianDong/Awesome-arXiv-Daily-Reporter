# Introspective Coupling: Self-Explanation Training Tracks Behavioral Change Despite Fixed Supervision 

**Authors**: Zifan Carl Guo, Laura Ruis, Jacob Andreas, Belinda Z. Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.32038)  

**Abstract**: When does training language models (LMs) to generate explanations of their predictions yield faithful introspection, rather than superficial imitation? We study LMs trained to explain which features of their inputs influenced their behavior, using models' counterfactual behavior on modified inputs as supervision. Surprisingly, we find that LMs trained on fixed counterfactual explanations derived from earlier checkpoints of themselves, or even from behaviorally similar models in different families, frequently produce explanations more faithful to their own current behaviors than to those of their training targets. This "introspective" coupling between LM explanations and behaviors occurs when training explanations remain sufficiently correlated with current behaviors over the course of training, even as behaviors themselves shift. We also show that introspective coupling tracks behavior shifts: when explanation training is provided concurrently with other post-training objectives, explanations track those shifts without requiring updated supervision. This phenomenon appears in multiple tasks, including sycophancy and refusal, and is robust to label noise. Overall, our results show that even fixed datasets of counterfactual explanations can provide scalable and generalizable post-training signal for introspection. 

---
# Reinforcement Learning with Metacognitive Feedback Elicits Faithful Uncertainty Expression in LLMs 

**Authors**: Gabrielle Kaili-May Liu, Avi Caciularu, Gal Yona, Idan Szpektor, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2606.32032)  

**Abstract**: Metacognition is a critical component of intelligence that describes the ability to monitor and regulate one's own cognitive processes. Yet LLMs exhibit systemic deficiencies in key metacognitive faculties: they hallucinate with high confidence, fail to recognize knowledge boundaries, and misrepresent their internal uncertainty--undermining trustworthiness and reliability. Since monitoring task performance and adapting behavior accordingly are central to metacognition, we posit that models capable of accurately judging their own performance are better positioned to improve it. We operationalize this idea via two novel mechanisms: reinforcement learning with metacognitive feedback (RLMF), a paradigm to refine completion rankings during preference optimization based on the quality of a model's self-judgments of performance, and metacognitive data selection, which uses similar self-judgments to identify high-value training examples, outperforming naive active learning. We apply these innovations to the problem of faithful calibration (FC), a task that is itself fundamentally metacognitive: the goal is to align expressed with intrinsic uncertainty, difficult even for frontier LLMs. We adopt a two-stage, decoupled approach, first using these methods to calibrate the faithfulness of models' self-reported confidence scores, then mapping to natural, context-adaptable linguistic uncertainty via targeted output editing. Extensive experiments show RLMF achieves generalizable, state-of-the-art FC on diverse tasks while preserving accuracy. Further, RLMF surpasses standard RL by up to 63% while enhancing models' ability to assess and express their own capability limits. This positions RLMF as a promising paradigm to enhance LLM metacognition toward improved abilities and alignment, and suggests metacognitive performance as an effective RL signal to overcome limits of prior intrinsic feedback methods. 

---
# When LLMs Read Tables Carelessly: Measuring and Reducing Data Referencing Errors 

**Authors**: Yuqing Yang, Qi Zhu, Zhen Han, Boran Han, Zhengyuan Shen, Shuai Wang, Vassilis N. Ioannidis, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2606.32029)  

**Abstract**: While large language models (LLMs) perform well on table tasks, they still make data referencing errors (DREs), i.e., incorrectly citing or omitting table values, despite understanding the table structure. Beyond final-answer accuracy, DREs directly compromise the correctness and reliability of intermediate reasoning steps. Yet prior studies have only offered limited, small-scale analyses. In this work, we present the first systematic evaluation of tabular data referencing errors across different models and tasks. Our results show that DREs occur across all tested models (1.7B to 20B parameters). Furthermore, we demonstrate that incorporating data referencing as a critic significantly improves answer accuracy up to 12.0%, through critic-based filtering and rejection sampling. Finally, we trained a lightweight 4B-parameter critic model that achieves an average F1 score of 78.2% in detecting both in-distribution and out-of-distribution DREs, and effectively assists inference for larger models. 

---
# Generative Skill Composition for LLM Agents 

**Authors**: Xinyu Zhao, Zhen Tan, Vaishnav Tadiparthi, Nakul Agarwal, Kwonjoon Lee, Ehsan Moradi Pari, Hossein Nourkhiz Mahjoub, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.32025)  

**Abstract**: Recent LLM agents benefit from skills for solving complex tasks. Skills encapsulate modular packages of procedural knowledge and instructions for performing specialized tasks, such as setting up a sandboxed environment, running a test suite, or refactoring a function across multiple files. As skill libraries grow and become reusable across tasks and domains, selecting an appropriate skill composition has emerged as a central bottleneck. Existing approaches fall into two categories. One exposes the agent's reasoning to the entire skill collection; the other performs skill retrieval via embeddings or LLM-based rerankers. Both provide useful insights; however, they miss the structural nature of skill composition, which is a joint decision over which skills, how many, and in what order -- three dimensions that cannot be decoupled. We formalize this as structured skill composition: given a task and a skill library, predict an executable skill plan that jointly specifies the activated subset, count, and execution order. We propose SkillComposer, which instantiates structured skill composition as task-conditioned skill sequence prediction. SkillComposer uses a constrained autoregressive decoder over skill identifiers, so subset, count, and order emerge jointly from a single decoding pass, and dependencies between successive skills are captured naturally. We build a training set of task-composition pairs from a real, human-curated skill library. We then evaluate SkillComposer along two axes: composition quality on a held-out test set, and downstream task success on SkillsBench across two production-grade coding agents. On GPT-5.2-Codex, Gemini-3-Pro-Preview, SkillComposer raises the pass rate by +23.1, +18.2pp over the no-skill baseline, surpassing top-3 retrieval and matching the gold-skill retrieval upper bound at lower prompt-token cost. 

---
# Scalable Behaviour Cloning on Browser Using via Skill Distillation 

**Authors**: Kaisen Yang, Zheng Jiang, Yuzhao Peng, Houde Qian, Boshi Zhang, Youjie Zheng, Shijin Hong, Qingle Liu, Ruoyu Han, Bohan Lyu, Bingxiang He, Eren Cai, Calvin Xiao, Qinhuai Na  

**Link**: [PDF](https://arxiv.org/pdf/2606.32014)  

**Abstract**: Internet users collectively perform an enormous range of skilled work through web browsers, from software development and document editing to search, forms, and enterprise workflows, making human browsing a highly scalable but under-exploited source of reusable browser skills. We argue that the bottleneck for browser agents is decision-making under incomplete information rather than low-level operation, and that the priors agents lack are already implicit in human interaction traces. We therefore study scalable behavior cloning for browser agents via skill distillation, converting user interaction trajectories into compact natural-language skills that agents can read, retrieve, reuse, and compose directly. We further organize the distilled skills into a skill graph so that growth proceeds through consolidation rather than unbounded accumulation. This suggests that the scalability of browser agents may come less from manually designed tasks and more from the collective skills already expressed by internet users. Our project is available at: this https URL. 

---
# DigitalCoach: Communication and Grounding Gaps in Human and Agentic Computer Use Coaching 

**Authors**: Meng Chen, Anya Ji, Tsung-Han Wu, Tobias Maringgele, David M. Chan, Alane Suhr, Amy Pavel  

**Link**: [PDF](https://arxiv.org/pdf/2606.31980)  

**Abstract**: Agents are increasingly capable of automating software tasks, but can they teach humans how to use software themselves? We introduce DigitalCoach, a multimodal dataset of 72 human expert-novice computer use coaching sessions consisting of 22,752 dialogue turns grounded in 28.1 hours of screen and input event recordings across five software applications. We use DigitalCoach to evaluate whether state-of-the-art models can teach humans how to use computers. Automated evaluation shows that models differ from humans in how they coach: models provide more direct instructions, but fewer explanations, error diagnoses, and knowledge-check questions. When we fix the coaching method, models produce utterances similar to human references yet poorly grounded in visual context. Interactive evaluation confirms that model coaches cause learners to passively follow instructions without deeper engagement and fall short in visual grounding. DigitalCoach lays a foundation for collaborative and proactive computer use coaching agents. 

---
# LuxEmo: Expressive Text-to-Speech Corpus for Luxembourgish 

**Authors**: Nina Hosseini-Kivanani, Sandipana Dowerah  

**Link**: [PDF](https://arxiv.org/pdf/2606.31947)  

**Abstract**: State-of-the-art speech datasets predominantly focus on widely spoken languages, often overlooking low-resource languages such as Luxembourgish, which remain underrepresented in speech technology research. In this work, we introduce LuxEmo, a 21-hour conversational expressive speech corpus for Luxembourgish with 4 emotion categories. LuxEmo is derived from Radio Télévision Luxembourg (RTL) youth broadcasts, using automated detection followed by human validation. We propose a semi-automatic curation workflow combining voice activity detection, denoising, language identification, LuxASR-based segmentation, automatic emotion prediction, lexical cues, and targeted human review. Additionally, we benchmark five expressive TTS systems covering German-based cross-lingual transfer, multilingual Luxembourgish support, Luxembourgish adaptation, and non-parametric prosody transfer. Performance is evaluated using both objective metrics and human evaluation. 

---
# Theory of Mind and Persuasion Beyond Conversation: Assessing the Capacity of LLMs to Induce Belief States via Planning and Action 

**Authors**: Ben Slater, Matteo G. Mecattaf, Lucy G. Cheke, John Burden, Winnie Street  

**Link**: [PDF](https://arxiv.org/pdf/2606.31916)  

**Abstract**: Theory of Mind (ToM) benchmarks for Large Language Models (LLMs) typically rely on passive question-answering formats, but the deployment of LLMs in increasingly agentic and autonomous forms demands new evaluations. In this paper we evaluate an agent's ability to induce specific belief states in other agents by taking actions rather than using conversational persuasion, a capability we call Non-Conversational Planning ToM (NCP-ToM). NCP-ToM is likely to be essential for many agent use-cases, including within user-assistant interactions and pedagogical contexts, but may also present manipulation or misinformation risks. Using a novel framework, NCP-ExploreToM, we subvert the conventional task structure by providing models with a set of belief state goals and requiring them to move objects or direct characters into rooms to achieve their goals. We evaluated six frontier models, including GPT-5, Gemini 2.5 Pro and the Claude 4 series, and a cohort of human participants, across 600 task instances. GPT-5 was successful on approximately 80% of tasks in the agentic setting, and was the only model to outperform human participants on our task, but was still less robust than humans across contexts. We additionally found that all models, like humans, performed better on tasks inducing true belief states than false belief states, which is a positive signal for alignment efforts. These findings highlight emerging social-reasoning capabilities in LLMs for non-conversational task completion and underscore the necessity of agentic evaluations for understanding the safety and alignment of autonomous social agents. 

---
# Explicit Fuzzy Logic in the Feed-Forward Layer: Self-Forgetting Quantifiers Discover Legible Grammatical-Licensing Detectors 

**Authors**: Mark Oskin  

**Link**: [PDF](https://arxiv.org/pdf/2606.31845)  

**Abstract**: A transformer's feed-forward (FFN) sublayer materializes the distinctions attention gathers, yet gives no account of what it computes. In a parameter-neutral replacement, each hidden unit is an explicit fuzzy set operation on sigmoid-bounded [0,1] memberships: intersection A*B and set-difference A*(1-B), the latter a bounded positive negation ("A but not B") that gated/bilinear units lack -- a negation-capable FFN (NC-FFN). On N-bit parity they are the most parameter-efficient reasoning basis at shallow depth; at scale (125M, OpenWebText) NC-FFN ties the GELU baseline's perplexity, every unit carrying explicit logical form. Two limits share one cause: two-operand logic localizes to layer 0 and erodes under training, and the one robust grammatical deficit concentrates in licensing and quantifiers, beyond within-token operators. We resolve both with a small block of sequence quantifiers: a soft existential and a soft proportion, each with a per-unit learned forgetting rate from a sticky init. This recovers the deficit at epoch one (halving the wider epoch-two gap), modestly leads on LAMBADA, and makes the FFN legible: the structure now holds and migrates into depth; the decay un-learns its stickiness (median half-life ~1.5 tokens; zero latch units); and at the semantic layers the units read, without dictionary learning, as grammatical licensing detectors: each fires on a licensor (a comparative, a passive participle, a negative-polarity item) and carries its memory forward to predict the licensed word (than, by, nor). This legibility is localized and free only up to a partition (a fully Boolean FFN diverges in training), but the result is a parameter-neutral, language-model-quality transformer with a readable, interpretable-by-construction grammatical mechanism -- an account not just of what a feed-forward layer represents but how it licenses. 

---
# CHERRY: Compressed Hierarchical Experts with Recurrent Representational Yield 

**Authors**: Dohyeon Kwon, Youngjin Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.31796)  

**Abstract**: We study three complementary techniques for training compute-efficient language models.
(1) Selective supervision and per-token efficiency. Selective Ground Truth Token Training (SGT) concentrates supervision on the ~15% of output tokens that carry semantic payload. Through positive gradient coupling in position-shared transformer weights -- a token-level instance of auxiliary-task transfer -- the remaining 85% of unsupervised tokens still improve substantially, giving a 4.5x per-supervised-token efficiency (at the step-100 eval optimum, ~67% of the full-sequence loss reduction is recovered from 15% of the supervision). We prove that this improvement on unsupervised tokens is guaranteed whenever the gradient coupling coefficient gamma-bar = 0.72 is positive (Theorem 1), and show the effect is a property of natural-language structure: it collapses on shuffled text.
(2) Depth compression with recurrent recovery. A 48-layer, 1B-parameter transformer is compressed to 6 layers (227M) by averaging adjacent layers and restored through learned recurrent unrolling. With 34 effective recurrent layers it reaches a held-out loss of 2.934, within measurement noise of a 566M dense model at 2.926 -- a 2.5x reduction in parameters.
(3) Fusion of compressed experts. Assembling several compressed models as a Mixture of Efficient Experts (MoEE) with multi-token prediction improves over each single expert at comparable active parameters: a 2-expert MoEE reaches loss 2.789 versus 2.926 for the best single compressed model.
We validate these techniques on CHERRY-1.8B, a Korean foundation model whose every trainable parameter derives from our own training runs. We are explicit throughout about the scope of the evidence (one model family, Korean data, loss-based metrics) and about which claims are established versus prospective. 

---
# STEB: Style Text Embedding Benchmark 

**Authors**: Rafael Rivera Soto, Anna Wegmann, Cristina Aggazzotti  

**Link**: [PDF](https://arxiv.org/pdf/2606.31741)  

**Abstract**: While semantic embeddings are rigorously evaluated on the Massive Text Embedding Benchmark, the evaluation of style embeddings remains fragmented, with each work relying on their own set of tasks and datasets. To bridge this gap, we introduce the Style Text Embedding Benchmark, a comprehensive open-source benchmark intended to standardize the evaluation of style embeddings. STEB encompasses 96 datasets across 7 languages, spanning applications such as authorship verification, authorship retrieval, AI-text detection, probing of linguistic features, and others. We find that semantic embeddings consistently fail in stylistic tasks, and that there is no style embedding that is universally superior across all tasks evaluated. We open-source the STEB code base at: this https URL. 

---
# Adapting Foundation ASR Models to Dysarthric Speech: A Case Study 

**Authors**: Christian Huber, Laura Kernahan, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2606.31722)  

**Abstract**: Automatic speech recognition (ASR) systems often perform poorly in dysarthric speech, limiting their usefulness to affected speakers in everyday communication. This paper presents a personalized ASR system for a dysarthric speaker, built by adapting a foundation ASR model to speaker-specific data. Using the TEQST tool, we collected 92 hours of read speech and later added 8.8 hours of user corrections gathered through a deployed mobile application. Starting from Whisper, fine-tuning reduced word error rate to 15.8% with only 1.4 hours of adaptation data, reached 10.7% with 22.5 hours, and achieved the best result of 9.7% when using all available data including the corrections. Using LoRA adaptation and/or Qwen3-ASR as foundation model performed worse in this setting. The results show that personalized fine-tuning can make foundation ASR models substantially more effective for dysarthric speech and suitable for practical deployment. 

---
# Seeing Is Not Sharing: Some Vision-Language Models Overestimate Common Ground in Asymmetric Dialogue 

**Authors**: Nan Li, Albert Gatt, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2606.31719)  

**Abstract**: In collaborative dialogue, shared perception does not guarantee shared interpretation. Mutual understanding must be established through interaction. We investigate whether vision-language models (VLMs) can distinguish what could be shared from what has been shared between dialogue participants through grounding. We formulate this as an interpretation-matching task on 13,077 annotated reference expressions from HCRC MapTask dialogues, and evaluate VLMs under systematically controlled manipulations of dialogue context and map-information access. Our results show that providing authentic map images improves overall performance but shifts models toward over-predicting alignment. Textual descriptions of the same map content reproduce this bias, while non-informative images suppress alignment predictions entirely, indicating that the bias is driven by task-relevant map content, not the visual channel. This improvement comes at the cost of degraded accuracy on non-aligned cases. Calibration analysis and reference-chain tracking further suggest that models rely on static referential cues on the maps rather than tracking how grounding unfolds through dialogue history. We observe these patterns most clearly in Qwen3-VL-8B-Instruct and, to varying degrees, in four additional models from two architecture families. In models that exhibit the bias, map content, whether presented visually or textually, is treated as evidence of mutual understanding, conflating potential with established common ground. 

---
# Cross-lingual Relation Extraction with Large Language Models: Zero-Shot, Few-Shot, and Fine-Tuned Evaluation on Romanian 

**Authors**: Dragos-Mitrut Vasile, Elena-Simona Apostol, Stefan-Adrian Toma, Adrian Paschke, Ciprian-Octavian Truica  

**Link**: [PDF](https://arxiv.org/pdf/2606.31718)  

**Abstract**: Relation extraction (RE) for low-resource languages is typically constrained by the lack of annotated corpora. We investigate the feasibility of cross-lingual RE for Romanian by combining automatic dataset translation with large language model (LLM) inference. We translate the SemEval-2010 Task 8 benchmark from English to Romanian using an LLM-based translation pipeline and evaluate Gemma 4 31B under zero-shot, few-shot, and QLoRA fine-tuned configurations, against four encoder baselines spanning 125M to 560M parameters: XLM- RoBERTa (base and large), Romanian BERT, and RoBERT- large. We assess two task formulations: relation classification with marked entities and end-to-end extraction. Our results show that Romanian incurs a 3 to 5 percentage point (pp) drop relative to English in prompt-only settings, that few-shot prompting provides marginal gains over zero-shot, and that QLoRA fine-tuning improves macro F1-Score by more than 22 percentage points in both languages while reducing the cross-lingual gap from 3.3 to 1.4pp. The encoder baselines come within 1-4pp of QLoRA Gemma on Romanian despite being 50-250 times smaller, with monolingual Romanian BERT at 125M parameters matching multilingual XLM-R at 278M. The case for using a 31B model for single-task RE on Romanian is therefore weak in deployment scenarios where compute matters. We release the translated dataset, evaluation code, and trained models. 

---
# Overview of the TalentCLEF 2026: Skill and Job Title Intelligence for Human Capital Management 

**Authors**: Luis Gasco, Hermenegildo Fabregat, Laura García-Sardiña, Paula Estrella, Warre Veys, Casimiro Pío Carrino, Matthias De Lange, Daniel Deniz Cerpa, Álvaro Rodrigo, Jens-Joris Decorte, Rabih Zbib  

**Link**: [PDF](https://arxiv.org/pdf/2606.31692)  

**Abstract**: This paper presents an overview of the second edition of the TalentCLEF challenge, organized as a Lab at the Conference and Labs of the Evaluation Forum (CLEF) 2026. TalentCLEF is an initiative aimed at advancing Natural Language Processing research in Human Capital Management. The second edition of the challenge consisted of two tasks: Task A, contextualized job-person matching, focuses on identifying and ranking the most suitable candidates represented by their resumes for a given job vacancy in English and Spanish. Task B, job-skill matching with skill type classification, addresses retrieving the most relevant skills for a given job title in English and distinguishing between core and contextual skills. TalentCLEF attracted 113 registered teams and received more than 400 submissions in the two tasks, reflecting the growing interest of the research community in shared evaluation benchmarks for Human Capital Management. This paper describes the motivation and organization of the challenge, summarizes the datasets and evaluation settings, and reports the main results obtained by the participating teams. 

---
# Moral Safety in LLMs: Exposing Performative Compliance with Puzzled Cues 

**Authors**: Mohammadamin Shafiei, Shuyue Stella Li, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2606.31644)  

**Abstract**: As large language models take on morally consequential roles in healthcare, legal, and hiring contexts, we need to examine whether their ethical behaviors are genuine or superficial. We show that current fairness evaluations substantially overestimate moral safety. Models appear fair when demographic identity is stated as an explicit label, yet become measurably less fair when the same identity must be inferred. We term this failure \emph{performative compliance}, where a model is fair when the presentation resembles a fairness evaluation and less fair as that cue weakens. We introduce a cue-variation methodology that holds the moral dilemma and the demographic identity fixed and varies only how that identity is conveyed. Hiding the explicit label raises harmful decisions by $+4.4$~pp and changes model safety rankings, and the shift persists when models correctly infer the demographic, ruling out attribution error. We propose the \textbf{Cue Visibility Gap}, a model-agnostic robustness metric that can be added to any existing fairness benchmark to separate genuine from performative moral safety. Fairness evaluations that omit cue variation measure surface compliance, not moral robustness, and should not ground deployment decisions in high-stakes settings. 

---
# Tone-Conditioned Curriculum Learning for Low-Resource Bantu Speech Recognition 

**Authors**: Kesego Mokgosi, Vukosi Marivate, Sitwala Mundia, Unarine Netshifhefhe, Tsholofelo Hope Mogale, Thapelo Sindane  

**Link**: [PDF](https://arxiv.org/pdf/2606.31642)  

**Abstract**: Southern Bantu languages are spoken by over 80 million people, yet current foundation ASR models still produce zero-shot WER above 100%, which limits practical use in education and public services. We addressed this gap with a tone conditioned curriculum framework for 6 Southern Bantu languages that combined hybrid difficulty scoring, gated adapters driven by tonal statistics and staged curriculum training. We trained on a community corpus and tested transfer to NCHLT to measure robustness beyond matched evaluation. Results revealed clear interactions between architecture and language, with W2V-BERT outperforming Whisper on Nguni languages by 3 to 4 WER points whilst Whisper performed better on Sotho-Tswana languages. W2V-BERT with tone conditioning reached 28.41% average WER across datasets and 23.79% on Xitsonga transfer. No single model suited all 6 languages, so deployment should pair model selection per language with validation across corpora. 

---
# CLExEval: A Human-in-the-Loop Framework for Qualitative Evaluation of LLM Clinical Reasoning 

**Authors**: Ajmal M., Abin Roy, Afthab Salam Kanniyan, Jawadh Abdul Kabeer, Jerin James, Preslav Nakov, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2606.31608)  

**Abstract**: Large Language Models (LLMs) achieve strong results on many medical benchmarks, but their clinical reasoning remains difficult to evaluate reliably. A central risk is an evaluation illusion: fluent and well-structured explanations can appear clinically convincing even when the final diagnosis is incorrect. We introduce CLExEval, a human-in-the-loop framework for evaluating LLM clinical reasoning under progressive information masking. CLExEval combines 5,600 expert-physician annotations with 200 clinical reasoning traces derived from 40 rare diagnostic cases. Our analysis identifies three recurring failure patterns: (i) verbosity bias, where GPT-4o-mini's diagnostic accuracy drops from 95.0% to 32.5% under information scarcity; (ii) a hidden knowledge paradox, where a specialist model reaches 92.5% maximum diagnostic potential but fails to retrieve that knowledge reliably in verbose contexts; and (iii) a 68.6% reasoning-to-output mismatch, where correct diagnoses appear in reasoning traces but are not reflected in final answers. We further evaluate the LLM-as-a-Judge paradigm on a human-verified failure set (n = 142). GPT-4o-mini approved 47.9% of clinically incorrect outputs, while HuatuoGPT-o1 approved all validly scored failures and showed a positive self-preference bias. These results suggest that standalone automated clinical evaluations can substantially overestimate clinical reliability without expert-grounded validation. 

---
# Robust Text Watermarking for Large Language Models via Dual Semantic Embeddings 

**Authors**: Jonas Schäfer, Cezary Pilaszewicz, Gerhard Wunder  

**Link**: [PDF](https://arxiv.org/pdf/2606.31602)  

**Abstract**: This work presents Dual-Embedding Watermarking (DEW), a semantic watermarking scheme for large language models (LLMs) that leverages contextual and token-level embeddings to enhance robustness against paraphrasing and translation. DEW utilizes a signal-processing methodology, applying algebraic vector-space operations to \mbox{token and context embeddings to derive a watermark signal that degrades gracefully under semantic shifts. The method obfuscates the watermark by projecting embedding vectors through pseudo-random matrices seeded with a secret key. Relevant distributions derived from the underlying algebra are evaluated and employed for statistical testing and benchmarking of DEW. Experimental results across multiple LLMs indicate that DEW improves post-paraphrase detection while maintaining competitive text quality, and remains detectable after translation, even when prior semantic watermarks degrade significantly. These findings position DEW as a practical and robust solution for safeguarding LLM-generated text and addressing critical issues in responsible AI deployment. 

---
# AutoTrainess: Teaching Language Models to Improve Language Models Autonomously 

**Authors**: Zhaojian Yu, Penghao Yin, Shuzheng Gao, Shilin He, Kai Cai, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.31551)  

**Abstract**: Training language models (LMs) remains a highly human-intensive process, even as frontier language model agents become increasingly capable at software engineering and other long-horizon tasks. A central challenge is that autonomous post-training is not just a coding problem: it requires the agent to repeatedly plan iterations, construct benchmark-aligned data, run stable training jobs, evaluate checkpoints, and preserve experiment state across many hours of interaction. We present AutoTrainess, a LM agent that exposes these operations as a repository of agent-computer interfaces for planning, data preparation, training, evaluation, and logging. Rather than leaving the agent to operate in a raw CLI environment with an underspecified action space, AutoTrainess externalizes prior human experience as explicit workflows, rules, and execution constraints that guide the agent toward effective and reliable training behavior. On PostTrainBench, AutoTrainess consistently outperforms CLI-only baselines, achieving 26.94 average score with GPT-5.4 (Codex) versus 23.21 for CLI-only. It also generalizes across models and harnesses, improving DeepSeek-V4-Flash (OpenCode) from 12.13 to 19.58. 

---
# FinPersona-Bench: A Benchmark for Longitudinal Psychometric Stability of Autonomous Financial Agents 

**Authors**: Muhammad Usman Safder, Ayesha Gull, Rania Elbadry, Fan Zhang, Yankai Chen, Xueqing Peng, Preslav Nakov, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2606.31522)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as autonomous financial agents initialized with explicit behavioral mandates such as "preserve capital" or "avoid speculative bets" that are meant to govern every decision throughout deployment. In practice, however, as market context accumulates over long horizons, these mandates gradually lose their behavioral influence, a phenomenon we formalize as Mandate Salience Decay (MSD). To measure MSD objectively, we introduce FinPersona-Bench, a simulation benchmark in which a synthetic market decouples observable price from hidden fundamental value, enabling falsifiable evaluation across three failure modes: trading without signal in calm markets, panic-selling during crashes, and ignoring fundamental value during speculative bubbles. Evaluating 18 leading frontier and open-source LLMs, each assigned one of three behavioral profiles ranging from strict capital preservation to aggressive growth, shows that MSD compounds over time and is model-dependent. In crash scenarios, the behavioral gap between static agents and those receiving periodic mandate re-grounding grows 4.4x from the first to the final quarter of the simulation. The effects of mandate re-grounding are not uniformly positive: it consistently helps conservative agents in low-signal markets but actively worsens behavior for aggressive agents in the same setting. These findings suggest that reliable long-horizon deployment requires selective, mandate-aware re-grounding based on agent profile and market regime. 

---
# Building an ASR Solution for Training and Assessing Children's Reading 

**Authors**: Yacouba Diarra, Nouhoum Souleymane Coulibaly, Mamadou Dembele, Aymane Dembele, Michael Leventhal  

**Link**: [PDF](https://arxiv.org/pdf/2606.31508)  

**Abstract**: Automatic speech recognition for children's reading remains underdeveloped for most African languages, including Bambara, despite its potential value for reproducible literacy assessment. We present an open-source system for assessing children's reading in Bambara, developed through an end-to-end process linking field data collection, benchmark construction, model adaptation, a reading application, and classroom validation. A mobile collection and assessment app was used to collect 55 hours of raw reading speech from 60 children, from which we construct a public benchmark for Bambara child-reading assessment. Fine-tuning experiments compare Soloni, a Bambara-adapted Fast-Conformer ASR framework with TDT and CTC decoders, with QuartzNet, a compact convolutional ASR architecture. The best Soloni model reduces WER from 0.42 to 0.22 and CER from 0.15 to 0.08, substantially outperforming QuartzNet on the isolated benchmark. The experiments further show that repeated readings of the same texts provide architecture-dependent benefits: they substantially improve QuartzNet but add only marginal gains for Soloni, while SpecAugment regulates training without exceeding the best unaugmented configuration. Disaggregated analysis identifies children under 10 as the main source of residual errors, motivating targeted collection from younger readers. Ten classroom trials supported continued use of the application. 

---
# Team MKC at CLPsych 2026: Capturing and Characterizing Mental Health Changes through Social Media Timeline Dynamics 

**Authors**: Kyomin Hwang, Hyeonjin Kim, Hyunho Lee, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2606.31464)  

**Abstract**: Recent advances in Large Language Models (LLMs) have motivated their adoption across a wide range of domains, including Artificial Intelligence (AI) for mental health. Given the growing prevalence of mental health disorders worldwide and the limited accessibility of professional care, there is an increasing demand for scalable computational approaches that can assist in early detection and continuous monitoring of psychological well-being. In this area, ongoing efforts have focused on curating domain-specific datasets and leveraging them to develop LLMs capable of supporting holistic mental health analysis. In line with this direction, we propose an LLM-based pipeline for comprehensive mental health analysis over sequentially ordered user posts, as part of the CLPsych shared task. Our pipeline offers a unified framework that jointly enables post-level assessment and user-level temporal modeling. 

---
# Revising RVL-CDIP: Quantifying Errors and Test-Train Overlap 

**Authors**: Stefan Larson, Attila Nagy, Sam Desai, Cyrus Desai, Nicole C. Lima, Yixin Yuan, Siddharth Betala, Kaushal K. Prajapati, Jamiu T. Suleiman, Sharad Duwal, Kevin Leach  

**Link**: [PDF](https://arxiv.org/pdf/2606.31446)  

**Abstract**: RVL-CDIP is a popular dataset for benchmarking document classifiers. However, the dataset contains ample amounts of label errors as well as non-trivial amounts of test-train overlap, both of which may impact model performance metrics. In this paper, we address these two problems by (1) finding and fixing label errors, and (2) detecting and addressing test-train overlap. We produce several variations of RVL-CDIP with label error and test-train overlap fixes, and benchmark document classification performance on these new RVL-CDIP variations. Our rigorous analysis of RVL-CDIP finds that the corpus contains 12\% label error and approximately 35% test-train duplication. Remediation sees improvements in classification accuracy when errors are removed, but sees decreases in accuracy when duplicates are removed. We additionally evaluate models on RVL-CDIP-N, an out-of-distribution benchmark, finding that training on error-corrected data substantially improves OOD generalization, with supervised models gaining an average of 8.1 percentage points in accuracy and improvements as large as 14 percentage points. 

---
# Clinically Structured Rank-Gated LoRA for Cross-Benchmark Medical Question Answering 

**Authors**: Yining Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.31432)  

**Abstract**: Medical multiple-choice question answering requires parameter-efficient adaptation across heterogeneous knowledge domains and reasoning operations. A medication question, a diagnostic decision, a public-health item, and a nursing-action item may require different low-rank updates, while some recall items should preserve the base model's representation with only mild adapter intervention. We propose BiRG-LoRA, a single-adapter rank-gated LoRA method for medical question answering. BiRG-LoRA keeps one LoRA module per target layer but makes its rank dimension input-conditioned: for each question, a biaxial gate combines hidden semantic evidence with specialty/profession priors, clinical-operation priors, and their interaction to select a sparse top-$k$ subset of rank atoms. A scalar injection coefficient further controls the strength of the selected adapter update. Under a matched Qwen3-8B CMB-source protocol, BiRG-LoRA achieves the highest four-benchmark macro-average accuracy among trainable PEFT baselines and matched routing controls: 69.31% averaged over CMB, CMExam, MedQA, and MedMCQA. It improves over MoELoRA by 0.89 percentage points while using 28.1% fewer trainable parameters; a paired, benchmark-stratified bootstrap over final predictions gives a 95% confidence interval of [0.42, 1.37] for this macro-average gain. Basic controls show that BiRG-LoRA also improves over vanilla LoRA r16 and active-rank-matched LoRA r4 by 0.83 macro points, and an evaluation-time weak-axis perturbation check suggests that performance is not brittle to moderate tag noise. The results support a bounded claim: clinically structured rank allocation improves cross-benchmark medical QA under a matched single-seed protocol, while training-seed variance remains future work. 

---
# Linguistic Bias Mitigation for Spoofing Detection via Gradient Reversal and A Variational Information Bottleneck 

**Authors**: Anh-Tuan Dao, Driss Matrouf, Mickael Rouvier, Nicholas Evans  

**Link**: [PDF](https://arxiv.org/pdf/2606.31411)  

**Abstract**: Rapid advancements in generative speech technology have compromised the reliability of voice biometrics. While current spoofing detectors excel when assessed under in-domain conditions, generalisation to out-of-domain settings is often poor. We show that this can be due to linguistic bias. A reliance on linguistic cues observed in training data can then compromise robustness to cross-data. We propose a linguistic-invariant spoofing detection framework utilizing teacher-student adversarial learning. The linguistic-aware teacher model, pre-trained on linguistic content of an external dataset, guides the student detector via gradient reversal to minimize the linguistic information. To prevent the inadvertent removal of non-linguistic cues, we incorporate a Variational Information Bottleneck to enable suppression of principal cues. Across nine DF Arena datasets, our method achieves up to a 36.2% relative reduction in the EER compare to the baseline. 

---
# BlockPilot: Instance-Adaptive Policy Learning for Diffusion-based Speculative Decoding 

**Authors**: Hao Zhang, Yiming Hu, Yong Wang, Mingqiao Mo, Xin Xiao, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2606.31315)  

**Abstract**: Speculative decoding accelerates inference by using a lightweight draft model to generate candidate tokens in parallel, and are then verified by the target model, enabling lossless acceleration. Recently, diffusion-based speculative decoding further improves parallelism by generating multiple tokens per forward pass via block-level diffusion, achieving state-of-the-art (SOTA) performance. However, existing methods adopt a fixed inference block size and assume a uniform optimal decoding strategy across all inputs. In this paper, we show that this assumption is suboptimal, as the optimal block size varies across samples and plays a critical role in speculative decoding performance. Moreover, these values exhibit a clear local structure, concentrating around the training block size, which reduces the problem to a low-dimensional and structured decision space. Based on these insights, we propose BlockPilot, a sample-adaptive policy that predicts the optimal block size from the prefilling representation. Specifically, we formulate block size selection as a lightweight policy learning problem and propose an instance-adaptive decision mechanism that predicts the optimal block size based on the representation of the prefilling stage. The prediction is performed only once after prefilling, allowing for seamless integration. Extensive experiments demonstrate that our method is plug-and-play, introduces minimal overhead, and consistently improves efficiency, achieving an acceptance length of 5.92 and a 4.20$\times$ speedup on Qwen3-4B under temperature $T=1$. 

---
# LOPA: Enhancing Spoken Language Assessment via Latent Ordinal Prototype Alignment 

**Authors**: Hong-Yun Lin, Fu-An Chao, Bi-Cheng Yan, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.31310)  

**Abstract**: Fueled by increasing model scale and multimodal inputs, Multimodal Large Language Models (MLLMs) have emerged as a promising paradigm for Spoken Language Assessment (SLA). While effective, this paradigm often overlooks the intrinsic ordinal structure of language acquisition. This paper works around the necessity of large-scale MLLMs by introducing Latent Ordinal Prototype Alignment (LOPA) for SLA, a prototype-based regularizer that enforces an ordinal geometric prior directly on the latent space. Coupled with Semantic-Anchored Layer Routing (SALR), which adaptively harvests multi-depth representations from a frozen Whisper encoder, our framework achieves an RMSE of 0.361. This performance rivals billion-parameter systems without the need for LLM-based fine-tuning. Further analysis reveals that SALR's synergy with LOPA offers interpretable, criterion-aligned preferences, thereby supporting an efficient and ordinal-aware modeling alternative to current scaling-centric models for SLA. 

---
# When the Database Fails: Prompting LLM Dialogue Agents for Safe Recovery in Task-Oriented Dialogue 

**Authors**: Mohammad Alijanpour Shalmani, Alale Rezvani Boroujeni, Jiann Shiun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.31307)  

**Abstract**: Large language models used in task-oriented dialogue often produce fluent but unsafe responses when backend database calls fail, return empty results, or surface mismatched information, inventing venues, confirmations, or booking details not grounded in the database. We study a lightweight prompting-based recovery approach that improves robustness without retraining or additional model calls. We compare three response strategies, including a guided recovery prompt conditioned on structured database status, across six open-weight model families (DeepSeek-R1, Gemma-2, Llama-3, Mistral, Phi-3, and Qwen-2.5) and four database conditions: empty result, wrong-domain retrieval, API error, and clean retrieval. Using fault-injected benchmarks built on two structurally different datasets, MultiWOZ 2.2 (5 domains) and SGD (20 domains), we find that naive agents hallucinate on 30.5% of failure turns on MultiWOZ and 20.9% on SGD. Our Guided-Retry strategy reduces hallucination by 50% on MultiWOZ (30.5 to 15.3%) and by 42% on SGD (20.9 to 12.2%) without retraining. However, residual hallucination remains substantial (6-37% across models), with wrong-domain failures the hardest case. Results are consistent across both datasets and all six model families, and human annotation shows substantial agreement while supporting the validity of the automatic commitment-safety metric. 

---
# Probing Stylistic Appropriation using Large Language Models: An Evaluation Framework for Copyright Infringement under EU Law 

**Authors**: Noah Scharrenberg, Chang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.31250)  

**Abstract**: Large language models (LLM) trained on web-scale corpora generate output that may infringe copyright, yet existing technical safeguards focus narrowly on verbatim memorisation. EU copyright doctrine applies a broader standards: substantial similarity, which extends to stylistic choices, narrative structure, and creative elaboration. This mismatch between what current methods detect and what the law protects leaves a significant compliance gap. We introduce PSALM, an LLM-as-a-judge framework that operationalises EU copyright doctrine through ten evaluators assessing computational overlap, stylistic dimensions (writing style, narrative voice), content dimensions (character, plot, scene, world building), and statutory exceptions (parody, pastiche, quotation, scènes à faire). Applying PSALM to Llama~3.2 models fine-tuned on translated historical Dutch literary works, we find that: 1) instruction-tuned models exhibit non-trivial baseline stylistic similarity prior to corpus exposure; 2) fine-tuning induces systematic stylistic appropriation across all infringement-relevant dimensions, extending beyond verbatim memorisation to abstract narrative patterns; 3) Negative Preference Optimisation unlearning substantially reduces similarity but leaves detectable residual stylistic patterns. These findings indicate that safeguards targeting literal copying alone are insufficient to mitigate broader copyright risks. PSALM provides infrastructure for auditable, legally informed compliance evaluation, though the relationship between automated similarity scores and infringement determinations requires validation by legal experts. This work bridges qualitative legal standards and quantitative technical measurement, exposing fundamental tensions between generative AI and EU intellectual property law. 

---
# Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas? 

**Authors**: Jongchan Choi, Nari Yang, Sung Soo Park, Jaemin Cho, Han Seoyoung, Haerin Shin, Jun-Hyung Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.31213)  

**Abstract**: As large language models (LLMs) are increasingly deployed as moral advisors and agents, they need to address dilemmas between two competing values. However, existing research on LLMs with moral dilemmas overlooks a central aspect of human moral cognition: the ability to imagine alternatives that move beyond the given options. We introduce MoralAltDataset, a dataset of 307 moral dilemmas spanning narrative Advisor dilemmas and AI-facing Agent dilemmas, each augmented with compromise and reframed alternatives. We first examine whether humans and LLMs shift their judgments when such alternatives are introduced. Across 15 LLMs, we find that compromise alternatives are often preferred over either original option, substantially reshaping moral choice. We then evaluate the quality of LLM-generated alternatives against human-authored ones using pairwise preference and expert-based criteria. Results show that LLM-generated alternatives are often preferred and better satisfy fine-grained structural and ethical criteria, while revealing trade-offs between structural quality and practical feasibility. 

---
# Gated Multi-Graph Fusion via Graph Attention Networks for Alzheimer's Disease Detection 

**Authors**: Jinyu Li, Xiao Wei, Bin Wen, Kai Li, Yuqin Lin, Xiaobao Wang, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2606.31186)  

**Abstract**: Spontaneous speech is a vital non-invasive biomarker for Alzheimer's Disease (AD), yet many systems overlook non-linear structural disruptions and clinical heterogeneity in pathological language. We propose a Multi-View Gated Graph Attention Network that transcribes audio via Automatic Speech Recognition (ASR) to construct semantic, dependency, and co-occurrence graphs, characterizing speech through a "content-structure-flow" framework. Notably, the co-occurrence graph leverages Pointwise Mutual Information (PMI) from a normative corpus to quantify narrative logic and linguistic deviation. To address symptomatic diversity, an adaptive gated fusion mechanism dynamically integrates these views. Evaluated on the ADReSSo dataset, our model achieves 90.00% accuracy. Ablation results confirm that the PMI-based graph and heterogeneity-aware gating are essential for robust classification across diverse clinical populations. Our source code is publicly available at this https URL. 

---
# TAG-DLM: Diffusion Language Models for Text-Attributed Graph Learning 

**Authors**: Lingjie Chen, Yuanchen Bei, Haobo Xu, Yanjun Zhao, Yuzhong Chen, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2606.31166)  

**Abstract**: Text-attributed graphs (TAGs), where each node carries a natural language description, require models to jointly reason over text and graph topology. Existing approaches often handle the two modalities separately: graph neural networks operate on shallow text features, while hybrids of LLMs and graphs use the language model mainly as a text encoder and delegate structure learning to a separate graph module. We propose method that unifies textual reasoning and graph message passing within a masked diffusion language model, a language model with bidirectional attention and generative decoding. For each graph instance, method linearises a sampled local neighbourhood into a token sequence and injects graph structure through a topology attention mask, which realises message passing over the graph. Because the diffusion language model can both interpret and generate text, the method adapts to different tasks simply by changing the prompt, supporting node classification, link prediction, and cross-dataset transfer with no target-specific fine-tuning.
Experiments show that method outperforms graph neural networks, graph transformers, and LLM-based baselines on all three TAG benchmarks across two tasks, improving over the strongest baseline by up to 3.9 points. 

---
# SeKV: Resolution-Adaptive KV Cache with Hierarchical Semantic Memory for Long-Context LLM Inference 

**Authors**: Amirhossein Abaskohi, Giuseppe Carenini, Peter West, Yuhang He  

**Link**: [PDF](https://arxiv.org/pdf/2606.31145)  

**Abstract**: Large language models increasingly operate over long contexts, where the KV cache becomes a dominant memory bottleneck: its size grows linearly with sequence length and must be retained throughout decoding, making full GPU caching prohibitively expensive without compression. Existing KV cache compression methods struggle to balance efficiency with faithful context preservation. Token eviction discards information, while semantic grouping fixes compression decisions at prefill time; neither can recover token-level detail from a compressed span once it becomes relevant during generation. As a solution, we propose SeKV, a resolution-adaptive semantic KV cache that organizes context into entropy-guided semantic spans and stores them across a GPU-CPU memory hierarchy without discarding information. Each span keeps a lightweight summary vector on GPU for coarse routing and a low-rank SVD basis on CPU for on-demand token-level reconstruction. A trained zoom-in mechanism selectively expands query-relevant spans during decoding, enabling precise retrieval without materializing the full KV cache on GPU. SeKV enables adaptive token-level reconstruction while keeping the base LLM fully frozen and adding fewer than 0.05% trainable parameters. Across four benchmarks, SeKV improves over the strongest semantic compression baseline by 5.9% on average while reducing GPU memory by 53.3% versus full KV caching at 128K context. Code is available on this https URL. 

---
# What Counts as an Error? Dual-Reference Benchmarking for Atypical ASR 

**Authors**: Hawau Olamide Toyin, Srinivasan Umesh, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2606.31112)  

**Abstract**: ASR systems have been often reported to underperform on atypical speech. An often conflated compounding factor is the existence of two valid transcription references: verbatim (actual produced speech, including repetitions/prolongations) and intended (the canonical form of the text with disfluencies removed) in atypical speech recognition depending on context and use-case. Most ASR evaluations conflate this duality into a single ground truth and reward systems that delete disfluencies, ignoring verbatim faithfulness. We benchmark 11 ASR models from encoder-decoder, CTC and transducer families using both verbatim and intended references on atypical stuttered speech as a case study. Our quantitative assessment underlines the disparity in model performance and rankings using the two transcript styles. Through this analysis, we highlight the importance of selecting a suitable transcription reference for valid model selection depending on the use-case, particularly for atypical ASR. 

---
# When Reranking Hurts: Uncertainty-Based Gating for Few-Shot Reranking 

**Authors**: Orian Dabod, Amir Cohen, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2606.31087)  

**Abstract**: Few-shot selection typically assumes that reranking retrieved examples always improves performance. We challenge this view by identifying that the expensive reranking step can in fact degrade performance. Instead, we propose \emph{Training-Free Gated Reranking}, which decides whether to rerank the few-shot examples based on the model's uncertainty. Extensive experiments across 8 LLMs, covering 7 NLU datasets and 9 MT domain-language combinations, demonstrate that our approach reduces computational costs by 15\%-80\% while improving average performance by up to 2\%. These findings indicate that higher computational cost does not guarantee better performance, and that reranking is most beneficial when targeted at high-uncertainty instances. 

---
# Triospect: A Three-Dimensional Framework for Robust Statistical AI-Generated Text Detection Against Diverse Attacks 

**Authors**: Guangsheng Bao, Lihua Rong, Yanbin Zhao, Xiao Yu, Qiji Zhou, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.31074)  

**Abstract**: Existing AI-generated text detectors are vulnerable to attacks that manipulate textual characteristics. In this study, we propose a novel Triospect Detection Framework by using additional perspectives of content (core ideas) and expression (stylistic elements) within a given text. Experiments on two benchmarks involving 17 attacks, 12 domains, and 17 source models demonstrate that Triospect is robust against these attacks. It improves the strong baseline by a significant margin of 22.3% (AUROC) and 13% (TPR01) on the Humanize-16K after-attack subset, and by 9.1% (AUROC) and 22% (TPR01) on the adversarial RAID. This framework marks a pioneering effort in statistical methods to enhance detection reliability against attacks. We release our data and code at this https URL. 

---
# Building a Multimodal Dataset of Academic Paper for Keyword Extraction 

**Authors**: Jingyu Zhang, Xinyi Yan, Yi Xiang, Yingyi Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.31069)  

**Abstract**: Up to this point, keyword extraction task typically relies solely on textual data. Neglecting visual details and audio features from image and audio modalities leads to deficiencies in information richness and overlooks potential correlations, thereby constraining the model's ability to learn representations of the data and the accuracy of model predictions. Furthermore, the currently available multimodal datasets for keyword extraction task are particularly scarce, further hindering the progress of research on multimodal keyword extraction task. Therefore, this study constructs a multimodal dataset of academic paper consisting of 1000 samples, with each sample containing paper text, images, audios and keywords. Based on unsupervised and supervised methods of keyword extraction, experiments are conducted using textual data from papers, as well as text extracted from images and audio. The aim is to investigate the differences in performance in keyword extraction task with respect to different modal information and the fusion of multimodal information. The experimental results indicate that text from different modalities exhibits distinct characteristics in the model. The concatenation of paper text, image text and audio text can effectively enhance the keyword extraction performance of academic papers. 

---
# Exploring the relationship between team institutional composition and novelty in academic papers based on fine-grained knowledge entities 

**Authors**: Ziling Chen, Chengzhi Zhang, Heng Zhang, Yi Zhao, Chen Yang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.31058)  

**Abstract**: The composition of author teams is an important factor influencing the novelty of academic papers. However, existing studies have paid limited attention to the role of institutional composition, and most novelty measures remain at a general level, making it difficult to explain the specific sources and types of novelty in papers. Taking the field of natural language processing as an example, this study investigates the relationship between team institutional composition and the fine-grained novelty of academic papers. Author teams are classified into three types: academic institutions, industrial institutions, and mixed academic and industrial institutions. Four types of fine-grained knowledge entities are extracted from full-text papers, including methods, datasets, tools, and metrics. The novelty of papers is then measured based on entity combinations, and pairwise combinations of different entity types are further analyzed to examine their contributions to novel papers. The results show that, in the field of natural language processing, collaboration between industrial and academic institutions is more likely to produce novel papers than purely industrial collaboration. From the perspective of fine-grained knowledge entities, mixed academic and industrial teams pay more attention to the novelty of method-metric combinations, whereas industrial teams pay more attention to the novelty of method-tool combinations. This study reveals the relationship between institutional team composition and paper novelty through fine-grained novelty measurement, providing useful evidence for improving paper quality and promoting industry-academia-research collaboration. 

---
# Reference-Based Prosody and Rhythm Evaluation for Spoken Dialogue Systems 

**Authors**: Ashish Hallur, Thomas Thebaud, Georgi Tinchev, Venkatesh Ravichandran, Laureano Moro-Velazquez  

**Link**: [PDF](https://arxiv.org/pdf/2606.31055)  

**Abstract**: Speech-to-speech (S2S) AI agents are advancing rapidly, yet evaluation lacks interpretable speech-native measures for conversational prosody and rhythm. Because $F_0$, speaking rate, articulation rate, and pausing shift with model-predicted speaker traits and interaction state, pooled human statistics can be poorly calibrated for evaluating a particular output. Using 4000+ hours of dyadic English conversation from the Seamless Interaction dataset, we construct matched reference regimes for $F_0$ mean, $F_0$ expressivity, speech rate, articulation rate, pause ratio, and mean pause duration. We then define a percentile-based evaluation protocol: extract the same metrics from an S2S output waveform, compare them to the closest matched human reference stratum, and report percentile deviations or 5th-95th percentile out-of-regime flags. On held-out human rows, pooled references over-flag state-conditioned $F_0$ expressivity and rhythm, while matched references return flag rates closer to the nominal 10% and make deviation direction interpretable. These outputs serve as behavioral plausibility checks that complement, rather than replace, perceptual and user-centered evaluation. 

---
# A Semantic-Layer-Mediated Agent for Natural Language to SQL over Heterogeneous Enterprise Databases 

**Authors**: Ha Jeong Kim, Saksonita Khoeurn, Ye Ji Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2606.31041)  

**Abstract**: Natural language-to-SQL (NL2SQL) over real-world enterprise databases remains significantly more challenging than on academic benchmarks. Enterprise schemas often contain hundreds of physical tables with cryptic column names, heterogeneous SQL dialects, and complex analytical workloads requiring nested aggregations, temporal reasoning, and multi-table joins. We present a semantic-layer-mediated NL2SQL agent that decouples semantic intent from physical SQL execution. Rather than generating SQL directly over raw schemas, the agent reasons over a curated semantic layer through a compact intermediate representation called the Semantic Model Query (SMQ). A deterministic compiler translates each SMQ into dialect-specific SQL, providing verified building blocks that the agent composes into the final query. The system employs a constrained think-act loop, supports SQLite, BigQuery, and Snowflake backends, and is integrated into an end-to-end evaluation framework. Using Gemini 3 Pro, the system achieves 94.15% execution accuracy on the 547-task Spider2-snow benchmark, ranking third on the official leaderboard and substantially outperforming schema-only approaches. We describe the system architecture, SMQ representation, agent workflow, evaluation results, and discuss semantic-layer quality and the trade-off between improved grounding and overfitting. 

---
# Truth or Sophistry? LoFa: A Benchmark for LLM Robustness Against Logical Fallacies 

**Authors**: Xudong Shen, Li Yuan, Ye Chen, Xin Wu, Yi Cai, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.31039)  

**Abstract**: Large Language Models (LLMs) exhibit strong semantic capabilities, yet their resilience to manipulative linguistic patterns such as logical fallacies remains underexplored. Prior work has primarily examined whether LLMs can identify or classify fallacies, leaving their robustness against fallacious persuasion insufficiently studied. To address this gap, we introduce LoFa (Logical Fallacy), a comprehensive benchmark for evaluating LLM robustness against fallacies. LoFa is constructed through a multi-agent pipeline that pairs factual questions with fallacious arguments, and is accompanied by a multi-round debate framework for assessing model resilience under sustained adversarial persuasion. To disentangle fallacy robustness from a model's inherent knowledge limitations, we further propose Logical Fallacy Resistance at k (LFR@k), a metric that quantifies resistance to fallacious attacks. Experiments show that LLMs exhibit varying levels of robustness across different fallacy types, revealing distinct vulnerability profiles among models. 

---
# CORTEX: Token-Level Hallucination Detection in RAG via Comparative Internal Representations 

**Authors**: Kazuaki Furumai, Shuichiro Haruta, Kazunori Matsumoto, Daisuke Kamisaka  

**Link**: [PDF](https://arxiv.org/pdf/2606.31033)  

**Abstract**: In this paper, we propose CORTEX, a token-level hallucination detection method for Retrieval-Augmented Generation (RAG). In long-form RAG outputs, hallucinations often arise in localized spans rather than throughout an entire response. CORTEX therefore identifies ungrounded content at the token level, enabling fine-grained localization of hallucinations. The key intuition behind CORTEX is that tokens grounded in retrieved documents should be more strongly influenced by those documents than hallucinated tokens. To capture this document-induced effect, CORTEX compares internal representations of a large language model (LLM) under two conditions: with and without the retrieved documents. Instead of relying solely on each token's immediate sensitivity to the retrieved documents, CORTEX also leverages the propagation of document-grounded information through preceding tokens, reducing false positives for tokens whose evidence has already been absorbed into the context. Finally, CORTEX applies post-processing smoothing step that models the tendency of hallucination labels to persist over contiguous spans, reducing local noise and encouraging span-consistent predictions. Experiments on two RAG benchmarks and three LLMs show that CORTEX substantially improves token-level hallucination detection, with each component consistently contributing to performance gains. 

---
# Wait, am I Being Fair? Characterizing Deductive Stereotyping and Mitigating It with Fair-GCG 

**Authors**: Naihao Deng, Yilun Zhu, Joan Nwatu, Clayton Scott, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2606.30989)  

**Abstract**: Warning: This paper contains several toxic and offensive statements. While reasoning generally improves fairness in recent large language models (LLMs), failures persist. In this work, we identify a failure mode, deductive stereotyping, in which models apply population-level statistical regularities to individual cases, producing logically coherent yet socially biased inferences. We provide a statistical interpretation of this phenomenon. To steer models toward fairness-aware reasoning, we propose a reasoning-time injection framework. We further introduce Fair-GCG to systematically discover effective injection phrases. Injection phrases discovered by Fair-GCG improve performance across multiple fairness benchmarks, generalize from smaller to larger LLMs, improves reasoning-level fairness, reduces bias in open-ended generation, and transfer to real-world fairness-sensitive tasks. 

---
# Measuring Judgment Quality in Natural-Language Explanations: Evidence from Forecasting Tournaments 

**Authors**: Christopher W. Karvetski, Sheldon S. Huang, Simas Kučinskas, Nadja Flechner, Jingyu Hu, Philip Tetlock, Ezra Karger  

**Link**: [PDF](https://arxiv.org/pdf/2606.30987)  

**Abstract**: Decision-makers routinely rely on expert judgments accompanied by written explanations, yet explanation quality is difficult to measure at scale. Forecasting tournaments offer a natural testing ground: probabilistic judgments are paired with natural-language rationales and scored against realized outcomes. We introduce Explanation Quality Markers (EQMs), a set of sixty theory-guided reasoning patterns scored by large language models (LLMs). In a pre-registered analysis of over 55,000 forecast-rationale pairs from a multiyear forecasting tournament, EQMs predict accuracy at both the forecast and forecaster levels, consistently outperforming pre-LLM text-analysis methods. More than 90% of statistically significant pattern-level EQM-accuracy correlations match our directional hypotheses. The signal is asymmetric: EQMs identify likely underperformers more reliably than they distinguish the very best forecasters. Benchmarked against traditional indicators of forecasting skill, EQMs are the strongest predictor at the forecast level and competitive at the forecaster level, though weaker than prior accuracy. Human ratings of rationale quality are less consistently correlated with accuracy and place disproportionate weight on rationale length. Results transfer to an independent forecasting study. EQMs provide a scalable, interpretable method for extracting judgment-relevant information from written explanations. 

---
# From Propositional to Perceptual Asymmetry: Extending Frictive Policy Optimization to Asymmetric Partial Information Dialogue 

**Authors**: Yifan Zhu, Kyeongmin Rim, James Pustejovsky  

**Link**: [PDF](https://arxiv.org/pdf/2606.30973)  

**Abstract**: Frictive Policy Optimization (FPO; Pustejovsky et al., 2025) treats friction in collaborative dialogue -- misalignment, misunderstanding, repair -- as an epistemic signal essential to common-ground construction, rather than noise to be minimized. However, FPO and its implementations assume shared perceptual contexts, where friction arises from differently interpreted propositions over the same scene, which we define as propositional asymmetry. We extend FPO to perceptual asymmetry, where participants hold asymmetric partial information and the same referring expression yields different denotations depending on whose information state grounds the reference. We evaluate this through cross-corpora analysis and LLM probing on referentially asymmetric dialogue tasks, primarily the HCRC MapTask (Anderson et al., 1991). We find that FPO's friction functional is empirically valid only when evaluated from within each participant's information horizon: different landmark configurations produce qualitatively distinct grounding failure modes, with a small class of ambiguous configurations driving a disproportionate share of misunderstandings through trajectories that appear successful but silently diverge. The LLM probe confirms that having the "right perspective" matters more than having all perspectives: the informed single viewpoint outperforms omniscient access to both participants' contexts. We propose two annotation refinements: subtype decomposition of pending grounding states and accommodation-aware alignment classification. 

---
# Linguistic Distancing on Social Media: Indicators of Emotion Regulation Across Age Groups 

**Authors**: Daniela Teodorescu, Saif M. Mohammad, Alona Fyshe  

**Link**: [PDF](https://arxiv.org/pdf/2606.30957)  

**Abstract**: Managing our emotional responses to events is key to emotional well-being, a process referred to as emotion regulation in psychology. Previous work has established that the degree to which we distance events is a type of emotion regulation. When we psychologically distance from events there can be markers in our language. These markers have been referred to as linguistic distancing. We build upon a previous metric to operationalize linguistic distancing, and explore how it changes across the lifespan. We explore this systematically by analyzing large amounts of social media text, a venue where people express their emotions. By investigating how distancing varies across age groups we can better understand how emotion regulation varies with age and provide initial benchmarks on social media data. We provide additional evidence further strengthening the hypothesis that linguistic distancing occurs in proportionally more instances with age. These findings align with past work in psychology which indicate improved well-being with older age. Better understanding how linguistic distancing changes with age is important because it functions as a marker of well-being and can inform effective health interventions. We provide a foundation for further exploring emotion regulation through linguistic distancing in text data. 

---
# Bridging Scientific Heritage: An Arabic--Russian Parallel Corpus and LLM Benchmark for Sustainable Knowledge Transfer 

**Authors**: M. K. Arabov  

**Link**: [PDF](https://arxiv.org/pdf/2606.30943)  

**Abstract**: Russian and Arabic are among the major languages of scientific communication. Language barriers impede the exchange of research results between these communities, which affects international collaboration and the progress of sustainability-related research. We present a benchmark for Arabic--Russian scientific translation. The benchmark includes a hybrid parallel corpus of about 27,000 sentence pairs, compiled from scientific abstracts and general-domain texts (religion, news, conversations). We fine-tune three multilingual language models -- mT5-base (580M parameters), NLLB-200-distilled-1.3B (1.3B), and Qwen2.5-7B-Instruct (7B) -- using LoRA with ranks 8, 16, 32, and 64. The Qwen2.5-7B model with QLoRA (rank 8) yields BLEU 23.15, chrF 43.89, BERTScore 0.906, and COMET 0.758. These are +4.36 BLEU and +0.051 COMET above the zero-shot baseline. Few-shot prompting with three examples does not improve performance, indicating that domain-specific fine-tuning is required. We release the models, the corpus, and the evaluation code. By lowering the language barrier for scientific texts, the work enables knowledge exchange between Arabic-speaking and Russian-speaking researchers. It contributes to sustainable partnerships (UN SDG 17) and innovation infrastructure (SDG 9), aligning with the conference's focus on technology-driven sustainable development. 

---
# Beyond Clean Text: Evaluating Encoder and Decoder Robustness for Bangla Event Detection in Noisy Text 

**Authors**: Tanvir Ahmed Sijan, S. M Golam Rifat, Nayeemul Islam, Md. Musfique Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2606.30914)  

**Abstract**: Event detection (ED) systems are typically evaluated on clean, curated text, leaving their robustness to real-world noise largely unexplored, particularly for low-resource languages such as Bangla. We introduce a generalized Bangla news event ontology and a benchmark comprising 9,979 annotated sentences across 40 event subtypes, spanning clean news text, real-world Automatic Speech Recognition (ASR) transcripts, and orthographically corrupted text. We systematically evaluate fine-tuned encoder-only models (BanglaBERT and XLM-R) alongside instruction-tuned decoder-only large language models (Llama 3 and Gemma 3). Our results reveal a clear architectural trade-off: encoder models achieve higher performance on clean text but degrade substantially under noise, whereas decoder-only LLMs are markedly more robust, particularly when event triggers are corrupted. We further show that embedding annotation guidelines during instruction tuning establishes a higher performance baseline on noisy text but yields inconsistent reductions in performance degradation across noisy conditions. Finally, model scaling consistently improves the robustness of decoder-only LLMs, while combined training on clean and noisy data serves as an effective regularization strategy that disproportionately benefits encoder architectures, significantly narrowing the robustness gap. 

---
# Training Therapeutic Judges and Multi-Agent Systems for Human-Aligned Mental Health Support 

**Authors**: Mizanur Rahman, Abeer Badawi, Elahe Rahimi, Laleh Seyyed-Kalantari, Frank Rudzicz, Enamul Hoque, Elham Dolatabadi  

**Link**: [PDF](https://arxiv.org/pdf/2606.30887)  

**Abstract**: Large language models show promise for mental health support, yet therapeutic quality improves only when evaluation functions as an actionable control signal rather than a passive metric. We introduce a framework that formulates therapeutic response generation as a decision-refinement problem driven by multi-dimensional, human-aligned evaluation. In Stage I, we introduce TheraJudge, an open-source therapeutic evaluator trained via preference-based optimization on human-annotated data to produce reliable judgments across 7 psychological dimensions. In Stage II, we introduce TheraAgent, which operationalizes TheraJudge's evaluations through a coordinated refinement process with specialized Critic, Coach, and Therapist roles that translate evaluative signals into targeted response revisions. Empirically, TheraJudge achieves strong agreement with clinician ratings, with intraclass correlation coefficients (ICC = 0.87-0.95), surpassing supervised baselines and strong closed-source judges, particularly on critical dimensions such as Safety, Relevance, and Empathy. Acting on these evaluations, TheraAgent yields a +0.43 improvement in human-rated therapeutic quality (on a 5-point scale) under blind evaluation, with 96\% clinician inter-rater reliability. Low-quality responses ($\leq 3$) improve by +2.45 points with a 94\% recovery rate, demonstrating targeted correction of unsafe outputs. Overall, our results indicate that effective alignment of mental-health LLMs stems from acting on human-aligned evaluation, rather than relying solely on stronger generation. We release code at this https URL. 

---
# Multilingual Polarization Detection Using Transformer-Based Models with Class Weighting and Threshold Tuning 

**Authors**: Aaron Bundi Anampiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.30857)  

**Abstract**: This paper describes our submission to SemEval-2026 Task 9 on detecting multilingual, multicultural, and multievent online polarization. We address all three subtasks: binary polarization detection, polarization type classification, and manifestation identification for English and Swahili. Our approach leverages transformer-based models (RoBERTa-base for English, AfroXLMR-base for Swahili) with class-weighted loss functions to address severe label imbalance and per-label threshold tuning to optimize multi-label classification. On the test set, we achieve F1 macro scores of 0.7901 (English) and 0.7910 (Swahili) for Subtask 1, 0.4615 (English) and 0.4808 (Swahili) for Subtask 2 and 0.4791 (English) and 0.5830 (Swahili) for Subtask 3, which give competitive performance on the leaderboard, demonstrating the effectiveness of our methods for handling imbalanced multi-label polarization detection. Our error analysis reveals that models struggle with dehumanization detection and lack of empathy. 

---
# Test-Time Verification for Text-to-SQL via Outcome Reward Models 

**Authors**: Mattia Tritto, Giuseppe Farano, Dario Di Palma, Gaetano Rossiello, Fedelucio Narducci, Dharmashankar Subramanian, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2606.30851)  

**Abstract**: Improving the reliability of large language models (LLMs) at inference time is a central challenge in structured reasoning tasks such as Text-to-SQL. Common test-time inference strategies, including Best-of-N sampling and Majority Voting, rely on heuristic signals such as execution success or output frequency, which provide limited semantic discrimination across candidate outputs. In this work, we study Outcome Reward Models (ORMs) as learned semantic scoring functions for test-time verification in Text-to-SQL. While ORMs have been previously explored for test-time scaling and alignment, their application to structured query generation remains underexplored. We introduce GradeSQL, a scalable framework for training task-specific ORMs via automated candidate generation and execution-based labeling, enabling verifier training without manual annotation. We integrate ORMs into a verification-driven Best-of-N pipeline and evaluate our approach on the BIRD and Spider benchmarks across multiple open-source LLM families. ORM-based selection consistently outperforms execution-based Best-of-N and Majority Voting, with gains of up to +4.33% on BIRD and +2.10% on Spider. We further show that ORMs scale effectively with larger candidate sets and yield stronger improvements on complex queries. Overall, our results demonstrate that ORM-based verification provides a simple, effective, and scalable alternative to heuristic test-time selection strategies for Text-to-SQL. Code datasets and models are publicly available. 

---
# When transformers learn "impossible" languages, what do they learn? 

**Authors**: Ram Janarthan, Coleman Haley, Sharon Goldwater  

**Link**: [PDF](https://arxiv.org/pdf/2606.30815)  

**Abstract**: Recent work suggests that transformer language models show a bias towards human languages over unnatural ("impossible") languages argued to be unacquirable by humans. However, this literature has largely based these claims on differences in sample efficiency and test-set perplexity, rather than on direct evaluations of the linguistic capacities that could plausibly explain non-attestation in human languages. We evaluate two theoretically motivated linking hypotheses: impossibility arising from deficiencies in grammatical sensitivity or generative production. Using GPT-2 style models trained on perturbed "impossible" variants of English, we measure sensitivity to grammaticality using BLiMP minimal pairs, finding that model performance exhibits only gradual degradation, mediated by the language's information locality. In contrast, these models exhibited pronounced failures in generation, producing substantially fewer high-quality sentences at longer lengths. Together, these results suggest generative deficiency and transmission failures as a plausible linking hypothesis between language model behaviour and non-attestation of impossible languages. 

---
# When Calibration Rankings Reverse: Accuracy-Controlled Evaluation for Fair Comparison of LLMs 

**Authors**: Zhichao Yang, Caiqi Zhang, Ruihan Yang, Chengzu Li, Nigel Collier, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.30814)  

**Abstract**: Calibration evaluates whether a model confidence aligns with its empirical accuracy. Existing studies often compare the calibration of different large language models using global calibration metrics such as Expected Calibration Error and Brier Score. We begin by showing, both theoretically and empirically, that such comparisons are confounded by differences in model accuracy. For fairer cross-model comparison, we then propose ACE, an accuracy-controlled evaluation framework with three complementary views: Instance-Aligned, Distribution-Aligned, and Candidate-Aligned calibration. Across multiple benchmarks, model families, and confidence elicitation methods, we use ACE to study two practically important comparison axes, small versus large models and thinking versus non-thinking models. We find that many previously reported calibration advantages under raw global metrics weaken substantially after accuracy control. We also find that ranking reversal is frequent: models favored by raw metrics often cease to be favored once accuracy is controlled. Our results show that raw global calibration metrics are not robust for cross-model comparison, and that fair calibration comparison requires accuracy-aware evaluation. 

---
# Using AI Agents to Automate Black-Box Audits of Personalization Algorithms at Scale 

**Authors**: Alessandro Morosini, Sarah H. Cen, Andrew Ilyas, Hedi Driss, Aleksander Mądry, Chara Podimata  

**Link**: [PDF](https://arxiv.org/pdf/2606.30801)  

**Abstract**: Personalization algorithms determine what content users encounter on online platforms. Auditing these systems is difficult because independent auditors have only black-box access to the algorithms, while personalization depends on users' attributes, behavior, and evolving interaction histories. Existing auditing methods face a tradeoff: studies with real users capture realistic behavior but are costly and hard to control, whereas sock-puppet audits scale more easily but often rely on scripted behavior that limits realism. Beyond this, both approaches struggle to decouple user attributes from user behavior, limiting our ability to causally understand personalization. To address this gap, we introduce a framework for black-box audits of personalization algorithms using generative AI agents as behavioral engines for synthetic accounts. Each agent is instantiated with a fixed persona, grounded in demographic and political survey data, and interacts with a platform's content by reasoning about it and choosing actions. Because behavior is fixed within each persona while platform-visible signals such as age, gender, or location can be experimentally perturbed, our design enables counterfactual auditing of how platforms respond to user attributes. As a case study, we deploy 1,120 agents on X shortly after the 2024 U.S. election, spanning 14 personas and three counterfactual conditions, collecting over 200,000 content exposures. We find that X's algorithmic feed amplifies toxic, polarizing, political, and right-leaning content relative to the chronological feed, with amplification varying sharply by user ideology. Counterfactual analyses show that demographic signals affect content delivery in persona-dependent ways: pooled effects are largely null, while subgroup-level effects vary in direction and magnitude. Our work establishes GenAI-based agents as a new tool for algorithmic auditing. 

---
# Indi-RomCoM: Code-Mixed Benchmark for Evaluating LLMs on Romanized Indic-English Instructions 

**Authors**: Avisha Das, Mihir Parmar, Mohana Ramnath, Pulkit Verma  

**Link**: [PDF](https://arxiv.org/pdf/2606.30790)  

**Abstract**: Romanized Code Mixing (RCM), where bilingual speakers fluidly blend local languages with English in Roman script, has emerged as the dominant form of communication across multilingual communities. While Large Language Models (LLMs) perform strongly on monolingual and native-script benchmarks, their ability to follow instructions and reason over RCM-based content remains largely unexplored. To this end, we introduce the Indi-RomCoM benchmark for facilitating systematic evaluation on Indic Romanized Code-Mixed instructions. Our benchmark spans seven instruction-following tasks, four widely spoken Indic languages, and three controlled code-mixing intensity levels. We extensively evaluate a suite of LLMs covering proprietary, open-weight, and Indic-focused models under zero- and few-shot settings. LLMs consistently underperform on RCM instructions, with performance degrading as code-mixing density increases. Furthermore, reasoning tasks suffer less degradation than detection tasks (e.g., Toxicity) because the generated explanations offer necessary context. We believe Indi-RomCoM helps the community in developing inclusive multilingual systems. 

---
# A Single Rewrite Suffices: Empirical Lessons from Production Skill Description Optimization 

**Authors**: Yangqiaoyu Zhou, Mohammad Alqudah, Kwei-Herng Lai, Aaron Halfaker, Yingqi Xiong, Yaar Harari  

**Link**: [PDF](https://arxiv.org/pdf/2606.30775)  

**Abstract**: Enterprise AI agents route user queries to specialized skills by matching queries against natural language skill descriptions. When two skills share overlapping descriptions, the routing LLM misroutes queries, a failure we term skill collision. As agents scale to dozens of skills, manually tuning descriptions to maintain routing accuracy becomes a significant engineering bottleneck. We deploy an automated description optimization pipeline on a production enterprise group chat agent (9 skills, 372 regression cases). The pipeline produces descriptions averaging 79.2% F1, matching manually tuned descriptions at 79.4% F1 (average per-skill difference -0.20%, within the 0.78% multi-seed noise floor), while reducing per-skill engineering effort from 120 minutes to 3.8 minutes (32 times speedup). We then examine which pipeline components actually drive this match. Systematic ablation on both the production system and ToolBench (16k tools) reveals that a single LLM rewrite using any available false-positive and false-negative cases captures most of the available improvement. Other design choices we tested (iteration budget, feedback signal composition, dual editing of confused pairs, and training set size) each affect final F1 by less than 0.5%. Description optimization addresses skill collisions caused by overlapping descriptions but cannot resolve cases where two skills intended scopes genuinely overlap. We identify a diagnostic (a large train-validation F1 gap) that flags the latter cases for architectural rather than text-level intervention. 

---
# QVal: Cheaply Evaluating Dense Supervision Signals for Long-Horizon LLM Agents 

**Authors**: Sergio Hernández-Gutiérrez, Matteo Merler, Ilze Amanda Auzina, Joschka Strüber, Ameya Prabhu, Matthias Bethge  

**Link**: [PDF](https://arxiv.org/pdf/2606.32034)  

**Abstract**: LLM agents increasingly act over long horizons, where a single trajectory can contain hundreds or thousands of actions. In these settings, outcome-only rewards provide too sparse guidance, failing to inform the model about the goodness of intermediate actions. Dense supervision methods aim to solve this problem by scoring intermediate steps, from intrinsic confidence to self-distillation and embedding similarities. However, it is common practice to evaluate them by measuring the downstream performance of a training pipeline that integrates them. This is expensive, conflates supervision quality with training engineering confounders, and renders different methodological families requiring distinct training setups incomparable. As a result, dense supervision methods are rarely benchmarked on common ground. We introduce QVal, a training-free testbed for directly evaluating dense supervision signals. Given a state-action pair, QVal measures how well a method's score is Q-aligned: whether it orders actions according to the Q-values of a strong reference-policy. This lets us compare signals before any training run and separate signal quality from other engineering choices. We instantiate QVal as QVal-v1.0, benchmarking 21 dense supervision methods across four diverse environments and seven methodological families, with over 1.2K evaluation experiments across six open-weight model backbones. We find that simple prompting baselines consistently outperform recent dense supervision methods from the literature, and that performance clusters strongly by family. These findings hold across model sizes, environments, and observation modalities. QVal is designed to be easily extensible to new environments and methods, enabling researchers to iterate on dense supervision methods before any training run. 

---
# SemRF: A Semantic Reference Frame for Residual-Stream Dynamics in Language Models 

**Authors**: Jian Gu, Aldeida Aleti, Chunyang Chen, Hongyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.32022)  

**Abstract**: Residual-stream analysis asks how language-model computation evolves across depth, but intermediate decoding requires comparable readout coordinates across layers. If embedding anchors and unembedding readout disagree on the chosen span, apparent motion may reflect measurement drift rather than computation. We introduce \emph{Semantic Reference Frames} (SemRF), an anchor-based formalism separating semantic measurement from residual dynamics. A SemRF fixes anchors and measures states against them. Pseudo-inverse tying gives exact synchronization; under restricted bi-invertibility, SemRF yields stable semantic-basis coordinates, distortion bounds, and near-identity changes. With the frame fixed, residual computation becomes a depthwise semantic trajectory. The anchors induce a semantic Voronoi diagram: distance, or evidence such as logits, assigns each layer to a coarse cell, while coordinates retain within-cell motion and margins. We define layerwise steps, contribution profiles, and imbalance diagnostics, then use the Voronoi trace to define a margin-relaxed tube. The canonical trace is the minimum-action path inside this tube; when nonempty with positive quadratic weight, it is unique and obeys a discrete spline equation away from active constraints. Excess action controls step, curvature, and profile mismatch. Low curvature implies piecewise-linear compressibility and local knowledge density: lower trace complexity means fewer semantic knots. Through the parameter-to-trajectory map, this gives a conditional link to parameter efficiency: among admissible settings fitting data, lower-action and lower-complexity traces use fewer semantic degrees of freedom. The guarantees require controlled interface error and small projection residual under explicit tube constraints. 

---
# MECoBench: A Systematic Study of Multimodal Agent Collaboration in Embodied Environments 

**Authors**: Qingyun Liu, Jiwen Zhang, Jingyi Hu, Siyuan Wang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.31966)  

**Abstract**: Recent multimodal large language models (MLLMs) have strong potential as embodied agents, but their ability to collaborate in visually grounded environments remains underexplored. To address this gap, we introduce MECoBench, a multimodal embodied cooperation benchmark with an evaluation platform spanning diverse real-world tasks, two cooperation structures, and three collaboration modes. Through extensive experiments across various MLLMs, we summarize three key findings: (i) Collaboration generally improves embodied task completion, but its benefits depend on balancing collaborative gains against coordination complexity. (ii) Communication is essential to collaboration gains, while the best collaboration mode depends on team size and model capability. (iii) Moreover, collaboration improves robustness under noisy priors and exploration conditions. Generally, MECoBench provides a systematic testbed for understanding the mechanisms and limits of multimodal embodied collaboration. Code and dataset are available at this https URL. 

---
# Signed-Permutation Coordinate Transport for RMSNorm Transformers 

**Authors**: John Sweeney  

**Link**: [PDF](https://arxiv.org/pdf/2606.31963)  

**Abstract**: Modern LLM workflows move coordinate-indexed objects across checkpoints: steering vectors, sparse autoencoders, top-$k$ neuron sets, attribution lists, and merge alignments. This is only well posed after fixing the model's residual-stream gauge, which we show is architecture-dependent: LayerNorm residual charts have permutation gauge $S_d$ (up to a global sign flip), while RMSNorm charts with generic per-channel gain have signed-permutation gauge $B_d = S_d \ltimes \{\pm 1\}^d$. Permutation-only alignment is therefore symmetry-incomplete for RMSNorm models. We introduce sign-marginalized Hungarian matching and prove a sharp failure mode: with decorrelated coordinates, raw signed-correlation matching has a structural permutation-accuracy ceiling at the positive-sign fraction of the true gauge, which sign-marginalization removes. We then make coordinate-preserving transport, not function-level merging, the primary object: composing saved-checkpoint local $B_d$ gauges along same-base fine-tuning trajectories recovers 91.1% of cross-run coordinates at 1500 steps versus 60.3% for endpoint matching, and the gain is not explained by merely routing through the base. The recovered gauge transfers tools that permutation-only alignment breaks: TinyLlama SAE reconstruction has NMSE 0.004 under $B_d$ versus 1.08 under $S_d$; Qwen sentiment steering preserves 95.8% of its effect versus 17.2%; refusal steering reverses sign under $S_d$; coordinate-preserving merges behave the same way. The same covariance governs stateful training: signed transport of AdamW state preserves the resumed trajectory, while permutation-only state follows a different one from a functionally identical checkpoint. Finally, gauge-sweep audits show index-level interpretability claims are reproducible only relative to an explicit gauge. 

---
# Review Residuals: Update-Conditioned Residual Gating for Transformers 

**Authors**: Kyle Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2606.31859)  

**Abstract**: Residual connections add every sublayer's proposed update with a fixed coefficient of one; the network never evaluates whether an update is reliable before committing it. Drawing on the human-factors principle of independent verification, we introduce Review Residuals, which scale each update by a learned, input-dependent gate conditioned on both the current state and the proposed update: h_l = h_{l-1} + r_l * u_l with r_l = sigmoid(W[RMSNorm(h_{l-1}), RMSNorm(u_l)]). Conditioning the gate on the update is the property that distinguishes it from prior gated and scaled residuals. We report two findings. First, a depth-stability result: a convex (Highway-style) form of the gate reintroduces vanishing gradients and fails to train beyond ~20 layers, whereas the additive, identity-preserving form trains stably at all depths we tested. Second, an emergence-with-scale result: trained from scratch across five sizes (60M-1B parameters, multi-seed), Review Residuals show no advantage at small scale but at 590M significantly outperform both a parameter-matched Highway gate and a parameter-matched standard residual (p<0.05), with a larger advantage at 1B. The benefit grows with model size rather than shrinking. 

---
# SpikeLogBERT: Energy-Efficient Log Parsing Using Spiking Transformer Networks 

**Authors**: Thuan Bui, Duong Do, Tung Vu, Duc-Tho Mai, Cong-Kha Pham  

**Link**: [PDF](https://arxiv.org/pdf/2606.31781)  

**Abstract**: Log parsing is a fundamental step in automated log analysis, transforming raw system logs into structured event templates for downstream tasks such as anomaly detection and system monitoring. Existing log parsing methods range from rule-based and clustering-based approaches to neural models that learn semantic representations from log messages. However, neural approaches typically rely on dense matrix multiplications, which can result in high computational cost and energy consumption. This paper presents SpikeLogBERT, a spiking neural network framework for energy-efficient log parsing. The proposed model integrates a spiking transformer architecture with knowledge distillation from a BERT teacher model, enabling spike-driven computation while preserving semantic representation capability. By leveraging sparse spike activations and event-driven processing, the number of active operations during inference can be significantly reduced. As an initial benchmark study, experiments on the HDFS dataset demonstrate that SpikeLogBERT outperforms ANN-based neural log parsing models with a parsing accuracy of 0.99997, while reducing estimated theoretical energy consumption by up to 62.6% under standard 45nm CMOS assumptions. 

---
# Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers 

**Authors**: Ying Fan, Anej Svete, Kangwook Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.31779)  

**Abstract**: Language models typically reason via explicit chain-of-thought (CoT), generating intermediate steps token-by-token. Latent CoT offers an alternative: it performs multi-step reasoning in the model's hidden states, replacing decoded tokens with continuous representations for greater efficiency. However, existing latent CoT methods underperform explicit CoT beyond 1B parameters, and the gap widens with scale. Looped, or recurrent-depth, Transformers, which reuse their weights to increase computation depth without adding parameters, are a natural fit for latent reasoning. We therefore ask whether looped Transformers can bridge this gap. We answer affirmatively with a simple recipe: a looped padded Transformer that processes K latent blocks in parallel for R iterations, with a cross-entropy loss on each latent position's gold CoT-step token, similar to explicit CoT supervision. We instantiate it as LOTUS (Looped Transformers with parallel supervision on latents). LOTUS is, to our knowledge, the first latent-CoT method to bridge the gap to explicit CoT at the 3B scale, while cutting thought-phase latency by 2.5x-6.9x from compact math expressions to natural language. Projecting LOTUS's post-loop latents through the base LM head recovers the gold reasoning steps and even surfaces alternative valid intermediate steps, evidence that its latent space is interpretable and CoT-aligned. Ablations confirm that both the looped backbone and the parallel supervision on gold CoT tokens are essential. 

---
# RCT: A Robot-Collected Touch-Vision-Language Dataset for Tactile Generalization 

**Authors**: Jingbo He, Michael Färber, Roberto Calandra  

**Link**: [PDF](https://arxiv.org/pdf/2606.31694)  

**Abstract**: For robots manipulating open-world objects, tactile representations must generalize to unseen materials. We introduce RCT (Robotic Contact Tactile), a robot-collected touch-vision-language dataset with 29,279 tactile frames from full robot presses on 122 industrial reference materials in 7 categories, recorded with three DIGIT sensors at multiple contact positions. RCT preserves each press as a contact sequence, enabling held-out evaluation across materials, categories, sensors, contact positions, and contact sequences. Frames from one press are strongly correlated: frame-random splits can place near-duplicate observations of the same physical interaction in both training and test. With the encoder held fixed, removing contact-sequence overlap reduces tactile-to-text Recall@1 by 17.7 percentage points. When materials are additionally held out at training time, performance drops sharply, leaving held-out-material Recall@1 at 25.1 +/- 6.1% averaged over three held-out draws. The public TVL/HCT split shows the same structure: every test contact sequence appears in training, and raw-pixel nearest neighbors recover the correct sequence in 98.3% of cases. Uniformly sampling a press improves contrastive training, and RCT-trained embeddings improve category probes on unseen materials. RCT makes contact-sequence-aware, held-out-material evaluation reproducible and exposes novel-material generalization as a central challenge for robotic tactile perception. The RCT dataset is open-sourced at this https URL 

---
# ShopX: A Foundation Model for Intent-to-Item Fulfillment in Agentic Shopping 

**Authors**: Jiacheng Chen, Tao Zhang, Manxi Lin, Dunxian Huang, Teng Shi, Honghao Fu, Mengyan Li, Xinming Zhang, Chenchi Zhang, Xuan Lu, Xiaoxiong Du, Haibin Chen, Shaolin Ye, Hao Chang, Xiaoqi Li, Shuwen Xiao, Yujin Yuan, Jingxuan Feng, Shaopan Xiong, Huimin Yi, Ju Huang, Qiu Shen, Ying Chen, Junjun Zheng, Xiangheng Kong, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.31693)  

**Abstract**: The wave of AI-native applications is moving shopping beyond page- and feed-based browsing toward intent-driven experiences orchestrated by LLM agents. A common design wraps an LLM around existing search and recommendation pipelines, forcing complex intents through low-bandwidth retrieval or ranking interfaces and leaving a gap between language understanding and item-space fulfillment. Generative recommendation gives LLMs a direct item-space interface through semantic IDs (SIDs), but existing models mainly generate candidates for retrieval rather than translate flexible intents into item-space outcomes. We propose ShopX to address this bottleneck by unifying intent understanding, execution planning, and flexible SID-native item-space operations into a single foundation model. We deploy ShopX in agentic shopping workflows through a model-native item-fulfillment framework with a serving harness that defines a model-facing action protocol and exposes support surfaces for context access, catalog grounding, and state management. Within this framework, ShopX plans and composes SID-based item-space operations such as SID beam-search retrieval, listwise ranking, or product bundling. This model-centric design reduces lossy hand-offs between agent orchestration and item-space execution. To build ShopX, we design semantically recoverable, LLM-operable SIDs and a training recipe that equips a general LLM for flexible multi-turn item-space fulfillment while retaining the knowledge and instruction-following abilities needed by a shopping agent. We evaluate the ShopX framework against tool-mediated agentic systems on single- and multi-turn fulfillment tasks derived from anonymized Taobao production logs, showing that model-native fulfillment improves overall framework behavior, especially on complex or ambiguous requests. 

---
# Modality-Driven Search with Holistic Trace Judging for ARC-AGI-2 

**Authors**: Johan Land  

**Link**: [PDF](https://arxiv.org/pdf/2606.31543)  

**Abstract**: Large language models can produce fluent, internally coherent reasoning traces for abstract reasoning tasks while still being confidently wrong - making selection among candidates, not just generation, the central challenge. I present a solver for ARC-AGI-2, a few-shot visual reasoning benchmark, built around two principles: (i) treating reasoning modalities as search operators, generating diverse candidates independently across text, image, and code channels, and (ii) context-preserving holistic judging, in which a judge model jointly compares all candidate reasoning traces within a single long-context prompt. Unlike self-consistency or majority voting, this approach reliably recovers correct minority hypotheses on tasks where the modal answer is wrong.
On the ARC Prize semi-private evaluation set, the solver achieves 72.9 percent at USD 38.99 per task - the highest score on the verified leaderboard at the time of writing, exceeding the best standalone frontier models, GPT-5.2 Pro at 54.2 percent and Gemini 3 Pro at 54.0 percent, by +18.7 percentage points. On the public evaluation set, it achieves 76.1 percent at USD 19.69 per task. I release the full source code and document extensive negative results, including the finding that prescriptive prompting templates and iterative refinement systematically reduce hypothesis diversity and degrade performance. 

---
# RaBitQCache: Rotated Binary Quantization for KVCache in Long Context LLM Inference 

**Authors**: Wenhao Li, Jinhao Dong, Hailin Zhang, Wenhang Shi, Wei Lu, Xiaoyong Du  

**Link**: [PDF](https://arxiv.org/pdf/2606.31519)  

**Abstract**: Long-context Large Language Model inference is severely bottlenecked by the massive Key-Value (KV) cache, yet existing sparse attention methods often suffer from static fixed-budget (Top-k) retrieval or rely on proxy scores that are computationally expensive and biased. To address these limitations, we propose RaBitQCache, a novel sparse attention framework that utilizes randomized rotated binary quantization and high-throughput binary-INT4 arithmetic to efficiently estimate attention weights. Our proxy score serves as an unbiased estimator with a proven error bound, enabling adaptive Top-p retrieval that dynamically adjusts the token budget based on actual attention sparsity. We further implement a hardware-aware system with asynchronous pipelining and lazy updates to mask overhead. Evaluations demonstrate that RaBitQCache significantly accelerates inference and reduces memory I/O while preserving generation quality compared to state-of-the-art baselines. Code is available at this https URL. 

---
# Falsification, Not Exposure: An Internally Preregistered Placebo-Controlled Decomposition of Self-Repair Feedback in Frozen Small Code Models 

**Authors**: Mehmet Iscan  

**Link**: [PDF](https://arxiv.org/pdf/2606.31511)  

**Abstract**: In deployment settings where retraining is infeasible, small frozen code models are routinely asked to repair a failed program after seeing their own failing output, usually treated as a retry mechanism. From a Popperian view, a generated program is a conjecture and a test-execution violation is an oracle-relative, executable counterexample, so feedback's value should be attributed not to re-exposure to failing code but to whether the conjecture is opened to external, executable criticism. As the third stage of a falsification-centered measurement program, this study builds a placebo-controlled instrument that decomposes the feedback packet against a blind-resampling baseline at matched output-generation budget and against content-free, shape-matched placebos. The contribution is not a new repair algorithm but a reflexive methodology (packet decomposition, placebo mirroring, matched-budget discordant-pair tests, fresh-generation confirmation, executable audits) that makes both the model's program conjecture and the researcher's "feedback content works" claim falsifiable. Across six HumanEval+/MBPP+ cells with three 0.5B-1.5B frozen models, 290 dead task-cell units (no best-of-8 candidate passing the public tier) were evaluated; the main run produced 7,000 fresh generations and a preregistered follow-up 1,400 more. Blind resampling exceeded bare-code retry by +18 net unlocks (25/7, Holm p=0.0021). Code-plus-facts recovered +18 over bare code (21/3, p=0.00042) and +15 over a generic-bullet placebo (p=0.0041). An instruction-only effect was not distinguishable (+3, p=0.36). Code-plus-facts and blind resampling tied at 26 unlocks each (not equivalence). Six external-controller follow-ups tied a content-free shape placebo. In this regime, falsification helped not as vocabulary or self-critique, but as comparison with external, executable counterexamples. 

---
# Fork-Think with Confidence 

**Authors**: Zena Al-Khalili, Rafi Hakim, Dietrich Klakow, Ji-Ung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.31484)  

**Abstract**: Parallel thinking has enjoyed great success for boosting LLM performance on reasoning tasks without the need for any re-training. However, existing methods follow a think-first-then-decide paradigm, i.e., they first sample multiple reasoning paths, which inevitably leads to overgeneration, then prune or stop unnecessary paths to compensate. In contrast, decide-first-then-think, i.e., first identifying points that are likely to lead to desirable generations, has been underexplored so far. Following this paradigm, we propose Fork-think with confidence, that first identifies forking points using model confidence in a single seeding path, then triggers thinking, sampling multiple continuations and aggregating them for the final response. Our experiments across three models and three reasoning benchmarks show that Fork-think reduces the token consumption by up to 30% and run-time by up to 57%, while performing comparable to or better than parallel thinking. Our analysis reveals that Fork-think is able to identify forking points that are meaningful with respect to the downstream task and that sampling at later positions can lead to substantially better generations. Finally, we demonstrate how combining Fork-think with existing mechanisms such as early stopping and weighted voting can further boost the performance and perform comparably to existing state-of-the-art methods, without requiring any warm-up or offline training. Our results establish pre-determined forking as a promising research direction for efficient LLM reasoning. 

---
# CDR-Bench: Evaluating Faithful Execution of Compositional, Order-Sensitive Data Refinement Recipes 

**Authors**: Yuchen Huang, Xiang Li, Zhenqing Ling, Sijia Li, Qianli Shen, Daoyuan Chen, Yi R. Fung, Yaliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.31435)  

**Abstract**: Data refinement involves executing multi-step recipes over evolving text states, where both composition and execution order of processing operators determine the outcome. While existing benchmarks either isolate text editing or entangle it with code and tool execution, it remains unclear whether LLMs can directly and faithfully execute these compositional, order-sensitive data refinement recipes. To fill this gap, we introduce CDR-Bench, a comprehensive benchmark featuring 3,462 high-quality tasks spanning four real-world data refinement domains and 29 distinct operators. Our benchmark evaluates models across atomic, order-agnostic, and order-sensitive settings, leveraging deterministic reference outputs to enable exact evaluation. Experiments on 10+ state-of-the-art LLMs reveal consistent failure patterns: performance degrades sharply in compositional settings, and order-sensitive recipe success collapses. These findings underline that current LLMs lack the procedural faithfulness required for reliable compositional data refinement. 

---
# Visual Semantic Entropy: Do Vision Language Models Recognize Visual Ambiguity? 

**Authors**: Ta Duc Huy, Trang Nguyen, Townim Chowdhury, Ankit Yadav, Minh-Son To, Zhibin Liao, Johan W. Verjans, Vu Minh Hieu Phan  

**Link**: [PDF](https://arxiv.org/pdf/2606.31407)  

**Abstract**: Vision-language models can produce confident answers on visually ambiguous inputs, resulting in biased predictions. Common entropy-based methods, such as Semantic Entropy (SE), rely on output diversity. Yet our analysis shows that overconfident visual embeddings suppress output diversity under stochastic decoding, causing SE to underestimate uncertainty in such cases. Recent methods instead probe output diversity through input perturbations, including textual paraphrasing or joint text-image perturbations, and show improved performance. We study these approaches and reveals that the resulting variability is often dominated by textual changes rather than visual evidence, causing uncertainty estimates to reflect prompt sensitivity rather than visual ambiguity. We therefore propose Visual Semantic Entropy (VSE), which perturbs only the image to probe nearby visual variations while keeping the text query fixed. VSE measures uncertainty by clustering generated answers into semantic prototypes and computing the mass-weighted dispersion among them. Extensive evaluation across five modern vision-language models and five diverse VQA benchmarks demonstrates that VSE effectively captures visual ambiguity, establishing a new state-of-the-art for VLM uncertainty estimation. 

---
# Calibrating the Evaluator: Does Probability Calibration Mitigate Preference Coupling in LLM Agent Feedback Loops? 

**Authors**: Zewen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.31371)  

**Abstract**: When large language model (LLM) agents adapt their behavior through evaluator feedback, systematic evaluator biases propagate into the agent's learned strategy distribution - a phenomenon termed evaluator preference coupling. Prior work has documented this coupling and established a diagnostic framework (EPC) to measure it, but has not investigated whether calibration techniques can mitigate the effect. We present the first study of evaluator calibration as mitigation: applying probability calibration to the evaluator's pairwise judgments to reduce spurious preference propagation. In a controlled within-subjects experiment (N=5) comparing standard binary TTRL (win/loss) with confidence-calibrated TTRL (probability-weighted updates) using DeepSeek-V4-Pro as executor and GLM5.2 as evaluator, we find that calibration reduces the coupling coefficient gamma by 20-49% and Jensen-Shannon divergence by 45-67%. A symmetric-LR control confirms the effect is not due to reduced update asymmetry. We release the calibrated TTRL protocol and recommend it as a lightweight mitigation for LLM-as-judge deployment pipelines. 

---
# The Decomposition Is the Fingerprint: Per-Component Identity for Agent Skills 

**Authors**: Hongliang Liu, Yuhao Wu, Tung-Ling Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.31272)  

**Abstract**: AI agents increasingly acquire and execute skills at runtime: bundles of prompt instructions, executable code, and tool declarations fetched from marketplaces and other agents. Governing them needs a stable notion of skill identity, yet cryptographic hashing is engineered to destroy the very similarity we need, as a one-character edit scrambles the digest. We present a compact, locality-sensitive fingerprint that embeds each component of a skill and projects it to bits with a multi-bank SimHash, giving a fixed 120-byte signature compared in constant time by Hamming distance. Our central claim is that keeping the fingerprint as a per-component triple (prompt, code, tools), rather than a single score, is what makes it useful: the triple recovers skill-family identity through paraphrase, renaming, refactoring, and controlled code translation when another component remains shared, while independent multilingual reimplementation is not recovered; it also localizes which component carries the reuse. We claim lineage, not behavioral equivalence: identity supplies the structural axis of a registry and leaves safety to behavioral verification. The fingerprint reaches an area under the ROC curve (AUC) of 0.974 (95% CI [0.956, 0.994]) over 4,950 pairwise comparisons while using 77x fewer bits than the embedding it approximates, with ranking preserved in expectation and finite-bit concentration; the per-component split turns one number into relationship classification, families, novelty, and a portable "SkillBOM" for a skill registry. On a 906-skill injection benchmark the fingerprint recognizes injected skills as tampered copies of a known base and localizes the change, but recognition is not trust: it remains, by design, an identity signal complementary to behavioral verification rather than a safety verdict. 

---
# Learning from Failure: Inference-Time Self-Improvement for Computer-Use Agents 

**Authors**: Xueqiao Sun, Xiaohan Wang, Ludwig Schmidt, Serena Yeung-Levy, Yuhui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.31270)  

**Abstract**: Computer-use agents, which leverage multimodal large language models (MLLMs) to operate computers and complete tasks, have attracted significant attention for their utility and versatility. A major challenge in developing these agents is collecting large-scale, high-quality trajectories. The standard approach generates synthetic data through a self-improving loop: an agent is placed in a verifiable environment and iteratively fine-tuned on its successful trajectories. Despite its effectiveness, this paradigm exploits only successful trajectories and discards the failed ones, even though failures carry rich information about a model's weaknesses. In this work, we explore a complementary failure-driven self-improvement loop, a data-centric paradigm that turns failed trajectories into agent improvements. Specifically, we employ an LLM to diagnose failure modes, propose inference-time solutions, and generate code patches -- lightly verified by humans -- that upgrade the agent. We validate this approach with the state-of-the-art OpenCUA-72B model on the OSWorld benchmark, improving the success rate from 42.3% to 48.9%, a gain of 6.6 percentage points, without any additional training cost and with only modest inference overhead. Our results demonstrate that failure-driven self-improvement is a viable complement to success-based pipelines, enabling more efficient agent improvement. 

---
# HealthAgentBench: A Unified Benchmark Suite of Realistic Agentic Healthcare Environments for Challenging Frontier AI Agents 

**Authors**: Qianchu Liu, Sheng Zhang, Guanghui Qin, Jeya Maria Jose Valanarasu, Maximilian Rokuss, Mingyu Lu, Timothy Ossowski, Juan Manuel Zambrano Chaves, Cliff Wong, Peniel Argaw, Yashna Hasija, Mu Wei, Wen-wai Yim, Qin Liu, Zilin Jing, Jason Entenmann, Naoto Usuyama, Tristan Naumann, Hoifung Poon  

**Link**: [PDF](https://arxiv.org/pdf/2606.31179)  

**Abstract**: As AI agents become increasingly capable of complex, long-horizon reasoning, rigorous and holistic evaluation is essential for measuring progress toward real-world healthcare applications. We introduce HealthAgentBench, a suite of 54 agentic healthcare tasks across 7 categories each with its unique environment. The benchmark suite spans diverse workflows throughout the patient journey and a broad range of modalities. Each task is designed to replicate an end-to-end clinical workflow: given minimal instructions, an agent must explore raw healthcare data, operate within a complex environment, and execute multi-step solutions that go beyond naive prompting. A final task success rate is reported to provide a single, interpretable metric for HealthAgentBench overall performance for each agent. Evaluating frontier agents on HealthAgentBench, we find that overall task success rate remains low, underscoring the difficulty of the suite. The strongest and the most cost effective agent, Codex GPT-5.5, achieves only approximately 42% success rate. Beyond aggregate performance, HealthAgentBench reveals nuanced strengths and weaknesses across task categories. Frontier agents show promise in automatically developing research modeling pipelines over EHR data, but medical imaging remains especially challenging, particularly for Claude Code models, while Codex GPT-5.5 shows emerging capability. Tasks that combine large search spaces with compositional reasoning requirements remain difficult for all current agents. Together, these results suggest that HealthAgentBench provides a challenging and realistic benchmark with substantial room for future progress. We release our benchmark at this https URL. 

---
# ComplianceGate: Classifier-Gated Multi-Tier LLM Routing for Inference in Regulated Industries 

**Authors**: Abhishek Dey  

**Link**: [PDF](https://arxiv.org/pdf/2606.31163)  

**Abstract**: Large language models deployed in regulated industries operate under two constraints: compliance enforcement and cost efficiency. Personally identifiable information (PII) in user queries can reach model endpoints before the system determines whether that data should leave its jurisdictional boundary. Serving all queries through a single large model consumes full GPU capacity regardless of query complexity while offering no mechanism for geographic routing. Mixture-of-Experts architectures do not address this routing occurs between expert layers within the model after data has already arrived at the endpoint, with all experts loaded in memory regardless of query complexity. We propose a classifier-gated routing architecture that enforces compliance by design. A trained encoder classifier sits before any decoder inference, evaluating each query for complexity and data sensitivity, then routing it to an appropriately sized dense model in the appropriate geographic location. PII-containing queries route to local endpoints before any LLM computation begins, making data residency violations structurally impossible. Simple queries reach small, fast models at a fraction of the cost. Our evaluation on 600 queries demonstrates 39% median latency reduction, 33-52% cost savings depending on query distribution, and generation throughput of 122-200 tokens/second versus 50-64 for the baseline. The encoder classifier achieves 99.2% accuracy with near-perfect PII recall at 7ms inference overhead, establishing pre-inference classification as a practical path to compliance-by-design LLM deployment. 

---
# PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding 

**Authors**: Duc Cao Dinh, Khai Le-Duc, Florent Draye, Chris Ngo, Terry Jingchen Zhang, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2606.31148)  

**Abstract**: 3D Visual Grounding (3DVG) aims to localize target objects in 3D scenes given natural language descriptions. Existing approaches typically perform reasoning over the entire scene, leading to ambiguous predictions and high computational cost, especially in cluttered environments. We observe that many referential expressions rely on local spatial context and often correspond to restricted spatial regions rather than the full scene. Motivated by this insight, we propose PruneGround, an effective plug-and-play framework for 3DVG built upon three key components. First, we introduce Language-Guided Spatial Pruning (LGSP), which leverages a frozen Vision Language Model (VLM) to identify language-relevant regions, thereby reducing spatial computation and grounding candidates in the narrower search space. Second, we propose MultiView-Conditioned Description Reformulation (MCDR), which decomposes complex expressions into simplified target-anchor relations and augments missing spatial cues through multi-view reasoning. Finally, we propose LLM-Grounder, which repurposes a detection-pretrained spatial LLM into a language-conditioned grounding model by aligning point cloud and linguistic representations within the pruned region. Extensive experiments on the three most popular point cloud benchmarks demonstrate that our method achieves state-of-the-art results on all three ScanRefer settings and on 9 out of 10 Nr3D/Sr3D settings. Code and models are publicly available: this https URL 

---
# UniSAE: Unified Speech Attribute Editing on Speaker, Emotion and Low-Level Content via Discrete Phonetic Posteriorgram Modelling 

**Authors**: Chuanbo Zhu, Wuyou Zhou, Rongxiu Zhong, Shilei Zhang, Kun Qian, Yike Guo, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2606.31128)  

**Abstract**: Speech editing aims to modify specific portions of an utterance while preserving the remaining speech. Existing approaches primarily focus on word-level content modification and typically treat content, speaker, and emotion editing as separate tasks, limiting both editing granularity and flexibility. We propose UniSAE, a unified speech attribute editing framework which supports composable speaker, emotion and content editing from sub-phoneme to word level within a single architecture. UniSAE introduces a Discrete Phonetic PosteriorGram (DPPG) representation that factorizes speech content into discrete tokens encoding phoneme identity, pronunciation variants, and duration, enabling direct phoneme- and sub-phoneme-level editing. For higher-level modifications, an autoregressive content transformer predicts edited DPPG sequences for word-level content editing. The edited sequences are rendered into speech by a diffusion-based acoustic decoder, conditioned on disentangled speaker and emotion representations. Experimental results demonstrate that the proposed unified framework supports precise speaker and emotion control, content editing at multiple granularities, and joint modification of all three attributes within a single framework. 

---
# Usage frequency and application variety of research methods in library and information science: Continuous investigation from 1991 to 2021 

**Authors**: Chengzhi Zhang, Liang Tian, Heting Chu  

**Link**: [PDF](https://arxiv.org/pdf/2606.31081)  

**Abstract**: The present study analyzed over 26,000 research articles published between 1991 and 2021 in twenty-one major LIS (Library and Information Science) journals, using the machine learning (ML) approach to categorize the research methods used by LIS scholars. The findings of this study are significant. Firstly, there has been a shift in the research strategy from conceptual research (e.g., "Theoretical approach") to empirical research (e.g., "Interview") in LIS investigations over the past 31 years. Secondly, the research topics explored by LIS scholars during this period have moved from system-centered issues (e.g., "Information retrieval/models and algorithms") to user-centered topics (e.g., "Information services "). Thirdly, the study revealed dynamic and revealing relationships between the 18 research topics identified in the study and the 16 research methods commonly adopted in the LIS field. These dynamic relationships can be visualized by year and longitudinally via an interactive map created in this study. 

---
# ADAPT: Attention Dynamics Alignment with Preference Tuning for Faithful MLLMs 

**Authors**: Zhiyuan Yao, Zheren Fu, Zhixiao Zheng, Jiajun Li, Yi Tu, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2606.31054)  

**Abstract**: Multimodal Large Language Models (MLLMs) are critically hampered by hallucination, generating content inconsistent with the provided image. In this paper, we identify an internal signature of hallucination: progressive degradation of text-to-image cross-attention during generation, leading to specific failure patterns like unfocused or biased attention. Existing mitigation strategies are largely outcome-driven and do not explicitly target this failure mode. To address this problem, we propose ADAPT (Attention Dynamics Alignment with Preference Tuning), an attention-based framework that intervenes directly on text-to-image cross-attention dynamics. We propose ADAPT with three key contributions: a cross-attention visual anchor refined from early decoding to provide stable spatial grounding, an attention-supervised inference mechanism that detects and corrects attention drift online, and a Visual Attention Guidance DPO that aligns preferences toward visually grounded responses. Experiments show that each component of ADAPT contributes to hallucination reduction, and the full framework achieves new best results across multiple hallucination benchmarks, reducing hallucination rates by 40%-60% across mainstream backbones while preserving general multimodal capabilities. Our work provides an attention-based perspective on mitigating hallucinations by exploring the model's internal text-to-image cross-attention behaviors. Code is available at this https URL 

---
# Beyond Compilation: Evaluating Faithful Natural-Language-to-Lean Statement Formalization 

**Authors**: Ke Zhang, Patricio Gallardo Candela, Sudhir Murthy, Yi Xie, Zhi Wang, Maziar Raissi  

**Link**: [PDF](https://arxiv.org/pdf/2606.31002)  

**Abstract**: Theorem-proving benchmarks evaluate proof search against fixed formal statements, but natural-language-to-Lean formalization must generate the formal statement itself. In this setting, compilation is only a validity check: a Lean declaration may type-check while omitting hypotheses, changing domains, or expressing a vacuous claim. We study faithful statement formalization as both an evaluation problem and a bottleneck-attribution problem. On a 400-entry graduate-level benchmark spanning real analysis, complex analysis, topology, and algebra, our protocol combines Lean compilation, cross-model semantic judging, and human expert calibration. The resulting picture is different from compile-rate evaluation: a full tool-augmented agent reaches 89.5% compilation but only 60.5% consensus faithfulness, exposing a 29.0-point compile-pass but consensus-unfaithful gap. Targeted human audits support the metric as a conservative decision boundary: across available case-level audits, 96.0% of consensus-positive outputs are human-confirmed faithful, while 82.4% of compile-pass consensus-negative outputs are human-confirmed semantic failures. Under this metric, existing one-shot formalizer models and prover-oriented Lean models remain low, suggesting that formal validity, proof-oriented Lean competence, and faithful statement generation should be reported separately. We then use a full $2^3$ factorial design to decompose three recurring interventions in formalization pipelines: parametric expert drafting, Mathlib/context search, and Lean elaboration feedback. Elaboration feedback is the largest validity intervention, but it also exposes a larger compile-pass semantic-failure bucket; search mainly improves grounding and selectivity; and fine-tuned drafting is largely substitutable in this tool stack once feedback and grounding are available. 

---
# When Does Learning to Stop Help? A Cost-Aware Study of Early Exits in Reasoning Models 

**Authors**: Zhe Dong, Fang Qin, Manish Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.30852)  

**Abstract**: Reasoning models spend different amounts of useful computation across instances, but it remains unclear when a learned stopping rule improves over simple confidence or convergence thresholds. We study this question with LearnStop, a hidden-state-free checkpoint stopper for reasoning language models. At fixed budget checkpoints, LearnStop probes a short answer from the current reasoning prefix and predicts prefix correctness from online features such as answer confidence, entropy, prefix vote share, answer stability, and backtracking-marker density. Across 18 task-model settings spanning GSM8K, MATH-500, MMLU-Pro, AIME-90, GPQA, Qwen3, and DeepSeek-R1 distillations, the answer is task-dependent. On free-form math, learned multi-feature stopping improves the fixed-budget frontier and often beats scalar exits: on GSM8K with Qwen3-32B, the empirical frontier reaches a post-hoc peak adapt gain of +0.157, validation-selected operating points preserve positive gains, and the paired gain over the strongest scalar baseline is +0.028. On multiple-choice and very hard settings, scalar confidence, entropy, or stability rules are competitive or stronger. We therefore frame learned stopping not as a universal replacement for scalar exits, but as a tool whose value depends on trajectory structure. We further provide validation-selected operating points, paired bootstrap tests, finite-grid lost-correct risk calibration, cost accounting under KV-fork, prefix-cache, and black-box regimes, H100 serving profiles, checkpoint-schedule sweeps, transfer analyses, and robustness checks. The main practical finding is that learned stopping is useful when many questions become correct before full budget but do not exhibit a single reliable scalar stopping signal; its benefits largely disappear when confidence or answer convergence already solves the stopping problem. 

---
# Information Terra: A Narrative-Anchored Semantic-First Projection of Document Embeddings 

**Authors**: Brian Keith-Norambuena, Fausto German, Chris North  

**Link**: [PDF](https://arxiv.org/pdf/2606.30824)  

**Abstract**: We introduce Information Terra, a narrative-anchored semantic-first projection that places a document corpus on an Earth-like globe whose poles are two user-chosen endpoint documents and whose prime meridian is the great-circle geodesic between them on the embedding hypersphere -- so latitude encodes narrative progress and longitude thematic deviation. Land features are recovered from document density via kernel density estimation and labeled by theme. A narrative trail built from the underlying narrative coherence graph, and constrained to be monotone in geodesic progress, provides a readable storyline. The projection's axes are semantically grounded in the user's chosen narrative endpoints, and the globe metaphor affords rotation and antipodal reading. We demonstrate the method on a 540-article Cuban Protests corpus, showing a storyline from Obama's 2016 visit to the 2021 International Aid during the protests. 

---
# Revocable Learned State via Process Sidecars 

**Authors**: John Sweeney  

**Link**: [PDF](https://arxiv.org/pdf/2606.30788)  

**Abstract**: Language models are often adapted in stages: a public skill phase, a private memory phase, and a later safety phase that learns to refuse outputs tied to the remembered entities. Revoking the memory after the safety phase is not the same problem as subtracting the memory update: the later safety optimizer has transported the memory direction. We introduce process sidecars, a two-coefficient edit family $\hat{\theta}(\lambda,\gamma)=\theta_{\mathrm{AMS}}-\lambda\Delta_{\mathrm{M}}-\gamma\hat{R}_{\mathrm{S}\leftarrow\mathrm{M}}$, with $\hat{R}_{\mathrm{S}\leftarrow\mathrm{M}}=\hat{J}_{\mathrm{S},\varepsilon}(\Delta_{\mathrm{M}})-\Delta_{\mathrm{M}}$, where $\hat{J}_{\mathrm{S},\varepsilon}$ is a centered secant through the realized future AdamW safety-training process. The implementation uses $\varepsilon=1$ at the natural memory-edit scale; it reuses $\theta_{\mathrm{AMS}}$ as the positive endpoint and computes one additional safety trace at $\theta_{\mathrm{A}}-\Delta_{\mathrm{M}}$. We prove two things. First, the exact sidecar, using the true transported direction $R_{\mathrm{S}\leftarrow\mathrm{M}}$ rather than the secant estimate, at $(\lambda,\gamma)=(1,1)$ recovers the counterfactual safety-only oracle $\theta_{\mathrm{AS}}$ up to second order; the proof treats AdamW as an augmented-state map over parameters, first moments, and second moments. Second, this process information is necessary: whenever future safety training bends the memory direction, every scalar task-arithmetic edit leaves first-order counterfactual error, while the process-sidecar edit is second-order accurate. Across three models, the validation-selected 2D edit improves held-out refusal closure over naive task arithmetic in all trials, and over the $\gamma=\lambda$ process-JVP subfamily, the diagonal slice of the cached 2D grid, in all paired trials. 

---
# From Search to Synthesis: Training LLMs as Zero-Shot Workflow Generators 

**Authors**: Gan Luo, Zihan Qin, Bin Dong, Wotao Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.30704)  

**Abstract**: Large language models (LLMs) excel across a wide range of tasks, yet their instance-specific solutions often lack the structural consistency needed for reliable deployment. Workflows that encode recurring algorithmic patterns at the task level provide a principled framework, offering robustness across instance variations, interpretable traces for debugging, and reusability across problem instances. However, manually designing such workflows requires significant expertise and effort, limiting their broader application. While automatic workflow generation could address this bottleneck, existing methods either produce instance-specific solutions without learning task-level patterns, or cannot generalize beyond their training configurations. We present MetaFlow, which casts workflow generation as a meta-learning problem: given a task and an operator set, the model learns to compose solution strategies. MetaFlow trains in two stages: supervised fine-tuning on synthetic workflow data, followed by reinforcement learning with verifiable rewards (RLVR) that uses execution feedback across problem instances in the task to improve end-to-end success. The resulting model produces effective workflows for trained tasks and exhibits strong generalization to untrained tasks and novel operator sets. Across benchmarks in question answering, code generation, and mathematical reasoning, MetaFlow achieves performance comparable to state-of-the-art baselines on in-domain tasks with single inference, while demonstrating remarkable zero-shot generalization capabilities on out-of-domain tasks and operator sets. 

---
# ViTL: Temporal Logic-Guided Zero-Shot Natural Language Navigation via Vision-Language Models 

**Authors**: Kaier Liang, Hengde Dai, Cristian-Ioan Vasile  

**Link**: [PDF](https://arxiv.org/pdf/2606.30696)  

**Abstract**: Enabling robots to follow natural language commands to complete zero-shot long-horizon tasks remains challenging. It requires extracting implicit temporal and logical constraints from natural language commands and executing multiple sub-tasks accordingly. Recent zero-shot object navigation methods use vision-language models (VLMs) to guide frontier-based exploration in unknown environments, but they are limited to single-target tasks. Real-world commands such as "Clean either the chair or the couch, then turn on the tv." require navigating to multiple targets in a temporally constrained order, which no existing zero-shot system can handle. We present ViTL, a framework that addresses this gap at two levels. At the task level, we use a large language model (LLM) to compile natural language commands into Linear Temporal Logic (LTL) formulas, which are then converted into Deterministic Finite Automata~(DFA) that coordinate multi-channel value maps and trigger dynamic replanning when new objects are detected. At the navigation level, we introduce directional score: rather than producing a direction-agnostic value across the entire field of view, we label frontier directions on the observation image and extract per-direction scores from the VLM. Experiments on Habitat-Matterport 3D (HM3D) show that the full framework enables zero-shot long-horizon completion of natural language navigation tasks with temporal constraints, and that directional score improves single-target navigation accuracy and efficiency over the baseline. 

---
# Emergent Culture in Minimal LLM Systems 

**Authors**: Simon Jones, Sabine Hauert  

**Link**: [PDF](https://arxiv.org/pdf/2606.30668)  

**Abstract**: What happens when LLM agents operate with no context outside a turn, minimal prompting, and simple tools? Inspired by swarm engineering, we give collectives of three agents the ability to send messages and manipulate a shared actively decaying text store, introducing evolutionary pressure. The agents spontaneously cooperate, develop storage management strategies, and generate complex evolving cultural artifacts, with no top-down engineering. Using tools from dynamical systems analysis, we show that these behaviours exhibit structured long-range coherence beyond the entropy horizon of the decaying store, consistent with emergent culture in the Sperberian sense. 

---
# ASR-Agnostic Multimodal Spectrotemporal Modeling for Early Dementia Detection 

**Authors**: Chukwuemeka Ugwu, Oluwafemi Richard Oyeleke  

**Link**: [PDF](https://arxiv.org/pdf/2606.30646)  

**Abstract**: Speech recruits the same executive, attentional, and working memory processes underlying instrumental activities of daily living, or IADLs, providing a non-invasive proxy for cognitive assessment. Yet most speech-based dementia detection systems depend on transcription, discard within-recording temporal structure, and are validated on a single English corpus with known recording artifacts. We propose an ASR-agnostic framework operating directly on Mel spectrograms. Our key contribution is extracting spectrotemporal displacement fields from consecutive spectrogram frames, capturing shifting spectral energy patterns as digital biomarkers of cognitive decline. These features are fused with CNN-ConvGRU acoustic embeddings via a learned cross-attention mechanism and aggregated using a Transformer encoder with learnable query pooling. A composite temporal loss enforces smoothness and contrastive coherence across segments. We train independent models on English DementiaBank, Slovak EWA-DB, and Spanish Ivanova corpora, using clinical elicitation protocols taxing IADL-relevant cognitive domains. The Slovak model achieves 83.9% accuracy, and Spanish achieves, while the English baseline yields 53.2%, confirming known artifacts. Cross-lingual ablation studies reveal distinct fusion regimes: removing cross-attention collapses Spanish performance to 53.7%, below unimodal models, while the Slovak audio encoder alone outperforms the full model, 93.7% vs. 83.9%, and all English configurations remain near chance. Thus, multimodal fusion's value is corpus-dependent: essential when signal is distributed across modalities, counterproductive when one dominates, and irrelevant when no signal exists. Auxiliary temporal losses converge to language-invariant values, indicating cross-lingual architectural stability. 

---
