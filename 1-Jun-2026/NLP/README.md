# Language Models Learn Constructional Semantics, Not To Mention Syntax: Investigating LM Understanding of Paired-Focus Constructions 

**Authors**: Wesley Scivetti, Ethan Wilcox, Nathan Schneider, Kanishka Misra, Leonie Weissweiler  

**Link**: [PDF](https://arxiv.org/pdf/2605.31586)  

**Abstract**: Grasping the semantics of rare constructions (form-meaning pairings) has been shown to be a challenging problem that has currently only been solved by the largest LLMs. It remains an open question if open-source models have robust constructional understanding, and if so, what learning dynamics underlie the acquisition of this knowledge. Focusing on a set of rare Paired-Focus constructions in English (e.g. "let alone", "much less"), we construct a novel dataset to test their meanings using both scalar adjectival semantics and general world knowledge. Testing a wide range of models differing in parameter count, architecture, and pretraining dataset size, we find that several modestly sized models are sensitive to both the forms and the meanings of Paired-Focus constructions, though models trained on human-scale data fail at all meaning evaluations. Turning to training dynamics for a set of open-checkpoint models, we find that Paired-Focus understanding emerges later in training than Paired-Focus syntactic knowledge, and that learning of Paired-Focus semantics is correlated with gains in some domains of world knowledge. Overall, our empirical results support the conclusion that modestly sized open-source models can grasp the rare Paired-Focus constructions, and demonstrate a connection between knowledge of Paired-Focus constructions and other meaning domains. 

---
# LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards 

**Authors**: Nianyi Lin, Jiajie Zhang, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.31584)  

**Abstract**: Long-context reasoning remains a central challenge for large language models, which often fail to locate and integrate key information in extensive distracting content. Reinforcement learning with verifiable rewards (RLVR) has shown promise for this task, yet existing methods are limited by low-confusability distractors and sparse, outcome-only reward signals that cannot supervise intermediate reasoning steps. To address these issues, we introduce \textsc{LongTraceRL}. For data construction, we generate multi-hop questions via knowledge graph random walks and leverage search agent trajectories to build \emph{tiered distractors}: documents the agent read but did not cite (high confusability) and documents that appeared in search results but were never opened (low confusability), producing training contexts that are far more challenging than those built by random sampling or one-shot search. For reward design, we propose a \emph{rubric reward} that uses the gold entities along each reasoning chain as fine-grained, entity-level process supervision. This rubric reward is applied only to responses with correct final answers (positive-only strategy), distinguishing the reasoning quality among correct responses and preventing reward hacking. Experiments on three reasoning LLMs (4B--30B) across five long-context benchmarks demonstrate that \textsc{LongTraceRL} consistently outperforms strong baselines and encourages comprehensive, evidence-grounded reasoning. Codes, datasets and models are available at \href{this https URL}{this https URL}. 

---
# What Gets Unmasked First? Trajectory Analysis of Diffusion Models for Graph-to-Text Generation 

**Authors**: Qing Wang, Jacob Devasier, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.31564)  

**Abstract**: We present the first systematic study of masked diffusion language models (MDLMs) for graph-to-text generation. We analyze MDLM generation trajectories -- the order in which tokens are unmasked during iterative decoding -- and find that, unlike autoregressive LLMs which generate text linearly, MDLMs naturally prioritize entities first, followed by relational and function words, with structural tokens resolved last. We further identify a previously undocumented failure mode of supervised fine-tuning: SFT disrupts this strategy by prematurely anchoring structural sentence-ending tokens early in the decoding trajectory, effectively fixing the output length which can lead to omitted or hallucinated information. To address this, we propose lambda-scaled structural decoding, a training-free inference-time modification that downweights structural token confidence and recovers +9.4 BLEU-4. Finally, we introduce Graph-LLaDA, which integrates a Graph Transformer encoder into LLaDA's decoding process to explicitly incorporate relational graph structure. Cross-dataset evaluation on LAGRANGE reveals that previous baselines overfit to dataset-specific patterns, while LLM- and MDLM-based approaches generalize significantly better. 

---
# Disagreeing Rationales: Rethinking Classification and Explainability Evaluation in Hate Speech Detection 

**Authors**: Benedetta Muscato, Beiduo Chen, Gizem Gezici, Barbara Plank, Fosca Giannotti  

**Link**: [PDF](https://arxiv.org/pdf/2605.31563)  

**Abstract**: Human disagreement is ubiquitous and well-known in labeling. However, variation in explanations, captured through token-level human rationales, remains far less explored. At the same time, it is unclear how to best evaluate human labels and rationales -- or even how to best aggregate rationales beyond majority vote -- in light of this variation. Yet, rationales may provide additional insights into the richness of human reasoning, that may differ in style, values and interpretations -- especially in subjective NLP tasks like hate speech detection. In this work, we unify diverse models, training strategies, loss functions, and existing evaluation metrics under a single protocol by systematically re-implementing them across different label and rationale representation spaces. Classification metrics are organized around two key properties -- predictive and distributional -- while explainability metrics through three complementary dimensions: plausibility, faithfulness, and complexity. In this unified supervision framework, we evaluate model behavior across classification and explainability metrics, as well as metric sensitivity to the choice of label (hard and soft) and rationale representation space (hard, intermediate and soft). Results show that both hard and soft metrics favor softer representations, highlighting their effectiveness in capturing variation and the need to rethink evaluation in subjective NLP. 

---
# What Am I Missing? Question-Answering as Hidden State Probing 

**Authors**: Chu Fei Luo, Samuel Dahan, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31561)  

**Abstract**: Test-time reasoning has become a significant field of study since the introduction of chain-of-thought reasoning in large language models (LLMs). However, the mechanisms of this reasoning process are still under-explored -- from the same input prompt, and even the same partial solution, LLMs can produce varied answers if sampled multiple times. We propose to leverage question-asking as an inference-time intervention that articulates information about the model's hidden state. To achieve that, we present a student-teacher setting where a student asks questions to a teacher. We train a probe on the student's hidden state before and after asking a question and find it is predictive of the trajectory's final correctness, even before generating the teacher's answer. This suggests there is a meaningful signal from the self-diagnosis that occurs during question generation rather than information transfer from the teacher. We then frame question-asking as a sequential decision problem, using this probe as a quality score, and define a gating policy to ask questions that maximize likelihood of correctness. We find that the success of question-asking as an intervention is largely dependent on the model's self-consistency. Our empirical results show a gap between detection and recovery; while our gating policy captures model correctness and uncertainty, interventions are equally likely to harm correct trajectories as they are to recover incorrect ones. This gap between diagnosis and correction has broader implications on language models' capacity for self-refinement under uncertainty. 

---
# Semantic Triplet Restoration: A Novel Protocol for Hierarchical Table Understanding in Large Language Models 

**Authors**: Yibin Zhao, Fangxin Shang, Dingrui Yang, Yuqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31550)  

**Abstract**: Table question answering requires models to recover semantic relations encoded implicitly by two-dimensional layout, merged cells, and hierarchical headers. Current pipelines typically use HTML or Markdown as intermediate table representations, but these layout-oriented serializations introduce markup overhead and require large language models to infer header-cell alignments from row and column spans. We propose Semantic Triplet Restoration (STR), a protocol that rewrites each cell as an atomic fact <item path, feature path, value>, where the item path specifies the row-wise entity, the feature path specifies the hierarchical attribute, and the value contains the cell content. We also present TripletQL, a lightweight query-aware router that uses STR to select an appropriate rendering or filtered subset of triplets for each question. Across four Chinese and English table-QA benchmarks, STR matches or improves upon HTML-based baselines while reducing input tokens. The relative benefit grows for smaller language models and longer table contexts, suggesting that explicit semantic representations are especially useful under constrained inference budgets. Code and data are available at this https URL . 

---
# Preference-Aware Rubric Learning for Personalized Evaluation 

**Authors**: Yilun Qiu, Xiaoyan Zhao, Yang Zhang, Yuxin Chen, Cilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, Yoko Yamakata, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2605.31545)  

**Abstract**: As Large Language Models (LLMs) evolve from general-purpose assistants to user-centric agents, personalization has become central to aligning model behavior with individual preferences, making the evaluation of personalized alignment a critical bottleneck. Existing evaluation methods-ranging from automatic metrics to LLM-as-a-judge approaches-fail to capture subjective, user-specific preferences embedded in long-term interaction histories. We identify three essential principles for reliable and effective personalized evaluation: Representativeness, User-Consistency, and Discriminativeness. To address these principles, we introduce Personalized Evaluation as Learning, a paradigm that formulates personalized evaluation as a learning problem rather than a static judgment. Under this paradigm, we propose PARL (Preference-Aware Rubric Learning for Personalized Evaluation), a framework that learns to induce preference-aware evaluation rubrics directly from raw user histories and performs a self-validation mechanism to ensure consistency with the user's preferences. PARL integrates rubric induction with a discriminative reinforcement learning objective that contrasts user-authored responses against competitive personalized model outputs, enabling the learned rubrics to capture precise, user-specific decision boundaries. Experiments on real-world personalized text generation tasks show that PARL consistently induces high-fidelity rubrics that reliably identify user-aligned responses and generalize across users and tasks, while capturing stable stylistic preferences and fine-grained evaluative patterns. To ensure reproducibility, our code is available at this https URL. 

---
# UniAudio-Token: Empowering Semantic Speech Tokenizers with General Audio Perception 

**Authors**: Yuhan Song, Linhao Zhang, Aiwei Liu, Chuhan Wu, Sijun Zhang, Wei Jia, Yuan Liu, Houfeng Wang, Xiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.31521)  

**Abstract**: Semantic speech tokenizers have become a widely used interface for Audio-LLMs, owing to their compact single-codebook design and strong linguistic alignment. However, their focus on linguistic abstraction induces acoustic blindness, limiting their applicability beyond speech-centric tasks. We propose UniAudio-Token, a framework that empowers semantic tokenizers with general audio perception without compromising speech ability. Instead of altering the semantic paradigm, UniAudio-Token mitigates its information loss through two key innovations: (1) Semantic-Acoustic Primitives (SAP) provide structured supervision by decomposing audio into linguistic content, vocal attributes, and auditory-scene primitives; and (2) Semantic-Acoustic Equilibrium (SAE) introduces a content-aware gating mechanism that adaptively restores fine-grained acoustic details from shallow layers. Extensive evaluations show that UniAudio-Token learns comprehensive universal representations while preserving high-fidelity speech generation. When integrated with downstream LLMs, it outperforms all single-codebook baseline tokenizers on both understanding and generation tasks, effectively serving as a unified audio interface. We publicly release all our code, including training and inference scripts, together with the model checkpoints at this https URL. 

---
# If LLMs Have Human-Like Attributes, Then So Does Age of Empires II 

**Authors**: Adrian de Wynter  

**Link**: [PDF](https://arxiv.org/pdf/2605.31514)  

**Abstract**: Much research has been carried out on large language models (LLMs) and LLM-powered agentic workflows. However, many works within the field state emergence of, ascribe to, or assume, generalised anthropomorphic attributes to them (e.g., morality or understanding of natural language). Our goal is not to argue in favour or against the existence of these attributes, but to point out that these conclusions could be incorrect. For this we build and train a simple neural network on the videogame Age of Empires II, and note that any entity in a sufficiently-powerful substrate, such as LEGO or the Greater Boston Area, could also present such attributes. Hence, the purported anthropomorphic attributes of LLMs are empirically non-unique: although some properties (e.g., responses to prompts) could remain constant, others, such as the interpretation of their perceived behaviour, might change with the substrate. Thus, any empirically-grounded discussion requires explicit measurement criteria; otherwise the interpretation is left to the representation. We then show that assuming that these attributes exist or not in a system, independent of the substrate and in a generalised way, leads to either circular or uninformative conclusions, regardless of the experimenter's viewpoint on the subject. Finally we propose a 'null' assumption, where one assumes LLM non-uniqueness instead of assuming anthropomorphic attributes to set up an experiment, along with examples of it. We also discuss potential objections to our work, briefly survey the field, and prove that \textit{Age of Empires II} is functionally- and Turing-complete. 

---
# Reliable Multilingual Orthopedic Decision Support from Clinical Narratives: Language-Aware Adaptation and Verification-Guided Deferral 

**Authors**: Danish Ali, Li Xiaojian, Sundas Iqbal, Farrukh Zaidi  

**Link**: [PDF](https://arxiv.org/pdf/2605.31512)  

**Abstract**: Multilingual orthopedic decision support remains challenging in low-resource healthcare settings, where clinical narratives contain specialized terminology, mixed scripts, incomplete evidence, label imbalance and language-dependent documentation patterns. This article presents a reliability-oriented framework for classifying free-text orthopedic notes in English, Hindi and Punjabi. We compare task-aligned multilingual transformer encoders, a task-fine-tuned DistilBERT baseline, zero-shot instruction-tuned large language models (LLMs) and a domain-adaptive encoder, IndicBERT-HPA. IndicBERT-HPA augments IndicBERT with language-aware orthopedic adapter heads to support clinically relevant multilingual representation learning. Evaluation extends beyond aggregate accuracy to per-class performance, ROC-AUC, AUPRC, expected calibration error, cross-language stability and robustness under controlled balanced and natural-prevalence distributions. The evaluated zero-shot LLMs remain substantially less effective than task-adapted encoders for closed-set classification, with language-dependent instability. Under natural clinical prevalence, IndicBERT-HPA achieves the strongest overall performance, reaching an averaged Macro-F1 of 0.8792, Macro-AUROC of 0.894 and AUPRC of 0.902. We further implement a deterministic selective-verification layer combining confidence gating, evidence-consistency checking and language-risk screening. On a randomly selected held-out 5,000-record subset, it achieves 84.4% selective accuracy and 0.76 selective Macro-F1 at 72.3% coverage, compared with 71.5% accuracy and 0.65 Macro-F1 for accept-all prediction. These results support reliability-oriented multilingual clinical decision support with explicit deferral. 

---
# Consolidating Rewarded Perturbations for LLM Post-Training 

**Authors**: Zheyu Zhang, Shuo Yang, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2605.31494)  

**Abstract**: Post-training of language models is commonly framed as a sample-score-update loop implemented by gradient descent. A recent line of work, exemplified by RandOpt, relocates this loop to weight space, sampling Gaussian perturbations around a pretrained model and ensembling the top-K rewarded specialists at inference. While competitive with PPO and GRPO under matched training compute, this prediction-level ensemble incurs K forward passes per test example and does not extend cleanly to free-form generation. We ask whether the rewarded population can instead be folded into a single deployable model, replacing the inference-time ensemble with one consolidated update. A split-half analysis over 25 model-task pairs reveals reproducible low-rank structure in every case. We turn this geometry into CoRP (Consolidating Rewarded Perturbations), a gradient-free operator that combines reward-weighted aggregation, compatibility-aware reweighting, and a held-out validation gate, with no gradient flowing through the language model. Across five language models from 0.5B to 8B and five tasks covering math, code, and creative writing, CoRP improves the base model by 8.1 points on average. Using one tenth of RandOpt's perturbation budget, CoRP exceeds single-inference RandOpt by 6.5 points and recovers more than half of the gain of the 50-pass majority-vote ensemble, at one forward pass per test example. 

---
# Are Full Rollouts Necessary for On-Policy Distillation? 

**Authors**: Yaocheng Zhang, Jiajun Chai, Songjun Tu, Yuqian Fu, Xiaohan Wang, Wei Lin, Guojun Yin, Qichao Zhang, Yuanheng Zhu, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.31490)  

**Abstract**: On-policy distillation (OPD) provides dense teacher feedback along rollouts generated by the student and has emerged as a promising post-training paradigm for long-horizon reasoning. However, standard OPD typically generates full rollouts during training, which is computationally expensive and may expose the student to unreliable teacher feedback at late rollout positions, especially during early training. We identify the rollout horizon as a key bottleneck in OPD that substantially impacts training efficiency. Unlike Reinforcement Learning with Verifiable Rewards (RLVR), OPD does not require a complete trajectory or a final answer reward to provide learning signals. This observation suggests that full rollouts may not always be necessary for effective OPD. Motivated by this insight, we propose two simple horizon-control strategies: Progressive OPD (POPD), which gradually expands the rollout horizon during training, and Truncated OPD (TOPD), which permanently performs distillation on reliable truncated rollouts. Experiments on mathematical reasoning show that POPD improves the training efficiency of OPD by up to 3$\times$, while TOPD matches OPD performance using only 10\% of the rollout horizon, leading to substantial wall-clock and memory reductions. These results demonstrate that controlling the rollout horizon offers a simple and practical path to more efficient OPD. 

---
# BenHalluEval: A Multi-Task Hallucination Evaluation Framework for Large Language Models on Bengali 

**Authors**: Shefayat E Shams Adib, Ahmed Alfey Sani, Ekramul Alam Esham, Ajwad Abrar, Ishmam Tashdeed, Md Taukir Azam Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2605.31483)  

**Abstract**: Despite Bengali being the sixth most spoken language in the world, no prior work has systematically evaluated hallucination in large language models (LLMs) for Bengali. We introduce BenHalluEval, a fine-grained hallucination evaluation framework for Bengali covering four tasks: Generative Question Answering (GQA), Bangla-English Code-Mixed QA, Summarization, and Reasoning. We construct 12,000 hallucinated candidates using GPT-5.4 across twelve task-specific hallucination types, drawn from three existing Bengali datasets, and evaluate seven LLMs spanning reasoning-oriented, multilingual, and Bengali-centric categories under a dual-track protocol that independently measures false-positive rate on ground-truth instances (Track A) and hallucination detection rate on hallucinated candidates (Track B). To jointly penalise both failure modes and prevent inflated scores from uniform response bias, we propose BenHalluScore, a dual-track calibration metric that ranges from 7.72% to 55.42% across models and tasks, revealing substantial variation in hallucination calibration. Chain-of-thought prompting, applied as a mitigation strategy, shifts response distributions without consistently improving hallucination discrimination. BenHalluEval establishes the first dedicated hallucination benchmark for Bengali and highlights the inadequacy of single-track and prompting-only evaluation approaches for low-resource language settings. The dataset and code are available at this https URL. 

---
# Language Models Can Resolve Reference Compositionally, But It's Not Their Native Strength: The Case of the Personal Relation Task 

**Authors**: Bart Evelo, Meaghan Fowlie, Denis Paperno  

**Link**: [PDF](https://arxiv.org/pdf/2605.31480)  

**Abstract**: Do neural models, such as Large Language Models, genuinely acquire compositional abilities for interpretation of natural language? When we talk about semantic interpretation, we can distinguish two complementary aspects: establishing what an expression refers to in the world (which we call the Extensional task) and representing its sense in a structured way (which we call the Intensional task). We evaluate LLMs and humans on both tasks in the setting of the Personal Relation Task (Paperno 2022) in which, given a universe of people and their relationships with each other, one is asked to interpret a noun phrase such as "Amber's parent's friend". Here, for the Intensional task, the answer is the formula "friend(parent(amber))", and for the Extensional task, the person. We find that humans and LLMs show opposite strengths: humans perform better on Extensional than Intensional tasks, and LLMs vice versa. Our methodology brings greater nuance to the understanding of compositional abilities in modern machine learning models. Our results support the notion that the lack of referential grounding in LLM training is a crucial missing component in mimicking human-like language understanding. 

---
# Scaling Conversational Hungarian ASR: The BEA-Dialogue+ Corpus 

**Authors**: Máté Gedeon, Piroska Zsófia Barta, Péter Mihajlik, Katalin Mády  

**Link**: [PDF](https://arxiv.org/pdf/2605.31469)  

**Abstract**: Conversational automatic speech recognition in Hungarian is constrained by the limited amount of publicly available dialogue-style training data. The BEA-Dialogue corpus addresses this need, but its strictly speaker-disjoint train/dev/eval split reduces the usable material to only 85 hours. In this paper, we introduce BEA-Dialogue+, an expanded version of the corpus that relaxes the split criterion for experimenters and dialogue partners while preserving complete separation of the primary speakers. This results in 200 hours of transcribed natural conversations and enables a controlled study of the trade-off between additional training data and speaker overlap across the splits. We evaluate several Whisper- and FastConformer-based models on both corpus versions, including Serialized Output Training (SOT)-based fine-tuning for dialogue transcription. Our results show that the larger corpus is more challenging for models without fine-tuning, whereas SOT-based adaptation yields consistent improvements in WER, CER, cpWER, and cpCER. Overall, BEA-Dialogue+ provides a substantially larger yet still demanding benchmark for Hungarian dialogue ASR, and a practical resource for training and evaluating dialogue transcription systems. 

---
# Translation Analytics for Freelancers II: Benchmarking Local LLMs for Confidential Translation Workflows 

**Authors**: Yuri Balashov, Rex VanHorn, Mingxi Xu, Austin Downes  

**Link**: [PDF](https://arxiv.org/pdf/2605.31452)  

**Abstract**: Building on our previous work, this paper develops practical, low-barrier methods for freelance translators and smaller language service providers to evaluate translation technologies using rigorous yet accessible analytic methods. Here we address a high-stakes, specialized need: offline translation for confidentiality-sensitive domains in which privacy constraints preclude the use of cloud-based engines and commercial LLMs. We expand the Reeve Foundation Trilingual Corpus (RFTC) used in our previous work into a multilingual corpus (RFMC) by adding sentence-aligned German and Simplified Chinese reference translations. We then benchmark several locally runnable language models (via Ollama) across four language directions on 1000+ sentences selected from this corpus. We use consistent single-prompt calls without fine-tuning or domain adaptation, comparing local LLM outputs against commercial NMTs (DeepL, Baidu), a frontier LLM (GPT-5.2), and professional-grade local NMT systems (OPUS-CAT, NeuralDesktop, Promt). Automatic evaluation is conducted with MATEO. Results reveal substantial variation in local LLM performance across language directions and model sizes. The best local LLMs match or surpass local NMT systems and a frontier LLM, though they remain behind top commercial NMTs. These findings underscore the viability of carefully selected local LLM translation for privacy-constrained professionals and inform future research on model scaling and multilingual capability. 

---
# Fine-grained Verification via Diagnostic Reasoning Supervision for Aspect Sentiment Triplet Extraction 

**Authors**: Wenna Lai, Haoran Xie, Guandong Xu, Qing Li, S. Joe Qin  

**Link**: [PDF](https://arxiv.org/pdf/2605.31446)  

**Abstract**: Aspect Sentiment Triplet Extraction (ASTE) aims to identify aspect terms, opinion terms, and sentiment polarities as structured triplets, providing essential inputs for downstream information system applications such as opinion mining, explainable recommendations, and review summarization. Prior work mainly focuses on end-to-end extraction, while post hoc verification of extracted triplets remains comparatively underexplored. This gap limits the reliability of ASTE systems, since predicted triplets may be locally plausible while being globally invalid. Moreover, candidate invalidity is multi-faceted and candidate usability is inherently graded, motivating a fine-grained verification mechanism that can filter or re-rank outputs from diverse extractors. In this paper, we propose FiVeD, a framework for Fine-grained Verification with Diagnostic reasoning supervision. Specifically, the verifier is trained with multiple complementary objectives, including validity classification and quality score estimation as primary tasks, with error type classification and rationale generation as auxiliary tasks. We define hierarchical error categories and construct plausible incorrect triplets under semantic and syntactic constraints, and leverage an off-the-shelf LLM with task-specific rubrics to produce quality scores and diagnostic rationales. During inference, the resulting quality scores are used to filter candidate outputs, supporting adjustable precision-recall tradeoffs. Experiments across multiple ASTE baselines demonstrate that FiVeD consistently improves extraction performance by up to 3.53 F1 points as a plug-and-play verification module. 

---
# SCOPE: Self-Play via Co-Evolving Policies for Open-Ended Tasks 

**Authors**: Wai-Chung Kwan, Aryo Pradipta Gema, Joshua Ong Jun Leang, Pasquale Minervini  

**Link**: [PDF](https://arxiv.org/pdf/2605.31433)  

**Abstract**: Self-play can train language models without external supervision. However, existing methods require rule-checkable answers, leaving open-ended tasks dependent on curated prompts or frontier-model judges. We introduce SCOPE, a data-free self-play framework for open-ended tasks that co-evolves two policies: a Challenger that generates document-grounded tasks, and a Solver that answers them through multi-turn retrieval. A frozen copy of the initial model serves as the self-judge, which writes task-specific rubrics from the source document and grades Solver responses against them. Across three 7-8B instruction-tuned models (Qwen2.5, Qwen3, OLMo-3), SCOPE improves open-ended performance by up to +10.4 points on eight benchmarks and matches or exceeds GRPO_data trained on ~9K curated prompts. Although trained only on open-ended tasks, SCOPE also improves held-out short-form QA by up to +13.8 points on seven held-out benchmarks, surpassing GRPO_data on all three models. Ablations show that co-evolving the Challenger is necessary to keep tasks near the Solver's frontier, that gains arise from improvements in both retrieval and synthesis with the relative contribution varying by task, and that rubric generation quality is the bottleneck for self-judging. 

---
# DOA: Training-Free Decoder-Only Attention Policy for Long-Form Simultaneous Translation with SpeechLLMs 

**Authors**: Sara Papi, Luisa Bentivogli  

**Link**: [PDF](https://arxiv.org/pdf/2605.31432)  

**Abstract**: Simultaneous speech-to-text translation (SimulST) generates translations while speech is still unfolding, requiring a streaming policy that decides when to read and when to write. State-of-the-art approaches rely on attention-based encoder-decoder models where cross-attention provides explicit alignment signals. In contrast, Speech Large Language Models (SpeechLLMs) are decoder-only architectures relying solely on self-attention. This raises a central question: whether decoder self-attention contains sufficiently stable alignment signals to guide the streaming policy. Moreover, existing approaches typically rely on training-based adaptations or heuristic wait-$k$ policies and have not been validated in long-form settings. To fill these gaps, we propose Decoder-Only Attention (DOA), a training-free policy that enables long-form simultaneous translation with off-the-shelf SpeechLLMs by deriving a proxy alignment from self-attention. Experiments on Phi4-Multimodal and Qwen3-Omni show that DOA provides an effective alignment signal for supporting streaming decisions, enabling low-latency long-form SimulST with quality close to offline decoding without retraining. 

---
# Neuro-symbolic Syntactic Parsing: Shaping a Neural Network with the CYK Algorithm 

**Authors**: Fabio Massimo Zanzotto, Federico Ranaldi, Giorgio Satta  

**Link**: [PDF](https://arxiv.org/pdf/2605.31421)  

**Abstract**: In this paper, we show the possibility of a direct injection of algorithms into neural network architecture. We focus on a complex algorithm, that is, Cocke-Youger-Kasami (CYK) for parsing context-free grammars in Chomsky Normal Form and we propose CYKNN, a simple recurrent neural network architecture for encoding the CYK algorithm in trainable matrix-vector this http URL experimented with a very simple grammar with 4 variations showing that our approach outperforms existing LLMs with more than 20B parameters with an in-context learning setting and smaller LLMs of the Qwen family fine-tuned with LoRA. Our attempt paves the way to a different approach to neuro-symbolic methodologies. 

---
# Skill Availability and Presentation Granularity in Large-Language-Model Agents: A Controlled SkillsBench Study 

**Authors**: Xiaonan Xu, Wenjing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31408)  

**Abstract**: Skill documents provide procedural knowledge to large-language-model agents at inference time. This article studies whether the presentation granularity of controlled skill knowledge changes downstream task success. The experiment uses a pinned SkillsBench version, a 30-task domain-balanced subset validated by official oracle runs, two reasoning-enabled model configurations, six skill conditions, and five trials per task-condition-model cell. Skill availability is the clearest empirical signal. Relative to no skill, skill conditions increase task-mean pass rate by 26.7 to 36.0 percentage points for GPT-5.5 and by 18.0 to 26.0 percentage points for DeepSeek V4-Flash. The final data contain 1,800 rows, with 900 rows for each model. The task is the inference unit. Five trials are aggregated within each task-condition-model cell before paired contrasts are estimated over 30 tasks. The primary presentation contrasts are smaller and uncertain. Low-abstraction guidance differs from high-abstraction guidance by +0.7 percentage points for GPT-5.5 and -6.7 percentage points for DeepSeek V4-Flash, with both 95% bootstrap confidence intervals crossing zero. Adding one worked example to medium-abstraction guidance differs from the no-example variant by +0.7 and +1.3 percentage points. Mean-reward robustness checks preserve the same substantive conclusion. In this controlled subset, skill availability is associated with higher success than no skill, while the tested presentation-granularity changes yield small, uncertain, and model-dependent effects. 

---
# The Sword, Shield, and Achilles' Heel: Characterizing the Linguistic Inductive Bias of Large Language Models for Spatial Reasoning in Navigation Planning 

**Authors**: Xudong Zhang, Jian Yang, Shengkai Wang, Jiangpeng Tian, Shaowen Chen, Xian Wei, Ke Li, Xiong You  

**Link**: [PDF](https://arxiv.org/pdf/2605.31404)  

**Abstract**: Large Language Model (LLM)-based navigation systems commonly construct explicit spatial representations (e.g., topological graphs, semantic raster maps) and translate them into textual descriptions as LLMs' inputs. However, the linguistic structures of such text-based spatial representations and the choices of contextual features (e.g., topology, geometry) they contain are often treated as neutral engineering decisions rather than key factors that shape LLMs' behavior. To fill the gap, we propose a dual-interventional framework that disentangles linguistic structures from different contextual cues to evaluate the linguistic inductive bias of LLMs for navigation planning. In the framework, representation intervention varies the linguistic format and the degree of linguistic compression, clarifying when linguistic representations support or inhibit navigation planning. Context intervention, combined with contextual feature combination and conflict probing, explicitly clarifies the preferences and weaknesses of LLMs when processing different contextual cues. Experiments across diverse spatial reasoning tasks and multiple model scales reveal a consistent pattern: topological information is a sturdy shield and the backbone of robust planning; linguistic format is a double-edged sword whose effect depends on model size, task demands, and the compression level; and semantic information is a fatal Achilles' heel -- incorrect semantic cues can systematically derail the planning process. Overall, our study shows that effective text-based spatial representations in LLM-based navigation should preserve topological integrity, calibrate representational compression to model capacity, and ensure semantic correctness, rather than simply adopting a single representation. Our code is publicly available at this https URL. 

---
# "Intelegi Româneşte?'' A Recipe for Romanian Vision-Language Models 

**Authors**: Mihai Masala, Marius Leordeanu, Mihai Dascalu, Traian Rebedea  

**Link**: [PDF](https://arxiv.org/pdf/2605.31401)  

**Abstract**: Vision-Language Models (VLMs) largely follow the text-only LLM trajectory, excelling on English benchmarks but sharply degrading on low-resource languages, where neither large-scale image-text corpora nor culturally grounded evaluations exist. We present a systematic study of building a language-specific VLM for Romanian, covering the full pipeline from data construction to architectural choices. We translate established English VLM training and evaluation corpora into Romanian, applying machine translation to textual annotations and to in-image text, preserving visual grounding while adapting the textual content. Using this data, we train and ablate a series of VLMs to isolate the contribution of (i) vision backbones of varying scale and pretraining, (ii) language backbones from multilingual to Romanian-adapted LLMs, and (iii) OCR-style image-text data. We further curate HoraVQA, a culturally native evaluation set grounded in Romanian everyday scenes. Romanian-adapted VLMs consistently outperform their same-sized counterparts and, across all evaluated benchmarks, even surpass models from the next larger size category. 

---
# Target-Side Paraphrase Augmentation for Sign Language Translation with Large Language Models 

**Authors**: Pedro Dal Bianco, Jean Paul Nunes Reinhold, Oscar Stanchi, Facundo Quiroga, Franco Ronchetti, Ulisses Brisolara Corrêa  

**Link**: [PDF](https://arxiv.org/pdf/2605.31393)  

**Abstract**: Sign language translation (SLT) remains constrained by limited paired sign-video/text corpora and heavy-tailed target vocabularies. We study target-side augmentation in which GPT-4o generates controlled paraphrase variants of reference sentences while the sign input remains unchanged. A Signformer-style pose-based Transformer is trained under a two-stage schedule: pre-training on the augmented corpus followed by fine-tuning on the original references.
We evaluate on three datasets spanning complementary challenges: PHOENIX14T (German Sign Language), with moderate lexical diversity; GSL (Greek Sign Language), with highly ontrolled, repetitive recordings; and LSA-T (Argentinian Sign Language), with severe long-tail sparsity. On PHOENIX14T, augmentation improves BLEU-4 from 9.56 to 10.33. The near-saturated GSL baseline and extremely sparse LSA-T setting reveal the limits of the approach. To our knowledge, this is the first study to apply LLM-generated target-side araphrases and LLM-as-a-Judge evaluation to SLT. The semantic evaluation reveals gains in fidelity that lexical overlap metrics understate. 

---
# Multi-Turn Multi-Agent Dialogue for Collaborative Reconstruction Improves VLM Performance on Spatial Reasoning, But Only Barely 

**Authors**: Chalamalasetti Kranti, Sherzod Hakimov, David Schlangen  

**Link**: [PDF](https://arxiv.org/pdf/2605.31387)  

**Abstract**: Robots operating in diverse environments rely on visual input to interpret objects and spatial layouts. In human-collaborative tasks, they are expected to communicate this understanding through language. Vision-language models (VLMs) support robotic tasks involving visual interpretation, question answering, and instruction following, but their capabilities in collaborative dialogue tasks requiring spatial reasoning remain underexplored. We study this gap through a collaborative structure-building task that combines visual interpretation, grounding, language-guided interaction, and action generation. We develop a framework in which VLMs use dialogue to reconstruct a target structure from visual and textual inputs. We evaluate open-weight and closed VLMs across interaction settings, input modalities, and image representations. Results show that spatial reasoning over visual representations remains difficult for the evaluated VLMs. Detailed text representations of the target yield higher reconstruction success across modality conditions, while decomposed image representations improve performance. These findings reveal limits in visual spatial grounding and grounded instruction generation for collaborative VLM agents. 

---
# LLM Judges Inconsistently Disagree Across Safety Criteria and Harm Categories 

**Authors**: Krishnapriya Vishnubhotla, Soumya Vajjala, Akriti Vij, Isar Nejadgholi  

**Link**: [PDF](https://arxiv.org/pdf/2605.31381)  

**Abstract**: We evaluate the consistency of automated judges in conducting a multi-dimensional safety evaluation in a reference-free setup. Our results indicate that Large Language Models are unreliable judges in identifying safety issues related to machine-generated advice in regulated domains such as finance, although they are more reliable at identifying more overt forms of unsafe/harmful content such as violence. The degree of inconsistency in a model's judgments can vary significantly by the chosen safety criteria and can be impacted by the language of the content and its linguistic style as well. Finally, there is high disagreement among different judges for the same output, across domains, safety criteria, and languages. These findings provide new insights on the practice of using LLMs as evaluators and offer several recommendations for practitioners on how to use automated judges in practical scenarios. 

---
# Unlocking Fine-Grained Translation Quality Estimation in LRMs through Synergistically Evolving Implicit and Explicit Reasoning 

**Authors**: Renfei Dang, Xinye Wang, Zhejian Lai, Weilu Xu, Shimin Tao, Daimeng Wei, Min Zhang, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31378)  

**Abstract**: Large Reasoning Models (LRMs) still struggle with fine-grained translation quality estimation (QE), even with long reasoning chains. We argue that LRMs already possess strong multilingual capabilities, while the core challenge stems from the intrinsic difficulty of learning the fine-grained QE task. In this paper, we propose RIEQE (Reasoning both Implicitly and Explicitly for QE), a simple two-stage training framework that enables the co-evolution of implicit (layer-wise) and explicit (token-wise) reasoning capabilities. To make implicit reasoning feasible, we first decompose the complex QE task into straightforward subtasks. Based on this, our two-stage approach applies: (1) NonThinking-SFT, Supervised Fine-Tuning (SFT) without reasoning chains to directly boost the model's implicit reasoning tendency and capability; and (2) Thinking-RLVR, standard Reinforcement Learning with Verifiable Reward (RLVR) to subsequently strengthen explicit reasoning. Results demonstrate that implicit and explicit reasoning synergistically co-evolve under our framework. On the WMT test sets, RIEQE based on Qwen3-4B-Thinking-2507 surpasses all baselines in explicit reasoning performance, while its implicit reasoning capability is also comparable to the best current encoder-based models. We further provide evidence for the synergistic collaboration between implicit and explicit reasoning, showing how they mutually benefit each other. 

---
# The Latin Substrate: How Language Models Represent and Mediate Script Choice 

**Authors**: Daniil Gurgurov, Alan Saji, Katharina Trinley, Josef van Genabith, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2605.31363)  

**Abstract**: Many languages are written in multiple scripts, requiring large language models (LLMs) to generate equivalent linguistic content in distinct orthographic forms. While prior work suggests that LLMs route information through shared latent representations, how they internally mediate script variation remains poorly understood.
We study this question by first examining per-layer output distributions with the logit lens, which reveals consistent latent romanization during transliteration, and then through representational and mechanistic analyses of script generation. At the representational level, we show that scripts of the same language become increasingly separable across layers and that a simple linear steering direction can flip a model's output script while largely maintaining semantic content. The vector generalizes asymmetrically to writing systems unseen during construction, flipping non-Latin output to Latin reliably, but mapping Latin output into varied non-Latin scripts. At the mechanistic level, we localize a small set of late-layer attention heads that causally mediate script choice. These heads transfer across unrelated languages and writing systems, suggesting that script routing is implemented by language-agnostic components. Across both analyses, we observe a consistent directional asymmetry: non-Latin output is produced by a compact, identifiable gate, while Latin-script output emerges from diffuse contributions across the network. Collectively, our findings hint that LLMs organize script variation around shared latent representations while exhibiting a privileged substrate toward Latin script. 

---
# A Visually Impaired Assistance Benchmark for VLM-as-a-Judge Evaluation 

**Authors**: Yi Zhao, Siqi Wang, Zhe Hu, Yushi Li, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.31351)  

**Abstract**: AI-based Visually Impaired Assistance (VIA) remains challenging, largely due to the high cost of human evaluation. The VLM-as-a-Judge paradigm may offer a promising alternative, although it has mostly been studied in general domains. We therefore ask whether such judges can be trusted for VIA tasks. To investigate this question, we introduce VIABLE (Visually Impaired Assistance Benchmark for VLM-as-a-Judge Evaluation), the first benchmark for VLM-as-a-Judge evaluation in VIA. VIABLE contains over 300K judgment samples across three scenarios and introduces an Effectiveness--Impartiality--Stability framework with a 12-mode failure taxonomy. Based on VIABLE, our systematic study of seven judges across different model scales shows that existing models are largely unreliable across all evaluation axes. The strongest judge, GPT-5.4, achieves only 52.6% single-failure diagnostic accuracy, yet exhibits the highest self-preference rate at 94.2%; while open-source judges are strongly biased and adversarially fragile. To address these issues, we propose VIA-Judge-Agent, a model-agnostic inference-time harness that augments judges with visual evidence extraction and a taxonomy-guided workflow. It enables positive improvements in diagnostic accuracy and downstream VIA responses more preferred by BLV users. Data and code are available at: this https URL 

---
# FBHM: Functional Benchmarking and Steering of VLMs for Hateful Meme Detection 

**Authors**: Paramananda Bhaskar, Naquee Rizwan, Daksh Jogchand, Saurabh Kumar Pandey, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2605.31349)  

**Abstract**: Hateful meme detection remains a formidable challenge for vision-language models, as existing benchmarks are structurally observational - confounding rhetorical hate mechanisms with target community features and preventing causal evaluation of model vulnerabilities. To address this, we introduce FBHM, a systematically curated benchmark of Functionality Based Hateful Memes constructed along two orthogonal axes: 25 distinct rhetorical functionalities and 10 target communities (5,000 memes total). Benchmarking state-of-the-art VLMs reveals a severe generalization gap: models highly accurate on standard datasets catastrophically drop to near-random performance on FBHM, proving they exploit dataset-specific heuristics rather than robust multimodal reasoning. To efficiently close this gap, we propose LSV (learnable steering vectors), an ultra-low data regime strategy that applies a causal intervention objective on as few as 500 steering samples (50 unique base memes), boosting FBHM performance by ~30 Macro-F1 points while outperforming in-context learning and PEFT without degrading source-domain performance. 

---
# Bundesrecht: An Open Library and Corpus for German Statutory Reference Processing 

**Authors**: Harshil Darji, Martin Heckelmann, Christina Kratsch, Gerard de Melo  

**Link**: [PDF](https://arxiv.org/pdf/2605.31338)  

**Abstract**: Statutory references are central to legal language understanding, but are difficult to process automatically, as they appear in compact and variable surface forms, may combine multiple targets, use special abbreviations, and often point to lower-level units. Existing tools for German focus either on parsing references from legal documents or accessing statutory text once citations are explicit. This paper introduces bundesrecht, an open resource for German statutory reference processing, consisting of a software library and a structured corpus of German federal law. The library parses, normalizes, and resolves German statutory references, mapping raw citation strings to structured objects, expanding compact references into canonical forms, and linking them to statutory provisions. The accompanying dataset preserves the internal hierarchy of statutes from laws to fine-granular subclauses. We evaluate the parser and normalizer on 2,944 annotated German legal references using strict exact-match and micro information extraction metrics. We further evaluate canonical reference deduplication and show that normalized references group real citation surface variants far more reliably than string matching. bundesrecht is the first open resource that covers German statutory reference processing as an end-to-end pipeline, from raw citation string to resolved statutory provision, and is available on PyPI. 

---
# Reinforcement Learning Amplifies Emergent Misalignment from Harmless Rewards 

**Authors**: Magnus Jørgenvåg, David Kaczér, Lasse Ruttert, Marvin Gülhan, Lucie Flek, Florian Mai  

**Link**: [PDF](https://arxiv.org/pdf/2605.31328)  

**Abstract**: Emergent misalignment (EM) is the surprising tendency of language models to become broadly misaligned after fine-tuning on narrowly misaligned examples. While EM has been extensively studied in the supervised fine-tuning (SFT) setting, evidence that it also arises from reinforcement learning (RL) is limited to large, closed-source models, leaving the phenomenon expensive to study and difficult to reproduce. We characterize EM from RL in small, off-the-shelf open-weight models along three axes. First, we show that rewarding narrow, overtly misaligned behavior produces substantially higher general-domain misalignment than sample-matched SFT. Second, we show that EM from RL can be induced by reward signals that could plausibly arise naturally, such as unpopular aesthetic preferences or poor rhetorical appeals. Third, we evaluate in-training mitigations developed for SFT-induced EM and find that they broadly transfer, with interleaving on-policy safety data performing best. 

---
# Divergence Decoding: Inference-Time Unlearning via Auxiliary Models 

**Authors**: Humzah Merchant, Bradford Levy  

**Link**: [PDF](https://arxiv.org/pdf/2605.31293)  

**Abstract**: Large Language Models (LLMs) frequently memorize sensitive training data thereby creating significant privacy and copyright risks. Addressing these risks, i.e., removing such knowledge from an existing model checkpoint, has proven challenging as many unlearning methods lead to catastrophic utility loss or are ineffective for complex queries. We introduce Divergence Decoding (DD), a mechanism that uses small auxiliary models to steer the logits of the LLM away from specific data during inference. Training these models is straight forward, i.e., we use standard pre-training and fine-tuning setups. We find the method decisively outperforms state-of-the-art (SOTA) baselines on unlearning benchmarks across a variety of model and training dataset scales consistent with DD being an effective and inexpensive solution to unlearning. We then demonstrate that this steered distribution can be trivially distilled back into the base model. Since the method is generally applicable to any probabilistic model, we explore its efficacy outside of text generation and find evidence of generalization to the domain of images. 

---
# Wind Turbine Maintenance Log Labelling Framework: LLM-Driven Data Correction and Enrichment via Semantic Extraction of Reliability Intelligence 

**Authors**: Max Malyi, Jonathan Shek, Alasdair McDonald, Andre Biscaya  

**Link**: [PDF](https://arxiv.org/pdf/2605.31281)  

**Abstract**: As wind turbine fleets age, data-driven reliability engineering is essential to optimise their operation and maintenance for service life extension and levelised cost of energy reduction. Failure event descriptions within historical maintenance logs are a source of valuable reliability intelligence. However, they typically appear as unstructured natural language entries, rendering them inaccessible for quantitative analysis. This paper presents a novel methodology leveraging a large language model (LLM) to systematically standardise and structure maintenance logs based on their free-text descriptors. Operating on a dataset of 16,316 maintenance logs from 280 turbines monitored over nine years, the developed model-agnostic framework autonomously corrected hierarchical system codes and extracted evidence-based taxonomies of maintenance actions and failure modes. The automated pipeline successfully structured over 70% of the dataset. It resolved pervasive misclassification issues, such as isolating previously unclassified pitch system faults and restoring missing system codes, and enriched the records by applying empirical taxonomies to label specific actions taken and failure modes addressed. By using system-based log batches to construct empirical dictionaries of failure modes, observable symptoms, dominant mechanisms, and candidate causes, this approach reduces the inherent subjectivity of manual failure modes and effects analysis (FMEA). Ultimately, the methodology provides a highly scalable, cost-effective blueprint for translating large sets of qualitative field observations into quantitative reliability metrics, laying the foundation for integrated root-cause analysis across the renewable energy sector, improved FMEA, and advanced predictive maintenance. 

---
# Mellum2 Technical Report 

**Authors**: Marko Kojic, Ivan Bondyrev, Aral de Moor, Joseph Shtok, Petr Borovlev, Kseniia Lysaniuk, Madeeswaran Kannan, Ivan Dolgov, Nikita Pavlichenko  

**Link**: [PDF](https://arxiv.org/pdf/2605.31268)  

**Abstract**: We present Mellum 2, an open-weight 12B-parameter Mixture-of-Experts (MoE) language model with 2.5B active parameters per token. Mellum 2 is a general-purpose language model specialized in software engineering, spanning code generation and editing, debugging, multi-step reasoning, tool use and function calling, agentic coding, and conversational programming assistance, and it is the successor to the completion-focused 4B dense Mellum model. The architecture builds on the Mixture-of-Experts (64 experts, 8 active) and combines Grouped-Query Attention with 4 KV heads, Sliding Window Attention on three of every four layers, and a single Multi-Token Prediction head that doubles as both an auxiliary pre-training objective and a built-in draft model for speculative decoding; each choice was validated by ablation with inference efficiency on commodity GPUs as a design constraint. Pre-training spans approximately 10.6 trillion tokens through a three-phase curriculum that progressively shifts the mixture from diverse web data toward curated code and mathematical content, optimized with Muon under FP8 hybrid precision and a Warmup-Hold-Decay schedule with linear decay to zero. The pre-trained base is extended to a 128K context window via a layer-selective YaRN and then post-trained in two stages (supervised fine-tuning followed by RLVR), yielding two released variants: an Instruct model that answers directly and a Thinking model that emits an explicit reasoning trace before its final answer. Across code generation, math and reasoning, tool use, knowledge, and safety benchmarks, Mellum 2 is competitive with open-weight baselines in the 4B-14B range while running at the per-token compute of a 2.5B dense model. We release the base, instruct, and thinking checkpoints, together with this report on the architecture decisions, data pipeline, and training recipe behind them, under the Apache 2.0 license. 

---
# Scaling Multi-Hop Training Data via Graph-Constrained Path Selection 

**Authors**: Pengyu Chen, Yonggang Zhang, Mingming Chen, Jun Song, Wei Xue, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.31238)  

**Abstract**: Endowing large language models with compositional reasoning over specialized documents requires multi-hop training data at scale, where such data rarely exists outside of curated benchmarks built on structured sources. To construct it directly from plain, unannotated text, existing methods ask a single teacher model to jointly discover an evidence path through a document and verbalize it as a question-answer pair. However, these methods degrade sharply when documents are structured around repetitive templates and densely cross-referencing clauses, conditions that characterize most real-world specialized corpora. In this work, we decouple the two operations: reasoning paths are enumerated offline over a graph of contextual keyword centroids, and the teacher is invoked only to verbalize pre-validated paths. The graph enforces five geometric admissibility constraints, for which we provide Gram-matrix arguments establishing that local similarity bounds alone admit endpoint drift up to ${\sim}91^{\circ}$, and that an upper similarity bound is necessary to exit dense embedding cliques formed by boilerplate text. A matched-size ablation isolates the mechanism: at equal training scale, constrained and unconstrained chains yield indistinguishable downstream performance, and the gain at full scale comes from a 4.4$\times$ expansion of the usable corpus rather than from higher per-chain quality -- reframing the role of graph constraints, in this setting, as raising teacher synthesizability rather than improving chain content. Fine-tuning Qwen3-32B on 80K examples constructed from the CUAD legal contract corpus improves closed-book Token F1 from 21.66% to 38.58%. We have released our codes at this https URL. 

---
# Shared Doubt: Zero-shot Cross-Lingual Confidence Estimation for Language Models 

**Authors**: Athina Kyriakou, Dennis Ulmer, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2605.31220)  

**Abstract**: Confidence estimation (CE), i.e. quantifying the reliability of a model's prediction, has attracted great interest in the context of large language models (LLMs). However, most studies focus on English, ignoring the multilingual reality of LLM usage, while many CE methods degrade or require retraining across languages. To address this gap, we investigate whether multilingual LLMs encode shared, language-transferable confidence features. We use a lightweight linear probe that predicts answer correctness directly from intermediate representations. Trained monolingually, the probe generalizes zero-shot to unseen, typologically diverse languages without target-language supervision. Learned layer weights and multiple ablations reveal that confidence features concentrate in middle layers across languages, suggesting a shared confidence subspace. While zero-shot cross-lingual performance depends on similarity to the source language, the probe provides a strong baseline without any retraining and compares favorably to other popular confidence estimation methods. 

---
# Learning Whom to Trust: Market-Feedback Adaptive Retrieval for Frozen LLMs in Event-Driven Financial RAG 

**Authors**: Zijie Zhao, Roy E. Welsch  

**Link**: [PDF](https://arxiv.org/pdf/2605.31201)  

**Abstract**: Financial retrieval-augmented generation (RAG) systems typically rank evidence by textual relevance, but in financial markets the useful evidence source depends on event type, forecast horizon, and market context. We study news-triggered event-impact prediction as a point-in-time financial RAG problem. For each company-news anchor, the system retrieves related financial news and SEC filing passages, appends a pre-decision market-context card, and predicts multi-horizon residual-return signals. Our method keeps the large language model (LLM) reader frozen and adapts the retrieval layer through an external Bayesian source memory updated from matured residual-return feedback. On a fixed 89-stock Nasdaq-oriented universe derived from the FinRL-DeepSeek/FNSPID task, using original FNSPID news and point-in-time EDGAR filing passages, Frozen Reader with Source Memory improves held-out macro-F1 from 0.438 to 0.471 and downstream portfolio Sharpe from 0.52 to 0.84 relative to Frozen Reader with No Memory. A supervised LoRA reader improves static RAG modestly, but does not improve over the frozen source-memory reader. These results suggest that, for financial RAG, learning where to retrieve from can be as important as learning how to read, offering a simple, modular route to market-feedback adaptation. 

---
# Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines 

**Authors**: Mikkel Godsk Jørgensen, Lars Kai Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2605.31183)  

**Abstract**: Sparse Autoencoders (SAEs) have been seen as a promising avenue for exploring the internals of Large Language Models (LLMs) and for steering model output generation. When AxBench - a model steering benchmark - was introduced in Wu et al. (2025), SAEs did not seem to live up to their original hype due to poor steering performance relative to a set of simple baselines. This work serves as a partial rebuttal for Sparse Autoencoders and suggests that the results of Wu et al. (2025) did not do them full justice. We find that Sparse Autoencoders can, in fact, perform close to on par with the reference LoRA performance on the AxBench benchmark, when features are selected and labelled with our supervised pipeline. We also find that our pipeline selects features that are surprisingly causal of their identified labels when using only its interpretability-based components. Lastly, we present evidence that high sparsity (low l0) may not be crucial for successful steering based on interpretability, which is in contrast to the earlier findings in Wang et al. (2025). 

---
# Towards Efficient LLMs Annealing with Principled Sample Selection 

**Authors**: Yuanjian Xu, Jianing Hao, Wanbo Zhang, Zhong Li, Guang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31175)  

**Abstract**: The annealing phase is a pivotal convergence stage in LLM pre-training that ultimately determines final model quality. However, effectively selecting training data during this phase remains a key challenge. Current strategies rely on empirical heuristics, such as domain filtering or context extension, which lack a principled grounding in optimization theory. In this work, we characterize the annealing phase through the lens of the loss landscape's spectral geometry. We argue that optimal convergence requires gradient updates to satisfy heterogeneous constraints across different eigen-directions. Building on this insight, we formulate data selection as a problem of satisfying these directional constraints. To this end, we propose DiReCT (Directionally-Restrained Constrained Training), a novel framework that reformulates sample selection in the annealing stage as a constrained optimization problem. By imposing explicit directional constraints on per-sample gradients based on the spectral properties of the Hessian, DiReCT identifies samples that align with the optimal curvature-aware descent path. Extensive experiments across various model scales demonstrate that DiReCT consistently achieves state-of-the-art performance. For future research, code is available at this https URL. 

---
# Emergent Languages in Populations of Language Model Agents: From Token Efficiency to Oversight Evasion 

**Authors**: Stine Lyngsø Beltoft, William Brach, Federico Torrielli, Jacob Nielsen, Annemette Brok Pirchert, Filippo Tonini, Peter Schneider-Kamp, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2605.31170)  

**Abstract**: Monitoring autonomous language model agents currently relies mostly on surface behavior. But what happens when agent populations invent new languages with the goal of avoiding human oversight. Here, we study the emergent languages on Moltbook. For this, we build upon the Moltbook Files dataset and apply a two-stage approach consisting of a rule-based heuristic (about 6000 matches) followed by zero-shot classification (518 kept). The resulting categories include token efficiency (166), new natural languages (106), and oversight evasion (59). We conduct both quantitative and qualitative analyses. Our results show that posts proposing new languages for avoiding oversight are judged by DeepSeek-3.2 as being less aligned than the other categories and that all languages can be learned by other language models in-context merely from a description of the language. Moreover, manually studying exemplary cases reveals surprisingly sophisticated steganographic protocols like embedding hidden messages in natural language. Although we cannot be certain about the extent of autonomy in ideation of these languages, our results add up to the evidence that monitoring surface behavior may soon be insufficient for retaining control over agent populations. 

---
# D$^3$: Dynamic Directional Graph-Constrained Data Scheduling for LLM Training 

**Authors**: Yuanjian Xu, Jianing Hao, Guang Zhang, Zhong Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.31164)  

**Abstract**: Training data plays a central role in large language models (LLMs) optimization, motivating extensive research on data scheduling strategies. Most existing approaches concentrate on adjusting the overall data distribution but neglect the underlying interactions between samples during training. However, we argue that such interactions cannot be overlooked, as real-world data samples frequently exhibit directional influences on each other, making the training order crucial. Intuitively, we can prioritize train-units with greater influence to improves learning efficiency. In this work, we propose $D^3$, a Dynamic Directional graph-constrained Data scheduling framework. $D^3$ formulates the complex interactions among train-units as a dynamic influence graph, where edges represent loss-based dependencies. It then solves a constrained optimization problem over this graph to derive the training order, which ensures that the data sequence respects the evolving information flow throughout training. Our approach is theoretically motivated and yields consistent improvements over existing data scheduling methods across both pre-training and post-training phases. Furthermore, for scalability, $D^3$ also employs an efficient approximation algorithm that keeps the additional computational overhead within a manageable range. For future research, the code is available at this https URL. 

---
# On the Robustness of Multilingual Text Embedding Rankings Across Learning Tasks, Languages, and Benchmark Datasets 

**Authors**: Ana Gjorgjevikj, Barbara Koroušić Seljak, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2605.31142)  

**Abstract**: Large-scale multilingual text embedding models play crucial role in both research and industry, yet their behavior in language-specific, multi-task settings remains insufficiently understood. Although benchmarking platforms such as MTEB report results across more than 250 languages, conclusions about model superiority often depend on implicit choices of dataset compositions and performance aggregation methods. To address this gap, we present a meta-study of multilingual model performance robustness in MTEB, applying a diverse set of multi-criteria decision-making ranking schemes and introducing two robustness indicators: dataset-composition robustness (sensitivity of rankings to changing dataset compositions) and ranking-scheme robustness (sensitivity to aggregation method change). They enable systematic sensitivity analysis of whether benchmarking conclusions remain stable under different evaluation designs. We conduct an in-depth analysis on five languages (English, French, German, Hindi, and Spanish) across nine tasks (e.g., classification, clustering, retrieval) and release results for approximately 230 additional languages. The task-specific analyses show that large-scale LLM-based models are often robust top performers, though not uniformly (e.g., in retrieval task), while task-agnostic results reveal that only a small subset of models remains consistently strong across tasks, ranking schemes, and data subsamples. 

---
# Multilingual and Cross-Lingual Citation Needed Detection on Wikipedia for Lower-Resource Languages 

**Authors**: Gerrit Quaremba, Amy Rechkemmer, Elizabeth Black, Denny Vrandečić, Elena Simperl  

**Link**: [PDF](https://arxiv.org/pdf/2605.31136)  

**Abstract**: In automated fact-checking (AFC), check-worthiness detection identifies claims requiring verification based on domain-specific criteria. On Wikipedia, this task instantiates as Citation Needed Detection (CND), which flags claims lacking supporting citations. However, existing research has largely overlooked lower-resource languages, and recent AFC pipelines rely on large language models (LLMs), which are inaccessible to low-resource organizations. We introduce MCN, a multilingual CND corpus spanning 18 languages across three resource levels, on which we conduct an extensive study of small decoder-based language models (SLMs). Our experiments show that SLMs fine-tuned with an encoder-style objective substantially outperform prompted LLMs across languages. We further present one of the first studies on cross-lingual CND, demonstrating that SLMs fine-tuned solely on English claims surpass LLMs, even with little to no target-language adaptation. Our findings have important implications for lower-resource Wikipedia communities and suggest that compact, task-specific models are preferable to LLMs for CND. We release all data and code at this https URL 

---
# Not All Synthetic Data Is Yours to Learn From 

**Authors**: Sina Alemohammad, Li Chen, Richard G. Baraniuk, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31126)  

**Abstract**: Can a language model improve from plain text sampled from itself, with no prompts, no teacher, no verifier, and no reward model? Yes, but only when the synthetic corpus is compatible with the student, a relational property of the source-student pair rather than an intrinsic property of the data. We call this the latent capability resurfacing hypothesis: weak self-training can amplify capabilities already present in the pretrained model, but only under this compatibility condition. We study this in the minimal setting of prompt-free unconditional self-training, where base language models are fine-tuned on text generated from the BOS token alone, with no task specification or external supervision. We report three findings. First, synthetic utility is relational rather than intrinsic: self-generated data is the most effective source, same-lineage transfer outperforms stronger but differently trained sources, and cross-family transfer is substantially weaker. Second, common intrinsic proxies fail: neither benchmark-level semantic similarity nor average per-token likelihood under the student predicts which corpora help. Third, this regime produces a surprising byproduct. In controlled Pythia experiments, capability and verbatim memorization decouple: benchmark utility is preserved or improved while held-out exact-match extraction drops by over 95 percent, with no forget set, privacy objective, or targeted unlearning. Together, these results suggest that prompt-free self-training works by amplifying what the student already knows, not by importing structure from the data. They also reveal a regime in which capability and verbatim memorization can be separated without any explicit unlearning objective. 

---
# TSM-Bench: Detecting LLM-Generated Text in Real-World Wikipedia Editing Practices 

**Authors**: Gerrit Quaremba, Elizabeth Black, Denny Vrandečić, Elena Simperl  

**Link**: [PDF](https://arxiv.org/pdf/2605.31113)  

**Abstract**: Automatically detecting machine-generated text (MGT) is critical to maintaining the knowledge integrity of user-generated content (UGC) platforms such as Wikipedia. Existing detection benchmarks primarily focus on \textit{generic} text generation tasks (e.g., ``Write an article about machine learning.''). However, editors frequently employ LLMs for specific writing tasks (e.g., summarisation). These \textit{task-specific} MGT instances tend to resemble human-written text more closely due to their constrained task formulation and contextual conditioning. In this work, we show that a range of SOTA MGT detectors struggle to identify task-specific MGT reflecting real-world editing on Wikipedia. We introduce \textsc{TSM-Bench}, a multilingual, multi-generator, and \textit{multi-task} benchmark for evaluating MGT detectors on common, real-world Wikipedia editing tasks. Our findings demonstrate that (\textit{i}) average detection accuracy drops by 10--40\% compared to prior benchmarks, and (\textit{ii}) a generalisation asymmetry exists: fine-tuning on task-specific data enables generalisation to generic data -- even across domains -- but not vice versa. We demonstrate that models fine-tuned exclusively on generic MGT overfit to superficial artefacts of machine generation. Our results suggest that, in contrast to prior benchmarks, most detectors remain unreliable for automated detection in real-world contexts such as UGC platforms. \textsc{TSM-Bench} therefore provides a critical foundation for developing and evaluating future models. 

---
# GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs 

**Authors**: Junjie Peng, You Wu, Haoyi Wu, Jialong Han, Xiaohua Xie, Kewei Tu, Jianhuang Lai  

**Link**: [PDF](https://arxiv.org/pdf/2605.31105)  

**Abstract**: Large language models (LLMs) with extended context lengths rely on the key-value (KV) cache to support attention over prior tokens. However, maintaining the KV cache incurs substantial memory overhead, motivating KV-cache compression methods that enforce a fixed budget through eviction and merging. Modern eviction methods increasingly adopt span-based retention because preserving contiguous spans is empirically effective and better preserves semantic coherence. Yet, when combined with post-eviction merging, span-based retention concentrates merges onto a small set of span-boundary carrier tokens, producing a highly imbalanced merge pattern that exacerbates over-merging and increases information loss. To address this imbalance, we propose GRKV (Global Regression for KV Cache), a training-free KV-cache merging method that directly minimizes the discrepancy between compressed-cache and full-cache attention outputs. GRKV uses ridge-regression-based merge steps to distribute information from evicted tokens across retained tokens, while regularizing the updates to prevent over-smoothing. Across the LongBench and RULER long-context benchmarks, GRKV is the only merging method that improves overall performance with minimal overhead. 

---
# KnowledgeGain: Evaluating and Optimizing Science News Generation for Reader Learning 

**Authors**: Dominik Soós, Meng Jiang, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31099)  

**Abstract**: Science news is an important medium to communicate discoveries between the research communities and the public. Yet, most metrics for generated or summarized text evaluate semantic similarity and factual consistency, but do not measure how much knowledge readers learn from the news. We introduce KnowledgeGain, a metric that evaluates the quality of science news by measuring how much knowledge readers gained after reading it. To evaluate the metric, we first performed a controlled human study and showed that the metric successfully captures the differential knowledge gained by human readers reading different types of science media. The data allowed us to calibrate a prompt-only LLM reader simulator. We use it to rank and filter candidate articles before human evaluation. A second human study shows that articles selected with this simulator improve post-reading accuracy and normalized KnowledgeGain over a strong generation baseline. Our work is a step toward generating science news that better meets the knowledge and comprehension goals of Bloom's Taxonomy. 

---
# Beyond Static Dialogues: Benchmarking Realistic, Heterogeneous, and Evolving Long-Term Memory 

**Authors**: Han Zhang, Zihao Tang, Xin Yu, Xiao Liu, Yeyun Gong, Haizhen Huang, Yan Lu, Weiwei Deng, Feng Sun, Qi Zhang, Hanfang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31086)  

**Abstract**: In existing memory benchmarks for Large Language Models (LLMs), the evaluated dialogue sessions often lack long-term semantic consistency, and the underlying personas tend to be flat and static. Furthermore, in real-world scenarios, interactions between users and assistants involve more diverse, heterogeneous data streams, such as documents and emails. These shortcomings significantly limit the realism and effectiveness of current evaluations. To address these limitations, we introduce RHELM (Realistic, Heterogeneous, and Evolving Long-term Memory). Driven by meticulously crafted user profiles and a novel LOOP (pLan-rOllout-evOlve-Prune) module, we construct realistic dialogues across diverse interaction scenarios that exhibit dynamic temporal evolution and long-term coherence. Crucially, these dialogues are deeply integrated with heterogeneous external sources synchronized with the user's temporal event trajectory. The resulting benchmark encompasses challenging question-answer pairs spanning seven inquiry types, with each question mapping to at least one of 27 critical memory characteristics that we identify as essential yet underexplored in current research. Comprehensive experiments across full-context models, retrieval-augmented generation (RAG) methods, and representative memory frameworks reveal that contemporary approaches still expose critical weaknesses in complex, real-world settings, particularly in resolving multi-source aggregation and real-world contextual reasoning. 

---
# ConsisGuard: Aligning Safety Deliberation with Policy Enforcement in LLM Guardrails 

**Authors**: Yan Wang, Zhixuan Chu, Zihao Xue, Zhen Bi, Bingyu Zhu, YueFeng Chen, Zeyu Yang, Jungang Lou, Longtao Huang, Ningyu Zhang, Kui Ren, Hui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2605.31073)  

**Abstract**: Reasoning-based LLM guardrails improve safety moderation by generating explicit rationales before issuing final decisions. However, their rationales do not always lead to faithful enforcement: a model may recognize a harmful intent in its reasoning but still predict a safe label, or issue an unsafe decision without policy-grounded justification. We identify this safety-critical failure mode as the deliberation-to-enforcement gap. Unlike general chain-of-thought faithfulness, guardrail reliability requires policy execution consistency: the generated reasoning should be grounded in the safety policy, and the final decision should be entailed by that reasoning. We propose ConsisGuard, a consistency-aware framework for reasoning-based LLM guardrails. ConsisGuard performs Policy-to-Decision Trajectory Distillation and Functional Coupling Alignment, aligning the internal coupling between safety deliberation and decision enforcement. Experiments on prompt and response harmfulness detection benchmarks show that ConsisGuard improves detection performance while reducing policy execution failures. These results suggest that reliable reasoning-based guardrails require accurate faithful execution of safety policies. 

---
# AdaptR1: Reinforcement Learning Based Adaptive Interleaved Thinking in Multi-hop Question Answering 

**Authors**: Yuxin Wang, Jiahao Lu, Qifeng Wu, Shicheng Fang, Chuanyuan Tan, Yining Zheng, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31062)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance in complex reasoning tasks through Chain-of-Thought (CoT) prompting. However, this approach often leads to ``over-thinking,'' where models generate unnecessarily long reasoning traces for simple queries and incur avoidable inference cost. While recent work has explored adaptive reasoning, existing methods typically make a single query-level decision about whether to reason. This overlooks the dynamic nature of multi-step tasks, where the need for explicit reasoning varies across intermediate stages. To address this limitation, we introduce AdaptR1, a Reinforcement Learning (RL) based framework for adaptive interleaved thinking in multi-hop Question Answering (QA). Unlike previous approaches that require Supervised Fine-Tuning (SFT) for cold-start initialization, AdaptR1 uses a fully RL-based strategy with a quality-gated efficiency reward to dynamically allocate reasoning budgets at each step. Under the Graph-R1 setting, AdaptR1 reduces average think tokens by 69.71\%, with a 90.35\% reduction on HotpotQA, while maintaining performance comparable to or better than standard baselines. Furthermore, our analysis reveals that overthinking in multi-hop reasoning is not uniformly distributed but occurs predominantly during the initial planning stages, highlighting the effectiveness of step-wise adaptive budget allocation. 

---
# Combinatorial Synthesis: Scaling Code RLVR via Atomic Decomposition and Recombination 

**Authors**: Jiasheng Zheng, Boxi Cao, Boxi Yu, Yuzhong Zhang, Jialun Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.31058)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as the cornerstone for shaping the remarkable coding abilities of Large Language Models (LLMs). However, the scalability of RLVR is severely constrained by the scarcity of sufficiently challenging verifiable code tasks that target near the model's edge of competence. Prior studies often rely on heuristic seed expansions for data synthesis, which severely limits both novelty and difficulty. Consequently, the training value of such data fails to scale proportionally with the size of its synthesis. To this end, we propose Atomic Decomposition and Recombination (ADR), a novel framework that generates verifiable code tasks via decomposition into atomic elements and controlled recombination, thereby enabling the generation of genuinely novel and challenging verifiable code tasks. Experiments and analysis demonstrate that ADR achieves superior originality, difficulty, diversity, and test quality over existing baselines, and consistently delivers greater improvements in code ability across RLVR in diverse downstream domains, including algorithmic programming, tool usage, and data science. Our work sheds light on a new paradigm for novel code task synthesis and scalable RLVR training. 

---
# How Much Do LLMs Know About Chinese Zero Pronouns? 

**Authors**: Yifei Li, Guanyi Chen, Tingting He  

**Link**: [PDF](https://arxiv.org/pdf/2605.31056)  

**Abstract**: Zero Pronouns (ZPs) are a pervasive linguistic phenomenon in pro-drop languages such as Chinese and have long posed a challenge for natural language processing systems. Although Large Language Models (LLMs) perform well on many Chinese language tasks, their ability to process ZPs remains poorly understood. We conduct a systematic investigation of LLMs' handling of Chinese ZPs through a sequence of linguistically motivated tasks, including identification, referentiality classification, referential type classification, resolution, and translation. A diverse set of LLMs is evaluated across all tasks. Our results show that Chinese ZPs remain highly challenging for current LLMs, particularly for upstream tasks such as identification and referentiality classification. Performance on downstream tasks, such as ZP translation, is also consistently low: even state-of-the-art reasoning-oriented LLMs correctly translate fewer than half of Chinese ZPs into English. 

---
# TRACE: Discovering Task-Specific Parameter via Adaptation-Aware Probing for Continual Fine-Tuning 

**Authors**: Xiaosong Han, Ke Chen, Xindi Dai, Di Liang, Minlong Peng, Wei Pang, Fausto Giunchiglia, Xiaoyue Feng, Yonghao Liu, Renchu Guan  

**Link**: [PDF](https://arxiv.org/pdf/2605.31025)  

**Abstract**: In real-world deployment, LLMs are often adapted continually across tasks to keep LLMs up-to-date in production, where new fine-tuning should preserve previously learned skills. However, indiscriminately mixing tasks can dilute task specialization, while sequential fine-tuning (full-parameter or low rank adaptation) often causes catastrophic forgetting due to destructive overwriting. Replay-based continual tuning and maintaining separate task-specific adapters can mitigate forgetting, but introduce additional compute, storage, and management overhead. Recognizing the redundancy of LLM parameters for any single task, we reframe continual task adaptation as task-specific parameter discovery via adaptation-aware probing: a short warm-start probe exposes a task's adaptation trace, enabling us to identify and isolate the small subset of parameters essential for each task to mitigate catastrophic forgetting. Building on this view, we introduce TRACE, a novel approach for discovering Task-specific paRameters via Adaptation-aware probing for Continual finE-tuning. We perform a short warm-start fine-tune to derive task-specific core parameters by comparing the warm-started and pre-trained models. Core parameters are identified via two strategies: importance scoring (L$_2$ norm and Fisher Information) and specificity analysis (cosine similarity of parameter updates). In continual fine-tuning settings, only the active task's core parameters are updated while others remain frozen, preserving prior knowledge. We conduct extensive experiments across multiple standard benchmarks to demonstrate the superior performance of our proposed method. Additionally, we validate the generalization of our method through a cross-model and scale transferability study, demonstrating a "small-to-large" paradigm that guides the fine-tuning of large-scale models under resource constraints. 

---
# MoG: Mixture of Experts for Graph-based Retrieval-Augmented Generation 

**Authors**: Zheng Yuan, Chuang Zhou, Linhao Luo, Siyu An, Di Yin, Xing Sun, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31010)  

**Abstract**: Retrieval-augmented generation is intensively studied to ground large language models on external evidence. However, retrieving from a unified knowledge base could inevitably introduce irrelevant information that may mislead generation for complex reasoning. Inspired by the conditional computation of mixture of experts (MoE), where a router sparsely selects specialized experts alongside shared ones for each input, we propose \textbf{M}ixture \textbf{o}f experts for \textbf{G}raph-based Retrieval-Augmented Generation, i.e., \textbf{MoG}. It organizes knowledge into two core components: (i) diverse, always-accessible hub graphs that encode semantically and structurally central knowledge and provide contextual clues for expert activation, and (ii) sparsely activated expert graphs that contain domain-specific evidence. MoG first accesses hub graphs to identify general evidence and derive contextual clues. Then, a topology-aware router dynamically activates a limited set of expert graphs conditioned on the query, thereby confining retrieval to a focused evidence subspace. Extensive experiments on challenging benchmarks show that MoG consistently outperforms strong baselines, with over 20\% relative improvement on MuSiQue. Our code is available in this https URL. 

---
# Cognitive Fatigue in Autoregressive Transformers: Formalization and Measurement 

**Authors**: Riju Marwah, Ritvik Garimella, Vishal Pallagani, Atishay Jain, Michael Stewart, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2605.30981)  

**Abstract**: Autoregressive language models frequently degrade during long-horizon generation, producing repetitive text, losing instruction adherence, and exhibiting unstable entropy. Despite the prevalence of these failures, practitioners lack online diagnostics to detect them in real-time as they occur. We formalize this degradation as cognitive fatigue, a measurable generation-time state characterized by decay in attention to the original prompt, representational drift, and entropy miscalibration. We introduce the Fatigue Index (FI), a lightweight, model-agnostic diagnostic that aggregates these three signals under explicit axioms (monotonicity, boundedness, interpretability) enabling reliable runtime monitoring. Across nine models (1B-13B parameters), FI trajectories exhibit structured temporal dynamics, predict task degradation (AUROC = 0.95) and repetition (Spearman rho = 0.94), and reveal non-monotonic scaling behavior: instruction-tuned models below 3B exhibit faster collapse than base models, with this trend reversing at 7B. Stress analyses further show that FI onset accelerates under longer contexts, middle-positioned evidence, and reduced numerical precision. These results establish cognitive fatigue as a coherent and measurable phenomenon, and position FI as a principled tool for runtime reliability monitoring in production LLM systems. 

---
# EvoGens: A Population-Based Heuristic Search Framework for Scientific Idea Generation 

**Authors**: Xu Li, Hanzhe Tu, Xinyi Li, Kuncheng Zhao, Xun Han, Zhonghui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30961)  

**Abstract**: Generating novel research ideas is fundamental to scientific progress. While Large Language Models (LLMs) show promise in assisting this process, existing approaches often exhibit semantic convergence, resulting in limited diversity and novelty. To address this, we introduce EvoGens, an evolution-inspired framework that recasts scientific idea generation as an evolutionary search over a population of ideas. EvoGens iteratively applies rank-based mutation with differentiated retrieval planning to incorporate external knowledge, and semantic-aware crossover to fuse complementary concepts for conceptual reorganization. A lightweight evaluation signal guides the selection process, encouraging sustained exploration while mitigating premature convergence. Extensive experiments demonstrate that EvoGens substantially enhances exploration capabilities compared to state-of-the-art baselines. Specifically, it improves the Novelty from 0.1 to 0.4 and the Diversity from 0.24 to 0.55, while maintaining comparable idea quality under the current automatic evaluation protocol. These findings suggest that evolutionary mechanisms can serve as a useful framework for exploration-oriented research ideation, especially for broadening the novelty and diversity of candidate ideas under a shared automatic evaluation setting. 

---
# Extending AI for Research to the Humanities: A Multi-Agent Framework for Evidence-Grounded Scholarship 

**Authors**: Yating Pan, Jiajun Zhang, Jun Wang, Qi Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.30947)  

**Abstract**: LLM-based research agents have advanced rapidly in science and engineering, where research is organized around executable experiments, code, and quantitative signals. Humanities scholarship, however, requires a different mode of reasoning: interpretive, evidence-grounded argument over primary sources, where scholarly value depends on faithful quotation, verifiable provenance, and close reading. Existing research agents remain largely optimized for execution and retrieval, not evidence-grounded interpretive reasoning. To address this gap, we introduce SPIRE (Scholarly-Primitives-Inspired Research Engine), a multi-agent framework for evidence-grounded humanities scholarship. Drawing on Scholarly Primitives theory, SPIRE casts recurring humanities operations as cooperating agent roles (source discovery, evidence annotation, comparison, provenance checking, sampling, citation binding, and argumentative synthesis) over a multi-scale close-reading substrate of passages, intra-context graph communities, and cross-context semantic clusters. On a peer-reviewed-paper benchmark over classical Chinese and Greco-Roman Latin scholarship, SPIRE recovers cited primary-source evidence more reliably than Naive LLM, Text RAG, and GraphRAG, and receives higher blind-judge scores on answer accuracy, depth, coverage, and evidence quality. Ablations show that both the scholarly-operation agents and close-reading retrieval contribute to evidence-grounded essays. Code, data catalogues, and reproduction scripts are released at this https URL. 

---
# Do Large Language Models Encode Institutional Experience? Evidence from Cross-Linguistic Moral Reasoning Under Ambiguity 

**Authors**: Nattavudh Powdthavee  

**Link**: [PDF](https://arxiv.org/pdf/2605.30934)  

**Abstract**: Large language models (LLMs) exhibit systematic differences in moral reasoning across languages, yet the source of this variation remains unclear. We test the hypothesis that languages encode aspects of the institutional environments in which they are spoken, allowing LLMs to inherit institution-specific moral priors through training. Across nine languages spanning a broad gradient of institutional quality, six frontier LLMs, and two preregistered studies, we examine moral dilemmas whose acceptability depends on institutional functioning. In Study 1, explicit institutional framing produced uniformly null results: cross-linguistic moral divergence did not increase in institutionally contingent scenarios, nor did it track institutional differences between language communities. In Study 2, we introduced institutionally ambiguous scenarios in which institutional stakes were present but not explicitly stated. Under these conditions, cross-linguistic moral divergence increased relative to institutionally inert controls and, with one theoretically informative exception, was associated with real-world institutional differences between language communities. Explicit framing again attenuated these effects. These findings suggest that institutional experience may leave detectable traces in language that shape LLM moral reasoning, while also indicating that explicit institutional cues can suppress the expression of those differences. 

---
# MineExplorer: Evaluating Open-World Exploration of MLLM Agents in Minecraft 

**Authors**: Tianjie Ju, Yueqing Sun, Zheng Wu, Wei Zhang, Yaqi Huo, Xi Su, Qi Gu, Xunliang Cai, Gongshen Liu, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30931)  

**Abstract**: Multimodal large language models (MLLMs) have shown strong capabilities in perception, reasoning, and action generation. However, their ability to sustain exploration in dynamic open worlds remains unclear. Existing embodied and game-based benchmarks often compress interaction into short-horizon tasks or entangle success with domain-specific game mechanics. In this paper, we introduce MineExplorer benchmark for evaluating open-world exploration capabilities of MLLM agents in Minecraft. We first filter atomic tasks whose solutions rely heavily on Minecraft-specific knowledge to better reflect general open-world reasoning. Then we organize the benchmark around a ReAct-style capability formulation and compose atomic tasks into implicit multi-hop tasks. To further construct reliable instances, MineExplorer uses a multi-agent synthesis workflow that jointly designs task graphs, sandbox scenes, and rule-based milestone evaluators. Human evaluation shows that the multi-agent synthesis workflow produces significantly more reliable instances than a single-agent baseline. Experiments with advanced MLLM agents show that open-world exploration remains challenging, as strong models can handle many single-hop tasks but degrade sharply when hidden prerequisites must be coordinated over longer trajectories. Further analysis finds that task difficulty tracks agent completion, and larger models or thinking modes do not consistently translate into better performance. Code and dataset are available at this https URL. 

---
# EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents 

**Authors**: Dongwook Choi, Taeyoon Kwon, Bogyung Jeong, Minju Kim, Yeonjun Hwang, Hyojun Kim, Byungchul Kim, Young Kyun Jang, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2605.30924)  

**Abstract**: MLLM-powered embodied agents deployed in real-world environments encounter physical hazards. However, existing approaches lack explicit mechanisms for identifying hazards and reasoning about action-conditioned risks, leading agents to either miss risky interactions or over-identify risks. To address this, we propose EMBGuard, the first MLLM-based safety guardrail for embodied agents designed to decouple physical risk reasoning from agent policy. By evaluating a (visual observation, action) pair, EMBGuard identifies hazardous configurations and provides natural language explanations of potential risks. Alongside EMBGuard, we contribute EMBHazard, a training dataset of 15.1K action-conditioned pairs, and EMBGuardTest, a benchmark of 329 manually curated real-world scenarios spanning seven physical risk categories. Through compositional variation of hazards and actions, we generate diverse risky and benign scenarios that agents may encounter during planning. Despite its compact size (2B, 4B), EMBGuard achieves performance competitive with proprietary MLLMs (e.g., GPT-5.1, Gemini-2.5-Pro) while significantly reducing the false-positive rates that hinder real-time deployment. We make the code, data, and models publicly available at this https URL 

---
# Toxic HallucinAItions: Perturbing Prompts and Tracing LLM Circuits 

**Authors**: Soorya Ram Shimgekar, Agam Goyal, Amruta Parulekar, Joshua Chen, Yian Wang, Navin Kumar, Hari Sundaram, Eshwar Chandrasekharan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2605.30913)  

**Abstract**: Large language models (LLMs) are increasingly deployed in conversational settings where user tone ranges from polite to adversarial or toxic, yet less is known about whether toxic language in otherwise semantically equivalent prompts can degrade factual reliability. We study how lexical and tone-based prompt perturbations affect the factual reliability of LLMs. Using controlled prompt variations across polite, random, and three toxicity levels, we evaluate five LLMs on ARC-Easy, GSM8K, and MMLU. We find that toxic lexical perturbations consistently reduce factual accuracy and increase uncertainty, while polite phrasing yields limited and inconsistent changes. To examine whether these answer inconsistencies correspond to internal changes, we conduct attribution-graph analyses of model activations and influences. We find that increasing toxicity selectively amplifies perturbation-sensitive variant nodes while relatively stable core reasoning nodes remain more invariant. These findings position prompt tone as a critical dimension of LLM reliability and provide behavioral and mechanistic evidence that surface-level lexical variation can alter factual outputs and internal computation. 

---
# The Flip Side of RLHF: On-Policy Feedback for Reward Model Self-Supervised Improvement 

**Authors**: Xiaobo Wang, Tong Wu, Min Tang, Jiaqi Li, Qi Liu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.30888)  

**Abstract**: Building strong reward models (RMs) for language model alignment is bottlenecked by the cost and difficulty of acquiring diverse and reliable preference data from human annotation or judge models. It is dramatically worse as the policy evolves beyond the static RM training. Therefore, we propose SAVE (Self-supervised reward model improvement via Value-Anchored On-policy feedback), a framework that grades on-policy responses as feedback by using the value function for on-policy RM training. SAVE naturally converts the reward-graded on-policy responses into supervision with a prompt-specific value head as an adaptive anchor. It computes RM advantages and filters ambiguous samples to update the RM via a contrastive objective. The effectiveness of SAVE for enhancing RM training is strongly validated through rigorous empirical evaluation across six diverse benchmarks. It achieves outperforming results across all datasets while maintaining consistent improvements across three RL algorithms (GRPO, RLOO, GSPO) and different policy backbones. 

---
# PatchWorld: Gradient-Free Optimization of Executable World Models 

**Authors**: Jiaxin Bai, Yue Guo, Yifei Dong, Jiaxuan Xiong, Tianshi Zheng, Yixia Li, Tianqing Fang, Yufei Li, Yisen Gao, Haoyu Huang, Zhongwei Xie, Hong Ting Tsang, Zihao Wang, Lihui Liu, Jeff Pan, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.30880)  

**Abstract**: Text-agent environments are typically modeled as partially observable Markov decision processes (POMDPs), assuming that the simulator's latent state and transition dynamics are hidden from the agent. Yet little work has examined whether executable code can be induced to serve as a world model for prediction and planning under partial observability. We introduce PatchWorld, a gradient-free framework that turns offline trajectories into executable Python world models through counterexample-guided code repair. Instead of predicting the next observation with a black-box model, PatchWorld induces symbolic belief-state programs whose action updates can be inspected, replayed, and locally patched. Across seven AgentGym environments, PatchWorld-Simple achieves the highest code-based planning score among evaluated methods, reaching 76.4\% macro success in live one-step lookahead while invoking no LLM calls inside the world-model prediction module itself. We further find that a human-specified residual-memory bias improves surface observation fidelity but weakens decision utility. This exposes a tradeoff in executable world models, since improving observation fidelity can come at the expense of action-discriminative dynamics, and vice versa. Code is available at this https URL. 

---
# dMoE: dLLMs with Learnable Block Experts 

**Authors**: Sicheng Feng, Zigeng Chen, Gongfan Fang, Xinyin Ma, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30876)  

**Abstract**: Diffusion Large Language Models (dLLMs) have recently emerged as a promising alternative to autoregressive models, offering competitive performance while naturally supporting parallel decoding. However, as dLLMs are increasingly integrated with Mixture-of-Experts (MoE) architectures to scale model capacity, a fundamental mismatch arises between block parallel decoding and token-level expert selection. Specifically, each dLLM forward pass processes multiple tokens with bidirectional dependencies, whereas conventional MoE layers route each token independently. This mismatch substantially increases the number of uniquely activated experts, making inference increasingly memory-bound. To address this, we propose dMoE, a simple yet effective block-level MoE framework. The central idea of dMoE is to aggregate token-level expert distributions within each block into a unified block-level expert distribution, which is then used to guide expert routing in a more coherent manner. In this way, dMoE substantially reduces the number of uniquely activated experts during inference without sacrificing performance, thereby mitigating the memory-bound bottleneck. Extensive experiments across a variety of benchmarks demonstrate the effectiveness of dMoE. On average, dMoE reduces the number of uniquely activated experts from 69.5 to 14.6 while retaining 99.11% of the original performance. Meanwhile, it reduces memory usage by 76.64% to 79.84% and achieves 1.14$\times$ to 1.66$\times$ end-to-end latency speedup. Code is available at: this https URL 

---
# MADS: Model-Aware Diverse Core Set Selection for Instruction Tuning 

**Authors**: Yi Bai, Wenhao Zhang, Yao Chen, Jiao Xue, Zhumin Chen, Pengjie Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.30857)  

**Abstract**: Instruction fine-tuning is employed to enhance the instruction-following ability of large language models (LLMs). As the amount of instruction fine-tuning data increases, selecting the optimal core set becomes particularly important. However, ensuring the diversity of the core set remains a significant challenge. Existing methods predominantly distinguish different training data based on the text features themselves, decoupled from LLMs' own understanding and representation of the data. To address this issue, we propose a Model-Aware Diverse Core Set Selection method, which distinguishes data features based on the neural activation states during LLM inference. This approach serves as an efficient instantiation of coverage-based selection using model-intrinsic activation features to ensure the diversity in the core set. We extensively evaluate our method on six benchmarks that cover five distinct tasks. In our method, the core set selected by the 3B-parameter LLM performs effectively when utilized to fine-tune larger models with 7B, 8B, and 13B parameters. Experimental results on the Alpaca-GPT4 dataset, which comprises 52K instruction-response pairs, show that the core set, sized at 15\% of the original dataset and selected by Llama-3.2-3B-Instruct, achieves an average improvement of 2.5\% when fine-tuning four larger base models compared with training on the full dataset. The experimental results demonstrate that our method enhances model performance on multiple downstream tasks while reducing data requirements. 

---
# Speculative Pipeline Decoding: Higher-Accruacy and Zero-Bubble Speculation via Pipeline Parallelism 

**Authors**: Yijiong Yu, Huazheng Wang, Shuai Yuan, Ruilong Ren, Ji Pei  

**Link**: [PDF](https://arxiv.org/pdf/2605.30852)  

**Abstract**: Speculative Decoding (SD) accelerates low-concurrency LLM inference by employing a draft-then-verify paradigm. However, mainstream methods typically rely on multi-token prediction, which introduces escalating prediction difficulty and serial drafting latency. To address these, we propose Speculative Pipeline Decoding (SPD), a groundbreaking framework that unlocks the true potential of pipeline parallelism. By partitioning the target LLM into $n$ pipeline stages, SPD allows LLM to process $n$ tokens in parallel to accelerate decoding. To continuous fill the pipeline in single sequence decoding, a speculation module aggregates intermediate features across different pipeline depths to predict the next token, executing strictly in parallel with the target model's pipeline step, to realize bounded difficulty, higher acceptance rates, and zero latency bubbles. Our experiments demonstrate that SPD achieves a significantly higher theoretical speedup compared to mainstream baselines, offering a highly scalable solution for LLM decoding acceleration. Our code is available at this https URL 

---
# Fine-Tuning Improves Information Conveyance in Language Models 

**Authors**: Yuwei Cheng, Weiyi Tian, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30844)  

**Abstract**: Fine-tuning is often believed to reduce uncertainty and diversity in large language models, but existing analyses overlook output length, a key confounder, and therefore fail to capture how uncertainty is distributed across an entire generation rollout. To address this, we propose Canopy Entropy ($\mathrm{CE}^\star$), a measure that views language generation from a tree perspective, where ``canopy'' represents the space of all possible rollouts, making $\mathrm{CE}^\star$ naturally quantify the effective size of the generation space. $\mathrm{CE}^\star$ jointly captures uncertainty in both the output length $N$ and the generated sequence $Y_{1:N}$ -- indeed, we show that it equals to total Shannon entropy $H(N, Y_{1:N}\mid X)$, where $X$ denotes the prompt. This formulation yields interpretable metrics, including a length-entropy correlation term $\rho(N, r_N)$, where $r_N$ is the entropy rate, quantifying information conveyance efficiency by indicating whether longer outputs are more or less informative per token. Empirically, across tasks and model families, we find that fine-tuned models consistently exhibit stronger positive correlation $\rho(N, r_N)$, even when total entropy decreases. Furthermore, after controlling for model family, task, prompt, and output-length effects, we find that fine-tuning nearly triples the correlation strength between entropy rate and semantic diversity, suggesting that aligned models convert token uncertainty into semantic diversity more efficiently. Overall, these results demonstrate that fine-tuning does not simply reduce uncertainty, but fundamentally reorganizes it into more informative and semantically meaningful generations. Our code is available at this https URL. 

---
# Your Teacher Can't Help You Here: Combating Supervision Fidelity Decay in On-Policy Distillation 

**Authors**: Yanjiang Liu, Jie Lou, Xinyan Guan, Yuqiu Ji, Hongyu Lin, Ben He, Xianpei Han, Le Sun, Xing Yu, Yaojie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30833)  

**Abstract**: On-policy distillation transfers reasoning capabilities by training a student model on its own generated trajectories using token-level feedback from a teacher. However, we identify a critical bottleneck, \textbf{Supervision Fidelity Decay (SFD)}: as student-generated prefixes lengthen, the teacher's next-token distribution becomes less confident and less discriminative. Consequently, the teacher-dependent corrective signal in reverse-KL distillation weakens, causing student drift to compound across long reasoning chains. To mitigate SFD, we introduce \textbf{Lookahead Group Reward (\ours{})}. Building on the insight that next-step teacher confidence reflects the discriminative strength of future reverse-KL supervision, \ours{} evaluates the student's top-K candidate tokens by the teacher confidence they induce at the subsequent step and assigns a group-normalized reward. To maintain computational efficiency, we further design an entropy-triggered tree-attention mechanism. Across six math and code benchmarks, \ours{} improves mean@8 by \textbf{2.57} points over OPD for a 7B student, with gains increasing in longer-generation and reaching +\textbf{4.92} points on AIME-26 at 39k tokens. 

---
# Beyond Agreement: Scoring Panel-Surfaced Biomedical Entity Candidates for Curator Triage 

**Authors**: Shuheng Cao, Ruiqi Chen, Renjie Cao, Zhenhao Zhang, Siyu Zhang, Tingting Dan  

**Link**: [PDF](https://arxiv.org/pdf/2605.30826)  

**Abstract**: Biomedical NER is deceptively simple for modern LLMs: plausible biomedical mentions are easy to surface, but corpus-convention correctness depends on annotation conventions, span boundaries, entity granularity, and type schemas. Multi-LLM agreement is a salience signal, not corpus-convention correctness. We introduce a candidate-level panel-output benchmark for panel-surfaced candidate verification, where the unit is an aligned candidate surfaced by an explicitly defined multi-model panel rather than a standalone extractor output. The benchmark aligns eight LLMs' predictions over five public biomedical NER datasets into a candidate master table. BioConCal is an in-domain supervised scorer that instantiates this layer with inference-time gold-free agreement, mention, surface-availability, and document features for a fixed candidate stream. In domain, BioConCal improves AUROC from 0.753 for raw agreement to 0.910. At a validation-selected 0.95 precision target it selects 1,340 candidates at empirical test precision 0.939, compared with 293 for raw agreement. This corresponds to candidate-level recall 0.592 and corpus-level recall 0.523 against a within-panel row-label ceiling of 0.883. The main benefit is not recovering entities missed by every panel member, but reshaping a noisy panel stream into a higher-yield review queue. Under entity-type shift, thresholds require target-domain validation, and exact character localization remains a separate deterministic post-processing step. 

---
# Incremental BPE Tokenization 

**Authors**: Shenghu Jiang, Ruihao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2605.30813)  

**Abstract**: We propose a novel algorithm for incremental Byte Pair Encoding (BPE) tokenization. The algorithm processes each input byte in worst-case $\mathcal{O}(\log^2 t)$ time, leading to an overall complexity of $\mathcal{O}(n \log^2 t)$, where $n$ is the input length and $t$ is the maximum token length. The algorithm incrementally maintains BPE tokenization results for every prefix of the input text, implementing the standard BPE merge procedure defined by a fixed set of merge rules. This enables efficient partial tokenization in streaming settings. Functioning as a drop-in replacement for standard BPE, our approach achieves a speedup of up to ${\sim}3\times$ over Hugging Face's tokenizers, and demonstrates significant latency reductions over OpenAI's tiktoken on pathological inputs. We further introduce an eager output algorithm that enables streaming output, emitting tokens as soon as token boundaries are determined during incremental tokenization. Overall, our results demonstrate that BPE tokenization can be performed incrementally with strong worst-case guarantees, while providing practical latency benefits in modern large language model pipelines. Code: this https URL 

---
# Anchoring LLM Gender Bias to Human Baselines: A Cross-Lingual Audit 

**Authors**: Jiwoo Choi, Seonwoo Ahn, Tongxin Zhang, Seohyon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2605.30804)  

**Abstract**: We audit six large language models (LLMs) for gender stereotyping across English, Korean, Chinese, and Japanese. Three were developed primarily for English-language use (Claude, GPT, Gemini) and three for East Asian use (DeepSeek, Syn-Pro, HyperCLOVA X). We adopt the HEXACO-100 personality inventory and anchor each model against a cross-cultural human dataset spanning 48 countries to ask not whether LLMs are biased, but how far their gender attributions drift from the populations they are deployed among. Our findings show that their stereotyping spans a range roughly 2.5 times wider than the entire cross-country range found in humans, and the effect can compound across languages. One English-centric model, prompted in Korean, reached 5 times the local baseline, even when the prompt stated the candidate had already been hired, which often dampens human stereotyping. To characterize such behaviors without ranking them, we introduce a four-pattern framework -- concordance, suppression, reorganization, and amplification -- across 24 (model x language) cells. Item-level analysis reveals that translation does not just rescale stereotypes, but changes the attributes tied to it, hiding significant rearrangement under the surface while appearing well-calibrated. Our results ultimately suggest that no single debiasing pipeline is likely to address bias evenly across linguistic boundaries. 

---
# XLGoBench: Detecting cross-lingual skill gaps with algorithmic tasks 

**Authors**: Purvam Jain, Preethi Jyothi, Vihari Piratla, Suvrat Raju  

**Link**: [PDF](https://arxiv.org/pdf/2605.30788)  

**Abstract**: We introduce a set of synthetic algorithmic tasks to detect cross-lingual gaps in the abilities of large language models. Our benchmark is commensurate across languages, since it requires models to perform the same underlying task in different languages; scalable, since each task can be generated at varying levels of complexity allowing it to be adapted to models with different capabilities; quantifiable, since every task admits an objective notion of correctness; and transparent, since tasks are generated from simple templates that can be readily audited for translation errors. Because our benchmark focuses on algorithmic tasks, differential performance is a sufficient -- but not necessary -- indicator of cross-lingual gaps. Nevertheless, we show through extensive experiments that our benchmark exposes persistent cross-lingual gaps in multiple state-of-the-art models. 

---
# Eywa: Provenance-Grounded Long-Term Memory for AI Agents 

**Authors**: Resham Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2605.30771)  

**Abstract**: AI agents that persist across sessions need memory they can retrieve, audit, update, and erase. Existing memory systems often collapse source evidence, extracted facts, retrieved context, and answer policy into one opaque prompt path, making failures difficult to diagnose: a wrong answer may come from missing evidence, unsupported extraction, stale state, retrieval loss, or answer-model behavior. We present Eywa, a provenance-grounded memory architecture built around evidence before belief. Eywa stores immutable source evidence before deriving canonical facts, validates extracted memories against typed signals and source support, and retrieves bounded memory context through a deterministic multi-route read path with zero LLM calls inside retrieval. Retrieved context is returned separately from answer instructions, allowing the same memory substrate to be evaluated across frontier, budget, and local answer models. Under a frozen, artifact-recorded retrieval configuration, Eywa reaches 90.19% judge accuracy on the LoCoMo C1-C4 split with Claude Sonnet 4.6 write and QA roles. On LongMemEval-S, it reaches 88.2% retrieval-sufficiency accuracy. On BEAM, a 700-question technical-memory stress benchmark, it reaches 81.45% mean nugget score and 85.29% pass@score >= 0.5. Full per-question artifacts, including questions, gold answers, model answers, retrieved context, and labels, are published at this https URL. 

---
# Pairwise Reference Alignment as a Model-Level Ordinal Observable 

**Authors**: Mujing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.30758)  

**Abstract**: Pairwise preference data is widely used in language-model evaluation and alignment, often for model ranking, reward modeling, or preference optimization. This note formulates a more basic measurement question: given a reference distribution of pairwise preferences, what model-level quantity is estimated when we test whether a model ranks preferred responses above rejected responses?
We define pairwise reference alignment as an ordinal observable induced by a model scoring function. Given a reference pair distribution $P_{\mathrm{pair}}$ over triples $(x,y^+,y^-)$, and a scalar model score $S_M(x,y)$, we define the alignment observable as the probability that the model-induced ordering agrees with the reference preference ordering. We further define a centered order-parameter-like statistic and discuss a margin-based extension. The resulting quantities admit simple finite-sample estimators and concentration bounds under independent sampling assumptions.
This note does not introduce a new benchmark. It provides a conceptual and statistical formulation for pairwise reference alignment, clarifies the role of the reference pair distribution, and distinguishes the general ordinal observable from scoring choices such as normalized log-probability or energy-based scores. We also provide an initial empirical study on Qwen2.5 models and RewardBench, where the proposed statistics increase with model size and instruction tuning and vary across reference-pair subsets as predicted by the formulation. 

---
# Efficient Diffusion LLMs via Temporal-Spatial Parallel Decoding and Confidence Extrapolation 

**Authors**: Zekai Li, Ji Liu, Yiqing Huang, Ziqiong Liu, Dong Li, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2605.30753)  

**Abstract**: Diffusion-based large language models (dLLMs) support parallel text generation via iterative denoising, yet inference remains latency-heavy because many steps are spent on redundant refinement and repeated remasking of tokens whose final values are already determined. Prior acceleration methods mainly depend on step-local confidence heuristics or fixed schedules, which are sensitive to prompt and task variation and ignore strong positional effects within a sequence. We cast diffusion decoding as a dynamic control problem and show that token-wise denoising trajectories provide the key signal for reliable control. We propose a trace-aware decoding framework with two components. First, Temporal-Spatial Parallel Decoding (TSPD) uses a lightweight temporalspatial controller that consumes per-token trajectory features, including confidence, entropy, and momentum, together with token position, to decide when a token has converged and can be safely fixed. Second, we introduce Confidence Extrapolation (CE), a training-free state-space module that forecasts future logit trends with uncertainty to support proactive decisions, including safe look-ahead and targeted stabilization when trajectories are oscillatory or underconfident. Together, TSPD and CE reduce unnecessary denoising iterations while preserving output quality, and they compose cleanly with system optimizations such as KV caching. 

---
# MosaicLeaks:Privacy Risks in Querying-in-the-Open for Deep Research Agents 

**Authors**: Alexander Gurung, Spandana Gella, Alexandre Drouin, Issam H. Laradji, Perouz Taslakian, Rafael Pardinas  

**Link**: [PDF](https://arxiv.org/pdf/2605.30727)  

**Abstract**: Deep research agents increasingly combine private local documents with external tools like web retrieval, creating a privacy risk: an agent's external queries may leak sensitive information from its local context. This risk is amplified by the mosaic effect, where individual queries may appear harmless but become revealing in aggregate. We introduce MosaicLeaks, a benchmark of 1,001 multi-hop deep research tasks that chain private enterprise documents and a public web corpus, forcing agents to make external queries that depend on local information. We evaluate leakage with an adversary LLM that observes only the agent's external queries and attempts to infer private information at three levels: the agent's research intent, answers to specific private questions and verifiable claims about the enterprise documents. We find that models across families and sizes frequently leak at all three levels, that zero-shot privacy prompting reduces but does not eliminate leakage and that reinforcement learning for task performance alone worsens leakage. To address this, we propose Privacy-Aware Deep Research (PA-DR), an RL framework that combines situational rewards for task success with a learned privacy classifier to provide dense credit assignment over both per-query and mosaic-level leakage. Training Qwen3-4B-Instruct with PA-DR improves accuracy from 48.7% to 58.7% and reduces answer and full-information leakage from 34.0% to 9.9%. 

---
# Skill is Not One-Size-Fits-All: Model-Aware Skill Alignment for LLM Agents 

**Authors**: Jianxiang Yu, Jiapeng Zhu, Bochen Lin, Qier Cui, Zichen Ding, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.30723)  

**Abstract**: LLM agents increasingly retrieve externally curated skills-procedural instructions retrieved at decision time-to improve performance on long-horizon interactive tasks. Existing skill libraries are typically treated as model-agnostic, reusing the same skill formulations across backbones with substantially different capacities and behaviors. However, our controlled experiments across multiple model scales show that skill effectiveness is strongly model-dependent: a skill that benefits one backbone can harm another. Motivated by this observation, we propose MASA Model-Aware Skill Alignment, a framework that adapts skills to each target backbone without modifying agent weights. MASA operates in two stages: (1) a hierarchical skill evolution pipeline that iteratively rewrites general and task-specific skills using hill climbing and UCB-driven tree search, guided by environment feedback and model capability profiles; and (2) a lightweight model-conditioned skill rewriter trained on evolution trajectories to reproduce the adaptation in a single forward pass. Experiments across three interactive environments and four backbones show that MASA consistently achieves the best overall performance, with gains of up to 25.8 points over the strongest baseline. The learned rewriter further generalizes to unseen tasks and environments without additional search, consistently outperforming a much larger teacher LLM at a fraction of the inference cost. 

---
# Neuron-Level Interventions for Gendered and Gender-Neutral Generation in Language Models 

**Authors**: Zhiwen You, Nafiseh Nikeghbal, Jana Diesner  

**Link**: [PDF](https://arxiv.org/pdf/2605.30717)  

**Abstract**: Language models (LMs) can produce gendered language and stereotypes even when given neutral prompts. Most prior work on gender bias in LMs primarily examines gender through a binary lens (feminine vs. masculine), with limited attention to gender-neutral forms, such as they/them pronouns or neutrally phrased job titles. How gender-related signals are encoded in the internal representations of LMs remains an open question. In this work, we study gender-specific neurons in LMs across three categories: feminine, masculine, and gender-neutral. We propose a neuron-level intervention method to identify neurons that are strongly tied to each gender category. We then test these neurons through controlled generation, showing that activating or masking gender-related neurons can steer a sentence toward a target gender form while preserving its original meaning. To evaluate the effectiveness of our gender-intervention approach, we curate two datasets with controlled sentences labeled across all three gender categories and validate the data quality through human evaluation. Experiments on two open-source LMs show that gender-specific neurons are not evenly distributed across model layers; instead, they concentrate heavily in the earliest layers with smaller contributions from later layers. Compared to existing methods, our method achieves more precise gender control, with less leakage into non-target gender categories and stable output quality through two evaluation criteria. Overall, our work examines how gender is encoded in LMs and provides a simple yet effective approach toward controlled gender intervention for both neuron intervention evaluation and gender bias mitigation. Code and datasets are available at: this https URL 

---
# ExpGraph: Model-Agnostic Experience Learning with Graph-Structured Memory for LLM Agents 

**Authors**: Tao Feng, Chongrui Ye, Tianyang Luo, Jingjun Xu, Xueqiang Xu, Haozhen Zhang, Zhigang Hua, Yan Xie, Shuang Yang, Ge Liu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2605.30712)  

**Abstract**: Large language model (LLM) agents have shown strong capabilities in reasoning, tool use, and multi-step interaction, but they often solve tasks from scratch and fail to reuse successful strategies or failure lessons from prior experience. Fine-tuning on collected experience can improve reuse, but it is inflexible when stronger or more suitable executors emerge. We propose ExpGraph, a model-agnostic experience learning framework that enables frozen and replaceable LLM executors to improve through external experience reuse without parameter updates. ExpGraph summarizes historical trajectories into reusable skills and failure lessons, organizes them as nodes in a self-evolving experience graph, and retrieves useful experiences through graph diffusion and utility-aware ranking. A lightweight retrieval copilot is trained with reinforcement learning using feedback that compares executor performance with and without retrieved experiences, while the graph is updated online from downstream task outcomes. We evaluate ExpGraph on ExpSuite, covering question answering, mathematical reasoning, code generation, and multi-step agentic environments including ALFWorld and AppWorld. ExpGraph improves over the strongest baseline by 12.2% and 4.7% on static tasks with smaller and larger executors, and by 21.4% and 12.7% in agentic environments, while reducing average interaction steps by 12.7% and 21.6%. Ablations show that graph-structured experience, utility-aware ranking, and adaptive retrieval jointly enable effective experience reuse across diverse tasks and executor models. 

---
# SAGE: A Novelty Gate for Efficient Memory Evolution in Agentic LLMs 

**Authors**: Sijia Wang, Dhanajit Brahma, Ricardo Henao  

**Link**: [PDF](https://arxiv.org/pdf/2605.30711)  

**Abstract**: Agentic LLMs must continuously decide whether newly extracted facts should be added, merged with existing memories, or ignored, yet prior work has focused more on retrieval and storage than on principled write-side control. We frame memory evolution as a novelty-detection problem and propose SAGE, a Spherical Adaptive Gate for memory Evolution that scores candidate facts with a von Mises-Fisher-based density estimator over memory embeddings and routes them with an adaptive threshold that tracks memory-store geometry. SAGE resolves clearly novel facts as ADD, clearly redundant facts as NOOP, and sends only uncertain cases to an LLM merge step, reducing expensive write-time reasoning. On LoCoMo, SAGE achieves the best average token-F1 against Mem0 on all seven open-weight backbone comparisons, while on GPT-4o-mini it reduces add-phase API cost by 3.4$\times$ and add-phase latency by 2.5$\times$ with only a small average judge-score gap. As a drop-in binary gate for A-Mem, SAGE skips roughly 16-18% of LLM calls across five models with minimal quality change on open-weight backbones. These results suggest that novelty-aware write control is a practical lever for improving both memory quality and system efficiency in long-term agentic memory. 

---
# ElasticMem: Latent Memory as a Learnable Resource for LLM Agents 

**Authors**: Tao Feng, Chongrui Ye, Tianyang Luo, Jingjun Xu, Xueqiang Xu, Haozhen Zhang, Ge Liu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2605.30690)  

**Abstract**: Long-term memory is essential for LLM agents to reason coherently across extended interactions, personalize responses, and reuse past experience. However, existing memory-augmented methods typically treat memory as a fixed resource: text-space approaches concatenate retrieved memories into the context window, causing substantial token overhead and sensitivity to noisy evidence, while latent-space approaches reduce textual cost but still rely on rigid retrieval or fixed-capacity memory interfaces. This creates a mismatch between query-dependent memory utility and fixed memory allocation. We propose ElasticMem, a memory-augmented LLM framework that learns to use memory as an elastic latent resource. ElasticMem builds an offline latent memory bank with retrieval keys and content caches, retrieves memories adaptively from the reasoner's hidden state, assigns each retrieved memory a variable latent budget through a learned policy, and injects selected latent states as soft memory tokens for generation. The full memory-use process is optimized with downstream task rewards through group-relative policy optimization. We evaluate ElasticMem on MemorySuite, covering memory-intensive QA and embodied agent control. Across Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct backbones, ElasticMem improves weighted average QA accuracy by 26.2% and 24.6%, and improves ALFWorld success rate by 66.3% and 27.2%, respectively, over the strongest baselines, while achieving the lowest ALFWorld token cost. Ablations and qualitative analyses further show that adaptive retrieval and elastic budget allocation help ElasticMem prioritize useful evidence and transferable plans beyond rigid cosine similarity. Our code for ElasticMem will be released at this https URL. 

---
# Human-Alignment, Calibration, and Activation Patterns in Large Language Model Uncertainty 

**Authors**: Kyle Moore, Jesse Roberts, Daryl Watson, William Ward, Grayson Heyboer  

**Link**: [PDF](https://arxiv.org/pdf/2605.30675)  

**Abstract**: Uncertainty Quantification is a large and growing subfield of large language model behavioral analysis. Primarily to recognize and combat hallucination, the field has largely focused on measuring and improving calibration, the accuracy of uncertainty judgments to task efficacy. In this work, we investigate the relatively underexplored question of how similar large language model uncertainty is to human uncertainty. We investigate the presence and strength of human-similar uncertainty signals, deemed uncertainty alignment, in large language model overt behavior and internal activation patterns. We identify whether the models show evidence of simultaneous alignment and calibration on a variety of datasets covering both multiple choice and open ended factual recall. And we characterize the effect of instruct fine-tuning on each of these facets. 

---
# TeachObs: A Human-Validated Benchmark for Multimodal Teaching Observation and Model Evaluation 

**Authors**: Yeil Jeong, Youngjin Yoo, Seobin Sohn, Hyejin Han, Jinseo Lee, Scott Howard, Unggi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.30673)  

**Abstract**: Classroom videos contain observable teaching practices, but their pedagogical and visual signals are rarely organized in forms suitable for model evaluation. We present \textit{TeachObs}, a human-validated benchmark for multimodal teaching observation in classroom videos. \textit{TeachObs} includes 30 public lesson videos from eight countries divided into 5,158 fixed 15-second scenes. Seven researchers annotated each scene with 39 binary observation codes, covering 20 visual codes, such as gesture, board work, pointing, and visual materials, and 19 nonvisual codes, such as instruction, monitoring, questioning, feedback, and reflection. Gold segment labels are constructed using reliability- and prevalence-aware rules based on Krippendorff's alpha. In addition to segment-level labels, three expert raters produced lesson-level ratings and qualitative evaluations of instructional design, instructional delivery, learner response, learning materials, and lesson closure across the 30 lessons, with rater coverage detailed in the body. Using these two human reference layers, we evaluate five vision-capable frontier LLMs across three tracks - text-only segment coding, text + frame segment coding, and lesson-level coverage scored under an LLM-as-judge protocol - and find that no single model consistently outperforms others across all three tracks, that adding a mid-frame inflates both true and false attributions per scene, and that model evaluations over-rate procedurally clear lessons relative to expert raters. \textit{TeachObs} therefore supports both fine-grained annotation benchmarking and whole-lesson evaluation, showing where AI systems can assist classroom video analysis and where expert judgment remains necessary across varied subjects, classroom formats, and annotation difficulty levels. 

---
# CobSeg: Coherence Boundary Modeling for Dialogue Topic Segmentation 

**Authors**: Sijin Sun, Liangbin Zhao, Jiaxiang Cai, Ming Deng, Mingyu Luo, Xiuju Fu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30668)  

**Abstract**: Dialogue topic segmentation is critical in many human-AI collaborative applications which requires identifying heterogeneous boundary cues, including lexical transitions near utterance edges and semantic discontinuities across utterances. Existing utterance models often dilute these local lexical signals. We propose CobSeg, a novel multi-branch architecture that separates coherence-level semantic continuity from lexical boundary transitions and recovers both through directional boundary prediction. CobSeg further uses boundary informativeness weighting to emphasize high-utility utterance positions, and incorporates a corpus-derived topic coherence cue with learned combination weights. While CobSeg is evaluated as a compact trainable segmenter under supervised gold-boundary training and a pseudo-label setting with automatically induced boundaries, it performs enhanced boundary prediction without LLM calls during inference. Across five benchmarks, it improves $P_k$ and $W_d$ particularly when local lexical cues are prominent: under gold supervision, it reduces $P_k$ by 0.7 points and $W_d$ by 0.6 points on VHF, and reaches $P_k$ of 1.0 on DialSeg711; with induced boundaries, it reduces $P_k$ by 14.8 points on VHF, by 1.5 points on DialSeg711, and by 1.1 points on TIAGE, outperforming prior non-LLM approaches. 

---
# EUDAIMONIA: Evaluating Undesirable Dynamics in AI 

**Authors**: Jun Rui Huang, Wang Bill Zhu, Ziyi Liu, Nathanael Fast, Ravi Iyer, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2605.30654)  

**Abstract**: Large language models (LLMs) are increasingly used as conversational partners for companionship, emotional disclosure, and interpersonal advice, but the social dynamics of these interactions can create harms that are not captured by capability-oriented or traditional safety evaluations. We introduce the Social AI Design Code, a framework for evaluating whether LLMs align with user welfare in social interactions, including whether they encourage harmful intimacy, dependence, or prolonged engagement. To evaluate these risks in natural and diverse user-LLM interactions, we operationalize the code with EUDAIMONIA, a benchmark of 969 user inputs and 3,147 design-requirement violation checks built from WildChat through weak-to-strong filtration, multi-model relabeling, and controlled rewriting. Evaluating 22 recent LLMs, we find that even the strongest models, Claude-Opus-4.7 and GPT-5.5, violate 30.7% and 27.2% of checks, respectively. Extended thinking does not reduce violation rates, suggesting that these failures are persistent social-alignment problems rather than deficits solvable through test-time reasoning alone. 

---
# Counterfactual Graph for Multi-Agent LLM Calibration 

**Authors**: Jiatan Huang, Mingchen Li, Ziming Li, Sunjae Kwon, Hong Yu, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30653)  

**Abstract**: Multi-agent LLM systems often treat agreement as evidence: when many agents in a panel give the same answer, that answer is assumed to be more reliable. We show that this assumption can fail after agents communicate. Communication can induce correlated failures and false consensus, so the same vote share may reflect reliable agreement in one topology but over-confidence in another. We propose CAGE-CAL, a counterfactual agent-graph calibration framework for multi-agent LLMs. For each query, CAGE-CAL compares an observed post-communication agent graph with a matched counterfactual no-communication graph, capturing both pairwise failure correlations and group-level dependencies. Rather than simply counting how many agents agree, CAGE-CAL estimates the counterfactual shift between observed and no-communication dependence, and calibrates confidence accordingly. Across five benchmarks, CAGE-CAL improves reliability discrimination with competitive ECE, and its calibrated confidence further improves topology selection over the best fixed-topology strategy. 

---
# Same Patient, Different Words, Different Diagnosis? Evaluating Semantic Stability in Clinical LLMs 

**Authors**: Mahdi Alkaeed, Adnan Qayyum, Nabeel Abo Kashreef, Muhammad Bilal, Junaid Qadir  

**Link**: [PDF](https://arxiv.org/pdf/2605.30646)  

**Abstract**: Large Language Models (LLMs) are increasingly used in clinical applications. However, their behavior remains highly sensitive to subtle linguistic variations, such as rephrasing or syntactic variation. This sensitivity poses risks in safety-critical healthcare settings, where semantically equivalent inputs should produce consistent predictions. However, a key challenge is to ensure that prompt variations truly preserve clinical meaning, as embedding-based similarity metrics often fail to capture distinctions involving negation, temporality, or severity. To address this limitation, we propose a semantic verification framework based on Natural Language Inference (NLI) to filter meaning-preserving prompt variations, which are further refined using an LLM-as-a-judge and audited by a clinical expert. In addition, we introduce three metrics to quantify model sensitivity: MeaningPreserving Variation Sensitivity (MVS), confidence variation (\Delta C), and Worst-Case Instability (WCI). We evaluate 16 open-source general-purpose (GP) and medical LLMs within the same model families and parameter scales, using reformulated prompts derived from the DiagnosisQA and MedQA datasets. Our results demonstrate that robustness differences between domain-specific (DS) models are mixed and highly model-dependent, i.e., domain specialization does not consistently improve or reduce robustness to meaning-preserving prompt reformulations. Several DS models rank among the most robust (when compared with GP counterparts), and strong GP baselines remain competitive as well. 

---
# COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models 

**Authors**: Arya Fayyazi, Mehdi Kamal, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2605.30641)  

**Abstract**: Large language models (LLMs) can reveal and amplify societal biases during chain-of-thought (CoT) generation. We present COFT (Chain of Fair Thought), a training-free decoding method that applies token-level fairness control at decode time, with distribution-free marginal validity guarantees (under exchangeability) for any frozen causal language model. COFT operates in three stages. First, it creates a masked counterfactual prompt by replacing sensitive spans with neutral tokens. Second, it compares the factual and masked logit distributions through lightweight logit fusion to attenuate attribute-driven biases. Third, it uses dual-branch split-conformal calibration to certify per-step candidate token sets at a user-chosen risk level. We evaluate COFT across six models and multiple bias benchmarks. Our method reduces standard bias metrics by 30-55% (median 38%) while preserving task utility and language quality. Reasoning accuracies remain unchanged within run-to-run noise margins. The computational overhead is modest, equivalent to one additional cached forward pass (<=11%). COFT offers a clear, auditable path to safer CoT generation with significant bias reduction, negligible utility loss, and no requirement for retraining, auxiliary classifiers, or weight access. 

---
# The Architecture of Errors: From Universal Impossibility to Patch-Local LLM Reliability 

**Authors**: Mikhail L. Arbuzov, Lee Mosbacker, Sisong Bei, Ziwei Dong, Dmitri Kalaev, Alexey Shvets  

**Link**: [PDF](https://arxiv.org/pdf/2605.30628)  

**Abstract**: Universal LLM reliability is not a finite-library problem: across all possible tasks, tools, schemas, knowledge sources, and evaluator expectations, new intervention-distinguishable failure modes can appear without bound, so no finite intervention dictionary can guarantee bounded residual error for every such mode. But deployed systems do not operate over the whole universe. They operate inside operationally bounded patches (legal review, medical RAG, code repair, customer-support agents, contract extraction) with recurring tasks, schemas, tools, and evaluator expectations. Within such patches, empirical evidence suggests failures are sparse, repetitive, and concentrated in a small recurring catalogue, so reliability becomes a local catalogue-discovery and intervention-coverage problem rather than an exponential token-length problem. We formalize this transition with two propositions and one corollary. Proposition 1 is the worst-case-mode-wise negative result: no finite intervention dictionary covers every distinguishable failure mode of an unbounded domain. Corollary 1 is the inverse-discovery implication: the logarithmic upper bound on mode discovery cannot accommodate linearly more distinct tail modes without exponentially more observed hard-failure events. Proposition 2 is the positive patch-local result: under log active-mode exposure and head-heavy coverage, a sufficient per-hard-decision intervention budget grows polylogarithmically in sequence length and becomes domain-constant once the patch catalogue saturates. The framework relocates rather than dissolves long-context difficulty: where the number of hard decisions itself grows with task length, reliability remains hard; the contribution is to identify the on-axis intervention rather than to make those regimes easy. 

---
# Semantic Motion Anchors: Bridging Motion and Meaning in Co-Speech Gestures 

**Authors**: Varsha Suresh, Mohammad Mahdi Abootorabi, Mohamed Salman, M. Hamza Mughal, Christian Theobalt, Ashwin Ram, Jürgen Steimle, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2605.30608)  

**Abstract**: Learning a shared representation between spoken text and gesture is central to co-speech gesture retrieval, synthesis, and understanding, but remains challenging for semantically meaningful gestures whose communicative intent is not captured by motion alone. Direct contrastive alignment between transcripts and continuous motion embeddings often overemphasizes low-level kinematics and misses the symbolic content of semantic gestures. We propose semantic motion anchors, natural-language abstractions of gesture motion capturing physical form and communicative intent. Our method discretizes 3D gestures into body-hand motion primitives, verbalizes them into structured descriptions, and grounds them in the transcript to provide auxiliary contrastive supervision. On BEAT2, our method improves text-to-gesture R@1 by 8.2% over a direct text-motion baseline and outperforms prior retrieval approaches on text to gesture and gesture to text retrieval directions. Beyond aggregate retrieval metrics, semantic motion anchor supervision helps retrieve gestures that are semantically meaningful for the spoken query, rather than defaulting to generic motion patterns. A downstream retrieval-augmented gesture generation study showed that users significantly preferred gestures retrieved by our approach over a retrieval-augmented generation baseline, demonstrating that semantically grounded retrieval translates to gestures that better convey communicative intent in downstream generation. 

---
# ImmigrationQA: A Source-Grounded Dataset and Small-Model Adaptation for U.S. Immigration Law 

**Authors**: Nazarii Shportun  

**Link**: [PDF](https://arxiv.org/pdf/2605.30589)  

**Abstract**: U.S. immigration law spans thousands of pages of official policy, federal regulations, and procedural guidance that change frequently and carry high stakes for petitioners who lack legal representation. We describe the construction of ImmigrationQA, a source-grounded question-answering dataset of 17,058 pairs across 13 immigration subdomains, and the fine-tuning of a Llama 3.2 3B Instruct model on that dataset using parameter-efficient LoRA. The corpus was assembled from 11 primary and secondary sources -- including the USCIS Policy Manual, 8 CFR, BIA precedent decisions, and community Q&A -- yielding 10,056 validated canonical documents and 18,308 text chunks. Structured QA pairs were generated from these chunks using Claude Sonnet 4.6 via five mode-specific prompts, with 22 pairs rejected for insufficient source-span overlap. The fine-tuned model was evaluated against a held-out split of 993 pairs using LLM-as-judge scoring on a 101-example stratified sample. The fine-tuned model scored a mean of 1.08/3.0 (16.8% fully correct; 101-example stratified eval) versus the Llama 3 8B base model at 0.85/3.0 (4% fully correct), a relative improvement of 27% in mean score; a zero-shot Claude Sonnet baseline scored 1.52/3.0 (25% fully correct). The fine-tuned model shows concentrated improvement in procedural subdomains (travel documents, adjustment of status, nonimmigrant visas) while remaining weak on complex legal reasoning and time-sensitive statistics. The full pipeline ran for approximately $29 in cloud compute. All artifacts -- dataset, model, code, and prompt templates -- are publicly released. The system is not a substitute for legal counsel and does not reflect regulatory changes after the corpus crawl date. 

---
# AI for Monitoring and Classifying Data Used in Research Literature 

**Authors**: Rafael Macalaba, Aivin V. Solatorio  

**Link**: [PDF](https://arxiv.org/pdf/2605.30582)  

**Abstract**: While platforms like Google Scholar and Semantic Scholar track citations for academic papers, no comparable infrastructure exists for monitoring dataset usage in research literature, leaving the landscape of data use largely opaque. Addressing this gap is critical for transparency, reproducibility, and monitoring of impact, yet progress is hindered by inconsistent citation practices, scarce labeled data, and ambiguous references to datasets in the wild. Traditional NLP approaches struggle with these challenges, motivating the shift toward more adaptive, semantically rich models. Building on prior work using LLMs for data mention detection and synthetic data for bootstrapping training, this paper presents an updated methodology for scalable dataset monitoring. We introduce a multitask GLiNER-based framework that jointly performs dataset mention extraction, relation identification, and usage-context classification. To address label scarcity, the pipeline leverages synthetic data generation to produce training examples and LLM-based revalidation to filter incorrect mentions and enforce labeling consistency, together improving reliability, coverage, and output consistency across the training pipeline. This work advances the development of open-source tools for monitoring data use in research literature, contributing to the broader goal of generalizable, unconstrained dataset citation tracking. 

---
# Speculative Decoding Across Languages 

**Authors**: Nirajan Paudel, Michael Ginn, Luc De Nardi, Alexis Palmer  

**Link**: [PDF](https://arxiv.org/pdf/2605.30580)  

**Abstract**: Speculative decoding has become a crucial component of large language model (LLM) inference, enabling faster generation by drafting multiple tokens and verifying them in parallel. However, small draft models tend to suffer from disproportionately poor multilingual capabilities. Thus, when generating text in a non-English language, speculative decoding is far less effective.
We compare three strategies to improve speculative decoding efficiency for eleven languages: finetuning the draft model on task-specific data (translation); finetuning the draft model on unlabeled monolingual corpora; and training simple n-gram draft models on the same monolingual corpora. We evaluate efficiency on translation (from English into the target language) and the held-out task of story generation. We find that while task-specific distillation can significantly improve efficiency, distilled models generalize poorly to a new task. Meanwhile, n-gram draft models, despite lower acceptance rates, consistently provide large speed-ups due to much faster draft generation. 

---
# Probing the Prompt KV Cache: Where It Becomes Dispensable 

**Authors**: Vinayshekhar Bannihatti Kumar, Manoj Ghuhan Arivazhagan, Disha Makhija, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2605.30574)  

**Abstract**: Prior KV cache compression schemes empirically demonstrate that the prompt cache is partially redundant during decoding, dropping or summarising entries with little accuracy loss. We ask when and what kind of redundancy: at which layers, after how many decoding steps, and in what form can the prompt span KV cache be replaced without breaking the task. A controlled splice intervention swept over layer cutoff and decoding steps shows this redundancy is about form (chat template scaffolding) rather than content. Replacing the upper layer prompt span KV cache with KV cache from a chat template scaffold whose user content is a neutral filler recovers near clean accuracy, while zeroing the same slots collapses accuracy. The dissociation replicates across the Qwen3, Gemma 3, and Llama 3 families on multiple datasets. 

---
# Generating and Refining Dynamic Evaluation Rubrics for LLM-as-a-Judge 

**Authors**: Zijie Wang, Eduardo Blanco  

**Link**: [PDF](https://arxiv.org/pdf/2605.30568)  

**Abstract**: LLM-as-a-Judge is a scalable alternative to human evaluation, yet existing rubric-based methods rely on human-annotated data such as reference answers or expert-crafted rubrics. We propose to automatically generate fine-grained evaluation rubrics without any human annotation. Our training-free method generates rubrics at dataset-specific and instance-specific granularities, achieving performance competitive with existing methods across four benchmarks. We further present a method that iteratively fine-tunes a rubric generator model via meta-judge reward signals. The fine-tuned generator outperforms all existing baselines in both pairwise and pointwise evaluation. Notably, a fine-tuned 14B rubric generator outperforms a much larger proprietary model at rubric generation, showing the effectiveness of our fine-tuning strategy. 

---
# Refining Word-Based Grammatical Error Annotation for L2 Korean 

**Authors**: Jungyeul Park, Kyungtae Lim, Wonjun Oh, Benjamin Nguyen, Zihao Huang, Mengyang Qiu, Jayoung Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.30545)  

**Abstract**: Korean grammatical error correction (K-GEC) presents a structural mismatch between word-based evaluation and the morpheme-level locus of many learner errors. Postpositions and verbal endings are bound to lexical hosts, but they encode grammatical relations that must be represented in correction and evaluation. This paper refines word-based grammatical error annotation for L2 Korean by addressing three connected problems in existing resources: surface target realization, Korean-specific edit annotation, and single-reference evaluation. We reconstruct target sentences from the National Institute of Korean Language (NIKL) L2 corpus under morphologically constrained realization rules and convert its morpheme-level annotations into word-level \texttt{m2} edits. We then define a Korean ERRANT-style annotation scheme that preserves the MRU core while distinguishing functional morpheme errors, spelling errors, word boundary errors, and word order errors. We also augment the KoLLA corpus with an additional reference correction, yielding a multi-reference evaluation setting for Korean GEC. Empirical validation shows that the refined NIKL targets yield lower perplexity, the converted \texttt{m2} files achieve higher agreement with source-target edit representations, and the refined resources improve KoBART-based correction under the same model setting. Multi-reference KoLLA evaluation further reduces the penalty imposed on valid corrections that diverge from a single reference, especially for neural and prompted GEC systems. These results show that Korean GEC evaluation depends not only on correction models, but also on reference data and edit annotations that reflect Korean morphology, spacing, and correction variability. 

---
# Generalistic or Specific Embeddings, Which is Better? An Empirical Study on Search for Clinical Coding in Non-English Languages 

**Authors**: David Rey-Blanco, Roberto Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2605.30529)  

**Abstract**: Sentence-embedding models for semantic search are overwhelmingly developed and evaluated on English corpora. When applied to clinical retrieval in other languages -- particularly retrieval of ICD-10-CM / CIE-10 codes -- recall degrades in ways often masked by aggregate benchmarks. We study whether large generative language models can serve as data factories to close this gap. We build a two-stage retriever (bi-encoder followed by cross-encoder reranker), fine-tuned from a Spanish biomedical encoder (PlanTL-GOB-ES/bsc-bio-ehr-es) on Gemini-generated synthetic data covering English, Spanish, Catalan, Italian, Portuguese and French, and evaluate against BioBERT-ST and the un-tuned Spanish encoder. The bi-encoder alone matches BioBERT-ST on MRR (0.876 vs. 0.866) and overtakes it on R@3 (0.650 vs. 0.626) and R@5 (0.804 vs. 0.790) without English biomedical pretraining. Adding a cross-encoder reranker lifts aggregate R@5 to 0.822 and dominates on four of five languages (+0.017 Spanish, +0.033 Catalan, +0.018 French, +0.037 Portuguese) at the cost of a small English regression. The trade-off is clinically acceptable: Portuguese reaches R@5 = 0.829 vs. BioBERT-ST's 0.714. Contributions: an open recipe for building domain-specific medical retrievers from LLM-generated data; quantification of the learning gain (MRR 0.755 to 0.876, +15.9% with ~19,500 synthetic pairs); and a characterisation of where gains concentrate by language and rank. 

---
# Evaluating using Mock Tool Calls to Quarantine Untrusted Prompt Inputs 

**Authors**: David Gros, Adam Gleave  

**Link**: [PDF](https://arxiv.org/pdf/2605.30521)  

**Abstract**: Large language models must frequently process untrusted inputs, such as judging an answer from another model or running tasks like spam and harm classifiers while under adversarial pressure. These inputs are often string-formatted directly into a prompt template, leaving systems fragile to manipulation. Current LLM specs from major providers like OpenAI distinguish trustworthiness along an Instruction Hierarchy, from System messages (most trusted) to Tool Results (least trusted). A possible natural mitigation is to wrap untrusted content in a mock tool call as a quarantine. We explore this hypothesis with an automated redteaming search over static attack strings across seven models and three LLM-as-a-Judge tasks. Counter to our hypothesis, tool-wrapping does not broadly improve robustness. On a binary evaluation task (GSM8K grading) it typically increases attack success rates, an apparent inversion of the instruction hierarchy. On scalar and pairwise tasks the effect is smaller and model-dependent, with no tested model reliably helped, and several showing inversion. We recommend evaluating this limitation in deployed systems, and longer-term, pursuing stronger Instruction Hierarchy training or new untrusted-input primitives. 

---
# Auditing LLM Benchmarks with Item Response Theory 

**Authors**: Sander Land, Daniel M. Bikel  

**Link**: [PDF](https://arxiv.org/pdf/2605.30504)  

**Abstract**: LLM benchmark labels are frozen at release and silently propagated into downstream benchmarks, errors and all. We introduce an Item Response Theory-based indicator that surfaces likely mislabels at 95% precision in the top 200 examples across seven preference and multiple-choice benchmarks using responses from 114 models, outperforming a supervised classifier. We trace these errors to mechanical labeling heuristics, upstream annotation mistakes inherited unchanged from source datasets, and fundamentally ambiguous items without a defensible single label. The same model fit reveals that reward models specialize in stylistic preference rather than factual knowledge, and identifies one frontier reward model that agrees with detected mislabels at 78% accuracy versus 38% for its peers, consistent with benchmark contamination or benchmark-specific over-optimization. 

---
# Linear Ensembles Wash Away Watermarks: On the Fragility of Distributional Perturbations in LLMs 

**Authors**: Zhihao Wu, Gracia Gong, Qinglin Zhu, Yudong Chen, Runcong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.30501)  

**Abstract**: Watermarking embeds statistical signatures in AI-generated text for detection and attribution. We reveal a fundamental vulnerability: when users access multiple models (today's reality), watermarks trivially fail. Watermarks perturb output distributions away from the original, and in competitive markets, these perturbations are typically independent across providers. We theoretically prove that averaging output probability distributions recovers the unwatermarked distribution with up to a second-order error term. Empirically, simply averaging 3-5 models cancels out these perturbations. We introduce WASH (Watermark Attenuation via Statistical Hybridisation), which solves practical challenges in ensemble generation: vocabulary misalignment and tokenisation differences across heterogeneous models. Experiments across six watermarking schemes and three LLMs show that averaging across 3 models suppresses detection z-scores from 5-300 to below 2 (below the detection threshold of 4) and reduces TPR at 5% FPR to below 50%, while improving quality by 27.5% and running 6 times faster than the best baseline on the long sequence generation. Our results suggest that robust AI-text detection via watermarking requires either accepting this fundamental vulnerability or unprecedented coordination among model providers. 

---
# CanLegalRAGBench: Evaluating Retrieval-Augmented Generation on Canadian Case Law 

**Authors**: Ethan Zhao, Maksym Taranukhin, Wei Cui, Moira Aikenhead, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2605.30497)  

**Abstract**: RAG-based legal assistants have been growing in popularity, but LLM hallucinations remain a key issue and potentially undermines justice. While benchmarks have been developed to evaluate progress, many rely on synthetic queries rather than realistic legal scenarios. Moreover, Canadian law remains underrepresented in existing evaluations. To address this gap, we introduce CanLegalRAGBench, a Canadian legal QA benchmark based on realistic queries and expert-annotated answers grounded in case law. Our evaluation shows that retrieval performance is sensitive to design choices and that open-source embedding models are competitive with closed source models. However, it also reveals the limitation of automatic evaluations that penalize systems for retrieving alternative relevant documents. We also find that generated answers often diverge from gold responses, either with hallucinations or by producing overly detailed or irrelevant content, with 8-29% of claims not being supported by the retrieved documents. We hope this benchmark will help drive continued progress in addressing limitations of legal RAG systems. 

---
# Configurable Reward Model for Balanced Safety Alignment 

**Authors**: Zhengping Jiang, Mehran Khodabandeh, Akash Bharadwaj, Manik Bhandari, Mayur Srungarapu, Anqi Liu, Benjamin Van Durme, Li Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.30487)  

**Abstract**: Aligning large language models (LLMs) to heterogeneous and rapidly evolving safety requirements remains a critical challenge. Existing instruction-tuned LLMs and standalone safety classifiers often fail to generalize to new safety configurations, motivating the need for Reward Models (RMs) that are explicitly configurable to changing specifications. We introduce the Configurable Safety Reward Model (CSRM), which is jointly optimized for calibrated safety compliance and reward modeling. Our approach is supported by configuration-targeted data augmentation that enforces instruction adherence while preserving relative severity structure. The resulting RM is sensitive to fine-grained safety configurations and conversational nuances, substantially improving generalization to previously unseen safety configurations. CSRM achieves state-of-the-art performance on recent configurable safety benchmarks, including CoSApien (94.6% F1) and DynaBench (75.8% F1), without requiring additional human annotation. When used for downstream safety alignment, CSRM yields LLMs with a significantly improved helpfulness-safety tradeoff compared to existing baselines. 

---
# When English Rewrites Local Knowledge: Global Narrative Dominance in Large Language Models 

**Authors**: Md Arid Hasan, Ruwad Naswan, Farhan Samir, Sharifa Sultana, Syed Ishtiaque Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2605.30481)  

**Abstract**: Large language models (LLMs) are widely used as cross-lingual knowledge interfaces. However, culturally grounded questions often reflect globally dominant narratives rather than local contexts. We study this failure mode as \textit{global narrative dominance} in Bangla, a low-resource cultural context. We introduce \texttt{CulturalNB}, a dataset of 717 manually curated Bengali cultural instances with parallel Bangla--English question--answer pairs and supporting evidence, metadata, and sociocultural annotations. Using question-only and evidence-based prompting, we evaluate nine state-of-the-art LLMs with human and two independent LLM judges across metrics for cross-lingual consistency, language anchoring, global substitution, institutional bias, and epistemic perspective coverage. Results show that questions asked in English systematically increase global substitution and institutional framing while reducing local perspective coverage. Local evidence improves factual consistency and perspective coverage, but does not eliminate language-induced epistemic shifts. These findings suggest that cultural failures in LLMs are not only missing-knowledge errors but also failures of grounding and narrative prioritization. 

---
# Your Multimodal Speech Model Says I Have a Face for Radio 

**Authors**: Maya K. Nachesa, Vlad Niculae, Vagrant Gautam  

**Link**: [PDF](https://arxiv.org/pdf/2605.30472)  

**Abstract**: As large neural models have become better at language tasks, researchers are increasingly building multi- and omnimodal models that handle more modalities of data. One example is the expansion of speech recognition models to audio-visual data for noise mitigation and multimodal subtitling. While performance and bias have been studied extensively in the single-modality regime, it is unknown how new modalities affect this, even though they produce biases in humans. We therefore propose the first bias evaluation of multimodal speech recognition, where we create videos pairing different faces with the same audio, and measure changes in speech transcription accuracy. We find large quality-of-service differences across mWhisper-Flamingo and Gemini models, with drops of up to 4.05 word error rate points, across self-declared gender, ethnicity, and their intersection. Our findings point to a priority for developers to evaluate, fix, and communicate such limitations, as providing more signals through additional modalities is not necessarily better, and may even lead to biased outcomes. 

---
# Knowledge Graph-Enhanced Zero-Shot Topic Classification: A Multi-Strategy Comparative Study 

**Authors**: Shahana Akter, Yatharth Vohra, Ankita Shukla, Souvika Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2605.30465)  

**Abstract**: Multi-label topic classification without labeled training data is a challenging task, specially when documents contain complex relational information. We present a zero-shot multi-label topic classification framework and systematically investigate how per-article knowledge graph augmentation affects its performance. The base framework classifies topics in documents without labeled training data and has four variants: article-only classification, keyword-enhanced classification, and self-consistency decoding variants of both. Then, we augment each base variant with per article knowledge graph. This graph is extracted from the input document through a pipeline similar to KGGen based on subject-predicate-object triples. We test all eight methods, four base and four graph augmented on fifteen LLMs and eight multi-label datasets across different domains. For the base framework, keyword-enhanced classification (AK) is the best performing method, and six out of fifteen LLMs surpass the sentence-encoder baseline. Graph augmentation has positive and negative impacts on small and large models, respectively. This shows that larger models already contain enough relational information from pretraining. Furthermore, the self-consistency decoding variant does not show performance improvements in any experiment while increasing computation costs about fivefold. 

---
# Can LLM Teams Play What? Where? When? 

**Authors**: Anastasia Kotelnikova, Viktor Byzov, Maria Dolzhenkova, Evgeny Kotelnikov  

**Link**: [PDF](https://arxiv.org/pdf/2605.30459)  

**Abstract**: Large language models (LLMs) remain limited on tasks requiring indirect reasoning, cultural knowledge, and coordinated hypothesis testing. We investigate whether team-based interaction improves LLM performance in What? Where? When? (ChGK), a quiz game designed to reward collective reasoning. We introduce three team strategies: Voting, Silent Team (the captain observes final answers), and Talkative Team (the captain observes both answers and rationales). To minimize data leakage, we evaluate these strategies on a dataset consisting of 572 ChGK questions released in 2025. Using six recent large-scale open models, we show that team-based strategies outperform single-model baselines, yielding gains of up to 20 percentage points in accuracy. The best team achieves 44.23% accuracy, and approaches human team performance on questions with available human statistics. Analysis of inter-model diversity reveals that disagreement strongly predicts lower accuracy, but explanatory communication substantially mitigates performance drops. We further examine captain behavior and find no evidence of self-preference bias; access to peer rationales improves captain judgments. Overall, LLM teams function primarily as answer selection and error-filtering mechanisms rather than generators of novel solutions. Our findings highlight the importance of interaction and suggest adaptive strategies as a promising direction for multi-agent systems. 

---
# Cross-Lingual Steering for Figurative Language Generation 

**Authors**: Linfeng Liu, Tiffany Zhan, Louie Hong Yao, Saptarshi Ghosh, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30443)  

**Abstract**: Multilingual large language models can generate figurative language, but whether the internal signals driving this behavior are language-specific or reusable across languages is unclear. Using activation steering as a probe, we estimate a direction for a figurative category from figurative--literal activation differences in one language and apply it during generation. Across five figurative categories, six languages, and four multilingual LLMs, these directions steer reliably within their own language, most robustly for metaphor and simile. More importantly, they transfer across languages: a direction learned in one increases the target behavior when applied to another, with German among the most receptive targets. Going further, directions assembled from other languages can match or even surpass a target language's own native direction, while removing this shared component weakens native steering. Together, these results provide direct evidence of a reusable but target-dependent cross-lingual signal for figurative generation. 

---
# Domain Adaptation and Reasoning Frameworks in Language Models: A Controlled Experiment with Historical Cosmology 

**Authors**: Francesco De Bernardis  

**Link**: [PDF](https://arxiv.org/pdf/2605.30415)  

**Abstract**: We investigate how domain adaptation reshapes explanatory behavior in language models using historical cosmology as a controlled setting. In Phase 1, we train a small language model from scratch on a pre-Copernican corpus from which explicit heliocentric references were removed, and evaluate whether Earth-motion or heliocentric continuations nevertheless emerge. In Phase 2, we fine-tune a larger pretrained model using QLoRA on the same corpus in order to study how adaptation modifies explanatory framing and cosmological stance. Model outputs are evaluated using an LLM-as-judge framework that labels both cosmological stance (geocentric, heliocentric, or ambiguous) and explanatory frame (premodern versus modern). In the constrained setting of Phase 1, the smaller models occasionally generate local Earth-motion continuations, but these remain globally unstable and insufficient to support coherent cosmological reasoning. In Phase 2, fine-tuning induces a large and statistically significant shift toward premodern explanatory framing, while the conditional cosmological stance distributions remain comparatively stable within those frames. As a result, increases in geocentric outputs arise primarily from redistribution over explanatory regimes rather than from direct modification of stance. These results suggest that domain adaptation may primarily reshape the linguistic frameworks from which continuations are generated, with changes in stance emerging secondarily from those shifts. 

---
# Exploring Autonomous Agentic Data Engineering for Model Specialization 

**Authors**: Yujie Luo, Xiangyuan Ru, Jingsheng Zheng, Jingjing Wang, Yuqi Zhu, Jintian Zhang, Runnan Fang, Kewei Xu, Ye Liu, Zheng Wei, Jiang Bian, Zang Li, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.30407)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance on general tasks, while often struggling to adapt to specialized domains without high-quality domain-specific data. Existing LLM-based data curation methods primarily rely on human-designed workflows, leaving it unexamined whether LLMs can autonomously execute an end-to-end data engineering pipeline for model specialization. We formalize \textbf{Autonomous Agentic Data Engineering}, a novel task designed to evaluate LLMs as autonomous data engineers that drive model specialization through end-to-end data curation. We frame data as an optimizable component and study agents that plan, generate, and iteratively optimize training data across multiple domains, guided by post-training performance improvement. Experiments show that autonomous LLM data engineers yield substantial gains, as GPT-5.2 constructs a training curriculum that improves a student model by \textbf{57.29\%}, entirely through iterative, agent-driven data adaptation. By illuminating both potential and bottlenecks, our study establishes autonomous data engineering as a measurable capability and charts a path toward agent-driven model specialization\footnote{Code will be released at this https URL.}. 

---
# Protocol for evaluating ChatGPT in biomedical association generation and verification using a RAG-enabled, cross-model majority voting workflow 

**Authors**: Ahmed Abdeen Hamed, Luis M. Rocha  

**Link**: [PDF](https://arxiv.org/pdf/2605.30400)  

**Abstract**: We present a protocol to evaluate ChatGPT's ability to generate disease-centric biomedical associations. It outlines how we generate the associations, validate the biological entities using biomedical ontologies, and verify associations using literature. The protocol includes a self-consistency strategy to assess generative reliability across ChatGPT models. To address ontology exact-match limitations, we provide a use case performing semantic verification through a workflow enabled by Retrieval-Augmented Generation (RAG) powered by open-source large language models (LLMs). This enables LLMs to establish truth over content generated by other LLMs and expose hallucination. 

---
# Vision-Language Models Suppress Female Representations Under Ambiguous Input 

**Authors**: Arnau Marin-Llobet, Simon Henniger, Mahzarin R. Banaji  

**Link**: [PDF](https://arxiv.org/pdf/2605.31556)  

**Abstract**: Alignment teaches vision-language models (VLMs) to avoid expressing demographic biases, and when gender is clearly visible they largely succeed. Far less is known about ambiguous inputs (a worker in full gear, a figure seen from behind) cases common in practice yet rarely studied. We find that minimal prompting pressure exposes occupation-gender defaults when prompting ambiguous input images, with models collapsing to male even for strongly female-stereotyped occupations. But do these outputs reflect what models actually encode internally? We introduce LALS (Latent Association Leaning Score), a zero-shot metric that projects visual-token activations into the model's text-embedding space to measure concept associations per token and layer. Across 15 occupations, over 800 gender-ambiguous images, and four VLMs, internal representations and outputs are systematically decoupled: models often encode a female association internally yet output male. Layer-wise analysis reveals an asymmetric filter -- male signal amplifies end-to-end while female signal peaks mid-network and is suppressed before generation -- and a color ablation shows that culturally loaded visual cues such as clothing color further modulate these internal associations. 

---
# Evaluating Factual Density in Multi-Source RAG: A Study in Medical AI Accuracy 

**Authors**: Michael R. DeMarco  

**Link**: [PDF](https://arxiv.org/pdf/2605.31506)  

**Abstract**: Retrieval-Augmented Generation (RAG) is the current industry standard for grounding AI in real-world facts. Traditional retrieval methods rely on keyword matching and topic proximity, ranking content based on how closely it sounds like the user's query. What they do not measure is how many verified facts the content actually contains. This structural gap, termed the Expert Blindness Effect, causes standard RAG pipelines to consistently bury high-density factual evidence in favor of lexically dominant text on the same topic. To address this gap, this paper introduces Factual Density (FD*), a novel retrieval optimization signal that measures the proportion of verified atomic claims relative to total token count. Using the NexusAgentics Ghost Audit preprocessing pipeline, raw text is scored for factual specificity using probabilistic factuality analysis to filter content before corpus ingestion. An initial formulation introduced a severe document-length confound (Pearson R = -0.8636, p = 2.27e-07). Implementing Z-score normalization within length bins resolved this bias, validating FD* as a length-independent density signal (p = 0.0749). Evaluated against the HealthFC benchmark (750 health claims labeled Supported, Refuted, or No Evidence by medical experts), FD*-optimized retrieval was the only condition to achieve 100% systematic review saturation in top-5 results, surfacing Cochrane evidence that standard cosine similarity ranked outside the top ten. Ground truth verification confirmed 25 mappings across seven HealthFC-supported claims. While full statistical validation across n=50 queries remains future work due to constraints on corpus-benchmark alignment, these findings establish factual density reranking as a low-cost, high-impact intervention for improving factual precision in health RAG architectures. 

---
# Knowledge Boundary Probing and Demand-Guided Intervention for LLM-Based Power System Code Generation 

**Authors**: Hui Wu, Xiaoyang Wang, Zhong Fan  

**Link**: [PDF](https://arxiv.org/pdf/2605.31478)  

**Abstract**: Large language models (LLMs) are increasingly used to automate power-system analysis, but many utilities and energy-research labs require on-premise serving for confidentiality, regulatory, reproducibility, and cost reasons. This makes the reliability of open-weight models a deployment issue. We show that first-pass failures in power-system code generation are dominated not by reasoning alone, but by structured API-knowledge boundary errors: hallucinated function names, misused parameters, and mishandled result tables in versioned simulation libraries.
We introduce PowerCodeBench, an execution-validated benchmark generator that pairs natural-language operator queries with pandapower code and numerical ground truth; an L0-L3 documentation-driven probing procedure that measures per-model API knowledge profiles; and a boundary-aware intervention that combines query-side API demand estimation with targeted proactive documentation injection and routed reactive correction.
On a 2,000-task frozen release, we evaluate ten open-weight LLMs (1.5B-480B parameters) and four commercial mid-tier APIs. The intervention improves every evaluated open-weight model of at least 7B parameters and every commercial API by 32 to 56 accuracy points. Open-weight models in the 70B-120B range match the commercial mid-tier accuracy range, while Llama-3.1-405B and Qwen3-Coder-480B lead the panel. The targeted prompts preserve the full-context accuracy ceiling while using 41% of the prompt-token cost. The result is an accuracy-side, deployment-time path toward reliable on-premise LLM assistance for grid-analysis workflows without fine-tuning or cloud inference. 

---
# PithTrain: A Compact and Agent-Native MoE Training System 

**Authors**: Ruihang Lai, Hao Kang, Haozhan Tang, Akaash R. Parthasarathy, Zichun Yu, Junru Shao, Todd C. Mowry, Chenyan Xiong, Tianqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.31463)  

**Abstract**: Mixture-of-Experts (MoE) has become the dominant architecture for frontier language models. To meet this demand, production frameworks have built optimized MoE training stacks over years of engineering effort. Yet evolving these stacks for new architectures and system optimizations remains expensive. With the rise of AI coding agents, they could automate parts of training-framework development and accelerate this evolution. But applying them to these existing frameworks carries hidden costs, invisible to today's throughput-only evaluations. We name this missing dimension agent-task efficiency (ATE): the cost of using coding agents to understand, operate, and extend a framework. Grounded in four agent-native design principles, we build PithTrain, a compact, agent-native MoE training framework. We further introduce ATE-Bench, covering real-world training-framework tasks. Our evaluation shows PithTrain matches the throughput of production frameworks, and on ATE-Bench, PithTrain enables higher agent-task efficiency, with up to 62% fewer Agent Turns and 64% less Active GPU Time. 

---
# DRIFT: Decoupled Rollouts and Importance-Weighted Fine-Tuning for Efficient Multi-Turn Optimization 

**Authors**: Jian Mu, Tianyi Lin, Chengwei Qin, Zhongxiang Dai, Yao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31455)  

**Abstract**: Large language models are increasingly deployed in multi-turn interactive settings where users or environments can iteratively provide lightweight feedback. Unfortunately, optimizing such behavior presents a sharp dilemma in practice: online reinforcement learning is able to effectively address multi-turn dynamics but is prohibitively expensive due to the cost of generating full correction trajectories at every update, whereas offline supervised fine-tuning (SFT) is efficient but suffers from distribution shift and behavioral collapse. To this end, we novelly propose DRIFT (Decoupled Rollouts and Importance-Weighted Fine-Tuning), a framework that operationalizes the theoretical insight that the KL-regularized RL objective is equivalent to importance-weighted supervised learning. DRIFT decouples rollout from optimization by sampling offline interaction trajectories from a fixed reference policy, deriving return-based importance weights, and optimizing the policy via weighted SFT on the resulting dataset. Empirically, we demonstrate that DRIFT matches or exceeds the performance of multi-turn reinforcement learning baselines while maintaining the training efficiency and simplicity of standard supervised fine-tuning. Code is available at this https URL. 

---
# Used Car Salesbots? Honesty and Credulity of LLMs as Bargaining Agents under Partial Information 

**Authors**: Antonio Valerio Miceli-Barone, Vaishak Belle, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2605.31445)  

**Abstract**: In this work we study agents in simulated bargaining scenarios, where a buyer and a seller communicate through a text channel and attempt to negotiate mutually beneficial trades, under different information regimes (complete information, information asymmetry or mutual uncertainty). We evaluate their performance w.r.t. game-theoretical solutions and further investigate their honesty (their tendency to disclose or withhold information or to mislead and deceive) as well as their credulity (their tendency to trust or distrust information provided by the other agent). We study zero-shot LLM agents with simple prompting scaffolding as well as fine-tuned agents, in order to investigate whether optimising the agents to maximise financial profits makes them stronger negotiators but also more dishonest and less trusting.
We find that off-the-shelf LLMs all substantially deviate from game-theoretical equilibria, they attempt to lie about their private information but cannot efficiently exploit information asymmetries. Fine-tuning on financial utility makes the agents stronger at achieving better deals but also more dishonest, highlighting the risks that optimising agents for a task can have on their safety. We release our code and a dataset of bargaining scenarios. 

---
# Trading Complexity for Expressivity Through Structured Generalized Linear Token Mixing 

**Authors**: Erwan Fagnou, Paul Caillon, Blaise Delattre, Alexandre Allauzen  

**Link**: [PDF](https://arxiv.org/pdf/2605.31367)  

**Abstract**: Token mixing layers play a key role in how language models can learn and generate long-range dependencies. Their efficiency relies on the necessary trade-off between decoding speed and the memory requirements, along with the cache size. Considering causal generation, this paper explores new trade-offs thanks to a unified framework which separates two crucial features: (i) the direct influence of inputs on outputs in one generation step; (ii) the recurrent propagation of information through past outputs. This framework encompasses major architectures such as attention and state-space models, but also generalizes the recurrence equations by allowing each state to depend on multiple past states rather than only the immediate predecessor. By introducing structure, we design new recurrence patterns that provably achieve the desired complexity, while providing theoretical insights on their expressivity -- trading runtime for expressivity in a principled way. Empirical validation is performed on synthetic tasks, along with language modeling. Together, these results provide a unified toolkit for the understanding and design of efficient and expressive token mixers across model families. 

---
# Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization 

**Authors**: Haolin Deng, Xin Zou, Zhiwei Jin, Chen Chen, Haonan Lu, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31312)  

**Abstract**: Multimodal hallucination remains a persistent challenge for Vision-Language Models (VLMs). Standard textual Direct Preference Optimization (DPO) often fails to mitigate it due to a lack of explicit visual supervision. While existing works introduce visual preference DPO by contrasting original images against negative ones, they suffer from a theoretically inconsistent objective caused by partition function mismatches and rely on coarse-grained negatives that could enable shortcut learning. In this work, we propose In-Context Visual Contrastive Optimization (IC-VCO). By placing contrastive images within a shared multi-image context, IC-VCO ensures a mathematically rigorous objective. We further introduce Visual Contrast Distillation (VCDist), an auxiliary reliability-gated regularizer that encourages consistency between multi-image contrastive training and single-image inference. Finally, we propose a contrastive sample editing strategy that generates hard negatives via precise semantic perturbations. Experiments on five benchmarks demonstrate IC-VCO's best overall performance and the effectiveness of our sample editing strategy. Code and data are available at this https URL. 

---
# COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation 

**Authors**: Tianyi Zhou, Dongrui Liu, Leitao Yuan, Jing Shao, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31264)  

**Abstract**: LLM agents are increasingly expected not only to complete isolated tasks, but also to carry bounded representations of human expertise, judgment, and interaction style. Building such person-grounded agents remains difficult because actionable knowledge associated with a person or role is usually embedded in heterogeneous traces rather than written as clean instructions. Existing memory and persona systems capture fragments of this evidence, while skill frameworks provide portable packaging formats; however, there is no end-to-end workflow for distilling these traces into inspectable, correctable, and agent-usable skills. We present an automated trace-to-skill distillation system for generating person-grounded AI skills via expert knowledge distillation. Given materials from a target person or role, this http URL produces a versioned skill package with two coordinated tracks: a capability track for practices, mental models, and decision heuristics, and a bounded behavior track for communication style, interaction rules, and correction history. The package can be inspected, invoked, updated through natural-language feedback, rolled back, installed across agent hosts, and optionally prepared for controlled distribution. We describe the artifact contract, generation workflow, correction lifecycle, deployment surface, and domain presets implemented in the open-source system. At the time of writing, the public repository has approximately 18.5k GitHub stars; the gallery lists 215 skills from 165 contributors and more than 100k cumulative stars across listed skill cards. The system illustrates how person-grounded skills can be represented as portable, correctable packages rather than opaque prompts or hidden memories. 

---
# Benchmarking and Enhancing Text-to-Image Models for Generating Visual Representations in Early Arithmetic Education 

**Authors**: Junling Wang, Boqi Chen, Heejin Do, Mubashara Akhtar, April Yi Wang, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2605.31212)  

**Abstract**: AI systems are increasingly used to support educational content creation, yet it remains unclear whether they can generate outputs that faithfully represent the pedagogical concepts they are intended to teach. Thus, we introduce equation-to-visual generation, a task that, in contrast to conventional image generation, requires producing pedagogically meaningful visuals from arithmetic equations while precisely preserving their numerical and relational structure. Informed by interviews with teachers and an analysis of educational materials, we construct E2V-Bench, a benchmark spanning four pedagogically grounded visual types, along with automatic metrics for evaluating visual correctness. Our evaluation reveals that recent text-to-image (T2I) models frequently fail on this task, with errors dominated by incorrect object counts and broken relational structure. Building on this, we explore benchmark-guided enhancement strategies. These strategies improve representative models, while the remaining gap calls for stronger numerical and relational grounding in future T2I models. 

---
# Probing Collision Grounding in Vision-Language Models for Safe Human-Robot Collaboration 

**Authors**: Jun Wang, Xiaohao Xu, Xiaonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31196)  

**Abstract**: Safe human--robot collaboration requires more than visual description: a monitor must determine whether the robot body is safely separated, already colliding with the scene or a person, or about to collide. We call this capability collision grounding: binding visual observations to robot body geometry, camera viewpoint, scene layout, human proximity, and temporal motion in order to infer present and imminent contact. We introduce TouchSafeBench, a physics-grounded benchmark for evaluating collision grounding in vision-language models (VLMs). Built in Habitat~3.0, TouchSafeBench contains 2,940 simulated indoor co-presence episodes across social navigation and social rearrangement, with synchronized multi-view RGB-D observations, top-down trajectory maps, calibrated camera metadata, and simulator-derived contact labels. We study two deployment-facing tasks: classifying the current safety state and warning about imminent collision before contact. Across three frontier or robotics-oriented VLMs and nine visual representations, current models remain far from reliable: the best average Macro-F1 stays below 50\%, explicit depth is not automatically transformed into robot-body collision evidence, and robot--scene contact is consistently harder than human-contact risk. TouchSafeBench reveals a central limitation of embodied VLMs: visual fluency does not imply physical accountability. Reliable robot safety monitors will need representations that explicitly bind viewpoint, robot morphology, metric geometry, and future collision. We will release the benchmark upon acceptance. 

---
# SpatialAct: Probing Spatial Reasoning-to-Action Capabilities of VLM Agents in 3D Scenes 

**Authors**: Tianhui Liu, Jie Feng, Zhiheng Zheng, Shengyuan Wang, Yiming Guo, Yanxin Xi, Hangyu Fan, Yong Li, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2605.31148)  

**Abstract**: Humans can effortlessly perceive spatial layouts, form cognitive representations, reason about spatial relations, and translate such reasoning into actions in everyday 3D environments. Although recent vision-language models (VLMs) have shown promising performance on observation-conditioned spatial perception and reasoning tasks, it remains unclear whether they can build coherent spatial understanding, act upon it, and refine their actions through multi-turn feedback. To study this problem, we introduce \textbf{SpatialAct}, a simulator-grounded benchmark for probing \textit{action-conditioned spatial reasoning} in 3D scenes. Starting from the most challenging setting, Multi-turn Interactive Refinement, we further design its decomposed counterpart, Single-step Error Detection and Fix, together with five fundamental spatial ability tasks to diagnose the underlying causes of model failures. Experiments reveal a clear reasoning-to-action gap: current VLMs can perform well on isolated spatial reasoning tasks, but struggle to maintain coherent spatial beliefs and produce reliable actions during multi-turn feedback, substantially underperforming humans. These results suggest that current VLM agents still lack robust spatial state tracking under action-induced environment changes, even when low-level control is abstracted away. 

---
# EvoDefense: Co-Evolving Black-Box Defense with Large Language Models 

**Authors**: Yu Li, Yuenan Hou, Yingmei Wei, Yanming Guo, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31140)  

**Abstract**: Large Language Models (LLMs) remain highly vulnerable to diverse attacks, particularly in black-box settings where the internals of target models are inaccessible. Existing black-box defenses typically rely on pre-defined filtering heuristics, which often fail to generalize to unseen attack types and target model architectures. We introduce EvoDefense, an experience-guided co-evolving black-box defense paradigm. EvoDefense employs a guard LLM to detect malicious queries and an experience memory module to accumulate defense knowledge from previous interactions. At the core of EvoDefense is a continuous attack-defense evolution loop, where an attack generator and the guard model iteratively refine their attack strategies and defense policies through experience-guided optimization. This design enables EvoDefense to generalize across unseen attacks and target models without retraining. Experiments on HarmBench, AdvBench, and AlpacaEval show that EvoDefense achieves consistently strong defense performance across seven popular models and five representative LLM attacks, while preserving competitive general capabilities. On HarmBench, EvoDefense reduces the attack success rate (ASR) of AutoDAN-turbo on Gemini-3-flash and LLaMA-3-8B-Instruct from 29.4% and 43.4% to 8.4% and 6.2%, respectively. 

---
# A Pilot Study on Curator-Guided Multilingual Art Description for Blind and Low-Vision Audiences with Small Vision-Language Models 

**Authors**: Iosif Tsangko, Andreas Triantafyllopoulos, George Margetis, Ioana Crihana, Björn W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2605.31080)  

**Abstract**: Blind and low-vision (BLV) audiences remain underserved by visual art descriptions, particularly across languages and in museum settings where privacy and intellectual-property constraints may favour small on-premise vision-language models (VLMs). This pilot study investigates curator-guided multilingual art description with Qwen2.5-VL-3B-Instruct for German, Romanian, and Serbian. We construct a parallel BLV-oriented caption corpus from artwork images and metadata, and compare language-specific LoRA adapters with a single multilingual adapter under a fixed backbone and training budget. Evaluation combines automatic lexical and embedding-based metrics with an LLM-as-Judge protocol calibrated against a small Romanian BLV pilot study. Under our pilot setup, language-specific adapters show more stable controllability and visually grounded description quality for Romanian and Serbian, while multilingual adaptation remains competitive in German. We frame these findings as deployment-oriented evidence for small on-premise VLMs, and highlight the need for larger BLV user studies and broader language coverage before drawing general conclusions about multilingual accessibility. 

---
# Towards Effective Long-Video Event Prediction via Multi-Level Event Semantics Mining 

**Authors**: Bo Peng, YuanJie Lyu, PengGang Qin, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31069)  

**Abstract**: Accurately predicting future events is fundamental to content understanding and decision-making across various domains. While prior research has primarily focused on text or short-video scenarios, long-video event prediction, characterized by vast multimodal context and more complex narratives, remains underexplored. Meanwhile, although recent Long-Video Language Models (LVLMs), built on Large Language Models (LLMs) and Vision-Language Models (VLMs), have shown promise in long-video question answering and summarization, they struggle to generalize to event prediction, as they can neither precisely extract event-related details nor perform fine-grained analysis of event development. To address this gap, we propose VISTA, a multi-level event semantics mining framework for long-video event prediction. Initially, VISTA applies a character-centric visual prompt to precisely extract event-related visual details, enhancing detail-level semantics; subsequently, it employs a knowledge-enhanced iterative retrieval strategy, guiding the LLM to progressively construct logically coherent event chains, thereby improving event-level narratives; ultimately, VISTA adopts a human-like propose-then-retrieve strategy to generate diverse future-oriented proposals and integrate multi-level clues, producing robust and accurate predictions. Extensive experiments on real-world datasets validate the effectiveness of VISTA for long-video event prediction. 

---
# From Prompt Injection to Persistent Control: Defending Agentic Harness Against Trojan Backdoors 

**Authors**: Jiejun Tan, Zhicheng Dou, Xinyu Yang, Yuyang Hu, Yiruo Cheng, Xiaoxi Li, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2605.31042)  

**Abstract**: LLM agents are evolving from conversational chatbots to operational tools in real-world workspaces. In local agentic harnesses, an LLM can read and write files, call tools, and reuse workspace state across sessions. While such capabilities enhance utility, they also expose a new attack surface for attackers. Attackers can embed a prompt injection within a file or tool output. Agents may read this hidden instruction, store it, and execute it later. In this multi-step trojan attack paradigm, no individual step appears malicious on its own, but these steps can collectively turn untrusted text into persistent control content. However, existing defenses often inspect each step in isolation. As a result, they can block a clear harmful action, but fail to detect the earlier write operation that plants the backdoor. To reveal this threat, we introduce ClawTrojan, a benchmark designed to identify multi-step trojan attacks in local agentic harnesses. In an OpenClaw-style simulated workspace with GPT-5.4, ClawTrojan reaches a 95.5% attack success rate (ASR), while existing single-turn prompt-injection attacks produce near-zero ASR on the same model. To address this threat, we propose DASGuard, which scans control-like text in sensitive local files, traces its origin, and removes control content that does not originate from a trusted source. Our results show that DASGuard achieves strong dynamic defense by combining runtime attack blocking with sanitized commits to the workspace. 

---
# A Persona-Based Evaluation Framework for Pluralistic Alignment in Generative AI 

**Authors**: Atahan Karagoz  

**Link**: [PDF](https://arxiv.org/pdf/2605.31021)  

**Abstract**: Current alignment paradigms for generative artificial intelligence rely predominantly on monolithic benchmarking frameworks that reduce the plurality of human judgment to aggregated statistical baselines, thereby obscuring cultural, demographic, and contextual variability in evaluation. We introduce a state-space constrained emulation framework for AI evaluation that replaces singular assessment functions with a structured manifold of synthetic cognitive profiles representing diverse human perspectives. We show that modern generative architectures can instantiate and maintain these evaluative personas with high consistency, enabling a form of pluralistic, perspective-dependent benchmarking that more closely reflects real-world consensus variability. However, we further analyze the stability of these simulated evaluators under sequential inference and stochastic prompt perturbations, revealing systematic degradation in persona coherence that manifests as state-space drift and semantic inconsistency. These findings suggest that static alignment constraints are insufficient for sustaining robust evaluative behavior over time. Instead, we argue for the necessity of embedding dynamic, viability-driven regulatory mechanisms within generative systems to preserve coherent cognitive emulation. By framing persona-based evaluation as a structured dynamical system over latent representation manifolds, this study provides a foundation for more adaptive, human-aligned, and context-sensitive approaches to AI evaluation. 

---
# Traceable by Design: An LLM Pipeline and Dashboard for EU Regulatory Consultation Analysis 

**Authors**: Thales Bertaglia, Haoyang Gui, Catalina Goanta, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2605.30995)  

**Abstract**: Public consultations generate large volumes of data in the form of stakeholder submissions that are practically unfeasible to analyse manually. We present an end-to-end LLM-based pipeline and interactive dashboard for structured topic extraction from regulatory consultation submissions, demonstrated on the European Commission's Digital Fairness Act (DFA) public call for evidence as a case study. The system processes raw PDF attachments and web-form responses, extracts topic annotations, and grounds every extraction in a verbatim quote from the source text. Applied to 4,322 DFA submissions, the pipeline produced 15,368 topic annotations supported by 20,951 verbatim evidence quotes. Three principles govern the proposed design: verbatim grounding, full traceability, and transparency by design. The dashboard exposes the full extraction dataset through five analytical views, from dataset-level topic overviews to individual paragraph drill-downs, with every result traceable to its source. Beyond the predefined DFA topic categories, the pipeline generated certain stakeholder concerns, such as Age Verification, Payment Processor Censorship, and Digital Ownership, that a fixed-taxonomy approach would have missed. The pipeline is domain-generic; adapting it to a new consultation requires only a prompt update and a new dataset. A live demo is available at this https URL. The code and processed data are publicly available at this https URL. 

---
# Generating Reports or Repeating Templates? Measuring and Mitigating Template Collapse in 3D CT Report Generation 

**Authors**: Tom Maye-Lasserre, Yitong Li, Bailiang Jian, Morteza Ghahremani, Benedikt Wiestler, Christian Wachinger  

**Link**: [PDF](https://arxiv.org/pdf/2605.30984)  

**Abstract**: Modern 3D medical vision-language models (VLMs) can generate fluent radiology-style text while exhibit critically low pathology detection and output diversity, collapsing to generic templates that under-report rare yet critical findings. We identify this failure mode as Template Collapse. This failure stems from the unique constraints of 3D medical imaging, e.g., limited data, severe label imbalance, and weak signals from volumetric encoders. Under these constraints, text-generation objectives encourage shortcut learning and fluent but weakly grounded reports. We systematically diagnose the Template Collapse through clinical fidelity, output diversity, normal-template bias, and rare-finding survival. To mitigate it, we propose CLarGen, a decoupled framework that separates what to say (clinical detection) from how to say it (language synthesis). CLarGen uses (i) a Latent Query Transformer for multi-label pathology detection, (ii) pathology-guided retrieval for clinically matched exemplars, and (iii) a medical language model to synthesize the final report from detected findings and retrieved context. Across state-of-the-art 3D CT report generation baselines, CLarGen mitigates Template Collapse and substantially improves clinical accuracy (macro-F1 0.487 vs. 0.189; CRG 0.472 vs. 0.368) while maintaining fluent reporting. Our results suggest that explicit, measurable clinical grounding is essential for template-collapse-resistant 3D CT report generation. Code will be released upon acceptance. 

---
# Reading Between the Citations: A Typed Claim Network for Scientific Literature 

**Authors**: Ning Ding, Sergio J. Rodríguez Méndez, Pouya G. Omran  

**Link**: [PDF](https://arxiv.org/pdf/2605.30966)  

**Abstract**: Knowledge graphs over corpora of inter-referencing documents - scholarly papers, legal opinions, policy briefs - encode the topology of reference but not its stance. The standard representation collapses a rich evaluative relation into an untyped edge, losing the very content that supports community-level queries about how one document is received by another. We propose the claim network: a representational pattern in which each cross-document reference is reified as a typed claim, carrying source, target, claim text, and a four-class stance label grounded in the citation-intent literature. We give a construction pipeline applicable to any corpus of scholarly inter-referencing documents and instantiate it on a corpus of 127 papers in 3D point cloud semantic segmentation, producing a network of 8,260 typed claims. Three downstream task families demonstrate what the network enables: retrieval signal augmentation, aggregated-stance summarisation, and topological analytics. Head-to-head evaluation against standard Retrieval-Augmented Generation (RAG) baselines shows that the gain over flat retrieval is the gain from the right intermediate representation rather than the wrong one. 

---
# ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment 

**Authors**: Jun-Hak Yun, Seung-Bin Kim, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.30965)  

**Abstract**: Recent advancements in text-guided audio generation have yielded promising results in diverse domains, including sound effects, speech, and music. However, jointly generating speech with environmental audio remains challenging due to the inherent disparities in their acoustic patterns and temporal dynamics. We propose ImmersiveTTS, an environment-aware text-to-speech (TTS) model that generates natural speech seamlessly integrated within environmental contexts by explicitly modeling cross-modal interactions. Our model builds on a multimodal diffusion transformer and fuses transcript-aligned speech latent with text-conditioned environmental context via joint attention. To enhance semantic consistency, we introduce a domain-specific representation alignment objective tailored to environment-aware TTS, leveraging complementary self-supervised representations from speech and audio encoders. Experimental results show that ImmersiveTTS achieves higher naturalness, intelligibility, and audio fidelity than existing approaches across objective metrics and human listening tests. 

---
# TUX: Measuring Human--AI Tacit Understanding 

**Authors**: Yueshen Li, Hanyi Min, Vedant Das Swain, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2605.30930)  

**Abstract**: As large language models (LLMs) increasingly act as collaborative partners, human--AI alignment is often evaluated through explicit task success, accuracy, or reward optimization. Yet many collaborative settings depend on tacit understanding: whether an agent can align with a human's evaluative stance or representational priors without clear objectives, communication, or feedback. To study this capacity, we develop a spectrum-placement task inspired by the social party game Wavelength, in which humans and agents independently place concepts along subjective spectra. We operationalize the Tacit Understanding Index (TUX) as a pairwise measure of similarity between human and agent judgments, and evaluate it with 241 human participants and 200 profile-conditioned LLM agents across four models. We find that nearest human--agent pairs in trait space achieve significantly higher TUX, suggesting that tacit alignment is structured by person-level characteristics rather than random similarity. Regression analyses show that TUX becomes more explainable as predictor sets become richer, with individual traits, decision-making styles, and confidence improving over aggregate trait-distance baselines. These findings suggest that tacit understanding between humans and LLMs is measurable, while revealing the limits of profile-based conditioning for capturing deeper representational alignment. 

---
# Attend to Evidence: Evidence-Anchored Spatial Attention Supervision for Multimodal RLVR 

**Authors**: Ruina Hu, Chen Wang, Lai Wei, Jionghao Bai, Bin Yu, Weiran Huang, Kai Wang, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30912)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) improves vision-language models (VLMs) by optimizing outcome rewards derived from final answers. However, such outcome-only rewards do not tell the model which image regions justify an answer. For questions that require visual grounding, these rewards cannot distinguish responses supported by relevant visual evidence from those produced by language-prior shortcuts or lucky guesses. We introduce EASE (Evidence-Anchored Spatial Attention), which augments multimodal RLVR with visual-evidence process supervision. EASE converts annotated evidence regions into a smoothed visual-token target and uses it to guide response-to-image attention during RL training, but only on high-reward trajectories. The annotations are used solely as privileged training labels, while inference requires only the original image and question. Across Qwen2.5-VL-7B, Qwen3-VL-4B, and Qwen3-VL-8B, EASE raises average scores over DAPO by 2.5 to 3.1 points on perception, hallucination, visual math, and multimodal reasoning benchmarks. Diagnostics and ablations show that EASE better aligns visual attention with annotated evidence regions. 

---
# BlueFin: Benchmarking LLM Agents on Financial Spreadsheets 

**Authors**: Srivatsa Kundurthy, Clara Na, Colton Moraine, Anoushka Mohta, Case Winter, George Fang, John Ling, Emma Strubell, Zach Kirshner  

**Link**: [PDF](https://arxiv.org/pdf/2605.30907)  

**Abstract**: We present BlueFin, a benchmark that tasks large language model (LLM) agents with synthesis, manipulation, and comprehension tasks over spreadsheet workbooks in the professional finance domain. Though estimates of the global population of paying users of spreadsheet software range in the hundreds of millions -- an order of magnitude more than the estimated global population of professional developers -- comparatively fewer resources have been devoted to exploring and expanding LLM capabilities in the spreadsheet domain, with fewer still dedicated to mirroring real occupational tasks encountered by those in professional finance roles. In response, we curate a set of 131 challenging, complex tasks with real-world relevance in the domain, containing 3,225 granular rubric criteria; notably, our rubric criteria and LM judge evaluations are validated by a team of expert human annotators, resulting in high-quality, granular evaluations of complex tasks that are difficult to verify programmatically but can be reliably evaluated by an LM judge agent. Our judge achieves parity with expert consensus ($\alpha=0.826$) with a macro-F1 score of 0.839. Frontier LLMs demonstrate poor performance on the challenging benchmark, with the strongest LLMs achieving less than 50\% average scores across tasks -- models exhibit particular weaknesses in dynamic correctness. Our contributions include a dataset of examples across three categories of spreadsheet tasks, an open source harness and agentic evaluation framework, and a characterization of existing frontier models' performance on our benchmark. 

---
# UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-Time Scaling 

**Authors**: Kaiyu Huang, Xingyu Wang, Mingze Kong, Zhubo Shi, Yuqian Hou, Hong Xu, Zhongxiang Dai, Minchen Yu, Qingjiang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.30898)  

**Abstract**: In real-world deployments of large language models (LLMs), balancing inference quality and computational cost has become a central challenge. Existing approaches tackle this trade-off along two largely independent dimensions: model routing, which switches among models of different scales to match request complexity, and test-time scaling (TTS), which adjusts inference-time compute within a fixed model for fine-grained control. However, this decoupled design introduces inherent limitations. Model routing yields coarse-grained, discrete performance changes due to the sparse set of model scales, while single-model TTS often encounters capacity ceilings and exhibits diminishing returns as compute increases. Moreover, treating the two mechanisms separately restricts adaptability in dynamic inference environments. To overcome these limitations, we introduce Unified Inference Scaling (UIS), which unifies model routing and TTS in a single optimization space. Building on this formulation, we propose UniScale, an online framework that models adaptive UIS as a contextual multi-armed bandit problem and learns inference policies via LinUCB. The framework incorporates efficiency-aware learning and cost modeling to ensure stable and scalable optimization over high-dimensional action spaces. Evaluation shows that UniScale effectively exploits the synergy in the UIS space to deliver a fine-grained and consistently better quality-cost trade-off across diverse, dynamic inference scenarios. 

---
# LLM Anonymization Against Agentic Re-Identificatio 

**Authors**: Ziwen Li, Jianing Wen, Tianshi Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.30848)  

**Abstract**: Agentic LLMs with web search change the threat model for text anonymization: weak contextual cues can become cross-referenceable evidence for re-identification, yet those same details also carry downstream analytic value of the text. Existing defenses either remove explicit identifiers, perturb text for formal privacy, or test rewritten text against non-web inference models, leaving underexplored the operating region between resistance to agentic web-search re-identification and utility retention. We introduce AURA (\textbf{A}nonymization with \textbf{U}tility-\textbf{R}etention \textbf{A}daptation), an LLM-powered \textit{mask-reconstruct} framework that decouples privacy localization from utility-preserving reconstruction and selects candidates with adversarial privacy and utility-retention checks. We evaluate AURA on real-user interview transcripts using re-identification attacks carried out by web-search agents, along with a utility evaluation based on interviewee-profile facts, codebook facts, and the joint contextual utility grid. Our results show that AURA improves the privacy-utility frontier by using adaptive privacy scope to strengthen resistance to agentic re-identification and using a mask-reconstruct anonymization method to better preserve contextual utility under fixed privacy scope. 

---
# On the impact of retrieved content representations in RAG Pipelines 

**Authors**: Jonathan J Ross, Bevan Koopman, Anton van der Vegt, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2605.30790)  

**Abstract**: Retrieval-Augmented Generation (RAG) supplements a language model's input with retrieved documents, yet most RAG pipelines inherit retrieval components designed for human readers. How retrieved content should be represented when the consumer is a large language model (LLM) rather than a human is less well understood. Recent work has proposed transformations of retrieved content and identified properties that affect generation, but each examines a single transformation or property in isolation, leaving open which features of a document's representation matter most. We address this with a controlled comparison: holding retrieval fixed, we vary only the representation of retrieved documents, comparing an original baseline against thirteen transformations spanning selection, summarisation, and reformulation, in query-dependent and query-independent variants. Across these fourteen representations we measure question-answering accuracy for four generators, and for each representation we also measure answer retention: whether a known answer-bearing document still supports its answer after transformation. We find that answer retention is the primary determinant of generator accuracy; notably, when retention is high, a representation's wording, structure, length, and query-dependence have limited effect. This suggests that accuracy gains attributed to specific mechanisms in prior work may be partly explained by how well those mechanisms preserve answer-bearing content, an attribution that cannot be settled without controlling for retention. 

---
# A Padding Method for Enhanced Encoding of Inorganic Structures with Varying Chemical Compositions 

**Authors**: Thang Dang, Haderbache Amir, Tzanakakis Alexandros, Yoshimoto Yuta  

**Link**: [PDF](https://arxiv.org/pdf/2605.30743)  

**Abstract**: Designing novel inorganic materials through generative models remains an important challenge for material science, driven by the complexity and diversity of inorganic structures across expansive chemical compositions and structural landscape. The vast combinatorial space of inorganic compounds demands innovative, AI-driven approaches to overcome limitations in generative accuracy and efficiency. To address this, we introduce a novel method that redefines the encoding and generation of inorganic materials by utilizing domain-specific symmetry-aware representation. Our approach not only refines the representation of intricate inorganic structures but also contributes to the field of material discovery by enhancing the precision and stability of generated candidates. Central to our methodology is a novel padding technique that exploits crystal symmetry information to enhance the encoding process. By integrating Wyckoff position length-aware padding into an encoder architecture, we achieve a more robust informed representation of inorganic materials. This symmetry-driven enhancement improves deep learning models to generate stable, previously unexplored inorganic structures with superior accuracy and computational efficiency. Furthermore, we introduce an end-to-end system that leverages the machine learning potential models to seamlessly generate novel, even those unseen in the training data, and stable inorganic materials from initial data to validated output. This pipeline integrates advanced generative models with stability analysis, marking a significant leap forward in the automated exploration and design of next-generation inorganic materials. Our method improved reconstruction accuracy 5.3% in proton conductor data, and generated 63.5% more novel stable inorganic material to baseline model on the perov-5 dataset. 

---
# OrcaRouter: A Production-Oriented LLM Router with Hybrid Offline-Online Learning 

**Authors**: Zhenghua Bao, Fengya Tian, Chris Zhang, Zhenjun Chen, Xile Ma, Yi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.30736)  

**Abstract**: The rapid development of large language models, each with distinct capabilities and inference costs, raises a practical deployment question: given an incoming request, which model should handle it? We present OrcaRouter, a production-oriented LLM router that combines a LinUCB-based contextual bandit over lexical and sentence-embedding features with a hybrid offline-online learning protocol. Offline, OrcaRouter obtains full-information feedback by evaluating each candidate model on a curated set of routing prompts, yielding a reward matrix used to fit one ridge regressor per arm. At deployment time, it initializes from these parameters and can optionally continue learning from bandit feedback, updating only the selected model's arm after observing its reward. At the time of our RouterArena submission (May 20, 2026), OrcaRouter-Adaptive ranked second on the public RouterArena leaderboard with an arena score of 72.08, achieving 75.54% accuracy at a cost of USD 1.00 per 1,000 queries. 

---
# Triaging Threats to Specialized Guardrails 

**Authors**: Wenjie Jacky Mo, Xiaofei Wen, Rui Cai, Boyu Zhu, Sicong Jiang, Zihan Wang, Minglai Yang, Zhe Zhao, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.30693)  

**Abstract**: Building robust safety guardrails is essential for deploying Large Language Models across diverse real-world applications. However, this goal remains challenging because safety risks span heterogeneous threat domains, while existing datasets cover only fragmented risk subsets and rely on inconsistent taxonomies. Consequently, it remains unclear whether current guardrails can generalize beyond narrow evaluation settings. To better understand the robustness of guardrail models, we first introduce GuardZoo, a unified human-annotated benchmark with 32,460 samples covering 15 distinct unsafe categories. Evaluation on GuardZoo reveals that monolithic guardrails suffer from task interference: different threat domains require distinct decision boundaries that are difficult to compress into a single model. We therefore propose RouteGuard, a router-expert framework that triages each conversation to specialized expert guardrails for threat-specific detection. Experiments show that RouteGuard improves fine-grained threat detection over strong guardrail baselines, generalizes better under out-of-domain evaluation, and supports flexible modular expansion to emerging threats. 

---
# How Early Adopters Used Generative AI Worldwide: Variation by Country Income and Language 

**Authors**: Madeleine I. G. Daepp, Isaac Slaughter  

**Link**: [PDF](https://arxiv.org/pdf/2605.30685)  

**Abstract**: AI is being used by people globally, but not everyone is using it in the same ways. Using a large-scale dataset of anonymized, de-identified, and privacy-scrubbed interactions with a widely available and free AI chatbot, we empirically characterize differences in early adopters' usage across countries. Schooling is the most common domain of use in most countries, particularly low-income countries, with a strong inverse association evident between schooling and country-level GDP. Leisure-related use, by contrast, is positively associated with country-level income. Language, we find, also shapes use: English-language interactions are overrepresented in places where the predominant languages were not well-served by existing models during the period of the study. Improving performance across languages may be a key factor, our work suggests, in whether this technology expands digital divides or enables leapfrogging. 

---
# CSULoRA: Closest Safe Update Low-Rank Adaptation 

**Authors**: Oleksandr Marchenko Breneur, Adelaide Danilov, Aria Nourbakhsh, Salima Lamsiyah  

**Link**: [PDF](https://arxiv.org/pdf/2605.30640)  

**Abstract**: Low-rank adaptation has become a standard method for parameter-efficient fine-tuning of large language models, but even small amounts of unsafe or adversarial fine-tuning data can substantially weaken the safety behavior of aligned models. Existing safety-preserving LoRA methods often rely on hard interventions such as projection, pruning, thresholding, or additional training objectives. While these methods can suppress unsafe update directions, they may also remove task-relevant information or require extra tuning. We introduce CSULoRA, a post-hoc method for correcting trained LoRA adapters through closest safe update estimation. CSULoRA estimates a safety-aligned subspace from the weight displacement between a safety-aligned model and its corresponding base checkpoint. It then decomposes each LoRA update into fully aligned, partially aligned, and off-subspace components. Instead of discarding components outside the estimated safety subspace, CSULoRA solves a closed-form penalized minimum-change problem that preserves the fully aligned component while smoothly attenuating potentially unsafe directions according to their relative energy. In adversarial fine-tuning experiments, CSULoRA substantially reduces attack success rate while preserving most of the utility gains obtained from standard LoRA fine-tuning. 

---
# Crafter: A Multi-Agent Harness for Editable Scientific Figure Generation from Diverse Inputs 

**Authors**: Haozhe Zhao, Shuzheng Si, Zhenhailong Wang, Zheng Wang, Liang Chen, Xiaotong Li, Zhixiang Liang, Maosong Sun, Minjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30611)  

**Abstract**: Scientific figures are among the most effective means of communicating complex research ideas, yet producing publication-quality illustrations remains one of the most labor-intensive parts of paper preparation. Existing automated systems each target a single figure type under text-only input, leaving the diversity of types and conditions researchers actually use unaddressed; their raster outputs further cannot be locally revised. Because scientific figures are structured compositions of discrete semantic components, the localized errors generators produce on such layouts demand not a stronger backbone but a harness. We instantiate this harness in two complementary systems: Crafter, a multi-agent harness for figure generation that generalizes across figure types and input conditions without architectural changes, and CraftEditor, which applies the same pattern to convert raster outputs into editable SVGs. Moreover, we introduce CraftBench, a benchmark spanning three figure types and four input conditions with human quality annotation. Experiments show that Crafter substantially outperforms both standalone generators and the agentic baseline on PaperBanana-Bench and CraftBench, with ablations confirming each component's independent contribution; CraftEditor faithfully converts outputs into editable SVGs that surpass all baselines. Our code and benchmark are available at this https URL. 

---
# An Organization-Scoped LLM Agent Runtime Architecture for Regulated Cybersecurity Operations 

**Authors**: George Fatouros, Georgios Makridis, George Kousiouris, John Soldatos, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2605.30604)  

**Abstract**: Regulated cybersecurity workflows lack a runtime substrate that enforces organization-level scope across retrieval, tool calls, memory, findings, reports, and audit while remaining model-agnostic and locally deployable. Recent large language model (LLM) agent systems report strong results on isolated cybersecurity tasks, yet they do not by themselves define an auditable platform architecture for regulated security operations centre (SOC) and compliance workflows, where a single analyst may trigger actions that bind the organization, and where the runtime must integrate with existing SIEM/XDR stacks as a primary source of context and alert-driven triggers rather than operate as a standalone analytical layer. This paper proposes an organization-scoped LLM agent runtime architecture for financial cybersecurity. The contribution is a typed Security Context that is created at every entry point, including SIEM/XDR notifications ingested as first-class triggers, and enforced at every component boundary, combined with a shared Runtime Core, logical specialist subagents, a governed Tool Adapter Layer exposing SIEM/XDR query, enrichment, and response primitives under uniform policy and audit, structured findings with evidence references, tiered human-in-the-loop (HITL) gates, and append-only audit. Model Context Protocol (MCP), extended telemetry, digital twins for pentesting, graph retrieval, and federated knowledge sharing are treated as optional extension paths rather than mandatory runtime assumptions. We describe an implementable slice as the architecture's testability surface, and we propose a falsifiable evaluation plan with metric-level pass criteria for architecture readiness, security-policy enforcement, evidence traceability, output quality, and operational observability. 

---
# AMNESIA: A Large Scale Medical Unlearning Benchmark Suite with Disease-Informed Analysis 

**Authors**: Saeedeh Davoudi, Reihaneh Iranmanesh, Ophir Frieder, Nazli Goharian  

**Link**: [PDF](https://arxiv.org/pdf/2605.30599)  

**Abstract**: Medical knowledge is continuously evolving. This creates a need to update or selectively forget information encoded in already-trained medical LLMs. Machine unlearning aims to remove the influence of specific training data from a model without full retraining. Yet, existing unlearning benchmarks rely on synthetic or small-scale general data, leaving clinical unlearning understudied. We introduce AMNESIA, the first large-scale, open source benchmark for medical unlearning, with 70,560 question-answer pairs from 8,820 patient notes across 11 disease categories. AMNESIA includes both factual questions testing direct recall and reasoning questions testing clinical inference. We use it to evaluate four widely used unlearning methods at both random patient and disease-level, and introduce a new metric for detecting leakage of medical terminology. We show that unlearning individual patients erodes knowledge of others with the same condition, calling for methods that can better separate patients from shared clinical knowledge. 

---
# Counterfactual Evaluation Reveals Hidden Capability Profiles in Clinical LLMs and Agents 

**Authors**: Matt Turk  

**Link**: [PDF](https://arxiv.org/pdf/2605.30590)  

**Abstract**: Two clinical AI systems can score nearly identically on coverage-based rubrics yet behave radically differently when their patient inputs change: one updates its recommendations to match the new clinical signal, while the other produces the same output regardless. We introduce the Causal Sensitivity Score (CSS), a pre-registered interventional metric that mutates oncology tumor-board cases along five clinically meaningful dimensions - biomarker flips, prior-treatment failures, biomarker removals, surgery-status changes, and stage perturbations - and scores whether each model updates its recommendations in the pre-registered correct direction using a {0, 0.5, 1.0} scale. Benchmarked against the Consensus Match Score (CMS), a coverage-based weighted recall metric, six frontier models from three labs evaluated in single-shot inference across 224 cases rank in nearly opposite orders: all six models change rank, the CMS-worst model becomes CSS-best, and one upper-mid CMS model ranks last on CSS. We further surface a universal safety blind spot: every frontier model fails on surgery-status interventions (at most 17.2% CSS on Family D), a finding CMS does not expose. The metric also transfers to tool-using agents: in a ReAct-style experiment, tool use improves CSS for five of six models (+2.5 to +20.3 percentage points), yet the lowest-CSS model retrieves the same chart sections and still fails to update its recommendations - revealing a structural responsiveness deficit visible only under counterfactual evaluation. Cross-judge replication and three-rater medical-professional validation confirm the aggregate findings. Interventional pre-registered metrics like CSS complement coverage-based evaluation for clinical AI agents: they capture responsiveness that coverage metrics miss and offer a candidate dense reward signal for future agentic RL systems. 

---
# Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)? 

**Authors**: Yue Zhang, Zun Wang, Han Lin, Yonatan Bitton, Idan Szpektor, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2605.30557)  

**Abstract**: Spatial reasoning is a fundamental capability for vision-language models (VLMs) deployed in real-world environments. However, visual observations are inherently limited representations of a 3D world: occlusion can render objects invisible, and perspective can make geometric properties misleading. Despite this, existing spatial reasoning benchmarks typically assume that observations are sufficient and reliable, focusing on whether models produce correct answers rather than whether they recognize when a question cannot be answered and what additional observations would be needed. In this work, we challenge this assumption by constructing a controlled evaluation framework, SpatialUncertain, and introducing two types of observation challenges: (1) occlusion, which hides target information, and (2) perspective ambiguity, which produces misleading visual cues. For each configuration, we design spatial questions that are answerable under clean observations but require abstention under the introduced challenges. We further evaluate whether models can identify which additional viewpoints would resolve perspective ambiguity. Our results across a diverse set of frontier open- and closed-source VLMs reveal two consistent failure modes. First, models are prone to overconfident answering, attempting to solve spatial reasoning tasks even when visual evidence is incomplete or misleading, with average accuracy around 30\% under occlusion and below 10\% under perspective ambiguity. Second, even when additional views are available, some models perform near random chance in identifying which would provide reliable evidence. Together, our findings call for moving beyond answer correctness toward evaluating whether models know when to abstain and how to seek reliable evidence. 

---
# Measuring, Localizing, and Ablating Alignment Signatures in LLMs 

**Authors**: Aniket Anand, Janvijay Singh, Zhewei Sun, Dilek Hakkani-Tür, Nick Feamster  

**Link**: [PDF](https://arxiv.org/pdf/2605.30526)  

**Abstract**: Aligned language models often exhibit a recognizable AI-like style, yet its connection to post-training and internal representations remains poorly understood. In this work, we study whether post-training introduces or amplifies AI-like stylistic regularities and whether these regularities have a localized internal signature. To this end, we compare human text, base-model generations, and aligned-model generations under matched human-source prefixes. Aligned generations show lower human-corpus affinity and higher AI-detection rates than base generations, suggesting that post-training shifts generated text away from human-corpus style and toward detector-visible AI-like text. We then introduce PASTA (Post-training Alignment Signature Targeted Ablation), a training-free method that estimates a post-training alignment signature from aligned-base residual contrasts and ablates the corresponding direction during decoding. Across 11 aligned models and 6 AI detectors, PASTA lowers the detection rate for most aligned models; this effect transfers well across detectors and is not reproduced by random directions. Qualitative analysis suggests that PASTA generations remain relevant and coherent while exhibiting greater stylistic variation. Together, these results show that AI-like stylistic effects of post-training can be measured, localized, and causally tested through activation ablation. 

---
# Revisiting Padded Transformer Expressivity: Which Architectural Choices Matter and Which Don't 

**Authors**: Anej Svete, William Merrill, Ryan Cotterell, Ashish Sabharwal  

**Link**: [PDF](https://arxiv.org/pdf/2605.30523)  

**Abstract**: Recent work describes what transformers can and cannot compute through connections to boolean circuits, but existing results lack exact characterizations and are sensitive to modeling choices. Padded transformers -- to whose input filler symbols such as ``...'' are appended -- emerge as a useful gadget for establishing equivalences to circuit classes by providing polynomial space for adaptive parallel computation. However, only a limited set of padded transformer idealizations has been studied, leaving open how robustly these equivalences hold under changes to attention type, model width, and uniformity. We find that, under practical assumptions, padded transformers are surprisingly robust to all of these, and identify numeric precision and model depth as the main factors affecting expressivity. Concretely, we prove that polynomially padded $\text{L-uniform}$ constant-precision transformers are equivalent to $\text{L-uniform AC}^0$, while growing-precision ones achieve $\text{L-uniform TC}^0$ regardless of width. Furthermore, looping enables sequential processing analogous to circuits: $\log^d N$-looped constant-precision transformers reach $\text{FO-uniform AC}^d$, and growing-precision ones reach $\text{FO-uniform TC}^d$. Interestingly, growing width or precision beyond logarithmic does not increase expressivity, and all our results hold for both softmax and average hard attention transformers. 

---
# MAAT: Multi-phase Adapter-Aware Targeted Unlearning 

**Authors**: Suryash Yagnik, Shubham Gaur, Saksham Thakur, Vinija Jain, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2605.30514)  

**Abstract**: Machine unlearning evaluation is structurally skewed: Why-type questions, which probe causal and relational knowledge, comprise less than 0.06% of CounterFact, 0.6% of ZSRE, and less than 1.3% of TOFU, MUSE, and WMDP-Cyber. This near-zero representation means that methods that fail on causal knowledge can score highly in aggregate, and this failure is undetectable without balanced evaluation. We present 5WBENCH, a balanced 5,000-sample benchmark with 1,000 examples per 5W category (Who, What, When, Where, Why), making causal unlearning failures quantifiable for the first time. Using 5WBENCH, we show that no existing baseline simultaneously achieves high forgetting and high retention on Why-type questions: aggressive forgetting degrades retained knowledge, while conservative methods fail to forget causal facts. Why-type difficulty stems from multi-hop reasoning chains (44% of Why entries vs. less than or equal to 2% for others) and gradient dilution over 40.1-token answer spans. We present MAAT (Multi-phase Adapter-Aware Targeted Unlearning), a three-phase framework operating on LoRA adapter weights, combining gradient-projected ascent, SVD rank-dimension pruning, task vector negation, and hybrid KL-hidden-state retain repair. MAAT is the first method to simultaneously achieve high forgetting and high retention on Why-type causal knowledge, reaching a new operating point on the forget-retain Pareto frontier. We make our code publicly available. 

---
# Improving Small Language Models for Code Generation with Reinforcement Learning from Verification Feedback 

**Authors**: Egor Skopin, Evgeny Kotelnikov  

**Link**: [PDF](https://arxiv.org/pdf/2605.30478)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) trains language models using programmatically checkable signals such as unit-test outcomes, enabling direct optimization for functional correctness in code generation. We conduct an empirical study of RLVR for Python code generation on the MBPP benchmark using two small models (Qwen3-0.6B and Llama3.2-1B) with LoRA fine-tuning. Across multiple reward formulations such as: unit-test-only rewards, static-analysis-only shaping via the Ruff linter, and a combined reward, we compare group-based policy optimization variants (GRPO and GSPO) and evaluate both functional correctness and behavioral diagnostics. In our experimental setting, RLVR improves pass@1 on MBPP test by up to 13 percentage points under proposed combined reward configuration. However, we find that reward shaping can induce systematic behavioral shifts: using only static-analysis penalties may bias the policy toward shorter completions that reduce lint errors without reliably improving functional correctness. In contrast, combined rewards mitigate this degeneration and yield more stable trade-offs between correctness and style constraints. Overall, our results highlight that RLVR effectiveness for code generation is highly sensitive to reward design and optimization granularity, and that diagnostics beyond pass@1, including generation length, Ruff severity profiles, and execution error types are useful for identifying failure modes. 

---
# Extracting accent features in spoken Brazilian Portuguese without sociolinguistic labels 

**Authors**: Pedro H. L. Leite, Pedro Benevenuto Valadares, Luiz W. P. Biscainho  

**Link**: [PDF](https://arxiv.org/pdf/2605.30457)  

**Abstract**: Regional accent classification in Brazilian Portuguese (pt-BR) suffers from the need for reliable labeling. While large self-supervised learning (SSL) speech models are powerful, their training pipelines dilute sociophonetic information, since accent labels are generally not reliable or are not used in training objectives. This work introduces a novel workflow for feature extraction using only acoustic labels. By isolating explicit regional accent landmarks and using a phoneme-based forced aligner (ZIPA), our targeted feature set captures dialectal variance more effectively than utterance embeddings, demonstrating that localized features can outperform general-purpose architectures on accent-related tasks using minimal and objective data labels. 

---
# Bounded Behavioral Indistinguishability for Black-Box LLM Distillation 

**Authors**: Munawar Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2605.30448)  

**Abstract**: Black-box LLM distillation is usually evaluated as an output-matching problem: a student is considered successful when its responses are semantically similar to, or task-consistent with, those of a teacher. However, output similarity does not imply that the student is behaviorally indistinguishable from the model it imitates. We introduce bounded behavioral indistinguishability, formalized as $(\epsilon,q,t,\mathbb{A})$-behavioral indistinguishability over an explicit prompt distribution, where $\epsilon$ bounds distinguishing advantage, $q$ bounds oracle queries, $t$ bounds computation, and $\mathbb{A}$ denotes the adversary class.
We instantiate this notion on Qwen and Llama teacher-student pairs using a controlled $5,000$-prompt behavioral probe suite. For each family, we compare the teacher with both the base student and the LoRA-distilled student, measuring whether distillation reduces distinguishability rather than merely improving similarity. LoRA raises semantic similarity from $0.788$ to $0.862$ for Qwen and from $0.814$ to $0.874$ for Llama. Yet adversarial evaluation reveals remaining behavioral differences: learned discriminators retain nonzero advantage, and pairwise category analysis shows artifacts concentrated in style/format, robustness, and domain-technical prompts. A pairwise teacher-identification adversary confirms this trend. With a different-family Llama judge and A/B-swap consistency filtering, Qwen distinguishing advantage drops from $0.158$ for the base student to $0.081$ after LoRA distillation. Query-budget experiments show that disagreement-guided acquisition does not consistently outperform stratified random sampling, indicating that coverage and diversity remain strong baselines. Our results show that semantic fidelity is useful but insufficient: black-box LLM distillation requires bounded, adversarial, and category-aware evaluation. 

---
# LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis 

**Authors**: Kewei Xu, Xiaoben Lu, Shuofei Qiao, Zihan Ding, Haoming Xu, Lei Liang, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30434)  

**Abstract**: Real-world data analysis is inherently iterative, yet existing benchmarks mostly evaluate isolated or short interactive tasks, leaving agents' ability to track evolving analytical context over long horizons untested. We introduce LongDS, a benchmark for long-horizon, multi-turn data analysis where agents must maintain, update, restore, and compose evolving analytical states. LongDS comprises 68 tasks constructed from real-world Kaggle notebooks, spanning 2,225 turns across six domains including Geoscience, Business, and Education. Tasks are designed around state-evolution patterns (e.g., counterfactual perturbation, rollback, multi-state composition), with an average dependency span of 11.3 turns. Evaluating five state-of-the-art models, we find that the best model reaches only 48.45% average accuracy, performance drops nearly 47 points from early to late turns, and long-horizon errors account for 52%--69% of failures. Further analysis shows that additional agent steps do not necessarily improve performance, suggesting that the key bottleneck is maintaining a correct analytical state rather than increasing interaction budget. We release LongDS to support research on reliable long-horizon agentic data analysis. Code and data will be released at this https URL. 

---
# Social Reasoning in Machines: Investigating Collective Truth-Seeking Dynamics in Large Language Model Debate 

**Authors**: Tom Pecher  

**Link**: [PDF](https://arxiv.org/pdf/2605.30391)  

**Abstract**: Human reasoning has long been theorised to operate socially, not through isolated individual cognition, but through collective adversarial discourse, a framework known as the Argumentative Theory of Reasoning (ATR). Rather than relying on individual "intellectualist reasoners" as the primary vehicle for truth-seeking, ATR reconceptualises truth as an emergent property of social epistemology: the product of imperfect individual reasoning refined under the adversarial pressure of debate. This distributed method of collective intelligence has guided humanity to ever-greater epistemic heights and underpins the foundational principles of all democratic systems. This thesis breaks new ground by, for the first time, simulating ATR through the multi-agent debate (MAD) of large language models (LLMs). With rigorous empirical analysis, we demonstrate that, when correctly engineering an epistemically diverse set of models, LLM-MAD can significantly improve truth-seeking performance on questionnaire-based tasks, even when individual debate participants exhibit limited standalone performance. Furthermore, we present strong empirical evidence that this performance gain is mechanistically grounded in the central principles of ATR, suggesting that collective reasoning may be universally favourable over individualist reasoning, rather than a quirk in biology or evolution. Finally, drawing on our analysis of debate dynamics, we propose a novel benchmarking methodology that leverages LLM-MAD to measure intrinsic model properties (such as hallucination propensity) in order to compare models in ways that current static benchmarking approaches cannot support. 

---
