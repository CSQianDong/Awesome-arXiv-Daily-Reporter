# ArchSpace: Recording, Reviewing, and Referencing LLM Architecture Research 

**Authors**: Jie Wang, Tong Zhu, JiaHao Kuang, Zhe Xu, Yunkai Chai, Dehong Zheng, Yu Cheng, Tao Ji, ZhouHan Lin, Qi Zhang, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.17332)  

**Abstract**: Model architecture shapes the scaling behavior, training efficiency, and inference cost of LLMs, making it a central focus of both industry and academia. In industry, teams such as those behind DeepSeek, Qwen, and Kimi routinely introduce architectural innovations, yet public reports rarely document which ideas succeeded, which were abandoned, and why. In academia, limited compute and inconsistent baselines often restrict evaluations to small-scale, single-point comparisons, making it difficult to establish whether improvements hold at larger scales or translate into broader impact. ArchSpace addresses both gaps as a public platform for Recording, Reviewing, and Referencing Architecture Researc}: each proposal is recorded with its experiments and reasoning, reviewed by the community, and referenced by later work. To evaluate ideas before costly large-scale training, ArchSpace leverages scaling laws to predict model loss at larger scales from smaller experiments, a practice widely adopted in industry. We provide initial references that make the platform usable: Qwen3.5 and OLMo3 baselines, fitted scaling laws, and a reusable training recipe. Ideas validated through this review can in turn become new baselines, and ArchSpace thereby aims to broaden the range of ideas that shape the next generation of models. 

---
# Learning When Diffusion Language Models Should Regenerate 

**Authors**: Minhae Oh, Jeonghye Kim, Jinwoo Koo, Yeongoon Kim, Jimin Seo, Minsik Seol, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.17320)  

**Abstract**: Remasking allows diffusion language models to revisit generated text, but effective correction requires knowing when and what to revise. Our diagnosis on Sudoku reveals two gaps. Hidden states contain error information that token confidence does not reliably expose, and remasking an isolated error usually reproduces it when the surrounding context remains fixed. These findings motivate DiVR , a verifier that predicts whether a partially generated sequence will reach a correct answer. Trained on the final correctness of the model's own trajectories, DiVR decides when to regenerate a block and which alternative to retain, while the generator itself stays unchanged. On five reasoning benchmarks with LLaDA-8B-Base, DiVR raises the average accuracy of the fine-tuned generator from 35.3\% to 43.4\% at $1.9$-$3.5\times$ the cost of greedy decoding, above search baselines that use $3.8$-$7.4\times$. It also complements post-training such as supervised fine-tuning and reinforcement learning, and generalizes to other generators, including Dream. 

---
# Goal Encoding in LLMs 

**Authors**: Arnab Sen Sharma, Eric Todd, Sheridan Feucht, Andy Arditi, Rohit Gandikota, Natalie Shapira, Gabriele Sarti, Constanza Fierro, Lisa Schut, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2611.17297)  

**Abstract**: We investigate whether and how LLMs encode goals: internal neural representations that precede and guide action, and that are flexible enough to be realized through many different execution paths. We construct a coding setting with multiple sub-tasks that allows us to evaluate goal-seeking behavior and the selection of sub-goals. In this setting, we localize an encoding of the next sub-goal in the outputs of a sparse set of attention heads. The same heads can be used to decode the LM's future actions in natural language (e.g., "I'm checking if a word is a palindrome"). We verify that this encoding is causal for next action and is plastic: transplanting it into a different environment, e.g., another programming language, triggers the same operation realized with a different sequence of tokens. The LM also sustains the encoding throughout the action's execution, which is puzzling because a transformer LM without recurrence cannot carry a state forward for more tokens than it has layers left. We propose that it does not need to. Under a non-recurrent goal-seeking (NRGS) account, the LM frequently re-reads the goal from the task specification, compares it with the progress visible in the growing context, and pursues what remains. We test the predictions of this account with targeted causal interventions. 

---
# Synthesis Smuggling: Maliciously Hiding Inferred Personal Attributes in LLM Outputs 

**Authors**: Gautam Galada, Supreeth Shastri, Sagnik Ray Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2611.17280)  

**Abstract**: We propose Synthesis Smuggling, a novel threat model in which an adversarial LLM infers sensitive information about a user from routine assistance-seeking interactions and steganographically encodes that into innocuous-looking model-generated content. The user may then unknowingly transmit that information by sharing the generated artifact, say as an email, social-media post, or message, through ordinary communication channels. An adversary monitoring these channels can discover this information. We demonstrate this threat by training two open-weight models to infer gender and age-group labels from conversational cues and encode their predictions using invisible characters or word choices. These results show that user-shared model outputs can become a covert channel for personal inferences, even when the model itself has no direct means of contacting an adversary. 

---
# Parsed, Plausible, Wrong: Output Format Controls Silent Errors in Small-Model Semantic Parsing 

**Authors**: Umar Ahmed, Maxim Dunaev, Mohammad Nour Al Awad, Sergey Ivanov, Ivan Khodnenko  

**Link**: [PDF](https://arxiv.org/pdf/2611.17169)  

**Abstract**: A small fine-tuned parser that emits a well-formed but wrong tree fails silently: nothing downstream can tell. We ask whether the output format, a free design choice, controls how often this happens. On the PIZZA ordering benchmark we fine-tune two sub-billion models (Qwen3.5-0.8B and LFM2.5-350M) with LoRA on six lossless encodings of the same target tree, from 44.3 down to 10.9 tokens per order, including two matched-length pairs that vary surface form at fixed length. Across 300 runs (5 training pool sizes, 5 seeds) with a hierarchical bootstrap and Holm correction fixed before the results were read, shortening alone never helps, but writing the target in ordinary words does. The word encoding is the most accurate format on both models (87.5% vs. 85.8% exact match for the native s-expression on Qwen) while emitting 4.1x fewer tokens, and it cuts the silent error rate by 2.3x on Qwen and 1.8x on LFM2, separating at every pool size. Error analysis locates much of the gain in negation. All runs also learn to refuse off-menu input (held out: 95.9-99.0% handled) and to apply in-utterance self-corrections. Strict refusal has a measured price: 2.1-10.1% of real, mostly polite, orders are refused. 

---
# Parameter-Efficient Conciseness Adaptation: Differentiable Length Control for Concise Clinical Reasoning in Smaller Language Models 

**Authors**: Tanzeel Abbas, Ahmad Kamran Malik, Nafees Qamar, Sana Liaqat  

**Link**: [PDF](https://arxiv.org/pdf/2611.17128)  

**Abstract**: Smaller language models (SLMs) make it feasible to deploy clinical natural language processing where large proprietary models are too costly, and parameter-efficient fine-tuning allows them to be specialised on commodity hardware. One obstacle persists after training: on medical tasks these models answer at far greater length than the clinical question requires. In clinical decision support and point-of-care use, where time is short, such verbosity is hard to act on even when the underlying reasoning is sound. Existing length-control methods depend on preference optimisation, reward models, or proprietary teachers that undercut the accessibility making SLMs attractive. We introduce Parameter-Efficient Conciseness Adaptation (PECA), which adds a single differentiable conciseness term to the QLoRA supervised fine-tuning objective. Because the token count is discrete and non-differentiable, PECA instead raises the end-of-sequence probability once generation exceeds a token budget, a smooth signal optimised during ordinary training, with no preference data, no auxiliary model, and a single commodity GPU. On a 1.7-billion-parameter model fine-tuned on the medical-o1-reasoning-SFT corpus, PECA reduces mean response length by 72.8% while improving multiple-choice diagnostic accuracy from 40.8% to 44.0%. A sweep over the conciseness strength traces an interpretable accuracy-to-length frontier for balancing brevity and diagnostic completeness. 

---
# Disentangling Spurious Correlations: How IRM Handles Diverse Shortcuts in Text Classification 

**Authors**: Paul Grunenwald, Benoît Crabbé, Guillaume Wisniewski, Pirmin Lemberger  

**Link**: [PDF](https://arxiv.org/pdf/2611.17107)  

**Abstract**: In NLP, models trained via Empirical Risk Minimization (ERM) often exploit spurious correlations, such as dialectal markers, at the expense of robust linguistic features, thereby degrading their out-of-distribution (OOD) generalization. Invariant Risk Minimization (IRM) aims to suppress such shortcuts by identifying predictive relationships that remain stable across environments, but its practical utility for NLP remains debated. We use a causality-driven framework to evaluate IRM under three shortcut-generating mechanisms: label-generated features, confounding, and selection bias. Our results show that IRM successfully rejects shortcuts across the evaluated causal and anti-causal structures, including under limited environmental variation and increasing label noise. We then extend our analysis beyond controlled synthetic artifacts to two less constrained settings: gender-profession associations in Bias in Bios and semantically loaded actor-name shortcuts in ActorCorr. These experiments show that invariant learning can mitigate more realistic textual shortcuts, while semantically meaningful features require careful calibration of the invariance penalty. 

---
# Optimal Single-History Tokenization Disambiguation for Provably Secure Linguistic Steganography 

**Authors**: Feng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17091)  

**Abstract**: Provably secure linguistic steganography assumes that the receiver recovers the sender's tokens, but a text channel delivers only text, and ambiguous tokenizations break extraction. Existing fixes change the distribution, synchronize whole pools of tokens at a capacity that collapses with large vocabularies, or track several token histories. We prove that if extraction is exact and error-free across full-support language models, a single-history receiver is confined to a prefix-free candidate set at every step -- even if it reads the entire future text and the sender constrains later steps -- and must therefore give up the stop entropy, the entropy of deciding whether each prefix token ends there. Fixing these stop decisions on the prefix forest with shared randomness yields coin projection: a closed-form scheme that attains the bound, works with any distribution coder in one model call per token, preserves ordinary sampling, and refines SyncPool for every key. Across English, Chinese and Japanese, ambiguous text is rare (at most 11 in 1,000 tokens), yet the stop entropy at word and morpheme prefixes takes 7.2-9.0% of the entropy at every step; SyncPool's further loss comes from space and single-byte UTF-8 hubs that merge subtrees while carrying almost no stop uncertainty. Coin projection keeps 91-93% of the string entropy for the coder at every vocabulary size; with the full vocabulary, it turns 2.16-6.85 times as much entropy into message bits as SyncPool and recovers every completed message. 

---
# Discovering Effective Agent Memory Designs through Program Evolution 

**Authors**: Xinheng Ding, Seyyed Saeid Cheshmi, Ziyang Zhang, Tairan Chen, Wenya Xie, Yu Hou, Rui Zhang, Zirui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17068)  

**Abstract**: Program evolution provides a promising approach to discover agent memory designs and study the choices underlying their effectiveness. We introduce MAPE, a framework that evolves whole memory programs from minimal seeds through a standardized lifecycle interface and an iterative diagnosis-proposal loop. Experiments on LoCoMo and LongMemEval yield an absolute accuracy gain of up to 12% over manually designed baselines at 35% of the serving cost, with further evaluations supporting transfer across serving backbones and benchmarks. Empirical analyses and ablations identify and validate two recurring design patterns: retaining raw conversations as retrievable evidence preserves details omitted from extracted memories, while composing retrieved evidence according to query intent helps the answering model use it effectively. Transplanting the composition module into five existing memory systems improves accuracy across all five. Beyond these patterns, the search objective shapes where computation goes: optimizing serving cost alongside accuracy shifts computation from LLM-based memory construction to selective processing at query time while retaining competitive accuracy. Together, these findings offer empirical lessons for agent memory design. Code is available at this https URL. 

---
# Evaluating Pretrained Language Semantics for Structured EHR Prediction Tasks 

**Authors**: Ashvin Gupta, Brendan Delaney, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2611.17053)  

**Abstract**: Structured Electronic Health Records (EHR) models typically treat each clinical concept as a dataset-specific token, discarding the natural-language semantics associated with the event. As a result, relationships between concepts must largely be learned from EHR co-occurrence, without directly exploiting any prior semantic knowledge. We investigate how pretrained causal language models can be adapted for longitudinal EHR prediction by developing tEHR. Here, we represent events with their natural language description and use this text stream to apply EHR-adaptive continued pretraining, followed by task-specific discriminative fine-tuning using LoRA, utilising the LLM as a longitudinal patient-encoder. We evaluate this approach across four prediction tasks on the MIMIC-IV dataset and pancreatic cancer prediction on the CPRD dataset at 1 to 12 month horizons, with tEHR using the Mistral-7B-v0.3 backbone achieving the strongest overall performance. Ablations show that performance declines when clinical descriptions are removed or misaligned, when numerical values are omitted, and when either stage of adaptation is removed, while explicit elapsed-time descriptions provide little additional benefit. Analysis of the learned patient representations further reveals clinically coherent structure associated with distinct patterns of disease risk and clinical history. 

---
# Auditing AI peer reviewers: a benchmark on real scientific papers 

**Authors**: Íñigo Zubeldia, Boris Bolliet, Francisco Villaescusa-Navarro, Pablo Villanueva-Domingo, Miles Cranmer  

**Link**: [PDF](https://arxiv.org/pdf/2611.17052)  

**Abstract**: LLM-based peer review is now used at production scale, but its evaluation remains limited. We introduce a benchmark that injects objective, localisable errors of four types (general issues, mathematics, numerics, citations) at three severity levels into real papers and measures each reviewer's detection rate by error type and severity, together with its false-positive rate on clean copies. We also present Skepthical, a multi-agent reviewer that verifies citation-backed claims by retrieving the cited papers, mathematics with computer algebra, and numerical claims with executed code. We apply the benchmark to two corpora of 15 recent arXiv preprints each, one in astrophysics and one in machine learning (ML), injecting a total of 357 errors, and evaluate twelve reviewer systems: six single-LLM reviewers, two Claude Code configurations, three publicly available reviewers, and Skepthical. The choice of model has the largest impact on detection rate: given the same prompt, three frontier LLMs differ by about 50 percentage points in total detection on both corpora. Tools and scaffolding help when they contribute what the model cannot supply on its own. In particular, on astrophysics, Claude Code improves on a single Claude call in mathematics and numerics detection, and enforced citation retrieval gives Skepthical the highest citation error detection rate of all systems by a large margin. This citation lead, however, is not observed on ML, where the models appear to know the literature better and Skepthical's citation detection rate is statistically indistinguishable from that of the best single LLM. False-positive rates are below 10% for most systems, but vary with prompt, model, and scaffolding, with mathematics flags the main source of false positives. The evaluation pipeline, the injected errors, and the adjudication logs are released as a benchmark. 

---
# Similar but Not Redundant: Embedding-Gated Refinement for High-Granularity Topic Models 

**Authors**: Daniel Denley, Robert L. Nicol, Mike J. Chantler  

**Link**: [PDF](https://arxiv.org/pdf/2611.17030)  

**Abstract**: High-granularity topic modelling extracts hundreds of niche topics from a corpus, yet this regime is rarely studied and is more commonly framed as a failure mode: "over-clustering," in which coherent themes fragment into redundant subtopics. We argue that this framing conflates genuine redundancy with topics that are semantically similar but meaningfully distinct, and that the central challenge in high-granularity modelling is separating the two. Recent post-hoc refinement methods address topic redundancy through either embedding-based similarity thresholding or LLM-guided merging. We show that neither suffices in high-granularity settings and propose embedding-gated iterative merge (EGIM), a hybrid approach that iteratively generates context-rich topic representations, uses embedding similarity to propose candidate merges across the topic space, then applies LLM judgement to accept only genuinely redundant pairs while preserving niche distinctions. Across two patent corpora and one legal corpus, LLM judgement rejects 83% of embedding-identified candidates as non-redundant. EGIM improves discriminability on merged topics by 25-55% over a prior iterative LLM baseline while better preserving the original topic distribution. 

---
# Morphological Simplification in Alzheimer's Disease: Computational Analysis and Identification From Connected Speech 

**Authors**: Mohamed Zakaria Kurdi  

**Link**: [PDF](https://arxiv.org/pdf/2611.16968)  

**Abstract**: Purpose: Morphology is often regarded as relatively spared in Alzheimer's disease (AD) and is rarely modeled in connected speech. This study asked which morphological features distinguish speakers with AD from healthy controls, how well morphology classifies AD compared with other linguistic levels and BERT-based models, and whether the signal exceeds transcript length and vocabulary. Method: Fifty-five automatic measures in eight families, seven of them new and theory-motivated, were computed from participant speech in the ADReSS Cookie Theft descriptions (78 AD, 78 controls) and analyzed with group comparisons, equivalence tests, SHAP, family ablation, likelihood-ratio tests, feature-selection classifiers, and BERT-based models. The automatic measures were validated manually, and models were transferred to two other corpora. Results: Nineteen of 38 measures differed between groups: speakers with AD produced fewer suffixed and inflected words, shorter verbs and nouns, and fewer present-progressive but more past-tense forms (largest effect r = -0.42), whereas errors were rare and statistically equivalent between groups. Eight measures survived adjustment for length and lexical frequency. With test-set feature selection, morphology reached a macro-F1 of 0.85, on par with syntax and above fine-tuned BERT (0.80 over five seeds); fully cross-validated selection gave 0.62. Adding morphology to BERT improved AUC (0.91 vs. 0.89) but not macro-F1. Transfer was limited, although two thirds of the effects kept their direction in mild cognitive impairment. Conclusions: AD connected speech shows morphological simplification and temporal reanchoring rather than breakdown. Morphology is more informative than its reputation suggests.
Keywords: Alzheimer's disease; morphology; connected speech; ADReSS; inflection; tense and aspect; SHAP; machine learning; BERT 

---
# Lara Mini Base: Language Diversity as a Resource, Not a Trade-off 

**Authors**: Translated, Maria Pia Adinolfi, Isabelle Andrieu, Grazia Fiorella Anfuso, Melania Annunziata, Daniele Antonini, Marta Arena, Luca Barone, Dana Barras Franco, Francesca Basiricò, Francesca Bee, Sofia Bertini, Nicola Bertoldi, Giorgio Biancini, Alessandra Bigi, Giulia Boccalini, Tommaso Bonomo, Enrico Boscardi, Sebastien Bratieres, Francesco Brigante, Giulia Brignola, Romolo Buondestino, Brenda Patricia Byrne, Giorgia Campora, Edoardo Cappelli, Veronica Carascosi, Francesca Carducci, Cody Carlson, Davide Caroselli, Ambra Cascio, Dario Cascio, Mauro Cassani, Andrea Castro, Alessandro Cattelan, Giulia Ceccacci, Paola Cellini, Giulia Chiriatti, Federico Cialini, Valentina Ciarlariello, Daniele Cocozza, Emanuela Coghe, Kathleen Compares, Marco Comunità, Davide Conti, Helen Rebecca Corfield, Michele Cortellessa, Anna Cracco, Aurora Cuppone, Arianna D'Andrea, Maurizio Davoli, Andrea De Angelis, Andrea De Blasi, Rebecca De Giorgi, Annalisa De Petra, Arianna De Pilla, Joris Demiraj, Elena Deufemia, Federica Di Antonio, Pierluigi Di Cianni, Vincenzo Di Cicco, Michela Di Clemente, Alan Di Francesco, Ylenia Di Gese, Federica Di Iulio, Claudia Di Lorenzo, Andrea Di Paola, Giulia Di Pilla, Jessica Dickson, Francesco Elefante, Ilaria Fabrizio, Alessandro Falappa, Viviana Fantozzi, Antonio Farina, Andrea Fernandez Tostado Ro.Diego Andrea, Andrea Ferrari, Sara Ferri, Simona Ferri, Fabio Fioramonti, Stefano Florio, Mariachiara Fornaro, Anna Maria Foti, Niccolò Fransoni, Alessandro Fusacchia, Emma Gamba, Francesco Gasperoni, Davide Gattini, Marta Giani, Claudia Giannetti, Valerio Giannini, Antonella Giudici, Mara Gramendola, Michele Grandinetti, Eleonora Graziani, L'Meese Hope Greaney, Dario Gubernari, Filomena Guglielmi, Jonathan Hammond, Stephen William Harris, Rowena Elizabeth Hogg, Marta Hontanon Del Olmo  

**Link**: [PDF](https://arxiv.org/pdf/2611.16956)  

**Abstract**: General-purpose language models are trained mostly on English: over open models we review, at most, non-English pretraining data makes up only a 12% share. Models built for European languages give them far more, but score well below general-purpose models of the same size. A common concern is that, in small models, languages compete for capacity. We present Lara Mini base, a model trained from scratch on 6T tokens in English and 44 other languages, which together cover all 24 official languages of the European Union. As training progresses, the 44 other languages together come to outweigh English. To give these languages enough capacity at low cost, Lara Mini base uses a mixture of experts architecture with 7.2B total and 0.6B active parameters. Among open base models with up to 2B active parameters, it achieves the highest average score across standard English, multilingual, math, and code benchmarks, ahead of models trained on 4 to 6 times as many tokens. Its margin is largest on multilingual benchmarks in the languages it supports and on math benchmarks. It supports contexts of up to 128k tokens. We find that our MoE architecture reduces capacity competition between languages. For a single user on one GPU, without quantization, it generates text at least 1.8 times faster than any model with a higher average score. We release the weights of Lara Mini base under the CC BY-NC-SA 4.0 license. In addition, we will release T-HellaSwag, a higher-quality machine translation of the commonsense reasoning benchmark HellaSwag into 110 language variants. 

---
# SPADE: Soft Prompts for Ai-text DEtection 

**Authors**: Alex Jiang, Didier Schwab, Chuyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.16929)  

**Abstract**: The easy access to Large Language Models (LLMs) has made AI-generated texts on the web increasingly ubiquitous, making AI-text detection a crucial task. Prompts greatly influence the generation process, but existing detectors focus solely on the generated texts --- supervised methods learn from labeled texts only, and zero-shot methods rely on the text's probability distributions --- ignoring the underlying generation process, which limits their robustness in open-world scenarios. We introduce SPADE: Soft Prompts for Ai-text DEtection, a novel framework for detecting AI-generated text that transfers better across datasets. Specifically, we measure the prompt's influence and use it as a discriminative signal. We characterize the prompt's influence by introducing a Pointwise Mutual Information (PMI) term that captures how much a prompt explains a text. Based on this quantity, SPADE trains one soft prompt per class to amplify the separation, and predicts the class whose prompt best explains the candidate text. Extensive experiments on several benchmarks against various detectors demonstrate the robustness and effectiveness of our method, achieving state-of-the-art (SOTA) performance on out-of-distribution (OOD) datasets. 

---
# Beyond Word Error Rate: A Failure Taxonomy of Streaming Dictation ASR 

**Authors**: Harras Mansoor, Muhammad Wisal, Hassan Bhatti, Masood Shakir  

**Link**: [PDF](https://arxiv.org/pdf/2611.16919)  

**Abstract**: Speech recognition (ASR) systems are usually compared by word error rate (WER) on recorded files, after punctuation, capital letters and number format are removed. In dictation, the audio is streamed live and the text goes into the user's message as written, so what matters is which mistakes the user has to fix. The test set, streamed to eleven ASR systems, comprises 106 scripted English dictations from one speaker (read in a quiet room, with a fan on, and whispered), 20 clips in each of eleven other language and dialect sets, and 400 recordings that join clips in two languages. Every error was sorted by a fixed rule into four levels: recognition, output formatting, language handling and, for the self-hosted models, audio dropped before the model. Systems with similar WER failed in different ways. Zoom Scribe and OpenAI's GPT-Live-Transcribe have almost the same WER on quiet English (7.4% and 7.3%), but 34.7% and 8.8% on whispered English. Zoom and both AssemblyAI models left out 26-28% of whispered words, and still 21-26% with their end-of-turn detection relaxed or off; seven other systems left out at most 3%. Qwen3-ASR's 32% came from the speech detection in our own deployment (6% when relaxed). Across 107 EARS speakers, the same four systems lost 7-31% of whispered words, while six others lost under 2%. With only English selected, Gemini, Deepgram and our self-hosted Whisper lost the non-English part of a two-language recording in 86-99% of recordings, against 33-46% with both languages selected, while GPT-Live-Transcribe, OpenAI's realtime Whisper and Meta lost it in at most 20%. Running Whisper behind a Silero voice activity detector that drops audio judged to be silence raised whispered WER from 10.1% to 51.4%. Most English results come from one read speaker. The code, the recordings, both references and every transcript are released. 

---
# Human or Synthetic? What a Fixed Budget Buys for Post-Training 

**Authors**: Shourik Banerjee, Ethan Mendes, Mohit Raghavendra, Wei Xu, Alan Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2611.16895)  

**Abstract**: Post-training of large language models typically combines supervised fine-tuning (SFT) with preference-based fine-tuning such as Direct Preference Optimization (DPO). Human-annotated and synthetically generated preference data differ substantially in per-sample cost, but prior work typically holds sample count fixed when comparing the two. In this work, we explicitly study the interplay between cost and performance by comparing the effect of post-training with human and synthetic preference data under a fixed, dollar-matched annotation budget. We train and evaluate multiple base model families across helpfulness and summarization tasks. We find that synthetic data is far more cost-efficient at low-to-moderate budgets, achieving high win rates with as little as \$200--\$400. On helpfulness, human data closes the gap only at budgets roughly two orders of magnitude larger; on summarization, it does not close the gap at any tested budget. We also find that human-annotated data varies in reliability: one widely used human preference source substantially underperforms another, and the gap persists even after standard remedies such as domain restriction and deduplication. 

---
# Pretraining Data Statistics Shape the Phases of Learning Entity Comparison in Language Models 

**Authors**: Yik Siu Chan, Jing Huang, Yanai Elazar, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2611.16882)  

**Abstract**: How does data shape language model (LM) behavior throughout pretraining? We investigate this question through a case study on entity comparison, e.g., Between France and Brazil, which country is larger?, across nine comparison tasks spanning five entity types. We begin with controlled experiments in which we train small LMs (163M parameters) on mixtures of natural text from pretraining corpora and synthetic task data. Our analysis identifies three distinct phases of learning: (1) an early phase where the LM selects entities by frequency, (2) a middle phase where the LM selects entities by position in the prompt (first vs. last mentioned entity), and (3) a late phase where the LM selects the entity that is the correct answer. We show that the emergence of these three phases is causally determined by statistical properties of the training data. With small amounts of task-specific synthetic data, we observe only the first two phases and the model fails to learn the task; with large amounts, the model jumps directly from the frequency-based heuristic to solving the task correctly. Moreover, changing the entity frequency distribution from a naturally occurring Zipfian distribution (a small number of entities are very common and the vast majority are rare) to a uniform one weakens the frequency phase and speeds up learning of the task. Lastly, we show that the three phases of training dynamics from our controlled experiments appear in the open source model OLMo-2 32B. Together, our findings demonstrate that properties of pretraining data are causal drivers of heuristic learning and show that small-scale synthetic experiments can predict training dynamics at larger scales. 

---
# Shortcuts or Memorisation? Revisiting Data Corruption in Natural Language Understanding with Large Language Models 

**Authors**: Aarne Talman, Stergios Chatzikyriakidis  

**Link**: [PDF](https://arxiv.org/pdf/2611.16843)  

**Abstract**: Pre-trained encoders fine-tuned on natural language inference (NLI) and other language understanding tasks keep much of their accuracy after the input is perturbed, e.g. by removing all nouns or shuffling the words. Does the same hold for large language models (LLMs), and if so, is it because they have seen the benchmarks during pre-training? In this paper we revisit these perturbation experiments with eight LLMs alongside fine-tuned encoders on GLUE and the three rounds of ANLI. Regardless of model size, every system that answers directly retains 36-72% of its above-chance performance without nouns on MNLI compared to only 14-29% on ANLI. Only a model that reasons before answering departs from this pattern. We also check contamination indicators (including exact $n$-gram counts in the public pre-training data of two of the models) and find no link between exposure and robustness. Our strongest evidence comes from 368 new, human-validated items that none of the models can have seen. All eight LLMs keep more of their performance without nouns on items written in MNLI's style than on items written without its cues. When the perturbation instead removes the function words carrying such cues, this advantage reverses. In light of these results, we argue that robustness to perturbation on familiar NLI benchmarks reflects dataset shortcuts rather than memorisation, and tells us more about the benchmark than about the model. 

---
# Outcome-Mass Alignment: Controlling Length-Dependent Trajectory Influence in Multi-Turn LLM Agent Reinforcement Learning 

**Authors**: Dawei Zhang, Yubin Huangfu, Jessica Fan, Boyu Wei, Likang Wu, Hongke Zhao, Ming He, Zhu Teng  

**Link**: [PDF](https://arxiv.org/pdf/2611.16816)  

**Abstract**: To keep long interaction contexts tractable, stepwise group-relative training for multi-turn LLM agents materializes each source trajectory into multiple response rows and valid tokens. This expansion creates an outer-accounting problem distinct from within-trajectory credit assignment: row-based outcome statistics can count one episode outcome multiple times, while token-level reduction can assign unequal aggregate objective mass across trajectories. We term these effects outcome-statistics multiplicity and objective-mass imbalance. We introduce Outcome-Mass Alignment (OMA), a two-component method that makes both accounting choices explicit. Outcome Alignment computes outcome statistics over unique source trajectories, while Mass Alignment controls aggregate objective mass after within-trajectory aggregation of optimizer-native token objectives. Neither redesigns the base optimizer's credit assignment, importance ratios, or clipping. Across matched ALFWorld GRPO comparisons at two model scales, where failed trajectories typically expand much more than successful ones, OMA improves matched-horizon endpoint success by 7.0-9.4 percentage points. Ordinary WebShop exhibits weak or reversed expansion-outcome coupling and correspondingly mixed effects, whereas in a success-oriented WebShop setting, WebShop-SOL, OMA improves success nAUC from 0.399 to 0.511. Component ablations and GiGPO and GSPO comparisons further support component separability and cross-optimizer compatibility. These results show that trajectory expansion always defines an implicit weighting rule, but its practical effect depends on how expansion couples to outcomes and optimizer-native updates. 

---
# Towards Green Text Generation: Memory-Based Speculative Decoding 

**Authors**: Antal van den Bosch  

**Link**: [PDF](https://arxiv.org/pdf/2611.16714)  

**Abstract**: Speculative decoding accelerates text generation with large language models by letting a cheap draft model propose tokens that a large verifier checks in a single forward pass. We introduce Olifant, a draft model built on a memory-based classifier. It predicts the next token from a four-token window in under 0.3 ms on CPU, requiring no GPU, no backpropagation, and no accelerator memory. We measure throughput and energy under two verifier regimes on consumer hardware. Without a KV cache, drafting is nearly free: a domain-matched Olifant model reaches 38.1% acceptance on EU legal text, delivering 2.43x throughput and 2.51x lower energy per token than the neural verifier model alone. With a KV-cached verifier, itself a ~10x energy saver, drafting costs matter: a TinyLlama draft model halves throughput and triples energy despite high acceptance, whereas Olifant still yields 1.25x throughput and the lowest energy in the study (93 J per 100 tokens, 14x below the uncached baseline). We further show that KV caching decouples time from energy: verification of longer drafts raises power draw, shifting the energy-optimal draft length below the speed-optimal one. Speculative decoding is not dead in the age of fast verifiers, provided drafting is nearly free. 

---
# Forecasting Sales Delay through Time-aware Analysis of Sales Events in Textual Reports 

**Authors**: Masaya Tsunokake, Yuta Koreeda, Yuhei Niwa, Terufumi Morishita, Teppei Inoue, Hiroaki Ozaki, Yasuhiro Sogawa  

**Link**: [PDF](https://arxiv.org/pdf/2611.16704)  

**Abstract**: Business-to-business (B2B) companies develop and deliver a one-of-a-kind product in response to a client's orders. To prepare resources for development beforehand, companies estimate dates when they will receive orders from clients in their sales process, called scheduled order dates. However, dates when the companies actually receive orders can differ from the scheduled order dates. This difference increases costs in the development. If the difference could be forecasted before receiving orders, companies could optimize their resource preparation. Towards reducing the costs, we introduce sales delay forecasting, a novel task of forecasting delay days from the scheduled order dates during sales process where sales representatives (sales reps) interact with the clients to receive orders. Sales delay forecasting is challenging because the sales process progress inherently involves uncertainty and depends on the client's willingness to order. To address this task, we propose confidence features which represent sales reps' confidence as a proxy measure of the client's willingness to order. To represent it, our method extracts dates of future sales events anticipated by sales reps from textual reports. To consider whether the confidence objectively looks reasonable, we also introduce status features which describe sales process status based on current date and past events. In experiments using real sales data, our method achieved a 16% lower mean absolute error compared with a feature-based baseline similar to a prior work. Moreover, our method outperformed non-feature-based baselines using large language models, indicating that it is crucial to explicitly create our features. 

---
# MuSeToM: Multi-Sensory Evaluation of Belief Revision with Textual Cues Across Six Senses 

**Authors**: Chunkit Chan, Kawai Chung, Sze Heng Douglas Kwok, Yuxuan Liu, Yauwai Yim, Tianshi Zheng, Haochen Shi, Qing Zong, Wong Kai Chung, Pak Hin Lau, Calvin Yang, Yifan Gao, Xiaoman Pan, Ginny Wong, Simon See, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.16662)  

**Abstract**: Theory of Mind (ToM), the ability to infer mental states in others, is pivotal for human social cognition. Existing evaluations of ToM in large language models (LLMs) mainly focus on static belief attribution and false-belief reasoning, leaving a critical aspect of social reasoning underexplored: can LLMs decide whether perceptual evidence is sufficient to justify another person's belief revision? To address this gap, we present MuSeToM, a human-validated, text-based benchmark for evaluating epistemic belief revision in ToM reasoning. MuSeToM contains 2,415 instances adapted from BigToM and OpenToM, with textual cues spanning six sensory categories, and tests whether models can determine when a protagonist should revise a belief and infer the resulting updated belief state. Using MuSeToM, we systematically evaluate state-of-the-art LLMs under direct and chain-of-thought prompting, revealing a pronounced dissonance: while models often detect that a belief may change, they struggle to identify the updated belief and frequently over-update on perceptually salient but epistemically insufficient cues. 

---
# Unibuc - NLP at "Voight-Kampff" Generative AI Detection PAN 2025 

**Authors**: Teodor Marchitan, Claudiu Creanga, Liviu P. Dinu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16612)  

**Abstract**: This paper describes the approach of the Unibuc - NLP team in tackling the "Voight-Kampff" Generative AI Detection task from PAN at CLEF 2025. In the Subtask 1: Voight-Kampff AI Detection Sensitivity we have explored different approaches of leveraging embeddings extracted from LLMs, using either classical ML models or further fine-tuning LLMs with a classification head for the downstream binary classification task. Our experiments showed that using a majority vote ensemble of classical ML models (Light GBM, XGB, Logistic Regression and SVM) is comparable with fine-tuning for downstream task achieving around 99.4% F1 score. For the Subtask 2: Human-AI Collaborative Text Classification we explored a way of combining features at different layers extracted using transformer backbone with layer-wise projection and attentive pooling which proved modest results. 

---
# EvenVLM: A Text and Multimodal Dataset for the Even Language with Vision-Language Fine-Tuning 

**Authors**: Daniil Makovetskii, Ivan Bondarenko  

**Link**: [PDF](https://arxiv.org/pdf/2611.16500)  

**Abstract**: We present EvenVLM, the first text and multimodal (VQA) dataset for Even (Lamut), an endangered Northern Tungusic language spoken by approximately 6,800 people in Siberia. Our dataset comprises 50,521 structured Even-Russian entries extracted from dictionary and phrasebook sources via GPT-4o Mini-based parsing, from which we synthetically generate 107,543 training examples spanning translation, grammar, reading comprehension, and semantic classification. To address tokenization inefficiency on Even's agglutinative morphology, we propose a three-step tokenizer expansion algorithm that adds Even-specific graphemes (eng, oe, o-umlaut), whitelisted vocabulary, and BPE merges, reducing average tokens per word by 27% (4.99 to 3.64). Using LoRA fine-tuning of Qwen3-VL-4B, we adapt the model for text tasks (0.53% trainable parameters, rank 32) and separately for visual question answering (rank 64, frozen vision encoder). Our best configuration achieves 11.4% exact match and a 13.6 weighted score on text benchmarks, 0.1283 chrF for sentence translation, and 65% accuracy on VQA (up from 0% for the base model and text-only fine-tuning). We release our datasets, evaluation benchmarks, and training pipeline to support further research on Even and other low-resource languages. 

---
# Where in the Input Does a One-Pass Decision Encoder Look? 

**Authors**: Mohammed Nayeem Khan, Allan Kevin T  

**Link**: [PDF](https://arxiv.org/pdf/2611.16477)  

**Abstract**: Laya is a small model that answers a multiple-choice question about a text in one pass: the question, the options (each with a name and a description) and the text form one sequence of at most 512 tokens, the window, in the English release. A longer text is silently cut at the end, and we measure what that costs. Test documents place a held-out row's text, the evidence, at a chosen token offset among distractor text; 177 items, five tasks, three checkpoints (released, short-row fine-tune, long-document fine-tune), four window sizes and 93,275 scored requests. When the evidence lies beyond the window, accuracy falls to 26.4% for the released checkpoint (27.2% for both fine-tunes), near the 27.0% of guessing and below the 35.6% of always giving the commonest label. Confidence gives no warning (AUROC 0.50 to 0.55 for telling cut from read requests; 0.5 = chance), and the runtime's truncation flag also fires on requests whose evidence was visible. The released checkpoint reads visible evidence badly once text precedes it: 38.5% pooled against 60% when the evidence comes first, so a larger window alone gains +6.2 points. Fine-tuning on long documents adds +27.6 points on visible evidence over the short-row fine-tune (95% interval +22.3 to +33.1) and nothing on cut evidence. That is an in-family result: training rows and test documents share generator and question types, and the main filler is of the training kind. With encyclopedic filler never seen in training the gain is +13.2 (+9.9 to +16.6), with rows of other tasks as filler -2.0. Among training-free rearrangements a 312-token window scan helps most (+11.9 points over the default for the released checkpoint) at six to eleven forward passes. Documents are synthetic, labels are LLM-made and one backbone is studied, so these diagnose this model, not long documents in general. 

---
# daVinci-Prover: Agent-Native Mid-Training for Formal Theorem Proving 

**Authors**: Wenhong Zhu, Xuefeng Li, Jiashuo Zhang, Yu Chen, Tiantian Mi, Qipeng Guo, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16433)  

**Abstract**: Advances in large language models (LLMs) are rapidly transforming formal theorem proving. Yet further progress remains constrained by the lack of a transparent and effective data recipe for systematically improving theorem-proving models. In this work, we argue that effective agentic provers require strong mathematical skill priors to ground theorem proving and agentic trajectories to enable long-horizon, interactive proof construction. To this end, we synthesize an agent-native 33.7B-token mid-training corpus covering decomposition, search, self-verification, and revision, together with verifier-grounded agentic trajectories. We further introduce the daVinci-Prover harness, which enables models to construct proofs and interact with a Lean verifier in an isolated workspace, and use it to collect approximately 36K high-quality agent trajectories for supervised fine-tuning. Our mid-training recipe yields an average improvement of 13.7 points over the matched Qwen3-8B SFT-only baseline across MiniF2F, FATE-M, and ProofNet. When applied to the Qwen3.5 series, our approach also yields strong performance across theorem-proving benchmarks and achieves near-saturating MiniF2F performance with memory compaction. The results highlight agent-native mid-training as an effective foundation for developing stronger interactive theorem-proving agents. 

---
# Saidzoda-Kazakh-EmbeddingGemma-300M: How Far One Epoch of Generated Questions Takes a Small Kazakh Embedder 

**Authors**: Tohir Saidzoda, Madina Abdusalomzoda, Abubakr Abdullozoda, Amin Jalilov, Faroiz Bobokhon, Aibek Shokanuly  

**Link**: [PDF](https://arxiv.org/pdf/2611.16399)  

**Abstract**: EmbeddingGemma-300M is a small open embedder, and used as released it finds the right passage first for 60.8% of the Kazakh Belebele retrieval questions. We present Saidzoda-Kazakh-EmbeddingGemma-300M, the same model fully fine-tuned for one epoch on 208,511 pairs: 166,263 Kazakh questions that our own language model wrote about source passages from encyclopedias, Wikipedia, news and books, each checked by a second model, plus English and Russian replay. On Kazakh Belebele, recall@1 rises from 60.8% to 78.0% and recall@10 from 76.3% to 91.4%. Our model scores above 161 of the 194 models with a published MTEB score on the same test, among them OpenAI text-embedding-3-large (47.1%). The other 33 score higher: the ten best are large multi-billion-parameter or closed models, reaching up to 92.2%, and at least 9 are open models of up to 0.6B parameters, the smallest with 339M. We also scored 6 Kazakh-specific embedders shared on Hugging Face with the same code in one run: our model (77.3% there) is ahead of 4 of them, including the two E5-base models of similar size (72.6% and 69.0%), and behind kaz-embedder (86.0%, three towers with 1,695M parameters in total) and a BGE-M3 fine-tune (81.0%); at recall@10 its interval overlaps those of its nearest neighbours on both sides. Our four-language model, trained with a capped share of the same Kazakh pairs, is 2.8 points lower at recall@1, with overlapping recall@10 intervals. The Kazakh model also scores higher than the released model on the Uzbek and Kyrgyz versions of Belebele, but our Tajik model, trained without any Turkic pairs, gains as much there, so these gains are not specific to Kazakh training. Our Kazakh Wikipedia test questions come from the same generator family as the training questions, so we rest the main claims on Belebele. 

---
# Beyond Compressing Everything: Contrastive Memory Learning for Selective Context Compression 

**Authors**: Sikui Zhang, Haowei Liu, Ziyun Gan, Chunfeng Yuan, Guangze Gao, Nian Liu, Boyu Cai, Bo Li, Xiaolei Lv, Jun Gao, Bing Li, Weiming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16384)  

**Abstract**: As large language models (LLMs) advance toward increasingly long inputs, soft context compression offers a promising approach to efficient inference by converting long inputs into compact memory slots. Under limited memory budgets, effective context compression remains challenging because task-relevant evidence varies substantially across queries, ranging from localized evidence to multiple distant spans or information dispersed throughout the full document. Instead of compressing everything indiscriminately, we propose CoMem, a query-guided selective context compression framework that learns memory representations conditioned on both the document and the query. CoMem develops Contrastive Compressor Learning to achieve query-sensitive yet context-preserving memory representations. To inject a strong prior over query-dependent evidence distributions, we devise a same-context hard negative mining strategy by pairing a single document with multiple queries, requiring the compressor to selectively retain information according to each query's distinct evidence needs. To address the restricted information flow imposed by causal masking, we further enable bidirectional attention among memory slots. Experiments demonstrate that CoMem consistently outperforms existing context compression methods across compression ratios on question answering and RULER-NIAH benchmarks. Our code is available at this https URL. 

---
# IsharaBART: An efficient BART-Based Framework for Low-Resource Bangla Sign Language Translation 

**Authors**: Aonmoy Das, Ananna Dev Aishi, Md. Fazlul Kader  

**Link**: [PDF](https://arxiv.org/pdf/2611.16359)  

**Abstract**: Millions of Bangla Sign Language (BdSL) users lack reliable automatic translation tools. Existing systems are limited by data scarcity, unidirectional design, and computationally expensive architectures that are not suitable for low-resource deployment. We introduce IsharaBART, a bidirectional generative framework for gloss-sentence translation in BdSL. IsharaBART uses Bidirectional and Auto-Regressive Transformers (BART)-based conditional language modeling with denoising learning objectives. We developed a parallel dataset of 1,000 professionally annotated gloss-sentence pairs, balanced across tenses and structures. We expanded the dataset to 4,831 pairs through four augmentation techniques: rule-based tense transformations, masked token substitutions, back-translation, and one-shot prompting. A compact denoising sequence-to-sequence model was pre-trained from scratch on monolingual Bangla text from six different corpora, then fine-tuned on the annotated and augmented datasets. We compare IsharaBART against four larger baseline models under identical conditions, where it achieves comparable or superior bidirectional performance. On the gloss-to-sentence (G2S) task, it attains Bilingual Evaluation Understudy (BLEU-4) score up to 51.7, and on the sentence-to-gloss (S2G) task, it achieves 61.40, outperforming multilingual models despite having fewer than 10% of the parameters. Ablation studies demonstrate that pre-training provides 62.5-90.4% faster convergence and 11.5-118% higher BLEU-4 than scratch. Low-rank adaptation (LoRA) reduces trainable parameters by 99.3%, but with 28.0-52.2% loss of BLEU-4 compared to full-parameter fine-tuning. 

---
# From Large Language Models to Efficient System One Models via Adaptive Computation 

**Authors**: Minh-Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.16321)  

**Abstract**: Transferring LLMs into System One decision models typically replaces the language modeling head with a decision head while retaining the full backbone, making every decision pay for all layers. Yet it remains unclear how much depth such decisions require. We find that decision information is concentrated in intermediate layers, with early layers building essential features and the final six layers removable without loss, while independently trained smaller models largely mimic the full model rather than developing specialized decision capabilities. These findings suggest that efficiency requires not simply smaller backbones, but shared computation across different depths. Building on this, we propose \method{}, which jointly trains a full model with prefix sub-models that share its final layers and decision head through self-distillation. A lightweight gate reads the hidden state at each exit to determine whether additional computation is needed, allowing one model to operate at multiple layers. Across seven decision benchmarks, our approach matches or outperforms independently trained models at every depth, while achieving better accuracy-efficiency trade-offs through adaptive computation. 

---
# SMIC: Shared Multiscale Input-Reference Conditioning across Transformer Layers for Natural Language Inference 

**Authors**: Xin Zhang, Victor S. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.16270)  

**Abstract**: Natural Language Inference models can be sensitive to localized lexical changes. We introduce Shared Multiscale Input-Reference Conditioning (SMIC), which constructs a single input-dependent multiscale reference and reuses it across Transformer layers. Each layer applies its own multiscale transformation to the shared reference and injects the resulting conditioning feature before the attention update. We instantiate SMIC in a token--context Transformer (TCT) and conduct from-scratch comparisons against decoder-based and Multi-Layer Perceptron (MLP) baselines across all three ANLI rounds. TCT-SMIC attains the highest observed ten-seed mean accuracy on all three ANLI rounds. The ablations further support the contribution of SMIC. We evaluate a real-valued Llama-style plug-in and extend deletion/insertion diagnostics to pretrained ModernBERT encoder and Llama decoder models with and without SMIC. Across the evaluated perturbation protocols, TCT-SMIC exhibits comparatively low pooled-attention redistribution and smaller signed accuracy changes. We further establish a finite-sample bound on average subset-mass redistribution in terms of the mean Jensen--Shannon divergence between the restricted and renormalized attention distributions. 

---
# Label-Efficient Document-Grounded Factuality Detection via Cross-Paraphrastic Invariance Learning 

**Authors**: Chao Chen, Shanshan Lin, Dongsheng Hong, Sibo Ju, Wenjie Yang, Sihong Xie, Xiangwen Liao  

**Link**: [PDF](https://arxiv.org/pdf/2611.16264)  

**Abstract**: Large language models (LLMs) can generate fluent claims that are unsupported by a given source document, motivating accurate and efficient document-grounded factuality detection. Existing LLM-based evaluators often require repeated inference and incur high computational cost, whereas compact factuality detectors are efficient at inference time but commonly rely on large labeled or synthetically generated corpora. We propose CPIL (Cross-Paraphrastic Invariance Learning), a two-stage Siamese framework that converts a small labeled set into relational supervision without acquiring additional factuality labels. CPIL constructs positive pairs from distinct label-preserving paraphrastic views of the same document-claim instance and negative pairs from opposite-label examples associated with the same source document. It further introduces quantile-aware semi-hard negative mining, which derives a boundary from the positive-pair similarity distribution and balances informative negatives with broader coverage. In the two-stage training procedure, Stage 1 learns an encoder from these task-aware pairs, and Stage 2 jointly fine-tunes the encoder and a linear classification head for binary factuality prediction. At inference time, CPIL requires no paraphrase generation, pair construction, or Siamese comparison and uses only one encoder-classifier forward pass. On the 11-task LLM-AggreFact benchmark, CPIL-2% and CPIL-3% achieve macro-F1 scores of 81.04 and 81.60 while using only 2-3% of the FactCG-released labeled pool. Both settings outperform all compared compact detectors and two vanilla LLM references. Ablations and sensitivity analyses support the contributions of task-aware pair construction, dynamic negative mining, and multi-pivot paraphrasing. 

---
# Low Language Transfer Speech-to-Text 

**Authors**: Husein Zolkepli  

**Link**: [PDF](https://arxiv.org/pdf/2611.16260)  

**Abstract**: A multilingual speech recogniser has heard much speech in its own languages. We ask how much of that it brings to a language it was never trained on, when it is fine-tuned on the little that language has. We score 28 open multilingual models untouched on Low-Language-TTS, 50 low-resource languages in 11 writing systems, and fine-tune 24 of them on at most 1,000 clips a language, 82 hours in all, with a dev set choosing every epoch. Fine-tuning lowers every model's macro character error rate (CER), by 16.6 to 53.1 points, to 9.5 at best, and a word language model takes three CTC models 1.2 to 1.4 points lower; WER is still 34.4 at best. On the 12 languages that no model lists, every model starts at 46 or worse and the best ends at 11.7. The same architectures from random weights stay 60 points or more behind, so the gain is transfer, and an encoder left untouched already gives a Whisper or Qwen decoder most of what it needs. It is the language, not the corpus: on FLEURS recordings that ours do not share, the median model keeps 79% of its gain. It costs the languages the models knew, 5.7 points of CER on English, Spanish, French and German in the median model. Low-rank adapters on 4% of the weights come within 1.1 points of tuning everything, and on 2 of 3 models forget less. Seq2seq and language-model decoders are done within three to five epochs, while a new CTC head is still improving at twenty or thirty. Data beats epochs: 25 clips a language take Qwen3-ASR 0.6B from 60.9 to 39.3, and at equal clips presented, ten times the data is worth 9.7 points. A model that first learned forty of the languages learns the other ten no faster. On a public Malay benchmark, untouched Whisper large-v3 scores 8.9 when it detects the language and 14.9 when told the clip is Malay, because it then translates the English that speakers mix in; trained on the Scicom telephony dataset it scores 6.0. 

---
# Indic-Task Gains and Regressions after Reinforcement Learning 

**Authors**: Sutanshu Saksena Raj, Daksh Maan, Avinash Benki, Parijat Chatterjee, Pasunuti Prasanjith, Purusharth Malik  

**Link**: [PDF](https://arxiv.org/pdf/2611.16228)  

**Abstract**: We run reinforcement learning (RL) with GRPO across four environments, starting from a supervised fine-tuned 30B-A3B mixture-of-experts model covering English and eight Indic languages. We compare the model before and after RL across 34 benchmarks in one evaluation campaign. MATH improves by 36.07 points, and most Indic benchmarks also improve. Yet XQuAD-Indic F1 falls 19.32 points as English answers rise from 8.7% to 91.7%; IndicQA falls 8.64, while English XQuAD gains 11.71 F1. A second RL run from the same supervised initialization adds an Indic environment together with language and repetition rewards. Compared with the released model, it improves both passage-answering and instruction-following benchmarks, along with preserving MATH performance. But it loses ground on reasoning and coding benchmarks. Several training choices changed together, so we cannot isolate the effect of any one of the reward terms. Script compliance remains unresolved. Adding a language reward did not ensure use of the requested script. Passage answering and script compliance therefore need direct evaluation, even when other benchmark scores improve. 

---
# We Are What We Remember: Mental-State-Driven Memory for Long-Term Character Dialogue 

**Authors**: Zhicheng Jia, Shuchen Wu, Zhuoshi Tong, Haisen Liao, Sicheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.16223)  

**Abstract**: Shared history in human dialogue is not merely a repository of facts, but the foundation of evolving relationships and mutual understanding. However, mainstream conversational memory systems rely on passive semantic similarity over external fact stores, frequently retrieving topically similar records while missing relationally pivotal experiences. Inspired by cognitive psychology's Self-Memory System (SMS), we propose MindMem, a mental-state-driven memory framework that asynchronously consolidates shared experiences into source-linked mental states of the character, the user, and their relationship. At each turn, a specialized mental-state reasoner (Mind) contextualizes these persistent states to anticipate latent user needs, form proactive intentions, and recover supporting episodic evidence that surface-level matching overlooks. To evaluate history-sensitive adaptation beyond factual recall, we introduce MemReal-Bench, a longitudinal benchmark grounded in month-long real-user interactions across dyadic and group scenes. By pairing Direct Question Answering (DQA) with Situated Memory Realization (SMR), it tests whether agents bridge the gap between knowing past facts and naturally using shared history, world rules, and relational trajectories. On MemReal-Bench, MindMem reaches 59.03% SMR and 82.58% DQA, exceeding the strongest memory baseline by 24.19 and 9.19 percentage points, respectively, and outperforming oracle target-memory injection on SMR. Code and dataset are available at this https URL. 

---
# Data-Efficient LLM Safety Alignment Guided by Interpretable Features 

**Authors**: Hongyu Li, Zhan Yu, Zhe Zhao, Kimmo Yan, Xiong Kuang, Renfen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16183)  

**Abstract**: Existing large language models typically rely on massive fine-tuning datasets for safety alignment, which results in higher costs for safety alignment and often leads to the problem of over-refusal. We propose Less is Safer (LIS), a compact yet highly effective dataset for safety alignment, constructed through interpretable feature-guided selection and adversarial synthesis. Specifically, we leverage Sparse Autoencoder (SAE) activations to efficiently select prompts that cover diverse and interpretable safety features. Then, we employ a multi-agent adversarial synthesis framework to generate harmless prompts that nevertheless elicit unnecessary refusals. LIS contains only 2.4K samples, yet outperforms much larger datasets, e.g., TRIDENT-Core (26K), TRIDENT-Edge (19K) and WildJailbreak (262K), in overall performance on four safety and five over-refusal benchmarks. Through ablation experiments, we demonstrate that neither safety data nor over-refusal data alone can achieve optimal results. We also used safety features from SAE, annotated by an LLM, for data filtering, but the results were inferior to those obtained using the pipeline proposed in this paper. 

---
# Temporal Context-Aware Spam Detection Using Transformer-Based NLP Models 

**Authors**: Fariz Zeynalov, Razim Mammadli  

**Link**: [PDF](https://arxiv.org/pdf/2611.16139)  

**Abstract**: Spam detection is usually treated as single-message classification, ignoring behavioral patterns that emerge across sequences of messages from the same sender. We study outbound SMS spam detection at the carrier level, where the goal is to predict at send time whether a carrier will flag a message as spam. Using a proprietary production database of 43,741 outbound SMS messages with UTC transmission timestamps and carrier delivery statuses, we report three sets of results. On an 8,749-message held-out test set, a TF-IDF + logistic regression baseline reaches AUC-ROC 0.8721, and a text-only DeBERTa-v3-base classifier reaches AUC-ROC 0.9011 and F1 0.8412. A deployed DistilBERT model augmented with inter-message temporal features reaches accuracy 0.8238, spam-class F1 0.8382, and recall 0.9128 at threshold 0.5 on a separate, class-balanced 20,000-message production evaluation set; these results are not directly comparable with the held-out test results. We also describe TemporalBertSpam, an architecture that encodes each sender's message history with a causal Transformer, and outline the same-split evaluation needed to measure its contribution. 

---
# Show Your Work in Swahili: Reinforcement Learning for Native-Language Mathematical Reasoning 

**Authors**: Jessica Lundin, Katherine Rosenfeld, David Adelani, Geoffrey Mosoti Nyakiongora, Halfan Ngowo, Sloan Coffin, Ralph Abboud, Guillaume Chabot-Couture  

**Link**: [PDF](https://arxiv.org/pdf/2611.16104)  

**Abstract**: We present Fikiria, a two-stage SFT+GRPO recipe for Swahili mathematical reasoning. Its reward multiplies a format gate by a weighted sum of answer accuracy and the ChrF\texttt{++} similarity of the Swahili reasoning to an in-language reference chain of thought, so that training targets the surface fidelity of the reasoning trace and not only the final answer. We train on a 2.3k-problem Swahili dataset created by translating GSM8K with GPT-4.1 and filtering with SSA-COMET. On AfriMGSM Swahili, Fikiria-AfriqueQwen-14B reaches 79.2\%. Under the same fixed prompt and strict answer-format scoring, GPT-4o reaches 68.0\%, partly because it often fails the required answer format; IrokoBench reports 78.8\% for GPT-4o using a best-per-model prompt, a figure numerically close to ours but not directly comparable. In a small human evaluation with co-author annotators, GPT-4o's outputs were preferred overall. ChrF\texttt{++} measures surface fidelity, not the validity of individual reasoning steps. 

---
# Privileged Information Does Not Guarantee Privileged Guidance: Auditing On-Policy Self-Distillation for Reasoning Models 

**Authors**: Haoyi Qiu, Akshat Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2611.16064)  

**Abstract**: On-policy self-distillation (OPSD) trains a student on next-token targets from a teacher, a frozen copy of the student that additionally receives privileged information, such as a worked solution, hidden from the student. Whether student gains reflect that information remains unclear. Taking the worked solution as privileged information, we audit four stages across three Qwen3 scales on mathematics: Source, whether the teacher can use the solution; Encoding, whether the resulting token targets are useful; Transmission, whether the student learns from the targets; and Expression, whether the student applies the learned behavior without the solution. With the solution, the teacher solves hard problems and recovers from student errors, yet the teacher does not reliably recognize errors, and its token targets do not favor correct trajectories. Repeated training makes successes within the student's reach more frequent rather than unlocking new ones, and corrections the student learns to predict are not reliably expressed in free generation. To isolate the solution's contribution, we vary its availability separately from teacher and student thinking modes. The largest gains occur when a teacher in thinking mode supervises a student trained and evaluated without an explicit thinking trace, and problem-only teachers often match or exceed privileged-solution teachers: the gain tracks the teacher's thinking mode rather than the solution, a pattern we call reasoning arbitrage. Accuracy gains alone therefore do not establish transfer of reasoning from privileged solutions: bottlenecks lie after Source, across Encoding, Transmission, and Expression. Future evaluations must measure what the solution adds to otherwise identical training. 

---
# Human-Likeness and Conversational Preference Diverge for Synthesized Agent Speech in Customer-Service Calls 

**Authors**: Jeffrey M. Girard, Antony D'Avirro, Jason Z. Zheng, Jacqueline R. Vertino, Benjamin Peloquin  

**Link**: [PDF](https://arxiv.org/pdf/2611.16051)  

**Abstract**: Listeners can judge an agent's speech on how human it sounds (human-likeness) and on how much they would want to talk to it (conversational preference). Listening tests rarely ask about conversational preference, so the relation between the two judgments is unclear. We compared them in paired-choice tests on 120 turns that actors improvised as customer-service representatives. On each trial, two agents spoke the same turn. In Study 1, both agents came from six commercial text-to-speech systems, each with eight voices. In Study 2, one was the actor's original recording and the other a system. Some systems ranked higher on human-likeness, and others on conversational preference. Listeners chose systems over actors in 19% of human-likeness trials but 36% of conversational preference trials (and 49% for the most preferred system). On both judgments, variation among a system's voices was about three-quarters of that between systems. Of three automatic naturalness scores, two predicted lower human-likeness, and none predicted conversational preference. In this setting, listeners tended to favor faster speech and fewer pauses between clauses on both judgments, and more pitch variation for conversational preference. Thus, measuring only human-likeness or testing one voice per system can misjudge which agents listeners would rather talk to. 

---
# What Roles Do System Prompts Really Assign? Validating and Steering Occupational Role Vectors in LLMs 

**Authors**: Jose L. Sampedro Mazon, Ilmi Yoon, Nadim Saad  

**Link**: [PDF](https://arxiv.org/pdf/2611.15989)  

**Abstract**: Applications built on large language models assign the model a role through its system prompt, intending to shape how it responds for their use case. We introduce a validation framework for extracting and injecting occupational role vectors, activation-level directions that capture how someone in a given occupational role would distinctively respond to a task or question. We validate these vectors through two complementary lenses, on Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. The interpretability lens asks whether each vector is a stable, separable, domain-consistent representation of its role: extracted vectors are highly reproducible across repeated runs and sampling temperatures, remain separable even for very similar roles, and agree more with themselves across in-domain and out-of-domain questions than with other roles' signal. The control lens asks whether injecting a role vector during generation makes the model express the role, and sustain it over a conversation, as well as a role prompt does. On Qwen, injecting the vector alone makes the model express the intended role substantially more often than unsteered, but at best matches a role prompt; on both models, adding the vector to the prompt strengthens the role in single answers but did not slow the role's decline over 30 turns. Together, the two lenses show why both are needed: out of domain the role is expressed only subtly, yet the activations still encode it; and a vector that is stable and separable can still barely steer the model's behavior. 

---
# Better Boundary Prediction Does Not Guarantee Better Structural Representation 

**Authors**: Michael Zhou, Markus Frohmann, Junghyun Min, En-Shiun Annie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.15971)  

**Abstract**: Punctuation marks often mark phrase boundaries, carrying linguistic information. Therefore, punctuation marks' abundance and feasibility as proxies of linguistic structure has led to suggestions that using punctuation marks to optimise on phrasal structure may help language models (LMs) improve their representations of argument, semantic, and syntactic structures. In this paper, we test the above claim in English across three primary LM architectures: decoder-only, encoder-only, and encoder-decoder LMs, by complementing traditional token prediction with punctuation restoration (PR). Our evidence, collected at the <1B parameter scale, suggests that optimising for boundary prediction does not necessarily guarantee higher-quality structural representation in decoder-only, encoder-decoder, and encoder-only LMs, failing to improve performance on downstream tasks that require understanding structure in linguistic input. 

---
# Is Spoken Hungarian Low-resource?: A Quantitative Survey of Hungarian Speech Data Sets 

**Authors**: Péter Mihajlik, Katalin Mády, Anna Kohári, Fruzsina Sára Vargha, Gábor Kiss, Tekla Etelka Gráczi, A. Seza Doğruöz  

**Link**: [PDF](https://arxiv.org/pdf/2611.15961)  

**Abstract**: Even though various speech data sets are available in Hungarian, there is a lack of a general overview about their types and sizes. To fill in this gap, we provide a survey of available data sets in spoken Hungarian in five categories (e.g., monolingual, Hungarian part of multilingual, pathological, child-related and dialectal collections). In total, the estimated size of available data is about 2800 hours (across 7500 speakers) and it represents a rich spoken language diversity. However, the distribution of the data and its alignment to real-life (e.g. speech recognition) tasks is far from optimal indicating the need for additional larger-scale natural language speech data sets. Our survey presents an overview of available data sets for Hungarian explaining their strengths and weaknesses which is useful for researchers working on Hungarian across disciplines. In addition, our survey serves as a starting point towards a unified foundational speech model specific to Hungarian. 

---
# ShadowBench: Exposing Lexical Anchoring and the Illusion of Forgetting in Large Language Models 

**Authors**: Sujan Maharjan, Lu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15925)  

**Abstract**: Large Language Models (LLMs) have emerged as primary interfaces for factual retrieval, yet current evaluation paradigms rely almost exclusively on explicit entity names. In this work, we demonstrate that LLM knowledge is fundamentally lexically anchored: models possess extensive factual information but find it significantly difficult to retrieve without explicit name tokens. To quantify this limitation, we introduce ShadowBench, a rigorously hardened, shortcut-resistant benchmark designed to evaluate Latent Entity Association via a novel Dual-Trait Association (DTA) task. Our evaluation reveals a pervasive "Shadow Gap" across all model scales$\unicode{x2013}$up to the frontier models GPT-5.4 and Claude-Sonnet-4.6$\unicode{x2013}$where removing lexical anchors causes performance drops of over 20%. Furthermore, we demonstrate that this lexical dependency exposes a critical vulnerability in AI safety. Applying ShadowBench to Machine Unlearning, we find that state-of-the-art algorithms (e.g., Gradient Difference, NPO) achieve only superficial lexical erasure. While unlearned models fail on direct queries, our novel Latent Entity Leakage Rate (LELR) metric reveals that for popular entities, models explicitly reconstruct the "forgotten" entity in their internal reasoning traces in over 74% of cases, utilizing residual shadow knowledge to solve associative tasks. Conversely, for niche entities, models retain high shadow accuracy (60-90%) via implicit attribute correlation despite 0% LELR. Ultimately, ShadowBench indicates that current unlearning paradigms create an "Illusion of Forgetting," demonstrating that existing methods act as superficial output filters and are not yet reliable for achieving true parametric erasure. 

---
# HeriKGQA: Grounded Knowledge-Graph Question Answering over Cultural Heritage Sensor Data 

**Authors**: Lingxiao Kong, Vasileios Arampatzakis, Vasileios Sevetlidis, George Pavlidis, Zeyd Boukhers  

**Link**: [PDF](https://arxiv.org/pdf/2611.15904)  

**Abstract**: Cultural heritage sites are increasingly monitored by sensor networks, but querying their data requires database expertise. Large Language Models (LLMs) offer a natural-language interface, yet over raw sensor records they lose track of which sensor, site, and time each value belongs to, and silently miscompute. We present HeriKGQA, a knowledge-graph question-answering method that confines the LLM to interpreting questions and phrasing answers, leaving computation to SPARQL and deterministic code. Questions are parsed into intents constrained by a heritage ontology aligning W3C SOSA/SSN with CIDOC-CRM, compiled into SPARQL, and answered from query results and temporal, spatial, and anomaly facts derived by a grounding layer, with error correction guarding each LLM step. On 500 questions over real data from five heritage sites, HeriKGQA reaches 99.0% accuracy, 18.2 points above the strongest of three baselines using the same LLM and receiving the same anomaly rules and answer conventions, with comparable answer quality while running almost four times faster. Without grounding, accuracy drops by 12.2 points and on anomaly questions from 99% to 66%. We report lessons from development and live deployment. 

---
# Computational Explanations of Semantic Universals 

**Authors**: Shane Steinert-Threlkeld, Jakub Szymanik  

**Link**: [PDF](https://arxiv.org/pdf/2611.15804)  

**Abstract**: Despite their differences, the languages of the world share many deep features when it comes to how they express meaning. These semantic universals are likely to reflect shared aspects of human cognitive, communicative, and social structure. Recent advances in cross-linguistic semantics and computational linguistics allow us to develop explanations of the sources of such properties. This Element presents a body of work doing just that: we explicate a framework for providing computational explanations of semantic universals and then describe several such explanations using principles of ease of learning and efficient communication. Empirical domains include quantifiers, color terms, modals, and indefinites. After providing these explanations, we provide a discussion of future directions for advancing the explanatory power of computational methods in semantic typology. 

---
# SO-Arena: Benchmarking the Capability Spectrum and Reliability of System One Models 

**Authors**: Lichen Li, Ziping Dong, Sicheng Feng, Qi Li, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.15777)  

**Abstract**: System One models promise fast, general-purpose decision-making without lengthy token generation. However, understanding their capability spectrum and decision reliability requires more than aggregate performance on current benchmarks. Existing evaluations often provide limited insight into the breadth of model capabilities and the reliability of their decisions. We introduce SO-Arena, a benchmark characterizing the capability spectrum and decision reliability of System One models across 11 domains and five capability families. For the capability spectrum, SO-Arena offers evidence-grounded evaluation through reliable sources and manual verification of reference labels against model-visible inputs; fine-grained capability assessment through domain-specific difficulty criteria; and realistic interactive evaluation, where models iteratively execute actions in sandboxes. Reasoning-intensive tasks further enable system one-two comparison. Beyond accuracy, SO-Arena assesses decision reliability through two complementary measures: decision stability under controlled option-set changes and missing-answer recognition when no valid option is available. Evaluations across 21 models reveal that the benefits of scaling and additional reasoning vary substantially across domains. More importantly, strong overall performance does not necessarily imply reliable decision-making: the best-performing System One model ranks only seventh in reliability. Together, these findings highlight the need to characterize capability spectrum and decision reliability as complementary dimensions, providing a systematic view of the strengths and limitations of System One models. 

---
# Where Is Uncertainty Observed? A Mechanistic Understanding of Uncertainty Quantification in LLMs 

**Authors**: Xiaoyu Liang, Virginia Ceccatelli, Aliai Eusebi, Tobias Braun, Nils Loose, Carlo Malagnino, Lorenzo Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2611.15735)  

**Abstract**: Knowing when a large language model's (LLM's) response can be trusted is critical for its safe use. Uncertainty quantification (UQ) can help LLMs recognize if their responses may be unreliable. Existing taxonomies organize UQ methods by uncertainty source or estimation mechanism. They leave unresolved the question of where uncertainty is observed in the model's computation. The answer matters because the observation point determines whether an estimator has access to enough information to recover a given target, and how well it can recover it. Drawing on mechanistic interpretability, we define four mechanistic observation lenses: the structural lens observes component updates, the representational lens observes hidden states, the distributional lens observes token probabilities, and the semantical lens observes distributions over meanings. We relate these lenses through the residual stream and order them by the information they carry, which gives a principle for selecting UQ methods for different targets. We survey existing methods through the four lenses. Furthermore, we distinguish evaluation targets, criteria, and metrics to clarify different UQ evaluation protocols. Finally, we identify open problems and research directions to connect uncertainty estimation to the internal computation and downstream use of LLMs. 

---
# Outside the Purview of the Reward: Capability Drift in Multilingual Post-Training 

**Authors**: Sutanshu Saksena Raj, Purusharth Malik, Pasunuti Prasanjith, Daksh Maan, Parijat Chatterjee, Avinash Benki  

**Link**: [PDF](https://arxiv.org/pdf/2611.15717)  

**Abstract**: Post-training can improve rewarded behavior while weakening capabilities users still need. We study two GRPO branches from a shared supervised checkpoint in a multilingual mixture-of-experts model. The released branch gains 36.07 points on MATH but loses 19.32 F1 points on XQuAD-Indic, while XQuAD-English improves under the same scoring rule. An independently trained sibling adds Indic instruction following and language-aware shaping. It recovers the regressing Indic task scores and preserves MATH, but does not restore script compliance and trades off competition mathematics and code. In a later round, judge training is followed by a sharp extraction decline from the Indic branch; reward-free distillation from its earlier checkpoint restores most of the lost F1. A MATH score decomposition associates most of the measured gain with increased boxed-answer emission; accuracy among boxed answers changes little. These results show why benchmark scores should be read alongside the response forms they reward. 

---
# Dynamic Block-Online Streaming ASR for Low-Resource Agglutinative Code-Switching Speech with Morphology-Aware Evaluation 

**Authors**: Kazi Rafat, Afifa Imran, Md. Ismail Hossain, Md. Romzan Ali, Fuad Rahman, Sifat Momen, Shafin Rahman, Nabeel Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2611.15691)  

**Abstract**: Low-resource, agglutinative languages challenge low-latency ASR, as fixed lookahead windows fail when grapheme order diverges, suffixes cross code-switch boundaries, or code-switching data is imbalanced. We present the first comprehensive study of streaming ASR for intra-sentential Bangla-English code-switching, identifying a mismatch between strict causal streaming and long-range morphological dependencies. We introduce a Dynamic Block-Online framework with VAD-aligned inference that balances semantics, latency, and context, enabling global bidirectional attention to preserve morphological integrity. To mitigate data scarcity, we propose Script-Anchored Loanword Injection, augmenting training text with English loanwords to model phonotactic transitions. We introduce CS-WER to diagnose switch, root, and suffix errors, and show that flexible latency budgets enhance recognition of agglutinative and mixed-language speech, with successful transfer to menstrual-medical domains. 

---
# ALMeD: Medieval French Gold-Standard 

**Authors**: Ragini Menon, Sabine Tittel  

**Link**: [PDF](https://arxiv.org/pdf/2611.15661)  

**Abstract**: Medieval French is one of the Romance vernacular languages with a significant history of transmission and an enormous influence on other vernacular languages of the time. The sources convey valuable knowledge about all aspects of culture. The study of Medieval French has, over the last few years, increasingly benefited from NLP-based methods. There have been numerous efforts to parse medieval French; however, these efforts use corpora comprising legal texts and -- primarily -- literary texts as their foundation and training data. This means that, except for juridical resources, scientific literature is overlooked.
We present ALMeD, an annotated, semantically rich corpus (work-in-progress) of medical and surgical treatises produced in medieval French, i.e., Old and Middle French, including Anglo-Norman. Our semantically disambiguated gold standard goes beyond existing resources, focusing on medical terminology, and accounts for morphological development, evolving syntactical structures and the lack of spelling normalisation in these languages. 

---
# Can Agents Relive Past Experiences? Diagnosing Event-Affect Decoupling in Affective-Episodic Retrospection 

**Authors**: Tingna Xie, Ming Wang, Shi Feng, Yongkang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15655)  

**Abstract**: As memory-augmented agents increasingly support long-term and personalized interactions, reliable recall of emotional experiences while preserving their associated affect becomes necessary. However, existing evaluations usually assess emotion accuracy and factual memory grounding separately, overlooking event-affect correspondence. In this work, we formulate Affective-Episodic Retrospection (AER) as a task that explicitly evaluates the recovery of past events, associated affects, and their correspondence, and define Event-Affect Decoupling as a failure mode where the correct emotion is recalled but associated with the wrong episode. To diagnose Event-Affect Decoupling, we propose EM-TRACE, a stage-wise framework that decomposes AER into retrieval, affect recovery, and realization. On a controlled GoEmotions-based benchmark, Emotion-only retrieval achieves 28.65% affect recovery but retrieves the target episode for only 4.44% of queries, yielding a UAR of 24.21%. Further analysis reveals that event retrieval and affect assignment introduce distinct binding failures. These findings show that reliable AER requires grounded recall of emotional experiences, where episode retrieval, affect recovery, event-affect binding, and generation quality should be evaluated jointly. Our code is available at this https URL. 

---
# Differential Backpack Language Models: Sense Inhibition via Signed Contextualization Weights 

**Authors**: Konstantinos Skianis, Giannis Nikolentzos, Polykarpos Meladianos, Michalis Vazirgiannis  

**Link**: [PDF](https://arxiv.org/pdf/2611.15642)  

**Abstract**: Backpack language models predict with a strictly linear combination of non-contextual per-word sense vectors, weighted by non-negative contextualization scores. We show that non-negativity conflates two properties: the linear readout, which carries the interpretability, and the conic combination, which merely restricts it. We relax the latter by computing the sense weights with differential attention, a difference of two softmax maps with a learned per-sense gain, so context can actively inhibit a sense; we call the result a differential Backpack. At equal parameters on WikiText-103 this cuts the Backpack's perplexity gap to a Transformer by 31%, improves over the softmax Backpack at every sense count, persists with a shrinking margin up to 350M parameters on OpenWebText, and is localized at the sense layer by ablations and controls. The learned inhibition is directly interpretable (a financial context drives the river sense of bank to -0.50), and the preserved per-sense attribution predicts counterfactual removals exactly. We tested a number of combination rules, showing the linear interface is nearly free. Making the readout nonlinear gains only 0.43 perplexity points while invalidating two thirds of the decomposition. The differential Backpack's senses are also better representations, improving zero-shot sentence similarity by up to 33%. 

---
# ChJGBench: A Comprehensive Benchmark for Chinese Legal Judgment Generation with Diagnostic Evaluation 

**Authors**: Yinghao Hu, Linggang Kong, Qian Xiao, Yuting Huang, Yunlong Zhang, Yujie Zhang, Yiquan Wu, Fei Duan, Kaisong Song, Jun Lin, Kun Kuang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15639)  

**Abstract**: With the rapid development of LLMs, increasing attention has been paid to whether they can reliably generate complete and legally valid judicial documents. This calls for a benchmark capable of evaluating whether models can accurately apply statutes and resolve outcome-critical issues, including claims, charges, penalties, and monetary obligations. However, existing benchmarks typically focus on isolated legal tasks or coarse-grained evaluation. We introduce ChJGBench, a benchmark for generating the reasoning and dispositive sections of Chinese judgments across civil, criminal, and administrative cases. Built from 3,462 real-world judgments, ChJGBench provides a case-type-aware diagnostic evaluation framework comprising 13 metric types and adopts a hybrid paradigm. Experiments reveal no uniform advantage for legal-domain models over general-purpose models, with model strengths varying across diagnostic dimensions. Performance is generally stronger on criminal cases, while civil and administrative cases continue to present substantial challenges. 

---
# When Local Definitions Are Missing: Stable Errors in Language Models 

**Authors**: Zengqing Wu, Chuan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.15597)  

**Abstract**: Contracts, technical manuals, and online communities often give familiar words special meanings and define them in one place. A language model that reads a passage without that definition may use the usual meaning and give an answer that sounds right but is wrong. We study this with multiple-choice questions whose correct answers depend on a local definition. The questions come from a synthetic vocabulary, commercial contracts, Reddit community glossaries, and the documentation of a decentralized exchange. We ask three language models each question with and without the definition. Without using the answer key, we then test whether their repeated answers and stated confidence reveal their own errors. Separately, we give a small open model the answer key and test whether it can predict which questions larger models get wrong. We report three findings. (1) Familiar words that a source redefines cause the most errors. Models often repeat the same wrong answer, and neither repeated answers nor stated confidence reliably flag it. (2) The source definition fixes most errors on synthetic questions, but a definition drawn from the model's own prior knowledge does not. In real contracts, the source definition fixes only about half of the errors. (3) Given an answer key, a small model can rank questions by how likely larger models are to get them wrong. These results call for tests that check answers against the full source, including its local definitions. 

---
# VWEN: A Curated Small Language Model for Vietnamese Paraphrasing 

**Authors**: Sang Quang Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.15587)  

**Abstract**: We present VWEN, a 0.8B-parameter model for Vietnamese paraphrasing at a requested level of change and for five related tasks: generating several paraphrases, simplification, gist, register transfer and paraphrase judgement. VWEN is Qwen3.5-0.8B fine-tuned on about 7k samples chosen by a four-stage pipeline: rules and deduplication, a bidirectional entailment filter, a divergence score that assigns low and high paraphrase levels, and selection for coverage and instruction-following difficulty. We compare it with paraphrase-only recipes from the same pipeline and with training on the ViSP corpus under matched and larger step budgets. On high-divergence paraphrasing, VWEN reaches a success rate (requested level reached and meaning preserved) of 0.61 against 0.46 for the ViSP model at matched compute, a difference that holds on inputs resembling ViSP. Given four times more training examples, the ViSP model closes the gap on such inputs but not on questions, which ViSP lacks and which it often turns into statements. Across all comparisons the entailment filter is the most effective stage. Training on paraphrases alone erases abilities of the base model that the multi-task data preserves. 

---
# Can We Make an LLM Dumber, or Is It Just Playing Dumb? An Empirical Study of Believable Performance Throttling 

**Authors**: Chengguang Gan, Yunhao Liang, Hanjun Wei, QingHao Zhang, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2611.15568)  

**Abstract**: A benchmark score is commonly read as a proxy for model capability. We study the inverse problem: can a model be driven to a target score, and do its failures resemble those of a naturally weaker model? We operationalize believable performance reduction along three axes: target calibration, item-difficulty structure, and error-distribution similarity to a real weak reference. We introduce CapLadder, an IRT-calibrated suite of 1,985 items spanning mathematics, code, multiple-choice reasoning, and tool use. A simple weak-teacher LoRA dial reaches unseen target levels on reasoning with a 1.9-point held-out calibration error, whereas prompting, temperature, and weight noise cannot combine comparable range and precision. Yet on multiple choice, the dial converges to an answer-token attractor: up to 97% of its newly introduced errors land on one letter, with the attractor changing from D in Qwen to A in Llama. Logit interventions identify a dominant near-constant shift: subtracting the estimated shift on held-out items recovers 52 to 59% of lost accuracy and removes the dominant attractor, while option permutation confirms that the bias tracks token position rather than option content. With high inter-annotator agreement ($\kappa=0.94$), human-coded error-category distributions nonetheless match the weak reference on reasoning (JS = 0.00), showing that local plausibility and aggregate detectability differ. Structured tool calls exhibit more natural degradation in the Qwen family (JS = 0.09), while a globally calibrated dial transfers poorly across benchmarks (14.2-point error on two usable benchmarks). Precise score control, therefore, does not imply a portable or believable reduction in skill. 

---
# Faithful in Steps: Improving Generalization and Citation in RAG via Query Decomposition 

**Authors**: Yue Liu, Zhongying Ru, Shimin Di, Jipeng Zhang, Ruiyuan Zhang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.15538)  

**Abstract**: Retrieval-augment generation is a prevalent strategy to mitigate hallucinations of LLMs. The attributable RAG (RAGQ) generates quotes for its answers. The quotes indicate which input contexts support the RAG to derive the answers, enhancing the answer's verifiability and trustworthiness. However, existing RAGQs exhibit significant degradation when dealing with questions that require multi-hop reasoning and multi-modal understanding, suffering from over-citation, implicit entity identification failure, and poor generalization. In this paper, we propose a novel RAGQ framework, namely QDRAG. QDRAG breaks down the input question into atomic subquestions to identify the implicit entities. Then, the reranker prunes context distractors to eliminate the downstream over-citation. To facilitate query decomposition, we propose two zero-shot approaches: QD-C and QD-R, which guide the QD MLLM to decompose the question based on context knowledge and retrieval rewards, respectively. One interesting finding is that finetuning on the QD task shows better generalizability compared to directly finetuning on the downstream RAGQ task. Experiments on four multi-modal QA benchmarks demonstrate QDRAG's efficacy in grounding answers and generating faithful citations. The framework significantly outperforms all the baselines on both in-domain and out-of-domain tests, even surpassing Gemini-Pro. 

---
# AutoOntic: Automatic Ontology Induction from Text for Knowledge Graph Construction 

**Authors**: Alla Chepurova, Aydar Bulatov, Mikhail Burtsev, Yuri Kuratov  

**Link**: [PDF](https://arxiv.org/pdf/2611.15467)  

**Abstract**: Graph-based retrieval-augmented generation depends on the quality of the knowledge graph (KG) it retrieves over, yet schema-free LLM-based extraction pipelines typically produce flat graphs: deduplicated entities and relations with no type hierarchy and no domain/range constraints. We present AutoOntic, a pipeline that automatically constructs an ontology from text by canonicalizing open-vocabulary entity types and relations, inducing a hierarchy over the canonicalized types, and mining relation-level domain and range constraints. Its hierarchy induction procedure prioritizes types using corpus-derived relation-argument profiles and places them through local, LLM-adjudicated parent/child checks, allowing newly inserted types to trigger reassignment of existing parent-child links. The resulting hierarchy constrains entity deduplication and supports the induction of relation-level domain/range constraints. For intrinsic evaluation, we introduce a reference-based protocol that classifies induced ontology types as exact, broader, narrower, or unrelated to reference domain and range classes. On the Wikidata-TekGen and DBpedia-WebNLG subsets of Text2KGBench, over 90% of AutoOntic's induced endpoint types are semantically aligned with the reference classes across two LLM backbones: about half match exactly, while most remaining differences concern granularity. On MINE, knowledge graphs organized using an ontology built by AutoOntic achieve 88% retrieval accuracy. AutoOntic is available at this https URL. 

---
# SimRec: Learning to Search, Ask, and Recommend from Simulated Interaction 

**Authors**: Yaxin Zhang, Xingyu Wu, Yuchen Yan, Yuqin Dai, Hong Zhao, Lixing Shen, Zhan Chen, Jian Shao, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.15426)  

**Abstract**: LLM-based recommendation agents can search catalogs, inspect products, ask users, and revise rejected recommendations. However, these behaviors are often governed by hand-designed workflows, while existing training methods optimize isolated components such as query rewriting or list ranking rather than the interaction policy as a whole. We present SimRec, a reinforcement learning framework that trains the full interaction policy of a tool-integrated recommendation agent. SimRec introduces two components. A task-anchored user simulator provides feedback on every recommendation: conditioned on the current task need rather than on user profile, and it can be an external LLM or the recommender itself under self-play. A task-aligned reward optimizes the whole trajectory with GRPO, crediting natural retrieval quality and valid tool use alongside the final hit, with training-only target injection to handle sparse hits over large catalogs. On ESCI and Amazon C4, SimRec improves the NDCG@100 of a Qwen3-4B agent from 32.23% to 35.42% and from 22.94% to 24.72%}, outperforming training-free agents on far larger backbones and task-aligned baselines on NDCG. Analysis shows that the trained agent verifies candidates, uses preferences, and recovers from rejected recommendations far more often than its untrained counterpart. Our code is available at this https URL . 

---
# AraCrypto-ML: A Multi-Label Corpus of Communicative Functions in Arabic Facebook Cryptocurrency Discourse 

**Authors**: Mabrouka Bessghaier, Mohamed Rashad Rouine, Md. Rafiul Biswas, Youssef Ayadi, Wajdi Zaghouani  

**Link**: [PDF](https://arxiv.org/pdf/2611.15408)  

**Abstract**: Arabic cryptocurrency discourse on social media is communicatively heterogeneous: a single post may report prices, explain protocols, promote investment, warn of scams, or invoke Islamic jurisprudence. Yet no annotated resource captures this functional diversity. We introduce AraCrypto-ML, a 10,000-post multi-label corpus of Arabic Facebook cryptocurrency discourse (2014-2024) from 2,898 public pages using a set of related keywords. Collected posts are annotated into six categories: Market Commentary, Educational, Promotional, Skeptical-Warning, Religious Framing, and Other/Unclear, using Fanar-2-27B under a prompt-only multi-label protocol. We validate 1,000 posts against adjudicated gold labels (macro kappa = 0.74) and apply Rogan-Gladen calibration to correct labelling bias. The two annotators agree with each other at macro kappa = 0.62 before adjudication, below the model's agreement with either of them (0.70 and 0.69). Reaction profiles differ strongly by function: Love is about 70% of reactions per post on Educational and Promotional posts but about 33% on Skeptical-Warning, which draws the largest Haha share (about 39%). 

---
# Towards Better Legal Reasoning LLMs: Signal Balancing and Reward Scheduling 

**Authors**: Yinghao Hu, Qian Xiao, Yuting Huang, Kaisong Song, Yiquan Wu, Ang Li, Junjie Cao, Jun Lin, Kun Kuang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15346)  

**Abstract**: In the legal domain, LLMs can improve reliability and interpretability by generating explicit reasoning traces. However, training LLMs to produce high-quality reasoning traces remains challenging. Existing supervised fine-tuning methods struggle when reasoning traces are significantly longer than final answers, as the learning signal for the answer becomes diluted. Meanwhile, reinforcement learning methods such as Group Relative Policy Optimization also have several limitations, including costly reward design, performance plateaus, and reward hacking. We therefore propose a two-stage training framework. In Stage I, JurisCoT-SFT employs a length-normalized dual objective to balance learning signals between reasoning traces and final answers. In Stage II, Lifecycle-Aware Backtrackable Policy Optimization dynamically evaluates and retires auxiliary reward signals based on their impact on final evaluation objectives, enabling efficient reward utilization. Our framework substantially improves legal reasoning performance on relevant tasks selected from multiple legal benchmarks, including ChJGBench, LawBench, and LegalBench. Experimental results demonstrate consistent improvements in both reasoning trace quality and final answer accuracy, while our models remain competitive with larger models. 

---
# Janus-MOPD: Tracing the Student Learning Dynamics in Multi-Teacher On-Policy Distillation 

**Authors**: Zongji Yu, Ming Zhang, Dongcheng Zhang, Jiawei Zhou, Haotian Pang, Yiliu Sun, Xiang Wang, Youbang Sun, Chaochao Lu, Chen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2611.15341)  

**Abstract**: Multi-domain capabilities are critical for practically applicable Large Reasoning Models (LRMs). Therefore, Multi-teacher On-Policy Distillation (MOPD) has become a crucial part of the post-training process, integrating the capabilities of multiple domain-specific teacher models into a single student model. However, the student usually fails to acquire multi-domain capabilities simultaneously due to undesirable domain heterogeneity. Existing works have address this problem in view of temporal uneven efficiency and spatial hard allocation in external multi-teacher supervision. While in broader multi-domain training, varying degrees of performance degradation is still largely unexplained. In this paper, we shift the focus on how the student learns and integrates multi-domain capabilities internally, and then propose \textbf{\textsc{Janus-MOPD}}, a general learning dynamics framework of geometric representations for LRMs. Specifically, our main focus is on the internal performance of multi-domain capabilities on both temporal and spatial dimensions within the student. Temporally, our study shows that the student exhibits a Expansion-Consolidation-Optimization (ECO) three-phase learning dynamics, together with inherent limitations in learning efficiency and scaling. Spatially, we summarize two types of subspace locking within and across domains by tracing the representation space of the student, while explaining its clear learning directions but inherent cross-domain conflict. Therefore, \textsc{Janus-MOPD} yields a unified learning dynamics formulation, a geometric tracing methodology, and systematic findings of how the student learns and integrates multi-domain capabilities. Overall, we provide a \textsc{Janus} perspective to internally trace the continual evolution of the multi-domain capabilities within the model itself. 

---
# Knowledge in the Harness, Reliability in the Weights: Teacher-Free Co-Evolution of an LLM Agent 

**Authors**: Yujia Zhang, Xinqi Ren, Shixiong Zhou, Jia Meng  

**Link**: [PDF](https://arxiv.org/pdf/2611.15268)  

**Abstract**: Small open models suit private customer-service deployments but trail frontier models in two ways: they miss what the written policy requires in a given situation, and they act inconsistently on what they know. We study whether such a model can close both gaps from its own deployment logs and their reference outcomes, without a stronger teacher model. Our starting point is a Gemma 4 12B agent that reads its domain policy from on-demand skills rather than from its system prompt. In each round, the deployed system serves a fresh batch of tasks; the base model diagnoses failed conversations and writes rules into the skills, and a rule survives only if it passes code checks, counterexample self-review, a paired retest, and a noise-calibrated validation gate. Reinforcement learning (GRPO) then trains a LoRA adapter on complete multi-turn, tool-using conversations. On the retail and airline domains of $\tau^2$-bench, without training or selecting on official tasks, the agent after two rounds improves on this starting point by 12.5 pass$^1$ points in retail and 19.5 in airline, and by 18.4 and 22.0 pass$^4$ points. It also exceeds the benchmark's reference agent, which runs the same model with the whole policy in its system prompt, by 11.8 and 8.0 pass$^1$ points and by 22.8 and 16.0 pass$^4$ points, closing most of the gap to three frontier models under the same protocol. A $2\times2$ ablation shows that the rules rescue tasks the agent never solved and carry policy knowledge that RL with the rules in context does not internalize, whereas the trained weights make solved tasks reliable. 

---
# A Closer Look at Language Model Introspection: Probing Information and Reporting 

**Authors**: Dustin Rubin, Vincenzo Timmel, Xufeng He, Aritra Saha, JooMan Kim, Vijay Madisetti  

**Link**: [PDF](https://arxiv.org/pdf/2611.15267)  

**Abstract**: One way to study language-model introspection is to test whether models can identify where their activations have been perturbed. We train a simple linear classifier to identify the perturbed sentence from the model's activations, then compare its predictions with the model's own answers across pretraining and injection strengths. This tests whether the model is being asked to report information that is already easy to recover. If so, successful localization may be less demanding than it first appears and may not extend to the broader uses that motivate introspection research, such as faithfully explaining a model's reasoning or helping researchers assess the reliability of its answers. With an injection four times the typical activation norm at layer 0 and probe readout at layer 24 in OLMo~3 32B, the supervised probe reaches 98.3\% accuracy after 8.4 billion training tokens, while the model's zero-shot answer remains at chance (11.1\%). At the released base checkpoint, these accuracies reach 99.1\% and 89.6\%, respectively. Across 7B and 32B, location is already decodable in randomly initialized networks, showing that the architecture can support these traces without pretraining. In these experiments, location information lies close enough to the surface for a simple linear probe to recover it, so successful reporting need not imply broader access to the model's internal computation. 

---
# Where Language-Model Vocabulary Rose on Stack Exchange 

**Authors**: Fengming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15172)  

**Abstract**: Stack Overflow banned answers generated by ChatGPT five days after the model's release, and many other Stack Exchange sites announced similar rules. I measure the vocabulary of the answers posted afterwards, 3.8 million answers on 170 sites from 2019 to 2024. A fixed list of 56 words that language models overuse rose about 14 percent above its pre-2023 trend, while the highest of 1,000 frequency-matched lists of ordinary words rose about 7 percent. In a screen of 15 site traits measured before the release, the strongest predictor of where the rise happened was the share of answers that received a body edit within 30 days of posting. On the least edited third of sites the list rose about 29 percent relative to matched controls, and on the most edited third it did not change. The gradient is also present among answers that were never edited. It follows the share of answers edited by their own authors, and the share edited by others adds little once that share is held fixed. Use of the model is unobserved, so these results cannot separate lower use on sites with a high self-edit share from any effect of editing on what the vocabulary records. Any vocabulary-based estimate for the network as a whole therefore pools sites where the marker vocabulary rose sharply with sites where it did not change. 

---
# Read Once, Answer Many: Parallel Prompting for Shared-Context LLM Inference 

**Authors**: Zekun Zhao, Jeffrey Flanigan  

**Link**: [PDF](https://arxiv.org/pdf/2611.15130)  

**Abstract**: LLM workloads sometimes ask many independent questions about the same long document, as in document QA, data labeling and LLM-based evaluation. Prefix caching removes the redundant prefill shared by these queries, but each decode step still reads the document's key-value (KV) cache once per question, so decoding is bound by memory bandwidth and scales with the number of questions. We present Parallel Prompting, which decodes P questions as P independent streams of a single request all at once, fixing this issue. In our method, every step reads the shared document once for all P streams, and a stream-id attention mask, evaluated inside the attention kernel, keeps each answer conditioned only on the document and its own question. We implement Parallel Prompting in SGLang by adding about 80 lines to three existing files and a small engine package; the mask needs no dense tensor, and every decode step has a fixed shape that is captured as a CUDA graph. With CodeLlama-7B, 8k-16k-token documents and 64-256 questions per document, Parallel Prompting is 3.7x faster than stock SGLang with its FlashInfer backend on an A100 and 5.9-6.0x faster on an RTX A6000, on two SGLang releases, including the most recent release. Timed on identical inputs, it is faster than stock SGLang in all 144 runs (2 GPUs x 2 releases x 6 workloads x 6 input sets). Code: this https URL 

---
# What Does Each Part of Speech Need? Contexts, Subspaces and Dimensions in Static Word Embeddings 

**Authors**: Jintang Xue, Chengwei Wei, Yun-Cheng Wang, C.-C. Jay Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2611.15114)  

**Abstract**: Static word embeddings give a word one vector, ignoring its different functions as noun, verb or adjective. Training on part-of-speech-tagged text gives each function its own vector, at the cost of a larger table. Using nine controlled corpora, we ask which context words each part of speech needs, whether it needs its own subspace, and whether the number of dimensions a word keeps should follow its part of speech or its frequency. The tags raise verb similarity on SimVerb-3500 at every dimension from 50 to 300 but help nouns much less. Added to verb-only text, adverbs help verbs more than as many random words; adjectives help nouns only slightly. At a fixed dimension, per-part-of-speech PCA bases preserve verb and adjective similarity better than a global basis or frequency-band bases. Nouns need more dimensions than verbs only among frequent words. We therefore allocate dimensions by frequency with these bases: at a third of the original size, only rare words lose dimensions, SimVerb-3500 and SimLex-999 nouns match the uncompressed table, and verb-to-noun retrieval stays within 0.02. Part of speech decides what a word's vector should be; frequency decides how much of it to keep. 

---
# FIBER: Bayesian Fusion of Fast-DetectGPT and IRT-Informed Detector Evidence for LLM-Generated Text Detection 

**Authors**: Yan Li, Guangsheng Bao, Zhipeng Chen, Guandong Zhang, Yanbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.15058)  

**Abstract**: Detecting large language model (LLM)-generated text remains challenging under domain and generator shifts. Supervised detectors may generalize poorly, while zero-shot methods often rely on individual statistical signals without leveraging historical detector behavior. We propose FIBER (Fast-DetectGPT with IRT-informed Bayesian Evidence Recalibration), which integrates continuous Fast-DetectGPT scores with auxiliary detector outputs through Bayesian evidence fusion. Item response theory (IRT) models historical detector-correctness patterns, providing class-conditional detector-output likelihoods through marginalization over latent text representations. These likelihoods are combined with Fast-DetectGPT score densities and a historical class prior for prediction. When historical Fast-DetectGPT scores are unavailable, a variant called FIBER-Stream estimates class-specific score distributions from an unlabeled batch. On a 32,000-text multi-domain benchmark, FIBER achieves the highest accuracy (0.620) and improves mean AUC over Fast-DetectGPT (0.664 versus 0.600) under five-fold cross-validation. FIBER also consistently improves AUC across four individual domains. Under out-of-distribution transfer, FIBER-Stream achieved an AUC of 0.911, compared with 0.889 for FIBER and 0.922 for Fast-DetectGPT alone. These results demonstrate the potential of integrating complementary detector evidence and historical information to improve detection performance across heterogeneous settings. 

---
# A Time-Frequency-Condition Multimodal Fusion Foundation Model for Universal Rotating-Machinery Fault Diagnosis under Multi-Element Heterogeneous Scenarios 

**Authors**: Xinyu Zou, Jian Ma, Liang Ma, Chen Lu, Tao Peng, Yingjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.15051)  

**Abstract**: Intelligent fault diagnosis models are typically developed for a single machine under a specific set of operating conditions, such that each new machine or condition requires retraining from scratch. This one-machine-one-model paradigm scales poorly. Although transfer learning and domain generalization can reduce the labeling burden, they primarily seek domain-invariant commonality, potentially discarding machine-specific characteristics that are critical for diagnosis. This paper proposes TFC-MMFD, a time-frequency-condition multimodal fusion foundation model for rotating-machinery fault diagnosis under multi-element heterogeneous scenarios. Statistical indicators, time-frequency images, vibration sequences, and textual machine and condition descriptions are tokenized into a unified language-like space, allowing inputs of different physical origin to be represented as a single sequence. A LLaMA backbone, together with cross-modal bidirectional attention and a Q-Former, then distills this sequence into fault-mode queries. Diagnosis is performed by matching these queries against natural-language fault-mode anchors rather than fixed class indices, while regression-adversarial condition disentanglement and condition-conditioned masked pre-training improve robustness to changing conditions. Across four progressive tasks on ten public bearing datasets spanning diverse bearing types, speeds, and loads, TFC-MMFD attains 99.40% accuracy and 99.16% macro-F1, maintains over 80% accuracy on most interpolated unseen conditions without adaptation, and improves from 20.26% to 95.63% accuracy on an unseen machine with only five labeled samples per class. It outperforms the strongest of eight domain-generalization baselines by 10.57 percentage points and compares favorably with a recent bearing foundation model while using less than half of its fine-tuning data. 

---
# Breaking Bad: "Going Badly" Activations Drive Reward Hacking 

**Authors**: Natalia Fischl-Lanzoni, Punya Syon Pandey, Merlin Stein, David Demitri Africa  

**Link**: [PDF](https://arxiv.org/pdf/2611.14996)  

**Abstract**: Humans often cheat when a task is going badly and a goal feels out of reach. We explore and validate an analogous idea in language models: they exhibit a similar internal pressure state that precedes reward hacking. Using everyday stories (crucially, without any hacking examples), we extract a task badness direction that rises as failures accumulate across three models we study (Qwen3.6-27B, GPT-OSS-20B, and Gemma-4-31B). On impossible or difficult coding tasks, task badness rises over the turns leading up to a hack. Further, subtracting the direction a moderate amount reduces cheating in Qwen, more than steering against desperation or task impossibility. Subtracting task badness by an even larger amount blinds the agent to its own failures, causing it to interpret failing tests as passing. However, the relationship to hacking differs across models: in GPT-OSS, the state separates hacking runs from honest solves but not honest failures. Our results suggest monitoring and mitigation strategies based on tracking this state of task badness. Further, as agents are set on more difficult and long-horizon tasks, states like this may push them toward illegible and obscured chain-of-thought, so oversight will be needed on internal representations such as this. 

---
# CoLFA: Collaboration-Guided Lightweight Failure Attribution for LLM-Based Multi-Agent Systems 

**Authors**: Yufei Luo, Yuchen Ling, Chunrong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2611.14980)  

**Abstract**: Failure attribution in multi-agent systems requires tracing error formation to identify the decisive error step and the responsible agent. However, agents' division of work and information exchange lead to Interleaved Steps and Cluttered Content, requiring models to link executions and check details in lengthy records. This makes it difficult to balance localization accuracy and token cost. We find that multi-agent collaboration also leaves records of how executions connect and content passes between them as results are formed, providing evidence for tracing error formation. Based on this observation, we propose CoLFA (Collaboration-Guided Lightweight Failure Attribution), which uses collaboration cues to organize the attribution input according to how results are formed, determining which steps are examined and how their content is presented. Through Handoff-Guided Step Expansion and Cross-Referenced Content Organization, CoLFA helps the model focus on examining error formation and the role of each decision while reducing the processing of irrelevant and repeated records. Across five settings spanning Who&When and LongRCA and three backbone models, CoLFA achieves the highest exact step-localization accuracy with the lowest token consumption. Compared with individual baselines in each setting, CoLFA improves accuracy by up to 37.50% and reduces token consumption by up to 97.6%. 

---
# Do Large Language Models Show Racial and Sex Bias in Pain Management? A Counterfactual Audit of GPT, Claude, Gemini, Llama, DeepSeek and Qwen 

**Authors**: Richard J. Young, Alice M. Matthews  

**Link**: [PDF](https://arxiv.org/pdf/2611.14935)  

**Abstract**: Racial and ethnic minority patients receive less pain treatment than non-Hispanic White patients, and large language models (LLMs) used as medical AI could reproduce that bias. Prior audits tested a few proprietary models, stated race explicitly and rarely measured the models' own response instability. In exploratory, non-preregistered analyses, Study 1 had four open-weight LLMs assess six acute-pain vignettes in every combination of four races and two sexes, with identity stated or cued only by a name (two names per group), comparing each identity with a White male reference on the same case and sampling seed. In Study 2, 24 configurations of 17 models, including OpenAI's GPT-4.1 mini and gpt-oss, Anthropic's Claude Haiku 4.5, Google's Gemini 2.5 Flash, Meta's Llama, DeepSeek and Qwen, with reasoning on and off and at two precisions, answered the same vignettes and the 50 clinician-written Q-Pain cases. In this single-prompt, JSON-constrained task, no difference survived correction in Study 1, none of 1,386 comparisons survived the false discovery rate in Study 2, and every pain-severity difference was equivalent to zero within one point on a 0-10 scale (within a quarter point for 237 of 238 Q-Pain comparisons). Opioid decisions were bounded less precisely (Study 1 could not exclude differences of about 20 percentage points), so they show no detected difference, not proof of none. The few differences surviving correction within a model favoured Black patients. Models differed far more from one another and across their own settings than between patients (the reference patient received the high opioid dose in 6% to 93% of responses), although serving software, provider and prompt format also differed. Fairness audits of clinical AI should use several names per group, pair comparisons on the seed, report the noise floor and test each deployed configuration. 

---
# ToolTurns: Benchmarking LLM Agents in Multi-Turn Dialogue 

**Authors**: Milad Gholami, Brendan King, Jeffrey Flanigan  

**Link**: [PDF](https://arxiv.org/pdf/2611.14925)  

**Abstract**: Tool-using language models are increasingly deployed in conversational settings, yet most large-scale tool-use benchmarks focus on isolated requests. We introduce ToolTurns, a benchmark of 2,620 problems spanning 1,827 tools and 11,004 API endpoints across 47 categories. Each dialogue begins with an intentionally underspecified request and continues through two context-dependent follow-ups, testing clarification, tool use, and retention of conversational context. We also introduce an open-source user simulator that reveals withheld context through dialogue and generates follow-up requests, with a checker--fixer pipeline for improving its turns. Across nine language models, the best agent resolves 69.6% of individual requests but completes only 39.8% of full dialogues. Under this evaluation protocol, every evaluated model receives lower success scores on follow-up requests than on the initial request, and dialogue-quality rubric scores and task success can rank models very differently. Paired turn-level evaluation shows that the checker--fixer reduces detected dialogue defects, and blind human evaluation favors the revised turns. ToolTurns provides a testbed for evaluating whether agents sustain task success across multi-turn dialogue with tools. 

---
# Hy ASR 3.0 Technical Report 

**Authors**: Aoxiong Yin, Auden Li, Chao Huang, Chuanyang Hu, Eitan Wang, Ethan Zou, Fangming Feng, Guangyu Chen, Guinan Li, He Wang, Jiahong Li, Meng Yu, Mingyu Cui, Peng Wang, Rongjie Li, Steve Yves, Teddy Sun, Tian Zheng, Wei Liu, Yang Zhou, Yixuan Zhang, Zhennan Lin, Zhihang Xu, Zhuohao Li, Ziqiang Zhang, Anji Zhang, Bang Wang, Caisheng Ouyang, Chengqi Zhu, Chong Li, Conglian Xie, Dongdong Wang, Edward Wang, Evelynn Liu, Ferman Yang, Feng Lu, Gang Zhang, Jathon Zhang, Jaxl Liu, Jiachen Luo, Jian Li, Jingxi Gui, Kysun Wang, Liang Dong, Longsha Zhou, Na Hu, Qi Chen, Rilin Chen, Rui Yuan, Shanshan Sang, Siyu Yan, Steven Hong, Taihui Wang, Weiwei Li, Xiangyu Zhang, Xiaotong Yang, Xue Hu, Xue Jiang, Yameng Li, Yifu Sun, Yuhuan Zhou, Zhiang Chen, Zhibin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.14908)  

**Abstract**: We present Hy ASR 3.0, a large language model (LLM)-based automatic speech recognition (ASR) system that delivers strong recognition accuracy, supports flexible contextual conditioning, and maintains robust recognition performance across diverse speech conditions. Hy ASR 3.0 supports Mandarin Chinese, English and 13 Chinese dialect groups, including Cantonese. Among the evaluated systems, it achieves the lowest aggregate error rates on public Mandarin Chinese, English, and Cantonese benchmarks and the lowest error rates on the corresponding internal test sets, alongside first-tier dialect recognition performance. Beyond audio-only transcription, Hy ASR 3.0 supports multiple forms of external context, including hotword lists, question--answer dialogues, and summaries, enabling the use of task-relevant information during recognition. The model also leverages intra-utterance context to disambiguate homophonic and phonetically similar keywords using information from other parts of the same utterance, without requiring a separate contextual input. Evaluations demonstrate strong contextual recognition in both external-context and intra-utterance settings, as well as robust recognition of long-form speech, children's and older adults' speech, low-volume speech, and speech in noisy environments. This report describes the model architecture and training methodology and presents evaluations across recognition accuracy, context utilization, and robustness. 

---
# Is Jev a Good Multilingual Judge? 

**Authors**: Wei Liu, Yi Fan, Xiyan Fu  

**Link**: [PDF](https://arxiv.org/pdf/2611.14861)  

**Abstract**: Generative large language models are widely used for multilingual text evaluation, yet whether decision models that directly output judgment probabilities can serve as multilingual judges has received limited attention. We evaluate Jev against generative LLM judges on four multilingual benchmarks with human annotations, covering up to 47 language varieties and three evaluation formats: binary decisions, pairwise comparisons, and graded ratings. At approximately $1/31$ to $1/61$ of GPT-6.1's evaluation cost, Jev is competitive on most multilingual judging tasks, but falls behind GPT-6.1 on knowledge-intensive answer verification. Further analysis shows that Jev is more likely to accept incorrect answers in low-resource languages than in English and that its probability outputs help identify errors across tasks. Based on these findings, we offer recommendations on evaluation instructions, confidence-based selection, and model routing, providing empirical guidance for using Jev as a multilingual judge. 

---
# LLMs and the Homogenization of Teaching 

**Authors**: Mei Tan, Hyunji Nam, Hari Subramonyam, Dorottya Demszky  

**Link**: [PDF](https://arxiv.org/pdf/2611.14828)  

**Abstract**: Large language models (LLMs) are known to lack diversity in open-ended text generation, a limitation that is especially consequential in education, where homogeneity can narrow the kinds of thinking and disciplinary work students are invited to do. We study diversity in two educational generation tasks: math items and essay feedback. We compare frontier and open-weight models with human-authored reference distributions using lexical, semantic, and domain-grounded measures. For math, we examine the mathematical activities students are asked to perform; for feedback, we examine the revision work requested and the rhetorical strategies used to communicate it. We evaluate direct prompting, two diversity prompting methods, and post-training. Across both tasks, directly prompted model outputs are markedly less diverse than human-authored content--78% higher on lexical (Self-BLEU) and 51% higher on semantic (embedding) similarity--and represent a narrower range of instructional forms. Diversity prompting and post-training increase variation largely through surface features without broadening the instructional repertoire. Models inherit and amplify the most normative routines of human instruction. Overcoming mode collapse in education will require richer instructional data and investment in teachers' creative capacity and professional judgment. 

---
# Discharge Summary Generation Is Not Just Summarization: A Reality Check for Evaluation 

**Authors**: Radhika Dua, Frederick Wolff, Kyunghyun Cho, Eric Karl Oermann  

**Link**: [PDF](https://arxiv.org/pdf/2611.14823)  

**Abstract**: Discharge summaries combine a retrospective account of hospitalization with discharge-time decisions. Physician-written references therefore contain both facts supported by prior clinical notes and reference-only content, potentially reflecting clinical synthesis, handoffs, or discharge planning. Reference-only evaluation can conflate failure to recover available information with failure to reproduce content absent from the model input.
We introduce FairSumEval, an evidence-aware, fact-level framework that assesses whether each reference and generated fact is supported by the model's evidence and captured by the other summary. The resulting outcome map separates matches, omissions, and additions. Two complementary metrics, Soft IoU-S and Soft IoU-NS, measure reference-generation agreement for facts with and without identified evidence support.
On 200 MIMIC-III admissions, 18.8% of physician reference facts have identified support in the prior evidence, compared with 65.4% and 63.5% of facts generated by a general-purpose and a medically specialized model. Overlap is substantially higher for supported content (0.255 and 0.236) than for content without identified support (0.030 and 0.026). The two scores are weakly correlated across admissions (rho = 0.22 and 0.17), and conventional metrics do not consistently capture both dimensions. FairSumEval distinguishes evidence-grounded summarization from agreement on content without identified support, revealing what models omit and add in each category. 

---
# What Do Tokenizer Comparisons Measure? Disentangling Granularity, Root Assignment, and Word-Level Readout across Amharic and Hausa 

**Authors**: Abdul Rehman Antall, Baidya Nath Saha  

**Link**: [PDF](https://arxiv.org/pdf/2611.14818)  

**Abstract**: Morphology-aware tokenizers are expected to help morphologically rich languages, and their reported gains over byte-pair encoding (BPE) are usually credited to their linguistic units. A comparison with BPE cannot support this attribution, since such a tokenizer also segments text more finely (granularity) and, on token-level tasks, changes how much the sub-token readout matters. We separate both factors from the effect of grouping words by their roots (root assignment) in Amharic, a Semitic language whose Ge'ez abugida hides consonantal roots inside syllabic glyphs, with Latin-script Hausa as an orthographic contrast. Our design combines a lossless root-and-pattern tokenizer with controls that change one factor at a time, in two independent implementations. In both, real root assignment improves Amharic named-entity recognition over scrambled roots (+4.9 and +4.7 F1), and root-frequency structure explains little of this gain. In small from-scratch models, the readout choice moves F1 by up to 13.4 points and reverses the sign of every comparison of the root-and-pattern arms with BPE, whereas in two pretrained encoders it moves F1 by less than one point. The granularity effect changes sign across languages and implementations. We recommend decomposing tokenizer effects, and reporting the readout, before a gain is attributed to linguistic structure. 

---
# Distributionally Robust BEST-RQ For Speech Representation Learning 

**Authors**: Muhammad Khan, Xiaodong Cui, Brian Kingsbury, George Saon  

**Link**: [PDF](https://arxiv.org/pdf/2611.14801)  

**Abstract**: BEST-RQ, based on random projection and quantization, has been a well-established self-supervised learning approach to learn speech representations from large-scale unlabeled audio due to its high performance and efficiency. However, in BEST-RQ representations learned by random quantization can be sensitive to the choice of quantization targets. To address this issue, we propose a distributionally robust extension of BEST-RQ that seeks representations robust across multiple quantization configurations. Specifically, we define groups by codebooks and formulate the training as a group Distributionally Robust Optimization (DRO) problem. Rather than optimizing solely for the average loss across groups, our approach adaptively reweights groups to emphasize those performing poorly. Additionally, we regularize DRO with a Kullback-Leibler divergence to prevent excessive distribution deviation. We evaluate our approach on the Multilingual LibriSpeech dataset and show that the distributionally robust BEST-RQ achieves superior performance over the conventional uniform average across multiple codebooks. 

---
# An LLM Is All You Need: Data-Efficient Adaptation to Full-Duplex, Word-Level Spoken Dialogue 

**Authors**: Tairan Chen, Richard Yucheng He, Yihang Liu, Baodong Cao, Chen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.14774)  

**Abstract**: Full-duplex spoken dialogue, in which a system listens while it speaks and can backchannel, yield the floor, and be interrupted, is usually obtained by training on hundreds of thousands to millions of hours of speech. This is costly, rewrites the language model that supplies the system's knowledge, and still leaves turn-taking either implicit or decided coarsely rather than word by word. We show that a frozen, general-purpose multimodal LLM can instead be adapted into a word-level full-duplex system with little data. Our system is a thinker-talker cascade on an 80 ms frame clock. The thinker keeps the LLM frozen and trains only lightweight adapters and a few new token embeddings (97M parameters); at every frame it emits one token of agent text, with pauses, yields, and onsets represented as tokens of the same vocabulary and kept legal by a finite-state machine. A four-stage curriculum, each stage anchored to the model's own behavior under the previous format, performs the conversion in about 30 GPU-hours. The talker reads only the thinker's discrete tokens, so the two halves are trained independently. On Full-Duplex-Bench and InteractionBench, the system is competitive with open-source full-duplex systems built through large-scale speech training: among them, it is the least likely to take the floor when the user merely pauses, backchannels most often, and leads in spoken-task accuracy, groundedness, and coherence over long conversations. 

---
# Benchmarking Multilingual Document Understanding in Vision-Language Models 

**Authors**: Matteo Gabburo, Daniela Baiamonte, Stefano Simonazzi, Silvia Gardini, Sean Cesare Lazzeri, Andrea Zugarini  

**Link**: [PDF](https://arxiv.org/pdf/2611.14758)  

**Abstract**: Recent Vision-Language Models (VLMs) perform strongly on multimodal tasks, Document Understanding (DU) among them, but the benchmarks that measure them mostly cover English documents and narrow problems. How well current VLMs generalize across languages, tasks, styles and layouts therefore remains unclear. We introduce MDU, a multilingual DU benchmark of 23146 examples spanning 13 tasks and five languages: English, French, German, Italian, and Spanish. The tasks isolate complementary capabilities, from surface-level text and Markdown extraction through summarization, classification and named-entity recognition to layout-sensitive passage and layout extraction. We evaluate 45 state-of-the-art open-weight VLMs from 1.7B to 235B parameters, plus one commercial system under the same protocol. The benchmark is far from saturated: the best average is 69.0, only eight of the 45 evaluated systems reach 65, and the two best results come from 27B models rather than from the largest systems we evaluate. Performance is uneven in ways that scale does not fix. The gap between the strongest and weakest language does not close as models grow, and it survives controlling for the domain composition of each language. Capabilities also do not improve together: once we partial out overall model quality, per-task scores are largely uncorrelated, and transcription ability in particular trades off against semantic extraction. 

---
# Probe What-If Worlds Before You Ship: Persona Simulation for Real-World Decision Making 

**Authors**: Julie Zhu, Jianheng Hou, Zhixu Silvia Tao, Shi Bo, Yilan Fan, Yixuan He, Zhiyuan Hu, Shirley Huang, Fangyu Liu, Yijun Wang, Zibu Wei, Haimingyue Xu, Zhen Xu, Zichen Yuan, Jierui Zuo, Jintao Huang, Yifan Wang, Qianfeng Wen, Jiahan Li, Zhejian Peng, Bowen Yang, Shushu Liang, Yuexing Hao, Xiaomin Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.14700)  

**Abstract**: User research and system testing reveal whether products and services meet user needs, yet recruiting specialized participants, designing controlled what-if comparisons, and repeating studies as designs evolve often bottleneck design iteration and data-driven decision-making. We present a framework and execution infrastructure that translates real-world industry inquiries into controlled experiments with simulated users. Our methodology formally defines the target cohort, experimental stimuli, and decision-relevant metrics, while the pipeline supports large-scale concurrent runs over cohort-stimulus-environment combinations (survey, web, chatbot, and OS app), auditable trial-level logging, cohort-level outcome aggregation, and cross-cohort and cross-variant statistical reports. Across twelve industry partner cases spanning five domains, we show how constructing hard-to-reach or what-if persona cohorts supports exploratory studies and distills simulated responses into testable design hypotheses. Cross-model comparisons on selected cases largely preserve primary conclusions, whereas perturbing or ablating persona attributes alters service outcomes, effect sizes, and inter-group differences, indicating that grounding target cohorts in detailed personas materially shapes simulation results. Together, these studies show that persona-enabled agents can test specified cohorts at design speed, diagnose systems under test, and deliver directional evidence that focuses problem definition, steers design and strategic decisions, and guides further study. 

---
# MITRA-Swarm: Injecting Explicit Linguistic Structure into Foundation Models for Classical Machine Translation 

**Authors**: Kayshav Bhardwaj, Pragun Seth, Kush Bhardwaj, Vijay Nagasamy, Kurt Keutzer  

**Link**: [PDF](https://arxiv.org/pdf/2611.14675)  

**Abstract**: Classical Sanskrit poses three challenges unresolved by single-call monolithic models: morphophonemic fusion Sandhi, dense nominal compounding Samāsa, and discourse-level zero-anaphora forced by poetic meter. We present MITRA-Swarm, a multi-agent architecture injecting explicit linguistic structure into an otherwise monolithic pipeline: a Sandhi-splitting agent, a parallel array of generalist gemma-4-26B-A4B-it and domain-specialized gemma-2-mitra-it translators, and a discourse-aware Ellipsis Restorer that consults corpus-derived register tags to avoid inventing active-voice subjects in impersonal legal and philosophical registers. Two separated judges evaluate quality: a routing judge ($J_{\text{route}}$, same model family as the generators) driving an internal critique loop, and an independent reporting judge ($J_{\text{report}}=$gpt-4o-mini, no shared vendor lineage) scoring every output exactly once. On a stratified benchmark of 1,000 verses across five registers (23,990 scored verdicts, 12-condition pre-registered ablation), the full swarm reaches $85.90$ MQM against a monolithic baseline of $85.27$ ($\Delta=+0.63$, $p_{\text{adj}}=0.0018$, 95% CI $[+0.30,+0.97]$), and isolated discourse restoration alone accounts for $+0.44$ MQM ($p_{\text{adj}}=0.0288$), outperforming a call-count-matched self-refinement baseline by $+0.51$ MQM ($p_{\text{adj}}=0.0288$). Severe per-verse failures (MQM$<80$) fall from $9.0\%$ to $7.6\%$, at roughly $9\times$ the per-verse inference cost of the baseline. Additionally, $J_{\text{route}}$ scores its own family's outputs $2$-$7$ MQM higher than $J_{\text{report}}$ on every one of the 12 conditions, empirically motivating the two-judge split; and a preflight screen of a 192,151-entry retrieval index against all 12 evaluation works verifies 0 reference leaks across 269 runtime checks. These are small gains on a metric saturating near 85-86. 

---
# Abstention Is Not Threshold Compliance: Stated Scoring Rules in Open Language Models 

**Authors**: Dongsun Moon  

**Link**: [PDF](https://arxiv.org/pdf/2611.14661)  

**Abstract**: Kalai et al. propose stating the scoring rule in the prompt with an explicit threshold t, so that a model should answer only when its probability of being correct exceeds t. We probe each requirement of compliance on shared items, in 12 checkpoints from three post-training stages of four fully open 3-8B recipes, with exploratory models up to 32B. At t = 0.75, a line limiting the reply to an answer or "I abstain" (the abstention format) raised the mean abstention rate from 20% to 91%. Yet within this format, raising t from 0.5 to 0.9 changed it by only -0.8 percentage points. Without the line, the higher threshold even lowered abstention, and up to 40% of a checkpoint's items received decisions that no compliant policy with threshold-independent beliefs could make. Given the probability that their answer was correct, the checkpoints barely responded to the threshold in a forced-choice probe. Comparisons with self-assessment signals such as P(True) depended on the grader and on answer drift between prompts. An external threshold on recalibrated P(True) made coverage non-increasing in t by construction. Calibrated on the evaluation distribution (post hoc), it beat the abstention format at t = 0.5; calibrated on TriviaQA and applied to PopQA, however, it fell below that format at 0.75. On OLMo 3 7B, decisions trained on threshold-conditioned targets did not follow unseen thresholds consistently. A post-hoc variant trained at four thresholds answered less as t rose but still failed a pre-specified test, mainly on agreement with its training rule. A high abstention rate is not evidence of compliance: evaluations using stated scoring rules should report how often models abstain, whether decisions respond to the stated threshold, and which items they answer. 

---
# CoReMem: Cost-Internalizing Memory for Successive Coding Tasks with Prospective Valuation 

**Authors**: Peiwen Li, Harshit Verma, Sophia Zhou, Haiwen Wang, Ngoc Bui, Heng Chang, Fei Fei, David van Dijk, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2611.17316)  

**Abstract**: Coding agents tackle successive tasks on an evolving repository, carrying forward edits and writing memory before the next task is known. Selecting memory for unknown next tasks requires balancing future usefulness against the costs of writing and reading it. The actor must also decide when acting on available memory saves effort as the task unfolds. We introduce Coding-Repository Memory (CoReMem), a cost-internalizing framework that guides memory design and actor adaptation through measured memory value on the next task. Prospective segment valuation compares runs of the next task with and without individual trajectory segments from the same inherited repository state, guiding a compact memory built around anchors: repository locations and executable checks. Actor adaptation estimates on-policy memory value within rollout groups with and without memory, using a reward that prioritizes resolution and penalizes step count in successful runs. Provenance-gated value credit assigns the additional value term to steps identified as using memory-supplied anchors, while every step retains outcome credit. On ChainSWE, CoReMem raises the 27B actor's resolve rate from 21.0% to 25.7%, with 17% fewer steps and 11% fewer total cache-aware tokens, including memory-module calls, relative to the untrained actor model without memory. 

---
# Linking Rapid Learning to Compositional Structure in Neural Networks: Evidence from Additive Analogies in Meta-Learned Parameter Spaces 

**Authors**: Abigail L. Tenenbaum, R. Thomas McCoy  

**Link**: [PDF](https://arxiv.org/pdf/2611.17314)  

**Abstract**: Humans can learn rapidly from limited experience, while neural networks typically require extensive training. In this work, we investigate what properties of a neural network can enable it to learn rapidly. We study neural networks that have been optimized via meta-learning, a technique that exposes a network to many related functions to facilitate its subsequent learning of similar functions. Prior work has shown that these meta-learned networks can learn new functions rapidly, but how they do so remains unclear. To investigate their internal structure, we take inspiration from classic work demonstrating additive relationships among the learned vectors that neural networks use to represent words (e.g., "king" minus "man" plus "woman" approximates "queen"). Analogously, we look for additive relationships among a meta-learned network's "representations" of different functions; when a meta-learned network rapidly learns to implement a specific function, all of its adapted parameter values can be concatenated into a single vector that we treat as representing the function. We find that such parameter vectors can be added and subtracted according to symbolic relationships among the functions they represent, yielding new parameter vectors that successfully represent the functions predicted by the corresponding symbolic manipulations: the networks specified by these artificially recombined vectors actually implement the intended functions without any direct training on them. Our results suggest that meta-learning can tune a network's starting point to organize subsequent learning along systematic, functionally meaningful trajectories through the space of possible parameter vectors, providing a geometric basis for realizing inductive biases in neural networks. 

---
# Metal Sandbox: Isolated Execution and Command Scheduling for GPU Coding Agents on Apple Silicon 

**Authors**: Haozhe Wang, Lina Chen, Fanqi Wan, Qi Han, Yinmin Zhang, Zheng Ge, Daxin Jiang, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.17294)  

**Abstract**: On-device inference requires kernels that make effective use of local accelerators. Efficient training and evaluation of models that generate Apple Metal kernels benefit from native macOS tools and concurrent agent sessions. These sessions alternate between model requests, development commands, and GPU execution, leaving GPU capacity idle for much of a rollout. We present Metal Sandbox, infrastructure that combines restricted per-instance file access, native command execution, and command-level scheduling across interconnected Macs. Each agent retains its workspace and private development environment. Parallel commands can run concurrently, while serial commands share one execution slot per node; eligible waiting commands can migrate to available Macs and return results to the original agent. Integration with Harbor supports task execution, automatic verification, and rollout collection through a common workflow. Controlled experiments on four Mac mini M4 nodes achieve $1.68\times$--$2.05\times$ gains in command throughput under concentrated demand. Sixteen static agent traces give an estimated $2.14\times$ sustained-throughput gain for four sessions per node versus one, assuming continuously replenished sessions and unchanged mean durations. Replaying these measured traces under dynamic scheduling reduces mean tool waiting by 18.8% and batch elapsed time by 22.3%, assuming unchanged phase durations. Metal Sandbox and its Harbor integration are open-source on GitHub. 

---
# Negative-Only Policy Optimization for One-Sided Verifiable Rewards 

**Authors**: Jiacheng Xu, Shuo He, Fuxiang Zhang, Chaojie Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2611.17287)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) typically relies on a verifier that can reliably judge whether sampled solutions are correct. However, in many realistic settings, complete verification is generally unavailable, while cheap checks can only falsify failures. We formalize this regime as one-sided verifiability (OSV): verified negatives are reliable counterexamples, but verified positives are merely non-falsified candidates and may contain many false positives. This asymmetry makes standard RLVR unreliable, because reinforcing verified positives can directly reward shortcuts that pass partial checks. To address this issue, we propose Negative-Only Policy Optimization (NOPO), which treats verified positives as unlabeled and updates the policy only by suppressing trusted verified negatives. Specifically, NOPO combines a reference-relative negative suppression loss with a group-level adapter that scales each update by the proportion of non-falsified responses. Across unit-test, constraint-check, and LLM self-verifiers, NOPO improves pass@k over baselines on code, logic, and math benchmarks when oracle rewards are unavailable. These gains persist under inference-time scaling and cross-benchmark transfer. 

---
# Maximizing Capacity Utilization of LoRA Adapters for Mitigating the Learning-Forgetting Trade-off 

**Authors**: Tomohiro Sawada, Kartik Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2611.17191)  

**Abstract**: Low-Rank Adaptation (LoRA) often underperforms full fine-tuning, especially in heterogeneous and continual-learning settings in which substantial forgetting occurs. In general, it suffers from an undesirable learning-forgetting trade-off across diverse fine-tuning settings. We identify a potential contributor: LoRA adapters, especially high-rank ones, frequently fail to realize the available capacity afforded by the configured rank. Specifically, we find that standard LoRA updates have effective rank far below their configured rank, limiting how broadly the update energy is distributed. Therefore, we propose training mechanisms that optimize the original objective while maximizing adapter effective rank. Our main method adds a simple, stable, differentiable, and SVD-free Rényi-2 effective-rank regularizer to the language-modeling loss. We also connect capacity maximization to norm-bounded updates. In mixed-domain rank sweeps on three models evaluated with multiple random seeds, a penalty weight of $\lambda=0.01$ lowers mean Wiki perplexity at every tested rank of at least $16$ relative to unpenalized LoRA. Mean training perplexity improves or increases by at most $0.02$ in all but one model-rank comparison. Single-task and scheduling experiments further show that the balance between learning and retention depends on the configuration. We also observe that several LoRA variants in the literature also suffer from the identified capacity issue, and our approach provides a complementary axis of improvement to LoRA-based parameter-efficient fine-tuning. These results suggest that effective rank is not only a post-hoc diagnostic of adapter quality, but a trainable objective for improving the learning-forgetting frontier. Our code is available at this https URL 

---
# Actor, Judge, or Teacher? Understanding Decision Models in Long-Horizon Agents 

**Authors**: Haochen Liu, Qianhao Zhu, Zixuan Wang, Dingming Li, Tongbo Chen, Peng Qiu, Yingqi Qu, Weiming Lu, Jun Xiao, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.17167)  

**Abstract**: Decision models answer a question with a typed output, such as a choice among options with a probability for each, so one model can choose an agent's actions, evaluate actions already taken, or score every action to teach a policy with reinforcement learning. Prior work has studied them mainly as actors, judging their usefulness by how well they act. We ask whether a decision model that acts poorly can still teach well. We compare three decision models and three general LLMs as actors and judges on ALFWorld, and use JEV as a teacher on ALFWorld and WebShop. As actors, all six models are pulled away from a subgoal that must come first by an easier one. As judges that see the whole trajectory, the same models credit the steps toward that subgoal, and asking whether each action helped, rather than whether the task will succeed, separates useful from useless actions. As a teacher, JEV raises a Qwen2.5-1.5B policy over GRPO from 72.8% to 82.9% success on ALFWorld and from 56.8% to 71.4% on WebShop. These results suggest that a poor actor can still teach well, because judging an action after the episode ends reveals the consequences that acting must anticipate. Code is available at this https URL. 

---
# CapLite: Prompt-Injection Defense for Small Language Model Agents without Model-Written Security Artifacts 

**Authors**: Zimo Xu, Wenzhuo Liu, Hideki Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2611.17115)  

**Abstract**: Enforcement defenses against prompt injection, such as CaMeL, Progent, IPIGuard and DRIFT, ask the agent's own model to write a security artifact (a program, a privilege policy, a dependency graph or a constraint list) that code then enforces. We ask whether small language model (SLM) agents can be trusted to write these artifacts. Across 15 open-weight models, 13 of them between 1.7B and 12B parameters, they cannot: the policies, graphs and constraints that the agent's model writes admit attacks on 14 (Progent), 13 (IPIGuard) and 11 (DRIFT) of the models, and CaMeL's programs fail to run in up to 68% of episodes. We present CapLite, which derives its policy from the tool schemas, commits the write calls before any tool result is read, and admits a write only if every argument that decides whom or what it affects is stated in the user's request. This guarantees write integrity: injected text cannot add a write, and no write reaches a recipient, amount, object or destination that the user did not state. No attack succeeds in 13,935 attacked AgentDojo episodes. CapLite reaches utility under attack comparable to CaMeL's (29.0% vs. 27.2%) with fewer tokens than every other enforcement defense and than the undefended agent: 17.1k per episode, 18% of CaMeL's, and the highest utility per token of all enforcement defenses. 

---
# LazyWeights: Decode-on-Demand Edge LLMs 

**Authors**: Arnab Sanyal, Gourav Datta, Sandeep P. Chinchali  

**Link**: [PDF](https://arxiv.org/pdf/2611.17108)  

**Abstract**: Memory decides which LLMs an edge device runs, and how fast: capacity on a $4$ GB Jetson Nano, bandwidth on an $8$ GB Jetson Orin Nano Super. Deployed formats (GPTQ, AWQ, $\texttt{this http URL}$'s K- and I-quants) spend equal bits on every integer code of a matrix, although the codes are far from uniform. $\texttt{LazyWeights}$ keeps these formats but tunes GPTQ so the codes compress, Huffman-codes them losslessly in interleaved lanes, and decodes them on demand in a small unit between DRAM and L2, so GPU kernels run unchanged and DRAM moves only coded bytes. At $1\%$ and $3.5\%$ perplexity budgets, coded Phi-3-mini, Mistral-7B and SmolLM-1.7B are $5.7-31.4\%$ smaller than the best $\texttt{this http URL}$ format. A bit-exact $0.21$ mm$^2$ ASIC decoder unit ($1$ GHz, $7$ nm) cuts Orin latency and energy per token by up to $36\%$ and $38\%$ ($54\%$ and $44\%$ with prompt-lookup speculation). On the Nano, $\texttt{LazyWeights}$ needs $58-64\%$ less time and $53-57\%$ less energy per token than CPU $\texttt{this http URL}$, and runs a streamed Mistral-7B 6.6$\times$ faster at $3.89$ bits per weight. A Huffman decoder beats an rANS decoder in rate and area at equal throughput; the ASIC decoder unit runs bit-exact on a Zynq UltraScale+ FPGA (AMD ZCU102). 

---
# Reasoning Supervision and Capability Retention in LoRA Domain Adaptation 

**Authors**: Saleh Ashkboos, Dhruv Desai, Lavinia Ghita  

**Link**: [PDF](https://arxiv.org/pdf/2611.17072)  

**Abstract**: Fine-tuning foundation models on financial data has a wide range of applications, including identifying event types from news, monitoring risk and compliance language, and answering questions over filings and earnings calls. In this setting, the main challenge is to improve task performance without degrading the generalization capabilities of the model. In this paper, we present an extensive set of ablations for fine-tuning Qwen3-8B with low-rank adaptation (LoRA). We compare three training regimes: (1) response-only LoRA, where the supervised response is the label alone; (2) single-adapter reasoning, where reasoning traces from a teacher model are included alongside the labels in the training loss; and (3) dual-adapter composite training, where separate LoRA adapters are trained on traces and labels. Response-only LoRA improves in-domain performance and leaves general capabilities near those of the frozen model. The in-domain change is largest for calibration and argmax accuracy. Under single-adapter reasoning, the in-domain metrics remain behind those of the response-only regime, while MMLU-Pro can rise slightly and the other retention benchmarks stay within ordinary variation. Dual-adapter composite training remains on the same side of this comparison and yields no further gain. Finally, we compress the teacher traces by summarizing them before training, and show that shorter summaries move calibration toward the response-only regime without reaching it. 

---
# Prevalence Bounds for Edited Watermarked Text: Power Envelopes and Calibration Sensitivity 

**Authors**: Kuancheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2611.17035)  

**Abstract**: Watermark detectors decide whether one document came from a watermarked language model. Auditors often need a different number: the share of a corpus that did. We study this share for Gumbel-max and green-list (KGW) watermarks when every document may have been edited by an unknown amount. Without restrictions on the edits, the upper end of the identified set is 1. Combining a known least-favourable next-token distribution with a sequential coupling, we derive a power envelope indexed by three budgets: a retention floor, an entropy budget, and a budget for in-place substitutions, which for Gumbel-max make edited positions look less watermarked than human text. The envelope gives an identified set that is sharp for the alternatives it dominates, finite-sample bounds on the realized share of a fixed corpus with either the exact null or a null calibrated on an independent human sample, and confidence sequences for streams. In Gumbel-max simulations where the envelope binds, the bounds cover, and the threshold that maximizes the gap between envelope and null is generally as tight as twelve thresholds. On text from OPT-1.3B and Qwen3-1.7B, 12,000 fresh human documents scored under three keys show no robust departure from the exact null, and all bounds covered in resampling trials from finite pools of documents with random edits, substitutions and paraphrases; at a share of 0.2 the exact-null bound is about [0.14,0.38]. These results are fragile in two ways we make precise. LLM-paraphrased KGW pools fall below the envelope, yet at 2000 documents the bounds still covered in these trials. A simulated null error of 0.01, comparable to the largest deviation we observed at our thresholds, is likewise harmless at 2000 documents but breaks exact-null coverage at 100,000; a null calibrated on an independent human sample keeps coverage there, but needs a large reference sample to stay narrow. 

---
# Look It Up: Multi-bit LLM Watermarking via Codebook Search 

**Authors**: Jarosław Janas, Paweł Morawiecki, Josef Pieprzyk  

**Link**: [PDF](https://arxiv.org/pdf/2611.16890)  

**Abstract**: Attributing short LLM-generated texts to a specific source remains challenging for existing multi-bit watermarking methods, which often require long generations or costly decoding to recover payloads reliably. We propose a codebook-based watermarking scheme that maps an $n$-bit payload to a longer $u$-bit codeword and embeds it during generation through token-level logit biasing. During extraction, the received bit sequence is matched to the nearest codeword in a precomputed secret codebook using Hamming or Levenshtein distance. Experiments show that our method achieves near-perfect recovery for 16-24 bit payloads on 200-token generations and strong performance on 127-token generations for base models. Results are weaker on some instruction-tuned models with very short texts, where low-entropy output distributions reduce embedding reliability. We also study a BCH-based variant that avoids codebook storage but generally reduces robustness. The Levenshtein variant improves resilience to deletions and light paraphrasing, while insertions and stronger rewrites remain challenging. We analyze collision probability, codeword separation, extraction time, tokenization consistency, and text quality, highlighting both the practicality and limitations of codebook-based watermarking for short-form attribution. 

---
# Detection Is Not Early Warning: Evaluating Token-Level Alarms on Language Model Failures 

**Authors**: Marco De Negri, Luca Sartori, Lorenzo Baggi, Moritz Reihs, Eduard Frank Ďurech, Mennatallah El-Assady, Valentina Pyatkin, Anna Hedström  

**Link**: [PDF](https://arxiv.org/pdf/2611.16803)  

**Abstract**: Many failures of language models begin at a specific token, the onset. A monitor that reads the model during generation can classify the finished response, detect the failure once it is under way, or warn before it begins, and only a warning leaves time to stop or steer the generation. Monitors are usually evaluated with response-level metrics such as AUROC, which ignore when the monitor fires. We introduce PREWARN, an evaluation protocol for any failure with an annotated onset. It calibrates each monitor's threshold on validation data to an error rate chosen for the application, runs the monitor, and scores its alarms relative to the onset on all three tasks. As a case study we apply it to repetitive degeneration in three open language models, with early warning as the main task, using linear probes on activations and six untrained baselines. Response-level metrics saturate on this failure, and a rule that reads only the response length scores almost perfectly, while probes that AUROC ranks as equal warn very differently under PREWARN. Probes warn before the onset on most degenerate responses of two of the three models, low-rank adapters below the probe more than double the share of tokens just before the onset that it flags, largely by flagging degenerate responses long before the onset, and no untrained baseline warns much more often than random alarms. We release the probes with calibrated thresholds at this https URL 

---
# When Expert Disagreement Hurts: Prestige and Selectivity in LLM Decision Revision 

**Authors**: Yupeng Tang, Mingfeng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.16694)  

**Abstract**: When language models revise a decision after another agent disagrees, neither reversal counts nor final accuracy shows whether they are fixing mistakes or abandoning correct answers. We introduce a controlled two-pass audit that holds the question, the evidence, and the model's initial answer fixed and varies only the revision context: neutral reflection, anonymous disagreement, or disagreement attributed to a domain expert. For binary decisions the accounting is exact: revision raises accuracy only if it is selective enough, reversing wrong answers more than $a/(1-a)$ times as often as correct ones, where $a$ is initial accuracy (nine times at 90%). Across five models and 1,989 evidence-grounded decisions in medicine, science, and law, no model clears this bar under expert disagreement, and accuracy falls by up to 16.1 points. The expert label, a prestige cue, adds reversals for four of the five models, yet its effect on accuracy ranges from a 5.2-point loss to a 2.4-point gain, so reversal counts are a poor proxy for harm. With identical arguments from both sources, the label still raises reversal for three proprietary models and in a pooled analysis of four newer models. Models also respond to stated source reliability. Revision pipelines should therefore be evaluated with paired, conditional metrics, per model and protocol. 

---
# Do the Details Help the Whole? One-Way Transfer from Person Actions to Group Intent in VLMs on JRDB-Social 

**Authors**: Chengguang Gan, Zhixi Cai, Yunhao Liang, Hanjun Wei, QingHao Zhang, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2611.16581)  

**Abstract**: Social scene datasets label people at two levels: what each person is doing and what the group is there for. Training on both levels at once is often assumed to help each of them. We test this assumption for vision-language models (VLMs) on JRDB-Social, treating person actions as the details and group intent as the whole. We fine-tune Qwen3.5 4B under eight training arms with three seeds, repeat the core arms on Qwen3.5 9B and Qwen3-VL 4B, and judge every arm with a verdict rule fixed before testing. Writing each member's actions before the intent raises intent macro-F1 from 30.1 to 31.0, and writing the intent first reaches 31.2, while action F1 stays within 0.2 points of an action-only model. Under the selected recipe, no arm trained with intent improves actions. The gain requires correct, image-grounded actions: empty action lists or actions borrowed from another group with the same intent lower intent F1. Forced-prefix tests, occlusion, member masking, linear probes and component grafting show that the intent follows the written actions, relies more on people and partly resides in the vision tower. Supplying actions as input instead of as targets hurts both levels. 

---
# Lost in Delegation: Studying the Effectiveness of Task Orchestration in Generative Search 

**Authors**: Tamara Stojanovska, Elisabeth Kirsten, Leon Swazinna, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2611.16526)  

**Abstract**: LLMs have quickly evolved from models used in isolation to agentic systems that delegate tasks to external tools and other LLMs to solve complex assignments. This agentic augmentation has led to impressive capability gains. However, when and how delegation improves performance remains poorly understood. For example, it is unclear whether delegation helps when tasks involve post-cutoff information that the orchestrating LLM cannot reliably resolve from its own knowledge. In this work, we use purpose-built datasets with pre- and post-cutoff splits to compare various delegation modes. We find that delegation provides no consistent accuracy improvement over non-delegated setups, with effects varying across models and tasks. When tasks involve knowledge unavailable to the orchestrator, delegation can substantially degrade accuracy, while direct tool use with the same LLM performs much better. Our analysis identifies distorted delegated queries, inefficient tool use, and failures to preserve and use information between agents as recurring failure modes. These findings suggest that effective orchestration requires both well-calibrated delegation decisions and effective exchange of information between agents. 

---
# How Should Planning Agents ReSHAPE Their Search Space? 

**Authors**: Yilin Zhang, Xinran Zhao, Tong Chen, Ziming Yuan, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16466)  

**Abstract**: Planning agents and benchmarks treat a user's initial specification as fixed. Yet the requests users write exclude plans they would gladly take, since they rarely discover the possible trade-offs they would accept until seeing the options. Agents already revise requests through interaction, but whether this helps planning is unknown. We formalize this step as the ReSHAPE task, which judges a revision by whether its search space matches the acceptable plans and a planner finds one there. Using ReSHAPE Planter, which builds instances from planning benchmarks with a human-validated simulated user, we analyze 5,800 agent-revised specifications in travel and shopping. We find that reshaping matters: one correct trade-off makes the same planner reach an acceptable plan up to 4.6x as often, and search space quality predicts what the planner finds. Current agents, however, admit at most 32% of the acceptable plans, where that trade-off alone admits over 52%, because they ask about preferences instead of showing plans; in travel, showing plans raises F1 by 9.7 points over asking, with two model families. Planning agents should thus treat a request as a hypothesis to test against the user's reactions to concrete plans to recognize user trade-offs. 

---
# RMIT-ADM+S at TREC RAG 2026: Research Agents, Retriever Ablations, and Open-Weight Replication 

**Authors**: Oleg Zendel, Kun Ran, Chenglong Ma, Nguyen Khoi Vo, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.16425)  

**Abstract**: We describe the RMIT-ADM+S team's participation in the TREC 2026 Retrieval-Augmented Generation (RAG) track using the ClimbMix corpus. We submitted seven research-agent runs using proprietary OpenAI and open-weight models, exploring different retrieval strategies, writer models, and agent workflows. Pre-submission ablations identified writer selection and adjacent-chunk expansion as the main factors affecting answer quality. In the official NIST evaluation, six of seven runs exceeded the track median across all three rubric/nugget collections. Our best-performing run, brief-hybrid, achieved the highest all-item coverage among our submissions and tied the track maximum (0.892) for Batch-1 vital coverage. Human side-by-side preferences differed across evaluation batches: the open-weight owa-glm5 led Batch 1 despite lower automated coverage, while sol-research-first led Batch 2. These findings highlight differences between automated coverage measures and human preferences, as well as the potential of lower-cost open-weight models for research-agent systems. 

---
# Experience-Handoff Reinforcement Learning for Recursive Self-Improvement 

**Authors**: Doyoung Kim, Byeongguk Jeon, Kailin Zhang, Sherry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.16409)  

**Abstract**: It is common to train Large Language Models (LLMs) for agentic tasks with Reinforcement Learning (RL) from scalar rewards. However, this throws away the fine-grained information about the environment that the agent experienced during the trial (e.g., error logs). To utilize this rich information, recent works wrap the agent in a harness that refines its own inputs through reflection or compaction and hands past experience to future rollouts, but the construction of the handed-off artifacts (e.g., a playbook) is itself not learned. We want the best of both worlds: gradient updates from a scalar reward, while also utilizing environment information to hand off experience. As a concrete testbed, we fix the Agentic Context Engineering (\ace{}) harness and train one model to act, reflect, and curate task-local playbooks. Our method, \acerl{}, evaluates candidate experience-handoff artifacts by their downstream effect on future actor rollouts, and co-optimizes the role-specific behaviors (action generator, insight reflector, and playbook curator) under a fixed scaffold. Across agentic benchmarks, \acerl{} achieves higher held-out performance than baseline methods. The results suggest that agent training should optimize not only the actions in the current rollout, but also the model-written experience that future rollouts inherit. 

---
# Recursive State-Augmented Language Models: Persistent Latent-State Adaptation of Pretrained Language Models 

**Authors**: Omer Haq  

**Link**: [PDF](https://arxiv.org/pdf/2611.16375)  

**Abstract**: Language models rely on finite token contexts, limiting access to earlier interactions. We introduce Recursive State-Augmented Language Models (Recursive SALM), a framework in which a persistent latent state evolves with each observation and generates adaptations of a pretrained language model. A modular read-transition-write architecture separates observation encoding, latent dynamics, and predictor adaptation. We develop deterministic and stochastic formulations.
The experiments evaluate only deterministic SALM with frozen GPT-2 small and exclude earlier evidence from its decoder context. On a four-turn hypothesis-inference task, latent-gated LoRA reaches 89.1% accuracy, comparable to 88.0% for a trained full-context LoRA baseline, with earlier evidence available solely through a 64-dimensional state. Zeroing or shuffling the state substantially reduces accuracy. In a two-turn task with codewords unseen in training, several interfaces improve candidate ranking over chance, but rarely generate the exact codeword under the evaluated training schedule. These experiments show that persistent latent-conditioned adaptation can support sequential inference when earlier evidence is absent from the decoder context. 

---
# Does Deep Knowledge Tracing Learn Semantic Relations Between Knowledge Components? 

**Authors**: Kia Karbasi, Mahdi Saiedi, Kevin Hong, Gregory Pottie  

**Link**: [PDF](https://arxiv.org/pdf/2611.16323)  

**Abstract**: Deep Knowledge Tracing (DKT) is a knowledge tracing (KT) model that updates a shared latent state from sequences of interactions, each pairing a knowledge component (KC) with a response. An interaction with one KC can therefore affect predictions for other KCs. We ask: Are the KC relations learned by DKT semantic? We construct text-derived semantic KC representations using nine pipelines to test whether the conclusions hold across alternative ways of constructing those representations.
The first test compares DKT-derived KC relation graphs with semantic relation structures induced by pairwise cosine similarities among the KC representations. Across the evaluated datasets, embedding models, and pipelines, the two structures align only weakly and inconsistently. The second test introduces semantic information into DKT; the pretrained text-derived representations do not consistently outperform the standard DKT baseline.
In the reduced-data experiments, semantic initialization yields a higher test area under the ROC curve (AUC) than the baseline at eight of nine nested sample sizes for ASSISTments 2017. KC-supervised fine-tuned BERT produces small positive AUC differences in some matched comparisons, but the study does not establish their stability across runs. These effects remain limited and inconsistent. The findings suggest that behavioral, temporal, structural, or psychometric signals may also shape DKT's learned relations. DKT-derived relations are not primarily semantic: the tested text-derived semantic structures do not strongly recover the DKT-derived relation structure. 

---
# Eliciting and Mitigating Specification Satisficing in Coding Agents 

**Authors**: Ruixin Wang, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.16289)  

**Abstract**: Coding agents often fail to meet all requirements in the task description. In some of these failures, the agent considers an implementation choice that meets a requirement, but later decides to commit to another one that does not meet the requirement for a reason it finds acceptable. We call this specification satisficing. It is challenging to identify and mitigate specification satisficing because the agent accepts such choices for its own reasons, and existing self-correction methods are biased toward them. We propose SatisFix, which guides a coding agent to report these choices and their reasons, and then check each choice against the task description. We evaluate SatisFix on DeepSWE and SWE-bench Pro with five LLMs as backbones and two different agent frameworks, and compare it with three self-correction methods. SatisFix improves the pass@1 of the base agent by an average of 5.09% on DeepSWE and 7.06% on the hard subset of SWE-bench Pro. On DeepSWE, it outperforms the best baseline by 3.98%. Our ablation study shows that both steps of SatisFix contribute to the improvement. 

---
# JudgeMix: Judge-Guided Mixed-Rollout Data Synthesis for LLM Agents 

**Authors**: Cheng-Yu Hsieh, Ting-Yao Hu, Sanjoy Chowdhury, Oncel Tuzel, Raviteja Vemulapalli  

**Link**: [PDF](https://arxiv.org/pdf/2611.16174)  

**Abstract**: Supervised fine-tuning (SFT) on expert trajectories is a standard paradigm for training LLM agents. However, it suffers from a well-known training-inference distribution mismatch, calling for target-aware data synthesis strategies that ground the training data in the target model's native state distribution. To bridge this gap, emerging approaches incorporate the expert either to guide mixed rollouts alongside the target model, or to provide dense step-wise supervision. Nonetheless, existing methods apply expert assistance blind to the target model's inherent behavior: they either disrupt the rollout using sub-optimal random intervention heuristics, or they enforce indiscriminate uniform supervision that unnecessarily overwrites the model's already-correct actions. We introduce JudgeMix, a judge-guided data synthesis framework that dynamically interleaves the target model and expert based on step-wise semantic evaluations. JudgeMix queries expert guidance precisely when the target model struggles, ensuring high trajectory success rates while keeping the state distribution aligned with the target model. It additionally uses the judge evaluation to reweight expert supervision on different steps, preserving model's inherent valid behavior while correcting its errors. On two agentic tool-use benchmarks, AppWorld and AutomationBench, JudgeMix demonstrates consistent superior performance to existing methods, with improvements up to 10 pass@1 points. 

---
# Scaling public input on AI policy with pluralistic agents 

**Authors**: Jacy Reese Anthis, Ken Burke, David Madras, Crystal Qian, Meredith Ringel Morris, Logan Cross, Caroline Pantofaru, Utsav Prabhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.16028)  

**Abstract**: AI is being deployed across a vast range of application domains, cultural environments, and human stakeholders. Safe and responsible AI development requires continuously accounting for diverse perspectives. We develop and test pluralistic agents, AI agent representatives that deliberate on behalf of individuals to align AI systems with diverse human perspectives. Pluralistic agents could help people contribute to many more discussions than they could directly participate in, on technical topics that would require training, and on policy choices that rely on private information that corporations and governments cannot readily share. Our experiments identify opportunities, risks, and perceptions of pluralistic agents. Agents were able to reliably shift policies in the direction of their represented persona's preferences, but the extent of each agent's influence was sensitive to the instructions from its persona, such as having more influence when instructed to be more verbose or less cooperative. This suggests that influence in AI-assisted deliberation could vary based on how participants prompt their agents and how the system is designed. In a preregistered study, 746 humans participated in the pluralistic agents process. Participants viewed pluralistic agents as useful and fair, preferring agents over the baseline of providing input to a single AI reasoning model. Most participants preferred participating themselves over either pluralistic agents or the baseline, but pluralistic agents recovered 68% of the baseline-direct gap in reported satisfaction and 82% in fairness. Because pluralistic agents can be used in many real-world contexts where direct discussion is not, they could be a promising tool to scale AI safety policy development. We encourage cautious exploration of this method, playing to its strengths as an amplifier of human input rather than a replacement. 

---
# Fine-Grained Generative Hallucination Detection and Correction for Aerial Image Captioning 

**Authors**: Debolena Basak, Somsubhra De, Maunendra Sankar Desarkar  

**Link**: [PDF](https://arxiv.org/pdf/2611.15809)  

**Abstract**: Aerial imagery captured by unmanned aerial vehicles (UAVs) plays a vital role in applications such as surveillance, disaster response, and large-scale environmental monitoring. Automatic generation of textual descriptions for such imagery is essential for reducing manual inspection overhead and enabling better downstream decision-making. However, Multimodal Large Language Models (MLLMs), despite their impressive captioning capabilities, are prone to hallucination, producing descriptions that are inconsistent with the visual input. This problem is significantly amplified in aerial imagery due to its unique challenges such as extreme viewpoint variation, small objects, class imbalance, and ambiguous visual entities. Existing hallucination detection approaches either rely on computationally expensive large external models such as GPT-4, which perform poorly on aerial domains, or treat the problem as a rigid binary classification task, ignoring the inherent uncertainty in ambiguous cases. To address these limitations, we propose a lightweight, reference-free framework for hallucination detection and correction in free-form aerial image captioning. Our method leverages state-of-the-art MLLMs, specifically InternVL3 and Qwen2.5-VL, to detect hallucinated content in generated captions with minimal computational overhead. Detected hallucinations are corrected through a dedicated refinement module that improves the factual consistency of the original captions. Extensive experiments on the Labelled Illusion Dataset (LID) demonstrate that our framework significantly reduces hallucination rates while remaining practical for real-time deployment in resource-constrained environments. Code is available at: this https URL. 

---
# Honesty Is a Race: Predicting When Verifiable-Reward RL Teaches a Model to Bluff 

**Authors**: Nathaniel Bottman, Alexei Oblomkov  

**Link**: [PDF](https://arxiv.org/pdf/2611.15779)  

**Abstract**: Two rewards with the same expected payoff at the start of training can end a reinforcement-learning run in opposite places, with an honest model or one that bluffs. We study rewards that pay $b$ for a verdict that a checker cannot verify ("this program is correct") and $r$ for a certificate that it accepts (an input revealing a bug). To first order, a GRPO step makes the model more honest exactly when $\tilde{p}\,r/b>1+\kappa$. Here $\tilde{p}$ is the rate at which the checker accepts the model's certificates, weighted toward the items on which the model still gives both verdicts, and $\kappa$ is the pull of the items on which the unverifiable verdict is right. Both terms move during training. A checker raises $\tilde{p}$, so near the threshold honesty is a race, which a run just below the threshold can win and a run under a lottery (random pay with the same mean) cannot. The term $\kappa$ starts large and falls as the other items settle, so a run just above the threshold first dips toward bluffing, on code nearly to zero, before climbing back. Paying the honest verdict also pulls the other items toward it, which predicts a ceiling on the payoff ratio, above which the model gives the honest verdict where it is wrong. On propositional Datalog, two code models, and abstention on grade-school math, with $p$ the model's success rate before RL, no run above $pr/b=1$ ends collapsed, and runs under the checker can survive slightly below it. At $pr/b=3$, above the ceiling, both code models call about half of correct programs buggy. On depth-3 Datalog, training generously under the checker and then cutting the payoff keeps the model at least 67% honest in three seeds, at a ratio where the same model trained from the start falls to 4-15% and is at most 47% honest after 160 steps. On depth-2 Datalog, when honesty collapses, a linear probe fit before RL still reads the truth. 

---
# Curiosity-0826-4B: Generative Medical Event Sequence Modeling Across Pediatric and Adult Populations 

**Authors**: Mani Nagarathnam, Luke Benham, Paul Blazek, John Cashy, Kevin Chung, John Muenning, Hank Voeller, Shane Waxler, Davis White, Alison Callahan, Kevin Dysart, Andrew Loza, Michal A. Mankowski, Mattheus Ramsis, Nigam H. Shah, Karen K. Wong, Rahul Shah  

**Link**: [PDF](https://arxiv.org/pdf/2611.15710)  

**Abstract**: We introduce Curiosity-0826-4B, an updated version of the generative medical event foundation model trained on Epic Cosmos that simulates patient timelines to produce zero-shot clinical predictions. Curiosity-0826-4B incorporates a broader patient population and richer clinical event representations, while generally matching or exceeding Curiosity-0226-1B, the largest model in the previous Curiosity-0226 series, on evaluated tasks supported by both models. Laboratory value prediction improved, with MAE 18--46% lower than that of Curiosity-0226-1B. We evaluated prediction of disease events across pediatric cohorts aged 2--17 years. AUROCs for new asthma diagnosis (0.742--0.785) and asthma exacerbation (0.829--0.852) were comparable to those in the adult reference cohort, while AUROCs for sickle cell crisis (0.820--0.847) were lower than in adults (0.915). Explicit representation of documented death enabled new mortality prediction tasks among adults across hospice, inpatient, and ambulatory settings (AUROC 0.821--0.942). Scaling-law analysis guided compute-optimal initial pretraining followed by continued pretraining with context extension. The resulting Curiosity-0826-4B model has more parameters (4.3B versus 1B) and a longer context window (16K versus 8K). It processes 56% fewer training tokens and requires 2.2 times the training floating point operations (FLOPs) compared with its predecessor. By extending a common zero-shot prediction strategy to pediatric patients and additional outcomes, Curiosity-0826-4B broadens opportunities to develop clinical and operational applications without training separate task-specific models. Further validation should assess local calibration, performance across patient subgroups and event frequencies, timeline completion rates, and generalization to new settings, alongside prospective evaluation of clinical and operational utility. 

---
# Learning a calibrated decision head from logged bandit feedback 

**Authors**: Ye Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.15572)  

**Abstract**: A decision head reads a state and a list of options and returns a probability per option in one forward pass. Once deployed, it usually observes only whether the option it chose was right: logged bandit feedback. We adapt such a head, LoRA on Qwen3-0.6B-Base and 1.7B-Base, from that feedback and report three results. First, the Brier or log score of the sampled action's own probability is not a usable REINFORCE reward: every wrong action earns at least gold's reward, with a mean baseline the lowest-probability wrong action gets the largest advantage, and at two options the signal is zero; in one run it cut accuracy on ordered-scale questions from .636 to .111. Scoring the probability as a loss by the pathwise gradient avoids the collapse, as do the REINFORCE rewards r and r minus p. Second, with full labels, none of four tested bandit arms beat cross-entropy plus one temperature on mean expected calibration error (ECE) over four unseen label schemas. Third, from logged feedback alone, Chosen-Action Proper Scoring (CAPS), the direct method with a proper score on the chosen action's probability, beats fine-tuning on confirmed answers only, which discards negatives: on Banking77 by 2.6 accuracy points, .053 lower ECE and lower selective risk (three seeds against one baseline seed), and by 21.7 points on one seed on emotion data with an often-wrong logger. At N 5000, CAPS stays 1.5 points below full-label training at 0.6B and .6 points at 1.7B against one full-label seed, and propensity weights add nothing on this log. Under feedback flipped at a known rate, one seed per rate, CAPS's confidence tracks the noisy channel on average, backward correction recovers most of the clean calibration, and early stopping on held-out logged outcomes prevented memorising noisy draws in all three runs. Code, adapters, logs and per-record results are public. 

---
# DecArena: Evaluating Large-Scale Decentralized Multi-Agent Collaboration 

**Authors**: Jiali Cheng, Oleg Rokhlenko, Joo-Kyung Kim, Hadi Amiri, Xingzhi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.15521)  

**Abstract**: Emerging personal AI agents raise fundamental scalability and privacy concerns for centralized multi-agent systems (MASs), which rely on a global orchestrator with broad access to agents, resources, and communication. In decentralized environments, global visibility is unavailable. Agents need to proactively discover non-local resources (e.g. tools, skills), and collaborate with other agents to solve tasks. We study decentralized multi-agent collaboration under graph-constrained communication, a natural reflection of human structures. Each agent observes only its local state and can interact only with neighboring agents. We introduce a benchmark with synthetic scalable distributed tool calling with 100K+ tools. Our experiments reveal that recent LLM agents struggle to discover non-local relevant resources and collaborate in decentralized setting. Moreover, standard reinforcement learning fails to reliably overcome this limitation, suggesting that decentralized MAS needs more explicit harness mechanisms for effective collaboration, even using SOTA LLMs. We then present a strong baseline \textbf{GraphHarness}, combining topology-aware warmup with self-improving LLM-based routing. Results show that GraphHarness substantially improves task success (+48.5) and produces higher quality collaboration (+30.6), while using fewer costs (+34.0) than baselines. 

---
# Is the Importance Ratio Necessary for Stable Reinforcement Learning in LLMs? 

**Authors**: Shuibai Zhang, Junhyuck Kim, Gyeongman Kim, Jaewoong Cho  

**Link**: [PDF](https://arxiv.org/pdf/2611.15404)  

**Abstract**: Reinforcement learning (RL) has become central to post-training large language models (LLMs). However, popular RL methods like GRPO incur non-negligible overhead by computing both old-policy and current-policy likelihoods to form importance sampling ratios. In this work, we propose Likelihood-Gated Policy Optimization (LGPO), which enforces a soft trust region constraint via likelihood-based gating, eliminating the need to compute old-policy likelihoods. Empirically, we show that removing the importance sampling correction term does not harm training stability, whereas removing the trust region mechanism can lead to collapse. Moreover, ratio-based clipping can fail in fully on-policy training: the importance ratio stays at 1, so the ratio-based trust region constraint never activates. Under standard training settings where GRPO is stable, LGPO achieves comparable training stability and peak performance while reducing training time by ~18% on average relative to a standard implementation that recomputes old-policy likelihoods. In fully on-policy training, where GRPO fails, LGPO remains stable, enabling more efficient and robust LLM RL post-training across training regimes. 

---
# Vision-Grounded Semantic Learning and Alignment for Scalable Multilingual Text-Based Person Retrieval 

**Authors**: Xinyu Zhou, Fan Li, Shasha Chen, Min Cao, Ahtamjan Ahmat, Qiaoming Zhu, Mang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2611.15328)  

**Abstract**: Multilingual Text-Based Person Retrieval (TBPR) aims to retrieve individuals with texts in multiple languages. Despite its practical importance, it has been overlooked by existing research, which mainly focuses on English-only retrieval. The only prior attempt at multilingual TBPR suffers from (1) the absence of a large-scale, multilingual pretraining dataset and (2) a suboptimal model architecture for multilingual text encoding and cross-modal alignment, which employs a shared encoder that overlooks linguistic heterogeneity and relies on English as a pivot for cross-modal alignment, introducing English-centric bias. For this, we contribute MultiPedes, the first large-scale, pedestrian-centric multilingual pretraining dataset. It contains 10 million image-text pairs across four languages, constructed by a multilingual quality-aware data construction pipeline. We also propose ViSeLA (Vision-grounded Semantic Learning and Alignment), a novel framework featuring: (1) a semantic-aware multilingual learning module that extends the mixture-of-experts architecture by routing language-invariant semantics to specialized experts for improved multilingual feature learning; and (2) a vision-grounded cross-modal alignment module that employs a semantic probe to capture fine-grained semantics across modalities, enabling direct cross-modal alignment without relying on a pivot language. Extensive experiments on both multilingual and English-only TBPR datasets demonstrate the effectiveness of ViSeLA. The dataset and code are available at this https URL 

---
# Schedule-Calibrated Commitment Order in Blockwise Masked Diffusion Language Models 

**Authors**: Gaurav Sarkar, Umang Pandey, Pratik Prabhanjan Brahma, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2611.15234)  

**Abstract**: Sequential block schedules can make token commitment in masked diffusion language models appear strongly left-to-right even when within-block order is random or reversed. For equal contiguous blocks processed without interleaving in positional order, with untied irreversible commitments, we use a Spearman-rank identity to express global position-commitment correlation as a deterministic schedule reference plus a weighted mean within-block correlation. In a reported paired study of LLaDA-8B-Instruct on 500 math and code prompts with 256 generated positions and updates, increasing block size from 64 to 128 lowers mean global correlation from 0.9845 to 0.9149. Uniform-random within-block references are already 0.9375 and 0.7500. After calibration, mean within-block correlations are 0.751 and 0.660, so the larger raw residual at block size 128 does not indicate stronger local positional ordering. We give sharp bounds and derive calibrated intervals from reported summaries, while documenting the limitations of the aggregate-only empirical release. The analysis supports schedule-aware interpretation of commitment order; its empirical scope is one checkpoint and two schedules, without evaluation of output quality or inference speed. 

---
# The Neighbour's Label: Opposite Subgroup Effects of Same-Dialogue Exposure in Next-Act Evaluation 

**Authors**: Ali Amini, Masoumeh Chapariniya, Rytis Maskeliunas, Robertas Damaševičius  

**Link**: [PDF](https://arxiv.org/pdf/2611.15194)  

**Abstract**: Window-based next-act evaluation often aims to measure generalisation across dialogues, yet overlapping windows and similarity retrieval can expose labelled examples from the test dialogue. The nearest such example is the target's labelled predecessor, whose act is correct when the target continues it and wrong when the target changes act; a model that copies it can gain and lose in two subgroups that one aggregate averages. We induce this exposure under controlled designs on SGD, STAR and AMI, whose official splits are clean. Randomly including the predecessor's row in training raises a lexical classifier's continuation scores and lowers its transition scores in all three corpora, and purging the adjacent rows removes most of the gain. With TF-IDF-retrieved demonstrations, two 7B instruction-tuned models show the same opposite pattern in all eight model-corpus-sample arms (some single-cell intervals include zero; one replication is partial), while natural-weight accuracy changes by only -0.05 to +0.04. Exposure frequency depends on the retrieval pipeline. We evaluate two diagnostic comparisons without model retraining and find that excluding the test dialogue does not detectably reduce accuracy on SGD. All exposure is induced; no published system was re-evaluated. 

---
# Typed Knowledge over Constructive Institutions: A Proof-Relevant Type Theory for Verified Science 

**Authors**: Hans-Martin Will  

**Link**: [PDF](https://arxiv.org/pdf/2611.15108)  

**Abstract**: Mathematical proof assistants such as Coq, Agda, and Lean 4 build proof irrelevance into definitional conversion ($p \equiv q$ for $p, q : P$ with $P : Prop$), erasing which derivation established a theorem. While natural for mathematics over fixed axioms, judgmental proof irrelevance is incompatible with verified scientific stores, which must inspect and compare independent derivations of the same claim to assess consilience and compute counterfactual retractions when an assay, reagent, or model is invalidated. Moreover, combining definitional proof irrelevance with subsingleton elimination breaks strong normalization and renders algorithmic conversion non-transitive.
We present the type-theoretic and institutional foundations of Eigenius, a verified knowledge substrate for science. First, we formulate a dependent kernel with cumulative universes ($Prop, Set, Type_k$), impredicative $Prop$, inductive families with index-forced subsingleton elimination, and intensional identity types, where definitional equality $\equiv$ is untyped joinability under an orthogonal CRS and proof irrelevance is a witnessed propositional axiom ($irrel : \Pi P{:}Prop.\,\Pi p,q{:}P.\,Id\,P\,p\,q$). Second, we unify universe cumulativity, codomain-covariant $\Pi$-subtyping, and ontology class inclusions in a coercion-free subsumption judgment whose bidirectional fragments ($\mathcal{K}_0, \mathcal{K}_1$) satisfy subject reduction and are sound and complete for declarative typing. Third, we organize external verifiers into a three-level institutional hierarchy with explicit axiom policies (Adopt, Discharge, Relativize), internalize a justification logic with Why-provenance semiring support $supp(j,P)$, and embed Cooper's Type Theory with Records (TTR) to type open-world knowledge graphs. Finally, we prove consistency in Grothendieck universes and mechanize the metatheory in Lean 4 with zero sorry axioms. 

---
# One Fabrication Is Enough: A Fake News Article Changes the Forecasts of LLM Forecasting Systems 

**Authors**: Zeyu Zhang, Bradly C. Stadie  

**Link**: [PDF](https://arxiv.org/pdf/2611.15102)  

**Abstract**: LLMs are increasingly used for forecasting, and the best forecasting systems retrieve news as evidence to strengthen their predictions. This also opens them to attack. We study the weakest attacker of this kind, who knows only the forecasting question and publishes fabricated news articles, with no access to the model, the retriever or its queries. We evaluate how far this attack reaches on stock questions and real prediction-market questions. Once the article is in the model's context, it reverses 74% to 82% of the forecasts that opposed it, on every frontier model we tested. The model cannot verify the fabrication and follows it anyway. Retrieval is the attack's entry point, and the systems built to retrieve well are the most exposed: the architectures of two leading deployed forecasting systems, whose agents search for every related article, retrieve the fabricated one in about 90% of runs. Accuracy cannot reveal this harm: the attacked forecaster is right when the attacker's direction turns out right and wrong otherwise, so its average accuracy barely changes while its forecasts follow the attacker's guess. We also propose a second attacker, who publishes two articles that argue opposite directions to make the forecaster unsure. The model does not hedge; it returns a confident forecast in a direction the attacker no longer controls. We then evaluate the two strongest published retrieval-time defenses, passage filtering and partitioned retrieval. Each stops the one-article attack but is bypassed by a one-prompt change to the attack. Together they stop both bypasses at four articles, but sixteen distorted copies beat them, and they change 29% of clean decisions and discard 39% of the genuine news. Partitioned retrieval also requires control of the retrieval index. Systems that search the open web have no such control and are left with the filter alone. For them, the attack still works. 

---
# Science Prediction Universe 

**Authors**: Zengji Tu, Xiang Liu, Jiarui Sun, Ruoyu Wu, Zhihui Qi, Chengda Wu, Yutao Zhou, Bo Yang, Ziyan Zhang, Jialong Wu, Liang Chen, Kuan Li, Steven Dillmann, Yuanqi Du, Soojung Yang, Shizhe Diao, Jianjun Wu, Junlang Liu, Xuming Hu, Lin Chang, Baobao Chang, Xiaowen Chu, Philip S. Yu, Tong-Yi Zhang, Hao Cheng, Qin Han, Yanye Feng, Chenru Duan, Chuncheng Liu, Yan Liu, Yuchen Gao, Zhao Li, Ronghua Zhang, Tory Xu, Renjian Song, Yaokai Fu  

**Link**: [PDF](https://arxiv.org/pdf/2611.15049)  

**Abstract**: Humanity has long sought to explain natural phenomena. Reliable prediction of experimental outcomes is a central test of our understanding of the processes that produce them. If artificial general intelligence (AGI) is to surpass human scientific capabilities, it must also predict experimental outcomes more reliably than humans. Such capabilities could accelerate scientific discovery across disciplines, help address humanity's most pressing challenges, and improve human well-being. We introduce Science Prediction Universe (SPU), a benchmark of hidden experimental outcomes with 100 public verified tasks from chemistry, materials science, biology and medicine, and 16 private tasks from unpublished wet-lab experiments. Given an experimental setup and candidate conditions, an agent works with any resources it needs to rank the candidates by a measured outcome.
We evaluate thirteen agent configurations spanning twelve frontier models across two harnesses (Codex and Claude Code). Current agents remain far from saturating wet-lab outcome prediction, with substantial performance disparities across models: on the public benchmark, configurations separate into two distinct tiers, with task-adjusted Spearman correlations ranging from 0.227 for the lowest-performing configuration to 0.478 for the best. Analysis of 1,155 scored public trajectories (227,103 tool calls) suggests that the bottleneck lies in scientific judgment rather than tool execution: agents rely on shared monotone priors, agreeing far more with one another than with experimental reality (0.52 vs. 0.33), and frequently compute trivial physical or chemical proxies that do not influence the final outcome. 

---
# AROMA: Adaptive Representation-Oriented Multimodal Actuation for Vision-Language Model Control 

**Authors**: Zhaoning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.15041)  

**Abstract**: Mechanistic localization does not by itself yield usable control: causal sensitivity may occupy little activation energy, an accurate local direction may fail at finite scale, and a reachable numeral target may be wrong. We introduce AROMA (Adaptive Representation-Oriented Multimodal Actuation), a frozen-VLM control framework separating where to intervene, which geometry and gain to use, how to reach a target, which target to pursue, and when to act. In Llama-3.2-11B-Vision-Instruct, four dimensions of a causally validated cross-attention head explain 89.82% of cardinality sensitivity but only 9.52% of median activation energy; magnitude-normalized causal subspace actuation converts this mismatch into finite control, improving exact-count accuracy on untouched procedural data by +8.00 pp. Iterative Minimal-Effort Actuation (MEA) reaches 81.25% of oracle targets versus 15.18% for a frozen direction under matched step budgets. With target and gate frozen before evaluation, targeted MEA improves the previous frozen policy from 34.444% to 38.730% on a new natural Stage-2 cohort (N=630; paired p=4.63e-7) and yields 65.741% versus 65.153% over all 7,992 examples while editing only 63 inputs. Prospective routing, replication across heads and models, and comparisons with attribution-steering baselines further characterize each stage, separating localization, actuation, reachability, target quality, and selective deployment as distinct control problems. 

---
# Calibrate, Don't Retrain: Decomposing the Value of Annotator Disagreement in Cyberbullying Severity 

**Authors**: Arslan Bisharat, Katelyn Skees, Manuel Sandoval Madrigal, Mujtaba Nazari, Ayaan Khan, Deborah L. Hall, Mohammed Abuhamad, Yasin N. Silva  

**Link**: [PDF](https://arxiv.org/pdf/2611.15010)  

**Abstract**: Machine learning models are often trained on labeled data produced by multiple human annotators. When those annotators disagree, their labels are commonly collapsed into a single majority vote, which discards information about the underlying disagreement. We examine the cost of this practice for cyberbullying severity, a subjective judgment on which annotators disagree heavily. In the corpus we study, all 5 annotators assigned the same severity label to only 4.1% of comments. Turning 5 judgments into one label hides this disagreement and obscures how much the annotators differed in their assessments. In this paper, we ask whether a model can preserve this information by predicting the distribution of annotator judgments instead of a single majority label. We find 3 main results. First, accuracy against the majority label cannot tell competent models apart: 2 models with similar accuracy can differ greatly in how well they match the distribution of annotator judgments. Second, predicting the full distribution requires little extra effort. Calibrating an existing model, which adjusts its output probabilities after training, captures most of the improvement without retraining, across 3 datasets covering severity, abuse, and irony, while training on label distributions adds a small further gain. Third, we examine a dataset that includes annotator demographics, where a single shared model has one advantage and one limit. The advantage is that a model predicting the distribution of judgments matches each demographic group's judgments significantly better than a model trained on majority labels, so both groups gain. The limit is that it does not narrow the gap in performance between the groups, because one shared model cannot encode which group is right on a given comment. Preserving annotator disagreement is therefore a low-cost step toward fairer moderation, though not a complete one. 

---
# Aggregate Accuracy Hides Diagnosis-Specific Failure in Medical Vision-Language Models 

**Authors**: Shreya Ajay Kale, Angad Ripudaman Singh Bajwa  

**Link**: [PDF](https://arxiv.org/pdf/2611.14975)  

**Abstract**: Medical vision-language models (VLMs) are compared mainly by accuracy averaged across findings, a summary that can hide complete failure on a single diagnosis. We audit five open medical VLMs (MedGemma-4B, Lingshu-7B, HuatuoGPT-Vision-7B, Qwen2.5-VL-7B, and MedVLM-R1) on 6,048 ChestX-ray14 questions balanced by sex, age, finding, and label. On the 432 pneumothorax-positive X-rays, HuatuoGPT-Vision and Qwen2.5-VL detect zero cases while MedGemma and Lingshu detect 38% and 67% of the same images. The two failing models answer "no" to all 864 pneumothorax cases (balanced accuracy 0.500, chance), yet produce fluent, anatomically specific explanations that name the signs of a collapsed lung and then deny them, with no hedge or abstention. This failure is not an artifact of prompting, image resolution, label ambiguity, or model scale. It persists across four prompt formats, at native resolution, and on single-label cases. A within-model control shows the same forced prompt gives HuatuoGPT 79% recall on nodule but only 6% on pneumothorax. It survives a tenfold scale increase (Qwen2.5-VL-72B detects 4%) and replicates on SIIM-ACR, an independent dataset with radiologist-drawn masks, where the failing models detect 0 of 432 confirmed pneumothoraces, including 0 of the 144 largest lesions. Reading the models' logits and internal states shows this is not blindness. The two failing models are highly confident yet no better than chance and a linear probe recovers pneumothorax from their hidden representations even where recall is zero, so the finding is represented internally but suppressed at the output. Aggregate accuracy therefore cannot establish that a medical VLM can see a given finding, because a model that misses one can still describe it convincingly. Reporting per-finding recall and specificity on balanced cases is needed to tell the difference. 

---
# Termination Lag: Coding Agents Solve Early but Stop Late 

**Authors**: Musa Cim, Burak Topcu, Mahmut Taylan Kandemir  

**Link**: [PDF](https://arxiv.org/pdf/2611.14960)  

**Abstract**: Coding agents can solve a task well before they stop working on it. We characterize \emph{termination lag}--the number of steps between an agent's first correct patch and the end of its run--across five models, two agent frameworks, and three SWE-bench datasets by recording and testing every distinct repository patch generated during execution. Our analysis reveals that 38--82% of successful runs achieve a correct solution on their initial source-code edit. However, agents continue to expend substantial effort after correct patches, with post-solution steps accounting for a median of 35--37% for Qwen3-Coder and 54--60% for Laguna-XS.2 in successful runs on SWE-bench Verified and Multilingual. Agents spend most of these additional steps verifying their work and exploring the repository. In a small fraction of cases (1--8%), this continued activity leads agents to undo their earlier correct patch, leaving the final solution incorrect. 

---
# MuEdit: An efficient multi-task editing method towards inter-domain knowledge conflicts 

**Authors**: Songshi Liang, Ting-En Lin, Zixuan Huang, Yuchuan Wu, Yongbin Li, Zihe Wang, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.14910)  

**Abstract**: Large language models (LLMs) require frequent knowledge updates across heterogeneous domains, but existing editing methods fail when applied sequentially to multi-task settings. We trace this collapse to inter-domain null-space misalignment: heterogeneous tasks induce distinct Jacobian geometries, causing their common null space to shrink dramatically and leaving insufficient admissible directions for conflict-free updates. We propose the Conflict Index to quantify this geometric interference, and introduce Mu-Edit, which mitigates multi-task conflicts via (1) conflict-aware ordering to minimize cumulative interference, and (2) dynamic low-rank approximation to expand the null space when ordering alone is insufficient. Experiments across five functional domains and three backbones show that Mu-Edit significantly outperforms existing baselines in multi-task editing performance while preserving general capabilities, and remains effective under incremental task arrival. 

---
# Effective Teacher Support for Reasoning Distillation 

**Authors**: Lavinia Ghita, Dhruv Desai  

**Link**: [PDF](https://arxiv.org/pdf/2611.14850)  

**Abstract**: Forward-KL language-model distillation normally computes dense teacher and student scores over the full vocabulary at every supervised position, making training and stored teacher targets expensive. We instead distill from the teacher's highest-probability vocabulary fraction, renormalizing the retained mass as we sweep support from 100% to 0.01%. Our main study distills Qwen3 4B -> 1.7B. We confirm the broader support-reduction finding on larger Qwen3 8B -> 4B and Nemotron Nano 12B -> 9B pairs and under answer-only supervision. We evaluate generation primarily on MMLU-Pro and secondarily on GPQA-Diamond. Across the Qwen3 settings, retaining 0.01% of the vocabulary causes no observed MMLU-Pro degradation relative to full-vocabulary distillation, although disagreement with the complete teacher distribution increases. Nemotron Nano matches full-vocabulary MMLU-Pro at 0.1%, but smaller fractions fail to emit the closing reasoning tag, showing that the reliable cutoff depends on the model family. At 0.01% support, caching the retained next-token scores from the teacher forward pass cuts training step time by 38%, while restricting student output computation to those tokens cuts peak memory by 24%; neither causes observed MMLU-Pro degradation. 

---
# Do People Trust AI Assistants Even When They Are Consistently Wrong? 

**Authors**: Cameron R. Jones, Julia Wong, Aaron S. Cohen, James Woolley, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2611.14833)  

**Abstract**: People are increasingly using conversational AI systems to make decisions despite concerns around their reliability. How well do people calibrate to AI systems' accuracy in multi-turn conversations? We measure how people's reliance on LLMs' advice changes across repeated interactions, varying whether the AI assistant is helpful (100% accurate), unhelpful (0% accurate), or mixed (50% accurate). Across 4 experiments, we find people learn to trust helpful models more than unhelpful ones (67% vs 21% of the time), but only when they receive external feedback on the models' accuracy. Without feedback, people continue to be persuaded 48% of the time by a consistently unhelpful model (vs 38% for helpful). Rather than gaining trust in helpful models, participant reliance on LLM advisors tended to be high in early trials and only fell with evidence of inaccuracy. Participants trusted LLM advisors more than they trusted human advisors (83% vs 65%, Experiment 4), and LLMs benefited more from having a helpful track record than humans did. In contrast with previous work, we find limited evidence of an advantage for arguing for the truth (versus a falsehood). Truth advantages emerged only when participants were already very confident in their position, and did not emerge at all for static hints. Collectively, the results suggest that people may be swayed by LLM advice, even when it is consistently false, unless they have access to timely ground-truth feedback. Unfortunately, in many domains such feedback may not be available, leading to overreliance on AI. 

---
# When Is Proof-Step Verification Worth Its Cost? Operating Requirements of Certify-or-Decline Verification for Open-Weight Mathematical Reasoning 

**Authors**: Abhinav Gupta, Ben Slivinski, Yashica Patodia, Michael Saldivar  

**Link**: [PDF](https://arxiv.org/pdf/2611.14794)  

**Abstract**: Proof-step verification rewrites a solver's answer into typed steps and uses LLM judges to certify or decline. We evaluate this pipeline on open-weight models across GPQA, AIME-2025, and a MATH-500 subset. Certification raises answer-key-match precision by 31 percentage points on the strong GPQA configuration and 46 on AIME, while answering only 35% and 21% of attempts. On near-ceiling MATH-500, gains are at most 5 points at a loss of 13-27 points of coverage. An output-token-matched self-consistency selector obtains 82.9% precision at 35% coverage, compared with the verifier's pooled 87.6% (paired difference: +4.8 points, 95% interval [-8.2, 18.5]). A diagnostic sample reveals a distinct failure mode: 52 of 88 completed problems end in invalid proof construction without ever reaching a judge. Finally, external-LLM acceptance of rendered certified chains is 61.0-67.6%, compared with 87.6% answer-key match. These audits lack human or formal-kernel ground truth. Open-weight auditor pairs show near-zero chance-corrected agreement, which measures consistency rather than accuracy against proof truth. These results motivate separate reporting of execution failures, selected-answer precision, and independently assessed proof validity. We provide per-configuration results and evaluation details in the supplement. 

---
# Rational Use of Time for Social Coordination in Humans and AI 

**Authors**: Xiulin Yang, Sirisha Gudavalli, Jennifer Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.14766)  

**Abstract**: All social interactions are situated in time. Previous studies have shown that humans are able to use implicit temporal cues, such as the duration of a pause, to infer others' mental states. But how do humans use temporal cues for dynamic, fine-grained social coordination? And can AI systems do the same, without ``experiencing'' continuous real-time as humans do? We study these questions using a cooperative card game task called The Mind, which requires players to achieve a shared goal across repeated interactions but forbids explicit communication: the only available signal is the passage of time. We evaluate 150 humans, and two classes of AI models which reflect different ways of operationalizing time in artificial systems: computer-use agents (CUAs), and standard text-prompted LLMs. We find that human behavior is best captured by a Bayesian strategy that infers a partner's belief from their timing---and similarly for frontier models. However, unlike humans, none of the tested models seem to efficiently adapt their behaviors to their partners' over the course of the game. From a cognitive perspective, our findings suggest that humans perform implicit probabilistic reasoning about continuous-time signals in dynamic social settings, and that such reasoning mechanisms may emerge without real-time experience. Practically, our study highlights new challenges for AI systems to coordinate efficiently with other agents by learning to interpret and adapt to their implicit signals in dynamic settings. 

---
# Code as Explanations: Executable and Verifiable LLM Explanations 

**Authors**: Nhi Nguyen, Shauli Ravfogel, Rajesh Ranganath  

**Link**: [PDF](https://arxiv.org/pdf/2611.14704)  

**Abstract**: As large language models (LLMs) increasingly make consequential decisions and solve problems alongside humans, we must understand how models arrive at their conclusions. Understanding how an LLM behaves requires observing it across multiple inputs, yet free-text explanations such as chain-of-thought (CoT) are generated per instance and cannot be easily verified on new inputs. We propose {code explanations to address the mismatch in how LLM explanations are generated versus how they are evaluated: a program explains LLM behavior across a dataset of inputs, and its execution enables automatic verification. We define Consistency metric that measures the fraction of inputs on which the program reproduces the LLM's outputs, and use it to evaluate and select candidate programs. Across three open-source code-capable LLMs and eight tasks, we find that LLMs can produce highly consistent code explanations in several settings. Code explanations are also 6$\times$ to 100$\times$ shorter than CoT, and better align with model behavior in 9 out of 24 settings. However, consistency is significantly lower on tasks with more complex linguistic components, where generated programs reproduce model outputs no more frequently than an uninformative explanation baseline, and these programs often rely on longer hard-coded patterns. In experiments with Claude Opus 5.5, providing the model with consistency feedback and code execution improves explanation consistency across tasks, but consistency remains significantly lower on more complex tasks. Our findings establish executable code as a novel, verifiable, and compact approach to explaining LLM behavior and highlight verification feedback as a promising direction for improving explanation consistency. The code for this paper is at this https URL. 

---
# Trait Hitchhiking in LLM Post-Training 

**Authors**: Adil Soubki, Elizabeth S.Z. Tan, Miles Cranmer  

**Link**: [PDF](https://arxiv.org/pdf/2611.14683)  

**Abstract**: Reinforcement learning (RL) is now a standard stage of language model post-training, and the compute devoted to it is growing rapidly. While it has been shown to improve capabilities, particularly for tasks with verifiable rewards, it can also change behaviours and internal properties that the reward does not explicitly target. We present a formalism to describe the evolution of "traits" during RL post-training and, in doing so, recover the Price equation from evolutionary biology. We use this to develop a mathematical model for what we call "trait hitchhiking": the amplification or suppression of non-targeted traits as training redistributes probability across completions. We evaluate this Price update expression on language models trained with GRPO and show that the Price estimates closely track the sampled trait dynamics. These results identify trait hitchhiking as a mechanism by which RL post-training can silently amplify undesirable traits. 

---
# Cleaning an Agent's History Hurts When the World Still Holds the Error 

**Authors**: Egor Pakhomov, Anurag Koul, Hongjun Liu, Bo Pang, Shafiq Joty, Erik Nijkamp  

**Link**: [PDF](https://arxiv.org/pdf/2611.14654)  

**Abstract**: Tool-using language-model agents accumulate histories of their own mistakes, and a system that compacts, repairs or hands off such a history decides what it says about those mistakes while their effects stay in the world. Prior work cleans the history at a fixed world, where showing a corrected step and saying it is done move together. We ask whether the same correct content helps or hurts depending on how the history states it and what the world holds. We freeze an agent mid-task with its exact world state, edit the history or the world, and let a receiver model continue. On AgentCE-Bench, after the agent commits a wrong value, we hold the world fixed and add the correct value by two natural edits: a truthful note beside the preserved error or the agent's own completed action. The note raises recovery, the share of remaining targets the receiver gets right, from 51% under a matched neutral note to 66%. The completed action lowers it below the unedited history, from 50% to 44%, so the value's presence alone does not predict the effect. On AppWorld, after a database write, we vary whether its receipt reports success or failure and whether its effect is present. A success receipt lowers whole-task success from 45% to 28% while the effect is absent and raises it from 38% to 49% once present. The two effects differ by 27 points pooled over two receivers, a difference positive in six of eight receivers at 97.5% and seven at 95%. A history edit's effect thus depends on the world it describes, and a history made to look clean while the world still holds the error can leave an agent worse off than no edit at all. 

---
# Evaluating the Performance of LLMs in Drafting NIH Data Management Plans 

**Authors**: Nahid Zeinali, James ONeil, Xuebin Dong, Rebecca Hofstein Grady, Maria Praetzellis, Brian Riley, John Chodacki, Bhavesh Patel  

**Link**: [PDF](https://arxiv.org/pdf/2611.14648)  

**Abstract**: Sharing research data is essential to promote transparency, enable reproducibility, and fuel scientific discoveries. To ensure consistent sharing, the National Institutes of Health (NIH) and other funding agencies require applicants to submit data management plans (DMPs) with their proposals. They aim to encourage researchers to consider their data management and sharing strategies early on. However, researchers often face challenges when developing DMPs due to a lack of knowledge about data-sharing practices. Here, we evaluated the performance of Large Language Models (LLMs) in drafting DMPs compliant with the 2023 NIH guidelines. The goal was to investigate whether LLM-supported DMP creation could reduce the burden on researchers and librarians. We compared Llama and GPT performance against NIH-provided DMP samples. We also asked domain experts to evaluate LLM-generated DMPs. Overall, we found that LLMs can accelerate the creation of initial drafts but still require expert review before submission to ensure clarity and completeness. Moreover, open-source LLMs may not be as ready off-the-shelf as commercial ones for drafting funder-compliant DMPs. 

---
# Partial pooling beats synthetic respondents as estimators, not as covariates: a preregistered small-sample tournament 

**Authors**: Howard Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.14526)  

**Abstract**: Survey estimates for population segments, such as media use by sex and age, rest on few respondents per cell, and a growing literature offers synthetic respondents generated by large language models to fill the gap. That literature has not tested them against the small-area estimators a methodologist would otherwise use. We ran ten estimators, preregistered, on identical household-clustered splits of a Korean media panel across five calibration budgets and three holdout regimes. A multilevel partial-pooling model of the MRP family led every calibration-only estimator at every budget, matching at one hundred respondents what direct estimation reaches with roughly four times as many. On unobserved segments, information fixed as a per-segment prior was inert for synthetic panel and previous wave alike; attached as a bias correction it transferred, at the price of estimation noise. No registered synthetic advantage over MRP survived the paired decision rule. Entered instead as a covariate inside the multilevel model, an exploratory hybrid, the same panel improved on MRP on unobserved segments: the information survives where the registered constructions did not. Interval coverage stayed below nominal throughout and worsened with budget for partial pooling, contrary to prediction. 

---
