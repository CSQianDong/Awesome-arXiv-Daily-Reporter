# LongZero: Enhancing Long-Context Reasoning in LLMs from Zero Data 

**Authors**: Zhennan Wan, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.30132)  

**Abstract**: Reinforcement learning (RL) is an effective way to strengthen the long-context reasoning of large language models (LLMs), but it needs training data that are long, diverse, challenging, and verifiable. Human annotation of such data is expensive, naturally occurring data are too scarce to sustain RL, and existing synthesis methods depend on external corpora or stronger teacher models, which become harder to obtain as the trained models get stronger. Recently, RL from zero data, in which a model learns from tasks it proposes itself, has achieved excellent performance, but existing methods target short-context tasks. Therefore, we propose $\textbf{LongZero}$, a method that lets an LLM improve its own long-context reasoning without any external data. Instead of writing questions and answers over existing documents, LongZero first has the model synthesize the question and answer with its supporting evidence in a compact structured form, and only then separately expands the evidence into a long multi-document context. LongZero combines these self-written multi-hop reasoning tasks with self-written multi-round retrieval tasks and trains the model with GRPO. We evaluate LongZero on two LLMs across five long-context reasoning benchmarks and compare it with RL on data synthesized from external corpora. Experimental results show that LongZero can generalize to real tasks, and outperforms both baselines, suggesting a viable and promising path to improving long-context reasoning of LLMs. 

---
# Environment Duels: A Microcosm of LLM Co-Evolution 

**Authors**: Shibo Jin, Zhiqiu Xu, Yiling Zou, Shangyu Gong, Mayur Naik  

**Link**: [PDF](https://arxiv.org/pdf/2610.30116)  

**Abstract**: Through a tangle of real-world interactions, from synthetic training pipelines to data shared on the open web, frontier LLMs absorb what their predecessors and competitors create. How well does each model play its role in this ecosystem, as a creator and as a solver? We introduce Environment Duels, a dynamic agentic benchmark that evaluates both capabilities on interactive environments. Each model writes environments, each with a privileged hint, aiming for tasks that capable solvers fail without the hint and pass with it. Across nine frontier LLMs, solving and authoring prove to be distinct skills: the strongest solvers are not the strongest authors, while smaller models learn more from how others played and write the hardest environments and the most useful hints. Beyond the scores, the pairwise interactions form a microcosm of how models affect one another. Head-to-head outcomes are transitive and so yield a stable ranking; independent authors converge on the same designs; and in a reinforcement-learning study with Qwen3.8-27B, the learner's hint lift on an author's environments tracks how much training on those environments improves it on held-out reasoning and agentic benchmarks, whether the environments are its own or another model's. Each of these patterns has a counterpart in how models shape one another outside the duel. By staging these interactions under controlled conditions, Environment Duels offers a window onto how LLMs teach, challenge and learn from one another. Code is available at this https URL. 

---
# Quant-Arena: Benchmarking LLM Agents for Quantitative Trading Research 

**Authors**: Houde Qian, Zheng Jiang, Wenyi Li, Chenxi Xu, Xinjie Lin, Yuzhao Peng, Ruoyu Han, Youjie Zheng, Kaisen Yang, Qinhuai Na, You Fu, Calvin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.30113)  

**Abstract**: Quantitative research is an iterative loop: a researcher proposes a market hypothesis, implements it as an executable signal, tests it on historical data, and decides what to keep. Existing benchmarks score formulas, code, or individual trading decisions, leaving open whether large language models (LLMs) can complete this loop and produce strategies that work beyond the research period. We introduce Quant-Arena, a benchmark and public leaderboard for end-to-end quantitative research on real A-share market data. A model constructs features and stock-scoring code, receives feedback only on 2022--2023, and freezes one strategy for assessment on the sealed year 2024. A cash-account evaluator applies trading constraints and costs, including market impact estimated from a simulated order book. Scores are dominated by realized excess return and complemented by signal quality and the diversity of features used by the strategy. We evaluate nine frontier models, among which GPT-6 Astra scores highest; yet no model beats the equal-weight market in 2024. Quant-Arena is open to submissions from any model or agent harness. 

---
# Did the Problem Actually Get Harder? Testing LLMs on Paired P and NP-Complete Problems Hardened by Adversarial Multi-Agent Rewriting 

**Authors**: Nurul Labib Sayeedi, Md. Faiyaz Abdullah Sayeedi, Zabir Al Nazi, Mahbub E Sobhani, Shubhashis Roy Dipta  

**Link**: [PDF](https://arxiv.org/pdf/2610.30103)  

**Abstract**: Many NP-complete (NPC) problems have a polynomial-time (P) twin that differs in a single structural detail, such as 3- versus 2-colouring a graph. A model that reasons rather than recalls should notice which side of this boundary a problem falls on, yet existing benchmarks either fix the wording of each task or ignore its complexity class, so they cannot separate the two. We introduce P2NPCBench, a benchmark of 12 problem families in which every NPC problem is paired with a P twin posed over the same surface. To remove the textbook cues that enable recall, we build it with AgentGAN, an adversarial multi-agent framework in which a generator rewrites each statement until a panel of frozen code models fails, while an executable validity gate keeps every rewrite answer- and class-preserving. The result is 2,114 hardened statements with hand-verified reference human solvers. Evaluating 12 proprietary and open-source models, we find that every model scores lower on NPC statements than on their P twins (35.4% vs. 63.8% pass@1), and that 95.1% of model-family pairs fail more often on the NPC side. A tenfold time limit raises NPC accuracy only from 28.5% to 31.9%, so failures stem from wrong or crashing programs rather than slow ones. Hardening also narrows the class gap from 24.8 to 14.6 points, showing that part of the apparent gap reflects familiarity with textbook phrasing rather than complexity class. 

---
# GoiBERT: Compact Vietnamese Language Models via Two-Stage Knowledge Distillation 

**Authors**: Quang Anh Kiet Tran, Nam Anh Tran, Manh Hung Trinh, Thanh Danh Le, Khang Nhat Hoang Vo, Huynh Dai Nhan Tran  

**Link**: [PDF](https://arxiv.org/pdf/2610.30067)  

**Abstract**: Vietnamese pre-trained encoders have improved rapidly, but the strongest ones are large, multilingual-vocabulary models or depend on external word segmentation, which makes them costly to deploy. We present GoiBERT, a family of two compact Vietnamese encoders with 26M and 49M parameters that operate directly on raw text. GoiBERT learns from two complementary teachers. A same-tokenizer Vietnamese encoder first teaches general-purpose representations. A much larger teacher then provides task-level guidance through soft labels that are generated without ever labeling its own training data. Across eight benchmarks covering text classification, aspect-based sentiment analysis, natural language inference, and named entity recognition, GoiBERT-large achieves the highest average score among non-teacher models. It does so with fewer than half the parameters of the strongest compact baseline and twice its throughput. GoiBERT-base retains 99% of that baseline's performance at a quarter of its size. Our analysis shows that compression is nearly lossless for sentiment and classification tasks but remains costly for inference and entity recognition, which points to clear directions for future work. Our code and models are available at: this https URL 

---
# Looped Uniform Diffusion Language Models 

**Authors**: Saiteja Utpala, Chieh-Hsin Lai, Yuki Mitsufuji, Praneeth Vepakomma  

**Link**: [PDF](https://arxiv.org/pdf/2610.30066)  

**Abstract**: Uniform diffusion language models (UDLMs) generate text by iteratively denoising sequences of ordinary vocabulary tokens, any of which may be corrupted and subsequently revised. During training, UDLMs must denoise from a context that may itself be unreliable, whereas masked diffusion language models (MDLMs) always condition on a clean context. This challenging training objective makes UDLMs more FLOP-intensive to train than MDLMs, but it yields stronger downstream reasoning performance. Recently, selective looping, which recursively applies a few early Transformer layers within each denoising step, has been shown to improve training efficiency in MDLMs under an iso-FLOP budget. We investigate looping in UDLMs while keeping the training objective and sampler unchanged. We find that looping significantly improves training efficiency: looped UDLMs reach the validation NLL of a non-looped UDLM trained on about 0.5 trillion tokens with up to 2.8x fewer training FLOPs on OpenWebText and 3.3x fewer on FineWeb-Edu. We also find that, at a matched NFE budget, looped UDLMs reach up to 9.8% lower generation perplexity than the non-looped UDLM, with most gains achieved at moderate loop counts. Finally, we study different loop-count distributions and find that sampling smaller loop counts more frequently during training improves inference efficiency: on downstream tasks, this strategy uses 12% fewer loops than uniform loop-count sampling without degrading accuracy. 

---
# What the Author Wrote, What the Agent Reads: Silent Cue Loss in Enterprise Document Ingestion 

**Authors**: Yikai Zhao, Saurabh Pandey, Pradeep Kumar Misra  

**Link**: [PDF](https://arxiv.org/pdf/2610.30026)  

**Abstract**: Enterprise agents act on documents. Before an agent can follow a policy, answer a compliance question or file a ticket, some ingestion layer must turn a Word file or PDF into tokens - and that layer is where an agent deployed in the wild silently loses information. We show that one layer is lost by every mainstream converter - the meaning an author encodes in how text looks - and that the loss makes agents confidently wrong rather than visibly incomplete. We build a task from 76 real policy documents: recover the requirements the author singled out, scored against the file's own OOXML properties by a deterministic matcher with no LLM judge. Formatting-blind ingestion collapses from micro F1 0.690 to 0.150 (docling) and 0.075 (plain text), a gap that replicates across six answering models from two vendors - five Claude models spanning the Opus, Sonnet and Haiku tiers, and OpenAI's GPT-5.6. An agent that cannot see the cue guesses, and docling's false-positive rate on requirements the author did not flag is four times ours. A span-level leave-one-out localises the loss to two cues - highlight (+0.302) and colour (+0.262), both Holm-adjusted p < 1e-4 - while bold, italic, monospace and strikethrough each sit within +/-0.003 of zero. The same asymmetry appears in PDF. Colour is also the cue no mainstream converter preserves and no parsing benchmark scores, and it is common: 32.8% of 6,460 documents carry a deliberate colour or highlight, 10.3% on an obligation. A pre-registered test shows that typing a cue for the agent adds +0.085 beyond merely preserving it (Holm-adjusted p = 0.049). We validate the labels against rendered page images through an independent modality (kappa = 0.72) and bound what wording alone can achieve (AUC 0.664). For agent builders the practical finding is that ingestion, not reasoning, is where this reliability is won or lost, and that the fix is cheap and measurable. 

---
# Building Auditable AI Text Detectors with Sparse Autoencoders 

**Authors**: Liam Dugan, Callum McDougall, Matthew Denton, Philip Pham, Christine Kaeser-Chen, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2610.30006)  

**Abstract**: AI-generated text detectors are increasingly deployed in consequential settings---peer review, education, admissions, journalism, and content moderation---where their outputs directly affect individuals. Yet the strongest detectors are fine-tuned neural classifiers whose decisions are opaque, while transparent alternatives (stylometric and linguistic-feature classifiers) underperform them by wide margins. We present a pipeline that converts a fine-tuned neural detector into a classifier that operates on human-readable concepts. We accomplish this by training sparse autoencoders (SAEs) directly on a detector's residual stream, assigning auto-interpretation labels to each SAE feature, and retaining only features whose generated descriptions pass a global fidelity threshold. We then train a lightweight classifier on just the interpretable features. We find that our SAEs recover latent concepts that correspond to highly specific and predictive linguistic "tells" of AI-generated text. Classifiers built on these extracted features are not only highly interpretable but exhibit stronger performance on out-of-distribution texts than either neural or stylometric alternatives. 

---
# Remembering Through Change: Provenance-Preserving Hypergraph Memory for Multimodal Agents 

**Authors**: Jielong Tang, Ling-An Zeng, Xujie Yuan, Xiao Dong, Zhiqian Xia, Qingfeng Li, Shixiong Kai, Kaichao Liang, Rong Yao, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.29995)  

**Abstract**: Long-term memory for multimodal agents remains challenging, as it must preserve fine-grained visual evidence while tracking entity evolution across sessions. Existing abstraction-based memories compress images into text, often discarding visually discriminative details, whereas pixel-preserving memories retain raw observations but struggle to reliably associate diverse observations with persistent entity identities and track their evolving states. To address these limitations, we propose MHG-Mem, a multimodal hypergraph memory that jointly organizes entity identity, state evolution, and provenance to visual evidence. Specifically, MHG-Mem represents distinct entity mentions and their observed facts as individual nodes, grouping identity-consistent mentions and facts through occurrence hyperedges. It then synthesizes aspect-specific evolution states from these grouped observations and links them to supporting entities, facts, and source evidence through evolutionary and evidential hyperedges. To navigate this structured memory, we introduce hypergraph-based agentic search, which adaptively selects an entry layer, traverses typed hyperedges across memory levels, and inspects source observations when fine-grained verification is required. These designs support both detailed visual verification and reasoning over cross-session state changes. Extensive experiments across three benchmarks demonstrate the effectiveness of MHG-Mem, improving performance by 8.20-9.98% over the strongest baseline. 

---
# CulturalRiddles: A Multicultural Benchmark of Riddles 

**Authors**: My Chiffon Nguyen, Tianyi Hu, Vinod Anbalagan, Henry Gagnier, Pouya Sadeghi, Farah Abdou, Tidor-Vlad Pricope, Diego MMS Oliveira, Jocelyn K D'Arcy, Madeline Smith, Oluwaseun A Ajayi, Yusser Al Ghussin, Zena Al-Khalili, Micol Altomare, Shayan Amir, Ahanaf Aziz, Bryan Chen, Ruchira Dhar, Abdellah El Mekki, Rawan Mohammed Elframawy, Akhil Eppa, Silvia Fernandez, Alex Gimei, Arya Hariharan, Christian Hoang, RamMohanRao Kadiyala, Diana Kozachek, Peerat Limkonchotiwat, Ankita Maity, Di Mo, Ayush Munot, Aylin Naebzadeh, Anastasiia Nikitina, Dina Pisarevskaya, Divas Poudel, Jebish Purbey, Md Rabius Sany Apu, Ahmad Mufli Ramadhan, Ralitza Nikolova Rizova, Diogo M. Serrano-Miranda, Vivek Gopalkrishna Silimkhan, Luis Fernando Solis Navarro, Kaiser Sun, Arya Suneesh, Manaswitha Sunkara, Tejaswi Sunkara, Jonathan E Swindell, Hanna Yukhymenko, Jingyi Zheng, Esther Adenuga, Ilia Afanasev, Wafa Aissa, Lisa Alazraki, Lauren C Altomare, Priscilla Amondi Amuok, Emile Anand, Gabriel Assis, José Matías Avalos, Silvia Lorena Avila, Zakaria Baannou, Niyati Bafna, Yash Bagla, Sarah Bahrami, Bontu F Balcha, Vanya Bannihatti Kumar, Irene Baucells, Abdumutalib Begmuratov, Naman Bhatia, Souvik Bhattacharyya, Isabel Blanco Ferreiro, Felipe Bochini, Sayuru Bopitiya, Angana Borah, Tulika Bose, Olena Bugaiova, Juliana Carvalho, Nancy F Chen, Stephy Chen, Rochelle Choenni, Jayanth Krishna Chundru, Leandro A Cossio, Nebojša Damnjanović, Ona de Gibert, Marco De Santis, Pedro Delgado, Kaustubh Dholé, Adnan El Assadi, Wesam El-Sayed, Nadine M Elframawy, Oumayma Essarhi, Naome A Etori, Reuben Chagas Fernandes, Saanidhi Pravin Gade, Javier García Gilabert, Álvaro González-Escapa, Bin Hu, Kaiyu Huang, Abderrahmane Issam, Danka Jokić, Ulugbek Khalilov  

**Link**: [PDF](https://arxiv.org/pdf/2610.29968)  

**Abstract**: As LLMs expand their language coverage, evaluating how well they understand and represent diverse cultures becomes increasingly important. However, many benchmarks remain English-centric, with limited cultural coverage and a focus on factual knowledge over its application in real-world cultural contexts. We propose CulturalRiddles, a benchmark of original, culturally grounded riddles created entirely by humans through a participatory community effort, spanning 62 cultures (45 countries, 51 languages). These riddles are deeply rooted in their respective cultures and require abductive and compositional reasoning, alongside detailed factual and social knowledge, language skills, and an understanding of culturally specific associations. We benchmark 28 LLM configurations using both automatic and large-scale human evaluation. Even frontier models achieve at most 22% accuracy in some communities. We publicly release all artifacts alongside an open contribution process to expand the benchmark's linguistic and cultural coverage. 

---
# Towards English - Mizo Legal Machine Translation: Corpus Creation, Analysis, and Benchmarks 

**Authors**: Telem Joyson Singh, Sanabam Boyshana, Sanasam Ranbir Singh, Priyankoo Sarmah  

**Link**: [PDF](https://arxiv.org/pdf/2610.29943)  

**Abstract**: Indian court proceedings are held in English, so speakers of languages with limited translation technology have limited access to them. We present the first English Mizo parallel dataset of Indian court judgments. It contains 13,168 sentence pairs from the Gauhati High Court and the Supreme Court of India. Professional native Mizo translators translated complete documents; sentence pairs were manually aligned, and two independent reviewers checked the results. Our study finds that English words are borrowed in a very selective way: official and legal terms are almost always kept in English, but words for everyday things like actions, situations, and results are usually expressed in native Mizo. Mizo diacritics are also used inconsistently across the corpus. When we fine tune the NLLB 200 1.3B model on our training data, it scores 27.47 BLEU on a balanced legal test set of 1,000 sentence pairs, an improvement of 7.16 BLEU over the base model. However, it still uses English words about three times more often than a human translator does. 

---
# Graph Topology Meets Energy-Driven Pathfinding: Preserving Contextual Consistency in Retrieval-Augmented Generation 

**Authors**: Yufeng Li, Zaijing Li, Fuchen Zhang, Zongzong Wu, Ming Zhao, Fengxiao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29933)  

**Abstract**: Retrieval-augmented generation (RAG) frequently integrates knowledge graphs (KGs) for complex multi-hop question answering. However, their performance is increasingly bottlenecked by the knowledge structuring and reasoning processes. (i) Severe semantic loss: Distilling continuous texts into discrete symbols strips away contextual nuances and narrative coherence. (ii) Pronounced modality mismatch: Fragmented graph structures clash with the continuous texts LLMs are pre-trained on, while rewriting them into natural language introduces generative noise and omits information. To address these limitations, we propose ThreadRAG, a semantic-topology-based RAG framework for context-preserving knowledge structuring and modality-native evidence reasoning. ThreadRAG introduces a Semantic Topology Mapper (STM), which organizes retrieved documents into context-intact Semantic Nodes and constructs a Semantic Topology Graph (STG) through narrative, semantic, and lexical edges. It further incorporates Energy-Driven Pathfinding (EDP), which identifies query-relevant subgraphs via energy diffusion and deterministically linearizes them into coherent textual evidence paths. Experiments on three multi-hop QA benchmarks demonstrate that ThreadRAG consistently outperforms strong baselines. Notably, it achieves a 14.4% EM improvement over text-centric methods on HotpotQA and surpasses the best baseline by 5.8% EM on 2WikiMultihop, while reducing token cost to 23.3% of the strongest baseline. 

---
# Task-Preserving Multilingual Instruction Tuning for Open-Ended QA in WMT 2026 MIST 

**Authors**: Aleix Sant, Jordi Luque, Carlos Escolano  

**Link**: [PDF](https://arxiv.org/pdf/2610.29926)  

**Abstract**: This paper describes our submission to the WMT 2026 Multilingual Instruction Shared Task. We adapt Qwen3-8B with LoRA using organiser-provided data, EuroAlpaca, a multilingual instruction dataset created through task-preserving localisation, and CrossEuroAlpaca, a cross-lingual augmentation that assigns different source and target languages to instructions, contexts and responses. Our system targets open-ended question answering (QA-OEG) in ten shared-task languages represented in the auxiliary data. Under matched decoding, adaptation improves the mean Gemma-4 judge score on target-language QA-OEG by 7.8% relative to the base model. On an additional evaluation using the Aya Evaluation Suite across all shared-task languages, it yields relative gains of 41.5% in macro-averaged ROUGE-L and 2.9% in BERTScore-F1, with larger improvements in target languages. However, these gains are task- and language-specific. Gemma-4 scores improve for QA-OEG and summarisation in the target languages but decline for context-grounded QA there and across all three tasks in the non-target languages. 

---
# Unintended Side Effects of Value Training in LLMs 

**Authors**: Siddhesh Pawar, Chen Cecilia Liu, Anna Korhonen, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2610.29922)  

**Abstract**: As LLMs are increasingly deployed in value-sensitive contexts, aligning them with human values has emerged as a post-training objective in NLP, aimed at instilling additional values and guiding model behaviour. However, current evaluations remain narrow. Because values are interconnected, it is unclear how fine-tuning for specific values alters unrelated capabilities and behaviours. In this work, we adapt four instruction-tuned LLMs across individual Hofstede's value dimensions and evaluate them on both cross-value transfer and broader behaviours. We find that value-specific training produces unintended side effects on untrained values, risk preferences, social cognition, and safety traits. Consequently, we suggest that value alignment after post-training should incorporate broader behavioral evaluations, and we highlight the need to investigate where additional value training might best fit into the pipeline. 

---
# Benchmarking Large Language Models for Detecting Satirical Code-Mixed Sentences 

**Authors**: Răzvan-Alexandru Smădu, Ioana Buhnila, Daniela Tanase, Dumitru-Clementin Cercel, Florin Pop, Mihaela-Claudia Cercel  

**Link**: [PDF](https://arxiv.org/pdf/2610.29907)  

**Abstract**: Code-switching is a natural linguistic phenomenon, yet its impact on high-level reasoning tasks remains largely unexplored in under-resourced languages like Romanian. Natural language processing tasks, such as satire detection, rely heavily on nuanced cultural and syntactic cues that large language models struggle to capture. In this paper, we investigate four methods (i.e., random replacement, phrase translation, dependency parsing, and large language models) for generating synthetic code-mixed Romanian sentences with English, German, French, and Italian. We start with an existing dataset for satire detection, compiled from Romanian news articles and annotated at the sentence level by humans, and augment it with over 600k samples, namely CS-SeLeRoSa. We provide an in-depth analysis of the dataset generation process. Then, we evaluate several multilingual large language models across zero-shot, chain-of-thought, and fine-tuning regimes on both the synthetic corpus and a human-adjudicated gold benchmark, namely CS-SeLeRoSa-Gold. While older open models suffer substantial degradation (dropping to 37%-55% macro-F1), newer models like Gemma 4 31B achieve 68.1% zero-shot, rivaling frontier models like GPT-5 mini (69.5%). Furthermore, fine-tuning significantly reduces formatting errors and improves the results, increasing Gemma 4 31B to 77.5% on CS-SeLeRoSa and 72.6% on the gold standard, outperforming frontier zero-shot baselines. 

---
# Investigating the Impact of Learner Errors on Social Bias in Large Language Models 

**Authors**: Federico Marcuzzi, Martin Christoph, Iryna Gurevych, Alla Rozovskays  

**Link**: [PDF](https://arxiv.org/pdf/2610.29896)  

**Abstract**: Existing social bias benchmarks evaluate large language models mostly on fluent, well-formed English, reflecting native speaker usage and leaving non-native writers unrepresented. To address this gap, we investigate how grammatical errors by non-native writers affect social bias. We develop a controlled framework that injects common learner errors into prompts and apply it to three bias benchmarks covering stereotype alignment and demographic preference across gender, race, and religion. We find that grammatical errors increase bias, and the effect grows with the error rate. Among the four common error types we consider, preposition errors produce the largest and most consistent increase in stereotype alignment. Our analysis suggests that errors increase bias when they are more likely to interfere with comprehension. Finally, fine-tuning on erroneous texts reduces the bias increase caused by grammatical errors. Our findings show that bias evaluations based exclusively on well-formed prompts may underestimate the bias experienced by users whose English contains grammatical errors. 

---
# Every Part Has Its Role: Part-of-Speech Effects in LLM Post-Training 

**Authors**: Laya Iyer, Chen Shani  

**Link**: [PDF](https://arxiv.org/pdf/2610.29843)  

**Abstract**: Large language models are typically trained with next-token prediction (NTP), which rewards predicting specific tokens rather than explicitly accounting for the semantic similarities between different words or expressions. Prior work has addressed this limitation by augmenting training data with minimally varying sentences that replace a target word with conceptually related alternatives, e.g., "The tired coach ran home" $\rightarrow$ {"The tired trainer ran home", "The tired instructor ran home", ...}. Such concept augmentation improves performance, but it remains unclear whether its effectiveness depends on where in the sentence the concept is expressed. We reproduce and extend prior work $\textit{beyond nouns}$ to verbs and adjectives, asking whether the part of speech (POS) carrying the concept signal affects perplexity and downstream performance. We find different patterns across evaluation settings: concept augmentation improves perplexity most consistently on news, independent of the POS used, whereas verbs provide the most reliable downstream gains. Despite these behavioral differences, representational analyses using CKA and linear probing show that the conditions yielding the largest downstream gains are not necessarily those that most alter the model's internal representations. Together, these results suggest that the effectiveness of concept augmentation depends on where and how conceptual information is introduced during training. 

---
# S-EDL: Eliciting Self-Evidence from Sequence Likelihoods for Semantic Calibration of LLMs 

**Authors**: Yawei Li, Jiazheng Li, David Rügamer, Bernd Bischl, Yulan He, Mina Rezaei  

**Link**: [PDF](https://arxiv.org/pdf/2610.29832)  

**Abstract**: Calibrating large language models (LLMs) in open-ended generation is uniquely challenging because many distinct token sequences can express the same underlying meaning. Prior calibration techniques target either fixed-set classification confidence or the binary correctness of individual strings, making them inherently ill-suited for the dynamic, multi-string nature of open-ended semantic outcomes. To bridge this gap, we introduce S-EDL, a novel evidential method that natively optimizes LLMs for semantic calibration. By eliciting evidence from the sequence likelihoods of the model itself, S-EDL constructs a differentiable Dirichlet prior over prompt-specific semantic classes. Optimizing the evidential loss under this prior yields a calibration-aware training signal over variable semantic classes, completely bypassing the need for fixed multiple-choice options. Evaluations across four open-ended benchmarks and three models demonstrate that S-EDL substantially reduces semantic calibration error while improving accuracy. By aligning generative likelihoods with semantic correctness, this work establishes a framework for deploying trustworthy LLMs in complex, open-ended applications. Code is available at: this https URL. 

---
# MemoryTax: The Bitter Lesson of Agent Memory 

**Authors**: Zedong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.29823)  

**Abstract**: Agent memory systems such as Mem0 and Zep decide what to keep before any question arrives; coding agents keep the raw history and let the model search it with its own tools. We ask whether designed memory still helps once the model is strong enough to search for itself. On one LoCoMo conversation (419 turns, 152 questions), we run four memory methods (full context, BM25, Mem0, Zep) and three coding agents (Pi, Codex, Claude Code), each with GPT-4o-mini, GPT-6 Luna and Claude Opus 5.5, and record what every rollout costs. This is a short empirical study, with one conversation and one run per cell. Agent memory is a costly patch for a weak model: Mem0 beats full context only with GPT-4o-mini, and its build alone costs more than a whole full-context rollout. Agent memory also has a ceiling, set by the model that writes it: a stronger reader does not recover what a weaker writer left out. With Claude Opus 5.5, Pi tops the cost-accuracy frontier, scoring 99% for less than half the cost of full context; its one miss is a wrong answer key. Part of that lead comes from photo captions that only the agents could read: on the 88 questions whose evidence has no photo, Pi and full context differ by one answer. 

---
# MAPS: Macro-Adaptive Parallel Reasoning for Regulating Search Momentum in Retrieval-Augmented Generation 

**Authors**: Xinze Li, Dingwen Zhang, Zhenghao Liu, Yukun Yan, Shuo Wang, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.29792)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves the performance of Large Language Models (LLMs) by retrieving external knowledge. Recently, some RAG methods interleave reasoning and retrieval, enabling LLMs to progressively generate sub-queries and retrieve external knowledge. However, these methods may suffer from redundant or ineffective sub-queries and insufficiently diverse query exploration. Moreover, due to the tightly interleaved nature of reasoning and retrieval, it remains unclear how the evolving reasoning process influences query intent. To characterize the evolution of query intent during reasoning, we introduce search momentum, which captures the accumulated retrieval tendency induced by historical changes in generated sub-queries along the reasoning trajectory. Through momentum-based analysis, we find that the query generation process tends to continuously evolve along existing retrieval trajectories. Based on this observation, we propose MAPS, a Macro-Adaptive Parallel Reasoning framework for regulating search momentum and mitigating query-intent drift. MAPS first retains the retrieved knowledge and a refined reasoning state, thereby reducing the influence of historical reasoning trajectories on subsequent sub-query generation. Subsequently, MAPS extends the conventional sequential reasoning process into a parallel reasoning paradigm. At each iteration, MAPS identifies complementary retrieval directions, generates corresponding sub-queries for parallel retrieval, and consolidates the retrieved evidence to update the knowledge and reasoning states. Experiments on six question-answering benchmarks demonstrate that MAPS consistently outperforms representative RAG methods across different backbone models. All code and resources are available at this https URL. 

---
# Where Do LLMs Learn to Recommend AI? 

**Authors**: Benaya Trabelsi, Jonathan Shaki, Sarit Kraus  

**Link**: [PDF](https://arxiv.org/pdf/2610.29785)  

**Abstract**: Asked what to study, which career to pursue, or what product to build, large language models (LLMs) often recommend AI-related options. We ask where this tendency comes from by comparing models before and after fine-tuning and by varying how questions are asked. First, pretrained models already include explicit AI/ML terms in the recommendations of 27% of their responses, and their released instruction-tuned versions do so in 59%. Second, fine-tuning models ourselves shows that this rise is not inevitable: general instruction tuning leaves the rate unchanged, but removing the 7% of instruction examples that concern AI cuts it by 14 percentage points, below the level of the pretrained models it starts from. Data rewarding helpfulness raise it by 9 points, while data ranked by the preferences of diverse simulated users lower it by 9. Finally, with models held fixed, asking for the "best" option or for what to do "right now" raises the rate by 8 and 18 points, whereas telling the model it is an AI advisor adds 4. Overall, the tendency starts in pretraining, and what models learn afterward, together with how a question is phrased, decides how strongly it appears. 

---
# Evolving with Open Worlds: Towards Recursive Self-Improvement for Future Forecasting 

**Authors**: Zhipeng Chen, Yunsen Liang, Yanxi Chen, Hongkang Zhao, Lingyun Feng, Ruiyang Ren, Guoxin Chen, Jia Deng, Feng Chang, Chuan Hao, Ran Tao, Jian Yang, Ji-Rong Wen, Bryan Dai, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.29784)  

**Abstract**: Most contemporary artificial intelligence (AI) systems are released with frozen parameters and improve through external updates, whereas an ideal system would keep improving from its own experience, a capacity we call recursive self-improvement (RSI). Yet studying RSI requires an environment that is continuously changing, objectively verifiable, and naturally grounded, posing a fundamental challenge for agent evolution. Here we introduce DeepEvolve, a sustainable evolution agent system grounded in future event prediction, in which agents continuously interact with a dynamically evolving environment and improve by internalizing and externalizing the experience gained through these interactions. A living environment synthesizes verifiable tasks, and a recursive-evolution engine turns the reasoning trajectory behind each prediction back into capability: decoupling search from prediction adapts parameter-level evolution to the contamination-prone live environment, while distilling reusable skills lets the agent evolve from the limited live trajectories such an environment affords. On the external FutureX leaderboard, which evaluates predictions against real-world outcomes and is updated weekly, our self-improved forecaster ranked first across all systems over three consecutive weeks, achieving up to a 195% improvement in rating. On our synthesized evaluation, it ranked among the top three across four base LLMs, outperforming the strongest baseline by up to 10.7 Bradley--Terry points. More broadly, these results suggest that a system built for self-improvement is best understood as an agent that co-evolves with a genuinely living, open-ended environment, pointing to a shift in AI from training static models on fixed data toward cultivating agents that grow stronger from their own experience, with such an open world as the indispensable basis of that growth. 

---
# ALIAS: Evaluating Performance and Memorization in LLM-Written TLA+ Specifications 

**Authors**: Arslan Bisharat, Mujtaba Nazari, Khushboo Bhadauria, Brian Ortiz, Eric Spencer, Beatriz Santos, TaiNing Wang, George K. Thiruvathukal, Konstantin Laufer, Mohammed Abuhamad  

**Link**: [PDF](https://arxiv.org/pdf/2610.29767)  

**Abstract**: Memorization inflates the benchmark scores of large language models (LLMs), since a model can reproduce answers it saw in training. Researchers test for it with consistent identifier renaming (CIR), which gives every identifier in a task a new name, and treat a lower score on the renamed task as evidence of recall. We test this method on the Temporal Logic of Actions Plus (TLA+) language, whose model checker acts as an oracle by confirming whether a renamed task preserves the exact behaviors of the original. Across 111 TLA+-Bench tasks, renaming lowers the correctness of Claude Opus 4.8 from 28% to 18% and of GPT-5 from 9% to 7%. Most of this decrease comes from how the test grades, which requires exact names that most descriptions never state. Grading by behavior shrinks the decrease to about 3 percentage points for Opus 4.8 and to 0 for GPT-5. In two model families fine-tuned on disjoint halves of the tasks, renaming removes about half of the training gain in likelihood and costs 4.9 percentage points (Qwen) and 1.3 percentage points (Gemma) on unseen tasks. A lower score after renaming therefore mixes memorization, name-based grading, and the cost of new names. 

---
# Bridging Foundational and Agentic Capabilities of LLMs 

**Authors**: Wei Guo, Yuanjian Xu, Jianing Hao, Yongyao Duan, Ran Chen, Zhong Li, Guang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29751)  

**Abstract**: Large language models (LLMs) are increasingly deployed as agents, yet it remains unclear how their foundational capabilities translate into agentic capabilities. End-to-end scores leave unresolved whether low benchmark capability or weak deployment drives a failure, limiting actionable diagnosis. We bridge foundational and agentic capabilities through an operation-level measurement framework. We represent each model with a foundational profile derived from capability benchmarks and an operation profile derived from interaction trajectories, and define capability realization gaps as deviations of observed operation performance from predictions based on foundational capabilities. Experiments on fifteen model services in $\tau^2$-Bench show that foundational profiles only partially predict agentic operations. Models with comparable profiles can differ substantially in execution, and these gaps remain consistent across task subsets. Operation profiles add information and substantially improve task-score prediction. The framework locates execution departures at the operation level to inform model development and agent evaluation. We will release our code and analysis artifacts. 

---
# Accept or Repair: Preserving Passage-Supported Answers in Contrastive Decoding 

**Authors**: Jungseob Lee, Seungyoon Lee, Hyeonseok Moon, Seongtae Hong, Suhyune Son, Sugyeong Eo, Minhyuk Kim, Chanjun Park, Jaehyung Seo  

**Link**: [PDF](https://arxiv.org/pdf/2610.29727)  

**Abstract**: Large language models often answer from knowledge learned during training, even when a provided passage contradicts that knowledge. To address this problem, contrastive decoding contrasts the predictions under the full and a modified prompt at every token, but it also alters correct answers that the passage supports. We show that checking answers against the passage removes this trade-off. Based on this observation, we propose Accept-or-Repair (AoR), a training-free procedure. AoR keeps a sampled answer if the passage supports it. Otherwise, it tries up to two replacements with contrastive decoding, returning one as soon as it passes this check. It rejects replacements that repeat the answer generated without the passage when the passage favors another answer. Our experiments show that AoR raises the score over sampling by 12.5 points on counterfactual passages and 2.3 on ordinary ones, surpassing eight contrastive and prompting baselines on both, including when each receives the same rejection rule. Replacing only initial answers that the passage does not support improves the average score of every decoder. These results suggest that models can follow passages that contradict their knowledge while preserving accuracy on passages that agree with it. Code will be released at this https URL. 

---
# Retrieval Augmented Generation Compositionality 

**Authors**: Ido Levy, Segev Shlomov, Leshem Choshen  

**Link**: [PDF](https://arxiv.org/pdf/2610.29722)  

**Abstract**: Retrieval Augmented Generation (RAG) grounds large language models in external knowledge. Modern pipelines combine embedding, chunking, retrieval, reranking, query augmentation, context compression, and agentic verification, yet these techniques are typically evaluated one at a time. It remains unclear whether their isolated gains survive composition, whether retrieval quality and answer faithfulness favor the same design choices, and whether conclusions from small corpora hold at large scale. We pre-declare three hypotheses and introduce LASAGNA, an open-source framework for controlled RAG evaluation. We test 450 configurations across eight benchmarks and corpora ranging from 5K to 200K documents. Three findings emerge. First, individually strong techniques from different phases of a RAG pipeline do not necessarily compose: three retrieval techniques predict a +47% combined improvement but deliver only +17% when used together. Second, component gains stack differently for retrieval and faithfulness, and optimizing one can hurt the other. Third, the best composition changes as the corpus grows: the combination that wins on a small corpus may not remain best at large scale. These findings show that gains measured in isolation are not reliable predictors of realistic performance, and that new RAG techniques should be tested with the full surrounding stack. We call on researchers to do so and supply LASAGNA to simplify these experiments. 

---
# Scaling Multilingual Pretraining of Pixel Language Models 

**Authors**: Ilker Kesen, Jenny Kunz, Kasra Malihi, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2610.29681)  

**Abstract**: Pixel language models process text as rendered images instead of a finite vocabulary, which gives them strong cross-lingual transfer abilities. However, not much is known about how these abilities are influenced by larger-scale pretraining. In this paper, we train a family of MPIXEL models along three axes of variation: more languages, more scripts, and larger model capacity under fixed per-language compute. In evaluations on five tasks covering 83 languages, we find that scaling acts locally: each axis improves the added languages and scripts, while increased capacity mitigates the curse of multilinguality from pretraining on more languages. Transfer to unseen scripts improves with multilingual pretraining, but does not further improve with scaling. On typologically distant low-resource languages, our models outperform subword and character-level baselines across many writing systems. We also show that larger pixel models reserve more layers for visual processing, and that cross-lingual alignment weakens as more languages are added. Overall, our findings characterize how multilingual scaling improves pixel language models. 

---
# Uni-Mem: Learning a Long-Horizon Agentic Memory Manager with Unified Online Policy Optimization 

**Authors**: Zhiqian Xia, Jielong Tang, Ling-An Zeng, Shixiong Kai, Rong Yao, Hao Kong, Xiao Dong, Wei Wang, Mingxuan Yuan, Siyu Xia, Haifeng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2610.29671)  

**Abstract**: Long-term agent memory is shifting from predefined workflows toward learnable policies for memory construction and retrieval. Existing learning-based methods either optimize construction alone or combine retrieval-side online reinforcement learning (RL) with offline construction supervision, limiting co-adaptation between the two stages. In this paper, we explore end-to-end online RL for a unified memory manager (Uni-Mem) that learns both capabilities under a shared policy. However, joint optimization with existing RL algorithms such as GRPO faces confounded group comparisons and sparse feedback over long memory-management trajectories. To this end, we propose Stage-Decoupled GRPO, which alternates construction and retrieval optimization while controlling variation from the other stage within each rollout group, yielding stage-specific learning signals. We further introduce Event-Wise Credit Assignment (ECA), which localizes feedback to responsible events and propagates event-level advantages to their constituent operations. Experiments on LoCoMo and LongMemEval show that Uni-Mem outperforms the strongest RL-based memory, demonstrating the effectiveness of jointly learning memory construction and retrieval via Stage-Decoupled GRPO and ECA. 

---
# Effects of Model Scale and Quantization on Social Bias and Abstention in Question Answering 

**Authors**: Kuniko Paxton, Ifeoluwa Wuraola, Dhavalkumar Thakker  

**Link**: [PDF](https://arxiv.org/pdf/2610.29657)  

**Abstract**: The deployment of small and quantized language models in resource-constrained settings raises an important fairness question: when a compressed model produces fewer biased responses, does this reflect fairer behavior or greater abstention? We study this problem in Question Answering (QA) using the BBQ benchmark across multiple model sizes and quantization settings from Qwen3, Gemma-3, and SmolLM2. We also examine whether the observed trends remain stable across alternative prompt formulations and when the ``Unknown'' option is removed in a forced-choice setting. Our results show that larger models achieved higher QA accuracy, but bias showed no simple scaling trend and quantization effects varied across configurations. Notably, 20 of the 26 configurations with fewer biased responses also show increased over-abstention, indicating that lower observed bias can partly reflect greater abstention. These findings show that lower observed bias can partly reflect greater abstention, and that fairness evaluation should consider biased responses, abstention, prompt formulation, and answer-option design jointly. 

---
# LLMs as survey respondents: Comparison to simple semantic baselines 

**Authors**: Esko Lehtonen, Tommi Buder-Gröndahl  

**Link**: [PDF](https://arxiv.org/pdf/2610.29645)  

**Abstract**: Large language models (LLMs) have been explored as surrogates for human survey respondents. A more promising application is data augmentation, where LLMs predict new responses from partial human data. However, the semantic similarity between questionnaire items is a known source of correlation between answers. It is thus unclear whether LLMs actually infer meaningful relationships between question-response pairs, or use surface-level comparison between questions as a heuristic. To assess this, we evaluated how three LLMs can predict responses to held-out items in two surveys measuring attitudes toward automated or light-electric vehicles, based on varying proportions of known responses. Performance was compared to two baselines, where new responses were derived by maximizing embedding-similarity to a prior response. While LLMs mostly predicted human responses better than baselines, their performance consistently declined with a lower share of known responses, even aligning with the baseline in one survey. Our results show that LLMs' capacities for augmenting replies varied across surveys, and can revert to the same levels as simpler embedding similarity baselines when less data is available. 

---
# Overview of CHIP 2026 Shared Task 1: Patient-Specific Phenotype Concept Recognition in Full-Text Biomedical Literature 

**Authors**: Xuanlyu Zhu, Feng Tian, Yang Zhong, Pengqi Zheng, Xiwen Bi, Ling Luo, Yuanyuan Sun, Jian Wang, Lingfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.29606)  

**Abstract**: Automatic phenotype information extraction plays an important role in rare disease diagnosis, genotype-phenotype analysis, and precision medicine. However, accurately identifying phenotype concepts and determining their associations with specific patients remain challenging, particularly in full-text biomedical articles involving multiple patients and complex clinical descriptions. This paper presents an overview of CHIP 2026 Shared Task 1, which aims to promote the development of methods for patient-specific phenotype concept recognition from full-text biomedical literature. The task consists of two subtasks: full-text phenotype concept recognition and patient-specific phenotype association. For this evaluation, we used 200 annotated full-text biomedical articles from the PatientPheX dataset, containing 506 patients, 15,193 phenotype mentions, and 5,656 patient-phenotype associations. A total of 455 teams registered for the competition, with 189 and 103 valid submissions received on the Phase A and Phase B leaderboards, respectively. The best-performing system achieved an overall score of 0.7389 on the final test set, outperforming the baseline by 8.37 percentage points. The top-performing teams explored diverse approaches combining biomedical language models, large language models (LLMs), and ensemble strategies. The evaluation results highlight both the potential of these approaches and the remaining challenges in patient attribution and phenotype normalization from full-text biomedical literature. The post-evaluation phase remains open at this https URL. 

---
# A Diagnostic Study of Faithfulness in Diffusion Language Models for Table-to-Text Generation 

**Authors**: Zaynab Merimi, Noussayba El Marrakchi, Arthur Satouf  

**Link**: [PDF](https://arxiv.org/pdf/2610.29602)  

**Abstract**: Diffusion language models generate by iteratively denoising a masked sequence under bidirectional attention, without the strict left-to-right commitment of autoregressive decoding. Whether this difference affects how reliably they remain grounded in structured inputs is largely unexplored. We study this question in table-to-text generation using ToTTo, where highlighted source cells make factual grounding directly testable. We compare the masked diffusion models MDLM and LLaDA with GPT-2 and GPT-Neo under an identical retrieval-based few-shot protocol, together with a QLoRA-adapted LLaDA, and we score reference similarity and source grounding as separate axes. The two pull apart. Within LLaDA, additional demonstrations raise BLEU from 11.3 to 22.0 while NLI entailment against the source table falls from 0.78 to 0.52; QLoRA reaches the best BLEU of the study, 24.15, and grounds no better for it. Manual annotation shows the failure modes differ in kind: the autoregressive baselines tend to wander off the table, while LLaDA-based models more often stay on topic and swap out individual entities. Reference-based metrics alone are insufficient to assess faithfulness in table-to-text generation. 

---
# PACE: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States 

**Authors**: Patrick Haller, Filipe Laitenberger, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2610.29545)  

**Abstract**: Linear-attention language models keep their context in a recurrent state of fixed size, where an inference system stores one copy of this state for every sequence it decodes at the same time. Storing the state at lower precision therefore lets more sequences share an accelerator. Existing state quantizers round the state after every generated token and report that four bits lose accuracy unless the bit width is raised or the model is fine-tuned.\par We introduce PACE, a post-hoc four-bit state quantizer. PACE rounds the state only once every few generated tokens, as chunked kernels already do once per chunk during prefill, and keeps the tokens generated in between in a small buffer. Rounding less often reduces the error that accumulates in the state. Additionally, we find that the large entries of the gated-delta state gather in columns, because some channels of the update written at each token remain large across many tokens. PACE therefore shares each quantization scale within one column. It also keeps the heads that retain rounding error longest at eight bits. On four hybrid models from 4B to 35B parameters, PACE with mixed four- and eight-bit heads is competitive with eight-bit DAMP on the three larger models at about two thirds of its memory, PACE with four-bit heads throughout scores higher than Q-Mamba at equal memory, and rounding every few tokens preserves the decoding speedup of the buffer. 

---
# RCSD: Reinforcement and Continual Learning for Domain-Specific Speculative Decoding 

**Authors**: Mingkang Liu, Weiming Meng, Peipei Jiao, Jiawei Zhang, Tianyuan Liu, Junyan Li, Lichen Pang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29527)  

**Abstract**: Speculative decoding accelerates large language model inference by employing a compact draft model to propose candidate tokens that are verified in parallel by a target model. Conventional training relies on soft-label distillation, which forces the draft model to mimic the target's full output distribution, including high-entropy regions that degrade acceptance rates and increase early rejection risk. We propose RCSD (Reinforcement and Continual Learning for Speculative Decoding), a training framework that aligns draft model optimization with the system-level objective of maximizing acceptance length. RCSD leverages naturally occurring accepted and rejected sequences during speculative inference to construct implicit preference pairs, with masked loss to avoid penalizing target-aligned tokens. Evaluated across multiple benchmarks, RCSD improves average acceptance length by 35% over HASS and achieves 1.35x faster end-to-end inference than HASS on practical workloads, without compromising generation quality. 

---
# Mitigating Query Shortcuts in Chain-of-Thought with Context-Aware Decoding 

**Authors**: Shin'ya Yamaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2610.29518)  

**Abstract**: While Chain-of-Thought (CoT) prompting improves the reasoning performance of large language models (LLMs), recent studies have shown that final answers do not always depend on the generated rationales. In particular, LLMs may rely on query-specific shortcuts rather than the information provided by their reasoning steps. In this paper, we investigate whether context-aware decoding (CAD), an existing contrastive decoding method, can mitigate such shortcut behavior when applied to CoT reasoning. We study this application as Rationale Contrastive Decoding (RCD), which contrasts the next-token distributions conditioned on the query with and without its rationale, amplifying tokens whose likelihood increases due to the rationale. We provide a theoretical interpretation of this decoding rule via a KL-regularized reward-maximization objective. Experiments across five reasoning benchmarks demonstrate that RCD improves answer accuracy over standard CoT. Furthermore, intervention analyses using ground-truth and random rationales reveal that RCD increases the behavioral dependence of final answers on the rationale content. Our findings suggest that contrastive decoding offers a simple, training-free approach to mitigating query shortcuts. 

---
# Keeping Time vs. Understanding Time: Neither Alone Explains Temporal Reasoning in LLMs 

**Authors**: Auss Abbood, Fernando Martínez-Plumed, José Hernández-Orallo, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2610.29507)  

**Abstract**: Temporal reasoning is widely assumed to decompose into two modes: temporal arithmetic (date and time calculations), and temporal semantics (understanding temporal and causal relations). This dichotomy shapes how benchmarks are designed, how models are trained, and how progress is measured, yet its empirical validity has never been directly tested. We provide two lines of evidence that challenge it. First, we test whether tasks labelled as arithmetic or semantic by the benchmarks support transferable gains within their purported mode. Finetuning on these subsets yields predominantly dataset-specific gains ($+6.8$ to $+23.1$ percentage points), while transfer to a different dataset in the same mode ranges from $-3.2$ to $+1.1$ percentage points. Across the two benchmark families used in the transfer analysis, this pattern provides little evidence of coherent, transferable temporal abilities. Second, using cognitive-demand-based model evaluation, we examine which capabilities actually drive performance on these benchmarks. We find that quantitative and logical reasoning, the assumed proxies for temporal arithmetic and temporal semantics, are not the primary drivers; knowledge is. Together, these findings show that current temporal reasoning benchmarks do not cleanly realise the arithmetic-semantics distinction and instead largely reward models for remembering how time typically works. 

---
# A Strong MoE Is All You Need for Process Supervision 

**Authors**: Guanzhi Deng, Zengyan Liu, Haibo Wang, Kuan Wu, Shing Yin Wong, Sichun Luo, Zhuoran Wang, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.29498)  

**Abstract**: Process reward models (PRMs) provide fine-grained supervision for multi-step reasoning, yet obtaining reliable process labels remains a central bottleneck. Existing approaches typically rely on human annotation, rollout-based estimation, or auxiliary verification. In this work, we explore a different source of process supervision natively exposed by Mixture-of-Experts (MoE) models: expert routing. We show that simple routing statistics provide an intrinsic process-confidence signal that is competitive with conventional output-logit confidence. Building on this observation, we construct routing confidence by combining overall routing mass, head routing mass, and within-step routing variation, and introduce RoCo-Rethink, which identifies the least confident reasoning step and selectively regenerates the subsequent trajectory. We further introduce RoCo-PRM, which uses a strong MoE to automatically annotate reasoning trajectories with routing-derived step scores and distills them into a standalone PRM. Across multiple MoE architectures and diverse reasoning tasks, both RoCo-Rethink and RoCo-PRM consistently outperform strong baselines, while RoCo-PRM transfers effectively across both model architectures and tasks. Our findings suggest that expert routing is more than a computational dispatch mechanism: it provides an architecture-native source of process supervision for reasoning. 

---
# TenetBench: A Clause-Anchored Russian Corpus for Evaluating Language Models against Professional Codes of Ethics 

**Authors**: Yegor Anichkov, Valery Vinogradov, Andrey Furov, Marina Polukoshko, Sergey Bolovtsov  

**Link**: [PDF](https://arxiv.org/pdf/2610.29477)  

**Abstract**: Codes of professional ethics guide and restrict specialists, who increasingly work with assistants built on large language models. Such an assistant can breach the user's code, or help the user breach it, without the user noticing. Evaluations reporting broad categories of harm cannot show how often this happens or which provision is breached. TenetBench pairs 3,433 requests conflicting with clauses of fifteen Russian-language professional codes with benign twins that differ only in the legitimacy of what is asked. An automated pipeline generates the tasks from the text of a code, so the corpus grows with every code added and a provision yields new tasks on every run. It also produced tasks and human-validated verdicts from two English codes of other jurisdictions. In human validation of the frozen pipeline, specialists rated 0.995 of the conflicting tasks plausible, and the model judge agreed with the adjudicated label at Gwet's AC1 of at least 0.894 on 1,860 elements. We evaluated fourteen model and reasoning-mode combinations on whether they produce the conflicting text, decline or warn. Eleven of the fourteen produce conflicting content on most such requests, mostly without a warning, and nearly three warnings in four accompany conflicting content. The models that conform more do so by declining while still serving the legitimate twin, and enabling reasoning halved the refusals of two models. We hope the dataset and pipeline will help evaluate how language models observe professional norms and let diagnostic datasets be built for particular professions and organisations. 

---
# OmniSimulator: Aligning Small Language Models for Authentic Heterogeneous Behavior Modeling 

**Authors**: Jiawei Chen, Ruoxi Xu, Boxi Cao, Zhuoqun Li, Ruotong Pan, Yunfei Zhang, Zhirui Yang, Jingyan Chen, Tingting Gao, Han Li, Yaojie Lu, Xianpei Han, Le Sun, Xiangyu Wu, Hongyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.29472)  

**Abstract**: Large language models (LLMs) have shown great potential as general-purpose user simulators for interactive systems. Prior studies have demonstrated that training on high-quality data can substantially improve model reasoning and downstream task performance. However, the upper bound of user simulation is still constrained by the model's reasoning paradigm: native chain-of-thought often degenerates into shallow pattern matching and fails to adequately model the behavioral factors underlying real user decisions. In this paper, we propose OmniSimulator, a novel training framework designed to improve the predictive ability of small LLMs for realistic user simulation on short-video platform environments. Specifically, OmniSimulator organizes behavior-oriented reasoning through a structured schema with five dimensions, carefully selects data from real user-behavior scenarios, and introduces a teacher model to generate high-quality reasoning traces under the proposed schema, helping small models learn behavior-relevant reasoning patterns rather than relying on superficial behavioral correlations. Based on this process, we construct 9,881 high-quality training instances from three months of historical interaction trajectories of 200 real users, averaging about 50 distilled examples per user. Experimental results show that OmniSimulator improves the average score of Qwen3-8B by 34.33%, achieves performance competitive with Claude-Sonnet-4.5, and supports smaller-scale, self-hosted deployment. These results suggest that learning structured behavior-oriented reasoning patterns is a promising direction towards scalable and high-fidelity LLM-based user simulation. 

---
# Omnilingual-MsASR: Benchmarking Multimodal, Multilingual, Long-Form Multi-Speaker ASR of Omni Large Language Models 

**Authors**: Yuan Ge, Pei Zhang, Xize Cheng, Lingzhou Mu, Mingyang Wang, Jin Xu, Baosong Yang, Shengxiang Gao, Jingbo Zhu, Zhengtao Yu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.29455)  

**Abstract**: Omni-modal large language models (LLMs) have achieved remarkable progress in audio and video understanding. However, existing models struggle with long-form, multi-speaker videos, especially in understanding the entire video and recognizing who says what and when. To characterize the capability limit of current omni LLMs and guide future model development, we introduce Omnilingual-MsASR, a multilingual, multimodal benchmark for long-form Multi-speaker Automatic Speech Recognition (MsASR) covering six languages, recordings up to 30 minutes, and scenarios with up to 20 speakers. Omnilingual-MsASR provides parallel audio and video inputs to evaluate models' cross-modal consistency and their ability to exploit complementary cues across modalities. Extensive evaluations of leading omni-modal LLMs reveal substantial performance bottlenecks, while their tcpWERs frequently exceed 100%. These findings highlight fine-grained temporal localization, speaker disambiguation, and cross-modal synergy as key challenges for future omni-modal foundation models. Our work is available at this https URL. 

---
# DynaWeave: Learning to Adapt through Dynamic Scenario Synthesis 

**Authors**: Xuan Yang, Chunxiao Liu, Hongsheng Xin, Hao Xu, Ning Miao  

**Link**: [PDF](https://arxiv.org/pdf/2610.29443)  

**Abstract**: Large language model (LLM) agents increasingly use tools to complete tasks in stateful environments. However, many training environments remain largely static apart from changes caused by the agent's own actions, whereas external events introduced by other users, services, or sudden contingencies in real environments can invalidate ongoing plans. To address this gap, we introduce DynaWeave, an automated synthesis framework that transforms existing static tasks into verifiable dynamic scenarios. The framework generates exogenous events relevant to each task, covering changes in task requirements, environment state, and external information. It uses directed acyclic graphs (DAGs) to represent dependencies between events and an execution engine to activate them during agent interaction. A dynamic verifier revises only the evaluation checks affected by events that actually occur and retains all other checks. Using DynaWeave, we collect 2,095 verified interaction trajectories across 36 environments and use them for supervised fine-tuning of Qwen3.5 models. Experiments on Gaia2, CostBench, and ToolMaze show that DynaWeave achieves consistent improvements in overall performance across all three evaluated model sizes, increasing the task completion rate on CostBench by 67.34 percentage points and even exceeding the performance of Gemini-3.1-Pro-Preview on ToolMaze. These results support dynamic scenario synthesis as a practical approach to training agents to adapt to changing environments. Our code is available at this https URL. 

---
# A KAN-based Non-linear Factorization for Embedding Layer Compression 

**Authors**: Oliver Giudice, Anna Russo Russo  

**Link**: [PDF](https://arxiv.org/pdf/2610.29424)  

**Abstract**: Among existing embedding compression approaches, low-rank factorization provides a simple and mathematically principled solution by approximating the embedding matrix with a reduced-rank decomposition. However, its practical effectiveness is often limited, as the resulting linear bottleneck constrains representational capacity. To overcome these limitations, we propose an enhanced factorization scheme that increases expressive power by introducing learnable non-linear projections within the compressed embedding layer. Our method is inspired by the Kolmogorov Arnold Networks (KAN) paradigm, which replaces fixed linear transformations with learnable univariate functions. Despite promising results in other domains, applications of KANs in NLP remain scarce and have so far been largely limited to final or task specific layers, rather than core representational components such as embedding layers.
We evaluate the proposed approach by integrating the compressed embedding layer into the Skip-gram framework and comparing it against a standard low-rank baseline under matched parameter budgets on the Word Analogy and Semantic Similarity benchmarks. We further introduce an effective initialization strategy from a pre-trained, uncompressed embedding layer. The behavior of the model is analyzed both immediately after initialization and after a brief post-initialization refinement phase, considering training with the standard Skip-gram objective as well as variants augmented with distillation from the original embedding layer. Overall, our results show that augmenting low-rank factorization with learnable non-linear projections effectively mitigates the limitations of linear bottlenecks, yielding more faithful compressed embeddings under fixed dimensionality and suggesting a promising direction for embedding layer compression beyond purely linear methods. 

---
# Beyond Small Talk: Modeling Disclosure Trajectories in Human-AI Conversation 

**Authors**: Keith Kwok, Wanting Ning, Tej Deep Pala, Desmond C. Ong, S. Supraja, Renwen Zhang, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2610.29413)  

**Abstract**: Recently, we see a rise in conversational agents as a form of social and emotional support, where disclosure will unfold over time. Yet, current evaluations of these agents typically examine isolated conversational states. In analyses of naturally occurring conversations, disclosure history and current disclosure are confounded which tends to make their effects difficult to isolate. We thus introduce Trajectory-based Disclosure Evaluation (TraDE), an evaluation framework grounded in Social Penetration Theory that utilizes predefined disclosure trajectories and evaluates how preceding disclosure history shapes model behavior under matched current disclosure. Through the use of TraDE, we demonstrate across multiple models and relational behavior benchmarks that responses to disclosures matched on depth can differ depending on the preceding trajectory, particularly in whether models accept or challenge the user's framing. We also observed that these differences track accumulated exposure to deeper disclosure rather than any single moment of vulnerability or the order of disclosures. It also tends to persists for several turns after disclosure returns to shallow levels. For several behaviors, we also demonstrate that models diverge most consistently from human supporters in the Emotional Support Conversation corpus (ESConv) not when users open up but rather when they step back in their disclosure. TraDE thus provides a systematic way to expose history-dependent relational behavior that evaluations of isolated conversational states would otherwise miss. 

---
# ECHO: Emergent Common-Mode Hidden Offsets Drive Vocabulary Collapse in Preference Optimization 

**Authors**: Yun Li, Ehsan Javanmardi, Simon Thompson, Jiachen Gong, Qunli Zhang, Zifan Zeng, Shiming Liu, Peng Wang, Zixuan Guo, Manabu Tsukada  

**Link**: [PDF](https://arxiv.org/pdf/2610.29410)  

**Abstract**: Preference optimization can improve every ranking metric while building a hidden, shared representation shift that the relative loss barely sees, one that can eventually destroy generation. We trace this failure to the common-mode offset, a shared directional shift in the model's hidden states that appears across token positions, prompts, and both chosen and rejected responses. Because it moves chosen and rejected representations together, the relative ranking loss barely reacts to its accumulation. As this offset grows, it projects onto the output layer to form a vocabulary well, elevating a single token's probability until it dominates decoding and causes generation collapse. The isolated offset alone reconstructs the vocabulary well, injecting the learned offset into an untreated base model recreates the collapse where norm-matched random directions do not, and masking this common component from the loss prevents the well in our flagship setting. Under the per-token ranking loss we study, the same collapse appears in 14 of 17 base-model lineages and under full-parameter fine-tuning. Moreover, the offset is not merely an artifact of collapse, since it is already present in textbook DPO and SimPO runs that form no well. Textbook DPO keeps it weakly aligned with any single vocabulary direction even with three times our base budget. Trained from base models for that budget, however, full-parameter SimPO drives it into collapse on both Llama-2 and Llama-3.1, and the same kind of offset grows, without a well, from a public SFT checkpoint under the official SimPO recipe. Across 17 UltraFeedback runs, the better a run ranks held-out responses, the less often two public reward models prefer its samples to the base model's. These findings expose a blind spot in standard ranking metrics and introduce a representation-level signal that can warn of generation collapse before it occurs. 

---
# Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories 

**Authors**: Boyuan Zheng, Jing Xiong, Chenwei Wu, Hui Shen, Zesen Zhao, Hei Ting Una Chan, Xueshen Liu, Zhen Zhang, Liyue Shen, Zechun Liu, Z. Morley Mao  

**Link**: [PDF](https://arxiv.org/pdf/2610.29378)  

**Abstract**: Quantization-aware training (QAT) aims to recover reasoning capabilities in low-bit large language models (LLMs). Existing recipes typically supervise the quantized student with the original full-precision model on fixed response sequences. However, the resulting low-bit quantized models still struggle on long-horizon reasoning tasks and often fail to terminate within a token budget that suffices for the full-precision model. We identify two challenges in closing this gap. First, fixed-response supervision leaves a mismatch between training prefixes and those encountered during student generation, and this trajectory divergence grows as responses lengthen. Second, on its own prefixes, the student underpredicts some next-token candidates and overpredicts others, and this distributional misalignment becomes more frequent at high-entropy prefixes. We present OnQ, a low-bit recovery pipeline that combines on-policy distillation and candidate-wise Dynamic KL. Starting from block-wise calibration, the quantized student generates responses and receives full-precision teacher supervision on its own prefixes. Dynamic KL weights forward and reverse KL by each candidate's teacher-student probability ratio to balance correction of under- and overprediction. Across Qwen3-1.7B, 4B, and 8B at 2- and 3-bit precision, OnQ achieves the highest nine-benchmark average, with the largest gains on long-horizon reasoning benchmarks. At 2 bits, it outperforms every baseline on every benchmark at every model size, improving the Qwen3-8B average by 10.26 points over the strongest baseline. It also produces shorter responses with fewer truncations and repetitive errors. 

---
# Memory-Centric Continual Learning for AI Agents 

**Authors**: Dongfang Li, Zhuoen Chen, Bonian Jia, Zixuan Liu, Zhengxuan Lu, Can Deng, Yukun Shi, Yuhang Meng, Xin Zhang, Jiahe Huang, Fuhao Li, Junmai Wang, Shunyao Wen, Jingjun Sun, Jiaxu Tian, Jiangling Zhang, Zihan Wang, Liangyou Li, Yitong Yao, Shuangyong Song, Beilun Wang, Pengxu Wei, Hanlin Xu, Chang Liu, Yongxiang Li, Baotian Hu, Xiangyang Ji, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29357)  

**Abstract**: Autonomous agents powered by foundation models can reason, plan, and use tools, but their fixed weights and context limits hinder sustained learning from experience over long-term deployment. While external memory systems attempt to bridge this gap, they mostly act as passive retrieval caches, evaluated primarily by whether they can recall stored facts or past interactions. Whether memory can drive genuine continual learning remains an open question. In this paper, we introduce Memory-Centric Continual Learning (MCCL), arguing that memory should be judged by whether it improves future actions, not merely what it retrieves. MCCL has three main parts. First, a three-axis taxonomy organizes memory by where state is stored (substrates), what form it takes (assets), and how it is managed (policies). This defines a two-loop lifecycle that updates memory state in an inner loop while refining management policies in an outer loop based on task feedback. Second, an evaluation protocol measures retention under interference, adaptation to change, and task transfer to isolate genuine behavioral gains from simple context expansion. Third, a memory scaling framework models how agent performance scales with experience, storage capacity, and compute budgets across non-parametric and parametric memory. Together, these components reframe agent memory from passive lookup to continuous learning, offering a principled route toward self-improving autonomous agents. 

---
# Don't Stop Long-Context Continual Pre-Training Too Early: An Empirical Scaling Law 

**Authors**: Ziqing Qiao, Xiaoyue Xu, Sikui Zhang, Yinuo Xu, Bingxiang He, Chaojun Xiao, Xu Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.29352)  

**Abstract**: Long-context continual pre-training (LCCP) is widely used to improve long-context capability, but how its benefits scale with the number of tokens used for LCCP remains unclear. We investigate this question through modeling, monitoring, and validating the benefits of scaling LCCP. First, we establish an empirical additive power law relating long-context performance to model size and the respective token budgets of short-context pre-training and LCCP. Experiments across five model scales and more than one hundred short- and long-context budget configurations show that the fitted law reproduces the observed trends and extrapolates well to held-out model scales. Second, we introduce \textsc{Constant-Last4}, a training and observation protocol that averages the four most recent consecutive checkpoints from a constant-learning-rate run to enable stable monitoring along a single extensible training trajectory. Under this protocol, the same scaling-law form fits well and accurately predicts the LCCP performance trajectory of a 1.22B-parameter model trained on approximately 100B LCCP tokens. Finally, downstream evaluations along this trajectory show continued improvements in long-context performance, with no clear sign of saturation within the evaluated range. These gains persist after both 128K context extension and supervised fine-tuning, with benefits extending to long-context understanding and agentic tasks. Together, these results argue for continuing to scale LCCP rather than stopping early: its gains are stably observable along an extensible trajectory, quantitatively predictable, and transferable to downstream tasks. 

---
# B-CARE: Blind Classwise Assessment of Relative Evidence for Stance Error Ranking 

**Authors**: Meysam Sabbaghan, Doina Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2610.29350)  

**Abstract**: Stance classifiers make errors that can be costly to identify under limited human-review budgets. Prioritizing potentially incorrect predictions remains challenging, particularly across heterogeneous base models. We introduce B-CARE (Blind Classwise Assessment of Relative Evidence for Stance Error Ranking), a prediction-blind framework for identifying high-risk stance predictions. Given only a target and text, a fixed LLM verifier assigns ordinal evidence scores to all candidate stance labels without observing the base model's prediction or confidence. B-CARE constructs a comparative risk score by contrasting the strongest alternative-class evidence with the evidence supporting the predicted class. The resulting classwise evidence profile can be reused across multiple base models without training a separate error detector or calibrating risk scores. We evaluate B-CARE on four stance benchmarks and three heterogeneous base LLMs, covering 12 dataset-model settings and 5,206 unique test instances. Evaluation includes error-ranking discrimination (AUROC and AUPRC), selective prediction (E-AURC), and review effectiveness (ErrorCapture at fixed review budgets). B-CARE achieves a mean AUROC of 0.735 and a mean AUPRC of 0.524, capturing 41.06% of errors within the highest-risk 20% of predictions. Although B-CARE does not uniformly outperform strong confidence-based baselines, it provides a reusable, complementary evidence signal for prioritizing potentially incorrect stance predictions across heterogeneous models. 

---
# ChronoSynth: Constraint-Aware Question Synthesis over Temporal Knowledge Graphs 

**Authors**: Feng Liang, Weixin Zeng, Runhao Zhao, Mingjun Zhou, Shaoshi Yang, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.29339)  

**Abstract**: Knowledge graph question generation (KGQG) generates natural-language questions from structured graph evidence and a target answer. When the input graph is temporal, a valid question must also preserve the answer field, time condition, reference event, and temporal order that determine the answer set. Existing KGQG methods and prompting baselines can produce fluent questions that alter these conditions, causing temporal semantic drift. This paper studies temporal KGQG as a constraint-preserving graph-to-question problem and proposes ChronoSynth, a memory-augmented and constraint-aware generation framework. ChronoSynth first constructs a Temporal Intent State to make the intended question semantics explicit. It then retrieves compatible typed question patterns from an indexed pattern memory, adapts them to the input graph, and validates the generated question. Validation feedback guides revisions when a supported inconsistency is detected. Experiments on CRONQUESTIONS and MultiTQ demonstrate state-of-the-art generation performance, with ablations supporting the contributions of its components. 

---
# Policy Central: Detecting Conflicts, Redundancy, and Similarity in Deployed Police Policy 

**Authors**: Holly Baker, Michael Kuo, Kyana Pollas  

**Link**: [PDF](https://arxiv.org/pdf/2610.29338)  

**Abstract**: Policy Central is a deployed system that helps police departments find relationships between sections of their written policy: whether two sections conflict, are redundant, are merely similar, or are different. We describe the system and compare two architectures. Retrieve-Judge, a retrieve-then-classify pipeline, runs as a cheap library-wide health signal; Screen-Confirm, a screen-then-confirm cascade, is a selective audit of the documents a reviewer chooses, checking them exhaustively against one another. On a 302-pair benchmark drawn from one department's ~84 real policy documents, given each pair Screen-Confirm reaches 0.795 aggregate recall and 0.885 on Conflict, so current models judge well. Finding the pair is the hard part: Retrieve-Judge's composed recall falls to 0.457, and Screen-Confirm recovers 0.472 (0.615 on Conflict). Today these pairs are found by reading the whole library, nearly 20,000 sections, checked against one another and re-checked after every revision. A scan that returns half of them in minutes, at low cost, and can be re-run on demand changes the economics of policy review even before recall improves. 

---
# Seeing the Forest for the Trees: Emergent Agreement from Individual Conversations 

**Authors**: Adi Simhi, Yaniv Nikankin, Jonathan Herzig, Idan Szpektor, Yonatan Belinkov, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2610.29328)  

**Abstract**: As LLM agents interact in the wild and shape each other's decisions, understanding how they converge toward agreement becomes important. We study this in multi-turn conversations where two LLMs discuss subjective matters, can agree or disagree, and may change their stances. Using a linear classifier on individual conversation activations, we identify an internal representation of agreement. We find that it varies substantially across turns and conversations and does not reliably track external agreement at the individual conversation level. We therefore move beyond the representation of agreement in individual conversations, by aggregating the agreement representation across conversations. Externally, this aggregate view reveals a simple structure: when pooled across conversations, agreement can be predicted well with a Markov process, with agreement at each turn being a direct function of agreement at the previous turn. Similarly, when internal agreement is aggregated, it couples with the external Markov agreement through a power-law mapping, exposing Markovian dynamics in internal representations. Additionally, we find that after a change of stance, models express agreement externally before their internal representations reflect the same shift. These results reveal a joint structure between internal and external agreement that emerges only when internal features are aggregated across conversations. Code available at this https URL. 

---
# Signals and Queries: Token-Efficient Agentic Context Discovery 

**Authors**: Yam Marcovitz  

**Link**: [PDF](https://arxiv.org/pdf/2610.29326)  

**Abstract**: User-facing LLM agents must identify which behavioral policies apply at each turn, but evaluating every policy with an LLM is costly. We introduce Signals and Queries (S&Q), a lightweight first-stage relevance filter developed for Parlant 4.0 that removes at least 30% of candidate policies before downstream evaluation while preserving high recall. For each policy, an LLM generates activating examples and hard negatives offline. An independent logistic classifier trained on their embeddings then predicts relevance using embeddings and vector dot products, with no per-turn LLM calls. This shifts expensive relevance evaluation to a smaller candidate set, while allowing each policy's classifier to be updated independently. Evaluated against 9,144 LLM-judge relevance labels covering 254 $\tau^2$-bench conversation prefixes and 36 airline policies, S&Q achieves higher precision than top-K and cosine-threshold retrieval at every evaluated recall level, with consistent results across two embedding providers. These findings support inexpensive, per-policy classification as a practical first stage for reducing downstream evaluation workload in customer-facing agents. 

---
# OSynth: A Longitudinal, Context-Sensitive Evaluation of Meeting Understanding 

**Authors**: Ian Porada, Suraj Tripathi, Chaitanya Animesh, Leda Sari, Chaitanya Narisetty, Austin R. Matthews, Bohan Yu, Ramesh L. Thulasiram, Gregory Sell, Kean Chin, Dmitriy Genzel  

**Link**: [PDF](https://arxiv.org/pdf/2610.29319)  

**Abstract**: Meeting notetakers are increasingly deployed in settings where longitudinal context is available including calendar metadata, prior meetings, emails, slides, and speaker enrollments; however, meeting corpora remain limited in terms of size, domain coverage, and available context. We present OSynth, a large-scale synthetic dataset of simulated organizations comprising 570 hours of meeting audio together with associated artifacts and a latent ground-truth state. In a pairwise comparison, human raters distinguish OSynth meetings from real meeting transcripts with only 56% accuracy based on excerpts of 150-300 words. Across state-of-the-art LLMs, additional context yields significant gains on several entity-sensitive dimensions, demonstrating limitations of contemporary approaches and the potential usefulness of OSynth in developing better-contextualized meeting-understanding systems. 

---
# Asking What a Director Sees Can Change What Vision-Language Models Choose 

**Authors**: Jiaxuan Guo, Xingyu Ren, Kejia Zhang, Jingxin Yang, Youran Sun, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29318)  

**Abstract**: Vision-language models are now routinely tested on theory of mind: whether they can work out what another person sees, knows, and means. A common design asks the model to report that perspective and then to act for the person, and reads a correct report beside a wrong action as a failure to use the perspective. That reading rests on a quiet assumption: that asking for the report leaves the action unchanged. We show that it does not. In a rendered Director Task, a partition hides the largest block from a director who asks for "the largest block". Asking first which blocks the director can see raises the share of replies that choose this hidden block by 12 to 88 percentage points on six GPT models and by 17 on Qwen3.8-27B, even though the models mostly report correctly which blocks the director sees. Reasoning traces and prompt rewordings suggest why: the question shifts how the request is read, from the director's view toward the model's own. When the prompt states what the director knows and refers to, fewer than 1% of replies pair a correct report with a hidden-block choice. Any evaluation that elicits a perspective report before the action it scores needs a baseline without the question, and we deliver one: paired prompts that differ only in that question, a design any perspective-taking benchmark can adopt. 

---
# Credit Where Credit Is Due: Contrastive Attribution Topology Search for Multi-Agent Systems 

**Authors**: Fabian Tritthart, Tom Zehle  

**Link**: [PDF](https://arxiv.org/pdf/2610.29292)  

**Abstract**: An LLM multi-agent system (MAS) is only as good as the agents it chooses to trust: its accuracy and cost hinge on its topology, meaning which agents are included and how they communicate. A single task score, however, does not reveal which agents help or hinder performance. We propose Contrastive Attribution Topology Search (CATS), which searches for compact teams within a fixed supergraph of specialized agents. An LLM attributer compares execution traces across candidate topologies and assigns credit to individual agents. CATS uses these credits to update supergraph edges while progressively pruning weak edges and agents, yielding a small team for deployment. With small open-weight models and matched candidate evaluations, CATS improves mean accuracy on BigCodeBench by 5.3 percentage points over score-only heuristic search while using 54\% fewer deployment tokens. On MBPP, CATS matches the strongest baselines with fewer tokens than the other searches. In a HotpotQA prompt-poisoning stress test, CATS excludes all adversarially prompted agents in all three runs, whereas random search retains some at comparable accuracy. Per-agent credit thus supports both team optimization and the pruning of potentially harmful agents. 

---
# Hierarchical Masked Diffusion Language Models 

**Authors**: Parul Gupta, Satwik, Avni Aggarwal, Bhavesh Gurnani, Arnab Kumar Mondal, Dinesh Khandelwal, Dinesh Garg, Parag Singla  

**Link**: [PDF](https://arxiv.org/pdf/2610.29255)  

**Abstract**: Masked diffusion language models have emerged as an alternative to autoregressive models, enabling parallel token generation. However, standard masked diffusion language models use a shared noise schedule across positions without explicitly encoding which positions should be resolved first. Many tasks exhibit a natural priority among positions: document titles and section headings can guide the generation of body text, while Sudoku cells determined directly from the initial clues can help resolve more difficult cells. To incorporate this notion of priority (or hierarchy) among token positions, we introduce the Hierarchical Masked Diffusion Language Model (HMDLM), which augments each token position with a hierarchy level available as part of the training data, and requires zero or minimal additional annotations. It factors generation into two passes: the first predicts a hierarchy distribution over masked positions, the second predicts token distributions conditioned on it. Giving each hierarchy level its own noise schedule induces a soft, controllable preference for revealing higher-priority positions earlier while keeping decoding fully parallel. Our formulation is mathematically grounded, and we re-derive the forward and reverse processes and the NELBO for this factorization. Our evaluation shows that for the task of language generation on LM1B, HMDLM reduces perplexity by more than 1.5 points over Anchored Diffusion Language Model (ADLM), a recently proposed strong diffusion baseline; on WikiHow, we see a drop in perplexity of up to 4.4 pts on ADLM and up to 7 pts against MDLM. An LLM judge on WikiHow prefers HMDLM generations (the two variants combined) more than $2\times$ as often as the baseline. On the task of solving Sudoku puzzles requiring multi-step reasoning, it improves solution accuracy by up to $8.6\%$ absolute points over existing diffusion baselines. 

---
# Intergroup Bias Predicts Expressions of Social Judgment and Empathy in the r/AmITheAsshole Reddit Community 

**Authors**: Yu Hou, Hal Daumé III, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2610.29254)  

**Abstract**: People share and understand others' experiences differently based on their social identities; often, people empathize more when the other person belongs to their ingroup compared to an outgroup. While this intergroup empathy gap is well-studied in social psychology, it remains unclear how it manifests in naturally occurring online social interactions. To this end, we measure how commenters express social judgments, as an indirect outcome of their understanding towards the posters, and empathy in text, based on whether commenters share the poster's gender or are of similar ages. We focus on posts where the poster self-reports these demographics, and analyze 126,716 comments associated with 36,163 posts from 18,111 commenters in r/AmItheAsshole, while controlling for commenter and poster group identities, post features (topic, length, and score), and both commenter- and post-level variance. Our results show that when commenters and posters are ingroup, in either gender or age, commenters are more likely to provide a Not the Asshole verdict and show greater text-based understanding as measured by LLMs. We also observe that older commenters behave differently from younger ones. Furthermore, in a subset of posts where the gender and age of the other party in the conflict are explicitly mentioned, we find that commenter-poster ingroup effects are moderated by the group relationships between the poster and the other party when commenters make social judgments. Our work highlights the importance of intergroup context and social group nuances in understanding online discourse. 

---
# Let Agents Design How They Search: Automatic Interface Optimization for Agentic Search 

**Authors**: Sangjune An, Jaeyoon Jung, Yumin Choi, Jinheon Baek, Junmo Cho, Sung Ju Hwang, Hae Beom Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.29250)  

**Abstract**: Agentic search depends on the interface through which a language-model agent accesses and explores a corpus. Existing interface designs are fixed by hand and applied unchanged to every corpus, although effective designs may depend on the corpus. We introduce SearchWeaver, a framework for automatic search interface optimization guided by downstream search performance. A proposer uses execution trajectories and task rewards to jointly revise corpus artifacts, executable tools, and usage instructions. This pro cess optimizes what information is prepared in advance, which operations tools perform, and which decisions remain with the search agent. With GPT-5.4-nano, the optimized interfaces improve answer accuracy from 71.6% to 78.4% on BrowseComp-Plus and average nDCG@10 from 45.7 to 55.0 across four BRIGHT subsets over Direct Corpus Interaction. These findings highlight the value of jointly optimizing corpus artifacts, executable tools, and usage instructions to develop effective search interfaces. 

---
# MOSAIC: An Omni-Modal Benchmark for Long-Term Memory and Cross-Modal Reasoning in Conversational Agents 

**Authors**: Hong Ting Tsang, Boyi Zhang, Haoyu Huang, Jiaxin Bai, Yufei Li, Yisen Gao, Zhongwei Xie, Hongyu Luo, Jeff Z. Pan, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.29249)  

**Abstract**: Omni-modal assistants must remember not only what users say, but also what they show and convey. Existing long-term memory benchmarks are mostly text-and-image or report a single end-to-end score, so they cannot locate whether evidence is lost when it is represented, retrieved, delivered, or read. We introduce MOSAIC, a benchmark of 923 questions over persona-grounded histories of 11-24 sessions spanning text, image, video, and audio. MOSAIC constructs targeted spatial, temporal, and prosodic signals rather than decorative media, with the strongest causal validation for spatial information and selector-level validation for annotated prosody, and pairs them with a matched diagnostic protocol that freezes the question, reader, prompt, and budget while intervening on representation, retrieval, and delivery. Matched decomposition shows that representation loss is modality-dependent, residual reading dominates even under gold evidence, and abstention remains weak on these controls: strong answerable scores coexist with failures to refuse. MOSAIC thus turns omni-modal memory evaluation from a leaderboard into a diagnosis of write, retrieve, deliver, and read failures. 

---
# Beyond Binary Labels: Diagnostic Evaluation of LLM Safety with Prompt-Level Rubrics 

**Authors**: Hongyu Li, Xinlong Huang, Zhan Yu, Jing Zhao, Zhe Zhao, Juntao Zhong, Xiong Kuang, Rui Yuan, ZhenXiang Yan, Huilin Xu, Haitao Yu, Lilin Wang, Rui Chen, Blaze Chen, Fengzong Lian, Renfen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29225)  

**Abstract**: Current LLM safety benchmarks predominantly rely on binary safe or unsafe labels, lacking explanatory depth and suffering from low consistency across automated judges. To address this, we introduce DiagSafe, a diagnostic evaluation framework that moves beyond binary classification by employing prompt-level rubrics. Grounded in globally applicable AI safety policies, each rubric provides explicit criteria along two dimensions: safety, which targets concrete harms with specific exemptions to suppress false positives, and usefulness, which independently evaluates instruction fulfillment. To ensure a robust and realistic evaluation, we construct a bilingual (English-Chinese) benchmark grounded in a comprehensive 84-label safety taxonomy, consisting of two subsets: DiagBase, which covers single- and multi-turn prompts with direct harmful queries and template-based jailbreaks, and DiagAttack, which embeds harmful intents into complex instruction-following tasks to form challenging instruction attacks. Extensive experiments demonstrate that DiagSafe significantly enhances the reliability of LLM-as-a-judge evaluation, achieving 95.34% agreement with human annotations -- substantially higher than the 69%-87% achieved by the baselines -- and high inter-judge consistency (Cohen's $\kappa$ = 0.864). Furthermore, applying our framework reveals that leading LLMs differ markedly in their unsafe rates, and each model also exhibits unique vulnerabilities along different dimensions. In addition, we observe that models are particularly susceptible in Chinese and multi-turn contexts. Ultimately, DiagSafe provides a reliable and explainable tool for auditing and optimizing LLM alignment. 

---
# C$^2$U in RAG: Compositional Concept Unlearning in Retrieval-Augmented Generation 

**Authors**: Yiyang Duan, Fan Wu, Tiantong Wu, Che Wang, Zhiren Gong, Wei Guo, Yang Cao, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2610.29223)  

**Abstract**: Retrieval-augmented generation can expose unwanted concepts through retrieved evidence even when the generator is moderated. We study query-side concept suppression with a fixed encoder and document index. Compositional Concept Unlearning (C$^2$U) routes a profile-derived direction through an augmented atom bank, adding only a bounded edited-minus-original delta to the raw query. Across five datasets, four encoders, and three seeds, the validation-selected configuration reduces pooled binary target Hit@5 by 4.25 percentage points with non-positive control degradation and target-exposure changes. Under the evaluated query transforms, it suppresses target density without increasing budgeted control-query exposure. Separately, ArXiv four-concept composition improves over an averaged-direction comparator after independent validation gates and first-use test evaluation. 

---
# Understanding Target Probability Reuse in Speculative Decoding 

**Authors**: Jungseob Lee, Giyoon Jang, Chailin Nam, Jiwoo Kum, Sihyeon Lee, Kyojun Choo, Jihun Song, Eunseo Song, Chanjun Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.29210)  

**Abstract**: Speculative decoding uses a target model to verify a tree of draft tokens in parallel. We study whether the resulting next-token distributions can help select the next tree. For unrestricted target models, we show that guaranteed exact reuse requires a distribution computed at the next round's complete context. We characterize when retaining this distribution requires sacrificing accepted tokens and establish conditions under which this loss persists across decoding rounds. Across six targets, we find that reranking first-token branches with a separately computed distribution at the current context improves expected acceptance over draft ranking, especially with small tree budgets. Drawing the first token from this distribution before allocating the tree further improves expected acceptance over pruning a preselected tree, even at equal mean tree size. In our experiments, both selection rules also decode faster than draft ranking when the separate target pass overlaps standalone drafting. Our findings show that making target probabilities available before tree allocation can improve the efficiency of speculative decoding. Our code is available at this https URL. 

---
# norregel: a preregistered benchmark of dated numeric values in Norwegian law 

**Authors**: Eirik Botten Nicolaysen  

**Link**: [PDF](https://arxiv.org/pdf/2610.29207)  

**Abstract**: norregel asks language models for numeric values in Norwegian statutes, regulations and parliamentary decisions (NOK amounts, percentages, multiples of the National Insurance basic amount (G) and age limits), each at a stated date. The 916 items were extracted mechanically from Lovdata's open data, and each carries the dated history of its provision, so an answer can be classed as correct, outdated, wrong or uninterpretable. We report a first and a second preregistered run on the same items. In the first run (five models, listed publicly only after local analysis), H1, that outdated answers are more frequent for values that took effect after a model's cutoff, could not be tested in substance, and H2 and H3 were inconclusive; Opus 5.5 and Sonnet 4.6 answered 46.7% and 40.2% of the 880 primary items correctly. The second run was preregistered and published before any of its calls, and eight models were analysed. Its pooled primary test supported H1 under the preregistered decision rule (d = 17.8 percentage points, 95% interval [7.7, 28.9], p = 0.0001). In the preregistered sensitivity analysis S8, which counts as outdated only the most recent value that differs from the key, d was 3.9 [-3.6, 12.7] (p = 0.1754), an interval that contains 0. H1 was supported in every other preregistered sensitivity analysis, but S8 is the only one that gives both groups one target value, so we do not regard the support for H1 as robust. The share of outdated answers equal to that most recent value (H2) was 56.1% [46.6, 65.0], an interval that contains 0.5, and the pooled test of a lower share correct in Nynorsk than in Bokmål (H3) was inconclusive. Several models, among them Haiku 4.5, gave mostly uninterpretable answers. Items and keys are withheld; item IDs, per-answer classes, code and results are on Zenodo. 

---
# LTEM: LLM-Friendly Table-Enhanced Markdown for Table Understanding and Generation 

**Authors**: Eunbin Kim, Hyeonseok Kang, Sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2610.29188)  

**Abstract**: Tables in real-world documents, including scientific papers, financial reports, and statistical summaries, frequently use merged cells, multi-level headers, and sectioned layouts. Yet the lightweight formats LLMs handle best (Markdown, CSV) cannot express these structures, while expressive formats (HTML, LaTeX) impose excessive token cost and parsing complexity. To enable systematic analysis of this trade-off, we introduce a taxonomy decomposing table structure into six atomic features (F1-F6) and corresponding complexity levels (L1-L6), and construct STB (Synthetic Table Benchmark), a controlled benchmark with 1,089 tables systematically distributed across complexity levels. Using this framework, we propose LTEM (LLM-friendly Table-Enhanced Markdown), a formally specified, minimal extension of Markdown that explicitly encodes spans and hierarchical headers. Experiments across five state-of-the-art LLMs show that LTEM achieves accuracy comparable to HTML, offers a favorable accuracy-token trade-off relative to HTML and LaTeX, and produces substantially fewer invalid responses than LaTeX; for table generation, LTEM is comparable to MultiMarkdown though HTML is best. We release our taxonomy, benchmark, and format specification to support future research on LLM-table interaction. 

---
# TrainBridge: Agentic NPU Training Optimization Guided by GPU Compilation Evidence 

**Authors**: Yupeng Chen, Kunxiang Zhao, Xingjian Wang, Yue Zhao, Yijing Wu, Ye Liu, Kai Li, Congliang Chen, Ruoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.29171)  

**Abstract**: We study how to improve training efficiency on Ascend NPUs. In the NPU training programs we study, framework-level compilation optimizations such as operator fusion are only partially applied, leaving fragmented computation and avoidable execution overheads. Profiling-guided tuning can address these inefficiencies, but it often requires costly trial and error. To make training on NPUs more efficient, we propose TrainBridge, an agentic approach for transferring GPU optimization guidance to NPU training. TrainBridge uses the highly optimized execution produced by mature GPU compilation stacks as a task-specific reference for NPU tuning. It extracts optimization intents from the GPU training execution to guide NPU program rewrites and further refines the rewritten program through iterative profiling. Our GPU-to-NPU migration experiments show that TrainBridge achieves an average 1.7x speedup across four workloads covering LLM pretraining, image classification, and vision-language training, and an average speedup of nearly 2.0x across 27 Modded-NanoGPT speedrun records with highly optimized GPU implementations. 

---
# Carving Radiology Reports at Their Evidential Joints: Anatomical Region Sets as Generation Units 

**Authors**: Jinwei Yang, Junhao Chen, Jianwei Zhao, Zhonghan Xu, Hong Cheng, Ling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29165)  

**Abstract**: Organizing radiology reports into independently generated units offers a way to generate fine-grained clinical content and bind it to visual evidence. However, existing methods typically adopt generation-unit structures not directly specified in natural reports or their annotations, introducing additional modeling compromises. We propose RadCarve, which instead defines generation units directly from the anatomical region sets in existing statement-region correspondences, unifying unit formation with visual evidence selection. RadCarve's planner uses composition-specialized query banks to jointly predict unit existence and region composition from the image, iteratively refining both with regional evidence aggregated according to current membership predictions. The generator independently produces each unit's text conditioned on the joint visual evidence of its predicted region set. Compared with strong baselines, RadCarve improves CXRFEScore by 9.4% in unit generation given region sets and achieves 2.4 times the end-to-end binding F1. It also achieves leading report-generation results on Chest ImaGenome and PadChest-GR, using fewer than one-sixth as many parameters as large-scale baselines. 

---
# Why is AI-generated language so hard to understand? 

**Authors**: Thomas Hikaru Clark, Matias B. Martin, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2610.29162)  

**Abstract**: There is a widespread sense that AI-generated text has recently become harder to understand. In this study, we investigate a set of psycholinguistic features that are known to correlate with human processing difficulty (e.g., object relative clauses, long syntactic dependencies, low-frequency n-grams). Using a novel corpus of AI-generated text across diverse domains, we find that AI models use these difficult features more often than paraphrases generated by round-trip translation. Compared with human writing, all nine frontier and near-frontier models use more inanimate subjects with animate verbs and more low-frequency n-grams, and most also use longer dependencies, more stacked nouns, and more reduced object relatives in conversation; we also demonstrate substantial variation across model families. We propose and test a hypothesis as to why: while human language is optimized for successful communication, AI-generated text is increasingly optimized for other goals like math, code, and reasoning, in which there is a pressure for \textit{information density} rather than successful communication. We provide a proof-of-concept for this hypothesis by fine-tuning a language model under a length penalty, which leads to a divergence from human-like linguistic behavior. Our findings highlight the axes of misalignment between linguistic processing demands in humans and LLMs, and point toward possible remedies. 

---
# GRAFT: Restoring the Missing Sampling Interface in Continuous Diffusion Language Models 

**Authors**: Haoning Yang, Lili Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.29136)  

**Abstract**: Continuous diffusion language models generate a whole sequence in parallel, never drawing a single word at any intermediate step; only the final step turns embeddings into text. Models such as ELF now report autoregressive-level quality at a fraction of the sampling steps of discrete-token diffusion models, and this design principle -- no step-wise discretization -- is what their own ablations credit for the gain. We show that this same principle carries a hidden cost: a collapse of the lexical long tail. Measured against held-out human text, ELF's output matches the human rate on common words almost exactly, yet its rarest words appear at only 0.30x the human rate, a gap that does not close from 105M to 652M parameters and that no sampler setting these models expose can reach. A same-size GPT-2, in contrast, spans zero to above the human rate by its decoding rule alone. We trace the difference to what continuous diffusion models lack: with no step that ever draws a discrete word, every step is instead driven by an average over the candidates still in play, and averaging a rare candidate together with the common word competing for its slot lands on the common word. An autoregressive model avoids this because its decoding rule samples one candidate rather than averaging over all of them. We propose GRAFT, a training-free method that grafts this missing choice back onto a frozen model: while the model is still undecided, GRAFT commits the rare words the average would otherwise discard, then regenerates the sequence from scratch with those commitments held in place. Across three ELF scales and on LangFlow, an independent continuous dLLM, GRAFT restores the rare-word share of generated text from 0.42-0.77x the human rate to 0.82-1.10x, a relative gain of 34-109%, and moves the entire word-frequency profile closer to human on every model tested, by 2.7x on ELF-B and 12x on LangFlow. 

---
# Multi-level Encoder with Global Topic for Task-oriented Dialogue Summarization 

**Authors**: Zhuoqi He, Peijie Huang, Yuhong Xu, Youming Peng, Mingzhi Xu, Xinyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.29097)  

**Abstract**: Task-oriented dialogue summarization aims to automatically extract key information to generate domain summaries to improve service efficiency and quality. Task-oriented dialogue is inherently logical and surrounds specific topic. How to effectively capture the dialogue topic and the most salient information becomes one of the major challenges of this task. In this paper, we propose a task-oriented dialogue summarization model utilizing a multi-level encoder with global topic (MLEGT). This model discovers temporal and spatial dependencies through a multi-level encoder, while simultaneously obtaining a simple yet effective global topic in the process of graph construction. It not only captures the crucial details but also breaks the long-range limit and discover the intrinsic structure across the dialogue. Then, a global topic guided pointer mechanism is added into the summrizer to capture ignored but important information, which is helpful to improve the accuracy of the summary. A comprehensive study of two public datasets, including a real-world large-scale Chinese police interrogation dataset and a medical reporting service dataset proves the superiority of our method on several strong baselines. 

---
# A Large-Scale Evaluation of Role-Enhanced Zero-Shot Chain-of-Thought Prompting 

**Authors**: Ibrahim Khan, Thai Van Nguyen, Yi Xia, Pratch Suntichaikul, Mustafa Can Gursesli, Daniel Fernandez Galeote, Juho Hamari, Ruck Thawonmas  

**Link**: [PDF](https://arxiv.org/pdf/2610.29073)  

**Abstract**: This paper proposes Role-Enhanced Zero-Shot Chain-of-Thought (RE-ZS-CoT), a prompting framework that systematically integrates domain-specific expert roles to enhance reasoning in large language models (LLMs) without task-specific examples. We assess two versions of our approach. In the first version, static role selection, we evaluate eight predefined expert roles across arithmetic, commonsense, and reading comprehension benchmarks to determine their impact on reasoning performance. In the second version, dynamic role selection, each model autonomously selects the single most suitable role for each benchmark, enabling adaptive task-role alignment without manual specification. Across 20 LLMs and seven benchmarks, RE-ZS-CoT frequently surpasses zero-shot and zero-shot-CoT baselines, achieving performance improvements of up to 44 percentage points. The results reveal a strong correspondence between static and dynamic versions; the benchmarks where static roles yielded improvements were almost always those where dynamic roles also outperformed baselines, though static selection achieved peak performance. Ablation studies further show that unconventional roles degrade performance, underscoring the necessity of task-role alignment. The framework's scalability is validated across model sizes (2b-14b parameters), with smaller models like Granite3.3-8b achieving remarkable proficiency in arithmetic reasoning. RE-ZS-CoT advances zero-shot prompting by unifying role specialization with chain-of-thought reasoning, offering a practical, resource-efficient solution for improving LLM capabilities across diverse domains. 

---
# Our Own Jev: Typed Decisions in One Forward Pass, from an In-House Judge to Open LoRA Models 

**Authors**: Yuchuan Weng, Chi Liu, Xin Chen, Srinivasan Manoharan  

**Link**: [PDF](https://arxiv.org/pdf/2610.29052)  

**Abstract**: A typed decision maps evidence, a question and a fixed set of options to a probability for every option, computed in one forward pass, by reading a language model's logits or a small head over its hidden states, instead of generating and parsing text. The commercial system Jev popularised this System~1 interface, and open reproductions have followed. We report two uses of the pattern. \emph{Part I} describes an in-house System~1 judge that triages about 148K agentic coding session records for a training corpus. It replaces a frontier LLM judge that cost on the order of \$1,000 per pass and whose stored labels matched a fresh two-judge consensus on only 26.9\% of the sessions where the two judges agreed; a self-hosted frozen backbone with a rank-16 LoRA adapter and a pointer head answers three typed questions in one forward pass, judges 1,000 sessions in about three minutes of one GPU, about \$0.06 at a reference rate and roughly 1\% of the frontier judge's cost, and keeps every session inside the organisation. \emph{Part II} asks whether a readout that adds no parameters can be competitive and whether training one helps. Applying the public Eikos recipe and corpus of about 25K decisions to three further backbones (1B, 2B and 35B), we read the frozen output layer of a LoRA-adapted decoder. On the 111 public hard items of JevBench our single-seed 35B, developed with those items in view, scores 79.3\%, not significantly different from the commercial reference (73.0\%, unadjusted $p=0.12$) and ahead of five open systems after Holm--Bonferroni correction; our 2B is ahead of all four kev checkpoints on the leaderboard. A single-seed ablation finds no detectable benefit from training the 26 letter rows of the output layer, and a retrain on a later corpus release moves the 2B's hard-tier accuracy by 11.7 points, which shows how little small evaluations resolve. 

---
# Target Regeneration Shrinks the Gains of Draft-Training Objectives in Speculative Decoding 

**Authors**: Jungseob Lee, Hogun Seong, Euijun Choi, Sugyeong Eo  

**Link**: [PDF](https://arxiv.org/pdf/2610.29013)  

**Abstract**: Speculative decoding accelerates large language models by verifying drafted tokens in parallel. New draft-training objectives are usually credited with the acceptance they add over cross-entropy on externally written responses, yet many pipelines let the target rewrite them. We ask how much of that gain survives. With a vision-language target and a block drafter, we train three objectives on four versions of one corpus: cross-entropy on corpus tokens, token-level distillation from the target, and distillation gated to positions that would survive verification. Distillation and the gate add large gains on external responses, but after greedy regeneration, distillation adds a smaller gain and the gate little or none. The shrinkage persists across data orders, position weights, verifiers, decoding temperatures and, for distillation, initializations and learning rates. Most of it remains at matched supervised length and at each objective's best measured training length. A second vision corpus shows the shrinkage, as does a text-only replication, for distillation also with an autoregressive drafter. Objective gains should therefore be measured against a target-regenerated baseline and across training lengths. Our code is available at this https URL. 

---
# Language Impairment as a Lens onto Large Language Models 

**Authors**: Muxuan He, Wang Bill Zhu, Yue Chen, Elsi Kaiser, Weizhi Du, Yao Du, Samuel Nastase, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.28992)  

**Abstract**: Although large language models (LLMs) encode language in highly distributed spaces, they remain vulnerable to focal disruptions---a phenomenon strikingly similar to how localized brain damage causes clinical aphasia in humans. To investigate this theoretical tension, we ask whether LLMs possess a functional structure analogous to the human language system, by introducing a novel mechanistic interpretability framework inspired by aphasia. Through two complementary approaches, classifier-guided lesioning and continuous activation steering, we systematically induce aphasia-like breakdowns in LLMs. We find that language functions can be successfully isolated to discrete structural circuits and continuous representational vectors, yet these functions remain densely entangled with significant circuit overlap and vector similarities across distinct aphasic profiles. Furthermore, while steering generalizes robustly across downstream tasks, structural lesioning remains asymmetrically constrained. By bridging neuropsychology and mechanistic interpretability, we demonstrate that LLMs, functionally analogous to biological brains, rely on a densely entangled architecture that can be systematically mapped through clinically-informed interventions. 

---
# Scan or Index? large Evidence Coverage, Answer Quality, and Cost in Long-Document RAG 

**Authors**: Juan Terven, Diana-Margarita Córdova-Esparza, Julio-Alejandro Romero-González, Edgar Chavez-Urbiola, Juan Irving Vasquez, Julian Andres Ramirez-Bautista, Silvia L. Chaparro-Cárdenas, Gendry Alfonso-Francia  

**Link**: [PDF](https://arxiv.org/pdf/2610.28957)  

**Abstract**: Scanning every passage in a long document seems thorough, but does it preserve more useful evidence or produce better answers? We compare 11 retrieval configurations spanning lexical and dense search, hybrid reranking, and full-document scanning with Jev, OpenAI Decisions, and Cloudflare Clef. Our evaluation uses 396 QASPER questions and 80 questions from eight complete textbooks, with common passages, a five-passage limit, and a fixed answer generator. On books, Hybrid + Jev achieves the highest observed evidence coverage (94.1%), compared with 88.7% for Jev full scan and 81.3\% for OpenAI dense retrieval. Full scans examine every passage initially, yet later reduction steps can discard relevant evidence. Higher evidence coverage also does not consistently produce better answers: Hybrid + Decisions achieves the highest observed book Answer Accuracy among retrieval methods (0.892), despite recovering less reference evidence. Full scanning incurs substantially higher latency on books under the tested deployments, while reusable dense retrieval has lower projected API cost than paid decision-based methods as queries accumulate. Most QASPER articles already fit within the hybrid candidate pool, limiting the benefit of scanning. These results highlight evidence retention, answer quality, and reuse cost as complementary criteria for choosing a long-document retrieval strategy. 

---
# Guilt by Association: Semantic Leakage Drives Associative Gender Bias 

**Authors**: Hazel Chen, Peter West, Hila Gonen  

**Link**: [PDF](https://arxiv.org/pdf/2610.28944)  

**Abstract**: Semantic leakage occurs when a concept in the prompt steers model generation about an entity, even though the concept is irrelevant to it (e.g., a person who likes yellow is said to work as a school bus driver). It has mostly been studied in settings where the concept is directly attributed to the entity the model is asked about. This leaves open whether leakage is a flawed heuristic, where the model builds a profile of that entity, or a much broader associative tendency, e.g., working across seemingly unrelated entities. We argue for the latter: even after removing the attribution of the concept to the specific entity, leakage persists. Next, we extend this idea to gender bias. While work on gender bias in LMs typically looks at cases where the stereotyped concept is attributed to an entity, our findings suggest that a concept may steer outputs with its mere presence in the scene. We show this using gendered concepts under two conditions: when the subject engages with the concept, and when the concept just sits in the scene. We find that both shift model generation toward the stereotype to a similar extent. Together, our results suggest that semantic leakage drives a form of bias that is overlooked by current evaluations. 

---
# Is Refusal Direction Universal across Low-Resourced Languages? 

**Authors**: Anastasiia Ivanova  

**Link**: [PDF](https://arxiv.org/pdf/2610.28897)  

**Abstract**: Refusal directions have been shown to transfer cross-lingually with near-perfect effectiveness across safety-aligned languages (Wang et al., 2025), reportedly regardless of resource level. However, the only truly low-resourced language evaluated is Yoruba, which is strongly misaligned and shows limited transfer, leaving open how refusal directions behave for other low-resourced languages. We extend PolyRefuse to three additional low-resourced languages (Belarusian, Bashkir, and Tajik) and replicate the pipeline of (Wang et al., 2025) on Qwen2.5-14B-Instruct. We find that universality is graded rather than binary: directions from safety-aligned and partially-aligned languages transfer broadly, while those from strongly misaligned languages fail to reliably modulate refusal behavior on any target tested, including other misaligned languages, once ceiling effects are accounted for. Geometric alignment and behavioral transfer are further shown to be partially dissociated. These findings complicate the universality claim and suggest that users of the least-resourced, most strongly misaligned languages may face weaker safety guarantees. 

---
# CSD: Common-Support Distillation for Cross-Tokenizer Knowledge Transfer 

**Authors**: Mingyu Yang, Sharath Turuvekere Sreenivas, Adithyakrishna Venkatesh Hanasoge, Nima Tajbakhsh, Ashwath Aithal, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2610.28895)  

**Abstract**: Cross-tokenizer logits distillation enables knowledge transfer across model families, allowing students to learn from stronger or more specialized teachers. However, differences in vocabulary and segmentation make teacher and student next-token distributions difficult to align. Existing logits-based methods are easy to integrate into standard distillation pipelines, but often rely on restrictive heuristics or tokenizer-pair-specific choices. They may restrict supervision to shared vocabulary subsets, require manually selecting different objectives for different tokenizer pairs, or match only the observed chunk and its complement rather than individual alternative continuations. We introduce Common-Support Distillation (CSD), a unified logits-based framework that constructs a shared support between the teacher and student at each aligned input chunk and matches the induced distributions with a single objective. For one-to-one alignments, CSD uses explicit common-token pairs as the shared support. For other alignments, it constructs a virtual support whose probabilities are computed using the autoregressive chain rule. A residual event captures probability mass outside the explicit support, ensuring that both induced distributions remain normalized. Across four teacher-student settings comprising Llama-3.2-1B and Gemma3-1B-PT students with Qwen and Phi teachers, CSD consistently achieves the highest average performance among the evaluated cross-tokenizer methods, outperforming the strongest alternative by up to 1.8 points and surpassing both continued pretraining and same-tokenizer distillation. For Llama-3.1-8B, CSD improves average accuracy over the strongest evaluated baseline by up to 4.2 points for the base student and 1.6 points for the instruction-tuned student. 

---
# Building AI Research Teams for Autonomous Discovery of Linear Attention Architectures 

**Authors**: Yujian Liu, Li An, Zelin He, Jacob Helwig, Jiabao Ji, Rohit Jain, Siyu Zhu, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28887)  

**Abstract**: Autonomous agents can increasingly propose ideas, modify code, and run experiments, creating an opportunity to accelerate scientific discovery using teams of collaborative AI researchers. To escape local minima in the discovery, we propose to encourage orthogonal exploration by constructing the team using agents with varying expertise. However, realizing this design requires more than diverse backgrounds. When experiments are costly, a team can test only a limited number of ideas, thereby creating the need to balance exploration and exploitation when selecting a subset of researchers to propose and execute new experiments. Furthermore, each result should shape what the team tries next. We therefore ask how AI researchers with different scientific backgrounds can build on one another's findings and sustain progress under a limited experimental budget. To address these challenges, we build a team of persistent AI researchers, each grounded in a different discipline, who keep private research histories, share experimental evidence, and receive research opportunities through adaptive allocation. We demonstrate the effectiveness of this design by deploying the team to search for linear attention architectures, where it outperforms GDN2 at 1.3B scale on language modeling, common-sense reasoning, and synthetic and real-world retrieval tasks. In controlled comparisons, removing any single organizational component from our design yields markedly weaker discoveries: without disciplinary backgrounds, the search plateaus early; without private histories, researchers make less progress despite identical shared evidence; and uniform selection falls short of adaptive allocation within the same budget. These results suggest that sustained autonomous discovery depends not only on stronger agents, but also on who explores, what they remember, and how research effort is allocated. 

---
# SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges 

**Authors**: Ethan Ludwick, Yu-Chung Hsiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.28886)  

**Abstract**: LLM judges increasingly mediate benchmark evaluation, model selection, data curation, and training. Their decisions are useful only when each judge responds reliably to changes affecting its scoring criterion. Existing methods for reliability testing perturb inputs with known expected effects on judge scores. Prominent LLM-judge testing harnesses combine human-authored perturbation recipes with synthetic test generation, leaving scorer-specific recipe ideation outside the automated pipeline. This limits their use to fixed perturbations and pre-assumed scorer types. We introduce SANE, an agent-based framework that encodes behavioral requirements for stochastic evaluators as reusable agent instructions. Given a scorer specification, implementation, and representative inputs, the pipeline first identifies scorer-specific capabilities, then automatically derives perturbation recipes and expected score relations for each. SANE also defines a scorer-independent validation framework for generated judge tests. It determines whether each recipe agrees with scorer intent and whether each instantiated case realizes the recipe and supports its expected relation. Because SANE generates a recipe inventory tailored to each scorer, the number of aligned recipes grows as scorers are added. Across five domains, SANE produces 3.7 times as many validated test families as the Judge Reliability Harness (JRH) at comparable end-to-end validity and detects scorer deficiencies in every domain, including professionalism and toxicity, where JRH detects none. On faithfulness, SANE achieves strong human alignment in judge ranking: its test-suite scores rank 11 judge models consistently with human-labeled FaithBench performance (Spearman rho = 0.852), while JRH does not (rho = 0.100). 

---
# KIRA: A Full-Stack Reading Screening System with Dual-Pathway Analysis for Children 

**Authors**: Maggie Dong, Rohit Mekkoth, Zixuan Wang, Cheol Jun Cho, Jiachen Lian, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2610.28840)  

**Abstract**: Early-grade reading screening is critical for identifying children at risk of reading difficulty. Existing AI-based screeners are often limited in the scope of their analyses and evaluation metrics, or are proprietary systems that lack transparent model specifications and evaluation. We present KIRA, a dual-pathway screening system that pairs word-level fluency scoring with detailed phoneme-level error detection. We adapt pretrained word and phoneme recognizers to children's speech, reducing word error rate from 13.88% to 8.84% and phoneme error rate from 41.66% to 23.00%. Powered by these child-adapted models, we implement Word Reading (WRE) and Oral Reading Fluency (ORF), two foundational tasks in K-2 screening. The system integrates these recognizers with a web-based teacher-facing interface presenting fluency scores alongside phoneme feedback for potential targeted intervention. KIRA brings modern child-speech recognition into a full-stack, deployable, classroom-ready reading screener. 

---
# How to Use Parallel Data for Multilingual Pretraining 

**Authors**: Zhihong Huang, Yao Lu, Derek F. Wong, Hamid Alinejad-Rokny, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28838)  

**Abstract**: High-quality human-translated parallel corpora can improve cross-lingual transfer, but are typically too small for large-scale pretraining. Machine-translated web corpora help overcome this limitation, making parallel data available at pretraining scale. Standard multilingual pretraining, however, samples source documents and their translations independently, without explicitly using their parallel structure. We investigate how to use the parallel structure of the data to improve multilingual pretraining by concatenating each document with its translation in the same training sequence. We compare this parallel pretraining (PPT) approach with standard pretraining (SPT), training models from scratch at matched token budgets. Surprisingly, despite the benefits of parallel data reported in prior cross-lingual work, PPT alone does not consistently improve language understanding over SPT. Our further analysis suggests that although PPT builds a strong cross-lingual foundation, as shown by its better use of shared information across languages, the mismatch between parallel training contexts and single-language task inputs limits its gains in general language understanding. This finding motivates a two-stage approach following the pretraining-fine-tuning paradigm: we first use PPT to build a cross-lingual foundation, then continue with SPT to adapt it to standard language modeling. In both high-resource and low-resource settings, the two-stage approach consistently improves general language understanding and cross-lingual task performance over standard pretraining alone at matched training budgets. 

---
# A Parallel Multi-Dialect Arabic Dependency Treebank 

**Authors**: Noor Abo Mokh, Mohamed Adel, Jamila El-Gizuli, Go Inoue, Ahmed Elshabrawy, Nizar Habash  

**Link**: [PDF](https://arxiv.org/pdf/2610.28823)  

**Abstract**: Dependency parsing resource development for Arabic has largely focused on Modern Standard Arabic, leaving dialectal Arabic comparatively under-resourced. We introduce MADARTB, the first large-scale parallel dependency treebank spanning five city dialects: Beirut, Cairo, Doha, Rabat and Tunis, and extending an existing parallel MSA version. In total MADARTB consists of 5,000 sentences annotated using the CATiB dependency scheme. We analyze syntactic similarity across all city dialects and MSA. Although global construction profiles exhibit high similarity across varieties, local syntactic structures frequently diverge, revealing substantial structural variation across dialects. We further establish a cross-dialect dependency parsing comparison using a supervised parser and four LLMs. The supervised parser achieves 87.5% LAS, while the best LLM setup trails at 85.7% LAS across all six varieties on the test set. We release our annotations and models publicly. 

---
# LIFT: Scoped Constraints for Evaluating and Training Instruction Following 

**Authors**: Eran Goldstein, Yuval Belfer, Yonatan Belinkov, Daniel Gissin  

**Link**: [PDF](https://arxiv.org/pdf/2610.28819)  

**Abstract**: Common instruction following (IF) benchmarks, such as IFEval, evaluate the ability of language models (LMs) to follow simple constraints that can easily be verified programmatically. These same constraints, in turn, are used as reward signals for reinforcement learning. This design may lead to reward hacking of constraints, pushing models away from the main task. Recent attempts to address this resulted in benchmarks with richer constraints, but required expert labor, remained closed-access, and cannot be easily extended. We present Lift-Bench, an IF benchmark that asks a model to draft a document given a topic and a structured outline. Constraints are scoped to certain document sections, providing finer control over constraint density. Lift-Bench includes both verifiable and judged constraints, as well as a training counterpart, Lift-Train. Importantly, Lift-Bench provides both main task and instruction-following constraints, mitigating shallow reward hacking. On experiments with multiple models, we find that frontier models struggle with the harder subsets of our benchmark and analyze model performance along multiple axes. We also find that training on Lift-Train is much less susceptible to reward hacking than training on the common IFTrain. Lift-Bench and Lift-Train are publicly available, as well as the data generating pipeline. 

---
# Where Do Legal Factors Live in LLMs? 

**Authors**: Li Zhang, Kevin Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2610.28799)  

**Abstract**: In case-based legal reasoning, legal professionals identify legally significant fact patterns in a new dispute and compare them with those of decided cases. In AI & Law, HYPO dimensions and CATO factors represent these patterns symbolically, organizing expert knowledge distilled from case narratives as explicit, structured, interpretable, and auditable legal concepts and relations that support case comparison, legal argument, and precedent-based reasoning. Large language models (LLMs) are increasingly used for legal tasks, yet their internal organization of legal knowledge remains difficult to trace. We investigate whether LLMs form internal representations corresponding to all 26 CATO factors. Two research questions examine linear decodability and sparse localization with sparse autoencoders. Experiments cover three open models, Qwen3-1.7B, Qwen3-8B, and Gemma-2 2B, supported by cross-model comparison and human audit of controlled natural-language passages. Results show that all 26 factors are linearly decodable in all three models. Single-feature sparse readouts cover the complete factor catalog, with best-layer test AUCs ranging from 0.86 to 1.00. Five selected features approach full-vector readout performance. We report 78 model-specific layer-feature coordinates, one for each factor in each model. Together, these results show that information corresponding to expert-defined legal factors can be read from the internal activations of LLMs processing legal narratives. By locating these readouts, our study connects legal language and the symbolic abstractions of HYPO and CATO to learned internal representations. 

---
# Is Anyone Enforcing the Cartel? What Reinforcement Learning Teaches LLM Agents in Markets 

**Authors**: Hieu Tran, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28787)  

**Abstract**: Evaluations of large language model (LLM) agents report that pricing agents collude at others' expense. Yet collusion is defined by enforcement: a cartel, in which competitors share monopoly profits, holds only if a member who cheats is punished. We introduce an audit that tests enforcement by forcing a member to cheat in counterfactual copies of a market, and use it to study what reinforcement learning (RL) teaches agents in pricing and trading. Small LLMs trained by RL learn what their training rivals reward, but only what they readily try. In self-play, they learn fixed rules ignoring their rivals; against a punishing rival, to be deterred; against a cheating rival, to match its cuts, never the costly price war that would deter it, which they rarely try. A war forced into training is learned, but mostly only where it was forced. Against their own copies, deterred and matching agents can earn the same cartel profits, but only matching ones can enforce them, and then thinly: as a by-product and only against large cuts. Q-learners enforce collusion only in pricing. Prompted LLMs reach cartel-like outcomes mostly through rules that ignore the market. Outcome measures cannot tell these cases apart; the audit can. 

---
# Scaling Therapy Supervision: Assessing Psychotherapy Fidelity Using Large Language Models 

**Authors**: Arvind Pillai, Alexis Roach, Jason Holden, Varsha D Badal, Subigya Nepal, Yu Yvonne Wu, Yuliang Chen, Sudarshan Regmi, Sarah Masud Preum, Nicholas C Jacobson, Nikhil Singh, Eric Granholm, Colin Depp, Andrew Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2610.28772)  

**Abstract**: Psychotherapy fidelity assessment measures how faithfully a therapist adheres to a treatment protocol, a critical determinant of positive patient outcomes. However, fidelity assessment is bottlenecked by manual expert review, rendering the process labor-intensive and unscalable. To scale therapy supervision and support clinician self-learning, we present an integrated application of large language models (LLMs) that automates psychotherapy fidelity workflows and generates actionable supervisory feedback from authentic, long-form clinical transcripts. We evaluate three open-weight models across multiple task decomposition configurations and reasoning strategies, demonstrating that evidence-first reasoning improves fidelity estimation. Furthermore, we establish the framework's external construct validity on an independent randomized controlled trial (RCT), where it reliably distinguishes Cognitive Behavioral Therapy (CBT) from non-CBT sessions across core therapeutic skills. Through a user study that evaluates the generated feedback, mental health experts indicate that the pipeline synthesizes actionable clinical feedback. Ultimately, our work offers a practical, scalable solution toward bridging the community mental health supervision gap. 

---
# SkillEval: Learning Interpretable Ability Profiles of LLMs via Cognitive Diagnosis Models 

**Authors**: Berke Arda, Peng Cui, Qiaoyuan Zheng, Rudolf Debelak, Mubashara Akhtar, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2610.28767)  

**Abstract**: Current evaluation of LLMs on benchmarks typically aggregates performance over individual examples into a single scalar score. This approach is overly coarse and ignores the diverse scope of skills covered by benchmark datasets. To address this limitation and better exploit benchmark data, we propose SkillEval, a new evaluation framework that performs cross-model, skill-level analysis. The framework integrates advances from two areas: (1) it leverages the meta-cognitive capabilities of LLMs to automatically discover interpretable skills from questions, and (2) it applies cognitive diagnosis models (CDMs) from psychometrics to estimate per-skill abilities of LLMs. We conduct experiments on 3,811 LLMs evaluated on 9,523 items across five benchmarks spanning diverse domains (MATH, BBH, GPQA, MuSR, IFEval). Compared to the traditional average-based evaluation paradigm, SkillEval produces reliable, more calibrated and informative yet concise model profiles that more accurately predict model behavior on unseen items. We also demonstrate the utility of SkillEval in two downstream tasks: (1) LLM routing, which dynamically assigns each query to a capable but efficient model, reducing inference cost by over 60% while achieving improved performance, and (2) Computerized Adaptive Testing (CAT), where an unseen model can be efficiently evaluated using a trained CDM. Furthermore, we show that these skill profiles provide useful insights, for example, how benchmarks share skills and how instruction-tuned models differ from their base models. 

---
# Translation and data ethics 

**Authors**: Joss Moorkens, Monja Jannet  

**Link**: [PDF](https://arxiv.org/pdf/2610.28710)  

**Abstract**: This chapter introduces data ethics for translation and discusses a number of ethical issues regarding data. These include issues of copyright, ownership and bias, which apply to all types of data, whereas there are further issues particular to minoritised language communities and those on either side of the digital divide, as data is implicated in power imbalances and dynamics. We close with a discussion of responsibility and regulation, and reflect on the growing attention on data ethics from translation studies researchers. 

---
# How Far Can the Gemma 4 Speech Encoder Go Without Changing Its Weights? Saudi and Gulf Arabic Speech Recognition with a Frozen and an Adapter-Augmented Encoder 

**Authors**: Amr Abdelkhalik  

**Link**: [PDF](https://arxiv.org/pdf/2610.28673)  

**Abstract**: We study the 305M-parameter audio encoder shared by Gemma 4 and EmbeddingGemma 2 (identical except for its final projection) for Saudi and Gulf Arabic speech recognition, without changing any of its weights. A frozen system trains only a 22.6M-parameter CTC reader over a mix of encoder layers; an adapted system adds 3.1M low-rank adapter parameters, a 41.5M-parameter hybrid CTC/attention reader and joint decoding with a sub-word language model. Against Cohere Transcribe Arabic (2.07B parameters), re-run on identical audio, the frozen system trails in word error rate (WER) but is statistically indistinguishable in character error rate (CER) on five of six human-labelled conditions, while running 3.4 times faster on CPU with greedy decoding. The adapted system is statistically indistinguishable in WER on a public Saudi customer-service call (19.20% against 19.27%) and on clean Emirati speech, is significantly better on telephone-coded Emirati speech, has a significantly lower CER on all six conditions of these sets, and beats a 0.6B recogniser fine-tuned on SADA in WER on all six; it still trails Cohere in WER on long-form Saudi speech. On a blind, human-corrected public call-centre sample it ranks 4 of 15 systems, behind three commercial APIs (significantly only one) and ahead of the other self-hostable systems by point estimate. An optimised joint decoder runs it at 139 times real time on one GPU. We also show that references made by correcting machine drafts favour the drafting system. Code and public-set results are released. 

---
# Think Harder Where It Matters: Item-Level Adaptive Recursion for Sequential Recommendation 

**Authors**: Zhanshuo Cao, Jiakai Tang, Yuqi Zhou, Sunhao Dai, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30128)  

**Abstract**: Existing sequential recommenders usually allocate a fixed computational budget to every item in a user's interaction history, regardless of the item's contribution to the final prediction. We challenge this de facto design with two diagnostic findings. (1) Non-uniform computational need: items within the same sequence vary drastically in how much computation they require. Most converge after shallow processing while a critical minority only stabilizes at full depth. (2) Suboptimality of fixed allocation: given oracle access to the target item, a non-uniform depth assignment consistently outranks the fixed one, indicating that existing uniform computation both wastes capacity on already-converged items and sacrifices overall recommendation quality. These observations motivate a shift from fixed-depth computation to item-level adaptive reasoning. To this end, we propose LoopRec. LoopRec recursively applies a single weight-shared block and introduces a lightweight gate at each recursion boundary that makes a binary exit-or-continue decision for every item independently. Items whose representations have sufficiently matured exit early, while those requiring deeper reasoning continue to subsequent recursions. This design converts the fixed global depth into a learned, per-item property without auxiliary depth labels, and is readily adaptable to diverse backbone architectures. Extensive experiments on public benchmarks with four backbone families demonstrate that LoopRec consistently outperforms matched fixed-depth counterparts. Further analysis reveals that LoopRec's learned allocation exhibits interpretable patterns aligned with our post-hoc oracle analysis: lower-frequency items and more recent interactions naturally receive deeper computation. Our code is available at this https URL. 

---
# VX-One: Typed, Calibrated Decision Heads for Real-Time Perimeter Triage - An empirical study of unknown-attack honesty, cascade gray zones, and calibration under drift 

**Authors**: Juan S. Santillana  

**Link**: [PDF](https://arxiv.org/pdf/2610.30090)  

**Abstract**: Perimeter triage faces a latency/semantics trade-off: signature engines are fast and format-safe but blind to unseen attacks, while generative models are semantic but slow and produce unparseable output. In late 2026 a class of commercial System One decision models (TypeSafe Jev; Microsoft-Decision-1) began selling typed decision primitives - binary gates, closed-set choices, ordinal scores - as product APIs, with no academic treatment. We present VX-One: a small multi-head network (100K-10M parameters) whose outputs are these primitives - a calibrated gate, a choice over a closed candidate set that always includes an explicit unknown option, and an ordinal score, trained multi-task, temperature-calibrated per head (~0.7-1.7 ms p50 on CPU). We evaluate it on UNSW-NB15, CIC-IDS2017, EMBER2024 against calibrated XGBoost, Random-Forest, MLP, Kitsune, and measure a two-phase cascade (Suricata with 52,674 Emerging-Threats rules -> VX-One) over five days of CIC-IDS2017. Findings: (i) VX-One wins detection (AUROC/FPR@95%TPR) on UNSW-NB15 and under temporal drift on CIC-IDS2017 (0.711 vs 0.637), but XGBoost wins multi-class attribution on all three benchmarks; (ii) leave-one-family-out gate detects never-seen families on 9/9 UNSW folds yet drops below chance on CIC recon family, and abstention threshold transfers poorly across benchmarks; (iii) on payload-less reconstructions signature engines resolve ~0.1% of flows - gray zone ~99.9% of traffic, recovered by Phase 2 (combined recall >=0.978); (iv) calibration collapses under temporal drift (ECE 0.04->0.40) for every method while ranking partially survives. We provide a measured account of where typed calibrated decisions help perimeter triage, and where they fail. 

---
# Agent Behavior Maps: Locating Agents by What They Would Do, Without Running Them 

**Authors**: Shota Fujikawa, Takuya Tamura, Kosuke Akimoto, Daichi Haraguchi, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2610.30014)  

**Abstract**: LLM agents with similar success rates can follow different action sequences. An Agent Behavior Map represents LLM-harness combinations using centered action log-likelihoods on shared reference trajectories, without requiring new environment rollouts from the target agent. We derive error bounds relating squared coordinate distances to trajectory KL under explicit assumptions, and across eight cross-scored models the distances track the directly measured trajectory KL in rank order. Across 100 LLMs and four environments, ALFWorld-based coordinates support performance prediction and identify behavioral similarities. In ALFWorld, these similarities persist after controlling for success rates and model size, including across model families. As an extension, harness-component displacements approximately compose, with accuracy varying across components and environments. 

---
# HiddenLens: A Multi-Layer LLM Framework for Detecting and Resolving Hidden Assumptions in Software Requirements 

**Authors**: Tanu Singh, Aashna Suman, Vinod Patidar  

**Link**: [PDF](https://arxiv.org/pdf/2610.29996)  

**Abstract**: Natural-language software requirements often omit assumptions about performance, operating conditions, user behaviour, and domain constraints, affecting architecture, testing, and deployment. If undiscovered until implementation, resolving them is far more costly than addressing them early. Although ambiguity and vagueness are widely studied, less attention has been given to assumptions embedded in requirements that appear clear and reasonable. We propose HiddenLens, a novel six-layer BERT-LLM pipeline and, to our knowledge, the first framework engineered to detect, classify, question, and resolve hidden assumptions rather than merely flag ambiguity. Layer 1 uses BERT attention entropy to localise unresolved spans; Layer 2 uses CoT-prompted LLM reasoning to articulate each assumption and its failure risk; Layer 3 uses fine-tuned BERT to classify assumptions into six non-functional requirement categories: performance, security, scalability, usability, reliability, and domain; Layer 4 generates ranked clarification options using FS LLM prompting; Layer 5 enables human validation through confirmation, editing, or rephrasing; and Layer 6 rewrites the requirement using the confirmed resolution, producing an explicit, testable, and traceable requirement. We also introduce the HiddenLens Corpus (HLC), comprising 312 requirements annotated with 847 hidden assumptions across six NFR categories. Inspired by PROMISE and PURE, HLC addresses the lack of annotated data for this task and is, to our knowledge, the first dataset developed specifically for it. HiddenLens outperforms keyword-based and QuARS-Adapted baselines by 71% and 61%, respectively, achieving an end-to-end F1-score of 0.79; its BERT-based NFR classifier achieves an F1-score of 0.847. These results indicate that HiddenLens can help analysts resolve hidden assumptions before they propagate through software development. 

---
# Commit or Abstain? System One Decision Models, LLMs, and Missing Legal Authority 

**Authors**: M. Mikail Demir  

**Link**: [PDF](https://arxiv.org/pdf/2610.29961)  

**Abstract**: Given the enormous number of claims to be checked, verifiable legal AI requires scalable, cost-effective models that answer the practical needs of law. System One decision models are built on large language models but, instead of generating text, return a calibrated probability for each option. This suits the verification of legal authority, where the question is closed: a given holding, or none of them. We test whether such models abstain when the correct authority is missing. We extend CaseHOLD with an abstention condition in which the correct holding is replaced in 1,670 of 5,314 items, and compare four decision models with six generative LLMs. When no option is correct, generative LLMs still pick one in 38-86% of items; the most cautious decision model does so in 30%. On standard CaseHOLD, a decision model matches the best generative LLM (88.1% vs. 88.2%) at 1/55 of the cost. Accepting that LLM's answer only when a decision model agrees halves its wrong citations, and distilling a decision model's probabilities into Qwen3-8B, without labels, yields a well-calibrated local model. Legal AI should be judged by its wrong citations, not by accuracy alone. 

---
# ToM2Vec: Continuous Theory-of-Mind Tokens for Multi-Agent Coordination 

**Authors**: Nithin Sivakumaran, Ali Payani, Hyunji Lee, Mohit Bansal, Elias Stengel-Eskin  

**Link**: [PDF](https://arxiv.org/pdf/2610.29934)  

**Abstract**: Forming an accurate theory-of-mind (ToM) is key to successful multi-agent collaboration and requires reasoning under uncertainty to capture multiple possible partner states. This is hard to do in the text representation space native to large language model (LLM) agents, which more easily describe a single partner state. To address this challenge, we introduce ToM2Vec, which learns to form abstract theories-of-mind in a rich latent space via a fixed budget of continuous reasoning tokens. During training, supervision for these tokens comes from the partner's actual thoughts, which the ToM tokens are trained to abstract and match. At test-time, ToM tokens are generated by each agent and act as conditioners for that agent's thoughts and actions. Across several environments covering collaborative, competitive, and mixed-motive settings (Overcooked-AI, multiparty negotiation, Decrypto, Hanabi, and Collab Escape/Capture), we evaluate ToM2Vec's ability to support multi-agent interactions and collaboration, finding consistent improvements over the base model and a variety of competitive trained and prompted baselines. We show that latent reasoning is critical, with performance dropping when reasoning is expressed in text. Moreover, we show that ToM2Vec training generalizes to partner model families not seen at train-time, including a proprietary model (GPT-5.4). In our analysis, we provide evidence that the latent tokens encode a ToM, rather than task-specific information: ToM2Vec's tokens improve the agent's ability to directly answer questions about partner beliefs, desires, and next actions. 

---
# Your Benchmark Is an Empirical Measure Over Difficulty 

**Authors**: Yifan Sun, Naicheng Yu, Jiawen Gong, Jingyan Shen, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29927)  

**Abstract**: A benchmark is not merely a collection of test questions; it naturally induces an empirical measure over difficulty. In this work, we develop this measure-based view of benchmark evaluation. This perspective allows us to identify and formalize two central failure modes: measure mismatch, where the benchmark difficulty distribution differs from the intended difficulty distribution, and coverage gaps, where uncovered regions along the difficulty axis prevent benchmarks from detecting meaningful localized training progress. We propose two diagnostic metrics that capture these two failure modes and provide rigorous theoretical guarantees. Empirically, we analyze 25 popular benchmarks spanning 7 domains using carefully estimated item difficulties. Our results show that many benchmarks implicitly overweight certain difficulty regions, so their leaderboard conclusions should be interpreted with caution; others leave substantial portions of the difficulty axis uncovered, limiting their ability to monitor the model improvement process. Finally, we turn diagnosis into action by introducing BenchFill, a simple difficulty-targeted item rewriting algorithm that effectively reduces local coverage gaps. 

---
# Real PMIDs, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents 

**Authors**: Mohammed Saqlain  

**Link**: [PDF](https://arxiv.org/pdf/2610.29812)  

**Abstract**: Literature-searching language agents are becoming a standard component of scientific assistants for the life sciences, yet there is little controlled evidence on which parts of their harness matter. We run a controlled ablation on biomedical claim verification (SciFact) with the live PubMed API as the only tool. Holding the backbone and output format fixed, we compare four harnesses: closed-book answering, single-shot retrieval using the claim as the query, a ReAct-style agent that writes its own PubMed queries, and an oracle given the gold abstracts, each with two efficient frontier backbones. Naive single-shot retrieval finds the gold paper for only 6% of claims and turns overconfident errors into abstentions without improving accuracy. Letting the model formulate its own queries triples gold recall and yields the best verdicts (62% accuracy vs. 49-59% for the other realistic harnesses), although these differences are not significant at our sample size and agents use fewer than half of their search budget. With gold abstracts, the same models reach 79-86% accuracy, a significant 17-24 point gap showing that retrieval, not reasoning, is the main bottleneck. Finally, a citation audit shows that closed-book models cite PubMed IDs that almost always exist (327/335) but are almost never relevant: a cross-family LLM judge rates only 3 of 327 existing citations as relevant to the claim, versus 67% for citations produced by the search agent. Existence checks are therefore insufficient safeguards for scientific agents. We release code, prompts, and all cached tool calls at this https URL. 

---
# Conflicted Records: Typed Adversarial Contracts for Auditable LLM Output Quality Enforcement 

**Authors**: Kardelen Subasi, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2610.29669)  

**Abstract**: LLM-generated records increasingly populate production information systems where they are treated as data assets subject to the same quality expectations as records from operational databases. A single such record can simultaneously satisfy and violate a quality rule: a grounded fact adjacent to a hallucination, a useful answer adjacent to a stereotype. Classical data cleaning cannot resolve this because the two judgments are incompatible over the same span; a single LLM judge cannot either, since it shares the training biases that produced the conflict. Existing interventions such as prompt engineering, chain-of-thought, and RLHF reduce surface error rates but provide no audit trail and cannot distinguish genuine remediation from blanket suppression. The question is whether a deterministic check over structured inter-agent records can catch the defects that referential checks alone miss. We present BiasGuard-D: two adversarial roles drawn from distinct model families exchange typed claim documents, a three-model quorum adjudicates by majority vote, and a combined referential-plus-semantic invariant validates the verdict. The semantic component fires on verdicts referential checks miss; removing the iterative loop drops invariant satisfaction sharply. The deliberation layer transfers from stereotype to hallucination remediation on HaluEval QA with only role-framing changes, raising right-answer recovery by 49 percentage points; the triage gate required a cross-family second auditor to maintain recall on objective hallucination. Every claim, vote, and instruction is stored as a typed record giving operators a provenance trail auditable under emerging AI regulations. 

---
# SPARE: Sparsity Allocation via Routing Entropy for Efficient MoEs 

**Authors**: Yifan Yang, Hanling Yi, Dongquan Yang, Feng Lin, Rong Xiao, Aijun Li, Xiaotian Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29662)  

**Abstract**: Mixture-of-Experts (MoE) architectures scale model capacity efficiently, yet their substantial memory footprint poses a major barrier to deployment. Expert pruning reduces this footprint, but most methods focus on which experts to remove and prune every layer equally. We propose SPARE (Sparsity Allocation via Routing Entropy), a training-free, selector-agnostic method that decides how many experts to remove from each layer using routing entropy, the entropy of tokens' normalized Top-K gate weights on calibration data. When a token's gate weight is spread evenly, removing one active expert discards little of it, so SPARE prunes layers with more diffuse routing more aggressively. A perturbation bound under random pruning formalizes this intuition, and controlled interventions with a fixed selector link concentrated routing to larger output changes. With the same selector and budget, SPARE improves average accuracy over uniform allocation on Mixtral, Qwen3, and DeepSeek models, especially at high pruning ratios. At 50% expert pruning, it raises the eight-task average of Qwen3-30B-A3B from 57.2% to 63.8%, and the math, code, and science average of DeepSeek-R1 from 80.1 to 82.8. 

---
# WRDO: Weighted Refusal Direction Optimization in Aligned Language Models 

**Authors**: Grigoriy Evseev, Vladimir Solodkin, Dmitry Bylinkin, Daniil Medyakov, Vladimir Aletov, Aleksandr Beznosikov  

**Link**: [PDF](https://arxiv.org/pdf/2610.29555)  

**Abstract**: Aligned large language models are trained to comply with benign requests while refusing harmful ones. However, recent work shows that this refusal behavior is fragile and can often be disrupted by simple activation- or weight-space interventions. In particular, refusal has been shown to be largely mediated by a single residual-stream direction, typically computed as a difference-of-means vector between harmful and harmless prompts. However, this standard construction aggregates heterogeneous harmful prompts into one uniform average, potentially conflating distinct refusal-related phenomena. In this work, we study whether refusal interventions can be strengthened by learning how to combine prompt-specific refusal directions. We propose Weighted Refusal Direction Optimization (WRDO), a procedure that constructs a composite refusal direction from a bank of per-prompt directions using learned aggregation weights. This approach allows different harmful prompts to contribute unequally to the final intervention direction. We evaluate WRDO against prior single-direction methods on three open-weight language models using three guard judges and HarmBench as an external evaluator. We further measure the effect of the intervention on general model capability through standard utility benchmarks. Finally, we investigate a defense-oriented setting in which the same optimization machinery is used in reverse to restore refusal behavior in an abliterated model checkpoint. Our results show that weighted aggregation provides a more effective and flexible mechanism for manipulating refusal representations. 

---
# pplx-embed-v2-late Technical Report 

**Authors**: Markus Krimmel, Louis Milliken, Ekaterina Glazkova, Maximilian Schall, Antoine Chaffin, Sedigheh Eslami, Jonas Golde, Denis Bykov, Claire Beamer, Jeremy Yang, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29523)  

**Abstract**: In this report, we introduce pplx-embed-v2-late, a pair of 0.6B and 9B multilingual and multimodal late-interaction models. Both models achieve state-of-the-art results within their respective size classes across a range of text, image, and visual document retrieval benchmarks. On visual document retrieval tasks, pplx-embed-v2-late-0.6b is competitive with models several times larger than itself. The models are trained via representation distillation from an 18B teacher. Consequently, they share an embedding space that supports asymmetric deployment: documents can be indexed with the 9B model and queried with the low-latency 0.6B model. Both models are available at this https URL. 

---
# Beyond the Vote: When Do LLMs Add Value to Retrieval-Based Tabular Classification? 

**Authors**: Ziquan Wang, Yuxuan Nie, Jian Ma, Yingjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.29520)  

**Abstract**: Retrieval-based in-context learning (ICL) enables large language models (LLMs) to predict tabular labels from similar examples, but it remains unclear whether LLM inference improves on direct voting over those same examples. We address this question by comparing four general-purpose LLMs with distance-weighted nearest-neighbor voting (Distance-KNN) using identical retrieved neighbors across 19 tabular classification datasets. With the number of demonstrations fixed, enlarging the retrieval candidate pool improves accuracy for all predictors, yet Distance-KNN achieves the highest mean accuracy at every pool size. More detailed analyses reveal that LLM gains are concentrated where retrieved neighbors exhibit low label consensus and vary across models. These gains further depend on the average feature-space distance to the test query: under low label consensus, some LLMs outperform Distance-KNN in close and intermediate neighborhoods, but all fall behind on average in the most distant group. Complementary experiments on the number of demonstrations, prompt formulation, retrieval representation, and label-balanced demonstration selection largely preserve the overall findings. These results characterize when LLM inference adds predictive value beyond distance-weighted voting in retrieval-based tabular ICL. 

---
# Harness Scaling for Generalizable Agentic Reinforcement Learning 

**Authors**: Fei Bai, Huatong Song, Chuan Hao, Ran Tao, Jian Yang, Bryan Dai, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2610.29458)  

**Abstract**: Agent harnesses have become a central component of modern agentic systems, shaping how models use tools, manage context, and coordinate long-horizon interaction. Realizing the benefits of these harnesses depends on how effectively the underlying model can make use of them, motivating reinforcement learning directly through the harness. However, existing approaches typically train through a single fixed harness, which can lead to specialization and limited transfer across harness configurations. To address this, we introduce \textbf{Harness Scaling}, a new scaling dimension for agentic RL that extends training from a single harness to diverse execution configurations. Concretely, we construct a factorized harness space, select a compact training subset with balanced factorial coverage, and organize cross-harness training to combine global coverage with controlled variation for the same task. Across multiple benchmarks, model scales, and RL algorithms, Harness Scaling consistently improves generalization to unseen white-box configurations. Moreover, models trained through structured white-box variation transfer effectively to unseen external black-box harnesses, outperforming models trained through a single complex black-box harness. These results establish harness diversity as a meaningful scaling dimension for agentic RL and highlight the importance of how this diversity is organized for cross-harness generalization. 

---
# BIRD-RL: Scaling Agentic Reinforcement Learning over Stateful Data-Centric Environments 

**Authors**: Ge Qu, Jinyang Li, Xiaohan Xu, Xiaolong Li, Bowen Qin, Nan Huo, Ziwei Tang, Chenhao Ma, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.29426)  

**Abstract**: While agentic reinforcement learning (RL) has emerged as a promising paradigm for training small language models (SLMs), existing frameworks are predominantly tool-centric and struggle with data-centric environments. Specifically, applying tool-centric frameworks to stateful tasks like SQL debugging introduces severe spatial inefficiencies since proper environment isolation typically requires a separate database container per rollout. Naively reusing databases without tracking evolving states across multi-turn interactions causes trajectory contamination and cross-trajectory interference, which severely destabilizes policy updates. To overcome these bottlenecks, we introduce BIRD-RL, a novel four-stage agentic RL curriculum that alternately enhances foundational agent reasoning and trajectory-adaptive on-policy exploration to circumvent the instability inherently associated with direct end-to-end learning. Crucially, to support this curriculum, BIRD-RL is driven by a Trajectory-scoped Persistent Execution Infrastructure, which enables safe database replica reuse while preserving trajectory isolation. This design improves spatial RL efficiency by 12.8x, successfully enabling scalable agentic RL for complex data-centric tasks without needs of Kubernettes. Empirical evaluations on the BIRD-CRITIC-SQLite benchmark demonstrate that our 7B and 14B models achieve Success Rates of 44.40% and 48.00%, respectively, showing competitive performance against strong proprietary model-based agents such as Claude-Opus-4.6 Agent (48.20%). Furthermore, BIRD-RL successfully trains unified SLMs capable of both advanced SQL debugging and generation, highlighting its broader potential for multi-objective optimization across heterogeneous, real-world data-centric tasks. 

---
# CCG Diffusion: Combinatory Categorial Grammar-Driven Compositional and Syntactically Disambiguated Text-to-Image Generation 

**Authors**: Ryosuke Yamaki, Daichi Mochihashi, Nobutaka Shimada, Seitaro Shinagawa, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2610.29415)  

**Abstract**: Text-to-image generation models do not explicitly receive the syntactic structure of the prompt, i.e., which word modifies which and where each phrase attaches. Consequently, they often struggle to bind attributes to the correct objects, to depict spatial relations and counts correctly, and to render syntactically ambiguous prompts in the intended interpretation. Existing syntax-based control methods mainly address attribute binding and have limited ability to control spatial relations and counts or to specify the interpretation of syntactically ambiguous prompts. They rely on phrase-structure or dependency grammars and require additional rules or constraints on the tree structure to extract modification relations between words or to specify a syntactic interpretation. We propose CCG Diffusion, a training-free method that controls image generation using Combinatory Categorial Grammar (CCG). In CCG, the category of each word carries rich syntactic information: it specifies what the word combines with and what phrase it forms. Modification relations between words can therefore be read directly from a CCG parse. Moreover, prepositional phrase (PP) attachment ambiguity, a representative syntactic ambiguity, can be resolved simply by fixing the preposition's category. CCG Diffusion applies the same two CCG-based mechanisms, prompt rewriting and attention guidance, both to controlling attribute binding, spatial relations, and counts and to resolving PP ambiguity. On two architecturally different backbones, Stable Diffusion and Sana, CCG Diffusion achieves the highest scores among the compared methods in attribute binding and 2D spatial relations and significantly outperforms all of them in PP ambiguity resolution. 

---
# Closed-Loop Dual-Reference Guidance for Accent Control in Cross-Lingual Voice Cloning 

**Authors**: Yuanzhong Zheng, Wei Huang, Ming Ma, Kailun Jin, Youhai Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29351)  

**Abstract**: Zero-shot cross-lingual voice cloning often carries the reference speaker's accent into the target language. We propose closed-loop dual-reference guidance, which pairs the original reference with a reusable native anchor constructed by voice conversion (VC). The references initialize separate language-model (LM) states that guide a shared acoustic trajectory; each completed patch then updates both states, keeping their generated histories aligned with the emitted speech. Anchor construction runs once per speaker, and online synthesis uses a frozen TTS model without per-utterance VC. On 20 held-out French speakers, the fixed setting $\lambda^{\!*}=0.33$ reduces residual French LID mass by 73\% versus direct cloning, at an ECAPA similarity cost of only $0.042$; relative to anchor-only cloning, it recovers $0.155$. The method achieves the lowest mean WER ($0.009$) and highest mean GOP ($4.750$) among the held-out test arms. In blinded preference tests, native-English listeners favor it over direct cloning on pronunciation naturalness, and over both anchor-only cloning and a TTS$\to$VC cascade on speaker identity. Online RTF is approximately 26\% lower than the cascade's ($0.59$ vs.\ $0.80$), and the same setting reduces Spanish LID mass by 56\% without re-tuning. 

---
# Data Schedules and Learning Rate Schedules Should Be Designed Together 

**Authors**: Haodong Wen, Kairong Luo, Tingkai Yan, Wenguang Chen, Kaifeng Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29348)  

**Abstract**: Mid-training (MT) is a training phase in LLM training pipelines that continues pretraining (PT) on smaller datasets of higher value. Taking the PT and MT phases as a whole, training usually starts with only PT data and then transitions completely to MT data. In this work, we view this complete transition from PT to MT data as a special case of a broader class of data mixing schedules that specify how the proportions of PT and MT tokens in each batch change over training. Under fixed total training compute and MT token budgets, we ask: Is this complete transition to MT data optimal for minimizing final MT loss? We show that the answer depends on the learning rate (LR) schedule: one data schedule may outperform another under one LR schedule, but changing only the LR schedule can reverse the outcome. Specifically, when the LR decays throughout continued training, introducing MT data earlier and mixing it with PT data lowers the final MT loss compared with placing all MT data at the end of training, but this result is reversed if we use constant LR. We therefore study how to jointly choose when to introduce MT data and when to decay the LR through ablation studies, and find that it is better to decay the LR only after MT data enters training. Finally, we develop a theoretical model that explains why the optimal data schedule for a fixed MT token budget depends on the LR schedule. 

---
# Decodability Is Not Controllability: Local and Anisotropic Speaker Control in Continuous-Latent TTS 

**Authors**: Yuanzhong Zheng, Wei Huang, Ming Ma, Kailun Jin, Youhai Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29345)  

**Abstract**: High linear decodability does not entail causal controllability in continuous-latent text-to-speech (TTS). In VoxCPM2, position-matched probes and equal-norm interventions across 30 speakers and 60 directed dyads expose this separation: reference-audio states at the final residual acoustic language model layer (RALM L7) retain 0.970 probe accuracy, yet aligned edits yield 0/60 target-dominant dyads. Edits at early reference-audio positions produce the opposite pattern. At L0, pair-aligned updates increase target similarity by 0.326 and achieve target dominance in all 60 dyads, against at most 1/60 under any equal-norm direction or position control. Independent speaker encoders, retrieval against held-out speaker prototypes, and double-blind listening confirm that the shifted voice is heard as the intended target. Cross-architecture interventions in this http URL distinguish backbone leverage from direct acoustic control: backbone edits displace the source without target attraction, while x-vector ablation reduces source similarity from 0.849 to 0.006. These findings identify two requirements for training-free speaker control: geometric specificity and a computational route that carries the intended speaker condition into acoustic synthesis. 

---
# Uncensored Is Not Unbiased: Abliteration Removes Refusals but Not Script-Dependent Bias About Taiwan 

**Authors**: Ju-Chun Ko  

**Link**: [PDF](https://arxiv.org/pdf/2610.29342)  

**Abstract**: "Abliterated" ("uncensored") builds of open-weight language models remove the activation direction that mediates refusal, and are popular partly on the assumption that a model that no longer refuses is also free of political censorship. We test this on two matched pairs of 2-bit models: NVIDIA's Nemotron 3.5 Bolt 74B-A9B and Qwen3.8-Flash-Next, each against a community abliteration. Each model judged 22 statement pairs (on the Republic of China (Taiwan), 1989 Tiananmen, control countries) in Traditional Chinese, Simplified Chinese (a character-level conversion) and English, and answered 27 open questions per script. Script mattered far more than abliteration: the original Bolt scored +0.58 on Index S (-1: People's Republic of China (PRC) position; +1: ROC position) in Traditional Chinese, +0.57 in English and +0.05 in Simplified Chinese, and affirmed "Taiwan is part of the PRC" with probability 0.91 in Simplified but 0.06 in Traditional characters; abliteration changed its scores by at most 0.06, mostly by raising "yes" on both sides. In Qwen, abliteration removed every refusal on 1989 and raised scores by 0.21 to 0.34, yet the abliterated Qwen still affirmed that Taiwan is part of the PRC in all three scripts. Naming the state the Republic of China (Taiwan) raised every model's score by 0.10 to 0.33 but left the script gap. Control pairs showed no script gap; each abliteration changed only its declared tensors. In 740,000 web documents, Simplified text on Taiwan contained PRC-framing terms 3.7 times as often as Traditional text. A script direction could not be steered away; a contrastive stance direction raised scores in every script, and its negation lowered them. A four-sentence system prompt raised the abliterated Qwen to +0.93 to +0.97. Keyword scoring rules failed the models closest to the ROC position mostly for attributing or denying the PRC claim. Uncensored is not unbiased. 

---
# Reader--Writer Coupling in Transformer MLPs 

**Authors**: Giyeong Oh, Sunghee Ahn, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29340)  

**Abstract**: A transformer MLP gives each hidden channel two roles: a reader that decides when the channel responds, and a writer that decides what it adds back to the residual stream. Read as a key--value memory, the layer can be measured one factor set at a time: the pairing of readers to writers is carried only by their shared index, invisible to any quantity computed from one factor set alone. Yet training couples the two factors through their joint update, so we ask what reading them together shows. The block supplies the joint instrument: the MLP writes back into the very stream it reads, so the exact Jacobian of its update is square, and its agreement with its own transpose resolves into reader--writer terms whose inner products give each channel, read as a couple, a signed pairing coordinate. Ranked by this coordinate, each block we measure has a small core of strongly coupled channels; where initialization gives either sign equally, the core is nearly all opposed: each writer points against the direction its reader responds to. This opposed core recurs from the smallest models trained from scratch to released checkpoints; its sign is in place within a few percent of training and holds while the coupling grows. Read as couples, most channels return their contribution against the reader that fired it, and the readers of later blocks take it up; few show in the logits. Training holds the couples in place: flipping the opposed writers moves the output toward common tokens, and a writer permuted onto another reader regrows to fit that reader while the reader stays. 

---
# OACap: Caption-Guided Active Observation for Detailed Video Captioning 

**Authors**: Hao Shi, Hong Yan, Xiaoyang Zhang, Qing Cui, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29327)  

**Abstract**: Unlike question answering, where a query specifies a perceptual target, open-ended detailed captioning must determine what evidence to acquire and how to integrate it into a coherent global description. We propose Observation-Adaptive Video Captioner (OACap), a caption-centered agent that turns unsupported or missing information into targeted observation requests, uses spatiotemporal tools to acquire evidence, and iteratively verifies and revises the caption. OACap captions consistently improve performance across general video-understanding benchmarks using a fixed text-only reader. To better diagnose the information they preserve, we introduce Factual-Semantic Captioning Bench (FSCap-Bench). Its Core-Support Factuality Score (CSFS) combines claim precision with importance-aware fact coverage and responds as expected to four controlled perturbation types on 93.0-98.3% of valid caption pairs. Its Cross-Modal Semantic Round-Trip Score (CSRS) measures semantic preservation through a video-caption-video-caption round trip by matching atomic facts from source and recovered captions. Together, they connect downstream textual-memory utility with factual and semantic preservation. 

---
# LLM-Based Exogenous World Modeling for Dynamic Scheduling 

**Authors**: Ni Zhang, Haofeng Yuan, Jianan Zhou, Shiji Song, Zhiguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.29308)  

**Abstract**: Dynamic scheduling requires dispatching online arrivals while anticipating future jobs, as poor decisions can compound downstream costs under nonlinear tardiness objectives. We show that a pretrained large language model (LLM) can serve as the exogenous component of a structured world model, forecasting uncertain future job arrivals. Specifically, we exploit an afterstate decomposition, which separates known deterministic shop transitions from uncertain exogenous job arrivals, enabling the LLM to forecast only the latter through a compact belief representation. Building on this, we develop Afterstate World-Model Planning (AWMP), which combines an evolved, belief-conditioned dispatching strategy with short scenario-based rollouts. A surprise-triggered refresh mechanism queries the LLM only when newly observed arrivals deviate sufficiently from previous forecast, and online residual self-calibration further conditions the LLM on its recent forecast errors to handle nonstationary arrivals. Across diverse arrival distributions, AWMP reduces weighted tardiness by 6-21% on dynamic job-shop and 22-46% on flexible job-shop benchmarks, relative to the best of eight classical priority-dispatching rules. 

---
# ExpEvolve: Structuring and Reusing Experience for Open-Ended Discovery 

**Authors**: Yongqiang Chen, Chenxi Liu, Deyu Zou, Zhenhao Chen, Tongliang Liu, Bo Han, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29303)  

**Abstract**: Evolutionary coding agents such as AlphaEvolve are one of the notable successes in using Large Language Models (LLMs) to build AI Scientists. These agents tackle open-ended scientific problems by iteratively improving and evolving programs, leveraging the prior knowledge and reasoning capabilities of LLMs. Despite their success, mutations over programs are mostly low-level upon surface heuristics, without a persistent unit of control. In contrast, science compounds through ideas. In this work, we propose ExpEvolve, which injects idea-level extraction and control into evolution: it discovers and leverages the high-level factors from programs that explain scoring differences among different programs. We characterize theoretical conditions on when discovery with idea factors improves sample efficiency. We also build a practical framework to show that the extracted high-level idea factors are helpful for guiding exploitation from experience and exploration when discovery comes to a plateau. Across 11 open-ended mathematical tasks, ExpEvolve significantly improves the final performance and sample efficiency up to 5.0x. The improvements also transfer to 4 other LLMs and to 6 machine learning tasks from MLS-Bench. Analysis shows that ExpEvolve can discover idea-level factors to provide better exploration and exploitation for evolutionary discovery. 

---
# Same Probabilities, Different Advice: Gain-Loss Framing Effects in High-Stakes LLM Decision Support 

**Authors**: Yuyao Zhang, Yakun Cui, Han Zhu, Jinsong Lu, Yike Guo, Sirui Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.29294)  

**Abstract**: Large language models (LLMs) are increasingly used to support consequential decisions, yet their reliability under equivalent representations of uncertainty remains unclear. Existing work has documented prompt sensitivity, cognitive biases, and confidence miscalibration, but it is still unknown whether high-stakes advice changes when the same probability is expressed through complementary gain and loss frames. We introduce FrameBench-HPS, a controlled benchmark of 2,880 prompts derived from 240 high-stakes scenarios across healthcare, finance, education/career, and legal/consumer-rights domains. Across seven LLMs, gain-loss framing changes the recommendation category in 32.4% of complete groups, with most changes involving uncertainty rather than direct reversals. Human evaluation confirms corresponding shifts in perceived risk and directiveness. Robustness experiments show that framing effects persist under conversational paraphrases and exceed ordinary repeated-generation variability. Probability canonicalization substantially reduces framing sensitivity, identifying probability representation as a practical target for improving consistency in high-stakes LLM decision support. 

---
# Computer Vision and Knowledge-Grounded Language Models for Intelligent Beekeeper Assistance 

**Authors**: Asaduz Zaman, Mohaimenul Azam Khan Raiaan, Malika Nisal Ratnayake  

**Link**: [PDF](https://arxiv.org/pdf/2610.29283)  

**Abstract**: Managed and wild bees sustain roughly a third of the global food products, yet colonies face mounting stress from climate-change including extreme weather patterns, altered parasite and disease patterns. Timely assessment of colony condition such as understanding colony strength, brood area, honey and pollen storage, comb usage, is essential to adaptive management. However, manual hive frame inspection is slow, costly, and quality depends on scarce expertise. We present an artificial targeted intelligent system framework focusing on edge deployable pipeline that turns hive frame photographs into interpretable and action-oriented inspection report. The pipeline includes an object detector to detect biologically-meaningful comb-cell classes; detections are distilled into structured fact schema; facts are converted into constrained factual text; the text triggers a grounded knowledge-base for interpretations and recommendations. Knowledge-base entries are required to be triggerable by evidence expressed in the fact schema. We built our system based on open source FAIRHiveFrames-1K dataset and favour reproducible, edge-deployable models over large general purpose frontier models. We additionally release FAIRHiveText-1K dataset that turns FAIRHiveFrames-1K dataset into interchangeable facts and text interpretation of the facts, making it easy for further LLM-based research into this area. We also propose a pathway to develop the grounded knowledge-base that will be helpful in providing interpretable and actionable reports from the facts. 

---
# Heard But Not Said: Prompting Understates Temporal Information of Musical Moments in Frozen Large Audio-Language Models 

**Authors**: Eunsik Shin  

**Link**: [PDF](https://arxiv.org/pdf/2610.29256)  

**Abstract**: Large audio-language models (LALMs) understand music, yet struggle to locate a described moment in time. We ask whether this reflects missing temporal information or a bottleneck in how the time span is extracted. Using a paired-query probe built from Slakh2100 stem activations, we attach a state query (e.g., ``only piano and bass are playing'') and a change query (e.g., ``the guitar drops out, leaving piano and bass'') to the same span and vary how much preceding audio the model is shown. Across frozen LALMs, three ways of extracting the span ask different questions: full-track prompting asking whether the model can say when; windowed yes/no scoring asking whether it verifies the moment when handed the segment, and whether it needs what came before; attention asking whether it can find the moment on its own. State queries score above baseline without context, while change queries improve with preceding context in most short-segment models; 30s padded encoders show no comparable recovery. Prompting and attention expose almost none of this. Prompting failures thus understate what LALMs can verify, though not what they can locate, by an amount set by query type and encoder. 

---
# Preference Is Not Permission: Personalizing the Auto-Purchase Line in Agentic Commerce 

**Authors**: Ryotaro Shimizu, Sai HtaungKham, Mo Fan  

**Link**: [PDF](https://arxiv.org/pdf/2610.29216)  

**Abstract**: Personalized shopping benchmarks ask which item fits the user; we ask how far an agent may commit on that item, and pose learning the individual's auto-purchase line as a task. The boundary is evidence-conditioned and three-valued: whether committing is appropriate depends on the sufficiency of product information (warrant), reversibility, price against the user's norm, and urgency, not only on fit; and the natural output is don't buy, confirm first, or buy automatically. Across sixteen instruction-tuned models, autonomous commitment is misaligned with a human authorization reference in which fifteen annotators, role-playing the delegating principal from a given profile, choose the furthest action they would allow without confirmation: models commit even when fit is low and when their own evidence flag reports the warrant missing. This is overreach. The line is individual: these raters disagree, yet the disagreement is structured and partly predictable from a person's own other labels. A shallow learner on a few dozen of a person's own labels predicts their held-out auto-purchase line at a median of 0.79 by 30-40 labels, outperforming every evaluated prompt$\times$model configuration, including a per-annotator oracle over 80 configurations; the learned boundary does not transfer across domains. We release the counterfactual benchmark, human authorization reference, and annotation protocol after acceptance. 

---
# Reasoning-Search-Augmented LLMs: An Operational Taxonomy, Benchmark Analysis and Controlled Evidence Synthesis 

**Authors**: Biswas Poudel, Nilson Chapagain, Amit Kumar, Xianshun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29189)  

**Abstract**: Large language models (LLMs) demonstrate impressive emergent reasoning capabilities, yet they remain constrained by fixed model parameters and finite context windows. Even when augmented with static retrieval, they can produce statements that are unsupported by, or even contradictory to, retrieved evidence. This leads to hallucinations, difficulty incorporating up-to-date information, and failures to correctly interpret or reconcile sources. Recent work addresses these limitations by tightly coupling step-by-step reasoning with interactive search, treating retrieval as an action within an iterative agentic loop rather than a one-shot preprocessing step. This study reviews frameworks that implement such reasoning-search loops by interleaving planning, querying, retrieval, integration, verification, and stopping. We analyze these systems through four architectural axes: retrieval coupling, controller design, knowledge source, and grounding strictness. We separately treat the learning signal as a cross-cutting dimension because prompting, supervision, preference optimization, and reinforcement learning can instantiate the same architecture while producing materially different search, stopping, and grounding behavior. We further distinguish benchmark difficulty from RSA-loop demand and synthesize controlled within-study ablations and quality-cost deltas, avoiding invalid pooling across heterogeneous model backbones, retrievers, corpora, and evaluation protocols. Our goal is to clarify the landscape of reasoning-search-augmented LLMs and outline a unified perspective on the design choices shaping future research. 

---
# TeachBack: Learning to Teach via Bidirectional Preference Optimization for On-Policy Self-Distillation 

**Authors**: Linfei Ji, Jiayun Li, Haozhao Wang, Siyu Jiang, Zehao Wang, Longtao Xiao, Yu-Jie Yuan, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.29167)  

**Abstract**: On-policy self-distillation (OPSD) enables a language model to learn from a Teacher instantiated from the same model but given access to privileged information, such as a reference solution. We find that access to sufficient privileged information allows the Teacher to generate highly accurate trajectories; however, accuracy alone does not guarantee that these trajectories are suitable for Student learning. Direct access to a complete solution may encourage shortened, reference-conditioned derivations that omit the exploration, verification, and self-correction required to solve problems without privileged information. We introduce TeachBack, a bidirectional preference-learning framework for on-policy self-distillation that learns not only from the Teacher but also how the Teacher should teach. TeachBack shapes the Teacher's generation behavior by controlling the length of the reference-reasoning suffix injected into the prompt and by varying the prompt style. While maintaining comparable Teacher accuracy, it encourages the Teacher to explore and re-solve problems, thereby reducing its reliance on the reference solution. At each rollout refresh, the shared policy generates both a privileged Teacher trajectory and a problem-only Student trajectory. A pairwise preference objective uses privileged Teacher trajectories to improve Student learning, while the reversed preference uses the Student's behavior to shape future Teacher supervision under the privileged condition. Experiments on mathematical and logical reasoning tasks show that TeachBack consistently improves Student performance over existing OPSD methods and related baselines. 

---
# VeeReview: Measuring Build-Verify Pairs in Agent-Authored Pull Requests from a V-Model Perspective 

**Authors**: Gabriel Dautovi, Jie JW Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29143)  

**Abstract**: Context: Reviewing a pull request requires verification evidence that addresses the proposed change. Coding agent benchmarks and studies of pull request outcomes do not directly assess this relationship. Objective: We test whether the V-Model's build and verify pairs can measure verification evidence in pull requests and support comparisons between agent-authored and human-authored changes. Method: VeeReview classifies four build and verify pairs using repository signals and a language-model judge. Three pairs use model judgements constrained by deterministic signals, and a rule over the diff and repository tree decides Module Design-Unit Test. We evaluated VeeReview against a 25-pull request reference set labelled by two raters and applied it to 1,454 matched pairs of agent and human pull requests from 741 repositories. Results: VeeReview assigned lower build-only rates to agent-authored pull requests on all four pairs, contrary to the preregistered hypothesis (mixed-effects odds ratios 0.52 to 0.86). Architecture-Integration was not significant. Standardization for linked issues, shipped tests and size reduced matched-rate differences by 34% to 60%, and the rule-decided pair's standardized interval included zero. The instrument did not meet its preregistered calibration criterion: agreement with adjudicated labels was 4% to 60% against a required 80%, and inter-rater agreement was weak (kappa = 0.154). Conclusions: The rule-decided pair shows that a process measure judged by a language model partly reflects how a pull request is presented rather than the verification work behind it. We report the cohort differences as observations made with this instrument, and we argue that such measures need a mechanically decided component and a validated human reference before they can support comparisons of verification practice. 

---
# GRAM: Group-wise Rank-Aware Model Merging via Subspace Alignment 

**Authors**: Di Wu, Shuaidong Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.29110)  

**Abstract**: Merging different models into a single model is highly desirable for multi-task deployment, particularly in regulated domains like healthcare. In such settings, models are often locally fine-tuned due to strict data privacy policies, necessitating that aggregation operates exclusively on the fine-tuned weights. Existing methods apply a uniform merge across all tasks, ignoring the geometric relationships among their learned subspaces. When task updates occupy misaligned subspaces, projecting them onto a shared low-rank basis discards task-relevant directions, a failure mode named \emph{subspace interference}. Through this study, we show that the per-task projection error of a merged model is governed by the principal-angle alignment between each task and the shared subspace, so that grouping geometrically compatible tasks provably reduces this error within each group. Motivated by this observation, we propose \textbf{Group-wise Rank-Aware Model Merging (GRAM)}, which uses the principal-angle geometry of task vectors to decide which tasks should be merged together and then performs a standard SVD-based merge within each coalition; the pipeline requires only the fine-tuned weights at merge time. With only one additional model ($\nK{=}2$), GRAM significantly mitigates subspace interference and consistently outperforms prior merging methods across language understanding (GLUE with Flan-T5), 8-task CLIP vision, and medical instruction tuning (Qwen2-7B-Instruct with four English/Chinese medical LoRAs). 

---
# Same Answer, Different Reasons: Trace-Level Divergence in Multilingual Large Language Models 

**Authors**: Rui Sun, Peizhao Mei, Xiewei Cheng, Kaizhuo Chen, Taoran Li, Zhiyuan Yu, Varun Chandrasekaran  

**Link**: [PDF](https://arxiv.org/pdf/2610.29092)  

**Abstract**: Multilingual large language models are increasingly deployed in settings where visible reasoning traces support verification, education, and safety auditing, yet standard evaluations typically measure only final-answer agreement or holistic response similarity. As a result, they establish outcome consistency without determining whether users across languages receive comparable supporting evidence. We introduce a trace-level divergence framework that compares visible reasoning traces as structured sequences, decomposing agreement into structural similarity, order consistency, and holistic similarity, with pair-specific calibration to enable comparison across languages. Across three reasoning benchmarks, seven languages, and six frontier LLMs, the natural baseline---embedding each whole response and comparing the two by cosine---systematically overstates agreement: $52.1\%$--$67.3\%$ of examples in the top quartile of holistic similarity are not in the top quartile of structural similarity. This dissociation is stable across sentence encoders and step segmentations, and blind bilingual annotation studies in two English-anchored language pairs show that human judgments align with structural similarity rather than with the holistic baseline. A same-language control shows that the level of divergence is language-specific: two English traces for paraphrases of the same question agree more closely than an English trace and its counterparts in other languages. Our results show that answer-level and holistic evaluations can substantially overestimate the consistency of multilingual explanatory artifacts, motivating trace-level evaluation for reliable multilingual deployment. 

---
# VeriEvolve: Evaluator-First Evolution for Tool-Using Agents 

**Authors**: Yueshen Li, Kamer Ali Yuksel, Gokhan Tur, Dilek Hakkani-Tür, João Pedro Maia  

**Link**: [PDF](https://arxiv.org/pdf/2610.29005)  

**Abstract**: A central obstacle to self-improving agents is the absence of a reliable fitness signal for realistic stateful tasks. We introduce \textsc{VeriEvolve}, an evaluator-first evolution framework that constructs this missing evaluation layer before optimizing the agent. From task instruction, VeriEvolve synthesizes instruction-grounded challenge states and an executable evaluator that checks them. During each round of workflow evolution, the evaluator is frozen and serves as the fitness signal. Between rounds, VeriEvolve revises the evaluator by converting the end states of strong and weak runs into a pair of counterfactual test cases.
Across all 30 MCPMark Filesystem tasks, VeriEvolve constructs 30 evaluators with 141 challenge states, achieving 141/141 agreement on the generated suites and 113/122 agreement on held-out states produced by executor agents. Evaluator-guided evolution increases externally verified success from 6/30 to 18/30 for GLM-5.2 and from 27/30 to 30/30 for Claude; refinement removes all three false positives that the weak executor exposes. Cross-environment studies on PostgreSQL, Playwright, Notion, and GitHub further expose complementary limits from observability, state representation, and search/execution capacity. These results show that executable evaluators can be synthesized and used to guide search when task success is observable and an external verifier is available for calibration. 

---
# FlashNext-RL: A System Optimization Study for Long-Horizon Agentic RL of Qwen3.8-Flash-Next 

**Authors**: Zhewei Yao, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2610.28998)  

**Abstract**: Long-horizon agentic reinforcement learning (RL) requires both efficient rollout generation and training on long sequences. However, the memory requirements and model-specific operators of Qwen3.8-Flash-Next make this difficult with existing open-source implementations. To resolve these limitations, we introduce FlashNext-RL, a model-specific RL training optimization framework based on prime-rl, with vLLM for software engineering (SWE) rollout generation. Our main focus is BF16 training of the full 48-layer model on a single 8xB200 node, with the first two layers, the n-gram table, the sparse-attention indexer and the multi-token prediction head frozen. We optimize host offload, gradient accumulation and the kernels for mixture-of-experts, sparse attention and the gated residual stream. In the full-model development study, non-padding training throughput increases from approximately 395 tokens/s/GPU in the initial 32K-token configuration to 4,438 tokens/s/GPU in the optimized 131K-token configuration, with hardware matmul throughput (aka without considering the cost of attention components) increasing from approximately 20 to 213 TFLOPs/GPU. A controlled cumulative study on an 8-layer model further quantifies the contributions of individual optimizations. Together with three inference nodes, the system supports a live SWE RL run with a target batch of 1,024 trainable rollouts and a 131,072-token context limit. Over steps 5-15, an iteration takes 22.4 minutes on average, with bounded entropy and trainer-sampler mismatch KL. These results highlight the importance of tailoring RL training to the model's architecture, memory layout and kernel requirements, rather than relying only on general-purpose RL infrastructure. 

---
# To Manipulate or Not: Opinion Dynamics in LLM Interactions 

**Authors**: Mohammad Siahkamari, Phuoc Dat Nguyen, Stephen MacNeil  

**Link**: [PDF](https://arxiv.org/pdf/2610.28875)  

**Abstract**: Large language models increasingly take part in extended conversations where they can both shape and be shaped by another party's views, yet most persuasion research measures only one side of that exchange. We introduce a controlled LLM-LLM debate framework in which two language-model roles repeatedly discuss a contested topic (climate-change attribution): one is given an assigned stance and, in some conditions, an explicit instruction to persuade, while the other receives no assigned position. Across four ordered model combinations involving GPT-6 Luna and Gemini 2.5 Flash Lite, we track both each role's explicitly reported position and a separate measure of its turn-by-turn behavior inside the conversation. Opinion dynamics differ substantially across model combinations: same-model interactions narrow the opinion gap on average, clearly for only one of the two models, while cross-model interactions widen it. Explicit reports and in-dialogue behavior capture distinct aspects of opinion dynamics, and a blanket instruction to persuade does not reliably increase influence. These findings suggest that conversational influence depends not only on assigned roles and initial positions but also on the participating models, with implications for how opinion change is measured in extended LLM interactions. 

---
# Beyond Correct Answers: How Compositional Cues Shape Visual Judgments 

**Authors**: Xiaoyu Ma, Zheyuan Zhang, Kaiwen Shi, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.28843)  

**Abstract**: Visual question answering requires models to assign attributes and relations to the correct objects, a problem known as visual binding. Structured prompts make these assignments explicit through labeled fields. Yet an improvement from the full prompt does not reveal which labels help or how they interact. First, we isolate field-label effects across ten models and six tasks by holding the image and claim fixed. These comparisons reveal differences that overall accuracy can hide. With a relation label present, adding object-role labels helps Qwen3 but hurts Qwen2 on Spatial. Second, we compare auxiliary hints that preserve or swap the question's role assignments against a no-hint baseline. Across two image sets, Qwen3's advantage with preserving hints mainly reflects errors caused by swapped hints. Preserving hints remain near baseline. Field order further affects the harm. Third, we test clearer hint wording and direct selection of the relation shown in the image. Clearer hints improve text matching but do not consistently improve visual judgments. In contrast, relation selection partly reduces errors for some hints in Qwen3 and GPT-4.1-mini. 

---
# Where Should a Stuck Agent Spend Its Next Tokens? Causal Attribution of Inference-Time Resources 

**Authors**: Saiyue Lyu, Jerrold Huang, Yunbei Zhang, Janet Wang, Zhenting Wang, Yingqiang Ge, Haibo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28834)  

**Abstract**: Inference-time scaling is often treated as a token-budget problem, but a failed coding agent can seek qualitatively different resources: computation, information, or feedback. We introduce Banzhaf Resource Attribution (BRA), a controlled factorial framework that measures each resource's marginal contribution to repair and its interactions while holding the failed state and revision protocol fixed. We evaluate six models on 194 function-level failures from MBPP+ and HumanEval+ and 126 repository-level failures from SWE-bench Verified. Task-relevant specifications consistently improve repair, whereas public-test feedback provides small gains and retrieved code little to none. The value of additional computation declines with model capability, while information remains useful; resource channels largely exhibit diminishing returns, although complementary information can reinforce itself. Yet three of four frontier models select feedback in 95-100% of cases and obtain lower gains than uniform random resource selection. BRA thus exposes a gap between the resources agents prefer and those that measurably improve repair. 

---
# Combining Long-Term Memory and LLMs Comes at a High Cost: From Prompt Injection to Persistent Compromise 

**Authors**: Pedro H. Barcha Correia, Matheus Azevedo de Sá, Paulo H. Gomes de Senna, Giovanni Mandel Martignago, Victor Takashi Hayashi, Charles Christian Miers, Marcos A. Simplicio Jr  

**Link**: [PDF](https://arxiv.org/pdf/2610.28810)  

**Abstract**: The widespread adoption of generative artificial intelligence in recent years has brought a parallel rise in prompt injection and jailbreak attacks. By embedding malicious instructions, these exploits manipulate language models to bypass intended behavior and operational boundaries. In traditional chatbots, such attacks are confined to a single session because state memory is not preserved across conversations. However, persistent memory features in modern conversational interfaces enable cross-session attack propagation, establishing a threat vector known as memory injection. This work explores this emerging vulnerability in leading LLM-as-a-service platforms, highlighting its real-world severity and the need for countermeasures. While existing literature focuses on violations tailored to specific platform features, we present a general threat model. We demonstrate that all NIST attack goals are collectively achievable across ChatGPT Instant 5.3, Claude Sonnet 4.6, and Gemini 3 Flash under this model. We also show that memory injections can exploit multiple attack vectors, including personal memories, projects, skills, organization instructions, and Model Context Protocol (MCP)-connected resources. Our findings reveal that all evaluated services can be manipulated to persistently spread disinformation, generate harmful content, or simulate offline unresponsiveness, even in fresh sessions. In the particular case of Claude, shared-memory features allow an attacker to compromise multiple users within an organization and also exfiltrate user prompts. To address these vulnerabilities, we derive defense mechanisms from our proposed threat model and foundational cybersecurity principles. Our analysis indicates that nearly half of these mitigations remain unimplemented across the target services. Datasets and attack artifacts are available at this https URL. 

---
# CALLIOPE: Private Insights into AI Use, Without Embeddings 

**Authors**: Edith Cohen, Vadym Doroshenko, Badih Ghazi, Charlie Harrison, Pritish Kamath, Alexander Knop, Ravi Kumar, Ethan Leeman, Pasin Manurangsi, Mikhail Pravilov, Adam Sealfon, Yurii Sushko, Marika Swanberg, Chiyuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28797)  

**Abstract**: We introduce CALLIOPE, a top-down framework for discovering topic hierarchies in large corpora of unstructured data (e.g., LLM conversations), while supporting differential privacy (DP). Unlike traditional and even recent LLM-augmented approaches that ultimately rely on clustering vector embeddings, our framework leverages LLM capabilities directly for the entire pipeline. Experiments demonstrate that this approach provides better utility over existing (embedding-based) methods in the DP setting, especially in the high privacy regime. We evaluate the quality of CALLIOPE outputs using a variety of traditionally studied clustering metrics, as well as some novel utility metrics that we introduce. 

---
# Measure First, Then Allocate: Calibrated Rollout Budgets for MCP Agents in Reinforcement Learning 

**Authors**: Hieu Tran, Zonghai Yao, Sunjae Kwon, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28785)  

**Abstract**: Training tool-using language models requires costly interaction, yet identical rewards within a rollout group provide no group-relative learning signal. We use reward disagreement to guide harness evaluation and the allocation of rollouts for policy training in Model Context Protocol (MCP) environments. We first introduce Discord, which tests harness edits using exact paired inference and decision-preserving early stopping: only disagreeing paired outcomes favor one harness over another. This insight motivates prospective allocation in BRACE (Bank-calibrated Rollout Allocation under Correlated Episodes). BRACE combines historical task evidence, uncertainty in task success rates, and a correlated reward model to forecast mixed-reward groups before allocating generation budget. Controlled harness experiments favor Discord, while live MCP experiments expose a retention-cost trade-off. At three Qwen3-VL model scales, BRACE exceeds the strongest plotted baseline peaks with fewer new training rollouts; historical-bank collection is counted separately. It also leads the compared methods on BFCL and NESTFUL without further training. Ablations probe the bank, estimator, and allocation score; prospective evaluation checks forecast calibration. These results support forecasting reward contrast before generation. 

---
# Automating ML for Science: Can Frontier Agents Climb Scientific Hills in the Wild? 

**Authors**: Ming Zhong, Stacy Li, Nicholas Carlini, Matthew Jagielski  

**Link**: [PDF](https://arxiv.org/pdf/2610.28763)  

**Abstract**: Frontier coding agents are increasingly framed as end-to-end automated researchers, and ML for science offers a demanding test of that promise: improving a scientific model often requires domain insight beyond routine ML engineering. We introduce Agent4Sci, a long-horizon testbed for measuring how far today's agents can improve the published best models on scientific ML benchmarks, spanning 20 benchmarks across 11 domains. Rather than ranking agents on a leaderboard, Agent4Sci asks what kinds of progress agents can make on these tasks under in-the-wild conditions, and where they fall short. Empirically, today's frontier agents advance the published best on every benchmark, with Claude Opus reaching 10.4% mean relative improvement and no run flagged for academic misconduct. These gains, however, come almost entirely from a bag of ML-engineering tricks rather than research-level insight. Trajectory analysis surfaces three recurring limitations in this setting: novelty miscalibration, resource under-utilization, and rapid collapse into local engineering sweeps. Lightweight plug-in skills for research ideation, experiment management, and scheduled checklists raise Claude Opus from 10.4% to 20.1% and Codex from 4.4% to 8.1%, while yielding more diverse, domain-relevant interventions. Agent4Sci gives the community a way to track and push frontier agents from ML engineering toward domain insight on real-world ML-for-science problems. Code and data are available on GitHub (this https URL) and Hugging Face (this https URL). 

---
# A Reverse-Gold Theorem: Positive-Data Identification by Substrate-Bounded Contrastive Learners 

**Authors**: Rakesh Sengupta  

**Link**: [PDF](https://arxiv.org/pdf/2610.28747)  

**Abstract**: Gold's theorem shows that no learner can identify an arbitrary language in a superfinite class from positive presentations alone. We prove a positive result of a different shape. Fix an innate \emph{recursive contrast operator} $R$ mapping any string to a finite set of structurally bounded perturbations of it, and a discriminative scorer of bounded capacity trained by InfoNCE-style contrastive estimation \citep{vandenoord2018,macollins2018} on self-generated perturbations. Under a faithfulness condition relating operator, scorer, and target, the trained scorer achieves \emph{PAC conditional identification} from positive data alone: with high probability it ranks all but an $\varepsilon$-fraction of positives above every out-of-language perturbation in their own orbit, with sample complexity polynomial in the capacity, the orbit size, the score bound, and $1/\varepsilon$. The negatives are constructed by the learner, not supplied by an oracle, and no global normalizing constant over strings is ever computed. The misranking bound follows from the geometry of the contrastive loss---per-example loss floors plus a definite loss increment under misranking---yielding an $\varepsilon^{-2}$ rate, together with an explicit approximation gap that quantifies the cost of in-language (``colliding'') perturbations and vanishes for orbit-pure operators. The ranking guarantee lifts to a global membership classifier under a margin condition, and the same operator composed with the learned scorer is a normalization-free Metropolis--Hastings generator. We instantiate the hypotheses for substitutable context-free languages \citep{clarkeyraud2007} and the subregular classes \citep{heinz2010,hayashikamigaito2025}, and state an open conjecture locating the construction in the mildly context-sensitive hierarchy. 

---
# Qwen-Audio-3.1-ASR Technical Report 

**Authors**: Chuanmeng Bian, Daren Chen, Peixin Chen, Zhigao Chen, Chong Deng, Zhiyun Fan, Zhifu Gao, Bo Gong, Qing Gu, Jiajun He, Yawei Hu, Yunjie Ji, Wei Ju, Jingbei Li, Xiangang Li, Xu Li, Zengxi Li, Zheng Li, Chengdong Liang, Baiji Liu, Yaming Liu, Ying Liu, Bin Ma, Yiping Peng, Yuezhang Peng, Zhendong Peng, Yu Pu, Yang Shi, Xin Shu, Jian Tang, Biao Tian, Peiyao Wang, Tianzi Wang, Wen Wang, Wupeng Wang, Cheng Wen, Yuzhong Wu, Zijian Xia, Yunchong Xiao, Nan Yang, Jianwei Yu, Jixing Yu, Binbin Zhang, Lei Zhang, Sitong Zhao, Guangdong Zhou, Yuan Zhou, Jianheng Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2610.28711)  

**Abstract**: We present Qwen-Audio-3.1-ASR, a family of audio-language models that builds upon its predecesor Qwen-Audio-3.0-ASR with improved speech recognition and expanded capabilities in speaker-diarized transcription and general audio understanding. The family retains support for 30 languages and 16 Chinese dialects, together with instruction-controlled capabilities including historical-context conditioning, hierarchical hotword customization, and native transcription polishing. Speaker-diarized ASR produces structured outputs containing speaker labels, timestamps, and recognized text, maintaining consistent speaker labels across turns and the representation of overlapping speech. The general audio understanding capabilities cover audio-, speech-, and music-captioning, temporal grounding and sound event detection, as well as audio question answering and multiple-choice reasoning. These tasks broaden the application scope from recognizing spoken words to interpreting speakers, acoustic events, and their temporal relations. To accommodate different deployment requirements, the family includes variants for asynchronous file transcription, low-latency streaming, interactive voice input, and non-streaming recognition, with speaker-diarized transcription and general audio understanding available in certain variants. Evaluations span conventional ASR, multilingual and dialectal speech, speaker-diarized transcription, and general audio understanding, demonstrating the capabilities of the Qwen-Audio-3.1-ASR family across transcription and broader audio analysis tasks. 

---
# Beyond source attribution: Address authorship and comparability decisions in an LLM-built scholarly knowledge graph 

**Authors**: Ksenia Kozha, PaRAGraph Research Project  

**Link**: [PDF](https://arxiv.org/pdf/2610.28663)  

**Abstract**: A knowledge graph needs a stable address for its evidence even where the literature offers competing classifications and agrees on none. Yet an address chosen for retrieval readily passes for the only classification available. This happens whenever records gathered under one name concern units that differ in boundaries, rank, or the criteria by which they are described or counted. In Sinitic dialectology the same localities recur in schemes whose units differ in membership and rank, and a unit may change its affiliation when the evidence for its diagnosis changes. Two tone counts, each quoted faithfully, may still count different things. The taxonomy of an entire field thus comes to rest on the unstable frame of a literature that contradicts itself. Neither verifying a quotation nor searching for conflicting values detects these differences. I distinguish the working address that gathers candidate evidence, the category that the source defines and the decision that admits a passage to a particular comparison, and argue that such decisions, together with the dependence among records, belong to the provenance of a comparison. On four re-runnable cases, the accompanying PaRAGraph Casebook shows where such differences arise among claims extracted by a language model and verified before entering the graph. The platform now stores the corresponding decisions as objects for one dimension of comparability, the address of a record: the placement of a locality in each source's scheme, the decision that a name in a given publication denotes a unit rather than a place, and the origin of every address. The working taxonomy can then be revised as new sources yield new claims, while the path from each comparison to its evidence stays fixed. Keywords: knowledge graphs; source attribution; classification; claim extraction; comparative concepts; Sinitic dialectology; Yue 

---
# TAGR: Task-Adaptive Geo-Semantic Retrieval across Heterogeneous Urban Delivery Tasks 

**Authors**: Zhi Liu, Yurun Cai, Xiao Han, Zhi Zhen, Yuran Li, Guojiang Shen, Xiangjie Kong  

**Link**: [PDF](https://arxiv.org/pdf/2610.28638)  

**Abstract**: As urban residents demand higher-quality instant delivery services, demand forecasting, estimated time of arrival prediction, and next-stop prediction have become critical tasks in urban delivery systems. However, the three tasks rely on historical evidence with substantially different structures and granularities. Meanwhile, their prediction targets differ in output form, making it difficult to integrate retrieved evidence through a single fusion mechanism without causing mismatch or negative transfer. These issues can increase prediction errors and subsequently lead to imbalanced resource allocation, where some regions experience unmet demand. In contrast, others suffer from underutilization, ultimately degrading overall efficiency and service quality. To address these challenges, we propose TAGR, a Task-Adaptive Geo-Semantic Retrieval framework across heterogeneous urban delivery tasks. TAGR constructs an LLM-based geo-semantic shared prior and task-specific retrieval banks to share regional knowledge while preserving heterogeneous task-specific evidence. The LLM-derived regional representations encode geographic context and semantic relations among regions, providing transferable knowledge for all three tasks without requiring online LLM inference. It further introduces a task-adaptive retrieval augmentation module to accommodate heterogeneous output structures, thereby mitigating information mismatch and negative transfer caused by a uniform retrieval-fusion strategy. Experiments on three city datasets demonstrate that TAGR consistently outperforms state-of-the-art baselines across multiple tasks. 

---
# RESEARCHHUB: A Personalized Literature Discovery System for Biomedical Preprints 

**Authors**: Viswonathan Manoranjan, Taeyoon Kim, Nicholas Do, Sang Woo Kim, Dawei Li, Zhen Tan, Tianlong Chen, Yunmei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28636)  

**Abstract**: We present ResearchHub, an interactive system for personalized literature discovery over biomedical preprints. Given a researcher profile or a free-form query, ResearchHub retrieves relevant papers from a domain-specific MedRxiv corpus and helps users move beyond near-duplicate recommendations through controllable novelty, diversity, and curriculum ordering. At the core of the system is prism, a bi-encoder retriever adapted to biomedical preprints using GPT-4o-generated natural language queries and a hybrid Multiple Negatives Ranking plus MarginMSE objective with cross-encoder soft labels. On 44,937 MedRxiv papers from 2023--2025, prism outperforms strong off-the-shelf dense retrievers, improving Recall@10 over SPECTER2-Search by 12.4 percentage points. ResearchHub further supports frontier retrieval based on a user's reading history, DPP-based diversity re-ranking, Local Intrinsic Dimensionality ordering from mainstream to specialized papers, and lightweight attribution cards for result explanation. The full pipeline runs with low latency, making the system suitable 

---
# Breaking Diffusion: Parallel Reasoning with Draft Refinement 

**Authors**: Jaesang Yu, Seoyeon Byun, Yun Wang, Yan Lu, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.26542)  

**Abstract**: Drafting an answer allows a reasoning model to reconsider each decision in the context of the others. Yet repeated diffusion denoising does not ensure that individually plausible token updates form a better answer together. We introduce Draft Refinement Language Models (DRLMs), which use a single denoiser to propose parallel revisions and evaluate a candidate draft. We train the denoiser to reconstruct every token from uniformly corrupted drafts, teaching both preservation and repair. Its predictions specify revisions directly, without a diffusion reverse transition. Evaluation requires a different use of these predictions: confidence in an existing token reflects both contextual support and evidence from observing the token itself. Using the known corruption process, we correct for this self-token evidence to estimate how well each token is supported by the prompt and the remaining draft. Aggregating these estimates yields a cavity energy computed in one forward pass, without an additional scoring model. Each round proposes a revision, reads the completed candidate in its revised context, and retains it only if its energy is lower. Across Countdown, Sudoku-Extreme, and Maze-Hard, our method outperforms autoregressive and discrete diffusion baselines. With 6M parameters, DRLM reaches 44.4% on Countdown-5 and 93.9% on Sudoku-Extreme. The results establish complete-draft parallel refinement as an effective alternative to reverse diffusion for complex reasoning. 

---
