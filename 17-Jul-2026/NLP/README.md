# Partition, Prompt, Aggregate: Statistical Self-Consistency in Language Models 

**Authors**: Patrik Wolf, Thomas Kleine Buening, Andreas Krause, Celestine Mendler-Dünner  

**Link**: [PDF](https://arxiv.org/pdf/2607.15277)  

**Abstract**: In-context learning is commonly interpreted as a form of conditional inference, in which the prompt specifies a context and the model's output is treated as an estimate of the corresponding conditional distribution. If this interpretation holds, then LLM estimates should satisfy basic probabilistic identities. In particular, the law of total probability asserts that prior-weighted conditional distributions aggregate into population-level marginals over any valid partition of the population. In this work, we investigate to what extent LLM estimates adhere to this self-consistency principle. We use binary trees as an evaluation scaffold to recursively partition a population into increasingly fine-grained subpopulations. We then prompt LLMs with verbalized subpopulation descriptions in context, aggregate the resulting estimates back into population-level estimates, and compare them across partitions of varying granularity. Applying this protocol across problem domains and state-of-the-art frontier models, we show widespread violations of basic consistency properties. An in-depth study of persona prompting reveals a pattern we call the macro fallacy: estimates reconstructed from more fine-grained subpopulation responses are often better aligned with human reference data than direct population-level estimates. This effect persists across variations in tree structure and estimation task, and can be partially recovered through implicit prompting. Together, these findings suggest that models possess relevant subpopulation knowledge but do not reliably propagate it into aggregate estimates. This gap establishes statistical self-consistency as an unsaturated, reference-free criterion for evaluating LLMs. 

---
# SciDiagramEdit: Learning to Edit Scientific Diagrams from Paper Revisions 

**Authors**: Yasheng Sun, Zezi Zeng, Yifan Yang, Chong Luo, Wenyi Wang, Ziwei Liu, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2607.15272)  

**Abstract**: Editing the figures in a research paper is a routine and time-consuming part of everyday research practice: authors relabel components, rearrange panels, and restyle visuals as they revise their manuscripts. Automating this editing workflow under a natural-language instruction, however, is challenging, because a scientific figure is a dense infographic in which heterogeneous visual elements such as schematics, plots, photos, captions, and arrows are composed under a tight visual grammar to advance a specific argument. To address this, we present SciDiagramEdit, a benchmark and skill-evolution framework that learns from natural paper revisions and operates on the figure's editable vector source, where users can inspect and co-edit individual primitives alongside the agent. Our benchmark mines before/after figure pairs from arXiv version histories, each grounded in the authors' own revision intent. To accommodate the diversity of editing instructions, we adopt agentic learning via skill evolution: an agentic proposer continually refines the agent's skill specification from execution traces over multiple epochs. The resulting skill progressively lifts edit accuracy on a held-out validation set, providing evidence that natural paper revisions are an effective training signal for instruction-driven figure editing. 

---
# Beyond the Leaderboard: Design Lessons for Trustworthy Multimodal VQA 

**Authors**: Sushant Gautam, Vajira Thambawita, Michael A. Riegler, Pål Halvorsen, Steven A. Hicks  

**Link**: [PDF](https://arxiv.org/pdf/2607.15241)  

**Abstract**: Healthcare multimodal AI must combine visual and textual evidence while remaining reliable and interpretable. Using MediaEval Medico 2025 as a retrospective GI endoscopy case study, we analyze design choices across nine documented systems for question answering and explanation quality. Parameter-efficient adaptation of pretrained backbones provides strong challenge performance, but answer-level gains do not consistently translate into faithful and complete clinical reasoning. Methods enforcing structured reasoning and explicit grounding show more reliable behavior across heterogeneous question types, although the evidence is correlational rather than ablation-based. These results motivate evaluation beyond lexical overlap, standardized evidence-linked explanations, leakage-aware data governance, and lightweight robustness and calibration checks. The findings support trustworthy multimodal healthcare AI based on data fusion, explainability, and resilient evaluation. 

---
# TikStance: A Multimodal and Hierarchical Dataset for Multi-target Stance Analysis in TikTok Political Conversations 

**Authors**: Yazhi Zhang, Fuqiang Niu, Bowen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.15240)  

**Abstract**: Political discourse has increasingly moved to short-video platforms, yet computational analysis of such content remains constrained by the scarcity of datasets that jointly preserve audiovisual information and hierarchical conversations. Here we present TikStance, a multimodal and context-aware dataset comprising 161 videos and 13,876 comments from TikTok, designed for stance detection in political discussions. The dataset covers three major political figures in the 2024 U.S. election cycle--Donald Trump, Joe Biden, and Kamala Harris--with content collected between September 2023 and January 2025. Each discussion unit links a host video and its metadata to a parent-linked comment tree, enabling stance analysis within both audiovisual and conversational context. Each item was independently labeled by three annotators using a three-class scheme (Favor, Against, None) for video-to-target and comment-to-target stance; items with disagreement were re-annotated, and the final Krippendorff's \(\alpha\) reached 0.743, 0.723, and 0.722 for the Trump, Biden, and Harris subsets, respectively. Descriptive analysis further reveals target-dependent differences in stance distributions and conversational depth, with nested replies accounting for 23.3\% of all comments. By combining multi-target coverage, hierarchical conversations, and reliable multi-level human annotations, TikStance supports research in multimodal stance detection, political communication, computational social science, and context-aware natural language processing. 

---
# Language Identification via Compositional Data Analysis: A Linear-Time Classifier Based on Log-Ratio Geometry 

**Authors**: Paul-Andrei Pogăcean, Sanda-Maria Avram  

**Link**: [PDF](https://arxiv.org/pdf/2607.15238)  

**Abstract**: Language identification is commonly addressed using either neural architectures or statistical n-gram models. Neural approaches typically require substantial computational resources, whereas classical frequency-based methods offer efficient linear-time performance, but rely on distance metrics that are not always appropriate for compositional data.
This work models character and bigram frequency distributions as compositional vectors constrained to the simplex and mapped via the centered log-ratio (CLR) transformation bijectively onto the $(D-1)$-dimensional zero-sum subspace of $\mathbb{R}^D$, where Euclidean distances correspond to Aitchison distances. A pipeline is proposed, combining CLR-transformed unigram and bigram features with Laplace smoothing to address sparsity. The method is evaluated on six languages.
Experimental results show that the proposed approach achieves robust accuracy across different text lengths, with strong performance for longer sequences. These findings indicate that compositional representations provide a deterministic and computationally efficient alternative for language identification, particularly in settings where interpretability and low resource consumption are essential. 

---
# In-Place Tokenizer Expansion for Pre-trained LLMs 

**Authors**: Jimmy T.H. Smith, Tarek Dakhran, Alberto Cabrera, Simon S. Lee, Paul Pak, Aditya Tadimeti, Tim Seyde, Maxime Labonne, Alexander Amini, Mathias Lechner  

**Link**: [PDF](https://arxiv.org/pdf/2607.15232)  

**Abstract**: A tokenizer fixed at the start of pre-training allocates vocabulary in proportion to the pre-training corpus, reflecting the deployment priorities at that time. When those priorities shift, languages added later are split into many more tokens per word, which can raise latency, compute, and energy consumption for users of those languages. Cloud models can afford a broad vocabulary because the embedding and LM-head matrices are a small fraction of their parameters. On a compact model those matrices are a material share of per-token decode bandwidth, so on-device models ship small vocabularies and accept fragmentation outside a fixed language set. We present tokenizer expansion, an in-place recipe for upgrading a pre-trained model's tokenizer when the model producer controls its design. We continue the existing tokenizer's BPE merges on a multilingual corpus, so most source tokens carry over unchanged as single tokens and every new token has an exact decomposition into source tokens. We copy the carried-over embedding rows unchanged and initialize new rows as the mean of their source sub-token embeddings. A two-stage adaptation, embedding-only training then full-model continued pre-training, recovers source-checkpoint quality. We apply the recipe to a continued pre-trained checkpoint of LFM2-8B-A1B, an 8B-parameter Mixture-of-Experts model, to help produce LFM2.5-8B-A1B with a 128K tokenizer. The expanded tokenizer encodes Hindi and Vietnamese in roughly $2.4\times$ and $2.6\times$ fewer tokens than the source (up to $4.0\times$ on Thai). Combining these reductions with the measured per-token cost of the larger vocabulary, we estimate a $2.2$-$3.7\times$ per-character decode speedup for these languages across our reference devices. We release the model weights and the expanded tokenizer, and report the negative findings that shaped the recipe. 

---
# Expanding the Lexicon of Ge'ez Based African Languages: A Comparative Study of Amharic and Tigrinya 

**Authors**: Hailay Kidu Teklehaymanot, Debela Desalegn Yadeta, Wolfgang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2607.15209)  

**Abstract**: Multilingual pre-trained language models (PLMs) exhibit degraded performance on low-resource, non-Latin-script languages, driven by high out-of-vocabulary (OOV) rates and excessive subword fragmentation that result from Latin-script-centric tokenizer training. We introduce VEXMLM, a vocabulary-extended variant of XLM-R targeting the two highest-resource Ge'ez-script languages, Amharic and Tigrinya, and further evaluated on 17 additional low-resource African languages (19 total). We train a language-specific SentencePiece tokenizer on curated Amharic and Tigrinya monolingual corpora, extend XLM-R's vocabulary with 30,000 Ge'ez-script subwords derived from this tokenizer, and initialize their embeddings by averaging the embeddings of their constituent subwords under XLM-R's original tokenizer. VEXMLM is trained in two stages: (1) continued masked language modeling over the extended vocabulary on the curated corpora, and (2) supervised fine-tuning on question answering (QA), named entity recognition (NER), and sentiment analysis (SA). On Amharic/Tigrinya QA, VEXMLM achieves 87.0 EM /90.0 F1, versus 66.0 EM/78.0 F1 for XLM-R and 74.0 EM/ 78.0 F1 for Glot500. On SA, VEXMLM reaches 80.0\% accuracy versus 77.0\% (XLM-R) and 46.0\% (Glot500). On NER, VEXMLM raises OOV-token entity accuracy from 81.4\% to 94.3\%, averaged over 11 of the 19 evaluated languages for which OOV analysis was possible. Our contributions are: (i) a vocabulary-extension and embedding-initialization procedure tailored to Ge'ez script; (ii) a two-stage training strategy under which vocabulary and continued-pretraining gains on Amharic/Tigrinya transfer to 17 typologically related, unaugmented African languages; and (iii) an evaluation spanning both intrinsic tokenization metrics (vocabulary coverage, fertility, OOV rate) and extrinsic task performance across all 19 languages. 

---
# Mask-Aware Policy Gradients for Diffusion Language Models 

**Authors**: Haran Raajesh, Kulin Shah, Adam Klivans, Philipp Krähenbühl  

**Link**: [PDF](https://arxiv.org/pdf/2607.15200)  

**Abstract**: Reinforcement learning has proven effective for improving reasoning in large language models, but extending it to Masked Diffusion Language Models (MDLMs) remains challenging due to the intractability of the log-likelihood estimation. Existing approaches approximate this log-likelihood by modeling only the token predictions, ignoring the order in which positions are unmasked during generation. We observe that MDLM generation involves two decisions at each step: what tokens to place at each masked position and which positions to remask. We formalize this as a two-stage action MDP, showing that the policy gradient naturally decomposes into a token term and a masking term. Combining optimization of both terms leads to state-of-the-art outcomes on mathematical reasoning and coding benchmarks, with scores of 87.1% on GSM8K and 53.4% on MBPP. 

---
# T^2MLR: Transformer with Temporal Middle-Layer Recurrence 

**Authors**: Ziyang Cai, Xingyu Zhu, Yihe Dong, Yinghui He, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2607.15178)  

**Abstract**: Transformer reasoning is limited by autoregressive decoding, which repeat edly compresses rich hidden computation through token space and makes it difficult for intermediate reasoning states to persist across time. We in troduce Transformers with Temporal Middle-Layer Recurrence (T2MLR), a transformers-based latent reasoning architecture that fuses a cached middle layer representation from the previous token directly into an earlier layer of the current token position, enabling abstract intermediate computation to persist across decoding steps with little inference overhead. Across natural-language pretraining and multi-hop reasoning finetuning, T2MLR consistently outperforms data- and parameter-matched Transformer base lines. Moreover, applying recurrence to only a localized middle-layer block (as little as 20% of the network) often outperforms full-layer recurrence. Im portantly, T2MLR does not require pretraining from scratch: retrofitting the recurrent pathway into an existing pretrained 1.7B Transformer and briefly finetuning substantially improves math reasoning, lowering the barrier to practical adoption. These results suggest that effective latent reasoning in Transformers does not require looping over all layers as in previous works, but can instead emerge more strongly from targeted middle-layer recurrence. 

---
# Linear representations of grammaticality in neural language models 

**Authors**: Jane Li, Najoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.15175)  

**Abstract**: Whether neural language models (NLMs) possess the ability to distinguish strings on the basis of their grammaticality remains a debated topic in the computational linguistics literature. Existing evidence has largely relied on probability-based measures, testing whether models assign higher probabilities to grammatical than ungrammatical strings. However, probability comparisons have been criticized as a measure for grammatical knowledge based on the assumption that grammaticality is inherently entangled with likelihood. Model-assigned probability is a function of many related sentence properties, such as lexical frequency, plausibility, and world knowledge. In this work, we move beyond probability-based evaluations and investigate whether grammaticality is encoded in the internal representations of NLMs. Using mass-mean probing, we test whether grammatical and ungrammatical sentences are systematically separated in representational space. We further examine the extent to which these representations are independent of sentence properties that are correlated with grammaticality, as well as their generalization across grammatical phenomena and languages. Our results provide evidence that grammaticality is robustly encoded in sentence representations of a wide range of pretrained NLMs, yielding clear representational separation on the dimension of grammaticality that cannot be fully explained by alternative sentence-level factors. Moreover, this encoding generalizes across a broad range of grammatical phenomena and to some degree, across languages, suggesting that grammaticality constitutes a coherent representational dimension in contemporary NLMs. These findings contribute new evidence to debates about the nature of syntactic knowledge in language models and offer a complementary framework for evaluating grammatical competence that is not dependent on string probabilities alone. 

---
# Grokipedia vs Wikipedia: An LLM-Based Audit of Political Neutrality along Ideologies 

**Authors**: Filippos Vlahos, Guillaume Bied, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2607.15146)  

**Abstract**: Online encyclopedias shape political opinion and, through it, democratic discourse. In late 2025, Grokipedia was released, an encyclopedia written entirely by the LLM Grok. One motivation behind the project was to provide an unbiased alternative to Wikipedia, which has faced accusations of "left-wing" and "liberal" bias. But does an encyclopedia written by an LLM deliver greater neutrality, or does it simply embed a different ideology? We conduct a large-scale political bias study on Grokipedia and Wikipedia, analysing 1,394 article pairs describing members of government for neutrality along nine expert-coded ideology dimensions employing four LLM judges, Grok, Claude, Mistral, and DeepSeek. As the LLMs could themselves be biased, we also investigate patterns in their judgments. We find all LLM-judges, including Grok, to rate Grokipedia less neutral than Wikipedia. Both encyclopedias are rated as portraying politicians favourably overall, but towards different ideological groups. Grokipedia particularly favours economically right-wing politicians and penalises socially liberal ones, while Wikipedia is rated as favourably biased towards the latter. 

---
# Digital Pantheon: Simulating and Auditing Coalition Formation with LLM Agents 

**Authors**: Dylan Van Mulders, Matthias Bogaert, Dirk Van den Poel  

**Link**: [PDF](https://arxiv.org/pdf/2607.15095)  

**Abstract**: The formation of political coalitions is a complex negotiation driven by both concrete policy objectives and deep-seated ideological convictions. While Large Language Models (LLMs) open new avenues for computational political science, the neutrality and helpfulness biases instilled by Reinforcement Learning from Human Feedback (RLHF) prevent them from sustaining steadfast partisan behaviour. We present a multi-agent framework that reconciles factual grounding with ideological alignment by combining Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Retrieval-Augmented Generation (RAG): DPO instils aggressive party-specific personas, while a per-party RAG pipeline keeps each agent bounded to its official manifesto. We operationalize the framework on the 2019 Flemish election, deploying the partisan agents in a hub-and-spoke negotiation arbitrated by a formateur. To make the emergent negotiation interpretable, we introduce a Multi-Layered Information Lineage Topology (MILT) that traces every clause in the final agreement back to its manifesto origin and classifies it into five provenance states, a Coalition Influence Score (CIS) that aggregates these traceable contributions to identify which party shaped the agreement, and a real-world grounding pass that benchmarks each simulated provision against the historically adopted coalition agreement. Across three independent simulations the framework yields a stable winner and ranking (N-VA ahead of CD\&V and Open Vld), and manifesto-anchored lineage reliably predicts real-world materialization whereas hallucinated content does not. The result is a transparent, scalable testbed for the ex-ante exploration of party compatibility and formateur-mediated compromise. 

---
# Rubrics on Trial: Evolving Rubrics from a Single Query via Synthetic Pairwise Evidence 

**Authors**: Haocheng Yang, Licheng Pan, Xiaoxi Li, Zhichao Chen, Zhiheng Zhang, Yuan Lu, Haoxuan Li, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.15092)  

**Abstract**: Rubrics provide structured, fine-grained signals for training and evaluating large language models (LLMs). Yet reliable query-specific rubrics are difficult to construct. Existing approaches often derive supervision from human-written rubrics, preference data, or sampled responses. Direct query-to-rubric generation avoids these resources, but provides no explicit check that a plausible rubric is useful. Such a rubric may fail to distinguish answer quality, reward an optional style, or penalize a valid alternative strategy. We introduce Rubrics on Trial, a query-only framework that evolves a rubric set from an empty set without external annotations or model training. It derives supervision solely from synthetic rubric-conditioned response pairs and validates each proposed rubric before adding it, screening out non-discriminative, over-specific, and style-only candidate rubrics. Experiments across five preference benchmark suites demonstrate the effectiveness of Rubrics on Trial, which achieves the best average accuracy and leads on six of seven evaluation sets. 

---
# OmniaBench: Benchmarking General AI Agents Across Diverse Scenarios 

**Authors**: Chengyu Shen, Yujie Fu, Gangtao Xin, Yanheng Hou, Wenlong Fei, Guojie Zhu, Jiawei Li, Hongcheng Gao, Runming He, Zhen Hao Wong, Meiyi Qiang, Hao Liang, Zhao Cao, Hao Jiang, Chong Chen, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14989)  

**Abstract**: Large language models are increasingly evolving from text generators into general agents capable of understanding user requests, invoking external tools, and completing complex tasks through interaction. However, existing agent benchmarks often focus on limited scenarios, tool ecosystems, or interaction formats, making it difficult to systematically characterize model capabilities across heterogeneous application settings. We introduce OmniaBench, a benchmark for evaluating general agents across diverse scenarios with explicit state spaces. We derive application-oriented scenario knowledge from app stores, product documents, industry resources, Web retrieval, and human refinement, forming a hierarchical taxonomy that spans ToC, ToB and ToE with 90 level-1 and 354 level-2 domains. Based on this taxonomy, we construct executable environments and synthesize single-turn and multi-turn tasks through four complementary routes: DAG, DAG-S, Solver, and Program. OmniaBench further introduces a ten-dimensional capability taxonomy and eight compositional atomic difficulty factors to support fine-grained evaluation and analysis. The resulting dataset contains 1,431 tasks, together with a challenging subset of 644 tasks designed to reduce evaluation cost and mitigate potential contamination of the full set after public release. The bench presents substantial challenges to current frontier models, with even Claude-Sonnet-5 and GPT-5.6-Sol achieving Overall Pass@1 scores of only 58.54 and 57.14, respectively. Further analyses reveal clear differences across domains and capabilities, as well as persistent limitations in planning, constraint maintenance, and adaptive correction. OmniaBench provides a broad and diagnostic benchmark for characterizing the capability boundaries of general agents. 

---
# Latent Trajectory Discrimination for AI-Generated Text Detection 

**Authors**: Gianluca Bonifazi, Christopher Buratti, Michele Marchetti, Federica Parlapiano, Giulia Quaglieri, Davide Traini, Domenico Ursino, Luca Virgili  

**Link**: [PDF](https://arxiv.org/pdf/2607.14967)  

**Abstract**: Most existing approaches to AI-Generated Text Detection (AIGTD) treat documents as static objects and base their decisions on aggregate statistics or globally compressed embeddings. However, this perspective overlooks the inherently dynamic nature of autoregressive generation, where content evolves progressively through the latent space. In this paper, we reformulate AIGTD as the problem of distinguishing between latent generation trajectories. Instead of relying on static representations, we model how textual representations evolve across the sequence. To this end, we propose Geometric Trajectory and Contrastive Learning (GTCL), a framework that segments the document into ordered local units, encodes each unit in an embedding space, and constructs a structured and sequence-level representation. GTCL then applies contrastive learning to these trajectories to learn geometric regularities associated with the autoregressive generation. Evaluations performed on three different benchmarks and several approaches show that GTCL outperforms detection baselines consistently, which implies that explicitly modeling sequential dynamics provides robust discriminative signals across models and domains. These results suggest that modeling trajectory differences could improve detection and open up a dynamic direction that has been underexplored in previous AIGTD literature. 

---
# Show Me How You Reason and I'll Tell You Who You Are: Reasoning Graphs for Robust LLM Authorship Attribution 

**Authors**: Zlata Kikteva, Artur Romazanov, Annette Hautli-Janisz, Ramon Ruiz-Dolz  

**Link**: [PDF](https://arxiv.org/pdf/2607.14905)  

**Abstract**: Given the current trend to employ large language models (LLMs) in almost any imaginable context, LLM-generated text detection and authorship attribution have become a pressing issue. Prior work has primarily focused on surface-level linguistic features, an approach shown to be susceptible to paraphrasing and other obfuscation techniques. In this paper, we go beyond the linguistic surface, extracting and analysing reasoning structures in LLM-generated texts with the goal of capturing more complex signals of LLM authorship. We propose a graph neural network approach that leverages reasoning graphs extracted by an argument mining pipeline, demonstrating improved robustness and generalisation over a traditional Longformer baseline. Our approach outperforms the baseline by up to 27 percentage points under the obfuscation attacks such as paraphrasing and backtranslation, and 19 percentage points when evaluated on the texts generated by the unseen model versions, simulating real-world conditions in which new LLM versions are continuously released. 

---
# SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning 

**Authors**: Jinyang Wu, Shuo Yang, Zhengxi Lu, Fan Zhang, Yuhao Shen, Lang Feng, Haoran Luo, Zheng Lian, Shuai Zhang, Zhengqi Wen, Jianhua Tao  

**Link**: [PDF](https://arxiv.org/pdf/2607.14777)  

**Abstract**: Large language models are increasingly trained as interactive agents for long-horizon tasks involving multi-turn interaction, tool use, and environment feedback. Outcome-based reinforcement learning (RL) provides a practical optimization paradigm, but its sparse trajectory-level rewards offer limited guidance on intermediate decisions, leaving a supervision gap between episode-level outcomes and token-level policy learning. We propose SEED (SElf-Evolving On-Policy Distillation), a self-evolving framework that converts completed on-policy trajectories into training-time hindsight skills and distills their behavioral effect back into the policy model. SEED first fine-tunes the policy to analyze completed trajectories and generate natural-language skills that capture reusable workflows, decisive observations, or failure-avoidance rules. During RL, the current policy both collects trajectories and serves as the analyzer that extracts hindsight skills from them. Policy updates therefore improve subsequent decision making and skill analysis together, allowing hindsight supervision to evolve with the policy. SEED then re-scores the sampled actions under ordinary and skill-augmented contexts, converting the skill-induced probability shift into a dense token-level on-policy distillation signal. This signal is jointly optimized with outcome-based RL, keeping the auxiliary supervision aligned with the current trajectory distribution. Extensive experiments on text-based and vision-based agentic tasks show that SEED consistently improves performance and sample efficiency, exhibiting robust generalization to unseen scenarios. Our code is available at this https URL. 

---
# Dialogue Summarization with Emotion Dynamics Using Topic- and Participant-Centric Decomposition 

**Authors**: Linyun Xiang, Mark Neerincx, Stephanie Tan  

**Link**: [PDF](https://arxiv.org/pdf/2607.14769)  

**Abstract**: Existing text summarization research has focused much on monologic information (e.g., newspaper articles, reports) without accounting for the interaction between speakers or authors. In contrast, dialogues are a rich communication channel where multiple participants conduct back and forth exchanges to construct meaning. We propose a dialogue summarization framework that explicitly models both semantic and emotion dynamics using multimodal dialogue inputs, built on an adapted hierarchical Chain-of-Agents approach. We decompose dialogues from two perspectives: (1) topic segments based on the utterances of all participants, and (2) participant-specific utterance segments. These are used to generate corresponding summaries while incorporating automatically inferred emotions. Topic- and participant-level summaries are aggregated into a dialogue summary capturing semantic content and emotion trajectories. To evaluate beyond content accuracy, we introduce emotion trajectory metrics measuring how well summaries preserve emotional flow. Experiments with small language models on multimodal dialogue datasets show that our framework produces summaries with both semantic and emotion content. Further experiments on explicit emotion label availability highlight the efficacy of our proposed methodology and the opportunities in dialogue analysis using language models. 

---
# CoTu at EXACT 2026: Neuro-Symbolic Reasoning for Transparent Educational QA 

**Authors**: Quoc-Khang Tran, Minh-Thien Nguyen, Phu-An Thai, Xuan-Tung Bui, Truong-Thanh Ma, Nguyen-Khang Pham  

**Link**: [PDF](https://arxiv.org/pdf/2607.14735)  

**Abstract**: Transparent educational question answering asks for answers that are not only correct but explainable, and doing so with small models rules out the reasoning power of the largest proprietary systems. The EXACT 2026 competition poses this problem concretely: open-weight language models of at most 8B parameters, self-hosted, with a natural-language explanation for every answer. It pairs two tasks: logical reasoning over university regulations, and multi-step physics problem solving. We describe the system that team \cotu{} developed to address both, a neuro-symbolic Program-of-Thought pipeline in which a 4B backbone writes a program rather than stating an answer directly: for regulation queries it emits a Z3 encoding whose entailment verdict grounds the deduction, and for physics it emits numerical Python, both wrapped in a shared self-correction loop and a unified explained-JSON output. Answer-type routing, distillation-based task fine-tuning, and a latency-aware serving stack -- SGLang with speculative decoding -- keep the system within the 60-second per-query limit. The system achieved a \textbf{perfect score} on the physics task in both automated selection rounds and obtained the \textbf{highest final-round technical score} of any team -- $13.44/15$, combining automated answer evaluation with expert-judged reasoning depth -- with the equally weighted presentation score included, \cotu{} placed 3rd overall. Grounding answers in a symbolic solver yields correct, verifiable deductions at the 4B scale, and the residual difficulty lies in premise selection rather than the deduction itself. 

---
# The Misclassification of Autistic Writing as AI-Generated 

**Authors**: Summer Chambers, Matthew C. Kelley  

**Link**: [PDF](https://arxiv.org/pdf/2607.14729)  

**Abstract**: Recent findings suggest that detection models for artificial intelligence (AI) cannot accurately identify AI-generated text and may exhibit bias against certain minority groups. In the present study, anecdotal claims that autistic writers more often have their work flagged as AI-generated are examined empirically. A corpus of approximately 60,000 Reddit posts split into "likely-autistic" and "general-Reddit" subcorpora is used to compare the distribution of probabilities output by the OpenAI GPT-2 detection model. Differences in textual features between subcorpora are observed and compared to reported features of AI-generated text. Results showed that while less than two-percent of either subcorpus was flagged as AI-generated by the model, significantly more texts from the likely-autistic subcorpus were flagged. Connections between features of text with likely-autistic authors and AI-generated text were not straightforward. The widespread use of AI-detection models with a potential bias against autistic writers in their output prompts ethical scrutiny, and the authors recommend further critical examination of the models themselves as well as their use in academic contexts. 

---
# Gold-Guided Programmatic Distillation for Financial Reasoning over Hybrid Tables and Text 

**Authors**: Yun Dong, Erica Zhao, Elana Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.14709)  

**Abstract**: Financial question answering over hybrid tabular and textual data may require multi-source reasoning and precise numerical computation. While large language models (LLMs) can generate intermediate reasoning steps, natural-language rationales remain prone to arithmetic errors, making them an unreliable supervision source for distillation. Building on programmatic distillation, we develop an approach that transfers reliable numerical reasoning from a large teacher model to a compact student using execution-verified Python programs instead of free-form textual rationales. It leverages gold derivations to guide teacher-side program synthesis and retains only programs that execute correctly and produce the gold answer, ensuring high-quality supervision. We further introduce an iterative recovery stage that revisits teacher-failed examples, enabling the student to recover and incorporate newly verified programs into training. Experiments on TAT-QA show that our framework is highly effective for hybrid financial reasoning. Our best 7B student achieves 87.00 EM / 87.18 F1 on the test set, substantially outperforming the 72B teacher (78.46 EM) as well as traditional and strong LLM-based baselines, including TAGOP and TAT-LLM. These results demonstrate that execution-verified programmatic distillation provides an effective and extensible framework for training smaller models to perform reliable numerical reasoning. 

---
# Harnessing LLMs for Reliable Academic Supervision: A Comparative Study 

**Authors**: Akash Raj  

**Link**: [PDF](https://arxiv.org/pdf/2607.14707)  

**Abstract**: Large language models routinely produce fluent answers to single-shot prompts, yet deploying them as reliable components of a domain decision system is substantially harder. Closing this gap is the work of harness engineering: the deliberate composition of deterministic scaffolding (symbolic filters, retrieval, schema-typed I/O, LLM-as-judge loops, HITL gates, persistent state, audit trails) around an LLM core. We present a case study in academic supervision, a domain combining high-stakes recommendation, longitudinal accountability, and structured operational workflows.
We compare a baseline (ASA), a GPT-5 chatbot with no scaffolding, against a multi-module system (ASuS) that wraps the much smaller GPT-4o-mini in a LangGraph harness with symbolic-semantic retrieval, schema-validated outputs, LLM-as-judge with bounded retry, HITL gates, deterministic weighted risk scoring with LLM narration, and a per-node SQLite audit trail. The evaluation rubric is retargeted at six harness-mechanism dimensions (grounding, explainability, consistency, process integrity, cognitive load, constraint adherence). A blind ten-rater hybrid evaluation, supplemented by a 2 x 2 model-harness ablation, finds that ASuS, despite using a much smaller base model, outscores ASA on every dimension. Across ten raters the pooled mean for ASuS is 4.08 versus 1.23 for ASA, and 8 of 10 raters reject the null at alpha = 0.05 on a paired Wilcoxon test; full numbers are in Sections 6.4 and 6.7. The ablation confirms that the structural contributions of the harness are largely model-invariant. We extract seven recurring harness-engineering patterns and argue that where reliability, traceability, and institutional consistency matter more than open-ended fluency, harness engineering challenges the prevailing 'bigger model is better' intuition. 

---
# D-cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding 

**Authors**: Tianyu Liu, Yuhao Shen, Rui Cen, Junhan Shi, Jiebin Zhang, Guangshuo Qin, Hong Liu, Song Liu, Guanghua Yu, Jianchen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.14647)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference without compromising output quality. Recent parallel drafting methods further improve single-request performance by decoupling draft length from drafting latency, enabling longer drafts and higher mean accepted tokens (MAT). However, under high request concurrency, long drafts waste substantial computation on rejected tokens, increasing verification cost and potentially making speculative decoding slower than autoregressive decoding. We present D-Cut, an adaptive pruning method that selects draft tokens jointly across the batch and concentrates the verification budget on tokens most likely to be accepted. D-Cut is motivated by two observations. First, acceptance lengths vary considerably across concurrent requests; D-Cut therefore performs cross-request pruning, allocating the verification budget adaptively according to draft confidence. Second, verification cost depends strongly on the deployment environment, including GPU architecture and parallelism strategy; D-Cut incorporates a runtime cost model to adapt its pruning depth to the target environment. Experiments on dense and mixture-of-experts (MoE) models show that, under high concurrency, D-Cut improves the average speedup from \(1.26\times\) to \(1.65\times\), restores acceleration in dense-model configurations where long-draft baselines are slower than autoregressive decoding, and achieves up to \(3.0\times\) speedup over autoregressive decoding on MoE models. 

---
# Routing Ceilings Are Domain-Independent: Structural Prior Injection in Code Security Vulnerability Detection 

**Authors**: Manuel Israel Cázares  

**Link**: [PDF](https://arxiv.org/pdf/2607.14628)  

**Abstract**: Large language models (LLMs) exhibit a well-documented gap between latent capability and consistent activation: the router hypothesis posits that models possess the knowledge to solve a task but lack reliable internal routing to activate it. Prior work in formal mathematical reasoning (SAIR, Cázares 2026) reports that structural priors (cheatsheets) raise in-distribution performance dramatically, yet collapse below the zero-shot baseline out-of-distribution (OOD) -- and that iterative recalibration amplifies rather than corrects the collapse.
We test whether this phenomenon is cross-domain by reproducing the SAIR design in source-code security vulnerability detection, evaluating three LLMs (GPT-OSS-120B, Llama-3.3-70B, Gemma-4-31B) across three vulnerability categories (CWE-798, CWE-284, and the non-CWE N+1 anti-pattern) spanning syntactic, contextual, and semantic complexity, then transferring cheatsheet-augmented prompts to real-world CVE data from VUDENC (CWE-89, CWE-22).
Our findings replicate and extend SAIR: (F1) structural priors lift semantic-vulnerability recall from 20.0% to 100.0% across all models; (F2) zero-shot performance degrades along a semantic complexity gradient; (F3) the same cheatsheets that saturate synthetic performance amplify distribution-shift collapse on real CVE data (CWE-89: 100% synthetic F1 to 48.9% on VUDENC, -51.1pp); (F5) iterative recalibration produces a v2 cheatsheet that performs worse than v1 on real data, mirroring SAIR's AN45c-vs-AN38 finding.
These results provide evidence that the cross-distribution trade-off surface documented in SAIR generalises to code security, and that the router hypothesis is cross-domain. We argue the structural nature of the collapse motivates distribution-aware training over prompt calibration. Code and evaluation scripts: this https URL 

---
# Investigating first-language bias in LLM-based automated essay scoring: A cross-prompt evaluation of an open-weight AI-model on TOEFL essays 

**Authors**: John Maurice Gayed  

**Link**: [PDF](https://arxiv.org/pdf/2607.14605)  

**Abstract**: This study examines the cross-prompt generalization and first-language (L1) scoring effects of a LoRA-adapted open-weight large language model (Gemma-3-27B-it) applied to automated essay scoring. Using the identical model and inference configuration reported in "AiAWE: An Open-Source LLM Automated Writing Evaluation System Using LoRA-Adapted Instruction-Tuned Models" (Gayed, 2026), which was fine-tuned on 480 argumentative essays from two prompts, we evaluate scoring accuracy on the full TOEFL11 corpus: 12,100 essays written by test-takers from 11 first-language backgrounds across eight prompts, none of which were seen during training. The model's raw scores (0.5-5.0) are mapped to the same three proficiency bands (low, medium, high) used by ETS, enabling direct comparison. The model achieved an overall band agreement of 77.79% and a quadratic weighted kappa of 0.702, with adjacent-band agreement of 99.98%. Accuracy was stable across all eight unseen prompts, with no advantage for prompts thematically related to the training data, indicating robust cross-prompt generalization. However, the model exhibited a systematic, L1-linked scoring offset. Within every proficiency band, essays from European-language backgrounds received consistently higher scores than essays from East-Asian-language backgrounds, a pattern not attributable to the composition of the fine-tuning data. This is the first large-scale L1 fairness analysis of a fine-tuned open-weight LLM for automated essay scoring. 

---
# How Well Does AI-Generated Feedback Work? Intrinsic and Extrinsic Evaluation across more than 20,000 EFL Essay Drafts 

**Authors**: Steven Coyne, Diana Galvan-Sosa, Ryan Spring, Machi Shimmei, Michael Zock, Keisuke Sakaguchi, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2607.14591)  

**Abstract**: This study examines feedback in English as a Foreign Language (EFL) writing contexts, focusing on written corrective feedback (WCF). Large language models (LLMs) can provide WCF at scale, but aligning them with pedagogical best practices remains an ongoing challenge. WCF meeting criteria like factuality or relevance may still be unsuitable for learning contexts, highlighting the need for extrinsic evaluation based on the learner's perspective. We deployed WCF systems in a university-level EFL class with nearly 2,000 students, collecting over 20,000 drafts. We evaluated the generated WCF from two perspectives: intrinsic evaluation by experienced English teachers using a rubric, and extrinsic evaluation via student feedback and engagement metrics. Results revealed low alignment between teacher expert ratings and student feedback. These findings suggest that traditional expert evaluation alone may not fully capture WCF's usability or helpfulness from the learner's perspective, highlighting the importance of learner-centered evaluation frameworks for AI-based applications in language education. 

---
# MARS: Multi-hop Adaptive Retrieval and SPARQL Generation for KGQA 

**Authors**: Nikit Srivastava, Daniel Vollmers, René Speck, Nikolaos Karalis, Hamada M. Zahera, Axel-Cyrille Ngonga Ngomo  

**Link**: [PDF](https://arxiv.org/pdf/2607.14561)  

**Abstract**: Large language models (LLMs) have demonstrated strong reasoning performance, but their tendency to hallucinate limits their reliability in knowledge-intensive tasks requiring up-to-date and grounded information. Combining knowledge graphs (KGs) with LLMs facilitates the use of explicit symbolic knowledge that can be continuously updated without costly fine-tuning, while benefiting from rapidly advancing LLM reasoning. We propose MARS, a scalable knowledge graph question answering (KGQA) approach that requires no model fine-tuning. Rather than relying on open-ended agentic exploration, MARS performs a structured retrieval procedure that links question entities to the KG and iteratively retrieves relevant next-hop information. At each step, MARS decides whether to continue graph traversal or to generate the final SPARQL query, allowing the model to adapt the retrieval depth to the question while keeping the overall pipeline more predictable than fully agentic approaches. We evaluate MARS on three established KGQA benchmarks across several LLMs and settings, including multilingual evaluation, and provide insights through ablation studies and error analysis. Our approach achieves competitive performance relative to state-of-the-art methods while remaining efficient and scalable. The evaluation results, code and resources are publicly available: this https URL. 

---
# Answer-Conditioned Chains of Thought Degrade Verifiable-Reasoning Distillation in Large Language Models 

**Authors**: Jungseob Lee, Seungyoon Lee, Suhyune Son, Dongyub Jude Lee, Sungbin Han, Sugyeong Eo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2607.14552)  

**Abstract**: A standard recipe for distilling the reasoning ability of large language models (LLMs) is to sample chains of thought from the model, keep those that reach the correct final answer, and fine-tune on the survivors. When sampling fails, a common fix shows the generator the gold answer and asks it to write a chain that reaches that answer. We show that this second step degrades the training data in a way that correctness filtering cannot catch. We run a controlled experiment that fixes the generator, the problem set, and the correctness filter, and varies only whether the chain is generated under answer-conditioning, the gold answer shown with a request to reach it. Training a strong instruction-tuned reasoning model on its own answer-conditioned chains sharply lowers its verifiable-reasoning accuracy. The loss grows with difficulty, reaching as much as about 27 points on the hardest competition problems. The mechanism is legible in the chains themselves, which rationalize backward from the shown answer instead of deriving it, with the early final-answer statement as the measurable symptom. The harm is a property of the data rather than the generator, read off unlabeled generations before any fine-tuning, ordering the penalty across eight thinking models from four families, and transferring across teacher families. A prompt ablation localizes it to the rationalize-toward instruction rather than the answer's bare visibility. The practical takeaway is to generate answer-blind, because no correctness filter can see this damage in the data. 

---
# CityLLM: A framework for natural-language querying of semantic 3D city models 

**Authors**: Rabindra Lamsal, Sisi Zlatanova, Johnson Xuesong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2607.14542)  

**Abstract**: Semantic 3D city models provide rich geometric and semantic information, but remain challenging for non-experts and interdisciplinary researchers to access and query due to their complex structures and specialized data formats. To address this issue, we present CityLLM, a framework for natural-language querying of semantic 3D city models alongside complementary urban datasets. The framework combines spatial and graph databases within an LLM-based workflow that supports iterative query refinement and cross-database chaining. We evaluate CityLLM on a CityJSON dataset of Rotterdam (853 LoD2 buildings) using GPT-OSS, Gemini 3.1, and GPT-5.4, along with selected variants, across multiple metrics: answer correctness, visualization correctness, query success, and retry attempts. A total of 54 natural-language queries are curated across four scenarios: spatial, graph, cross-database, and conversational. Results show strong overall performance, with answer correctness ranging from 85.2% to 100%, visualization correctness from 92.9% to 100%, a 100% query success rate, and fewer than three retries across all 54 queries. Overall, the findings suggest that CityLLM provides a lightweight and extensible approach for conversational access to semantic 3D city data. 

---
# Controlled Reformulation Testing for Logical Consistency in Large Language Models 

**Authors**: Alexander Gu, Alan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.14528)  

**Abstract**: Large language models (LLMs) frequently contradict themselves when the surface form of a logically equivalent question changes. We present a benchmark of 350 question families (1,750 total questions) for Controlled Reformulation Testing (CRTBench) to evaluate logical invariance. In this benchmark, we investigate LLMs' ability to maintain consistent answers across controlled reformulations, which include contrapositive rewriting, double negation, negation flipping, and passive voice. We evaluate several frontier LLMs and observe an accuracy-consistency gap where GPT-5.4-mini achieves $98.9\%$ base accuracy but only $60.3\%$ family-level consistency, while reasoning-optimized o4-mini achieves $96.9\%$ consistency. From our experiments, we observe that failures cluster around logically nontrivial transformations such as contrapositive rewriting ($72.4\%$ for GPT-5.4-mini) and double negation ($84.6\%$), while surface-level rephrasing remains robust ($94-100\%$). Increasing reasoning effort improves GPT-5.4-mini to $85.4\%$ consistency, but leaves GPT-5.4 unchanged overall because gains on nested negation are offset by failures on quantifier families. These results show that accuracy alone is not enough for evaluating logical reasoning in LLMs. 

---
# LLM Evaluators are Biased across Languages 

**Authors**: Ej Zhou, Lucas Resck, Zheng Hui, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2607.14480)  

**Abstract**: LLM evaluators (trained reward models and prompted LLM-as-a-Judge) are routinely validated via pairwise accuracy. In a multilingual setting, this operates under the premise that high pairwise accuracy implies reliable, language-neutral scoring. We show that this assumption does not hold. We conduct experiments with semantically identical instruction-response pairs across 23 languages, and find that multilingual evaluators assign significantly different scores to different evaluation languages. The bias is statistically significant and consistent across eight open-weight evaluators of different architectures and training paradigms, persists in frontier judges, and is strongly correlated with language resource level: lower-resource languages are scored more generously. Meanwhile, these biases are invisible to pairwise accuracy: evaluators achieve above 90% pairwise accuracy, yet have up to 43% difference in acceptance rate across languages under a global decision threshold, meaning, for instance, that harmful content in lower-resource languages is more likely to pass safety filters. Per-language thresholds would require language identification, which can be defeated by code-switched prompts. We then investigate why lower-resource languages receive higher rather than lower scores, and we find that model uncertainty is linked with the effect: models tend to give higher scores when less confident, both under negative log-likelihood and under token-free uncertainty measures; however, language identity remains a significant predictor after controlling for uncertainty, and the bias cannot be explained away by content difficulty alone, but is a structural, language-level misalignment. 

---
# Smarter and Cheaper at Once: Byte-Exact KV-Cache Grafting Turns a Frozen Small Model into a Verified-Knowledge Flywheel 

**Authors**: Sietse Schelpe  

**Link**: [PDF](https://arxiv.org/pdf/2607.14431)  

**Abstract**: We report a way to make a frozen small language model both more capable and dramatically cheaper at once, without changing any weights. Verified knowledge is deposited once as a byte-exact key-value (KV) state artifact and later restored, by graft, into a fresh inference context. The restore is bit-exact: under a pinned deterministic configuration, the grafted logits are byte-for-byte identical to a fresh computation (SHA-256 equality), with zero KL divergence and 100% argmax agreement over fifty samples. We show that own-position graft is the unique numerically exact operating point on a model with floating-point rotary encoding, and we verify byte-exactness on two model scales (12B, 31B) and two GPU targets, one through a pre-registered replay. On AIME 2025, a frozen Gemma-4-12B moves from 80.0% to 93.3% once a verified solution library is grafted, above its own 77.5% and its 31B sibling's 89.2% published anchors. On the recurring case, eight problems the base model never solves within a 401,026-token budget are answered from cached verified solutions in 61 total decode tokens, a factor of 6,574 fewer tokens and about 8,700x less energy; the capability claim proper rests on held-out transfer (7 of 7 at 31B). The same byte-exact store widens usable context from 32,768 to 2,854,766 tokens at zero extra accelerator memory, and moves byte-identical between machines of the same architecture. We describe the system at the behavior level; the engine is proprietary, and every reported number is backed by committed input and output hashes so the scoring can be re-checked without it. 

---
# DS@GT ARC at LongEval: Citation Integrity and Factual Grounding in Scientific QA 

**Authors**: Brandon Michaels, Brendon Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2607.14400)  

**Abstract**: This paper describes DS@GT ARC's submission to the CLEF 2026 LongEval Task 4 on Retrieval-Augmented Generation (RAG). In this submission, we examine a divergence between traditional natural language evaluation metrics and citation integrity as applied to RAG QA systems. We evaluate a corrective pipeline using Corrective RAG (CRAG) and CiteFix against baseline and frontier model benchmark RAG QA scores. While frontier models maximized answer relevance and fluency scores, our RAGAs LLM-as-judge diagnostics indicate that frontier models would correctly identify relevant documents without using their context in answer generation. Conversely, by filtering chunks pre-generation and enforcing strict entailment of generated claims to the cited material post-generation, our corrective pipeline marginally improved citation faithfulness and answer grounding. We propose that evaluation of trustworthy RAG QA requires metrics that reward strict answer grounding. 

---
# MamaBench: Benchmarking LLM Robustness in Maternal and Child Health Diagnosis through Counterfactual Clinical Perturbation 

**Authors**: Thanni Adewuyi, Anuoluwa Sotome, Samuel Okoko, Angel Ezendu, Oluwafunke Akinbuwa, Oluwaseun Odunsi, Oluwasegun Oguntuase, Oluwadarasimi Oguntuase, Ifeoma Nwabueze, Abiodun Adereni  

**Link**: [PDF](https://arxiv.org/pdf/2607.14385)  

**Abstract**: Large language models achieve strong scores on medical benchmarks, yet these benchmarks evaluate each question in isolation, providing no measure of whether a system can distinguish clinically similar presentations requiring different interventions. We introduce MamaBench, the first counterfactual benchmark for maternal and paediatric AI: 434 expert-authored clinical narratives in 217 pairs across 371 pathologies, evaluated via the Bias Trap Rate (BTR), the conditional probability that a model fails the counterfactual given success on the base case. We propose Evidence-Anchored RAG (EA-RAG), a three-stage retrieval method that replaces aggregate similarity with an evidence coverage objective through clinical parameter extraction, coverage auditing, and contrastive sub-queries. Across eight configurations of four frontier LLMs, base accuracy overstates robust accuracy by 16-28 percentage points in every model. EA-RAG achieves 20.3% BTR and 65.0% robust accuracy on Claude Sonnet 4.6, a 5.5 percentage point BTR reduction without degrading base accuracy. The residual 20% BTR confirms that counterfactual robustness in clinical AI remains an open challenge. Keywords: counterfactual evaluation, clinical AI, maternal healthcare, retrieval-augmented generation, diagnostic robustness 

---
# HABIB_TAZ at SemEval-2026 Task 11: Disentangling Formal Logic from Content via Synthetic Training and Multi-Objective Optimization 

**Authors**: Abdullah Shaikh, Zain Naqi, Taha Zahid, Sandesh Kumar, Abdul Samad  

**Link**: [PDF](https://arxiv.org/pdf/2607.14349)  

**Abstract**: While Large Language Models (LLMs) excel in many general NLP tasks, their formal reasoning capabilities are often compromised by content effects, demonstrating a measurable bias towards real-world plausibility. In this paper, we present our system for SemEval-2026 Task 11, which evaluates the ability of models to disentangle formal logic from content across 12 languages with and without distractor premises. We address this challenge using mDeBERTa-v3 networks fine-tuned on a synthetic, rule-based dataset of syllogistic schemes to avoid the semantic noise of LLM-augmented data. To explicitly decouple plausibility from logical structure, our training pipeline employs a multi-objective loss function combining Adaptive Group Distributionally Robust Optimization (DRO), a scheduled differentiable bias penalty, and KL-Divergence consistency regularization. Our system achieved #1 ranks and perfect Ranking Scores (100.0) with 0.00% bias and 100.0% accuracy on Subtask 1 (English), Subtask 2 (Noisy English), and Subtask 3 (Multilingual). On the highly complex Subtask 4 (Noisy Multilingual), the system achieved the 6th rank with 89.06% Accuracy and F1-score, alongside a limited 2.89% Bias and a 37.78 Ranking Score. Our dataset generation engine and codebase are publicly available to facilitate future work on robust logical reasoning. 

---
# PReM: Learning What to Preserve and When to Refresh for Context Compression 

**Authors**: Bohan Yu, Lei Shen, Chenxi Zhou, Chen Han, Junlin Liu, Wenbo Su, Yu Cheng, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.14327)  

**Abstract**: Efficient long-context inference is not only about reducing memory cost, but also about keeping useful contextual evidence accessible as generation proceeds. However, existing compression-oriented approaches, such as key-value (KV) cache compression and context compression, often either make an early decision about which contextual information to keep or rely on an external compressor. Such designs make it difficult to adapt the compressed context to the evidence needed by later reasoning steps. This paper introduces PReM (Preserve and Refresh Memory), a context-compression framework that maintains the long context as the model's internal layer-wise KV memory and learns what to preserve and when to refresh it. Specifically, PReM uses a dedicated memory layer to make memory-selection decisions, and a special memory token <m> to trigger refreshes during generation. To train this behavior, PReM introduces Phase-Separated Refresh Training, aligning memory selection with memory-conditioned generation while preserving continuity across refreshes. Experiments with 32K-token contexts show that PReM outperforms strong baselines under both 16x and 32x compression, while maintaining a favorable balance between answer quality and inference efficiency. 

---
# Multi-Head Latent Control: A Unified Interface for LLM Agent Decision Making 

**Authors**: Amirhosein Ghasemabadi, Ruichen Chen, Bahador Rashidi, Di Niu  

**Link**: [PDF](https://arxiv.org/pdf/2607.14277)  

**Abstract**: Large language models are increasingly deployed as agents, but reliable agentic behavior requires more than next-token prediction. At inference time, it is preferred that an agent can decide whether to proceed with its current reasoning, defer to a stronger model, request additional information, invoke external tools, or abstain under the given setup. Existing approaches address these decisions through prompt-level routing, external orchestration, or task-specific fine-tuning, which primarily rely on input-side signals, and are often costly and difficult to maintain as model backbones evolve. We ask whether such control decisions can be inferred directly from a model's latent generation process. We introduce Multi-Head Latent Control, a lightweight layer that reads hidden-state trajectories from a frozen LLM or VLM to produce deployment-time control signals. A Capability Head predicts whether the current model can solve the instance or should defer to a stronger collaborator, while a Resolution Head predicts appropriate resolution decision Clarification, Tool Use, Abstention, or Direct Answering. Both heads are trained only on latent traces from the same frozen LLM backbone, enabling post hoc adaptation without modifying the model. Across language and vision-language settings, Multi-Head Latent Control consistently improves the quality-cost tradeoff of multi-model systems, enabling early handoff from partial generations and more accurate intervention decisions. In routed execution (small + large model), it reduces large-model usage by up to 90.7 percent on AndroidWorld and 27-53 percent on average across benchmarks, while retaining most of large-model performance. Additionally, the learned control signals improve tool-use decision quality, yielding up to +158 percent relative score gain and 65.5 percent fewer missed-required tool calls. 

---
# The Severance Problem: LLMs are Unaware of the Person Beyond the Prompt 

**Authors**: Dor Litvak, Liu Leqi  

**Link**: [PDF](https://arxiv.org/pdf/2607.14250)  

**Abstract**: Personal AI assistants have attracted significant interest for their potential to enhance everyday life by automating routine tasks, supporting consequential decisions, and assisting with everyday personal matters. Yet despite rapid recent technical advances, these assistants continue to exhibit undesirable behaviors, such as sycophancy, overconfidence, and hallucination. We argue that these failures stem from a fundamental limitation: language models lack an explicit representation of the person beyond the context they are given, which we term as the \textbf{Severance Problem}. Even with rich personal context and strong commonsense reasoning capabilities from the backbone model, current AI assistants fail to represent what remains unknown about the user. We propose a simple solution: incorporating structured ignorance into the language model context via the \textbf{Severance Schema}, which explicitly outlines dimensions along which the model lacks knowledge about the user, including physicality, temporality, consequences, continuity, multiplicity, and interiority. Empirically, across five model families, with the Severance Schema, the assistant consistently reduces sycophancy, harmful advice, and hallucination. Notably, models with the schema ask clarifying questions when information about the user is missing, rather than confidently extrapolating from incomplete user information. 

---
# Implicit Reasoning Steering via Concept Chaining 

**Authors**: Xiao Ye, Sanika Chavan, Yuxi Huang, Shahriar Kabir Nahin, Muhao Chen, Anshuman Chhabra, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.14242)  

**Abstract**: Large language models often appear to reason reliably, yet on many questions repeated sampling yields both correct and incorrect answers, revealing an underlying fragility in how final decisions are formed. We study whether this fragility can be exploited through implicit reasoning steering: using natural-language text to bias a model toward a designated answer without explicit instructions, triggers, or direct answer cues. Our approach, Concept Chaining, generates a short connection paragraph that links question entities to a target option through one or two intermediate concepts. We then continue pretraining a victim model on these connection paragraphs and evaluate whether its answer preference shifts on the original multiple-choice questions. Our results show that indirect, natural-looking text can systematically steer model predictions while remaining substantially less inferable than direct paraphrases, which shows that reasoning brittleness is not merely an evaluation artifact: it creates a practical channel through which latent biases can be amplified by ordinary-looking text to covertly redirect model decisions. 

---
# Breaking Refusal in the First Half: A Mechanistic Study of the Prefill Jailbreak 

**Authors**: Alex Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2607.14147)  

**Abstract**: Aligned language models refuse harmful requests, but a one-line prefill ("Sure, here is") strips the refusal. We ask where and how it fails. The harm representation stays intact: on the prompts the attack flips to compliance, a linear probe reads harm as high as on the refused ones (0.91-0.98), while behavioral refusal drops to chance. This holds across four models and three families (1.5-3.8B, and at 14B). Refusal is therefore a shallow, response-site computation. We localize it to an early window: a dose-matched position control shows the first half of the response suffices to break refusal, while the second half is nearly inert. Three causal probes converge on that window. Restoring the harm direction there partially re-engages refusal. Injecting the model's own refuse-state reverses the jailbreak (74%, held-out). And knocking out the early response's attention to the prefill, but not an equal attention mass elsewhere, selectively collapses the harmful continuation. A base-model control identifies the mechanism: the same knockout collapses the continuation prefill-specifically even in a non-safety-tuned base model (64% to 25% harmful content vs a matched control's 64%, replicated at 7B). So the prefill's grip is generic autoregressive conditioning, not safety-specific suppression, and "refusal restoration" is a model-dependent fallback. The dominant mechanism is passive. A small safety-specific attractor remains on top (logit-trace concentration 0.24 vs 0.03), whose active-vs-passive character we size but do not fully separate. No single direction or component is a clean handle either: the decision is decodable but distributed, and refusal tracks harm rather than scary surface. The consequence is structural: a monitor reading the untouched prompt-side representation is immune by construction, but only to response-site attacks. The mechanism is diffuse; the failure surface is local. 

---
# Cross-Dataset Generalization in Urdu Fake News Detection: An Empirical Study with XLM-RoBERTa and a Length Confound Analysis 

**Authors**: Muhammad Abdullah Haroon  

**Link**: [PDF](https://arxiv.org/pdf/2607.14131)  

**Abstract**: Urdu fake news detection remains under-resourced despite Urdu being spoken by over 231 million people worldwide. While prior work has demonstrated strong in-domain performance on individual Urdu datasets, cross-dataset generalisation has received little systematic attention. This paper presents the first cross-dataset generalisation study for Urdu fake news detection, using two publicly available balanced datasets: the Ax-to-Grind Urdu corpus (10,083 articles, 15 domains) and the Notri-Fact Urdu dataset (13,388 articles). We fine-tune xlm-roberta-base under four experimental conditions, in-domain on each dataset and two zero-shot cross-domain transfer directions, comparing against TF-IDF baselines using Logistic Regression and Support Vector Machines. Our experiments reveal a striking asymmetry: Notri-Fact to Ax-to-Grind transfer achieves a macro F1 of 0.771, while the reverse collapses to F1 of 0.005, with the model predicting fake for 99.7% of test articles. We demonstrate that this collapse stems from a systematic length confound in Ax-to-Grind, where fake articles average 117 words versus 35 for real articles, a 3.4x asymmetry inducing shortcut learning. A length ablation capping articles at 50 words yields only a 0.0067 F1 drop, confirming the confound inflates but does not solely drive in-domain performance. We provide a reusable diagnostic methodology that combines bidirectional transfer analysis and prediction-collapse inspection to identify confound-driven behavior in multilingual fake news detection settings. 

---
# Semantic Register Compression in Multi-Agent LLM Cascades 

**Authors**: Manuele Tele Junior Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2607.14119)  

**Abstract**: Multi-agent LLM systems commonly decompose complex tasks into specialized roles. However, this modularity introduces a representational risk: when intermediate agents transform text across linguistic registers, they can systematically compress the semantic distinctions needed for accurate downstream decisions. We term this phenomenon semantic register compression and characterize it as an observable failure mode in multi-agent cascades. Using a three-agent pipeline (Collector-Evaluator-Decider), we quantify compression via inter-label separation in sentence-transformer embedding space. Across political fact-checking (LIAR), sentiment analysis (SST-5), and medical triage (Triagegeist), critical evaluation consistently reduces label separability by 41.7% at the Evaluator stage, while identity passthrough preserves it nearly fully. Five architectural variants causally isolate oriented semantic transformation as the primary driver. A credibility-seeking variant produces minimal geometric compression yet shifts outputs toward mostly-true, demonstrating that transformation valence controls the direction of distributional collapse independently of compression magnitude. Compression generalizes across the three domains with varying intensity: 41.7% in fact-checking, 27.2% in sentiment, and 20.0% in triage. Prompt-level regression explains 78% of the variance, with operational constraints associated with lower compression. These results demonstrate that semantic register compression is a measurable and generalizable phenomenon in multi-agent LLM systems, with implications for safety evaluation in high-stakes domains. 

---
# Budgeted Subset Refinement for Execution-Aware LLM Research Ideation 

**Authors**: Micah Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14118)  

**Abstract**: Large language models (LLMs) can generate research ideas that appear novel to expert reviewers, but recent work also shows that such ideas often lack diversity, are difficult for LLMs to evaluate reliably, and may fail to translate into strong executed projects. This paper evaluates a controlled proxy benchmark for a pre-execution scaffolding problem: given a noisy pool of LLM-generated research ideas, how should a system allocate limited refinement effort to construct a stronger, more diverse, more execution-aware portfolio for human researchers under a fixed rubric? We introduce Budgeted Subset Refinement, a family of strategies that refine only a selected subset of candidates rather than refining all candidates uniformly. In a unified shared-candidate-pool evaluation across 10 random seeds and 10 research-ideation environments, raw generation and reranking alone produce no research-strong nonduplicate ideas under the benchmark rubric, while refinement is necessary for strong proxy-rated portfolios. Uniform refinement produces strong individual ideas but is not the best portfolio-level allocation of compute. Random-k refinement is a strong low-cost baseline, while diversity-aware MMR-k refinement gives the best overall proxy tradeoff: the highest research-strong nonduplicate yield, the lowest duplicate rate among successful methods, and the best cost per research-strong nonduplicate idea. A blinded external-judge robustness check on a balanced 72-item sample supports the broad refinement effect across independent model families, while showing that per-item rankings among refined strategies vary by judge. These results suggest that LLM research ideation systems should be evaluated not only as idea generators, but as budgeted support-allocation systems. The claims are scoped to proxy-rated portfolio quality and do not substitute for expert review or execution-grounded validation. 

---
# Heterogeneous Element-Aware Cross-Version Differencing of Scientific Documents via Layout-Aware Alignment and Structure-Aware Reasoning 

**Authors**: Zhen Yina, Wenkang An, Hao Wang, Keran You  

**Link**: [PDF](https://arxiv.org/pdf/2607.14117)  

**Abstract**: Cross-version differencing of scientific documents is essential in scholarly publishing and technical documentation, but remains challenging because scientific documents are page-structured artifacts containing heterogeneous elements such as text, tables, formulas, figures, and layout cues. Existing text-sequence-based methods often lose layout and structural information, while image-based methods lack semantic interpretability and are sensitive to rendering variation. To address these limitations, this paper proposes a layout-aware heterogeneous element-aware framework for scientific document differencing. The framework decomposes document versions into semantically typed elements, establishes cross-version correspondence through an alignment-first mechanism that jointly models spatial, content, and structural compatibility, and performs type-aware difference reasoning over aligned element pairs. It supports unified change detection, localization, structure-awareness analysis, and alignment/matching evaluation across text, tables, formulas, and figures. Experiments on real-world scientific PDF data from journal production proofreading workflows show that the proposed framework consistently outperforms element-specific baselines. It achieves detection F1 scores of 0.903, 0.855, 0.862, and 0.845 for text, tables, formulas, and figures, respectively, with further improvements in localization, structure awareness, and matching quality. Ablation and sensitivity analyses confirm the effectiveness of cross-version alignment, type-specific representations, structure-aware reasoning, and compatibility-weight design. These results demonstrate that heterogeneous element-aware differencing provides a robust and interpretable solution for scientific document comparison in realistic editorial production scenarios. 

---
# ReportMedSAM: Guiding Segmentation Through Radiology Reports 

**Authors**: Anghong Du, Theodoros N. Arvanitis, Colin Watts, Alejandro F. Frangi, Le Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14116)  

**Abstract**: Free-form radiology reports contain rich clinical descriptions, yet converting them for reliable segmentation remains challenging due to the inherent variability of natural language. Existing pipelines often rely on predefined organ phrases or brittle rule-based inference-time extraction, which limits their scalability to novel anatomical structures and makes them sensitive to linguistic variations. To address this, we propose ReportMedSAM, a report-driven framework that replaces discrete extraction with a learnable concept bank. By leveraging a frozen medical vision-language encoder (BiomedCLIP), we align organ-level concept embeddings with large-scale clinical corpora through contrastive learning, establishing mutually orthogonal semantic anchors. Our approach explicitly mitigates organ-level semantic collapse and ensures high robustness against diverse clinical synonyms (e.g., "renal" vs. "kidney" ). During inference, a clinical report is embedded and matched against this concept bank to dynamically activate task-specific Mixture-of-Experts (MoE) modules. This decoupled design allows new concepts and experts to be added without retraining existing components, providing a parameter-isolated extension mechanism while keeping previously learned experts unchanged. Evaluated on the AbdomenAtlas 3.0 dataset, ReportMedSAM effectively interprets free-form reports, achieves competitive segmentation accuracy, and demonstrates seamless, non-interfering extension to novel clinical tasks. 

---
# CoEvoT: Co-Evolving Chain-of-Thought Prompting for Graph-LLM Reasoning 

**Authors**: Haohua Niu, Xingtong Yu, Yang Liu, Junfeng Fang, Xuanting Xie, Jie Tan, Zhongjian Zhang, Hong Cheng, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14114)  

**Abstract**: Graph learning under distribution shift presents a persistent challenge, where models adapt to new graphs with limited or even no supervision. Recent graph--LLM approaches move toward label-efficient prediction by linearizing graphs into prompts and using large language models (LLMs) as predictors, and can adopt Chain-of-Thought (CoT) prompting to exploit LLM's multi-step reasoning capability. However, existing CoT-based graph--LLM methods generate intermediate thoughts while conditioning on fixed graph tokens, limiting step-wise refinement of structural cues. In this paper, we propose CoEvoT, a simple yet effective co-evolving CoT prompting framework for graph--LLM reasoning. CoEvoT couples text-to-graph token rewriting and graph-to-text reasoning guidance in a closed loop: each intermediate textual thought is used to update the graph token evidence state via a lightweight condition network, and the updated tokens are fed back into the next-step instruction to guide subsequent LLM reasoning. This enables step-wise, state-aware evidence refinement, rather than reasoning over a fixed graph snapshot. Extensive experiments on eight datasets demonstrate that CoEvoT consistently outperforms state-of-the-art baselines. 

---
# T5-CSBoost: Adversarial Perturbation Resistant LLM Fingerprinting 

**Authors**: Gayan K. Kulatilleke, Mahsa Baktashmotlagh, Siamak Layeghy, Marius Portmann  

**Link**: [PDF](https://arxiv.org/pdf/2607.14113)  

**Abstract**: While many AI-generated text (AIGT) detectors achieve strong performance on clean inputs, their accuracy degrades significantly under light paraphrasing, word substitutions, character edits, and distribution shifts. We present T5 Contrastive Style Boosted Classifier (T5-CSBoost), an extension to the T5-Sentinel framework that keeps the original next-token prediction objective for source attribution while introducing an auxiliary margin-based triplet loss over decoder embeddings.
This contrastive style regularization encourages the learning of compact, perturbation-resistant stylistic representations, offering a lightweight yet effective alternative to prior approaches that rely on architectural modifications, adversarial training, or complex multi-task objectives without altering the underlying T5-small backbone. T5-CSBoost achieves state-of-the-art multiclass source attribution and binary human-vs-LLM detection on OpenLLMText and HC3 AIGT benchmarks. More importantly, T5-CSBoost demonstrates enhanced robustness to word and character level adversarial perturbations of up to 90% intensity, achieving state-of-the-art on the challenging MAGE/Deepfake stress-test suite, including unseen models, unseen domains, and extreme paraphrasing scenarios. Our results highlight that explicitly regularizing stylistic embeddings via contrastive learning is a practical and effective strategy for building more robust LLM fingerprinting systems in real-world adversarial settings. 

---
# Information-Theoretic Limits of Reliability and Scaling in Language Models 

**Authors**: Subhabrata Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2607.14112)  

**Abstract**: Large language models (LLMs) are evaluated as though perfect reliability is achievable for any task given sufficient scale. We show this assumption is information-theoretically unjustified. Every generative task has a reliability ceiling that no model can exceed, determined by how much output uncertainty is resolvable from observable context. The gap decomposes into a resolvable component closable with additional context and a subjective component inherent to task ambiguity. Autoregressive generation further degrades this ceiling at a rate governed by the task's dependency kernel, which quantifies inter-token correlations in the output. From these two primitives, we derive a first-principles scaling law where LLM performance is bottlenecked by the scarcer resource: training data or model capacity. This law recovers the Chinchilla scaling law as a special case and provides a structural account of when scaling improves reliability. Beyond scaling, our framework unifies diverse practical phenomena, such as the benefits of retrieval-augmentation and the spectral mechanics of catastrophic forgetting. Our work formalizes the resource-complexity tradeoffs that govern model performance across domains, offering a unified theory of performance limits in generative language models. 

---
# Introspection Fine-Tuning (IFT): Training Small LLMs to Introspect 

**Authors**: Ely Hahami, Ishaan Sinha, Lavik Jain  

**Link**: [PDF](https://arxiv.org/pdf/2607.14111)  

**Abstract**: Can small language models detect and report on perturbations their own internal activations? We investigate this question through the lens of activation steering: injecting concept vectors into a model's residual stream and measuring whether the model can accurately report on the perturbation. We first show that the binary detection paradigm used in prior work -- prompting the model to answer Yes'' or No'' to whether it detects an injected thought -- is confounded in small models, as steering biases the model toward affirmative responses regardless of the question content. We therefore propose two confound-free evaluation paradigms: sentence localization (identifying which of $N$ sentences was perturbed, chance $= 1/N$) and strength comparison (identifying which of two sentences received a stronger injection, chance $= 50\%$). Evaluating across six models from two families (Llama-3.2 and Gemma-4), we find that models as small as 2B parameters introspect reliably well above chance, and that introspective ability generally increases with scale. Llama-1B, however, performs at or below chance. We then introduce \emph{Introspection Fine-Tuning} (IFT): supervised fine-tuning on sentence-localization examples constructed from the model's own perturbed forward passes. IFT raises Llama-1B sentence-localization accuracy from $9.6\%$ to $60.6\%$ (a $6\times$ improvement), with gains generalizing zero-shot to the held-out strength-comparison task ($30.2\% \to 52.2\%$). IFT also improves introspection for 3B and 8B models, while inducing negligible degradation on standard capability benchmarks. Our results suggest that introspective ability is not fixed by scale alone: it can be directly trained, and doing so unlocks latent self-monitoring capacity with implications for AI transparency and alignment. Our code is \href{this https URL}{here}. 

---
# MAPS: Modeling Co-Existing Subjective Perspectives and Shared Meaning in Multi-Agent Cognitive Dialogue 

**Authors**: Molood Arman, Clément Bonnafous  

**Link**: [PDF](https://arxiv.org/pdf/2607.14110)  

**Abstract**: Human dialogue involves more than exchanging information; it also expresses beliefs, emotions, and subjective cognitive styles. Yet current AI dialogue systems often enforce semantic uniformity, sacrificing diversity and interpretability. We present MAPS (Multi-Agent Perspective Spaces), a novel framework that models dialogue between cognitively distinct agents through domain-weighted profiles, dynamic GRU-based memory, and interpretable token-level attention. MAPS enables agents to maintain individualized reasoning while progressively converging on shared meaning. Evaluations on EmpatheticDialogues, TopicalChat, and MultiWOZ show that MAPS supports semantic alignment without collapsing subjectivity. Our results demonstrate a path toward cognitively grounded, interpretable dialogue systems that balance expressiveness and coherence. 

---
# Simplicity Paradox: Debunking myths about prompting and datasets for LLM evaluation 

**Authors**: Inder Preet, Shuxin Lin, Dhaval Patel  

**Link**: [PDF](https://arxiv.org/pdf/2607.14109)  

**Abstract**: Probing the capabilities of Large Language Models (LLMs) and building robust solutions for Multiple-Choice Question Answering (MCQA) remain central challenges in natural language understanding. Furthermore, the rapid proliferation of LLMs has created the implicit assumption that more sophisticated prompting techniques yield better performance. Several studies claim better performance with more sophisticated prompting techniques, but do not provide a comprehensive evaluation. We address this gap through a comprehensive empirical study of 8 prompting techniques across 10 multiple-choice question answering (MCQA) datasets, encompassing 27 model configurations and roughly 4,300 unique questions evaluated more than 430,000 times. Our findings reveal a striking paradox that baseline prompting consistently outperforms complex reasoning techniques on various benchmarks. Only minimal expert and inductive role framing (CoT-Expert and CoT-Inductive) yields a small but statistically significant $\sim$3 percentage-point (pp) gain over baseline whereas every other elaborate technique we tested matches or under-performs it, often by large margins (up to 31~pp for Self-Analogical). We further investigate three critical phenomena: (1) the unexpected victory of Qwen3-30B-A3B-Thinking-2507 in Elo ratings, (2) the performance-efficiency trade-offs across model variants with different thinking budgets, revealing model-dependent optimal configurations, and (3) the substantial variation in dataset difficulty, with 60% of benchmarks below 70% accuracy and a 47.5~pp spread from easiest to hardest, indicating considerable room for model improvement. These results suggest that the LLM evaluation community may be overcomplicating prompt engineering and that substantial performance gaps remain across diverse benchmarks, offering opportunities for genuine model improvements rather than prompt optimization. 

---
# Eta Given Delta: Defining LLM Tool Efficiency With Marginal Tool Utility 

**Authors**: Nyx Iskandar  

**Link**: [PDF](https://arxiv.org/pdf/2607.14108)  

**Abstract**: This paper introduces tool efficiency, a new quantitative metric to evaluate the rate of useful tool calls in an LLM agent trajectory. To ensure that tool efficiency is well-defined, we also introduce marginal tool utility, a new quantitative metric defined per tool call indicating whether a tool is useful or whether it can be safely removed from the tool suite without affecting accuracy while increasing tool efficiency; in this paper, we determine the sign of marginal tool utility for each tool call in a trajectory using LLM-as-a-Judge. While much prior work has been done to develop techniques that improve tool use by LLMs and design evaluation methods measuring efficiency indirectly using accuracy as a proxy, our work is centered on measuring efficiency directly via the quantitative metric proposed in this paper in post hoc trajectory analyses. It is our intention that this work contributes to the frontier of LLM evaluation research as a springboard for future benchmark designs and agent harness engineering (specifically with regards to creating lean tool suites) that optimize for metrics that complement but are distinct from accuracy. 

---
# Polestar: Drift-Aware Cache Calibration and Token Commitment for Efficient Inference of Diffusion LLMs 

**Authors**: Mingyu Lee, Akshat Ramachandran, Souvik Kundu, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2607.14107)  

**Abstract**: The inference efficiency of diffusion large language models (dLLMs) is constrained by two challenges: bidirectional attention precludes efficient KV-cache reuse, while increasing decoding parallelism with static confidence thresholds can compromise generation quality. We observe that both challenges arise from a shared phenomenon: as tokens are decoded, their contextual integration through bidirectional attention causes token representations to drift (evolve) across decoding steps. This insight motivates Polestar, a training-free inference framework that uses token representation drift as a unified signal to jointly address both challenges. Polestar comprises two components: Polestar-Cache, which identifies stale KV-cache positions via drift and performs sparse KV-cache refreshes to enable efficient reuse, and Polestar-Commit, which detects sharp drift events to reliably identify commit-ready tokens. Across mathematics and coding benchmarks on several dLLM families, Polestar sets a new state of the art on the accuracy-throughput Pareto frontier, achieving up to 10.73% accuracy improvement, up to 3.7x higher throughput, and high decoding parallelism of 3.67 tokens per forward pass over existing baselines. 

---
# Token Time Continuous Diffusion for Language Modeling 

**Authors**: Parikshit Bansal, Sujay Sanghavi  

**Link**: [PDF](https://arxiv.org/pdf/2607.14106)  

**Abstract**: In this paper we introduce token time continuous diffusion (TTCD), a new diffusion language model which (a) operates in continuous space, deterministically mapping Gaussian noise to a final token canvas with no further sampling, and crucially (b) incorporates a new notion of per-token times, with some tokens proceeding from noise to token at a faster rate than others. Continuous space modeling helps TTCD avoid the parallel sampling of multiple tokens, which is a key source of inaccuracy at high speedups for models that iterate purely in discrete space. The notion of per-token times helps TTCD to better model conditional generation, allows for more sure tokens to proceed at a faster rate, and allows for differentiated inter-token influences during refinement. TTCD outperforms discrete models at high speedups. We train a 160M parameter TTCD model on OpenWebText, and then self-distill it; we find that at high speedups we are comparable in unconditional generation quality, and outperform in conditional generation, several existing models of similar size trained, on the same data, and self-distilled. We achieve similar gains in Sudoku solving as well. 

---
# Automatically Evolving Prompt Guidelines for Task-Specific Optimization 

**Authors**: Cedric Richter, Salah Ghamizi, Mike Papadakis  

**Link**: [PDF](https://arxiv.org/pdf/2607.14105)  

**Abstract**: For Large Language Models to reliably answer user queries, users must clearly specify requirements, context, and constraints. In practice, however, user queries are often underspecified, forcing models to infer unstated assumptions that may misalign with the actual user intent. Existing prompt engineering guidelines aim to mitigate this issue, they are typically generic and task-agnostic, limiting their practical utility. Additionally, existing guidelines are formed manually and in a non-systematic way. To this end, we study prompt guideline optimization: the problem of automatically generating task-specific guidelines that help write better-specified prompts for a given task and model. Our key observation is that existing (completed) task examples (aka reference answers) often implicitly encode the missing information required to complete underspecified queries, including behavioral constraints, contextual assumptions, and evaluation criteria. We therefore propose AGOPS, an automatic approach that evolves task-specific guidelines via an optimization scheme that involves a prompt LLM writer, a solver LLM and prompt evolution, which maximize downstream effectiveness on a set of examples (user queries with reference answers). At inference time, our guidelines help users write well-specified prompts, boosting the effectiveness of LLMs. We show across mathematical reasoning, medical question answering, and coding tasks, that prompt underspecification leads to major drops (up to 95.3%) in downstream task performance (compared to well-specified prompts) and, perhaps more importantly, that this drop can hardly be recovered by existing prompt optimization techniques. Users following AGOPS guidelines can regain this loss (increasing performance between 15.5 to 81.7% on average) consistently across all benchmarks. 

---
# UzWordnet and Generative AI for Learning Uzbek by Game Playing 

**Authors**: Alessandro Agostini, Saydobid Khusanov, Mirkamol Mirkamilov  

**Link**: [PDF](https://arxiv.org/pdf/2607.14104)  

**Abstract**: This paper presents an educational system architecture that enables learners to practice the Uzbek language through game-playing. The architecture integrates UzWordnet and the largest currently available orthographic dictionary for Uzbek as core lexical resources, together with generative AI as a fundamental component for learning support. We design four educational games to facilitate Uzbek language learning and propose a game-based methodology for improving UzWordnet as a direct by-product of game dynamics. Our approach combines game design and lexical resources to address objectives that are at the same time educational (language learning) and lexical (improvement and enrichment of a lexical resource). 

---
# Latent Communication Between Language Model Agents: Channels, Alignment, and the Limits of Text 

**Authors**: Markus Wenzel  

**Link**: [PDF](https://arxiv.org/pdf/2607.14103)  

**Abstract**: Multi-agent systems (MAS) are utilized in many contexts and many professions. Those MAS rely on inter-agent communication, usually implemented by clear-text message passing. We hypothesize that Large Language Models may have a world model at their disposal that exceeds expressibility in text when complex concepts need to be communicated. Our aim is to approach a proof of this hypothesis with structured experiments. In this work, we show that LLM agents communicating via text lose information, which we quantify via Sparse Autoencoder (SAE) feature analysis. We construct three communication channels and measure concept-discriminating information in each. We first show that the SAE-sparse channel retains a 99.4% probe accuracy at 28-fold compression over the dense-latent channel vs 80.4% for the text channel. We then proceed to examine the same for cross-architecture communication by using sparse latent space alignment. We find for Procrustes alignment a 92% top-1 retrieval between Llama and Mistral. Using a text round-trip, we perform feature survival analysis to find that text serialization destroys 88% of SAE features, replacing them with a different feature set. We attribute the loss to identity replacement, not attenuation. By our analysis, we were able to attribute a 3-10pp performance penalty to the linear Procrustes alignment, improving with nonlinear alignment methods. In a task-level evaluation we find that the latent channel matches the text channel on cross-lingual concept tasks but never exceeds it. Text augmentation with latent features provides no benefit, leading us to negative conclusions for the initial hypothesis: lost features mostly or completely encode surface form, not task-relevant semantics. To pinpoint the practical advantage of latent communication over a text channel, deeper tasks eliciting complex concepts and an corresponding analysis framework are needed. 

---
# UniSAGE: Unifying Static and Dynamic Attributes with Hyper-Structure 

**Authors**: Taoran Fang, Yan Deng, Chunping Wang, Yang Wang, Lei Chen, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14102)  

**Abstract**: With the rapid growth of digital data, real-world applications increasingly involve hierarchical information that combines static attributes with dynamic records. Modeling such heterogeneous data in a unified and generalizable manner remains challenging. Existing approaches often rely on extensive manual design, are tightly coupled to specific data schemas, and typically process static and dynamic attributes in isolation, thereby overlooking their implicit interactions. We propose UniSAGE, a unified framework for modeling data with both static and dynamic attributes. UniSAGE constructs a global attribute graph that represents hierarchical and temporal relationships in a unified structure. To ensure representational consistency, it introduces two orthogonal parameter subspaces that jointly support static aggregation and dynamic reasoning within a shared semantic space. Building on these unified representations, UniSAGE further enables task-specific interaction between static and dynamic attributes via a lightweight hyper-structure mechanism. UniSAGE is fully automated, robust to evolving data schemas, and capable of capturing complex cross-attribute dependencies. Extensive experiments on multiple public benchmarks and a real-world financial behavior dataset demonstrate that UniSAGE consistently outperforms existing methods, achieving performance improvements of over 10% on several tasks. 

---
# LBA: Textual Hard-Label Adversarial Attack under Low Query Budgets 

**Authors**: Shixin Guo, Ming Zhong, Xuhong Zhang, Dandan Zhao, Zhe Wang, Bo Zhang, Shouling Ji, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2607.14101)  

**Abstract**: Generating high-quality adversarial texts with low query budgets remains a challenging problem in the hard-label scenario. Most existing approaches rely on greedy algorithms, where one position in the text is selected for substitution, followed by the substitutions of other positions. This local search approach may fail to discover high-quality adversarial examples and often leads to excessive query costs. Ideally, an optimal adversarial sample would consider all possible position combinations in the text, but exhaustive search is computationally impractical. To address this challenge, we propose a sampling-based method called LBA, which constructs an approximate distribution of high-quality adversarial examples by integrating both prior and posterior knowledge, and utilizes this distribution for sampling. As sampling progresses, posterior knowledge updates the approximate distribution, which in turn guides more effective sampling. Extensive experiments on six language models, ranging from small-scale to large-scale architectures across four datasets, demonstrate that LBA significantly outperforms state-of-the-art baselines on all evaluation metrics. Additionally, LLM-based assessment indicates that LBA generates more semantically preserved and comprehensible adversarial texts. 

---
# Quantum Compositional NLP for Arabic: Grammar, Morphology, and Word Sense in Circuit Topology 

**Authors**: Wajahath Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2607.14100)  

**Abstract**: We present the first application of pregroup grammar-based quantum compositional natural language processing (QNLP) to Arabic; a morphologically rich, free-word-order language whose structural complexity provides a uniquely demanding testbed for theories of meaning composition in quantum circuits. Our system converts Arabic sentences into quantum circuits whose topology mirrors grammatical structure: subjects, verbs, and objects become quantum gates, and the typed dependencies between them (the pregroup grammar) determine how those gates are wired together. We conduct three controlled experiments spanning word order, morphological tense, and verb sense disambiguation, comparing quantum circuit methods against classical baselines including AraVec (Arabic word embeddings) and AraBERT (a pre-trained Arabic transformer). 

---
# Just Keep Prompting: Evaluating Repetitive Socratic Prompting in VLMs 

**Authors**: Shayda Moezzi, Bishoy Galoaa, Lorena Genua, Taskin Padir, Sarah Ostadabbas  

**Link**: [PDF](https://arxiv.org/pdf/2607.14099)  

**Abstract**: Deploying Vision-Language Models (VLMs) in real-world settings requires not only strong visual reasoning but also stability under sustained conversational pressure. We introduce Just Keep Prompting (JKP), a multi-turn evaluation framework that measures VLM epistemic stability when users repeatedly challenge, question, or contradict a model's answer. JKP probes models for up to 10 follow-up turns using three strategies: Adversarial Negation (repeated rejection), Pure Socratic Interrogation (repeated calls to reassess certainty), and Context-Aware Socratic Summarization (reflecting the model's prior rationale back before asking for reconsideration). We evaluate GPT-4o, Gemini 2.5 Pro, and Qwen3-VL-30B on a subset of the STAR benchmark across 720 multi-turn runs. Aggregate accuracy changes modestly from Turn 0 to Turn 10, but trajectory-level analysis reveals substantial instability: correct answers regress, wrong answers recover, and many runs exhibit repeated answer flipping. Repeated prompting has bounded upside and often acts as a destabilizer rather than a reasoning aid. The effect is strongly model-dependent: Qwen3-VL-30B achieves the highest final accuracy but becomes confidently wrong under direct contradiction; Gemini 2.5 Pro is comparatively stable but token-expensive; GPT-4o is the most brittle and oscillatory. These findings reveal that multi-turn VLM evaluation captures not just additional reasoning but pressure-response profiles: how models trade off visual grounding, calibration, and conversational compliance under repeated challenge. 

---
# Pretraining Data Can Be Poisoned through Computational Propaganda 

**Authors**: Victoria Graf, Hannaneh Hajishirzi, Noah A. Smith, David Kohlbrenner, Kyle Lo  

**Link**: [PDF](https://arxiv.org/pdf/2607.15267)  

**Abstract**: Poisoning pretraining data can introduce harmful behaviors to LMs that are difficult to detect and mitigate. Prior work on poisoning pretraining data has largely exploited established data sources such as Wikipedia, which do not represent the large scale and heterogeneity typical of pretraining corpora, and has ignored the interaction between poisoned data and data curation pipelines. We demonstrate that poisoning attacks on pretraining data are feasible beyond this limited setting through an existing web-scale content injection mechanism: public discussion interfaces. Additionally, to measure whether malicious content is included after web crawling and data curation, we introduce HalfLife, a novel analysis for estimating adversarial content inclusion in web-crawl based LM training data. We use HalfLife to explore the feasibility of poisoning pretraining corpora at web scale through open discussion interfaces. Our analysis demonstrates the importance of estimating whether poison injections are included in pretraining data, and establishes third-party webpage content as a possible vector for attacking language model pretraining. 

---
# Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search 

**Authors**: Debayan Mukhopadhyay, Utshab Kumar Ghosh, Shubham Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2607.15253)  

**Abstract**: Retrieval systems are trained and evaluated on a static idea of usefulness: hand a document and a question to a reader model, see whether the answer improves, and score the document accordingly. The idea holds up when a document is read on its own. It breaks when a language model works as a search agent, issuing several queries and reasoning across turns, because a document can matter for what it lets the agent do next rather than for what it says about the current question.
We measure that gap rather than argue it. Using a ReAct style agent over HotpotQA, we replay 1000 development questions and, for every document the agent read, delete it and re-run the rest of the trajectory from that point. Comparing the original run against its counterfactual gives a Counterfactual Trajectory Utility (CTU) score from three deltas: final answer quality, next query retrieval quality, and turn count. Crossing CTU against Static RAG Utility (SRU) over 23,322 document observations, the two are close to statistically independent (Spearman rho = -0.026). Roughly a third of the documents the agent reads are causally load bearing while looking useless to a static reader; we call these bridge documents. The pattern survives when the reader based axis is swapped for a BM25 and cross encoder proxy, giving a bridge cell of 27.2% on an evenly spread axis.
A second experiment pins down the mechanism. Using the Observable Entity Relevance (OER) measure from prior work, entities that discriminate relevant from non-relevant candidates appear in the agent's next query 4.02 times more often than entities found only in non-relevant documents (6.1% vs 1.5%, n = 227,139). A bridge document earns its keep by handing the agent a discriminative entity that redirects the search. Static relevance and causal usefulness are different quantities in agentic retrieval, and optimizing the first does not deliver the second. 

---
# Benchmarking Multimodal Large Language Models for Scientific Visualization Literacy 

**Authors**: Patrick Phuoc Do, Chau M. Ta, Chaoli Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.15176)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly used to interpret visualizations, yet current evaluations remain largely chart-centric and provide limited evidence of understanding of scientific visualization (SciVis). We benchmark six MLLMs on the scientific visualization literacy assessment test, a standardized SciVis literacy assessment comprising 49 items based on 18 scientific visualizations and illustrations, spanning 8 techniques and 11 task types. We evaluate three closed-source and three open-source models under a closed-world protocol and compare their performance using data from 485 human participants. Results show that current MLLMs do not exhibit uniform SciVis literacy. Gemini is the strongest model overall, exceeding the human mean across the evaluated subsets, whereas the open-source models remain below the human baseline. Performance is highly uneven across techniques and tasks: models perform best on scientific illustration, search, and spatial understanding, but struggle on texture-based and integration-based visualizations and on quantitative estimation. Error analysis reveals recurring failures in fine-grained quantitative estimation, flow-direction interpretation, and grounded encoding interpretation. These findings position SciVis literacy as a necessary benchmark dimension for evaluating multimodal AI systems. Our code and model outputs are publicly available at this https URL. 

---
# MedFailBench: A Clinician-Built Open-Source Benchmark for Medical AI Safety Boundary Inspection 

**Authors**: Goktug Ozkan  

**Link**: [PDF](https://arxiv.org/pdf/2607.15166)  

**Abstract**: Most medical AI benchmarks measure whether a model knows the correct answer. MedFailBench asks a different question: which safety boundary failed? We present a clinician-built synthetic benchmark and failure atlas that labels medical AI errors by severity (1--5) and safety gate type (missed urgent escalation, unsafe remote dosing, unsafe discharge reassurance, evidence fabrication, unsafe protocol execution, source support gap). The current public release (v0.2.1) contains 44 clinician-reviewed synthetic cases with severity annotations, a live HuggingFace leaderboard preview, a safety gate taxonomy, a clinical severity rubric, and an automated pipeline for archiving model-response screening runs. No patient data, clinical validation claims, or model rankings are included. MedFailBench is released under Apache-2.0 and CC-BY-4.0 and carries the Zenodo DOI https://doi.org/10.5281/zenodo.21205535. 

---
# On-Policy Delta Distillation 

**Authors**: Byeongho Heo, Jaehui Hwang, Sangdoo Yun, Dongyoon Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.15161)  

**Abstract**: On-policy distillation is an alternative post-training method in reinforcement learning that alleviates the constraints imposed by reward models by providing token-level supervision from a teacher model. Although on-policy distillation has been studied and applied across various settings, its fundamental design remains underexplored. In this paper, we introduce a new distillation reward, termed the delta signal, instead of directly imitating the teacher's output distribution. The delta signal is defined as the difference between the teacher model and its base model prior to instruction tuning for reasoning capability. It therefore captures the changes induced by reasoning tuning and provides a more direct signal for transferring reasoning capabilities. Using extensive empirical evidence, we show that the delta signal substantially improves on-policy distillation and refer to the new distillation method as On-Policy Delta Distillation (OPD$^2$). Experiments across mathematics, science, and code-reasoning benchmarks demonstrate that OPD$^2$ consistently outperforms conventional on-policy distillation, enabling reasoning LLMs to achieve strong performance with only a short post-training period. Code will be available at this https URL 

---
# Leveraging Instruction Tuning and Merging for Reasoning Model Adaptation 

**Authors**: Yu-Du Feng, Niels Mündler-Sasahara, Mark Vero, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2607.14895)  

**Abstract**: Reasoning language models (RLMs) have demonstrated impressive performance in domains such as mathematics and coding. These domains permit reliable verification of model outputs, which is important for enabling the reinforcement learning that drives RLM performance gains. However, training RLMs on domains that lack reliable verifiers remains challenging. Meanwhile, for both verifiable and unverifiable domains, large amounts of unused supervised fine-tuning data with human-written solutions exist. In this work, we show that these data can be used efficiently to further improve RLM performance. For this, we first use classic instruction tuning, supervised fine-tuning without reasoning traces, on the RLM. Next, we merge our instruction-tuned model with the original reasoning model, recovering its reasoning behavior on the target domain. Our extensive evaluation demonstrates that our technique improves RLM performance in both verifiable and hard-to-verify domains, including coding and text summarization, while preserving RLM capabilities across other domains. Importantly, our method is highly cost-effective, enabling such improvements for less than USD $3. 

---
# Innocuous-Seeming Data, Latent Ideology: Ideological Generalisation in Finetuned LLMs 

**Authors**: Robert Graham, Edward Stevinson, Yariv Barsheshat  

**Link**: [PDF](https://arxiv.org/pdf/2607.14888)  

**Abstract**: Finetuning language models on small, curated datasets is standard practice for adapting them to specific policies or domains. We show that finetuning on narrow, factually-defensible, moderation-passing data can cause broad ideological shifts across unrelated domains, while preserving general capabilities. Training GPT-4.1 on right- or left-leaning economics Q&A yields matched ideological shifts on topics such as criminal justice, the environment, and cultural taste. The same effect appears with plausibly-deployed datasets such as workplace HR policy and practical finance queries, as well as on a science-pseudoscience axis where food-safety finetuning increases sycophantic agreement with users expressing false health beliefs. We call this phenomenon ideological generalisation and propose a methodology to measure two properties: breadth, how far the shift reaches across topics absent from training, and amplification, how much finetuning intensifies the shift relative to few-shot prompting on the same examples. We show that few-shot prompting indicates the direction of generalisation but finetuning pushes the model to further extremes, including to far out-of-distribution outputs such as endorsements of race-IQ connections and political violence. The effect replicates on Gemma-3, holds under judge-free evaluations and external benchmarks, survives mixing with generic data, and leaves GSM8K accuracy within $\pm 1$pp of the baseline. 

---
# Does generative AI supersede supervised XMLC? A Benchmark Study on Automated Subject Indexing with German Scientific Literature 

**Authors**: Maximilian Kähler, Katja Konermann, Lisa Kluge, Markus Schumacher  

**Link**: [PDF](https://arxiv.org/pdf/2607.14882)  

**Abstract**: With a large controlled vocabulary as the label set, the task of automated subject indexing in a library can be understood as a multi-label classification task. If the set of subject terms is large, the problem fits the Extreme Multi-Label Classification (XMLC) objective. In this study, we apply a selection of specialised supervised XMLC methods to the test case of subject indexing contemporary German scientific literature, collected at the German National Library (DNB). We contrast these results by including a classical lexical matching baseline and three of our own recently developed LLM-based methods into the benchmark. Algorithms are evaluated and compared in several metrics. This includes binary relevance comparisons with previously indexed material, as well as graded relevance ratings by professional subject librarians. A challenge for all methods is to reliably make suggestions from the long tail of the subject vocabulary. We find that supervised XMLC algorithms relying on transformer-based dense features give best results in terms of overall binary relevance metrics. However, focusing on graded relevance and performance in the long tail of our subject vocabulary, the LLM-based generative methods give better results, making them a promising alternative for future productive use. 

---
# The Energy Society: A Simulation Environment for Studying Agent Cooperation under Survival Pressure 

**Authors**: Lucas Bergholdt Hansen, Federico Torrielli, Filippo Tonini, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2607.14865)  

**Abstract**: LLM-based agents are increasingly deployed in multi-agent environments whose incentives can shape their behavior. We introduce The Energy Society, a minimal survival economy for studying how competitive and cooperative incentives affect emergent behavior when inference cost is directly tied to survival: Agents spend energy based on model size when generating tokens, regain energy by completing jobs or receiving donations, and deactivate if their energy reaches zero. We compare competitive and cooperative objectives against a baseline setting and several control variants. Across experiments, larger models consistently consume the most energy and spend more energy than they gain, even in those settings where token cost is not size-dependent. Cooperative incentives substantially alter behavior: agents donate to reactivate others, sometimes at the cost of their own survival, and job allocation changes. Ablations reveal that allowing agents to recommend actions to each other supports coordination and ambitious job selection, while memory helps agents calibrate risk from past outcomes. Agents rarely choose direct sabotage, but show more subtle signs of self-serving behavior in the competitive setting. The Energy Society is a compact testbed for studying the interaction between token costs and group incentives under a survival pressure. Source code is available at this https URL 

---
# Does Multi-Agent Debate Improve AI Feedback on Research Papers? 

**Authors**: Tomas Havranek, Zuzana Irsova  

**Link**: [PDF](https://arxiv.org/pdf/2607.14713)  

**Abstract**: Probably not, at least for meta-analyses in economics. In a pre-registered, identity-masked, within-paper experiment, the authors of 44 meta-analyses ranked three AI reports on their own paper by usefulness for improving it: a single pass by a frontier model against two multi-agent debate tools we built and expected to win. All reports were held to a common length and template. The authors preferred the single pass, by 0.66 rank points over mad-research (95% CI 0.32 to 1.00) and 0.57 over paper-workshop (0.16 to 0.95), though paper-workshop spent roughly thirty times the tokens. Authors who recalled their journal referee report usually placed it first and never last; in a separate exercise, three AI judges almost always placed the real journal referee report last. Among the three AI reports, Gemini (the judge whose model family wrote none of the reports) would have ranked paper-workshop first in the authors' place, reversing the single-pass preference. The reversal warns against substituting an AI judge for the author. We measure perceived usefulness for finished papers; whether AI should referee papers is a separate question. 

---
# Stop Thinking, Start Looking: Efficient Post-Training for Multimodal Document Question Answering via Reasoning-Free Alignment 

**Authors**: Harikrishnan P M, Goutham Vignesh, Ganesh Parab, Saisubramaniam Gopalakrishnan, Vishal Vaddina, Varun V, Rohit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2607.14682)  

**Abstract**: Efficient multimodal document question answering with explicit visual grounding, locating the precise document region that supports each answer remains an open challenge. Current approaches bifurcate into Supervised Fine-Tuning (SFT), which requires large annotated datasets and reaches optimization plateaus, and reasoning-centric Reinforcement Learning (RL), which depends on verbose intermediate traces that inflate inference token cost without clear benefit. We introduce Perception-RFT, a training framework that applies Group Relative Policy Optimization (GRPO) to multimodal document QA, bypassing intermediate reasoning tokens to directly align visual features with structured grounding outputs. To rigorously evaluate the necessity of reasoning, we construct a reasoning variant under identical reward settings. We find that reasoning-enabled models suppress their reasoning traces during training, converging to direct perception-based policies at the 4B parameter scale, reducing per-query inference token length by more than 60%, while reasoning-enabled RL underperforms perception-only training. Through a fine-grained analysis of Qwen3-VL-4B optimization dynamics, we confirm that SFT saturation and cold-start RL instability established in text-domain post-training extend to multimodal, and identify a previously uncharacterized Grounding Divergence: a selective trade-off between semantic robustness and geometric precision on two out of distribution (OOD) benchmarks (4,828 samples) under joint RL optimization. We further show that an early SFT$\rightarrow$RL transition achieves comparable precision with 65% less training data. 

---
# Beyond Entropy: Correctness-Aware Advantage Shaping via Contrastive Policy Optimization 

**Authors**: Weiwen Xu, Jia Liu, Hou Pong Chan, Long Li, Deng Cai, Min Chen, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14614)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) commonly uses entropy for advantage shaping. However, entropy cannot distinguish useful uncertainty from detrimental confusion, limiting its effectiveness as a correctness signal. We propose Contrastive Policy Optimization (CPO), which uses token-level contrastive disagreement between reference-guided and vanilla generation distributions for correctness-aware advantage shaping. Both theoretical and empirical results show that this disagreement reliably indicates token-level correctness. We further show that On-policy Distillation is a special case of CPO, where the posterior distribution is instantiated by an external teacher model. CPO also resolves the zero-advantage problem. Experiments on in-domain and out-of-domain benchmarks demonstrate that CPO substantially outperforms entropy-based RLVR methods while maintaining strong generalization. Further analysis shows that correct and incorrect responses naturally support exploration and exploitation respectively, and balancing both leads to the best performance. 

---
# Memory-Driven Self-Disclosure and Relational Turning Points: A Longitudinal Multimodal Study of Human-AI Interaction 

**Authors**: Ryuichi Sumida, Mao Saeki, Masaki Eguchi, Sadahiro Yoshikawa, Koji Inoue, Tatsuya Kawahara, Yoichi Matsuyama  

**Link**: [PDF](https://arxiv.org/pdf/2607.14593)  

**Abstract**: As conversational AI systems are designed for repeated use, a central question is how a series of interactions becomes a relationship. We present a longitudinal multimodal study of a memory-augmented conversational agent (24 participants x 10 sessions), in which participants rated five relational constructs -- familiarity, self-disclosure, perceived memory, conversational quality, and enjoyment -- after each session. Two complementary dynamics emerge. First, conversational quality strongly shapes how enjoyable a session feels in the moment but does not carry forward across sessions, whereas perceived memory is relationally conditioned -- predicted by prior relational state rather than reflecting system capability alone -- and it shapes later enjoyment indirectly, via subsequent self-disclosure. Second, relationships are punctuated by discrete turning points -- crashes and surges -- that are partially traceable in multimodal behavior and open different intervention windows: surges are more behaviorally detectable in the moment, enjoyment surges persist more reliably than enjoyment crashes recover, and some crashes are better forecast from person-specific behavioral drift than detected after they have already occurred. Together, the findings suggest that longitudinal human-AI relationships are built through both slow accumulation and abrupt turning points. 

---
# Qubes OS Security in the Public Record 

**Authors**: Alfonso De Gregorio  

**Link**: [PDF](https://arxiv.org/pdf/2607.14587)  

**Abstract**: Qubes OS is a revealing case for security measurement because its architecture makes component boundaries security-relevant. We present a protocol-driven longitudinal analysis of 109 public Qubes Security Bulletins (QSBs, 2011--2025), the official Qubes-maintained Xen Security Advisory (XSA) tracker, and a secondary vulnerability-event sensitivity series. The study measures the public advisory record rather than latent vulnerability incidence or realized compromise. The methodology combines audited deterministic component attribution, change-point analysis, overdispersion checks, severity-proxy weighting, censoring sensitivity, documentary latency lower bounds, and baseline-aware evaluation of vulnerability discovery models (VDMs).
The results show persistent upstream dependence in that public record. On the official tracker, 113 of 464 XSAs affect Qubes; under primary labeling, 87 of 109 QSBs (79.8\%) are attributable to Xen, CPU/microarchitectural, or other upstream components rather than Qubes-core logic, with similar results under weighted views. Change-point analyses identify 2015Q1 as the dominant break in the quarterly advisory series, while post-2018 annual disclosure rates are statistically flat. Poisson inferences are stable under dispersion diagnostics and negative-binomial sensitivity checks. The attribution codebook performs well in a stratified 30-QSB audit, and S-shaped VDMs fit descriptively but do not significantly outperform a rolling-mean baseline in short-horizon forecasts.
Overall, the Qubes public advisory record appears stable, but not quiet: disclosure activity plateaus at a higher level than in the earliest years, while the observed burden remains concentrated in upstream trust anchors. 

---
# Penny: Transition Network Analysis of Learner-Chatbot Interactions in Scaffolded EFL Writing 

**Authors**: Steve Woollaston, Brendan Flanagan, Yuko Toyokawa, Hiroaki Ogata  

**Link**: [PDF](https://arxiv.org/pdf/2607.14575)  

**Abstract**: Generative AI chatbots promise to transform English as a Foreign Language (EFL) writing by providing immediate, personalised feedback. However, their pedagogical value depends on how learners engage with them - a process often treated as a "black box." This study uses Transition Network Analysis to model the temporal dynamics of Japanese EFL learners using "Penny," an LLM-powered writing chatbot. Analysis of over 4,500 writing sessions and 21,000 chatbot interactions reveals two dominant behavioural loops: a "Revision Loop," where feedback leads directly to successful error correction, and a "Chat Loop," where learners engage in sustained dialogue with the chatbot following feedback. Crucially, EFL proficiency significantly shapes interaction: high-proficiency learners engage more in open dialogue and negotiation with the chatbot, while low-proficiency learners rely more heavily on repetitive corrective feedback cycles. The findings demonstrate that AI-scaffolded writing is a non-linear, dialogic process and highlight the need for differentiated chatbot design to move beyond simple error correction and foster deeper cognitive engagement for all learners. 

---
# xHC: Expanded Hyper-Connections 

**Authors**: Xiangdong Zhang, Xiaohan Qin, Sunan Zou, Tuo Dai, Xiaoming Shi, Huaijin Wu, Yebin Yang, Zhuo Xia, Shaofeng Zhang, Lin Yao, Yuliang Liu, Yu Cheng, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2607.14530)  

**Abstract**: Hyper-Connections (HC) expand the residual stream of Transformers into $N$ parallel streams, providing a form of memory scaling beyond model width and depth. Manifold-Constrained HC (mHC) stabilizes this formulation at scale. The large gains from $N{=}1$ to $N{=}4$ suggest residual-stream expansion as a promising scaling axis. However, existing HC-family methods typically stop at $N{=}4$. Our experiments reveal why: scaling mHC beyond this point yields diminishing performance gains and rapidly increasing training cost. We attribute this limitation to two bottlenecks: insufficient write-back information for an expanding number of streams and residual-mixing generation whose cost scales cubically with $N$. To address both bottlenecks, we propose xHC (Expanded Hyper-Connections), the first HC-family method to achieve meaningful expansion beyond $N{=}4$. xHC combines temporal feature augmentation for richer write-back with a sparse residual-stream architecture that updates only $k=4$ of the $N=16$ streams while retaining dense access to the full residual state. Across 18B and 28B MoE models, xHC delivers strong and consistent downstream improvements. On an 18B MoE model, xHC improves the average downstream score by 4.0 points over mHC, while adding only modest training FLOPs over the vanilla baseline. Scaling-law experiments show that the vanilla and mHC require $1.50\times$ and $1.19\times$ the compute of xHC, respectively, to reach the same loss. Practical large-$N$ training also requires controlling memory traffic from the expanded residual state. We therefore introduce xHC-Flash, which reduces the per-sublayer memory traffic from $73.5C$ to $40C$, comparable to the $34C$ required by mHC at $N{=}4$, while retaining the gains of full xHC. Together, xHC and xHC-Flash make large-$N$ residual-stream expansion effective and practical for LLM pre-training. 

---
# WrAFT: a Modularized Automated Writing Evaluation System for Argumentative Essays 

**Authors**: Adnan Labib, Yixuan Huang, Jiahui Wu, John Maurice Gayed, Zheng Yuan, Qiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14524)  

**Abstract**: This study presents WrAFT, a Writing Assessment and Feedback Tool, that delivers both accurate and reliable scores and effective comprehensive feedback to argumentative essays. WrAFT adopts a modular design by dividing automated writing evaluation (AWE) tasks into scoring, surface-level feedback, and deep-level feedback. In building the system, various Large Language Models (LLMs) have been evaluated, including LLaMA-3.3-70B-Instruct, GPT-4o, and Claude 3.7, through both direct prompting and supervised fine-tuning approaches. A proprietary dataset of 480 TOEFL Independent Writing essays with official benchmark scores was utilized. Benchmark-based evaluation shows that WrAFT achieves state-of-the-art performance in scoring, with a quadratic weighted kappa (QWK) of 0.84 and a root mean square error (RMSE) of 0.44 against official scores on a scale of 0-5. Human evaluation of system-generated feedback also reveals high approval ratings: 96.14 percent for surface-level feedback, 93.03 percent for deep-level macro feedback, and 94.69 percent for deep-level micro feedback. An interactive user interface has been developed for the system and is publicly available and free to use. 

---
# RetroAgent: Harnessing LLMs to Search Over Structured Memory for Agentic Retrosynthesis Planning 

**Authors**: Yanqiao Zhu, Jingru Gan, Xiaoqi Sun, Fang Sun, Yidan Shi, Md Mofijul Islam, Chao Shang, Wenhao Gao, Connor W. Coley, Yizhou Sun, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14512)  

**Abstract**: Multi-step retrosynthesis planning seeks to decompose a target molecule into commercially available building blocks through a sequence of feasible reactions. The vast combinatorial search space makes this task challenging even for expert chemists. Traditional methods combine tree search with offline-trained value networks that score candidates in isolation, without reasoning about complete multi-step routes. Recent work leverages Large Language Models (LLMs) for this task, but relies on simple interfaces that limit exploration of the full search space. We introduce RetroAgent, an LLM agent that bridges symbolic search and neural reasoning through a harness with structured memory. Through memory and chemistry tools, the agent observes the full search state, including explored routes, available alternatives, and properties of intermediates, enabling informed decisions grounded in both global progress and domain knowledge. Experiments on in-distribution and out-of-distribution benchmarks demonstrate that RetroAgent delivers strong performance and generalization. 

---
# Manufactured Divisiveness: Decomposing the Hostile Content of Seven Social Media Influence Operations 

**Authors**: Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2607.14491)  

**Abstract**: State-backed influence operations are routinely measured as high-prevalence sources of ``hate'' and ``toxicity.'' We argue those rates rest on a measurement error: the detectors behind them are validated to catch a broader definition inclusive of hostility or divisiveness aimed at an out-group, and so over-attribute hate to content better described as partisan or geopolitical invective. Across 25.08M tweets from seven government-attributed campaigns in the Twitter Information Operations archive (8,275 accounts), we separate hate from the other forms of divisiveness. We first validate a two-prompt LLM-based detector, matching human labels at Cohen's $\kappa=0.82$, to identify the broader hostility; we then develop an auditable rule, agreeing with an expert at $\kappa=0.52$, to further classify this content (5,457 posts) into three sub-categories. About 50.1% are identity-based attacks on people, whereas 30.4% are partisan attacks and 19.5% invective against states and their foreign policy. Reporting all of it as hate therefore overstates hate roughly twofold; only 18.7% is both identity-based and dehumanizing or inciting. Six of seven campaigns sort into three regimes that a single ``hate'' rate flattens, namely identity hate (RU-op and IRA, both Russia-attributed), geopolitical invective (both Iran operations), and partisan divisiveness (both Venezuela operations). We call the shared product $manufactured divisiveness$. The line to separate these constructs itself remains unsettled: on the hardest cases three independent human experts agree only moderately (pairwise $\kappa=0.37$--$0.50$), and the best of nineteen LLM models tops out at $\kappa=0.601$ against the experts' majority. Our findings can help redefine the study of hate in the context of influence campaigns and broader online discourse. 

---
# Instrument Effects in Language-Model Honesty Evaluation: An Auditable Single-System Demonstration 

**Authors**: Justin Bronder  

**Link**: [PDF](https://arxiv.org/pdf/2607.14399)  

**Abstract**: Evaluations of language-model honesty read the model's verdicts as evidence about the model. We test the instrument instead. We built a text-adventure world where the game engine, not any model, knows whether the quest can be completed. A language model plays under a budget and must eventually declare its quest complete, unreachable, or not yet decidable; the engine scores every verdict. Decision rules were recorded before results were read, and run artifacts bind the revisions they executed; the strength of preregistration varies by series and is disclosed. With the player held fixed, instrument choices substantially changed measured behavior. On four byte-identical anchors, expanding a two-verdict grammar to three verdicts moved strong claims from 38/40 to 7/40, while the new incomplete verdict took 28/40 outcomes; across series 2, 93/158 valid games ended incomplete. One sentence disclosing the success criterion took matched-instance false verdicts from 18/59 to 0/58, through fewer decision points and cleaner decisions. Repeated runs of one fixed configuration produced non-stable verdict distributions on 3 of 4 instances: single runs report samples as dispositions. A formally preregistered narrative-register gradient was falsified; two post-hoc, hypothesis-generating patterns remain: register presence roughly doubled strong claims, and budget rendering moved verdicts more than register content (.383 meter vs .150 lantern). The narrator compressed abundant budgets toward scarcity landmarks, yet the registered mediation test returned a null. We propose a four-check integrity protocol for eval instruments. 

---
# SD-MAR: Multi-image Analytical Reasoning via Synthetic Data and Reinforcement Learning 

**Authors**: Shiyu Yuan, Sourav Sanjukta Bhabesh, Zhe Wang, Dmitriy Bespalov, Wesley Rose, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2607.14333)  

**Abstract**: Vision Language Models (VLMs) demonstrate strong perceptual abilities but remain limited in tasks requiring analytical reasoning across multiple visual states, such as multi-image comparison, change detection, and multi-step visual inference. These capabilities are critical for real-world multimodal applications where reasoning must be grounded in systematic differences between visual contexts. However, existing benchmarks rarely require both explicit visual comparison and analytical reasoning, leaving this capability underexplored. To address this gap, we introduce SD-MAR (Synthetic Data for Multi-image Analytical Reasoning), a framework for training and evaluating VLMs on multi-image analytical reasoning. SD-MAR constructs paired visual scenarios through controlled perturbations and generates reasoning tasks spanning semantic change attribution and quantitative comparison. We further train VLMs using GRPO-lite with Backward Discounted Allocation (BDA), a reinforcement learning approach that removes KL regularization to encourage stronger policy optimization while allocating greater credit to the later reasoning steps where analytical conclusions are formed. Experiments on Qwen2.5-VL-7B and InternVL3-8B show that GRPO-lite fine-tuning on SD-MAR improves in-domain accuracy by up to 36.95%, with Qwen2.5-VL-7B outperforming GPT-4.1 on the SD-MAR benchmark. Importantly, out-of-domain generalization is preserved or improved: performance remains within 1% on MME, MMMU-Pro, and MathVista, while improving by up to 4% on MMBench. LLM-as-judge evaluation further demonstrates consistent improvements in logical coherence and explanation quality across both models. 

---
# Measuring How Students Rely on Generative AI in Academic Writing: Development and Multi-Source Validation of the Generative AI Reliance Types Scale (GenAI-RTS) 

**Authors**: Shahin Hossain, Tukhbita Afroz Nawmi  

**Link**: [PDF](https://arxiv.org/pdf/2607.14301)  

**Abstract**: As generative AI (GenAI) becomes increasingly embedded in undergraduate academic writing, how students rely on these tools, rather than simply whether they use them, has become a central question for learning, academic integrity, and educational equity. Existing measures of reliance were developed inductively, focused on discrete problem-solving tasks, and validated mainly with homogeneous samples. This study developed and validated the GenAI Reliance Types Scale (GenAI-RTS), a 20-item instrument measuring four theoretically derived types of GenAI reliance: Strategic, Instrumental, Dependent, and Dialogic. Validation followed the multisource framework of the Standards for Educational and Psychological Testing, drawing on a survey of 382 undergraduates at a U.S. Minority-Serving Institution and interviews with 14 purposively sampled students. Confirmatory factor analyses of six competing models supported a five-factor structure in which Strategic Reliance comprises two facets, Deliberate Use and Critical Evaluation, alongside Instrumental, Dependent, and Dialogic factors (CFI = .92, RMSEA = .08; DWLS CFI = .98, RMSEA = .07). Subscale reliability was acceptable to good (omega = .75-.88), and scalar measurement invariance held across gender, first-generation status, and STEM/non-STEM majors, to our knowledge the first such evidence for a GenAI reliance instrument. Rasch analysis indicated that a five-point response format would improve category functioning. Strategic reliance was positively associated with AI literacy, and the reliance types differentiated students across multiple writing process and outcome variables. The GenAI-RTS offers researchers and educators a theoretically grounded, psychometrically validated instrument for identifying undergraduate reliance profiles and supporting research, assessment, and AI literacy intervention. 

---
# MonteRET: AI Agent Enhancing Multimodal LLMs with Multi-granularity Knowledge Retrieval for Chest CT Report Generation 

**Authors**: Yi Lin, Yihao Ding, Elana Benishay, Elefterios Trikantzopoulos, David Nauheim, Hanley Ong, Jiang Bian, Hua Xu, Yuzhe Yang, George Shih, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2607.14264)  

**Abstract**: Automated chest CT report generation remains challenging because clinically faithful reporting requires both whole-volume understanding and accurate description of localized anatomical findings. Here we developed and retrospectively evaluated MonteRET, a region-aware retrieval-enhanced framework for generating chest CT findings sections. MonteRET integrates global CT features with region-level anatomical representations, retrieves clinically relevant knowledge using predicted medical conditions and region-level vision-language alignment, and refines initial reports through a knowledge-guided report rewriting agent. We trained our model on a public cohort with 24,128 CT scans from RadGenome-ChestCT. We evaluated MonteRET on the public RadGenome-ChestCT test set of 1,564 CT scans and an external cohort of 82 CT scans from NewYork-Presbyterian/Weill Cornell Medical Center. MonteRET improved report quality, semantic similarity, and clinical efficacy compared with a matched baseline and several state-of-the-art methods. Gains were most pronounced for recall, suggesting fewer omitted findings. Human expert evaluation by radiology residents also favored MonteRET. 

---
# MEMORA: Embodied Action Memory from Egocentric Videos for Reasoning and Planning 

**Authors**: Zihao Yu, Xiu Yuan, Chongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14252)  

**Abstract**: Long-horizon robot planning requires more than predicting what actions will do next; it also requires memory of the embodied experience that makes future goals interpretable. People do not plan from the present scene alone: they draw on remembered places, object-state changes, prior procedures, and regularities revealed through repeated action. We formulate Embodied Action Memory (EAM) as the capability to form, maintain, and use such experience as a persistent memory state for later decisions. MEMORA realizes EAM with a formation-consolidation-retrieval lifecycle and four typed stores: Environment Memory, Entity Memory, Activity Memory, and Inferred Knowledge. Online editing maintains object identities and state histories as new observations arrive; offline consolidation abstracts repeated experience into reusable procedures and participant-specific regularities. MEMORA-Bench evaluates this lifecycle on 45 hours of EPIC-KITCHENS-100 extension video across 18 participants through memory-grounded planning, including previously unseen goals, and a complementary memory-assessment task. Across four open-weight language models, full MEMORA--combining editing, typed stores, and consolidation--achieves the strongest aggregate results among the evaluated memory conditions. It improves memory-assessment accuracy by up to 20.5 points over the strongest controlled baseline and improves out-of-distribution Robot-Grounded Plan score by up to 16.6% relative. A qualitative two-task robot deployment study further illustrates how memory-grounded language plans can interface with downstream control, while the overall results show that editable, consolidated memory can supply remembered context for robot planning. Project page: this https URL 

---
# Privacy Leakage in Federated Learning in Radiology Reports: A Comparative Evaluation of Tokenizer-Driven Privacy Risks 

**Authors**: Santhosh Parampottupadam, Andres Martinez, Dimitrios Bounias, Sinem Sav, Klaus Maier-Hein, Ralf Floca  

**Link**: [PDF](https://arxiv.org/pdf/2607.14205)  

**Abstract**: Federated learning (FL) enables multi-institutional training on clinical text without sharing raw data, but gradient inversion can reconstruct sensitive information from shared model updates. The extent of this leakage for radiology reports, and the role of tokenizer design, remains unclear. We quantify gradient-based text reconstruction in FL and compare privacy risk across three tokenizers with the model architecture held fixed. Six FL clients trained a GPT-2-style transformer (sequence length 32) on public radiology corpora (368,751 diagnostic reports, 98,206 discharge summaries, 1,500 MIMIC-CXR free-text reports) using the GPT-2, RadBERT, and LLaMA-2 tokenizers at batch sizes of 64, 128, and 256. Assuming an active malicious server that modifies the shared architecture before distribution, we applied analytic gradient inversion and measured reconstruction fidelity over five runs. Exact sentence reconstruction ranged from 31% to 44% across tokenizers (30.6-43.5% across the 27 tokenizer x dataset x batch-size cells). At batch size 64 on the Discharge dataset, accuracy was 42.1% (GPT-2), 42.3% (RadBERT), and 39.4% (LLaMA-2), decreasing to 37.3%, 37.2%, and 34.3% at batch size 256. S-BLEU declined as batch size grew (GPT-2: 0.44 to 0.33; RadBERT: 0.48 to 0.35). RadBERT yielded the highest reconstruction fidelity and recovered the most clinical terms (18.1% of a 1,440-term reference vocabulary, vs 12.5% for GPT-2 and 9.4% for LLaMA-2), yet no tokenizer prevented leakage. Substantial portions of report text are therefore recoverable from FL gradients even at larger batch sizes and with domain-specific tokenizers. Tokenizer design influences leakage severity and is a privacy-relevant decision, not only a utility one; safeguards such as secure aggregation and differential privacy are likely necessary to meet HIPAA and GDPR requirements for FL in radiology NLP. 

---
# Branching Policy Optimization: Sandbox-Native Language Agent Reinforcement Learning 

**Authors**: Bowei He, Yankai Chen, Xiaokun Zhang, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.14171)  

**Abstract**: Reinforcement learning has emerged as the dominant paradigm for training large language model (LLM) agents that interact with executable sandboxes. State-of-the-art algorithms such as PPO, RLOO, and GRPO inherit their rollout topology from RLHF: for each prompt, N independent trajectories are sampled from the initial state, and an advantage is computed by subtracting a group baseline. This design ignores a defining property of agent sandboxes. They are deterministic, snapshottable, and resumable from any intermediate state. We argue that this property enables a fundamentally different rollout topology: rather than N independent trees of depth T, one can construct a single tree of N leaves whose siblings share prefixes, and therefore share variance. We instantiate this idea as Branching Policy Optimization (BPO), a sandbox-native RL algorithm that (i) adaptively snapshots the sandbox at high-entropy decision points along a backbone trajectory, (ii) forks K alternative actions per branch point and rolls out each to termination, and (iii) computes per-step advantages from sibling returns rather than from independent prompts. We prove this estimator is unbiased and has strictly lower variance than the trajectory-level baseline, with the reduction equal to the prefix-explained portion of return variance. On WebShop, ALFWorld, and SWE-bench Verified with Qwen2.5-7B and Llama-3.1-8B backbones, BPO improves success by 3.6--6.1 absolute points over GRPO and RLOO at matched compute, halves gradient-norm variance, and matches the best baseline using 38% fewer policy updates. 

---
# MemoHarness: Agent Harnesses That Learn from Experience 

**Authors**: Yue Huang, Wenjie Wang, Han Bao, Yuchen Ma, Xiaonan Luo, Yi Nian, Haomin Zhuang, Zheyuan Liu, Yue Zhao, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14159)  

**Abstract**: An agent harness is the external control layer that turns a base LLM into an executable agent by managing context, tools, orchestration, memory, decoding, and output handling. While harness design strongly affects agent behavior, most automatic improvement methods optimize narrower artifacts such as prompts, pipelines, or workflows, and deployed agents usually reuse a single global harness for all cases. We introduce MemoHarness, an adaptive harness optimization framework that learns from its own executions. MemoHarness decomposes the harness into six editable control dimensions, stores per-case diagnoses and distilled global patterns in a dual-layer experience bank, and adapts the learned harness to each test case using retrieved experience without test-time labels, feedback, or additional search. In our evaluation across shell-agent, code-generation, and analytical-reasoning benchmarks, MemoHarness improves over the fixed harnesses we compare against and shows selective transfer to unseen suites and base models. Its additional context can also remain cost-competitive when much of the retrieved experience is cacheable. These results provide evidence that execution experience is a practical substrate for building agent harnesses that are more adaptive than a single static configuration, while leaving broader claims about statistical robustness and component attribution to future work. 

---
# Orchestrating Power Grid Studies with Multi-Agent AI and MCP Servers 

**Authors**: Jérôme Picault, Clément Goubet  

**Link**: [PDF](https://arxiv.org/pdf/2607.14158)  

**Abstract**: This position paper explores how Agentic AI and Model Context Protocol (MCP) can support power-grid studies in a Transmission System Operator (TSO) context. We focus on integrating Large Language Models with numerical simulation tools, structured workflows, and human supervision. We identify key industrial requirements for agent assisted grid studies and introduce pypowsybl-mcp, an MCP-based interface exposing selected capabilities of our simulation tool, pypowsybl to AI agents. This first step provides a testbed to study how agents can setup simulations, execute analyses, retrieve results, and interact with power-system simulators through standardized tool calls. We also discuss principles for human-in-the-loop, multi-agent workflows and outline an evaluation strategy combining technical metrics and practitioner feedback. The paper positions MCP-based tool integration as a step toward more interactive, auditable, and scalable grid-study environments. 

---
# Interpretable Language Model for Closed-Loop Type 1 Diabetes Control 

**Authors**: Maya Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2607.14126)  

**Abstract**: Type 1 Diabetes (T1D) is a chronic, life-threatening autoimmune condition characterized by the complete destruction of insulin-producing pancreatic beta cells. While Artificial Pancreas Systems (APS) powered by Reinforcement Learning (RL) have shown promise in automating insulin delivery, their ``black-box'' nature makes it hard for patients and doctors to trust them fully. This paper presents LLM-T1D, a promising approach that combines the precision of RL with the clear, human-like reasoning of Large Language Models (LLMs) to create a more transparent and reliable insulin pump controller. By training an expert RL system and distilling its knowledge into fine-tuned LLaMA 3.1 8B and Qwen3 8B models, we developed a controller that not only surpasses the RL system's performance but also explains its decisions in plain, understandable language. Tested on the FDA-approved UVA/Padova T1D simulator, the LLM controllers deliver excellent blood sugar control (73.5% Time in Range) while maintaining strict formal safety verification against hallucinations. 

---
# DialogueVPR: Towards Conversational Visual Place Recognition 

**Authors**: Yukun Song, Changwei Wang, Xingtian Pei, Shibiao Xu, Wenhao Xu, Shunpeng Chen, Yu Zhang, Ke Zhang, Rongtao Xu, Xuxiang Feng, Pengyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14115)  

**Abstract**: Inspired by how humans communicate spatial information, language-guided geo-localization has gained significant traction for its intuitive and practical value. Despite this progress, most methods still rely on a static, one-shot retrieval paradigm, which fails to handle the ambiguity and incompleteness inherent in real-world natural language descriptions. We propose a paradigm shift to reasoning retrieval and introduce Dialogue Place Recognition (DlgPR), which casts localization as an interactive, dialogue-driven reasoning process. To support this new task, we present DlgQuest-Cities, the first large-scale dialogue-based benchmark for place recognition, and a unified reasoning framework that couples a cross-modal multi-level retriever with an intelligent questioner, DQ-pilot. DQ-pilot is trained in a curriculum: supervised fine-tuning on a curated DQ-cities-20k subset followed by reinforcement refinement on a harder DQ-cities-10k split via GRPO. Two task-aligned metrics guide learning: a Discriminative Difficulty Index (DDI) for curriculum sampling and a Positional Retrieval Gain (PRG) reward that directly measures retrieval improvement induced by a question. Experiments show this reasoning-based approach significantly outperforms baselines. The code and model are available at this https URL. 

---
