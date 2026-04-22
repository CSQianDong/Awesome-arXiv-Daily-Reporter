# Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views 

**Authors**: Feihao Fang, My T. Thai, Yuanyuan Lei  

**Link**: [PDF](https://arxiv.org/pdf/2604.19716)  

**Abstract**: Large Language Models (LLMs) still struggle with multi-step logical reasoning. Existing approaches either purely refine the reasoning chain in natural language form or attach a symbolic solver as an external module. In this work, we instead ask whether LLMs contain a shared internal logical subspace that simultaneously aligns natural-language and symbolic-language views of the reasoning process. Our hypothesis is that this logical subspace captures logical reasoning capabilities in LLMs that are shared across views while remaining independent of surface forms. To verify this, we employ Canonical Correlation Analysis on the paired residual activations from natural-language and symbolic-language reasoning chains, learning a low-dimensional subspace with maximum cross-view correlation. Furthermore, we design a training-free approach that steers LLMs reasoning chain along this logical subspace, thereby leveraging the complementary reasoning signals from both views. Experiments on four logical reasoning benchmarks demonstrate the effectiveness of our approach, improving accuracy by up to 11 percentage points and generalizing well on out-of-domain problems. 

---
# Epistemic orientation in parliamentary discourse is associated with deliberative democracy 

**Authors**: Segun Aroyehun, Stephan Lewandowsky, David Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2604.19699)  

**Abstract**: The pursuit of truth is central to democratic deliberation and governance, yet political discourse reflects varying epistemic orientations, ranging from evidence-based reasoning grounded in verifiable information to intuition-based reasoning rooted in beliefs and subjective interpretation. We introduce a scalable approach to measure epistemic orientation using the Evidence--Minus--Intuition (EMI) score, derived from large language model (LLM) ratings and embedding-based semantic similarity. Applying this approach to 15 million parliamentary speech segments spanning 1946 to 2025 across seven countries, we examine temporal patterns in discourse and its association with deliberative democracy and governance. We find that EMI is positively associated with deliberative democracy within countries over time, with consistent relationships in both contemporaneous and lagged analyses. EMI is also positively associated with the transparency and predictable implementation of laws as a dimension of governance. These findings suggest that the epistemic nature of political discourse is crucial for both the quality of democracy and governance. 

---
# An Answer is just the Start: Related Insight Generation for Open-Ended Document-Grounded QA 

**Authors**: Saransh Sharma, Pritika Ramu, Aparna Garimella, Koyel Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2604.19685)  

**Abstract**: Answering open-ended questions remains challenging for AI systems because it requires synthesis, judgment, and exploration beyond factual retrieval, and users often refine answers through multiple iterations rather than accepting a single response. Existing QA benchmarks do not explicitly support this refinement process. To address this gap, we introduce a new task, document-grounded related insight generation, where the goal is to generate additional insights from a document collection that help improve, extend, or rethink an initial answer to an open-ended question, ultimately supporting richer user interaction and a better overall question answering experience. We curate and release SCOpE-QA (Scientific Collections for Open-Ended QA), a dataset of 3,000 open-ended questions across 20 research collections. We present InsightGen, a two-stage approach that first constructs a thematic representation of the document collection using clustering, and then selects related context based on neighborhood selection from the thematic graph to generate diverse and relevant insights using LLMs. Extensive evaluation on 3,000 questions using two generation models and two evaluation settings shows that InsightGen consistently produces useful, relevant, and actionable insights, establishing a strong baseline for this new task. 

---
# Exploring Language-Agnosticity in Function Vectors: A Case Study in Machine Translation 

**Authors**: Nurkhan Laiyk, Gerard I. Gállego, Javier Ferrando, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2604.19678)  

**Abstract**: Function vectors (FVs) are vector representations of tasks extracted from model activations during in-context learning. While prior work has shown that multilingual model representations can be language-agnostic, it remains unclear whether the same holds for function vectors. We study whether FVs exhibit language-agnosticity, using machine translation as a case study. Across three decoder-only multilingual LLMs, we find that translation FVs extracted from a single English$\rightarrow$Target direction transfer to other target languages, consistently improving the rank of correct translation tokens across multiple unseen languages. Ablation results show that removing the FV degrades translation across languages with limited impact on unrelated tasks. We further show that base-model FVs transfer to instruction-tuned variants and partially generalize from word-level to sentence-level translation. 

---
# Chat2Workflow: A Benchmark for Generating Executable Visual Workflows with Natural Language 

**Authors**: Yi Zhong, Buqiang Xu, Yijun Wang, Zifei Shan, Shuofei Qiao, Guozhou Zheng, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19667)  

**Abstract**: At present, executable visual workflows have emerged as a mainstream paradigm in real-world industrial deployments, offering strong reliability and controllability. However, in current practice, such workflows are almost entirely constructed through manual engineering: developers must carefully design workflows, write prompts for each step, and repeatedly revise the logic as requirements evolve-making development costly, time-consuming, and error-prone. To study whether large language models can automate this multi-round interaction process, we introduce Chat2Workflow, a benchmark for generating executable visual workflows directly from natural language, and propose a robust agentic framework to mitigate recurrent execution errors. Chat2Workflow is built from a large collection of real-world business workflows, with each instance designed so that the generated workflow can be transformed and directly deployed to practical workflow platforms such as Dify and Coze. Experimental results show that while state-of-the-art language models can often capture high-level intent, they struggle to generate correct, stable, and executable workflows, especially under complex or changing requirements. Although our agentic framework yields up to 5.34% resolve rate gains, the remaining real-world gap positions Chat2Workflow as a foundation for advancing industrial-grade automation. Code is available at this https URL. 

---
# Pause or Fabricate? Training Language Models for Grounded Reasoning 

**Authors**: Yiwen Qiu, Linjuan Wu, Yizhou Liu, Yuchen Yan, Jin Ma, Xu Tan, Yao Hu, Daoxin Zhang, Wenqi Zhang, Weiming Lu, Jun Xiao, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.19656)  

**Abstract**: Large language models have achieved remarkable progress on complex reasoning tasks. However, they often implicitly fabricate information when inputs are incomplete, producing confident but unreliable conclusions -- a failure mode we term ungrounded reasoning. We argue that this issue arises not from insufficient reasoning capability, but from the lack of inferential boundary awareness -- the ability to recognize when the necessary premises for valid inference are missing. To address this issue, we propose Grounded Reasoning via Interactive Reinforcement Learning (GRIL), a multi-turn reinforcement learning framework for grounded reasoning under incomplete information. GRIL decomposes the reasoning process into two stages: clarify and pause, which identifies whether the available information is sufficient, and grounded reasoning, which performs task solving once the necessary premises are established. We design stage-specific rewards to penalize hallucinations, enabling models to detect gaps, stop proactively, and resume reasoning after clarification. Experiments on GSM8K-Insufficient and MetaMATH-Insufficient show that GRIL significantly improves premise detection (up to 45%), leading to a 30% increase in task success while reducing average response length by over 20%. Additional analyses confirm robustness to noisy user responses and generalization to out-of-distribution tasks. 

---
# The signal is the ceiling: Measurement limits of LLM-predicted experience ratings from open-ended survey text 

**Authors**: Andrew Hong, Jason Potteiger, Luis E. Zapata  

**Link**: [PDF](https://arxiv.org/pdf/2604.19645)  

**Abstract**: An earlier paper (Hong, Potteiger, and Zapata 2026) established that an unoptimized GPT 4.1 prompt predicts fan-reported experience ratings within one point 67% of the time from open-ended survey text. This paper tests the relative impact of prompt design and model selection on that performance. We compared four configurations on approximately 10,000 post-game surveys from five MLB teams: the original baseline prompt and a moderately customized version, crossed with three GPT models (4.1, 4.1-mini, 5.2). Prompt customization added roughly two percentage points of within +/-1 agreement on GPT 4.1 (from 67% to 69%). Both model swaps from that best configuration degraded performance: GPT 5.2 returned to the baseline, and GPT 4.1-mini fell six percentage points below it. Both levers combined were dwarfed by the input itself: across capable configurations, accuracy varied more than an order of magnitude more by the linguistic character of the text than by the choice of prompt or model. The ceiling has two parts. One is a bias in how the model reads text, which prompt design can correct. The other is a difference between what fans write about and what they actually decide, which no engineering can close because the missing information is not in the text. Prompt customization moved the first part; model selection moved neither reliably. The result is not that "prompt engineering helps a little" but that prompt engineering helps in a specific and predictable way, on the part of the ceiling it can reach. 

---
# Micro Language Models Enable Instant Responses 

**Authors**: Wen Cheng, Tuochao Chen, Karim Helwani, Sriram Srinivasan, Luke Zettlemoyer, Shyamnath Gollakota  

**Link**: [PDF](https://arxiv.org/pdf/2604.19642)  

**Abstract**: Edge devices such as smartwatches and smart glasses cannot continuously run even the smallest 100M-1B parameter language models due to power and compute constraints, yet cloud inference introduces multi-second latencies that break the illusion of a responsive assistant. We introduce micro language models ($\mu$LMs): ultra-compact models (8M-30M parameters) that instantly generate the first 4-8 words of a contextually grounded response on-device, while a cloud model completes it; thus, masking the cloud latency. We show that useful language generation survives at this extreme scale with our models matching several 70M-256M-class existing models. We design a collaborative generation framework that reframes the cloud model as a continuator rather than a respondent, achieving seamless mid-sentence handoffs and structured graceful recovery via three error correction methods when the local opener goes wrong. Empirical results show that $\mu$LMs can initiate responses that larger models complete seamlessly, demonstrating that orders-of-magnitude asymmetric collaboration is achievable and unlocking responsive AI for extremely resource-constrained devices. The model checkpoint and demo are available at this https URL. 

---
# The "Small World of Words" German Free-Association Norms 

**Authors**: Samuel Aeschbach, Rui Mata, Kaidi Lõo, Simon De Deyne, Dirk U. Wulff  

**Link**: [PDF](https://arxiv.org/pdf/2604.19620)  

**Abstract**: Free-association norms provide essential empirical data for investigating linguistic, semantic, and cultural phenomena in the cognitive sciences. Although large-scale norms exist for languages such as English, Dutch, Spanish, and Mandarin Chinese, no comparable resource has been available for German. To address this gap, we present free-association norms for 5,877 German cue words as part of the German version of the multilingual Small World of Words (SWOW) project. We describe the data collection procedures, participant characteristics, and our comprehensive preprocessing pipeline before introducing the resulting SWOW-DE data set. Using data from three established psycholinguistic paradigms, we show that SWOW-DE norms robustly predict performance in lexical decision tasks, relatedness judgments, and psycholinguistic word ratings. Furthermore, we demonstrate that SWOW-DE responses compare favorably with existing German resources and provide a preliminary cross-linguistic comparison revealing both shared and language-specific association patterns, highlighting promising directions for future research. Overall, SWOW-DE represents the largest collection of German free associations to date and offers a unique resource for linguistic, psychological, and cross-cultural research. 

---
# Cross-Model Consistency of AI-Generated Exercise Prescriptions: A Repeated Generation Study Across Three Large Language Models 

**Authors**: Kihyuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.19598)  

**Abstract**: This study compared repeated generation consistency of exercise prescription outputs across three large language models (LLMs), specifically GPT-4.1, Claude Sonnet 4.6, and Gemini 2.5 Flash, under temperature=0 conditions. Each model generated prescriptions for six clinical scenarios 20 times, yielding 360 total outputs analyzed across four dimensions: semantic similarity, output reproducibility, FITT classification, and safety expression. Mean semantic similarity was highest for GPT-4.1 (0.955), followed by Gemini 2.5 Flash (0.950) and Claude Sonnet 4.6 (0.903), with significant inter-model differences confirmed (H = 458.41, p < .001). Critically, these scores reflected fundamentally different generative behaviors: GPT-4.1 produced entirely unique outputs (100%) with stable semantic content, while Gemini 2.5 Flash showed pronounced output repetition (27.5% unique outputs), indicating that its high similarity score derived from text duplication rather than consistent reasoning. Identical decoding settings thus yielded fundamentally different consistency profiles, a distinction that single-output evaluations cannot capture. Safety expression reached ceiling levels across all models, confirming its limited utility as a differentiating metric. These results indicate that model selection constitutes a clinical rather than merely technical decision, and that output behavior under repeated generation conditions should be treated as a core criterion for reliable deployment of LLM-based exercise prescription systems. 

---
# RoLegalGEC: Legal Domain Grammatical Error Detection and Correction Dataset for Romanian 

**Authors**: Mircea Timpuriu, Dumitru-Clementin Cercel  

**Link**: [PDF](https://arxiv.org/pdf/2604.19593)  

**Abstract**: The importance of clear and correct text in legal documents cannot be understated, and, consequently, a grammatical error correction tool meant to assist a professional in the law must have the ability to understand the possible errors in the context of a legal environment, correcting them accordingly, and implicitly needs to be trained in the same environment, using realistic legal data. However, the manually annotated data required by such a process is in short supply for languages such as Romanian, much less for a niche domain. The most common approach is the synthetic generation of parallel data; however, it requires a structured understanding of the Romanian grammar. In this paper, we introduce, to our knowledge, the first Romanian-language parallel dataset for the detection and correction of grammatical errors in the legal domain, RoLegalGEC, which aggregates 350,000 examples of errors in legal passages, along with error annotations. Moreover, we evaluate several neural network models that transform the dataset into a valuable tool for both detecting and correcting grammatical errors, including knowledge-distillation Transformers, sequence tagging architectures for detection, and a variety of pre-trained text-to-text Transformer models for correction. We consider that the set of models, together with the novel RoLegalGEC dataset, will enrich the resource base for further research on Romanian. 

---
# A Bolu: A Structured Dataset for the Computational Analysis of Sardinian Improvisational Poetry 

**Authors**: Silvio Calderaro, Johanna Monti  

**Link**: [PDF](https://arxiv.org/pdf/2604.19584)  

**Abstract**: The growing interest of Natural Language Processing (NLP) in minority languages has not yet bridged the gap in the preservation of oral linguistic heritage. In particular, extemporaneous poetry - a performative genre based on real-time improvisation, metrical-rhetorical competence - remains a largely unexplored area of computational linguistics. This methodological gap necessitates the creation of specific resources to document and analyse the structures of improvised poetry. This is the context in which A Bolu was created, the first structured corpus of extemporaneous poetry dedicated to cantada logudorese, a variant of the Sardinian language. The dataset comprises 2,835 stanzas for a total of 141,321 tokens. The study presents the architecture of the corpus and applies a multidimensional analysis combining descriptive statistical indices and computational linguistics techniques to map the characteristics of the poetic text. The results indicate that the production of Sardinian extemporaneous poets is characterised by recurring patterns that support Parry and Lord's theory of formulaicity. This evidence not only provides a new key to understanding oral creativity, but also offers a significant contribution to the development of NLP tools that are more inclusive and sensitive to the specificities of less widely spoken languages. 

---
# Impact of large language models on peer review opinions from a fine-grained perspective: Evidence from top conference proceedings in AI 

**Authors**: Wenqing Wu, Chengzhi Zhang, Yi Zhao, Tong Bao  

**Link**: [PDF](https://arxiv.org/pdf/2604.19578)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), the academic community has faced unprecedented disruptions, particularly in the realm of academic communication. The primary function of peer review is improving the quality of academic manuscripts, such as clarity, originality and other evaluation aspects. Although prior studies suggest that LLMs are beginning to influence peer review, it remains unclear whether they are altering its core evaluative functions. Moreover, the extent to which LLMs affect the linguistic form, evaluative focus, and recommendation-related signals of peer-review reports has yet to be systematically examined. In this study, we examine the changes in peer review reports for academic articles following the emergence of LLMs, emphasizing variations at fine-grained level. Specifically, we investigate linguistic features such as the length and complexity of words and sentences in review comments, while also automatically annotating the evaluation aspects of individual review sentences. We also use a maximum likelihood estimation method, previously established, to identify review reports that potentially have modified or generated by LLMs. Finally, we assess the impact of evaluation aspects mentioned in LLM-assisted review reports on the informativeness of recommendation for paper decision-making. The results indicate that following the emergence of LLMs, peer review texts have become longer and more fluent, with increased emphasis on summaries and surface-level clarity, as well as more standardized linguistic patterns, particularly reviewers with lower confidence score. At the same time, attention to deeper evaluative dimensions, such as originality, replicability, and nuanced critical reasoning, has declined. 

---
# A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression 

**Authors**: Jincheng Ren, Siwei Wu, Yizhi Li, Kang Zhu, Shu Xu, Boyu Feng, Ruibin Yuan, Wei Zhang, Riza Batista-Navarro, Jian Yang, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.19572)  

**Abstract**: As model capabilities advance, research has increasingly shifted toward long-horizon, multi-turn terminal-centric agentic tasks, where raw environment feedback is often preserved in the interaction history to support future decisions. However, repeatedly retaining such feedback introduces substantial redundancy and causes cumulative token cost to grow quadratically with the number of steps, hindering long-horizon reasoning. Although observation compression can mitigate this issue, the heterogeneity of terminal environments makes heuristic-based or fixed-prompt methods difficult to generalize. We propose TACO, a plug-and-play, self-evolving Terminal Agent Compression framework that automatically discovers and refines compression rules from interaction trajectories for existing terminal agents. Experiments on TerminalBench (TB 1.0 and TB 2.0) and four additional terminal-related benchmarks (i.e., SWE-Bench Lite, CompileBench, DevEval, and CRUST-Bench) show that TACO consistently improves performance across mainstream agent frameworks and strong backbone models. With MiniMax-2.5, it improves performance on most benchmarks while reducing token overhead by around 10%. On TerminalBench, it brings consistent gains of 1%-4% across strong agentic models, and further improves accuracy by around 2%-3% under the same token budget. These results demonstrate the effectiveness and generalization of self-evolving, task-aware compression for terminal agents. 

---
# Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps 

**Authors**: Jonas Waldendorf, Bashar Awwad Shiekh Hasan, Evgenii Tsymbalov  

**Link**: [PDF](https://arxiv.org/pdf/2604.19565)  

**Abstract**: Hallucinations in Speech Large Language Models (SpeechLLMs) pose significant risks, yet existing detection methods typically rely on gold-standard outputs that are costly or impractical to obtain. Moreover, hallucination detection methods developed for text-based LLMs do not directly capture audio-specific signals. We investigate four attention-derived metrics: AUDIORATIO, AUDIOCONSISTENCY, AUDIOENTROPY, and TEXTENTROPY, designed to capture pathological attention patterns associated with hallucination, and train lightweight logistic regression classifiers on these features for efficient inference-time detection. Across automatic speech recognition and speech-to-text translation tasks, evaluations on Qwen-2-Audio and Voxtral-3B show that our approach outperforms uncertainty-based and prior attention-based baselines on in-domain data, achieving improvements of up to +0.23 PR-AUC, and generalises to out-of-domain ASR settings. We further find that strong performance can be achieved with approximately 100 attention heads, improving out-of-domain generalisation compared to using all heads. While effectiveness is model-dependent and task-specific training is required, our results demonstrate that attention patterns provide a valuable tool for hallucination detection in SpeechLLMs. 

---
# Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment 

**Authors**: Bobo Li, Rui Wu, Zibo Ji, Meishan Zhang, Hao Fei, Min Zhang, Mong-Li Lee, Wynne Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19548)  

**Abstract**: Large Language Model agents have rapidly evolved from static text generators into dynamic systems capable of executing complex autonomous workflows. To enhance reliability, multi-agent frameworks assigning specialized roles are increasingly adopted to enable self-reflection and mutual auditing. While such role-playing effectively leverages domain expert knowledge, we find it simultaneously induces a human-like cognitive bias known as Actor-Observer Asymmetry (AOA). Specifically, an agent acting as an actor (during self-reflection) tends to attribute failures to external factors, whereas an observer (during mutual auditing) attributes the same errors to internal faults. We quantify this using our new Ambiguous Failure Benchmark, which reveals that simply swapping perspectives triggers the AOA effect in over 20% of cases for most models. To tame this bias, we introduce ReTAS (Reasoning via Thesis-Antithesis-Synthesis), a model trained through dialectical alignment to enforce perspective-invariant reasoning. By integrating dialectical chain-of-thought with Group Relative Policy Optimization, ReTAS guides agents to synthesize conflicting viewpoints into an objective consensus. Experiments demonstrate that ReTAS effectively mitigates attribution inconsistency and significantly improves fault resolution rates in ambiguous scenarios. 

---
# Emotion-Cause Pair Extraction in Conversations via Semantic Decoupling and Graph Alignment 

**Authors**: Tianxiang Ma, Weijie Feng, Xinyu Wang, Zhiyong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.19547)  

**Abstract**: Emotion-Cause Pair Extraction in Conversations (ECPEC) aims to identify the set of causal relations between emotion utterances and their triggering causes within a dialogue. Most existing approaches formulate ECPEC as an independent pairwise classification task, overlooking the distinct semantics of emotion diffusion and cause explanation, and failing to capture globally consistent many-to-many conversational causality. To address these limitations, we revisit ECPEC from a semantic perspective and seek to disentangle emotion-oriented semantics from cause-oriented semantics, mapping them into two complementary representation spaces to better capture their distinct conversational roles. Building on this semantic decoupling, we naturally formulate ECPEC as a global alignment problem between the emotion-side and cause-side representations, and employ optimal transport to enable many-to-many and globally consistent emotion-cause matching. Based on this perspective, we propose a unified framework SCALE that instantiates the above semantic decoupling and alignment principle within a shared conversational structure. Extensive experiments on several benchmark datasets demonstrate that SCALE consistently achieves state-of-the-art performance. Our codes are released at this https URL. 

---
# Bangla Key2Text: Text Generation from Keywords for a Low Resource Language 

**Authors**: Tonmoy Talukder, G M Shahariar  

**Link**: [PDF](https://arxiv.org/pdf/2604.19508)  

**Abstract**: This paper introduces \textit{Bangla Key2Text}, a large-scale dataset of $2.6$ million Bangla keyword--text pairs designed for keyword-driven text generation in a low-resource language. The dataset is constructed using a BERT-based keyword extraction pipeline applied to millions of Bangla news texts, transforming raw articles into structured keyword--text pairs suitable for supervised learning. To establish baseline performance on this new benchmark, we fine-tune two sequence-to-sequence models, \texttt{mT5} and \texttt{BanglaT5}, and evaluate them using multiple automatic metrics and human judgments. Experimental results show that task-specific fine-tuning substantially improves keyword-conditioned text generation in Bangla compared to zero-shot large language models. The dataset, trained models, and code are publicly released to support future research in Bangla natural language generation and keyword-to-text generation tasks. 

---
# Beyond Rating: A Comprehensive Evaluation and Benchmark for AI Reviews 

**Authors**: Bowen Li, Haochen Ma, Yuxin Wang, Jie Yang, Xinchi Chen, Xuanjing Huang, Yining Zheng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19502)  

**Abstract**: The rapid adoption of Large Language Models (LLMs) has spurred interest in automated peer review; however, progress is currently stifled by benchmarks that treat reviewing primarily as a rating prediction task. We argue that the utility of a review lies in its textual justification--its arguments, questions, and critique--rather than a scalar score. To address this, we introduce Beyond Rating, a holistic evaluation framework that assesses AI reviewers across five dimensions: Content Faithfulness, Argumentative Alignment, Focus Consistency, Question Constructiveness, and AI-Likelihood. Notably, we propose a Max-Recall strategy to accommodate valid expert disagreement and introduce a curated dataset of paper with high-confidence reviews, rigorously filtered to remove procedural noise. Extensive experiments demonstrate that while traditional n-gram metrics fail to reflect human preferences, our proposed text-centric metrics--particularly the recall of weakness arguments--correlate strongly with rating accuracy. These findings establish that aligning AI critique focus with human experts is a prerequisite for reliable automated scoring, offering a robust standard for future research. 

---
# Rank-Turbulence Delta and Interpretable Approaches to Stylometric Delta Metrics 

**Authors**: Dmitry Pronin, Evgeny Kazartsev  

**Link**: [PDF](https://arxiv.org/pdf/2604.19499)  

**Abstract**: This article introduces two new measures for authorship attribution - Rank-Turbulence Delta and Jensen-Shannon Delta - which generalise Burrows's classical Delta by applying distance functions designed for probabilistic distributions. We first set out the theoretical basis of the measures, contrasting centred and uncentred z-scoring of word-frequency vectors and re-casting the uncentred vectors as probability distributions. Building on this representation, we develop a token-level decomposition that renders every Delta distance numerically interpretable, thereby facilitating close reading and the validation of results. The effectiveness of the methods is assessed on four literary corpora in English, German, French and Russian. The English, German and French datasets are compiled from Project Gutenberg, whereas the Russian benchmark is the SOCIOLIT corpus containing 755 works by 180 authors spanning the eighteenth to the twenty-first centuries. Rank-Turbulence Delta attains attribution accuracy comparable with Cosine Delta; Jensen-Shannon Delta consistently matches or exceeds the performance of canonical Burrows's Delta. Finally, several established attribution algorithms are re-evaluated on the extended SOCIOLIT corpus. 

---
# LePREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues 

**Authors**: Fanyu Wang, Xiaoxi Kang, Paul Burgess, Aashish Srivastava, Chetan Arora, Adnan Trakic, Lay-Ki Soon, Md Khalid Hossain, Lizhen Qu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19464)  

**Abstract**: More than half of the global population struggles to meet their civil justice needs due to limited legal resources. While Large Language Models (LLMs) have demonstrated impressive reasoning capabilities, significant challenges remain even at the foundational step of legal issue identification. To investigate LLMs' capabilities in this task, we constructed a dataset from 769 real-world Malaysian Contract Act court cases, using GPT-4o to extract facts and generate candidate legal issues, annotated by senior legal experts, which reveals a critical limitation: while LLMs generate diverse issue candidates, their precision remains inadequate (GPT-4o achieves only 62%). To address this gap, we propose LePREC (Legal Professional-inspired Reasoning Elicitation and Classification), a neuro-symbolic framework combining neural generation with structured statistical reasoning. LePREC consists of: (1) a neuro component leverages LLMs to transform legal descriptions into question-answer pairs representing diverse analytical factors, and (2) a symbolic component applies sparse linear models over these discrete features, learning explicit algebraic weights that identify the most informative reasoning factors. Unlike end-to-end neural approaches, LePREC achieves interpretability through transparent feature weighting while maintaining data efficiency through correlation-based statistical classification. Experiments show a 30-40% improvement over advanced LLM baselines, including GPT-4o and Claude, confirming that correlation-based factor-issue analysis offers a more data-efficient solution for relevance decisions. 

---
# 'The Order in the Horse's Heart': A Case Study in LLM-Assisted Stylometry for the Discovery of Biblical Allusion in Modern Literary Fiction 

**Authors**: Ewan Cameron  

**Link**: [PDF](https://arxiv.org/pdf/2604.19447)  

**Abstract**: We present a dual-track pipeline for detecting biblical allusions in literary fiction and apply it to the novels of Cormac McCarthy. A bottom-up embedding track uses inverse document frequency to identify rare vocabulary shared with the King James Bible, embeds occurrences in their local context for sense disambiguation, and passes candidate passage pairs through cascaded LLM review. A top-down register track asks an LLM to read McCarthy's prose undirected to any specific biblical passage for comparison, catching allusions not distinguished by word or phrase rarity. Both tracks are cross-validated by a long-context model that holds entire novels alongside the KJV in a single pass, and every finding is checked against published scholarship. Restricting attention to allusions that carry a textual echo--shared phrasing, reworked vocabulary, or transplanted cadence--and distinguishing literary allusions proper from signposted biblical references (similes naming biblical figures, characters overtly citing scripture), the pipeline surfaces 349 allusions across the corpus. Among a target set of 115 previously documented allusions retrieved through human review of the academic literature, the pipeline independently recovers 62 (54% recall), with recall varying by connection type from 30% (transformed imagery) to 80% (register collisions). We contextualise these results with respect to the value-add from LLMs as assistants to mechanical stylometric analyses, and their potential to facilitate the statistical study of intertextuality in massive literary corpora. 

---
# What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search 

**Authors**: Xinhao Zhang, Xi Chen, François Portet, Maxime Peyrard  

**Link**: [PDF](https://arxiv.org/pdf/2604.19440)  

**Abstract**: Recent work has demonstrated the promise of orchestrating large language models (LLMs) within evolutionary and agentic optimization systems. However, the mechanisms driving these optimization gains remain poorly understood. In this work, we present a large-scale study of LLM-guided evolutionary search, collecting optimization trajectories for 15 LLMs across 8 tasks. Although zero-shot problem-solving ability correlates with final optimization outcomes, it explains only part of the variance: models with similar initial capability often induce dramatically different search trajectories and outcomes. By analyzing these trajectories, we find that strong LLM optimizers behave as local refiners, producing frequent incremental improvements while progressively localizing the search in semantic space. Conversely, weaker optimizers exhibit large semantic drift, with sporadic breakthroughs followed by stagnation. Notably, various measures of solution novelty do not predict final performance; novelty is beneficial only when the search remains sufficiently localized around high-performing regions of the solution space. Our results highlight the importance of trajectory analysis for understanding and improving LLM-based optimization systems and provide actionable insights for their design and training. 

---
# Lost in Translation: Do LVLM Judges Generalize Across Languages? 

**Authors**: Md Tahmid Rahman Laskar, Mohammed Saidul Islam, Mir Tafseer Nayeem, Amran Bhuiyan, Mizanur Rahman, Shafiq Joty, Enamul Hoque, Jimmy Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19405)  

**Abstract**: Automatic evaluators such as reward models play a central role in the alignment and evaluation of large vision-language models (LVLMs). Despite their growing importance, these evaluators are almost exclusively assessed on English-centric benchmarks, leaving open the question of how well these evaluators generalize across languages. To answer this question, we introduce MM-JudgeBench, the first large-scale benchmark for multilingual and multimodal judge model evaluation, which includes over 60K pairwise preference instances spanning 25 typologically diverse languages. MM-JudgeBench integrates two complementary subsets: a general vision-language preference evaluation subset extending VL-RewardBench, and a chart-centric visual-text reasoning subset derived from OpenCQA, enabling systematic analysis of reward models (i.e., LVLM judges) across diverse settings. We additionally release a multilingual training set derived from MM-RewardBench, disjoint from our evaluation data, to support domain adaptation. By evaluating 22 LVLMs (15 open-source, 7 proprietary), we uncover substantial cross-lingual performance variance in our proposed benchmark. Our analysis further shows that model size and architecture are poor predictors of multilingual robustness, and that even state-of-the-art LVLM judges exhibit inconsistent behavior across languages. Together, these findings expose fundamental limitations of current reward modeling and underscore the necessity of multilingual, multimodal benchmarks for developing reliable automated evaluators. 

---
# Does Self-Consistency Improve the Recall of Encyclopedic Knowledge? 

**Authors**: Sho Hoshino, Ukyo Honda, Peinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19395)  

**Abstract**: While self-consistency is known to improve performance on symbolic reasoning, its effect on the recall of encyclopedic knowledge is unclear due to a lack of targeted evaluation grounds. To address this, we establish such a knowledge recall split for the popular MMLU benchmark by applying a data-driven heuristic from prior work. We validate this split by showing that the performance patterns on the symbolic reasoning and knowledge recall subsets mirror those of GSM8K and MedMCQA, respectively. Using this solid ground, we find that self-consistency consistently improves performance across both symbolic reasoning and knowledge recall, even though its underlying CoT prompting is primarily effective for symbolic reasoning. As a result, we achieve an 89\% accuracy on MMLU, the best performance to date with the use of GPT-4o. 

---
# Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain? 

**Authors**: Niclas Doll, Jasper Schulze Buschhoff, Shalaka Satheesh, Hammam Abdelwahab, Héctor Allende-Cid, Katrin Klug  

**Link**: [PDF](https://arxiv.org/pdf/2604.19394)  

**Abstract**: This paper narrows the performance gap between small, specialized models and significantly larger general-purpose models through domain adaptation via continual pre-training and merging. We address the scarcity of specialized non-English data by constructing a high-quality German medical corpus (FineMed-de) from FineWeb2. This corpus is used to continually pre-train and merge three well-known LLMs (ranging from $7B$ to $24B$ parameters), creating the DeFineMed model family. A comprehensive evaluation confirms that specialization dramatically enhances $7B$ model performance on German medical benchmarks. Furthermore, the pairwise win-rate analysis of the Qwen2.5-based models demonstrates an approximately $3.5$-fold increase in the win-rate against the much larger Mistral-Small-24B-Instruct through domain adaptation. This evidence positions specialized $7B$ models as a competitive, resource-efficient solution for complex medical instruction-following tasks. While model merging successfully restores instruction-following abilities, a subsequent failure mode analysis reveals inherent trade-offs, including the introduction of language mixing and increased verbosity, highlighting the need for more targeted fine-tuning in future work. This research provides a robust, compliant methodology for developing specialized LLMs, serving as the foundation for practical use in German-speaking healthcare contexts. 

---
# DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing 

**Authors**: Jinyu Guo, Zhihan Zhang, Yutong Li, Jiehui Xie, Md. Tamim Iqbal, Dongshen Han, Lik-Hang Lee, Sung-Ho Bae, Jie Zou, Yang Yang, Chaoning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19351)  

**Abstract**: The quadratic computational complexity of the standard attention mechanism constitutes a fundamental bottleneck for large language models in long-context inference. While existing KV cache compression methods alleviate memory pressure, they often sacrifice generation quality and fail to address the high overhead of floating-point arithmetic. This paper introduces DASH-KV, an innovative acceleration framework that reformulates attention as approximate nearest-neighbor search via asymmetric deep hashing. Under this paradigm, we design an asymmetric encoding architecture that differentially maps queries and keys to account for their distinctions in precision and reuse characteristics. To balance efficiency and accuracy, we further introduce a dynamic mixed-precision mechanism that adaptively retains full-precision computation for critical tokens. Extensive experiments on LongBench demonstrate that DASH-KV significantly outperforms state-of-the-art baseline methods while matching the performance of full attention, all while reducing inference complexity from O(N^2) to linear O(N). The code is available at this https URL 

---
# Are Large Language Models Economically Viable for Industry Deployment? 

**Authors**: Abdullah Mohammad, Sushant Kumar Ray, Pushkar Arora, Rafiq Ali, Ebad Shabbir, Gautam Siddharth Kashyap, Jiechao Gao, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2604.19342)  

**Abstract**: Generative AI-powered by Large Language Models (LLMs)-is increasingly deployed in industry across healthcare decision support, financial analytics, enterprise retrieval, and conversational automation, where reliability, efficiency, and cost control are critical. In such settings, models must satisfy strict constraints on energy, latency, and hardware utilization-not accuracy alone. Yet prevailing evaluation pipelines remain accuracy-centric, creating a Deployment-Evaluation Gap-the absence of operational and economic criteria in model assessment. To address this gap, we present EDGE-EVAL-a industry-oriented benchmarking framework that evaluates LLMs across their full lifecycle on legacy NVIDIA Tesla T4 GPUs. Benchmarking LLaMA and Qwen variants across three industrial tasks, we introduce five deployment metrics-Economic Break-Even (Nbreak), Intelligence-Per-Watt (IPW ), System Density (\r{ho}sys), Cold-Start Tax (Ctax), and Quantization Fidelity (Qret)-capturing profitability, energy efficiency, hardware scaling, serverless feasibility, and compression safety. Our results reveal a clear efficiency frontier-models in the <2B parameter class dominate larger baselines across economic and ecological dimensions. LLaMA-3.2-1B (INT4) achieves ROI break-even in 14 requests (median), delivers 3x higher energy-normalized intelligence than 7B models, and exceeds 6,900 tokens/s/GB under 4-bit quantization. We further uncover an efficiency anomaly-while QLoRA reduces memory footprint, it increases adaptation energy by up to 7x for small models-challenging prevailing assumptions about quantization-aware training in edge deployment. 

---
# Evaluating LLM-Driven Summarisation of Parliamentary Debates with Computational Argumentation 

**Authors**: Eoghan Cunningham, Derek Greene, James Cross, Antonio Rago  

**Link**: [PDF](https://arxiv.org/pdf/2604.19331)  

**Abstract**: Understanding how policy is debated and justified in parliament is a fundamental aspect of the democratic process. However, the volume and complexity of such debates mean that outside audiences struggle to engage. Meanwhile, Large Language Models (LLMs) have been shown to enable automated summarisation at scale. While summaries of debates can make parliamentary procedures more accessible, evaluating whether these summaries faithfully communicate argumentative content remains challenging. Existing automated summarisation metrics have been shown to correlate poorly with human judgements of consistency (i.e., faithfulness or alignment between summary and source). In this work, we propose a formal framework for evaluating parliamentary debate summaries that grounds argument structures in the contested proposals up for debate. Our novel approach, driven by computational argumentation, focuses the evaluation on formal properties concerning the faithful preservation of the reasoning presented to justify or oppose policy outcomes. We demonstrate our methods using a case-study of debates from the European Parliament and associated LLM-driven summaries. 

---
# Rethinking Scale: Deployment Trade-offs of Small Language Models under Agent Paradigms 

**Authors**: Xinlin Wang, Mats Brorsson  

**Link**: [PDF](https://arxiv.org/pdf/2604.19299)  

**Abstract**: Despite the impressive capabilities of large language models, their substantial computational costs, latency, and privacy risks hinder their widespread deployment in real-world applications. Small Language Models (SLMs) with fewer than 10 billion parameters present a promising alternative; however, their inherent limitations in knowledge and reasoning curtail their effectiveness. Existing research primarily focuses on enhancing SLMs through scaling laws or fine-tuning strategies while overlooking the potential of using agent paradigms, such as tool use and multi-agent collaboration, to systematically compensate for the inherent weaknesses of small models. To address this gap, this paper presents the first large-scale, comprehensive study of <10B open-source models under three paradigms: (1) the base model, (2) a single agent equipped with tools, and (3) a multi-agent system with collaborative capabilities. Our results show that single-agent systems achieve the best balance between performance and cost, while multi-agent setups add overhead with limited gains. Our findings highlight the importance of agent-centric design for efficient and trustworthy deployment in resource-constrained settings. 

---
# IndiaFinBench: An Evaluation Benchmark for Large Language Model Performance on Indian Financial Regulatory Text 

**Authors**: Rajveer Singh Pall  

**Link**: [PDF](https://arxiv.org/pdf/2604.19298)  

**Abstract**: We introduce IndiaFinBench, to our knowledge the first publicly available evaluation benchmark for assessing large language model (LLM) performance on Indian financial regulatory text. Existing financial NLP benchmarks draw exclusively from Western financial corpora (SEC filings, US earnings reports, and English-language financial news), leaving a significant gap in coverage of non-Western regulatory frameworks. IndiaFinBench addresses this gap with 406 expert-annotated question-answer pairs drawn from 192 documents sourced from the Securities and Exchange Board of India (SEBI) and the Reserve Bank of India (RBI), spanning four task types: regulatory interpretation (174 items), numerical reasoning (92 items), contradiction detection (62 items), and temporal reasoning (78 items). Annotation quality is validated through a model-based secondary pass (kappa=0.918 on contradiction detection) and a 60-item human inter-annotator agreement evaluation (kappa=0.611; 76.7% overall agreement). We evaluate twelve models under zero-shot conditions, with accuracy ranging from 70.4% (Gemma 4 E4B) to 89.7% (Gemini 2.5 Flash). All models substantially outperform a non-specialist human baseline of 60.0%. Numerical reasoning is the most discriminative task, with a 35.9 percentage-point spread across models. Bootstrap significance testing (10,000 resamples) reveals three statistically distinct performance tiers. The dataset, evaluation code, and all model outputs are available at this https URL 

---
# Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs 

**Authors**: Guy Mor-Lan, Omer Goldman, Matan Eyal, Adi Mayrav Gilady, Sivan Eiger, Idan Szpektor, Avinatan Hassidim, Yossi Matias, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2604.19292)  

**Abstract**: Multilingual large language models (LLMs) have minimized the fluency gap between languages. This advancement, however, exposes models to the risk of biased behavior, as knowledge and norms may propagate across languages. In this work, we aim to quantify models' inter- and intra-lingual biases, via their ability to answer locale-ambiguous questions. To this end, we present LocQA, a test set containing 2,156 questions in 12 languages, referring to various locale-dependent facts such as laws, dates, and measurements. The questions do not contain indications of the locales they relate to, other than the querying language itself. LLMs' responses to LocQA locale-ambiguous questions thus reveal models' implicit priors. We used LocQA to evaluate 32 models, and detected two types of structural biases. Inter-lingually, we show a global bias towards answers relevant to the US-locale, even when models are asked in languages other than English. Moreover, we discovered that this global bias is exacerbated in models that underwent instruction tuning, compared to their base counterparts. Intra-lingually, we show that when multiple locales are relevant for the same language, models act as demographic probability engines, prioritizing locales with larger populations. Taken together, insights from LocQA may help in shaping LLMs' desired local behavior, and in quantifying the impact of various training phases on different kinds of biases. 

---
# HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human-LLM Collaborative Writing 

**Authors**: Euntae Kim, Soomin Han, Buru Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19274)  

**Abstract**: Large language models (LLMs) are increasingly used as co-authors in collaborative writing, where users begin with rough drafts and rely on LLMs to complete, revise, and refine their content. However, this capability poses a serious safety risk: malicious users could jailbreak the models-filling incomplete drafts with dangerous content-to force them into generating harmful outputs. In this paper, we identify the vulnerability of current LLMs to such draft-based co-authoring jailbreak attacks and introduce HarDBench, a systematic benchmark designed to evaluate the robustness of LLMs against this emerging threat. HarDBench spans a range of high-risk domains-including Explosives, Drugs, Weapons, and Cyberattacks-and features prompts with realistic structure and domain-specific cues to assess the model susceptibility to harmful completions. To mitigate this risk, we introduce a safety-utility balanced alignment approach based on preference optimization, training models to refuse harmful completions while remaining helpful on benign drafts. Experimental results show that existing LLMs are highly vulnerable in co-authoring contexts and our alignment method significantly reduces harmful outputs without degrading performance on co-authoring capabilities. This presents a new paradigm for evaluating and aligning LLMs in human-LLM collaborative writing settings. Our new benchmark and dataset are available on our project page at this https URL 

---
# CulturALL: Benchmarking Multilingual and Multicultural Competence of LLMs on Grounded Tasks 

**Authors**: Peiqin Lin, Chenyang Lyu, Wenjiang Luo, Haotian Ye, Md Mehrab Hossain, Chunlan Ma, Shaoxiong Ji, Younes Samih, Bo Zeng, Fan Jiang, Yuanbin Cao, Dilda Duisenbek, Adrian Neo Sau Xun, Daria Pozdniakova, Liubou Misevich, Nevena Marinković, Ngoc Gia Linh Nguyen, Thi Khanh Linh Do, Sarakmatak Sophy, Baotian Hu, Guanhua Chen, Gongbo Tang, Alham Fikri Aji, Longyue Wang, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.19262)  

**Abstract**: Large language models (LLMs) are now deployed worldwide, inspiring a surge of benchmarks that measure their multilingual and multicultural abilities. However, these benchmarks prioritize generic language understanding or superficial cultural trivia, leaving the evaluation of grounded tasks -- where models must reason within real-world, context-rich scenarios -- largely unaddressed. To fill this gap, we present CulturALL, a comprehensive and challenging benchmark to assess LLMs' multilingual and multicultural competence on grounded tasks. CulturALL is built via a human--AI collaborative framework: expert annotators ensure appropriate difficulty and factual accuracy, while LLMs lighten the manual workload. By incorporating diverse sources, CulturALL ensures comprehensive scenario coverage. Each item is carefully designed to present a high level of difficulty, making CulturALL challenging. CulturALL contains 2,610 samples in 14 languages from 51 regions, distributed across 16 topics to capture the full breadth of grounded tasks. Experiments show that the best LLM achieves 44.48% accuracy on CulturALL, underscoring substantial room for improvement. 

---
# Towards a Linguistic Evaluation of Narratives: A Quantitative Stylistic Framework 

**Authors**: Alessandro Maisto  

**Link**: [PDF](https://arxiv.org/pdf/2604.19261)  

**Abstract**: The evaluation of narrative quality remains a complex challenge, as it involves subjective factors such as plot, character development, and emotional impact. This work proposes a quantitative approach to narrative assessment by focusing on the linguistic dimension as a primary indicator of quality. The paper presents a methodology for the automatic evaluation of narrative based on the extraction of a comprehensive set of 33 quantitative linguistic features categorized into lexical, syntactic, and semantic groups. To test the model, an experiment was conducted on a specialized corpus of 23 books, including canonical masterpieces and self-published works. Through a similarity matrix, the system successfully clustered the narratives, distinguishing almost perfectly between professionally edited and self-published texts. Furthermore, the methodology was validated against a human-annotated dataset; it significantly outperforms traditional story-level evaluation metrics, demonstrating the effectiveness of quantitative linguistic features in assessing narrative quality. 

---
# ShadowPEFT: Shadow Network for Parameter-Efficient Fine-Tuning 

**Authors**: Xianming Li, Zongxi Li, Tsz-fung Andrew Lee, Jing Li, Haoran Xie, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.19254)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) reduces the training cost of full-parameter fine-tuning for large language models (LLMs) by training only a small set of task-specific parameters while freezing the pretrained backbone. However, existing approaches, such as Low-Rank Adaptation (LoRA), achieve adaptation by inserting independent low-rank perturbations directly to individual weights, resulting in a local parameterization of adaptation. We propose ShadowPEFT, a centralized PEFT framework that instead performs layer-level refinement through a depth-shared shadow module. At each transformer layer, ShadowPEFT maintains a parallel shadow state and evolves it repeatedly for progressively richer hidden states. This design shifts adaptation from distributed weight-space perturbations to a shared layer-space refinement process. Since the shadow module is decoupled from the backbone, it can be reused across depth, independently pretrained, and optionally deployed in a detached mode, benefiting edge computing scenarios. Experiments on generation and understanding benchmarks show that ShadowPEFT matches or outperforms LoRA and DoRA under comparable trainable-parameter budgets. Additional analyses on shadow pretraining, cross-dataset transfer, parameter scaling, inference latency, and system-level evaluation suggest that centralized layer-space adaptation is a competitive and flexible alternative to conventional low-rank PEFT. 

---
# Talking to a Know-It-All GPT or a Second-Guesser Claude? How Repair reveals unreliable Multi-Turn Behavior in LLMs 

**Authors**: Clara Lachenmaier, Hannah Bultmann, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2604.19245)  

**Abstract**: Repair, an important resource for resolving trouble in human-human conversation, remains underexplored in human-LLM interaction. In this study, we investigate how LLMs engage in the interactive process of repair in multi-turn dialogues around solvable and unsolvable math questions. We examine whether models initiate repair themselves and how they respond to user-initiated repair. Our results show strong differences across models: reactions range from being almost completely resistant to (appropriate) repair attempts to being highly susceptible and easily manipulated. We further demonstrate that once conversations extend beyond a single turn, model behavior becomes more distinctive and less predictable across systems. Overall, our findings indicate that each tested LLM exhibits its own characteristic form of unreliability in the context of repair. 

---
# Headlines You Won't Forget: Can Pronoun Insertion Increase Memorability? 

**Authors**: Selina Meyer, Magdalena Abel, Michael Roth  

**Link**: [PDF](https://arxiv.org/pdf/2604.19189)  

**Abstract**: For news headlines to influence beliefs and drive action, relevant information needs to be retained and retrievable from memory. In this probing study we draw on experiment designs from cognitive psychology to examine how a specific linguistic feature, namely direct address through first- and second-person pronouns, affects memorability and to what extent it is feasible to use large language models for the targeted insertion of such a feature into existing text without changing its core meaning. Across three controlled memorization experiments with a total of 240 participants, yielding 7,680 unique memory judgments, we show that pronoun insertion has mixed effects on memorability. Exploratory analyses indicate that effects differ based on headline topic, how pronouns are inserted and their immediate contexts. Additional data and fine-grained analysis is needed to draw definitive conclusions on these mediating factors. We further show that automatic revisions by LLMs are not always appropriate: Crowdsourced evaluations find many of them to be lacking in content accuracy and emotion retention or resulting in unnatural writing style. We make our collected data available for future work. 

---
# SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization 

**Authors**: Bo-Jyun Wang, Ying-Jia Lin, Hung-Yu Kao  

**Link**: [PDF](https://arxiv.org/pdf/2604.19185)  

**Abstract**: Small language models (SLMs), such as BART, can achieve summarization performance comparable to large language models (LLMs) via distillation. However, existing LLM-based ranking strategies for summary candidates suffer from instability, while classical metrics (e.g., ROUGE) are insufficient to rank high-quality summaries. To address these issues, we introduce \textbf{SCURank}, a framework that enhances summarization by leveraging \textbf{Summary Content Units (SCUs)}. Instead of relying on unstable comparisons or surface-level overlap, SCURank evaluates summaries based on the richness and semantic importance of information content. We investigate the effectiveness of SCURank in distilling summaries from multiple diverse LLMs. Experimental results demonstrate that SCURank outperforms traditional metrics and LLM-based ranking methods across evaluation measures and datasets. Furthermore, our findings show that incorporating diverse LLM summaries enhances model abstractiveness and overall distilled model performance, validating the benefits of information-centric ranking in multi-LLM distillation. The code for SCURank is available at this https URL. 

---
# Mind the Unseen Mass: Unmasking LLM Hallucinations via Soft-Hybrid Alphabet Estimation 

**Authors**: Hongxing Pan, Yingying Guo, Wenqing Kuang, Jiashi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19162)  

**Abstract**: This paper studies uncertainty quantification for large language models (LLMs) under black-box access, where only a small number of responses can be sampled for each query. In this setting, estimating the effective semantic alphabet size--that is, the number of distinct meanings expressed in the sampled responses--provides a useful proxy for downstream risk. However, frequency-based estimators tend to undercount rare semantic modes when the sample size is small, while graph-spectral quantities alone are not designed to estimate semantic occupancy accurately. To address this issue, we propose SHADE (Soft-Hybrid Alphabet Dynamic Estimator), a simple and interpretable estimator that combines Generalized Good-Turing coverage with a heat-kernel trace of the normalized Laplacian constructed from an entailment-weighted graph over sampled responses. The estimated coverage adaptively determines the fusion rule: under high coverage, SHADE uses a convex combination of the two signals, while under low coverage it applies a LogSumExp fusion to emphasize missing or weakly observed semantic modes. A finite-sample correction is then introduced to stabilize the resulting cardinality estimate before converting it into a coverage-adjusted semantic entropy score. Experiments on pooled semantic alphabet-size estimation against large-sample references and on QA incorrectness detection show that SHADE achieves the strongest improvements in the most sample-limited regime, while the performance gap narrows as the number of samples increases. These results suggest that hybrid semantic occupancy estimation is particularly beneficial when black-box uncertainty quantification must operate under tight sampling budgets. 

---
# Voice of India: A Large-Scale Benchmark for Real-World Speech Recognition in India 

**Authors**: Kaushal Bhogale, Manas Dhir, Amritansh Walecha, Manmeet Kaur, Vanshika Chhabra, Aaditya Pareek, Hanuman Sidh, Sagar Jain, Bhaskar Singh, Utkarsh Singh, Tahir Javed, Shobhit Banga, Mitesh M. Khapra  

**Link**: [PDF](https://arxiv.org/pdf/2604.19151)  

**Abstract**: Existing Indic ASR benchmarks often use scripted, clean speech and leaderboard driven evaluation that encourages dataset specific overfitting. In addition, strict single reference WER penalizes natural spelling variation in Indian languages, including non standardized spellings of code-mixed English origin words. To address these limitations, we introduce Voice of India, a closed source benchmark built from unscripted telephonic conversations covering 15 major Indian languages across 139 regional clusters. The dataset contains 306230 utterances, totaling 536 hours of speech from 36691 speakers with transcripts accounting for spelling variations. We also analyze performance geographically at the district level, revealing disparities. Finally, we provide detailed analysis across factors such as audio quality, speaking rate, gender, and device type, highlighting where current ASR systems struggle and offering insights for improving real world Indic ASR systems. 

---
# How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs for Quantitative Reasoning 

**Authors**: Haoyang Chen, Yi Liu, Jianzhi Shao, Tao Zhang, Chengfu Huo, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19149)  

**Abstract**: Thinking LLMs produce reasoning traces before answering. Prior activation steering work mainly targets on shaping these traces. It remains less understood how answer tokens actually read and integrate the reasoning to produce reliable outcomes. Focusing on quantitative reasoning, we analyze the answer-to-reasoning attention and observe a benign self-reading pattern aligned with correctness, characterized by a forward drift of the reading focus along the reasoning trace and a persistent concentration on key semantic anchors, whereas incorrect solutions exhibit diffuse and irregular attention pattern. We interpret this as internal certainty during answer decoding, where the model commits to a viable solution branch and integrates key evidence. Following this, we propose a training-free steering method driven by Self-Reading Quality (SRQ) scores combining geometric metrics for process control with semantic metrics for content monitoring. SRQ selects data to build steering vectors that guide inference toward benign self-reading and away from uncertain and disorganized reading. Experiments show that our method yields consistent accuracy gains. 

---
# ReflectMT: Internalizing Reflection for Efficient and High-Quality Machine Translation 

**Authors**: Kunquan Li, Yingxue Zhang, Fandong Meng, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.19144)  

**Abstract**: Recent years have witnessed growing interest in applying Large Reasoning Models (LRMs) to Machine Translation (MT). Existing approaches predominantly adopt a "think-first-then-translate" paradigm. Although explicit reasoning trajectories significantly enhance translation quality, they incur prohibitive inference costs and latency. To address these limitations, we propose ReflectMT, a two-stage reflection internalization algorithm for machine translation that employs a "translate-first-think-later" paradigm. Our approach develops the model's "translate-reflect-refine" capability through reinforcement learning. In the first stage, we cultivate the model's capacity for high-quality reflection and refinement, thereby enhancing its semantic comprehension and task-specific knowledge. In the second stage, we train the model to internalize the knowledge acquired during reflection. As a result, during inference, ReflectMT operates in a direct translation mode, producing high-quality translations on the first attempt without any explicit reasoning steps. Experimental results on datasets such as WMT24 demonstrate that our model's first-pass translations during inference outperform multi-step reasoning LRMs such as DeepSeek-R1 in both automatic metrics and GPT-based evaluation, achieving a 2.16-point improvement in GPT-based translation quality evaluation while reducing token consumption by 94.33%. 

---
# The Rise of Verbal Tics in Large Language Models: A Systematic Analysis Across Frontier Models 

**Authors**: Shuai Wu, Xue Li, Yanna Feng, Yufang Li, Zhijun Wang, Ran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19139)  

**Abstract**: As Large Language Models (LLMs) continue to evolve through alignment techniques such as Reinforcement Learning from Human Feedback (RLHF) and Constitutional AI, a growing and increasingly conspicuous phenomenon has emerged: the proliferation of verbal tics -- repetitive, formulaic linguistic patterns that pervade model outputs. These range from sycophantic openers ("That's a great question!", "Awesome!") to pseudo-empathetic affirmations ("I completely understand your concern", "I'm right here to catch you") and overused vocabulary ("delve", "tapestry", "nuanced"). In this paper, we present a systematic analysis of the verbal tic phenomenon across eight state-of-the-art LLMs: GPT-5.4, Claude Opus 4.7, Gemini 3.1 Pro, Grok 4.2, Doubao-Seed-2.0-pro, Kimi K2.5, DeepSeek V3.2, and MiMo-V2-Pro. Utilizing a custom evaluation framework for standardized API-based evaluation, we assess 10,000 prompts across 10 task categories in both English and Chinese, yielding 160,000 model responses. We introduce the Verbal Tic Index (VTI), a composite metric quantifying tic prevalence, and analyze its correlation with sycophancy, lexical diversity, and human-perceived naturalness. Our findings reveal significant inter-model variation: Gemini 3.1 Pro exhibits the highest VTI (0.590), while DeepSeek V3.2 achieves the lowest (0.295). We further demonstrate that verbal tics accumulate over multi-turn conversations, are amplified in subjective tasks, and show distinct cross-lingual patterns. Human evaluation (N = 120) confirms a strong inverse relationship between sycophancy and perceived naturalness (r = -0.87, p < 0.001). These results underscore the "alignment tax" of current training paradigms and highlight the urgent need for more authentic human-AI interaction frameworks. 

---
# Construction of Knowledge Graph based on Language Model 

**Authors**: Qiubai Zhu, Qingwang Wang, Haibin Yuan, Wei Chen, Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.19137)  

**Abstract**: Knowledge Graph (KG) can effectively integrate valuable information from massive data, and thus has been rapidly developed and widely used in many fields. Traditional KG construction methods rely on manual annotation, which often consumes a lot of time and manpower. And KG construction schemes based on deep learning tend to have weak generalization capabilities. With the rapid development of Pre-trained Language Models (PLM), PLM has shown great potential in the field of KG construction. This paper provides a comprehensive review of recent research advances in the field of construction of KGs using PLM. In this paper, we explain how PLM can utilize its language understanding and generation capabilities to automatically extract key information for KGs, such as entities and relations, from textual data. In addition, We also propose a new Hyper-Relarional Knowledge Graph construction framework based on lightweight Large Language Model (LLM) named LLHKG and compares it with previous methods. Under our framework, the KG construction capability of lightweight LLM is comparable to GPT3.5. 

---
# Do Emotions Influence Moral Judgment in Large Language Models? 

**Authors**: Mohammad Saim, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19125)  

**Abstract**: Large language models have been extensively studied for emotion recognition and moral reasoning as distinct capabilities, yet the extent to which emotions influence moral judgment remains underexplored. In this work, we develop an emotion-induction pipeline that infuses emotion into moral situations and evaluate shifts in moral acceptability across multiple datasets and LLMs. We observe a directional pattern: positive emotions increase moral acceptability and negative emotions decrease it, with effects strong enough to reverse binary moral judgments in up to 20% of cases, and with susceptibility scaling inversely with model capability. Our analysis further reveals that specific emotions can sometimes behave contrary to what their valence would predict (e.g., remorse paradoxically increases acceptability). A complementary human annotation study shows humans do not exhibit these systematic shifts, indicating an alignment gap in current LLMs. 

---
# Detoxification for LLM: From Dataset Itself 

**Authors**: Wei Shao, Yihang Wang, Gaoyu Zhu, Ziqiang Cheng, Lei Yu, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.19124)  

**Abstract**: Existing detoxification methods for large language models mainly focus on post-training stage or inference time, while few tackle the source of toxicity, namely, the dataset itself. Such training-based or controllable decoding approaches cannot completely suppress the model's inherent toxicity, whereas detoxifying the pretraining dataset can fundamentally reduce the toxicity that the model learns during training. Hence, we attempt to detoxify directly on raw corpora with SoCD (Soft Contrastive Decoding), which guides an LLM to localize and rewrite toxic spans in raw data while preserving semantics, in our proposed HSPD (Hierarchical Semantic-Preserving Detoxification) pipeline, yielding a detoxified corpus that can drop-in replace the original for fine-tuning or other training. On GPT2-XL, HSPD attains state-of-the-art detoxification, reducing Toxicity Probability (TP) from 0.42 to 0.18 and Expected Maximum Toxicity (EMT) from 0.43 to 0.20. We further validate consistent best-in-class results on LLaMA2-7B, OPT-6.7B, and Falcon-7B. These findings show that semantics-preserving, corpus-level rewriting with HSPD effectively suppresses downstream toxicity while retaining data utility and allowing seamless source-level mitigation, thereby reducing the cost of later model behavior adjustment. (Code is available at: this https URL) 

---
# SAHM: A Benchmark for Arabic Financial and Shari'ah-Compliant Reasoning 

**Authors**: Rania Elbadry, Sarfraz Ahmad, Ahmed Heakl, Dani Bouch, Momina Ahsan, Muhra AlMahri, Marwa Elsaid khalil, Yuxia Wang, Salem Lahlou, Sophia Ananiadou, Veselin Stoyanov, Jimin Huang, Xueqing Peng, Preslav Nakov, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.19098)  

**Abstract**: English financial NLP has progressed rapidly through benchmarks for sentiment, document understanding, and financial question answering, while Arabic financial NLP remains comparatively under-explored despite strong practical demand for trustworthy finance and Islamic-finance assistants. We introduce SAHM, a document-grounded benchmark and instruction-tuning dataset for Arabic financial NLP and Shari'ah-compliant reasoning. SAHM contains 14,380 expert-verified instances spanning seven tasks: AAOIFI standards QA, fatwa-based QA/MCQ, accounting and business exams, financial sentiment analysis, extractive summarization, and event-cause reasoning, curated from authentic regulatory, juristic, and corporate sources. We evaluate 19 strong open and proprietary LLMs using task-specific metrics and rubric-based scoring for open-ended outputs, and find that Arabic fluency does not reliably translate to evidence-grounded financial reasoning: models are substantially stronger on recognition-style tasks than on generation and causal reasoning, with the largest gaps on event-cause reasoning. We release the benchmark, evaluation framework, and an instruction-tuned model to support future research on trustworthy Arabic financial NLP. 

---
# HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing 

**Authors**: Andrew Zhuoer Feng, Cunxiang Wang, Yu Luo, Lin Fan, Yilin Zhou, Zikang Wang, Xiaotao Gu, Jie Tang, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19071)  

**Abstract**: Evaluating the writing capabilities of large language models (LLMs) remains a significant challenge due to the multidimensional nature of writing skills and the limitations of existing metrics. LLM's performance in thousand-words level and open-ended writing is inadequately assessed by traditional reference-based metrics or modern LLM-as-a-judge methods. We propose Tree-of-Writing (ToW), to resolve the implicit inconsistency often found when LLM-as-a-judge aggregates all sub-features in text evaluation. ToW incorporates a tree-structured workflow by explicitly modeling the aggregation weights of sub-features. We also present HowToBench, a large-scale Chinese writing benchmark encompassing 12 genres and 1302 instructions across three task categories: contextual completion, outline-guided writing, and open-ended generation. ToW successfully mitigates the biases, achieving a 0.93 Pearson correlation with human judgments. Furthermore, we detect that both overlap-based text generation metrics and popular LLM-as-a-judge practices are vulnerable to textual disturbances, while ToW is robust to them. We also uncover a negative correlation between input length and content-related scores in the Guide task, showcasing that it cannot be simply improved by input-side information piling. 

---
# TRN-R1-Zero: Text-rich Network Reasoning via LLMs with Reinforcement Learning Only 

**Authors**: Yilun Liu, Ruihong Qiu, Zi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19070)  

**Abstract**: Zero-shot reasoning on text-rich networks (TRNs) remains a challenging frontier, as models must integrate textual semantics with relational structure without task-specific supervision. While graph neural networks rely on fixed label spaces and supervised objectives, recent large language model (LLM)-based approaches often overlook graph context or depend on distillation from larger models, limiting generalisation. We propose TRN-R1-Zero, a post-training framework for TRN reasoning trained solely via reinforcement learning. TRN-R1-Zero directly optimises base LLMs using a Neighbour-aware Group Relative Policy Optimisation objective that dynamically adjusts rewards based on a novel margin gain metric for the informativeness of neighbouring signals, effectively guiding the model toward relational reasoning. Unlike prior methods, TRN-R1-Zero requires no supervised fine-tuning or chain-of-thought data generated from large reasoning models. Extensive experiments across citation, hyperlink, social and co-purchase TRN benchmarks demonstrate the superiority and robustness of TRN-R1-Zero. Moreover, relying strictly on node-level training, TRN-R1-Zero achieves zero-shot inference on edge- and graph-level tasks, extending beyond cross-domain transfer. The codebase is publicly available at this https URL. 

---
# Product-of-Experts Training Reduces Dataset Artifacts in Natural Language Inference 

**Authors**: Aby Mammen Mathew  

**Link**: [PDF](https://arxiv.org/pdf/2604.19069)  

**Abstract**: Neural NLI models overfit dataset artifacts instead of truly reasoning. A hypothesis-only model gets 57.7% in SNLI, showing strong spurious correlations, and 38.6% of the baseline errors are the result of these artifacts. We propose Product-of-Experts (PoE) training, which downweights examples where biased models are overconfident. PoE nearly preserves accuracy (89.10% vs. 89.30%) while cutting bias reliance by 4.71% (bias agreement 49.85% to 45%). An ablation finds lambda = 1.5 that best balances debiasing and accuracy. Behavioral tests still reveal issues with negation and numerical reasoning. 

---
# Cell-Based Representation of Relational Binding in Language Models 

**Authors**: Qin Dai, Benjamin Heinzerling, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2604.19052)  

**Abstract**: Understanding a discourse requires tracking entities and the relations that hold between them. While Large Language Models (LLMs) perform well on relational reasoning, the mechanism by which they bind entities, relations, and attributes remains unclear. We study discourse-level relational binding and show that LLMs encode it via a Cell-based Binding Representation (CBR): a low-dimensional linear subspace in which each ``cell'' corresponds to an entity--relation index pair, and bound attributes are retrieved from the corresponding cell during inference. Using controlled multi-sentence data annotated with entity and relation indices, we identify the CBR subspace by decoding these indices from attribute-token activations with Partial Least Squares regression. Across domains and two model families, the indices are linearly decodable and form a grid-like geometry in the projected space. We further find that context-specific CBR representations are related by translation vectors in activation space, enabling cross-context transfer. Finally, activation patching shows that manipulating this subspace systematically changes relational predictions and that perturbing it disrupts performance, providing causal evidence that LLMs rely on CBR for relational binding. 

---
# SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning 

**Authors**: Boyan Shi, Wei Chen, Shuyuan Zhao, Junfeng Shen, Shengnan Guo, Shaojiang Wang, Huaiyu Wan  

**Link**: [PDF](https://arxiv.org/pdf/2604.19048)  

**Abstract**: The combination of Mixture-of-Experts (MoE) and Low-Rank Adaptation (LoRA) has shown significant potential for enhancing the multi-task learning capabilities of Large Language Models. However, existing methods face two primary challenges: (1)Imprecise Routing in the current MoE-LoRA method fails to explicitly match input semantics with expert capabilities, leading to weak expert specialization. (2)Uniform weight fusion strategies struggle to provide adaptive update strengths, overlooking the varying complexity of different tasks. To address these limitations, we propose SAMoRA (Semantic-Aware Mixture of LoRA Experts), a novel parameter-efficient fine-tuning framework tailored for task-adaptive learning. Specifically, A Semantic-Aware Router is proposed to explicitly align textual semantics with the most suitable experts for precise routing. A Task-Adaptive Scaling mechanism is designed to regulate expert contributions based on specific task requirements dynamically. In addition, a novel regularization objective is proposed to jointly promote expert specialization and effective scaling. Extensive experiments on multiple multi-task benchmarks demonstrate that SAMoRA significantly outperforms the state-of-the-art methods and holds excellent task generalization capabilities. Code is available at this https URL 

---
# RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora 

**Authors**: Hanjun Cho, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.19047)  

**Abstract**: Existing QA benchmarks typically assume distinct documents with minimal overlap, yet real-world retrieval-augmented generation (RAG) systems operate on corpora such as financial reports, legal codes, and patents, where information is highly redundant and documents exhibit strong inter-document similarity. This mismatch undermines evaluation validity: retrievers can be unfairly undervalued even when they retrieve documents that provide sufficient evidence, because redundancy across documents is not accounted for in evaluation. On the other hand, retrievers that perform well on standard benchmarks often generalize poorly to real-world corpora with highly similar and redundant documents. We present RARE (Redundancy-Aware Retrieval Evaluation), a framework for constructing realistic benchmarks by (i) decomposing documents into atomic facts to enable precise redundancy tracking and (ii) enhancing LLM-based data generation with CRRF. RAG benchmark data usually requires multiple quality criteria, but LLMs often yield trivial outputs. CRRF scores criteria separately and fuses decisions by rank, improving the reliability of generated data. Applying RARE to Finance, Legal, and Patent corpora, we introduce RedQA, where a strong retriever baseline drops from 66.4% PerfRecall@10 on 4-hop General-Wiki to 5.0-27.9% PerfRecall@10 at 4-hop depth, revealing robustness gaps that current benchmarks fail to capture. RARE enables practitioners to build domain-specific RAG evaluations that faithfully reflect real-world deployment conditions. 

---
# AlignCultura: Towards Culturally Aligned Large Language Models? 

**Authors**: Gautam Siddharth Kashyap, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2604.19016)  

**Abstract**: Cultural alignment in Large Language Models (LLMs) is essential for producing contextually aware, respectful, and trustworthy outputs. Without it, models risk generating stereotyped, insensitive, or misleading responses that fail to reflect cultural diversity w.r.t Helpful, Harmless, and Honest (HHH) paradigm. Existing benchmarks represent early steps toward cultural alignment; yet, no benchmarks currently enables systematic evaluation of cultural alignment in line with UNESCO's principles of cultural diversity w.r.t HHH paradigm. Therefore, to address this gap, we built Align-Cultura, two-stage pipeline for cultural alignment. Stage I constructs CULTURAX, the HHH-English dataset grounded in the UNESCO cultural taxonomy, through Query Construction, which reclassifies prompts, expands underrepresented domains (or labels), and prevents data leakage with SimHash. Then, Response Generation pairs prompts with culturally grounded responses via two-stage rejection sampling. The final dataset contains 1,500 samples spanning 30 subdomains of tangible and intangible cultural forms. Stage II benchmarks CULTURAX on general-purpose models, culturally fine-tuned models, and open-weight LLMs (Qwen3-8B and DeepSeek-R1-Distill-Qwen-7B). Empirically, culturally fine-tuned models improve joint HHH by 4%-6%, reduce cultural failures by 18%, achieve 10%-12% efficiency gains, and limit leakage to 0.3%. 

---
# Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection 

**Authors**: Yixuan Tang, Yirui Zhang, Hang Feng, Anthony K.H. Tung  

**Link**: [PDF](https://arxiv.org/pdf/2604.19005)  

**Abstract**: Half-truths, claims that are factually correct yet misleading due to omitted context, remain a blind spot for fact verification systems focused on explicit falsehoods. Addressing such omission-based manipulation requires reasoning not only about what is said, but also about what is left unsaid. We propose RADAR, a role-anchored multi-agent debate framework for omission-aware fact verification under realistic, noisy retrieval. RADAR assigns complementary roles to a Politician and a Scientist, who reason adversarially over shared retrieved evidence, moderated by a neutral Judge. A dual-threshold early termination controller adaptively decides when sufficient reasoning has been reached to issue a verdict. Experiments show that RADAR consistently outperforms strong single- and multi-agent baselines across datasets and backbones, improving omission detection accuracy while reducing reasoning cost. These results demonstrate that role-anchored, retrieval-grounded debate with adaptive control is an effective and scalable framework for uncovering missing context in fact verification. The code is available at this https URL. 

---
# When Safety Fails Before the Answer: Benchmarking Harmful Behavior Detection in Reasoning Chains 

**Authors**: Ishita Kakkar, Enze Zhang, Rheeya Uppaal, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19001)  

**Abstract**: Large reasoning models (LRMs) produce complex, multi-step reasoning traces, yet safety evaluation remains focused on final outputs, overlooking how harm emerges during reasoning. When jailbroken, harm does not appear instantaneously but unfolds through distinct behavioral steps such as suppressing refusal, rationalizing compliance, decomposing harmful tasks, and concealing risk. However, no existing benchmark captures this process at sentence-level granularity within reasoning traces -- a key step toward reliable safety monitoring, interventions, and systematic failure diagnosis. To address this gap, we introduce HarmThoughts, a benchmark for step-wise safety evaluation of reasoning traces. \ourdataset is built on our proposed harm taxonomy of 16 harmful reasoning behaviors across four functional groups that characterize how harm propagates rather than what harm is produced. The dataset consists of 56,931 sentences from 1,018 reasoning traces generated by four model families, each annotated with fine-grained sentence-level behavioral labels. Using HarmThoughts, we analyze harm propagation patterns across reasoning traces, identifying common behavioral trajectories and drift points where reasoning transitions from safe to unsafe. Finally, we systematically compare white-box and black-box detectors on the task of identifying harmful reasoning behaviours on HarmThoughts. Our results show that existing detectors struggle with fine-grained behavior detection in reasoning traces, particularly for nuanced categories within harm emergence and execution, highlighting a critical gap in process-level safety monitoring. HarmThoughts is available publicly at: this https URL 

---
# $R^2$-dLLM: Accelerating Diffusion Large Language Models via Spatio-Temporal Redundancy Reduction 

**Authors**: Zhenbang Du, Kejing Xia, Xinrui Zhong, Yonggan Fu, Nicolai Oswald, Binfei Ji, Brucek Khailany, Pavlo Molchanov, Yingyan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.18995)  

**Abstract**: Diffusion Large Language Models (dLLMs) have emerged as a promising alternative to autoregressive generation by enabling parallel token prediction. However, practical dLLM decoding still suffers from high inference latency, which limits deployment. In this work, we observe that a substantial part of this inefficiency comes from recurring redundancy in the decoding process, including spatial redundancy caused by confidence clusters and positional ambiguity, and temporal redundancy caused by repeatedly remasking predictions that have already stabilized. Motivated by these patterns, we propose $R^2$-dLLM, a unified framework for reducing decoding redundancy from both inference and training perspectives. At inference time, we introduce training-free decoding rules that aggregate local confidence and token predictions, and finalize temporally stable tokens to avoid redundant decoding steps. We further propose a redundancy-aware supervised fine-tuning pipeline that aligns the model with efficient decoding trajectories and reduces reliance on manually tuned thresholds. Experiments demonstrate that $R^2$-dLLM consistently reduces the number of decoding steps by up to 75% compared to existing decoding strategies, while maintaining competitive generation quality across different models and tasks. These results validate that decoding redundancy is a central bottleneck in dLLMs, and that explicitly reducing it yields substantial practical efficiency gains. 

---
# STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming 

**Authors**: MinJae Jung, YongTaek Lim, Chaeyun Kim, Junghwan Kim, Kihyun Kim, Minwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.18976)  

**Abstract**: While Large Language Models (LLMs) are widely used, they remain susceptible to jailbreak prompts that can elicit harmful or inappropriate responses. This paper introduces STAR-Teaming, a novel black-box framework for automated red teaming that effectively generates such prompts. STAR-Teaming integrates a Multi-Agent System (MAS) with a Strategy-Response Multiplex Network and employs network-driven optimization to sample effective attack strategies. This network-based approach recasts the intractable high-dimensional embedding space into a tractable structure, yielding two key advantages: it enhances the interpretability of the LLM's strategic vulnerabilities, and it streamlines the search for effective strategies by organizing the search space into semantic communities, thereby preventing redundant exploration. Empirical results demonstrate that STAR-Teaming significantly surpasses existing methods, achieving a higher attack success rate (ASR) at a lower computational cost. Extensive experiments validate the effectiveness and explainability of the Multiplex Network. The code is available at this https URL. 

---
# Assessing Capabilities of Large Language Models in Social Media Analytics: A Multi-task Quest 

**Authors**: Ramtin Davoudi, Kartik Thakkar, Nazanin Donyapour, Tyler Derr, Hamid Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2604.18955)  

**Abstract**: In this study, we present the first comprehensive evaluation of modern LLMs - including GPT-4, GPT-4o, GPT-3.5-Turbo, Gemini 1.5 Pro, DeepSeek-V3, Llama 3.2, and BERT - across three core social media analytics tasks on a Twitter (X) dataset: (I) Social Media Authorship Verification, (II) Social Media Post Generation, and (III) User Attribute Inference. For the authorship verification, we introduce a systematic sampling framework over diverse user and post selection strategies and evaluate generalization on newly collected tweets from January 2024 onward to mitigate "seen-data" bias. For post generation, we assess the ability of LLMs to produce authentic, user-like content using comprehensive evaluation metrics. Bridging Tasks I and II, we conduct a user study to measure real users' perceptions of LLM-generated posts conditioned on their own writing. For attribute inference, we annotate occupations and interests using two standardized taxonomies (IAB Tech Lab 2023 and 2018 U.S. SOC) and benchmark LLMs against existing baselines. Overall, our unified evaluation provides new insights and establishes reproducible benchmarks for LLM-driven social media analytics. The code and data are provided in the supplementary material and will also be made publicly available upon publication. 

---
# A Mechanism and Optimization Study on the Impact of Information Density on User-Generated Content Named Entity Recognition 

**Authors**: Jiang Xiaobo, Dinghong Lai, Song Qiu, Yadong Deng, Xinkai Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2604.18944)  

**Abstract**: Named Entity Recognition (NER) models trained on clean, high-resource corpora exhibit catastrophic performance collapse when deployed on noisy, sparse User-Generated Content (UGC), such as social media. Prior research has predominantly focused on point-wise symptom remediation -- employing customized fine-tuning to address issues like neologisms, alias drift, non-standard orthography, long-tail entities, and class imbalance. However, these improvements often fail to generalize because they overlook the structural sparsity inherent in UGC. This study reveals that surface-level noise symptoms share a unified root cause: low Information Density (ID). Through hierarchical confounding-controlled resampling experiments (specifically controlling for entity rarity and annotation consistency), this paper identifies ID as an independent key factor. We introduce Attention Spectrum Analysis (ASA) to quantify how reduced ID causally leads to ``attention blunting,'' ultimately degrading NER performance. Informed by these mechanistic insights, we propose the Window-Aware Optimization Module (WOM), an LLM-empowered, model-agnostic framework. WOM identifies information-sparse regions and utilizes selective back-translation to directionally enhance semantic density without altering model architecture. Deployed atop mainstream architectures on standard UGC datasets (WNUT2017, Twitter-NER, WNUT2016), WOM yields up to 4.5\% absolute F1 improvement, demonstrating robustness and achieving new state-of-the-art (SOTA) results on WNUT2017. 

---
# Disparities In Negation Understanding Across Languages In Vision-Language Models 

**Authors**: Charikleia Moraitaki, Sarah Pan, Skyler Pulling, Gwendolyn Flusche, Kumail Alhamoud, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2604.18942)  

**Abstract**: Vision-language models (VLMs) exhibit affirmation bias: a systematic tendency to select positive captions ("X is present") even when the correct description contains negation ("no X"). While prior work has documented this failure mode in English and proposed solutions, negation manifests differently across languages through varying morphology, word order, and cliticization patterns, raising the question of whether these solutions serve all linguistic communities equitably. We introduce the first human-verified multilingual negation benchmark, spanning seven typologically diverse languages: English, Mandarin Chinese, Arabic, Greek, Russian, Tagalog, and Spanish. Evaluating three VLMs - CLIP, SigLIP, and MultiCLIP - we find that standard CLIP performs at or below chance on non-Latin-script languages, while MultiCLIP achieves the highest and most uniform accuracy. We also evaluate SpaceVLM, a proposed negation correction, and find that it produces substantial improvements for several languages - particularly English, Greek, Spanish, and Tagalog - while showing varied effectiveness across typologically different languages. This variation reveals that linguistic properties like morphology, script, and negation structure interact with model improvements in fairness-relevant ways. As VLMs are deployed globally, multilingual benchmarks are essential for understanding not just whether solutions work, but for whom. 

---
# Proposing Topic Models and Evaluation Frameworks for Analyzing Associations with External Outcomes: An Application to Leadership Analysis Using Large-Scale Corporate Review Data 

**Authors**: Yura Yoshida, Masato Kanai, Masataka Nakayama, Haruki Ohsawa, Yukiko Uchida, Arata Yuminaga, Gakuse Hoshina, Nobuo Sayama  

**Link**: [PDF](https://arxiv.org/pdf/2604.18919)  

**Abstract**: Analyzing topics extracted from text data in relation to external outcomes is important across fields such as computational social science and organizational research. However, existing topic modeling methods struggle to simultaneously achieve interpretability, topic specificity (alignment with concrete actions or characteristics), and polarity stance consistency (absence of mixed positive and negative evaluations within a topic). Focusing on leadership analysis using corporate review data, this study proposes a method leveraging large language models to generate topics that satisfy these properties, along with an evaluation framework tailored to external outcome analysis. The framework explicitly incorporates topic specificity and polarity stance consistency as evaluation criteria and examines automated evaluation methods based on existing metrics. Using employee reviews from OpenWork, a major corporate review platform in Japan, the proposed method achieves improved interpretability, specificity, and polarity consistency compared to existing approaches. In analyses of external outcomes such as employee morale, it also produces topics with higher explanatory power. These results suggest that the proposed method and evaluation framework provide a generalized approach for topic analysis in applications involving external outcomes. 

---
# MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation 

**Authors**: Mehul Agarwal, Aditya Aggarwal, Arnav Goel, Medha Hira, Anubha Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.18914)  

**Abstract**: While multilingual large language models (LLMs) perform well on high-level tasks like translation and question answering, their ability to handle grammatical gender and morphological agreement remains underexplored. In morphologically rich languages, gender influences verb conjugation, pronouns, and even first-person constructions with explicit and implicit mentions of gender. We introduce MORPHOGEN, a morphologically grounded large-scale benchmark dataset for evaluating gender-aware generation in three typologically diverse grammatically gendered languages: French, Arabic, and Hindi. The core task, GENFORM, requires models to rewrite a first-person sentence in the opposite gender while preserving its meaning and structure. We construct a high-quality synthetic dataset spanning these three languages and benchmark 15 popular multilingual LLMs (2B-70B) on their ability to perform this transformation. Our results reveal significant gaps and interesting insights into how current models handle morphological gender. MORPHOGEN provides a focused diagnostic lens for gender-aware language modeling and lays the groundwork for future research on inclusive and morphology-sensitive NLP. 

---
# LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval 

**Authors**: He Cheng, Yifu Wu, Saksham Khatwani, Maya Kruse, Dmitriy Dligach, Timothy A. Miller, Majid Afshar, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.18913)  

**Abstract**: Knowledge graphs (KGs) are increasingly integrated with large language models (LLMs) to provide structured, verifiable reasoning. A core operation in this integration is multi-hop retrieval, yet existing systems struggle to balance efficiency, scalability, and interpretability. We introduce LogosKG, a novel, hardware-aligned framework that enables scalable and interpretable k-hop retrieval on large KGs by building on symbolic KG formulations and executing traversal as hardware-efficient operations over decomposed subject, object, and relation representations. To scale to billion-edge graphs, LogosKG integrates degree-aware partitioning, cross-graph routing, and on-demand caching. Experiments show substantial efficiency gains over CPU and GPU baselines without loss of retrieval fidelity. With proven performance in KG retrieval, a downstream two-round KG-LLM interaction demonstrates how LogosKG enables large-scale, evidence-grounded analysis of how KG topology, such as hop distribution and connectivity, shapes the alignment between structured biomedical knowledge and LLM diagnostic reasoning, thereby opening the door for next-generation KG-LLM integration. The source code is publicly available at this https URL, and an online demo is available at this https URL. 

---
# Less Is More: Cognitive Load and the Single-Prompt Ceiling in LLM Mathematical Reasoning 

**Authors**: Manuel Israel Cazares  

**Link**: [PDF](https://arxiv.org/pdf/2604.18897)  

**Abstract**: We present a systematic empirical study of prompt engineering for formal mathematical reasoning in the context of the SAIR Equational Theories Stage 1 competition. The task requires deciding whether one equational law implies another over all magmas -- a problem that is undecidable in general but decidable for FALSE via finite model search. Over five weeks, we designed, tested, and analyzed more than 40 prompt variants, ranging from 0 to 4,878 bytes, across four evaluation splits and three language models (gpt-oss-120b, Llama 3.3 70B, Gemma 4 31B).
Our central finding is a single-prompt ceiling: despite substantial engineering effort, balanced hard accuracy plateaus in an empirical saturation region of approximately 60--79% for gpt-oss-120b, compared to a 59.75% no-cheatsheet baseline. We identify three mechanisms underlying this ceiling: (1) the mathematical undecidability of the TRUE case limits what any finite prompt can encode; (2) complex rule systems decrease performance on weaker models (Llama 3.3 70B collapses to 0% TRUE recall with prompts exceeding 2KB); and (3) prompt ordering effects interact with model attention in fragile, non-monotonic ways.
Our best submission (AN45c, 2,252 bytes) achieves 79.25% accuracy on hard3 (n=400; 95% CI: [75.0%, 82.9%]), with TRUE recall of 95.9% and FALSE recall of 63.4%, representing a +19.5 percentage-point improvement over the no-cheatsheet baseline (59.75%). We release all prompt variants, evaluation scripts, and results at this https URL 

---
# Prioritizing the Best: Incentivizing Reliable Multimodal Reasoning by Rewarding Beyond Answer Correctness 

**Authors**: Mengzhao Jia, Zhihan Zhang, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18892)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) improves multimodal reasoning by rewarding verifiable final answers. Yet answer-correct trajectories may still rely on incomplete derivations, weak evidence, or statements that contradict their conclusions. This gap between answer correctness and reasoning validity, which we call reasoning-answer inconsistency, motivates trajectory supervision in multimodal RL. We compare two main approaches: reward models (RMs), and Generative Rewards (GRs). RMs are efficient and help early in training, but their gains weaken as the policy distribution shifts; GRs improve performance, but may give unstable rewards and computationally expensive. We therefore propose Groupwise Ranking Reward, which ranks verifier-passed trajectories for the same prompt in one pass and redistributes reward accordingly. Groupwise comparison better separates stronger and weaker correct trajectories with lower judge overhead than GRs. Experiments show that RLVR aggravates reasoning-answer inconsistency, while trajectory supervision alleviates it. Groupwise Ranking Reward performs best overall, improving reliability-conditioned accuracy from 47.4% to 54.7% over RLVR. 

---
# Where Fake Citations Are Made: Tracing Field-Level Hallucination to Specific Neurons in LLMs 

**Authors**: Yuefei Chen, Yihao Quan, Xiaodong Lin, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18880)  

**Abstract**: LLMs frequently generate fictitious yet convincing citations, often expressing high confidence even when the underlying reference is wrong. We study this failure across 9 models and 108{,}000 generated references, and find that author names fail far more often than other fields across all models and settings. Citation style has no measurable effect, while reasoning-oriented distillation degrades recall. Probes trained on one field transfer at near-chance levels to the others, suggesting that hallucination signals do not generalize across fields. Building on this finding, we apply elastic-net regularization with stability selection to neuron-level CETT values of Qwen2.5-32B-Instruct and identify a sparse set of field-specific hallucination neurons (FH-neurons). Causal intervention further confirms their role: amplifying these neurons increases hallucination, while suppressing them improves performance across fields, with larger gains in some fields. These results suggest a lightweight approach to detecting and mitigating citation hallucination using internal model signals alone. 

---
# LegalBench-BR: A Benchmark for Evaluating Large Language Models on Brazilian Legal Decision Classification 

**Authors**: Pedro Barbosa de Carvalho Neto  

**Link**: [PDF](https://arxiv.org/pdf/2604.18878)  

**Abstract**: We introduce LegalBench-BR, the first public benchmark for evaluating language models on Brazilian legal text classification. The dataset comprises 3,105 appellate proceedings from the Santa Catarina State Court (TJSC), collected via the DataJud API (CNJ) and annotated across five legal areas through LLM-assisted labeling with heuristic validation. On a class-balanced test set, BERTimbau-LoRA, updating only 0.3% of model parameters, achieves 87.6% accuracy and 0.87 macro-F1 (+22pp over Claude 3.5 Haiku, +28pp over GPT-4o mini). The gap is most striking on administrativo (administrative law): GPT-4o mini scores F1 = 0.00 and Claude 3.5 Haiku scores F1 = 0.08 on this class, while the fine-tuned model reaches F1 = 0.91. Both commercial LLMs exhibit a systematic bias toward civel (civil law), absorbing ambiguous classes rather than discriminating them, a failure mode that domain-adapted fine-tuning eliminates. These results demonstrate that general-purpose LLMs cannot substitute for domain-adapted models in Brazilian legal classification, even when the task is a simple 5-class problem, and that LoRA fine-tuning on a consumer GPU closes the gap at zero marginal inference cost. We release the full dataset, model, and pipeline to enable reproducible research in Portuguese legal NLP. 

---
# Semantic Needles in Document Haystacks: Sensitivity Testing of LLM-as-a-Judge Similarity Scoring 

**Authors**: Sinan G. Aksoy, Alexandra A. Sabrio, Erik VonKaenel, Lee Burke  

**Link**: [PDF](https://arxiv.org/pdf/2604.18835)  

**Abstract**: We propose a scalable, multifactorial experimental framework that systematically probes LLM sensitivity to subtle semantic changes in pairwise document comparison. We analogize this as a needle-in-a-haystack problem: a single semantically altered sentence (the needle) is embedded within surrounding context (the hay), and we vary the perturbation type (negation, conjunction swap, named entity replacement), context type (original vs. topically unrelated), needle position, and document length across all combinations, testing five LLMs on tens of thousands of document pairs. Our analysis reveals several striking findings. First, LLMs exhibit a within-document positional bias distinct from previously studied candidate-order effects: most models penalize semantic differences more harshly when they occur earlier in a document. Second, when the altered sentence is surrounded by topically unrelated context, it systematically lowers similarity scores and induces bipolarized scores that indicate either very low or very high similarity. This is consistent with an interpretive frame account in which topically-related context may allow models to contextualize and downweight the alterations. Third, each LLM produces a qualitatively distinct scoring distribution, a stable "fingerprint" that is invariant to perturbation type, yet all models share a universal hierarchy in how leniently they treat different perturbation types. Together, these results demonstrate that LLM semantic similarity scores are sensitive to document structure, context coherence, and model identity in ways that go beyond the semantic change itself, and that the proposed framework offers a practical, LLM-agnostic toolkit for auditing and comparing scoring behavior across current and future models. 

---
# Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models 

**Authors**: Seyedali Mohammadi, Manas Gaur, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2604.18786)  

**Abstract**: Scientific feasibility assessment asks whether a claim is consistent with established knowledge and whether experimental evidence could support or refute it. We frame feasibility assessment as a diagnostic reasoning task in which, given a hypothesis, a model predicts feasible or infeasible and justifies its decision. We evaluate large language models (LLMs) under controlled knowledge conditions (hypothesis-only, with experiments, with outcomes, or both) and probe robustness by progressively removing portions of the experimental and/or outcome context. Across multiple LLMs and two datasets, providing outcome evidence is generally more reliable than providing experiment descriptions. Outcomes tend to improve accuracy beyond what internal knowledge alone provides, whereas experimental text can be brittle and may degrade performance when the context is incomplete. These findings clarify when experimental evidence benefits LLM-based feasibility assessment and when it introduces fragility. 

---
# Mango: Multi-Agent Web Navigation via Global-View Optimization 

**Authors**: Weixi Tong, Yifeng Di, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18779)  

**Abstract**: Existing web agents typically initiate exploration from the root URL, which is inefficient for complex websites with deep hierarchical structures. Without a global view of the website's structure, agents frequently fall into navigation traps, explore irrelevant branches, or fail to reach target information within a limited budget. We propose Mango, a multi-agent web navigation method that leverages the website structure to dynamically determine optimal starting points. We formulate URL selection as a multi-armed bandit problem and employ Thompson Sampling to adaptively allocate the navigation budget across candidate URLs. Furthermore, we introduce an episodic memory component to store navigation history, enabling the agent to learn from previous attempts. Experiments on WebVoyager demonstrate that Mango achieves a success rate of 63.6% when using GPT-5-mini, outperforming the best baseline by 7.3%. Furthermore, on WebWalkerQA, Mango attains a 52.5% success rate, surpassing the best baseline by 26.8%. We also demonstrate the generalizability of Mango using both open-source and closed-source models as backbones. Our data and code are open-source and available at this https URL. 

---
# An Empirical Study of Multi-Generation Sampling for Jailbreak Detection in Large Language Models 

**Authors**: Hanrui Luo, Shreyank N Gowda  

**Link**: [PDF](https://arxiv.org/pdf/2604.18775)  

**Abstract**: Detecting jailbreak behaviour in large language models remains challenging, particularly when strongly aligned models produce harmful outputs only rarely. In this work, we present an empirical study of output based jailbreak detection under realistic conditions using the JailbreakBench Behaviors dataset and multiple generator models with varying alignment strengths. We evaluate both a lexical TF-IDF detector and a generation inconsistency based detector across different sampling budgets. Our results show that single output evaluation systematically underestimates jailbreak vulnerability, as increasing the number of sampled generations reveals additional harmful behaviour. The most significant improvements occur when moving from a single generation to moderate sampling, while larger sampling budgets yield diminishing returns. Cross generator experiments demonstrate that detection signals partially generalise across models, with stronger transfer observed within related model families. A category level analysis further reveals that lexical detectors capture a mixture of behavioural signals and topic specific cues, rather than purely harmful behaviour. Overall, our findings suggest that moderate multi sample auditing provides a more reliable and practical approach for estimating model vulnerability and improving jailbreak detection in large language models. Code will be released. 

---
# Model-Agnostic Meta Learning for Class Imbalance Adaptation 

**Authors**: Hanshu Rao, Guangzeng Han, Xiaolei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18759)  

**Abstract**: Class imbalance is a widespread challenge in NLP tasks, significantly hindering robust performance across diverse domains and applications. We introduce Hardness-Aware Meta-Resample (HAMR), a unified framework that adaptively addresses both class imbalance and data difficulty. HAMR employs bi-level optimizations to dynamically estimate instance-level weights that prioritize genuinely challenging samples and minority classes, while a neighborhood-aware resampling mechanism amplifies training focus on hard examples and their semantically similar neighbors. We validate HAMR on six imbalanced datasets covering multiple tasks and spanning biomedical, disaster response, and sentiment domains. Experimental results show that HAMR achieves substantial improvements for minority classes and consistently outperforms strong baselines. Extensive ablation studies demonstrate that our proposed modules synergistically contribute to performance gains and highlight HAMR as a flexible and generalizable approach for class imbalance adaptation. Code is available at this https URL. 

---
# Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation 

**Authors**: Abhishek Purushothama, Emma Thronson, Alexia Guo, Amir Zeldes  

**Link**: [PDF](https://arxiv.org/pdf/2604.18758)  

**Abstract**: Low-resource machine translation requires methods that differ from those used for high-resource languages. This paper proposes a novel in-context learning approach to support low-resource machine translation of the Coptic language to English, with syntactic augmentation from Universal Dependencies parses of input sentences. Building on existing work using bilingual dictionaries to support inference for vocabulary items, we add several representations of syntactic analyses to our inputs , specifically exploring the inclusion of raw parser outputs, verbalizations of parses in plain English, and targeted instructions of difficult constructions identified in sub-trees and how they can be translated. Our results show that while syntactic information alone is not as useful as dictionary-based glosses, combining retrieved dictionary items with syntactic information achieves significant gains across model sizes, achieving new state-of-the-art translation results for Coptic. 

---
# Remask, Don't Replace: Token-to-Mask Refinement in Masked Diffusion Language Models 

**Authors**: Lin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.18738)  

**Abstract**: Masked diffusion language models such as LLaDA2.1 rely on Token-to-Token (T2T) editing to correct their own generation errors: whenever a different token crosses a confidence threshold, the committed token is overwritten. We identify three structural failure modes of this rule. The trigger cannot fire when no single alternative is confident enough; the replacement is computed under a context that may itself contain errors; and the uniform perturbations used to train the T2T stream do not resemble the coherent, semantically plausible mistakes that the model actually makes at inference. As an alternative, we propose Token-to-Mask (T2M) remasking. Rather than overwriting a suspect token with a new guess, T2M resets the position to the mask state, so that the next denoising step re-predicts it from an in-distribution context. The method is training-free, modifies only the editing rule, and introduces no new parameters. We pair it with three detection heuristics and give a short theoretical account of why a mask is a better conditioning signal than an erroneous token. Across 8 benchmarks, T2M improves accuracy on tasks that require exact token-level output. Its largest gain is +5.92 points on CMATH, where we attribute 79.9% of baseline errors to last-mile corruption (correct reasoning followed by a garbled final answer); T2M repairs 41.3% of these cases. 

---
# Investigating Counterfactual Unfairness in LLMs towards Identities through Humor 

**Authors**: Shubin Kim, Yejin Son, Junyeong Park, Keummin Ka, Seungbeen Lee, Jaeyoung Lee, Hyeju Jang, Alice Oh, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18729)  

**Abstract**: Humor holds up a mirror to social perception: what we find funny often reflects who we are and how we judge others. When language models engage with humor, their reactions expose the social assumptions they have internalized from training data. In this paper, we investigate counterfactual unfairness through humor by observing how the model's responses change when we swap who speaks and who is addressed while holding other factors constant. Our framework spans three tasks: humor generation refusal, speaker intention inference, and relational/societal impact prediction, covering both identity-agnostic humor and identity-specific disparagement humor. We introduce interpretable bias metrics that capture asymmetric patterns under identity swaps. Experiments across state-of-the-art models reveal consistent relational disparities: jokes told by privileged speakers are refused up to 67.5% more often, judged as malicious 64.7% more frequently, and rated up to 1.5 points higher in social harm on a 5-point scale. These patterns highlight how sensitivity and stereotyping coexist in generative models, complicating efforts toward fairness and cultural alignment. 

---
# Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP 

**Authors**: Thanmay Jayakumar, Deepon Halder, Raj Dabre  

**Link**: [PDF](https://arxiv.org/pdf/2604.18722)  

**Abstract**: Cross-lingual transfer in NLP is often hindered by the ``script barrier'' where differences in writing systems inhibit transfer learning between languages. Transliteration, the process of converting the script, has emerged as a powerful technique to bridge this gap by increasing lexical overlap. This paper provides a comprehensive survey of the application of transliteration in cross-lingual NLP. We present a taxonomy of key motivations to utilize transliterations in language models, and provide an overview of different approaches of incorporating transliterations as input. We analyze the evolution and effectiveness of these methods, discussing the critical trade-offs involved, and contextualize their need in modern LLMs. The review explores various settings that show how transliteration is beneficial, including handling code-mixed text, leveraging language family relatedness, and pragmatic gains in inference efficiency. Based on this analysis, we provide concrete recommendations for researchers on selecting and implementing the most appropriate transliteration strategy based on their specific language, task, and resource constraints. 

---
# Characterizing AlphaEarth Embedding Geometry for Agentic Environmental Reasoning 

**Authors**: Mashrekur Rahman, Samuel J. Barrett, Christina Last  

**Link**: [PDF](https://arxiv.org/pdf/2604.18715)  

**Abstract**: Earth observation foundation models encode land surface information into dense embedding vectors, yet the geometric structure of these representations and its implications for downstream reasoning remain underexplored. We characterize the manifold geometry of Google AlphaEarth's 64-dimensional embeddings across 12.1 million Continental United States samples (2017--2023) and develop an agentic system that leverages this geometric understanding for environmental reasoning. The manifold is non-Euclidean: effective dimensionality is 13.3 (participation ratio) from 64 raw dimensions, with local intrinsic dimensionality of approximately 10. Tangent spaces rotate substantially, with 84\% of locations exceeding 60\textdegree{} and local-global alignment (mean$|\cos\theta| = 0.17$) approaching the random baseline of 0.125. Supervised linear probes indicate that concept directions rotate across the manifold, and compositional vector arithmetic using both PCA-derived and probe-derived directions yields poor precision. Retrieval instead produces physically coherent results, with local geometry predicting retrieval coherence ($R^2 = 0.32$). Building on this characterization, we introduce an agentic system with nine specialized tools that decomposes environmental queries into reasoning chains over a FAISS-indexed embedding database. A five-condition ablation (120 queries, three complexity tiers) shows that embedding retrieval dominates response quality ($\mu = 3.79 \pm 0.90$ vs.\ $3.03 \pm 0.77$ parametric-only; scale 1--5), with peak performance on multi-step comparisons ($\mu = 4.28 \pm 0.43$). A cross-model benchmark show that geometric tools reduce Sonnet 4.5's score by 0.12 points but improve Opus 4.6's by 0.07, with Opus achieving higher geometric grounding (3.38 vs.\ 2.64), suggesting that the value of geometric characterization scales with the reasoning capability of the consuming model. 

---
# Probing for Reading Times 

**Authors**: Eleftheria Tsipidi, Samuel Kiegeland, Francesco Ignazio Re, Tianyang Xu, Mario Giulianelli, Karolina Stanczak, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2604.18712)  

**Abstract**: Probing has shown that language model representations encode rich linguistic information, but it remains unclear whether they also capture cognitive signals about human processing. In this work, we probe language model representations for human reading times. Using regularized linear regression on two eye-tracking corpora spanning five languages (English, Greek, Hebrew, Russian, and Turkish), we compare the representations from every model layer against scalar predictors -- surprisal, information value, and logit-lens surprisal. We find that the representations from early layers outperform surprisal in predicting early-pass measures such as first fixation and gaze duration. The concentration of predictive power in the early layers suggests that human-like processing signatures are captured by low-level structural or lexical representations, pointing to a functional alignment between model depth and the temporal stages of human reading. In contrast, for late-pass measures such as total reading time, scalar surprisal remains superior, despite its being a much more compressed representation. We also observe performance gains when using both surprisal and early-layer representations. Overall, we find that the best-performing predictor varies strongly depending on the language and eye-tracking measure. 

---
# Two-dimensional early exit optimisation of LLM inference 

**Authors**: Jan Hůla, David Adamczyk, Tomáš Filip, Martin Pavlíček, Petr Sosík  

**Link**: [PDF](https://arxiv.org/pdf/2604.18592)  

**Abstract**: We introduce a two-dimensional (2D) early exit strategy that coordinates layer-wise and sentence-wise exiting for classification tasks in large language models. By processing input incrementally sentence-by-sentence while progressively activating deeper layers, our method achieves multiplicative computational savings that exceed those from optimizing either dimension independently. Experimental evaluation across four state-of-the-art LLMs (Llama 3.1, Llama 3.2, Gemma, Qwen; 3B-8B parameters) on three sentiment classification datasets demonstrates additional speed-ups of 1.4--2.3$\times$ over optimal layer-wise early exit for simpler tasks with vanilla models, with graceful degradation on complex multi-class problems. Fine-tuning reduces but does not eliminate this advantage. The approach is model-agnostic, requires only lightweight classification adapters, and is orthogonal to complementary efficiency methods such as quantization and pruning. Our findings indicate that 2D early exit strategies excel when semantic information accumulates predictably across input structure, suggesting possible applicability to sequence-processing tasks beyond sentiment classification. 

---
# SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models 

**Authors**: Josue Torres-Fonseca, Naihao Deng, Yinpei Dai, Shane Storks, Yichi Zhang, Rada Mihalcea, Casey Kennington, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2604.19638)  

**Abstract**: Multimodal Large Language Models are increasingly adopted as autonomous agents in interactive environments, yet their ability to proactively address safety hazards remains insufficient. We introduce SafetyALFRED, built upon the embodied agent benchmark ALFRED, augmented with six categories of real-world kitchen hazards. While existing safety evaluations focus on hazard recognition through disembodied question answering (QA) settings, we evaluate eleven state-of-the-art models from the Qwen, Gemma, and Gemini families on not only hazard recognition, but also active risk mitigation through embodied planning. Our experimental results reveal a significant alignment gap: while models can accurately recognize hazards in QA settings, average mitigation success rates for these hazards are low in comparison. Our findings demonstrate that static evaluations through QA are insufficient for physical safety, thus we advocate for a paradigm shift toward benchmarks that prioritize corrective actions in embodied contexts. We open-source our code and dataset under this https URL 

---
# Diagnosable ColBERT: Debugging Late-Interaction Retrieval Models Using a Learned Latent Space as Reference 

**Authors**: François Remy  

**Link**: [PDF](https://arxiv.org/pdf/2604.19566)  

**Abstract**: Reliable biomedical and clinical retrieval requires more than strong ranking performance: it requires a practical way to find systematic model failures and curate the training evidence needed to correct them. Late-interaction models such as ColBERT provide a first solution thanks to the interpretable token-level interaction scores they expose between document and query tokens. Yet this interpretability is shallow: it explains a particular document--query pairwise score, but does not reveal whether the model has learned a clinical concept in a stable, reusable, and context-sensitive way across diverse expressions. As a result, these scores provide limited support for diagnosing misunderstandings, identifying irreasonably distant biomedical concepts, or deciding what additional data or feedback is needed to address this. In this short position paper, we propose Diagnosable ColBERT, a framework that aligns ColBERT token embeddings to a reference latent space grounded in clinical knowledge and expert-provided conceptual similarity constraints. This alignment turns document encodings into inspectable evidence of what the model appears to understand, enabling more direct error diagnosis and more principled data curation without relying on large batteries of diagnostic queries. 

---
# Enhancing Construction Worker Safety in Extreme Heat: A Machine Learning Approach Utilizing Wearable Technology for Predictive Health Analytics 

**Authors**: Syed Sajid Ullah, Amir Khan  

**Link**: [PDF](https://arxiv.org/pdf/2604.19559)  

**Abstract**: Construction workers are highly vulnerable to heat stress, yet tools that translate real-time physiological data into actionable safety intelligence remain scarce. This study addresses this gap by developing and evaluating deep learning models, specifically a baseline Long Short-Term Memory (LSTM) network and an attention-based LSTM, to predict heat stress among 19 workers in Saudi Arabia. Using Garmin Vivosmart 5 smartwatches to monitor metrics such as heart rate, HRV, and oxygen saturation, the attention-based model outperformed the baseline, achieving 95.40% testing accuracy and significantly reducing false positives and negatives. With precision, recall, and F1 scores of 0.982, this approach not only improves predictive performance but also offers interpretable results suitable for integration into IoT-enabled safety systems and BIM dashboards, advancing proactive, informatics-driven safety management in the construction industry. 

---
# Enhancing Unsupervised Keyword Extraction in Academic Papers through Integrating Highlights with Abstract 

**Authors**: Yi Xiang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19505)  

**Abstract**: Automatic keyword extraction from academic papers is a key area of interest in natural language processing and information retrieval. Although previous research has mainly focused on utilizing abstract and references for keyword extraction, this paper focuses on the highlights section - a summary describing the key findings and contributions, offering readers a quick overview of the research. Our observations indicate that highlights contain valuable keyword information that can effectively complement the abstract. To investigate the impact of incorporating highlights into unsupervised keyword extraction, we evaluate three input scenarios: using only the abstract, the highlights, and a combination of both. Experiments conducted with four unsupervised models on Computer Science (CS), Library and Information Science (LIS) datasets reveal that integrating the abstract with highlights significantly improves extraction performance. Furthermore, we examine the differences in keyword coverage and content between abstract and highlights, exploring how these variations influence extraction outcomes. The data and code are available at this https URL. 

---
# EVPO: Explained Variance Policy Optimization for Adaptive Critic Utilization in LLM Post-Training 

**Authors**: Chengjun Pan, Shichun Liu, Jiahang Lin, Dingwei Zhu, Jiazheng Zhang, Shihan Dou, Songyang Gao, Zhenhua Han, Binghai Wang, Rui Zheng, Xuanjing Huang, Tao Gui, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.19485)  

**Abstract**: Reinforcement learning (RL) for LLM post-training faces a fundamental design choice: whether to use a learned critic as a baseline for policy optimization. Classical theory favors critic-based methods such as PPO for variance reduction, yet critic-free alternatives like GRPO have gained widespread adoption due to their simplicity and competitive performance. We show that in sparse-reward settings, a learned critic can inject estimation noise that exceeds the state signal it captures, increasing rather than reducing advantage variance. By casting baseline selection as a Kalman filtering problem, we unify PPO and GRPO as two extremes of the Kalman gain and prove that explained variance (EV), computable from a single training batch, identifies the exact boundary: positive EV indicates the critic reduces variance, while zero or negative EV signals that it inflates variance. Building on this insight, we propose Explained Variance Policy Optimization (EVPO), which monitors batch-level EV at each training step and adaptively switches between critic-based and batch-mean advantage estimation, provably achieving no greater variance than the better of the two at every step. Across four tasks spanning classical control, agentic interaction, and mathematical reasoning, EVPO consistently outperforms both PPO and GRPO regardless of which fixed baseline is stronger on a given task. Further analysis confirms that the adaptive gating tracks critic maturation over training and that the theoretically derived zero threshold is empirically optimal. 

---
# Deep Supervised Contrastive Learning of Pitch Contours for Robust Pitch Accent Classification in Seoul Korean 

**Authors**: Hyunjung Joo, GyeongTaek Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.19477)  

**Abstract**: The intonational structure of Seoul Korean has been defined with discrete tonal categories within the Autosegmental-Metrical model of intonational phonology. However, it is challenging to map continuous $F_0$ contours to these invariant categories due to variable $F_0$ realizations in real-world speech. Our paper proposes Dual-Glob, a deep supervised contrastive learning framework to robustly classify fine-grained pitch accent patterns in Seoul Korean. Unlike conventional local predictive models, our approach captures holistic $F_0$ contour shapes by enforcing structural consistency between clean and augmented views in a shared latent space. To this aim, we introduce the first large-scale benchmark dataset, consisting of manually annotated 10,093 Accentual Phrases in Seoul Korean. Experimental results show that our Dual-Glob significantly outperforms strong baseline models with state-of-the-art accuracy (77.75%) and F1-score (51.54%). Therefore, our work supports AM-based intonational phonology using data-driven methodology, showing that deep contrastive learning effectively captures holistic structural features of continuous $F_0$ contours. 

---
# Do LLMs Game Formalization? Evaluating Faithfulness in Logical Reasoning 

**Authors**: Kyuhee Kim, Auguste Poiroux, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2604.19459)  

**Abstract**: Formal verification guarantees proof validity but not formalization faithfulness. For natural-language logical reasoning, where models construct axiom systems from scratch without library constraints, this gap between valid proofs and faithful translations is especially acute. We investigate whether frontier models exploit this gap when generating Lean 4 proofs, a behavior we term formalization gaming.
We evaluate GPT-5 and DeepSeek-R1 on 303 first-order logic problems (203 from FOLIO, 100 from Multi-LogiEval), comparing unified generation against a two-stage pipeline that separates formalization from proving. Despite compilation rates of 87-99%, we find no evidence of systematic gaming in unified generation: models prefer reporting failure over forcing proofs, even under prompting designed to encourage it. However, unfaithfulness that evades our detection signals may still occur. The two-stage pipeline reveals two distinct modes of unfaithfulness: GPT-5 fabricates axioms during proof generation, a reactive fallback detectable via cross-stage comparison, while DeepSeek-R1 mistranslates premises during formalization, producing internally consistent outputs that evade detection entirely. These findings show that high compilation rates or accuracies should not be equated with faithful reasoning. Code and data are available at this https URL. 

---
# VCE: A zero-cost hallucination mitigation method of LVLMs via visual contrastive editing 

**Authors**: Yanbin Huang, Yisen Li, Guiyao Tie, Xiaoye Qu, Pan Zhou, Hongfei Wang, Zhaofan Zou, Hao Sun, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.19412)  

**Abstract**: Large vision-language models (LVLMs) frequently suffer from Object Hallucination (OH), wherein they generate descriptions containing objects that are not actually present in the input image. This phenomenon is particularly problematic in real-world applications such as medical imaging and autonomous driving, where accuracy is critical. Recent studies suggest that the hallucination problem may stem from language priors: biases learned during pretraining that cause LVLMs to generate words based on their statistical co-occurrence. To mitigate this problem, we propose Visual Contrastive Editing (VCE), a novel post-hoc method that identifies and suppresses hallucinatory tendencies by analyzing the model's response to contrastive visual perturbations. Using Singular Value Decomposition (SVD), we decompose the model's activation patterns to isolate hallucination subspaces and apply targeted parameter edits to attenuate its influence. Unlike existing approaches that require fine-tuning or labeled data, VCE operates as a label-free intervention, making it both scalable and practical for deployment in resource-constrained settings. Experimental results demonstrate that VCE effectively reduces object hallucination across multiple benchmarks while maintaining the model's original computational efficiency. 

---
# RDP LoRA: Geometry-Driven Identification for Parameter-Efficient Adaptation in Large Language Models 

**Authors**: Yusuf Çelebi, Yağız Asker, Özay Ezerceli, Mahmoud ElHussieni, Selva Taş, Reyhan Bayraktar, Fatma Betül Terzioğlu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19321)  

**Abstract**: Fine-tuning Large Language Models (LLMs) remains structurally uncertain despite parameter-efficient methods such as Low-Rank Adaptation (LoRA), as the layer-specific roles of internal representations are poorly understood, leading to heuristic decisions about where adaptation should be applied. We model the evolution of hidden states as a high-dimensional geometric trajectory and propose using the Ramer-Douglas-Peucker (RDP) algorithm, a parameter-free and training-free polygon simplification method that preserves global structural transitions while eliminating locally redundant changes, to identify critical breakpoints along the representation path. Crucially, we use these geometric pivots not merely for analysis, but as a direct decision signal for determining which layers should be adapted during parameter-efficient fine-tuning. By integrating this geometry-aware layer selection strategy into LoRA fine-tuning of Qwen3-8B-Base, we achieve superior performance on MMLU-Math using only 13 RDP-selected layers (81.67%), significantly outperforming both full 36-layer adaptation (79.32%) and random 13-layer selection (75.56%), as well as the baseline Qwen3-8B-Base model (74.25%). These results demonstrate that leveraging the intrinsic geometry of representation trajectories provides a robust, interpretable, and training-free signal for optimizing layer selection during model adaptation. 

---
# Beyond Semantic Similarity: A Component-Wise Evaluation Framework for Medical Question Answering Systems with Health Equity Implications 

**Authors**: Abu Noman Md Sakib, Md. Main Oddin Chisty, Zijie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19281)  

**Abstract**: The use of Large Language Models (LLMs) to support patients in addressing medical questions is becoming increasingly prevalent. However, most of the measures currently used to evaluate the performance of these models in this context only measure how closely a model's answers match semantically, and therefore do not provide a true indication of the model's medical accuracy or of the health equity risks associated with it. To address these shortcomings, we present a new evaluation framework for medical question answering called VB-Score (Verification-Based Score) that provides a separate evaluation of the four components of entity recognition, semantic similarity, factual consistency, and structured information completeness for medical question-answering models. We perform rigorous reviews of the performance of three well-known and widely used LLMs on 48 public health-related topics taken from high-quality, authoritative information sources. Based on our analyses, we discover a major discrepancy between the models' semantic and entity accuracy. Our assessments of the performance of all three models show that each of them has almost uniformly severe performance failures when evaluated against our criteria. Our findings indicate alarming performance disparities across various public health topics, with most of the models exhibiting 13.8% lower performance (compared to an overall average) for all the public health topics that relate to chronic conditions that occur in older and minority populations, which indicates the existence of what's known as condition-based algorithmic discrimination. Our findings also demonstrate that prompt engineering alone does not compensate for basic architectural limitations on how these models perform in extracting medical entities and raise the question of whether semantic evaluation alone is a sufficient measure of medical AI safety. 

---
# Reducing the Offline-Streaming Gap for Unified ASR Transducer with Consistency Regularization 

**Authors**: Andrei Andrusenko, Vladimir Bataev, Lilit Grigoryan, Nune Tadevosyan, Vitaly Lavrukhin, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2604.19079)  

**Abstract**: Unification of automatic speech recognition (ASR) systems reduces development and maintenance costs, but training a single model to perform well in both offline and low-latency streaming settings remains challenging. We present a Unified ASR framework for Transducer (RNNT) training that supports both offline and streaming decoding within a single model, using chunk-limited attention with right context and dynamic chunked convolutions. To further close the gap between offline and streaming performance, we introduce an efficient Triton implementation of mode-consistency regularization for RNNT (MCR-RNNT), which encourages agreement across training modes. Experiments show that the proposed approach improves streaming accuracy at low latency while preserving offline performance and scaling to larger model sizes and training datasets. The proposed Unified ASR framework and the English model checkpoint are open-sourced. 

---
# Superficial Success vs. Internal Breakdown: An Empirical Study of Generalization in Adaptive Multi-Agent Systems 

**Authors**: Namyoung So, Seokgyu Jang, Taeuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.18951)  

**Abstract**: Adaptive multi-agent systems (MAS) are increasingly adopted to tackle complex this http URL, the narrow task coverage of their optimization raises the question of whether they can function as general-purpose this http URL address this gap, we conduct an extensive empirical study of adaptive MAS, revealing two key findings: (1) topological overfitting -- they fail to generalize across different domains; and (2) illusory coordination -- they achieve reasonable surface-level accuracy while the underlying agent interactions diverge from ideal MAS behavior, raising concerns about their practical this http URL findings highlight the pressing need to prioritize generalization in MAS development and motivate evaluation protocols that extend beyond simple final-answer correctness. 

---
# Personalized Benchmarking: Evaluating LLMs by Individual Preferences 

**Authors**: Cristina Garbacea, Heran Wang, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.18943)  

**Abstract**: With the rise in capabilities of large language models (LLMs) and their deployment in real-world tasks, evaluating LLM alignment with human preferences has become an important challenge. Current benchmarks average preferences across all users to compute aggregate ratings, overlooking individual user preferences when establishing model rankings. Since users have varying preferences in different contexts, we call for personalized LLM benchmarks that rank models according to individual needs. We compute personalized model rankings using ELO ratings and Bradley-Terry coefficients for 115 active Chatbot Arena users and analyze how user query characteristics (topics and writing style) relate to LLM ranking variations. We demonstrate that individual rankings of LLM models diverge dramatically from aggregate LLM rankings, with Bradley-Terry correlations averaging only $\rho = 0.04$ (57\% of users show near-zero or negative correlation) and ELO ratings showing moderate correlation ($\rho = 0.43$). Through topic modeling and style analysis, we find users exhibit substantial heterogeneity in topical interests and communication styles, influencing their model preferences. We further show that a compact combination of topic and style features provides a useful feature space for predicting user-specific model rankings. Our results provide strong quantitative evidence that aggregate benchmarks fail to capture individual preferences for most users, and highlight the importance of developing personalized benchmarks that rank LLM models according to individual user preferences. 

---
# Comparison of sEMG Encoding Accuracy Across Speech Modes Using Articulatory and Phoneme Features 

**Authors**: Chenqian Le, Ruisi Li, Beatrice Fumagalli, Xupeng Chen, Amirhossein Khalilian-Gourtani, Tianyu He, Adeen Flinker, Yao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18920)  

**Abstract**: We test whether Speech Articulatory Coding (SPARC) features can linearly predict surface electromyography (sEMG) envelopes across aloud, mimed, and subvocal speech in twenty-four subjects. Using elastic-net multivariate temporal response function (mTRF) with sentence-level cross-validation, SPARC yields higher prediction accuracy than phoneme one-hot representations on nearly all electrodes and in all speech modes. Aloud and mimed speech perform comparably, and subvocal speech remains above chance, indicating detectable articulatory activity. Variance partitioning shows a substantial unique contribution from SPARC and a minimal unique contribution from phoneme features. mTRF weight patterns reveal anatomically interpretable relationships between electrode sites and articulatory movements that remain consistent across modes. This study focuses on representation/encoding analysis (not end-to-end decoding) and supports SPARC as a robust and interpretable intermediate target for sEMG-based silent-speech modeling. 

---
# Harmful Intent as a Geometrically Recoverable Feature of LLM Residual Streams 

**Authors**: Isaac Llorente-Saguer  

**Link**: [PDF](https://arxiv.org/pdf/2604.18901)  

**Abstract**: Harmful intent is geometrically recoverable from large language model residual streams: as a linear direction in most layers, and as angular deviation in layers where projection methods fail. Across 12 models spanning four architectural families (Qwen2.5, Qwen3.5, Llama-3.2, Gemma-3) and three alignment variants (base, instruction-tuned, abliterated), under single-turn, English evaluation, we characterise this geometry through six direction-finding strategies. Three succeed: a soft-AUC-optimised linear direction reaches mean AUROC 0.98 and TPR@1\%FPR 0.80; a class-mean probe reaches 0.98 and 0.71 at <1ms fitting cost; a supervised angular-deviation strategy reaches AUROC 0.96 and TPR of 0.61 along a representationally distinct direction ($73^\circ$ from projection-based solutions), uniquely sustaining detection in middle layers where projection methods collapse.
Detection remains stable across alignment variants, including abliterated models from which refusal has been surgically removed: harmful intent and refusal behaviour are functionally dissociated features of the representation. A direction fitted on AdvBench transfers to held-out HarmBench and JailbreakBench with worst-case AUROC 0.96. The same picture holds at scale: across Qwen3.5 from 0.8B to 9B parameters, AUROC remains $\geq$0.98 and cross-variant transfer stays within 0.018 of own-direction performance
This is consistent with a simple account: models acquire a linearly decodable representation of harmful intent as part of general language understanding, and alignment then shapes what they do with such inputs without reorganising the upstream recognition signal. As a practical consequence, AUROC in the 0.97+ regime can substantially overestimate operational detectability; TPR@$1\%$FPR should accompany AUROC in safety-adjacent evaluation. 

---
# Human-Guided Harm Recovery for Computer Use Agents 

**Authors**: Christy Li, Sky CH-Wang, Andi Peng, Andreea Bobu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18847)  

**Abstract**: As LM agents gain the ability to execute actions on real computer systems, we need ways to not only prevent harmful actions at scale but also effectively remediate harm when prevention fails. We formalize a solution to this neglected challenge in post-execution safeguards as harm recovery: the problem of optimally steering an agent from a harmful state back to a safe one in alignment with human preferences. We ground preference-aligned recovery through a formative user study that identifies valued recovery dimensions and produces a natural language rubric. Our dataset of 1,150 pairwise judgments reveals context-dependent shifts in attribute importance, such as preferences for pragmatic, targeted strategies over comprehensive long-term approaches. We operationalize these learned insights in a reward model, re-ranking multiple candidate recovery plans generated by an agent scaffold at test time. To evaluate recovery capabilities systematically, we introduce BackBench, a benchmark of 50 computer-use tasks that test an agent's ability to recover from harmful states. Human evaluation shows our reward model scaffold yields higher-quality recovery trajectories than base agents and rubric-based scaffolds. Together, these contributions lay the foundation for a new class of agent safety methods -- ones that confront harm not only by preventing it, but by navigating its aftermath with alignment and intent. 

---
# Towards Understanding the Robustness of Sparse Autoencoders 

**Authors**: Ahson Saiyed, Sabrina Sadiekh, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2604.18756)  

**Abstract**: Large Language Models (LLMs) remain vulnerable to optimization-based jailbreak attacks that exploit internal gradient structure. While Sparse Autoencoders (SAEs) are widely used for interpretability, their robustness implications remain underexplored. We present a study of integrating pretrained SAEs into transformer residual streams at inference time, without modifying model weights or blocking gradients. Across four model families (Gemma, LLaMA, Mistral, Qwen) and two strong white-box attacks (GCG, BEAST) plus three black-box benchmarks, SAE-augmented models achieve up to a 5x reduction in jailbreak success rate relative to the undefended baseline and reduce cross-model attack transferability. Parametric ablations reveal (i) a monotonic dose-response relationship between L0 sparsity and attack success rate, and (ii) a layer-dependent defense-utility tradeoff, where intermediate layers balance robustness and clean performance. These findings are consistent with a representational bottleneck hypothesis: sparse projection reshapes the optimization geometry exploited by jailbreak attacks. 

---
# Beyond Indistinguishability: Measuring Extraction Risk in LLM APIs 

**Authors**: Ruixuan Liu, David Evans, Li Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.18697)  

**Abstract**: Indistinguishability properties such as differential privacy bounds or low empirically measured membership inference are widely treated as proxies to show a model is sufficiently protected against broader memorization risks. However, we show that indistinguishability properties are neither sufficient nor necessary for preventing data extraction in LLM APIs. We formalize a privacy-game separation between extraction and indistinguishability-based privacy, showing that indistinguishability and inextractability are incomparable: upper-bounding distinguishability does not upper-bound extractability. To address this gap, we introduce $(l, b)$-inextractability as a definition that requires at least $2^b$ expected queries for any black-box adversary to induce the LLM API to emit a protected $l$-gram substring. We instantiate this via a worst-case extraction game and derive a rank-based extraction risk upper bound for targeted exact extraction, as well as extensions to cover untargeted and approximate extraction. The resulting estimator captures the extraction risk over multiple attack trials and prefix adaptations. We show that it can provide a tight and efficient estimation for standard greedy extraction and an upper bound on the probabilistic extraction risk given any decoding configuration. We empirically evaluate extractability across different models, clarifying its connection to distinguishability, demonstrating its advantage over existing extraction risk estimators, and providing actionable mitigation guidelines across model training, API access, and decoding configurations in LLM API deployment. Our code is publicly available at: this https URL. 

---
# Owner-Harm: A Missing Threat Model for AI Agent Safety 

**Authors**: Dongcheng Zhang, Yiqing Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18658)  

**Abstract**: Existing AI agent safety benchmarks focus on generic criminal harm (cybercrime, harassment, weapon synthesis), leaving a systematic blind spot for a distinct and commercially consequential threat category: agents harming their own deployers. Real-world incidents illustrate the gap: Slack AI credential exfiltration (Aug 2024), Microsoft 365 Copilot calendar-injection leaks (Jan 2024), and a Meta agent unauthorized forum post exposing operational data (Mar 2026). We propose Owner-Harm, a formal threat model with eight categories of agent behavior damaging the deployer. We quantify the defense gap on two benchmarks: a compositional safety system achieves 100% TPR / 0% FPR on AgentHarm (generic criminal harm) yet only 14.8% (4/27; 95% CI: 5.9%-32.5%) on AgentDojo injection tasks (prompt-injection-mediated owner harm). A controlled generic-LLM baseline shows the gap is not inherent to owner-harm (62.7% vs. 59.3%, delta 3.4 pp) but arises from environment-bound symbolic rules that fail to generalize across tool vocabularies. On a post-hoc 300-scenario owner-harm benchmark, the gate alone achieves 75.3% TPR / 3.3% FPR; adding a deterministic post-audit verifier raises overall TPR to 85.3% (+10.0 pp) and Hijacking detection from 43.3% to 93.3%, demonstrating strong layer complementarity. We introduce the Symbolic-Semantic Defense Generalization (SSDG) framework relating information coverage to detection rate. Two SSDG experiments partially validate it: context deprivation amplifies the detection gap 3.4x (R = 3.60 vs. R = 1.06); context injection reveals structured goal-action alignment, not text concatenation, is required for effective owner-harm detection. 

---
# Unlocking the Edge deployment and ondevice acceleration of multi-LoRA enabled one-for-all foundational LLM 

**Authors**: Sravanth Kodavanti, Sowmya Vajrala, Srinivas Miriyala, Utsav Tiwari, Uttam Kumar, Utkarsh Kumar Mahawar, Achal Pratap Singh, Arya D, Narendra Mutyala, Vikram Nelvoy Rajendiran, Sharan Kumar Allur, Euntaik Lee, Dohyoung Kim, HyeonSu Lee, Gyusung Cho, JungBae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.18655)  

**Abstract**: Deploying large language models (LLMs) on smartphones poses significant engineering challenges due to stringent constraints on memory, latency, and runtime flexibility. In this work, we present a hardware-aware framework for efficient on-device inference of a LLaMA-based multilingual foundation model supporting multiple use cases on Samsung Galaxy S24 and S25 devices with SM8650 and SM8750 Qualcomm chipsets respectively. Our approach integrates application-specific LoRAs as runtime inputs to a single frozen inference graph, enabling dynamic task switching without recompilation or memory overhead. We further introduce a multi-stream decoding mechanism that concurrently generates stylistic variations - such as formal, polite, or jovial responses - within a single forward pass, reducing latency by up to 6x. To accelerate token generation, we apply Dynamic Self-Speculative Decoding (DS2D), a tree-based strategy that predicts future tokens without requiring a draft model, yielding up to 2.3x speedup in decode time. Combined with quantization to INT4 and architecture-level optimizations, our system achieves 4-6x overall improvements in memory and latency while maintaining accuracy across 9 languages and 8 tasks. These results demonstrate practical feasibility of deploying multi-use-case LLMs on edge devices, advancing the commercial viability of Generative AI in mobile platforms. 

---
# Who Shapes Brazil's Vaccine Debate? Semi-Supervised Modeling of Stance and Polarization in YouTube's Media Ecosystem 

**Authors**: Geovana S. de Oliveira, Ana P. C. Silva, Fabricio Murai, Carlos H. G. Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2604.18586)  

**Abstract**: Vaccination remains a cornerstone of global public health, yet the COVID-19 pandemic exposed how online misinformation, political polarization, and declining institutional trust can undermine immunization efforts. Most of the prior computational studies that analyzed vaccine discourse on social platforms focus on English-language data, specific vaccines, or short time windows, impairing our understanding of long-term dynamics in high-impact, non-English contexts like Brazil, home to one of the world's most comprehensive immunization systems. We here present the largest longitudinal study of Brazil's vaccine discourse on YouTube, leveraging a semi-supervised stance detection framework that combines self-labeling and self-training to classify nearly 1.4 million comments. By integrating stance with temporal patterns, engagement metrics, and channel taxonomy (legacy media, science communicators, digital-native outlets), we map how pro- and anti-vaccine narratives evolve and circulate within a hybrid media ecosystem. Our results show that semi-supervised learning substantially improves stance classification robustness, enabling fine-grained tracking of public attitudes across Brazil's full immunization schedule. Polarization spikes during epidemiological crises, especially COVID-19, but becomes fragmented across vaccines and interaction patterns in the post-pandemic period. Notably, science communication and digital-native channels emerge as the primary loci of both supportive and oppositional engagement, revealing structural vulnerabilities in contemporary health communication. Thus, our work advances computational methods for large-scale stance modeling while offering actionable evidence for public health agencies, platform governance, and online information ecosystems. 

---
