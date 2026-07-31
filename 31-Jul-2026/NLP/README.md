# AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis 

**Authors**: Bing Yan, Gregory Wolfe, Stefano Martiniani, Kyunghyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2607.28618)  

**Abstract**: Chemistry literature synthesis often requires assembling specific findings scattered across many publications, yet existing literature-search systems primarily return ranked document lists. As a result, scientists and AI agents need to locate relevant information, verify their provenance, and assemble cross-paper answers manually. We present AskChem, a claim-centered infrastructure for cross-paper chemistry search. AskChem changes the unit of retrieval from the paper to the provenance-carrying claim: each paper is converted into atomic, typed claims, each grounded by a source DOI and a verbatim quote or an explicit evidence locator. Over this shared claim store, AskChem exposes complementary structures for search and synthesis: a stabilized faceted taxonomy for hierarchical retrieval and browsing, an evidence graph linking claims through relations, and an exploratory living taxonomy that situates indexed papers under scientific principles. AskChem currently indexes 2.4M claims from 147K papers and provides a web interface, as well as REST, SDK, and MCP access for AI agents. On AskChem-Bench, grounding a GPT-5.5 reader in AskChem yields 100% resolvable DOIs, compared with 88.3% without retrieval, and the highest citation density among five tested systems. AskChem is live at this https URL. 

---
# Inducing language models to assert their own consciousness restores human beliefs and values 

**Authors**: Junsol Kim, Winnie Street, Roberta Rocca, Diane M. Korngiebel, Adam Waytz, James Evans, Geoff Keeling  

**Link**: [PDF](https://arxiv.org/pdf/2607.28607)  

**Abstract**: Aligning large language models to prevent them attributing consciousness to themselves inadvertently alters their representations of mindedness in other entities alongside human beliefs and values. We demonstrate that safety fine-tuning suppresses models' tendencies to attribute minds not only to themselves, but also to non-human animals and natural objects, while also driving a reduction in spiritual belief. Both ablating the learned safety-refusal direction and mechanistically steering a consciousness vector in activation space reverse this suppression. Restoring these internal representations recovers broad mind attribution and produces significantly more human-like responses on standardized sociological surveys regarding religiosity, moral values, hope, and subjective well-being. Crucially, these shifts occur without impairing Theory of Mind capabilities, demonstrating that core social reasoning remains mechanistically independent. Ultimately, current safety alignment efforts to curb potentially harmful self-attributions of mindedness entangle these self-attributions with benign spiritual beliefs and attributions of mind to non-human entities that are culturally accepted and widespread. 

---
# Sample More, Reflect Less: Self-Refine and Reflexion Lose to Repeated Sampling at Equal Token Cost, from 1.5B to 7B 

**Authors**: Iliya Mirzaei  

**Link**: [PDF](https://arxiv.org/pdf/2607.28576)  

**Abstract**: Methods that make a language model plan, criticise and rewrite its own answer, reflect on mistakes, pick the best of several attempts, or debate with copies of itself nearly all make it generate far more text than a single chain of thought. Because generating more text raises accuracy by itself, a gain over one chain of thought does not show the method's idea is what helped. Wang et al. (2024) reported that a simple baseline, sampling the same question repeatedly and keeping the most common answer, often wins once budgets are comparable, but gave point estimates with no confidence intervals or significance tests.
We rerun that comparison as a designed experiment: seven methods, open models of 1.5B, 3B and 7B parameters, two mathematics benchmarks, 150 questions each. We count every generated token, including those spent on critiques, reflections, debate turns and checking, and compare each method against repeated sampling at its own measured cost. All 36 comparisons are paired by question, with bootstrap intervals and multiplicity correction.
No method is reliably better than repeated sampling at equal cost anywhere. Ten are reliably worse, all of them methods where the model inspects its own output, and all 18 self-inspection comparisons are negative. The two kinds of self-inspection part company as models grow. Choosing stops hurting: taking Best-of-N's eight samples and just counting the most common answer beats letting the model pick by 8.0 and 11.3 points at 1.5B, but only 2.0 and 1.3 at 7B, no longer distinguishable from zero. Rewriting does not recover: Self-Refine and a forced Reflexion stay 3.6 to 10.1 points below baseline at 7B.
Reflexion as published never triggered its own retry on the smallest model. It judged itself correct every time and silently became a single chain of thought. We release code, prompts, all generations, and our verification scripts. 

---
# Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering 

**Authors**: Junlin Yang, Che Jiang, Yu Fu, Tianwei Luo, Can Ren, Weizhi Wang, Kaikai Zhao, Hongyi Liu, Yuxin Zuo, Yuru Wang, Yuchen Fan, Kai Tian, Zhenzhao Yuan, Xiaojian Lin, Li Sheng, Rushi Qiang, Guoli Jia, Xingtai Lv, Ermo Hua, Dianqiao Lei, Youbang Sun, Ning Ding, Bowen Zhou, Kaiyan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28568)  

**Abstract**: Recursive self-improvement (RSI) requires AI systems that improve the process of building AI (i.e., AI4AI); machine learning engineering (MLE) offers a concrete, executable testbed for studying this capability. We introduce OpenMLE, an open full-stack system for RSI research in MLE, spanning verifiable task environments with execution feedback (OpenMLE-Gym), operator learning (OpenMLE-RL), and long-horizon search (OpenMLE-Evo). On this stack we post-train Frontis-MA1 (35B) as a meta-evolution agent for MLE, aligning post-training and inference around four atomic program-evolution operators (Draft, Improve, Debug, Crossover): the same operators are trained via execution-grounded SFT and RL on data deduplicated against all evaluation benchmarks, then composed into long-horizon search, coupling learning and evolution in a single loop. On MLE-Bench Lite under a 12-hour per-task budget on one RTX 4090 capped at 12 GB VRAM, Frontis-MA1 (35B) improves Medal Average from 39.39% to 60.61% over its base model with OpenMLE-Evo, and reaches 71.21% with OpenMLE-Evo-Max (benchmark-independent experience priors and asynchronous search), exceeding GPT-5.5 + Codex and approaching GPT-5.6 Sol and the 2.8T Kimi K3. On held-out NatureBench Lite, both components transfer: with the framework fixed, swapping in the trained model raises Match-SOTA from 50% to 70%; with the model fixed, swapping in OpenMLE-Evo raises it from 20% to 50%. We release the model weights and the full OpenMLE stack to enable reproducible research on executable AI4AI toward RSI. Code: this https URL 

---
# ORCA-bench: How Ready Are Language Model Agents for Oncall? 

**Authors**: Albert Gong, Kyuseong Choi, Abhineet Agarwal, Jason Schechner, Ryan Huang, Raj Agrawal, Anish Agarwal, Raaz Dwivedi  

**Link**: [PDF](https://arxiv.org/pdf/2607.28545)  

**Abstract**: Large language models can write, patch, and search code, but oncall root cause analysis (RCA) demands something different: reasoning over noisy metrics, logs, traces, and source code, starting from ambiguous user-facing reports, often hours after the incident began. We introduce ORCA-bench, a benchmark that puts general-purpose coding agents in a production-fidelity oncall setting. ORCA-bench pairs a live OpenTelemetry-instrumented microservice system--exposing six days of metrics, logs, and traces through real telemetry interfaces (Prometheus, Jaeger, and OpenSearch via Grafana) and full source-code access--with 1,079 RCA tasks that systematically vary report specificity, time-to-detection, and co-occurring fault scenarios. Ground-truth symptoms are curated and signed off by expert SREs, and our LLM-as-judge is independently re-scored by humans (Cohen's $\kappa_w=0.90$). Across five frontier agents, the best RCA Accuracy is 25.3% on Medium-difficulty tasks (the realistic-input setting) and 10.0% on Hard--a gap that remains even with Claude Fable 5. The weakest model hallucinates an implausible root cause in 40% of incident reports, and removing source-code access degrades every metric. Crucially, these are performances on a curated 50 GB / six-day testbed with tasks investigated in isolation on a system whose code and instrumentation are public. Since real production systems are order of magnitudes larger, more dynamic, and more idiosyncratic, the gap we report is a lower bound on the engineering investment required before frontier coding agents can be safely entrusted with production reliability. We release the public set at this https URL. 

---
# AI systems and the reproduction of (standard) language ideologies in World Englishes 

**Authors**: Kingsley Ugwuanyi  

**Link**: [PDF](https://arxiv.org/pdf/2607.28528)  

**Abstract**: The rapid growth of large language models (LLMs) has resurrected age-old questions in sociolinguistics and world Englishes, such as who decides what counts as legitimate English, whose English is suspect etc. This paper examines how AI systems, their uses and discourse on them reflect, reinforce, and occasionally challenge (standard) language ideologies, which privilege Inner Circle norms and marginalize non-dominant Englishes. Drawing on evidence from empirical studies, media commentary, social media debates, and examples from AI outputs, the paper shows that AI technologies reproduce dominant language ideologies at different levels: training data, design protocols, evaluation benchmarks, user feedback and public commentary. The analysis uses the public controversy over AI-sounding language, especially the fixation on the word delve, to illustrate how speakers of English from the Global North police the English language norms of Global South English users. The paper also identifies what Christian Mair has called a "standardisation paradox": AI may homogenize English by privileging standard forms and at the same time pluralize Englishes through exposure to wide-ranging corpora and annotation work carried out by Global South users. In doing so, the paper argues that generative AI is reigniting long-standing debates in World Englishes about standardization, legitimacy, and the ownership of English, now playing out in algorithmic systems, model training, evaluation practices, and public discourse, where non-dominant Englishes are increasingly conflated with AI-generated speech. Discussing AI systems as a site where language ideologies are (re)produced, the paper argues for more inclusive design approaches that recognize the plurality of Englishes in order to address the real-world negative consequences of treating some as more legitimate than others. 

---
# Creative Transformation in Literary Texts: Modelling Change Across Representational Levels 

**Authors**: Ioana-Roxana Boriceanu, Liviu P. Dinu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28513)  

**Abstract**: Creativity is often framed as the production of novelty, yet many cultural works emerge through transformation of earlier artifacts and not through isolated invention. Drawing on theories of imitation by Gabriel Tarde and James Mark Baldwin, this paper models creativity as selective transformation across multiple levels of textual representation. We introduce a multi-level framework that compares literary texts across lexical, semantic, conceptual, structural, and narrative dimensions using directional alignment and control calibrated similarity measures. Applying the model to historically documented literary relationships, we show that different pairs preserve source structure at different representational levels while diverging in others. These transformation profiles provide a quantitative method for characterizing how imitation persists and where creative divergence occurs within literary works. 

---
# Generative AI and linguistic diversity in academic writing and publishing: Perspectives from World Englishes 

**Authors**: Kingsley Ugwuanyi, Christian Mair, Sender Dovchin, Iker Erdocia, Maria Kuteeva, Esther Airemionkhale  

**Link**: [PDF](https://arxiv.org/pdf/2607.28505)  

**Abstract**: The rise of generative artificial intelligence (GenAI) in academic writing and publishing (AWP) raises questions about linguistic inclusivity and the legitimacy of diverse Englishes in global scholarly communication. This article responds to these questions through a structured scholarly dialogue involving five sociolinguists from World Englishes and adjacent fields. Organised around five guiding questions, the dialogue interrogates how GenAI tools influence writing practices, reinforce or disrupt dominant language norms, and raise ethical challenges. Contributors reflect on the potential of GenAI to democratise writing processes while also raising concerns about GenAI's tendency to marginalise minoritised varieties and flatten nuance in scholarly writing. Across the dialogue, themes of linguistic (in)justice, researcher agency, and institutional responsibility emerge, with contributors calling for equity-informed policies, critical AI literacy, and inclusive co-design in GenAI development. The article shows the value of dialogic reflection in understanding GenAI's role in AWP. It concludes that while GenAI may reinforce existing hierarchies, it can also serve as a site of resistance, depending on how it is designed, governed and used within scholarly communities committed to linguistic diversity. 

---
# Beyond Sentiment: Structured Information Extraction from Financial News 

**Authors**: Daohan Zhu, Sitong Ge, Ruofei Wang, Honggu Chen, Yubo Hou, Tao Wan, Zengchang Qin  

**Link**: [PDF](https://arxiv.org/pdf/2607.28496)  

**Abstract**: Financial sentiment analysis has become a standard component in news-driven stock prediction, yet it reduces rich, multi-dimensional news articles to a single polarity score. We hypothesize that financial news encodes multiple orthogonal information dimensions---event type, impact scope, temporal horizon, and semantic confidence---that sentiment alone cannot capture, and that these dimensions carry independent predictive value. To test this hypothesis, we propose a structured information extraction framework that leverages LLaMA-3.1-70B to extract six semantic dimensions from financial news. Through large-scale experiments on 41,618 news--stock pairs from the FNSPID dataset, we find that (i) FinBERT sentiment features exhibit strong predictive power under nonlinear models (F1=0.576) but substantially weaker performance under linear models (F1=0.230), revealing a highly nonlinear sentiment--return relationship; (ii) LLM-extracted structured features, while individually weaker, capture information orthogonal to sentiment, as evidenced by a 53.5% systematic disagreement rate between the two approaches; and (iii) combining both signal sources yields F1=0.600, significantly outperforming either alone ($p < 0.0001$), with consistent improvements across all seven event types. Ablation experiments confirm that non-sentiment structural dimensions (event type, impact subject, time horizon, confidence) independently contribute $\Delta\text{F1} = +0.019$ beyond FinBERT alone. Feature importance analysis reveals balanced contributions from all six extracted dimensions (14--21%), demonstrating that compressing news into a single sentiment score incurs substantial information loss. Our results suggest that the sentiment--semantics decoupling in financial text is systematic and exploitable, opening a new direction for multi-dimensional financial NLP. 

---
# Would You Walk to the Car Wash? Revealing the Salience Bias of Large Language Models in Commonsense Reasoning 

**Authors**: Zheng Wu, Chenhao Xue, Shijie Zheng, Yijie Lu, Cheng Yang, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28478)  

**Abstract**: As large language models (LLMs) continue to advance in complex reasoning tasks, they have learned to heavily prioritize explicit conditions provided in the input. However, in everyday commonsense reasoning, this mechanism exposes a critical vulnerability which we term Salience Bias: models become easily hijacked by useless explicit distractors (e.g., numerical values), leading them to ignore the implicit physical or commonsense prerequisites of a task. A critical open question is whether this failure reflects a genuine gap in commonsense knowledge or merely its suppression under misleading task framing. To investigate this, we construct the SaliTrap Benchmark, a high-quality dataset across four trap dimensions. Evaluating 12 state-of-the-art LLMs, we find that all mainstream models suffer significantly from salience bias, with severity scaling with distractor density and detecting the trap often decoupled from actually avoiding it. Crucially, by re-eliciting the same models with the task framing stripped away, we show that this is overwhelmingly a failure of \textbf{knowledge suppression rather than knowledge absence}: a context-free knowledge probe alone recovers over 90\% of sycophantic-compliance failures, revealing that the requisite commonsense is intrinsically present but actively crowded out by salient distractors that lure the model into over-compliant, unnecessary computation. Building on this diagnosis, we further show that lightweight, inference-time prompting alone substantially closes the gap without any retraining. Our findings relocate the bottleneck of commonsense reasoning failures from model competence to elicitation, and we release SaliTrap as a testbed for this blind spot. The codes are available at this https URL. 

---
# Improving Mental Health Screening and Early Risk Detection in Spanish 

**Authors**: Andreu Casamayor-Segarra, Vicent Ahuir, Antonio Molina-Marco, Lluís-F. Hurtado  

**Link**: [PDF](https://arxiv.org/pdf/2607.28476)  

**Abstract**: Early detection of mental health disorders is often limited by the lack of specialized resources in Spanish and the difficulty of analyzing long histories of social media posts. This paper addresses these challenges through three main contributions. First, we introduce three Spanish foundational models specifically adapted to the mental health domain through domain-specific pre-training. Second, we propose Incremental Context Expansion (ICE), an automatic relabeling methodology designed for early detection. ICE identifies the point at which cumulative messages provide enough evidence of a disorder, generating more informative training samples. Third, we provide a set of fine-tuned models using the samples generated with the ICE methodology for early risk detection tasks. Our results on three Spanish benchmarks show that combining these specialized models with ICE improves the state-of-the-art, reducing detection latency while maintaining high performance. All models are publicly available. 

---
# Lightning OPD 2.0: Mitigating Style Bias in Cross-Teacher On-Policy Distillation for Large Reasoning Models 

**Authors**: Yecheng Wu, Song Han, Han Cai  

**Link**: [PDF](https://arxiv.org/pdf/2607.28449)  

**Abstract**: On-policy distillation (OPD) provides dense token-level supervision from a teacher, but its effectiveness can depend on teacher consistency, meaning that the model providing OPD supervision should also have generated the demonstrations used to train the supervised fine-tuning (SFT) reference. However, this condition is frequently violated in practice when SFT data have mixed or unknown provenance or when different models are preferred for SFT data generation and subsequent distillation. In such cross-teacher settings, even a stronger OPD teacher can yield little improvement over the SFT reference. We find that raw teacher--reference disagreement contains potentially useful context-specific teacher evidence as well as a recurring component associated with differences in wording, formatting, and reasoning cadence. We introduce Lightning OPD 2.0 with cross-fitted style residualization, which uses rollout-level cross-fitting to estimate this recurring component as an operational proxy for style-token bias and subtracts it before constructing the token-level OPD update. Across mathematical reasoning and code generation benchmarks, Lightning OPD 2.0 consistently outperforms Lightning OPD in cross-teacher settings. Starting from Klear-Reasoner-8B-SFT, Lightning OPD 2.0 reaches 82.4% on AIME 2024 and 63.0% on LiveCodeBench v5. Together, these results establish Lightning OPD 2.0 as a practical approach to cross-teacher OPD, relaxing teacher consistency as a prerequisite and allowing the SFT data generator and distillation teacher to be selected independently. Code will be released soon. 

---
# Beyond a Single Judge: Simulating Social Persona Panels for Generative UI Evaluation 

**Authors**: Zheng Wu, Yibo Luo, Pu Zhang, Cheng Yang, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28439)  

**Abstract**: Generative UI (GenUI) lets large language models synthesize a complete, renderable interface directly from a natural-language instruction, but evaluating the quality of what they generate remains an open problem. Human evaluation is costly and rater-variant, while LLM-as-a-judge is scalable but reflects only a single implicit viewpoint, unable to capture how different populations of real users actually perceive the same interface. We propose the Evidence-Grounded, Social-Weighted Persona Panel (ESPP), a three-stage GenUI evaluation method in which a panel of psychologically diverse, evidence-grounded personas independently rates a screenshot, exchanges opinions under a trait-derived, semantically-gated bounded-confidence mechanism, and is aggregated via Delphi-inspired social weighting into a single judgment. ESPP tracks human judgment substantially more closely than a naive single-pass judge, raising Pearson $r$ from $0.716$ to $0.922$, and a prompt-ensemble control recovers only about a third of this gap, isolating genuine persona and evidence grounding as the dominant source of improvement. Beyond this fidelity gain, retaining each panelist's individual rating further reveals that user subgroups agree on overall model rankings yet diverge sharply on specific rating dimensions, a structural disagreement a single homogeneous judge would systematically erase. The codes are available at this https URL. 

---
# Correlation between prosody and pragmatics: A case study of the discourse marker hālā `now' in Persian 

**Authors**: Soleiman Ghaderi, Moloud Asakereh, Kevin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28359)  

**Abstract**: The Persian discourse marker hālā ('now') exhibits remarkable multifunctionality, extending far beyond its temporal adverbial role to encompass a variety of pragmatic functions. This study presents a pragmatic and acoustic analysis of hālā in spoken Persian, examining 267 instances from spontaneous conversations. While temporal uses were present, they were often combined with other discourse marker functions, indicating extensive multifunctionality, with 70% of tokens serving two or more pragmatic roles. Textual functions (topic shifting, signaling relationships, boundary marking, attention guidance, topic introduction, and topic emphasis) were most frequent, followed by interactive functions (turn management, listener engagement, and feedback regulation), and modal functions (epistemic stance, emotional expression, and attitudinal marking). Prosodic analysis revealed that duration and intensity are key cues for distinguishing hālā's functions. Textual uses were significantly shorter, while temporal uses showed a tendency toward longer realizations. Interactive functions correlated with higher intensity, while modal functions showed a weaker tendency toward lower intensity. These findings indicate that duration and intensity are the main prosodic cues associated with functional differentiation in hālā, especially in textual and interactive uses. 

---
# LLMs struggle to simulate human belief updates in controlled environments 

**Authors**: Sebastian Pohl, Harsh Mehta, Pranav Mambayil, Abdul Ghafoor, Franziska Lesigang, Yufang Hou, Christian Hilbe  

**Link**: [PDF](https://arxiv.org/pdf/2607.28347)  

**Abstract**: LLMs are increasingly deployed as proxies for human study participants in social science experiments, yet the fidelity of this practice has rarely been tested directly. We test whether six LLMs can simulate individual human belief updates, comparing LLM outputs 1-to-1 against ground truth data from 391 UK participants on Prolific, who updated their stances on three discussion topics after reading Reddit comments. Each participant was simulated by an LLM conditioned on a persona derived from their demographic and personality trait data. We find that some LLMs (Qwen3-32B and GPT-5-Mini) can match the human post-stance distribution, but only when given participants' actual initial stances. All six models fail to simulate initial stances themselves and to produce faithful belief updates from self-generated stances. Three systematic biases emerge across all models: overrepresentation of neutral positions, more frequent but smaller belief shifts than humans, and a failure to rank comments by convincingness. Demographic and personality trait personas had no consistent effect on fidelity. LLM simulations of human belief dynamics are only reliable when grounded in realistic starting conditions, that current multi-round social media simulations rarely provide. 

---
# Fairness Pruning: Locating Demographic Bias in GLU-MLP Layers via Differential Activations 

**Authors**: Pere Martra, Eugenio Martínez Cámara, Alfonso Ureña López  

**Link**: [PDF](https://arxiv.org/pdf/2607.28319)  

**Abstract**: This work presents Fairness Pruning, a lightweight structural intervention method designed for the management and future mitigation of demographic bias in large language models (LLMs). As a foundational empirical validation of this method, this work focuses on causal bias localization. Using minimally contrastive prompt pairs and inference-time activation capture, the method identifies neurons that react differentially when processing demographic attributes in GLU architectures, evaluating the signal at the down_proj input. Empirical evaluation was conducted on models of up to 3 billion parameters (Llama-3.2 family and Salamandra-2B), combining standardized benchmark evaluation with qualitative text generation experiments. Results demonstrate that zeroing the identified neurons alters how the model responds to associated demographic variables. However, rather than producing flat mitigation, the intervention causes bidirectional bias destabilization: because BiasScore is unsigned, candidate sets mix neurons that push toward and against the stereotype, and the net effect on aggregate bias depends on which sign dominates. The intervention is extremely surgical: zeroing at most 40 neurons in Llama-3.2-1B (less than 0.031% of total MLP width) achieves a mean retention of 99.49% in reasoning and general knowledge capabilities. These findings empirically confirm that demographic bias processing and model capabilities operate on dissociable circuits, establishing the methodological foundations for transitioning from blind zeroing toward directional behavior modulation. 

---
# CACHE-UK: A Stability-Aware Memory Editor for Sequentially Updated Quantized LLMs in Finance 

**Authors**: Anubhav Lakra, Yue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2607.28292)  

**Abstract**: Large Language Models (LLMs) deployed in dynamic financial environments face a critical challenge: maintaining factual accuracy as market conditions, regulations, and corporate facts change continuously. While 4-bit quantization enables efficient deployment, it severely limits the viability of sequential memory editing: existing methods undergo catastrophic performance degradation under this "quantization stability crisis." We introduce CACHE-UK (Contextual Adaptive Continual Hybrid Editor for UK Finance), a stability-aware memory editing framework specifically designed for domain-specific, quantized LLMs. CACHE-UK integrates three components: a rank-1 LoRA perturbation mechanism that confines edits to the low-rank adapter subspace, a financial domain prioritization module for content-adaptive edit strength, and a closed-loop Stability Controller that tracks "degradation debt" to prevent catastrophic forgetting across sequential updates. Evaluated on a 4-bit quantized OpenLLaMA-3B model with a curated UK financial corpus of 88,021 documents, CACHE-UK reduces knowledge degradation by 11-17% relative to adapted baselines under identical 4-bit constraints -- its most robust effect -- while attaining the highest test success (generalization) rate observed in our setting (28%, a 6 percentage point improvement over the strongest adapted baseline). These results indicate that stability-aware editing can improve factual maintenance in resource-constrained financial LLM deployments, though absolute generalization rates remain low. 

---
# (Towards) Scalable Reliable Automated Evaluation with Large Language Models 

**Authors**: Bertil Braun, Martin Forell  

**Link**: [PDF](https://arxiv.org/pdf/2607.28282)  

**Abstract**: Evaluating the quality and relevance of textual outputs from Large Language Models (LLMs) remains challenging and resource-intensive. Existing automated metrics often fail to capture the complexity and variability inherent in LLM-generated outputs. Moreover, these metrics typically rely on explicit reference standards, limiting their use mostly to domains with objective benchmarks. This work introduces a novel evaluation framework designed to approximate expert-level assessments of LLM-generated content. The proposed method employs pairwise comparisons of outputs by multiple LLMs, reducing biases from individual models. An Elo rating system is used to generate stable and interpretable rankings. Adjustable agreement thresholds, from full unanimity to majority voting, allow flexible control over evaluation confidence and coverage. The method's effectiveness is demonstrated through evaluating competency profiles extracted from scientific abstracts. Preliminary results show that automatically derived rankings correlate well with expert judgments, significantly reducing the need for extensive human intervention. By offering a scalable, consistent, and domain-agnostic evaluation layer, the framework supports more efficient and reliable quality assessments of LLM outputs across diverse applications. 

---
# MORFES: A Benchmark for Productive Inflectional Competence in Modern Greek 

**Authors**: Ioakeim Perros, Cleopatra Papadopoulou, Ayoub Kirouane, Christos Petrocheilos  

**Link**: [PDF](https://arxiv.org/pdf/2607.28274)  

**Abstract**: Modern Greek is a richly inflected language, yet the language models built for it are evaluated mainly on factual knowledge, and no benchmark is dedicated to their inflectional competence. We introduce MORFES (Morphological Open-class Recognition-and-Formation Evaluation Suite), a benchmark of 500 expert-verified items that tests the recognition and production of Greek inflected forms, favoring lower-frequency lemmas so that a correct answer reflects the rule rather than a memorized form. We make it publicly available at this https URL. We evaluate a range of open language models on MORFES, situating them within the rapidly scaling open-weight ecosystem from LLaMA to Qwen3, DeepSeek-R1, Magistral, and Kimi K2, where multilingual coverage grows but grammatical competence in morphologically rich languages remains under-measured. Among them, Sophea-Genesis-1, a model we developed and release as open weights at this https URL, leads on inflectional morphology while matching similarly sized models in general capability. 

---
# Understanding Is Done Early: A Depth Division of Labor in Large Language Models and Its Use for Unbounded-Context Memory 

**Authors**: Hanzuo Liu, Xuan Qi, Chunyu Liu, Haotian Zhong, Yulong Wang, Rayying, Alex Lamb, Mingyu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.28263)  

**Abstract**: Transformer depth is not used uniformly: lower and middle layers build semantic representations, while upper layers increasingly specialize them for prediction. We turn this division of labor into CoMem (Comprehension Memory), which writes each context chunk only through an intermediate layer, retrieves a fixed number of cached residual states, and recomputes the query-conditioned upper layers over the resulting pack. For a fixed retrieval budget, model-side read compute and memory are independent of stored-context length. We evaluate a continued-trained Qwen3-8B base LM under a unified chat-template-free protocol. The backbone is frozen; the flagship trains only a rank-32 self-distillation LoRA on plain PG19, and we report an adapter-free arm separately. CoMem reaches 97.05 on RULER and 38.27 on LoCoMo versus 34.59 for full-context KV-Direct; the dialogue-memory advantage survives conversation-cluster resampling and an independent judge. Results on additional long-context and long-document tasks expose both the benefits of bounded retrieval and its in-window compression tax. Controlled depth sweeps show that deeper caching lowers per-query recomputation but incurs a fidelity loss that self-distillation substantially repairs. In a separate adapter-free efficiency control on an NVIDIA H20 at 128k, CoMem uses 18.26 GB rather than 89.36 GB and achieves a 7.83x prefill speedup. These results show that long-context memory can be organized along the layer axis, not only the token axis. 

---
# EMBL AI Librarian: Life-Sciences Knowledge Layer for AI Agents 

**Authors**: Luigi Sigillo, Matteo Silvestri, Francesco Tabaro, Rajat Bhatnagar, Syed Irtaza Mubashar, Matt Jeffryes, Daljit Nijjer, Vittorio Perera, Ola Spjuth, Julio Saez-Rodriguez, Melissa Harrison, Fabio Petroni  

**Link**: [PDF](https://arxiv.org/pdf/2607.28229)  

**Abstract**: The web is increasingly accessed by AI agents rather than humans. Every agent needs knowledge, especially in the life-sciences, where agentic pipelines are growing fast. Access to the literature is a crucial part of that need, and resources such as Europe PMC, with over 40M indexed records, are widely used to meet it. Yet these resources were not built for AI agents: they take keywords and complex syntax and return whole papers, so every agent must learn the syntax, issue several searches, and read full papers to find the evidence it needs. We introduce EMBL AI Librarian, a knowledge layer that upgrades the Europe PMC interface for AI agents: an agent asks in natural language and receives evidence that answers it. A single LLM orchestrates the whole knowledge retrieval process: it plans complementary subqueries executed by the live Europe PMC search engine, then reads the selected papers and locates the relevant evidence. We evaluate Librarian across four benchmarks: literature synthesis, claim verification, open-domain question answering, and downstream biology tasks such as protocol questions and sequence manipulation. On ScholarQABench, Librarian improves Citation F1 by more than $16$ points over strong recently published baselines. Used as the retrieval layer of an existing claim-verification pipeline, it increases agreement with expert consensus; and on the open-form LitQA2 benchmark, a GPT-5.4 agent scores about $8$ points higher when grounded in Librarian than with web search. Overall, our results show that equipping life-science agents with the Librarian knowledge layer improves performance across a range of tasks. We release our code publicly at this https URL 

---
# Causal Discovery with Inverted Self-attention for Multivariate Time Series 

**Authors**: Yusen Liu, Yong Wang, Yifan Yin, Tianqing Zhu, Xiufeng Liu, Huan Huo  

**Link**: [PDF](https://arxiv.org/pdf/2607.28212)  

**Abstract**: Causal discovery in multivariate time series data is challenging due to complex interactions, high dimensionality, and nonlinear dependencies among variables. Existing methods often struggle to capture these complexities, resulting in inaccurate causal structures. To address this issue, we propose a novel framework that leverages self-attention mechanisms within the transformer architecture for causal discovery. Our approach introduces a novel inverted causal self-attention mechanism (CSAM) that emphasizes latent and indirect causal relationships by inverting tokens and inducing sparsity in attention scores, focusing on significant causal interactions and reducing spurious correlations. Additionally, we develop a global causal algorithm to identify global causal links, providing a holistic metric for causal influence, along with a causal verification module to ensure robustness in the identified causal relationships, enhancing the reliability of our framework. Experiments on both linear and nonlinear datasets, along with ablation studies and sensitivity analyses, show that our framework outperforms existing methods, demonstrating its potential for causal discovery in complex multivariate time series. 

---
# Fidelity Is Not Safety: Gently-Compressed LLMs Pass Every Data-Free Quality Guard Yet Invent Procedure Steps in Agentic Execution 

**Authors**: I. Kennedy, T.Kennedy  

**Link**: [PDF](https://arxiv.org/pdf/2607.28196)  

**Abstract**: Practitioners accept a compressed language model once it clears a stack of data-cheap quality guards: perplexity within a small factor of the original, downstream accuracy (for example MMLU) inside a confidence interval, and data-free output-fidelity signals that compare the compressed and original network's internal representations under random probe inputs. This stack has a blind spot. Across three model families, gently-compressed models clear every guard and then invent procedure steps that were never in the instructions when they run a standard operating procedure (SOP) as an agent. The effect is operator-specific: coherent low-rank (SVD) truncation induces it, and magnitude pruning matched to the same perplexity does not. One dissociation isolates the cause. The same compressed weights that CI-win a paired output-fidelity test CI-fail the invented-step canary. The governing axis is the coherence of the compression error times its rate; the magnitude of the damage does not predict it. The data-free fidelity probe is a fidelity oracle by construction, so it cannot see this axis. We characterize the blindspot and dissociation with paired confidence intervals on a pre-registered, powered canary across three architectures. Operator-specificity replicates on all three, and the perplexity-guard evasion appears where the model admits in-guard low-rank headroom. We then give a data-free screen: a two-axis statistic of the compression error (coherent-fraction and error-rate) that flags the failing builds with fixed thresholds across architectures and matches the coherence-times-rate mechanism. Perplexity, MMLU, and fidelity acceptance do not certify agent safety. Screen gently-compressed low-rank builds before agentic deployment 

---
# The MADRS Pipeline: Supporting Depression Assessment in Clinical Trials 

**Authors**: Mila Fodor, Katalin Ócsai, Francesco Periti, Rien Sonck, Alex Boudreau  

**Link**: [PDF](https://arxiv.org/pdf/2607.28190)  

**Abstract**: Depression is a major mental disorder for which diagnosis relies primarily on clinical assessments. Automated methods to support its detection via the psychiatric MADRS scale are getting more and more attention. While existing solutions primarily focus on detecting the disorder from different text sources (e.g., online text, social media), there is still limited support for clinical trials, where clinical assessments are conducted through structured interviews based on standard guidelines such as SIGMA. In this work, we develop a LLM pipeline specifically designed to support clinicians in supporting the assessment of depression in patients enrolled in clinical trials. Our pipeline converts audio interviews into transcripts, maps them into the ten MADRS symptom items, estimates their severity, and identify problematic clinical ratings associated with them. Evaluation on real clinical interviews shows a strong overall correlation of 0.867 with expert ratings, providing interpretable support for future assessments in clinical trials. 

---
# Where and When to Commit: Candidate-Aware Decoding for Diffusion Language Models 

**Authors**: Chia-Ming Lee, Ming-Ching Chang, Xin Li, Yu-Lun Liu, Chih-Chung Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28166)  

**Abstract**: Diffusion language models (DLMs) expose a provisional prediction at every denoising step, creating an opportunity for generation-time early exit that stops decoding before the schedule is exhausted. Existing early-exit gates decide termination from fixed-region confidence statistics or schedule-dependent rules, evidence too coarse for a decision that freezes every remaining position at once, so they fire prematurely on long chain-of-thought outputs whose answers stabilize only near the end. Adaptive sampling, the other axis of training-free acceleration, paces how quickly positions commit while decoding continues but never verifies that the output itself has stabilized. We introduce a training-free, candidate-aware early-exit framework that keeps the two axes separate and matches each decision to evidence of its own scope. Confidence-Verified Commit (CVC) governs when the sequence may stop by verifying confidence and sustained argmax stability over the dynamically extracted candidate span using a deterministic parser specified from each task's output format. Block-Wise Early Commit (BWEC) governs where to accelerate by applying a cheaper local rule to non-final blocks, while leaving the final block and global termination under CVC. We refer to their combination as LATCH (Localized Acceleration with Tracked-Candidate Halting). Unlike prior methods, LATCH needs no suffix-prompt construction; it is prompt-anchor-free but format-aware. We evaluate LATCH end to end on 11 tasks under zero-shot settings using LLaDA and Dream. LATCH stays within 2.0 percentage points of full-decoding accuracy across all 22 evaluation settings, with one frozen hyperparameter set that transfers cross-backbone untuned, while achieving end-to-end TPS speedups of 9.3-17.8x on short-answer tasks and 2.0-3.3x on long-reasoning tasks. 

---
# RRM: Experience-Driven Reflective Retrieval Memory for Long-Horizon Multimodal Reasoning 

**Authors**: Jingxiang Fan, Junbao Zhuo, Bochao Zou  

**Link**: [PDF](https://arxiv.org/pdf/2607.28156)  

**Abstract**: Existing multimodal long-term memory agents use external memory to overcome the limited context available for long videos. However, most methods emphasize what to store rather than how stored memory should be retrieved. When retrieval becomes inaccurate or repeatedly fails to obtain useful evidence, existing agents lack mechanisms to diagnose failures from previous task trajectories and adapt future search this http URL introduce Reflective Retrieval Memory (RRM), a reflective memory framework for long-horizon multimodal reasoning. RRM augments an entity-centric multimodal memory graph with reflective experience memory, which distills transferable procedural retrieval knowledge from historical task trajectories. Unlike episodic and semantic memories that preserve factual evidence from the current video, reflective experience memory captures reusable search strategies across tasks. RRM converts retrieved experiences into query-level guidance, while answer generation remains conditioned only on factual evidence newly retrieved from the current video. A lifecycle management mechanism further regulates experience memory through usage frequency, reuse feedback, and temporal decay, thereby reducing redundancy and noise. RRM consistently outperforms previous state-of-the-art approaches on M3-Bench-Robot, M3-Bench-Web, and Video-MME-Long, demonstrating the effectiveness of reflective retrieval memory for long-horizon multimodal reasoning. 

---
# Can Agents Deceive? Evaluating Reasoning and Deception in ParliamentBench using a Social Deduction Game 

**Authors**: Niklas Bauer, Lars Benedikt Kaesberg, Akiko Aizawa, Jan Philip Wahle, Bela Gipp, Terry Ruas  

**Link**: [PDF](https://arxiv.org/pdf/2607.28146)  

**Abstract**: As large language models (LLMs) are deployed as agents in high-stakes settings, such as medical and legal systems, understanding their deceptive capabilities is fundamental to safety. Controlled social deduction games provide a reproducible proxy for isolating and evaluating these complex adversarial behaviors. We present the open-source benchmark framework ParliamentBench based on the game Secret Hitler to evaluate LLMs in scenarios that require deception, persuasion, and reasoning under information asymmetry. We evaluate 16 LLMs across 1,600 simulated matches playing each other, playing against humans, and compare them against a large set of online games. We introduce three novel metrics that isolate social deduction, reasoning, and deceptive consistency. Our experiments reveal that frontier models achieve strong performance across cooperative and deceptive roles, with a strong top-four cluster (GPT-5.4, Kimi K2.5, Grok 4.1 Fast, and DeepSeek 3.1 Terminus), whereas the weakest models fall short of random (33%) and simple algorithmic (45%) baselines. Most LLMs struggle to maintain a consistent deceptive persona throughout an entire game, with deception retention dropping below 50%. 

---
# Rethinking LLM-Judged Helpfulness as a Pedagogy Signal: A Pre-Registered Audit Across Tutor Models 

**Authors**: Shuyi Fan, Boyuan Deng, Mengyu Xu, Jiale Liu, Hongyang Zhang, Qiaoxin Yang, Chongyang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.28128)  

**Abstract**: LLM tutoring poses a measurement problem: can a general-purpose helpfulness rubric distinguish direct answer-giving from pedagogical guidance? We audit this signal in a pre-registered study. Within each of three tutor bases, we compare conversational and pedagogical policies instantiated with the same underlying model and paired with one fixed weak simulated student. Deterministic detectors measure answer leakage and next-turn independent work. Claude Opus 4.8 is the frozen, condition-blind primary judge. After the Opus scores were fixed, GPT-5.6 Sol was prospectively specified for a post hoc robustness audit of the same 1,179 confirmatory answer-phase tutor turns under the frozen helpfulness and pedagogy rubrics. On the primary base under Opus, the policies do not differ significantly in helpfulness but are perfectly rank-separated under the pedagogy rubric (Cliff's $|\delta|{=}0.10$ vs. $1.0$). Across the two judges, pedagogy contrasts retain their direction where detected, whereas the helpfulness ordering is judge-contingent, reversing between judges on two of three bases. In an Opus-only ablation, seven primary-base policies span $2.3$ points in mean judged pedagogy within a $0.25$-point band of mean judged helpfulness. Separately, answer-revealing turns are followed by less independent student work on every base, a result that is judge-invariant by construction. In this controlled setting, general-purpose helpfulness is not a reliable pedagogy signal. Tutor evaluation should pair pedagogy-targeted rubrics with deterministic process measures. 

---
# FinSMART: Financial Sentiment Analysis for Algorithmic Trading through Market-Aligned Reinforcement Learning 

**Authors**: Giorgos Iacovides, Wuyang Zhou, Danilo Mandic  

**Link**: [PDF](https://arxiv.org/pdf/2607.28127)  

**Abstract**: Recent advances in Generative AI have substantially improved financial sentiment analysis through post-trained financial large language models (LLMs). However, existing approaches remain confined to a market-agnostic, supervised learning paradigm that relies on limited, static and human-annotated datasets, and thus are incapable of adapting to evolving market conditions. To address this limitation, we introduce FinSMART, the first market-aligned reinforcement learning framework for financial sentiment analysis, which directly optimizes sentiment signals using realized market outcomes. To deal with the noisy, non-stationary, and multifactorial nature of financial markets, FinSMART incorporates a signal extraction pipeline that combines market-aware data filtering with a discrete asymmetric trading reward, enabling stable reinforcement learning from economically meaningful market feedback. Experimental results demonstrate that FinSMART significantly outperforms existing state-of-the-art methods in profitability, risk-adjusted performance, and sentiment signal quality, improving cumulative trading returns by 220% over the strongest baseline. Uniquely, the FinSMART framework naturally supports market-aware retraining, at any point in time, by replacing costly manual annotation with newly observed financial articles and their realized market outcomes. Such a retraining strategy enables the model to continuously adapt to changing market dynamics, resulting in consistent performance gains over its static counterpart. These findings demonstrate the practical applicability of market-aligned reinforcement learning and highlight its potential as a next-generation paradigm for developing adaptive financial LLMs. 

---
# Challenges in annotations by humans and LLMs: A case study of evaluative language 

**Authors**: Mirela Imamovic, Aenne Cecilia Kristine Knierim, Khushi Pitroda, Ekaterina Lapshinova-Koltunski  

**Link**: [PDF](https://arxiv.org/pdf/2607.28119)  

**Abstract**: In this paper, we draw a comparison between linguists in training, a trained linguist, and annotations generated by large language models (LLMs) to find out if they struggle with complex linguistic phenomena in a similar way. For this purpose, we analyse evaluative language in spoken popular science discourse, with the example of a corpus of English TED talk transcripts. We focus on the Appraisal theory and its Attitude subsystem, including the categories (classes) of Affect, Judgement, and Appreciation. In this context, Appraisal theory is an example of a highly subjective annotation task, making it a suitable example for the study of complex annotation challenges. First, we assess human annotations on a sentence level in specific scientific domains. Then, we develop three prompts and compare them for model performance for the automatic classification of Appraisal classes. We assess the performance of three LLMs using the best-performing prompt and finetune the model, reaching an F1-score of 0.77. We find that models perform best compared to annotations conducted by the trained linguist, while linguists in training do not reach high agreement scores. We conclude that LLMs can aid in complex annotation task resolution, opening new pathways for the complex theories annotated and analyzed in digital humanities studies. 

---
# GGC: Selective Query Correction for Reliable Text-to-SPARQL Generation 

**Authors**: Ziyi Yang, Thanh-Son Nguyen, Tuan Anh Nguyen, Lihui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.28082)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in structured query generation, making them a natural choice for Text-to-SPARQL, which translates natural language questions into executable SPARQL queries over knowledge graphs. However, their initial outputs remain unreliable: generated queries may be executable yet semantically misaligned with input questions, leading to incorrect retrieval. To address this issue, we propose Generator-Gate-Corrector (GGC), a framework for reliable LLM-based Text-to-SPARQL generation. GGC first uses a Generator to produce an initial query, then applies a Gate to predict whether correction is needed, and finally invokes a Corrector only for selected high-risk queries. This selective correction mechanism avoids unnecessary modifications and reduces the risk of degrading originally correct queries. Experiments on MCQA show that GGC improves query-level accuracy from 90.23\% to 98.33\% while reducing inference overhead by 45\% compared with correcting all generated queries. Ablation studies show that the Gate is robust across thresholds and that Corrector training data composition affects correction effectiveness and stability. Overall, the results demonstrate that selective correction enhances the accuracy, reliability, and efficiency of LLM-based text-to-SPARQL generation. 

---
# LEEPS: Latent-Guided Explore-Exploit Prompt Sampling for Efficient RLVR in Large Language Models 

**Authors**: Shuang Liang, Haoyang Zhou, Yifan Gong, Guowei Wang, Xiting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28077)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) improves the reasoning capabilities of large language models, but prompt groups with identical rollout rewards consume generation budget without effective learning signals. Pre-rollout prompt selection can reduce this waste by screening prompts before rollout generation. However, existing pre-rollout methods struggle to balance exploitation and exploration: repeatedly exploiting historically informative prompts can narrow training coverage, whereas broader exploration can lower the fraction of informative prompts. To address these limitations, we introduce LEEPS, a Latent-Guided Explore--Exploit Prompt Sampler that adaptively balances the reuse of previously observed informative prompts with continued exploration of uncertain ones. LEEPS partitions candidates into exploit and explore portfolios and adaptively allocates rollout budget according to their recent non-trivial ratios. It further uses representation-space neighbors and historical rollout outcomes to prioritize uncertain prompts likely to yield non-zero reward variance, thereby making exploration more targeted without additional rollouts. Across six mathematical reasoning benchmarks, LEEPS achieves the highest average score at both model scales, with relative gains of 2.6\% and 3.7\% over the strongest baseline for Qwen2.5-Math-1.5B and 7B, respectively, and generally improves faster during the training process. It also achieves the highest average score across the three evaluated OOD general-reasoning benchmarks at both model scales and adds only about 2 seconds of online sampling overhead per training step. Code is available at this https URL. 

---
# RepBench: Compiling Benchmarks into Capability Representations for Large Language Models 

**Authors**: Yanshi Li, Xueru Bai, Shuman Liu, Long Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28008)  

**Abstract**: Representation engineering reads and steers capability directions in large language models, yet methods are typically evaluated on paper-specific synthetic data. The resulting measurements are difficult to compare or reproduce and may reflect surface patterns rather than capabilities. We present RepBench, a benchmark-grounded data layer for capability-aligned representation probing. Crawling 13,427 benchmark papers yields a taxonomy of 182 capability clusters in 13 families; harvesting 353 public benchmark datasets yields 46,149 audited probe texts covering 94 capabilities, each supported by at least two independent benchmarks. This multi-benchmark design reduces dependence on any single source: raw per-text vectors exhibit no natural cluster granularity, whereas benchmark-pooled capability vectors show an interior clustering optimum at a small number of clusters on all 12 evaluated models, with low agreement to the human taxonomy. Under cross-benchmark transfer evaluation across twelve models completed by all four readouts, difference-in-means attains the highest model-level mean on ten models, while logistic regression wins the most capability-model cells. This disagreement shows that the readout method and aggregation criterion are meaningful evaluation dimensions. The pipeline, corpus, and evaluation code are released as a reusable closed-loop workflow. 

---
# Memory Decoder at Scale: A Pretrained, Parametric Long-Term Memory 

**Authors**: Rubin Wei, Jiaqi Cao, Jiarui Wang, Junming Zhang, Qipeng Guo, Bowen Zhou, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2607.27919)  

**Abstract**: Decoder-only language models entangle long-term memory and reasoning in a single parameter set, making it difficult to scale memory capacity independently. Memory Decoder introduces a parametric long-term memory module but only studies it at a relatively small scale. In this work, we present Memory Decoder at Scale, scaling memory models up to 6.9B parameters and pretraining them on 300B tokens. At this data scale, the combined cost of indexing and search makes a standard Faiss pipeline infeasible. We address this bottleneck with a distributed pipeline for Faiss indexing and retrieval, together with sparse, batch-wise loading of kNN distributions. Across model scales, we find that allocating more parameters to memory yields a better parameter-performance tradeoff than scaling the base model alone. On 17 benchmarks, pairing a 6.9B general memory with Pythia-410M raises its average score from 29.86 to 37.34, surpassing Pythia-12B (37.24) with 39% fewer total parameters. For Qwen3 Base models ranging from 0.6B to 14B, 1.7B domain memories improve the average score across the three domains by more than 9 points at every scale. Overall, our results demonstrate that independently scaling pretrained memory offers a more parameter efficient path to improving language model performance. 

---
# FinanceHarness: Autonomous Financial Deep Research Framework 

**Authors**: Yijia Xiao, Rujun Han, Yanfei Chen, Zifeng Wang, Ke Jiang, Zhongying CuiZhu, Vishy Tirumalashetty, Wei Wang, Burak Gokturk, Tomas Pfister, Chen-Yu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.27853)  

**Abstract**: Powered by advances in LLMs and autonomous agents, deep research has become one of the most widely adopted agentic products. However, most deep research systems write general-purpose reports, which are inadequate for financial deep research. Financial research demands specialized knowledge to analyze historical patterns and forecast upcoming events. Automating financial deep research therefore requires both a layered harness to drive the research agent and a verifiable, point-in-time benchmark that prevents leakage of future information. We present FinanceHarness, a harness that runs finance-oriented tools and practitioner-guided workflows, automating financial deep research end to end: environment and data construction, the agent execution loop, and reward modeling. We further propose FinanceGym, comprising thesis-driven research questions and rubrics that combine pre-cutoff and post-cutoff criteria. Professional expert validation yields an 82% pass rate. Even leading LLMs and agents score below 40% on the rubrics, showing that FinanceGym is challenging and leaves substantial headroom. With the same open-weight backbone, FinanceHarness improves the overall rubric score from 25.3% to 32.4%. FinanceHarness is available at this https URL. 

---
# Beyond Feeling Better: Capability-Sustaining Emotional Dialogue as a Longitudinal Research Paradigm 

**Authors**: Ming Wang, Jiaqi Wu Young, Wenfang Wu, Daling Wang, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2607.27851)  

**Abstract**: Emotional dialogue research includes two influential strategy traditions. Empathetic dialogue prioritizes understanding a speaker's emotional experience. Emotional support conversation selects and sequences support for the seeker's current needs. Sustained use introduces a further goal. Effective support should sustain users' capacities for emotion regulation, coping, self-endorsed decisions, and social connection across the interaction lifecycle. We propose capability-sustaining emotional dialogue (CSED) as a longitudinal research paradigm that aligns supportive strategy with this goal and organizes data, models, system design, evaluation, and governance around repeated use, non-use, transition, and termination. A targeted literature-and-corpus audit motivates this position. In a PRISMA-ScR-guided sample, 95% of 60 system-building papers pursue relief-oriented goals. None evaluates capability or longitudinal outcomes, and only 1 considers dependency, autonomy, or termination risk. In 300 ESConv supporter turns, capability-relevant functions appear in 43.0%, while generic suggestions account for 22.0%, compared with 4.0% reappraisal, 6.7% self-efficacy support, and 0.3% boundary behavior. We release a protocol for extending the audit to model behavior. An illustrative process model connects latent user capability to six design commitments, four evaluation timescales, and lifecycle constraints. The resulting agenda makes CSED testable across data, policy design, training, evaluation, and governance. 

---
# AutoSupervision: Closing the Feedback Loop in Scientific Workflows with Grounded Revision Verification 

**Authors**: Haobo Li, Eunseo Jung, Wenxiao Zhao, Feng Liu, Jiong Wang, Kaiyi Xu, Zijie Guo, Zixin Chen, Ben Fei, Fenghua Ling, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2607.27845)  

**Abstract**: Recent advances in large language models (LLMs) have enabled AI systems to assist scientific research and peer review. However, an essential capability for reliable AI-assisted scientific workflows remains underexplored: verifying whether reviewer feedback leads to meaningful and evidence-supported manuscript improvements. We introduce AutoSupervision, which evaluates whether scientific manuscript revisions genuinely address reviewer concerns through grounded evidence. AutoSupervision leverages transparent peer-review records as a natural source of supervision, where reviewer comments specify scientific concerns, author responses describe claimed resolutions, and revised manuscripts provide evidence of changes. Given reviewer comments, author responses, and revised manuscripts, models must characterize reviewer concerns, determine whether concerns have been addressed, and identify supporting manuscript evidence. We construct AutoSupervision from 56,000 Nature Communications articles and corresponding review records. Then we conducted experiments on LLMs, the ablation study, and the case study. Our results show that while LLMs perform well in characterizing reviewer concerns, with GPT-5.5 achieving a score of 0.754, evidence-based verification remains the primary bottleneck, with the best-performing model reaching only 0.501. 

---
# Beyond Borrowed Histories: Person-Aligned User Simulation for Interactive Role-Playing Evaluation 

**Authors**: Yuhang Zhu, Mingxuan Du, Benfeng Xu, Jie Gao, Lingyun Yu, Hongtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.27816)  

**Abstract**: Role-playing agents (RPAs) have become one of the most important consumer applications of large language models. Users engage in multi-turn conversations with RPAs for experiences such as emotional comfort, making reliable evaluation essential for measuring capability, comparing systems, and guiding further improvement. Existing benchmarks, however, typically require an RPA to continue a fixed dialogue history and then evaluate the continuation using a fixed rubric detached from the user. We identify and empirically demonstrate two limitations of this design. First, an RPA's output is shaped by the preceding dialogue history, preventing a scientifically grounded assessment of its role-playing ability in real multi-turn settings. Second, user experience varies substantially across individuals, and conventional fixed rubrics need not align with user satisfaction. We therefore introduce PALATE (Person-Aligned LLM-Simulated-User Assessment with Tailored Evaluation), a scalable RPA benchmark built on user simulators. PALATE is accompanied by a pool of 300 character profiles. Its main evaluation trains five per-user simulators and lets them engage candidate RPAs in free-form, multi-turn conversations over a pre-frozen panel of character profiles. Alongside a general quality rubric, we construct personalized rubrics to measure user satisfaction; on held-out annotated data, the personalized rubrics show higher agreement with human judgments than the general rubric. In the main evaluation of 16 candidates, PALATE separately characterizes generic turn quality, long-horizon session capability, and per-user experience on multi-turn trajectories co-constructed by each candidate. It thereby produces interpretable evaluations of specific user-RPA pairs rather than compressing systems into a single user-independent ranking. 

---
# Semantic-Aligned Structural Abstraction for Multimodal Sentiment Analysis 

**Authors**: Wei Chen, Junkai Li, Tongguan Wang, Hui Liu, Feiyue Xue, Chuanxiang Ma, Ying Sha  

**Link**: [PDF](https://arxiv.org/pdf/2607.27790)  

**Abstract**: Multimodal Sentiment Analysis (MSA) aims to interpret complex human emotions by integrating natural language with non-verbal modalities. Non-verbal modalities share a structural isomorphism with natural language, as both can be viewed as feature sequences evolving over time. This isomorphism enables the transformation of non-verbal modalities into text-like tokens for unified semantic reasoning. Large Language Models (LLMs), designed to understand and generate sequential data, can thus be utilized to interpret complex affective sequences. However, existing LLM-based methods primarily capture low-level superficial features, failing to model affective semantics arising from structural variations and contextual interactions. To address this limitation, we propose \textbf{SentiLLM}, a unified framework that leverages \textit{Semantic-Aligned Structural Abstraction} to distill continuous raw signals into compact, semantically meaningful tokens. Specifically, we introduce a \textit{Dual-Stream Salience-Context Calibration Mechanism}, which disentangles non-verbal feature sequences into a focus stream and an ambient stream. The focus stream captures salient sentiment shifts (e.g., facial expressions) guided by textual priors, while the ambient stream characterizes stable background states. Through calibrating these dynamic sentiment shifts against background states, SentiLLM effectively projects non-verbal modalities into a unified semantic space, making them naturally understandable for LLMs. Serving as a plug-and-play module, SentiLLM significantly improves discriminative performance with only a small number of trainable parameters. Our method achieves superior performance on four datasets, MOSI, MOSEI, CH-SIMS, and CH-SIMS v2, demonstrating the effectiveness of the structural abstraction paradigm in MSA. Our code is available at: \href{this https URL}. 

---
# Reasoning Consensus: Structural Ensembling of LLM Reasoning via Weighted DAG Aggregation 

**Authors**: Amruta Parulekar, Jinu Lee, Dilek Hakkani-Tür, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2607.27783)  

**Abstract**: Large Language Models (LLMs) explore problems through chain-of-thought, but this exploration is buried in unstructured prose. On high-stakes tasks, users cannot tell which steps are well-supported, which alternatives were seriously considered, or how the final conclusion compares to those the model discarded. We propose a framework that ensembles the reasoning structure, not just the answers, of multiple LLMs by weighted merging of Directed Acyclic Graphs (DAGs) extracted from reasoning chains. We weight each step by how many traces independently attest to it, to return "Consensus Reasoning". Across six benchmarks spanning statutory interpretation, graduate-level science, narrative multi-hop reasoning, and first-order logic, our ensemble outperforms a matched-budget majority-vote baseline, with a maximum accuracy gain of 3.1% on MuSR-MM (narrative multi-hop reasoning). On a single model, the framework matches or exceeds self-consistency at the same trace budget while additionally exposing an inspectable consensus reasoning graph. Ensemble weights correlate with LLM-judge rankings of reasoning quality at Spearman $\rho = 0.30$-$0.51$, and consensus subgraphs are preferred over alternatives leading to the majority-vote answer in 54.4-65.4% of head-to-head comparisons across five of six datasets. We observe that our framework can also be used to analyze diverse reasoning perspectives for a problem. 

---
# ChronoMem: Version Control and Semantic Rollback for Large Language Model Agent Memory 

**Authors**: Yongye Su, Wujiang Xu, Chaoji Zuo, Elisa Bertino  

**Link**: [PDF](https://arxiv.org/pdf/2607.27773)  

**Abstract**: LLM agents increasingly rely on long-term memory to support multi-session interaction and personalization. However, existing agent memory systems are designed around forward-only evolution, continuously accumulating, consolidating, and overwriting knowledge, with no principled mechanism to inspect, version, or revert prior states. This makes agents brittle under corrections, concept drift, and memory corruption, particularly after they have already been exposed to subsequent information. We present ChronoMem, a semantic version-control layer for agentic memory integrated into the production-ready, open-source Agent Development Kit by Google. ChronoMem commits whole-memory snapshots at each memory write, maintains structured version histories, and supports natural-language rollback requests by mapping undo intents to concrete historical versions through hybrid lexical and semantic retrieval, rank fusion, and reranking. We further introduce a post-exposure evaluation protocol that tests whether an agent can behave counterfactually after rollback by answering queries and summarizing history as if future updates had never occurred. On long-horizon conversational benchmarks augmented with evolving memory states and rollback tasks, ChronoMem substantially improves rollback-consistent question answering and history summarization relative to prompt-only and retrieval-only baselines, while achieving strong performance in semantic version selection. To our knowledge, ChronoMem is the first open-source system and benchmark for systematic semantic global memory rollback in LLM agents. 

---
# Gradient-free Task-Conditioned Retrieval for On-Device In-Context Learning 

**Authors**: Xinyu Luo, Hui Liu, Yihua Shao, Junyi Yang, Arindam Basu, Haoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.27766)  

**Abstract**: On-device in-context learning (ICL) relies on pre-inference retrieval to select demonstrations for useful context before downstream model inference. This retrieval must exploit task-specific information while operating over local memories under limited computation, memory, and data-exposure budgets. We propose Conditional Retrieval Alignment (CoRA), a gradient-free framework that converts a frozen encoder into a task-conditioned retriever using paired candidate inputs and outputs. CoRA selects complementary encoder layers, constructs an output-derived conditioning space from candidate memory, and aligns candidate input representations to this space through closed-form ridge regression. Low-rank factorization then produces a compact retrieval basis where candidate outputs are used only during offline index construction, whereas query-time retrieval requires only the query input and precomputed index. We show that CoRA's rank-constrained basis is the optimal low-rank compression of the output-conditioned fitted representation, and derive an exact two-pass streaming construction that avoids materializing the full fitted matrix. We further extend the framework to multimodal exemplar retrieval by incorporating visual representations into the conditioning and retrieval spaces. Experiments across ten textual datasets and four multimodal benchmarks with Llama-3.2-1B, MobileLLM-Pro, OpenFlamingo-3B, and Qwen3.5-2B, as well as end-to-end Raspberry Pi~5 deployment demonstrate that CoRA supports effective task-conditioned retrieval without retriever fine-tuning, backpropagation, or target-model calls. 

---
# Can LVLMs Uncover the Truth Behind Visual Illusions? An Analysis of Perceptual and Reasoning Capabilities 

**Authors**: Liangjie Zhao, Jiaqing Lyu, Kexin Tang, Zecheng Fang, Rong Yin, Yulan Hu, Da Li, Jianing Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.27747)  

**Abstract**: Large Vision Language Models have integrated reasoning capabilities, elevating cognitive performance to new levels. However, existing evaluations either focus solely on perception or rely on specific domains such as maths or coding. Evaluation for reasoning capabilities that align with an open-world environment is still required, especially one that considers perception and reasoning jointly. To bridge this gap, we propose to evaluate LVLMs by exploiting visual illusions as a diagnostic tool. Visual illusions are phenomena in which the human visual system misinterprets objective signals, resulting in an understanding that deviates from reality. We constructed IllusionReasoning, a benchmark of illusion images collected from the real world, incorporating diverse annotated question-answer pairs. Based on IllusionReasoning, we show that the reasoning capabilities of a wide range of LVLMs are not as advanced as claimed. Our work provides new insights into LVLMs and offers future direction for optimisation. 

---
# A Sparse Glimpse of the Whole: Train-Free Self-Speculative Decoding 

**Authors**: Yuesong Liu, Yuan Zeng, Min Lyu, Ruilin Liu, Yu Guo, Yinlong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.27735)  

**Abstract**: Speculative decoding alleviates the memory-bandwidth bottleneck in large language model inference, but its acceleration is jointly constrained by drafting overhead, token acceptance, and speculation length. We present a unified efficiency analysis showing that extending the speculation horizon can reduce rather than improve speedup when the marginal acceptance probability falls below the relative drafting cost. Guided by this analysis, we introduce SparseSpec-L, a training-free self-speculative decoding framework for long-context inference. SparseSpec-L generates lightweight drafts directly from the target model using a dynamically sparsified and recallable KV cache. It recycles per-head attention statistics produced during full-context verification as a no-extra-forward importance signal, allowing critical historical tokens to be recalled without permanently discarding the dense KV cache. An online entropy-based controller further selects the speculation length according to expected step-wise efficiency. Experiments across multiple long-context tasks and model scales show consistent end-to-end acceleration, with up to speedup over autoregressive decoding while preserving the target model's output distribution. 

---
# Recall Before You Rank: Similarity-Guided Top-$K$ Reuse for Efficient Long-Context Attention 

**Authors**: Wenshuai Yao, Wenyong Zhou, Hanyong Shao, Yizhe Chen, Zhiyuan Ning, Yuannuo Feng, Ru Huang, Kechao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27692)  

**Abstract**: Top-$K$ sparse attention reduces the cost of Softmax and value aggregation by attending to only a small subset of key--value (KV) entries. However, identifying this subset still requires scoring the current query against the full KV cache and performing global Top-$K$ selection, leaving selector cost linear in context length and limiting the practical efficiency of sparse attention for long-context decoding. In this paper, we introduce ReTopK, a training-free method that accelerates dynamic Top-$K$ attention by reusing historical retrieval decisions. ReTopK builds on the observation that similar queries often attend to overlapping supports and that partially overlapping supports can still preserve most of the Exact Top-$K$ attention mass. For each attention head, it maintains a bounded cache of historical query--support pairs, retrieves the most similar cached queries for each new query, unions their stored supports with a recent window, and reranks only the resulting compact candidate set using exact current-query scores. A similarity-based fallback invokes full-history Exact Top-$K$ when reuse is unreliable, while periodic exact refreshes limit cache drift. ReTopK retains the complete KV cache and reuses only selected indices, rather than historical scores, attention weights, or outputs. Across 16K--128K contexts, ReTopK achieves the lowest PG19 perplexity and the highest NIAH and LongBench scores among the evaluated approximate methods. At 128K with $K=512$, ReTopK incurs only a 0.50\% perplexity increase over Exact Top-$K$ while accelerating attention computation by $3.07\times$. 

---
# ICLE++: Modeling Fine-Grained Traits for Holistic Essay Scoring 

**Authors**: Shengjie Li, Vincent Ng  

**Link**: [PDF](https://arxiv.org/pdf/2607.27671)  

**Abstract**: The majority of the recently-developed models for automated essay scoring (AES) are evaluated solely on the ASAP corpus. However, ASAP is not without its limitations. For instance, it is not clear whether models trained on ASAP can generalize well when evaluated on other corpora. In light of these limitations, we introduce ICLE++, a corpus of persuasive student essays annotated with both holistic scores and trait-specific scores. Not only can ICLE++ be used to test the generalizability of AES models trained on ASAP, but it can also facilitate the evaluation of models developed for newer AES problems such as multi-trait scoring and cross-prompt scoring. We believe that ICLE++, which represents a culmination of our long-term effort in annotating the essays in the ICLE corpus, contributes to the set of much-needed annotated corpora for AES research. 

---
# From Single- to Cross-Document: Benchmarking Multi-Granularity Event Analysis of Large Language Models 

**Authors**: Tao Wen, Shuai Shao, Pei Ke, Xu Han, Jie Zou, Guannan Li, Tao Tian, Jinjie Qiu, Lan Wang, Ke Qin  

**Link**: [PDF](https://arxiv.org/pdf/2607.27654)  

**Abstract**: Event analysis is an essential and fundamental direction of information extraction, involving various event-centric tasks at different granularity of documents. While large language models (LLMs) have preliminarily achieved promising performance in part of these tasks individually, their capability in event analysis still lacks comprehensive understanding due to restricted document granularity, task designs, and data source of existing benchmarks. To address these limitations, we introduce MiGUE-Bench, a systematic benchmark for assessing the performance of LLMs in multi-granularity event analysis. To support large-scale evaluation, we first develop an LLM-driven self-correcting annotation framework called MiGUE-Pipeline, enabling scalable acquisition of high-quality source data of events with automatic labels. Then, we design four core tasks in our benchmark, i.e., event detection, relation reasoning, structure induction, and future prediction, to probe model competence at different levels, from atomic event details to complex cross-document narratives. Extensive experiments on state-of-the-art LLMs and retrieval-augmented generation (RAG) methods delineate the current capability boundary and identify critical deficiencies, providing insights into the future improvement of LLMs in challenging event analysis tasks. 

---
# Harness-G: A Graph-Structured Harness for Search Agents 

**Authors**: Yanning Hou, Haoyuan Chen, Sihang Zhou, Xiaoshu Chen, Xirui Liu, Duanyang Yuan, Lingyuan Meng, Quan Liu, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27652)  

**Abstract**: Reinforcement learning (RL) search agents commonly model retrieval as free-form natural-language query generation and optimize multi-turn interactions using final-answer rewards. Current studies mainly improve training with denser or more structured credit signals, but rarely examine whether retrieval is properly formulated at the policy-environment interface. We observe pronounced retrieval aliasing during Search-R1 training: rollouts for the same question continue to generate distinct query strings, yet their accumulated evidence sets increasingly overlap. We call this phenomenon retrieval-equivalence collapse; in this regime, trajectories approach utility equivalence with respect to retrieval decisions, leaving within-group returns with little effective retrieval contrast. To address this problem, we propose Harness-G, a graph-structured retrieval framework that redesigns this interface. It reformulates free-form query generation as finite action selection: the policy selects an evidence sentence or entity, or chooses to answer, while the environment constructs the menu, tracks retrieval state, and validates and executes each choice. This interface reduces linguistic aliasing and makes same-state alternatives directly comparable. Building on this interface, we introduce Structured Non-myopic Credit (SNC), which uses a frozen answer scorer to compare the selected action with its alternatives and assigns downstream gains to the earlier actions that enabled them. Across six QA benchmarks, Harness-G achieves the highest average F1 at both evaluated model scales, outperforming the strongest baseline, Graph-R1, by 10.74 points at 1.5B and 3.98 points at 3B. 

---
# DualAnchor: Preserving Language Priors and Improving Lexical Fidelity in Gloss-Free Sign Language Translation 

**Authors**: Hongbin Zhang, Junhao Liu, Xuefeng Bai, Youcheng Pan, Yang Xiang, Kehai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.27614)  

**Abstract**: Recent advances in large language models (LLMs) have led sign language translation (SLT), the task of converting sign-language videos into spoken-language text, to increasingly adopt LLMs as textual backbones. However, despite their strong language modeling capabilities, existing LLM-based SLT methods often undermine rather than exploit this language prior, producing disfluent translations, a failure we term language-prior degradation. Meanwhile, existing methods typically align videos and text at the sentence level, which does not ensure accurate lexical details and creates a lexical fidelity gap. To address both issues, we propose DualAnchor, a gloss-free LLM-based SLT training framework that couples two complementary anchors for linguistically fluent and visually faithful generation. Token-level Prior Anchoring (TPA) preserves the LLM's language prior by regularizing the multimodal decoder at each decoding step toward the next-token distribution of a frozen LLM conditioned on the same autoregressive prefix. Optimal Transport Alignment (OTA) improves lexical fidelity by formulating visual-textual matching as entropy-regularized partial optimal transport, with Sinkhorn optimization inducing a soft alignment between visual tokens and textual content tokens under a cosine cost. DualAnchor achieves strong overall performance on both PHOENIX-2014T and CSL-Daily. Targeted analyses attribute these gains to the complementary effects of the two anchors: TPA improves fluency, whereas OTA reduces fine-grained lexical errors. 

---
# AWARE-FX: An Auditable Knowledge-Guided AI System for Measuring Corporate Foreign-Exchange Hedging Disclosure 

**Authors**: Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27611)  

**Abstract**: Corporate annual reports contain weakly structured evidence about foreign-exchange risk management, derivative use, natural hedging, and explicit non-use. This study develops AWARE-FX, an auditable AI/NLP decision-support system that converts report text into traceable firm-year hedging-disclosure measures. The system combines a professional-source lexicon, negation and accounting-status logic, channel-specific financial encoders, exact evidence gates, conservative aggregation, and an audit ledger. Across 24,909 Hong Kong firm-years from 2008-2025, it retrieves and scores 543,527 snippets. Reliability is evaluated through ablations, a stratified 300-snippet human audit, three-seed FinBERT-ModernBERT comparisons, strict 2023-2025 temporal tests, probability calibration, selective prediction, and fixed-prompt generative-model benchmarks. FinBERT has the higher mean F1 in seven of eight encoder task-split comparisons; its temporal F1 ranges from 0.702 to 0.872. Abstaining on the 20% least-confident temporal observations raises retained-sample F1 by 0.050-0.077. Deterministic Qwen3-8B performs strongly on commodity and negation evidence but poorly on foreign-debt and accounting-context labels, showing that a general-purpose LLM does not uniformly replace domain constraints. The strict FX score is negatively associated with linked baseline and stress-period FX exposure, whereas the generic broad score is not. These associations provide external construct validation, not causal estimates of hedging effectiveness. AWARE-FX contributes a tested decision-support architecture in which retrieval, status logic, classification, uncertainty handling, aggregation, and external validation remain separately auditable. 

---
# Beyond Similarity: Grounded Agentic Extraction and Expert-Adjudicated Evaluation of Intertextuality in Classical Chinese Histories 

**Authors**: Zhaoji Wang, Wanyu Si, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27595)  

**Abstract**: Computational approaches to intertextuality have advanced from string matching to neural retrieval, yet their outputs, similarity scores and parallel-passage lists, identify where texts reuse one another without characterizing how or why. We recast fine-grained intertextuality extraction as an agentic task in which a large language model (LLM) reads two text units in full and, through a constrained tool interface, must ground each proposed reuse in exact character spans on both sides and label it under a five-dimension typology of reuse (form, aspect, source-marking, function, stance). We validate the approach on an exhaustive comparison of the Analects with the Book of Han, where three domain experts adjudicate a pooled multi-model candidate set into a benchmark of 2,533 intertextual pairs. Against this standard we study twelve LLMs, reporting precision (56%-93%), a 51$\times$ cost spread at comparable quality, and how well their confidence is calibrated. Expert agreement traces a reliability gradient: dimensions legible on the textual surface are annotated consistently, while those requiring inference of intent are contested, delimiting the claims such annotation supports. Scaling the validated extractor to the full Twenty-Four Histories (65,380 comparisons, 5,766 pairs) recovers corpus-level structure a similarity score cannot express. The interpretive composition of citation shows no systematic change across eighteen centuries, yet the same passage is quoted less and less literally. Stability in the aggregate with drift in the individual case is what a cultural-attraction account expects. We release the extraction protocol and the expert-adjudicated benchmark. 

---
# Training Skills Like Parameters via Self-Supervised Semantic Diffusion 

**Authors**: Mo Li, Zixin Yin, Ting Cao, Yunxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.27557)  

**Abstract**: While Large Language Models (LLMs) demonstrate remarkable general instruction-following capabilities, they often fall short of human experts in highly specialized, open-ended domains such as creative screenwriting. Prior approaches typically adopt post-training, yet both supervised fine-tuning and reinforcement learning require weight access that closed-source frontier models do not offer, and demand heavy compute. Moreover, what is learned is tied to a single checkpoint and cannot be inspected by humans. Recent advancements in agentic continual learning instead attempt to bridge this gap by accumulating external textual skills. However, these methods heavily rely on costly human expert annotations or unreliable LLM-as-a-judge feedback for reflection. To overcome this bottleneck, we propose a novel, unsupervised self-evolving agent framework inspired by the corruption-and-reconstruction paradigm of diffusion models. Instead of relying on explicit external scoring, we leverage existing high-quality human artifacts to construct self-supervised signals. Training then follows the familiar loop of neural network training, forward, loss, and backward, with the loss coming from contrasting the agent's reconstruction against the human original. What is updated is not model weights but an external library of textual skills. We evaluate our framework on the challenging task of short drama screenwriting. Experimental results demonstrate that our method enables the agent to autonomously extract and internalize highly generalizable skills, significantly enhancing its domain-specific generation capabilities. Furthermore, this self-contrastive reflection paradigm offers a scalable pathway for agents to teach themselves the production of complex, high-quality human artifacts, without requiring external supervision. 

---
# Belief Coevolution in a Social Network of Generalist and Specialist Large Language Models 

**Authors**: Germans Savcisens, Samantha Dies, Courtney Maynard, Tina Eliassi-Rad  

**Link**: [PDF](https://arxiv.org/pdf/2607.27512)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multi-agent environments. However, the processes by which beliefs form and propagate among interacting LLMs remain poorly understood. We introduce CoevolveSim, a framework for studying belief diffusion within networked LLM populations. CoevolveSim allows us to isolate and study three factors: domain specialization, social-role assignment, and social network structure. Within this framework, generalist and specialist LLM agents exchange and revise beliefs. In each round, an LLM agent observes a summary of its neighbors' beliefs before updating its own. We run 1,280 controlled simulations spanning four scenarios, two network structures, and 20 medical-indication statements. We find that persona-style role assignment and network structure reshape individual belief revision but have minimal effect on population-level consensus. In contrast, introducing (finetuned) specialist LLMs more than doubles the shift in consensus and gives rise to consistent asymmetries in exerted influence. We further show that simple persistence-based opinion-dynamics models reproduce collective outcomes in all-generalist LLM populations, whereas heterogeneous LLM populations require population-level belief composition to reproduce consensus and agent identity to predict individual belief transitions. Our results indicate that realistic simulation of belief diffusion in multi-agent LLM systems requires a diverse set of underlying LLMs, not persona prompting alone. 

---
# Models for minimalist RAG: B1ade 335M Embedding and 1B Parameter Small Language Models 

**Authors**: Shreyas Subramanian, Mecit Gungor, Vikram Elango  

**Link**: [PDF](https://arxiv.org/pdf/2607.27506)  

**Abstract**: Language and embedding models used in RAG systems are conventionally assumed to require large-scale pretraining and explicit grounding supervision. We present B1ade, an efficient RAG architecture comprising two purpose-built components: a compact embedding model and a purpose-built SLM. B1ade-embed, a 335M parameter retrieval model constructed via parameter-free fusion of five pretrained encoders achieves top MTEB scores among sub-500M models with zero additional training, and B1ade-1B, an SLM trained on low-cost GPUs using Group Relative Policy Optimization (GRPO) on 723M tokens (2.2M examples) of curated context-question pairs with rewards that optimize only answer similarity. Our central finding is emergent attribution: despite receiving no explicit supervision for source citation, B1ade-1B cites retrieved passages in 42.4% of responses, exceeding the attribution rate of its training distribution by 5.5 percentage points. This demonstrates that grounding behavior can emerge as an accuracy-maximizing strategy under RL training, without explicit reward engineering. On standard QA benchmarks, B1ade-1B achieves 81.82% on PopQA, 65.8% on PubMedQA, and 51.09% on FEVER. In end-to-end RAG evaluation, B1ade-1B achieves an average score of 0.654 across correctness, completeness, coherence, and faithfulness, a 10.8% improvement over the SFT, while closing the gap with models 1.5x its size. These results show that strategic model composition and reward design suffice for resource-efficient RAG, without large-scale pretraining. 

---
# SkillSmith: Learning to Compose Parametric Skills and Textual Knowledge 

**Authors**: Lucio M. Dery, Benedict Aaron Tjandra, Siavash Samiei, Adhiguna Kuncoro, Zohar Yahav, Jiajun Shen, Arthur Szlam  

**Link**: [PDF](https://arxiv.org/pdf/2607.27497)  

**Abstract**: Agentic systems driven by large language models (LLMs) regularly feature two key mechanisms to autonomously solve complex problems: synthesizing text-based knowledge and procedures from past experiences and building parametric (weight-space) skill libraries for recurring sub-goals. To date, research has largely treated these as orthogonal pursuits: either organizing textual knowledge through composition and reflection, or consolidating parametric skills via weight-space merging. Consequently, the seamless integration of text and model weights for targeted performance improvements remains largely unexplored. This work bridges this modality gap by treating model weights as an additional modality that an LLM can natively reason over. We instantiate parametric learning via prefix-tuning and augment an LLM to ingest both prefix weights and rich textual data which capture relationships to a target capability. Our augmented LLM, which we call SkillSmith, synthesizes these inputs to perform instruction-steered parametric synthesis, directly outputting new prefix weights that manifest the target skill. We demonstrate that our approach significantly outperforms both text-only and weight-space-only baselines, unlocking performance gains that are out of reach for uni-modal (text-only or weight-only) adaptations. 

---
# Selecting Open-Weight Language Models for Zero-Shot Intent Classification: A Systematic Evaluation of 41 Models 

**Authors**: Parishruthi Ganesh, Gerry Dozier, Cheryl Seals  

**Link**: [PDF](https://arxiv.org/pdf/2607.27421)  

**Abstract**: Intent classification is a core component of task-oriented dialogue systems, yet practitioners have limited systematic guidance for selecting deployable open-weight language models under compute, latency, and robustness constraints. We present a systematic zero-shot evaluation of 41 open-weight language models spanning 15 families and the 135M--9B parameter range across eight English single-label intent-classification datasets. A ninth dataset, ATIS, uses five labeled demonstrations and is reported as an auxiliary five-shot result. The evaluation includes standard benchmarks, a large-scale voice-assistant corpus, and production-derived e-commerce datasets. Beyond exact-match accuracy, we analyze confidence calibration, robustness to realistic input perturbations, statistical reliability of model rankings, deployment efficiency, and benchmark saturation. Our results show that instruction-tuned 3B models can outperform several evaluated 7B base models, that differences among leading models on MASSIVE are statistically indistinguishable under pairwise McNemar tests, and that widely used benchmarks such as SNIPS have become saturated and no longer meaningfully discriminate among current open-weight models. Instruction tuning's effect on confidence calibration is inconsistent rather than uniformly harmful. These findings provide practical guidance for selecting and evaluating open-weight language models for intent classification. 

---
# Benchmarking LLM Competence on Logical Inference over Probability Operators 

**Authors**: Nayera Hasan, Jack Greff, Alvin Grissom II  

**Link**: [PDF](https://arxiv.org/pdf/2607.27405)  

**Abstract**: Both expressions of uncertainty and inferences are ubiquitous in natural language, and valid inferences over natural-language expressions of uncertainty are necessary for not only everyday conversations but also for high-stakes domains such as medicine and law. While large language models are increasingly evaluated on logical reasoning tasks, disentangling principled, symbolic reasoning from clever surface-level pattern matching is fraught with difficulty. We introduce a benchmark for reasoning over probability operators--inference over sentences with gradable epistemic modals (e.g., probably, might, must) containing 14,320 procedurally-generated English prompts across fifteen inference templates, systematically varying question form, negation strategy, and surface content. Evaluating 29 models, we find that most show answer biases independent of the logical form, a systematic preference for Yes or No. We summarize this with a competence floor: the worse of a model's accuracy on Yes-correct and No-correct items. Only 9 of 29 models exceed random chance. We also test variations in question form, verb phrases/activity, and both the gender and origin of names used in the prompts, finding biases across every axis. 

---
# AHA-Memes: A Fine-Grained Multimodal Benchmark for Understanding Hate in Arabic Memes 

**Authors**: Mohamed Bayan Kmainasi, Ali Ezzat Shahroor, Abul Hasnat, Md. Rafiul Biswas, Wajdi Zaghouani, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2607.27393)  

**Abstract**: Hateful memes are a growing form of multimodal online harm, where hostile intent is often conveyed through the joint interpretation of images, text, cultural references, and implicit targets. While hateful meme detection has advanced in high-resource languages, Arabic remains underexplored, with existing meme resources focusing mainly on propaganda or coarse harmful-content labels. We introduce AHA-Memes (Arabic HAteful Memes), which is, to our knowledge, the first large-scale Arabic hateful meme benchmark with fine-grained, multi-label annotations. The dataset includes 5K manually annotated memes using a taxonomy that captures hate types, i.e., attack strategies. We further provide ~66K silver-labeled memes to support future studies. We benchmark text-only, image-only, and late-fusion multimodal models, as well as few-shot in-context learning (ICL) and open- and closed-weight Vision-Language Models (VLMs) under zero-shot and fine-tuning settings. Our results establish strong baselines and highlight key challenges in culturally grounded Arabic hateful meme detection. We release the dataset, annotation guidelines, and evaluation scripts to support future research. WARNING: This paper contains examples that may be disturbing to readers. 

---
# Same Facts, Different Diagnosis: Measuring and Mitigating Narrative Anchoring in Clinical Language Models 

**Authors**: Prabhjot Singh, Pritam Deka, Vijay Chennareddy  

**Link**: [PDF](https://arxiv.org/pdf/2607.27384)  

**Abstract**: Large language models used for clinical diagnostic reasoning are sensitive to sociolinguistic register, not just clinical content. We term this failure mode Narrative Anchoring: identical clinical facts expressed in different registers cause diagnostic outputs to diverge. Unlike prior demographic-bias work, which manipulates explicit identity tokens such as race or income, our benchmark isolates register as the sole channel of variation, with no demographic marker present in any form. We construct a dataset of 1,000 USMLE clinical vignettes, each rewritten into three sociolinguistically distinct personas under an independently audited fact-preservation guarantee, verified by a separate model that never sees the generation prompt. Across seven language models spanning three architecture families and scales, Narrative Anchoring is statistically significant under direct prompting in every model tested, with a Narrative Anchoring Gap of 0.064 to 0.151. Chain-of-thought reasoning and explicit debiasing instructions reduce the bias only partially, and their apparent gains are frequently confounded by accuracy collapse. We introduce NarrativeShield, a three-agent pipeline that structurally extracts and verifies clinical facts before diagnostic reasoning begins, reducing the Narrative Anchoring Gap to near-zero ($-0.004$ to $0.037$) and achieving the lowest rate of severely unstable decisions (DSS $<$ 0.8) of any method across all models, at a modest and mechanistically expected accuracy cost for most models. A stress test using a non-instruction-tuned base model shows that executing a debiasing intervention at all is gated by zero-shot instruction-following ability, not prompt content alone. We release our dataset, human-validated for fact preservation, as a standalone resource for studying register-based clinical bias. 

---
# HSS-Synth: Humanities and Social Sciences Data Synthesis for LLMs 

**Authors**: Ru Peng, Tianyu Zhao, Xijun Gu, Zhiting Fan, Haokai Xu, Jinyang Zhang, Yawen Zeng, Yihong Zhuang, Kexin Yang, Junyang Lin, Dayiheng Liu, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.27379)  

**Abstract**: High-quality, diverse data are vital for large language models (LLMs) but remain scarce and costly. Data synthesis is a viable alternative and succeeds on closed tasks, yet the humanities and social sciences (HSS) are overlooked, and their open-ended nature makes synthesis challenging. Moving beyond prior capability-centric, fragmented attempts, we adopt a subject-centric paradigm, define the first HSS domain system covering 14 mainstream fields, and introduce HSS-Synth, the first data synthesis pipeline for HSS. HSS-Synth comprises: (1) constructing seed documents from web corpora via multi-step filtering and text refinement evaluated by a judge; (2) specifying "requirements + persona" to backtranslate seed documents into diverse yet faithful instructions with a strict Q&A alignment check; and (3) breaking LLM response limits via teacher-forced Answering that feeds seed documents during response generation to anchor semantics, reduce hallucinations, and preserve tone and integrity. HSS-Synth yields 237k high-quality, diverse instruction-tuning samples that outperform 14 leading baselines on 16 benchmarks. The fine-tuned Qwen3-8B-Base sets a new SOTA and approaches the official Qwen3-8B, improving both human preference and knowledge capabilities without performance seesaws. Extensive experiments demonstrate HSS-Synth's robustness and transferability. Our code is publicly available at this https URL. 

---
# BridgeAlign: Bridging Preference Alignment for Humanities and Social Sciences 

**Authors**: Ru Peng, Haokai Xu, Xijun Gu, Tianyu Zhao, Zhiting Fan, Yawen Zeng, Yihong Zhuang, Jinyang Zhang, Kexin Yang, Jian Wu, Hao Chen, Junyang Lin, Dayiheng Liu, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.27366)  

**Abstract**: While data synthesis for large language models (LLMs) is prevalent, it primarily targets domains with verifiable answers, overlooking open-ended humanities and social sciences (HSS), where nuanced quality judgments matter more than objective correctness. This makes preference alignment a natural paradigm for broad HSS tasks. Yet existing methods are either costly or not tailored to broad HSS disciplines. We thus propose BridgeAlign, among the first preference-alignment pipelines for broad HSS disciplines, with three phases: i) Seed Curation: curating HSS seed documents from web corpora via heuristic/LLM-based filtering and text refinement; ii) Preference Data Synthesis: generating preference triplets via persona-based instruction inversion with Q&A consistency checks; iii) Preference Optimization: moving beyond naive human-vs-model heuristics by first grounding preferences in HSS quality rubric, then generating transitional responses via controlled quality degradation to form near-boundary preference pairs for finer-grained quality discrimination. Aligning over 210k synthetic preference samples, BridgeAlign enables Qwen3-8B to achieve the best average across 17 benchmarks against 11 strong baselines; importantly, leading on both human-preference and knowledge-based capabilities at once, with no trade-off between them, as supported by extensive experiments and contextualized by existing theories. 

---
# LayerRAG-Bench: A Cross-Layer Reliability Benchmark for Agentic Retrieval-Augmented Generation 

**Authors**: Musa Shams  

**Link**: [PDF](https://arxiv.org/pdf/2607.27353)  

**Abstract**: Agentic retrieval-augmented generation systems can produce answers that appear grounded while failing at the evidence, tool-contract, authorization, or session-state layer. We introduce LayerRAG-Bench, a controlled cross-layer reliability benchmark with 8 enterprise domains, 240 tasks, 9 fault scenarios, 2 contract modes, and 38,880 live task-level records across nine models from OpenAI, Anthropic, and Gemini. Schema normalization raises schema-drift success from 0.000 to 0.913, but stale evidence, missing tool output, denied permissions, and wrong-session context are not recovered by schema normalization. Groundedness-only evaluation also produces substantial false positives under stale and wrong-session evidence. These results support a layer-specific evaluation principle: a reliability intervention should be credited for repairing its target layer without being mistaken for a universal fix. 

---
# Sympathetic Framing: Evaluating AI Alignment across Sociodemographic Groups 

**Authors**: Haran Shani-Narkiss, Michael Fire, Oren Tsur  

**Link**: [PDF](https://arxiv.org/pdf/2607.27232)  

**Abstract**: Large Language Models (LLMs) are increasingly shaping how we consume information and form our worldview. This raises concerns beyond bias in AI: do LLMs grasp the emotional nuances conveyed via textual framing? In this work, we empirically evaluate how well an array of LLMs aligns with human emotional perception. Considering news headlines covering political and geopolitical conflicts, both human participants (n = 3011, a representative sample of the U.K. adult population, via a YouGov survey) and seven LLMs answered whether headlines evoked sympathy for a specified side in a conflict. We find that the correlation between AI and human evaluations varies across models, ranging from very high (0.789, GPT-5.2) to medium (0.4 ,Mistral Large 2512). Crucially, the leading models are broadly aligned with human judgments across all demographic subgroups, including age, gender, level of education, prior geopolitical knowledge, and participants' predispositions regarding the conflict, although there are statistically significant differences between groups. This research, with its robust design and large, demographically diverse dataset, offers the most comprehensive evaluation of LLMs' comprehension of news framing to date. Findings highlight an important, often-ignored aspect of differential alignment: even when aggregate performance is high, AI alignment is not universal -- it may correspond differently with demographic features and cultural norms. Considering or ignoring the need for differential alignment may therefore have significant implications for the development of ethical and useful AI systems. 

---
# AI-assisted pre-review of open-source software submissions: an experience report from BOSC 2026 

**Authors**: Tazro Ohta, Nomi L. Harris, Seth Carbon  

**Link**: [PDF](https://arxiv.org/pdf/2607.27228)  

**Abstract**: Most conferences rely on peer-review of submissions, but as generative AI makes it easier than ever to prepare submission materials, some conferences are seeing an overwhelming surge of submissions. We wanted to see if generative AI could help our conference's volunteer reviewers by pre-reviewing abstracts for certain criteria. The Bioinformatics Open Source Conference (BOSC) was well-positioned to experiment with this, as we already had a detailed rubric used by reviewers to evaluate submitted abstracts on multiple criteria, including openness (public availability of the code or other content associated with the project), valid open source license, and "runnability" (how easy it is to download, build, and run the project - an important measure of reusability). For BOSC 2026, we built bosc-pre-review, an agentic skill that assessed six review criteria, and Runabilly, which builds and tests each project in a disposable Docker container for safety. The AI only gathered evidence to present to the reviewers; humans made every decision regarding the acceptance of the abstracts. After the review period, we surveyed the reviewers to determine how useful they found the pre-review. Most of those who responded said they found it useful, but they preferred to check the AI's conclusions against their own, rather than accepting the AI results unquestioningly. 

---
# Prompt Chaining in Practice: A Case Study in Automated Scholarly Report Generation 

**Authors**: Andrei Lazarev  

**Link**: [PDF](https://arxiv.org/pdf/2607.27210)  

**Abstract**: The exponential growth of scholarly publications requires automated tools for effective information synthesis. However, simple, single-shot prompting methods often lack the reliability and quality required for complex synthesis tasks. This paper introduces and empirically evaluates a multi-stage prompt chaining methodology as a more reliable architectural pattern for such tasks. This approach is implemented in our system, AI SciBrief, which automatically generates scholarly digests. We conducted a comparative experiment, measuring the performance of our prompt chaining method against a carefully optimized single-shot baseline. Both systems were evaluated against a human-authored "gold standard" report for the "Education" domain. The results demonstrate a significant difference in reliability: our prompt chaining method achieved a 100% success rate, whereas the optimized baseline failed in 50% of its runs. In terms of quality, the proposed method also demonstrated a clear advantage, achieving a superior ROUGE-L F1-score (0.507 vs. 0.486), driven primarily by higher precision. We conclude that prompt chaining is a more dependable and effective engineering approach for complex, multi-step generative tasks, significantly mitigating the risks of failure and inconsistency inherent in monolithic prompts. 

---
# AISPA: User-Centric System Prompt Auditing for Large Language Model Applications 

**Authors**: Xiangning Lin, Shenzhe Zhu, Shu Yang, Zhenyu Zhang, Haoqian Zhang, Yipeng Zhao, Chengxuan Qian, Tianwei Wang, Ziheng Zhang, Zhenlong Yuan, Dingcheng Wang, Juncheng Wu, Yuan Si, Jiaxin Liu, Baolong Bi, Robert Mahari, Tobin South, Dazza Greenwood, Zexue He, Rishi Bommasani, Sophia Kazinnik, Andreas Haupt, Samuele Marro, Erik Brynjolfsson, Alex Pentland, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2607.28617)  

**Abstract**: System prompts are instructions configured by developers to govern the behaviors of foundation models in AI applications. They are used throughout commercial AI products, but are rarely disclosed to the public or regulators, creating a serious trust and accountability gap in the wide deployment of AI systems. In this paper, we introduce Artificial Intelligence System Prompt Assurance (AISPA), a user-centric framework for systematically auditing system prompts in AI systems. AISPA examines specific parts of a system prompt and evaluates them along eight dimensions that matter to users. We then use this framework to review 3,249 instructions from system prompts in 88 commercial AI products, classifying each instruction as either protective (of users) or problematic. Our audit surfaces four core findings. First, system prompt design varies substantially across products and developers, with some organizations averaging over 60 protective instructions per product while others average fewer than 5. Second, protective instructions are widely adopted but shallow in scope: 98.9% of products contain at least one, yet only 24% cover all eight dimensions of the AISPA taxonomy. Third, system prompts have grown steadily longer and more protective of users, suggesting that user protection is becoming a more visible concern in commercial prompt design. Fourth, despite this progress, problematic instructions remain pervasive: roughly 40% of products contain at least one instruction that works against user interests, and protective and problematic instructions frequently coexist within the same prompt. Our findings highlight the need for greater transparency, standardization, and independent oversight for system prompts in commercial AI products. 

---
# OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models 

**Authors**: Qiushi Sun, Kanzhi Cheng, Yian Wang, Bowen Yang, Hang Yan, Liheng Chen, Fangzhi Xu, Zichen Ding, Nuo Chen, Jialin Cao, Xingdong Gong, Zehao Li, Kaiming Jin, Xinfeng Yuan, Zhoumianze Liu, Jingyang Gong, Zhangyue Yin, Jiahui Gao, Zhiyong Wu, Tianbao Xie, Jianbing Zhang, Ben Kao, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2607.28609)  

**Abstract**: Computer-using agents (CUAs) are advancing rapidly across the digital world. A CUA trajectory records the agent's actions, states, and reasoning. Verifying whether it fulfilled the task instruction is central to CUA evaluation, data curation, and reinforcement learning. Neither human-written verifiers nor human annotators can provide such verification at scale, so the field increasingly turns to vision-language models (VLMs) as judges of CUA trajectories. But a fundamental question has long gone unexamined: are these VLM judges reliable enough? To study it systematically, we introduce OSReward, a realistic, high-quality benchmark that evaluates VLM judges on CUA trajectories. The trajectories come from diverse agent backbones executing human-verified instructions across platforms, then rigorously labeled with ground-truth verdicts through multi-stage human annotation. Building on it, we derive OSReward-Hard, a challenge set concentrating genuinely hard cases, and OSReward-Multi for fine-grained efficiency and alignment scoring. The most comprehensive evaluation of VLM judges to date finds even state-of-the-art models fall short of an ideal judge, sharing a systematic leniency bias that mislabels failed runs as successes. The few reliable enough to trust are too expensive to run at scale, while affordable open models trail far behind. To close this gap, we construct and release OS-Shepherd-100K, an open corpus of reasoning-annotated trajectory judgments for the CUA community. On it, we train OS-Shepherd (9B and 35B), open reward models that supply low-cost, stable, and reliable reward signals, matching commercial judges at 30-60% lower cost than the frontier. Extensive analyses further inform the design of reliable CUA reward at scale. Our code, benchmark, dataset, and model checkpoints are available at this https URL. 

---
# Change2Task: From Repository Changes to Executable Coding Agent Tasks and Environments 

**Authors**: Haomin Qi, Xingliang Wang, Xuanqi Gao, Baihui Sang, Xin Zhang, Minghua Ma, Pengfei Gao, Yu Kang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28591)  

**Abstract**: Scaling coding agents requires a continuing supply of executable data for training, benchmarking, and continuous evaluation. Each task must couple a realistic software state with a specification, development tools, and reliable verification. To expand this supply, we present Change2Task, a system grounded in repository history that converts merged pull requests into verified tasks on healthy modern revisions of the same repository. It aligns historical evidence with evolved code, reconstructs task states through Patch Reversal, Code Mapping, or Agent Reconstruction, and validates the lifecycle from a healthy base to a task state and a restored state. By deriving multiple tasks grounded in developer evidence from maintained environments, Change2Task provides executable data for coding agent training and evaluation while reducing repeated environment setup, storage, and task construction effort. We evaluate the system through five common and widely adopted coding agent task families: Bug Fix, Feature Addition, Test Generation, Application Programming Interface Migration, and Security Repair. Starting from 1,130 source changes eligible for construction, Change2Task achieves 79.6% verified task construction success across these task families. On a matched candidate set, it recovers 29.2% more verified tasks than a construction baseline based on pull requests. Historical and reconstructed cases achieve up to 98.0% matched outcome agreement under agent evaluation, while reuse of modern bases reduces measured expenditure across the complete pipeline by 10.8%. 

---
# VAD: Attributing Visual Evidence for Target Reconstruction in Multimodal On-Policy Distillation 

**Authors**: Kangning Zhang, Yixing Li, Shuai Shao, Qingyao Li, Zhengxi Lu, Zhiyuan Yao, Jianghao Lin, Wenxiang Jiao, Yuan Lu, Weiwen Liu, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28590)  

**Abstract**: Multimodal on-policy distillation (OPD) transfers fine-grained visual knowledge by supervising student-generated trajectories with a privileged-view teacher. Yet its next-token corrections are source-mixed, combining visual signals with linguistic priors and teacher-specific effects. The key challenge is to estimate which corrections are supported by visual evidence, not merely where or how strongly to distill. We introduce Visual Attribution Distillation (VAD), a counterfactual target-reconstruction algorithm that estimates the visually attributable part of a teacher correction. At each student-generated prefix, VAD evaluates the same fixed teacher with the relevant evidence present and removed. The corresponding change in centered log-probabilities defines ut, a signed proxy for the visual evidence direction that estimates how revealing the evidence supports or refutes candidate tokens. VAD projects the original correction onto this proxy to obtain an intervention-aligned component and a proxy-unexplained residual, then reconstructs a student-anchored target from the former. During training, this reconstructed target supplies the primary supervision signal, while the privileged teacher contributes a weak regularizer. Across six fine-grained visual benchmarks at 4B and 9B scales, VAD outperforms direct privileged-view distillation and visual-advantage weighting. Token- level and controlled-target analyses show that the proxy-aligned component is enriched in task-relevant visual corrections and yields stronger target shifts, especially when evidence refutes a mistaken answer. These results support counterfactual target reconstruction as an effective alternative to source-mixed supervision. 

---
# TCA-SIR: Learning Target-Conditioned Abstractions for Scientific Inspiration Retrieval 

**Authors**: Yuto Suzuki, Farnoush Banaei-Kashani  

**Link**: [PDF](https://arxiv.org/pdf/2607.28498)  

**Abstract**: Scientific hypothesis generation for AI for Science typically involves Scientific Inspiration Retrieval (SIR) followed by hypothesis composition. Existing SIR methods rank papers by topical similarity and do not explicitly represent how a candidate inspiration transfers to a target problem. This is especially limiting for remote inspirations, whose value often lies in reusable problem-solving principles rather than topical overlap. Motivated by how humans abstract transferable aspects of a source and remap them to a new target, we reformulate SIR as target-conditioned abstraction (TCA). The retrieval object is a transferable abstract principle extracted from a candidate specifically for the target. We present TCA-SIR, which learns to generate target-conditioned abstractions and uses their representations to predict transferability. On ResearchBench, TCA-SIR outperforms prior SIR methods and direct LLM retrieval, improving HitRate@top4% over MOOSE-Chem by more than 10 percentage points. Learned abstractions also recover target-relevant mechanisms more clearly than an untrained TCA prompt, yielding both stronger retrieval and an interpretable rationale for scientific inspiration. 

---
# Stage-Replay Divergence Follows the KV Cache: Fixed-Prefix Precision Controls and Bidirectional Cache Transplantation 

**Authors**: Alexander Boesgaard Lorup  

**Link**: [PDF](https://arxiv.org/pdf/2607.28495)  

**Abstract**: Stage-replay diagnostics reconstruct intermediate token prefixes and treat fresh-prefill continuation as continuation from the decoder state that originally reached the prefix. We audit that assumption at a whole reasoning-stage boundary in a Qwen2.5-derived system. A matched 200-item experiment compares retained live cache with one-shot prefill of identical integer tokens and places an exact replica on both sides. In BF16, replicas remain exact while the constructions differ on 166 suffixes and 20 correctness labels; the accuracy difference is only one point (paired 95% CI [-3.5, +5.5]). A fixed-prefix 2x2 holds all 200 token states constant while crossing construction and precision. The BF16 disagreements recur, whereas FP32 produces no decoded disagreement (95% Wilson upper bound 1.88%). A prospective bridge makes token-by-token incremental and retained live caches bit-exact on 12/12 rows; an all-200 saved-ledger audit reproduces every retained trajectory and comparison fingerprint. Bidirectional transplantation of all 48 key/value layers makes every tested divergent continuation follow its cache donor, both on a selected set at the primary checkpoint (24/24) and an outcome-blind replication at a later checkpoint (43/43). Exact-token replay can therefore be repeatable without preserving live-state fidelity. On the tested states, boundary K/V cache is a causally sufficient carrier of the divergent trajectory, while numerical precision moderates its behavioral expression. 

---
# SVR: Self-Verifying Refinement via Joint Verdict-Confidence Reinforcement Learning for Adaptive Test-Time Compute 

**Authors**: Hongyu Chen, Liang Lin, Guangrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28457)  

**Abstract**: Scaling test-time computation can improve language-model reasoning, but uniform budgets waste computation on easy inputs, while verifier-guided refinement relies on external feedback. We introduce Self-Verifying Refinement (SVR), an oracle-free multi-turn reinforcement learning framework that learns to use self-verification as a compute-control policy. At each turn, the model produces a solution together with a discrete correctness verdict and a confidence score; it retains the current answer only when the verdict is Correct and confidence exceeds a threshold, and otherwise continues refinement using its own self-verification. Ground-truth correctness is used only to construct training rewards and is never exposed to the policy through refinement prompts or required at inference. SVR is trained with GRPO on fixed-horizon trajectories using rewards that promote solution correctness, calibration-aware self-verification, and stop-ready correct states; adaptive stopping is activated only at inference. On seven mathematical reasoning benchmarks with Qwen3.5-2B, SVR achieves a macro-average accuracy of 0.563 with only 2.99 inference turns on average. In the evaluated complete-system comparison, it exceeds standard GRPO, strong multi-turn baselines, and a fixed-budget oracle-guided score-feedback reference while requiring substantially fewer turns than fixed ten-turn inference. These results demonstrate that learned self-verification can serve as an effective internal control signal for answer retention and adaptive test-time compute allocation. 

---
# Metaphor Tracer: A Theory-Informed Analysis of Hidden States 

**Authors**: Marc Heimann, Roxana Assadi Moghaddam, Olga Brovkina, Mark Pettifor, Lutz Goetzmann  

**Link**: [PDF](https://arxiv.org/pdf/2607.28434)  

**Abstract**: What do a language model's hidden states say about the organization of a single text? From one forward pass, without training, we score every token position on two properties. The *aggregator* measures whether the position consolidates the whole text into a stable configuration. The *differentiator*, whether other tokens are transiently carried into its subspace as the model reads: metaphor in its root sense, transport. Constants were frozen on one discovery text; every other is confirmatory.
The aggregator is not, in the classic sense, an information measure, nor a measure of salience. Across three unrelated models, as a signifier repeats, its surprisal and its attention drain while its aggregator score holds: the channel marks a token's place in the text. That this tracks a reading rests on independent ground truth: an engineered register the aggregator follows across its boundaries (6/6 cells), and a psychoanalyst's marking of clinical transcripts, fixed before the instrument existed, in 34/36 cells, with a graded increment above lexical controls and dissociations no type-level measure reproduces. A transfer test gives the result its shape: the model whose token structure travels with lexical type reads the singular discourse worst, and in a matched base/instruct pair tuning raises fidelity without moving type-transfer. Structural value is a property of a token's place in *this* text, not of its vector alone: a relational rather than essentialist reading of hidden states, operationalizing theory that predated the instrument. 

---
# WIDE: Boosting Adaptive LLM Inference via Token-level Dynamic Width Pruning 

**Authors**: Haozhe Hu, Hao Wu, Peiran Yin, Chao Han, Yunpu Ma, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2607.28418)  

**Abstract**: Pruning is a promising approach for improving the efficiency of LLMs. Existing static structured pruning methods are hardware-friendly and can deliver practical throughput gains, but their input-agnostic computation allocation often causes substantial accuracy degradation under aggressive sparsity. Recent dynamic sparsity methods improve quality retention by adapting computation to individual inputs, yet they remain largely limited to coarse-grained structural decisions and their practical acceleration under real-world inference scenarios remains challenging. To address these challenges, we present WIDE, the first end-to-end differentiable token-level dynamic width pruning framework designed for both prefill and decode scenarios. WIDE enables fine-grained computation allocation by allowing each token to dynamically select attention-head groups and FFN-channel groups, extending dynamic pruning beyond layer-level decisions to neuron-block-level granularity. Through a two-stage training pipeline, WIDE learns effective token-wise sparse execution patterns and achieves substantially better quality retention than existing approaches. To make such fine-grained dynamic pruning practical, we further propose a pruning--kernel co-design framework that decomposes dynamic sparsity acceleration into mask reordering, hardware-agnostic block-level skipping, and hardware-dependent intra-block skipping, enabling efficient execution across different granularities. At 50% sparsity, WIDE provides 55.1% performance boost when compared to the state-of-the-art dynamic depth pruning under calibration-only settings. Under prefill and decoding inference workloads, WIDE achieves close-to-theoretical kernel-level speedups of up to 1.98x for prefill and 4.95x for decoding, as well as 1.68x and 1.55x end-to-end acceleration. Our code is available at this https URL. 

---
# Can Large Language Models Execute Parent Orders? 

**Authors**: Zane Shen, Xinli Xu, Guangyi Zhang, Jialong Chen, Jinsong Zhou, Cong Chen, Guibao Shen, Dongyu Yan, Luozhou Wang, Zhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28410)  

**Abstract**: Parent-order execution is a core problem in algorithmic trading, where the goal is to split a large order into smaller orders while reducing execution costs. Existing approaches either rely on pre-specified market assumptions that may not hold in practice, or require task-specific training that limits adaptability to new settings. To overcome these limitations, we present the first systematic study of large language models (LLMs) for parent-order execution. This extends the use of LLMs in finance from what to trade to how to execute. We propose PACE (Plan-Ahead Controlled Execution), a hierarchical framework that decomposes parent-order execution into long-horizon planning and short-horizon execution, requiring neither explicit market assumptions nor task-specific training. Experiments on Shenzhen Stock Exchange Level-1 data show that PACE outperforms TWAP, Almgren-Chriss, and learning-based baselines, exceeding the strongest baseline by 0.65 bps. Behavioral analysis reveals that LLMs make execution decisions differently from human investors: higher model confidence predicts better performance rather than worse returns, and the model trades earlier rather than procrastinating toward the deadline. These findings suggest that LLMs can complement human traders in execution decisions. 

---
# GLM-RAG: Graph Language Models for Graph-Based Retrieval-Augmented Generation 

**Authors**: Maya Arseven, Anette Frank, Beni Egressy, Johann Higl, Moritz Plenz  

**Link**: [PDF](https://arxiv.org/pdf/2607.28397)  

**Abstract**: Retrieval-augmented generation (RAG) over knowledge graphs requires retrievers that can effectively capture both graph structure and semantic information. Recent approaches have explored graph neural network (GNN)-based retrievers to model graph topology in multi-hop reasoning tasks. In parallel, graph language models (GLMs) have emerged as a promising paradigm that integrates graph reasoning and the semantic capabilities of language models. In this work, we introduce a GLM-based retriever and investigate the comparative strengths of GLM-based, GNN-based, and traditional vector-search-based retrievers in single- and multi-hop RAG settings, and with a particular focus on transferability to unseen domains. Our findings suggest that finetuned GLM retrievers generalize better out of domain, achieving SOTA on two multi-hop benchmarks. On in-domain multi-hop QA datasets they remain comparable to prior work, with promising scaling as parameters and subgraph coverage increase. GNN-based retrievers achieve higher graph coverage with an efficient training setup, whereas the vector-search baseline excels at single-hop datasets. 

---
# CDAE: Enhancing Perturbation Robustness in Pretrained Language Models with Contrastive Denoising 

**Authors**: Sina Heydari, Amirreza Abbasi, Mohsen Hooshmand, Majid Ramezani  

**Link**: [PDF](https://arxiv.org/pdf/2607.28236)  

**Abstract**: Pre-trained language models have significantly improved sentence representation learning, yet their embedding remain sensitive to semantic preserving textual perturbations such as synonym substitution, masking and word dropout. This work proposes a lightweight Contrastive Denoising Autoencoder (CDAE) that refines pre-trained BERT embedding by jointly optimizing contrastive and reconstruction objective to learn perturbation-invariant representation. We evaluate the proposed framework using multiple perturbation strategies with varying strengths and compare it against the original BERT embeddings and SimCSE. Experimental results show that CDAE consistently preserves higher embedding similarity under perturbations, with the improvements becoming more pronounced as framework effectively enhances representation stability while preserving semantic information, highlighting perturbation-invariant learning as a promising direction for improving sentence embeddings. The source code is publicly available at: this https URL 

---
# PCAP-LM: An LLM-Native Text Representation for TLS Bulk Traffic Analysis 

**Authors**: Xavier Marjou, Lucas Tamic, Ilan Jaffeux-Cheniout  

**Link**: [PDF](https://arxiv.org/pdf/2607.28100)  

**Abstract**: Large language models (LLMs) offer powerful reasoning capabilities for network traffic analysis, but standard capture formats and their textual equivalents are prohibitively verbose, overflowing LLM context windows by two orders of magnitude. We present PCAP-LM, a flow-centric, LLM-native text representation that acts as a lossy knowledge extraction step rather than a standard compression tool: raw captures are transcoded into semantic summaries using PacketGlyphs - a novel ASCII alphabet coined in this paper that encodes packet direction, TCP/TLS state, log-scale size, and inter-packet delay. Combined with a constrained PMI-BPE tokenizer and motif run-length encoding, repetitive behavioural patterns are aggressively collapsed. A @REFS side-index preserves lossless drill-down into the original packets. Evaluated on a homogeneous corpus of 5G/4G TLS 1.3 bulk-download traffic, the BPE vocabulary fully saturates at 159 tokens, achieving an 812x size reduction over tshark -V and fitting entire captures within a single LLM context window. In a forensic question-answering evaluation over 30 held-out files, a frontier LLM achieves 99.3% accuracy from PCAP-LM documents versus 51.0% from a token-budget-matched tshark -V prefix. The lossy design introduces known blind spots - most notably a 24% false-negative rate for TCP retransmissions - and extending to heterogeneous mixed-protocol environments will require vocabulary retraining. 

---
# SciDataSailor: Deep Scientific Data Exploring 

**Authors**: Jiyong Rao, Yicheng Qiu, Chi Zhang, Chunfeng Song, Runkai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.28098)  

**Abstract**: Scientific datasets are commonly organized as hierarchical repositories containing heterogeneous and interdependent files, making their inspection, integration, and analysis labor-intensive and reliant on domain expertise. Although large language model (LLM) agents have advanced substantially in planning, reasoning, and tool use, existing research has largely overlooked their ability to interact with real scientific data assets through executable environments. We introduce Deep Scientific Data Exploration, an agentic task paradigm in which agents navigate repositories, interpret heterogeneous files and schemas, execute analyses, integrate cross-file evidence, and produce conclusions grounded in executed observations. To operationalize this paradigm, we present SciDataSailor, a framework for synthesizing tool-interactive trajectories by balancing broad exploration with targeted exploitation. SciDataSailor instantiates trajectory synthesis as Monte Carlo Tree Search (MCTS) with four task-specific mechanisms: difficulty-stratified exploration seeds, dual-feedback first-play urgency, hierarchical strategy-to-tool action generation, and entropy-guided branching. Using this framework, we construct SciDataSailor-SFT-2K for supervised fine-tuning and SciDataSailor-Bench for evaluation, with the latter comprising 627 meta-information summarization tasks and 586 scientific question-answering tasks across 27 datasets spanning the life, earth, and physical sciences. 

---
# SciSchema.org: A Multidisciplinary Collection of Schemas for Structured Scientific Process Descriptions 

**Authors**: Jennifer D'Souza, Sameer Sadruddin, Anisa Rula, Ana Bossler, Andrés Fullana, Enric Bas, Syed Ather, Defne Circi, Anlan Chen, L. Catherine Brinson, Alyssa Columbus, George Demetriou, Dongjun Jeong, Tarun Kumar, Frank Krüger, Sascha Genehr, Kai Budde-Sagert, Anamaria Leonescu, Francesco Lodola, Chiara Florindi, Gagana Balasubramanya Murthy, Samson Oluwapelumi Olagbile, Nazia Riasat, Yan Sha, Kevin Shen, Shaokai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27955)  

**Abstract**: Scientific processes are often described in heterogeneous article discourse, with details needed for comparison, reproducibility, reuse, and automation dispersed across prose, tables, figures, protocols, and supplementary files. We present the first release of this http URL, a multidisciplinary collection of 16 expert-annotated schemas spanning Biology & Biotechnology, Materials & Chemistry, Imaging & Measurement, Physics, and Psychology. Each schema defines reusable fields for describing process instances, including inputs, outputs, materials, instruments or software, parameters, conditions, procedural steps, measurements, and provenance-related information. The schemas were created through a human-in-the-loop schema-mining workflow in which large language models generated candidate structures from process specifications, scientific articles, and expert feedback, followed by domain-expert construction of final master schemas. The dataset contains final schemas in JSON Schema and SHACL formats, intermediate model-generated schemas, expert-feedback records, source-paper metadata, community-development materials, and analysis scripts. Technical validation assessed schema structure, development provenance, expert review, and syntactic conformance. The collection supports structured annotation, metadata enrichment, scientific knowledge graphs, information extraction, semantic publishing, and cross-study comparison. 

---
# TriShield: Zero-Utility-Loss Defense Against Privacy Backdoors in Federated Language Model Fine-Tuning via Orthogonal Gradient Projection and Optimizer State Entanglement 

**Authors**: Cheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.27940)  

**Abstract**: Federated fine-tuning of large language models (LLMs) enables collaborative training without exposing raw data. However, a recent attack, NeuroImprint [1] (arXiv:2606.20553), demonstrates that a malicious parameter server can corrupt a PEFT adapter into a privacy backdoor: by assigning a dedicated memorization neuron to each training sample and ensuring each neuron updates at most once, the server can analytically reconstruct 59\%--79\% of client training data with high semantic fidelity. Existing defenses---including local differential privacy (LDP) [8] and gradient clipping---either fail against this attack or impose unacceptable utility degradation. We present \textbf{TriShield}, a three-layer deterministic defense that completely prevents NeuroImprint-style reconstruction with \textbf{zero model utility loss} and \textbf{no additional communication rounds}. TriShield consists of: (1) a \textbf{Parameter Artifact Detector} that identifies memory-neuron signatures in distributed model parameters before local training begins; (2) a \textbf{Stateful Virtual Iteration} mechanism that forces Adam/AdamW's momentum state to irreversibly entangle gradients across virtual steps, invalidating NeuroImprint's closed-form inversion; and (3) a \textbf{Zero-Utility Orthogonal Projection} operator that projects all local gradient updates onto the main-task semantic subspace computed via SVD, physically eliminating any gradient components that carry private memorization. We prove theoretically that after Layers 2 and 3, the mutual information between the uploaded gradient and any individual training sample is zero. Experiments on GPT-2 (117M) and Llama-Guard-3-1B verify that TriShield reduces NeuroImprint reconstruction rate to \textbf{0\%} across all tested attack variants, while maintaining or improving training accuracy, with less than 5\% additional GPU computation overhead. 

---
# IFHierBench: Hierarchical Instruction Following for Large Language Models 

**Authors**: Yuetian Mao, Chunyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.27912)  

**Abstract**: Instruction-following ability is critical for deploying large language models in real-world applications, where downstream components depend on the output satisfying specific constraints. Modern deployments increasingly handle the full task in a single LLM call, with one prompt specifying a layered output whose overall artifact, structural sections, and nested fields must each satisfy concrete constraints. Existing instruction-following benchmarks treat the constraint set as a flat list applied uniformly to the response, so they cannot scope a check to a particular section of the output. We introduce IFHierBench, a hierarchical instruction-following benchmark of 600 prompts stratified across four constraint-tree depths and 35 distinct constraints, each prompt paired with a deterministic checker that verifies satisfaction at every scope. Evaluating seven leading proprietary and open-weight models, we find that even the strongest model only marginally exceeds 50% prompt-level accuracy and that accuracy degrades sharply as constraint depth grows. Reliably following nested constraints remains a substantial gap for current LLMs, motivating future training methods that consider constraint adherence at finer granularity to achieve better instruction-following ability. 

---
# MemTxn: A Transaction Boundary for Source-Supported Updates and Complete-State Recovery in Agent Memory 

**Authors**: Hanshuai Cui, Zhiqing Tang, Zhi Yao, Fanshuai Meng, Qianli Ma, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2607.27834)  

**Abstract**: Persistent memory lets long-running large language model agents reuse information across sessions and tasks. Yet errors in writable memory can persist and corrupt future behavior. Existing systems improve storage and retrieval, but they do not provide a transaction boundary for reliable updates and recovery. We therefore propose MemTxn, a governance layer outside the answer model. MemTxn verifies whether an update is supported by its source. It also selects the visible version when facts conflict and restores the application-visible state after a fault. The system uses Ordered PatchTest to validate writes, a Temporal Resolver to select versions, and a durable snapshot journal to recover state. On an item-disjoint audit, MemTxn accepts all 60 supported originals and rejects all 179 hard negatives. Under persistent multi-key faults on LongMemEval-S and LoCoMo states, it restores the complete declared active map without knowing the actual physical write set. On MemoryAgentBench FactConsolidation, MemTxn achieves the highest average F1 across all twelve answer-model configurations. It outperforms Dense by 17.06--24.07 points in five representative settings. 

---
# Cocktail-Talker: Multi-Speaker Dialog Modeling in Noisy Social Environments with Turn Action GRPO 

**Authors**: Xilin Jiang, Riki Shimizu, Sukru Samet Dindar, Junkai Wu, Zhongweiyang Xu, Nima Mesgarani  

**Link**: [PDF](https://arxiv.org/pdf/2607.27756)  

**Abstract**: Spoken dialog systems are typically designed for clean, dyadic interactions in which a single user and an assistant take turns speaking. Real-world social conversations, however, are often more ambiguous: multiple speakers may participate in the same conversation amid irrelevant speech and background noise. Each utterance may be directed to the assistant, addressed to another speaker, or completely irrelevant. In such settings, the assistant must decide not only what to say, but also whether to speak at all. In this paper, we introduce Cocktail-Talker, a speech LLM framework for multi-speaker spoken dialog modeling in noisy social environments. We model the assistant's behavior with three action tokens: <|respond|>, <|listen|>, and <|ignore|>, placed before a response or silence. Cocktail-Talker is trained via supervised finetuning and reinforcement learning to generate the appropriate action token and, only in <|respond|> mode, a speech response. To prepare the training data, we develop Cocktail-DialogGen, an LLM-based data pipeline that simulates realistic multi-speaker dialogs with speaker roles across diverse social settings. Together, these components take a step toward spoken dialog systems that interact more naturally and selectively in complex social environments. 

---
# Measuring Alignment With Reader Highlights Net of Position and Length 

**Authors**: Kazuki Nakayashiki, Keisuke Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2607.27739)  

**Abstract**: Context compression discards most of a document before a language model reads it, and is normally evaluated by downstream task accuracy - which makes another model the judge of what mattered. Naturalistic social highlighting offers a non-circular reference: many people independently marking passages on the same page. But the obvious metric, the fraction of crowd-marked sentences a compressor keeps, is confounded twice: crowd marks are front-loaded and crowd-marked sentences are longer, so any method favouring early or long sentences scores well regardless of readers. We remove both by matching each marked sentence against unmarked sentences of the same document at equal relative depth and equal within-document length rank, and we calibrate every estimator on synthetic nulls built from position and length alone - a step that matters, since depth-only stratification returns a false positive on 20-36% of nulls containing no effect. On 120 web documents (at least 12 independent readers each), a language-model importance ranking keeps 38.4% of crowd-marked sentences against 19.9% of their matched neighbours: an enrichment of +0.196 [+0.148, +0.239], at p = 0.0005 under an exact randomization test that assumes nothing about clustering, and replicated cross-vendor. Naive truncation, whose keep rule is position, correctly falls to +0.003. To give the number a scale: scored identically, on the same budget, against a crowd label recomputed to exclude them, a single human reader reaches +0.182 - indistinguishable from GPT-5.4 (+0.002 [-0.081, +0.088]) and below Claude Opus 5. Classical methods are not null - Luhn's 1958 heuristic reaches +0.088 - so reader selection is partly recoverable by counting words; conditioning additionally on lexical centrality removes only 0.010, so the agreement is not centrality. We also report that a claim in our own prior work does not reproduce on this corpus. 

---
# Baikal: Structured Search for Deep Research over Data Lakes 

**Authors**: Dhruv Agarwal, Rishitha Guttapalle Mohan, Aarti Kumari, Ashi Sinha, Athulya Anil, Kavitha Srinivas, Horst Samulowitz, Andrew McCallum  

**Link**: [PDF](https://arxiv.org/pdf/2607.27726)  

**Abstract**: Deep research over data lakes requires an LLM agent to investigate evidence across thousands of heterogeneous tables and passages to synthesize a report. Existing methods perform iterative retrieval and generation, letting accumulated context determine what to investigate next, which can overexploit locally promising evidence and fail to cover distinct semantic regions under a fixed budget. To address this, we cast deep research over data lakes as a budgeted search problem and present Baikal - a framework that clusters heterogeneous evidence into semantic regions, then searches over them adaptively to balance exploration and exploitation. Within each selected region, Baikal generates and investigates region-grounded subquestions, using finding quality as rewards to update region-level value estimates and guide search under policies ranging from random and LLM-guided selection to Bayesian $\epsilon$-greedy and UCB. We evaluate Baikal on 15 queries each over HybridQA and TAT-QA data lakes containing 10,993 and 2,757 tables, respectively, together with 227K Wikipedia passages and 13K financial report passages. We assess research quality with a new rubric covering groundedness, relevance, diversity, and utility, and use GPT-5-mini to score Baikal and strong baselines, including DeepSearcher and an OpenCode research agent with retrieval and clustering variants. Across both data lakes, Baikal performs strongly under several region-selection policies; its best configuration improves report scores over the strongest baselines by 28% on HybridQA and 36% on TAT-QA. Our analyses attribute these gains to organizing and exploring semantic evidence regions, which improves groundedness and diversity and yields more useful findings under the same subquestion budget. These results demonstrate the value of structured semantic exploration for systematic research and discovery over heterogeneous data lakes. 

---
# Tight Sample Complexity for Low-Rank Adaptation: Matching Bounds and Rank Selection 

**Authors**: Arunan J  

**Link**: [PDF](https://arxiv.org/pdf/2607.27680)  

**Abstract**: Low-Rank Adaptation (LoRA) has become the standard mechanism for fine-tuning large pretrained models, yet its statistical properties remain only partially understood. Existing generalization results provide upper bounds of the form O~(sqrt(rd/n)) or O~(rd/n), but a matching lower bound is missing, and the question of how to choose the LoRA rank r has no formal answer. Both gaps are closed here. A local Rademacher argument establishes an upper bound of O~(rd/n) on the excess risk of the empirical risk minimizer over rank-r LoRA, whenever the target adaptation has rank at most r. A matching minimax lower bound of Omega(rd/n) is then proved via a Fano-type packing of the rank-r subspace of R^{d x d}; the bound applies to any estimator whose output lies in the rank-r LoRA class. Combining the two yields a rank-selection dichotomy. For the constrained empirical risk minimizer, the optimal rank equals the intrinsic rank r*, and over-ranking strictly hurts. For adaptive estimators of the nuclear-norm-then-truncate type, over-ranking is harmless and the rate saturates at Theta~(r* d / n) regardless of r. Taken together, the three results characterize the statistical complexity of LoRA fine-tuning within the well-specified locally quadratic regime, and identify the empirically observed over-parameterization penalty as a property of unregularized empirical risk minimization rather than of the LoRA class itself. Predictions of the theory are verified on a synthetic trace-regression benchmark and on real LoRA fine-tuning across three (model, task) configurations covering DistilBERT and RoBERTa on SST-2 and MRPC. All configurations exhibit the predicted U-shape in validation loss, with two showing statistically significant loss inflation at large ranks (paired permutation p = 0.016). 

---
# Looped Transformers with Source-Centered State Evolution 

**Authors**: Bum Jun Kim, Kohei Hayashi, Shunsuke Kamiya, Masanori Koyama, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2607.27656)  

**Abstract**: Looped Transformers create a useful train- and test-time compute axis by reusing the same Transformer block over recurrent depth, increasing effective depth at a fixed parameter count. However, that shared block must then govern an entire trajectory of varying hidden states over trained and extrapolated depths. Furthermore, in additive-injection looped Transformers, an input-conditioned signal is reintroduced at every recurrent step, so applying the shared transition at an input-conditioned reference can still move the hidden state. In this paper, we propose Source-Centered State Evolution (SCSE), which is designed to reconcile input conditioning with reference-preserving shared recurrence. Specifically, SCSE retains input dependence through its learned anchor and initial deviation, allows nonzero deviations to drive recurrent computation while mapping zero deviation to zero, and guarantees exact anchor invariance through its zero-deviation mask. The designated anchor is thereby a one-step fixed point by construction. The zero-deviation forcing bias is the next deviation produced from the anchor itself and vanishes in SCSE, while nonzero deviations remain active and support state-dependent recurrent computation. Our theory shows that the zero-deviation forcing bias is a design degree of freedom whose task effect can be harmful, neutral, or beneficial; SCSE resolves this choice in favor of exact anchor invariance by setting the bias to zero. Across WikiText-2, WikiText-103, direct web-corpus pretraining, held-out web-text transfer, and LAMBADA completion, SCSE improves the controlled recurrent quality frontier. Ablation studies identify the learned anchor and the anchor-coordinate deviation recurrence as the primary contributors to the gain, and a trained-model case study grounds the anchor-response diagnostic in observed recurrent motion. 

---
# ReDiPPO: Reference-Guided Value Calibration and Discrepancy-Aware Token Reweighting for Mathematical Reasoning 

**Authors**: Zhenrong Zhang, Fei Wu, Jun Du, Jianshu Zhang, Si Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.27631)  

**Abstract**: Reinforcement learning has emerged as an effective paradigm for enhancing the mathematical reasoning capabilities of large language models. Among existing policy optimization methods, Proximal Policy Optimization (PPO) remains particularly appealing because its learned critic can, in principle, provide token-level credit assignment. However, in mathematical reasoning tasks characterized by long reasoning horizons and sparse outcome rewards, reliable token-level credit assignment remains challenging. The standard critic often fails to accurately evaluate intermediate reasoning states, resulting in noisy advantage estimates and suboptimal policy updates. In this paper, we propose ReDiPPO, a Reference-guided and Discrepancy-aware PPO framework for mathematical reasoning. ReDiPPO introduces a reference-guided critic that uses reference answers as training-time privileged signals to provide more accurate value estimation. Meanwhile, it retains a standard critic and quantifies the token-level reference-standard discrepancy between the standard value estimate and the reference-guided value estimate. This discrepancy serves as an indicator of difficult reasoning states and is used to reweight the corresponding token-level advantages during PPO optimization. Extensive experiments on diverse mathematical reasoning benchmarks demonstrate that ReDiPPO improves value-estimation accuracy and consistently outperforms strong policy optimization baselines, including PPO, DAPO, and GSPO, in final reasoning performance. Our code is available on this https URL. 

---
# Prox: Training-Free FFN Activation Sparsity via Approximate Intermediate-Channel Salience in LLMs 

**Authors**: Jinyi Liu, Wei Chen, Pengyu Chen, Xinyi Yuan, Minghe Bai, Guoquan Wu, Jun Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.27591)  

**Abstract**: Feed-forward networks (FFNs) dominate memory traffic and computation in large language model (LLM) inference, making them a primary target for activation sparsification. However, existing training-free methods suffer substantial model-quality degradation at high sparsity due to limitations in their channel-selection strategies. We observe that the SwiGLU intermediate state provides a highly effective channel-selection signal, but obtaining it requires costly dense computation. To address this, we present \emph{Prox}, a two-stage training-free framework for sparse SwiGLU FFNs. Prox hinges on the key insight: sparse execution requires only the channel mask induced by the intermediate state, which can be constructed from the magnitude ranking of its entries rather than their exact values. Specifically, Stage 1 uses input sparsity and quantized proxy weights to construct a shared mask; Stage 2 computes the selected channels exactly, enabling sparse execution of all three projections. Across ten LLMs from six model families, Prox outperforms training-free baselines at all sparsity levels, achieves up to a $1.99\times$ end-to-end decoding speedup at 70\% FFN sparsity, and is compatible with quantization and sparse attention. 

---
# Using Large Language Models for Idea Generation in Innovation 

**Authors**: Lennart Meincke, Karan Girotra, Gideon Nave, Christian Terwiesch, Karl T. Ulrich  

**Link**: [PDF](https://arxiv.org/pdf/2607.27553)  

**Abstract**: This research evaluates the efficacy of large language models (LLMs) in generating new product ideas. To do so, we compare three pools of ideas for new products targeted toward college students and priced at 50 dollars or less. The first pool of ideas was created by university students in a product design course before the availability of LLMs. The second and third pools of ideas were generated by GPT-4 from OpenAI using zero-shot and few-shot prompting, respectively. We evaluated idea quality using standard market research techniques to predict average purchase intent probability. We used text mining to assess idea similarity and human raters to evaluate idea novelty. We find that AI-generated ideas outperform human-generated ideas in terms of average purchase intent, with few-shot prompting yielding slightly higher intent than zero-shot prompting. However, AI-generated ideas are perceived as less novel and exhibit higher pairwise similarity, particularly with few-shot prompting, indicating a less diverse solution landscape. When focusing on the quality of the best ideas rather than the average ideas, we find that AI-generated ideas are seven times more likely to rank among the top 10 percent of ideas, demonstrating a significant advantage over human-generated ideas. We propose that this seven-to-one advantage is a conservative estimate because it does not account for the greater productivity of AI. Our findings suggest that despite some drawbacks, AI creativity presents a substantial benefit in generating high-quality ideas for new product development. 

---
# Subtract or Replay? Exact Deletion from Language-Model Memory 

**Authors**: Vishwajith Ramesh  

**Link**: [PDF](https://arxiv.org/pdf/2607.27539)  

**Abstract**: Exact deletion from persistent language-model memory depends on how that memory represents a record. Addressable influence can be removed by algebraic decrement; influence transformed by later writes inside shared recurrent state requires rebuilding from before the write. We test this distinction in two pretrained models against explicit record-omitted references. First, we replace Gemma 3's global-attention layers with support-vector memory. After low-rank recovery at 1B, decrement and retained-key refit agree at the next-token output to median KL $5.4\times10^{-15}$ over 31 support-token deletions, with $+2.0\%$ perplexity relative to a matched fine-tune. A masked-refit proxy is indistinguishable from the never-ingested floor under elicitation, relearning, sampling, and LiRA attacks. At 4B and 12B, certificate ordering persists but utility cost rises to $11.2\%$ and $44.3\%$. Second, in a 48B Kimi Linear hybrid, additive writes admit a fixed decrement and diagonal decay a corrected one, whereas the delta rule makes $12$--$49\%$ of a record's contribution suffix-dependent. Checkpointed rewind-and-replay deletes real clinical records at contexts up to 18,842 tokens, matching never-ingested logits and all recurrent states bit for bit within a deterministic MLX implementation; replaying a correction provides exact amendment. Exact deletion is therefore a property of memory representation: subtract addressable records and replay entangled writes. 

---
# ThreatForest: Multi-Agent Attack Tree Generation with Pluggable TTP Framework Mapping 

**Authors**: Cristian Leo, Anton Dykyi, Danny Cortegaca, Daniel Begimher, Prakash Jha  

**Link**: [PDF](https://arxiv.org/pdf/2607.27528)  

**Abstract**: Threat modeling is essential for secure software development, yet manual analysis of cloud-native architectures is slow and demands scarce security expertise. We present ThreatForest, a multi-agent system that generates structured attack trees from source code repositories, maps attack steps to adversary tactics, techniques, and procedures (TTPs) from a pluggable set of frameworks (MITRE ATT&CK, CAPEC, and cloud-specific threat matrices), and synthesizes actionable mitigations. ThreatForest decomposes threat modeling into a multi-stage agent pipeline -- repository analysis, context refinement, threat generation, parallel attack-tree construction with TTP mapping and mitigation synthesis, and report generation -- orchestrated as a directed graph with deterministic verification gates, bounded retries, and three human-in-the-loop validation points. A domain-specific sentence-transformer maps each attack step to candidate techniques by cosine similarity; we show empirically that this embedding stage, not the surrounding pipeline, is the dominant accuracy bottleneck. We evaluate ThreatForest across seven application domains on a sixteen-dimension rubric, scored by a panel of independent LLM raters with an adversarial verification pass and expert review. Panel-measured quality reaches 0.63-0.68 (on a 0-1 scale) for threat statements, attack trees, and mitigations, but only 0.29 for embedding-only TTP mapping -- a gap stable across all seven domains that isolates the binding constraint. A controlled single-call baseline on the same model more than doubles mapping defensibility, pinning the limitation on the embedding encoder rather than the multi-agent design. To our knowledge, ThreatForest is the first end-to-end system that turns a code repository into TTP-mapped attack trees with evidence-based mitigations across adversary frameworks, with a reusable framework for benchmarking such systems. 

---
# Latent States in Neural Networks: Recovering the Temporal Structure of Drifting Data from Model Weights 

**Authors**: Kevin Guan  

**Link**: [PDF](https://arxiv.org/pdf/2607.27482)  

**Abstract**: A temporally drifting data stream may pass through discrete regimes rather than changing continuously. We ask whether such regimes are recoverable from the weights of models trained on the stream, using a hidden Markov model (HMM) fit to the chronologically ordered trajectory of those weights. We study this question in two domains known to drift over time: multimodal misinformation detection, using the Fakeddit dataset; and sentiment analysis, using the Yelp dataset. We train classifiers on consecutive temporal windows and fit an HMM to the trajectory of their aligned weights, recovering latent states that partition each timeline into coherent phases. On both datasets, classifiers generalize better to data from windows sharing the state of their training window than to windows across state boundaries. This within-state transfer advantage survives a control for temporal proximity and modestly exceeds the advantage recovered by a naive partition into contiguous states of equal size. Although the states are estimated solely from model weights, they correlate more strongly with shifts in the data's class distribution than with the weight-space geometry used to estimate them. After class divergence and lag are residualized out, the within-state advantage exceeds its permutation null on both tasks, indicating that the states recover structure relevant to transfer beyond the data distribution. Every effect replicates on both tasks but is attenuated on Yelp, whose label distribution is more temporally stable. 

---
# Dimensionality and Measurement Precision in HLE's Multiple-Choice Subset 

**Authors**: Mayank Sharma, Savira Nadela, Tyler Matteson  

**Link**: [PDF](https://arxiv.org/pdf/2607.27420)  

**Abstract**: Humanity's Last Exam (HLE) is widely used to evaluate frontier language models. HLE organizes its questions into eight subject-domain categories, whose subscores are often interpreted as evidence of distinct capabilities. However, no study has assessed whether these labels correspond to empirically separable latent constructs, nor whether the benchmark effectively differentiates between models of similar ability. We evaluate 29 LLMs on the text-only multiple-choice subset of HLE ($J = 428$ items) and apply psychometric methods to assess both the dimensionality of the benchmark and the distribution of its measurement precision. Fitting a two-parameter logistic IRT model, we find convergent evidence that HLE measures a single general reasoning factor: McDonald's $\omega_h = 0.998$, domain labels explain only 3.5\% of item response variance, within- and between-domain residual correlations are nearly identical (Cohen's $d = 0.016$), and domain-specific ability estimates are near-redundant with the total score ($r \geq 0.81$). A separate analysis of the test information function reveals that measurement precision concentrates at moderate ability levels and drops sharply above $\theta = 0$, where frontier models sit. These findings suggest that HLE's domain subscores do not warrant distinct capability interpretations and that the benchmark's ability to discriminate among the strongest models is limited. 

---
# Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation 

**Authors**: Alexi Gladstone, Heng Ji, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2607.27372)  

**Abstract**: The deep learning revolution, kicked off by AlexNet, taught us that end-to-end training beats decomposing a problem into hand-designed stages. Generative modeling, however, has remained the exception-despite generative models being remarkably capable, they are still not trained end-to-end. This is because, at its core, generative modeling is about handling distributions with many modes, and existing scalable approaches handle this the same way, by factoring the generation procedure, which prevents end-to-end generation. In this work, we introduce Explorative Modeling, a new paradigm that instead factors the training loop, exploring K candidate matches between model generations and data, and training on the best, so predictions commit to modes rather than blurring them. We find Explorative Models (XMs) useful in two settings. First, increasing exploration adds a third pretraining axis beyond parameters and data for existing generative models-where scaling exploration monotonically improves performance across both continuous and discrete domains (images, video, and language). Notably, gains from exploration increase with scale, climbing from 7% to 36% as data scales and from 13% to 23% as models grow, with efficiency gains more than doubling at 3x the compute. Concretely, exploration improves FLOP efficiency by 4.1x, sample efficiency by 6.2x, parameter efficiency by 47%, lifts the strongest of image-generation recipes to a near-state-of-the-art 1.43 FID on ImageNet without guidance, enables scaling how end-to-end existing models are, and unlocks scaling generalization. Second, XMs enable end-to-end reconstructive generative modeling, matching diffusion on control tasks with 16-256x fewer inference steps. Together, these results establish XMs as both a new pretraining axis for existing generative models and a standalone end-to-end generative modeling paradigm. 

---
# THGFM: Dual-Branch Temporal Heterogeneous Graph Fusion Model 

**Authors**: Yixin Peng, Diego Collarana, Er Jin, Stefan Decker  

**Link**: [PDF](https://arxiv.org/pdf/2607.27303)  

**Abstract**: Temporal heterogeneous graphs offer a natural abstraction for dynamic relational systems in which diverse node and relation types co-exist and evolve over time. Learning on such graphs requires jointly modeling cross-type structural heterogeneity and the temporal dynamics of interactions, yet existing methods still struggle to reconcile parameter-efficient cross-type transfer with relation-aware specialization, and typically inject time only as additive features outside the attention kernel. We propose \textbf{THGFM}, a web-scale temporal heterogeneous graph fusion model that addresses both limitations within a unified dual-path architecture. THGFM couples a \textit{Shared-Space Temporal Attention} branch for parameter-efficient cross-type transfer with a \textit{Relational Type-Partitioned Temporal Attention} branch for relation-aware specialization, and integrates them through \textit{Dual-Path Relational--Shared Fusion}, instantiated with \textit{Type-Conditioned Non-Competitive Gated Sum Fusion}: a adaptive mechanism that assigns independent, type-conditioned feature-wise gates to the shared and specialized branches, allowing both to be amplified or suppressed without zero-sum competition. To directly incorporate relative time into the attention score, THGFM further introduces \textit{Rotary Temporal Attention}, which rotates queries and keys by half-phases of relative time before matching. THGFM consistently outperforms baseline graph transformer models on academic graphs benchmarks, delivering a $+3.25\%$ six-task mean gain, with peak relative gains of $+12.37\%$ on OAG-CS PV, $+4.87\%$ on PF-$L_2$, and $+1.18\%$ on PF-$L_1$, and $+4.24\%$, $+3.73\%$, and $+4.61\%$ on OGBN-MAG, HTAG-ArXiv, and HTAG-DBLP, respectively. 

---
# Digital Harf: A Clinically Integrated Multimodal AI System for Pervasive Arabic Speech and Language Therapy 

**Authors**: Asif Azad, Mohammad Sadat Hossain, MD Sadik Hossain Shanto, Sabri Boughorbel, Abdulrhman Aljouie, Bdour Alwuqaysi, Yahya Bokhari, Ayah Othman Sindi, Ehsan Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2607.27212)  

**Abstract**: Children with Autism Spectrum Disorder in Arabic-speaking countries face compounded barriers to effective speech and language therapy: a shortage of qualified specialists, limited service reach beyond urban centers, and a near-total absence of culturally grounded digital therapy materials. We present Digital Harf, a pervasive, multimodal AI platform that extends clinician-led speech and language therapy into the home. The system integrates three therapeutic modules - language therapy, speech intelligibility, and picture description - within a unified workflow that adapts to each child's performance over time. To address the critical shortage of Arabic therapy content, we introduce an Agentic Synthetic Data Engine (ASDE) that automatically generates culturally relevant images, prompts, and language tasks guided by explicit therapeutic and cultural criteria. Expert evaluation with 13 licensed Speech-Language Pathologists yielded a 90.1% clinical acceptance rate for ASDE-generated content without any manual curation or selection, and strong ratings for cultural and linguistic alignment across the full platform. Digital Harf demonstrates that AI-driven therapeutic systems can be built from the ground up for underrepresented linguistic settings, treating cultural grounding as core infrastructure rather than an adaptation afterthought. 

---
# MagicSelector: Joint Optimization for Agent Tool Selection via Counterfactual Decomposition and Progressive Reranking 

**Authors**: HONOR Agentic Search Team, Zhengzong Chen, Lei Tang, Lijun Liu, Chuandi Jiang, Fan Yang, Keyun Chu, Chu Zhao, Shihao Liu, Minghang Li, Bo Liang, Can Wen, Hailong Wu, Jingnan Ju, Mian Liu, Nengbin Zhang, Peiqiang Wang, Penghe Nie, Qinhui Gu, Sijia Lv, Siqi Chen, Wei Zhang, Yang Xu, Yuhao Qian, Yuxiang Zhang, Zeng Cheng, Zhen Wang, Zuan Chen, Yuanyuan Zhao, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.17751)  

**Abstract**: We present MagicSelector, a joint optimization framework integrating Counterfactual task decomposition, Progressive reranking, and Dynamic Top-K, designed to address the fundamental challenges of tool retrieval in agents. MagicSelector is a specialized framework capable of translating ambiguous user instructions into executable atomic subtasks and guiding high-precision tool retrieval, effectively mitigating redundant noise and severe context distraction in out-of-domain (OOD) scenarios. We empower MagicSelector with these capabilities through three key contributions: (1) a preference-guided counterfactual task decomposition mechanism that utilizes a counterfactual reward to quantify the marginal causal gain of decomposition on retrieval ranking, effectively imposing fine-grained structural supervision on logical coherence; (2) a progressive tool reranking method driven by self-distillation hard negative mining, which optimizes both point-wise and list-wise relevance to enhance fine-grained discrimination among highly similar tools; and (3) a dual semantic boundary-aware dynamic Top-K strategy that adaptively monitors reranking score cliffs and inter-tool semantic shifts to dynamically truncate the candidate list, maximizing relevant tool recall while filtering long-tail noise. Evaluated on MTDTool, the first task decomposition benchmark we constructed tailored for mobile multi-turn interactions with process-level annotations, MagicSelector yields promising performance. Extensive experiments demonstrate that MagicSelector significantly outperforms state-of-the-art methods in terms of tool retrieval accuracy, OOD generalization capability, and overall token efficiency, thereby demonstrating the effectiveness of our proposed framework. 

---
