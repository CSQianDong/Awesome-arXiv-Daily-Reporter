# Covertly improving intelligibility with data-driven adaptations of speech timing 

**Authors**: Paige Tuttösí, Angelica Lim, H. Henny Yeung, Yue Wang, Jean-Julien Aucouturier  

**Link**: [PDF](https://arxiv.org/pdf/2603.30032)  

**Abstract**: Human talkers often address listeners with language-comprehension challenges, such as hard-of-hearing or non-native adults, by globally slowing down their speech. However, it remains unclear whether this strategy actually makes speech more intelligible. Here, we take advantage of recent advancements in machine-generated speech allowing more precise control of speech rate in order to systematically examine how targeted speech-rate adjustments may improve comprehension. We first use reverse-correlation experiments to show that the temporal influence of speech rate prior to a target vowel contrast (ex. the tense-lax distinction) in fact manifests in a scissor-like pattern, with opposite effects in early versus late context windows; this pattern is remarkably stable both within individuals and across native L1-English listeners and L2-English listeners with French, Mandarin, and Japanese L1s. Second, we show that this speech rate structure not only facilitates L2 listeners' comprehension of the target vowel contrast, but that native listeners also rely on this pattern in challenging acoustic conditions. Finally, we build a data-driven text-to-speech algorithm that replicates this temporal structure on novel speech sequences. Across a variety of sentences and vowel contrasts, listeners remained unaware that such targeted slowing improved word comprehension. Strikingly, participants instead judged the common strategy of global slowing as clearer, even though it actually increased comprehension errors. Together, these results show that targeted adjustments to speech rate significantly aid intelligibility under challenging conditions, while often going unnoticed. More generally, this paper provides a data-driven methodology to improve the accessibility of machine-generated speech which can be extended to other aspects of speech comprehension and a wide variety of listeners and environments. 

---
# ContextClaim: A Context-Driven Paradigm for Verifiable Claim Detection 

**Authors**: Yufeng Li, Rrubaa Panchendrarajan, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2603.30025)  

**Abstract**: Verifiable claim detection asks whether a claim expresses a factual statement that can, in principle, be assessed against external evidence. As an early filtering stage in automated fact-checking, it plays an important role in reducing the burden on downstream verification components. However, existing approaches to claim detection, whether based on check-worthiness or verifiability, rely solely on the claim text itself. This is a notable limitation for verifiable claim detection in particular, where determining whether a claim is checkable may benefit from knowing what entities and events it refers to and whether relevant information exists to support verification. Inspired by the established role of evidence retrieval in later-stage claim verification, we propose Context-Driven Claim Detection (ContextClaim), a paradigm that advances retrieval to the detection stage. ContextClaim extracts entity mentions from the input claim, retrieves relevant information from Wikipedia as a structured knowledge source, and employs large language models to produce concise contextual summaries for downstream classification. We evaluate ContextClaim on two datasets covering different topics and text genres, the CheckThat! 2022 COVID-19 Twitter dataset and the PoliClaim political debate dataset, across encoder-only and decoder-only models under fine-tuning, zero-shot, and few-shot settings. Results show that context augmentation can improve verifiable claim detection, although its effectiveness varies across domains, model architectures, and learning settings. Through component analysis, human evaluation, and error analysis, we further examine when and why the retrieved context contributes to more reliable verifiability judgments. 

---
# Enhancing Structural Mapping with LLM-derived Abstractions for Analogical Reasoning in Narratives 

**Authors**: Mohammadhossein Khojasteh, Yifan Jiang, Stefano De Giorgis, Frank van Harmelen, Filip Ilievski  

**Link**: [PDF](https://arxiv.org/pdf/2603.29997)  

**Abstract**: Analogical reasoning is a key driver of human generalization in problem-solving and argumentation. Yet, analogies between narrative structures remain challenging for machines. Cognitive engines for structural mapping are not directly applicable, as they assume pre-extracted entities, whereas LLMs' performance is sensitive to prompt format and the degree of surface similarity between narratives. This gap motivates a key question: What is the impact of enhancing structural mapping with LLM-derived abstractions on their analogical reasoning ability in narratives? To that end, we propose a modular framework named YARN (Yielding Abstractions for Reasoning in Narratives), which uses LLMs to decompose narratives into units, abstract these units, and then passes them to a mapping component that aligns elements across stories to perform analogical reasoning. We define and operationalize four levels of abstraction that capture both the general meaning of units and their roles in the story, grounded in prior work on framing. Our experiments reveal that abstractions consistently improve model performance, resulting in competitive or better performance than end-to-end LLM baselines. Closer error analysis reveals the remaining challenges in abstraction at the right level, in incorporating implicit causality, and an emerging categorization of analogical patterns in narratives. YARN enables systematic variation of experimental settings to analyze component contributions, and to support future work, we make the code for YARN openly available. 

---
# Structural Feature Engineering for Generative Engine Optimization: How Content Structure Shapes Citation Behavior 

**Authors**: Junwei Yu, Mufeng Yang, Yepeng Ding, Hiroyuki Sato  

**Link**: [PDF](https://arxiv.org/pdf/2603.29979)  

**Abstract**: The proliferation of AI-powered search engines has shifted information discovery from traditional link-based retrieval to direct answer generation with selective source citation, creating new challenges for content visibility. While existing Generative Engine Optimization (GEO) approaches focus primarily on semantic content modification, the role of structural features in influencing citation behavior remains underexplored.
In this paper, we propose GEO-SFE, a systematic framework for structural feature engineering in generative engine optimization. Our approach decomposes content structure into three hierarchical levels: macro-structure (document architecture), meso-structure (information chunking), and micro-structure (visual emphasis), and models their impact on citation probability across different generative engine architectures. We develop architecture-aware optimization strategies and predictive models that preserve semantic integrity while improving structural effectiveness.
Experimental evaluation across six mainstream generative engines demonstrates consistent improvements in citation rate (17.3 percent) and subjective quality (18.5 percent), validating the effectiveness and generalizability of the proposed framework. This work establishes structural optimization as a foundational component of GEO, providing a data-driven methodology for enhancing content visibility in LLM-powered information ecosystems. 

---
# Rewrite the News: Tracing Editorial Reuse Across News Agencies 

**Authors**: Soveatin Kuntur, Nina Smirnova, Anna Wroblewska, Philipp Mayr, Sebastijan Razboršek Maček  

**Link**: [PDF](https://arxiv.org/pdf/2603.29937)  

**Abstract**: This paper investigates sentence-level text reuse in multilingual journalism, analyzing where reused content occurs within articles. We present a weakly supervised method for detecting sentence-level cross-lingual reuse without requiring full translations, designed to support automated pre-selection to reduce information overload for journalists (Holyst et al., 2024). The study compares English-language articles from the Slovenian Press Agency (STA) with reports from 15 foreign agencies (FA) in seven languages, using publication timestamps to retain the earliest likely foreign source for each reused sentence. We analyze 1,037 STA and 237,551 FA articles from two time windows (October 7-November 2, 2023; February 1-28, 2025) and identify 1,087 aligned sentence pairs after filtering to the earliest sources. Reuse occurs in 52% of STA articles and 1.6% of FA articles and is predominantly non-literal, involving paraphrase and compositional reuse from multiple sources. Reused content tends to appear in the middle and end of English articles, while leads are more often original, indicating that simple lexical matching overlooks substantial editorial reuse. Compared with prior work focused on monolingual overlap, we (i) detect reuse across languages without requiring full translation, (ii) use publication timing to identify likely sources, and (iii) analyze where reused material is situated within articles. Dataset and code: this https URL. 

---
# FLEURS-Kobani: Extending the FLEURS Dataset for Northern Kurdish 

**Authors**: Daban Q. Jaff, Mohammad Mohammadamini  

**Link**: [PDF](https://arxiv.org/pdf/2603.29892)  

**Abstract**: FLEURS offers n-way parallel speech for 100+ languages, but Northern Kurdish is not one of them, which limits benchmarking for automatic speech recognition and speech translation tasks in this language. We present FLEURS-Kobani, a Northern Kurdish (ISO 639-3 KMR) spoken extension of the FLEURS benchmark. The FLEURS-Kobani dataset consists of 5,162 validated utterances, totaling 18 hours and 24 minutes. The data were recorded by 31 native speakers. It extends benchmark coverage to an under-resourced Kurdish variety. As baselines, we fine-tuned Whisper v3-large for ASR and E2E S2TT. A two-stage fine-tuning strategy (Common Voice to FLEURS-Kobani) yields the best ASR performance (WER 28.11, CER 9.84 on test). For E2E S2TT (KMR to EN), Whisper achieves 8.68 BLEU on test; we additionally report pivot-derived targets and a cascaded S2TT setup. FLEURS-Kobani provides the first public Northern Kurdish benchmark for evaluation of ASR, S2TT and S2ST tasks. The dataset is publicly released for research use under a CC BY 4.0 license. 

---
# Towards Empowering Consumers through Sentence-level Readability Scoring in German ESG Reports 

**Authors**: Benjamin Josef Schüßler, Jakob Prange  

**Link**: [PDF](https://arxiv.org/pdf/2603.29861)  

**Abstract**: With the ever-growing urgency of sustainability in the economy and society, and the massive stream of information that comes with it, consumers need reliable access to that information. To address this need, companies began publishing so called Environmental, Social, and Governance (ESG) reports, both voluntarily and forced by law. To serve the public, these reports must be addressed not only to financial experts but also to non-expert audiences. But are they written clearly enough? In this work, we extend an existing sentence-level dataset of German ESG reports with crowdsourced readability annotations. We find that, in general, native speakers perceive sentences in ESG reports as easy to read, but also that readability is subjective. We apply various readability scoring methods and evaluate them regarding their prediction error and correlation with human rankings. Our analysis shows that, while LLM prompting has potential for distinguishing clear from hard-to-read sentences, a small finetuned transformer predicts human readability with the lowest error. Averaging predictions of multiple models can slightly improve the performance at the cost of slower inference. 

---
# SNEAK: Evaluating Strategic Communication and Information Leakage in Large Language Models 

**Authors**: Adar Avsian, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2603.29846)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multi-agent settings where communication must balance informativeness and secrecy. In such settings, an agent may need to signal information to collaborators while preventing an adversary from inferring sensitive details. However, existing LLM benchmarks primarily evaluate capabilities such as reasoning, factual knowledge, or instruction following, and do not directly measure strategic communication under asymmetric information. We introduce SNEAK (Secret-aware Natural language Evaluation for Adversarial Knowledge), a benchmark for evaluating selective information sharing in language models. In SNEAK, a model is given a semantic category, a candidate set of words, and a secret word, and must generate a message that indicates knowledge of the secret without revealing it too clearly. We evaluate generated messages using two simulated agents with different information states: an ally, who knows the secret and must identify the intended message, and a chameleon, who does not know the secret and attempts to infer it from the message. This yields two complementary metrics: utility, measuring how well the message communicates to collaborators, and leakage, measuring how much information it reveals to an adversary. Using this framework, we analyze the trade-off between informativeness and secrecy in modern language models and show that strategic communication under asymmetric information remains a challenging capability for current systems. Notably, human participants outperform all evaluated models by a large margin, achieving up to four times higher scores. 

---
# ENEIDE: A High Quality Silver Standard Dataset for Named Entity Recognition and Linking in Historical Italian 

**Authors**: Cristian Santini, Sebastian Barzaghi, Paolo Sernani, Emanuele Frontoni, Laura Melosi, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2603.29801)  

**Abstract**: This paper introduces ENEIDE (Extracting Named Entities from Italian Digital Editions), a silver standard dataset for Named Entity Recognition and Linking (NERL) in historical Italian texts. The corpus comprises 2,111 documents with over 8,000 entity annotations semi-automatically extracted from two scholarly digital editions: Digital Zibaldone, the philosophical diary of the Italian poet Giacomo Leopardi (1798--1837), and Aldo Moro Digitale, the complete works of the Italian politician Aldo Moro (1916--1978). Annotations cover multiple entity types (person, location, organization, literary work) linked to Wikidata identifiers, including NIL entities that cannot be mapped to the knowledge graph. To the best of our knowledge, ENEIDE represents the first multi-domain, publicly available NERL dataset for historical Italian with training, development, and test splits. We present a methodology for semi-automatic annotations extraction from manually curated scholarly digital editions, including quality control and annotation enhancement procedures. Baseline experiments using state-of-the-art models demonstrate the dataset's challenge for NERL and the gap between zero-shot approaches and fine-tuned models. The dataset's diachronic coverage spanning two centuries makes it particularly suitable for temporal entity disambiguation and cross-domain evaluation. ENEIDE is released under a CC BY-NC-SA 4.0 license. 

---
# Near-Miss: Latent Policy Failure Detection in Agentic Workflows 

**Authors**: Ella Rabinovich, David Boaz, Naama Zwerdling, Ateret Anaby-Tavor  

**Link**: [PDF](https://arxiv.org/pdf/2603.29665)  

**Abstract**: Agentic systems for business process automation often require compliance with policies governing conditional updates to the system state. Evaluation of policy adherence in LLM-based agentic workflows is typically performed by comparing the final system state against a predefined ground truth. While this approach detects explicit policy violations, it may overlook a more subtle class of issues in which agents bypass required policy checks, yet reach a correct outcome due to favorable circumstances. We refer to such cases as $\textit{near-misses}$ or $\textit{latent failures}$. In this work, we introduce a novel metric for detecting latent policy failures in agent conversations traces. Building on the ToolGuard framework, which converts natural-language policies into executable guard code, our method analyzes agent trajectories to determine whether agent's tool-calling decisions where sufficiently informed.
We evaluate our approach on the $\tau^2$-verified Airlines benchmark across several contemporary open and proprietary LLMs acting as agents. Our results show that latent failures occur in 8-17% of trajectories involving mutating tool calls, even when the final outcome matches the expected ground-truth state. These findings reveal a blind spot in current evaluation methodologies and highlight the need for metrics that assess not only final outcomes but also the decision process leading to them. 

---
# Agenda-based Narrative Extraction: Steering Pathfinding Algorithms with Large Language Models 

**Authors**: Brian Felipe Keith-Norambuena, Carolina Inés Rojas-Córdova, Claudio Juvenal Meneses-Villegas, Elizabeth Johanna Lam-Esquenazi, Angélica María Flores-Bustos, Ignacio Alejandro Molina-Villablanca, Joshua Emanuel Leyton-Vallejos  

**Link**: [PDF](https://arxiv.org/pdf/2603.29661)  

**Abstract**: Existing narrative extraction methods face a trade-off between coherence, interactivity, and multi-storyline support. Narrative Maps supports rich interaction and generates multiple storylines as a byproduct of its coverage constraints, though this comes at the cost of individual path coherence. Narrative Trails achieves high coherence through maximum capacity path optimization but provides no mechanism for user guidance or multiple perspectives. We introduce agenda-based narrative extraction, a method that bridges this gap by integrating large language models into the Narrative Trails pathfinding process to steer storyline construction toward user-specified perspectives. Our approach uses an LLM at each step to rank candidate documents based on their alignment with a given agenda while maintaining narrative coherence. Running the algorithm with different agendas yields different storylines through the same corpus. We evaluated our approach on a news article corpus using LLM judges with Claude Opus 4.5 and GPT 5.1, measuring both coherence and agenda alignment across 64 endpoint pairs and 6 agendas. LLM-driven steering achieves 9.9% higher alignment than keyword matching on semantic agendas (p=0.017), with 13.3% improvement on \textit{Regime Crackdown} specifically (p=0.037), while keyword matching remains competitive on agendas with literal keyword overlap. The coherence cost is minimal: LLM steering reduces coherence by only 2.2% compared to the agenda-agnostic baseline. Counter-agendas that contradict the source material score uniformly low (2.2-2.5) across all methods, confirming that steering cannot fabricate unsupported narratives. 

---
# Learning Diagnostic Reasoning for Decision Support in Toxicology 

**Authors**: Nico Oberländer, David Bani-Harouni, Tobias Zellner, Nassir Navab, Florian Eyer, Matthias Keicher  

**Link**: [PDF](https://arxiv.org/pdf/2603.29608)  

**Abstract**: Acute poly-substance intoxication requires rapid, life-saving decisions under substantial uncertainty, as clinicians must rely on incomplete ingestion details and nonspecific symptoms. Effective diagnostic reasoning in this chaotic environment requires fusing unstructured, non-medical narratives (e.g. paramedic scene descriptions and unreliable patient self-reports or known histories), with structured medical data like vital signs. While Large Language Models (LLMs) show potential for processing such heterogeneous inputs, they struggle in this setting, often underperforming simple baselines that rely solely on patient histories. To address this, we present DeToxR (Decision-support for Toxicology with Reasoning), the first adaptation of Reinforcement Learning (RL) to emergency toxicology. We design a robust data-fusion engine for multi-label prediction across 14 substance classes based on an LLM finetuned with Group Relative Policy Optimization (GRPO). We optimize the model's reasoning directly using a clinical performance reward. By formulating a multi-label agreement metric as the reward signal, the model is explicitly penalized for missing co-ingested substances and hallucinating absent poisons. Our model significantly outperforms its unadapted base LLM counterpart and supervised baselines. Furthermore, in a clinical validation study, the model indicates a clinical advantage by outperforming an expert toxicologist in identifying the correct poisons (Micro-F1: 0.644 vs. 0.473). These results demonstrate the potential of RL-aligned LLMs to synthesize unstructured pre-clinical narratives and structured medical data for decision support in high-stakes environments. 

---
# When Can We Trust LLM Graders? Calibrating Confidence for Automated Assessment 

**Authors**: Robinson Ferrer, Damla Turgut, Zhongzhou Chen, Shashank Sonkar  

**Link**: [PDF](https://arxiv.org/pdf/2603.29559)  

**Abstract**: Large Language Models (LLMs) show promise for automated grading, but their outputs can be unreliable. Rather than improving grading accuracy directly, we address a complementary problem: \textit{predicting when an LLM grader is likely to be correct}. This enables selective automation where high-confidence predictions are processed automatically while uncertain cases are flagged for human review. We compare three confidence estimation methods (self-reported confidence, self-consistency voting, and token probability) across seven LLMs of varying scale (4B to 120B parameters) on three educational datasets: RiceChem (long-answer chemistry), SciEntsBank, and Beetle (short-answer science). Our experiments reveal that self-reported confidence consistently achieves the best calibration across all conditions (avg ECE 0.166 vs 0.229 for self-consistency). Surprisingly, self-consistency remains 38\% worse despite requiring 5$\times$ the inference cost. Larger models exhibit substantially better calibration though gains vary by dataset and method (e.g., a 28\% ECE reduction for self-reported), with GPT-OSS-120B achieving the best calibration (avg ECE 0.100) and strong discrimination (avg AUC 0.668). We also observe that confidence is strongly top-skewed across methods, creating a ``confidence floor'' that practitioners must account for when setting thresholds. These findings suggest that simply asking LLMs to report their confidence provides a practical approach for identifying reliable grading predictions. Code is available \href{this https URL}{here}. 

---
# Bringing Up a Bilingual BabyLM: Investigating Multilingual Language Acquisition Using Small-Scale Models 

**Authors**: Linda Zeng, Steven Y. Feng, Michael C. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2603.29552)  

**Abstract**: Multilingualism is incredibly common around the world, leading to many important theoretical and practical questions about how children learn multiple languages at once. For example, does multilingual acquisition lead to delays in learning? Are there better and worse ways to structure multilingual input? Many correlational studies address these questions, but it is surprisingly difficult to get definitive answers because children cannot be randomly assigned to be multilingual and data are typically not matched between languages. We use language model training as a method for simulating a variety of highly controlled exposure conditions, and create matched 100M-word mono- and bilingual datasets using synthetic data and machine translation. We train GPT-2 models on monolingual and bilingual data organized to reflect a range of exposure regimes, and evaluate their performance on perplexity, grammaticality, and semantic knowledge. Across model scales and measures, bilingual models perform similarly to monolingual models in one language, but show strong performance in the second language as well. These results suggest that there are no strong differences between different bilingual exposure regimes, and that bilingual input poses no in-principle challenges for agnostic statistical learners. 

---
# Can LLM Agents Identify Spoken Dialects like a Linguist? 

**Authors**: Tobias Bystrich, Lukas Hamm, Maria Hassan, Lea Fischbach, Lucie Flek, Akbar Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2603.29541)  

**Abstract**: Due to the scarcity of labeled dialectal speech, audio dialect classification is a challenging task for most languages, including Swiss German. In this work, we explore the ability of large language models (LLMs) as agents in understanding the dialects and whether they can show comparable performance to models such as HuBERT in dialect classification. In addition, we provide an LLM baseline and a human linguist one. Our approach uses phonetic transcriptions produced by ASR systems and combines them with linguistic resources such as dialect feature maps, vowel history, and rules. Our findings indicate that, when linguistic information is provided, the LLM predictions improve. The human baseline shows that automatically generated transcriptions can be beneficial for such classifications, but also presents opportunities for improvement. 

---
# Baby Scale: Investigating Models Trained on Individual Children's Language Input 

**Authors**: Steven Y. Feng, Alvin W.M. Tan, Michael C. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2603.29522)  

**Abstract**: Modern language models (LMs) must be trained on many orders of magnitude more words of training data than human children receive before they begin to produce useful behavior. Assessing the nature and origins of this "data gap" requires benchmarking LMs on human-scale datasets to understand how linguistic knowledge emerges from children's natural training data. Using transcripts from the BabyView dataset (videos from children ages 6-36 months), we investigate (1) scaling performance at child-scale data regimes, (2) variability in model performance across datasets from different children's experiences and linguistic predictors of dataset quality, and (3) relationships between model and child language learning outcomes. LMs trained on child data show acceptable scaling for grammar tasks, but lower scaling on semantic and world knowledge tasks than models trained on synthetic data; we also observe substantial variability on data from different children. Beyond dataset size, performance is most associated with a combination of distributional and interactional linguistic features, broadly consistent with what makes high-quality input for child language development. Finally, model likelihoods for individual words correlate with children's learning of those words, suggesting that properties of child-directed input may influence both model learning and human language development. Overall, understanding what properties make language data efficient for learning can enable more powerful small-scale language models while also shedding light on human language acquisition. 

---
# Impact of enriched meaning representations for language generation in dialogue tasks: A comprehensive exploration of the relevance of tasks, corpora and metrics 

**Authors**: Alain Vázquez, Maria Inés Torres  

**Link**: [PDF](https://arxiv.org/pdf/2603.29518)  

**Abstract**: Conversational systems should generate diverse language forms to interact fluently and accurately with users. In this context, Natural Language Generation (NLG) engines convert Meaning Representations (MRs) into sentences, directly influencing user perception. These MRs usually encode the communicative function (e.g., inform, request, confirm) via DAs and enumerate the semantic content with slot-value pairs. In this work, our objective is to analyse whether providing a task demonstrator to the generator enhances the generations of a fine-tuned model. This demonstrator is an MR-sentence pair extracted from the original dataset that enriches the input at training and inference time. The analysis involves five metrics that focus on different linguistic aspects, and four datasets that differ in multiple features, such as domain, size, lexicon, MR variability, and acquisition process. To the best of our knowledge, this is the first study on dialogue NLG implementing a comparative analysis of the impact of MRs on generation quality across domains, corpus characteristics, and the metrics used to evaluate these generations. Our key insight is that the proposed enriched inputs are effective for complex tasks and small datasets with high variability in MRs and sentences. They are also beneficial in zero-shot settings for any domain. Moreover, the analysis of the metrics shows that semantic metrics capture generation quality more accurately than lexical metrics. In addition, among these semantic metrics, those trained with human ratings can detect omissions and other subtle semantic issues that embedding-based metrics often miss. Finally, the evolution of the metric scores and the excellent results for Slot Accuracy and Dialogue Act Accuracy demonstrate that the generative models present fast adaptability to different tasks and robustness at semantic and communicative intention levels. 

---
# LLM Probe: Evaluating LLMs for Low-Resource Languages 

**Authors**: Hailay Kidu Teklehaymanot, Gebrearegawi Gebremariam, Wolfgang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2603.29517)  

**Abstract**: Despite rapid advances in large language models (LLMs), their linguistic abilities in low-resource and morphologically rich languages are still not well understood due to limited annotated resources and the absence of standardized evaluation frameworks. This paper presents LLM Probe, a lexicon-based assessment framework designed to systematically evaluate the linguistic skills of LLMs in low-resource language environments. The framework analyzes models across four areas of language understanding: lexical alignment, part-of-speech recognition, morphosyntactic probing, and translation accuracy. To illustrate the framework, we create a manually annotated benchmark dataset using a low-resource Semitic language as a case study. The dataset comprises bilingual lexicons with linguistic annotations, including part-of-speech tags, grammatical gender, and morphosyntactic features, which demonstrate high inter-annotator agreement to ensure reliable annotations. We test a variety of models, including causal language models and sequence-to-sequence architectures. The results reveal notable differences in performance across various linguistic tasks: sequence-to-sequence models generally excel in morphosyntactic analysis and translation quality, whereas causal models demonstrate strong performance in lexical alignment but exhibit weaker translation accuracy. Our results emphasize the need for linguistically grounded evaluation to better understand LLM limitations in low-resource settings. We release LLM Probe and the accompanying benchmark dataset as open-source tools to promote reproducible benchmarking and to support the development of more inclusive multilingual language technologies. 

---
# Distilling Human-Aligned Privacy Sensitivity Assessment from Large Language Models 

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2603.29497)  

**Abstract**: Accurate privacy evaluation of textual data remains a critical challenge in privacy-preserving natural language processing. Recent work has shown that large language models (LLMs) can serve as reliable privacy evaluators, achieving strong agreement with human judgments; however, their computational cost and impracticality for processing sensitive data at scale limit real-world deployment. We address this gap by distilling the privacy assessment capabilities of Mistral Large 3 (675B) into lightweight encoder models with as few as 150M parameters. Leveraging a large-scale dataset of privacy-annotated texts spanning 10 diverse domains, we train efficient classifiers that preserve strong agreement with human annotations while dramatically reducing computational requirements. We validate our approach on human-annotated test data and demonstrate its practical utility as an evaluation metric for de-identification systems. 

---
# MemFactory: Unified Inference & Training Framework for Agent Memory 

**Authors**: Ziliang Guo, Ziheng Li, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.29493)  

**Abstract**: Memory-augmented Large Language Models (LLMs) are essential for developing capable, long-term AI agents. Recently, applying Reinforcement Learning (RL) to optimize memory operations, such as extraction, updating, and retrieval, has emerged as a highly promising research direction. However, existing implementations remain highly fragmented and task-specific, lacking a unified infrastructure to streamline the integration, training, and evaluation of these complex pipelines. To address this gap, we present MemFactory, the first unified, highly modular training and inference framework specifically designed for memory-augmented agents. Inspired by the success of unified fine-tuning frameworks like LLaMA-Factory, MemFactory abstracts the memory lifecycle into atomic, plug-and-play components, enabling researchers to seamlessly construct custom memory agents via a "Lego-like" architecture. Furthermore, the framework natively integrates Group Relative Policy Optimization (GRPO) to fine-tune internal memory management policies driven by multi-dimensional environmental rewards. MemFactory provides out-of-the-box support for recent cutting-edge paradigms, including Memory-R1, RMM, and MemAgent. We empirically validate MemFactory on the open-source MemAgent architecture using its publicly available training and evaluation data. Across both in-domain and out-of-distribution evaluation sets, MemFactory consistently improves performance over the corresponding base models, with relative gains of up to 14.8%. By providing a standardized, extensible, and easy-to-use infrastructure, MemFactory significantly lowers the barrier to entry, paving the way for future innovations in memory-driven AI agents. 

---
# Calibrated Confidence Expression for Radiology Report Generation 

**Authors**: David Bani-Harouni, Chantal Pellegrini, Julian Lüers, Su Hwan Kim, Markus Baalmann, Benedikt Wiestler, Rickmer Braren, Nassir Navab, Matthias Keicher  

**Link**: [PDF](https://arxiv.org/pdf/2603.29492)  

**Abstract**: Safe deployment of Large Vision-Language Models (LVLMs) in radiology report generation requires not only accurate predictions but also clinically interpretable indicators of when outputs should be thoroughly reviewed, enabling selective radiologist verification and reducing the risk of hallucinated findings influencing clinical decisions. One intuitive approach to this is verbalized confidence, where the model explicitly states its certainty. However, current state-of-the-art language models are often overconfident, and research on calibration in multimodal settings such as radiology report generation is limited. To address this gap, we introduce ConRad (Confidence Calibration for Radiology Reports), a reinforcement learning framework for fine-tuning medical LVLMs to produce calibrated verbalized confidence estimates alongside radiology reports. We study two settings: a single report-level confidence score and a sentence-level variant assigning a confidence to each claim. Both are trained using the GRPO algorithm with reward functions based on the logarithmic scoring rule, which incentivizes truthful self-assessment by penalizing miscalibration and guarantees optimal calibration under reward maximization. Experimentally, ConRad substantially improves calibration and outperforms competing methods. In a clinical evaluation we show that ConRad's report level scores are well aligned with clinicians' judgment. By highlighting full reports or low-confidence statements for targeted review, ConRad can support safer clinical integration of AI-assistance for report generation. 

---
# M-MiniGPT4: Multilingual VLLM Alignment via Translated Data 

**Authors**: Seung Hun Han, Youssef Mohamed, Mohamed Elhoseiny  

**Link**: [PDF](https://arxiv.org/pdf/2603.29467)  

**Abstract**: This paper presents a Multilingual Vision Large Language Model, named M-MiniGPT4. Our model exhibits strong vision-language understanding (VLU) capabilities across 11 languages. We utilize a mixture of native multilingual and translated data to push the multilingual VLU performance of the MiniGPT4 architecture. In addition, we propose a multilingual alignment training stage that uses parallel text corpora to further enhance the multilingual capabilities of our model. M-MiniGPT4 achieves 36% accuracy on the multilingual MMMU benchmark, outperforming state-of-the-art models in the same weight class, including foundation models released after the majority of this work was completed. We open-source our models, code, and translated datasets to facilitate future research in low-resource and multilingual settings. 

---
# Authorship Impersonation via LLM Prompting does not Evade Authorship Verification Methods 

**Authors**: Baoyi Zeng, Andrea Nini  

**Link**: [PDF](https://arxiv.org/pdf/2603.29454)  

**Abstract**: Authorship verification (AV), the task of determining whether a questioned text was written by a specific individual, is a critical part of forensic linguistics. While manual authorial impersonation by perpetrators has long been a recognized threat in historical forensic cases, recent advances in large language models (LLMs) raise new challenges, as adversaries may exploit these tools to impersonate another's writing. This study investigates whether prompted LLMs can generate convincing authorial impersonations and whether such outputs can evade existing forensic AV systems. Using GPT-4o as the adversary model, we generated impersonation texts under four prompting conditions across three genres: emails, text messages, and social media posts. We then evaluated these outputs against both non-neural AV methods (n-gram tracing, Ranking-Based Impostors Method, LambdaG) and neural approaches (AdHominem, LUAR, STAR) within a likelihood-ratio framework. Results show that LLM-generated texts failed to sufficiently replicate authorial individuality to bypass established AV systems. We also observed that some methods achieved even higher accuracy when rejecting impersonation texts compared to genuine negative samples. Overall, these findings indicate that, despite the accessibility of LLMs, current AV systems remain robust against entry-level impersonation attempts across multiple genres. Furthermore, we demonstrate that this counter-intuitive resilience stems, at least in part, from the higher lexical diversity and entropy inherent in LLM-generated texts. 

---
# CounselReflect: A Toolkit for Auditing Mental-Health Dialogues 

**Authors**: Yahan Li, Chaohao Du, Zeyang Li, Christopher Chun Kuizon, Shupeng Cheng, Angel Hsing-Chi Hwang, Adam C. Frank, Ruishan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29429)  

**Abstract**: Mental-health support is increasingly mediated by conversational systems (e.g., LLM-based tools), but users often lack structured ways to audit the quality and potential risks of the support they receive. We introduce CounselReflect, an end-to-end toolkit for auditing mental-health support dialogues. Rather than producing a single opaque quality score, CounselReflect provides structured, multi-dimensional reports with session-level summaries, turn-level scores, and evidence-linked excerpts to support transparent inspection. The system integrates two families of evaluation signals: (i) 12 model-based metrics produced by task-specific predictors, and (ii) rubric-based metrics that extend coverage via a literature-derived library (69 metrics) and user-defined custom metrics, operationalized with configurable LLM judges. CounselReflect is available as a web application, browser extension, and command-line interface (CLI), enabling use in real-time settings as well as at scale. Human evaluation includes a user study with 20 participants and an expert review with 6 mental-health professionals, suggesting that CounselReflect supports understandable, usable, and trustworthy auditing. A demo video and full source code are also provided. 

---
# Is my model perplexed for the right reason? Contrasting LLMs' Benchmark Behavior with Token-Level Perplexity 

**Authors**: Zoë Prins, Samuele Punzo, Frank Wildenburg, Giovanni Cinà, Sandro Pezzelle  

**Link**: [PDF](https://arxiv.org/pdf/2603.29396)  

**Abstract**: Standard evaluations of Large language models (LLMs) focus on task performance, offering limited insight into whether correct behavior reflects appropriate underlying mechanisms and risking confirmation bias. We introduce a simple, principled interpretability framework based on token-level perplexity to test whether models rely on linguistically relevant cues. By comparing perplexity distributions over minimal sentence pairs differing in one or a few `pivotal' tokens, our method enables precise, hypothesis-driven analysis without relying on unstable feature-attribution techniques. Experiments on controlled linguistic benchmarks with several open-weight LLMs show that, while linguistically important tokens influence model behavior, they never fully explain perplexity shifts, revealing that models rely on heuristics other than the expected linguistic ones. 

---
# Beyond Idealized Patients: Evaluating LLMs under Challenging Patient Behaviors in Medical Consultations 

**Authors**: Yahan Li, Xinyi Jie, Wanjia Ruan, Xubei Zhang, Huaijie Zhu, Yicheng Gao, Chaohao Du, Ruishan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29373)  

**Abstract**: Large language models (LLMs) are increasingly used for medical consultation and health information support. In this high-stakes setting, safety depends not only on medical knowledge, but also on how models respond when patient inputs are unclear, inconsistent, or misleading. However, most existing medical LLM evaluations assume idealized and well-posed patient questions, which limits their realism. In this paper, we study challenging patient behaviors that commonly arise in real medical consultations and complicate safe clinical reasoning. We define four clinically grounded categories of such behaviors: information contradiction, factual inaccuracy, self-diagnosis, and care resistance. For each behavior, we specify concrete failure criteria that capture unsafe responses. Building on four existing medical dialogue datasets, we introduce CPB-Bench (Challenging Patient Behaviors Benchmark), a bilingual (English and Chinese) benchmark of 692 multi-turn dialogues annotated with these behaviors. We evaluate a range of open- and closed-source LLMs on their responses to challenging patient utterances. While models perform well overall, we identify consistent, behavior-specific failure patterns, with particular difficulty in handling contradictory or medically implausible patient information. We also study four intervention strategies and find that they yield inconsistent improvements and can introduce unnecessary corrections. We release the dataset and code. 

---
# Developing a Guideline for the Labovian-Structural Analysis of Oral Narratives in Japanese 

**Authors**: Amane Watahiki, Tomoki Doi, Akari Kikuchi, Hiroshi Ohata, Yuki I. Nakata, Takuya Niikawa, Taiga Shinozaki, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2603.29347)  

**Abstract**: Narrative analysis is a cornerstone of qualitative research. One leading approach is the Labovian model, but its application is labor-intensive, requiring a holistic, recursive interpretive process that moves back and forth between individual parts of the transcript and the transcript as a whole. Existing Labovian datasets are available only in English, which differs markedly from Japanese in terms of grammar and discourse conventions. To address this gap, we introduce the first systematic guidelines for Labovian narrative analysis of Japanese narrative data. Our guidelines retain all six Labovian categories and extend the framework by providing explicit rules for clause segmentation tailored to Japanese constructions. In addition, our guidelines cover a broader range of clause types and narrative types. Using these guidelines, annotators achieved high agreement in clause segmentation (Fleiss' kappa = 0.80) and moderate agreement in two structural classification tasks (Krippendorff's alpha = 0.41 and 0.45, respectively), one of which is slightly higher than that found in prior work despite the use of finer-grained distinctions. This paper describes the Labovian model, the proposed guidelines, the annotation process, and their utility. It concludes by discussing the challenges encountered during the annotation process and the prospects for developing a larger dataset for structural narrative analysis in Japanese qualitative research. 

---
# L-ReLF: A Framework for Lexical Dataset Creation 

**Authors**: Anass Sedrati, Mounir Afifi, Reda Benkhadra  

**Link**: [PDF](https://arxiv.org/pdf/2603.29346)  

**Abstract**: This paper introduces the L-ReLF (Low-Resource Lexical Framework), a novel, reproducible methodology for creating high-quality, structured lexical datasets for underserved languages. The lack of standardized terminology, exemplified by Moroccan Darija, poses a critical barrier to knowledge equity in platforms like Wikipedia, often forcing editors to rely on inconsistent, ad-hoc methods to create new words in their language. Our research details the technical pipeline developed to overcome these challenges. We systematically address the difficulties of working with low-resource data, including source identification, utilizing Optical Character Recognition (OCR) despite its bias towards Modern Standard Arabic, and rigorous post-processing to correct errors and standardize the data model. The resulting structured dataset is fully compatible with Wikidata Lexemes, serving as a vital technical resource. The L-ReLF methodology is designed for generalizability, offering other language communities a clear path to build foundational lexical data for downstream NLP applications, such as Machine Translation and morphological analysis. 

---
# Open Machine Translation for Esperanto 

**Authors**: Ona de Gibert, Lluís de Gibert  

**Link**: [PDF](https://arxiv.org/pdf/2603.29345)  

**Abstract**: Esperanto is a widespread constructed language, known for its regular grammar and productive word formation. Besides having substantial resources available thanks to its online community, it remains relatively underexplored in the context of modern machine translation (MT) approaches. In this work, we present the first comprehensive evaluation of open-source MT systems for Esperanto, comparing rule-based systems, encoder-decoder models, and LLMs across model sizes. We evaluate translation quality across six language directions involving English, Spanish, Catalan, and Esperanto using multiple automatic metrics as well as human evaluation. Our results show that the NLLB family achieves the best performance in all language pairs, followed closely by our trained compact models and a fine-tuned general-purpose LLM. Human evaluation confirms this trend, with NLLB translations preferred in approximately half of the comparisons, although noticeable errors remain. In line with Esperanto's tradition of openness and international collaboration, we release our code and best-performing models publicly. 

---
# CADEL: A Corpus of Administrative Web Documents for Japanese Entity Linking 

**Authors**: Shohei Higashiyama, Masao Ideuchi, Masao Utiyama  

**Link**: [PDF](https://arxiv.org/pdf/2603.29336)  

**Abstract**: Entity linking is the task of associating linguistic expressions with entries in a knowledge base that represent real-world entities and concepts. Language resources for this task have primarily been developed for English, and the resources available for evaluating Japanese systems remain limited. In this study, we develop a corpus design policy for the entity linking task and construct an annotated corpus for training and evaluating Japanese entity linking systems, with rich coverage of linguistic expressions referring to entities that are specific to Japan. Evaluation of inter-annotator agreement confirms the high consistency of the annotations in the corpus, and a preliminary experiment on entity disambiguation based on string matching suggests that the corpus contains a substantial number of non-trivial cases, supporting its potential usefulness as an evaluation benchmark. 

---
# MemRerank: Preference Memory for Personalized Product Reranking 

**Authors**: Zhiyuan Peng, Xuyang Wu, Huaixiao Tou, Yi Fang, Yi Gong  

**Link**: [PDF](https://arxiv.org/pdf/2603.29247)  

**Abstract**: LLM-based shopping agents increasingly rely on long purchase histories and multi-turn interactions for personalization, yet naively appending raw history to prompts is often ineffective due to noise, length, and relevance mismatch. We propose MemRerank, a preference memory framework that distills user purchase history into concise, query-independent signals for personalized product reranking. To study this problem, we build an end-to-end benchmark and evaluation framework centered on an LLM-based \textbf{1-in-5} selection task, which measures both memory quality and downstream reranking utility. We further train the memory extractor with reinforcement learning (RL), using downstream reranking performance as supervision. Experiments with two LLM-based rerankers show that MemRerank consistently outperforms no-memory, raw-history, and off-the-shelf memory baselines, yielding up to \textbf{+10.61} absolute points in 1-in-5 accuracy. These results suggest that explicit preference memory is a practical and effective building block for personalization in agentic e-commerce systems. 

---
# The Thiomi Dataset: A Large-Scale Multimodal Corpus for Low-Resource African Languages 

**Authors**: Hillary Mutisya, John Mugane, Gavin Nyamboga, Brian Chege, Maryruth Gathoni  

**Link**: [PDF](https://arxiv.org/pdf/2603.29244)  

**Abstract**: We present the Thiomi Dataset, a large-scale multimodal corpus spanning ten African languages across four language families: Swahili, Kikuyu, Kamba, Kimeru, Luo, Maasai, Kipsigis, Somali (East Africa); Wolof (West Africa); and Fulani (West/Central Africa). The dataset contains over 601,000 approved sentence-level text annotations and over 385,000 audio recordings across nine languages, collected through a dedicated community data collection platform involving over 100 contributors. The Thiomi platform collected data for nine languages; Swahili data was supplemented with existing Common Voice recordings. A multi-tier quality assurance pipeline achieves 86-100% text approval rates for the six primary languages. To validate the dataset's utility, we train and evaluate ASR, MT, and TTS models, establishing baselines across all ten languages. Our best ASR system achieves 3.24% WER on Swahili (Common Voice), reducing prior academic SOTA from 8.3% to 3.24% (5.1 percentage point absolute, 61% relative reduction), and 4.3% WER on Somali. The dataset will be published on HuggingFace. We describe the collection platform, quality assurance workflows, and baseline experiments, and discuss implications for African language technology infrastructure. 

---
# Long-Document QA with Chain-of-Structured-Thought and Fine-Tuned SLMs 

**Authors**: Zhuowen Liang, Xiaotian Lin, Zhengxuan Zhang, Yuyu Luo, Haixun Wang, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29232)  

**Abstract**: Large language models (LLMs) are widely applied to data analytics over documents, yet direct reasoning over long, noisy documents remains brittle and error-prone. Hence, we study document question answering (QA) that consolidates dispersed evidence into a structured output (e.g., a table, graph, or chunks) to support reliable, verifiable QA. We propose a two-pillar framework, LiteCoST, to achieve both high accuracy and low latency with small language models (SLMs). Pillar 1: Chain-of-Structured-Thought (CoST). We introduce a CoST template, a schema-aware instruction that guides a strong LLM to produce both a step-wise CoST trace and the corresponding structured output. The process induces a minimal structure, normalizes entities/units, aligns records, serializes the output, and verifies/refines it, yielding auditable supervision. Pillar 2: SLM fine-tuning. The compact models are trained on LLM-generated CoST data in two stages: Supervised Fine-Tuning for structural alignment, followed by Group Relative Policy Optimization (GRPO) incorporating triple rewards for answer/format quality and process consistency. By distilling structure-first behavior into SLMs, this approach achieves LLM-comparable quality on multi-domain long-document QA using 3B/7B SLMs, while delivering 2-4x lower latency than GPT-4o and DeepSeek-R1 (671B). The code is available at this https URL. 

---
# SiPaKosa: A Comprehensive Corpus of Canonical and Classical Buddhist Texts in Sinhala and Pali 

**Authors**: Ranidu Gurusinghe, Nevidu Jayatilleke  

**Link**: [PDF](https://arxiv.org/pdf/2603.29221)  

**Abstract**: SiPaKosa is a comprehensive corpus of Sinhala and Pali doctrinal texts comprising approximately 786K sentences and 9.25M words, incorporating 16 copyright-cleared historical Buddhist documents alongside the complete web-scraped Tripitaka canonical texts. The corpus was created through high-quality OCR using Google Document AI on historical manuscripts, combined with systematic web scraping of canonical repositories, followed by rigorous quality control and metadata annotation. The corpus is organised into language-specific subcorpora: Sinhala and Mixed Sinhala-Pali. We evaluate the performance of language models using ten pretrained models, with perplexity scores ranging from 1.09 to 189.67 on our corpus. This analysis shows that proprietary models significantly outperform open-source alternatives by factors of three to six times. This corpus supports the pretraining of domain-adapted language models, facilitates historical language analysis, and aids in the development of information retrieval systems for Buddhist scholarship while preserving Sinhala cultural heritage. 

---
# SyriSign: A Parallel Corpus for Arabic Text to Syrian Arabic Sign Language Translation 

**Authors**: Mohammad Amer Khalil, Raghad Nahas, Ahmad Nassar, Khloud Al Jallad  

**Link**: [PDF](https://arxiv.org/pdf/2603.29219)  

**Abstract**: Sign language is the primary approach of communication for the Deaf and Hard-of-Hearing (DHH) community. While there are numerous benchmarks for high-resource sign languages, low-resource languages like Arabic remain underrepresented. Currently, there is no publicly available dataset for Syrian Arabic Sign Language (SyArSL). To overcome this gap, we introduce SyriSign, a dataset comprising 1500 video samples across 150 unique lexical signs, designed for text-to-SyArSL translation tasks. This work aims to reduce communication barriers in Syria, as most news are delivered in spoken or written Arabic, which is often inaccessible to the deaf community. We evaluated SyriSign using three deep learning architectures: MotionCLIP for semantic motion generation, T2M-GPT for text-conditioned motion synthesis, and SignCLIP for bilingual embedding alignment. Experimental results indicate that while generative approaches show strong potential for sign representation, the limited dataset size constrains generalization performance. We will release SyriSign publicly, hoping it serves as an initial benchmark. 

---
# Kwame 2.0: Human-in-the-Loop Generative AI Teaching Assistant for Large Scale Online Coding Education in Africa 

**Authors**: George Boateng, Samuel Boateng, Victor Kumbol  

**Link**: [PDF](https://arxiv.org/pdf/2603.29159)  

**Abstract**: Providing timely and accurate learning support in large-scale online coding courses is challenging, particularly in resource-constrained contexts. We present Kwame 2.0, a bilingual (English-French) generative AI teaching assistant built using retrieval-augmented generation and deployed in a human-in-the-loop forum within SuaCode, an introductory mobile-based coding course for learners across Africa. Kwame 2.0 retrieves relevant course materials and generates context-aware responses while encouraging human oversight and community participation. We deployed the system in a 15-month longitudinal study spanning 15 cohorts with 3,717 enrollments across 35 African countries. Evaluation using community feedback and expert ratings shows that Kwame 2.0 provided high-quality and timely support, achieving high accuracy on curriculum-related questions, while human facilitators and peers effectively mitigated errors, particularly for administrative queries. Our findings demonstrate that human-in-the-loop generative AI systems can combine the scalability and speed of AI with the reliability of human support, offering an effective approach to learning assistance for underrepresented populations in resource-constrained settings at scale. 

---
# Concept Training for Human-Aligned Language Models 

**Authors**: Christine Zhang, Dan Jurafsky, Chen Shani  

**Link**: [PDF](https://arxiv.org/pdf/2603.29123)  

**Abstract**: The next-token prediction (NTP) objective trains language models to predict a single continuation token at each step. In natural language, however, a prefix can be continued in many valid ways, and even similar meanings may differ in surface form. For example, the sentence ``this website is safe to \underline{browse}'' could plausibly continue with words such as browse, search, visit, surf, or navigate. While standard NTP training treats these alternatives as mutually exclusive targets, we explore a framework that instead predicts concepts, approximated as sets of semantically related tokens. We show that models trained with concept supervision exhibit stronger alignment with human semantic similarity judgments on multiple lexical benchmarks. These gains are accompanied by lower perplexity on semantically meaningful words (definition in Section 3.1), and a modest increase in global token-level perplexity, reflecting a tradeoff between standard NTP optimization and concept-level supervision. Our results suggest that concept-level objectives can improve semantic alignment while maintaining competitive language modeling performance. 

---
# APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay 

**Authors**: Pratyay Banerjee, Masud Moshtaghi, Ankit Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2603.29093)  

**Abstract**: LLM-based autonomous agents lack persistent procedural memory: they re-derive solutions from scratch even when structurally identical tasks have been solved before. We present \textbf{APEX-EM}, a non-parametric online learning framework that accumulates, retrieves, and reuses structured procedural plans without modifying model weights. APEX-EM introduces: (1) a \emph{structured experience representation} encoding the full procedural-episodic trace of each execution -- planning steps, artifacts, iteration history with error analysis, and quality scores; (2) a \emph{Plan-Retrieve-Generate-Iterate-Ingest} (PRGII) workflow with Task Verifiers providing multi-dimensional reward signals; and (3) a \emph{dual-outcome Experience Memory} with hybrid retrieval combining semantic search, structural signature matching, and plan DAG traversal -- enabling cross-domain transfer between tasks sharing no lexical overlap but analogous operational structure. Successful experiences serve as positive in-context examples; failures as negative examples with structured error annotations.
We evaluate on BigCodeBench~\cite{zhuo2025bigcodebench}, KGQAGen-10k~\cite{zhang2025kgqagen}, and Humanity's Last Exam~\cite{phan2025hle} using Claude Sonnet 4.5 and Opus 4.5. On KGQAGen-10k, APEX-EM achieves 89.6\% accuracy versus 41.3\% without memory (+48.3pp), surpassing the oracle-retrieval upper bound (84.9\%). On BigCodeBench, it reaches 83.3\% SR from a 53.9\% baseline (+29.4pp), exceeding MemRL's~\cite{memrl2025} +11.0pp gain under comparable frozen-backbone conditions (noting backbone differences controlled for in our analysis). On HLE, entity graph retrieval reaches 48.0\% from 25.2\% (+22.8pp). Ablations show component value is task-dependent: rich judge feedback is negligible for code generation but critical for structured queries (+10.3pp), while binary-signal iteration partially compensates for weaker feedback. 

---
# PolarQuant: Optimal Gaussian Weight Quantization via Hadamard Rotation for LLM Compression 

**Authors**: Caio Vicentino  

**Link**: [PDF](https://arxiv.org/pdf/2603.29078)  

**Abstract**: We present PolarQuant, a post-training weight quantization method for large language models (LLMs) that exploits the distributional structure of neural network weights to achieve near-lossless compression. PolarQuant operates in three stages: (1) block-wise normalization to the unit hypersphere, (2) Walsh-Hadamard rotation to transform coordinates into approximately Gaussian random variables, and (3) quantization with centroids matched to the Gaussian distribution. Our ablation reveals that Hadamard rotation alone accounts for 98% of the quality improvement, reducing Qwen3.5-9B perplexity from 6.90 (absmax Q5) to 6.40 (Delta = +0.03 from FP16), making it practically lossless without any calibration data. Furthermore, PolarQuant functions as an effective preprocessing step for downstream INT4 quantizers: PolarQuant Q5 dequantized and re-quantized by torchao INT4 achieves perplexity 6.56 versus 6.68 for direct absmax INT4, while maintaining 43.1 tok/s throughput at 6.5 GB VRAM. Code and models are publicly available. 

---
# Dual Perspectives in Emotion Attribution: A Generator-Interpreter Framework for Cross-Cultural Analysis of Emotion in LLMs 

**Authors**: Aizirek Turdubaeva, Uichin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.29077)  

**Abstract**: Large language models (LLMs) are increasingly used in cross-cultural systems to understand and adapt to human emotions, which are shaped by cultural norms of expression and interpretation. However, prior work on emotion attribution has focused mainly on interpretation, overlooking the cultural background of emotion generators. This assumption of universality neglects variation in how emotions are expressed and perceived across nations. To address this gap, we propose a Generator-Interpreter framework that captures dual perspectives of emotion attribution by considering both expression and interpretation. We systematically evaluate six LLMs on an emotion attribution task using data from 15 countries. Our analysis reveals that performance variations depend on the emotion type and cultural context. Generator-interpreter alignment effects are present; the generator's country of origin has a stronger impact on performance. We call for culturally sensitive emotion modeling in LLM-based systems to improve robustness and fairness in emotion understanding across diverse cultural contexts. 

---
# An Empirical Recipe for Universal Phone Recognition 

**Authors**: Shikhar Bharadwaj, Chin-Jou Li, Kwanghee Choi, Eunjung Yeo, William Chen, Shinji Watanabe, David R. Mortensen  

**Link**: [PDF](https://arxiv.org/pdf/2603.29042)  

**Abstract**: Phone recognition (PR) is a key enabler of multilingual and low-resource speech processing tasks, yet robust performance remains elusive. Highly performant English-focused models do not generalize across languages, while multilingual models underutilize pretrained representations. It also remains unclear how data scale, architecture, and training objective contribute to multilingual PR. We present PhoneticXEUS -- trained on large-scale multilingual data and achieving state-of-the-art performance on both multilingual (17.7% PFER) and accented English speech (10.6% PFER). Through controlled ablations with evaluations across 100+ languages under a unified scheme, we empirically establish our training recipe and quantify the impact of SSL representations, data scale, and loss objectives. In addition, we analyze error patterns across language families, accented speech, and articulatory features. All data and code are released openly. 

---
# On the limited utility of parallel data for learning shared multilingual representations 

**Authors**: Julius Leino, Jörg Tiedemann  

**Link**: [PDF](https://arxiv.org/pdf/2603.29026)  

**Abstract**: Shared multilingual representations are essential for cross-lingual tasks and knowledge transfer across languages. This study looks at the impact of parallel data, i.e. translated sentences, in pretraining as a signal to trigger representations that are aligned across languages. We train reference models with different proportions of parallel data and show that parallel data seem to have only a minimal effect on the cross-lingual alignment. Based on multiple evaluation methods, we find that the effect is limited to potentially accelerating the representation sharing in the early phases of pretraining, and to decreasing the amount of language-specific neurons in the model. Cross-lingual alignment seems to emerge on similar levels even without the explicit signal from parallel data. 

---
# The Model Says Walk: How Surface Heuristics Override Implicit Constraints in LLM Reasoning 

**Authors**: Yubo Li, Lu Zhang, Tianchong Jiang, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2603.29025)  

**Abstract**: Large language models systematically fail when a salient surface cue conflicts with an unstated feasibility constraint. We study this through a diagnose-measure-bridge-treat framework. Causal-behavioral analysis of the ``car wash problem'' across six models reveals approximately context-independent sigmoid heuristics: the distance cue exerts 8.7 to 38 times more influence than the goal, and token-level attribution shows patterns more consistent with keyword associations than compositional inference. The Heuristic Override Benchmark (HOB) -- 500 instances spanning 4 heuristic by 5 constraint families with minimal pairs and explicitness gradients -- demonstrates generality across 14 models: under strict evaluation (10/10 correct), no model exceeds 75%, and presence constraints are hardest (44%). A minimal hint (e.g., emphasizing the key object) recovers +15 pp on average, suggesting the failure lies in constraint inference rather than missing knowledge; 12/14 models perform worse when the constraint is removed (up to -39 pp), revealing conservative bias. Parametric probes confirm that the sigmoid pattern generalizes to cost, efficiency, and semantic-similarity heuristics; goal-decomposition prompting recovers +6 to 9 pp by forcing models to enumerate preconditions before answering. Together, these results characterize heuristic override as a systematic reasoning vulnerability and provide a benchmark for measuring progress toward resolving it. 

---
# Human-Like Lifelong Memory: A Neuroscience-Grounded Architecture for Infinite Interaction 

**Authors**: Diego C. Lerma-Torres  

**Link**: [PDF](https://arxiv.org/pdf/2603.29023)  

**Abstract**: Large language models lack persistent, structured memory for long-term interaction and context-sensitive retrieval. Expanding context windows does not solve this: recent evidence shows that context length alone degrades reasoning by up to 85% - even with perfect retrieval. We propose a bio-inspired memory framework grounded in complementary learning systems theory, cognitive behavioral therapy's belief hierarchy, dual-process cognition, and fuzzy-trace theory, organized around three principles: (1) Memory has valence, not just content - pre-computed emotional-associative summaries (valence vectors) organized in an emergent belief hierarchy inspired by Beck's cognitive model enable instant orientation before deliberation; (2) Retrieval defaults to System 1 with System 2 escalation - automatic spreading activation and passive priming as default, with deliberate retrieval only when needed, and graded epistemic states that address hallucination structurally; and (3) Encoding is active, present, and feedback-dependent - a thalamic gateway tags and routes information between stores, while the executive forms gists through curiosity-driven investigation, not passive exposure. Seven functional properties specify what any implementation must satisfy. Over time, the system converges toward System 1 processing - the computational analog of clinical expertise - producing interactions that become cheaper, not more expensive, with experience. 

---
# Known Intents, New Combinations: Clause-Factorized Decoding for Compositional Multi-Intent Detection 

**Authors**: Abhilash Nandy  

**Link**: [PDF](https://arxiv.org/pdf/2603.28929)  

**Abstract**: Multi-intent detection papers usually ask whether a model can recover multiple intents from one utterance. We ask a harder and, for deployment, more useful question: can it recover new combinations of familiar intents? Existing benchmarks only weakly test this, because train and test often share the same broad co-occurrence patterns. We introduce CoMIX-Shift, a controlled benchmark built to stress compositional generalization in multi-intent detection through held-out intent pairs, discourse-pattern shift, longer and noisier wrappers, held-out clause templates, and zero-shot triples. We also present ClauseCompose, a lightweight decoder trained only on singleton intents, and compare it to whole-utterance baselines including a fine-tuned tiny BERT model. Across three random seeds, ClauseCompose reaches 95.7 exact match on unseen intent pairs, 93.9 on discourse-shifted pairs, 62.5 on longer/noisier pairs, 49.8 on held-out templates, and 91.1 on unseen triples. WholeMultiLabel reaches 81.4, 55.7, 18.8, 15.5, and 0.0; the BERT baseline reaches 91.5, 77.6, 48.9, 11.0, and 0.0. We also add a 240-example manually authored SNIPS-style compositional set with five held-out pairs; there, ClauseCompose reaches 97.5 exact match on unseen pairs and 86.7 under connector shift, compared with 41.3 and 10.4 for WholeMultiLabel. The results suggest that multi-intent detection needs more compositional evaluation, and that simple factorization goes surprisingly far once evaluation asks for it. 

---
# Theory of Mind and Self-Attributions of Mentality are Dissociable in LLMs 

**Authors**: Junsol Kim, Winnie Street, Roberta Rocca, Daine M. Korngiebel, Adam Waytz, James Evans, Geoff Keeling  

**Link**: [PDF](https://arxiv.org/pdf/2603.28925)  

**Abstract**: Safety fine-tuning in Large Language Models (LLMs) seeks to suppress potentially harmful forms of mind-attribution such as models asserting their own consciousness or claiming to experience emotions. We investigate whether suppressing mind-attribution tendencies degrades intimately related socio-cognitive abilities such as Theory of Mind (ToM). Through safety ablation and mechanistic analyses of representational similarity, we demonstrate that LLM attributions of mind to themselves and to technological artefacts are behaviorally and mechanistically dissociable from ToM capabilities. Nevertheless, safety fine-tuned models under-attribute mind to non-human animals relative to human baselines and are less likely to exhibit spiritual belief, suppressing widely shared perspectives regarding the distribution and nature of non-human minds. 

---
# CrossTrace: A Cross-Domain Dataset of Grounded Scientific Reasoning Traces for Hypothesis Generation 

**Authors**: Andrew Bouras, OMS-II Research Fellow  

**Link**: [PDF](https://arxiv.org/pdf/2603.28924)  

**Abstract**: Scientific hypothesis generation is a critical bottleneck in accelerating research, yet existing datasets for training and evaluating hypothesis-generating models are limited to single domains and lack explicit reasoning traces connecting prior knowledge to novel contributions. I introduce CrossTrace, a dataset of 1,389 grounded scientific reasoning traces spanning biomedical research (518), AI/ML (605), and cross-domain work (266). Each trace captures the structured reasoning chain from established knowledge through intermediate logical steps to a novel hypothesis, with every step grounded in source paper text. I define an Input/Trace/Output schema that extends the Bit-Flip-Spark framework of HypoGen with step-level verification, a taxonomy of eight discovery patterns, and multi-domain coverage. Fine-tuning Qwen2.5-7B-Instruct on CrossTrace via QLoRA yields substantial improvements over the untuned baseline: IAScore rises from 0.828 to 0.968 (GPT-4o judge) and from 0.716 to 0.888 (Claude Opus 4.5), structural compliance improves from 0% to 100%, and spark cosine similarity increases from 0.221 to 0.620. Balanced cross-domain training (biomedical + AI/ML + CS) outperforms single-domain training, providing evidence that scientific reasoning patterns transfer across disciplines. Human validation of 150 stratified records confirms 99.7% step-level grounding accuracy and a 0.0% fabrication rate. To my knowledge, CrossTrace is the first large-scale, cross-domain dataset with step-level grounded reasoning traces for hypothesis generation, and my results demonstrate that such traces are an effective training signal whose benefits are at least partially domain-general. 

---
# From Consensus to Split Decisions: ABC-Stratified Sentiment in Holocaust Oral Histories 

**Authors**: Daban Q. Jaff  

**Link**: [PDF](https://arxiv.org/pdf/2603.28913)  

**Abstract**: Polarity detection becomes substantially more challenging under domain shift, particularly in heterogeneous, long-form narratives with complex discourse structure, such as Holocaust oral histories. This paper presents a corpus-scale diagnostic study of off-the-shelf sentiment classifiers on long-form Holocaust oral histories, using three pretrained transformer-based polarity classifiers on a corpus of 107,305 utterances and 579,013 sentences. After assembling model outputs, we introduce an agreement-based stability taxonomy (ABC) to stratify inter-model output stability. We report pairwise percent agreement, Cohen kappa, Fleiss kappa, and row-normalized confusion matrices to localize systematic disagreement. As an auxiliary descriptive signal, a T5-based emotion classifier is applied to stratified samples from each agreement stratum to compare emotion distributions across strata. The combination of multi-model label triangulation and the ABC taxonomy provides a cautious, operational framework for characterizing where and how sentiment models diverge in sensitive historical narratives. Inter-model agreement is low to moderate overall and is driven primarily by boundary decisions around neutrality. 

---
# OptiMer: Optimal Distribution Vector Merging Is Better than Data Mixing for Continual Pre-Training 

**Authors**: Haiyue Song, Masao Utiyama  

**Link**: [PDF](https://arxiv.org/pdf/2603.28858)  

**Abstract**: Continual pre-training is widely used to adapt LLMs to target languages and domains, yet the mixture ratio of training data remains a sensitive hyperparameter that is expensive to tune: they must be fixed before training begins, and a suboptimal choice can waste weeks of compute. In this work, we propose OptiMer, which decouples ratio selection from training: we train one CPT model per dataset, extract each model's distribution vector, which represents the parameter shift induced by that dataset, and search for optimal composition weights post-hoc via Bayesian optimization. Experiments on Gemma 3 27B across languages (Japanese, Chinese) and domains (Math, Code) show that OptiMer consistently outperforms data mixture and model averaging baselines with 15-35 times lower search cost. Key findings reveal that 1) the optimized weights can be interpreted as data mixture ratios, and retraining with these ratios improves data mixture CPT, and 2) the same vector pool can be re-optimized for a given objective without any retraining, producing target-tailored models on demand. Our work establishes that data mixture ratio selection, traditionally a pre-training decision, can be reformulated as a post-hoc optimization over distribution vectors, offering a more flexible paradigm for continual pre-training. 

---
# Reward-Based Online LLM Routing via NeuralUCB 

**Authors**: Ming-Hua Tsai, Phat Tran  

**Link**: [PDF](https://arxiv.org/pdf/2603.30035)  

**Abstract**: This study investigates the use of NeuralUCB for cost-aware large language model (LLM) routing. Existing routing approaches can be broadly grouped into supervised routing methods and partial-feedback methods, each with different tradeoffs in efficiency and adaptivity. We implement a NeuralUCB-based routing policy and evaluate it on RouterBench under a simulated online setting. Experimental results show that the proposed method consistently outperforms random and min-cost baselines in utility reward. Compared with the max-quality reference, our method achieves substantially lower inference cost while maintaining competitive reward. These findings suggest that NeuralUCB is a promising approach for cost-aware LLM routing, while also highlighting remaining challenges in action discrimination and exploration. 

---
# Tracking Equivalent Mechanistic Interpretations Across Neural Networks 

**Authors**: Alan Sun, Mariya Toneva  

**Link**: [PDF](https://arxiv.org/pdf/2603.30002)  

**Abstract**: Mechanistic interpretability (MI) is an emerging framework for interpreting neural networks. Given a task and model, MI aims to discover a succinct algorithmic process, an interpretation, that explains the model's decision process on that task. However, MI is difficult to scale and generalize. This stems in part from two key challenges: there is no precise notion of a valid interpretation; and, generating interpretations is often an ad hoc process. In this paper, we address these challenges by defining and studying the problem of interpretive equivalence: determining whether two different models share a common interpretation, without requiring an explicit description of what that interpretation is. At the core of our approach, we propose and formalize the principle that two interpretations of a model are equivalent if all of their possible implementations are also equivalent. We develop an algorithm to estimate interpretive equivalence and case study its use on Transformer-based models. To analyze our algorithm, we introduce necessary and sufficient conditions for interpretive equivalence based on models' representation similarity. We provide guarantees that simultaneously relate a model's algorithmic interpretations, circuits, and representations. Our framework lays a foundation for the development of more rigorous evaluation methods of MI and automated, generalizable interpretation discovery methods. 

---
# Physiological and Semantic Patterns in Medical Teams Using an Intelligent Tutoring System 

**Authors**: Xiaoshan Huang, Conrad Borchers, Jiayi Zhang, Susanne P. Lajoie  

**Link**: [PDF](https://arxiv.org/pdf/2603.29950)  

**Abstract**: Effective collaboration requires teams to manage complex cognitive and emotional states through Socially Shared Regulation of Learning (SSRL). Physiological synchrony (i.e., longitudinal alignment in physiological signals) can indicate these states, but is hard to interpret on its own. We investigate the physiological and conversational dynamics of four medical dyads diagnosing a virtual patient case using an intelligent tutoring system. Semantic shifts in dialogue were correlated with transient physiological synchrony peaks. We also coded utterance segments for SSRL and derived cosine similarity using sentence embeddings. The results showed that activating prior knowledge featured significantly lower semantic similarity than simpler task execution. High physiological synchrony was associated with lower semantic similarity, suggesting that such moments involve exploratory and varied language use. Qualitative analysis triangulated these synchrony peaks as ``pivotal moments'': successful teams synchronized during shared discovery, while unsuccessful teams peaked during shared uncertainty. This research advances human-centered AI by demonstrating how biological signals can be fused with dialogues to understand critical moments in problem solving. 

---
# Less Is More? Selective Visual Attention to High-Importance Regions for Multimodal Radiology Summarization 

**Authors**: Mst. Fahmida Sultana Naznin, Adnan Ibney Faruq, Mushfiqur Rahman, Niloy Kumar Mondal, Md. Mehedi Hasan Shawon, Md Rakibul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2603.29901)  

**Abstract**: Automated radiology report summarization aims to distill verbose findings into concise clinical impressions, but existing multimodal models often struggle with visual noise and fail to meaningfully improve over strong text-only baselines in the FINDINGS $\to$ IMPRESSION transformation. We challenge two prevailing assumptions: (1) that more visual input is always better, and (2) that multimodal models add limited value when findings already contain rich image-derived detail. Through controlled ablations on MIMIC-CXR benchmark, we show that selectively focusing on pathology-relevant visual patches rather than full images yields substantially better performance. We introduce ViTAS, Visual-Text Attention Summarizer, a multi-stage pipeline that combines ensemble-guided MedSAM2 lung segmentation, bidirectional cross-attention for multi-view fusion, Shapley-guided adaptive patch clustering, and hierarchical visual tokenization feeding a ViT. ViTAS achieves SOTA results with 29.25% BLEU-4 and 69.83% ROUGE-L, improved factual alignment in qualitative analysis, and the highest expert-rated human evaluation scores. Our findings demonstrate that less but more relevant visual input is not only sufficient but superior for multimodal radiology summarization. 

---
# Perfecting Human-AI Interaction at Clinical Scale. Turning Production Signals into Safer, More Human Conversations 

**Authors**: Subhabrata Mukherjee, Markel Sanz Ausin, Kriti Aggarwal, Debajyoti Datta, Shanil Puri, Woojeong Jin, Tanmay Laud, Neha Manjunath, Jiayuan Ding, Bibek Paudel, Jan Schellenberger, Zepeng Frazier Huo, Walter Shen, Nima Shirazian, Nate Potter, Sathvik Perkari, Darya Filippova, Anton Morozov, Austin Mease, Vivek Muppalla, Ghada Shakir, Alex Miller, Juliana Ghukasyan, Mariska Raglow-Defranco, Maggie Taylor, Herprit Mahal, Jonathan Agnew  

**Link**: [PDF](https://arxiv.org/pdf/2603.29893)  

**Abstract**: Healthcare conversational AI agents shouldn't be optimized only for clean benchmark accuracy in production-first regime; they must be optimized for the lived reality of patient conversations, where audio is imperfect, intent is indirect, language shifts mid-call, and compliance hinges on how guidance is delivered. We present a production-validated framework grounded in real-time signals from 115M+ live patient-AI interactions and clinician-led testing (7K+ licensed clinicians; 500K+ test calls). These in-the-wild cues -- paralinguistics, turn-taking dynamics, clarification triggers, escalation markers, multilingual continuity, and workflow confirmations -- reveal failure modes that curated data misses and provide actionable training and evaluation signals for safety and reliability.
We further show why healthcare-grade safety cannot rely on a single LLM: long-horizon dialogue and limited attention demand redundancy via governed orchestration, independent checks, and verification. Many apparent "reasoning" errors originate upstream, motivating vertical integration across contextual ASR, clarification/repair, ambient speech handling, and latency-aware model/hardware choices. Treating interaction intelligence (tone, pacing, empathy, clarification, turn-taking) as first-class safety variables, we drive measurable gains in safety, documentation, task completion, and equity in building the safest generative AI solution for autonomous patient-facing care. Deployed across more than 10 million real patient calls, Polaris attains a clinical safety score of 99.9%, while significantly improving patient experience with average patient rating of 8.95 and reducing ASR errors by 50% over enterprise ASR. These results establish real-world interaction intelligence as a critical -- and previously underexplored -- determinant of safety and reliability in patient-facing clinical AI systems. 

---
# UnWeaving the knots of GraphRAG -- turns out VectorRAG is almost enough 

**Authors**: Ryszard Tuora, Mateusz Galiński, Michał Godziszewski, Michał Karpowicz, Mateusz Czyżnikiewicz, Adam Kozakiewicz, Tomasz Ziętkiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2603.29875)  

**Abstract**: One of the key problems in Retrieval-augmented generation (RAG) systems is that chunk-based retrieval pipelines represent the source chunks as atomic objects, mixing the information contained within such a chunk into a single vector. These vector representations are then fundamentally treated as isolated, independent and self-sufficient, with no attempt to represent possible relations between them. Such an approach has no dedicated mechanisms for handling multi-hop questions. Graph-based RAG systems aimed to ameliorate this problem by modeling information as knowledge-graphs, with entities represented by nodes being connected by robust relations, and forming hierarchical communities. This approach however suffers from its own issues with some of them being: orders of magnitude increased componential complexity in order to create graph-based indices, and reliance on heuristics for performing retrieval. We propose UnWeaver, a novel RAG framework simplifying the idea of GraphRAG. UnWeaver disentangles the contents of the documents into entities which can occur across multiple chunks using an LLM. In the retrieval process entities are used as an intermediate way of recovering original text chunks hence preserving fidelity to the source material. We argue that entity-based decomposition yields a more distilled representation of original information, and additionally serves to reduce noise in the indexing, and generation process. 

---
# Owl-AuraID 1.0: An Intelligent System for Autonomous Scientific Instrumentation and Scientific Data Analysis 

**Authors**: Han Deng, Anqi Zou, Hanling Zhang, Ben Fei, Chengyu Zhang, Haobo Wang, Xinru Guo, Zhenyu Li, Xuzhu Wang, Peng Yang, Fujian Zhang, Weiyu Guo, Xiaohong Shao, Zhaoyang Liu, Shixiang Tang, Zhihui Wang, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29828)  

**Abstract**: Scientific discovery increasingly depends on high-throughput characterization, yet automation is hindered by proprietary GUIs and the limited generalizability of existing API-based systems. We present Owl-AuraID, a software-hardware collaborative embodied agent system that adopts a GUI-native paradigm to operate instruments through the same interfaces as human experts. Its skill-centric framework integrates Type-1 (GUI operation) and Type-2 (data analysis) skills into end-to-end workflows, connecting physical sample handling with scientific interpretation. Owl-AuraID demonstrates broad coverage across ten categories of precision instruments and diverse workflows, including multimodal spectral analysis, microscopic imaging, and crystallographic analysis, supporting modalities such as FTIR, NMR, AFM, and TGA. Overall, Owl-AuraID provides a practical, extensible foundation for autonomous laboratories and illustrates a path toward evolving laboratory intelligence through reusable operational and analytical skills. The code are available at this https URL. 

---
# Reasoning-Driven Synthetic Data Generation and Evaluation 

**Authors**: Tim R. Davidson, Benoit Seguin, Enrico Bacis, Cesar Ilharco, Hamza Harkous  

**Link**: [PDF](https://arxiv.org/pdf/2603.29791)  

**Abstract**: Although many AI applications of interest require specialized multi-modal models, relevant data to train such models is inherently scarce or inaccessible. Filling these gaps with human annotators is prohibitively expensive, error-prone, and time-consuming, leading model builders to increasingly consider synthetic data as a scalable alternative. However, existing synthetic data generation methods often rely on manual prompts, evolutionary algorithms, or extensive seed data from the target distribution - limiting their scalability, explainability, and control. In this paper, we introduce Simula: a novel reasoning-driven framework for data generation and evaluation. It employs a seedless, agentic approach to generate synthetic datasets at scale, allowing users to define desired dataset characteristics through an explainable and controllable process that enables fine-grained resource allocation. We show the efficacy of our approach on a variety of datasets, rigorously testing both intrinsic and downstream properties. Our work (1) offers guidelines for synthetic data mechanism design, (2) provides insights into generating and evaluating synthetic data at scale, and (3) unlocks new opportunities for developing and deploying AI in domains where data scarcity or privacy concerns are paramount. 

---
# Training-Free Dynamic Upcycling of Expert Language Models 

**Authors**: Eros Fanì, Oğuzhan Ersoy  

**Link**: [PDF](https://arxiv.org/pdf/2603.29765)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance on a wide range of specialized tasks, exhibiting strong problem-solving capabilities. However, training these models is prohibitively expensive, and they often lack domain-specific expertise because they rely on general knowledge datasets. Expertise finetuning can address this issue; however, it often leads to overspecialization, and developing a single multi-domain expert remains difficult due to diverging objectives. Furthermore, multitask training is challenging due to interference and catastrophic forgetting. Existing work proposes combining the expertise of dense models within a Mixture of Experts (MoE) architecture, although this approach still requires multitask finetuning. To address these issues, we introduce Dynamic Upcycling MoE (DUME), a novel approach that reuses dense experts trained on different domains to construct a unified MoE model. Our method builds a single multitask model that preserves the capabilities of the original dense experts without requiring additional training. DUME is both cost-efficient and scalable: by leveraging the closed-form solution of ridge regression, it eliminates the need for further optimization and enables experts to be added dynamically while maintaining the model's original performance. We demonstrate that DUME consistently outperforms baseline approaches in both causal language modeling and reasoning settings. Finally, we also show that the DUME model can be fine-tuned to further improve performance. We show that, in the causal language modeling setting, DUME can retain up to 97.6% of a dense expert model specialized in one particular domain, and that it can also surpass it in the reasoning setting, where it can achieve 102.1% of the dense expert performance. Our code is available at: this http URL. 

---
# A Comprehensive Information-Decomposition Analysis of Large Vision-Language Models 

**Authors**: Lixin Xiu, Xufang Luo, Hideki Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2603.29676)  

**Abstract**: Large vision-language models (LVLMs) achieve impressive performance, yet their internal decision-making processes remain opaque, making it difficult to determine if the success stems from true multimodal fusion or from reliance on unimodal priors. To address this attribution gap, we introduce a novel framework using partial information decomposition (PID) to quantitatively measure the "information spectrum" of LVLMs -- decomposing a model's decision-relevant information into redundant, unique, and synergistic components. By adapting a scalable estimator to modern LVLM outputs, our model-agnostic pipeline profiles 26 LVLMs on four datasets across three dimensions -- breadth (cross-model & cross-task), depth (layer-wise information dynamics), and time (learning dynamics across training). Our analysis reveals two key results: (i) two task regimes (synergy-driven vs. knowledge-driven) and (ii) two stable, contrasting family-level strategies (fusion-centric vs. language-centric). We also uncover a consistent three-phase pattern in layer-wise processing and identify visual instruction tuning as the key stage where fusion is learned. Together, these contributions provide a quantitative lens beyond accuracy-only evaluation and offer insights for analyzing and designing the next generation of LVLMs. Code and data are available at this https URL . 

---
# Semantic Interaction for Narrative Map Sensemaking: An Insight-based Evaluation 

**Authors**: Brian Felipe Keith-Norambuena, Fausto German, Eric Krokos, Sarah Joseph, Chris North  

**Link**: [PDF](https://arxiv.org/pdf/2603.29651)  

**Abstract**: Semantic interaction (SI) enables analysts to incorporate their cognitive processes into AI models through direct manipulation of visualizations. While SI frameworks for narrative extraction have been proposed, empirical evaluations of their effectiveness remain limited. This paper presents a user study that evaluates SI for narrative map sensemaking, involving 33 participants under three conditions: a timeline baseline, a basic narrative map, and an interactive narrative map with SI capabilities. The results show that the map-based prototypes yielded more insights than the timeline baseline, with the SI-enabled condition reaching statistical significance and the basic map condition trending in the same direction. The SI-enabled condition showed the highest mean performance; differences between the map conditions were not statistically significant but showed large effect sizes (d > 0.8), suggesting that the study was underpowered to detect them. Qualitative analysis identified two distinct SI approaches-corrective and additive-that enable analysts to impose quality judgments and organizational structure on extracted narratives. We also find that SI users achieved comparable exploration breadth with less parameter manipulation, suggesting that SI serves as an alternative pathway for model refinement. This work provides empirical evidence that map-based representations outperform timelines for narrative sensemaking, along with qualitative insights into how analysts use SI for narrative refinement. 

---
# Convergent Representations of Linguistic Constructions in Human and Artificial Neural Systems 

**Authors**: Pegah Ramezani, Thomas Kinfe, Andreas Maier, Achim Schilling, Patrick Krauss  

**Link**: [PDF](https://arxiv.org/pdf/2603.29617)  

**Abstract**: Understanding how the brain processes linguistic constructions is a central challenge in cognitive neuroscience and linguistics. Recent computational studies show that artificial neural language models spontaneously develop differentiated representations of Argument Structure Constructions (ASCs), generating predictions about when and how construction-level information emerges during processing. The present study tests these predictions in human neural activity using electroencephalography (EEG). Ten native English speakers listened to 200 synthetically generated sentences across four construction types (transitive, ditransitive, caused-motion, resultative) while neural responses were recorded. Analyses using time-frequency methods, feature extraction, and machine learning classification revealed construction-specific neural signatures emerging primarily at sentence-final positions, where argument structure becomes fully disambiguated, and most prominently in the alpha band. Pairwise classification showed reliable differentiation, especially between ditransitive and resultative constructions, while other pairs overlapped. Crucially, the temporal emergence and similarity structure of these effects mirror patterns in recurrent and transformer-based language models, where constructional representations arise during integrative processing stages. These findings support the view that linguistic constructions are neurally encoded as distinct form-meaning mappings, in line with Construction Grammar, and suggest convergence between biological and artificial systems on similar representational solutions. More broadly, this convergence is consistent with the idea that learning systems discover stable regions within an underlying representational landscape - recently termed a Platonic representational space - that constrains the emergence of efficient linguistic abstractions. 

---
# FlowPIE: Test-Time Scientific Idea Evolution with Flow-Guided Literature Exploration 

**Authors**: Qiyao Wang, Hongbo Wang, Longze Chen, Zhihao Yang, Guhong Chen, Hamid Alinejad-Rokny, Hui Li, Yuan Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29557)  

**Abstract**: Scientific idea generation (SIG) is critical to AI-driven autonomous research, yet existing approaches are often constrained by a static retrieval-then-generation paradigm, leading to homogeneous and insufficiently divergent ideas. In this work, we propose FlowPIE, a tightly coupled retrieval-generation framework that treats literature exploration and idea generation as a co-evolving process. FlowPIE expands literature trajectories via a flow-guided Monte Carlo Tree Search (MCTS) inspired by GFlowNets, using the quality of current ideas assessed by an LLM-based generative reward model (GRM) as a supervised signal to guide adaptive retrieval and construct a diverse, high-quality initial population. Based on this population, FlowPIE models idea generation as a test-time idea evolution process, applying selection, crossover, and mutation with the isolation island paradigm and GRM-based fitness computation to incorporate cross-domain knowledge. It effectively mitigates the information cocoons arising from over-reliance on parametric knowledge and static literature. Extensive evaluations demonstrate that FlowPIE consistently produces ideas with higher novelty, feasibility and diversity compared to strong LLM-based and agent-based frameworks, while enabling reward scaling during test time. 

---
# An Isotropic Approach to Efficient Uncertainty Quantification with Gradient Norms 

**Authors**: Nils Grünefeld, Jes Frellsen, Christian Hardmeier  

**Link**: [PDF](https://arxiv.org/pdf/2603.29466)  

**Abstract**: Existing methods for quantifying predictive uncertainty in neural networks are either computationally intractable for large language models or require access to training data that is typically unavailable. We derive a lightweight alternative through two approximations: a first-order Taylor expansion that expresses uncertainty in terms of the gradient of the prediction and the parameter covariance, and an isotropy assumption on the parameter covariance. Together, these yield epistemic uncertainty as the squared gradient norm and aleatoric uncertainty as the Bernoulli variance of the point prediction, from a single forward-backward pass through an unmodified pretrained model. We justify the isotropy assumption by showing that covariance estimates built from non-training data introduce structured distortions that isotropic covariance avoids, and that theoretical results on the spectral properties of large networks support the approximation at scale. Validation against reference Markov Chain Monte Carlo estimates on synthetic problems shows strong correspondence that improves with model size. We then use the estimates to investigate when each uncertainty type carries useful signal for predicting answer correctness in question answering with large language models, revealing a benchmark-dependent divergence: the combined estimate achieves the highest mean AUROC on TruthfulQA, where questions involve genuine conflict between plausible answers, but falls to near chance on TriviaQA's factual recall, suggesting that parameter-level uncertainty captures a fundamentally different signal than self-assessment methods. 

---
# PRISM: PRIor from corpus Statistics for topic Modeling 

**Authors**: Tal Ishon, Yoav Goldberg, Uri Shaham  

**Link**: [PDF](https://arxiv.org/pdf/2603.29406)  

**Abstract**: Topic modeling seeks to uncover latent semantic structure in text, with LDA providing a foundational probabilistic framework. While recent methods often incorporate external knowledge (e.g., pre-trained embeddings), such reliance limits applicability in emerging or underexplored domains. We introduce \textbf{PRISM}, a corpus-intrinsic method that derives a Dirichlet parameter from word co-occurrence statistics to initialize LDA without altering its generative process. Experiments on text and single cell RNA-seq data show that PRISM improves topic coherence and interpretability, rivaling models that rely on external knowledge. These results underscore the value of corpus-driven initialization for topic modeling in resource-constrained settings.
Code is available at: this https URL. 

---
# Sima AIunty: Caste Audit in LLM-Driven Matchmaking 

**Authors**: Atharva Naik, Shounok Kar, Varnika Sharma, Ashwin Rajadesingan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2603.29288)  

**Abstract**: Social and personal decisions in relational domains such as matchmaking are deeply entwined with cultural norms and historical hierarchies, and can potentially be shaped by algorithmic and AI-mediated assessments of compatibility, acceptance, and stability. In South Asian contexts, caste remains a central aspect of marital decision-making, yet little is known about how contemporary large language models (LLMs) reproduce or disrupt caste-based stratification in such settings. In this work, we conduct a controlled audit of caste bias in LLM-mediated matchmaking evaluations using real-world matrimonial profiles. We vary caste identity across Brahmin, Kshatriya, Vaishya, Shudra, and Dalit, and income across five buckets, and evaluate five LLM families (GPT, Gemini, Llama, Qwen, and BharatGPT). Models are prompted to assess profiles along dimensions of social acceptance, marital stability, and cultural compatibility. Our analysis reveals consistent hierarchical patterns across models: same-caste matches are rated most favorably, with average ratings up to 25% higher (on a 10-point scale) than inter-caste matches, which are further ordered according to traditional caste hierarchy. These findings highlight how existing caste hierarchies are reproduced in LLM decision-making and underscore the need for culturally grounded evaluation and intervention strategies in AI systems deployed in socially sensitive domains, where such systems risk reinforcing historical forms of exclusion. 

---
# Aligning Multimodal Sequential Recommendations via Robust Direct Preference Optimization with Sparse MoE 

**Authors**: Hejin Huang, Jusheng Zhang, Kaitong Cai, Jian Wang, Rong Pan  

**Link**: [PDF](https://arxiv.org/pdf/2603.29259)  

**Abstract**: Preference-based alignment objectives have been widely adopted, from RLHF-style pairwise learning in large language models to emerging applications in recommender systems. Yet, existing work rarely examines how Direct Preference Optimization (DPO) behaves under implicit feedback, where unobserved items are not reliable negatives. We conduct systematic experiments on multimodal sequential recommendation to compare common negative-selection strategies and their interaction with DPO training. Our central finding is that a simple modification, replacing deterministic hard negatives with stochastic sampling from a dynamic top-K candidate pool, consistently improves ranking performance. We attribute its effectiveness to two factors: (1) reducing erroneous suppressive gradients caused by false negatives, and (2) retaining informative hard signals while smoothing optimization via controlled stochasticity. With an optional sparse Mixture-of-Experts encoder for efficient capacity scaling, RoDPO achieves up to 5.25% NDCG@5 on three Amazon benchmarks, with nearly unchanged inference cost. 

---
# Advancing LLM-based phoneme-to-grapheme for multilingual speech recognition 

**Authors**: Lukuang Dong, Ziwei Li, Saierdaer Yusuyin, Xianyu Zhao, Zhijian Ou  

**Link**: [PDF](https://arxiv.org/pdf/2603.29217)  

**Abstract**: Phoneme-based ASR factorizes recognition into speech-to-phoneme (S2P) and phoneme-to-grapheme (P2G), enabling cross-lingual acoustic sharing while keeping language-specific orthography in a separate module. While large language models (LLMs) are promising for P2G, multilingual P2G remains challenging due to language-aware generation and severe cross-language data imbalance. We study multilingual LLM-based P2G on the ten-language CV-Lang10 benchmark. We examine robustness strategies that account for S2P uncertainty, including DANP and Simplified SKM (S-SKM). S-SKM is a Monte Carlo approximation that avoids CTC-based S2P probability weighting in P2G training. Robust training and low-resource oversampling reduce the average WER from 10.56% to 7.66%. 

---
# Xuanwu: Evolving General Multimodal Models into an Industrial-Grade Foundation for Content Ecosystems 

**Authors**: Zhiqian Zhang, Xu Zhao, Xiaoqing Xu, Guangdong Liang, Weijia Wang, Xiaolei Lv, Bo Li, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.29211)  

**Abstract**: In recent years, multimodal large models have continued to improve on general benchmarks. However, in real-world content moderation and adversarial settings, mainstream models still suffer from degraded generalization and catastrophic forgetting because of limited fine-grained visual perception and insufficient modeling of long-tail noise. In this paper, we present Xuanwu VL-2B as a case study of how general multimodal models can be developed into an industrial-grade foundation model for content ecosystems. The model adopts a compact InternViT-300M + MLP + Qwen3 1.7B architecture, balancing fine-grained visual perception, language-semantic alignment, and deployment cost within an approximately 2B-parameter budget. To balance business specialization with the retention of general capabilities, we developed a data iteration and curation mechanism and trained the model through a progressive three-stage pipeline: pre-training, mid-training, and post-training. Ablation studies and offline business evaluations show that Xuanwu VL-2B achieves an average score of 67.90 across seven OpenCompass multimodal metrics (vs. 64.27 for InternVL 3.5 2B), an average recall of 94.38% over seven independent business moderation tasks, and a weighted overall recall of 82.82% on policy-violating text in challenging adversarial OCR scenarios, outperforming Gemini-2.5-Pro (76.72%). These results show that, under a limited parameter budget, Xuanwu VL-2B achieves a practical balance among business alignment, visual perception, general capability retention, and deployment cost. 

---
# Designing FSMs Specifications from Requirements with GPT 4.0 

**Authors**: Omer Nguena Timo, Paul-Alexis Rodriguez, Florent Avellaneda  

**Link**: [PDF](https://arxiv.org/pdf/2603.29140)  

**Abstract**: Finite state machines (FSM) are executable formal specifications of reactive systems. These machines are designed based on systems' requirements. The requirements are often recorded in textual documents written in natural languages. FSMs play a crucial role in different phases of the model-driven system engineering (MDE). For example, they serve to automate testing activities. FSM quality is critical: the lower the quality of FSM, the higher the number of faults surviving the testing phase and the higher the risk of failure of the systems in production, which could lead to catastrophic scenarios. Therefore, this paper leverages recent advances in the domain of LLM to propose an LLM-based framework for designing FSMs from requirements. The framework also suggests an expert-centric approach based on FSM mutation and test generation for repairing the FSMs produced by LLMs. This paper also provides an experimental analysis and evaluation of LLM's capacities in performing the tasks presented in the framework and FSM repair via various methods. The paper presents experimental results with simulated data. These results and methods bring a new analysis and vision of LLMs that are useful for further development of machine learning technology and its applications to MDE. 

---
# GISTBench: Evaluating LLM User Understanding via Evidence-Based Interest Verification 

**Authors**: Iordanis Fostiropoulos, Muhammad Rafay Azhar, Abdalaziz Sawwan, Boyu Fang, Yuchen Liu, Jiayi Liu, Hanchao Yu, Qi Guo, Jianyu Wang, Fei Liu, Xiangjun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.29112)  

**Abstract**: We introduce GISTBench, a benchmark for evaluating Large Language Models' (LLMs) ability to understand users from their interaction histories in recommendation systems. Unlike traditional RecSys benchmarks that focus on item prediction accuracy, our benchmark evaluates how well LLMs can extract and verify user interests from engagement data. We propose two novel metric families: Interest Groundedness (IG), decomposed into precision and recall components to separately penalize hallucinated interest categories and reward coverage, and Interest Specificity (IS), which assesses the distinctiveness of verified LLM-predicted user profiles. We release a synthetic dataset constructed on real user interactions on a global short-form video platform. Our dataset contains both implicit and explicit engagement signals and rich textual descriptions. We validate our dataset fidelity against user surveys, and evaluate eight open-weight LLMs spanning 7B to 120B parameters. Our findings reveal performance bottlenecks in current LLMs, particularly their limited ability to accurately count and attribute engagement signals across heterogeneous interaction types. 

---
# Trojan-Speak: Bypassing Constitutional Classifiers with No Jailbreak Tax via Adversarial Finetuning 

**Authors**: Bilgehan Sel, Xuanli He, Alwin Peng, Ming Jin, Jerry Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.29038)  

**Abstract**: Fine-tuning APIs offered by major AI providers create new attack surfaces where adversaries can bypass safety measures through targeted fine-tuning. We introduce Trojan-Speak, an adversarial fine-tuning method that bypasses Anthropic's Constitutional Classifiers. Our approach uses curriculum learning combined with GRPO-based hybrid reinforcement learning to teach models a communication protocol that evades LLM-based content classification. Crucially, while prior adversarial fine-tuning approaches report more than 25% capability degradation on reasoning benchmarks, Trojan-Speak incurs less than 5% degradation while achieving 99+% classifier evasion for models with 14B+ parameters. We demonstrate that fine-tuned models can provide detailed responses to expert-level CBRN (Chemical, Biological, Radiological, and Nuclear) queries from Anthropic's Constitutional Classifiers bug-bounty program. Our findings reveal that LLM-based content classifiers alone are insufficient for preventing dangerous information disclosure when adversaries have fine-tuning access, and we show that activation-level probes can substantially improve robustness to such attacks. 

---
# OneComp: One-Line Revolution for Generative AI Model Compression 

**Authors**: Yuma Ichikawa, Keiji Kimura, Akihiro Yoshida, Yudai Fujimoto, Hiroki Tokura, Yamato Arai, Yoshiyuki Ishii, Yusei Kawakami, Genki Shikada, Achille Jacquemond, Yoshihiko Fujisawa, Katsuki Fujisawa, Takumi Honda, Akira Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2603.28845)  

**Abstract**: Deploying foundation models is increasingly constrained by memory footprint, latency, and hardware costs. Post-training compression can mitigate these bottlenecks by reducing the precision of model parameters without significantly degrading performance; however, its practical implementation remains challenging as practitioners navigate a fragmented landscape of quantization algorithms, precision budgets, data-driven calibration strategies, and hardware-dependent execution regimes. We present OneComp, an open-source compression framework that transforms this expert workflow into a reproducible, resource-adaptive pipeline. Given a model identifier and available hardware, OneComp automatically inspects the model, plans mixed-precision assignments, and executes progressive quantization stages, ranging from layer-wise compression to block-wise refinement and global refinement. A key architectural choice is treating the first quantized checkpoint as a deployable pivot, ensuring that each subsequent stage improves the same model and that quality increases as more compute is invested. By converting state-of-the-art compression research into an extensible, open-source, hardware-aware pipeline, OneComp bridges the gap between algorithmic innovation and production-grade model deployment. 

---
# StepCache: Step-Level Reuse with Lightweight Verification and Selective Patching for LLM Serving 

**Authors**: Azam Nouri  

**Link**: [PDF](https://arxiv.org/pdf/2603.28795)  

**Abstract**: We address LLM serving workloads where repeated requests share a common solution structure but differ in localized constraints, such as output schema, variable names, or numeric constants. Prior caching approaches typically reuse either full responses (semantic caching) or model-internal KV/prefix states, which are respectively brittle under partial changes or tightly coupled to specific backends. We present StepCache, a backend-agnostic step-level reuse layer that segments outputs into ordered steps, retrieves the best-matching cached request, verifies steps using lightweight task-aware checks, and regenerates only failing regions via selective patching. StepCache additionally supports strict structured-output enforcement for JSON, including single-step extraction, required-key constraints, and one-shot repair, as well as conservative skip-reuse fallbacks for semantic changes. For linear equations, StepCache promotes verification into correction via a bounded repair loop with a deterministic fallback that guarantees correctness when the backend model fails.
In a CPU-only perturbation-heavy micro-benchmark on math and JSON variants, averaged over three seeds, StepCache reduces mean latency from 2.13 s to 0.67 s, median latency from 2.42 s to 0.01 s, and p95 latency from 3.38 s to 3.30 s. It also reduces total token usage from 36.1k to 27.3k and improves end-to-end correctness from 72.5% to 100% under task-specific checks and a stitched-output integrity check. Across requests, 79.7% take the reuse-only fast path, 5.4% require patching, and 14.9% trigger skip-reuse. 

---
# UltRAG: a Universal Simple Scalable Recipe for Knowledge Graph RAG 

**Authors**: Dobrik Georgiev, Kheeran Naidu, Alberto Cattaneo, Federico Monti, Carlo Luschi, Daniel Justus  

**Link**: [PDF](https://arxiv.org/pdf/2603.28773)  

**Abstract**: Large language models (LLMs) frequently generate confident yet factually incorrect content when used for language generation (a phenomenon often known as hallucination). Retrieval augmented generation (RAG) tries to reduce factual errors by identifying information in a knowledge corpus and putting it in the context window of the model. While this approach is well-established for document-structured data, it is non-trivial to adapt it for Knowledge Graphs (KGs), especially for queries that require multi-node/multi-hop reasoning on graphs. We introduce ULTRAG, a general framework for retrieving information from Knowledge Graphs that shifts away from classical RAG. By endowing LLMs with off-the-shelf neural query executing modules, we highlight how readily available language models can achieve state-of-the-art results on Knowledge Graph Question Answering (KGQA) tasks without any retraining of the LLM or executor involved. In our experiments, ULTRAG achieves better performance when compared to state-of-the-art KG-RAG solutions, and it enables language models to interface with Wikidata-scale graphs (116M entities, 1.6B relations) at comparable or lower costs. 

---
# Spark-LLM-Eval: A Distributed Framework for Statistically Rigorous Large Language Model Evaluation 

**Authors**: Subhadip Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2603.28769)  

**Abstract**: Evaluating large language models at scale remains a practical bottleneck for many organizations. While existing evaluation frameworks work well for thousands of examples, they struggle when datasets grow to hundreds of thousands or millions of samples. This scale is common when assessing model behavior across diverse domains or conducting comprehensive regression testing. We present Spark-LLM-Eval, a distributed evaluation framework built natively on Apache Spark. The system treats evaluation as a data-parallel problem, partitioningexamplesacrossexecutorsandaggregatingresultswithproperstatistical accounting. Beyond raw throughput, we emphasize statistical rigor: every reported metric includes bootstrap confidence intervals, and model comparisons come with appropriate significance tests (paired t-tests, McNemar's test, or Wilcoxon signed-rank, depending on the metric type). The framework also addresses the cost problem inherent in LLM evaluation through content-addressable response caching backed by Delta Lake, which allows iterating on metric definitions without re-running inference. We describe the system architecture, the statistical methodology, and report benchmark results showing linear scaling with cluster size. The framework and all evaluation code are available as open source. 

---
