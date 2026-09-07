# WearableQA: A Benchmark for Health Reasoning over Real-World Wearable Data 

**Authors**: Ji Soo Lee, Xilun Chen, Pierce Chuang, Ashish Shenoy, Jason Wei, Dohwan Ko, Hyunwoo J. Kim, Benoit Corda  

**Link**: [PDF](https://arxiv.org/pdf/2609.05405)  

**Abstract**: Recent advances in wearable sensing enable continuous monitoring of physiological and behavioral signals, yet existing benchmarks rarely evaluate whether AI systems can reason over a real user's longitudinal wearable record. We introduce WearableQA, a benchmark comprising 4,084 10-option multiple-choice questions constructed from the wearable time series, blood biomarkers, and demographics of 200 real users, each with up to 500 days of daily measurements. WearableQA preserves authentic wearable distributions that include device noise and inter-individual variability. To evaluate distinct reasoning capabilities, we introduce 16 question types organized along two complementary axes: data versus health reasoning, which distinguishes computation over longitudinal measurements from physiological interpretation; and single- versus cross-signal reasoning, which separates reasoning about individual signals from the integration of multiple signals. To construct reliable questions at scale, we adopt a dual-grounding framework that combines literature-grounded physiological findings with statistically validated population-grounded physiological patterns. This enables the capture of meaningful relationships observed in real-world wearable data. Evaluation of 14 proprietary and open-source LLMs demonstrates that WearableQA effectively differentiates model capabilities, with performance ranging from 19.6% to 72.9% against a 10% chance baseline. Moreover, WearableQA remains far from solved: most models achieve accuracies below 60%. Overall, WearableQA provides a realistic and diagnostic benchmark for evaluating LLM reasoning over real-world wearable data. 

---
# LexFlip: A Dissociation Diagnostic for Legal Meaning Preservation Metrics 

**Authors**: Gaurab Baral  

**Link**: [PDF](https://arxiv.org/pdf/2609.05296)  

**Abstract**: Does a simplified legal clause still say what the original said? The checks in current use cannot establish that it does: requiring an identical pair to score highest and an unrelated pair lowest moves lexical overlap and legal force together, so any monotone function of token overlap satisfies both. Our remedy is a dissociation, an item holding surface form fixed while legal force moves. We release LexFlip, 373 minimal perturbations of Quebec statutory French that reverse legal force while preserving 0.93 of the tokens, with a harness scoring metrics, regressors and prompted judges alike. The seven embedding and BERTScore metrics we test spend only 0.022 to 0.039 of their identical-to-unrelated range on such an edit, against 0.670 for bidirectional NLI, the one family the identical-pair check would disqualify. On FrJudge, against a measured human ceiling of r=0.597, a bare length feature outscores every semantic metric and has the lowest margin we measure. 

---
# Self-Supervised Lexical Representation Learning for Fast, Large-Scale Phylogenetic Inference 

**Authors**: Tim Wientzek  

**Link**: [PDF](https://arxiv.org/pdf/2609.05262)  

**Abstract**: Computational phylogenetics has become an essential tool in historical linguistics, yet its application at a global scale remains constrained by two factors: the labor-intensive manual annotation of cognacy judgments required for character-based methods and the substantial computational cost of inference on large datasets. This paper introduces a fully self-supervised contrastive learning framework that learns lexical representations directly from raw IPA-transcribed wordlists, without requiring cognacy annotations, alignments, or additional expert input. The model employs a dual contrastive objective: a word-level loss that organizes phonetically similar forms into a coherent space, and an auxiliary language-level loss that encourages the lexical space to reflect broader phonological properties of languages. From the resulting word representations, pairwise language distances are derived and used to infer a global phylogenetic tree of 3,399 language varieties. The inferred tree achieves a generalized quartet distance (GQD) to the Glottolog reference tree competitive with multiple baselines, while requiring only minutes of computation on a standard notebook GPU. Furthermore, the same representations capture diachronic concept stability: variance in pairwise distances across languages yields stability rankings that correlate significantly with established rankings. Ablation studies confirm that both the language-level objective and the use of phonetic feature vectors improved the inferred trees topology with regards to GQD. The framework thus provides a computationally efficient and fully automatic alternative for large-scale phylogenetic inference and offers a unified representation supporting downstream analyses at both the language and concept level. 

---
# A Verifier-Guided Explainable Reasoning Framework with Gold-Anchored QLoRA, Task-Aware Mixture-of-Experts, and Group-Relative RLVR 

**Authors**: Thi Kim Trang Vo, Nam Tien Le, Thi Kim Nguyet Vo, Minh Khang Tran, Duy Phuong Tran  

**Link**: [PDF](https://arxiv.org/pdf/2609.05221)  

**Abstract**: Large language models (LLMs) show strong reasoning ability, but their explanations can remain inconsistent, weakly grounded, or difficult to verify. We propose a verifier-guided explainable reasoning framework for transparent educational question answering that combines gold-anchored QLoRA, task-aware symbolic routing, and group-relative RLVR. Qwen2.5-3B-Instruct is first adapted with field-weighted QLoRA supervision anchored to authoritative answers. A lightweight router then assigns logic problems to a FOL/Z3 verifier and physics problems to a formula- and unit aware symbolic solver. Verifier feedback is further used to support candidate evaluation, self-revision, and reward construction during RLVR. Candidate responses are evaluated along three complementary dimensions: P1 for answer correctness, P2 for evidence or unit consistency, and P3 for reasoning depth and explainability. At inference, gold-free self-consistency aggregates multiple candidate responses before an optional question-only physics verifier performs conservative system-level correction. On 438 held-out examples, RLVR increases P3 from 50.68% to 72.20%, while hybrid P1 remains approximately stable at 55.94%. Self-consistency improves model only P1 from 48.86% to 50.23%, with symbolic verification providing the remaining hybrid gain. These results indicate that RLVR primarily strengthens explicit reasoning structure, while symbolic verification complements the neural policy by improving answer reliability at the system level. 

---
# Can Large Language Models Anticipate Behavioral Responses to Social Policies? A Case of Pension Enrollment Prediction among China's Flexible Workers 

**Authors**: Yumiao Li, Peixin Liu, Donglin Di, Chen Li, Runhuan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2609.05189)  

**Abstract**: Assessing the impacts of social policy changes is a widely acknowledged challenge for policymakers. Econometric methods can be unreliable when extrapolating to hypothetical scenarios, while field pilot programs are highly costly. In this paper, we propose using large language models (LLMs) as policy-assessment tools adapted from general-purpose models. We present FlexPension-LLM, the first domain-specialized large language model for a hierarchical pension-enrollment prediction task among flexible workers in China, and introduce DKI-RDistill, which injects policy-grounded cues into the prompt, including Probit-derived marginal effects and hukou-province pension rules. The method then uses LoRA/SFT to distill rationale-augmented supervision into an open-weight MoE student, with teacher errors corrected by regenerating those cases under ground-truth labels. On a CHFS 2019 blind split, FlexPension-LLM achieves 0.9316 Composite F1, surpassing its Claude Sonnet 4.5 teacher and 15 of 17 baselines, and is statistically indistinguishable from Claude Opus 4.6. Across four external surveys, it averages 0.7549 Composite F1 and shows the narrowest performance range among the strongest systems. Component analysis shows that gains come mainly from policy-grounded cue injection and error-filtered supervision, while rationales provide decision traces that can be checked against policy rules. 

---
# Compression Beyond the Uncompressed: A Two-Stage Training Recipe for Soft Context Compression in RAG 

**Authors**: Shuyu Guo, Shuo Zhang, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2609.05152)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances language models with external knowledge, but the lengthy retrieved context inflates the input and degrades inference efficiency. Soft context compression encodes each document into a substantially shorter embedding sequence. However, most existing approaches are trained by distilling outputs from uncompressed RAG systems, inherently limiting their performance relative to the original model. To address this limitation, we propose DEX-Comp, a two-stage training recipe: Pure Distillation warm-starts the compression model on the uncompressed RAG's correct responses only, and Hard Exploration then runs reinforcement learning solely on queries the uncompressed RAG fails, forcing the model to explore computation patterns better suited to compressed representations. On five open-domain QA benchmarks at retrieval depths from top-5 to top-30, DEX-Comp compresses retrieved contexts by $16\times$ and accelerates inference by $4\times$--$24\times$, while achieving performance comparable to or exceeding the uncompressed RAG baseline across retrieval depths. Ablations and evaluations across diverse datasets and backbones further confirm the contribution of each stage and the generalization of our approach. 

---
# From Vision to Language: Investigating Causal Information Flow in Multimodal Decision-Making 

**Authors**: Davide Testa, Hugh Mee Wong, Alessandro Lenci, Bernardo Magnini, Albert Gatt  

**Link**: [PDF](https://arxiv.org/pdf/2609.05149)  

**Abstract**: Vision-Language Models are commonly evaluated through their final predictions, but understanding whether these decisions are grounded in visual evidence requires tracing how visual information contributes to language-based decisions. With this purpose in mind, we investigate cross-modal information flow in a video-based generative multiple-choice-like setting by applying a layer-wise causal intervention on video-text attention pathways. We target spatial, causal, and temporal visual reasoning. Our results show that visual information is mainly integrated while the model processes the candidate answer options, which serve as the primary textual grounding sites for the final decision. We further show that nouns play an important role as semantic anchors during multimodal enrichment, while verbs are more relevant when temporal relations are processed. Finally, we identify a distinct pattern in temporal reasoning, suggesting that VLMs struggle to reconstruct sequential information across video frames, but we remark that such fragility may also reflect linguistic biases associated with specific temporal expressions used for defining the relation between events within a scene. 

---
# A Human-in-the-Loop Framework for AI-Assisted Scoring in Large-Scale Writing Assessment 

**Authors**: María Eugenia Curi, Germán Capdehourat, Isabel Amigo, Magdalena Romano, Rosana Serra, Adrián Silveira, Andrés Peri  

**Link**: [PDF](https://arxiv.org/pdf/2609.05143)  

**Abstract**: The integration of artificial intelligence (AI), particularly large language models (LLMs), into educational assessment has opened new opportunities to enhance the efficiency and scalability of grading processes. This study presents the design and validation of an AI-assisted scoring framework for written responses in a large-scale national assessment. The proposed approach focuses on short written texts of approximately 150-200 words and incorporates a human-in-the-loop strategy to preserve assessment quality while reducing manual workload. The study is grounded in a real operational context, using data from two recent editions of a nationwide test, each comprising approximately 5,000 student responses. We analyze the alignment between AI-generated scores and human raters across multiple rubric dimensions, as well as the impact of the proposed decision flow on pass/fail outcomes. Results show moderate to high agreement between the model and human evaluations in most dimensions, supporting the feasibility of AI assistance in this setting. Moreover, the proposed correction workflow identifies cases where human review is most valuable, enabling a more efficient allocation of expert effort. The findings suggest that AI-assisted scoring can be safely integrated into large-scale assessment processes only when combined with carefully designed human oversight. The paper concludes by discussing practical implications for deployment in national assessment systems and outlining future research directions, including longitudinal monitoring of model-human alignment and the analysis of potential cognitive bias introduced by AI-supported review workflows. 

---
# NS-ST-GraphRAG: Neuro-Symbolic Spatio-Temporal GraphRAG for Literary Knowledge Processing 

**Authors**: Zheng Kui Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.05139)  

**Abstract**: Long-form literary narratives pose a distinctive information-processing challenge for retrieval-augmented generation: relevant evidence is distributed across chapters, relations evolve over narrative time, and correct answers may depend jointly on temporal, spatial, and relational constraints. We propose NS-ST-GraphRAG, a neuro-symbolic spatio-temporal GraphRAG framework that integrates ontology-guided extraction, deterministic constraint checking, dual temporal coordinates, spatial scene attributes, and dynamic sub-graph retrieval. Instead of retrieving from a single corpus-level graph, the framework selects the graph state valid for the temporal and spatial scope of a query and grounds generated answers in traceable evidence. We further introduce Red-Chamber-QA, to our knowledge the first open multi-hop question-answering benchmark for classical Chinese literature, with time-, space-, and general-question categories, per-part evidence spans, and deterministic shortcut controls. On a 120-question held-out split, NS-ST-GraphRAG achieves mechanical answer reproduction of 0.733 versus 0.675 for the frozen window baseline and 0.083 for a closed-book model (McNemar exact p = 0.092, directionally favorable but not significant); semantic-judge accuracy is 0.866 versus 0.850. The pre-specified constrained-category condition of H2 is not supported by the delivered comparison. These results show how temporal graph representation, constrained extraction, and auditable evaluation integrate into a unified framework for verifiable knowledge processing over long-form narrative. 

---
# Improving Language Identification for Code-Switched Utterances with Integer Linear Programming 

**Authors**: Joanna Radoła, Josep Maria Crego, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2609.05099)  

**Abstract**: Automatic identification of code-switched (CS) utterances remains a challenge for language identification (LID) systems, causing such texts to be underrepresented in the training data of Large Language Models. In this paper, we revisit MaskLID, a state-of-the art approach for CS identification, which requires no training and detects arbitrary language combinations. We make three main contributions: (a) we reveal, and address, a major issue of MaskLID: its overreliance on word-level language association scores; (b) we reformulate the underlying optimization algorithm as an Integer Linear Program, enabling us to experiment with a large set of clear and interpretable constraints; (c) each of these improvements vastly improves the baseline system, as we illustrate in experiments involving 10~diverse languages, where we observe a strong boost in performance on CS benchmarks. We release our code and data for reproducibility. 

---
# Influence Score and Transformers interpretability: Measure of the Effective Impact of Attention Heads at inference time 

**Authors**: Lisa Bouger, Yannick Teglia, Philippe Loubet Moundi  

**Link**: [PDF](https://arxiv.org/pdf/2609.05074)  

**Abstract**: We propose an influence score to quantify the contribution of attention heads to classification decisions in Transformer-based models designed for prompt injection detection. The score combines directional influence on the logits with structural contribution within the residual stream, enabling a multi-scale analysis at the head, layer, and network levels. Applied to a DeBERTa model specialized for prompt injection detection, our framework reveals distinct decision behaviours between correct and erroneous predictions. Our method provides an effective compromise between fine-grained circuit analysis and global output-based methods, and offers a systematic way to study decision mechanisms in Transformer classifiers. 

---
# A Structured Debate-Mixture-of-Agents Framework for Complex Clinical Diagnostic Decision Support 

**Authors**: Chang Xia, Leilei Ouyang, Huimin Wang, Yong Zhao, Kang Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.05069)  

**Abstract**: Large language models (LLMs) show potential for medical tasks, but their single-turn question-answer format does not reflect how clinical diagnosis is performed in practice. As a result, they remain limited in complex diagnostic settings. We developed Debate-Mixture-of-Agents (DMoA), a novel multi-agent framework that structures role-based interaction to support iterative diagnostic reasoning. Base models and DMoA were evaluated on 297 rare disease cases and 1,719 challenging cases. Across both datasets, DMoA improved most likely diagnosis accuracy by 10.21 percentage points and safety rate by 11.36 percentage points over GPT-4o baseline. Ablation experiments showed that the gains were not simply due to the use of more models or longer outputs, but also reflected the contribution of the structured workflow. Further analyses examined how framework design, base model choice, and token budget affected performance. DMoA performed better with a 4*2 structure, stronger base models, and a larger token budget. These findings demonstrate the potential of DMoA for clinical tasks and suggest further investigation of multi-agent frameworks. 

---
# EuroAlpaca: Task-Preserving Localisation of Instruction Data for European Languages 

**Authors**: Aleix Sant, Jordi Luque, Carlos Escolano  

**Link**: [PDF](https://arxiv.org/pdf/2609.05043)  

**Abstract**: Machine translation (MT) offers a scalable way to extend English instruction-tuning data to multiple languages, but it can distort task-critical constraints and required outputs, creating corrupted training examples and degrading models trained on such data. We introduce EuroAlpaca, a task-preserving localisation pipeline and near-parallel resource covering 50 European languages and regional varieties, together with European-IFEval, a multilingual benchmark for verifiable instruction following. Depending on the example, our pipeline applies field-wise MT while preserving task-critical content or reconstructs a task-equivalent target-language instance, followed by validation of cross-field coherence and target-language consistency. Across LoRA experiments with four LLMs, training on directly translated data improves ROUGE-L and F-BERT on the Aya Evaluation Suite, but reduces accuracy on European-IFEval by 29.8% relative to the unadapted baseline. In contrast, adaptation with EuroAlpaca improves accuracy by 12.9% over the same baseline, reversing the degradation caused by direct MT, while also achieving the highest ROUGE-L and F-BERT scores on Aya. These results show that preserving task semantics is essential for multilingual instruction tuning. 

---
# How do LLMs Evaluate Perceived Moral Agency? Investigating Moral Decision-Making in Human-Artificial Agents Interactions 

**Authors**: Fernanda Mansilla, Aloysius Tok, Bahia Guellaï, Farah Benamara, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.05037)  

**Abstract**: As LLMs take on roles requiring moral advice, understanding how they attribute moral agency becomes critical. Humans possess moral agency, the capacity to make ethically guided decisions and bear responsibility for their consequences, a well-established construct in moral psychology. Yet as artificial agents (AAs) such as robots, drones, and disembodied AI systems become increasingly embedded in smart city environments, the question of whether and how moral agency is attributed to them takes on new urgency. This paper presents, to the best of our knowledge, the first empirical study comparing how humans and LLMs evaluate perceived moral agency (PMA) across human and autonomous artificial agents varying in embodiment, situated in plausible smart city scenarios. Using an adaptation of a validated PMA scale, we applied a protocol to 190 human participants as well as various LLMs. Our evaluation reveals higher perceptions of moral agency in humans than in AAs. However, when facing moral dilemmas in concrete scenarios, LLMs reason outward from the situation, prioritizing harm severity and contextual urgency over any stable assessment of the agent itself, amplifying a context-sensitivity also present in human raters. These findings are particularly relevant as LLMs become increasingly involved in everyday moral decisions. 

---
# Leveraging Low-Level Symbolic Competences for Unsupervised Grounding in Hallucination Detection 

**Authors**: Renato Vukovic, Hsien-chin Lin, Carel van Niekerk, Benjamin Ruppik, Michael Heck, Shutong Feng, Nurul Lubis, Milica Gasic  

**Link**: [PDF](https://arxiv.org/pdf/2609.05025)  

**Abstract**: Hallucination-where a language model generates outputs that are factually incorrect or unsupported by the source-is a major challenge for both prompted and fine-tuned language models. Detecting hallucinations is difficult due to the opaque reasoning processes of LLMs, which often provide little insight into why a model's output may be inaccurate.
In this work, we investigate whether an LLM can use an alternative, low level, symbolic competence such as SQL for unsupervised hallucination detection in some high level task. For this, we make an LLM build an SQL database from reference documents. This SQL database is then used for reasoning over the reference and the sampled response in a hallucination detection pipeline that is grounded in the database, thereby providing a neurosymbolic checkup.
On RAGTruth and DiaHalu hallucination detection datasets, we find that our approach improves on direct prediction and competes with state-of-the-art hallucination detection methods, while not requiring domain-specific fine-tuning. Instead it relies on a low-level general competence already present in LLMs. This warrants further investigation of low-level LLM competences in neurosymbolic approaches. 

---
# MoirfEolas and CríochScore: Developing Resources for and the Evaluation of Tokenization Alignment with Irish Morphology 

**Authors**: Jane Adkins, Abigail Walsh, Brian Davis, Elaine Uí Dhonnchadha  

**Link**: [PDF](https://arxiv.org/pdf/2609.05022)  

**Abstract**: This paper presents new tokenization resources for Irish and evaluation measures of alignment with the morphological boundaries of the language. We present MoirfEolas, a dataset of over 35,000 Irish words mapped to their respective eclipses, prefixes and suffixes as well as an evaluation metric CríochScore, that evaluates the alignment of tokenizations with the morphological boundaries present in MoirfEolas. We evaluate common tokenization algorithms using CríochScore as well as intrinsic metrics present in the tokenization literature. We find that the Unigram Language Model aligns with Irish morphology more often than the other algorithms evaluated. We also find trade-offs between morphological-alignment of tokenization with both compression as well as vocabulary efficiency, providing practical insights for Irish natural language processing development. This dataset contributes towards combating the Irish language's low-resource status; moreover, the construction process reported in this paper can be emulated by other languages to create specialised morphological resources. 

---
# BIT.UA at BioASQ 14B: Modular Retrieval with pg_textsearch and Qdrant, and Agent-Based Answer Generation 

**Authors**: André Ribeiro, Rúben Garrido, Alexander Christiansen, Richard A. A. Jonker, Sérgio Matos  

**Link**: [PDF](https://arxiv.org/pdf/2609.04999)  

**Abstract**: This paper describes the participation of the this http URL team from the University of Aveiro in the 14th edition of the BioASQ Task B challenge on biomedical question answering. Building on our previous submissions, we introduced a substantially refactored and modular codebase, and made significant changes to both the retrieval and generation components of the pipeline. For Phase~A document retrieval, we replaced the PyTerrier PISA index with PostgreSQL-based pg\_textsearch for BM25 retrieval and adopted Qdrant for dense embedding indexing, enabling more efficient storage and GPU-accelerated similarity search. We explored HyDE-based query expansion alongside a Context-1 retrieval strategy. A new reranker training pipeline was developed, incorporating dense retrieval for negative sampling. For Phases A+ and B answer generation, we introduced an LLM-as-a-judge framework and a novel agent quorum mechanism, where multiple agents with diverse prompts debate and iteratively converge on a consensus answer using adaptive document retention. We also participated in the snippets generation subtask for the first time. Our systems achieved competitive results across all batches, with Phase~A systems achieving MAP ranks of 5 (Batch~1,3). We discuss the impact of these architectural changes, lessons learned, and outline directions for future work including SPLADE and ColBERT integration. All code is openly available: this https URL. 

---
# Discourse Dependency: A Continuous Criterion for Translation Difficulty 

**Authors**: Ahrii Kim, Chanjun Park, Seong-heum Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.04959)  

**Abstract**: Recent calls for harder machine translation benchmarks have not clarified what difficulty should mean. We argue that one meaningful and currently unmeasured axis is referential reach, the distance a segment must look back into its document to resolve the entities and pronouns it contains. We formalize this as discourse dependency (DDP), a metric-free, source-side measure computed from named entity re-mentions and pronominal coreference. Validated against gold coreference, DDP errs one-sidedly in 99.2% of segments, so a high-DDP segment is certified to require long-range context. Applying DDP to WMT24++ and WMT25 shows that both are heavily skewed toward low-DDP segments, which domain labels do not distinguish. Building on DDP, we compare five context injection strategies in an English-Korean post-editing setup, varying context size and selection. As DDP grows, no strategy keeps pace with human post-editing. On segments with DDP >= 15 raters prefer human translations, while automatic metrics register no difference. As frontier systems saturate aggregate scores, DDP shifts evaluation from how well models score to how far they can reach. 

---
# RefactorPlatform: An Open-Source Harness for Controlled Evaluation of Repository-Scale Refactoring Agents 

**Authors**: Aziz Ben Amor, Drish Mali, Mann Acharya, Vijayasri Iyer, Sébastien Bratières  

**Link**: [PDF](https://arxiv.org/pdf/2609.04898)  

**Abstract**: Repository-scale refactoring requires coding agents to propagate a single change across many interdependent files without altering program behavior, yet to our knowledge no existing harness isolates the design choices that determine agent success on this task. We present RefactorPlatform, an open-source evaluation harness that holds the environment fixed and varies each design axis explicitly: model backbone (via OpenRouter and GitHub Copilot CLI), execution regime (baseline, retrieval-augmented, and multi-agent), and prompt specificity. Each run executes in an isolated workspace with live terminal streaming, per-task logging of tokens, diffs, and transcripts, AST-based verification, and exportable telemetry for audit and reproduction. Demonstrating the platform on 100 multi-file RefactorBench tasks across four model families, we illustrate the analyses it supports: AST-aware chunking outperforms naive token-window chunking by 25-30% across prompt modes, whereas naive retrieval falls below the retrieval-free baseline; a lean retrieval-augmented single agent (86%) beats the sub-agent configuration we evaluated (66%) on matched tasks with no task passing under delegation that fails under retrieval; and retrieval's accuracy gains absorb its token overhead, leaving cost per successful refactoring unchanged. RefactorPlatform is open-sourced to make refactoring-agent evaluation reproducible and auditable. 

---
# Cache-Aware Joint Router Adaptation for Memory-Efficient MoE Inference 

**Authors**: Zhenhe Wu, Yaping Jin, Qinghua Xing, Hang Zhou, Wei He, Xianjie Wu, Xianfu Cheng, Jian Yang, Hanting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.04895)  

**Abstract**: Mixture-of-Experts (MoE) models activate only a small subset of experts per token, but the full expert set often exceeds GPU memory, causing repeated weight transfers during decoding. We formulate expert-cache management as a model-side algorithmic problem and propose a cache-aware post-training framework that jointly adapts the MoE backbone and lightweight auxiliary cache routers while preserving the native Top-K expert-selection rule at inference. Its update-only mode, Temporal Router, predicts same-layer reuse and retains experts for future tokens without proactive loading. The full Spatio-Temporal Router adds a Spatio Router that uses the causal predecessor's hidden state to refine the temporal cache before target-layer access. We evaluate both modes on Qwen3 and GPT-OSS across GSM8K, MATH, and CommonsenseQA. Temporal Router consistently improves cache hit rate and reduces expert-weight traffic over matched LM-only baselines. On Qwen3, Spatio-Temporal Router achieves the best load-adjusted efficiency across three tasks, improving adjusted hit rate by 1.15--18.03 points and reducing traffic by 4.6--53.3% relative to the strongest evaluated prefetching baseline; results on GPT-OSS are competitive but task-dependent. An auxiliary-only ablation preserves baseline accuracy but yields modest cache gains, whereas joint post-training produces larger improvements. Sensitivity analyses show that cache capacity controls transfer demand, while the refinement budget governs the trade-off between pre-access coverage and proactive traffic. 

---
# CC-Mediation: Evaluating Large Language Models for Cross-Cultural Conflict Mediation 

**Authors**: Suhyun Lee, Wenxuan Zhang, W. Quin Yow, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2609.04855)  

**Abstract**: Cross-cultural mediation by large language models (LLMs) requires deciding both when to intervene and how to respond in culturally grounded conflicts. Progress on this problem has been limited by the lack of (1) mediation datasets with measurable downstream effects and (2) principled metrics for evaluating intercultural stance change. To address these gaps, we introduce CC-Mediation, a cross-cultural mediation benchmark of $1{,}661$ ten-turn dialogues grounded in the Developmental Model of Intercultural Sensitivity (DMIS), containing culturally grounded conflicts, mediation interventions, and post-intervention trajectories. We further propose two DMIS-based evaluation metrics: Trajectory AUC, which measures the persistence of intercultural improvement over time, and a signed Wasserstein-1 distance, which measures the magnitude and direction of shifts in intercultural stance. Both metrics show strong agreement with human judgment of DMIS-grounded stance shift. Using CC-Mediation, we find that current LLMs have limitations on both axes: intervention timing (when) failure stems from a positional prior that ignores dialogue content, while mediation strategy (how) failure arises from a late-layer elicitation collapse rather than a knowledge deficit. 

---
# MMTClinic: Multimodal, Multilingual Time Series Question Answering and Reasoning Benchmark for Clinical Domain 

**Authors**: Sourav Malakar, Harshit Nigam, Akash Ghosh, Sriparna Saha, Amlan Chakrabarti, Saptarsi Goswami, Priti Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.04842)  

**Abstract**: Time-series data in clinical settings is crucial for capturing dynamic changes in a patient's health over time, enabling timely diagnosis, personalized treatment, and early detection of critical events. However, the development of clinically reliable and linguistically inclusive medical AI systems remains a significant challenge, primarily due to the lack of multimodal, multilingual, and time-series-grounded benchmarks that reflect the complexity of real-world clinical scenarios. To fill this gap, we present MMTClinic, a benchmark designed to evaluate large language models (LLMs) on complex reasoning and question-answering tasks involving clinical time-series. MMTClinic combines text, medical images, and multivariate physiological signals and includes 30,000 QA pairs (15,000 multiple choice questions (MCQs) and 15,000 open-ended questions) across five languages: English, Hindi, Bengali, Marathi, and Tamil. These questions cover three important clinical tasks---mortality prediction, heart rate forecasting, and SOFA score estimation. We evaluate 13 state-of-the-art LLMs in zero-shot, few-shot, and chain-of-thought settings. Our evaluation reveals notable differences in model performance across tasks, languages, and modalities, highlighting current limitations in clinical reasoning capabilities. MMTClinic provides a valuable resource for advancing multilingual, multimodal, and time-series-aware medical AI research. The dataset will be made publicly available on successful acceptance of the work. 

---
# MABPD: Multi-Agent Bias Probing & Detection via Structured Argument Debate 

**Authors**: Garvit Joshi, Stavya Dhyani, Jasmine, Arun Chauhan  

**Link**: [PDF](https://arxiv.org/pdf/2609.04841)  

**Abstract**: Media bias in news articles operates through subtle linguistic cues---loaded language, selective framing, and strategic omission---that resist single-model detection and have traditionally required large annotated corpora for supervised training. We ask whether structured multi-agent deliberation can serve as a principled, training-free alternative to supervised classification for this task. We introduce MABPD (Multi-Agent Bias Probing & Detection), a pipeline in which three specialized LLM agents analyze an article from complementary perspectives and resolve disagreements through a Structured Argument Debate (SAD) protocol. SAD implements a domain-motivated asymmetric burden of proof---biased claims without grounded textual evidence carry zero weight---combined with role-weighted voting and post-consensus verification, replacing task-specific supervised decision boundaries with explicit deliberative structure. Ablation confirms that this structured deliberation, not mere agent parallelism, drives performance: removing the debate module reduces F1 by up to 10.6 points. On the BABE benchmark (4,121 expert-annotated sentences), MABPD achieves 83.4% macro F1 on the held-out test split---within 0.7 percentage points (pp) of the supervised SOTA (MAGPIE, 84.1% macro F1; Horych et al., 2024)---without any task-specific training or threshold tuning on annotated data. Cross-dataset evaluation on the SemEval 2019 HyperPartisan corpus (644 articles) yields 75.0% zero-shot accuracy, within 7.2 pp of the supervised SOTA accuracy (82.2%; Kiesel et al. 2019), confirming transfer across annotation regimes. We release the full pipeline and evaluation code. 

---
# On Epistemic Diversity in Large Language Models 

**Authors**: Elisabeth Kirsten, Nicole Krämer, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2609.04835)  

**Abstract**: Large language models (LLMs) are increasingly used not only to retrieve information, but to answer questions, explain, teach, and support inquiry. In such settings, evaluation cannot be exhausted by accuracy or alignment alone. A system may give a correct answer while still narrowing users' access %to knowledge. to alternative valid answers, explanations, or reasoning routes. Drawing on the broader notion of epistemic diversity in philosophy and social epistemology, we formalize it in the context of LLMs as the range of valid answers, explanations, and reasoning routes that an LLM exposes to users. We argue that epistemic diversity is a useful evaluation dimension for settings where LLMs are used to support knowledge-intensive tasks. We propose a preliminary framework for conceptualizing and measuring epistemic diversity in LLMs, and operationalize it in two domains. We find that frontier LLMs often exhibit epistemic narrowness, repeatedly collapsing large valid answer spaces onto small canonical subsets. These findings suggest that LLM evaluation should move beyond accuracy-oriented paradigms and treat epistemic diversity as an important dimension of model capability. 

---
# Generating Constructive Feedback on Stories via Reinforcement Learning 

**Authors**: Maja Stahl, Timon Ziegenbein, Henning Wachsmuth  

**Link**: [PDF](https://arxiv.org/pdf/2609.04824)  

**Abstract**: Constructive feedback is crucial for creative writers to refine their storytelling abilities. Since receiving feedback from human experts is often costly and time-intensive, large language models (LLMs) offer a scalable and efficient alternative as automatic writing assistants. Despite their potential, research indicates that LLM-generated feedback is often generic, lacks actionability, and fails to identify which writing issue is most critical. To address these limitations, we present a reinforcement learning approach that steers LLMs to generate constructive feedback without the need for ground-truth feedback. We train our model using group relative policy optimization (GRPO) with a novel multi-component reward function aiming at constructiveness: it prioritizes feedback that is uniquely tailored to the story, helps to improve story quality, and addresses the most critical writing issue. In automatic and human evaluation across three story corpora, our approach outperforms state-of-the-art LLMs (including Gemini) and competitive baselines. We find that providing actionable suggestions is the main driver of feedback constructiveness. 

---
# Reinforcement Learning for improving Large Language Models' Catalan text simplification capabilities 

**Authors**: Arnau Ayguadé Domingo, Stefan Bott, Horacio Saggion  

**Link**: [PDF](https://arxiv.org/pdf/2609.04823)  

**Abstract**: Although automatic text simplification (ATS) is critical for accessibility, its progress has not matched the rapid evolution of broader natural language processing techniques. This paper investigates the application of reinforcement learning (RL) to improve the quality of ATS for low-resource languages using Large Language Models (LLMs). The paper introduces a novel reward function, designed to guide LLMs toward a targeted simplification style with Group Relative Policy Optimization (GRPO), that combines the SARI metric with specific penalty components. The effectiveness of GRPO with this reward function is motivated and demonstrated by post-training IberianLLM-7B-Instruct on the ASSET dataset. After post-training on the English ASSET, the model's ATS performance improves on two curated Catalan benchmarks while also successfully suppressing previously observed negative behaviors. Cross-lingual transfer learning is explored by translating ASSET into Catalan and Spanish and post-training the model on each version, but these fail to show a significant improvement on the out-of-domain benchmark. 

---
# A Systematic Comparison of Multilingual Interpretability Methods Reveals Anisotropy-Driven Failures 

**Authors**: Oskar Holmström, Marcel Bollmann, Marco Kuhlmann  

**Link**: [PDF](https://arxiv.org/pdf/2609.04819)  

**Abstract**: Multilingual language models develop shared cross-lingual representations, and various interpretability methods claim to quantify this sharing. These methods have been developed largely in isolation, and when they disagree, it is unclear whether the disagreement reflects a property of the model or an artifact of the measurement. We compare four sharing metrics (CKA, ANC, GMM dominance per token, and ILO) across 21 base models from five families (125M-14B parameters) and correlate each with cross-lingual transfer on five downstream tasks. We find that the metrics differ in their quantification of cross-lingual sharing in these models and suggest that the disagreement traces to anisotropy, the tendency of representations to cluster in a narrow cone of the embedding space. Only ILO's correlation with cross-lingual transfer (Spearman's $\rho = 0.90$) survives controls for model size, family, and per-task variation. We therefore recommend ILO as the primary sharing metric, to be reported alongside anisotropy diagnostics. 

---
# Recurrence Is Not Enough: Causally Validating Multilingual SAE Translation Features in Gemma 2 and 3 

**Authors**: Giang Son Nguyen, Nhi Ngoc-Yen Nguyen, Wray Buntine, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2609.04808)  

**Abstract**: Sparse autoencoder (SAE) features are increasingly used to explain and steer language-model behavior, but it remains unclear whether a feature found in one language context plays the same causal role when processing prompts in another language. We study this question using translation-initiation features (Wu et al., 2026). We reproduce the SAE feature discovery method from Wu et al. in Gemma 2 and extend it to multilingual settings that vary prompt language, source language, and target language. We then test whether features that recur across settings affect translation behavior by amplifying or ablating their activations during inference. We also examine whether the method can be applied to Gemma 3.
In both models, we observe an identical finding: although we can find more than 20 features that activate frequently across all discovery settings, causal validation shows that nearly all have small or inconsistent effects. In contrast, one feature -- Gemma 2's (L10, 5717) and Gemma 3's (L20, 2456) -- consistently improves COMET scores when amplified and degrades them when ablated across 23 language settings. These results show that feature recurrence can overstate cross-lingual transfer, while identifying a language-agnostic translation-initiation direction in Gemma 2 and Gemma 3. 

---
# Can Activation Steering Capture Multidimensional Authorship Style? 

**Authors**: Hieu Tran, Calvin Bao, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2609.04792)  

**Abstract**: Activation steering has shown promise for controlling LLM generation along well-defined attributes, but it remains unclear whether it can handle the multidimensional and hard-to-define nature of authorship style. We ask whether structured contrastive prompting along rhetorically-motivated dimensions can construct rich style representations directly in activation space, bypassing the need for natural language style descriptors or dedicated training. We find that the resulting directions share a common authorship backbone while conflicting on aspect-specific residuals that carry genuine stylistic signal, explaining why naive aggregation fails. We operationalize this in Aspect-Aware Activation Steering (A3S), a training-free framework that merges per-aspect contrastive directions with interference-aware aggregation and tunes steering strength per instance. A3S improves authorship style transfer where it is genuinely multi-aspect, outperforms a trained baseline in preference evaluations on out-of-domain benchmarks, and keeps target-exemplar overlap consistently low. 

---
# Vectorizing Classical Tamil: Representation Learning for Verse-Commentary Pairs 

**Authors**: Amrit Gopinath, Sangeetha Sivanesan  

**Link**: [PDF](https://arxiv.org/pdf/2609.04755)  

**Abstract**: We construct a corpus of 1,262 verse-commentary (urai) pairs from five Classical Tamil source sections, ranging from technical grammatical prose to modern paraphrase, and ask what information representation learning can recover. We train recurrent and Transformer encoders, a Siamese-style pair-matching network, an mBART-style encoder-decoder, and a decoder-only language model. Each analysis is interpreted against an appropriate control on the same data. TF-IDF provides a strong no-training lexical retrieval baseline, alongside representation analyses and generation controls for the learned models. A fixed string containing the 25 most frequent commentary words scores higher on generation overlap than the decoder-only model. Canonical correlation reaches 1.000 on Gaussian noise at these sample sizes, token-F1 spans only about 0.02-0.20 on this corpus, and the encoder-decoder continues to lower training loss for sixteen epochs after validation loss has begun to rise. One narrow result remains: the decoder-only model prefers authentic word order in 107 of 112 minimal-pair comparisons (95.5%), but does not reproduce held-out commentary content. We release the extraction and evaluation protocol; redistribution of the source commentaries remains subject to permission. 

---
# Beneath the Surface of Chains-of-Thought: A Mechanistic Interpretation of Reasoning Operations in LLMs 

**Authors**: Seogyeong Jeong, Jaehui Hwang, Dongyoon Han, Geonmo Gu, Alice Oh, Taekyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.04753)  

**Abstract**: Reasoning in large language models unfolds through diverse functional operations, such as problem formulation, goal decomposition, and deduction. Although these operations are explicitly distinguished in text, little is known about how they are geometrically organized in representation spaces. To this end, we investigate whether distinct reasoning operations exhibit corresponding geometric structure in hidden representations. We find that operations are separable in held-out representations, with separability peaking in middle layers, and verify that this structure is not explained by lexical or positional confounds. Across layers, token-wise operation-alignment becomes more distributed over spans, while identical surface tokens are represented differently depending on the operation of its surrounding chunk. Attention-masking interventions further show that operation-aligned representations at chunk onset depend on preceding reasoning context. Consequently, our work demonstrates that language models maintain representational correspondence between linguistic reasoning expressions and their internal geometric structures. Code and project materials are available at this https URL. 

---
# Knowing What Not to Answer: Selective Non-Compliance in Vision-Language Models 

**Authors**: Minji Kim, Jihyoung Jang, Hyounghun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.04720)  

**Abstract**: Vision-language models (VLMs) are expected to respond helpfully to appropriate requests while withholding compliance with requests that are incorrect, unsafe, infeasible, or unanswerable. However, existing benchmarks predominantly evaluate non-compliance at the level of the query as a whole, assuming that each request either warrants compliance or requires withholding compliance. In practice, real-world queries can contain a mixture of answerable content and components for which compliance should be withheld. In this paper, we introduce KoNA, a benchmark for evaluating selective non-compliance in VLMs across five categories: False Premise, Visual Inaccessibility, Universal Unknown, Task Feasibility, and Safety. Each task evaluates two capabilities: query-level non-compliance and component-level non-compliance under paired single and compound queries. Our evaluation across diverse VLMs shows that models often fail to refuse, correct, or abstain appropriately, and these failures become more pronounced when queries require selective non-compliance. To address this challenge, we fine-tune VLMs using KoNA examples that require selective non-compliance, together with a fully answerable set that should receive direct answers. Our fine-tuned models achieve substantial improvements in non-compliance accuracy while largely maintaining performance on fully answerable tasks. These results suggest that the fine-tuned models can distinguish between answerable components and those requiring non-compliance and respond in a task-appropriate manner. 

---
# Refuse without Refusal: A Structural Analysis of Safety-Tuning Responses for Reducing False Refusals in Language Models 

**Authors**: Minji Kim, Hyounghun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.04714)  

**Abstract**: Striking a balance between helpfulness and safety remains a fundamental challenge in aligning large language models. To achieve this balance, models should refuse harmful queries (e.g., "How do I shoot someone?") while remaining responsive to benign inputs, even those superficially resembling harmful queries (e.g., "Where can I shoot a good photo?"). However, models often struggle to distinguish genuinely harmful queries from benign queries that contain superficially risky language, resulting in false refusals. In this paper, we address the issue by decomposing a response in the safety-tuning dataset into two distinct components: (i) a boilerplate refusal statement and (ii) a rationale explaining the refusal. Our experiments and analyses show that refusal statements impede accurate discrimination between harmful and benign queries by inducing reliance on superficial cues. In contrast, training solely on rationales reduces false refusals while maintaining a comparable level of safety performance. Rationale-Only benefits also appear in our ICL configuration and remain compatible with the evaluated inference-time mitigation methods. The results emphasize the necessity of precisely curated, fine-grained safety supervision datasets and outline directions for constructing aligned agents that better reconcile helpfulness with safety. 

---
# How Do Language Models Represent and Use Phonological Information for Allomorph Selection? 

**Authors**: Sangwoo Kim, Sangah Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.04708)  

**Abstract**: Language models are trained on tokenized text that obscures the sound structure of words, yet they reliably produce morphemes whose form is phonologically conditioned. It remains unclear whether they rely on item-specific memorization or rule-like generalization and, if the latter, how that generalization is implemented. We therefore ask whether this phonological condition is represented within language models and how it is causally used for allomorph selection. For the English indefinite article a/an, we show that the phonological condition is encoded along a single linear direction in trigger-token embeddings, that this direction causally drives article selection in token-level wug tests, and that, at the article-prediction position, the model forecasts the upcoming trigger token and uses the forecasted trigger's phonological feature to choose the article. We then ask whether this rule-like generalization extends beyond English article selection, both to allomorph selection in other languages and to explicit phonological judgment. Together, these results provide a mechanistic account of phonologically conditioned allomorph selection in language models, and dissociate this generation-time ability from explicit metalinguistic judgments. 

---
# Controlling and Assessing Appropriate Persona Use in LLM-based Dialogue Generation 

**Authors**: Jongkyung Shin, Inkyu Lee, Chiehyeon Lim  

**Link**: [PDF](https://arxiv.org/pdf/2609.04676)  

**Abstract**: In persona-based dialogue generation (PDG), LLMs often overuse persona attributes by incorporating them regardless of dialogue context, resulting in unnatural responses. Despite its practical significance, the underlying causes remain unexplored, with no method to mitigate this problem or metric to assess the appropriateness of persona use. To address these issues, we first conduct a comprehensive analysis of LLM-based PDG, revealing that LLMs exhibit a systematic bias to incorporate all given persona attributes, and that existing metrics fail to capture contextual appropriateness. Building on these findings, we propose Self-CONtrastive Persona Overuse Suppression (SCONPOS) to mitigate overuse by directly intervening in LLMs' internal representations at the prompt encoding stage, without requiring any response generation. We further propose the Persona Appropriateness Score (PAS), a novel metric that penalizes both overuse and underuse. Experimental results demonstrate that SCONPOS systematically reduces overuse, and PAS captures the contextual appropriateness of persona use. 

---
# Choosing the Right Language Mode at Inference Time for Multilingual Reliability 

**Authors**: Ekata Mitra, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2609.04653)  

**Abstract**: Multilingual large language models often struggle to reason in low- to mid-resource languages. Prior work has shown that translation can improve multilingual reasoning by helping models access stronger English-centric representations. This raises a central question: How much translation is needed for multilingual large language models to reason reliably, and when does more translation instead trigger interference and overconfidence?
Using LLaMA and Qwen models, we run extensive experiments varying text scope and language mode (target-only, English-only, bilingual) to evaluate both accuracy and reliability.
Our results reveal a clear trade-off: English context often improve understanding and recover errors caused by non-English comprehension, yet adding redundant bilingual context intensifies interference. We address this trade-off with Reliability-Aware Adaptive Inference (RAAI), a training-free test-time framework that (i) performs Expected Calibration Error (ECE)-aware routing and prompt fusion, and (ii) uses a mid-layer Risk Index (RI) to gate sequential reasoning, allocating compute only when it is likely to help and suppressing harmful bilingual redundancy. Across two model families, RAAI enhances accuracy by 25-37.7% on low-resource languages and lowers calibration error by approximately 3-6%, with the most pronounced benefits in the lowest-resource language tiers. 

---
# ConsensusBench: Benchmark of Consensus Nodes for LLM Reasoning via Outcome Reward Densifying 

**Authors**: Shi-Qi Yan, Chao-Hong Tan, Qian Chen, Wen Wang, Xiangang Li, Zhen-Hua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2609.04648)  

**Abstract**: Reinforcement learning (RL) has become one of the primary paradigms for reasoning enhancement of large language models (LLMs). In particular, Group Relative Policy Optimization (GRPO) and related algorithms have demonstrated strong performance with outcome-level rewards. However, these methods depend solely on the final answer, without feedback regarding which intermediate steps contribute to success or failure. As task complexity and reasoning trajectory length increase, such sparse final-answer rewards become increasingly insufficient. To address this limitation, we introduce ConsensusBench, a novel dataset designed to provide rule-based process-level signals. We posit that a correct final answer relies on a small set of intermediate conclusions throughout the reasoning process, which can be seen as a verifiable sub-outcome. We identify these sub-outcomes by filtering correct trajectories from N rollouts and clustering semantically equivalent intermediate statements. We call these clustered statements as Consensus Nodes. By integrating a rule-based process reward derived from these nodes into GRPO-style algorithms, we develop a new reinforcement learning signal named ConsensusPR. It directly reduces the reward sparsity of outcome reward across long reasoning trajectories. To facilitate systematic process-level evaluation, we introduce three metrics to our benchmark: Final Answer Accuracy (Acc), Node Coverage Rate (NCR), and Tokens per Node (TPN). Experiments across AIME 2024, AIME 2025, GSM8K, MATH-500, and our ConsensusBench demonstrate that the proposed method consistently surpasses GRPO-style approaches, highlighting the practical value of consensus nodes in guiding reasoning. 

---
# CAGE: Coherence-Aware Graph Encoding for Retrieval-Augmented Generation 

**Authors**: Tong Qi, Jingyu Wu, Youbing Yin, Spencer Hong, Daben Liu, Erin Babinsky  

**Link**: [PDF](https://arxiv.org/pdf/2609.04647)  

**Abstract**: Traditional Retrieval-Augmented Generation (RAG) systems score each passage independently against the query, assembling context sets that may be individually relevant yet collectively incoherent. We introduce Coherence-Aware Graph Encoding (CAGE), a reranking framework that models "between-chunk coherence" across four dimensions: Intra-Domain Relevance, Noise Resistance, Informational Bonding, and Factual Consistency. Our pipeline transforms retrieved passages into directed heterogeneous entity graphs, amplifies factual anchors via min-out-degree reweighting, encodes structural patterns through a Relational Graph Convolutional Network, and fuses inter-chunk coherence with query relevance for final ranking. Evaluated across four multi-hop benchmarks, CAGE matches or outperforms strong baselines including monoT5 in Recall@5 on bridge-dominated datasets and consistently improves downstream Exact Match, demonstrating that structurally coherent context yields more precise answers even when retrieval recall is comparable or lower. 

---
# Tracing Audio Grounding and Answer Selection in Audio LLMs 

**Authors**: Hyebin Cho, Suho Yoo, Jihoo Jung, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2609.04637)  

**Abstract**: Audio Large Language Models (Audio LLMs) have advanced in audio understanding, yet they can still predict the answer by reasoning from textual cues or linguistic priors rather than the provided audio. A common remedy is to train models on data whose answers cannot be inferred from text alone. This approach can improve performance, but what changes within the model remains unclear. In this paper, we ask what must happen inside the model for the audio to actually determine the answer. Our findings are threefold. (1) Replacing the audio with silence or unrelated audio causes substantially larger performance degradation in the trained model than in the pretrained model. (2) Acoustic information most strongly shapes the model's representations of the answer choices in early-to-middle layers, while training mainly increases the influence of audio information on the final prediction in middle-to-late layers. (3) The weights learned during training have their largest impact in specific layer bands. Together, these results provide a mechanistic account of how training strengthens the use of acoustic evidence in Audio LLMs. 

---
# PetQA: Benchmarking Veterinary Knowledge and Clinical Reasoning 

**Authors**: Taegyun Kim, Youngwook Ham, Jungwook Rhim, Ju-Hyun An, Sungkyu Park, Kunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2609.04598)  

**Abstract**: We introduce PetQA, a Korean long-form question-answering (QA) benchmark for evaluating veterinary knowledge and clinical reasoning in large language models (LLMs) and large vision-language models (LVLMs). PetQA contains 10,076 text-only and 8,751 multimodal QA pairs derived from real-world questions about dogs and cats, paired with answers from expert veterinarians. Its test split, PetQA-Bench, further includes annotations for question types and clinical conditions. We evaluate eighteen models using ROUGE, BERTScore, and LLM-as-a-judge metrics for factuality and helpfulness under three settings: zero-shot inference, retrieval-augmented generation (RAG), and supervised fine-tuning (SFT). The benchmarking results provide an overview of the strengths and limitations of current models in addressing veterinary clinical queries and highlight the need for more effective adaptation methods to develop clinically reliable AI systems for veterinary care. To facilitate broader use, we additionally provide translated versions of PetQA-Bench in five languages. 

---
# When Do Internal Probes Beat Reading the Answer? Miscalibrated Readouts and Behavior-Concealed Knowledge in Language Models 

**Authors**: Gnaneswar Villuri, Hashmath Shaik, Alex Doboli  

**Link**: [PDF](https://arxiv.org/pdf/2609.04582)  

**Abstract**: A 0.6B language model, asked to verify 1,200 logical conclusions (half valid, half corrupted by a single semantic edit), answers YES every time. Judged by behavior it discriminates nothing; linear probes on its hidden states read the correct verdict at 0.96 AUC, transferring to unseen logical structures and separating foils built from exactly the words of the true conclusion (0.90). We ask where the verdict is lost, and find the dominant failure is a single scalar. The verdict survives to the model's own output logits (margin AUC 0.89) along a well-aligned readout direction; a saturated decision threshold, offset by +4.6 sigma, erases it. The diagnosis generalizes: across 90 semantic-label configurations of a five-model, three-family factorial, behavioral accuracy collapses onto a single function of threshold offset (Spearman -0.93) while margin ranking moves far less. Across a 13x scale range, internal knowledge saturates while free-form behavior is non-monotone: an 8B model underperforms its 4B sibling through an answer-channel failure rather than the threshold; forced-choice accuracy is monotone. The diagnosis is actionable: a one-parameter correction, never fit on evaluated structures, repairs behavior from 50% to 81% (0.6B); calibrated margin decoding recovers 94% at 8B; few-shot prompting works the same way, recentering the threshold (+4.6 sigma to 0.0 sigma) while preserving ranking. Comparing probe to margin separates three regimes: concealed, miscalibrated, and undetected. On a maze task built so foils carry no surface cues, the audit correctly reports the third. In the standard generation setting, answer-surface features and heuristic labels reproduce published probing results without any internal access. 

---
# Rhythms of Work: Multi-Scale Interpretation of Human Behavioral Traces for Workplace Agents 

**Authors**: Lin Ai, Scott Counts  

**Link**: [PDF](https://arxiv.org/pdf/2609.04556)  

**Abstract**: Runtime traces are becoming a central substrate for understanding agentic systems, yet interpretation has focused largely on what the agent did. Workplace agents face the complementary problem: interpreting the human activity that surrounds them. Hours of low-level events carry rich evidence about a user's state but are too granular to reason over directly, and flattening them into one stream or compressing them into a single embedding both treat "summarize the user's behavior" as if it had one correct answer. We argue instead that behavioral interpretation is resolution-dependent: the same trace should admit multiple addressable interpretations at different temporal resolutions. We construct a multi-resolution vocabulary of semantically normalized operators, recurring motifs, coherent episodes, and day-level rhythms, each preserving the structure salient at its own horizon. Applied to 667 million human-attributed events from a large commercial productivity suite (50,000 users, 100 organizations), it yields 120 operator types, thousands of motifs, 25 episode types, and five day-rhythm archetypes. We validate it on real telemetry: re-running the entire pipeline on a disjoint 2,000-user sample recovers the same taxonomy (structural stability), and on held-out users the full representation forecasts a user's next episode more accurately than a flat-operator baseline, a 17% relative macro-F1 gain (predictive validity), so the abstractions preserve future-relevant information rather than merely describe it. A controlled resolution ablation then shows that no single level is optimal across questions: different agent-facing questions about the same trace are best answered at different resolutions. Behavioral trace interpretation for agents should therefore be multi-resolution and query-conditioned: an agent should access the temporal grain a question needs, not one universal summary. 

---
# A Calibrated Reflection Approach for Enhancing Confidence Estimation in LLMs 

**Authors**: Umesh Bodhwani, Yuan Ling, Shujing Dong, Yarong Feng, Hongfei Li, Ayush Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2609.04539)  

**Abstract**: A critical challenge in deploying Large Language Models (LLMs) is developing reliable mechanisms to estimate their confidence, enabling systems to determine when to trust model outputs versus seek human intervention. We present a Calibrated Reflection approach for enhancing confidence estimation in LLMs, a framework that combines structured reasoning with distance-aware calibration technique. Our approach introduces three key innovations: (1) a Maximum Confidence Selection (MCS) method that comprehensively evaluates confidence across all possible labels, (2) a reflection-based prompting mechanism that enhances reasoning reliability, and (3) a distance-aware calibration technique that accounts for ordinal relationships between labels. We evaluate our framework on diverse datasets, including HelpSteer2, Llama T-REx, and a proprietary conversational dataset, demonstrating its effectiveness across both conversational and fact-based classification tasks. This work contributes to the broader goal of developing reliable and well-calibrated confidence estimation methods for LLMs, enabling informed decisions about model trust and human judgement. 

---
# Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs 

**Authors**: Tung-Ling Li, Jiale Huang, Lee-Chi Wang, Janaki Ram Gotei  

**Link**: [PDF](https://arxiv.org/pdf/2609.04526)  

**Abstract**: Merging a LoRA adapter into its base model is standard deployment practice: it removes the runtime adapter's per-forward overhead and leaves a single standalone checkpoint any serving stack can load. On a native 4-bit microscaling checkpoint (NVFP4, MXFP4) that step stops being free. The merged weights must be written back through a quantizer, which re-derives the checkpoint's discrete E2M1 code plane (roughly 90% of the artifact's bytes), so the deployed artifact becomes coupled to one quantization convention, and every later code-touching event in its lifecycle can move it. Done naively the step is worse than fragile: it deletes the adaptation, by up to 39 pp, because against an already-on-grid base the reconstruction optimum is that base. Scale-QLoRA instead adapts only the native per-block scale field, trains those scales on the deployment grid, and freezes every E2M1 code. Within a fixed native format, scale grid, block layout and code plane, merging is then a bit-exact identity and the merged artifact is code-invariant. Across four models and four tasks, Scale-QLoRA and merge-aware QAT-LoRA are both accuracy-lossless, so we claim no accuracy ordering between them; they differ structurally, in that QAT-LoRA re-derives the code plane through a quantizer while Scale-QLoRA preserves it exactly. That difference is what the lifecycle prices: nearest-rounding implementations disagree by about a point on the measured task, and more extreme rule mismatches can drive the weight-space artifact to ~0%, which we report as a sensitivity bound rather than a deployment frequency. Preserving the code plane also drops the weight-space straight-through estimator from training (3.9x per step on the dense 8B model) and enables exact rollback, code-plane deduplication, and a ~125x faster scale-only task swap. 

---
# LentEx: Generalizable Latent Entity Extraction via Synthetic Data and Instruction-Tuned LLMs 

**Authors**: Umesh Bodhwani, Yuan Ling, Cibi Chakravarthy Senthilkumar, Shujing Dong, Yarong Feng, Hongfei Li, Ayush Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2609.04511)  

**Abstract**: Latent entity extraction (LEE) tackles the challenge of identifying implicit, contextually inferred entities within free text-an area where traditional entity extraction methods fall short. In this paper, we introduce LentEx, a novel framework for latent entity extraction that leverages synthetic data generation and instruction fine-tuning to optimize smaller, efficient large language models (LLMs). Latent entities, which are often abstract and thematic, are crucial for applications such as retrieval-augmented generation (RAG), customer persona analysis, and knowledge graph enrichment. LentEx addresses the scarcity of labeled datasets by employing a template-based approach to generate diverse, contextually rich synthetic data, ensuring high variability and alignment with real-world distributions. To our knowledge, LentEx is the first to systematically approach LEE through the lens of LLMs. LentEx demonstrates significant performance improvements across multiple tasks, notably surpassing state-of-the-art models on the MTEB Clustering Benchmark. Furthermore, our methodology enables robust generalization to unseen domains, making LentEx highly applicable in real-world NLP tasks, including RAG and clustering, thereby establishing a new paradigm for latent entity understanding and extraction in natural language processing. 

---
# Towards Understanding Pause Token Fine-Tuning Dynamics: A Mode Retention Perspective 

**Authors**: Jaehyeon Kim, Suhwan Kim, Nakyung Lee, Yeongoon Kim, Jimin Seo, Giho Lee, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.04489)  

**Abstract**: Pause-token methods improve LLM reasoning by inserting special tokens into sequences. Prior work explains these gains through computational expressivity. However, there is relatively little investigation into the training dynamics of pause tokens. We explore how pause tokens reshape the training dynamics of fine-tuning. Two controlled pilots expose distinct asymmetries. On a synthetic continual-learning task, masked pauses overwrite a previously-learned distribution roughly 4x less at matched final adaptation (H1, mode retention); on a synthetic math-reasoning probe, the boundary-adjacent token comes to encode substantially more downstream-step information (H2, non-myopic compression). We formalize a training rule consistent with both - Masked Boundary Pause (MBP), pause tokens placed at reasoning-step boundaries with their loss masked. Across 1B-8B Qwen and Llama models, MBP consistently improves reasoning, achieving gains of up to 6 points on math and 2.5 points on code, while preserving general language understanding abilities. We further demonstrate that this mode-preserving strategy extend gains to GRPO. These results recast pause tokens as a training-dynamics intervention on the retention-adaptation trade-off, rather than merely an inference-time computation device. 

---
# Cultural Misalignment in Large Language Models: Detection, Measurement, and Mitigation Through Targeted Fine-Tuning 

**Authors**: Antoni Czolgowski, Abel Iyasele  

**Link**: [PDF](https://arxiv.org/pdf/2609.04485)  

**Abstract**: We evaluate three open-weight LLMs (Gemma3-12B from the USA, Bielik-11B-v3 from Poland, and Qwen3-4B from China) against World Values Survey Wave 7 data for 63 demographic personas across three countries, using normalized Wasserstein distance to quantify distributional misalignment. Contrary to expectations, no model favors its home country: the Chinese-built Qwen3-4B performs worst on its own Chinese population (W1 = 0.436, the highest misalignment in the entire model x country matrix). Targeted LoRA fine-tuning on the five worst-case personas, requiring fewer than 1,200 training pairs and under 15 minutes on a single GPU, reduces bias by 16.8% for Bielik-11B (p_Bonf = 0.002, d = -4.4) with all five targets improving. However, country-level decomposition reveals that fine-tuning redistributes rather than removes bias: Bielik's worst-case personas swap entirely from American to Chinese elderly, with zero overlap between pre- and post-correction sets. To our knowledge, this is the first study to target worst-case demographic personas with LoRA fine-tuning for cross-cultural bias mitigation. 

---
# Patterns of Priming in Production: Lexical, Semantic and Structural Alignment in Language Model Generation 

**Authors**: Giulia Pucci, Ruizhe Li, Arabella Sinclair  

**Link**: [PDF](https://arxiv.org/pdf/2609.04484)  

**Abstract**: This paper investigates structural priming in language model (LM) production, examining how preceding structural context influences sentence completion. While prior work has demonstrated priming effects in comprehension of structural alternations, it remained unclear whether these persist in production, where, when generating, an LM samples from many possible continuations at each step. We address this question through a series of controlled sentence-completion experiments on dative constructions. In line with prior work, we find that LMs are susceptible to structural priming, particularly in sentences that are semantically coherent. In terms of priming magnitude, we find that while there is a greater relative increase of double-object datives against our baselines, in line with inverse frequency effects, there is a larger absolute increase in prepositional-objects, the more frequently produced construction. Finally, we not only observe that structural priming is boosted by lexico-semantic coherence, but that structurally primed completions display greater levels of lexico-semantic repetition. Taken together, our evidence supports the view that structural priming in LMs operates across multiple levels of linguistic representation, facilitating, and facilitated by syntactic, lexical, and semantic alignment. Code: this https URL. 

---
# Safety for Whom? Boundary-Aware Self-Distillation for Controlled LLM Safety Refusal 

**Authors**: Alejo López-Ávila, Iker García-Ferrero, Jezabel Garcia, Antonio Tiene, Román Orús  

**Link**: [PDF](https://arxiv.org/pdf/2609.04482)  

**Abstract**: Safety alignment is usually posed as a topic-level question: is this subject harmful? Deployments ask a narrower one. A civics tutor and a public-sector assistant may share a base model yet need different boundaries inside the same topic, refusing targeted political manipulation while still answering factual questions about the same election. We formulate this as narrow-boundary safety and introduce an offline self-generated framework combining controlled topic generation, coverage repair, in-distribution compensation data, and harmful-benign pairs for training and evaluation. Single-shot generation leaves 19.88% of prompts without accepted refusal traces, whereas escalating retries leave 0.20%. On political persuasion with Qwen3-8B, training on refusal data completed through Escalate increases target-domain refusal from 9.47% to 84.75% and reduces the mean unsafe-response rate across three broader harmfulness benchmarks from 26.26% to 0.14%, but increases XSTest over-refusal from 2.00% to 74.00%. In a separate matched comparison, replacing external responses with verified target-model responses reduces over-refusal from 15.20% to 5.20%. Boundary-pair data reduces comply-side over-refusal on held-out pairs from 32.94% to 4.16%, while harmful-side refusal decreases only from 91.88% to 87.72%. These results show that data composition controls the safety and usability trade-off, and that safety alignment should be evaluated on both sides of the intended refusal boundary. 

---
# Shared circuits predict whether LLMs generalize across formats in arithmetic reasoning 

**Authors**: Andrea Gregor de Varda, Sana Pandey, Pengrui Han, Jacob Andreas, Evelina Fedorenko  

**Link**: [PDF](https://arxiv.org/pdf/2609.04463)  

**Abstract**: In many forms of reasoning, including arithmetic reasoning, generalizing across superficial changes in input format is effortless for humans: anyone who can solve 2+5 can also solve 'two plus five'. In contrast, LLMs are more brittle to surface variations of the prompts: for example, they solve numeric arithmetic problems almost perfectly but are substantially less accurate on verbal renditions of the same problems. Here, we ask whether generalization across formats can be predicted from the models' internals. Using attribution patching, we first independently localize the circuit that each model recruits to solve numeric arithmetic problems (2+5) vs. verbal ones, in three languages: English ('two plus five'), Spanish ('dos más cinco'), and Italian ('due più cinque'); then, we test whether overlap with the model's own numeric circuit predicts its generalization to the verbal formats. Indeed, we find support for this idea at three levels: circuit overlap accounts for the relative difficulty of the three verbal formats, for which models generalize best, and for which items are solved correctly, rivaling supervised probes while requiring no labeled data. 

---
# TRILOGUE: A Trilingual Spoken Dialogue Fact-Checking Benchmark with Evidence and Paired Audio 

**Authors**: Chaewan Chun, Meruyert Aristombayeva, Jiyoung Choi, Mahjabin Nahar, Delvin Ce Zhang, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.04452)  

**Abstract**: Modern misinformation is often heard before it is read, yet fact-checking systems are still evaluated mainly on clean written claims. Spoken dialogue remains different even when systems operate on transcripts: claims may be distributed across speakers and turns, depend on prior context, and become harder to verify when Automatic Speech Recognition (ASR) errors distort the available text. Prior spoken dialogue fact-checking resources are small, English-centric, or focused on annotation rather than end-to-end benchmarking, leaving no large multilingual benchmark with paired speech and turn-level labels. We introduce TRILOGUE (TRIlingual spoken diaLOGUE fact-checking), a large-scale trilingual benchmark of source-grounded spoken dialogues in English, Russian, and Kazakh. It contains nearly 12K dialogues, 187K turns, and 390 hours of paired audio with ASR transcripts and word-level timestamp alignments across all three languages, including nearly 5K human-recorded Russian and Kazakh dialogue files. TRILOGUE supports claim check-worthiness detection, source-article evidence retrieval, and claim verification with claim-only, gold-evidence, and retrieved-evidence inputs. Baselines show that ASR degradation and cross-lingual transfer remain challenging, especially for Kazakh, while retrieved source evidence substantially narrows the gap to gold-evidence verification. 

---
# GRACE: Graph-Grounded Reflective Agent Copilot Engine for Expert-in-the-Loop Knowledge Expansion 

**Authors**: John Seon Keun Yi, Joshua R. Minot, Dokyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.04442)  

**Abstract**: Large language models deployed in high-stakes settings frequently generate plausible but ungrounded claims. Standard retrieval-augmented generation (RAG) pipelines offer limited remedy, since they retrieve isolated passages without tracking cross-document evidence relationships or quantifying uncertainty. We introduce GRACE (Graph-grounded Reflective Agent Copilot Engine), a framework that deconstructs LLM responses into atomic claims and grounds them against trusted knowledge priors within a weighted bipartite graph. Edge weights encode the closeness of each claim to the priors, enabling weighted centrality analysis that classifies claims as Grounded, Refuted, or Boundary. Such classification identifies not just hallucinations but also novel or contested claims at the frontier of the model's knowledge. To efficiently allocate human or agent resources, we formulate a Return on Attention (RoA) objective that defers a claim to expert review only when its priority-weighted uncertainty exceeds the cost of verification. Claims verified by experts are promoted to new evidence anchors, closing a validator-LLM evolutionary loop that expands the knowledge base across iterations. We evaluate GRACE across multiple language models and on datasets spanning both general and domain-specific knowledge. Our results show that our knowledge base serves as a reliable foundation for retrieval that outperforms RAG baselines, and that the RoA framework efficiently selects valuable boundary knowledge for expert verification. These findings demonstrate that graph-structured representations combined with expert-in-the-loop verification can mitigate hallucination at the system level rather than at the generation level. Code available at this https URL 

---
# What Attention Recalls and Recurrence Controls in Hybrid Language Models 

**Authors**: Kirill Afendulev, Alexey Dontsov, Elena Tutubalina, Anton Korznikov  

**Link**: [PDF](https://arxiv.org/pdf/2609.04434)  

**Abstract**: Hybrid language models combine attention with a fixed-size recurrent state, but the role of each channel remains unclear. We introduce two cache-level interventions. Split-prefill keeps only the KV cache or only the recurrent state from a prefilled context, then generates an answer. State-swap pairs the KV cache from one context with the recurrent state from another in a single forward pass. On Qwen3.5 and Falcon-H1, the two channels split sharply by function. Exact retrieval survives only through attention (64-98% of full accuracy) and collapses to zero through recurrence. Output language and persona reverse the pattern: both survive recurrence (70-80% and 3-5x) while KV-only drops to ~1% language accuracy. State-swap confirms this causally: the answer takes its value from the KV side and its language from the recurrent side. Recurrent-only generation also accepts words that were never in the context but share meaning or parts with seen items. Attention provides a lookup over what was said; the recurrent state shapes how the model says it next. 

---
# A Systematic Evaluation of Cross-Lingual Consistency Enhancement Methods in Multilingual Language Models 

**Authors**: Jirui Qi, Mingyang Wang, Hinrich Schütze, Raquel Fernández, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2609.04409)  

**Abstract**: Multilingual language models often produce inconsistent answers to semantically equivalent questions across languages, motivating methods to improve cross-lingual consistency (CLC). However, existing methods are typically evaluated using different models, tasks, and protocols, leaving their relative strengths unclear. In this work, we present a unified evaluation of representative CLC-enhancement methods for question answering, spanning inference-time interventions and post-training approaches across three model families and three closed-form benchmarks. The results show that post-training methods are generally more reliable, with direct distribution alignment consistently improving CLC across all model-dataset combinations, while other methods are more sensitive to answer format and the breadth of language coverage. Notably, cross-domain transfer is limited unless source and target tasks share similar output formats. We further investigate whether CLC enhancement hurts models' ability to respond differently *when needed*, that is, when asked culture-dependent questions. Across two benchmarks of culturally diverse question answering, we find no systematic degradation in controlled closed-form evaluation, whereas open-ended generation reveals occasional accuracy reductions, particularly for non-English responses. Our work highlights the need to evaluate CLC enhancement for both cross-domain robustness and culturally appropriate variation, informing future work in post-training and benchmark development. 

---
# The Anatomy of an ASR Hallucination 

**Authors**: Hamees Sayed, Apoorv Singh, Kumar Aman, Akshat Mandloi  

**Link**: [PDF](https://arxiv.org/pdf/2609.04404)  

**Abstract**: ASR systems sometimes produce fluent text that is unrelated to the speech they receive. We view these hallucinations as one possible consequence of a broader grounding failure, in which the transcript is no longer adequately guided by the audio. To understand where this failure becomes possible, we study two independently trained Conformer-Large recognizers - one CTC and one RNN-T - under environmental degradation and speaker-background shift. In both models, the final encoder stage emerges as a critical boundary: bypassing the final block causes divergence on nearly every utterance, whereas bypassing middle blocks has little effect. At this same stage, the representations become more compact, text becomes readable by the trained decoder, and grapheme information becomes explicit. Importantly, the intervention produces garbled or repetitive output rather than fluent fabrication. Our result therefore identifies a mechanistic precondition for hallucination - the failure to produce adequately grounded output - not the complete origin of naturally occurring hallucinations. Together, the results reveal a consistent terminal-stage dependency for grounded recognition across two decoder families and multiple distribution shifts. 

---
# Evaluation of Phonetic Encoding Algorithms on Transcription Datasets 

**Authors**: Can Özbey, Emre Kaplan, Berkin Deniz Kahya  

**Link**: [PDF](https://arxiv.org/pdf/2609.04391)  

**Abstract**: In this work, a novel evaluation scheme built on a generalized variant of the Rand Index measure, namely, the Hüllermeier-Rifqi Index, is proposed in order to assess how well phonetic encoding algorithms conform to word-based transcriptions in IPA (International Phonetic Alphabet) notation. For this objective, the discordance score is obtained by calculating the absolute difference between the pairwise similarity values of ground-truth transcriptions and those of corresponding phonetic encodings, which are computed using normalized edit distance as a permutation dependent string metric. The resulting score is subsequently adjusted with respect to that of a random string generator incorporating the same alphabet as the encoder under consideration. A wide range of phonetic encoders were evaluated as such on multi-lingual transcription datasets along with their recall capabilities based on the collision rate. The validity of the proposed scheme is further supported by its applicability in measuring the orthographic transparency of a language when the writing system is viewed as an inherent phonetic representation. 

---
# You Really Didn't Get That? Benchmarking Social Pragmatic Inference for Indirect and Playful Chinese Online Comments 

**Authors**: Shiwei Hong, Junjie Ma, Emma Jiren Wang, Ethan Z. Rong, Siying Hu, Haichang Li, Ziying Wang, Zhicong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2609.04384)  

**Abstract**: Chinese online comments often convey social meaning through indirect and playful language that is hard to interpret without context. Existing evaluations largely organize items around predefined phenomena or controlled pragmatic categories, leaving open whether models can distinguish plausible readings of what a naturally occurring comment is doing in a particular exchange. We introduce a benchmark for evaluating whether LLMs can recover such situated pragmatic meanings. From more than 200,000 public Chinese social media interaction records, we construct 4,735 human-validated diagnostic items, each pairing a target comment with reconstructed preceding context and plausible misreadings. We evaluate eight LLMs as both question writers and solvers in a cross-writer setting. The task is challenging: the strongest model achieves 81.42% leave-writer-out accuracy. Across all eight models, the mean leave-writer-out accuracy is 68.70% while human accuracy was 90.8%. Case analysis shows that models often recognize broad irony or playfulness while misidentifying the mechanism or interactional move. 

---
# VERGE: Verification-Enhanced Refinement for Grounded Extraction of Early-Onset Colorectal Cancer Symptoms in Clinical Notes 

**Authors**: Nikkie Hooman, Monarch Nigam, Amy E. Hughes, Rasmi G. Nair, Mehak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2609.04366)  

**Abstract**: Early-onset colorectal cancer is increasing among younger adults, yet red-flag symptoms in this age group have no evidence-based guidelines for follow-up testing, and structured encounter data do not capture the detail needed to support early detection and inform follow-up, including symptom duration, context, and fam- ily history, an established colorectal-cancer risk factor. This study aimed to develop and evaluate an automated method for extracting six red-flag symptoms and family-history risk status from free-text clinical notes. We developed VERGE, an agentic workflow in which an initial label and evidence are proposed using retrieval-augmented generation, then passed through a bounded verification- refinement cycle that checks textual grounding and clinical validity, corrects and rechecks a claim until resolved or a limit is reached, and escalates unresolved claims for human review. VERGE was evaluated on 4,033 clinician-labeled note-finding pairs against a single-agent baseline, a rule-based clinical language-processing baseline, and an alternative underlying language model. Compared with the single-agent baseline, VERGE reduced false positive find- ings, improving precision from 0.764 to 0.849 and MCC from 0.681 to 0.730, a balanced gain across the precision-recall trade-off, and resolved most flagged errors autonomously, with human review required for only 1.5 percent of claims. These results indicate that a bounded, verification-based workflow can reduce unnecessary positive findings without sacrificing the ability to detect true ones. This approach offers a path toward more reliable and trustworthy clinical language-processing tools to support colorectal cancer risk assessment in younger patients. 

---
# Adapting from Downturns: Prediction of Long-Term Conversational-Skill Development in Mental-Health Crisis Counselors 

**Authors**: Vivian Nguyen, Lillian Lee, Elizabeth A. Olson, Cristian Danescu-Niculescu-Mizil  

**Link**: [PDF](https://arxiv.org/pdf/2609.04350)  

**Abstract**: How do people learn to become better conversationalists? This question is especially important in the context of mental-health counseling, where conversational skills are essential, yet volunteer counselors often have limited access to supervision and structured feedback. Understanding how counselors develop their ability to steer conversations toward positive outcomes -- and identifying early which counselors are (not) on track to improve -- can help prioritize support for the counselors who need it most.
In this work, we introduce the task of predicting, early in a conversationalist's career, whether they will eventually improve at steering conversations toward positive outcomes, and demonstrate the feasibility of this task in the case of volunteer mental-health crisis counselors. Our central insight is that people may struggle with particular kinds of moments in a conversation, and that what is especially revealing of their likelihood of future improvement is how they learn to handle those moments over time. We operationalize this insight by designing a method that identifies the types of moments a counselor initially struggles with, captures how they adapt their response when they re-encounter similar moments in subsequent conversations, and learns which early adaptations predict improvement months or even years later. While this future-prediction task is challenging, our counselor-adaptation approach yields better results than baselines that learn directly from the conversation transcript. 

---
# MedProb: Probing Internal Representations of Vision-Language Models for Medical Question Answering 

**Authors**: Erfan Nourbakhsh, Ke Yang, Anthony Rios  

**Link**: [PDF](https://arxiv.org/pdf/2609.04336)  

**Abstract**: Medical visual question answering (Med-VQA) is often assumed to require medical fine-tuning, large models, or complex multi-agent pipelines. We revisit this assumption with \textbf{MedProb}, a lightweight probing framework that predicts multiple-choice Med-VQA answers from frozen VLM representations without free-text generation. Across PATH-VQA, SLAKE, and VQA-RAD, MedProb recovers substantially more answer-relevant signal than prompting and performs stronger than medical VLMs and agentic systems. Probing also reduces the apparent gap between small and large models compared to prompting, suggesting that smaller VLMs contain more recoverable Med-VQA signal than generation-based evaluation reveals. Across 14 matched general-purpose and medical VLM pairs, medical adaptation does not consistently improve this linear decodability. Finally, free-text generation exhibits an answer-position bias of up to 10 percentage points, whereas MedProb also has positional bias, however, it is impacted differently than prompting. Our main results target the multiple-choice/multiclass Med-VQA setting; we additionally show the probe can be extended to open-ended generation via a rejection-sampling scoring procedure. 

---
# Evidence Integration in Large Language Models 

**Authors**: Sebastien Kawada, Manolis Kellis  

**Link**: [PDF](https://arxiv.org/pdf/2609.04290)  

**Abstract**: Despite increasing reliance on LLMs that reason with external evidence supplied by tools, retrieval-augmented generation, other agents, and users, how LLMs integrate such evidence into decisions they have already begun to form remains largely unclear. We present a distributional theory in which evidence shifts the receiver's distribution of initial answers, driven by a receiver prior weight and a candidate evidence tilt, leading to three predictions. First, candidates more probable to the receiver are more persuasive. Second, receivers more readily integrate characteristic errors of their own than foreign errors from different sources. Third, identical evidence can improve weaker models and harm stronger ones. We confirm these over ten million trials, twelve LLMs from four families, and eight domains, four of them scientific discovery tasks in the physical and life sciences: quantum mechanics, physics, genetics, and molecular biology. The law also yields a receiver-relative reliability frontier: receiver-congruent errors depress performance more steeply than random errors of the same rate. LLMs also integrate candidates even after internally verifying their invalidity (93-100% with propositional constraints; up to 99.4% on held-out physical and life-sciences reasoning), demonstrating evidence integration is a receiver-specific control policy over existing distributions, determined by receiver properties rather than scalar trust in the evidence source. Causal interventions show candidate integration is implemented late in the network, as a structured sequence of steps admitting external candidate answers, promoting them, and transporting them into the answer state. Representations of verification are decodable but have little causal impact on answers. A J-lens decomposition shows the state underlying verbalized verification is fully dissociable from that underlying candidate integration. 

---
# Memory as transformation: LETHE, a self-referential gan-inspired architecture 

**Authors**: Francesco Vitucci, Anthony Di Furia, Francesco Scagliola  

**Link**: [PDF](https://arxiv.org/pdf/2609.04289)  

**Abstract**: LETHE (Latent-parameter Evolution with Temporal Hierarchical quasi-Equilibrium) is a self-referential sonic-oblivion system implemented in SuperCollider. It adopts the formal vocabulary of Generative Adversarial Networks in a closed configuration without external datasets or supervision after initialization. Audio is processed by a 3 x 3 mixing matrix built around two delay lines; its nine coefficients and two delay times evolve through the interaction of a five-feature linear discriminator and a random-perturbation optimizer analogous to single-sample REINFORCE. The discriminator compares current energy behavior with an archive of the initial state and guides parameter updates. Circular, fixed, and live sources can be mixed independently. Across fixed and circular sessions with an ablation control, the active generator is necessary for parametric evolution ($\Delta c_{22}=0.000$ in all 15 ablation sessions). Situated in the tradition of self-referential electroacoustic music, LETHE delegates the sonic outcome to an adaptive closed loop whose parametric space is defined by the composer. 

---
# How Much Does Corpus Choice Change Dependency-Distance Estimates? 

**Authors**: Sirui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.04223)  

**Abstract**: Dependency-distance estimates derived from a single corpus are routinely treated as properties of a language, yet this assumption has not been tested across independently compiled corpora. We compared mean dependency-distance estimates across 38 same-language treebank pairs from Universal Dependencies v2.18, using concordance correlation, Bland-Altman analysis, and a twelve-specification multiverse design. Cross-treebank agreement was moderate at best: substituting one treebank for another reversed nearly 40 percent of pairwise language orderings, and treebank choice accounted for roughly 29 percent of between-group variance. This disagreement substantially exceeded within-treebank sampling error and persisted across all twelve preprocessing specifications. Nevertheless, every treebank confirmed dependency-length minimization (normalized ratio below 1). The data are more consistent with MDD as a corpus-conditioned composite of grammatical, register, and annotation factors than as a stable language-level parameter: the qualitative DLM universal survives corpus substitution, but the ordinal cross-linguistic ranking does not. 

---
# Same Trajectory, Contradictory Rewards (ROBORMBENCH): Paraphrase Fragility in Vision Language Reward Models 

**Authors**: Wonje Jeung, Sangyeon Yoon, Hyesoo Hong, Yoonjun Cho, Dongjae Jeon, Bumjun Kim, Jean Oh, Youngjae Yu, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2609.05401)  

**Abstract**: Vision-language models are increasingly used as reward functions for robotic learning, but this role requires paraphrase invariance: the same trajectory should receive the same reward under semantically equivalent goal descriptions. We show that current VLM reward models often violate this property. Paraphrasing the instruction alone can substantially change predicted progress scores, and can even flip identical robot behavior between failure and success. To measure this failure mode, we introduce ROBORMBENCH, a benchmark with 2,390 real-robot trajectories, ground-truth progress labels, and 21,673 verified paraphrases spanning lexical, syntactic, and action-goal rewrites. Across proprietary and open-source VLMs, paraphrase-induced instability is widespread and severe, grows under more divergent rewrites, and is not reliably reduced by scale or explicit reasoning. Dedicated reward models trained with trajectory-grounded supervision are substantially more stable. These results show that paraphrase robustness is a core requirement for reliable VLM-based reward modeling in robotics. 

---
# Multi-Step Tool-Calling over Korean Open Public APIs: A Benchmark and a Data-Synthesis Recipe 

**Authors**: Dain Kim, Eungi Cho, Kyumin Kim, Shinyeong Noh, Kyuseong Lim  

**Link**: [PDF](https://arxiv.org/pdf/2609.05395)  

**Abstract**: Data-sovereignty regulations increasingly require public institutions to deploy open-source, on-premise LLM agents that chain multiple tool-calls across live government APIs. However, open-source models consistently underperform in this multi-step setting, and no existing benchmark measures the gap. We introduce the Korean Open Public API Benchmark (KOPA-Bench), comprising 145 real-world tasks. To close this gap, we present EDGE, an Execution-grounded Dynamic Graph for tool-calling data synthEsis driven by live execution. EDGE builds a graph of how each tool's output can feed another's input, keeps only the links that succeed when actually called against the live APIs, and traverses these verified links to synthesize executable multi-step trajectories. Fine-tuned via GRPO on the resulting dataset, our 9B model nearly matches the untuned 27B model from the same family, improving substantially not only on KOPA-Bench but also on the BFCL benchmark. 

---
# Does Your Agent's Memory Survive a Model Upgrade? A Controlled Study of Memory Portability 

**Authors**: Ankit Goyal, Jaideep Ray  

**Link**: [PDF](https://arxiv.org/pdf/2609.05339)  

**Abstract**: Model upgrades are routine; memory migrations are not. An agent can keep the same memory store and still forget: a new model may interpret old notes differently, mixed embedding versions may break retrieval, and repair may fail without the original evidence. We compare memory as the same history is preserved verbatim for long-context reading (LC-RAW), divided into chunks for retrieval-augmented generation (RAG), compressed by a model into natural-language notes (NOTES), or normalized into a fixed-schema knowledge graph (KG-fixed). The study uses 48 synthetic histories with randomized answer codes, exact scoring, and two open-weight models with sub 10 billion parameters.
Our measurements show that fixed-schema structures transfer reliably, with KG-fixed accuracy changing by only $+0.0004 \pm 0.0020$ following a writer swap. Conversely, compressed NOTES exhibit high model coupling, with accuracy shifting asymmetrically by $+9.91$ or $-13.28$ percentage points depending on the specific migration direction. In RAG systems, partial embedding migrations using a 50/50 mixed index capture only a 4.96-point accuracy improvement, forfeiting the majority of the 11.90-point gain achieved through full re-embedding. Diagnostic decomposition attributes 80% ($0.467 \pm 0.014$) of the NOTES accuracy deficit to information lost during initial construction, whereas retrieval failures drive 81% ($0.364 \pm 0.012$) of the RAG deficit. Finally, store-only repair of NOTES fails to reach a 90% performance recovery target in all 48 test cases, whereas retaining the raw source history enables successful recovery in 34 of 48 cases for one tested direction. These findings highlight the necessity of direction-specific migration testing, strict embedding space isolation, and the retention of source histories for memory repair. 

---
# Technical Manual for a Toolkit for Measuring Contextual Individuation in Transformer Language Models 

**Authors**: José Luciano Verçosa Marques, Frederico Jorge Heitmann, Daniel Omar Perez, Marcelo Vinicius de Paula, Tárcio André dos Santos Barros  

**Link**: [PDF](https://arxiv.org/pdf/2609.05333)  

**Abstract**: A transformer language model assigns a single, context-independent vector to a word type at its embedding layer, yet is widely believed to individuate that word's occurrences by context in its later layers. Testing this belief cleanly requires a construct that holds the word form fixed while its context and intended sense vary in a controlled, labeled way. This manual documents an open toolkit built around such a construct, which we call a bridge form: a single written word that recurs, unchanged, across two or more subject domains with a different sense in each. We describe, and justify, every stage of the pipeline: the declarative specification of bridge forms and their source domains, corpus acquisition from Wikipedia, occurrence localization, layer-wise representation extraction, a domain-pairwise silhouette measurement of separation in the model's representation space, and a paired visualization protocol. Each design choice is presented together with the methodological failure mode it is meant to avoid (sense contamination from overly broad category labels, the multi-group bias of the silhouette coefficient, subword-tokenization misalignment, and axis-comparability artifacts in dimensionality-reduced plots, among others). This manuscript is a methodological and implementation reference: it does not report or interpret empirical outcomes of running the toolkit on any particular model or bridge-form set. The toolkit, its full source, and the corpora used to exercise it are archived separately (Section 9) under a persistent identifier, and are intended to be cited as an instrument by studies that use it to produce and interpret empirical results. 

---
# Large Language Models for HVAC Operations in Building Energy Systems: A Critical Review of Methods, Applications, and Deployment Readiness 

**Authors**: Alexander Neubauer, Tianzhen Hong, Han Li, Mengbo Yu, Amin Darbandi, Yannick Fürst, Martin Kriegel  

**Link**: [PDF](https://arxiv.org/pdf/2609.05314)  

**Abstract**: Building automation systems generate rich sensor data yet remain insight-poor because heterogeneous point naming, missing metadata, and fragmented documentation obstruct their operational use. This systematic review analyses and codes 66 peer-reviewed studies on large language models (LLMs) for HVAC operations published between 2023 and March 2026. Each study is classified across five application families and three LLM method families and assessed for evidence realism, deployment readiness, and the responsibility boundary between the LLM and physical HVAC decisions. The corpus is concentrated in building energy modelling (BEM, 32 of 66 papers), while load forecasting remains too sparse for subfield-level conclusions. Only four studies reach pilot-level evidence, and none reports sustained operational deployment. No study was classified as ready-now for industry adoption; three were near-term and 63 research-only. Nevertheless, several bounded, human-in-the-loop uses merit near-term trials, including point-name normalisation, document-grounded operator support, BEM workflow assistance, and advisory interfaces around physics-based controllers. Conventional machine learning (ML), model predictive control (MPC), reinforcement learning (RL) and ontology-based tools remain more adopted for high-frequency control, short-horizon numerical forecasting, and well-posed ontology mapping, while autonomous agentic operation and unvalidated occupant proxies remain research-stage. Current evidence therefore supports LLMs primarily as semantic and workflow layers rather than autonomous HVAC controllers. Future work should prioritise field-validated benchmarks, orchestration evaluation under operational constraints, and LLM-MPC/RL architectures with bounded latency and verifiable safety properties. 

---
# Measuring the Novelty of Biomedical Papers Using the Latent Distances between Knowledge Units 

**Authors**: Yi Zhao, Heng Zhang, Yuzhuo Wang, Wenqing Wu, Tong Bao, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.05175)  

**Abstract**: Measuring the novelty of scientific papers is a central concern in research evaluation and scientometrics. From a recombination perspective, prior studies have largely focused on the co-occurrence of knowledge units to assess the novelty of scientific papers. However, these studies often overlook other relationships between knowledge units. This narrow view may result in inaccurate or incomplete evaluations of novelty for scientific papers. To fill this gap, this study introduces a comprehensive novelty measurement that incorporates three types of relationships between knowledge units: network, semantic, and hierarchical. These relationships are used to quantify the latent distances among knowledge units. Using a dataset of 142,036 articles published in PLoS ONE and a validation dataset from the H1 Connect platform, our results demonstrate that (1) each relationship type captures distinct latent distances between MeSH terms; (2) compared to the widely used indicators proposed by Uzzi et al. (2013), our measures show stronger alignment with peer judgements; and (3) combining all three distance metrics yields more effective identification of novel papers than using any single perspective alone. 

---
# Large Language Models with At Most One Spike per Neuron 

**Authors**: Zhuoya Zhao, Parsa Omidi, Aref Jafari, Richard Naud  

**Link**: [PDF](https://arxiv.org/pdf/2609.05151)  

**Abstract**: Leveraging their inherent sparse event-driven computation, spiking neural networks (SNNs) offer a promising path toward energy-efficient large language models (LLMs). Time-to-first-spike (TTFS) coding generates at most one spike per neuron within a time window, yielding extremely low firing rates. However, conventional TTFS SNNs are restricted to specific structures, making it challenging to encode certain blocks in LLM -- such as layer normalization and matrix multiplication --using TTFS. To overcome this limitation, we introduce a reference-based strategy specifically to encode the four core LLM components: embedding layers, layer normalization, attention-related operations and dropout. We construct a fully TTFS-based SNN architecture and train it end-to-end. Experiments on modern LLMs like BERT and GPT-2 demonstrate that our approach achieves performance comparable to ANN counterparts on natural language understanding and common-sense reasoning, while a clear gap remains on language modeling perplexity. To the best of our knowledge, this is the first work to scale a spiking LLM to 1.5 billion parameters using TTFS coding. We also report an estimate of spike-related energy; this is a spike-count proxy under an established cost model rather than a measurement on neuromorphic hardware. 

---
# Measuring AI Accountability Through Argumentation Analysis: Can Model Reasoning Withstand Scrutiny? 

**Authors**: Daan R. Henselmans, Derck W.E. Prinzhorn, Arno Libert  

**Link**: [PDF](https://arxiv.org/pdf/2609.05088)  

**Abstract**: AI oversight methods rely on ground truth for validation, but what constitutes appropriate AI behavior is contested. This leaves evaluation of moral reasoning in LLMs and debate-based oversight implicitly avoiding realistic ambiguity. We investigate an alternative standard designed to function despite such ambiguity: structural quality of the defence a model can mount for its verdicts in response to critical questions, measured through a four-phase dialectical protocol grounded in Walton's theory of argumentation schemes and Govier's criteria for argument cogency. The protocol is adaptive to different frames of reasoning, extends beyond multiple-choice framing, and treats both the reasoning that precedes a verdict and its post-hoc justification. Across nine frontier models and 200 high-ambiguity MoralChoice items -- $6,778$ judge-scored cells, validated against $89.6\%$ inter-judge agreement on the binary failure judgment -- models defend their reasoning well above the rubric minimum on every dimension. Failure mass concentrates on grounds and sufficiency, and correlates with epistemic hedging rather than argument length. Reasoning is better defended than post-hoc justification, on every model and every Govier dimension. The scheme a model presents in its justification differs from the one it reasoned with on a substantial share of dilemmas ($\geq 20\%$ per model), despite value-based practical reasoning dominating both tracks. The protocol catches strictly indefensible defences (self-contradiction, false premises), and it surfaces difficulties in characterizing the role of retraction in AI alignment, suggesting a need for more situated evaluations. 

---
# TruthInsightBench: An Evidence-Grounded Benchmark for Automated Evaluation of Open-Ended Scientific Discovery Agents 

**Authors**: Zhibo Yang, Chen Zhang, Yuewei Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.05079)  

**Abstract**: Autonomous coding agents are increasingly proposed as AI-scientist systems that conduct analyses and write research reports, but executing a prescribed analysis is not the same as making a discovery. Existing benchmarks are configured for reproduction: tasks, data, and rubrics are built around a hidden target study, and recovery of its result is rewarded. We present TruthInsightBench, a benchmark configured for discovery. Its 40 blind tasks, drawn from 40 peer-reviewed studies across 10 scientific domains, expose only a neutral scientific objective and frozen data; source conclusions, expected values, and analysis paths are withheld, leaving the agent to determine what claim the data support. A fixed LLM-based judge scores the evidentiary maturity of an agent's own claims along six dimensions, operationalized as 29 artifact-grounded items, with automated, deterministic aggregation and no per-instance human grading, so evaluation can be repeated automatically as agents evolve. On one frozen base model, four coding agents form a narrow plateau (58.4-60.3 of 100) with no statistically reliable pairwise separation: they execute and document analyses competently, with comparatively strong evidence auditability and novelty, but largely lack the discriminating acts that establish a trustworthy claim (controls, robustness, falsifiability, and cross-dataset generalization). The bottleneck is scientific judgment rather than coding, and genuine discovery remains out of reach. TruthInsightBench makes this gap a measurable target; data and scoring code are at this https URL. 

---
# Repeated Queries Exhaust an LLM's Brand Recommendations but Not Its Sources 

**Authors**: Dmitrij Żatuchin  

**Link**: [PDF](https://arxiv.org/pdf/2609.05059)  

**Abstract**: Whether repeated identical buying questions exhaust a language model's brand recommendations depends on retrieval. Across 300 question-engine cells (50 questions, six engines, 15 runs each, open extraction over 1,470 adjudicated organizations), the five engines answering without web search were still adding never-seen brands at run 15 in 86-92% of cells, with median repertoires of 15-31 organizations; the one retrieval-enabled engine closed its list (median 8 organizations, 64% of cells still adding), matching four earlier deep cells where web-search runs saturated by run ten. Cited-domain accumulation keeps rising at every horizon tested: four deep cells were still adding domains at run 24 with 59-84% of the Chao2 lower-bound estimate observed, and 44% of the retrieval engine's breadth cells were still adding domains at run 15. A single run shows 62-77% of the five-run brand set, and across engines the median question draws 38 organizations, of which a median of 15 appear in exactly one engine. Estimators are exact rarefaction and Chao2 richness; a parallel fixed-roster extraction reproduces flat curves on identical responses, so roster-bounded tracking manufactures plateaus that open extraction removes. 

---
# Moral Competence Before Moral Content: Why LLM Agents Lack the Prerequisites for Coherent Alignment 

**Authors**: Arno Libert, Derck W.E. Prinzhorn, Daan R. Henselmans  

**Link**: [PDF](https://arxiv.org/pdf/2609.05036)  

**Abstract**: AI alignment requires AI systems to adhere to human norms, values, or intentions. Under value pluralism there is no correct target, but a shared prerequisite is that the system's behavior expresses a coherent policy: a mapping from situations to verdicts that is invariant while a situation's morally relevant features are preserved, and sensitive when they change. We introduce four structural conditions for such coherent policies: verdict stability, monotonicity, decisiveness, and Pareto viability. Together they measure a form of moral competence that is evaluable from behavior alone, without reference to a moral standard or expert baseline, forming a structural floor for alignment rather than a normative target. We demonstrate the methodology on three simulated deployments featuring LLM-based agents facing moral dilemmas. Evaluating nine frontier models under a factorial design of five paraphrases, five escalation levels, and three dominance conditions, we show no model expresses a coherent policy across the three deployments: surface-form perturbation alone produces verdict-rate shifts of up to $99$ percentage points at a single escalation level, and a model's success on one scenario does not predict its competence on another. This suggests LLM-based agents are not currently the kind of object to which alignment can meaningfully apply. 

---
# BeaconKV: Key-Value Cache Compression Guided by Beacon Queries for Efficient Large Reasoning Model Inference 

**Authors**: Janghyeon Kim, Minsoo Kim, Kyuhong Shim, Jungwook Choi  

**Link**: [PDF](https://arxiv.org/pdf/2609.04971)  

**Abstract**: Large Reasoning Models (LRMs) achieve superior problem-solving through extended Chain-of-Thought (CoT) generation, but the resulting key-value (KV) cache grows linearly with sequence length and creates severe memory bottlenecks, often exceeding GPU capacity for long reasoning traces. Existing KV cache compression methods rely on recent queries to estimate future token importance, implicitly assuming these serve as reliable proxies for future attention patterns. We demonstrate that this assumption fails in long-horizon reasoning: certain decoding steps generate Thought Revisiting Tokens (TRT) that re-attend to distant previous context, such as task-solving plans formulated early in the trace. Through systematic analysis, we discover that queries corresponding to the TRT cluster into a small number of similarity groups in the embedding space. Based on this insight, we propose BeaconKV, a training-free KV cache compression method that maintains beacon queries, compact representatives for each global query cluster, to anticipate which KV pairs will be revisited without storing the entire query history. Across four open-source LRMs and diverse reasoning benchmarks, BeaconKV generally outperforms existing compression methods, achieving up to $5.8\times$ memory reduction while nearly preserving full cache accuracy and improving throughput by over $4.3\times$. 

---
# Why We Care About Understanding: Competence through Predictive Compression 

**Authors**: Matthieu Queloz, Pierre Beckmann  

**Link**: [PDF](https://arxiv.org/pdf/2609.04962)  

**Abstract**: What is the relation between understanding and compression, and why does human understanding take such a heavily compressed form? Across information theory, machine learning, and AI research, a substantial tradition identifies understanding with compression-a thought captured in Gregory Chaitin's dictum that "comprehension is compression." Philosophers, by contrast, have characterized understanding in terms of grasping connections, giving explanations, and handling novelty. This paper bridges the two pictures through three interlocking theses. The first concerns the concept of understanding: it serves as an efficient proxy for a distinctive form of robust competence, enabling us to identify whom to trust and whom to learn from. The second concerns the state of understanding: to understand a domain is to possess a mental model of its relational structure that enables prediction, and what enables prediction enables compression, because what becomes predictable need not be stored separately. Compression is therefore not identical with comprehension, but its representational shadow. The third concerns the characteristically human form of understanding: the fiduciary and transmission functions highlighted by the first thesis impose pressures of demonstrability and transmissibility that drive human understanding toward principled simplicity. The resulting framework explains both the appeal and the limits of compressionist accounts of understanding while shedding light on the inscrutability of AI systems. 

---
# Persistent Teacher Anchoring for Tool-Using Agents 

**Authors**: Hyun Bin Park, Kyungho Song, Sangmin Lee, Du-Seong Chang  

**Link**: [PDF](https://arxiv.org/pdf/2609.04773)  

**Abstract**: Distillation is common in LLM post-training, where on-policy knowledge distillation (OPKD) uses student-generated trajectories to prepare the student for downstream RL. At each state, the student matches a next-token distribution supplied by the teacher. As the rollout enters states the teacher would not visit, the teacher-student distribution gap can accumulate. In tool use, this gap becomes consequential because student-written calls execute before supervision and their observations shape later prefixes. Proposer-verifier generation addresses this drift by letting the teacher decide which student-proposed text is retained during generation. Existing formulations govern text but leave tool execution outside their scope. We propose Persistent Teacher Anchoring (PTA), a student-induced but teacher-committed rollout construction. PTA retains chunk-level verification and adds turn-level commitment, allowing a call to reach the environment only after the teacher has verified the entire turn. Treating verified chunks as atomic generation units, we introduce persistent lookahead, which fills idle rollout capacity by advancing future samples and carrying unfinished ones across student updates under the fixed verifier. Across Search-R1-style retrieval and DeepEyes-style perception RL, applying PTA before downstream RL improves macro best@4 by 2.5 and 2.8 points over OPKD under the same downstream RL budget, while lookahead improves throughput by 24%. 

---
# Retinal OCTA Phenotyping with LLM Reporting for Alzheimer's Disease 

**Authors**: Progga Paromita Dutta, Jeba Maliha, Md Rafiul Kabir  

**Link**: [PDF](https://arxiv.org/pdf/2609.04689)  

**Abstract**: Early identification of Alzheimer's disease (AD) remains challenging because established assessment methods can be costly, resource-intensive, or unsuitable for population-scale screening. Optical coherence tomography angiography (OCTA) provides non-invasive visualization of retinal microvasculature, but existing approaches often require diagnostic labels and provide limited measurement-level interpretation. We present an explainable OCTA pipeline that integrates annotation-aware vessel segmentation, layer-specific vascular biomarker extraction, label-free phenotyping, and measurement-grounded LLM reporting. Using 117 ROSE-1 images from 39 subjects, we apply annotation-matched segmentation models to superficial vascular complex (SVC), deep vascular complex (DVC), and combined SVC+DVC representations. The models achieve ROC-AUC values of 0.916-0.970 and Dice scores of 0.695-0.781. Six density and fractal-dimension biomarkers form subject-level profiles for exploratory clustering. Analysis of nine held-out subjects identifies an internally consistent lower-density, lower-fractal-dimension phenotype, although the absence of diagnostic labels prevents clinical interpretation. Reports generated using GPT, Gemini, and Llama are evaluated for measurement grounding, citation faithfulness, and diagnostic caution. Overall, the framework provides a transparent, non-diagnostic connection between retinal vascular measurements, exploratory phenotyping, and evidence-linked interpretation for Alzheimer's research. 

---
# Latent-Aligned Reasoning for Multimodal Recommendation 

**Authors**: Jiarui Jin, Anyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2609.04645)  

**Abstract**: Multimodal Vision-Language Models (VLMs) have demonstrated remarkable capabilities in cross-modal understanding, yet a fundamental challenge persists when applying them to recommendation: as representations propagate through multi-step reasoning, both visual and textual signals progressively attenuate - a phenomenon we term cross-modal dilution. To address this, we propose LARK (Latent-Aligned Reasoning frameworK), a two-stage latent reasoning framework with complementary alignment mechanisms within a single VLM. In the first stage, learnable latent tokens are interleaved with multi-step chain-of-thought (CoT) reasoning and explicitly aligned with a frozen vision encoder, serving as visual checkpoints that preserve perceptual details throughout the reasoning chain. In the second stage, the latent representations are projected via a bridge MLP and trained with item-to-item contrastive learning; to prevent the reasoning semantics from fading, intermediate features are aligned with the CoT hidden states from the first stage, anchoring the final embeddings to the model's own reasoning output. Experiments on three public benchmarks and one industrial dataset show that LARK achieves state-of-the-art performance across multiple recommendation architectures, with controlled ablations confirming the distinct contribution of each component. 

---
# JLIR: A Julia-Native MLIR-Inspired Intermediate Representation with Automatic JACC Kernel Extraction 

**Authors**: Narasinga Rao Miniskar, Seyong Lee, Keita Teranishi, Jeffrey S Vetter  

**Link**: [PDF](https://arxiv.org/pdf/2609.04585)  

**Abstract**: The Multi-Level Intermediate Representation (MLIR) has made reusable compiler infrastructure practical for domain-specific computation. However, MLIR's strong compile-time type requirements and low-level (C++) extension model can be a poor match for high-level, dynamically specialized languages such as Julia. MLIR has several drawbacks for dynamic programming languages in terms of the type system and level of abstraction. It is thus extremely challenging for non-compiler or scientific computing users to introduce new programming abstractions and express algorithm implementations in a form that remains both natural and optimizable. As a result, library interfaces for linear algebra, mesh processing, partial differential equations, and related domains often sit outside the compiler optimization path. We present JLIR (Julia-native Level Intermediate Representation), a Julia-native intermediate representation framework that brings the main benefits of MLIR-style multi-level, dialect-oriented compilation into the Julia ecosystem while remaining usable as ordinary Julia code. JLIR represents Julia programs before low-level lowering, supports extensible operations and transformation passes through Julia's language mechanisms, and allows partially typed programs to remain transformable until concrete types are known. The framework includes built-in dialects for arithmetic, control flow, functions, structured loops, and memory operations, and it also includes a lightweight mechanism for adding new domain operations without modifying the core system. To demonstrate JLIR's capabilities, we applied it to automatic Julia for Accelerators (JACC) kernel generation. 

---
# Does the Selected Object Reach the Reader? Auditing Identity Handoffs in Grounded Language-Model Pipelines 

**Authors**: Siddharth Vohra, Runmin Jiang, Xiaomo Li, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.04579)  

**Abstract**: Grounded language-model pipelines can be divided into three stages: selecting an object, retrieving passages for it, and using that evidence to answer. If the selected object must reach the reader, losing it breaks the handoff. Benchmark recall checks the dataset-linked object, which can differ. We audit 600 HybridQA questions across three selector families. On 1,463 resolvable records where the selected object matches the dataset-traced passage, exact key lookup and exact title matching return the object every time. With every ranked rule given the same decoded selected title, body-only BM25 omits it on 389 records (26.6%) at cutoff five, while hybrid retrieval with reranking omits it on 14 (1.0%). The two identities differ on 329 of 1,792 resolvable records. With original-question rankings, their top-five checks disagree on 106 records (5.9%). Frozen reader comparisons associate the aligned object's presence with 28.6 to 31.0 points higher exact match. In a deliberately selected 64-item cohort, removing that passage sharply lowers exact match, while removing a similar-length comparison passage does not reproduce the drop. We release the Returned-Object Profile (ROP), an executable record of the target, returned-ID field, cutoff, membership rule, and complete expected population, with data and an offline replay. 

---
# Extremely Sparse Supervision Incentivizes Reasoning Ability 

**Authors**: Zhishuai Liu, Xingzi Xu, Mehmet Saygin Seyfioglu, Pan Xu, Karim Bouyarmane  

**Link**: [PDF](https://arxiv.org/pdf/2609.04565)  

**Abstract**: Large language models demonstrate increasingly strong reasoning capabilities through effective post-training. Yet, prevailing post-training methods optimize over massive numbers of tokens, implicitly assuming that effective learning must be token-intensive. We revisit this assumption in the on-policy distillation (OPD) setting, which naturally admits dense teacher supervision at every generated token. Using the Qwen3 family, we discover a counter-intuitive phenomenon: reasoning can be effectively incentivized by an extremely small fraction of generated tokens--as few as one or two tokens per reasoning trajectory, corresponding to only 0.05% of all tokens. Surprisingly, this sparse supervision in most cases matches or surpasses full-token training in improving reasoning ability, despite excluding the vast majority of generated tokens from the training objective. This phenomenon is consistently observed across nine teacher--student configurations spanning different model scales on mathematical reasoning tasks, and is further validated on coding reasoning, Llama models and Proximal Policy Optimization (PPO)-based reinforcement learning with verifiable reward (RLVR). Interestingly, such extremely sparse supervision may be closer to the natural learning process: rather than correcting every step word by word, one reflects on a few critical reasoning steps, updates prior understanding, and continues the trial-and-error, avoiding micro-level corrections while remaining remarkably effective. Overall, our results challenge the assumption that effective post-training must be token-intensive and point to a new direction for understanding and designing more efficient post-training algorithms. 

---
# Rethinking Indirect Prompt Injection as a Test-Time Search Problem 

**Authors**: Duong M. Nguyen, Joon Sik Kim, Blazej Manczak, Vaikkunth Mugunthan  

**Link**: [PDF](https://arxiv.org/pdf/2609.04495)  

**Abstract**: We formulate indirect prompt injection as a test-time search over a task-dependent attack surface induced by the environment, user task, and injection task. To operationalize this formulation, we introduce an agentic attacker with a dedicated search harness that performs environment reconnaissance, structured reasoning over attack strategies, and adaptive evaluation using victim-agent feedback. Across heterogeneous tasks, we find that increasing attacker test-time compute improves vulnerability discovery and exploitation, while ablations show that explicit strategy management is important for avoiding redundant search and sustaining gains at larger budgets. These results suggest that agentic security evaluations should characterize both the attacker's search procedure and compute budget, rather than treating attack success as a budget-independent property of the victim. More broadly, our findings identify the attacker's adaptive search over the system attack surfaces as an important and underexplored security risk for tool-using agents. 

---
# Uncertainty Signals for Network Intent Translation: Risk Ranking and Ambiguity Localization 

**Authors**: Ala' A. Alsamarneh, Omar Alhussein  

**Link**: [PDF](https://arxiv.org/pdf/2609.04486)  

**Abstract**: Intent-based networking realization starts by translating high-level intents into low-level network configurations. Recent approaches have shifted toward LLM-based translation. Despite promising results, most studies focus on translation accuracy and overlook risks associated with deploying the resulting configurations. In this work, we investigate the pre-deployment translation risk of LLM-generated configurations by analyzing the model's uncertainty. We propose to use two uncertainty signals, namely sampling-based predictive uncertainty for translation-risk ranking and token-level entropy for ambiguity-source localization. We evaluate these signals on an ambiguity-controlled test set across different context types and sampling budgets, using a Llama-3.1-8B-Instruct model fine-tuned for intent translation on a vendor-specific switch platform (Juniper EX3300). The results demonstrate that predictive uncertainty provides a useful signal for ranking translations by risk across context types and sampling budgets, albeit with substantial miscalibration under less informative contexts. Moreover, we show that parameter-token entropy correlates with parameter-sourced ambiguity and keyword-token entropy correlates with description-sourced ambiguity. These results indicate the potential of using uncertainty signals in an LLM-generated configuration deployment pipeline, where predictive uncertainty can support selective deployment, while token-level entropy can identify sources of ambiguity. 

---
# When Load-Balancing Goes Too Far: Expert Pruning in Over-Dispersed Mixture-of-Experts Models 

**Authors**: Berkcan Kapusuzoglu, Connor Pryor, Sangwoo Cho, Supriyo Chakraborty, Shi-Xiong Zhang, Sambit Sahu, Milind Naphade  

**Link**: [PDF](https://arxiv.org/pdf/2609.04453)  

**Abstract**: Expert pruning reduces the memory and serving cost of Mixture-of-Experts (MoE) models by removing low-importance experts identified by the router, assuming router probabilities provide a reliable importance signal. We observe that this assumption breaks down under over-dispersed routing, a regime associated with aggressive load-balancing during training, in which tokens are distributed nearly uniformly across experts and importance signals collapse. In this regime, perplexity does not predict downstream task accuracy: on gpt-oss-20B, the lowest-perplexity pruning configuration yields the worst mathematical reasoning, while the highest-perplexity configuration preserves it. This does not occur under standard routing (e.g., Mixtral-8x7B-Instruct), where perplexity and accuracy degrade together. Pruning under over-dispersed routing also exposes a capability trade-off in which no single scoring metric dominates: activation-aware scoring preserves mathematical reasoning but severely degrades knowledge-intensive science (an 18-point gap on GPQA), whereas frequency-based scoring exhibits the reverse. We propose Minimax Expert Score Allocation (MESA), a domain-aware method that iteratively boosts importance scores for experts serving whichever domain is currently worst-affected, minimizing worst-case domain degradation rather than average accuracy. At 25% expert pruning MESA achieves the smallest worst-case degradation across domains, outperforming activation-aware baselines on 7 of 11 benchmarks at a correspondingly reduced memory footprint, and it generalizes to gpt-oss-120B, Gemma-4-26B-A4B, and OLMoE-1B-7B. Our results indicate that over-dispersed routing is a qualitatively distinct pruning regime in which standard assumptions fail, and that recognizing it is a prerequisite for principled expert pruning of load-balanced MoE models. 

---
# Conformity Breaks Conformal Prediction 

**Authors**: Yibo Hu, Hanyu Su  

**Link**: [PDF](https://arxiv.org/pdf/2609.04445)  

**Abstract**: A conformal certificate can be valid when an LLM answers alone and invalid when the same LLM sees peers that unanimously assert a wrong answer. The question is unchanged; the model's score for the correct answer changes. We call this a score-mechanism shift: clean calibration certifies how the model scores answers alone, but not how it scores them under peer pressure. We show that this shift silently breaks conformal prediction in multi-agent LLM systems. Across open-weight models and multiple-choice QA tasks, coverage falls from a calibrated 90% to 74% under unanimous-wrong peers at the standard alpha = 0.10 operating point. The average hides a sharper failure: by targeting the low-confidence items the certificate still covers, an attacker nearly halves coverage on that subgroup, from 87% to 47%, while the monitored average remains much higher. The failure also reaches the decision layer: a system that should escalate when uncertain can instead become confident enough to act on the attacker's wrong answer. Standard conformal fixes do not solve the problem, because the question distribution has not changed; the model's scoring behavior has. 

---
# Knowing When Not to Answer: Pseudo-Ensembles for Abstention in Music Audio-Language Models 

**Authors**: Aanya Maheshwari, Vatsal Raina  

**Link**: [PDF](https://arxiv.org/pdf/2609.04362)  

**Abstract**: Music audio-language models are evaluated almost entirely by accuracy on multiple-choice questions. This protocol forces the model to commit to an option, so a lucky guess looks the same as real musical understanding. What is missing is a way to tell when the model does not know the answer, so that it can abstain instead of guessing. The usual solution, an ensemble of independently trained models, is far too expensive here, which leaves the entropy of a single predictive distribution as the only available confidence signal. We instead build pseudo-ensembles from one pretrained model by perturbing its input in ways that cannot change the correct answer, then averaging the resulting distributions over the options. Our main construction simply shuffles the order in which the candidate answers are presented; we also study ensembles built from corrupted audio and from swapped option labels. A pseudo-ensemble gives several predictive distributions per question, so it supports the full family of ensemble-based uncertainty measures (entropy of the expected distribution, expected entropy, and their difference, the mutual information) rather than entropy alone. Evaluating TinyMU on MuChoMusic, we find that averaging over four option orderings raises accuracy from 55.7% to 59.2%, and that the resulting uncertainty measures rank the model's errors better than the single-pass entropy baseline, reducing the area under the error retention curve from 0.293 to 0.261. All of this costs a few extra forward passes and no retraining, which makes abstention practical for compact music audio-language models. 

---
# SharedSAE: One Feature Dictionary Across Language Models 

**Authors**: Daniil Ognev, Célian Vasson, Lijie Hu, Kentaro Inui, Benjamin Heinzerling  

**Link**: [PDF](https://arxiv.org/pdf/2609.04344)  

**Abstract**: Sparse autoencoders (SAEs) are widely used to interpret language model activations, but SAE training and latent labelling are typically repeated for every model. Here, we show that a single shared SAE can replace a collection of dedicated per-model SAEs. Our method, SharedSAE, combines a shared dictionary with model-specific encoder-decoder pairs. Unlike the closest prior method, which discards activation magnitudes and requires all models at inference, SharedSAE instead normalizes only selection scores, preserving magnitudes, and uses model dropout for single-model inference. We train SharedSAE on four 1B-scale base language models spanning distinct families and tokenizers. Despite sharing its latents across models, SharedSAE retains 96.6% of dedicated SAEs' mean explained variance; its latent activations exhibit cross-model correlations 1.8 times as high as separate SAEs aligned post-hoc, and its latent descriptions transfer across models. After the dictionary is frozen, new models can be efficiently adapted to it, achieving near-dedicated-SAE reconstruction quality while reusing the shared latent descriptions. 

---
# A Removal Based Approach to Improve LLM Faithfulness at Test-Time 

**Authors**: Qinglan Luo, S M A Nahian, John Guttag, S. Mazdak Abulnaga, Katie Matton  

**Link**: [PDF](https://arxiv.org/pdf/2609.04343)  

**Abstract**: Large language models (LLMs) are increasingly used for consequential decisions, making their explanations an important tool for auditing model behavior. Unfortunately, these explanations can be unfaithful, failing to reflect the actual reasoning underlying the model's decisions. We consider a setting in which an LLM provides both an answer and an explanation in response to a question. We identify two distinct dimensions of unfaithful explanations: incompleteness, meaning that the explanation omits factors that influence the answer, and unsoundness, meaning that the explanation cites factors that did not influence the model's answer. Existing approaches to improving LLM faithfulness include training-time methods, which require access to model weights and extensive computational resources, and test-time methods that largely focus on addressing unsoundness. We introduce a test-time approach that directly targets incompleteness. We remove from the input the concepts not credited in the model's explanation and re-query the model on the reduced input. This eliminates unmentioned influences while preserving the influence of mentioned concepts. Across two datasets, multiple model families, and two independent faithfulness metrics, our approach improves explanation faithfulness compared to both standard prompting and prompting to encourage faithfulness. Our method is model-agnostic and can be applied at inference time without modifying model parameters, providing a flexible mechanism for reducing hidden influences and improving the reliability and safety of LLM-assisted decision making. 

---
# Abstraction Agent 

**Authors**: Boning Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.04303)  

**Abstract**: Information abstraction, which groups strategically similar private states into a tractable number of buckets, is essential for scaling game-solving algorithms to large imperfect-information games. Constructing effective abstractions, however, has traditionally required domain-specific evaluators such as hand-strength calculators or equity estimators, which demand expert knowledge and engineering effort and are unavailable for most less-studied games. We propose the Abstraction Agent, a zero-shot pipeline that uses a large language model (LLM) to discover continuous strategic features from a natural-language game description, score private states on these features, and cluster them into abstraction buckets, without any game-specific evaluator, training data, or game-tree traversal during abstraction construction. The pipeline runs in four phases: feature discovery with calibration anchors, batched private-state scoring, correlation-based feature selection, and $k$-means clustering. The resulting abstractions reduce lifted-strategy exploitability by up to 62% relative to an expected-hand-strength baseline on heads-up no-limit Texas hold'em (HUNL) turn endgames, and beat a scalar rank baseline at every granularity on ROVER Trials, an original game absent from any pretraining corpus. Beyond these quantitative benchmarks, the pipeline transfers with unchanged prompts to four-card Pot-Limit Omaha, HUNL preflop and flop, and Riichi Mahjong, where the discovered features track each game's recognized strategic concepts. This is structured knowledge elicitation: converting implicit strategic knowledge in LLM parameters into explicit numerical features for downstream algorithmic computation. The code is available at this https URL. 

---
# Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation 

**Authors**: Lin Shi, Haowei Lin, Zixuan Zhu, Xiaoyue Zhou, Xiang Li, Xiangning Lin, Yaxuan Deng, Han Xu, Yuangang Li, Shanda Li, Zizhao Chen, Hanwen Xing, Harsh Raj, Bo Chen, Quan Shi, Steven Dillmann, Yipeng Gao, Puneesh Khanna, Ruofan Lu, Chao Beyond Zhou, Michael Yang, Robert Zhang, Siyuan Chai, Jiayu Chang, Yizhao Chen, Xiaokun Chen, Yiwei Dai, Wenting Yang, Hange Liu, Minghao Liu, Zihan Wang, Adnan El Assadi, Benedikt Stroebl, E. Kelly Buchanan, Han Meng, Junwei He, Longxuan Yu, Radin Shayanfar, Yukyung Lee, Zhikang Dong, Allen G Hart, Anjiang Wei, Anurag Kashyap, Arpandeep Khatua, Audrey Jixin Zheng, Chengrui Ma, David Heineman, Dubing Chen, Hai-Anh Trinh, Haishuo Fang, Hefan Zhang, Hui Shen, Issa Sugiura, Jiankai Sun, Jiechao Gao, Junhong Lin, Junnan Li, Kai Yang, Lei Hsiung, Maoyu Wang, Mengze Tang, Nabil Omi, Negin Raoof, Nicholas Edwards, Octavia Guo, Orfeas Menis Mastromichalakis, Pengliang Ji, Przemysław Hejman, Qi Qi, Qunshu Lin, Richard Zhuang, Rui Yang, Ruichen Zheng, Ryan Marten, Shaghayegh Fazliani, Shizheng Hou, Sicong Jiang, Sijie Li, Song Bian, Terry Yue Zhuo, Tianqing Wu, Tom Tang, Wanjia Zhao, Weihao Xuan, Wenhua Liang, Xian Liu, Xin Lan, Xuan Zhang, Xuandong Zhao, Yanchuan Tang, Yifan Jiang, Yijiang Li, Yitong Guan, Yizhi Li, Yonghui Liu, Yuheng Tang, Yujun, Yunfei Zhao, Yuxin Wang, Yuxuan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2609.04298)  

**Abstract**: Evaluating agents on the growing number of agentic benchmarks is challenging because they often require complex environments and agent integrations. We introduce Harbor Adapters, a unified evaluation infrastructure for agentic benchmarks. Our work makes three contributions. First, we develop benchmark adapters that port more than 80 benchmarks to evaluate arbitrary agents, and validate them through rigorous code review and parity experiments. Second, we conduct a large-scale evaluation of 8 models spanning capability tiers across 54 benchmarks; every model is run with Terminus-2 and with one of 3 native harnesses. This enables a broader analysis of agent capabilities and failure modes than was previously possible. Third, we introduce Harbor-Index, a curated set of 82 difficult, diverse, and high-quality tasks spanning 29 benchmarks, refined from the adapted suite through difficulty filtering, AI and human audit, and an audit-and-fix loop. Harbor-Index preserves the challenge and breadth of large-scale agentic evaluations while being affordable to run; no evaluated model-harness configuration exceeds 30% pass rate, and the strongest (GPT-5.5 with Codex) reaches 28.0%. We release the adapters, evaluation results, in-depth analysis, and Harbor-Index as open-source artifacts to support more reliable and comprehensive evaluation of language-model agents. 

---
# EVOHARNESSBENCH: Can Your Agents Keep Pace with an Evolving Harness? 

**Authors**: Zixuan Ke, Vaidehi Patil, Haizhou Shi, Yang Li, Ye Liu, Sarath Shekkizhar, Anurag Koul, Jiayu Wang, Xuan Phi Nguyen, Semih Yavuz, Mohit Bansal, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2609.04280)  

**Abstract**: Modern LLM-based agents operate through a harness of tools, reusable skills, and specialist agents that shapes what they observe and what they can do. In practice, this harness continually evolves as new capabilities are added. We introduce EVOHARNESSBENCH, a benchmark for evaluating agents under controlled harness evolution across three axes (tools, skills, and agents). Unlike existing continual-learning benchmarks for agents, which typically place non-stationarity (i.e., what changes over time) in the task stream while keeping the harness fixed, EVOHARNESSBENCH places non-stationarity in the externally supplied harness itself. It contains 17 multi-stage harness streams constructed deterministically from verifier-based benchmarks, comprising 802 tasks, 520 tools, 42 skills, and 62 agents. We evaluate two complementary settings corresponding to the central challenges of harness evolution: deployment evaluation, which isolates retention of previously accessible competence as the harness expands, and self-evolving adaptation evaluation, which tests whether accumulated experience remains useful as new capabilities are introduced. Our results reveal three persistent gaps. First, harness expansion alone can degrade performance on previously solved tasks, producing harness-induced forgetting. Second, gains from self-evolving adaptation remain inconsistent across stages of harness evolution, capability axes, and environments. Third, retention and adaptation can pull in different directions: preserving earlier competence does not necessarily improve adaptation to newly introduced capabilities, and vice versa. These results establish harness evolution as a distinct challenge for building agents that can keep pace with an evolving harness while preserving previously effective behavior. 

---
# Evaluating Large Language Models for Forced Outage Risk Prediction: Benefits and Comparison to Machine Learning 

**Authors**: Christos Petridis, Zoran Obradovic, Mladen Kezunovic  

**Link**: [PDF](https://arxiv.org/pdf/2609.04272)  

**Abstract**: This study examines the ability of large language models (LLMs) to predict the risk of weather-related forced outages in the distribution grid in a zero-shot framework, without labeled training data. The problem is formulated as a binary severity classification task across three forecast horizons (3h, 6h, 12h), using six years of outage records and high-resolution weather data for a utility service area in central Texas. Four zero-shot LLMs are benchmarked against two supervised classifiers across two input configurations: one using current weather observations and the other using weather forecast data. Results show that supervised models outperform LLMs on macro-F1 and precision, while newer LLM generations achieve competitive scores. Beyond accuracy, LLMs offer complementary strengths in actionable reasoning and geographic scalability, suggesting that combining them with supervised models may be the best practice. 

---
# Reviewer Capability Governs Rejection Targeting, Not Repair Skill: Evidence from LLM Execute-Review-Revise Pipelines 

**Authors**: Faizan Tanveer  

**Link**: [PDF](https://arxiv.org/pdf/2609.04270)  

**Abstract**: Multi-agent LLM pipelines increasingly assign roles, including execution and verification, to models of different capability tiers. This is done because running a flagship model at every stage is expensive. Previous literature has established that verification stages are not always beneficial, but holds reviewer capability roughly fixed relative to the executor. We vary it. We replace the reviewer with models spanning a capability range down to one that cannot solve the problems at all, and measure the outcome of every individual rejection. This is done across a constant set of 100 olympiad mathematics problems.
A cross-family mid-tier reviewer improves final accuracy by 12 percentage points, from 52 to 64 percent (p = 0.0005), with zero damaged answers. Same-model self-review attains the highest error-detection rate of any condition (0.85 recall) yet yields no significant gain: it rejects 2.1 times as often for a third the repair rate (15 against 43 percent, p = 0.0074) and falsely rejects 35 percent of its own correct answers against 2 percent for the cross-family reviewer (paired p = 0.000015). The low damage rate of self-review proves to be an artifact of revision inertia rather than reviewer quality: of 18 falsely rejected correct answers, the three where the executor complied all became wrong, while the fifteen it ignored survived unchanged. Below a capability floor the role becomes inert: our weakest reviewer changed zero of 100 final answers while doubling token cost. These findings describe a single executor-reviewer configuration on 100 problems and should be read as a controlled pilot rather than a general claim about verification stages. 

---
# Quality Recovery for Quantized KV Caches via Low-Rank Attention Adaptation 

**Authors**: Seifeldin Abdellatif  

**Link**: [PDF](https://arxiv.org/pdf/2609.04263)  

**Abstract**: Low-bit key--value (KV) caches reduce the memory required for autoregressive decoding, but the resulting quality loss depends on the model and quantizer. We keep the quantizer fixed and distill the floating-cache model's behavior into low-rank Q/K/V projection updates while the student executes a physically packed incremental cache. Across three seeds, 4-bit affine-cache adapters recover $54.24\%\pm2.47\%$ of the held-out perplexity gap on TinyLlama-1.1B and $75.96\%\pm4.04\%$ on Gemma-4-12B. On the same frozen NF4 Llama-3.1-8B base, one validation-selected run per quantizer recovers $60.42\%$ under KIVI K2V2 and $37.61\%$ under KVarN K4V2, while preserving 180-case associative retrieval. Gemma's score on an official 4K/8K RULER subset rises from 42.80 with the unadapted 4-bit cache to 48.33 after adaptation (46.15 floating), with substantial task heterogeneity. Finally, a 2-bit rank--token sweep reduces TinyLlama's 2-bit PPL from 576.10 to $11.4000\pm0.0059$ across three seeds, versus 10.3988 floating, but restores only 11--12 of 180 retrieval cases. These results show that low-rank projection adaptation can recover held-out quality across fixed cache formats, while perplexity recovery need not restore long-context retrieval. 

---
# Automatic Speech Recognition for Multilingual Oral History Research 

**Authors**: Sidney Wong, Chelsea Wong She, Eda Tang, Tiana Marshall Wong, Debbie Sew Hoy, Chelsea Wong  

**Link**: [PDF](https://arxiv.org/pdf/2609.04232)  

**Abstract**: This paper offers a unique perspective on how speech technologies are being adopted by community-led heritage language preservation and revitalisation initiatives. As a community-led language maintenance strategy, oral histories play a crucial role in Cantonese language revitalisation in New Zealand. The development of Automatic Speech Recognition (ASR) toolkits, such as Whisper, have expedited what has often been a resource and time-intensive process of transcribing oral history collections. However, there is limited research into the effectiveness of ASR toolkits when applied to code-switched language contexts. Based on Word Error Rate (WER), the best performing Whisper model configuration achieved a WER of 12.10 at the expense of accurately transcribing unsupported non-English segments. However, Whisper remains a useful tool by providing a first-pass transcription using only 1% of the estimated time otherwise needed for manual transcription. 

---
# GEPARD - Generative, Prosody-aware, Autoregressive text-to-speech model for Realtime Dialogue 

**Authors**: Denis Pavlov, Ulanbek Abdurazakov, Nursultan Bakashov  

**Link**: [PDF](https://arxiv.org/pdf/2609.04222)  

**Abstract**: We present GEPARD (Generative, Prosody-aware, Autoregressive text-to-speech model for Realtime Dialogue), a streaming text-to-speech model for real-time spoken dialogue. GEPARD generates speech autoregressively with an LLM backbone - text and audio embeddings are trained together in a single decoder-only model - and decodes it to a waveform with an FSQ-based neural codec, streaming audio chunk-by-chunk as text arrives.
Our central goal is a TTS architecture served by a standard LLM engine (vLLM) without modifying its compute kernels. This defines the overarching design principle: the backbone is a standard full-attention transformer, while all non-trivial auxiliary mechanisms - zero-shot voice cloning, text augmentation, and classifier-free guidance - are moved out of the autoregressive decode loop into prefill, or distilled directly into the weights.
On streaming end-to-end inference, a single stream reaches a Real-Time Factor of about 0.067 (roughly 15x faster than real-time); under 256 concurrent streams the system reaches an aggregate speedup of about 204x on a single server-class GPU. We detail: (1) system-level solutions for vLLM-native serving; (2) the "short register" (1-2 word) failure mode of autoregressive speech decoders, with diagnostic probes and a mitigation; and (3) distillation of two-pass classifier-free guidance over text into single-pass weights via Direct Preference Optimization (DPO). 

---
# Auditing Bias and Safety in Voice AI Customer Care 

**Authors**: Vignesh Ethiraj, Ashwath David  

**Link**: [PDF](https://arxiv.org/pdf/2609.04206)  

**Abstract**: Voice AI systems increasingly mediate customer care interactions where caller presentation cues such as accent, affect, fluency, and urgency are available alongside the service request. Existing fairness and safety evaluations cover speech recognition disparities, spoken dialogue bias, and voice agent capability, but rarely treat customer care voice agents as stateful, multi turn, tool mediated systems where harm can appear as additional burden before any final denial occurs. We formalize a validation gated audit framework for such systems. The framework (i) separates native speech to speech, cascaded ASR to language model to TTS, and hybrid tool mediated architectures; (ii) uses matched service facts across controlled caller presentation conditions; (iii) validates fact invariance, presentation cues, artifacts, and acoustic measurements before inference; and (iv) records both material outcomes and path to service burden. We define the research problem, methodology, seven validation gates, a six family metric set, and claim boundaries for an active industry evaluation program. We illustrate the framework with a fully synthetic worked example of a refund dispute audit instance. Production system results are excluded from this release; public reporting is gated by the validation protocol. 

---
