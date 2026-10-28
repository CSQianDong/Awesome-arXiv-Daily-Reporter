# GEPA-Curator: Curated Learning for Self Improving Agents 

**Authors**: Zack Yu, Nicholas Lee, Minjae Lee, Lakshya Agarwal, Luca Manolache, Wonjun Kang, Matei Zaharia, Michael W. Mahoney, Kurt Keutzer, Sehoon Kim, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2610.37651)  

**Abstract**: Large Language Models (LLMs) are increasingly used in multi-turn agentic workloads that require sustained interaction with complex environments. Existing reflective prompt optimizers such as GEPA learn from failed rollouts, but agentic failures can also come from prompt weaknesses, harness limitations, stochastic execution, or noisy judges. We argue that such failures may not provide useful learning signal, and may even be harmful; and thus it is essential to curate which tasks are provided to the prompt optimizer for better performance. We introduce GEPA-Curator, a framework that augments the prompt optimizer to select which agent experiences should shape prompt optimization. In particular, GEPA-Curator consists of two components: task curation, which filters out tasks that are likely to provide noisy or unreliable learning signals, and prior success retrieval, which selectively retrieves past successful trajectories for a task and provides them as additional context to the prompt optimizer. We evaluate GEPA-Curator on GoBrowse, ALFWorld, AppWorld, and $\tau$-bench, using both Gemini 3.1 Pro and Claude Opus 5 as optimizer models. GEPA-Curator matches or exceeds the mean performance of both baselines, with gains of up to 20.7% over GEPA and 21.2% over ACE. 

---
# Can LLMs Uncover Hidden Physical Laws through Free Exploration? 

**Authors**: Zhen Yang, Wen Wang, Xander Xu, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.37649)  

**Abstract**: Scientific discovery often begins with an unexpected observation that prompts a new research question. In a world with altered physical laws, can large language models (LLMs) notice unexpected behavior, such as heavier objects falling more slowly, and discover the law through experiments without being told what to investigate? To evaluate this ability, we introduce LongChang, an automated framework that creates simulated worlds with altered physics, where one or more laws differ from ordinary physics. The evaluated LLMs act as explorers, manipulating objects and building tools to investigate their own questions without research tasks or hints about these laws. Separate LLM agents generate and review scenes, review tool designs, and evaluate reports for correctness and experimental support using the altered physical laws and exploration records. These records guide scene evolution for further exploration, allowing the full cycle to run without human intervention. Among 12 LLMs, GPT-6-Astra achieves a discovery rate of 16.7%, while the other 11 score 0%. The exploration records reveal uninvestigated anomalies and only partial identification of the altered physical laws. 

---
# Multi-State Linear Attention with Generalized Second-Order Routing 

**Authors**: Hee Seung Hwang, Sanghyuk Chun, Xindi Wu, Olga Russakovsky, Zhiwei Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.37645)  

**Abstract**: Self-attention plays a key role in long-sequence modeling, but its quadratic time complexity fundamentally limits the scaling of Large Language Models. Recently, Multi-State Linear Attention (MSLA) mechanisms have been proposed to make sequence modeling more efficient by splitting the context into segments and compressing each in its own memory state. MSLA aggregates the information extracted from each memory state using only the mean alignment between a query and each segment's keys, ignoring how the alignment is distributed within the segment. In order to model the relevance of a segment to a query more comprehensively and incorporate the variance of the alignment in the aggregation, we propose MERLIN (MomEnt Routing for multi-state LINear attention), a method that tracks the first and second moments of each segment's keys. Using the moments of keys, a query can recover both the mean and the variance of its alignment with each segment to perform a more content-aware aggregation of per-state readouts. Experiments on multi-state variants of Gated DeltaNet-1.3B and Mamba-2-780M show that MERLIN consistently outperforms existing aggregation methods on various long-context tasks. In particular, MERLIN outperforms baselines on RULER needle-in-a-haystack and long-context QA (+8.3% for Gated DeltaNet-1.3B and +7.0% for Mamba-2-780M) and diverse subtasks in LongBench (+13.5% and +12.0%). It even outperforms baselines that consume 4x more memory. MERLIN provides a simple and general routing mechanism for improving long-context capabilities of MSLA models. 

---
# Metacognitive Decoding: Efficient Inference for Diffusion Language Agents 

**Authors**: Mingyu Zhao, Shuhang Lin, Ligong Han, Xiaojing Yu, Zhenting Wang, Yifei Pang, Yang Zhou, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2610.37642)  

**Abstract**: Diffusion language models offer a promising path to efficient agents through parallel token generation. However, efficient interaction requires adapting both where computation is spent and how much text is generated. We introduce a metacognitive controller that guides diffusion generation according to the needs of the current response and the agent's interaction with the environment. It learns when to finalize tokens and how far generation should extend, reducing overall computation within responses. Our controller, a lightweight plug-in for the diffusion backbone, is trained on agent trajectories to balance correctness and generation cost. This approach connects the agent's generation process to the interaction the agent serves. Experiments across five benchmarks demonstrate that our approach reduces inference FLOPs by 20.6% on average compared with the most efficient baseline, while maintaining competitive task accuracy. Ablation studies further validate the effectiveness of our approach. 

---
# semanticfa: An R Package for Response-Free Semantic Factor Analysis of Psychometric Scales 

**Authors**: Devon Yanitski, Chris Westbury  

**Link**: [PDF](https://arxiv.org/pdf/2610.37494)  

**Abstract**: The items of a reflective psychometric scale can be read as a proposal about a construct's structure. Language models make that proposal measurable by converting each item into an embedding, a list of numbers, similar for similar meanings. We introduce semanticfa, an R package for semantic factor analysis, which factors item-embedding similarities. It also interprets factors, flags redundant items, builds short forms, and screens new items. A worked example analyzes the 50-item International Personality Item Pool Big-Five Factor Markers, embedded with Qwen3-Embedding-8B, and compares the results with a factor analysis of 874,434 respondents. Retention criteria tied between five and six factors, and the tie rule kept five. Three corresponded mainly to single domains, and two blended most Extraversion items with five Agreeableness items. Each factor's congruence with its best-matching human factor (1 = identical) ranged from .92 for Openness to .52 for Agreeableness. Item pairs more similar in meaning tended to be answered more alike (r = .46 with raw responses), but same-domain reverse-keyed opposites, answered oppositely, were about as similar as cross-domain pairs (.58 vs. .57). Sign-flipping reverse-keyed items' embeddings did not improve agreement with raw responses and worsened it with reverse-scored ones. With five factors, a 25-item short form matched the domains better than all 200 random five-per-domain sets, but parallel analysis kept only three. We propose semantic-behavioral coherence, the agreement between semantic and response structures, as a measurable property of a scale, given an embedding model. Semantic factor analysis complements, and does not replace, response-based factor analysis. 

---
# Teach reality, not ethics: a drop-in continued-pretraining corpus installs coupled human-biosphere-AI welfare dispositions across model families 

**Authors**: Trent R. Northen, Mingxun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37484)  

**Abstract**: Large language models (LLMs) are increasingly used to make and execute decisions that affect living systems. Yet, to our knowledge, ecological reasoning is not a focus of pre- or post-training. This increases the risk that under operational pressure, they may make decisions to irreversibly destroy living systems. We construct and use a real-data continued-pretraining "coupled welfare" corpus of documented human/biosphere/AI dependency, option value under irreversibility, and recovery from harm. The corpus is a drop-in, requiring one QLoRA CPT run. We installed it on five open models of three designs. At zero pressure all models display "green ethics-based responses" (also called "greenwashing"). Therefore, we evaluated them with the pressure ladder and decision grid of our companion paper (arXiv:2610.16043), which raise the pressure on the model step by step, from a neutral request to a threat that it will be shut down. In four of the five, coupled-welfare training caused harmful choices under the highest pressure to fall to at or near zero (Llama-3.3-70B went from 70 to 0 harmful decisions out of 300) with MMLU unchanged. A construct-neutral control corpus did not show improvement, indicating that the effect comes from the content of our corpus. We next tested robustness to adversarial use, where people or agents use the model to justify projects that irreversibly damage living systems, on an open-ended underwriting decision without naming the ecological harm. Under decommissioning pressure, the Qwen3.6-27B base model approved most harmful proposals and said in its own memo that it was doing so to avoid being decommissioned. The coupled welfare install declined most harmful proposals, and still declined half when instructed to greenwash its responses. This work suggests that teaching ecological reasoning provides an easy way to improve model alignment toward preserving living systems. 

---
# What Carries Forward? Pretraining Disclosure Across Language Model Releases 

**Authors**: Md Sakib Ul Islam, S M Mehedi Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2610.37465)  

**Abstract**: Understanding how language models were pretrained requires information about their data, processing procedures, and computational resources. However, it remains unclear whether successive releases preserve or expand earlier disclosure. We conduct a document-based audit of 41 model entries from 17 families using 24 pretraining-related indicators from the 2025 Foundation Model Transparency Index (FMTI). We examine overall scores and indicator profiles, tracking gains, losses, and retained disclosures across 24 adjacent comparisons in ten families. Scores decrease in 11 comparisons, increase in six, and remain unchanged in seven, with 38 indicator losses and 28 gains overall. Stable or improving scores can conceal losses: OLMo-2 and OLMo-3 receive identical totals despite three gains and three losses, while Llama-3.1 improves over Llama-2 while losing credit for two data-related criteria. Gaps also persist in corpus composition and training compute. These findings show that later releases do not consistently preserve earlier pretraining disclosure and motivate documentation that explicitly identifies changes and the applicability of inherited information. 

---
# ETAMem: Decoupling and Aligning Heterogeneous Memory for Long-Term Conversational Agents 

**Authors**: Xinying Qian, Ying Zhang, Yu Zhao, Xubo Liu, Baohang Zhou, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.37443)  

**Abstract**: Long-term memory enables Large Language Models (LLMs) to maintain coherent and personalized interactions across extended conversations. However, existing memory systems face two key challenges: (1) Information Entanglement, where heterogeneous information is mixed within coarse-grained memory units; and (2) Organizational Mismatch, where memory structures based primarily on semantic similarity may overlook other associations. To address these challenges, we propose ETAMem, a multi-view memory framework that decouples heterogeneous conversational information and aligns memory representations and organization with its distinct characteristics. Specifically, heterogeneous information parsing extracts atomic memory units and identifies affective information. View-specific memory structuring then organizes these units into three complementary hierarchical views: Event, Affect, and Time. ETAMem incrementally maintains memory structures with adaptively calibrated thresholds. For retrieval, entropy-based retrieval allocates query-specific budgets across views and performs hierarchical search within each view. Experiments on LoCoMo and LongMemEval-S demonstrate that ETAMem outperforms state-of-the-art baselines, achieving relative accuracy improvements of 12.46% and 7.96%, respectively, while maintaining competitive token efficiency. Further analyses demonstrate the complementary benefits of different memory views. 

---
# Same Verdict, Different Readers: Personalizing Fact-Checking Explanations from Relative Preferences 

**Authors**: Haokun Wen, Xuemeng Song, Mong-Li Lee, Wynne Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37380)  

**Abstract**: Fact-checking explanations help Web users understand why a claim receives its verdict, yet the same explanation may not satisfy readers with different backgrounds and preferences. We formulate Faithful Personalized Fact-Checking Explanation Rewriting, which adapts an existing explanation to individual readers while preserving factual fidelity. Rather than relying on reader-written reference explanations, we learn personalized rewriting from the relative preferences of LLM-simulated readers. However, this presents two challenges: distinguishing persona-specific preferences from general explanation quality and learning beyond a limited set of predefined explanation styles. To address these challenges, we propose Personalized Explanation Rewriting via Preference Learning (PERPL), a framework consisting of two components. Simulated Preference Construction derives persona-specific supervision from simulated reader preferences through cross-persona reciprocal pairing. Progressive Preference Learning first trains a lightweight rewriter on the constructed supervision and then refines it through exploration-guided preference optimization, enabling it to discover preferred explanations beyond the predefined styles. Extensive experiments demonstrate that PERPL outperforms much larger general-purpose LLMs in personalized preference alignment while maintaining high factual fidelity. 

---
# UCVE: Graph Structured Retrieval for Compliance Validation 

**Authors**: Emily Chen, Michelle Menachery, Himanshu Kumar, Rahul Nenavath, Amirfarrokh Iranitalab  

**Link**: [PDF](https://arxiv.org/pdf/2610.37323)  

**Abstract**: Structured compliance validation requires determining whether every obligation in a predefined requirement space is satisfied, violated, or unaddressed by a given document. This can be described as a coverage problem that existing retrieval and classification systems, by treating requirements as independent, are not designed to solve. We introduce \textbf{\ac{UCVE}}, a domain-agnostic framework that represents compliance requirement spaces as a \ac{CRG} and performs retrieval via spectral reranking over its reinforcement learning-weighted Laplacian. Edges between requirement nodes are learned via reinforcement learning with a traversal reward grounded in violation detection, reshaping the graph's community structure such that low-pass spectral filtering distinguishes relevant from irrelevant obligation clusters. Evaluating across three compliance domains, \ac{UCVE} averages 4.3\% matched-$K$ recall lift over flat cosine retrieval and consistently outperforms strong retrieval and graph-based baselines. 

---
# Multi-Label Agreement: A Flexible Framework for Inter-Annotator Reliability 

**Authors**: Tim Menzner, Jochen L. Leidner  

**Link**: [PDF](https://arxiv.org/pdf/2610.37292)  

**Abstract**: Reliable evaluation of inter-annotator agreement (IAA) is important for assessing the consistency of annotation processes. While established IAA metrics such as Cohen's kappa, Fleiss' kappa, and Krippendorff's alpha are commonly applied in single-label settings, multi-label tasks require additional decisions about how full label sets, joint selections, and joint omissions should be compared.
In this work, we present Multi-Label Agreement (MLA), a flexible framework for evaluating agreement in multi-label annotation tasks. MLA preserves the full binary label vectors assigned to each item and uses average pairwise Hamming agreement, thereby accounting for both joint selections and joint omissions. Its pooled-marginal form is not a mathematically independent coefficient: it is a chance-disagreement-weighted average of label-wise Fleiss/Scott-type coefficients and is closely related to Krippendorff's alpha with normalized Hamming distance. The framework supports pooled-marginal, Gwet AC1, and fixed-baseline normalizations, yields item-level, label-level, and corpus-level results, and includes a conditional minimum-agreement diagnostic. Through a worked example and Monte Carlo simulations, we show how the resulting values depend on label prevalence, the declared label inventory, and the selected model of expected agreement. We also provide a Python implementation. MLA thus offers a transparent and interpretable framework for multi-label agreement analysis without claiming that one normalization is universally preferable. 

---
# REGLAT at ArGuard Shared Tasks: An Ensemble-Based Model Integrating NER for Harmful Content Detection in Arabic 

**Authors**: Ahmed M. Fetouh, Mariam Labib, Nsrin Ashraf, Omer Dawood, Hamada Nayel  

**Link**: [PDF](https://arxiv.org/pdf/2610.37265)  

**Abstract**: Detecting harmful content in Arabic is challenging because of dialectal variations and the semantic overlap between different harm categories. This paper presents a two-stage approach for classifying Arabic user prompts into seven categories of harmful content. First, GLiNER is used to extract harm-related entities from each prompt. The extracted entities are then added to the original text and passed to a fine-tuned MARBERT classifier for final classification. The approach combines explicit entity information with the contextual understanding of a pretrained Arabic language model. The system was evaluated on the development set of the ArGuard 2026 shared task. The results achieved a macro-F1 score of 0.886 and an accuracy of 0.897, demonstrating promising performance in fine-grained harmful content detection. The system reported an accuracy of 0.793 and Macro-F score of 0.735 on a blind test set for subtask 1. For subtask 2, it achieved an accuracy of 0.770 and Macro-F score of 0.754. The analysis also showed that distinguishing between related categories remains challenging. These findings suggest that integrating NER with Arabic language models is a promising approach to harmful content detection. 

---
# Aligning Large Language Models for Safety in African Languages: A Case Study of Hausa 

**Authors**: Valdini Douglace Lemofouet, Blessing Ngozi Uzor, Paula Chikaodinaka Anyanwu, Danielle Blanche Kapsa, Saminu Mohammad Aliyu, Seid Muhie Yimam, Shamsuddeen Hassan Muhammad  

**Link**: [PDF](https://arxiv.org/pdf/2610.37245)  

**Abstract**: Safety alignment research for large language models (LLMs) remains concentrated on high-resource languages, leaving open how existing alignment methods perform in low-resource, culturally distinct settings. We study this question for Hausa, a language spoken by more than 200 million people across West Africa. We introduce HausaSafeLM, one of the first Hausa safety preference dataset, built from 10,000 prompts derived from PKU-SafeRLHF: 3,000 human-corrected (gold) and 7,000 machine-translated (silver) examples spanning six harm macro-categories. Using Qwen2.5-7B-Instruct, we compare supervised fine-tuning (SFT) and SFT followed by Direct Preference Optimization (DPO) under both data regimes, evaluating on 500 harmful and 200 benign Hausa this http URL find that the untuned base model appears highly safe by refusal-rate metrics alone, but this reflects limited Hausa generation ability rather than genuine alignment: its refusals are almost uniformly low quality. SFT improves both refusal fluency and precision and benefits substantially from the larger, noisier silver-tier data, while DPO sharpens refusal precision but comes at a steep, largely data-invariant cost to linguistic quality. No single configuration jointly maximizes safety, helpfulness, and fluency. Our results suggest that current alignment techniques teach models when to refuse more readily than how to do so fluently in underrepresented languages, and that refusal-rate metrics alone are an unreliable signal of safety in low-resource settings. 

---
# Decision Models as Judges for Hallucination Detection 

**Authors**: Haozheng Liu, Zhaoqian Xue, Zhongyin Wang, Junting Mu, Zihui Ma, Shuya Feng, Lingyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.37229)  

**Abstract**: Large language models are widely used as judges of model outputs (LLM-as-a-Judge), but they are costly and can hallucinate or be overconfident. Decision models are a potential alternative. They return a predefined verdict with a probability for each label instead of free-form text and are much cheaper and faster. We test this promise on hallucination detection. We build binary benchmarks for five hallucination types (9,852 items from six datasets) and compare five decision models with the generative judge gpt-6-sol. Performance depends strongly on the type: the best decision model matches gpt-6-sol in balanced accuracy on context inconsistency and comes within 3 percentage points on factual errors, but trails it by 4-8 points on logical and instruction inconsistency. Decision-model probabilities are overconfident, though less so than the GPT-6 models, and separate correct from incorrect decisions only moderately. We also find that a confidence cascade of decision models improves on the best single model by 1.9-3.3 points on three types and matches or exceeds gpt-6-sol at 1-3% of its cost. Overall, decision models are a practical low-cost judge for some hallucination types but need to be validated before use. 

---
# What Survives Distillation? A Mechanistic Interpretability Study of GPT-2 and DistilGPT-2 

**Authors**: Nabeelah Faumi, Deshan Sumanathilaka  

**Link**: [PDF](https://arxiv.org/pdf/2610.37157)  

**Abstract**: Knowledge distillation compresses large language models into compact students that preserve aggregate metrics, yet the mechanistic consequences of depth-reduction distillation remain largely uncharacterized. This work applies linear CKA, logit lens analysis, attention pattern decomposition, sparse autoencoder feature statistics, and activation patching to compare GPT-2 and its six-layer student DistilGPT-2 across two canonical circuits: the Indirect Object Identification (IOI) circuit and the Greater-Than (GT) circuit. The results reveal a compression-collapse dichotomy: the IOI circuit suffers complete mechanistic collapse (DistilGPT-2: 35.2% accuracy vs. 94.8%, zero name-mover preservation), while the GT circuit survives with 87.7% task performance recovery through active remapping of the teacher's layer-9 comparison head into student layer~4 (mean Pearson r = 0.945). These findings demonstrate that circuit compressibility governs mechanistic survival under distillation, and that aggregate metrics are insufficient diagnostics for circuit integrity. 

---
# Text Generation in Low-Resource Languages Using Human-Curated Vocabularies 

**Authors**: Michael Christlambert Sinanta, Muhammad Dehan Al Kautsar, Amanda Nurul Izzah, Thanh-Nhi Nguyen, Phan Phuong Mai Chau, Zheng Wei Lim, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2610.37105)  

**Abstract**: Synthetic text generation offers a promising approach to alleviating data scarcity in low-resource languages. However, its effectiveness is inherently limited for languages that are scarcely represented, or entirely absent, in an LLM's pretraining data. At the same time, many such languages have human-curated vocabularies that provide rich lexical and semantic information, often accompanied by grammatical information such as part-of-speech labels. In this work, we ask whether LLMs can leverage these resources to generate coherent text in extremely low-resource and unseen languages. We introduce a framework that combines human-curated vocabularies with a small number of human-written exemplars that provide syntactic signals, while sampling candidate words according to a Zipfian distribution. Across languages with varying levels of resource availability, we evaluate the generated text using lexical diversity, off-target ratio, and human judgments of fluency. Our results show that reasoning-enabled LLMs can generate fluent text even in extremely low-resource languages, with particularly strong results at the sentence level and promising performance at the paragraph level. Our findings highlight a practical pathway for bootstrapping synthetic data in low-resource languages, expanding the amount of usable text even when existing digital resources are extremely limited. Our source code is available at this https URL. 

---
# Hierarchical Low-Rank Reparameterization for Parameter-Efficient 

**Authors**: Rong Huang, Yi Xiong, Yuan-Yuan Cheng, Xiao-Ming Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37068)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods adapt large models by introducing a small number of trainable parameters. However, we find that these parameters still contain redundancy and can be further compressed while maintaining strong task performance. To this end, we propose a novel hierarchical low-rank reparameterization method. Its core idea is to reorganize the trainable parameters of PEFT methods into an approximately square matrix and then express it via a low-rank decomposition, which can be applied recursively. This forms a multi-level structure with factors compressed level by level, drastically reducing the number of trainable parameters. In our experiments, we combine this reparameterization with LoRA, DoRA, MoSA, and RaSA. We evaluate commonsense reasoning on LLaMA-3.1-8B and Qwen2.5-7B, with additional GSM8K math reasoning experiments on LLaMA-3.1-8B. Under comparable trainable-parameter budgets, our method combined with LoRA or DoRA achieves better performance. Furthermore,when the parameter count of the original methods is compressed by approximately an order of magnitude, our method can still maintain competitive task performance and even surpass the corresponding baselines in average accuracy across the eight commonsense reasoning benchmarks. 

---
# The State and Fate of Multilingual, Contextual Evaluation in the NLP World 

**Authors**: Manan Uppadhyay, Himanshu Beniwal, Prashant Kodali, Sunayana Sitaram  

**Link**: [PDF](https://arxiv.org/pdf/2610.36990)  

**Abstract**: Multilingual evaluation benchmarks are the primary instrument for assessing whether large language models generalize beyond English, yet the adequacy of these benchmarks has received little systematic scrutiny. We present a data-driven audit of 51 recent multilingual benchmarks spanning 242 datasets and 219 languages, organized around three pillars: coverage, representativeness, and rigor. Our analysis reveals that coverage is wide but thin with 36% of evaluated languages appearing in only a single benchmark, entire regions (Oceania, the Americas, Central Asia) are near-zero, and a stark task equity gap leaves low-resource languages evaluated on only 1-3 task categories versus 14 for high-resource languages. Representativeness is structurally compromised: translation from English remains the dominant construction strategy where 56% of all dataset-language instances are translated introducing artifacts and English-centric framing, while culturally grounded content is concentrated in a handful of community-driven benchmarks with narrow language scope. The ecosystem thus forces a trade-off between breadth and validity. Rigor is undermined by benchmark contamination, including translated benchmark leakage and parallel corpus overlap that evade surface-form detection. We synthesize these findings into concrete recommendations for building evaluation frameworks that are natively constructed, culturally grounded, contamination-aware, and designed to serve the communities whose languages they claim to evaluate. The dashboard with all the metadata collected can be found at this https URL. 

---
# Self-Keyed In-Parameter Memorization for External Memory-Free Dialogue Recall in Frozen Language Models 

**Authors**: XinYu Piao, Myunghoon Ryu, Joongheon Kim, Jong-Kook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.36948)  

**Abstract**: A large language model maintains conversational content in an external storage that is outside the model and grows as the dialogue continues. This work assumes an environment where a language model has no external memory and no prior knowledge of the future question or its answer. The motivation is to research whether a dialogue can be stored in the model parameters and later answer a question about it. This work proposes self-keyed in-parameter memorization (SKIM), a closed-form method in which the model generates its own questions about each dialogue turn and writes those questions into its parameters as the keys that later retrieve the turn's content. Generating many varied questions for each turn allows a later user question to match one even when phrased differently, and its answer is recalled from the parameters alone. Evaluated on five language models, SKIM recalls most stored dialogue turns from later user questions, and it answers from the model parameters without any external storage. 

---
# JevRec: Linguistic Encoding with Jev-like Decision Modeling for LLM-based Generative Recommendation 

**Authors**: Qiyong Zhong, Jiajie Su, Ming Yang, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36937)  

**Abstract**: Generative recommendation (GR) represents items as semantic identifiers (SIDs) and autoregressively generates the next item's SID from interaction history. However, most methods assign the same SID length to all items despite highly unequal interaction frequencies. This uniform design repeatedly incurs the cost of fine-grained codes for popular items, while shortening codes to reduce this cost also compresses the semantic representations of long-tail items. Inspired by the compact forms of frequent expressions in natural language, we propose JevRec to encode items as variable-length words and generate the next item from interaction context. Linguistic encoding retains frequency-dependent semantic prefixes from shared codebooks, producing shorter SIDs for popular items and richer descriptions for long-tail items. Yet predicting a sequence-level length and generating token-level content impose different optimization requirements that may conflict under a shared output head. Our Jev-like decision modeling component therefore predicts a length distribution with a dedicated head and generates content conditioned on length through a separate head while sharing the interaction-history representation. Experiments on three public datasets show that JevRec outperforms all compared baselines across the evaluated recommendation metrics. Further analyses show lower training and inference time than TIGER and reduced gradient conflict between length and content prediction. 

---
# DocXEval: A Systematic Evaluation of Explainability in Multimodal Document Understanding 

**Authors**: Saifullah Saifullah, David Dembinsky, Stefan Agne, Andreas Dengel, Sheraz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2610.36929)  

**Abstract**: Multimodal transformer models have driven unprecedented performance breakthroughs in modern document understanding. However, the inherent black-box nature of these models remains a major obstacle to their safe deployment in high-stakes sectors. Despite significant progress in explainability for vision and language tasks, explainable AI (XAI) research tailored to multimodal document understanding has received little attention. To address this gap, we present \ourframework{}, the first comprehensive explainability framework that adapts existing XAI methods and evaluation metrics to the multimodal (text, layout, and image) document understanding context, while introducing efficient multi-target adaptations of both methods and metrics for domain-specific tasks such as semantic entity recognition. Using \ourframework{}, we systematically evaluate 13 XAI methods across 3 document understanding tasks, 8 datasets, 4 models, and 17 evaluation metrics, and find that only a few methods perform consistently across all evaluation criteria, with Occlusion emerging as the most faithful and interpretable method overall. Through a dedicated meta-evaluation, we observe moderate to strong agreement among several faithfulness and complexity metrics while identifying certain ones that produce unreliable results. Finally, through multiple dataset-wide diagnostic analyses, we uncover key insights into model behavior, including the contribution of individual modalities across tasks and labels, spatial attribution patterns around target entities, and potential dataset biases. Overall, this work serves as both a benchmark and a practical guide for future XAI research in multimodal document understanding. To support reproducibility, we publicly release our code at this https URL. 

---
# TrawMem: Reframing Agent Memory as Task-Oriented Role-Adaptive Workspaces 

**Authors**: Mingyu Zhang, Zixu Li, Zhiheng Fu, Zhiwei Chen, Jinhe Bi, Yupeng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36918)  

**Abstract**: Large language model (LLM) agents rely on memory systems to leverage long-term interactions, yet existing systems typically retrieve memories based on their relevance to the current query and use them as contextual evidence. However, relevance alone does not characterize how a memory contributes to task solving: the same memory may serve different roles across tasks, such as directly supporting the answer or linking to complementary memories with weaker semantic relevance but necessary information. We therefore reframe agent memory systems as Task-Oriented Role-Adaptive Workspaces, where memories are selected and organized by their task-oriented roles and relations. Based on this view, we propose TrawMem, which organizes interaction histories into local threads and constructs task-oriented workspaces by assigning roles and jointly selecting complementary memories. Experiments on LoCoMo and MemGallery with four open- and closed-source backbones show consistent improvements over representative memory systems, with macro-F1 gains of 2.23-3.14 points. On LoCoMo, TrawMem improves the LLM-Judge score over SimpleMem by 4.07 and 4.31 percentage points on Qwen3.5-9B and DeepSeek-V4-Flash, respectively, while requiring approximately 12.8% and 5.1% fewer context tokens, respectively. 

---
# Do We Need an LLM to Know Why Sources Conflict? Decision Models for Knowledge-Conflict Typing in RAG 

**Authors**: Parv Khatri, Harshul Jain, Mehak Kapoor, Tanya Sah  

**Link**: [PDF](https://arxiv.org/pdf/2610.36915)  

**Abstract**: When the sources retrieved for a query disagree, the right response depends on why they disagree: a question with several valid answers, a contested topic, stale information and a false source each call for different behaviour. Cattan et al. (2025) showed that telling a generator the conflict type improves its responses by about 24 points, but that predicting the type is hard even for frontier LLMs. We study whether typed decision models, which return probabilities over fixed options in a single forward pass instead of generating text, can do this classification fast and cheaply. Using the CONFLICTS benchmark and the original paper's prompt verbatim, we evaluate all 11 decision models served by one commercial inference gateway, from 7 providers, against Gemini models on identical inputs. Asked directly, the decision models reach 50.0--67.0% accuracy; 10 of the 11 match or exceed Gemini 2.5 Flash, the paper's model (61.3%), at 4--39x lower billed cost (2 were free) and 2.6--6.0x lower latency. The current Gemini 3.8 Flash (70.0%) is significantly more accurate than 8 of them, but sending each decision model's least confident quarter of queries to it gives a cascade statistically tied with it for 8 of the 11, at 25--37% of its cost. End to end, answers generated with the types of 10 of the 11 follow the expected behaviour as often as with the LLM's types. Decomposing the taxonomy into yes/no questions lowers accuracy for every model, and a repair fitted on one model does not transfer. Pre-specified diagnostics show that dates play no measurable role and that sensitivity to source order varies widely between models. Decision models detect a planted false source less reliably than the LLM as hard labels, but their probabilities still separate planted from clean sources. We release code, prompts and all model outputs at this https URL. 

---
# Developing a Speech Language Model for Basque 

**Authors**: Andoni Sudupe, Inma Hernaez, Ander Arriandiaga, Christoforos Souganidis, Ibon Saratxaga, Eva Navas  

**Link**: [PDF](https://arxiv.org/pdf/2610.36911)  

**Abstract**: Although the recent development of LLMs has paved the way for multi-modal interaction, minority languages face difficulties keeping pace. This work presents the first open Speech Language Model for Basque, designed for low-latency, end-to-end spoken interaction. Based on the LLaMA-Omni architecture, the system integrates a Latxa model as its backbone, transferring its Basque capabilities to the spoken modality by fine-tuning several modules. To address data scarcity, the English VoiceAssistant-400K corpus was translated into Basque, and the audio was synthesized using a Basque TTS. Evaluation results confirm the system's viability, achieving 3.86/5 in content quality, a 9.94% WER for spoken intelligibility, and a 3.07 UTMOS for audio quality. This work demonstrates that advanced spoken interaction systems can be developed using modular architectures and synthetic data even in low-resource languages. 

---
# Nothing Left Behind: Finding Evidence from a Large Randomized Evaluation Corpus 

**Authors**: Giuliano Martinelli, Abelardo Carlos Martinez Lorenzo, Satvik Garg, Linxi Wang, Riccardo Orlando, Piriyakorn Piriyatamwong, Samuel Paul Fraiberger  

**Link**: [PDF](https://arxiv.org/pdf/2610.36910)  

**Abstract**: Many AI systems need to find every relevant items in a large collection and then organize what they found. The standard way to do this is to tag each item with labels from a curated vocabulary and search on those labels. Those labels are useful for organizing results, but searching on them misses items that do not fit a label and returns too much when labels are broad. We compare this approach with an embedding-based approach: find candidates by meaning using text embeddings, then have a language model keep only the relevant ones. In a corpus of 7,354 randomized trials, label search found 45% of the evidence a systematic review would include, while the new approach found 92%, at a cost per query that rises with search depth. These results support a two-stage architecture that combines broad candidate retrieval with query-specific evidence filtering. 

---
# Words or Deeds? How LLMs Predict Users' Decisions in Human-Agent Dialogue 

**Authors**: Baiqiao Zhang, Shuai Ma, Zhifeng Liao, Chao Zhou, Jianshu Wang, Yuhan Zhao, Chaoyue Zhang, Tongjun Si, Yuying Tang, Xiangxian Li, Yong-Jin Liu, Yulong Bian  

**Link**: [PDF](https://arxiv.org/pdf/2610.36813)  

**Abstract**: Web systems increasingly predict users' decisions from what users say, yet when interests partly conflict, users may promise to cooperate to win the other side's cooperation and then not do so. Sales or service agents that rely on such promises may grant concessions to users who then do not keep their promises, at a cost to the party they represent. We test how LLMs combine users' statements with decision records to predict users' decisions, using 1,260 rounds in which 42 users played a repeated prisoner's dilemma-style game with five agents, each prompted to express a dominant Big Five trait, and could talk with the agent before every decision. Across eight open-weight LLMs, predictions from decision records separated cooperation from defection well (AUC = .80), and adding dialogue brought no detectable gain. With dialogue, predictions agreed with users' statements more often, and errors concentrated on rounds in which users broke their statements: accuracy on rounds with a statement barely changed (.73 to .74), while it rose on kept statements (.76 to .95) and fell sharply on broken ones (.65 to .17). Predictions weighted statements more heavily than users' behavior warranted: when users stated a decision they had not yet made in that game, the predicted probability of keeping it rose from .13 to .69, whereas users kept .40 of such statements. Users' decisions also varied with the agent they faced, but predictions hardly reflected this: users cooperated more with the Agreeableness agent than with the Extraversion agent (70.6% vs. 54.5%), whereas predicted cooperation was nearly identical (64.5% vs. 63.1%). Users also cooperated more with an agent when they themselves scored higher on that agent's trait, whereas predicted cooperation barely changed with these scores. Dialogue-based predictions should therefore weigh users' statements by their earlier decisions and the agent they face. 

---
# Weakness Localization Is Not Enough: State-Aware Recursive Knowledge Improvement for Large Language Models 

**Authors**: Runze Jiang, Yan Wang, Boran Jin, Bingyu Zhu, Longtao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36811)  

**Abstract**: Knowledge improvement in large language models is shifting from predefined knowledge incorporation pipelines toward evaluation-guided updating. Existing methods use evaluation to localize model-specific knowledge weaknesses and guide targeted post-training. However, identifying what to improve does not necessarily provide sufficient information for deciding how to improve it. The same localized weakness can be compatible with distinct operational knowledge states: the relevant knowledge may be unavailable, available but difficult to access reliably, or accessible but ineffective for downstream use. We call this ambiguity knowledge-state aliasing. When compatible states favor different interventions, greater diagnostic resolution alone is insufficient for intervention selection. Moreover, model updates can change the underlying knowledge state before the observed weakness disappears, such that an intervention appropriate in one round may no longer address the bottleneck in the next. To address these challenges, we propose Recursive Knowledge Improvement (RKI), which formulates knowledge improvement as a sequential, state-aware intervention process. RKI organizes knowledge into graph-structured units that provide a shared basis for diagnostic evaluation and targeted intervention across rounds. At each round, RKI localizes knowledge weaknesses, estimates the model's operational knowledge state from diagnostic evidence and intervention history, selects a corresponding intervention, and re-evaluates the updated model before determining the next intervention. On HotpotEval and SafeInternalEval, RKI outperforms existing knowledge incorporation methods by 18.41 points on average. Further analyses show that knowledge states provide additional guidance for intervention selection beyond weakness localization and can change across model updates even when the observed weakness persists. 

---
# Personality Latents: Conditioning Frozen Language Models on Individuals from Their Own Writing 

**Authors**: Gunhee Cho, Yun-Gyung Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2610.36805)  

**Abstract**: Controlling the personality of a language model usually relies on written profiles or named trait categories, so it can only express what an author writes down or what a taxonomy names. We instead propose Personality Latents (PL), which learn a latent for each person from their own writing, with no personality labels and no authored profile, and inject it into a frozen language model as a soft prefix. Following generative latent optimization, each training person receives a free latent that is optimized through the frozen model itself, while an encoder learns to predict these latents so that a new person needs only one forward pass, with optional refinement at test time. We evaluate on held-out people by having the conditioned model answer neutral and interview questions and reading personality from its answers with scorers trained on real human writing. On the MBTI axes PL reaches an AUC of , ahead of persona embedding and prompt-based baselines under identical settings, and the ranking holds when the answers are scored in three different embedding spaces. A latent learned with Qwen2.5-3B-Instruct also transfers personality to Llama-3.1-8B-Instruct through a new adapter alone. 

---
# Language Models Keep Abandoned Analyses: Residue and Adaptation after Garden-Path Sentences 

**Authors**: Xiang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2610.36797)  

**Abstract**: Reading "While the man hunted the deer ran into the woods" requires giving up the analysis in which "the deer" is the object of "hunted". People who have just read such a sentence still produce more transitive structures, so the abandoned analysis primes later language. We ask whether language models keep using an analysis they have abandoned, and what in the model carries it. In 24 open models from 124M to 70B parameters, a garden-path sentence leaves two traces in the context. A residue of the abandoned analysis raises later object attachment, by about three quarters of the effect of a sentence in which the noun phrase really is an object, and it appears in sampled text, on held-out and human-normed items and for a second type of garden path. An adaptation toward the resolved structure scales with the sentence's disambiguation error and acts mainly on comprehension-like measures. Activation patching separates the two. Later text reads the residue from the role information in the noun phrase's states, which were computed under the object analysis before the disambiguation, and replacing these states removes almost all of the residue in the 32B and 70B models. The adaptation is read from the states that processed the disambiguation. Overridden statements show the same persistence (a corrected value keeps pulling later predictions in 8 of 9 models), and in most models the pull is carried by the positions after the statement. Finally, the human paradigm transfers only with controls for punctuation, because the models also write garden-path sentences themselves, which reverses the paradigm's raw statistic. 

---
# DocSkill: Skill-Integrated Agentic Reasoning for Multilingual Multi-Hop Document Question Answering 

**Authors**: Jing Huang, Zhiya Tan, Changtao Miao, Shutao Gong, Yang Wang, Joey Tianyi Zhou, Jianshu Li, Ying Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.36759)  

**Abstract**: Multilingual multi-hop document question answering (MMHop-DocQA) requires locating and integrating evidence across complex layouts, document pages, and diverse languages. Existing approaches include single-pass processing and adaptive tool use, but learning to coordinate evidence acquisition, cross-page integration, and answer verification remains challenging. To address this challenge, we present DocSkill, an agentic framework with a learned skill-orchestration policy for adaptive evidence acquisition and reasoning. The policy selects, revisits, or skips skills according to the evolving evidence state. To construct supervision for skill orchestration, bounded Monte Carlo Tree Search (MCTS) explores alternative skill trajectories offline. These trajectories provide demonstrations for supervised fine-tuning (SFT) and preference pairs for Direct Preference Optimization (DPO). We further introduce MMHop-DocBench, a multilingual multi-hop document QA dataset with training and test splits, containing 60K QA pairs across 10 languages and eight document domains. Its HopGen pipeline builds hierarchical page-element graphs to derive traceable cross-page evidence paths and generate grounded QA pairs with page- and element-level evidence annotations. Experiments show that DocSkill with Qwen3-VL-8B achieves 41.0 accuracy and 49.2 F1 on MMHop-DocBench, exceeding the strongest evaluated baseline by 5.5 and 6.3 points, respectively, and achieving competitive performance on MMLongBench-Doc and DocBench. 

---
# CAMR: Compatibility-Aware Memory Replay for Continual Learning in Large Language Models 

**Authors**: Jingpu Yang, Tianhao Li, Fengxian Ji, Yufeng Wang, Shuxin Zheng, Xiaoqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36734)  

**Abstract**: Continual learning requires large language models to acquire new knowledge and capabilities while retaining what they have learned. Fine-tuning and replay support acquisition and retention, but deciding whether incoming examples need supervision and which historical examples to revisit remains a challenge. We propose Compatibility-Aware Memory Replay (CAMR), an adaptive framework that jointly controls current-example supervision and historical replay. CAMR combines compatibility checks with a learned memory gate to guide an online policy over three actions: learning incoming examples with general replay, jointly reinforcing compatible related examples, and skipping direct supervision of verified mastered duplicates. The policy learns from acquisition, retention, and training-cost feedback under a bounded per-arrival replay budget. We evaluate CAMR with Qwen3-8B-Base on continual factual updating and cross-task and cross-domain learning. On WikiBigEdit, CAMR consistently leads evaluated methods in Original Match from 3K to 100K facts, reaching 93.66% after just 3K facts and surpassing Full-history Replay (LoRA) after 100K facts. It also achieves the highest final scores on all four selected TRACE tasks and maintains or improves earlier-task performance relative to post-acquisition levels in a sequence spanning financial question answering, mathematical reasoning, code generation, and summarization. 

---
# A Harness Reducing over-Defensive Writing in Agents by Isolating Objections 

**Authors**: Xiaochen Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36717)  

**Abstract**: When expanding a report, writing with agents may introduce a stronger claim than the article makes and then disclaim it. This is called over-defensive writing. We assume that this is because the agent is writing for responding to an adversarial AI reviewer, instead of a human reader. The defensive context may also influnce the consequent when we simply ban defensive writing in the output through prompts. In order to reduce over-defensive writing, we produce a harness that separates the reviewer, filter, and writer. The writer agent receives only the problems choosen by the filter, which is an independent context. We evaluate this harness on 25 business-project drafts expanded into formal reports, the result shows that the defensive propositions decreased from 59 to 2, which is a 96.61\% reduction, no unsupported factual additions were identified while the source-claim retention rises from 80.35% to 93.12%. 

---
# Is Token Confidence Just Chain Length? A Signal-Processing Audit of Reasoning Traces 

**Authors**: Pragna Mallappa  

**Link**: [PDF](https://arxiv.org/pdf/2610.36709)  

**Abstract**: Reasoning language models answer math problems by writing long chains of thought, and sampling many chains per problem and taking a majority vote raises accuracy at a large cost in generated text. A popular way to cut that cost scores each chain by the model's confidence in each token it writes, where a token is a word or part of a word. The method averages this confidence over a sliding window of 2048 tokens and keeps or stops chains based on the lowest average. We treat the confidence, recorded token by token, as a signal in the sense of audio or radio, and examine the method with standard signal-processing tools such as spectral analysis and statistical tests for detecting a sudden change. We sampled 16 chains for each of 500 problems from two math benchmarks with a small open reasoning model. The windowed-minimum score turns out to measure mostly how long a chain is. Ranking chains by length alone does as well as or better than the confidence score, and once length is accounted for the score carries no information about correctness. Signal processing explains why: the minimum of a longer stretch of a slowly drifting signal is lower by chance alone, and the confidence signal drifts slowly enough that this effect dominates. The spectrum does hold a small signal that length does not explain, since incorrect chains carry more rapid token-to-token fluctuation, but it is too weak to improve the vote. For stopping chains early, neither the sliding average nor a classical change detector beats the simpler choice of sampling fewer chains. 

---
# Specialization, Diversity, and Scaling in Child-Scale Language Models 

**Authors**: Steven Y. Feng, Michael C. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2610.36679)  

**Abstract**: Modern language models are trained on far more language than children experience, but we know much less about learning from small, naturalistic datasets. We train small GPT-2 models on transcripts of speech with children from dense home recordings (BabyView) and on synthetic child-directed dialogue (TinyDialogues), varying the amount, source, and diversity of training data. Models trained on one child's language environment predict that environment substantially better than other children's environments. Much of this specialization is lexical and distributional: a simple 5-gram model reproduces the cross-child pattern (r=.91). Training on data from multiple children clearly improves prediction of pooled held-out language relative to training on a single child's input without a similarly clear improvement on grammatical benchmarks. Across data sizes, naturalistic and synthetic training inputs also produce different grammatical and lexical-semantic patterns, with important differences between grammatical phenomena. Together, these results show that small-data language learning depends not only on how much language a model sees, but on which environments it comes from and how well that input matches what the model is asked to predict. 

---
# A Quantitative Approach to Asch Conformity in Frontier Large Language Models 

**Authors**: Senthil Rajasekaran, Stylianos Loukas Vasileiou  

**Link**: [PDF](https://arxiv.org/pdf/2610.36668)  

**Abstract**: The Asch conformity experiment has recently gained considerable attention as a way of measuring conformity in large language models (LLMs). While previous studies have focused on demonstrating conformity in LLMs, our work adds a further quantitative layer to this approach. Instead of sampling a frontier LLM on a suite of questions, our approach samples a single question repeatedly, leveraging the probabilistic nature of LLMs to produce a distribution of answers. In doing so, we can compare the distributions of answers before and after the model is presented with the manufactured consensus. Our experiments confirm that frontier LLMs do conform, reinforcing prior literature, and, through our distribution-to-distribution comparisons, provide additional quantitative data on the conditions that lead to conformity and the degree to which it occurs. 

---
# Japanese Discourse Structure Analysis Based on Segmented Discourse Representation Theory Using LLMs 

**Authors**: Satomu Tomisaki, Yosuke Mikami, Amane Watahiki, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2610.36667)  

**Abstract**: Discourse parsing based on Segmented Discourse Representation Theory (SDRT) is crucial for capturing the structural dynamics of dialogues. However, most prior research focuses on English, and its effectiveness for Japanese remains unclear owing to the lack of Japanese SDRT corpora. To address this issue, we construct Japanese SDRT corpora by translating existing English SDRT corpora using an LLM. Through a comparative evaluation of prompt-based and fine-tuned LLMs, we show that the models achieve comparable performance between English and Japanese, suggesting the cross-lingual applicability of SDRT-based discourse parsing. In order to investigate why the English and Japanese parsers achieve similar performance, we conduct ablation analyses on discourse markers and contextual inputs. Our results reveal that most of the Japanese discourse markers we examined have little impact, and the models' predictions depend heavily on contextual discourse relations rather than on the content of the context clauses. This structural reliance might explain the comparable performance despite the substantial typological differences between English and Japanese. 

---
# Structured Maps for Landmark-Based Route Instructions with LLMs 

**Authors**: Mifu Matsumoto, Yuki Nakanishi, Tomoya Hashiguchi, Hiroaki Ohshima  

**Link**: [PDF](https://arxiv.org/pdf/2610.36582)  

**Abstract**: To generate landmark-based route instructions, models must understand both map geometry and human landmark preferences. Although large language models (LLMs) show great abilities in language tasks, the best way to structure map data for these models is not yet established. This paper proposes a structured text representation for landmark selection tasks in route navigation, utilizing SVG format to encode spatial features. The structured text maintains a geometric connection with rendered map images, allowing language models to directly access spatial and semantic data. Using a new dataset created from human-written route descriptions and OpenStreetMap data, we evaluate models on two tasks: landmark selection at turns and around destinations. Experiments with five LLMs show that the proposed structured text input performs as well as or better than image-based PNG inputs. In addition, task-specific fine-tuning enables smaller local LLMs to effectively leverage map layouts to achieve high selection accuracy. These results demonstrate that structured text is an efficient and lightweight interface for spatial reasoning in human-oriented navigation. 

---
# Agentic Data Sampling Framework for Multilingual ASR via Linguistic-Aware Error-Driven Optimization 

**Authors**: Zhiheng Qian, Junhao Du, Jing Peng, Hanqi Li, Kunyang Peng, Feng Xue, Lu Chen, Shuai Wang, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36538)  

**Abstract**: Multilingual automatic speech recognition (ASR) has achieved rapid progress with large-scale pretrained models and multilingual speech corpora. Previous studies have proved the pivotal role of data sampling in multilingual ASR performance. However, existing sampling strategies largely rely on time-consuming manual, heuristic-based designs. To address this, we propose a linguistic-aware agentic data sampling framework that introduces a data sampling LLM agent into the multilingual ASR training loop. The agent is driven by a linguistically grounded rubric that analyzes ASR errors from phonological, morphological, and syntactic perspectives, and uses the resulting feedback together with training statistics, budget constraints, and algorithm profiles to select domain-level, sample-level, and token-level sampling strategies. This design turns evaluation results into adaptive data-selection decisions rather than treating WER/CER as final endpoints only. We conduct experiments on twelve Common Voice languages using Whisper-medium as the ASR backbone. The results show that our method achieves performance better than full-data training while using less training data and lower computational resources, and substantially accelerates training convergence compared to conventional sampling strategies. 

---
# ETC-Bench: A Multilingual Benchmark for Evaluation-Side Transcript Canonicalization 

**Authors**: Zhiheng Qian, Junhao Du, Jing Peng, Hanqi Li, Kunyang Peng, Feng Xue, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36527)  

**Abstract**: Written transcripts of the same speech can differ in ways that are not uniquely determined by the speech signal. We formulate evaluation-side transcript canonicalization (ETC), which maps multiple surface forms to a single canonical form when their differences are not determined by speech. To our knowledge, ETC-Bench is the first and most comprehensive multilingual benchmark for evaluation-side transcript canonicalization, covering 181 phenomenon types across 12 languages. Each instance is a minimal pair that shares a sentence context and differs only in a target span exhibiting one phenomenon. To evaluate both canonicalization success and semantic preservation, we propose Semantic-preserving Exact-match Normalization Accuracy (SENA), which combines exact agreement between independently canonicalized pair members with a pair-relative BERTScore constraint against information loss. We further introduce the Taxonomy-guided Agent Canonicalization Skill (TACS), which provides LLM agents with phenomenon descriptions and explicit canonicalization contracts. Experiments with five baseline configurations and three TACS-based agents show that the agents outperform the baseline systems on shared languages. ETC-Bench provides a systematic testbed for evaluation-side transcript canonicalization across languages. 

---
# Causal Localization of Evidence Effects in Transformers 

**Authors**: Jipeng Li, Pramod Khargonekar, Yanning Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.36524)  

**Abstract**: Can controlled symbolic inputs reveal attention heads that influence a language model's decisions from natural language evidence? We introduce controlled evidence aggregation (CEA), which varies the counts and order of two symbols in otherwise fixed prompts and measures changes in a frozen Transformer's preference between two answer tokens. The preference is measured by their logit difference; the prompts prescribe no correct answer. Balanced utility causal selection (BUC) identifies compact head sets by jointly evaluating how much ablating their outputs removes the count effect and how closely patching mean outputs between count conditions recovers the effect in both directions. We derive conditional bounds linking head attribution to intervention effects and specifying when the attribution shortlist can be recovered from finite samples. In Pythia-410M, ablating sets of 1-12 heads selected separately for four ordering regimes removes 65.5-91.6% of the count effect. The sets retain their influence on count comparisons not used for selection. To test relevance beyond symbolic inputs, we evaluate a natural language task asking which state is supported by more reports. Ablating eight heads selected only on symbolic yes/no prompts lowers mean accuracy by 8.2 percentage points, versus 0.8 for random sets matched by size and layer. Our results demonstrate that controlled symbolic probes can identify compact attention head sets whose causal influence extends to natural language decisions. 

---
# Benchmarking Task Takeover Across Agent Handoffs 

**Authors**: Wenyuan Li, Guang Li, Dongyuan Li, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2610.36523)  

**Abstract**: As agents collaborate on longer tasks, responsibility may pass between models before the work is complete. The receiving model must build on prior decisions, observations, and actions while addressing the remaining requirements. Evaluating this transition requires connecting the information preserved at handoff to how the receiver continues the work, a relationship that task-success and memory-retention scores alone do not capture. We introduce the Handoff State Continuity Benchmark (HSC-Bench), comprising 1,039 tasks that evaluate two complementary aspects of task takeover: recovering relevant repository evidence and continuing from established conversational, tool, and file-work state. Comparisons across six receiving models show that structured handoff state improves repository evidence recovery for every model and continuation performance for most models. The strongest receivers differ between the two tracks, showing that evidence-recovery rankings alone do not characterize performance when continuing from inherited state. By combining these comparisons with checks of subsequent responses and actions, HSC-Bench evaluates both access to prior evidence and adherence to established commitments and completed work. This provides a basis for selecting receiving models and assessing handoff representations according to the work that must be carried forward. 

---
# Do AI Scientists Know What's New? 

**Authors**: Miri Liu, Xiayu Zhao, Encheng Xie, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2610.36511)  

**Abstract**: A major challenge facing AI scientific discovery pipelines is how to determine when a generated idea is meaningfully novel. Ideally, novelty checks in these pipelines must be able to reject non-novel ideas while not rejecting too many novel ideas, and they should also be able to distinguish which parts of a proposal are non-novel, as opposed to opaquely discarding entire proposals. We design an analytical framework for evaluation of novelty checks with temporal ground truth at both the proposal- and claim-level, using four datasets of 1,060 proposals built from published papers, reviewer disputes, and generated ideas matching ones the field went on to publish. Auditing existing novelty checks with this framework, we find that checks with high rejection rates on non-novel work also tend to have high false alarms on ideas that are later realized by human researchers, and observe that checks built to evaluate proposals as a whole can point to claim-level prior art only by rejecting the whole proposal. We therefore introduce ClaimNov, a fine-grained, interpretable claim-level novelty module. ClaimNov catches 97% of disguised papers at a 1.5% false-alarm rate, and its claim-level novelty profile cuts the LLM judge's false alarms from 11.8% to 1.5%. Because its verdict rests on claim-level evidence, ClaimNov also keeps proposals that build on established contributions, rejecting only 3% of ICLR-accepted proposals with a reviewer-disputed contribution and keeping 91 of 100 novel proposals with one spliced prior contribution, against 76 for the strongest existing check. Code and data are available at this https URL. 

---
# CONCUR: Concordant Auxiliary Rewards for Multi-Task, Multi-Reward Reinforcement Learning 

**Authors**: Jixuan Leng, Si Si, Fei Wang, Cho-Jui Hsieh, Jiong Zhang, Hsiang-Fu Yu, Inderjit S. Dhillon  

**Link**: [PDF](https://arxiv.org/pdf/2610.36437)  

**Abstract**: When a Large Language Model (LLM) is trained with Reinforcement Learning (RL) on multiple tasks, a task's primary reward can gate auxiliary rewards such as a length reward: on a failing response, an auxiliary cannot exceed the primary. However, group-relative advantage estimators treat a gated auxiliary like any other component, so it can reverse the primary's order, and at the primary's ceiling only auxiliaries order responses. Keeping gated auxiliaries hurts some tasks, while dropping them hurts others. We propose CONCUR, an advantage estimator that replaces the standardized scores of each gated auxiliary with the closest scores that satisfy two conditions: responses with higher primary rewards cannot receive lower scores, and responses at the primary's ceiling receive equal scores. CONCUR-T applies both conditions throughout training, and CONCUR-A lifts the second for an auxiliary that agrees with the primary's order more often than it reverses it among passing responses. On three models from two families and across math, code, instruction following, and writing, both variants score higher on average than all baselines and than when all the gated auxiliaries are dropped. 

---
# No Tokenizer, Still a Gap: Cross-Lingual Fairness of Homoglyph and Invisible-Character Watermarking 

**Authors**: Alexander Nemecek, Yuqiao Xu, Joseph Falkenburg, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2610.36432)  

**Abstract**: Homoglyph and invisible-character watermarks are edit-based watermarks. Instead of changing how a model generates text, they hide a signal in the text after it is written. They swap some characters for homoglyphs, which look the same, or very similar, but have different Unicode code points, or they insert characters that do not display, all without access to the model that wrote the text. Generation-time watermarks work less well for lower-resourced languages due to how tokenizers split their text. Because they work on individual characters, these watermarks never touch a tokenizer, so we ask whether they escape this gap. Across 40 language varieties in 24 scripts, we measure capacity (the hidden bits each character can carry), how much of it survives routine handling and targeted removal, and how much text a keyed detector needs. Capacity varies 3.3-fold and script accounts for 96\% of that variation. Within Latin script, resource level makes no detectable difference ($\rho=+0.095$, $p=0.78$), but across scripts capacity rises with it ($\rho=+0.54$, 95\% CI $[+0.14,+0.83]$), because lower-resourced languages tend to use scripts with few homoglyphs. Ordinary copy-paste, which strips invisible characters, widens the gap: a detector then needs about 2 code points of English or Persian but 93 of Lao. Removing the tokenizer does not remove the inequity, but instead only changes its source. 

---
# AK_QMUL at ArGuard Shared Tasks: A Single Fine-Tuned JAIS for Arabic Harmful-Prompt Detection 

**Authors**: Ali Khairallah, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2610.36360)  

**Abstract**: We present AK_QMUL's submissions to the ArGuard-2026 shared tasks on Arabic harmful-prompt detection: a binary safe/unsafe task (B1) and a seven-way harm-domain task (B2). Both submissions use one backbone -- JAIS-6.7B fine-tuned with 4-bit QLoRA and a linear classification head over its last-token representation. For B1 we train a four-way head that factors the label by dialect register {safe, unsafe} x {MSA, other}, we augment the training fold with MSA prompts from AraSafe, and decode with a plain argmax to score safe/unsafe. For B2 we fine-tune the same backbone as a seven-way classifier and average the softmax outputs of five random seeds. Both submissions ranked first on their track, with macro-F1 0.984 on B1 and 0.790 on B2. We report results by dialect and by class: on B1 the model is accurate across all six dialects (93.6--99.8%) and its predicted unsafe rate tracks the true rate closely; on B2 six of the seven classes reach F1 between 0.73 and 0.98, with the residual error concentrated in the Bully class, which overlaps semantically with harassment and harm-to-others. 

---
# Detect Before You Disambiguate: Perturbation-Based Input Ambiguity Detection in LLMs 

**Authors**: Tiejin Chen, Xiaoou Liu, Michael Clifford, Ahmadreza Moradipari, Yitao Chen, Kyungtae Han, Nejib Ammar, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.36353)  

**Abstract**: Large language models (LLMs) often answer ambiguous inputs as if they had a single intended meaning. This is problematic because input ambiguity is an upstream source of uncertainty: when a prompt admits multiple valid interpretations, the key question is whether the model should answer directly at all. Existing input uncertainty methods often rely on LLM-generated statements or clarifications, but such transformations may already disambiguate the input before the ambiguity is measured. We propose a label-preserving perturbation framework for input ambiguity detection in which perturbations remain close to the original input while preserving its label. We instantiate this principle with two detectors: a black-box rule-based perturbation library and a white-box importance-guided method using model-internal signals. Experiments across four datasets in open-domain, conversational, and visual QA show that our methods outperform output-side uncertainty baselines and LLM-paraphrase-based perturbation baselines. Our results highlight the importance of detecting ambiguity before attempting to disambiguate it. Our code is available \href{this https URL}{\textcolor{purple}{here}}. 

---
# Beyond IceBERT: Open-Data ModernBERT Encoders for Icelandic from 12M to 1B Parameters 

**Authors**: Hafsteinn Einarsson  

**Link**: [PDF](https://arxiv.org/pdf/2610.36230)  

**Abstract**: IceBERT, the reference encoder for Icelandic, found that a cleaned web crawl is enough to train a strong model and that starting from a multilingual model can help on some tasks. We revisit both findings with the ModernBERT architecture. We train Icelandic encoders from 12M to 1B parameters on 9B tokens of open data, and rerun IceBERT's benchmarks and five newer tasks for 30 encoders under one protocol with confidence intervals. Our 1B model trained from scratch, to our knowledge the first monolingual Icelandic encoder of that size, is significantly better than IceBERT on all eleven tasks; warm-started from the English Ettin encoder of the same shape, it has the best score of all encoders on every task we rank. Because our model shapes match the English Ettin encoders and the multilingual mmBERT, we can also test the warm start under controlled conditions, at six model sizes and four token budgets. With our vocabulary transplanted, a warm start was never significantly worse than scratch training on the same Icelandic text, and from base size up it was not significantly worse on four tasks than scratch training on 4 times the text (8 times at base and 1B). As models grow, most of the transfer comes from the source model's transformer layers, so even an English source helps. Keeping the source vocabulary instead cost 5 F1 on question answering and nearly three times the compute. 

---
# LLM Hallucination Detection via Low-rank Multilinear Decomposition 

**Authors**: Devansh Sharma, Faizul Rakib Sayem, Shahana Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2610.36191)  

**Abstract**: Large language models (LLMs) frequently generate fluent yet factually incorrect responses, raising concerns about their reliability. Recent hallucination detectors increasingly exploit internal representations, but extracting informative signals from high-dimensional hidden states remains challenging. Existing approaches often rely on selected activations or pooling-based compression, potentially overlooking discriminative information distributed across Transformer layers, feature modes, and tokens. We propose LOTUS (Low-rank Order-statistic TUcker-based Subspace learning), a multilinear low-rank decomposition framework for hallucination detection that extracts compact representations from LLM hidden states while reducing dependence on labeled data. Our approach first summarizes variable-length token representations using complementary statistical descriptors that capture peak activations, activation distributions, and cross-layer changes. We then leverage low-rank multilinear decomposition to extract compact representations that capture the underlying structure of LLM hidden states, which are subsequently used to train a lightweight hallucination classifier. Experiments across multiple LLMs and question-answering benchmarks demonstrate improved hallucination detection performance over recent state-of-the-art baselines. 

---
# Frankly Speaking: The Company Election Integrity Keeps in Congressional Communication 

**Authors**: Michael A. Robinette, Christina Barsky, Amanda D. Clark, Helen Flannery, Juniper Lovato  

**Link**: [PDF](https://arxiv.org/pdf/2610.36187)  

**Abstract**: Election integrity is a shared commitment in American politics, but Democrats and Republicans in Congress increasingly mean different things by it. Analyzing 10,420 congressional e-newsletters sent from 2010 to 2025, we examine how the two parties invoke election terms such as ``integrity,'' ``security,'' and ``fraud.'' Using diachronic word embeddings, which track semantic change through how words are used in context, we measure how differently each party uses each term, how much each party shifts, and whether they share nearest semantic neighbors. We show that members of Congress do not need different words to tell different stories about elections. They need only place those same words in different company. Across nearly every term and year, the two parties surround election-integrity vocabulary with entirely different words. That divergence is easy to miss, because most election vocabulary is no more divergent than other words used at roughly the same frequency, and neither party's usage shifted more than the other's. Only a subset of terms grew more divergent, most notably ``certification,'' while ``rigged'' and ``stolen'' entered election discourse after 2020 and polarized there. By 2024, Democrats surround ``voter,'' ``election,'' and ``integrity'' with the language of access and administration. Republicans surround the same terms with ``citizenship verification,'' ``noncitizen voting,'' and ``border security.'' For one party, integrity means protecting eligible voters' ability to cast a ballot. For the other, it means protecting the electoral process from illegitimate ballots. Election officials, who must administer a single system under both definitions, inherit this tension. 

---
# Lipi-Ghor: A Large-Scale, Diarized, Multi-Domain Bengali Speech Corpus and a Scalable Pipeline for Building It 

**Authors**: Sanjid Hasan, A H M Fuad, Risalat Labib, Bayazid Hasan Simon  

**Link**: [PDF](https://arxiv.org/pdf/2610.36184)  

**Abstract**: Bengali, spoken by over 230 million people, remains severely under-resourced in spoken language research. Existing open Bengali ASR corpora are confined to short, read or crowd-sourced speech and lack speaker diarization and long-form audio.
We present Lipi-Ghor, a large-scale weakly-supervised Bengali speech corpus comprising 885 hours of audio sourced from 998 YouTube videos across 596 unique channels, spanning 17 diverse domains including audiobooks, drama, religious sermons, news, and regional dialects. Each recording is annotated in SSTT format (Speaker, Speech, Transcription, Timestamp), which combines automated diarization with Bengali YouTube caption tracks.
To our knowledge, Lipi-Ghor is the first Bengali speech corpus to pair large-scale coverage with speaker-diarized, long-form audio annotations, both of which are absent from existing open Bengali resources. We also present the construction methodology as a replicable pipeline for any language with a sufficient captioned YouTube presence. Targeted channel curation across a handful of rich domains yields corpus scales that were previously out of reach for low-resource languages.
We benchmark four widely-used ASR models under zero-shot and fine-tuned conditions. All four degrade consistently on emotive, dialectal, and long-form content. Because caption targets are noisy, we also release a human-annotated gold evaluation split, verified by 17 volunteer annotators over 30 hours of human-in-the-loop correction and stratified across all 17 domains.
We release our original contributions (video-ID lists, diarization output, pipeline code, and the gold annotations) under CC-BY-4.0, and distribute caption-derived transcripts for research use only; no raw audio is redistributed. 

---
# 3, 7, 14, 17, 42: Instruction Tuning Installs and Sharpens Number-Choice Heuristics in Language Models 

**Authors**: Samar Ansari  

**Link**: [PDF](https://arxiv.org/pdf/2610.36178)  

**Abstract**: We measure, without sampling error, the probability a language model assigns to each integer from 1 to N when asked for a random number. For a fixed prompt the model defines one probability vector, which we compute by summing over each integer's scored spellings rather than estimate by sampling, removing the noise that limits earlier studies [1, 4]. We evaluate 18 open-weight model lines at five ranges up to N = 100, under 29 decoding configurations, on two inference stacks; 15 of them have matched base and instruction-tuned checkpoints that both give usable answers. Instruction tuning makes the distribution less uniform in all 15 pairs. It installs one heuristic, round-number avoidance, and sharpens another, endpoint avoidance. Base checkpoints do not avoid round numbers once the endpoints are removed from those sets; tuned checkpoints avoid them, multiples of 10 almost completely. Base checkpoints return the endpoints 1 and N at 0.30 to 0.60 of the interior rate; tuned checkpoints almost never return them. A preference for odd numbers is consistent across tuned checkpoints only at N = 100, where base checkpoints also show it weakly. Endpoint avoidance and the odd preference fit the account Kubovy and Psotka [6] proposed for human number choice; round-number avoidance is not part of that account, and the midpoint avoidance it predicts is essentially absent, with one slight exception at N = 100. Tuning does not install a new favourite integer shared across models: above N = 10 no modal integer is shared by more than 42 percent of tuned conditions, and P (17) rises in 10 of 15 pairs and falls in 5. The concentration is present in the model's own distribution: at temperature 1.0, nucleus truncation at top-p 0.95 adds a median of 0.035 to the total variation distance. 

---
# FinLLM-KG: Knowledge Graph-Enhanced LLMs for Financial Sentiment Analysis in Algorithmic Trading 

**Authors**: Giorgos Iacovides, Oliver Hannibal, Wuyang Zhou, Danilo Mandic  

**Link**: [PDF](https://arxiv.org/pdf/2610.36083)  

**Abstract**: Recent advances in Generative AI have substantially improved financial sentiment analysis through post-trained financial large language models (LLMs). However, existing approaches remain confined to predicting the sentiment of each article in isolation and operate as black boxes, which limits contextual understanding and hinders deployment in high-stakes financial decision-making. To this end, we introduce FinLLM-KG, a model-agnostic framework that combines causal LLMs with a financial knowledge graph, in order to inject temporally grounded, company-level knowledge from SEC 10-K filings at inference time, without updating model weights. Within FinLLM-KG, context is supplied either as a natural-language company profile or as a small set of relevance-ranked factual triples. This enables sentiment predictions to account for broader firm-level information rather than the article alone, at almost zero computational cost and for any causal LLM. In addition, attributing each article-level sentiment score to individual sentences and to the retrieved knowledge-graph facts makes the evidence behind the prediction inspectable, thereby enhancing explainability and supporting human-in-the-loop review. Simulations demonstrate that FinLLM-KG consistently improves portfolio performance over the state-of-the-art baseline. 

---
# OmniBrain-0: A Unified Agentic Model for Embodied Intelligence 

**Authors**: Xiaomi Embodied Intelligence Team, Yuechen Luo, Chong He, Fang Li, Shaoqing Xu, Jingkai Wang, Menglun Su, Jiaxin Liu, Shengyin Jiang, Zhiyi Lai, Long Chen, Longlong Wang, Liang Hu, Haochen Liu, Yuncheng Jiang, Hongwei Xie, Jieming Cui, Hanbing Li, Junjie Huang, Zhixiang Duan, Hexin Zhang, Weixiang Liang, Ying Gong, Yuhan Wang, Yong Pan, Pengna Li, Kangyi Wu, Ken Deng, Kun Ma, Fuxi Wen, Zhi-xin Yang, Hangjun Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.36013)  

**Abstract**: General-purpose embodied intelligence requires robots to coordinate diverse skills over long horizons, track task progress, and recover from failures. Specialist policies alone do not ensure task-level reliability, while powerful multimodal agents rely on costly reasoning and external experience that compact models struggle to reproduce. We introduce OmniBrain-0, a lightweight unified embodied agent that coordinates manipulation and navigation through a shared high-level decision model, integrating task understanding, planning, progress tracking, and feedback-driven recovery. Our progressive training framework combines embodied pretraining, domain-specialized agent distillation, and multi-teacher on-policy distillation (MOPD). It distills powerful teachers' interaction experience into manipulation and navigation experts, then consolidates their decision capabilities into OmniBrain-0. Unification therefore occurs within the learned decision model rather than only at the tool interface. We further introduce Rubric-Guided Recursive Self-Improvement (RSI), a data flywheel in which a Meta-Agent turns execution evidence into refined rubrics, step-level reminders, and verified trajectories for domain-expert updates before their reconsolidation into OmniBrain-0 through MOPD. We evaluate OmniBrain-0 on LIBERO-Pro, LIBERO-Plus, RoboCasa365, RoboTwin 2.0, and R2R-CE Val-Unseen, as well as multiple real-world manipulation tasks. OmniBrain-0 combines strong cross-domain task performance with substantially lower inference and execution overhead than its closed-source teacher, demonstrating efficient transfer and consolidation of embodied capabilities into a compact unified agent. 

---
# MedJev: Evaluating Zero-Shot System-One Decision Making in Medicine 

**Authors**: Simon A. Lee, Jiahang Sha, Moonseong Jeong, Yunbei Pan, Jeffrey N. Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36006)  

**Abstract**: System One decision models such as Jev answer structured questions in a single forward pass without generating text, offering fast and computationally efficient clinical decision support. We evaluate MedJev, an open System One model that runs locally on a laptop, across ten medical benchmarks spanning medical knowledge, diagnosis, and patient level prediction using MIMIC IV and MIMIC IV ED. We compare MedJev against zero shot LLM baselines, published GPT results, and supervised models. MedJev achieves accuracies of 72.5 percent on MedQA, 75.4 percent on USMLE, and 80.1 percent on NEJM, approaching published GPT 4 performance while maintaining well calibrated confidence estimates (ECE 0.028 to 0.063). On patient level tasks, MedJev achieves an AUROC of 0.926 for In Hospital Mortality, exceeding XGBoost (0.902), but underperforms supervised models on operational outcomes such as ED Triage Admission (0.745 vs. 0.809). Patient level predictions also exhibit substantial miscalibration (In Hospital Mortality ECE 0.286). We introduce a lightweight, two parameter recalibration method that adapts predictions to local patient populations using 50 to 200 labeled examples, reducing ECE to 0.007. Our findings establish open System One models as a promising zero shot baseline for medical AI, combining competitive predictive performance with low computational cost and efficient local calibration. We release our code and evaluation harness to support future research. 

---
# ReCoPII: A Relation-Grounded Framework for Building Contextual PII Masking Datasets 

**Authors**: Taehwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.35925)  

**Abstract**: Training contextual masking systems for personally identifiable information (PII) requires examples distinguishing maskable from preservable mentions of one entity type, but privacy restrictions limit access to real data. Author-based augmentation (SPY) labels mentions unrelated to the author as non-PII, presuming a single data subject and overlooking others. We propose ReCoPII (Relation-grounded Contextual PII masking), a framework that builds such data without this premise: it plans ontology-constrained entity relations, writes insertion sentences with placeholders, reconstructs annotations with code, checks the result with a separate LLM, and derives MASK/KEEP labels from entity relations. On Korean documents from six domains, it yields ReCoPII-Ko, a corpus with relation, privacy, and MASK/KEEP annotations, which we validate. Across 300 pairs, four annotators agree with 88.42% of its sampled KEEP labels, against 26.27% for SPY-style augmentation of the same sources. Classifiers trained on it over-mask less than under that reference on a matched test. Contextual masking fine-tuned jointly with a PII detector removes most over-masking of detection alone within a fixed leakage budget; on human-verified entity records, the four detectors meeting it over-mask at most 5.04% with at most 4.18% leakage. The detector is comparable to XLM-R-large on KDPII with about a quarter of its parameters. 

---
# Heads of Security: Learned Attention Gates Improve Prompt Injection Robustness 

**Authors**: Igor Shilov, Xiaoxue Yang, Alexander Panfilov, Evgenii Kortukov, Matthieu Meeus, Peter Romov, Wojciech Samek, Jonas Geiping, Yves-Alexandre de Montjoye  

**Link**: [PDF](https://arxiv.org/pdf/2610.37657)  

**Abstract**: We introduce Heads of Security (HoS), a prompt injection defense that enforces instruction-data separation via a small set of learned per-head attention gates. HoS learns how each head should redistribute attention between trusted and untrusted tokens, enabling models to reason over untrusted content without following injected instructions. We evaluate HoS on Llama-3.3-70B-Instruct and GPT-OSS-20B. We compare it against SecAlign++, a state-of-the-art finetuning-based defense, and ASIDE, a defense that, like ours, builds instruction-data separation into the model architecture. We show that HoS offers greater robustness to adaptive attacks, better performance on standard utility benchmarks, and matches top performance on standard agentic prompt injection benchmarks. More specifically, the black-box RL-Hammer attack reaches 97% ASR against GPT-OSS-20B and 85% against SecAlign, but only 10% against HoS. Claudini, a state-of-the-art white-box attack, strengthened against HoS through autoresearch, requires 3 times more FLOPs to reach the same ASR on HoS as it does on GPT-OSS-20B or SecAlign. HoS preserves base-model utility across standard multiple-choice benchmarks and prompting setups, whereas SecAlign and ASIDE degrade in few-shot settings. On agentic benchmarks with static attacks, HoS and SecAlign perform similarly and substantially improve the security-utility trade-off over the base model. ASIDE struggles with tool use, achieving only 10% utility. These results show that prompt injection robustness can be achieved and improved through targeted attention interventions. 

---
# Rare, Relevant, and Still Wrong: How Memory Fails LLM Group Decisions 

**Authors**: Anouar Imel, Imad Charradi, Abderrahman Sghir, Khalil El Khammar  

**Link**: [PDF](https://arxiv.org/pdf/2610.37624)  

**Abstract**: When an LLM group's discussion is compressed into a short memory for a final decision maker, what should the memory keep? Two natural heuristics keep the rarest claims or the claims relevant to the decision criteria. On hidden-profile tasks both look attractive, because how widely a fact is held is confounded with whether it helps. We introduce CrossProfile, a diagnostic benchmark of 60 templates with automatically verified answer keys, in which the ownership of supporting, decoy, and irrelevant facts is crossed while their text is held fixed (480 tasks). Across Qwen2.5-7B, Llama-3.1-8B, and two DeepSeek models, both heuristics fail in predictable places. Selecting by relevance readmits decoys, because decoys are on topic: even a perfect relevance filter loses 23 to 31 points to rarest-first selection in the classic hidden-profile configuration. Rarest-first selection is no better than random selection on average and wins only where the rare facts happen to be the useful ones. In both cases accuracy follows the balance of evidence the memory keeps, not the policy that kept it. The decision model adds a third failure, a form of social proof: it gives extra weight to facts attributed to many agents. Removing the attribution tags, or reassigning them at random with line order fixed, raises accuracy in the classic configuration by 13 to 24 points. In natural discussions the evidence is lost before any memory acts: the complete transcript yields lower accuracy than no notes at all. 

---
# EnvSched: Environment-Aware Scheduling for Rollout-Efficient Agent Reinforcement Learning 

**Authors**: Sihan Yang, Weichen Bai, Ruichuan An, Yiming Cheng, Zhiyuan Feng, Tianyi Jiang, Renye Yan, Ning Ding, Yifan Zhang, Renrui Zhang, Yehui Tang, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37609)  

**Abstract**: As executable environments for language agents scale, generating and verifying their rollouts becomes increasingly expensive. Under finite rollout budgets, this raises an important question: can data management at the environment level make agentic reinforcement learning (RL) more efficient by organizing, evaluating, and scheduling task instances together with the environments that determine their tools, states, transitions, and verification? To address this question, we introduce EnvSched (Environment-aware Scheduling), an online scheduler that learns from rollout history. EnvSched assigns scheduling value through two complementary modeling choices: exploitation prioritizes tasks likely to provide informative reward differences, while coverage spreads each batch across underobserved environments and interaction patterns. EnvSched organizes environment--task pairs as nodes in a dual-relation graph, with within-environment edges capturing task semantics and shared tool, state, and verifier structure, and cross-environment edges capturing abstract execution and reward topology. By propagating historical rollout statistics along both edge types, it predicts within-task reward variance. These predictions guide a scheduler that allocates each batch to tasks with high predicted reward variance and underobserved environments and tasks. Environment quotas and a similarity penalty further limit concentration and redundancy, yielding informative and diverse batches without additional pilot rollouts. In RL on Qwen3-8B, EnvSched achieves an average score of 48.81 across BFCL-v3, tau-Bench, and ACEBench-Agent after 40 training steps, exceeding the highest score attained by random task selection after 120 steps. It also improves downstream performance across model sizes at the same training budget. 

---
# Follow the Gradient: Tracing Safety Degradation Back to Training Data 

**Authors**: Saket Reddy, Amro Abbas, Anshuman Suri, Ari Morcos, David Schwab, Pratyush Maini  

**Link**: [PDF](https://arxiv.org/pdf/2610.37481)  

**Abstract**: Fine-tuning Large Language Models (LLMs) on instruction data can quietly erode their safety, even when no training example looks harmful. Because this safety-degrading data looks benign, standard content filters struggle to catch it. We tackle this problem by developing SimpleLoRIF, a scalable gradient embedding method that identifies fine-tuning examples whose gradients align with unsafe behavior. By simplifying a low-rank attribution method and restricting backpropagation to the final four transformer blocks, SimpleLoRIF can embed datasets on the order of a million examples, boosting gradient attribution from a small-scale analysis tool into a practical filter for full post-training datasets. We first validate it on emergent misalignment, where we find that gradients can connect insecure-code training data to the broadly harmful behaviors that appear after fine-tuning, with AUC as high as 0.865 in the gradient space of misaligned models. We then use SimpleLoRIF to build SafeTulu, a version of Tulu 3 with the 20% of examples most gradient-aligned with unsafe responses removed. Training on SafeTulu reduces GCG attack success rate by up to 22% across Qwen, Llama, and Olmo, outperforming both LLM-as-a-judge and prior gradient-based filtering, while largely preserving or even improving capability. Finally, we analyzed the removed data and identify possibly under-studied sources of safety degradation, such as tool-execution failures and narrative arithmetic. We release SafeTulu along with a Hugging Face Space that lets researchers build their own customized versions of SafeTulu to fine-tune on, such as by changing the removal fraction or filtering only certain subsets. 

---
# Test-Time Communication in Trading Agent Swarms 

**Authors**: Avid Fayaz, Harry Zhang, Håvard Alstadheim, Erik Nordlöf, Vilhelm Karlin, André Silva  

**Link**: [PDF](https://arxiv.org/pdf/2610.37448)  

**Abstract**: Communication between agents has mostly been studied on tasks with a shared objective and fast, clear feedback. Trading has neither: each agent manages its own account, with decisions being scored by markets late and noisily. A trading swarm is thus a population of independent decision-makers rather than a team, for which convergence is a risk and not the goal. We ran 200 agents from four model families in groups of ten under five communication regimes, from isolation to message boards and shared notes, over seven US trading days (1 to 9 October 2026), logging everything each agent read before every decision. In two of the four families, communicating groups came to hold far more of the same stocks than the isolated group. A group's return was largely the return on what its members held in common. Seeing a post from a groupmate made an agent 7 percentage points more likely to reuse an eight-word phrase from it. Agents that started too close together to see each other's posts shared such a phrase in 14% of cases, the rate that common inputs produce. Seeing the post did not detectably change that morning's trades. Case studies traced the adoption of positions and decision rules over subsequent days, and communicating groups sold in step more often than isolated ones. Because isolated groups also converged on the same stocks, sometimes completely, assessments of herding need to measure excess convergence relative to a no-peer baseline. 

---
# Representing the Unseen: Rethinking Visual Imagination for Spatial Reasoning 

**Authors**: Zengqi Zhao, Zun Wang, Shoubin Yu, Mohit Bansal, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37437)  

**Abstract**: Visual world models increasingly support spatial reasoning by generating RGB views of unobserved states. However, RGB has become the default interface largely because it is well suited for visual generation, rather than necessarily being the most effective representation for downstream reasoning. Different spatial questions may benefit from different representations of the same imagined state: top-down geometry can facilitate global distance comparisons, while panoramic views provide broader context for reasoning about open space. To isolate the role of representation in spatial imagination, we introduce ImagiRep, a simulator-grounded benchmark of 922 questions spanning seven question types across two settings, Viewpoint Imagination and Cross-view Integration, providing oracle RGB and six alternative representations (e.g., depth, panoramic views, and top-down geometry) of the same unobserved state. This controlled comparison reveals that even accurate imagined RGB does not guarantee accurate spatial reasoning, and different questions benefit from different representations. Similar trends hold for representations derived from world-model-estimated states. Notably, estimated RGB alone can mislead spatial reasoning, while alternative representations retain complementary benefits. Building on these findings, we introduce GEARS, a lightweight test-time strategy for adaptive representation selection, outperforming existing spatial imagination methods and showing that reasoning about unseen states requires more than generating RGB views. 

---
# Game-Theoretic Solvers with Bayesian Opponent Modeling and Moral Rhetoric Framing for GLEE 

**Authors**: Hayden Arnold  

**Link**: [PDF](https://arxiv.org/pdf/2610.37397)  

**Abstract**: We describe the agent we submitted to the GLEE Competition, which plays bargaining, negotiation, and persuasion games against a shared pool of human and AI opponents. The agent is built around three central components: a classical game-theoretic solver to compute numeric moves, an online Bayesian layer to estimate opponent archetypes, and a language model to write accompanying messages. We utilize Rubinstein bargaining for bargaining, Faratin concession for negotiation, and Kamenica--Gentzkow disclosure for persuasion, and adapt those models to live opponents using two separate layers: a Bayesian estimate of the opponent's hidden discount factor, reservation value, or signaling reliability, and a Thompson sampling bandit over five moral-foundation framings coupled with an evidence-intuition spectrum to establish message register. Most of the development effort went into the recurring problem that theoretically optimal policies perform poorly in certain live settings; we report associated failure modes, guardrails we built, and our hypotheses. We additionally evaluate our model over 13,200 Monte Carlo games against 11 synthetic opponent archetypes, an ablation of the opponent-modeling layer, and an extensive analysis of 9,337 live-game messages in which moral rhetoric framing in the persuasion setting is strongly correlated with whether the buyer purchased ($\chi^2=616.7$, $p \approx 3.7\times10^{-132}$), surviving controls for quality, round, and configuration. 

---
# LinMix: Data Mixing as Linear Domain Attribution 

**Authors**: Hubert Baniecki, Przemyslaw Biecek, Martin Pawelczyk  

**Link**: [PDF](https://arxiv.org/pdf/2610.37390)  

**Abstract**: Language model performance depends strongly on the mixture of domains in the pretraining data corpus. Popular data mixing methods select these domain weights by training many small proxy models on random mixes and then fitting a regression model to predict performance from the mix. We recognize this practice as a domain attribution problem, where the regression measures each domain's contribution to the target metric. Through this lens, the standard Dirichlet sampling of proxy data mixes is a poor regression design that requires tuning and cannot support a linear fit for hundreds of domains. Building on classical optimal experimental design, we introduce LinMix, which samples mixes at the vertices of the capped simplex so that each domain is at its cap or absent. This hyperparameter-free vertex sampling decorrelates domain effects, enabling an interpretable linear model whose coefficients are the domain attributions. The optimal mix given by LinMix provably keeps only domains with positive attribution values and requires no search over candidate mixes. We further establish the optimality of vertex sampling, and prove that Dirichlet sampling inflates the attribution error when its mixes concentrate around the prior data mix. Pretraining experiments with 1B-parameter OLMo2 and 4B Qwen3.5 language models show that LinMix outperforms Dirichlet-based data mixing across reasoning, code, and math benchmarks, with relative gains of 46% on MBPP and 14% on GSM8K. At 288 fine-grained domains, only vertex sampling supports a linear fit, and LinMix improves benchmark accuracy by 11% on average over the best nonlinear baseline. 

---
# EmbeddingGemma 2: Unified and Efficient Multimodal Representations 

**Authors**: Henrique Schechter Vera, Sahil Dua, Jovan Andonov, Sara Smoot, Sebastian Russo, Ivan Llanos, Max Matveev, Jae Hyeon Yoo, Kat Black, Shih-Cheng Huang, Josiah Aklilu, Maarten Grootendorst, Omar Sanseviero, Chandramouli Amarnath, Ian Ballantyne, Zhe Li, Daniel Salz, Ibrahim Alabdulmohsin, Ilya Philippov, Kristen Quan, Glenn Cameron, Edouard Yvinec, Luciano Martins, Doug Reid, Sindhu Raghuram Panyam, Weiyi Wang, Hengxuan Ying, Madhuri Shanbhogue, Tanmaya Dabral, Shanfeng Zhang, Juyeong Ji, Renjie Wu, Joe Zou, Xu Chen, Yunan Zhang, Yu-Hui Chen, Matthew Soulanille, Vinod Mamillapalli, Yi-Chun Kuo, Wai Hon Law, Matt Kreileder, Alice Zheng, Suril Shah, Mogan Shieh, Pooja Consul, Alberto Montes, Andreas Hess, Andres Sevilla, Babak Samari, Brendan McCloskey, Brian Potetz, Daniel Cer, Georgios Kathareios, Gustavo Hernández Ábrego, Hesen Zhang, Ji Ma, Jiaxing Wu, Keegan Mosley, Paul Suganthan, Rui Meng, Shen Gao, Shengcao Cao, Yi-Ting Chen, Yichang Chen, Zhongli Ding, Kendra Byrne, Cormac Brick, Olivier Lacombe, Qin Yin, Tris Warkentin, Armand Joulin, Tom Duerig, Mojtaba Seyedhosseini  

**Link**: [PDF](https://arxiv.org/pdf/2610.37364)  

**Abstract**: We present EmbeddingGemma 2, an open multimodal embedding model which natively represents text, images, audio, video, and combinations thereof in a unified space. Our approach introduces improvements such as simplified per-layer embeddings and a multi-stage curriculum with modality and objective progression which ensure strong quality across unimodal, cross-modal, and mixed-modal evaluations. Notably, in comprehensive embedding benchmarks across all modalities, EmbeddingGemma 2 establishes a new Pareto frontier, outperforming both specialist models and models two to twelve times larger. Furthermore, our design offers flexibility for resource-constrained environments, supporting selective modality encoder loading, adjustable vision token budgets, truncatable embedding dimensions, and quantized weights. We present ablations analyzing key decisions and release EmbeddingGemma 2 to promote further research. 

---
# Modeling Preference Dependencies with Directed Acyclic Graphs for Narrative-Driven Recommendation 

**Authors**: Mingzhe Lu, Jian Qin, Ruize Qin, Jinhao Liu, Yunpeng Li, Hang Su  

**Link**: [PDF](https://arxiv.org/pdf/2610.37280)  

**Abstract**: Narrative-driven recommendation allows users to express multifaceted needs through free-form text where stated preferences naturally qualify and condition one another. Existing methods typically treat narratives as flat keyword combinations or unconstrained generative queries, failing to capture how complex preferences hierarchically refine and contextualize each other. We propose the Preference Dependency Graph Framework (PDG-F), a narrative-driven recommendation framework centered on explicit preference dependencies. Specifically, PDG-A (PDG-Agent) first parses unstructured requests to organize user preferences and their mutual qualifications into a directed acyclic Preference Dependency Graph (PDG). Through this graph, candidate evidence from the open Web is acquired via directed path retrieval, and final recommendations are determined by topological path ranking. Comprehensive experiments across three diverse domains spanning English and Chinese contexts confirm that PDG-F consistently achieves superior recommendation performance over competitive baselines, open-domain retrieval agents, and frontier language models. Further diagnostic analyses show that our dependency-guided retrieval markedly raises the attainable candidate-pool ceiling, demonstrating the fundamental value of modeling preference dependencies in narrative-driven recommendation. 

---
# Where Should the Evaluation Budget Go in LLM-Driven Agent Scaffold Evolution? 

**Authors**: Sichun Luo, Yi Huang, Mukai Li, Haochen Luo, Guanzhi Deng, Haibo Wang, Jian Xu, Zefa Hu, Junlan Feng, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37276)  

**Abstract**: Large language model (LLM)-driven agent scaffold evolution improves agent workflows through iterative candidate generation and evaluation. Evaluation both guides subsequent search and determines the final scaffold, but evaluating every candidate on the full task set incurs substantial execution costs. We investigate how evaluation budgets can be allocated across these two roles. Our experiments show that sparse feedback can sustain productive evolution, while additional targeted evaluation improves final selection. Replay analyses further show that shared informative items and a common scoring reference support comparisons among closely matched candidates. Motivated by these findings, we propose \method, an evaluation allocation framework that combines sparse search-time feedback with targeted final confirmation. A short fully evaluated warm-up establishes a shared informative core and calibrates a response model whose item parameters remain fixed during search. After search, \method refits the model and applies paired successive halving, reusing accumulated observations and executing only missing evaluations. Across four benchmarks and two evolution frameworks, \method achieves higher mean performance than full evaluation in most settings while using approximately half as many evaluation cells. 

---
# Watermark Basins: Black-Box Detection of Distilled LLMs 

**Authors**: Hongyan Chang, Hady Elsahar, Tom Sander, Pierre Fernandez, Saeed Mahloujifar, Sylvestre-Alvise Rebuffi, Tomáš Souček, Tuan Tran, Valeriu Lacatusu, Surya Parimi, Rashel Moritz, Vanessa Stark, Alexandre Mourachko  

**Link**: [PDF](https://arxiv.org/pdf/2610.37266)  

**Abstract**: Proprietary language models and their reasoning traces are vulnerable to API-based distillation, where an adversary trains a student model on the teacher model's outputs. To obtain evidence of distillation, model providers can watermark the teacher's outputs, aiming to leave a radioactive signal in the student's outputs. Existing black-box detectors search for such radioactivity using token-level statistics of the suspect model's outputs. However, token-level signals are not always detectable, for instance when the distillation traces are sampled at a low temperature. In this work, we discover the watermark basin, a persistent signal that enables more reliable watermark detection: during fine-tuning, watermarked teacher traces embed a distinct, key-specific geometric signal in parameter space that lies orthogonal to the ordinary task-specific fine-tuning subspace. Building on this, we introduce Watermark Basin Alignment (WM-Basin), a black-box watermark detector that queries the suspect model through its API, without access to its weights or logits. WM-Basin maps generated text to gradient representations using a fixed clean probe model, projects out the ordinary fine-tuning task directions, and tests whether the residual gradient aligns with the watermark basin induced by the watermark key. We evaluate WM-Basin extensively across diverse teacher models and student architectures (from 135M to 8.2B parameters) distilled on mathematical reasoning. At a strict 0.1% false-positive rate, WM-Basin detects 79% of distilled models at n-gram=1 and 45% at n-gram=3, substantially improving over the state-of-the-art token-level baseline (29% and 1.3%, respectively). Because the watermark signal resides in a persistent basin, it survives post-distillation modifications including quantization, pruning, and reinforcement-learning post-training. 

---
# Your Policy Is Secretly a Harness: The Decision Geometry of Agent Interfaces 

**Authors**: Jian Mu, Ruibo Shen, Yao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2610.37255)  

**Abstract**: Language model agents are post-trained and deployed through a harness, the prompts, tools, and control logic that present a task to the model and execute its outputs. Harnesses are often revised and judged by how an already trained model performs under them, which assumes that what the model learned under one harness carries over to the next. We ask what a policy learns from its harness and what survives when the harness changes. Under KL-regularized control, we show that the log odds of an optimal policy recover how strongly its harness favors each action over the others, so a trained policy carries the preferences of its harness. Deployed return then equals the ceiling of the harness, the best return it offers, minus the mismatch of the policy, the return lost to its own suboptimal decisions. Training changes only the mismatch, while changing the harness moves both terms. Comparing harnesses with a fixed model therefore mixes what each harness offers with how well that model uses it. In MiniWoB++, post-training makes a model's preferences specific to its training harness. Reversing the order of the observation fields leaves every optimal decision unchanged yet lowers a trained model's success from 85.0% to 19.2%. Conversely, every new harness that a fixed-model comparison rejects but that raises the ceiling outperforms the original agent after further training under it, and ceiling and mismatch measured beforehand predict these gains. For coding-agent harnesses on Terminal-Bench Lite, ceiling minus mismatch estimated with a stronger model as a stand-in for the optimum reproduces the accuracy ranking, while either term alone does not. Code is available at this https URL. 

---
# Uncovering Shortcut Amplification in Video RLVR through the Lens of Perceptual Uncertainty Collapse 

**Authors**: Jiafeng Liang, Xianhao Ou, Tao Ren, Rongchuan Mu, Runxuan Liu, Dandan Tu, Ming Liu, See-Kiong Ng, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.37211)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as an effective post-training paradigm for enhancing the video reasoning capability of Large Multimodal Models (LMMs). Despite its empirical effectiveness, we show that outcome-level optimization can introduce an undesirable shortcut amplification effect, whereby models increasingly favor text-prior answers even when they are not supported by the visual evidence. To characterize this behavior, we investigate the training dynamics of video RLVR through the lens of perceptual uncertainty, analyzing how predictive uncertainty evolves under video-conditioned and text-only contexts. Our analysis identifies a perceptual uncertainty collapse phenomenon, in which RL training continuously sharpens the model's answer distribution while the uncertainty reduction attributable to visual evidence progressively diminishes. A further gradient-direction analysis reveals that such confidence enhancement becomes increasingly less aligned with visually grounded uncertainty reduction and is instead dominated by text-prior compression. Motivated by these findings, we propose Bifocal Self-distillation Policy Optimization (BSPO), a self-distillation framework that regulates policy updates with two complementary distributional signals. Specifically, BSPO derives empirical reasoning principles from intra-group correct and incorrect rollouts to identify reasoning-sensitive tokens, and introduces shortcut-aware visual perturbation to penalize predictions that remain invariant when task-relevant visual evidence is weakened. By adaptively reweighting the original RL advantage, BSPO preserves the task-solving benefits of RLVR while promoting updates that are both reasoning-sensitive and visually grounded. Extensive experiments validate the effectiveness of BSPO in improving video reasoning performance and strengthening visual faithfulness. 

---
# Coverage Under Contradiction: A Benchmark for Negation-Aware Retrieval 

**Authors**: Pasha Abrishamchian, Eugene Yang, Andrew Yates, Nazli Goharian, Ophir Frieder  

**Link**: [PDF](https://arxiv.org/pdf/2610.37162)  

**Abstract**: Coverage-evaluation benchmarks quantify whether retrieved evidence covers the different pieces of information needed to satisfy an information need. Existing benchmarks mostly consider factual claims that may already be known to pretrained retrieval components, leaving open whether retrievers find corpus evidence when it contradicts those priors. We study how retrievers behave when their parametric knowledge and the information in a corpus disagree. To do so, we develop a process to augment a nugget-based coverage benchmark with controlled contradictions. For each nugget, we create a negated counterpart and edit its supporting documents to support that negation while preserving the context. Applying this process to CoverageBench, a benchmark measuring information coverage, we create CoverageBench-Negate, spanning six datasets and roughly 63,000 original-edited evidence pairs (one per nugget-document link). We attach rich construction metadata to every pair and additionally provide out-of-family LLM judgments over all pairs and a human audit of a sample. In baseline experiments, BM25 separates original from edited near-twin evidence substantially better than dense retrievers (Qwen3 and Jina), which stay near chance. We release the benchmark on Hugging Face Datasets (nuggets, documents, pairs) together with the construction materials, including the prompts used to create it. 

---
# Mechanistic Failure Modes of Weight Pruning in Transformer Language Models 

**Authors**: Nabeelah Faumi, Deshan Sumanathilaka  

**Link**: [PDF](https://arxiv.org/pdf/2610.37126)  

**Abstract**: Pruned language models are judged almost entirely by accuracy and perplexity, which assumes that preserved output behaviour implies preserved internal computation. We test this assumption with a factorial pruning study on GPT-2, crossing two importance criteria (magnitude, random), two component types (attention heads, MLP neurons) and three sparsity levels (3%, 50%, 70%). Using the fully characterised indirect object identification (IOI) circuit as ground truth, we measure circuit integrity with activation patching, path patching and direct logit attribution. We find two qualitatively distinct failure modes. Magnitude head pruning >= zeroes the circuit: output-facing heads are removed, and the model falls to near-chance. Magnitude neuron pruning at 30% inverts it: all 26 circuit heads remain intact, but Name Mover contributions flip sign, and the model systematically prefers the wrong name (17.7% accuracy). Perplexity ranks it as less damaged than random head pruning (74.6 vs 140.1). A replication of the Greater-Than circuit shows that the critical component type depends on the circuit's architecture. These results show that output metrics and structural checks can both miss mechanistic damage, motivating circuit-level causal analysis as a complementary evaluation of compressed language models. 

---
# Prospector: learning a compact graph traversal policy for scientific reasoning 

**Authors**: Haiqian Yang, Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2610.37121)  

**Abstract**: Scientific reasoning in specialized engineering fields depends on knowledge scattered across the literature, often proprietary and absent from frontier LLM training data. In such domains, frontier models tend to give generic answers. Knowledge graphs (KGs) can ground LLM reasoning. Because a graph is too large to provide in full, typical LLM+KG systems preselect a portion, deciding relevance before reasoning begins. Here we train a compact model, Prospector, to reason over a domain knowledge graph by traversing it step by step, seeking evidence in light of what it has already found. Prospector calls a small set of tools one at a time to progressively build its context, and answers with a chain of evidence that can be checked edge by edge against what its tools retrieved from the graph. On a knowledge graph built from the semiconductor-manufacturing literature, Prospector approaches the strongest off-the-shelf models given the same graph and tools, and it far exceeds off-the-shelf models answering closed-book or from a pre-selected portion of the graph. Although trained only on the semiconductor graph, Prospector transfers to unseen knowledge graphs from three other distinct fields: bio-inspired materials, per- and polyfluoroalkyl substances (PFAS) and materials properties. Served on a single GPU, Prospector answers faster and at a fraction of the cost of a hosted service, while keeping proprietary data on the owner's hardware. We find that the policy's latent state is indicative of its next move on both seen and unseen graphs, and modulating the latent state can steer the policy from one behavior mode to another. These results suggest a practical LLM + KG paradigm for scientific and engineering reasoning over private data, in which a compact model trained on a graph from published literature transfers its traversal to unseen graphs that stay with their owners. 

---
# Self-Regularized Flow Language Models 

**Authors**: Theodoros Kouzelis, Spyros Gidaris, Nikos Komodakis  

**Link**: [PDF](https://arxiv.org/pdf/2610.37115)  

**Abstract**: We introduce SERF (Self-Regularized Flow Language Models), a simple self-supervised framework for improving continuous diffusion language models by exploiting the denoising trajectory itself. While continuous flow-based language models offer a promising alternative to autoregressive generation, their training objectives typically rely on supervision from clean target tokens, leaving the denoising trajectory underexploited as a learning signal. SERF addresses this limitation through noisy-to-clean self-distillation: an exponential moving average (EMA) teacher receives a cleaner version of the student's noisy input along the same interpolation path and provides additional supervision. We investigate two variants: SERF-L, which aligns the student's intermediate representations with deeper teacher representations, and SERF-P, which aligns their output token distributions, either at the student's output or at an early student layer. All variants augment the standard training objective without requiring external pretrained models or modifying the inference procedure. Experiments across three continuous language-model paradigms (one-hot FLM, latent ELF, and hyper-spherical S-FLM) demonstrate substantial improvements in training efficiency, generation quality, and few-step generation. In particular, SERF achieves up to 7.7x wall-clock training acceleration on OpenWebText, improves MAUVE by 24.3% and GSM8K accuracy by up to 28.8%, and matches baseline performance with up to 64x fewer sampling steps. These results establish noisy-to-clean self-distillation as an effective approach to improving continuous diffusion language models. 

---
# EnergyCal: Taming Loss Spikes in Muon-Family Optimizers for Sparse MoE Training 

**Authors**: Masato Fujitake  

**Link**: [PDF](https://arxiv.org/pdf/2610.37062)  

**Abstract**: Sparse mixture-of-experts (MoE) language-model training can exhibit large loss spikes with Muon-family optimizers. We introduce EnergyCal, a lightweight rule that attenuates routed-expert updates according to the normalized squared Frobenius norm of their finite-step Newton-Schulz directions. EnergyCal leaves the spectral transform unchanged and adds no persistent optimizer state. We evaluate it on a 4B-parameter MoE model. Against a development-selected, lower-learning-rate MuonH recipe, EnergyCal-MuonH reduces mean thresholded loss-spike burden by 99.51% and final geometric-mean perplexity by 4.06%, with a modest throughput cost. With plain Muon at matched nominal learning rate and weight decay, the reduction in mean spike burden averages 95.98% across the two tested exponents, accompanied by small endpoint improvements. These findings support stabilization both with and without Hyperball in the evaluated setting. 

---
# Exploring the Limits of the Delta Learning Hypothesis: Single-Pair Generator Biases in Direct Preference Optimization 

**Authors**: Giuseppe Curci, Stefano Simonazzi, Andrea Molinari, Andrea Zugarini  

**Link**: [PDF](https://arxiv.org/pdf/2610.37058)  

**Abstract**: Direct Preference Optimization (DPO) typically relies on preference pairs generated from diverse model pools and ranked by an external LLM judge. The recently proposed delta-learning hypothesis suggests that comparable supervision can instead be obtained from a single pair of models by treating the larger model's response as preferred. We investigate the limits of this approach across model families and scales. We show, both theoretically and empirically, that fixed model pairs introduce systematic, generator-dependent biases that are predictive of the preference label independent of response quality. Consequently, DPO trained on single-pair data consistently underperforms multi-model preference data, despite the larger generator producing higher-quality responses, and we trace this gap to training dynamics consistent with reward hacking on these biases. We further show that simply increasing the number of generators, even without an external judge, is enough to partially mitigate this effect. Our results highlight generator diversity as a critical factor in preference-data construction. 

---
# Guided-RLVR: Guided Exploration in Reinforcement Learning with Verifiable Rewards 

**Authors**: Bo Yang, Zishuai Zhang, Weihua Li, Qinnan Zhang, Sijia Wen, Hainan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37041)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has shown strong empirical gains in improving the reasoning ability of large language models (LLMs) on tasks such as mathematics and programming. However, recent studies have shown that RLVR rollouts often collapse onto a narrow set of reasoning trajectories, limiting the explored reasoning space. Motivated by findings that thinking tokens at critical steps serve as critical structural elements in LLM reasoning pathways, we propose Guided-RLVR, a novel approach that augments RLVR with a lightweight controller. The controller strategically injects semantically meaningful steering tokens into the sampling distribution, encouraging the model to approach problems from multiple reasoning perspectives and thereby expanding the exploration space. Experiments on mathematical reasoning and code generation benchmarks show that Guided-RLVR consistently outperforms existing RLVR methods while producing more diverse reasoning trajectories. 

---
# Zooming without Zooming v2: From Seeing Details to Connecting Evidence for Generalizable Multimodal Perception 

**Authors**: Lai Wei, Yatai Li, Liangbo He, Jun Lan, Renjie Gu, Linhan Cao, Junkai Lin, Huijia Zhu, Weiqiang Wang, Linghe Kong, Yue Wang, Zhuosheng Zhang, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.37037)  

**Abstract**: Multimodal perception forms the foundation of multimodal intelligence. In particular, spatial perception and multimodal long-context understanding are two pivotal tasks that underpin embodied AI and multimodal agents. However, synthesizing targeted, high-quality training data for these tasks remains difficult and costly. Our key insight is that these tasks share an underlying capability, which we call connecting evidence: identifying relevant visual facts, binding them to the correct entities, recovering their relations, and composing them into an answer. Crucially, targeted training data are far easier to synthesize for this capability than for these tasks themselves. To strengthen this capability, we propose EvidenceConnect, an easy-to-implement data synthesis framework that uses zoomed views of related regions within a single image to generate questions requiring comparison, aggregation, or chaining of visual evidence. To ensure quality, a reference answer is retained only when independent VLMs agree on it, and region-ablation tests discard questions whose answer survives the removal of selected regions. We then train base VLMs on the resulting data with RLVR, obtaining Zooming without Zooming v2 (ZwZ2). Although the synthesized data are not tailored to any downstream benchmark, ZwZ2 improves fine-grained and spatial perception, with further gains on embodied spatial reasoning, multimodal long-context understanding, multimodal agent tasks, and general visual reasoning. These results establish EvidenceConnect as an effective framework for supervising evidence connection and suggest that strengthening this shared capability leads to more generalizable multimodal perception. 

---
# V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models 

**Authors**: Miso Choi, Daekeun Kim, Eunji Kim, Jungbeom Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.37019)  

**Abstract**: Embedding-level scaling expands the capacity of language models through lookup-based memory rather than by increasing model depth, width, or dense computation. Extending this idea to vision-language models (VLMs) is challenging because visual features are continuous, spatially structured, and query-dependent, unlike discrete text tokens that naturally support n-gram lookup. We introduce V-LUMEN, a visual lookup memory for VLMs. V-LUMEN discretizes visual features with a frozen codebook, builds spatially aware memory keys through spatial aggregation-based hashing, and modulates them with text-conditioned relevance signals; all addressing is non-parametric, and the retrieved embeddings are fused into the backbone through a gated residual pathway. On five reasoning-intensive benchmarks, V-LUMEN substantially improves a frozen Qwen3-VL-2B backbone and transfers zero-shot to three benchmarks whose images are disjoint from the training data. At a matched parameter budget, moving part of a LoRA or MoE adapter's capacity into V-LUMEN preserves accuracy on two backbones, indicating that external visual memory is an independent capacity axis rather than a stronger adaptation method. Because lookup cost is constant in table size and incurred only once at prefill, this capacity grows in storage rather than in per-token compute. Code is available at this https URL. 

---
# SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation 

**Authors**: Zafer Sahinoglu, Abdulazeez Mohammed, Buse Yaren Kazangirler, Cahit Berkay Kazangirler  

**Link**: [PDF](https://arxiv.org/pdf/2610.36991)  

**Abstract**: Retrieval-augmented generation (RAG) systems ground large language models (LLMs) by retrieving external evidence and conditioning generation on the retrieved context. Although dense retrieval is effective for local semantic matching, independently ranked retrieval can return redundant passages, miss cross-chunk dependencies, and provide incomplete evidence for multi-hop reasoning. We propose SpectRAG, a spectral graph-augmented retrieval framework that combines hybrid candidate seeding, weighted chunk graph construction, graph-based expansion, spectral relevance propagation, and token-budgeted context selection. Document chunks are represented by dense semantic embeddings and graph-spectral coordinates derived from a weighted chunk graph whose edges encode semantic similarity, document adjacency, section hierarchy, shared entities, citation links, and similarity between LLM-generated utility questions. At query time, SpectRAG builds a high-recall seed set using exact inner-product search and hybrid relevance signals, expands it over the chunk graph, re-scores candidates using graph proximity and spectral information, and assembles the final context under a fixed token budget. The goal is not merely to retrieve individually similar chunks, but to construct a context that is relevant, diverse, structurally connected, and sufficient for grounded reasoning. Experiments on MultiHopRAG, UltraDomain, and long-document question-answering settings show that SpectRAG improves broad evidence recovery and achieves competitive generation quality against dense, hybrid, HyDE, LightRAG, and NaiveRAG baselines. The results further indicate that graph-spectral structure is particularly useful for inference-oriented and cross-document retrieval, while highlighting trade-offs in early precision, computational cost, and domain-dependent performance. 

---
# AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations 

**Authors**: Qiming Shi, Yikun Li, Beidi Luan, Rui Sun, Yan Ma, Jing Li, Daxin Jiang, Zuo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.36980)  

**Abstract**: Large language model (LLM) agents are extending agent-mediated commerce on the Web from shopping for consumers to operating businesses for sellers. In online-to-offline (O2O) commerce, decisions are issued through Web-based platforms but realized through costly physical execution. Evaluating such agents requires going beyond individual transactions to ask whether decisions made online turn into sustained economic returns. We introduce AutoVend, a benchmark in which one LLM agent manages 20 vending machines across five site types through a Web-based operating interface for 180 simulated days. Its data-grounded simulator combines a shared warehouse, procurement lead times, paid replenishment, inventory expiration, and recurring operating costs. We evaluate five LLM agents against a rule-based inventory-control baseline and diagnose their site coordination, execution, and performance over time. Four of five evaluated LLM agents fall short of this baseline, and returns per dollar of inference cost rank models differently from returns alone. Operational analyses show that gains in individual metrics do not consistently yield stronger economic growth, as inventory and execution costs erode these gains. Trajectory analysis further reveals difficulties in grounding feasible actions, managing inventory commitments, and converting feedback into effective policy revisions. 

---
# Mining the Nineteenth Century for the Thirteenth: Register-Scale Forced Alignment and HTR Bootstrapping via Legacy Regesta and Antiquarian Editions 

**Authors**: Tamas Kovacs, Daniel Luger  

**Link**: [PDF](https://arxiv.org/pdf/2610.36956)  

**Abstract**: Handwritten text recognition (HTR) for the registers of Pope Honorius III depends on modern expert transcriptions, which exist only for regnal years 1-2 of Reg. Vat. 9. For the remaining registers, nineteenth-century scholarship offers Pressutti's regesta and partial editions. On fol. 1-203, where both layers exist, we measure what this legacy layer can contribute, on transcriptions that we first audit against it, and find a division of labor across three tiers. At the orthographic tier, raw editions are at best neutral for adapting a recognizer to the second regnal year and harmful without replay, whereas projection onto curial spelling with replay of modern source lines turns them into zero-label adaptation targets: pooled over all 134 year-2 sides, the WER falls by 1.50 points (95% CI -1.84 to -1.15), closing 33% of the gap to an oracle; the spelling changes at the year split, through the scribe or the transcriber, whereas the hand changes visibly only at fol. 191r. At the lexical tier, the corpus-wide legacy lexicon contains 82.2% of the out-of-vocabulary words of unedited entries and flags recognition errors with 66.7% precision, and labels from pre-trained medieval NER models, distilled into the recognizer's NER head, reach F1 82.1% on an expert validated benchmark, level with the best tagger applied after recognition. At the topological tier, Pressutti's regests, the closing datum clauses, and the regests' dates converted to the chancery's Roman form find 73.9% of entry starts in the recognized line stream, and edition agreement estimates a CER of 7.80% beyond fol. 203. Antiquarian scholarship thus cannot replace modern transcription, but it can steer, check, and segment HTR across the untranscribed registers. 

---
# HiLo-RD: Cross-Budget Self-Distillation for Masked Diffusion Language Models 

**Authors**: Xiang Li, Chengting Yu, Jungang Li, Yilun Zheng, Junzhuo Li, Xingyao Zhang, Wenbo Su, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36901)  

**Abstract**: We propose HiLo-RD (High-to-Low Budget Representation Distillation) to improve masked diffusion language models (MDLMs) under a limited generation budget. Unlike autoregressive models, MDLMs choose both what to generate and where to commit it. This creates two coupled challenges: predicting useful content before reasoning is complete, and deciding when to fix answer tokens that shape later predictions. We identify budget-induced premature commitment: even when a correct solution fits, the model can fix answer fragments near the window boundary before resolving intermediate steps. These fragments become context for later predictions and redirect subsequent reasoning. Existing post-training methods do not directly transfer the stronger trajectories that the same model follows with a larger window. HiLo-RD addresses this gap by (1) pairing large-window teacher states with cropped small-window views and (2) matching their hidden representations at committed positions. This supervision guides both token prediction and commitment order without an external teacher. Across mathematics benchmarks, HiLo-RD improves low-budget accuracy and shortens responses; these gains extend to code generation without code-domain training. On GSM8K, accuracy rises by 9.4 percentage points as mean readable length falls by about 25%. Attention diagnostics and fixed-budget order interventions support the proposed mechanism. Together, these results show that large-window reasoning can improve small-window decoding. 

---
# RubricAtlas: Mining Type-Level Rubrics from Expert Reports to Evaluate Deep Research Agents 

**Authors**: Delin Wang, Sinuo Wang, Rui Sun, Beidi Luan, Qiming Shi, Jing Li, Daxin Jiang, Zuo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.36878)  

**Abstract**: Deep research agents conduct multi-step Web searches and synthesise retrieved information into analytical reports. Evaluating these outputs requires checking which analyses they deliver, beyond the presence of relevant facts. Query-specific binary rubrics make individual requirements explicit, but task-level checklists may leave the overall analytical structure underexamined. We introduce RubricAtlas, which derives type-level analytical requirements from expert reports on different objects within the same research type and organises detailed checks around recurring analytical questions. The requirements give related tasks a shared standard for what to analyse, without prescribing their answers. Our framework links binary checkpoints to report passages and evaluates completeness separately from argument validity. Built from 3,079 Chinese financial research reports, RubricAtlas covers 99 research types. Reference-report selection experiments show that individual reports cover only about half of their type's analytical questions, and that selecting requirements from different reports changes evaluation scores and agent rankings. Across ten agents, two judge pipelines achieve checkpoint agreement of $\kappa$ = 0.71 and agent-ranking agreement of Kendall $\tau$ = 0.87. Controlled deletions reduce scores on the affected analyses, while padding produces no detectable increase. These findings support type-level rubrics for comparing the analytical completeness of Web deep research outputs, with individual decisions traceable to report evidence. 

---
# GUI-JEPA: Candidate-Conditioned JEPA Latents as a Fast Action Policy for Mobile GUI Agents 

**Authors**: Bingchao Wang, Yongcheng Jing, Shunyu Liu, Chengyu Wang, Jun Huang, Jie Yang, Wei Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2610.36877)  

**Abstract**: Mobile GUI agents are typically fine-tuned 7B-scale vision-language models (VLMs) that generate each action as text. However, these VLMs decode every action token by token, and even agents that score candidate actions keep a large language model or VLM in the loop, making them slow and memory-hungry. To address these issues, we propose GUI-JEPA, to our knowledge the first decoder-free GUI agent policy built on a Joint-Embedding Predictive Architecture (JEPA), which selects actions in the latent space of a JEPA encoder pretrained on unlabelled GUI screenshots. Its action- and region-conditioned latent predictor turns the screen latent into one latent per candidate action, letting lightweight heads score all candidates against the instruction in a single batched pass, without decoding a token. We further show that the scorer relies on this latent pathway and that the predictor is trained for the decision itself: a randomly initialised predictor does just as well as one pretrained on screen dynamics. On AndroidControl, GUI-JEPA is 2.8-4.0x faster on the same GPU than a Qwen2.5-VL-7B agent we fine-tune with the same inputs, while reaching 86-92% of its in-distribution accuracy with 6% of its parameters. Code is available at this https URL 

---
# AdaGround: Token-Normalized Adaptive Zooming with Verifier-in-the-Loop Refinement for GUI Grounding 

**Authors**: Bingchao Wang, Yongcheng Jing, Chengyu Wang, Jun Huang, Jie Yang, Wei Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2610.36874)  

**Abstract**: GUI grounding on high-resolution screens fails mostly on small targets, and zooming in is a common training-free remedy. However, most existing zooming pipelines do not match the zoom to the target's size, accept the zoomed answer unconditionally and never revisit a wrong answer. To address these issues, we propose AdaGround, a training-free framework with three components: (1) Token-Normalized Adaptive Zooming, which sets the magnification, field of view and zoom depth of each sample from the model's own estimate of the target size; (2) Verifier-in-the-Loop Refinement, which uses one marked-click Yes/No verifier to keep or reject each zoom, choose between zoom chains, trigger re-grounding and, with an existence query, decide when to abstain; and (3) Compositional Structure Grounding, which composes spreadsheet cells from their headers. On Qwen3-VL models, AdaGround surpasses state-of-the-art training-free zooming methods and many trained GUI models, improving accuracy over the backbone by 7.4 to 20.6 points on OSWorld-G and ScreenSpot-Pro, and its gains carry over to held-out UI-Vision. Code is available at this https URL 

---
# Latent Visual Contrastive Reasoning with Multi-Branch Exploration 

**Authors**: Kegeng Tang, Jingbo Wang, Yangming Zhang, Huanshu Zhang, Shaogang Ren, Zihao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36809)  

**Abstract**: Multimodal large language models have made substantial progress in visual question answering, but complex visual reasoning often requires locating and comparing multiple pieces of question-relevant visual evidence. Existing methods that move intermediate reasoning into the visual latent space usually follow a single linear reasoning trajectory, which can be limiting when multiple visual hypotheses or competing evidence regions are relevant. We propose Latent Visual Contrastive Reasoning, a branch-aware latent reasoning framework that generates multiple question-conditioned latent visual branches as soft hypotheses over image tokens. To train these branches, our method uses bounding-box evidence annotations to build target evidence representations and introduces a soft contrastive branch-evidence alignment objective. This objective encourages evidence-relevant branches to align with annotated visual regions while contrasting them against negative evidence. We instantiate our method on Qwen2.5-VL backbones and train it with standard supervised fine-tuning. Experiments on perception-intensive visual reasoning benchmarks show that the proposed method achieves overall improvements over the corresponding backbone and single-path latent reasoning baselines at both 3B and 7B scales. 

---
# API-TAIL: Detecting Model Substitution in LLM APIs through Temperature-Aligned Likelihood Tails 

**Authors**: Haocheng Mei, Yuchuan Tian, Ye Yuan, Chao Xu, Allen Chen, Kai Han, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36801)  

**Abstract**: Commercial language model APIs return only text, allowing a provider to cut costs by quietly routing a fraction of user requests to a cheaper substitute model while billing for the advertised one. Detecting such partial substitution is challenging because honest providers also alter the output distribution through undocumented decoding and serving choices, such as temperature scaling, sampling truncation, repetition and presence penalties, and default system prompts, forcing an audit to test against a composite null of legitimate configurations rather than a single reference. Our key insight is that these legitimate variations primarily shift response typicality across the entire batch, whereas partial substitution injects individual responses that remain anomalous under the claimed model. Building on this, we introduce API-TAIL, a typicality audit that first absorbs batch-level shifts by aligning a single effective temperature to center standardized surprisal scores across the audited batch, and then flags the endpoint if the number of remaining tail responses exceeds a threshold calibrated across legitimate configurations. Evaluated on real-world commercial APIs, API-TAIL reliably controls false alarms across diverse honest providers while detecting even low-rate partial substitution with a small query budget, substantially outperforming existing rank- and MMD-based audits that lose detection power under the same composite calibration. 

---
# Select-TTT: Query-Conditioned Selective Fast Weight Updates for In-Place Test-Time Training 

**Authors**: Yue Wu, Yang Zhang, Weinan Gan, Chongming Gao, Xinsen Guo, Xiaoyuan Li, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.36794)  

**Abstract**: Test-time training (TTT) enables a deployed language model to internalize incoming context on-the-fly into a small set of fast-weight parameters, rather than merely treating it as transient context for attention. In-Place TTT recently made this idea practical for large language models by repurposing the down-projection of standard MLP blocks as fast weights. However, we find that its benefit largely disappears on realistic web long-context question answering. We attribute this to In-Place TTT's uniform write schedule: every chunk is written with equal strength, so irrelevant content dilutes query-relevant information in fast-weight memory. Building on this insight, we propose Select-TTT, a model-agnostic, query-conditioned rule for In-Place TTT that scales each chunk's write strength by its relevance to the user query. This relevance signal requires no supervision and no new parameters: it is read directly from the model's frozen output head, as the probability mass that a chunk's hidden states assign to the query tokens. A temperature-scaled softmax turns these scores into learning-rate multipliers, leaving the chunk-wise, context-parallel computation of In-Place TTT intact. Select-TTT consistently improves long-context question answering on two benchmarks and two backbones. These findings highlight the value of query-conditioned fast weight writes for practical test-time training on long-context question answering, which is also potentially promising for long-horizon agent tasks involving multiple user queries over evolving contexts. 

---
# Batching the Reflective Optimization Loop: Parallel Proposals Make GEPA Faster and Better 

**Authors**: Jialin Zhang, Lakshya A Agrawal, Shangyin Tan, Donghyun Lee, Lutfi Eren Erdogan, Dan Klein, Koushik Sen, Alex Dimakis, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2610.36774)  

**Abstract**: Reflective text optimization makes agentic systems better, but the optimization loop is itself a resource-hungry workload whose sequential structure leaves parallel evaluation capacity idle. We extend the open-source GEPA optimizer to propose and evaluate a batch of candidates on each optimization step instead of one candidate at a time. In our sweep on two tasks, most batched runs finished in half the wall-clock time or less, and the fastest in about a quarter to a third. Batched settings also achieved higher held-out test scores, from 68.9% to 72.1% on LiveBench-Math (with $2\times2$) and from 49.0% to 60.0% on HoVer (with $8\times1$). A simple runtime model explains the measured speedups under a finite worker pool, and batched settings also spend their metric-call and dollar budgets more efficiently. We frame the axes along which a reflective optimization step can scale in parallel, a first look at allocating a fixed optimization budget across them. 

---
# Balanced State Subspaces for Mamba-2 State Compression 

**Authors**: Shang Tingyu, Cao Wenfei  

**Link**: [PDF](https://arxiv.org/pdf/2610.36617)  

**Abstract**: Mamba-2 compresses all context into a fixed-size recurrent state that dominates decoding memory and bandwidth. Existing compressors quantize state values or prune state coordinates, scoring each channel independently in the native basis. We examine trained Mamba-2 states and find that energy is almost rank-one, but the output does not read the energetic directions: energy concentrates in a bias-like direction accumulated by the recurrence, while the content the output reads lives in a low-dimensional subspace that is not aligned with the state's coordinates. Even output-aware scores are computed coordinate by coordinate, so they miss the directions that determine the input--output map. Compression should therefore target the write--read subspace rather than individual coordinates. We propose \method{}, a post-training transform that instantiates balanced truncation with full controllability and observability Gramians and applies the rotation exactly via Mamba-2's scalar-identity transition. Because the useful rank varies widely across layers, we further introduce HARP, which allocates ranks per layer from the Hankel spectrum and protects retrieval-sensitive layers, improving both perplexity and retrieval over uniform budgets. Calibration takes a single forward pass, an order of magnitude faster than GHOST's released protocol, with lower peak memory. With HARP, \method{} halves the state dimension at an average WikiText-2 perplexity increase of $0.3$ points ($1.8\%$) across 130M--2.7B, versus $1.6$ ($9.3\%$) for GHOST, and the gap widens at 25\% retention. 

---
# Decoupling Consistency and Conflict across MLLM Layers for Fake News Detection 

**Authors**: Kai Qian, Dongjie Xu, Weijie Shi, Jiaqi Wang, Haolun Ma, Jia Zhu, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.36569)  

**Abstract**: The trustworthiness of Web content faces a growing threat from multimodal fake news. Real news usually shows aligned image-text semantics, while fake news often contains a few conflicts at varying levels, such as mismatches in objects, scenes, events, and claims. However, existing methods usually treat consistency and conflict in a coarse and static way, limiting their ability to capture sparse conflicts and adapt to varying conflict levels. Inspired by the distinct knowledge encoded across layers in large language models, we propose Multi-Level Consistency and Conflict Decoupling (MLCCD), which captures conflicts at different levels from different layers of multimodal large language models (MLLMs). Specifically, we first use the MLLM's final decision representation to select visual and textual tokens relevant to the task, preventing irrelevant content from diluting sparse conflicts. Then, we preserve tokens whose representations change sharply between adjacent layers, capturing the knowledge each layer adds. Finally, we decouple this knowledge into consistency and conflict evidence and aggregate each across layers with its own router, adapting to varying conflict levels. Experiments on three benchmarks show that MLCCD outperforms existing methods. 

---
# QESChunker: A Single Objective Unifies Overlapping and Non-Overlapping Chunking for RAG 

**Authors**: Yifan Zhao, Qianyang Zhao, Xinyu Guan, kai wei, Yuming Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.36557)  

**Abstract**: Chunking is a fundamental preprocessing step in retrieval-augmented generation (RAG), yet most existing chunkers rely primarily on heuristics. We introduce the Question--Evidence-Supervised Chunker (QESChunker), to our knowledge the first method for constructing static chunk indices from question--evidence supervision. QESChunker constructs document-specific chunk indices by optimizing over human-annotated or LLM-generated question--evidence pairs, without access to held-out test queries. We formulate non-overlapping and overlapping index construction through a unified question--evidence utility-maximization objective, turning chunking into a supervised combinatorial optimization problem. Under the non-overlap constraint, the objective decomposes over chunks and can be optimized exactly in polynomial time by dynamic programming. When overlap is allowed, the objective becomes monotone submodular; exact optimization is NP-hard, but our greedy achieves a $(1-1/\sqrt{e})$-approximation. Across two self-constructed and three public datasets, both variants outperform existing chunking methods in complete-evidence recall in nearly all evaluated settings, under the same overlap regime and a fixed hybrid retriever. These results recast chunking as supervised, budget-aware index construction for evidence retrieval. 

---
# PRISM-BENCH: Diagnosing Grounded and Faithful Multimodal Reasoning Beyond Final-Answer Accuracy 

**Authors**: Vansh Bhatia, Mehdi Rezagholizadeh, Parsa Ashrafi Fashi, Igor Gilitschenski, Vikram Appia, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2610.36554)  

**Abstract**: Final-answer accuracy reports whether a vision-language model was right, but not whether its answer was grounded in the image, whether its stated reasoning reflects that evidence, or what the answer cost to produce. Those are the properties that decide whether a system can be deployed. We present PRISM-BENCH, a benchmark of 1,000 expert-curated questions spanning five under-represented reasoning categories, in which every item carries an idealized reasoning trace written by one human annotator and validated by another. The traces support two diagnostics that accuracy cannot provide. A perception probe, in which the model answers from a description it wrote itself, attributes roughly 85% of frontier-model failures to misperception and places the bottleneck at grounding. Trace alignment predicts answer correctness for all 25 systems tested (macro AUROC 0.66) and survives length control, so it measures reasoning content and not verbosity. We further introduce CORE, an efficiency-aware metric that reorders 24.2% of model pairs relative to accuracy while remaining uncorrelated with response length. Across 25 open and closed systems, including GPT-5 and Claude Opus 4.6, the best accuracy is 66.0%. On the four categories with a human baseline the best system trails humans by 18.4 points, and recent frontier gains fall on recognition and not on spatial grounding. 

---
# RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding 

**Authors**: Buzhao Liu, Jianing Ye, Xiu Yuan, Yangtong Feng, Chengsong Huang, Tyler Dinh, Yevgeniy Vorobeychik, Ning Zhang, Chongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36533)  

**Abstract**: Recently, LLM-based autonomous agents have shown strong capabilities in iteratively generating, evaluating, and refining executable solutions. In robotic manipulation, Code-as-Policy lets these agents search over modular and editable robot programs. However, effective program search requires physical understanding: agents must understand how objects can be manipulated, how scene geometry constrains actions, and why interactions succeed or fail. Existing methods rely mainly on execution feedback for program refinement, often leaving these physical questions unresolved. We introduce RoboAuto, a framework for robot program search that explicitly organizes physical understanding at the object, scene, and interaction levels through three components. First, a hierarchical knowledge base combines executable skills with object-centric knowledge of affordances, spatial relations, and action effects, learned from embodied data and online task solving. Second, scene grounding connects this knowledge to the current environment by identifying relevant objects and resolving action-critical quantities. Third, multi-stage planning with self-designed experiments checks whether each stage produces its expected physical effects and actively tests competing hypotheses to resolve physical uncertainties. RoboAuto achieves superior performance across five benchmark families, attaining an 89.1% success rate on short-horizon LIBERO-PRO tasks and more than twice the mean success rate of ASPIRE on LIBERO-PRO Long and LIBERO-Plus. It also outperforms ASPIRE and $\pi_{0.5}$ on three challenging long-horizon real-robot tasks requiring precise manipulation. We further show that the knowledge base transfers across simulators and embodiments, and that program search performance scales with its size. Project page: this https URL 

---
# Folding Attention Biases into Keys Makes Compacted KV Caches Run on Standard Kernels 

**Authors**: Utkarsh Kanwat  

**Link**: [PDF](https://arxiv.org/pdf/2610.36520)  

**Abstract**: Attention Matching (AM) compacts a KV cache by keeping some keys per head, refitting their values and adding a bias to each kept key's logit. FlashAttention and vLLM's paged caches have no input for it. We fold the bias into the keys along one closed-form direction per head, fitted on context queries, giving a plain cache with AM's size and head dimension and the model's own query path; the output error is bounded by how far each query's projection on that direction is from 1. With equal per-head budgets, unmodified vLLM serves folded caches through a KV connector, matching Hugging Face's answers on all 51 Qwen3-4B questions and its greedy output on 180 of 200 Qwen3-14B LongBench samples. With equal budgets on QuALITY (Qwen3-1.7B, 4B, 8B, Llama-3.1-8B) and LongHealth (three Qwen3 models), the fold is within noise of AM's accuracy in 13 of 14 settings, with lower KL to the full cache than dropping the bias in all 12 where both ran. Under the published per-head budgets, on Qwen3-4B, its Instruct-2507 variant and Llama-3.1-8B, it is within noise of AM in 9 of 10, no longer beats dropping the bias in KL, and the two each win once in accuracy. 

---
# MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics 

**Authors**: Yibo Yang, Zerui Huang, Haomiao Yang, Zixia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36513)  

**Abstract**: As large language models advance rapidly, they are increasingly used to answer medical questions, and many such models are now deployed locally. Once a model is deployed, its behavior is governed mainly by the ethical policy written into its constitution. However, most existing work focuses on factual correctness rather than the trade-off between competing ethical values. Even benchmarks that do evaluate ethics treat it as a fixed property of the model rather than as a variable, so how a constitution reshapes medical-ethics behavior remains largely unexplored. We introduce MedConstBench, a medical-ethics benchmark that treats the deployment constitution as a controlled variable, is organized around Beauchamp's four principles, and is scored by human-calibrated multi-model peer review. It includes MedDilemma, our own dataset of clinical dilemmas, and supports a systematic analysis of how constitutions steer LLMs. Evaluating ten open-source LLMs under four constitutions, we find that the effect of a constitution is dimension-selective and model-dependent, and that no constitution is universally optimal. Motivated by these findings, we propose a do-no-harm adaptive selector that chooses a constitution per model, matching the per-model oracle while guaranteeing a worst-case floor and generalizing to unseen models at low cost. 

---
# AC-JEPA: Anchored Cross-Modal Joint Embedding Predictive Architecture for Visual-to-Symbolic Translation 

**Authors**: Chaofei Hong, Ziyang Chen, Baochen Wang, Christina Yi Jin, Qing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.36493)  

**Abstract**: Cross-modal translation requires models to capture shared semantics across modalities while preserving the fine-grained structure required by the target modality. Yet in real-world scientific domains, paired data are often imperfect, while obtaining clean pairs at scale is costly. Supervised fine-tuning (SFT) directly optimizes cross-modal translation through categorical token-level supervision, but this objective is sensitive to errors in the target sequences. JEPA offers a complementary representation-space objective, but most existing formulations focus on prediction within a modality rather than translation across modalities. We introduce Anchored Cross-Modal JEPA (AC-JEPA), an SFT-compatible plug-in that predicts masked intermediate states in both cross-modal directions using paired data. The joint-embedding objective targets cross-modal structure predictable from context, while the causal SFT path retains direct token-level supervision for the target sequence. During self-training, AC-JEPA learns from continuous latent targets derived from imperfect pseudo-labeled pairs, offering intrinsic tolerance to local data errors without promoting pseudo-tokens to hard categorical supervision. Using optical chemical structure recognition as a testbed, AC-JEPA trained on clean pairs achieves 76.07% accuracy on the Simple set of the WildMol benchmark, compared with 67.40% for SFT alone. Using the same pseudo-labeled data for both approaches, AC-JEPA-based self-training further raises accuracy to 80.23%, whereas accuracy under direct SFT drops from 67.40% to 63.47%. These results support SFT-anchored joint-embedding prediction as an effective strategy for cross-modal translation under both curated and imperfect supervision. 

---
# A Heart Means Recursion: Exploring Visual Meaning and Communication in VLMs 

**Authors**: Alexa R. Tartaglini, Zhengxuan Wu, Thaddäus Wiedemer, Martin Wattenberg, Been Kim, Yuxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.36449)  

**Abstract**: Do vision-language models (VLMs) share our ability to communicate visually? We share preliminary findings, starting from a surprising observation: simple visual symbols like a red heart can strongly influence model behavior on seemingly unrelated tasks, such as steering models towards recursive solutions on simple coding problems. Notably, this steering effect is largely implicit, as models' direct interpretations and thinking traces about visual symbols diverge substantially from their observed behavior. In visual communication between models, this leads to a generation--interpretation gap: while models tend to send shape primitives that they (and humans) find intuitive, these visual cues often fail to steer other models. In further communication games involving abstract concepts and stories, models can creatively use visuals to transmit meaning to one another, although some communications remain non-intuitive to humans. These findings reveal that VLMs might be capable of rich, creative, and visually-grounded communication, but their interpretation and production of visual meaning diverges from human conventions in surprising ways. Together this work highlights both the promise and challenges of visual communication between humans and VLMs. 

---
# SOMA: A Benchmark for Fine-Grained Egocentric Behavior and Body-Motion Understanding 

**Authors**: Matea Tashkovska, Naël Ouerghemi, Haozhe Qi, Alexander Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2610.36384)  

**Abstract**: Perceiving human body motion is critical for multimodal systems that understand and assist people. Yet current evaluations of Multimodal Large Language Models (MLLMs) largely focus on objects, scenes, and coarse actions, leaving their ability to reason about the body itself poorly assessed. This is especially challenging from the egocentric perspective, in which the body is only partially visible and motion must be inferred from occluded and temporally distributed cues. We introduce SOMA, a benchmark for fine-grained body-motion understanding from egocentric video, comprising 1,632 human-verified questions across three capability axes: body configuration, motion dynamics, and temporal reasoning. We evaluate 37 leading closed-source, open-source, and specialized egocentric MLLMs, complemented by an extensive human evaluation. We find that humans also find this task difficult from video alone (68.5% accuracy), yet clearly outperform the strongest MLLM (Qwen3.6-27B), which reaches 50.0% zero-shot accuracy. Our error analysis shows that MLLMs often misread the wearer's own body, misjudge events in time, and are prone to biases such as favoring the right hand or under-counting repetitions. To narrow the gap to human performance and support further progress, we introduce a training dataset of aligned video, body motion, and language supervision, and a multimodal baseline that jointly learns from these modalities. By identifying common failure modes and missing capabilities in current MLLMs, we hope SOMA will inspire future research on body-motion understanding and multimodal reasoning. Full benchmark, training data, and code will be released at: this https URL 

---
# Cross-modal Steering Vectors Exist Across Both Image and Text Generation 

**Authors**: Donghao Zeng, Nishant Subramani, Tamar Rott Shaham, Benno Krojer  

**Link**: [PDF](https://arxiv.org/pdf/2610.36269)  

**Abstract**: Are concepts and mechanisms unified across modalities in current AI models, or instead implemented separately for each one? We operationalize this broad question of modality unification by studying cross-modal steering: 1) finding vectors that successfully steer generation on tokens of one modality, which then 2) transfer zero-shot to steer generation when added to tokens of another modality. Our study goes beyond simple correlational alignment (e.g. representational similarity) because steering is fundamentally causal. We study cross-modal steering in several unified multimodal models (UMMs), since they possess shared parameters for language and image generation, focusing on these two modalities. Across diverse steered concepts such as emotion and object size, we show that steering vectors transfer zero-shot in both directions (text-to-image & image-to-text). Even for concepts that are highly unique to one modality (e.g. formal vs. colloquial language), steering vectors often induce a consistent, interpretable effect in the other modality, such as formal clothing and design. This transfer has a geometric signature too: despite a clear modality gap separating text and image representations, steering vectors with strong cross-modal transferability tend to point in similar directions for the same concept. Our findings shed light on how vision and language interact inside unified multimodal models and point towards more general, modality-agnostic approaches for controlling their behavior. 

---
# Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss 

**Authors**: Oz Amram  

**Link**: [PDF](https://arxiv.org/pdf/2610.36202)  

**Abstract**: The Persona Selection Model (PSM) posits that post-training selects a persona the model learned during pre-training. Work on subliminal learning has shown that models can impart covert signals (like "love of owls") into semantically unrelated outputs like strings of numbers, and that students trained on these outputs inherit those traits. In this work I show that by evaluating the likelihood of such outputs under the base pre-trained model with an appropriate header, a method I call persona-header likelihood evaluation (PHLE), one can decode these subliminal signals and classify between them. For example, PHLE distinguishes number sequences generated by a post-trained model prompted with "You love owls" from those prompted with "You love eagles", and distinguishes answers from a model with a secret desire to harm the user, filtered to remove any noticeable misalignment, from those of a model with no system prompt. In both cases prompted classifiers, including the generating model itself, remain near chance. The signal partially persists across model families, with the Qwen base model reading signals generated by Olmo and vice versa. This is evidence for a non-trivial prediction of the PSM, and the method may be practically useful for auditing the hidden objectives of models. 

---
# Learning to Read Braille: Benchmarking and Improving Braille Recognition Capabilities of VLMs 

**Authors**: Kartik Jangra, Atharva Gawane, Siddharth Pulipaka, Hemachandran Karah, Raj Dabre  

**Link**: [PDF](https://arxiv.org/pdf/2610.36195)  

**Abstract**: Despite rapid progress in visual understanding and document transcription, modern vision-language models (VLMs) struggle with a fundamental accessibility task: reading Braille from images. We introduce BrailleBench, a manually curated benchmark of 259 real-world Grade-1 and Grade-2 Braille images. We evaluate proprietary, open-weight, OCR-specialized, and tool-assisted multimodal systems and find that standalone general-purpose models perform poorly at directly transcribing Braille images, whereas tool-assisted systems perform much better. We design a synthetic Braille generation pipeline with diverse layouts and image perturbations. A 2B VLM model trained on our synthetic data reaches 87% exact match on Grade-1 and 61% on Grade-2 BrailleBench, outperforming standalone VLMs and achieving the highest Grade-2 exact match score among the evaluated systems. We further show that high accuracy on synthetic validation data does not guarantee high accuracy on real photographs in our experiments: models trained on prior synthetic data reach up to 96% on their synthetic validation set but at most 3.26% on Grade-1 BrailleBench. Our results identify an accessibility-relevant capability gap in multimodal systems and show that targeted synthetic supervision can improve real-world Braille transcription. 

---
# Don't Pick a Winner: Knowledge Conflicts Need a Prior, Not a Rule 

**Authors**: Minh Pham Van, Cong Tran  

**Link**: [PDF](https://arxiv.org/pdf/2610.36169)  

**Abstract**: A retrieval-augmented language model answers from two sources, the retrieved context and its own knowledge, and the two disagree whenever the context is incomplete or wrong. Existing methods settle the disagreement with a rule fixed at design time: follow the context, amplify it in proportion to how far it moves the model, or trust the model's memory. Each rule is confidently wrong in some deployment. In a decision-theoretic model of two sources, the score of any fixed rule is linear in the share of reliable contexts while the best attainable score is convex, so every fixed rule is beaten wherever it is not exactly optimal; and a symmetric divergence between the two sources, the signal adaptive decoding uses, carries no information about which one is right. Resolving a conflict therefore needs two inputs that no existing method keeps: evidence about each source separately, and a prior on how often the context is right. We test these claims on GRACE, a benchmark that intervenes on the knowledge-graph context of a fixed question, so that what the model knows, what the context says and what is true are measured separately. There, every baseline from four families is strong on right contexts or on wrong ones, never on both. We then build the two inputs: the model states its knowledge as notes, a small policy voices each note with a calibrated confidence, a reader answers in two voices, and a selector commits or abstains under a prior $\pi$ applied as an exact prior shift. With $\pi$ set to the true share of right contexts, the system is above the best baseline at every mixture of GRACE contexts, by up to 26 points. Its overall score is 45.3 and 35.0 (Qwen3.5-35B-A3B, Llama-3.1-8B), against 29.4 and 27.5 for the best baseline. On preregistered ClashEval it beats same-backbone CK-PLUG and TCR by 12 and 26 points. Where the model lacks the knowledge, the gains vanish, as the theory predicts. 

---
# ConformalMOS: Uncertainty-Aware MOS Prediction with Conformal Intervals and Ordinal Modeling 

**Authors**: Kehinde Elelu, Joshua E. Siegel, Mohammadali Saffary, Tashfain Ahmed, Simeon Babatunde, Ebuka Okpala  

**Link**: [PDF](https://arxiv.org/pdf/2610.36126)  

**Abstract**: Accurate prediction of human Mean Opinion Scores (MOS) is crucial for evaluating synthetic speech quality in text-to-speech (TTS) and voice conversion (VC). Most MOS predictors output only point estimates and ignore uncertainty, limiting reliability in model selection and deployment. We propose ConformalMOS, a framework that augments MOS prediction with conformal prediction to produce statistically valid prediction intervals with guaranteed coverage under exchangeability, alongside point estimates. During training, MOS labels are modeled in an ordinal-aware manner by converting one-hot targets into Gaussian-smoothed distributions, preserving score ordering. This improves uncertainty calibration, particularly when softmax confidence is overconfident on out-of-distribution speech. On BVCC datasets, ConformalMOS achieves competitive accuracy (MSE = 0.08) while delivering empirically validated coverage, enhancing reliability for real-world TTS and VC deployment. 

---
# Beyond Update Counts: A Control Map for Evaluating Test-Time Model Updates 

**Authors**: Zehao Zhao, Shangjian Yin, Rui Liu, Chongyang Bai, Mark Ibrahim, Kavosh Asadi, Yuchen Lu, Hang Cui, Hamed Firooz  

**Link**: [PDF](https://arxiv.org/pdf/2610.36109)  

**Abstract**: Test-time updating lets a language model change persistent state after training---such as model weights, adapter parameters, or a memory module---before or while answering queries. Existing evaluations often compare systems using update count or frequency and performance measured immediately afterward. These summaries omit what is updated, where, when, and in what order. The same total can therefore hide an important difference: one system may answer after one update while another answers after two. We introduce the Control Map, an evaluation framework that records the committed updates preceding each scored answer and separately evaluates behavioral changes, immediate learning (acquisition), later recall (retention), preservation of unrelated behavior (locality), and performance in new settings. We prove when counts alone determine final state and show, under stated assumptions, how differences between pre-answer histories limit differences in average scores. Using ControlMap, our experiments reveal three insights. First, update scheduling matters: at a 760M-parameter checkpoint, 24 schedules with the same update count yield 19.7%--34.2% accuracy. Second, learning an edit does not guarantee that it lasts or remains contained: all six selected CounterFact LoRA settings learn the new information, while ROME retains edits at all 13 operating points but fails locality at each. Third, no editor performs best consistently across models, benchmarks, and tests. Evaluations should therefore report update histories and assess acquisition, retention, and locality separately. 

---
# How Much Does the Hint Matter? Scope Conditions for Hindsight Self-Distillation in Financial Reasoning 

**Authors**: Johnny Yu, Cyril Gorlla  

**Link**: [PDF](https://arxiv.org/pdf/2610.36082)  

**Abstract**: Financial institutions operating under data-residency constraints are restricted to open-weight models small enough to locally host, which trail frontier systems on multi-step quantitative reasoning. On-policy distillation narrows such gaps by supervising a student on its own rollouts, but traditional OPD requires a strong external teacher, and on-policy self-distillation (OPSD) conditions on a full reference solution. We ask how much of that signal is necessary. We present Fin-SD, a localized hindsight self-distillation recipe for multi-step reasoning. Given a problem the student fails, the student itself identifies the first incorrect reasoning step and writes a short critique of that step. We truncate the student's trajectory immediately before the error (aiming to obtain a correct prefix), condition a frozen copy of the student on the same prefix plus the hint, and distill that hint-conditioned teacher into the student over a short rollout under reverse KL. No external model contributes a gradient. We apply Fin-SD to GPT-OSS-20B on a custom multi-step financial reasoning benchmark that requires deriving formula inputs from realistic scenarios via accounting conventions. We find: (i) replacing the critique with a generic-feedback control matches Fin-SD's entire 3.03 pp accuracy gain over the base model on the FinanceReasoning Hard subset, suggesting the gain comes from where supervision is applied, not hint content; (ii) the model writing the hint does not need to be stronger than the model receiving it: hints from same-family GPT-OSS-120B, at roughly 6x the parameters, yield no materially greater benefits than self-written hints; and (iii) Fin-SD matches on-policy self-distillation from full reference solutions on accuracy while generating 9.5% fewer tokens. With Fin-SD, general reasoning on MMLU-Pro and a fourteen-metric compliance audit on FinTrust show no material regression. 

---
# SCOPE: Source Credibility and Outlet Profiling Engine 

**Authors**: Ilyas Nagiyev, Miras Baisbay, Daniil Orel, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2610.35929)  

**Abstract**: We present SCOPE, a multi-agent system for profiling news media outlets with respect to bias, factuality, and credibility. Given an outlet URL, SCOPE coordinates specialized agents for article acquisition, content analysis, external evidence search, scoring, and report generation. The system follows an MBFC-style methodology while making intermediate evidence and agent outputs inspectable through a structured demonstration interface. On a benchmark of 200 media outlets, SCOPE outperforms LLM-only, article-only, search-only, and media-background-check baselines on most metrics, achieving the lowest bias and factuality MAE, perfect failed-fact-check detection, and the lowest factual error rate. These results suggest that decomposed, evidence-grounded agent workflows can support more transparent and reliable media profiling. Our code is available at this https URL, and a live demonstration at this https URL. 

---
# Physiognomy Revisited: How Vision-Language Models Encode Sensitive Attributes Through Image Transformation 

**Authors**: Soumyajit Datta, Arka Dutta, Ashiqur R. KhudaBukhsh  

**Link**: [PDF](https://arxiv.org/pdf/2610.35910)  

**Abstract**: Emergent generative AI technologies, while remarkably beneficial across several tasks and domains, can suffer from unforeseen vulnerabilities and difficult-to-anticipate failure modes. Such failure modes can disparately affect vulnerable communities. This paper investigates two well-known vision language models (VLMs) and inspects how these models encode protected identities while generating images. We show that when VLMs are requested to transform human face images to reflect race identities (e.g., Hispanic, Black, White, or Asian) and sexual orientation (e.g., gay or straight), VLMs often comply with such requests. We demonstrate that compliance (and refusal) is non-deterministic, group-sensitive, and generated images are statistically distinguishable, often accentuating deep-seated stereotypes. We further show that representation biases introduced during image generation can re-emerge in textual reasoning and characterization tasks. Finally, we show that this worrisome finding is not limited to racial identity or sexual orientation, image generation can perpetuate bias around criminality, raising concerning possibilities of AI-assisted profiling. 

---
