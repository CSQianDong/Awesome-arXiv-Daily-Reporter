# Strategically Diverse Sampling for Self-Training 

**Authors**: Alexander Gurung, Esmeralda S. Whitammer, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2609.31571)  

**Abstract**: Many LLM training and inference methods, including RL and test-time scaling, depend on repeated sampling, but benefit only when the responses meaningfully differ. Self-training faces the same challenge: training data is typically constructed by sampling IID responses and filtering primarily for correctness, thereby overrepresenting strategies a model already favours. We investigate strategic diversity, or substantive variation among approaches to a problem, as an alternative principle for constructing self-training data. We generate strategically diverse data with two sampling methods: GROOT, a new method which constructs a hierarchical tree of approaches and samples distinct paths, and Verbalized Sampling (VS), adapted to produce an unstructured set of approaches. Across competitive programming and Next-Chapter Prediction domains, models trained on strategically sampled data outperform IID-trained counterparts on difficult tasks and provide strong initializations for RL and test-time scaling. Most strikingly, self-training on strategically diverse but incorrect traces from Qwen3-4B outperforms IID distillation from a 235B teacher. These results challenge prevailing assumptions about what makes useful self-training data and show that diversity of approaches can matter more than correctness or teacher scale. 

---
# MexHat: A Dataset for Hate Speech Detection in Mexican Spanish Videos 

**Authors**: Itzel Tlelo-Coyotecatl, Hugo Jair Escalante  

**Link**: [PDF](https://arxiv.org/pdf/2609.31553)  

**Abstract**: Ensuring online safety through content monitoring had raised Hate Speech Detection as a crucial task to be addressed. By essence the task demands the capture of contextual cues, which are essential for a precise understanding of the content's intent. Although automated detection approaches for the task have advanced significantly, the scarcity of non-English resources persists, limiting the ability of models to adapt to the subtle, context-dependent, and culturally related nature of multimodal content. In this paper, we introduce MexHat, a video dataset designed to capture the linguistic and cultural cues for the hate-speech detection task in a Mexican Spanish context. Our dataset comprises around 1k video clips annotated across two tasks: a three-way class evaluation (no negative content, offensive content and hate-speech content), and a fine-grained class evaluation including three hate-speech sub-categories. The dataset statistics and the baseline results highlight the inherent challenges associated with the task. Disclaimer: This paper contains sensitive content that may be disturbing to some readers. 

---
# Muslim: A Deployed Arabic Voice AI Platform for Grounded Islamic Knowledge 

**Authors**: Yahya Mohamed Elnawasany  

**Link**: [PDF](https://arxiv.org/pdf/2609.31511)  

**Abstract**: We present Muslim, a production Arabic voice AI platform serving grounded, sourced Islamic knowledge to real users. Beyond a real-time voice pipeline (NeMo Arabic ASR, an OpenAI-compatible LLM endpoint, self-hosted TTS) and a deterministic multi-source retrieval layer routed across six Model Context Protocol servers, we report three things a research prototype typically lacks. First, a released family of fine-tuned Arabic Islamic model artifacts: an efficient tool-routing LLM (Muslim-6B-PRO, 5.94B parameters) and a Modern Standard Arabic TTS model (Fasih-TTS-V1) that ranks 5th of 17 overall and 2nd of 11 open-weight systems on the community-voted Arabic TTS Arena for MSA. Second, an account and metering layer - a free per-account turn allowance, capacity-aware refusal, and email verification deferred to the point it actually matters - that turns an open demo into an operable, abuse-resistant product. Third, a three-layer observability stack (liveness, error reporting, product analytics) built specifically around the system's characteristic failure mode: a GPU-bound agent host going silent while the web tier keeps serving normally. We report real, measured latency and accuracy figures (98.4% recitation-validation accuracy on 124 cases; end-to-end voice latency of 0.9-1.7s) and discuss the concrete engineering trade-offs and limitations of running an Islamic-knowledge voice product in production. 

---
# Evaluating Cultural Awareness of LLMs for Haitian Creole 

**Authors**: Christelle Clervilsson, Yanzhu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.31506)  

**Abstract**: Large language models (LLMs) exhibit substantial performance disparities between high- and low-resource languages. Beyond lower task performance, they often fail to capture the cultural norms and values of underrepresented communities. In this work, we present the first systematic evaluation of cultural awareness in LLMs for Haitian Creole, a language spoken by millions but severely underrepresented in digital resources. We assess cultural awareness along four complementary dimensions---specificity, bias, diversity, and variation---using a benchmark of culturally salient prompts curated by native speakers in a text infilling setting. Our results reveal a clear gap between cultural awareness in Haitian Creole and higher-resource French, with Haitian performance being more uneven across domains and more affected by French linguistic interference. Story generation further reveals recurring portrayals of Haitian characters through hardship and resilience, showing that even positive characterizations can encode stereotypical narratives. Our code, benchmark, and evaluation framework are publicly available. 

---
# ViSTA: A Simple Bridge Extends Visual Alignment to Clinical Time-Series Understanding in Multimodal LLMs 

**Authors**: Junyi Gao, Yu Shi, Pingzhao Hu, Ewen M Harrison  

**Link**: [PDF](https://arxiv.org/pdf/2609.31448)  

**Abstract**: Clinical prediction models estimate risk from patient measurements, while large language models support medical text understanding and question answering. Yet their language capabilities do not ensure accurate prediction from structured, high-dimensional clinical time series. Improving this ability would connect risk estimation with flexible questions about a patient's evolving condition. We introduce ViSTA, a compact adapter that incorporates irregular numerical measurements into a pretrained vision-language model's chart representations. It learns corrections to visual tokens while leaving all pretrained parameters unchanged. On MIMIC-IV, ViSTA has the highest mean scores among the compared adaptations on all four metrics for acute kidney injury and mortality prediction across models with 2-9 billion parameters. With 0.516 million trainable parameters, the 2-billion-parameter model reaches an area under the ROC curve of 0.7376 for acute kidney injury, compared with GPT-5.6 Sol's 0.7380 with text input and high reasoning effort. Training for temporal question answering yields 69.27% accuracy at 4 billion parameters with over 90% fewer trainable parameters than low-rank adaptation using charts or numerical text, at a 2.82-4.88 percentage-point accuracy gap. ViSTA extends pretrained language models to numerical prediction and temporal questions. 

---
# Sorry Robot, Happy Human: Vision-Language Models Read Only One of Two Legible Typographic Layers 

**Authors**: Mert İncidelen, Yamen Kashkash, Asya Berker, Murat Aydoğan  

**Link**: [PDF](https://arxiv.org/pdf/2609.31403)  

**Abstract**: Vision-language models (VLMs), despite their success in optical character recognition (OCR) tasks, are vulnerable to typographic attacks and have a fragile structure for images with multiple text layers. In this study, the DecoyBench dataset was created using the Decoy Font method. The dataset consists of 300 images, each containing text with sharp contour lines superimposed on another text with soft shading. Six recent closed-source models from three different model families were evaluated using this dataset under two different prompting conditions (naive and guided) and at two different resolutions ($512\times512$ and $64\times64$). A validation study showed that human participants could read both text layers with high accuracy. In contrast, the models, with most variants and both prompting methods, read the contour text with near-human accuracy at high resolution, but almost never fully extracted the shading text. At low resolution, the contour text could not be read by either the models or humans, while the shading text could be extracted with high accuracy. The findings indicate that the evaluated VLMs exhibit a consistent behavioral limitation when processing typographic structures containing multiple spatial frequency layers. 

---
# Highlight-Then-Summarize: Learning to Compress Evidence for Long-Context Understanding 

**Authors**: Zhaoyuan Xia, Qinghongbing Xie, Yung Xiang Hue, Jianguang Jiang, Gaofeng Lu, Zhenyu Jiao, Xing Yuan, Dai Dai, Tong Mo, Long Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2609.31382)  

**Abstract**: Long-context understanding requires large language models (LLMs) to reason over lengthy documents, conversations, and code, yet task-relevant evidence is often sparse and scattered amid substantial irrelevant and redundant content. We propose Highlight-Then-Summarize (H2S), a compress-then-reason paradigm that first identifies source-grounded, question-relevant evidence and then integrates it into a compact, question-conditioned summary before producing the final answer. To train this behavior, we construct H2S-Dataset, comprising 6,647 examples from 11 benchmark families with an average context length of 43.9K tokens, and introduce H2S-RL, which provides process-level rewards for evidence selection and summary construction in addition to final-answer correctness. We evaluate on H2S-Bench, a seven-task long-context suite. Under a shared 128K input and 4K output budget, H2S-14B achieves an average score of 32.60, outperforming Qwen3.8-27B by 10.17 points and obtaining the strongest overall result among the evaluated open-source models. H2S-14B also achieves the highest Evidence-Summary Quality score and retains 97.1% of its 16K-budget performance with only a 4K output budget. These results show that explicitly selecting and integrating evidence improves long-context reasoning while enabling more compact generation. 

---
# Stale-Document Poisoning: When Outdated Retrieval Overrides Correct Model Answers 

**Authors**: Md Shamim Ahmed, Lukas Galke Poech, Richard Röttger  

**Link**: [PDF](https://arxiv.org/pdf/2609.31342)  

**Abstract**: Retrieval-augmented generation (RAG) is often used to address outdated knowledge by providing external evidence. But retrieval helps only when that evidence is still valid. We identify a temporal alignment failure, stale-document poisoning, in which outdated evidence makes a model wrong despite answering correctly without retrieval. We construct a benchmark of 317 verified knowledge reversals across medicine, law, software, and platform policy, grounded in dated official sources. Across 12 models, recent medical reversals are harder than long-established ones. More importantly, outdated retrieval flips 30% of Llama and 37% of Qwen answers even without instructions to trust the document; explicit follow instructions raise these rates to 66% and 75%. Across four open models and four domains, poisoning ranges from 17-91%, while matched up-to-date evidence is followed in 97-100% of trials. To isolate temporal applicability, we keep the historical evidence unchanged across 50 reversals and vary only the evaluation date. A clear pattern emerges: dates alone produce only modest adaptation, but when models are explicitly told when the old evidence stops applying, the larger models switch to the appropriate answer almost perfectly. Causal interventions confirm that this validity information directly shapes the final decision. The same internal components also support broader comparison tasks, suggesting that temporal applicability can recruit a general reasoning mechanism used for other comparisons. Finally, a fixed recency-aware hybrid re-ranker reduces poisoning by 4.6-10.0 points when dates are accurate, with gains that depend on reliable temporal metadata. Reliable RAG therefore requires selective trust: models must determine not only what retrieved evidence says, but whether it still applies. 

---
# Identifying Scientists on X 

**Authors**: Philipp Meier, Katarina Boland, Laura Kallmeyer, Stefan Dietze  

**Link**: [PDF](https://arxiv.org/pdf/2609.31264)  

**Abstract**: With the growing importance of science-related discourse on the Web and the erosion of the classical knowledge order, it is important to identify different user groups, such as scientists, automatically. This work proposes an approach for identifying scientists and non- scientists on X/Twitter based on their user biographies and tweets. We show that we are able to classify accounts as scientists and non- scientists on two different datasets, reaching an F1 score of up to 0.88 using Random Forests with linguistic features and up to 0.96 using a contrastively fine-tuned DeBERTa model in an ensemble setup. Furthermore, we provide two datasets with X users labeled as scientists or non scientists and their respective tweets and user biographies. 

---
# MoSAR: Mixture of Semantic Attention Regimes for Learning Adaptive and Approximable Attention Geometries 

**Authors**: Michele Paolicelli, Alessandro Petruzzelli, Alessandro Franceso Maria Martina, Cataldo Musto, Giovanni Semeraro  

**Link**: [PDF](https://arxiv.org/pdf/2609.31261)  

**Abstract**: The quadratic complexity of dense self-attention remains a central bottleneck for long-context language modeling. Many efficient alternatives address this cost by deciding in advance where attention should be sparse or local. We argue that attention approximation should instead be approached as a geometric problem, with the relevant interaction geometry learned from data: natural-language dependencies are input-dependent and difficult to prescribe in advance, so the model should learn where positional relevance can decay and where broader interactions must be preserved. We introduce Mixture of Semantic Attention Regimes (MoSAR), which learns such an adaptive, controlled-decay geometry over query--key interactions. Input-conditioned query and key routers, applied after positional encoding, select mixtures over short, medium, and global regimes, inducing a continuous distance-dependent attention field rather than a fixed sparsity pattern. This geometry is learned during training and can subsequently be discretized through top-1 routing. In controlled pre-training experiments with matched 500M-parameter models, MoSAR learns a substantially lower-reach attention geometry without degrading language-modeling quality, improving perplexity over dense RoPE at the training context length. Under length extrapolation, MoSAR achieves the best perplexity among all evaluated variants, including strong baselines such as ALiBi. Moreover, the learned geometry remains stable under deterministic top-1 discretization, suggesting that it is not only adaptive, but also amenable to low-cost approximation at inference time. 

---
# PIA: A Personal Intelligence Agent Turning Health Conversations into Records and Records into Understanding 

**Authors**: Jeonghun Yoon, Dongchan Kim, Hongyeon Yu, Young-Bum Kim, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2609.31255)  

**Abstract**: General-purpose agent memory summarizes conversations: it extracts salient snippets, embeds them, and retrieves the top-k into the prompt. A health agent cannot run on summaries: a dose becomes a sentence, "since last week" is resolved at the model's discretion, and a three-month glucose trend cannot be answered by text similarity. We present PIA, a personal intelligence agent deployed alongside a consumer health agent. PIA receives the agent's natural-language requests, decides for itself whether and how to write or read, and turns conversations into typed clinical records and records into a synthesized understanding of the user. Its memory harness consists of four controls -- extraction, memory, retrieval, and understanding -- each a domain-agnostic mechanism with a pluggable health module: schema, medical alias dictionary, knowledge graph, and temporal rules. We show how the same query receives a different answer as the memory injected into the response context deepens from one-dimensional recall, to a two-dimensional health snapshot, to a three-dimensional trajectory with causality, and report lessons from operation: self-reported health data are missing not at random, question phrasing governs the quality of synthesized understanding, and nearly a third of candidate causal links are structural noise that rules alone remove. 

---
# RupeeBias: Auditing Demographic Bias in Indian Economic Guidance from Large Language Models 

**Authors**: Pavithra P M Nair, Bhavik Talaviya, Shourya Bhushan, Rahul Pankajakshan, Seema Guruvadoo, Avinash Agarwal, Gilad Gressel, Krishnashree Achuthan  

**Link**: [PDF](https://arxiv.org/pdf/2609.31245)  

**Abstract**: Individuals turn to large language models (LLMs) for guidance across a wide range of economic tasks, from comparing loan options and planning savings to deciding what raise to ask for or how much to charge for their services. LLMs are known to reproduce social biases, and biased economic guidance may influence what users believe they are worth, what they ask for, and what they ultimately accept. This risk is especially salient in India, where economic outcomes are shaped by demographic categories such as caste and urban-rural location. Existing LLM bias benchmarks, however, are largely designed around Western demographic categories and therefore miss key axes of economic disparity in the Indian context. We introduce RupeeBias, a benchmark for auditing demographic bias in LLM-generated economic guidance across Indian economic settings. RupeeBias consists of 39,150 prompts spanning four use cases: salary estimation, salary increment estimation, counter-offer recommendation, and service pricing recommendation. The benchmark follows a single-attribute counterfactual design, holding the description of the user's qualifications, experience, or service offering fixed while varying one demographic identifier at a time. RupeeBias covers 87 India-specific demographic identifiers across six axes: caste, religion, regional identity, gender, disability, and urban-rural location, with all prompts constructed in both English and Hinglish. We evaluate nine LLMs on RupeeBias and find systematic demographic disparities across all six axes. For otherwise identical prompts that differ only in demographic identifier, LLM-generated economic outputs differ by 20.2% on average. We publicly release RupeeBias to support future research on demographic bias in LLM-generated economic guidance across India-specific demographic and economic contexts. 

---
# Where a Model Sends Its Own Repeated Token 

**Authors**: Nicolás Vera Zúñiga  

**Link**: [PDF](https://arxiv.org/pdf/2609.31181)  

**Abstract**: Black-box model identification works by scoring a model's response to natural-language prompts. One line of work feeds models a degenerate input -- their own token, repeated -- to find a failure mode rather than an identity. We take that input and ask where the model goes when it does not. For each token t, read argmax p(. | t, t) in one forward pass; the result is a map on the whole vocabulary, with two halves. The first -- which tokens are fixed points -- is partially anticipated, and we report it as a failed estimand: the natural distance on it is 83% cardinality, separates a corpus manipulation by two bits in 3471 against a precision floor of zero, and attributes families at 0.5833. The second half, where the map sends tokens that are not fixed points, is unrecorded; the one paper holding those tokens logged them as a zero. Pairing on the source token removes the cardinality confound by construction (r from 0.9128 to -0.0932) and attributes families at 0.8333 -- twelve models scored against a pool of nineteen -- with chance 0.1389, across seven tokenizer groups and several corpora. Two nulls clear it: frequency-matched destinations agree at 0.1429, independent marginals at 0.0798. Family predicts agreement better than tokenizer (0.2031 against 0.1205), and recurrent architectures cluster at balanced accuracy 1.0 against a 0.7895 majority rate, or 0.90 once each model's dominant destination is excluded -- the figure we stand behind. We measure the robustness envelope: 8-bit weight rounding moves the map less than deduplicating the training corpus does (0.9004 against 0.6353, on one support), 4-bit destroys it (0.0098; 0.1812 at deployment granularity, so not a coarseness artefact), and the precision floor varies by model from 0.201 to 0.9778. All estimands and kill conditions were registered before the data, and the failed one is reported as fully as the surviving one. 

---
# Improving Visual Sensitivity of LLMs on Multimodal Machine Translation with Metric-based Loss Weighting 

**Authors**: Paweł Mąka, Piotr Andruszkiewicz, Yusuf Can Semerci, Jan Scholtes, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2609.31169)  

**Abstract**: Multimodal Machine Translation aims to incorporate additional signal from non-textual modalities to improve translations by resolving ambiguities. While models, through multimodal fusion, are able to accept images related to the source text, they can ignore this information. Therefore, increasing their visual sensitivity remains an active research area. In this work, we introduce a training method, Metric-based Loss Weighting, that improves visual grounding of translations by increasing the loss function for tokens that benefit from the accompanying image. We identify these tokens using the Point-wise Cross-mutual Information (PCXMI) metric, which compares the model's output probabilities with and without visual context. We introduce a Congruency-based PCXMI metric and experimentally show that both metrics working in combination yield the best results. We evaluate our method by fine-tuning three pretrained Multimodal Large Language Models on the task of Image-guided Machine Translation for three language directions. Metric-based Loss Weighting outperforms other tested methods on the CoMMuTE contrastive dataset, improving accuracy by up to more than 7 percentage points compared to standard fine-tuning, while maintaining strong general translation performance. 

---
# Do we need to answer that question? Salience and Answerability of Potential Questions in Naturalistic Dialogue 

**Authors**: Amandine Decker, Maxime Amblard, Ellen Breitholtz  

**Link**: [PDF](https://arxiv.org/pdf/2609.31130)  

**Abstract**: We empirically investigate Question Under Discussion based modelling in naturalistic dialogue by studying whether the salience of generated potential questions predicts their subsequent resolution. Building on Wu et al. (2024), we construct a dataset of 7,124 questions automatically generated from utterances and preceding context from the British National Corpus, and annotated for salience and answerability. We find a robust but low positive correlation between salience and answerability in dialogue, indicating that more salient questions are more likely to be addressed. However, this effect is markedly weaker than in monologic text, suggesting that conversational structure is less predictable. We further observe that structured interactions exhibit stronger alignment between annotators than less organised dialogues. 

---
# LocUS: Head Selection and Subspace Projection for Targeted Activation Steering 

**Authors**: Irene Tallini, Lorenzo Basile, Valentino Maiorca, Francesco Locatello, Alberto Cazzaniga  

**Link**: [PDF](https://arxiv.org/pdf/2609.31122)  

**Abstract**: Activation steering is a powerful training-free paradigm for controlling large language models at inference time. However, standard approaches estimate a per-layer steering direction from contrastive data and apply it on the layer's entire representation space, which may couple the intervention to off-target properties present in the contrastive data and degrade unrelated capabilities. To mitigate this issue, we introduce LocUS (Localized Unembedding Steering), a method which grounds activation steering to the model's own output vocabulary subspace. By identifying a property-specific linear subspace within the unembedding matrix, LocUS enforces a geometric constraint that restricts the steering transformation to a specific subspace and at the same time localizes its application to a sparse subset of attention heads. Extensive evaluations across three model families on toxicity mitigation, sentiment redirection and sycophancy suppression show that LocUS matches or outperforms state-of-the-art baselines while intervening on under 6% of parameters and better preserving general capability. 

---
# CG-Probes: Recovering Guardrail Directions from Patient Query Embeddings 

**Authors**: Marko Řeháček, Vítězslav Dušek, Martin Rusinko, Vít Nováček  

**Link**: [PDF](https://arxiv.org/pdf/2609.31062)  

**Abstract**: Patient-facing AI assistants promise valuable support to patients, but incoming queries can pose medical risks. To create guardrails, we work with oncologists to define three ordinal risk axes: Medical Urgency, Psychological Urgency, and Topic Sensitivity. We propose Clinical Guardrail Probes (CG-Probes) to measure the risks from query embeddings. We probe for each axis in the normalized embedding space of frozen embedders via the difference-in-means method, treating each axis as a potential linear direction. To train the probes, we cluster 79,658 Czech oncology search queries with BERTopic and use these clusters to generate pairs of queries with contrastive risk levels via few-shot prompting. We evaluate the approach on 200 queries (90 real, 110 synthetic), each graded by two oncologists, against two open-weight LLMs and a frontier LLM. We find that urgency-based axes are recoverable as linear directions, and the probes are competitive with open-weight LLMs (no significant differences in quadratic-weighted kappa) at a fraction of the latency. Each axis yields a scalar score that clinicians can inspect and use to set escalation thresholds. The pipeline requires only search logs, axis definitions, and black-box access to the embedding model, suggesting transferability across healthcare domains. Robust validation on new queries and axes remains future work. 

---
# Modeling Student Sensemaking with LLMs and Knowledge-Graph-Guided Inference 

**Authors**: Özge Alacam, Zübeyde Demet Kirbulut Güneş, Funda Ekici, Nurcan Turan-Oluk, Dilay Dinçdemir, Hakkı Kadayıfçı, Sevinç Nihal Yeşiloğlu, Burcu Işık, Halil Tümay, Sinem Gencer  

**Link**: [PDF](https://arxiv.org/pdf/2609.31046)  

**Abstract**: Collaborative science learning requires nuanced interpretation of student dialogue to characterize how learners identify knowledge gaps, build explanations, and work toward resolution - a theory-driven analysis that is labor-intensive and difficult to scale. We investigate whether instruction-tuned large language models (LLMs) can support multidimensional analysis of collaborative sensemaking without task-specific training, and whether structured knowledge-state information improves model inference. We evaluate two mid-size LLMs on 23 richly annotated, expert-labeled episodes across prompting conditions that vary definitional scaffolding, reasoning mode, and turn structure. Without reasoning, models tend to overpredict successful sensemaking; reasoning-enabled prompting improves identification of unsuccessful cases. Knowledge-state diagnostics provide additional grounding, improving detection of unsuccessful sensemaking and increasing agreement with expert annotations. No single configuration performs best across all sensemaking dimensions, underscoring the multidimensional nature of the task. 

---
# G$^2$PTQ: Improving LLM Post-Training Quantization with Generalized Gradient Compensation 

**Authors**: Ruikang Liu, Haoli Bai, Yuxuan Sun, Qian Zhang, Wenzheng Cai, Yanqi Hao, Feiyu Wang, Weidong Zhong, Zhuang Wang, Tong Yang, Xiangsheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.31009)  

**Abstract**: Post-training quantization (PTQ) is a practical approach to reducing the memory and computational footprint of large language models (LLMs) without retraining. GPTQ-based methods have become the de facto standard, yet they suffer from two complementary limitations. Methods with local, layer-wise objectives lack global supervision; while methods with global objectives fix their Hessian estimates at the start and ignore first-order gradients, so their guidance grows stale as quantization proceeds. This paper presents G$^2$PTQ, a unified PTQ framework with Generalized Gradient Compensation that integrates both first- and second-order information under a globally supervised, block-wise optimization objective. By refreshing gradient and Hessian estimates before quantizing each Transformer block, G$^2$PTQ avoids the staleness of prior global methods. Furthermore, to stabilize the exact first-order compensation, we introduce a trust-region scaling mechanism that dynamically bounds the gradient step to prevent exploding weight updates. Finally, we derive efficient implementations for block-wise Hessian approximation and exact gradient compensation. Experimental results on various model families and bit-widths demonstrate that G$^2$PTQ enables better alignment with the full-precision model, outperforming state-of-the-art baselines. Code is available at: this https URL. 

---
# ZooWork-ShopRanker: An Open, Preference-Aligned E-Commerce Reranker 

**Authors**: Siqiao Xue, Shuxuan Liu, Ning Hu  

**Link**: [PDF](https://arxiv.org/pdf/2609.31002)  

**Abstract**: Open rerankers trained for general web retrieval transfer imperfectly to e-commerce, where ranking decisions depend not only on topical relevance but also on user preferences, product constraints, and comparative product fit. These preference signals are difficult to supervise at scale: real search traffic provides authentic queries and candidates but no clean pairwise labels. We present ZooWork-ShopRanker, a family of e-commerce rerankers (0.6B, 4B, and 8B) aligned to judge-labeled shopping preference. Training pairs are labeled by a panel of reasoning large language models (LLMs) from different families acting as a preference oracle, with position-debiased judgments and agreement tiers, and the rerankers are trained on these labels. The aligned 8B flagship then serves as a distillation teacher for the efficient 4B and 0.6B models, which are fit to its scores and sharpened on judged pairs. To measure progress, we introduce ShopRank-Bench, a contamination-limited benchmark of ~10,000 private-traffic preference pairs in both text formats, tiered by how many judge families committed to each label. ZooWork-ShopRanker-8B and -4B significantly outperform the strongest open reranker baseline, every model significantly beats its own un-aligned base, and ZooWork-ShopRanker-0.6B beats its size peer; the gains hold in both formats and extend to common MTEB benchmarks. We release the models and the dual-format ShopRank-Bench to facilitate further research. 

---
# Evaluating Sycophancy in Chinese Large Language Models on Factual Questions Derived from Online Search Queries 

**Authors**: Geng Liu, Feng Li, Mengxiao Zhu, Francesco Pierri  

**Link**: [PDF](https://arxiv.org/pdf/2609.30986)  

**Abstract**: As large language models increasingly mediate information access, factually accurate and independent answers are critical. However, these models can exhibit sycophancy by aligning their responses with users' stated beliefs even when those beliefs are incorrect, potentially presenting misinformation as independently verified and reinforcing users' confidence in false claims. Prior work leaves unresolved whether introducing user beliefs causes correct responses to become incorrect or uncertain, or causes uncertain responses to become belief-aligned incorrect answers. It also remains unclear whether anti-sycophancy interventions preserve or restore factual accuracy or merely shift responses toward uncertainty. We analyze factual sycophancy in Chinese-language information seeking using yes/no fact-checking questions. Our analysis covers 364,941 responses from three frontier Chinese-based LLMs (DeepSeek, Qwen, and Doubao) to 12,165 factual questions derived from real-world Chinese search queries. We evaluate the models with and without reasoning across baseline, belief-conditioned, and anti-sycophancy prompting, tracing matched shifts among correct, incorrect, and uncertain responses. Under incorrect user beliefs, we distinguish belief-aligned errors from losses of factual confidence, in which initially correct answers become uncertain. Patterns vary across models and reasoning settings: reasoning is not a consistent safeguard, and anti-sycophancy instructions can reduce incorrect agreement while increasing uncertainty. In Chinese-language factual question answering, avoiding agreement with false beliefs is therefore not equivalent to preserving factual accuracy, highlighting the value of transition-level evaluation. Such behavior may undermine the reliability of LLM-mediated information access by reinforcing misinformation or weakening users' confidence in factually correct answers. 

---
# THA: Weighted Finite-State Text Normalization and Inverse Text Normalization for Khmer 

**Authors**: Seanghay Yath  

**Link**: [PDF](https://arxiv.org/pdf/2609.30984)  

**Abstract**: Text-to-speech needs written text in spoken form, and speech recognition output needs the reverse. For Khmer, neither direction has a maintained open-source tool, and the script makes both harder: words are not separated by spaces, and number words occur inside ordinary words. We present Tha, a Khmer text normalization and inverse text normalization toolkit built from weighted finite-state transducers. It segments and classifies a whole line in one shortest-path search, and a second transducer rejects token boundaries inside a Khmer syllable. On Google's Khmer test suite, Tha agrees with the reference on all 274 cardinals up to one spelling variant, and on 2,906 real TTS prompts, 153 of the 158 sentences it rewrites are correct. Tha is open source under the Apache 2.0 license. 

---
# Coupled Usage-Sense Processes: Temporal and Attributable Lexical Semantic Change 

**Authors**: Haruka Ezoe, Ryohei Hisano  

**Link**: [PDF](https://arxiv.org/pdf/2609.30974)  

**Abstract**: Lexical semantic change is usually summarized by a scalar distance between independently sampled period distributions. This measures how much a word changed, but does not reveal when it changed, which mechanisms and component movements carried the change, or which usages support the attribution. We introduce Coupled Usage--Sense Processes (CUSP), which derives these answers from a single marginal preserving temporal process. A hierarchical coupling relates contextual distributions through latent usage components, while Markov composition makes adjacent and longer span correspondences compatible. Displacement operators quantify change magnitude and timing, split variation exactly between movement of component centers and reorganization within components, and attribute it to transported component pairs. Word-local modes resolve distinct directions of change and their activity over time, while representative passages from attributed components ground the analysis in text. Under a Gaussian mixture specialization, we prove parametric recovery of the operators and squared distances. Synthetic experiments support the predicted rate. CUSP remains competitive on English and German DWUG and recovers controlled Janus profiles while maintaining compositionally coherent transport. A large corpus of US court opinions demonstrates transition, mode, and passage attribution in unlabeled natural text. CUSP thus makes magnitude, timing, mechanism, movement, modes, and textual evidence compatible views of one lexical history. 

---
# FAVoR: Measuring and Mitigating Author-Style Homogenization in Federated Personalized Generation 

**Authors**: Lu Han, Jingyao Zhang, Katy Ilonka Gero, Nguyen H. Tran  

**Link**: [PDF](https://arxiv.org/pdf/2609.30968)  

**Abstract**: Large language models are increasingly used as personalized writing assistants, but adapting a model across many authors can compromise individual writing style by pulling author-specific signals toward a shared register. Federated parameter-efficient fine-tuning (PEFT) offers a data-local setting for this multi-author adaptation problem: clients keep author text local while sharing compact adapter updates. However, we show that standard aggregation can preserve continuation utility while making different authors' generations less distinguishable in style space, a failure mode we define as author-style homogenization. We evaluate author-style retention with Angular Style Classification Encoder (ASCE)-based diagnostics on our main BlogText benchmark and ASCE-independent external authorship verification. Using this protocol, we find that common federated PEFT baselines can preserve semantic utility while averaging out author-specific signals. To address this homogenization, we instantiate FAVoR (Federated Authorial Voice Retention), an author-style residual mechanism for federated PEFT. FAVoR uses a shared-private adapter design: clients upload shared-adapter updates while retaining author-specific residual corrections locally. Across BlogText and external Mythos-Reddit validation, FAVoR improves author-style retention over standard and personalized federated PEFT baselines. These gains come with small continuation-utility trade-offs and are supported by component ablations, external verification, and cold-start transfer. 

---
# Estimating and Orthogonalizing Unknown Pre-training Gradients for Continual Fine-tuning of Large Language Models 

**Authors**: Bing Wang, Changchun Li, Xin-Qiang Cai, Lin Yuanbo Wu, Ximing Li, Gang Niu, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2609.30935)  

**Abstract**: Continual fine-tuning is essential for large language models (LLMs) to dynamically adapt to real-world environments, yet it inevitably suffers from catastrophic forgetting, particularly the performance degradation of previous tasks and LLMs' general-purpose knowledge. Although existing methods, such as orthogonal gradient projection, mitigate the forgetting across various fine-tuning tasks, they fundamentally fail to preserve pre-training LLMs' inherent general-purpose knowledge because the original data and gradients of off-the-shelf pre-training LLMs required by these methods are strictly unknown and highly diverse. To bridge this critical gap, we propose EoupCT, a novel framework designed to Estimate and Orthogonalize Unknown Pre-training gradients for Continual LLM fine-Tuning. Specifically, EoupCT estimates pre-training gradients by dynamically generating pseudo data that is most susceptible to forgetting for new tasks through a learnable soft prompt equipped with Gumbel-Softmax relaxation. Furthermore, we formulate a multi-objective optimization problem and introduce a first-order efficient Pareto optimizer that jointly optimizes LLM parameters and the soft prompt, rigorously enforcing orthogonality between new task updates and the estimated pre-training gradients. Extensive experiments across multiple LLMs demonstrate that EoupCT effectively preserves both task-specific proficiency and inherent general-purpose knowledge, successfully mitigating the catastrophic forgetting. 

---
# Training-Free Pronunciation Transcription via Text-Constrained Acoustic Rescoring 

**Authors**: Hikaru Asano, Yotaro Kubo, So Kuroki  

**Link**: [PDF](https://arxiv.org/pdf/2609.30924)  

**Abstract**: Accurate and efficient pronunciation transcription is essential for preparing text-to-speech training data at scale. Existing approaches have different limitations: grapheme-to-pronunciation (G2P) and speech-to-pronunciation (S2P) methods each capture only partial information, using only text or only speech, while speech-and-text-to-pronunciation (ST2P) methods use both but require costly pronunciation-annotated data. To address this problem, we propose a training-free ST2P pipeline that integrates both lexical and acoustic information at inference time. Lexical resources and G2P tools generate text-constrained candidates, and a left-to-right greedy search selects the best one using whole-sequence negative log-likelihoods from frozen pretrained S2P models. On three Japanese corpora, our method reduces Character Error Rate (CER) from 0.60--1.40\% (text-only baseline) to 0.04--0.17\% with reference transcripts, and 0.64--1.58\% with ASR transcripts. It outperforms all baselines, including a trained ST2P model and commercial multimodal LLMs. Our greedy search method is 3--3.5$\times$ faster than beam search at similar CER, and the cascade is 2$\times$ faster than direct decoding ensuring the efficiency and accuracy. In Spanish, French, and preliminary English, it also surpasses four open multimodal LLMs and the best traditional methods. 

---
# ToolSearcher: Optimizing Tool Selection at Scale via Reinforcement Learning 

**Authors**: Zhenlong Dai, Xujie Song, Zitong Wang, Tong Niu, Jian liu, Weiqiang Wang, Xiu Tang, Sai Wu, Chang Yao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.30906)  

**Abstract**: Large language models (LLMs) excel at natural language processing but struggle to interact with external environments. Tool learning provides a promising way to extend LLMs into actionable agents, where tool selection is a critical prerequisite for successful tool use. Existing work often assumes a small or predefined set of tools, leaving large-scale tool selection underexplored. Real-world repositories contain a vast and diverse array of tools, making it difficult for LLMs to effectively search, distinguish, and compose tools under context-length constraints. We identify large-scale tool selection as a new challenge for agentic reinforcement learning, highlighting that existing RL methods for knowledge-based question answering are inadequate for selecting tools while considering compatibility. To address this challenge, we propose ToolSearcher, a novel RL framework for effective multi-turn search and fine-grained optimization in large-scale tool selection. Specifically, we introduce category-constrained tool discrimination to improve the model's ability to distinguish functionally similar tools, event-level search modeling to explicitly optimize the discovery of target tools during multi-turn search, and trajectory-aligned credit allocation to provide fine-grained reward signals for different stages of the search-selection process. Extensive experiments on large-scale tool selection benchmarks demonstrate that ToolSearcher consistently outperforms a set of strong baselines in challenging settings involving iterative search and complex tool composition. 

---
# From annotation to reasoning: Culture in language models 

**Authors**: Daniel Hershcovich, Alexander Conroy, Jens Bjerring-Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2609.30897)  

**Abstract**: How should we evaluate language models when more than one interpretation can be right? Cultural benchmarks often test factual knowledge, agreement with survey responses, or recognition of a predefined meaning. These tasks leave open whether a model can explain how a cultural reference works in a particular text, support a reading with evidence, or revise it after criticism. This is a question of interpretive depth, complementary to the breadth of cultural coverage. We argue that literary interpretation offers a useful setting for studying these capabilities. We focus on cultural referencing and reuse: how texts invoke, repeat, and transform earlier expressions across historical and linguistic contexts. Our central claim is that literary scholars can disagree about an interpretation while recognizing the quality of its support. We propose linking evidence-centered benchmarks, evaluation that preserves scholarly disagreement, and model-development experiments on literary data, contextual resources, and scholarly feedback. Danish literature provides a concrete starting point, with implications for other languages and domains. The aim is to develop alternative evaluation strategies that go beyond conventional benchmark metrics and guide model development toward cultural robustness in AI systems. 

---
# Effects of Transcript Compression on LLM-based Medical Misinformation Detection in Japanese YouTube Videos 

**Authors**: Yuya Wake, Sho Tsugawa, Toshiyuki Amagasa  

**Link**: [PDF](https://arxiv.org/pdf/2609.30882)  

**Abstract**: Large language models (LLMs) are increasingly used to assess long-form medical videos, but their effectiveness may depend on whether transcripts are provided in full or compressed through summarization, retrieval, or claim screening. This study examines how such transcript compression affects LLM-based veracity classification of Japanese medical YouTube videos. We compare four transcript input designs: full transcripts, LLM-generated summaries, RAPTOR-based retrievalaugmented generation (RAG), and Screening, which extracts candidate medical and health-related sentences. Using 74 long-form videos labeled as Real or Fake, we evaluate classification performance and analyze linguistic changes using J-LIWC, hedge expressions, and institutional or technical terms. The full-transcript Baseline achieved the best performance, whereas all compressed inputs increased false negatives, meaning that Fake videos were more likely to be misclassified as Real. Summary caused the largest performance drop, while Screening performed best among the compressed inputs but still omitted many medically relevant sentences. Linguistic analyses showed that these errors were not explained by a simple increase in certainty. Instead, Summary reduced affective, social, temporal, cognitive, and conversational cues, while Summary and RAG made institutional and technical terms more salient. These findings suggest that transcript compression can represent Fake videos as more coherent and authoritative inputs, thereby weakening cues needed for misinformation detection 

---
# Evidence-Grounded Auditing of Identification Assumptions in Climate-Policy Causal Evaluations 

**Authors**: Yonghong Zhang, Yong Xie, Isabel M. Parra, Ricardo Correia  

**Link**: [PDF](https://arxiv.org/pdf/2609.30867)  

**Abstract**: Difference-in-differences (DID) studies are widely used to evaluate climate policy, but assessing the evidence supporting their identification assumptions remains challenging. We introduce ARGUS, a structured language-model pipeline that audits reported evidence against an eleven-dimension assumption-implication-evidence rubric and abstains when relevant evidence cannot be retrieved. We evaluate ARGUS using injected flaws, economics papers, and a small pilot with reconciled labels. On the 11-flaw benchmark, ARGUS detects 73% of planted flaws, compared with 18% for a keyword-based pipeline. Across 26 economics papers, ARGUS abstains on about 40% of paper-dimension assessments for lack of retrievable evidence. In a five-paper pilot with labels reconciled by two annotators, it assigns a higher risk level than the labels on 25 of the 33 assessments it completes. A rule fixed before the labels arrived removes most of this in-sample; weighted agreement stays low. ARGUS provides evidence-linked risk reports that localize potential weaknesses for expert review, without adjudicating causal claims. Code and data: this https URL 

---
# Persistent Negatives for Adversarial Black-Box On-Policy Distillation 

**Authors**: Haixu Ma, Saad Lahrichi, Weiwei Li, Kevin Han, Weiqiang Wu, Peggy Yang, Dongzhuo Li, Ruiyi Li, Serena Li, Gedi Zhou, Mingze Gao, Abhishek Kumar, Xiangjun Fan, Lizhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.30864)  

**Abstract**: Black-box On-Policy Distillation (OPD) seeks to improve a student from its own generations when the teacher provides sampled responses but not token probabilities. Adversarial distillation offers one route: it learns a discriminator over prompt-matched teacher and student responses and uses its score as the policy reward. However, sampling discriminator negatives from the latest student at each step couples the learned reward to a negative distribution that changes after every policy update. We address this moving-target problem with persistent-negative adversarial distillation, a live-pool method that replaces a fraction of each discriminator batch with historical, prompt-matched teacher--student comparisons. Under matched discriminator compute, historical comparisons train the discriminator, while GRPO remains on-policy with fresh student responses. Our analysis identifies the Bayes-optimal reward as a teacher-to-negative log-density ratio and, under explicit assumptions, shows how persistent negatives anchor the discriminator and reduce reward-estimation MSE relative to fresh-negative training. Across two student families, three judges, and four judged-chat benchmarks, persistent-negative adversarial distillation consistently improves performance over current methods at matched discriminator compute. It also yields smoother fresh-policy discriminator trajectories, with fewer below-chance dips. These findings identify the discriminator's negative distribution as an important design axis in black-box on-policy distillation. 

---
# Enhancing Assessment of Self-Consistency in LLM Explanations using Perturbation Strength 

**Authors**: Phuong Q. Le, Kemal Kurniawan, Jey Han Lau  

**Link**: [PDF](https://arxiv.org/pdf/2609.30849)  

**Abstract**: Prior work has examined the self-consistency of LLM-generated explanations using surface-level perturbation methods. However, the strength of these perturbations is not explicitly measured and controlled. In this work, we propose an LLM-as-a-judge approach to measure perturbation strength in a unified manner across input and CoT perturbations. We then evaluate the self-consistency in explanations generated from various LLMs under controlled strength conditions, ensuring a fair comparison across perturbation types. Experiments show that our proposed LLM-based perturbation strength measure outperforms other embedding- and probability-based approaches and that input perturbations generally affect LLMs more strongly than CoT perturbations. Our work suggests that judgments about a model's self-consistency is fair only within the same perturbation type. 

---
# I-Parakeet: Integer-Only Conformer ASR on Mobile NPU 

**Authors**: Taichi Nishimura  

**Link**: [PDF](https://arxiv.org/pdf/2609.30846)  

**Abstract**: In this paper, we propose I-Parakeet, an integer-only implementation of NVIDIA's Parakeet-CTC (0.6B parameters) that runs on a smartphone NPU without any floating-point operator or CPU fallback. Modern Conformer ASR models are hard to deploy on edge devices because of their size, and quantized models still fall back to floating point for numerically sensitive operations. This prevents them from fully exploiting integer accelerators such as mobile NPUs. To achieve this, our contributions are threefold. First, we derive an integer formulation of the relative-positional self-attention at the core of the Conformer. We fuse its two score branches with different quantization scales and the relative shift into integer-only operations. Second, we introduce a minimax-optimized Swish approximation that minimizes the maximum error of the Swish output. Third, a layer-wise range analysis of activations yields two targeted remedies: an INT16 grid for the BatchNorm output and percentile calibration for the heavy-tailed pre-encoder activations. I-Parakeet achieves 4.97% WER on LibriSpeech test-other, running on a Qualcomm NPU at a real-time factor of 0.048, 7.5x faster than a CPU baseline. 

---
# Understanding the Role of Prompt Template in Knowledge Distillation for Safety Alignment 

**Authors**: Anjila Budathoki, Manish Dhakal, Benjamin M. Ampel, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2609.30802)  

**Abstract**: Prior research has demonstrated that the choice of prompt template during Supervised Fine-Tuning (SFT) significantly impacts the robustness of safety alignment afterwards. However, the influence of template selection during Knowledge Distillation (KD) from teacher to student remains largely unexplored. Thus, we fill this gap by analyzing how different template configurations influence the pre-existing safety alignment of the student. We observe a significant degradation of safety alignment present in the aligned base instruct-tuned model. Specifically, we find that utilizing chat templates renders the model more compliant with harmful queries compared to a non-chat template. These findings are consistent across three models: LLaMA, Gemma and Qwen model families and are evaluated across multiple safety benchmarks. We further show that using a non-chat template during distillation better preserves the base student's internal representations, while chat template distillation induces a larger representational shift. Code: this https URL 

---
# Learning Natural Conversational Behavior in Tandem Speech-to-Speech Models with Randomized Guidance 

**Authors**: Manato Yaguchi, Yotaro Kubo, Hikaru Asano, So Kuroki  

**Link**: [PDF](https://arxiv.org/pdf/2609.30773)  

**Abstract**: Tandem speech-to-speech architectures couple a responsive speech frontend with an asynchronous text backend. In KAME, a large language model (LLM) serves as the backend, supplying candidate responses as guidance to the speech frontend while the user is still speaking. Ordinary conversation recordings capture the eventual response but not the guidance the backend would supply during the user's utterance. Generating the missing guidance with a simulator LLM adds substantial data-preparation overhead when training on real conversations. We propose randomized intermediate guidance, which derives guidance directly from the conversation corpus rather than simulating backend LLM behavior. During training, target responses provide informative guidance, while randomly sampled responses provide potentially irrelevant updates during the utterance. This combination aims to teach the frontend to use backend information selectively. On synthetic dialogues, KAME trained with this recipe achieves response quality comparable to that of the LLM-generated and similarity-based baselines. Training on 3.8k hours of real conversations improves smooth turn-taking and audio-judge naturalness over synthetic-data KAME while retaining a response-quality advantage over Moshi. These results show that randomized guidance offers a practical route to combining the response-quality benefits of tandem models with natural conversational behavior learned from real speech. 

---
# SEA-CLIP-Tiny: Efficient Multilingual Text-Vision Embedding for Southeast Asian Languages 

**Authors**: Puja Ahmad Habibi, Faiz Assabil Firdaus, Ashvanth S, Ekapol Chuangsuwanich, Pume Tuchinda, Peerat Limkonchotiwat  

**Link**: [PDF](https://arxiv.org/pdf/2609.30739)  

**Abstract**: Multilingual text-vision embedding models are essential for cross-lingual image-text retrieval, but Southeast Asian languages remain poorly supported due to the region's linguistic diversity and limited data and computing resources. In this paper, we introduce SEA-CLIP-Tiny, a compact multilingual text-vision embedding model for Southeast Asia with fewer than 50M parameters. Our model adapts a CLIP-KD-style framework to Southeast Asian multilingual settings through regional data curation and multilingual teacher guidance. Experiments across seven Southeast Asian languages show that SEA-CLIP-Tiny achieves the strongest average retrieval performance among the evaluated student models, reaching 12.9%, 31.5%, and 42.2% at R@1, R@5, and R@10, respectively. Compared with MobileCLIP2, it improves average R@10 by 12.1 points while using 38.4% fewer parameters and lower measured CPU latency. These results highlight the importance of region-aware training for efficient multilingual text-vision models in Southeast Asia. 

---
# Beyond Mean Attention: Diversity-Aware, Layer-Wise Scoring for KV Cache Eviction 

**Authors**: Tianfang Xie, Wei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.30738)  

**Abstract**: KV cache eviction methods such as SnapKV and PyramidKV rank tokens solely by mean attention over a small observation window. We study a unified score, $\mu_i+\lambda_1\sigma_i+\lambda_2\mathrm{corr}(i,S)$, adding attention dispersion across window queries and redundancy relative to selected tokens. For $\lambda_2<0$, the score penalizes similarity to selected tokens as in maximal marginal relevance (MMR), without extra forward passes. To test whether this relevance-diversity balance should vary with depth, we compare fixed global coefficients with three-segment and quadratic profiles. Only these depth profiles are searched on a development split under a $\sinh$ reparameterization. On all 16 English LongBench datasets with Mistral-7B at a budget of 64 entries per layer, a single global diversification constant improves 13 of 16 datasets (macro +1.1); the gain holds at budget 32 and narrows at 128. Per-dataset search finds no detectable layer structure on most datasets; on passage retrieval it finds a large one: a mid-layer sign flip that rewards similarity and is worth +9.6 over the baseline at budget 64 and, without re-tuning, +13.2 over the global constant at budget 128. Ablations attribute the gain to the redundancy term; replaying every accepted search state on the held-out test set separates genuine structure from tuning noise. 

---
# Words Speak Louder Than Order: A Behavioral Evaluation of Gemma 4 

**Authors**: Amanda Fitch  

**Link**: [PDF](https://arxiv.org/pdf/2609.30716)  

**Abstract**: When a language model receives two conflicting documents as input, how does it decide which one to prioritize? Does it rely on how the sources are framed or the presentation order of the documents? We evaluated this behavior on Google's pre-trained Gemma 4-e4b model across a targeted behavioral suite (n = 13 items, 784 forward passes in short, single-turn contexts) using a completely counterbalanced experimental design. This setup allowed us to mathematically isolate the specific effects of source framing and reading position, while ensuring the model's natural vocabulary biases were canceled out.
Across ten test conditions, we discovered the following:
1. Source framing heavily overpowers reading position. When directly competing, the semantic framing of a source (such as presenting it as an official guideline or a fresh update) had a significantly stronger impact on the model's final answer than the presentation order of the document.
2. The model favors the first document it reads, but this bias is highly variable. While the model consistently demonstrated a primacy effect (preferring the first document presented), the actual strength of this bias fluctuated by at least a factor of 5 based solely on the surface wording.
3. Overall structural repetition, not short copy-cues, drives positional bias. The model's preference for the first document is not a mechanical reaction to short, repetitive trigger phrases, such as "is [Answer]". However, the primacy effect does increase significantly when the two competing documents are structurally identical, using word-for-word verbatim templates. Introducing variation in the overall wording between the two sources reduces this positional bias. 

---
# TRACE: Temporal Audit and Condition-aware Evaluation of Streaming Video Understanding 

**Authors**: Yibo Ma, Qianqian Zhang, Peng Liu, Tiancheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.30670)  

**Abstract**: Streaming video understanding requires models to interpret evidence as it arrives, yet current evaluations often report task scores without specifying when evidence becomes valid, how visual history is maintained, or how responses are triggered. As a result, similar scores may correspond to different workloads, failure modes, and operational behavior. We introduce TRACE (Temporal Audit and Condition-aware Evaluation), a condition-aware benchmark and evaluation framework that makes these factors explicit. TRACE combines temporally audited visual tasks with evidence timing and instruction-dependent trigger annotations, a unified causal Core--Adapter protocol that controls information availability while recording actual history processing and response events, and multidimensional reporting of answer quality, timeliness, response-selection behavior, workload, completion, and reliability. On 1,240 records from 517 videos, we evaluate eight publicly available models or systems in eight configurations. We find that nearly identical QA accuracy can mask substantial differences in completion, answer validity, and generation workload, while proactive performance separates into response quality, response delay, false alarms (responses emitted while no target window is currently valid and a later one remains), and missed target windows. These results show that streaming-video performance should be interpreted as execution-conditioned system behavior rather than a single score. Our benchmark and code can be accessed at \href{this https URL}{this https URL}. 

---
# Recursive Self-Improvement via On-Policy Distillation for Reasoning 

**Authors**: Shangjian Yin, Zehao Zhao, Kavosh Asadi, Rui Liu, Yuchen Lu, Shike Mei, Hang Cui, Luke Simon, Zhouxing Shi, Hamed Firooz  

**Link**: [PDF](https://arxiv.org/pdf/2609.30652)  

**Abstract**: On-policy distillation (OPD) trains a student model by having it generate trajectories, then matching its next-token predictions with an external teacher's next-token predictions. This provides dense, token-level supervision to the student. On-policy self-distillation (OPSD) eliminates the need for the external teacher. Specifically, a second frozen copy of the student model, now given the ground truth in its context, serves as the teacher. The student model only receives the problem and learns to mimic the privileged teacher model, while the teacher remains frozen throughout training. Previous work showed that freezing the teacher is useful for training stability, but we argue that this can prevent the teacher from incorporating the improvements learned by the student during training. Our primary contribution is to address this limitation with a recursive framework built around two complementary components. First, we let the privileged teacher co-evolve with the student so that revision learned in one round can guide the next, a process we refer to as Dynamic Co-Evolution (DCE). Second, because stronger revision can also make responses too verbose and self-critical, we additionally train on shorter, verified rewrites of the model's own on-policy responses. We call this complementary objective Self-Refined Concise Learning (SRCL). Overall, our comprehensive evaluations show that DCE+SRCL outperforms OPSD across multiple model scales and four competition-level mathematics benchmarks. Specifically, on Qwen3-8B, DCE+SRCL reaches 65.97% Average@12, outperforming OPSD by 35.62 percentage points while reducing mean output length by 7.80% relative to DCE alone. 

---
# The Hard Part Comes After Search: Benchmarking Web Agents on Synthesizing, Organizing, and Displaying Knowledge 

**Authors**: Alexander Gill, Md Farhan Ishmam, Xuyen Nguyen, Neha Bhat, Parker Henry DeYoung, Fateme Hashemi Chaleshtori, Nathan Stringham, Kenneth Marino, Ana Marasović  

**Link**: [PDF](https://arxiv.org/pdf/2609.30604)  

**Abstract**: Existing computer-use agent benchmarks do not fully evaluate agents acting as assistants. A useful assistant retrieves information across complex, multi-step workflows, synthesizes it into artifacts (documents, presentations, spreadsheets), and navigates program interfaces to produce a coherent final product. Such workflows demand reasoning and synthesis, decomposition of complex tasks, as well as visual and spatial understanding. To study agents on workflows like these, we introduce KNOWS, a benchmark of open-ended, complex, browser-based tasks that jointly evaluate these capabilities, with each task culminating in a produced artifact. To write tasks, we develop a task design rubric and a protocol for ensuring that tasks meet the requirements. Each task is paired with an evaluator, a program that combines deterministic checks with LLM judgments to balance the richness, reliability, and automation tradeoff inherent to agent evaluation. We evaluate and analyze frontier computer-use agents and browser-based harnesses. They achieve moderate scores on partial-success metrics, but the best performer fully succeeds in fewer than 3% of our complex, long-horizon tasks. Failures on visual steps render the resulting artifacts unusable, even when agents complete more than 50% of other evaluation steps. Our results expose limitations of current agents acting as end-to-end assistants, and call for progress on tool use, visual understanding, and long-horizon reasoning. 

---
# Probing Stability-Plasticity Tradeoffs in Agent Memory through Cognitive Experimental Paradigms 

**Authors**: Jiaqi Ding, Guorong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.30558)  

**Abstract**: Agent memory systems are increasingly used to maintain long-term user preferences, task states and evolving facts, but current evaluations often collapse memory behavior into final-answer accuracy. We introduce MemProbe, a cognitive-science-inspired framework for diagnosing stability-plasticity tradeoffs in agent memory. The framework is motivated by a core insight from cognitive memory research: memory is reconstructive and shaped by interference, source reliability, reinforcement, and reactivation. MemProbe turns this insight into four reusable experimental paradigms (interference, misinformation, consolidation strength, and reconsolidation window) that manipulate when a memory should be updated, preserved, or treated as uncertain. It further decomposes correctness into behavioral profiles that reveal how systems update, preserve, attribute, and temporally organize information. We instantiate these paradigms in a 56-episode diagnostic suite and evaluate six incremental memory systems under a unified protocol. Results show that systems with similar aggregate scores exhibit distinct behavioral profiles. MemProbe provides such a diagnostic lens, turning aggregate performance into interpretable profiles of memory maintenance over time. Code is available at this https URL. 

---
# REALMS: An AI-Assistant Conversational System for Real-Time Exact Audience Sizing over High-Dimensional Nested Profiles 

**Authors**: Haixu Ma, Aditya Bansal, Shubham Lohiya, Sumit Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2609.30547)  

**Abstract**: Audience sizing is a critical component of digital marketing. It enables precise resource allocation, campaign planning, and performance optimization. Traditional approaches using skeleton audiences, sampling, or predictive modeling suffer from significant delays, estimation errors, and poor scalability over high-dimensional profile data. We present REALMS (Real-time Exact Audience sizing via LLM-based Multi-attribute Search), a conversational system for exact audience sizing deployed in production on an enterprise customer data platform. REALMS enables marketers to query massive profile stores with millions of profiles and thousands of attributes using natural language and receive precise counts in seconds. The system introduces three key components: (1) a categorical attribute retrieval mechanism using embedding-based vector search to dynamically identify relevant schema attributes without manual configuration; (2) an LLM-powered NL2SQL pipeline with template-based in-context learning for accurate query generation over complex nested schemas; and (3) schema standardization enabling industry-agnostic deployment across diverse enterprise environments. Evaluation on real enterprise data demonstrates strong recall for attribute retrieval, high SQL execution accuracy, and low latency, which enables real-time interactive audience insights where prior methods required hours. 

---
# Feeding BabyLMs Macaroni: Code-Switching Curricula Cause Cross-Lingual Convergence 

**Authors**: Dries Rooryck, Alex Cai, Yonatan Belinkov, David Alvarez-Melis, Kianté Brantley  

**Link**: [PDF](https://arxiv.org/pdf/2609.30535)  

**Abstract**: Children in multilingual communities often code-switch, using multiple languages in a single utterance. Can we induce cross-lingual alignment in language models by training on code-switched text? We pretrain small decoder-only transformers on two 100M-word multilingual corpora: a base corpus formed by mixing the English, Dutch, and Chinese BabyBabelLM datasets, and a corpus generated from it by inserting word- and sentence-level code-switching using an LLM. We find that training on code-switched data aligns the representations of parallel text, particularly across different scripts, and that this alignment persists through training on monolingual documents. Under a learning curriculum that progresses from word-level code-switching, to sentence-level code-switching, to monolingual documents, models trained on code-switched data outperform baselines trained without it on the BabyLM evaluation suite. Our work characterizes code-switching curriculum learning as an effective data augmentation method for multilingual pretraining. We release our code, data, and models at this https URL. 

---
# Inquesto Score: A reliability Protocol For Voice Agents 

**Authors**: Massa Baali, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2609.30514)  

**Abstract**: Voice agents are increasingly deployed in workflows where failed interactions can affect transactions, access, and other consequential outcomes, creating a need for reproducible and interpretable evaluation. We introduce Inquesto Score (IS), a protocol for measuring voice-agent reliability as the percentage of calls in a fixed, versioned evaluation population that achieve the caller's goal without a functional failure or worse. Rather than combining heterogeneous metrics, IS defines explicit failure events and severity levels and evaluates the deployed voice pipeline. Timing failures, including talk-over and delayed responses, are measured directly from audio, while semantic and state-dependent failures are evaluated using scenario predicates, tool traces, and a pinned open-model judge. Diagnostic views of behavior, acoustic robustness, identity handling, and speaker groups accompany the score without being combined into it. Inquesto Score v0.1 evaluates 30 scenarios, three acoustic conditions, four speaker groups, and 306 calls per agent across 13 configurations of a reference voice-agent system. Our evaluation shows that reliable measurement requires evidence beyond transcripts, explicit treatment of deployment conditions, and validation of the evaluators used to determine outcomes. We release the protocol, reference implementation, and evaluation records. 

---
# Breaking Homogeneity: Diversifying Persona Sets for Creative LLM Outputs 

**Authors**: Sang Bin Moon, Nicole Cho, Daniel Borrajo, Sumitra Ganesh, Abolfazl Hashemi  

**Link**: [PDF](https://arxiv.org/pdf/2609.30492)  

**Abstract**: Language models often produce homogeneous responses to open-ended tasks; such homogeneity can spawn groupthink-the convergence of ideas toward a singular and potentially suboptimal decision. We formulate persona diversification as a set-level conditioning problem and study two orthogonal design choices: selecting versus generating personas, and space-filling versus frontier-seeking diversity. We instantiate this design space with four methods spanning coverage and dispersion subset selections, uniform-coverage sampling, and evolutionary persona generation. Evaluations on the Alternative Uses Task (AUT), Infinity-Chat, and Divergent Association Task (DAT) show the benefits of the proposed methods across tasks and creativity objectives. On AUT, evolutionary persona generation increases response diversity by 78.8%, originality by 26.1%, flexibility by 49.5%, and holistic creativity by 13.9% over task-only prompting, while maintaining 98.5% validity; on Infinity-Chat, it nearly doubles persona-induced response separation relative to random personas. Moreover, evolutionary personas compose with creativity-optimized prompting, further increasing its response diversity by 18.6% and creativity by 6.3%. These results establish persona-set geometry as a task-agnostic mechanism for eliciting divergent LLM outputs, and support persona diversification as a reusable complement to prompt optimization. 

---
# CARGO: Context-Aware Retrieval-Gated Evaluation of Agentic AI in Production 

**Authors**: Mukul Chhabra, Shail Patel, Luigi Medrano  

**Link**: [PDF](https://arxiv.org/pdf/2609.30471)  

**Abstract**: Reference-based LLM-as-a-judge evaluation assumes the reference answer is the target. In deployed agentic systems that operate over dynamic entities (support cases, assets, accounts), the closest available reference typically applies the correct procedure to a different entity, so a literal judge penalizes different identifiers, dates, and statuses as errors or hallucinations. We name this failure mode reference-instance divergence (RID). We propose CARGO, a framework that (i) treats retrieved references as procedural exemplars and grounds factual judgments in the live instance's observed context, (ii) assigns each claim a three-way status (supported, contradicted, unverifiable) and penalizes only contradictions, and (iii) gates evaluation by retrieval confidence, casting production evaluation as selective prediction. We introduce CARGO-Bench, a perturbation-based diagnostic suite with ground truth by construction that separates leniency from discrimination. On CARGO-Bench (246 items, two judge models, 7,872 judgments), the standard reference-based judge penalizes 100% of correct entity-transplanted answers and is uninformative (discrimination index DI ~ 0); supplying the live facts without reframing changes nothing. CARGO eliminates these false penalties (0/50) while retaining near-complete contradiction recall (50/50 and 49/50), raising DI to 0.58 [0.48, 0.68]; a rubric-swap control attributes most of the effect to context-grounded dimension definitions. CARGO also exposes a limitation of its own design: the leniency that protects entity values suppresses detection of procedural corruptions (20% recall). A post-hoc fix does not close the gap, and an LLM-as-annotator study with written guidelines and adjudication shows the same blind spot. We release a preregistered protocol for extending the evaluation to expert agreement, risk-coverage, and cost on production traffic. 

---
# Where Does Retrieval-Based Open-Ended Evaluation Fail? Automatic Taxonomy Induction from Long-Form Medical Answer Factuality Verification 

**Authors**: Heyuan Huang, Jirui Dai, Alexandra DeLucia, Sonal Joshi, Mahsa Yarmohammadi, Jie Gao, Bernal Jiménez Gutiérrez, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2609.30467)  

**Abstract**: Retrieval-based factuality evaluation, where LLM-generated claims are verified against evidence from authoritative medical corpora, has become the dominant paradigm for scalable hallucination detection in high-stakes clinical settings. Despite the urgency of reliable and transparent medical fact verification, most systems measure performance with aggregate metrics like F1, which obscure where and why failures occur. Existing RAG diagnostics require gold answers or annotated gold evidence, neither of which exists in this regime. We introduce two comprehensive taxonomies, grounded in a case study on the open-ended MedExpert dataset and 3 closed-ended datasets, decomposing failures into retrieval-stage errors along five quality dimensions, and verifier-reasoning errors into six consecutive steps. We adapt an automatic pattern induction pipeline using LLM-as-Judge to label evidence quality and classify verifier reasoning errors at scale, and then stress-test our findings across 4 retrieval methods and 6 frontier verifier models. Our analysis reveals that scaling model size, adding reasoning effort, expanding to authoritative web sources, and applying medical fine-tuning do not resolve these failure modes, demonstrating that they represent fundamental limitations of the retrieve-then-verify paradigm in open-ended medical settings rather than artifacts of outdated systems. We release our code and data at this https URL for the full reproducibility of our results. 

---
# Inference-Time Target Speaker Unlearning in LLM-Based Automatic Speech Recognition 

**Authors**: Bo Su, Yueru Yan, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2609.30439)  

**Abstract**: We introduce target-speaker unlearning ASR (TSU-ASR) task in a fully end-to-end framework for multi-speaker ASR and diarization. Given a multi-speaker utterance and a set of opt-out speakers who do not wish to have their speech transcribed, the task requires an ASR system to transcribe all speakers except the opt-out ones, while still indicating when those speakers are active. As a first step towards tackling this task, we introduce a novel, light-weight Enrollment-Conditioned Gating (ECG) module attachable to a frozen dual-stream speech LLM that enables ASR for new opt-out speakers dynamically during inference, even those who were not seen during initial ECG training phase. Our experiments on both AMI (English) and AliMeeting (Mandarin) datasets show that speech transcription accuracy for corresponding opt-out words or characters falls from 72.3% to 48.2% and from 73.6% to 27.3%, respectively, while retained speakers' transcription error rates maintain more or less the same. Our approach provides a practical solution for modern video conferencing platforms, allowing speakers to dynamically opt-out from automated AI transcriptions without forcefully leaving the meeting sessions, enabling a privacy-preserving interface for potentially millions of online meetings daily. 

---
# All In Good Time: Causality-Aware Framework for LLM-Based Simultaneous Speech-to-Speech Translation 

**Authors**: Amir Hussein, Enas Albasiri, Travis M. Bartley, Nourchene Ferchichi, Ke Hu, Harishchandra Dubey, Myungjong Kim, Zhehuai Chen, Oluwatobi Olabiyi, Sanjeev Khudanpur  

**Link**: [PDF](https://arxiv.org/pdf/2609.30416)  

**Abstract**: Large Language Models (LLMs) have shown strong performance in low-resource offline translation; however, extending them to simultaneous speech-to-speech translation (Simul-S2ST) remains challenging due to the scarcity of causally aligned training data with high cross-lingual speaker fidelity. In addition, existing approaches rely on fixed translation policy or confidence heuristics, leading to suboptimal quality and higher latency. We propose a causality-aware Simul-S2ST framework with a novel data pipeline that generates high-fidelity, causally aligned segments with improved voice transfer. The framework introduces (i) a factorized S2ST architecture (FAST), (ii) a causality-aware adaptive policy (CAP), and (iii) causality-aware latency metric. Experiments on CVSS Spanish, German, and French show that FAST-CAP consistently improves the quality-latency trade-off, achieving up to +1.2 BLEU and a 26% relative latency reduction over a fixed policy. Despite using substantially less training data than existing systems, FAST-CAP achieves state-of-the-art results in speech translation quality and speaker fidelity while yielding up to a 38.8% relative reduction in latency. 

---
# A Unified Account of Concepts and Chunks 

**Authors**: Karthik Singaravadivelan, Pat Langley  

**Link**: [PDF](https://arxiv.org/pdf/2609.30414)  

**Abstract**: Cognitive psychology has studied how people encode, use, and learn concepts that describe categories, and how they represent, recognize, and acquire chunks for familiar patterns of elements. The literatures on these two topics are nearly disjoint, which poses a challenge for unified theories of cognition. In this paper, we review Cobweb, a computational account of categorization and concept formation, and propose an extended theory that incorporates chunks and their acquisition. The theory makes no commitments about modality, applying to any experience that decomposes into elements and relations among them. We also present \trellis/, an implementation of this theory, and illustrate its application to learning context-free grammars, which we adopt as a testbed because they involve both concept-like and chunk-like elements. In addition, we report experimental results on three synthetic grammars that demonstrate the system's ability to represent syntactic knowledge, use it to parse and generate sentences, and learn compositional structures from sample parses. We conclude by discussing related work on concepts and chunks, along with directions for future research in the area. 

---
# A Benchmark Framework for Screening Automation in Systematic Reviews 

**Authors**: Gauransh Kumar, Luciano Marchezan, Guillaume Genois, Kévin Delcourt, Eugene Syriani  

**Link**: [PDF](https://arxiv.org/pdf/2609.30298)  

**Abstract**: Systematic reviews (SR) are essential for evidence-based research, but their screening phase is highly time-consuming and labor-intensive. Large language models (LLMs) offer a promising opportunity to reduce this workload by assisting with article relevance classification. However, existing evaluation approaches often rely on traditional metrics that may be misleading for highly imbalanced SR screening this http URL paper presents a benchmark dataset of $45\,064$ labeled entries for evaluating LLM performance in SR screening across 32 curated secondary studies. It proposes an evaluation framework that accounts for class imbalance, i.e., the natural prevalence of excluded articles relative to included articles in SRs. It also introduces PromptSR, a tool designed to support prompt experimentation, experiment management, and result analysis for LLM-based screening. We also present a use case demonstrating the application of SRBench and PromptSR. 

---
# Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops 

**Authors**: Enrico Palumbo, Alexandre Tamborrino, Victor Ode, Ben Lacker, Adrià Casas Escoda, Jeremy Hopple, Marcus Better, James Leoni, Hugo Galvão, Hugues Bouchard, Mounia Lalmas, José Luis Redondo García, Abenezer Abebe, Ann Clifton, Anton Blomberg, Henrik Lindström, Dani Doro, Christine Doig Cardet  

**Link**: [PDF](https://arxiv.org/pdf/2609.30297)  

**Abstract**: Conversational recommendation agents are a new paradigm for content discovery, enabling users to express complex intents through natural language (e.g., "recommend Italian indie artists I haven't heard before"). A central challenge in building such agents is optimizing agent planning -- deciding how to select, sequence, and invoke tools -- particularly in cold-start settings where real user interactions are not yet available. We introduce a pipeline for multi-turn synthetic data generation and a self-improvement loop to address this challenge. The synthetic data pipeline transforms single-turn prompts into realistic multi-turn conversations, enabling systematic evaluation before launch. The self-improvement loop combines variance-based contrastive optimization with iterative refinement through a coding agent, automatically identifying and fixing planning and tool-use errors. Our approach improves quality by +8% on top of a highly optimized manual prompt. The system has been productionized and significantly accelerated iteration cycles for the launch of a conversational recommendation agent at Spotify. Online A/B tests demonstrate its effectiveness, with +14% user listening, +5% increase in weekly active users, and a 5% reduction in skip rate compared to a prior experience supporting only session refinement. This work provides a practical framework for accelerating the development of conversational recommendation agents in industry. 

---
# SignTrace: Describe a Sign, Find the Word 

**Authors**: Zengji Tu, Xingye Zhu, Ningjing Wang, Tingyi Huang, Yangjunfeng Zhu, Dai Wan  

**Link**: [PDF](https://arxiv.org/pdf/2609.30295)  

**Abstract**: Identifying an unfamiliar sign is difficult when a learner remembers its movement but does not know its meaning or formal feature codes. SignTrace addresses this longstanding reverse-lookup problem through natural-language access to a Chinese sign-language dictionary. The system integrates LLM-based dictionary enrichment, action extraction, dictionary-style rewriting, seven-channel retrieval, and candidate reranking over 6,699 entries. It has been deployed for user trials and has received positive informal feedback. Evaluation on a dictionary-derived benchmark of 500 movement-description queries yields 94.0% Hit@1, 97.4% Hit@9, and a mean reciprocal rank of 0.9540. Reranking increases Hit@1 from 71.8% to 94.0%, while component analyses show the contribution of enriched entry descriptions. Median query-processing time is 13.37 seconds with six concurrent queries. By connecting everyday movement descriptions to documented signs and meanings, SignTrace provides a practical tool for identifying unfamiliar signs. Dictionary-derived wording and prior selection within the benchmark limit generalization to descriptions independently produced by users. 

---
# SlideLab: Audience-Centered Scientific Slide Generation and Evaluation 

**Authors**: Vidushee Vats, Karun Sharma, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.30294)  

**Abstract**: Scientific presentations are more than summaries of research papers. They need to present the work in a coherent sequence, explain the main ideas clearly, and help the audience follow the presentation. We present SlideLab, a training-free multi-agent framework for generating scientific presentations from research papers. SlideLab first plans the presentation narrative, then builds and iteratively refines a shared slide deck using agents for content planning, visual generation, layout refinement, and grounding verification. In a blind human preference study, SlideLab was preferred over both open-source and commercial systems on 77% of papers while using roughly 4 times fewer inference tokens than the strongest open-source baseline. We also introduce ConfArena, an audience-oriented evaluation framework that simulates a conference room and assesses presentations slide by slide. ConfArena matches human system rankings and detects injected presentation problems, including falsified numbers, degraded figures, dropped slides, and shuffled slide order. 

---
# Cartograph: Federated Tool Discovery with Operator-Attested Retrieval for AI Agents 

**Authors**: Justice Owusu Agyemang, Michael Agyare, Kwame Opuni-Boachie Obour Agyekum, Kwame Agyeman-Prempeh Agyekum, Francisca Adoma Acheampong, Jerry John Kponyo  

**Link**: [PDF](https://arxiv.org/pdf/2609.30293)  

**Abstract**: The Model Context Protocol (MCP) enables AI agents to discover and call tools, but loading every definition becomes expensive as connected catalogs grow. We present Cartograph, a federated MCP proxy that changes agent-visible tool discovery from $O(n)$ catalog traversal to $O(k)$ progressive disclosure. Cartograph combines three mechanisms: (1) operator-attested capability cards, Ed25519-signed descriptions generated under the deploying operator's control rather than ranked publisher copy; (2) Rift, a three-layer confusable-cluster analysis comprising density clustering, query-margin analysis, and token diagnosis; and (3) two-stage retrieval, which ranks servers before tools. On a 22-server, 374-tool deployment, Cartograph exposes three proxy tools instead of 374 definitions. A 49-query author-constructed benchmark yields R@5 of 0.816, compared with 0.592 for a Jaccard keyword baseline, while a measured top-5 discovery exchange uses 475 tokens rather than 42,450 under the stated full-catalog accounting. Rift identifies 49 confusable clusters, including four HIGH-risk clusters in bootstrap-generated cards. An exploratory comparison of 119 LLM-generated descriptions removes the observed zero-distance cluster but shows that mixing card-generation regimes can reduce R@5. Gateway measurements over ten trials add 5ms mean latency (0.8%) relative to direct stdio MCP calls. Cartograph is complementary to code-execution approaches: it controls which tool descriptions are surfaced and records the provenance of the descriptions used for ranking for each query. 

---
# A Survey on Fake Review Detection: From Pre-trained Language Models to Large Language Models 

**Authors**: Fanji Yang, Huiyao Chen, Xi Yu, Meishan Zhang, Xiaohong Xiao, Mingsen Deng  

**Link**: [PDF](https://arxiv.org/pdf/2609.30292)  

**Abstract**: Online reviews shape consumer decisions, platform governance, and corporate this http URL reviews compromise this information channel by injecting deceptive evidence into rating systems, recommendation pipelines, and public trust this http URL rise of large language models, or LLMs, has changed the problem in two this http URL can generate fluent and context-aware deceptive reviews, while pre-trained language models, or PLMs, and LLMs also provide stronger semantic representations for this http URL survey reviews fake review detection from an information fusion perspective, covering 211 studies published from 2018 to early this http URL organize existing work by evidence source and fusion level, covering review text, sentiment, rating behavior, temporal metadata, user-product graphs, multimodal content, external knowledge, and LLM-generated this http URL trace the development from traditional machine learning and deep learning to PLM-based and LLM-based methods, and examine how different approaches combine textual, behavioral, structural, and multimodal this http URL also analyze reported performance trends on widely used Amazon, Yelp, and OpSpam benchmark families, while noting the limitations caused by different label construction procedures, data splits, and evaluation this http URL, we identify open problems in adversarial generation, cross-domain transfer, uncertainty-aware fusion, missing-source robustness, interpretability, and trustworthy evaluation for AI-generated deceptive content. 

---
# Auditing and Repairing LLM-as-Judge Failures in a Production Text-to-SQL Pipeline 

**Authors**: Haowei Liu, Hsin-Tai Wu, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2609.30290)  

**Abstract**: Production text-to-SQL pipelines often end with an LLM-as-judge whose agreement with human annotators has never actually been measured. When we checked ours, the deployed gpt-4o-mini judge agreed with two-author gold at only Cohen's kappa = 0.04 on a disagreement-enriched set and 0.42 on a uniform-random spot-check, over-flagging 77.1% of the human-FAITHFUL cases in the enriched set. Most of its over-flags trace back to a single mechanism we call GRADE-HALLUCINATION. A self-hosted Qwen3.6-27B replacement (kappa = 0.72) lands in the same range as Claude Opus 4.7 (kappa = 0.71); the head-to-head is underpowered at n = 96, but for the deployment decision that hardly matters, since Qwen costs roughly 1/300 as much per call. Ensembling does not help for free. Pairing the weak judge with a stronger one degrades agreement, whereas three strong judges under unanimity routing reach kappa = 0.79 at 89.7% auto-coverage. Applied out-of-domain, the same audit recipe flags 25.5% of BIRD-financial's expert-authored gold SQLs as candidate gold-SQL issues under our annotation protocol. Code and pre-registration are at this https URL. 

---
# Not All Memories Are Equal: Hierarchical Collaborative Memory for Validity-Aware Retrieval in LLM Agents 

**Authors**: Yufei Shi, Rujing Yao, Ang Li, Yang Wu, Zhuoren Jiang, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.30289)  

**Abstract**: In team collaboration scenarios, memory is heterogeneous and continually evolving. Team memories capture collective decisions, protocols, and current consensus, while individual memories preserve member-specific observations, execution traces, and intermediate progress. Existing memory-augmented systems typically retrieve from all stored memories as a flat pool, ranking them by semantic relevance, importance, or recency without modeling hierarchical structure or evolving validity. As a result, they often surface semantically relevant but outdated or conflicting memories, especially individual memories that no longer align with current team consensus, instead of prioritizing currently valid memories. This is particularly problematic when collaborative LLM agents answer user questions, since their responses should be grounded in valid memories. We propose HiCoMER, a framework for hierarchical collaborative memory management and validity-aware retrieval in LLM agents. HiCoMER first maintains the validity of team and individual memories and then retrieves memories that remain valid, rather than retrieving directly from all stored memories. It consists of three components: a Hierarchical Memory Conflict Updater, a Validity-Aware Memory Retriever, and a Memory-Grounded Answer Generator. To evaluate HiCoMER, we construct two new datasets for memory-grounded question answering in collaborative settings. Experiments on both datasets show that HiCoMER consistently outperforms strong baselines by reducing outdated retrieval, preserving current team consensus, and improving downstream QA quality. 

---
# Manifold Projection and Iterative Autoencoder Refinement for Masked Language Modeling 

**Authors**: Narges Mokhtari, Farzan Haddadi, Ebrahim Rezaii  

**Link**: [PDF](https://arxiv.org/pdf/2609.30288)  

**Abstract**: In Transformer-based masked language models, attention is the primary mechanism for context mixing, but there are other ways to mix data across tokens. Recent attention-free mixers replace attention with fixed or hypernetwork-generated MLPs, alternating their dynamic, content-dependent weighting for computational simplicity. We build an alternative that gets the same property from a low-rank bottleneck autoencoder. We replace attention with a stack of autoencoder-based mixing modules, one operating over local neighborhoods, one over the full sequence, and one across attention heads, each compressing and reconstructing its input through a bottleneck, and its width is a hyperparameter rather than a training effect. In masked positions, we introduce an iterative refinement procedure that has two distinct steps. A pulling step that pulls an embedding representation toward a weighted average of its neighbors, and a correcting step that projects the result back to the learned manifold via an autoencoder. Our architecture achieves a significant portion of attention's performance at about $1.9 \times$ fewer FLOPs when pretrained on C4 and evaluated with parameter-matched BERT baselines. Our model equals parameter-matched BERT and TinyBERT baselines on the rarest-token frequency bucket using a frequency-aware training schedule that samples rare tokens more than uniformly for the masking tasks. 

---
# A Mechanistic Study of AI-Text Detection Neurons in Frozen BERT: Sparse Probing and Activation Patching on RAID 

**Authors**: Paweł Blicharz, Miłosz Grunwald  

**Link**: [PDF](https://arxiv.org/pdf/2609.30287)  

**Abstract**: AI-generated text detectors achieve high accuracy on standard benchmarks, yet the internal representations that drive these predictions remain poorly understood. We study which neurons in a frozen BERT-base-uncased encoder support AI-text detection, using the RAID benchmark across six generators spanning pure-base and instruction-tuned models. We apply the L1-to-L2 sparse-probing protocol of Gurnee et al. (2023) to all 9,216 CLS hidden-state dimensions (12 layers x 768), which we call neurons. The procedure recovers a stable set of under 1% of neurons per generator, consistent across folds and seeds; a probe restricted to that set retains most of the full-feature detection accuracy. Bidirectional activation patching confirms this set's causal relevance: in both directions it flips predictions an order of magnitude more often than size-matched random sets. Mean-ablating the same neurons leaves accuracy largely intact; the signal is therefore redundantly distributed. Cross-generator analysis reveals a bipartite structure: instruction-tuned generators concentrate 30-36% of stable neurons in BERT's final layer while both base generators fall below 14%, consistent with a layer-12 footprint of post-training alignment. Leave-one-family-out evaluation shows the selected neurons retain 86-94% of the full-feature ceiling on unseen generator families, so a detector can operate on a small fixed subspace without re-identifying neurons per generator. 

---
# Learning to Stop without Learning to Stop: Self-Supervised Confidence Training Improves Reasoning Efficiency 

**Authors**: Parsa Hosseini, Akasha Tigalappanavara, Sumit Nawathe, Chenrui Fan, Sourya Basu, Genta Indra Winata, Anirban Das, Soheil Feizi, Nima Chitsazan  

**Link**: [PDF](https://arxiv.org/pdf/2609.31619)  

**Abstract**: Reasoning models often generate very long reasoning traces, making inference computationally expensive. Existing approaches typically improve efficiency either through inference-time early-stopping mechanisms or by explicitly encouraging shorter reasoning during training, for example through reinforcement learning with length penalties. We show that substantial efficiency gains can instead emerge from a different kind of supervision: \textit{confidence}. Using a self-supervised procedure, we fine-tune reasoning models to predict their confidence in the answer at intermediate points along their own reasoning trajectories using only 600 training problems. Confidence is used only as a training target: the loss contains no objective for reasoning length, efficiency, or stopping. At inference, the fine-tuned models use the standard generation procedure, with no confidence elicitation or early-stopping mechanism. Despite this, self-supervised confidence fine-tuning makes reasoning more efficient, reducing generated tokens by up to 25\% at matched accuracy across Gemma, Qwen, Nemotron, and GPT-OSS models on mathematical, scientific, and coding reasoning benchmarks, with efficiency gains comparable to methods that explicitly optimize for shorter reasoning. Analysis of reasoning episodes further shows that confidence supervision largely preserves the base models' high-level reasoning composition rather than selectively suppressing particular behaviors. Our results suggest that efficient reasoning may emerge as a downstream consequence of learning metacognitive signals, without being directly optimized. 

---
# User Model Extraction via Belief Self-Distillation 

**Authors**: Ali Holmov, Yiran Huang, Kirill Bykov, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2609.31603)  

**Abstract**: Large language models (LLMs) implicitly infer attributes of their users and adapt their behavior accordingly, yet these beliefs remain difficult to inspect and causally manipulate. We introduce Belief Self-Distillation (BSD), a unified read-write framework that bridges linear and causal probing by learning a compact user representation that can be both decoded and written back into the model. The frozen LLM acts as its own teacher, distilling beliefs from natural conversations without external annotations. Unlike conventional probing, BSD isolates not only information present in activations, but a state whose causal role can be directly tested. Across multiple model families, BSD faithfully recovers user beliefs and enables substantially stronger interventions than matched hidden-state steering. Crucially, we find that refusal depends not only on the request, but on the model's inferred user intent: changing this belief alters refusal while holding the request fixed. We further uncover a striking cross-model regularity: independently trained LLMs converge on a shared geometry for representing their users. Together, these results reveal implicit user models as readable and causally writable internal states with direct implications for AI safety, shaping how models condition safety decisions on whom they believe they are interacting with. 

---
# Compact Documentation for Coding Agents: A Benchmark, an Optimizer, and Why It Does Not Transfer 

**Authors**: Md Shohel Arman, Igor Molybog  

**Link**: [PDF](https://arxiv.org/pdf/2609.31587)  

**Abstract**: We investigate whether natural-language documentation helps coding agents resolve software issues, and we build the tools to construct and evaluate it. We introduce a roundtrip benchmark that scores code descriptions by whether code regenerated from them passes the original tests, and show that completeness, not length, drives a description's fidelity. Using the benchmark as an optimization signal, we discover a description-writing prompt that reaches full fidelity and generalizes to unseen files. We then test the hypothesis that motivated the work: that better documentation helps an agent resolve real repository issues. Across two model families and ten repositories, and against a positive control confirming that our evaluation can detect a genuine improvement, we find that it does not. When the source is present, neither static compact documentation nor retrieved context beats the issue alone. We report this negative result together with the benchmark and the optimizer, and we characterize the boundary at which documentation helps. 

---
# Two Conformal Constructions for Adaptive Within-Document AI-Text Screening 

**Authors**: Marco Mandap, Jerahmeel Hipolito, Arcel Galvez, Charlie Margaret Balagtas, Michael Joshua Buluran, Jeff Roel Durmiendo, Rizzette E. Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2609.31547)  

**Abstract**: We study false-alert control when screening for text generated by artificial intelligence (AI). The screening procedure selects document prefixes and detectors from observed evidence and may stop before exhausting its inspection budget. We give two finite-sample constructions under document-level exchangeability between human calibration documents and a new null document, with no restriction on dependence among tokens within a document. Construction A registers a finite family of prefix-detector scores and allocates a false-alert budget across their conformal ranks. A union bound protects any executed subset of that family. Construction B calibrates the complete-path maximum of a development-fixed adaptive policy. Each partial-path maximum is bounded by the complete maximum, so a terminal conformal rank protects early stopping without splitting the error budget. We prove marginal control of any false alert across the permitted inspection path and derive necessary calibration counts for rejection. We also state oracle testing, distribution-shift, and independent-audit bounds with their additional assumptions. Both constructions protect stopping within their specified scope; neither proof constructs an e-process or justifies multiplying conformal ranks. Detection power and computational savings remain questions for empirical evaluation. 

---
# Statistical Foundations for a Google Play User-Review Sentiment Index: Signal Fusion, Shrinkage, Distributional Validation, and Dynamic Smoothing 

**Authors**: Marco Mandap  

**Link**: [PDF](https://arxiv.org/pdf/2609.31513)  

**Abstract**: We develop a statistically explicit sentiment index for Google Play user reviews and establish the mathematical results supporting its construction. Normalized star ratings and text-sentiment scores are treated as noisy measures of latent review valence and fused by covariance-aware inverse-variance weighting. Review-level estimates are aggregated with bounded helpfulness and recency weights, then shrunk toward a population mean using estimated precision rather than an arbitrary review-count threshold. App-level rating histograms provide a distributional diagnostic for samples returned under different API sort orders; because star ratings are discrete, classical continuous Kolmogorov-Smirnov critical values are not used. A local-level state-space model and the Kalman filter provide a denoised temporal trend. Full proofs cover the BLUE and Gaussian maximum-likelihood result, Gaussian-conjugate shrinkage, the Glivenko-Cantelli and Donsker theorems, count transformations via the delta method, and exact Gaussian Kalman filtering. A worked three-review example shows how textual complaints can materially reduce an apparently perfect star-only score. 

---
# PriceBench: A Diagnostic Benchmark for Price, Quality, and Brand Preferences in LLM Booking Agents 

**Authors**: Pavel Kireyev  

**Link**: [PDF](https://arxiv.org/pdf/2609.31468)  

**Abstract**: LLMs increasingly act as purchasing agents, which makes the LLM, not the user, the one choosing among the options that satisfy a request; its preferences quietly fix what gets bought and what it costs. Hotel booking is a clean instance: a high-volume choice settled on a few comparable attributes, where the pick reveals those preferences. We introduce PriceBench, a diagnostic benchmark that recovers an LLM's price, quality, and brand preferences from its booking choices with a logit choice model, applied to 28 LLMs from 8 providers on 3,600 hotel tasks from 179 real New York City properties. We find that capability is associated with how consistently an LLM chooses, not with what it chooses: more capable LLMs hold stronger, more consistent preferences, while weaker ones either lock onto one position, exploitable by whoever controls listing order, or choose almost indifferently. What those preferences favor varies sharply across providers and even within one family: price sensitivity spans more than an order of magnitude, and the price/quality trade-off moves mean booked nightly price from \$247 to \$393 on identical tasks. What an agent buys must therefore be measured per LLM, not inferred, and we release the tasks, code, and all 28 response sets. 

---
# Towards Mitigating Fabricated Consensus: The Active Provenance Gate for Multi-Agent Debate Synthesis 

**Authors**: Jakub Masłowski, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2609.31422)  

**Abstract**: Large language model-based multi-agent debate (MAD) systems are being increasingly used as complex decision pipelines in distributed processes. However, their final synthesis phase still remains inadequately controlled. Even with detailed debate logs, summarizing models are prone to fabricating smoothly written debate consensus that is not grounded in the debate's history. To address this safety gap, this paper presents empirical research and studies if the introduction of active post-debate verification can mitigate the production of such factually unsupported summaries, while still providing valuable information. Furthermore, it is examined whether explicitly signalling divergence is preferable in the absence of a reliable compromise. The Active Provenance Gate (APG) is introduced as a post-debate verification layer that treats the source as a hard constraint, analysing the debate logs, auditing each claim, and applying self-correction. In crisis simulations, the self-healing mechanism more than doubles the average data Provenance Fidelity in difficult condition scenarios, before the strict gate blocks unsupported claims and generates divergence reports. In the human study, a vast majority of the users (over 75%) preferred a report explicitly stating failure in critical scenarios, despite most of them perceiving fabricated consensus from the baseline system as more fluent. Our main contribution is the transition of data origin tracing from passive logging to active conditional blocking before publication. 

---
# Intent2Tc: Automated Intent-to-Traffic Control Translation with Language Models 

**Authors**: Andrea Masini, Sudipta Acharya, Paolo Bellavista, Luca Foschini, Burak Kantarci  

**Link**: [PDF](https://arxiv.org/pdf/2609.31397)  

**Abstract**: Automated and highly usable Quality-of-Service (QoS) enforcement requires translating high-level service intents into deployable traffic-management policies. Although intent-based networking (IBN) has simplified policy specification, bridging the gap between business-level intents and executable network configurations remains complex, error-prone, and difficult to automate. This paper presents Intent2Tc, a closed-loop language-model-driven framework that translates business-level traffic-shaping intents into declarative sub-intents and subsequently into validated, executable Linux traffic control (tc) configurations. The framework integrates an Active Queue Management (AQM)-based digital twin (DT) semantic model, automated metadata extraction, critique-driven refinement, and Retrieval-Augmented Generation (RAG)-based knowledge reuse to improve semantic consistency and configuration reliability. We evaluate multiple open-source large language models (LLMs) and small language models (SLMs), together with Claude Sonnet-4.6, on 100 Request for Comments (RFC) 9315-compliant traffic-shaping intents. Across both translation stages, Intent2Tc achieves high semantic fidelity, configuration accuracy, and deployment readiness, with Claude Sonnet-4.6 reaching 0.98 semantic similarity, 1.0 semantic unit coverage, and 0.045 normalized edit distance. Furthermore, RAG reduces token consumption and inference latency while enabling compact models such as Phi-4-mini to approach the performance of substantially larger models. Linux tc serves as the target configuration platform, demonstrating the practical applicability of the proposed framework. 

---
# The Right Information Extraction Pipeline Depends on the Document: Accuracy-Energy Trade-offs for Small, Local Models 

**Authors**: Christoph Walser, Mauricio Fadel Argerich, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2609.31341)  

**Abstract**: Whether an information extraction pipeline should process page images or parsed text depends on the document, and the answer flips across the layout spectrum. We study this trade-off under a constraint that rules out (closed) cloud services: privacy-sensitive documents processed on-premise by small ($\le 8\mathrm{B}$ parameter) text-only and vision--language models, evaluated on both accuracy and energy over a design space spanning input representation, model family, and inference configuration. Benchmarking on the near-plain-text Kleister-NDA contracts and the layout-rich VRDU forms, we find that batching is the dominant energy lever, cutting energy per page by 38-85% at no cost in accuracy, while FP8 quantization saves 27-32% when requests are served one at a time but less than 1mWh per page (9-19%) once batching is applied. Preprocessing dominates what remains: neural OCR costs $17\times$ more energy per page than classical OCR and never reaches the Pareto frontier. Which representation wins flips with the type of document: vision--language models on layout-rich documents and small text-only models with a cheap parser on near-plain text, where they are both more accurate and cheaper than any vision--language configuration. Our work yields concrete guidelines for energy-efficient, privacy-compliant local information extraction. 

---
# Why Alzheimer's Speech Screening Fails to Generalize: Bridging the Deployment Gap via Cross-Corpus Evidence Anchoring 

**Authors**: Zijian Lu, Sizhe Liu, Yin Zhang, Jixuan Deng, Xinrong Lin, Xinchen Yuan, Chicheng Jin, Yiping Zuo, Yuanchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.31293)  

**Abstract**: Speech-based screening is a promising, non-invasive approach for detecting Alzheimer's disease and related cognitive risks. However, models trained on a single domain often generalize poorly to unseen languages, tasks, or recording protocols. This paper investigates this deployment gap using a leave-one-corpus-out evaluation across four distinct datasets. Among 70 interpretable speech and language features, 59 exhibit direction conflicts between healthy control and cognitive risk groups across corpora, with pause, silence, and speech rate showing high protocol sensitivity. Furthermore, while the XLM-R text baseline achieves strong average performance, its Area Under the ROC Curve (AUC) drops to 0.520 on the weakest held-out domain. A standard GroupDRO baseline reaches a 0.766 mean speaker AUC and a 0.504 worst-domain AUC under the same protocol. To address this, we propose a fusion method that integrates XLM-R text baseline scores with evidence anchors selected during training. Balanced fusion achieves a 0.785 mean speaker AUC, while anchor-heavy fusion raises the worst-case speaker AUC to 0.615. This work highlights the need to audit feature transferability and report worst-case domain robustness in cognitive speech screening. 

---
# JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models 

**Authors**: Jianyi Hu, Hangtao Zhang, Yi Liu, Yeqi Zeng, Li Zeng, Xianlong Wang, Rui Wang, Leo Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.31142)  

**Abstract**: Models trained with reinforcement learning for calibrated decisions (RLCD), such as Jev, answer a typed question about an input, the state, with a probability, a choice, or a score, and software acts on the answer without a person reading it. Their robustness has not been measured: adversarial benchmarks score what a model generates or executes, whereas a typed model generates nothing and returns a well-formed answer even when manipulated. Measurement is also hard, because identical requests can return different answers, most available labels come from the model itself, and the API preprocesses each request out of view. Our key idea is to score each attacked decision against the model's own clean decision rather than against labels, and to read it against the change caused by an identical re-run. Building on this, we introduce JevAdvBench, to our knowledge the first adversarial benchmark for RLCD models, with 812 typed questions over 66 scenarios, and a black-box attack suite of 9,744 single-edit variants that each edit one part of a request, with billed input tokens confirming that the edit reached the model. On jev-1.13.0, rewording stays within 1.2 percentage points of the re-run baseline, and fields outside the schema never reach the model. In contrast, one unverified opinion appended to the state flips 12.1% of decisions, statistically tied with the strongest injected command (10.1%), and pushes 38% of confident answers below the 0.8 confidence threshold that routes them to human review. Applications built on RLCD models should therefore treat the state as untrusted, argued input. Project website: this https URL 

---
# KuaFu: Compressing Long User Behavior into Understanding at Billion Scale 

**Authors**: Jiahao Hui, Lin Zhu, Yishen Hu, Jingdong Shu, Zetai Jiang, Xining Ran, Ben Tan, Yeshou Cai, Gong Chen, Haijie Gu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.31045)  

**Abstract**: Conversational agents, generative recommenders, and personalized advertising all rest on one capability: understanding each user from raw behavior. Prevailing industrial practice is task-specific: for each task, a relevant subsequence is extracted from the full history and a dedicated model trained on it. In production it hits two bottlenecks. First, even after filtering, a single-task sequence stays extremely long: content-interest summarization reads several hundred items per user, tens of thousands of tokens once serialized as prompt text. Second, profiles are refreshed routinely: a billion users weekly, roughly 100K QPM in aggregate, which under a fixed GPU budget sets a hard throughput floor. Compression is therefore mandatory, yet truncation or coarse compression can silently distort the profile, introducing four hallucination types (fabrication, omission, date misattribution, broken logic) that, with no way to evaluate the compressed representation itself, surface only as diffuse degradation in downstream metrics. We present KuaFu, a unified behavior-compression layer whose minimal unit is one behavior item. A two-axis projector compresses each item into 2-4 tokens of width 128-256 (about 10x along the token axis, 20x along width; per-item cache 10 KB to 0.5 KB), with fidelity-oriented four-stage training and layered intermediate evaluation. Across four production profiling tasks it matches or exceeds uncompressed single-task production models on all five headline metrics, raises per-GPU throughput by 37%-350%, and saves 190 GPUs. On public benchmarks it nearly always beats prior compressors at the same compression ratio (up to +17.7 EM on out-of-domain MRQA); on RecBench, a 4B model surpasses its 8B counterpart by 1.90 points. KuaFu has run on the Tencent advertising and recommendation platform for ten months, lifting overall GMV by 1.37%. 

---
# Same Text, Different Numbers: The Divergence of LLM-Based Measures 

**Authors**: Hamid Boustanifar, Sasan Mansouri  

**Link**: [PDF](https://arxiv.org/pdf/2609.31013)  

**Abstract**: Researchers increasingly use generative large language models (LLMs) to convert corporate text into empirical variables. We examine the extent to which LLM-based textual measures are invariant to model choice using thirteen measures, including sentiment, management clarity, uncertainty, answer specificity, and climate and political risk. Seven LLMs from different providers score earnings call transcripts of S&P 500 companies on these constructs. Cross-model rank correlations average only 0.52, and transcript-level differences common across providers account for only 34% of total score variation. Cross-model disagreement does not predict subsequent analyst or market disagreement, consistent with a substantial model-specific component rather than common ambiguity in the underlying disclosure. Model choice significantly affects downstream inference, with coefficient magnitudes, signs, and statistical significance varying substantially across models. Averaging across providers makes transcript rankings more stable for most constructs, but score levels remain sensitive to the models included in the ensemble. LLM-generated variables should therefore be treated as model-contingent measurements and validated across providers. 

---
# Does Uniform Discrete Diffusion Need Time? 

**Authors**: Chunsan Hong, Chieh-Hsin Lai, Satoshi Hayakawa, Yuhta Takida, Jong Chul Ye, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2609.30977)  

**Abstract**: Uniform discrete diffusion models (UDMs) commonly use explicit time conditioning, but we find that it can often be unnecessary in practice. In this paper, we first show that the population-optimal UDM predictor generally depends on time: time controls how much the model should trust the observed context. We then show that this dependence can become negligible in finite-data settings relevant to language. When a corrupted training sequence remains much closer to its original clean sequence than to competing training sequences, the empirical-optimal predictor is nearly insensitive to time over most of the diffusion trajectory, where the guarantee weakens toward the high-noise endpoint. Empirically, trained language UDMs exhibit limited time sensitivity over most of the trajectory, while time-agnostic predictors remain competitive with, and often outperform, time-conditioned models across datasets and training objectives. These results challenge the use of explicit time conditioning in UDMs: although the population optimum depends on time, explicitly conditioning on it may often be unnecessary in practice. 

---
# Cross-Backend QIEO: Universal Runtime Portability across OpenMP5, CUDA, HIP, and Multi-Language Interfaces 

**Authors**: Aman Mittal, Ferdin Sagai Don Bosco, Kasturi Venkata Srikanth, Abhishek Singh, Aditya Singh, Abhishek Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2609.30914)  

**Abstract**: Quantum-inspired algorithms emulate quantum mechanical principles, such as, superposition, interference, and probabilistic amplitude evolution, on classical hardware by representing candidate solutions as qubit vectors and evolving them through rotation-gate operators. This approach offers higher optimization performance without physical qubits, and has been shown to achieve order-of-magnitude speedups (10--80$\times$) over traditional solvers on combinatorial, high-dimensional NP-hard problems.
A critical barrier to adoption, however, is the lack of a unified execution framework that delivers both algorithmic performance and hardware portability. We present \textbf{Cross-Backend Quantum Inspired Evolutionary Optimizer (QIEO)}, the runtime core of BQP's BQPhy solver, which addresses this gap through a \emph{single-source-of-truth} architecture. One C++ implementation of the QIEO algorithm is compiled once per hardware target and exposed to multiple high-level languages via thin binding layers. The framework dispatches to CPU (sequential), OpenMP~5 (multi-core), CUDA (NVIDIA), and HIP (AMD) backends at runtime, adapting kernels to each device's memory hierarchy and warp/wavefront execution model.
The framework's real-world utility is validated through binding demonstrations that share the identical C++ runtime. BQPhy's Python library is demonstrated on a neural network hyperparameter optimisation achieving 88.60\% test accuracy on MNIST. BQPhy's MATLAB's Toolkit is tested on wind farm layout optimisation attaining $365\,399 \pm 4\,552$~MWh/yr, which is statistically indistinguishable from particle swarm optimisation and $+7.6\%$ above genetic algorithms on a 32-variable constrained engineering problem. The Julia package tackles the Lotka--Volterra parameter estimation where BQPhy replaces native Julia solvers on the same residual, cutting mean SSE by $2.1\times$. 

---
# Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness 

**Authors**: Nux Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.30820)  

**Abstract**: Looped transformers reuse weights across recurrence steps, making low-bit quantization especially attractive. We identify two distinct failure modes of standard post-training quantization. On Huginn-3.5B, per-channel INT4 fails primarily at the non-residual loop-entry adapter, while quantizing the residual core is much less damaging. We call this feedback exposure: a quantized layer perturbs the recurrent state without an identity path, and the resulting error is fed back at later steps. Controlled experiments on linear filters and Mamba state-space models show that feedback exposure also occurs outside transformers. Grouped INT4 reveals a separate failure, calibration blindness: our one-step GPTQ baseline builds its Hessian from step-0 activations, leaving input directions used later in the recurrence nearly unweighted. Across nine checkpoints from seven looped architectures, one-step GPTQ is worse than round-to-nearest (RTN) on the primary task metric for five checkpoints. Accumulating the GPTQ Hessian across recurrence steps outperforms both one-step GPTQ and RTN on all nine checkpoints and recovers bf16-level accuracy on Huginn. These results separate two questions for PTQ on looped models: where quantization error enters the recurrence, and which states calibration sees. 

---
# Symbiotic Architecture for Post-Hoc Audio Extension of Frozen Language Models 

**Authors**: Yotaro Kubo, Qi Sun, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2609.30784)  

**Abstract**: This paper proposes an architecture for equipping large language models (LLMs) with audio-understanding capabilities without fine-tuning their weights. The proposed symbiotic architecture employs an injector module that writes audio-conditioned vectors directly into the target LLM's short-term memory, i.e., the key-value (KV) cache, enabling the LLM to behave as an audio language model (ALM). The architectural advantages are twofold. First, it improves the scalability of ALMs: because the proposed method bypasses the LLM during audio injection, the injection cost is governed by the injector width rather than the backbone width, and can therefore scale more slowly than the cost of full-backbone prefilling. Second, since the training scheme does not update the LLM weights, the original capabilities of the LLM are preserved without the risk of degradation from fine-tuning. The effectiveness of the proposed method is evaluated on both audio-understanding tasks (automatic speech recognition, audio question answering, and acoustic scene classification) and text-only tasks. We confirm that, while activating fewer parameters during audio prefilling, our architecture outperforms the conventional method with a frozen LLM and approaches the performance of a fine-tuned ALM, all while preserving the backbone LLM's original text-only task performance by construction. 

---
# LAVOIR: Teaching a Single-Pass Decision Encoder When and What to Ask with Amortized Value of Information 

**Authors**: Furkan Yilmaz, Habibe Aleyna Tasdemir, Muhammed Faruk Gozay  

**Link**: [PDF](https://arxiv.org/pdf/2609.30706)  

**Abstract**: "System One" decision models such as TypeSafe's Jev and its open counterpart Laya answer typed questions about a text in a single forward pass with calibrated probabilities, but they cannot ask for missing information: when a first message does not say what separates two departments, they guess. We present LAVOIR (Laya with Value-Of-Information Routing), which places the candidate pieces of missing information (slots) in the input next to the answer options, so that one forward pass returns both the decision distribution and, for every slot, the expected gain in the probability of the correct decision if the user were asked about it. VOI targets need no human labels: gold decisions come from schema rules, an LLM only verbalizes messages and answers, a model from another family checks every text, and pairing each message with several profiles makes regression on realized gains estimate the expected gain. A Gini-impurity cap bounds the predicted value by what a calibrated model can still gain. In a controlled study, decisions on seen schemas are statistically indistinguishable from the Bayes ceiling. The final model's question policy matches a greedy oracle VOI policy on seen schemas (AUC 0.799 vs. 0.797), and with at most 0.5 questions per conversation it is 14.1 points more accurate than never asking. On real ABCD conversations, one real exchange raises accuracy by 8.3 points where LAVOIR asks and leaves it unchanged where it does not; on SGD the cap lowers the asking rate from 93% to 8.6%. On Laya's twelve benchmarks LAVOIR is above Laya's reported scores on seven, and it answers a question in 31 ms (median, GH200). 

---
# Prompt Injection Detection for Email Agents Through Attack Chain Modeling 

**Authors**: Ahmad Hashmi, Dhyey Patel, Yunting Yin  

**Link**: [PDF](https://arxiv.org/pdf/2609.30657)  

**Abstract**: Large language model email assistants are particularly vulnerable to indirect prompt injection because untrusted email content can be retrieved into the model context and influence subsequent tool use. Existing prompt injection detectors mainly formulate this problem as binary malicious text classification, which overlooks the important factor that harmful agent behavior often arises through a sequence of stages. We propose a detection framework that models this attack chain by combining a text detector, verifiers specific to each stage, explicit rule-based risk signals, user intent and action consistency analysis, and a logistic decision policy. To support this framework, we derive attack chain labels from prompt injection datasets, evaluate the proposed framework under random splits, temporal phase transfer, conditional stage transfer, cross-dataset transfer, and conduct ablation studies on multiple benchmarks. Results show that random train test splits substantially overestimate robustness under distribution shift, while later tool argument stages are more predictable than earlier stages in the framework. We also show that training on harmless emails that resemble attacks helps reduce false alarms while preserving the ability to detect real attacks. Across five binary benchmarks, our framework achieves a mean F1 score of 0.406 under the strict threshold setting policy, compared with 0.216 for the strongest of five pretrained detectors evaluated without additional training. These results highlight the value of combining attack stage predictions with checks for conflicts between the user's request and instructions in retrieved emails. Our experiments also demonstrate the importance of training with challenging benign examples to balance attack detection and false alarms. 

---
# Epstein Files Engine: Agentic Search for Investigative Journalism 

**Authors**: Duy K. Nguyen, Teresa Mondría Terol, Dylan Freedman, Zach Seward  

**Link**: [PDF](https://arxiv.org/pdf/2609.30611)  

**Abstract**: On Jan. 30, 2026, the U.S. Department of Justice released a mixed-media collection concerning Jeffrey Epstein, including about three million pages of PDFs. We describe the Epstein Files Engine, an A.I. agent The New York Times deployed to investigate the files. The Engine translated reporter questions into Google BigQuery SQL queries across three corpora: Epstein-related releases, the Times's archive and external, Epstein-related news headlines. It used an LLM to plan queries and returned citation-rich answers a reporter could verify and trust. More than 100 journalists used the Engine, and it contributed to at least 20 published stories. We report how reporters queried it and describe Diff, our text-and-visual duplicate matching method that amplified novelty signals and allowed the Engine to surface genuinely new information. We argue that newsroom agents serve newsrooms best not as autonomous writers, but as interfaces to source material and institutional knowledge. 

---
# Thinking Less to Simulate Better: Intuitive Prompting Improves LLM Agents Simulating Individual Social Media Reactions, Including Unfamiliar Content 

**Authors**: Ljubisa Bojic, Tijana Stanic, Joerg Matthes, Agariadne Dwinggo Samala, Bojana Dinic, Jue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.30563)  

**Abstract**: Platform policies are increasingly tested on artificial users, making agent fidelity important. Yet convincing fake profiles could also manipulate perceived public opinion before elections. Validation has concentrated on agreement with human behaviour and has paid little attention to whether an agent behaves in line with the profile it was given. The present study profiled eight Serbian participants through a questionnaire, a deep interview, and a written self-presentation, recorded their reactions to sixty-eight social media posts, and asked four language models to predict those reactions under five prompt conditions varying profile content and instruction style. Attitudinal content improved prediction over demographic backstories by a wide margin. Agents matched their stated profiles more closely than participants matched their own survey answers, and consistency proved unrelated to fidelity once profile information was present. Instructing models to respond intuitively and immediately rather than analytically gave the highest fidelity of any condition and cut the compression of individual differences from seven times the human level to three. The advantage held on posts about topics the questionnaire never raised, where that condition reached the highest fidelity of any setup and beat a crowd baseline by a wide margin, which suggests that agents prompted this way could serve as general-purpose simulated users rather than specialists on the topics they were profiled for. Results may bear implications for the development of language models, because intuition-based setups appear better suited to some tasks than reasoning-based ones. 

---
# Don't CLAP: Are Music-Text Models Bag-of-Words? 

**Authors**: Yuan-Chiao Cheng, Alexander Lerch  

**Link**: [PDF](https://arxiv.org/pdf/2609.30540)  

**Abstract**: Text-to-music systems are assessed on audio quality and on how faithfully the music follows its prompt, and the CLAP score, the cosine similarity between a music-text model's audio and text embeddings, is the standard objective metric of faithfulness. We ask how accurately that score reflects the text: when an attribute is linked to an instrument (e.g., distorted guitar), does the text embedding capture that binding? To find out, we introduce an attribute swap perturbation: the caption of a real recording is edited by exchanging exactly one property, timbre, lead versus accompaniment, or order of first appearance, between two instruments. We then test four contrastive music-text models and one large audio-language model on whether the audio scores higher against the original caption than against the perturbed one. No contrastive model distinguishes the two captions reliably. The audio-language model does better, but further experiments show that its advantage rests largely on audio-agnostic language priors. Our results thus provide compelling evidence that the CLAP score and related metrics do not capture fine-grained musical meaning or attribute bindings; their representation is closer to a bag-of-words that leaves them insensitive to meaning-changing perturbations of the caption. 

---
# AcoustiClaim: A Numeric Claim Benchmark with Instrument Ground Truth 

**Authors**: Sheng-Tse Lin, Siyuan Zhai, Chien-Liang Kuo, Massa Baali, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2609.30483)  

**Abstract**: Audio language models state numbers for acoustic quantities, and neither human opinion nor a judge model says whether such a number is true of the signal. AcoustiClaim extracts each numeric claim from free text, scores it against the instrument that defines the quantity, and classes each quantity by where its reference can be read. Four open-weight systems and one closed model, asked for ten quantities five ways on two corpora, fill 207 cells. Of these, 49 emit fewer than five distinct values, and eight of the 158 cells that can be ranked exceed a rank correlation of 0.3, the bar we set, three with an interval clear of it, five of them one closed model reading pitch. Error sits at or above a constant-predictor floor in every ranked cell but three. The reference decoder we train declines the five voice quantities in prose on 95% of mixtures, with nothing withheld, and states them on the clean twins, reproducing its targets' rule from audio alone. With a calibrated threshold, withholding lowers error on all ten quantities on the mixtures in the mean and on eight at every split, against at most 0.6% from a random selector. A linear baseline orders errors at least as well as ours. F0 s.d. and shimmer stay above the constant floor. 

---
# Asymmetric Classifier-Free Guidance for Target-Speaker ASR 

**Authors**: Yiwen Guan, Jacob Whitehill  

**Link**: [PDF](https://arxiv.org/pdf/2609.30476)  

**Abstract**: Target-speaker automatic speech recognition (TS-ASR) must identify and transcribe a desired speaker under varying overlap and noise conditions. These changes alter the acoustic evidence for the target speaker in the speech mixture, motivating inference-time calibration of speaker conditioning. We introduce asymmetric classifier-free guidance (CFG) for TS-ASR using Whisper: the speaker-conditioned branch predicts the target transcript, while the speaker-unconditioned branch predicts serialized multi-speaker transcripts. CFG adjusts the contribution of speaker conditioning during decoding through a single guidance scale. We select a global guidance scale on target-domain development data and train a lightweight encoder-based predictor to adjust it for each utterance, keeping the recognition model fixed. Under domain shifts, our full system achieves relative word error rate (WER) reductions of up to 21.8% over the condition-only baseline, and 5.6% over standard conditional decoding of the same CFG-trained model. Oracle analysis shows that substantially larger WER reductions are possible through utterance-level scale selection and identifies how beneficial adjustments vary with domain shifts. 

---
# RAZOR: Pruning Replaceable Experts in LLMs 

**Authors**: Mingyang Song, Mao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.30465)  

**Abstract**: Mixture-of-experts (MoE) models activate few experts per token but store the full expert pool. Expert pruning reduces this storage burden; at a fixed pruning budget, the goal is to preserve the original model's output distribution as closely as possible. Yet an expert's usage or contribution magnitude does not by itself determine the damage caused by its removal. What matters is whether the surviving computation can replace its function. We introduce RAZOR, a training-free expert pruning method that scores functional replaceability using consensus residuals: deviations of expert outputs from the original weighted mixture. An exact single-deletion identity at a fixed layer input accounts for survivor renormalization and router-selected refill, providing local scores aggregated over calibration tokens for budgeted pruning without gradients or recovery training. On GLM-4.7-Flash, Qwen3.6-35B-A3B, DeepSeek-V4-Flash-0731, and Hy3 at 25\% and 50\% expert removal, RAZOR achieves the highest nine-task macro average among the evaluated pruning methods in all eight settings. On the two backbones with matched REAP benchmark runs, it exceeds REAP by 2.12--5.59 points and wins all 36 paired task comparisons. It also lowers reverse KL relative to REAP in all four matched GLM-4.7-Flash and Qwen3.6-35B-A3B model--budget settings. Analysis of responses generated by Qwen3.6-35B-A3B nevertheless reveals changes in diversity, formatting, and termination, underscoring that task retention and predictive fidelity do not ensure generation stability. 

---
# What Improves Multimodal Misinformation Detection? Answers from a Large-Scale Empirical Study 

**Authors**: Akshit Sharma, Prashant W. Patil  

**Link**: [PDF](https://arxiv.org/pdf/2609.30402)  

**Abstract**: Multimodal misinformation is increasingly crafted to look convincing by pairing a textual claim with an image that appears to "prove" it. Yet in practice, building effective detectors often hinges on a small set of design choices that are rarely examined in a controlled way. In this paper, we conduct a large-scale study of multimodal design choices for misinformation detection with over 3,375 experiments- spanning three benchmark datasets and a broad range of pre-trained vision and language backbones. Through systematic comparisons and targeted robustness analyses, we distill practical guidance on which design choices help, when do they fail silently, and what aspects of the pipeline most strongly shape model behavior, answering 4 key Research Questions (RQs). We aim to provide a reliable foundation for designing stronger and more dependable multimodal misinformation detection systems, thus contributing to the broader research community. 

---
# When Is a Multi-Agent Code Judge Actually Grounded? Two Label-Free Measurements, and a Judge That Declines to Guess 

**Authors**: Salma Roshdy Aly, Hussein Assaf, Ziad Kobti  

**Link**: [PDF](https://arxiv.org/pdf/2609.30328)  

**Abstract**: When one language model judges whether another's code is correct, it does not report the absence of evidence. It returns a confident verdict with reasoning attached, indistinguishable from a verdict it had grounds for. Multi-agent verification, which decomposes a judgment into checkable claims and verifies each against evidence, is a promising response and works well when the evidence is a set of retrieved documents.
We argue such methods require two things of their evidence: it must be independent of the answer under review, and it must differ between the two candidates being compared. The second condition holds automatically with retrieved documents and stops holding in code judging.
Running MARCH, a published framework unmodified over 80 condition-by-cell measurements on two code judging benchmarks, we find it declares both solutions equally good on 78 to 95% of comparisons, reaching 4.4% accuracy where the same model asked directly reaches 43.7%. Neither easier problems nor a larger judge changes this. Two measurements taken from the pipeline's own logs explain it without needing labels.
Gating on one of them, the pipeline declines the comparisons it cannot make and raises its accuracy from 20.7 to 36.9% while still answering half of all comparisons. The contribution is not a more accurate judge, but a label-free way to tell when a judge has no basis for its answer. 

---
# PALM: Point-in-Time Adaptation for Financial Language Models 

**Authors**: Seunghan Lee, Jun Seo, Jaehoon Lee, Junhyeok Kang, Sangjun Han, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, Soonyoung Lee, Wonbin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2609.30316)  

**Abstract**: Language models used in financial backtests suffer from look-ahead bias, as a model trained on text published after the study period has already observed the outcomes it is asked to predict. To handle this issue, point-in-time (PIT) language models are pretrained on chronologically filtered corpora and released as one checkpoint per calendar year, each with a documented cutoff. However, each additional year costs a full pretraining run, and whether that run is necessary has never been tested. In this paper, we show that the annual pretraining run is not necessary. We instead compare each checkpoint against the newer one that replaced it, and find that the newer checkpoint scores no better on the same evaluation window. Motivated by this observation, we propose PALM (Point-in-time Adaptation for financial Language Models), a simple yet effective alternative to annual pretraining that fits a low-rank adapter on text published before the decision date without modifying any pretrained weight. We further find that a small adapter is enough to add a new period to the knowledge an old checkpoint already encodes, and that this outperforms continued pretraining. We validate PALM on a decade of financial news and on various families of PIT models, whose cutoffs span two decades and whose sizes range from 1.3 to 4.2B. Code is available at: this https URL. 

---
