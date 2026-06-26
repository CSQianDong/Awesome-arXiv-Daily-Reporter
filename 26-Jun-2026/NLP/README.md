# Mapping Political-Elite Networks in Europe with a Multilingual Joint Entity-Relation Extraction Pipeline 

**Authors**: Kirill Solovev, Jana Lasser  

**Link**: [PDF](https://arxiv.org/pdf/2606.27347)  

**Abstract**: Whether political elites organise into rent-seeking coalitions that capture public resources or civic networks that sustain governance is a central question in comparative politics. Yet observing these complex, informal, and adversarial ties at scale has historically required intensive manual coding, while automated text-as-data methods have largely been limited to simple co-occurrence. Recent large language model (LLM) approaches offer a path forward but often rely on proprietary APIs, lack cross-lingual capability, and struggle with scalable entity resolution. We present a modular, fully open-weight pipeline for multilingual joint entity-relation extraction that builds signed, temporal knowledge graphs from massive unstructured news corpora. It combines span-based named-entity recognition (NER) with a three-stage linking cascade mapping mentions to language-independent Wikidata identifiers; a high-throughput, ontology-constrained mixture-of-experts model then uses guided decoding to extract directed, signed relationships grounded in a domain ontology. A full-coverage spot-check against a 3491-relation gold standard shows high textual correctness (68.2% strict to 93.7% lenient). Two large-scale case studies validate the pipeline against the public record. In Austria, it reconstructs a political party's complete lifecycle, dating internal fractures and tracking personnel into successor factions and court convictions. In a Polish corpus, it uncovers the overlapping economic and governance networks of state-enterprise patronage, alongside the structurally balanced, signed conflict network of the polarized Civic Platform (Platforma Obywatelska, PO)--Law and Justice (Prawo i Sprawiedliwość, PiS) duopoly. By bridging raw multilingual text and structured relational data, our framework provides a robust, replicable foundation for cross-national empirical computational social science. 

---
# Empowering GUI Agents via Autonomous Experience Exploration and Hindsight Experience Utilization for Task Planning 

**Authors**: Tianyi Men, Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.27330)  

**Abstract**: Multimodal web agents can assist humans in operating repetitive GUI tasks, where effective task planning is essential for decomposing complex tasks into executable actions. While small open source MLLMs are cost efficient and privacy preserving compared with commercial large models, they suffer from weak planning and limited cross website generalization. To address these limitations, we introduce the planning experience exploration and utilization (PEEU) method, which autonomously explores environments to discover experiences and utilizes hindsight experience to synthesize strictly aligned, high level training data. To quantitatively analyze the generalization behaviors driving this performance, we propose the task decomposition hierarchical analysis framework (TDHAF) to systematically study compositional generalization across three task granularities: low, middle and high levels. Our analysis reveals that mastering low level atomic skills does not guarantee high level planning competence, while high level task training yields stronger OOD generalization. Experiments on real world benchmarks demonstrate PEEU's superior effectiveness: our 7B model achieves 30.6% accuracy, outperforming the much larger Qwen2.5-VL-32B model. These demonstrate constructing hindsight high level tasks and leveraging experiences is crucial for OOD planning abilities of small MLLMs. 

---
# LLM-Based Examination of Eligibility Criteria from Securities Prospectuses at the German Central Bank 

**Authors**: Serhii Hamotskyi, Akash Kumar Gautam, Christian Hänig  

**Link**: [PDF](https://arxiv.org/pdf/2606.27316)  

**Abstract**: Verifying the eligibility of securities as collateral is a key responsibility of the German Central Bank. However, manually verifying these assets against legal and financial criteria within lengthy, semi-structured, and often bilingual prospectuses is a resource-intensive task. While previous efforts utilized traditional Named Entity Recognition (NER) for information extraction, these methods can struggle with OCR noise, linguistic variance, and rigid span-based constraints, and the need for manually annotated training data for each relevant annotation type. In this paper, we present the first case study applying Large Language Models (LLMs) to the eligibility examination process, shifting the paradigm toward a generative Information Extraction pipeline. Our approach decomposes the task into extraction, normalization, and interpretation, allowing for greater flexibility in handling noisy text and interleaved German-English content. We further introduce a value-based evaluation methodology using LLM-as-a-judge, which offers a more semantic assessment than location-based metrics. Our results demonstrate that LLM-based systems achieve high precision (up to 91%) in document-level eligibility, exhibiting a conservative operating profile that minimizes false acceptance. 

---
# Beyond Surface Forms: A Comprehensive, Mechanism-Oriented Taxonomy of Indirect Linguistic Encoding for LLM-Based Coded Language Detection 

**Authors**: Hamid Reza Firoozfar, Mohammadsadegh Abolhasani, Reza Mousavi, Paul Jen-Hwa Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27314)  

**Abstract**: To avoid moderation and surveillance on social media, some users routinely invent indirect linguistic expressions (ILE) that camouflage sensitive meanings. Such expressions surface as algospeak, euphemisms, and adversarial obfuscation, depending on intent and context, and they involve recurring encoding mechanisms. We propose a comprehensive, mechanism-oriented taxonomy of ILE that abstracts away from communicative goals and instead categorizes the underlying operations through which meaning is encoded and recovered. We evaluate the taxonomy by incorporating it into LLM prompts and comparing it with four existing taxonomies and a no-taxonomy baseline, using 2,000 manually annotated TikTok and Bluesky posts. The proposed taxonomy attains the strongest document- and span-level performance across the three LLMs, achieving an improvement of 4.7% in accuracy and 5.4% in F1 over the best-performing benchmark. The empirical results reveal the importance of a comprehensive, mechanism-oriented taxonomy as a stable scaffold for detecting emerging coded language and a useful input to content moderation. Disclaimer: This paper contains content that may be profane, vulgar, or offensive. 

---
# Multilingual Reasoning Cascades Need More Context 

**Authors**: Arnav Mazumder, Dengjia Zhang, Shuyue Stella Li, Yulia Tsvetkov, Niyati Bafna  

**Link**: [PDF](https://arxiv.org/pdf/2606.27306)  

**Abstract**: Translation cascades for reasoning translate the query from another language to English, reason in English, and translate the answer back to the original language. This is a competitive approach to multilingual reasoning, but structurally lossy, since each stage discards information later stages may need, including cues for cultural grounding, register, and disambiguation. We examine the benefits of a simple and training-free intervention: a context-aware translation cascade, which additionally provides the original question, the English translated question, and the reasoning trace to the context of the final translation module. We evaluate gains across nine multilingual benchmarks including various task types, three backbone models, and 285 high-, mid-, and low-resource languages, and demonstrate strong gains for open-ended generation across models and resource regimes. We show that the original language question carries most of the beneficial context. Our study emphasizes the need to better design information flow in machine translation cascades for mitigating error propagation, and provides a simple and actionable default strategy: preserve the original user question until the end of the pipeline. 

---
# How Surprising Is Historical Italian to Language Models? Tokenization Tax, Comprehension Tax, and a Simple Mitigation 

**Authors**: Maria Levchenko  

**Link**: [PDF](https://arxiv.org/pdf/2606.27275)  

**Abstract**: Large language models (LLMs) are increasingly critical to digital library workflows, yet their ability to process historical language remains poorly understood. Historical difficulty is typically treated as a monolithic barrier, conflating orthographic variation, linguistic distance, and pretraining exposure. In this paper, we propose a diagnostic framework that decomposes this difficulty into four distinct dimensions: tokenization cost, predictive uncertainty (surprisal), semantic robustness, and context sensitivity.
We evaluate this framework on three datasets spanning three centuries: (1) a newly curated corpus of 17th-century Italian texts (1610-1689) digitized from original page images; (2) canonical 19th-century Italian "I Promessi Sposi" serving as a high-exposure control; and (3) 18th-century Russian civil print books as a contrastive orthographic stress test.
Our results reveal a distinct dissociation between encoding cost and comprehension. While Russian and early modern Italian incur comparable tokenization penalties (25-30% inflation), their predictive difficulty diverges sharply. 17th-century Italian is on average 2.4 times more surprising than its modern equivalent - with academic prose reaching 3.2 times - whereas Russian shows only a modest increase. But predictive uncertainty does not imply representational degradation: embedding similarity remains robust (> 0.85) across all datasets, confirming that models can represent historical meaning even when generation is unstable.
Finally, we demonstrate that a minimal temporal context prompt reduces historical surprisal by approximately 60%, offering a simple, model-agnostic mitigation. These findings suggest that while historical text imposes a consistent encoding tax, digital libraries can safely deploy LLMs for semantic retrieval tasks, provided that generative applications are carefully adapted. 

---
# LMs as Task-Specific Knowledge Bases: An Interpretability Analysis 

**Authors**: Amit Elhelo, Amir Globerson, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2606.27237)  

**Abstract**: Language models (LMs) capture large amounts of factual knowledge applicable to a wide range of tasks, motivating the view of their parameters as a knowledge base. An important property of knowledge bases is that different queries for the same fact return consistent results, drawing on a single source of truth. We investigate whether LMs satisfy this property through behavioral and mechanistic analyses. Our results suggest that they encode knowledge in a task-specific manner. Behaviorally, facts acquired on one task frequently fail to co-emerge on others during training. Parameter localization experiments suggest a mechanistic explanation, revealing distinct parameter subsets underlying different tasks for the same fact. Finally, we show that chain-of-thought reasoning draws part of its effectiveness from engaging task-specific parameters beyond those tied to the evaluation task. Our findings suggest that what the model knows and how it is asked are intertwined in parameter space, undermining the "knowledge base" analogy and carrying implications for the reliability and controllability of factual knowledge in LMs. 

---
# Bridging Talk and Thought: Understanding Dialogue Dynamics Across Collaborative Problem-Solving Contexts 

**Authors**: Zhengyuan Liu, Stella Xin Yin, Min-Yen Kan, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.27233)  

**Abstract**: We present a conceptual framework for analyzing dialogue in collaborative problem-solving contexts, with an emphasis on the emerging dynamics of human-AI and multi-agent collaboration. As intelligent systems become active agents capable of autonomous reasoning and strategic cooperation, understanding the dialogic interaction during collaborative problem solving is increasingly important for optimizing and evaluating such partnerships. Our framework addresses key limitations in current analytical approaches through a hierarchical two-layer coding scheme that integrates cognitive and non-cognitive problem solving with metacognitive regulatory mechanisms. We demonstrate its effectiveness and generalizability across nine datasets spanning multiple domains, and provide insights into how humans and agents coordinate their knowledge, skills, and efforts to solve complex problems, showing in particular that metacognitive regulation can be an essential discriminator of deeper collaboration. 

---
# CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention 

**Authors**: Sayak Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2606.27229)  

**Abstract**: Recurrent models must forget in order to remember, yet the state of the art decides what to erase without consulting what is stored -- the gate sees only the arriving token, not the memory it is about to modify. This memory-blind gating is one of three coupled defects in the leading delta-rule architecture (GDN-2): the value-axis erase mask wastes parameters at the scale of the value projection, and -- as we prove -- mathematically prevents the WY-form triangular chunk solver that makes recurrent training competitive with Transformers.
We introduce CARVE (Content-Aware Recurrent with Value Efficiency), which resolves all three problems through one principle: erase only on the key axis. This is provably necessary and sufficient for the WY-form solver to remain valid. Within it, CARVE reuses the recurrent output tensor -- already written to GPU memory -- as a free content signal for the erase gate, and replaces the per-value write-gate projection with a single scalar per head. At initialisation CARVE is bit-identical to GDN-2; any quality difference emerges from what the content gate learns.
At 1.3B parameters trained on 100B tokens, CARVE achieves WikiText perplexity 15.72 (minus 0.18 vs. GDN-2, a 4.5-sigma effect), leads every recurrent baseline on nine common-sense reasoning benchmarks, and sets state of the art on every RULER retrieval probe -- at 0.4% throughput overhead, 13% lower peak memory, and 19% fewer parameters. Six formal theorems cover memory capacity, Lyapunov stability, gradient flow, expressivity separation, Pareto-optimal chunk size, and hybrid optimality. 

---
# Compositionality and the lexicon in evolutionary semantics 

**Authors**: Fausto Carcassi  

**Link**: [PDF](https://arxiv.org/pdf/2606.27228)  

**Abstract**: Formal semantics has shown that sentence meanings arise by recursively composing lexical meanings, yet much of the literature on semantic universals models either lexicons with fixed signal structures or holistic composition without interpretable lexical parts. We introduce a framework that integrates this fundamental insight of formal semantics in evolutionary modeling, by allowing lexical meanings and a composition function to co-evolve under pressures for conceptual simplicity and communicative accuracy. We apply this framework to the evolution of quantificational meaning. Analyzing the Pareto frontier, we find that the most well-known semantic universal, conservativity, emerges as an efficient system-wide abstraction. The account is sensitive to syntactic structure and helps reconcile tensions between empirical evidence on quantifier learnability and prior evolutionary models. More broadly, the results demonstrate that the picture of sentential meaning developed in formal semantics can be productively combined with evolutionary modeling. The framework offers a template for studying universals that involve global compression within a grammatical category, semantic specialization of syntactic arguments, and the co-evolution of lexical and compositional meaning. 

---
# Paved with True Intents: Intent-Aware Training Improves LLM Safety Classification Across Training Regimes 

**Authors**: Jeremias Ferrao, Niclas Müller-Hof, Iustin Sîrbu, Traian Rebedea, Yftah Ziser  

**Link**: [PDF](https://arxiv.org/pdf/2606.27210)  

**Abstract**: We argue that safety classifiers should model user intent as an explicit signal between the prompt and the final label. To study this, we introduce AIMS, a human-annotated dataset of 1,724 difficult safety prompts, each paired with an intent description and harm label. We use AIMS to evaluate intent-aware training across supervised fine-tuning, preference learning, reasoning distillation, and reinforcement learning. Despite its size, AIMS enables competitive safety classifiers across training regimes: DPO from model-generated intent errors improves over SFT, and intent-conditioned distillation outperforms reasoning-only distillation in most teacher-student pairs. Most notably, directly rewarding intent faithfulness with GRPO yields the strongest average performance across five external safety benchmarks, while our intent-aware models form the inference latency-F1 Pareto frontier. These results show that faithful intent modeling is a compact, high-quality supervision signal for more robust safety classifiers. 

---
# Syntactic Belief Update as the Driver of Garden Path Processing Difficulty 

**Authors**: Alan Zhou, Miloš Stanojević, John T. Hale  

**Link**: [PDF](https://arxiv.org/pdf/2606.27206)  

**Abstract**: Garden path sentences present a processing difficulty for humans -- the sentence prefix leads the listener towards one interpretation, until the listener hears a critical word that shows that the initial interpretation was wrong. Lexical surprisal, a measure that usually predicts sentence processing difficulty quite well, fails to provide good predictions for garden path sentences.
We propose an alternative that actively predicts a probability distribution over syntactic trees (its syntactic belief) and updates that distribution after each new word. If a processor is led down a garden path, syntactic beliefs will be wrong and will require a large update at the critical word. The magnitude of the update is measured with a generalized Rényi divergence. Crucially, this metric is dependent on lexical items, but is fully independent of the probability of lexical items. This Syntactic Belief Update provides a better fit to the human reading time data on garden path sentences. This suggests a new research direction examining purely non-lexical alternatives to surprisal for psycholinguistics. 

---
# Forecasting With LLMs: Improved Generalization Through Feature Steering 

**Authors**: Humzah Merchant, Bradford Levy  

**Link**: [PDF](https://arxiv.org/pdf/2606.27199)  

**Abstract**: Successful forecasting involves identifying patterns between historical and future states of the world which generalize to future observations. We apply LLMs to a variety of forecasting tasks and inspect their internal states using sparse autoencoders to understand whether they appear to rely on time-specific pieces of knowledge versus generalizable patterns. Our analyses identify features associated with both time-aware reasoning and look-ahead-biased reasoning. We then apply the LLMs to an entirely different domain and intervene on these features. We find that amplifying time-awareness features substantially reduces look-ahead bias on forecasting prompts while preserving general reasoning performance. In contrast, steering the candidate look-ahead-bias features does not produce an effect. These results suggest that interpretable temporal features can be used to causally shift LLMs toward more historically grounded reasoning. 

---
# The Riddle Riddle: Testing Flexible Reasoning in Large Language Models and Humans 

**Authors**: Bella Fascendini, Kathryn McGregor, Max D. Gupta, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2606.27103)  

**Abstract**: Humans flexibly adapt their reasoning strategies to the requirements of a given problem. Large language models (LLMs) have performed well on many cognitive tasks, however, it is unclear whether this accuracy is a result of pattern matching from training data or flexible reasoning. Here, we introduce a novel paradigm to test this question: the riddle riddle paradigm. Riddle riddles are word problems written to mimic popular riddles, but altered so their answers only require literal interpretations. Identifying correct answers requires looking past the structure of each question and flexibly apply different reasoning strategies based on the content. If LLMs respond to surface features, such as form, a riddle-like structure should cause models to use an inventive reasoning strategy even when a literal interpretation suffices. Alternatively, if LLMs reason based on content, they should flexibly switch strategies when appropriate. Across two experiments with nine state-of-the-art LLMs and 100 human participants, we show humans and LLMs fail on this paradigm in opposite directions. LLMs were far more accurate on genuine riddles than on riddle riddles (84.9% vs. 50.7%); whereas humans showed the reverse effect (50.5% vs. 80.5%). Error analysis shows that 90.8% of LLM errors on riddle riddles (the condition where they show diminished performance) were due to inappropriate use of inventive reasoning while only 57.6% of human errors on genuine riddles were due to overextending literal reasoning. Thus, while both groups make mistakes, reasoning mistakes are made more often by LLMs than by humans. Overall, LLMs' strong performance on genuine riddles may reflect memory retrieval rather than flexible strategy selection, and without stimuli designed to elicit this contrast, it becomes easy to conflate LLM-generated outputs that look like reasoning with genuine reasoning. 

---
# Towards Explainable Adjudicative Variance: Quantifying Judicial Discretion via Gated Multi-Task Learning 

**Authors**: Stanisław Sójka, Felix Steffek, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2606.27069)  

**Abstract**: Legal outcome prediction must disentangle objective case facts from adjudicative context. Merit-based rulings rely on factual evidence while technical disposals may hinge on judicial discretion. We propose a Judge-Aware Gated Multi-Task Learning architecture that explicitly models this distinction. We introduce a fine-grained outcome taxonomy to supervise the encoder, enforcing a structural regularization that disentangles distinct semantic pathways. This granular legal curriculum enables our Gated Fusion mechanism to dynamically modulate reliance on judge identity. We evaluate our approach on 13,937 UK Employment Tribunal decisions. We benchmark our design against supervised fine-tuning (SFT) of a Gemma-4 26B-A4B backbone, in which judge identity and the taxonomy are injected as prompt tokens or autoregressive output targets. The two contextual signals compose only weakly when forced through a single autoregressive channel. In contrast, coupling a LoRA-adapted Gemma-4 encoder with our gated architecture defines a new state of the art on this benchmark while requiring an order of magnitude fewer trainable parameters than the generative SFT baselines, with gains concentrated on the most ambiguous and rarest outcome classes. Beyond accuracy, the architecture is interpretable; learned judge embeddings and calibration profiles localize the cases where adjudicative context drives the prediction. These results indicate that, for identity-conditioned classification of legal outcomes, the choice of conditioning interface dominates scale: differentiable structured composition yields more accurate, more parameter-efficient models than prompt-based composition over a substantially larger backbone. 

---
# NuclearQAv2: A Structured Benchmark for Evaluating Domain-Science Competence in Large Language Models 

**Authors**: Henry Shaowu Yuchi, Michal Kucer, Benjamin H. Sims, Selma Peterson, Emily Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2606.27047)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance across a wide range of tasks, but ensuring their reliability in highly technical domains remains a significant challenge. In nuclear engineering, problem solving often requires not only factual knowledge but also quantitative reasoning and conceptual understanding. To address the need for systematic evaluation in this domain, we introduce NuclearQAv2, a benchmark for assessing LLMs on nuclear engineering knowledge. The benchmark comprises approximately 1,240 question-answer pairs spanning three categories: boolean, numeric, and verbal. NuclearQAv2 is constructed using a hybrid pipeline that combines expert-authored questions, existing datasets, and LLM-assisted generation from domain-specific technical corpora. By leveraging structured prompting for both automated question generation and response evaluation, the proposed framework enables scalable benchmark construction and evaluation. We evaluate a diverse set of LLMs using NuclearQAv2 and observe substantial performance differences across task types. While the models generally perform well on factual questions, quantitative reasoning and conceptual understanding remain considerably more challenging. These results highlight the importance of multi-faceted evaluation frameworks and establish NuclearQAv2 as a scalable benchmark for assessing LLM capabilities in technical domains. 

---
# Improving General Role-Playing Agents via Psychology-Grounded Reasoning and Role-Aware Policy Optimization 

**Authors**: Zhenhua Xu, Dongsheng Chen, Jian Li, Yitong Lin, Zhebo Wang, Jiafu Wu, Yizhang Jin, Chengjie Wang, Meng Han, Yabiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27025)  

**Abstract**: Building general-purpose role-playing agents that faithfully portray any character from a natural-language profile remains challenging. The dominant paradigm -- supervised fine-tuning -- encourages behavioral mimicry without deep, human-like internal thought processes, resulting in poor out-of-distribution generalization. Therefore, we propose \textbf{Psy-CoT}, a psychology-grounded chain-of-thought framework that decomposes pre-response reasoning into three role-specific steps -- \emph{Interaction Perception}, \emph{Psychological Empathy}, and \emph{Logical Construction} -- so that the model \emph{thinks dynamically} from the profile rather than merely mimicking surface patterns. While structured reasoning provides a foundation, it alone is insufficient; reinforcement learning is essential to further align the model with character fidelity. However, we observe that under LLM-based reward models, both generic phrases that hack the reward model and genuinely role-specific phrases receive identical gradient signals -- this hacking accumulates over training, misleading the model into treating both as equally optimal choices. To address this, we propose \textbf{Role-Aware Policy Optimization (RAPO)}, which uses profile--token mutual information to weight gradients asymmetrically -- amplifying role-specific tokens under positive advantage while attenuating them under negative advantage. Experiments on CoSER, CharacterBench, and CharacterEval demonstrate that Psy-CoT outperforms existing role-playing CoT methods, and RAPO consistently surpasses GRPO across multiple model scales. 

---
# MinGram: A Minimalist Unigram Tokenizer with High Compression and Competitive Morphological Alignment 

**Authors**: Sander Land  

**Link**: [PDF](https://arxiv.org/pdf/2606.27019)  

**Abstract**: The Unigram tokenizer uses an elegant representation which makes it straightforward to edit vocabularies, but its training is comparatively heavy and complex. We introduce MinGram (Minimalist Unigram), which keeps the token-list representation but simplifies training using a BPE-derived seed vocabulary, Hard EM on a minimum-token path, and a single flat score-pruning step. This removes the suffix array, the forward-backward pass, and the iterative prune loop, leaving a procedure that requires little beyond tokenizer inference itself. By making token count the primary objective and using a Unigram score only as a tiebreak, MinGram keeps the compression of pure token-count methods while retaining much of the morphological alignment and downstream quality of probabilistic ones. Across six languages, MinGram compresses better than both BPE and standard Unigram, and a compression-oriented variant matches the strongest token-count compressors while retaining substantially higher morphological alignment. In controlled downstream language-model training, Unigram-family tokenizers, with MinGram among the best, consistently beat BPE in bits-per-byte. 

---
# Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs 

**Authors**: Sinie van der Ben, Raphaël Baur, Yannick Metz, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2606.26987)  

**Abstract**: Recent work identified emotion vectors in Claude Sonnet 4.5, which are internal representations that encode emotion concepts, causally influence behavior, and exhibit geometry mirroring human psychological structure. We test the generality of these findings in two open-weight models, Apertus-8B-Instruct-2509 and Gemma-4-E4B-it, extracting emotion contrast vectors across all layers, using two model-generated corpora. We recover valence geometry for both models, with peak PC1--valence correlations of $r = 0.76$ and $r = 0.83$, approaching the $r = 0.81$ reported for this http URL replication, we observe notable differences in how valence representations emerge across model depth. In Gemma-4-E4B-it, valence is strongly encoded in early layers but collapses towards later layers, whereas Apertus-8B-Instruct-2509 exhibits the opposite pattern, with valence representations absent in early layers, but emerging at mid depths. Arousal encoding, in contrast, is sensitive to the extraction corpus: both models show stronger PC2--arousal alignment with Gemma-generated stories ($r$ up to $0.45$) than Apertus-generated ones ($r \leq 0.21$), suggesting arousal-relevant cues are unevenly distributed across generated corpora. We open-source our experiment code and dataset for reproducible investigation of emotion representations across language model architectures. 

---
# ReaORE: Reasoning-Guided Progressive Open Relation Extraction Empowered by Large Reasoning Models 

**Authors**: Xin Lin, Liang Zhang, Guoqi Ma, Hongyao Tu, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2606.26986)  

**Abstract**: Open Relation Extraction (OpenRE) requires a model to extract unseen relations between head and tail entities from unstructured text for real-world applications. The core challenge of OpenRE lies in achieving reliable generalization to unseen relation types. Current OpenRE approaches either employ clustering techniques, which cannot generate relation labels and suffer from poor generalization, or rely on direct relation label generation via Large Language Models (LLMs), which lack sufficient discriminative capacity to distinguish easily confused relations. To address these limitations, we propose Reasoning-guided progressive OpenRE (ReaORE), a framework for performing relation extraction through coarse-to-fine relation reasoning. Specifically, ReaORE consists of two key stages: (i) relation filtering, which reasons over multiple aspects to understand relations and instances, yielding an initial relation set, and further supplements and filters relations via embedding-based similarity to ensure the target relation is included; (ii) relation prediction, which aims to predict the target relations from the above set via fine-grained comparative reasoning to better distinguish easily confused relations. Extensive experiments on two widely used OpenRE datasets demonstrate that ReaORE outperforms existing baselines. 

---
# Auditing Framing-Sensitive Behavioral Instability in Large Language Models for Mental Health Interactions 

**Authors**: Abla Bedoui, Ashley L. Greene, Mohammed Cherkaoui  

**Link**: [PDF](https://arxiv.org/pdf/2606.26982)  

**Abstract**: Large language models (LLMs) are increasingly being integrated into mental health support tools and other psychologically sensitive conversational applications. In such settings, behavioral stability and consistency are important for trustworthy human-AI interaction. However, semantically similar concerns can be presented through different contextual framings, potentially eliciting different model responses. Such framing-sensitive variability may challenge user expectations regarding system behavior and complicate the assessment of AI reliability. While prior studies have primarily examined such effects at the behavioral level, less is known about how framing-related variation is reflected in the internal representations of aligned language models. In this work, we investigate these effects using controlled matched prompts spanning multiple contextual framing conditions across several instruction-tuned model families. Across architectures, framing systematically alters interpretive response tendencies. Layer-wise probing analyses show that behavior-associated information remains decodable throughout transformer depth, with architecture-dependent variation in decoding strength. Moreover, held-out framing probes remained consistently above chance across architectures despite strong lexical baselines. Activation steering experiments further suggest that framing-associated representational directions can partially modulate downstream behavioral outcomes. Finally, these findings indicate that robustness to contextual variation may represent an important consideration when evaluating the consistency and trustworthiness of conversational AI systems deployed in mental-health-oriented interactions. 

---
# RedVox: Safety and Fairness Gaps in Speech Models Across Languages 

**Authors**: Beatrice Savoldi, Sara Papi, Wafa Aissa, Matteo Negri, Luisa Bentivogli  

**Link**: [PDF](https://arxiv.org/pdf/2606.26968)  

**Abstract**: Speech-capable models are increasingly deployed in real-world applications across languages. Yet their safety and fairness beyond English settings and under naturalistic conditions remain understudied. We survey safety reporting practices across state-of-the-art speech model releases, finding that only 8% document any multilingual analysis. To address this gap, we introduce RedVox, a multilingual safety and fairness benchmark for audio and speech built on real voices, covering unsafe and unfair stereotypical requests across five languages (English, French, Italian, Spanish, and German). Evaluating eight state-of-the-art models, we find that vulnerabilities persist even under non-adversarial conditions, worsen in non-English languages, and are amplified when the request comes from a spoken input. Finally, by surveying the participants who contributed to RedVox, we document the unique personal and privacy challenges of collecting speech data with human participants, pointing to broader sociotechnical challenges in naturalistic speech safety research. 

---
# Term-Centric Hierarchy Induction from Heterogeneous Corpora 

**Authors**: Elena Senger, Yuri Campbell, Jan-Peter Bergmann, Rob van der Goot, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2606.26963)  

**Abstract**: Organizing knowledge from diverse text sources into interpretable hierarchies is crucial for tasks such as policy analysis, innovation monitoring, and exploratory domain mapping. Existing taxonomy induction methods typically rely on document-level representations that capture entire documents rather than the specific domain concepts relevant for knowledge organization, limiting their ability to generalize across heterogeneous sources. We propose a term-centric framework for inducing hierarchical taxonomies from heterogeneous corpora that scales to massive document collections. Our approach maps documents from diverse sources into a shared representation space using automatic term extraction, enabling robust cross-source alignment. Based on these representations, we construct interpretable hierarchies that integrate domain priors with datadriven clustering. Experiments on a novel English and German multi-source benchmark of over one million documents demonstrate that our method improves cross-source coherence and hierarchy quality over text- and summarybased baselines. A case study on German regional innovation analysis further demonstrates its practical utility for technology landscape mapping. 

---
# GAVEL: Grounded Caption Error Verification and Localization 

**Authors**: Zixian Gao, Atsushi Hashimoto, Kuniaki Saito  

**Link**: [PDF](https://arxiv.org/pdf/2606.26923)  

**Abstract**: Vision-language models (VLMs) often produce hallucinated or inconsistent outputs, where text and images are not properly aligned. Addressing this issue requires not only detecting misalignment but also explaining the discrepancy and localizing its visual evidence. We introduce GAVEL (Grounded Caption Error Verification and Localization), a task that jointly addresses verification, explanation, and localization for image-text pairs. To support systematic evaluation, we also present a corresponding dataset and benchmark. We further train a supervised baseline on the human-annotated training split to assess whether GAVEL provides learnable supervision for these abilities. Experiments show that even strong closed-source models struggle on GAVEL, while the supervised baseline yields consistent improvements across grounding and explanation metrics. 

---
# SamaVaani: Auditing and Debiasing Multilingual Clinical ASR for Indian Languages 

**Authors**: Subham Kumar, Prakrithi Shivaprakash, Abhishek Manoharan, Astut Kurariya, Diptadhi Mukherjee, Prabhat Chand, Pratima Murthy, Koustav Rudra, Lekhansh Shukla, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.26901)  

**Abstract**: Automatic Speech Recognition (ASR) is increasingly used to document clinical encounters, yet its reliability in multilingual and demographically diverse Indian healthcare context remains largely unknown. In this study, we first conduct the systematic audit of ASR performance on real-world psychiatric interview data spanning Kannada, Hindi and Indian English, comparing eight state-of-the-art models including IndicWhisper, WhisperLargeV3, Sarvam, GoogleS2T, Gemma3n, OmniLingual, Vaani, and Gemini. Our results reveal substantial variability across models and languages, with some systems performing competitively in Indian English but failing in regional speech. We further fine-tune two of the best performing opensource models, i.e., Gemma3n and OmniLingual, using various methods. With this, we uncover systematic performance gaps tied to speaker role and gender, raising concerns about equitable deployment in clinical settings, which are further mitigated by fairness-aware fine-tuning. To this end, we propose SamaVaani, a unified debiasing technique that simultaneously improves ASR performance and improves fairness across demographic groups. 

---
# Heterogeneous Neural Predictivity from Language Models During Naturalistic Comprehension 

**Authors**: Xiao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2606.26880)  

**Abstract**: Language-model representations provide structured, high-dimensional annotations of naturalistic language stimuli and can serve as informative neural predictors during comprehension. We analyzed locked derived data from Brain Treebank, MEG-MASC, and Podcast ECoG with eight frozen language models, blocked encoding models, and matched temporal, nuisance, and representation-capacity controls. Positive held-out prediction and gains over low-level baselines were widespread in source-level summaries. Across Brain Treebank and Podcast ECoG, 67 of 432 evaluable rows met a controlled predictive-only criterion, and model-side feature ablations changed prediction scores in most evaluable source rows. Brain-derived, timing-linked, acoustic, and implanted-signal controls confirmed component-level sensitivity of the analysis pipeline. These findings show that language-model-derived quantities can annotate neural activity during natural speech and text comprehension. Participant-level matched-control advantages were localized rather than uniform, response-profile and feature-specificity contrasts bounded representational or computational interpretations, and complete co-indexed integrated interpretation will require future jointly indexed coverage. Together, the analyses identify language-model features as useful neural predictors and separate predictive usefulness from claims about shared neural organization or language-processing computations. 

---
# Information-Aware KV Cache Compression for Long Reasoning 

**Authors**: Jushi Kai, Zhuiri Xiao, Alexandra Birch, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.26875)  

**Abstract**: Reasoning capability has advanced rapidly in large language models (LLMs), leading to an increasing size of key-value (KV) cache in both prefilling and decoding stages. Existing KV cache compression methods mainly rely on attention weights to estimate token importance. While attention effectively captures contextual relevance, it overlooks complementary information-theoretic signals related to predictive uncertainty and token informativeness. In this paper, we revisit token importance from a forward-looking perspective and introduce \textit{Forward Influence}, a metric that measures how compressed tokens affect future contexts. Our analysis reveals that tokens selected by attention scores mainly influence nearby contexts, whereas tokens associated with high predictive uncertainty exhibit substantially stronger influence on distant future contexts. Based on the observation, we propose \textbf{InfoKV}, an entropy-aware KV cache compression framework that incorporates information-theoretic signals. It combines token-level predictive uncertainty with layer-wise representation evolution and integrates the resulting entropy scores with attention scores during reasoning. Experiments on long-context reasoning benchmarks with Llama-3.1, Llama-3.2, and DeepSeek-R1 demonstrate that InfoKV consistently outperforms existing attention-based KV compression methods in both long prefilling and decoding scenarios. 

---
# Cascaded Multi-Granularity Pruning for On-Device LLM Inference in Industrial IoT 

**Authors**: Jinghan Wang, Yanjun Chen, Wei Zhang, Xiaotong Huang, Tianchen Liu, Gaoliang Peng  

**Link**: [PDF](https://arxiv.org/pdf/2606.26861)  

**Abstract**: Deploying large language models (LLMs) on Industrial Internet of Things (IIoT) edge devices demands extreme compression, yet existing structured pruning methods collapse at high compression ratios due to one-shot importance estimation, and their cross-architecture behavior remains unpredictable. This article presents a cascaded multi-granularity pruning framework that removes layers, attention heads, and feed-forward channels in coarse-to-fine order, with lightweight low-rank recovery between stages to re-estimate component importance. An information-theoretic analysis motivates this ordering, and the Structural Independence Assumption (SIA) is formalized as a checkable condition predicting whether per-component pruning criteria are reliable for a given architecture: Multi-Head Attention (MHA)+GELU designs satisfy the SIA, whereas Grouped Query Attention (GQA)+SwiGLU designs violate it. On bearing fault diagnosis spanning 88M to 6.25B-parameter models, the framework extends achievable compression to 13.8 times on MHA+GELU architectures with 83.82% accuracy (+3.70 percentage points (pp) over the strongest baseline), while exposing a ~74pp accuracy collapse on GQA+SwiGLU architectures that violate the SIA. Deployed on an industrial slewing bearing fault diagnosis platform with NVIDIA DGX Spark, compressed models reduce inference latency by up to 67.2% and peak memory by 62.5%, demonstrating viability for IIoT edge inference. 

---
# FBK's Long-form SpeechLLMs for IWSLT 2026 Instruction Following 

**Authors**: Zhihang Xie, Marco Gaido, Sara Papi, Matteo Negri, Luisa Bentivogli  

**Link**: [PDF](https://arxiv.org/pdf/2606.26819)  

**Abstract**: This paper describes our submission to the IWSLT 2026 Instruction Following shared task. SpeechLLMs are developed for both short-form and long-form speech instruction following under constrained settings. For the short track, strong performance is achieved on MCIF, with a SIFS score of 2.0708. For the long track, three speech segmentation methods are explored, and the HIFS score is introduced to account for unstable long-form generation. Experimental results show that fixed 30-second segmentation provides the most robust long-form performance, achieving the highest HIFS score of 2.0663. Further analysis shows that hallucination mainly manifests as repetitive insertions in generated outputs, substantially affecting ASR and SSUM, while short-form capabilities are largely retained after long-form extension. 

---
# From Vajrayana Tara to Bengali Baul: A Computational Study of Lexical Transmission Across Buddhist, Shakta, and Vaishnava Traditions in Bengal 

**Authors**: Joy Bose  

**Link**: [PDF](https://arxiv.org/pdf/2606.26803)  

**Abstract**: We present a computational corpus study of vocabulary relationships across eight tradition layers of Bengali and Sanskrit devotional literature spanning the 8th to 19th centuries, encompassing Buddhist Vajrayana, Shakta Tantra, Vaishnava, and Baul traditions. Using a corpus of 75 texts and TF-IDF character n-gram vectorization with cosine similarity analysis, we address the historically argued but previously unquantified claim that Buddhist Vajrayana vocabulary survived the collapse of the Pala monasteries and was absorbed into the Shakta Tantra tradition of Bengal. The central finding is a specificity result: the Gitagovinda (Vaishnava Sanskrit, 12th century) has zero cosine similarity to Shakta Kali texts, while Bridge Tara texts (Buddhist-Shakta transitional, same century, same language) have cosine similarity 0.54 to Shakta Kali. This 8.5-fold contrast between two Sanskrit traditions from the same century demonstrates that the Buddhist-Shakta vocabulary overlap is not a generic property of Sanskrit devotional literature but is specific to the Buddhist-Shakta transmission chain. Three Brihannilatantra Tara texts show Shakta-to-Buddhist vocabulary ratios of 2.0 to 4.0, constituting measurable evidence of lexical transition within that chain. Ramprasad Sen's 18th-century Bengali Kali songs preserve Buddhist vocabulary residue including 56 occurrences of Tara alongside 103 occurrences of Kali. The Vaishnava Bengali tradition contributes a parallel chain to modern Baul vocabulary (similarity 0.29), slightly weaker than the Buddhist Sahajiya chain via Charyapada (0.31). These results provide the first quantitative multi-tradition corroboration of historically argued Buddhist-Shakta syncretism in Bengal. 

---
# OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning 

**Authors**: Shuo Yang, Jinyang Wu, Zhengxi Lu, Yuhao Shen, Fan Zhang, Lang Feng, Shuai Zhang, Haoran Luo, Zheng Lian, Zhengqi Wen, Jianhua Tao  

**Link**: [PDF](https://arxiv.org/pdf/2606.26790)  

**Abstract**: Outcome-based reinforcement learning provides a stable optimization backbone for language agents, but its sparse trajectory-level rewards provide little guidance on which intermediate decisions should be reinforced or suppressed. On-policy self-distillation offers dense token-level supervision, yet existing skill-conditioned variants often rely on external skill memories or retrieved privileged context, which are costly to maintain and can be mismatched with the state distribution induced by the current policy in multi-turn interaction. We propose \textbf{OPID} (\textbf{O}n-\textbf{P}olicy Sk\textbf{i}ll \textbf{D}istillation), a framework that extracts skill supervision directly from completed on-policy trajectories. OPID represents trajectory hindsight as hierarchical skills: episode-level skills capture global workflows or failure-avoidance rules, while step-level skills capture local decision knowledge at critical timesteps. A critical-first routing mechanism uses step-level skills when critical decisions are identified and falls back to episode-level skills as default guidance otherwise. The selected skill is injected into the interaction history, allowing the old policy to re-score the same sampled response under both original and skill-augmented contexts. The resulting log-probability shift yields a token-level self-distillation advantage, which is combined with the outcome advantage for policy optimization. OPID thus preserves RL as the primary training objective while introducing dense, distribution-matched hindsight supervision. Experiments on ALFWorld, WebShop and Search-based QA demonstrate that OPID generally improves agent performance, sample efficiency, and robustness over outcome-only RL and existing skill-distillation baselines. Our code is available at this https URL. 

---
# Evaluation Pitfalls and Challenges in Multimedia Event Extraction 

**Authors**: Philipp Seeberger, Steffen Freisinger, Tobias Bocklet, Korbinian Riedhammer  

**Link**: [PDF](https://arxiv.org/pdf/2606.26775)  

**Abstract**: Multimedia event extraction aims to jointly identify events and their arguments across multiple modalities, such as text and images, to support more comprehensive event understanding. While recent work reports steady and substantial progress, the reliability and comparability of these results critically depend on consistent and rigorous evaluation. In this work, we present the first systematic analysis of evaluation pitfalls in multimedia event extraction and identify three major sources of issues: inconsistent data processing, inconsistent task assumptions, and overly relaxed evaluation settings. We demonstrate, through a series of controlled experiments under a strict evaluation framework, that minor evaluation choices can cause large performance variations and lead to overestimation of a model's ability to ground real-world events across modalities. Our findings highlight the need for comparable evaluation standards and encourage a shift toward more rigorous evaluation in multimedia event extraction. 

---
# ConvMemory v3: A Validity Context Layer for Conversational Memory via Target-Conditioned Relation Verification 

**Authors**: Taiheng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2606.26753)  

**Abstract**: Conversational memory retrieval optimizes relevance, yet a retrieved memory can be relevant and simultaneously outdated: a later turn updates, corrects, or supersedes it. ConvMemory v3 adds a validity context layer that detects and surfaces this update evidence through target-conditioned relation verification, sitting after the v1/v2 retrieval path. The core mechanism is a dual-evidence gate that conditions a relation judgment on the specific target proposition, scoring a (target, source) pair through the product of a MiniLM slot head and a DeBERTa-v3 slot head and gating it by conservative event/operation evidence. On a synthetic multi-hop validity benchmark the gate reaches 90.12% +/- 1.73 accuracy; through a real-data feedback loop that mines failure patterns but trains on synthetic pairs only, the verifier transfers to Memora role binding with zero target-side labels, reaching 98.8% +/- 0.9 group-all-correct. The deployed layer preserves retrieval by default: a context mode attaches structured validity metadata while keeping the candidate set and rank order fixed, and a query-conditioned demote mode is an explicit opt-in for dense current-state workloads, where it raises current-active H@1 from a never-demote baseline of 45.1% to 95.7% +/- 1.2 while protecting non-superseded memories at 99.4% recall. Six machine-verifiable safety contracts pin the layer's behavior. Multi-hop graph propagation is validated as a mechanism; fully automatic construction of strict prerequisite edges is characterized as a boundary, since strict necessity requires counterfactual world knowledge. This report extends ConvMemory v1 (arXiv:2605.28062) and v2 (arXiv:2606.10842). 

---
# Beyond Logical Forms: LLM-Extracted Patterns for Fallacy Classification 

**Authors**: Eleni Papadopulos, Firoj Alam, Giovanni Da San Martino  

**Link**: [PDF](https://arxiv.org/pdf/2606.26698)  

**Abstract**: In today's fast-paced information era, logical fallacies, defined as defective patterns of reasoning, inevitably contribute to the growth of information disorder. However, often fallacies appear in nuanced forms that complicate automated classification. In this study, we investigate whether merging abstract logical structures with context-level linguistic cues proves beneficial for fallacy classification, developing a framework that inductively extracts such patterns from fallacious examples and their explanations using Large Language Models (LLMs). We evaluate the impact of these patterns across different LLMs and experimental zero- and one-shot configurations, showing statistically significant improvements over zero-shot baselines and outperforming competing approaches. Cross-dataset experiments validate generalization, establishing data-driven pattern extraction as an effective method for generating logical representations. 

---
# SocialPersona: Benchmarking Personalized Profiling and Response with Multimodal Social-Media Context 

**Authors**: Qinkai Zhang, Yanyan Zhao, Xin Lu, Yulin Hu, Pengtao Han, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2606.26654)  

**Abstract**: Personalized language-model assistants are often evaluated through a memory lens: can a model recall preferences users have explicitly stated in dialogue? More comprehensive personalization demands a harder capability -- inferring what users care about from the multimodal traces they naturally leave behind. We introduce SocialPersona, a benchmark for evaluating whether multimodal large language models (MLLMs) can recover revealed preferences from longitudinal social-media timelines and use them in dialogue. Built from longitudinal timelines of 171 everyday, non-promotional social-media users, SocialPersona contains text, images, timestamps, and 2,597 human-verified preference tags across seven interest domains, separating stable interests from recent interests. It supports two tasks: constructing structured user profiles from multimodal context and generating responses aligned with inferred profiles. Experiments with proprietary and open-weight MLLMs show that models can identify broad interest domains, yet their performance drops on fine-grained and recent interests and degrades further when inferred profiles must be used to personalize dialogue. Together with evidence that text and images provide complementary preference signals, these results indicate that robust cross-modal, long-horizon user modeling remains a key challenge, and that SocialPersona can help measure and advance progress toward assistants that infer and act on revealed preferences. 

---
# CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs 

**Authors**: Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, Anbang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2606.26650)  

**Abstract**: In this paper, we present CAT-Q, Cost-efficient and Accurate Ternary Quantization, for compressing and accelerating LLMs. Unlike existing state-of-the-art ternary quantization methods that rely on data-intensive and costly quantization-aware training to mitigate severe performance degradation, CAT-Q is a simple yet effective post-training quantization scheme that is readily applicable to LLMs with diverse architectures and model sizes. It has two key components, learnable modulation (LM) and softened ternarization (ST), which are coupled from an optimization perspective. LM leverages a composition of learnable factors to modulate the distribution of pre-trained high-precision weights and the ternary threshold, making them less sensitive to ternarization. ST further introduces a differentiable transition function to guide the ternarization process toward stable convergence. We show that, for pre-trained LLMs with 1.7B to 8B parameters, CAT-Q can efficiently quantize them into ternary models using only 512 calibration samples, while achieving superior performance than the seminal BitNet 1.58-bit v1 and v2 families (with 1.3B to 7B parameters) trained with 100B tokens, yielding about a 100,000X reduction in training tokens. Moreover, we show for the first time that CAT-Q can quantize much larger pre-trained LLMs having 14B to 235B parameters into leading ternary models within just 8 to 60 hours on 8 A100-80GB GPUs. Code is available at this https URL. 

---
# Closing the Quality Gap in Low-Resource Text-to-Speech: LoRA Fine-Tuning of VoxCPM2 for Khmer and Korean 

**Authors**: Phannet Pov, Sovandara Chhoun, Hyun Woo Park, Wan-Sup Cho, Saksonita Khoeurn  

**Link**: [PDF](https://arxiv.org/pdf/2606.26618)  

**Abstract**: Large pretrained text-to-speech (TTS) models sound almost human for well-resourced languages, but much worse for languages that are rare in their training data. We study this quality gap for Khmer and Korean using VoxCPM2, a 2.4B-parameter, tokenizer-free TTS model that joins a MiniCPM-4 language-model backbone with a flow-matching diffusion decoder. We build one shared, language-tagged corpus of about 26 hours and adapt VoxCPM2 with a single Low-Rank Adaptation (LoRA) adapter, trained on both languages at once and added to both the language model and the decoder. The adapter is zero-initialized, so training starts exactly at the original (zero-shot) model. In native-speaker listening tests, the Khmer Mean Opinion Score (MOS) rises from 3.85 to 4.23 with the best adapter (rank 64), a highly significant gain (paired Wilcoxon test, p<0.001), while training only 0.19 to 3.03 percent of the parameters. The automatic loss and the human ratings, however, disagree on the best rank: validation loss is lowest at rank 128, yet MOS peaks at rank 64. The same adapter brings no gain for Korean, a language the base model already handles well, and at a high rank it even degrades quality. Adaptation therefore helps mainly where the base model is genuinely weak. 

---
# Zero-shot Tweet-Level Stance Detection Enhanced by External Knowledge and Reflective Chain-of-Thought Reasoning 

**Authors**: Yiju Huang, Wenxian Wang, Lijun Zhou, Rui Tang, Xiao Lan, Tao Zhang, Haizhou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26571)  

**Abstract**: Zero-shot tweet-level stance detection confronts two primary challenges: (1) mitigating the context sparsity inherent in short texts, and (2) establishing the relevance between implicit targets and textual content. While existing methods primarily focus on incorporating external knowledge, they neglect the intrinsic semantic cues embedded within key intra-textual entities. Furthermore, current models exhibit limited capability in determining the relevance of unseen targets to the given text, thereby struggling to differentiate between "neutral" and "irrelevant" stance labels. To address these issues, we first construct a four-class, multi-topic Japanese tweet dataset. To our knowledge, this is the first Japanese tweet-level dataset for stance detection. We then propose KIRP, a zero-shot stance detection framework. It integrates external knowledge with entity reorganization for data augmentation and employs prompt chaining for reasoning. Specifically, the framework incorporates knowledge graphs to supplement and reorganize key textual entities, while reflective Chain-of-Thought (CoT) reasoning extracts and validates implicit targets. To better distinguish "neutral" from "irrelevant" labels, we adopt stance-aware contrastive learning to capture discriminative features and design a three-layer iterative prototype network for fine-grained classification. Experimental results on SemEval-2016, WT-WT, and KIRP-D show that KIRP achieves state-of-the-art performance. KIRP obtains F1 scores of 84.05% (three-class) on SemEval-2016, and 84.99% and 79.18% (four-class) on WT-WT and KIRP-D, respectively. 

---
# Erase-then-Delta Attention: Decoupling Erase and Write Addresses in Delta-Rule Linear Attention 

**Authors**: Xiao Li, Chengruidong Zhang, Hao Luo, Xi Lin, Zekun Wang, Zihan Qiu, Yunfei Mao, Langshi Chen, Man Yuan, Minmin Sun, Huiqiang Jiang, Siqi Zhang, Rui Men, Wei Hu, Gong Cheng, Bo Zheng, Dayiheng Liu, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.26560)  

**Abstract**: Delta-rule linear attention improves recurrent memory updates by correcting what is already stored at the current write address before writing new content. However, the active correction is still anchored to that same write address. As a result, stale information stored at a different address cannot be actively removed before new content is written elsewhere. We propose Erase-then-Delta Attention (EDA), a memory update rule that decouples where to erase from where to write. The key insight is that recurrent memory models should not only correct the current write, but also selectively suppress outdated memory at an independently chosen address. Concretely, our method first applies a targeted erase step along a learned erase direction, and then performs the standard delta-style corrective write along the current write direction. This preserves the corrective behavior of delta-rule updates while expanding their memory-management capacity. Language-model pretraining experiments across dense 2.5B and MoE 25B-A2.8B model families show that EDA performs best in both settings. The gain persists after 80B-token long-context midtraining of the MoE models, where EDA also performs best in long-context evaluations from 4k to 128k contexts. A compact update analysis and memory-state probes suggest why: EDA keeps the delta-rule corrective write intact while allocating an additional cleanup path most strongly when passive decay is weak. These results suggest that recurrent memory models should decide not only what to write, but also what stale information to erase and where. 

---
# \textsc{DiARC}: Distinguishing Positive and Negative Samples Helps Improving ARC-like Reasoning Ability of Large Language Models 

**Authors**: Yuxuan Yang, Feiyang Li, Yile Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26530)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC;~\citealp{chollet2019measure}) contains tasks that require summarizing patterns from limited grid samples and predicting output grids. Recently, many large language model based approaches have attempted to transform it into a text-based reasoning task. However, methods based on open-source models have generally yielded unsatisfactory results, while those relying on closed-source models are too costly. Current efforts mainly focus on data augmentation, constructing ARC-like data for more comprehensive supervised fine-tuning. In this work, we argue that solving ARC-like problems requires not only \textit{positive} sample supervision but also the ability to improve model reasoning by distinguishing \textit{negative} samples. To this end, we draw on the idea of preference alignment and propose \textsc{DiARC}, a method that constructs preference pairs to enable the model to distinguish between them. Specifically, we propose three ways to construct negative samples, including output-level visual transformations, DSL-level rule inversion, and task-specific rule editing. The resulting negative samples provide informative near-miss alternatives while keeping the observed demonstrations unchanged. Experimental results across multiple ARC-like benchmarks show that \textsc{DiARC} consistently improves performance over baseline models. The code is released at this https URL. 

---
# The Inattentional Gap: Task-Conditioned Language and Vision Models Omit the Safety-Critical Signals They Can Otherwise Report 

**Authors**: Kwan Soo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2606.26529)  

**Abstract**: AI safety is evaluated by how reliably a model detects the hazards it is told to find, yet accidents often arise from the hazard no one specified. We show that conditioning a language or vision model on a narrow task suppresses its reporting of co-present, safety-critical signals it can otherwise report, a machine analogue of human inattentional blindness arising from a different mechanism. Across radiology and driving text scenarios and chest-radiograph vision tasks, suppression appeared in every model tested, did not diminish with scale, persisted in a reasoning model, and varied more by model family than by size, while the same models reported these signals at substantially higher rates when unconstrained. We name this dissociation the Inattentional Gap and argue that it decouples measured benchmark safety from real-world safety: a system can score near-perfectly on the hazards an evaluation specifies while remaining blind to those that cause harm. 

---
# Assessing Post-Reform Changes in Risk Disclosure Quality with a Multidimensional Text Analysis Approach 

**Authors**: Nobuhiro Aikawa, Mitsuo Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2606.26522)  

**Abstract**: While corporate narrative disclosures provide crucial information to capital markets, comprehensively evaluating their qualitative changes over time remains challenging. Narrative text is inherently multidimensional, meaning that an improvement in one textual dimension often occurs alongside changes in others. To capture these underlying dynamics, we propose a longitudinal text analysis approach combining Japanese-language NLP metric extraction with paired testing, shift function analysis, and inter-metric correlation. Our framework extends prior indicator sets by incorporating a cross-section relevance indicator to measure topical alignment between risk disclosures and management strategies. Applying this approach to evaluate Japan's 2019 disclosure reforms, we analyze 19,770 firm-year observations over a 10-year period (FY2015-FY2024). The joint analysis reveals complex shifts in disclosure patterns that are frequently masked by conventional single-indicator methods. Specifically, we find that while disclosure volume increased substantially, it was accompanied by a decline in readability. Furthermore, although the overall information structure improved, specific descriptive quality stagnated, and the degree of adaptation varied across market segments. 

---
# Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors for AI Agents over Evolving Knowledge 

**Authors**: Neeraj Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2606.26511)  

**Abstract**: Retrieval-augmented generation (RAG) gives agents access to accumulated knowledge, but has no model of time. When a fact changes (e.g., a function is renamed or API restructured), RAG retrieves both the stale and current value with near-identical embedding similarity. The agent then either abstains or serves the superseded fact. We show this is a structural problem: on a calibrated dataset, cosine similarity distinguishes a contradicted fact from a duplicated one with AUROC 0.59 (near chance), as contradictions are often more embedding-similar to the original than rephrased duplicates.
We present MemStrata, a retrieval memory maintaining temporal validity. It stores facts like RAG, preserving static recall, but when a fact's value is contradicted, a deterministic (subject, relation, object) supersession rule retires the stale value in a bi-temporal ledger - with no similarity threshold and no LLM call. Across six benchmarks run locally with a 7B model, MemStrata ties RAG on static knowledge and reaches 0.95-1.00 accuracy on evolving knowledge (where RAG reaches 0.20-0.47). The central result is the stale-fact-error rate: when required to answer, RAG serves superseded values 15-40% of the time; MemStrata drives this to ~0%, a failure class RAG cannot avoid. MemStrata achieves this at retrieval latency (~2.1s) versus ~16-18s for LLM-reranking baselines. We release the harness, datasets, and a marker-free evaluation protocol for memory under knowledge evolution. 

---
# Nemotron-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context 

**Authors**: Fitsum Reda, John Kamalu, Roger Waleffe, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2606.26493)  

**Abstract**: Diffusion language models offer a promising alternative to autoregressive models due to their potential for parallel and iterative generation. However, existing approaches use a single network for both context representation and iterative denoising, forcing one model to serve both roles and limiting its capacity for either role. We propose TwoTower, a block-wise autoregressive diffusion model that decouples these roles into two towers: a frozen AR context tower that causally processes clean tokens, and a trainable diffusion denoiser tower with bidirectional block attention that refines noisy blocks via cross-attention to the context. Built on Nemotron-3-Nano-30B-A3B, an open-weight 30B hybrid Mamba-Transformer MoE model, and trained on approximately 2.1T tokens, Nemotron-TwoTower retains 98.7% of the autoregressive baseline's quality while offering 2.42X higher wall-clock generation throughput. We release the code and model weights at this https URL. 

---
# Comparing BERT Sentence-Pair Classification and Few-Shot LLM Prompting for Detecting Threat and Solution Framing in German Climate News 

**Authors**: Raven Adam, David Maier, Marie Kogler  

**Link**: [PDF](https://arxiv.org/pdf/2606.26489)  

**Abstract**: News media play a central role in shaping public perceptions of climate change, and whether coverage emphasizes threats or solutions has measurable effects on audience engagement and policy support. Automated detection of these framing patterns at the sentence level would allow researchers to analyze large corpora that are infeasible to code manually. We present a systematic comparison of two approaches for classifying sentences from German-language climate news articles as threat-oriented, solution-oriented, both, or neither. The first approach uses few-shot prompting with an open-weights large language model (Llama 4 Maverick), employing chain-of-thought reasoning and structured output with confidence scoring. The second approach fine-tunes a German BERT model (deepset/gbert-large) for sentence-pair classification, where the preceding sentence provides contextual information for the target sentence. Both approaches implement two independent binary classifiers, one for threat framing and one for solution framing. We evaluate both methods on a corpus of 440 Austrian newspaper articles that were manually coded following a detailed coding scheme developed with domain experts. The fine-tuned BERT classifiers achieve an F1 score of 0.83 for both the threat and solution tasks, while the LLM-based classifiers reach an F1 of 0.78. An ablation study confirms that providing the preceding sentence as context improves BERT classification performance substantially compared to single-sentence input. These results contribute to the growing body of work comparing fine-tuned encoder models with prompted generative models for text classification in computational social science. 

---
# Speaking Numbers to LLMs: Multi-Wavelet Number Embeddings for Time Series Forecasting 

**Authors**: Defu Cao, Zijie Lei, Muyan Weng, Jiao Sun, Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.26487)  

**Abstract**: Large language models (LLMs) are attractive for context-aware time series forecasting because they can integrate heterogeneous textual signals, yet their discrete, language-oriented tokenization and embedding interfaces are misaligned with continuous numerical values, often harming numerical ordering and forecasting reliability. We propose TempoWave, a plug-and-play temporal wavelet digit interface that maps each scalar observation into digit-wise embeddings constructed from multi-wavelet, multi-scale coefficients. By directly overriding standard token representations, TempoWave seamlessly exposes both fine-grained local fluctuations and macro global structures in a transformer-compatible form, ensuring that precise numerical formatting, distinct digit identity, and robustness to common normalization operations are maintained throughout the LLM pipeline. Experiments across five context-enriched forecasting benchmarks demonstrate that TempoWave consistently improves LLM-based forecasters over standard numeric tokenization and alternative embedding interfaces, achieving a new state-of-the-art. These results highlight the numeric interface as a key bottleneck and suggest that principled multi-resolution embeddings can better couple LLMs' contextual reasoning with precise forecasting. Our code is available at this https URL and our model can be accessed at this https URL. 

---
# Utilizing Cognitive Signals Generated during Human Reading to Enhance Keyphrase Extraction from Microblogs 

**Authors**: Xinyi Yan, Yingyi Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26485)  

**Abstract**: Microblogging platforms generate massive amounts of short, noisy, and dispersed user content, making automatic keyphrase extraction (AKE) an important but challenging task. Prior studies have used eye-tracking signals to improve microblog-based AKE because such signals reflect readers' attention to salient words. However, eye tracking alone is limited by physiological, acquisition, and feature-decoding constraints. To address this issue, we investigate whether electroencephalogram (EEG) signals can complement eye-tracking signals for AKE. Using the ZuCo cognitive language processing corpus, we select 8 EEG features and 17 eye-tracking features and incorporate them into microblog-based AKE models. To reduce possible distortion of cognitive signals by model structures, we inject these features into the input of the soft-attention layer and the query vectors of the self-attention layer. We then evaluate different combinations of cognitive signals across AKE models. The results show that cognitive signals produced during reading consistently improve AKE performance, regardless of feature combinations and model architectures. EEG features bring the largest gains, while combining EEG and eye-tracking features yields performance between the two individual signal types, suggesting partial complementarity but also possible redundancy or noise. These findings indicate that EEG signals provide useful cognitive evidence for microblog-based AKE and that multimodal cognitive signals deserve further investigation. 

---
# Extracting Problem and Method Sentence from Scientific Papers: A Context-enhanced Transformer Using Formulaic Expression Desensitization 

**Authors**: Yingyi Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26481)  

**Abstract**: Billions of scientific papers lead to the need to identify essential parts from the massive text. Scientific research is an activity from putting forward problems to using methods. To learn the main idea from scientific papers, we focus on extracting problem and method sentences. Annotating sentences within scientific papers is labor-intensive, resulting in small-scale datasets that limit the amount of information models can learn. This limited information leads models to rely heavily on specific forms, which in turn reduces their generalization capabilities. This paper addresses the problems caused by small-scale datasets from three perspectives: increasing dataset scale, reducing dependence on specific forms, and enriching the information within sentences. To implement the first two ideas, we introduce the concept of formulaic expression (FE) desensitization and propose FE desensitization-based data augmenters to generate synthetic data and reduce models' reliance on FEs. For the third idea, we propose a context-enhanced transformer that utilizes context to measure the importance of words in target sentences and to reduce noise in the context. Furthermore, this paper conducts experiments using large language model (LLM) based in-context learning (ICL) methods. Quantitative and qualitative experiments demonstrate that our proposed models achieve a higher macro F1 score compared to the baseline models on two scientific paper datasets, with improvements of 3.71% and 2.67%, respectively. The LLM based ICL methods are found to be not suitable for the task of problem and method extraction. 

---
# Soft Token Alignment for Cross-Lingual Reasoning 

**Authors**: Jiayi He, Jungsoo Park, Wei Xu, Alan Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2606.26466)  

**Abstract**: Multilingual large language models often produce inconsistent reasoning and answers for semantically equivalent prompts in different languages. Prior work suggests that intermediate representations can be relatively language-agnostic, but generation becomes increasingly language-specific as models commit to discrete output tokens. This is problematic because language-specific lexical choices can cause semantically equivalent reasoning paths to diverge across languages. These divergences motivate searching for a cross-lingual alignment signal that is less tied to any single vocabulary item or script. We propose SOLAR, an auxiliary objective for supervised fine-tuning that aligns soft-token representations across languages, using English as a pivot. Soft tokens are probability-weighted mixtures over the vocabulary embeddings, yielding continuous representations that can aggregate information from semantically related tokens across languages. We then align each non-English soft-token summary to its English counterpart in the shared embedding space. Across four multilingual reasoning benchmarks, SOLAR improves accuracy by up to +17.7 points over the base model and +3.8 over standard supervised fine-tuning, with the largest gains on low-resource languages. SOLAR also strengthens final-layer cross-lingual similarity and substantially reduces language-cluster separability, suggesting that aligning soft-token representations helps preserve shared semantic structure during multilingual reasoning. 

---
# AnySimLite: A Lightweight Few-Shot Similarity Encoder for On-Device Speech-Adjacent Classification 

**Authors**: Sourav Ghosh, Yash Bhatia, Keshav Goyal, Sahil Singh Bagri, Mohamed Akram Ulla Shariff, Saravana Balaji Shanmugam  

**Link**: [PDF](https://arxiv.org/pdf/2606.26452)  

**Abstract**: To minimize privacy concerns and inference latency on edge devices like smartphones, lightweight on-device models remain important for end-user applications. Many of these applications involve natural language classification, but deploying multiple specialized models creates a memory footprint challenge. We investigate: Can a single lightweight architecture solve multiple Speech-Adjacent (SA) classification tasks through reduction to a nuanced text similarity formulation? We propose AnySimLite, a lightweight similarity encoder that combines word-level and character-level channels. Together with a dataset transformation strategy, we evaluate AnySimLite across multiple SA classification tasks and show that it consistently achieves state-of-the-art (SOTA) or SOTA-competitive performance in few-shot settings while maintaining a low memory footprint. Even in the worst case, the performance drop remains below 7% while using $<\frac{1}{250}^{\mathrm{th}}$ of the model size of the SOTA qLLaMA_LoRA-7B baseline. 

---
# ProvenAI: Provenance-Native Traces of Evidence in Generated Answers 

**Authors**: Mohammad Faizan, Dalal Alharthi  

**Link**: [PDF](https://arxiv.org/pdf/2606.26449)  

**Abstract**: Retrieval-augmented systems routinely present citations alongside generated answers, yet a citation does not confirm that the corresponding source meaningfully shaped the output. This paper introduces ProvenAI, a framework that decomposes transparency in multi-hop question answering into three independently measurable layers: answer correctness, citation fidelity against benchmark supporting evidence, and per-document influence under leave-one-resource-out intervention. Targeting the HotpotQA distractor benchmark through a seven-stage pipeline covering data normalisation, retrieval indexing, citation-aware answer generation, attribution auditing, ablation-based influence estimation, batch evaluation, and interactive inspection, ProvenAI evaluates 7,405 validation examples drawn from a canonical corpus of 509,300 passages. The system achieves 53.53% answer accuracy alongside a mean citation-fidelity score of 71.55%, and a worked example surfaces what we call the citation-influence gap: a clean citation audit co-occurring with a profile in which one cited source registers only weak influence while seven uncited sources demonstrably shift the output. We formalise the relationship between the implemented surface proxy and a token-level KL-divergence target through a stated faithfulness condition, ground the framework in causal-mediation analysis and database-provenance theory, and discuss how the three measurement layers compose with cryptographic provenance architectures emerging in autonomous scientific discovery. ProvenAI establishes that meaningful transparency in retrieval-grounded QA requires traceable links across retrieved, cited, and behaviourally influential evidence as three distinct, independently measured layers. 

---
# ConflictScore: Identifying and Measuring How Language Models Handle Conflicting Evidence 

**Authors**: Siyi Liu, Aaron Halfaker, Dan Roth, Patrick Xia  

**Link**: [PDF](https://arxiv.org/pdf/2606.26437)  

**Abstract**: Existing metrics for factuality and faithfulness evaluate whether an answer is supported or contradicted by its grounding documents, but they fail to capture when both supporting and contradicting evidence coexist. We introduce ConflictScore, a novel metric that quantifies how well a model's response acknowledges conflicting evidence in its grounding documents. Our framework decomposes responses into atomic claims, labels each claim against each grounding document, and then aggregates these labels into two complementary measures: ConflictScore-Count (CS-C), the proportion of claims exhibiting conflicts, and ConflictScore-Ratio (CS-R), the balance between supporting and contradicting evidence. We develop ConflictBench, a benchmark covering diverse forms of conflicts such as ambiguity, contradiction, and divergent opinions, to systematically evaluate our metric. Experiments show that ConflictScore effectively detects overconfident claims across domains and can serve as a corrective feedback mechanism that improves truthfulness on TruthfulQA. 

---
# ProfileFoundry: A Synthetic Person-Object Substrate for Privacy, Memory, and Tool-Use Evaluation in LLM Agent 

**Authors**: Sriram Selvam, Anneswa Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2606.26403)  

**Abstract**: Foundation-model research increasingly needs data about people: user state, personal histories, relationships, contact-like fields, documents, and longitudinal updates. Real user data is difficult to share, perturb, audit, or redistribute responsibly, while independently generated fake fields rarely preserve the cross-field and temporal consistency needed for controlled evaluation. We present PROFILEFOUNDRY, a deterministic generator and fixed reference release of 100,000 adult synthetic Person Objects across eight locales. Each object combines a typed current snapshot, household, family, and employer links, snapshot-aligned events, normalized relational views, and generation provenance. The release contains 709,228 events, 40,338 households, 52,491 employers, and 518,564 directed relationship edges. We report evidence in separate categories: selected population-marginal comparisons, per-object invariant checks, release-wide referential and temporal closure, and coincidence/provenance screens. PROFILEFOUNDRY is not a population-fidelity model, a rendered-text corpus, or a formal privacy mechanism. Instead, it is a responsible synthetic source layer for constructing downstream foundation-model evaluations involving memory, privacy, document understanding, record linkage, and agent state while keeping the synthetic person behind each artifact inspectable 

---
# Charting the Growth of Social-Physical HRI (spHRI): A Systematic Review Pipeline Augmented by Small Language Models 

**Authors**: Mayumi Mohan, Ju-Hung Chen, Alexis E. Block  

**Link**: [PDF](https://arxiv.org/pdf/2606.26382)  

**Abstract**: Social-physical human-robot interaction (spHRI) has grown rapidly across robotics, human-computer interaction, human-robot interaction, and haptics. Yet, fragmented terminology and inconsistent methodologies make systematic synthesis difficult. To support scalable review practices, we evaluated the extent to which small language models (SLMs; < 1.5B parameters) can assist with title and abstract screening for a large spHRI systematic review. While no SLMs matched human reviewers' performance, the models operated locally and screened papers orders of magnitude faster. The combined SLM ensemble identified 39 papers reviewers missed, representing 10.29% of the final relevant dataset. These results demonstrate that SLMs can augment, rather than replace, expert reviewers and make large-scale literature reviews accessible and sustainable. 

---
# Phonetic and semantic analyses of spoken corpora of Beijing and Taiwan Mandarin indicate that the neutral tone is a lexical tone 

**Authors**: Yuxin Lu, Zhexuan Li, R. Harald Baayen  

**Link**: [PDF](https://arxiv.org/pdf/2606.26360)  

**Abstract**: The neutral, or floating, tone of Mandarin Chinese is a tone with an enigmatic set of properties. It has been described as a reduced tone, or as a tone that sometimes is lexically fixed but that can also be toneless. In two-syllable words, it is found only on the second syllable, but single-syllable words can also have the neutral tone. We present a corpus-based study of the phonetic realization of the neutral tone in spontaneous conversational speech corpora of Beijing Mandarin and Taiwan Mandarin. We show that the neutral tone has its own tonal target, just as the four lexical tones of Mandarin. We also show that disyllabic words with a neutral tone have pitch contours that have a pitch component that depends on the tone on the first syllable, just as has been observed for two-syllable words with a lexical tone on the second syllable (Chuang et al., 2026). Furthermore, words with a floating tone have word-specific pitch signatures, which have also been documented for single-syllable words (Jin et al., 2026) as well as two-syllable words (Lu et al., 2026b). These word-specific pitch signatures are shown to be predictable to some extent from words' contextualized embeddings, as previously reported for lexical tones (Chuang et al., 2026; Lu et al., 2026b). As there is also considerable variability in the realization of lexical tones, we propose that the neutral tone is, in fact, a lexical tone in both Taiwan Mandarin and Beijing Mandarin. We document both similarities and differences in the realization of the floating tone in these two varieties and provide evidence, using contextualized embeddings, that some of the observed differences may arise from differences in the meanings of the words as used in the two corpora. 

---
# From Structure to Synergy: A Survey of Vision-Language Perception Paradigm Evolution in Multimodal Large Language Models 

**Authors**: Haoxiang Sun, Tao Wang, Li Yuan, Jian Zhao, Jiancheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2606.26196)  

**Abstract**: Multimodal Large Language Models (MLLMs) have recently made remarkable progress in unifying vision-language understanding and reasoning, especially following the introduction of models such as OpenAI's O-series and DeepSeek's R-series, which have driven a paradigm shift toward perception-centric intelligence. However, there remains a lack of systematic surveys that examine perception from a truly unified vision-language perspective -- one that treats vision and language as an inseparable modality. Existing reviews are often fragmented, focusing separately on either vision or language, and thus rarely capture the cross-modal evolution of perception as an integrated capability. To bridge this gap, we present the first systematic survey of unified vision-language perception in MLLMs. Specifically, we (1) formalize MLLM perception as an intrinsic, unified vision-language capability analogous to human innate perception, (2) introduce a five-stage taxonomy tracing the paradigm evolution of MLLM perception and survey representative methods and milestones at each phase, and (3) identify open challenges and outline promising research directions toward truly general, unified multimodal intelligence. We hope our study will provide both a foundational understanding and an actionable roadmap to foster further innovation on the path toward artificial general intelligence (AGI). 

---
# Thinking Like a Scientist? A Structural Study of LLM-Generated Research Methods 

**Authors**: Francesca Carlon, Brecht Verbeken, Vincent Ginis, Andres Algaba  

**Link**: [PDF](https://arxiv.org/pdf/2606.26130)  

**Abstract**: Large Language Models (LLMs) are increasingly used to guide research methodology, yet their default methodological tendencies under minimal prompting remain unclear. Here, we prompt GPT-5.1, Gemini 3 Pro, and DeepSeek-V3.2 with an LLM-extracted research question from each of 1,000 recent arXiv computer-science papers and compare the resulting methodology suggestions against a paper-derived experimental inventory. Since we provide only the research question, the differences we measure reflect initial suggestions and not how optimal those suggestions are. We extract structured method features from both sources, map them into a shared taxonomy, and quantify divergence across multiple taxonomy dimensions including model provider, dataset task type, and evaluation metric type. The strongest imbalance appears in provider choice, with Jensen-Shannon divergence about 3-5x larger than any other taxonomy dimension. Other/Academic single-occurrence models are underrepresented by 23-24 percentage points, while reused academic/community models are slightly overrepresented (4-6pp). LLMs also suggest a much narrower range of methods overall: the effective number of model entities contracts from 1,232 to 59-96, and inter-LLM rank correlations (0.55-0.68) generally exceed LLM-to-paper correlations (0.33-0.56), so the distortions are largely shared across models. Popularity baselines, BM25 retrieval calibration, and paper-level similarity tests confirm that the outputs are query-specific responses, but filtered through a narrower set of options. Researchers who rely on LLM suggestions without cross-checking therefore risk narrowing their methodological search space toward a more concentrated default. 

---
# Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM 

**Authors**: Tianyi Wu, Xiaoxi Sun, Yanhua Jiao, Yulin Li, Yixin Chen, YunHao Cao, YiQi Hu, Zhuotao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2606.26120)  

**Abstract**: Diffusion Large Language Models (dLLMs) offer a promising alternative to autoregressive models, excelling in text generation tasks due to their bidirectional attention mechanisms. However, their computational complexity scales on the order of L cubed with the sequence length L. This poses significant challenges for long-sequence and real-time applications, primarily due to the lack of compatibility with key-value caching and the non-autoregressive nature of denoising steps. Existing acceleration methods rely on static caching or parallel decoding strategies, which fail to account for the dynamic behavior of token properties across layers and decoding steps. We propose Dynamic-dLLM, a training-free framework that enhances dLLM inference efficiency through two components: Dynamic Cache Updating (DCU), which adaptively allocates cache-update budgets based on layer-wise token dynamics, and Adaptive Parallel Decoding (APD), which dynamically calibrates decoding thresholds to balance generation quality and efficiency. Extensive experiments on models like LLaDA-8B-Instruct, LLaDA-1.5, and Dream-v0-7B-Instruct across benchmarks such as MMLU, GSM8K, and HumanEval demonstrate that Dynamic-dLLM significantly improves inference speed. It attains an average speedup exceeding 3 times while maintaining performance. Dynamic-dLLM outperforms state-of-the-art acceleration methods and provides a plug-and-play solution for efficient dLLM deployment without compromising performance. The code is available at this https URL. 

---
# From Lexicon to AI: A Structured-Data Pipeline for Specialized Conversational Systems in Low-Resource Languages 

**Authors**: Siddhant Hitesh Mantri, Dhara Gorasiya, Malhar Kulkarni, Pushpak Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2606.26112)  

**Abstract**: Low-resource languages face a critical challenge in AI development: creating specialized conversational systems without access to massive training corpora. We present a systematic methodology for transforming structured linguistic resources into specialized AI systems, demonstrating that expert-curated lexical databases can serve as effective foundations for conversational AI development. Our approach converts Hindi WordNet into 1.25 million diverse instruction-response pairs, fine-tunes a 12B-parameter language model using resource-efficient LoRA with 4-bit quantization. Evaluation through a Hindi language learning chatbot demonstrates that structured-knowledge-based systems achieve superior pedagogical effectiveness (91.0 vs. 79.4-83.6 for general-purpose models) while maintaining competitive semantic performance and exceptional consistency. The complete pipeline demonstrates a proof-of-concept methodology using Hindi for developing specialized AI systems for any languages with WordNet resources. This work addresses the critical gap in AI accessibility for low-resource languages, offering a practical alternative to corpus-intensive approaches and potentially enabling specialized AI development for the hundreds of languages with existing WordNet resources. 

---
# Where Larger Models Excel: The Primacy of Constraint-Guided Reasoning 

**Authors**: Guan-Yi Lin, Hen-Hsen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26108)  

**Abstract**: Larger language models consistently outperform smaller ones on reasoning benchmarks, yet the reasoning differences underlying this gap remain underexplored. Across benchmarks in mathematics, physics, chemistry, and programming, we observe stable performance gaps: averaged over datasets, Qwen3-32B outperforms Qwen3-8B by 6.43%, while GPT-OSS-120B exceeds GPT-OSS-20B by 7.38%. To study the reasoning differences behind these gains, we develop AdvCluster, an automated framework that identifies questions where the larger model shows a stable advantage, extracts fine-grained advantage descriptions from paired reasoning traces produced by larger and smaller models, and organizes them through semantic clustering with quantitative evaluation and selection guided by a reviewer model. Our analysis yields a systematic taxonomy of larger model reasoning advantages, spanning both common advantages that recur across domains and specialized advantages associated with particular domains. Across these patterns, a recurring theme is Constraint-Guided Reasoning: larger models are better at identifying explicit and implicit constraints, organizing them into structured reasoning, and using them to rule out infeasible paths and verify intermediate steps. 

---
# Low Resource Multimodal Translation of Nepali Spoken Words into Emotion-Conditioned Sign Language Avatars 

**Authors**: Jatin Bhusal, Salma Tamang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26107)  

**Abstract**: Sign language communication systems, that integrate emotional expression remain underexplored, particularly for low-resource languages. This pilot study presents NEST-V1 (Nepali Emotion and Speech Transformer - Version 1), a proof-of-concept multimodal framework that demonstrates the feasibility of generating emotion-conditioned Nepali Sign Language avatars from spoken input. As a preliminary investigation, we focus on four common Nepali words ("thank you", "hello", "house", "me") across three emotional states (happy, neutral, sad) to validate our core technical approach. Our lightweight architecture employs a shared acoustic encoder for simultaneous Automatic Speech Recognition and emotion classification, achieving 81.1% ASR accuracy and 79.21% emotion recognition accuracy on a dataset of 600 labeled audio samples from 50 speakers. The system demonstrates 37% parameter efficiency compared to separate model architectures while maintaining a lightweight footprint with only 22.1M parameters suitable for edge deployment. This pilot work establishes the technical foundation for emotion-aware sign language translation in low-resource settings and provides a scalable framework for future expansion to larger vocabularies and more diverse emotional expressions. Our preliminary results indicate the viability of real-time, emotionally expressive sign language communication systems for the hearing-impaired community, with clear pathways for enhancement in subsequent development phases. 

---
# Reducing Conversational Escalation in Large Language Model Dialogue with Nonviolent Communication Constraints 

**Authors**: Zhixing Sun, Shenghe Xu, Tao Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.26106)  

**Abstract**: Large language models (LLMs) are increasingly used in emotionally charged situations involving interpersonal conflict, frustration, and distress. While prior safety research has focused on preventing explicit harms such as toxic or policy-violating content, less attention has been paid to conversational behaviors that may unintentionally escalate conflict. In this paper, we investigate whether LLMs can be guided toward more de-escalating dialogue behavior through lightweight prompt-level constraints derived from Nonviolent Communication (NVC). We reformulate NVC principles as process-oriented guidelines that discourage blame attribution, emphasize attention to users' emotional experiences, and encourage clarification before advice. Using a dual-agent simulation framework across multiple instruction-tuned models and user resistance levels, we show that NVC-constrained prompting consistently reduces conversational escalation and stabilizes interactions with highly resistant users. These results suggest that simple communication constraints can meaningfully improve the trustworthiness of LLM dialogue in conflict-prone settings. 

---
# Context Recycling for Long-Horizon LLM Inference 

**Authors**: Derek Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2606.26105)  

**Abstract**: Large language models (LLMs) exhibit strong capabilities in short-context reasoning but degrade in performance over long conversational horizons due to context window limitations and inefficient token usage. We introduce ContextForge, a system for context recycling that maintains task-relevant information across turns by combining structured query generation, external memory retrieval, and controlled synthesis. The system enables efficient reuse of prior computation without relying on full context replay, reducing token overhead while preserving answer quality. We evaluate ContextForge using a 15-turn conversational benchmark that tests multi-turn reasoning, back-references, and domain shifts across structured healthcare queries. Compared to a baseline agent using identical underlying models, ContextForge demonstrates improved consistency and reduced token consumption, while maintaining comparable response accuracy. These results suggest that context recycling provides a practical approach for extending LLM capabilities in long-horizon tasks without requiring larger context windows or model retraining. Code and evaluation artifacts are available at this https URL. 

---
# Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfare 

**Authors**: Jasmine Brazilek, Harper Dunn  

**Link**: [PDF](https://arxiv.org/pdf/2606.26104)  

**Abstract**: Animal-welfare advocates produce a lot of writing, and increasingly that writing trains the language models that millions of people then ask about animal welfare. Using vocabulary-matched stance-contrast probes on a held-out animal-welfare benchmark, we measure how each of ten linguistic features changes Llama-3.2-1B's preference for pro-animal-welfare reasoning when used as fine-tuning data. Eight of the ten features produce statistically significant shifts. Seven move the model toward stronger pro-animal-welfare reasoning: assertive certainty, explicit moral vocabulary, emotion words, evaluative claims, narrative structure, depicted harm severity, and immediate temporal framing. Two move it the other way: hedged language and concrete sensory description both dilute the pro-animal-welfare stance. First-person perspective has no statistically significant effect. The practical recommendation for anyone writing animal-welfare text that may end up in LLM training corpora: assert a position rather than describe a scene neutrally. The features that shift the model are the ones that make the writer's position explicit; the features that dilute it hold animal-welfare content but withhold stance. 

---
# Investigating LLM's Problem Solving Capability -- a Study on Statics Questions 

**Authors**: Tanner Culleton, Hung-Fu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26103)  

**Abstract**: Large Language Models (LLMs) have rapidly influenced many aspects of society, particularly education, due to their demonstrated ability to complete assignments and examinations across a wide range of subjects. Although prior studies have examined the educational impact of LLMs, much of the existing work relies on public or open problem datasets and lacks topic-specific analysis. In engineering education, especially within mechanical engineering, systematic investigations of LLM performance on specific problem types remain limited. Instead of using traditional methods that directly ask textbook questions to an LLM tool, our study adopts a model distillation process to evaluate LLM capabilities in solving statics problems. By distilling ChatGPT, we extracted 25 text-only statics questions and further constructed two additional datasets by adding diagrams and modifying their numerical values. Experimental results show that while LLMs perform well on text-only statics problems, their accuracy decreases when diagrams are introduced and the problems require multi-step reasoning. Further analysis suggests that this performance drop is not primarily caused by limitations in image recognition, but rather by difficulties in multi-step reasoning and in consistently applying extracted visual information across successive solution stages. 

---
# Helpfulness Hurts: Domain-Dependent Degradation of Mid-Trained Compassion Values Under Post-Training 

**Authors**: Jasmine Brazilek, Juliana Seawell  

**Link**: [PDF](https://arxiv.org/pdf/2606.26102)  

**Abstract**: Standard post-training pipelines apply supervised fine-tuning (SFT) and reinforcement learning (RL) to make language models helpful, but these processes may inadvertently degrade values instilled during pre-training. We investigate whether the domain of post-training data differentially affects the retention of animal compassion values in a Llama 3.1 8B model mid-trained on compassion-oriented synthetic data, using both SFT (helpfulness via Dolly-15k vs. coding via Magicoder-110K) and GRPO (helpfulness via RLHFlow vs. coding via Magicoder), evaluated on the Animal Harm Benchmark (AHB 2.2) and MORU benchmark (Moral Reasoning Under Uncertainty). Helpfulness training significantly degrades animal compassion relative to coding training on AHB (SFT: 35.7% vs. 65.2%; GRPO: 18.7% vs. 32.0%), replicating across two independent helpfulness datasets and two training paradigms. On English MORU items, helpfulness training degrades general moral reasoning by 25.5 percentage points (46.4% vs. 71.9%), a striking gap that rivals the compassion effect in magnitude. However, this effect does not transfer cross-lingually: on the multilingual MORU benchmark, the domain effect disappears (SFT: 52.3% vs. 51.2%). In contrast, the animal compassion effect transfers consistently across languages, with Magicoder's AHB percentage-point gain over the base model 4.5 times larger on non-English items than English items. This divergence suggests that values instilled through mid-training are encoded more deeply and cross-lingually than reasoning improvements from domain-specific post-training. These results suggest that, for labs building on value-laden mid-training, coding-domain post-training may better preserve mid-trained values than helpfulness post-training without harming general reasoning capabilities. 

---
# Know2Guess: A Contamination-Aware Multi-Zone Benchmark for Knowledge-Boundary Evaluation in Large Language Models 

**Authors**: Renwei Meng, Bowen Zhang, Jian Wang, Xican Wang, Haoyi Wu, Xuanyan Qiu, Shengan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26101)  

**Abstract**: Reliable evaluation of large language models should separate supported answering from unsupported guessing without conflating either with data contamination, prompt idiosyncrasy, or generic refusal behavior. We present a contamination-aware, multi-zone benchmark for measuring the transition from answerable knowledge to abstention-expected unknowns under frozen build-time labels. The benchmark contains 1,200 items across five domains, explicit abstention expectations, contamination-risk metadata, and dual parsing with an official strict parser plus a normalized robustness parser. We evaluate FLAN-T5, Qwen2.5-Instruct, and Llama-3-Instruct models under locked answer-or-abstain prompts, answer-only controls, and prompt-template variants. The benchmark is not solved by generic non-answer behavior: FLAN baselines remain weak on productive abstention, while stronger instruction-tuned models expose a selective but incomplete transition from answering to abstaining. Qwen2.5-3B-Instruct achieves the best overall reliability, but answer-expected zones remain difficult, calibration remains poor, and benign-item refusal persists. Prompt and parser robustness analyses preserve the main ranking and qualitative conclusions. The benchmark therefore provides a reproducible protocol for auditing answerability, abstention, refusal, and contamination as distinct but interacting dimensions of LLM this http URL dataset is publicly available at this https URL. 

---
# HierBias: Context-Conditioned Hierarchical Media Bias Detection with Multi-Task Type Classification 

**Authors**: Kaining Li, Ruichen Yan, Yuxin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.26100)  

**Abstract**: Media bias detection is a critical task for ensuring fair and balanced information dissemination, yet existing sentence-level approaches classify each sentence independently, ignoring inter-sentence contextual signals that human annotators naturally exploit. We present \textbf{HierBias}, a hierarchical context-conditioned media bias detector that formally models document context in bias prediction. We introduce the \emph{context-conditioned bias probability} and prove theoretically that leveraging document context strictly reduces the Bayes error of sentence-level classification when inter-sentence mutual information is non-zero. A multi-task generalization bound further establishes that jointly training binary bias detection and fine-grained bias type classification improves sample efficiency on small annotated corpora. Architecturally, HierBias pairs a sentence-level RoBERTa encoder with a cross-sentence Transformer aggregator and dual output heads for binary detection and four-class type classification. Evaluated on BABE and BASIL, HierBias achieves 0.853 F1 and 0.723 MCC, surpassing the state-of-the-art bias-detector by $+2.6\%$ F1 and $+4.3\%$ MCC (McNemar's test, $p < 0.05$). Ablation experiments confirm that each theoretical component contributes independently and consistently. 

---
# DanceOPD: On-Policy Generative Field Distillation 

**Authors**: Wei Zhou, Xiongwei Zhu, Zelin Xu, Bo Dong, Lixue Gong, Yongyuan Liang, Meng Chu, Leigang Qu, Lingdong Kong, Wei Liu, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2606.27377)  

**Abstract**: Modern image generation demands a single model that unifies diverse capabilities, including text-to-image (T2I), local editing, and global editing. However, these capabilities are rarely naturally aligned and often conflict. For instance, editing tends to degrade T2I performance, while global and local editing interfere with each other. Consequently, effectively composing these capabilities has become a central challenge for image generation model training. To tackle this, we introduce DanceOPD, an on-policy generative field distillation framework for flow-matching models that routes each sample to one capability field, queries one low-noise student-induced state, and trains with a simple velocity MSE objective. With each capability source defined as a velocity field over the shared flow state space, the student learns from fields queried on its own rollout states to compose expert capabilities. This formulation also absorbs operator-defined fields such as classifier-free guidance. Comprehensive experiments on T2I, editing, realism-field absorption, and CFG absorption show that our approach improves multi-capability composition, strengthening target capabilities while preserving anchor generation quality. We believe this work establishes a practical route for generative field distillation in flow-matching models. 

---
# The Geometry of Updates: Fisher Alignment at Vocabulary Scale 

**Authors**: John Sweeney  

**Link**: [PDF](https://arxiv.org/pdf/2606.27242)  

**Abstract**: Training-free source selection for LLM families with shared vocabularies arises in scientific string domains such as SMILES, protein, and genomic sequences, where candidate corpora share a tokenizer but differ in prediction targets. This creates an activation-dark regime: representation-similarity metrics can be uninformative without assumptions about label-conditioned error geometry, while classical update-geometry metrics are computationally prohibitive at vocabulary scale. We show that, in a shared-output head setting, representation metrics (e.g., CKA) are non-identifiable for transfer; models can share identical representations yet have orthogonal head updates. The key identity is that head Fisher alignment is exactly a cosine between kernel mean embeddings in the joint activation-error space, exposing activation, error, and coupling factors rather than requiring a materialized Fisher matrix. FisherSketch estimates this cosine directly in a single streaming pass, making K=128,256 head Fisher alignment practical with a 16 KB task signature (m=4096) and a 192 KB per-task streaming state, small enough to store next to a model hash, but encoding transfer-relevant update structure. Beyond source selection, the same signatures and marginals provide a diagnostic instrument for studying whether LLM task similarity is driven by activations, errors, or their coupling; shared-parameter and internal-layer validations, together with Llama-3.1-8B verbalizer-shift experiments, show that FisherSketch remains informative when activation similarity cannot distinguish tasks. 

---
# Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement 

**Authors**: Sangwoo Cho, Kushal Chawla, Pengshan Cai, Zefang Liu, Chenyang Zhu, Shi-Xiong Zhang, Sambit Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27226)  

**Abstract**: Evaluating LLM outputs remains a major bottleneck in NLP: human evaluation is expensive and slow, lexical metrics correlate poorly with human judgments on open-ended generation, and holistic LLM judges often produce opaque scores that are hard to debug. We propose BINEVAL, a framework that decomposes evaluation criteria into atomic binary questions and aggregates the resulting verdicts into interpretable, multi-dimensional scores. Given a task prompt, a meta-prompt generates fine-grained evaluation questions, and an LLM answers them independently for each output, yielding transparent question-level feedback together with calibrated overall scores. This decomposition makes evaluation easier to inspect, easier to diagnose, and directly usable for prompt improvement. Across SummEval, Topical-Chat, and QAGS, BINEVAL matches or outperforms strong baselines including UniEval and G-Eval, with especially strong results on factual consistency benchmarks such as QAGS. Beyond competitive correlation with human judgments, BINEVAL better matches human score distributions and avoids the ceiling effects common in prior LLM judges, leading to better discrimination between borderline and clearly flawed outputs. We further show that the same question-level feedback supports iterative prompt optimization, improving evaluator prompts on summarization and generation prompts on IFBench under both self-update and cross-model update settings. Overall, BINEVAL provides a task-agnostic, training-free, and interpretable evaluation framework that combines strong empirical performance with practical diagnostic and optimization value. 

---
# HarmVideoBench: Benchmarking Harmful Video Understanding in Large Multimodal Models 

**Authors**: Jiajun Wu, Haoyu Kang, Yining Sun, Jiacheng Hou, Heng Zhang, Danyang Zhang, Zhenjun Zhao, Haochi Zhang, Leixin Sun, Eric Hanchen Jiang, Yushan Li, Ruiyu Li, Mengkai Huang, Yan Gao, Xu Zhang, Guancheng Wan  

**Link**: [PDF](https://arxiv.org/pdf/2606.27187)  

**Abstract**: Large vision-language models (LVLMs) have recently shown immense potential in automated content moderation, sparking growing interest in developing harmful-video benchmarks. However, we identify two primary limitations in existing works: 1) The multi-layered characteristics of harmful videos are overlooked. Existing benchmarks predominantly formulate evaluation as a binary classification task, failing to capture implicit or deep contextual harms. 2) Explanatory rationales are completely absent. Current frameworks measure exclusively whether a model flags a video correctly rather than explaining why, turning evaluation into a black box where models can succeed through superficial shortcuts. To address these problems, we present HarmVideoBench, a multi-layered diagnostic benchmark comprising 1,379 videos paired with 4,137 multiple-choice questions. HarmVideoBench benchmarks three hierarchical dimensions: Observable Evidence, Clip-Internal Meaning, and Beyond-Clip Reasoning, aiming to evaluate models' deep understanding beyond surface cues with carefully balanced and curated samples. We evaluate 19 leading models on HarmVideoBench to assess their multidimensional understanding of harmful videos. Moreover, we introduce BCR, a benchmark-aligned method that predicts reasoning boundaries and dynamically retrieves context only when needed. Experimental results show that BCR substantially improves the base model's performance in harmful video understanding, raising the macro average from 61.7 percent to a state-of-the-art 84.4 percent. 

---
# Just how sure are you? Improving Verbalized Uncertainty Calibration in Medical VQA 

**Authors**: Eren Senoglu, Federico Toschi, Nicolo Brunello, Andrea Sassella, Mark James Carman  

**Link**: [PDF](https://arxiv.org/pdf/2606.27023)  

**Abstract**: Multimodal large language models (MLLMs) applied to Medical Visual Question Answering (VQA) tend to produce overconfident outputs regardless of actual correctness, and existing verbalized confidence calibration methods, developed primarily for text only LLMs, do not account for the multimodal nature of medical image understanding.
This work proposes a training based framework that finetunes MLLMs to improve their calibration using a composite loss function combining a Brier style calibration term, an anchor regularizer that prevents confidence collapse toward extreme values, a contrastive image text alignment term, and a KL based model stabilization term. The alignment signal is derived from a $2 \times 2$ factorial perturbation design that crosses image presence with text integrity, probing the reliance of the model on visual modality input versus language priors. Finally, a top K KL divergence regularizer is used to protect the answering ability of the model during finetuning.
Across three Medical VQA benchmarks and two architectures (MedGemma 4B IT and Qwen2 VL 7B Instruct), our method reduces calibration error by 60% or more, and improves discrimination by 26% or more, while preserving predictive accuracy. On average across benchmarks, the technique outperforms prompting based, sampling based, and training based approaches, and ablation experiments confirm that each component of the loss function is indeed necessary for improving the calibration. All code for the experiments is publicly available. 

---
# Einstein World Models 

**Authors**: Munachiso Samuel Nwadike, Zangir Iklassov, Ali Mekky, Zayd M. Kawakibi Zuhri, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2606.26969)  

**Abstract**: Does intelligence require the ability to reason about phenomena beyond direct experience? It is natural to suspect that some complex thought cannot be captured through language alone. However, of particular concern to this work, is whether visualising counterfactual events can complement language as a mechanism for complex thought. We ask whether LLMs can be trained to utilise such visualisation mechanisms, in a way that benefits their reasoning abilities. Motivated by this question, we propose Einstein World Models. EWMs are a blueprint for LLM-based reasoning systems that place visual-temporal rollouts inside the reasoning trace, allowing them to reason in ways that text alone may not support well. In an EWM, the LLM calls a world-module (not to be confused with a world model), to produce short rollouts of scenes under consideration. The returned rollout is treated not as the answer, but as an inspectable hypothesis that can support later reasoning. Einstein World Models extend the capability of LLMs for tool calling (such as web search or code execution), into the domain of visual thought experiments. 

---
# Jailbreaking for the Average Jane: Choosing Optimal Jailbreaks via Bandit Algorithms for Automatically Enhanced Queries 

**Authors**: Prarabdh Shukla, Ritik, Suhas Rao, Arpit Agarwal, Arjun Bhagoji  

**Link**: [PDF](https://arxiv.org/pdf/2606.26936)  

**Abstract**: With a profusion of jailbreaks for LLMs now widely known, a growing concern is that non-expert malicious actors ("the average Jane") could elicit actionable responses to malicious requests. In this work, we examine whether this concern is justified. A non-expert malicious actor requires two ingredients for a successful attack: a powerful jailbreak for their target model, acting on an effective malicious query. For the former, we propose a novel attack strategy based on the multi-armed bandit framework. This allows efficient online learning of the optimal jailbreak from a large choice set via noisy exploration on a small number of queries, with subsequent application of the learnt policy on an exploitation set. For the latter, we curate $\mathrm{FrankensteinBench}$, a safety benchmark of $11,279$ malicious queries drawn from manual curation over $7$ existing benchmarks, along with automated enhancement and generation. Each query is categorized as simple or complex by the technical expertise required to craft it. Our findings confirm the concern. Our bandit-based attack achieves success rates as high as $97\%$ on average over $15$ SoTA open-weight LLMs. Moreover, adding complexity to queries raises the attack success rate by up to $26\%$ on average across models -- making it an effective, automatable prompting strategy. 

---
# AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems 

**Authors**: Changxin Lao, Fei Pan, Guozhuang Ma, Han Li, Huihuang Lin, Jijun Shi, Kangzhi Zhao, Kun Gai, Mo Zhou, Qinqin Zhou, Quan Chen, Ruochen Yang, Shifu Bie, Shuang Yang, Shuo Yang, Wenhao Li, Wentao Xie, Xiao Lv, Xuming Wang, Yijun Wang, Yiming Chen, Yusheng Huang, Zhongyuan Wang, Zibo Zhao, Zijie Zhuang, Baoning Xia, Chao Liu, Chaoyi Ma, Chubo He, Dawei Cong, Feng Jiang, Gang Wang, Guilin Xia, Hanwen Xu, Jiahong Xie, Jiahui Qiao, Jian Liang, Jiangfan Yue, Jing Wang, Jinghan Yang, Jinghui Jia, Kan Qin, Lei Wang, Ming Li, Peilin Song, Pengbo Xu, Qiang Luo, Ruiming Tang, Shiyang Liu, Shuxian Jin, Tao Wang, Tao Zhang, Xiang Gao, Xianghan Li, Yingsong Luo, Yiwen Ning, Yongcheng Liu, Yuan Guo, Zhaojie Liu, Zhenkai Cui  

**Link**: [PDF](https://arxiv.org/pdf/2606.26859)  

**Abstract**: Recommendation algorithm iteration is moving from an artisanal, engineer-bound process toward an industrialized research loop, but this transition remains blocked by a structural execution bottleneck: the idea-to-launch cycle still depends on human engineers to generate hypotheses, modify production code, launch A/B experiments, and attribute online results. Innovation therefore scales linearly with headcount rather than compounding with evidence, compute, and accumulated experimental knowledge. We present AgentX, a production-deployed multi-agent system that fundamentally restructures this production function. AgentX operates as a self-evolving development engine: it autonomously generates, implements, evaluates, and learns from recommendation experiments at a scale and pace that no manual workflow can sustain.
The system orchestrates four tightly coupled stages in a closed loop. A Brainstorm Agent synthesizes evidence from historical experiments, system architecture, data analysis, and external research into ranked, executable proposals. A Developing Agent translates each proposal into production-ready code through repository-grounded generation and multi-dimensional reliability verification. An Evaluation Agent conducts safe online rollout with guardrail-vetoed A/B judgment, converting both successes and failures into structured knowledge assets. A Harness Evolution layer (SGPO) then distills execution trajectories into semantic-gradient updates that continuously sharpen the agents themselves -- making the system not merely automated, but self-improving. 

---
# KARLA: Knowledge-base Augmented Retrieval for Language Models 

**Authors**: Francois Crespin, Fabian M. Suchanek, Nils Holzenberger  

**Link**: [PDF](https://arxiv.org/pdf/2606.26807)  

**Abstract**: We propose a new method that allows an LLM to automatically pull in factual knowledge from a knowledge base during token generation. This means that (1)~factual knowledge in the LLM output can be updated without retraining the LLM, (2)~facts in the LLM output can be traced to the knowledge base for transparency and explainability, and (3)~smaller models can achieve the same factual accuracy as larger models. Our core idea is to train the model to produce special tokens that trigger a query to the knowledge base. Our experiments show that our method improves factual grounding in both short and long-form generation, and allows factual revisions to take effect through KB edits rather than parameter updates. 

---
# AIGP: An LLM-Based Framework for Long-Term Value Alignment in E-Commerce Pricing 

**Authors**: Chennan Ma, Yanning Zhang, Siqi Hong, Xiuchong Wang, Fei Xiao, Keping Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26787)  

**Abstract**: Traditional dynamic pricing models in large-scale e-commerce suffer from limited interpretability, poor utilization of unstructured information, and misalignment with long-term business objectives such as cumulative Gross Merchandise Value (GMV), Return on Investment (ROI) and milestone achievement. We propose AIGP, a novel framework that leverages a Large Language Model (LLM) prompted with domain knowledge, structured data and textual context to make interpretable, knowledge-aware pricing decisions. For efficient deployment while maintaining high-quality outputs, we employ supervised fine-tuning for knowledge distillation. Central to AIGP is the Long-Term Value Estimator (LTVE), trained via offline reinforcement learning on historical data, which serves as a reward model to score candidate pricing actions and select preference pairs for Direct Preference Optimization (DPO), thereby aligning the pricing policy with long-term business objectives. Extensive offline evaluations and large-scale online A/B tests on Tao Factory demonstrate that AIGP achieves significant improvements: +13.21% in GMV, +7.59% in ROI, and +8.20% in milestone achievement rate over 14 days compared to the production baseline, while simultaneously providing interpretable and transparent pricing rationales. 

---
# Reproducibility Study of "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models" 

**Authors**: Ananth K S, Arya Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2606.26783)  

**Abstract**: Fang et al. (2025) introduced a null-space constrained projection, named AlphaEdit, for locate-then-edit knowledge editing methods, theoretically guaranteeing that edits do not disrupt previously preserved knowledge, and reports substantial gains over existing editing methods on LLaMA3, GPT2-XL, and GPT-J. In this work, we present a reproducibility study of AlphaEdit, reproducing its reported results under the original experimental setup and extending the evaluation along three axes: new model architectures, additional downstream benchmarks, and substantially longer sequential editing horizons. We successfully reproduce AlphaEdit's reported metrics across the original models, though we identify a discrepancy in the reported fluency and consistency metric. Extending AlphaEdit to newer model families, we find that its advantage does not generalize uniformly, which we trace to architectural assumptions in the locate-then-edit paradigm that are violated by these newer models. We further stress-test AlphaEdit's central sequential-editing claim by extending the number of edits well beyond those evaluated in the original paper, and find that performance, which is stable at the originally reported scale, degrades as edits reach a much higher count, indicating that the null-space projection's protection against catastrophic forgetting is bounded rather than unconditional. Finally, we extend evaluation of edited models on three extra benchmarks, namely, BoolQ, HellaSwag, and XSTest, and we find that large-scale sequential editing degrades both general downstream task competence and safety-relevant refusal behavior. Our results confirm that AlphaEdit performs as reported within its original scope, while showing that its core theoretical guarantees are sensitive to model architecture and editing scale in ways that have practical implications for its deployment. 

---
# Structure Before Collapse: Transient semantic geometry in next-token prediction 

**Authors**: Yize Zhao, Isabel Papadimitriou, Christos Thrampoulidis  

**Link**: [PDF](https://arxiv.org/pdf/2606.26749)  

**Abstract**: Neural Collapse predicts that balanced one-hot classification pushes model representations to be equally far from each other; a symmetric configuration that depends only on the output label and ignores any semantic similarity in the inputs. This creates a puzzle: next-token prediction language models are trained predominantly (as context length increases) with one-hot labels: the same context is very unlikely to appear twice in training with different labels. However, they clearly learn latent structural features. That is, despite the one-hot training regime, a language model's contextual embeddings represent the fact that the next word in ''Mary broke the ___'' is likely to be filled by tokens in the latent classes of a) medium-sized, b) rigid, c) inanimate nouns. How does gradient descent find such categorical semantic structure when co-occurrence statistics collapse to one-hot sparsity, eliminating any shared next-tokens among different contexts? To investigate this tension we identify three synthetic controlled settings where inputs have latent semantic factors but are mapped to distinct one-hot labels. We find that semantic geometry emerges early in training, and that representations cluster by shared attributes despite receiving no explicit supervision to do so. This structure is transient: with sufficient capacity and time, the model eventually reaches the predicted symmetric state where all representations are equally separated. We study this phase transition through Gram matrix analysis and propose a preliminary modification to the commonly used unconstrained features model to capture the emergent semantic geometry. 

---
# HyperDFlash: MHC-Aligned Block Speculative Decoding with Gated Residual Reduction 

**Authors**: Luxi Lin, Shuang Peng, Rui Ma, Junhao Hua, Shuwei Fan, Zhengda Qin, Qiang Wang, Hongjian Sun, Fangmin Chen, Songwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.26744)  

**Abstract**: We present HyperDFlash, a block-parallel speculative decoding framework tailored to the novel multi-hyper-connection (MHC) architecture proposed by DeepSeek-V4. Despite the strong initial-token drafting performance of the native Multi-Token Prediction (MTP) module in DeepSeek-V4, its draft accuracy degrades sharply at later positions, as error accumulation from unverified intermediate tokens harms acceptance rates. Although the original DFlash method supports efficient one-pass block drafting, it cannot be seamlessly adapted to the MHC paradigm, since the multi-path residual stream of DeepSeek-V4 induces feature misalignment with conventional drafting designs. To resolve this mismatch, we propose two model-aligned optimizations for MHC residual streams. First, we adopt pre-collapse residual states as the exclusive conditioning signal, preserving multi-path structural information and aligning the drafter with the native prediction pathway of the target model. Second, we replace the heavy generic linear compressor with a lightweight gated residual reducer, whose parameters are inherited from the built-in hyper-connection head. This design yields input-aware path aggregation with three orders of magnitude fewer parameters while maintaining architectural alignment. We further enhance training via a targeted KL distillation loss applied to the LM-head, which regularizes predictions against the full target probability distribution and improves draft quality at early training stages. Experiments across math reasoning, code synthesis, and conversational benchmarks show that HyperDFlash consistently outperforms both the native MTP baseline and vanilla DFlash adaptation. It achieves substantial gains in average accepted draft length and decoding speedup, validating the effectiveness of MHC alignment, gated reduction, and targeted distillation for high-performance speculative decoding. 

---
# Do Safety Guardrails Need to Reason? LeanGuard: A Fast and Light Approach for Robust Moderation 

**Authors**: Dongbin Na  

**Link**: [PDF](https://arxiv.org/pdf/2606.26686)  

**Abstract**: In order to screen a prompt or a response, the recent guardrail methods generate a chain-of-thought (CoT) before they issue a verdict. This design follows a common belief that step-by-step reasoning improves a decision. However, CoT also makes the guard heavy and slow, because the model must generate many tokens before it decides. This may not match how guardrails are actually deployed. A guardrail sometimes should not be heavy and slow, and it often runs on-device, for example on an embodied robot. In this paper, we pose a question whether a safety guardrail really needs to reason. To answer this question, we train a lightweight bidirectional encoder and a reasoning guard on the same corpus, and we then remove only the reasoning while we keep everything else fixed. With this controlled same-base comparison, we show that the chain does not improve moderation accuracy. We name the resulting guard LeanGuard. A 395M label-only encoder reaches an average F1 of 82.90 $\pm$ 0.26 over public benchmarks. It matches a reasoning guard that is built on a much larger decoder, while it uses only a single forward pass over an input of at most 512 tokens. This is about a ~100x reduction in inference compute. We further show that this label-only encoder stays robust under training-label noise and retains far more recall at a strict false-positive rate than the reasoning guard, so a heavier reasoning guard is not the more robust choice either. Our finding suggests that the current guardrail benchmarks may not be hard enough to reward reasoning, and that the necessity of CoT for moderation is still not proven. We release all source codes and models including LeanGuard at this https URL. 

---
# From Weights to Features: SAE-Guided Activation Regularization for LLM Continual Learning 

**Authors**: Evan Ning, Wei Xue, Dong Lou, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.26629)  

**Abstract**: Weight-space regularization methods such as Elastic Weight Consolidation (EWC) are the standard approach to catastrophic forgetting in continual learning. However, those methods tend to underperform when applied to large language models. We argue that such underperformance can be partly explained by the ``polysemantic'' nature of large language models: per-weight importance estimates utilized by EWC-style regularization are too coarse and cannot isolate the knowledge that needs protection. In this paper, we propose regularizing instead in the model's activation space, using pretrained Sparse Autoencoders (SAEs) as a monosemantic feature dictionary. From the perspective of constrained optimization, we derive a new loss function that uses the SAE feature dictionary to explicitly balance stability and plasticity, and show that EWC is a special case in the one-sided weight-space penalty setting. Unlike replay-based methods that store or revisit examples from earlier tasks, our method requires no previous-task data after mask construction: current-task data is used to compute a compact SAE feature mask, and only this mask is retained for later training. Further, since the feature space has significantly lower dimensionality than the parameter space, the proposed method is more memory efficient. On the TRACE and MedCL continual learning benchmarks, the method achieves the strongest result among approaches without introducing task-specific architectural components, also surpassing traditional weight-space regularization methods like EWC. Beyond performance comparisons, we provide empirical evidence for the polysemanticity thesis: task-relevant representations are linearly separable in the SAE feature basis but indistinguishable from chance in the weight basis, and weight-space protection is nearly non-selective at the concept level. 

---
# Adversarial Diffusion Across Modalities: A Fusion Survey of Attacks, Defenses, and Evaluation for Text, Vision, and Vision-Language Models 

**Authors**: Abrar Alotaibi, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2606.26566)  

**Abstract**: Adversarial evaluation of AI systems has matured along four largely disconnected tracks: diffusion-based attacks on text and large language models (LLMs), diffusion-based attacks on image classifiers, jailbreak pipelines against vision-language models, and diffusion-based input purification defenses. Each has developed its own vocabulary, threat models, and benchmarks, with denoising diffusion models emerging as a shared generative mechanism whose recipes are now actively ported between communities. This survey performs an information-fusion exercise at the meta-research level: we integrate these four tracks into a single conceptual framework with a unified taxonomy, evaluation criteria, and research agenda, focusing on the LLM-side slice. We catalog fifty published papers across four scope areas (text/LLM, image classifier, vision-language model, defense), plus four diffusion-LLM-as-victim entries and ten non-diffusion baselines against which any new attack must be compared. We propose a six-class taxonomy of diffusion roles in adversarial pipelines, augmented by a threat-model axis recording attacker knowledge, query budget, and target accessibility, and apply a five-dimension framework (attack success rate, transferability, query budget, perplexity, defense-evasion) uniformly across modalities. The review adopts a dual attacker-defender perspective: alongside the attack catalog we cover four diffusion-based defenses that form the natural evaluation backdrop for new attacks. Our critical analysis identifies five recurring weaknesses of the current LLM-side literature, and we close with a research agenda of open questions and concrete experimental designs. The companion catalog and spreadsheet are released with the paper. We are explicit that this is a narrative review with quality assessment, not a PRISMA-compliant systematic review, and discuss the implications for replication. 

---
# Compiler-Driven Approximation Tuning for Hyperdimensional Computing 

**Authors**: Xavier Routh, Abdul Rafae Noor, Akash Kothari, Zheyu Li, Mahbod Afarin, Tajana Rosing, Vikram Adve  

**Link**: [PDF](https://arxiv.org/pdf/2606.26547)  

**Abstract**: As Moore's law reaches its physical and economic limits, domain-specific approaches are increasingly employed to accelerate machine learning workloads. Hyperdimensional Computing (HDC) represents one such emerging paradigm, offering an alternative to conventional deep learning techniques. Rooted in cognitive models of computation, HDC is designed bottom-up with hardware efficiency as a first-class objective. HDC workloads map naturally to heterogeneous hardware platforms, including CPUs, GPUs, and FPGAs, as well as emerging in-memory computing technologies such as Resistive RAM (ReRAM) and Phase-Change Memory (PCM). HDC algorithms are intrinsically tolerant to noise and approximation, enabling substantial performance gains with minimal accuracy loss. In this work, we introduce ApproxHDC, a framework for automated identification and application of domain-specific approximations in HDC workloads. ApproxHDC extends the HPVM-HDC compiler infrastructure to enable retargetable compilation across diverse hardware backends, including CPUs, GPUs, and simulated ReRAM and PCM-based accelerators. The space of possible approximations is exponentially large; ApproxHDC employs efficient search and analysis to navigate it and identify high-impact configurations spanning both software and hardware levels. 

---
# Humans Disengage, Reasoning Models Persist: Separating Difficulty Registration from Deliberation Allocation 

**Authors**: Han-yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26502)  

**Abstract**: Large reasoning models (LRMs) take longer on harder problems, just as humans do. This surface similarity hides an opposite pattern within items. When an LRM gets a problem wrong, it spends more tokens than when it gets the same problem right; humans do the reverse, spending less time on the trials they get wrong. We separate two levels of deliberation: how response time tracks difficulty across items (registration), and, with item identity held fixed, whether an agent spends more on its own failures or successes (allocation). On a public matched human-LRM corpus, humans and all five thinking LRMs reproduce the known cross-item alignment (registration) but diverge within items (allocation): every LRM shows a large wrong-vs-right effect (Cohen's d = 1.47-3.13 on H-ARC) while humans show the opposite sign. The comparison stays inside each agent's own scale; we never put seconds and tokens on one axis. The dissociation holds under item fixed effects, replicates across datasets, and is absent in a non-thinking baseline. We read the human pattern as engagement versus abandonment: people stay on items they expect to solve and give up on the rest. We read the LRM pattern as length driven by uncertainty: chains grow when the model is unsure, which is exactly when it tends to fail. Both policies produce the same cross-item correlation with difficulty, so they look aligned on the measure prior work has used; the divergence shows up only once item identity is fixed. Under resource-rational metareasoning, the split is between two stopping policies that share a difficulty signal but implement opposite control; trace length captures the signal and misses the control. 

---
# Adaptive Evaluation of Out-of-Band Defenses Against Prompt Injection in LLM Agents 

**Authors**: Praneeth Narisetty, Shiva Nagendra Babu Kore, Uday Kumar Reddy Kattamanchi, Jayaram Kumarapu  

**Link**: [PDF](https://arxiv.org/pdf/2606.26479)  

**Abstract**: Recent work (2024 to 2026) has converged on a strategy for defending tool-using LLM agents against indirect prompt injection: rather than training the model to refuse malicious instructions, enforce security outside the model with a deterministic policy that mediates the agent's actions. Systems such as CaMeL, FIDES, Progent, RTBAS, and FORGE realize this with capabilities, information-flow labels, and reference monitors, and several report near-elimination of attacks on the AgentDojo benchmark. We make two contributions. First, we organize these out-of-band defenses as instances of classical integrity protection (Biba), reference monitoring, and least privilege, yielding a structured comparison of what they do and do not cover. Second, we warn that every one of them is validated only on static benchmarks (a fixed set of injection attempts), the same methodology that made in-band defenses look strong until adaptive, defense-aware attacks broke twelve of them at over 90% success; we specify the threat model and protocol an adaptive evaluation requires. We then run that protocol as an independent reproduction and extension of Progent's own adaptive-attack analysis, on AgentDojo, with an open-weight agent (Qwen2.5-7B) self-hosted on a single H200, a setting its authors did not test. Averaged over three runs, the defense held: Progent cut mean attack success roughly sixfold (25.8% to 4.2%), and a hand-crafted adaptive attack did not raise it (2.6%). This is one small-scale data point on a weak model with a single black-box attack template; a stronger optimized (white-box GCG) attack remains open. The result is consistent with, but does not establish, the hypothesis that deterministic out-of-band enforcement is a harder target for an adaptive attacker than in-band detection. 

---
# Epiphany-Aware KV Cache Eviction Without the Attention Matrix 

**Authors**: Steven Kolawole, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2606.26472)  

**Abstract**: As reasoning models emit chains of thought tens of thousands of tokens long, KV cache increasingly becomes a deployment bottleneck. Existing cache eviction methods rank tokens by attention weight, which is a noisy importance proxy in long reasoning traces, and prohibits the use of fused kernels in production inference by forcing the model to materialize the attention matrix. In this work, we instead score tokens with a metric we term the epiphany score: the change in the model's internal representation, read directly from the forward pass with no attention matrix and negligible extra state. Our resulting cache eviction method, EpiKV, requires no training, classifier, or custom kernel, and can be used directly in FlashAttention inference stacks unchanged -- scaling to a 16x longer feasible context than attention-based scoring. upper-mid layers negatively) and remove a positional trend with a causal rolling z-score. At a 4096-token cache EpiKV reaches 72% on MATH-500, matching the strongest attention-based baseline (ThinKV 71%, H2O 67%); a lag-normalized KV variant reaches 37% on AIME-2024 at 8192 tokens against the best of them (33%), at up to 2.8x the speed. 

---
# DualEval: Joint Model-Item Calibration for Unified LLM Evaluation 

**Authors**: Aaron J. Li, Hao Huang, Youngmin Park, Yitong Ma, Wei-Lin Chiang, Li Chen, Cho-Jui Hsieh, Bin Yu, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2606.26429)  

**Abstract**: Current LLM evaluation relies on two complementary but often disconnected signals: static benchmarks with objective correctness labels and arena-style preference data that better reflect open-ended user interactions. We introduce DualEval, a latent model-item calibration framework that represents models and evaluation items in a shared space, jointly estimating model ability together with item difficulty and sharpness. We apply DualEval across four domains: coding, math, miscellaneous domain-knowledge tasks, and generic everyday user queries. Our evaluation uses 18 frontier LLMs, static benchmark labels, and reward-model scores validated against held-out human preferences for open-ended model responses. Empirically, our framework produces reliable and balanced model rankings, and its learned item-level profiles support downstream applications such as benchmark compression for sample-efficient evaluation and anomaly detection for contamination or outlier analysis. Overall, DualEval unifies static and arena-style evaluation through joint model-item calibration, producing model rankings and item-level diagnostics that support more sample-efficient, interpretable, and auditable evaluation pipelines. 

---
# Staying VIGILant: Mitigating Visual Laziness via Counterfactual Visual Alignment in MLLMs 

**Authors**: Xi Xiao, Chen Liu, Chih-Ting Liao, Yunbei Zhang, Qizhen Lan, Yuxiang Wei, Lin Zhao, Janet Wang, Jianyang Gu, Muchao Ye, Tianyang Wang, Hao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.26387)  

**Abstract**: Multimodal large language models (MLLMs) extend large language models (LLMs) with visual perception, enabling joint reasoning over images and text. Despite inheriting strong reasoning capabilities from LLMs, they remain prone to hallucinations that contradict their visual inputs. Mechanistic studies indicate that this weakness stems from visual laziness: MLLMs encode the correct visual evidence internally, but overly rely on strong language priors during response. Existing alignment methods, such as direct preference optimization, primarily optimize outcome-level rewards based on text. This introduces an optimization bias toward linguistic shortcuts, leading to responses that often contradict the visual evidence. To address this, we propose Visual Information Gain In aLignment (VIGIL), a reinforcement-learning (RL) post-training framework that shifts the focus from numerical reward fitting to causal visual grounding. VIGIL introduces a geometric constraint that explicitly maximizes the mutual information between the visual input and the generated response. We achieve this by penalizing "blind confidence" instances where the model remains improperly certain even when textual-visual attention is masked to create a counterfactual blind state. Extensive experiments show that VIGIL consistently outperforms recent alignment methods across hallucination and reasoning benchmarks without compromising text-only capabilities. Our approach matches the full-data performance of state-of-the-art methods using only 25% of the preference data and even demonstrates emergent spatial grounding capabilities without explicit bounding box supervision. 

---
# Narration-of-Thought: Inference-Time Scaffolding for Defeasible Ethical Reasoning in Large Language Models 

**Authors**: Patrick Cooper, Alvaro Velasquez  

**Link**: [PDF](https://arxiv.org/pdf/2606.26366)  

**Abstract**: Standard chain-of-thought on moral dilemmas exhibits two failure modes: stakeholder collapse (the trace names at most one party with a stake in the outcome) and uncertainty suppression (no explicit unknowns or hedges before committing to an action). We introduce narration-of-thought (NoT), a system prompt that structures chain-of-thought into five sections: protagonist, stakeholders, two-step consequences, uncertainty, then commitment. NoT adds no training, parameters, or fine-tuning. On 100 DailyDilemmas scenarios across four generators from three vendors, NoT cuts stakeholder collapse from up to 31% to under 1% and uncertainty suppression from up to 72% to 1-24% on every model. A matched-budget verbose-CoT control rules out token spend as the active ingredient; NoT retains Cliff's delta advantages of +0.79 to +0.90 on stakeholder count and +0.65 to +0.93 on uncertainty score for three of four generators, and a section ablation attributes each shift to its specific sub-instruction. Textual-gradient descent initialised at NoT improves the scaffold further; a cross-family training judge (different vendor from the generator) dominates an in-family one on every measured axis. Extended to a five-round multi-stakeholder debate protocol, the scaffold converts a 6% standoff into 95% full consensus on a calibration set and 100% combined convergence on a DailyDilemmas replication. The resulting traces externalise the stakeholders, consequences, and uncertainty grounding each commitment, providing an auditable substrate for dependable agentic deployment. 

---
# Axon: A Synthesizing Superoptimizer for Tensor Programs 

**Authors**: Akash Kothari, Shaowei Zhu, Daniel Kroening, Chungha Sung  

**Link**: [PDF](https://arxiv.org/pdf/2606.26344)  

**Abstract**: Writing high performance kernels for AI accelerators requires deep expertise in tiling, instruction selection, data layout, and operator fusion placing a significant burden on programmers. In this paper, we focus on tile based AI accelerator programs and present Axon, a synthesizing superoptimizer for tensor programs: it uses program synthesis to automatically generate target instructions from semantics specifications, and explores semantically equivalent program variants to select the best performing kernel empirically. Axon discovers algebraic transformations by propagating operators through computation graphs and uses SMT over unbounded tensors to guarantee that all transformations preserve semantics without requiring hand crafted rewrite rules. It then lowers tensor operations to target ISA instructions, explores tiling configurations constrained by hardware descriptions, and fuses operators and instructions to minimize memory traffic. 

---
# The Verification Horizon: No Silver Bullet for Coding Agent Rewards 

**Authors**: Binghai Wang, Chenlong Zhang, Dayiheng Liu, Jiajun Zhang, Jiawei Chen, Mouxiang Chen, Rongyao Fang, Siyuan Zhang, Xuwu Wang, Yuheng Jing, Zeyao Ma, Zeyu Cui  

**Link**: [PDF](https://arxiv.org/pdf/2606.26300)  

**Abstract**: A classical intuition holds that verifying a solution is easier than producing one. For today's coding agents, this intuition is being inverted: as foundation models develop stronger reasoning capabilities and engineering harnesses grow more sophisticated, generating complex candidate solutions is no longer difficult -- reliably verifying them has become the harder problem. Every verifier we can build is only a proxy for human intent, never the intent itself. This makes verification subject to a twofold difficulty: first, intent is underspecified by nature, making it inherently hard to faithfully check whether it has been fulfilled; second, during model training, optimization widens the gap between proxy and intent -- manifesting as reward hacking or signal saturation. To address this, we characterize the quality of verification signals along three dimensions -- scalability, faithfulness, and robustness -- and argue that achieving all three simultaneously is the central challenge. We further study four reward constructions: a test verifier for general coding tasks, a rubric verifier for frontend tasks, the user as verifier for real-world agent tasks, and an automated agent verifier for long-horizon tasks. Across different task types and policy capability levels, we conduct in-depth analysis and experiments on the core challenges of reward design and how to more effectively leverage reward signals. Experiments show that targeted verification design can effectively suppress reward hacking, improve task completion quality, and achieve significant gains across multiple internal and public benchmarks. These experiences collectively point to a core observation: no fixed reward function can remain effective as policy capability continues to grow; and verification must co-evolve with the generator. 

---
# From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations 

**Authors**: Dianjing Fan, Yao Li, Kyaw Hpone Myint, Dwipam Katariya, Alexandre G.R. Day, Pranab Mohanty, Giri Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2606.26277)  

**Abstract**: Sequential user behavior modeling is widely adopted in industrial recommender systems; however, significant gaps remain in financial services, where pre-login web interactions and authenticated in-app experiences differ drastically. Specifically, pre-login web users typically explore new products, whereas logged-in app users focus on account servicing. Due to the challenge of cross-channel entity resolution (e.g., matching anonymous web sessions to authenticated mobile accounts), web-based intent signals remain underutilized for post-authentication personalization. Existing methods for capturing web-based intent are often ad-hoc and narrow, lacking the flexibility to support both quantitative downstream recommendations and qualitative understanding at scale. In this work, we propose a scalable and dual-purpose intent prediction framework for web-based interactions and demonstrate its applicability for personalization. Our approach transforms raw web clickstreams into two outputs: a self-supervised Transformer encodes multi-modal clickstreams into a compact session embedding, while an LLM-based taxonomy generation and distillation pipeline produces interpretable intent labels. Our system demonstrates that self-supervised clickstream representations combined with LLM-distilled taxonomies can jointly serve quantitative tasks and qualitative understanding in production: on the mobile homepage tile ranking task, the session embedding improves macro Recall@1 by 1.88% and reduces Log Loss by 13.38% over production baselines. On the user conversion prediction task, the embedding outperforms the LLM labels by 4.3% on micro F1, while the distillation layer delivers interpretable labels at ultra-low latency with only a 7% performance drop. 

---
# Neural Speaker Diarization via Multilingual Training: Evaluation on Low-Resource Nepali-Hindi Speech 

**Authors**: Samip Neupane, Sandesh Pokhrel, Sandesh Pyakurel, Basanta Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2606.26144)  

**Abstract**: Speaker diarization, the task of determining "who spoke when" in a multi-speaker recording, is a critical component in applications such as meeting transcription, accessibility tools, and multilingual information retrieval. While end-to-end neural diarization systems have achieved strong performance for English and other high-resource languages, their effectiveness degrades substantially for underrepresented languages where annotated speech data is scarce.
This paper investigates speaker diarization for low-resource Nepali-Hindi speech through a multilingual training approach, comparing two modern architectures: EEND with encoder-decoder attractors (EEND-EDA) and EEND with Perceiver-based attractors (DiaPer). Both models are trained on a multilingual corpus combining English speech from LibriSpeech, diverse speaker recordings from VoxCeleb, and separately collected Nepali and Hindi audio, a setup designed to reduce language bias and encourage cross-lingual generalization. We evaluate both models across 2-speaker, 3-speaker, 4-speaker, and mixed-speaker scenarios on LibriSpeech, VoxCeleb, and Nepali-Hindi (NeHi) test sets. DiaPer achieves stronger overall performance than EEND-EDA, particularly in more challenging multi-speaker conditions, obtaining DERs of 3.28%, 2.02%, 4.05%, and 4.76% on NeHi 2-speaker, 3-speaker, 4-speaker, and mixed-speaker settings, respectively, compared to 1.50%, 9.68%, 16.17%, and 11.19% for EEND-EDA. These results demonstrate the viability of Perceiver-based end-to-end neural diarization for low-resource multilingual speech processing. 

---
