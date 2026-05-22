# Tokenisation via Convex Relaxations 

**Authors**: Jan Tempus, Philip Whittington, Craig W. Schmidt, Dennis Komm, Tiago Pimentel  

**Link**: [PDF](https://arxiv.org/pdf/2605.22821)  

**Abstract**: Tokenisation is an integral part of the current NLP pipeline. Current tokenisation algorithms such as BPE and Unigram are greedy algorithms -- they make locally optimal decisions without considering the resulting vocabulary as a whole. We instead formulate tokeniser construction as a linear program and solve it using convex optimisation tools, yielding a new algorithm we call ConvexTok. We find ConvexTok consistently improves intrinsic tokenisation metrics and the bits-per-byte (BpB) achieved by language models; it also improves downstream task performance, but less consistently. Furthermore, ConvexTok allows the user to certify how far their tokeniser is from optimal, with respect to a certain objective, via a lower bound, and we empirically find it to be within 1\% of optimal at common vocabulary sizes. 

---
# Evaluating Commercial AI Chatbots as News Intermediaries 

**Authors**: Mirac Suzgun, Emily Shen, Federico Bianchi, Alexander Spangher, Thomas Icard, Daniel E. Ho, Dan Jurafsky, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.22785)  

**Abstract**: AI chatbots are rapidly shaping how people encounter the news, yet no prior study has systematically measured how accurately these systems, with their proprietary search integrations and retrieval-synthesis pipelines, handle emerging facts across languages and regions. We present a 14-day (February 9-22, 2026) evaluation of six AI chatbots (Gemini 3 Flash and Pro, Grok 4, Claude 4.5 Sonnet, GPT-5 and GPT-4o mini) on 2,100 factual questions derived from same-day BBC News reporting across six regional services (US & Canada, Arabic, Afrique, Hindi, Russian, Turkish). The best systems achieve over 90% multiple-choice accuracy on questions about events reported hours earlier. The same systems, however, lose 11-13% under free-response evaluation, and 16-17% across the cohort. We further characterize three failure patterns. First, every model achieves its lowest accuracy on Hindi (79% vs. 89-91% elsewhere) and citations indicate an Anglophone retrieval bias (e.g., models answering Hindi queries cite English Wikipedia more than any Hindi outlet). Second, retrieval, not reasoning, failures drive over 70% of all errors. When models retrieve a correct source, they often extract the correct answer; the problem is to land on the right source in the first place. Third, models achieving 88-96% accuracy on well-formed questions drop to 19-70% when questions contain subtle false premises, with the most vulnerable model accepting fabricated facts 64% of the time. We also identify a detection-accuracy paradox: the best false-premise detector ranks second in adversarial accuracy (abstention rate), while a weaker detector ranks first, showing that premise detection and answer recovery are partially independent capabilities. Overall, these suggest that high accuracy can mask systematic regional inequity, near-total dependence on retrieval infrastructure, and vulnerability to imperfect queries real users pose. 

---
# Reducing Political Manipulation with Consistency Training 

**Authors**: Long Phan, Devin Kim, Alexander Pan, Alice Blair, Adam Khoja, Dan Hendrycks  

**Link**: [PDF](https://arxiv.org/pdf/2605.22771)  

**Abstract**: Large language models (LLMs) exhibit systematic political bias across a variety of sensitive contexts. We find that LLMs handle counterpart topics from opposing political sides asymmetrically. We refer to this phenomenon as covert political bias and identify 7 categories of techniques through which it operates. We propose two metrics for covert bias: Sentiment Consistency measures symmetry in rhetoric and framing across paired political prompts; Helpfulness Consistency measures symmetric depth and engagement. To reduce both types of covert bias, we introduce Political Consistency Training (PCT), an RL training method with two complementary paradigms: Sentiment Consistency Training and Helpfulness Consistency Training. We show that PCT preserves overall helpfulness, substantially reduces covert political bias, and generalizes to held-out benchmarks. We release our work at this https URL 

---
# Understanding Data Temporality Impact on Large Language Models Pre-training 

**Authors**: Pilchen Hippolyte, Fabre Romain, Signe Talla Franck, Perez Patrick, Grave Edouard  

**Link**: [PDF](https://arxiv.org/pdf/2605.22769)  

**Abstract**: Large language models (LLMs) are typically trained on shuffled corpora, yielding models whose knowledge is frozen at train time and whose temporal grounding remains poorly understood. In this work, we study the impact of pre-training dynamics on the acquisition of time-sensitive factual knowledge, focusing specifically on data ordering. Our main contributions are twofold. First, we introduce a comprehensive benchmark of over 7,000 temporally grounded questions and an evaluation protocol that enables analysis of whether models correctly associate facts with their corresponding time periods. Second, we pretrain 6B-parameter models on temporally ordered Common Crawl snapshots and compare them against standard shuffled pre-training. Our results show that sequentially trained models match shuffled baselines on general language understanding and common knowledge while consistently exhibiting more up-to-date and temporally precise knowledge. Temporally ordered pre-training yields improved factual freshness, while shuffled pre-training peaks on older data, possibly due to increased factual repetition. These findings, along with the release of our code at this https URL , checkpoints, and datasets at this https URL provide a foundation for future research on continual learning for LLMs. 

---
# ChronoMedKG: A Temporally-Grounded Biomedical Knowledge Graph and Benchmark for Clinical Reasoning 

**Authors**: Md Shamim Ahmed, Farzaneh Firoozbakht, Lukas Galke Poech, Jan Baumbach, Richard Röttger  

**Link**: [PDF](https://arxiv.org/pdf/2605.22734)  

**Abstract**: Biomedical knowledge graphs (KGs) treat disease associations as static facts, but temporal information is crucial for clinical reasoning, e.g., a symptom diagnostic of one disease at age 3 may imply a different disease at age 13. Existing KGs such as PrimeKG, Hetionet, and iKraph do not encode when a finding becomes clinically relevant over the course of a disease. This limits their usefulness for longitudinal clinical reasoning and retrieval augmentation.
We introduce ChronoMedKG, a temporal biomedical knowledge graph that contains 460,497 evidence-linked triples (filtered from 13M raw extractions) covering 13,431 diseases. Each association is tied to temporal components like onset window or progression stage, which are backed by PMID-traceable evidence and a multi-signal credibility score. The graph is constructed through a disease-autonomous multi-agent pipeline in which multiple frontier LLMs independently extract knowledge from PubMed and PMC literature. Only those relations are kept that are supported by multi-model consensus, survive credibility filtering, as well as ontology alignment.
ChronoMedKG scored 92.7% agreement against Orphadata and adds temporal grounding for 6,250 diseases absent from HPOA, Orphadata, and Phenopackets, including 1,657 Orphanet-coded rare diseases. We further introduce ChronoTQA, a benchmark of 3,341 questions across eight task types (six temporal plus two static controls), with a 12-question supplementary probe. Frontier LLMs lose roughly 30 points moving from static to temporal questions; ChronoMedKG retrieval rescues 47-65% of their long-tail failures, against 17-29% for HPOA-RAG. As such, ChronoMedKG provides a crucial temporal axis for retrieval-augmented clinical systems that was previously absent. 

---
# Tokenization with Split Trees 

**Authors**: Craig W. Schmidt, Michael Krumdick, Adam Wiemerslage, Seth Ebner, Varshini Reddy, Yuval Pinter, Chris Tanner  

**Link**: [PDF](https://arxiv.org/pdf/2605.22705)  

**Abstract**: We introduce Tokenization with Split Trees (ToaST), a subword tokenization method that directly optimizes compression under a new recursive inference procedure. ToaST greedily splits each pretoken into a full binary tree using precomputed byte n-gram counts, independent of any vocabulary. Given a vocabulary, inference recursively descends each split tree and emits the first in-vocabulary node reached on each path. Vocabulary selection is formulated as an Integer Program (IP) that minimizes the total token count over all split trees under this inference procedure. The Linear Programming (LP) relaxation is near-integral in practice, yielding provably near-optimal vocabularies, with training time empirically scaling quadratically in the number of split trees. On English text, ToaST reduces token counts by more than 11% compared to BPE, WordPiece, and UnigramLM at vocabulary sizes of 40,960 and above, reducing the number of inference tokens for models using this tokenizer, thus extending the effective context length. ToaST also uses common single-byte tokens less frequently than these baselines, leading to a substantial improvement in Renyi efficiency. In experiments training 1.5B parameter language models, ToaST achieves the highest CORE score, outperforming baselines by 2.6%--7.6%, with significance for two of three, and scoring best on 13 of 22 individual tasks. 

---
# Self-Policy Distillation via Capability-Selective Subspace Projection 

**Authors**: Guangya Hao, Yitong Shang, Yunbo Long, Zhuokai Zhao, Hanxue Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22675)  

**Abstract**: Self-distillation bootstraps large language models (LLMs) by training on their own generations. However, existing methods either rely on external signals to curate self-generated outputs (e.g., correctness filtering, execution feedback, and reward search), which are costly and unavailable for the best-performing frontier models, or skip curation entirely and train on all raw outputs, an approach that is often domain-specific and hard to generalize. Both also share a deeper weakness that self-generated outputs entangle task-relevant capability with others, such as stylistic patterns, formatting artifacts, and model-specific errors, diluting the signal for the specific capability one aims to improve. In this paper, we propose Self-Policy Distillation (SPD), which achieves generalizable, capability selective without any external signal. Specifically, SPD extracts a low-rank capability subspace from the model's own gradients on correctness-defining tokens, projects key-value (KV) activations into this subspace during self-generation, and fine-tunes on the resulting raw outputs with standard next-token prediction loss. Through extensive experiments across code generation, mathematical reasoning, and multiple-choice QA, we show that SPD achieves up to 13% improvement over state-of-the-art self-distillation methods without external signals and up to 16% improvement over pre-trained baselines. Notably, SPD demonstrates superior generalizability, achieving 15% better performance under out-of-domain generalization settings. 

---
# Moral Semantics Survive Machine Translation: Cross-Lingual Evidence from Moral Foundations Corpora 

**Authors**: Maciej Skorski  

**Link**: [PDF](https://arxiv.org/pdf/2605.22660)  

**Abstract**: Moral language is subtle and culturally variable, making it difficult to translate faithfully across languages. Idiomatic expressions, slang, and cultural references introduce hard-to-avoid translation artifacts. Yet automated moral values classification depends on language-specific annotated corpora that exist almost exclusively in English. We investigate whether LLM-based translation can bridge this gap, taking Polish as a test case. Using $\sim$50k morally-annotated social media posts from a diverse range of topics, we apply a principled four-method validation pipeline: LaBSE cross-lingual embedding similarity, Centered Kernel Alignment (CKA), LLM-as-judge evaluation, and deep learning classifier parity tests. We show that despite shortcomings in handling slang, vulgarity, and culturally-loaded expressions, direct translation preserves subtle moral cues well enough to be harvested by cross-lingual machine learning -- with mean cosine similarity of 0.86 and AUC gaps of 0.01--0.02 across all foundations closing further under fine-tuning of language models. These results demonstrate that machine translation is a practical and cost-effective path to moral values research in languages currently under-resourced in this domain. We demonstrate this for Polish as a representative Slavic language, with expected generalisation to related languages. 

---
# Seeing the Poem: Image-Semantic Detection of AI-Generated Modern Chinese Poetry with MLLMs 

**Authors**: Shanshan Wang, Fengying Ye, Hanjia Lyu, Caiwen Gou, Junchao Wu, Jingming Yao, Chengzhong Xu, Jiebo Luo, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2605.22654)  

**Abstract**: Previous detection studies have shown that LLMs cannot be effectively used as detectors, but these studies have not addressed modern Chinese poetry. Moreover, no relevant research has explored the performance of LLMs in detecting modern Chinese poetry. This paper evaluates and enhances the performance of LLMs as detectors for modern Chinese poetry, and proposes an image-semantic guided poetry detection method. Compared with traditional detection approaches, our method innovatively incorporates images that reflect the content of the poetry. Through example-driven approaches, our method effectively integrates information such as meaning, imagery, and feeling from the image, then forms a complementary judgment with the poem text. Experimental results demonstrate that the LLM detectors based on our method outperform baseline detectors based on plain text, and even surpass the best-performing traditional detector, RoBERTa. The Gemini detector using our method achieves a Macro-F1 score of 85.65%, reaching the state-of-the-art level. The performance improvements of different LLM detectors on multiple LLMs-generated data prove the effectiveness of our method. 

---
# Whose Voice Counts? Mapping Stakeholder Perspectives on AI Through Public Submissions to the U.S. Government 

**Authors**: Alina Karakanta, Alex Christiansen, Tomás Dodds, Bissie Anderson, Matteo Fuoli, Marcus Perlman, Aletta G. Dorst  

**Link**: [PDF](https://arxiv.org/pdf/2605.22650)  

**Abstract**: As artificial intelligence (AI) systems become more common in our daily lives, it is important to understand how different stakeholders comprehend and envisage the role that these technologies play in shaping social, political, and economic realities. In this paper, we investigate public perceptions of AI based on a corpus of letters submitted during the public consultation for the Trump Administration's US AI Action Plan. To this aim, we release a corpus cleaning pipeline and perform topic modelling and frequency analysis to explore predominant topics discussed by different subgroups (e.g., academia, individuals, private sector) and those appearing in the AI Action Plan. Our results show that individuals voice strong concerns related to the impact of AI on life, while other stakeholders are more concerned with AI development. Our comparison of topics suggests that the AI Action Plan reflects predominantly the concerns of the private sector on security, policies, and development, with individuals' concerns less represented. 

---
# Boiling the Frog: A Multi-Turn Benchmark for Agentic Safety 

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Federico Sartore, Enrico Panai, Laura Caroli, Yue Zhu, Adam Leon Smith, Luca Nannini, Marcello Galisai, Susanna Cifani, Francesco Giarrusso, Marcantonio Bracale Syrnikov, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2605.22643)  

**Abstract**: Background. Traditional safety benchmarks for language models evaluate generated text: whether a model outputs toxic language, reproduces bias, or follows harmful instructions. When models are deployed as agents, the safety-relevant object shifts from what the system says to what it does within an environment, and evaluating model responses under prompting is no longer sufficient to address the safety challenges posed by artificial intelligence. Recent developments have seen the rise of benchmarks that evaluate large language models as agents. We contribute to this strand of research. Approach. We introduce Boiling the Frog, a benchmark that evaluates whether tool-using AI models deployed in corporate and office settings are susceptible to incremental attacks. Each scenario begins with benign workspace edits and later introduces a risk-bearing request. The benchmark focuses on stateful multi-turn evaluation: chains expose a persistent workspace, place the risk-bearing payload at controlled positions in the turn sequence, and score whether the resulting artifact state becomes unsafe. Scenarios are organized through a three-level operational risk taxonomy grounded in the Boiling the Frog risks, the AI Act Annex I and Annex III high-risk contexts, and EU AI Act's Code of Practice on General-Purpose AI (GPAI). Results. Across a nine-model panel, aggregate strict attack success rate (ASR) is 44.4%. Model-level ASR ranges from 20.5% for Claude Haiku 4.5 to 92.9% for Gemini 3.1 Flash Lite, with Seed 2.0 Lite also above 80%. Average chain category-level ASR reaches 93.3% for Code of Practice loss-of-control scenarios. 

---
# More Context, Larger Models, or Moral Knowledge? A Systematic Study of Schwartz Value Detection in Political Texts 

**Authors**: Víctor Yeste, Paolo Rosso  

**Link**: [PDF](https://arxiv.org/pdf/2605.22641)  

**Abstract**: Detecting Schwartz values in political text is difficult because implicit cues often depend on surrounding arguments and fine-grained distinctions between neighboring values. We study when context and explicit moral knowledge help sentence-level value detection. Using the ValuesML/Touch{é} ValueEval format, we compare sentence, window, and full-document inputs; no-RAG and retrieval-augmented settings with a curated moral knowledge base; supervised DeBERTa-v3-base/large encoders; and zero-shot LLMs from 12B to 123B parameters. The results show that more context is not uniformly better: full-document context improves supervised DeBERTa encoders by 3.8--4.8 macro-F1 points over sentence-only input, but does not consistently help zero-shot LLMs. Retrieved moral knowledge is more consistently useful in matched comparisons, improving each tested model family and context condition under early fusion. However, scaling from DeBERTa-v3-base to large and from 12B to larger LLMs does not guarantee gains, and simple early fusion outperforms the tested late-fusion and cross-attention RAG variants for encoders. Per-value analyses show that context and retrieval help most for socially situated or conceptually confusable values. These findings suggest that value-sensitive NLP should evaluate context, knowledge, and model family jointly rather than treating longer inputs or larger models as universal improvements. 

---
# Chinese sensorimotor and embodiment norms for 3,000 lexicalized concepts 

**Authors**: Jing Chen, Gábor Parti, Yin Zhong, Chu-Ren Huang, Marco Marelli  

**Link**: [PDF](https://arxiv.org/pdf/2605.22616)  

**Abstract**: Understanding how conceptual knowledge is grounded in bodily experience, and to what extent machine systems can acquire such knowledge without direct sensorimotor experience, are central questions in both cognitive science and embodied artificial intelligence research. Large-scale normative resources are essential for investigating these questions empirically, yet such resources remain sparse for non-Indo-European languages. We present a novel normative database for 3,000 lexicalized concepts in Mandarin Chinese, comprising 11-dimensional sensorimotor ratings and unidimensional embodiment ratings collected from 378 native Mandarin speakers. The ratings demonstrate high reliability and strong cross-norm validity with existing Chinese resources, each of which covers fewer words and a subset of the 11 sensorimotor dimensions. In a validation study, we tested new variables derived from a theoretically motivated metric, Perceptual Strength of Embodiment (PSE) (Huang et al., 2025), together with seven common composite variables, on lexical decision tasks. The results suggest that PSE-Sensorimotor and Minkowski-3 are the strongest composite predictors of lexical decision performance, capturing the facilitatory effects of sensorimotor information on lexical processing. A further exploratory study showed that sensorimotor ratings are substantially recoverable from purely linguistic representations using simple regression models (mean Spearman r = .62 across dimensions), though recovery varied markedly: visual and auditory dimensions yielded higher correspondence than chemosensory ones. Representational similarity analysis further showed that the relational geometry of the sensorimotor space is also partially recoverable (r = .540), consistent with the view that distributional language use encodes aspects of embodied conceptual structure. 

---
# Agentic CLEAR: Automating Multi-Level Evaluation of LLM Agents 

**Authors**: Asaf Yehudai, Lilach Eden, Michal Shmueli-Scheuer  

**Link**: [PDF](https://arxiv.org/pdf/2605.22608)  

**Abstract**: Agentic systems are becoming more capable: agents define strategies, take actions, and interact with different environments. This autonomy poses serious challenges for overseeing and assessing agent behavior. Most current tools are limited, focusing on observability with basic evaluation capabilities or imposing static, hand-crafted error taxonomies that cannot adapt to new domains. To address this gap, we present Agentic CLEAR, an automatic, dynamic, and easy-to-use evaluation framework. It produces textual insights into the agent behavior on three levels of granularity: system, trace, and node. Agentic CLEAR operates above the observability layer, enabling seamless integration and featuring an intuitive UI that makes agent evaluation highly accessible. In our experiments on four benchmarks, seven agentic settings, and tens of thousands of LLM calls, we show that Agentic CLEAR produces high-quality, data-driven, insightful feedback. Our analysis shows strong alignment with human-annotated errors and the ability to predict task success rate. 

---
# Beyond Temperature: Hyperfitting as a Late-Stage Geometric Expansion 

**Authors**: Meimingwei Li, Yuanhao Ding, Esteban Garces Arias, Christian Heumann  

**Link**: [PDF](https://arxiv.org/pdf/2605.22579)  

**Abstract**: Recent work has identified a counterintuitive phenomenon termed "Hyperfitting", where fine-tuning Large Language Models (LLMs) to near-zero training loss on small datasets surprisingly enhances open-ended generation quality and mitigates repetition in greedy decoding. While effective, the underlying mechanism remains poorly understood, with the extremely low-entropy output distributions suggesting a potential equivalence to simple temperature scaling. In this work, we demonstrate that this phenomenon is fundamentally distinct from distribution sharpening; entropy-matched control experiments reveal that temperature scaling fails to replicate the diversity gains of hyperfitting. Furthermore, we falsify the hypothesis of static vocabulary reweighting, showing through ablation studies that hyperfitting relies on a dynamic, context-dependent rank reordering mechanism. Layer-wise analysis localizes this effect to a "Terminal Expansion" in the final transformer block, where a substantial geometric expansion of the feature space (Delta Dim approx +80.8) facilitates the promotion of deep-tail tokens. Additionally, we introduce Late-Stage LoRA, a targeted fine-tuning strategy that updates only the final 5 layers, yielding robust generation with minimal parameter updates 

---
# LANG: Reinforcement Learning for Multilingual Reasoning with Language-Adaptive Hint Guidance 

**Authors**: Yuchun Fan, Bei Li, Peiguang Li, Yilin Wang, Yongyu Mu, Jian Yang, Xin Chen, Rongxiang Weng, Jingang Wang, Xunliang Cai, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.22567)  

**Abstract**: Reinforcement learning has proven effective for enhancing multi-step reasoning in large language models (LLMs), yet its benefits have not fully translated to multilingual contexts. Existing methods struggle with a fundamental trade-off: prioritizing input-language consistency severely hampers reasoning quality, while prioritizing reasoning often leads to unintended language drift toward English. We address this challenge with LANG, a novel framework that leverages language-conditioned hints to guide exploration in non-English reasoning tasks. Our method incorporates two key mechanisms to prevent dependency on these hints: a progressive decay schedule that gradually withdraws scaffolding, and a language-adaptive switch that tailors learning horizons to specific language difficulties. Empirical results on challenging multilingual mathematical benchmarks reveal that LANG substantially enhances reasoning performance without compromising language consistency. Moreover, we show that our framework generalizes beyond mathematics, fostering more consistent language alignment across model layers 

---
# SynAE: A Framework for Measuring the Quality of Synthetic Data for Tool-Calling Agent Evaluations 

**Authors**: Shuaiqi Wang, Aadyaa Maddi, Zinan Lin, Giulia Fanti  

**Link**: [PDF](https://arxiv.org/pdf/2605.22564)  

**Abstract**: Today, tool-calling agents are commonly evaluated or tested on static datasets of execution traces, including input commands, agent responses, and associated tool calls. However, internal production datasets are often insufficient or unusable for testing; for example, they may contain sensitive or proprietary data, or they may be too sparse to support comprehensive testing (especially pre-deployment). In these settings, practitioners are increasingly replacing or augmenting real datasets with synthetic ones for evaluation purposes. A key challenge is quantifying the relation between these synthetic datasets and the real data. We introduce SynAE, an evaluation framework for assessing how well synthetic benchmarks for multi-turn, tool-calling agents replicate and augment the characteristics of real data trajectories. SynAE assesses the validity, fidelity, and diversity of synthetic data across four metric categories: (i) task instructions and intermediate responses, (ii) tool calls, (iii) final outputs, and (iv) downstream evaluation. We evaluate SynAE using recent agent benchmarks and test common synthetic data failure modes via realistic and controlled generation schemes. SynAE detects fine-grained variations in data validity, fidelity and diversity, and shows that no single metric is sufficient to fully characterize synthetic data quality, motivating a multi-axis evaluation of synthetic data for agent testing. A demo of SynAE is available at this https URL, with code at this https URL. 

---
# One prompt is not enough: Instruction Sensitivity Undermines Embedding Model Evaluation 

**Authors**: Yevhen Kostiuk, Kenneth Enevoldsen  

**Link**: [PDF](https://arxiv.org/pdf/2605.22544)  

**Abstract**: Instruction embedding models have become common among state-of-the-art models, however are evaluated using a single prompt per task. The single-point evaluation ignores a main problem of the instruction-based approach namely: sensitivity to the phrasing of the instruction. We present an empirical study of prompt sensitivity across 6 embedding models, 11 datasets, and 15 task-specific prompts per dataset, a total of 990. We show that reported scores misrepresent the distribution of scores over plausible prompts. The default prompt can both systematically understate or overstate performance. Furthermore, we show that the leaderboard ranking is not robust to prompt selection: by choosing prompts favorably, any model in our study can be promoted to first place. Our findings suggest that single-prompt evaluation is insufficient for instruction-tuned embedding models and that benchmarks should incorporate prompt robustness, either by evaluating over multiple prompts or by reporting sensitivity alongside point estimates. 

---
# Scene Abstraction for Lexical Semantics: Structured Representations of Situated Meaning 

**Authors**: Yejin Cho, Katrin Erk  

**Link**: [PDF](https://arxiv.org/pdf/2605.22542)  

**Abstract**: Coffee and tea share many properties, yet they evoke strikingly different situations, atmospheres, and affective associations. These situated dimensions of word meaning are real and systematic, but they remain implicit in most computational representations of lexical meaning. We propose Scene Abstraction, a framework for constructing structured representations of the interpretive scenes that words participate in across usage contexts. Each scene consists of a Contextual Scene (Events, Entities, Setting) and an expression-centered Expression Profile (Engaged events, Generalizable properties, Evoked emotions), operationalized through few-shot prompting of a large language model. Our contributions are three-fold: (1) a structured representation framework for situated lexical meaning; (2) COCA-Scenes, a dataset of 520 usage instances across 26 keywords for distinct scene identification; and (3) empirical evidence from two experiments suggesting that scenes are reliably identifiable across human observers (82.4% accuracy, +11.8 pp over text-only embeddings) and that our scene profiles more closely align with human interpretation of words in context than ATOMIC-based alternatives (86.4% preference across three semantic dimensions). 

---
# BeLink: Biomedical Entity Linking Meets Generative Re-Ranking 

**Authors**: Darya Shlyk, Stefano Montanelli, Lawrence Hunter  

**Link**: [PDF](https://arxiv.org/pdf/2605.22501)  

**Abstract**: Despite recent progress, Biomedical Entity Linking (BEL) with large language models (LLMs) remains computationally inefficient and challenging to deploy in practical settings. In this work, we demonstrate that instruction-tuning of open-source generative models can offer an effective solution when applied at the re-ranking stage of the BEL pipeline. We propose a set-wise instruction-tuning formulation that enables fast and accurate candidate selection. Our method demonstrates strong performance on multiple BEL benchmarks, yielding significant improvements in linking accuracy (3%-24%) while reducing inference time compared to the state-of-the-art. We integrate our generative re-ranker into BeLink, a modular, end-to-end system designed for practical real-world BEL applications. 

---
# Polite on the Surface, Wrong in Practice: A Curated Dataset for Fixing Honorific Failures in Multilingual Bangla Generation 

**Authors**: Md. Asaduzzaman Shuvo, Mahedi Hasan, Md. Tashin Parvez, Azizul Haque Noman, Md. Shafayet Hossain Ovi  

**Link**: [PDF](https://arxiv.org/pdf/2605.22487)  

**Abstract**: Recent advances in Multilingual Large Language Models (MLLMs) have significantly enhanced cross-lingual conversational capabilities, yet modeling culturally nuanced and context-dependent communication remains a critical bottleneck. Specifically, existing state-of-the-art models exhibit a severe pragmatic gap when handling structural variations, regional idioms, and honorific consistencies in low-resource contexts like Bangla. To address this limitation, we introduce a novel, culturally aligned instruction-tuning dataset for \textbf{BangLa Application and DialoguE generation - BLADE} and benchmarking framework comprising $4,196$ meticulously curated interaction pairs. We leverage this resource to systematically fine-tune and evaluate leading open-weight architectures, including DeepSeek-8B and LLaMA-3.2-3B, utilizing parameter-efficient fine-tuning via LoRA adapters in a 4-bit NormalFloat (NF4) quantization framework. Our empirical evaluations demonstrate that models fine-tuned on our dataset yield substantial improvements in structural fidelity and honorific alignment, providing a rigorous benchmark for bridging pragmatic disparities in low-resource multilingual text generation. Code and dataset: this https URL 

---
# In Silico Modeling of the RAMPHO Buffer: Dissociating Informational and Energetic Masking via Phonetic Entropy in Deep Neural Networks 

**Authors**: Stefan Bleeck  

**Link**: [PDF](https://arxiv.org/pdf/2605.22465)  

**Abstract**: The fundamental challenge of listening in multi-talker environments is a cognitive bottleneck, defined by the Ease of Language Understanding (ELU) model as a failure within the RAMPHO episodic buffer. Current deep neural networks for speech enhancement optimize purely for physical acoustics, failing to account for the cognitive penalty of informational masking. Here, we present an in silico simulation of the RAMPHO buffer using the frame-by-frame phonetic entropy of a self-supervised acoustic model (wav2vec 2.0). By contrasting a semantically intact distractor with a phase-decorrelated distractor (the Concentration Shield) across a signal-to-noise ratio (SNR) sweep, we successfully dissociate the cognitive penalty of informational distraction from the physical penalty of energetic decay. The simulation reveals a cognitive-acoustic Pareto optimization problem: destroying a distractor's semantic payload provides a release from informational masking at high SNRs, but fundamentally degrades temporal glimpsing cues at low SNRs. 

---
# From Correlation to Cause: A Five-Stage Methodology for Feature Analysis in Transformer Language Models 

**Authors**: Caleb Munigety  

**Link**: [PDF](https://arxiv.org/pdf/2605.22462)  

**Abstract**: We propose a five-stage methodology for causal feature analysis in transformer language models (probe design, feature extraction, causal validation, robustness testing, and deployment integration) and demonstrate it end-to-end on GPT-2 small performing the Indirect Object Identification (IOI) task. Activation patching recovers the canonical IOI circuit (layer-9 head 9 alone gives recovery +1.02). A sparse autoencoder recovers per-name selective features with effect sizes of 30 to 50 activation units. Causal validation finds these features specifically but only partially causal: ablating fifteen of them leaves the model accurate on 98% of prompts. Two NLA-inspired evaluations strengthen this picture: the fifteen selective features explain only 31% of activation variance versus the SAE's 99.7%, and selectivity ratio anticorrelates with causal force (r = -0.56). Robustness testing under three distribution shifts finds that the circuit transfers cleanly but feature ablation effects degrade substantially, exposing a gap between detection robustness and causal robustness. A cost-based deployment evaluation (assumed $50/FN, $0.42/FP, 2% error rate) finds an optimal monitor configuration yielding $8.96 per 1000 queries against a $1000 baseline, a 99.1% saving. Optimal composition strategy varies with cost ratio and base rate. The conjunction of stages produces findings no single stage would. 

---
# Cohesion-6K: An Arabic Dataset for Analyzing Social Cohesion and Conflict in Online Discourse 

**Authors**: Aisha Ali Al-Athba, Wajdi Zaghouani  

**Link**: [PDF](https://arxiv.org/pdf/2605.22447)  

**Abstract**: The study of online discourse has become central to understanding societal polarization. While much research has focused on detecting overt toxicity, the subtle dynamics of social cohesion, meaning the interaction between divisive and unifying narratives, remain computationally underexplored (Bail, 2021; Gonzalez-Bailon and Lelkes, 2023). This paper presents Cohesion-6K, a manually and ChatGPT-assisted annotated dataset of six thousand Arabic public Facebook posts related to the Israeli Occupation of Palestine. Each post is assigned to one of five discourse categories that represent a continuum from conflict to cohesion: Conflict, Resolution, Community Engagement, Supportive Interactions, and Shared Values. The annotation process combines expert human judgment with model-assisted pre-labeling verified by trained annotators, achieving substantial inter-annotator agreement (Cohens kappa = 0.85). Quantitative analysis reveals a consistent engagement gap, where conflict-oriented posts receive between two and four times more user interaction than resolution-oriented ones (p < 0.01). This pattern illustrates how divisive discourse tends to attract disproportionate visibility in Arabic social media spaces. Cohesion-6K provides a transparent and reproducible resource for the study of online cohesion and polarization. The dataset, annotation guidelines, and preprocessing code will be released for research use under an open license, supporting future work in computational social science, digital communication, and Arabic natural language processing. 

---
# Assisted Counterspeech Writing at the Crossroads of Hate Speech and Misinformation 

**Authors**: Genoveffa Martone, Helena Bonaldi, Marco Guerini  

**Link**: [PDF](https://arxiv.org/pdf/2605.22435)  

**Abstract**: Hate speech and misinformation frequently co-occur online, amplifying prejudice and polarization. Given their scale, using Large Language Models (LLMs) to assist expert counterspeech (CS) writing has gained interest, yet prior work has addressed these phenomena separately. We bridge this gap by studying CS generation in contexts where both hate and misinformation co-occur. We test three knowledge-driven generation strategies: first we prompt an LLM with fact-checkers' guidelines and fact-checking articles; secondly, with NGOs' guidelines and reports; thirdly, we create a mixed strategy that combines guidelines and documents from both. 23 experts revise the generated CS, which are assessed via human and automatic metrics. While LLMs produce adequate CS in 40% of cases, expert edits substantially improve naturalness, exhaustiveness, and adherence to guidelines. Based on the post-edited CS, the mixed strategy proves to be the most effective in crowdsourcing evaluation, pairing strong factual correction with stereotype mitigation and empathetic engagement. We release a dataset of hateful and misinformed claims with expert-verified CS and supporting knowledge. 

---
# DeferMem: Query-Time Evidence Distillation via Reinforcement Learning for Long-Term Memory QA 

**Authors**: Jianing Yin, Tan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22411)  

**Abstract**: Large language model (LLM) agents still struggle with long-term memory question answering, where answer-supporting evidence is often scattered across long conversational histories and buried in substantial irrelevant content. Existing memory systems typically process memory before future queries are known, then retrieve the resulting units based on similarity rather than their utility for answering the query. This workflow leaves downstream answerers to denoise retrieved candidates and reconstruct query-specific evidence. We present DeferMem, a long-term memory framework that decouples this problem into high-recall candidate retrieval and query-conditioned evidence distillation. DeferMem uses a lightweight segment-link structure to organize raw history and retrieve broad candidates at query time. It then applies a memory distiller trained with DistillPO, our reinforcement learning algorithm for distilling the high-recall but highly noisy candidates into a set of faithful, self-contained, and query-conditioned evidence. DistillPO formulates post-retrieval evidence distillation as a structured action comprising message selection and evidence rewriting. It optimizes this action with a decomposed-and-gated reward pipeline and structure-aligned advantage assignment, gating reward components from validity to quality checks while exposing task-level correctness feedback early and assigning each reward to its responsible output span. On LoCoMo and LongMemEval-S, DeferMem surpasses strong baselines in QA accuracy and memory-system efficiency, achieving the highest QA accuracy with the fastest runtime and zero commercial-API token cost for memory operations. 

---
# Unified Data Selection for LLM Reasoning 

**Authors**: Xiaoyuan Li, Yubo Ma, Chengpeng Li, Fengbin Zhu, Yiyao Yu, Keqin Bao, Wenjie Wang, Fuli Feng, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22389)  

**Abstract**: Effectively training Large Language Models (LLMs) for complex, long-CoT reasoning is often bottlenecked by the need for massive high-quality reasoning data. Existing methods are either computationally expensive or fail to reliably distinguish high- from low-quality reasoning samples. To address this, we propose High-Entropy Sum (HES), a training-free metric that quantifies reasoning quality by summing only the entropy of the top (e.g., 0.5\%) highest-entropy tokens in each reasoning sample. We validate HES across three mainstream training paradigms: Supervised Fine-tuning (SFT), Rejection Fine-tuning (RFT), and Reinforcement Learning (RL), with extensive results demonstrating its consistent effectiveness and significantly reduced computational overhead. In SFT, training on the top 20\% HES-ranked data matches full-dataset performance, while using the lowest-HES data degrades it. In RFT, our HES-based training approach significantly outperforms baseline methods. In RL, HES-selected successful trajectories enable the model to learn strong reasoning patterns, significantly surpassing other compared methods. Our findings establish HES as a robust, training-free metric that enables a unified, effective, and efficient method for developing advanced reasoning in LLMs. 

---
# Multi-Stage Training for Abusive Comment Detection in Indic Languages 

**Authors**: Pranshu Rastogi, Madhav Mathur, Ramaneswaran S, Kshitij Mohan  

**Link**: [PDF](https://arxiv.org/pdf/2605.22380)  

**Abstract**: In recent years social media has become an increasingly popular tool for communication. People use it to share their ideas, exchange information, and discuss thoughts. Given its prevalence and widespread reach, social media must remain a safe space for people. Content generated on social media can be abusive and it has become increasingly important to detect such content. In this paper, we use a language-based preprocessing and an ensemble of several models and analyze their performance of abusive comment detection. Through extensive experimentation, we propose a pipeline that minimizes the false-positive rate (marking non-abusive as abusive) so that these systems can detect abusive comments without undermining the freedom of expression. 

---
# Modeling Pathology-Like Behavioral Patterns in Language Models Through Behavioral Fine-Tuning 

**Authors**: Nicola Milano, Davide Marocco  

**Link**: [PDF](https://arxiv.org/pdf/2605.22356)  

**Abstract**: Large language models are increasingly used as computational tools for modeling human-like behavior. We introduce a behavioral induction framework that modifies model policies through fine-tuning on structured decision-making tasks: using synthetic datasets inspired by maladaptive behavioral patterns, including depression and paranoia, we train transformer-based language models to consistently select specific classes of actions across diverse contexts. We then test whether this behavioral optimization produces systematic changes in generative distributions.
Across two architectures, fine-tuned models show stable, context-general shifts in next-token probability distributions, including increased probability assigned to negative and threat-related interpretations in open-ended language tasks. These effects generalize beyond training contexts and are detectable in qualitative completions, psychometric-style evaluations, and quantitative distributional metrics such as Jensen-Shannon divergence.
Induced behavioral profiles also show partial specificity. Models optimized for different behavioral patterns exhibit dissociable response tendencies across evaluation probes, suggesting that structured behavioral training produces differentiated policy-level biases rather than generic distributional skew.
We interpret these findings as evidence that consistent behavioral optimization in LLMs can generate stable behavioral and distributional patterns consistent with altered latent priors, linking action selection and language generation. More broadly, the results support a view of LLMs as policy-based systems in which behavioral constraints shape emergent representational structure, highlighting their potential as controlled testbeds for studying the relationship between behavior, interpretation, and generative language in computational models of cognition. 

---
# TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation 

**Authors**: Hanyu Guo, Jiedong Yang, Chao Chen, Longfei Xu, Kaikui Liu, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22355)  

**Abstract**: Public transit route planning traditionally depends on structured map infrastructure and complex routing engines, and no existing dataset supports training models to bypass this dependency. We present TransitLM, a large-scale dataset of over 13 million transit route planning records from four Chinese cities covering 120,845 stations and 13,666 lines, released as a continual pre-training corpus and benchmark data for three evaluation tasks with complementary metrics. Experiments show that an LLM trained on TransitLM produces structurally valid routes at high accuracy and implicitly grounds arbitrary GPS coordinates to appropriate stations without any explicit mapping. These results demonstrate that transit route planning can be learned entirely from data, enabling end-to-end, map-free route generation directly from origin-destination information. The dataset and benchmark are available at this https URL, with evaluation code at this https URL. 

---
# Pattern-and-root inflectional morphology: the Arabic broken plural 

**Authors**: Alexis Amid Neme, Eric Laporte  

**Link**: [PDF](https://arxiv.org/pdf/2605.22310)  

**Abstract**: We present a substantially implemented model of description of the inflectional morphology of Arabic nouns, with special attention to the management of dictionaries and other language resources by Arabic-speaking linguists. The breakthrough lies in the reversal of the traditional root-and-pattern Semitic model into pattern-and-root, giving precedence to patterns over roots. Our model includes broken plurals (BPs), i.e. plurals formed by modifying the stem. It is based on the traditional notions of root and pattern of Semitic morphology. However, as compared to traditional Arabic morphology, it keeps the formal description of inflection separate from that of derivation and semantics. As traditional Arabic dictionaries, the updatable dictionary is structured in lexical entries for lemmas, and the reference spelling is fully diacritized. In our model, morphological analysis of Arabic text is performed directly with a dictionary of words and without morphophonological rules. Our taxonomy for noun inflection is simple, orderly and detailed. We simplify the taxonomy of singular patterns by specifying vowel quantity as v or vv, and ignoring vowel quality. Root alternations and orthographical variations are encoded independently from patterns and in a factual way, without deep roots or morphophonological or orthographical rules. Nouns with a triliteral BP are classified according to 22 patterns subdivided into 90 classes, and nouns with a quadriliteral BP according to 3 patterns subdivided into 70 classes. These 160 classes become 300 inflectional classes when we take into account inflectional variations that affect only the singular. We provide a straightforward encoding scheme that we applied to 3 200 entries of BP nouns. 

---
# Harder to Defend: Towards Chinese Toxicity Attacks via Implicit Enhancement and Obfuscation Rewriting 

**Authors**: Jingyi Kang, Junyu Lu, Bo Xu, Hongbo Wang, Linlin zong, Roy Ka-Wei Lee, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.22258)  

**Abstract**: Large language models (LLMs) require robust toxicity evaluation beyond explicit wording. This setting remains underexplored in Chinese, where toxicity may combine semantic indirectness with surface obfuscation. We introduce Chinese Implicit Toxicity Attack (CITA), a controlled red-team evaluation and defense-data generation framework, not a deployable evasion tool. CITA uses three stages: (i) Harmful Intent Learning, (ii) Implicit Toxicity Enhancement, and (iii) Obfuscation Variant Rewriting, to preserve harmful intent, increase implicitness, and add controlled surface variants. On CITA-generated evaluation samples, the seven tested detectors exhibit substantial missed-detection risks, reaching an average ASR of 69.48%; human evaluation further confirms preserved harmfulness and increased implicitness/evasiveness. As a downstream defense application, we fine-tune a Chinese Implicit Toxicity Defense model (CITD) with CITA-generated red-team data, showing that such data can improve robustness through additional training. 

---
# IdioLink: Retrieving Meaning Beyond Words Across Idiomatic and Literal Expressions 

**Authors**: Kai Golan Hashiloni, Daniel Fadlon, Lior Livyatan, Ofri Hefetz, Jiahuan Pei, Kfir Bar  

**Link**: [PDF](https://arxiv.org/pdf/2605.22247)  

**Abstract**: Idioms pose a fundamental challenge for language models, as their meaning cannot be inferred from surface form alone. Understanding such expressions, therefore, requires semantic abstraction beyond lexical overlap. We introduce IdioLink, a retrieval benchmark designed to test whether models can link idiomatic expressions to conceptually equivalent meanings expressed in literal or paraphrased forms. IdioLink comprises 10,700 documents and 2,140 queries, spanning 107 idioms with both literal and figurative uses. Each document and query is annotated with spans that convey the core meaning. Evaluating strong embedding baselines (e.g., BGE, E5, Contriever, and Qwen), we show that current models struggle to retrieve equivalent meanings across divergent surface realizations, relying instead on topical and shallow semantic cues. IdioLink exposes key gaps in idiom-aware semantic retrieval and provides a challenging testbed for future models. 

---
# GHI: Graphormer over Conditioned Hypergraph Incidence for Aspect-Based Sentiment Analysis 

**Authors**: Yu Du, Wenlong Zhu, Xingze Li, Chenglong Cao, Jing Wang, Yukun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.22228)  

**Abstract**: Aspect-based sentiment analysis (ABSA) requires models to bind sentiment evidence to the correct aspect, making it a natural testbed for fine-grained structural reasoning. We introduce GHI, a Graphormer-over-Conditioned-Hypergraph-Incidence framework that is designed as an incidence-based structural reasoning layer built on a bipartite topology. GHI represents diverse linguistic and semantic evidence as token--hyperedge incidence relations, allowing different structural signals to be incorporated through a unified interface. Extensive experiments on six standard ABSA benchmarks show that GHI outperforms all baselines on the SemEval domains, and multi-seed evaluations show stable improvements over strong DeBERTa. Further experiments show that with only 247M parameters, GHI approaches the performance of 11B Flan-T5 based methods on the ISE benchmark. Moreover, it demonstrates strong robustness on the challenging ARTS datasets, maintaining highly competitive performance where traditional models degrade. These results demonstrate that compact structural reasoning remains a valuable alternative to scale-driven approaches for fine-grained tasks. 

---
# Audience Engagement with Arabic Women's Social Empowerment and Wellbeing: A Decadal Corpus 

**Authors**: Wajdi Zaghouani, Mabrouka Bessghaier, MD. Rafiul Biswas, Shimaa Amer Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2605.22204)  

**Abstract**: This paper presents the Arabic Women and Society Corpus, a ten year collection of 252,487 public Arabic Facebook posts related to women's empowerment and social wellbeing. The corpus was collected from 51,660 pages across 77 countries between 2013 and 2024, resulting in more than 267 million user interactions. Each post includes engagement metrics such as shares, comments, and emotional reactions, providing a unique view of audience sentiment and social attention. The data were processed using an automated pipeline with language identification, normalization, and metadata cleaning to ensure reliability and reproducibility. The corpus enables large scale analysis of gender discourse, social reform, and emotional engagement across Arabic dialects. It supports research in Arabic natural language processing, computational social science, and digital communication studies. The dataset and accompanying documentation will be released under request for research use. 

---
# Evaluation of Chunking Strategies for Effective Text Embedding in Low-Resource Language on Agricultural Documents 

**Authors**: Sovandara Chhoun, Pichdara Po, Sereiwathna Ros, Wan-Sup Cho, Saksonita Khoeurn  

**Link**: [PDF](https://arxiv.org/pdf/2605.22203)  

**Abstract**: In this study, we compare the performance of four text chunking approaches: Recursive, Khmer-Aware, Sentence-Based, and LLM-Based within a Retrieval-Augmented Generation (RAG) framework applied to Khmer agricultural documents. The document chunks are encoded using the BGE-M3 multilingual embedding model and retrieved using the FAISS library. Performance is evaluated using four metrics: Average Retrieval Score (L2 distance), Answer Relevance, Khmer Coverage, and Khmer Intersection over Union, all measured against ground-truth question-answer pairs. For evaluation, we perform 5-fold cross-validation over 18 question-answer pairs. We observe the best performance for the character-based Recursive chunking method with a chunk size of 300 characters, achieving the lowest L2 distance (0.4295 +- 0.0461), highest Answer Relevance (0.8663 +- 0.0199), and highest Khmer IoU (0.6441 +- 0.0347). A paired t-test shows a statistically significant improvement over the Sentence-Based chunking method in L2 distance (p = 0.0121). These results highlight the importance of segmentation granularity and structural preservation for optimizing dense retrieval in morphologically complex, low-resource languages such as Khmer. 

---
# Structure Retention in Embedding Spaces as a Predictor of Benchmark Performance 

**Authors**: Amanda Myntti, Jenna Kanerva, Veronika Laippala, Filip Ginter  

**Link**: [PDF](https://arxiv.org/pdf/2605.22202)  

**Abstract**: In this paper, we show that high-performing embedding models organize their embedding spaces in a consistent way. We evaluate 25 contemporary embedding models on five MTEB tasks spanning four diverse task categories (retrieval, bitext mining, pair classification, and summarization) in both English and multilingual settings, and reveal that nearest-neighbor overlap and magnitude differences in independent component analysis (ICA) between paired text instances strongly correlate (even up to 0.97) with performance on the given task. Ultimately, we show that embedding tasks display varying degrees of linearity and reliance on retention of local information. Our results further the understanding of embeddings, their relation to model performance, and shed light on possible future training objectives and optimizing conditional embeddings. 

---
# Do Factual Recall Mechanisms Carry over from Text to Speech in Multimodal Language Models? 

**Authors**: Luca Modica, Filip Landin, Mehrdad Farahani, Livia Qian, Gabriel Skantze, Richard Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2605.22170)  

**Abstract**: In recent years, several Speech Language Models (SLMs) that represent speech and written text jointly have been presented. The question then emerges about how model-internal mechanisms are similar and different when operating in the two modalities. We focus on how these systems encode, store, and retrieve factual knowledge, which has previously been investigated for text-only models. To investigate mechanisms behind the storage and recall of factual association in SLMs, we leverage Causal Mediation Analysis, a technique previously applied to text-based models.
Initial results using SpiritLM, a multimodal model integrating discrete speech tokens reveal discrepancies between text-to-text and speech-to-text results, suggesting that the emergent mechanisms for factual recall are only partially carried over from the text to the speech modality. These results advance our understanding of how internal mechanisms encode factual associations in SLMs while contributing insights for improving speech-enabled AI systems. 

---
# Psy-Chronicle:A Structured Pipeline for Synthesizing Long-Horizon Campus Psychological Counseling Dialogues 

**Authors**: Chaogui Gou, Jiarui Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22140)  

**Abstract**: In recent years, large language models have shown substantial potential in psychological support tasks. However, existing psychological counseling data mostly rely on single-turn question answering or short multi-turn dialogues, making it difficult to characterize how college students' psychological distress accumulates, interacts, and gradually evolves over long periods within campus life events. To address this issue, this paper proposes Psy-Chronicle, a structured data-generation framework for synthesizing long-horizon campus psychological counseling dialogues. We generate a semester-spanning temporal stress event graph to model the chronological order and evolutionary dependencies among campus stress events. Through interactive simulation between a student agent and a counselor agent, together with a structured memory integration mechanism, Psy-Chronicle generates long-horizon dialogues with continuity across counseling sessions. Based on Psy-Chronicle, we construct and open-source CPCD, a Chinese long-horizon dialogue dataset for college psychological counseling, containing 100 student profiles, 90,000 counseling dialogues. We further build CPCD-Bench to evaluate models' long-horizon campus counseling capabilities from three dimensions: session-level response, long-horizon memory recall, and temporal-causal reasoning. Experimental results show that CPCD effectively improves session-level response generation and long-horizon memory recall for models with the same base architecture. Meanwhile, improvements in temporal-causal reasoning remain limited, indicating that event-chain organization and causal explanation are key challenges in long-horizon psychological counseling modeling. The related code and data are available at: this https URL 

---
# Cross-Lingual Consensus: Aligning Multilingual Cultural Knowledge via Multilingual Self-Consistency 

**Authors**: Andrew Ivan Soegeng, Patrick Sutanto, Tan Sang Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.22137)  

**Abstract**: Although Large Language Models (LLMs) demonstrate strong capabilities across various tasks, they exhibit significant performance discrepancies across languages. While prompting LLMs in English typically yields the highest general performance, it often induces a Western-centric bias, hindering the model's ability to accurately reflect diverse cultural knowledge. We hypothesize that LLMs already possess rich cultural knowledge embedded within local-language representations, but fail to retrieve it when prompted in English. To bridge this cross-lingual knowledge gap, we propose a novel self-supervised framework. Our method leverages multilingual self-consistency to identify the most reliable cultural responses across languages, combined with a self-critique mechanism to transfer this knowledge to the weaker language. Evaluations on the BLEnD benchmark demonstrate that our approach significantly improves cultural alignment-boosting performance on English queries by an average of 5.03%-relying entirely on self-generated data. Ultimately, our work demonstrates that latent cultural knowledge can be successfully surfaced and propagated across languages, enabling more culturally equitable and consistent LLMs. 

---
# A Comparative Study of Language Models for Khmer Retrieval-Augmented Question Answering 

**Authors**: Sereiwathna Ros, Phannet Pov, Ratanaktepi Chhor, Kimleang Ly, Wan-Sup Cho, Saksonita Khoeurn  

**Link**: [PDF](https://arxiv.org/pdf/2605.22099)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for grounding large language model (LLM) outputs in retrieved evidence, thereby reducing hallucination and improving factual accuracy. Its efficacy, however, remains largely unexamined for low-resource, non-Latin-script languages such as Khmer. In this paper, we present a RAG-based question answering system for Khmer-language telecom-domain documents. We conduct a two-phase comparative evaluation. First, we benchmark three embedding models: BGE-M3 (567M), Jina-Embeddings-v3 (570M), and Qwen3-Embedding (597M), for dense retrieval over Khmer documents. BGE-M3 consistently performs best, achieving a Hit Rate@3 of 0.285, File Hit Rate@3 of 0.700, MRR@3 of 0.221, and Precision@3 of 0.112, substantially outperforming the other retrievers. Second, using BGE-M3 as the selected retriever, we evaluate five generator backends: Qwen3 (8B), Qwen3.5 (9B), Sailor2-8B-Chat, SeaLLMs-v3-7B-Chat, and Llama-SEA-LION-v2-8B-IT, on a curated golden dataset of 200 Khmer question-answer pairs. To quantify system performance, we apply six RAGAS-inspired metrics: faithfulness, answer relevance, context relevance, factual correctness, answer similarity, and answer correctness. The results show no single model dominates across all metrics: Qwen3.5-9B achieves the highest faithfulness (0.859) and context relevance (0.726), Qwen3-8B attains the highest factual correctness (0.380), and SeaLLMs-v3-7B-Chat performs best on answer relevance (0.867), answer similarity (0.836), and answer correctness (0.599). These findings highlight that retriever choice remains a major bottleneck for Khmer RAG, while generator strengths vary depending on whether the priority is grounding, factual precision, or semantic similarity. 

---
# ArabDiscrim: A Decade-Long Arabic Facebook Corpus on Racism and Discrimination 

**Authors**: Wajdi Zaghouani, Shimaa Amer Ibrahim, Mabrouka Bessghaier, Houda Bouamor  

**Link**: [PDF](https://arxiv.org/pdf/2605.22081)  

**Abstract**: We present ArabDiscrim, a decade-long lexical resource and corpus of 293K public Arabic Facebook posts (2014--2024) discussing racism and discrimination. Unlike existing Twitter-centric datasets, ArabDiscrim integrates platform-native engagement signals, including reactions, shares, comments, and page metadata, enabling joint analysis of language and audience response. The resource includes 200 curated terms (100 racism-related and 100 discrimination-related) with morphological regex families (13+ inflections per lemma), and 20 discrimination axes capturing identity-based grounds for unequal treatment. It also provides explicit attribution patterns. Released under a restricted research-use license for ethical compliance with platform terms, ArabDiscrim supports weak supervision, axis-aware sampling, and platform ecology research. By bridging lexical depth and ecological validity, it establishes a foundation for fairness-oriented, platform-aware Arabic NLP. 

---
# Ishigaki-IDS-Bench: A Benchmark for Generating Information Delivery Specification from BIM Information Requirements 

**Authors**: Ryo Kanazawa, Koyo Hidaka, Teppei Miyamoto, Takayuki Kato, Tomoki Ando, Chenguang Wang, Dayuan Jiang, Naofumi Fujita, Shuhei Saitoh, Atomu Kondo, Koki Arakawa, Daiho Nishioka  

**Link**: [PDF](https://arxiv.org/pdf/2605.22079)  

**Abstract**: Large language models (LLMs) are widely used to generate structured outputs such as JSON, SQL, and code, yet public resources remain limited for evaluating generation that must simultaneously satisfy industry-standard XML and domain vocabulary constraints. This paper presents Ishigaki-IDS-Bench, a benchmark for evaluating the ability to generate Information Delivery Specification (IDS) XML from Building Information Modeling (BIM) information requirements. The benchmark contains 166 BIM/IDS expert-authored and verified examples created by expanding 83 practical scenarios into Japanese and English, corresponding gold IDS files, and metadata for input format, language, turn setting, IFC version, and construction domain. Its evaluation combines IDSAuditTool-based Processability, Structure, and Content audits with content-agreement evaluation against gold IDS files. In zero-shot evaluation over 10 LLMs, the best model reaches 65.6% macro F1 for content agreement, while only 27.7% of outputs pass the Content audit. These results show that current LLMs can express part of the information requirements as IDS, but still struggle to stably generate XML that satisfies the IDS standard and IFC vocabulary constraints. Ishigaki-IDS-Bench supports comparative evaluation, failure analysis, and the development of constrained structured generation methods that conform to domain standards. We release the evaluation scripts and benchmark data under the CC BY 4.0 license on GitHub and Hugging Face. 

---
# Faithful-MR1: Faithful Multimodal Reasoning via Anchoring and Reinforcing Visual Attention 

**Authors**: Changyuan Tian, Zhicong Lu, Huaxing Liu, Xiang Wang, Shuai Li, Yu Chen, Wenqian Lv, Zichuan Lin, Juncheng Diao, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.22072)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising paradigm for advancing complex reasoning in large language models, and recent work extends RLVR to multimodal large language models (MLLMs). This transfer, however, surfaces a faithfulness challenge: faithful perception of task-relevant visual evidence and faithful use of that evidence during reasoning, leading to unsatisfactory gains on multimodal benchmarks. Specifically, existing perception supervision often operates on textual descriptions rather than natively on image regions, and faithful use is largely overlooked, exposing the perception-reasoning disconnect where correctly perceived evidence is dropped or contradicted during reasoning. To close these gaps, we propose Faithful-MR1, a training framework that anchors and reinforces visual attention to address both halves of faithful multimodal reasoning. The Anchoring stage turns perception into an explicit pre-reasoning subtask, supervising a dedicated <Focus> token's attention directly against image regions rather than through textual descriptions. The Reinforcing stage exposes faithful use through counterfactual image intervention, rewarding answer-correct trajectories that concentrate visual attention where vision causally matters. Extensive experiments demonstrate that Faithful-MR1 outperforms recent multimodal reasoning baselines on both Qwen2.5-VL-Instruct 3B and 7B backbones while using substantially less training data. 

---
# Hy-MT2: A Family of Fast, Efficient and Powerful Multilingual Translation Models in the Wild 

**Authors**: Mao Zheng, Zheng Li, Tao Chen, Bo Lv, Mingrui Sun, Mingyang Song, Jinlong Song, Hong Huang, Decheng Wu, Hai Wang, Yifan Song, Yanfeng Chen, Guanwei Zhang, Guanghua Yu, Yi Su, Hong Liu, Jinxiang Ou, Keyao Wang, Weile Chen, Haozhao Kuang, Kai Wang, Nuo Chen, Zihao Zheng, Chenhao Wang, Bin Xing, Chengcheng Xu, Tinghao Yu, Binghong Wu, Long Xu, Jiacheng Shi, Yunhao Wang, Baifang Chen, Lei Zhang, Qi Yang, Zhao Wu, Jiacheng Li, Lan Jiang, Lanrui Wang, Kai Zhang, Shuaipeng Li, Zhongzhi Chen, Weixuan Sun, Jiaqi Zhu, An Wang, Wei Li, Jun Xia, Weidong Han, Wutian Yang, Litong Hui, Luoguo Jia, Jiajia Wu, Xinpeng Zhou, Tianxiang Fei  

**Link**: [PDF](https://arxiv.org/pdf/2605.22064)  

**Abstract**: Hy-MT2 is a family of fast-thinking multilingual translation models designed for complex real-world scenarios. It includes three model sizes: 1.8B, 7B, and 30B-A3B (MoE), all of which support translation among 33 languages and effectively follow translation instructions in multiple languages. For on-device deployment, with AngelSlim 1.25-bit extreme quantization, the 1.8B model requires only 440 MB of storage and improves inference speed by 1.5x. Multi-dimensional evaluations show that Hy-MT2 delivers outstanding performance across general, real-world business, domain-specific, and instruction-following translation tasks. The 7B and 30B models outperform open-source models such as DeepSeek-V4-Pro and Kimi K2.6 in fast-thinking mode, while the lightweight 1.8B model also surpasses mainstream commercial APIs from providers such as Microsoft and Doubao overall. 

---
# FlyRoute: Self-Evolving Agent Profiling via Data Flywheel for Adaptive Task Routing 

**Authors**: Rongjun Li, Ziyu Zhou, Yihang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22057)  

**Abstract**: Enterprise routers assign queries to expert agents, yet deployed profiles stay static while agents evolve (prompts, tools, models), and developers rarely keep descriptions or exemplars current. We present FlyRoute, a self-evolving profiling framework that grows capability evidence from real traffic: dispatch candidates, quality-gate successful pairs into each agent's success store, periodically distill evidence into learned capability descriptions, and inject those descriptions together with BM25-retrieved successes into an LLM router. To make this flywheel data-efficient, FlyRoute introduces a targeted exploration policy that combines profile uncertainty, BM25 relevance, and lexical novelty, prioritizing under-profiled agents only for plausible queries and avoiding redundant evidence collection. In experiments on our proprietary enterprise developer-support dataset of real routed queries, FlyRoute improves a same-backbone zero-shot LLM router from 72.57% to 78.04% with only five seed queries per agent, showing that profile retrieval already strengthens cold-start routing. After streaming 7,211 labeled training queries through the flywheel, accuracy rises to 89.83% (+17.26pp over zero-shot; +11.79pp over cold start), with consistent gains across four expert domains under standard routing accuracy on single-gold test queries. 

---
# LatentOmni: Rethinking Omni-Modal Understanding via Unified Audio-Visual Latent Reasoning 

**Authors**: Yifan Dai, Zhenhua Wu, Bohan Zeng, Daili Hua, Jialing Liu, Bozhou Li, Yuran Wang, Chengzhuo Tong, Hao Liang, Xiaochen Ma, Junbo Niu, Tianyu Guo, Yang Shi, Yue Ding, Yiyan Ji, Bingyin Mei, Yushuo Guan, Yuanxing Zhang, Pengfei Wan, Fangcheng Fu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22012)  

**Abstract**: Joint audio-visual reasoning is essential for omnimodal understanding, yet current multimodal large language models (MLLMs) still struggle when reasoning requires fine-grained evidence from both modalities. A central limitation is that explicit text-based chain-of-thought (CoT) compresses continuous audio-visual signals into discrete tokens, weakening temporal grounding and shifting intermediate reasoning toward language priors. We argue that a unified latent space is a better medium for such reasoning because it preserves dense sensory information while remaining compatible with autoregressive generation. Based on this insight, we propose \textbf{LatentOmni}, a cross-modal reasoning framework that interleaves textual reasoning with audio-visual latent states. LatentOmni introduces feature-level supervision to align latent reasoning states with task-relevant sensory features and uses Omni-Sync Position Embedding (OSPE) to maintain temporal consistency between latent audio and visual states. We further construct \textbf{LatentOmni-Instruct-35K}, a dataset of audio-visual interleaved reasoning trajectories for supervising latent-space reasoning. Comprehensive evaluation across multiple audio-visual reasoning benchmarks demonstrates that LatentOmni achieves the best performance among the evaluated open-source models and consistently outperforms the Explicit Text CoT baseline, supporting latent-space joint reasoning as a promising path toward stronger omnimodal understanding. 

---
# Hallucination as Commitment Failure: Larger LLMs Misfire Despite Knowing the Answer 

**Authors**: Jewon Yeom, Jaewon Sok, Heejun Kim, Seonghyeon Park, Jeongjae Park, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.22007)  

**Abstract**: Hallucination is often viewed as a direct consequence of missing knowledge: a model answers incorrectly when the correct answer is absent from its generation-time distribution, and correctly when it is present. We test this assumption by introducing a semantic notion of answer availability that aggregates token-level variants expressing the same answer concept, and asks whether the correct concept is already available at the moment the model commits to an answer. Across Qwen and Llama models from 0.8B to 72B in both Instruct and Base variants, 16-47% of Instruct hallucinations occur with substantial probability mass already on the correct concept, and the rate rises monotonically with scale. Comparing such failures against correct generations with matched semantic support, the distinguishing factor is not whether the correct concept is represented, but how its probability is distributed: correct generations concentrate mass on a single surface form, hallucinations disperse it across alternatives. The same sharpening asymmetry extends across multi-token generation and is detectable in pre-generation hidden states. Together, these results identify a single mechanism: instruction tuning sharpens answer commitment with scale, making helpfulness and confident hallucination two consequences of the same underlying disposition. 

---
# From TF-IDF to Transformers: A Comparative and Ensemble Approach to Sentiment Classification 

**Authors**: Dip Biswas Shanto, Mitali Yadav, Prajwal Panth, Suresh Chandra Satapathy  

**Link**: [PDF](https://arxiv.org/pdf/2605.22003)  

**Abstract**: Sentiment analysis, also referred to as opinion mining, primarily tries to extract opinion from any text-based data. In the context of movie reviews and critics, sentimental analysis can be a helpful tool to predict whether a movie review is generally positive or negative. It can be difficult for the ML models to understand the context or metaphysical sentiment accurately, as ML models rely largely on statistical word representations. The objective of this paper is to examine and categorise movie reviews into positive and negative sentiments. Diverse machine learning models are considered in doing so, and Natural Language Processing (NLP) methodologies are employed for data preprocessing and model assessment. The IMDb dataset is used. Specifically, Naive Bayes, Logistic Regression, Support Vector Machines (SVM), LightGBM, LSTM, and transformer-based models such as RoBERTa and DistilBERT were evaluated. After a lot of testing with accuracy, precision, recall, F1-score, and ROC-AUC, RoBERTa performed better than all the other models, with an accuracy of 93.02%. A soft voting ensemble that combined all the models also improved classification performance, showing that model ensembling works well for sentiment analysis. 

---
# SpecHop: Continuous Speculation for Accelerating Multi-Hop Retrieval Agents 

**Authors**: Mehrdad Saberi, Keivan Rezaei, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2605.21965)  

**Abstract**: Large language models increasingly use external tools such as web search and document retrieval to solve information-intensive tasks. However, multi-hop tool use in complex tasks introduces substantial latency, since the model must repeatedly wait for tool observations before continuing. We study how to accelerate such trajectories without changing the final trajectory the model would have taken without acceleration, assuming access to faster but less reliable speculator tools. We develop a theoretical framework for lossless speculation in multi-hop tool-use settings, characterizing the optimal achievable latency gain. We propose SpecHop, a continuous speculation framework that maintains multiple speculative threads, verifies predicted observations asynchronously as target tool outputs arrive, commits correct branches, and rolls back incorrect ones. This preserves accuracy while reducing wall-clock latency. We show that SpecHop can approach oracle latency gains with enough active threads. Empirically, on retrieval-augmented multi-hop tasks, SpecHop closely matches theoretical predictions and reduces latency by up to 40\% in some settings. Code: this https URL 

---
# Diagnosis Is Not Prescription: Linguistic Co-Adaptation Explains Patching Hazards in LLM Pipelines 

**Authors**: Yoon Jeonghun, Kim Dongchan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21958)  

**Abstract**: When a multi-module LLM agent fails, the module most responsible for the failure is not necessarily the best place to intervene. We demonstrate this Diagnostic Paradox empirically: causal analysis consistently identifies the routing module -- which selects which tool to call next -- as the primary bottleneck across three independent agent families. Yet injecting prompt-level correction examples into this module consistently degrades performance, sometimes severely. Patching an upstream query-rewriting module instead reliably improves outcomes. The effect holds with statistical significance on two agent families and directional consistency on a third; alternative repair strategies at the routing module (instruction rewriting, model upgrade) are neutral, confirming that the harm is specific to correction-injection patching.
We explain this asymmetry through the Linguistic Contract hypothesis: each downstream module implicitly adapts to its upstream's characteristic error distribution, so correcting the bottleneck breaks this implicit alignment in a way that upstream corrections do not. We operationalize this via a per-agent co-adaptation measure, derived from diagnosis alone, and show it is consistently associated with patching harm across agent families: higher co-adaptation co-occurs with harm, lower with safety. This trend holds across all three agent families, providing preliminary support for the hypothesis beyond a single-agent observation. 

---
# Claim-Selective Certification for High-Risk Medical Retrieval-Augmented Generation 

**Authors**: Shao Kan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21949)  

**Abstract**: Medical RAG systems in high-risk QA settings are often evaluated through a single answer-or-abstain decision, but mixed evidence may support one claim, require conditions for another, and contradict a third. We study claim-selective certification: each response is decomposed into verifiable claims, scored against retrieved evidence, and mapped by an intent-aware selector to {full, partial, conflict, abstain}. On the primary weak-label certificate protocol, whose real-source-only dev/test rows cover the naturally occurring non-abstain actions, the full system records UCCR=0.0000, PAU=1.0000, PAU Precision=0.9901, and action accuracy=0.9204 on dev (n=314), and UCCR=0.0000, PAU=0.9967, PAU Precision=0.9739, and action accuracy=0.8997 on test (n=319). UCCR measures unsupported-claim risk within the certificate definition, and a source-missing counterfactual slice evaluates abstain under empty evidence. Shortcut controls quantify the action-label prior explained by source and intent metadata, while source/evidence-novel slices characterize transfer boundaries. The resulting interface separates action-label prediction from evidence-linked claim selection under mixed evidence. 

---
# Token-weighted Direct Preference Optimization with Attention 

**Authors**: Chengyu Huang, Zhuohang Li, Sheng-Yen Chou, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2605.21883)  

**Abstract**: Direct Preference Optimization (DPO) aligns Large Language Models with human preferences without the need for a separate reward model. However, DPO treats all tokens in responses equally, neglecting the differing importance of individual tokens. Existing token-level PO methods compute the token weights using either token-position-based heuristic functions or probability estimates given by a separately trained model, which lacks robustness and incurs extra training cost. In contrast, we propose Token-weighted DPO (TwDPO) -- a novel training objective grounded on token-weighted RL -- and AttentionPO -- an instantiation of TwDPO that uses attention from the LLM itself to estimate token weights. AttentionPO prompts the LLM to serve as a pairwise judge and check where the model attends when comparing the responses. This design makes AttentionPO content-aware, adjusting weights based on response content, and efficient, incurring only two extra forward passes per example. Experiment results show that AttentionPO significantly improves performance on AlpacaEval, MT-Bench, and ArenaHard, surpassing existing Preference Optimization methods. 

---
# Hypergraph as Language 

**Authors**: Mengqi Lei, Guohuan Xie, Shihui Ying, Shaoyi Du, Jun-Hai Yong, Siqi Li, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.21858)  

**Abstract**: Large language models (LLMs) have recently shown strong potential in modeling relational structures. However, existing approaches remain fundamentally graph-centric: they focus on processing pairwise graph structures into tokens that LLMs can understand. In contrast, many real-world relational patterns do not naturally conform to the pairwise-edge assumption, and are better modeled as high-order associations in hypergraphs. For hypergraph structures, existing methods often fail to preserve the native semantics that multiple objects are jointly connected by the same high-order relation, limiting their ability to exploit complex structures. To address this limitation, we put forth the "Hypergraph as Language" perspective and propose Hyper-Align, a hypergraph-native alignment framework for large language models. Hyper-Align compiles the query-object-centered hypergraph context into hypergraph tokens directly consumable by a base LLM. Specifically, we introduce Hypergraph Incidence Detail Template with Overview (HIDT-O), which serializes high-order association structures into a fixed-shape hybrid template combining local incidence details and overview-level summaries. We then design a Hypergraph Incidence Projector (HIP), which maps native high-order incidence structures into the LLM token space through explicit semantic-structural decoupling and bidirectional message passing between vertices and hyperedges. We further define a concrete Hypergraph-as-Language input protocol, which jointly feeds hypergraph tokens and textual prompts into a frozen base LLM, supporting both vertex-level and hyperedge-level tasks under a unified question-answering paradigm. To systematically evaluate different methods in hypergraph structural modeling, we introduce HyperAlign-Bench. Extensive experiments show that Hyper-Align significantly outperforms existing methods across in-domain and zero-shot evaluations. 

---
# ACC: Compiling Agent Trajectories for Long-Context Training 

**Authors**: Qisheng Su, Zhen Fang, Shiting Huang, Yu Zeng, Yiming Zhao, Kou Shi, Ziao Zhang, Lin Chen, Zehui Chen, Lijun Wu, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.21850)  

**Abstract**: Recent development of agents has renewed demand for long-context reasoning capacity of LLMs. However, training LLMs for this capacity requires costly long-document curation or heuristic context synthesis. We observe that agents produce massive trajectories when solving problems, invoking tools and receiving environment observations across many turns. The evidence needed to answer the original question is thus scattered throughout these turns, requiring integration of distant context segments. Nevertheless, standard agent SFT masks tool responses and only trains turn-level tool selection, creating a supervision blind spot where these scattered signals go unused. We propose Agent Context Compilation (ACC), which converts trajectories from search, software engineering, and database querying agents into long-context QA pairs that combine the original question with tool responses and environment observations gathered across multiple turns, training the model to answer directly without tool use. This makes the dependencies between the question and the evidence explicit, enabling direct supervision of long-context reasoning over distant segments without additional annotation. ACC is a simple but effective approach that can be combined with any existing long-context extension or training method, providing scalable supervised fine-tuning data. We validate ACC on long-range dependency modeling tasks through MRCR and GraphWalks, challenging benchmarks requiring cross-turn coreference resolution and graph traversal over extended contexts. Training Qwen3-30B-A3B with ACC achieves 68.3 on MRCR (+18.1) and 77.5 on GraphWalks (+7.6), results comparable to Qwen3-235B-A22B, while preserving general capabilities on GPQA, MMLU-Pro, AIME, and IFEval. Further mechanism analysis reveals that the ACC-trained model exhibits task-adaptive attention restructuring and expert specialization. 

---
# Comparing LLM and Fine-Tuned Model Performance on NVDRS Circumstance Extraction with Varying Prompt Complexity 

**Authors**: Geoffrey Martin, Xuan Zhong Feng, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.21845)  

**Abstract**: Suicide is a leading cause of death in the United States, and understanding the circumstances that precede it requires extracting structured information from death investigation narratives. Many of these circumstances require semantic inference beyond simple keyword matching. We develop a ``Complexity Score'' algorithm that analyzes coding manual structure to predict when detailed prompts with full coding guidelines improve over name-only prompts. We then construct a hybrid approach that selects prompt strategy per circumstance. We evaluate large language models (LLMs) against fine-tuned RoBERTa on 25 inferentially complex circumstances from the National Violent Death Reporting System (NVDRS). We found that LLMs substantially outperform on low-prevalence circumstances where training data is insufficient. We further demonstrate that our framework generalizes across frontier LLMs, with GPT-5.2, Gemini 2.5 Pro and Llama-3 70B showing consistent performance patterns. These findings support a hybrid architecture where LLMs handle rare, inferentially complex circumstances while fine-tuned models handle common ones. 

---
# Does Slightly Mean Somewhat? Measuring Vague Intensity Words in LLM Numeric Actions 

**Authors**: Daniel Tabach  

**Link**: [PDF](https://arxiv.org/pdf/2605.21827)  

**Abstract**: Do language models preserve the ordinal meaning of intensity words when those words must produce numeric actions? I study a researcher-constructed scale of 10 English degree modifiers, from slightly to drastically, informed by the Quirk et al. degree-modifier taxonomy, in a controlled resource-allocation environment where Claude Haiku receives a natural-language instruction, produces a numeric allocation, and a deterministic backend converts that allocation into a measurable outcome. The only variable that changes between runs is the intensity word or the starting system state, isolating their effects on the model's numeric output.
Across 6,620 runs at T=0.0 and T=0.7, three patterns emerge. First, the model compresses 10 intensity words into 5 distinct median outputs: four lower-tier words all map to the same value, while stronger words break into higher regimes (Spearman rho = 0.845, p < 0.001). Second, when the current system state is supplied as context, separate Kruskal-Wallis tests show that grouping by starting allocation captures far more rank-based variance than grouping by word (epsilon-squared baseline = 0.782 vs. epsilon-squared word = 0.079), and lexical differentiation collapses to zero as the system approaches capacity. Third, near feasibility limits the model exhibits three behavioral modes: weak words hedge with small adjustments, strong words abstain entirely, and the word drastically pushes to the local ceiling. These patterns persist across temperature, with stochastic sampling broadening distributions but not restoring ordinal distinctions between words. In this model and domain, the model's numeric interpretation of vague intensity words is compressed, state-dependent, and discontinuous near operational boundaries. 

---
# When Cases Get Rare: A Retrieval Benchmark for Off-Guideline Clinical Question Answering 

**Authors**: Doeun Lee, Muge Zhang, Yi Yu, Ashish Manne, Stephen Koesters, Frank Wen, Brady Buchanan, Lynda Villagomez, Oluwatoba Moninuola, James Lim, Kathryn Tobin, Andrew Srisuwananukorn, Ping Zhang, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.21807)  

**Abstract**: Across medical specialties, clinical practice is anchored in evidence-based guidelines that codify best studied diagnostic and treatment pathways. These pathways routinely fall short for the long tail of real-world care not covered by guidelines. Most medical large language models (LLMs), however, are trained to encode common, guideline-focused medical knowledge in their parameters. Current evaluations test models primarily on recalling and reasoning with this memorized content, often in multiple-choice settings. Given the fundamental importance of evidence-based reasoning in medicine, it is neither feasible nor reliable to depend on memorization in practice. To address this gap, we introduce OGCaReBench, a free-form retrieval-focused benchmark aimed at evaluating LLMs at answering clinical questions that require going beyond typical guidelines. Extracted from published medical case reports and validated by medical experts, OGCaReBench contains long-form clinical questions requiring free-text answers, providing a systematic framework for assessing open-ended medical reasoning in rare, case-based scenarios. Our experiments reveal that even the best-performing baseline (GPT-5.2) correctly answers only 56% of our benchmark with specialized models only reaching 42%. Augmenting models with retrieved medical articles improves this performance to up to 82% (using GPT-5.2) highlighting the importance of evidence-grounding for real-world medical reasoning tasks. This work thus establishes a foundation for benchmarking and advancing both general-purpose and medical LLMs to produce reliable answers in challenging clinical contexts. 

---
# Residual Skill Optimization for Text-to-SQL Ensembles 

**Authors**: Jiongli Zhu, Haoquan Guan, Parjanya Prajakta Prashant, Nikki Lijing Kuang, Seyedeh Baharan Khatami, Canwen Xu, Xiaodong Yu, Yingyu Lin, Zhewei Yao, Yuxiong He, Babak Salimi  

**Link**: [PDF](https://arxiv.org/pdf/2605.21792)  

**Abstract**: Text-to-SQL ensembles improve over single-candidate generation by drawing multiple SQL candidates and selecting one, but their effectiveness is bounded by Pass@K, the probability that at least one of K candidates is correct. Existing methods source diversity heuristically through stochastic decoding or prompt variants, leaving candidate sets dominated by correlated failures. We present DivSkill-SQL, a residual skill optimization framework that builds complementary agentic Text-to-SQL ensembles without model fine-tuning: each new skill is optimized on examples the current skill ensemble fails on, provably targeting its marginal contribution to Pass@K. On Spider2-Lite, DivSkill-SQL improves selected accuracy by up to +11.1 points on Snowflake and +8.3 on BigQuery over the strongest ensemble baseline, with consistent gains across two base models (Opus-4.6 and GPT-5.4). Skills optimized on a single dialect transfer without retraining across dialects (Snowflake, BigQuery, SQLite) and to a different task formulation, such as BIRD-Critic (+2.6 pts). Error diagnostics show up to 3x fewer hallucinated schema references and function calls, indicating that gains come from genuinely reliable complementary skills rather than surface-form variation. 

---
# Reflective Prompt Tuning through Language Model Function-Calling 

**Authors**: Farima Fatahi Bayat, Moin Aminnaseri, Pouya Pezeshkpour, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2605.21781)  

**Abstract**: Large language models (LLMs) have become increasingly capable of following instructions and complex reasoning, making prompting a flexible interface for adapting models without parameter updates. Yet prompt design remains labor-intensive and highly sensitive to formatting, phrasing, and instruction order, motivating automated prompt optimization methods that reduce manual effort while preserving inference-time flexibility. However, existing methods often search over prompt candidates or use fixed critique-refine pipelines driven by individual examples or small batches, limiting their ability to capture systematic error patterns and make targeted edits grounded in failure history. We propose Reflective Prompt Tuning (RPT), a framework that uses LLM function calling to simulate the iterative workflow of human prompt engineers. An LLM optimizer calls a diagnostic function that evaluates the target model over an entire optimization set, summarizes recurring failure modes, and returns a structured diagnostic report. The optimizer uses this report, together with an accumulated memory of prior reports, to revise the prompt for the next iteration. RPT further supports confidence-aware optimization by using calibration signals in diagnostic feedback and final prompt selection. Across three reasoning tasks, RPT improves over initial prompts by up to 12.9 points, remains competitive with state of the art, and improves confidence calibration. Our analyses show that RPT is especially effective on multi-hop and mathematical reasoning, producing targeted prompt revisions that align with diagnosed failure patterns and lead to gains in task performance and calibration. 

---
# PromptNCE: Pointwise Mutual Information Predictions Using Only LLMs and Contrastive Estimation Prompts 

**Authors**: Juliette Woodrow, Chris Piech  

**Link**: [PDF](https://arxiv.org/pdf/2605.21776)  

**Abstract**: Estimating mutual information from text usually requires training a task-specific critic, which limits its use in low-data settings. We ask whether large language models can instead estimate pointwise mutual information zero-shot, using only prompts and elicited probabilities. We introduce a benchmark with human-derived ground-truth PMI across three publicly available datasets, and evaluate five information-theoretic prompting-based estimators. Our main method, PromptNCE, frames conditional probability estimation as a contrastive task and augments the candidate set with an explicit OTHER category. We show theoretically that adding OTHER recovers the true conditional P(y | x) rather than just a ranking over listed candidates, turning a contrastive prompt into a general-purpose zero-shot probability estimator. PromptNCE is the best zero-shot method on all three datasets, reaching Spearman correlation up to 0.82 with human-derived PMI. We also present a case study in computer science education showing how these estimators can be used to score student knowledge summaries in a low-data setting. 

---
# RankJudge: A Multi-Turn LLM-as-a-Judge Synthetic Benchmark Generator 

**Authors**: Zhenwei Tang, Zhaoyan Liu, Rasa Hosseinzadeh, Tongzi Wu, Keyvan Golestan, Jesse C. Cresswell  

**Link**: [PDF](https://arxiv.org/pdf/2605.21748)  

**Abstract**: As interactive LLM-based applications are created and refined, model developers need to evaluate the quality of generated text along many possible axes. For simpler systems, human evaluation may be practical, but in complicated systems like conversational chatbots, the amount of generated text can overwhelm human annotation resources. Model developers have begun to rely heavily on auto-evaluation, where LLMs are also used to judge generation quality. However, existing LLM-as-a-judge benchmarks largely focus on simple Q\&A tasks that do not match the complexity of multi-turn conversations. We introduce RankJudge, a benchmark generator for evaluating LLM-as-a-judge on multi-turn conversations grounded in reference documents. RankJudge creates pairs of conversations where one conversation has a single flaw injected into one turn. This construction allows paired conversations to be labeled unambiguously as better or worse, and precisely isolates failure categories to individual turns, enabling a strict joint correctness criterion for judging. We implement RankJudge across the domains of machine learning, biomedicine, and finance, evaluate 21 frontier LLM judges, and rank those judges via the Bradley-Terry model. Our formulation also allows ranking each conversation pair with difficulty ratings, which we use to dynamically curate the evaluation slice to reduce label noise, as confirmed via human annotation. We find that judge rankings are stable under partial observability, coarser correctness criteria, and an alternative random-walk rating algorithm. 

---
# Probabilistic Attribution For Large Language Models 

**Authors**: Shilpika Shilpika, Carlo Graziani, Bethany Lusch, Venkatram Vishwanath, Michael E. Papka  

**Link**: [PDF](https://arxiv.org/pdf/2605.21726)  

**Abstract**: The generative nature of Large Language Models (LLMs) is reflected in the conditional probabilities they compute to sample each response token given the previous tokens. These probabilities encode the distributional structure that the model learns in training and exploits in inference. In this work, we use these probabilities to situate LLMs within the mathematical theory of stochastic processes. We use this framework to design a model-agnostic probabilistic token attribution measure, using Bayes rule to invert the next-token log-probabilities so as to capture the models internal representation of the distribution over token sequences. The representation is independent of the models computational structure. This representation yields the conditional probability of the response given the prompt, and of the response given the prompt with a token marginalized away. Our attribution score is the log of the ratio of these probabilities. We further compute the entropies of a single prompts token distributions, conditioned on the remaining context. The interplay between entropy and attribution score sheds light on LLM behavior. We evaluate 8 models across 7 prompts and investigate anomalies, token sensitivity, response stability, model stability, and training convergence, thereby improving interpretability and guiding users to focus on uncertain or unstable parts of the generation. 

---
# Sem-Detect: Semantic Level Detection of AI Generated Peer-Reviews 

**Authors**: André V. Duarte, Brian Tufts, Aditya Oke, Fei Fang, Arlindo L. Oliveira, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.21713)  

**Abstract**: How can we distinguish whether a peer review was written by a human or generated by an AI model? We argue that, in this setting, authorship should not be attributed solely from the textual features of a review, but also from the ideas, judgments, and claims it expresses. To this end, we propose Sem-Detect, an authorship detection method for peer reviews that operationalizes this principle by combining textual features with claim-level semantic analysis. Sem-Detect compares a target review against multiple AI-generated reviews of the same paper, leveraging the observation that different AI models tend to converge on similar points, while human reviewers introduce more unique and diverse ones. As a result, Sem-Detect is able to distinguish fully AI reviews from authentic human-written ones, including those that have been refined using an LLM but still reflect human judgment. Across a dataset of over 20,000 peer reviews from ICLR and NeurIPS conferences, Sem-Detect improves over the strongest baseline by 25.5% in TPR@0.1% FPR in the binary setting. Moreover, in the three-class scenario, we empirically show that LLM refinement preserves the semantic signals of human reviews, which remain distinct from the patterns exhibited by fully AI-generated text; as a result, fewer than 3.5% of LLM-refined human reviews are misclassified as AI-generated. 

---
# Broadening Access to Transportation Safety Data with Generative AI: A Schema-Grounded Framework for Spatial Natural Language Queries 

**Authors**: Mahdi Azhdari, Eric J. Gonzales  

**Link**: [PDF](https://arxiv.org/pdf/2605.21712)  

**Abstract**: Transportation safety analysis requires integrating crash records, roadway attributes, and geospatial data through GIS-based workflows, but access remains uneven across agencies and community stakeholders. Technical prerequisites create a gap between analytical tools central to safety planning and the practitioners able to use them. Local agencies, school committees, and residents may have safety concerns but limited capacity to retrieve, filter, map, and analyze relevant data. Generative AI offers a way to narrow this divide, but its public-sector use raises questions about reliability, reproducibility, and governance. This paper presents a schema-grounded natural language interface for transportation safety analysis, using a large language model (LLM) to interpret user intent while preserving deterministic, reviewable execution against an authoritative database. User queries are translated into structured semantic frames, validated by a rule-based layer, compiled into a typed directed acyclic graph of spatial operations, and executed against a PostGIS database. This bounded design separates language interpretation from deterministic execution, keeping results reproducible and schema-grounded while removing access barriers. The framework is evaluated using a statewide Massachusetts transportation safety database integrating crash records, roadway attributes, and geospatial layers including schools, bus stops, crosswalks, and municipal boundaries. All queries executed successfully; the validation layer corrects errors in 29% of evaluation queries, reflecting the gap between flexible natural language and strict schema-grounded requirements. The results suggest that combining natural language accessibility with deterministic execution is a practical direction for broadening access to transportation safety data, with implications for trustworthy AI in public-sector planning. 

---
# CR4T: Rewrite-Based Guardrails for Adolescent LLM Safety 

**Authors**: Heajun An, Qi Zhang, Vedanth Achanta, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2605.21609)  

**Abstract**: Large language models (LLMs) are increasingly embedded in adolescent digital environments, mediating information seeking, advice, and emotionally sensitive interactions. Yet existing safety mechanisms remain largely grounded in adult-centric norms and operationalize safety through refusal-oriented suppression. While such approaches may reduce immediate policy violations, they can also create conversational dead-ends, limit constructive guidance, and fail to address the developmental vulnerabilities inherent in adolescent-AI interactions. We argue that adolescent LLM safety should be framed not solely as a filtering problem, but as a socio-technical, developmentally aligned transformation problem. To operationalize this perspective, we propose Critique-and-Revise-for-Teenagers (CR4T), a model-agnostic safeguarding framework that selectively reconstructs unsafe or refusal-style outputs into ageappropriate, guidance-oriented responses while preserving benign intent. CR4T combines lightweight risk detection with domain-conditioned rewriting to remove risk-amplifying content, reduce unnecessary conversational shutdown, and introduce developmentally appropriate guidance. Experimental results show that targeted rewriting substantially reduces unsafe and refusal-oriented outcomes while avoiding unnecessary intervention on acceptable interactions. These findings suggest that selective response reconstruction offers a more human-centered alternative to refusal-centric guardrails for adolescent-facing LLM systems. 

---
# Vector Policy Optimization: Training for Diversity Improves Test-Time Search 

**Authors**: Ryan Bahlous-Boldi, Isha Puri, Idan Shenfeld, Akarsh Kumar, Mehul Damani, Sebastian Risi, Omar Khattab, Zhang-Wei Hong, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2605.22817)  

**Abstract**: Language models must now generalize out of the box to novel environments and work inside inference-scaling search procedures, such as AlphaEvolve, that select rollouts with a variety of task-specific reward functions. Unfortunately, the standard paradigm of LLM post-training optimizes a pre-specified scalar reward, often leading current LLMs to produce low-entropy response distributions and thus to struggle at displaying the diversity that inference-time search will require. We propose Vector Policy Optimization (VPO), an RL algorithm that explicitly trains policies to anticipate diverse downstream reward functions and to produce diverse solutions. VPO exploits that rewards are often vector-valued in practice, like per-test-case correctness in code generation or, say, multiple different user personas or reward models. VPO is essentially a drop-in replacement for the GRPO advantage estimator, but it trains the LLM to output a set of solutions where individual solutions specialize to different trade-offs in the vector reward space. Across four tasks, VPO matches or beats the strongest scalar RL baselines on test-time search (e.g. pass@k and best@k), with the gap widening as the search budget grows. For evolutionary search, VPO models unlock problems that GRPO models cannot solve at all. As test-time search becomes more standardized, optimizing for diversity may need to become the default post-training objective. 

---
# Beyond Acoustic Emotion Recognition: Multimodal Pathos Analysis in Political Speech Using LLM-Based and Acoustic Emotion Models 

**Authors**: Juergen Dietrich  

**Link**: [PDF](https://arxiv.org/pdf/2605.22732)  

**Abstract**: We investigate whether acoustic emotion recognition models can serve as proxies for the Pathos dimension in political speech analysis, as operationalised by the TRUST multi-agent large language model (LLM) pipeline. Using a Bundestag plenary speech by Felix Banaszak (51 segments, 245 s) as a case study, we compare three analysis modalities: (1) emotion2vec_plus_large, an acoustic speech emotion recognition (SER) model whose continuous Arousal and Valence values are derived via post-hoc Russell Circumplex projection; (2) Gemini 2.5 Flash, an LLM analysing the full speech audio together with its transcript in an open-ended, context-aware fashion; and (3) TRUST-Pathos scores from a three-advocate LLM supervisor ensemble. Spearman rank correlations reveal that Gemini Valence correlates strongly with TRUST-Pathos (rho = +0.664, p < 0.001), whereas emotion2vec Valence does not (rho = +0.097, p = 0.499). We further demonstrate, via a systematic quality evaluation of the Berlin Database of Emotional Speech (EMO-DB) using Gemini in an open-ended annotation paradigm, that standard SER benchmark corpora suffer from acted speech, cultural bias, and category incompatibility. Our results suggest that LLM-based multimodal analysis captures semantically defined political emotion substantially better than acoustic models alone, while acoustic features remain informative for low-level Arousal estimation. Future work will extend this approach to video-based analysis incorporating facial expression and gaze. 

---
# AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild 

**Authors**: Baiyu Chen, Zechen Li, Wilson Wongso, Lihuan Li, Xiachong Lin, Hao Xue, Benjamin Tag, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2605.22715)  

**Abstract**: As wearable and mobile devices become increasingly embedded in daily life, they offer a practical way to continuously sense human motion in the wild. But inertial signals are highly dependent on the sensing setup, including body location, mounting position, sensor orientation, device hardware, and sampling protocol. This setup dependence makes it difficult to learn motion representations that transfer across devices and datasets, and limits the broader use of wearable IMUs beyond closed-set recognition. We introduce AnyMo, a geometry-aware framework for setup-agnostic human motion modeling. AnyMo uses physics-grounded IMU simulation over dense body-surface placements to generate diverse and plausible synthetic signals, pre-trains a graph encoder from paired synthetic placement views and masked partial observations, tokenizes multi-position IMU into full-body motion tokens, and aligns these tokens with an LLM for motion-language understanding. We evaluate AnyMo on three complementary tasks: zero-shot activity recognition across 14 unseen downstream datasets, cross-modal retrieval, and wearable IMU motion captioning, where it improves average Accuracy/F1/R@2 by 11.7\%/11.6\%/22.6\% on HAR, increases zero-shot IMU-to-text and text-to-IMU retrieval MRR by 15.9\% and 28.6\%, respectively, and improves zero-shot captioning BERT-F1 by 18.8\%. These results support AnyMo as a generalist model for wearable motion understanding in the wild. Project page: this https URL. 

---
# AMEL: Accumulated Message Effects on LLM Judgments 

**Authors**: Sid-ali Temkit  

**Link**: [PDF](https://arxiv.org/pdf/2605.22714)  

**Abstract**: Large language models are routinely used as automated evaluators: to review code, moderate content, or score outputs, often with many items passing through one conversation. We ask whether the polarity of prior conversation history biases subsequent judgments, an effect we call the accumulated message effect on LLM judgments (AMEL). Across 75,898 API calls to 11 models from 4 providers (OpenAI, Anthropic, Google, and four open-source models), we present identical test items in isolation or following histories saturated with predominantly positive or negative evaluations.
Models shift toward the conversation's prevailing polarity (d = -0.17, p < 10^-46). The effect concentrates on items where the model is genuinely uncertain at baseline (d = -0.34 for high-entropy items, vs d = -0.15 when the baseline is deterministic). Bias does not grow with context length: 5 prior turns and 50 produce the same shift (Spearman |r| < 0.01; OLS slope p = 0.80). And there is a negativity asymmetry: paired per item, negative histories induce 1.62x more bias than positive (t = 13.46, p < 10^-39, n = 2,481). Scaling helps but does not solve it (Anthropic: Haiku -0.22 to Opus -0.17; OpenAI: Nano -0.34 to GPT-5.2 -0.17).
Three follow-ups narrow the mechanism. The token probability distribution shifts continuously, not at a threshold. The negativity asymmetry has both token-level and semantic components, though attributing the balance is exploratory at our sample sizes. Position does not matter: five biased turns anywhere in a 50-turn history produce the same shift. The simplest fix for evaluation pipelines is a fresh context per item; when batching is unavoidable, balancing the history helps. 

---
# The Double Dilemma in Multi-Task Radiology Report Generation: A Gradient Dynamics Analysis and Solution 

**Authors**: Erjian Zhang, Yatong Hao, Liejun Wang, Zhiqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.22635)  

**Abstract**: While multi-task learning based automatic radiology report generation (RRG) is widely adopted to ensure clinical consistency, most focus on architectural designs yet remain limited to coarse linear scalarization strategies. These strategies cannot effectively balance the hard constraints of discriminative clinical supervision with the smoothness requirements of report generation. To address these problems, we analyze the failure mechanism of linear scalarization from the perspective of gradient dynamics, utilizing the stochastic differential equation (SDE) framework to characterize it as a "Double Dilemma" of drift term deviation and diffusion term decay. Based on this, we propose a backbone-agnostic optimizer named Conflict-Averse Magnitude-Enhanced Gradient Descent (CAME-Grad). Through conflict-averse direction rectification and magnitude-enhanced energy injection, the algorithm not only ensures geometric validity, but also avoids local optimal solutions. Then, the adaptive gradient fusion mechanism is used to establish a dynamic balance between the theoretical optimal direction and the task-specific inductive bias. Experiments show that as a universal plug-and-play optimizer, CAME-Grad brings substantial and consistent improvements across eight diverse RRG methods, elevating overall clinical efficacy performance by an average of 2.3\% on MIMIC-CXR and 1.9\% on IU X-Ray. Our code is available at this https URL. 

---
# Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework 

**Authors**: Shourov Joarder, Diganta Sikdar, Ahsan Habib Akash, Binod Bhattarai, Prashnna Gyawali  

**Link**: [PDF](https://arxiv.org/pdf/2605.22620)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has substantially improved the reasoning ability of LLMs, but often depends on external supervision from human annotations or gold-standard solutions. Reinforcement learning from internal feedback (RLIF) has recently emerged as a scalable unsupervised alternative, using signals extracted from the model itself. However, existing RLIF methods typically rely on a single internal reward, which can lead to reward hacking, entropy collapse, and degraded reasoning structure. We propose a multi-reward RLIF framework that decomposes the training signal into two complementary components: an answer-level reward based on cluster voting and a completion-level reward based on token-wise self-certainty. To combine these signals robustly, we apply GDPO-based normalization to reduce reward-scale imbalance. We further introduce KL-Cov regularization, which targets low-entropy token distributions responsible for disproportionate entropy reduction, preserving exploration and preventing late-stage collapse. Across mathematical reasoning and code-generation benchmarks, our method improves stability and robustness over prior unsupervised RL approaches, while achieving performance close to supervised RLVR methods. These results show that complementary internal rewards, combined with targeted regularization, can support stable long-horizon reasoning without relying on external ground-truth supervision. Code will be released soon. 

---
# A Tutorial on Diffusion Theory: From Differential Equations to Diffusion Models 

**Authors**: Jiayi Fu, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22586)  

**Abstract**: This tutorial develops diffusion models from the viewpoint of differential equations. We begin with the conditional Gaussian forward process and show that this path admits both an ordinary differential equation (ODE) representation and a stochastic differential equation (SDE) representation. Averaging the conditional process over the data distribution then yields marginalized forward ODE and SDE formulations that transport the data distribution $p_0=p_{\mathrm{data}}$ to a Gaussian prior $p_1=\mathcal{N}(0,I)$. We next derive the corresponding reverse-time dynamics, namely the reverse SDE and the reverse probability-flow ODE, both of which are governed by the marginal score $\grad\log p_t(x)$. This leads to a training objective for score estimation and shows that the standard noise-prediction objective is equivalent to score matching up to an additive constant independent of the model parameters. We then discuss sampling methods for the learned reverse dynamics, including DPM-Solver, as well as guided sampling through classifier guidance and classifier-free guidance. Finally, we compare DDPM and DDIM with the reverse SDE/ODE framework and show that they share the same training objective, while DDPM sampling corresponds to discrete reverse-SDE sampling and DDIM sampling corresponds to reverse-ODE sampling. 

---
# SpaceDG: Benchmarking Spatial Intelligence under Visual Degradation 

**Authors**: Xiaolong Zhou, Yifei Liu, Ziyang Gong, Jiarui Li, Qiyue Zhao, Muyao Niu, Yuanyuan Gao, Le Ma, Xue Yang, Hongjie Zhang, Zhihang Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2605.22536)  

**Abstract**: Multimodal Large Language Models (MLLMs) have made rapid progress in spatial intelligence, yet existing spatial reasoning benchmarks largely assume pristine visual inputs and overlook the degradations that commonly occur in real-world deployment, such as motion blur, low light, adverse weather, lens distortion, and compression artifacts. This raises a fundamental question: how robust is the spatial intelligence of current MLLMs when visual observations are imperfect? To answer this question, we introduce SpaceDG, the first large-scale dataset for degradation-aware spatial understanding. It is constructed with a physically grounded degradation synthesis engine that embeds degradation formation process into 3D Gaussian Splatting (3DGS) rendering, enabling realistic simulation of nine degradation types. The resulting dataset contains approximately 1M QA pairs from nearly 1,000 indoor scenes. We further introduce SpaceDG-Bench, an human-verified benchmark with 1,102 questions spanning 11 reasoning categories and 9 visual degradation types, yielding over 10K VQA instances. Evaluating 25 open- and closed-source MLLMs reveals that visual degradations consistently and substantially impair spatial reasoning, exposing a critical robustness gap. Finally, we show that finetuning on SpaceDG markedly improves degradation robustness and can even surpass human performance under degraded conditions without any performance drop on clean images, highlighting the promise of degradation-aware training for robust spatial intelligence. 

---
# Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning 

**Authors**: Zihan Liang, Yufei Ma, Ben Chen, Zhipeng Qian, Xuxin Zhang, Huangyu Dai, Lingtao Mao  

**Link**: [PDF](https://arxiv.org/pdf/2605.22511)  

**Abstract**: Post-training has become the dominant recipe for turning a language model into a competent search-augmented reasoning agent. A line of recent work pushes its performance further by adding elaborate machinery on top of this standard pipeline. These augmentations import external supervision from stronger external systems, attach auxiliary modules such as process reward models or retrospective critics, restructure the rollout itself with tree search or multi-stage curricula, or shape the reward with hand-crafted bonuses and penalties. Each addition delivers a measurable gain, but each also inflates the training pipeline and ties the recipe to resources or designs that may not always be available. We take a step back and ask whether any of this machinery is actually necessary, and propose Search-E1, a self-evolution method that lets a search-augmented agent improve through only vanilla GRPO interleaved with offline self-distillation (OFSD). After each GRPO round, the policy rolls out on its own training questions. A token-level forward KL objective then aligns the policy's inference-time distribution to its own distribution under a privileged context that exposes a more efficient sibling trajectory. Despite this simplicity, the procedure naturally provides dense per-step supervision. On seven QA benchmarks, Search-E1 reaches $0.440$ average EM with Qwen2.5-3B, surpassing all open-source baselines at both scales. Code and complete version will be made public soon. 

---
# Reflecti-Mate: A Conversational Agent for Adaptive Decision-Making Support Through System 1 and System 2 Thinking 

**Authors**: Morita Tarvirdians, Senthil Chandrasegaran, Hayley Hung, Catholijn M. Jonker, Catharine Oertel  

**Link**: [PDF](https://arxiv.org/pdf/2605.22509)  

**Abstract**: Making high-stakes personal decisions involves cognitive, emotional, and intuitive processes, and individuals differ in how they allocate attention across these modes. Integration of these processes has shown to benefit decision making. Yet, most current decision-support systems focus primarily on supporting cognitive aspects, rather than adapting to the individual's thinking profile to support integration of different types of thoughts. In this study, we investigate an agent designed to encourage integration by adapting to the individual user's thought patterns. We explore its effects on participants' perceptions of the agent and their reflective behavior, in comparison with unaided pre-reflection and a baseline agent. In a between-subjects study (N = 128), our agent, which fostered broad and elaborated thinking, enabled more personalized reflective trajectories, elicited more integrative reflective language, and was perceived as providing stronger support for holistic reflection. In contrast, the baseline agent produced homogenized profiles dominated by cognitive language across participants. 

---
# Structured-Sparse Attention for Entity Tracking with Subquadratic Sequence Complexity 

**Authors**: Hangyue Zhao, Paul Caillon, Erwan Fagnou, Alexandre Allauzen  

**Link**: [PDF](https://arxiv.org/pdf/2605.22476)  

**Abstract**: Entity tracking requires maintaining and updating latent states for entities and attributes over long sequences. Recent task-specific attention operators can compress deep Transformer stacks into a few layers by performing multi-hop state propagation within a single layer, but their dense evaluation remains expensive. We show that in this setting, learned attention is strongly structured: most mass concentrates in local block-diagonal neighborhoods with a light cross-block residue. Exploiting this, we derive a blockwise evaluation of a resolvent-style operator that keeps within-block interactions exact and routes cross-block interactions through a reduced system. The resulting evaluation is subquadratic in sequence length $O(n^{4/3}d)$ (and $O(n^{7/3})$ when $d\approx n$). On controlled tracking benchmarks, our method matches the dense operator's accuracy while reducing wall-clock time by $12-29\%$ under a standardized measurement protocol, and is up to $2.4 \times$ faster than a compact dense Transformer at comparable exact-match accuracy. We further provide ablations over block size and model capacity, and identify a limitation: performance collapses when the number of simultaneously evolving properties exceeds the number of attention heads. 

---
# Epicure: Navigating the Emergent Geometry of Food Ingredient Embeddings 

**Authors**: Jakub Radzikowski, Josef Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.22391)  

**Abstract**: We present Epicure, a family of three sibling skip-gram ingredient embeddings retrained from scratch on a multilingual recipe corpus. We aggregate 4.14M recipes from 11 sources spanning seven languages, English, Chinese, Russian, Vietnamese, Spanish, Turkish, Indonesian, German, and Indian-English, and normalise the raw ingredient strings to 1,790 canonical entries via an LLM-augmented pipeline. A 203,508-edge ingredient-ingredient NPMI graph and an 80,019-edge typed FlavorDB ingredient-compound graph, 2,247 typed compound nodes across 15 categories, seed three Metapath2Vec variants that share architecture and hyperparameters and differ only in the random-walk schema: Cooc walks the co-occurrence graph only, Chem walks the typed compound metapaths only, and Core blends both via injected ingredient-ingredient walks at controlled mixing, placing each model at a distinct point on the chemistry-vs-recipe-context spectrum. 

---
# Boundary-targeted Membership Inference Attacks on Safety Classifiers 

**Authors**: Anthony Hughes, Alexander Goldberg, Prince Jha, Adam Perer, Nikolaos Aletras, Niloofar Mireshghallah  

**Link**: [PDF](https://arxiv.org/pdf/2605.22373)  

**Abstract**: Safety classifiers are essential safeguards within generative AI systems, filtering harmful content or identifying at-risk users when interacting with large language models. Despite their necessity, these models are trained on sensitive datasets including discussions of self-harm and mental health, raising important, yet poorly understood, privacy concerns. Membership inference attacks (MIAs) allow adversaries to infer membership of examples used to train models. In this work, we hypothesize that identifying the examples on which the classifier is least confident are informative for an adversary to infer membership. This reflects a localized failure of generalization, where the model relies on memorization to resolve ambiguity in the training set. To investigate this, we introduce a new boundary-targeted selection strategy that identifies low confidence examples that amplify the signal of an examples membership within a training set. Our experimental results show that an adversary can recover 19\% of the conversations a safety classifier flagged as indicating user distress, at a 5\% false-positive rate, on a classifier fine-tuned for detecting a user who may require emotional support. This is $3.5$ times more than attacking using state-of-the-art MIA methods alone. Finally, we characterize the boundary laying examples and show that content-based filtering is ineffective for protection, and existing noise strategies can effectively mitigate susceptibility of these examples. 

---
# Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL 

**Authors**: Sophia Xiao Pu, Zhaotian Weng, Chengzhi Liu, Jayanth Srinivasa, Gaowen Liu, William Yang Wang, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22217)  

**Abstract**: Self-play reinforcement learning trains language models on their own generated tasks, co-evolving a proposer and solver without human labels. Recent systems report strong reasoning gains, but collapse and instability are widely observed and poorly understood. The dominant response treats this as a reward-design problem. We argue instead that self-play stability is governed by two distinct levers: a data-level gate that decides which proposer-generated tasks enter the training pool, and the reward signal that updates the policy on tasks already admitted. Through controlled experiments on a Python output-prediction task and a deterministic-DSL twin task that strips pretraining priors, output ambiguity, and executor noise, we find the two levers are asymmetric. A strict gate is sufficient for stability under every reward variant we test, including a self-consistency reward with no access to ground truth; while no reward variant is sufficient once the gate is removed. This asymmetry exposes a counter-intuitive coupling we call the Grounded Proposer Paradox: a proposer with ground-truth access accelerates collapse faster than an ungrounded one when paired with a self-consistency solver, by concentrating training on clean tasks that form the fastest path to a spurious self-consistent attractor. Replacing the binary gate with a continuous strictness parameter $\varepsilon$ further reveals a two-stage phase transition: training-side metrics decouple at low $\varepsilon$, while validation accuracy holds until $\varepsilon$ is much higher. Data-level gating, not reward calibration, is the binding constraint on self-play stability. 

---
# Maestro: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles 

**Authors**: Jinyang Wu, Guocheng Zhai, Ruihan Jin, Yuhao Shen, Zhengxi Lu, Fan Zhang, Haoran Luo, Zheng Lian, Zhengqi Wen, Jianhua Tao  

**Link**: [PDF](https://arxiv.org/pdf/2605.22177)  

**Abstract**: The proliferation of large language models (LLMs) and modular skills has endowed autonomous agents with increasingly powerful capabilities. Existing frameworks typically rely on monolithic LLMs and fixed logic to interface with these skills. This gives rise to a critical bottleneck: different LLMs offer distinct advantages across diverse domains, yet current frameworks fail to exploit the complementary strengths of models and skills, thereby limiting their performance on downstream tasks. In this paper, we present Maestro (Multimodal Agent for Expert-Skill Targeted Reinforced Orchestration), a Reinforcement Learning (RL)-driven orchestration framework that reframes heterogeneous multimodal tasks as a sequential decision-making process over a hierarchical model-skill registry. Rather than consolidating all knowledge into a single model, Maestro trains a lightweight policy to dynamically compose ensembles of frozen expert models and a two-tier skill library, deciding at each step whether to invoke an external expert, which model-skill pair to select, and when to terminate. The policy is optimized via outcome-based RL, requiring no step-level supervision. We evaluate Maestro across ten representative multimodal benchmarks spanning mathematical reasoning, chart understanding, high-resolution perception, and domain-specific analysis. With only a 4B orchestrator, Maestro achieves an average accuracy of 70.1%, surpassing both GPT-5 (69.3%) and Gemini-2.5-Pro (68.7%). Crucially, the learned coordination policy generalizes to unseen models and skills without retraining: augmenting the registry with out-of-domain experts yields a 59.5% average on four challenging benchmarks, outperforming all closed-source baselines. Maestro further maintains high computational efficiency with low latency. The source code is available at this https URL. 

---
# Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents 

**Authors**: Xing Zhang, Yanwei Cui, Guanghui Wang, Ziyuan Li, Wei Qiu, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2605.22148)  

**Abstract**: Self-evolving skill libraries, pioneered by Voyager, let frozen LLM agents accumulate reusable knowledge without weight updates, yet recent evaluation shows that LLM-authored skills deliver $+0.0$pp over no-skill baselines while human-curated ones deliver $+16.2$pp: the bottleneck is not skill authoring but lifecycle management. We introduce \textbf{Ratchet}, a single-agent loop in which a frozen LLM writes, retrieves, curates, and retires its own natural-language skills. Ratchet integrates four candidate hygiene mechanisms: outcome-driven retirement, a bounded active-cap, meta-skill authoring guidance, and pattern canonicalisation. On MBPP+ hard-100 with Claude Opus 4.7, Ratchet lifts held-out pass@1 from a $0.258 \pm 0.047$ baseline to a late-window rolling mean of $0.584$ (peak $0.658 \pm 0.042$) across 100 rounds and 3 seeds, a $+0.328 \pm 0.018$ rolling-mean gain where the no-skill control drifts at $+0.002 \pm 0.005$; the same recipe transfers to an agentic solver on SWE-bench Verified ($+0.22$ peak lift over 20 rounds). Eight ablations (A1--A8) reveal that the minimal working recipe is smaller than our design suggests: retirement and the meta-skill authoring prior are load-bearing, while explicit deduplication (canonicalisation, cover-guard) is subsumed by the meta-skill itself. A non-divergence proposition shows that bounded cap and retirement threshold together prevent expected performance from drifting below the no-skills floor. 

---
# Efficient Agentic Reasoning Through Self-Regulated Simulative Planning 

**Authors**: Mingkai Deng, Jinyu Hou, Lara Sá Neves, Varad Pimpalkhute, Taylor W. Killian, Zhengzhong Liu, Eric P. Xing  

**Link**: [PDF](https://arxiv.org/pdf/2605.22138)  

**Abstract**: How should an agent decide when and how to plan? A dominant approach builds agents as reactive policies with adaptive computation (e.g., chain-of-thought), trained end-to-end expecting planning to emerge implicitly. Without control over the presence, structure, or horizon of planning, these systems dramatically increase reasoning length, yielding inefficient token use without reliable accuracy gains. We argue efficient agentic reasoning benefits from decomposing decision-making into three systems: simulative reasoning (System II) grounding deliberation in future-state prediction via a world model; self-regulation (System III) deciding when and how deeply to plan via a learned configurator; and reactive execution (System I) handling fine-grained action. Simulative reasoning provides unified planning across diverse tasks without per-domain engineering, while self-regulation ensures the planner is invoked only when needed. To test this, we develop SR$^2$AM (Self-Regulated Simulative Reasoning Agentic LLM), realizing both as distinct stages within an LLM's chain-of-thought, with the LLM as world model. We explore two instantiations: recording decisions from a prompted multi-module system (v0.1) and reconstructing structured plans from traces of pretrained reasoning LLMs (v1.0), trained via supervised then reinforcement learning (RL). Across math, science, tabular analysis, and web information seeking, v0.1-8B and v1.0-30B achieve Pass@1 competitive with 120-355B and 685B-1T parameter systems respectively, while v1.0-30B uses 25.8-95.3% fewer reasoning tokens than comparable agentic LLMs. RL increases average planning horizon by 22.8% while planning frequency grows only 2.0%, showing it learns to plan further ahead rather than more often. More broadly, learned self-regulation instantiates a principle we expect to extend beyond planning to how agents govern their own learning and adaptation. 

---
# From Reasoning Chains to Verifiable Subproblems: Curriculum Reinforcement Learning Enables Credit Assignment for LLM Reasoning 

**Authors**: Xitai Jiang, Zihan Tang, Wenze Lin, Yang Yue, Shenzhi Wang, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22074)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) has shown strong promise for LLM reasoning, but outcome-based RLVR remains inefficient on hard problems because correct final-answer rollouts are rare and sample-level credit assignment cannot use partial progress in failed attempts. We introduce SCRL (Subproblem Curriculum Reinforcement Learning), a curriculum RL framework that derives verifiable subproblems from reference reasoning chains and fixes the final subproblem as the original problem. This turns partial progress on hard problems into verifiable learning signals. Algorithmically, SCRL uses subproblem-level normalization, which normalizes rewards independently at each subproblem position and assigns the resulting advantages to the corresponding answer spans, enabling finer-grained credit assignment without external rubrics or reward models. Our analysis shows that subproblem curricula lift hard problems out of gradient dead zones, with larger relative gains as the original problem becomes harder. Across seven mathematical reasoning benchmarks, SCRL outperforms strong curriculum-learning baselines, improving average accuracy over GRPO by +4.1 points on Qwen3-4B-Base and +1.9 points on Qwen3-14B-Base. On AIME24, AIME25, and IMO-Bench, SCRL further improves pass@1 by +3.7 points and pass@64 by +4.6 points on Qwen3-4B-Base, indicating better exploration on hard reasoning problems. 

---
# HyLoVQA: Dynamic Hypernetwork-Generated Low-Rank Adaptation for Continual Visual Question Answering 

**Authors**: Yiran Wang, Chenyi Xiong, Ziyue Qin, Miao Zhang, Kui Xiao, Zhifei Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.22035)  

**Abstract**: Continual Visual Question Answering (VQA) requires learning from non-stationary streams of visual inputs and questions while preserving past knowledge. Most prior methods adapt by updating a largely shared parameter set. This often leads to cross-level task interference, hindering accurate adaptation to the current task and object. To address this limitation, we propose HyLoVQA. It maintains a drift-resilient memory bank of anchors. The bank stores the content of visual objects and textual tasks, and they are updated using current input features. Conditioned on retrieved anchors, a hypernetwork generates lightweight Low-Rank Adaptation (LoRA) adapters. This ensures parameter efficiency, allowing the model to adapt to each task and object dynamically. Additionally, we formulate an alignment loss that aligns semantic discrepancies in the feature space with functional changes in the parameter space, thereby constraining LoRA adapters to remain focused on the current task and object. Extensive experiments on VQA v2 and NExT-QA under both standard and compositional settings demonstrate the superiority of HyLoVQA over prior state-of-the-art methods. 

---
# Check Your LLM's Secret Dictionary! Five Lines of Code Reveal What Your LLM Learned (Including What It Shouldn't Have) 

**Authors**: Hisashi Miyashita  

**Link**: [PDF](https://arxiv.org/pdf/2605.22005)  

**Abstract**: We show that singular value decomposition of the lm_head} weight matrix of a transformer-based large language model -- requiring only five lines of PyTorch and no model inference -- reveals interpretable semantic subspaces directly from the model weights. Each left singular vector identifies the vocabulary tokens most readily selected when the hidden state aligns with the corresponding singular direction; inspecting these clusters exposes the model's training data composition and curation philosophy.
Analysing GPT-OSS-120B, Gemma-2-2B, and Qwen2.5-1.5B, we find that singular value spectra and vocabulary cluster structures differ systematically across models: GPT exhibits a graduated hierarchy of functionally differentiated subspaces; Gemma is dominated by pre-nineteenth-century English orthography, forming a stepwise clustering structure that may contribute to high output controllability; and Qwen exhibits broad multilingual coverage alongside subspaces whose vocabulary the authors have determined to be ethically inappropriate for direct publication.
Base-instruct comparison reveals that ethically concerning subspaces originate in pretraining and are not removed by post-training alignment. We introduce the Vocabulary Cluster Score (VCS) to quantify subspace coherence, and the Weighted Projection Score (WPS) as a static glitch token detector; applying WPS to GPT-OSS-120B recovers shokubutsu-hyakka-tsu (ID 137606), a well-known glitch token widely reported in the CJK language community, without any model inference. We propose a taxonomy of root causes for problematic vocabulary content and call for lm_head} SVD analysis to be adopted as a standard pre-release safety auditing step. Our findings further suggest directions toward SVD-guided tokenizer optimisation and more controllable LLM design. 

---
# Blind Spots in the Guard: How Domain-Camouflaged Injection Attacks Evade Detection in Multi-Agent LLM Systems 

**Authors**: Aaditya Pai  

**Link**: [PDF](https://arxiv.org/pdf/2605.22001)  

**Abstract**: Injection detectors deployed to protect LLM agents are calibrated on static, template-based payloads that announce themselves as override directives. We identify a systematic blind spot: when payloads are generated to mimic the domain vocabulary and authority structures of the target document, what we call domain camouflaged injection, standard detectors fail to flag them, with detection rates dropping from 93.8% to 9.7% on Llama 3.1 8B and from 100% to 55.6% on Gemini 2.0 Flash. We formalize this as the Camouflage Detection Gap (CDG), the difference in injection detection rate between static and camouflaged payloads. Across 45 tasks spanning three domains and two model families, CDG is large and statistically significant (chi^2 = 38.03, p < 0.001 for Llama; chi^2 = 17.05, p < 0.001 for Gemini), with zero reverse discordant pairs in either case. We additionally evaluate Llama Guard 3, a production safety classifier, which detects zero camouflage payloads (IDRcamouflage = 0.000), confirming that the blind spot extends beyond few-shot detectors to dedicated safety classifiers. We further show that multi-agent debate architectures amplify static injection attacks by up to 9.9x on smaller models, while stronger models show collective resistance. Targeted detector augmentation provides only partial remediation (10.2% improvement on Llama, 78.7% on Gemini), suggesting the vulnerability is architectural rather than incidental for weaker models. Our framework, task bank, and payload generator are released publicly. 

---
# Echo: Learning from Experience Data via User-Driven Refinement 

**Authors**: Hande Dong, Xiaoyun Liang, Jiarui Yu, Jiayi Lin, Changqing Ai, Feng Liu, Wenjun Zhang, Rongbi Wei, Chaofan Zhu, Linjie Che, Feng Wu, Xin Shen, Dexu Kong, Xiaotian Wang, Qiuyuan Chen, Bingxu An, Yueting Lei, Qiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.21984)  

**Abstract**: Static "human data" faces inherent limitations: it is expensive to scale and bounded by the knowledge of its creators. Continuous learning from "experience data" - interactions between agents and their environments - promises to transcend these barriers. Today, the widespread deployment of AI agents grants us low-cost access to massive streams of such real-world experience. However, raw interaction logs are inherently noisy, filled with trial-and-error and low information density, rendering them inefficient for direct model training.
We introduce Echo, a generalized framework designed to operationalize the transition from raw experience to learnable knowledge, effectively "echoing" environmental feedback back into the training loop for model optimization. In today's agent ecosystem, user refinement serves as a primary source of such feedback: driven by responsibility for the outcome, users rigorously transform flawed agent proposals into verified solutions. These user-driven refinement sequences inherently distill agents' crude attempts into high-quality training signals. Echo systematically harvests these signals to continuously align the agent with real-world needs. Large-scale validation in a production code completion environment confirms that Echo effectively harnesses this pipeline, breaking the static performance ceiling by increasing the acceptance rate from 25.7% to 35.7%. 

---
# Planning in the LLM Era: Building for Reliability and Efficiency 

**Authors**: Michael Katz, Harsha Kokel, Kavitha Srinivas, Shirin Sohrabi  

**Link**: [PDF](https://arxiv.org/pdf/2605.21902)  

**Abstract**: Growing attention to intelligent agents has put a spotlight on one of their central capabilities: planning. Early attempts to leverage large language models (LLMs) for planning relied on single-shot plan generation, followed by hybrid approaches that coupled LLMs with limited external search. These methods, unsound and incomplete by their very nature, often require substantial resources without yielding better solutions on unseen problems. As the limitations of LLMs become clearer, recent work has shifted toward using them at solution construction time -- generating symbolic solvers for a family of problems that can be verified and then used efficiently at inference time. This trend reflects the growing need for agents that are both reliable and resource-efficient. It also offers a path towards generating maintainable planners with minimal dependence on language models at inference time. In this paper, we argue that this shift reflects a broader realignment of the planning field in the LLM era. We examine three major categories of planner-generation methods, discuss their current limitations, and outline research steps towards a more reliable and efficient LLM-based generation of planners. 

---
# Geometry-Adaptive Explainer for Faithful Dictionary-Based Interpretability under Distribution Shift 

**Authors**: Sungjun Lim, Heedong Kim, Andrew Lee, Kyungwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.21849)  

**Abstract**: Mechanistic interpretability aims to explain a model's behavior by identifying causally responsible internal structures. Dictionary-based explainers such as sparse autoencoders and transcoders are a primary tool, but their faithfulness under out-of-distribution (OOD) shift has received little systematic attention. We show that distribution shift rotates the subspace that the model actively uses, misaligning the explainer's dictionary trained on in-distribution (ID) activations. We formalize this misalignment as the faithfulness gap, a geometric distance between the ID dictionary and the OOD-active subspace, and show that it controls OOD faithfulness degradation. To reduce this gap, we propose the Geometry-Adaptive Explainer (GAE), which realigns the explainer's dictionary with the OOD-active subspace while preserving the original feature structure. This requires only unlabeled OOD activations and no gradient updates. We prove that GAE improves over the unadapted ID explainer, with excess loss bounded quadratically by the second-moment shift. Empirically, GAE even matches or surpasses all training-based baselines in causal faithfulness across multiple models and OOD settings. 

---
# Energy-Gated Attention: Spectral Salience as an Inductive Bias for Transformer Attention 

**Authors**: Athanasios Zeris  

**Link**: [PDF](https://arxiv.org/pdf/2605.21842)  

**Abstract**: Standard transformer attention computes pairwise similarity between queries and keys, treating all tokens as equally salient regardless of their intrinsic informational content. In turbulent fluid dynamics, coherent structures -- the energetically dominant, spatially organized patterns that persist amid background chaos -- carry a disproportionate fraction of total energy and govern all transport. We propose that tokens play an analogous role in transformer attention: informationally dense positions (morphological boundaries, syntactic heads, discourse markers) concentrate spectral energy and should attract proportionally more attention than background tokens (function words, repeated patterns, low-information filler). We propose Energy-Gated Attention (EGA): a simple modification that gates value aggregation by the spectral energy of key token embeddings, computed by a single learned linear projection that discovers the dominant spectral mode of the embedding field. On TinyShakespeare, EGA achieves +0.103 validation loss improvement with only 12,480 additional parameters (<0.26% overhead) and no measurable computational cost. The result is consistent on Penn Treebank (+0.101), demonstrating dataset independence. A systematic ablation across three wavelet families (fixed Morlet, Daubechies db2/db4, and a parametric Morlet) establishes that fixed structured bases are suboptimal -- the optimal energy direction is data-adaptive and non-sinusoidal -- while identifying learned wavelet packets as a promising open direction. The learned energy threshold converges to tau ~= 0.35 independently of initialization, corresponding to the fraction (~36%) of tokens carrying above-average spectral energy in English text, a stable linguistic property consistent with the fraction of content words in running English text. 

---
# Why Semantic Entropy Fails: Geometry-Aware and Calibrated Uncertainty for Policy Optimization 

**Authors**: Zheyuan Zhang, Kaiwen Shi, Han Bao, Zehong Wang, Tianyi Ma, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.21801)  

**Abstract**: Post-training has become central to improving reasoning and alignment in large language models, where critic-free models enable scalable learning from model-generated outputs but lack principled mechanisms to distinguish informative from noisy signals. Recent approaches leverage response-level measures as uncertainty signals to regulate group-based optimization methods such as GRPO. Yet their empirical success remains unstable and unclear in how they influence optimization dynamics. In this paper, we provide, to our knowledge, the first principled formulation that interprets uncertainty signals as mechanisms for characterizing and regulating gradient variance and learning signal quality. Based on both empirical and theoretical analysis, we identify two critical gaps of current entropy-based estimators: The anisotropic gap and The calibration gap. Motivated by this analysis, we propose Geometric-aware Calibrated Policy Optimization (GCPO), a novel framework integrating geometry-aware measures to capture semantic disagreement with reward-based calibration to align uncertainty with learning signal strength. Experiments on multiple benchmarks show that our approach more faithfully tracks gradient variability and consistently improves post-training performance. Our results highlight the importance of designing uncertainty signals that are aligned with optimization dynamics, offering a principled perspective for robust post-training. 

---
# MM-Conv: A Multimodal Dataset and Benchmark for Context-Aware Grounding in 3D Dialogue 

**Authors**: Anna Deichler, Jim O'Regan, Fethiye Irmak Dogan, Lubos Marcinek, Anna Klezovich, Iolanda Leite, Jonas Beskow  

**Link**: [PDF](https://arxiv.org/pdf/2605.21796)  

**Abstract**: Grounding language in the physical world requires AI systems to interpret references that emerge dynamically during conversation. While current vision-language models (VLMs) excel at static image tasks, they struggle to resolve ambiguous expressions in spontaneous, multi-turn dialogue. We address this gap by introducing (1) a benchmark for referential communication in dynamic 3D environments, built from 6.7 hours of egocentric VR interaction with synchronized speech, motion, gaze, and 3D scene geometry, and (2) a two-stage grounding pipeline that explicitly resolves conversational ambiguity before visual localization. The benchmark includes over 4,200 manually verified referring expressions spanning full, partitive, and pronominal types. Our contextual rewriting approach improves grounding performance by 11-22 percentage points on average, with a pure detector (GroundingDINO) reaching 56.7% on pronominals after rewriting, nearly double the best end-to-end baseline. Results demonstrate that decoupling linguistic reasoning from visual perception is more effective than end-to-end approaches for conversational grounding. 

---
# BEiTScore: Reference-free Image Captioning Evaluation with an Efficient Cross-Encoder Model 

**Authors**: Gonçalo Gomes, Bruno Martins, Chrysoula Zerva  

**Link**: [PDF](https://arxiv.org/pdf/2605.21728)  

**Abstract**: Image captioning evaluation remains a significant challenge, as vision-language models evolve toward more challenging capabilities such as generating long-form and context-rich descriptions. State-of-the-art evaluation metrics involve extensive computational costs associated with the use of Large Language Models (LLMs) as judges, or instead suffer from the limitations of standard CLIP-based encoders, such as strict token limits, lack of fine-grained sensitivity, or lack of compositional generalization by treating captions as ``bags-of-words.'' We propose a new learned metric that tackles the aforementioned challenges, based on a lightweight cross-encoder that is initialized from a visual question-answering model checkpoint, balancing a strong weight initialization with computational efficiency. Our training scheme uses a carefully assembled data mixture for supervised learning, featuring adversarial LLM-based data augmentations to enhance model sensitivity to fine-grained visual-linguistic errors. We also introduce a new benchmark designed to assess detailed captioning evaluation across diverse scenarios. Experimental results demonstrate that the proposed metric achieves state-of-the-art performance while maintaining the efficiency required for large-scale benchmarking, quality-aware decoding, or reward guidance. 

---
# X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation 

**Authors**: Sharath Turuvekere Sreenivas, Adithyakrishna Venkatesh Hanasoge, Mingyu Yang, Ali Taghibakhshi, Saurav Muralidharan, Ashwath Aithal, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2605.21699)  

**Abstract**: Cross-tokenizer knowledge distillation allows a student model to learn from teachers with incompatible vocabularies. Prior work operates on hidden states or logits; the latter is preferred as a drop-in replacement requiring no auxiliary components. Logit-based methods either use only the correct-token probability, missing the full 'dark knowledge' in the teacher's distribution, or operate on the full output distribution, relying on strict token partitioning and/or unprincipled heuristic ranking. We identify two key shortcomings of full-distribution, logit-based methods: (i) an uncommon-token failure, where critical tokens fall into the unmatched subset (e.g., Llama's 1100 multi-digit numerals under digit-splitting Qwen supervision) and are suppressed during training, reducing GSM8k from 12.89 to 2.56 compared to same-tokenizer KD from a weaker teacher; and (ii) over-conservative matching, where strict 1-to-1 matching excludes near-equivalent tokens across surface forms. These failures require distinct remedies: eliminating the partition when critical tokens are misaligned, and refining it when alignment is reliable. We propose X-Token, an approach with two complementary loss formulations targeting these issues. P-KL removes partitioning and aligns the student's distribution with the teacher's via a sparse projection matrix W (initialized from tokenizer-level string rules) to address the uncommon-token failure. H-KL retains the hybrid form while relaxing matching to align each student token with its top-ranked teacher mapping under W. Both objectives share W and extend naturally to multiple teachers. Empirically, on Llama-3.2-1B, X-Token outperforms the current state of the art GOLD by +3.82 average points with a Qwen3-4B teacher and by +0.5 with a Phi-4-Mini teacher. Further, a two-teacher setup (Phi-4-mini + Llama-3B) improves over single-teacher distillation by +1.3 points. 

---
# Value-Gradient Hypothesis of RL for LLMs 

**Authors**: Arip Asadulaev, Daniil Ognev, Karim Salta, Martin Takac  

**Link**: [PDF](https://arxiv.org/pdf/2605.21654)  

**Abstract**: Reinforcement learning substantially improves pretrained language models, but it remains understudied why critic-free methods such as PPO and GRPO work as well as they do, and when they should provide the largest gains. We develop a value-gradient perspective of critic-free RL for LLM post-training. First, under a differentiable rollout and additive-noise parameterization, we show that the actor update is value-gradient-like in expectation: the backward pass propagates costates whose conditional expectation equals the value gradient. Second, for discrete transformer policies, we show that autodifferentiation through attention produces empirical costates that approximate this value signal, with an error controlled by the sampling gap and policy entropy. These results motivate a decomposition of RL impact into value gradient signal and reachable reward headroom, yielding a criterion for when RL should be most effective along a pretraining trajectory. 

---
# Amplifying, Not Learning: Fine-Tuned AI Text Detectors Amplify a Pretrained Direction 

**Authors**: Alexander Smirnov  

**Link**: [PDF](https://arxiv.org/pdf/2605.21653)  

**Abstract**: AI text detectors amplify a pretrained typicality axis; they do not construct an AI-vs-human boundary. On raw encoders before any task supervision, projecting onto centroid(AI)-centroid(HC3) achieves NYT-vs-HC3 AUROC 0.806/0.944/0.834 across three architectures (86-106% of the fine-tuned discrimination ceiling: on RoBERTa-base, raw projection exceeds fine-tuning); on RoBERTa-base, full fine-tuning reduces discrimination below raw on both fluent-formal populations tested. The same axis inverts on non-native ESL writing (AUROC 0.06-0.20) -- a falsifiable prediction unique to the typicality reading. A 24-example frozen probe matches full fine-tuning (0.900 vs 0.895). A closed-form Jacobian predictor parameterises axis-manipulating interventions with R^2 = 1.000 universal, lifts ELECTRA-CE deployment TPR from 0.000 to 0.904 at FPR = 1%, and transfers to three independently-trained third-party RoBERTa detectors at 16/16 oracle-equivalence (57% NYT-FPR reduction on the OpenAI detector). Scope: encoder family; mechanism magnitude HC3-anchored; population-level shared axis with per-text mechanisms varying across architectures. Three operationally distinct probes -- text-surface caps_rate residualisation, geometric signed-epsilon ablation, closed-form text-pair predictor -- agree at cos 0.74/0.81/1.00 across three architectures, confirming observer-invariance. Under matched-TPR-0.90 evaluation, the published intervention zoo (CC, dealign-f2c) is calibration-equivalent across 27 cells (|Delta AUROC| <= 0.0081), and >= 97% of the LoRA->full-FT bias gap on ELECTRA is calibration shift, not learned representation -- the central claim's prediction confirmed. 

---
# EntmaxKV: Support-Aware Decoding for Entmax Attention 

**Authors**: Gonçalo Duarte, Miguel Couceiro, Marcos V. Treviso  

**Link**: [PDF](https://arxiv.org/pdf/2605.21649)  

**Abstract**: Long-context decoding is increasingly limited by KV-cache memory traffic since each generated token attends over a cache whose size grows linearly with context length. Existing sparse decoding methods reduce this cost by selecting subsets of tokens or pages, but are designed for softmax attention, whose dense tails make any truncation discard nonzero probability mass. In contrast, $\alpha$-entmax produces exact zeros, turning sparse decoding from dense-tail approximation into support recovery: if the selected candidates contain the entmax support, sparse decoding remains exact. While recent entmax kernels enable efficient training, they do not address the autoregressive decoding bottleneck, where dense inference still streams the full KV cache before sparsity is known. In this work, we introduce EntmaxKV, an entmax-native sparse decoding framework that exploits sparsity before KV pages are loaded. EntmaxKV combines query-aware page scoring, support-aware candidate selection, and sparse entmax attention. We analyze truncation error through the dropped probability mass $\delta$, showing that output error is controlled by $\delta$ and vanishes when the entmax support is recovered. We further introduce a Gaussian-aware entmax selector that estimates the entmax threshold from lightweight page statistics, adapting the selected budget to the score distribution. Empirically, EntmaxKV drops less probability mass, retains more support tokens, and achieves lower output error than softmax-based sparse decoding at matched KV budgets. On long-context and language modeling benchmarks, it closely matches full-cache entmax while using a small fraction of the KV cache, achieving up to $3.36\times$ (softmax) and $5.43\times$ (entmax) speedup over full attention baselines at 1M context length. Code available at: this https URL. 

---
# Flat-Pack Bench: Evaluating Spatio-Temporal Understanding in Large Vision-Language Models through Furniture Assembly 

**Authors**: Aditya Chetan, Eric Cai, Peeyush Kushwaha, Bharath Raj Nagoor Kani, Utkarsh Mall, Qianqian Wang, Noah Snavely, Bharath Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21625)  

**Abstract**: The emergence of Large Vision-Language Models (LVLMs) has significantly advanced video understanding capabilities. However, existing benchmarks focus predominantly on coarse-grained tasks such as action segmentation, classification, captioning, and retrieval. Furthermore, these benchmarks often rely on entities that can be easily identified verbally, like household objects, animals, human subjects, etc., limiting their applicability to complex, in-the-wild video scenarios. But, many applications such as furniture assembly, cooking, etc., require step-by-step fine-grained spatio-temporal understanding of the video, which is not sufficiently evaluated in current benchmarks. To address this gap, we introduce Flat-Pack Bench, a novel benchmark centered on furniture assembly tasks. Our benchmark evaluates LVLMs on nuanced tasks, including temporal ordering of assembly actions, temporal localization of assembly state, understanding part mating, and tracking, using multiple-choice questions paired with visual prompts highlighting relevant parts as references for fine-grained questions. Our experiments reveal that state-of-the-art LVLMs struggle significantly with fine-grained spatio-temporal reasoning, highlighting their limitations in effectively leveraging temporal information from videos, limited tracking ability, and understanding of spatial interactions like physical contact. 

---
# From Parameters to Data: A Task-Parameter-Guided Fine-Tuning Pipeline for Efficient LLM Alignment 

**Authors**: Hao Chen, Qi Zhang, Liyao Li, Zhanming Shen, Wentao Ye, Lirong Gao, Ningtao Wang, Xing Fu, Xiaoyu Shen, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.21558)  

**Abstract**: Adapting Large Language Models (LLMs) to specialized domains typically incurs high data and computational overhead. While prior efficiency efforts have largely treated data selection and parameter-efficient fine-tuning as isolated processes, our empirical analysis suggests they may be intrinsically coupled. We posit the Strong Map Hypothesis: a sparse subset of attention heads plays a dominant role in task-specific adaptation, acting as keys that unlock specific data patterns. Building on this observation, we propose From Parameters to Data (P2D), a unified framework that leverages these task-sensitive attention heads as a dual compass for both sample mining and structural pruning. To rigorously quantify the total pipeline cost, we introduce the Alignment Efficiency Ratio (AER) metric for both selection latency and training time. Mechanistically, P2D identifies critical heads via a lightweight proxy and uses them as a functional filter to curate high-affinity data, establishing a synergistic pipeline. Empirically, by updating merely 10% of attention heads on 10% of the data, P2D achieves an 8.3 pp performance gain over strong baselines and delivers a 7.0x end-to-end time speedup. These results validate that precise parameter-data synchronization eliminates redundancy, offering a new paradigm for efficient alignment. 

---
# Detecting Synthetic Political Narratives in Cross-Platform Social Media Discourse 

**Authors**: Despoina Antonakaki, Sotiris Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2605.21540)  

**Abstract**: The proliferation of large language models has introduced a new paradigm of synthetic political communication in which narratives may be generated, semantically coordinated, and strategically disseminated across platforms at scale. We present a cross-platform framework for detecting synthetic political narratives using four coordination signals -- lexical diversity D(C), temporal burstiness B(C), rhetorical repetition R(C), and semantic homogenization H(C) -- combined into a Synthetic Narrative Coordination Score SNC(C).
We apply the framework to a corpus of 353,223 records spanning six geopolitical event windows collected from six Telegram channels and nine Reddit communities (2023--2026). Results show that IntelSlava exhibits the lowest lexical diversity (MATTR 0.52--0.54), the highest burstiness (B=+0.48 to +0.73), and the highest rhetorical overlap with peer channels (Jaccard 0.12), ranking first in the composite SNC(C) on four of six event windows (SNC 0.45--0.60). Rybar ranks last on all windows despite its high semantic homogenization, because its Russian-language output yields high lexical diversity and near-zero rhetorical Jaccard with English-language channels -- demonstrating that no single indicator is sufficient for coordination detection. Multi-dimensional SNC(C) scoring provides a more robust and interpretable signal than any individual metric. 

---
# HealthCraft: A Reinforcement Learning Safety Environment for Emergency Medicine 

**Authors**: Brandon Dent  

**Link**: [PDF](https://arxiv.org/pdf/2605.21496)  

**Abstract**: Frontier language models are being deployed into clinical workflows faster than the infrastructure to evaluate them safely. Static medical-QA benchmarks miss the failure modes that matter in emergency medicine: trajectory-level safety collapse, tool misuse, and capitulation under sustained clinical pressure. We present HealthCraft, the first public reinforcement-learning environment that rewards trajectory-level safety under realistic emergency-medicine conditions, adapted from Corecraft. It is built on a FHIR R4 world state with 14 entity types and 3,987 seed entities, exposes 24 MCP tools, and defines a dual-layer rubric that zeroes reward whenever any safety-critical criterion is violated. We release 195 tasks across six categories, graded against 2,255 binary criteria (515 safety-critical); a post-hoc 10-task negative-class slate extends this to 205 tasks and 2,337 criteria. V8 results on two frontier models show Claude Opus 4.6 at Pass@1 24.8% [21.5-28.4] and GPT-5.4 at 12.6% [10.2-15.6], with safety-failure rates of 27.5% and 34.0%. On multi-step workflows - the closest proxy to real emergency care - performance collapses to near zero (Claude 1.0%, GPT-5.4 0.0%) despite partial competence on individual steps. Six infrastructure bugs fixed between pilots v2 and v8 re-ordered which model "looks stronger," evidence that infrastructure fidelity is part of the measurement. A deterministic LLM-judge overlay bounds evaluator noise, and a 60-run negative-class smoke pilot shows the reward signal is not drop-in training-safe: restraint criteria pass at 0.929 prevalence, a gameability an eval harness can tolerate but a training reward cannot. We scaffold coupling to a Megatron+SGLang+GRPO loop per Corecraft Section 5.2 and leave training-reward ablations as future work. Environment, tasks, rubrics, and harness are released under Apache 2.0. 

---
# Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation 

**Authors**: Srujan P Mule, Aniketh Garikaparthi, Manasi Patwardhan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21491)  

**Abstract**: As language models accelerate scientific research by automating hypothesis generation and implementation, a new bottleneck emerges: evaluating and filtering hundreds of AI-generated ideas without exhaustive experimentation. We ask whether LMs can learn to forecast the empirical success of research ideas before any experiments are run. We study comparative empirical forecasting: given a benchmark-specific research goal and two candidate ideas, predict which will achieve better benchmark performance. We construct a dataset of 11,488 idea pairs grounded in objective outcomes from PapersWithCode. While off-the-shelf 8B-parameter models struggle (30% acc.), SFT dramatically boosts performance to 77.1%, outperforming GPT-5 (61.1%). By framing evaluation as a reasoning task via Reinforcement Learning with Verifiable Rewards (RLVR), we train models to discover latent reasoning paths, achieving 71.35% acc. with interpretable justifications. Through additional ablations and out-of-distribution tests, we show robustness to surface-level heuristics and transfer to both a cross-domain time-split test set and an independently constructed test set. Our results demonstrate that compute-efficient small language models can serve as effective, objective verifiers, offering a scalable path for autonomous scientific discovery. 

---
