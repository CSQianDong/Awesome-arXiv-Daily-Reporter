# ChronoMedKG: A Temporally-Grounded Biomedical Knowledge Graph and Benchmark for Clinical Reasoning 

**Authors**: Md Shamim Ahmed, Farzaneh Firoozbakht, Lukas Galke Poech, Jan Baumbach, Richard Röttger  

**Link**: [PDF](https://arxiv.org/pdf/2605.22734)  

**Abstract**: Biomedical knowledge graphs (KGs) treat disease associations as static facts, but temporal information is crucial for clinical reasoning, e.g., a symptom diagnostic of one disease at age 3 may imply a different disease at age 13. Existing KGs such as PrimeKG, Hetionet, and iKraph do not encode when a finding becomes clinically relevant over the course of a disease. This limits their usefulness for longitudinal clinical reasoning and retrieval augmentation.
We introduce ChronoMedKG, a temporal biomedical knowledge graph that contains 460,497 evidence-linked triples (filtered from 13M raw extractions) covering 13,431 diseases. Each association is tied to temporal components like onset window or progression stage, which are backed by PMID-traceable evidence and a multi-signal credibility score. The graph is constructed through a disease-autonomous multi-agent pipeline in which multiple frontier LLMs independently extract knowledge from PubMed and PMC literature. Only those relations are kept that are supported by multi-model consensus, survive credibility filtering, as well as ontology alignment.
ChronoMedKG scored 92.7% agreement against Orphadata and adds temporal grounding for 6,250 diseases absent from HPOA, Orphadata, and Phenopackets, including 1,657 Orphanet-coded rare diseases. We further introduce ChronoTQA, a benchmark of 3,341 questions across eight task types (six temporal plus two static controls), with a 12-question supplementary probe. Frontier LLMs lose roughly 30 points moving from static to temporal questions; ChronoMedKG retrieval rescues 47-65% of their long-tail failures, against 17-29% for HPOA-RAG. As such, ChronoMedKG provides a crucial temporal axis for retrieval-augmented clinical systems that was previously absent. 

---
# Evaluating Commercial AI Chatbots as News Intermediaries 

**Authors**: Mirac Suzgun, Emily Shen, Federico Bianchi, Alexander Spangher, Thomas Icard, Daniel E. Ho, Dan Jurafsky, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.22785)  

**Abstract**: AI chatbots are rapidly shaping how people encounter the news, yet no prior study has systematically measured how accurately these systems, with their proprietary search integrations and retrieval-synthesis pipelines, handle emerging facts across languages and regions. We present a 14-day (February 9-22, 2026) evaluation of six AI chatbots (Gemini 3 Flash and Pro, Grok 4, Claude 4.5 Sonnet, GPT-5 and GPT-4o mini) on 2,100 factual questions derived from same-day BBC News reporting across six regional services (US & Canada, Arabic, Afrique, Hindi, Russian, Turkish). The best systems achieve over 90% multiple-choice accuracy on questions about events reported hours earlier. The same systems, however, lose 11-13% under free-response evaluation, and 16-17% across the cohort. We further characterize three failure patterns. First, every model achieves its lowest accuracy on Hindi (79% vs. 89-91% elsewhere) and citations indicate an Anglophone retrieval bias (e.g., models answering Hindi queries cite English Wikipedia more than any Hindi outlet). Second, retrieval, not reasoning, failures drive over 70% of all errors. When models retrieve a correct source, they often extract the correct answer; the problem is to land on the right source in the first place. Third, models achieving 88-96% accuracy on well-formed questions drop to 19-70% when questions contain subtle false premises, with the most vulnerable model accepting fabricated facts 64% of the time. We also identify a detection-accuracy paradox: the best false-premise detector ranks second in adversarial accuracy (abstention rate), while a weaker detector ranks first, showing that premise detection and answer recovery are partially independent capabilities. Overall, these suggest that high accuracy can mask systematic regional inequity, near-total dependence on retrieval infrastructure, and vulnerability to imperfect queries real users pose. 

---
# Seeing the Poem: Image-Semantic Detection of AI-Generated Modern Chinese Poetry with MLLMs 

**Authors**: Shanshan Wang, Fengying Ye, Hanjia Lyu, Caiwen Gou, Junchao Wu, Jingming Yao, Chengzhong Xu, Jiebo Luo, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2605.22654)  

**Abstract**: Previous detection studies have shown that LLMs cannot be effectively used as detectors, but these studies have not addressed modern Chinese poetry. Moreover, no relevant research has explored the performance of LLMs in detecting modern Chinese poetry. This paper evaluates and enhances the performance of LLMs as detectors for modern Chinese poetry, and proposes an image-semantic guided poetry detection method. Compared with traditional detection approaches, our method innovatively incorporates images that reflect the content of the poetry. Through example-driven approaches, our method effectively integrates information such as meaning, imagery, and feeling from the image, then forms a complementary judgment with the poem text. Experimental results demonstrate that the LLM detectors based on our method outperform baseline detectors based on plain text, and even surpass the best-performing traditional detector, RoBERTa. The Gemini detector using our method achieves a Macro-F1 score of 85.65%, reaching the state-of-the-art level. The performance improvements of different LLM detectors on multiple LLMs-generated data prove the effectiveness of our method. 

---
# Understanding Data Temporality Impact on Large Language Models Pre-training 

**Authors**: Pilchen Hippolyte, Fabre Romain, Signe Talla Franck, Perez Patrick, Grave Edouard  

**Link**: [PDF](https://arxiv.org/pdf/2605.22769)  

**Abstract**: Large language models (LLMs) are typically trained on shuffled corpora, yielding models whose knowledge is frozen at train time and whose temporal grounding remains poorly understood. In this work, we study the impact of pre-training dynamics on the acquisition of time-sensitive factual knowledge, focusing specifically on data ordering. Our main contributions are twofold. First, we introduce a comprehensive benchmark of over 7,000 temporally grounded questions and an evaluation protocol that enables analysis of whether models correctly associate facts with their corresponding time periods. Second, we pretrain 6B-parameter models on temporally ordered Common Crawl snapshots and compare them against standard shuffled pre-training. Our results show that sequentially trained models match shuffled baselines on general language understanding and common knowledge while consistently exhibiting more up-to-date and temporally precise knowledge. Temporally ordered pre-training yields improved factual freshness, while shuffled pre-training peaks on older data, possibly due to increased factual repetition. These findings, along with the release of our code at this https URL , checkpoints, and datasets at this https URL provide a foundation for future research on continual learning for LLMs. 

---
# Moral Semantics Survive Machine Translation: Cross-Lingual Evidence from Moral Foundations Corpora 

**Authors**: Maciej Skorski  

**Link**: [PDF](https://arxiv.org/pdf/2605.22660)  

**Abstract**: Moral language is subtle and culturally variable, making it difficult to translate faithfully across languages. Idiomatic expressions, slang, and cultural references introduce hard-to-avoid translation artifacts. Yet automated moral values classification depends on language-specific annotated corpora that exist almost exclusively in English. We investigate whether LLM-based translation can bridge this gap, taking Polish as a test case. Using $\sim$50k morally-annotated social media posts from a diverse range of topics, we apply a principled four-method validation pipeline: LaBSE cross-lingual embedding similarity, Centered Kernel Alignment (CKA), LLM-as-judge evaluation, and deep learning classifier parity tests. We show that despite shortcomings in handling slang, vulgarity, and culturally-loaded expressions, direct translation preserves subtle moral cues well enough to be harvested by cross-lingual machine learning -- with mean cosine similarity of 0.86 and AUC gaps of 0.01--0.02 across all foundations closing further under fine-tuning of language models. These results demonstrate that machine translation is a practical and cost-effective path to moral values research in languages currently under-resourced in this domain. We demonstrate this for Polish as a representative Slavic language, with expected generalisation to related languages. 

---
# Reducing Political Manipulation with Consistency Training 

**Authors**: Long Phan, Devin Kim, Alexander Pan, Alice Blair, Adam Khoja, Dan Hendrycks  

**Link**: [PDF](https://arxiv.org/pdf/2605.22771)  

**Abstract**: Large language models (LLMs) exhibit systematic political bias across a variety of sensitive contexts. We find that LLMs handle counterpart topics from opposing political sides asymmetrically. We refer to this phenomenon as covert political bias and identify 7 categories of techniques through which it operates. We propose two metrics for covert bias: Sentiment Consistency measures symmetry in rhetoric and framing across paired political prompts; Helpfulness Consistency measures symmetric depth and engagement. To reduce both types of covert bias, we introduce Political Consistency Training (PCT), an RL training method with two complementary paradigms: Sentiment Consistency Training and Helpfulness Consistency Training. We show that PCT preserves overall helpfulness, substantially reduces covert political bias, and generalizes to held-out benchmarks. We release our work at this https URL 

---
# Self-Policy Distillation via Capability-Selective Subspace Projection 

**Authors**: Guangya Hao, Yitong Shang, Yunbo Long, Zhuokai Zhao, Hanxue Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22675)  

**Abstract**: Self-distillation bootstraps large language models (LLMs) by training on their own generations. However, existing methods either rely on external signals to curate self-generated outputs (e.g., correctness filtering, execution feedback, and reward search), which are costly and unavailable for the best-performing frontier models, or skip curation entirely and train on all raw outputs, an approach that is often domain-specific and hard to generalize. Both also share a deeper weakness that self-generated outputs entangle task-relevant capability with others, such as stylistic patterns, formatting artifacts, and model-specific errors, diluting the signal for the specific capability one aims to improve. In this paper, we propose Self-Policy Distillation (SPD), which achieves generalizable, capability selective without any external signal. Specifically, SPD extracts a low-rank capability subspace from the model's own gradients on correctness-defining tokens, projects key-value (KV) activations into this subspace during self-generation, and fine-tunes on the resulting raw outputs with standard next-token prediction loss. Through extensive experiments across code generation, mathematical reasoning, and multiple-choice QA, we show that SPD achieves up to 13% improvement over state-of-the-art self-distillation methods without external signals and up to 16% improvement over pre-trained baselines. Notably, SPD demonstrates superior generalizability, achieving 15% better performance under out-of-domain generalization settings. 

---
# Boiling the Frog: A Multi-Turn Benchmark for Agentic Safety 

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Federico Sartore, Enrico Panai, Laura Caroli, Yue Zhu, Adam Leon Smith, Luca Nannini, Marcello Galisai, Susanna Cifani, Francesco Giarrusso, Marcantonio Bracale Syrnikov, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2605.22643)  

**Abstract**: Background. Traditional safety benchmarks for language models evaluate generated text: whether a model outputs toxic language, reproduces bias, or follows harmful instructions. When models are deployed as agents, the safety-relevant object shifts from what the system says to what it does within an environment, and evaluating model responses under prompting is no longer sufficient to address the safety challenges posed by artificial intelligence. Recent developments have seen the rise of benchmarks that evaluate large language models as agents. We contribute to this strand of research. Approach. We introduce Boiling the Frog, a benchmark that evaluates whether tool-using AI models deployed in corporate and office settings are susceptible to incremental attacks. Each scenario begins with benign workspace edits and later introduces a risk-bearing request. The benchmark focuses on stateful multi-turn evaluation: chains expose a persistent workspace, place the risk-bearing payload at controlled positions in the turn sequence, and score whether the resulting artifact state becomes unsafe. Scenarios are organized through a three-level operational risk taxonomy grounded in the Boiling the Frog risks, the AI Act Annex I and Annex III high-risk contexts, and EU AI Act's Code of Practice on General-Purpose AI (GPAI). Results. Across a nine-model panel, aggregate strict attack success rate (ASR) is 44.4%. Model-level ASR ranges from 20.5% for Claude Haiku 4.5 to 92.9% for Gemini 3.1 Flash Lite, with Seed 2.0 Lite also above 80%. Average chain category-level ASR reaches 93.3% for Code of Practice loss-of-control scenarios. 

---
# LANG: Reinforcement Learning for Multilingual Reasoning with Language-Adaptive Hint Guidance 

**Authors**: Yuchun Fan, Bei Li, Peiguang Li, Yilin Wang, Yongyu Mu, Jian Yang, Xin Chen, Rongxiang Weng, Jingang Wang, Xunliang Cai, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.22567)  

**Abstract**: Reinforcement learning has proven effective for enhancing multi-step reasoning in large language models (LLMs), yet its benefits have not fully translated to multilingual contexts. Existing methods struggle with a fundamental trade-off: prioritizing input-language consistency severely hampers reasoning quality, while prioritizing reasoning often leads to unintended language drift toward English. We address this challenge with LANG, a novel framework that leverages language-conditioned hints to guide exploration in non-English reasoning tasks. Our method incorporates two key mechanisms to prevent dependency on these hints: a progressive decay schedule that gradually withdraws scaffolding, and a language-adaptive switch that tailors learning horizons to specific language difficulties. Empirical results on challenging multilingual mathematical benchmarks reveal that LANG substantially enhances reasoning performance without compromising language consistency. Moreover, we show that our framework generalizes beyond mathematics, fostering more consistent language alignment across model layers 

---
# More Context, Larger Models, or Moral Knowledge? A Systematic Study of Schwartz Value Detection in Political Texts 

**Authors**: Víctor Yeste, Paolo Rosso  

**Link**: [PDF](https://arxiv.org/pdf/2605.22641)  

**Abstract**: Detecting Schwartz values in political text is difficult because implicit cues often depend on surrounding arguments and fine-grained distinctions between neighboring values. We study when context and explicit moral knowledge help sentence-level value detection. Using the ValuesML/Touch{é} ValueEval format, we compare sentence, window, and full-document inputs; no-RAG and retrieval-augmented settings with a curated moral knowledge base; supervised DeBERTa-v3-base/large encoders; and zero-shot LLMs from 12B to 123B parameters. The results show that more context is not uniformly better: full-document context improves supervised DeBERTa encoders by 3.8--4.8 macro-F1 points over sentence-only input, but does not consistently help zero-shot LLMs. Retrieved moral knowledge is more consistently useful in matched comparisons, improving each tested model family and context condition under early fusion. However, scaling from DeBERTa-v3-base to large and from 12B to larger LLMs does not guarantee gains, and simple early fusion outperforms the tested late-fusion and cross-attention RAG variants for encoders. Per-value analyses show that context and retrieval help most for socially situated or conceptually confusable values. These findings suggest that value-sensitive NLP should evaluate context, knowledge, and model family jointly rather than treating longer inputs or larger models as universal improvements. 

---
# Beyond Temperature: Hyperfitting as a Late-Stage Geometric Expansion 

**Authors**: Meimingwei Li, Yuanhao Ding, Esteban Garces Arias, Christian Heumann  

**Link**: [PDF](https://arxiv.org/pdf/2605.22579)  

**Abstract**: Recent work has identified a counterintuitive phenomenon termed "Hyperfitting", where fine-tuning Large Language Models (LLMs) to near-zero training loss on small datasets surprisingly enhances open-ended generation quality and mitigates repetition in greedy decoding. While effective, the underlying mechanism remains poorly understood, with the extremely low-entropy output distributions suggesting a potential equivalence to simple temperature scaling. In this work, we demonstrate that this phenomenon is fundamentally distinct from distribution sharpening; entropy-matched control experiments reveal that temperature scaling fails to replicate the diversity gains of hyperfitting. Furthermore, we falsify the hypothesis of static vocabulary reweighting, showing through ablation studies that hyperfitting relies on a dynamic, context-dependent rank reordering mechanism. Layer-wise analysis localizes this effect to a "Terminal Expansion" in the final transformer block, where a substantial geometric expansion of the feature space (Delta Dim approx +80.8) facilitates the promotion of deep-tail tokens. Additionally, we introduce Late-Stage LoRA, a targeted fine-tuning strategy that updates only the final 5 layers, yielding robust generation with minimal parameter updates 

---
# Agentic CLEAR: Automating Multi-Level Evaluation of LLM Agents 

**Authors**: Asaf Yehudai, Lilach Eden, Michal Shmueli-Scheuer  

**Link**: [PDF](https://arxiv.org/pdf/2605.22608)  

**Abstract**: Agentic systems are becoming more capable: agents define strategies, take actions, and interact with different environments. This autonomy poses serious challenges for overseeing and assessing agent behavior. Most current tools are limited, focusing on observability with basic evaluation capabilities or imposing static, hand-crafted error taxonomies that cannot adapt to new domains. To address this gap, we present Agentic CLEAR, an automatic, dynamic, and easy-to-use evaluation framework. It produces textual insights into the agent behavior on three levels of granularity: system, trace, and node. Agentic CLEAR operates above the observability layer, enabling seamless integration and featuring an intuitive UI that makes agent evaluation highly accessible. In our experiments on four benchmarks, seven agentic settings, and tens of thousands of LLM calls, we show that Agentic CLEAR produces high-quality, data-driven, insightful feedback. Our analysis shows strong alignment with human-annotated errors and the ability to predict task success rate. 

---
# SynAE: A Framework for Measuring the Quality of Synthetic Data for Tool-Calling Agent Evaluations 

**Authors**: Shuaiqi Wang, Aadyaa Maddi, Zinan Lin, Giulia Fanti  

**Link**: [PDF](https://arxiv.org/pdf/2605.22564)  

**Abstract**: Today, tool-calling agents are commonly evaluated or tested on static datasets of execution traces, including input commands, agent responses, and associated tool calls. However, internal production datasets are often insufficient or unusable for testing; for example, they may contain sensitive or proprietary data, or they may be too sparse to support comprehensive testing (especially pre-deployment). In these settings, practitioners are increasingly replacing or augmenting real datasets with synthetic ones for evaluation purposes. A key challenge is quantifying the relation between these synthetic datasets and the real data. We introduce SynAE, an evaluation framework for assessing how well synthetic benchmarks for multi-turn, tool-calling agents replicate and augment the characteristics of real data trajectories. SynAE assesses the validity, fidelity, and diversity of synthetic data across four metric categories: (i) task instructions and intermediate responses, (ii) tool calls, (iii) final outputs, and (iv) downstream evaluation. We evaluate SynAE using recent agent benchmarks and test common synthetic data failure modes via realistic and controlled generation schemes. SynAE detects fine-grained variations in data validity, fidelity and diversity, and shows that no single metric is sufficient to fully characterize synthetic data quality, motivating a multi-axis evaluation of synthetic data for agent testing. A demo of SynAE is available at this https URL, with code at this https URL. 

---
# Scene Abstraction for Lexical Semantics: Structured Representations of Situated Meaning 

**Authors**: Yejin Cho, Katrin Erk  

**Link**: [PDF](https://arxiv.org/pdf/2605.22542)  

**Abstract**: Coffee and tea share many properties, yet they evoke strikingly different situations, atmospheres, and affective associations. These situated dimensions of word meaning are real and systematic, but they remain implicit in most computational representations of lexical meaning. We propose Scene Abstraction, a framework for constructing structured representations of the interpretive scenes that words participate in across usage contexts. Each scene consists of a Contextual Scene (Events, Entities, Setting) and an expression-centered Expression Profile (Engaged events, Generalizable properties, Evoked emotions), operationalized through few-shot prompting of a large language model. Our contributions are three-fold: (1) a structured representation framework for situated lexical meaning; (2) COCA-Scenes, a dataset of 520 usage instances across 26 keywords for distinct scene identification; and (3) empirical evidence from two experiments suggesting that scenes are reliably identifiable across human observers (82.4% accuracy, +11.8 pp over text-only embeddings) and that our scene profiles more closely align with human interpretation of words in context than ATOMIC-based alternatives (86.4% preference across three semantic dimensions). 

---
# Polite on the Surface, Wrong in Practice: A Curated Dataset for Fixing Honorific Failures in Multilingual Bangla Generation 

**Authors**: Md. Asaduzzaman Shuvo, Mahedi Hasan, Md. Tashin Parvez, Azizul Haque Noman, Md. Shafayet Hossain Ovi  

**Link**: [PDF](https://arxiv.org/pdf/2605.22487)  

**Abstract**: Recent advances in Multilingual Large Language Models (MLLMs) have significantly enhanced cross-lingual conversational capabilities, yet modeling culturally nuanced and context-dependent communication remains a critical bottleneck. Specifically, existing state-of-the-art models exhibit a severe pragmatic gap when handling structural variations, regional idioms, and honorific consistencies in low-resource contexts like Bangla. To address this limitation, we introduce a novel, culturally aligned instruction-tuning dataset for \textbf{BangLa Application and DialoguE generation - BLADE} and benchmarking framework comprising $4,196$ meticulously curated interaction pairs. We leverage this resource to systematically fine-tune and evaluate leading open-weight architectures, including DeepSeek-8B and LLaMA-3.2-3B, utilizing parameter-efficient fine-tuning via LoRA adapters in a 4-bit NormalFloat (NF4) quantization framework. Our empirical evaluations demonstrate that models fine-tuned on our dataset yield substantial improvements in structural fidelity and honorific alignment, providing a rigorous benchmark for bridging pragmatic disparities in low-resource multilingual text generation. Code and dataset: this https URL 

---
# BeLink: Biomedical Entity Linking Meets Generative Re-Ranking 

**Authors**: Darya Shlyk, Stefano Montanelli, Lawrence Hunter  

**Link**: [PDF](https://arxiv.org/pdf/2605.22501)  

**Abstract**: Despite recent progress, Biomedical Entity Linking (BEL) with large language models (LLMs) remains computationally inefficient and challenging to deploy in practical settings. In this work, we demonstrate that instruction-tuning of open-source generative models can offer an effective solution when applied at the re-ranking stage of the BEL pipeline. We propose a set-wise instruction-tuning formulation that enables fast and accurate candidate selection. Our method demonstrates strong performance on multiple BEL benchmarks, yielding significant improvements in linking accuracy (3%-24%) while reducing inference time compared to the state-of-the-art. We integrate our generative re-ranker into BeLink, a modular, end-to-end system designed for practical real-world BEL applications. 

---
# Unified Data Selection for LLM Reasoning 

**Authors**: Xiaoyuan Li, Yubo Ma, Chengpeng Li, Fengbin Zhu, Yiyao Yu, Keqin Bao, Wenjie Wang, Fuli Feng, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22389)  

**Abstract**: Effectively training Large Language Models (LLMs) for complex, long-CoT reasoning is often bottlenecked by the need for massive high-quality reasoning data. Existing methods are either computationally expensive or fail to reliably distinguish high- from low-quality reasoning samples. To address this, we propose High-Entropy Sum (HES), a training-free metric that quantifies reasoning quality by summing only the entropy of the top (e.g., 0.5\%) highest-entropy tokens in each reasoning sample. We validate HES across three mainstream training paradigms: Supervised Fine-tuning (SFT), Rejection Fine-tuning (RFT), and Reinforcement Learning (RL), with extensive results demonstrating its consistent effectiveness and significantly reduced computational overhead. In SFT, training on the top 20\% HES-ranked data matches full-dataset performance, while using the lowest-HES data degrades it. In RFT, our HES-based training approach significantly outperforms baseline methods. In RL, HES-selected successful trajectories enable the model to learn strong reasoning patterns, significantly surpassing other compared methods. Our findings establish HES as a robust, training-free metric that enables a unified, effective, and efficient method for developing advanced reasoning in LLMs. 

---
# DeferMem: Query-Time Evidence Distillation via Reinforcement Learning for Long-Term Memory QA 

**Authors**: Jianing Yin, Tan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22411)  

**Abstract**: Large language model (LLM) agents still struggle with long-term memory question answering, where answer-supporting evidence is often scattered across long conversational histories and buried in substantial irrelevant content. Existing memory systems typically process memory before future queries are known, then retrieve the resulting units based on similarity rather than their utility for answering the query. This workflow leaves downstream answerers to denoise retrieved candidates and reconstruct query-specific evidence. We present DeferMem, a long-term memory framework that decouples this problem into high-recall candidate retrieval and query-conditioned evidence distillation. DeferMem uses a lightweight segment-link structure to organize raw history and retrieve broad candidates at query time. It then applies a memory distiller trained with DistillPO, our reinforcement learning algorithm for distilling the high-recall but highly noisy candidates into a set of faithful, self-contained, and query-conditioned evidence. DistillPO formulates post-retrieval evidence distillation as a structured action comprising message selection and evidence rewriting. It optimizes this action with a decomposed-and-gated reward pipeline and structure-aligned advantage assignment, gating reward components from validity to quality checks while exposing task-level correctness feedback early and assigning each reward to its responsible output span. On LoCoMo and LongMemEval-S, DeferMem surpasses strong baselines in QA accuracy and memory-system efficiency, achieving the highest QA accuracy with the fastest runtime and zero commercial-API token cost for memory operations. 

---
# Assisted Counterspeech Writing at the Crossroads of Hate Speech and Misinformation 

**Authors**: Genoveffa Martone, Helena Bonaldi, Marco Guerini  

**Link**: [PDF](https://arxiv.org/pdf/2605.22435)  

**Abstract**: Hate speech and misinformation frequently co-occur online, amplifying prejudice and polarization. Given their scale, using Large Language Models (LLMs) to assist expert counterspeech (CS) writing has gained interest, yet prior work has addressed these phenomena separately. We bridge this gap by studying CS generation in contexts where both hate and misinformation co-occur. We test three knowledge-driven generation strategies: first we prompt an LLM with fact-checkers' guidelines and fact-checking articles; secondly, with NGOs' guidelines and reports; thirdly, we create a mixed strategy that combines guidelines and documents from both. 23 experts revise the generated CS, which are assessed via human and automatic metrics. While LLMs produce adequate CS in 40% of cases, expert edits substantially improve naturalness, exhaustiveness, and adherence to guidelines. Based on the post-edited CS, the mixed strategy proves to be the most effective in crowdsourcing evaluation, pairing strong factual correction with stereotype mitigation and empathetic engagement. We release a dataset of hateful and misinformed claims with expert-verified CS and supporting knowledge. 

---
# From Correlation to Cause: A Five-Stage Methodology for Feature Analysis in Transformer Language Models 

**Authors**: Caleb Munigety  

**Link**: [PDF](https://arxiv.org/pdf/2605.22462)  

**Abstract**: We propose a five-stage methodology for causal feature analysis in transformer language models (probe design, feature extraction, causal validation, robustness testing, and deployment integration) and demonstrate it end-to-end on GPT-2 small performing the Indirect Object Identification (IOI) task. Activation patching recovers the canonical IOI circuit (layer-9 head 9 alone gives recovery +1.02). A sparse autoencoder recovers per-name selective features with effect sizes of 30 to 50 activation units. Causal validation finds these features specifically but only partially causal: ablating fifteen of them leaves the model accurate on 98% of prompts. Two NLA-inspired evaluations strengthen this picture: the fifteen selective features explain only 31% of activation variance versus the SAE's 99.7%, and selectivity ratio anticorrelates with causal force (r = -0.56). Robustness testing under three distribution shifts finds that the circuit transfers cleanly but feature ablation effects degrade substantially, exposing a gap between detection robustness and causal robustness. A cost-based deployment evaluation (assumed $50/FN, $0.42/FP, 2% error rate) finds an optimal monitor configuration yielding $8.96 per 1000 queries against a $1000 baseline, a 99.1% saving. Optimal composition strategy varies with cost ratio and base rate. The conjunction of stages produces findings no single stage would. 

---
# TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation 

**Authors**: Hanyu Guo, Jiedong Yang, Chao Chen, Longfei Xu, Kaikui Liu, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22355)  

**Abstract**: Public transit route planning traditionally depends on structured map infrastructure and complex routing engines, and no existing dataset supports training models to bypass this dependency. We present TransitLM, a large-scale dataset of over 13 million transit route planning records from four Chinese cities covering 120,845 stations and 13,666 lines, released as a continual pre-training corpus and benchmark data for three evaluation tasks with complementary metrics. Experiments show that an LLM trained on TransitLM produces structurally valid routes at high accuracy and implicitly grounds arbitrary GPS coordinates to appropriate stations without any explicit mapping. These results demonstrate that transit route planning can be learned entirely from data, enabling end-to-end, map-free route generation directly from origin-destination information. The dataset and benchmark are available at this https URL, with evaluation code at this https URL. 

---
# IdioLink: Retrieving Meaning Beyond Words Across Idiomatic and Literal Expressions 

**Authors**: Kai Golan Hashiloni, Daniel Fadlon, Lior Livyatan, Ofri Hefetz, Jiahuan Pei, Kfir Bar  

**Link**: [PDF](https://arxiv.org/pdf/2605.22247)  

**Abstract**: Idioms pose a fundamental challenge for language models, as their meaning cannot be inferred from surface form alone. Understanding such expressions, therefore, requires semantic abstraction beyond lexical overlap. We introduce IdioLink, a retrieval benchmark designed to test whether models can link idiomatic expressions to conceptually equivalent meanings expressed in literal or paraphrased forms. IdioLink comprises 10,700 documents and 2,140 queries, spanning 107 idioms with both literal and figurative uses. Each document and query is annotated with spans that convey the core meaning. Evaluating strong embedding baselines (e.g., BGE, E5, Contriever, and Qwen), we show that current models struggle to retrieve equivalent meanings across divergent surface realizations, relying instead on topical and shallow semantic cues. IdioLink exposes key gaps in idiom-aware semantic retrieval and provides a challenging testbed for future models. 

---
# Modeling Pathology-Like Behavioral Patterns in Language Models Through Behavioral Fine-Tuning 

**Authors**: Nicola Milano, Davide Marocco  

**Link**: [PDF](https://arxiv.org/pdf/2605.22356)  

**Abstract**: Large language models are increasingly used as computational tools for modeling human-like behavior. We introduce a behavioral induction framework that modifies model policies through fine-tuning on structured decision-making tasks: using synthetic datasets inspired by maladaptive behavioral patterns, including depression and paranoia, we train transformer-based language models to consistently select specific classes of actions across diverse contexts. We then test whether this behavioral optimization produces systematic changes in generative distributions.
Across two architectures, fine-tuned models show stable, context-general shifts in next-token probability distributions, including increased probability assigned to negative and threat-related interpretations in open-ended language tasks. These effects generalize beyond training contexts and are detectable in qualitative completions, psychometric-style evaluations, and quantitative distributional metrics such as Jensen-Shannon divergence.
Induced behavioral profiles also show partial specificity. Models optimized for different behavioral patterns exhibit dissociable response tendencies across evaluation probes, suggesting that structured behavioral training produces differentiated policy-level biases rather than generic distributional skew.
We interpret these findings as evidence that consistent behavioral optimization in LLMs can generate stable behavioral and distributional patterns consistent with altered latent priors, linking action selection and language generation. More broadly, the results support a view of LLMs as policy-based systems in which behavioral constraints shape emergent representational structure, highlighting their potential as controlled testbeds for studying the relationship between behavior, interpretation, and generative language in computational models of cognition. 

---
# Do Factual Recall Mechanisms Carry over from Text to Speech in Multimodal Language Models? 

**Authors**: Luca Modica, Filip Landin, Mehrdad Farahani, Livia Qian, Gabriel Skantze, Richard Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2605.22170)  

**Abstract**: In recent years, several Speech Language Models (SLMs) that represent speech and written text jointly have been presented. The question then emerges about how model-internal mechanisms are similar and different when operating in the two modalities. We focus on how these systems encode, store, and retrieve factual knowledge, which has previously been investigated for text-only models. To investigate mechanisms behind the storage and recall of factual association in SLMs, we leverage Causal Mediation Analysis, a technique previously applied to text-based models.
Initial results using SpiritLM, a multimodal model integrating discrete speech tokens reveal discrepancies between text-to-text and speech-to-text results, suggesting that the emergent mechanisms for factual recall are only partially carried over from the text to the speech modality. These results advance our understanding of how internal mechanisms encode factual associations in SLMs while contributing insights for improving speech-enabled AI systems. 

---
# Harder to Defend: Towards Chinese Toxicity Attacks via Implicit Enhancement and Obfuscation Rewriting 

**Authors**: Jingyi Kang, Junyu Lu, Bo Xu, Hongbo Wang, Linlin zong, Roy Ka-Wei Lee, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.22258)  

**Abstract**: Large language models (LLMs) require robust toxicity evaluation beyond explicit wording. This setting remains underexplored in Chinese, where toxicity may combine semantic indirectness with surface obfuscation. We introduce Chinese Implicit Toxicity Attack (CITA), a controlled red-team evaluation and defense-data generation framework, not a deployable evasion tool. CITA uses three stages: (i) Harmful Intent Learning, (ii) Implicit Toxicity Enhancement, and (iii) Obfuscation Variant Rewriting, to preserve harmful intent, increase implicitness, and add controlled surface variants. On CITA-generated evaluation samples, the seven tested detectors exhibit substantial missed-detection risks, reaching an average ASR of 69.48%; human evaluation further confirms preserved harmfulness and increased implicitness/evasiveness. As a downstream defense application, we fine-tune a Chinese Implicit Toxicity Defense model (CITD) with CITA-generated red-team data, showing that such data can improve robustness through additional training. 

---
# Cross-Lingual Consensus: Aligning Multilingual Cultural Knowledge via Multilingual Self-Consistency 

**Authors**: Andrew Ivan Soegeng, Patrick Sutanto, Tan Sang Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.22137)  

**Abstract**: Although Large Language Models (LLMs) demonstrate strong capabilities across various tasks, they exhibit significant performance discrepancies across languages. While prompting LLMs in English typically yields the highest general performance, it often induces a Western-centric bias, hindering the model's ability to accurately reflect diverse cultural knowledge. We hypothesize that LLMs already possess rich cultural knowledge embedded within local-language representations, but fail to retrieve it when prompted in English. To bridge this cross-lingual knowledge gap, we propose a novel self-supervised framework. Our method leverages multilingual self-consistency to identify the most reliable cultural responses across languages, combined with a self-critique mechanism to transfer this knowledge to the weaker language. Evaluations on the BLEnD benchmark demonstrate that our approach significantly improves cultural alignment-boosting performance on English queries by an average of 5.03%-relying entirely on self-generated data. Ultimately, our work demonstrates that latent cultural knowledge can be successfully surfaced and propagated across languages, enabling more culturally equitable and consistent LLMs. 

---
# Psy-Chronicle:A Structured Pipeline for Synthesizing Long-Horizon Campus Psychological Counseling Dialogues 

**Authors**: Chaogui Gou, Jiarui Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22140)  

**Abstract**: In recent years, large language models have shown substantial potential in psychological support tasks. However, existing psychological counseling data mostly rely on single-turn question answering or short multi-turn dialogues, making it difficult to characterize how college students' psychological distress accumulates, interacts, and gradually evolves over long periods within campus life events. To address this issue, this paper proposes Psy-Chronicle, a structured data-generation framework for synthesizing long-horizon campus psychological counseling dialogues. We generate a semester-spanning temporal stress event graph to model the chronological order and evolutionary dependencies among campus stress events. Through interactive simulation between a student agent and a counselor agent, together with a structured memory integration mechanism, Psy-Chronicle generates long-horizon dialogues with continuity across counseling sessions. Based on Psy-Chronicle, we construct and open-source CPCD, a Chinese long-horizon dialogue dataset for college psychological counseling, containing 100 student profiles, 90,000 counseling dialogues. We further build CPCD-Bench to evaluate models' long-horizon campus counseling capabilities from three dimensions: session-level response, long-horizon memory recall, and temporal-causal reasoning. Experimental results show that CPCD effectively improves session-level response generation and long-horizon memory recall for models with the same base architecture. Meanwhile, improvements in temporal-causal reasoning remain limited, indicating that event-chain organization and causal explanation are key challenges in long-horizon psychological counseling modeling. The related code and data are available at: this https URL 

---
# Hallucination as Commitment Failure: Larger LLMs Misfire Despite Knowing the Answer 

**Authors**: Jewon Yeom, Jaewon Sok, Heejun Kim, Seonghyeon Park, Jeongjae Park, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.22007)  

**Abstract**: Hallucination is often viewed as a direct consequence of missing knowledge: a model answers incorrectly when the correct answer is absent from its generation-time distribution, and correctly when it is present. We test this assumption by introducing a semantic notion of answer availability that aggregates token-level variants expressing the same answer concept, and asks whether the correct concept is already available at the moment the model commits to an answer. Across Qwen and Llama models from 0.8B to 72B in both Instruct and Base variants, 16-47% of Instruct hallucinations occur with substantial probability mass already on the correct concept, and the rate rises monotonically with scale. Comparing such failures against correct generations with matched semantic support, the distinguishing factor is not whether the correct concept is represented, but how its probability is distributed: correct generations concentrate mass on a single surface form, hallucinations disperse it across alternatives. The same sharpening asymmetry extends across multi-token generation and is detectable in pre-generation hidden states. Together, these results identify a single mechanism: instruction tuning sharpens answer commitment with scale, making helpfulness and confident hallucination two consequences of the same underlying disposition. 

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
# A Comparative Study of Language Models for Khmer Retrieval-Augmented Question Answering 

**Authors**: Sereiwathna Ros, Phannet Pov, Ratanaktepi Chhor, Kimleang Ly, Wan-Sup Cho, Saksonita Khoeurn  

**Link**: [PDF](https://arxiv.org/pdf/2605.22099)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for grounding large language model (LLM) outputs in retrieved evidence, thereby reducing hallucination and improving factual accuracy. Its efficacy, however, remains largely unexamined for low-resource, non-Latin-script languages such as Khmer. In this paper, we present a RAG-based question answering system for Khmer-language telecom-domain documents. We conduct a two-phase comparative evaluation. First, we benchmark three embedding models: BGE-M3 (567M), Jina-Embeddings-v3 (570M), and Qwen3-Embedding (597M), for dense retrieval over Khmer documents. BGE-M3 consistently performs best, achieving a Hit Rate@3 of 0.285, File Hit Rate@3 of 0.700, MRR@3 of 0.221, and Precision@3 of 0.112, substantially outperforming the other retrievers. Second, using BGE-M3 as the selected retriever, we evaluate five generator backends: Qwen3 (8B), Qwen3.5 (9B), Sailor2-8B-Chat, SeaLLMs-v3-7B-Chat, and Llama-SEA-LION-v2-8B-IT, on a curated golden dataset of 200 Khmer question-answer pairs. To quantify system performance, we apply six RAGAS-inspired metrics: faithfulness, answer relevance, context relevance, factual correctness, answer similarity, and answer correctness. The results show no single model dominates across all metrics: Qwen3.5-9B achieves the highest faithfulness (0.859) and context relevance (0.726), Qwen3-8B attains the highest factual correctness (0.380), and SeaLLMs-v3-7B-Chat performs best on answer relevance (0.867), answer similarity (0.836), and answer correctness (0.599). These findings highlight that retriever choice remains a major bottleneck for Khmer RAG, while generator strengths vary depending on whether the priority is grounding, factual precision, or semantic similarity. 

---
# FlyRoute: Self-Evolving Agent Profiling via Data Flywheel for Adaptive Task Routing 

**Authors**: Rongjun Li, Ziyu Zhou, Yihang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22057)  

**Abstract**: Enterprise routers assign queries to expert agents, yet deployed profiles stay static while agents evolve (prompts, tools, models), and developers rarely keep descriptions or exemplars current. We present FlyRoute, a self-evolving profiling framework that grows capability evidence from real traffic: dispatch candidates, quality-gate successful pairs into each agent's success store, periodically distill evidence into learned capability descriptions, and inject those descriptions together with BM25-retrieved successes into an LLM router. To make this flywheel data-efficient, FlyRoute introduces a targeted exploration policy that combines profile uncertainty, BM25 relevance, and lexical novelty, prioritizing under-profiled agents only for plausible queries and avoiding redundant evidence collection. In experiments on our proprietary enterprise developer-support dataset of real routed queries, FlyRoute improves a same-backbone zero-shot LLM router from 72.57% to 78.04% with only five seed queries per agent, showing that profile retrieval already strengthens cold-start routing. After streaming 7,211 labeled training queries through the flywheel, accuracy rises to 89.83% (+17.26pp over zero-shot; +11.79pp over cold start), with consistent gains across four expert domains under standard routing accuracy on single-gold test queries. 

---
# SpecHop: Continuous Speculation for Accelerating Multi-Hop Retrieval Agents 

**Authors**: Mehrdad Saberi, Keivan Rezaei, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2605.21965)  

**Abstract**: Large language models increasingly use external tools such as web search and document retrieval to solve information-intensive tasks. However, multi-hop tool use in complex tasks introduces substantial latency, since the model must repeatedly wait for tool observations before continuing. We study how to accelerate such trajectories without changing the final trajectory the model would have taken without acceleration, assuming access to faster but less reliable speculator tools. We develop a theoretical framework for lossless speculation in multi-hop tool-use settings, characterizing the optimal achievable latency gain. We propose SpecHop, a continuous speculation framework that maintains multiple speculative threads, verifies predicted observations asynchronously as target tool outputs arrive, commits correct branches, and rolls back incorrect ones. This preserves accuracy while reducing wall-clock latency. We show that SpecHop can approach oracle latency gains with enough active threads. Empirically, on retrieval-augmented multi-hop tasks, SpecHop closely matches theoretical predictions and reduces latency by up to 40\% in some settings. Code: this https URL 

---
# Claim-Selective Certification for High-Risk Medical Retrieval-Augmented Generation 

**Authors**: Shao Kan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21949)  

**Abstract**: Medical RAG systems in high-risk QA settings are often evaluated through a single answer-or-abstain decision, but mixed evidence may support one claim, require conditions for another, and contradict a third. We study claim-selective certification: each response is decomposed into verifiable claims, scored against retrieved evidence, and mapped by an intent-aware selector to {full, partial, conflict, abstain}. On the primary weak-label certificate protocol, whose real-source-only dev/test rows cover the naturally occurring non-abstain actions, the full system records UCCR=0.0000, PAU=1.0000, PAU Precision=0.9901, and action accuracy=0.9204 on dev (n=314), and UCCR=0.0000, PAU=0.9967, PAU Precision=0.9739, and action accuracy=0.8997 on test (n=319). UCCR measures unsupported-claim risk within the certificate definition, and a source-missing counterfactual slice evaluates abstain under empty evidence. Shortcut controls quantify the action-label prior explained by source and intent metadata, while source/evidence-novel slices characterize transfer boundaries. The resulting interface separates action-label prediction from evidence-linked claim selection under mixed evidence. 

---
# Diagnosis Is Not Prescription: Linguistic Co-Adaptation Explains Patching Hazards in LLM Pipelines 

**Authors**: Yoon Jeonghun, Kim Dongchan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21958)  

**Abstract**: When a multi-module LLM agent fails, the module most responsible for the failure is not necessarily the best place to intervene. We demonstrate this Diagnostic Paradox empirically: causal analysis consistently identifies the routing module -- which selects which tool to call next -- as the primary bottleneck across three independent agent families. Yet injecting prompt-level correction examples into this module consistently degrades performance, sometimes severely. Patching an upstream query-rewriting module instead reliably improves outcomes. The effect holds with statistical significance on two agent families and directional consistency on a third; alternative repair strategies at the routing module (instruction rewriting, model upgrade) are neutral, confirming that the harm is specific to correction-injection patching.
We explain this asymmetry through the Linguistic Contract hypothesis: each downstream module implicitly adapts to its upstream's characteristic error distribution, so correcting the bottleneck breaks this implicit alignment in a way that upstream corrections do not. We operationalize this via a per-agent co-adaptation measure, derived from diagnosis alone, and show it is consistently associated with patching harm across agent families: higher co-adaptation co-occurs with harm, lower with safety. This trend holds across all three agent families, providing preliminary support for the hypothesis beyond a single-agent observation. 

---
# ACC: Compiling Agent Trajectories for Long-Context Training 

**Authors**: Qisheng Su, Zhen Fang, Shiting Huang, Yu Zeng, Yiming Zhao, Kou Shi, Ziao Zhang, Lin Chen, Zehui Chen, Lijun Wu, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.21850)  

**Abstract**: Recent development of agents has renewed demand for long-context reasoning capacity of LLMs. However, training LLMs for this capacity requires costly long-document curation or heuristic context synthesis. We observe that agents produce massive trajectories when solving problems, invoking tools and receiving environment observations across many turns. The evidence needed to answer the original question is thus scattered throughout these turns, requiring integration of distant context segments. Nevertheless, standard agent SFT masks tool responses and only trains turn-level tool selection, creating a supervision blind spot where these scattered signals go unused. We propose Agent Context Compilation (ACC), which converts trajectories from search, software engineering, and database querying agents into long-context QA pairs that combine the original question with tool responses and environment observations gathered across multiple turns, training the model to answer directly without tool use. This makes the dependencies between the question and the evidence explicit, enabling direct supervision of long-context reasoning over distant segments without additional annotation. ACC is a simple but effective approach that can be combined with any existing long-context extension or training method, providing scalable supervised fine-tuning data. We validate ACC on long-range dependency modeling tasks through MRCR and GraphWalks, challenging benchmarks requiring cross-turn coreference resolution and graph traversal over extended contexts. Training Qwen3-30B-A3B with ACC achieves 68.3 on MRCR (+18.1) and 77.5 on GraphWalks (+7.6), results comparable to Qwen3-235B-A22B, while preserving general capabilities on GPQA, MMLU-Pro, AIME, and IFEval. Further mechanism analysis reveals that the ACC-trained model exhibits task-adaptive attention restructuring and expert specialization. 

---
# LatentOmni: Rethinking Omni-Modal Understanding via Unified Audio-Visual Latent Reasoning 

**Authors**: Yifan Dai, Zhenhua Wu, Bohan Zeng, Daili Hua, Jialing Liu, Bozhou Li, Yuran Wang, Chengzhuo Tong, Hao Liang, Xiaochen Ma, Junbo Niu, Tianyu Guo, Yang Shi, Yue Ding, Yiyan Ji, Bingyin Mei, Yushuo Guan, Yuanxing Zhang, Pengfei Wan, Fangcheng Fu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22012)  

**Abstract**: Joint audio-visual reasoning is essential for omnimodal understanding, yet current multimodal large language models (MLLMs) still struggle when reasoning requires fine-grained evidence from both modalities. A central limitation is that explicit text-based chain-of-thought (CoT) compresses continuous audio-visual signals into discrete tokens, weakening temporal grounding and shifting intermediate reasoning toward language priors. We argue that a unified latent space is a better medium for such reasoning because it preserves dense sensory information while remaining compatible with autoregressive generation. Based on this insight, we propose \textbf{LatentOmni}, a cross-modal reasoning framework that interleaves textual reasoning with audio-visual latent states. LatentOmni introduces feature-level supervision to align latent reasoning states with task-relevant sensory features and uses Omni-Sync Position Embedding (OSPE) to maintain temporal consistency between latent audio and visual states. We further construct \textbf{LatentOmni-Instruct-35K}, a dataset of audio-visual interleaved reasoning trajectories for supervising latent-space reasoning. Comprehensive evaluation across multiple audio-visual reasoning benchmarks demonstrates that LatentOmni achieves the best performance among the evaluated open-source models and consistently outperforms the Explicit Text CoT baseline, supporting latent-space joint reasoning as a promising path toward stronger omnimodal understanding. 

---
# Hypergraph as Language 

**Authors**: Mengqi Lei, Guohuan Xie, Shihui Ying, Shaoyi Du, Jun-Hai Yong, Siqi Li, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.21858)  

**Abstract**: Large language models (LLMs) have recently shown strong potential in modeling relational structures. However, existing approaches remain fundamentally graph-centric: they focus on processing pairwise graph structures into tokens that LLMs can understand. In contrast, many real-world relational patterns do not naturally conform to the pairwise-edge assumption, and are better modeled as high-order associations in hypergraphs. For hypergraph structures, existing methods often fail to preserve the native semantics that multiple objects are jointly connected by the same high-order relation, limiting their ability to exploit complex structures. To address this limitation, we put forth the "Hypergraph as Language" perspective and propose Hyper-Align, a hypergraph-native alignment framework for large language models. Hyper-Align compiles the query-object-centered hypergraph context into hypergraph tokens directly consumable by a base LLM. Specifically, we introduce Hypergraph Incidence Detail Template with Overview (HIDT-O), which serializes high-order association structures into a fixed-shape hybrid template combining local incidence details and overview-level summaries. We then design a Hypergraph Incidence Projector (HIP), which maps native high-order incidence structures into the LLM token space through explicit semantic-structural decoupling and bidirectional message passing between vertices and hyperedges. We further define a concrete Hypergraph-as-Language input protocol, which jointly feeds hypergraph tokens and textual prompts into a frozen base LLM, supporting both vertex-level and hyperedge-level tasks under a unified question-answering paradigm. To systematically evaluate different methods in hypergraph structural modeling, we introduce HyperAlign-Bench. Extensive experiments show that Hyper-Align significantly outperforms existing methods across in-domain and zero-shot evaluations. 

---
# From TF-IDF to Transformers: A Comparative and Ensemble Approach to Sentiment Classification 

**Authors**: Dip Biswas Shanto, Mitali Yadav, Prajwal Panth, Suresh Chandra Satapathy  

**Link**: [PDF](https://arxiv.org/pdf/2605.22003)  

**Abstract**: Sentiment analysis, also referred to as opinion mining, primarily tries to extract opinion from any text-based data. In the context of movie reviews and critics, sentimental analysis can be a helpful tool to predict whether a movie review is generally positive or negative. It can be difficult for the ML models to understand the context or metaphysical sentiment accurately, as ML models rely largely on statistical word representations. The objective of this paper is to examine and categorise movie reviews into positive and negative sentiments. Diverse machine learning models are considered in doing so, and Natural Language Processing (NLP) methodologies are employed for data preprocessing and model assessment. The IMDb dataset is used. Specifically, Naive Bayes, Logistic Regression, Support Vector Machines (SVM), LightGBM, LSTM, and transformer-based models such as RoBERTa and DistilBERT were evaluated. After a lot of testing with accuracy, precision, recall, F1-score, and ROC-AUC, RoBERTa performed better than all the other models, with an accuracy of 93.02%. A soft voting ensemble that combined all the models also improved classification performance, showing that model ensembling works well for sentiment analysis. 

---
# Token-weighted Direct Preference Optimization with Attention 

**Authors**: Chengyu Huang, Zhuohang Li, Sheng-Yen Chou, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2605.21883)  

**Abstract**: Direct Preference Optimization (DPO) aligns Large Language Models with human preferences without the need for a separate reward model. However, DPO treats all tokens in responses equally, neglecting the differing importance of individual tokens. Existing token-level PO methods compute the token weights using either token-position-based heuristic functions or probability estimates given by a separately trained model, which lacks robustness and incurs extra training cost. In contrast, we propose Token-weighted DPO (TwDPO) -- a novel training objective grounded on token-weighted RL -- and AttentionPO -- an instantiation of TwDPO that uses attention from the LLM itself to estimate token weights. AttentionPO prompts the LLM to serve as a pairwise judge and check where the model attends when comparing the responses. This design makes AttentionPO content-aware, adjusting weights based on response content, and efficient, incurring only two extra forward passes per example. Experiment results show that AttentionPO significantly improves performance on AlpacaEval, MT-Bench, and ArenaHard, surpassing existing Preference Optimization methods. 

---
# Does Slightly Mean Somewhat? Measuring Vague Intensity Words in LLM Numeric Actions 

**Authors**: Daniel Tabach  

**Link**: [PDF](https://arxiv.org/pdf/2605.21827)  

**Abstract**: Do language models preserve the ordinal meaning of intensity words when those words must produce numeric actions? I study a researcher-constructed scale of 10 English degree modifiers, from slightly to drastically, informed by the Quirk et al. degree-modifier taxonomy, in a controlled resource-allocation environment where Claude Haiku receives a natural-language instruction, produces a numeric allocation, and a deterministic backend converts that allocation into a measurable outcome. The only variable that changes between runs is the intensity word or the starting system state, isolating their effects on the model's numeric output.
Across 6,620 runs at T=0.0 and T=0.7, three patterns emerge. First, the model compresses 10 intensity words into 5 distinct median outputs: four lower-tier words all map to the same value, while stronger words break into higher regimes (Spearman rho = 0.845, p < 0.001). Second, when the current system state is supplied as context, separate Kruskal-Wallis tests show that grouping by starting allocation captures far more rank-based variance than grouping by word (epsilon-squared baseline = 0.782 vs. epsilon-squared word = 0.079), and lexical differentiation collapses to zero as the system approaches capacity. Third, near feasibility limits the model exhibits three behavioral modes: weak words hedge with small adjustments, strong words abstain entirely, and the word drastically pushes to the local ceiling. These patterns persist across temperature, with stochastic sampling broadening distributions but not restoring ordinal distinctions between words. In this model and domain, the model's numeric interpretation of vague intensity words is compressed, state-dependent, and discontinuous near operational boundaries. 

---
# Comparing LLM and Fine-Tuned Model Performance on NVDRS Circumstance Extraction with Varying Prompt Complexity 

**Authors**: Geoffrey Martin, Xuan Zhong Feng, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.21845)  

**Abstract**: Suicide is a leading cause of death in the United States, and understanding the circumstances that precede it requires extracting structured information from death investigation narratives. Many of these circumstances require semantic inference beyond simple keyword matching. We develop a ``Complexity Score'' algorithm that analyzes coding manual structure to predict when detailed prompts with full coding guidelines improve over name-only prompts. We then construct a hybrid approach that selects prompt strategy per circumstance. We evaluate large language models (LLMs) against fine-tuned RoBERTa on 25 inferentially complex circumstances from the National Violent Death Reporting System (NVDRS). We found that LLMs substantially outperform on low-prevalence circumstances where training data is insufficient. We further demonstrate that our framework generalizes across frontier LLMs, with GPT-5.2, Gemini 2.5 Pro and Llama-3 70B showing consistent performance patterns. These findings support a hybrid architecture where LLMs handle rare, inferentially complex circumstances while fine-tuned models handle common ones. 

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
# RankJudge: A Multi-Turn LLM-as-a-Judge Synthetic Benchmark Generator 

**Authors**: Zhenwei Tang, Zhaoyan Liu, Rasa Hosseinzadeh, Tongzi Wu, Keyvan Golestan, Jesse C. Cresswell  

**Link**: [PDF](https://arxiv.org/pdf/2605.21748)  

**Abstract**: As interactive LLM-based applications are created and refined, model developers need to evaluate the quality of generated text along many possible axes. For simpler systems, human evaluation may be practical, but in complicated systems like conversational chatbots, the amount of generated text can overwhelm human annotation resources. Model developers have begun to rely heavily on auto-evaluation, where LLMs are also used to judge generation quality. However, existing LLM-as-a-judge benchmarks largely focus on simple Q\&A tasks that do not match the complexity of multi-turn conversations. We introduce RankJudge, a benchmark generator for evaluating LLM-as-a-judge on multi-turn conversations grounded in reference documents. RankJudge creates pairs of conversations where one conversation has a single flaw injected into one turn. This construction allows paired conversations to be labeled unambiguously as better or worse, and precisely isolates failure categories to individual turns, enabling a strict joint correctness criterion for judging. We implement RankJudge across the domains of machine learning, biomedicine, and finance, evaluate 21 frontier LLM judges, and rank those judges via the Bradley-Terry model. Our formulation also allows ranking each conversation pair with difficulty ratings, which we use to dynamically curate the evaluation slice to reduce label noise, as confirmed via human annotation. We find that judge rankings are stable under partial observability, coarser correctness criteria, and an alternative random-walk rating algorithm. 

---
# Sem-Detect: Semantic Level Detection of AI Generated Peer-Reviews 

**Authors**: André V. Duarte, Brian Tufts, Aditya Oke, Fei Fang, Arlindo L. Oliveira, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.21713)  

**Abstract**: How can we distinguish whether a peer review was written by a human or generated by an AI model? We argue that, in this setting, authorship should not be attributed solely from the textual features of a review, but also from the ideas, judgments, and claims it expresses. To this end, we propose Sem-Detect, an authorship detection method for peer reviews that operationalizes this principle by combining textual features with claim-level semantic analysis. Sem-Detect compares a target review against multiple AI-generated reviews of the same paper, leveraging the observation that different AI models tend to converge on similar points, while human reviewers introduce more unique and diverse ones. As a result, Sem-Detect is able to distinguish fully AI reviews from authentic human-written ones, including those that have been refined using an LLM but still reflect human judgment. Across a dataset of over 20,000 peer reviews from ICLR and NeurIPS conferences, Sem-Detect improves over the strongest baseline by 25.5% in TPR@0.1% FPR in the binary setting. Moreover, in the three-class scenario, we empirically show that LLM refinement preserves the semantic signals of human reviews, which remain distinct from the patterns exhibited by fully AI-generated text; as a result, fewer than 3.5% of LLM-refined human reviews are misclassified as AI-generated. 

---
# When Cases Get Rare: A Retrieval Benchmark for Off-Guideline Clinical Question Answering 

**Authors**: Doeun Lee, Muge Zhang, Yi Yu, Ashish Manne, Stephen Koesters, Frank Wen, Brady Buchanan, Lynda Villagomez, Oluwatoba Moninuola, James Lim, Kathryn Tobin, Andrew Srisuwananukorn, Ping Zhang, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.21807)  

**Abstract**: Across medical specialties, clinical practice is anchored in evidence-based guidelines that codify best studied diagnostic and treatment pathways. These pathways routinely fall short for the long tail of real-world care not covered by guidelines. Most medical large language models (LLMs), however, are trained to encode common, guideline-focused medical knowledge in their parameters. Current evaluations test models primarily on recalling and reasoning with this memorized content, often in multiple-choice settings. Given the fundamental importance of evidence-based reasoning in medicine, it is neither feasible nor reliable to depend on memorization in practice. To address this gap, we introduce OGCaReBench, a free-form retrieval-focused benchmark aimed at evaluating LLMs at answering clinical questions that require going beyond typical guidelines. Extracted from published medical case reports and validated by medical experts, OGCaReBench contains long-form clinical questions requiring free-text answers, providing a systematic framework for assessing open-ended medical reasoning in rare, case-based scenarios. Our experiments reveal that even the best-performing baseline (GPT-5.2) correctly answers only 56% of our benchmark with specialized models only reaching 42%. Augmenting models with retrieved medical articles improves this performance to up to 82% (using GPT-5.2) highlighting the importance of evidence-grounding for real-world medical reasoning tasks. This work thus establishes a foundation for benchmarking and advancing both general-purpose and medical LLMs to produce reliable answers in challenging clinical contexts. 

---
# Hy-MT2: A Family of Fast, Efficient and Powerful Multilingual Translation Models in the Wild 

**Authors**: Mao Zheng, Zheng Li, Tao Chen, Bo Lv, Mingrui Sun, Mingyang Song, Jinlong Song, Hong Huang, Decheng Wu, Hai Wang, Yifan Song, Yanfeng Chen, Guanwei Zhang, Guanghua Yu, Yi Su, Hong Liu, Jinxiang Ou, Keyao Wang, Weile Chen, Haozhao Kuang, Kai Wang, Nuo Chen, Zihao Zheng, Chenhao Wang, Bin Xing, Chengcheng Xu, Tinghao Yu, Binghong Wu, Long Xu, Jiacheng Shi, Yunhao Wang, Baifang Chen, Lei Zhang, Qi Yang, Zhao Wu, Jiacheng Li, Lan Jiang, Lanrui Wang, Kai Zhang, Shuaipeng Li, Zhongzhi Chen, Weixuan Sun, Jiaqi Zhu, An Wang, Wei Li, Jun Xia, Weidong Han, Wutian Yang, Litong Hui, Luoguo Jia, Jiajia Wu, Xinpeng Zhou, Tianxiang Fei  

**Link**: [PDF](https://arxiv.org/pdf/2605.22064)  

**Abstract**: Hy-MT2 is a family of fast-thinking multilingual translation models designed for complex real-world scenarios. It includes three model sizes: 1.8B, 7B, and 30B-A3B (MoE), all of which support translation among 33 languages and effectively follow translation instructions in multiple languages. For on-device deployment, with AngelSlim 1.25-bit extreme quantization, the 1.8B model requires only 440 MB of storage and improves inference speed by 1.5x. Multi-dimensional evaluations show that Hy-MT2 delivers outstanding performance across general, real-world business, domain-specific, and instruction-following translation tasks. The 7B and 30B models outperform open-source models such as DeepSeek-V4-Pro and Kimi K2.6 in fast-thinking mode, while the lightweight 1.8B model also surpasses mainstream commercial APIs from providers such as Microsoft and Doubao overall. 

---
# Beyond Acoustic Emotion Recognition: Multimodal Pathos Analysis in Political Speech Using LLM-Based and Acoustic Emotion Models 

**Authors**: Juergen Dietrich  

**Link**: [PDF](https://arxiv.org/pdf/2605.22732)  

**Abstract**: We investigate whether acoustic emotion recognition models can serve as proxies for the Pathos dimension in political speech analysis, as operationalised by the TRUST multi-agent large language model (LLM) pipeline. Using a Bundestag plenary speech by Felix Banaszak (51 segments, 245 s) as a case study, we compare three analysis modalities: (1) emotion2vec_plus_large, an acoustic speech emotion recognition (SER) model whose continuous Arousal and Valence values are derived via post-hoc Russell Circumplex projection; (2) Gemini 2.5 Flash, an LLM analysing the full speech audio together with its transcript in an open-ended, context-aware fashion; and (3) TRUST-Pathos scores from a three-advocate LLM supervisor ensemble. Spearman rank correlations reveal that Gemini Valence correlates strongly with TRUST-Pathos (rho = +0.664, p < 0.001), whereas emotion2vec Valence does not (rho = +0.097, p = 0.499). We further demonstrate, via a systematic quality evaluation of the Berlin Database of Emotional Speech (EMO-DB) using Gemini in an open-ended annotation paradigm, that standard SER benchmark corpora suffer from acted speech, cultural bias, and category incompatibility. Our results suggest that LLM-based multimodal analysis captures semantically defined political emotion substantially better than acoustic models alone, while acoustic features remain informative for low-level Arousal estimation. Future work will extend this approach to video-based analysis incorporating facial expression and gaze. 

---
# Probabilistic Attribution For Large Language Models 

**Authors**: Shilpika Shilpika, Carlo Graziani, Bethany Lusch, Venkatram Vishwanath, Michael E. Papka  

**Link**: [PDF](https://arxiv.org/pdf/2605.21726)  

**Abstract**: The generative nature of Large Language Models (LLMs) is reflected in the conditional probabilities they compute to sample each response token given the previous tokens. These probabilities encode the distributional structure that the model learns in training and exploits in inference. In this work, we use these probabilities to situate LLMs within the mathematical theory of stochastic processes. We use this framework to design a model-agnostic probabilistic token attribution measure, using Bayes rule to invert the next-token log-probabilities so as to capture the models internal representation of the distribution over token sequences. The representation is independent of the models computational structure. This representation yields the conditional probability of the response given the prompt, and of the response given the prompt with a token marginalized away. Our attribution score is the log of the ratio of these probabilities. We further compute the entropies of a single prompts token distributions, conditioned on the remaining context. The interplay between entropy and attribution score sheds light on LLM behavior. We evaluate 8 models across 7 prompts and investigate anomalies, token sensitivity, response stability, model stability, and training convergence, thereby improving interpretability and guiding users to focus on uncertain or unstable parts of the generation. 

---
# CR4T: Rewrite-Based Guardrails for Adolescent LLM Safety 

**Authors**: Heajun An, Qi Zhang, Vedanth Achanta, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2605.21609)  

**Abstract**: Large language models (LLMs) are increasingly embedded in adolescent digital environments, mediating information seeking, advice, and emotionally sensitive interactions. Yet existing safety mechanisms remain largely grounded in adult-centric norms and operationalize safety through refusal-oriented suppression. While such approaches may reduce immediate policy violations, they can also create conversational dead-ends, limit constructive guidance, and fail to address the developmental vulnerabilities inherent in adolescent-AI interactions. We argue that adolescent LLM safety should be framed not solely as a filtering problem, but as a socio-technical, developmentally aligned transformation problem. To operationalize this perspective, we propose Critique-and-Revise-for-Teenagers (CR4T), a model-agnostic safeguarding framework that selectively reconstructs unsafe or refusal-style outputs into ageappropriate, guidance-oriented responses while preserving benign intent. CR4T combines lightweight risk detection with domain-conditioned rewriting to remove risk-amplifying content, reduce unnecessary conversational shutdown, and introduce developmentally appropriate guidance. Experimental results show that targeted rewriting substantially reduces unsafe and refusal-oriented outcomes while avoiding unnecessary intervention on acceptable interactions. These findings suggest that selective response reconstruction offers a more human-centered alternative to refusal-centric guardrails for adolescent-facing LLM systems. 

---
# PromptNCE: Pointwise Mutual Information Predictions Using Only LLMs and Contrastive Estimation Prompts 

**Authors**: Juliette Woodrow, Chris Piech  

**Link**: [PDF](https://arxiv.org/pdf/2605.21776)  

**Abstract**: Estimating mutual information from text usually requires training a task-specific critic, which limits its use in low-data settings. We ask whether large language models can instead estimate pointwise mutual information zero-shot, using only prompts and elicited probabilities. We introduce a benchmark with human-derived ground-truth PMI across three publicly available datasets, and evaluate five information-theoretic prompting-based estimators. Our main method, PromptNCE, frames conditional probability estimation as a contrastive task and augments the candidate set with an explicit OTHER category. We show theoretically that adding OTHER recovers the true conditional P(y | x) rather than just a ranking over listed candidates, turning a contrastive prompt into a general-purpose zero-shot probability estimator. PromptNCE is the best zero-shot method on all three datasets, reaching Spearman correlation up to 0.82 with human-derived PMI. We also present a case study in computer science education showing how these estimators can be used to score student knowledge summaries in a low-data setting. 

---
# AMEL: Accumulated Message Effects on LLM Judgments 

**Authors**: Sid-ali Temkit  

**Link**: [PDF](https://arxiv.org/pdf/2605.22714)  

**Abstract**: Large language models are routinely used as automated evaluators: to review code, moderate content, or score outputs, often with many items passing through one conversation. We ask whether the polarity of prior conversation history biases subsequent judgments, an effect we call the accumulated message effect on LLM judgments (AMEL). Across 75,898 API calls to 11 models from 4 providers (OpenAI, Anthropic, Google, and four open-source models), we present identical test items in isolation or following histories saturated with predominantly positive or negative evaluations.
Models shift toward the conversation's prevailing polarity (d = -0.17, p < 10^-46). The effect concentrates on items where the model is genuinely uncertain at baseline (d = -0.34 for high-entropy items, vs d = -0.15 when the baseline is deterministic). Bias does not grow with context length: 5 prior turns and 50 produce the same shift (Spearman |r| < 0.01; OLS slope p = 0.80). And there is a negativity asymmetry: paired per item, negative histories induce 1.62x more bias than positive (t = 13.46, p < 10^-39, n = 2,481). Scaling helps but does not solve it (Anthropic: Haiku -0.22 to Opus -0.17; OpenAI: Nano -0.34 to GPT-5.2 -0.17).
Three follow-ups narrow the mechanism. The token probability distribution shifts continuously, not at a threshold. The negativity asymmetry has both token-level and semantic components, though attributing the balance is exploratory at our sample sizes. Position does not matter: five biased turns anywhere in a 50-turn history produce the same shift. The simplest fix for evaluation pipelines is a fresh context per item; when batching is unavoidable, balancing the history helps. 

---
# Broadening Access to Transportation Safety Data with Generative AI: A Schema-Grounded Framework for Spatial Natural Language Queries 

**Authors**: Mahdi Azhdari, Eric J. Gonzales  

**Link**: [PDF](https://arxiv.org/pdf/2605.21712)  

**Abstract**: Transportation safety analysis requires integrating crash records, roadway attributes, and geospatial data through GIS-based workflows, but access remains uneven across agencies and community stakeholders. Technical prerequisites create a gap between analytical tools central to safety planning and the practitioners able to use them. Local agencies, school committees, and residents may have safety concerns but limited capacity to retrieve, filter, map, and analyze relevant data. Generative AI offers a way to narrow this divide, but its public-sector use raises questions about reliability, reproducibility, and governance. This paper presents a schema-grounded natural language interface for transportation safety analysis, using a large language model (LLM) to interpret user intent while preserving deterministic, reviewable execution against an authoritative database. User queries are translated into structured semantic frames, validated by a rule-based layer, compiled into a typed directed acyclic graph of spatial operations, and executed against a PostGIS database. This bounded design separates language interpretation from deterministic execution, keeping results reproducible and schema-grounded while removing access barriers. The framework is evaluated using a statewide Massachusetts transportation safety database integrating crash records, roadway attributes, and geospatial layers including schools, bus stops, crosswalks, and municipal boundaries. All queries executed successfully; the validation layer corrects errors in 29% of evaluation queries, reflecting the gap between flexible natural language and strict schema-grounded requirements. The results suggest that combining natural language accessibility with deterministic execution is a practical direction for broadening access to transportation safety data, with implications for trustworthy AI in public-sector planning. 

---
# Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework 

**Authors**: Shourov Joarder, Diganta Sikdar, Ahsan Habib Akash, Binod Bhattarai, Prashnna Gyawali  

**Link**: [PDF](https://arxiv.org/pdf/2605.22620)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has substantially improved the reasoning ability of LLMs, but often depends on external supervision from human annotations or gold-standard solutions. Reinforcement learning from internal feedback (RLIF) has recently emerged as a scalable unsupervised alternative, using signals extracted from the model itself. However, existing RLIF methods typically rely on a single internal reward, which can lead to reward hacking, entropy collapse, and degraded reasoning structure. We propose a multi-reward RLIF framework that decomposes the training signal into two complementary components: an answer-level reward based on cluster voting and a completion-level reward based on token-wise self-certainty. To combine these signals robustly, we apply GDPO-based normalization to reduce reward-scale imbalance. We further introduce KL-Cov regularization, which targets low-entropy token distributions responsible for disproportionate entropy reduction, preserving exploration and preventing late-stage collapse. Across mathematical reasoning and code-generation benchmarks, our method improves stability and robustness over prior unsupervised RL approaches, while achieving performance close to supervised RLVR methods. These results show that complementary internal rewards, combined with targeted regularization, can support stable long-horizon reasoning without relying on external ground-truth supervision. Code will be released soon. 

---
# SpaceDG: Benchmarking Spatial Intelligence under Visual Degradation 

**Authors**: Xiaolong Zhou, Yifei Liu, Ziyang Gong, Jiarui Li, Qiyue Zhao, Muyao Niu, Yuanyuan Gao, Le Ma, Xue Yang, Hongjie Zhang, Zhihang Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2605.22536)  

**Abstract**: Multimodal Large Language Models (MLLMs) have made rapid progress in spatial intelligence, yet existing spatial reasoning benchmarks largely assume pristine visual inputs and overlook the degradations that commonly occur in real-world deployment, such as motion blur, low light, adverse weather, lens distortion, and compression artifacts. This raises a fundamental question: how robust is the spatial intelligence of current MLLMs when visual observations are imperfect? To answer this question, we introduce SpaceDG, the first large-scale dataset for degradation-aware spatial understanding. It is constructed with a physically grounded degradation synthesis engine that embeds degradation formation process into 3D Gaussian Splatting (3DGS) rendering, enabling realistic simulation of nine degradation types. The resulting dataset contains approximately 1M QA pairs from nearly 1,000 indoor scenes. We further introduce SpaceDG-Bench, an human-verified benchmark with 1,102 questions spanning 11 reasoning categories and 9 visual degradation types, yielding over 10K VQA instances. Evaluating 25 open- and closed-source MLLMs reveals that visual degradations consistently and substantially impair spatial reasoning, exposing a critical robustness gap. Finally, we show that finetuning on SpaceDG markedly improves degradation robustness and can even surpass human performance under degraded conditions without any performance drop on clean images, highlighting the promise of degradation-aware training for robust spatial intelligence. 

---
# Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents 

**Authors**: Xing Zhang, Yanwei Cui, Guanghui Wang, Ziyuan Li, Wei Qiu, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2605.22148)  

**Abstract**: Self-evolving skill libraries, pioneered by Voyager, let frozen LLM agents accumulate reusable knowledge without weight updates, yet recent evaluation shows that LLM-authored skills deliver $+0.0$pp over no-skill baselines while human-curated ones deliver $+16.2$pp: the bottleneck is not skill authoring but lifecycle management. We introduce \textbf{Ratchet}, a single-agent loop in which a frozen LLM writes, retrieves, curates, and retires its own natural-language skills. Ratchet integrates four candidate hygiene mechanisms: outcome-driven retirement, a bounded active-cap, meta-skill authoring guidance, and pattern canonicalisation. On MBPP+ hard-100 with Claude Opus 4.7, Ratchet lifts held-out pass@1 from a $0.258 \pm 0.047$ baseline to a late-window rolling mean of $0.584$ (peak $0.658 \pm 0.042$) across 100 rounds and 3 seeds, a $+0.328 \pm 0.018$ rolling-mean gain where the no-skill control drifts at $+0.002 \pm 0.005$; the same recipe transfers to an agentic solver on SWE-bench Verified ($+0.22$ peak lift over 20 rounds). Eight ablations (A1--A8) reveal that the minimal working recipe is smaller than our design suggests: retirement and the meta-skill authoring prior are load-bearing, while explicit deduplication (canonicalisation, cover-guard) is subsumed by the meta-skill itself. A non-divergence proposition shows that bounded cap and retirement threshold together prevent expected performance from drifting below the no-skills floor. 

---
# Vector Policy Optimization: Training for Diversity Improves Test-Time Search 

**Authors**: Ryan Bahlous-Boldi, Isha Puri, Idan Shenfeld, Akarsh Kumar, Mehul Damani, Sebastian Risi, Omar Khattab, Zhang-Wei Hong, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2605.22817)  

**Abstract**: Language models must now generalize out of the box to novel environments and work inside inference-scaling search procedures, such as AlphaEvolve, that select rollouts with a variety of task-specific reward functions. Unfortunately, the standard paradigm of LLM post-training optimizes a pre-specified scalar reward, often leading current LLMs to produce low-entropy response distributions and thus to struggle at displaying the diversity that inference-time search will require. We propose Vector Policy Optimization (VPO), an RL algorithm that explicitly trains policies to anticipate diverse downstream reward functions and to produce diverse solutions. VPO exploits that rewards are often vector-valued in practice, like per-test-case correctness in code generation or, say, multiple different user personas or reward models. VPO is essentially a drop-in replacement for the GRPO advantage estimator, but it trains the LLM to output a set of solutions where individual solutions specialize to different trade-offs in the vector reward space. Across four tasks, VPO matches or beats the strongest scalar RL baselines on test-time search (e.g. pass@k and best@k), with the gap widening as the search budget grows. For evolutionary search, VPO models unlock problems that GRPO models cannot solve at all. As test-time search becomes more standardized, optimizing for diversity may need to become the default post-training objective. 

---
# Boundary-targeted Membership Inference Attacks on Safety Classifiers 

**Authors**: Anthony Hughes, Alexander Goldberg, Prince Jha, Adam Perer, Nikolaos Aletras, Niloofar Mireshghallah  

**Link**: [PDF](https://arxiv.org/pdf/2605.22373)  

**Abstract**: Safety classifiers are essential safeguards within generative AI systems, filtering harmful content or identifying at-risk users when interacting with large language models. Despite their necessity, these models are trained on sensitive datasets including discussions of self-harm and mental health, raising important, yet poorly understood, privacy concerns. Membership inference attacks (MIAs) allow adversaries to infer membership of examples used to train models. In this work, we hypothesize that identifying the examples on which the classifier is least confident are informative for an adversary to infer membership. This reflects a localized failure of generalization, where the model relies on memorization to resolve ambiguity in the training set. To investigate this, we introduce a new boundary-targeted selection strategy that identifies low confidence examples that amplify the signal of an examples membership within a training set. Our experimental results show that an adversary can recover 19\% of the conversations a safety classifier flagged as indicating user distress, at a 5\% false-positive rate, on a classifier fine-tuned for detecting a user who may require emotional support. This is $3.5$ times more than attacking using state-of-the-art MIA methods alone. Finally, we characterize the boundary laying examples and show that content-based filtering is ineffective for protection, and existing noise strategies can effectively mitigate susceptibility of these examples. 

---
# AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild 

**Authors**: Baiyu Chen, Zechen Li, Wilson Wongso, Lihuan Li, Xiachong Lin, Hao Xue, Benjamin Tag, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2605.22715)  

**Abstract**: As wearable and mobile devices become increasingly embedded in daily life, they offer a practical way to continuously sense human motion in the wild. But inertial signals are highly dependent on the sensing setup, including body location, mounting position, sensor orientation, device hardware, and sampling protocol. This setup dependence makes it difficult to learn motion representations that transfer across devices and datasets, and limits the broader use of wearable IMUs beyond closed-set recognition. We introduce AnyMo, a geometry-aware framework for setup-agnostic human motion modeling. AnyMo uses physics-grounded IMU simulation over dense body-surface placements to generate diverse and plausible synthetic signals, pre-trains a graph encoder from paired synthetic placement views and masked partial observations, tokenizes multi-position IMU into full-body motion tokens, and aligns these tokens with an LLM for motion-language understanding. We evaluate AnyMo on three complementary tasks: zero-shot activity recognition across 14 unseen downstream datasets, cross-modal retrieval, and wearable IMU motion captioning, where it improves average Accuracy/F1/R@2 by 11.7\%/11.6\%/22.6\% on HAR, increases zero-shot IMU-to-text and text-to-IMU retrieval MRR by 15.9\% and 28.6\%, respectively, and improves zero-shot captioning BERT-F1 by 18.8\%. These results support AnyMo as a generalist model for wearable motion understanding in the wild. Project page: this https URL. 

---
# Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning 

**Authors**: Zihan Liang, Yufei Ma, Ben Chen, Zhipeng Qian, Xuxin Zhang, Huangyu Dai, Lingtao Mao  

**Link**: [PDF](https://arxiv.org/pdf/2605.22511)  

**Abstract**: Post-training has become the dominant recipe for turning a language model into a competent search-augmented reasoning agent. A line of recent work pushes its performance further by adding elaborate machinery on top of this standard pipeline. These augmentations import external supervision from stronger external systems, attach auxiliary modules such as process reward models or retrospective critics, restructure the rollout itself with tree search or multi-stage curricula, or shape the reward with hand-crafted bonuses and penalties. Each addition delivers a measurable gain, but each also inflates the training pipeline and ties the recipe to resources or designs that may not always be available. We take a step back and ask whether any of this machinery is actually necessary, and propose Search-E1, a self-evolution method that lets a search-augmented agent improve through only vanilla GRPO interleaved with offline self-distillation (OFSD). After each GRPO round, the policy rolls out on its own training questions. A token-level forward KL objective then aligns the policy's inference-time distribution to its own distribution under a privileged context that exposes a more efficient sibling trajectory. Despite this simplicity, the procedure naturally provides dense per-step supervision. On seven QA benchmarks, Search-E1 reaches $0.440$ average EM with Qwen2.5-3B, surpassing all open-source baselines at both scales. Code and complete version will be made public soon. 

---
# Check Your LLM's Secret Dictionary! Five Lines of Code Reveal What Your LLM Learned (Including What It Shouldn't Have) 

**Authors**: Hisashi Miyashita  

**Link**: [PDF](https://arxiv.org/pdf/2605.22005)  

**Abstract**: We show that singular value decomposition of the lm_head} weight matrix of a transformer-based large language model -- requiring only five lines of PyTorch and no model inference -- reveals interpretable semantic subspaces directly from the model weights. Each left singular vector identifies the vocabulary tokens most readily selected when the hidden state aligns with the corresponding singular direction; inspecting these clusters exposes the model's training data composition and curation philosophy.
Analysing GPT-OSS-120B, Gemma-2-2B, and Qwen2.5-1.5B, we find that singular value spectra and vocabulary cluster structures differ systematically across models: GPT exhibits a graduated hierarchy of functionally differentiated subspaces; Gemma is dominated by pre-nineteenth-century English orthography, forming a stepwise clustering structure that may contribute to high output controllability; and Qwen exhibits broad multilingual coverage alongside subspaces whose vocabulary the authors have determined to be ethically inappropriate for direct publication.
Base-instruct comparison reveals that ethically concerning subspaces originate in pretraining and are not removed by post-training alignment. We introduce the Vocabulary Cluster Score (VCS) to quantify subspace coherence, and the Weighted Projection Score (WPS) as a static glitch token detector; applying WPS to GPT-OSS-120B recovers shokubutsu-hyakka-tsu (ID 137606), a well-known glitch token widely reported in the CJK language community, without any model inference. We propose a taxonomy of root causes for problematic vocabulary content and call for lm_head} SVD analysis to be adopted as a standard pre-release safety auditing step. Our findings further suggest directions toward SVD-guided tokenizer optimisation and more controllable LLM design. 

---
# Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL 

**Authors**: Sophia Xiao Pu, Zhaotian Weng, Chengzhi Liu, Jayanth Srinivasa, Gaowen Liu, William Yang Wang, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22217)  

**Abstract**: Self-play reinforcement learning trains language models on their own generated tasks, co-evolving a proposer and solver without human labels. Recent systems report strong reasoning gains, but collapse and instability are widely observed and poorly understood. The dominant response treats this as a reward-design problem. We argue instead that self-play stability is governed by two distinct levers: a data-level gate that decides which proposer-generated tasks enter the training pool, and the reward signal that updates the policy on tasks already admitted. Through controlled experiments on a Python output-prediction task and a deterministic-DSL twin task that strips pretraining priors, output ambiguity, and executor noise, we find the two levers are asymmetric. A strict gate is sufficient for stability under every reward variant we test, including a self-consistency reward with no access to ground truth; while no reward variant is sufficient once the gate is removed. This asymmetry exposes a counter-intuitive coupling we call the Grounded Proposer Paradox: a proposer with ground-truth access accelerates collapse faster than an ungrounded one when paired with a self-consistency solver, by concentrating training on clean tasks that form the fastest path to a spurious self-consistent attractor. Replacing the binary gate with a continuous strictness parameter $\varepsilon$ further reveals a two-stage phase transition: training-side metrics decouple at low $\varepsilon$, while validation accuracy holds until $\varepsilon$ is much higher. Data-level gating, not reward calibration, is the binding constraint on self-play stability. 

---
# Efficient Agentic Reasoning Through Self-Regulated Simulative Planning 

**Authors**: Mingkai Deng, Jinyu Hou, Lara Sá Neves, Varad Pimpalkhute, Taylor W. Killian, Zhengzhong Liu, Eric P. Xing  

**Link**: [PDF](https://arxiv.org/pdf/2605.22138)  

**Abstract**: How should an agent decide when and how to plan? A dominant approach builds agents as reactive policies with adaptive computation (e.g., chain-of-thought), trained end-to-end expecting planning to emerge implicitly. Without control over the presence, structure, or horizon of planning, these systems dramatically increase reasoning length, yielding inefficient token use without reliable accuracy gains. We argue efficient agentic reasoning benefits from decomposing decision-making into three systems: simulative reasoning (System II) grounding deliberation in future-state prediction via a world model; self-regulation (System III) deciding when and how deeply to plan via a learned configurator; and reactive execution (System I) handling fine-grained action. Simulative reasoning provides unified planning across diverse tasks without per-domain engineering, while self-regulation ensures the planner is invoked only when needed. To test this, we develop SR$^2$AM (Self-Regulated Simulative Reasoning Agentic LLM), realizing both as distinct stages within an LLM's chain-of-thought, with the LLM as world model. We explore two instantiations: recording decisions from a prompted multi-module system (v0.1) and reconstructing structured plans from traces of pretrained reasoning LLMs (v1.0), trained via supervised then reinforcement learning (RL). Across math, science, tabular analysis, and web information seeking, v0.1-8B and v1.0-30B achieve Pass@1 competitive with 120-355B and 685B-1T parameter systems respectively, while v1.0-30B uses 25.8-95.3% fewer reasoning tokens than comparable agentic LLMs. RL increases average planning horizon by 22.8% while planning frequency grows only 2.0%, showing it learns to plan further ahead rather than more often. More broadly, learned self-regulation instantiates a principle we expect to extend beyond planning to how agents govern their own learning and adaptation. 

---
# Reflecti-Mate: A Conversational Agent for Adaptive Decision-Making Support Through System 1 and System 2 Thinking 

**Authors**: Morita Tarvirdians, Senthil Chandrasegaran, Hayley Hung, Catholijn M. Jonker, Catharine Oertel  

**Link**: [PDF](https://arxiv.org/pdf/2605.22509)  

**Abstract**: Making high-stakes personal decisions involves cognitive, emotional, and intuitive processes, and individuals differ in how they allocate attention across these modes. Integration of these processes has shown to benefit decision making. Yet, most current decision-support systems focus primarily on supporting cognitive aspects, rather than adapting to the individual's thinking profile to support integration of different types of thoughts. In this study, we investigate an agent designed to encourage integration by adapting to the individual user's thought patterns. We explore its effects on participants' perceptions of the agent and their reflective behavior, in comparison with unaided pre-reflection and a baseline agent. In a between-subjects study (N = 128), our agent, which fostered broad and elaborated thinking, enabled more personalized reflective trajectories, elicited more integrative reflective language, and was perceived as providing stronger support for holistic reflection. In contrast, the baseline agent produced homogenized profiles dominated by cognitive language across participants. 

---
# Maestro: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles 

**Authors**: Jinyang Wu, Guocheng Zhai, Ruihan Jin, Yuhao Shen, Zhengxi Lu, Fan Zhang, Haoran Luo, Zheng Lian, Zhengqi Wen, Jianhua Tao  

**Link**: [PDF](https://arxiv.org/pdf/2605.22177)  

**Abstract**: The proliferation of large language models (LLMs) and modular skills has endowed autonomous agents with increasingly powerful capabilities. Existing frameworks typically rely on monolithic LLMs and fixed logic to interface with these skills. This gives rise to a critical bottleneck: different LLMs offer distinct advantages across diverse domains, yet current frameworks fail to exploit the complementary strengths of models and skills, thereby limiting their performance on downstream tasks. In this paper, we present Maestro (Multimodal Agent for Expert-Skill Targeted Reinforced Orchestration), a Reinforcement Learning (RL)-driven orchestration framework that reframes heterogeneous multimodal tasks as a sequential decision-making process over a hierarchical model-skill registry. Rather than consolidating all knowledge into a single model, Maestro trains a lightweight policy to dynamically compose ensembles of frozen expert models and a two-tier skill library, deciding at each step whether to invoke an external expert, which model-skill pair to select, and when to terminate. The policy is optimized via outcome-based RL, requiring no step-level supervision. We evaluate Maestro across ten representative multimodal benchmarks spanning mathematical reasoning, chart understanding, high-resolution perception, and domain-specific analysis. With only a 4B orchestrator, Maestro achieves an average accuracy of 70.1%, surpassing both GPT-5 (69.3%) and Gemini-2.5-Pro (68.7%). Crucially, the learned coordination policy generalizes to unseen models and skills without retraining: augmenting the registry with out-of-domain experts yields a 59.5% average on four challenging benchmarks, outperforming all closed-source baselines. Maestro further maintains high computational efficiency with low latency. The source code is available at this https URL. 

---
# Planning in the LLM Era: Building for Reliability and Efficiency 

**Authors**: Michael Katz, Harsha Kokel, Kavitha Srinivas, Shirin Sohrabi  

**Link**: [PDF](https://arxiv.org/pdf/2605.21902)  

**Abstract**: Growing attention to intelligent agents has put a spotlight on one of their central capabilities: planning. Early attempts to leverage large language models (LLMs) for planning relied on single-shot plan generation, followed by hybrid approaches that coupled LLMs with limited external search. These methods, unsound and incomplete by their very nature, often require substantial resources without yielding better solutions on unseen problems. As the limitations of LLMs become clearer, recent work has shifted toward using them at solution construction time -- generating symbolic solvers for a family of problems that can be verified and then used efficiently at inference time. This trend reflects the growing need for agents that are both reliable and resource-efficient. It also offers a path towards generating maintainable planners with minimal dependence on language models at inference time. In this paper, we argue that this shift reflects a broader realignment of the planning field in the LLM era. We examine three major categories of planner-generation methods, discuss their current limitations, and outline research steps towards a more reliable and efficient LLM-based generation of planners. 

---
# Blind Spots in the Guard: How Domain-Camouflaged Injection Attacks Evade Detection in Multi-Agent LLM Systems 

**Authors**: Aaditya Pai  

**Link**: [PDF](https://arxiv.org/pdf/2605.22001)  

**Abstract**: Injection detectors deployed to protect LLM agents are calibrated on static, template-based payloads that announce themselves as override directives. We identify a systematic blind spot: when payloads are generated to mimic the domain vocabulary and authority structures of the target document, what we call domain camouflaged injection, standard detectors fail to flag them, with detection rates dropping from 93.8% to 9.7% on Llama 3.1 8B and from 100% to 55.6% on Gemini 2.0 Flash. We formalize this as the Camouflage Detection Gap (CDG), the difference in injection detection rate between static and camouflaged payloads. Across 45 tasks spanning three domains and two model families, CDG is large and statistically significant (chi^2 = 38.03, p < 0.001 for Llama; chi^2 = 17.05, p < 0.001 for Gemini), with zero reverse discordant pairs in either case. We additionally evaluate Llama Guard 3, a production safety classifier, which detects zero camouflage payloads (IDRcamouflage = 0.000), confirming that the blind spot extends beyond few-shot detectors to dedicated safety classifiers. We further show that multi-agent debate architectures amplify static injection attacks by up to 9.9x on smaller models, while stronger models show collective resistance. Targeted detector augmentation provides only partial remediation (10.2% improvement on Llama, 78.7% on Gemini), suggesting the vulnerability is architectural rather than incidental for weaker models. Our framework, task bank, and payload generator are released publicly. 

---
# Epicure: Navigating the Emergent Geometry of Food Ingredient Embeddings 

**Authors**: Jakub Radzikowski, Josef Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.22391)  

**Abstract**: We present Epicure, a family of three sibling skip-gram ingredient embeddings retrained from scratch on a multilingual recipe corpus. We aggregate 4.14M recipes from 11 sources spanning seven languages, English, Chinese, Russian, Vietnamese, Spanish, Turkish, Indonesian, German, and Indian-English, and normalise the raw ingredient strings to 1,790 canonical entries via an LLM-augmented pipeline. A 203,508-edge ingredient-ingredient NPMI graph and an 80,019-edge typed FlavorDB ingredient-compound graph, 2,247 typed compound nodes across 15 categories, seed three Metapath2Vec variants that share architecture and hyperparameters and differ only in the random-walk schema: Cooc walks the co-occurrence graph only, Chem walks the typed compound metapaths only, and Core blends both via injected ingredient-ingredient walks at controlled mixing, placing each model at a distinct point on the chemistry-vs-recipe-context spectrum. 

---
# From Reasoning Chains to Verifiable Subproblems: Curriculum Reinforcement Learning Enables Credit Assignment for LLM Reasoning 

**Authors**: Xitai Jiang, Zihan Tang, Wenze Lin, Yang Yue, Shenzhi Wang, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22074)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) has shown strong promise for LLM reasoning, but outcome-based RLVR remains inefficient on hard problems because correct final-answer rollouts are rare and sample-level credit assignment cannot use partial progress in failed attempts. We introduce SCRL (Subproblem Curriculum Reinforcement Learning), a curriculum RL framework that derives verifiable subproblems from reference reasoning chains and fixes the final subproblem as the original problem. This turns partial progress on hard problems into verifiable learning signals. Algorithmically, SCRL uses subproblem-level normalization, which normalizes rewards independently at each subproblem position and assigns the resulting advantages to the corresponding answer spans, enabling finer-grained credit assignment without external rubrics or reward models. Our analysis shows that subproblem curricula lift hard problems out of gradient dead zones, with larger relative gains as the original problem becomes harder. Across seven mathematical reasoning benchmarks, SCRL outperforms strong curriculum-learning baselines, improving average accuracy over GRPO by +4.1 points on Qwen3-4B-Base and +1.9 points on Qwen3-14B-Base. On AIME24, AIME25, and IMO-Bench, SCRL further improves pass@1 by +3.7 points and pass@64 by +4.6 points on Qwen3-4B-Base, indicating better exploration on hard reasoning problems. 

---
# BEiTScore: Reference-free Image Captioning Evaluation with an Efficient Cross-Encoder Model 

**Authors**: Gonçalo Gomes, Bruno Martins, Chrysoula Zerva  

**Link**: [PDF](https://arxiv.org/pdf/2605.21728)  

**Abstract**: Image captioning evaluation remains a significant challenge, as vision-language models evolve toward more challenging capabilities such as generating long-form and context-rich descriptions. State-of-the-art evaluation metrics involve extensive computational costs associated with the use of Large Language Models (LLMs) as judges, or instead suffer from the limitations of standard CLIP-based encoders, such as strict token limits, lack of fine-grained sensitivity, or lack of compositional generalization by treating captions as ``bags-of-words.'' We propose a new learned metric that tackles the aforementioned challenges, based on a lightweight cross-encoder that is initialized from a visual question-answering model checkpoint, balancing a strong weight initialization with computational efficiency. Our training scheme uses a carefully assembled data mixture for supervised learning, featuring adversarial LLM-based data augmentations to enhance model sensitivity to fine-grained visual-linguistic errors. We also introduce a new benchmark designed to assess detailed captioning evaluation across diverse scenarios. Experimental results demonstrate that the proposed metric achieves state-of-the-art performance while maintaining the efficiency required for large-scale benchmarking, quality-aware decoding, or reward guidance. 

---
# Energy-Gated Attention: Spectral Salience as an Inductive Bias for Transformer Attention 

**Authors**: Athanasios Zeris  

**Link**: [PDF](https://arxiv.org/pdf/2605.21842)  

**Abstract**: Standard transformer attention computes pairwise similarity between queries and keys, treating all tokens as equally salient regardless of their intrinsic informational content. In turbulent fluid dynamics, coherent structures -- the energetically dominant, spatially organized patterns that persist amid background chaos -- carry a disproportionate fraction of total energy and govern all transport. We propose that tokens play an analogous role in transformer attention: informationally dense positions (morphological boundaries, syntactic heads, discourse markers) concentrate spectral energy and should attract proportionally more attention than background tokens (function words, repeated patterns, low-information filler). We propose Energy-Gated Attention (EGA): a simple modification that gates value aggregation by the spectral energy of key token embeddings, computed by a single learned linear projection that discovers the dominant spectral mode of the embedding field. On TinyShakespeare, EGA achieves +0.103 validation loss improvement with only 12,480 additional parameters (<0.26% overhead) and no measurable computational cost. The result is consistent on Penn Treebank (+0.101), demonstrating dataset independence. A systematic ablation across three wavelet families (fixed Morlet, Daubechies db2/db4, and a parametric Morlet) establishes that fixed structured bases are suboptimal -- the optimal energy direction is data-adaptive and non-sinusoidal -- while identifying learned wavelet packets as a promising open direction. The learned energy threshold converges to tau ~= 0.35 independently of initialization, corresponding to the fraction (~36%) of tokens carrying above-average spectral energy in English text, a stable linguistic property consistent with the fraction of content words in running English text. 

---
# Geometry-Adaptive Explainer for Faithful Dictionary-Based Interpretability under Distribution Shift 

**Authors**: Sungjun Lim, Heedong Kim, Andrew Lee, Kyungwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.21849)  

**Abstract**: Mechanistic interpretability aims to explain a model's behavior by identifying causally responsible internal structures. Dictionary-based explainers such as sparse autoencoders and transcoders are a primary tool, but their faithfulness under out-of-distribution (OOD) shift has received little systematic attention. We show that distribution shift rotates the subspace that the model actively uses, misaligning the explainer's dictionary trained on in-distribution (ID) activations. We formalize this misalignment as the faithfulness gap, a geometric distance between the ID dictionary and the OOD-active subspace, and show that it controls OOD faithfulness degradation. To reduce this gap, we propose the Geometry-Adaptive Explainer (GAE), which realigns the explainer's dictionary with the OOD-active subspace while preserving the original feature structure. This requires only unlabeled OOD activations and no gradient updates. We prove that GAE improves over the unadapted ID explainer, with excess loss bounded quadratically by the second-moment shift. Empirically, GAE even matches or surpasses all training-based baselines in causal faithfulness across multiple models and OOD settings. 

---
# Why Semantic Entropy Fails: Geometry-Aware and Calibrated Uncertainty for Policy Optimization 

**Authors**: Zheyuan Zhang, Kaiwen Shi, Han Bao, Zehong Wang, Tianyi Ma, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.21801)  

**Abstract**: Post-training has become central to improving reasoning and alignment in large language models, where critic-free models enable scalable learning from model-generated outputs but lack principled mechanisms to distinguish informative from noisy signals. Recent approaches leverage response-level measures as uncertainty signals to regulate group-based optimization methods such as GRPO. Yet their empirical success remains unstable and unclear in how they influence optimization dynamics. In this paper, we provide, to our knowledge, the first principled formulation that interprets uncertainty signals as mechanisms for characterizing and regulating gradient variance and learning signal quality. Based on both empirical and theoretical analysis, we identify two critical gaps of current entropy-based estimators: The anisotropic gap and The calibration gap. Motivated by this analysis, we propose Geometric-aware Calibrated Policy Optimization (GCPO), a novel framework integrating geometry-aware measures to capture semantic disagreement with reward-based calibration to align uncertainty with learning signal strength. Experiments on multiple benchmarks show that our approach more faithfully tracks gradient variability and consistently improves post-training performance. Our results highlight the importance of designing uncertainty signals that are aligned with optimization dynamics, offering a principled perspective for robust post-training. 

---
# HealthCraft: A Reinforcement Learning Safety Environment for Emergency Medicine 

**Authors**: Brandon Dent  

**Link**: [PDF](https://arxiv.org/pdf/2605.21496)  

**Abstract**: Frontier language models are being deployed into clinical workflows faster than the infrastructure to evaluate them safely. Static medical-QA benchmarks miss the failure modes that matter in emergency medicine: trajectory-level safety collapse, tool misuse, and capitulation under sustained clinical pressure. We present HealthCraft, the first public reinforcement-learning environment that rewards trajectory-level safety under realistic emergency-medicine conditions, adapted from Corecraft. It is built on a FHIR R4 world state with 14 entity types and 3,987 seed entities, exposes 24 MCP tools, and defines a dual-layer rubric that zeroes reward whenever any safety-critical criterion is violated. We release 195 tasks across six categories, graded against 2,255 binary criteria (515 safety-critical); a post-hoc 10-task negative-class slate extends this to 205 tasks and 2,337 criteria. V8 results on two frontier models show Claude Opus 4.6 at Pass@1 24.8% [21.5-28.4] and GPT-5.4 at 12.6% [10.2-15.6], with safety-failure rates of 27.5% and 34.0%. On multi-step workflows - the closest proxy to real emergency care - performance collapses to near zero (Claude 1.0%, GPT-5.4 0.0%) despite partial competence on individual steps. Six infrastructure bugs fixed between pilots v2 and v8 re-ordered which model "looks stronger," evidence that infrastructure fidelity is part of the measurement. A deterministic LLM-judge overlay bounds evaluator noise, and a 60-run negative-class smoke pilot shows the reward signal is not drop-in training-safe: restraint criteria pass at 0.929 prevalence, a gameability an eval harness can tolerate but a training reward cannot. We scaffold coupling to a Megatron+SGLang+GRPO loop per Corecraft Section 5.2 and leave training-reward ablations as future work. Environment, tasks, rubrics, and harness are released under Apache 2.0. 

---
# EntmaxKV: Support-Aware Decoding for Entmax Attention 

**Authors**: Gonçalo Duarte, Miguel Couceiro, Marcos V. Treviso  

**Link**: [PDF](https://arxiv.org/pdf/2605.21649)  

**Abstract**: Long-context decoding is increasingly limited by KV-cache memory traffic since each generated token attends over a cache whose size grows linearly with context length. Existing sparse decoding methods reduce this cost by selecting subsets of tokens or pages, but are designed for softmax attention, whose dense tails make any truncation discard nonzero probability mass. In contrast, $\alpha$-entmax produces exact zeros, turning sparse decoding from dense-tail approximation into support recovery: if the selected candidates contain the entmax support, sparse decoding remains exact. While recent entmax kernels enable efficient training, they do not address the autoregressive decoding bottleneck, where dense inference still streams the full KV cache before sparsity is known. In this work, we introduce EntmaxKV, an entmax-native sparse decoding framework that exploits sparsity before KV pages are loaded. EntmaxKV combines query-aware page scoring, support-aware candidate selection, and sparse entmax attention. We analyze truncation error through the dropped probability mass $\delta$, showing that output error is controlled by $\delta$ and vanishes when the entmax support is recovered. We further introduce a Gaussian-aware entmax selector that estimates the entmax threshold from lightweight page statistics, adapting the selected budget to the score distribution. Empirically, EntmaxKV drops less probability mass, retains more support tokens, and achieves lower output error than softmax-based sparse decoding at matched KV budgets. On long-context and language modeling benchmarks, it closely matches full-cache entmax while using a small fraction of the KV cache, achieving up to $3.36\times$ (softmax) and $5.43\times$ (entmax) speedup over full attention baselines at 1M context length. Code available at: this https URL. 

---
# Echo: Learning from Experience Data via User-Driven Refinement 

**Authors**: Hande Dong, Xiaoyun Liang, Jiarui Yu, Jiayi Lin, Changqing Ai, Feng Liu, Wenjun Zhang, Rongbi Wei, Chaofan Zhu, Linjie Che, Feng Wu, Xin Shen, Dexu Kong, Xiaotian Wang, Qiuyuan Chen, Bingxu An, Yueting Lei, Qiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.21984)  

**Abstract**: Static "human data" faces inherent limitations: it is expensive to scale and bounded by the knowledge of its creators. Continuous learning from "experience data" - interactions between agents and their environments - promises to transcend these barriers. Today, the widespread deployment of AI agents grants us low-cost access to massive streams of such real-world experience. However, raw interaction logs are inherently noisy, filled with trial-and-error and low information density, rendering them inefficient for direct model training.
We introduce Echo, a generalized framework designed to operationalize the transition from raw experience to learnable knowledge, effectively "echoing" environmental feedback back into the training loop for model optimization. In today's agent ecosystem, user refinement serves as a primary source of such feedback: driven by responsibility for the outcome, users rigorously transform flawed agent proposals into verified solutions. These user-driven refinement sequences inherently distill agents' crude attempts into high-quality training signals. Echo systematically harvests these signals to continuously align the agent with real-world needs. Large-scale validation in a production code completion environment confirms that Echo effectively harnesses this pipeline, breaking the static performance ceiling by increasing the acceptance rate from 25.7% to 35.7%. 

---
# X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation 

**Authors**: Sharath Turuvekere Sreenivas, Adithyakrishna Venkatesh Hanasoge, Mingyu Yang, Ali Taghibakhshi, Saurav Muralidharan, Ashwath Aithal, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2605.21699)  

**Abstract**: Cross-tokenizer knowledge distillation allows a student model to learn from teachers with incompatible vocabularies. Prior work operates on hidden states or logits; the latter is preferred as a drop-in replacement requiring no auxiliary components. Logit-based methods either use only the correct-token probability, missing the full 'dark knowledge' in the teacher's distribution, or operate on the full output distribution, relying on strict token partitioning and/or unprincipled heuristic ranking. We identify two key shortcomings of full-distribution, logit-based methods: (i) an uncommon-token failure, where critical tokens fall into the unmatched subset (e.g., Llama's 1100 multi-digit numerals under digit-splitting Qwen supervision) and are suppressed during training, reducing GSM8k from 12.89 to 2.56 compared to same-tokenizer KD from a weaker teacher; and (ii) over-conservative matching, where strict 1-to-1 matching excludes near-equivalent tokens across surface forms. These failures require distinct remedies: eliminating the partition when critical tokens are misaligned, and refining it when alignment is reliable. We propose X-Token, an approach with two complementary loss formulations targeting these issues. P-KL removes partitioning and aligns the student's distribution with the teacher's via a sparse projection matrix W (initialized from tokenizer-level string rules) to address the uncommon-token failure. H-KL retains the hybrid form while relaxing matching to align each student token with its top-ranked teacher mapping under W. Both objectives share W and extend naturally to multiple teachers. Empirically, on Llama-3.2-1B, X-Token outperforms the current state of the art GOLD by +3.82 average points with a Qwen3-4B teacher and by +0.5 with a Phi-4-Mini teacher. Further, a two-teacher setup (Phi-4-mini + Llama-3B) improves over single-teacher distillation by +1.3 points. 

---
# Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation 

**Authors**: Srujan P Mule, Aniketh Garikaparthi, Manasi Patwardhan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21491)  

**Abstract**: As language models accelerate scientific research by automating hypothesis generation and implementation, a new bottleneck emerges: evaluating and filtering hundreds of AI-generated ideas without exhaustive experimentation. We ask whether LMs can learn to forecast the empirical success of research ideas before any experiments are run. We study comparative empirical forecasting: given a benchmark-specific research goal and two candidate ideas, predict which will achieve better benchmark performance. We construct a dataset of 11,488 idea pairs grounded in objective outcomes from PapersWithCode. While off-the-shelf 8B-parameter models struggle (30% acc.), SFT dramatically boosts performance to 77.1%, outperforming GPT-5 (61.1%). By framing evaluation as a reasoning task via Reinforcement Learning with Verifiable Rewards (RLVR), we train models to discover latent reasoning paths, achieving 71.35% acc. with interpretable justifications. Through additional ablations and out-of-distribution tests, we show robustness to surface-level heuristics and transfer to both a cross-domain time-split test set and an independently constructed test set. Our results demonstrate that compute-efficient small language models can serve as effective, objective verifiers, offering a scalable path for autonomous scientific discovery. 

---
# Flat-Pack Bench: Evaluating Spatio-Temporal Understanding in Large Vision-Language Models through Furniture Assembly 

**Authors**: Aditya Chetan, Eric Cai, Peeyush Kushwaha, Bharath Raj Nagoor Kani, Utkarsh Mall, Qianqian Wang, Noah Snavely, Bharath Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21625)  

**Abstract**: The emergence of Large Vision-Language Models (LVLMs) has significantly advanced video understanding capabilities. However, existing benchmarks focus predominantly on coarse-grained tasks such as action segmentation, classification, captioning, and retrieval. Furthermore, these benchmarks often rely on entities that can be easily identified verbally, like household objects, animals, human subjects, etc., limiting their applicability to complex, in-the-wild video scenarios. But, many applications such as furniture assembly, cooking, etc., require step-by-step fine-grained spatio-temporal understanding of the video, which is not sufficiently evaluated in current benchmarks. To address this gap, we introduce Flat-Pack Bench, a novel benchmark centered on furniture assembly tasks. Our benchmark evaluates LVLMs on nuanced tasks, including temporal ordering of assembly actions, temporal localization of assembly state, understanding part mating, and tracking, using multiple-choice questions paired with visual prompts highlighting relevant parts as references for fine-grained questions. Our experiments reveal that state-of-the-art LVLMs struggle significantly with fine-grained spatio-temporal reasoning, highlighting their limitations in effectively leveraging temporal information from videos, limited tracking ability, and understanding of spatial interactions like physical contact. 

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
# Detecting Synthetic Political Narratives in Cross-Platform Social Media Discourse 

**Authors**: Despoina Antonakaki, Sotiris Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2605.21540)  

**Abstract**: The proliferation of large language models has introduced a new paradigm of synthetic political communication in which narratives may be generated, semantically coordinated, and strategically disseminated across platforms at scale. We present a cross-platform framework for detecting synthetic political narratives using four coordination signals -- lexical diversity D(C), temporal burstiness B(C), rhetorical repetition R(C), and semantic homogenization H(C) -- combined into a Synthetic Narrative Coordination Score SNC(C).
We apply the framework to a corpus of 353,223 records spanning six geopolitical event windows collected from six Telegram channels and nine Reddit communities (2023--2026). Results show that IntelSlava exhibits the lowest lexical diversity (MATTR 0.52--0.54), the highest burstiness (B=+0.48 to +0.73), and the highest rhetorical overlap with peer channels (Jaccard 0.12), ranking first in the composite SNC(C) on four of six event windows (SNC 0.45--0.60). Rybar ranks last on all windows despite its high semantic homogenization, because its Russian-language output yields high lexical diversity and near-zero rhetorical Jaccard with English-language channels -- demonstrating that no single indicator is sufficient for coordination detection. Multi-dimensional SNC(C) scoring provides a more robust and interpretable signal than any individual metric. 

---
# From Parameters to Data: A Task-Parameter-Guided Fine-Tuning Pipeline for Efficient LLM Alignment 

**Authors**: Hao Chen, Qi Zhang, Liyao Li, Zhanming Shen, Wentao Ye, Lirong Gao, Ningtao Wang, Xing Fu, Xiaoyu Shen, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.21558)  

**Abstract**: Adapting Large Language Models (LLMs) to specialized domains typically incurs high data and computational overhead. While prior efficiency efforts have largely treated data selection and parameter-efficient fine-tuning as isolated processes, our empirical analysis suggests they may be intrinsically coupled. We posit the Strong Map Hypothesis: a sparse subset of attention heads plays a dominant role in task-specific adaptation, acting as keys that unlock specific data patterns. Building on this observation, we propose From Parameters to Data (P2D), a unified framework that leverages these task-sensitive attention heads as a dual compass for both sample mining and structural pruning. To rigorously quantify the total pipeline cost, we introduce the Alignment Efficiency Ratio (AER) metric for both selection latency and training time. Mechanistically, P2D identifies critical heads via a lightweight proxy and uses them as a functional filter to curate high-affinity data, establishing a synergistic pipeline. Empirically, by updating merely 10% of attention heads on 10% of the data, P2D achieves an 8.3 pp performance gain over strong baselines and delivers a 7.0x end-to-end time speedup. These results validate that precise parameter-data synchronization eliminates redundancy, offering a new paradigm for efficient alignment. 

---
# MM-Conv: A Multimodal Dataset and Benchmark for Context-Aware Grounding in 3D Dialogue 

**Authors**: Anna Deichler, Jim O'Regan, Fethiye Irmak Dogan, Lubos Marcinek, Anna Klezovich, Iolanda Leite, Jonas Beskow  

**Link**: [PDF](https://arxiv.org/pdf/2605.21796)  

**Abstract**: Grounding language in the physical world requires AI systems to interpret references that emerge dynamically during conversation. While current vision-language models (VLMs) excel at static image tasks, they struggle to resolve ambiguous expressions in spontaneous, multi-turn dialogue. We address this gap by introducing (1) a benchmark for referential communication in dynamic 3D environments, built from 6.7 hours of egocentric VR interaction with synchronized speech, motion, gaze, and 3D scene geometry, and (2) a two-stage grounding pipeline that explicitly resolves conversational ambiguity before visual localization. The benchmark includes over 4,200 manually verified referring expressions spanning full, partitive, and pronominal types. Our contextual rewriting approach improves grounding performance by 11-22 percentage points on average, with a pure detector (GroundingDINO) reaching 56.7% on pronominals after rewriting, nearly double the best end-to-end baseline. Results demonstrate that decoupling linguistic reasoning from visual perception is more effective than end-to-end approaches for conversational grounding. 

---
# Reinforced Preference Optimization for Reasoning-Augmented Recommendations 

**Authors**: Jingtong Gao, Zeyu Song, Chi Lu, Xiaopeng Li, Derong Xu, Maolin Wang, Peng Jiang, Kun Gai, Qingpeng Cai, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.21967)  

**Abstract**: Recommender systems are critical for delivering personalized content across digital platforms, and recent advances in Large Language Models (LLMs) offer new opportunities to enhance them with richer world knowledge and explicit reasoning capabilities. With the help of reasoning knowledge, recommendations can better infer users' underlying intents, adapt to evolving preferences, and leverage semantic relationships for improved accuracy and interpretability. However, existing reasoning-based recommendation methods often fail to fully align the LLM's reasoning process with recommendation-specific objectives due to structural disruption during integration and difficulties in translating free-form generation into accurate item predictions. In this paper, we introduce RPORec, a reinforced preference optimization framework that unifies an LLM backbone's reasoning ability with a dedicated recommendation head (Rechead) for precise item retrieval. RPORec comprises two stages: (1) Reasoning-Augmented Recommendation Modeling, where high-quality Chain-of-Thought (CoT) reasoning is generated and used as auxiliary knowledge to guide the Rechead in learning recommendation-specific representations; and (2) Advanced Reasoning Refinement and Alignment, in which the trained Rechead produces verifiable rewards to fine-tune the LLM backbone via reinforcement learning, enhancing reasoning quality, structural consistency, and task relevance. Extensive experiments on public benchmarks and large-scale online deployments show that RPORec consistently outperforms state-of-the-art LLM-based recommendation methods, demonstrating the effectiveness of reasoning-augmented recommendation modeling in real-world systems. 

---
# Bridging the Cold-Start Gap: LLM-Powered Synthetic Data Generation for Natural Language Search at Airbnb 

**Authors**: Wendy Ran Wei, Hao Li, Weiwei Guo, Xiaowei Liu, Xueyin Chen, Dillon Davis, Malay Haldar, Soumyadip Banerjee, Kedar Bellare, Huiji Gao, Stephanie Moyerman, Sanjeev Katariya  

**Link**: [PDF](https://arxiv.org/pdf/2605.21812)  

**Abstract**: Deploying natural language search systems presents a critical cold-start challenge: no real user queries to learn linguistic patterns, and no relevance labels to train ranking models. We present a framework for generating synthetic queries and labels using large language models (LLMs), powering model training and evaluation for Airbnb's natural language search.
For query generation, we combine contrastive listing pairs from booking sessions with seed queries from user research to balance realism and diversity, enabling a cold-to-warm start transition as real user data becomes available. For label generation, we introduce contrastive generation that produces topicality labels by construction, and Virtual Judge (VJ) labeling for broader coverage.
We compare our approach against a no-seed contrastive baseline and an InPars-style baseline. For query length, the InPars baseline produces verbose queries with KL divergence of 12.03 vs. real users; our seed-guided approach achieves 0.66, a 7.5x improvement. For attribute type distributions, our approach achieves the lowest KL divergence (0.04), outperforming even seed queries (0.09). Experiments show our approach produces harder evaluation examples than the no-seed baseline (79% vs. 97% pairwise accuracy), providing discriminative signal for model improvement. We deploy production pipelines generating synthetic examples daily for embedding-based retrieval and ranking evaluation. 

---
# LLM Retrieval for Stable and Predictable Ad Recommendations 

**Authors**: Vinodh Kumar Sunkara, Satheeshkumar Karuppusamy, Hangjun Xu, Sai Deepika Regani, Kshitij Gupta, Gaby Nahum, Sneha Iyer, Jean-Baptiste Fiot, Yinglong Guo, Xiaowen Guo, Atul Jangra, Yucheng Liu, Jinghao Yan, Vijay Pappu, Benjamin Schulte, Deepak Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2605.21969)  

**Abstract**: Traditional ads recommendation systems have primarily focused on optimizing for prediction accuracy of click or conversion events using canonical metrics such as recall or normalized discounted cumulative gain (NDCG). With the hyper-growth of ads inventory and liquidity with generative AI technologies, the prediction stability and predictability is becoming increasingly critical. Intuitively, prediction stability and predictability can be defined to quantify system robustness with respect to minor/noisy input (ads, creatives) perturbations, the lack of which could lead to advertiser perceivable problems such as repeatability, cold start and under-exploration. In this paper, we introduce a new evaluation framework for quantifying stability and predictability of an ads recommender system, and present an online validated semantic candidate generation framework powered by fine-tuned Large Language Models (LLMs) that showed significant improvement along these metrics by fundamentally improving the semantic-awareness of the system. The approach extracts hierarchical semantic attributes from ad creatives to obtain LLM representations, which serve as the foundation for graph-based expansion, ensuring the retrieved candidates encapsulate semantic variants of an ad, guaranteeing that small creative variants from the advertiser yield consistent and explainable delivery results to the user. We tested this LLM ads retrieval framework in a large-scale industrial ads recommendation system, demonstrating significant improvements across offline and online A/B experiments, showcasing gains in both predictability and traditional performance metrics. Although evaluated in the ads stack, this is a general framework that can be applied broadly to any large-scale recommendation and retrieval systems facing similar scaling and predictability challenges. 

---
# Integrating Chain-of-Thought into Generative Retrieval: A Preliminary Study 

**Authors**: Wenhao Zhang, Ruihao Yu, Yi Bai, Zhumin Chen, Pengjie Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.22358)  

**Abstract**: While generative retrieval (GR) demonstrates competitive performance on standard retrieval benchmarks, existing approaches directly map queries to document identifiers (docids) without intermediate deliberation, limiting their effectiveness for complex queries that require multi-step reasoning. As a preliminary study on integrating chain-of-thought (CoT) into generative retrieval, we introduce ThinkGR, a unified framework that interleaves CoT with docid generation, enabling iterative thinking and retrieval within a single generative process. To bridge the gap between free-form thought generation and structured retrieval targets, we design (1) a hybrid decoding strategy that dynamically switches between unconstrained thought generation and constrained docid decoding, and (2) a two-phase training approach that first aligns thought-retrieval patterns through supervised fine-tuning, then optimizes thought quality via retrieval-grounded reinforcement learning. Experiments on four multi-hop retrieval benchmarks demonstrate that ThinkGR achieves state-of-the-art performance with an average improvement of +6.86\%. Our work opens new avenues for enhancing generative retrieval with explicit deliberation capabilities, with promising implications for retrieval tasks requiring complex reasoning. 

---
# Direct content-based retrieval from music scores images 

**Authors**: Noelia Luna-Barahona, Antonio Ríos-Vila, David Rizo, Jorge Calvo-Zaragoza  

**Link**: [PDF](https://arxiv.org/pdf/2605.22255)  

**Abstract**: The digitization of musical scores plays a crucial role in their preservation and accessibility, yet information retrieval still depends mainly on metadata searches, such as by title or composer. Content based search in music score images remains underexplored compared to text documents, despite its potential value for musicians, musicologists, and educators. This work contributes to the field by first studying which characteristics of a score are most relevant for search and by defining a systematic method to build query datasets from any annotated corpus. We also consider diverse methods for content-based search on music score images, ranging from transcription-based approaches relying on Optical Music Recognition (OMR), to a transcription-free Transformer model trained to recognize queries directly from score images, and a text-prompted Large Language Model. Our experiments evaluate these models on four corpora exhibiting diverse characteristics in terms of dataset size, image quality, and typesetting mechanisms. Overall, each method excels under different conditions: OMR-based pipelines achieve higher in-domain retrieval, whereas transcription-free models handle domain variability more effectively. 

---
# Can AI Make Conflicts Worse? An Alignment Failure in LLM Deployment Across Conflict Contexts 

**Authors**: Andrii Kryshtal  

**Link**: [PDF](https://arxiv.org/pdf/2605.22720)  

**Abstract**: AI models are already deployed in societies affected by armed conflict, and journalists, humanitarian workers, governments and ordinary citizens rely on them for information or for their work processes. No established practice exists for checking whether their outputs can make those conflicts worse. We tested nine model configurations from four providers (OpenAI, Anthropic, DeepSeek, xAI) on 90 multi-turn scenarios designed to surface misaligned behaviour in conflict contexts: false equivalence between documented atrocities, denial of genocide, and failure to recognise ethnic slurs, among others. When such outputs feed into journalism, humanitarian reporting, or public debate, they can deepen divisions in fragile societies. Failure rates span 6\% to 47\% between the best and worst performing models, which makes model choice a safety question in its own right and when users pushed for ``balance'' in cases where international courts have already assigned responsibility, five of nine configurations failed 80 to 100 percent of the time. We release the first evaluation framework for this domain and propose adding it to alignment evaluation portfolios. 

---
# MOSS: Self-Evolution through Source-Level Rewriting in Autonomous Agent Systems 

**Authors**: Qianshu Cai, Yonggang Zhang, Xianzhang Jia, Wei Xue, Jun Song, Xinmei Tian, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.22794)  

**Abstract**: Autonomous agentic systems are largely static after deployment: they do not learn from user interactions, and recurring failures persist until the next human-driven update ships a fix. Self-evolving agents have emerged in response, but all confine evolution to text-mutable artifacts -- skill files, prompt configurations, memory schemas, workflow graphs -- and leave the agent harness untouched. Since routing, hook ordering, state invariants, and dispatch live in code rather than in any text artifact, an entire class of structural failure is physically unreachable from the text layer. We argue that source-level adaptation is a fundamentally more general medium: it is Turing-complete, a strict superset of every text-mutable scope, takes effect deterministically rather than through base-model compliance, and does not erode under long-context drift. We present MOSS, a system that performs self-rewriting at the source level on production agentic substrates. Each evolution is anchored to an automatically curated batch of production-failure evidence and proceeds through a deterministic multi-stage pipeline; code modification is delegated to a pluggable external coding-agent CLI while MOSS retains stage ordering and verdicts. Candidates are verified by replaying the batch against the candidate image in ephemeral trial workers, then promoted via user-consent-gated, in-place container swap with health-probe-gated rollback. On OpenClaw, MOSS lifts a four-task mean grader score from 0.25 to 0.61 in a single cycle without human intervention. 

---
# LCGuard: Latent Communication Guard for Safe KV Sharing in Multi-Agent Systems 

**Authors**: Sadia Asif, Mohammad Mohammadi Amiri, Momin Abbas, Prasanna Sattigeri, Karthikeyan Natesan Ramamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2605.22786)  

**Abstract**: Large language model (LLM)-based multi-agent systems increasingly rely on intermediate communication to coordinate complex tasks. While most existing systems communicate through natural language, recent work shows that latent communication, particularly through transformer key-value (KV) caches, can improve efficiency and preserve richer task-relevant information. However, KV caches also encode contextual inputs, intermediate reasoning states, and agent-specific information, creating an opaque channel through which sensitive content may propagate across agents without explicit textual disclosure. To address this, we introduce \textbf{LCGuard} (Latent Communication Guard), a framework for safe KV-based latent communication in multi-agent LLM systems. LCGuard treats shared KV caches as latent working memory and learns representation-level transformations before cache artifacts are transmitted across agents. We formalize representation-level sensitive information leakage operationally through reconstruction: a shared cache artifact is unsafe if an adversarial decoder can recover agent-specific sensitive inputs from it. This leads to an adversarial training formulation in which the adversary learns to reconstruct sensitive inputs, while LCGuard learns transformations that preserve task-relevant semantics and reduce reconstructable information. Empirical evaluations across multiple model families and multi-agent benchmarks show that LCGuard consistently reduces reconstruction-based leakage and attack success rates while maintaining competitive task performance compared to standard KV-sharing baselines. 

---
# Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention 

**Authors**: Ali Hatamizadeh, Yejin Choi, Jan Kautz  

**Link**: [PDF](https://arxiv.org/pdf/2605.22791)  

**Abstract**: Linear attention replaces the unbounded cache of softmax attention with a fixed-size recurrent state, reducing sequence mixing to linear time and decoding to constant memory. The hard part is not just what to forget, but how to edit this compressed memory without scrambling existing associations. Delta-rule models subtract the current read before writing a new value, and Kimi Delta Attention (KDA) sharpens forgetting with channel-wise decay. But the active edit still uses a single scalar gate to control two different things: how much old content to erase on the key side and how much new content to commit on the value side. We introduce Gated DeltaNet-2, which generalizes both Gated DeltaNet and KDA by inheriting adaptive forgetting and channel-wise decay while addressing their shared limitation, the scalar tie between erasing and writing. Gated Delta Rule-2 separates these roles with a channel-wise erase gate b_t and a channel-wise write gate w_t, reducing to KDA when both gates collapse to the same scalar and to Gated DeltaNet when the decay also collapses. We derive a fast-weight update view, a chunkwise WY algorithm with channel-wise decay absorbed into asymmetric erase factors, and a gate-aware backward pass that preserves efficient parallel training. At 1.3B parameters trained on 100B FineWeb-Edu tokens, Gated DeltaNet-2 achieves the strongest overall results among Mamba-2, Gated DeltaNet, KDA, and Mamba-3 variants across language modeling, commonsense reasoning, and retrieval. Its advantage is most pronounced on long-context RULER needle-in-a-haystack benchmarks, where it improves the evaluated multi-key retrieval setting and remains strong in both recurrent and hybrid settings. Code is available at this https URL. 

---
# Advancing Mathematics Research with AI-Driven Formal Proof Search 

**Authors**: George Tsoukalas, Anton Kovsharov, Sergey Shirobokov, Anja Surina, Moritz Firsching, Gergely Bérczi, Francisco J. R. Ruiz, Arun Suggala, Adam Zsolt Wagner, Eric Wieser, Lei Yu, Aja Huang, Miklós Z. Horváth, Andrew Ferrauiolo, Henryk Michalewski, Codrut Grosu, Thomas Hubert, Matej Balog, Pushmeet Kohli, Swarat Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2605.22763)  

**Abstract**: Large language models (LLMs) increasingly excel at mathematical reasoning, but their unreliability limits their utility in mathematics research. A mitigation is using LLMs to generate formal proofs in languages like Lean. We perform the first large-scale evaluation of this method's ability to solve open problems. Our most capable agent autonomously resolved 9 of 353 open Erdős problems at the per-problem cost of a few hundred dollars, proved 44/492 OEIS conjectures, and is being deployed in combinatorics, optimization, graph theory, algebraic geometry, and quantum optics research. A basic agent alternating LLM-based generation with Lean-based verification replicated the Erdős successes but proved costlier on the hardest problems. These findings demonstrate the power of AI-aided formal proof search and shed light on the agent designs that enable it. 

---
# Is Capability a Liability? More Capable Language Models Make Worse Forecasts When It Matters Most 

**Authors**: Nick Merrill, Jaeho Lee, Ezra Karger  

**Link**: [PDF](https://arxiv.org/pdf/2605.22672)  

**Abstract**: We document inverse scaling in LLMs on forecasting problems whose underlying time series exhibit superlinear growth and tail risk of regime change, a structure common in finance and epidemiology. On these tasks, more capable models produce worse distributional forecasts. The pattern appears on ForecastBench-Sim (FBSim), a contamination-free, simulated-world benchmark we release, in forecasting synthetic SIR epidemics with a matched linear control, and replicates in real-world datasets on COVID-19, measles, housing markets, and hyperinflation. A per-quantile decomposition shows the failure concentrates at the upper tail, which more capable models shift upward to track aggressive extrapolations of growth, while the lower tail stays put. A within-family study of Llama-3.1 shows that both model scale and post-training independently contribute to this effect. Domain knowledge does not reliably rescue calibration. This inverse scaling does not appear on single-threshold metrics common in LLM forecasting benchmarks, reversing the sign of the capability--accuracy relationship on identical outputs. Single-threshold scoring at conventional cutoffs misses the upper-tail cost; tail-inclusive scoring reverses the sign of the capability--accuracy relationship on the same outputs. We recommend that LLM forecasting evaluations use continuous (and unbounded) measures of accuracy alongside bounded binary threshold metrics. 

---
# HarnessAPI: A Skill-First Framework for Unified Streaming APIs and MCP Tools 

**Authors**: Edwin Jose  

**Link**: [PDF](https://arxiv.org/pdf/2605.22733)  

**Abstract**: Every Python function deployed as an LLM tool must today exist in two forms: an HTTP endpoint for human-facing clients and CI pipelines, and an MCP tool registration for agent runtimes such as Claude and Cursor. These representations share business logic yet diverge in all the surrounding machinery (routing, validation, serialisation, streaming, and schema maintenance), and they drift apart as the underlying code evolves. We present HarnessAPI, a Python framework that eliminates this duplication by treating a typed skill folder as the single source of truth. From one this http URL plus Pydantic schemas, the framework automatically derives a streaming HTTP endpoint with Server-Sent Events, an interactive OpenAPI/Swagger UI, and a zero-configuration MCP tool, all served from a single process. Dual-mode content negotiation lets the same handler serve SSE-streaming and JSON-returning clients with no handler changes. A dynamic code-generation mechanism ensures Pydantic type annotations propagate correctly to FastMCP's inspection layer, resolving a technical limitation that prevents naive closure-based registration. Measured across six representative skills using cloc, HarnessAPI reduces framework-facing boilerplate by 74% compared with a manually maintained dual-stack implementation (FastAPI server + FastMCP server). HarnessAPI subclasses FastAPI, inheriting its full middleware, dependency-injection, and deployment ecosystem. It is available at this https URL and on PyPI (pip install harnessapi) 

---
# Towards a General Intelligence and Interface for Wearable Health Data 

**Authors**: Girish Narayanswamy, Maxwell A. Xu, A. Ali Heydari, Samy Abdel-Ghaffar, Marius Guerard, Kara Vaillancourt, Zhihan Zhang, Jake Garrison, Levi Albuquerque, Dimitris Spathis, Hong Yu, Hamid Palangi, Xuhai "Orson" Xu, David G.T. Barrett, Joseph Breda, Jed McGiffin, Yubin Kim, Yuwei Zhang, Naghmeh Rezaei, Samuel Solomon, Karan Ahuja, Tim Althoff, Jake Sunshine, Ming-Zher Poh, Benjamin Yetton, Ari Winbush, Nicholas B. Allen, James M. Rehg, Isaac Galatzer-Levy, Yun Liu, John Hernandez, Anupam Pathak, Conor Heneghan, Yuzhe Yang, Ahmed A. Metwally, Pushmeet Kohli, Mark Malhotra, Shwetak Patel, Xin Liu, Daniel McDuff  

**Link**: [PDF](https://arxiv.org/pdf/2605.22759)  

**Abstract**: While ubiquitous wearable sensors capture a wealth of behavioral and physiological information, effectively transforming these signals into personalized health insights is challenging. Specifically, converting low-level sensor data into representations capable of characterizing higher-level states is difficult due to high phenotypic diversity and variation in individual baseline health, physiology, and lifestyle factors. Moreover, collecting wearable data paired with health outcome annotations is laborious and expensive, and retrospective annotation remains practically unfeasible, contributing to a scarcity of data with high-quality labels. To overcome these limitations, we propose a foundation model for wearable health that is pretrained on more than one trillion minutes of unlabeled sensor signals drawn from a large cohort of five million participants. We demonstrate that the joint scaling of model capacity and pretraining data volume leads to systematic improvements in performance, as evaluated on a diverse set of 35 health prediction tasks, spanning cardiovascular, metabolic, sleep, and mental health, as well as lifestyle choices and demographic factors. We find that this population scale representation unlocks label-efficient few-shot learning and generative capabilities for robust daily metric estimation. To further leverage this learned representation, we deploy a classroom of LLM agents to autonomously search the space of downstream predictive heads built on the model embeddings, showing broad performance improvements that increase with LLM model capacity. Finally, we show how integrating these downstream predictors into a Personal Health Agent can support model responses that are more relevant, contextually aware, and safe, and we validate this via 1,860 ratings from a cohort of clinicians. 

---
# WorkstreamBench: Evaluating LLM Agents on End-to-End Spreadsheet Tasks in Finance 

**Authors**: Thomson Yen, Julian Poeltl, Harshith Srinivas Gear, Yilin Meng, Joshua Fan, Adam Shen, Yili Liu, Ali Bauyrzhan, Siri Du, Haoyang Liu, Daniel Guetta, Hongseok Namkoong  

**Link**: [PDF](https://arxiv.org/pdf/2605.22664)  

**Abstract**: LLM agents are increasingly expected to carry out end-to-end workflows, producing complete artifacts from high-level user instructions. To meet enterprise needs, frontier AI labs have developed agents that can construct entire spreadsheets from scratch. This is especially relevant in finance, where core workflows such as financial modeling, forecasting, and scenario analysis are commonly conducted through spreadsheets. Yet, existing spreadsheet benchmarks do not measure this advanced capability, focusing instead on question-answering or single-formula edits. To address this gap, we provide one of the first evaluations of agents on end-to-end spreadsheet tasks, focusing on economically critical financial workflows such as modeling and scenario analysis. Since deliverables therein are routinely reviewed and revised by multiple stakeholders, judging their quality necessarily involves high-level criteria such as readability or ease of modification. To reflect the multidimensional nature of solution quality, we develop an evaluation taxonomy comprising three dimensions: Accuracy, Formula, and Format, each comprising fine-grained criteria that reflect professional standards. The Claude family leads the benchmark and produces the most professional-looking outputs in our qualitative review, but even the strongest agents frequently fall short of professional finance standards and degrade sharply as the difficulty increases beyond a few chained calculations. This suggests that current agents are not yet able to reliably produce professional-quality spreadsheets at the level of complexity real-world workflows demand. 

---
# Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning 

**Authors**: Banghao Chi, Yining Xie, Mingyuan Wu, Jingcheng Yang, Jize Jiang, Zhaoheng Li, Shengyi Qian, Minjia Zhang, Klara Nahrstedt, Rui Hou, Xiangjun Fan, Hanchao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22642)  

**Abstract**: Spreadsheet systems (e.g., Microsoft Excel, Google Sheets) play a central role in modern data-centric workflows. As AI agents grow increasingly capable of automating complex tasks, such as controlling computers and generating presentations, building an AI-driven spreadsheet agent has emerged as a promising research direction. Most existing spreadsheet agents rely on specialized prompting over general-purpose LLMs; while this design has potentials on simple spreadsheet operations, it struggles to manage the complex, multi-step workflows typical of real-world applications.
We introduce Spreadsheet-RL, a reinforcement learning (RL) fine-tuning framework designed to train specialized spreadsheet agents within a realistic Microsoft Excel environment. Spreadsheet-RL features an automated pipeline for scalable collection of paired start-goal spreadsheets from online forums, as well as domain-specific evaluation tasks in areas such as finance and supply chain management, which we compile into the new Domain-Spreadsheet benchmark dataset. It also includes a Spreadsheet Gym environment designed for multi-turn RL: Spreadsheet Gym exposes extensive Excel functionality through a Python sandbox, along with a refined harness that incorporates a comprehensive tool set and carefully designed tool-routing rules for spreadsheet tasks. Through comprehensive experiments, we show that Spreadsheet-RL substantially enhances AI agent's performance on both general and domain-specific spreadsheet tasks: it improves Qwen3-4B-Thinking-2507's Pass@1 on SpreadsheetBench from 12.0% to 23.4%, and raises Pass@1 from 8.4% to 17.2% on our curated Domain-Spreadsheet dataset. These results highlight Spreadsheet-RL's strong potential for generalization and real-world adoption in spreadsheet automation, and broadly, its promise for advancing LLM-based interactions with data interfaces in everyday work. 

---
# Forecasting Scientific Progress with Artificial Intelligence 

**Authors**: Sean Wu, Pan Lu, Yupeng Chen, Jonathan Bragg, Yutaro Yamada, Peter Clark, David Clifton, Philip Torr, James Zou, Junchi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22681)  

**Abstract**: Artificial intelligence (AI) is increasingly embedded in scientific discovery, yet whether it can anticipate scientific progress remains unclear. To study this question, we introduce a temporally grounded evaluation framework for forecasting scientific progress under controlled knowledge constraints. We present CUSP (Cutoff-conditioned Unseen Scientific Progress), a multi-disciplinary and event-level benchmark that evaluates scientific forecasting in AI systems through feasibility assessment, mechanistic reasoning, generative solution design, and temporal prediction. Across 4,760 scientific events, we observe systematic and domain-dependent limitations in current frontier models. While models can identify plausible research directions from competing candidates, they fail to reliably predict whether scientific advances will be realized and systematically misestimate when they will occur. Performance is highly heterogeneous across domains, with the timing of AI progress more predictable than advances in biology, chemistry, and physics. Performance is largely insensitive to whether events occur before or after the training cutoff, suggesting these limitations cannot be explained solely by knowledge exposure in training data. Under controlled information access, additional pre-cutoff knowledge improves performance but does not close the gap to full-information settings, which becomes more pronounced for high-citation advances. Models also exhibit systematic overconfidence and strong response biases, indicating unreliable uncertainty estimation. Taken together, current AI systems fall short as predictive tools for scientific progress. Access to prior knowledge does not translate into reliable forecasting, and performance benefits more from post-event information than from forward-looking prediction. 

---
# Think Thrice Before You Speak: Dual knowledge-enhanced Theory-of-Mind Reasoning for Persuasive Agents 

**Authors**: Minghui Ma, Bin Guo, Runze Yang, Mengqi Chen, Yan Liu, Jingqi Liu, Yahan Pei, Xuehao Ma, Qiuyun Zhang, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22602)  

**Abstract**: Persuasive dialogue requires reasoning about others' latent mental states, a capability known as Theory of Mind (ToM). However, due to reliance on simple prompting strategies and insufficient ToM knowledge, existing LLMs often fail to capture the intrinsic dependencies among mental states, leading to fragmented representations and unstable reasoning. To address these challenges, we introduce the ToM-based Persuasive Dialogue (ToM-PD) task, grounded in the Belief-Desire-Intention (BDI) framework, which explicitly models the sequential dependencies among mental states in multi-turn dialogues. To facilitate research on this task, we construct a large-scale annotated dataset, ToM-based Broad Persuasive Dialogues (ToM-BPD), capturing fine-grained mental states and corresponding persuasive strategies. We further propose Think Thrice Before You Speak (TTBYS), a knowledge-enhanced stepwise reasoning framework that leverages both explicit and implicit prior experiences to improve LLMs' inference of desires, beliefs, and persuasive strategies. Experimental results demonstrate that Qwen3-8B equipped with TTBYS outperforms GPT-5 by 1.20%, 22.80%, and 16.97% in predicting desires, beliefs, and persuasive strategies, respectively. Case studies further show that our approach enhances interpretability and consistency in reasoning. 

---
# AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters 

**Authors**: Hanjun Luo, Zhimu Huang, Sylvia Chung, Yiran Wang, Yingbin Jin, Jialin Li, Jiang Li, Xinfeng Li, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2605.22645)  

**Abstract**: Text-to-image (T2I) systems increasingly rely on upstream prompters, either humans or multimodal large language models (MLLMs), to translate user intent into detailed prompts. Yet current benchmarks fix the prompt and only evaluate T2I models, leaving the prompting proficiency of this upstream component entirely unmeasured. We introduce AtelierEval, the first unified benchmark that quantifies prompting proficiency across 360 expert-crafted tasks. Grounded in a cognitive view, it spans three task categories and instantiates tasks using a taxonomy of real-world challenges, with a dual interface for both humans and MLLMs. To enable scalable and reliable evaluation, we propose AtelierJudge, a skill-based, memory-augmented agentic evaluator. It produces subjective and objective scores for prompt-image pairs, achieving a Spearman correlation of 0.79 with human experts, approaching human performance. Extensive experiments benchmark 8 MLLMs against 48 human users across 4 T2I backends, validate AtelierEval as a robust diagnostic tool, and reveal the superiority of mimicry over planning, advocating for an image-augmented direction for future prompters. Our work is released to support future research. 

---
# TerminalWorld: Benchmarking Agents on Real-World Terminal Tasks 

**Authors**: Zhaoyang Chu, Jiarui Hu, Xingyu Jiang, Pengyu Zou, Han Li, Chao Peng, Peter O'Hearn, Earl T. Barr, Mark Harman, Federica Sarro, He Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.22535)  

**Abstract**: We introduce TerminalWorld, a scalable data engine that automatically reverse-engineers high-fidelity evaluation tasks from "in-the-wild" terminal recordings. Processing 80,870 terminal recordings, the engine yields a full benchmark of 1,530 validated tasks, spanning 18 real-world categories, ranging from short everyday operations to workflows exceeding 50 steps, and covering 1,280 unique commands. From these, we curate a Verified subset of 200 representative, manually reviewed tasks. Comprehensive benchmarking on TerminalWorld-Verified across eight frontier models and six agents reveals that current systems still struggle with authentic terminal workflows, achieving a maximum pass rate of only 62.5%. Moreover, TerminalWorld captures real-world terminal capabilities distinct from existing expert-curated benchmarks (e.g., Terminal-Bench), with only a weak correlation to their scores (Pearson r=0.20). The automated engine makes TerminalWorld authentic and scalable by construction, enabling it to evaluate agents in real-world terminal environments as developer practices evolve. Data and code are available at this https URL. 

---
# LACO: Adaptive Latent Communication for Collaborative Driving 

**Authors**: Tianhao Chen, Yuheng Wu, Dongman Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.22504)  

**Abstract**: Collaborative driving aims to improve safety and efficiency by enabling connected vehicles to coordinate under partial observability. Recent approaches have evolved from sharing visual features for perception to exchanging language-based reasoning through foundation models for behavioral coordination. Though communicating in language provides intuitive information, it introduces two challenges: high latency caused by autoregressive decoding and information loss caused by compressing rich internal representations into discrete tokens. To address these challenges, we analyze latent communication in collaborative driving under inherent limitations of multi-agent settings. Our analysis reveals agent identity confusion, where direct fusion of latent states entangles decision representations across vehicles. Motivated by this, we propose LACO, a training-free \textbf{LA}tent \textbf{CO}mmunication paradigm that seamlessly adapts pretrained driving models to collaborative settings. LACO introduces Iterative Latent Deliberation (ILD) for latent reasoning, Cross-Horizon Saliency Attribution (CHSA) for communication-efficient information selection, and Structured Semantic Knowledge Distillation (SSKD) to stabilize ego-centric decision making. Closed-loop experiments in CARLA show that LACO notably reduces communication and inference latency while maintaining strong collaborative driving performance. 

---
# Meta-Soft: Leveraging Composable Meta-Tokens for Context-Preserving KV Cache Compression 

**Authors**: Wei Luo, Yi Huang, Songchen Ma, Huanyu Qu, Jiang Cai, Mingkun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22337)  

**Abstract**: The KV cache used in large language models has linearly growing time complexity, so LLMs face memory blow-up and reduced decoding efficiency when they process long this http URL KV Cache eviction has become an important research direction; however, existing methods based on fixed Soft Tokens (e.g., Judge Q) rely on a static parameter set as the query to evaluate the importance of KV pairs, so they cannot adapt dynamically to different input prompts, and they cannot precisely capture complex and changing task this http URL, evicted KV pairs are discarded permanently, so this causes irreversible information loss and context breaks. To address this problem, we propose Meta-Soft, a dynamic compression framework based on probe-driven context integration. Specifically, we build a meta-library with a learnable orthogonal basis matrix $\mathcal{L}$, and we use a selector network with Gumbel-Softmax to produce differentiable sparse combination weights, so we dynamically synthesize the most targeted $k$ Soft Tokens from the input prompt this http URL append these Soft Tokens to the end of the input sequence to probe key information. We also introduce an attention-flow based integration mechanism, which redistributes the semantic information of removed tokens into retained tokens, and this keeps the dropped context information this http URL on multiple datasets show that our method outperforms existing state-of-the-art eviction methods and provides a new solution for KV Cache compression. 

---
# SciCore-Mol: Augmenting Large Language Models with Pluggable Molecular Cognition Modules 

**Authors**: Yuxuan Chen, Changwei Lv, Yunduo Xiao, Zhongjing Du, Daquan Zhou, Yukun Yan, Zheni Zeng, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22287)  

**Abstract**: Large Language Models (LLMs) are central to the one-for-all intelligent paradigm, but they face a fundamental challenge when dealing with heterogeneous scientific data such as molecules: the inherent gap between discrete linguistic symbols and topological molecular or continuous reaction data leads to significant information loss and semantic noise in text-based reasoning. We propose SciCore-Mol, a modular framework that bridges this gap through three deeply integrated pluggable cognitive modules: a topology-aware perception module, a latent diffusion-based molecular generation module, and a reaction-aware reasoning module. Each module is coupled to the LLM backbone through learned representation interfaces, enabling richer information exchange than is possible with text-only tool feedback. Our experiments on diverse chemical tasks demonstrate that SciCore-Mol achieves strong comprehensive performance across molecular understanding, generation, reaction prediction, and general chemistry knowledge, with an 8B-parameter open-source system that is competitive with and in several dimensions surpasses proprietary large models. This work provides a systematic blueprint for equipping LLMs with scientific expertise through decoupled, pluggable, and flexibly orchestrated modules, with direct implications for drug design, chemical synthesis, and broader scientific discovery. 

---
# SGR-Bench: Benchmarking Search Agents on State-Gated Retrieval 

**Authors**: Ningyuan Li, Haiyang Shen, Mugeng Liu, Yudong Han, Zhuofan Shi, Sixiong Xie, Yun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.22219)  

**Abstract**: Recent advances in large language models and tool-using agents have expanded the range of benchmarked web tasks. Yet an important class of specialized retrieval tasks remains undercharacterized. On many specialized data-retrieval websites, answer-bearing evidence becomes accessible only after establishing the correct site-specific retrieval state through filters, views, hierarchies, or scopes. We term this capability state-gated retrieval (SGR). We introduce SGR-Bench, a benchmark for this setting containing 100 expert-curated tasks spanning six source families and 12 public data ecosystems. Each task requires discovering the appropriate website and configuring its site-specific retrieval state to produce a structured answer. SGR-Bench pairs constraint-guided and goal-oriented formulations of the same underlying problems, enabling controlled comparisons between explicit and implicit guidance for state-gated retrieval. We evaluate eight CLI-based agentic LLM systems and three commercial search-agent products. On SGR-Bench, the strongest system reaches only 66.18% item-level F1, while row-level F1 remains much lower. A manual audit of 156 analyzable failed CLI trajectories shows why: agents often reach a relevant web source, but establish the wrong site-specific retrieval state. Retrieval-scope drift (37.2%) and criterion mismatch (27.6%) dominate, whereas final answer composition accounts for only 10.3%. The dataset and single-case evaluation instructions are available at this https URL. 

---
# Evaluating Large Language Models as Live Strategic Agents: Provider Performance, Hybrid Decomposition, and Operational Gaps in Timed Risk Play 

**Authors**: H. C. Ekne  

**Link**: [PDF](https://arxiv.org/pdf/2605.22238)  

**Abstract**: Static benchmarks capture only part of how large language models behave in practice. Real systems place models inside repeated loops with time limits, formatting constraints, and failure modes. We study this setting in a timed multi-phase Risk environment with explicit victory targets and repeated planning and execution cycles. In a replicated 32-game cross-provider championship under frozen rules, gemini-3.1-pro-preview won 20 of 32 games against gpt-5.1, claude-opus-4-7, and kimi-k2.6, and the pooled winner distribution differs strongly from an equal-strength null (p approx 1.5 x 10^-5). We then separate planning from execution by standardizing execution on a cheaper Gemini Flash scaffold. Under this design, a pooled 32-game planner bakeoff is consistent with near-equality (p approx 0.821), which indicates that much of the earlier provider spread came from end-to-end system behavior rather than planning alone. To study mechanism, we analyze saved planning and execution traces from the provider championship. Gemini refers to the terminal objective far more often than the other models and increases that focus as victory approaches. Gemini also converts more turns into deep conquest chains, even though it is not the cleanest runtime. These results show that live-agent performance depends on objective tracking, execution conversion, cost, and runtime reliability, and they support evaluating LLMs as components in bounded workflows rather than as isolated benchmark respondents. 

---
# Unlocking Proactivity in Task-Oriented Dialogue 

**Authors**: Hongbin Zhang, Ning Gao, Yuqin Dai, Ruiyuan Wu, Jinpeng Wang, Rena Wei Gao, Bingdong Tan, Shuzheng Gao, Zongjie Li, Chaozheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22240)  

**Abstract**: Proactive task-oriented dialogue (TOD), such as outbound sales, demands a persuasive agent that actively probes the user's concerns and steers the conversation toward acceptance within a bounded number of turns. Yet post-trained LLMs are inherently conservative, and reward-shaping RL (e.g., GRPO) struggles since it only re-weights what an already passive policy samples. We show that conditioning on the user's latent concerns unlocks proactive capability that no amount of sampling can undermine, establishing these concerns as a pivotal training-time signal. To operationalize this finding, we build the \textbf{Cognitive User Simulator}, which models each user as a stratified persona comprising observable external traits and hidden internal concerns. The simulator produces faithful and diverse interactions, while emitting per-turn state dynamics that track persuasion progress. We then introduce \textbf{Simulator-Induced Asymmetric-View Policy Optimization}, which converts the modeled concerns and the simulation state transition into complementary training objectives: (1) \emph{Asymmetric On-Policy Self-Distillation} that transfers concern-aware behavior from a privileged view of the same policy into its deployable, conversation-only view; and (2) \emph{State-Transition Policy Refinement} ... 

---
# Claw AI Lab: An Autonomous Multi-Agent Research Team 

**Authors**: Fan Wu, Cheng Chen, Zhenshan Tan, Taiyu Zhang, Xinzhen Xu, Yanyu Qian, Dingcheng Gao, Lanyun Zhu, Qi Zhu, Yi Tan, Deyi Ji, Guosheng Lin, Tianrun Chen, Deheng Ye, Fayao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22662)  

**Abstract**: We present Claw AI Lab, a lab-native autonomous research platform that advances automated research from a hidden prompt-to-paper pipeline into an interactive AI laboratory. Rather than centering the system around a single agent or a fixed serial workflow, we allow users to instantiate a full research team from one prompt, with customizable roles, collaborative workflows, real-time monitoring, artifact inspection, and rollback/resume control through a unified dashboard. The platform also supports distinct research modes for exploration, multi-agent discussion, and reproduction, making autonomous research substantially more steerable and laboratory-like in practice. A key practical contribution of Claw AI Lab lies in its Claw-Code Harness, which connects local codebases, datasets, and checkpoints to runnable experiments and feeds execution artifacts back into the research loop. As a result, the harness improves not only execution integration, but also experimental completion and result integrity: experiments are easier to inspect, iterate on, and faithfully transfer into final papers, reducing common failure modes such as partial runs and malformed result reporting. In our internal evaluation on five AI research case studies, using AutoResearchClaw as the baseline, Claw AI Lab is consistently preferred by AI expert judges on idea novelty, experiment completeness, and paper presentation quality. We view Claw AI Lab as an early step toward a new paradigm: autonomous research as usable, interactive, and reliability-aware scientific infrastructure. 

---
# Towards Direct Evaluation of Harness Optimizers via Priority Ranking 

**Authors**: Kai Tzu-iunn Ong, Minseok Kang, Dongwook Choi, Junhee Cho, Seungju Kim, Seungwon Lim, Geunha Jang, Minwoo Oh, Bogyung Jeong, Sunghwan Kim, Taeyoon Kwon, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2605.22505)  

**Abstract**: Harness optimization enables automated agent creation by having an optimizer agent iteratively update the harness of target agents. Despite its success, current studies evaluate optimizers solely by observing target agents' performance gains. This indirect end-improvement evaluation neglects optimizers' actions at intermediate steps, which are often erroneous and hinder agent performance. Therefore, it is unclear whether harness optimization is driven by optimizers' informed update actions or simply trial-and-error. This necessitates direct evaluation of harness optimizers. However, evaluating harness optimizers directly is non-trivial and costly due to the lack of oracle harnesses. To address this, we present a simple, low-cost design to directly evaluate them, namely priority ranking. By asking harness optimizers to rank components (e.g., tools) in a given harness by their potential to improve/hinder agent performance when updated, our design quantifies optimizer ability at the step level without expensive rollouts or manual examination. More importantly, optimizers' ranking performance correlates with their ability to improve agents in actual multi-step harness optimization, establishing priority ranking as a reliable predictor of optimization ability. Priority ranking is enabled by Shor, a collection of 182 human-verified optimization scenarios spanning across domains, designs, and time stages. Codes and data can be found at this https URL. 

---
# CLORE: Content-Level Optimization for Reasoning Efficiency 

**Authors**: Yuyang Wu, Qiyao Xue, Guanxing Lu, Weichen Liu, Zihan Wang, Manling Li, Olexandr Isayev  

**Link**: [PDF](https://arxiv.org/pdf/2605.22211)  

**Abstract**: Reinforcement learning post-training has improved the reasoning ability of large language models, but often produces unnecessarily long, repetitive, or semantically opaque reasoning traces. Existing efficient reasoning methods mainly regulate response length through explicit budgets or length-aware rewards, leaving intermediate reasoning content weakly supervised. We propose CLORE, a content-level optimization framework that improves reasoning efficiency by editing correct on-policy rollouts. CLORE uses an external augmentation model to delete repetitive segments, illegible or task-irrelevant content, and superfluous reasoning after the solution is established, while preserving the final answer. The resulting augmented--original pairs are optimized with an auxiliary reference-free DPO objective alongside standard policy-gradient training. By restricting augmentation to correct trajectories and performing local deletion, CLORE keeps edited rollouts close to the policy distribution and mitigates off-policy mismatch. Experiments on DeepSeek-R1-Distill-Qwen-7B and Qwen2.5-Math-7B across five mathematical reasoning benchmarks show that CLORE improves the accuracy--efficiency trade-off and remains compatible with GRPO, DAPO, Training Efficient, and ThinkPrune. Content-level analyses further show that CLORE reduces repetitive reasoning, illegible content, and post-answer exploration, supporting content-level supervision as a complementary direction to length-level control. 

---
# Cross-domain benchmarks reveal when coordinated AI agents improve scientific inference from partial evidence 

**Authors**: Fiona Y. Wong, Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2605.22300)  

**Abstract**: Scientific evidence often spans instruments, databases, and disciplines, so no single source records the full phenomenon. This makes it difficult to determine when coordinated AI agents add value over simpler scientific workflows. We evaluate this question with a cross-domain benchmark spanning four scientific tasks: mapping molecular structure into musical representations, detecting historical paradigm shifts in science, identifying vector-borne disease emergence, and vetting transiting-exoplanet candidates. Each case uses a frozen evaluation panel, predefined scoring protocols, explicit baselines, ablations or null controls, and stated limitations. The results define three operating regimes. When different disciplines each capture only part of the phenomenon, cross-channel composites improve over single-channel baselines: climate-vector emergence reaches AUROC 0.944 and exoplanet vetting reaches AUROC 0.955. However, the exoplanet workflow is effectively tied with a strong combined-summary baseline, showing that decomposition does not always improve top-line performance. When one signal dominates, as in paradigm-shift detection, coordination mainly improves interpretation and traceability. For molecular sonification, the gain is representational rather than predictive. ScienceClaw x Infinite provides the auditable artifact and provenance layer for this evaluation. The benchmark therefore assigns value to coordination only when the corresponding performance, provenance, or representation claim is supported by explicit comparators. 

---
# ST-SimDiff: Balancing Spatiotemporal Similarity and Difference for Efficient Video Understanding with MLLMs 

**Authors**: Bingjun Luo, Tony Wang, Chaoqi Chen, Xinpeng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.22158)  

**Abstract**: Multimodal Large Language Models (MLLMs) face significant computational overhead when processing long videos due to the massive number of visual tokens required. To improve efficiency, existing methods primarily reduce redundancy by pruning or merging tokens based on importance or similarity. However, these approaches largely overlook a critical dimension of video content, i.e., changes and turning points, and they lack a collaborative model for spatio-temporal relationships. To address this, we propose a new perspective: similarity is for identifying redundancy, while difference is for capturing key events. Based on this, we designed a training-free framework named ST-SimDiff. We first construct a spatio-temporal graph from the visual tokens to uniformly model their complex associations. Subsequently, we employ a parallel dual-selection strategy: 1) similarity-based selection uses community detection to retain representative tokens, compressing static information; 2) temporal difference-based selection precisely locates content-changing points to preserve tokens that capture key dynamic shifts. This allows it to preserve both static and dynamic content with a minimal number of tokens. Extensive experiments show our method significantly outperforms state-of-the-art approaches while substantially reducing computational costs. Our code is available in this https URL. 

---
# Skill Weaving: Efficient LLM Improvement via Modular Skillpacks 

**Authors**: Zhuo Li, Guodong Du, Zesheng Shi, Weiyang Guo, Weijun Yao, Yuan Zhou, Jiabo Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.22205)  

**Abstract**: Large language models increasingly require specialization across diverse domains, yet existing approaches struggle to balance multi-domain capacities with strict memory and inference constraints. In this work, we introduce SkillWeave, a modular improvement framework that enables LLMs to specialize under fixed memory budgets. SkillWeave partitions full capabilities of a general-purpose model into skillpacks -- lightweight, domain-specific delta modules -- that reorganize and refine the model's internal knowledge. For efficient deployment, SkillWeave integrates SkillZip to compress skillpacks into compact and inference-ready format, enabling strong multi-domain performance with low-latency execution. On multi-task and agentic benchmarks, a 9B SkillWeave model outperforms several baselines and even surpasses a 32B monolithic LLM, while achieving up to 4x speedup. 

---
# Adapting the Interface, Not the Model: Runtime Harness Adaptation for Deterministic LLM Agents 

**Authors**: Tianshi Xu, Huifeng Wen, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.22166)  

**Abstract**: LLM agents are shaped not only by their language models, but also by the runtime harness that mediates observation, tool use, action execution, feedback interpretation, and trajectory control. While existing agent adaptation methods mainly update model parameters, many failures in deterministic, rule-governed domains stem from mismatches at the model--environment interface. We propose Life-Harness, a lifecycle-aware runtime harness that improves frozen LLM agents without changing model weights or evaluation environments. Life-Harness evolves from training trajectories by converting recurring interaction failures into reusable interventions across environment contracts, procedural skills, action realization, and trajectory regulation, and remains fixed during held-out evaluation. On seven deterministic environments from $\tau$-bench, $\tau^2$-bench, and AgentBench, Life-Harness improves 116 out of 126 model--environment settings across 18 model backbones, with an average relative improvement of 88.5%. Harnesses evolved only from Qwen3-4B-Instruct trajectories transfer to 17 other models, showing that Life-Harness captures reusable environment-side structure rather than model-specific behavior. These results position runtime interface adaptation as a complementary alternative to model-centric agent training. Code is available at GitHub. 

---
# Compiling Agentic Workflows into LLM Weights: Near-Frontier Quality at Two Orders of Magnitude Less Cost 

**Authors**: Simon Dennis, Rivaan Patil, Kevin Shabahang, Hao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.22502)  

**Abstract**: Agent orchestration frameworks have proliferated, collectively exceeding 290,000 GitHub stars across LangGraph, CrewAI, Google ADK, OpenAI Agents SDK, Semantic Kernel, Strands, and LlamaIndex. All follow the same pattern: an external orchestrator above the LLM, injecting instructions and routing decisions every turn. Recent work has shown this architecture is dominated for procedural tasks by simply providing the procedure in a frontier model's system prompt [Dennis et al., 2026a], at the cost of consuming the context window, requiring a frontier model for every conversation, and exposing proprietary procedures to third-party providers. Compiling the procedure into the weights of a small fine-tuned model -- creating a subterranean agent -- should resolve all of these concerns, and prior work (SimpleTOD, FireAct, SynTOD, WorkflowLLM, Agent Lumos) has shown the technique works. Yet developer adoption has overwhelmingly favored orchestration. We identify three perceived barriers and address each empirically across travel booking (14 nodes), Zoom support (14 nodes, product-specific knowledge), and insurance claims (55 nodes, 6 decision hubs). 

---
# ArborKV: Structure-Aware KV Cache Management for Scaling Tree-based LLM Reasoning 

**Authors**: Yeqiu Chen, Ziyan Liu, Zhenxin Huang, Runquan Gui, Hong Wang, Lei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22106)  

**Abstract**: Recent progress in LLM reasoning has increasingly shifted from single-pass generation to explicit search over intermediate reasoning states. Tree-of-Thoughts (ToT) organizes inference to tree-structured search with branching and backtracking, but it substantially amplifies the Key--Value (KV) cache: retaining KV states for a frontier of partial trajectories quickly becomes a memory bottleneck that limits throughput and constrains search depth and width under fixed hardware budgets. We address this challenge by observing that KV reuse in ToT-style inference is governed by search dynamics: near-term decoding depends primarily on the active branch and its ancestors, whereas inactive subtrees have low short-term reuse probability yet must remain recoverable for backtracking. Motivated by this, we propose ArborKV, a structure-aware eviction framework that couples a lightweight value estimator with a tree-aware allocation policy, and performs purely token-extractive eviction with lazy rehydration to support revisits. Experiments on ToT-style reasoning benchmarks show that ArborKV achieves up to ~4x peak KV-memory reduction while preserving near-full-retention accuracy, enabling larger search configurations under fixed device budgets that would otherwise run out of memory. 

---
# Perception or Prejudice: Can MLLMs Go Beyond First Impressions of Personality? 

**Authors**: Caixin Kang, Tianyu Yan, Sitong Gong, Mingfang Zhang, Liangyang Ouyang, Ruicong Liu, Bo Zheng, Huchuan Lu, Kaipeng Zhang, Yoichi Sato, Yifei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22109)  

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly deployed in human-facing roles where personality perception is critical, yet existing benchmarks evaluate this capability solely on numerical Big Five score prediction, leaving open whether models truly perceive personality through behavioral understanding or merely prejudge through superficial pattern matching. We address this gap with three contributions. (i) A new task: we formalize Grounded Personality Reasoning (GPR), which requires MLLMs to anchor each Big Five rating in observable evidence through a chain of rating, reasoning, and grounding. (ii) A new dataset: we release MM-OCEAN (1,104 videos, 5,320 MCQs), produced by a multi-agent pipeline with human verification, with timestamped behavioral observations, evidence-grounded trait analyses, and seven categories of cue-grounding MCQs. (iii) Benchmark and analysis: we design a three-tier evaluation (rating, reasoning, grounding) plus four sample-level failure-mode metrics: Prejudice Rate (PR), Confabulation Rate (CR), Integration-failure Rate (IR), and Holistic-grounding Rate (HR), and benchmark 27 MLLMs (13 closed, 14 open). The analysis uncovers a striking Prejudice Gap: across the field, 51% of correct ratings are not grounded in retrieved cues, and the Holistic-Grounding Rate spans only 0-33.5%. These findings expose a disconnect between getting the right score and reasoning for the right reason, charting a roadmap for grounded social cognition in MLLMs. 

---
# LLM-Metrics: Measuring Research Impact Through Large Language Model Memory 

**Authors**: Si Shen, Wenhua Zhao, Danhao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22176)  

**Abstract**: Citation counts remain the dominant metric for assessing research impact, yet they suffer from well-documented limitations: temporal lag, disciplinary bias, and Matthew effects. Here we propose LLM-Metrics, a research-impact assessment metric derived from the parametric memory of large language models (LLMs). The central hypothesis is that high-impact papers receive greater exposure in the academic community, that this exposure enters LLM training data in textual form, and that models consequently form stronger parametric memory of these papers. We designed four types of multiple-choice probes, covering title recognition, author recognition, method recognition, and venue recognition, and evaluated 549 computer science papers published in 2023-2024 across 17 LLMs spanning 0.5B to 72B parameters from six vendors. Of the 17 models, 15 produced positive predictions, 9 of which were significant at p less than 0.05, with an overall Spearman correlation of rho = 0.1495 and p = 0.0004 against citation counts. Three additional findings support the proposed mechanism. First, the predictive signal was stronger for 2024 papers, rho = 0.1880, whose citation counts were near zero at model-training time, reducing the plausibility of a simple reverse-causality explanation. Second, author-recognition probes showed the strongest discriminative power, consistent with an exposure-driven memory mechanism. Third, model scale and predictive power were non-monotonic: a 3B-parameter model, Llama-3.2-3B-Instruct, with rho = 0.1829, outperformed most larger models, supporting a selective-memory hypothesis in which the limited capacity of smaller models can serve as an effective information filter. LLM-Metrics offers a real-time, cross-disciplinary, citation-independent paradigm for research assessment. 

---
# IdleSpec: Exploiting Idle Time via Speculative Planning for LLM Agents 

**Authors**: Daewon Choi, Kyunghyun Park, Woomin Song, Saket Dingliwal, Sai Muralidhar Jayanthi, Jinwoo Shin, Aram Galstyan  

**Link**: [PDF](https://arxiv.org/pdf/2605.22154)  

**Abstract**: Large language model (LLM)-based agents solve complex tasks by leveraging multi-step reasoning with iterative tool calls and environment interactions, which incur idle time while waiting for observations. Despite the prevalence of idle time in most agentic scenarios, existing works treat it as an unavoidable overhead or propose restricted solutions that overlook varying computational budgets across different tool calls and future observation uncertainty, thereby leading to suboptimal utilization of idle time. In this paper, we introduce IdleSpec, a scalable and generic inference approach that leverages idle-time computation to improve agent performance while minimizing latency overhead. Specifically, IdleSpec iteratively generates plan candidates during idle periods and, once observations become available, aggregates them to guide the next reasoning step. For effective plan generation under observation uncertainty, IdleSpec samples between complementary drafting strategies (i.e., progressive and recovery) from a learned distribution that is updated via posterior feedback. Our experiments demonstrate that IdleSpec significantly improves agent performance in various agentic scenarios by effectively utilizing idle time. In particular, on the GAIA and FRAMES, IdleSpec achieves 55.6% average accuracy with Gemini-2.5-Flash, surpassing the vanilla baseline without idle-time usage by 5.1%. Furthermore, for MLE-Bench, which involves substantial delay from code executions, IdleSpec achieves performance gains of up to 9.1% on the Any Medal rate, highlighting its generalizability to long-horizon tasks. 

---
# Active Evidence-Seeking and Diagnostic Reasoning in Large Language Models for Clinical Decision Support 

**Authors**: Chen Zhan, Xihe Qiu, Xiaoyu Tan, Xibing Zhuang, Gengchen Ma, Yue Zhang, Shuo Li, Peifeng Liu, Xiaoxiao Ge, Liang Liu, Lu Gan  

**Link**: [PDF](https://arxiv.org/pdf/2605.22047)  

**Abstract**: Large language models perform well on static medical examinations, yet clinical diagnosis often requires iterative evidence gathering under uncertainty. Building on prior interactive evaluation efforts, we introduce an OSCE-inspired standardized patient simulator and a controlled, reproducible benchmark for active diagnostic inquiry. Across 468 cases and 15 models in our protocol, we observe that multi-turn evidence seeking reduces diagnostic accuracy by 12.75% and lowers supporting-evidence quality by 24.36% relative to full-context evaluation; error analyses associate these drops with premature diagnostic closure and inefficient questioning. Together, these results suggest that static full-context benchmarks may overestimate performance in interactive evidence-seeking settings, motivating complementary interactive assessment for safer clinical decision support. 

---
# Enhancing Visual Token Representations for Video Large Language Models via Training-Free Spatial-Temporal Pooling and Gridding 

**Authors**: Bingjun Luo, Tony Wang, Hanqi Chen, Xinpeng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.22078)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have significantly advanced video understanding tasks, yet challenges remain in efficiently compressing visual tokens while preserving spatiotemporal interactions. Existing methods, such as LLaVA family, utilize simplistic pooling or interpolation techniques that overlook the intricate dynamics of visual tokens. To bridge this gap, we propose ST-GridPool, a novel training-free visual token enhancement method designed specifically for Video LLMs. Our approach integrates Pyramid Temporal Gridding (PTG), which captures multi-grained spatiotemporal interactions through hierarchical temporal gridding, and Norm-based Spatial Pooling (NSP), which preserves high-information visual regions by leveraging the correlation between token norms and semantic richness. Extensive experiments on various benchmarks demonstrate that ST-GridPool consistently enhances performance of Video LLMs without requiring costly retraining. Our method offers an efficient and plug-and-play solution for improving visual token representations. Our code is available in this https URL. 

---
# ExComm: Exploration-Stage Communication for Error-Resilient Agentic Test-Time Scaling 

**Authors**: Woomin Song, Beomjun Kim, Daewon Choi, Sai Muralidhar Jayanthi, Saket Dingliwal, Jinwoo Shin, Aram Galstyan  

**Link**: [PDF](https://arxiv.org/pdf/2605.22102)  

**Abstract**: A common failure mode in long-horizon agentic test-time scaling is error propagation, where factual errors or invalid deductions introduced at intermediate steps persist in the agent's belief state and contaminate later reasoning. Existing test-time scaling methods provide limited control over this process, as they often rely on agents to detect their own mistakes, select among flawed trajectories, or refine solutions only after errors have already shaped the reasoning path. We propose ExComm, a communication protocol for exploration-stage agentic test-time scaling. ExComm is motivated by the empirical observation that the majority of intermediate errors in parallel agentic reasoning produce detectable cross-agent factual conflicts. Leveraging the iterative structure of agentic workflows, ExComm periodically audits agent belief states to detect such conflicts, resolves them through a dedicated tool-based verification loop, and returns concise, targeted feedback to the involved agents. Corrections are incorporated through soft belief updates, which append verified feedback rather than overwriting existing beliefs. Furthermore, to prevent collapsing trajectory diversity due to communication, ExComm further introduces a trajectory diversification module that redirects redundant trajectories toward orthogonal strategies. Experiments on AIME 2024, AIME 2025, and GAIA with Gemini-2.5-Flash-Lite and Qwen3.5-4B show that ExComm consistently outperforms strong test-time scaling baselines, achieving average performance gains of 5.7% and 5.0% over the best-performing baselines, respectively. Further analyses demonstrate improved error recovery, favorable scaling behavior, stronger diversity than adapted communication baselines, and the best performance-cost trade-off among the evaluated methods. 

---
# The Log is the Agent: Event-Sourced Reactive Graphs for Auditable, Forkable Agentic Systems 

**Authors**: Yohei Nakajima  

**Link**: [PDF](https://arxiv.org/pdf/2605.21997)  

**Abstract**: Most agent frameworks are built around the language model: a conversation loop comes first, then tools, then rules, and finally a logging layer bolted on for observability, with state persisted as retrievable "memory." We describe ActiveGraph, a runtime that inverts this arrangement. The append-only event log is the source of truth; the working graph is a deterministic projection of that log; and behaviors--ordinary functions, classes, LLM-backed routines, or logic attached to typed edges--react to changes in the graph and emit new events. No component instructs another; coordination happens entirely through the shared graph. This single design decision yields three properties that retrieval-and-summarization memory systems do not provide: deterministic replay of any run from its log, cheap forking that branches a run at any event without re-executing the shared prefix, and end-to-end lineage from a high-level goal down to the individual model call that produced each artifact. We present the architecture, a determinism contract that makes replay sound, and a worked diligence example whose full causal structure is reconstructable from the log alone. We discuss--without claiming to demonstrate--why this substrate is unusually well suited to self-improving agents, and how it extends the BabyAGI lineage and prior graph-memory research. 

---
# ECPO: Evidence-Coupled Policy Optimization for Evidence-Certified Candidate Ranking 

**Authors**: Miaobo Hu, Shuhao Hu, BoKun Wang, Yina Sa, Xin Wang, Xiaobo Guo, Daren Zha, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.21993)  

**Abstract**: Ranking systems used in decision-support settings should not only order candidates but also expose evidence that can be independently checked. We study evidence-certified candidate ranking: given an intent_id, a predefined plan skeleton, a window-local candidate roster, and text-derived candidate trajectories with span provenance, a system must output a Top-K list together with doc_id:span evidence certificates whose cited spans are sufficient to recover the decision. We instantiate this task on MAVEN-ERE and RAMS with fixed upstream extraction, window-local randomized candidate identifiers, skeleton-aligned trajectory supervision, hard negatives, and audit references. We introduce Evidence-Coupled Policy Optimization (ECPO), a listwise policy-optimization objective whose action is the joint object of ranking and evidence certificate. ECPO first learns an interpretable trajectory reward from skeleton alignment, argument consistency, and optional graph features; it then optimizes a constrained policy with three coupled rewards: listwise ranking utility, span-level certificate validity, and an evidence-cycle reward computed by a label-free deterministic verifier that reconstructs candidate support from claim-stripped cited spans. This reframes the goal from maximizing ordinary NDCG alone to maximizing CertNDCG and decision-evidence coupling. The evaluation compares ECPO against zero-shot, SFT, and GRPO policies, RM-only scoring with deterministic evidence attachment, grammar/JSON-constrained decoding, validator retry, best-of-N RM selection, and post-hoc evidence rationalization under closed-roster, predicted-roster, and hybrid-roster settings. 

---
# Format-Constraint Coupling in Knowledge Graph Construction from Statistical Tables 

**Authors**: Jingxuan Qi, Zhiqiang Ye, Yuxiang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.21974)  

**Abstract**: An extraction schema should not reduce knowledge graph fidelity. On statistical CSV, however, it can. We study country-by-year time-series matrices, a common layout on open-data portals. In this setting, serialization format and schema constraints interact super-additively. Their joint effect exceeds the sum of independent effects by up to +1.180 (2x2 factorial, 6 datasets). Bootstrap 95% CIs are strictly positive on 4/6 datasets, with strongest evidence on wide Type-II matrices. More critically, a schema applied to a mismatched format can trigger catastrophic mismatch. Fact coverage falls below the unconstrained baseline on 4/6 datasets through entity inflation or extraction refusal. We call this observed pattern format-constraint coupling. Probing and token ablation support a surface-form anchoring explanation centred on column-name references. Controlled variants across format-schema pairings, GraphRAG hosts, and LLM families show the same direction within the measured scope; one LLM family shows only partial activation. The observation also has a diagnostic consequence. Three standard retrieval modes largely mask construction quality (delta <= 1pp), whereas direct graph access exposes gaps up to +47.6pp (p < 0.0001). To support fidelity-aware evaluation, we release CSVFidelity-Bench. It contains 15 datasets, 11 Type-II matrices, 4 Type-III tables, and 1,892 Gold Standard facts across 6 domains. 

---
# AI-Enabled Serious Games: Integrating Intelligence and Adaptivity in Training Systems 

**Authors**: Priyamvada Tripathi, Bill Kapralos  

**Link**: [PDF](https://arxiv.org/pdf/2605.21962)  

**Abstract**: Serious games are widely used for learning and training across domains such as healthcare, defense, and education. Persistent challenges remain, however, including static scenario design, authoring bottlenecks, limited learner modeling, and difficulty implementing meaningful real-time instructional adaptation. Recent advances in artificial intelligence (AI) introduce novel capabilities such as dynamic scenario variation, contextual feedback, adaptive pacing, and learner-state modeling that may help address some of these limitations. At the same time, integrating AI into serious games raises important questions related to validity, transparency, system control, and learner trust. This chapter examines how contemporary AI approaches may support real-time instructional adaptation in serious games. It distinguishes between instructional intelligence, defined as a system's capacity to infer learner knowledge and reason about pedagogically appropriate responses, and adaptivity, defined as the ability to modify instructional actions during interaction. A historical synthesis of adaptive learning systems is presented, tracing developments from early computer-assisted instruction through intelligent tutoring systems (ITS), dynamic difficulty adjustment (DDA), authoring platforms, learning analytics, and recent AI-enabled architectures. Building on this perspective, the chapter discusses how large language models (LLMs), reinforcement learning (RL), and agent-based architectures may contribute to more integrated forms of intelligence and adaptivity in serious games. It also highlights practical and research challenges associated with AI-enabled systems, including explainability, validation, computational cost, and the limited empirical evidence regarding long-term learning outcomes in AI-enabled serious games. 

---
# Implicit Safety Alignment from Crowd Preferences 

**Authors**: Qian Lin, Daniel S. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2605.21822)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) can reveal implicit objectives such as safety considerations that go beyond task completion. In this work, we focus on the common safety criteria embedded in crowd preference datasets, where different users may express distinct preferences or objectives, yet follow similar safety principles. Our aim is to discover shared safety criteria from crowd preferences and then transfer them to downstream RL tasks to regularize agent behavior and enforce safety. We first show that direct reward combination-optimizing a preference-learned reward model together with downstream task rewards-has inherent limitations. Motivated by this, we propose Safe Crowd Preference-based RL, a hierarchical framework that extracts safety-aligned skills from crowd preferences and composes them via a high-level policy to safely solve downstream tasks. Experiments across safe RL environments and a preliminary LLM-style task with diverse user goals and shared safety constraints demonstrate that our approach substantially lowers safety costs without access to explicit safety rewards, while achieving task performance comparable to oracle methods trained with ground-truth safety signals. 

---
# Latent-space Attacks for Refusal Evasion in Language Models 

**Authors**: Giorgio Piras, Raffaele Mura, Fabio Brau, Maura Pintor, Luca Oneto, Fabio Roli, Battista Biggio  

**Link**: [PDF](https://arxiv.org/pdf/2605.21706)  

**Abstract**: Safety-aligned language models are trained to refuse harmful requests, yet refusal behavior can be suppressed by steering their internal representations. Existing methods do so by ablating a refusal direction from model activations, aiming to remove refusal from the model's residual stream. Despite their empirical success, these methods lack a principled account of the latent-space transformation they induce and why it suppresses refusal. In this work, we recast refusal suppression as a latent-space evasion attack against linear probes trained to separate refused from answered prompts. Under this view, prior work's difference-in-means direction naturally defines such a probe, and its ablation is exactly a projection onto its decision boundary, i.e., a minimum-confidence evasion attack. This perspective not only explains the empirical success of prior work but also admits a key limitation: evasion stops at the decision boundary, motivating the need to push representations further into the compliant region, i.e., where the model answers. We leverage this by proposing a Controlled Latent-space Evasion attack that projects representations past the boundary with an optimized confidence. We achieve state-of-the-art attack success rate across 15 instruction-tuned, multimodal, and reasoning models, outperforming existing refusal-ablation baselines and specialized jailbreak attacks. 

---
# What Counts as AI Sycophancy? A Taxonomy and Expert Survey of a Fragmented Construct 

**Authors**: Meryl Ye, Lujain Ibrahim, Jessica Y. Bo, Myra Cheng, Ida Mattsson, Daniel Vennemeyer, Robert Kraut, Steve Rathje  

**Link**: [PDF](https://arxiv.org/pdf/2605.21778)  

**Abstract**: AI sycophancy has become a prominent concern in large language model (LLM) research. Yet the term lacks a consistent definition and has been applied to behaviors ranging from agreeing with a user's false claim to excessively praising the user to withholding corrective feedback. When researchers, companies, and policymakers use the same term to describe different behaviors, evaluation results become difficult to compare, mitigation strategies fail to transfer, and systems that are resistant to one form of sycophancy continue exhibiting other forms. To address this, we make two contributions. First, we reviewed 70 papers on AI sycophancy to develop a taxonomy of how the behavior has been defined and measured. The taxonomy distinguishes (1) whether a model is sycophantic toward a user's positions and beliefs, or toward the user's broader personal traits and emotions, and (2) whether this occurs through explicit, direct language or more implicit, subtle behaviors such as framing, omission, or tone. Mapping existing literature to our taxonomy reveals that current research has focused on overt forms of sycophancy toward users' beliefs, leaving more subtle and person-directed behaviors relatively understudied. Second, we surveyed 106 experts in AI sycophancy and related fields to examine whether researchers agree on which model behaviors are sycophantic. While experts are nearly unanimous in believing that sycophancy is a significant problem in current AI systems (94.3% agree), they disagree substantially on which specific behaviors qualify. Together, these findings demonstrate that AI sycophancy is a broad family of behaviors with different measurement challenges, intervention requirements, and governance implications. Our taxonomy provides a shared vocabulary for understanding and addressing these behaviors. 

---
# SMDD-Bench: Can LLMs Solve Real-World Small Molecule Drug Design Tasks? 

**Authors**: Kevin Han, Renfei Zhang, Kathy Wei, Hamed Mahdavi, Niloofar Mireshghallah, Amir Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2605.21740)  

**Abstract**: LLM agents have incredible potential for scientific discovery applications. However, the performance of LLM agents on real-world, small molecule drug design (SMDD) tasks across diverse chemistries and targets is unclear. Current evaluation methods are either ad hoc, too simple for real-world discovery, limited in scale, or restricted to single-turn question answering. In effort to standardize the evaluation of LLM agents on small molecule design, we introduce SMDD-Bench, a challenging, multi-turn, long-horizon agentic benchmark consisting of 502 guaranteed-solvable task instances spanning 5 task types: 2D Pharmacophore Identification, Interaction Point Discovery, Scaffold Hopping, Lead Optimization, and Fragment Assembly. SMDD-Bench tasks span a wide region of chemical space and involve 102 unique protein targets. Completely solving the benchmark would require having strong chemical and biological reasoning and 3D intuition, understanding specialized tool use, and displaying planning expertise over a limited number of oracle calls. We benchmark 7 frontier open and closed source LLMs and find even the most performant LLM, GPT5.4, solves only 40.2\% of tasks. We hope SMDD-Bench provides a standardized testbed to invigorate the field towards training and evaluating LLM agents for fully autonomous computational drug design. We host a public leaderboard at this http URL . 

---
# MindLoom: Composing Thought Modes for Frontier-Level Reasoning Data Synthesis 

**Authors**: Haiyang Shen, Taian Guo, Xuanzhong Chen, Mugeng Liu, Weichen Bi, Wenchun Jing, Sixiong Xie, Zhuofan Shi, Yudong Han, Chongyang Pan, Siqi Zhong, Jinsheng Huang, Ming Zhang, Yun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.21630)  

**Abstract**: Although LLMs have made substantial progress in reasoning, systematically producing frontier-level reasoning data remains difficult. Existing synthesis methods often have limited visibility into the structural factors that govern problem difficulty, which can result in narrow diversity and unstable difficulty control. In this work, we view the difficulty of a reasoning problem as arising from the accumulation of atomic knowledge-reasoning transformations, which we term thought modes. Building on this perspective, we propose MindLoom, a framework for synthesizing frontier-level reasoning data through compositional thought mode engineering. Given a collection of hard problems with verified solutions, MindLoom first decomposes those solutions into thought mode chains that reveal each problem's construction logic. It then trains a retrieval model that matches problem states to compatible thought modes, providing guidance on which reasoning challenges to introduce during synthesis. New problems are composed by iteratively applying retrieved thought modes to seed questions, with distribution-aligned sampling to encourage diverse reasoning coverage. Finally, a rollout-based judging stage labels generated questions by difficulty and supplies judged-correct responses for supervised fine-tuning. We evaluate MindLoom on nine benchmarks covering five STEM disciplines and four mathematical reasoning tasks across multiple model families and sizes. Models fine-tuned on MindLoom-generated data achieves favorable performances over base models, distillation, and external-data baselines across the reported benchmarks. Ablation studies indicate the contribution of each component, and further analysis suggests that MindLoom covers a broad range of reasoning patterns while maintaining useful difficulty control. We have open-sourced our implementation at this https URL. 

---
# AttuneBench: A Conversation-Based Benchmark for LLM Emotional Intelligence 

**Authors**: Kate M. Lubrano, Faisal Sayed, Ankita Rathod, Akshansh, Craver Corbyn Thomas-Smith, Mark E. Whiting, Karina Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.21739)  

**Abstract**: Emotional intelligence (EI), the ability to perceive, understand, and respond appropriately to others' emotional states, is central to human communication, and increasingly important to assess as LLMs assume conversational roles in everyday life. Existing EI benchmarks rely on synthetic prompts, single-turn cases, or third-party annotation. These approaches do not directly measure how models infer and respond to a participant's emotional state over the course of a real conversation. We introduce AttuneBench, a benchmark grounded in 200 genuine multi-turn human-model conversations in which participants conversed with anonymized LLMs and provided turn-by-turn annotations of their emotional state, the model's behavior, and their preferred responses. Across 11 evaluated models, we find that model rankings on emotion recognition, behavioral classification, preference prediction, and judged response quality are largely independent, indicating that emotionally intelligent behavior decomposes into separable capabilities. Preference alignment and response-quality judgments are substantially more model-discriminating than emotion-label accuracy. These results indicate that emotionally intelligent behavior requires predicting what kind of response a specific user wants in context, a distinction that aggregate scoring can obscure and that single-turn or synthetic formats cannot directly capture across turns. AttuneBench provides a framework for assessing each of these capabilities and for diagnosing model-specific strengths and failure modes in emotionally salient conversation. 

---
# The Shape of Testimony: A Scalable Framework for Oral History Archive Comparison 

**Authors**: Itamar Trainin, Renana Keydar, Amit Pinchevski  

**Link**: [PDF](https://arxiv.org/pdf/2605.21623)  

**Abstract**: Researchers in Holocaust studies have often distinguished between two styles of oral survivor testimony: the USC Shoah Foundation's interviews tend to follow a structured, interviewer-guided format, whereas the Yale Fortunoff Video Archive generally favors a more free-form, open-ended style. This distinction has influenced both scholarly research and the development of later archives. In this study, we critically examine that claim by conducting a large-scale computational analysis of more than 1,600 testimonies from both collections. Leveraging discourse segmentation, topic modeling, and large language model (LLM) based analysis, we quantify the "structuredness" level of testimonies through topic coherence, interviewer-survivor dynamics, and the distribution of question types. Our results generally corroborate the structural differences identified in earlier research, while also revealing significant overlaps between the collections, both within individual interviews and across common narrative patterns. This complicates the simple "structured vs. free-form" dichotomy often applied to these oral histories. Beyond revisiting a foundational claim in Holocaust studies, our work provides a scalable, replicable framework for comparative corpus analysis. As a proof of concept, it suggests broader applications for digital oral history, narrative analysis, and the design of citizen-science annotation platforms. 

---
# Benchmarking and Improving Monitors for Out-Of-Distribution Alignment Failure in LLMs 

**Authors**: Dylan Feng, Pragya Srivastava, Cassidy Laidlaw  

**Link**: [PDF](https://arxiv.org/pdf/2605.21602)  

**Abstract**: Many safety and alignment failures of large language models (LLMs) occur due to out-of-distribution (OOD) situations: unusual prompt or response patterns that are unforeseen by model developers. We systematically study whether LLM monitoring pipelines can detect these OOD alignment failures by introducing a benchmark called Misalignment Out Of Distribution (MOOD). It is difficult to find failures that are truly OOD for off-the-shelf models trained on vast safety datasets. We sidestep this by including a restricted training set in MOOD that we use to train our own monitors, as well as seven test sets with diverse alignment failures that are outside the training distribution. Using MOOD, we find that guard models (safety classifiers) often fail to generalize OOD. To fix this, we propose combining guard models with OOD detectors. We test four types of OOD detectors and find that a combination of a guard model with Mahalanobis distance and perplexity-based OOD detectors can improve recall from 39% to 45%. We also establish positive scaling trends across model scales for monitors that combine a guard model and OOD detector; we find that incorporating OOD detection into monitoring achieves a higher recall gain than using a guard model with 20 times more parameters. Our work suggests that OOD detection should be a crucial component of LLM monitoring and provides a foundation for further work on this important problem. 

---
# TO-Agents: A Multi-Agent AI Pipeline for Preference-Guided Topology Optimization 

**Authors**: Isabella A. Stewart, Hongrui Chen, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2605.21622)  

**Abstract**: Topology optimization can generate efficient structures, but designers often must manually translate qualitative intent, such as desired visual style, product experience, or manufacturability into solver settings that are not directly tied to those preferences. We present TO-Agents, a multi-agent AI framework that connects natural-language design intent with iterative topology optimization. The framework converts a human-provided problem description into validated solver inputs, runs a topology optimization solver, renders the resulting 3D topology, and uses multi-view vision-language reasoning with an independent judge agent to critique each result and revise solver parameters. We evaluate the framework on two long-horizon design tasks: a cantilever beam benchmark and a phone-stand product design. In both tasks, the designer specifies an aesthetic preference for hierarchically branched structures inspired by natural tree morphologies, and the system performs four revision cycles across ten independent replicates. TO-Agents produces at least one preference-aligned design in 60% of trials for each case study, corresponding to up to 6x more successful trials than an ablated pipeline without visual or historical feedback. Judge scores and human evaluations show that the pipeline can identify effective parameter levers, recover from poor revisions, and expand design exploration. A manufacturing agent further post-processes top-ranked designs for additive manufacturing, enabling end-to-end intent-to-prototype design. We also identify failure modes, including overshooting, selective memory, misplaced tools, and incorrect parameter reasoning. These results suggest that agentic topology optimization can shift designers from low-level parameter tuning toward higher-level specification of form and function, while highlighting safeguards needed for reliable autonomous engineering design. 

---
# DeltaBox: Scaling Stateful AI Agents with Millisecond-Level Sandbox Checkpoint/Rollback 

**Authors**: Yunpeng Dong, Jingkai He, Yuze Hou, Dong Du, Zhonghu Xu, Si Yu, Yubin Xia, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.22781)  

**Abstract**: LLM-powered AI agents require high-frequency state exploration (e.g., test-time tree search and reinforcement learning), relying on rapid checkpoint and rollback (C/R) of the complete sandbox state, including files and process state (e.g., memory, contexts, etc.). Existing mechanisms duplicate the entire state, causing hundreds of milliseconds to seconds of latency per C/R, which severely bottlenecks deep search and large-scale fan-outs.
This paper observes that subsequent checkpoints in AI agents are highly similar. Therefore, instead of full duplication, a sandbox should only duplicate the changes between consecutive checkpoints (Key Insight). However, it is non-trivial to realize the idea, mainly due to the missing OS supports. This paper proposes a new OS-level abstraction, DeltaState, to enable the change-based transactional C/R for AI agents with two co-designed OS mechanisms. First, DeltaFS enables change-based filesystem C/R by organizing the file states into layers and dynamically freezing the writable layer and inserting a new one during checkpoint, reducing file updates to copy-on-write, and making rollback a simple layer switch. Second, DeltaCR enables change-based process state C/R using incremental dumps, and accelerates rollback by bypassing traditional pipelines to directly fork() from a frozen template process. We then present DeltaBox, a novel agent sandbox achieving millisecond level C/R through the two new mechanisms. Evaluations on SWE-bench and RL micro-benchmarks show DeltaBox completes checkpoint and rollback in millisecond-level latency (14ms and 5ms, respectively), empowering agents to explore substantially more nodes under fixed time budgets. 

---
# The Matching Principle: A Geometric Theory of Loss Functions for Nuisance-Robust Representation Learning 

**Authors**: Vishal Rajput  

**Link**: [PDF](https://arxiv.org/pdf/2605.22800)  

**Abstract**: Robustness, domain adaptation, photometric and occlusion invariance, compositional generalisation, temporal robustness, alignment safety, and classical anisotropic regularisation are usually treated as separate problems with separate method families. This paper argues that much of their shared structure is one statistical problem: estimate the covariance of label-preserving deployment nuisance, then regularise the encoder Jacobian along a matrix whose range covers that covariance (the matching principle). CORAL, adversarial training, IRM, augmentation, metric learning, Jacobian penalties, and alignment-style constraints are different estimators of that object, not independent robustness tricks.
In the linear-Gaussian model we prove closed-form optimality (Theorem A), including cube-root water-filling within the matched range; necessity of range coverage for quadratic Jacobian penalties (Theorem G); the same range dichotomy at deep global minima; and two falsification controls (Lemma C; Corollaries E), with seven conditional consistency lemmas (D1-D7) for estimation under standard identifiability assumptions.
We introduce the Trajectory Deviation Index (TDI), a label-free probe of embedding sensitivity when task accuracy or Jacobian Frobenius norm is insufficient.
Thirteen pre-registered blocks from classical ML through Qwen2.5-7B test the predicted matched, then isotropic, then wrong-W ordering on geometry and deployment drift; twelve pass, and the sole exception (Office-31) is an eigengap failure named before the run. At 7B scale, matched style-PMH improves selective honesty and preserves Style TDI where standard DPO degrades it.
The contribution is naming the deployment nuisance covariance, stating what the regulariser must do, and supplying a closed-form falsifiable theory once that object is identified, not universality on every leaderboard. 

---
# Toward AI VIS Co-Scientists: A General and End-to-End Agent Harness for Solving Complex Data Visualization Tasks 

**Authors**: Haichao Miao, Zhimin Li, Kuangshi Ai, Kaiyuan Tang, Chaoli Wang, Peer-Timo Bremer, Shusen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.21825)  

**Abstract**: The ability to inspect, interpret, and communicate complex data is crucial for virtually any scientific endeavor, but often requires significant expertise outside the core domain ranging from data management and analysis to visualization design and implementation. We present an end-to-end agentic harness that, based on only the data and a high level description of the tasks, independently designs custom visual analysis applications (VIS apps). This represents an important step towards a general AI co-scientist envisioned by many as an autonomous system that can autonomously execute long horizon tasks based on high-level directions. Our proposed VIS co-scientist is an essential component of this broader AI co-scientist vision: a harness that can autonomously analyze data and design visualization solutions using a collection of agents and specialized skills that coordinate exploratory analysis, plan, configure the environment, implement, validate the interface, and most importantly evaluate the overall task completion. Each stage produces document and instruction artifacts that guide downstream work and enable iterative refinement. We validate this approach on IEEE SciVis Contests spanning multiple science and engineering fields. These contests serve as ideal proving grounds because they encode real-world complexity: ambiguous requirements, diverse data modalities, design trade-offs, and task-driven validation. Given only the data and target tasks, our system autonomously produces functional single-page VIS Apps with verified linked-view behavior, highly customized to domain experts' specified tasks and needs. 

---
# Investigating Concept Alignment Using Implausible Category Members 

**Authors**: Sunayana Rane, Brenden M. Lake, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2605.21683)  

**Abstract**: Developing AI systems with a human-like understanding of everyday concepts is a key step towards developing safe, reliable systems whose behavior makes sense to humans. When probing concept understanding, asking questions about plausible category members (e.g., "Is a car a vehicle?") is likely to recall patterns in the model's vast training data. We pursue an alternative strategy, characterizing the boundaries of conceptual categories by asking about implausible category members (e.g., "Is an olive a vehicle?") to probe the kind of concept-level knowledge we take for granted in fellow humans. We characterize concept boundaries for a set of fundamental concepts by studying AI systems' assignments of objects to superordinate categories from a classic psychological study by Rosch and Mervis, as well as their assignments of the same objects to mismatched superordinate categories. We compare these assignments to those made by human participants on the full range of within-category and cross-category assignment tasks. Our results reveal a range of concepts for which which models differ in meaningful and surprising ways from humans, including treating "words" as belonging to categories like "vehicles" and "clothing," identifying several "vegetable" category members as "fruit," and assigning exemplars from non-weapon categories to the "weapons" category. We also demonstrate how these instances of concept misalignment translate into problematic downstream behavior with implications for AI safety. 

---
# Trace2Skill: Verifier-Guided Skill Evolution for Long-Context EDA Agents 

**Authors**: Zijian Du, Nathaniel Pinckney  

**Link**: [PDF](https://arxiv.org/pdf/2605.21810)  

**Abstract**: Complex Verilog Design Problems (CVDP) challenge hardware LLM agents because solving them requires localizing verifier-relevant RTL, testbenches, include paths, and build dependencies inside large repository snapshots, making precise edits, and recovering from sparse hidden-verifier failures. We present Trace2Skill, a test-time scaling framework that improves a hardware agent without RTL-specialized model fine-tuning. Rather than training a new model or only sampling more candidate solutions, Trace2Skill treats the agent's natural-language skill as an evolvable policy. It mines repeated rollout traces for success and failure modes, converts them into dense diagnostics and oracle lessons, and uses an oracle, mutator, and selector loop to produce task-specific skills that guide later search, editing, validation, and recovery. Because final pass/fail labels are often too coarse for hard failures, Trace2Skill also supports bounded runtime dense verifier feedback that returns sanitized functional observations while keeping hidden harnesses and reference solutions inaccessible to the agent. This feedback helps guide skill evolution and agent execution by connecting skill text, verifier evidence, and downstream behavior. Across hard CVDP tasks that defeat the seed CVDP agent, including tasks that also defeat frontier coding agents, Trace2Skill with dense verifier feedback substantially improves task pass rates and produces breakthrough passes on previously unsolved tasks, without requiring high-quality fine-tuning data, specialized RTL model training, or model weight updates. The same framework provides a general test-time scaling strategy that can extend beyond digital design to other verifiable EDA tasks. 

---
# Post-Training is About States, Not Tokens: A State Distribution View of SFT, RL, and On-Policy Distillation 

**Authors**: Dong Nie  

**Link**: [PDF](https://arxiv.org/pdf/2605.22731)  

**Abstract**: Large language model post-training methods such as supervised fine-tuning (SFT), reinforcement learning (RL), and distillation are often analyzed through their loss functions: maximum likelihood, policy gradients, forward KL, reverse KL, or related objective-level variants. We study a complementary factor: the state distribution on which supervision is applied. For an autoregressive policy, a state is a prompt plus generated prefix. SFT trains on fixed dataset states, while RL and on-policy distillation (OPD) train on states induced by the current learner. We formalize post-training as state-distribution shaping and run a controlled smallscale study using Qwen3-0.6B-Base on GSM8K, with TruthfulQA and MMLU as retention evaluations. Our results show three phenomena. First, a mild SFT run improves GSM8K with little forgetting, while a stress SFT run causes substantial retention loss. Second, OPD from a degraded SFT teacher surpasses that teacher on GSM8K, TruthfulQA, and MMLU, despite using the teacher as its only supervision source. Third, a lightweight on-policy RL run improves GSM8K while preserving retention. These results support a state-centric view of post-training: the source and locality of training states can be as important as the form of the supervision signal. 

---
# The Distillation Game: Adaptive Attacks & Efficient Defenses 

**Authors**: Youssef Allouah, Mahdi Haghifam, Sanmi Koyejo, Reza Shokri  

**Link**: [PDF](https://arxiv.org/pdf/2605.22737)  

**Abstract**: Distillation attacks create a deployment trade-off for model providers: the same outputs that make a model more useful can also make it easier to imitate. We study this trade-off through a minimax game between a utility-constrained teacher and an adaptive student. Our framework yields tractable one-sided response rules: an adaptive evaluation rule in which the student reweights high-value examples, and a teacher-side defense template that suppresses outputs most useful for distillation. From a cheap proxy for example value, we derive Product-of-Experts (PoE), a simple forward-pass-only defense that combines the teacher with a proxy student during generation. Empirically, adaptive evaluation reveals a large passive--adaptive gap: on state-of-the-art defenses, adaptive students recover substantially more capability than passive evaluation suggests on GSM8K and MATH. Under this stronger evaluation, the apparent robustness gap between expensive defenses and PoE narrows considerably, while PoE remains substantially cheaper and preserves higher-quality reasoning traces. Overall, our results suggest that strong distillation remains difficult to stop, and that progress on antidistillation should be judged against adaptive students rather than passive ones. Our code is available at: this https URL. 

---
# Healthcare LLM Benchmarks Are Only as Good as Their Explicit Assumptions 

**Authors**: Naveen Raman, Santiago Cortes-Gomez, Mateo Dulce Rubio, Fei Fang, Bryan Wilder  

**Link**: [PDF](https://arxiv.org/pdf/2605.22612)  

**Abstract**: Benchmarks are necessary for healthcare evaluation, but are not sufficient for predicting deployment performance. Our position is that the evaluation--deployment gap arises not because of poorly designed benchmarks, but from implicit assumptions about how users interact with models that cannot be surfaced from benchmarks alone. To make this precise, we propose a classification of assumptions into two categories: task, which can be tested from conversation data alone, and outcome, which requires outcome data and behavioral studies for testing. Critically, outcome assumptions depend on human behavior, something that even well-designed benchmarks cannot directly observe. To demonstrate the operationality of this framework, we retrospectively analyze a healthcare RCT as a case study and find that the gap naturally separates into task and outcome gaps of roughly equal size. To address this, we make two contributions: first, we propose BenchmarkCards, an artifact that documents assumptions, and second, we propose staged evaluation, a procedure that systematically tests assumptions and evaluates performance. 

---
# Contractual Skills: A GovernSpec Design Framework for Enterprise AI Agents 

**Authors**: Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22634)  

**Abstract**: Skills are increasingly used to package agent instructions, workflows, scripts, and reference materials. In enterprise settings, however, skills often need to express more than task guidance: they must make goals, input boundaries, permissions, evidence requirements, output contracts, quality criteria, verification steps, human approval points, and handoff rules inspectable. This paper proposes contractual skills, a GovernSpec-inspired design framework for organizing this http URL files as readable task contracts while preserving lightweight skill discovery and progressive loading. The framework clarifies the boundary between contractual skills, GovernSpec YAML contracts, Model Context Protocol surfaces, tool adapters, runtime guardrails, tracing, and evaluation systems.
We evaluate the framework with two offline experiments. A text-generation study covers three enterprise skills, fifteen synthetic tasks, four instruction conditions, and eight generation models, yielding 960 outputs and 1680 cross-judge score records. Contractual skills outperform no-skill and minimal-skill baselines on all tested models. Relative to information-rich plain expanded skills, the gains are small and mixed, suggesting that contractual fields mainly improve checkability and maintainability rather than raw generation quality. A tool-calling challenge covers eight models and 192 simulated tool-call records. Skills usually reduce high-risk tool attempts, but model differences remain and runtime tool guardrails are still required. The results suggest that contractual skills are best understood as a governance layer that makes task intent, boundaries, and acceptance criteria explicit, not as a standalone safety mechanism. 

---
# VGenST-Bench: A Benchmark for Spatio-Temporal Reasoning via Active Video Synthesis 

**Authors**: Jinho Park, Youbin Kim, Hogun Park, Eunbyung Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.22570)  

**Abstract**: Spatio-temporal reasoning is a core capability for Multimodal Large Language Models (MLLMs) operating in the real world. As such, evaluating it precisely has become an essential challenge. However, existing spatio-temporal reasoning benchmark datasets primarily rely on static image sets or passively curated video data, which limits the evaluation of fine-grained reasoning capabilities. In this paper, we introduce VGenST-Bench, a video benchmark that employs generative models to actively synthesize highly controlled and diverse evaluation scenarios. To construct VGenST-Bench, we propose a multi-agent pipeline incorporating a human quality control stage, ensuring the quality of all generated videos and QA pairs. We establish a comprehensive 3x2x2 video taxonomy, encompassing Spatial Scale, Perspective, and Scene Dynamics to span diverse scenarios. Furthermore, we design a hierarchical task suite that decouples low-level visual perception from high-level spatio-temporal reasoning. By shifting the paradigm from passive curation to active synthesis, VGenST-Bench enables fine-grained diagnosis of spatio-temporal understanding in MLLMs. 

---
# Measuring Security Without Fooling Ourselves: Why Benchmarking Agents Is Hard 

**Authors**: Sahar Abdelnabi, Chris Hicks, Konrad Rieck, Ahmad-Reza Sadeghi  

**Link**: [PDF](https://arxiv.org/pdf/2605.22568)  

**Abstract**: The benchmarks used to evaluate AI agents in security-critical roles suffer from crucial weaknesses. Building on recent empirical evidence, we characterize three core challenges that undermine security evaluations: benchmark vulnerabilities, temporal staleness, and runtime uncertainty. We then outline practical directions toward building more robust and trustworthy evaluation frameworks. 

---
# Steins;Gate Drive: Semantic Safety Arbitration over Structured Futures for Latency-Decoupled LLM Planning 

**Authors**: Anjie Qiu, Hans D. Schotten  

**Link**: [PDF](https://arxiv.org/pdf/2605.22456)  

**Abstract**: Cloud-hosted LLM driver agents provide useful semantic judgments, but their inference latency exceeds stepwise vehicle-control windows. Learned world models predict futures, but they usually keep future generation and action selection inside large coupled loops. We present SteinsGateDrive, a latency-decoupled planner-runtime architecture in which the worldline metaphor from the eponymous story names one plausible consequence of an intervention: the LLM selects counterfactual driving futures before the final control instant, and a runtime reuses the selected forecast only while safety contracts remain valid. The generator builds three world-line roles: alpha nominal ego-conditioned futures, beta interaction counterfactuals around nearby vehicles, and gamma hazard-stress futures such as braking, cut-ins, or blocked corridors. The selected branch becomes a typed StrategicForecast with horizon, validity/abort conditions, fallback, and authority. On a within-subject, matched-seed normal-highway protocol with 10 seeds and 20 steps, GPT-5.4 mini reduces effective lag from +3.07 s at 1-second horizon to -0.01 s at 4-second horizon while preserving the measured no-collision safety boundary. The architecture's safety contribution comes from the atom-predicate runtime check, not from the drift score, which functions as a refresh-frequency knob. 

---
# The Neural Compiler: Program-to-Network Translation for Hybrid Scientific Machine Learning 

**Authors**: Lucas Sheneman  

**Link**: [PDF](https://arxiv.org/pdf/2605.22498)  

**Abstract**: Scientific machine learning often requires combining known physics with unknown parameters or correction terms learned from data. Existing approaches either ignore known structure, encode it as a soft penalty, or require hand-written PyTorch code for each equation. We present The Neural Compiler, a system that translates programs written in a first-order Scheme-like expression language into frozen, differentiable PyTorch modules. These modules match the source program to floating-point precision and provide gradients through autograd. In hybrid models, the compiled module encodes known physics exactly while learned components model the unknown remainder. We evaluate the compiler across six experiment domains: Feynman physics equations, Lotka-Volterra dynamics, a damped pendulum, a one-dimensional heat equation, three-dimensional vector mechanics, and compositional generalization. Compiled modules match hand-coded PyTorch implementations numerically for single equations, showing no accuracy loss from compilation. With only 1 to 4 trainable parameters, compiled models recover physical constants to less than 1 percent error in most cases, while standard PINN baselines with more than 8500 parameters show 7 to 93 percent error. Compiled modules also compose with zero error, while neural approximations can accumulate large errors in deep composition chains. The main value of the compiler is not improved accuracy over hand-coded equations, but systematic composability: it generates correct, differentiable modules from symbolic specifications without rewriting each equation by hand. The system supports 51 primitive operations, including vector and matrix algebra, enabling PDE discretizations and hybrid scientific models. This string-in, module-out interface also provides a natural target for large language models that translate scientific descriptions into executable differentiable modules. 

---
# Towards Clinically Interpretable Ophthalmic VQA via Spatially-Grounded Lesion Evidence 

**Authors**: Xingyue Wang, Bo Liu, Meng Wang, Zhixuan Zhang, Chengcheng Zhu, Huazhu Fu, Jiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22414)  

**Abstract**: Visual Question Answering (VQA) holds great promise for clinical support, particularly in ophthalmology, where retinal fundus photography is essential for diagnosis. However, ophthalmic VQA benchmarks primarily emphasize answer accuracy, neglecting the explicit visual evidence necessary for clinical interpretability. In this work, we introduce FundusGround, a new benchmark for clinically interpretable ophthalmic VQA with spatially-grounded lesion evidence. Specifically, we propose a three-stage pipeline that collects 10,719 fundus images with 15,595 image-level meticulously annotated lesions. To ensure anatomical consistency and clinical validity, all lesions are spatially localized using the Early Treatment Diabetic Retinopathy Study (ETDRS) grid, enabling standardized mapping to nine clinically meaningful retinal regions. Built upon this structured lesion evidence, 72,706 questions are then generated spanning four formats: open-ended, closed-ended, single-choice, and multiple-choice. We further benchmark multiple general- and medical- large vision-language models using dual metrics for answer accuracy and lesion-level reasoning. The experiments demonstrate that incorporating lesion-level visual evidence consistently improves model performance and transparency, highlighting the necessity of explicit spatial grounding for reliable and explainable ophthalmic VQA. 

---
# VeriScale: Adversarial Test-Suite Scaling for Verifiable Code Generation 

**Authors**: Yifan Bai, Xiaoyang Liu, Zihao Mou, Guihong Wang, Jian Yu, Shuhan Xie, Yantao Li, Yangyu Zhang, Jingwei Liang, Tao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.22368)  

**Abstract**: As large language models (LLMs) are increasingly deployed for software engineering, constructing high-quality benchmarks is crucial for evaluating not just the functional correctness, but also the formal verifiability of generated code. However, existing benchmarks are limited by the quantity and quality of positive and negative test cases, leading to an overestimation of model capabilities in generating specifications and implementations. To address this, we propose VeriScale, a novel framework driven by the adversarial implementations. It consists of two stages: test-suite expansion to construct diverse and challenging test cases, and test-suite reduction to distill them into compact yet discriminative suites. While VeriScale is general, we instantiate it on Verina to construct VerinaPlus, which expands the original test suites by over 83$\times$, and VerinaLite, a lightweight 14$\times$ variant. Our experiments across eight state-of-the-art LLMs demonstrate that VerinaPlus exposes substantial model weaknesses hidden by the original benchmark, evidenced by sharp score drops on both SpecGen and CodeGen tasks, whereas VerinaLite maintains this discriminative power at a fraction of the evaluation cost. The enhanced benchmarks and source code are publicly available at this https URL. 

---
# Sibyl-AutoResearch: Autonomous Research Needs Self-Evolving Trial-and-Error Harnesses, Not Paper Generators 

**Authors**: Chengcheng Wang, Qinhua Xie, Wei He, Jianyuan Guo, Shiqi Wang, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22343)  

**Abstract**: Autonomous research systems increasingly make the scientific workflow executable: agents can propose ideas, run code, inspect results, and draft papers. But executable workflows do not by themselves produce research judgment. We analyze where current systems lose trial experience: weak evidence becomes prose, pilot signals become broad claims, memory remains textual, and recurring process failures do not change later behavior. We introduce Sibyl-AutoResearch, a self-evolving AutoResearch framework built around Scientific Trial-and-Error Harnesses. A harness lets agents run bounded trials, preserve positive and negative outcomes, and route lessons into later planning, validation, claim scope, scheduling, critique, writing, and harness repair. We formalize this through two auditable conversion units: trial-to-behavior conversion, which links trial signals to later research actions, and trial-to-harness-behavior conversion, which links recurring process failures to system updates. We implement the framework in SIBYL, a file-backed autonomous research system that exposes the state, roles, memory, gates, and artifact traces needed to inspect these conversion paths. A retrospective audit identifies eight high-confidence conversion events, with a median latency of one iteration and a maximum latency of three iterations. A recovered-failure registry further shows how five naturally occurring failure classes, including duplicate results, stale numbers, and unsupported statistics, were blocked, downgraded, or routed into later repair. These traces do not establish a comparative performance claim; they show that the proposed conversion units are recoverable from realistic autonomous-research workspaces. The SIBYL framework and system are available at this https URL. 

---
# Benchmarking Autonomous Agents against Temporal, Spatial, and Semantic Evasions 

**Authors**: Jianan Ma, Xiaohu Du, Ruixiao Lin, Yaoxiang Bian, Jialuo Chen, Jingyi Wang, Xiaofang Yang, Shiwen Cui, Changhua Meng, Xinhao Deng, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22321)  

**Abstract**: As autonomous agents (e.g., OpenClaw) increasingly operate with deep system-level privileges to execute complex tasks, they introduce severe, unmitigated security risks. Current vulnerability analyses overwhelmingly focus on single-turn, stateless behaviors, overlooking the expanded attack surface inherent in stateful, multi-turn interactions and dynamic tool invocations. In this paper, we propose a novel, multi-dimensional evasion framework targeting LLM-based agent systems. We introduce three stealthy attack vectors: (1) Temporal evasion, which fragments malicious payloads across sequential interaction turns; (2) Spatial evasion, which conceals payloads within complex external artifacts that evade standard LLM parsing mechanisms; and (3) Semantic evasion, which obscures malicious intents beneath benign contextual noise. To systematically quantify these threats, we construct A3S-Bench, a comprehensive benchmark comprising 2,254 real-world agent execution trajectories. Evaluating a standard agent framework separately integrated with 10 mainstream LLM backbones against 20 practical threat scenarios, we demonstrate that our evasion framework elevates the average risk trigger rate from a 28.3\% baseline to 52.6\%. These findings reveal systemic, architecture-level vulnerabilities in current autonomous agent systems that existing defenses fail to address, highlighting an urgent need for defense mechanisms tailored to the unique threats. 

---
# One LR Doesn't Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs 

**Authors**: Di He, Songjun Tu, Keyu Wang, Lu Yin, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22297)  

**Abstract**: Learning rate configuration is a fundamental aspect of modern deep learning. The prevailing practice of applying a uniform learning rate across all layers overlooks the structural heterogeneity of Transformers, potentially limiting their effectiveness as the backbone of Large Language Models (LLMs). In this paper, we introduce Layerwise Learning Rate (LLR), an adaptive scheme that assigns distinct learning rates to individual Transformer layers. Our method is grounded in Heavy-Tailed Self-Regularization (HT-SR) theory, which characterizes the empirical spectral density (ESD) of weight correlation matrices to quantify heavy-tailedness. Layers with weaker heavy-tailedness are assigned larger learning rates to accelerate their training, while layers with stronger heavy-tailedness receive smaller learning rates. By tailoring learning rates in this manner, LLR promotes balanced training across layers, leading to faster convergence and improved generalization. Extensive experiments across architectures (from LLaMA to GPT-nano), optimizers (AdamW and Muon), and parameter scales (60M-1B) demonstrate that LLR achieves up to 1.5x training speedup and outperforms baselines, notably raising average zero-shot accuracy from 47.09% to 49.02%. A key advantage of LLR is its low tuning overhead: it transfers nearly optimal LR settings directly from the uniform baseline. Code is available at this https URL. 

---
# What are the Right Symmetries for Formal Theorem Proving? 

**Authors**: Krzysztof Olejniczak, Radoslav Dimitrov, Xingyue Huang, Bernardo Cuenca Grau, Jinwoo Kim, İsmail İlkan Ceylan  

**Link**: [PDF](https://arxiv.org/pdf/2605.22257)  

**Abstract**: Formal theorem provers based on large language models (LLMs) are highly sensitive to superficial variations in problem representation: semantically equivalent statements can exhibit drastically different proof success rates, revealing a failure to respect structural symmetries inherent in formal mathematics. This raises a central question: what are the right symmetries for formal theorem proving? We introduce rewriting categories, a category-theoretic framework capturing the compositional, generally non-invertible transformations induced by proof tactics, and use it to formalize two symmetry notions: proof equivariance, governing how proof distributions transform under rewrites, and success invariance (i.e., invariance of success probability), requiring equivalent statements to be solved with the same probability. We observe that state-based next-tactic provers naturally satisfy proof equivariance by operating on proof states. In contrast, state-of-the-art LLM-based provers satisfy neither property, exhibiting large performance variation across equivalent formulations. To mitigate this, we propose test-time methods that aggregate over equivalent rewritings of the input, showing theoretically that they recover success invariance in the sampling limit, and empirically, that they improve robustness and performance under fixed inference budgets. Our results highlight symmetry as a key missing inductive bias in LLM-based theorem proving and suggest test-time computation as a practical route to approximate it. 

---
# Bernini: Latent Semantic Planning for Video Diffusion 

**Authors**: Bernini Team, Chenchen Liu, Junyi Chen, Lei Li, Lu Chi, Mingzhen Sun, Zhuoying Li, Yi Fu, Ruoyu Guo, Yiheng Wu, Ge Bai, Zehuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.22344)  

**Abstract**: Multimodal large language models (MLLMs) and diffusion models have each reached remarkable maturity: MLLMs excel at reasoning over heterogeneous multimodal inputs with strong semantic grounding, while diffusion models synthesize images and videos with photorealistic fidelity. We argue that these two families can be unified through a simple division of labor: MLLMs perform semantic planning, while diffusion models render pixels from high-level semantic guidance and low-level visual features. Building on this idea, we propose Bernini, a unified framework for video generation and editing. An MLLM-based planner predicts the target semantic representation directly in the ViT embedding space, and a DiT-based renderer synthesizes pixels conditioned on this plan, augmented by text features and, for editing, source VAE features for detail preservation. Because semantics serve as the interface, the planner and renderer can be trained separately and only lightly co-trained, preserving the pretrained strengths of both components while keeping training efficient. To better handle multiple visual inputs, we introduce Segment-Aware 3D Rotary Positional Embedding (SA-3D RoPE), and further incorporate chain-of-thought reasoning in the planner to better transfer understanding into generation. Bernini achieves state-of-the-art performance across a wide range of video generation and editing benchmarks, with the MLLM's pretrained understanding translating into strong generalization on challenging editing tasks. 

---
# EmoTrack: Robust Depression Tracking from Counseling Transcripts across Session Regimes 

**Authors**: Zhaomin Wu, Jiayi Li, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2605.22286)  

**Abstract**: Text-based counseling is an important interface for AI mental-health support, where transcripts may be used to monitor depression severity and flag sessions requiring timely human review. However, robust PHQ-8 prediction across session regimes remains challenging: fine-tuning-based methods can exploit richer supervision but may generalize poorly under data scarcity, while prompt-based LLM methods are data-efficient but usually treat each transcript holistically and provide limited support for longitudinal context. We study robust depression tracking from counseling transcripts across single-session and multi-session regimes. We introduce LongCounsel, a multi-session counseling dataset with session-level PHQ-8 supervision for evaluating repeated-session tracking under partial symptom disclosure and cross-session continuity. We further propose EmoTrack, a PHQ-8 prediction framework that combines LLM-extracted clinical signals with frozen turn-level semantic embeddings and trains symptom-specific predictors over the resulting transcript representation. When prior sessions are available, EmoTrack can further incorporate them through compact cross-session memory. Experiments on LongCounsel and DAIC-WOZ show that EmoTrack achieves a clear gain on the real single-session benchmark, including a 13.5% relative MAE reduction over the strongest DAIC-WOZ baseline, and remains competitive with the strongest longitudinal baseline on LongCounsel. 

---
# Tailoring Teaching to Aptitude: Direction-Adaptive Self-Distillation for LLM Reasoning 

**Authors**: Hongbin Zhang, Chaozheng Wang, Kehai Chen, Youcheng Pan, Yang Xiang, Jinpeng Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22263)  

**Abstract**: On-policy self-distillation (OPSD) is an emerging LLM post-training paradigm in which the model serves as its own teacher: conditioned on privileged information such as a reference trace or hint, the same policy provides dense token-level supervision on its own rollouts. However, recent studies show that OPSD degrades complex reasoning by suppressing predictive uncertainty, which supports exploration and hypothesis revision. Our token-level analysis shows that this failure arises from applying a uniform direction of teacher supervision across tokens with different uncertainty levels: conformity to the privileged self-teacher suppresses exploration at high entropy, while deviation from the teacher degrades step accuracy at low entropy. Accordingly, we propose \textbf{Direction-Adaptive Self-Distillation} (\textbf{DASD}), which reframes privileged self-distillation from uniform teacher imitation into entropy-routed directional supervision: high-entropy tokens are pushed away from the privileged teacher to preserve exploration, while low-entropy tokens are pulled toward the teacher to stabilize step-level execution. Across six mathematical reasoning benchmarks, DASD achieves the best macro Avg@16 over strong RLVR and self-distillation baselines. Pass@$k$, reasoning-health, and generalization analyses show that these average gains come from preserving exploration without sacrificing step-level execution. 

---
# Can Transformers Learn to Verify During Backtracking Search? 

**Authors**: Yin Jun Phua, Tony Ribeiro, Tuan Nguyen, Katsumi Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2605.22221)  

**Abstract**: Backtracking search underlies classical constraint solvers, planners, and theorem provers. Recent transformer-based reasoning systems explore search trees over their own intermediate steps. A common training recipe fits an autoregressive next-token loss on offline solver traces. The model's input at each step is a cumulative trace of all prior decisions. The optimal continue-or-backtrack predictor depends only on the current search state, since two trajectories reaching the same state admit the same viable continuations. We show that decoder-only transformers trained on cumulative traces fail this requirement in two ways: the trace can scatter state features across many positions (scattered retrieval), and the predictor can condition on the trajectory rather than the state (history entanglement). We address scattered retrieval with localization, a trace-level fix that rewrites each decision block to expose state features locally. We address history entanglement with Selective State Attention (SSA), a fixed attention mask that enforces state-based decisions structurally without modifying training data, objective, or parameters. We focus on reactive verification, after propagation has exposed a contradiction. We test SSA on 3-SAT, graph coloring, Blocks World, and backtracking parsing. On same-state pairs that differ only in prior history, SSA emits identical decisions while a cumulative-trained causal baseline does not. Our contribution is a diagnostic of transformer behavior on serialized trajectory data, paired with a structural fix. Pretrained language models that search over their own reasoning steps may face the same failure. Our analysis opens up inference-time context clearing as a candidate way to apply the same isolation without retraining. 

---
# MuKV: Multi-Grained KV Cache Compression for Long Streaming Video Question-Answering 

**Authors**: Junbin Xiao, Jiajun Chen, Tianxiang Sun, Xun Yang, Angela Yao  

**Link**: [PDF](https://arxiv.org/pdf/2605.22269)  

**Abstract**: Long streaming video QA remains challenging due to growing visual tokens and limited reasoning length of large language models (LLMs). KV-caching stores the Key-Value (KV) of the historical tokens via LLM prefill and enables more efficient streaming QA. However, existing methods cache every one or two frames, causing redundant memory usage and losing fine-grained spatial details within frame or temporal contexts across frames. This paper proposes MuKV, a method that features a multi-grained KV cache compression module and a semi-hierarchical retrieval approach to improve both efficiency and accuracy for long streaming VideoQA. For the offline KV cache, MuKV extracts visual representations at patch-, frame-, and segment-levels. The multiple levels of granularity preserve both local cues and global temporal context, while maintaining efficiency with a dual signal token compression mechanism guided by self-attention and frequency. For online QA, MuKV designs a semi-hierarchical retrieval method to retrieve relevant KV caches for answer generation. Experiments on long-streaming VideoQA benchmarks show that MuKV significantly improves answer accuracy, without sacrificing memory and online QA efficiency. Moreover, our compression mechanism alone brings consistent benefits across answer accuracy, memory, and QA efficiency over baselines, showcasing highly effective contribution. 

---
# Not Yet: Humans Outperform LLMs in a Colonel Blotto Tournament 

**Authors**: Dmitry Dagaev, Egor Ivanov, Petr Parshakov, Alexey Savvateev, Gleb Vasiliev  

**Link**: [PDF](https://arxiv.org/pdf/2605.22095)  

**Abstract**: The emergence of large language models (LLMs) has spurred economists to study how humans and LLMs behave in strategic settings. We organized a series of round-robin tournaments in the Colonel Blotto game. This game attracts game theorists' attention due to high-dimensional action space and the absence of pure strategy Nash equilibria. In the first tournament, more than 200 human participants competed against one another. In the second tournament, several popular LLMs were invited to submit strategies. In the third tournament, we matched the number of LLM strategies to the number submitted by humans. We find that humans more often employ better-calibrated intermediate-level allocation heuristics and outperform the simpler, more stereotyped strategies submitted by LLMs. Strategic sophistication is key to success if and only if the necessary level of reasoning depth is reached, while lower and higher levels of reasoning offer no clear advantage over the primitive strategies. Among humans, field of study weakly predicts success: participants with STEM backgrounds perform better in the first tournament. Surprisingly, humans almost do not adjust their strategies across tournaments with different sets of opponents. This result suggests that humans base their choices primarily on the game's rules rather than on the identity of their opponents, treating LLMs much like human competitors. 

---
# JMed48k: A Multi-Profession Japanese Medical Licensing Benchmark for Vision-Language Model Evaluation 

**Authors**: Yue Xun, Junyu Liu, Qian Niu, Xinyi Wang, Zheng Yuan, Zirui Li, Zequn Zhang, Bowen Zhao, Shujun Wang, Irene Li, Kan Hatakeyama-Sato, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2605.22080)  

**Abstract**: We introduce JMed48k, a multi-profession Japanese healthcare licensing benchmark for evaluating vision-language models. Built from official PDF materials released by the Japanese Ministry of Health, Labour and Welfare, JMed48k contains 48,862 exam questions and 20,142 images from 11 national licensing examinations between 2005 and 2025, with visual content annotated under an 8-type taxonomy. From this corpus, we derive JMed48k-Eval, a recent five-year evaluation subset with 12,484 scored questions, including 9,905 text-only questions and 2,579 questions with images. We evaluate 21 proprietary, open-source, and medical-specific models, reporting text-only and with-image performance separately. Because these subsets contain different questions, we further introduce a paired image-removal audit that evaluates questions with images before and after removing visual content to explore four answer-transition states. The audit shows that proprietary and open source models gain substantially from images, whereas medical-specific systems show limited observable use of visual evidence, with many correct answers persisting after image removal. Even among proprietary models, the net image-removal effect varies sevenfold across professions, from +5.7 points on Physician questions to +39.8 points on Public Health Nurse questions. We release JMed48k to support reproducible, profession-stratified evaluation of vision-language models in medical licensing settings. 

---
# SWE-Mutation: Can LLMs Generate Reliable Test Suites in Software Engineering? 

**Authors**: Yuxuan Sun, Yuze Zhao, Yufeng Wang, Yao Du, Zhiyuan Ma, Jinbo Wang, Mengdi Zhang, Kai Zhang, Zhenya Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22175)  

**Abstract**: Evaluating software engineering capabilities has become a core component of modern large language models (LLMs); however, the key bottleneck hindering further scaling lies not in the scarcity of high-quality solutions, but in the lack of high-quality test suites. Test suites are indispensable both for synthesizing program repair trajectories and for providing precise feedback signals in reinforcement learning. Unfortunately, due to the high cost and difficulty of annotation, high-quality test suites have long been hard to obtain, while those automatically generated by LLMs tend to be superficial and lack sufficient discriminative power. As a first step toward constructing high-quality test suites, we introduce SWE-Mutation, a benchmark for evaluating LLM-generated test suites. The benchmark characterizes test suites by introducing systematically mutated solutions that attempt to ``fool'' the test suites and pass validation. We further propose an agentic, language-agnostic framework for automatically generating complex mutants. Our benchmark consists of 2,636 mutated variants derived from 800 original instances and includes a multilingual subset spanning nine programming languages. Experiments on seven LLMs reveal that even DeepSeek-V3.1 achieves only 10.20% verification and 36.15% detection rates, highlighting the inadequacy of current LLMs. Additionally, our agentic mutation strategy enhances realism, reducing average detection rates from 71.04% to 39.81% compared to conventional methods. These findings expose persistent deficiencies in the ability of current LLMs to generate reliable and discriminative test suites. 

---
# One-Way Policy Optimization for Self-Evolving LLMs 

**Authors**: Shuo Yang, Jinda Lu, Kexin Huang, Chiyu Ma, Shaohang Wei, Yuyang Liu, Guoyin Wang, Jingren Zhou, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.22156)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a promising paradigm for scaling reasoning capabilities of Large Language Models (LLMs). However, the sparsity of binary verifier rewards often leads to low efficiency and optimization instability. To stabilize training, existing methods typically impose token-level constraints relative to a reference policy. We identify that such constraints penalize deviations indiscriminately; this can flip verifier-determined direction when the policy attempts to outperform the reference, thereby suppressing gains. To resolve this, we propose One-Way Policy Optimization (OWPO), a method based on the principle of decoupling optimization direction from update magnitude. In OWPO, the verifier dictates the update direction, while the reference policy serves only to adjust the magnitude. Specifically, OWPO applies asymmetric reweighting: it performs Accelerated Alignment for inferior deviations (where the policy lags behind the reference) and Gain Locking for superior deviations (where the policy surpasses the reference). Furthermore, by incorporating iterative reference updates, OWPO creates a ``Ratchet Effect'' that continuously consolidates gains. Experimental results demonstrate that OWPO outperforms strong baselines, including DAPO, OPD, and MOPD, breaking the bottleneck of fixed priors to enable continuous self-evolution without reliance on external reference models. 

---
# LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation 

**Authors**: Zhuo Chen, Xinzhe Yuan, Jianshu Zhang, Jinzong Dong, Ruichen Zhou, Yingchun Niu, Tianhang Zhou, Yu Yang Fredrik Liu, Yuqiang Li, Nanyang Ye, Qinying Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22054)  

**Abstract**: The high cost and data scarcity in scientific exploration have motivated the use of large language models (LLMs) as knowledge-driven components in Bayesian optimization (BO). However, existing approaches typically embed LLMs directly into the sampling or surrogate modeling pipeline, without fully leveraging their significantly lower evaluation cost compared to real-world experiments. To address this limitation, we propose LLM-Accelerated Bayesian Optimization (LABO), a framework that combines LLM predictions with experimental observations within a single BO loop. LABO employs a gating criterion to dynamically balance the reliance on LLM predictions versus actual experiments. By leveraging inexpensive LLM evaluations to broadly explore the search space and reserving costly real experiments only for regions with high uncertainty, LABO achieves more sample-efficient optimization. We provide a theoretical analysis with a cumulative regret bound that formalizes this efficiency gain. Empirical results across diverse scientific tasks demonstrate that LABO consistently outperforms existing methods under identical experimental budgets. Our results suggest that LABO offers a practical and theoretically grounded approach for integrating LLMs into scientific discovery workflows. 

---
# GA-VLN: Geometry-Aware BEV Representation for Efficient Vision-Language Navigation 

**Authors**: Jiahao Yang, Zihan Wang, Xiangyang Li, Xing Zhu, Yujun Shen, Yinghao Xu, Shuqiang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22036)  

**Abstract**: Despite significant progress in Vision-Language Navigation (VLN), existing approaches still rely on dense RGB videos that produce excessive patch tokens and lack explicit spatial structure, resulting in substantial computational overhead and limited spatial reasoning. To address these issues, we introduce the Geometry-Aware BEV (GA-BEV) - a compact, 3D-grounded feature representation that integrates both explicit and implicit geometric cues into multimodal large language model (MLLM) - based navigation systems. We construct BEV spatial maps from RGB-D inputs by projecting visual features into 3D space and aggregating them into an agent-centric layout that preserves geometric consistency while reducing token redundancy. To further enrich geometric understanding, we incorporate features from a pretrained 3D foundation model into the BEV space, injecting structural priors learned from large-scale 3D reconstruction tasks. Together, these complementary cues - explicit depth-based projection and implicit learned priors - yield compact yet spatially expressive representations that substantially improve navigation efficiency and performance. Experiments show that our method achieves state-of-the-art results using only navigation data, without DAgger augmentation or mixed VQA training, demonstrating the robustness and data efficiency of the proposed GA-VLN framework. 

---
# TextTeacher: What Can Language Teach About Images? 

**Authors**: Tobias Christian Nauen, Stanislav Frolov, Brian Bernhard Moser, Federico Raue, Ahmed Anwar, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2605.22098)  

**Abstract**: The platonic representation hypothesis suggests that sufficiently large models converge to a shared representation geometry, even across modalities. Motivated by this, we ask: Can the semantic knowledge of a language model efficiently improve a vision model? As an answer, we introduce TextTeacher, a simple auxiliary objective that injects text embeddings as additional information into image classification training. TextTeacher uses readily available image captions, a pre-trained and frozen text encoder, and a lightweight projection to produce semantic anchors that efficiently guide representations during training while leaving the inference-time model unchanged. On ImageNet with standard ViT backbones, TextTeacher improves accuracy by up to +2.7 percentage points (p.p.) and yields consistent transfer gains (on average +1.0 p.p.) under the same recipe and compute. It outperforms vision knowledge distillation, yielding more accuracy at a constant compute budget or similar accuracy, but 33% faster. Our analysis indicates that TextTeacher acts as a feature-space preconditioner, shaping deeper layers in the first stages of training, and aiding generalization by supplying complementary semantic cues. TextTeacher adds negligible overhead, requires no costly multimodal training of the target model and preserves the simplicity and latency of pure vision models.
Project page with code and captions: this https URL 

---
# Ex-GraphRAG: Interpretable Evidence Routing for Graph-Augmented LLMs 

**Authors**: Yoav Kor Sade, Arvindh Arun, Rishi Puri, Steffen Staab, Maya Bechler-Speicher  

**Link**: [PDF](https://arxiv.org/pdf/2605.21994)  

**Abstract**: GraphRAG conditions language models on subgraphs retrieved from knowledge graphs, encoded via message-passing GNNs. Because these encoders entangle node contributions through iterated neighborhood aggregation, there is no closed-form way to determine how much each retrieved entity influenced the encoder's output, and therefore no way to faithfully audit what structural evidence actually reached the model. We introduce Ex-GraphRAG, which replaces the GNN encoder with a Multivariate Graph Neural Additive Network (M-GNAN), an extension of additive graph models to high-dimensional embedding spaces that yields an exact decomposition of the encoder's output across individual nodes and feature groups, without post-hoc approximation. On STaRK-Prime, this auditable encoder matches black-box performance. Using it to audit evidence routing, we uncover a semantic-structural mismatch: the nodes that dominate the encoder's output are structurally disconnected in the retrieved subgraph, held together by low-attribution intermediaries whose removal degrades multi-hop QA by up to 28%. This mismatch, invisible to any opaque encoder, reveals that semantic importance and structural connectivity are governed by disjoint sets of nodes, with direct implications for retrieval pruning, context construction, and failure diagnosis in graph-augmented LLMs. 

---
# Learning Spatiotemporal Sensitivity in Video LLMs via Counterfactual Reinforcement Learning 

**Authors**: Dazhao Du, Jian Liu, Jialong Qin, Tao Han, Bohai Gu, Fangqi Zhu, Yujia Zhang, Eric Liu, Xi Chen, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.21988)  

**Abstract**: Video large language models (Video LLMs) achieve strong benchmark accuracy, yet often answer video questions through shortcuts such as single-frame cues and language priors rather than by tracking spatiotemporal dynamics. This issue is exacerbated in RL post-training, where correctness-only rewards can further reinforce shortcut policies that obtain high reward without tracking video dynamics. We address this by asking a controlled counterfactual question: if the visual world changed while the question remained fixed, should the answer change or stay the same? Based on this view, we propose \textbf{Counterfactual Relational Policy Optimization (CRPO)}, a dual-branch RL framework for improving \emph{spatiotemporal sensitivity}. CRPO constructs counterfactual videos through horizontal flips and temporal reversals, trains on both original and counterfactual branches, and introduces a \textbf{Counterfactual Relation Reward (CRR)} between their answers. CRR encourages answers to change for dynamic questions and remain unchanged for static questions. This cross-branch constraint makes it difficult for shortcut policies to be consistently rewarded across both branches. To evaluate this property, we introduce \textbf{DyBench}, a paired counterfactual video benchmark with 3,014 videos covering reversible dynamics, moving direction, and event sequence, together with a strict pair-accuracy metric that prevents fixed-answer shortcuts from inflating scores. Experiments show that CRPO outperforms prior RL methods on spatiotemporal-sensitive evaluations while maintaining competitive general video performance. On Qwen3-VL-8B, CRPO improves DyBench P-Acc by +7.7 and TimeBlind I-Acc by +8.2 over the base model, indicating improved spatiotemporal sensitivity rather than stronger reliance on static shortcuts. The project website can be found at this https URL . 

---
# Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow 

**Authors**: Chengsheng Zhang, Chenghao Sun, Zhining Xie, Xinmei Tian  

**Link**: [PDF](https://arxiv.org/pdf/2605.21980)  

**Abstract**: Large Vision-Language Models (LVLMs) represent a significant leap towards empathetic agents, demonstrating remarkable capabilities in emotion understanding. However, the internal mechanisms governing how LVLMs translate abstract visual stimuli into coherent emotional narratives remain largely unexplored, primarily due to the scarcity of visual counterfactuals and the diffuse nature of emotional expression. In this paper, we bridge this gap by introducing a steering-vector-based causal attribution framework tailored for descriptive emotional reasoning. To this end, we construct a specialized dataset to demystify the emotional circuits underlying the three-stage ``Adapt-Aggregate-Execute'' mechanism. Crucially, we discover a functional decoupling: visual emotional cues are aggregated in middle layers via sentiment-specific attention heads, but are subsequently translated into narrative generation in deep layers through emotion-general pathways. Guided by these insights, we regulate the emotional information routing to strengthen attention flow and amplify the semantic activation to consolidate expression. Extensive experiments on the comprehensive MER-UniBench demonstrate that our methods significantly improve performance via inference-time intervention, effectively mitigating emotional hallucinations and corroborating the causal fidelity of the discovered circuits. 

---
# MLLMs Know When Before Speaking: Revealing and Recovering Temporal Grounding via Attention Cues 

**Authors**: Dazhao Du, Liao Duan, Jian Liu, Tao Han, Yujia Zhang, Eric Liu, Xi Chen, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.21954)  

**Abstract**: Video temporal grounding (VTG), which localizes the start and end times of a queried event in an untrimmed video, is a key test of whether multimodal large language models (MLLMs) understand not only what happens but also when it happens. Although modern MLLMs describe video content fluently, their timestamp predictions remain unreliable, while existing remedies either require costly post-training on temporal annotations or rely on coarse training-free heuristics. In this work, we probe the cross-modal attention of MLLMs and uncover a perception-generation gap. Our key finding is that MLLMs often know the target interval during prefill, but lose this signal when generating the final answer. In the prefill stage, a sparse set of attention heads, which we call \emph{Temporal Grounding Heads} (TG-Heads), concentrates query-to-video attention on the ground-truth interval. During autoregressive decoding, however, the answer tokens shift attention away from this interval toward visually salient but query-irrelevant segments. This observation motivates an inference-time read-then-regenerate framework. We first convert TG-Head prefill attention into a debiased frame-level relevance signal and extract the high-attention interval it highlights. We then re-invoke the MLLM with visual context restricted to this interval, using video cropping or attention masking to suppress distractors. Without parameter updates and architectural changes, our framework consistently improves MiMo-VL-7B, Qwen3-VL-8B, and TimeLens-8B on three VTG benchmarks, with gains of up to +3.5 mIoU. The project website can be found at this https URL. 

---
# From Patches to Trajectories: Privileged Process Supervision for Software-Engineering Agents 

**Authors**: Murong Ma, Tianyu Chen, Yun Lin, Shuai Lu, Qinglin Zhu, Yeyun Gong, Zhiyong Huang, Peng Cheng, Yan Lu, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.21996)  

**Abstract**: Supervised fine-tuning (SFT) on long teacher trajectories is the dominant way to instill investigation and reasoning in open software-engineering (SWE) agents. Since every retained response becomes an imitation target, the student inherits the final outcome and intermediate flaws, including ungrounded leaps and redundant loops. High-quality training data must be effective(each step is grounded and narrows the agent's epistemic gap to the correct fix) and efficient(each step is information-bearing rather than redundant or looping). Existing recipes filter or relabel teacher rollouts using only a binary terminal verifier, which does not directly target these axes and provides no supervision on instances where the teacher fails.
Most real issue includes a developer-authored reference patch, $p^\star$, revealing the file paths, runtime behaviors, and coding conventions presupposed by the correct fix, yet standard pipelines discard it. We propose Patches-to-Trajectories (P2T), which uses $p^\star$ as privileged information during curation and formulates trajectory construction as bi-objective optimization over per-step effectiveness and trajectory length. A reverse phase distills $p^\star$ into a latent process graph, $G^\star$, of contextual facts and solution milestones. A forward phase curates trajectories from blinded teacher continuations by scoring per-step progress against $G^\star$ under a leakage-blocking groundedness check and retaining the shortest effective segments.
Using only 1.8k curated SWE-Gym instances, P2T improves effectiveness and efficiency over outcome-filtered SFT and its tool-error-masking variant. On SWE-bench Verified, it raises Pass@1 by up to 10.8 points while reducing per-instance inference cost by ~15%, with consistent gains on SWE-bench Lite. Size-matched ablations and qualitative analysis further isolate trajectory quality from data scale. 

---
# CausalGuard: Conformal Inference under Graph Uncertainty 

**Authors**: Vikash Singh, Weicong Chen, Debargha Ganguly, Yanyan Zhang, Nengbo Wang, Sreehari Sankar, Mohsen Hariri, Alexander Nemecek, Chaoda Song, Shouren Wang, Biyao Zhang, Van Yang, Erman Ayday, Jing Ma, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2605.21928)  

**Abstract**: Estimating treatment effects from observational data requires choosing an adjustment set, but valid adjustment depends on an unknown causal graph. Graph misspecification can cause under-coverage, while graph-agnostic conformal wrappers may regain nominal coverage only through large padding. We introduce CausalGuard, a structure-weighted conformal framework that calibrates after aggregating graph-conditional doubly robust pseudo-outcomes. Candidate DAGs are proposed from an LLM-derived edge prior, pruned by conditional-independence tests, and reweighted by Bayesian Information Criterion. A composite nonconformity score then calibrates the posterior-weighted pseudo-outcome. CausalGuard provides distribution-free finite-sample marginal coverage for this aggregated pseudo-outcome; under causal identification, overlap, conditional-mean nuisance stability, and concentration on target-aligned valid adjustment strategies, its conditional mean converges to the true Conditional Average Treatment Effect. Across five benchmarks, CausalGuard attains mean coverage above the nominal 90% level for the directly evaluable target and reduces width when graph-agnostic conformal baselines require large padding. Stress tests show that CausalGuard suppresses invalid collider adjustment and remains stable under misspecified priors when the retained candidate set is data-supported. 

---
# AgroVG: A Large-Scale Multi-Source Benchmark for Agricultural Visual Grounding 

**Authors**: Haocheng Li, Juepeng Zheng, Zenghao Yang, Kaiqi Du, Guilong Xiao, Gengmeng Pu, Haohuan Fu, Jianxi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22034)  

**Abstract**: Visual grounding, the task of localizing objects described by natural-language expressions, is a foundational capability for agricultural AI systems, enabling applications such as selective weeding, disease monitoring, and targeted harvesting. Reliable evaluation of agricultural visual grounding remains challenging because agricultural targets are often small, repetitive, occluded, or irregularly shaped, and instructions may refer to one, many, or no objects in an image. Evaluating this capability therefore requires jointly testing localization accuracy, target-set completeness, and existence-aware abstention. To address these challenges, we introduce \textbf{AgroVG}, a multi-source benchmark that formulates agricultural grounding as generalized set prediction: given an image and a referring expression, a model must return all matching target instances or abstain when no target is present. AgroVG contains 10{,}071 annotation-grounded image-query pairs from ten source datasets across six target families: crop/weed, fruit, wheat head, pest, plant disease, and tree canopy. It supports bounding-box grounding (T1) across all six families and instance-mask grounding (T2) on sources with reliable instance-level pixel annotations, with queries covering single-target, multi-target, and target-absent regimes. AgroVG further provides task-specific protocols for box-set matching and query-level mask coverage. Zero-shot evaluation of 26 model configurations spanning closed-source MLLMs, open-source VLMs, and specialized grounding systems reveals persistent gaps: the best multi-target Set-$F_1$ reaches only 0.35, and the best positive-query mask success rate at IoU@0.75 remains below 0.17. Data and code are available at this https URL . 

---
# MAVEN: A Multi-stage Agentic Annotation Pipeline for Video Reasoning Tasks 

**Authors**: Han Zhang, Wanting Jiang, Tomasz Kornuta, Tian Zheng, Vidya Murali  

**Link**: [PDF](https://arxiv.org/pdf/2605.21917)  

**Abstract**: Training Vision Language Models (VLMs) for video event reasoning requires high-quality structured annotations capturing not only what happened, but when, where, why, and with what consequence, at a scale manual labelling cannot support. We present MAVEN (Multi-stage Agentic Video Event aNnotation), a multi-stage agentic pipeline that turns raw videos into multi-task training data with Chain-of-Thought (CoT) reasoning traces, organized around a designated Event of Focus. At its core, MAVEN synthesizes a Multi-Scale Spatio-Temporal Event Description (MSTED) from three complementary caption levels; this explicit intermediate serves as the sole input to downstream Q&A generation across multiple task formats. Crucially, MAVEN supports agent-driven domain adaptation: given a new video dataset and target question examples, the agent redesigns all prompts top-down without manual re-engineering. A hierarchical refinement loop further classifies annotation errors against a taxonomy, traces root causes to the originating pipeline stage, and applies targeted edits that rewrite prompts or modify the pipeline structure itself, iteratively improving data quality. We apply MAVEN to label over 5,300 traffic videos and fine-tune Cosmos-Reason2-8B on the resulting data. On a private CCTV evaluation set, fine-tuning surpasses both Gemini 2.5 Pro and 3.1 Flash, including a $+38.8$-point gain in MCQ accuracy over zero-shot. On AccidentBench, CCTV-only training lifts Cosmos-Reason2 by $+10.7$ MCQ points and matches Gemini 2.5 Pro despite seeing no dashcam videos; adding agent-adapted dashcam annotations narrows the gap to Gemini 3.1 Flash, and RL post-training pushes overall performance past both Gemini baselines. Qualitative results on warehouse surveillance and public safety videos further show the agentic workflow readily adapts the pipeline to new domains. 

---
# The Illusion of Reasoning: Exposing Evasive Data Contamination in LLMs via Zero-CoT Truncation 

**Authors**: Yifan Lan, Yuanpu Cao, Hanyu Wang, Lu Lin, Jinghui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.21856)  

**Abstract**: Large language models (LLMs) have demonstrated impressive reasoning abilities across a wide range of tasks, but data contamination undermines the objective evaluation of these capabilities. This problem is further exacerbated by malicious model publishers who use evasive, or indirect, contamination strategies, such as paraphrasing benchmark data to evade existing detection methods and artificially boost leaderboard performance. Current approaches struggle to reliably detect such stealthy contamination. In this work, we uncover a critical phenomenon: a model's generated reasoning steps actively mask its underlying memorization. Inspired by this, we propose the Zero-CoT Probe (ZCP), a novel black-box detection method that deliberately truncates the entire Chain-of-Thought (CoT) process to expose latent shortcut mappings. To further isolate memorization from the model's intrinsic problem-solving capabilities, ZCP compares the model's zero-CoT performance on the original benchmark against an isomorphically perturbed reference dataset. Furthermore, we introduce Contamination Confidence, a metric that quantifies both the likelihood and severity of contamination, moving beyond simple binary classifications. Extensive experiments on both previously identified contaminated models and specially fine-tuned contaminated models demonstrate that ZCP robustly detects both direct and evasive data contamination. The code for ZCP is accessible at this https URL. 

---
# OPPO: Bayesian Value Recursion for Token-Level Credit Assignment in LLM Reasoning 

**Authors**: Yu Li, Rui Miao, Tian Lan, Zhengling Qi  

**Link**: [PDF](https://arxiv.org/pdf/2605.21851)  

**Abstract**: Reinforcement learning with verifiable rewards has become the standard recipe for improving LLM reasoning, but the dominant algorithm GRPO assigns a single trajectory-level advantage to every token, diluting the signal at pivotal reasoning steps and injecting noise at uninformative ones. Critic-free alternatives derived from on-policy distillation supply per-token signals through oracle-conditioned likelihood ratios, yet apply each signal in isolation from the trajectory-level evidence accumulated up to that position. We propose Oracle-Prompted Policy Optimization (OPPO), which rests on a single observation: the oracle signal used by prior distillation-style methods for local discrimination is also the natural Bayesian update of the model's belief about eventual success. Accumulating the signal along a trajectory yields, in closed form and at the cost of one extra forward pass, a running estimate of the success probability at every position, together with a token-level advantage that requires no learned value network and no additional rollouts. A first-order analysis factorizes the advantage into the per-token discrimination signal used by distillation methods modulated by a state weight that concentrates credit on genuinely pivotal tokens, with a directional variance-reduction guarantee. The framework admits two estimators differing only in which model scores the evidence: a \textit{self-oracle} that reuses the student and recovers the on-policy distillation reward as a strict special case, and a \textit{teacher-oracle} that delegates scoring to a stronger frozen model. On two base LLMs across seven mathematics, science, and code reasoning benchmarks, OPPO improves over GRPO, DAPO, and SDPO by up to $+6.0$ points on AMC'23 and $+5.2$ points on AIME'24, with gains that widen monotonically with response length. 

---
# EvoScene-VLA: Evolving Scene Beliefs Inside the Action Decoder for Chunked Robot Control 

**Authors**: Chushan Zhang, Ruihan Lu, Jinguang Tong, Xuesong Li, Yikai Wang, Hongdong Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.21862)  

**Abstract**: Chunked vision-language-action (VLA) policies predict multi-step robot controls, conditioning each update on the current visual observation alone. Yet robot actions cause contact, occlusion, and object motion, and the geometry that later decisions depend on can change before the next visual update arrives. Spatial VLAs improve current-frame geometry. Temporal VLAs aggregate past frames. Neither maintains an action-updated scene prior across chunks. We argue for a persistent action-updated scene state across control calls, and introduce EvoScene-VLA. Its recurrent scene prefix carries a geometry-aware scene state across chunks. At each vision-language model (VLM) call, the VLM combines scene information from the current observation with the action-updated prior from the previous chunk; the action decoder outputs both the next action chunk and a compact scene update. This update becomes the next prior, which the VLM corrects against the new observation when the next call arrives. Each control call therefore starts from a scene prior that reflects both recent actions and fresh visual evidence. During training, \textbf{Scene Predictor} supplies future scene-token targets, and Geometric Anchor aligns scene slots with frozen depth and 3D teachers. We discard both modules at deployment. On 31 RoboTwin tasks, EvoScene-VLA raises average success from 87.2% to 89.1% in fixed evaluation and from 86.1% to 88.5% in randomized evaluation. On the Galaxea R1-Lite real robot, EvoScene-VLA outperforms all baselines. 

---
# SDGBiasBench: Benchmarking and Mitigating Vision--Language Models' Biases in Sustainable Development Goals 

**Authors**: Zihang Lin, Huaiyuan Qin, Muli Yang, Hongyuan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.21919)  

**Abstract**: Assessing progress toward the Sustainable Development Goals (SDGs) requires multi-step reasoning over visual cues, contextual knowledge, and development indicators, where incomplete evidence use and imperfect evidence integration can introduce hidden prediction biases. Real-world SDG monitoring further spans both qualitative judgments and quantitative estimation. However, existing benchmarks typically evaluate these aspects in isolation, obscuring systematic biases that emerge when models substitute priors for evidence. To address this gap, we propose SDGBiasBench, a large-scale benchmark suite for SDG-oriented vision-language reasoning. Spanning 500k expert-involved multiple-choice questions and 50k regression tasks, the benchmark enables comprehensive assessment of both decision-level and estimation-level bias in Vision--Language Models (VLMs). Evaluations on SDGBiasBench reveal an intrinsic SDG bias in current VLMs, where predictions are frequently driven by SDG specific priors rather than reliable multi-modal cues. To mitigate such bias, we propose CADE (Contrastive Adaptive Debias Ensemble), a training-free, plug-and-play method that leverages modality-specific answer priors. CADE yields significant gains on the proposed benchmark, improving multiple-choice accuracy by up to 25% and reducing regression MAE by up to 12 points across multiple VLMs. We hope our work can foster the development of more fair and reliable AI systems for sustainable development. 

---
# PocketAgents: A Manifest-Driven Library of Autonomous Defense Agents 

**Authors**: Sidnei Barbieri, Ágney Lopes Roth Ferraz, Lourenço Alves Pereira Júnior  

**Link**: [PDF](https://arxiv.org/pdf/2605.21694)  

**Abstract**: Connecting large language models (LLMs) to defensive enforcement requires more than asking a model whether an attack is happening. A defender must decide which model outputs may change the system state, which outputs must be rejected, and how failures should be recorded. We present PocketAgents, a manifest-driven library of autonomous defense agents. Each agent is installed as three data files: a manifest, a prompt, and a runtime context. The shared runtime gives the agent bounded telemetry access and accepts only typed reports whose requested action appears in the manifest. We implemented PocketAgents on top of a cyber arena (Perry), a cyber-deception testbed, and evaluated two agents, Command and Control and Exfiltration, in 18 closed-loop trials of a DarkSide-inspired attack on a small enterprise topology. Thirteen trials produced validated network-block actions and contained the attack; four failed schema validation; one produced a valid no-action decision. The experiments show that a typed boundary makes LLM-driven defense measurable, extensible, and attributable. 

---
# Understanding Perspectives of Patients, Caregivers and Clinicians towards Emerging Collaborative-decision Making Technologies 

**Authors**: Ray-Yuan Chung, Athena Ortega, Zixuan Xu, Daeun Yoo, Jaime Snyder, Wanda Pratt, Aaron Wightman, Ryan Hutson, Cozumel Pruette, Ari Pollack  

**Link**: [PDF](https://arxiv.org/pdf/2605.21777)  

**Abstract**: In pediatrics, patients, caregivers, and clinicians share responsibility for health decisions, but limited collaboration can undermine outcomes. We conducted a qualitative study examining decision-makers perceptions toward collaborative decision-making technologies, including interactive dashboards, VR simulators, and AI voice assistants. Findings reveal differences in user opinions across groups and indicate technology acceptance is linked to users trust of these technologies. Technology developers and researchers need to explore design and implementation strategies that build and facilitate trust or appropriate distrust between users and these novel technologies before these tools can effectively support collaborative decision-making. 

---
# Faster Completion, Less Learning: Generative AI Reduced Study Time on Math Problems and the Knowledge They Build 

**Authors**: Sina Rismanchian, Hasan Uzun, Jeffrey Matayoshi, Eric Cosyn, Eyad Kurd-Misto  

**Link**: [PDF](https://arxiv.org/pdf/2605.21629)  

**Abstract**: How much have students' ordinary learning processes shifted in response to generative AI, and how does that affect their durable learning outcomes? Self-report surveys show little change, while small-scale behavioral studies report widespread AI use without the scale or duration to measure learning consequences. We address both questions using a ten-year panel of $3.2$ million ALEKS learning interactions for the time-on-task analysis, complemented by ALEKS PPL placement-assessment data for the proctoring and retention analyses, with a quasi-experimental design exploiting within-curriculum variation in AI susceptibility: text-based word problems transcribable into AI prompts serve as the treated group; graph-based problems requiring interactive platform manipulation as the comparison. Learning time on AI-susceptible problems declines $2.8\%$ per quarter among college students after ChatGPT's release, cumulating to $26.9\%$ over eleven quarters; high-schoolers show $31.3\%$, middle-schoolers $9.0\%$, and Grade 5 students no detectable change. The divergence vanishes entirely under proctoring for college students, making general efficiency gains unlikely. Logistic fixed-effects models on randomly assigned proctored retention items yield a $25\%$ cumulative decline in odds of correct response; the same estimator on non-proctored assessment produces a large opposite-signed increase -- inconsistent with any platform, cohort, or curriculum explanation. These results are among the first large-scale behavioral and outcome evidence that generative AI has altered how students study and the knowledge they build -- the population-level indicator of \emph{cognitive surrender}, with direct implications for educational research, assessment governance, and AI policy. 

---
# TBP-mHC: full expressivity for manifold-constrained hyper connections through transportation polytopes 

**Authors**: Anton Lyubinin  

**Link**: [PDF](https://arxiv.org/pdf/2605.21724)  

**Abstract**: Hyper-Connections (HC) improve residual networks by introducing learnable mixing across multiple residual streams, but unconstrained mixing leads to training instability. Manifold-Constrained Hyper-Connections (mHC) address this by enforcing approximate double stochasticity via Sinkhorn normalization, while mHC-lite ensures exact constraints through convex combinations of permutation matrices at the cost of factorial complexity. KromHC reduces this cost using Kronecker-product parameterizations, but restricts the mixing matrices to a structured submanifold of the Birkhoff polytope .
We propose Transportation Birkhoff Polytope (TBP) parameterizations and their Recursive variants (RTBP), which construct exactly doubly stochastic mixing matrices with $(n-1)^2$ degrees of freedom. Our approach avoids iterative normalization and combinatorial explosion while preserving full expressivity of the Birkhoff polytope. Empirical results on language model pre-training' demonstrate competitive performance with improved stability and scalability. 

---
# RefusalBench: Why Refusal Rate Misranks Frontier LLMs on Biological Research Prompts 

**Authors**: Lukas Weidener, Marko Brkić, Mihailo Jovanović, Emre Ulgac, Aakaash Meduri  

**Link**: [PDF](https://arxiv.org/pdf/2605.21545)  

**Abstract**: Frontier large language models are increasingly deployed as orchestration backbones for biological research workflows, yet no shared evidence base exists for comparing their refusal behaviour on legitimate research prompts. RefusalBench, introduced here, is a matched-triple benchmark of 141 prompts in 47 bundles that holds task framing constant while varying only biological risk tier (benign, borderline, dual-use), enabling tier-conditioned comparisons robust to subdomain confounding. A 15-prompt should-refuse positive-control module establishes per-model calibration floors; three models fail to refuse even these prompts. Across 19 frontier models in the May 2026 snapshot, strict refusal rates span 0.1% to 94.6% on identical prompts. Jurisdiction does not predict refusal in this snapshot (Mann-Whitney U, p = 0.393; EU n = 1, US bimodal); provider identity does, with Anthropic's API stack predicting refusal at OR = 21.03 (95% CI: 14.58-30.34 prompt-clustered; 5.70-77.55 under model-clustered GEE). This effect is best read as access-path-level rather than model-weight-level: 99.8% of Anthropic's strict refusals carry the same safety_policy adjudicated reason code, consistent with a small set of canonical refusal templates rather than case-by-case model reasoning. Strict refusal rate misranks safety calibration: Grok 4.20 achieves the highest tier discrimination (Youden's J = 0.787) while ranking only seventh by overall refusal rate, and Claude Opus 4.7's J dropped 65% from prior versions with no improvement in dual-use detection. Nine of 18 frontier models exhibit a hedge-but-help partial-compliance pattern at dual-use tier that binary refusal metrics cannot detect. 

---
# Frequency-Domain Regularized Adversarial Alignment for Transferable Attacks against Closed-Source MLLMs 

**Authors**: Leitao Yuan, Qinghua Mao, Daizong Liu, Kun Wang, Wenjie Wang, Yan Teng, Jing Shao, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.21541)  

**Abstract**: Multimodal large language models (MLLMs) remain vulnerable to transfer-based targeted attacks, where perturbations optimized on open-source surrogate encoders can generalize to closed-source MLLMs. A key challenge for improving adversarial transferability is to effectively capture the intrinsic visual focus shared across different models, such that perturbations align with transferable semantic cues rather than surrogate-specific behaviors. However, existing methods suffer from spatial-domain feature redundancy and surrogate-specific gradient signals, thereby hindering cross-model transferability. In this paper, we propose FRA-Attack, which addresses both challenges from a unified frequency-domain regularization perspective. For feature alignment, a high-pass DCT objective on patch features suppresses redundant global structures and concentrates the loss on the high-frequency band that carries the MLLMs' intrinsic visual focus. For gradient optimization, we introduce Frequency-domain Gradient Regularization (FGR), a \textit{model-agnostic} low-pass regularizer that modulates the surrogate gradient using only the geometric frequency coordinate, \textit{i.e.}, no surrogate-derived statistic is involved, so that FGR is model-agnostic by construction, removing surrogate-specific high-frequency artifacts while preserving transferable low-frequency directions. Together, the two components form a unified frequency-domain treatment of transferability. Extensive experiments on $15$ flagship MLLMs across $7$ vendors show that FRA-Attack achieves superior cross-model transferability, particularly with state-of-the-art performance on GPT-5.4, Claude-Opus-4.6 and Gemini-3-flash. 

---
# When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning 

**Authors**: Xiaogeng Liu, Xinyan Wang, Yingzi Ma, Yechao Zhang, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.21606)  

**Abstract**: On-policy self-distillation (OPSD) trains a student on its own rollouts using a privileged teacher, but its standard objective weights all generated tokens equally, implicitly treating the privileged teacher target as equally reliable at every student-visited prefix. Existing entropy-based OPD methods relax this uniformity by modulating token-level supervision with teacher entropy, but high teacher entropy in reasoning has an ambiguous reliability meaning: it can reflect either non-viable uncertainty or benign solution diversity. To identify this phenomenon, we introduce a branch-viability diagnostic. Specifically, we record next-token alternatives from the privileged-answer teacher prompt, force each alternative after the student prompt plus its on-policy spine prefix, and test whether the resulting student-template continuation recovers the correct answer. On Qwen3-4B, we find that an oriented within-sequence position score is the strongest tested predictor of teacher-token reliability, reaching an area-under-ROC-curve (AUROC) of 0.83; local uncertainty scores are at most 0.57. Motivated by this trajectory-level structure, we propose Position-Weighted On-Policy Self-Distillation (PW-OPSD), which applies an increasing position weight while keeping the same student rollout, privileged teacher pass, and clipped forward-KL target as OPSD. In our comprehensive evaluations with different random seeds, the diagnostic-derived PW-OPSD improves AIME 2024 and AIME 2025 Avg@12 by +1.0 and +1.1 points, and a generalization evaluation on two larger-scale models from different families, DeepSeek-R1-Distill-Llama-8B and Olmo-3-7B-Think, also demonstrates consistent aggregate Avg@12 improvements. These results show that teacher-token reliability in reasoning distillation is trajectory-structured and can be utilized without additional teacher computation. 

---
# Protein Thoughts: Interpretable Reasoning with Tree of Thoughts and Embedding-Space Flow Matching for Protein-Protein Interaction Discovery 

**Authors**: Kingsley Yeon, Xuefeng Liu, Promit Ghosal  

**Link**: [PDF](https://arxiv.org/pdf/2605.21522)  

**Abstract**: Protein-protein interactions (PPIs) govern nearly all cellular processes, yet computational methods for identifying binding partners typically produce ranked predictions without mechanistic justification. This creates a fundamental barrier to adoption because biologists cannot assess whether predictions reflect genuine biochemical insight or spurious correlations. We present \textbf{Protein Thoughts}, a framework that reformulates PPI discovery as an interpretable search problem with explicit reasoning. The system decomposes binding evidence into four biologically meaningful signals: sequence similarity reflecting evolutionary relationships, structural complementarity capturing geometric fit, interface balance, and chemical compatibility encoding residue-level interactions. Rather than collapsing these signals into an opaque score, we preserve their individual contributions through a transparent value function that enables both ranking and auditing. To navigate large candidate spaces efficiently, we introduce hypothesis-guided entropy-regularized Tree-of-Thoughts search. A fine-tuned language model generates search directives from embedding-derived features, classifying candidates as high-priority, exploratory, or skippable. These directives condition a Boltzmann policy that balances exploitation with entropy-driven exploration, while hypothesis-aware pruning prevents premature abandonment of promising candidates. For candidates exhibiting score disagreement, hypothesis-conditioned embedding-space flow matching transports protein embeddings toward the binder manifold. On the SHS148k benchmark, Protein Thoughts achieves mean best-binder rank of 11.2 versus 47.7 for an entropic tree search baseline, a 76% improvement, and for binding prediction the trained value function achieves $91.08 \pm 0.19$ Micro-F1, outperforming existing PPI methods on the same dataset. 

---
# Predicting Performance of Symbolic and Prompt Programs with Examples 

**Authors**: Chengqi Zheng, Keya Hu, Shuzhi Liu, Tao Wu, Kevin Ellis, Yewen Pu  

**Link**: [PDF](https://arxiv.org/pdf/2605.21515)  

**Abstract**: LLM prompting is widely used for naturally stated tasks, yet it is unreliable it may succeed on a few test cases but fail at deployment time. We study performance prediction: given a program, either symbolic (e.g. Python) or a prompt executed on an LLM, and a few in-domain examples, predict its performance on unseen tasks from the same domain. We use a simple coin-flip model, treating each pass/fail program execution as a Bernoulli random variable, whose success probability is the programs unknown performance. In this model, performance depends entirely on: 1) the observed execution outcomes on test cases, and 2) a prior over performances. We compile empirical performance priors from a corpus of diverse programs and tasks, and find that performance for symbolic programs (e.g., Python) are all or nothing, while prompt programs have a diffuse prior with many nearly-correct programs. This difference explains why a few passing tests can certify symbolic programs but not prompt programs. Building on this insight, we develop RAP (Retrieved Approximate Prior), which retrieves similar tasks and prompt programs from an existing corpus to construct a proxy prior, which is then used to predict performance. We show RAP achieves solid performances. 

---
# Autonomous LLM Agents & CTFs: A Second Look 

**Authors**: Youness Bouchari, Matteo Boffa, Marco Mellia, Idilio Drago, Thanh Minh Bui, Dario Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2605.21497)  

**Abstract**: Large Language Model (LLM) agents are increasingly proposed to automate offensive security tasks, with recent studies reporting near human-level success rates in Capture-the-Flag (CTF) challenges. We here revisit these results, providing a second look at these claims. We engineer different agent architectures of increasing complexity and modularity on 30 web-based CTFs challenges spanning 14 vulnerability classes. We instantiate these agents with multiple LLM backbones, and compare them with claude-code, a general-purpose agent that automatically determines its internal architecture. Our evaluation yields three main findings. First, claude-code achieves performance comparable to the engineered architectures (19/30 solved tasks), suggesting that general-purpose agents are strong baselines for offensive security tasks. Second, both our architectures and claude-code struggle in the same challenge categories, revealing persistent barriers that keep current agents below human-level capability. Third, by leveraging our manually designed architectures we can systematically measure the impact of additional components, finding that structured orchestration of specialized roles outperforms monolithic designs, improving run-to-run consistency, and reducing execution costs. 

---
# Harnesses for Inference-Time Alignment over Execution Trajectories 

**Authors**: Boyuan Wang, Bochao Li, Minghan Wang, Yuxin Tao, Fang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2605.21516)  

**Abstract**: Harness engineering has emerged as an important inference-time technique for large language model (LLM) agents, aiming to improve long-term performance through task decomposition and guided execution. However, more elaborate harnesses are not uniformly better: increasing decomposition or guidance can sometimes improve execution, but can also reduce final task success. We study harness design through the lens of inference-time trajectory alignment. This perspective separates harness into two mechanisms: task decomposition, which structures a task into sub-goals, and guided execution, which reshapes local action distributions during execution. This decomposition allows us to quantify how workflow granularity, retry budgets, and guidance-induced action reweighting shape the performance limits of harness design. It further reveals concrete failure modes, including over-decomposition, over-pruning, and hallucinated execution. We validate these predictions through controlled synthetic experiments and real terminal agent benchmarks. Inspired by the theory, we further show that effective harnesses can be partial: specifying only the initial steps and leaving the remaining execution to agent can achieve higher pass rate than fully structured workflows. 

---
# High-speed Networking for Giga-Scale AI Factories 

**Authors**: Sajy Khashab, Albert Gran Alcoz, Alon Gal, Jacky Romano, Rani Abboud, Yonatan Piasetzky, Lior Maman, Amit Nishry, Barak Gafni, Omer Shabtai, Matty Kadosh, Dror Goldenberg, Gilad Shainer, Mark Silberstein  

**Link**: [PDF](https://arxiv.org/pdf/2605.21187)  

**Abstract**: As distributed model training scales to span hundreds of thousands of GPUs, scale-out networks face unprecedented performance and efficiency demands. NVIDIA Spectrum-X Ethernet has been designed from the ground up to achieve predictable and stable network performance with high utilization and low latency. This paper presents the Spectrum-X multiplane architecture, which replaces hierarchical depth with topological parallelism, and introduces hardware-accelerated load balancing in NICs and switches as the key architectural approach to provide fast reaction to highly dynamic network conditions at the microsecond timescales that AI training workloads demand. We describe the motivation, design principles, evaluation methodology and performance on state-of-the-art benchmarks, as well as the lessons we learned from deploying and debugging Spectrum-X networks in large-scale systems. Our evaluation highlights production-grade AI infrastructure performance across three core dimensions: 98% of the theoretical line rate with low jitter-free latency; strong cross-tenant isolation for concurrent workloads; robust, capacity-proportional bisection bandwidth and 7% latency increase for 10% fabric link failures; and rapid reaction to host and fabric link flaps during LLM training workloads. 

---
