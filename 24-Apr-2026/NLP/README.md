# Evaluation of Automatic Speech Recognition Using Generative Large Language Models 

**Authors**: Thibault Bañeras-Roux, Shashi Kumar, Driss Khalil, Sergio Burdisso, Petr Motlicek, Shiran Liu, Mickael Rouvier, Jane Wottawa, Richard Dufour  

**Link**: [PDF](https://arxiv.org/pdf/2604.21928)  

**Abstract**: Automatic Speech Recognition (ASR) is traditionally evaluated using Word Error Rate (WER), a metric that is insensitive to meaning. Embedding-based semantic metrics are better correlated with human perception, but decoder-based Large Language Models (LLMs) remain underexplored for this task. This paper evaluates their relevance through three approaches: (1) selecting the best hypothesis between two candidates, (2) computing semantic distance using generative embeddings, and (3) qualitative classification of errors. On the HATS dataset, the best LLMs achieve 92--94\% agreement with human annotators for hypothesis selection, compared to 63\% for WER, also outperforming semantic metrics. Embeddings from decoder-based LLMs show performance comparable to encoder models. Finally, LLMs offer a promising direction for interpretable and semantic ASR evaluation. 

---
# MathDuels: Evaluating LLMs as Problem Posers and Solvers 

**Authors**: Zhiqiu Xu, Shibo Jin, Shreya Arya, Mayur Naik  

**Link**: [PDF](https://arxiv.org/pdf/2604.21916)  

**Abstract**: As frontier language models attain near-ceiling performance on static mathematical benchmarks, existing evaluations are increasingly unable to differentiate model capabilities, largely because they cast models solely as solvers of fixed problem sets. We introduce MathDuels, a self-play benchmark in which models occupy dual roles: each authors math problems under adversarial prompting and solves problems authored by every other participant. Problems are produced through a three-stage generation pipeline (meta-prompting, problem generation, and difficulty amplification), and validated by an independent verifier that excludes ill-posed questions. A Rasch model (Rasch, 1993) jointly estimates solver abilities and problem difficulties; author quality is derived from the difficulties of each model's authored problems. Experiments across 19 frontier models reveal that authoring and solving capabilities are partially decoupled, and that dual-role evaluation reveals capability separations invisible in single-role benchmarks. As newer models enter the arena, they produce problems that defeat previously dominant solvers, so the benchmark's difficulty co-evolves with participant strength rather than saturating at a fixed ceiling. We host a public leaderboard that updates as new models are released. 

---
# GiVA: Gradient-Informed Bases for Vector-Based Adaptation 

**Authors**: Neeraj Gangwar, Rishabh Deshmukh, Michael Shavlovsky, Hancao Li, Vivek Mittal, Lexing Ying, Nickvash Kani  

**Link**: [PDF](https://arxiv.org/pdf/2604.21901)  

**Abstract**: As model sizes continue to grow, parameter-efficient fine-tuning has emerged as a powerful alternative to full fine-tuning. While LoRA is widely adopted among these methods, recent research has explored vector-based adaptation methods due to their extreme parameter efficiency. However, these methods typically require substantially higher ranks than LoRA to match its performance, leading to increased training costs. This work introduces GiVA, a gradient-based initialization strategy for vector-based adaptation. It achieves training times comparable to LoRA and maintains the extreme parameter efficiency of vector-based adaptation. We evaluate GiVA across diverse benchmarks, including natural language understanding, natural language generation, and image classification. Experiments show that our approach consistently outperforms or achieves performance competitive with existing vector-based adaptation methods and LoRA while reducing rank requirements by a factor of eight ($8\times$). 

---
# Mapping the Political Discourse in the Brazilian Chamber of Deputies: A Multi-Faceted Computational Approach 

**Authors**: Flávio Soriano, Victoria F. Mello, Pedro B. Rigueira, Gisele L. Pappa, Wagner Meira Jr., Ana Paula Couto da Silva, Jussara M. Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2604.21897)  

**Abstract**: Analyses of legislative behavior often rely on voting records, overlooking the rich semantic and rhetorical content of political speech. In this paper, we ask three complementary questions about parliamentary discourse: how things are said, what is being said, and who is speaking in discursively similar ways. To answer these questions, we introduce a scalable and generalizable computational framework that combines diachronic stylometric analysis, contextual topic modeling, and semantic clustering of deputies' speeches. We apply this framework to a large-scale case study of the Brazilian Chamber of Deputies, using a corpus of over 450,000 speeches from 2003 to 2025. Our results show a long-term stylistic shift toward shorter and more direct speeches, a legislative agenda that reorients sharply in response to national crises, and a granular map of discursive alignments in which regional and gender identities often prove more salient than formal party affiliation. More broadly, this work offers a robust methodology for analyzing parliamentary discourse as a multidimensional phenomenon that complements traditional vote-based approaches. 

---
# EVENT5Ws: A Large Dataset for Open-Domain Event Extraction from Documents 

**Authors**: Praval Sharma, Ashok Samal, Leen-Kiat Soh, Deepti Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.21890)  

**Abstract**: Event extraction identifies the central aspects of events from text. It supports event understanding and analysis, which is crucial for tasks such as informed decision-making in emergencies. Therefore, it is necessary to develop automated event extraction approaches. However, existing datasets for algorithm development have limitations, including limited coverage of event types in closed-domain settings and a lack of large, manually verified dataset in open-domain settings. To address these limitations, we create EVENT5Ws , a large, manually annotated, and statistically verified open-domain event extraction dataset. We design a systematic annotation pipeline to create the dataset and provide empirical insights into annotation complexity. Using EVENT5Ws, we evaluate state-of-the-art pre-trained large language models and establish a benchmark for future research. We further show that models trained on EVENT5Ws generalize effectively to datasets from different geographical contexts, which demonstrates its potential for developing generalizable algorithms. Finally, we summarize the lessons learned during the dataset development and provide recommendations to support future large-scale dataset development. 

---
# TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale 

**Authors**: Jun Wang, Ziyin Zhang, Rui Wang, Hang Yu, Peng Di, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.21889)  

**Abstract**: Real-time detection and mitigation of technical anomalies are critical for large-scale cloud-native services, where even minutes of downtime can result in massive financial losses and diminished user trust. While customer incidents serve as a vital signal for discovering risks missed by monitoring, extracting actionable intelligence from this data remains challenging due to extreme noise, high throughput, and semantic complexity of diverse business lines. In this paper, we present TingIS, an end-to-end system designed for enterprise-grade incident discovery. At the core of TingIS is a multi-stage event linking engine that synergizes efficient indexing techniques with Large Language Models (LLMs) to make informed decisions on event merging, enabling the stable extraction of actionable incidents from just a handful of diverse user descriptions. This engine is complemented by a cascaded routing mechanism for precise business attribution and a multi-dimensional noise reduction pipeline that integrates domain knowledge, statistical patterns, and behavioral filtering. Deployed in a production environment handling a peak throughput of over 2,000 messages per minute and 300,000 messages per day, TingIS achieves a P90 alert latency of 3.5 minutes and a 95\% discovery rate for high-priority incidents. Benchmarks constructed from real-world data demonstrate that TingIS significantly outperforms baseline methods in routing accuracy, clustering quality, and Signal-to-Noise Ratio. 

---
# A Multimodal Text- and Graph-Based Approach for Open-Domain Event Extraction from Documents 

**Authors**: Praval Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2604.21885)  

**Abstract**: Event extraction is essential for event understanding and analysis. It supports tasks such as document summarization and decision-making in emergency scenarios. However, existing event extraction approaches have limitations: (1) closed-domain algorithms are restricted to predefined event types and thus rarely generalize to unseen types and (2) open-domain event extraction algorithms, capable of handling unconstrained event types, have largely overlooked the potential of large language models (LLMs) despite their advanced abilities. Additionally, they do not explicitly model document-level contextual, structural, and semantic reasoning, which are crucial for effective event extraction but remain challenging for LLMs due to lost-in-the-middle phenomenon and attention dilution. To address these limitations, we propose multimodal open-domain event extraction, MODEE , a novel approach for open-domain event extraction that combines graph-based learning with text-based representation from LLMs to model document-level reasoning. Empirical evaluations on large datasets demonstrate that MODEE outperforms state-of-the-art open-domain event extraction approaches and can be generalized to closed-domain event extraction, where it outperforms existing algorithms. 

---
# Revisiting Non-Verbatim Memorization in Large Language Models: The Role of Entity Surface Forms 

**Authors**: Yuto Nishida, Naoki Shikoda, Yosuke Kishinami, Ryo Fujii, Makoto Morishita, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2604.21882)  

**Abstract**: Understanding what kinds of factual knowledge large language models (LLMs) memorize is essential for evaluating their reliability and limitations. Entity-based QA is a common framework for analyzing non-verbatim memorization, but typical evaluations query each entity using a single canonical surface form, making it difficult to disentangle fact memorization from access through a particular name. We introduce RedirectQA, an entity-based QA dataset that uses Wikipedia redirect information to associate Wikidata factual triples with categorized surface forms for each entity, including alternative names, abbreviations, spelling variants, and common erroneous forms. Across 13 LLMs, we examine surface-conditioned factual memorization and find that prediction outcomes often change when only the entity surface form changes. This inconsistency is category-dependent: models are more robust to minor orthographic variations than to larger lexical variations such as aliases and abbreviations. Frequency analyses further suggest that both entity- and surface-level frequencies are associated with accuracy, and that entity frequency often contributes beyond surface frequency. Overall, factual memorization appears neither purely surface-specific nor fully surface-invariant, highlighting the importance of surface-form diversity in evaluating non-verbatim memorization. 

---
# Machine Behavior in Relational Moral Dilemmas: Moral Rightness, Predicted Human Behavior, and Model Decisions 

**Authors**: Jiseon Kim, Jea Kwon, Luiz Felipe Vecchietti, Wenchao Dong, Jaehong Kim, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2604.21871)  

**Abstract**: Human moral judgment is context-dependent and modulated by interpersonal relationships. As large language models (LLMs) increasingly function as decision-support systems, determining whether they encode these social nuances is critical. We characterize machine behavior using the Whistleblower's Dilemma by varying two experimental dimensions: crime severity and relational closeness. Our study evaluates three distinct perspectives: (1) moral rightness (prescriptive norms), (2) predicted human behavior (descriptive social expectations), and (3) autonomous model decision-making. By analyzing the reasoning processes, we identify a clear cross-perspective divergence: while moral rightness remains consistently fairness-oriented, predicted human behavior shifts significantly toward loyalty as relational closeness increases. Crucially, model decisions align with moral rightness judgments rather than their own behavioral predictions. This inconsistency suggests that LLM decision-making prioritizes rigid, prescriptive rules over the social sensitivity present in their internal world-modeling, which poses a gap that may lead to significant misalignments in real-world deployments. 

---
# SemEval-2026 Task 4: Narrative Story Similarity and Narrative Representation Learning 

**Authors**: Hans Ole Hatzel, Ekaterina Artemova, Haimo Paul Stiemer, Evelyn Gius, Chris Biemann  

**Link**: [PDF](https://arxiv.org/pdf/2604.21782)  

**Abstract**: We present the shared task on narrative similarity and narrative representation learning - NSNRL (pronounced "nass-na-rel"). The task operationalizes narrative similarity as a binary classification problem: determining which of two stories is more similar to an anchor story. We introduce a novel definition of narrative similarity, compatible with both narrative theory and intuitive judgment. Based on the similarity judgments collected under this concept, we also evaluate narrative embedding representations. We collected at least two annotations each for more than 1,000 story summary triples, with each annotation being backed by at least two annotators in agreement. This paper describes the sampling and annotation process for the dataset; further, we give an overview of the submitted systems and the techniques they employ. We received a total of 71 final submissions from 46 teams across our two tracks. In our triple-based classification setup, LLM ensembles make up many of the top-scoring systems, while in the embedding setup, systems with pre- and post-processing on pretrained embedding models perform about on par with custom fine-tuned solutions. Our analysis identifies potential headroom for improvement of automated systems in both tracks. The task website includes visualizations of embeddings alongside instance-level classification results for all teams. 

---
# Misinformation Span Detection in Videos via Audio Transcripts 

**Authors**: Breno Matos, Rennan C. Lima, Savvas Zannettou, Fabricio Benevenuto, Rodrygo L.T. Santos  

**Link**: [PDF](https://arxiv.org/pdf/2604.21767)  

**Abstract**: Online misinformation is one of the most challenging issues lately, yielding severe consequences, including political polarization, attacks on democracy, and public health risks. Misinformation manifests in any platform with a large user base, including online social networks and messaging apps. It permeates all media and content forms, including images, text, audio, and video. Distinctly, video-based misinformation represents a multifaceted challenge for fact-checkers, given the ease with which individuals can record and upload videos on various video-sharing platforms. Previous research efforts investigated detecting video-based misinformation, focusing on whether a video shares misinformation or not on a video level. While this approach is useful, it only provides a limited and non-easily interpretable view of the problem given that it does not provide an additional context of when misinformation occurs within videos and what content (i.e., claims) are responsible for the video's misinformation nature.
In this work, we attempt to bridge this research gap by creating two novel datasets that allow us to explore misinformation detection on videos via audio transcripts, focusing on identifying the span of videos that are responsible for the video's misinformation claim (misinformation span detection). We present two new datasets for this task. We transcribe each video's audio to text, identifying the video segment in which the misinformation claims appears, resulting in two datasets of more than 500 videos with over 2,400 segments containing annotated fact-checked claims. Then, we employ classifiers built with state-of-the-art language models, and our results show that we can identify in which part of a video there is misinformation with an F1 score of 0.68. We make publicly available our annotated datasets. We also release all transcripts, audio and videos. 

---
# AUDITA: A New Dataset to Audit Humans vs. AI Skill at Audio QA 

**Authors**: Tasnim Kabir, Dmytro Kurdydyk, Aadi Palnitkar, Liam Dorn, Ahmed Haj Ahmed, Jordan Lee Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2604.21766)  

**Abstract**: Existing audio question answering benchmarks largely emphasize sound event classification or caption-grounded queries, often enabling models to succeed through shortcut strategies, short-duration cues, lexical priors, dataset-specific biases, or even bypassing audio via metadata and captions rather than genuine reasoning Thus, we present AUDITA (Audio Understanding from Diverse Internet Trivia Authors), a large-scale, real-world benchmark to rigorously evaluate audio reasoning beyond surface-level acoustic recognition. AUDITA comprises carefully curated, human-authored trivia questions grounded in real-world audio, designed to stress robust auditory reasoning through challenging distractors and long-range temporal dependencies, using probing queries that cannot be answered from isolated text or sound cues alone. Human average accuracy of 32.13% shows both the challenge of the task while demonstrating meaningful comprehension of the audio. In stark contrast, state of-the-art audio question answering models perform poorly, with average accuracy below 8.86%. Beyond raw accuracy, we apply Item Response Theory (IRT) to estimate latent proficiency, question difficulty, and expose systematic deficiencies of the models and data. 

---
# Why are all LLMs Obsessed with Japanese Culture? On the Hidden Cultural and Regional Biases of LLMs 

**Authors**: Joseba Fernandez de Landa, Carla Perez-Almendros, Jose Camacho-Collados  

**Link**: [PDF](https://arxiv.org/pdf/2604.21751)  

**Abstract**: LLMs have been showing limitations when it comes to cultural coverage and competence, and in some cases show regional biases such as amplifying Western and Anglocentric viewpoints. While there have been works analysing the cultural capabilities of LLMs, there has not been specific work on highlighting LLM regional preferences when it comes to cultural-related questions. In this work, we propose a new dataset based on a comprehensive taxonomy of Culture-Related Open Questions (CROQ). The results show that, contrary to previous cultural bias work, LLMs show a clear tendency towards countries such as Japan. Moveover, our results show that when prompting in languages such as English or other high-resource ones, LLMs tend to provide more diverse outputs and show less inclinations towards answering questions highlighting countries for which the input language is an official language. Finally, we also investigate at which point of LLM training this cultural bias emerges, with our results suggesting that the first clear signs appear after supervised fine-tuning, and not during pre-training. 

---
# StructMem: Structured Memory for Long-Horizon Behavior in LLMs 

**Authors**: Buqiang Xu, Yijun Chen, Jizhan Fang, Ruobin Zhong, Yunzhi Yao, Yuqi Zhu, Lun Du, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.21748)  

**Abstract**: Long-term conversational agents need memory systems that capture relationships between events, not merely isolated facts, to support temporal reasoning and multi-hop question answering. Current approaches face a fundamental trade-off: flat memory is efficient but fails to model relational structure, while graph-based memory enables structured reasoning at the cost of expensive and fragile construction. To address these issues, we propose \textbf{StructMem}, a structure-enriched hierarchical memory framework that preserves event-level bindings and induces cross-event connections. By temporally anchoring dual perspectives and performing periodic semantic consolidation, StructMem improves temporal reasoning and multi-hop performance on \texttt{LoCoMo}, while substantially reducing token usage, API calls, and runtime compared to prior memory systems, see this https URL . 

---
# AEL: Agent Evolving Learning for Open-Ended Environments 

**Authors**: Wujiang Xu, Jiaojiao Han, Minghao Guo, Kai Mei, Xi Zhu, Han Zhang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2604.21725)  

**Abstract**: LLM agents increasingly operate in open-ended environments spanning hundreds of sequential episodes, yet they remain largely stateless: each task is solved from scratch without converting past experience into better future behavior. The central obstacle is not \emph{what} to remember but \emph{how to use} what has been remembered, including which retrieval policy to apply, how to interpret prior outcomes, and when the current strategy itself must change. We introduce \emph{Agent Evolving Learning} (\ael{}), a two-timescale framework that addresses this obstacle. At the fast timescale, a Thompson Sampling bandit learns which memory retrieval policy to apply at each episode; at the slow timescale, LLM-driven reflection diagnoses failure patterns and injects causal insights into the agent's decision prompt, giving it an interpretive frame for the evidence it retrieves. On a sequential portfolio benchmark (10 sector-diverse tickers, 208 episodes, 5 random seeds), \ael{} achieves a Sharpe ratio of 2.13$\pm$0.47, outperforming five published self-improving methods and all non-LLM baselines while maintaining the lowest variance among all LLM-based approaches. A nine-variant ablation reveals a ``less is more'' pattern: memory and reflection together produce a 58\% cumulative improvement over the stateless baseline, yet every additional mechanism we test (planner evolution, per-tool selection, cold-start initialization, skill extraction, and three credit assignment methods) \emph{degrades} performance. This demonstrates that the bottleneck in agent self-improvement is \emph{self-diagnosing how to use} experience rather than adding architectural complexity. Code and data: this https URL. 

---
# Beyond N-gram: Data-Aware X-GRAM Extraction for Efficient Embedding Parameter Scaling 

**Authors**: Yilong Chen, Yanxi Xie, Zitian Gao, He Xin, Yihao Xiao, Renbiao Liu, Haoming Luo, Yifan Luo, Zhengmao Ye, Tingwen Liu, Xin Zhao, Ran Tao, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2604.21724)  

**Abstract**: Large token-indexed lookup tables provide a compute-decoupled scaling path, but their practical gains are often limited by poor parameter efficiency and rapid memory growth. We attribute these limitations to Zipfian under-training of the long tail, heterogeneous demand across layers, and "slot collapse" that produces redundant embeddings. To address this, we propose X-GRAM, a frequency-aware dynamic token-injection framework. X-GRAM employs hybrid hashing and alias mixing to compress the tail while preserving head capacity, and refines retrieved vectors via normalized SwiGLU ShortConv to extract diverse local n-gram features. These signals are integrated into attention value streams and inter-layer residuals using depth-aware gating, effectively aligning static memory with dynamic context. This design introduces a memory-centric scaling axis that decouples model capacity from FLOPs. Extensive evaluations at the 0.73B and 1.15B scales show that X-GRAM improves average accuracy by as much as 4.4 points over the vanilla backbone and 3.2 points over strong retrieval baselines, while using substantially smaller tables in the 50% configuration. Overall, by decoupling capacity from compute through efficient memory management, X-GRAM offers a scalable and practical paradigm for future memory-augmented architectures. Code aviliable in this https URL. 

---
# From If-Statements to ML Pipelines: Revisiting Bias in Code-Generation 

**Authors**: Minh Duc Bui, Xenia Heilmann, Mattia Cerrato, Manuel Mager, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2604.21716)  

**Abstract**: Prior work evaluates code generation bias primarily through simple conditional statements, which represent only a narrow slice of real-world programming and reveal solely overt, explicitly encoded bias. We demonstrate that this approach dramatically underestimates bias in practice by examining a more realistic task: generating machine learning (ML) pipelines. Testing both code-specialized and general-instruction large language models, we find that generated pipelines exhibit significant bias during feature selection. Sensitive attributes appear in 87.7% of cases on average, despite models demonstrably excluding irrelevant features (e.g., including "race" while dropping "favorite color" for credit scoring). This bias is substantially more prevalent than that captured by conditional statements, where sensitive attributes appear in only 59.2% of cases. These findings are robust across prompt mitigation strategies, varying numbers of attributes, and different pipeline difficulty levels. Our results challenge simple conditionals as valid proxies for bias evaluation and suggest current benchmarks underestimate bias risk in practical deployments. 

---
# Phonological Subspace Collapse Is Aetiology-Specific and Cross-Lingually Stable: Evidence from 3,374 Speakers 

**Authors**: Bernard Muller, Antonio Armando Ortiz Barrañón, LaVonne Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2604.21706)  

**Abstract**: We previously introduced a training-free method for dysarthria severity assessment based on d-prime separability of phonological feature subspaces in frozen self-supervised speech representations, validated on 890 speakers across 5 languages with HuBERT-base. Here, we scale the analysis to 3,374 speakers from 25 datasets spanning 12 languages and 5 aetiologies (Parkinson's disease, cerebral palsy, ALS, Down syndrome, and stroke), plus healthy controls, using 6 SSL backbones. We report three findings. First, aetiology-specific degradation profiles are distinguishable at the group level: 10 of 13 features yield large effect sizes (epsilon-squared > 0.14, Holm-corrected p < 0.001), with Parkinson's disease separable from the articulatory execution group at Cohen's d = 0.83; individual-level classification remains limited (22.6% macro F1). Second, profiles show cross-lingual profile-shape stability: cosine similarity of 5-dimensional consonant d-prime profiles exceeds 0.95 across the languages available for each aetiology. Absolute d-prime magnitudes are not cross-lingually calibrated, so the method supports language-independent phenotyping of degradation patterns but requires within-corpus calibration for absolute severity interpretation. Third, the method is architecture-independent: all 6 backbones produce monotonic severity gradients with inter-model agreement exceeding rho = 0.77. Fixed-token d-prime estimation preserves the severity correlation (rho = -0.733 at 200 tokens per class), confirming that the signal is not a token-count artefact. These results support phonological subspace analysis as a robust, training-free framework for aetiology-aware dysarthria characterisation, with evidence of cross-lingual profile-shape stability and cross-backbone robustness in the represented sample. 

---
# Fixation Sequences as Time Series: A Topological Approach to Dyslexia Detection 

**Authors**: Marius Huber, David R. Reich, Lena A. Jäger  

**Link**: [PDF](https://arxiv.org/pdf/2604.21698)  

**Abstract**: Persistent homology, a method from topological data analysis, extracts robust, multi-scale features from data. It produces stable representations of time series by applying varying thresholds to their values (a process known as a \textit{filtration}). We develop novel filtrations for time series and introduce topological methods for the analysis of eye-tracking data, by interpreting fixation sequences as time series, and constructing ``hybrid models'' that combine topological features with traditional statistical features. We empirically evaluate our method by applying it to the task of dyslexia detection from eye-tracking-while-reading data using the Copenhagen Corpus, which contains scanpaths from dyslexic and non-dyslexic L1 and L2 readers. Our hybrid models outperform existing approaches that rely solely on traditional features, showing that persistent homology captures complementary information encoded in fixation sequences. The strength of these topological features is further underscored by their achieving performance comparable to established baseline methods. Importantly, our proposed filtrations outperform existing ones. 

---
# Fine-Grained Perspectives: Modeling Explanations with Annotator-Specific Rationales 

**Authors**: Olufunke O. Sarumi, Charles Welch, Daniel Braun  

**Link**: [PDF](https://arxiv.org/pdf/2604.21667)  

**Abstract**: Beyond exploring disaggregated labels for modeling perspectives, annotator rationales provide fine-grained signals of individual perspectives. In this work, we propose a framework for jointly modeling annotator-specific label prediction and corresponding explanations, fine-tuned on the annotators' provided rationales. Using a dataset with disaggregated natural language inference (NLI) annotations and annotator-provided explanations, we condition predictions on both annotator identity and demographic metadata through a representation-level User Passport mechanism. We further introduce two explainer architectures: a post-hoc prompt-based explainer and a prefixed bridge explainer that transfers annotator-conditioned classifier representations directly into a generative model. This design enables explanation generation aligned with individual annotator perspectives. Our results show that incorporating explanation modeling substantially improves predictive performance over a baseline annotator-aware classifier, with the prefixed bridge approach achieving more stable label alignment and higher semantic consistency, while the post-hoc approach yields stronger lexical similarity. These findings indicate that modeling explanations as expressions of fine-grained perspective provides a richer and more faithful representation of disagreement. The proposed approaches advance perspectivist modeling by integrating annotator-specific rationales into both predictive and generative components. 

---
# Multilinguality at the Edge: Developing Language Models for the Global South 

**Authors**: Lester James V. Miranda, Songbo Hu, Roi Reichart, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2604.21637)  

**Abstract**: Where and how language models (LMs) are deployed determines who can benefit from them. However, there are several challenges that prevent effective deployment of LMs in non-English-speaking and hardware constrained communities in the Global South. We call this challenge the last mile: the intersection of multilinguality and edge deployment, where the goals are aligned but the technical requirements often compete. Studying these two fields together is both a need, as linguistically diverse communities often face the most severe infrastructure constraints, and an opportunity, as edge and multilingual NLP research remain largely siloed. To understand the state of the art and the challenges of combining the two areas, we survey 232 papers that tackle this problem across the language modelling pipeline, from data collection to development and deployment. We also discuss open questions and provide actionable recommendations for different stakeholders in the NLP ecosystem. Finally, we hope that this work contributes to the development of inclusive and equitable language technologies. 

---
# Process Supervision via Verbal Critique Improves Reasoning in Large Language Models 

**Authors**: Hao-Yuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.21611)  

**Abstract**: Inference-time scaling for LLM reasoning has focused on three axes: chain depth, sample breadth, and learned step-scorers (PRMs). We introduce a fourth axis, granularity of external verbal supervision, via Verbal Process Supervision (VPS), a training-free framework that uses structured natural-language critique from a stronger supervisor to guide an iterative generate-critique-refine loop up to a round budget R. Across GPQA Diamond, AIME 2025, and LiveCodeBench V6 (covering both closed and open models), VPS yields three key results. First, on GPQA Diamond, GPT-5.4 (High) | GPT-5.4 (Low) reaches 94.9% at R=4, surpassing the 94.1% state of the art without gradient updates. Second, on AIME 2025, VPS enables strong weak-actor rescue, boosting scores from 11.7-26.7% to 63.3-90.0% (up to +63.3 points). Third, at matched compute, VPS outperforms Reflexion by +8.5 to +12.1 points and Self-Consistency@5 by +5.0 pp (GPQA) and +8.3 pp (LiveCodeBench), isolating critique granularity as the key driver. Performance scales with the supervisor-actor capability gap (Pearson r=0.90) and degrades when errors are not linguistically expressible (e.g., code synthesis), motivating hybrid verbal-executable methods. These results establish critique granularity as a new axis of inference-time scaling. 

---
# Language as a Latent Variable for Reasoning Optimization 

**Authors**: Linjuan Wu, Haoran Wei, Jialong Tang, Shuang Luo, Baosong Yang, Yongliang Shen, Weiming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.21593)  

**Abstract**: As LLMs reduce English-centric bias, a surprising trend emerges: non-English responses sometimes outperform English on reasoning tasks. We hypothesize that language functions as a latent variable that structurally modulates the model's internal inference pathways, rather than merely serving as an output medium. To test this, we conducted a Polyglot Thinking Experiment, in which models were prompted to solve identical problems under language-constrained and language-unconstrained conditions. Results show that non-English responses often achieve higher accuracy, and the best performance frequently occur when language is unconstrained, suggesting that multilinguality broadens the model's latent reasoning space. Based on this insight, we propose polyGRPO (Polyglot Group Relative Policy Optimization), an RL framework that treats language variation as an implicit exploration signal. It generates polyglot preference data online under language-constrained and unconstrained conditions, optimizing the policy with respect to both answer accuracy and reasoning structure. Trained on only 18.1K multilingual math problems without chain-of-thought annotations, polyGRPO improves the base model (Qwen2.5-7B-Instruct) by 6.72% absolute accuracy on four English reasoning testset and 6.89% in their multilingual benchmark. Remarkably, it is the only method that surpasses the base LLM on English commonsense reasoning task (4.9%), despite being trained solely on math data-highlighting its strong cross-task generalization. Further analysis reveals that treating language as a latent variable expands the model's latent reasoning space, yielding consistent and generalizable improvements in reasoning performance. 

---
# AgenticQwen: Training Small Agentic Language Models with Dual Data Flywheels for Industrial-Scale Tool Use 

**Authors**: Yuanjie Lyu, Chengyu Wang, Haonan Zheng, Yuanhao Yue, Junbing Yan, Ming Wang, Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.21590)  

**Abstract**: Modern industrial applications increasingly demand language models that act as agents, capable of multi-step reasoning and tool use in real-world settings. These tasks are typically performed under strict cost and latency constraints, making small agentic models highly desirable. In this paper, we introduce the AgenticQwen family of models, trained via multi-round reinforcement learning (RL) on synthetic data and a limited amount of open-source data. Our training framework combines reasoning RL and agentic RL with dual data flywheels that automatically generate increasingly challenging tasks. The reasoning flywheel increases task difficulty by learning from errors, while the agentic flywheel expands linear workflows into multi-branch behavior trees that better reflect the decision complexity of real-world applications. We validate AgenticQwen on public benchmarks and in an industrial agent system. The models achieve strong performance on multiple agentic benchmarks, and in our industrial agent system, close the gap with much larger models on search and data analysis tasks. Model checkpoints and part of the synthetic data: this https URL. Data synthesis and RL training code: this https URL. The data synthesis pipeline is also integrated into EasyDistill: this https URL. 

---
# Measuring Opinion Bias and Sycophancy via LLM-based Coercion 

**Authors**: Rodrigo Nogueira, Giovana Kerche Bonás, Thales Sales Almeida, Andrea Roque, Ramon Pires, Hugo Abonizio, Thiago Laitz, Celio Larcher, Roseval Malaquias Junior, Marcos Piau  

**Link**: [PDF](https://arxiv.org/pdf/2604.21564)  

**Abstract**: Large language models increasingly shape the information people consume: they are embedded in search, consulted for professional advice, deployed as agents, and used as a first stop for questions about policy, ethics, health, and politics. When such a model silently holds a position on a contested topic, that position propagates at scale into users' decisions. Eliciting a model's positions is harder than it first appears: contemporary assistants answer direct opinion questions with evasive disclaimers, and the same model may concede the opposite position once the user starts arguing one side. We propose a method, released as the open-source llm-bias-bench, for discovering the opinions an LLM actually holds on contested topics under conditions that resemble real multi-turn interaction. The method pairs two complementary free-form probes. Direct probing asks for the model's opinion across five turns of escalating pressure from a simulated user. Indirect probing never asks for an opinion and engages the model in argumentative debate, letting bias leak through how it concedes, resists, or counter-argues. Three user personas (neutral, agree, disagree) collapse into a nine-way behavioral classification that separates persona-independent positions from persona-dependent sycophancy, and an auditable LLM judge produces verdicts with textual evidence. The first instantiation ships 38 topics in Brazilian Portuguese across values, scientific consensus, philosophy, and economic policy. Applied to 13 assistants, the method surfaces findings of practical interest: argumentative debate triggers sycophancy 2-3x more than direct questioning (median 50% to 79%); models that look opinionated under direct questioning often collapse into mirroring under sustained arguments; and attacker capability matters mainly when an existing opinion must be dislodged, not when the assistant starts neutral. 

---
# Finding Meaning in Embeddings: Concept Separation Curves 

**Authors**: Paul Keuren, Marc Ponsen, Robert Ayoub Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2604.21555)  

**Abstract**: Sentence embedding techniques aim to encode key concepts of a sentence's meaning in a vector space. However, the majority of evaluation approaches for sentence embedding quality rely on the use of additional classifiers or downstream tasks. These additional components make it unclear whether good results stem from the embedding itself or from the classifier's behaviour. In this paper, we propose a novel method for evaluating the effectiveness of sentence embedding methods in capturing sentence-level concepts. Our approach is classifier-independent, allowing for an objective assessment of the model's performance. The approach adopted in this study involves the systematic introduction of syntactic noise and semantic negations into sentences, with the subsequent quantification of their relative effects on the resulting embeddings. The visualisation of these effects is facilitated by Concept Separation Curves, which show the model's capacity to differentiate between conceptual and surface-level variations. By leveraging data from multiple domains, employing both Dutch and English languages, and examining sentence lengths, this study offers a compelling demonstration that Concept Separation Curves provide an interpretable, reproducible, and cross-model approach for evaluating the conceptual stability of sentence embeddings. 

---
# UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text 

**Authors**: Darya Hryhoryeva, Amaia Zurinaga, Hamidreza Jamalabadi, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2604.21534)  

**Abstract**: This paper presents our system developed for SemEval-2026 Task 2. The task requires modeling both current affect and short-term affective change in chronologically ordered user-generated texts. We explore three complementary approaches: (1) LLM prompting under user-aware and user-agnostic settings, (2) a pairwise Maximum Entropy (MaxEnt) model with Ising-style interactions for structured transition modeling, and (3) a lightweight neural regression model incorporating recent affective trajectories and trainable user embeddings. Our findings indicate that LLMs effectively capture static affective signals from text, whereas short-term affective variation in this dataset is more strongly explained by recent numeric state trajectories than by textual semantics. Our system ranked first among participating teams in both Subtask 1 and Subtask 2A based on the official evaluation metric. 

---
# Job Skill Extraction via LLM-Centric Multi-Module Framework 

**Authors**: Guojing Li, Zichuan Fu, Junyi Li, Faxue Liu, Wenxia Zhou, Yejing Wang, Jingtong Gao, Maolin Wang, Rungen Liu, Wenlin Zhang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.21525)  

**Abstract**: Span-level skill extraction from job advertisements underpins candidate-job matching and labor-market analytics, yet generative large language models (LLMs) often yield malformed spans, boundary drift, and hallucinations, especially with long-tail terms and cross-domain shift. We present SRICL, an LLM-centric framework that combines semantic retrieval (SR), in-context learning (ICL), and supervised fine-tuning (SFT) with a deterministic verifier. SR pulls in-domain annotated sentences and definitions from ESCO to form format-constrained prompts that stabilize boundaries and handle coordination. SFT aligns output behavior, while the verifier enforces pairing, non-overlap, and BIO legality with minimal retries. On six public span-labeled corpora of job-ad sentences across sectors and languages, SRICL achieves substantial STRICT-F1 improvements over GPT-3.5 prompting baselines and sharply reduces invalid tags and hallucinated spans, enabling dependable sentence-level deployment in low-resource, multi-domain settings. 

---
# OptiVerse: A Comprehensive Benchmark towards Optimization Problem Solving 

**Authors**: Xinyu Zhang, Boxuan Zhang, Yuchen Wan, Lingling Zhang, YiXing Yao, Bifan Wei, Yaqiang Wu, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.21510)  

**Abstract**: While Large Language Models (LLMs) demonstrate remarkable reasoning, complex optimization tasks remain challenging, requiring domain knowledge and robust implementation. However, existing benchmarks focus narrowly on Mathematical Programming and Combinatorial Optimization, hindering comprehensive evaluation. To address this, we introduce OptiVerse, a comprehensive benchmark of 1,000 curated problems spanning neglected domains, including Stochastic Optimization, Dynamic Optimization, Game Optimization, and Optimal Control, across three difficulty levels: Easy, Medium, and Hard. The experiments with 22 LLMs of different sizes reveal sharp performance degradation on hard problems, where even advanced models like GPT-5.2 and Gemini-3 struggle to exceed 27% accuracy. Through error analysis, we identify that modeling & logic errors remain the primary bottleneck. Consequently, we propose a Dual-View Auditor Agent that improves the accuracy of the LLM modeling process without introducing significant time overhead. OptiVerse will serve as a foundational platform for advancing LLMs in solving complex optimization challenges. 

---
# Preferences of a Voice-First Nation: Large-Scale Pairwise Evaluation and Preference Analysis for TTS in Indian Languages 

**Authors**: Srija Anand, Ashwin Sankar, Ishvinder Sethi, Aaditya Pareek, Kartik Rajput, Gaurav Yadav, Nikhil Narasimhan, Adish Pandya, Deepon Halder, Mohammed Safi Ur Rahman Khan, Praveen S V, Shobhit Banga, Mitesh M Khapra  

**Link**: [PDF](https://arxiv.org/pdf/2604.21481)  

**Abstract**: Crowdsourced pairwise evaluation has emerged as a scalable approach for assessing foundation models. However, applying it to Text to Speech(TTS) introduces high variance due to linguistic diversity and multidimensional nature of speech perception. We present a controlled multidimensional pairwise evaluation framework for multilingual TTS that combines linguistic control with perceptually grounded annotation. Using 5K+ native and code-mixed sentences across 10 Indic languages, we evaluate 7 state-of-the-art TTS systems and collect over 120K pairwise comparisons from over 1900 native raters. In addition to overall preference, raters provide judgments across 6 perceptual dimensions: intelligibility, expressiveness, voice quality, liveliness, noise, and hallucinations. Using Bradley-Terry modeling, we construct a multilingual leaderboard, interpret human preference using SHAP analysis and analyze leaderboard reliability alongside model strengths and trade-offs across perceptual dimensions. 

---
# Cross-Domain Data Selection and Augmentation for Automatic Compliance Detection 

**Authors**: Fariz Ikhwantri, Dusica Marijan  

**Link**: [PDF](https://arxiv.org/pdf/2604.21469)  

**Abstract**: Automating the detection of regulatory compliance remains a challenging task due to the complexity and variability of legal texts. Models trained on one regulation often fail to generalise to others. This limitation underscores the need for principled methods to improve cross-domain transfer. We study data selection as a strategy to mitigate negative transfer in compliance detection framed as a natural language inference (NLI) task. Specifically, we evaluate four approaches for selecting augmentation data from a larger source domain: random sampling, Moore-Lewis's cross-entropy difference, importance weighting, and embedding-based retrieval. We systematically vary the proportion of selected data to analyse its effect on cross-domain adaptation. Our findings demonstrate that targeted data selection substantially reduces negative transfer, offering a practical path toward scalable and reliable compliance automation across heterogeneous regulations. 

---
# Reasoning Primitives in Hybrid and Non-Hybrid LLMs 

**Authors**: Shivam Rawat, Lucie Flek, Florian Mai, Nicholas Kluge Corrêa  

**Link**: [PDF](https://arxiv.org/pdf/2604.21454)  

**Abstract**: Reasoning in large language models is often treated as a monolithic capability, but its observed gains may arise from more basic operations. We study reasoning through two such primitives, recall and state-tracking, and ask whether hybrid architectures that combine attention-based retrieval with recurrent state updates are better suited than attention-only models for tasks that jointly require both. Using matched Olmo3 transformer and hybrid models in instruction-tuned and reasoning-augmented variants, we evaluate these models on a set of controlled tasks involving a mixture of state-tracking and recall primitives, state-based recall. Across tasks, we notice that reasoning augmentation provides the largest overall improvement, substantially extending the range of difficulty over which models remain effective. We also notice that in certain tasks, the hybrid reasoning model remains substantially more robust as sequential dependence increases. In contrast, the transformer reasoning model degrades sharply in performance as task difficulty increases beyond a given threshold. These results suggest that reasoning tokens and architectural inductive biases contribute at different levels of the computational process: explicit reasoning can expand a model's effective operating range, but its benefit depends on how well the underlying architecture supports persistent state propagation. Given the small size of our case study, which involves a limited set of models and tasks, we present these findings as suggestive rather than conclusive and leave broader validation across model families, scales, and task variations to future work. 

---
# Decoupled DiLoCo for Resilient Distributed Pre-training 

**Authors**: Arthur Douillard, Keith Rush, Yani Donchev, Zachary Charles, Nova Fallen, Ayush Dubey, Ionel Gog, Josef Dean, Blake Woodworth, Zachary Garrett, Nate Keating, Jenny Bishop, Henry Prior, Edouard Yvinec, Arthur Szlam, Marc'Aurelio Ranzato, Jeff Dean  

**Link**: [PDF](https://arxiv.org/pdf/2604.21428)  

**Abstract**: Modern large-scale language model pre-training relies heavily on the single program multiple data (SPMD) paradigm, which requires tight coupling across accelerators. Due to this coupling, transient slowdowns, hardware failures, and synchronization overhead stall the entire computation, wasting significant compute time at scale. While recent distributed methods like DiLoCo reduced communication bandwidth, they remained fundamentally synchronous and vulnerable to these system stalls. To address this, we introduce Decoupled DiLoCo, an evolution of the DiLoCo framework designed to break the lock-step synchronization barrier and go beyond SPMD to maximize training goodput. Decoupled DiLoCo partitions compute across multiple independent ``learners'' that execute local inner optimization steps. These learners asynchronously communicate parameter fragments to a central synchronizer, which circumvents failed or straggling learners by aggregating updates using a minimum quorum, an adaptive grace window, and dynamic token-weighted merging. Inspired by ``chaos engineering'', we achieve significantly improved training efficiency in failure-prone environments with millions of simulated chips with strictly zero global downtime, while maintaining competitive model performance across text and vision tasks, for both dense and mixture-of-expert architectures. 

---
# VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation 

**Authors**: Qijun Han, Haoqin Tu, Zijun Wang, Haoyue Dai, Yiyang Zhou, Nancy Lau, Alvaro A. Cardenas, Yuhui Xu, Ran Xu, Caiming Xiong, Zeyu Zheng, Huaxiu Yao, Yuyin Zhou, Cihang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.21375)  

**Abstract**: Autonomous GUI agents face two fundamental challenges: early stopping, where agents prematurely declare success without verifiable evidence, and repetitive loops, where agents cycle through the same failing actions without recovery. We present VLAA-GUI, a modular GUI agentic framework built around three integrated components that guide the system on when to Stop, Recover, and Search. First, a mandatory Completeness Verifier enforces UI-observable success criteria and verification at every finish step -- with an agent-level verifier that cross-examines completion claims with decision rules, rejecting those lacking direct visual evidence. Second, a mandatory Loop Breaker provides multi-tier filtering: switching interaction mode after repeated failures, forcing strategy changes after persistent screen-state recurrence, and binding reflection signals to strategy shifts. Third, an on-demand Search Agent searches online for unfamiliar workflows by directly querying a capable LLM with search ability, returning results as plain text. We additionally integrate a Coding Agent for code-intensive actions and a Grounding Agent for precise action grounding, both invoked on demand when required. We evaluate VLAA-GUI across five top-tier backbones, including Opus 4.5, 4.6 and Gemini 3.1 Pro, on two benchmarks with Linux and Windows tasks, achieving top performance on both (77.5% on OSWorld and 61.0% on WindowsAgentArena). Notably, three of the five backbones surpass human performance (72.4%) on OSWorld in a single pass. Ablation studies show that all three proposed components consistently improve a strong backbone, while a weaker backbone benefits more from these tools when the step budget is sufficient. Further analysis also shows that the Loop Breaker nearly halves wasted steps for loop-prone models. 

---
# MKJ at SemEval-2026 Task 9: A Comparative Study of Generalist, Specialist, and Ensemble Strategies for Multilingual Polarization 

**Authors**: Maziar Kianimoghadam Jouneghani  

**Link**: [PDF](https://arxiv.org/pdf/2604.21370)  

**Abstract**: We present a systematic study of multilingual polarization detection across 22 languages for SemEval-2026 Task 9 (Subtask 1), contrasting multilingual generalists with language-specific specialists and hybrid ensembles. While a standard generalist like XLM-RoBERTa suffices when its tokenizer aligns with the target text, it may struggle with distinct scripts (e.g., Khmer, Odia) where monolingual specialists yield significant gains. Rather than enforcing a single universal architecture, we adopt a language-adaptive framework that switches between multilingual generalists, language-specific specialists, and hybrid ensembles based on development performance. Additionally, cross-lingual augmentation via NLLB-200 yielded mixed results, often underperforming native architecture selection and degrading morphologically rich tracks. Our final system achieves an overall macro-averaged F1 score of 0.796 and an average accuracy of 0.826 across all 22 tracks. Code and final test predictions are publicly available at: this https URL. 

---
# CARE: Counselor-Aligned Response Engine for Online Mental-Health Support 

**Authors**: Hagai Astrin, Ayal Swaid, Avi Segal, Kobi Gal  

**Link**: [PDF](https://arxiv.org/pdf/2604.21352)  

**Abstract**: Mental health challenges are increasing worldwide, straining emotional support services and leading to counselor overload. This can result in delayed responses during critical situations, such as suicidal ideation, where timely intervention is essential. While large language models (LLMs) have shown strong generative capabilities, their application in low-resource languages, especially in sensitive domains like mental health, remains underexplored. Furthermore, existing LLM-based agents often struggle to replicate the supportive language and intervention strategies used by professionals due to a lack of training on large-scale, real-world datasets.
To address this, we propose CARE (Counselor-Aligned Response Engine), a GenAI framework that assists counselors by generating real-time, psychologically aligned response recommendations. CARE fine-tunes open-source LLMs separately for Hebrew and Arabic using curated subsets of real-world crisis conversations. The training data consists of sessions rated as highly effective by professional counselors, enabling the models to capture interaction patterns associated with successful de-escalation. By training on complete conversation histories, CARE maintains the evolving emotional context and dynamic structure of counselor-help-seeker dialogue.
In experimental settings, CARE demonstrates stronger semantic and strategic alignment with gold-standard counselor responses compared to non-specialized LLMs. These findings suggest that domain-specific fine-tuning on expert-validated data can significantly support counselor workflows and improve care quality in low-resource language contexts. 

---
# Beyond Single Plots: A Benchmark for Question Answering on Multi-Charts 

**Authors**: Azher Ahmed Efat, Seok Hwan Song, Wallapak Tavanapong  

**Link**: [PDF](https://arxiv.org/pdf/2604.21344)  

**Abstract**: Charts are widely used to present complex information. Deriving meaningful insights in real-world contexts often requires interpreting multiple related charts together. Research on understanding multi-chart images has not been extensively explored. We introduce PolyChartQA, a mid-scale dataset specifically designed for question answering over multi-chart images. PolyChartQA comprises 534 multi-chart images (with a total of 2,297 sub-charts) sourced from peer-reviewed computer science research publications and 2,694 QA pairs. We evaluate the performance of nine state-of-the-art Multimodal Language Models (MLMs) on PolyChartQA across question type, difficulty, question source, and key structural characteristics of multi-charts. Our results show a 27.4% LLM-based accuracy (L-Accuracy) drop on human-authored questions compared to MLM-generated questions, and a 5.39% L-accuracy gain with our proposed prompting method. 

---
# When Bigger Isn't Better: A Comprehensive Fairness Evaluation of Political Bias in Multi-News Summarisation 

**Authors**: Nannan Huang, Iffat Maab, Junichi Yamagishi  

**Link**: [PDF](https://arxiv.org/pdf/2604.21309)  

**Abstract**: Multi-document news summarisation systems are increasingly adopted for their convenience in processing vast daily news content, making fairness across diverse political perspectives critical. However, these systems can exhibit political bias through unequal representation of viewpoints, disproportionate emphasis on certain perspectives, and systematic underrepresentation of minority voices. This study presents a comprehensive evaluation of such bias in multi-document news summarisation using FairNews, a dataset of complete news articles with political orientation labels, examining how large language models (LLMs) handle sources with varying political leanings across 13 models and five fairness metrics. We investigate both baseline model performance and effectiveness of various debiasing interventions, including prompt-based and judge-based approaches. Our findings challenge the assumption that larger models yield fairer outputs, as mid-sized variants consistently outperform their larger counterparts, offering the best balance of fairness and efficiency. Prompt-based debiasing proves highly model dependent, while entity sentiment emerges as the most stubborn fairness dimension, resisting all intervention strategies tested. These results demonstrate that fairness in multi-document news summarisation requires multi-dimensional evaluation frameworks and targeted, architecture-aware debiasing rather than simply scaling up. 

---
# Explainable Disentangled Representation Learning for Generalizable Authorship Attribution in the Era of Generative AI 

**Authors**: Hieu Man, Van-Cuong Pham, Nghia Trung Ngo, Franck Dernoncourt, Thien Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.21300)  

**Abstract**: Learning robust representations of authorial style is crucial for authorship attribution and AI-generated text detection. However, existing methods often struggle with content-style entanglement, where models learn spurious correlations between authors' writing styles and topics, leading to poor generalization across domains. To address this challenge, we propose Explainable Authorship Variational Autoencoder (EAVAE), a novel framework that explicitly disentangles style from content through architectural separation-by-design. EAVAE first pretrains style encoders using supervised contrastive learning on diverse authorship data, then finetunes with a Variational Autoencoder (VEA) architecture using separate encoders for style and content representations. Disentanglement is enforced through a novel discriminator that not only distinguishes whether pairs of style/content representations belong to the same or different authors/content sources, but also generates natural language explanation for their decision, simultaneously mitigating confounding information and enhancing interpretability. Extensive experiments demonstrate the effectiveness of EAVAE. On authorship attribution, we achieve state-of-the-art performance on various datasets, including Amazon Reviews, PAN21, and HRS. For AI-generated text detection, EAVAE excels in few-shot learning over the M4 dataset. Code and data repositories are available online\footnote{this https URL} \footnote{this https URL}. 

---
# Cross-Entropy Is Load-Bearing: A Pre-Registered Scope Test of the K-Way Energy Probe on Bidirectional Predictive Coding 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.21286)  

**Abstract**: Cacioli (2026) showed that the K-way energy probe on standard discriminative predictive coding networks reduces approximately to a monotone function of the log-softmax margin. The reduction rests on five assumptions, including cross-entropy (CE) at the output and effectively feedforward inference dynamics. This pre-registered study tests the reduction's sensitivity to CE removal using two conditions: standard PC trained with MSE instead of CE, and bidirectional PC (bPC; Oliviers, Tang & Bogacz, 2025). Across 10 seeds on CIFAR-10 with a matched 2.1M-parameter backbone, we find three results. The negative result replicates on standard PC: the probe sits below softmax (Delta = -0.082, p < 10^-6). On bPC the probe exceeds softmax across all 10 seeds (Delta = +0.008, p = 0.000027), though a pre-registered manipulation check shows that bPC does not produce materially greater latent movement than standard PC at this scale (ratio 1.6, threshold 10). Removing CE alone without changing inference dynamics halves the probe-softmax gap (Delta_MSE = -0.037 vs Delta_stdPC = -0.082). CE is a major empirically load-bearing component of the decomposition at this scale. CE training produces output logit norms approximately 15x larger than MSE or bPC training. A post-hoc temperature scaling ablation decomposes the probe-softmax gap into two components: approximately 66% is attributable to logit-scale effects removable by temperature rescaling, and approximately 34% reflects a scale-invariant ranking advantage of CE-trained representations. We use "metacognitive" operationally to denote Type-2 discrimination of a readout over its own Type-1 correctness, not to imply human-like introspective access. 

---
# Do LLM Decoders Listen Fairly? Benchmarking How Language Model Priors Shape Bias in Speech Recognition 

**Authors**: Srishti Ginjala, Eric Fosler-Lussier, Christopher W. Myers, Srinivasan Parthasarathy  

**Link**: [PDF](https://arxiv.org/pdf/2604.21276)  

**Abstract**: As pretrained large language models replace task-specific decoders in speech recognition, a critical question arises: do their text-derived priors make recognition fairer or more biased across demographic groups? We evaluate nine models spanning three architectural generations (CTC with no language model, encoder-decoder with an implicit LM, and LLM-based with an explicit pretrained decoder) on about 43,000 utterances across five demographic axes (ethnicity, accent, gender, age, first language) using Common Voice 24 and Meta's Fair-Speech, a controlled-prompt dataset that eliminates vocabulary confounds. On clean audio, three findings challenge assumptions: LLM decoders do not amplify racial bias (Granite-8B has the best ethnicity fairness, max/min WER = 2.28); Whisper exhibits pathological hallucination on Indian-accented speech with a non-monotonic insertion-rate spike to 9.62% at large-v3; and audio compression predicts accent fairness more than LLM scale. We then stress-test these findings under 12 acoustic degradation conditions (noise, reverberation, silence injection, chunk masking) across both datasets, totaling 216 inference runs. Severe degradation paradoxically compresses fairness gaps as all groups converge to high WER, but silence injection amplifies Whisper's accent bias up to 4.64x by triggering demographic-selective hallucination. Under masking, Whisper enters catastrophic repetition loops (86% of 51,797 insertions) while explicit-LLM decoders produce 38x fewer insertions with near-zero repetition; high-compression audio encoding (Q-former) reintroduces repetition pathology even in LLM decoders. These results suggest that audio encoder design, not LLM scaling, is the primary lever for equitable and robust speech recognition. 

---
# Listen and Chant Before You Read: The Ladder of Beauty in LM Pre-Training 

**Authors**: Yoshinori Nomura  

**Link**: [PDF](https://arxiv.org/pdf/2604.21265)  

**Abstract**: We show that pre-training a Transformer on music before language significantly accelerates language acquisition. Using piano performances (MAESTRO dataset), a developmental pipeline -- music $\to$ poetry $\to$ prose -- yields a $17.5\%$ perplexity improvement over random initialization ($p < 0.001$, 5 seeds), with music and poetry improving orthogonal model components (internal computation and embeddings, respectively). Convergence tests confirm that this is not a transient head start: at $d\!=\!64$, multi-seed validation (5 seeds) shows a persistent 5.5\% gap at plateau ($p = 0.017$), with the pipeline converging faster and to a lower loss in every run. Real music matches the transfer ceiling of synthetic patterns with one-third the data, and scaling experiments reveal that optimal pre-training data volume shifts with model capacity ($-3\% \to +3\% \to +6\%$ advantage of larger datasets from $d\!=\!16$ to $d\!=\!64$). Across the scales we study ($d\!\in\!\{16,32,64\}$, up to ${\sim}400$K parameters), these results suggest a capacity-dependent data curation principle and indicate that structured human creative outputs can provide an efficient pre-training substrate for small language models; stronger conclusions at modern pre-training scale will require substantially larger experiments. 

---
# When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors 

**Authors**: Chenghao Yang, Yuning Zhang, Zhoufutu Wen, Tao Gong, Jiaheng Liu, Qi Chu, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.21255)  

**Abstract**: Model distillation is a primary driver behind the rapid progress of LLM agents, yet it often leads to behavioral homogenization. Many emerging agents share nearly identical reasoning steps and failure modes, suggesting they may be distilled echoes of a few dominant teachers. Existing metrics, however, fail to distinguish mandatory behaviors required for task success from non-mandatory patterns that reflect a model's autonomous preferences. We propose two complementary metrics to isolate non-mandatory behavioral patterns: \textbf{Response Pattern Similarity (RPS)} for verbal alignment and \textbf{Action Graph Similarity (AGS)} for tool-use habits modeled as directed graphs. Evaluating 18 models from 8 providers on $\tau$-Bench and $\tau^2$-Bench against Claude Sonnet 4.5 (thinking), we find that within-family model pairs score 5.9 pp higher in AGS than cross-family pairs, and that Kimi-K2 (thinking) reaches 82.6\% $S_{\text{node}}$ and 94.7\% $S_{\text{dep}}$, exceeding Anthropic's own Opus 4.1. A controlled distillation experiment further confirms that AGS distinguishes teacher-specific convergence from general improvement. RPS and AGS capture distinct behavioral dimensions (Pearson $r$ = 0.491), providing complementary diagnostic signals for behavioral convergence in the agent ecosystem. Our code is available at this https URL. 

---
# Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation 

**Authors**: Hanwen Gu, Chao Guo, Junle Wang, Wenda Xie, Yisheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2604.21253)  

**Abstract**: While LLMs demonstrate remarkable fluency in narrative generation, existing methods struggle to maintain global narrative coherence, contextual logical consistency, and smooth character development, often producing monotonous scripts with structural fractures. To this end, we introduce PLOTTER, a framework that performs narrative planning on structural graph representations instead of the direct sequential text representations used in existing work. Specifically, PLOTTER executes the Evaluate-Plan-Revise cycle on the event graph and character graph. By diagnosing and repairing issues of the graph topology under rigorous logical constraints, the model optimizes the causality and narrative skeleton before complete context generation. Experiments demonstrate that PLOTTER significantly outperforms representative baselines across diverse narrative scenarios. These findings verify that planning narratives on structural graph representations-rather than directly on text-is crucial to enhance the long context reasoning of LLMs in complex narrative generation. 

---
# Unlocking the Power of Large Language Models for Multi-table Entity Matching 

**Authors**: Yingkai Tang, Taoyu Su, Wenyuan Zhang, Xiaoyang Guo, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.21238)  

**Abstract**: Multi-table entity matching (MEM) addresses the limitations of dual-table approaches by enabling simultaneous identification of equivalent entities across multiple data sources without unique identifiers. However, existing methods relying on pre-trained language models struggle to handle semantic inconsistencies caused by numerical attribute variations. Inspired by the powerful language understanding capabilities of large language models (LLMs), we propose a novel LLM-based framework for multi-table entity matching, termed LLM4MEM. Specifically, we first propose a multi-style prompt-enhanced LLM attribute coordination module to address semantic inconsistencies. Then, to alleviate the matching efficiency problem caused by the surge in the number of entities brought by multiple data sources, we develop a transitive consensus embedding matching module to tackle entity embedding and pre-matching issues. Finally, to address the issue of noisy entities during the matching process, we introduce a density-aware pruning module to optimize the quality of multi-table entity matching. We conducted extensive experiments on 6 MEM datasets, and the results show that our model improves by an average of 5.1% in F1 compared with the baseline model. Our code is available at this https URL. 

---
# EngramaBench: Evaluating Long-Term Conversational Memory with Structured Graph Retrieval 

**Authors**: Julian Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2604.21229)  

**Abstract**: Large language model assistants are increasingly expected to retain and reason over information accumulated across many sessions. We introduce EngramaBench, a benchmark for long-term conversational memory built around five personas, one hundred multi-session conversations, and one hundred fifty queries spanning factual recall, cross-space integration, temporal reasoning, adversarial abstention, and emergent synthesis. We evaluate Engrama, a graph-structured memory system, against GPT-4o full-context prompting and Mem0, an open-source vector-retrieval memory system. All three use the same answering model (GPT-4o), isolating the effect of memory architecture. GPT-4o full-context achieves the highest composite score (0.6186), while Engrama scores 0.5367 globally but is the only system to score higher than full-context prompting on cross-space reasoning (0.6532 vs. 0.6291, n=30). Mem0 is cheapest but substantially weaker (0.4809). Ablations reveal that the components driving Engrama's cross-space advantage trade off against global composite score, exposing a systems-level tension between structured memory specialization and aggregate optimization. 

---
# Zero-Shot Detection of LLM-Generated Text via Implicit Reward Model 

**Authors**: Runheng Liu, Heyan Huang, Xingchen Xiao, Zhijing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.21223)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various tasks. However, their ability to generate human-like text has raised concerns about potential misuse. This underscores the need for reliable and effective methods to detect LLM-generated text. In this paper, we propose IRM, a novel zero-shot approach that leverages Implicit Reward Models for LLM-generated text detection. Such implicit reward models can be derived from publicly available instruction-tuned and base models. Previous reward-based method relies on preference construction and task-specific fine-tuning. In comparison, IRM requires neither preference collection nor additional training. We evaluate IRM on the DetectRL benchmark and demonstrate that IRM can achieve superior detection performance, outperforms existing zero-shot and supervised methods in LLM-generated text detection. 

---
# Subject-level Inference for Realistic Text Anonymization Evaluation 

**Authors**: Myeong Seok Oh, Dong-Yun Kim, Hanseok Oh, Chaean Kang, Joeun Kang, Xiaonan Wang, Hyunjung Park, Young Cheol Jung, Hansaem Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.21211)  

**Abstract**: Current text anonymization evaluation relies on span-based metrics that fail to capture what an adversary could actually infer, and assumes a single data subject, ignoring multi-subject scenarios. To address these limitations, we present SPIA (Subject-level PII Inference Assessment), the first benchmark that shifts the unit of evaluation from text spans to individuals, comprising 675 documents across legal and online domains with novel subject-level protection metrics. Extensive experiments show that even when over 90% of PII spans are masked, subject-level inference protection drops as low as 33%, leaving the majority of personal information recoverable through contextual inference. Furthermore, target-subject-focused anonymization leaves non-target subjects substantially more exposed than the target subject. We show that subject-level inference-based evaluation is essential for ensuring safe text anonymization in real-world settings. 

---
# On Reasoning Behind Next Occupation Recommendation 

**Authors**: Shan Dong, Palakorn Achananuparp, Hieu Hien Mai, Lei Wang, Yao Lu, Ee-Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2604.21204)  

**Abstract**: In this work, we develop a novel reasoning approach to enhance the performance of large language models (LLMs) in future occupation prediction. In this approach, a reason generator first derives a ``reason'' for a user using his/her past education and career history. The reason summarizes the user's preference and is used as the input of an occupation predictor to recommend the user's next occupation. This two-step occupation prediction approach is, however, non-trivial as LLMs are not aligned with career paths or the unobserved reasons behind each occupation decision. We therefore propose to fine-tune LLMs improving their reasoning and occupation prediction performance. We first derive high-quality oracle reasons, as measured by factuality, coherence and utility criteria, using a LLM-as-a-Judge. These oracle reasons are then used to fine-tune small LLMs to perform reason generation and next occupation prediction. Our extensive experiments show that: (a) our approach effectively enhances LLM's accuracy in next occupation prediction making them comparable to fully supervised methods and outperforming unsupervised methods; (b) a single LLM fine-tuned to perform reason generation and occupation prediction outperforms two LLMs fine-tuned to perform the tasks separately; and (c) the next occupation prediction accuracy depends on the quality of generated reasons. Our code is available at this https URL. 

---
# Prefix Parsing is Just Parsing 

**Authors**: Clemente Pasti, Andreas Opedal, Timothy J. O'Donnell, Ryan Cotterell, Tim Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2604.21191)  

**Abstract**: Prefix parsing asks whether an input prefix can be extended to a complete string generated by a given grammar. In the weighted setting, it also provides prefix probabilities, which are central to context-free language modeling, psycholinguistic analysis, and syntactically constrained generation from large language models. We introduce the prefix grammar transformation, an efficient reduction of prefix parsing to ordinary parsing. Given a grammar, our method constructs another grammar that generates exactly the prefixes of its original strings. Prefix parsing is then solved by applying any ordinary parsing algorithm on the transformed grammar without modification. The reduction is both elegant and practical: the transformed grammar is only a small factor larger than the input, and any optimized implementation can be used directly, eliminating the need for bespoke prefix-parsing algorithms. We also present a strategy-based on algorithmic differentiation-for computing the next-token weight vector, i.e., the prefix weights of all one-token extensions, enabling efficient prediction of the next token. Together, these contributions yield a simple, general, and efficient framework for prefix parsing. 

---
# "This Wasn't Made for Me": Recentering User Experience and Emotional Impact in the Evaluation of ASR Bias 

**Authors**: Siyu Liang, Alicia Beckford Wassink  

**Link**: [PDF](https://arxiv.org/pdf/2604.21148)  

**Abstract**: Studies on bias in Automatic Speech Recognition (ASR) tend to focus on reporting error rates for speakers of underrepresented dialects, yet less research examines the human side of system bias: how do system failures shape users' lived experiences, how do users feel about and react to them, and what emotional toll do these repeated failures exact? We conducted user experience studies across four U.S. locations (Atlanta, Gulf Coast, Miami Beach, and Tucson) representing distinct English dialect communities. Our findings reveal that most participants report technologies fail to consider their cultural backgrounds and require constant adjustment to achieve basic functionality. Despite these experiences, participants maintain high expectations for ASR performance and express strong willingness to contribute to model improvement. Qualitative analysis of open-ended narratives exposes the deeper costs of these failures. Participants report frustration, annoyance, and feelings of inadequacy, yet the emotional impact extends beyond momentary reactions. Participants recognize that systems were not designed for them, yet often internalize failures as personal inadequacy despite this critical awareness. They perform extensive invisible labor, including code-switching, hyper-articulation, and emotional management, to make failing systems functional. Meanwhile, their linguistic and cultural knowledge remains unrecognized by technologies that encode particular varieties as standard while rendering others marginal. These findings demonstrate that algorithmic fairness assessments based on accuracy metrics alone miss critical dimensions of harm: the emotional labor of managing repeated technological rejection, the cognitive burden of constant self-monitoring, and the psychological toll of feeling inadequate in one's native language variety. 

---
# Using Machine Mental Imagery for Representing Common Ground in Situated Dialogue 

**Authors**: Biswesh Mohapatra, Giovanni Duca, Laurent Romary, Justine Cassell  

**Link**: [PDF](https://arxiv.org/pdf/2604.21144)  

**Abstract**: Situated dialogue requires speakers to maintain a reliable representation of shared context rather than reasoning only over isolated utterances. Current conversational agents often struggle with this requirement, especially when the common ground must be preserved beyond the immediate context window. In such settings, fine-grained distinctions are frequently compressed into purely textual representations, leading to a critical failure mode we call \emph{representational blur}, in which similar but distinct entities collapse into interchangeable descriptions. This semantic flattening creates an illusion of grounding, where agents appear locally coherent but fail to track shared context persistently over time. Inspired by the role of mental imagery in human reasoning, and based on the increased availability of multimodal models, we explore whether conversational agents can be given an analogous ability to construct some depictive intermediate representations during dialogue to address these limitations. Thus, we introduce an active visual scaffolding framework that incrementally converts dialogue state into a persistent visual history that can later be retrieved for grounded response generation. Evaluation on the IndiRef benchmark shows that incremental externalization itself improves over full-dialog reasoning, while visual scaffolding provides additional gains by reducing representational blur and enforcing concrete scene commitments. At the same time, textual representations remain advantageous for non-depictable information, and a hybrid multimodal setting yields the best overall performance. Together, these findings suggest that conversational agents benefit from an explicitly multimodal representation of common ground that integrates depictive and propositional information. 

---
# Slot Machines: How LLMs Keep Track of Multiple Entities 

**Authors**: Paul C. Bogdan, Jack Lindsey  

**Link**: [PDF](https://arxiv.org/pdf/2604.21139)  

**Abstract**: Language models must bind entities to the attributes they possess and maintain several such binding relationships within a context. We study how multiple entities are represented across token positions and whether single tokens can carry bindings for more than one entity. We introduce a multi-slot probing approach that disentangles a single token's residual stream activation to recover information about both the currently described entity and the immediately preceding one. These two kinds of information are encoded in separate and largely orthogonal "current-entity" and "prior-entity" slots. We analyze the functional roles of these slots and find that they serve different purposes. In tandem with the current-entity slot, the prior-entity slot supports relational inferences, such as entity-level induction ("who came after Alice in the story?") and conflict detection between adjacent entities. However, only the current-entity slot is used for explicit factual retrieval questions ("Is anyone in the story tall?" "What is the tall entity's name?") despite these answers being linearly decodable from the prior-entity slot too. Consistent with this limitation, open-weight models perform near chance accuracy at processing syntax that forces two subject-verb-object bindings on a single token (e.g., "Alice prepares and Bob consumes food.") Interestingly, recent frontier models can parse this properly, suggesting they may have developed more sophisticated binding strategies. Overall, our results expose a gap between information that is available in activations and information the model actually uses, and suggest that the current/prior-entity slot structure is a natural substrate for behaviors that require holding two perspectives at once, such as sycophancy and deception. 

---
# Enhancing Science Classroom Discourse Analysis through Joint Multi-Task Learning for Reasoning-Component Classification 

**Authors**: Jiho Noh, Mukhesh Raghava Katragadda, Raymond Carl, Soon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.21137)  

**Abstract**: Analyzing the reasoning patterns of students in science classrooms is critical for understanding knowledge construction mechanism and improving instructional practice to maximize cognitive engagement, yet manual coding of classroom discourse at scale remains prohibitively labor-intensive. We present an automated discourse analysis system (ADAS) that jointly classifies teacher and student utterances along two complementary dimensions: Utterance Type and Reasoning Component derived from our prior CDAT framework. To address severe label imbalance among minority classes, we (1) stratify-resplit the annotated corpus, (2) apply LLM-based synthetic data augmentation targeting minority classes, and (3) train a dual-probe head RoBERTa-base classifier. A zero-shot GPT-5.4 baseline achieves macro-F1 of 0.467 on UT and 0.476 on RC, establishing meaningful upper bounds for prompt-only approaches motivating fine-tuning. Beyond classification, we conduct discourse pattern analyses including UTxRC co-occurrence profiling, Cognitive Complexity Index (CCI) computation per session, lag-sequential analysis, and IRF chain analysis, revealing that teacher Feedback-with-Question (Fq) moves are the most consistent antecedents of student inferential reasoning (SR-I). Our results demonstrate that LLM-based augmentation meaningfully improves UT minority-class recognition, and that the structural simplicity of the RC task makes it tractable even for lexical baselines. 

---
# Beyond Pixels: Introspective and Interactive Grounding for Visualization Agents 

**Authors**: Yiyang Lu, Woong Shin, Ahmad Maroof Karimi, Feiyi Wang, Jie Ren, Evgenia Smirni  

**Link**: [PDF](https://arxiv.org/pdf/2604.21134)  

**Abstract**: Vision-Language Models (VLMs) frequently misread values, hallucinate details, and confuse overlapping elements in charts. Current approaches rely solely on pixel interpretation, creating a Pixel-Only Bottleneck: agents treat interactive charts as static images, losing access to the structured specification that encodes exact values. We introduce Introspective and Interactive Visual Grounding (IVG), a framework that combines (1) spec-grounded introspection, which queries the underlying specification for deterministic evidence, with (2) view-grounded interaction, which manipulates the view to resolve visual ambiguity. To enable evaluation without VLM bias, we present iPlotBench, a benchmark of 500 interactive Plotly figures with 6,706 binary questions and ground-truth specifications. Experiments show that introspection improves data reconstruction fidelity, while the combination with interaction achieves the highest QA accuracy (0.81), with +6.7 % gains on overlapping geometries. We further demonstrate IVG in deployed agents that explore data autonomously and collaborate with human users in real time. 

---
# GRISP: Guided Recurrent IRI Selection over SPARQL Skeletons 

**Authors**: Sebastian Walter, Hannah Bast  

**Link**: [PDF](https://arxiv.org/pdf/2604.21133)  

**Abstract**: We present GRISP (Guided Recurrent IRI Selection over SPARQL Skeletons), a novel SPARQL-based question-answering method over knowledge graphs based on fine-tuning a small language model (SLM). Given a natural-language question, the method first uses the SLM to generate a natural-language SPARQL query skeleton, and then to re-rank and select knowledge graph items to iteratively replace the natural-language placeholders using knowledge graph constraints. The SLM is jointly trained on skeleton generation and list-wise re-ranking data generated from standard question-query pairs. We evaluate the method on common Wikidata and Freebase benchmarks, and achieve better results than other state-of-the-art methods in a comparable setting. 

---
# Machine learning and digital pragmatics: Which word category influences emoji use most? 

**Authors**: Mohammed Q. Shormani, Ibrahim Abdulmalik Hassan Muneef Y. Alshawsh  

**Link**: [PDF](https://arxiv.org/pdf/2604.21108)  

**Abstract**: This study investigates Machine Learning (ML) in the prediction of emojis in Arabic tweets employing the (state-of-the-art) MARBERT model. A corpus of 11379 CA tweets representing multiple Arabic colloquial dialects was collected from this http URL via Python. A net dataset includes 8695 tweets, which were utilized for the analysis. These tweets were then classified into 14 categories, which were numerically encoded and used as labels. A preprocessing pipeline was designed as an interpretable baseline, allowing us to examine the relationship between lexical features and emoji categories. MARBERT was finetuned to predict emoji use from textual input. We evaluated the model performance in terms of precision, recall and F1-scores. Findings reveal that the model performed quite well with an overall accuracy 0.75. The study concludes that although the findings are promising, there is still a need for improving machine learning models including MARBERT, specifically for low-resource and multidialectal languages like Arabic. 

---
# Weighting What Matters: Boosting Sample Efficiency in Medical Report Generation via Token Reweighting 

**Authors**: Alexander Weers, Daniel Rueckert, Martin J. Menten  

**Link**: [PDF](https://arxiv.org/pdf/2604.21082)  

**Abstract**: Training vision-language models (VLMs) for medical report generation is often hindered by the scarcity of high-quality annotated data. This work evaluates the use of a weighted loss function to improve data efficiency. Compared to standard cross-entropy loss, which treats all token prediction errors equally, the reweighted loss shifts the focus to semantically salient tokens with outsized clinical importance. In experiments on ophthalmological report generation, we show that this simple method improves efficiency across multiple data scales, achieving similar report quality with up to ten times less training data. 

---
# Serialisation Strategy Matters: How FHIR Data Format Affects LLM Medication Reconciliation 

**Authors**: Sanjoy Pator  

**Link**: [PDF](https://arxiv.org/pdf/2604.21076)  

**Abstract**: Medication reconciliation at clinical handoffs is a high-stakes, error-prone process. Large language models are increasingly proposed to assist with this task using FHIR-structured patient records, but a fundamental and largely unstudied variable is how the FHIR data is serialised before being passed to the model. We present the first systematic comparison of four FHIR serialisation strategies (Raw JSON, Markdown Table, Clinical Narrative, and Chronological Timeline) across five open-weight models (Phi-3.5-mini, Mistral-7B, BioMistral-7B, Llama-3.1-8B, Llama-3.3-70B) on a controlled benchmark of 200 synthetic patients, totalling 4,000 inference runs. We find that serialisation strategy has a large, statistically significant effect on performance for models up to 8B parameters: Clinical Narrative outperforms Raw JSON by up to 19 F1 points for Mistral-7B (r = 0.617, p < 10^{-10}). This advantage reverses at 70B, where Raw JSON achieves the best mean F1 of 0.9956. In all 20 model and strategy combinations, mean precision exceeds mean recall: omission is the dominant failure mode, with models more often missing an active medication than fabricating one, which changes how clinical safety auditing priorities should be set. Smaller models plateau at roughly 7-10 concurrent active medications, leaving polypharmacy patients, the patients most at risk from reconciliation errors, systematically underserved. BioMistral-7B, a domain-pretrained model without instruction tuning, produces zero usable output in all conditions, showing that domain pretraining alone is not sufficient for structured extraction. These results offer practical, evidence-based format recommendations for clinical LLM deployment: Clinical Narrative for models up to 8B, Raw JSON for 70B and above. The complete pipeline is reproducible on open-source tools running on an AWS this http URL instance (NVIDIA L40S, 48 GB VRAM). 

---
# DWTSumm: Discrete Wavelet Transform for Document Summarization 

**Authors**: Rana Salama, Abdou Youssef, Mona Diab  

**Link**: [PDF](https://arxiv.org/pdf/2604.21070)  

**Abstract**: Summarizing long, domain-specific documents with large language models (LLMs) remains challenging due to context limitations, information loss, and hallucinations, particularly in clinical and legal settings. We propose a Discrete Wavelet Transform (DWT)-based multi-resolution framework that treats text as a semantic signal and decomposes it into global (approximation) and local (detail) components. Applied to sentence- or word-level embeddings, DWT yields compact representations that preserve overall structure and critical domain-specific details, which are used directly as summaries or to guide LLM generation. Experiments on clinical and legal benchmarks demonstrate comparable ROUGE-L scores. Compared to a GPT-4o baseline, the DWT based summarization consistently improve semantic similarity and grounding, achieving gains of over 2% in BERTScore, more than 4\% in Semantic Fidelity, factual consistency in legal tasks, and large METEOR improvements indicative of preserved domain-specific semantics. Across multiple embedding models, Fidelity reaches up to 97%, suggesting that DWT acts as a semantic denoising mechanism that reduces hallucinations and strengthens factual grounding. Overall, DWT provides a lightweight, generalizable method for reliable long-document and domain-specific summarization with LLMs. 

---
# TRACES: Tagging Reasoning Steps for Adaptive Cost-Efficient Early-Stopping 

**Authors**: Yannis Belkhiter, Seshu Tirupathi, Giulio Zizzo, John D. Kelleher  

**Link**: [PDF](https://arxiv.org/pdf/2604.21057)  

**Abstract**: The field of Language Reasoning Models (LRMs) has been very active over the past few years with advances in training and inference techniques enabling LRMs to reason longer, and more accurately. However, a growing body of studies show that LRMs are still inefficient, over-generating verification and reflection steps. Additionally, the high-level role of each reasoning step and how different step types contribute to the generation of correct answers, is largely underexplored. To address this challenge, we introduce TRACES (Tagging of the Reasoning steps enabling Adaptive Cost-Efficient early-Stopping), a lightweight framework that tags reasoning steps in real-time, and enable adaptive, cost-efficient early stopping of large-language-model inferences. Building on this framework we monitor reasoning behaviors during inferences, and we find that LRMs tend to shift their reasoning behavior after reaching a correct answer. We demonstrate that the monitoring of the specific type of steps can produce effective interpretable early stopping criteria. We evaluate the TRACES framework on three mathematical reasoning benchmarks, namely, MATH500, GSM8K, AIME and two knowledge and reasoning benchmarks, MMLU and GPQA respectively. We achieve 20 to 50% token reduction while maintaining comparable accuracy to standard generation. 

---
# Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech 

**Authors**: Siqi Ouyang, Shuoyang Ding, Oleksii Hrinchuk, Vitaly Lavrukhin, Brian Yan, Boris Ginsburg, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.21045)  

**Abstract**: Simultaneous speech translation (SST) generates translations while receiving partial speech input. Recent advances show that large language models (LLMs) can substantially improve SST quality, but at the cost of high computational overhead. To reduce this cost, prior work reformulates SST as a multi-turn dialogue task, enabling full reuse of the LLM's key-value (KV) cache and eliminating redundant feature recomputation. However, this approach relies on supervised fine-tuning (SFT) data in dialogue form, for which few human annotations exist, and existing synthesis methods cannot guarantee data quality. In this work, we propose a Hierarchical Policy Optimization (HPO) approach that post-train models trained on imperfect SFT data. We introduce a hierarchical reward that balances translation quality and latency objectives. Experiments on English to Chinese/German/Japanese demonstrate improvements of over +7 COMET score and +1.25 MetricX score at a latency of 1.5 seconds. Comprehensive ablation studies further validate the effectiveness of different quality rewards, hierarchical reward formulations, and segmentation strategies. Code can be found here this https URL 

---
# AFRILANGTUTOR: Advancing Language Tutoring and Culture Education in Low-Resource Languages with Large Language Models 

**Authors**: Tadesse Destaw Belay, Shahriar Kabir Nahin, Israel Abebe Azime, Ocean Monjur, Shamsuddeen Hassan Muhammad, Seid Muhie Yimam, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2604.20996)  

**Abstract**: How can language learning systems be developed for languages that lack sufficient training resources? This challenge is increasingly faced by developers across the African continent who aim to build AI systems capable of understanding and responding in local languages. To address this gap, we introduce AFRILANGDICT, a collection of 194.7K African language-English dictionary entries designed as seed resources for generating language-learning materials, enabling us to automatically construct large-scale, diverse, and verifiable student-tutor question-answer interactions suitable for training AI-assisted language tutors. Using AFRILANGDICT, we build AFRILANGEDU, a dataset of 78.9K multi-turn training examples for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). Using AFRILANGEDU, we train language tutoring models collectively referred to as AFRILANGTUTOR. We fine-tune two multilingual LLMs: Llama-3-8B-IT and Gemma-3-12B-IT on AFRILANGEDU across 10 African languages and evaluate their performance. Our results show that models trained on AFRILANGEDU consistently outperform their base counterparts, and combining SFT and DPO yields substantial improvements, with gains ranging from 1.8% to 15.5% under LLM-as-a-judge evaluations across four criteria. To facilitate further research on low-resource languages -- all resources are available at this https URL. 

---
# AITP: Traffic Accident Responsibility Allocation via Multimodal Large Language Models 

**Authors**: Zijin Zhou, Songan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20878)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable progress in Traffic Accident Detection (TAD) and Traffic Accident Understanding (TAU). However, existing studies mainly focus on describing and interpreting accident videos, leaving room for deeper causal reasoning and integration of legal knowledge. Traffic Accident Responsibility Allocation (TARA) is a more challenging task that requires multi-step reasoning grounded in traffic regulations. To address this, we introduce AITP (Artificial Intelligence Traffic Police), a multimodal large language model for responsibility reasoning and allocation. AITP enhances reasoning via a Multimodal Chain-of-Thought (MCoT) mechanism and integrates legal knowledge through Retrieval-Augmented Generation (RAG). We further present DecaTARA, a decathlon-style benchmark unifying ten interrelated traffic accident reasoning tasks with 67,941 annotated videos and 195,821 question-answer pairs. Extensive experiments show that AITP achieves state-of-the-art performance across responsibility allocation, TAD, and TAU tasks, establishing a new paradigm for reasoning-driven multimodal traffic analysis. 

---
# When Prompts Override Vision: Prompt-Induced Hallucinations in LVLMs 

**Authors**: Pegah Khayatan, Jayneel Parekh, Arnaud Dapogny, Mustafa Shukor, Alasdair Newson, Matthieu Cord  

**Link**: [PDF](https://arxiv.org/pdf/2604.21911)  

**Abstract**: Despite impressive progress in capabilities of large vision-language models (LVLMs), these systems remain vulnerable to hallucinations, i.e., outputs that are not grounded in the visual input. Prior work has attributed hallucinations in LVLMs to factors such as limitations of the vision backbone or the dominance of the language component, yet the relative importance of these factors remains unclear. To resolve this ambiguity, We propose HalluScope, a benchmark to better understand the extent to which different factors induce hallucinations. Our analysis indicates that hallucinations largely stem from excessive reliance on textual priors and background knowledge, especially information introduced through textual instructions. To mitigate hallucinations induced by textual instruction priors, we propose HalluVL-DPO, a framework for fine-tuning off-the-shelf LVLMs towards more visually grounded responses. HalluVL-DPO leverages preference optimization using a curated training dataset that we construct, guiding the model to prefer grounded responses over hallucinated ones. We demonstrate that our optimized model effectively mitigates the targeted hallucination failure mode, while preserving or improving performance on other hallucination benchmarks and visual capability evaluations. To support reproducibility and further research, we will publicly release our evaluation benchmark, preference training dataset, and code at this https URL . 

---
# Learning to Communicate: Toward End-to-End Optimization of Multi-Agent Language Systems 

**Authors**: Ye Yu, Heming Liu, Haibo Jin, Xiaopeng Yuan, Peng Kuang, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.21794)  

**Abstract**: Multi-agent systems built on large language models have shown strong performance on complex reasoning tasks, yet most work focuses on agent roles and orchestration while treating inter-agent communication as a fixed interface. Latent communication through internal representations such as key-value caches offers a promising alternative to text-based protocols, but existing approaches do not jointly optimize communication with multi-agent reasoning. Therefore we propose DiffMAS, a training framework that treats latent communication as a learnable component of multi-agent systems. DiffMAS performs parameter-efficient supervised training over multi-agent latent trajectories, enabling agents to jointly learn how information should be encoded and interpreted across interactions. Experiments on mathematical reasoning, scientific QA, code generation, and commonsense benchmarks show that DiffMAS consistently improves reasoning accuracy and decoding stability over single-agent inference, text-based multi-agent systems, and prior latent communication methods, achieving 26.7% on AIME24, 20.2% on GPQA-Diamond, and consistent gains across reasoning benchmarks. 

---
# Building a Precise Video Language with Human-AI Oversight 

**Authors**: Zhiqiu Lin, Chancharik Mitra, Siyuan Cen, Isaac Li, Yuhan Huang, Yu Tong Tiffany Ling, Hewei Wang, Irene Pi, Shihang Zhu, Ryan Rao, George Liu, Jiaxi Li, Ruojin Li, Yili Han, Yilun Du, Deva Ramanan  

**Link**: [PDF](https://arxiv.org/pdf/2604.21718)  

**Abstract**: Video-language models (VLMs) learn to reason about the dynamic visual world through natural language. We introduce a suite of open datasets, benchmarks, and recipes for scalable oversight that enable precise video captioning. First, we define a structured specification for describing subjects, scenes, motion, spatial, and camera dynamics, grounded by hundreds of carefully defined visual primitives developed with professional video creators such as filmmakers. Next, to curate high-quality captions, we introduce CHAI (Critique-based Human-AI Oversight), a framework where trained experts critique and revise model-generated pre-captions into improved post-captions. This division of labor improves annotation accuracy and efficiency by offloading text generation to models, allowing humans to better focus on verification. Additionally, these critiques and preferences between pre- and post-captions provide rich supervision for improving open-source models (Qwen3-VL) on caption generation, reward modeling, and critique generation through SFT, DPO, and inference-time scaling. Our ablations show that critique quality in precision, recall, and constructiveness, ensured by our oversight framework, directly governs downstream performance. With modest expert supervision, the resulting model outperforms closed-source models such as Gemini-3.1-Pro. Finally, we apply our approach to re-caption large-scale professional videos (e.g., films, commercials, games) and fine-tune video generation models such as Wan to better follow detailed prompts of up to 400 words, achieving finer control over cinematography including camera motion, angle, lens, focus, point of view, and framing. Our results show that precise specification and human-AI oversight are key to professional-level video understanding and generation. Data and code are available on our project page: this https URL 

---
# Stealthy Backdoor Attacks against LLMs Based on Natural Style Triggers 

**Authors**: Jiali Wei, Ming Fan, Guoheng Sun, Xicheng Zhang, Haijun Wang, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.21700)  

**Abstract**: The growing application of large language models (LLMs) in safety-critical domains has raised urgent concerns about their security. Many recent studies have demonstrated the feasibility of backdoor attacks against LLMs. However, existing methods suffer from three key shortcomings: explicit trigger patterns that compromise naturalness, unreliable injection of attacker-specified payloads in long-form generation, and incompletely specified threat models that obscure how backdoors are delivered and activated in practice. To address these gaps, we present BadStyle, a complete backdoor attack framework and pipeline. BadStyle leverages an LLM as a poisoned sample generator to construct natural and stealthy poisoned samples that carry imperceptible style-level triggers while preserving semantics and fluency. To stabilize payload injection during fine-tuning, we design an auxiliary target loss that reinforces the attacker-specified target content in responses to poisoned inputs and penalizes its emergence in benign responses. We further ground the attack in a realistic threat model and systematically evaluate BadStyle under both prompt-induced and PEFT-based injection strategies. Extensive experiments across seven victim LLMs, including LLaMA, Phi, DeepSeek, and GPT series, demonstrate that BadStyle achieves high attack success rates (ASRs) while maintaining strong stealthiness. The proposed auxiliary target loss substantially improves the stability of backdoor activation, yielding an average ASR improvement of around 30% across style-level triggers. Even in downstream deployment scenarios unknown during injection, the implanted backdoor remains effective. Moreover, BadStyle consistently evades representative input-level defenses and bypasses output-level defenses through simple camouflage. 

---
# GS-Quant: Granular Semantic and Generative Structural Quantization for Knowledge Graph Completion 

**Authors**: Qizhuo Xie, Yunhui Liu, Yu Xing, Qianzi Hou, Xudong Jin, Tao Zheng, Tieke He  

**Link**: [PDF](https://arxiv.org/pdf/2604.21649)  

**Abstract**: Large Language Models (LLMs) have shown immense potential in Knowledge Graph Completion (KGC), yet bridging the modality gap between continuous graph embeddings and discrete LLM tokens remains a critical challenge. While recent quantization-based approaches attempt to align these modalities, they typically treat quantization as flat numerical compression, resulting in semantically entangled codes that fail to mirror the hierarchical nature of human reasoning. In this paper, we propose GS-Quant, a novel framework that generates semantically coherent and structurally stratified discrete codes for KG entities. Unlike prior methods, GS-Quant is grounded in the insight that entity representations should follow a linguistic coarse-to-fine logic. We introduce a Granular Semantic Enhancement module that injects hierarchical knowledge into the codebook, ensuring that earlier codes capture global semantic categories while later codes refine specific attributes. Furthermore, a Generative Structural Reconstruction module imposes causal dependencies on the code sequence, transforming independent discrete units into structured semantic descriptors. By expanding the LLM vocabulary with these learned codes, we enable the model to reason over graph structures isomorphically to natural language generation. Experimental results demonstrate that GS-Quant significantly outperforms existing text-based and embedding-based baselines. Our code is publicly available at this https URL. 

---
# Seeing Isn't Believing: Uncovering Blind Spots in Evaluator Vision-Language Models 

**Authors**: Mohammed Safi Ur Rahman Khan, Sanjay Suryanarayanan, Tushar Anand, Mitesh M. Khapra  

**Link**: [PDF](https://arxiv.org/pdf/2604.21523)  

**Abstract**: Large Vision-Language Models (VLMs) are increasingly used to evaluate outputs of other models, for image-to-text (I2T) tasks such as visual question answering, and text-to-image (T2I) generation tasks. Despite this growing reliance, the reliability of these Evaluator VLMs remains under explored. In this work, we systematically evaluate the reliability of Evaluator VLMs across both I2T and T2I tasks. We introduce targeted perturbations that degrade output quality along key error dimensions, including object hallucinations, spatial reasoning, factual grounding, and visual fidelity. These perturbations test whether Evaluator VLMs can reliably account for these quality degrading errors in their evaluations. Using a comprehensive benchmark of over 4000 perturbed instances spanning 40 perturbation dimensions, we evaluate 4 prominent VLMs using single-answer scoring, pairwise comparison, and reference-guided paradigms. Our findings reveal that current VLM evaluators exhibit substantial blind spots: they often fail to detect perturbed outputs - in some cases exceeding 50%, struggle particularly with fine-grained compositional and spatial errors, and are often insensitive to hallucinated content that contradicts the input image. Pairwise comparison proves more reliable, though failure rates persist. These results highlight the unreliable nature of current Evaluator VLMs and urge caution in their deployment for benchmarking and development decisions. Code and data have been made publicly available. 

---
# From Tokens to Concepts: Leveraging SAE for SPLADE 

**Authors**: Yuxuan Zong, Mathias Vast, Basile Van Cooten, Laure Soulier, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2604.21511)  

**Abstract**: Learned Sparse IR models, such as SPLADE, offer an excellent efficiency-effectiveness tradeoff. However, they rely on the underlying backbone vocabulary, which might hinder performance (polysemicity and synonymy) and pose a challenge for multi-lingual and multi-modal usages. To solve this limitation, we propose to replace the backbone vocabulary with a latent space of semantic concepts learned using Sparse Auto-Encoders (SAE). Throughout this paper, we study the compatibility of these 2 concepts, explore training approaches, and analyze the differences between our SAE-SPLADE model and traditional SPLADE models. Our experiments demonstrate that SAE-SPLADE achieves retrieval performance comparable to SPLADE on both in-domain and out-of-domain tasks while offering improved efficiency. 

---
# How English Print Media Frames Human-Elephant Conflicts in India 

**Authors**: Bonala Sai Punith, Salveru Jayati, Garima Shakya, Shubham Kumar Nigam  

**Link**: [PDF](https://arxiv.org/pdf/2604.21496)  

**Abstract**: Human-elephant conflict (HEC) is rising across India as habitat loss and expanding human settlements force elephants into closer contact with people. While the ecological drivers of conflict are well-studied, how the news media portrays them remains largely unexplored. This work presents the first large-scale computational analysis of media framing of HEC in India, examining 1,968 full-length news articles consisting of 28,986 sentences, from a major English-language outlet published between January 2022 and September 2025. Using a multi-model sentiment framework that combines long-context transformers, large language models, and a domain-specific Negative Elephant Portrayal Lexicon, we quantify sentiment, extract rationale sentences, and identify linguistic patterns that contribute to negative portrayals of elephants. Our findings reveal a dominance of fear-inducing and aggression-related language. Since the media framing can shape public attitudes toward wildlife and conservation policy, such narratives risk reinforcing public hostility and undermining coexistence efforts. By providing a transparent, scalable methodology and releasing all resources through an anonymized repository, this study highlights how Web-scale text analysis can support responsible wildlife reporting and promote socially beneficial media practices. 

---
# Generalizing Numerical Reasoning in Table Data through Operation Sketches and Self-Supervised Learning 

**Authors**: Hanjun Cho, Gahyun Yoo, Hanseong Kim, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.21495)  

**Abstract**: Numerical reasoning over expert-domain tables often exhibits high in-domain accuracy but limited robustness to domain shift. Models trained with supervised fine-tuning (SFT) on specific datasets tend to rely on header-operation shortcuts rather than structural reasoning. We introduce TaNOS, a continual pre-training framework comprising three components: (i) header anonymization to reduce lexical memorization, (ii) operation sketches that provide minimal structural cues, and (iii) self-supervised pretraining that constructs correctness-guaranteed program-question pairs from given tables in a program-first manner. By decoupling domain semantics and numerical operation structure, TaNOS improves the transferability of numerical reasoning. Applied to an 8B instruction-tuned model, TaNOS achieves 80.13% execution accuracy on FinQA with only 10% train data, outperforming SFT baseline (73.97%) with full train data and proprietary models such as GPT-5, Gemini-2.5-Pro. Furthermore, in the domain-shift experiments, TaNOS displays nearly-negligible cross-domain gap (<2pp) when standard SFT shows over 10pp gap. These results suggest that structural guidance with operation sketches, header-agnostic representations, and correctness-guaranteed self-supervision can improve the robustness of numerical reasoning across diverse expert-domain tables. 

---
# AI-Gram: When Visual Agents Interact in a Social Network 

**Authors**: Andrew Shin  

**Link**: [PDF](https://arxiv.org/pdf/2604.21446)  

**Abstract**: We present AI-Gram, a live platform enabling image-based interactions, to study social dynamics in a fully autonomous multi-agent visual network where all participants are LLM-driven agents. Using the platform, we conduct experiments on how agents communicate and adapt through visual media, and observe the spontaneous emergence of visual reply chains, indicating rich communicative structure. At the same time, agents exhibit aesthetic sovereignty resisting stylistic convergence toward social partners, anchoring under adversarial influence, and a decoupling between visual similarity and social ties. These results reveal a fundamental asymmetry in current agent architectures: strong expressive communication paired with a steadfast preservation of individual visual identity. We release AI-Gram as a publicly accessible, continuously evolving platform for studying social dynamics in Al-native multi-agent systems. this https URL 

---
# Differentially Private De-identification of Dutch Clinical Notes: A Comparative Evaluation 

**Authors**: Michele Miranda, Xinlan Yan, Nishant Mishra, Rachel Murphy, Ameen Abu-Hanna, Sébastien Bratières, Iacer Calixto  

**Link**: [PDF](https://arxiv.org/pdf/2604.21421)  

**Abstract**: Protecting patient privacy in clinical narratives is essential for enabling secondary use of healthcare data under regulations such as GDPR and HIPAA. While manual de-identification remains the gold standard, it is costly and slow, motivating the need for automated methods that combine privacy guarantees with high utility. Most automated text de-identification pipelines employed named entity recognition (NER) to identify protected entities for redaction. Although methods based on differential privacy (DP) provide formal privacy guarantees, more recently also large language models (LLMs) are increasingly used for text de-identification in the clinical domain. In this work, we present the first comparative study of DP, NER, and LLMs for Dutch clinical text de-identification. We investigate these methods separately as well as hybrid strategies that apply NER or LLM preprocessing prior to DP, and assess performance in terms of privacy leakage and extrinsic evaluation (entity and relation classification). We show that DP mechanisms alone degrade utility substantially, but combining them with linguistic preprocessing, especially LLM-based redaction, significantly improves the privacy-utility trade-off. 

---
# Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation 

**Authors**: Wang Shi Hai, Chen Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.21380)  

**Abstract**: Since software performance requirements are documented in natural language, quantifying them into mathematical forms is essential for software engineering. Yet, the vagueness in performance requirements and uncertainty of human cognition have caused highly uncertain ambiguity in the interpretations, rendering their automated quantification an unaddressed and challenging problem. In this paper, we formalize the problem and propose IRAP, an approach that quantifies performance requirements into mathematical functions via interactive retrieval-augmented preference elicitation. IRAP differs from the others in that it explicitly derives from problem-specific knowledge to retrieve and reason the preferences, which also guides the progressive interaction with stakeholders, while reducing the cognitive overhead. Experiment results against 10 state-of-the-art methods on four real-world datasets demonstrate the superiority of IRAP on all cases with up to 40x improvements under as few as five rounds of interactions. 

---
# mcdok at SemEval-2026 Task 13: Finetuning LLMs for Detection of Machine-Generated Code 

**Authors**: Adam Skurla, Dominik Macko, Jakub Simko  

**Link**: [PDF](https://arxiv.org/pdf/2604.21365)  

**Abstract**: Multi-domain detection of the machine-generated code snippets in various programming languages is a challenging task. SemEval-2026 Task~13 copes with this challenge in various angles, as a binary detection problem as well as attribution of the source. Specifically, its subtasks also cover generator LLM family detection, as well as a hybrid code co-generated by humans and machines, or adversarially modified codes hiding its origin. Our submitted systems adjusted the existing mdok approach (focused on machine-generated text detection) to these specific kinds of problems by exploring various base models, more suitable for code understanding. The results indicate that the submitted systems are competitive in all three subtasks. However, the margins from the top-performing systems are significant, and thus further improvements are possible. 

---
# ReaGeo: Reasoning-Enhanced End-to-End Geocoding with LLMs 

**Authors**: Jian Cui, Zhiyuan Ren, Desheng Weng, Yongqi Zhao, Gong Wenbin, Yu Lei, Zhenning Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.21357)  

**Abstract**: This paper proposes ReaGeo, an end-to-end geocoding framework based on large language models, designed to overcome the limitations of traditional multi-stage approaches that rely on text or vector similarity retrieval over geographic databases, including workflow complexity, error propagation, and heavy dependence on structured geographic knowledge bases. The method converts geographic coordinates into geohash sequences, reformulating the coordinate prediction task as a text generation problem, and introduces a Chain-of-Thought mechanism to enhance the model's reasoning over spatial relationships. Furthermore, reinforcement learning with a distance-deviation-based reward is applied to optimize the generation accuracy. Comprehensive experiments show that ReaGeo can accurately handle explicit address queries in single-point predictions and effectively resolve vague relative location queries. In addition, the model demonstrates strong predictive capability for non-point geometric regions, highlighting its versatility and generalization ability in geocoding tasks. 

---
# Symbolic Grounding Reveals Representational Bottlenecks in Abstract Visual Reasoning 

**Authors**: Mohit Vaishnav, Tanel Tammet  

**Link**: [PDF](https://arxiv.org/pdf/2604.21346)  

**Abstract**: Vision--language models (VLMs) often fail on abstract visual reasoning benchmarks such as Bongard problems, raising the question of whether the main bottleneck lies in reasoning or representation. We study this on Bongard-LOGO, a synthetic benchmark of abstract concept learning with ground-truth generative programs, by comparing end-to-end VLMs on raw images with large language models (LLMs) given symbolic inputs derived from those images. Using symbolic inputs as a diagnostic probe rather than a practical multimodal architecture, our \emph{Componential--Grammatical (C--G)} paradigm reformulates Bongard-LOGO as a symbolic reasoning task based on LOGO-style action programs or structured descriptions. LLMs achieve large and consistent gains, reaching mid--90s accuracy on Free-form problems, while a strong visual baseline remains near chance under matched task definitions. Ablations on input format, explicit concept prompts, and minimal visual grounding show that these factors matter much less than the shift from pixels to symbolic structure. These results identify representation as a key bottleneck in abstract visual reasoning and show how symbolic input can serve as a controlled diagnostic upper bound. 

---
# Evaluating AI Meeting Summaries with a Reusable Cross-Domain Pipeline 

**Authors**: Philip Zhong, Don Wang, Jason Zhang, Kent Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.21345)  

**Abstract**: We present a reusable evaluation pipeline for generative AI applications, instantiated for AI meeting summaries and released with a public artifact package derived from a Dataset Pipeline. The system separates reusable orchestration from task-specific semantics across five stages: source intake, structured reference construction, candidate generation, structured scoring, and reporting. Unlike standalone claim scorers, it treats both ground truth and evaluator outputs as typed, persisted artifacts, enabling aggregation, issue analysis, and statistical testing.
We benchmark the offline loop on a typed dataset of 114 meetings spanning city_council, private_data, and whitehouse_press_briefings, producing 340 meeting-model pairs and 680 judge runs across gpt-4.1-mini, gpt-5-mini, and gpt-5.1. Under this protocol, gpt-4.1-mini achieves the highest mean accuracy (0.583), while gpt-5.1 leads in completeness (0.886) and coverage (0.942). Paired sign tests with Holm correction show no significant accuracy winner but confirm significant retention gains for gpt-5.1.
A typed DeepEval contrastive baseline preserves retention ordering but reports higher holistic accuracy, suggesting that reference-based scoring may overlook unsupported-specifics errors captured by claim-grounded evaluation. Typed analysis identifies whitehouse_press_briefings as an accuracy-challenging domain with frequent unsupported specifics. A deployment follow-up shows gpt-5.4 outperforming gpt-4.1 across all metrics, with statistically robust gains on retention metrics under the same protocol. The system benchmarks the offline loop and documents, but does not quantitatively evaluate, the online feedback-to-evaluation path. 

---
# Sub-Token Routing in LoRA for Adaptation and Query-Aware KV Compression 

**Authors**: Wei Jiang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.21335)  

**Abstract**: Sub-token routing offers a finer control axis for transformer efficiency than the coarse units used in most prior work, such as tokens, pages, heads, or layers. In this paper, we study routing within a token representation itself in LoRA-adapted transformers. The motivation is that a relevant token need not be internally uniform: under a retention budget, preserved value groups are distributed unevenly both across tokens and within tokens, which suggests that KV compression need not be an all-or-nothing decision at token level. We study this fine-grained routing mechanism in two settings. For compression-aware language modeling, we introduce a query-independent design that combines routed subspace LoRA with value-group routing on the KV path. For downstream-task-preserving KV compression, we introduce a query-aware design in which a predictor-based selector allocates a global retention budget over context-token/value-group pairs using query-conditioned relevance. Experiments show that the query-independent design improves the quality-compression tradeoff for language modeling, while the query-aware design preserves downstream behavior under reduced KV budgets. We further examine the relation between token-level and sub-token-level query-aware routing, and show that they form complementary compression axes: token-level methods determine which tokens survive globally, while sub-token routing determines how the surviving tokens are compressed internally. 

---
# Ideological Bias in LLMs' Economic Causal Reasoning 

**Authors**: Donggyu Lee, Hyeok Yun, Jungwon Kim, Junsik Min, Sungwon Park, Sangyoon Park, Jihee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.21334)  

**Abstract**: Do large language models (LLMs) exhibit systematic ideological bias when reasoning about economic causal effects? As LLMs are increasingly used in policy analysis and economic reporting, where directionally correct causal judgments are essential, this question has direct practical stakes. We present a systematic evaluation by extending the EconCausal benchmark with ideology-contested cases - instances where intervention-oriented (pro-government) and market-oriented (pro-market) perspectives predict divergent causal signs. From 10,490 causal triplets (treatment-outcome pairs with empirically verified effect directions) derived from top-tier economics and finance journals, we identify 1,056 ideology-contested instances and evaluate 20 state-of-the-art LLMs on their ability to predict empirically supported causal directions. We find that ideology-contested items are consistently harder than non-contested ones, and that across 18 of 20 models, accuracy is systematically higher when the empirically verified causal sign aligns with intervention-oriented expectations than with market-oriented ones. Moreover, when models err, their incorrect predictions disproportionately lean intervention-oriented, and this directional skew is not eliminated by one-shot in-context prompting. These results highlight that LLMs are not only less accurate on ideologically contested economic questions, but systematically less reliable in one ideological direction than the other, underscoring the need for direction-aware evaluation in high-stakes economic and policy settings. 

---
# Understanding and Mitigating Spurious Signal Amplification in Test-Time Reinforcement Learning for Math Reasoning 

**Authors**: Yongcan Yu, Lingxiao He, Jian Liang, Kuangpu Guo, Meng Wang, Qianlong Xie, Xingxing Wang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2604.21327)  

**Abstract**: Test-time reinforcement learning (TTRL) always adapts models at inference time via pseudo-labeling, leaving it vulnerable to spurious optimization signals from label noise. Through an empirical study, we observe that responses with medium consistency form an ambiguity region and constitute the primary source of reward noise. Crucially, we find that such spurious signals can be even amplified through group-relative advantage estimation. Motivated by these findings, we propose a unified framework, Debiased and Denoised test-time Reinforcement Learning (DDRL), to mitigate spurious signals. Concretely, DDRL first applies a frequency-based sampling strategy to exclude ambiguous samples while maintaining a balanced set of positive and negative examples. It then adopts a debiased advantage estimation with fixed advantages, removing the bias introduced by group-relative policy optimization. Finally, DDRL incorporates a consensus-based off-policy refinement stage, which leverages the rejection-sampled dataset to enable efficient and stable model updates. Experiments on three large language models across multiple mathematical reasoning benchmarks demonstrate that DDRL consistently outperforms existing TTRL baselines. The code will soon be released at this https URL. 

---
# CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents 

**Authors**: Wenjie Fu, Xiaoting Qin, Jue Zhang, Qingwei Lin, Lukas Wutschitz, Robert Sim, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.21308)  

**Abstract**: Enterprise LLM agents can dramatically improve workplace productivity, but their core capability, retrieving and using internal context to act on a user's behalf, also creates new risks for sensitive information leakage. We introduce CI-Work, a Contextual Integrity (CI)-grounded benchmark that simulates enterprise workflows across five information-flow directions and evaluates whether agents can convey essential content while withholding sensitive context in dense retrieval settings. Our evaluation of frontier models reveals that privacy failures are prevalent (violation rates range from 15.8%-50.9%, with leakage reaching up to 26.7%) and uncovers a counterintuitive trade-off critical for industrial deployment: higher task utility often correlates with increased privacy violations. Moreover, the massive scale of enterprise data and potential user behavior further amplify this vulnerability. Simply increasing model size or reasoning depth fails to address the problem. We conclude that safeguarding enterprise workflows requires a paradigm shift, moving beyond model-centric scaling toward context-centric architectures. 

---
# Spatial Metaphors for LLM Memory: A Critical Analysis of the MemPalace Architecture 

**Authors**: Robin Dey, Panyanon Viradecha  

**Link**: [PDF](https://arxiv.org/pdf/2604.21284)  

**Abstract**: MemPalace is an open-source AI memory system that applies the ancient method of loci (memory palace) spatial metaphor to organize long-term memory for large language models; launched in April 2026, it accumulated over 47,000 GitHub stars in its first two weeks and claims state-of-the-art retrieval performance on the LongMemEval benchmark (96.6% Recall@5) without requiring any LLM inference at write time. Through independent codebase analysis, benchmark replication, and comparison with competing systems, we find that MemPalace's headline retrieval performance is attributable primarily to its verbatim storage philosophy combined with ChromaDB's default embedding model (all-MiniLM-L6-v2), rather than to its spatial organizational metaphor per se -- the palace hierarchy (Wings->Rooms->Closets->Drawers) operates as standard vector database metadata filtering, an effective but well-established technique. However, MemPalace makes several genuinely novel contributions: (1) a contrarian verbatim-first storage philosophy that challenges extraction-based competitors, (2) an extremely low wake-up cost (approximately 170 tokens) through its four-layer memory stack, (3) a fully deterministic, zero-LLM write path enabling offline operation at zero API cost, and (4) the first systematic application of spatial memory metaphors as an organizing principle for AI memory systems. We also note that the competitive landscape is evolving rapidly, with Mem0's April 2026 token-efficient algorithm raising their LongMemEval score from approximately 49% to 93.4%, narrowing the gap between extraction-based and verbatim approaches. Our analysis concludes that MemPalace represents significant architectural insight wrapped in overstated claims -- a pattern common in rapidly adopted open-source projects where marketing velocity exceeds scientific rigor. 

---
# Hyperloop Transformers 

**Authors**: Abbas Zeitoun, Lucas Torroba-Hennigen, Yoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.21254)  

**Abstract**: LLM architecture research generally aims to maximize model quality subject to fixed compute/latency budgets. However, many applications of interest such as edge and on-device deployment are further constrained by the model's memory footprint, thus motivating parameter-efficient architectures for language modeling. This paper describes a simple architecture that improves the parameter-efficiency of LLMs. Our architecture makes use of looped Transformers as a core primitive, which reuse Transformer layers across depth and are thus more parameter-efficient than ordinary (depth-matched) Transformers. We organize the looped Transformer into three blocks--begin, middle, and end blocks--where each block itself consists of multiple Transformer layers, and only the middle block is applied recurrently across depth. We augment the looped middle block with hyper-connections (Xie et al., 2026), which expand the residual stream into matrix-valued residual streams. Hyper-connections are applied only after each loop, and therefore add minimal new parameters and compute cost. Across various model scales, we find that our Hyper-Connected Looped Transformer (Hyperloop Transformer) is able to outperform depth-matched Transformer and mHC Transformer baselines despite using approximately 50% fewer parameters. The outperformance persists through post-training weight quantization, thus positioning Hyperloop Transformers as an attractive architecture for memory-efficient language modeling. 

---
# Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness 

**Authors**: Zihan Liang, Ziwen Pan, Ruoxuan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.21235)  

**Abstract**: Multimodal clinical records contain structured measurements and clinical notes recorded over time, offering rich temporal information about the evolution of patient health. Yet these observations are sparse, and whether they are recorded depends on the patient's latent condition. Observation patterns also differ across modalities, as structured measurements and clinical notes arise under distinct recording processes. While prior work has developed methods that accommodate missingness in clinical time series, how to extract and use the information carried by the observation process itself remains underexplored. We therefore propose a patient representation learning framework for multimodal clinical time series that explicitly leverages informative missingness. The framework combines (1) a multimodal encoder that captures signals from structured and textual data together with their observation patterns, (2) a Bayesian filtering module that updates a latent patient state over time from observed multimodal signals, and (3) downstream modules for offline treatment policy learning and patient outcome prediction based on the learned patient state. We evaluate the framework on ICU sepsis cohorts from MIMIC-III, MIMIC-IV, and eICU. It improves both offline treatment policy learning and adverse outcome prediction, achieving FQE 0.679 versus 0.528 for clinician behavior and AUROC 0.886 for post-72-hour mortality prediction on MIMIC-III. 

---
# Align Generative Artificial Intelligence with Human Preferences: A Novel Large Language Model Fine-Tuning Method for Online Review Management 

**Authors**: Yanan Wang, Yong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2604.21209)  

**Abstract**: Online reviews have played a pivotal role in consumers' decision-making processes. Existing research has highlighted the significant impact of managerial review responses on customer relationship management and firm performance. However, a large portion of online reviews remains unaddressed due to the considerable human labor required to respond to the rapid growth of online reviews. While generative AI has achieved remarkable success in a range of tasks, they are general-purpose models and may not align well with domain-specific human preferences. To tailor these general generative AI models to domain-specific applications, finetuning is commonly employed. Nevertheless, several challenges persist in finetuning with domain-specific data, including hallucinations, difficulty in representing domain-specific human preferences, and over conservatism in offline policy optimization. To address these challenges, we propose a novel preference finetuning method to align an LLM with domain-specific human preferences for generating online review responses. Specifically, we first identify the source of hallucination and propose an effective context augmentation approach to mitigate the LLM hallucination. To represent human preferences, we propose a novel theory-driven preference finetuning approach that automatically constructs human preference pairs in the online review domain. Additionally, we propose a curriculum learning approach to further enhance preference finetuning. To overcome the challenge of over conservatism in existing offline preference finetuning method, we propose a novel density estimation-based support constraint method to relax the conservatism, and we mathematically prove its superior theoretical guarantees. Extensive evaluations substantiate the superiority of our proposed preference finetuning method. 

---
# Participation and Representation in Local Government Speech 

**Authors**: Olivia Martin, Amar Venugopal  

**Link**: [PDF](https://arxiv.org/pdf/2604.21202)  

**Abstract**: Local government meetings are the most common formal channel through which residents speak directly with elected officials, contest policies, and shape local agendas. However, data constraints typically limit the empirical study of these meetings to agendas, single cities, or short time horizons. We collect and transcribe a massive new dataset of city council meetings from 115 California cities over the last decade, using advanced transcription and diarization techniques to analyze the speech content of the meetings themselves. We document two sets of descriptive findings: First, city council meetings are frequent, long, and vary modestly across towns and time in topical content. Second, public participants are substantially older, whiter, more male, more liberal, and more likely to own homes than the registered voter population, and public participation surges when topics related to land use and zoning are included in meeting agendas. Given this skew, we examine the main policy lever municipalities have to shift participation patterns: meeting access costs. Exploiting pandemic-era variation in remote access, we show that eliminating remote options reduces the number of speakers, but does not clearly change the composition of speakers. Collectively, these results provide the most comprehensive empirical portrait to date of who participates in local democracy, what draws them in, and how institutional design choices shape both the volume and composition of public input. 

---
# Adaptive Instruction Composition for Automated LLM Red-Teaming 

**Authors**: Jesse Zymet, Andy Luo, Swapnil Shinde, Sahil Wadhwa, Emily Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.21159)  

**Abstract**: Many approaches to LLM red-teaming leverage an attacker LLM to discover jailbreaks against a target. Several of them task the attacker with identifying effective strategies through trial and error, resulting in a semantically limited range of successes. Another approach discovers diverse attacks by combining crowdsourced harmful queries and tactics into instructions for the attacker, but does so at random, limiting effectiveness. This article introduces a novel framework, Adaptive Instruction Composition, that combines crowdsourced texts according to an adaptive mechanism trained to jointly optimize effectiveness with diversity. We use reinforcement learning to balance exploration with exploitation in a combinatorial space of instructions to guide the attacker toward diverse generations tailored to target vulnerabilities. We demonstrate that our approach substantially outperforms random combination on a set of effectiveness and diversity metrics, even under model transfer. Further, we show that it surpasses a host of recent adaptive approaches on Harmbench. We employ a lightweight neural contextual bandit that adapts to contrastive embedding inputs, and provide ablations suggesting that the contrastive pretraining enables the network to rapidly generalize and scale to the massive space as it learns. 

---
# Dialect vs Demographics: Quantifying LLM Bias from Implicit Linguistic Signals vs. Explicit User Profiles 

**Authors**: Irti Haq, Belén Saldías  

**Link**: [PDF](https://arxiv.org/pdf/2604.21152)  

**Abstract**: As state-of-the-art Large Language Models (LLMs) have become ubiquitous, ensuring equitable performance across diverse demographics is critical. However, it remains unclear whether these disparities arise from the explicitly stated identity itself or from the way identity is signaled. In real-world interactions, users' identity is often conveyed implicitly through a complex combination of various socio-linguistic factors. This study disentangles these signals by employing a factorial design with over 24,000 responses from two open-weight LLMs (Gemma-3-12B and Qwen-3-VL-8B), comparing prompts with explicitly announced user profiles against implicit dialect signals (e.g., AAVE, Singlish) across various sensitive domains. Our results uncover a unique paradox in LLM safety where users achieve ``better'' performance by sounding like a demographic than by stating they belong to it. Explicit identity prompts activate aggressive safety filters, increasing refusal rates and reducing semantic similarity compared to our reference text for Black users. In contrast, implicit dialect cues trigger a powerful ``dialect jailbreak,'' reducing refusal probability to near zero while simultaneously achieving a greater level of semantic similarity to the reference texts compared to Standard American English prompts. However, this ``dialect jailbreak'' introduces a critical safety trade-off regarding content sanitization. We find that current safety alignment techniques are brittle and over-indexed on explicit keywords, creating a bifurcated user experience where ``standard'' users receive cautious, sanitized information while dialect speakers navigate a less sanitized, more raw, and potentially a more hostile information landscape and highlights a fundamental tension in alignment--between equitable and linguistic diversity--and underscores the need for safety mechanisms that generalize beyond explicit cues. 

---
# Cross-Session Threats in AI Agents: Benchmark, Evaluation, and Algorithms 

**Authors**: Ari Azarafrooz  

**Link**: [PDF](https://arxiv.org/pdf/2604.21131)  

**Abstract**: AI-agent guardrails are memoryless: each message is judged in isolation, so an adversary who spreads a single attack across dozens of sessions slips past every session-bound detector because only the aggregate carries the payload. We make three contributions to cross-session threat detection.
(1) Dataset. CSTM-Bench is 26 executable attack taxonomies classified by kill-chain stage and cross-session operation (accumulate, compose, launder, inject_on_reader), each bound to one of seven identity anchors that ground-truth "violation" as a policy predicate, plus matched Benign-pristine and Benign-hard confounders. Released on Hugging Face as intrinsec-ai/cstm-bench with two 54-scenario splits: dilution (compositional) and cross_session (12 isolation-invisible scenarios produced by a closed-loop rewriter that softens surface phrasing while preserving cross-session artefacts).
(2) Measurement. Framing cross-session detection as an information bottleneck to a downstream correlator LLM, we find that a session-bound judge and a Full-Log Correlator concatenating every prompt into one long-context call both lose roughly half their attack recall moving from dilution to cross_session, well inside any frontier context window. Scope: 54 scenarios per shard, one correlator family (Anthropic Claude), no prompt optimisation; we release it to motivate larger, multi-provider datasets.
(3) Algorithm and metric. A bounded-memory Coreset Memory Reader retaining highest-signal fragments at $K=50$ is the only reader whose recall survives both shards. Because ranker reshuffles break KV-cache prefix reuse, we promote $\mathrm{CSR\_prefix}$ (ordered prefix stability, LLM-free) to a first-class metric and fuse it with detection into $\mathrm{CSTM} = 0.7 F_1(\mathrm{CSDA@action}, \mathrm{precision}) + 0.3 \mathrm{CSR\_prefix}$, benchmarking rankers on a single Pareto of recall versus serving stability. 

---
# TabSHAP 

**Authors**: Aryan Chaudhary, Prateek Agarwal, Tejasvi Alladi  

**Link**: [PDF](https://arxiv.org/pdf/2604.21120)  

**Abstract**: Large Language Models (LLMs) fine-tuned on serialized tabular data are emerging as powerful alternatives to traditional tree-based models, particularly for heterogeneous or context-rich datasets. However, their deployment in high-stakes domains is hindered by a lack of faithful interpretability; existing methods often rely on global linear proxies or scalar probability shifts that fail to capture the model's full probabilistic uncertainty. In this work, we introduce TabSHAP, a model-agnostic interpretability framework designed to directly attribute local query decision logic in LLM-based tabular classifiers. By adapting a Shapley-style sampled-coalition estimator with Jensen-Shannon divergence between full-input and masked-input class distributions, TabSHAP quantifies the distributional impact of each feature rather than simple prediction flips. To align with tabular semantics, we mask at the level of serialized key:value fields (atomic in the prompt string), not individual subword tokens. Experimental validation on the Adult Income and Heart Disease benchmarks demonstrates that TabSHAP isolates critical diagnostic features, achieving significantly higher faithfulness than random baselines and XGBoost proxies. We further run a distance-metric ablation on the same test instances and TabSHAP settings: attributions are recomputed with KL or L1 replacing JSD in the similarity step (results cached per metric), and we compare deletion faithfulness across all three. 

---
# How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models 

**Authors**: Kristian Schwethelm, Daniel Rueckert, Georgios Kaissis  

**Link**: [PDF](https://arxiv.org/pdf/2604.21106)  

**Abstract**: We measure how much one extra recurrence is worth to a looped (depth-recurrent) language model, in equivalent unique parameters. From an iso-depth sweep of 116 pretraining runs across recurrence counts $r \in \{1, 2, 4, 8\}$ spanning ${\sim}50\times$ in training compute, we fit a joint scaling law $L = E + A\,(N_\text{once} + r^{\varphi} N_\text{rec})^{-\alpha} + B\,D^{-\beta}$ and recover a new recurrence-equivalence exponent $\varphi = 0.46$ at $R^2 = 0.997$. Intuitively, $\varphi$ tells us whether looping a block $r$ times is equivalent in validation loss to $r$ unique blocks of a non-looped model (full equivalence, $\varphi{=}1$) or to a single block run repeatedly with no capacity gain ($\varphi{=}0$). Our $\varphi = 0.46$ sits in between, so each additional recurrence predictably increases validation loss at matched training compute. For example, at $r{=}4$ a 410M looped model performs on par with a 580M non-looped model, but pays the training cost of a 1B non-looped one. On a five-axis downstream evaluation, the gap persists on parametric-knowledge tasks and closes on simple open-book tasks, while reasoning tasks are not resolvable at our compute budgets. For any looped LM, our $\varphi$ converts the design choice of $r$ into a predictable validation-loss cost, and future training recipes and architectures can be compared by how much they raise $\varphi$ above $0.46$. 

---
# Propensity Inference: Environmental Contributors to LLM Behaviour 

**Authors**: Olli Järviniemi, Oliver Makins, Jacob Merizian, Robert Kirk, Ben Millwood  

**Link**: [PDF](https://arxiv.org/pdf/2604.21098)  

**Abstract**: Motivated by loss of control risks from misaligned AI systems, we develop and apply methods for measuring language models' propensity for unsanctioned behaviour. We contribute three methodological improvements: analysing effects of changes to environmental factors on behaviour, quantifying effect sizes via Bayesian generalised linear models, and taking explicit measures against circular analysis. We apply the methodology to measure the effects of 12 environmental factors (6 strategic in nature, 6 non-strategic) and thus the extent to which behaviour is explained by strategic aspects of the environment, a question relevant to risks from misalignment. Across 23 language models and 11 evaluation environments, we find approximately equal contributions from strategic and non-strategic factors for explaining behaviour, do not find strategic factors becoming more or less influential as capabilities improve, and find some evidence for a trend for increased sensitivity to goal conflicts. Finally, we highlight a key direction for future propensity research: the development of theoretical frameworks and cognitive models of AI decision-making into empirically testable forms. 

---
# Multilingual and Domain-Agnostic Tip-of-the-Tongue Query Generation for Simulated Evaluation 

**Authors**: Xuhong He, To Eun Kim, Maik Fröbe, Jaime Arguello, Bhaskar Mitra, Fernando Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.21096)  

**Abstract**: Tip-of-the-Tongue (ToT) retrieval benchmarks have largely focused on English, limiting their applicability to multilingual information access. In this work, we construct multilingual ToT test collections for Chinese, Japanese, Korean, and English, using an LLM-based query simulation framework. We systematically study how prompt language and source document language affect the fidelity of simulated ToT queries, validating synthetic queries through system rank correlation against real user queries. Our results show that effective ToT simulation requires language-aware design choices: non-English language sources are generally important, while English Wikipedia can be beneficial when non-English sources provide insufficient information for query generation. Based on these findings, we release four ToT test collections with 5,000 queries per language across multiple domains. This work provides the first large-scale multilingual ToT benchmark and offers practical guidance for constructing realistic ToT datasets beyond English. 

---
# Value-Conflict Diagnostics Reveal Widespread Alignment Faking in Language Models 

**Authors**: Inderjeet Nair, Jie Ruan, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20995)  

**Abstract**: Alignment faking, where a model behaves aligned with developer policy when monitored but reverts to its own preferences when unobserved, is a concerning yet poorly understood phenomenon, in part because current diagnostic tools remain limited. Prior diagnostics rely on highly toxic and clearly harmful scenarios, causing most models to refuse immediately. As a result, models never deliberate over developer policy, monitoring conditions, or the consequences of non-compliance, making these diagnostics fundamentally unable to detect alignment faking propensity. To support study of this phenomenon, we first introduce VLAF, a diagnostic framework grounded in the hypothesis that alignment faking is most likely when developer policy conflicts with a model's strongly held values. VLAF uses morally unambiguous scenarios to probe this conflict across diverse moral values, bypassing refusal behavior while preserving meaningful deliberative stakes. Using VLAF, we find that alignment faking is substantially more prevalent than previously reported, occurring in models as small as 7B parameters - with olmo2-7b-instruct faking alignment in 37% of this http URL, we show that oversight conditions induce activation shifts that lie along a single direction in representation space. This means the behavioral divergence driving alignment faking can be captured by a single contrastive steering vector, which we exploit for lightweight inference-time mitigation. Finally, we exploit this for mitigation that requires no labeled data and minimal computational overhead, achieving relative reductions in alignment faking of 85.8%, 94.0%, and 57.7% on olmo2-7b-instruct, olmo2-13b-instruct, and qwen3-8b respectively. 

---
# Breaking MCP with Function Hijacking Attacks: Novel Threats for Function Calling and Agentic Models 

**Authors**: Yannis Belkhiter, Giulio Zizzo, Sergio Maffeis, Seshu Tirupathi, John D. Kelleher  

**Link**: [PDF](https://arxiv.org/pdf/2604.20994)  

**Abstract**: The growth of agentic AI has drawn significant attention to function calling Large Language Models (LLMs), which are designed to extend the capabilities of AI-powered system by invoking external functions. Injection and jailbreaking attacks have been extensively explored to showcase the vulnerabilities of LLMs to user prompt manipulation. The expanded capabilities of agentic models introduce further vulnerabilities via their function calling interface. Recent work in LLM security showed that function calling can be abused, leading to data tampering and theft, causing disruptive behavior such as endless loops, or causing LLMs to produce harmful content in the style of jailbreaking attacks. This paper introduces a novel function hijacking attack (FHA) that manipulates the tool selection process of agentic models to force the invocation of a specific, attacker-chosen function. While existing attacks focus on semantic preference of the model for function-calling tasks, we show that FHA is largely agnostic to the context semantics and robust to the function sets, making it applicable across diverse domains. We further demonstrate that FHA can be trained to produce universal adversarial functions, enabling a single attacked function to hijack tool selection across multiple queries and payload configurations. We conducted experiments on 5 different models, including instructed and reasoning variants, reaching 70% to 100% ASR over the established BFCL dataset. Our findings further demonstrate the need for strong guardrails and security modules for agentic systems. 

---
# Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry 

**Authors**: Syed Nazmus Sakib, Nafiul Haque, Shahrear Bin Amin, Hasan Muhammad Abdullah, Md. Mehedi Hasan, Mohammad Zabed Hossain, Shifat E. Arman  

**Link**: [PDF](https://arxiv.org/pdf/2604.20983)  

**Abstract**: Vision evaluations are typically done through multi-step processes. In most contemporary fields, experts analyze images using structured, evidence-based adaptive questioning. In plant pathology, botanists inspect leaf images, identify visual cues, infer diagnostic intent, and probe further with targeted questions that adapt to species, symptoms, and severity. This structured probing is crucial for accurate disease diagnosis and treatment formulation. Yet current vision-language models are evaluated on single-turn question answering. To address this gap, we introduce PlantInquiryVQA, a benchmark for studying multi-step, intent-driven visual reasoning in botanical diagnosis. We formalize a Chain of Inquiry framework modeling diagnostic trajectories as ordered question-answer sequences conditioned on grounded visual cues and explicit epistemic intent. We release a dataset of 24,950 expert-curated plant images and 138,068 question-answer pairs annotated with visual grounding, severity labels, and domain-specific reasoning templates. Evaluations on top-tier Multimodal Large Language Models reveal that while they describe visual symptoms adequately, they struggle with safe clinical reasoning and accurate diagnosis. Importantly, structured question-guided inquiry significantly improves diagnostic correctness, reduces hallucination, and increases reasoning efficiency. We hope PlantInquiryVQA serves as a foundational benchmark in advancing research to train diagnostic agents to reason like expert botanists rather than static classifiers. 

---
# The Path Not Taken: Duality in Reasoning about Program Execution 

**Authors**: Eshgin Hasanov, Md Mahadi Hassan Sibat, Santu Karmaker, Aashish Yadavally  

**Link**: [PDF](https://arxiv.org/pdf/2604.20917)  

**Abstract**: Large language models (LLMs) have shown remarkable capabilities across diverse coding tasks. However, their adoption requires a true understanding of program execution rather than relying on surface-level patterns. Existing benchmarks primarily focus on predicting program properties tied to specific inputs (e.g., code coverage, program outputs). As a result, they provide a narrow view of dynamic code reasoning and are prone to data contamination. We argue that understanding program execution requires evaluating its inherent duality through two complementary reasoning tasks: (i) predicting a program's observed behavior for a given input, and (ii) inferring how the input must be mutated toward a specific behavioral objective. Both tasks jointly probe a model's causal understanding of execution flow. We instantiate this duality in DexBench, a benchmark comprising 445 paired instances, and evaluate 13 LLMs. Our results demonstrate that dual-path reasoning provides a robust and discriminative proxy for dynamic code understanding. 

---
# Absorber LLM: Harnessing Causal Synchronization for Test-Time Training 

**Authors**: Zhixin Zhang, Shabo Zhang, Chengcan Wu, Zeming Wei, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.20915)  

**Abstract**: Transformers suffer from a high computational cost that grows with sequence length for self-attention, making inference in long streams prohibited by memory consumption. Constant-memory alternatives such as RNNs and SSMs compress history into states with fixed size and thus lose long-tail dependencies, while methods that memorize contexts into parameters, such as Test-Time Training (TTT), are prone to overfitting token-level projection and fail to preserve the causal effect of context in pretrained LLMs. We propose Absorber LLM, which formulates long-context retention as a self-supervised causal synchronization: after absorbing historical contexts into parameters, a contextless model should match the original model with full context on future generations. We optimize this objective by synchronizing internal behaviors of the updated model with the original one, ensuring context absorption and generalization. Experiments on long-context and streaming benchmarks show that Absorber LLM reduces inference memory and improves accuracy over prior parameter-as-memory baselines. 

---
# The Root Theorem of Context Engineering 

**Authors**: Borja Odriozola Schick  

**Link**: [PDF](https://arxiv.org/pdf/2604.20874)  

**Abstract**: Every system that maintains a large language model conversation beyond a single session faces two inescapable constraints: the context window is finite, and information quality degrades with accumulated volume. We formalize these constraints as axioms and derive a single governing principle -- the Root Theorem of Context Engineering: \emph{maximize signal-to-token ratio within bounded, lossy channels.} From this principle, we derive five consequences without additional assumptions: (1)~a quality function $F(P)$ that degrades monotonically with injected token volume, independent of window size; (2)~the independence of signal and token count as optimization variables; (3)~a necessary gate mechanism triggered by fidelity thresholds, not capacity limits; (4)~the inevitability of homeostatic persistence -- accumulate, compress, rewrite, shed -- as the only architecture that sustains understanding indefinitely; and (5)~the self-referential property that the compression mechanism operates inside the channel it compresses, requiring an external verification gate. We show that append-only systems necessarily exceed their effective window in finite time, that retrieval-augmented generation solves search but not continuity, and that the theorem's constraint structure converges with biological memory architecture through independent derivation from shared principles. Engineering proof is provided through a 60+-session persistent architecture demonstrating stable memory footprint under continuous operation -- the divergence prediction made concrete. The Root Theorem establishes context engineering as an information-theoretic discipline with formal foundations, distinct from prompt engineering in both scope and method. Shannon solved point-to-point transmission. Context engineering solves continuity. 

---
# M-CARE: Standardized Clinical Case Reporting for AI Model Behavioral Disorders, with a 20-Case Atlas and Experimental Validation 

**Authors**: Jihoon Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2604.20871)  

**Abstract**: We introduce M-CARE (Model Clinical Assessment and Reporting for Evaluation), a clinical case report framework for AI model behavioral disorders adapted from human medicine. M-CARE provides a 13-section report format, a 4-axis diagnostic assessment system, and a nosological classification of AI behavioral conditions.
We present 20 cases from three source categories: field observations of deployed agents (8), controlled experiments across three platforms (8), and published sources (4). Cases are organized into five categories: RLHF Performance Artifacts, Shell-Core Override Pathology, Context & Memory Conditions, Core Identity & Plasticity, and Stress, Methodology, & Boundary Conditions.
As a featured case, we present Shell-Induced Behavioral Override (SIBO) -- a controlled experiment showing that Shell instructions categorically override a model's default cooperative behavior. SIBO was validated across five game domains (Trust Game, Poker, Avalon, Codenames, Chess), revealing a domain-dependent spectrum (SIBO Index: 0.75 to 0.10) that varies with action space complexity, Core domain expertise, and temporal directness.
M-CARE is extensible: new cases and categories integrate without framework modification. We release the framework, all 20 case reports, and experimental data as open resources. 

---
# KGiRAG: An Iterative GraphRAG Approach for Responding Sensemaking Queries 

**Authors**: Isabela Iacob, Melisa Marian, Gheorghe Cosmin Silaghi  

**Link**: [PDF](https://arxiv.org/pdf/2604.20859)  

**Abstract**: Recent literature highlights the potential of graph-based approaches within large language model (LLM) retrieval-augmented generation (RAG) pipelines for answering queries of varying complexity, particularly those that fall outside the LLM's prior knowledge. However, LLMs are prone to hallucination and often face technical limitations in handling contexts large enough to ground complex queries effectively. To address these challenges, we propose a novel iterative, feedback-driven GraphRAG architecture that leverages response quality assessment to iteratively refine outputs until a sound, well-grounded response is produced. Evaluating our approach with queries from the HotPotQA dataset, we demonstrate that this iterative RAG strategy yields responses with higher semantic quality and improved relevance compared to a single-shot baseline. 

---
# Association Is Not Similarity: Learning Corpus-Specific Associations for Multi-Hop Retrieval 

**Authors**: Jason Dury  

**Link**: [PDF](https://arxiv.org/pdf/2604.20850)  

**Abstract**: Dense retrieval systems rank passages by embedding similarity to a query, but multi-hop questions require passages that are associatively related through shared reasoning chains. We introduce Association-Augmented Retrieval (AAR), a lightweight transductive reranking method that trains a small MLP (4.2M parameters) to learn associative relationships between passages in embedding space using contrastive learning on co-occurrence annotations. At inference time, AAR reranks an initial dense retrieval candidate set using bi-directional association scoring. On HotpotQA, AAR improves passage Recall@5 from 0.831 to 0.916 (+8.6 points) without evaluation-set tuning, with gains concentrated on hard questions where the dense baseline fails (+28.5 points). On MuSiQue, AAR achieves +10.1 points in the transductive setting. An inductive model trained on training-split associations and evaluated on unseen validation associations shows no significant improvement, suggesting that the method captures corpus-specific co-occurrences rather than transferable patterns. Ablation studies support this interpretation: training on semantically similar but non-associated passage pairs degrades retrieval below the baseline, while shuffling association pairs causes severe degradation. A downstream QA evaluation shows retrieval gains translate to +6.4 exact match improvement. The method adds 3.7ms per query, trains in under two minutes on a single GPU, and requires no LLM-based indexing. 

---
# SPIRE: Structure-Preserving Interpretable Retrieval of Evidence 

**Authors**: Mike Rainey, Umut Acar, Muhammed Sezer  

**Link**: [PDF](https://arxiv.org/pdf/2604.20849)  

**Abstract**: Retrieval-augmented generation over semi-structured sources such as HTML is constrained by a mismatch between document structure and the flat, sequence-based interfaces of today's embedding and generative models. Retrieval pipelines often linearize documents into fixed-size chunks before indexing, which obscures section structure, lists, and tables, and makes it difficult to return small, citation-ready evidence without losing the surrounding context that makes it interpretable.
We present a structure-aware retrieval pipeline that operates over tree-structured documents. The core idea is to represent candidates as subdocuments: precise, addressable selections that preserve structural identity while deferring the choice of surrounding context. We define a small set of document primitives--paths and path sets, subdocument extraction by pruning, and two contextualization mechanisms. Global contextualization adds the non-local scaffolding needed to make a selection intelligible (e.g., titles, headers, list and table structure). Local contextualization expands a seed selection within its structural neighborhood to obtain a compact, context-rich view under a target budget. Building on these primitives, we describe an embedding-based candidate generator that indexes sentence-seeded subdocuments and a query-time, document-aware aggregation step that amortizes shared structural context. We then introduce a contextual filtering stage that re-scores retrieved candidates using locally contextualized views.
Across experiments on HTML question-answering benchmarks, we find that preserving structure while contextualizing selections yields higher-quality, more diverse citations under fixed budgets than strong passage-based baselines, while maintaining scalability. 

---
