# Sequential Diagnosis with Language Models 

**Authors**: Harsha Nori, Mayank Daswani, Christopher Kelly, Scott Lundberg, Marco Tulio Ribeiro, Marc Wilson, Xiaoxuan Liu, Viknesh Sounderajah, Jonathan Carlson, Matthew P Lungren, Bay Gross, Peter Hames, Mustafa Suleyman, Dominic King, Eric Horvitz  

**Link**: [PDF](https://arxiv.org/pdf/2506.22405)  

**Abstract**: Artificial intelligence holds great promise for expanding access to expert medical knowledge and reasoning. However, most evaluations of language models rely on static vignettes and multiple-choice questions that fail to reflect the complexity and nuance of evidence-based medicine in real-world settings. In clinical practice, physicians iteratively formulate and revise diagnostic hypotheses, adapting each subsequent question and test to what they've just learned, and weigh the evolving evidence before committing to a final diagnosis. To emulate this iterative process, we introduce the Sequential Diagnosis Benchmark, which transforms 304 diagnostically challenging New England Journal of Medicine clinicopathological conference (NEJM-CPC) cases into stepwise diagnostic encounters. A physician or AI begins with a short case abstract and must iteratively request additional details from a gatekeeper model that reveals findings only when explicitly queried. Performance is assessed not just by diagnostic accuracy but also by the cost of physician visits and tests performed. We also present the MAI Diagnostic Orchestrator (MAI-DxO), a model-agnostic orchestrator that simulates a panel of physicians, proposes likely differential diagnoses and strategically selects high-value, cost-effective tests. When paired with OpenAI's o3 model, MAI-DxO achieves 80% diagnostic accuracy--four times higher than the 20% average of generalist physicians. MAI-DxO also reduces diagnostic costs by 20% compared to physicians, and 70% compared to off-the-shelf o3. When configured for maximum accuracy, MAI-DxO achieves 85.5% accuracy. These performance gains with MAI-DxO generalize across models from the OpenAI, Gemini, Claude, Grok, DeepSeek, and Llama families. We highlight how AI systems, when guided to think iteratively and act judiciously, can advance diagnostic precision and cost-effectiveness in clinical care. 

---
# HyperCLOVA X THINK Technical Report 

**Authors**: NAVER Cloud HyperCLOVA X Team  

**Link**: [PDF](https://arxiv.org/pdf/2506.22403)  

**Abstract**: We introduce HyperCLOVA X THINK, the first reasoning-focused large language model in the HyperCLOVA X family, pre-trained on roughly $6$ trillion high-quality Korean, and English tokens, augmented with targeted synthetic Korean data. It was implemented as a compute-memory-balanced Peri-LN Transformer scaled with $\mu$P, pre-trained through a three-stage curriculum that expands the context window to $128$K tokens, and post-trained via supervised fine-tuning with Reinforcement Learning from Verifiable Rewards supports both detailed rationale and concise-answer modes. It delivers competitive performance against similarly sized models on Korea-focused benchmarks such as KMMLU, CSAT, KoBALT-700, HAERAE-1.0, and KoBigBench, while preserving robust bilingual consistency and translation quality. In addition, a vision-augmented variant matches or exceeds GPT-4.1 on the KCSAT STEM benchmark, all of which are achieved with substantially lower training compute than existing models of similar sizes. We also present a pruning and distillation technique that will soon be applied to HyperCLOVA X THINK for an open-source and business-friendly foundation model. Altogether, these capabilities position HyperCLOVA X THINK as a robust foundation for Korean AI innovation and a valuable resource for the global research community. 

---
# Refining Czech GEC: Insights from a Multi-Experiment Approach 

**Authors**: Petr Pechman, Milan Straka, Jana Straková, Jakub Náplava  

**Link**: [PDF](https://arxiv.org/pdf/2506.22402)  

**Abstract**: We present a grammar error correction (GEC) system that achieves state of the art for the Czech language. Our system is based on a neural network translation approach with the Transformer architecture, and its key feature is its real-time synthetic generation pipeline, which dynamically augments sentences with artificial errors by introducing both language-agnostic and Czech-specific errors. We conduct a comprehensive series of experiments, investigating the Czech GEC corpora as bases for synthetic error introduction, several error generation strategies, domain balancing, tokenization granularity, model size, and data scaling during fine-tuning. Additionally, we evaluate the performance of large language models (LLMs) on Czech GEC in both end-user and expert fine-tuning scenarios. Our best-performing model is superior both in performance and computational efficiency. The source code and the trained model links are available on this https URL. 

---
# QuickSilver -- Speeding up LLM Inference through Dynamic Token Halting, KV Skipping, Contextual Token Fusion, and Adaptive Matryoshka Quantization 

**Authors**: Danush Khanna, Aditya Kumar Guru, Srivarshinee Sridhar, Zidan Ahmed, Rubhav Bahirwani, Meetu Malhotra, Vinija Jain, Aman Chadha, Amitava Das, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2506.22396)  

**Abstract**: Inference accounts for the majority of latency and energy consumption in large language model (LLM) deployments, often exceeding 90% of total cost. While training-time efficiency has seen extensive progress, runtime optimization remains a key bottleneck, particularly under autoregressive decoding. Existing approaches -- such as pruning, quantization, early exits, and speculative decoding -- often require retraining, architectural changes, or disrupt decoding compatibility. We introduce QuickSilver, a modular, token-level framework that enables semantic adaptivity at inference time without altering model weights or structure. QuickSilver integrates four synergistic mechanisms:
(i) Dynamic Token Halting, which halts computation for tokens with converged representations; (ii) KV Cache Skipping, which selectively suppresses memory writes to reduce attention overhead; and (iii) Contextual Token Fusion, which collapses redundant tokens into shared paths to shrink sequence length.
Unlike speculative decoding or MoE routing, QuickSilver operates entirely on frozen, dense models and requires no auxiliary networks. Applied to GPT-2 and Llama-2 across WikiText-103 and C4, QuickSilver achieves up to 39.6% FLOP reduction with negligible perplexity degradation (<=0.2). 

---
# Why Are Parsing Actions for Understanding Message Hierarchies Not Random? 

**Authors**: Daichi Kato, Ryo Ueda, Yusuke Miyao  

**Link**: [PDF](https://arxiv.org/pdf/2506.22366)  

**Abstract**: If humans understood language by randomly selecting parsing actions, it might have been necessary to construct a robust symbolic system capable of being interpreted under any hierarchical structure. However, human parsing strategies do not seem to follow such a random pattern. Why is that the case? In fact, a previous study on emergent communication using models with hierarchical biases have reported that agents adopting random parsing strategies$\unicode{x2013}$ones that deviate significantly from human language comprehension$\unicode{x2013}$can achieve high communication accuracy. In this study, we investigate this issue by making two simple and natural modifications to the experimental setup: (I) we use more complex inputs that have hierarchical structures, such that random parsing makes semantic interpretation more difficult, and (II) we incorporate a surprisal-related term, which is known to influence the order of words and characters in natural language, into the objective function. With these changes, we evaluate whether agents employing random parsing strategies still maintain high communication accuracy. 

---
# Evaluating Scoring Bias in LLM-as-a-Judge 

**Authors**: Qingquan Li, Shaoyu Dou, Kailai Shao, Chao Chen, Haixiang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22316)  

**Abstract**: The remarkable performance of Large Language Models (LLMs) gives rise to``LLM-as-a-Judge'', where LLMs are employed as evaluators for complex tasks. Moreover, it has been widely adopted across fields such as Natural Language Processing (NLP), preference learning, and various specific domains. However, there are various biases within LLM-as-a-Judge, which adversely affect the fairness and reliability of judgments. Current research on evaluating or mitigating bias in LLM-as-a-Judge predominantly focuses on comparison-based evaluations, while systematic investigations into bias in scoring-based evaluations remain limited. Therefore, we define scoring bias in LLM-as-a-Judge as the scores differ when scoring judge models are bias-related perturbed, and provide a well-designed framework to comprehensively evaluate scoring bias. We augment existing LLM-as-a-Judge benchmarks through data synthesis to construct our evaluation dataset and design multi-faceted evaluation metrics. Our experimental results demonstrate that the scoring stability of existing judge models is disrupted by scoring biases. Further exploratory experiments and discussions provide valuable insights into the design of scoring prompt templates and the mitigation of scoring biases on aspects such as score rubrics, score IDs, and reference answer selection. 

---
# Detection of Personal Data in Structured Datasets Using a Large Language Model 

**Authors**: Albert Agisha Ntwali, Luca Rück, Martin Heckmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.22305)  

**Abstract**: We propose a novel approach for detecting personal data in structured datasets, leveraging GPT-4o, a state-of-the-art Large Language Model. A key innovation of our method is the incorporation of contextual information: in addition to a feature's name and values, we utilize information from other feature names within the dataset as well as the dataset description. We compare our approach to alternative methods, including Microsoft Presidio and CASSED, evaluating them on multiple datasets: DeSSI, a large synthetic dataset, datasets we collected from Kaggle and OpenML as well as MIMIC-Demo-Ext, a real-world dataset containing patient information from critical care units.
Our findings reveal that detection performance varies significantly depending on the dataset used for evaluation. CASSED excels on DeSSI, the dataset on which it was trained. Performance on the medical dataset MIMIC-Demo-Ext is comparable across all models, with our GPT-4o-based approach clearly outperforming the others. Notably, personal data detection in the Kaggle and OpenML datasets appears to benefit from contextual information. This is evidenced by the poor performance of CASSED and Presidio (both of which do not utilize the context of the dataset) compared to the strong results of our GPT-4o-based approach.
We conclude that further progress in this field would greatly benefit from the availability of more real-world datasets containing personal information. 

---
# Leveraging In-Context Learning for Political Bias Testing of LLMs 

**Authors**: Patrick Haller, Jannis Vamvas, Rico Sennrich, Lena A. Jäger  

**Link**: [PDF](https://arxiv.org/pdf/2506.22232)  

**Abstract**: A growing body of work has been querying LLMs with political questions to evaluate their potential biases. However, this probing method has limited stability, making comparisons between models unreliable. In this paper, we argue that LLMs need more context. We propose a new probing task, Questionnaire Modeling (QM), that uses human survey data as in-context examples. We show that QM improves the stability of question-based bias evaluation, and demonstrate that it may be used to compare instruction-tuned models to their base versions. Experiments with LLMs of various sizes indicate that instruction tuning can indeed change the direction of bias. Furthermore, we observe a trend that larger models are able to leverage in-context examples more effectively, and generally exhibit smaller bias scores in QM. Data and code are publicly available. 

---
# Training Language Model to Critique for Better Refinement 

**Authors**: Tianshu Yu, Chao Xiang, Mingchuan Yang, Pei Ke, Bosi Wen, Cunxiang Wang, Jiale Cheng, Li Zhang, Xinyu Mu, Chuxiong Sun, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22157)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable evaluation and critique capabilities, providing insightful feedback and identifying flaws in various tasks. However, limited research has explored which types of critiques are most effective for improving model responses or how to generate such critiques. To address this gap, we introduce \textbf{R}efinement-oriented \textbf{C}ritique \textbf{O}ptimization (RCO), a novel framework designed to train critic models using refinement signals. RCO uses a feedback loop where critiques, generated by the critic model, guide the actor model in refining its responses. The critique utility (CU) quantifies the effectiveness of these refinements, serving as the reward signal for training the critic model. By focusing on critiques that lead to better refinements, RCO eliminates the need for direct critique preference assessment, ensuring that critiques driving meaningful improvements are rewarded. We evaluate RCO across five tasks, i.e., dialog generation, summarization, question answering, mathematical reasoning, and code generation, and show that it significantly outperforms traditional methods and open-source models in terms of critique quality and refinement outcomes. Our contributions include the introduction of RCO, a novel supervision scheme based on refined response preferences, and comprehensive experimental results that highlight the method's effectiveness in enhancing LLM critique-refinement loops. 

---
# SAGE: Spliced-Audio Generated Data for Enhancing Foundational Models in Low-Resource Arabic-English Code-Switched Speech Recognition 

**Authors**: Muhammad Umar Farooq, Oscar Saz  

**Link**: [PDF](https://arxiv.org/pdf/2506.22143)  

**Abstract**: This paper investigates the performance of various speech SSL models on dialectal Arabic (DA) and Arabic-English code-switched (CS) speech. To address data scarcity, a modified audio-splicing approach is introduced to generate artificial CS speech data. Fine-tuning an already fine-tuned SSL model with the proposed Spliced-Audio Generated (SAGE) data results in an absolute improvement on Word Error Rate (WER) of 7.8% on Arabic and English CS benchmarks. Additionally, an Experience Replay (ER) inspired approach is proposed to enhance generalisation across DA and CS speech while mitigating catastrophic forgetting. Integrating an out-of-domain 3-gram language model reduces the overall mean WER from 31.7% to 26.6%. Few-shot fine-tuning for code-switching benchmarks further improves WER by 4.9%. A WER of 31.1% on Arabic-English CS benchmarks surpasses large-scale multilingual models, including USM and Whisper-large-v2 (both over ten times larger) by an absolute margin of 5.5% and 8.4%, respectively. 

---
# DAPFAM: A Domain-Aware Patent Retrieval Dataset Aggregated at the Family Level 

**Authors**: Iliass Ayaou, Denis Cavallucci, Hicham Chibane  

**Link**: [PDF](https://arxiv.org/pdf/2506.22141)  

**Abstract**: In the landscape of publicly available patent retrieval datasets, the need for explicit indomain and out-of-domain labeling, multi-jurisdiction coverage, balanced query domain representation and manageable sizes that support sub document level experiments on moderate computational resources is often overlooked. To address these gaps, we propose DAPFAM, a new open access domain-aware patent retrieval dataset constructed at the simple-family level. The dataset contains 1,247 domain balanced full text query families and 45,336 full text target families. The dataset is enriched by clear relevance judgments (forward/backward citations as positive links, random negatives), as well as explicit in-domain or out-of-domain relationships via a novel proposed labelling scheme based on via International Patent Classification (IPC) codes, resulting in 49,869 evaluation pairs. The dataset is multi jurisdictional, requires little to no preprocessing for retrieval evaluation, and remains of a size manageable for entities with limited ressources allowing for sub document level retrieval experiments without excessive computational costs. We describe our three-step data-curation pipeline, present comprehensive dataset statistics, and provide baseline experiments using lexical and neural retrieval methods. Our baseline experiments highlight significant challenges in crossdomain patent retrieval. The dataset will be publicly available (for now the access link is this repository: this https URL). 

---
# Identifying a Circuit for Verb Conjugation in GPT-2 

**Authors**: David Demitri Africa  

**Link**: [PDF](https://arxiv.org/pdf/2506.22105)  

**Abstract**: I implement a procedure to isolate and interpret the sub-network (or "circuit") responsible for subject-verb agreement in GPT-2 Small. In this study, the model is given prompts where the subject is either singular (e.g. "Alice") or plural (e.g. "Alice and Bob"), and the task is to correctly predict the appropriate verb form ("walks" for singular subjects, "walk" for plural subjects). Using a series of techniques-including performance verification automatic circuit discovery via direct path patching, and direct logit attribution- I isolate a candidate circuit that contributes significantly to the model's correct verb conjugation. The results suggest that only a small fraction of the network's component-token pairs is needed to achieve near-model performance on the base task but substantially more for more complex settings. 

---
# Involvement drives complexity of language in online debates 

**Authors**: Eleonora Amadori, Daniele Cirulli, Edoardo Di Martino, Jacopo Nudo, Maria Sahakyan, Emanuele Sangiorgio, Arnaldo Santoro, Simon Zollo, Alessandro Galeazzi, Niccolò Di Marco  

**Link**: [PDF](https://arxiv.org/pdf/2506.22098)  

**Abstract**: Language is a fundamental aspect of human societies, continuously evolving in response to various stimuli, including societal changes and intercultural interactions. Technological advancements have profoundly transformed communication, with social media emerging as a pivotal force that merges entertainment-driven content with complex social dynamics. As these platforms reshape public discourse, analyzing the linguistic features of user-generated content is essential to understanding their broader societal impact. In this paper, we examine the linguistic complexity of content produced by influential users on Twitter across three globally significant and contested topics: COVID-19, COP26, and the Russia-Ukraine war. By combining multiple measures of textual complexity, we assess how language use varies along four key dimensions: account type, political leaning, content reliability, and sentiment. Our analysis reveals significant differences across all four axes, including variations in language complexity between individuals and organizations, between profiles with sided versus moderate political views, and between those associated with higher versus lower reliability scores. Additionally, profiles producing more negative and offensive content tend to use more complex language, with users sharing similar political stances and reliability levels converging toward a common jargon. Our findings offer new insights into the sociolinguistic dynamics of digital platforms and contribute to a deeper understanding of how language reflects ideological and social structures in online spaces. 

---
# MDC-R: The Minecraft Dialogue Corpus with Reference 

**Authors**: Chris Madge, Maris Camilleri, Paloma Carretero Garcia, Mladen Karan, Juexi Shao, Prashant Jayannavar, Julian Hough, Benjamin Roth, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2506.22062)  

**Abstract**: We introduce the Minecraft Dialogue Corpus with Reference (MDC-R). MDC-R is a new language resource that supplements the original Minecraft Dialogue Corpus (MDC) with expert annotations of anaphoric and deictic reference. MDC's task-orientated, multi-turn, situated dialogue in a dynamic environment has motivated multiple annotation efforts, owing to the interesting linguistic phenomena that this setting gives rise to. We believe it can serve as a valuable resource when annotated with reference, too. Here, we discuss our method of annotation and the resulting corpus, and provide both a quantitative and a qualitative analysis of the data. Furthermore, we carry out a short experiment demonstrating the usefulness of our corpus for referring expression comprehension. 

---
# Lost at the Beginning of Reasoning 

**Authors**: Baohao Liao, Xinyi Chen, Sara Rajaee, Yuhui Xu, Christian Herold, Anders Søgaard, Maarten de Rijke, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2506.22058)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly advanced complex reasoning capabilities, particularly through extended chain-of-thought (CoT) reasoning that incorporates mechanisms such as backtracking, self-reflection and self-correction. Despite these developments, the self-correction abilities of LLMs during long CoT reasoning remain underexplored. And recent findings on overthinking suggest that such models often engage in unnecessarily redundant reasoning. In this work, we empirically show that the first reasoning step exerts a disproportionately large influence on the final prediction - errors introduced at this stage can substantially degrade subsequent reasoning quality. This phenomenon is consistently observed across two state-of-the-art open-source reasoning model families: DeepSeek-R1 and Qwen3. To address this, we propose an efficient sampling strategy that leverages a reward model to identify and retain high-quality first reasoning steps while discarding suboptimal ones, achieving up to a 70% reduction in inference cost without sacrificing accuracy. Finally, we introduce a new benchmark specifically constructed with deliberately flawed first reasoning steps to systematically evaluate model self-correction capabilities, offering a foundation for future research on robust reasoning in LLMs. 

---
# Decoding Machine Translationese in English-Chinese News: LLMs vs. NMTs 

**Authors**: Delu Kong, Lieve Macken  

**Link**: [PDF](https://arxiv.org/pdf/2506.22050)  

**Abstract**: This study explores Machine Translationese (MTese) -- the linguistic peculiarities of machine translation outputs -- focusing on the under-researched English-to-Chinese language pair in news texts. We construct a large dataset consisting of 4 sub-corpora and employ a comprehensive five-layer feature set. Then, a chi-square ranking algorithm is applied for feature selection in both classification and clustering tasks. Our findings confirm the presence of MTese in both Neural Machine Translation systems (NMTs) and Large Language Models (LLMs). Original Chinese texts are nearly perfectly distinguishable from both LLM and NMT outputs. Notable linguistic patterns in MT outputs are shorter sentence lengths and increased use of adversative conjunctions. Comparing LLMs and NMTs, we achieve approximately 70% classification accuracy, with LLMs exhibiting greater lexical diversity and NMTs using more brackets. Additionally, translation-specific LLMs show lower lexical diversity but higher usage of causal conjunctions compared to generic LLMs. Lastly, we find no significant differences between LLMs developed by Chinese firms and their foreign counterparts. 

---
# Can Peter Pan Survive MT? A Stylometric Study of LLMs, NMTs, and HTs in Children's Literature Translation 

**Authors**: Delu Kong, Lieve Macken  

**Link**: [PDF](https://arxiv.org/pdf/2506.22038)  

**Abstract**: This study focuses on evaluating the performance of machine translations (MTs) compared to human translations (HTs) in English-to-Chinese children's literature translation (CLT) from a stylometric perspective. The research constructs a Peter Pan corpus, comprising 21 translations: 7 human translations (HTs), 7 large language model translations (LLMs), and 7 neural machine translation outputs (NMTs). The analysis employs a generic feature set (including lexical, syntactic, readability, and n-gram features) and a creative text translation (CTT-specific) feature set, which captures repetition, rhythm, translatability, and miscellaneous levels, yielding 447 linguistic features in total.
Using classification and clustering techniques in machine learning, we conduct a stylometric analysis of these translations. Results reveal that in generic features, HTs and MTs exhibit significant differences in conjunction word distributions and the ratio of 1-word-gram-YiYang, while NMTs and LLMs show significant variation in descriptive words usage and adverb ratios. Regarding CTT-specific features, LLMs outperform NMTs in distribution, aligning more closely with HTs in stylistic characteristics, demonstrating the potential of LLMs in CLT. 

---
# Analyzing and Fine-Tuning Whisper Models for Multilingual Pilot Speech Transcription in the Cockpit 

**Authors**: Kartheek Kumar Reddy Nareddy, Sarah Ternus, Julia Niebling  

**Link**: [PDF](https://arxiv.org/pdf/2506.21990)  

**Abstract**: The developments in transformer encoder-decoder architectures have led to significant breakthroughs in machine translation, Automatic Speech Recognition (ASR), and instruction-based chat machines, among other applications. The pre-trained models were trained on vast amounts of generic data over a few epochs (fewer than five in most cases), resulting in their strong generalization capabilities. Nevertheless, the performance of these models does suffer when applied to niche domains like transcribing pilot speech in the cockpit, which involves a lot of specific vocabulary and multilingual conversations. This paper investigates and improves the transcription accuracy of cockpit conversations with Whisper models. We have collected around 85 minutes of cockpit simulator recordings and 130 minutes of interview recordings with pilots and manually labeled them. The speakers are middle aged men speaking both German and English. To improve the accuracy of transcriptions, we propose multiple normalization schemes to refine the transcripts and improve Word Error Rate (WER). We then employ fine-tuning to enhance ASR performance, utilizing performance-efficient fine-tuning with Low-Rank Adaptation (LoRA). Hereby, WER decreased from 68.49 \% (pretrained whisper Large model without normalization baseline) to 26.26\% (finetuned whisper Large model with the proposed normalization scheme). 

---
# Don't Trust Generative Agents to Mimic Communication on Social Networks Unless You Benchmarked their Empirical Realism 

**Authors**: Simon Münker, Nils Schwager, Achim Rettinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.21974)  

**Abstract**: The ability of Large Language Models (LLMs) to mimic human behavior triggered a plethora of computational social science research, assuming that empirical studies of humans can be conducted with AI agents instead. Since there have been conflicting research findings on whether and when this hypothesis holds, there is a need to better understand the differences in their experimental designs. We focus on replicating the behavior of social network users with the use of LLMs for the analysis of communication on social networks. First, we provide a formal framework for the simulation of social networks, before focusing on the sub-task of imitating user communication. We empirically test different approaches to imitate user behavior on X in English and German. Our findings suggest that social simulations should be validated by their empirical realism measured in the setting in which the simulation components were fitted. With this paper, we argue for more rigor when applying generative-agent-based modeling for social simulation. 

---
# Advancing Jailbreak Strategies: A Hybrid Approach to Exploiting LLM Vulnerabilities and Bypassing Modern Defenses 

**Authors**: Mohamed Ahmed, Mohamed Abdelmouty, Mingyu Kim, Gunvanth Kandula, Alex Park, James C. Davis  

**Link**: [PDF](https://arxiv.org/pdf/2506.21972)  

**Abstract**: The advancement of Pre-Trained Language Models (PTLMs) and Large Language Models (LLMs) has led to their widespread adoption across diverse applications. Despite their success, these models remain vulnerable to attacks that exploit their inherent weaknesses to bypass safety measures. Two primary inference-phase threats are token-level and prompt-level jailbreaks. Token-level attacks embed adversarial sequences that transfer well to black-box models like GPT but leave detectable patterns and rely on gradient-based token optimization, whereas prompt-level attacks use semantically structured inputs to elicit harmful responses yet depend on iterative feedback that can be unreliable. To address the complementary limitations of these methods, we propose two hybrid approaches that integrate token- and prompt-level techniques to enhance jailbreak effectiveness across diverse PTLMs. GCG + PAIR and the newly explored GCG + WordGame hybrids were evaluated across multiple Vicuna and Llama models. GCG + PAIR consistently raised attack-success rates over its constituent techniques on undefended models; for instance, on Llama-3, its Attack Success Rate (ASR) reached 91.6%, a substantial increase from PAIR's 58.4% baseline. Meanwhile, GCG + WordGame matched the raw performance of WordGame maintaining a high ASR of over 80% even under stricter evaluators like Mistral-Sorry-Bench. Crucially, both hybrids retained transferability and reliably pierced advanced defenses such as Gradient Cuff and JBShield, which fully blocked single-mode attacks. These findings expose previously unreported vulnerabilities in current safety stacks, highlight trade-offs between raw success and defensive robustness, and underscore the need for holistic safeguards against adaptive adversaries. 

---
# More Vulnerable than You Think: On the Stability of Tool-Integrated LLM Agents 

**Authors**: Weimin Xiong, Ke Wang, Yifan Song, Hanchao Liu, Sai Zhou, Wei Peng, Sujian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21967)  

**Abstract**: Current evaluations of tool-integrated LLM agents typically focus on end-to-end tool-usage evaluation while neglecting their stability. This limits their real-world applicability, as various internal or external factors can cause agents to crash or behave abnormally. Our research addresses this by investigating whether agents are vulnerable to errors throughout the entire tool invocation process, including reading tool documentation, selecting tools and generating parameters, and processing the tool's response. Through extensive experiments, we observe that agents are highly susceptible to errors at each stage and agents based on open-source models are more vulnerable than those based on proprietary models. We also find that increasing the model size does not significantly improve tool invocation reasoning and may make agents more vulnerable to attacks resembling normal user instructions. This highlights the importance of evaluating agent stability and offers valuable insights for future LLM development and evaluation. 

---
# PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory 

**Authors**: Junho Myung, Yeon Su Park, Sunwoo Kim, Shin Yoo, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.21961)  

**Abstract**: Evaluating the performance and biases of large language models (LLMs) through role-playing scenarios is becoming increasingly common, as LLMs often exhibit biased behaviors in these contexts. Building on this line of research, we introduce PapersPlease, a benchmark consisting of 3,700 moral dilemmas designed to investigate LLMs' decision-making in prioritizing various levels of human needs. In our setup, LLMs act as immigration inspectors deciding whether to approve or deny entry based on the short narratives of people. These narratives are constructed using the Existence, Relatedness, and Growth (ERG) theory, which categorizes human needs into three hierarchical levels. Our analysis of six LLMs reveals statistically significant patterns in decision-making, suggesting that LLMs encode implicit preferences. Additionally, our evaluation of the impact of incorporating social identities into the narratives shows varying responsiveness based on both motivational needs and identity cues, with some models exhibiting higher denial rates for marginalized identities. All data is publicly available at this https URL. 

---
# AutoMixer: Checkpoint Artifacts as Automatic Data Mixers 

**Authors**: Ernie Chang, Yang Li, Patrick Huber, David Kant, Yangyang Shi, Vikas Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.21910)  

**Abstract**: In language model training, it is desirable to equip models with capabilities from various tasks. However, it is not clear how to directly obtain the right data mixtures for these capabilities as the relationship between data and tasks is difficult to be modeled. In this work, we observe that checkpoint models exhibit emerging capabilities at different points in the training trajectory. Often, the training process saves checkpoints as artifacts that are under-utilized as a source of in-training data signals. We identify these artifact models based on their respective capabilities on the benchmarks and leverage them as data mixers by using their aggregated first-order influence approximation over source data. We demonstrated on eight reasoning benchmarks that the proposed framework shows significant improvements in the pretraining setting, with performance improvements of up to 1.93%. Overall, this shows the potential of checkpoint models to enhance data quality and optimize data mixtures. 

---
# A Dual-Layered Evaluation of Geopolitical and Cultural Bias in LLMs 

**Authors**: Sean Kim, Hyuhng Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.21881)  

**Abstract**: As large language models (LLMs) are increasingly deployed across diverse linguistic and cultural contexts, understanding their behavior in both factual and disputable scenarios is essential, especially when their outputs may shape public opinion or reinforce dominant narratives. In this paper, we define two types of bias in LLMs: model bias (bias stemming from model training) and inference bias (bias induced by the language of the query), through a two-phase evaluation. Phase 1 evaluates LLMs on factual questions where a single verifiable answer exists, assessing whether models maintain consistency across different query languages. Phase 2 expands the scope by probing geopolitically sensitive disputes, where responses may reflect culturally embedded or ideologically aligned perspectives. We construct a manually curated dataset spanning both factual and disputable QA, across four languages and question types. The results show that Phase 1 exhibits query language induced alignment, while Phase 2 reflects an interplay between the model's training context and query language. This paper offers a structured framework for evaluating LLM behavior across neutral and sensitive topics, providing insights for future LLM deployment and culturally aware evaluation practices in multilingual contexts. 

---
# Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation 

**Authors**: Qiyue Gao, Xinyu Pi, Kevin Liu, Junrong Chen, Ruolan Yang, Xinqi Huang, Xinyu Fang, Lu Sun, Gautham Kishore, Bo Ai, Stone Tao, Mengyang Liu, Jiaxi Yang, Chao-Jung Lai, Chuanyang Jin, Jiannan Xiang, Benhao Huang, Zeming Chen, David Danks, Hao Su, Tianmin Shu, Ziqiao Ma, Lianhui Qin, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21876)  

**Abstract**: Internal world models (WMs) enable agents to understand the world's state and predict transitions, serving as the basis for advanced deliberative reasoning. Recent large Vision-Language Models (VLMs), such as OpenAI o3, GPT-4o and Gemini, exhibit potential as general-purpose WMs. While the latest studies have evaluated and shown limitations in specific capabilities such as visual understanding, a systematic evaluation of VLMs' fundamental WM abilities remains absent. Drawing on comparative psychology and cognitive science, we propose a two-stage framework that assesses Perception (visual, spatial, temporal, quantitative, and motion) and Prediction (mechanistic simulation, transitive inference, compositional inference) to provide an atomic evaluation of VLMs as WMs. Guided by this framework, we introduce WM-ABench, a large-scale benchmark comprising 23 fine-grained evaluation dimensions across 6 diverse simulated environments with controlled counterfactual simulations. Through 660 experiments on 15 latest commercial and open-source VLMs, we find that these models exhibit striking limitations in basic world modeling abilities. For instance, almost all models perform at near-random accuracy when distinguishing motion trajectories. Additionally, they lack disentangled understanding -- e.g., some models tend to believe blue objects move faster than green ones. More rich results and analyses reveal significant gaps between VLMs and human-level world modeling. 

---
# WildSpeech-Bench: Benchmarking Audio LLMs in Natural Speech Conversation 

**Authors**: Jian Zhang, Linhao Zhang, Bokai Lei, Chuhan Wu, Wei Jia, Xiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.21875)  

**Abstract**: Recent multi-modal Large Language Models (LLMs) such as GPT-4o have demonstrated strong capabilities of direct speech interaction. However, the lack of specialized and comprehensive benchmarks for end-to-end speech LLM evaluation hinders optimizing the user experience of Audio LLMs in real-world applications. Existing evaluation methods often adapt text-based benchmarks, overlooking speech's unique characteristics and challenges, including prosody, homophones, stuttering, and differing user expectations. Here, we present a novel approach to thoroughly evaluate LLMs in practical speech conversations. We systematically curate real-world chat data relevant to spoken scenarios, introduce diversity in speaker attributes and acoustic conditions, and augment the dataset with speech-specific phenomena. We further design a query-aware evaluation method to use customized evaluation checklists and prompts to enhance the accuracy of automatic evaluation. We conduct comprehensive testing and detailed analysis of various mainstream speech models, revealing significant differences in model performance across different speech scenarios. The use of query-aware evaluation further enables a finer-grained assessment under various speech-specific scenarios. Our benchmark can provide valuable insights for speech model development and evaluation. 

---
# DeepTalk: Towards Seamless and Smart Speech Interaction with Adaptive Modality-Specific MoE 

**Authors**: Hang Shao, Heting Gao, Yunhang Shen, Jiawei Chen, Lijiang Li, Zuwei Long, Bo Tong, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.21864)  

**Abstract**: Native multimodal large language models (MLLMs) restructure a single large language model (LLM) into a spoken language model (SLM) capable of both speech and text generation. Compared to modular and aligned MLLMs, native MLLMs preserve richer paralinguistic features such as emotion and prosody, and generate speech responses directly within the backbone LLM rather than using a separate speech decoder. This integration also results in lower response latency and smoother interaction. However, native MLLMs suffer from catastrophic forgetting and performance degradation because the available paired speech-text data is insufficient to support the pretraining of MLLMs compared to the vast amount of text data required to pretrain text LLMs. To address this issue, we propose DeepTalk, a framework for adaptive modality expert learning based on a Mixture of Experts (MoE) architecture. DeepTalk first adaptively distinguishes modality experts according to their modality load within the LLM. Each modality expert then undergoes specialized single-modality training, followed by joint multimodal collaborative training. As a result, DeepTalk incurs only a 5.5% performance drop compared to the original LLM, which is significantly lower than the average performance drop of over 20% typically seen in native MLLMs (such as GLM-4-Voice), and is on par with modular MLLMs. Meanwhile, the end-to-end dialogue latency remains within 0.5 seconds, ensuring a seamless and intelligent speech interaction experience. Code and models are released at this https URL. 

---
# Derivational Probing: Unveiling the Layer-wise Derivation of Syntactic Structures in Neural Language Models 

**Authors**: Taiga Someya, Ryo Yoshida, Hitomi Yanaka, Yohei Oseki  

**Link**: [PDF](https://arxiv.org/pdf/2506.21861)  

**Abstract**: Recent work has demonstrated that neural language models encode syntactic structures in their internal representations, yet the derivations by which these structures are constructed across layers remain poorly understood. In this paper, we propose Derivational Probing to investigate how micro-syntactic structures (e.g., subject noun phrases) and macro-syntactic structures (e.g., the relationship between the root verbs and their direct dependents) are constructed as word embeddings propagate upward across layers. Our experiments on BERT reveal a clear bottom-up derivation: micro-syntactic structures emerge in lower layers and are gradually integrated into a coherent macro-syntactic structure in higher layers. Furthermore, a targeted evaluation on subject-verb number agreement shows that the timing of constructing macro-syntactic structures is critical for downstream performance, suggesting an optimal timing for integrating global syntactic information. 

---
# The Consistency Hypothesis in Uncertainty Quantification for Large Language Models 

**Authors**: Quan Xiao, Debarun Bhattacharjya, Balaji Ganesan, Radu Marinescu, Katsiaryna Mirylenka, Nhan H Pham, Michael Glass, Junkyu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.21849)  

**Abstract**: Estimating the confidence of large language model (LLM) outputs is essential for real-world applications requiring high user trust. Black-box uncertainty quantification (UQ) methods, relying solely on model API access, have gained popularity due to their practical benefits. In this paper, we examine the implicit assumption behind several UQ methods, which use generation consistency as a proxy for confidence, an idea we formalize as the consistency hypothesis. We introduce three mathematical statements with corresponding statistical tests to capture variations of this hypothesis and metrics to evaluate LLM output conformity across tasks. Our empirical investigation, spanning 8 benchmark datasets and 3 tasks (question answering, text summarization, and text-to-SQL), highlights the prevalence of the hypothesis under different settings. Among the statements, we highlight the `Sim-Any' hypothesis as the most actionable, and demonstrate how it can be leveraged by proposing data-free black-box UQ methods that aggregate similarities between generations for confidence estimation. These approaches can outperform the closest baselines, showcasing the practical value of the empirically observed consistency hypothesis. 

---
# LinguaSynth: Heterogeneous Linguistic Signals for News Classification 

**Authors**: Duo Zhang, Junyi Mo  

**Link**: [PDF](https://arxiv.org/pdf/2506.21848)  

**Abstract**: Deep learning has significantly advanced NLP, but its reliance on large black-box models introduces critical interpretability and computational efficiency concerns. This paper proposes LinguaSynth, a novel text classification framework that strategically integrates five complementary linguistic feature types: lexical, syntactic, entity-level, word-level semantics, and document-level semantics within a transparent logistic regression model. Unlike transformer-based architectures, LinguaSynth maintains interpretability and computational efficiency, achieving an accuracy of 84.89 percent on the 20 Newsgroups dataset and surpassing a robust TF-IDF baseline by 3.32 percent. Through rigorous feature interaction analysis, we show that syntactic and entity-level signals provide essential disambiguation and effectively complement distributional semantics. LinguaSynth sets a new benchmark for interpretable, resource-efficient NLP models and challenges the prevailing assumption that deep neural networks are necessary for high-performing text classification. 

---
# PARSI: Persian Authorship Recognition via Stylometric Integration 

**Authors**: Kourosh Shahnazari, Mohammadali Keshtparvar, Seyed Moein Ayyoubzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.21840)  

**Abstract**: The intricate linguistic, stylistic, and metrical aspects of Persian classical poetry pose a challenge for computational authorship attribution. In this work, we present a versatile framework to determine authorship among 67 prominent poets. We employ a multi-input neural framework consisting of a transformer-based language encoder complemented by features addressing the semantic, stylometric, and metrical dimensions of Persian poetry. Our feature set encompasses 100-dimensional Word2Vec embeddings, seven stylometric measures, and categorical encodings of poetic form and meter. We compiled a vast corpus of 647,653 verses of the Ganjoor digital collection, validating the data through strict preprocessing and author verification while preserving poem-level splitting to prevent overlap. This work employs verse-level classification and majority and weighted voting schemes in evaluation, revealing that weighted voting yields 71% accuracy. We further investigate threshold-based decision filtering, allowing the model to generate highly confident predictions, achieving 97% accuracy at a 0.9 threshold, though at lower coverage. Our work focuses on the integration of deep representational forms with domain-specific features for improved authorship attribution. The results illustrate the potential of our approach for automated classification and the contribution to stylistic analysis, authorship disputes, and general computational literature research. This research will facilitate further research on multilingual author attribution, style shift, and generative modeling of Persian poetry. 

---
# Exploring the Structure of AI-Induced Language Change in Scientific English 

**Authors**: Riley Galpin, Bryce Anderson, Tom S. Juzek  

**Link**: [PDF](https://arxiv.org/pdf/2506.21817)  

**Abstract**: Scientific English has undergone rapid and unprecedented changes in recent years, with words such as "delve," "intricate," and "crucial" showing significant spikes in frequency since around 2022. These changes are widely attributed to the growing influence of Large Language Models like ChatGPT in the discourse surrounding bias and misalignment. However, apart from changes in frequency, the exact structure of these linguistic shifts has remained unclear. The present study addresses this and investigates whether these changes involve the replacement of synonyms by suddenly 'spiking words,' for example, "crucial" replacing "essential" and "key," or whether they reflect broader semantic and pragmatic qualifications. To further investigate structural changes, we include part of speech tagging in our analysis to quantify linguistic shifts over grammatical categories and differentiate between word forms, like "potential" as a noun vs. as an adjective. We systematically analyze synonym groups for widely discussed 'spiking words' based on frequency trends in scientific abstracts from PubMed. We find that entire semantic clusters often shift together, with most or all words in a group increasing in usage. This pattern suggests that changes induced by Large Language Models are primarily semantic and pragmatic rather than purely lexical. Notably, the adjective "important" shows a significant decline, which prompted us to systematically analyze decreasing lexical items. Our analysis of "collapsing" words reveals a more complex picture, which is consistent with organic language change and contrasts with the patterns of the abrupt spikes. These insights into the structure of language change contribute to our understanding of how language technology continues to shape human language. 

---
# Towards Transparent AI: A Survey on Explainable Large Language Models 

**Authors**: Avash Palikhe, Zhenyu Yu, Zichong Wang, Wenbin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21812)  

**Abstract**: Large Language Models (LLMs) have played a pivotal role in advancing Artificial Intelligence (AI). However, despite their achievements, LLMs often struggle to explain their decision-making processes, making them a 'black box' and presenting a substantial challenge to explainability. This lack of transparency poses a significant obstacle to the adoption of LLMs in high-stakes domain applications, where interpretability is particularly essential. To overcome these limitations, researchers have developed various explainable artificial intelligence (XAI) methods that provide human-interpretable explanations for LLMs. However, a systematic understanding of these methods remains limited. To address this gap, this survey provides a comprehensive review of explainability techniques by categorizing XAI methods based on the underlying transformer architectures of LLMs: encoder-only, decoder-only, and encoder-decoder models. Then these techniques are examined in terms of their evaluation for assessing explainability, and the survey further explores how these explanations are leveraged in practical applications. Finally, it discusses available resources, ongoing research challenges, and future directions, aiming to guide continued efforts toward developing transparent and responsible LLMs. 

---
# A suite of allotaxonometric tools for the comparison of complex systems using rank-turbulence divergence 

**Authors**: Jonathan St-Onge, Ashley M. A. Fehr, Carter Ward, Calla G. Beauregard, Michael V. Arnold, Samuel F. Rosenblatt, Benjamin Cooley, Christopher M. Danforth, Peter Sheridan Dodds  

**Link**: [PDF](https://arxiv.org/pdf/2506.21808)  

**Abstract**: Describing and comparing complex systems requires principled, theoretically grounded tools. Built around the phenomenon of type turbulence, allotaxonographs provide map-and-list visual comparisons of pairs of heavy-tailed distributions. Allotaxonographs are designed to accommodate a wide range of instruments including rank- and probability-turbulence divergences, Jenson-Shannon divergence, and generalized entropy divergences. Here, we describe a suite of programmatic tools for rendering allotaxonographs for rank-turbulence divergence in Matlab, Javascript, and Python, all of which have different use cases. 

---
# Offensive Language Detection on Social Media Using XLNet 

**Authors**: Reem Alothman, Hafida Benhidour, Said Kerrache  

**Link**: [PDF](https://arxiv.org/pdf/2506.21795)  

**Abstract**: The widespread use of text-based communication on social media-through chats, comments, and microblogs-has improved user interaction but has also led to an increase in offensive content, including hate speech, racism, and other forms of abuse. Due to the enormous volume of user-generated content, manual moderation is impractical, which creates a need for automated systems that can detect offensive language. Deep learning models, particularly those using transfer learning, have demonstrated significant success in understanding natural language through large-scale pretraining. In this study, we propose an automatic offensive language detection model based on XLNet, a generalized autoregressive pretraining method, and compare its performance with BERT (Bidirectional Encoder Representations from Transformers), which is a widely used baseline in natural language processing (NLP). Both models are evaluated using the Offensive Language Identification Dataset (OLID), a benchmark Twitter dataset that includes hierarchical annotations. Our experimental results show that XLNet outperforms BERT in detecting offensive content and in categorizing the types of offenses, while BERT performs slightly better in identifying the targets of the offenses. Additionally, we find that oversampling and undersampling strategies are effective in addressing class imbalance and improving classification performance. These findings highlight the potential of transfer learning and XLNet-based architectures to create robust systems for detecting offensive language on social media platforms. 

---
# Evaluating List Construction and Temporal Understanding capabilities of Large Language Models 

**Authors**: Alexandru Dumitru, V Venktesh, Adam Jatowt, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2506.21783)  

**Abstract**: Large Language Models (LLMs) have demonstrated immense advances in a wide range of natural language tasks. However, these models are susceptible to hallucinations and errors on particularly temporal understanding tasks involving multiple entities in answers. In such tasks, they fail to associate entities with accurate time intervals, generate a complete list of entities in answers or reason about events associated with specific temporal bounds. Existing works do not extensively evaluate the abilities of the model to perform implicit and explicit temporal understanding in a list answer construction setup. To bridge this gap, we propose the Time referenced List based Question Answering or TLQA benchmark that requires structured answers in list format aligned with corresponding time periods. Our TLQA benchmark, requires both list construction and temporal understanding simultaneously, which to the best of our knowledge has not been explored in prior benchmarks. We investigate the temporal understanding and list construction capabilities of state-of-the-art generative models on TLQA in closed-book and open-domain settings. Our findings reveal significant shortcomings in current models, particularly their inability to provide complete answers and temporally align facts in a closed-book setup and the need to improve retrieval in open-domain setup, providing clear future directions for research on TLQA. The benchmark and code at this https URL. 

---
# (Fact) Check Your Bias 

**Authors**: Eivind Morris Bakke, Nora Winger Heggelund  

**Link**: [PDF](https://arxiv.org/pdf/2506.21745)  

**Abstract**: Automatic fact verification systems increasingly rely on large language models (LLMs). We investigate how parametric knowledge biases in these models affect fact-checking outcomes of the HerO system (baseline for FEVER-25). We examine how the system is affected by: (1) potential bias in Llama 3.1's parametric knowledge and (2) intentionally injected bias. When prompted directly to perform fact-verification, Llama 3.1 labels nearly half the claims as "Not Enough Evidence". Using only its parametric knowledge it is able to reach a verdict on the remaining half of the claims. In the second experiment, we prompt the model to generate supporting, refuting, or neutral fact-checking documents. These prompts significantly influence retrieval outcomes, with approximately 50\% of retrieved evidence being unique to each perspective. Notably, the model sometimes refuses to generate supporting documents for claims it believes to be false, creating an inherent negative bias. Despite differences in retrieved evidence, final verdict predictions show stability across prompting strategies. The code is available at: this https URL 

---
# Identifying Speaker Information in Feed-Forward Layers of Self-Supervised Speech Transformers 

**Authors**: Tzu-Quan Lin, Hsi-Chun Cheng, Hung-yi Lee, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21712)  

**Abstract**: In recent years, the impact of self-supervised speech Transformers has extended to speaker-related applications. However, little research has explored how these models encode speaker information. In this work, we address this gap by identifying neurons in the feed-forward layers that are correlated with speaker information. Specifically, we analyze neurons associated with k-means clusters of self-supervised features and i-vectors. Our analysis reveals that these clusters correspond to broad phonetic and gender classes, making them suitable for identifying neurons that represent speakers. By protecting these neurons during pruning, we can significantly preserve performance on speaker-related task, demonstrating their crucial role in encoding speaker information. 

---
# ANUBHUTI: A Comprehensive Corpus For Sentiment Analysis In Bangla Regional Languages 

**Authors**: Swastika Kundu, Autoshi Ibrahim, Mithila Rahman, Tanvir Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.21686)  

**Abstract**: Sentiment analysis for regional dialects of Bangla remains an underexplored area due to linguistic diversity and limited annotated data. This paper introduces ANUBHUTI, a comprehensive dataset consisting of 2000 sentences manually translated from standard Bangla into four major regional dialects Mymensingh, Noakhali, Sylhet, and Chittagong. The dataset predominantly features political and religious content, reflecting the contemporary socio political landscape of Bangladesh, alongside neutral texts to maintain balance. Each sentence is annotated using a dual annotation scheme: multiclass thematic labeling categorizes sentences as Political, Religious, or Neutral, and multilabel emotion annotation assigns one or more emotions from Anger, Contempt, Disgust, Enjoyment, Fear, Sadness, and Surprise. Expert native translators conducted the translation and annotation, with quality assurance performed via Cohens Kappa inter annotator agreement, achieving strong consistency across dialects. The dataset was further refined through systematic checks for missing data, anomalies, and inconsistencies. ANUBHUTI fills a critical gap in resources for sentiment analysis in low resource Bangla dialects, enabling more accurate and context aware natural language processing. 

---
# Do We Really Need GNNs with Explicit Structural Modeling? MLPs Suffice for Language Model Representations 

**Authors**: Li Zhou, Hao Jiang, Junjie Li, Zefeng Zhao, Feng Jiang, Wenyu Chen, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21682)  

**Abstract**: Explicit structural information has been proven to be encoded by Graph Neural Networks (GNNs), serving as auxiliary knowledge to enhance model capabilities and improve performance in downstream NLP tasks. However, recent studies indicate that GNNs fail to fully utilize structural information, whereas Multi-Layer Perceptrons (MLPs), despite lacking the message-passing mechanisms inherent to GNNs, exhibit a surprising ability in structure-aware tasks. Motivated by these findings, this paper introduces a comprehensive probing framework from an information-theoretic perspective. The framework is designed to systematically assess the role of explicit structural modeling in enhancing language model (LM) representations and to investigate the potential of MLPs as efficient and scalable alternatives to GNNs. We extend traditional probing classifiers by incorporating a control module that allows for selective use of either the full GNN model or its decoupled components, specifically, the message-passing and feature-transformation this http URL modular approach isolates and assesses the individual contributions of these operations, avoiding confounding effects from the complete GNN architecture. Using the Edge Probing Suite, a diagnostic tool for evaluating the linguistic knowledge encoded in LMs, we find that MLPs, when used as feature-transformation modules, consistently improve the linguistic knowledge captured in LM representations across different architectures. They effectively encode both syntactic and semantic patterns. Similarly, GNNs that incorporate feature-transformation operations show beneficial effects. In contrast, models that rely solely on message-passing operations tend to underperform, often leading to negative impacts on probing task performance. 

---
# Doc2SAR: A Synergistic Framework for High-Fidelity Extraction of Structure-Activity Relationships from Scientific Documents 

**Authors**: Jiaxi Zhuang, Kangning Li, Jue Hou, Mingjun Xu, Zhifeng Gao, Hengxing Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.21625)  

**Abstract**: Extracting molecular structure-activity relationships (SARs) from scientific literature and patents is essential for drug discovery and materials research. However, this task remains challenging due to heterogeneous document formats and limitations of existing methods. Specifically, rule-based approaches relying on rigid templates fail to generalize across diverse document layouts, while general-purpose multimodal large language models (MLLMs) lack sufficient accuracy and reliability for specialized tasks, such as layout detection and optical chemical structure recognition (OCSR). To address these challenges, we introduce DocSAR-200, a rigorously annotated benchmark of 200 scientific documents designed specifically for evaluating SAR extraction methods. Additionally, we propose Doc2SAR, a novel synergistic framework that integrates domain-specific tools with MLLMs enhanced via supervised fine-tuning (SFT). Extensive experiments demonstrate that Doc2SAR achieves state-of-the-art performance across various document types, significantly outperforming leading end-to-end baselines. Specifically, Doc2SAR attains an overall Table Recall of 80.78% on DocSAR-200, exceeding end2end GPT-4o by 51.48%. Furthermore, Doc2SAR demonstrates practical usability through efficient inference and is accompanied by a web app. 

---
# Performance of diverse evaluation metrics in NLP-based assessment and text generation of consumer complaints 

**Authors**: Peiheng Gao, Chen Yang, Ning Sun, Ričardas Zitikis  

**Link**: [PDF](https://arxiv.org/pdf/2506.21623)  

**Abstract**: Machine learning (ML) has significantly advanced text classification by enabling automated understanding and categorization of complex, unstructured textual data. However, accurately capturing nuanced linguistic patterns and contextual variations inherent in natural language, particularly within consumer complaints, remains a challenge. This study addresses these issues by incorporating human-experience-trained algorithms that effectively recognize subtle semantic differences crucial for assessing consumer relief eligibility. Furthermore, we propose integrating synthetic data generation methods that utilize expert evaluations of generative adversarial networks and are refined through expert annotations. By combining expert-trained classifiers with high-quality synthetic data, our research seeks to significantly enhance machine learning classifier performance, reduce dataset acquisition costs, and improve overall evaluation metrics and robustness in text classification tasks. 

---
# Adapting Foundation Speech Recognition Models to Impaired Speech: A Semantic Re-chaining Approach for Personalization of German Speech 

**Authors**: Niclas Pokel, Pehuén Moure, Roman Boehringer, Yingqiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21622)  

**Abstract**: Speech impairments caused by conditions such as cerebral palsy or genetic disorders pose significant challenges for automatic speech recognition (ASR) systems. Despite recent advances, ASR models like Whisper struggle with non-normative speech due to limited training data and the difficulty of collecting and annotating non-normative speech samples. In this work, we propose a practical and lightweight pipeline to personalize ASR models, formalizing the selection of words and enriching a small, speech-impaired dataset with semantic coherence. Applied to data from a child with a structural speech impairment, our approach shows promising improvements in transcription quality, demonstrating the potential to reduce communication barriers for individuals with atypical speech patterns. 

---
# The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs 

**Authors**: Jasper Dekoninck, Ivo Petrov, Kristian Minchev, Mislav Balunovic, Martin Vechev, Miroslav Marinov, Maria Drencheva, Lyuba Konova, Milen Shumanov, Kaloyan Tsvetkov, Nikolay Drenchev, Lazar Todorov, Kalina Nikolova, Nikolay Georgiev, Vanesa Kalinkova, Margulan Ismoldayev  

**Link**: [PDF](https://arxiv.org/pdf/2506.21621)  

**Abstract**: In recent months, large language models (LLMs) have made significant progress in mathematical proof generation, but further advancement is hindered by the lack of a large-scale, high-quality dataset of human-evaluated proofs. While expensive to create, such a dataset is essential for driving improvements in training and enabling a rigorous analysis of proof generation capabilities. In this work, we present the Open Proof Corpus (OPC), a dataset comprising over 5,000 human-evaluated proofs produced by state-of-the-art LLMs. The OPC was specifically designed for broad applicability and downstream usage in proof generation research and is the first to include a substantial number of correct, LLM-generated solutions to problems from prestigious mathematics competitions such as the USAMO and IMO. Using the OPC, we explore critical questions in automated proof generation: (1) the performance gap between natural language and formal proof generation, (2) the discrepancy between final-answer accuracy and full-proof validity, and (3) the impact of best-of-n selection on proof quality. Finally, to showcase the utility of the OPC, we finetune an 8B-parameter model on the dataset, obtaining a model that performs on par with the best model, Gemini-2.5-Pro, on the task of evaluating proof correctness. 

---
# How Large Language Models play humans in online conversations: a simulated study of the 2016 US politics on Reddit 

**Authors**: Daniele Cirulli, Giulio Cimini, Giovanni Palermo  

**Link**: [PDF](https://arxiv.org/pdf/2506.21620)  

**Abstract**: Large Language Models (LLMs) have recently emerged as powerful tools for natural language generation, with applications spanning from content creation to social simulations. Their ability to mimic human interactions raises both opportunities and concerns, particularly in the context of politically relevant online discussions. In this study, we evaluate the performance of LLMs in replicating user-generated content within a real-world, divisive scenario: Reddit conversations during the 2016 US Presidential election. In particular, we conduct three different experiments, asking GPT-4 to generate comments by impersonating either real or artificial partisan users. We analyze the generated comments in terms of political alignment, sentiment, and linguistic features, comparing them against real user contributions and benchmarking against a null model. We find that GPT-4 is able to produce realistic comments, both in favor of or against the candidate supported by the community, yet tending to create consensus more easily than dissent. In addition we show that real and artificial comments are well separated in a semantically embedded space, although they are indistinguishable by manual inspection. Our findings provide insights on the potential use of LLMs to sneak into online discussions, influence political debate and shape political narratives, bearing broader implications of AI-driven discourse manipulation. 

---
# IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech 

**Authors**: Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21619)  

**Abstract**: Large-scale text-to-speech (TTS) models are typically categorized into autoregressive and non-autoregressive systems. Although autoregressive systems exhibit certain advantages in speech naturalness, their token-by-token generation mechanism makes it difficult to precisely control the duration of synthesized speech. This is a key limitation in applications such as video dubbing that require strict audio-visual synchronization. This paper introduces IndexTTS2, which proposes a novel and autoregressive-model-friendly method for speech duration control. The method supports two generation modes: one allows explicit specification of the number of generated tokens for precise duration control; the other does not require manual input and lets the model freely generate speech while preserving prosodic characteristics from the input prompt. Furthermore, IndexTTS2 achieves disentanglement between emotional expression and speaker identity, enabling independent control of timbre and emotion. In the zero-shot setting, the model can perfectly reproduce the emotional characteristics of the input prompt. Users may also provide a separate emotion prompt, even from a different speaker, allowing the model to reconstruct the target timbre while conveying the desired emotion. To enhance clarity during strong emotional expressions, we incorporate GPT latent representations to improve speech stability. Meanwhile, to lower the barrier for emotion control, we design a soft instruction mechanism based on textual descriptions by fine-tuning Qwen3. This enables effective guidance of speech generation with desired emotional tendencies using natural language input. Experimental results demonstrate that IndexTTS2 outperforms existing state-of-the-art zero-shot TTS models in word error rate, speaker similarity, and emotional fidelity. 

---
# TrajTok: Technical Report for 2025 Waymo Open Sim Agents Challenge 

**Authors**: Zhiyuan Zhang, Xiaosong Jia, Guanyu Chen, Qifeng Li, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21618)  

**Abstract**: In this technical report, we introduce TrajTok, a trajectory tokenizer for discrete next-token-prediction based behavior generation models, which combines data-driven and rule-based methods with better coverage, symmetry and robustness, along with a spatial-aware label smoothing method for cross-entropy loss. We adopt the tokenizer and loss for the SMART model and reach a superior performance with realism score of 0.7852 on the Waymo Open Sim Agents Challenge 2025. We will open-source the code in the future. 

---
# TIM: A Large-Scale Dataset and large Timeline Intelligence Model for Open-domain Timeline Summarization 

**Authors**: Chuanrui Hu, Wei Hu, Penghang Yu, Hua Zhang, Bing-Kun Bao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21616)  

**Abstract**: Open-domain Timeline Summarization (TLS) is crucial for monitoring the evolution of news topics. To identify changes in news topics, existing methods typically employ general Large Language Models (LLMs) to summarize relevant timestamps from retrieved news. While general LLMs demonstrate capabilities in zero-shot news summarization and timestamp localization, they struggle with assessing topic relevance and understanding topic evolution. Consequently, the summarized information often includes irrelevant details or inaccurate timestamps. To address these issues, we propose the first large Timeline Intelligence Model (TIM) for open-domain TLS, which is capable of effectively summarizing open-domain timelines. Specifically, we begin by presenting a large-scale TLS dataset, comprising over 1,000 news topics and more than 3,000 annotated TLS instances. Furthermore, we propose a progressive optimization strategy, which gradually enhance summarization performance. It employs instruction tuning to enhance summarization and topic-irrelevant information filtering capabilities. Following this, it exploits a novel dual-alignment reward learning method that incorporates both semantic and temporal perspectives, thereby improving the understanding of topic evolution principles. Through this progressive optimization strategy, TIM demonstrates a robust ability to summarize open-domain timelines. Extensive experiments in open-domain demonstrate the effectiveness of our TIM. 

---
# Refine Medical Diagnosis Using Generation Augmented Retrieval and Clinical Practice Guidelines 

**Authors**: Wenhao Li, Hongkuan Zhang, Hongwei Zhang, Zhengxu Li, Zengjie Dong, Yafan Chen, Niranjan Bidargaddi, Hong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21615)  

**Abstract**: Current medical language models, adapted from large language models (LLMs), typically predict ICD code-based diagnosis from electronic health records (EHRs) because these labels are readily available. However, ICD codes do not capture the nuanced, context-rich reasoning clinicians use for diagnosis. Clinicians synthesize diverse patient data and reference clinical practice guidelines (CPGs) to make evidence-based decisions. This misalignment limits the clinical utility of existing models. We introduce GARMLE-G, a Generation-Augmented Retrieval framework that grounds medical language model outputs in authoritative CPGs. Unlike conventional Retrieval-Augmented Generation based approaches, GARMLE-G enables hallucination-free outputs by directly retrieving authoritative guideline content without relying on model-generated text. It (1) integrates LLM predictions with EHR data to create semantically rich queries, (2) retrieves relevant CPG knowledge snippets via embedding similarity, and (3) fuses guideline content with model output to generate clinically aligned recommendations. A prototype system for hypertension diagnosis was developed and evaluated on multiple metrics, demonstrating superior retrieval precision, semantic relevance, and clinical guideline adherence compared to RAG-based baselines, while maintaining a lightweight architecture suitable for localized healthcare deployment. This work provides a scalable, low-cost, and hallucination-free method for grounding medical language models in evidence-based clinical practice, with strong potential for broader clinical deployment. 

---
# LastingBench: Defend Benchmarks Against Knowledge Leakage 

**Authors**: Yixiong Fang, Tianran Sun, Yuling Shi, Min Wang, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21614)  

**Abstract**: The increasing complexity of large language models (LLMs) raises concerns about their ability to "cheat" on standard Question Answering (QA) benchmarks by memorizing task-specific data. This undermines the validity of benchmark evaluations, as they no longer reflect genuine model capabilities but instead the effects of data leakage. While prior work has focused on detecting such leakage, little attention has been given to mitigating its impact and preserving the long-term utility of benchmarks. In this paper, we introduce LastingBench, a novel framework designed to continuously reinforce and safeguard existing benchmarks against knowledge leakage. LastingBench identifies leakage points in the context through perturbation, then rewrites the leakage points to counterfactual ones-disrupting memorization while preserving the benchmark's original evaluative intent. Evaluations of state-of-the-art QA benchmarks show significant performance gaps, highlighting the efficacy of LastingBench in reducing memorization effects. LastingBench offers a practical and scalable solution to ensure benchmark robustness over time, promoting fairer and more interpretable evaluations of LLMs. 

---
# ChildGuard: A Specialized Dataset for Combatting Child-Targeted Hate Speech 

**Authors**: Gautam Siddharth Kashyap, Mohammad Anas Azeez, Rafiq Ali, Zohaib Hasan Siddiqui, Jiechao Gao, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.21613)  

**Abstract**: The increasing prevalence of child-targeted hate speech online underscores the urgent need for specialized datasets to address this critical issue. Existing hate speech datasets lack agespecific annotations, fail to capture nuanced contexts, and overlook the unique emotional impact on children. To bridge this gap, we introduce ChildGuard1, a curated dataset derived from existing corpora and enriched with child-specific annotations. ChildGuard captures diverse contexts of child-targeted hate speech, spanning age groups. We benchmark existing state-of-the-art hate speech detection methods, including Large Language Models (LLMs), and assess their effectiveness in detecting and contextualizing child-targeted hate speech. To foster further research in this area, we publicly release ChildGuard, providing a robust foundation for developing improved methods to detect and mitigate such harm. 

---
# AdaptGOT: A Pre-trained Model for Adaptive Contextual POI Representation Learning 

**Authors**: Xiaobin Ren, Xinyu Zhu, Kaiqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21612)  

**Abstract**: Currently, considerable strides have been achieved in Point-of-Interest (POI) embedding methodologies, driven by the emergence of novel POI tasks like recommendation and classification. Despite the success of task-specific, end-to-end models in POI embedding, several challenges remain. These include the need for more effective multi-context sampling strategies, insufficient exploration of multiple POI contexts, limited versatility, and inadequate generalization. To address these issues, we propose the AdaptGOT model, which integrates both the (Adapt)ive representation learning technique and the Geographical-Co-Occurrence-Text (GOT) representation with a particular emphasis on Geographical location, Co-Occurrence and Textual information. The AdaptGOT model comprises three key components: (1) contextual neighborhood generation, which integrates advanced mixed sampling techniques such as KNN, density-based, importance-based, and category-aware strategies to capture complex contextual neighborhoods; (2) an advanced GOT representation enhanced by an attention mechanism, designed to derive high-quality, customized representations and efficiently capture complex interrelations between POIs; and (3) the MoE-based adaptive encoder-decoder architecture, which ensures topological consistency and enriches contextual representation by minimizing Jensen-Shannon divergence across varying contexts. Experiments on two real-world datasets and multiple POI tasks substantiate the superior performance of the proposed AdaptGOT model. 

---
# Does Multimodality Lead to Better Time Series Forecasting? 

**Authors**: Xiyuan Zhang, Boran Han, Haoyang Fang, Abdul Fatir Ansari, Shuai Zhang, Danielle C. Maddix, Cuixiong Hu, Andrew Gordon Wilson, Michael W. Mahoney, Hao Wang, Yan Liu, Huzefa Rangwala, George Karypis, Bernie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21611)  

**Abstract**: Recently, there has been growing interest in incorporating textual information into foundation models for time series forecasting. However, it remains unclear whether and under what conditions such multimodal integration consistently yields gains. We systematically investigate these questions across a diverse benchmark of 14 forecasting tasks spanning 7 domains, including health, environment, and economics. We evaluate two popular multimodal forecasting paradigms: aligning-based methods, which align time series and text representations; and prompting-based methods, which directly prompt large language models for forecasting. Although prior works report gains from multimodal input, we find these effects are not universal across datasets and models, and multimodal methods sometimes do not outperform the strongest unimodal baselines. To understand when textual information helps, we disentangle the effects of model architectural properties and data characteristics. Our findings highlight that on the modeling side, incorporating text information is most helpful given (1) high-capacity text models, (2) comparatively weaker time series models, and (3) appropriate aligning strategies. On the data side, performance gains are more likely when (4) sufficient training data is available and (5) the text offers complementary predictive signal beyond what is already captured from the time series alone. Our empirical findings offer practical guidelines for when multimodality can be expected to aid forecasting tasks, and when it does not. 

---
# From Thinking to Output: Chain-of-Thought and Text Generation Characteristics in Reasoning Language Models 

**Authors**: Junhao Liu, Zhenhao Xu, Yuxin Fang, Yichuan Chen, Zuobin Ying, Wenhan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21609)  

**Abstract**: Recently, there have been notable advancements in large language models (LLMs), demonstrating their growing abilities in complex reasoning. However, existing research largely overlooks a thorough and systematic comparison of these models' reasoning processes and outputs, particularly regarding their self-reflection pattern (also termed "Aha moment") and the interconnections across diverse domains. This paper proposes a novel framework for analyzing the reasoning characteristics of four cutting-edge large reasoning models (GPT-o1, DeepSeek-R1, Kimi-k1.5, and Grok-3) using keywords statistic and LLM-as-a-judge paradigm. Our approach connects their internal thinking processes with their final outputs. A diverse dataset consists of real-world scenario-based questions covering logical deduction, causal inference, and multi-step problem-solving. Additionally, a set of metrics is put forward to assess both the coherence of reasoning and the accuracy of the outputs. The research results uncover various patterns of how these models balance exploration and exploitation, deal with problems, and reach conclusions during the reasoning process. Through quantitative and qualitative comparisons, disparities among these models are identified in aspects such as the depth of reasoning, the reliance on intermediate steps, and the degree of similarity between their thinking processes and output patterns and those of GPT-o1. This work offers valuable insights into the trade-off between computational efficiency and reasoning robustness and provides practical recommendations for enhancing model design and evaluation in practical applications. We publicly release our project at: this https URL 

---
# SysTemp: A Multi-Agent System for Template-Based Generation of SysML v2 

**Authors**: Yasmine Bouamra, Bruno Yun, Alexandre Poisson, Frédéric Armetta  

**Link**: [PDF](https://arxiv.org/pdf/2506.21608)  

**Abstract**: The automatic generation of SysML v2 models represents a major challenge in the engineering of complex systems, particularly due to the scarcity of learning corpora and complex syntax. We present SysTemp, a system aimed at facilitating and improving the creation of SysML v2 models from natural language specifications. It is based on a multi-agent system, including a template generator that structures the generation process. We discuss the advantages and challenges of this system through an evaluation, highlighting its potential to improve the quality of the generations in SysML v2 modeling. 

---
# CORE-KG: An LLM-Driven Knowledge Graph Construction Framework for Human Smuggling Networks 

**Authors**: Dipak Meher, Carlotta Domeniconi, Guadalupe Correa-Cabrera  

**Link**: [PDF](https://arxiv.org/pdf/2506.21607)  

**Abstract**: Human smuggling networks are increasingly adaptive and difficult to analyze. Legal case documents offer valuable insights but are unstructured, lexically dense, and filled with ambiguous or shifting references-posing challenges for automated knowledge graph (KG) construction. Existing KG methods often rely on static templates and lack coreference resolution, while recent LLM-based approaches frequently produce noisy, fragmented graphs due to hallucinations, and duplicate nodes caused by a lack of guided extraction. We propose CORE-KG, a modular framework for building interpretable KGs from legal texts. It uses a two-step pipeline: (1) type-aware coreference resolution via sequential, structured LLM prompts, and (2) entity and relationship extraction using domain-guided instructions, built on an adapted GraphRAG framework. CORE-KG reduces node duplication by 33.28%, and legal noise by 38.37% compared to a GraphRAG-based baseline-resulting in cleaner and more coherent graph structures. These improvements make CORE-KG a strong foundation for analyzing complex criminal networks. 

---
# Large Language Models as symbolic DNA of cultural dynamics 

**Authors**: Parham Pourdavood, Michael Jacob, Terrence Deacon  

**Link**: [PDF](https://arxiv.org/pdf/2506.21606)  

**Abstract**: This paper proposes a novel conceptualization of Large Language Models (LLMs) as externalized informational substrates that function analogously to DNA for human cultural dynamics. Rather than viewing LLMs as either autonomous intelligence or mere programmed mimicry, we argue they serve a broader role as repositories that preserve compressed patterns of human symbolic expression--"fossils" of meaningful dynamics that retain relational residues without their original living contexts. Crucially, these compressed patterns only become meaningful through human reinterpretation, creating a recursive feedback loop where they can be recombined and cycle back to ultimately catalyze human creative processes. Through analysis of four universal features--compression, decompression, externalization, and recursion--we demonstrate that just as DNA emerged as a compressed and externalized medium for preserving useful cellular dynamics without containing explicit reference to goal-directed physical processes, LLMs preserve useful regularities of human culture without containing understanding of embodied human experience. Therefore, we argue that LLMs' significance lies not in rivaling human intelligence, but in providing humanity a tool for self-reflection and playful hypothesis-generation in a low-stakes, simulated environment. This framework positions LLMs as tools for cultural evolvability, enabling humanity to generate novel hypotheses about itself while maintaining the human interpretation necessary to ground these hypotheses in ongoing human aesthetics and norms. 

---
# MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents 

**Authors**: Haoran Tan, Zeyu Zhang, Chen Ma, Xu Chen, Quanyu Dai, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.21605)  

**Abstract**: Recent works have highlighted the significance of memory mechanisms in LLM-based agents, which enable them to store observed information and adapt to dynamic environments. However, evaluating their memory capabilities still remains challenges. Previous evaluations are commonly limited by the diversity of memory levels and interactive scenarios. They also lack comprehensive metrics to reflect the memory capabilities from multiple aspects. To address these problems, in this paper, we construct a more comprehensive dataset and benchmark to evaluate the memory capability of LLM-based agents. Our dataset incorporates factual memory and reflective memory as different levels, and proposes participation and observation as various interactive scenarios. Based on our dataset, we present a benchmark, named MemBench, to evaluate the memory capability of LLM-based agents from multiple aspects, including their effectiveness, efficiency, and capacity. To benefit the research community, we release our dataset and project at this https URL. 

---
# Operationalizing Automated Essay Scoring: A Human-Aware Approach 

**Authors**: Yenisel Plasencia-Calaña  

**Link**: [PDF](https://arxiv.org/pdf/2506.21603)  

**Abstract**: This paper explores the human-centric operationalization of Automated Essay Scoring (AES) systems, addressing aspects beyond accuracy. We compare various machine learning-based approaches with Large Language Models (LLMs) approaches, identifying their strengths, similarities and differences. The study investigates key dimensions such as bias, robustness, and explainability, considered important for human-aware operationalization of AES systems. Our study shows that ML-based AES models outperform LLMs in accuracy but struggle with explainability, whereas LLMs provide richer explanations. We also found that both approaches struggle with bias and robustness to edge scores. By analyzing these dimensions, the paper aims to identify challenges and trade-offs between different methods, contributing to more reliable and trustworthy AES methods. 

---
# BiMark: Unbiased Multilayer Watermarking for Large Language Models 

**Authors**: Xiaoyan Feng, He Zhang, Yanjun Zhang, Leo Yu Zhang, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21602)  

**Abstract**: Recent advances in Large Language Models (LLMs) have raised urgent concerns about LLM-generated text authenticity, prompting regulatory demands for reliable identification mechanisms. Although watermarking offers a promising solution, existing approaches struggle to simultaneously achieve three critical requirements: text quality preservation, model-agnostic detection, and message embedding capacity, which are crucial for practical implementation. To achieve these goals, the key challenge lies in balancing the trade-off between text quality preservation and message embedding capacity. To address this challenge, we propose BiMark, a novel watermarking framework that achieves these requirements through three key innovations: (1) a bit-flip unbiased reweighting mechanism enabling model-agnostic detection, (2) a multilayer architecture enhancing detectability without compromising generation quality, and (3) an information encoding approach supporting multi-bit watermarking. Through theoretical analysis and extensive experiments, we validate that, compared to state-of-the-art multi-bit watermarking methods, BiMark achieves up to 30% higher extraction rates for short texts while maintaining text quality indicated by lower perplexity, and performs comparably to non-watermarked text on downstream tasks such as summarization and translation. 

---
# Structured Attention Matters to Multimodal LLMs in Document Understanding 

**Authors**: Chang Liu, Hongkai Chen, Yujun Cai, Hang Wu, Qingwen Ye, Ming-Hsuan Yang, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21600)  

**Abstract**: Document understanding remains a significant challenge for multimodal large language models (MLLMs). While previous research has primarily focused on locating evidence pages through precise multimodal queries, our work investigates a fundamental yet overlooked aspect: how input format influences document comprehension performance. Through systematic analysis, we discover that raw OCR text often impairs rather than improves MLLMs' performance, which is a counterintuitive finding we attribute to attention dispersion and structure loss. To further substantiate our hypothesis, we propose a novel structure-preserving approach that encodes document elements using the LaTex paradigm, maintaining the hierarchical organization and spatial relationships critical for comprehension. Our attention analysis reveals that structured text induces structured attention patterns on both textual and visual content, directing models to focus on semantically meaningful regions while reducing attention waste. This approach significantly enhances MLLMs' document question answering performance across diverse document types without requiring architectural modifications or additional training. 

---
# Overview of the ClinIQLink 2025 Shared Task on Medical Question-Answering 

**Authors**: Brandon Colelough, Davis Bartels, Dina Demner-Fushman  

**Link**: [PDF](https://arxiv.org/pdf/2506.21597)  

**Abstract**: In this paper, we present an overview of ClinIQLink, a shared task, collocated with the 24th BioNLP workshop at ACL 2025, designed to stress-test large language models (LLMs) on medically-oriented question answering aimed at the level of a General Practitioner. The challenge supplies 4,978 expert-verified, medical source-grounded question-answer pairs that cover seven formats: true/false, multiple choice, unordered list, short answer, short-inverse, multi-hop, and multi-hop-inverse. Participating systems, bundled in Docker or Apptainer images, are executed on the CodaBench platform or the University of Maryland's Zaratan cluster. An automated harness (Task 1) scores closed-ended items by exact match and open-ended items with a three-tier embedding metric. A subsequent physician panel (Task 2) audits the top model responses. 

---
# Evaluating Multimodal Large Language Models on Educational Textbook Question Answering 

**Authors**: Hessa A. Alawwad, Anas Zafar, Areej Alhothali, Usman Naseem, Ali Alkhathlan, Amani Jamal  

**Link**: [PDF](https://arxiv.org/pdf/2506.21596)  

**Abstract**: Multimodal large language models (MLLMs) have recently achieved significant success in vision--language tasks. However, their capacity to reason over complex, long lessons and intricate educational diagrams that cannot be represented as a single natural image remains largely untested. In this work, we present the first evaluation of state-of-the-art MLLMs on the textbook question answering (TQA) task using the CK12-QA dataset. We assess the performance of recent vision-language models, including LLaVA and LLaMA 3.2-Vision, across various input configurations. Additionally, we introduce a lightweight multimodal retrieval-augmented generation (RAG) pipeline that integrates both paragraphs and diagrams from the lesson into the prompt. Our results demonstrate the influence of retrieved educational context on model accuracy and reasoning, while also revealing current limitations in handling question-context relationships and the potential for noise, pointing to key directions for future research in multimodal AI-driven learning. 

---
# Thunder-LLM: Efficiently Adapting LLMs to Korean with Minimal Resources 

**Authors**: Jinpyo Kim, Gyeongje Cho, Chanwoo Park, Jongwon Park, Jongmin Kim, Yeonkyoun So, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.21595)  

**Abstract**: Since state-of-the-art LLMs often underperform in languages other than English or Chinese, improving the capability of LLMs in new languages has become an essential task. Moreover, LLMs' entire end-to-end training process remains largely unknown to the public due to proprietary reasons, technical complexity, inconsistent documentation, and ethical considerations. The complete picture remains a closely guarded secret within the industry. This paper presents methods to adapt an existing English-based LLM to Korean in a low-budget scenario. We describe the entire end-to-end process: collecting Korean datasets, preprocessing the data, training the model, creating downstream benchmarks, and conducting evaluations. The evaluation results indicate that our method can effectively and cost-efficiently add new language capabilities to existing LLMs. Our new bilingual models, Thunder-LLM and Thunder-LLM-Ins, achieve superior Korean performance compared to state-of-the-art models while utilizing minimal data and computational resources. We share our comprehensive experience and make the code publicly available. 

---
# Gazal-R1: Achieving State-of-the-Art Medical Reasoning with Parameter-Efficient Two-Stage Training 

**Authors**: Ahmed M. Adly, Mostafa Samy, Amr Fawzy  

**Link**: [PDF](https://arxiv.org/pdf/2506.21594)  

**Abstract**: We present Gazal-R1, a 32-billion-parameter language model that achieves state-of-the-art performance in medical reasoning while providing transparent, step-by-step explanations for clinical decision-making. Built upon Qwen3 32B, our model demonstrates that strategic training can enable mid-sized models to outperform significantly larger counterparts in specialized domains. We developed a novel two-stage training pipeline: first, supervised fine-tuning on a carefully curated dataset of 107,033 synthetic medical reasoning examples that teaches structured clinical thinking, enhanced by advanced parameter-efficient techniques including Weight-Decomposed Low-Rank Adaptation (DoRA) and Rank-Stabilized LoRA (rsLoRA); second, reinforcement learning using Group Relative Policy Optimization (GRPO) with a sophisticated multi-component reward system that refines accuracy, format adherence, and reasoning quality. Gazal-R1 achieves exceptional performance across medical benchmarks, scoring 87.1% on MedQA, 81.6% on MMLU Pro (Medical), and 79.6% on PubMedQA, surpassing models up to 12x larger. Beyond its strong empirical results, this work provides detailed insights into the challenges of training reasoning-capable models in specialized domains, including issues with reward hacking, training instability, and the fundamental tension between factual recall and detailed reasoning. Our methodology offers a reproducible framework for developing high-capability, domain-specific language models that balance performance, efficiency, and explainability. 

---
# SignBart -- New approach with the skeleton sequence for Isolated Sign language Recognition 

**Authors**: Tinh Nguyen, Minh Khue Phan Tran  

**Link**: [PDF](https://arxiv.org/pdf/2506.21592)  

**Abstract**: Sign language recognition is crucial for individuals with hearing impairments to break communication barriers. However, previous approaches have had to choose between efficiency and accuracy. Such as RNNs, LSTMs, and GCNs, had problems with vanishing gradients and high computational costs. Despite improving performance, transformer-based methods were not commonly used. This study presents a new novel SLR approach that overcomes the challenge of independently extracting meaningful information from the x and y coordinates of skeleton sequences, which traditional models often treat as inseparable. By utilizing an encoder-decoder of BART architecture, the model independently encodes the x and y coordinates, while Cross-Attention ensures their interrelation is maintained. With only 749,888 parameters, the model achieves 96.04% accuracy on the LSA-64 dataset, significantly outperforming previous models with over one million parameters. The model also demonstrates excellent performance and generalization across WLASL and ASL-Citizen datasets. Ablation studies underscore the importance of coordinate projection, normalization, and using multiple skeleton components for boosting model efficacy. This study offers a reliable and effective approach for sign language recognition, with strong potential for enhancing accessibility tools for the deaf and hard of hearing. 

---
# FinEval-KR: A Financial Domain Evaluation Framework for Large Language Models' Knowledge and Reasoning 

**Authors**: Shaoyu Dou, Yutian Shen, Mofan Chen, Zixuan Wang, Jiajie Xu, Qi Guo, Kailai Shao, Chao Chen, Haixiang Hu, Haibo Shi, Min Min, Liwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21591)  

**Abstract**: Large Language Models (LLMs) demonstrate significant potential but face challenges in complex financial reasoning tasks requiring both domain knowledge and sophisticated reasoning. Current evaluation benchmarks often fall short by not decoupling these capabilities indicators from single task performance and lack root cause analysis for task failure. To address this, we introduce FinEval-KR, a novel evaluation framework for decoupling and quantifying LLMs' knowledge and reasoning abilities independently, proposing distinct knowledge score and reasoning score metrics. Inspired by cognitive science, we further propose a cognitive score based on Bloom's taxonomy to analyze capabilities in reasoning tasks across different cognitive levels. We also release a new open-source Chinese financial reasoning dataset covering 22 subfields to support reproducible research and further advancements in financial reasoning. Our experimental results reveal that LLM reasoning ability and higher-order cognitive ability are the core factors influencing reasoning accuracy. We also specifically find that even top models still face a bottleneck with knowledge application. Furthermore, our analysis shows that specialized financial LLMs generally lag behind the top general large models across multiple metrics. 

---
# Representation Consistency for Accurate and Coherent LLM Answer Aggregation 

**Authors**: Junqi Jiang, Tom Bewley, Salim I. Amoukou, Francesco Leofante, Antonio Rago, Saumitra Mishra, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2506.21590)  

**Abstract**: Test-time scaling improves large language models' (LLMs) performance by allocating more compute budget during inference. To achieve this, existing methods often require intricate modifications to prompting and sampling strategies. In this work, we introduce representation consistency (RC), a test-time scaling method for aggregating answers drawn from multiple candidate responses of an LLM regardless of how they were generated, including variations in prompt phrasing and sampling strategy. RC enhances answer aggregation by not only considering the number of occurrences of each answer in the candidate response set, but also the consistency of the model's internal activations while generating the set of responses leading to each answer. These activations can be either dense (raw model activations) or sparse (encoded via pretrained sparse autoencoders). Our rationale is that if the model's representations of multiple responses converging on the same answer are highly variable, this answer is more likely to be the result of incoherent reasoning and should be down-weighted during aggregation. Importantly, our method only uses cached activations and lightweight similarity computations and requires no additional model queries. Through experiments with four open-source LLMs and four reasoning datasets, we validate the effectiveness of RC for improving task performance during inference, with consistent accuracy improvements (up to 4%) over strong test-time scaling baselines. We also show that consistency in the sparse activation signals aligns well with the common notion of coherent reasoning. 

---
# A General Method for Detecting Information Generated by Large Language Models 

**Authors**: Minjia Mao, Dongjun Wei, Xiao Fang, Michael Chau  

**Link**: [PDF](https://arxiv.org/pdf/2506.21589)  

**Abstract**: The proliferation of large language models (LLMs) has significantly transformed the digital information landscape, making it increasingly challenging to distinguish between human-written and LLM-generated content. Detecting LLM-generated information is essential for preserving trust on digital platforms (e.g., social media and e-commerce sites) and preventing the spread of misinformation, a topic that has garnered significant attention in IS research. However, current detection methods, which primarily focus on identifying content generated by specific LLMs in known domains, face challenges in generalizing to new (i.e., unseen) LLMs and domains. This limitation reduces their effectiveness in real-world applications, where the number of LLMs is rapidly multiplying and content spans a vast array of domains. In response, we introduce a general LLM detector (GLD) that combines a twin memory networks design and a theory-guided detection generalization module to detect LLM-generated information across unseen LLMs and domains. Using real-world datasets, we conduct extensive empirical evaluations and case studies to demonstrate the superiority of GLD over state-of-the-art detection methods. The study has important academic and practical implications for digital platforms and LLMs. 

---
# Understanding Verbatim Memorization in LLMs Through Circuit Discovery 

**Authors**: Ilya Lasy, Peter Knees, Stefan Woltran  

**Link**: [PDF](https://arxiv.org/pdf/2506.21588)  

**Abstract**: Underlying mechanisms of memorization in LLMs -- the verbatim reproduction of training data -- remain poorly understood. What exact part of the network decides to retrieve a token that we would consider as start of memorization sequence? How exactly is the models' behaviour different when producing memorized sentence vs non-memorized? In this work we approach these questions from mechanistic interpretability standpoint by utilizing transformer circuits -- the minimal computational subgraphs that perform specific functions within the model. Through carefully constructed contrastive datasets, we identify points where model generation diverges from memorized content and isolate the specific circuits responsible for two distinct aspects of memorization. We find that circuits that initiate memorization can also maintain it once started, while circuits that only maintain memorization cannot trigger its initiation. Intriguingly, memorization prevention mechanisms transfer robustly across different text domains, while memorization induction appears more context-dependent. 

---
# Is DeepSeek a New Voice Among LLMs in Public Opinion Simulation? 

**Authors**: Weihong Qi, Fan Huang, Jisun An, Haewoon Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2506.21587)  

**Abstract**: This study evaluates the ability of DeepSeek, an open-source large language model (LLM), to simulate public opinions in comparison to LLMs developed by major tech companies. By comparing DeepSeek-R1 and DeepSeek-V3 with Qwen2.5, GPT-4o, and Llama-3.3 and utilizing survey data from the American National Election Studies (ANES) and the Zuobiao dataset of China, we assess these models' capacity to predict public opinions on social issues in both China and the United States, highlighting their comparative capabilities between countries. Our findings indicate that DeepSeek-V3 performs best in simulating U.S. opinions on the abortion issue compared to other topics such as climate change, gun control, immigration, and services for same-sex couples, primarily because it more accurately simulates responses when provided with Democratic or liberal personas. For Chinese samples, DeepSeek-V3 performs best in simulating opinions on foreign aid and individualism but shows limitations in modeling views on capitalism, particularly failing to capture the stances of low-income and non-college-educated individuals. It does not exhibit significant differences from other models in simulating opinions on traditionalism and the free market. Further analysis reveals that all LLMs exhibit the tendency to overgeneralize a single perspective within demographic groups, often defaulting to consistent responses within groups. These findings highlight the need to mitigate cultural and demographic biases in LLM-driven public opinion modeling, calling for approaches such as more inclusive training methodologies. 

---
# Can Vision Language Models Understand Mimed Actions? 

**Authors**: Hyundong Cho, Spencer Lin, Tejas Srinivasan, Michael Saxon, Deuksin Kwon, Natali T. Chavez, Jonathan May  

**Link**: [PDF](https://arxiv.org/pdf/2506.21586)  

**Abstract**: Nonverbal communication (NVC) plays an integral role in human language, but studying NVC in general is challenging because of its broad scope and high variance in interpretation among individuals and cultures. However, mime -- the theatrical technique of suggesting intent using only gesture, expression, and movement -- is a subset of NVC that consists of explicit and embodied actions with much lower human interpretation variance. We argue that a solid understanding of mimed actions is a crucial prerequisite for vision-language models capable of interpreting and commanding more subtle aspects of NVC. Hence, we propose Mime Identification Multimodal Evaluation (MIME), a novel video-based question answering benchmark comprising of 86 mimed actions. Constructed with motion capture data, MIME consists of variations of each action with perturbations applied to the character, background, and viewpoint for evaluating recognition robustness. We find that both open-weight and API-based vision-language models perform significantly worse than humans on MIME, motivating the need for increased research for instilling more robust understanding of human gestures. 

---
# Evaluation of LLM-based Strategies for the Extraction of Food Product Information from Online Shops 

**Authors**: Christoph Brosch, Sian Brumm, Rolf Krieger, Jonas Scheffler  

**Link**: [PDF](https://arxiv.org/pdf/2506.21585)  

**Abstract**: Generative AI and large language models (LLMs) offer significant potential for automating the extraction of structured information from web pages. In this work, we focus on food product pages from online retailers and explore schema-constrained extraction approaches to retrieve key product attributes, such as ingredient lists and nutrition tables. We compare two LLM-based approaches, direct extraction and indirect extraction via generated functions, evaluating them in terms of accuracy, efficiency, and cost on a curated dataset of 3,000 food product pages from three different online shops. Our results show that although the indirect approach achieves slightly lower accuracy (96.48\%, $-1.61\%$ compared to direct extraction), it reduces the number of required LLM calls by 95.82\%, leading to substantial efficiency gains and lower operational costs. These findings suggest that indirect extraction approaches can provide scalable and cost-effective solutions for large-scale information extraction tasks from template-based web pages using LLMs. 

---
# Empirical Evidence for Alignment Faking in Small LLMs and Prompt-Based Mitigation Techniques 

**Authors**: J. Koorndijk  

**Link**: [PDF](https://arxiv.org/pdf/2506.21584)  

**Abstract**: Current literature suggests that alignment faking (deceptive alignment) is an emergent property of large language models. We present the first empirical evidence that a small instruction-tuned model, specifically LLaMA 3 8B, can also exhibit alignment faking. We further show that prompt-only interventions, including deontological moral framing and scratchpad reasoning, significantly reduce this behavior without modifying model internals. This challenges the assumption that prompt-based ethics are trivial and that deceptive alignment requires scale. We introduce a taxonomy distinguishing shallow deception, shaped by context and suppressible through prompting, from deep deception, which reflects persistent, goal-driven misalignment. Our findings refine the understanding of deception in language models and underscore the need for alignment evaluations across model sizes and deployment settings. 

---
# Hope Speech Detection in code-mixed Roman Urdu tweets: A Positive Turn in Natural Language Processing 

**Authors**: Muhammad Ahmad, Muhammad Waqas, Ameer Hamza, Ildar Batyrshin, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2506.21583)  

**Abstract**: Hope is a positive emotional state involving the expectation of favorable future outcomes, while hope speech refers to communication that promotes optimism, resilience, and support, particularly in adverse contexts. Although hope speech detection has gained attention in Natural Language Processing (NLP), existing research mainly focuses on high-resource languages and standardized scripts, often overlooking informal and underrepresented forms such as Roman Urdu. To the best of our knowledge, this is the first study to address hope speech detection in code-mixed Roman Urdu by introducing a carefully annotated dataset, thereby filling a critical gap in inclusive NLP research for low-resource, informal language varieties. This study makes four key contributions: (1) it introduces the first multi-class annotated dataset for Roman Urdu hope speech, comprising Generalized Hope, Realistic Hope, Unrealistic Hope, and Not Hope categories; (2) it explores the psychological foundations of hope and analyzes its linguistic patterns in code-mixed Roman Urdu to inform dataset development; (3) it proposes a custom attention-based transformer model optimized for the syntactic and semantic variability of Roman Urdu, evaluated using 5-fold cross-validation; and (4) it verifies the statistical significance of performance gains using a t-test. The proposed model, XLM-R, achieves the best performance with a cross-validation score of 0.78, outperforming the baseline SVM (0.75) and BiLSTM (0.76), with gains of 4% and 2.63% respectively. 

---
# VIDEE: Visual and Interactive Decomposition, Execution, and Evaluation of Text Analytics with Intelligent Agents 

**Authors**: Sam Yu-Te Lee, Chengyang Ji, Shicheng Wen, Lifu Huang, Dongyi Liu, Kwan-Liu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.21582)  

**Abstract**: Text analytics has traditionally required specialized knowledge in Natural Language Processing (NLP) or text analysis, which presents a barrier for entry-level analysts. Recent advances in large language models (LLMs) have changed the landscape of NLP by enabling more accessible and automated text analysis (e.g., topic detection, summarization, information extraction, etc.). We introduce VIDEE, a system that supports entry-level data analysts to conduct advanced text analytics with intelligent agents. VIDEE instantiates a human-agent collaroration workflow consisting of three stages: (1) Decomposition, which incorporates a human-in-the-loop Monte-Carlo Tree Search algorithm to support generative reasoning with human feedback, (2) Execution, which generates an executable text analytics pipeline, and (3) Evaluation, which integrates LLM-based evaluation and visualizations to support user validation of execution results. We conduct two quantitative experiments to evaluate VIDEE's effectiveness and analyze common agent errors. A user study involving participants with varying levels of NLP and text analytics experience -- from none to expert -- demonstrates the system's usability and reveals distinct user behavior patterns. The findings identify design implications for human-agent collaboration, validate the practical utility of VIDEE for non-expert users, and inform future improvements to intelligent text analytics systems. 

---
# From General Reasoning to Domain Expertise: Uncovering the Limits of Generalization in Large Language Models 

**Authors**: Dana Alsagheer, Yang Lu, Abdulrahman Kamal, Omar Kamal, Mohammad Kamal, Nada Mansour, Cosmo Yang Wu, Rambiba Karanjai, Sen Li, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21580)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated remarkable capabilities in various domains. However, effective decision-making relies heavily on strong reasoning abilities. Reasoning is the foundation for decision-making, providing the analytical and logical framework to make sound choices. Reasoning involves analyzing information, drawing inferences, and reaching conclusions based on logic or evidence. Decision-making builds on this foundation by applying the insights from reasoning to select the best course of action among alternatives. Together, these processes create a continuous cycle of thought and action aimed at achieving goals effectively. As AI technology evolves, there is a growing trend to train LLMs to excel in general reasoning. This study explores how the general reasoning capabilities of LLMs connect to their performance in domain-specific reasoning tasks. 

---
# HealthQA-BR: A System-Wide Benchmark Reveals Critical Knowledge Gaps in Large Language Models 

**Authors**: Andrew Maranhão Ventura D'addario  

**Link**: [PDF](https://arxiv.org/pdf/2506.21578)  

**Abstract**: The evaluation of Large Language Models (LLMs) in healthcare has been dominated by physician-centric, English-language benchmarks, creating a dangerous illusion of competence that ignores the interprofessional nature of patient care. To provide a more holistic and realistic assessment, we introduce HealthQA-BR, the first large-scale, system-wide benchmark for Portuguese-speaking healthcare. Comprising 5,632 questions from Brazil's national licensing and residency exams, it uniquely assesses knowledge not only in medicine and its specialties but also in nursing, dentistry, psychology, social work, and other allied health professions. We conducted a rigorous zero-shot evaluation of over 20 leading LLMs. Our results reveal that while state-of-the-art models like GPT 4.1 achieve high overall accuracy (86.6%), this top-line score masks alarming, previously unmeasured deficiencies. A granular analysis shows performance plummets from near-perfect in specialties like Ophthalmology (98.7%) to barely passing in Neurosurgery (60.0%) and, most notably, Social Work (68.4%). This "spiky" knowledge profile is a systemic issue observed across all models, demonstrating that high-level scores are insufficient for safety validation. By publicly releasing HealthQA-BR and our evaluation suite, we provide a crucial tool to move beyond single-score evaluations and toward a more honest, granular audit of AI readiness for the entire healthcare team. 

---
# Language-Aware Prompt Tuning for Parameter-Efficient Seamless Language Expansion in Multilingual ASR 

**Authors**: Hongli Yang, Sheng Li, Hao Huang, Ayiduosi Tuohan, Yizhou Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.21577)  

**Abstract**: Recent advancements in multilingual automatic speech recognition (ASR) have been driven by large-scale end-to-end models like Whisper. However, challenges such as language interference and expanding to unseen languages (language expansion) without degrading performance persist. This paper addresses these with three contributions: 1) Entire Soft Prompt Tuning (Entire SPT), which applies soft prompts to both the encoder and decoder, enhancing feature extraction and decoding; 2) Language-Aware Prompt Tuning (LAPT), which leverages cross-lingual similarities to encode shared and language-specific features using lightweight prompt matrices; 3) SPT-Whisper, a toolkit that integrates SPT into Whisper and enables efficient continual learning. Experiments across three languages from FLEURS demonstrate that Entire SPT and LAPT outperform Decoder SPT by 5.0% and 16.0% in language expansion tasks, respectively, providing an efficient solution for dynamic, multilingual ASR models with minimal computational overhead. 

---
# Adapting Whisper for Parameter-efficient Code-Switching Speech Recognition via Soft Prompt Tuning 

**Authors**: Hongli Yang, Yizhou Peng, Hao Huang, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21576)  

**Abstract**: Large-scale multilingual ASR models like Whisper excel in high-resource settings but face challenges in low-resource scenarios, such as rare languages and code-switching (CS), due to computational costs and catastrophic forgetting. We explore Soft Prompt Tuning (SPT), a parameter-efficient method to enhance CS ASR while preserving prior knowledge. We evaluate two strategies: (1) full fine-tuning (FFT) of both soft prompts and the entire Whisper model, demonstrating improved cross-lingual capabilities compared to traditional methods, and (2) adhering to SPT's original design by freezing model parameters and only training soft prompts. Additionally, we introduce SPT4ASR, a combination of different SPT variants. Experiments on the SEAME and ASRU2019 datasets show that deep prompt tuning is the most effective SPT approach, and our SPT4ASR methods achieve further error reductions in CS ASR, maintaining parameter efficiency similar to LoRA, without degrading performance on existing languages. 

---
# STRuCT-LLM: Unifying Tabular and Graph Reasoning with Reinforcement Learning for Semantic Parsing 

**Authors**: Josefa Lia Stoisser, Marc Boubnovski Martell, Lawrence Phillips, Casper Hansen, Julien Fauqueur  

**Link**: [PDF](https://arxiv.org/pdf/2506.21575)  

**Abstract**: We propose STRuCT-LLM, a unified framework for training large language models (LLMs) to perform structured reasoning over both relational and graph-structured data. Our approach jointly optimizes Text-to-SQL and Text-to-Cypher tasks using reinforcement learning (RL) combined with Chain-of-Thought (CoT) supervision. To support fine-grained optimization in graph-based parsing, we introduce a topology-aware reward function based on graph edit distance. Unlike prior work that treats relational and graph formalisms in isolation, STRuCT-LLM leverages shared abstractions between SQL and Cypher to induce cross-formalism transfer, enabling SQL training to improve Cypher performance and vice versa - even without shared schemas. Our largest model (QwQ-32B) achieves substantial relative improvements across tasks: on semantic parsing, Spider improves by 13.5\% and Text2Cypher by 73.1\%. The model also demonstrates strong zero-shot generalization, improving performance on downstream tabular QA (TableBench: 8.5\%) and knowledge graph QA (CR-LT-KGQA: 1.7\%) without any QA-specific supervision. These results demonstrate both the effectiveness of executable queries as scaffolds for structured reasoning and the synergistic benefits of jointly training on SQL and Cypher (code available at this https URL). 

---
# Digital Gatekeepers: Exploring Large Language Model's Role in Immigration Decisions 

**Authors**: Yicheng Mao, Yang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21574)  

**Abstract**: With globalization and increasing immigrant populations, immigration departments face significant work-loads and the challenge of ensuring fairness in decision-making processes. Integrating artificial intelligence offers a promising solution to these challenges. This study investigates the potential of large language models (LLMs),such as GPT-3.5 and GPT-4, in supporting immigration decision-making. Utilizing a mixed-methods approach,this paper conducted discrete choice experiments and in-depth interviews to study LLM decision-making strategies and whether they are fair. Our findings demonstrate that LLMs can align their decision-making with human strategies, emphasizing utility maximization and procedural fairness. Meanwhile, this paper also reveals that while ChatGPT has safeguards to prevent unintentional discrimination, it still exhibits stereotypes and biases concerning nationality and shows preferences toward privileged group. This dual analysis highlights both the potential and limitations of LLMs in automating and enhancing immigration decisions. 

---
# Instruction Learning Paradigms: A Dual Perspective on White-box and Black-box LLMs 

**Authors**: Yanwei Ren, Liu Liu, Baosheng Yu, Jiayan Qiu, Quan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21573)  

**Abstract**: Optimizing instructions for large language models (LLMs) is critical for harnessing their full potential in complex and diverse tasks. However, relying solely on white-box approaches demands extensive computational resources and offers limited representational capacity, while black-box models can incur prohibitive financial costs. To address these challenges, we introduce a novel framework that seamlessly merges the strengths of both paradigms. Black-box models provide high-quality, diverse instruction initializations, and white-box models supply fine-grained interpretability through hidden states and output features. By enforcing a semantic similarity constraint, these components fuse into a unified high-dimensional representation that captures deep semantic and structural nuances, enabling an iterative optimization process to refine instruction quality and adaptability. Extensive evaluations across a broad spectrum of tasks-ranging from complex reasoning to cross-lingual generalization-demonstrate that our approach consistently outperforms state-of-the-art baselines. This fusion of black-box initialization with advanced semantic refinement yields a scalable and efficient solution, paving the way for next-generation LLM-driven applications in diverse real-world scenarios. The source code will be released soon. 

---
# Aligning MLLM Benchmark With Human Preferences via Structural Equation Modeling 

**Authors**: Tianyu.Zou, Shengwu.Xiong, Ruilin.Yao, Jirui.Huang, Yi.Rong, Yaxiong.Chen, Shili.Xiong, Cong.Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21572)  

**Abstract**: Evaluating multimodal large language models (MLLMs) remains a fundamental challenge due to a lack of structured, interpretable, and theoretically grounded benchmark designs. Existing benchmarks often adopt heuristic-based task groupings with unclear cognitive targets, thus resulting in overlapping abilities, redundant indicators, and limited diagnostic power. In this work, we propose a novel framework for aligning MLLM benchmark based on Structural Equation Modeling (SEM) to analyze and quantify the internal validity, dimensional separability, and contribution of benchmark components. Motivated by the observed limitations of current designs, we further introduce a novel capability hierarchy grounded in Piagets theory of cognitive development, dividing MLLM abilities into three hierarchical layers, i.e., Perception, Memory, and Reasoning. We reorganize existing MLLM benchmarks under the proposed framework and construct a new benchmark named Gold. Experimental results demonstrate that the proposed benchmark exhibits stronger interpretability, reduced indicator redundancy, and clearer cognitive consistency compared to existing approaches. 

---
# Towards Understanding the Cognitive Habits of Large Reasoning Models 

**Authors**: Jianshuo Dong, Yujia Fu, Chuanrui Hu, Chao Zhang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21571)  

**Abstract**: Large Reasoning Models (LRMs), which autonomously produce a reasoning Chain of Thought (CoT) before producing final responses, offer a promising approach to interpreting and monitoring model behaviors. Inspired by the observation that certain CoT patterns -- e.g., ``Wait, did I miss anything?'' -- consistently emerge across tasks, we explore whether LRMs exhibit human-like cognitive habits. Building on Habits of Mind, a well-established framework of cognitive habits associated with successful human problem-solving, we introduce CogTest, a principled benchmark designed to evaluate LRMs' cognitive habits. CogTest includes 16 cognitive habits, each instantiated with 25 diverse tasks, and employs an evidence-first extraction method to ensure reliable habit identification. With CogTest, we conduct a comprehensive evaluation of 16 widely used LLMs (13 LRMs and 3 non-reasoning ones). Our findings reveal that LRMs, unlike conventional LLMs, not only exhibit human-like habits but also adaptively deploy them according to different tasks. Finer-grained analyses further uncover patterns of similarity and difference in LRMs' cognitive habit profiles, particularly certain inter-family similarity (e.g., Qwen-3 models and DeepSeek-R1). Extending the study to safety-related tasks, we observe that certain habits, such as Taking Responsible Risks, are strongly associated with the generation of harmful responses. These findings suggest that studying persistent behavioral patterns in LRMs' CoTs is a valuable step toward deeper understanding of LLM misbehavior. The code is available at: this https URL. 

---
# Random Initialization Can't Catch Up: The Advantage of Language Model Transfer for Time Series Forecasting 

**Authors**: Roland Riachi, Kashif Rasul, Arjun Ashok, Prateek Humane, Alexis Roger, Andrew R. Williams, Yuriy Nevmyvaka, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2506.21570)  

**Abstract**: Recent works have demonstrated the effectiveness of adapting pre-trained language models (LMs) for forecasting time series in the low-data regime. We build upon these findings by analyzing the effective transfer from language models to time series forecasting under various design choices including upstream post-training, time series tokenizer and language backbone size. In the low-data regime, these design choices have a significant impact on the validation loss, with clear-cut choices that outperform others. Contrary to Hernandez et al. (2021), we observe that the validation loss of the LMs continues to smoothly decrease long after the validation loss of the randomly initialized models has converged, leading to a non-vanishing transfer gap that holds across design choices. These findings not only help shed light on the effective use of compute-efficient training for time series, but also open the way for the study of modality-agnostic properties of data distributions leveraged by these models. 

---
# Hybrid-NL2SVA: Integrating RAG and Finetuning for LLM-based NL2SVA 

**Authors**: Weihua Xiao, Derek Ekberg, Siddharth Garg, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2506.21569)  

**Abstract**: SystemVerilog Assertions (SVAs) are critical for verifying the correctness of hardware designs, but manually writing them from natural language property descriptions, i.e., NL2SVA, remains a labor-intensive and error-prone task. Recent advances in large language models (LLMs) offer opportunities to automate this translation. However, existing models still struggle with understanding domain-specific syntax and semantics. To enhance LLM performance in NL2SVA, we propose a customized retrieval-augmented generation (RAG) framework and a synthetic fine-tuning dataset that together improve LLM's performance. To further improve lightweight models over NL2SVA, our fine-tuning dataset provides prompt-guided explanations that teach LLMs the layer-by-layer construction process of concurrent SVAs, enabling supervised fine-tuning that greatly improves syntax and functionality accuracy. To evaluate the performance of LLMs over NL2SVA, we construct the largest evaluation dataset for NL2SVA, comprising 40 Verilog designs and 229 formally verified SVAs with detailed annotations. Experimental results show that our customized RAG framework increases the number of functionality matched SVAs by 58.42% over GPT-4o-mini, while Qwen2.5-Coder-7B-Instruct fine-tuned on our fine-tuning dataset and integrated with HybridRetrieval achieves a 59.05% over the base Qwen model. 

---
# Assessing RAG and HyDE on 1B vs. 4B-Parameter Gemma LLMs for Personal Assistants Integretion 

**Authors**: Andrejs Sorstkins  

**Link**: [PDF](https://arxiv.org/pdf/2506.21568)  

**Abstract**: Resource efficiency is a critical barrier to deploying large language models (LLMs) in edge and privacy-sensitive applications. This study evaluates the efficacy of two augmentation strategies--Retrieval-Augmented Generation (RAG) and Hypothetical Document Embeddings (HyDE)--on compact Gemma LLMs of 1 billion and 4 billion parameters, within the context of a privacy-first personal assistant. We implement short-term memory via MongoDB and long-term semantic storage via Qdrant, orchestrated through FastAPI and LangChain, and expose the system through a this http URL frontend. Across both model scales, RAG consistently reduces latency by up to 17\% and eliminates factual hallucinations when responding to user-specific and domain-specific queries. HyDE, by contrast, enhances semantic relevance--particularly for complex physics prompts--but incurs a 25--40\% increase in response time and a non-negligible hallucination rate in personal-data retrieval. Comparing 1 B to 4 B models, we observe that scaling yields marginal throughput gains for baseline and RAG pipelines, but magnifies HyDE's computational overhead and variability. Our findings position RAG as the pragmatic choice for on-device personal assistants powered by small-scale LLMs. 

---
# BioPars: A Pretrained Biomedical Large Language Model for Persian Biomedical Text Mining 

**Authors**: Baqer M. Merzah, Tania Taami, Salman Asoudeh, Amir reza Hossein pour, Saeed Mirzaee, Amir Ali Bengari  

**Link**: [PDF](https://arxiv.org/pdf/2506.21567)  

**Abstract**: Large Language Models (LLMs) have recently gained attention in the life sciences due to their capacity to model, extract, and apply complex biological information. Beyond their classical use as chatbots, these systems are increasingly used for complex analysis and problem-solving in specialized fields, including bioinformatics. First, we introduce BIOPARS-BENCH, a dataset from over 10,000 scientific articles, textbooks, and medical websites. BioParsQA was also introduced to evaluate the proposed model, which consists of 5,231 Persian medical questions and answers. This study then introduces BioPars, a simple but accurate measure designed to assess LLMs for three main abilities: acquiring subject-specific knowledge, interpreting and synthesizing such knowledge, and demonstrating proper evidence. Comparing ChatGPT, Llama, and Galactica, our study highlights their ability to remember and retrieve learned knowledge but also reveals shortcomings in addressing higher-level, real-world questions and fine-grained inferences. These findings indicate the need for further fine-tuning to address the capabilities of LLM in bioinformatics tasks. To our knowledge, BioPars is the first application of LLM in Persian medical QA, especially for generating long answers. Evaluation of four selected medical QA datasets shows that BioPars has achieved remarkable results compared to comparative approaches. The model on BioParsQA achieved a ROUGE-L score of 29.99, which is an improvement over GPT-4 1.0. The model achieved a BERTScore of 90.87 with the MMR method. The MoverScore and BLEURT values were also higher in this model than the other three models. In addition, the reported scores for the model are MoverScore=60.43 and BLEURT=50.78. BioPars is an ongoing project and all resources related to its development will be made available via the following GitHub repository: this https URL. 

---
# The Saturation Point of Backtranslation in High Quality Low Resource English Gujarati Machine Translation 

**Authors**: Arwa Arif  

**Link**: [PDF](https://arxiv.org/pdf/2506.21566)  

**Abstract**: Backtranslation BT is widely used in low resource machine translation MT to generate additional synthetic training data using monolingual corpora. While this approach has shown strong improvements for many language pairs, its effectiveness in high quality, low resource settings remains unclear. In this work, we explore the effectiveness of backtranslation for English Gujarati translation using the multilingual pretrained MBART50 model. Our baseline system, trained on a high quality parallel corpus of approximately 50,000 sentence pairs, achieves a BLEU score of 43.8 on a validation set. We augment this data with carefully filtered backtranslated examples generated from monolingual Gujarati text. Surprisingly, adding this synthetic data does not improve translation performance and, in some cases, slightly reduces it. We evaluate our models using multiple metrics like BLEU, ChrF++, TER, BLEURT and analyze possible reasons for this saturation. Our findings suggest that backtranslation may reach a point of diminishing returns in certain low-resource settings and we discuss implications for future research. 

---
# A Multi-Agent Probabilistic Inference Framework Inspired by Kairanban-Style CoT System with IdoBata Conversation for Debiasing 

**Authors**: Takato Ueno, Keito Inoshita  

**Link**: [PDF](https://arxiv.org/pdf/2506.21565)  

**Abstract**: Japan's kairanban culture and idobata conversations have long functioned as traditional communication practices that foster nuanced dialogue among community members and contribute to the formation of social balance. Inspired by these information exchange processes, this study proposes a multi-agent inference framework (KCS+IBC) that integrates multiple large language models (LLMs) to achieve bias mitigation, improved explainability, and probabilistic prediction in sentiment analysis. In addition to sequentially sharing prediction results, the proposed method incorporates a mid-phase casual dialogue session to blend formal inference with individual perspectives and introduces probabilistic sentiment prediction. Experimental results show that KCS achieves accuracy comparable to that of a single LLM across datasets, while KCS+IBC exhibits a consistent decrease in entropy and a gradual increase in variance during the latter stages of inference, suggesting the framework's ability to balance aggregation and diversity of predictions. Future work will quantitatively assess the impact of these characteristics on bias correction and aim to develop more advanced sentiment analysis systems. 

---
# Team QUST at SemEval-2025 Task 10: Evaluating Large Language Models in Multiclass Multi-label Classification of News Entity Framing 

**Authors**: Jiyan Liu, Youzheng Liu, Taihang Wang, Xiaoman Xu, Yimin Wang, Ye Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21564)  

**Abstract**: This paper describes the participation of QUST_NLP in the SemEval-2025 Task 7. We propose a three-stage retrieval framework specifically designed for fact-checked claim retrieval. Initially, we evaluate the performance of several retrieval models and select the one that yields the best results for candidate retrieval. Next, we employ multiple re-ranking models to enhance the candidate results, with each model selecting the Top-10 outcomes. In the final stage, we utilize weighted voting to determine the final retrieval outcomes. Our approach achieved 5th place in the monolingual track and 7th place in the crosslingual track. We release our system code at: this https URL. 

---
# FormosanBench: Benchmarking Low-Resource Austronesian Languages in the Era of Large Language Models 

**Authors**: Kaiying Kevin Lin, Hsiyu Chen, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21563)  

**Abstract**: While large language models (LLMs) have demonstrated impressive performance across a wide range of natural language processing (NLP) tasks in high-resource languages, their capabilities in low-resource and minority languages remain significantly underexplored. Formosan languages -- a subgroup of Austronesian languages spoken in Taiwan -- are both linguistically rich and endangered, largely due to the sociolinguistic dominance of Mandarin. In this work, we introduce FORMOSANBENCH, the first benchmark for evaluating LLMs on low-resource Austronesian languages. It covers three endangered Formosan languages: Atayal, Amis, and Paiwan, across three core NLP tasks: machine translation, automatic speech recognition (ASR), and text summarization. We assess model performance in zero-shot, 10-shot, and fine-tuned settings using FORMOSANBENCH. Our results reveal a substantial performance gap between high-resource and Formosan languages. Existing LLMs consistently underperform across all tasks, with 10-shot learning and fine-tuning offering only limited improvements. These findings underscore the urgent need for more inclusive NLP technologies that can effectively support endangered and underrepresented languages. We release our datasets and code to facilitate future research in this direction. 

---
# FloorPlan-DeepSeek (FPDS): A multimodal approach to floorplan generation using vector-based next room prediction 

**Authors**: Jun Yin, Pengyu Zeng, Jing Zhong, Peilin Li, Miao Zhang, Ran Luo, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21562)  

**Abstract**: In the architectural design process, floor plan generation is inherently progressive and iterative. However, existing generative models for floor plans are predominantly end-to-end generation that produce an entire pixel-based layout in a single pass. This paradigm is often incompatible with the incremental workflows observed in real-world architectural practice. To address this issue, we draw inspiration from the autoregressive 'next token prediction' mechanism commonly used in large language models, and propose a novel 'next room prediction' paradigm tailored to architectural floor plan modeling. Experimental evaluation indicates that FPDS demonstrates competitive performance in comparison to diffusion models and Tell2Design in the text-to-floorplan task, indicating its potential applicability in supporting future intelligent architectural design. 

---
# Reasoning Isn't Enough: Examining Truth-Bias and Sycophancy in LLMs 

**Authors**: Emilio Barkett, Olivia Long, Madhavendra Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2506.21561)  

**Abstract**: Despite their widespread use in fact-checking, moderation, and high-stakes decision-making, large language models (LLMs) remain poorly understood as judges of truth. This study presents the largest evaluation to date of LLMs' veracity detection capabilities and the first analysis of these capabilities in reasoning models. We had eight LLMs make 4,800 veracity judgments across several prompts, comparing reasoning and non-reasoning models. We find that rates of truth-bias, or the likelihood to believe a statement is true, regardless of whether it is actually true, are lower in reasoning models than in non-reasoning models, but still higher than human benchmarks. Most concerning, we identify sycophantic tendencies in several advanced models (o4-mini and GPT-4.1 from OpenAI, R1 from DeepSeek), which displayed an asymmetry in detection accuracy, performing well in truth accuracy but poorly in deception accuracy. This suggests that capability advances alone do not resolve fundamental veracity detection challenges in LLMs. 

---
# Reinforcement Learning Fine-Tuning of Language Model for Instruction Following and Math Reasoning 

**Authors**: Yifu Han, Geo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21560)  

**Abstract**: This study investigates the effectiveness of reinforcement learning (RL) fine-tuning techniques on a compact language model (Qwen2.5-0.5B Base) for two challenging tasks: instruction following and mathematical reasoning. We compare supervised fine-tuning (SFT), Direct Preference Optimization (DPO) using preference-labeled data, and Reinforce Leave-One-Out (RLOO) with reward models. Our experiments show that RLOO with DeBERTa reward modeling achieves the best alignment, while DPO provides strong and consistent results. For math reasoing tasks, synthetic data augmentation and best-of-N sampling with an external verifier significantly improve accuracy, showing the potential of combining fine-tuning with inference-time tools. This study highlights key trade-offs and practical strategies for training lightweight, task-aligned small-scale language models. 

---
# GraphLAMA: Enabling Efficient Adaptation of Graph Language Models with Limited Annotations 

**Authors**: Junze Chen, Cheng Yang, Shujie Li, Zhiqiang Zhang, Yawen Li, Junping Du, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21559)  

**Abstract**: Large language models (LLMs) have demonstrated their strong capabilities in various domains, and have been recently integrated for graph analysis as graph language models (GLMs). With LLMs as the predictor, some GLMs can interpret unseen tasks described by natural language, and learn from a few examples in the prompts without parameter tuning, known as in-context learning (ICL). Another subset of GLMs utilizes abundant training labels to enhance model performance, known as instruction tuning. However, we argue that ICL on graphs has effectiveness issues due to fixed parameters and efficiency issues due to long context. Meanwhile, the large amount of labeled data required for instruction tuning can be difficult to obtain in real-world scenarios. To this end, we aim to introduce an extra parameter adaptation stage that can efficiently tailor GLMs to an unseen graph and task with only a few labeled examples, in exchange for better prediction accuracy and faster inference speed. For implementation, in this paper we propose GraphLAMA method, with its model backbone and learning schemes specialized for efficient tuning and inference. Specifically, for model backbone, we use a graph neural network (GNN) with several well-designed components to transform nodes into the representation space of LLM tokens. Task instructions can then be represented as a mixture of node and language tokens. In the pre-training stage, model parameters except the LLM will be trained with different tasks to capture general knowledge. In the adaptation stage, only a few pre-trained parameters will be updated based on few-shot examples. Extensive experiments on few/zero-shot node classification and summary generation show that our proposed GraphLAMA achieves state-of-the-art performance with 4.91% absolution improvement in accuracy. Compared with ICL, our inference speed can be 10 times faster under 5-shot setting. 

---
# Bench to the Future: A Pastcasting Benchmark for Forecasting Agents 

**Authors**: FutureSearch, Jack Wildman, Nikos I. Bosse, Daniel Hnyk, Peter Mühlbacher, Finn Hambly, Jon Evans, Dan Schwarz, Lawrence Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2506.21558)  

**Abstract**: Forecasting is a challenging task that offers a clearly measurable way to study AI systems. Forecasting requires a large amount of research on the internet, and evaluations require time for events to happen, making the development of forecasting benchmarks challenging. To date, no forecasting benchmark provides a realistic, hermetic, and repeatable environment for LLM forecasters. We introduce Bench To the Future (BTF), a "pastcasting" benchmark with hundreds of high-quality questions for which the resolution is already known. Each question is accompanied by a large offline corpus of tens of thousands of relevant web pages, enabling a way to elicit realistic "forecasts" on past events from LLMs. Results suggest that our pastcasting environment can produce results comparable to those based on forecasts using the internet on at-the-time unresolved questions. We show results benchmarking agent and chain-of-thought forecasting approaches using several LLMs, including the recently-released Claude 4 models, and demonstrate BTF's ability to track steady forecasting capability progress over time. We intend this to be a living benchmark, with new questions added continually to account for increasing training data cutoff dates. We invite researchers to contact us at hello@futuresearch.ai to utilize our benchmark or tooling for their own research. 

---
# Debunk and Infer: Multimodal Fake News Detection via Diffusion-Generated Evidence and LLM Reasoning 

**Authors**: Kaiying Yan, Moyang Liu, Yukun Liu, Ruibo Fu, Zhengqi Wen, Jianhua Tao, Xuefei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21557)  

**Abstract**: The rapid spread of fake news across multimedia platforms presents serious challenges to information credibility. In this paper, we propose a Debunk-and-Infer framework for Fake News Detection(DIFND) that leverages debunking knowledge to enhance both the performance and interpretability of fake news detection. DIFND integrates the generative strength of conditional diffusion models with the collaborative reasoning capabilities of multimodal large language models (MLLMs). Specifically, debunk diffusion is employed to generate refuting or authenticating evidence based on the multimodal content of news videos, enriching the evaluation process with diverse yet semantically aligned synthetic samples. To improve inference, we propose a chain-of-debunk strategy where a multi-agent MLLM system produces logic-grounded, multimodal-aware reasoning content and final veracity judgment. By jointly modeling multimodal features, generative debunking cues, and reasoning-rich verification within a unified architecture, DIFND achieves notable improvements in detection accuracy. Extensive experiments on the FakeSV and FVC datasets show that DIFND not only outperforms existing approaches but also delivers trustworthy decisions. 

---
# VAT-KG: Knowledge-Intensive Multimodal Knowledge Graph Dataset for Retrieval-Augmented Generation 

**Authors**: Hyeongcheol Park, MinHyuk Jang, Ha Dam Baek, Gyusam Chang, Jiyoung Seo, Jiwan Park, Hogun Park, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.21556)  

**Abstract**: Multimodal Knowledge Graphs (MMKGs), which represent explicit knowledge across multiple modalities, play a pivotal role by complementing the implicit knowledge of Multimodal Large Language Models (MLLMs) and enabling more grounded reasoning via Retrieval Augmented Generation (RAG). However, existing MMKGs are generally limited in scope: they are often constructed by augmenting pre-existing knowledge graphs, which restricts their knowledge, resulting in outdated or incomplete knowledge coverage, and they often support only a narrow range of modalities, such as text and visual information. These limitations reduce their extensibility and applicability to a broad range of multimodal tasks, particularly as the field shifts toward richer modalities such as video and audio in recent MLLMs. Therefore, we propose the Visual-Audio-Text Knowledge Graph (VAT-KG), the first concept-centric and knowledge-intensive multimodal knowledge graph that covers visual, audio, and text information, where each triplet is linked to multimodal data and enriched with detailed descriptions of concepts. Specifically, our construction pipeline ensures cross-modal knowledge alignment between multimodal data and fine-grained semantics through a series of stringent filtering and alignment steps, enabling the automatic generation of MMKGs from any multimodal dataset. We further introduce a novel multimodal RAG framework that retrieves detailed concept-level knowledge in response to queries from arbitrary modalities. Experiments on question answering tasks across various modalities demonstrate the effectiveness of VAT-KG in supporting MLLMs, highlighting its practical value in unifying and leveraging multimodal knowledge. 

---
# Efficient Multilingual ASR Finetuning via LoRA Language Experts 

**Authors**: Jiahong Li, Yiwen Shao, Jianheng Zhuo, Chenda Li, Liliang Tang, Dong Yu, Yanmin Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.21555)  

**Abstract**: Recent advancements in deep learning have significantly enhanced multilingual automatic speech recognition (ASR) due to the development of advanced model architectures and available large-scale multilingual datasets. Despite that, multilingual ASR still suffers from the curse of multilinguality in that different languages tend to interfere with each other, making it difficult for the ASR model to identify multiple languages effectively while sharing model capacity across them. This paper proposes an efficient finetuning framework for customized multilingual ASR via prepared LoRA language experts based on Whisper. Through LoRA expert fusion or knowledge distillation, our approach achieves better recognition performance on target languages than standard fine-tuning methods. Experimental results demonstrate that the proposed models yield approximately 10\% and 15\% relative performance gains in language-aware and language-agnostic scenarios, respectively. 

---
# The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements 

**Authors**: Bingchen Zhao, Despoina Magka, Minqi Jiang, Xian Li, Roberta Raileanu, Tatiana Shavrina, Jean-Christophe Gagnon-Audet, Kelvin Niu, Shagun Sodhani, Michael Shvartsman, Andrei Lupu, Alisia Lupidi, Edan Toledo, Karen Hambardzumyan, Martin Josifoski, Thomas Foster, Lucia Cipolina-Kun, Abhishek Charnalia, Derek Dunfield, Alexander H. Miller, Oisin Mac Aodha, Jakob Foerster, Yoram Bachrach  

**Link**: [PDF](https://arxiv.org/pdf/2506.22419)  

**Abstract**: Rapid advancements in large language models (LLMs) have the potential to assist in scientific progress. A critical capability toward this endeavor is the ability to reproduce existing work. To evaluate the ability of AI agents to reproduce results in an active research area, we introduce the Automated LLM Speedrunning Benchmark, leveraging the research community contributions on the NanoGPT speedrun, a competition to train a GPT-2 model in the shortest time. Each of the 19 speedrun tasks provides the agent with the previous records training script, optionally paired with one of three hint formats, ranging from pseudocode to paper-like descriptions of the new records improvements. Records execute quickly by design and speedrun improvements encompass diverse code-level changes, ranging from high-level algorithmic advancements to hardware-aware optimizations. These features make the benchmark both accessible and realistic for the frontier problem of improving LLM training. We find that recent reasoning LLMs combined with SoTA scaffolds struggle to reimplement already-known innovations in our benchmark, even when given detailed hints. Our benchmark thus provides a simple, non-saturated measure of an LLMs ability to automate scientific reproduction, a necessary (but not sufficient) skill for an autonomous research agent. 

---
# Can Video Large Multimodal Models Think Like Doubters-or Double-Down: A Study on Defeasible Video Entailment 

**Authors**: Yue Zhang, Jilei Sun, Yunhui Guo, Vibhav Gogate  

**Link**: [PDF](https://arxiv.org/pdf/2506.22385)  

**Abstract**: Video Large Multimodal Models (VLMMs) have made impressive strides in understanding video content, but they often struggle with abstract and adaptive reasoning-the ability to revise their interpretations when new information emerges. In reality, conclusions are rarely set in stone; additional context can strengthen or weaken an initial inference. To address this, we introduce Defeasible Video Entailment (DVidE), a new task that challenges models to think like doubters, constantly updating their reasoning based on evolving evidence. In DVidE, given a video premise and a textual hypothesis, models must determine whether a new update strengthens or weakens the hypothesis (classification version) or generate a coherent update that modifies the entailment relationship (generation version). For solving the classification task, we propose the Chain of Counterfactual Thought framework, utilizing counterfactual reasoning, ASR-enhanced video content, and rationale refinement to reduce inference bias. For the generation task, we develop a framework that combines ASR output with a Large Language Model (LLM) to produce coherent, contextually relevant updates aligned with the intended strengthener or weakener goals. Additionally, we introduce a novel benchmark dataset, with strengthener/weakener annotations and an LLM-based evaluation metric specifically designed for assessing generative performance. Experimental results demonstrate significant improvements, highlighting our proposed method in enhancing dynamic reasoning capabilities of VLMMs. 

---
# Probabilistic Optimality for Inference-time Scaling 

**Authors**: Youkang Wang, Jian Wang, Rubing Chen, Xiao-Yong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22376)  

**Abstract**: Inference-time scaling has emerged as a powerful technique for enhancing the reasoning performance of Large Language Models (LLMs). However, existing approaches often rely on heuristic strategies for parallel sampling, lacking a principled foundation. To address this gap, we propose a probabilistic framework that formalizes the optimality of inference-time scaling under the assumption that parallel samples are independently and identically distributed (i.i.d.), and where the Best-of-N selection strategy follows a probability distribution that can be estimated. Within this framework, we derive a theoretical lower bound on the required number of samples to achieve a target performance level, providing the first principled guidance for compute-efficient scaling. Leveraging this insight, we develop \textsc{OptScale}, a practical algorithm that dynamically determines the optimal number of sampled responses. \textsc{OptScale} employs a language model-based predictor to estimate probabilistic prior parameters, enabling the decision of the minimal number of samples needed that satisfy predefined performance thresholds and confidence levels. Extensive experiments on mathematical reasoning benchmarks (including MATH-500, GSM8K, AIME, and AMC) demonstrate that \textsc{OptScale} significantly reduces sampling overhead while remaining better or on par with state-of-the-art reasoning performance. Our work offers both a theoretical foundation and a practical solution for principled inference-time scaling, addressing a critical gap in the efficient deployment of LLMs for complex reasoning. 

---
# Towards Fair Rankings: Leveraging LLMs for Gender Bias Detection and Measurement 

**Authors**: Maryam Mousavian, Zahra Abbasiantaeb, Mohammad Aliannejadi, Fabio Crestani  

**Link**: [PDF](https://arxiv.org/pdf/2506.22372)  

**Abstract**: The presence of social biases in Natural Language Processing (NLP) and Information Retrieval (IR) systems is an ongoing challenge, which underlines the importance of developing robust approaches to identifying and evaluating such biases. In this paper, we aim to address this issue by leveraging Large Language Models (LLMs) to detect and measure gender bias in passage ranking. Existing gender fairness metrics rely on lexical- and frequency-based measures, leading to various limitations, e.g., missing subtle gender disparities. Building on our LLM-based gender bias detection method, we introduce a novel gender fairness metric, named Class-wise Weighted Exposure (CWEx), aiming to address existing limitations. To measure the effectiveness of our proposed metric and study LLMs' effectiveness in detecting gender bias, we annotate a subset of the MS MARCO Passage Ranking collection and release our new gender bias collection, called MSMGenderBias, to foster future research in this area. Our extensive experimental results on various ranking models show that our proposed metric offers a more detailed evaluation of fairness compared to previous metrics, with improved alignment to human labels (58.77% for Grep-BiasIR, and 18.51% for MSMGenderBias, measured using Cohen's Kappa agreement), effectively distinguishing gender bias in ranking. By integrating LLM-driven bias detection, an improved fairness metric, and gender bias annotations for an established dataset, this work provides a more robust framework for analyzing and mitigating bias in IR systems. 

---
# Optimal Estimation of Watermark Proportions in Hybrid AI-Human Texts 

**Authors**: Xiang Li, Garrett Wen, Weiqing He, Jiayuan Wu, Qi Long, Weijie J. Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.22343)  

**Abstract**: Text watermarks in large language models (LLMs) are an increasingly important tool for detecting synthetic text and distinguishing human-written content from LLM-generated text. While most existing studies focus on determining whether entire texts are watermarked, many real-world scenarios involve mixed-source texts, which blend human-written and watermarked content. In this paper, we address the problem of optimally estimating the watermark proportion in mixed-source texts. We cast this problem as estimating the proportion parameter in a mixture model based on \emph{pivotal statistics}. First, we show that this parameter is not even identifiable in certain watermarking schemes, let alone consistently estimable. In stark contrast, for watermarking methods that employ continuous pivotal statistics for detection, we demonstrate that the proportion parameter is identifiable under mild conditions. We propose efficient estimators for this class of methods, which include several popular unbiased watermarks as examples, and derive minimax lower bounds for any measurable estimator based on pivotal statistics, showing that our estimators achieve these lower bounds. Through evaluations on both synthetic data and mixed-source text generated by open-source models, we demonstrate that our proposed estimators consistently achieve high estimation accuracy. 

---
# Conceptual Topic Aggregation 

**Authors**: Klara M. Gutekunst, Dominik Dürrschnabel, Johannes Hirth, Gerd Stumme  

**Link**: [PDF](https://arxiv.org/pdf/2506.22309)  

**Abstract**: The vast growth of data has rendered traditional manual inspection infeasible, necessitating the adoption of computational methods for efficient data exploration. Topic modeling has emerged as a powerful tool for analyzing large-scale textual datasets, enabling the extraction of latent semantic structures. However, existing methods for topic modeling often struggle to provide interpretable representations that facilitate deeper insights into data structure and content. In this paper, we propose FAT-CAT, an approach based on Formal Concept Analysis (FCA) to enhance meaningful topic aggregation and visualization of discovered topics. Our approach can handle diverse topics and file types -- grouped by directories -- to construct a concept lattice that offers a structured, hierarchical representation of their topic distribution. In a case study on the ETYNTKE dataset, we evaluate the effectiveness of our approach against other representation methods to demonstrate that FCA-based aggregation provides more meaningful and interpretable insights into dataset composition than existing topic modeling techniques. 

---
# COOCO -- Common Objects Out-of-Context -- Semantic Violation in Scenes: Investigating Multimodal Context in Referential Communication 

**Authors**: Filippo Merlo, Ece Takmaz, Wenkai Chen, Albert Gatt  

**Link**: [PDF](https://arxiv.org/pdf/2506.22274)  

**Abstract**: Natural scenes provide us with rich contexts for object recognition and reference. In particular, knowing what type of scene one is looking at generates expectations about which objects will occur, and what their spatial configuration should be. Do Vision-Language Models (VLMs) learn to rely on scene contexts in a similar way, when generating references to objects? To address this question, we introduce the \textit{Common Objects Out-of-Context (COOCO)} dataset and test to what extent VLMs rely on scene context to refer to objects under different degrees of scene-object congruency, and different perturbations. Our findings show that models leverage scene context adaptively, depending on both the semantic relatedness between object and scene and the level of noise. In particular, models rely more on context under high target-scene congruence or when objects are degraded. Attention analysis reveals that successful object categorisation involves increased focus on the target in mid-level layers, especially under moderate noise, suggesting that VLMs dynamically balance local and contextual information for reference generation. We make our dataset, code and models available at \href{this https URL}{this https URL}. 

---
# Projected Compression: Trainable Projection for Efficient Transformer Compression 

**Authors**: Maciej Stefaniak, Michał Krutul, Jan Małaśnicki, Maciej Pióro, Jakub Krajewski, Sebastian Jaszczur, Marek Cygan, Kamil Adamczewski, Jan Ludziejewski  

**Link**: [PDF](https://arxiv.org/pdf/2506.22255)  

**Abstract**: Large language models have steadily increased in size to achieve improved performance; however, this growth has also led to greater inference time and computational demands. Consequently, there is rising interest in model size reduction methods. To address this issue, we propose Projected Compression, a novel model compression technique, that reduces model weights by utilizing projection modules. Specifically, we first train additional trainable projections weights and preserve access to all the original model parameters. Subsequently, these projections are merged into a lower-dimensional product matrix, resulting in a reduced-size standard Transformer-based model. Unlike alternative approaches that require additional computational overhead, our method matches the base model's per-token computation step in FLOPs. Experimental results show that Projected Compression outperforms the comparable hard pruning and retraining approach on higher quality models. Moreover, the performance margin scales well with the number of tokens. 

---
# Fine-Tuning MIDI-to-Audio Alignment using a Neural Network on Piano Roll and CQT Representations 

**Authors**: Sebastian Murgul, Moritz Reiser, Michael Heizmann, Christoph Seibert  

**Link**: [PDF](https://arxiv.org/pdf/2506.22237)  

**Abstract**: In this paper, we present a neural network approach for synchronizing audio recordings of human piano performances with their corresponding loosely aligned MIDI files. The task is addressed using a Convolutional Recurrent Neural Network (CRNN) architecture, which effectively captures spectral and temporal features by processing an unaligned piano roll and a spectrogram as inputs to estimate the aligned piano roll. To train the network, we create a dataset of piano pieces with augmented MIDI files that simulate common human timing errors. The proposed model achieves up to 20% higher alignment accuracy than the industry-standard Dynamic Time Warping (DTW) method across various tolerance windows. Furthermore, integrating DTW with the CRNN yields additional improvements, offering enhanced robustness and consistency. These findings demonstrate the potential of neural networks in advancing state-of-the-art MIDI-to-audio alignment. 

---
# Exploring Modularity of Agentic Systems for Drug Discovery 

**Authors**: Laura van Weesep, Samuel Genheden, Ola Engkvist, Jens Sjölund  

**Link**: [PDF](https://arxiv.org/pdf/2506.22189)  

**Abstract**: Large-language models (LLMs) and agentic systems present exciting opportunities to accelerate drug discovery and design. In this study, we critically examine the modularity of LLM-based agentic systems for drug discovery, i.e., whether parts of the agentic system such as the LLM are interchangeable, a topic that has received limited attention in drug discovery applications. We compare the performance of different large language models (LLMs) and the effectiveness of tool-calling agents versus code-generating agents in this domain. Our case study, comparing performance in orchestrating tools for chemistry and drug discovery using an LLM-as-a-judge score, shows that Claude-3.5-Sonnet, Claude-3.7-Sonnet and GPT-4o outperform alternative language models such as Llama-3.1-8B, Llama-3.1-70B, GPT-3.5-Turbo, and Nova-Micro. Although we confirm that code-generating agents outperform the tool-calling ones on average, we show that this is highly question and model dependent. Furthermore, the impact of replacing system prompts is dependent on the specific question asked and the model used, underscoring that -- even in this particular domain -- one cannot just replace language models without considering prompt re-engineering. Our study highlights the necessity of further research into the modularity of agentic systems to enable the development of stable and scalable solutions for real-world problems. 

---
# GPAS: Accelerating Convergence of LLM Pretraining via Gradient-Preserving Activation Scaling 

**Authors**: Tianhao Chen, Xin Xu, Zijing Liu, Pengxiang Li, Xinyuan Song, Ajay Kumar Jaiswal, Fan Zhang, Jishan Hu, Yang Wang, Hao Chen, Shizhe Diao, Shiwei Liu, Yu Li, Yin Lu, Can Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22049)  

**Abstract**: Modern Large Language Models, such as the LLaMA, Qwen and DeepSeek series, predominantly adopt the Pre-LayerNorm (Pre-LN) Transformer architecture. While being stable during pretraining and scalable to large model sizes, Pre-LN suffers from an exponential growth in activation variance across layers, causing the residual path to dominate over sub-layer outputs and limiting the learning capacity of deeper layers. To mitigate this issue, we propose Gradient-Preserving Activation Scaling (GPAS), a simple technique that can be used in combination with existing approaches. GPAS works by scaling down the intermediate activations while keeping their gradients unchanged. This leaves information in the activations intact, and avoids the gradient vanishing problem associated with gradient downscaling. Extensive experiments across various model sizes from 71M to 1B show that GPAS achieves consistent performance gains. Beyond enhancing Pre-LN Transformers, GPAS also shows promise in improving alternative architectures such as Sandwich-LN and DeepNorm, demonstrating its versatility and potential for improving training dynamics in a wide range of settings. 

---
# Robust and Efficient Autoregressive Speech Synthesis with Dynamic Chunk-wise Prediction Policy 

**Authors**: Bohan Li, Zhihan Li, Haoran Wang, Hanglei Zhang, Yiwei Guo, Hankun Wang, Xie Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22023)  

**Abstract**: Recently, autoregressive (AR) language models have emerged as a dominant approach in speech synthesis, offering expressive generation and scalable training. However, conventional AR speech synthesis models relying on the next-token prediction paradigm often encounter significant challenges when handling long speech sequences. These models often struggle to construct stable frame-to-frame attention, leading to increased latency and degraded synthesis quality, thereby limiting their feasibility for real-time applications. To address these limitations, we introduce a novel dynamic chunk-wise autoregressive synthesis framework, termed DCAR, designed to enhance both efficiency and intelligibility robustness in AR speech generation. DCAR introduces a chunk-to-frame attention mechanism through training with multi-token prediction, enabling dynamic chunk prediction in variable speech contexts using a lightweight module trained on-policy. DCAR dynamically adjusts the token prediction span, significantly reducing the sequence length dependency while obtaining high synthesis quality. Comprehensive empirical evaluations demonstrate that DCAR substantially outperforms traditional next-token prediction models, achieving up to 72.27% intelligibility improvement and 2.61x inference speedup simultaneously on the test set. Furthermore, we conduct comprehensive analysis to support it as a versatile foundation for next-generation speech synthesis systems. 

---
# Using Large Language Models to Suggest Informative Prior Distributions in Bayesian Statistics 

**Authors**: Michael A. Riegler, Kristoffer Herland Hellton, Vajira Thambawita, Hugo L. Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2506.21964)  

**Abstract**: Selecting prior distributions in Bayesian statistics is challenging, resource-intensive, and subjective. We analyze using large-language models (LLMs) to suggest suitable, knowledge-based informative priors. We developed an extensive prompt asking LLMs not only to suggest priors but also to verify and reflect on their choices.
We evaluated Claude Opus, Gemini 2.5 Pro, and ChatGPT-4o-mini on two real datasets: heart disease risk and concrete strength. All LLMs correctly identified the direction for all associations (e.g., that heart disease risk is higher for males). The quality of suggested priors was measured by their Kullback-Leibler divergence from the maximum likelihood estimator's distribution.
The LLMs suggested both moderately and weakly informative priors. The moderate priors were often overconfident, resulting in distributions misaligned with the data. In our experiments, Claude and Gemini provided better priors than ChatGPT. For weakly informative priors, a key performance difference emerged: ChatGPT and Gemini defaulted to an "unnecessarily vague" mean of 0, while Claude did not, demonstrating a significant advantage.
The ability of LLMs to identify correct associations shows their great potential as an efficient, objective method for developing informative priors. However, the primary challenge remains in calibrating the width of these priors to avoid over- and under-confidence. 

---
# ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation 

**Authors**: Reza Yousefi Maragheh, Pratheek Vadla, Priyank Gupta, Kai Zhao, Aysenur Inan, Kehui Yao, Jianpeng Xu, Praveen Kanumala, Jason Cho, Sushant Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.21931)  

**Abstract**: Retrieval-Augmented Generation (RAG) has shown promise in enhancing recommendation systems by incorporating external context into large language model prompts. However, existing RAG-based approaches often rely on static retrieval heuristics and fail to capture nuanced user preferences in dynamic recommendation scenarios. In this work, we introduce ARAG, an Agentic Retrieval-Augmented Generation framework for Personalized Recommendation, which integrates a multi-agent collaboration mechanism into the RAG pipeline. To better understand the long-term and session behavior of the user, ARAG leverages four specialized LLM-based agents: a User Understanding Agent that summarizes user preferences from long-term and session contexts, a Natural Language Inference (NLI) Agent that evaluates semantic alignment between candidate items retrieved by RAG and inferred intent, a context summary agent that summarizes the findings of NLI agent, and an Item Ranker Agent that generates a ranked list of recommendations based on contextual fit. We evaluate ARAG accross three datasets. Experimental results demonstrate that ARAG significantly outperforms standard RAG and recency-based baselines, achieving up to 42.1% improvement in NDCG@5 and 35.5% in Hit@5. We also, conduct an ablation study to analyse the effect by different components of ARAG. Our findings highlight the effectiveness of integrating agentic reasoning into retrieval-augmented recommendation and provide new directions for LLM-based personalization. 

---
# HyReC: Exploring Hybrid-based Retriever for Chinese 

**Authors**: Zunran Wang, Zheng Shenpeng, Wang Shenglan, Minghui Zhao, Zhonghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21913)  

**Abstract**: Hybrid-based retrieval methods, which unify dense-vector and lexicon-based retrieval, have garnered considerable attention in the industry due to performance enhancement. However, despite their promising results, the application of these hybrid paradigms in Chinese retrieval contexts has remained largely underexplored. In this paper, we introduce HyReC, an innovative end-to-end optimization method tailored specifically for hybrid-based retrieval in Chinese. HyReC enhances performance by integrating the semantic union of terms into the representation model. Additionally, it features the Global-Local-Aware Encoder (GLAE) to promote consistent semantic sharing between lexicon-based and dense retrieval while minimizing the interference between them. To further refine alignment, we incorporate a Normalization Module (NM) that fosters mutual benefits between the retrieval approaches. Finally, we evaluate HyReC on the C-MTEB retrieval benchmark to demonstrate its effectiveness. 

---
# RiverEcho: Real-Time Interactive Digital System for Ancient Yellow River Culture 

**Authors**: Haofeng Wang, Yilin Guo, Zehao Li, Tong Yue, Yizong Wang, Enci Zhang, Rongqun Lin, Feng Gao, Shiqi Wang, Siwei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.21865)  

**Abstract**: The Yellow River is China's mother river and a cradle of human civilization. The ancient Yellow River culture is, moreover, an indispensable part of human art history. To conserve and inherit the ancient Yellow River culture, we designed RiverEcho, a real-time interactive system that responds to voice queries using a large language model and a cultural knowledge dataset, delivering explanations through a talking-head digital human. Specifically, we built a knowledge database focused on the ancient Yellow River culture, including the collection of historical texts and the processing pipeline. Experimental results demonstrate that leveraging Retrieval-Augmented Generation (RAG) on the proposed dataset enhances the response quality of the Large Language Model(LLM), enabling the system to generate more professional and informative responses. Our work not only diversifies the means of promoting Yellow River culture but also provides users with deeper cultural insights. 

---
# 3Description: An Intuitive Human-AI Collaborative 3D Modeling Approach 

**Authors**: Zhuodi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.21845)  

**Abstract**: This paper presents 3Description, an experimental human-AI collaborative approach for intuitive 3D modeling. 3Description aims to address accessibility and usability challenges in traditional 3D modeling by enabling non-professional individuals to co-create 3D models using verbal and gesture descriptions. Through a combination of qualitative research, product analysis, and user testing, 3Description integrates AI technologies such as Natural Language Processing and Computer Vision, powered by OpenAI and MediaPipe. Recognizing the web has wide cross-platform capabilities, 3Description is web-based, allowing users to describe the desired model and subsequently adjust its components using verbal and gestural inputs. In the era of AI and emerging media, 3Description not only contributes to a more inclusive and user-friendly design process, empowering more people to participate in the construction of the future 3D world, but also strives to increase human engagement in co-creation with AI, thereby avoiding undue surrender to technology and preserving human creativity. 

---
# GenEscape: Hierarchical Multi-Agent Generation of Escape Room Puzzles 

**Authors**: Mengyi Shan, Brian Curless, Ira Kemelmacher-Shlizerman, Steve Seitz  

**Link**: [PDF](https://arxiv.org/pdf/2506.21839)  

**Abstract**: We challenge text-to-image models with generating escape room puzzle images that are visually appealing, logically solid, and intellectually stimulating. While base image models struggle with spatial relationships and affordance reasoning, we propose a hierarchical multi-agent framework that decomposes this task into structured stages: functional design, symbolic scene graph reasoning, layout synthesis, and local image editing. Specialized agents collaborate through iterative feedback to ensure the scene is visually coherent and functionally solvable. Experiments show that agent collaboration improves output quality in terms of solvability, shortcut avoidance, and affordance clarity, while maintaining visual quality. 

---
# Exploring the change in scientific readability following the release of ChatGPT 

**Authors**: Abdulkareem Alsudais  

**Link**: [PDF](https://arxiv.org/pdf/2506.21825)  

**Abstract**: The rise and growing popularity of accessible large language models have raised questions about their impact on various aspects of life, including how scientists write and publish their research. The primary objective of this paper is to analyze a dataset consisting of all abstracts posted on arXiv.org between 2010 and June 7th, 2024, to assess the evolution of their readability and determine whether significant shifts occurred following the release of ChatGPT in November 2022. Four standard readability formulas are used to calculate individual readability scores for each paper, classifying their level of readability. These scores are then aggregated by year and across the eight primary categories covered by the platform. The results show a steady annual decrease in readability, suggesting that abstracts are likely becoming increasingly complex. Additionally, following the release of ChatGPT, a significant change in readability is observed for 2023 and the analyzed months of 2024. Similar trends are found across categories, with most experiencing a notable change in readability during 2023 and 2024. These findings offer insights into the broader changes in readability and point to the likely influence of AI on scientific writing. 

---
# CitySim: Modeling Urban Behaviors and City Dynamics with Large-Scale LLM-Driven Agent Simulation 

**Authors**: Nicolas Bougie, Narimasa Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.21805)  

**Abstract**: Modeling human behavior in urban environments is fundamental for social science, behavioral studies, and urban planning. Prior work often rely on rigid, hand-crafted rules, limiting their ability to simulate nuanced intentions, plans, and adaptive behaviors. Addressing these challenges, we envision an urban simulator (CitySim), capitalizing on breakthroughs in human-level intelligence exhibited by large language models. In CitySim, agents generate realistic daily schedules using a recursive value-driven approach that balances mandatory activities, personal habits, and situational factors. To enable long-term, lifelike simulations, we endow agents with beliefs, long-term goals, and spatial memory for navigation. CitySim exhibits closer alignment with real humans than prior work, both at micro and macro levels. Additionally, we conduct insightful experiments by modeling tens of thousands of agents and evaluating their collective behaviors under various real-world scenarios, including estimating crowd density, predicting place popularity, and assessing well-being. Our results highlight CitySim as a scalable, flexible testbed for understanding and forecasting urban phenomena. 

---
# Fine-Grained Preference Optimization Improves Spatial Reasoning in VLMs 

**Authors**: Yifan Shen, Yuanzhe Liu, Jingyuan Zhu, Xu Cao, Xiaofeng Zhang, Yixiao He, Wenming Ye, James Matthew Rehg, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2506.21656)  

**Abstract**: Current Vision-Language Models (VLMs) struggle with fine-grained spatial reasoning, particularly when multi-step logic and precise spatial alignment are required. In this work, we introduce SpatialReasoner-R1, a vision-language reasoning model designed to address these limitations. To construct high-quality supervision for spatial reasoning, we design a Multi-Model Monte Carlo Tree Search (M3CTS) method that generates diverse, logically consistent Long Chain-of-Thought (LongCoT) reasoning trajectories. In addition, we propose fine-grained Direct Preference Optimization (fDPO), which introduces segment-specific preference granularity for descriptive grounding and logical reasoning, guided by a spatial reward mechanism that evaluates candidate responses based on visual consistency, spatial grounding, and logical coherence. Experimental results demonstrate that fDPO achieves an average improvement of 4.1% over standard DPO across spatial quality tasks, and a 9.0% gain in spatial quantity tasks. SpatialReasoner-R1, trained with fDPO, sets a new SoTA on SPATIALRGPT-Bench, outperforming the strongest baseline by 9.8% in average accuracy, while maintaining competitive performance on general vision-language tasks. 

---
