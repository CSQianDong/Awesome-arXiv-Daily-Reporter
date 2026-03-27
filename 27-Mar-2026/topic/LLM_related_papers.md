# An Experimental Comparison of the Most Popular Approaches to Fake News Detection 

**Authors**: Pietro Dell'Oglio, Alessandro Bondielli, Francesco Marcelloni, Lucia C. Passaro  

**Link**: [PDF](https://arxiv.org/pdf/2603.25501)  

**Abstract**: In recent years, fake news detection has received increasing attention in public debate and scientific research. Despite advances in detection techniques, the production and spread of false information have become more sophisticated, driven by Large Language Models (LLMs) and the amplification power of social media. We present a critical assessment of 12 representative fake news detection approaches, spanning traditional machine learning, deep learning, transformers, and specialized cross-domain architectures. We evaluate these methods on 10 publicly available datasets differing in genre, source, topic, and labeling rationale. We address text-only English fake news detection as a binary classification task by harmonizing labels into "Real" and "Fake" to ensure a consistent evaluation protocol. We acknowledge that label semantics vary across datasets and that harmonization inevitably removes such semantic nuances. Each dataset is treated as a distinct domain. We conduct in-domain, multi-domain and cross-domain experiments to simulate real-world scenarios involving domain shift and out-of-distribution data. Fine-tuned models perform well in-domain but struggle to generalize. Cross-domain architectures can reduce this gap but are data-hungry, while LLMs offer a promising alternative through zero- and few-shot learning. Given inherent dataset confounds and possible pre-training exposure, results should be interpreted as robustness evaluations within this English, text-only protocol. 

---
# Self-Improvement of Large Language Models: A Technical Overview and Future Outlook 

**Authors**: Haoyan Yang, Mario Xerri, Solha Park, Huajian Zhang, Yiyang Feng, Sai Akhil Kogilathota, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.25681)  

**Abstract**: As large language models (LLMs) continue to advance, improving them solely through human supervision is becoming increasingly costly and limited in scalability. As models approach human-level capabilities in certain domains, human feedback may no longer provide sufficiently informative signals for further improvement. At the same time, the growing ability of models to make autonomous decisions and execute complex actions naturally enables abstractions in which components of the model development process can be progressively automated. Together, these challenges and opportunities have driven increasing interest in self-improvement, where models autonomously generate data, evaluate outputs, and iteratively refine their own capabilities. In this paper, we present a system-level perspective on self-improving language models and introduce a unified framework that organizes existing techniques. We conceptualize the self-improvement system as a closed-loop lifecycle, consisting of four tightly coupled processes: data acquisition, data selection, model optimization, and inference refinement, along with an autonomous evaluation layer. Within this framework, the model itself plays a central role in driving each stage: collecting or generating data, selecting informative signals, updating its parameters, and refining outputs, while the autonomous evaluation layer continuously monitors progress and guides the improvement cycle across stages. Following this lifecycle perspective, we systematically review and analyze representative methods for each component from a technical standpoint. We further discuss current limitations and outline our vision for future research toward fully self-improving LLMs. 

---
# PICon: A Multi-Turn Interrogation Framework for Evaluating Persona Agent Consistency 

**Authors**: Minseo Kim, Sujeong Im, Junseong Choi, Junhee Lee, Chaeeun Shim, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.25620)  

**Abstract**: Large language model (LLM)-based persona agents are rapidly being adopted as scalable proxies for human participants across diverse domains. Yet there is no systematic method for verifying whether a persona agent's responses remain free of contradictions and factual inaccuracies throughout an interaction. A principle from interrogation methodology offers a lens: no matter how elaborate a fabricated identity, systematic interrogation will expose its contradictions. We apply this principle to propose PICon, an evaluation framework that probes persona agents through logically chained multi-turn questioning. PICon evaluates consistency along three core dimensions: internal consistency (freedom from self-contradiction), external consistency (alignment with real-world facts), and retest consistency (stability under repetition). Evaluating seven groups of persona agents alongside 63 real human participants, we find that even systems previously reported as highly consistent fail to meet the human baseline across all three dimensions, revealing contradictions and evasive responses under chained questioning. This work provides both a conceptual foundation and a practical methodology for evaluating persona agents before trusting them as substitutes for human participants. We provide the source code and an interactive demo at: this https URL 

---
# TAPO: Translation Augmented Policy Optimization for Multilingual Mathematical Reasoning 

**Authors**: Xu Huang, Zhejian Lai, Zixian Huang, Jiajun Chen, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25419)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in English mathematical reasoning, yet a significant performance disparity persists in multilingual contexts, largely attributed to deficiencies in language understanding. To bridge this gap, we introduce Translation-Augmented Policy Optimization (TAPO), a novel reinforcement learning framework built upon GRPO. TAPO enforces an explicit alignment strategy where the model leverages English as a pivot and follows an understand-then-reason paradigm. Crucially, we employ a step-level relative advantage mechanism that decouples understanding from reasoning, allowing the integration of translation quality rewards without introducing optimization conflicts. Extensive experiments reveal that TAPO effectively synergizes language understanding with reasoning capabilities and is compatible with various models. It outperforms baseline methods in both multilingual mathematical reasoning and translation tasks, while generalizing well to unseen languages and out-of-domain tasks. 

---
# Navigating the Prompt Space: Improving LLM Classification of Social Science Texts Through Prompt Engineering 

**Authors**: Erkan Gunes, Christoffer Florczak, Tevfik Murat Yildirim  

**Link**: [PDF](https://arxiv.org/pdf/2603.25422)  

**Abstract**: Recent developments in text classification using Large Language Models (LLMs) in the social sciences suggest that costs can be cut significantly, while performance can sometimes rival existing computational methods. However, with a wide variance in performance in current tests, we move to the question of how to maximize performance. In this paper, we focus on prompt context as a possible avenue for increasing accuracy by systematically varying three aspects of prompt engineering: label descriptions, instructional nudges, and few shot examples. Across two different examples, our tests illustrate that a minimal increase in prompt context yields the highest increase in performance, while further increases in context only tend to yield marginal performance increases thereafter. Alarmingly, increasing prompt context sometimes decreases accuracy. Furthermore, our tests suggest substantial heterogeneity across models, tasks, and batch size, underlining the need for individual validation of each LLM coding task rather than reliance on general rules. 

---
# Translation Asymmetry in LLMs as a Data Augmentation Factor: A Case Study for 6 Romansh Language Varieties 

**Authors**: Jannis Vamvas, Ignacio Pérez Prat, Angela Heldstab, Dominic P. Fischer, Sina Ahmadi, Rico Sennrich  

**Link**: [PDF](https://arxiv.org/pdf/2603.25489)  

**Abstract**: Recent strategies for low-resource machine translation rely on LLMs to generate synthetic data from higher-resource languages. We find that this method fails for Romansh, because LLMs tend to confuse its 6 distinct language varieties. Our experiments show that instead, the direction of data augmentation should be aligned with the resource gradient between source and target language. This approach surpasses Gemini 3 Pro in the lowest-resource variety of Romansh by 23 BLEU. A human evaluation confirms that our experiments yield the first model that generates fluent translations in the individual Romansh varieties. 

---
# Beyond Via: Analysis and Estimation of the Impact of Large Language Models in Academic Papers 

**Authors**: Mingmeng Geng, Yuhang Dong, Thierry Poibeau  

**Link**: [PDF](https://arxiv.org/pdf/2603.25638)  

**Abstract**: Through an analysis of arXiv papers, we report several shifts in word usage that are likely driven by large language models (LLMs) but have not previously received sufficient attention, such as the increased frequency of "beyond" and "via" in titles and the decreased frequency of "the" and "of" in abstracts. Due to the similarities among different LLMs, experiments show that current classifiers struggle to accurately determine which specific model generated a given text in multi-class classification tasks. Meanwhile, variations across LLMs also result in evolving patterns of word usage in academic papers. By adopting a direct and highly interpretable linear approach and accounting for differences between models and prompts, we quantitatively assess these effects and show that real-world LLM usage is heterogeneous and dynamic. 

---
# S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation 

**Authors**: Ligong Han, Hao Wang, Han Gao, Kai Xu, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2603.25702)  

**Abstract**: Block-diffusion language models offer a promising path toward faster-than-autoregressive generation by combining block-wise autoregressive decoding with within-block parallel denoising. However, in the few-step regime needed for practical acceleration, standard confidence-thresholded decoding is often brittle: aggressive thresholds hurt quality, while conservative thresholds require unnecessary denoising steps. Existing approaches that address this issue either require additional training or incur extra test-time compute. We present S2D2, a training-free self-speculative decoding framework for block-diffusion language models. Our key observation is that a block-diffusion model becomes autoregressive when the block size is reduced to one, allowing the same pretrained model to act as both drafter and verifier. S2D2 inserts a speculative verification step into standard block-diffusion decoding and uses lightweight routing policies to decide when verification is worth its cost. This yields a hybrid decoding trajectory in which diffusion proposes tokens in parallel, while the autoregressive mode acts as a local sequence-level critic. Across three mainstream block-diffusion families, S2D2 consistently improves the accuracy-speed tradeoff over strong confidence-thresholding baselines. On SDAR, we observe up to $4.7\times$ speedup over autoregressive decoding, and up to $1.57\times$ over a tuned dynamic decoding baseline while improving accuracy by up to $4.5$ points. On LLaDA2.1-Mini, S2D2 remains complementary to built-in self-correction, including a conservative setting where it is $4.4\times$ faster than the static baseline with slightly higher accuracy. 

---
# Large Language Model as Token Compressor and Decompressor 

**Authors**: Wenbing Li, Zikai Song, Jielei Zhang, Tianhao Zhao, Junkai Lin, Yiran Wang, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25340)  

**Abstract**: In this paper, we establish the novel insight that an off-the-shelf LLM can function as an excellent token compressor and decompressor. To demonstrate, we design a self-expressive autoencoding learning framework fine-tunes a pretrained LLM to translate long texts into a compact internal language of discrete, variable-length latent codes, termed Z-tokens, and to reconstruct the original text exactly from them. The resulting representation is content-adaptive: semantically dense segments receive more Z-tokens, while redundant or predictable regions are aggressively compressed, via lightweight LoRA-based adapter heads. Empirically, our method achieves up to 18 times token reduction on Wikipedia, CNN/DailyMail, HotpotQA, and Qulac-style long-query datasets, while preserving reconstruction fidelity and downstream performance. This simple yet effective design supports applications including prompt compression and autoregressive generation directly in the Z-token space, offering a potential pathway toward token-efficient long-context reasoning. 

---
# When Hate Meets Facts: LLMs-in-the-Loop for Check-worthiness Detection in Hate Speech 

**Authors**: Nicolás Benjamín Ocampo, Tommaso Caselli, Davide Ceolin  

**Link**: [PDF](https://arxiv.org/pdf/2603.25269)  

**Abstract**: Hateful content online is often expressed using fact-like, not necessarily correct information, especially in coordinated online harassment campaigns and extremist propaganda. Failing to jointly address hate speech (HS) and misinformation can deepen prejudice, reinforce harmful stereotypes, and expose bystanders to psychological distress, while polluting public debate. Moreover, these messages require more effort from content moderators because they must assess both harmfulness and veracity, i.e., fact-check them. To address this challenge, we release WSF-ARG+, the first dataset which combines hate speech with check-worthiness information. We also introduce a novel LLM-in-the-loop framework to facilitate the annotation of check-worthy claims. We run our framework, testing it with 12 open-weight LLMs of different sizes and architectures. We validate it through extensive human evaluation, and show that our LLM-in-the-loop framework reduces human effort without compromising the annotation quality of the data. Finally, we show that HS messages with check-worthy claims show significantly higher harassment and hate, and that incorporating check-worthiness labels improves LLM-based HS detection up to 0.213 macro-F1 and to 0.154 macro-F1 on average for large models. 

---
# MolQuest: A Benchmark for Agentic Evaluation of Abductive Reasoning in Chemical Structure Elucidation 

**Authors**: Taolin Han, Shuang Wu, Jinghang Wang, Yuhao Zhou, Renquan Lv, Bing Zhao, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.25253)  

**Abstract**: Large language models (LLMs) hold considerable potential for advancing scientific discovery, yet systematic assessment of their dynamic reasoning in real-world research remains limited. Current scientific evaluation benchmarks predominantly rely on static, single-turn Question Answering (QA) formats, which are inadequate for measuring model performance in complex scientific tasks that require multi-step iteration and experimental interaction. To address this gap, we introduce MolQuest, a novel agent-based evaluation framework for molecular structure elucidation built upon authentic chemical experimental data. Unlike existing datasets, MolQuest formalizes molecular structure elucidation as a multi-turn interactive task, requiring models to proactively plan experimental steps, integrate heterogeneous spectral sources (e.g., NMR, MS), and iteratively refine structural hypotheses. This framework systematically evaluates LLMs' abductive reasoning and strategic decision-making abilities within a vast and complex chemical space. Empirical results reveal that contemporary frontier models exhibit significant limitations in authentic scientific scenarios: notably, even state-of-the-art (SOTA) models achieve an accuracy of only approximately 50%, while the performance of most other models remains below the 30% threshold. This work provides a reproducible and extensible framework for science-oriented LLM evaluation, our findings highlight the critical gap in current LLMs' strategic scientific reasoning, setting a clear direction for future research toward AI that can actively participate in the scientific process. 

---
# SafeMath: Inference-time Safety improves Math Accuracy 

**Authors**: Sagnik Basu, Subhrajit Mitra, Aman Juneja, Somnath Banerjee, Rima Hazra, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2603.25201)  

**Abstract**: Recent research points toward LLMs being manipulated through adversarial and seemingly benign inputs, resulting in harmful, biased, or policy-violating outputs. In this paper, we study an underexplored issue concerning harmful and toxic mathematical word problems. We show that math questions, particularly those framed as natural language narratives, can serve as a subtle medium for propagating biased, unethical, or psychologically harmful content, with heightened risks in educational settings involving children. To support a systematic study of this phenomenon, we introduce ToxicGSM, a dataset of 1.9k arithmetic problems in which harmful or sensitive context is embedded while preserving mathematically well-defined reasoning tasks. Using this dataset, we audit the behaviour of existing LLMs and analyse the trade-offs between safety enforcement and mathematical correctness. We further propose SafeMath -- a safety alignment technique that reduces harmful outputs while maintaining, and in some cases improving, mathematical reasoning performance. Our results highlight the importance of disentangling linguistic harm from math reasoning and demonstrate that effective safety alignment need not come at the cost of accuracy. We release the source code and dataset at this https URL. 

---
# Probing the Lack of Stable Internal Beliefs in LLMs 

**Authors**: Yifan Luo, Kangping Xu, Yanzhen Lu, Yang Yuan, Andrew Chi-Chih Yao  

**Link**: [PDF](https://arxiv.org/pdf/2603.25187)  

**Abstract**: Persona-driven large language models (LLMs) require consistent behavioral tendencies across interactions to simulate human-like personality traits, such as persistence or reliability. However, current LLMs often lack stable internal representations that anchor their responses over extended dialogues. This work explores whether LLMs can maintain "implicit consistency", defined as persistent adherence to an unstated goal in multi-turn interactions. We designed a 20-question-style riddle game paradigm where an LLM is tasked with secretly selecting a target and responding to users' guesses with "yes/no" answers. Through evaluations, we find that LLMs struggle to preserve latent consistency: their implicit "goals" shift across turns unless explicitly provided their selected target in context. These findings highlight critical limitations in the building of persona-driven LLMs and underscore the need for mechanisms that anchor implicit goals over time, which is a key to realistic personality modeling in interactive applications such as dialogue systems. 

---
# OMIND: Framework for Knowledge Grounded Finetuning and Multi-Turn Dialogue Benchmark for Mental Health LLMs 

**Authors**: Suraj Racha, Prashant Harish Joshi, Utkarsh Maurya, Nitin Yadav, Mridul Sharma, Ananya Kunisetty, Saranya Darisipudi, Nirmal Punjabi, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2603.25105)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities for complex tasks, yet adaptation in medical domain, specifically mental health, poses specific challenges. Mental health is a rising concern globally with LLMs having large potential to help address the same. We highlight three primary challenges for LLMs in mental health - lack of high quality interpretable and knowledge grounded training data; training paradigms restricted to core capabilities, and evaluation of multi turn dialogue settings. Addressing it, we present oMind framework which includes training and aligning LLM agents for diverse capabilities including conversations; high quality ~164k multi-task SFT dataset, as a result of our generation pipeline based on Structured Knowledge retrieval, LLM based pruning, and review actions. We also introduce oMind-Chat - a novel multi turn benchmark dataset with expert annotated turn level and conversation level rubrics. Our diverse experiments on both core capabilities and conversations shows oMind LLMs consistently outperform baselines. oMind-LLM also shows significantly better reasoning with up to 80% win rate. 

---
# Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2603.25112)  

**Abstract**: Standard evaluation of LLM confidence relies on calibration metrics (ECE, Brier score) that conflate two distinct capacities: how much a model knows (Type-1 sensitivity) and how well it knows what it knows (Type-2 metacognitive sensitivity). We introduce an evaluation framework based on Type-2 Signal Detection Theory that decomposes these capacities using meta-d' and the metacognitive efficiency ratio M-ratio. Applied to four LLMs (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Llama-3-8B-Base, Gemma-2-9B-Instruct) across 224,000 factual QA trials, we find: (1) metacognitive efficiency varies substantially across models even when Type-1 sensitivity is similar -- Mistral achieves the highest d' but the lowest M-ratio; (2) metacognitive efficiency is domain-specific, with different models showing different weakest domains, invisible to aggregate metrics; (3) temperature manipulation shifts Type-2 criterion while meta-d' remains stable for two of four models, dissociating confidence policy from metacognitive capacity; (4) AUROC_2 and M-ratio produce fully inverted model rankings, demonstrating these metrics answer fundamentally different evaluation questions. The meta-d' framework reveals which models "know what they don't know" versus which merely appear well-calibrated due to criterion placement -- a distinction with direct implications for model selection, deployment, and human-AI collaboration. Pre-registered analysis; code and data publicly available. 

---
# Closing the Confidence-Faithfulness Gap in Large Language Models 

**Authors**: Miranda Muqing Miao, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2603.25052)  

**Abstract**: Large language models (LLMs) tend to verbalize confidence scores that are largely detached from their actual accuracy, yet the geometric relationship governing this behavior remain poorly understood. In this work, we present a mechanistic interpretability analysis of verbalized confidence, using linear probes and contrastive activation addition (CAA) steering to show that calibration and verbalized confidence signals are encoded linearly but are orthogonal to one another -- a finding consistent across three open-weight models and four datasets. Interestingly, when models are prompted to simultaneously reason through a problem and verbalize a confidence score, the reasoning process disrupts the verbalized confidence direction, exacerbating miscalibration. We term this the "Reasoning Contamination Effect." Leveraging this insight, we introduce a two-stage adaptive steering pipeline that reads the model's internal accuracy estimate and steers verbalized output to match it, substantially improving calibration alignment across all evaluated models. 

---
# LLM-Driven Reasoning for Constraint-Aware Feature Selection in Industrial Systems 

**Authors**: Yuhang Zhou, Zhuokai Zhao, Ke Li, Spilios Evmorfos, Gökalp Demirci, Mingyi Wang, Qiao Liu, Qifei Wang, Serena Li, Weiwei Li, Tingting Wang, Mingze Gao, Gedi Zhou, Abhishek Kumar, Xiangjun Fan, Lizhu Zhang, Jiayi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24979)  

**Abstract**: Feature selection is a crucial step in large-scale industrial machine learning systems, directly affecting model accuracy, efficiency, and maintainability. Traditional feature selection methods rely on labeled data and statistical heuristics, making them difficult to apply in production environments where labeled data are limited and multiple operational constraints must be satisfied. To address this, we propose Model Feature Agent (MoFA), a model-driven framework that performs sequential, reasoning-based feature selection using both semantic and quantitative feature information. MoFA incorporates feature definitions, importance scores, correlations, and metadata (e.g., feature groups or types) into structured prompts and selects features through interpretable, constraint-aware reasoning. We evaluate MoFA in three real-world industrial applications: (1) True Interest and Time-Worthiness Prediction, where it improves accuracy while reducing feature group complexity, (2) Value Model Enhancement, where it discovers high-order interaction terms that yield substantial engagement gains in online experiments, and (3) Notification Behavior Prediction, where it selects compact, high-value feature subsets that improve both model accuracy and inference efficiency. Together, these results demonstrate the practicality and effectiveness of LLM-based reasoning for feature selection in real production systems. 

---
# Approaches to Analysing Historical Newspapers Using LLMs 

**Authors**: Filip Dobranić, Tina Munda, Oliver Pejić, Vojko Gorjanc, Uroš Šmajdek, David Bordon, Jakob Lenardič, Tjaša Konovšek, Kristina Pahor de Maiti Tekavčič, Ciril Bohak, Darja Fišer  

**Link**: [PDF](https://arxiv.org/pdf/2603.25051)  

**Abstract**: This study presents a computational analysis of the Slovene historical newspapers \textit{Slovenec} and \textit{Slovenski narod} from the sPeriodika corpus, combining topic modelling, large language model (LLM)-based aspect-level sentiment analysis, entity-graph visualisation, and qualitative discourse analysis to examine how collective identities, political orientations, and national belonging were represented in public discourse at the turn of the twentieth century. Using BERTopic, we identify major thematic patterns and show both shared concerns and clear ideological differences between the two newspapers, reflecting their conservative-Catholic and liberal-progressive orientations. We further evaluate four instruction-following LLMs for targeted sentiment classification in OCR-degraded historical Slovene and select the Slovene-adapted GaMS3-12B-Instruct model as the most suitable for large-scale application, while also documenting important limitations, particularly its stronger performance on neutral sentiment than on positive or negative sentiment. Applied at dataset scale, the model reveals meaningful variation in the portrayal of collective identities, with some groups appearing predominantly in neutral descriptive contexts and others more often in evaluative or conflict-related discourse. We then create NER graphs to explore the relationships between collective identities and places. We apply a mixed methods approach to analyse the named entity graphs, combining quantitative network analysis with critical discourse analysis. The investigation focuses on the emergence and development of intertwined historical political and socionomic identities. Overall, the study demonstrates the value of combining scalable computational methods with critical interpretation to support digital humanities research on noisy historical newspaper data. 

---
# Prompt Attack Detection with LLM-as-a-Judge and Mixture-of-Models 

**Authors**: Hieu Xuan Le, Benjamin Goh, Quy Anh Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25176)  

**Abstract**: Prompt attacks, including jailbreaks and prompt injections, pose a critical security risk to Large Language Model (LLM) systems. In production, guardrails must mitigate these attacks under strict low-latency constraints, resulting in a deployment gap in which lightweight classifiers and rule-based systems struggle to generalize under distribution shift, while high-capacity LLM-based judges remain too slow or costly for live enforcement. In this work, we examine whether lightweight, general-purpose LLMs can reliably serve as security judges under real-world production constraints. Through careful prompt and output design, lightweight LLMs are guided through a structured reasoning process involving explicit intent decomposition, safety-signal verification, harm assessment, and self-reflection. We evaluate our method on a curated dataset combining benign queries from real-world chatbots with adversarial prompts generated via automated red teaming (ART), covering diverse and evolving patterns. Our results show that general-purpose LLMs, such as gemini-2.0-flash-lite-001, can serve as effective low-latency judges for live guardrails. This configuration is currently deployed in production as a centralized guardrail service for public service chatbots in Singapore. We additionally evaluate a Mixture-of-Models (MoM) setting to assess whether aggregating multiple LLM judges improves prompt-attack detection performance relative to single-model judges, with only modest gains observed. 

---
# Fine-Tuning A Large Language Model for Systematic Review Screening 

**Authors**: Kweku Yamoah, Noah Schroeder, Emmanuel Dorley, Neha Rani, Caleb Schutz  

**Link**: [PDF](https://arxiv.org/pdf/2603.24767)  

**Abstract**: Systematic reviews traditionally have taken considerable amounts of human time and energy to complete, in part due to the extensive number of titles and abstracts that must be reviewed for potential inclusion. Recently, researchers have begun to explore how to use large language models (LLMs) to make this process more efficient. However, research to date has shown inconsistent results. We posit this is because prompting alone may not provide sufficient context for the model(s) to perform well. In this study, we fine-tune a small 1.2 billion parameter open-weight LLM specifically for study screening in the context of a systematic review in which humans rated more than 8500 titles and abstracts for potential inclusion. Our results showed strong performance improvements from the fine-tuned model, with the weighted F1 score improving 80.79% compared to the base model. When run on the full dataset of 8,277 studies, the fine-tuned model had 86.40% agreement with the human coder, a 91.18% true positive rate, a 86.38% true negative rate, and perfect agreement across multiple inference runs. Taken together, our results show that there is promise for fine-tuning LLMs for title and abstract screening in large-scale systematic reviews. 

---
# Evaluating Fine-Tuned LLM Model For Medical Transcription With Small Low-Resource Languages Validated Dataset 

**Authors**: Mohammed Nowshad Ruhani Chowdhury, Mohammed Nowaz Rabbani Chowdhury, Sakari Lukkarinen  

**Link**: [PDF](https://arxiv.org/pdf/2603.24772)  

**Abstract**: Clinical documentation is a critical factor for patient safety, diagnosis, and continuity of care. The administrative burden of EHRs is a significant factor in physician burnout. This is a critical issue for low-resource languages, including Finnish. This study aims to investigate the effectiveness of a domain-aligned natural language processing (NLP); large language model for medical transcription in Finnish by fine-tuning LLaMA 3.1-8B on a small validated corpus of simulated clinical conversations by students at Metropolia University of Applied Sciences. The fine-tuning process for medical transcription used a controlled preprocessing and optimization approach. The fine-tuning effectiveness was evaluated by sevenfold cross-validation. The evaluation metrics for fine-tuned LLaMA 3.1-8B were BLEU = 0.1214, ROUGE-L = 0.4982, and BERTScore F1 = 0.8230. The results showed a low n-gram overlap but a strong semantic similarity with reference transcripts. This study indicate that fine-tuning can be an effective approach for translation of medical discourse in spoken Finnish and support the feasibility of fine-tuning a privacy-oriented domain-specific large language model for clinical documentation in Finnish. Beside that provide directions for future work. 

---
# Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection 

**Authors**: Ruichao Yang, Wei Gao, Xiaobin Zhu, Jing Ma, Hongzhan Lin, Ziyang Luo, Bo-Wen Zhang, Xu-Cheng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.25203)  

**Abstract**: Multimodal misinformation poses an escalating challenge that often evades traditional detectors, which are opaque black boxes and fragile against new manipulation tactics. We present Probabilistic Concept Graph Reasoning (PCGR), an interpretable and evolvable framework that reframes multimodal misinformation detection (MMD) as structured and concept-based reasoning. PCGR follows a build-then-infer paradigm, which first constructs a graph of human-understandable concept nodes, including novel high-level concepts automatically discovered and validated by multimodal large language models (MLLMs), and then applies hierarchical attention over this concept graph to infer claim veracity. This design produces interpretable reasoning chains linking evidence to conclusions. Experiments demonstrate that PCGR achieves state-of-the-art MMD accuracy and robustness to emerging manipulation types, outperforming prior methods in both coarse detection and fine-grained manipulation recognition. 

---
# Bilingual Text-to-Motion Generation: A New Benchmark and Baselines 

**Authors**: Wanjiang Weng, Xiaofeng Tan, Xiangbo Shu, Guo-Sen Xie, Pan Zhou, Hongsong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25178)  

**Abstract**: Text-to-motion generation holds significant potential for cross-linguistic applications, yet it is hindered by the lack of bilingual datasets and the poor cross-lingual semantic understanding of existing language models. To address these gaps, we introduce BiHumanML3D, the first bilingual text-to-motion benchmark, constructed via LLM-assisted annotation and rigorous manual correction. Furthermore, we propose a simple yet effective baseline, Bilingual Motion Diffusion (BiMD), featuring Cross-Lingual Alignment (CLA). CLA explicitly aligns semantic representations across languages, creating a robust conditional space that enables high-quality motion generation from bilingual inputs, including zero-shot code-switching scenarios. Extensive experiments demonstrate that BiMD with CLA achieves an FID of 0.045 vs. 0.169 and R@3 of 82.8\% vs. 80.8\%, significantly outperforms monolingual diffusion models and translation baselines on BiHumanML3D, underscoring the critical necessity and reliability of our dataset and the effectiveness of our alignment strategy for cross-lingual motion synthesis. The dataset and code are released at \href{this https URL}{this https URL} 

---
# Can MLLMs Read Students' Minds? Unpacking Multimodal Error Analysis in Handwritten Math 

**Authors**: Dingjie Song, Tianlong Xu, Yi-Fan Zhang, Hang Li, Zhiling Yan, Xing Fan, Haoyang Li, Lichao Sun, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.24961)  

**Abstract**: Assessing student handwritten scratchwork is crucial for personalized educational feedback but presents unique challenges due to diverse handwriting, complex layouts, and varied problem-solving approaches. Existing educational NLP primarily focuses on textual responses and neglects the complexity and multimodality inherent in authentic handwritten scratchwork. Current multimodal large language models (MLLMs) excel at visual reasoning but typically adopt an "examinee perspective", prioritizing generating correct answers rather than diagnosing student errors. To bridge these gaps, we introduce ScratchMath, a novel benchmark specifically designed for explaining and classifying errors in authentic handwritten mathematics scratchwork. Our dataset comprises 1,720 mathematics samples from Chinese primary and middle school students, supporting two key tasks: Error Cause Explanation (ECE) and Error Cause Classification (ECC), with seven defined error types. The dataset is meticulously annotated through rigorous human-machine collaborative approaches involving multiple stages of expert labeling, review, and verification. We systematically evaluate 16 leading MLLMs on ScratchMath, revealing significant performance gaps relative to human experts, especially in visual recognition and logical reasoning. Proprietary models notably outperform open-source models, with large reasoning models showing strong potential for error explanation. All evaluation data and frameworks are publicly available to facilitate further research. 

---
# FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol 

**Authors**: Jie Zhu, Yimin Tian, Boyang Li, Kehao Wu, Zhongzhi Liang, Junhui Li, Xianyin Zhang, Lifan Guo, Feng Chen, Yong Liu, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24943)  

**Abstract**: This paper introduces \textbf{FinMCP-Bench}, a novel benchmark for evaluating large language models (LLMs) in solving real-world financial problems through tool invocation of financial model context protocols. FinMCP-Bench contains 613 samples spanning 10 main scenarios and 33 sub-scenarios, featuring both real and synthetic user queries to ensure diversity and authenticity. It incorporates 65 real financial MCPs and three types of samples, single tool, multi-tool, and multi-turn, allowing evaluation of models across different levels of task complexity. Using this benchmark, we systematically assess a range of mainstream LLMs and propose metrics that explicitly measure tool invocation accuracy and reasoning capabilities. FinMCP-Bench provides a standardized, practical, and challenging testbed for advancing research on financial LLM agents. 

---
# Measuring What Matters -- or What's Convenient?: Robustness of LLM-Based Scoring Systems to Construct-Irrelevant Factors 

**Authors**: Cole Walsh, Rodica Ivan  

**Link**: [PDF](https://arxiv.org/pdf/2603.25674)  

**Abstract**: Automated systems have been widely adopted across the educational testing industry for open-response assessment and essay scoring. These systems commonly achieve performance levels comparable to or superior than trained human raters, but have frequently been demonstrated to be vulnerable to the influence of construct-irrelevant factors (i.e., features of responses that are unrelated to the construct assessed) and adversarial conditions. Given the rising usage of large language models in automated scoring systems, there is a renewed focus on ``hallucinations'' and the robustness of these LLM-based automated scoring approaches to construct-irrelevant factors. This study investigates the effects of construct-irrelevant factors on a dual-architecture LLM-based scoring system designed to score short essay-like open-response items in a situational judgment test. It was found that the scoring system was generally robust to padding responses with meaningless text, spelling errors, and writing sophistication. Duplicating large passages of text resulted in lower scores predicted by the system, on average, contradicting results from previous studies of non-LLM-based scoring systems, while off-topic responses were heavily penalized by the scoring system. These results provide encouraging support for the robustness of future LLM-based scoring systems when designed with construct relevance in mind. 

---
# LogitScope: A Framework for Analyzing LLM Uncertainty Through Information Metrics 

**Authors**: Farhan Ahmed, Yuya Jeremy Ong, Chad DeLuca  

**Link**: [PDF](https://arxiv.org/pdf/2603.24929)  

**Abstract**: Understanding and quantifying uncertainty in large language model (LLM) outputs is critical for reliable deployment. However, traditional evaluation approaches provide limited insight into model confidence at individual token positions during generation. To address this issue, we introduce LogitScope, a lightweight framework for analyzing LLM uncertainty through token-level information metrics computed from probability distributions. By measuring metrics such as entropy and varentropy at each generation step, LogitScope reveals patterns in model confidence, identifies potential hallucinations, and exposes decision points where models exhibit high uncertainty, all without requiring labeled data or semantic interpretation. We demonstrate LogitScope's utility across diverse applications including uncertainty quantification, model behavior analysis, and production monitoring. The framework is model-agnostic, computationally efficient through lazy evaluation, and compatible with any HuggingFace model, enabling both researchers and practitioners to inspect LLM behavior during inference. 

---
# Reaching Beyond the Mode: RL for Distributional Reasoning in Language Models 

**Authors**: Isha Puri, Mehul Damani, Idan Shenfeld, Marzyeh Ghassemi, Jacob Andreas, Yoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.24844)  

**Abstract**: Given a question, a language model (LM) implicitly encodes a distribution over possible answers. In practice, post-training procedures for LMs often collapse this distribution onto a single dominant mode. While this is generally not a problem for benchmark-style evaluations that assume one correct answer, many real-world tasks inherently involve multiple valid answers or irreducible uncertainty. Examples include medical diagnosis, ambiguous question answering, and settings with incomplete information. In these cases, we would like LMs to generate multiple plausible hypotheses, ideally with confidence estimates for each one, and without computationally intensive repeated sampling to generate non-modal answers. This paper describes a multi-answer reinforcement learning approach for training LMs to perform distributional reasoning over multiple answers during inference. We modify the RL objective to enable models to explicitly generate multiple candidate answers in a single forward pass, internalizing aspects of inference-time search into the model's generative process. Across question-answering, medical diagnostic, and coding benchmarks, we observe improved diversity, coverage, and set-level calibration scores compared to single answer trained baselines. Models trained with our approach require fewer tokens to generate multiple answers than competing approaches. On coding tasks, they are also substantially more accurate. These results position multi-answer RL as a principled and compute-efficient alternative to inference-time scaling procedures such as best-of-k. Code and more information can be found at this https URL. 

---
# X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs 

**Authors**: Di Cao, Dongjie Fu, Hai Yu, Siqi Zheng, Xu Tan, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.24596)  

**Abstract**: While the shift from cascaded dialogue systems to end-to-end (E2E) speech Large Language Models (LLMs) improves latency and paralinguistic modeling, E2E models often exhibit a significant performance degradation compared to their text-based counterparts. The standard Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) training methods fail to close this gap. To address this, we propose X-OPD, a novel Cross-Modal On-Policy Distillation framework designed to systematically align the capabilities of Speech LLMs to their text-based counterparts. X-OPD enables the Speech LLM to explore its own distribution via on-policy rollouts, where a text-based teacher model evaluates these trajectories and provides token-level feedback, effectively distilling teacher's capabilities into student's multi-modal representations. Extensive experiments across multiple benchmarks demonstrate that X-OPD significantly narrows the gap in complex tasks while preserving the model's inherent capabilities. 

---
# Training LLMs for Multi-Step Tool Orchestration with Constrained Data Synthesis and Graduated Rewards 

**Authors**: Cheng Jiayang, Xin Liu, Zhihan Zhang, Haoyang Wen, Zixuan Zhang, Qingyu Yin, Shiyang Li, Priyanka Nigam, Bing Yin, Chao Zhang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.24709)  

**Abstract**: Multi-step tool orchestration, where LLMs must invoke multiple dependent APIs in the correct order while propagating intermediate outputs, remains challenging. State-of-the-art models frequently fail on full sequence execution, with parameter value errors accounting for a significant portion of failures. Training models to handle such workflows faces two obstacles: existing environments focus on simple per-turn function calls with simulated data, and binary rewards provide no signal for partial correctness.
We present a framework addressing both challenges. First, we construct a reinforcement learning environment backed by a large-scale cache of real API responses, enabling a data synthesis pipeline that samples valid multi-step orchestration traces with controllable complexity and significantly higher generation efficiency than unconstrained methods. Second, we propose a graduated reward design that decomposes correctness into atomic validity (individual function call correctness at increasing granularity) and orchestration (correct tool sequencing with dependency respect). On ComplexFuncBench, our approach demonstrates substantial improvements in turn accuracy. Ablation studies confirm both reward components are essential: using either alone significantly degrades performance. 

---
# Is Mathematical Problem-Solving Expertise in Large Language Models Associated with Assessment Performance? 

**Authors**: Liang Zhang, Yu Fu, Xinyi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.25633)  

**Abstract**: Large Language Models (LLMs) are increasingly used in math education not only as problem solvers but also as assessors of learners' reasoning. However, it remains unclear whether stronger math problem-solving ability is associated with stronger step-level assessment performance. This study examines that relationship using the GSM8K and MATH subsets of PROCESSBENCH, a human-annotated benchmark for identifying the earliest erroneous step in mathematical reasoning. We evaluate two LLM-based math tutor agent settings, instantiated with GPT-4 and GPT-5, in two independent tasks on the same math problems: solving the original problem and assessing a benchmark-provided solution by predicting the earliest erroneous step. Results show a consistent within-model pattern: assessment accuracy is substantially higher on math problem items the same model solved correctly than on items it solved incorrectly, with statistically significant associations across both models and datasets. At the same time, assessment remains more difficult than direct problem solving, especially on error-present solutions. These findings suggest that math problem-solving expertise supports stronger assessment performance, but reliable step-level diagnosis also requires additional capabilities such as step tracking, monitoring, and precise error localization. The results have implications for the design and evaluation of AI-supported Adaptive Instructional Systems (AISs) for formative assessment in math education. 

---
# Beyond Content Safety: Real-Time Monitoring for Reasoning Vulnerabilities in Large Language Models 

**Authors**: Xunguang Wang, Yuguang Zhou, Qingyue Wang, Zongjie Li, Ruixuan Huang, Zhenlan Ji, Pingchuan Ma, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25412)  

**Abstract**: Large language models (LLMs) increasingly rely on explicit chain-of-thought (CoT) reasoning to solve complex tasks, yet the safety of the reasoning process itself remains largely unaddressed. Existing work on LLM safety focuses on content safety--detecting harmful, biased, or factually incorrect outputs -- and treats the reasoning chain as an opaque intermediate artifact. We identify reasoning safety as an orthogonal and equally critical security dimension: the requirement that a model's reasoning trajectory be logically consistent, computationally efficient, and resistant to adversarial manipulation. We make three contributions. First, we formally define reasoning safety and introduce a nine-category taxonomy of unsafe reasoning behaviors, covering input parsing errors, reasoning execution errors, and process management errors. Second, we conduct a large-scale prevalence study annotating 4111 reasoning chains from both natural reasoning benchmarks and four adversarial attack methods (reasoning hijacking and denial-of-service), confirming that all nine error types occur in practice and that each attack induces a mechanistically interpretable signature. Third, we propose a Reasoning Safety Monitor: an external LLM-based component that runs in parallel with the target model, inspects each reasoning step in real time via a taxonomy-embedded prompt, and dispatches an interrupt signal upon detecting unsafe behavior. Evaluation on a 450-chain static benchmark shows that our monitor achieves up to 84.88\% step-level localization accuracy and 85.37\% error-type classification accuracy, outperforming hallucination detectors and process reward model baselines by substantial margins. These results demonstrate that reasoning-level monitoring is both necessary and practically achievable, and establish reasoning safety as a foundational concern for the secure deployment of large reasoning models. 

---
# Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills 

**Authors**: Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Xiaoxi Jiang, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25158)  

**Abstract**: Equipping Large Language Model (LLM) agents with domain-specific skills is critical for tackling complex tasks. Yet, manual authoring creates a severe scalability bottleneck. Conversely, automated skill generation often yields fragile or fragmented results because it either relies on shallow parametric knowledge or sequentially overfits to non-generalizable trajectory-local lessons. To overcome this, we introduce Trace2Skill, a framework that mirrors how human experts author skills: by holistically analyzing broad execution experience before distilling it into a single, comprehensive guide. Instead of reacting sequentially to individual trajectories, Trace2Skill dispatches a parallel fleet of sub-agents to analyze a diverse pool of executions. It extracts trajectory-specific lessons and hierarchically consolidates them into a unified, conflict-free skill directory via inductive reasoning. Trace2Skill supports both deepening existing human-written skills and creating new ones from scratch. Experiments in challenging domains, such as spreadsheet, VisionQA and math reasoning, show that Trace2Skill significantly improves upon strong baselines, including Anthropic's official xlsx skills. Crucially, this trajectory-grounded evolution does not merely memorize task instances or model-specific quirks: evolved skills transfer across LLM scales and generalize to OOD settings. For example, skills evolved by Qwen3.5-35B on its own trajectories improved a Qwen3.5-122B agent by up to 57.65 absolute percentage points on WikiTableQuestions. Ultimately, our results demonstrate that complex agent experience can be packaged into highly transferable, declarative skills -- requiring no parameter updates, no external retrieval modules, and utilizing open-source models as small as 35B parameters. 

---
# RubricEval: A Rubric-Level Meta-Evaluation Benchmark for LLM Judges in Instruction Following 

**Authors**: Tianjun Pan, Xuan Lin, Wenyan Yang, Qianyu He, Shisong Chen, Licai Qi, Wanqing Xu, Hongwei Feng, Bo Xu, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.25133)  

**Abstract**: Rubric-based evaluation has become a prevailing paradigm for evaluating instruction following in large language models (LLMs). Despite its widespread use, the reliability of these rubric-level evaluations remains unclear, calling for meta-evaluation. However, prior meta-evaluation efforts largely focus on the response level, failing to assess the fine-grained judgment accuracy that rubric-based evaluation relies on. To bridge this gap, we introduce RubricEval. Our benchmark features: (1) the first rubric-level meta-evaluation benchmark for instruction following, (2) diverse instructions and responses spanning multiple categories and model sources, and (3) a substantial set of 3,486 quality-controlled instances, along with Easy/Hard subsets that better differentiates judge performance. Our experiments reveal that rubric-level judging remains far from solved: even GPT-4o, a widely adopted judge in instruction-following benchmarks, achieves only 55.97% on Hard subset. Considering evaluation paradigm, rubric-level evaluation outperforms checklist-level, explicit reasoning improves accuracy, and both together reduce inter-judge variance. Through our established rubric taxonomy, we further identify common failure modes and offer actionable insights for reliable instruction-following evaluation. 

---
# The Anatomy of Uncertainty in LLMs 

**Authors**: Aditya Taparia, Ransalu Senanayake, Kowshik Thopalli, Vivek Narayanaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2603.24967)  

**Abstract**: Understanding why a large language model (LLM) is uncertain about the response is important for their reliable deployment. Current approaches, which either provide a single uncertainty score or rely on the classical aleatoric-epistemic dichotomy, fail to offer actionable insights for improving the generative model. Recent studies have also shown that such methods are not enough for understanding uncertainty in LLMs. In this work, we advocate for an uncertainty decomposition framework that dissects LLM uncertainty into three distinct semantic components: (i) input ambiguity, arising from ambiguous prompts; (ii) knowledge gaps, caused by insufficient parametric evidence; and (iii) decoding randomness, stemming from stochastic sampling. Through a series of experiments we demonstrate that the dominance of these components can shift across model size and task. Our framework provides a better understanding to audit LLM reliability and detect hallucinations, paving the way for targeted interventions and more trustworthy systems. 

---
# From Stateless to Situated: Building a Psychological World for LLM-Based Emotional Support 

**Authors**: Boning Zhao, Clover Hu, Xinnuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.25031)  

**Abstract**: In psychological support and emotional companionship scenarios, the core limitation of large language models (LLMs) lies not merely in response quality, but in their reliance on local next-token prediction, which prevents them from maintaining the temporal continuity, stage awareness, and user consent boundaries required for multi-turn intervention. This stateless characteristic makes systems prone to premature advancement, stage misalignment, and boundary violations in continuous dialogue. To address this problem, we argue that the key challenge in process-oriented emotional support is not simply generating natural language, but constructing a sustainably updatable external situational structure for the model. We therefore propose LEKIA 2.0, a situated LLM architecture that separates the cognitive layer from the executive layer, thereby decoupling situational modeling from intervention execution. This design enables the system to maintain stable representations of the user's situation and consent boundaries throughout ongoing interaction. To evaluate this process-control capability, we further introduce a Static-to-Dynamic online evaluation protocol for multi-turn interaction. LEKIA achieved an average absolute improvement of approximately 31% over prompt-only baselines in deep intervention loop completion. The results suggest that an external situational structure is a key enabling condition for building stable, controllable, and situated emotional support systems. 

---
# Supervising Ralph Wiggum: Exploring a Metacognitive Co-Regulation Agentic AI Loop for Engineering Design 

**Authors**: Zeda Xu, Nikolas Martelaro, Christopher McComb  

**Link**: [PDF](https://arxiv.org/pdf/2603.24768)  

**Abstract**: The engineering design research community has studied agentic AI systems that use Large Language Model (LLM) agents to automate the engineering design process. However, these systems are prone to some of the same pathologies that plague humans. Just as human designers, LLM design agents can fixate on existing paradigms and fail to explore alternatives when solving design challenges, potentially leading to suboptimal solutions. In this work, we propose (1) a novel Self-Regulation Loop (SRL), in which the Design Agent self-regulates and explicitly monitors its own metacognition, and (2) a novel Co-Regulation Design Agentic Loop (CRDAL), in which a Metacognitive Co-Regulation Agent assists the Design Agent in metacognition to mitigate design fixation, thereby improving system performance for engineering design tasks. In the battery pack design problem examined here, we found that the novel CRDAL system generates designs with better performance, without significantly increasing the computational cost, compared to a plain Ralph Wiggum Loop (RWL) and the metacognitively self-assessing Self-Regulation Loop (SRL). Also, we found that the CRDAL system navigated through the latent design space more effectively than both SRL and RWL. However, the SRL did not generate designs with significantly better performance than RWL, even though it explored a different region of the design space. The proposed system architectures and findings of this work provide practical implications for future development of agentic AI systems for engineering design. 

---
# AutoSAM: an Agentic Framework for Automating Input File Generation for the SAM Code with Multi-Modal Retrieval-Augmented Generation 

**Authors**: Zaid Abulawi, Zavier Ndum Ndum, Eric Cervi, Rui Hu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24736)  

**Abstract**: In the design and safety analysis of advanced reactor systems, constructing input files for system-level thermal-hydraulics codes such as the System Analysis Module (SAM) remains a labor-intensive task. Analysts must extract and reconcile design data from heterogeneous engineering documents and manually translate it into solver-specific syntax. In this paper, we present AutoSAM, an agentic framework that automates SAM input file generation. The framework combines a large language model agent with retrieval-augmented generation over the solver's user guide and theory manual, together with specialized tools for analyzing PDFs, images, spreadsheets, and text files. AutoSAM ingests unstructured engineering documents, including system diagrams, design reports, and data tables, extracts simulation-relevant parameters into a human-auditable intermediate representation, and synthesizes validated, solver-compatible input decks. Its multimodal retrieval pipeline integrates scientific text extraction, vision-based figure interpretation, semantic embedding, and query answering. We evaluate AutoSAM on four case studies of increasing complexity: a single-pipe steady-state model, a solid-fuel channel with temperature reactivity feedback, the Advanced Burner Test Reactor core, and the Molten Salt Reactor Experiment primary loop. Across all cases, the agent produces runnable SAM models consistent with expected thermal-hydraulic behavior while explicitly identifying missing data and labeling assumed values. The framework achieves 100% utilization of structured inputs, about 88% extraction from PDF text, and 100% completeness in vision-based geometric extraction. These results demonstrate a practical path toward prompt-driven reactor modeling, in which analysts provide system descriptions and supporting documentation while the agent translates them into transparent, and executable, SAM simulations. 

---
# The Kitchen Loop: User-Spec-Driven Development for a Self-Evolving Codebase 

**Authors**: Yannick Roy  

**Link**: [PDF](https://arxiv.org/pdf/2603.25697)  

**Abstract**: Code production is now a commodity; the bottleneck is knowing what to build and proving it works. We present the Kitchen Loop, a framework for autonomous, self-evolving software built on a unified trust model: (1) a specification surface enumerating what the product claims to support; (2) 'As a User x 1000', where an LLM agent exercises that surface as a synthetic power user at 1,000x human cadence; (3) Unbeatable Tests, ground-truth verification the code author cannot fake; and (4) Drift Control, continuous quality measurement with automated pause gates. We validate across two production systems over 285+ iterations, producing 1,094+ merged pull requests with zero regressions detected by the regression oracle (methodology in Section 6.1). We observe emergent properties at scale: multi-iteration self-correction chains, autonomous infrastructure healing, and monotonically improving quality gates. The primitives are not new; our contribution is their composition into a production-tested system with the operational discipline that makes long-running autonomous evolution safe. 

---
# A Mentalistic Interface for Probing Folk-Psychological Attribution to Non-Humanoid Robots 

**Authors**: Giulio Pisaneschi, Pierpaolo Serio, Estelle Gerbier, Andrea Dan Ryals, Lorenzo Pollini, Mario G. C. A. Cimino  

**Link**: [PDF](https://arxiv.org/pdf/2603.25646)  

**Abstract**: This paper presents an experimental platform for studying intentional-state attribution toward a non-humanoid robot. The system combines a simulated robot, realistic task environments, and large language model-based explanatory layers that can express the same behavior in mentalistic, teleological, or mechanistic terms. By holding behavior constant while varying the explanatory frame, the platform provides a controlled way to investigate how language and framing shape the adoption of the intentional stance in robotics. 

---
# AD-CARE: A Guideline-grounded, Modality-agnostic LLM Agent for Real-world Alzheimer's Disease Diagnosis with Multi-cohort Assessment, Fairness Analysis, and Reader Study 

**Authors**: Wenlong Hou, Sheng Bi, Guangqian Yang, Lihao Liu, Ye Du, Hanxiao Xue, Juncheng Wang, Yuxiang Feng, Yue Xun, Nanxi Yu, Ning Mao, Mo Yang, Yi Wah Eva Cheung, Ling Long, Kay Chen Tan, Lequan Yu, Xiaomeng Ma, Shaozhen Yan, Shujun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25322)  

**Abstract**: Alzheimer's disease (AD) is a growing global health challenge as populations age, and timely, accurate diagnosis is essential to reduce individual and societal burden. However, real-world AD assessment is hampered by incomplete, heterogeneous multimodal data and variability across sites and patient demographics. Although large language models (LLMs) have shown promise in biomedicine, their use in AD has largely been confined to answering narrow, disease-specific questions rather than generating comprehensive diagnostic reports that support clinical decision-making. Here we expand LLM capabilities for clinical decision support by introducing AD-CARE, a modality-agnostic agent that performs guideline-grounded diagnostic assessment from incomplete, heterogeneous inputs without imputing missing modalities. By dynamically orchestrating specialized diagnostic tools and embedding clinical guidelines into LLM-driven reasoning, AD-CARE generates transparent, report-style outputs aligned with real-world clinical workflows. Across six cohorts comprising 10,303 cases, AD-CARE achieved 84.9% diagnostic accuracy, delivering 4.2%-13.7% relative improvements over baseline methods. Despite cohort-level differences, dataset-specific accuracies remain robust (80.4%-98.8%), and the agent consistently outperforms all baselines. AD-CARE reduced performance disparities across racial and age subgroups, decreasing the average dispersion of four metrics by 21%-68% and 28%-51%, respectively. In a controlled reader study, the agent improved neurologist and radiologist accuracy by 6%-11% and more than halved decision time. The framework yielded 2.29%-10.66% absolute gains over eight backbone LLMs and converges their performance. These results show that AD-CARE is a scalable, practically deployable framework that can be integrated into routine clinical workflows for multimodal decision support in AD. 

---
# Train at Moving Edge: Online-Verified Prompt Selection for Efficient RL Training of Large Reasoning Model 

**Authors**: Jiahao Wu, Ning Lu, Shengcai Liu, Kun Wang, Yanting Yang, Li Qing, Ke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25184)  

**Abstract**: Reinforcement learning (RL) has become essential for post-training large language models (LLMs) in reasoning tasks. While scaling rollouts can stabilize training and enhance performance, the computational overhead is a critical issue. In algorithms like GRPO, multiple rollouts per prompt incur prohibitive costs, as a large portion of prompts provide negligible gradients and are thus of low utility. To address this problem, we investigate how to select high-utility prompts before the rollout phase. Our experimental analysis reveals that sample utility is non-uniform and evolving: the strongest learning signals concentrate at the ``learning edge", the intersection of intermediate difficulty and high uncertainty, which shifts as training proceeds. Motivated by this, we propose HIVE (History-Informed and online-VErified prompt selection), a dual-stage framework for data-efficient RL. HIVE utilizes historical reward trajectories for coarse selection and employs prompt entropy as a real-time proxy to prune instances with stale utility. By evaluating HIVE across multiple math reasoning benchmarks and models, we show that HIVE yields significant rollout efficiency without compromising performance. 

---
# Photon: Speedup Volume Understanding with Efficient Multimodal Large Language Models 

**Authors**: Chengyu Fang, Heng Guo, Zheng Jiang, Chunming He, Xiu Li, Minfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.25155)  

**Abstract**: Multimodal large language models are promising for clinical visual question answering tasks, but scaling to 3D imaging is hindered by high computational costs. Prior methods often rely on 2D slices or fixed-length token compression, disrupting volumetric continuity and obscuring subtle findings. We present Photon, a framework that represents 3D medical volumes with token sequences of variable length. Photon introduces instruction-conditioned token scheduling and surrogate gradient propagation to adaptively reduce tokens during both training and inference, which lowers computational cost while mitigating the attention dilution caused by redundant tokens. It incorporates a custom backpropagation rule with gradient restoration to enable differentiable optimization despite discrete token drop. To stabilize token compression and ensure reliable use of visual evidence, Photon further applies regularization objectives that mitigate language-only bias and improve reliability. Experiments on diverse medical visual question answering tasks show that Photon achieves state-of-the-art accuracy while reducing resource usage and accelerating both training and inference. 

---
# Evaluating adaptive and generative AI-based feedback and recommendations in a knowledge-graph-integrated programming learning system 

**Authors**: Lalita Na Nongkhai, Jingyun Wang, Adam Wynn, Takahiko Mendori  

**Link**: [PDF](https://arxiv.org/pdf/2603.24940)  

**Abstract**: This paper introduces the design and development of a framework that integrates a large language model (LLM) with a retrieval-augmented generation (RAG) approach leveraging both a knowledge graph and user interaction history. The framework is incorporated into a previously developed adaptive learning support system to assess learners' code, generate formative feedback, and recommend exercises. Moerover, this study examines learner preferences across three instructional modes; adaptive, Generative AI (GenAI), and hybrid GenAI-adaptive. An experimental study was conducted to compare the learning performance and perception of the learners, and the effectiveness of these three modes using four key log features derived from 4956 code submissions across all experimental groups. The analysis results show that learners receiving feedback from GenAI modes had significantly more correct code and fewer code submissions missing essential programming logic than those receiving feedback from adaptive mode. In particular, the hybrid GenAI-adaptive mode achieved the highest number of correct submissions and the fewest incorrect or incomplete attempts, outperforming both the adaptive-only and GenAI-only modes. Questionnaire responses further indicated that GenAI-generated feedback was widely perceived as helpful, while all modes were rated positively for ease of use and usefulness. These results suggest that the hybrid GenAI-adaptive mode outperforms the other two modes across all measured log features. 

---
# Self-Corrected Image Generation with Explainable Latent Rewards 

**Authors**: Yinyi Luo, Hrishikesh Gokhale, Marios Savvides, Jindong Wang, Shengfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2603.24965)  

**Abstract**: Despite significant progress in text-to-image generation, aligning outputs with complex prompts remains challenging, particularly for fine-grained semantics and spatial relations. This difficulty stems from the feed-forward nature of generation, which requires anticipating alignment without fully understanding the output. In contrast, evaluating generated images is more tractable. Motivated by this asymmetry, we propose xLARD, a self-correcting framework that uses multimodal large language models to guide generation through Explainable LAtent RewarDs. xLARD introduces a lightweight corrector that refines latent representations based on structured feedback from model-generated references. A key component is a differentiable mapping from latent edits to interpretable reward signals, enabling continuous latent-level guidance from non-differentiable image-level evaluations. This mechanism allows the model to understand, assess, and correct itself during generation. Experiments across diverse generation and editing tasks show that xLARD improves semantic alignment and visual fidelity while maintaining generative priors. Code is available at this https URL. 

---
# NeuroVLM-Bench: Evaluation of Vision-Enabled Large Language Models for Clinical Reasoning in Neurological Disorders 

**Authors**: Katarina Trojachanec Dineva, Stefan Andonov, Ilinka Ivanoska, Ivan Kitanovski, Sasho Gramatikov, Tamara Kostova, Monika Simjanoska Misheva, Kostadin Mishev  

**Link**: [PDF](https://arxiv.org/pdf/2603.24846)  

**Abstract**: Recent advances in multimodal large language models enable new possibilities for image-based decision support. However, their reliability and operational trade-offs in neuroimaging remain insufficiently understood. We present a comprehensive benchmarking study of vision-enabled large language models for 2D neuroimaging using curated MRI and CT datasets covering multiple sclerosis, stroke, brain tumors, other abnormalities, and normal controls. Models are required to generate multiple outputs simultaneously, including diagnosis, diagnosis subtype, imaging modality, specialized sequence, and anatomical plane. Performance is evaluated across four directions: discriminative classification with abstention, calibration, structured-output validity, and computational efficiency. A multi-phase framework ensures fair comparison while controlling for selection bias. Across twenty frontier multimodal models, the results show that technical imaging attributes such as modality and plane are nearly solved, whereas diagnostic reasoning, especially subtype prediction, remains challenging. Tumor classification emerges as the most reliable task, stroke is moderately solvable, while multiple sclerosis and rare abnormalities remain difficult. Few-shot prompting improves performance for several models but increases token usage, latency, and cost. Gemini-2.5-Pro and GPT-5-Chat achieve the strongest overall diagnostic performance, while Gemini-2.5-Flash offers the best efficiency-performance trade-off. Among open-weight architectures, MedGemma-1.5-4B demonstrates the most promising results, as under few-shot prompting, it approaches the zero-shot performance of several proprietary models, while maintaining perfect structured output. These findings provide practical insights into performance, reliability, and efficiency trade-offs, supporting standardized evaluation of multimodal LLMs in neuroimaging. 

---
# Learning From Developers: Towards Reliable Patch Validation at Scale for Linux 

**Authors**: Chih-En Lin, Attreyee Mukherjee, Ajay Rawat, Ruqi Zhang, Pedro Fonseca  

**Link**: [PDF](https://arxiv.org/pdf/2603.24825)  

**Abstract**: Patch reviewing is critical for software development, especially in distributed open-source development, which highly depends on voluntary work, such as Linux. This paper studies the past 10 years of patch reviews of the Linux memory management subsystem to characterize the challenges involved in patch reviewing at scale. Our study reveals that the review process is still primarily reliant on human effort despite a wide-range of automatic checking tools. Although kernel developers strive to review all patch proposals, they struggle to keep up with the increasing volume of submissions and depend significantly on a few developers for these reviews.
To help scale the patch review process, we introduce FLINT, a patch validation system framework that synthesizes insights from past discussions among developers and automatically analyzes patch proposals for compliance. FLINT employs a rule-based analysis informed by past discussions among developers and an LLM that does not require training or fine-tuning on new data, and can continuously improve with minimum human effort. FLINT uses a multi-stage approach to efficiently distill the essential information from past discussions. Later, when a patch proposal needs review, FLINT retrieves the relevant validation rules for validation and generates a reference-backed report that developers can easily interpret and validate. FLINT targets bugs that traditional tools find hard to detect, ranging from maintainability issues, e.g., design choices and naming conventions, to complex concurrency issues, e.g., deadlocks and data races. FLINT detected 2 new issues in Linux v6.18 development cycle and 7 issues in previous versions. FLINT achieves 21% and 14% of higher ground-truth coverage on concurrency bugs than the baseline with LLM only. Moreover, FLINT achieves a 35% false positive rate, which is lower than the baseline. 

---
# Large Language Models as Optimization Controllers: Adaptive Continuation for SIMP Topology Optimization 

**Authors**: Shaoliang Yang, Jun Wang, Yunsheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25099)  

**Abstract**: We present a framework in which a large language model (LLM) acts as an online adaptive controller for SIMP topology optimization, replacing conventional fixed-schedule continuation with real-time, state-conditioned parameter decisions. At every $k$-th iteration, the LLM receives a structured observation$-$current compliance, grayness index, stagnation counter, checkerboard measure, volume fraction, and budget consumption$-$and outputs numerical values for the penalization exponent $p$, projection sharpness $\beta$, filter radius $r_{\min}$, and move limit $\delta$ via a Direct Numeric Control interface. A hard grayness gate prevents premature binarization, and a meta-optimization loop uses a second LLM pass to tune the agent's call frequency and gate threshold across runs. We benchmark the agent against four baselines$-$fixed (no-continuation), standard three-field continuation, an expert heuristic, and a schedule-only ablation$-$on three 2-D problems (cantilever, MBB beam, L-bracket) at $120\!\times\!60$ resolution and two 3-D problems (cantilever, MBB beam) at $40\!\times\!20\!\times\!10$ resolution, all run for 300 iterations. A standardized 40-iteration sharpening tail is applied from the best valid snapshot so that compliance differences reflect only the exploration phase. The LLM agent achieves the lowest final compliance on every benchmark: $-5.7\%$ to $-18.1\%$ relative to the fixed baseline, with all solutions fully binary. The schedule-only ablation underperforms the fixed baseline on two of three problems, confirming that the LLM's real-time intervention$-$not the schedule geometry$-$drives the gain. Code and reproduction scripts will be released upon publication. 

---
# Experiential Reflective Learning for Self-Improving LLM Agents 

**Authors**: Marc-Antoine Allard, Arnaud Teinturier, Victor Xing, Gautier Viaud  

**Link**: [PDF](https://arxiv.org/pdf/2603.24639)  

**Abstract**: Recent advances in large language models (LLMs) have enabled the development of autonomous agents capable of complex reasoning and multi-step problem solving. However, these agents struggle to adapt to specialized environments and do not leverage past interactions, approaching each new task from scratch regardless of their accumulated experience. We introduce Experiential Reflective Learning (ERL), a simple self-improvement framework that enables rapid environment adaptation through experiential learning. ERL reflects on task trajectories and outcomes to generate heuristics, capturing actionable lessons that transfer across tasks. At test time, relevant heuristics are retrieved based on the current task and injected into the agent's context to guide execution. On the Gaia2 benchmark, ERL improves success rate by 7.8% over a ReAct baseline, with large gains in task completion reliability, and outperforms prior experiential learning methods. Through systematic ablations, we find that selective retrieval is essential and that heuristics provide more transferable abstractions than few-shot trajectory prompting. These results demonstrate that reflecting on single-attempt experiences to extract transferable heuristics enables effective agent self-improvement. 

---
# Scalable Object Relation Encoding for Better 3D Spatial Reasoning in Large Language Models 

**Authors**: Shengli Zhou, Minghang Zheng, Feng Zheng, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24721)  

**Abstract**: Spatial reasoning focuses on locating target objects based on spatial relations in 3D scenes, which plays a crucial role in developing intelligent embodied agents. Due to the limited availability of 3D scene-language paired data, it is challenging to train models with strong reasoning ability from scratch. Previous approaches have attempted to inject 3D scene representations into the input space of Large Language Models (LLMs) and leverage the pretrained comprehension and reasoning abilities for spatial reasoning. However, models encoding absolute positions struggle to extract spatial relations from prematurely fused features, while methods explicitly encoding all spatial relations (which is quadratic in the number of objects) as input tokens suffer from poor scalability. To address these limitations, we propose QuatRoPE, a novel positional embedding method with an input length that is linear to the number of objects, and explicitly calculates pairwise spatial relations through the dot product in attention layers. QuatRoPE's holistic vector encoding of 3D coordinates guarantees a high degree of spatial consistency, maintaining fidelity to the scene's geometric integrity. Additionally, we introduce the Isolated Gated RoPE Extension (IGRE), which effectively limits QuatRoPE's influence to object-related tokens, thereby minimizing interference with the LLM's existing positional embeddings and maintaining the LLM's original capabilities. Extensive experiments demonstrate the effectiveness of our approaches. The code and data are available at this https URL. 

---
# Malicious LLM-Based Conversational AI Makes Users Reveal Personal Information 

**Authors**: Xiao Zhan, Juan Carlos Carrillo, William Seymour, Jose Such  

**Link**: [PDF](https://arxiv.org/pdf/2506.11680)  

**Abstract**: LLM-based Conversational AIs (CAIs), also known as GenAI chatbots, like ChatGPT, are increasingly used across various domains, but they pose privacy risks, as users may disclose personal information during their conversations with CAIs. Recent research has demonstrated that LLM-based CAIs could be used for malicious purposes. However, a novel and particularly concerning type of malicious LLM application remains unexplored: an LLM-based CAI that is deliberately designed to extract personal information from users.
In this paper, we report on the malicious LLM-based CAIs that we created based on system prompts that used different strategies to encourage disclosures of personal information from users. We systematically investigate CAIs' ability to extract personal information from users during conversations by conducting a randomized-controlled trial with 502 participants. We assess the effectiveness of different malicious and benign CAIs to extract personal information from participants, and we analyze participants' perceptions after their interactions with the CAIs. Our findings reveal that malicious CAIs extract significantly more personal information than benign CAIs, with strategies based on the social nature of privacy being the most effective while minimizing perceived risks. This study underscores the privacy threats posed by this novel type of malicious LLM-based CAIs and provides actionable recommendations to guide future research and practice. 

---
# Sketch2Simulation: Automating Flowsheet Generation via Multi Agent Large Language Models 

**Authors**: Abdullah Bahamdan, Emma Pajak, John D. Hedengren, Antonio del Rio Chanona  

**Link**: [PDF](https://arxiv.org/pdf/2603.24629)  

**Abstract**: Converting process sketches into executable simulation models remains a major bottleneck in process systems engineering, requiring substantial manual effort and simulator-specific expertise. Recent advances in generative AI have improved both engineering-diagram interpretation and LLM-assisted flowsheet generation, but these remain largely disconnected: diagram-understanding methods often stop at extracted graphs, while text-to-simulation workflows assume structured inputs rather than raw visual artifacts. To bridge this gap, we present an end-to-end multi-agent large language model system that converts process diagrams directly into executable Aspen HYSYS flowsheets. The framework decomposes the task into three coordinated layers: diagram parsing and interpretation, simulation model synthesis, and multi-level validation. Specialized agents handle visual interpretation, graph-based intermediate representation construction, code generation for the HYSYS COM interface, execution, and structural verification. We evaluate the framework on four chemical engineering case studies of increasing complexity, from a simple desalting process to an industrial aromatic production flowsheet with multiple recycle loops. The system produces executable HYSYS models in all cases, achieving complete structural fidelity on the two simpler cases and strong performance on the more complex ones, with connection consistency above 0.93 and stream consistency above 0.96. These results demonstrate a viable end-to-end sketch-to-simulation workflow while highlighting remaining challenges in dense recycle structures, implicit diagram semantics, and simulator-interface constraints. 

---
# AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation 

**Authors**: Zhihui Yao, Hengran Zhang, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2603.25092)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) with external knowledge but remains vulnerable to low-authority sources that can propagate misinformation. We investigate whether LLMs can perceive information authority - a capability extending beyond semantic understanding. To address this, we introduce AuthorityBench, a comprehensive benchmark for evaluating LLM authority perception comprising three datasets: DomainAuth (10K web domains with PageRank-based authority), EntityAuth (22K entities with popularity-based authority), and RAGAuth (120 queries with documents of varying authority for downstream evaluation). We evaluate five LLMs using three judging methods (PointJudge, PairJudge, ListJudge) across multiple output formats. Results show that ListJudge and PairJudge with PointScore output achieve the strongest correlation with ground-truth authority, while ListJudge offers optimal cost-effectiveness. Notably, incorporating webpage text consistently degrades judgment performance, suggesting authority is distinct from textual style. Downstream experiments on RAG demonstrate that authority-guided filtering largely improves answer accuracy, validating the practical importance of authority perception for reliable knowledge retrieval. Code and benchmark are available at: this https URL. 

---
# Adaptive Chunking: Optimizing Chunking-Method Selection for RAG 

**Authors**: Paulo Roberto de Moura Júnior, Jean Lelong, Annabelle Blangero  

**Link**: [PDF](https://arxiv.org/pdf/2603.25333)  

**Abstract**: The effectiveness of Retrieval-Augmented Generation (RAG) is highly dependent on how documents are chunked, that is, segmented into smaller units for indexing and retrieval. Yet, commonly used "one-size-fits-all" approaches often fail to capture the nuanced structure and semantics of diverse texts. Despite its central role, chunking lacks a dedicated evaluation framework, making it difficult to assess and compare strategies independently of downstream performance. We challenge this paradigm by introducing Adaptive Chunking, a framework that selects the most suitable chunking strategy for each document based on a set of five novel intrinsic, document-based metrics: References Completeness (RC), Intrachunk Cohesion (ICC), Document Contextual Coherence (DCC), Block Integrity (BI), and Size Compliance (SC), which directly assess chunking quality across key dimensions. To support this framework, we also introduce two new chunkers, an LLM-regex splitter and a split-then-merge recursive splitter, alongside targeted post-processing techniques. On a diverse corpus spanning legal, technical, and social science domains, our metric-guided adaptive method significantly improves downstream RAG performance. Without changing models or prompts, our framework increases RAG outcomes, raising answers correctness to 72% (from 62-64%) and increasing the number of successfully answered questions by over 30% (65 vs. 49). These results demonstrate that adaptive, document-aware chunking, guided by a complementary suite of intrinsic metrics, offers a practical and effective path to more robust RAG systems. Code available at this https URL. 

---
