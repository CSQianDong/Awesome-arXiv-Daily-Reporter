# Real-World Summarization: When Evaluation Reaches Its Limits 

**Authors**: Patrícia Schmidtová, Ondřej Dušek, Saad Mahamood  

**Link**: [PDF](https://arxiv.org/pdf/2507.11508)  

**Abstract**: We examine evaluation of faithfulness to input data in the context of hotel highlights: brief LLM-generated summaries that capture unique features of accommodations. Through human evaluation campaigns involving categorical error assessment and span-level annotation, we compare traditional metrics, trainable methods, and LLM-as-a-judge approaches. Our findings reveal that simpler metrics like word overlap correlate surprisingly well with human judgments (Spearman correlation rank of 0.63), often outperforming more complex methods when applied to out-of-domain data. We further demonstrate that while LLMs can generate high-quality highlights, they prove unreliable for evaluation as they tend to severely under- or over-annotate. Our analysis of real-world business impacts shows incorrect and non-checkable information pose the greatest risks. We also highlight challenges in crowdsourced evaluations. 

---
# HKGAI-V1: Towards Regional Sovereign Large Language Model for Hong Kong 

**Authors**: Sirui Han, Junqi Zhu, Ruiyuan Zhang, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.11502)  

**Abstract**: This paper presents the development of HKGAI-V1, a foundational sovereign large language model (LLM), developed as part of an initiative to establish value-aligned AI infrastructure specifically tailored for Hong Kong. Addressing the region's unique multilingual environment (Cantonese, Mandarin, and English), its distinct socio-legal context under the "one country, two systems" framework, and specific local cultural and value considerations, the model is built upon the DeepSeek architecture and systematically aligned with regional norms through a multifaceted full parameter fine-tuning process. It is further integrated with a retrieval-augmented generation (RAG) system to ensure timely and factually grounded information access. The core contribution lies in the design and implementation of a comprehensive, region-specific AI alignment and safety framework, demonstrated through two key achievements: 1) The successful development of HKGAI-V1 itself - which outper-forms general-purpose models in handling Hong Kong-specific culturally sensitive queries, and embodies a "governance-embedded" approach to digital sovereignty - empowers Hong Kong to exercise control over AI applications in critical sectors including public services, legal systems, and edu-cation. 2) The development of the proprietary Adversarial HK Value Benchmark, a rigorous tool for evaluating model alignment with local ethical and legal stand-ards under challenging conditions. By documenting these achievements, the paper provides not only a technological artifact but also a replicable blueprint for developing advanced, regionally focused AI systems deeply rooted in their local identities. 

---
# Reasoning Strategies in Large Language Models: Can They Follow, Prefer, and Optimize? 

**Authors**: Yanjian Zhang, Guillaume Wisniewski, Nadi Tomeh, Thierry Charnois  

**Link**: [PDF](https://arxiv.org/pdf/2507.11423)  

**Abstract**: Human reasoning involves different strategies, each suited to specific problems. Prior work shows that large language model (LLMs) tend to favor a single reasoning strategy, potentially limiting their effectiveness in diverse reasoning challenges. In this work, we investigate whether prompting can control LLMs reasoning strategies and assess its impact on logical problem-solving. While our experiments show that no single strategy consistently improves accuracy, performance could be enhanced if models could adaptively choose the optimal strategy. We propose methods to guide LLMs in strategy selection, highlighting new ways to refine their reasoning abilities. 

---
# Seq vs Seq: An Open Suite of Paired Encoders and Decoders 

**Authors**: Orion Weller, Kathryn Ricci, Marc Marone, Antoine Chaffin, Dawn Lawrie, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2507.11412)  

**Abstract**: The large language model (LLM) community focuses almost exclusively on decoder-only language models, since they are easier to use for text generation. However, a large subset of the community still uses encoder-only models for tasks such as classification or retrieval. Previous work has attempted to compare these architectures, but is forced to make comparisons with models that have different numbers of parameters, training techniques, and datasets. We introduce the SOTA open-data Ettin suite of models: paired encoder-only and decoder-only models ranging from 17 million parameters to 1 billion, trained on up to 2 trillion tokens. Using the same recipe for both encoder-only and decoder-only models produces SOTA recipes in both categories for their respective sizes, beating ModernBERT as an encoder and Llama 3.2 and SmolLM2 as decoders. Like previous work, we find that encoder-only models excel at classification and retrieval tasks while decoders excel at generative tasks. However, we show that adapting a decoder model to encoder tasks (and vice versa) through continued training is subpar compared to using only the reverse objective (i.e. a 400M encoder outperforms a 1B decoder on MNLI, and vice versa for generative tasks). We open-source all artifacts of this study including training data, training order segmented by checkpoint, and 200+ checkpoints to allow future work to analyze or extend all aspects of training. 

---
# KisMATH: Do LLMs Have Knowledge of Implicit Structures in Mathematical Reasoning? 

**Authors**: Soumadeep Saha, Akshay Chaturvedi, Saptarshi Saha, Utpal Garain, Nicholas Asher  

**Link**: [PDF](https://arxiv.org/pdf/2507.11408)  

**Abstract**: Chain-of-thought traces have been shown to improve performance of large language models in a plethora of reasoning tasks, yet there is no consensus on the mechanism through which this performance boost is achieved. To shed more light on this, we introduce Causal CoT Graphs (CCGs), which are directed acyclic graphs automatically extracted from reasoning traces that model fine-grained causal dependencies in the language model output. A collection of $1671$ mathematical reasoning problems from MATH500, GSM8K and AIME, and their associated CCGs are compiled into our dataset -- \textbf{KisMATH}. Our detailed empirical analysis with 15 open-weight LLMs shows that (i) reasoning nodes in the CCG are mediators for the final answer, a condition necessary for reasoning; and (ii) LLMs emphasise reasoning paths given by the CCG, indicating that models internally realise structures akin to our graphs. KisMATH enables controlled, graph-aligned interventions and opens up avenues for further investigation into the role of chain-of-thought in LLM reasoning. 

---
# EXAONE 4.0: Unified Large Language Models Integrating Non-reasoning and Reasoning Modes 

**Authors**: LG AI Research, Kyunghoon Bae, Eunbi Choi, Kibong Choi, Stanley Jungkyu Choi, Yemuk Choi, Kyubeen Han, Seokhee Hong, Junwon Hwang, Taewan Hwang, Joonwon Jang, Hyojin Jeon, Kijeong Jeon, Gerrard Jeongwon Jo, Hyunjik Jo, Jiyeon Jung, Euisoon Kim, Hyosang Kim, Jihoon Kim, Joonkee Kim, Seonghwan Kim, Soyeon Kim, Sunkyoung Kim, Yireun Kim, Yongil Kim, Youchul Kim, Edward Hwayoung Lee, Gwangho Lee, Haeju Lee, Honglak Lee, Jinsik Lee, Kyungmin Lee, Sangha Park, Young Min Paik, Yongmin Park, Youngyong Park, Sanghyun Seo, Sihoon Yang, Heuiyeen Yeen, Sihyuk Yi, Hyeongu Yun  

**Link**: [PDF](https://arxiv.org/pdf/2507.11407)  

**Abstract**: This technical report introduces EXAONE 4.0, which integrates a Non-reasoning mode and a Reasoning mode to achieve both the excellent usability of EXAONE 3.5 and the advanced reasoning abilities of EXAONE Deep. To pave the way for the agentic AI era, EXAONE 4.0 incorporates essential features such as agentic tool use, and its multilingual capabilities are extended to support Spanish in addition to English and Korean. The EXAONE 4.0 model series consists of two sizes: a mid-size 32B model optimized for high performance, and a small-size 1.2B model designed for on-device applications. The EXAONE 4.0 demonstrates superior performance compared to open-weight models in its class and remains competitive even against frontier-class models. The models are publicly available for research purposes and can be easily downloaded via this https URL. 

---
# DCR: Quantifying Data Contamination in LLMs Evaluation 

**Authors**: Cheng Xu, Nan Yan, Shuhao Guan, Changhong Jin, Yuke Mei, Yibing Guo, M-Tahar Kechadi  

**Link**: [PDF](https://arxiv.org/pdf/2507.11405)  

**Abstract**: The rapid advancement of large language models (LLMs) has heightened concerns about benchmark data contamination (BDC), where models inadvertently memorize evaluation data, inflating performance metrics and undermining genuine generalization assessment. This paper introduces the Data Contamination Risk (DCR) framework, a lightweight, interpretable pipeline designed to detect and quantify BDC across four granular levels: semantic, informational, data, and label. By synthesizing contamination scores via a fuzzy inference system, DCR produces a unified DCR Factor that adjusts raw accuracy to reflect contamination-aware performance. Validated on 9 LLMs (0.5B-72B) across sentiment analysis, fake news detection, and arithmetic reasoning tasks, the DCR framework reliably diagnoses contamination severity and with accuracy adjusted using the DCR Factor to within 4% average error across the three benchmarks compared to the uncontaminated baseline. Emphasizing computational efficiency and transparency, DCR provides a practical tool for integrating contamination assessment into routine evaluations, fostering fairer comparisons and enhancing the credibility of LLM benchmarking practices. 

---
# Addressing Data Imbalance in Transformer-Based Multi-Label Emotion Detection with Weighted Loss 

**Authors**: Xia Cui  

**Link**: [PDF](https://arxiv.org/pdf/2507.11384)  

**Abstract**: This paper explores the application of a simple weighted loss function to Transformer-based models for multi-label emotion detection in SemEval-2025 Shared Task 11. Our approach addresses data imbalance by dynamically adjusting class weights, thereby enhancing performance on minority emotion classes without the computational burden of traditional resampling methods. We evaluate BERT, RoBERTa, and BART on the BRIGHTER dataset, using evaluation metrics such as Micro F1, Macro F1, ROC-AUC, Accuracy, and Jaccard similarity coefficients. The results demonstrate that the weighted loss function improves performance on high-frequency emotion classes but shows limited impact on minority classes. These findings underscore both the effectiveness and the challenges of applying this approach to imbalanced multi-label emotion detection. 

---
# What is the Best Process Model Representation? A Comparative Analysis for Process Modeling with Large Language Models 

**Authors**: Alexis Brissard, Frédéric Cuppens, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2507.11356)  

**Abstract**: Large Language Models (LLMs) are increasingly applied for Process Modeling (PMo) tasks such as Process Model Generation (PMG). To support these tasks, researchers have introduced a variety of Process Model Representations (PMRs) that serve as model abstractions or generation targets. However, these PMRs differ widely in structure, complexity, and usability, and have never been systematically compared. Moreover, recent PMG approaches rely on distinct evaluation strategies and generation techniques, making comparison difficult. This paper presents the first empirical study that evaluates multiple PMRs in the context of PMo with LLMs. We introduce the PMo Dataset, a new dataset containing 55 process descriptions paired with models in nine different PMRs. We evaluate PMRs along two dimensions: suitability for LLM-based PMo and performance on PMG. \textit{Mermaid} achieves the highest overall score across six PMo criteria, whereas \textit{BPMN text} delivers the best PMG results in terms of process element similarity. 

---
# Automated Novelty Evaluation of Academic Paper: A Collaborative Approach Integrating Human and Large Language Model Knowledge 

**Authors**: Wenqing Wu, Chengzhi Zhang, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11330)  

**Abstract**: Novelty is a crucial criterion in the peer review process for evaluating academic papers. Traditionally, it's judged by experts or measure by unique reference combinations. Both methods have limitations: experts have limited knowledge, and the effectiveness of the combination method is uncertain. Moreover, it's unclear if unique citations truly measure novelty. The large language model (LLM) possesses a wealth of knowledge, while human experts possess judgment abilities that the LLM does not possess. Therefore, our research integrates the knowledge and abilities of LLM and human experts to address the limitations of novelty assessment. The most common novelty in academic papers is the introduction of new methods. In this paper, we propose leveraging human knowledge and LLM to assist pretrained language models (PLMs, e.g. BERT etc.) in predicting the method novelty of papers. Specifically, we extract sentences related to the novelty of the academic paper from peer review reports and use LLM to summarize the methodology section of the academic paper, which are then used to fine-tune PLMs. In addition, we have designed a text-guided fusion module with novel Sparse-Attention to better integrate human and LLM knowledge. We compared the method we proposed with a large number of baselines. Extensive experiments demonstrate that our method achieves superior performance. 

---
# Internal Value Alignment in Large Language Models through Controlled Value Vector Activation 

**Authors**: Haoran Jin, Meng Li, Xiting Wang, Zhihao Xu, Minlie Huang, Yantao Jia, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2507.11316)  

**Abstract**: Aligning Large Language Models (LLMs) with human values has attracted increasing attention since it provides clarity, transparency, and the ability to adapt to evolving scenarios. In this paper, we introduce a Controlled Value Vector Activation (ConVA) method that directly aligns the internal values of LLMs by interpreting how a value is encoded in their latent representations and modifies relevant activations to ensure consistent values in LLMs. To ensure an accurate and unbiased interpretation, we propose a context-controlled value vector identification method. To consistently control values without sacrificing model performance, we introduce a gated value vector activation method for effective and minimum degree of value control. Experiments show that our method achieves the highest control success rate across 10 basic values without hurting LLM performance and fluency, and ensures target values even with opposite and potentially malicious input prompts. Source code and data are available at~ this https URL. 

---
# Dr.Copilot: A Multi-Agent Prompt Optimized Assistant for Improving Patient-Doctor Communication in Romanian 

**Authors**: Andrei Niculae, Adrian Cosma, Cosmin Dumitrache, Emilian Rǎdoi  

**Link**: [PDF](https://arxiv.org/pdf/2507.11299)  

**Abstract**: Text-based telemedicine has become increasingly common, yet the quality of medical advice in doctor-patient interactions is often judged more on how advice is communicated rather than its clinical accuracy. To address this, we introduce this http URL , a multi-agent large language model (LLM) system that supports Romanian-speaking doctors by evaluating and enhancing the presentation quality of their written responses. Rather than assessing medical correctness, this http URL provides feedback along 17 interpretable axes. The system comprises of three LLM agents with prompts automatically optimized via DSPy. Designed with low-resource Romanian data and deployed using open-weight models, it delivers real-time specific feedback to doctors within a telemedicine platform. Empirical evaluations and live deployment with 41 doctors show measurable improvements in user reviews and response quality, marking one of the first real-world deployments of LLMs in Romanian medical settings. 

---
# Fine-Grained Chinese Hate Speech Understanding: Span-Level Resources, Coded Term Lexicon, and Enhanced Detection Frameworks 

**Authors**: Zewen Bai, Liang Yang, Shengdi Yin, Yuanyuan Sun, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.11292)  

**Abstract**: The proliferation of hate speech has inflicted significant societal harm, with its intensity and directionality closely tied to specific targets and arguments. In recent years, numerous machine learning-based methods have been developed to detect hateful comments on online platforms automatically. However, research on Chinese hate speech detection lags behind, and interpretability studies face two major challenges: first, the scarcity of span-level fine-grained annotated datasets limits models' deep semantic understanding of hate speech; second, insufficient research on identifying and interpreting coded hate speech restricts model explainability in complex real-world scenarios. To address these, we make the following contributions: (1) We introduce the Span-level Target-Aware Toxicity Extraction dataset (STATE ToxiCN), the first span-level Chinese hate speech dataset, and evaluate the hate semantic understanding of existing models using it. (2) We conduct the first comprehensive study on Chinese coded hate terms, LLMs' ability to interpret hate semantics. (3) We propose a method to integrate an annotated lexicon into models, significantly enhancing hate speech detection performance. Our work provides valuable resources and insights to advance the interpretability of Chinese hate speech detection research. 

---
# FMC: Formalization of Natural Language Mathematical Competition Problems 

**Authors**: Jiaxuan Xie, Chengwu Liu, Ye Yuan, Siqi Li, Zhiping Xiao, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11275)  

**Abstract**: Efficient and accurate autoformalization methods, which leverage large-scale datasets of extensive natural language mathematical problems to construct formal language datasets, are key to advancing formal mathematical reasoning. In this paper, we propose an autoformalization pipeline based on large language models with error feedback, achieving a fully automatic and training-free formalization approach. Using this pipeline, we curate an Olympiad-level dataset aligning natural language problems with Lean formalizations. The dataset comprises $3,922$ mathematical problems in natural language and $9,787$ in Lean, of which $64.46\%$ were assessed as at least above-average quality, making it suitable as a benchmark for automated theorem provers. Additionally, we investigate the formalization and reasoning capabilities of various LLMs and empirically demonstrate that few-shot learning, error feedback, and increasing sampling numbers enhance the autoformalization process. Experiments of three automated theorem provers on the \dataset\ dataset also highlight its challenging nature and its value as a benchmark for formal reasoning tasks. 

---
# KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding 

**Authors**: Luohe Shi, Zuchao Li, Lefei Zhang, Guoming Liu, Baoyuan Qi, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11273)  

**Abstract**: Large language models (LLMs) based on Transformer Decoders have become the preferred choice for conversational generative AI. Despite the overall superiority of the Decoder architecture, the gradually increasing Key-Value (KV) cache during inference has emerged as a primary efficiency bottleneck, both in aspects of memory consumption and data transfer bandwidth limitations. To address these challenges, we propose a paradigm called KV-Latent. By down-sampling the Key-Value vector dimensions into a latent space, we can significantly reduce the KV Cache footprint and improve inference speed, only with a small amount of extra training, less than 1\% of pre-training takes. Besides, we enhanced the stability of Rotary Positional Embedding applied on lower-dimensional vectors by modifying its frequency sampling mechanism, avoiding noise introduced by higher frequencies while retaining position attenuation. Our experiments, including both models with Grouped Query Attention and those without, have yielded satisfactory results. Finally, we conducted comparative experiments to study the impact of separately reducing Key and Value components on model's performance. Our approach allows for the construction of more efficient language model systems, and opens the new possibility on KV Cache saving and efficient LLMs. Our code is available at this https URL. 

---
# Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages 

**Authors**: Lyzander Marciano Andrylie, Inaya Rahmanisa, Mahardika Krisna Ihsani, Alfan Farizki Wicaksono, Haryo Akbarianto Wibowo, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2507.11230)  

**Abstract**: Understanding the multilingual mechanisms of large language models (LLMs) provides insight into how they process different languages, yet this remains challenging. Existing studies often focus on individual neurons, but their polysemantic nature makes it difficult to isolate language-specific units from cross-lingual representations. To address this, we explore sparse autoencoders (SAEs) for their ability to learn monosemantic features that represent concrete and abstract concepts across languages in LLMs. While some of these features are language-independent, the presence of language-specific features remains underexplored. In this work, we introduce SAE-LAPE, a method based on feature activation probability, to identify language-specific features within the feed-forward network. We find that many such features predominantly appear in the middle to final layers of the model and are interpretable. These features influence the model's multilingual performance and language output and can be used for language identification with performance comparable to fastText along with more interpretability. Our code is available at this https URL . 

---
# An Agentic Flow for Finite State Machine Extraction using Prompt Chaining 

**Authors**: Fares Wael, Youssef Maklad, Ali Hamdi, Wael Elsersy  

**Link**: [PDF](https://arxiv.org/pdf/2507.11222)  

**Abstract**: Finite-State Machines (FSMs) are critical for modeling the operational logic of network protocols, enabling verification, analysis, and vulnerability discovery. However, existing FSM extraction techniques face limitations such as scalability, incomplete coverage, and ambiguity in natural language specifications. In this paper, we propose FlowFSM, a novel agentic framework that leverages Large Language Models (LLMs) combined with prompt chaining and chain-of-thought reasoning to extract accurate FSMs from raw RFC documents. FlowFSM systematically processes protocol specifications, identifies state transitions, and constructs structured rule-books by chaining agent outputs. Experimental evaluation across FTP and RTSP protocols demonstrates that FlowFSM achieves high extraction precision while minimizing hallucinated transitions, showing promising results. Our findings highlight the potential of agent-based LLM systems in the advancement of protocol analysis and FSM inference for cybersecurity and reverse engineering applications. 

---
# EsBBQ and CaBBQ: The Spanish and Catalan Bias Benchmarks for Question Answering 

**Authors**: Valle Ruiz-Fernández, Mario Mina, Júlia Falcão, Luis Vasquez-Reina, Anna Sallés, Aitor Gonzalez-Agirre, Olatz Perez-de-Viñaspre  

**Link**: [PDF](https://arxiv.org/pdf/2507.11216)  

**Abstract**: Previous literature has largely shown that Large Language Models (LLMs) perpetuate social biases learnt from their pre-training data. Given the notable lack of resources for social bias evaluation in languages other than English, and for social contexts outside of the United States, this paper introduces the Spanish and the Catalan Bias Benchmarks for Question Answering (EsBBQ and CaBBQ). Based on the original BBQ, these two parallel datasets are designed to assess social bias across 10 categories using a multiple-choice QA setting, now adapted to the Spanish and Catalan languages and to the social context of Spain. We report evaluation results on different LLMs, factoring in model family, size and variant. Our results show that models tend to fail to choose the correct answer in ambiguous scenarios, and that high QA accuracy often correlates with greater reliance on social biases. 

---
# Temperature and Persona Shape LLM Agent Consensus With Minimal Accuracy Gains in Qualitative Coding 

**Authors**: Conrad Borchers, Bahar Shahrokhian, Francesco Balzan, Elham Tajik, Sreecharan Sankaranarayanan, Sebastian Simon  

**Link**: [PDF](https://arxiv.org/pdf/2507.11198)  

**Abstract**: Large Language Models (LLMs) enable new possibilities for qualitative research at scale, including coding and data annotation. While multi-agent systems (MAS) can emulate human coding workflows, their benefits over single-agent coding remain poorly understood. We conducted an experimental study of how agent persona and temperature shape consensus-building and coding accuracy of dialog segments based on a codebook with 8 codes. Our open-source MAS mirrors deductive human coding through structured agent discussion and consensus arbitration. Using six open-source LLMs (with 3 to 32 billion parameters) and 18 experimental configurations, we analyze over 77,000 coding decisions against a gold-standard dataset of human-annotated transcripts from online math tutoring sessions. Temperature significantly impacted whether and when consensus was reached across all six LLMs. MAS with multiple personas (including neutral, assertive, or empathetic), significantly delayed consensus in four out of six LLMs compared to uniform personas. In three of those LLMs, higher temperatures significantly diminished the effects of multiple personas on consensus. However, neither temperature nor persona pairing lead to robust improvements in coding accuracy. Single agents matched or outperformed MAS consensus in most conditions. Only one model (OpenHermesV2:7B) and code category showed above-chance gains from MAS deliberation when temperature was 0.5 or lower and especially when the agents included at least one assertive persona. Qualitative analysis of MAS collaboration for these configurations suggests that MAS may nonetheless aid in narrowing ambiguous code applications that could improve codebooks and human-AI coding. We contribute new insight into the limits of LLM-based qualitative methods, challenging the notion that diverse MAS personas lead to better outcomes. We open-source our MAS and experimentation code. 

---
# What Should LLMs Forget? Quantifying Personal Data in LLMs for Right-to-Be-Forgotten Requests 

**Authors**: Dimitri Staufer  

**Link**: [PDF](https://arxiv.org/pdf/2507.11128)  

**Abstract**: Large Language Models (LLMs) can memorize and reveal personal information, raising concerns regarding compliance with the EU's GDPR, particularly the Right to Be Forgotten (RTBF). Existing machine unlearning methods assume the data to forget is already known but do not address how to identify which individual-fact associations are stored in the model. Privacy auditing techniques typically operate at the population level or target a small set of identifiers, limiting applicability to individual-level data inquiries. We introduce WikiMem, a dataset of over 5,000 natural language canaries covering 243 human-related properties from Wikidata, and a model-agnostic metric to quantify human-fact associations in LLMs. Our approach ranks ground-truth values against counterfactuals using calibrated negative log-likelihood across paraphrased prompts. We evaluate 200 individuals across 15 LLMs (410M-70B parameters), showing that memorization correlates with subject web presence and model scale. We provide a foundation for identifying memorized personal data in LLMs at the individual level, enabling the dynamic construction of forget sets for machine unlearning and RTBF requests. 

---
# MSA at ImageCLEF 2025 Multimodal Reasoning: Multilingual Multimodal Reasoning With Ensemble Vision Language Models 

**Authors**: Seif Ahmed, Mohamed T. Younes, Abdelrahman Moustafa, Abdelrahman Allam, Hamza Moustafa  

**Link**: [PDF](https://arxiv.org/pdf/2507.11114)  

**Abstract**: We present a robust ensemble-based system for multilingual multimodal reasoning, designed for the ImageCLEF 2025 EXAMS V challenge. Our approach integrates Gemini 2.5 Flash for visual description, Gemini 1.5 Pro for caption refinement and consistency checks, and Gemini 2.5 Pro as a reasoner which handles final answer selection, all coordinated through carefully engineered few-shot and zero-shot prompts. We conducted an extensive ablation study, training several large language models (Gemini 2.5 Flash, Phi 4, Gemma 3, Mistral) on an English dataset and its multilingual augmented version. Additionally, we evaluated Gemini 2.5 Flash in a zero-shot setting for comparison and found it to substantially outperform the trained models. Prompt design also proved critical: enforcing concise, language-normalized formats and prohibiting explanatory text boosted model accuracy on the English validation set from 55.9% to 61.7%. On the official leaderboard, our system (Team MSA) achieved first place overall in the multilingual track with 81.4% accuracy, and led 11 out of 13 individual language tracks, with top results such as 95.07% for Croatian and 92.12% for Italian. These findings highlight that lightweight OCR-VLM ensembles, when paired with precise prompt strategies and cross-lingual augmentation, can outperform heavier end-to-end models in high-stakes, multilingual educational settings. 

---
# Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs 

**Authors**: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2507.11112)  

**Abstract**: Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning. 

---
# The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs 

**Authors**: Zichen Wen, Jiashu Qu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11097)  

**Abstract**: Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at this https URL. 

---
# Beyond Traditional Algorithms: Leveraging LLMs for Accurate Cross-Border Entity Identification 

**Authors**: Andres Azqueta-Gavaldón, Joaquin Ramos Cosgrove  

**Link**: [PDF](https://arxiv.org/pdf/2507.11086)  

**Abstract**: The growing prevalence of cross-border financial activities in global markets has underscored the necessity of accurately identifying and classifying foreign entities. This practice is essential within the Spanish financial system for ensuring robust risk management, regulatory adherence, and the prevention of financial misconduct. This process involves a labor-intensive entity-matching task, where entities need to be validated against available reference sources. Challenges arise from linguistic variations, special characters, outdated names, and changes in legal forms, complicating traditional matching algorithms like Jaccard, cosine, and Levenshtein distances. These methods struggle with contextual nuances and semantic relationships, leading to mismatches. To address these limitations, we explore Large Language Models (LLMs) as a flexible alternative. LLMs leverage extensive training to interpret context, handle abbreviations, and adapt to legal transitions. We evaluate traditional methods, Hugging Face-based LLMs, and interface-based LLMs (e.g., Microsoft Copilot, Alibaba's Qwen 2.5) using a dataset of 65 Portuguese company cases. Results show traditional methods achieve accuracies over 92% but suffer high false positive rates (20-40%). Interface-based LLMs outperform, achieving accuracies above 93%, F1 scores exceeding 96%, and lower false positives (40-80%). 

---
# Social Media Sentiments Analysis on the July Revolution in Bangladesh: A Hybrid Transformer Based Machine Learning Approach 

**Authors**: Md. Sabbir Hossen, Md. Saiduzzaman, Pabon Shaha  

**Link**: [PDF](https://arxiv.org/pdf/2507.11084)  

**Abstract**: The July Revolution in Bangladesh marked a significant student-led mass uprising, uniting people across the nation to demand justice, accountability, and systemic reform. Social media platforms played a pivotal role in amplifying public sentiment and shaping discourse during this historic mass uprising. In this study, we present a hybrid transformer-based sentiment analysis framework to decode public opinion expressed in social media comments during and after the revolution. We used a brand new dataset of 4,200 Bangla comments collected from social media. The framework employs advanced transformer-based feature extraction techniques, including BanglaBERT, mBERT, XLM-RoBERTa, and the proposed hybrid XMB-BERT, to capture nuanced patterns in textual data. Principle Component Analysis (PCA) were utilized for dimensionality reduction to enhance computational efficiency. We explored eleven traditional and advanced machine learning classifiers for identifying sentiments. The proposed hybrid XMB-BERT with the voting classifier achieved an exceptional accuracy of 83.7% and outperform other model classifier combinations. This study underscores the potential of machine learning techniques to analyze social sentiment in low-resource languages like Bangla. 

---
# LLM-Augmented Symptom Analysis for Cardiovascular Disease Risk Prediction: A Clinical NLP 

**Authors**: Haowei Yang, Ziyu Shen, Junli Shao, Luyao Men, Xinyue Han, Jing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.11052)  

**Abstract**: Timely identification and accurate risk stratification of cardiovascular disease (CVD) remain essential for reducing global mortality. While existing prediction models primarily leverage structured data, unstructured clinical notes contain valuable early indicators. This study introduces a novel LLM-augmented clinical NLP pipeline that employs domain-adapted large language models for symptom extraction, contextual reasoning, and correlation from free-text reports. Our approach integrates cardiovascular-specific fine-tuning, prompt-based inference, and entity-aware reasoning. Evaluations on MIMIC-III and CARDIO-NLP datasets demonstrate improved performance in precision, recall, F1-score, and AUROC, with high clinical relevance (kappa = 0.82) assessed by cardiologists. Challenges such as contextual hallucination, which occurs when plausible information contracts with provided source, and temporal ambiguity, which is related with models struggling with chronological ordering of events are addressed using prompt engineering and hybrid rule-based verification. This work underscores the potential of LLMs in clinical decision support systems (CDSS), advancing early warning systems and enhancing the translation of patient narratives into actionable risk assessments. 

---
# Journalism-Guided Agentic In-Context Learning for News Stance Detection 

**Authors**: Dahyun Lee, Jonghyeon Choi, Jiyoung Han, Kunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.11049)  

**Abstract**: As online news consumption grows, personalized recommendation systems have become integral to digital journalism. However, these systems risk reinforcing filter bubbles and political polarization by failing to incorporate diverse perspectives. Stance detection -- identifying a text's position on a target -- can help mitigate this by enabling viewpoint-aware recommendations and data-driven analyses of media bias. Yet, existing stance detection research remains largely limited to short texts and high-resource languages. To address these gaps, we introduce \textsc{K-News-Stance}, the first Korean dataset for article-level stance detection, comprising 2,000 news articles with article-level and 19,650 segment-level stance annotations across 47 societal issues. We also propose \textsc{JoA-ICL}, a \textbf{Jo}urnalism-guided \textbf{A}gentic \textbf{I}n-\textbf{C}ontext \textbf{L}earning framework that employs a language model agent to predict the stances of key structural segments (e.g., leads, quotes), which are then aggregated to infer the overall article stance. Experiments show that \textsc{JoA-ICL} outperforms existing stance detection methods, highlighting the benefits of segment-level agency in capturing the overall position of long-form news articles. Two case studies further demonstrate its broader utility in promoting viewpoint diversity in news recommendations and uncovering patterns of media bias. 

---
# Team HUMANE at AVeriTeC 2025: HerO 2 for Efficient Fact Verification 

**Authors**: Yejun Yoon, Jaeyoon Jung, Seunghyun Yoon, Kunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.11004)  

**Abstract**: This paper presents HerO 2, Team HUMANE's system for the AVeriTeC shared task at the FEVER-25 workshop. HerO 2 is an enhanced version of HerO, the best-performing open-source model from the previous year's challenge. It improves evidence quality through document summarization and answer reformulation, optimizes veracity prediction via post-training quantization under computational constraints, and enhances overall system performance by integrating updated language model (LM) backbones. HerO 2 ranked second on the leaderboard while achieving the shortest runtime among the top three systems, demonstrating both high efficiency and strong potential for real-world fact verification. The code is available at this https URL. 

---
# Mario at EXIST 2025: A Simple Gateway to Effective Multilingual Sexism Detection 

**Authors**: Lin Tian, Johanne R. Trippas, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10996)  

**Abstract**: This paper presents our approach to EXIST 2025 Task 1, addressing text-based sexism detection in English and Spanish tweets through hierarchical Low-Rank Adaptation (LoRA) of Llama 3.1 8B. Our method introduces conditional adapter routing that explicitly models label dependencies across three hierarchically structured subtasks: binary sexism identification, source intention detection, and multilabel sexism categorization. Unlike conventional LoRA applications that target only attention layers, we apply adaptation to all linear transformations, enhancing the model's capacity to capture task-specific patterns. In contrast to complex data processing and ensemble approaches, we show that straightforward parameter-efficient fine-tuning achieves strong performance. We train separate LoRA adapters (rank=16, QLoRA 4-bit) for each subtask using unified multilingual training that leverages Llama 3.1's native bilingual capabilities. The method requires minimal preprocessing and uses standard supervised learning. Our multilingual training strategy eliminates the need for separate language-specific models, achieving 1.7-2.4\% F1 improvements through cross-lingual transfer. With only 1.67\% trainable parameters compared to full fine-tuning, our approach reduces training time by 75\% and model storage by 98\%, while achieving competitive performance across all subtasks (ICM-Hard: 0.6774 for binary classification, 0.4991 for intention detection, 0.6519 for multilabel categorization). 

---
# Teach Me Sign: Stepwise Prompting LLM for Sign Language Production 

**Authors**: Zhaoyi An, Rei Kawakami  

**Link**: [PDF](https://arxiv.org/pdf/2507.10972)  

**Abstract**: Large language models, with their strong reasoning ability and rich knowledge, have brought revolution to many tasks of AI, but their impact on sign language generation remains limited due to its complexity and unique rules. In this paper, we propose TEAch Me Sign (TEAM-Sign), treating sign language as another natural language. By fine-tuning an LLM, we enable it to learn the correspondence between text and sign language, and facilitate generation. Considering the differences between sign and spoken language, we employ a stepwise prompting strategy to extract the inherent sign language knowledge within the LLM, thereby supporting the learning and generation process. Experimental results on How2Sign and Phoenix14T datasets demonstrate that our approach effectively leverages both the sign language knowledge and reasoning capabilities of LLM to align the different distribution and grammatical rules between sign and spoken language. 

---
# DS@GT at eRisk 2025: From prompts to predictions, benchmarking early depression detection with conversational agent based assessments and temporal attention models 

**Authors**: Anthony Miyaguchi, David Guecha, Yuwen Chiu, Sidharth Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2507.10958)  

**Abstract**: This Working Note summarizes the participation of the DS@GT team in two eRisk 2025 challenges. For the Pilot Task on conversational depression detection with large language-models (LLMs), we adopted a prompt-engineering strategy in which diverse LLMs conducted BDI-II-based assessments and produced structured JSON outputs. Because ground-truth labels were unavailable, we evaluated cross-model agreement and internal consistency. Our prompt design methodology aligned model outputs with BDI-II criteria and enabled the analysis of conversational cues that influenced the prediction of symptoms. Our best submission, second on the official leaderboard, achieved DCHR = 0.50, ADODL = 0.89, and ASHR = 0.27. 

---
# Modeling Understanding of Story-Based Analogies Using Large Language Models 

**Authors**: Kalit Inani, Keshav Kabra, Vijay Marupudi, Sashank Varma  

**Link**: [PDF](https://arxiv.org/pdf/2507.10957)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have brought them closer to matching human cognition across a variety of tasks. How well do these models align with human performance in detecting and mapping analogies? Prior research has shown that LLMs can extract similarities from analogy problems but lack robust human-like reasoning. Building on Webb, Holyoak, and Lu (2023), the current study focused on a story-based analogical mapping task and conducted a fine-grained evaluation of LLM reasoning abilities compared to human performance. First, it explored the semantic representation of analogies in LLMs, using sentence embeddings to assess whether they capture the similarity between the source and target texts of an analogy, and the dissimilarity between the source and distractor texts. Second, it investigated the effectiveness of explicitly prompting LLMs to explain analogies. Throughout, we examine whether LLMs exhibit similar performance profiles to those observed in humans by evaluating their reasoning at the level of individual analogies, and not just at the level of overall accuracy (as prior studies have done). Our experiments include evaluating the impact of model size (8B vs. 70B parameters) and performance variation across state-of-the-art model architectures such as GPT-4 and LLaMA3. This work advances our understanding of the analogical reasoning abilities of LLMs and their potential as models of human reasoning. 

---
# HanjaBridge: Resolving Semantic Ambiguity in Korean LLMs via Hanja-Augmented Pre-Training 

**Authors**: Seungho Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.10920)  

**Abstract**: Large language models (LLMs) often show poor performance in low-resource languages like Korean, partly due to unique linguistic challenges such as homophonous Sino-Korean words that are indistinguishable in Hangul script. To address this semantic ambiguity, we propose HanjaBridge, a novel meaning-injection technique integrated into a continual pre-training (CPT) framework. Instead of deterministically mapping a word to a single Hanja (Chinese character), HanjaBridge presents the model with all possible Hanja candidates for a given homograph, encouraging the model to learn contextual disambiguation. This process is paired with token-level knowledge distillation to prevent catastrophic forgetting. Experimental results show that HanjaBridge significantly improves Korean language understanding, achieving a 21\% relative improvement on the KoBALT benchmark. Notably, by reinforcing semantic alignment between Korean and Chinese through shared Hanja, we observe a strong positive cross-lingual transfer. Furthermore, these gains persist even when Hanja augmentation is omitted at inference time, ensuring practical efficiency with no additional run-time cost. 

---
# How Stylistic Similarity Shapes Preferences in Dialogue Dataset with User and Third Party Evaluations 

**Authors**: Ikumi Numaya, Shoji Moriya, Shiki Sato, Reina Akama, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2507.10918)  

**Abstract**: Recent advancements in dialogue generation have broadened the scope of human-bot interactions, enabling not only contextually appropriate responses but also the analysis of human affect and sensitivity. While prior work has suggested that stylistic similarity between user and system may enhance user impressions, the distinction between subjective and objective similarity is often overlooked. To investigate this issue, we introduce a novel dataset that includes users' preferences, subjective stylistic similarity based on users' own perceptions, and objective stylistic similarity annotated by third party evaluators in open-domain dialogue settings. Analysis using the constructed dataset reveals a strong positive correlation between subjective stylistic similarity and user preference. Furthermore, our analysis suggests an important finding: users' subjective stylistic similarity differs from third party objective similarity. This underscores the importance of distinguishing between subjective and objective evaluations and understanding the distinct aspects each captures when analyzing the relationship between stylistic similarity and user preferences. The dataset presented in this paper is available online. 

---
# LLMs on Trial: Evaluating Judicial Fairness for Large Language Models 

**Authors**: Yiran Hu, Zongyue Xue, Haitao Li, Siyuan Zheng, Qingjing Chen, Shaochun Wang, Xihan Zhang, Ning Zheng, Yun Liu, Qingyao Ai, Yiqun Liu, Charles L.A. Clarke, Weixing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.10852)  

**Abstract**: Large Language Models (LLMs) are increasingly used in high-stakes fields where their decisions impact rights and equity. However, LLMs' judicial fairness and implications for social justice remain underexplored. When LLMs act as judges, the ability to fairly resolve judicial issues is a prerequisite to ensure their trustworthiness. Based on theories of judicial fairness, we construct a comprehensive framework to measure LLM fairness, leading to a selection of 65 labels and 161 corresponding values. Applying this framework to the judicial system, we compile an extensive dataset, JudiFair, comprising 177,100 unique case facts. To achieve robust statistical inference, we develop three evaluation metrics, inconsistency, bias, and imbalanced inaccuracy, and introduce a method to assess the overall fairness of multiple LLMs across various labels. Through experiments with 16 LLMs, we uncover pervasive inconsistency, bias, and imbalanced inaccuracy across models, underscoring severe LLM judicial unfairness. Particularly, LLMs display notably more pronounced biases on demographic labels, with slightly less bias on substance labels compared to procedure ones. Interestingly, increased inconsistency correlates with reduced biases, but more accurate predictions exacerbate biases. While we find that adjusting the temperature parameter can influence LLM fairness, model size, release date, and country of origin do not exhibit significant effects on judicial fairness. Accordingly, we introduce a publicly available toolkit containing all datasets and code, designed to support future research in evaluating and improving LLM fairness. 

---
# Testing Hypotheses from the Social Approval Theory of Online Hate: An Analysis of 110 Million Posts from Parler 

**Authors**: David M. Markowitz, Samuel Hardman Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2507.10810)  

**Abstract**: In this paper, we explored how online hate is motivated by receiving social approval from others. We specifically examined two central tenets of Walther's (2024) social approval theory of online hate: (H1a) more signals of social approval on hate messages predicts more subsequent hate messages, and (H1b) as social approval increases, hate speech messages become more extreme. Using over 110 million posts from Parler (2018-2021), we observed that the number of upvotes a person received on a hate speech post was unassociated with the amount of hate speech in their next post and posts during the next week, month, three months, and six months. Between-person effects revealed an average negative relationship between social approval and hate speech production at the post level, but this relationship was mixed at other time intervals. Social approval reinforcement mechanisms of online hate may operate differently on niche social media platforms. 

---
# Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers 

**Authors**: Yilun Zhao, Chengye Wang, Chuhan Li, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2507.10787)  

**Abstract**: This paper introduces MISS-QA, the first benchmark specifically designed to evaluate the ability of models to interpret schematic diagrams within scientific literature. MISS-QA comprises 1,500 expert-annotated examples over 465 scientific papers. In this benchmark, models are tasked with interpreting schematic diagrams that illustrate research overviews and answering corresponding information-seeking questions based on the broader context of the paper. We assess the performance of 18 frontier multimodal foundation models, including o4-mini, Gemini-2.5-Flash, and Qwen2.5-VL. We reveal a significant performance gap between these models and human experts on MISS-QA. Our analysis of model performance on unanswerable questions and our detailed error analysis further highlight the strengths and limitations of current models, offering key insights to enhance models in comprehending multimodal scientific literature. 

---
# Applying Text Embedding Models for Efficient Analysis in Labeled Property Graphs 

**Authors**: Michal Podstawski  

**Link**: [PDF](https://arxiv.org/pdf/2507.10772)  

**Abstract**: Labeled property graphs often contain rich textual attributes that can enhance analytical tasks when properly leveraged. This work explores the use of pretrained text embedding models to enable efficient semantic analysis in such graphs. By embedding textual node and edge properties, we support downstream tasks including node classification and relation prediction with improved contextual understanding. Our approach integrates language model embeddings into the graph pipeline without altering its structure, demonstrating that textual semantics can significantly enhance the accuracy and interpretability of property graph analysis. 

---
# Language Models for Adult Service Website Text Analysis 

**Authors**: Nickolas Freeman, Thanh Nguyen, Gregory Bott, Jason Parton, Collin Francel  

**Link**: [PDF](https://arxiv.org/pdf/2507.10743)  

**Abstract**: Sex trafficking refers to the use of force, fraud, or coercion to compel an individual to perform in commercial sex acts against their will. Adult service websites (ASWs) have and continue to be linked to sex trafficking, offering a platform for traffickers to advertise their victims. Thus, organizations involved in the fight against sex trafficking often use ASW data when attempting to identify potential sex trafficking victims. A critical challenge in transforming ASW data into actionable insight is text analysis. Previous research using ASW data has shown that ASW ad text is important for linking ads. However, working with this text is challenging due to its extensive use of emojis, poor grammar, and deliberate obfuscation to evade law enforcement scrutiny. We conduct a comprehensive study of language modeling approaches for this application area, including simple information retrieval methods, pre-trained transformers, and custom transformer models. We demonstrate that characteristics of ASW text data allow efficient custom transformer models to be trained with relatively small GPU resources and used efficiently for inference on consumer hardware. Our custom models outperform fine-tuned variants of well-known encoder-only transformer models, including BERT-base, RoBERTa, and ModernBERT, on accuracy, recall, F1 score, and ROC AUC. We demonstrate the use of our best-performing custom configuration on three tasks related to ASW data analysis: (i) decomposing the giant component in a graph representation of ASW data, (ii) clustering ASW ad text, and (iii) using the learned token embeddings to understand the use of emojis in the illicit context we study. The models we develop represent a significant advancement in ASW text analysis, which can be leveraged in a variety of downstream applications and research. 

---
# Emergence of Hierarchical Emotion Organization in Large Language Models 

**Authors**: Bo Zhao, Maya Okawa, Eric J. Bigelow, Rose Yu, Tomer Ullman, Ekdeep Singh Lubana, Hidenori Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2507.10599)  

**Abstract**: As large language models (LLMs) increasingly power conversational agents, understanding how they model users' emotional states is critical for ethical deployment. Inspired by emotion wheels -- a psychological framework that argues emotions organize hierarchically -- we analyze probabilistic dependencies between emotional states in model outputs. We find that LLMs naturally form hierarchical emotion trees that align with human psychological models, and larger models develop more complex hierarchies. We also uncover systematic biases in emotion recognition across socioeconomic personas, with compounding misclassifications for intersectional, underrepresented groups. Human studies reveal striking parallels, suggesting that LLMs internalize aspects of social perception. Beyond highlighting emergent emotional reasoning in LLMs, our results hint at the potential of using cognitively-grounded theories for developing better model evaluations. 

---
# PLEX: Perturbation-free Local Explanations for LLM-Based Text Classification 

**Authors**: Yogachandran Rahulamathavan, Misbah Farooq, Varuna De Silva  

**Link**: [PDF](https://arxiv.org/pdf/2507.10596)  

**Abstract**: Large Language Models (LLMs) excel in text classification, but their complexity hinders interpretability, making it difficult to understand the reasoning behind their predictions. Explainable AI (XAI) methods like LIME and SHAP offer local explanations by identifying influential words, but they rely on computationally expensive perturbations. These methods typically generate thousands of perturbed sentences and perform inferences on each, incurring a substantial computational burden, especially with LLMs. To address this, we propose \underline{P}erturbation-free \underline{L}ocal \underline{Ex}planation (PLEX), a novel method that leverages the contextual embeddings extracted from the LLM and a ``Siamese network" style neural network trained to align with feature importance scores. This one-off training eliminates the need for subsequent perturbations, enabling efficient explanations for any new sentence. We demonstrate PLEX's effectiveness on four different classification tasks (sentiment, fake news, fake COVID-19 news and depression), showing more than 92\% agreement with LIME and SHAP. Our evaluation using a ``stress test" reveals that PLEX accurately identifies influential words, leading to a similar decline in classification accuracy as observed with LIME and SHAP when these words are removed. Notably, in some cases, PLEX demonstrates superior performance in capturing the impact of key features. PLEX dramatically accelerates explanation, reducing time and computational overhead by two and four orders of magnitude, respectively. This work offers a promising solution for explainable LLM-based text classification. 

---
# Anthropomimetic Uncertainty: What Verbalized Uncertainty in Language Models is Missing 

**Authors**: Dennis Ulmer, Alexandra Lorson, Ivan Titov, Christian Hardmeier  

**Link**: [PDF](https://arxiv.org/pdf/2507.10587)  

**Abstract**: Human users increasingly rely on natural language interactions with large language models (LLMs) in order to receive help on a large variety of tasks and problems. However, the trustworthiness and perceived legitimacy of LLMs is undermined by the fact that their output is frequently stated in very confident terms, even when its accuracy is questionable. Therefore, there is a need to signal the confidence of the language model to a user in order to reap the benefits of human-machine collaboration and mitigate potential harms. Verbalized uncertainty is the expression of confidence with linguistic means, an approach that integrates perfectly into language-based interfaces. Nevertheless, most recent research in natural language processing (NLP) overlooks the nuances surrounding human uncertainty communication and the data biases that influence machine uncertainty communication. We argue for anthropomimetic uncertainty, meaning that intuitive and trustworthy uncertainty communication requires a degree of linguistic authenticity and personalization to the user, which could be achieved by emulating human communication. We present a thorough overview over the research in human uncertainty communication, survey ongoing research, and perform additional analyses to demonstrate so-far overlooked biases in verbalized uncertainty. We conclude by pointing out unique factors in human-machine communication of uncertainty and deconstruct anthropomimetic uncertainty into future research directions for NLP. 

---
# AutoRAG-LoRA: Hallucination-Triggered Knowledge Retuning via Lightweight Adapters 

**Authors**: Kaushik Dwivedi, Padmanabh Patanjali Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2507.10586)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable fluency across a range of natural language tasks, yet remain vulnerable to hallucinations - factual inaccuracies that undermine trust in real world deployment. We present AutoRAG-LoRA, a modular framework for Retrieval-Augmented Generation (RAG) that tackles hallucination in large language models through lightweight LoRA-based adapters and KL-regularized training. Our pipeline integrates automated prompt rewriting, hybrid retrieval, and low-rank adapter tuning to ground responses in retrieved evidence. A hallucination detection module, using both classifier-based and self-evaluation techniques, assigns confidence scores to generated outputs, triggering an optional feedback correction loop. This loop enforces factual alignment via contrastive KL loss and adapter fine tuning. We demonstrate that AutoRAG-LoRA significantly reduces the factual drift while preserving the efficiency and modularity of the model. 

---
# A Taxonomy for Design and Evaluation of Prompt-Based Natural Language Explanations 

**Authors**: Isar Nejadgholi, Mona Omidyeganeh, Marc-Antoine Drouin, Jonathan Boisvert  

**Link**: [PDF](https://arxiv.org/pdf/2507.10585)  

**Abstract**: Effective AI governance requires structured approaches for stakeholders to access and verify AI system behavior. With the rise of large language models, Natural Language Explanations (NLEs) are now key to articulating model behavior, which necessitates a focused examination of their characteristics and governance implications. We draw on Explainable AI (XAI) literature to create an updated XAI taxonomy, adapted to prompt-based NLEs, across three dimensions: (1) Context, including task, data, audience, and goals; (2) Generation and Presentation, covering generation methods, inputs, interactivity, outputs, and forms; and (3) Evaluation, focusing on content, presentation, and user-centered properties, as well as the setting of the evaluation. This taxonomy provides a framework for researchers, auditors, and policymakers to characterize, design, and enhance NLEs for transparent AI systems. 

---
# Transforming Sensitive Documents into Quantitative Data: An AI-Based Preprocessing Toolchain for Structured and Privacy-Conscious Analysis 

**Authors**: Anders Ledberg, Anna Thalén  

**Link**: [PDF](https://arxiv.org/pdf/2507.10582)  

**Abstract**: Unstructured text from legal, medical, and administrative sources offers a rich but underutilized resource for research in public health and the social sciences. However, large-scale analysis is hampered by two key challenges: the presence of sensitive, personally identifiable information, and significant heterogeneity in structure and language. We present a modular toolchain that prepares such text data for embedding-based analysis, relying entirely on open-weight models that run on local hardware, requiring only a workstation-level GPU and supporting privacy-sensitive research.
The toolchain employs large language model (LLM) prompting to standardize, summarize, and, when needed, translate texts to English for greater comparability. Anonymization is achieved via LLM-based redaction, supplemented with named entity recognition and rule-based methods to minimize the risk of disclosure. We demonstrate the toolchain on a corpus of 10,842 Swedish court decisions under the Care of Abusers Act (LVM), comprising over 56,000 pages. Each document is processed into an anonymized, standardized summary and transformed into a document-level embedding. Validation, including manual review, automated scanning, and predictive evaluation shows the toolchain effectively removes identifying information while retaining semantic content. As an illustrative application, we train a predictive model using embedding vectors derived from a small set of manually labeled summaries, demonstrating the toolchain's capacity for semi-automated content analysis at scale.
By enabling structured, privacy-conscious analysis of sensitive documents, our toolchain opens new possibilities for large-scale research in domains where textual data was previously inaccessible due to privacy and heterogeneity constraints. 

---
# An Offline Mobile Conversational Agent for Mental Health Support: Learning from Emotional Dialogues and Psychological Texts with Student-Centered Evaluation 

**Authors**: Vimaleswar A, Prabhu Nandan Sahu, Nilesh Kumar Sahu, Haroon R Lone  

**Link**: [PDF](https://arxiv.org/pdf/2507.10580)  

**Abstract**: Mental health plays a crucial role in the overall well-being of an individual. In recent years, digital platforms have been increasingly used to expand mental health and emotional support. However, there are persistent challenges related to limited user accessibility, internet connectivity, and data privacy, which highlight the need for an offline, smartphone-based solution. To address these challenges, we propose EmoSApp (Emotional Support App): an entirely offline, smartphone-based conversational app designed for mental health and emotional support. The system leverages Large Language Models (LLMs), specifically fine-tuned, quantized and deployed using Torchtune and Executorch for resource-constrained devices, allowing all inferences to occur on the smartphone. To equip EmoSApp with robust domain expertise, we fine-tuned the LLaMA-3.2-1B-Instruct model on our custom curated ``Knowledge dataset'' of 14,582 mental-health QA pairs, along with the multi-turn conversational data.
Through qualitative human evaluation with the student population, we demonstrate that EmoSApp has the ability to respond coherently, empathetically, maintain interactive dialogue, and provide relevant suggestions to user's mental health problems. Additionally, quantitative evaluations on nine standard commonsense and reasoning benchmarks demonstrate the efficacy of our fine-tuned, quantized model in low-resource settings. By prioritizing on-device deployment and specialized domain adaptation, EmoSApp serves as a blueprint for future innovations in portable, secure, and highly tailored AI-driven mental health solutions. 

---
# Truth Sleuth and Trend Bender: AI Agents to fact-check YouTube videos and influence opinions 

**Authors**: Logé Cécile, Ghori Rehan  

**Link**: [PDF](https://arxiv.org/pdf/2507.10577)  

**Abstract**: Misinformation poses a significant threat in today's digital world, often spreading rapidly through platforms like YouTube. This paper introduces a novel approach to combating misinformation by developing an AI-powered system that not only fact-checks claims made in YouTube videos but also actively engages users in the comment section and challenge misleading narratives. Our system comprises two main agents: Truth Sleuth and Trend Bender.
Truth Sleuth extracts claims from a YouTube video, uses a Retrieval-Augmented Generation (RAG) approach - drawing on sources like Wikipedia, Google Search, Google FactCheck - to accurately assess their veracity and generates a nuanced and comprehensive report. Through rigorous prompt engineering, Trend Bender leverages this report along with a curated corpus of relevant articles to generate insightful and persuasive comments designed to stimulate a productive debate. With a carefully set up self-evaluation loop, this agent is able to iteratively improve its style and refine its output.
We demonstrate the system's capabilities through experiments on established benchmark datasets and a real-world deployment on YouTube, showcasing its potential to engage users and potentially influence perspectives. Our findings highlight the high accuracy of our fact-checking agent, and confirm the potential of AI-driven interventions in combating misinformation and fostering a more informed online space. 

---
# AirLLM: Diffusion Policy-based Adaptive LoRA for Remote Fine-Tuning of LLM over the Air 

**Authors**: Shiyi Yang, Xiaoxue Yu, Rongpeng Li, Jianhang Zhu, Zhifeng Zhao, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11515)  

**Abstract**: Operating Large Language Models (LLMs) on edge devices is increasingly challenged by limited communication bandwidth and strained computational and memory costs. Thus, cloud-assisted remote fine-tuning becomes indispensable. Nevertheless, existing Low-Rank Adaptation (LoRA) approaches typically employ fixed or heuristic rank configurations, and the subsequent over-the-air transmission of all LoRA parameters could be rather inefficient. To address this limitation, we develop AirLLM, a hierarchical diffusion policy framework for communication-aware LoRA adaptation. Specifically, AirLLM models the rank configuration as a structured action vector that spans all LoRA-inserted projections. To solve the underlying high-dimensional sequential decision-making problem, a Proximal Policy Optimization (PPO) agent generates coarse-grained decisions by jointly observing wireless states and linguistic complexity, which are then refined via Denoising Diffusion Implicit Models (DDIM) to produce high-resolution, task- and channel-adaptive rank vectors. The two modules are optimized alternatively, with the DDIM trained under the Classifier-Free Guidance (CFG) paradigm to maintain alignment with PPO rewards. Experiments under varying signal-to-noise ratios demonstrate that AirLLM consistently enhances fine-tuning performance while significantly reducing transmission costs, highlighting the effectiveness of reinforcement-driven, diffusion-refined rank adaptation for scalable and efficient remote fine-tuning over the air. 

---
# SWE-MERA: A Dynamic Benchmark for Agenticly Evaluating Large Language Models on Software Engineering Tasks 

**Authors**: Pavel Adamenko, Mikhail Ivanov, Aidar Valeev, Rodion Levichev, Pavel Zadorozhny, Ivan Lopatin, Dmitry Babayev, Alena Fenogenova, Valentin Malykh  

**Link**: [PDF](https://arxiv.org/pdf/2507.11059)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) in software engineering has revealed critical limitations in existing benchmarks, particularly the widely used SWE-bench dataset. Recent studies have uncovered severe data contamination issues, e.g. SWE-bench reports 32.67% of successful patches involve direct solution leakage and 31.08\% pass due to inadequate test cases. We introduce SWE-MERA, a dynamic, continuously updated benchmark designed to address these fundamental challenges through an automated collection of real-world GitHub issues and rigorous quality validation. Our approach implements a reliable pipeline that ensures quality while minimizing contamination risks, resulting in approximately 10,000 potential tasks with 300 samples currently available. Evaluation using the Aider coding agent demonstrates strong discriminative power in state-of-the-art models. We report performance across a dozen recent LLMs evaluated on tasks collected between September 2024 and June 2025. 

---
# First-Order Error Matters: Accurate Compensation for Quantized Large Language Models 

**Authors**: Xingyu Zheng, Haotong Qin, Yuye Li, Jiakai Wang, Jinyang Guo, Michele Magno, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11017)  

**Abstract**: Post-training quantization (PTQ) offers an efficient approach to compressing large language models (LLMs), significantly reducing memory access and computational costs. Existing compensation-based weight calibration methods often rely on a second-order Taylor expansion to model quantization error, under the assumption that the first-order term is negligible in well-trained full-precision models. However, we reveal that the progressive compensation process introduces accumulated first-order deviations between latent weights and their full-precision counterparts, making this assumption fundamentally flawed. To address this, we propose FOEM, a novel PTQ method that explicitly incorporates first-order gradient terms to improve quantization error compensation. FOEM approximates gradients by directly computing the difference between latent and full-precision weights, avoiding the high cost and limited generalization of backpropagation-based gradient computation. This approach introduces minimal additional computational overhead. Moreover, FOEM leverages precomputed Cholesky factors to efficiently recover the inverse of Hessian submatrices in real time. Extensive experiments across a wide range of models and benchmarks demonstrate that FOEM consistently outperforms the classical GPTQ method. In 3-bit weight-only quantization, FOEM reduces the perplexity of Llama3-8B by 89.6%, and improves the 5-shot MMLU accuracy of Llama3-70B from 51.7% to 74.9%, approaching the full-precision performance of 78.6%. Furthermore, FOEM can be seamlessly integrated with advanced techniques such as GPTAQ and SpinQuant, yielding additional improvements under the challenging W4A4KV4 setting, and further narrowing the accuracy gap with full-precision baselines beyond what current state-of-the-art methods achieve. The code is available at this https URL. 

---
# LiLM-RDB-SFC: Lightweight Language Model with Relational Database-Guided DRL for Optimized SFC Provisioning 

**Authors**: Parisa Fard Moshiri, Xinyu Zhu, Poonam Lohan, Burak Kantarci, Emil Janulewicz  

**Link**: [PDF](https://arxiv.org/pdf/2507.10903)  

**Abstract**: Effective management of Service Function Chains (SFCs) and optimal Virtual Network Function (VNF) placement are critical challenges in modern Software-Defined Networking (SDN) and Network Function Virtualization (NFV) environments. Although Deep Reinforcement Learning (DRL) is widely adopted for dynamic network decision-making, its inherent dependency on structured data and fixed action rules often limits adaptability and responsiveness, particularly under unpredictable network conditions. This paper introduces LiLM-RDB-SFC, a novel approach combining Lightweight Language Model (LiLM) with Relational Database (RDB) to answer network state queries to guide DRL model for efficient SFC provisioning. Our proposed approach leverages two LiLMs, Bidirectional and Auto-Regressive Transformers (BART) and the Fine-tuned Language Net T5 (FLAN-T5), to interpret network data and support diverse query types related to SFC demands, data center resources, and VNF availability. Results demonstrate that FLAN-T5 outperforms BART with a lower test loss (0.00161 compared to 0.00734), higher accuracy (94.79% compared to 80.2%), and less processing time (2h 2min compared to 2h 38min). Moreover, when compared to the large language model SQLCoder, FLAN-T5 matches the accuracy of SQLCoder while cutting processing time by 96% (SQLCoder: 54 h 43 min; FLAN-T5: 2 h 2 min). 

---
# NavComposer: Composing Language Instructions for Navigation Trajectories through Action-Scene-Object Modularization 

**Authors**: Zongtao He, Liuyi Wang, Lu Chen, Chengju Liu, Qijun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.10894)  

**Abstract**: Language-guided navigation is a cornerstone of embodied AI, enabling agents to interpret language instructions and navigate complex environments. However, expert-provided instructions are limited in quantity, while synthesized annotations often lack quality, making them insufficient for large-scale research. To address this, we propose NavComposer, a novel framework for automatically generating high-quality navigation instructions. NavComposer explicitly decomposes semantic entities such as actions, scenes, and objects, and recomposes them into natural language instructions. Its modular architecture allows flexible integration of state-of-the-art techniques, while the explicit use of semantic entities enhances both the richness and accuracy of instructions. Moreover, it operates in a data-agnostic manner, supporting adaptation to diverse navigation trajectories without domain-specific training. Complementing NavComposer, we introduce NavInstrCritic, a comprehensive annotation-free evaluation system that assesses navigation instructions on three dimensions: contrastive matching, semantic consistency, and linguistic diversity. NavInstrCritic provides a holistic evaluation of instruction quality, addressing limitations of traditional metrics that rely heavily on expert annotations. By decoupling instruction generation and evaluation from specific navigation agents, our method enables more scalable and generalizable research. Extensive experiments provide direct and practical evidence for the effectiveness of our method. 

---
# Domain-Adaptive Small Language Models for Structured Tax Code Prediction 

**Authors**: Souvik Nath, Sumit Wadhwa, Luiz Perez  

**Link**: [PDF](https://arxiv.org/pdf/2507.10880)  

**Abstract**: Every day, multinational firms process thousands of transactions, each of which must adhere to tax regulations that vary by jurisdiction and are often nuanced. The determination of product and service tax codes, such as HSN or SAC is a major use case in Tax compliance. An accurate determination of such codes is imperative to avoid any tax penalties. This paper proposes a domain-adaptive small language model (SLM) with an encoder-decoder architecture for the enhanced prediction of product and service tax codes. In this approach, we address the problem of predicting hierarchical tax code sequences using unstructured product and services data. We employ an SLM based upon encoder-decoder architecture as this enables sequential generation of tax codes to capture the hierarchical dependencies present within the tax codes. Our experiments demonstrate that encoder-decoder SLMs can be successfully applied to the sequential prediction of structured tax codes, a domain that remains comparatively unexplored in current NLP research. In this paper, we demonstrate the superior performance of the domain-adaptive encoder-decoder SLMs over flat classifiers when applied to the Harmonized System of Nomenclature (HSN), and achieve superior results compared to decoder-only and encoder-only architectures for structured sequence generation tasks. This approach can also be scaled to other government-mandated tax commodity codes, such as United Nations Standard Products and Services Codes (UNSPSC), or Brazil's Nomenclatura Comum do Mercosul (NCM). 

---
# Overview of the TREC 2022 deep learning track 

**Authors**: Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, Jimmy Lin, Ellen M. Voorhees, Ian Soboroff  

**Link**: [PDF](https://arxiv.org/pdf/2507.10865)  

**Abstract**: This is the fourth year of the TREC Deep Learning track. As in previous years, we leverage the MS MARCO datasets that made hundreds of thousands of human annotated training labels available for both passage and document ranking tasks. In addition, this year we also leverage both the refreshed passage and document collections that were released last year leading to a nearly $16$ times increase in the size of the passage collection and nearly four times increase in the document collection size. Unlike previous years, in 2022 we mainly focused on constructing a more complete test collection for the passage retrieval task, which has been the primary focus of the track. The document ranking task was kept as a secondary task, where document-level labels were inferred from the passage-level labels. Our analysis shows that similar to previous years, deep neural ranking models that employ large scale pretraining continued to outperform traditional retrieval methods. Due to the focusing our judging resources on passage judging, we are more confident in the quality of this year's queries and judgments, with respect to our ability to distinguish between runs and reuse the dataset in future. We also see some surprises in overall outcomes. Some top-performing runs did not do dense retrieval. Runs that did single-stage dense retrieval were not as competitive this year as they were last year. 

---
# MultiVox: Benchmarking Voice Assistants for Multimodal Interactions 

**Authors**: Ramaneswaran Selvakumar, Ashish Seth, Nishit Anand, Utkarsh Tyagi, Sonal Kumar, Sreyan Ghosh, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2507.10859)  

**Abstract**: The rapid progress of Large Language Models (LLMs) has empowered omni models to act as voice assistants capable of understanding spoken dialogues. These models can process multimodal inputs beyond text, such as speech and visual data, enabling more context-aware interactions. However, current benchmarks fall short in comprehensively evaluating how well these models generate context-aware responses, particularly when it comes to implicitly understanding fine-grained speech characteristics, such as pitch, emotion, timbre, and volume or the environmental acoustic context such as background sounds. Additionally, they inadequately assess the ability of models to align paralinguistic cues with complementary visual signals to inform their responses. To address these gaps, we introduce MultiVox, the first omni voice assistant benchmark designed to evaluate the ability of voice assistants to integrate spoken and visual cues including paralinguistic speech features for truly multimodal understanding. Specifically, MultiVox includes 1000 human-annotated and recorded speech dialogues that encompass diverse paralinguistic features and a range of visual cues such as images and videos. Our evaluation on 9 state-of-the-art models reveals that, although humans excel at these tasks, current models consistently struggle to produce contextually grounded responses. 

---
# Automated Thematic Analyses Using LLMs: Xylazine Wound Management Social Media Chatter Use Case 

**Authors**: JaMor Hairston, Ritvik Ranjan, Sahithi Lakamana, Anthony Spadaro, Selen Bozkurt, Jeanmarie Perrone, Abeed Sarker  

**Link**: [PDF](https://arxiv.org/pdf/2507.10803)  

**Abstract**: Background Large language models (LLMs) face challenges in inductive thematic analysis, a task requiring deep interpretive and domain-specific expertise. We evaluated the feasibility of using LLMs to replicate expert-driven thematic analysis of social media data. Methods Using two temporally non-intersecting Reddit datasets on xylazine (n=286 and n=686, for model optimization and validation, respectively) with twelve expert-derived themes, we evaluated five LLMs against expert coding. We modeled the task as a series of binary classifications, rather than a single, multi-label classification, employing zero-, single-, and few-shot prompting strategies and measuring performance via accuracy, precision, recall, and F1-score. Results On the validation set, GPT-4o with two-shot prompting performed best (accuracy: 90.9%; F1-score: 0.71). For high-prevalence themes, model-derived thematic distributions closely mirrored expert classifications (e.g., xylazine use: 13.6% vs. 17.8%; MOUD use: 16.5% vs. 17.8%). Conclusions Our findings suggest that few-shot LLM-based approaches can automate thematic analyses, offering a scalable supplement for qualitative research. Keywords: thematic analysis, large language models, natural language processing, qualitative analysis, social media, prompt engineering, public health 

---
# Theory of Mind and Self-Disclosure to CUIs 

**Authors**: Samuel Rhys Cox  

**Link**: [PDF](https://arxiv.org/pdf/2507.10773)  

**Abstract**: Self-disclosure is important to help us feel better, yet is often difficult. This difficulty can arise from how we think people are going to react to our self-disclosure. In this workshop paper, we briefly discuss self-disclosure to conversational user interfaces (CUIs) in relation to various social cues. We then, discuss how expressions of uncertainty or representation of a CUI's reasoning could help encourage self-disclosure, by making a CUI's intended "theory of mind" more transparent to users. 

---
# From Semantic Web and MAS to Agentic AI: A Unified Narrative of the Web of Agents 

**Authors**: Tatiana Petrova, Aleksandr Puzikov, Boris Bliznukov, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2507.10644)  

**Abstract**: The concept of the Web of Agents (WoA), which transforms the static, document-centric Web into an environment of autonomous agents acting on users' behalf, has attracted growing interest as large language models (LLMs) become more capable. However, research in this area is still fragmented across different communities. Contemporary surveys catalog the latest LLM-powered frameworks, while the rich histories of Multi-Agent Systems (MAS) and the Semantic Web are often treated as separate, legacy domains. This fragmentation obscures the intellectual lineage of modern systems and hinders a holistic understanding of the field's trajectory. We present the first comprehensive evolutionary overview of the WoA. We show that modern protocols like A2A and the MCP, are direct evolutionary responses to the well-documented limitations of earlier standards like FIPA standards and OWL-based semantic agents. To systematize this analysis, we introduce a four-axis taxonomy (semantic foundation, communication paradigm, locus of intelligence, discovery mechanism). This framework provides a unified analytical lens for comparing agent architectures across all generations, revealing a clear line of descent where others have seen a disconnect. Our analysis identifies a paradigm shift in the 'locus of intelligence': from being encoded in external data (Semantic Web) or the platform (MAS) to being embedded within the agent's core model (LLM). This shift is foundational to modern Agentic AI, enabling the scalable and adaptive systems the WoA has long envisioned. We conclude that while new protocols are essential, they are insufficient for building a robust, open, trustworthy ecosystem. Finally, we argue that the next research frontier lies in solving persistent socio-technical challenges, and we map out a new agenda focused on decentralized identity, economic models, security, and governance for the emerging WoA. 

---
# Scalpel vs. Hammer: GRPO Amplifies Existing Capabilities, SFT Replaces Them 

**Authors**: Neel Rajani, Aryo Pradipta Gema, Seraphina Goldfarb-Tarrant, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2507.10616)  

**Abstract**: Training large language models (LLMs) for reasoning via maths and code datasets has become a major new focus in LLM post-training. Two particularly popular approaches are reinforcement learning (RL) and supervised fine-tuning (SFT), but their training dynamics are poorly understood. We present a comparative analysis of RL and SFT on the same maths problems with the same model and similar hyperparameters. We find that RL yields minor in-domain gains on maths and slight degradation on knowledge-intensive benchmarks like MMLU, while both trends are more pronounced in SFT. We also analyse model parameters across checkpoints, observing that both algorithms modify query and key weights the most. Meanwhile, SFT exhibits greater updates and also affects mid-layer MLPs more, leading us to hypothesise that this may have caused the out-of-domain degradation. We therefore investigate whether freezing parts of the model during training can mitigate the reduced performance on knowledge-intensive benchmarks. However, our results are inconclusive, with benefits on GPQA:Diamond and degradation on other benchmarks. Taken together, our observations provide a preliminary indication for why RL amplifies existing capabilities, while SFT replaces old skills with new ones. 

---
# Findings of the BEA 2025 Shared Task on Pedagogical Ability Assessment of AI-powered Tutors 

**Authors**: Ekaterina Kochmar, Kaushal Kumar Maurya, Kseniia Petukhova, KV Aditya Srivatsa, Anaïs Tack, Justin Vasselli  

**Link**: [PDF](https://arxiv.org/pdf/2507.10579)  

**Abstract**: This shared task has aimed to assess pedagogical abilities of AI tutors powered by large language models (LLMs), focusing on evaluating the quality of tutor responses aimed at student's mistake remediation within educational dialogues. The task consisted of five tracks designed to automatically evaluate the AI tutor's performance across key dimensions of mistake identification, precise location of the mistake, providing guidance, and feedback actionability, grounded in learning science principles that define good and effective tutor responses, as well as the track focusing on detection of the tutor identity. The task attracted over 50 international teams across all tracks. The submitted models were evaluated against gold-standard human annotations, and the results, while promising, show that there is still significant room for improvement in this domain: the best results for the four pedagogical ability assessment tracks range between macro F1 scores of 58.34 (for providing guidance) and 71.81 (for mistake identification) on three-class problems, with the best F1 score in the tutor identification track reaching 96.98 on a 9-class task. In this paper, we overview the main findings of the shared task, discuss the approaches taken by the teams, and analyze their performance. All resources associated with this task are made publicly available to support future research in this critical domain. 

---
# Can Large Language Models Understand As Well As Apply Patent Regulations to Pass a Hands-On Patent Attorney Test? 

**Authors**: Bhakti Khera, Rezvan Alamian, Pascal A. Scherz, Stephan M. Goetz  

**Link**: [PDF](https://arxiv.org/pdf/2507.10576)  

**Abstract**: The legal field already uses various large language models (LLMs) in actual applications, but their quantitative performance and reasons for it are underexplored. We evaluated several open-source and proprietary LLMs -- including GPT-series, Anthropic, Deepseek and Llama-3, variants -- on parts of the European Qualifying Examination (EQE) for future European Patent Attorneys. OpenAI o1 led with 0.82 accuracy and 0.81 F1 score, whereas (Amazon Web Services) AWS Llama 3.1 8B lagged at 0.50 accuracy, and a Python-deployed Llama 3.1 8B scored 0.55. The latter two are within the range of mere guessing for the two-answer forced-choice design. None of the evaluated models could have passed the examination fully, as accuracy never exceeded the average threshold of 0.90 required for professional-level standards -- also not models that are regularly promoted for their assumed beyond-PhD- and bar-admitted-lawyer-level performance. GPT-4o excelled at integrating text and graphics, while Claude 3 Opus often lost formatting coherence. Human patent experts evaluated the textual justifications and uncovered various critical shortcomings of each model. They valued clarity and legal rationale over the raw correctness of the answers, which revealed misalignment between automatic metrics and expert judgment. Model outputs were sensitive to modest temperature changes and prompt wording, which underscores the remaining necessity of expert oversight. Future work should target logical consistency, robust multimodality, and adaptive prompting to approach human-level patent proficiency. In summary, despite the outstanding performance of recent large models, the general public might overestimate their performance. The field has a long way to go to develop a virtual patent attorney. This paper wants to point out several specific limitations that need solutions. 

---
# Orchestrator-Agent Trust: A Modular Agentic AI Visual Classification System with Trust-Aware Orchestration and RAG-Based Reasoning 

**Authors**: Konstantinos I. Roumeliotis, Ranjan Sapkota, Manoj Karkee, Nikolaos D. Tselikas  

**Link**: [PDF](https://arxiv.org/pdf/2507.10571)  

**Abstract**: Modern Artificial Intelligence (AI) increasingly relies on multi-agent architectures that blend visual and language understanding. Yet, a pressing challenge remains: How can we trust these agents especially in zero-shot settings with no fine-tuning? We introduce a novel modular Agentic AI visual classification framework that integrates generalist multimodal agents with a non-visual reasoning orchestrator and a Retrieval-Augmented Generation (RAG) module. Applied to apple leaf disease diagnosis, we benchmark three configurations: (I) zero-shot with confidence-based orchestration, (II) fine-tuned agents with improved performance, and (III) trust-calibrated orchestration enhanced by CLIP-based image retrieval and re-evaluation loops. Using confidence calibration metrics (ECE, OCR, CCC), the orchestrator modulates trust across agents. Our results demonstrate a 77.94\% accuracy improvement in the zero-shot setting using trust-aware orchestration and RAG, achieving 85.63\% overall. GPT-4o showed better calibration, while Qwen-2.5-VL displayed overconfidence. Furthermore, image-RAG grounded predictions with visually similar cases, enabling correction of agent overconfidence via iterative re-evaluation. The proposed system separates perception (vision agents) from meta-reasoning (orchestrator), enabling scalable and interpretable multi-agent AI. This blueprint is extensible to diagnostics, biology, and other trust-critical domains. All models, prompts, results, and system components including the complete software source code are openly released to support reproducibility, transparency, and community benchmarking at Github: this https URL 

---
# NLP Meets the World: Toward Improving Conversations With the Public About Natural Language Processing Research 

**Authors**: Shomir Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2507.10559)  

**Abstract**: Recent developments in large language models (LLMs) have been accompanied by rapidly growing public interest in natural language processing (NLP). This attention is reflected by major news venues, which sometimes invite NLP researchers to share their knowledge and views with a wide audience. Recognizing the opportunities of the present, for both the research field and for individual researchers, this paper shares recommendations for communicating with a general audience about LLMs' capabilities and limitations. These recommendations cover three themes: vague terminology as an obstacle to public understanding, unreasonable expectations as obstacles to sustainable growth, and ethical failures as obstacles to continued support. Published NLP research and popular news coverage are cited to illustrate these themes with examples. The recommendations promote effective, transparent communication with the general public about NLP, in order to strengthen public understanding and encourage support for research. 

---
