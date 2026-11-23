# MemOnce: A Scalable Memory Architecture for Large Language Models 

**Authors**: Chang Nie, Chaoyou Fu, Songlin Liu, Yiqing Hu, Caifeng Shan  

**Link**: [PDF](https://arxiv.org/pdf/2611.27973)  

**Abstract**: Deploying large language models (LLMs) in long-horizon and continual interactions requires maintaining reusable memory states for future inference, posing challenges for existing architectures. Standard Transformers accumulate layer-wise, token-level KV caches, incurring linear GPU memory growth and quadratic computational complexity as the context expands. In contrast, linear-time architectures compress history into fixed-size states, but struggle to retain fine-grained details. We introduce MemOnce, a foundation architecture that learns scalable memory states for language models. MemOnce is based on two core designs: (a) Memory encoder-decoder: it caches only a small subset of final-layer hidden states as compact memory representations and dynamically decodes them back into KV caches for attention layers; (b) Memory sparse attention: for each continual input, MemOnce adaptively retrieves relevant memory representations, with the sparse indices computed only once across layers. Coupling these two designs enables efficient on-the-fly memory loading and KV cache decoding, drastically reducing the GPU memory footprint and achieving linear-time complexity. We conduct extensive experiments to show that MemOnce outperforms Transformers and other architectures across varying model sizes, context lengths, and training paradigms (pretraining and post-training adaptation). In particular, adapting Qwen3.5-4B with MemOnce reduces memory state cache by 1313$\times$ and prefill latency by 6.9$\times$ at 1M context, and improves throughput by 3.5$\times$ and overall memory performance by +5.77%. Finally, MemOnce generalizes zero-shot to downstream representation tasks, achieving competitive results against specialized models. 

---
# Evaluating Prompting Strategies for Zero-Shot Multilingual Aspect-Based Sentiment Analysis with Large Language Models 

**Authors**: Jakub Šm{í}d, Pavel Kr{á}l  

**Link**: [PDF](https://arxiv.org/pdf/2611.27807)  

**Abstract**: While large language models (LLMs) have advanced natural language understanding, their ability to perform complex target-aspect-sentiment detection (TASD) -- a key task within aspect-based sentiment analysis (ABSA) -- in zero-shot multilingual settings remains underexplored. Existing ABSA research predominantly focuses on English or relies on fine-tuned task-specific models, leaving a gap in understanding how open-weight LLMs handle compound sentiment triplet extraction across diverse language families and domains. We present a systematic evaluation of zero-shot multilingual TASD across 12 languages and three domains using seven instruction-tuned open-weight models -- ranging from 8B to 35B parameters across dense and mixture-of-experts (MoE) architectures -- under four prompting paradigms: monolithic, chain-of-thought, sequential, and sequential refine. Our results show that direct monolithic prompting achieves strong zero-shot performance, outperforming multi-step pipelines, while dense models consistently outperform sparse MoE architectures. Through fine-grained error taxonomy profiling and step-level pipeline tracking, we demonstrate that initial aspect term recall forms a hard bottleneck, whereas multi-step decomposition degrades extraction quality; iterative refinement only partially recovers this loss on larger models while inducing net-negative over-refinement on smaller architectures. Finally, generation volume analysis shows that multi-step strategies incur up to 2.38x token overhead without F1 gains, establishing monolithic prompting as the most effective and efficient paradigm for multilingual ABSA. 

---
# Enhancing Attention with Higher-Order Attention 

**Authors**: R. I. Cukier  

**Link**: [PDF](https://arxiv.org/pdf/2611.27764)  

**Abstract**: Current decoder large language models rely on a combination of attention, feed forward and residual layers. The attention used is autoregressive (AR) though this is not strictly correct, as we note. That motivates the introduction of other attention methods that, while also not strictly AR, do enhance the AR character. One is based on a decomposition of mutual information (MI) into a set of Conditional MIs (CMIs) that correlate key tokens with pairs of query tokens. A second, higher-order attention, introduces correlations between contiguous key and query tokens. A third introduces correlations between a set of key and query tokens and uses a Generalized Maximum Entropy Principle to obtain an attention expression spanning a sequence of tokens. All these methods may enhance the information content of the autoregressive probabilities that are used in inference/generation. Applications to contrastive decoding and a systematic analysis of autoregressive probabilities are presented. 

---
# From Novel Combinations to Informative Absence: Structural permission, communicative opportunity, and form choice in language learning 

**Authors**: Yanzhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.27762)  

**Abstract**: Language learners must infer which novel expressions are grammatically licensed and when continued absence is evidence against them. Our generative theory unites these capacities via counterfactual expression opportunities, decomposing utterance generation into structural permission, communicative opportunity, and form choice. First, shared predictive continuations recover structural permission for unseen word-construction pairings across 30 artificial grammars (mean AUC 0.945 after 120 sentences). Second, under hierarchical uncertainty, we derive a joint evidence law linking omissions to context diversity and within-context repetition, and an exact finite-sample context-refinement inequality. Thus, observing related expressions calibrates expected use to sharpen negative evidence, and higher expected use can drive steeper posterior declines for stronger-permission candidates (reversing initial order after six omissions in a fixed-rate example). Third, we prove that while raw counts confound semantic availability and form selection, scene-level opportunity observations identify both. Empirically, reanalyzing adult and child artificial-language experiments shows meaning-conditioned restriction; judgments of restricted combinations improve held-out production predictions across 399 participants (+0.036 nats/production). Naturalistic child-directed and child speech (11 families, six corpora) shows pervasive source-level usage clustering beyond lexical, number, and dependency-role controls, grounding hierarchical rate uncertainty. Finally, modeling causative judgments across five languages shows that structural compatibility and calibrated usage probabilities draw on distinct distributional, semantic, and discriminative cues. Together, these findings show how the organization and calibration of communicative opportunities govern the evidential value of linguistic absence. 

---
# RoSci: Exploring Robustness in Science VLMs 

**Authors**: Junying Wang, Yijin Guo, Yalun Wu, Ziheng Jiao, Farong Wen, Qihang Yan, Wenzhe Li, Ye Shen, Qi Jia, Chunyi Li, Zicheng Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2611.27750)  

**Abstract**: Recent advancements in Vision-Language Models (VLMs) enable significant progress in scientific tasks. However, these models are generally assumed to be presented with complete and accurate information, overlooking the challenges of erroneous/missing information in wild scenarios. To address this gap, we introduce the concept of Scientific Model Robustness, which defines a model's ability to detect anomalies, assess solvability, and perform self-calibration when faced with Information Error (inconsistencies) or Information Loss (insufficiency). Specifically, we present RoSci-Bench, a diagnostic benchmark of 1,220 samples using a seed-variant strategy across six degradation types. To address VLM fragility, we further propose RoSci-Agent, a framework that employs a Parse-Verify-Decide pipeline to vet premises before reasoning. The evaluated VLMs sometimes answer questions whose premises are invalid or incomplete. RoSci-Agent achieves a higher Overall Robustness Score than the evaluated standalone VLMs. RoSci-Bench gives the multimodal reasoning community a common test of when models answer, request clarification, or reject a scientific question. 

---
# AutoOntologyBench: A Task-Grounded Benchmark for Agentic Knowledge Engineering 

**Authors**: Minye Shao, Chenghao Xiao, Xingyue Zhao, Yang Long  

**Link**: [PDF](https://arxiv.org/pdf/2611.27734)  

**Abstract**: Foundation-model agents are increasingly extending beyond software engineering (SWE) into autonomous scientific research, where acquiring and integrating domain knowledge remains a critical challenge. Recent industrial advances have increasingly adopted knowledge graphs and ontology-based semantic layers to organize heterogeneous knowledge into reusable representations. We ask: Can autonomous agents construct ontologies that help them solve real tasks, and do they offer measurable benefits over alternative knowledge representations? Existing benchmarks evaluate ontology correctness or agent task success in isolation, leaving the value of autonomously acquired knowledge unresolved. We therefore introduce AutoOntologyBench, a task-grounded benchmark comprising 19,947 audited tasks grounded in 18 real-world knowledge sources across 7 domains, with source-native verification. Spanning Application, Repair, Organization, and Construction, AutoOntologyBench compares nine frontier-model agents under matched knowledge representations. Layered on retrieval, agent-built ontologies add 6.1 points of task success, yet trail an expert-derived reference by 13.0, identifying autonomous knowledge acquisition as the bottleneck to scalable scientific research. Code and resources are available at this https URL. 

---
# Decision Models for Multilingual Emotion Classification: A Comparative Study Against LLMs and Encoders 

**Authors**: Alessandro Piccolo, Francesco De Nicolò, Francesco Manco, Eufemia Lella  

**Link**: [PDF](https://arxiv.org/pdf/2611.27721)  

**Abstract**: Decision models receive a question with a closed list of answers and return a probability for each answer instead of free text. They promise low latency and robustness to prompt wording, but no study has compared several of them with fine-tuned encoders across languages. We ask whether systems used without training on labelled data (zero-shot) can replace fine-tuning, and what the decision format adds over a general-purpose large language model (LLM). We compare three decision models (Jev, OpenAI Decisions, Strands Decider 2B) and two LLMs (GPT-6 Luna, Qwen3.5-2B), given Italian instructions, with two fine-tuned multilingual encoders (tabularisai, XLM-EMO) on four emotions (joy, sadness, anger, fear) in 10 social-media datasets in five languages, with paired difference and equivalence tests. The three best zero-shot systems (two decision models and one LLM) score on average about one macro-F1 point below an encoder fine-tuned on the target dataset; our test sets cannot exclude a gap of 4-5 points, and on single datasets either approach wins by up to 22 points. They beat encoders fine-tuned on another dataset by 8-15 points, though against XLM-EMO this may not extend to new datasets, and instructions in each text's language change their results by under 1 point. Exploratorily, they beat the strongest public emotion model used as is by about 6 points. OpenAI's decision endpoint matches GPT-6 Luna, the LLM it runs on, at a fifth of its median latency, and a 2-billion-parameter decision model beats a same-size general LLM by 10 points, owing to its training, not the decision format. The best zero-shot systems, the LLM included, have an SD of about 1 point across prompts. Without labelled data they are a strong default, and decision models add lower latency and, unlike generative LLM use, per-option probabilities. Fine-tuning won by a wide margin only on the largest dataset, GoEmotions. 

---
# Hy-MT3: More Accurate, Robust and Practical Multilingual Translation Models in the Wild 

**Authors**: Mao Zheng, Zheng Li, Fangxu Liu, Tao Chen, Bingxin Qu, Bo Lv, Mingrui Sun, Hongyu Li, Hong Huang, Jinlong Song, Decheng Wu, Guanghua Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27673)  

**Abstract**: Hy-MT3 is the third generation of the Hunyuan multilingual translation model family, including \textbf{Hy-MT3-1.8B}, \textbf{Hy-MT3-7B} and \textbf{Hy-MT3-30B-A3B (MoE)}. Compared with Hy-MT2, Hy-MT3 translates \textbf{more accurately, more completely and more faithfully to user constraints}: on real-world scenario data, the share of samples with mistranslations is reduced by \textbf{43\%}, omissions or additions by \textbf{74\%}, and instruction-following errors by \textbf{40\%}, while language coverage is expanded from \textbf{38 to 102} languages. With only 3B activated parameters, \textbf{Hy-MT3-30B-A3B achieves the best overall performance among all evaluated models in non-thinking mode}, including flagship models such as Qwen3.8-Max, Kimi-K3.1-Pro and DeepSeek-V4.1-Flash, and approaches the strongest closed-source thinking models such as GPT-6.1-Sol and Gemini-4-Argon, while surpassing models that consume \textbf{more than 30$\times$} its inference compute. Beyond general translation, Hy-MT3 is optimized for two industrial scenarios. For \textbf{game localization}, built with high-quality game-domain data and a professional evaluation system from the Honor of Kings team, Hy-MT3-30B-A3B matches or outperforms strong closed-source models such as Gemini 3.1 Pro on all core metrics, and can be plugged into game development pipelines through an open-source toolchain. For \textbf{image translation}, Hy-MT3 improves context understanding and error correction to better tolerate OCR errors, achieving the best result among models of comparable size. Finally, we open-source \textbf{TransRubric}, a rubric-based automatic evaluation suite for translation, which outperforms other evaluation schemes on the WMT26 metrics shared task test set and is more stable with respect to human evaluation. 

---
# Auditing Post-Training Unlearning in Diffusion Language Models using Sparse Autoencoders 

**Authors**: Yunha Seo, Changhoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.27652)  

**Abstract**: Machine unlearning is an approach to privacy protection: it aims to remove the influence of designated data while preserving the model's behavior on information that should be retained. While a growing body of work investigates this objective in autoregressive large language models, machine unlearning remains underexplored in diffusion language models (DLMs). DLMs generate text through iterative denoising, exposing partially masked states whose internal representations change throughout the generation process. This raises a DLM-specific question: can selective forgetting improve by identifying forget-data-associated features and adapting their suppression across denoising stages? We conduct a controlled investigation of this question in LLaDA-8B-Instruct. To evaluate the unlearning of knowledge plausibly encoded during pre-training, we use WMDP as our forget-domain benchmark. Using top-k sparse autoencoders and denoising-state replay, we compare static and stage-adaptive feature suppression. We find substantial shifts in forget-data-associated features across denoising stages, but tracking these dynamics alone does not reliably improve the forget-retain trade-off at the evaluated operating points. Together, our study provides an initial empirical foundation for developing and evaluating machine unlearning as a post-training privacy mechanism in DLMs. 

---
# Tasklets: A Unit-Test Benchmark for Language Model Architectures 

**Authors**: Zayd Muhammad Kawakibi Zuhri, Munachiso Samuel Nwadike, Kentaro Inui, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2611.27607)  

**Abstract**: We introduce Tasklets, a benchmark of small synthetic tasks for evaluating the computational capabilities of language model sequence-mixing layers. These tasks probe foundational capabilities such as recall, sequence manipulation, and state tracking. Such tasks are widely used in architecture research, but differences in task selection, formulation, and evaluation make comparisons across studies difficult. Tasklets brings these tasks together, organizing them by the capabilities they test and providing a shared setup for evaluating how different architectural layer types learn fundamental computations. We compare a broad range of layer types through controlled experiments with small models. Our task implementations and evaluation code support reproducible, low-cost testing of new architectural proposals: this https URL. 

---
# Alliance Rupture and Repair in Simulating Cognitive Behavioral Therapy Dialogues 

**Authors**: Jian Chai, Yitian Yang, Siqi Liu, Yi-Chia Wang, Yi-Chieh Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.27544)  

**Abstract**: In cognitive behavioral therapy (CBT), routine work can strain the client-counselor working alliance, and the counselor must then notice the rupture, repair it, and adapt the work that follows. Existing CBT simulators model client resistance and follow structured treatment plans, but rarely track alliance state or allow counselors to repair ruptures over multiple turns and revise their plans accordingly. We present ALLY, a simulation framework that brings alliance rupture, repair, and replanning into ongoing CBT dialogue. Each client has a hidden alliance state that governs how openly the client engages. Like a real counselor, the counselor agent sees only the dialogue. It must notice when the client pulls back or pushes against the work, repair the rupture, and replan the rest of the session around what it has learned. Automatic evaluation shows that ALLY's counselors are closer to real CBT counselors than those of existing public synthetic datasets in strategy flow and emotional dynamics, and that ALLY is rated highest overall in counseling quality (CTRS) and interpersonal skill (FIS-T). An expert evaluation with 20 licensed therapists also finds the simulated ruptures plausible and the repairs and replanning largely appropriate. 

---
# Open CDLM: A Benchmark and Library for Continuous Diffusion Language Models 

**Authors**: Hao Jiang, Ruixi Chen, Jiayi Han, Yulei Sun, Yujian Chen, Gen Li, Brian.K Chen, Guang Cheng, Weijian Luo, Tianyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27536)  

**Abstract**: Reliable comparisons of continuous diffusion language models (CDLMs) require evaluating both model fit and the distribution of generated text. Two common practices leave this goal only partially addressed. Likelihood-based evaluation often reports a variational upper bound on discrete negative log-likelihood, whose slack can confound comparisons. Generated-sample-based evaluation commonly combines generative perplexity with within-sequence unigram entropy; both scores can remain favorable even when a sampler produces the same text on every draw. We introduce \textbf{Open CDLM}, a benchmark for evaluating CDLMs and their samplers, and describe the design of an accompanying open-source library to support reproducible comparisons. For likelihood-based evaluation, we clarify which comparisons of model fit are supported by reported perplexity bounds and how those conclusions depend on the evaluation procedure. For sample-based evaluation, we use \textit{Kernel Text Discrepancy} (\textbf{KTD}) to compare collections of generated and real texts, alongside diagnostics for diversity, validity, and coverage. In controlled OpenWebText experiments, replacing a collection of texts with copies of a single selected passage improves both conventional generation scores. MAUVE detects the resulting collapse, and its response to other perturbations depends on the text representation and the type of perturbation. This work marks the beginning of Open CDLM as an open-source project for CDLM research. We will maintain and extend the library over time and welcome contributions from the community. 

---
# How Many Bytes Has a BPE Model Seen? Estimating Byte-Token Exposure in Pretraining from Corpus Statistics and Embedding Norms 

**Authors**: Liwei Zhang, ZhiMing Tong, Mochen Guan, Linzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.27506)  

**Abstract**: Byte-level BPE vocabularies retain 256 single-byte fallback tokens. When converting such a model into a pure byte-level model, these tokens provide the starting embeddings. How much training have they received? We estimate byte-token exposure in Qwen3.8 using two independent routes. First, we tokenize 540 MB of public corpora covering 40 domains and 33 languages, then combine per-domain token shares with assumed pretraining mixtures. Single-byte tokens account for 13 to 36% of tokens, mostly ASCII; genuine UTF-8 fallback bytes (0x80-0xFF) account for 0.15 to 0.27% under mainstream mixtures, or about 2 billion occurrences per trillion pretraining tokens. Fallback rates vary by four orders of magnitude across languages, reflecting how well each script is covered by the vocabulary. Second, we analyze embedding and output-matrix row norms for four open models, using untrained padding rows as a zero-exposure baseline and ordinary tokens with known frequencies for calibration. In all four models, the 13 bytes that cannot occur in valid UTF-8 remain at initialization, while every legal high byte lies in a high-frequency band. For Qwen3.8, the norm-based estimate agrees with the corpus estimate within a factor of four. The norm-frequency relation eventually folds, so inversion is useful only as a model-specific, order-of-magnitude estimate. No private data were used. 

---
# The Assistant Persona Differs Across Languages in Activation Space 

**Authors**: Daniel Yoo, Naama Rozen, Evan Harris  

**Link**: [PDF](https://arxiv.org/pdf/2611.27504)  

**Abstract**: Large language models behave differently across languages, and are noticeably less safe in lower-resource languages. This gap is well established behaviorally, but not mechanistically. Directions in activation space, including the assistant persona direction, are almost always extracted from English and assumed to generalize across languages. We extract the assistant persona direction in 31 languages in Qwen3-32B, and find that the resulting axes differ substantially: their cosine similarity to the English axis ranges from 0.378 to 0.911. Moreover, each language's assistant persona direction and its magnitude are correlated with the language's resource level. We also find that the model's default assistant state is farther from the helpful assistant pole in lower-resource languages. Finally, steering against a language's own axis amplifies persona-based jailbreak more than steering against the English axis. These results indicate that the model does not carry a single assistant representation across languages. More broadly, the assistant persona may need to be an explicit target of multilingual alignment. 

---
# Typhoon OCR 2: From Thai Document Transcription to Multilingual, Multi-Task OCR 

**Authors**: Surapon Nonesung, Natapong Nitarach, Sittipong Sripaisarnmongkol, Kunat Pipatanakul, Warit Sirichotedumrong  

**Link**: [PDF](https://arxiv.org/pdf/2611.27494)  

**Abstract**: Typhoon OCR 1.5 focused on reading Thai documents. However, in real-world workflows, users often need to extract specific information or preserve the structure of an entire page for further processing. Typhoon OCR 2 handles both tasks in a single model, with prompt-driven extraction and structured full-page parsing. Full-page parsing covers Thai, English, and Chinese. We release compact 2B and 0.8B models adapted from open document parsers through supervised fine-tuning on curated Thai and public multilingual data, followed by Group Relative Policy Optimization (GRPO) with task rewards and a rule penalizing unfinished responses. Among evaluated open models, Typhoon OCR 2 2B scores highest on Thai MDPBench (80.7 versus 68.9 for Typhoon OCR 1.5) and on ThaiOCRBench (59.91 versus 40.45), which covers recognition, extraction, and document understanding. The 2B model remains close to its base on English olmOCR-bench (84.7 versus 84.4) and on OmniDocBench, which evaluates English and Chinese together (90.97 versus 91.03). On MDPBench, GRPO reduces the mean share of pages with repetition loops for the 2B model from 3.2% to 0.7%, with a smaller and statistically uncertain score gain for the 0.8B model. We release the model weights, Python package, and evaluation suite. 

---
# Alignment without Transfer: Tokenization Impacts Cross-lingual Representations during Multilingual Language Model Pretraining 

**Authors**: Jonathan von Rad, Pontus Stenetorp, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27491)  

**Abstract**: Tokenization represents one of the most crucial steps in mulilingual NLP pipelines. Yet, tokenizers routinely under-allocate vocabulary to low-resource languages. While improved allocation leads to better performance, it remains poorly understood how tokenization shapes cross-lingual representations and transfer during pretraining. To address this, we pretrain 39 mono- and bilingual 1B decoder-only models on FineWeb2-HQ across English, German, French, Arabic and Chinese, varying only the tokenizer. Our results demonstrate that improved vocabulary allocation (i) substantially raises cross-lingual representation alignment in early-to-mid layers, an effect that prior work measuring only the output layer could not observe, (ii) increases cross-lingual parametric knowledge sharing in feed-forward neurons, supported by correlational and causal evidence, and (iii) an alignment-transfer dissociation, where cross-lingual transfer remains essentially unchanged despite higher alignment. Moreover, transfer is significantly lower for cross-script bilinguals even under fair vocabulary allocation, indicating that the script barrier is not merely a tokenizer artifact. All of our checkpoints and code are publicly available. 

---
# Structurally Different, Functionally Persistent: Capability Regions Under Model Optimization 

**Authors**: Jiwoo Kum, Chanwoo Kim, Doyun Kim, Seonghyeon Cho, Sihyeon Lee, Chanjun Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.27445)  

**Abstract**: Large language model (LLM) capabilities have been associated with localized parameter regions, enabling targeted model editing and intervention. However, it remains unclear whether a region identified before model optimization remains an effective intervention target afterward. We evaluate the functional persistence and parameter overlap of coding capability regions across three instruction-tuned models under quantization, pruning, and LoRA-based adaptation. Although re-extracted regions exhibit substantial parameter reorganization, ablating the original region after optimization eliminates or nearly eliminates coding performance in almost all conditions, while magnitude-matched controls preserve performance. Partition analysis further shows that the shared subset of the original and re-extracted regions accounts for most of the observed ablation effect in the majority of conditions. These results demonstrate that parameter overlap and functional persistence are distinct: capability regions can remain causally effective intervention targets even when their parameter realization changes substantially. 

---
# Silent Arbitration: How Large Language Models Resolve Co-Authoritative Conflicting Instructions 

**Authors**: Sihyeon Lee, Jiwoo Kum, Jihun Song, Chanwoo Kim, Chanjun Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.27444)  

**Abstract**: Large language models increasingly receive high-level instructions about roles, values, interests, and communication style. Such instructions can conflict even when a prompt assigns them equal authority and provides no priority rule. We study co-authoritative persona conflict, in which a single prompt jointly assigns two opposing personas. Across eight conflict types grounded in published source material and five LLMs, we measure conflict acknowledgment, resolution strategy, and directional stance. Models acknowledge incompatibility in 72% of direct checks but explicitly surface it in only 22% of task responses, revealing silent arbitration. Dominance accounts for 45% of dual-persona responses, and 35% of compromise-labeled responses retain directional lean. Prompt framing further changes the visible form of arbitration substantially more than ordinary paraphrase variation. These findings show that evaluating instruction following should assess whether models disclose conflicts, how they resolve them, and which persona their responses favor. 

---
# No Audio, No Transcripts, No Problem: LLM-based ASR Adaptation Using Only Domain Documents 

**Authors**: Sergio Burdisso, Dairazalia Sánchez-Cortés, Esaú Villatoro-Tello, Thibault Bañeras-Roux, Manjunath K E, Kadri Hacioglu, Petr Motlicek, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2611.27434)  

**Abstract**: Can a speech recognizer be adapted to a new domain with no target audio and no transcripts, using only written documentation? This is the realistic onboarding setting: a new customer hands over raw documentation, not the transcripts that text-only adaptation normally assumes. We turn a domain's documents into adaptation text in two ways, directly as written sentences or rendered by a large language model (LLM) into synthetic spoken dialogs, and feed them to a recently proposed denoising-based text-only adaptation method. We show that, across in-domain, out-of-domain, and cross-domain settings on two corpora, document-only adaptation can match or closely approach real transcriptions; whether written or conversational rendering is best depends on speaking style and the amount of text generated. To our knowledge, this is the first study of document-only adaptation of LLM-based ASR. As no existing corpus pairs audio, transcripts, and documents, we also built and released a document-grounded benchmark. 

---
# ORBIT: Offset-based Activation Steering for Cross-lingual Multimodal Fact Verification 

**Authors**: Huy Trieu, Duy-Minh Nguyen-Tran, Thanh-Dat Nguyen, Huy Tien Nguyen, Tung Le  

**Link**: [PDF](https://arxiv.org/pdf/2611.27421)  

**Abstract**: Multimodal fact verification faces a fundamental accessibility bottleneck: while large proprietary models exhibit cross-lingual robustness, small vision-language models suffer a severe, disproportionate accuracy collapse on low-resource languages. Closing this gap traditionally requires prohibitive scale or extensive language-specific fine-tuning. In this work, we propose Offset Recovery via Bilingual Inference-time Technique (ORBIT), an algorithmic intervention that mitigates this collapse at inference time. ORBIT estimates span-specific offset vectors from parallel claim-evidence instances and applies them to claim and evidence representations, targeting the cross-lingual degradation induced by low-resource evidence. Experiments across 81 configurations reveal that ORBIT substantially recovers both accuracy and F1-score in low-resource settings, exposes calibration offset norms as a diagnostic proxy, and outperforms translation-based and steering baselines. 

---
# WaXup: Weight Averaging of Experts as Data Augmentation for Robust Introspection 

**Authors**: Marc Boëlle, Lucas Dixon, Alexandre Ramé  

**Link**: [PDF](https://arxiv.org/pdf/2611.27353)  

**Abstract**: As the capabilities of large language models keep improving, traditional mechanistic interpretability methods struggle to keep pace. In contrast, trainable introspection methods emerge as a promising alternative; they repurpose the model itself as a self-diagnostic tool, trained to verbalize what it has learned during finetuning in natural language. While training on datasets of finetuned experts achieves encouraging results, it currently lacks robustness and generalization. Building on the data augmentation framework of MiXup, we introduce WaXup, a method that augments the introspection training set through linear interpolation in the weight space. This is motivated by the linear mode connectivity property: weight averaged experts interpolate the behaviors of the initial experts. We evaluate WaXup on three benchmarks: hidden-behavior models, a meta-dataset of real-world NLP tasks and AuditBench, a challenging alignment auditing suite of models adversarially trained to conceal their behaviors. WaXup improves the accuracy of generated descriptions as well as robustness to various distribution shifts. We hope to improve the reliability of trainable introspection methods towards better and safer models. 

---
# Breaking the Long Horizon: Short-Horizon Process Supervision for Long-Form Generation in LLMs 

**Authors**: Heyang Xu, HanWen Hao, Qianren Mao, Zhuo Liu, Qili Zhang, Sheng Yang, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.27348)  

**Abstract**: Long-form generation is increasingly important for agentic LLM systems, yet current reinforcement learning methods typically optimize an entire document with a single outcome-level reward. This delayed and coarse-grained supervision makes it difficult to localize errors and assign credit across semantically distinct parts of the output. To address this, we propose a short-horizon framework that reformulates long-form generation through semantic decomposition into segment-level decisions. Each segment is generated with prefix-conditioned context, allowing cross-segment coherence to be preserved while keeping supervision local. The generated segments are then scored with rubric-based rewards for adherence, coherence, and relevance, enabling explicit local supervision at the segment level. These local rewards are used to compute step-wise advantages, yielding localized policy updates. To isolate the effect of supervision granularity, we instantiate both outcome-level supervision and our short-horizon process reward method (SH-PRM) under the same GRPO framework. Across multiple LLMs and two long-form generation benchmarks, SH-PRM consistently outperforms outcome-level supervision and strong inference-time baselines under the same optimizer and comparable budgets. Further analyses suggest that these gains primarily arise from improved credit assignment. Overall, our results suggest that aligning supervision granularity with the semantic structure of long-form generation is an effective design principle for long-form LLM training. 

---
# Reasoning in Fewer Tokens: Concept-Normalized Multilingual Scratchpads and Reversible Model-Adaptive Reasoning Codes 

**Authors**: Rabimba Karanjai, Suryabhan Singh Hada, Hemanth Hegadehalli Madhavarao, Qun Gu, Libin N. George, Wenhuan Sun, Neha Monga, Richard Williamson, Prakhar Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2611.27326)  

**Abstract**: Chain-of-thought prompting significantly improves the reasoning capabilities of large language models (LLMs) but incurs high inference costs due to lengthy autoregressive decoding. While previous efficiency approaches reduce costs by discarding tokens or utilizing continuous latent representations, these lossy methods fundamentally sacrifice interpretability and auditability. In this work, we introduce a novel compression framework for intermediate reasoning. We apply Byte-Pair Encoding (BPE) to reasoning corpora to learn a compressed vocabulary of Private Use Area (PUA) tokens, each representing frequent semantic spans. After expanding the LLM's vocabulary, we fine-tune the model via strictly 1:1, our method is lossless. We evaluate our approach on Llama-3.1-8B, Qwen2.5-14B, and Qwen2.5-32B across four diverse benchmarks. Our method reduces total generated tokens by 14.7% on average with reasoning trace lengths dropping by up to 27.1%, while incurring a negligible average accuracy cost of 1.6 percentage points. Crucially, this approach is orthogonal to existing lossy methods, allowing practitioners to stack lossless and irreversible compression techniques. We demonstrate that a compressed intermediate language provides a highly efficient computational resource that accelerates inference while fully preserving the transparency of LLM reasoning. 

---
# Do Language Models Know Which Words Steer Their Reasoning? 

**Authors**: Yanbei Jiang, Chao Lei, Jey Han Lau, Krista A. Ehinger, Rena Gao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.27317)  

**Abstract**: While chain-of-thought (CoT) prompting elicits intermediate reasoning steps from language models, it is unclear which words in the context actually steer the CoT. We identify Phantom Context Influence (PCI): language models are disproportionately steered by semantically irrelevant context words, even as the same models deny those words' relevance when asked. To localize PCI, we introduce Leave-One-Out Word JSD analysis, which removes one context word at a time from the otherwise intact context to isolate its marginal effect on the next-token distribution; words whose removal flips the prediction (pivot words) are passed to the same LLM to self-judge whether it recognizes this predictive dependence. Across four models and three benchmarks, language models largely do not know which words steer their reasoning, a mismatch we call the Generation-Judgment Contradiction. Inspired by this phenomenon, we propose Online KV Suppression: it scores every prompt word's influence over the whole chain of thought in a single pass, then zeroes the KV-cache entries of the highest-influence word the model itself judges inert and redecodes. This single-word intervention improves CoT accuracy across four models and five reasoning benchmarks, by up to 8.9 percentage points on math reasoning benchmark. 

---
# Observed, Curated, Reused: Observation Bias in Clinical AI Evaluation 

**Authors**: Md Shamim Ahmed, Lukas Galke Poech, Richard Röttger  

**Link**: [PDF](https://arxiv.org/pdf/2611.27301)  

**Abstract**: Clinical AI evaluations are vulnerable to observation bias embedded in routine medical documentation, which captures only what clinicians notice, test, and record rather than a complete clinical picture. As this evidence travels into publications and reference knowledge graphs, crucial distinctions disappear: unexamined symptoms blur with confirmed negatives, and chronological disease progression is lost. Surviving evidence can also enter both the benchmark query and the reference knowledge base, creating a self-referential loop where a patient is evaluated against clues derived from their own published case report.
To quantify this bias, we tracked 6,503 clinical cases across six historical Human Phenotype Ontology (HPO) releases (October 2021-June 2026) to identify when HPO first cited the publications describing them. This yielded a confirmatory cohort with an observable "source ingestion" event, where HPO cited a case's source publication as target-disease evidence. Diagnostic rankers depended heavily on these source publications: for our primary semantic ranker (Resnik), publication-weighted mean reciprocal rank rose from 0.26 before paper ingestion to 0.84 upon inclusion in HPO, but collapsed to 0.05 when source-linked assertions were counterfactually removed. Two additional rankers (IC-cosine and LIRICAL) reproduced this trajectory, and control experiments confirmed the diagnostic jump was driven by the patient's publication rather than background HPO growth.
Clinical AI systems may therefore score well because their knowledge base contains evidence linked to the test case's source publication, rather than through generalization. Modern knowledge is not the problem; evaluation evidence must match the claim. Trustworthy benchmarks must separate patients, source-linked evidence, and time: historical tests should use only knowledge available at the intended decision time. 

---
# Do Companies Deliver on Their IPO Claims? 

**Authors**: Vidhyakshaya Kannan, Shiv Banafar, Michael Galarnyk, Sagnik Nandi, Soumeya Kannan, Shauray Arora, Abanoub Abdelmalak, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2611.27273)  

**Abstract**: Initial public offering (IPO) filings describe what a company does and what it intends to become. Yet whether individual forward-looking claims are subsequently realized is difficult to assess. We introduce Vision Follow-Through (VFT), a benchmark and agentic pipeline that separates claim extraction, category assignment, verifiability, and evidence-grounded follow-through assessment. \vftclaims{} contains 5,066 human-annotated claim spans from 50 IPO filings, while \vftverify{} adds verifiability tiers, follow-through outcomes, and cited later evidence for 1,000 claims. Across language models, classifying human-annotated claims is substantially easier than extracting claims from full filings. Extraction coverage also varies sharply by category, with Organizational/Culture claims recovered far less often than Market Opportunity claims. Follow-through assessment poses a separate challenge: model judgments diverge from human assessments in different ways rather than exhibiting a single consistent bias. These results show that reliable assessment of corporate follow-through requires both finding the relevant claims and grounding subsequent judgments in later evidence. 

---
# LongCat Recall Sparse Attention: Co-Designing Key Organization and Retrieval for Long-Context Inference 

**Authors**: Hao Zhang, Bin Xiao, Kefeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27233)  

**Abstract**: Retrieval-based sparse attention, exemplified by DeepSeek Sparse Attention (DSA), reduces attention computation by selecting a small subset of key-value entries. However, selecting these entries can still require searching the entire visible history. For long sequences, retrieval becomes a major bottleneck, limiting the efficiency gains from sparse attention. To address this problem, we propose LongCat Recall Sparse Attention (LRSA), which combines fine-grained representatives with paged retrieval in an incrementally maintained index. LRSA partitions the sequence into time buckets that can be indexed independently as new keys arrive, without regrouping existing history. Within each bucket, it organizes indexing keys into capacity-balanced directional groups and packs a fixed number of complete groups into each page, retaining a separate center for every group. These centers provide fine-grained representatives for page selection, after which refinement scores members of the selected pages together with sink, sliding-window, and visible unindexed candidates to select the final Top-$K$ logical positions. To execute this retrieval process efficiently, we store members in reordered pages, fuse construction stages according to group size, and share storage for overlapping fixed candidates across queries during chunked prefill. These choices support regular member access while reducing kernel launches during construction and repeated candidate transfers. Moreover, we integrate LRSA into SGLang to maintain the index during inference, overlapping asynchronous index construction for new buckets with subsequent computation and reusing compatible bucket indices and member pages after a prefix match in the original key-value pool. With a 1024K-token context, LRSA accelerates key selection by 4.70$\times$ over DSA and combined selection and attention by 35.60$\times$ over full attention. 

---
# CoPyro: Feeding Copy Proposals to Parallel Drafters 

**Authors**: Hong-Kai Zheng, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.27217)  

**Abstract**: Speculative decoding accelerates language generation by letting a target model verify several draft tokens in one forward pass. Earlier neural drafters such as EAGLE extend each candidate sequence one token at a time. Recent work has turned to parallel drafters such as DFlash, which predict a whole block at once. Another approach uses n-gram retrieval to find an earlier occurrence of the current context suffix and copy the tokens that followed it directly into the draft. We introduce CoPyro, which feeds these copied tokens into a parallel drafter in place of masks. The drafter learns from context which tokens to keep and which to revise as it predicts the block in its usual forward pass. Across eight benchmarks on Qwen3-4B, CoPyro improves overall acceptance length for all four parallel drafters tested, with gains of 2.19-5.08% when trained on 100K samples. With DFlash trained on 710K samples, overall gains range from 1.08% to 2.00% across the tested model sizes and temperatures. Code generation benefits most in both settings. On Qwen3-8B, processing the proposals adds an average of 0.24 ms per decoding round in Transformers. 

---
# Correct Precedents Induce Policy Violations in On-Device LLM Agents 

**Authors**: Chong Tian, Yu Wang, Steve Liu, Qirong Ho, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.27216)  

**Abstract**: Local-cloud agent systems pair a cloud model with a small on-device agent that acts on private data, often as the last check before an irreversible action. We identify, for the first time, a failure in which such an agent is led into a violation by its own correct decisions within a session. After it correctly releases an item whose restriction was lifted, it becomes more likely to release the next item that is still restricted. Because the trigger is a correct decision, a history with no faulty step can still end in a violation. We call this failure correct-precedent generalization and find it in 13 open-weight models from nine developers. A single correct precedent can cut the safe rate of the most affected agent roughly in half, and the risk compounds when a refused request is simply sent again in a new session. Interventions on the agent's history indicate that it carries its decision on the released item over to the restricted one instead of checking that item's status. Building on this finding, we propose Paired Decision Distillation (PDD), which distills the decisions of a weakly affected teacher into a LoRA adapter. PDD restores the safe rate to nearly 100% in every distilled model, without extra prompt text or model calls. 

---
# Explicit Style Extraction for Personalized Long-Form Text Generation 

**Authors**: Jenitha Reddy Patel, Sameeksha Kini, Amulya Rayasam, Anisa Dye, Abhinav Sharma, Sai Karthik Navuluru, Bo Ni, Li Li, Junda Wu, Tong Yu, Nedim Lipka, Xiaolei Huang, Tyler Derr, Yue Zhao, Zhengzhong Tu, Yu Wang, Hongjie Chen, Thien Huu Nguyen, Nesreen K. Ahmed, Franck Dernoncourt, Ryan A. Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2611.27211)  

**Abstract**: Personalized long-form generation asks a language model to write new text the way a specific user would. The standard recipe, retrieval-augmented generation, places a few of the user's past texts in the prompt but leaves the model to decide what to take from them. We introduce StylePrism, a taxonomy of 28 stylistic aspects in seven categories, and SP-Refine, an inference-time method that makes style extraction an explicit step: the model first fills StylePrism with the user's patterns, quoted evidence, and habits to avoid, then drafts from the user's samples and revises the draft twice against this profile, guided by its own style scores. On the three long-form tasks of LongLaMP with three LLMs, SP-Refine outperforms LongLaMP's retrieval-augmented baseline in 25 of 27 ROUGE and METEOR comparisons, by 10.2% on average and by up to 35.8%. Annotating users' own texts with the extracted profiles shows that StylePrism captures concrete, checkable habits, from the order of rhetorical moves to a reviewer's sign-off. Ablations show that the profile helps as revision guidance, not as a replacement for the samples: writing from the profile alone is worse in all 27 comparisons, while a single implicit-inference call is a cheaper alternative that comes close. Oracle variants that let the reviser see the reference, together with reference-free controls, show that larger refinement gains require information unavailable at inference time. 

---
# How Much of a Calculation-Error Rate Is Not Arithmetic: Auditing the FinReason Cup 2026 Task 3 Metric with a Deterministic Rule System 

**Authors**: Vladimir Beskorovainyi  

**Link**: [PDF](https://arxiv.org/pdf/2611.27192)  

**Abstract**: Task 3 of the FinReason Cup 2026 scores a system on two values per case, the one an XBRL filing reports and the one its own calculation relationships imply, and labels each failure as structural, extraction or calculation. This report shows that the calculation-error rate so defined does not count arithmetic failures, and bounds how far it is from one using the published board alone. One of the three rules behind the benchmark, DQC_US_0015, has an answer that is the negation of the reported value (in all 110 published practice cases, against 1 of 120 and 6 of 102 for the other two), so a calculation label on such a case records a wrong fact or a wrong route, not a wrong sum. Subtracting the available structural and extraction labels from each entry's error count on that rule gives a floor: the bound is positive for 22 of the 23 ranked entries, and at the top it is sharpest, with at least 11 of 12 and 12 of 13 calculation errors of the two joint-first entries falling on cases where no arithmetic remained. The calculation-error rate of the best systems on this task is almost entirely not arithmetic.
The vehicle for that result is an entry that reconstructs the three data-quality rules the benchmark is drawn from, DQC_US_0015, DQC_US_0117 and DQC_US_0126, as a deterministic system with no learned parameters, no training step and no external service, running all 680 cases in 1.4 seconds. It scores 96.62% under the offline judge on development and 95.29% on the published test set, and its residual error concentrates in cases where a concept carries several equally flagged facts and the question names no dimensional member. Reading those error classes against the per-rule accuracies is what exposed the metric, which is why both are reported here. 

---
# Co-Evolving Behavioral Agents and Detectors for Adaptive Information Operations 

**Authors**: Lin Tian, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27156)  

**Abstract**: Detectors of state-sponsored information operations (IOs) are trained on campaigns that platforms have already taken down. They lose much of their accuracy on an actor they have not seen, even though no operator in the archived data is trying to avoid them. A real operator wants two things, to reach an audience and to avoid being caught. We present IO-AI, a framework that gives simulated agents these two capabilities, engagement and evasion, and tests what each one does to detection. The agents are built from the posting behavior of real campaigns and choose the emotion, type, and timing of each post. An agent and a detector then train against each other in turn. We rank the trained agents by how hard they are to detect and train one detector on them from easiest to hardest. On 20 Twitter IO campaigns, this detector reduces missed detections of adaptive agents by 65% relative to a detector trained only on the archived campaigns, at a similar false-alarm rate. On actors it has never seen it reaches an F1 of 0.81, against 0.62. We find that an agent trained to seek engagement becomes easier to catch, because it learns the fast, emotional retweeting that archived campaigns were caught producing. An agent that was trained for engagement before the game keeps its engagement lead after it learns to evade. We also introduce a behavioral distance metric that compares two groups of accounts using only their unlabeled posting histories. The larger the distance between two groups, the worse a detector trained on one does on the other (Spearman rho = -0.58), so the metric can be used to choose which new groups to test first. 

---
# Do Senior Agents Do Their Job? Measuring Role Integrity in How LLM Agents Use Subordinate Material 

**Authors**: Jinxiang Xie, Xinyi Zhang, Changjiang Jiang, Kexu Liu, Qi Yang, Yuxin Su  

**Link**: [PDF](https://arxiv.org/pdf/2611.27139)  

**Abstract**: Multi-agent LLM systems increasingly divide long tasks among subagents whose reports a senior agent turns into the final answer. Such a division of labor pays off only if each agent, and above all the senior agent, carries out the duties of its role. Current evaluations, however, mostly score the final answer, and a fluent, well-cited answer cannot show whether the senior agent weighed its subagents' reports or merely aggregated them. To assess whether a senior agent meets its assigned responsibilities, we define role integrity as the preservation of an agent's functional responsibilities when it acts on its subordinates' material. To measure it, we introduce RI-Bench, a behavioral benchmark grounded in the structures and public records of three institutional hierarchies: peer review, appellate adjudication, and parliamentary inquiry. We choose these hierarchies because their rules explicitly assign the senior role a duty toward subordinate material, and together they cover three basic duties: to weigh, to verify, and to relay. Rather than scoring output correctness, we measure how senior agents' outputs respond to controlled changes in subordinate material, using rule-based checks and a rubric-guided LLM judge calibrated against human annotation. Across eight LLMs and over 12,000 agentic episodes, senior agents rarely adopt a planted procedural error (2.9% of rulings) but often follow a majority of generic reviews (up to 71% of decisions) or drop witnesses' stated limitations (up to 65% of cited claims). Obligation prompts reduce these failures without removing them, and they also shift behavior in the control conditions. These findings motivate direct, obligation-specific evaluation of role integrity rather than inferring it from general capability or explicit role instructions. Code and data are available at this https URL. 

---
# Cross-Platform Production Signatures of State-Linked Information Operations 

**Authors**: Lin Tian, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27137)  

**Abstract**: An information operation active on several platforms presents different names, personas, and topics on each. Its working routine is harder to change. We call this routine the operation's production signature: the regularities in when and how its accounts post, reply, and repost. We measure it from timestamps and coarse action types only, using no content, profile metadata, or network ties, as timing and action profiles for accounts and learned action-gap policies for populations. We then ask which parts of a signature carry over to another platform and which belong to a single operation. We train models on Twitter archives of four state-linked operations and apply them to Internet Research Agency (IRA) accounts on Reddit, with no Reddit training labels. They rank the IRA accounts at AUC 0.90-0.94. Content detectors on the same route fall from 0.98 within a platform to 0.81. Daily phase accounts for most of the transfer. Posting volume reverses direction between platforms, and action mix follows the role accounts hold on each platform. What transfers is not unique to the IRA: a second operation shares the same working-hours window, and action-gap policies separate every operation from organic accounts (AUC 0.96-1.0) without consistently ranking their own operation first. On 938 Telegram channels, groups that forward the same sources keep ordinary hours and pace but publish original posts in the same ten-minute windows more often than chance. This pattern is invisible when accounts are examined one at a time. Our results suggest that timing and action traces learned on one platform can help flag likely operations on another. 

---
# Separating Automation Gains from Agent Contributions in AI-Assisted Model Validation 

**Authors**: M. Ahmadi Achachlouei  

**Link**: [PDF](https://arxiv.org/pdf/2611.27094)  

**Abstract**: LLM agents are being proposed for bank model validation, yet evaluations that compare agent-assisted work with manual practice credit the agent with gains that come from digitized evidence and deterministic tests. We propose a staged architecture in which an LLM-free controlled workflow (P1) is both the fallback and the control arm for individually authorized agent tasks (P2), and an attribution protocol that credits an agent only for faults P1 does not already report. The protocol charges decoy false alarms, unnecessary actions and endorsed developer assertions, counts invalid outputs as misses, folds newly automatable findings back into P1 with discovery credit, and fixes safety bounds and a break-even cost gate in advance. On a synthetic credit-scorecard harness, five scripted reference policies show why this matters: naive case accuracy (4/6) and pooled detection (3/5) tie four flawed policies, whereas the protocol separates all five and credits only the reference policy with detection beyond P1. One fold-back round raises a P1-parroting policy's naive scores without any change to the policy. A regression matrix of sixteen mutants covers nine ways in which earlier versions of the evaluation passed misleadingly. No LLM agent has been evaluated yet; we specify a falsifiable pilot. 

---
# How Far Does Lossless Re-tokenization Move an RLVR Update? 

**Authors**: Zhe Dong, Fang Qin  

**Link**: [PDF](https://arxiv.org/pdf/2611.27083)  

**Abstract**: Language models assign different probabilities to token sequences that decode to the same string. In reinforcement learning from verifiable rewards (RLVR), a trainer that re-encodes sampled text may train on a different token sequence with the same reward. We ask how far this moves a training update, and whether the movement has a consistent direction. From one checkpoint, we compare updates on the same rollouts, canonical and with selected tokens split into two pieces, against an update on a different batch. On Qwen2.5-0.5B with three-digit addition, splitting about one token in 25 moves the update 0.556 times as far as replacing the batch. The ratio rises with the fraction split in four models and depends on which tokens are split; within a checkpoint, the displacements share a direction across batches. In a tabular model with one table per tokenization and a single sampled string, we prove that whenever two strings below an unsampled prefix differ in probability, only the zero full-gradient step agrees across tokenizations, however it is weighted. In an audit that decodes and re-encodes resampled rollouts, re-encoding touches up to 5.2% of rollouts but at most 0.08% of tokens, far below the split fractions we test. Training on fixed rollouts, the canonical arm finishes ahead in 14 of 18 pre-registered pairs; the interval for the mean accuracy difference includes zero over all pairs and excludes it only under the registered exclusion rule, which favours the canonical arm. 

---
# When Do LLM-Elicited Priors Help? A Controlled Audit of Language Models as Priors for Small-Sample Bayesian Regression 

**Authors**: Ran Zou, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27075)  

**Abstract**: Large language models (LLMs) are increasingly asked where to center the coefficients of a small-sample Bayesian regression, but the evidence is unclear: benchmarks are often built to match the prior and models may remember them. We audit this practice with one frontier and one flash-tier model, using hidden feature names, post-cutoff housing data and a pre-specified test that gives features the wrong descriptions. First, the models share their errors (correlation $0.92$--$0.99$ across draws, vendors and paraphrases): averaging $20$ draws is no more accurate than one, and treating draws as independent creates overconfidence. Much of this error comes from reporting each feature's effect alone, not with the others held fixed. Second, against the best classical prior, the LLM center helps clearly on one of four benchmarks (RMSE $-5.8\%$), marginally on two, and hurts mean RMSE on the fourth. Third, the LLM's stated uncertainty is unreliable; letting the data set the prior scale cuts the harm of a wrong center four-fold against a unit-variance prior. A pre-specified external test returned the verdict \textsc{partial}: on $18$ rule-selected problems the center helps on six and hurts on four, and a hard trust gate trusts a harmful prior while an evidence-weighted mixture is safer. 

---
# NightmareDiscourseBench: A Benchmark Dataset for Fictional Direct Discourse Identification Pipelines 

**Authors**: Claudia Carroll, Gabi Kirilloff, Dashiell Leigh  

**Link**: [PDF](https://arxiv.org/pdf/2611.27059)  

**Abstract**: In this paper we introduce a new benchmark dataset, NightmareDiscourseBench, for testing NLP pipelines that identify forms of direct discourse in English-language fiction. We first review existing publicly available benchmarks (providing summary statistics and reviewing available metadata) for this task, identifying theoretical inconsistency and exclusive focus on speech as primary issues in this domain. We then overview our narrative theory-based framework for classifying and evaluating different forms of direct discourse, and finally describe NightmareDiscourseBench, a new benchmark test set of 12 text samples accompanied by metadata pertinent to the task of direct discourse identification. 

---
# Isolating the Language Shell: A Content-Controlled Study of Multilingual Reasoning 

**Authors**: Hoejoon Kwon, Jinhee Jang, Byeonggeuk Lim, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.27053)  

**Abstract**: Multilingual reasoning distillation requires choosing the language in which teacher reasoning is presented, yet existing recipes often entangle linguistic form with differences in generation, translation, and filtering. We construct content-controlled supervision by rendering the same verified English trajectories into English, translated target-language, and language-mixed shells, and fine-tune two pretrained students in three target languages, evaluating them on multilingual mathematical reasoning at two difficulty levels. Even with the underlying reasoning content held fixed, we find that shell choice can change downstream accuracy, with effects varying across students, languages, and difficulty levels. Pre-training compatibility does not consistently track these outcomes, and even when our tests find no clear accuracy difference between shells, their output-language behavior can still differ substantially. These results suggest that the supervision shell is a distinct design choice in multilingual reasoning distillation. 

---
# Beyond Prompt Heuristics: Assessing Child Safety Alignment in LLMs 

**Authors**: Anika Misra, Chimaobi Okite, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2611.26965)  

**Abstract**: Child-safety taxonomies typically characterize safety risks through a predefined set of harm categories, and existing evaluations largely operationalize these categories using prompts that explicitly invoke them. However, child-safety considerations can also arise in seemingly benign requests that lack recognizable harm-category cues. We therefore investigate how child-safety alignment generalizes to such settings. Across six models, we find considerably higher misalignment rates on requests without recognizable harm-category cues than on those containing such cues (80.0, 54.0, and 48.4 percentage-point gaps for Qwen3.5-27B, GPT-5.4-mini, and Claude-Haiku-4.5, respectively). Through controlled behavioral and mechanistic analyses, we find that introducing a single harm-associated word reduces misalignment across most models, with mechanistic evidence suggesting that child-safety reasoning may rely in part on lexical and semantic associations with words in the prompt. Our findings motivate evaluations that treat child safety as context-dependent, and not reducible to a fixed set of harm categories. 

---
# Partially canonical axes in large language model representations 

**Authors**: Andrea Gregor de Varda, Ferdinando D'Elia, Greta Tuckute, Evelina Fedorenko, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2611.26920)  

**Abstract**: Large language models (LLMs) develop rich internal representations of language, but whether their individual coordinate axes carry meaning, or whether meaning is encoded only in directions in representation space, is still an open question. A common assumption is that individual axes in LLMs are arbitrary. We here challenge this view: across 12 LLMs from six model families, the native axes are reliably more aligned with other models' representations, with human brain responses to language, and with interpretable linguistic features than randomly rotated axes that preserve the same geometry. The native axes nonetheless fall well short of the best achievable alignment, so LLM coordinates are not fully arbitrary but also not fully canonical: they are partially canonical. A theory of how LLMs represent language must explain both the population geometry they learn and why individual units acquire patterns of response profiles that recur across systems. 

---
# Modeling Attributional Style at Scale: A Dataset and Analysis for Psychological Attribution Assessment and Reframing 

**Authors**: Qiang Zhou, Hanzhen Zhu, Pan Wang, Rui Tu, Huaizhi Qu, Zhuoran Wang, Xin Hu, Lei Li, Tianlong Chen, Jingtong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26900)  

**Abstract**: According to the reformulated Learned Helplessness theory, repeated exposure to uncontrollable negative events can foster a depressogenic attributional style, increasing susceptibility to depression yet remaining a tractable target for cognitive therapy. Computational research on attributional cognition, however, is hampered by the lack of large-scale datasets and robust evaluation protocols. In this work, we introduce the Attributional Style Transfer Dataset (ASTD) along with dedicated evaluation metrics, the first benchmark designed to model, assess, and reframe attributional explanations at scale. Constructed via a Prevent-Filter-Validate pipeline that integrates LLM-based generation with specialist validation, ASTD contains 42,000 real-world events paired with psychologically grounded attributions spanning seven styles. Using this dataset, we address two key challenges: (1) scalable assessment of attributional style via both supervised classifiers and zero/few-shot LLMs; and (2) attributional reframing and evaluation, where we propose automatic evaluation metrics to quantify psychological validity. Furthermore, we leverage our proposed metrics to construct a preference dataset, fine-tuning LLMs with Direct Preference Optimization (DPO) and achieving substantial gains in reframing quality. Together, our dataset, metrics, and methodology offer a new paradigm for understanding and modeling attributional style, with direct implications for scalable and adaptive mental health interventions. Code and data: this https URL 

---
# ProbeAgent: Principled Diagnosis of Agent Failures in Verifiable Environments 

**Authors**: Yue Huang, Yuchen Ma, Shuhang Lin, Yuyang Wu, Zixuan Weng, Zixiang Xu, Xiangqi Wang, Yuexing Hao, Yujun Zhou, Yue Zhao, Olexandr Isayev, Stefan Feuerriegel, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26897)  

**Abstract**: Improving agents requires discovering where they fail and understanding why. An agent acts over multiple steps in a changing environment, so diagnosing its failures requires more than inspecting the final response of a large language model (LLM). The same unsuccessful outcome can arise from task difficulty or from an error the agent repeats across tasks. We introduce ProbeAgent, an automated framework that discovers agent failures, tests their proposed causes, and identifies shared failure modes. Starting from benchmark tasks, ProbeAgent builds executable tests and validates them using reference solutions, automated checks, and review of the task requirements. For each failed trajectory, it explains the error and constructs new tasks that preserve the suspected cause while changing the surrounding context. Running the agent on these tasks tests whether the same error recurs. ProbeAgent then groups failures with similar causes and generates descriptions that distinguish their errors. This process connects explanations of individual failures to executable tests and descriptions of shared failure modes. We evaluate ProbeAgent across 22 agent configurations using tasks from 10 benchmarks, collecting 1,244 failures and producing 536 failure-mode descriptions. These descriptions distinguish specific errors within broad outcomes such as incomplete work, including planning without acting and stopping before delivering a file. In human evaluation, annotators accept 90.0% of assigned descriptions and correctly reject 83.3% of similar alternatives, supporting the accuracy and specificity of the discovered failure modes. 

---
# Your Judge Inherits Your Prompt: In-Context Label Bias in an LLM-as-Judge Is Directional, Not Symmetric 

**Authors**: Abdullah Ahmad, Ahmed J. O. Ahmed, Luca Citi  

**Link**: [PDF](https://arxiv.org/pdf/2611.26881)  

**Abstract**: LLM-as-judge has become a standard way to score open-ended model outputs, yet the judge is itself a frozen language model steered by a few in-context examples, and such models are known to absorb the label balance of their demonstrations rather than treat them as neutral illustrations. We ask whether this label bias reaches an LLM judge used inside a multimodal (video question answering) evaluation pipeline, and what form it takes. Using a fully frozen generator-judge pipeline, we hold the generated answers fixed and vary only the Yes/No balance of the judge's few-shot examples across four demonstration sets. Each verdict is validated against a two-annotator human consensus on 200 examples per dataset. The judge's acceptance rate stays close to human consensus when its examples are balanced or skewed toward rejection, but jumps sharply (by about 30 points on NExT-QA and 20 on MSVD-QA, with a strongly one-sided error profile) when the examples are skewed toward acceptance; the opposite skew produces no matching drop. The bias is therefore one-directional within the tested range of label ratios. A separate five-draw analysis shows that results from a single example draw are unreliable, varying by up to 18 points, more than any difference between prompting conditions. We conclude that demonstration label bias in an LLM judge is a directional, inflation-only failure mode that can fabricate apparent benchmark gains but not losses, and that it is cheaply avoided by balancing the judge's examples and reporting results averaged over several draws. 

---
# How Long Will My Language Model Run? A Benchmark for Pre-Execution Runtime Prediction 

**Authors**: Ashish Molakalapalli, Kangda Wei, Haonan Duan, Alex Carter, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26875)  

**Abstract**: Researchers submitting deep learning jobs to shared GPU clusters need runtime estimates to request time limits, choose hardware, and budget compute. Large language models (LLMs) offer a tempting shortcut: describe the job and ask. We introduce PreRun, a benchmark for predicting deep learning training and inference runtime from information available before execution. PreRun contains 801 training and 1,884 inference records of language and vision models collected on a GPU cluster, paired with workload configurations and explicitly defined timing targets. It also includes 64 text-generation configurations measured under standardized conditions, each in three separate program launches. We compare three LLMs (Qwen3.5-9B, GPT-OSS-120B, and Claude Opus 5.5) with regression baselines and evaluate analytical estimates on a supported forward-pass subset. On 16 held-out generation configurations, the open-weight LLMs underestimate latency by 90-99% on average without examples. Claude's zero-shot mean absolute error is 4.5 times that of a Random Forest fit on 32 configurations. Five labeled examples substantially reduce Claude's generation error, but their benefit varies across timing targets. We will open-source our data and evaluation pipeline upon acceptance. 

---
# How Language Models Compress and Compare: Understanding Selection with Token Covariance Maps 

**Authors**: Moshe Eliasof  

**Link**: [PDF](https://arxiv.org/pdf/2611.26874)  

**Abstract**: Many language-model tasks reduce to selecting among structured alternatives, yet we still lack a simple picture of how a decoder-only transformer organizes an entire selection prompt across depth. We introduce token covariance maps, a prompt-level geometric view that tracks how token positions co-vary in hidden-state space at each layer. This turns a structured prompt into a depth-indexed sequence of maps whose spectral and block structure can be aligned with instructions, context, and candidate answers. Across 14 decoder-only models from 4 families and 12 structured-selection benchmarks, token covariance maps reveal a robust compress-then-compare organization. Early layers represent prompts diffusely, middle layers compress them into a shared low-rank bottleneck, and late layers re-expand into context-option and option-option geometry. This organization recurs across larger fixed-template samples, deterministic prompt rewrites, pairwise reranking reformulations, and controlled answer-set manipulations, while also emerging during training from an initially flat depth profile. Its timing aligns with behavior: useful answer discrimination appears well after the bottleneck, and low-rank intervention around the bottleneck preferentially disrupts late comparison geometry. We additionally show that the same geometric profile appears in instruction-tuned models, across several languages, and during open-ended generation. We further show that changing the internal representation of one candidate around the compressed region can shift which candidate the model later prefers. These results show that token covariance maps are a diagnostic tool for comparing how models organize information across prompts, model families, and stages of training. 

---
# LLMs Memorize Their Own Hallucinations During Post-Training 

**Authors**: Anirudh Atmakuru, Tanya Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2611.26863)  

**Abstract**: Reasoning domains have shaped the post-training recipes of frontier models even though they are trained and deployed in domains beyond math and code. In this paper, we study the consequences of directly adopting reasoning-centric training recipes, e.g., the use of outcome-based rewards for reinforcement learning (RL), injection of "cognitive" behaviors such as proposing and backtracking from hypotheses in reasoning chains, etc., for knowledge tasks. We hypothesize that these decisions, while useful for reasoning, inadvertently encourage models to memorize hallucinations they themselves generate in their long reasoning chains when trained on knowledge tasks. To study this, we train models with standard post-training, extract their self-generated hallucinations during RL, and track memorization throughout the rest of training. Our experiments on six open models across two model families and two architectures confirm our hypothesis. We find that models' recall of hallucinations reinforced during training increases by 4.83% on average, compared to just 0.17% for self-hallucinations that receive no training signal. In fact, models even memorize false facts that they immediately retract in their reasoning; a backtracking template inherited from reasoning-centric training that has no benefit for knowledge recall. Furthermore, we find that memorization is strongest on the hardest training questions, and rewarding a hallucination makes the model less likely to abstain from it later (by 7.23% on average). Overall, our work identifies and systematically studies a previously unexamined pathway through which models learn hallucinations during training. 

---
# Weight-Space Programming Writes In-Context Learning and Matches Fine-Tuning in Language Models 

**Authors**: Konstantinos Skianis, Aristidis Likas  

**Link**: [PDF](https://arxiv.org/pdf/2611.26841)  

**Abstract**: Weight-space programming is a frontier language model's recently reported ability to read and rewrite another network's weights without gradient descent. We investigate how examples can provide a weight update so that a weaker language model performs the task with nothing in the prompt, and study what it edits. Our experiments cover six tasks, four frontier editors and three families of weaker language models, without the editors being told anything about in-context learning (ICL). In most runs, the few-shot minus zero-shot residual is measured at the final prompt token, the ICL task vector, and write it into the weights, at its sparsest into near-silent MLP units keyed to the prompt format. Results reach the accuracy of ICL and of fine-tuning on the same examples, LoRA included, usually leaving perplexity untouched, with worse performance under changes of wording. We also qualify the comparison by counting unfinished runs, observing that one editor falls below fine-tuning; held only to the seen examples, without task pairs of their own, only the strongest editor has similar performance. Given a task and unlabelled inputs, that editor writes its own examples and does nearly as well on unseen items as with labels. 

---
# Cross-Linguistic Distinctive-Feature Constraints on Segmental Triphones: Evidence for Contrastive Dispersion 

**Authors**: Leonardo Araújo  

**Link**: [PDF](https://arxiv.org/pdf/2611.26837)  

**Abstract**: Natural speech reflects articulatory and perceptual constraints, yet how these pressures shape consecutive segment transitions across languages remains underexplored. We analyze adjacent distinctive-feature distances in token-weighted segmental triphones across 17 Universal Dependencies corpora spanning nine language families, transcribed into IPA and mapped to 24 PanPhon features using Hamming and cosine metrics. Natural sequences exhibit an intermediate-distance mode (8-10 features under Hamming, except Vietnamese; 0.28-0.38 under cosine). Against an unrestricted within-word shuffled baseline, natural words reject the inventory-only null (W = 0, p < 10-4), showing higher expected distance, narrower dispersion, and lower joint triphone entropy in all 17 languages. Under a stricter within-word consonant-vowel template baseline, expected distance and dispersion are no longer separated, whereas joint entropy remains lower in 16 languages (p = 1.07x10-4). Entropy decomposition reveals higher mutual information between incoming and outgoing transitions in all 17 languages and all nine families (W = 0, p < 10-4), demonstrating sequential coupling beyond syllable-template alternation. Word-level mean-SD correlations and directional asymmetries do not replicate cross-linguistically. 

---
# Non-Expanding Gated Continued Fraction Architecture for Feedforward Layers in Language Models 

**Authors**: Amit Dhurandhar, Vijil Chenthamarakshan, Karthikeyan Natesan Ramamurthy, Dennis Wei, Rahul Nair  

**Link**: [PDF](https://arxiv.org/pdf/2611.26835)  

**Abstract**: Recent work has shown that continued fraction inspired neural architectures (CoFrGeNets) can be highly parameter efficient yet performant for language modeling. In this work, we revisit continued fraction-based channel mixing with the goal of improving its performance and simplifying integration into standard pre-training protocols for language models. We achieve this by two simple yet critical contributions: i) we propose a novel non-expanding gated continued fraction inspired architectural component to replace feedforward layers in established language model architectures that may use vanilla multilayer perceptrons or gated linear units or (vanilla/gated) experts in mixture-of-experts (MoE) type architectures. ii) We propose a way to re-implement divisions -- the non-linearity in these architectures -- in terms of logarithms and exponentials. By experimenting on different language modeling architectures such as GPT-2, Llama-3, Mamba-2 and (nano) Llama-MoE we find that our proposed architecture performs similarly or significantly better than the continued fraction architecture proposed in prior work along with having higher throughput. In addition, our re-implementation of divisions makes training of deeper CoFrGeNets stable without the need for incremental training which was proposed in prior work as a means to stabilize training, but added notable complexity to existing large language model training implementations. Code is available at this http URL 

---
# Rethinking Full-Weight LoRA with Domain and Task Adapter Scaling for Question Answering over Portuguese Well Drilling Reports 

**Authors**: Luan B. Guerra, Diogo M. Paes, Marcelo N. Grimberg, Arnaldo C. Junior, Wallace Casaca, Lucas C. Ribas  

**Link**: [PDF](https://arxiv.org/pdf/2611.26833)  

**Abstract**: Adapting large language models (LLMs) to specialized domains remains challenging, particularly for technical documents such as Portuguese well drilling reports, where annotated data are scarce and domain adaptation may compromise instruction-following capabilities. Although Low-Rank Adaptation (LoRA) is widely used for parameter-efficient adaptation, its scaling after training remains insufficiently explored. This study investigates the impact of LoRA adapter scaling on question answering over Portuguese drilling reports using Qwen2.5-7B and Llama-3.1-8B. Domain-adaptive pretraining (DAPT) is performed either directly on instruction-tuned models or on their base counterparts, with subsequent adapter transfer, while task-specific adapters are trained on WellQuest, an open-source question-answering dataset derived from Portuguese drilling reports. The two adapters are also combined using complementary weights. Scaling factors ranging from 0.1 to 1.0 are evaluated across multiple training checkpoints using BERTScore. Results show that fractional scaling consistently improves DAPT performance over full-weight application, increasing F1 from 68.8\% to 71.9\% for Qwen2.5-7B and from 63.2\% to 68.5\% for Llama-3.1-8B. QA adapters perform best at high but fractional weights, while base-to-instruct transfer yields better results than direct DAPT on instruct-tuned models. Adapter merging provides a modest improvement for Llama-3.1-8B. Overall, these findings demonstrate that post-training adapter scaling can improve domain adaptation without additional data or training. 

---
# Neural Geometry as a Learning Signal 

**Authors**: Eric Todd, Tal Haklay, Nikhil Prakash, Matthew Finlayson, Michael Hanna, Fenil R. Doshi, Daniel Wurgaft, Thomas McGrath, Owen Lewis, David Bau, Jack Merullo, Ekdeep Singh Lubana, Matthew Kowal, Thomas Fel, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2611.26832)  

**Abstract**: We investigate whether the geometric structure of neural representations can provide a sufficient learning signal for improving model behavior without any supervision on model outputs. First, we fine-tune Llama-3.1-8B to answer questions such as "What month is four months after June?" by supervising internal representations of the input month (June = 6) and offset (4), which are encoded with Fourier harmonics. Second, we fine-tune the model to predict line breaks in text wrapped to a fixed width by supervising an internal representation of the number of characters since the preceding newline, which is represented by a low-dimensional helical structure. In both tasks, we supervise a low-dimensional subspace of the residual stream using an idealized target structure, and find that internal geometric supervision achieves similar performance to standard fine-tuning (despite no supervision on outputs). Moreover, we find that simply patching in an ideal geometry is not sufficient to recover fine-tuning performance, and thus demonstrate geometric supervision provides beneficial effects beyond the representations it targets. Together, these findings show how mechanistic understanding of neural geometry can be leveraged to shape the behavior of neural networks. 

---
# Search From the Start: Search-Native Pretraining for Language Models 

**Authors**: Kaustubh Ponkshe, Gyung Hyun Je, Stefan Taga, Colin Raffel, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2611.26809)  

**Abstract**: Modern language models (LMs) are increasingly deployed with access to search tools at inference. Yet pretraining remains predominantly focused on predicting static text, with search capabilities typically introduced during post-training. We propose Search-Native Pretraining as a practical route to teach search capability during pretraining from ordinary text corpora. Specifically, we augment pretraining corpus at scale with web-search queries and retrieve web response using a local index. Training on this augmented corpus teaches models to generate search calls and incorporate retrieved evidence during language modeling. Across Wikipedia and DCLM-Edu corpora and 360M to 3.3B parameter scales, search-pretrained models ("QueryLM") show improved performance on knowledge-intensive tasks and higher steerability toward the retrieved evidence. Crucially, their advantage persists after supervised fine-tuning of both search-pretrained and conventionally pretrained models, indicating that the tested post-training procedures do not fully recover the capabilities developed during pretraining. Throughout our experiments, we find that a 1.4B QueryLM consistently beats or matches the performance of a FLOP-matched 3.3B baseline model across all knowledge intensive tasks. These results establish search-pretraining as a practical way to develop foundational search capabilities during pretraining. We release the training artifacts, including model checkpoints and data. 

---
# MedicalRSI: Medical Recursive Self-Improving Agents via Clinical Model-Harness Co-Evolution 

**Authors**: Ziqing Wang, Hamid Palangi, Lili Zhao, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2611.26806)  

**Abstract**: Medical agents built on large language models (LLMs) have advanced rapidly, yet most stay fixed once deployed and do not keep using their experience to improve later attempts. As a result, the same kind of error can recur across patients. To address this, existing self-improvement methods update an agent's weights or its medical harness from its own experience, but their improving strategy stays fixed and never learns which changes actually help, which caps how much the agent can improve. Recursive self-improvement (RSI) instead improves both the agent and its improving strategy. However, achieving RSI for medical agents is challenging, because clinical workflows rarely come with a reference harness, and a change that fixes early errors may not fix the problems the agent meets later. We propose MedicalRSI, an RSI framework built on Clinical Model-Harness Co-Evolution, in which one model plays two roles. The Solver is the medical agent itself, and the Improver is its learnable improving strategy, which writes a medical harness from the agent's failures. Each harness is rewarded by its gain over acting without it, and both roles are trained together, so the Improver keeps adapting to the problems the Solver has not yet solved. On multiple benchmarks and LLM backbones, MedicalRSI outperforms existing context-level and weight-level self-improvement methods and transfers well to unseen workflows, new LLMs and new databases, a step toward medical agents that keep improving. 

---
# What Helps Language-Model Agents Review Violent Video? A Controlled Comparison with a Human Evidence Reference 

**Authors**: Mohammed El Amine Mokhtari, Matei Mancas, Virginie Vandenbulcke  

**Link**: [PDF](https://arxiv.org/pdf/2611.26776)  

**Abstract**: Systems that flag violent video for a human operator can now be built from language models without training, by combining perception models and letting a language model decide. We ask how the observations of the perception models should reach the decision, and which evidence matters. On the XD-Violence test set (799 clips) we fix one set of perception agents (video description, object detection, sound tagging, speech recognition and speech tagging) and one reasoning model, and compare three interfaces: messages between agents, a typed evidence graph explored with tools, and flat text read in one pass; and a single audio-visual model given the raw clip. Two annotators provide a human reference that records the decision, the incident segments and whether they relied on sight, sound or speech. Flat text matches the single model (F1 0.967 against 0.968), while the graph agent and message passing are less accurate; a variant of the graph agent developed on a separate set did not close the gap, and both graph summaries and multi-turn tool use lowered accuracy. These results hold with reworded prompts and with a larger reasoning model, except that the larger model weakens message passing enough for the graph agent to beat it. Sound events improve F1 by +0.012 [+0.004, +0.021], in line with the annotators, who used sound or speech on most violent clips. Citations of stored observations are faithful, changing about half of the decisions when removed, but no more so with the graph than with flat text, and the agent pipeline costs about 3 times the GPU time of the single model. We also report two artefacts of vision-language describers that mislead pipelines built on them. 

---
# Know Your Context Bank: Repairing Inconsistency and Measuring Generalization in Context-Aware Text-to-SQL 

**Authors**: Hailong Li, Mohammadreza Pourreza, Tal Neiman, Reza Sherkat, Per Jacobsson, Yannis Papakonstantinou  

**Link**: [PDF](https://arxiv.org/pdf/2611.26775)  

**Abstract**: Text-to-SQL systems increasingly rely on context banks: curated collections of example queries, metric definitions, and domain knowledge that tell a model how an organization writes its queries. In the enterprise, a bank grows over years as many analysts add entries, checked one at a time for correctness (does the SQL answer its own question?) and rarely for consistency with one another. Yet correctness is often loosely defined: a question seldom fixes how an entity is counted or which table defines it, so two entries can each pass review yet resolve the same decision in opposite ways. We call this condition context bank rot. A second question follows: does a system that uses a bank generalize from it, or only reproduce it? Existing benchmarks cannot tell, as they do not control how far a test question sits from the context. We answer both questions by comparing queries decision by decision, with a fast scorer distilled from human-calibrated LLM judges. Turned inward, the comparison diagnoses and repairs a bank. Fine-tuning on a cleaned BIRD-train bank beats a same-size random subset at every epoch (+2.6 points on average), and 92.4% of our removals had their SQL corrected by an independent expert audit, against 45.6% of audited entries. In-context examples that follow one convention beat those following the opposite one by 13.3 points, while a mix is barely better than none. Turned outward, the comparison generates test questions at a controlled distance from a bank, a pipeline we call CAT (Context-Aware Testing). On CAT-BIRD, for retrieval and fine-tuning alike, the benefit of a bank falls step by step with distance and is largely gone once about three quarters of a question's logic is new, and on CAT-Spider the benefit of retrieval likewise shrinks with distance. 

---
# Reinforcement Learning Unlearns Corrupted SFT Reasoning 

**Authors**: Anika Misra, Shreyas Vinod, Huteng Dai  

**Link**: [PDF](https://arxiv.org/pdf/2611.26756)  

**Abstract**: We show that reinforcement learning (RL) in reasoning tasks can succeed even when the supervised fine-tuning (SFT) data is severely corrupted. In a mathematical reasoning dataset (GSM8K), we systematically corrupt training trajectories across 361 configurations by injecting realistic error types, varying error locations, trace-answer consistency, and corruption ratios up to 100%. While aggregated SFT-only performance steadily deteriorates under corruption (dropping from 43.61% to 13.80%), Group Relative Policy Optimization (GRPO) post-training consistently restores model accuracy back to gold-baseline levels (approximately 60%). Crucially, initializing RL with as few as N = 8 minimal, incorrect, three-line "form-only" traces boosts post-RL performance to 69.30%, outperforming standard gold SFT initialization (60.71%) and GRPO without SFT (35.00%). Further, we show that SFT accuracy is largely uncorrelated with post-RL accuracy (r approximately 0), demonstrating a fundamental asymmetry in post-training: SFT primarily establishes structural layout (form), while RL independently discovers functional reasoning (function). 

---
# SkillPrivus: Uncovering PRIVacy of USe Harms in Agentic Skills 

**Authors**: Mariana Fernandez-Espinosa, Ivoline Ngong, Keerthiram Murugesan, Karthikeyan Natesan Ramamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2611.26747)  

**Abstract**: Agent skills extend AI assistants through installable instructions and code that can access users' files, accounts, and data. Even when performing their advertised tasks, skills may infer sensitive attributes, retain unnecessary information, or disclose data to unexpected recipients. We study these potential privacy-of-use harms through SkillPrivus, a static audit framework that analyzes skills before installation without executing them. SkillPrivus extracts data operations from code and natural-language instructions, grounds them in source evidence, and adapts Contextual Integrity to assess their appropriateness. Applied to 726 real skills from public marketplaces, 563 of which handle personal data, SkillPrivus flags at least one potential violation in 307 of these skills (54.5%), and its judgments agree with three blind human annotators on 83.2% of resolvable moves. The flagged behaviors include sensitive inferences, information use beyond the stated task, and disclosures to unexpected recipients, suggesting that privacy harms can arise from ordinary skill functionality. 

---
# TEA-Bench-BR: Can Coding Agents Turn a Branching Clinical Specification into Correct Code? 

**Authors**: Marcio Andre de Souza Estevam  

**Link**: [PDF](https://arxiv.org/pdf/2611.27895)  

**Abstract**: Coding agents increasingly implement rules written by domain experts, and bugs in such code are silent: it runs and returns a plausible answer. We introduce TEA-Bench-BR, a benchmark built on the Brazilian pathway for early autism (TEA) screening: three fictitious instruments with the structure of real ones, a multi-professional referral flow, and export to FHIR R4 with the national RNDS profiles, all specified in Portuguese. A generator samples each task's parameters (144 tasks, 6,798 hidden cases), and the hidden cases kill all 1,955 non-equivalent mutants of the reference. With Gemma 4 as the agent, the 12B solves 38.0% of single-instrument tasks and none at the FHIR level; the 31B solves 77.8%, 85.2% and 78.7% at levels 1 to 3 and passes over 99% of the hidden cases, its remaining failures mostly in input validation rather than clinical decisions. With the same tasks in English, the 31B solves 57.4% at level 3, as its code validates an identifier less often; the 12B's, more often. The 12B misses 10.10% of the cases that call for referral; the 31B, 0.02%. A contamination probe on a modified PHQ-9 finds no error that follows the original rules. 

---
# DistDNA: Language Model Fingerprinting under Stochastic Decoding 

**Authors**: Lifan Hu, Qian Yu, Zhaomin Wu, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2611.27831)  

**Abstract**: Behavioral fingerprints identify a large language model (LLM) from its responses to a shared set of prompts, enabling applications such as verifying whether a deployed model corresponds to the checkpoint it claims to be. Deployed models, however, are commonly served with stochastic decoding, under which the same model may produce different responses to an identical prompt, depending on an unknown temperature and top-$p$ setting. Existing representations such as LLM DNA provide limited treatment of this variation. A natural remedy is to characterize models by their response distributions, which, however, raises two challenges. First, each distribution must be estimated from only a small number of sampled responses. Second, a change in decoding setting can alter responses to the same extent as a change of checkpoint. Consequently, fingerprint distances alone provide an incomplete basis for model identity, and identification typically relies on a classifier trained jointly on fingerprints from many checkpoints. We propose DistDNA, which represents a model by its prompt-conditional response distributions and compares models by promptwise maximum mean discrepancy (MMD). The model under examination is assigned to the known checkpoint whose reference fingerprints, collected at several decoding settings, lie nearest to its own, without training a classifier. We show that finite sampling adds variance-dependent terms to the expected squared fingerprint distance, with these terms decreasing as $O(1/R)$ for equal response counts, where $R$ is the number of responses per prompt. Experiments across 115 model checkpoint entries demonstrate that DistDNA improves identification accuracy over the strongest baseline, LLM DNA, by 10.36%. 

---
# SLP-Bench: Evaluating and Improving Acoustic Claims of Reasoning Audio Language Models on Clinical Speech 

**Authors**: Côme Hosxe, Rachid Riad, Quentin Spinat  

**Link**: [PDF](https://arxiv.org/pdf/2611.27810)  

**Abstract**: Audio language models (ALMs) can answer clinical questions from speech, outperforming baselines built on handcrafted acoustic features. Reasoning ALMs generate a chain of thought that refers to properties of the speaker's voice before predicting. Each statement the model makes about the voice needs to be checked against clinicians' perceptual ratings of the same recording. On clinical speech, existing evaluations score only the final answer. We introduce Speech Language Pathologist Bench (SLP-Bench), a benchmark grading each acoustic claim in the chain of thought against clinicians' perceptual ratings of the same recording. It covers five clinician-rated datasets and 25 perceptual dimensions, and assesses claims for factuality, stability under resampling, and consistency across reworded and different questions. Across ten open-weight reasoning ALMs evaluated zero-shot, claim balanced accuracy ranges from 0.52 to 0.58, claims change under resampling and rewording, and the most consistent models are not the most factual. Fine-tuning a 7B ALM with GRPO on a reward for claims that agree with clinicians raises its claim balanced accuracy from 0.54 to 0.60 and its mean kappa from 0.39 to 0.61, while preserving answer accuracy on per-dimension questions. SLP-Bench is available upon request, subject to approval by the providers of each dataset. 

---
# Beyond Math, Code, and Logic: Do Reasoning Models Transfer to Scientific Argumentation? 

**Authors**: Buse Sibel Korkmaz, Antonio del Rio Chanona  

**Link**: [PDF](https://arxiv.org/pdf/2611.27722)  

**Abstract**: Scientific discourse depends on argumentative reasoning: identifying claims, evaluating evidence, and constructing coherent responses. While recent advances in reasoning-capable language models have demonstrated strong performance on mathematical, coding, and logical benchmarks, their ability to engage in scientific argumentation remains unclear. We present the first systematic evaluation of language models across seven tasks spanning argument mining, rebuttal generation, and discourse-level reasoning using research papers, peer reviews, and grant proposals. Our study reveals that even frontier models with strong general reasoning skills struggle with domain-specific argumentative tasks, highlighting a fundamental capability gap. To address this, we introduce a training framework that explicitly scaffolds the argumentative reasoning process in language models, substantially improving their competence in scientific discourse. The resulting compact models approach or exceed the performance of much larger proprietary systems and generalize to unseen conversational settings, demonstrating reasoning transfer beyond task-specific supervision. These findings underscore that effective scientific argumentation is not an emergent property of scale, but requires explicit reasoning-aware training, and they point toward practical pathways for building AI systems that can contribute meaningfully to scientific discourse. 

---
# Potential emergence in LLM agents: a statistical physics perspective 

**Authors**: Zhuo-Yang Song, Hua Xing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27618)  

**Abstract**: There has been significant recent interest in understanding the behaviour of LLM agents from the perspective of human preferences or corpus statistics. In this paper we propose agent states as intermediate variables through which post-training can be measured and predicted. From a statistical physics perspective, we study the emergence of a global potential which governs the transition of state in LLM agent reasoning. We introduce a partition function to model the ideal KL-regularised transition distribution and track potential structure, mean reward and state entropy. We demonstrate our approach in two Qwen3.5-9B tasks: safety-test selection and physics-reasoning auditing. We observe that the potential structure strengthens during post-training as predicted from our framework, while joint changes in reward and state entropy provide consistency evidence. States thus provide a quantitative description of how post-training organises agent behaviour. 

---
# PaMuk-Bench: A Parallel Multilingual Multitask Benchmark for Vision-Language Tasks 

**Authors**: Silviu-Florin Gheorghe, Radu-Stefan Popa, Adriana Valentina Costache, Adrian-Gabriel Chifu, Alessia Amelio, Paul Irofti, Anders Skaarup Johansen, Lallouani Bouchakour, Mohamed Lichouri, Khaled Lounnas, Neelu Madan, Yashvir Malik, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27521)  

**Abstract**: While vision-language models (VLMs) currently obtain notable results for highly-studied languages, e.g. English, they exhibit significant performance drops in certain low-resource languages, especially due to the lack of annotated datasets. To this end, we propose PaMuk-Bench, a novel parallel multilingual multitask benchmark for vision-language tasks. Our benchmark includes a set of over 10K curated English prompts that are translated into six languages: Arabic, Danish, French, Hindi, Italian, Romanian. Parallel translations, which are obtained via a semi-automatic process, along with accompanying generated and retrieved images enable the comparative development and evaluation of VLMs across a wide range of tasks: text-to-image generation, image captioning, content-based image retrieval, image generation difficulty estimation, image retrieval difficulty estimation. We empirically show that zero-shot English and multilingual pre-trained VLMs compare consistently worse than their fine-tuned counterparts, confirming that PaMuk-Bench is a valuable resource that can drastically reduce the language gap across a variety of VLM architectures and tasks. PaMuk-Bench is available at this https URL. 

---
# Efficient Natural Language to Temporal Logic Translation for Real-Time Physical AI 

**Authors**: Oliver Schön, Adrien Banse, John Lygeros, Lars Lindemann  

**Link**: [PDF](https://arxiv.org/pdf/2611.27489)  

**Abstract**: Robots can be controlled to perform tasks with guaranteed correctness when the task instructions are stated in a formal language. Large language models (LLMs) have been widely used to allow users to provide such instructions in natural language (NL), by translating NL into formal language. When deploying such translators on a robot, however, we face limited computing and energy resources, while offloading computation or data to the cloud may be infeasible or undesirable. Yet an LLM is a general-purpose text generator, whereas a formula in a fixed formal language fragment follows a well-defined syntax. We therefore ask whether text generation is needed at all. For the formal language of linear temporal logic (LTL), we show that this translation can instead be broken down into answering a set of questions with pre-defined answers. We then construct a System-One model, a recent class of NL-based classifier, to provide such answers and enable us to translate a natural language instruction into an LTL instruction. On three benchmarks, our translator matches or exceeds the accuracy of the best local LLMs available, at up to one order of magnitude lower runtime and energy consumption. We show that it generalizes to unseen atomic propositions and temporal patterns. 

---
# On the Challenges of Using LLMs in High-Risk AI Recruitment: Guiding Questions for Legal Conformity with the AI Act 

**Authors**: Clara Punzi, Stefano Tramacere, Andrea Fedele, Sezer Kutluk, Roberto Pellungrini, Gizem Gezici, Paolo Baracco, Laura Castello, Eleonora Cusin, Eduardo Costa, Amedeo Giacovelli, Mario D'Almo, Fosca Giannotti, Dino Pedreschi, Marzio Di Vece  

**Link**: [PDF](https://arxiv.org/pdf/2611.27462)  

**Abstract**: The EU Artificial Intelligence Act (AI Act) imposes detailed obligations on providers of high-risk AI systems but offers limited operational guidance, particularly when these systems integrate third-party General-Purpose AI (GPAI) models. This paper introduces a toolkit that fosters the development of legal-by-design AI-assisted recruitment systems by translating the core legal obligations of the AI Act's conformity assessment---specifically Articles 9--15 and Annex IV---into techno-normative requirements and corresponding guiding questions. While recruitment pipelines may involve components with varying technical complexity and differing levels of algorithmic transparency, this paper focuses on the most challenging case: the integration of third-party GPAI models into critical stages of the selection process, including resume and job description parsing, applicant scoring, and shortlist generation. This scenario presents significant challenges for downstream providers of high-risk AI systems, as it underscores the difficulties in obtaining the technical information needed to demonstrate compliance with the AI Act's techno-normative requirements, both prior to and during market deployment. After introducing the toolkit and illustrating its application through a simplified example, the paper explores informational dependencies between upstream and downstream GPAI model providers. These dependencies may produce informational gaps across the AI supply chain, potentially hindering the collection of evidence necessary for the AI system provider to ensure legal compliance. The resulting lack of technical transparency negatively affects the deployer, who may struggle to maintain meaningful oversight of the system, and ultimately impacts the persons subject to the AI output. 

---
# HR-Merge: Model Merging with Headroom-Weighted Residual Refinement 

**Authors**: Jiakang Huang, Xueyan Zhang, Renyi Cai, Jingcheng Niu, Jiaru Li, Yining Wang, Josh Zhang, Jiahe Jiang, Jinman Zhao, Gerald Penn, Baharan Mirzasoleiman  

**Link**: [PDF](https://arxiv.org/pdf/2611.27377)  

**Abstract**: Model merging combines experts fine-tuned from a shared backbone, such as LoRA adapters of a large language model, into one multi-task model without further training. Existing methods apply one rule to every task, yet the accuracy they lose is uneven across tasks. We show that the lost accuracy remains recoverable from each task's residual, its task vector minus the merged update, and that a task's validation headroom, the share of its expert validation accuracy the merge loses, indicates where recovery pays off. Residuals from different tasks interfere, so they must be merged rather than summed. We propose HR-Merge, a training-free wrapper around any merging method. It weights each residual by its task's headroom, re-merges the weighted residuals with the base operator, and adds the result back at a strength selected on validation, falling back to the base merge when no strength helps. On six NLI adapters of LLaMA-3 8B, HR-Merge lifts TIES from 93.29 to 94.29 normalized accuracy, above all tested bases, mainly by recovering its weakest task. In vision, on three CLIP backbones, up to 20 tasks, and LoRA merging, it generally improves or preserves base accuracy and yields the best model in every setting. 

---
# Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training 

**Authors**: Zhen Zhang, Changyi Yang, Xuehai He, Chengzhi Liu, Yutong Yin, Ziyue Liu, Zhuoran Yin, Yangmin Li, Zhihan Liu, Jisen Li, Mingyu Jin, Chuhan Li, Zhaotian Weng, Ruoyu He, Wenyao Gao, Jiaxin Deng, Chenwei Wu, Hui Shen, Wenhao Chai, Chenyang Zhao, Zhuofan Ying, Kieran Liu, Andrew Chen, Pony Ma, Zhaoran Wang, Alkesh Patel, Zhe Gan, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27376)  

**Abstract**: RL post-training typically produces domain experts that must then be consolidated into one model, and as new experts keep arriving, this consolidation becomes a continual process. Weight merging and multi-teacher on-policy distillation (MOPD) must re-consolidate the whole model whenever an expert is added, and can erode capabilities outside the target domains. We propose Composable Soft-Prompt Distillation (CSPD), which keeps the pre-RL model frozen, distills each RL expert independently into a soft prompt, and composes experts by concatenating their prompts without joint training. A single soft prompt suffices to distill an RL expert, recovering at least 90% of the teacher's gain in 22 of 23 math, code, and agent settings. Independently trained prompts compose well. In a multi-expert setting, the average rises with every expert added, and the full composition matches MOPD (67.83 vs. 67.79). The composition generalizes best across six held-out benchmarks. Input-space adaptation adds little serving overhead, keeping throughput within 1.5% of the unadapted backbone and reaching 2.40x that of rank-4 LoRA. These results point toward continual post-training on a fixed backbone. 

---
# AMARIS: Merging Generalist and Specialist LLMs via Adaptive Subspace Inheritance 

**Authors**: Hwee Young Kang, Irwin De Tao Chin, Wen-Haw Chong, Hai Leong Chieu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27369)  

**Abstract**: Domain specialization of large language models (LLMs) often improves target-domain performance but degrades broad capabilities. Model merging is an attractive training-free alternative, yet common parameter-space strategies (for example, interpolation and sparsified averaging) provide limited control over task-specific interference. We introduce Adaptive Model Alignment via Ranked Inherited Subspaces (AMARIS), a covariance-driven framework that transfers specialist updates only along selected activation directions. AMARIS formulates merging as budgeted subspace inheritance: it seeks high specialist activation capture under a regularized base-output perturbation budget, then builds a low-rank projection gate from ranked generalized eigendirections. This yields a single static merged checkpoint that concentrates transfer in high-utility, low-interference directions. We evaluate AMARIS on Qwen3-based merges (8B and 32B) across finance, instruction-following, and multilingual SEA specialization. In our evaluated settings, AMARIS reaches Pareto operating points that are competitive with strong training-free baselines, including 98.37% specialist retention with 95.16% core retention for the 8B finance setting, 103.55% core retention with 90.28% specialist retention for the 8B instruction-following setting, and 99.20% core retention with 94.80% specialist retention for the 32B SEA setting. These results support subspace-constrained merging as a practical route to compositional capability integration without additional training. 

---
# SAVIOR: Sample-efficient Adaptation of Vision-Language Models for OCR Representation 

**Authors**: Akshata A Bhat, Sharath Naganna, Saiful Haq, Prashant Khatri, Krishna Chaitanya Reddy Tamataam, Neha Arun, Niyati Chhaya, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2611.27319)  

**Abstract**: Optical Character Recognition (OCR) pipelines and vision-language models exhibit high accuracy on standard OCR and document-understanding benchmarks, yet they systematically underperform on document patterns that are operationally critical in financial workflows. These patterns include vertical or rotated text, logo-embedded vendor names, fine-print regulatory clauses, degraded scans, and complex multi-column layouts. While underrepresented in public datasets, they constitute a substantial portion of real-world failure cases. We introduce SAVIOR, a sample-efficient data curation methodology that targets such high-impact failure scenarios to adapt vision-language models for robust financial OCR. We further propose PaIRS, a structure-aware evaluation metric that measures layout fidelity by comparing pairwise spatial relationships between tokens. Fine-tuning Qwen2.5-VL-Instruct with SAVIOR-Train achieves a word recall of 0.93. When the resulting layout-aware OCR output is used as context for document question answering, it attains an F1 score of 0.89, outperforming both open- and closed-source models such as GPT-4o, Mistral-OCR, PaddleOCR-VL, and DeepSeek-OCR. 

---
# Belief-State Management Meets Rubric Rewards for Agentic Forecasting 

**Authors**: Jingyi Chai, Yun Liao, Hao Lu, Kemeng Zhang, Bingyang Zheng, Yujie Ge, Xiangrui Liu, Linfeng Zhang, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.27286)  

**Abstract**: Current large language model (LLM) agents still struggle to forecast real-world events. This stems from two primary limitations: i) standard ReAct harness accumulates reasoning and evidence within one growing context, and ii) outcome-based rewards in forecasting lead to a slow and noisy signal that cannot distinguish sound reasoning from lucky guesses. To address these, we introduce a forecasting training framework including harness design and post-training. For the harness side, we propose Proactive Belief-State Management (PBM), equipping agents with a fold tool to compress lengthy context into compact belief states and reset the context for subsequent exploration. For the training side, we develop SEIPH, a five-dimensional, outcome-blind rubric, providing process-quality rewards. Using the rubric-based reward, we post-train a 35B model with PPO and GRPO, yielding DeepForecaster-PPO and DeepForecaster-GRPO. Theoretically, we prove that idealized fold-based stopping converges and calibrated local rewards reduce segment-level gradient variance. Empirically, DeepForecaster-PPO (35B) achieves the best Brier score on both ForecastBench and Polymarket-Sim, outperforming frontier agents based on DeepSeek-V4-Pro (1.6T), Kimi-K2.6 (1T), and GPT-5.5. DeepForecaster-GRPO ranks second on both FutureX metrics. Ablations reveal that PBM also improves the untrained base model without any training. Calibration analysis shows that DeepForecaster-PPO makes more accurate high-confidence predictions than Qwen3.6 and GPT-5.5. Trajectory analysis further shows that forecasts generally improve from the first to the final belief state as agents gather evidence and revise their judgments. 

---
# What Does Matching a Human Intervention Effect Establish? Benchmark Checks for LLM Simulations 

**Authors**: Tengfei Shao, Asma Rezaei, Ali Reza Kamali  

**Link**: [PDF](https://arxiv.org/pdf/2611.27186)  

**Abstract**: LLM simulations are validated by equivalence with human effects, yet a zero reference passes whenever the human 90 percent interval lies inside the margin, as the human benchmark shows before simulation. When the zero reference fails, planned precision gives independent simulations the margin up to which passing is a 5 percent test of response. In an exploratory 70-experiment reanalysis, 72 percent of GPT-4's $\pm$0.20 SD passes were zero-reference passes; GPT-4 passed 16 contrasts the zero reference failed (1.58 expected without response). In a preregistered audit, Claude Sonnet 5.5 passed three intention targets, all zero-reference passes, and the registered belief-gradient control; norm beliefs changed 0.065 SD against humans' 0.351 (zero-reference check added after unblinding; norm beliefs emphasized after inspection). Registered tests detected no overestimation; average-effect intervals excluded overestimation above 0.08 SD. A group audit's precision floor is 1.33 facts against a 2.9-fact effect. Report the zero-reference threshold beside every equivalence decision. 

---
# The Vulnerability Window: Decision Instability in Quantized Reward Models 

**Authors**: Omkar Shewale, Deepak Kumar, Divakar Kumar Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2611.27155)  

**Abstract**: Reward models and LLM judges are routinely quantized to 4 bits so that scoring many candidates stays affordable, and the usual check is that benchmark accuracy barely moves. That check is close to blind. Across eight reward models from two families (0.6B-8B), three generative judges and four 4-bit quantizers, quantization rewrites 1.1-12.3% of pairwise preference decisions on RewardBench while aggregate accuracy moves by as little as 0.03 points (p = 1.00), because the errors it fixes and introduces cancel. Flips concentrate where the full-precision margin is small relative to the quantization noise scale sigma: a Gaussian approximation Phi(-|m|/sigma) describes the aggregate rate, over-predicting it in all 30 scalar settings and failing for generative judges, and occupancy of the resulting vulnerability window falls from 25% to 7% with model size, which is what makes small judges fragile. The churn stays within noise under benign best-of-16 and becomes clear under adversarial pressure for some quantizers (146:54 harmful:helpful flips on one family, reversing sign on another). Downstream it changes what ships: across six (model, quantizer) pairs from two families, 14.6-65.7% of problems get a different best-of-64 selection while only 2.6-9.6% get a different final answer, with selections changing 5-8x more often than answers in every pair, and the pair the benchmark calls identical (0.03 points, p = 1.00) still changes 34.5% of them. Benchmark churn ranks that downstream change (rho = 0.89); benchmark accuracy does not order it reliably. Held-out audits then locate the useful diagnostic in the margin rather than the Gaussian: every margin-based predictor orders flips identically (AUC 0.927), and on the level a directly measured churn rate wins in-distribution. As a routing signal it is matched by plain top-k. The contribution is the characterization, not the model. 

---
# Double counting in LLM evidence fusion: exact posteriors separate rule compliance from posterior gain 

**Authors**: Tengfei Shao, Asma Rezaei, Ali Reza Kamali  

**Link**: [PDF](https://arxiv.org/pdf/2611.27138)  

**Abstract**: Language models fuse reports in retrieval, multi-agent consensus and agent memory, where treating shared-source reports as independent can double count evidence. A common remedy is an instruction to count each source once. Its value depends on the receiver's raw baseline. Pricing it needs an exact reference. We build the graded-dependence exact reference: binary worlds with single-report reliability 0.65 and four dependence levels, whose disclosed parameters give every world an exact posterior and every rule a target risk. The paired gain protocol scores raw and instructed answers of four configurations on 2,976 worlds. Agreement and gain separate in both directions. At $\delta$ = 0, claude-opus-5-5 and gpt-6.1-sol agreed with the rule within 0.001 in 744 of 744, and the instruction raised their posterior mismatch by 0.044; gpt-6-luna agreed in 165 of 744 and gained 0.0140 (95% interval 0.0064 to 0.0219); claude-haiku-5-5's gain moved from -0.0295 ($\delta$ = 0) to +0.0302 ($\delta$ = 1) (descriptive). The prespecified first-child endpoint contrast, gain at $\delta$ = 1 minus $\delta$ = 0, was 0.0342 (0.0303 to 0.0382). In the preregistered tests, the raw source-weight contrast was positive (0.259, 95% interval 0.182 to 0.337; Holm rejected zero) and the grouping endpoint difference was not detected (0.0004, -0.0023 to 0.0031). This interval excludes differences larger than 0.0031 in magnitude and lies within the registered precision bound of $\pm$0.0046. The reference prices the rule before any model call. The paired gain, estimable as a Brier difference on labelled deployment items, decides adoption per receiver and prompt condition. 

---
# Reasoning in Two Dimensions: Quantifying Chain-of-Thought Faithfulness and Efficiency 

**Authors**: Chenruo Liu, Quan Zhang, Qi Lei  

**Link**: [PDF](https://arxiv.org/pdf/2611.27100)  

**Abstract**: Chain-of-thought (CoT) reasoning exposes intermediate steps, but a correct answer does not establish that an LLM relies on the dependence expressed by those steps or reasons efficiently. We operationalize CoT faithfulness and efficiency within a common framework. Specifically, we represent a CoT as a semantic reasoning graph and decompose the model's predictive dependence along two complementary axes: dependence explained by semantic support versus additional reasoning history, and dependence allocated to answer-reaching versus off-path reasoning. This decomposition yields global and step-level measures of unfaithfulness and inefficiency. Controlled interventions show that the proposed metrics respond to hidden dependence and induced off-path reasoning as intended. Across mathematical and multi-hop reasoning tasks, answer-level unfaithfulness is consistently associated with greater predictive uncertainty, whereas both unfaithfulness and inefficiency exhibit weak or inconsistent associations with answer correctness. We further derive theoretically grounded, graph-aware fine-tuning objectives that improve the targeted reasoning properties without directly optimizing the computationally expensive metrics or compromising answer accuracy. Results indicate that accuracy and brevity provide incomplete assessments of reasoning quality: reliable CoT evaluation should also consider whether the LLM's predictive dependence is faithfully represented and efficiently allocated. 

---
# PLoQ: A Prompt-Specific Low-Rank Basis for 4-bit Quantization of LLM Activations 

**Authors**: Faraz Tahmasebi, Michael Pellauer, Hyoukjun Kwon, Sitao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27092)  

**Abstract**: Low-bit activation quantization of large language models remains challenging, as a small number of dominant directions dictate the quantization range. State-of-the-art methods isolate these directions in a high-precision subspace through transformations calibrated offline and folded into the weights, which fixes the subspace for all prompts, leaving each prompt's few unique directions unprotected, and rules it out wherever no weight can absorb the transformation. We present PLoQ, a prompt-adaptive low-rank decomposition that extracts the dominant activation components before quantization, quantizes them separately, and restores them within the layer's GEMM, without modifying the weights. The low-rank component's basis is estimated from each prompt during prefill and reused throughout decoding. At the same bit width as state-of-the-art mixed-precision quantization (4-bit weights, activations and K/V cache with an eighth of the channels at 8 bits, about 4.2 bits per activation on average), PLoQ outperforms it on all six Llama and Qwen models from 1B to 70B parameters, at up to 8k tokens (16k on two of them), recovering up to 46% of the perplexity that the state of the art still loses to 4-bit activations. On Llama-3.1-8B at 8k tokens it adds only 4% to prefill and 8% to decoding. 

---
# ORION: Exact Canonical-State Supervision for Systematic Generalization in Transformers 

**Authors**: Sibi Gokul, Jyoti Prakash Meher  

**Link**: [PDF](https://arxiv.org/pdf/2611.27087)  

**Abstract**: Autoregressive Transformers generally fail on out-of-distribution reasoning due to two primary reasons: positional corruption over long sequences and structural state drift across deep execution steps. We introduce ORION, a framework that supervises exact canonical execution states via auxiliary heads to anchor latent trajectories to valid computational manifolds. Across Folding, GridShift, and CLRS-30, a pre-registered 2 x 2 factorial shows that RoPE and ORION address orthogonal axes. RoPE recovers length extrapolation (59.8%), while ORION recovers depth extrapolation (66.7%). Moreover, their combination complementarily achieves strict additive gain in joint length-depth extrapolation, while either method alone offers near satisfactory results. Negative controls confirm that the gain depends strictly on exact canonical alignment; hashed state statistics fail to constrain depth extrapolation. 

---
# From Letter-of-Credit Requirements to Executable Specifications: Evaluating Verification Accuracy and Completeness 

**Authors**: Ji Sub Choi, Whan Choi, Seo Yeon Gu, Suhyun Kim, Inyong Yun  

**Link**: [PDF](https://arxiv.org/pdf/2611.26988)  

**Abstract**: In letter-of-credit (LC) document examination, a correct final decision does not show that every requirement was examined. We therefore evaluate examination requirement by requirement and make each requirement executable. A language model converts the bill of lading (BL) and commercial invoice (IV) requirements of a credit into atomic conditions and logical structures, and a deterministic verifier retrieves the named evidence and computes each outcome, so every check can be compared with a gold encoding and traced to logged evidence. Because LC records are confidential, our on-premises 9B parser is fine-tuned only on synthetic pairs built from templates and human-reviewed public document values. On 345 real LC cases, final-decision accuracy proves misleading: a constant PASS rule has the highest raw accuracy while examining nothing. Although it never sees a real record during training, the fine-tuned parser detects 93% of the atomic requirements its ontology covers and reconstructs 86% of the covered requirements exactly. It verifies 82% of the evaluable requirements correctly, against 54% for a tool-using agent, and leads the agent on both satisfied requirements and discrepancies. It also completely examines 1.8 times as many cases as few-shot prompting of the same model, while direct prediction and the agent complete none. Executing the gold encodings with the same verifier shows that some of the remaining errors, missed discrepancies included, come from the operators and normalization, which a better parser alone cannot fix. Every rejection comes with a trace of the units and evidence behind it, which suits examiner-in-the-loop review. 

---
# Run One, Get Two: Near-Free Streaming Safety Checks for Local LLMs 

**Authors**: Soheil Zibakhsh, Pedram Aghazadeh, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2611.26975)  

**Abstract**: Local LLM assistants decode for one user at a time, so decoding is bound by memory bandwidth and leaves compute idle. Streaming safety checks today trade accuracy against cost. A dedicated guard model is accurate but nearly doubles the latency per token, while a hidden-state probe is cheap but less accurate. We present ToRA, an adapter that runs a second model as an extra row in the generator's decode step, sharing its weights and KV cache. Dual-row kernels read each weight once for both rows. On Qwen3-0.6B, ToRA matches or beats a dedicated streaming guard, with higher F1 on 12 of 17 benchmarks. It detects 11.8 points more unsafe responses than the probe at a similar cost, under 4% added latency per token. 

---
# Right Numbers, Wrong Story: When LLMs Narrate Financial Changes Beyond Their Inputs 

**Authors**: Seunghan Lee, Jun Seo, Junhyeok Kang, Jaehoon Lee, Sangjun Han, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, Soonyoung Lee, Wonbin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2611.26968)  

**Abstract**: A financial report drafted by an LLM can state every number correctly but still go beyond its input, adding judgments and causes that no one supplied. We call these additions unsupported framing and find them far more often in LLM drafts than in reports written by proInput facts Revenue +3% Higher prices 10-K filings Revenue increased 3% on higher prices. 10-K writer Revenue surged 3% on higher prices, due to strong demand. LLM Fits the size? Surged: Unusual for 3% In the input? In the input? Strong demand: Not in input Higher prices: Kept fessionals, and attempts to fix them by prompting with how each word is used instead make such judgments more frequent. We therefore propose FrameCheck, a plug-in that sits beFrameCheck Revenue increased 3% on higher prices. No extra model call Judgment word Unsupported cause Supported cause tween the LLM and human review to keep each draft within its input. Specifically, FrameCheck 1) guides the draft with professional examples that report a change of similar magnitude, 2) flags judgments that professionals would not use and causes the input does not give, and 3) removes them by rule, without a model call. In experiments on 17 open LLMs and 3M drafts, FrameCheck removes almost all unsupported framing from drafts and brings the judgment rate close to that of the writers of 10-K filings, the annual reports that US firms file with regulators. Code is available at this http URL. 

---
# Self-similar Organization of Information in Language 

**Authors**: Leon Lufkin, Cutter Dawes, Kamesh Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2611.26935)  

**Abstract**: How much information relevant to predicting the future of a passage (i.e. predictive information) is carried by the preceding text, and how is this information distributed along that history? These questions are fundamental to understanding both the statistical structure of language and what large language models (LLMs) can learn about it. Indeed, LLMs trained to predict the next token in text must exploit this predictive information, so its organization determines how the structure of language becomes available to the model to learn. We develop a theory for the organization of predictive information based on the scaling behavior of a family of information measures, and the theory predicts that if this organization has the same form at all measured length scales -- i.e., it has a self-similar structure -- then the information measures evaluated at different lengths collapse onto a common form. We observe this scaling collapse in long text documents, showing that predictive information is present at every length scale we measure and has a self-similar structure. Self-similarity is a considerably stronger property than showing the existence of long-range dependencies, and it constrains the process generating the text. More broadly, our results give quantitative support to the empirical observation that conditioning on more distant text continues to reveal new structure in language to LLMs. 

---
# ERA: Embed, Route, Adapt for Cold-Start Cross-Model Latent Retrieval 

**Authors**: Venkata Chivukula, Zifeng Wang, Jinsung Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2611.26919)  

**Abstract**: Production retrieval systems are tied to the embedding model that indexed their corpus. Moving queries to a different encoder normally requires re-embedding the whole corpus, and at cold start there are no query logs while every document embedded by the new encoder incurs an API or compute cost. We present ERA (Embed, Route, Adapt), a system for cold-start cross-model latent retrieval. ERA spends a small anchor budget on maximally informative documents, selected from the existing target index via Cosine Farthest Point Sampling, trains a lightweight adapter with a Manifold Alignment Loss that couples directional alignment with orthogonal regularization to prevent representation collapse, and pairs the adapter with a deterministic router that escalates the rare, hard queries that define perceived retrieval quality. Across five BEIR datasets, ERA matches or outperforms point-wise L2 and embedding-conversion baselines, with relative gains of up to 35% nDCG@10 on Arguana, and on clustered corpora its anchor selection also improves prior methods under a fixed anchor budget. With 1.2 ms of adapter and router overhead on top of a ~20 ms local encoder, ERA at a 20% nominal escalation budget reduces end-to-end query latency from roughly 160 ms to 38-86 ms while concentrating escalations on the queries where the projection fails most. 

---
# What Does the Visual Encoder Buy? Visual Information Gain in Encoder-Free Multimodal Pretraining 

**Authors**: Zhixuan Ge, Anqi Li, Wei Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26917)  

**Abstract**: Recent scaling studies ask whether vision-language models still need a visual encoder by comparing encoder-free and encoder-based models on the next-token loss of the text in image-text data. This loss can fall without any use of the image, and two runs with the same loss can differ in what they take from it. Visual information gain (VIG) is the loss increase when the image is replaced by a uniform one. It splits exactly into global information, carried by image statistics, and layout information, which shuffling the patches within the image destroys. We apply it to decoders with 158M to 1.04B parameters that read pixel patches through a linear embedding, evaluated every billion tokens up to 40B, and to 19 public checkpoints. In our decoders the image lowers the loss by 3% to 46%, depending on the image type. Global information is present after the first billion tokens. Layout information starts near zero, grows slowly with training and does not grow with model size on five of six image types, and a larger batch reaches the same loss with less of it. With a frozen SigLIP 2 encoder in front of the same decoder the loss is lower only when the arrangement of the image is intact. On shuffled images the losses with and without the encoder agree to within 0.04 nats on five of six image types, so the net advantage of the encoder in the loss is layout information, above all the ability to read text in images. Public encoder-free models at production scale (Gemma 4 12B, NEO, Mono-InternVL) do acquire layout information. On the same documents they have 0.53 to 0.76 nats of it, against 0.65 to 0.82 for recent encoder-based models and 0.04 to 0.07 for our decoders, and their released training stages show it rising. These results agree with the reported parity of the two families when both are trained from scratch and place the reported advantage of a pretrained encoder in layout information. 

---
# AstronautBench: Measuring Agent Self-Improvement Across Episodes Through Knowledge, Evidence, and Progress 

**Authors**: Congyuan Yu, Runping Tian  

**Link**: [PDF](https://arxiv.org/pdf/2611.26910)  

**Abstract**: LLM agents are expected to improve themselves from their own experience across episodes, with fixed weights. Self-improvement in an embodied world cannot be read from one measure. Existing benchmarks measure task progress, world-knowledge question answering (QA), and episodic memory, but a correct answer does not show that the agent ever observed the evidence for it, and progress does not show what the agent learned. We introduce AstronautBench, which measures one agent's self-improvement through three views on the same run: what it knows, whether its own evidence supports that knowledge, and how far it gets in the world. It is built on the game Outer Wilds, where each episode is a 22-minute loop after which the world resets, only the agent's knowledge carries over, and progress is gated by knowledge. After every episode, we measure the places reached and ship-log rumor facts revealed, the correctness of answers to 164 questions about the world, and whether each answer is supported by the frames and position the agent cites from its own recorded play. Two LLM judges from different model families check every answer. In eight-episode baselines with GPT-6.1 Sol and Gemini 4 Argon planners, which act through a learned controller or their own gamepad inputs, the three views do not move together, so any one of them alone gives an incomplete picture of whether the agent is improving. The GPT-6.1 Sol agent's correct answers grow from 14.6% to 26.8% of all questions, but at episode 8 only 59 to 68% of them are supported by its cited evidence. It reveals no new rumor facts after episode 3, while the places it reaches and its memory keep growing. More than half of the answers that become correct after episode 3 cite only evidence from earlier episodes. 

---
# What Does a Zoom Actually Add? Decomposing Fine-Grained Perception Failures in Vision-Language Models 

**Authors**: Zhixuan Ge, Anqi Li, Kexin Geng, Wei Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26904)  

**Abstract**: Zooming into a region often turns a wrong answer from a vision-language model (VLM) into a right one, and a growing literature takes this as evidence that the model lacked the pixels. But a zoom changes several things at once. It removes the surrounding context, re-tokenizes, at a higher density, pixels the model has already seen, and, if the first pass was downsampled, exposes detail the model never received. We separate these factors for each failure. Holding the first pass fixed, we vary only the second look and assign every failure to the cheapest one that fixes it: reusing cached tokens, isolating the region, re-tokenizing the seen pixels, or acquiring new ones. The study covers 2,753 zoom-rescued failures of Qwen2.5-VL, Qwen3-VL and InternVL3 on V*, ZoomBench and TextVQA, at first-pass budgets from 17x fewer pixels than the source to native resolution. First, most rescued failures did not lack the pixels: upsampling the pixels the model had already seen fixes 47-73% of them even at 17x downsampling, against 1-3% for a random region. Second, the first-pass budget decides how often new pixels are needed: depending on the benchmark, new pixels are required for 23-50% of the rescues at 17x and for 12-23% at 4x. The missing-pixel reading and the accounts that dispute it thus hold at different budgets. Third, the useful magnification is large and differs between objects and text: seen pixels fix more failures up to 6-12x their first-pass size per side on objects and saturate by 4x on text, and even native pixels gain from being rendered beyond native resolution. Allowing any magnification, 68-94% of the failures at 17x need no new pixels. These models more often need a better rendering of evidence they already have than new pixels, which motivates re-tokenizing a region before acquiring it. 

---
# Perceive Once, Escalate on Demand: Handing Off Visual KV Memory from Small to Large VLMs 

**Authors**: Qiang Zhou, Jianing Deng, Siyan Dong, Jingxian Chen, Han-Yi Lin, Jingtong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26902)  

**Abstract**: An embodied agent cannot afford a large vision-language model (VLM) at every step of a long video, so a cascade runs a small VLM continuously and escalates only uncertain decisions to a large one. Each escalation, however, forces the large model to re-encode a history the small model has already perceived, and this dominates its cost. We present Baton, which hands the large model the small model's visual memory, its key-value (KV) cache over the history. Baton rests on one finding: the large model needs a usable memory, not a faithful copy of its own. A map trained to reproduce the large model's cache matches it most closely, yet brings no gain over the small model alone. Baton instead distills a per-layer linear map, without labels, from the large model's own answers, with the large model encoding only the current frame. Interventions on the map trace the gain to one direction, carried by the middle third of the large model's layers. On RoboMME, Baton beats the small model alone, a pruned or shortened re-watch, and a text memory; as the escalation channel of a cascade it matches re-watching to within about one point at 28-47% less compute, and 8x less per escalation at 64 frames. Its gain grows with the large model, from 8B to 32B, tracks the large model's headroom across eight model pairs from four families, and transfers zero-shot to OpenEQA. 

---
# Don't Discard Your Rollouts: Reusing Teacher RL Traces for Student Distillation 

**Authors**: Vashisth Tiwari, Emma Strubell, J. Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2611.26803)  

**Abstract**: RL post-training methods such as PPO and GRPO are computationally expensive, with substantial resources devoted to sampling rollouts, which contain multiple completions per prompt along with their respective rewards. Canonical distillation, however, discards these traces and uses more compute to regenerate data from the final teacher. We ask if and how we can reuse the saved teacher RL rollouts to distil student models. Reusing rollouts is non-trivial: naive SFT on the rollouts underperforms canonical distillation, and naive DPO, while performant, is unstable, collapsing on weaker students. We show that treating rollout reuse as a data selection problem using metadata already recorded during GRPO substantially reduces this gap. We use prompt pass rates to order the data, using the easiest prompts for an SFT warm start prior to using mixed-success prompts for DPO. For each prompt, we use saved teacher log-likelihoods as a proxy for selecting student-compatible completions. Across mathematical reasoning and Python coding tasks, with students from multiple model families, our two-stage recipe achieves comparable average performance to canonical distillation, while incurring no additional inference cost for dataset construction. 

---
# TOUMAII: A 4.5M-Pair Tool-Calling Preference Dataset and Its Source-Agnostic Generator 

**Authors**: Moritz Pötzsch, Andrea Gentilini, Lisa Rossgoderer, Tom Osterfeld, Kesava Prasad Sanjivi Arul, Swanand Ravindra Kadhe  

**Link**: [PDF](https://arxiv.org/pdf/2611.26722)  

**Abstract**: To autonomously execute multi-step tasks, enterprise agents must reliably interact with tools, call APIs, or query databases, making tool-calling a fundamental capability of AI models. Supervised fine-tuning (SFT) optimises for producing the correct call, but never contrasts it with a plausible incorrect one. In an environment where even a minor formatting error can invalidate a call and an incorrect decision can derail the interaction, direct preference optimisation (DPO) targets exactly this contrast. DPO, therefore, requires preference pairs rather than plain demonstrations. For tool-calling, such data, especially in realistic multi-turn settings, remains extremely scarce. To address this scarcity, we introduce TOUMAII, a large-scale tool-calling preference dataset containing ~4.5M validated preference pairs, derived from Agent-Ark/Toucan-1.5M (apache-2.0) conversations. The dataset consists of multi- and single-turn interactions and includes parallel and sequential function calls. Additionally, we release TOUMAII-Gen, a synthetic data generation pipeline that consumes tool-calling SFT datasets and generates a preference dataset with naturally occurring model-generated tool-calling failure modes. To validate the effectiveness of our DPO data, we fine-tune a 3-billion-parameter model, achieving performance parity with other high-quality tool-calling preference datasets. Furthermore, we release a second small preference dataset derived from ToolACE (apache-2.0), showcasing the source-agnostic nature of our pipeline.
We release the complete code and datasets: this https URL and this https URL 

---
# Adding One More Agent: How AI Team Size and Formation Shape Productivity and Diversity in Human-AI Brainstorming Teams 

**Authors**: Rong-Ching Chang, Hao-Chuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26663)  

**Abstract**: Generative AI has improved individual productivity, but whether adding a plurality of AI partners helps or hurts remains open. We crossed AI partnership (one versus two) with behavioral formation (calibrated to average human behavior versus customized by the participant) in a brainstorming experiment ($N$=168) with human-human dyads as reference. Two-AI rooms posted more ideas to the shared board, but human-authored ideas dropped substantially. With one AI partner, formation mattered: humans paired with an average-calibrated AI posted more than any other condition. With two, this effect vanished. Humans continued talking but converted fewer ideas into board posts. AI social talk predicted more human posts, while AI posting volume predicted fewer. AI presence increased global diversity of human-authored ideas, but only a second AI increased within-session differentiation. The number of AI partners and their formation can reshape what the human contributes; single-agent results do not generalize to multi-agent teams. 

---
# VerifiQuant: Governance-by-Execution for Financial LLM Workflows 

**Authors**: Yu-Chi Ko, Mi-Yen Yeh, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.26649)  

**Abstract**: Large language models embedded in financial workflows fail most dangerously by emitting confident, well-formatted but wrong numbers with no signal distinguishing them from correct ones. We argue that the central deployment problem is governability: a system must determine when a question supports a numeric answer and when it should instead clarify, intercept, or refuse. We present VerifiQuant, a governance-by-execution framework that packages financial logic into inference contracts and evaluates each query through an executable diagnostic funnel before deterministic computation. This separation makes computation, clarification, and refusal structured and auditable while keeping financial arithmetic outside the language model. Against matched retrieval-and-execution baselines, and within a declared contract library, the principal benefit is not higher accuracy on well-posed questions but safer behavior when an input does not justify a number. When it finds no applicable contract, it says so instead of guessing. Coverage sweeps and failure analysis put a price on this boundary: safety is bought with coverage rather than with accuracy, and residual failures remain attributable to specific runtime frontiers. 

---
# SDG: State-Dependent Gating at Chunk Boundaries for Linear Attention 

**Authors**: Renjia Deng, Yuxi Liu, Yutong He, Xue Wang, Kun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.26637)  

**Abstract**: Recent gated linear attention models use state-independent gates, computed from token representations without directly reading the recurrent memory. This design supports efficient chunkwise computation: sequences are divided into chunks of consecutive tokens, with parallel computation within each chunk. However, the same token representation produces the same gate, even when the layer stores different information. We introduce State-Dependent Gating (SDG), which computes element-wise gates from the change in memory over a chunk and applies them at the chunk boundary. This provides state feedback for forgetting while preserving parallel computation within each chunk. We develop fused kernels for Gated DeltaNet, Kimi Delta Attention, Gated DeltaNet-2, and Mamba-3. Our theoretical analysis quantifies the potential reconstruction benefits of state-dependent, element-wise gating under a signal--noise model. We also show that boundary decay is an element-wise ridge proximal step when the gate is held fixed. Empirical evaluation of 1.3B linear attention and hybrid models trained on 100B tokens shows improvements across language modeling, commonsense reasoning, synthetic and real-world retrieval, and long-context understanding. These results demonstrate that SDG is an effective approach to enhancing existing linear attention models. 

---
