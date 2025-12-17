# MMGR: Multi-Modal Generative Reasoning 

**Authors**: Zefan Cai, Haoyi Qiu, Tianyi Ma, Haozhe Zhao, Gengze Zhou, Kung-Hsiang Huang, Parisa Kordjamshidi, Minjia Zhang, Xiao Wen, Jiuxiang Gu, Nanyun Peng, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14691)  

**Abstract**: Video foundation models generate visually realistic and temporally coherent content, but their reliability as world simulators depends on whether they capture physical, logical, and spatial constraints. Existing metrics such as Frechet Video Distance (FVD) emphasize perceptual quality and overlook reasoning failures, including violations of causality, physics, and global consistency. We introduce MMGR (Multi-Modal Generative Reasoning Evaluation and Benchmark), a principled evaluation framework based on five reasoning abilities: Physical, Logical, 3D Spatial, 2D Spatial, and Temporal. MMGR evaluates generative reasoning across three domains: Abstract Reasoning (ARC-AGI, Sudoku), Embodied Navigation (real-world 3D navigation and localization), and Physical Commonsense (sports and compositional interactions). MMGR applies fine-grained metrics that require holistic correctness across both video and image generation. We benchmark leading video models (Veo-3, Sora-2, Wan-2.2) and image models (Nano-banana, Nano-banana Pro, GPT-4o-image, Qwen-image), revealing strong performance gaps across domains. Models show moderate success on Physical Commonsense tasks but perform poorly on Abstract Reasoning (below 10 percent accuracy on ARC-AGI) and struggle with long-horizon spatial planning in embodied settings. Our analysis highlights key limitations in current models, including overreliance on perceptual data, weak global state consistency, and objectives that reward visual plausibility over causal correctness. MMGR offers a unified diagnostic benchmark and a path toward reasoning-aware generative world models. 

---
# Spoken DialogSum: An Emotion-Rich Conversational Dataset for Spoken Dialogue Summarization 

**Authors**: Yen-Ju Lu, Kunxiao Gao, Mingrui Liang, Helin Wang, Thomas Thebaud, Laureano Moro-Velazquez, Najim Dehak, Jesus Villalba  

**Link**: [PDF](https://arxiv.org/pdf/2512.14687)  

**Abstract**: Recent audio language models can follow long conversations. However, research on emotion-aware or spoken dialogue summarization is constrained by the lack of data that links speech, summaries, and paralinguistic cues. We introduce Spoken DialogSum, the first corpus aligning raw conversational audio with factual summaries, emotion-rich summaries, and utterance-level labels for speaker age, gender, and emotion. The dataset is built in two stages: first, an LLM rewrites DialogSum scripts with Switchboard-style fillers and back-channels, then tags each utterance with emotion, pitch, and speaking rate. Second, an expressive TTS engine synthesizes speech from the tagged scripts, aligned with paralinguistic labels. Spoken DialogSum comprises 13,460 emotion-diverse dialogues, each paired with both a factual and an emotion-focused summary. The dataset is available online at this https URL. Baselines show that an Audio-LLM raises emotional-summary ROUGE-L by 28% relative to a cascaded ASR-LLM system, confirming the value of end-to-end speech modeling. 

---
# Fast and Accurate Causal Parallel Decoding using Jacobi Forcing 

**Authors**: Lanxiang Hu, Siqi Kou, Yichao Fu, Samyam Rajbhandari, Tajana Rosing, Yuxiong He, Zhijie Deng, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14681)  

**Abstract**: Multi-token generation has emerged as a promising paradigm for accelerating transformer-based large model inference. Recent efforts primarily explore diffusion Large Language Models (dLLMs) for parallel decoding to reduce inference latency. To achieve AR-level generation quality, many techniques adapt AR models into dLLMs to enable parallel decoding. However, they suffer from limited speedup compared to AR models due to a pretrain-to-posttrain mismatch. Specifically, the masked data distribution in post-training deviates significantly from the real-world data distribution seen during pretraining, and dLLMs rely on bidirectional attention, which conflicts with the causal prior learned during pretraining and hinders the integration of exact KV cache reuse. To address this, we introduce Jacobi Forcing, a progressive distillation paradigm where models are trained on their own generated parallel decoding trajectories, smoothly shifting AR models into efficient parallel decoders while preserving their pretrained causal inference property. The models trained under this paradigm, Jacobi Forcing Model, achieves 3.8x wall-clock speedup on coding and math benchmarks with minimal loss in performance. Based on Jacobi Forcing Models' trajectory characteristics, we introduce multi-block decoding with rejection recycling, which enables up to 4.5x higher token acceptance count per iteration and nearly 4.0x wall-clock speedup, effectively trading additional compute for lower inference latency. Our code is available at this https URL. 

---
# TiME: Tiny Monolingual Encoders for Efficient NLP Pipelines 

**Authors**: David Schulmeister, Valentin Hartmann, Lars Klein, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2512.14645)  

**Abstract**: Today, a lot of research on language models is focused on large, general-purpose models. However, many NLP pipelines only require models with a well-defined, small set of capabilities. While large models are capable of performing the tasks of those smaller models, they are simply not fast enough to process large amounts of data or offer real-time responses. Furthermore, they often use unnecessarily large amounts of energy, leading to sustainability concerns and problems when deploying them on battery-powered devices. In our work, we show how to train small models for such efficiency-critical applications. As opposed to many off-the-shelf NLP pipelines, our models use modern training techniques such as distillation, and offer support for low-resource languages. We call our models TiME (Tiny Monolingual Encoders) and comprehensively evaluate them on a range of common NLP tasks, observing an improved trade-off between benchmark performance on one hand, and throughput, latency and energy consumption on the other. Along the way, we show that distilling monolingual models from multilingual teachers is possible, and likewise distilling models with absolute positional embeddings from teachers with relative positional embeddings. 

---
# JMMMU-Pro: Image-based Japanese Multi-discipline Multimodal Understanding Benchmark via Vibe Benchmark Construction 

**Authors**: Atsuyuki Miyai, Shota Onohara, Jeonghun Baek, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2512.14620)  

**Abstract**: This paper introduces JMMMU-Pro, an image-based Japanese Multi-discipline Multimodal Understanding Benchmark, and Vibe Benchmark Construction, a scalable construction method. Following the evolution from MMMU to MMMU-Pro, JMMMU-Pro extends JMMMU by composing the question image and question text into a single image, thereby creating a benchmark that requires integrated visual-textual understanding through visual perception. To build JMMMU-Pro, we propose Vibe Benchmark Construction, a methodology in which an image generative model (e.g., Nano Banana Pro) produces candidate visual questions, and humans verify the outputs and, when necessary, regenerate with adjusted prompts to ensure quality. By leveraging Nano Banana Pro's highly realistic image generation capabilities and its ability to embed clean Japanese text, we construct a high-quality benchmark at low cost, covering a wide range of background and layout designs. Experimental results show that all open-source LMMs struggle substantially with JMMMU-Pro, underscoring JMMMU-Pro as an important benchmark for guiding future efforts in the open-source community. We believe that JMMMU-Pro provides a more rigorous evaluation tool for assessing the Japanese capabilities of LMMs and that our Vibe Benchmark Construction also offers an efficient guideline for future development of image-based VQA benchmarks. 

---
# Towards Nepali-language LLMs: Efficient GPT training with a Nepali BPE tokenizer 

**Authors**: Adarsha Shrestha, Basanta Pokharel, Binit Shrestha, Smriti Adhikari, Dinesh Gothe  

**Link**: [PDF](https://arxiv.org/pdf/2512.14585)  

**Abstract**: Nepali, a low-resource language spoken by over 32 million people, continues to face challenges in natural language processing (NLP) due to its complex grammar, agglutinative morphology, and limited availability of high-quality corpora. Most efforts to date have centered on basic encoder architectures; they remain insufficient for Nepali-specific text generation. This study presents a GPT-2-based Nepali language model trained using several training strategies inspired by GPT-3, including optimized learning rate schedules, batch scaling, and architectural refinements. A custom 16k Byte-Pair Encoding (BPE) tokenizer was trained exclusively on Nepali text to ensure more consistent segmentation and improved input representation. The model was pretrained on a combined dataset comprising a 10.75GB cleaned NepBERTa corpus and additional web-scraped Nepali news articles. FlashAttention was integrated to reduce memory usage and stabilize training. After two epochs, the model achieved a training loss of 3.168177, a validation loss of 3.081982, and a final perplexity of 21.80, demonstrating its capability to generate coherent Nepali news-style text. 

---
# Low-Resource, High-Impact: Building Corpora for Inclusive Language Technologies 

**Authors**: Ekaterina Artemova, Laurie Burchell, Daryna Dementieva, Shu Okabe, Mariya Shmatova, Pedro Ortiz Suarez  

**Link**: [PDF](https://arxiv.org/pdf/2512.14576)  

**Abstract**: This tutorial (this https URL) is designed for NLP practitioners, researchers, and developers working with multilingual and low-resource languages who seek to create more equitable and socially impactful language technologies. Participants will walk away with a practical toolkit for building end-to-end NLP pipelines for underrepresented languages -- from data collection and web crawling to parallel sentence mining, machine translation, and downstream applications such as text classification and multimodal reasoning. The tutorial presents strategies for tackling the challenges of data scarcity and cultural variance, offering hands-on methods and modeling frameworks. We will focus on fair, reproducible, and community-informed development approaches, grounded in real-world scenarios. We will showcase a diverse set of use cases covering over 10 languages from different language families and geopolitical contexts, including both digitally resource-rich and severely underrepresented languages. 

---
# Polypersona: Persona-Grounded LLM for Synthetic Survey Responses 

**Authors**: Tejaswani Dash, Dinesh Karri, Anudeep Vurity, Gautam Datla, Tazeem Ahmad, Saima Rafi, Rohith Tangudu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14562)  

**Abstract**: This paper introduces PolyPersona, a generative framework for synthesizing persona-conditioned survey responses across multiple domains. The framework instruction-tunes compact chat models using parameter-efficient LoRA adapters with 4-bit quantization under a resource-adaptive training setup. A dialogue-based data pipeline explicitly preserves persona cues, ensuring consistent behavioral alignment across generated responses. Using this pipeline, we construct a dataset of 3,568 synthetic survey responses spanning ten domains and 433 distinct personas, enabling controlled instruction tuning and systematic multi-domain evaluation. We evaluate the generated responses using a multi-metric evaluation suite that combines standard text generation metrics, including BLEU, ROUGE, and BERTScore, with survey-specific metrics designed to assess structural coherence, stylistic consistency, and sentiment this http URL results show that compact models such as TinyLlama 1.1B and Phi-2 achieve performance comparable to larger 7B to 8B baselines, with a highest BLEU score of 0.090 and ROUGE-1 of 0.429. These findings demonstrate that persona-conditioned fine-tuning enables small language models to generate reliable and coherent synthetic survey data. The proposed framework provides an efficient and reproducible approach for survey data generation, supporting scalable evaluation while facilitating bias analysis through transparent and open protocols. 

---
# Agreement Between Large Language Models and Human Raters in Essay Scoring: A Research Synthesis 

**Authors**: Hongli Li, Che Han Chen, Kevin Fan, Chiho Young-Johnson, Soyoung Lim, Yali Feng  

**Link**: [PDF](https://arxiv.org/pdf/2512.14561)  

**Abstract**: Despite the growing promise of large language models (LLMs) in automatic essay scoring (AES), empirical findings regarding their reliability compared to human raters remain mixed. Following the PRISMA 2020 guidelines, we synthesized 65 published and unpublished studies from January 2022 to August 2025 that examined agreement between LLMs and human raters in AES. Across studies, reported LLM-human agreement was generally moderate to good, with agreement indices (e.g., Quadratic Weighted Kappa, Pearson correlation, and Spearman's rho) mostly ranging between 0.30 and 0.80. Substantial variability in agreement levels was observed across studies, reflecting differences in study-specific factors as well as the lack of standardized reporting practices. Implications and directions for future research are discussed. 

---
# VLegal-Bench: Cognitively Grounded Benchmark for Vietnamese Legal Reasoning of Large Language Models 

**Authors**: Nguyen Tien Dong, Minh-Anh Nguyen, Thanh Dat Hoang, Nguyen Tuan Ngoc, Dao Xuan Quang Minh, Phan Phi Hai, Nguyen Thi Ngoc Anh, Dang Van Tu, Binh Vu  

**Link**: [PDF](https://arxiv.org/pdf/2512.14554)  

**Abstract**: The rapid advancement of large language models (LLMs) has enabled new possibilities for applying artificial intelligence within the legal domain. Nonetheless, the complexity, hierarchical organization, and frequent revisions of Vietnamese legislation pose considerable challenges for evaluating how well these models interpret and utilize legal knowledge. To address this gap, Vietnamese Legal Benchmark (VLegal-Bench) is introduced, the first comprehensive benchmark designed to systematically assess LLMs on Vietnamese legal tasks. Informed by Bloom's cognitive taxonomy, VLegal-Bench encompasses multiple levels of legal understanding through tasks designed to reflect practical usage scenarios. The benchmark comprises 10,450 samples generated through a rigorous annotation pipeline, where legal experts label and cross-validate each instance using our annotation system to ensure every sample is grounded in authoritative legal documents and mirrors real-world legal assistant workflows, including general legal questions and answers, retrieval-augmented generation, multi-step reasoning, and scenario-based problem solving tailored to Vietnamese law. By providing a standardized, transparent, and cognitively informed evaluation framework, VLegal-Bench establishes a solid foundation for assessing LLM performance in Vietnamese legal contexts and supports the development of more reliable, interpretable, and ethically aligned AI-assisted legal systems. 

---
# Dual Language Models: Balancing Training Efficiency and Overfitting Resilience 

**Authors**: David Samuel, Lucas Georges Gabriel Charpentier  

**Link**: [PDF](https://arxiv.org/pdf/2512.14549)  

**Abstract**: This paper combines autoregressive and masked-diffusion training objectives without any architectural modifications, resulting in flexible language models that outperform single-objective models. Autoregressive modeling has been a popular approach, partly because of its training efficiency; however, that comes at the cost of sensitivity to overfitting. On the other hand, masked-diffusion models are less efficient to train while being more resilient to overfitting. In this work, we demonstrate that dual-objective training achieves the best of both worlds. To derive the optimal ratio between both objectives, we train and evaluate 50 language models under varying levels of data repetition. We show that it is optimal to combine both objectives under all evaluated settings and that the optimal ratio is similar whether targeting autoregressive or masked-diffusion downstream performance. 

---
# VersatileFFN: Achieving Parameter Efficiency in LLMs via Adaptive Wide-and-Deep Reuse 

**Authors**: Ying Nie, Kai Han, Hongguang Li, Hang Zhou, Tianyu Guo, Enhua Wu, Xinghao Chen, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14531)  

**Abstract**: The rapid scaling of Large Language Models (LLMs) has achieved remarkable performance, but it also leads to prohibitive memory costs. Existing parameter-efficient approaches such as pruning and quantization mainly compress pretrained models without enhancing architectural capacity, thereby hitting the representational ceiling of the base model. In this work, we propose VersatileFFN, a novel feed-forward network (FFN) that enables flexible reuse of parameters in both width and depth dimensions within a fixed parameter budget. Inspired by the dual-process theory of cognition, VersatileFFN comprises two adaptive pathways: a width-versatile path that generates a mixture of sub-experts from a single shared FFN, mimicking sparse expert routing without increasing parameters, and a depth-versatile path that recursively applies the same FFN to emulate deeper processing for complex tokens. A difficulty-aware gating dynamically balances the two pathways, steering "easy" tokens through the efficient width-wise route and allocating deeper iterative refinement to "hard" tokens. Crucially, both pathways reuse the same parameters, so all additional capacity comes from computation rather than memory. Experiments across diverse benchmarks and model scales demonstrate the effectiveness of the method. The code will be available at this https URL. 

---
# Linguists should learn to love speech-based deep learning models 

**Authors**: Marianne de Heer Kloots, Paul Boersma, Willem Zuidema  

**Link**: [PDF](https://arxiv.org/pdf/2512.14506)  

**Abstract**: Futrell and Mahowald present a useful framework bridging technology-oriented deep learning systems and explanation-oriented linguistic theories. Unfortunately, the target article's focus on generative text-based LLMs fundamentally limits fruitful interactions with linguistics, as many interesting questions on human language fall outside what is captured by written text. We argue that audio-based deep learning models can and should play a crucial role. 

---
# C-ing Clearly: Enhanced Binary Code Explanations using C code 

**Authors**: Teodor Poncu, Ioana Pintilie, Marius Dragoi, Dragos Tantaru, Florin Brad  

**Link**: [PDF](https://arxiv.org/pdf/2512.14500)  

**Abstract**: Large Language Models (LLMs) typically excel at coding tasks involving high-level programming languages, as opposed to lower-level programming languages, such as assembly. We propose a synthetic data generation method named C-ing Clearly, which leverages the corresponding C code to enhance an LLM's understanding of assembly. By fine-tuning on data generated through our method, we demonstrate improved LLM performance for binary code summarization and vulnerability detection. Our approach demonstrates consistent gains across different LLM families and model sizes. 

---
# SASQ: Static Activation Scaling for Quantization-Aware Training in Large Language Models 

**Authors**: Shizhuo Mao, Song Chen, Yi Kang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14481)  

**Abstract**: Large language models (LLMs) excel at natural language tasks but face deployment challenges due to their growing size outpacing GPU memory advancements. Model quantization mitigates this issue by lowering weight and activation precision, but existing solutions face fundamental trade-offs: dynamic quantization incurs high computational overhead and poses deployment challenges on edge devices, while static quantization sacrifices accuracy. Existing approaches of quantization-aware training (QAT) further suffer from weight training costs. We propose SASQ: a lightweight QAT framework specifically tailored for activation quantization factors. SASQ exclusively optimizes only the quantization factors (without changing pre-trained weights), enabling static inference with high accuracy while maintaining deployment efficiency. SASQ adaptively truncates some outliers, thereby reducing the difficulty of quantization while preserving the distributional characteristics of the activations. SASQ not only surpasses existing SOTA quantization schemes but also outperforms the corresponding FP16 models. On LLaMA2-7B, it achieves 5.2% lower perplexity than QuaRot and 4.7% lower perplexity than the FP16 model on WikiText2. 

---
# Effect of Document Packing on the Latent Multi-Hop Reasoning Capabilities of Large Language Models 

**Authors**: Gabriele Prato, Shagun Sodhani, Alessandro Sordoni, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2512.14427)  

**Abstract**: The standard practice for training large language models involves packing multiple documents together to optimize computational efficiency. However, the impact of this process on the models' capabilities remains largely unexplored. To address this gap, we investigate how different document-packing strategies influence the latent multi-hop reasoning abilities of LLMs. Our findings indicate that packing can improve model performance compared to training on individual documents, at the expense of more compute. To further understand the underlying mechanisms, we conduct an ablation study, identifying key factors that explain the advantages of packing. Ultimately, our research deepens the understanding of LLM training dynamics and provides practical insights for optimizing model development. 

---
# Step-Tagging: Toward controlling the generation of Language Reasoning Models through step monitoring 

**Authors**: Yannis Belkhiter, Seshu Tirupathi, Giulio Zizzo, John D. Kelleher  

**Link**: [PDF](https://arxiv.org/pdf/2512.14332)  

**Abstract**: The field of Language Reasoning Models (LRMs) has been very active over the past few years with advances in training and inference techniques enabling LRMs to reason longer, and more accurately. However, a growing body of studies show that LRMs are still inefficient, over-generating verification and reflection steps. To address this challenge, we introduce the Step-Tagging framework, a lightweight sentence-classifier enabling real-time annotation of the type of reasoning steps that an LRM is generating. To monitor reasoning behaviors, we introduced ReasonType: a novel taxonomy of reasoning steps. Building on this framework, we demonstrated that online monitoring of the count of specific steps can produce effective interpretable early stopping criteria of LRM inferences. We evaluate the Step-tagging framework on three open-source reasoning models across standard benchmark datasets: MATH500, GSM8K, AIME and non-mathematical tasks (GPQA and MMLU-Pro). We achieve 20 to 50\% token reduction while maintaining comparable accuracy to standard generation, with largest gains observed on more computation-heavy tasks. This work offers a novel way to increase control over the generation of LRMs, and a new tool to study behaviors of LRMs. 

---
# Inflation Attitudes of Large Language Models 

**Authors**: Nikoleta Anesti, Edward Hill, Andreas Joseph  

**Link**: [PDF](https://arxiv.org/pdf/2512.14306)  

**Abstract**: This paper investigates the ability of Large Language Models (LLMs), specifically GPT-3.5-turbo (GPT), to form inflation perceptions and expectations based on macroeconomic price signals. We compare the LLM's output to household survey data and official statistics, mimicking the information set and demographic characteristics of the Bank of England's Inflation Attitudes Survey (IAS). Our quasi-experimental design exploits the timing of GPT's training cut-off in September 2021 which means it has no knowledge of the subsequent UK inflation surge. We find that GPT tracks aggregate survey projections and official statistics at short horizons. At a disaggregated level, GPT replicates key empirical regularities of households' inflation perceptions, particularly for income, housing tenure, and social class. A novel Shapley value decomposition of LLM outputs suited for the synthetic survey setting provides well-defined insights into the drivers of model outputs linked to prompt content. We find that GPT demonstrates a heightened sensitivity to food inflation information similar to that of human respondents. However, we also find that it lacks a consistent model of consumer price inflation. More generally, our approach could be used to evaluate the behaviour of LLMs for use in the social sciences, to compare different models, or to assist in survey design. 

---
# From Context to EDUs: Faithful and Structured Context Compression via Elementary Discourse Unit Decomposition 

**Authors**: Yiqing Zhou, Yu Lei, Shuzheng Si, Qingyan Sun, Wei Wang, Yifei Wu, Hao Wen, Gang Chen, Fanchao Qi, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.14244)  

**Abstract**: Managing extensive context remains a critical bottleneck for Large Language Models (LLMs), particularly in applications like long-document question answering and autonomous agents where lengthy inputs incur high computational costs and introduce noise. Existing compression techniques often disrupt local coherence through discrete token removal or rely on implicit latent encoding that suffers from positional bias and incompatibility with closed-source APIs. To address these limitations, we introduce the EDU-based Context Compressor, a novel explicit compression framework designed to preserve both global structure and fine-grained details. Our approach reformulates context compression as a structure-then-select process. First, our LingoEDU transforms linear text into a structural relation tree of Elementary Discourse Units (EDUs) which are anchored strictly to source indices to eliminate hallucination. Second, a lightweight ranking module selects query-relevant sub-trees for linearization. To rigorously evaluate structural understanding, we release StructBench, a manually annotated dataset of 248 diverse documents. Empirical results demonstrate that our method achieves state-of-the-art structural prediction accuracy and significantly outperforms frontier LLMs while reducing costs. Furthermore, our structure-aware compression substantially enhances performance across downstream tasks ranging from long-context tasks to complex Deep Search scenarios. 

---
# Two CFG Nahuatl for automatic corpora expansion 

**Authors**: Juan-José Guzmán-Landa, Juan-Manuel Torres-Moreno, Miguel Figueroa-Saavedra, Ligia Quintana-Torres, Graham Ranger Martha-Lorena Avendaño-Garrido  

**Link**: [PDF](https://arxiv.org/pdf/2512.14239)  

**Abstract**: The aim of this article is to introduce two Context-Free Grammars (CFG) for Nawatl Corpora expansion. Nawatl is an Amerindian language (it is a National Language of Mexico) of the $\pi$-language type, i.e. a language with few digital resources. For this reason the corpora available for the learning of Large Language Models (LLMs) are virtually non-existent, posing a significant challenge. The goal is to produce a substantial number of syntactically valid artificial Nawatl sentences and thereby to expand the corpora for the purpose of learning non contextual embeddings. For this objective, we introduce two new Nawatl CFGs and use them in generative mode. Using these grammars, it is possible to expand Nawatl corpus significantly and subsequently to use it to learn embeddings and to evaluate their relevance in a sentences semantic similarity task. The results show an improvement compared to the results obtained using only the original corpus without artificial expansion, and also demonstrate that economic embeddings often perform better than some LLMs. 

---
# Ladder Up, Memory Down: Low-Cost Fine-Tuning With Side Nets 

**Authors**: Estelle Zheng, Nathan Cerisara, Sébastien Warichet, Emmanuel Helbert, Christophe Cerisara  

**Link**: [PDF](https://arxiv.org/pdf/2512.14237)  

**Abstract**: Fine-tuning large language models (LLMs) is often limited by the memory available on commodity GPUs. Parameter-efficient fine-tuning (PEFT) methods such as QLoRA reduce the number of trainable parameters, yet still incur high memory usage induced by the backward pass in the full model. We revisit Ladder Side Tuning (LST), a rarely explored PEFT technique that adds a lightweight side network, and show that it matches QLoRA's compute scaling slope while cutting peak memory by 50\%. Across different downstream benchmarks spanning natural language understanding, mathematical and LLM-critic tasks, LST has competitive performance with QLoRA's accuracy on average while being much more memory-efficient. This efficiency enables fine-tuning of 7B-parameter models on a single 12 GB consumer GPU with 2k-token contexts, requiring no gradient checkpointing\textemdash conditions under which QLoRA exhausts memory. Beyond memory efficiency, we also establish scaling laws showing that LST scales similarly to QLoRA. We exploit Ladder's architectural flexibility by introducing xLadder, a depth-extended variant that increases effective depth via cross-connections and shortens chain-of-thought (CoT) at fixed parameter count. Ladder is strong when memory is the bottleneck; xLadder builds on this by enabling deeper reasoning without additional memory overhead. 

---
# A Comparative Analysis of Retrieval-Augmented Generation Techniques for Bengali Standard-to-Dialect Machine Translation Using LLMs 

**Authors**: K. M. Jubair Sami, Dipto Sumit, Ariyan Hossain, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2512.14179)  

**Abstract**: Translating from a standard language to its regional dialects is a significant NLP challenge due to scarce data and linguistic variation, a problem prominent in the Bengali language. This paper proposes and compares two novel RAG pipelines for standard-to-dialectal Bengali translation. The first, a Transcript-Based Pipeline, uses large dialect sentence contexts from audio transcripts. The second, a more effective Standardized Sentence-Pairs Pipeline, utilizes structured local\_dialect:standard\_bengali sentence pairs. We evaluated both pipelines across six Bengali dialects and multiple LLMs using BLEU, ChrF, WER, and BERTScore. Our findings show that the sentence-pair pipeline consistently outperforms the transcript-based one, reducing Word Error Rate (WER) from 76\% to 55\% for the Chittagong dialect. Critically, this RAG approach enables smaller models (e.g., Llama-3.1-8B) to outperform much larger models (e.g., GPT-OSS-120B), demonstrating that a well-designed retrieval strategy can be more crucial than model size. This work contributes an effective, fine-tuning-free solution for low-resource dialect translation, offering a practical blueprint for preserving linguistic diversity. 

---
# Astraea: A State-Aware Scheduling Engine for LLM-Powered Agents 

**Authors**: Hongqiu Ni, Jiabao Zhang, Guopeng Li, Zilong Wang, Ruiqi Wu, Chi Zhang, Haisheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2512.14142)  

**Abstract**: Large Language Models (LLMs) are increasingly being deployed as intelligent agents. Their multi-stage workflows, which alternate between local computation and calls to external network services like Web APIs, introduce a mismatch in their execution pattern and the scheduling granularity of existing inference systems such as vLLM. Existing systems typically focus on per-segment optimization which prevents them from minimizing the end-to-end latency of the complete agentic workflow, i.e., the global Job Completion Time (JCT) over the entire request lifecycle. To address this limitation, we propose Astraea, a service engine designed to shift the optimization from local segments to the global request lifecycle. Astraea employs a state-aware, hierarchical scheduling algorithm that integrates a request's historical state with future predictions. It dynamically classifies requests by their I/O and compute intensive nature and uses an enhanced HRRN policy to balance efficiency and fairness. Astraea also implements an adaptive KV cache manager that intelligently handles the agent state during I/O waits based on the system memory pressure. Extensive experiments show that Astraea reduces average JCT by up to 25.5\% compared to baseline methods. Moreover, our approach demonstrates strong robustness and stability under high load across various model scales. 

---
# CogMem: A Cognitive Memory Architecture for Sustained Multi-Turn Reasoning in Large Language Models 

**Authors**: Yiran Zhang, Jincheng Hu, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2512.14118)  

**Abstract**: Large language models (LLMs) excel at single-turn reasoning but often lose accuracy and coherence over extended, multi-turn interactions. Recent evaluations such as TurnBench highlight recurring failure modes-reasoning bias, task drift, hallucination, overconfidence, and memory decay. Current approaches typically append full conversational histories, causing unbounded context growth, higher computational costs, and degraded reasoning efficiency. We introduce CogMem, a cognitively inspired, memory-augmented LLM architecture that supports sustained iterative reasoning through structured, persistent memory. CogMem incorporates three layers: a Long-Term Memory (LTM) that consolidates cross-session reasoning strategies; a Direct Access (DA) memory that maintains session-level notes and retrieves relevant long-term memories; and a Focus of Attention (FoA) mechanism that dynamically reconstructs concise, task-relevant context at each turn. Experiments on TurnBench show that this layered design mitigates reasoning failures, controls context growth, and improves consistency across extended reasoning chains, moving toward more reliable, human-like reasoning in LLMs. 

---
# Multilingual and Continuous Backchannel Prediction: A Cross-lingual Study 

**Authors**: Koji Inoue, Mikey Elmers, Yahui Fu, Zi Haur Pang, Taiga Mori, Divesh Lala, Keiko Ochi, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2512.14085)  

**Abstract**: We present a multilingual, continuous backchannel prediction model for Japanese, English, and Chinese, and use it to investigate cross-linguistic timing behavior. The model is Transformer-based and operates at the frame level, jointly trained with auxiliary tasks on approximately 300 hours of dyadic conversations. Across all three languages, the multilingual model matches or surpasses monolingual baselines, indicating that it learns both language-universal cues and language-specific timing patterns. Zero-shot transfer with two-language training remains limited, underscoring substantive cross-lingual differences. Perturbation analyses reveal distinct cue usage: Japanese relies more on short-term linguistic information, whereas English and Chinese are more sensitive to silence duration and prosodic variation; multilingual training encourages shared yet adaptable representations and reduces overreliance on pitch in Chinese. A context-length study further shows that Japanese is relatively robust to shorter contexts, while Chinese benefits markedly from longer contexts. Finally, we integrate the trained model into a real-time processing software, demonstrating CPU-only inference. Together, these findings provide a unified model and empirical evidence for how backchannel timing differs across languages, informing the design of more natural, culturally-aware spoken dialogue systems. 

---
# A Unified Sparse Attention via Multi-Granularity Compression 

**Authors**: Siran Liu, Zane Cao, Yongchao He  

**Link**: [PDF](https://arxiv.org/pdf/2512.14082)  

**Abstract**: Efficient long-context understanding and reasoning are increasingly vital for large language model (LLM) applications such as multi-turn dialogue and program analysis. However, the core self-attention mechanism scales quadratically with sequence length, creating a fundamental computational bottleneck. Existing sparse attention methods alleviate this issue but face trade-offs: training-based methods are costly and cannot be directly applied as acceleration plugins for other models, while inference-time methods often compromise efficiency or cross-modal generality. To address these limitations, we present UniSparse, a unified mechanism that introduces the notion of composite tokens--compact representations that aggregate multi-granularity contextual information. Building on this abstraction, UniSparse dynamically constructs sparse attention through multi-granularity compression and block-level selection, enabling efficient and hardware-friendly execution on GPU. Across multiple modalities and tasks ranging from synthetic benchmarks to real-world applications, UniSparse consistently surpasses state-of-the-art sparse attention methods (e.g., MInference, XAttention, FlexPrefill) in both accuracy and efficiency, achieving $\ge$ 99% of full-attention accuracy and up to 2.61$\times$ faster attention computation than FlashAttention. 

---
# Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed 

**Authors**: Yonggan Fu, Lexington Whalen, Zhifan Ye, Xin Dong, Shizhe Diao, Jingyu Liu, Chengyue Wu, Hao Zhang, Enze Xie, Song Han, Maksim Khadkevich, Jan Kautz, Yingyan Celine Lin, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2512.14067)  

**Abstract**: Diffusion language models (dLMs) have emerged as a promising paradigm that enables parallel, non-autoregressive generation, but their learning efficiency lags behind that of autoregressive (AR) language models when trained from scratch. To this end, we study AR-to-dLM conversion to transform pretrained AR models into efficient dLMs that excel in speed while preserving AR models' task accuracy. We achieve this by identifying limitations in the attention patterns and objectives of existing AR-to-dLM methods and then proposing principles and methodologies for more effective AR-to-dLM conversion. Specifically, we first systematically compare different attention patterns and find that maintaining pretrained AR weight distributions is critical for effective AR-to-dLM conversion. As such, we introduce a continuous pretraining scheme with a block-wise attention pattern, which remains causal across blocks while enabling bidirectional modeling within each block. We find that this approach can better preserve pretrained AR models' weight distributions than fully bidirectional modeling, in addition to its known benefit of enabling KV caching, and leads to a win-win in accuracy and efficiency. Second, to mitigate the training-test gap in mask token distributions (uniform vs. highly left-to-right), we propose a position-dependent token masking strategy that assigns higher masking probabilities to later tokens during training to better mimic test-time behavior. Leveraging this framework, we conduct extensive studies of dLMs' attention patterns, training dynamics, and other design choices, providing actionable insights into scalable AR-to-dLM conversion. These studies lead to the Efficient-DLM family, which outperforms state-of-the-art AR models and dLMs, e.g., our Efficient-DLM 8B achieves +5.4%/+2.7% higher accuracy with 4.5x/2.7x higher throughput compared to Dream 7B and Qwen3 4B, respectively. 

---
# What Affects the Effective Depth of Large Language Models? 

**Authors**: Yi Hu, Cai Zhou, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14064)  

**Abstract**: The scaling of large language models (LLMs) emphasizes increasing depth, yet performance gains diminish with added layers. Prior work introduces the concept of "effective depth", arguing that deeper models fail to fully utilize their layers for meaningful computation. Building on this, we systematically study how effective depth varies with model scale, training type, and task difficulty. First, we analyze the model behavior of Qwen-2.5 family (1.5B-32B) and find that while the number of effective layers grows with model size, the effective depth ratio remains stable. Besides, comparisons between base and corresponding long-CoT models show no increase in effective depth, suggesting that improved reasoning stems from longer context rather than deeper per-token computation. Furthermore, evaluations across tasks of varying difficulty indicate that models do not dynamically use more layers for harder problems. Our results suggest that current LLMs underuse available depth across scales, training paradigms and tasks of varying difficulties, pointing out research opportunities on increasing the layer utilization rate of LLMs, model pruning, and early exiting. Our code is released at this https URL. 

---
# Structure-Aware Decoding Mechanisms for Complex Entity Extraction with Large-Scale Language Models 

**Authors**: Zhimin Qiu, Di Wu, Feng Liu, Chenrui Hu, Yuxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.13980)  

**Abstract**: This paper proposes a structure-aware decoding method based on large language models to address the difficulty of traditional approaches in maintaining both semantic integrity and structural consistency in nested and overlapping entity extraction tasks. The method introduces a candidate span generation mechanism and structured attention modeling to achieve unified modeling of entity boundaries, hierarchical relationships, and cross-dependencies. The model first uses a pretrained language model to obtain context-aware semantic representations, then captures multi-granular entity span features through candidate representation combinations, and introduces hierarchical structural constraints during decoding to ensure consistency between semantics and structure. To enhance stability in complex scenarios, the model jointly optimizes classification loss and structural consistency loss, maintaining high recognition accuracy under multi-entity co-occurrence and long-sentence dependency conditions. Experiments conducted on the ACE 2005 dataset demonstrate significant improvements in Accuracy, Precision, Recall, and F1-Score, particularly in nested and overlapping entity recognition, where the model shows stronger boundary localization and structural modeling capability. This study verifies the effectiveness of structure-aware decoding in complex semantic extraction tasks, provides a new perspective for developing language models with hierarchical understanding, and establishes a methodological foundation for high-precision information extraction. 

---
# Olmo 3 

**Authors**: Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, Pete Walsh, Pradeep Dasigi, Robert Berry, Saumya Malik, Saurabh Shah, Scott Geng, Shane Arora, Shashank Gupta, Taira Anderson, Teng Xiao, Tyler Murray, Tyler Romero, Victoria Graf, Akari Asai, Akshita Bhagia, Alexander Wettig, Alisa Liu, Aman Rangapur, Chloe Anastasiades, Costa Huang, Dustin Schwenk, Harsh Trivedi, Ian Magnusson, Jaron Lochner, Jiacheng Liu, Lester James V. Miranda, Maarten Sap, Malia Morgan, Michael Schmitz, Michal Guerquin, Michael Wilson, Regan Huff, Ronan Le Bras, Rui Xin, Rulin Shao, Sam Skjonsberg, Shannon Zejiang Shen, Shuyue Stella Li, Tucker Wilde, Valentina Pyatkin, Will Merrill, Yapei Chang, Yuling Gu, Zhiyuan Zeng, Ashish Sabharwal, Luke Zettlemoyer, Pang Wei Koh, Ali Farhadi, Noah A. Smith, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2512.13961)  

**Abstract**: We introduce Olmo 3, a family of state-of-the-art, fully-open language models at the 7B and 32B parameter scales. Olmo 3 model construction targets long-context reasoning, function calling, coding, instruction following, general chat, and knowledge recall. This release includes the entire model flow, i.e., the full lifecycle of the family of models, including every stage, checkpoint, data point, and dependency used to build it. Our flagship model, Olmo 3 Think 32B, is the strongest fully-open thinking model released to-date. 

---
# FiNERweb: Datasets and Artifacts for Scalable Multilingual Named Entity Recognition 

**Authors**: Jonas Golde, Patrick Haller, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2512.13884)  

**Abstract**: Recent multilingual named entity recognition (NER) work has shown that large language models (LLMs) can provide effective synthetic supervision, yet such datasets have mostly appeared as by-products of broader experiments rather than as systematic, reusable resources. We introduce FiNERweb, a dataset-creation pipeline that scales the teacher-student paradigm to 91 languages and 25 scripts. Building on FineWeb-Edu, our approach trains regression models to identify NER-relevant passages and annotates them with multilingual LLMs, resulting in about 225k passages with 235k distinct entity labels. Our experiments show that the regression model achieves more than 84 F1, and that models trained on FiNERweb obtain comparable or improved performance in zero shot transfer settings on English, Thai, and Swahili, despite being trained on 19x less data than strong baselines. In addition, we assess annotation quality using LLM-as-a-judge and observe consistently high scores for both faithfulness (3.99 out of 5) and completeness (4.05 out of 5), indicating reliable and informative annotations. Further, we release the dataset with both English labels and translated label sets in the respective target languages because we observe that the performance of current state-of-the-art models drops by 0.02 to 0.09 F1 when evaluated using target language labels instead of English ones. We release FiNERweb together with all accompanying artifacts to the research community in order to facilitate more effective student-teacher training for multilingual named entity recognition. 

---
# TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs 

**Authors**: Jun Zhang, Teng Wang, Yuying Ge, Yixiao Ge, Xinhao Li, Ying Shan, Limin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14698)  

**Abstract**: This paper does not introduce a novel method but instead establishes a straightforward, incremental, yet essential baseline for video temporal grounding (VTG), a core capability in video understanding. While multimodal large language models (MLLMs) excel at various video understanding tasks, the recipes for optimizing them for VTG remain under-explored. In this paper, we present TimeLens, a systematic investigation into building MLLMs with strong VTG ability, along two primary dimensions: data quality and algorithmic design. We first expose critical quality issues in existing VTG benchmarks and introduce TimeLens-Bench, comprising meticulously re-annotated versions of three popular benchmarks with strict quality criteria. Our analysis reveals dramatic model re-rankings compared to legacy benchmarks, confirming the unreliability of prior evaluation standards. We also address noisy training data through an automated re-annotation pipeline, yielding TimeLens-100K, a large-scale, high-quality training dataset. Building on our data foundation, we conduct in-depth explorations of algorithmic design principles, yielding a series of meaningful insights and effective yet efficient practices. These include interleaved textual encoding for time representation, a thinking-free reinforcement learning with verifiable rewards (RLVR) approach as the training paradigm, and carefully designed recipes for RLVR training. These efforts culminate in TimeLens models, a family of MLLMs with state-of-the-art VTG performance among open-source models and even surpass proprietary models such as GPT-5 and Gemini-2.5-Flash. All codes, data, and models will be released to facilitate future research. 

---
# Segmental Attention Decoding With Long Form Acoustic Encodings 

**Authors**: Pawel Swietojanski, Xinwei Li, Mingbin Xu, Takaaki Hori, Dogan Can, Xiaodan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14652)  

**Abstract**: We address the fundamental incompatibility of attention-based encoder-decoder (AED) models with long-form acoustic encodings. AED models trained on segmented utterances learn to encode absolute frame positions by exploiting limited acoustic context beyond segment boundaries, but fail to generalize when decoding long-form segments where these cues vanish. The model loses ability to order acoustic encodings due to permutation invariance of keys and values in cross-attention. We propose four modifications: (1) injecting explicit absolute positional encodings into cross-attention for each decoded segment, (2) long-form training with extended acoustic context to eliminate implicit absolute position encoding, (3) segment concatenation to cover diverse segmentations needed during training, and (4) semantic segmentation to align AED-decoded segments with training segments. We show these modifications close the accuracy gap between continuous and segmented acoustic encodings, enabling auto-regressive use of the attention decoder. 

---
# RecGPT-V2 Technical Report 

**Authors**: Chao Yi, Dian Chen, Gaoyang Guo, Jiakai Tang, Jian Wu, Jing Yu, Mao Zhang, Wen Chen, Wenjun Yang, Yujie Luo, Yuning Jiang, Zhujin Gao, Bo Zheng, Binbin Cao, Changfa Wu, Dixuan Wang, Han Wu, Haoyi Hu, Kewei Zhu, Lang Tian, Lin Yang, Qiqi Huang, Siqi Yang, Wenbo Su, Xiaoxiao He, Xin Tong, Xu Chen, Xunke Xi, Xiaowei Huang, Yaxuan Wu, Yeqiu Yang, Yi Hu, Yujin Yuan, Yuliang Yan, Zile Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.14503)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable potential in transforming recommender systems from implicit behavioral pattern matching to explicit intent reasoning. While RecGPT-V1 successfully pioneered this paradigm by integrating LLM-based reasoning into user interest mining and item tag prediction, it suffers from four fundamental limitations: (1) computational inefficiency and cognitive redundancy across multiple reasoning routes; (2) insufficient explanation diversity in fixed-template generation; (3) limited generalization under supervised learning paradigms; and (4) simplistic outcome-focused evaluation that fails to match human standards.
To address these challenges, we present RecGPT-V2 with four key innovations. First, a Hierarchical Multi-Agent System restructures intent reasoning through coordinated collaboration, eliminating cognitive duplication while enabling diverse intent coverage. Combined with Hybrid Representation Inference that compresses user-behavior contexts, our framework reduces GPU consumption by 60% and improves exclusive recall from 9.39% to 10.99%. Second, a Meta-Prompting framework dynamically generates contextually adaptive prompts, improving explanation diversity by +7.3%. Third, constrained reinforcement learning mitigates multi-reward conflicts, achieving +24.1% improvement in tag prediction and +13.0% in explanation acceptance. Fourth, an Agent-as-a-Judge framework decomposes assessment into multi-step reasoning, improving human preference alignment. Online A/B tests on Taobao demonstrate significant improvements: +2.98% CTR, +3.71% IPV, +2.19% TV, and +11.46% NER. RecGPT-V2 establishes both the technical feasibility and commercial viability of deploying LLM-powered intent reasoning at scale, bridging the gap between cognitive exploration and industrial utility. 

---
# RePo: Language Models with Context Re-Positioning 

**Authors**: Huayang Li, Tianyu Zhao, Richard Sproat  

**Link**: [PDF](https://arxiv.org/pdf/2512.14391)  

**Abstract**: In-context learning is fundamental to modern Large Language Models (LLMs); however, prevailing architectures impose a rigid and fixed contextual structure by assigning linear or constant positional indices. Drawing on Cognitive Load Theory (CLT), we argue that this uninformative structure increases extraneous cognitive load, consuming finite working memory capacity that should be allocated to deep reasoning and attention allocation. To address this, we propose RePo, a novel mechanism that reduces extraneous load via context re-positioning. Unlike standard approaches, RePo utilizes a differentiable module, $f_\phi$, to assign token positions that capture contextual dependencies, rather than replying on pre-defined integer range. By continually pre-training on the OLMo-2 1B backbone, we demonstrate that RePo significantly enhances performance on tasks involving noisy contexts, structured data, and longer context length, while maintaining competitive performance on general short-context tasks. Detailed analysis reveals that RePo successfully allocate higher attention to distant but relevant information, assign positions in dense and non-linear space, and capture the intrinsic structure of the input context. Our code is available at this https URL. 

---
# SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions 

**Authors**: Panayiotis Smeros, Vincent Emonet, Ruijie Wang, Ana-Claudia Sima, Tarcisio Mendes de Farias  

**Link**: [PDF](https://arxiv.org/pdf/2512.14277)  

**Abstract**: The advent of large language models is contributing to the emergence of novel approaches that promise to better tackle the challenge of generating structured queries, such as SPARQL queries, from natural language. However, these new approaches mostly focus on response accuracy over a single source while ignoring other evaluation criteria, such as federated query capability over distributed data stores, as well as runtime and cost to generate SPARQL queries. Consequently, they are often not production-ready or easy to deploy over (potentially federated) knowledge graphs with good accuracy. To mitigate these issues, in this paper, we extend our previous work and describe and systematically evaluate SPARQL-LLM, an open-source and triplestore-agnostic approach, powered by lightweight metadata, that generates SPARQL queries from natural language text. First, we describe its architecture, which consists of dedicated components for metadata indexing, prompt building, and query generation and execution. Then, we evaluate it based on a state-of-the-art challenge with multilingual questions, and a collection of questions from three of the most prevalent knowledge graphs within the field of bioinformatics. Our results demonstrate a substantial increase of 24% in the F1 Score on the state-of-the-art challenge, adaptability to high-resource languages such as English and Spanish, as well as ability to form complex and federated bioinformatics queries. Furthermore, we show that SPARQL-LLM is up to 36x faster than other systems participating in the challenge, while costing a maximum of $0.01 per question, making it suitable for real-time, low-cost text-to-SPARQL applications. One such application deployed over real-world decentralized knowledge graphs can be found at this https URL. 

---
# Scalable Frameworks for Real-World Audio-Visual Speech Recognition 

**Authors**: Sungnyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.14083)  

**Abstract**: The practical deployment of Audio-Visual Speech Recognition (AVSR) systems is fundamentally challenged by significant performance degradation in real-world environments, characterized by unpredictable acoustic noise and visual interference. This dissertation posits that a systematic, hierarchical approach is essential to overcome these challenges, achieving the robust scalability at the representation, architecture, and system levels. At the representation level, we investigate methods for building a unified model that learns audio-visual features inherently robust to diverse real-world corruptions, thereby enabling generalization to new environments without specialized modules. To address architectural scalability, we explore how to efficiently expand model capacity while ensuring the adaptive and reliable use of multimodal inputs, developing a framework that intelligently allocates computational resources based on the input characteristics. Finally, at the system level, we present methods to expand the system's functionality through modular integration with large-scale foundation models, leveraging their powerful cognitive and generative capabilities to maximize final recognition accuracy. By systematically providing solutions at each of these three levels, this dissertation aims to build a next-generation, robust, and scalable AVSR system with high reliability in real-world applications. 

---
# Grammar Search for Multi-Agent Systems 

**Authors**: Mayank Singh, Vikas Yadav, Shiva Krishna Reddy Malay, Shravan Nayak, Sai Rajeswar, Sathwik Tejaswi Madhusudhan, Eduardo Blanco  

**Link**: [PDF](https://arxiv.org/pdf/2512.14079)  

**Abstract**: Automatic search for Multi-Agent Systems has recently emerged as a key focus in agentic AI research. Several prior approaches have relied on LLM-based free-form search over the code space. In this work, we propose a more structured framework that explores the same space through a fixed set of simple, composable components. We show that, despite lacking the generative flexibility of LLMs during the candidate generation stage, our method outperforms prior approaches on four out of five benchmarks across two domains: mathematics and question answering. Furthermore, our method offers additional advantages, including a more cost-efficient search process and the generation of modular, interpretable multi-agent systems with simpler logic. 

---
# HyperVL: An Efficient and Dynamic Multimodal Large Language Model for Edge Devices 

**Authors**: HyperAI Team, Yuchen Liu, Kaiyang Han, Zhiqiang Xia, Yuhang Dong, Chen Song, Kangyu Tang, Jiaming Xu, Xiushi Feng, WenXuan Yu, Li Peng, Mingyang Wang, Kai Wang, Changpeng Yang, Yang Li, Haoyu Lu, Hao Wang, Bingna Xu, Guangyao Liu, Long Huang, Kaibin Guo, Jinyang Wu, Dan Wu, Hongzhen Wang, Peng Zhou, Shuai Nie, Shande Wang, Runyu Shi, Ying Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.14052)  

**Abstract**: Current multimodal large lanauge models possess strong perceptual and reasoning capabilities, however high computational and memory requirements make them difficult to deploy directly on on-device environments. While small-parameter models are progressively endowed with strong general capabilities, standard Vision Transformer (ViT) encoders remain a critical bottleneck, suffering from excessive latency and memory consumption when processing high-resolution this http URL address these challenges, we introduce HyperVL, an efficient multimodal large language model tailored for on-device inference. HyperVL adopts an image-tiling strategy to cap peak memory usage and incorporates two novel techniques: (1) a Visual Resolution Compressor (VRC) that adaptively predicts optimal encoding resolutions to eliminate redundant computation, and (2) Dual Consistency Learning (DCL), which aligns multi-scale ViT encoders within a unified framework, enabling dynamic switching between visual branches under a shared LLM. Extensive experiments demonstrate that HyperVL achieves state-of-the-art performance among models of comparable size across multiple benchmarks. Furthermore, it significantly significantly reduces latency and power consumption on real mobile devices, demonstrating its practicality for on-device multimodal inference. 

---
# Hierarchical Multi-agent Large Language Model Reasoning for Autonomous Functional Materials Discovery 

**Authors**: Samuel Rothfarb, Megan C. Davis, Ivana Matanovic, Baikun Li, Edward F. Holby, Wilton J.M. Kort-Kamp  

**Link**: [PDF](https://arxiv.org/pdf/2512.13930)  

**Abstract**: Artificial intelligence is reshaping scientific exploration, but most methods automate procedural tasks without engaging in scientific reasoning, limiting autonomy in discovery. We introduce Materials Agents for Simulation and Theory in Electronic-structure Reasoning (MASTER), an active learning framework where large language models autonomously design, execute, and interpret atomistic simulations. In MASTER, a multimodal system translates natural language into density functional theory workflows, while higher-level reasoning agents guide discovery through a hierarchy of strategies, including a single agent baseline and three multi-agent approaches: peer review, triage-ranking, and triage-forms. Across two chemical applications, CO adsorption on Cu-surface transition metal (M) adatoms and on M-N-C catalysts, reasoning-driven exploration reduces required atomistic simulations by up to 90% relative to trial-and-error selection. Reasoning trajectories reveal chemically grounded decisions that cannot be explained by stochastic sampling or semantic bias. Altogether, multi-agent collaboration accelerates materials discovery and marks a new paradigm for autonomous scientific exploration. 

---
# Generative AI for Video Translation: A Scalable Architecture for Multilingual Video Conferencing 

**Authors**: Amirkia Rafiei Oskooei, Eren Caglar, Ibrahim Sahin, Ayse Kayabay, Mehmet S. Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2512.13904)  

**Abstract**: The real-time deployment of cascaded generative AI pipelines for applications like video translation is constrained by significant system-level challenges. These include the cumulative latency of sequential model inference and the quadratic ($\mathcal{O}(N^2)$) computational complexity that renders multi-user video conferencing applications unscalable. This paper proposes and evaluates a practical system-level framework designed to mitigate these critical bottlenecks. The proposed architecture incorporates a turn-taking mechanism to reduce computational complexity from quadratic to linear in multi-user scenarios, and a segmented processing protocol to manage inference latency for a perceptually real-time experience. We implement a proof-of-concept pipeline and conduct a rigorous performance analysis across a multi-tiered hardware setup, including commodity (NVIDIA RTX 4060), cloud (NVIDIA T4), and enterprise (NVIDIA A100) GPUs. Our objective evaluation demonstrates that the system achieves real-time throughput ($\tau < 1.0$) on modern hardware. A subjective user study further validates the approach, showing that a predictable, initial processing delay is highly acceptable to users in exchange for a smooth, uninterrupted playback experience. The work presents a validated, end-to-end system design that offers a practical roadmap for deploying scalable, real-time generative AI applications in multilingual communication platforms. 

---
# Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs 

**Authors**: Rachit Bansal, Aston Zhang, Rishabh Tiwari, Lovish Madaan, Sai Surya Duvvuri, Devvrit Khatri, David Brandfonbrener, David Alvarez-Melis, Prajjwal Bhargava, Mihir Sanjay Kale, Samy Jelassi  

**Link**: [PDF](https://arxiv.org/pdf/2512.13898)  

**Abstract**: Progress on training and architecture strategies has enabled LLMs with millions of tokens in context length. However, empirical evidence suggests that such long-context LLMs can consume far more text than they can reliably use. On the other hand, it has been shown that inference-time compute can be used to scale performance of LLMs, often by generating thinking tokens, on challenging tasks involving multi-step reasoning. Through controlled experiments on sandbox long-context tasks, we find that such inference-time strategies show rapidly diminishing returns and fail at long context. We attribute these failures to score dilution, a phenomenon inherent to static self-attention. Further, we show that current inference-time strategies cannot retrieve relevant long-context signals under certain conditions. We propose a simple method that, through targeted gradient updates on the given context, provably overcomes limitations of static self-attention. We find that this shift in how inference-time compute is spent leads to consistently large performance improvements across models and long-context benchmarks. Our method leads to large 12.6 and 14.1 percentage point improvements for Qwen3-4B on average across subsets of LongBench-v2 and ZeroScrolls benchmarks. The takeaway is practical: for long context, a small amount of context-specific training is a better use of inference compute than current inference-time scaling strategies like producing more thinking tokens. 

---
# EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery 

**Authors**: Kamer Ali Yuksel  

**Link**: [PDF](https://arxiv.org/pdf/2512.13857)  

**Abstract**: Large language models (LLMs) are increasingly used to evolve programs and multi-agent systems, yet most existing approaches rely on overwrite-based mutations that maintain only a single candidate at a time. Such methods discard useful variants, suffer from destructive edits, and explore a brittle search space prone to structural failure. We introduce EvoLattice, a framework that represents an entire population of candidate programs or agent behaviors within a single directed acyclic graph. Each node stores multiple persistent alternatives, and every valid path through the graph defines a distinct executable candidate, yielding a large combinatorial search space without duplicating structure. EvoLattice enables fine-grained alternative-level evaluation by scoring each alternative across all paths in which it appears, producing statistics that reveal how local design choices affect global performance. These statistics provide a dense, data-driven feedback signal for LLM-guided mutation, recombination, and pruning, while preserving successful components. Structural correctness is guaranteed by a deterministic self-repair mechanism that enforces acyclicity and dependency consistency independently of the LLM. EvoLattice naturally extends to agent evolution by interpreting alternatives as prompt fragments or sub-agent behaviors. Across program synthesis (proxy and optimizer meta-learning), EvoLattice yields more stable evolution, greater expressivity, and stronger improvement trajectories than prior LLM-guided methods. The resulting dynamics resemble quality-diversity optimization, emerging implicitly from EvoLattice's internal multi-alternative representation rather than an explicit external archive. 

---
# Mitigating Catastrophic Forgetting in Mathematical Reasoning Finetuning through Mixed Training 

**Authors**: John Graham Reynolds  

**Link**: [PDF](https://arxiv.org/pdf/2512.13706)  

**Abstract**: When finetuning large language models for specialized tasks such as mathematical reasoning, models exhibit catastrophic forgetting, losing previously learned capabilities. We investigate this by finetuning Flan-T5-Base (250M parameters) on the DeepMind Mathematics dataset and measuring forgetting on MultiNLI. Math-only training improves mathematical accuracy from 3.1\% to 12.0\% but causes NLI accuracy to collapse from 81.0\% to 16.5\%--a 64.5 percentage point drop occurring within the first 1,000 training steps. We propose mixed training strategies that interleave mathematical and NLI examples during training. Our results demonstrate that mixed training completely eliminates catastrophic forgetting while maintaining equivalent mathematical performance: the balanced 1:1 ratio achieves 12.0\% math accuracy (matching math-only) while preserving 86.2\% NLI accuracy. We systematically explore mixing ratios from 1:1 to 15:1, finding that even minimal NLI exposure (6.2\%) provides effective regularization. These findings demonstrate that specialization need not require forgetting general capabilities, with implications for scaling to larger models where mixed training may confer additional benefits beyond forgetting prevention. 

---
# Leveraging LLMs for Structured Data Extraction from Unstructured Patient Records 

**Authors**: Mitchell A. Klusty, Elizabeth C. Solie, Caroline N. Leach, W. Vaiden Logan, Lynnet E. Richey, John C. Gensel, David P. Szczykutowicz, Bryan C. McLellan, Emily B. Collier, Samuel E. Armstrong, V.K. Cody Bumgardner  

**Link**: [PDF](https://arxiv.org/pdf/2512.13700)  

**Abstract**: Manual chart review remains an extremely time-consuming and resource-intensive component of clinical research, requiring experts to extract often complex information from unstructured electronic health record (EHR) narratives. We present a secure, modular framework for automated structured feature extraction from clinical notes leveraging locally deployed large language models (LLMs) on institutionally approved, Health Insurance Portability and Accountability Act (HIPPA)-compliant compute infrastructure. This system integrates retrieval augmented generation (RAG) and structured response methods of LLMs into a widely deployable and scalable container to provide feature extraction for diverse clinical domains. In evaluation, the framework achieved high accuracy across multiple medical characteristics present in large bodies of patient notes when compared against an expert-annotated dataset and identified several annotation errors missed in manual review. This framework demonstrates the potential of LLM systems to reduce the burden of manual chart review through automated extraction and increase consistency in data capture, accelerating clinical research. 

---
# Writing in Symbiosis: Mapping Human Creative Agency in the AI Era 

**Authors**: Vivan Doshi, Mengyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.13697)  

**Abstract**: The proliferation of Large Language Models (LLMs) raises a critical question about what it means to be human when we share an increasingly symbiotic relationship with persuasive and creative machines. This paper examines patterns of human-AI coevolution in creative writing, investigating how human craft and agency are adapting alongside machine capabilities. We challenge the prevailing notion of stylistic homogenization by examining diverse patterns in longitudinal writing data. Using a large-scale corpus spanning the pre- and post-LLM era, we observe patterns suggestive of a "Dual-Track Evolution": thematic convergence around AI-related topics, coupled with structured stylistic differentiation. Our analysis reveals three emergent adaptation patterns: authors showing increased similarity to AI style, those exhibiting decreased similarity, and those maintaining stylistic stability while engaging with AI-related themes. This Creative Archetype Map illuminates how authorship is coevolving with AI, contributing to discussions about human-AI collaboration, detection challenges, and the preservation of creative diversity. 

---
# Shakespeare, Entropy and Educated Monkeys 

**Authors**: Ioannis Kontoyiannis  

**Link**: [PDF](https://arxiv.org/pdf/2512.11880)  

**Abstract**: It has often been said, correctly, that a monkey forever randomly typing on a keyboard would eventually produce the complete works of William Shakespeare. Almost just as often it has been pointed out that this "eventually" is well beyond any conceivably relevant time frame. We point out that an educated monkey that still types at random but is constrained to only write "statistically typical" text, would produce any given passage in a much shorter time. Information theory gives a very simple way to estimate that time. For example, Shakespeare's phrase, Better three hours too soon than a minute too late, from The Merry Wives of Windsor, would take the educated monkey only 73 thousand years to produce, compared to the beyond-astronomical $2.7 \times 10^{63}$ years for the randomly typing one. Despite the obvious improvement, it would still take the educated monkey an unimaginably long $10^{42,277}$ years to produce all of Hamlet. 

---
