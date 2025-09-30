# InfoAgent: Advancing Autonomous Information-Seeking Agents 

**Authors**: Gongrui Zhang, Jialiang Zhu, Ruiqi Yang, Kai Qiu, Miaosen Zhang, Zhirong Wu, Qi Dai, Bei Liu, Chong Luo, Zhengyuan Yang, Linjie Li, Lijuan Wang, Weizhu Chen, Yuan Zhang, Xin Li, Zhaoyi Liu, Xin Geng, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.25189)  

**Abstract**: Building Large Language Model agents that expand their capabilities by interacting with external tools represents a new frontier in AI research and applications. In this paper, we introduce InfoAgent, a deep research agent powered by an innovative data synthesis pipeline and orchestrated web search tools. To construct challenging, hard-to-find queries,we build entity trees and apply sub-tree sampling with entity fuzzification to systematically increase question difficulty. Unlike prior work that relies heavily on commercial search tools, we develop a dedicated self-hosted search infrastructure, enhancing transparency of agent environments and facilitating further advancement of agent capacity. We evaluate the effectiveness of our data pipeline by measuring the average number of tool calls required to correctly answer a question, and also show that our agent yields better performance when equipped with our tools. Our \mbox{InfoAgent} is post-trained from Qwen3-14B using a two-stage recipe: cold-start supervised finetuning to instill long-horizon search behaviors, followed by reinforcement learning which significantly improves reasoning-driven tool use. With our methods, InfoAgent achieves 15.3\% accuracy on BrowseComp, 29.2\% on BrowseComp-ZH, and 40.4\% on Xbench-DS, outperforming prior open-source deep research agents such as WebSailor-72B and DeepDive-32B. 

---
# Learning to Parallel: Accelerating Diffusion Large Language Models via Adaptive Parallel Decoding 

**Authors**: Wenrui Bao, Zhiben Chen, Dan Xu, Yuzhang Shang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25188)  

**Abstract**: Autoregressive decoding in large language models (LLMs) requires $\mathcal{O}(n)$ sequential steps for $n$ tokens, fundamentally limiting inference throughput. Recent diffusion-based LLMs (dLLMs) enable parallel token generation through iterative denoising. However, current parallel decoding strategies rely on fixed, input-agnostic heuristics (e.g., confidence thresholds), which fail to adapt to input-specific characteristics, resulting in suboptimal speed-quality trade-offs across diverse NLP tasks. In this work, we explore a more flexible and dynamic approach to parallel decoding. We propose Learning to Parallel Decode (Learn2PD), a framework that trains a lightweight and adaptive filter model to predict, for each token position, whether the current prediction matches the final output. This learned filter approximates an oracle parallel decoding strategy that unmasks tokens only when correctly predicted. Importantly, the filter model is learned in a post-training manner, requiring only a small amount of computation to optimize it (minute-level GPU time). Additionally, we introduce End-of-Text Prediction (EoTP) to detect decoding completion at the end of sequence, avoiding redundant decoding of padding tokens. Experiments on the LLaDA benchmark demonstrate that our method achieves up to 22.58$\times$ speedup without any performance drop, and up to 57.51$\times$ when combined with KV-Cache. 

---
# Incentive-Aligned Multi-Source LLM Summaries 

**Authors**: Yanchen Jiang, Zhe Feng, Aranyak Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2509.25184)  

**Abstract**: Large language models (LLMs) are increasingly used in modern search and answer systems to synthesize multiple, sometimes conflicting, texts into a single response, yet current pipelines offer weak incentives for sources to be accurate and are vulnerable to adversarial content. We introduce Truthful Text Summarization (TTS), an incentive-aligned framework that improves factual robustness without ground-truth labels. TTS (i) decomposes a draft synthesis into atomic claims, (ii) elicits each source's stance on every claim, (iii) scores sources with an adapted multi-task peer-prediction mechanism that rewards informative agreement, and (iv) filters unreliable sources before re-summarizing. We establish formal guarantees that align a source's incentives with informative honesty, making truthful reporting the utility-maximizing strategy. Experiments show that TTS improves factual accuracy and robustness while preserving fluency, aligning exposure with informative corroboration and disincentivizing manipulation. 

---
# NAIPv2: Debiased Pairwise Learning for Efficient Paper Quality Estimation 

**Authors**: Penghai Zhao, Jinyu Tian, Qinghua Xing, Xin Zhang, Zheng Li, Jianjun Qian, Ming-Ming Cheng, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.25179)  

**Abstract**: The ability to estimate the quality of scientific papers is central to how both humans and AI systems will advance scientific knowledge in the future. However, existing LLM-based estimation methods suffer from high inference cost, whereas the faster direct score regression approach is limited by scale inconsistencies. We present NAIPv2, a debiased and efficient framework for paper quality estimation. NAIPv2 employs pairwise learning within domain-year groups to reduce inconsistencies in reviewer ratings and introduces the Review Tendency Signal (RTS) as a probabilistic integration of reviewer scores and confidences. To support training and evaluation, we further construct NAIDv2, a large-scale dataset of 24,276 ICLR submissions enriched with metadata and detailed structured content. Trained on pairwise comparisons but enabling efficient pointwise prediction at deployment, NAIPv2 achieves state-of-the-art performance (78.2% AUC, 0.432 Spearman), while maintaining scalable, linear-time efficiency at inference. Notably, on unseen NeurIPS submissions, it further demonstrates strong generalization, with predicted scores increasing consistently across decision categories from Rejected to Oral. These findings establish NAIPv2 as a debiased and scalable framework for automated paper quality estimation, marking a step toward future scientific intelligence systems. Code and dataset are released at this https URL. 

---
# EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering 

**Authors**: Haolei Xu, Xinyu Mei, Yuchen Yan, Rui Zhou, Wenqi Zhang, Weiming Lu, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.25175)  

**Abstract**: Large language model (LLM) steering has emerged as a promising paradigm for controlling model behavior at inference time through targeted manipulation of hidden states, offering a lightweight alternative to expensive retraining. However, existing steering frameworks suffer from critical limitations: computational inefficiency, limited extensibility, and restricted functionality that hinder both research progress and practical deployment. We present EasySteer, a unified framework for high-performance, extensible LLM steering built on vLLM. Our system features modular architecture with pluggable interfaces for both analysis-based and learning-based methods, fine-grained parameter control, pre-computed steering vectors for eight application domains, and an interactive demonstration system. Through deep integration with vLLM's optimized inference engine, EasySteer achieves 5.5-11.4$\times$ speedup over existing frameworks. Extensive experiments demonstrate its effectiveness in overthinking mitigation, hallucination reduction, and other key applications. EasySteer transforms steering from research technique to production-ready capability, establishing critical infrastructure for deployable, controllable language models. 

---
# Pretraining Large Language Models with NVFP4 

**Authors**: NVIDIA, Felix Abecassis, Anjulie Agrusa, Dong Ahn, Jonah Alben, Stefania Alborghetti, Michael Andersch, Sivakumar Arayandi, Alexis Bjorlin, Aaron Blakeman, Evan Briones, Ian Buck, Bryan Catanzaro, Jinhang Choi, Mike Chrzanowski, Eric Chung, Victor Cui, Steve Dai, Bita Darvish Rouhani, Carlo del Mundo, Deena Donia, Burc Eryilmaz, Henry Estela, Abhinav Goel, Oleg Goncharov, Yugi Guvvala, Robert Hesse, Russell Hewett, Herbert Hum, Ujval Kapasi, Brucek Khailany, Mikail Khona, Nick Knight, Alex Kondratenko, Ronny Krashinsky, Ben Lanir, Simon Layton, Michael Lightstone, Daniel Lo, Paulius Micikevicius, Asit Mishra, Tim Moon, Deepak Narayanan, Chao Ni, Abhijit Paithankar, Satish Pasumarthi, Ankit Patel, Mostofa Patwary, Ashwin Poojary, Gargi Prasad, Sweta Priyadarshi, Yigong Qin, Xiaowei Ren, Oleg Rybakov, Charbel Sakr, Sanjeev Satheesh, Stas Sergienko, Pasha Shamis, Kirthi Shankar, Nishant Sharma, Mohammad Shoeybi, Michael Siu, Misha Smelyanskiy, Darko Stosic, Dusan Stosic, Bor-Yiing Su, Frank Sun, Nima Tajbakhsh, Shelby Thomas, Przemek Tredak, Evgeny Tsykunov, Gandhi Vaithilingam, Aditya Vavre, Rangharajan Venkatesan, Roger Waleffe, Qiyu Wan, Hexin Wang, Mengdi Wang, Lizzie Wei, Hao Wu, Evan Wu, Keith Wyss, Ning Xu, Jinze Xue, Charlene Yang, Yujia Zhai, Ruoxi Zhang, Jingyang Zhu, Zhongbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25149)  

**Abstract**: Large Language Models (LLMs) today are powerful problem solvers across many domains, and they continue to get stronger as they scale in model size, training set size, and training set quality, as shown by extensive research and experimentation across the industry. Training a frontier model today requires on the order of tens to hundreds of yottaflops, which is a massive investment of time, compute, and energy. Improving pretraining efficiency is therefore essential to enable the next generation of even more capable LLMs. While 8-bit floating point (FP8) training is now widely adopted, transitioning to even narrower precision, such as 4-bit floating point (FP4), could unlock additional improvements in computational speed and resource utilization. However, quantization at this level poses challenges to training stability, convergence, and implementation, notably for large-scale models trained on long token horizons.
In this study, we introduce a novel approach for stable and accurate training of large language models (LLMs) using the NVFP4 format. Our method integrates Random Hadamard transforms (RHT) to bound block-level outliers, employs a two-dimensional quantization scheme for consistent representations across both the forward and backward passes, utilizes stochastic rounding for unbiased gradient estimation, and incorporates selective high-precision layers. We validate our approach by training a 12-billion-parameter model on 10 trillion tokens -- the longest publicly documented training run in 4-bit precision to date. Our results show that the model trained with our NVFP4-based pretraining technique achieves training loss and downstream task accuracies comparable to an FP8 baseline. These findings highlight that NVFP4, when combined with our training approach, represents a major step forward in narrow-precision LLM training algorithms. 

---
# Paired by the Teacher: Turning Unpaired Data into High-Fidelity Pairs for Low-Resource Text Generation 

**Authors**: Yen-Ju Lu, Thomas Thebaud, Laureano Moro-Velazquez, Najim Dehak, Jesus Villalba  

**Link**: [PDF](https://arxiv.org/pdf/2509.25144)  

**Abstract**: We present Paired by the Teacher (PbT), a two-stage teacher-student pipeline that synthesizes accurate input-output pairs without human labels or parallel data. In many low-resource natural language generation (NLG) scenarios, practitioners may have only raw outputs, like highlights, recaps, or questions, or only raw inputs, such as articles, dialogues, or paragraphs, but seldom both. This mismatch forces small models to learn from very few examples or rely on costly, broad-scope synthetic examples produced by large LLMs. PbT addresses this by asking a teacher LLM to compress each unpaired example into a concise intermediate representation (IR), and training a student to reconstruct inputs from IRs. This enables outputs to be paired with student-generated inputs, yielding high-quality synthetic data. We evaluate PbT on five benchmarks-document summarization (XSum, CNNDM), dialogue summarization (SAMSum, DialogSum), and question generation (SQuAD)-as well as an unpaired setting on SwitchBoard (paired with DialogSum summaries). An 8B student trained only on PbT data outperforms models trained on 70 B teacher-generated corpora and other unsupervised baselines, coming within 1.2 ROUGE-L of human-annotated pairs and closing 82% of the oracle gap at one-third the annotation cost of direct synthesis. Human evaluation on SwitchBoard further confirms that only PbT produces concise, faithful summaries aligned with the target style, highlighting its advantage of generating in-domain sources that avoid the mismatch, limiting direct synthesis. 

---
# Investigating Language and Retrieval Bias in Multilingual Previously Fact-Checked Claim Detection 

**Authors**: Ivan Vykopal, Antonia Karamolegkou, Jaroslav Kopčan, Qiwei Peng, Tomáš Javůrek, Michal Gregor, Marián Šimko  

**Link**: [PDF](https://arxiv.org/pdf/2509.25138)  

**Abstract**: Multilingual Large Language Models (LLMs) offer powerful capabilities for cross-lingual fact-checking. However, these models often exhibit language bias, performing disproportionately better on high-resource languages such as English than on low-resource counterparts. We also present and inspect a novel concept - retrieval bias, when information retrieval systems tend to favor certain information over others, leaving the retrieval process skewed. In this paper, we study language and retrieval bias in the context of Previously Fact-Checked Claim Detection (PFCD). We evaluate six open-source multilingual LLMs across 20 languages using a fully multilingual prompting strategy, leveraging the AMC-16K dataset. By translating task prompts into each language, we uncover disparities in monolingual and cross-lingual performance and identify key trends based on model family, size, and prompting strategy. Our findings highlight persistent bias in LLM behavior and offer recommendations for improving equity in multilingual fact-checking. To investigate retrieval bias, we employed multilingual embedding models and look into the frequency of retrieved claims. Our analysis reveals that certain claims are retrieved disproportionately across different posts, leading to inflated retrieval performance for popular claims while under-representing less common ones. 

---
# Knowledge Extraction on Semi-Structured Content: Does It Remain Relevant for Question Answering in the Era of LLMs? 

**Authors**: Kai Sun, Yin Huang, Srishti Mehra, Mohammad Kachuee, Xilun Chen, Renjie Tao, Zhaojiang Lin, Andrea Jessee, Nirav Shah, Alex Betty, Yue Liu, Anuj Kumar, Wen-tau Yih, Xin Luna Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.25107)  

**Abstract**: The advent of Large Language Models (LLMs) has significantly advanced web-based Question Answering (QA) systems over semi-structured content, raising questions about the continued utility of knowledge extraction for question answering. This paper investigates the value of triple extraction in this new paradigm by extending an existing benchmark with knowledge extraction annotations and evaluating commercial and open-source LLMs of varying sizes. Our results show that web-scale knowledge extraction remains a challenging task for LLMs. Despite achieving high QA accuracy, LLMs can still benefit from knowledge extraction, through augmentation with extracted triples and multi-task learning. These findings provide insights into the evolving role of knowledge triple extraction in web-based QA and highlight strategies for maximizing LLM effectiveness across different model sizes and resource settings. 

---
# Towards Personalized Deep Research: Benchmarks and Evaluations 

**Authors**: Yuan Liang, Jiaxian Li, Yuqing Wang, Piaohong Wang, Motong Tian, Pai Liu, Shuofei Qiao, Runnan Fang, He Zhu, Ge Zhang, Minghao Liu, Yuchen Eleanor Jiang, Ningyu Zhang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.25106)  

**Abstract**: Deep Research Agents (DRAs) can autonomously conduct complex investigations and generate comprehensive reports, demonstrating strong real-world potential. However, existing evaluations mostly rely on close-ended benchmarks, while open-ended deep research benchmarks remain scarce and typically neglect personalized scenarios. To bridge this gap, we introduce Personalized Deep Research Bench, the first benchmark for evaluating personalization in DRAs. It pairs 50 diverse research tasks across 10 domains with 25 authentic user profiles that combine structured persona attributes with dynamic real-world contexts, yielding 250 realistic user-task queries. To assess system performance, we propose the PQR Evaluation Framework, which jointly measures (P) Personalization Alignment, (Q) Content Quality, and (R) Factual Reliability. Our experiments on a range of systems highlight current capabilities and limitations in handling personalized deep research. This work establishes a rigorous foundation for developing and evaluating the next generation of truly personalized AI research assistants. 

---
# Towards Trustworthy Lexical Simplification: Exploring Safety and Efficiency with Small LLMs 

**Authors**: Akio Hayakawa, Stefan Bott, Horacio Saggion  

**Link**: [PDF](https://arxiv.org/pdf/2509.25086)  

**Abstract**: Despite their strong performance, large language models (LLMs) face challenges in real-world application of lexical simplification (LS), particularly in privacy-sensitive and resource-constrained environments. Moreover, since vulnerable user groups (e.g., people with disabilities) are one of the key target groups of this technology, it is crucial to ensure the safety and correctness of the output of LS systems. To address these issues, we propose an efficient framework for LS systems that utilizes small LLMs deployable in local environments. Within this framework, we explore knowledge distillation with synthesized data and in-context learning as baselines. Our experiments in five languages evaluate model outputs both automatically and manually. Our manual analysis reveals that while knowledge distillation boosts automatic metric scores, it also introduces a safety trade-off by increasing harmful simplifications. Importantly, we find that the model's output probability is a useful signal for detecting harmful simplifications. Leveraging this, we propose a filtering strategy that suppresses harmful simplifications while largely preserving beneficial ones. This work establishes a benchmark for efficient and safe LS with small LLMs. It highlights the key trade-offs between performance, efficiency, and safety, and demonstrates a promising approach for safe real-world deployment. 

---
# jina-reranker-v3: Last but Not Late Interaction for Document Reranking 

**Authors**: Feng Wang, Yuqing Li, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25085)  

**Abstract**: jina-reranker-v3 is a 0.6B parameter multilingual document reranker that introduces a novel last but not late interaction. Unlike late interaction models such as ColBERT that perform separate encoding followed by multi-vector matching, our approach conducts causal self-attention between query and documents within the same context window, enabling rich cross-document interactions before extracting contextual embeddings from the last token of each document. This compact architecture achieves state-of-the-art BEIR performance with 61.94 nDCG@10 while being ten times smaller than generative listwise rerankers. 

---
# Scaling Generalist Data-Analytic Agents 

**Authors**: Shuofei Qiao, Yanqiu Zhao, Zhisong Qiu, Xiaobin Wang, Jintian Zhang, Zhao Bin, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.25084)  

**Abstract**: Data-analytic agents are emerging as a key catalyst for automated scientific discovery and for the vision of Innovating AI. Current approaches, however, rely heavily on prompt engineering over proprietary models, while open-source models struggle to face diverse-format, large-scale data files and long-horizon, multi-step reasoning that real-world analytics demands. This paper introduces DataMind, a scalable data synthesis and agent training recipe designed to build generalist data-analytic agents. DataMind tackles three key challenges in building open-source data-analytic agents, including insufficient data resources, improper training strategy, and unstable code-based multi-turn rollout. Concretely, DataMind applies 1) a fine-grained task taxonomy and a recursive easy-to-hard task composition mechanism to increase the diversity and difficulty of synthesized queries; 2) a knowledge-augmented trajectory sampling strategy followed by model-based and rule-based filtering; 3) a dynamically adjustable training objective combining both SFT and RL losses; 4) a memory-frugal and stable code-based multi-turn rollout framework. Built on DataMind, we curate DataMind-12K, a high-quality trajectory set spanning diverse domains, task categories, and data file formats for data-analytic tasks. Trained on DataMind-12K, our DataMind-14B achieves state-of-the-art with an average score of 71.16% on multiple data analysis benchmarks, outperforming the strongest proprietary baselines DeepSeek-V3.1 and GPT-5. Our DataMind-7B also performs best among all open-source models with a score of 68.10%. We also incorporate some empirical insights gained from our exploratory trials into the analysis experiments, aiming to provide actionable insights about agentic training for the community. We will release DataMind-12K and DataMind-7B,14B for the community's future research. 

---
# An empirical study on the limitation of Transformers in program trace generation 

**Authors**: Simeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.25073)  

**Abstract**: We study Transformers on the task \emph{program trace generation} (PTG), where models produce step-by-step execution traces for synthetic programs. Unlike existing algorithmic problems, PTG externalizes reasoning through long traces where each step is trivial. We train small Transformers with diverse modifications, including alternative position encodings, softmax replacements, hybrid model, and short convolutions. While these models achieve strong in-distribution accuracy, they exhibit systematic failures when generalizing to various factors (e.g., program length, trace steps), though some designs significantly improve generalization. 

---
# Confidence-Guided Error Correction for Disordered Speech Recognition 

**Authors**: Abner Hernandez, Tomás Arias Vergara, Andreas Maier, Paula Andrea Pérez-Toro  

**Link**: [PDF](https://arxiv.org/pdf/2509.25048)  

**Abstract**: We investigate the use of large language models (LLMs) as post-processing modules for automatic speech recognition (ASR), focusing on their ability to perform error correction for disordered speech. In particular, we propose confidence-informed prompting, where word-level uncertainty estimates are embedded directly into LLM training to improve robustness and generalization across speakers and datasets. This approach directs the model to uncertain ASR regions and reduces overcorrection. We fine-tune a LLaMA 3.1 model and compare our approach to both transcript-only fine-tuning and post hoc confidence-based filtering. Evaluations show that our method achieves a 10% relative WER reduction compared to naive LLM correction on the Speech Accessibility Project spontaneous speech and a 47% reduction on TORGO, demonstrating the effectiveness of confidence-aware fine-tuning for impaired speech. 

---
# Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures 

**Authors**: Marco Bronzini, Carlo Nicolini, Bruno Lepri, Jacopo Staiano, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2509.25045)  

**Abstract**: Despite their capabilities, Large Language Models (LLMs) remain opaque with limited understanding of their internal representations. Current interpretability methods, such as direct logit attribution (DLA) and sparse autoencoders (SAEs), provide restricted insight due to limitations such as the model's output vocabulary or unclear feature names. This work introduces Hyperdimensional Probe, a novel paradigm for decoding information from the LLM vector space. It combines ideas from symbolic representations and neural probing to project the model's residual stream into interpretable concepts via Vector Symbolic Architectures (VSAs). This probe combines the strengths of SAEs and conventional probes while overcoming their key limitations. We validate our decoding paradigm with controlled input-completion tasks, probing the model's final state before next-token prediction on inputs spanning syntactic pattern recognition, key-value associations, and abstract inference. We further assess it in a question-answering setting, examining the state of the model both before and after text generation. Our experiments show that our probe reliably extracts meaningful concepts across varied LLMs, embedding sizes, and input domains, also helping identify LLM failures. Our work advances information decoding in LLM vector space, enabling extracting more informative, interpretable, and structured features from neural representations. 

---
# GateMABSA: Aspect-Image Gated Fusion for Multimodal Aspect-based Sentiment Analysis 

**Authors**: Adamu Lawan, Haruna Yunusa  

**Link**: [PDF](https://arxiv.org/pdf/2509.25037)  

**Abstract**: Aspect-based Sentiment Analysis (ABSA) has recently advanced into the multimodal domain, where user-generated content often combines text and images. However, existing multimodal ABSA (MABSA) models struggle to filter noisy visual signals, and effectively align aspects with opinion-bearing content across modalities. To address these challenges, we propose GateMABSA, a novel gated multimodal architecture that integrates syntactic, semantic, and fusion-aware mLSTM. Specifically, GateMABSA introduces three specialized mLSTMs: Syn-mLSTM to incorporate syntactic structure, Sem-mLSTM to emphasize aspect--semantic relevance, and Fuse-mLSTM to perform selective multimodal fusion. Extensive experiments on two benchmark Twitter datasets demonstrate that GateMABSA outperforms several baselines. 

---
# Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct 

**Authors**: Haoyang Zheng, Xinyang Liu, Cindy Xiangrui Kong, Nan Jiang, Zheyuan Hu, Weijian Luo, Wei Deng, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.25035)  

**Abstract**: Fast generation of language texts is the holy grail that people pursue in the AI era. In this work, we introduced Discrete Diffusion Divergence Instruct (DiDi-Instruct), a training-based method that leads to fast language generation models by initializing from a pre-trained (masked) discrete diffusion language model (dLLM). The resulting DiDi-Instruct model outperforms the dLLM counterparts and the GPT-2 baseline with 64x acceleration. In the theoretical part of the paper, we build the foundation of DiDi-Instruct in a framework of integral KL-divergence minimization, with practical training algorithms. We also introduce techniques like grouped reward normalization, intermediate-state matching, and the reward-guided ancestral sampler (RGAS) that significantly improve the training stability, the model coverage, and the inference performances. On OpenWebText, DiDi-Instruct outperforms all accelerated language generation models as well as the GPT-2 baseline and the standard dLLMs, achieving sample perplexities ranging from 62.2 (8 NFEs) to 18.4 (128 NFEs). These performance gains are accomplished with a negligible entropy loss of about 1% and 20x less additional training wall-clock time. We further validate the robustness and effectiveness of DiDi-Instruct through extensive ablation studies, model scaling, and the generation of discrete protein sequences. In conclusion, DiDi-Instruct is an efficient yet effective distillation method, enabling language generation in the blink of an eye. We will release both code and models at this http URL. 

---
# Circuit Distillation 

**Authors**: Somin Wadhwa, Silvio Amir, Byron C. Wallace  

**Link**: [PDF](https://arxiv.org/pdf/2509.25002)  

**Abstract**: Model distillation typically focuses on behavioral mimicry, where a student model is trained to replicate a teacher's output while treating its internal computations as a black box. In this work we propose an alternative approach: Distilling the underlying computational mechanisms implemented by a teacher model. Specifically, we propose circuit distillation, which introduces an objective to align internal representations between analogous circuit components in teacher and student models. We propose a method to match ``functionally correspondent'' circuit components and introduce a loss reflecting similarities between the representations that these induce. We evaluate circuit distillation on entity tracking and theory of mind (ToM) tasks using models from the Llama3 family. Our results demonstrate that circuit distillation outperforms standard distillation, successfully transferring algorithmic capabilities by adjusting only a small, targeted subset of student model parameters. This work establishes the feasibility of transferring mechanisms, which may in turn allow for efficient distillation of targeted teacher capabilities via interpretable and controllable internal student mechanisms. 

---
# Generalized Correctness Models: Learning Calibrated and Model-Agnostic Correctness Predictors from Historical Patterns 

**Authors**: Hanqi Xiao, Vaidehi Patil, Hyunji Lee, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2509.24988)  

**Abstract**: Generating accurate and calibrated confidence estimates is critical for deploying LLMs in high-stakes or user-facing applications, and remains an open challenge. Prior research has often framed confidence as a problem of eliciting a model's "self-knowledge", i.e., the ability of an LLM to judge whether its own answers are correct; this approach implicitly assumes that there is some privileged information about the answer's correctness that is accessible to the model itself. However, our experiments reveal that an LLM attempting to predict the correctness of its own outputs generally performs no better than an unrelated LLM. Moreover, we hypothesize that a key factor in building a "Correctness Model" (CM) is exposure to a target model's historical predictions. We propose multiple methods to inject this historical correctness information, creating a Generalized Correctness Model (GCM). We first show that GCMs can be trained on the correctness data from many LLMs and learn patterns for correctness prediction applicable across datasets and models. We then use CMs as a lens for studying the source of correctness prediction ability and its generalization, systematically controlling their training data and finding that answer phrasing is a strong predictor for correctness. We further explore alternative methods of injecting history without training an LLM, finding that including history as in-context examples can help improve correctness prediction, and post-hoc calibration can provide complementary reductions in calibration error. We evaluate GCMs based on Qwen3-8B across 5 model families and the MMLU and TriviaQA datasets, as well as on a downstream selective prediction task, finding that reliable LLM confidence estimation is a generalizable and model-agnostic skill learned by systematically encoding correctness history rather than a model-specific skill reliant on self-introspection. 

---
# SemanticShield: LLM-Powered Audits Expose Shilling Attacks in Recommender Systems 

**Authors**: Kaihong Li, Huichi Zhou, Bin Ma, Fangjun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24961)  

**Abstract**: Recommender systems (RS) are widely used in e-commerce for personalized suggestions, yet their openness makes them susceptible to shilling attacks, where adversaries inject fake behaviors to manipulate recommendations. Most existing defenses emphasize user-side behaviors while overlooking item-side features such as titles and descriptions that can expose malicious intent. To address this gap, we propose a two-stage detection framework that integrates item-side semantics via large language models (LLMs). The first stage pre-screens suspicious users using low-cost behavioral criteria, and the second stage employs LLM-based auditing to evaluate semantic consistency. Furthermore, we enhance the auditing model through reinforcement fine-tuning on a lightweight LLM with carefully designed reward functions, yielding a specialized detector called SemanticShield. Experiments on six representative attack strategies demonstrate the effectiveness of SemanticShield against shilling attacks, and further evaluation on previously unseen attack methods shows its strong generalization capability. Code is available at this https URL. 

---
# The Dialogue That Heals: A Comprehensive Evaluation of Doctor Agents' Inquiry Capability 

**Authors**: Linlu Gong, Ante Wang, Yunghwei Lai, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24958)  

**Abstract**: An effective physician should possess a combination of empathy, expertise, patience, and clear communication when treating a patient. Recent advances have successfully endowed AI doctors with expert diagnostic skills, particularly the ability to actively seek information through inquiry. However, other essential qualities of a good doctor remain overlooked. To bridge this gap, we present MAQuE(Medical Agent Questioning Evaluation), the largest-ever benchmark for the automatic and comprehensive evaluation of medical multi-turn questioning. It features 3,000 realistically simulated patient agents that exhibit diverse linguistic patterns, cognitive limitations, emotional responses, and tendencies for passive disclosure. We also introduce a multi-faceted evaluation framework, covering task success, inquiry proficiency, dialogue competence, inquiry efficiency, and patient experience. Experiments on different LLMs reveal substantial challenges across the evaluation aspects. Even state-of-the-art models show significant room for improvement in their inquiry capabilities. These models are highly sensitive to variations in realistic patient behavior, which considerably impacts diagnostic accuracy. Furthermore, our fine-grained metrics expose trade-offs between different evaluation perspectives, highlighting the challenge of balancing performance and practicality in real-world clinical settings. 

---
# MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes 

**Authors**: Changsheng Zhao, Ernie Chang, Zechun Liu, Chia-Jung Chang, Wei Wen, Chen Lai, Rick Cao, Yuandong Tian, Raghuraman Krishnamoorthi, Yangyang Shi, Vikas Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2509.24945)  

**Abstract**: The paradigm shift in large language models (LLMs) from instinctive responses to chain-of-thought (CoT) reasoning has fueled two prevailing assumptions: (1) reasoning capabilities only emerge in sufficiently large models, and (2) such capabilities require training on massive datasets. While the first assumption has already been challenged by recent sub-billion-parameter reasoning models such as Qwen3-0.6B and DeepSeek distilled variants, the second remains largely unquestioned. In this work, we revisit the necessity of scaling to extremely large corpora (>10T tokens) for reasoning emergence. By carefully curating and resampling open-source datasets that we identify as beneficial under our designed metrics, we demonstrate that strong reasoning abilities can emerge with far less data. Specifically, we show that only ~2T tokens of high-quality data are sufficient, and pre-training with 4.2T tokens on the dataset resampled from these ~2T tokens, followed by a established post-training procedure, enables the development of MobileLLM-R1, a series of sub-billion-parameter reasoning models that substantially outperform prior models trained on fully open-sourced data. For example, MobileLLM-R1-950M achieves an AIME score of 15.5, compared to just 0.6 for OLMo-2-1.48B and 0.3 for SmolLM-2-1.7B. Remarkably, despite being trained on only 11.7% of the tokens compared to Qwen3's proprietary 36T-token corpus for pretraining, MobileLLM-R1-950M matches or surpasses Qwen3-0.6B across multiple reasoning benchmarks. To facilitate further research in this direction, we have released the complete training recipe, data sources, data mixing ratio, and model checkpoints, together with the key insights obtained throughout this study. 

---
# How Well Do LLMs Imitate Human Writing Style? 

**Authors**: Rebira Jemama, Rajesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.24930)  

**Abstract**: Large language models (LLMs) can generate fluent text, but their ability to replicate the distinctive style of a specific human author remains unclear. We present a fast, training-free framework for authorship verification and style imitation analysis. The method integrates TF-IDF character n-grams with transformer embeddings and classifies text pairs through empirical distance distributions, eliminating the need for supervised training or threshold tuning. It achieves 97.5\% accuracy on academic essays and 94.5\% in cross-domain evaluation, while reducing training time by 91.8\% and memory usage by 59\% relative to parameter-based baselines. Using this framework, we evaluate five LLMs from three separate families (Llama, Qwen, Mixtral) across four prompting strategies - zero-shot, one-shot, few-shot, and text completion. Results show that the prompting strategy has a more substantial influence on style fidelity than model size: few-shot prompting yields up to 23.5x higher style-matching accuracy than zero-shot, and completion prompting reaches 99.9\% agreement with the original author's style. Crucially, high-fidelity imitation does not imply human-like unpredictability - human essays average a perplexity of 29.5, whereas matched LLM outputs average only 15.2. These findings demonstrate that stylistic fidelity and statistical detectability are separable, establishing a reproducible basis for future work in authorship modeling, detection, and identity-conditioned generation. 

---
# BOE-XSUM: Extreme Summarization in Clear Language of Spanish Legal Decrees and Notifications 

**Authors**: Andrés Fernández García, Javier de la Rosa, Julio Gonzalo, Roser Morante, Enrique Amigó, Alejandro Benito-Santos, Jorge Carrillo-de-Albornoz, Víctor Fresno, Adrian Ghajari, Guillermo Marco, Laura Plaza, Eva Sánchez Salido  

**Link**: [PDF](https://arxiv.org/pdf/2509.24908)  

**Abstract**: The ability to summarize long documents succinctly is increasingly important in daily life due to information overload, yet there is a notable lack of such summaries for Spanish documents in general, and in the legal domain in particular. In this work, we present BOE-XSUM, a curated dataset comprising 3,648 concise, plain-language summaries of documents sourced from Spain's ``Bolet\'ın Oficial del Estado'' (BOE), the State Official Gazette. Each entry in the dataset includes a short summary, the original text, and its document type label. We evaluate the performance of medium-sized large language models (LLMs) fine-tuned on BOE-XSUM, comparing them to general-purpose generative models in a zero-shot setting. Results show that fine-tuned models significantly outperform their non-specialized counterparts. Notably, the best-performing model -- BERTIN GPT-J 6B (32-bit precision) -- achieves a 24\% performance gain over the top zero-shot model, DeepSeek-R1 (accuracies of 41.6\% vs.\ 33.5\%). 

---
# Expanding Computation Spaces of LLMs at Inference Time 

**Authors**: Yoonna Jang, Kisu Yang, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2509.24884)  

**Abstract**: Chain-of-thought (CoT) rationale enables language models to use additional task-related text for problem-solving, benefiting not only from detailed reasoning steps but also from the expanded computational space of longer inputs. Prior work has trained filler or special tokens to serve as additional computation spaces. In this study, we investigate whether language models can leverage artificially inserted sequences of filler tokens solely at inference. We first identify effective token types, numbers, and insertion locations, then examine at what stage of training models begin to exploit the expanded computation space, and finally analyze dynamics within these spaces via attention maps. Experiments on models ranging from 1.7B to 32B across open-domain QA and math tasks show that appropriate token types and counts vary, but placing filler tokens directly before the final 'Answer:' token is most effective. Smaller models benefit most, up to 12.372 percentage points in SmolLM2-1.7B-Instruct, indicating that these spaces act as additional computational capacity rather than redundant input. Attention maps reveal that expanded spaces often continue the original attention mechanism and sometimes focus on questions or answer options, suggesting meaningful computation for problem-solving. 

---
# Metaphor identification using large language models: A comparison of RAG, prompt engineering, and fine-tuning 

**Authors**: Matteo Fuoli, Weihang Huang, Jeannette Littlemore, Sarah Turner, Ellen Wilding  

**Link**: [PDF](https://arxiv.org/pdf/2509.24866)  

**Abstract**: Metaphor is a pervasive feature of discourse and a powerful lens for examining cognition, emotion, and ideology. Large-scale analysis, however, has been constrained by the need for manual annotation due to the context-sensitive nature of metaphor. This study investigates the potential of large language models (LLMs) to automate metaphor identification in full texts. We compare three methods: (i) retrieval-augmented generation (RAG), where the model is provided with a codebook and instructed to annotate texts based on its rules and examples; (ii) prompt engineering, where we design task-specific verbal instructions; and (iii) fine-tuning, where the model is trained on hand-coded texts to optimize performance. Within prompt engineering, we test zero-shot, few-shot, and chain-of-thought strategies. Our results show that state-of-the-art closed-source LLMs can achieve high accuracy, with fine-tuning yielding a median F1 score of 0.79. A comparison of human and LLM outputs reveals that most discrepancies are systematic, reflecting well-known grey areas and conceptual challenges in metaphor theory. We propose that LLMs can be used to at least partly automate metaphor identification and can serve as a testbed for developing and refining metaphor identification protocols and the theory that underpins them. 

---
# Between Help and Harm: An Evaluation of Mental Health Crisis Handling by LLMs 

**Authors**: Adrian Arnaiz-Rodriguez, Miguel Baidal, Erik Derner, Jenn Layton Annable, Mark Ball, Mark Ince, Elvira Perez Vallejos, Nuria Oliver  

**Link**: [PDF](https://arxiv.org/pdf/2509.24857)  

**Abstract**: The widespread use of chatbots powered by large language models (LLMs) such as ChatGPT and Llama has fundamentally reshaped how people seek information and advice across domains. Increasingly, these chatbots are being used in high-stakes contexts, including emotional support and mental health concerns. While LLMs can offer scalable support, their ability to safely detect and respond to acute mental health crises remains poorly understood. Progress is hampered by the absence of unified crisis taxonomies, robust annotated benchmarks, and empirical evaluations grounded in clinical best practices. In this work, we address these gaps by introducing a unified taxonomy of six clinically-informed mental health crisis categories, curating a diverse evaluation dataset, and establishing an expert-designed protocol for assessing response appropriateness. We systematically benchmark three state-of-the-art LLMs for their ability to classify crisis types and generate safe, appropriate responses. The results reveal that while LLMs are highly consistent and generally reliable in addressing explicit crisis disclosures, significant risks remain. A non-negligible proportion of responses are rated as inappropriate or harmful, with responses generated by an open-weight model exhibiting higher failure rates than those generated by the commercial ones. We also identify systemic weaknesses in handling indirect or ambiguous risk signals, a reliance on formulaic and inauthentic default replies, and frequent misalignment with user context. These findings underscore the urgent need for enhanced safeguards, improved crisis detection, and context-aware interventions in LLM deployments. Our taxonomy, datasets, and evaluation framework lay the groundwork for ongoing research and responsible innovation in AI-driven mental health support, helping to minimize harm and better protect vulnerable users. 

---
# Hierarchical Error Correction for Large Language Models: A Systematic Framework for Domain-Specific AI Quality Enhancement 

**Authors**: Zhilong Zhao, Yindi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24841)  

**Abstract**: Large Language Models face significant performance challenges in specialized domains, with state-of-the-art models achieving only 45.9% accuracy on medical coding tasks. This study proposes a Hierarchical Error Correction (HEC) framework that addresses domain-specific AI limitations through systematic error analysis and targeted intervention strategies.
We analyze error patterns across four specialized domains and find that AI errors follow consistent hierarchical structures: Knowledge-layer errors (58.4%), Reasoning-layer errors (39.6%), and Complexity-layer errors (2.0%). Based on these patterns, we develop a three-stage correction framework that addresses errors according to their hierarchical importance and demonstrates that framework effectiveness correlates inversely with baseline task performance.
Experimental validation across medical transcription (4,921 cases), legal document classification (1,000 cases), political bias detection (645 cases), and legal reasoning (1,000 cases) shows consistent improvements. Cross-model validation across five LLM architectures demonstrates average improvements of 11.2 percentage points (p < 0.001). However, analysis reveals framework limitations in high-baseline tasks (>75% accuracy), where hierarchical intervention may interfere with effective reasoning processes.
The results suggest that systematic error analysis can guide effective AI enhancement strategies in specialized domains, particularly for moderate-baseline tasks, while highlighting the importance of understanding framework boundaries for optimal deployment. 

---
# SemShareKV: Efficient KVCache Sharing for Semantically Similar Prompts via Token-Level LSH Matching 

**Authors**: Xinye Zhao, Spyridon Mastorakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.24832)  

**Abstract**: As large language models (LLMs) continue to scale, the memory footprint of key-value (KV) caches during inference has become a significant bottleneck. Existing approaches primarily focus on compressing KV caches within a single prompt or reusing shared prefixes or frequently ocurred text segments across prompts. However, such strategies are limited in scenarios where prompts are semantically similar but lexically different, which frequently occurs in tasks such as multi-document summarization and conversational agents. We propose \textit{SemShareKV}, a KV cache sharing and compression framework that accelerates LLM inference by reusing KVCache in semantically similar prompts. Instead of relying on exact token matches, SemShareKV applies fuzzy token matching using locality-sensitive hashing (LSH) on token embeddings and incorporates Rotary Position Embedding (RoPE) to better preserve positional information. By selectively reusing relevant key-value pairs from a reference prompt's cache, SemShareKV reduces redundant computation while maintaining output quality. Experiments on diverse summarization datasets show up to 6.25$\times$ speedup and 42\% lower GPU memory usage with 5k tokens input, with negligible quality degradation. These results highlight the potential of semantic-aware cache sharing for efficient LLM inference. 

---
# DiaCDM: Cognitive Diagnosis in Teacher-Student Dialogues using the Initiation-Response-Evaluation Framework 

**Authors**: Rui Jia, Yuang Wei, Ruijia Li, Yuang-Hao Jiang, Xinyu Xie, Yaomin Shen, Min Zhang, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24821)  

**Abstract**: While cognitive diagnosis (CD) effectively assesses students' knowledge mastery from structured test data, applying it to real-world teacher-student dialogues presents two fundamental challenges. Traditional CD models lack a suitable framework for handling dynamic, unstructured dialogues, and it's difficult to accurately extract diagnostic semantics from lengthy dialogues. To overcome these hurdles, we propose DiaCDM, an innovative model. We've adapted the initiation-response-evaluation (IRE) framework from educational theory to design a diagnostic framework tailored for dialogue. We also developed a unique graph-based encoding method that integrates teacher questions with relevant knowledge components to capture key information more precisely. To our knowledge, this is the first exploration of cognitive diagnosis in a dialogue setting. Experiments on three real-world dialogue datasets confirm that DiaCDM not only significantly improves diagnostic accuracy but also enhances the results' interpretability, providing teachers with a powerful tool for assessing students' cognitive states. The code is available at this https URL. 

---
# KnowGuard: Knowledge-Driven Abstention for Multi-Round Clinical Reasoning 

**Authors**: Xilin Dang, Kexin Chen, Xiaorui Su, Ayush Noori, Iñaki Arango, Lucas Vittor, Xinyi Long, Yuyang Du, Marinka Zitnik, Pheng Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2509.24816)  

**Abstract**: In clinical practice, physicians refrain from making decisions when patient information is insufficient. This behavior, known as abstention, is a critical safety mechanism preventing potentially harmful misdiagnoses. Recent investigations have reported the application of large language models (LLMs) in medical scenarios. However, existing LLMs struggle with the abstentions, frequently providing overconfident responses despite incomplete information. This limitation stems from conventional abstention methods relying solely on model self-assessments, which lack systematic strategies to identify knowledge boundaries with external medical evidences. To address this, we propose \textbf{KnowGuard}, a novel \textit{investigate-before-abstain} paradigm that integrates systematic knowledge graph exploration for clinical decision-making. Our approach consists of two key stages operating on a shared contextualized evidence pool: 1) an evidence discovery stage that systematically explores the medical knowledge space through graph expansion and direct retrieval, and 2) an evidence evaluation stage that ranks evidence using multiple factors to adapt exploration based on patient context and conversation history. This two-stage approach enables systematic knowledge graph exploration, allowing models to trace structured reasoning paths and recognize insufficient medical evidence. We evaluate our abstention approach using open-ended multi-round clinical benchmarks that mimic realistic diagnostic scenarios, assessing abstention quality through accuracy-efficiency trade-offs beyond existing closed-form evaluations. Experimental evidences clearly demonstrate that KnowGuard outperforms state-of-the-art abstention approaches, improving diagnostic accuracy by 3.93\% while reducing unnecessary interaction by 7.27 turns on average. 

---
# Evaluating Spatiotemporal Consistency in Automatically Generated Sewing Instructions 

**Authors**: Luisa Geiger, Mareike Hartmann, Michael Sullivan, Alexander Koller  

**Link**: [PDF](https://arxiv.org/pdf/2509.24792)  

**Abstract**: In this paper, we propose a novel, automatic tree-based evaluation metric for LLM-generated step-by-step assembly instructions, that more accurately reflects spatiotemporal aspects of construction than traditional metrics such as BLEU and BERT similarity scores. We apply our proposed metric to the domain of sewing instructions, and show that our metric better correlates with manually-annotated error counts as well as human quality ratings, demonstrating our metric's superiority for evaluating the spatiotemporal soundness of sewing instructions. Further experiments show that our metric is more robust than traditional approaches against artificially-constructed counterfactual examples that are specifically constructed to confound metrics that rely on textual similarity. 

---
# SeaPO: Strategic Error Amplification for Robust Preference Optimization of Large Language Models 

**Authors**: Jun Rao, Yunjie Liao, Xuebo Liu, Zepeng Lin, Lian Lian, Dong Jin, Shengjun Cheng, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24781)  

**Abstract**: Existing alignment methods for preference optimization of large language models (LLMs) aim to enhance model performance by utilizing pairs of positive and negative samples. However, due to the limited capacity of models in scoring or generating responses, the quality of positive and negative samples may become similar during training, which complicates optimization for preference learning. To address this issue, we introduce SeaPO, a Strategic Error Amplification method that leverages three error types commonly occurring in LLMs to introduce specific error patterns into the model Preference Optimization. This strategy ensures that negative samples are more erroneous than positive samples and preference-based training is employed to mitigate the occurrence of these errors, thereby enhancing model performance. Evaluations across five capability dimensions and different model scales (1.5B to 14B) demonstrate that the generated data significantly improved overall model performance, particularly in terms of truthfulness, with improvements of 5-10 percentage points observed. Further analysis reveals that task performance varies depending on the error types introduced. Injecting the most common error types improves performance in related tasks, while a mix of error types leads to a broader performance enhancement: most tasks show stable improvements, while a few tasks exhibit significant gains. 

---
# LatentEvolve: Self-Evolving Test-Time Scaling in Latent Space 

**Authors**: Guibin Zhang, Fanci Meng, Guancheng Wan, Zherui Li, Kun Wang, Zhenfei Yin, Lei Bai, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24771)  

**Abstract**: Test-time Scaling (TTS) has been demonstrated to significantly enhance the reasoning capabilities of Large Language Models (LLMs) during the inference phase without altering model parameters. However, existing TTS methods are largely independent, implying that LLMs have not yet evolved to progressively learn how to scale more effectively. With the objective of evolving LLMs to learn ``how to scale test-time computation,'' we propose LatentEvolve, a self-evolving latent TTS framework inspired by the complementary learning system (CLS) theory. Analogous to the human brain's dual system of a fast-recall hippocampus and a slow-consolidating neocortex, LatentEvolve comprises two evolutionary components: \textit{daytime scaling}, which rapidly retrieves historical latent representations to better guide current LLM reasoning; and \textit{nighttime scaling}, which integrates past latent optimizations in a manner akin to the human brain's consolidation of experiences during sleep. The alternation of daytime and nighttime processes facilitates a fast and slow evolution of LLM TTS, mirroring human cognitive dynamics in a fully unsupervised manner. Extensive experiments across eight benchmarks and five model backbones demonstrate that our LatentEvolve surpasses state-of-the-art TTS methods such as LatentSeek and TTRL by up to $13.33\%$ and exhibits exceptional cross-domain and cross-backbone generalization. 

---
# ProxyAttn: Guided Sparse Attention via Representative Heads 

**Authors**: Yixuan Wang, Huang He, Siqi Bao, Hua Wu, Haifeng Wang, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2509.24745)  

**Abstract**: The quadratic complexity of attention mechanisms limits the efficiency of Large Language Models (LLMs) on long-text tasks. Recently, methods that dynamically estimate block importance have enabled efficient block sparse attention, leading to significant acceleration in long-text pre-filling of LLMs. However, their coarse-grained estimation inevitably leads to performance degradation at high sparsity rates. In this work, we propose ProxyAttn, a training-free sparse attention algorithm that achieves more precise block estimation by compressing the dimension of attention heads. Based on our observation of the similarity among multiple attention heads, we use the scores of pooled representative heads to approximate the scores for all heads. To account for the varying sparsity among heads, we also propose a block-aware dynamic budget estimation method. By combining the scores from representative proxy heads with multi-head dynamic budgets, we achieve a more fine-grained block importance evaluation at low computational cost. Experiments on a variety of mainstream models and extensive benchmarks confirm the underlying similarity among attention heads. Leveraging a fine-grained estimation, the proposed method achieves substantial gains in performance and efficiency compared to existing methods. More precisely, ProxyAttn can achieve up to 10.3x attention acceleration and 2.4x prefilling acceleration without significant performance loss. Our code is available at this https URL. 

---
# Socratic-Zero : Bootstrapping Reasoning via Data-Free Agent Co-evolution 

**Authors**: Shaobo Wang, Zhengbo Jiao, Zifan Zhang, Yilang Peng, Xu Ze, Boyu Yang, Wei Wang, Hu Wei, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24726)  

**Abstract**: Recent breakthroughs in large language models (LLMs) on reasoning tasks rely heavily on massive, high-quality datasets-typically human-annotated and thus difficult to scale. While data synthesis or distillation offers a promising alternative, existing methods struggle with inconsistent data quality and an inability to dynamically adapt to the evolving capabilities of the model, leading to suboptimal training signals. To address these limitations, we introduce Socratic-Zero, a fully autonomous framework that generates high-quality training data from minimal seed examples through the co-evolution of three agents: the Teacher, the Solver, and the Generator. The Solver continuously refines its reasoning by learning from preference feedback on both successful and failed trajectories; the Teacher adaptively crafts increasingly challenging questions based on the Solver's weaknesses; and the Generator distills the Teacher's question-design strategy to enable scalable, high-fidelity curriculum generation. This closed-loop system produces a self-improving curriculum-requiring no pre-existing tasks or labels. Remarkably, starting from only 100 seed questions, our Socratic-Solver-8B achieves an average gain of +20.2 percentage points over prior data synthesis methods across seven mathematical reasoning benchmarks (AMC23, AIME24-25, Olympiad, MATH-500, Minerva, and GSM8K), with consistent gains on both Qwen3 and GLM4 series models. Even more surprisingly, synthetic data from Socratic-Generator-32B enables student LLMs to achieve superior performance compared to other state-of-the-art (SOTA) commercial LLMs on these benchmarks, including Qwen3-235B-A22B, DeepSeek-V3.1-671B, GPT-5, Gemini-2.5-Pro, Grok-4, and Claude-4.1-Opus. 

---
# MemGen: Weaving Generative Latent Memory for Self-Evolving Agents 

**Authors**: Guibin Zhang, Muxin Fu, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24704)  

**Abstract**: Agent memory shapes how Large Language Model (LLM)-powered agents, akin to the human brain, progressively refine themselves through environment interactions. Existing paradigms remain constrained: parametric memory forcibly adjusts model parameters, and retrieval-based memory externalizes experience into structured databases, yet neither captures the fluid interweaving of reasoning and memory that underlies human cognition. To address this gap, we propose MemGen, a dynamic generative memory framework that equips agents with a human-esque cognitive faculty. It consists of a \textit{memory trigger}, which monitors the agent's reasoning state to decide explicit memory invocation, and a \textit{memory weaver}, which takes the agent's current state as stimulus to construct a latent token sequence as machine-native memory to enrich its reasoning. In this way, MemGen enables agents to recall and augment latent memory throughout reasoning, producing a tightly interwoven cycle of memory and cognition. Extensive experiments across eight benchmarks show that MemGen surpasses leading external memory systems such as ExpeL and AWM by up to $38.22\%$, exceeds GRPO by up to $13.44\%$, and exhibits strong cross-domain generalization ability. More importantly, we find that without explicit supervision, MemGen spontaneously evolves distinct human-like memory faculties, including planning memory, procedural memory, and working memory, suggesting an emergent trajectory toward more naturalistic forms of machine cognition. 

---
# Reference-Free Rating of LLM Responses via Latent Information 

**Authors**: Leander Girrbach, Chi-Ping Su, Tankred Saanum, Richard Socher, Eric Schulz, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2509.24678)  

**Abstract**: How reliable are single-response LLM-as-a-judge ratings without references, and can we obtain fine-grained, deterministic scores in this setting? We study the common practice of asking a judge model to assign Likert-scale scores to free-text responses and show two systematic issues: scores are unstable under sampling and poorly calibrated, leading to compression near the top of the scale and frequent ties. We then propose and evaluate Latent Judges, which derive scalar ratings from internal model signals: (i) probability-weighted scores over integer ratings, (ii) verifier-style probabilities of "yes", and (iii) linear probes trained on model activations at the rating position. Across a broad suite of pairwise and single-rating benchmarks, latent methods match or surpass standard prompting, with consistent gains on pairwise accuracy and listwise ranking relevant to Best-of-N selection. Probability-weighted scores achieve the strongest single-rating correlations, while probes recover useful signals when output logits are miscalibrated. These results indicate that latent information provides deterministic and more discriminative signals for reference-free evaluation, and can improve selection and training approaches like Best-of-$N$, multi-teacher distillation, and routing. 

---
# Understanding the Dilemma of Unlearning for Large Language Models 

**Authors**: Qingjie Zhang, Haoting Qian, Zhicong Huang, Cheng Hong, Minlie Huang, Ke Xu, Chao Zhang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24675)  

**Abstract**: Unlearning seeks to remove specific knowledge from large language models (LLMs), but its effectiveness remains contested. On one side, "forgotten" knowledge can often be recovered through interventions such as light fine-tuning; on the other side, unlearning may induce catastrophic forgetting that degrades general capabilities. Despite active exploration of unlearning methods, interpretability analyses of the mechanism are scarce due to the difficulty of tracing knowledge in LLMs' complex architectures. We address this gap by proposing unPact, an interpretable framework for unlearning via prompt attribution and contribution tracking. Typically, it quantifies each prompt token's influence on outputs, enabling pre- and post-unlearning comparisons to reveal what changes. Across six mainstream unlearning methods, three LLMs, and three benchmarks, we find that: (1) Unlearning appears to be effective by disrupting focus on keywords in prompt; (2) Much of the knowledge is not truly erased and can be recovered by simply emphasizing these keywords in prompts, without modifying the model's weights; (3) Catastrophic forgetting arises from indiscriminate penalization of all tokens. Taken together, our results suggest an unlearning dilemma: existing methods tend either to be insufficient - knowledge remains recoverable by keyword emphasis, or overly destructive - general performance collapses due to catastrophic forgetting, still leaving a gap to reliable unlearning. 

---
# InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation 

**Authors**: Weilin Zhao, Zihan Zhou, Zhou Su, Chaojun Xiao, Yuxuan Li, Yanghao Li, Yudi Zhang, Weilun Zhao, Zhen Li, Yuxiang Huang, Ao Sun, Xu Han, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24663)  

**Abstract**: Long-sequence processing is a critical capability for modern large language models. However, the self-attention mechanism in the standard Transformer architecture faces severe computational and memory bottlenecks when processing long sequences. While trainable sparse attention methods offer a promising solution, existing approaches such as NSA introduce excessive extra parameters and disrupt the conventional \textit{pretrain-on-short, finetune-on-long} workflow, resulting in slow convergence and difficulty in acceleration. To overcome these limitations, we introduce dense-sparse switchable attention framework, termed as InfLLM-V2. InfLLM-V2 is a trainable sparse attention that seamlessly adapts models from short to long sequences. Specifically, InfLLM-V2 reuses dense attention parameters through parameter-free architecture modification, maintaining consistency between short and long sequence processing. Additionally, InfLLM-V2 ensures computational efficiency across all sequence lengths, by using dense attention for short inputs and smoothly transitioning to sparse attention for long sequences. To achieve practical acceleration, we further introduce an efficient implementation of InfLLM-V2 that significantly reduces the computational overhead. Our experiments on long-context understanding and chain-of-thought reasoning demonstrate that InfLLM-V2 is 4$\times$ faster than dense attention while retaining 98.1% and 99.7% of the performance, respectively. Based on the InfLLM-V2 framework, we have trained and open-sourced MiniCPM4.1 (this https URL), a hybrid reasoning model, providing a reproducible implementation for the research community. 

---
# Hype or not? Formalizing Automatic Promotional Language Detection in Biomedical Research 

**Authors**: Bojan Batalo, Erica K. Shimomoto, Neil Millar  

**Link**: [PDF](https://arxiv.org/pdf/2509.24638)  

**Abstract**: In science, promotional language ('hype') is increasing and can undermine objective evaluation of evidence, impede research development, and erode trust in science. In this paper, we introduce the task of automatic detection of hype, which we define as hyperbolic or subjective language that authors use to glamorize, promote, embellish, or exaggerate aspects of their research. We propose formalized guidelines for identifying hype language and apply them to annotate a portion of the National Institutes of Health (NIH) grant application corpus. We then evaluate traditional text classifiers and language models on this task, comparing their performance with a human baseline. Our experiments show that formalizing annotation guidelines can help humans reliably annotate candidate hype adjectives and that using our annotated dataset to train machine learning models yields promising results. Our findings highlight the linguistic complexity of the task, and the potential need for domain knowledge and temporal awareness of the facts. While some linguistic works address hype detection, to the best of our knowledge, we are the first to approach it as a natural language processing task. 

---
# HiKE: Hierarchical Evaluation Framework for Korean-English Code-Switching Speech Recognition 

**Authors**: Gio Paik, Yongbeom Kim, Soungmin Lee, Sangmin Ahn, Chanwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.24613)  

**Abstract**: Despite advances in multilingual automatic speech recognition (ASR), code-switching (CS), the mixing of languages within an utterance common in daily speech, remains a severely underexplored challenge. In this paper, we introduce HiKE: the Hierarchical Korean-English code-switching benchmark, the first globally accessible evaluation framework for Korean-English CS, aiming to provide a means for the precise evaluation of multilingual ASR models and to foster research in the field. The proposed framework not only consists of high-quality, natural CS data across various topics, but also provides meticulous loanword labels and a hierarchical CS-level labeling scheme (word, phrase, and sentence) that together enable a systematic evaluation of a model's ability to handle each distinct level of code-switching. Through evaluations of diverse multilingual ASR models and fine-tuning experiments, this paper demonstrates that while most multilingual ASR models initially struggle with CS-ASR, this capability can be enabled through fine-tuning with CS data. HiKE will be available at this https URL. 

---
# Inducing Dyslexia in Vision Language Models 

**Authors**: Melika Honarmand, Ayati Sharma, Badr AlKhamissi, Johannes Mehrer, Martin Schrimpf  

**Link**: [PDF](https://arxiv.org/pdf/2509.24597)  

**Abstract**: Dyslexia, a neurodevelopmental disorder characterized by persistent reading difficulties, is often linked to reduced activity of the visual word form area in the ventral occipito-temporal cortex. Traditional approaches to studying dyslexia, such as behavioral and neuroimaging methods, have provided valuable insights but remain limited in their ability to test causal hypotheses about the underlying mechanisms of reading impairments. In this study, we use large-scale vision-language models (VLMs) to simulate dyslexia by functionally identifying and perturbing artificial analogues of word processing. Using stimuli from cognitive neuroscience, we identify visual-word-form-selective units within VLMs and demonstrate that targeted ablation of these units, unlike ablation of random units, leads to selective impairments in reading tasks while general visual and language comprehension abilities remain intact. In particular, the resulting model matches dyslexic humans' phonological deficits without a significant change in orthographic processing. Taken together, our modeling results replicate key characteristics of dyslexia and establish a computational framework for investigating reading disorders. 

---
# AdaThink-Med: Medical Adaptive Thinking with Uncertainty-Guided Length Calibration 

**Authors**: Shaohao Rui, Kaitao Chen, Weijie Ma, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24560)  

**Abstract**: Recent advances in inference time scaling with extended long chain-of thought have significantly improved the reasoning capabilities of both general and medical large language models (LLMs). However, these models tend to engage in lengthy reasoning processes regardless of the difficulty of the input question, leading to increased inference costs in real-world applications. Therefore, enabling adaptive thinking where models think less for simpler questions and think more for complex ones is critical for the effective use of medical LLMs in practice. Despite its importance, there is a lack of end-to-end approaches designed to enhance the adaptive thinking capabilities of medical LLMs while providing a comprehensive examination of the trade-off between performance and computational cost. To bridge this gap, we propose AdaThink-Med, the first end-to-end framework designed to enhance adaptive thinking ability in medical reasoning models with uncertainty-guided length calibration. AdaThink-Med first generates multiple candidate outputs for each question, evaluates the correctness and uncertainty of each candidate, and then estimates problem difficulty via an uncertainty-guided length calibration module. For outputs with low difficulty and correct answers, the framework penalizes longer reasoning paths; whereas for those with high difficulty and incorrect answers, it encourages extending the chain of thought to explore alternative solutions. On six public medical QA benchmarks, AdaThink-Med achieves up to 6.4x length reduction on average while retaining performance with only minimal degradation. Intriguingly, we observe that AdaThink-Med spontaneously develops two distinct reasoning modes, which we characterize as "non-thinking" and "thinking", demonstrating the model's ability to suppress redundant reasoning processes dynamically. 

---
# Building Benchmarks from the Ground Up: Community-Centered Evaluation of LLMs in Healthcare Chatbot Settings 

**Authors**: Hamna, Gayatri Bhat, Sourabrata Mukherjee, Faisal Lalani, Evan Hadfield, Divya Siddarth, Kalika Bali, Sunayana Sitaram  

**Link**: [PDF](https://arxiv.org/pdf/2509.24506)  

**Abstract**: Large Language Models (LLMs) are typically evaluated through general or domain-specific benchmarks testing capabilities that often lack grounding in the lived realities of end users. Critical domains such as healthcare require evaluations that extend beyond artificial or simulated tasks to reflect the everyday needs, cultural practices, and nuanced contexts of communities. We propose Samiksha, a community-driven evaluation pipeline co-created with civil-society organizations (CSOs) and community members. Our approach enables scalable, automated benchmarking through a culturally aware, community-driven pipeline in which community feedback informs what to evaluate, how the benchmark is built, and how outputs are scored. We demonstrate this approach in the health domain in India. Our analysis highlights how current multilingual LLMs address nuanced community health queries, while also offering a scalable pathway for contextually grounded and inclusive LLM evaluation. 

---
# Knowledge Editing with Subspace-Aware Key-Value Mappings 

**Authors**: Haewon Park, Sangwoo Kim, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.24502)  

**Abstract**: Knowledge editing aims to efficiently correct factual errors in Language Models (LMs). The popular locate-then-edit approach modifies an MLP layer by finding an optimal mapping between its input vector (key) and output vector (value) that leads to the expression of the edited knowledge. However, existing methods without any constraints on the key and value vectors cause significant perturbations to the edited model. To address this, we propose Subspace Knowledge Edit (SUIT), a method that identifies and modifies only the subspace of critical features relevant to the edit. Our empirical results on LLaMA-3-8B, GPT-J-6B, and Qwen2.5-7B models show that SUIT dramatically improves knowledge preservation over strong baselines while maintaining high edit efficacy. This effectiveness confirms that SUIT successfully identifies the critical subspace for the edit. Further analyses provide additional validation for our approach. The source code and data will be released to the public upon publication of the paper. 

---
# GRPO-MA: Multi-Answer Generation in GRPO for Stable and Efficient Chain-of-Thought Training 

**Authors**: Hongcheng Wang, Yinuo Huang, Sukai Wang, Guanghui Ren, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.24494)  

**Abstract**: Recent progress, such as DeepSeek-R1, has shown that the GRPO algorithm, a Reinforcement Learning (RL) approach, can effectively train Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) and Vision-Language Models (VLMs). In this paper, we analyze three challenges of GRPO: gradient coupling between thoughts and answers, sparse reward signals caused by limited parallel sampling, and unstable advantage estimation. To mitigate these challenges, we propose GRPO-MA, a simple yet theoretically grounded method that leverages multi-answer generation from each thought process, enabling more robust and efficient optimization. Theoretically, we show that the variance of thought advantage decreases as the number of answers per thought increases. Empirically, our gradient analysis confirms this effect, showing that GRPO-MA reduces gradient spikes compared to GRPO. Experiments on math, code, and diverse multimodal tasks demonstrate that GRPO-MA substantially improves performance and training efficiency. Our ablation studies further reveal that increasing the number of answers per thought consistently enhances model performance. 

---
# Sanitize Your Responses: Mitigating Privacy Leakage in Large Language Models 

**Authors**: Wenjie Fu, Huandong Wang, Junyao Gao, Guoan Wan, Tao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24488)  

**Abstract**: As Large Language Models (LLMs) achieve remarkable success across a wide range of applications, such as chatbots and code copilots, concerns surrounding the generation of harmful content have come increasingly into focus. Despite significant advances in aligning LLMs with safety and ethical standards, adversarial prompts can still be crafted to elicit undesirable responses. Existing mitigation strategies are predominantly based on post-hoc filtering, which introduces substantial latency or computational overhead, and is incompatible with token-level streaming generation. In this work, we introduce Self-Sanitize, a novel LLM-driven mitigation framework inspired by cognitive psychology, which emulates human self-monitor and self-repair behaviors during conversations. Self-Sanitize comprises a lightweight Self-Monitor module that continuously inspects high-level intentions within the LLM at the token level via representation engineering, and a Self-Repair module that performs in-place correction of harmful content without initiating separate review dialogues. This design allows for real-time streaming monitoring and seamless repair, with negligible impact on latency and resource utilization. Given that privacy-invasive content has often been insufficiently focused in previous studies, we perform extensive experiments on four LLMs across three privacy leakage scenarios. The results demonstrate that Self-Sanitize achieves superior mitigation performance with minimal overhead and without degrading the utility of LLMs, offering a practical and robust solution for safer LLM deployments. Our code is available at the following link: this https URL 

---
# A Text-To-Text Alignment Algorithm for Better Evaluation of Modern Speech Recognition Systems 

**Authors**: Lasse Borgholt, Jakob Havtorn, Christian Igel, Lars Maaløe, Zheng-Hua Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24478)  

**Abstract**: Modern neural networks have greatly improved performance across speech recognition benchmarks. However, gains are often driven by frequent words with limited semantic weight, which can obscure meaningful differences in word error rate, the primary evaluation metric. Errors in rare terms, named entities, and domain-specific vocabulary are more consequential, but remain hidden by aggregate metrics. This highlights the need for finer-grained error analysis, which depends on accurate alignment between reference and model transcripts. However, conventional alignment methods are not designed for such precision. We propose a novel alignment algorithm that couples dynamic programming with beam search scoring. Compared to traditional text alignment methods, our approach provides more accurate alignment of individual errors, enabling reliable error analysis. The algorithm is made available via PyPI. 

---
# Bias Mitigation or Cultural Commonsense? Evaluating LLMs with a Japanese Dataset 

**Authors**: Taisei Yamamoto, Ryoma Kumon, Danushka Bollegala, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2509.24468)  

**Abstract**: Large language models (LLMs) exhibit social biases, prompting the development of various debiasing methods. However, debiasing methods may degrade the capabilities of LLMs. Previous research has evaluated the impact of bias mitigation primarily through tasks measuring general language understanding, which are often unrelated to social biases. In contrast, cultural commonsense is closely related to social biases, as both are rooted in social norms and values. The impact of bias mitigation on cultural commonsense in LLMs has not been well investigated. Considering this gap, we propose SOBACO (SOcial BiAs and Cultural cOmmonsense benchmark), a Japanese benchmark designed to evaluate social biases and cultural commonsense in LLMs in a unified format. We evaluate several LLMs on SOBACO to examine how debiasing methods affect cultural commonsense in LLMs. Our results reveal that the debiasing methods degrade the performance of the LLMs on the cultural commonsense task (up to 75% accuracy deterioration). These results highlight the importance of developing debiasing methods that consider the trade-off with cultural commonsense to improve fairness and utility of LLMs. 

---
# Alternatives To Next Token Prediction In Text Generation - A Survey 

**Authors**: Charlie Wyatt, Aditya Joshi, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2509.24435)  

**Abstract**: The paradigm of Next Token Prediction (NTP) has driven the unprecedented success of Large Language Models (LLMs), but is also the source of their most persistent weaknesses such as poor long-term planning, error accumulation, and computational inefficiency. Acknowledging the growing interest in exploring alternatives to NTP, the survey describes the emerging ecosystem of alternatives to NTP. We categorise these approaches into five main families: (1) Multi-Token Prediction, which targets a block of future tokens instead of a single one; (2) Plan-then-Generate, where a global, high-level plan is created upfront to guide token-level decoding; (3) Latent Reasoning, which shifts the autoregressive process itself into a continuous latent space; (4) Continuous Generation Approaches, which replace sequential generation with iterative, parallel refinement through diffusion, flow matching, or energy-based methods; and (5) Non-Transformer Architectures, which sidestep NTP through their inherent model structure. By synthesizing insights across these methods, this survey offers a taxonomy to guide research into models that address the known limitations of token-level generation to develop new transformative models for natural language processing. 

---
# CDT: A Comprehensive Capability Framework for Large Language Models Across Cognition, Domain, and Task 

**Authors**: Haosi Mo, Xinyu Ma, Xuebo Liu, Derek F. Wong, Yu Li, Jie Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24422)  

**Abstract**: Recent advances in Large Language Models (LLMs) have significantly enhanced their capabilities, highlighting the need for comprehensive evaluation frameworks that extend beyond task-specific benchmarks. However, existing benchmarks often focus on isolated abilities, lacking a holistic framework for assessing LLM capabilities. To address this gap, we propose the Cognition-Domain-Task (CDT) framework, which comprehensively measures a model's capabilities across three dimensions. We expand the scope of model capability definitions at the cognitive level by incorporating the Cattell-Horn-Carroll cognitive theory, refining the categorization of model capabilities. We apply CDT in two directions: dataset capability evaluation and data selection. Experiments show that our capability metrics correlate well with downstream performance and can support effective dataset analysis and construction. The experiments on data selection also show significant improvements in both general and specific benchmarks, achieving scores of 44.3 and 45.4, with an increase of 1.6 and 2.2 points over the baselines, respectively. These results validate the effectiveness and practicality of CDT. Source code and models are available at this https URL. 

---
# Multilingual Text-to-SQL: Benchmarking the Limits of Language Models with Collaborative Language Agents 

**Authors**: Khanh Trinh Pham, Thu Huong Nguyen, Jun Jo, Quoc Viet Hung Nguyen, Thanh Tam Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24405)  

**Abstract**: Text-to-SQL enables natural access to databases, yet most benchmarks are English-only, limiting multilingual progress. We introduce MultiSpider 2.0, extending Spider 2.0 to eight languages (English, German, French, Spanish, Portuguese, Japanese, Chinese, Vietnamese). It preserves Spider 2.0's structural difficulty while adding linguistic and dialectal variability, demanding deeper reasoning for complex SQL. On this benchmark, state-of-the-art LLMs (such as DeepSeek-R1 and OpenAI o1) reach only 4\% execution accuracy when relying on intrinsic reasoning, versus 60\% on MultiSpider 1.0. Therefore, we provide a collaboration-driven language agents baseline that iteratively refines queries, improving accuracy to 15\%. These results reveal a substantial multilingual gap and motivate methods that are robust across languages and ready for real-world enterprise deployment. Our benchmark is available at this https URL. 

---
# Agentar-Scale-SQL: Advancing Text-to-SQL through Orchestrated Test-Time Scaling 

**Authors**: Pengfei Wang, Baolin Sun, Xuemei Dong, Yaxun Dai, Hongwei Yuan, Mengdie Chu, Yingqi Gao, Xiang Qi, Peng Zhang, Ying Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24403)  

**Abstract**: State-of-the-art (SOTA) Text-to-SQL methods still lag significantly behind human experts on challenging benchmarks like BIRD. Current approaches that explore test-time scaling lack an orchestrated strategy and neglect the model's internal reasoning process. To bridge this gap, we introduce Agentar-Scale-SQL, a novel framework leveraging scalable computation to improve performance. Agentar-Scale-SQL implements an Orchestrated Test-Time Scaling strategy that synergistically combines three distinct perspectives: i) Internal Scaling via RL-enhanced Intrinsic Reasoning, ii) Sequential Scaling through Iterative Refinement, and iii) Parallel Scaling using Diverse Synthesis and Tournament Selection. Agentar-Scale-SQL is a general-purpose framework designed for easy adaptation to new databases and more powerful language models. Extensive experiments show that Agentar-Scale-SQL achieves SOTA performance on the BIRD benchmark, reaching 81.67\% execution accuracy on the test set and ranking first on the official leaderboard, demonstrating an effective path toward human-level performance. 

---
# LLaDA-MoE: A Sparse MoE Diffusion Language Model 

**Authors**: Fengqi Zhu, Zebin You, Yipeng Xing, Zenan Huang, Lin Liu, Yihong Zhuang, Guoshan Lu, Kangyu Wang, Xudong Wang, Lanning Wei, Hongrui Guo, Jiaqi Hu, Wentao Ye, Tieyuan Chen, Chenchen Li, Chengfu Tang, Haibo Feng, Jun Hu, Jun Zhou, Xiaolu Zhang, Zhenzhong Lan, Junbo Zhao, Da Zheng, Chongxuan Li, Jianguo Li, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24389)  

**Abstract**: We introduce LLaDA-MoE, a large language diffusion model with the Mixture-of-Experts (MoE) architecture, trained from scratch on approximately 20T tokens. LLaDA-MoE achieves competitive performance with significantly reduced computational overhead by maintaining a 7B-parameter capacity while activating only 1.4B parameters during inference. Our empirical evaluation reveals that LLaDA-MoE achieves state-of-the-art performance among diffusion language models with larger parameters, surpassing previous diffusion language models LLaDA, LLaDA 1.5, and Dream across multiple benchmarks. The instruct-tuned model LLaDA-MoE-7B-A1B-Instruct demonstrates capabilities comparable to Qwen2.5-3B-Instruct in knowledge understanding, code generation, mathematical reasoning, agent and alignment tasks, despite using fewer active parameters. Our results show that integrating a sparse MoE architecture into the training objective of masked diffusion language models still brings out MoE's strengths under efficient inference with few active parameters, and opens ample room for further exploration of diffusion language models. LLaDA-MoE models are available at Huggingface. 

---
# HarmMetric Eval: Benchmarking Metrics and Judges for LLM Harmfulness Assessment 

**Authors**: Langqi Yang, Tianhang Zheng, Kedong Xiu, Yixuan Chen, Di Wang, Puning Zhao, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.24384)  

**Abstract**: The alignment of large language models (LLMs) with human values is critical for their safe deployment, yet jailbreak attacks can subvert this alignment to elicit harmful outputs from LLMs. In recent years, a proliferation of jailbreak attacks has emerged, accompanied by diverse metrics and judges to assess the harmfulness of the LLM outputs. However, the absence of a systematic benchmark to assess the quality and effectiveness of these metrics and judges undermines the credibility of the reported jailbreak effectiveness and other risks. To address this gap, we introduce HarmMetric Eval, a comprehensive benchmark designed to support both overall and fine-grained evaluation of harmfulness metrics and judges. Our benchmark includes a high-quality dataset of representative harmful prompts paired with diverse harmful and non-harmful model responses, alongside a flexible scoring mechanism compatible with various metrics and judges. With HarmMetric Eval, our extensive experiments uncover a surprising result: two conventional metrics--METEOR and ROUGE-1--outperform LLM-based judges in evaluating the harmfulness of model responses, challenging prevailing beliefs about LLMs' superiority in this domain. Our dataset is publicly available at this https URL, and the code is available at this https URL. 

---
# Reinforcement Mid-Training 

**Authors**: Yijun Tian, Shaoyu Chen, Zhichao Xu, Yawei Wang, Jinhe Bi, Peng Han, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24375)  

**Abstract**: The development of state-of-the-art large language models is commonly understood as a two-stage process involving pre-training and post-training. We point out the need for an additional intermediate stage called reinforcement mid-training with potential for strong performance gains. In this paper, we formally define the problem and identify three key challenges: (1) inefficient training due to excessive reasoning steps, (2) disregard of the imbalanced token entropy distribution, and (3) underutilization of token information. To address these challenges, we propose RMT, a framework for efficient, adaptive, and unified reinforcement mid-training with various innovative components. In particular, we first introduce a dynamic token budget mechanism that constrains unnecessary reasoning steps and mitigates model overthinking. Next, we design a curriculum-based adaptive sampling method that fosters a progressive learning trajectory from easy to hard tokens. Finally, we present a dual training strategy that combines reinforcement learning with next-token prediction, ensuring targeted learning on key tokens and full exploitation of all token information. Extensive experiments demonstrate the superiority of RMT over state-of-the-art methods, achieving up to +64.91% performance improvement with only 21% of the reasoning length in language modeling. We also show that checkpoints obtained after reinforcement mid-training can benefit the subsequent post-training, yielding up to +18.76% improvement in the mathematical domain. 

---
# Beyond Repetition: Text Simplification and Curriculum Learning for Data-Constrained Pretraining 

**Authors**: Matthew Theodore Roque, Dan John Velasco  

**Link**: [PDF](https://arxiv.org/pdf/2509.24356)  

**Abstract**: Most studies on language model pretraining focus on large datasets, leaving open questions about optimization in data-constrained settings. In such settings, the effects of training data order and of including alternative versions of the same text remain underexplored. We address this by studying curriculum learning in pretraining, focusing on text-complexity ordering and data augmentation via simplification. We ask: (1) Does simplifying texts enhance representation quality more than reusing the original data? and (2) Does ordering data by text complexity yield better representations? To answer, we build on a pair of parallel corpora where human-written paragraphs are aligned with LLM-simplified variants, and test four data schedules: repeated exposure, low-to-high complexity, high-to-low, and interleaved. We analyze models' representation quality from a sample efficiency perspective via fine-tuning, as well as its zero-shot performance on linguistic knowledge, entity tracking, world knowledge, and commonsense reasoning. Our findings show that adding simplified data improves fine-tuning and zero-shot performance over a repeated-exposure baseline: smaller models benefit from low-to-high complexity, while larger models perform better with interleaved ordering. 

---
# AlignX: Advancing Multilingual Large Language Models with Multilingual Representation Alignment 

**Authors**: Mengyu Bu, Shaolei Zhang, Zhongjun He, Hua Wu, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.24338)  

**Abstract**: Multilingual large language models (LLMs) possess impressive multilingual understanding and generation capabilities. However, their performance and cross-lingual alignment often lag for non-dominant languages. A common solution is to fine-tune LLMs on large-scale and more balanced multilingual corpus, but such approaches often lead to imprecise alignment and suboptimal knowledge transfer, struggling with limited improvements across languages. In this paper, we propose AlignX to bridge the multilingual performance gap, which is a two-stage representation-level framework for enhancing multilingual performance of pre-trained LLMs. In the first stage, we align multilingual representations with multilingual semantic alignment and language feature integration. In the second stage, we stimulate the multilingual capability of LLMs via multilingual instruction fine-tuning. Experimental results on several pre-trained LLMs demonstrate that our approach enhances LLMs' multilingual general and cross-lingual generation capability. Further analysis indicates that AlignX brings the multilingual representations closer and improves the cross-lingual alignment. 

---
# Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding 

**Authors**: Sungkyun Kim, Jaemin Kim, Dogyung Yoon, Jiho Shin, Junyeol Lee, Jiwon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2509.24328)  

**Abstract**: LLMs have low GPU efficiency and high latency due to autoregressive decoding. Speculative decoding (SD) mitigates this using a small draft model to speculatively generate multiple tokens, which are then verified in parallel by a target model. However, when speculation accuracy is low, the overhead from rejected tokens can offset the benefits, limiting SD's effectiveness, especially at large batch sizes. To address this, we propose Speculative Verification (SV), an efficient augmentation to SD that dynamically predicts speculation accuracy and adapts the verification length to maximize throughput. SV introduces a companion model - a small auxiliary model similar in size to the draft model - to estimate the alignment between draft and target model distributions. By maximizing the information gain from quantifying this alignment, SV refines verification decisions, reducing wasted computation on rejected tokens and improving decoding efficiency. Moreover, SV requires no modifications to the draft or target models and is compatible with existing SD variants. We extensively evaluated SV on publicly available LLMs across three NLP tasks using nine combinations of draft, companion, and target models, including 13B-72B target models and three types of variations: base (no finetuning), instruction-tuned, and task fine-tuned. Across all experiments and batch sizes (4-80), SV consistently outperforms both SD and standard decoding with the target model. It improves SD performance by up to 2$\times$, with an average speedup of 1.4 $\times$ in large-batch settings (batch sizes 32-80). These results demonstrate SV's robustness, scalability, and practical utility for efficient LLM inference. 

---
# Multimodal Large Language Models Meet Multimodal Emotion Recognition and Reasoning: A Survey 

**Authors**: Yuntao Shou, Tao Meng, Wei Ai, Keqin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.24322)  

**Abstract**: In recent years, large language models (LLMs) have driven major advances in language understanding, marking a significant step toward artificial general intelligence (AGI). With increasing demands for higher-level semantics and cross-modal fusion, multimodal large language models (MLLMs) have emerged, integrating diverse information sources (e.g., text, vision, and audio) to enhance modeling and reasoning in complex scenarios. In AI for Science, multimodal emotion recognition and reasoning has become a rapidly growing frontier. While LLMs and MLLMs have achieved notable progress in this area, the field still lacks a systematic review that consolidates recent developments. To address this gap, this paper provides a comprehensive survey of LLMs and MLLMs for emotion recognition and reasoning, covering model architectures, datasets, and performance benchmarks. We further highlight key challenges and outline future research directions, aiming to offer researchers both an authoritative reference and practical insights for advancing this domain. To the best of our knowledge, this paper is the first attempt to comprehensively survey the intersection of MLLMs with multimodal emotion recognition and reasoning. The summary of existing methods mentioned is in our Github: \href{this https URL}{this https URL}. 

---
# Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in LLMs 

**Authors**: Jongwook Han, Jongwon Lim, Injin Kong, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.24319)  

**Abstract**: Large language models (LLMs) can express different values in two distinct ways: (1) intrinsic expression, reflecting the model's inherent values learned during training, and (2) prompted expression, elicited by explicit prompts. Given their widespread use in value alignment and persona steering, it is paramount to clearly understand their underlying mechanisms, particularly whether they mostly overlap (as one might expect) or rely on substantially different mechanisms, but this remains largely understudied. We analyze this at the mechanistic level using two approaches: (1) value vectors, feature directions representing value mechanisms extracted from the residual stream, and (2) value neurons, MLP neurons that contribute to value expressions. We demonstrate that intrinsic and prompted value mechanisms partly share common components that are crucial for inducing value expression, but also possess unique elements that manifest in different ways. As a result, these mechanisms lead to different degrees of value steerability (prompted > intrinsic) and response diversity (intrinsic > prompted). In particular, components unique to the intrinsic mechanism seem to promote lexical diversity in responses, whereas those specific to the prompted mechanism primarily strengthen instruction following, taking effect even in distant tasks like jailbreaking. 

---
# Q-Mirror: Unlocking the Multi-Modal Potential of Scientific Text-Only QA Pairs 

**Authors**: Junying Wang, Zicheng Zhang, Ye Shen, Yalun Wu, Yingji Liang, Yijin Guo, Farong Wen, Wenzhe Li, Xuezhi Zhao, Qi Jia, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2509.24297)  

**Abstract**: High-quality, multi-modal benchmarks are crucial for advancing scientific reasoning in large models yet their manual creation is costly and unscalable. To address this bottleneck, we explore the potential for transforming Text-Only QA Pairs (TQAs) into high-quality Multi-Modal QA Pairs (MMQAs), which include three parts: 1) Task Definition \& Evaluation Rubric: We develop a TQA-to-MMQA framework and establish a comprehensive, multi-dimensional MMQA quality rubric that provides principles for the transformation. 2) Benchmark Construction: Then we construct two extensive benchmarks to rigorously evaluate state-of-the-art generation \& understanding models on the distinct tasks of MMQA generation \& MMQA quality evaluation. 3) Preliminary Solution: We develop an agentic system (Q-Mirror), which operationalizes our framework by integrating MMQA generation and evaluation into a closed loop for iterative refinement. Our experiments show that while state-of-the-art models can generate MMQAs, their outputs still leave substantial gaps, underscoring the need for reliable evaluation. We further demonstrate that top-tier understanding models align closely with human judgment in MMQA quality assessment. Leveraging both insights, the Q-Mirror agent raises average scores from 78.90 to 85.22 and pass rates from 72\% to 95\%, offering a practical path to large-scale scientific benchmarks. 

---
# DiffuGuard: How Intrinsic Safety is Lost and Found in Diffusion Large Language Models 

**Authors**: Zherui Li, Zheng Nie, Zhenhong Zhou, Yufei Guo, Yue Liu, Yitong Zhang, Yu Cheng, Qingsong Wen, Kun Wang, Jiaheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24296)  

**Abstract**: The rapid advancement of Diffusion Large Language Models (dLLMs) introduces unprecedented vulnerabilities that are fundamentally distinct from Autoregressive LLMs, stemming from their iterative and parallel generation mechanisms. In this paper, we conduct an in-depth analysis of dLLM vulnerabilities to jailbreak attacks across two distinct dimensions: intra-step and inter-step dynamics. Experimental results reveal a harmful bias inherent in the standard greedy remasking strategy and identify a critical phenomenon we term Denoising-path Dependence, where the safety of early-stage tokens decisively influences the final output. These findings also indicate that while current decoding strategies constitute a significant vulnerability, dLLMs possess a substantial intrinsic safety potential. To unlock this potential, we propose DiffuGuard, a training-free defense framework that addresses vulnerabilities through a dual-stage approach: Stochastic Annealing Remasking dynamically introduces controlled randomness to mitigate greedy selection bias, while Block-level Audit and Repair exploits internal model representations for autonomous risk detection and guided correction. Comprehensive experiments on four dLLMs demonstrate DiffuGuard's exceptional effectiveness, reducing Attack Success Rate against six diverse jailbreak methods from 47.9% to 14.7% while preserving model utility and efficiency. Our code is available at: this https URL. 

---
# LOGOS: LLM-driven End-to-End Grounded Theory Development and Schema Induction for Qualitative Research 

**Authors**: Xinyu Pi, Qisen Yang, Chuong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24294)  

**Abstract**: Grounded theory offers deep insights from qualitative data, but its reliance on expert-intensive manual coding presents a major scalability bottleneck. Current computational tools stop short of true automation, keeping researchers firmly in the loop. We introduce LOGOS, a novel, end-to-end framework that fully automates the grounded theory workflow, transforming raw text into a structured, hierarchical theory. LOGOS integrates LLM-driven coding, semantic clustering, graph reasoning, and a novel iterative refinement process to build highly reusable codebooks. To ensure fair comparison, we also introduce a principled 5-dimensional metric and a train-test split protocol for standardized, unbiased evaluation. Across five diverse corpora, LOGOS consistently outperforms strong baselines and achieves a remarkable $88.2\%$ alignment with an expert-developed schema on a complex dataset. LOGOS demonstrates a powerful new path to democratize and scale qualitative research without sacrificing theoretical nuance. 

---
# Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement 

**Authors**: Yu-Che Tsai, Kuan-Yu Chen, Yuan-Chi Li, Yuan-Hao Chen, Ching-Yu Tsai, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.24291)  

**Abstract**: Existing large language model (LLM)-based embeddings typically adopt an encoder-only paradigm, treating LLMs as static feature extractors and overlooking their core generative strengths. We introduce GIRCSE (Generative Iterative Refinement for Contrastive Sentence Embeddings), a novel framework that leverages autoregressive generation to iteratively refine semantic representations. By producing sequences of soft tokens optimized under contrastive objective, GIRCSE captures latent concepts and implicit semantics that encoder-only methods often miss. To guide this process, we propose an Iterative Contrastive Refinement (ICR) objective that encourages each refinement step to yield better representations. Extensive experiments show that GIRCSE outperforms strong LLM-based embedding baselines on the MTEB benchmark and instruction-following tasks. Moreover, GIRCSE exhibits an emergent test-time scaling property: generating more tokens at inference steadily improves embedding quality. Our results establish generative iterative refinement as a new paradigm for representation learning. 

---
# SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents 

**Authors**: Gyuhyeon Seo, Jungwoo Yang, Junseong Pyo, Nalim Kim, Jonggeun Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.24282)  

**Abstract**: Large Language Model (LLM) agents excel at multi-step, tool-augmented tasks. However, smart homes introduce distinct challenges, requiring agents to handle latent user intents, temporal dependencies, device constraints, scheduling, and more. The main bottlenecks for developing smart home agents with such capabilities include the lack of a realistic simulation environment where agents can interact with devices and observe the results, as well as a challenging benchmark to evaluate them. To address this, we introduce $\textbf{SimuHome}$, a time-accelerated home environment that simulates smart devices, supports API calls, and reflects changes in environmental variables. By building the simulator on the Matter protocol (the global industry standard for smart home communication), SimuHome provides a high-fidelity environment, and agents validated in SimuHome can be deployed on real Matter-compliant devices with minimal adaptation. We provide a challenging benchmark of 600 episodes across twelve user query types that require the aforementioned capabilities. Our evaluation of 11 agents under a unified ReAct framework reveals that while models perform well on simple tasks, they struggle with latent intent inference, state verification, and especially temporal scheduling. Even the top-performing model, GPT-4.1, reaches only 54% success rate. These findings highlight a critical need for methods that can reliably verify the current state via tools before acting and coordinate time-dependent actions. 

---
# MRAG-Suite: A Diagnostic Evaluation Platform for Visual Retrieval-Augmented Generation 

**Authors**: Yuelyu Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.24253)  

**Abstract**: Multimodal Retrieval-Augmented Generation (Visual RAG) significantly advances question answering by integrating visual and textual evidence. Yet, current evaluations fail to systematically account for query difficulty and ambiguity. We propose MRAG-Suite, a diagnostic evaluation platform integrating diverse multimodal benchmarks (WebQA, Chart-RAG, Visual-RAG, MRAG-Bench). We introduce difficulty-based and ambiguity-aware filtering strategies, alongside MM-RAGChecker, a claim-level diagnostic tool. Our results demonstrate substantial accuracy reductions under difficult and ambiguous queries, highlighting prevalent hallucinations. MM-RAGChecker effectively diagnoses these issues, guiding future improvements in Visual RAG systems. 

---
# Prompt and Parameter Co-Optimization for Large Language Models 

**Authors**: Xiaohe Bo, Rui Li, Zexu Sun, Quanyu Dai, Zeyu Zhang, Zihang Tian, Xu Chen, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.24245)  

**Abstract**: Prompt optimization and fine-tuning are two major approaches to improve the performance of Large Language Models (LLMs). They enhance the capabilities of LLMs from complementary perspectives: the former through explicit natural language, and the latter through implicit parameter updates. However, prior work has typically studied them in isolation, leaving their synergistic potential largely underexplored. To bridge this gap, in this paper, we introduce MetaTuner, a novel framework that jointly integrates prompt optimization and fine-tuning for LLM training. Specifically, we introduce two neural networks to generate prompts and parameters, respectively, while allowing them to share a common bottom encoding layer to enable knowledge sharing. By the guidance of the final supervised signals, our framework is optimized to discover the optimal combinations between the prompts and parameters. Given that prompt learning involves discrete optimization while fine-tuning operates in a continuous parameter space, we design a supervised regularization loss to train our framework effectively. Extensive experiments across diverse benchmarks show that our method consistently outperforms the baselines. 

---
# Model Fusion with Multi-LoRA Inference for Tool-Enhanced Game Dialogue Agents 

**Authors**: Kangxu Wang, Ze Chen, Chengcheng Wei, Jiewen Zheng, Jiarong He, Max Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.24229)  

**Abstract**: This paper presents the opdainlp team's solution for the GPU track of the CPDC 2025 challenge. The challenge consists of three tasks, aiming to build an in-game conversational AI that adheres to character personas, aligns with the game's worldview, and supports function calling. Considering both effectiveness and resource/time constraints during inference, we synthesized data for some of the tasks based on the datasets provided by the competition organizers. We employed Qwen3-14B with LoRA fine-tuning and model fusion, and utilized a base model integrated with multiple LoRA adapters during inference. Specifically, in the competition, we used three distinct LoRA adapters to handle tool calling, response generation with tool call results, and response generation without tool call results, respectively. MultiLoRA inference was implemented using vLLM. Our solution achieved the first place in Task 1 and Task 3, and the second place in Task 2 of the GPU track. 

---
# MoVa: Towards Generalizable Classification of Human Morals and Values 

**Authors**: Ziyu Chen, Junfei Sun, Chenxi Li, Tuan Dung Nguyen, Jing Yao, Xiaoyuan Yi, Xing Xie, Chenhao Tan, Lexing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.24216)  

**Abstract**: Identifying human morals and values embedded in language is essential to empirical studies of communication. However, researchers often face substantial difficulty navigating the diversity of theoretical frameworks and data available for their analysis. Here, we contribute MoVa, a well-documented suite of resources for generalizable classification of human morals and values, consisting of (1) 16 labeled datasets and benchmarking results from four theoretically-grounded frameworks; (2) a lightweight LLM prompting strategy that outperforms fine-tuned models across multiple domains and frameworks; and (3) a new application that helps evaluate psychological surveys. In practice, we specifically recommend a classification strategy, all@once, that scores all related concepts simultaneously, resembling the well-known multi-label classifier chain. The data and methods in MoVa can facilitate many fine-grained interpretations of human and machine communication, with potential implications for the alignment of machine behavior. 

---
# ScenarioBench: Trace-Grounded Compliance Evaluation for Text-to-SQL and RAG 

**Authors**: Zahra Atf, Peter R Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2509.24212)  

**Abstract**: ScenarioBench is a policy-grounded, trace-aware benchmark for evaluating Text-to-SQL and retrieval-augmented generation in compliance contexts. Each YAML scenario includes a no-peek gold-standard package with the expected decision, a minimal witness trace, the governing clause set, and the canonical SQL, enabling end-to-end scoring of both what a system decides and why. Systems must justify outputs using clause IDs from the same policy canon, making explanations falsifiable and audit-ready. The evaluator reports decision accuracy, trace quality (completeness, correctness, order), retrieval effectiveness, SQL correctness via result-set equivalence, policy coverage, latency, and an explanation-hallucination rate. A normalized Scenario Difficulty Index (SDI) and a budgeted variant (SDI-R) aggregate results while accounting for retrieval difficulty and time. Compared with prior Text-to-SQL or KILT/RAG benchmarks, ScenarioBench ties each decision to clause-level evidence under strict grounding and no-peek rules, shifting gains toward justification quality under explicit time budgets. 

---
# BeyondBench: Benchmark-Free Evaluation of Reasoning in Language Models 

**Authors**: Gaurav Srivastava, Aafiya Hussain, Zhenyu Bi, Swastik Roy, Priya Pitre, Meng Lu, Morteza Ziyadi, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24210)  

**Abstract**: Evaluating language models fairly is becoming harder as static benchmarks available on the internet risk contamination by training data. This makes it unclear whether models are truly reasoning or just recalling answers. In this paper, we introduce BeyondBench, an evaluation framework that avoids this problem by using algorithmic problem generation. Unlike traditional benchmarks that risk contamination from internet-scale training data, BeyondBench creates mathematically grounded problems on the fly, ensuring each test remains fresh and uncontaminated. Our framework covers 44 algorithmic tasks with a total of 117 variations, grouped into three difficulty levels: the Easy Suite (29 tasks) for basic arithmetic and statistics, the Medium Suite (5 tasks, 49 variations) for sequence patterns and reasoning, and the Hard Suite (10 tasks, 68 variations) tackling NP-complete and constraint satisfaction problems. Each task generates problems from a combinatorial space larger than 10^15 unique instances, with solutions verified deterministically by mathematical proofs. We evaluated 101 language models, including 85 open-source and 16 closed-source models, spanning sizes from 0.5B to 141B parameters and multiple quantization schemes. Our results show consistent reasoning deficiencies across model families, with performance degrading sharply as problem complexity increases from polynomial to exponential. In our Hard Suite evaluations, models such as Gemini-2.5-pro, Llama-3.3-70B, and Qwen2.5-72B achieved average accuracies of 56.38%, 26.91%, and 33.60%, respectively. Moreover, we observe that performance drops drastically without tool usage, with GPT-5, GPT-5-mini, and GPT-5-nano showing a decline of 16.81%, 28.05%, and 47.59% accuracy on the hard suite. Our leaderboard is publicly available at this https URL 

---
# Can Large Language Models Express Uncertainty Like Human? 

**Authors**: Linwei Tao, Yi-Fan Yeh, Bo Kai, Minjing Dong, Tao Huang, Tom A. Lamb, Jialin Yu, Philip H.S. Torr, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24202)  

**Abstract**: Large language models (LLMs) are increasingly used in high-stakes settings, where overconfident responses can mislead users. Reliable confidence estimation has been shown to enhance trust and task accuracy. Yet existing methods face practical barriers: logits are often hidden, multi-sampling is computationally expensive, and verbalized numerical uncertainty (e.g., giving a 0-100 score) deviates from natural communication. We revisit linguistic confidence (LC), where models express uncertainty through hedging language (e.g., probably, might), offering a lightweight and human-centered alternative. To advance this direction, we (1) release the first diverse, large-scale dataset of hedging expressions with human-annotated confidence scores, and (2) propose a lightweight mapper that converts hedges into confidence scores at near-zero cost. Building on these resources, we (3) conduct the first systematic study of LC across modern LLMs and QA benchmarks, revealing that while most LLMs underperform in expressing reliable LC, carefully designed prompting achieves competitive calibration and discriminability. Finally, we (4) introduce a fine-tuning framework that further improves LC reliability. Taken together, our work positions linguistic confidence as a scalable, efficient, and human-aligned approach to LLM uncertainty estimation, and calls for deeper exploration of this promising yet underexplored direction. 

---
# AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play 

**Authors**: Ran Xu, Yuchen Zhuang, Zihan Dong, Jonathan Wang, Yue Yu, Joyce C. Ho, Linjun Zhang, Haoyu Wang, Wenqi Shi, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24193)  

**Abstract**: Search-augmented LLMs often struggle with complex reasoning tasks due to ineffective multi-hop retrieval and limited reasoning ability. We propose AceSearcher, a cooperative self-play framework that trains a single large language model (LLM) to alternate between two roles: a decomposer that breaks down complex queries and a solver that integrates retrieved contexts for answer generation. AceSearcher couples supervised fine-tuning on a diverse mixture of search, reasoning, and decomposition tasks with reinforcement fine-tuning optimized for final answer accuracy, eliminating the need for intermediate annotations. Extensive experiments on three reasoning-intensive tasks across 10 datasets show that AceSearcher outperforms state-of-the-art baselines, achieving an average exact match improvement of 7.6%. Remarkably, on document-level finance reasoning tasks, AceSearcher-32B matches the performance of the DeepSeek-V3 model using less than 5% of its parameters. Even at smaller scales (1.5B and 8B), AceSearcher often surpasses existing search-augmented LLMs with up to 9x more parameters, highlighting its exceptional efficiency and effectiveness in tackling complex reasoning tasks. Our code will be published at this https URL and this https URL. 

---
# PET: Preference Evolution Tracking with LLM-Generated Explainable Distribution 

**Authors**: Luyang Zhang, Siyuan Peng, Jialu Wang, Shichao Zhu, Beibei Li, Zhongcun Wang, Guangmou Pan, Yan Li, Song Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24189)  

**Abstract**: Understanding how user preference evolves over time is a fundamental challenge central to modern digital ecosystems, for which Large Language Models (LLMs) are an increasingly prominent and popular approach due to their ability to comprehend the rich semantic context within behavioral data. A common practice is to use LLMs to predict a user's next action by directly generating a ranked list of preferred items. Although effective for short-term prediction, the end-to-end generation paradigm inherently limits personalization. Its opaque decision-making process obscures holistic user profiling and exacerbates popularity bias. To address these limitations, we propose Preference Evolution Tracking (PET), a framework that reframes the task as inferring a dynamic probability distribution over a stable and interpretable lattice of preference clusters. By applying logit-probing and generative classification techniques, PET infers a user's preference as a probability distribution, enabling transparent preference learning. On public benchmarks (Yelp, MovieLens), PET improves ranking quality by up to 40% in NDCG over direct generation baselines. On a large-scale, real-world dataset from a short-video platform, it excels at ranking long-tail contents, significantly outperforming a SOTA production model by 7 times in the NDCG score. Ultimately, PET transforms the user profile model from direct preference list generation to a transparent distributional preference mapping, paving the way for more explainable, fair, and diverse personalization systems. 

---
# Beyond Overall Accuracy: A Psychometric Deep Dive into the Topic-Specific Medical Capabilities of 80 Large Language Models 

**Authors**: Zhimeng Luo, Lixin Wu, Adam Frisch, Daqing He  

**Link**: [PDF](https://arxiv.org/pdf/2509.24186)  

**Abstract**: As Large Language Models (LLMs) are increasingly proposed for high-stakes medical applications, there has emerged a critical need for reliable and accurate evaluation methodologies. Traditional accuracy metrics fail inadequately as they neither capture question characteristics nor offer topic-specific insights. To address this gap, we introduce \textsc{MedIRT}, a rigorous evaluation framework grounded in Item Response Theory (IRT), the gold standard in high-stakes educational testing. Unlike previous research relying on archival data, we prospectively gathered fresh responses from 80 diverse LLMs on a balanced, 1,100-question USMLE-aligned benchmark. Using one unidimensional two-parameter logistic IRT model per topic, we estimate LLM's latent model ability jointly with question difficulty and discrimination, yielding more stable and nuanced performance rankings than accuracy alone. Notably, we identify distinctive ``spiky'' ability profiles, where overall rankings can be misleading due to highly specialized model abilities. While \texttt{GPT-5} was the top performer in a majority of domains (8 of 11), it was outperformed in Social Science and Communication by \texttt{Claude-3-opus}, demonstrating that even an overall 23rd-ranked model can hold the top spot for specific competencies. Furthermore, we demonstrate IRT's utility in auditing benchmarks by identifying flawed questions. We synthesize these findings into a practical decision-support framework that integrates our multi-factor competency profiles with operational metrics. This work establishes a robust, psychometrically grounded methodology essential for the safe, effective, and trustworthy deployment of LLMs in healthcare. 

---
# Retrieval-augmented GUI Agents with Generative Guidelines 

**Authors**: Ran Xu, Kaixin Ma, Wenhao Yu, Hongming Zhang, Joyce C. Ho, Carl Yang, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24183)  

**Abstract**: GUI agents powered by vision-language models (VLMs) show promise in automating complex digital tasks. However, their effectiveness in real-world applications is often limited by scarce training data and the inherent complexity of these tasks, which frequently require long-tailed knowledge covering rare, unseen scenarios. We propose RAG-GUI , a lightweight VLM that leverages web tutorials at inference time. RAG-GUI is first warm-started via supervised finetuning (SFT) and further refined through self-guided rejection sampling finetuning (RSF). Designed to be model-agnostic, RAG-GUI functions as a generic plug-in that enhances any VLM-based agent. Evaluated across three distinct tasks, it consistently outperforms baseline agents and surpasses other inference baselines by 2.6% to 13.3% across two model sizes, demonstrating strong generalization and practical plug-and-play capabilities in real-world scenarios. 

---
# Task Vectors, Learned Not Extracted: Performance Gains and Mechanistic Insight 

**Authors**: Haolin Yang, Hakaze Cho, Kaize Ding, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2509.24169)  

**Abstract**: Large Language Models (LLMs) can perform new tasks from in-context demonstrations, a phenomenon known as in-context learning (ICL). Recent work suggests that these demonstrations are compressed into task vectors (TVs), compact task representations that LLMs exploit for predictions. However, prior studies typically extract TVs from model outputs or hidden states using cumbersome and opaque methods, and they rarely elucidate the mechanisms by which TVs influence computation. In this work, we address both limitations. First, we propose directly training Learned Task Vectors (LTVs), which surpass extracted TVs in accuracy and exhibit superior flexibility-acting effectively at arbitrary layers, positions, and even with ICL prompts. Second, through systematic analysis, we investigate the mechanistic role of TVs, showing that at the low level they steer predictions primarily through attention-head OV circuits, with a small subset of "key heads" most decisive. At a higher level, we find that despite Transformer nonlinearities, TV propagation is largely linear: early TVs are rotated toward task-relevant subspaces to improve logits of relevant labels, while later TVs are predominantly scaled in magnitude. Taken together, LTVs not only provide a practical approach for obtaining effective TVs but also offer a principled lens into the mechanistic foundations of ICL. 

---
# Localizing Task Recognition and Task Learning in In-Context Learning via Attention Head Analysis 

**Authors**: Haolin Yang, Hakaze Cho, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2509.24164)  

**Abstract**: We investigate the mechanistic underpinnings of in-context learning (ICL) in large language models by reconciling two dominant perspectives: the component-level analysis of attention heads and the holistic decomposition of ICL into Task Recognition (TR) and Task Learning (TL). We propose a novel framework based on Task Subspace Logit Attribution (TSLA) to identify attention heads specialized in TR and TL, and demonstrate their distinct yet complementary roles. Through correlation analysis, ablation studies, and input perturbations, we show that the identified TR and TL heads independently and effectively capture the TR and TL components of ICL. Using steering experiments with geometric analysis of hidden states, we reveal that TR heads promote task recognition by aligning hidden states with the task subspace, while TL heads rotate hidden states within the subspace toward the correct label to facilitate prediction. We further show how previous findings on ICL mechanisms, including induction heads and task vectors, can be reconciled with our attention-head-level analysis of the TR-TL decomposition. Our framework thus provides a unified and interpretable account of how large language models execute ICL across diverse tasks and settings. 

---
# Your thoughts tell who you are: Characterize the reasoning patterns of LRMs 

**Authors**: Yida Chen, Yuning Mao, Xianjun Yang, Suyu Ge, Shengjie Bi, Lijuan Liu, Saghar Hosseini, Liang Tan, Yixin Nie, Shaoliang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2509.24147)  

**Abstract**: Current comparisons of large reasoning models (LRMs) focus on macro-level statistics such as task accuracy or reasoning length. Whether different LRMs reason differently remains an open question. To address this gap, we introduce the LLM-proposed Open Taxonomy (LOT), a classification method that uses a generative language model to compare reasoning traces from two LRMs and articulate their distinctive features in words. LOT then models how these features predict the source LRM of a reasoning trace based on their empirical distributions across LRM outputs. Iterating this process over a dataset of reasoning traces yields a human-readable taxonomy that characterizes how models think. We apply LOT to compare the reasoning of 12 open-source LRMs on tasks in math, science, and coding. LOT identifies systematic differences in their thoughts, achieving 80-100% accuracy in distinguishing reasoning traces from LRMs that differ in scale, base model family, or objective domain. Beyond classification, LOT's natural-language taxonomy provides qualitative explanations of how LRMs think differently. Finally, in a case study, we link the reasoning differences to performance: aligning the reasoning style of smaller Qwen3 models with that of the largest Qwen3 during test time improves their accuracy on GPQA by 3.3-5.7%. 

---
# Beyond Magic Words: Sharpness-Aware Prompt Evolving for Robust Large Language Models with TARE 

**Authors**: Guancheng Wan, Lucheng Fu, Haoxin Liu, Yiqiao Jin, Hui Yi Leong, Eric Hanchen Jiang, Hejia Geng, Jinhe Bi, Yunpu Ma, Xiangru Tang, B. Aditya Prakash, Yizhou Sun, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24130)  

**Abstract**: The performance of Large Language Models (LLMs) hinges on carefully engineered prompts. However, prevailing prompt optimization methods, ranging from heuristic edits and reinforcement learning to evolutionary search, primarily target point-wise accuracy. They seldom enforce paraphrase invariance or searching stability, and therefore cannot remedy this brittleness in practice. Automated prompt search remains brittle: small, semantically preserving paraphrases often cause large performance swings. We identify this brittleness as the textual sharpness of the prompt landscape. In this work, we provide the first formal treatment of textual sharpness in the discrete, semantic space of prompts, together with an operational robustness criterion over a semantic neighborhood; the design is black-box or API-only, requiring no gradients to update the model's parameters. Then we introduce TARE (Textual Sharpness-Aware Evolving), a derivative-free framework that alternates between an inner, sampling-based adversarial search that stresses a prompt with hard paraphrases and an outer, robust selection that prefers candidates whose neighborhoods remain strong. We further propose ATARE, which learns anisotropic weights to shape the semantic neighborhood and adapts its radius over time to balance exploration and fidelity. Diverse tasks evaluate our methods, whose design for minimizing textual sharpness gap leads to prompts that preserve accuracy under paraphrasing, outperforming accuracy-only prompt search while remaining computationally practical. 

---
# EduVidQA: Generating and Evaluating Long-form Answers to Student Questions based on Lecture Videos 

**Authors**: Sourjyadip Ray, Shubham Sharma, Somak Aditya, Pawan Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2509.24120)  

**Abstract**: As digital platforms redefine educational paradigms, ensuring interactivity remains vital for effective learning. This paper explores using Multimodal Large Language Models (MLLMs) to automatically respond to student questions from online lectures - a novel question answering task of real world significance. We introduce the EduVidQA Dataset with 5252 question-answer pairs (both synthetic and real-world) from 296 computer science videos covering diverse topics and difficulty levels. To understand the needs of the dataset and task evaluation, we empirically study the qualitative preferences of students, which we provide as an important contribution to this line of work. Our benchmarking experiments consist of 6 state-of-the-art MLLMs, through which we study the effectiveness of our synthetic data for finetuning, as well as showing the challenging nature of the task. We evaluate the models using both text-based and qualitative metrics, thus showing a nuanced perspective of the models' performance, which is paramount to future work. This work not only sets a benchmark for this important problem, but also opens exciting avenues for future research in the field of Natural Language Processing for Education. 

---
# Dual-Scale World Models for LLM Agents Towards Hard-Exploration Problems 

**Authors**: Minsoo Kim, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24116)  

**Abstract**: LLM-based agents have seen promising advances, yet they are still limited in "hard-exploration" tasks requiring learning new knowledge through exploration. We present GLoW, a novel approach leveraging dual-scale world models, maintaining a trajectory frontier of high-value discoveries at the global scale, while learning from local trial-and-error in exploration through a Multi-path Advantage Reflection mechanism which infers advantage-based progress signals to guide exploration. To evaluate our framework for hard-exploration, we tackle the Jericho benchmark suite of text-based games, where GLoW achieves a new state-of-theart performance for LLM-based approaches. Compared to state-of-the-art RLbased methods, our approach achieves comparable performance while requiring 100-800x fewer environment interactions. 

---
# Pragmatic Inference for Moral Reasoning Acquisition: Generalization via Distributional Semantics 

**Authors**: Guangliang Liu, Xi Chen, Bocheng Chen, Xitong Zhang, Kristen Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2509.24102)  

**Abstract**: Moral reasoning has emerged as a promising research direction for Large Language Models (LLMs), yet achieving generalization remains a central challenge. From a linguistic standpoint, this difficulty arises because LLMs are adept at capturing distributional semantics, which fundamentally differs from the morals which operate at the pragmatic level. This paper investigates how LLMs can achieve generalized moral reasoning despite their reliance on distributional semantics. We propose pragmatic inference methods grounded in moral foundations theory, which leverage contextual information at each step to bridge the pragmatic gap and guide LLMs in connecting moral foundations with moral reasoning objectives. Experimental results demonstrate that our approach significantly enhances LLMs' generalization in moral reasoning, providing a foundation for future research grounded in moral foundations theory. 

---
# BTC-SAM: Leveraging LLMs for Generation of Bias Test Cases for Sentiment Analysis Models 

**Authors**: Zsolt T. Kardkovács, Lynda Djennane, Anna Field, Boualem Benatallah, Yacine Gaci, Fabio Casati, Walid Gaaloul  

**Link**: [PDF](https://arxiv.org/pdf/2509.24101)  

**Abstract**: Sentiment Analysis (SA) models harbor inherent social biases that can be harmful in real-world applications. These biases are identified by examining the output of SA models for sentences that only vary in the identity groups of the subjects. Constructing natural, linguistically rich, relevant, and diverse sets of sentences that provide sufficient coverage over the domain is expensive, especially when addressing a wide range of biases: it requires domain experts and/or crowd-sourcing. In this paper, we present a novel bias testing framework, BTC-SAM, which generates high-quality test cases for bias testing in SA models with minimal specification using Large Language Models (LLMs) for the controllable generation of test sentences. Our experiments show that relying on LLMs can provide high linguistic variation and diversity in the test sentences, thereby offering better test coverage compared to base prompting methods even for previously unseen biases. 

---
# GEAR: A General Evaluation Framework for Abductive Reasoning 

**Authors**: Kaiyu He, Peilin Wu, Mian Zhang, Kun Wan, Wentian Zhao, Xinya Du, Zhiyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24096)  

**Abstract**: Since the advent of large language models (LLMs), research has focused on instruction following and deductive reasoning. A central question remains: can these models discover new knowledge, and how can we evaluate this ability? We address this by studying abductive reasoning-the generation of plausible hypotheses to explain observations-and introduce GEAR (General Evaluation for Abductive Reasoning), a general-purpose, fully automated, transparent, and label-free evaluation paradigm. GEAR scores hypothesis sets by three metrics: consistency (each hypothesis explains the observations), generalizability (consistent hypotheses make meaningful predictions on unseen inputs), and diversity (the set covers distinct predictions and patterns). Built this way, GEAR is scalable (no human gold answers), reliable (deterministic scoring aligned with classical abduction), and open-ended (scores improve only when models produce new plausible hypotheses, unlike static benchmarks that saturate once accuracy is high). Using GEAR, we conduct a fine-grained study of nine LLMs on four abduction benchmarks with 1,500 problems, generating over 50,000 candidate hypotheses and revealing model differences obscured by gold-answer or purely human evaluations. We further propose a momentum-based curriculum that adjusts GEAR-derived training data by learning velocity: it starts with what the model learns quickly and shifts toward harder objectives such as generating diverse hypotheses once the model is confident on foundational objectives. Without gold-label supervision, this strategy improves all GEAR objectives and these gains transfer to established abductive reasoning benchmarks. Taken together, GEAR provides a principled framework that evaluates abduction and supplies label-free, scalable training signals that help LLMs produce more diverse and reliable hypotheses. 

---
# Large-Scale Constraint Generation - Can LLMs Parse Hundreds of Constraints? 

**Authors**: Matteo Boffa, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2509.24090)  

**Abstract**: Recent research has explored the constrained generation capabilities of Large Language Models (LLMs) when explicitly prompted by few task-specific requirements. In contrast, we introduce Large-Scale Constraint Generation (LSCG), a new problem that evaluates whether LLMs can parse a large, fine-grained, generic list of constraints. To examine the LLMs' ability to handle an increasing number constraints, we create a practical instance of LSCG, called Words Checker. In Words Checker, we evaluate the impact of model characteristics (e.g., size, family) and steering techniques (e.g., Simple Prompt, Chain of Thought, Best of N) on performance. We also propose FoCusNet, a small and dedicated model that parses the original list of constraints into a smaller subset, helping the LLM focus on relevant constraints. Experiments reveal that existing solutions suffer a significant performance drop as the number of constraints increases, with FoCusNet showing an 8-13% accuracy boost. 

---
# Ensembling Multilingual Transformers for Robust Sentiment Analysis of Tweets 

**Authors**: Meysam Shirdel Bilehsavar, Negin Mahmoudi, Mohammad Jalili Torkamani, Kiana Kiashemshaki  

**Link**: [PDF](https://arxiv.org/pdf/2509.24080)  

**Abstract**: Sentiment analysis is a very important natural language processing activity in which one identifies the polarity of a text, whether it conveys positive, negative, or neutral sentiment. Along with the growth of social media and the Internet, the significance of sentiment analysis has grown across numerous industries such as marketing, politics, and customer service. Sentiment analysis is flawed, however, when applied to foreign languages, particularly when there is no labelled data to train models upon. In this study, we present a transformer ensemble model and a large language model (LLM) that employs sentiment analysis of other languages. We used multi languages dataset. Sentiment was then assessed for sentences using an ensemble of pre-trained sentiment analysis models: bert-base-multilingual-uncased-sentiment, and XLM-R. Our experimental results indicated that sentiment analysis performance was more than 86% using the proposed method. 

---
# ResFormer: All-Time Reservoir Memory for Long Sequence Classification 

**Authors**: Hongbo Liu, Jia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24074)  

**Abstract**: Sequence classification is essential in NLP for understanding and categorizing language patterns in tasks like sentiment analysis, intent detection, and topic classification. Transformer-based models, despite achieving state-of-the-art performance, have inherent limitations due to quadratic time and memory complexity, restricting their input length. Although extensive efforts have aimed at reducing computational demands, processing extensive contexts remains challenging.
To overcome these limitations, we propose ResFormer, a novel neural network architecture designed to model varying context lengths efficiently through a cascaded methodology. ResFormer integrates an reservoir computing network featuring a nonlinear readout to effectively capture long-term contextual dependencies in linear time. Concurrently, short-term dependencies within sentences are modeled using a conventional Transformer architecture with fixed-length inputs.
Experiments demonstrate that ResFormer significantly outperforms baseline models of DeepSeek-Qwen and ModernBERT, delivering an accuracy improvement of up to +22.3% on the EmoryNLP dataset and consistent gains on MultiWOZ, MELD, and IEMOCAP. In addition, ResFormer exhibits reduced memory consumption, underscoring its effectiveness and efficiency in modeling extensive contextual information. 

---
# SparseD: Sparse Attention for Diffusion Language Models 

**Authors**: Zeqing Wang, Gongfan Fang, Xinyin Ma, Xingyi Yang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24014)  

**Abstract**: While diffusion language models (DLMs) offer a promising alternative to autoregressive models (ARs), existing open-source DLMs suffer from high inference latency. This bottleneck is mainly due to the attention's quadratic complexity with respect to context length in computing all query-key pairs. Intuitively, to reduce this complexity, a natural strategy is to restrict attention to sparse patterns that retain only the most relevant connections. Such approaches are well-established in ARs, where attention follows fixed and clearly defined sparse patterns. However, in DLMs, we observe distinct sparsity behaviors: (1) attention patterns vary across heads, (2) attention patterns in each head remain highly similar across denoising steps, and (3) early denoising steps are critical for generation. These findings render sparse attention methods designed for ARs largely incompatible with DLMs, as they fail to capture head-specific structures and risk degrading generation when applied in early denoising steps. To address these challenges, we propose SparseD, a novel sparse attention method for DLMs. Leveraging the observations, SparseD only requires pre-computing head-specific sparse patterns one time, and reuses them across all steps. This prevents recomputing sparse patterns at each denoising step. Meanwhile, SparseD uses full attention in the early steps, then switches to sparse attention later to maintain generation quality. Together, these establish SparseD as a practical and efficient solution for deploying DLMs in long-context applications. Experimental results demonstrate that SparseD achieves lossless acceleration, delivering up to $1.50\times$ speedup over FlashAttention at a 64k context length with 1,024 denoising steps. 

---
# Sequential Diffusion Language Models 

**Authors**: Yangzhou Liu, Yue Cao, Hao Li, Gen Luo, Zhe Chen, Weiyun Wang, Xiaobo Liang, Biqing Qi, Lijun Wu, Changyao Tian, Yanting Zhang, Yuqiang Li, Tong Lu, Yu Qiao, Jifeng Dai, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24007)  

**Abstract**: Diffusion language models (DLMs) have strong theoretical efficiency but are limited by fixed-length decoding and incompatibility with key-value (KV) caches. Block diffusion mitigates these issues, yet still enforces a fixed block size and requires expensive training. We introduce Next Sequence Prediction (NSP), which unifies next-token and next-block prediction, enabling the model to adaptively determine the generation length at each step. When the length is fixed to 1, NSP reduces to standard next-token prediction. Building on NSP, we propose Sequential Diffusion Language Model (SDLM), which can retrofit pre-trained autoregressive language models (ALMs) at minimal cost. Specifically, SDLM performs diffusion inference within fixed-size mask blocks, but dynamically decodes consecutive subsequences based on model confidence, thereby preserving KV-cache compatibility and improving robustness to varying uncertainty and semantics across the sequence. Experiments show that SDLM matches or surpasses strong autoregressive baselines using only 3.5M training samples, while achieving 2.1 higher throughput than Qwen-2.5. Notably, the SDLM-32B model delivers even more pronounced efficiency gains, demonstrating the strong scalability potential of our modeling paradigm. Project page and codes: this https URL 

---
# MCPMark: A Benchmark for Stress-Testing Realistic and Comprehensive MCP Use 

**Authors**: Zijian Wu, Xiangyan Liu, Xinyuan Zhang, Lingjun Chen, Fanqing Meng, Lingxiao Du, Yiran Zhao, Fanshi Zhang, Yaoqi Ye, Jiawei Wang, Zirui Wang, Jinjie Ni, Yufan Yang, Arvin Xu, Michael Qizhe Shieh  

**Link**: [PDF](https://arxiv.org/pdf/2509.24002)  

**Abstract**: MCP standardizes how LLMs interact with external systems, forming the foundation for general agents. However, existing MCP benchmarks remain narrow in scope: they focus on read-heavy tasks or tasks with limited interaction depth, and fail to capture the complexity and realism of real-world workflows. To address this gap, we propose MCPMark, a benchmark designed to evaluate MCP use in a more realistic and comprehensive manner. It consists of $127$ high-quality tasks collaboratively created by domain experts and AI agents. Each task begins with a curated initial state and includes a programmatic script for automatic verification. These tasks demand richer and more diverse interactions with the environment, involving a broad range of create, read, update, and delete (CRUD) operations. We conduct a comprehensive evaluation of cutting-edge LLMs using a minimal agent framework that operates in a tool-calling loop. Empirical results show that the best-performing model, gpt-5-medium, reaches only $52.56$\% pass@1 and $33.86$\% pass^4, while other widely regarded strong models, including claude-sonnet-4 and o3, fall below $30$\% pass@1 and $15$\% pass^4. On average, LLMs require $16.2$ execution turns and $17.4$ tool calls per task, significantly surpassing those in previous MCP benchmarks and highlighting the stress-testing nature of MCPMark. 

---
# The AI Agent Code of Conduct: Automated Guardrail Policy-as-Prompt Synthesis 

**Authors**: Gauri Kholkar, Ratinder Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2509.23994)  

**Abstract**: As autonomous AI agents are increasingly deployed in industry, it is essential to safeguard them. We introduce a novel framework that automates the translation of unstructured design documents into verifiable, real-time guardrails. We introduce "Policy as Prompt," a new approach that uses Large Language Models (LLMs) to interpret and enforce natural language policies by applying contextual understanding and the principle of least privilege. Our system first ingests technical artifacts to construct a verifiable policy tree, which is then compiled into lightweight, prompt-based classifiers that audit agent behavior at runtime. We validate our approach across diverse applications, demonstrating a scalable and auditable pipeline that bridges the critical policy-to-practice gap, paving the way for verifiably safer and more regulatable AI. 

---
# The Hidden Costs of Translation Accuracy: Distillation, Quantization, and Environmental Impact 

**Authors**: Dhaathri Vijay, Anandaswarup Vadapalli  

**Link**: [PDF](https://arxiv.org/pdf/2509.23990)  

**Abstract**: The rapid expansion of large language models (LLMs) has heightened concerns about their computational and environmental costs. This study investigates the trade-offs between translation quality and efficiency by comparing full-scale, distilled, and quantized models using machine translation as a case study. We evaluated performance on the Flores+ benchmark and through human judgments of conversational translations in French, Hindi, and Kannada. Our analysis of carbon emissions per evaluation run revealed that the full 3.3B fp32 model, while achieving the highest BLEU scores, incurred the largest environmental footprint (about 0.007-0.008 kg CO2 per run). The distilled models achieved an inference of up to 4.5x faster than the full 3.3B model, with only minimal reductions in BLEU scores. Human evaluations also showed that even aggressive quantization (INT4) preserved high levels of accuracy and fluency, with differences between models generally minor. These findings demonstrate that model compression strategies can substantially reduce computational demands and environmental impact while maintaining competitive translation quality, though trade-offs are more pronounced in low-resource settings. We argue for evaluation frameworks that integrate efficiency and sustainability alongside objective metrics as central dimensions of progress in NLP. 

---
# Toward Preference-aligned Large Language Models via Residual-based Model Steering 

**Authors**: Lucio La Cava, Andrea Tagarelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.23982)  

**Abstract**: Preference alignment is a critical step in making Large Language Models (LLMs) useful and aligned with (human) preferences. Existing approaches such as Reinforcement Learning from Human Feedback or Direct Preference Optimization typically require curated data and expensive optimization over billions of parameters, and eventually lead to persistent task-specific models. In this work, we introduce Preference alignment of Large Language Models via Residual Steering (PaLRS), a training-free method that exploits preference signals encoded in the residual streams of LLMs. From as few as one hundred preference pairs, PaLRS extracts lightweight, plug-and-play steering vectors that can be applied at inference time to push models toward preferred behaviors. We evaluate PaLRS on various small-to-medium-scale open-source LLMs, showing that PaLRS-aligned models achieve consistent gains on mathematical reasoning and code generation benchmarks while preserving baseline general-purpose performance. Moreover, when compared to DPO-aligned models, they perform better with huge time savings. Our findings highlight that PaLRS offers an effective, much more efficient and flexible alternative to standard preference optimization pipelines, offering a training-free, plug-and-play mechanism for alignment with minimal data. 

---
# ByteSized32Refactored: Towards an Extensible Interactive Text Games Corpus for LLM World Modeling and Evaluation 

**Authors**: Haonan Wang, Junfeng Sun, Xingdi Yuan, Ruoyao Wang, Ziang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23979)  

**Abstract**: Simulating interactive world models remains a core challenge in Large Language Models(LLMs). In this work, we introduce the ByteSized32Refactored, a refactored, modular, and extensible implementation of the original ByteSized32 corpus to explore the task of text game generation. We further optimize the code structure of each text game and create the this http URL foundation library, which centralizes common logic across all 32 games by abstracting 7 base classes (GameObject, etc.) into reusable modules, thereby reducing from 20k to 10k total lines of Python code compared to the original Bytesized32. Our refactored implementation enables extendability - with our centralized design, ByteSized32Refactored can be more efficiently extended to include text games of new scenarios and specifications by reusing the shared logic and functionalities. Extensive experiments with GPT-4o demonstrate a mix of performance - with Bytesized32Refactored, the generated text games for unseen scenarios showcase quality improvements on two of the four evaluation dimensions while decreases on the other two, indicating that the hierarchical structure of the refactored code presents new challenges for LLMs. Overall, we highlight that our extensible code structure, centered on the foundation library and the modular optimization, not only facilitates LLM adaptation to environment specifications but also establishes a scalable environment that supports future extensions. 

---
# HiPO: Hybrid Policy Optimization for Dynamic Reasoning in LLMs 

**Authors**: Ken Deng, Zizheng Zhan, Wen Xiang, Wenqiang Zhu, Tianhao Peng, Xinping Lei, Weihao Li, Jingxuan Xu, Kun Wu, Yifan Yao, Haoyang Huang, Huaixi Tang, Kepeng Lei, Zhiyi Lai, Songwei Yu, Zongxian Feng, Zuchen Gao, Weihao Xie, Chenchen Zhang, Yanan Wu, Yuanxing Zhang, Lecheng Huang, Yuqun Zhang, Jie Liu, Zhaoxiang Zhang, Haotian Zhang, Bin Chen, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23967)  

**Abstract**: Large Language Models (LLMs) increasingly rely on chain-of-thought (CoT) reasoning to improve accuracy on complex tasks. However, always generating lengthy reasoning traces is inefficient, leading to excessive token usage and higher inference costs. This paper introduces the Hybrid Policy Optimization (i.e., HiPO), a framework for adaptive reasoning control that enables LLMs to selectively decide when to engage in detailed reasoning (Think-on) and when to respond directly (Think-off). Specifically, HiPO combines a hybrid data pipelineproviding paired Think-on and Think-off responseswith a hybrid reinforcement learning reward system that balances accuracy and efficiency while avoiding over-reliance on detailed reasoning. Experiments across mathematics and coding benchmarks demonstrate that HiPO can substantially reduce token length while maintaining or improving accuracy. Finally, we hope HiPO a can be a principled approach for efficient adaptive reasoning, advancing the deployment of reasoning-oriented LLMs in real-world, resource-sensitive settings. 

---
# Vision-Grounded Machine Interpreting: Improving the Translation Process through Visual Cues 

**Authors**: Claudio Fantinuoli  

**Link**: [PDF](https://arxiv.org/pdf/2509.23957)  

**Abstract**: Machine Interpreting systems are currently implemented as unimodal, real-time speech-to-speech architectures, processing translation exclusively on the basis of the linguistic signal. Such reliance on a single modality, however, constrains performance in contexts where disambiguation and adequacy depend on additional cues, such as visual, situational, or pragmatic information. This paper introduces Vision-Grounded Interpreting (VGI), a novel approach designed to address the limitations of unimodal machine interpreting. We present a prototype system that integrates a vision-language model to process both speech and visual input from a webcam, with the aim of priming the translation process through contextual visual information. To evaluate the effectiveness of this approach, we constructed a hand-crafted diagnostic corpus targeting three types of ambiguity. In our evaluation, visual grounding substantially improves lexical disambiguation, yields modest and less stable gains for gender resolution, and shows no benefit for syntactic ambiguities. We argue that embracing multimodality represents a necessary step forward for advancing translation quality in machine interpreting. 

---
# Easy Turn: Integrating Acoustic and Linguistic Modalities for Robust Turn-Taking in Full-Duplex Spoken Dialogue Systems 

**Authors**: Guojian Li, Chengyou Wang, Hongfei Xue, Shuiyuan Wang, Dehui Gao, Zihan Zhang, Yuke Lin, Wenjie Li, Longshuai Xiao, Zhonghua Fu, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.23938)  

**Abstract**: Full-duplex interaction is crucial for natural human-machine communication, yet remains challenging as it requires robust turn-taking detection to decide when the system should speak, listen, or remain silent. Existing solutions either rely on dedicated turn-taking models, most of which are not open-sourced. The few available ones are limited by their large parameter size or by supporting only a single modality, such as acoustic or linguistic. Alternatively, some approaches finetune LLM backbones to enable full-duplex capability, but this requires large amounts of full-duplex data, which remain scarce in open-source form. To address these issues, we propose Easy Turn, an open-source, modular turn-taking detection model that integrates acoustic and linguistic bimodal information to predict four dialogue turn states: complete, incomplete, backchannel, and wait, accompanied by the release of Easy Turn trainset, a 1,145-hour speech dataset designed for training turn-taking detection models. Compared to existing open-source models like TEN Turn Detection and Smart Turn V2, our model achieves state-of-the-art turn-taking detection accuracy on our open-source Easy Turn testset. The data and model will be made publicly available on GitHub. 

---
# Assessing Large Language Models in Updating Their Forecasts with New Information 

**Authors**: Zhangdie Yuan, Zifeng Ding, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2509.23936)  

**Abstract**: Prior work has largely treated future event prediction as a static task, failing to consider how forecasts and the confidence in them should evolve as new evidence emerges. To address this gap, we introduce EVOLVECAST, a framework for evaluating whether large language models appropriately revise their predictions in response to new information. In particular, EVOLVECAST assesses whether LLMs adjust their forecasts when presented with information released after their training cutoff. We use human forecasters as a comparative reference to analyze prediction shifts and confidence calibration under updated contexts. While LLMs demonstrate some responsiveness to new information, their updates are often inconsistent or overly conservative. We further find that neither verbalized nor logits-based confidence estimates consistently outperform the other, and both remain far from the human reference standard. Across settings, models tend to express conservative bias, underscoring the need for more robust approaches to belief updating. 

---
# Taming Masked Diffusion Language Models via Consistency Trajectory Reinforcement Learning with Fewer Decoding Step 

**Authors**: Jingyi Yang, Guanxu Chen, Xuhao Hu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23924)  

**Abstract**: Masked diffusion language models (MDLMs) have recently emerged as a promising alternative to autoregressive (AR) language models, offering properties such as parallel decoding, flexible generation orders, and the potential for fewer inference steps. Despite these advantages, decoding strategies and reinforcement learning (RL) algorithms tailored for MDLMs remain underexplored. A naive approach is to directly transfer techniques well-established for AR models to MDLMs. However, this raises an immediate question: Is such a naive transfer truly optimal? For example, 1) Block-wise and semi-AR decoding strategies are not employed during the training of MDLMs, so why do they outperform full diffusion-style decoding during inference? 2) Applying RL algorithms designed for AR models directly to MDLMs exhibits a training-inference inconsistency, since MDLM decoding are non-causal (parallel). This results in inconsistencies between the rollout trajectory and the optimization trajectory. To address these challenges, we propose EOS Early Rejection (EOSER) and Ascending Step-Size (ASS) decoding scheduler, which unlock the potential of MDLMs to perform full diffusion-style decoding, achieving competitive performance with fewer decoding steps. Additionally, we introduce Consistency Trajectory Group Relative Policy Optimization (CJ-GRPO) for taming MDLMs, which emphasizes the consistency between rollout trajectory and optimization trajectory, and reduces the optimization errors caused by skip-step optimization. We conduct extensive experiments on reasoning tasks, such as mathematical and planning benchmarks, using LLaDA-8B-Instruct. The results demonstrate that the proposed EOSER and ASS mechanisms, together with CJ-GRPO, hold significant promise for effectively and efficiently taming MDLMs. Code: this https URL. 

---
# DocPruner: A Storage-Efficient Framework for Multi-Vector Visual Document Retrieval via Adaptive Patch-Level Embedding Pruning 

**Authors**: Yibo Yan, Guangwei Xu, Xin Zou, Shuliang Liu, James Kwok, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23883)  

**Abstract**: Visual Document Retrieval (VDR), the task of retrieving visually-rich document pages using queries that combine visual and textual cues, is crucial for numerous real-world applications. Recent state-of-the-art methods leverage Large Vision-Language Models (LVLMs) in a multi-vector paradigm, representing each document as patch-level embeddings to capture fine-grained details. While highly effective, this approach introduces a critical challenge: prohibitive storage overhead, as storing hundreds of vectors per page makes large-scale deployment costly and impractical. To address this, we introduce DocPruner, the first framework to employ adaptive patch-level embedding pruning for VDR to effectively reduce the storage overhead. DocPruner leverages the intra-document patch attention distribution to dynamically identify and discard redundant embeddings for each document. This adaptive mechanism enables a significant 50-60% reduction in storage for leading multi-vector VDR models with negligible degradation in document retrieval performance. Extensive experiments across more than ten representative datasets validate that DocPruner offers a robust, flexible, and effective solution for building storage-efficient, large-scale VDR systems. 

---
# Winning the Pruning Gamble: A Unified Approach to Joint Sample and Token Pruning for Efficient Supervised Fine-Tuning 

**Authors**: Shaobo Wang, Jiaming Wang, Jiajun Zhang, Cong Wang, Yue Min, Zichen Wen, Fei Huang, Huiqiang Jiang, Junyang Lin, Dayiheng Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23873)  

**Abstract**: As supervised fine-tuning (SFT) evolves from a lightweight post-training step into a compute-intensive phase rivaling mid-training in scale, data efficiency has become critical for aligning large language models (LLMs) under tight budgets. Existing data pruning methods suffer from a fragmented design: they operate either at the sample level or the token level in isolation, failing to jointly optimize both dimensions. This disconnect leads to significant inefficiencies--high-value samples may still contain redundant tokens, while token-level pruning often discards crucial instructional or corrective signals embedded in individual examples. To address this bottleneck, we introduce the Error-Uncertainty (EU) Plane, a diagnostic framework that jointly characterizes the heterogeneous utility of training data across samples and tokens. Guided by this insight, we propose Quadrant-based Tuning (Q-Tuning), a unified framework that strategically coordinates sample pruning and token pruning. Q-Tuning employs a two-stage strategy: first, it performs sample-level triage to retain examples rich in informative misconceptions or calibration signals; second, it applies an asymmetric token-pruning policy, using a context-aware scoring mechanism to trim less salient tokens exclusively from misconception samples while preserving calibration samples in their entirety. Our method sets a new state of the art across five diverse benchmarks. Remarkably, on SmolLM2-1.7B, Q-Tuning achieves a +38\% average improvement over the full-data SFT baseline using only 12.5\% of the original training data. As the first dynamic pruning approach to consistently outperform full-data training, Q-Tuning provides a practical and scalable blueprint for maximizing data utilization in budget-constrained LLM SFT. 

---
# SPELL: Self-Play Reinforcement Learning for evolving Long-Context Language Models 

**Authors**: Ziyi Yang, Weizhou Shen, Ruijun Chen, Chenliang Li, Fanqi Wan, Ming Yan, Xiaojun Quan, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23863)  

**Abstract**: Progress in long-context reasoning for large language models (LLMs) has lagged behind other recent advances. This gap arises not only from the intrinsic difficulty of processing long texts, but also from the scarcity of reliable human annotations and programmatically verifiable reward signals. In this paper, we propose SPELL, a multi-role self-play reinforcement learning framework that enables scalable, label-free optimization for long-context reasoning. SPELL integrates three cyclical roles-questioner, responder, and verifier-within a single model to enable continual self-improvement. The questioner generates questions from raw documents paired with reference answers; the responder learns to solve these questions based on the documents; and the verifier evaluates semantic equivalence between the responder's output and the questioner's reference answer, producing reward signals to guide continual training. To stabilize training, we introduce an automated curriculum that gradually increases document length and a reward function that adapts question difficulty to the model's evolving capabilities. Extensive experiments on six long-context benchmarks show that SPELL consistently improves performance across diverse LLMs and outperforms equally sized models fine-tuned on large-scale annotated data. Notably, SPELL achieves an average 7.6-point gain in pass@8 on the strong reasoning model Qwen3-30B-A3B-Thinking, raising its performance ceiling and showing promise for scaling to even more capable models. 

---
# Open-DeBias: Toward Mitigating Open-Set Bias in Language Models 

**Authors**: Arti Rani, Shweta Singh, Nihar Ranjan Sahoo, Gaurav Kumar Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2509.23805)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success on question answering (QA) tasks, yet they often encode harmful biases that compromise fairness and trustworthiness. Most existing bias mitigation approaches are restricted to predefined categories, limiting their ability to address novel or context-specific emergent biases. To bridge this gap, we tackle the novel problem of open-set bias detection and mitigation in text-based QA. We introduce OpenBiasBench, a comprehensive benchmark designed to evaluate biases across a wide range of categories and subgroups, encompassing both known and previously unseen biases. Additionally, we propose Open-DeBias, a novel, data-efficient, and parameter-efficient debiasing method that leverages adapter modules to mitigate existing social and stereotypical biases while generalizing to unseen ones. Compared to the state-of-the-art BMBI method, Open-DeBias improves QA accuracy on BBQ dataset by nearly $48\%$ on ambiguous subsets and $6\%$ on disambiguated ones, using adapters fine-tuned on just a small fraction of the training data. Remarkably, the same adapters, in a zero-shot transfer to Korean BBQ, achieve $84\%$ accuracy, demonstrating robust language-agnostic generalization. Through extensive evaluation, we also validate the effectiveness of Open-DeBias across a broad range of NLP tasks, including StereoSet and CrowS-Pairs, highlighting its robustness, multilingual strength, and suitability for general-purpose, open-domain bias mitigation. The project page is available at: this https URL 

---
# Transformer Tafsir at QIAS 2025 Shared Task: Hybrid Retrieval-Augmented Generation for Islamic Knowledge Question Answering 

**Authors**: Muhammad Abu Ahmad, Mohamad Ballout, Raia Abu Ahmad, Elia Bruni  

**Link**: [PDF](https://arxiv.org/pdf/2509.23793)  

**Abstract**: This paper presents our submission to the QIAS 2025 shared task on Islamic knowledge understanding and reasoning. We developed a hybrid retrieval-augmented generation (RAG) system that combines sparse and dense retrieval methods with cross-encoder reranking to improve large language model (LLM) performance. Our three-stage pipeline incorporates BM25 for initial retrieval, a dense embedding retrieval model for semantic matching, and cross-encoder reranking for precise content retrieval. We evaluate our approach on both subtasks using two LLMs, Fanar and Mistral, demonstrating that the proposed RAG pipeline enhances performance across both, with accuracy improvements up to 25%, depending on the task and model configuration. Our best configuration is achieved with Fanar, yielding accuracy scores of 45% in Subtask 1 and 80% in Subtask 2. 

---
# Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions 

**Authors**: Yoonah Park, Haesung Pyun, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.23782)  

**Abstract**: Large Language Models (LLMs) often fail on multiple-choice questions (MCQs) despite demonstrating correct knowledge in other contexts, such as free-form generation. To investigate the mechanism underlying this knowledge-prediction gap on MCQs and alleviate it, we conduct a probing analysis and find that residual streams in certain layers contain a subspace spanned by two important bases: a \emph{knowledge basis} that encodes the probability of the ground-truth answer for a given MCQ and a \emph{prediction basis} that encodes the probability of the answer choice predicted by the model. We observe that incorrect predictions arise from a misalignment of the model's hidden states along these two bases. Hence, we introduce \textbf{KAPPA} (Knowledge-Aligned Prediction through Projection-based Adjustment), a parameter-free intervention that transforms the hidden states to align the prediction coordinate with the knowledge coordinate within this subspace. Experiments on binary-choice reformulations of Big-Bench-Hard and ARC-Challenge show that KAPPA substantially improves accuracy and consistently outperforms baselines. While optimal subspaces differ across tasks, subspaces generalize to some extent, as supported by cross-dataset experiments. Moreover, KAPPA extends its effectiveness to free-form questions beyond MCQs. Our work provides a new geometric understanding of the knowledge-prediction gap and offers a practical method for better aligning model behavior with its latent knowledge. 

---
# From Personal to Collective: On the Role of Local and Global Memory in LLM Personalization 

**Authors**: Zehong Wang, Junlin Wu, ZHaoxuan Tan, Bolian Li, Xianrui Zhong, Zheli Liu, Qingkai Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23767)  

**Abstract**: Large language model (LLM) personalization aims to tailor model behavior to individual users based on their historical interactions. However, its effectiveness is often hindered by two key challenges: the \textit{cold-start problem}, where users with limited history provide insufficient context for accurate personalization, and the \textit{biasing problem}, where users with abundant but skewed history cause the model to overfit to narrow preferences. We identify both issues as symptoms of a common underlying limitation, i.e., the inability to model collective knowledge across users. To address this, we propose a local-global memory framework (LoGo) that combines the personalized local memory with a collective global memory that captures shared interests across the population. To reconcile discrepancies between these two memory sources, we introduce a mediator module designed to resolve conflicts between local and global signals. Extensive experiments on multiple benchmarks demonstrate that LoGo consistently improves personalization quality by both warming up cold-start users and mitigating biased predictions. These results highlight the importance of incorporating collective knowledge to enhance LLM personalization. 

---
# Knowledge-Level Consistency Reinforcement Learning: Dual-Fact Alignment for Long-Form Factuality 

**Authors**: Junliang Li, Yucheng Wang, Yan Chen, Yu Ran, Ruiqing Zhang, Jing Liu, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23765)  

**Abstract**: Hallucination and factuality deficits remain key obstacles to the reliability of large language models (LLMs) in long-form generation. Existing reinforcement learning from human feedback (RLHF) frameworks primarily rely on preference rewards, yet they often overlook the model's internal knowledge boundaries, exacerbating the so-called "hallucination tax". To address this challenge, we propose Knowledge-Level Consistency Reinforcement Learning Framework (KLCF), a novel framework that focuses on the knowledge consistency between the policy model's expressed knowledge and the base model's parametric knowledge, and introduces a Dual-Fact Alignment mechanism to jointly optimize factual recall and precision. Specifically, KLCF leverages pretrained knowledge boundaries to construct fact checklist, guiding online reinforcement learning to improve factual coverage and recall; simultaneously, it trains a self-assessment module based on the base model's internal knowledge to enhance factual precision during generation. Unlike prior methods that rely on external retrieval or heavy verification, our reward design is fully external-knowledge-free and lightweight, making KLCF efficient and easily scalable to large-scale training. Experimental results demonstrate that KLCF substantially improves factuality metrics across multiple long-form benchmarks and effectively alleviates model hallucinations. 

---
# Understanding Textual Capability Degradation in Speech LLMs via Parameter Importance Analysis 

**Authors**: Chao Wang, Rui-Chen Zheng, Yang Ai, Zhen-Hua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.23755)  

**Abstract**: The integration of speech into Large Language Models (LLMs) has substantially expanded their capabilities, but often at the cost of weakening their core textual competence. This degradation limits the ability of speech-enabled LLMs to fully exploit their pre-trained text-based knowledge. In this work, we analyze the underlying mechanisms of this issue through a focused study of the widely used encoder-adaptor paradigm. We propose an analytical framework based on parameter importance estimation, which reveals that fine-tuning for speech introduces a textual importance distribution shift: the layer-wise allocation of parameters critical to textual reasoning is disrupted. Building on this insight, we investigate two mitigation strategies: layer-wise learning rate scheduling and Low-Rank Adaptation (LoRA), both aim to preserve the original parameter distribution. Experimental results show that both approaches better maintain textual competence than full fine-tuning, while also improving downstream spoken question answering performance. Furthermore, our analysis offers a principled explanation for the effectiveness of the proposed mitigation strategies, linking their benefits to the structural properties of textual knowledge in LLMs. 

---
# Compose and Fuse: Revisiting the Foundational Bottlenecks in Multimodal Reasoning 

**Authors**: Yucheng Wang, Yifan Hou, Aydin Javadov, Mubashara Akhtar, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23744)  

**Abstract**: Multimodal large language models (MLLMs) promise enhanced reasoning by integrating diverse inputs such as text, vision, and audio. Yet cross-modal reasoning remains underexplored, with conflicting reports on whether added modalities help or harm performance. These inconsistencies stem from a lack of controlled evaluation frameworks and analysis of models' internals to isolate when and why modality interactions support or undermine reasoning. We address this gap through a logic-grounded evaluation framework that categorizes multimodal reasoning into six interaction patterns, varying how facts are distributed across modalities and logically combined. Empirically, additional modalities enhance reasoning only when they provide independent and sufficient reasoning paths, while redundant or chained entailment support often hurts performance. Moreover, reasoning degrades in three systematic ways: weaker modalities drag down overall performance, conflicts bias preference toward certain modalities, and joint signals from different modalities fail to be integrated effectively. Therefore, we identify two core failures: task-composition bottleneck, where recognition and reasoning cannot be jointly executed in one pass, and fusion bottleneck, where early integration introduces bias. For further investigation, we find that attention patterns fail to encode fact usefulness, but a simple two-step prompting (recognize then reason) restores performance, confirming the task-composition bottleneck. Moreover, modality identity remains recoverable in early layers, and softening attention in early fusion improves reasoning, highlighting biased fusion as another failure mode. Overall, our findings show that integration, not perception, is the main barrier to multimodal reasoning, suggesting composition-aware training and early fusion control as promising directions. 

---
# Do LLMs Understand Romanian Driving Laws? A Study on Multimodal and Fine-Tuned Question Answering 

**Authors**: Eduard Barbu, Adrian Marius Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2509.23715)  

**Abstract**: Ensuring that both new and experienced drivers master current traffic rules is critical to road safety. This paper evaluates Large Language Models (LLMs) on Romanian driving-law QA with explanation generation. We release a 1{,}208-question dataset (387 multimodal) and compare text-only and multimodal SOTA systems, then measure the impact of domain-specific fine-tuning for Llama 3.1-8B-Instruct and RoLlama 3.1-8B-Instruct. SOTA models perform well, but fine-tuned 8B models are competitive. Textual descriptions of images outperform direct visual input. Finally, an LLM-as-a-Judge assesses explanation quality, revealing self-preference bias. The study informs explainable QA for less-resourced languages. 

---
# Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion 

**Authors**: Zhiqiang Liu, Yichi Zhang, Mengshu Sun, Lei Liang, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23714)  

**Abstract**: Multi-modal knowledge graph completion (MMKGC) aims to discover missing facts in multi-modal knowledge graphs (MMKGs) by leveraging both structural relationships and diverse modality information of entities. Existing MMKGC methods follow two multi-modal paradigms: fusion-based and ensemble-based. Fusion-based methods employ fixed fusion strategies, which inevitably leads to the loss of modality-specific information and a lack of flexibility to adapt to varying modality relevance across contexts. In contrast, ensemble-based methods retain modality independence through dedicated sub-models but struggle to capture the nuanced, context-dependent semantic interplay between modalities. To overcome these dual limitations, we propose a novel MMKGC method M-Hyper, which achieves the coexistence and collaboration of fused and independent modality representations. Our method integrates the strengths of both paradigms, enabling effective cross-modal interactions while maintaining modality-specific information. Inspired by ``quaternion'' algebra, we utilize its four orthogonal bases to represent multiple independent modalities and employ the Hamilton product to efficiently model pair-wise interactions among them. Specifically, we introduce a Fine-grained Entity Representation Factorization (FERF) module and a Robust Relation-aware Modality Fusion (R2MF) module to obtain robust representations for three independent modalities and one fused modality. The resulting four modality representations are then mapped to the four orthogonal bases of a biquaternion (a hypercomplex extension of quaternion) for comprehensive modality interaction. Extensive experiments indicate its state-of-the-art performance, robustness, and computational efficiency. 

---
# VIVA+: Human-Centered Situational Decision-Making 

**Authors**: Zhe Hu, Yixiao Ren, Guanzhong Liu, Jing Li, Yu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.23698)  

**Abstract**: Multimodal Large Language Models (MLLMs) show promising results for embodied agents in operating meaningfully in complex, human-centered environments. Yet, evaluating their capacity for nuanced, human-like reasoning and decision-making remains challenging. In this work, we introduce VIVA+, a cognitively grounded benchmark for evaluating the reasoning and decision-making of MLLMs in human-centered situations. VIVA+ consists of 1,317 real-world situations paired with 6,373 multiple-choice questions, targeting three core abilities for decision-making: (1) Foundational Situation Comprehension, (2) Context-Driven Action Justification, and (3) Reflective Reasoning. Together, these dimensions provide a systematic framework for assessing a model's ability to perceive, reason, and act in socially meaningful ways. We evaluate the latest commercial and open-source models on VIVA+, where we reveal distinct performance patterns and highlight significant challenges. We further explore targeted training and multi-step reasoning strategies, which yield consistent performance improvements. Finally, our in-depth analysis highlights current model limitations and provides actionable insights for advancing MLLMs toward more robust, context-aware, and socially adept decision-making in real-world settings. 

---
# TF-Bench: Evaluating Program Semantics Reasoning with Type Inference in System F 

**Authors**: Yifeng He, Luning Yang, Christopher Castro Gaw Gonzalo, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23686)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into the software engineering ecosystem. Their test-time compute (TTC) reasoning capabilities show significant potential for understanding program logic and semantics beyond mere token recognition. However, current benchmarks for code reasoning lack a formal, program-centric deductive framework to ensure sound evaluation, and are incapable of assessing whether models genuinely reason about program semantics or merely exploit superficial associations between natural language and code tokens. To bridge this gap, we introduce TF-Bench, a benchmark designed to evaluate LLM reasoning based on type inference in System F, a task we refer to as program semantics reasoning. By employing verified transformations to remove semantically irrelevant natural language, we construct TF-Bench_pure, a purely semantics-driven variant of TF-Bench. Our analysis reveals substantial limitations in state-of-the-art LLMs, with the best-performing LLM (Claude-3.7-sonnet) achieving only 55.85% accuracy on TF-Bench_pure. Additionally, we propose two novel metrics to assess robustness and the effectiveness of test-time reasoning, underscoring critical limitations in current LLM capabilities and highlighting essential directions for future research. 

---
# Aligning LLMs for Multilingual Consistency in Enterprise Applications 

**Authors**: Amit Agarwal, Hansa Meghwani, Hitesh Laxmichand Patel, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2509.23659)  

**Abstract**: Large language models (LLMs) remain unreliable for global enterprise applications due to substantial performance gaps between high-resource and mid/low-resource languages, driven by English-centric pretraining and internal reasoning biases. This inconsistency undermines customer experience and operational reliability in multilingual settings such as customer support, content moderation, and information retrieval. Even with advanced Retrieval-Augmented Generation (RAG) systems, we observe up to an 29% accuracy drop in non-English languages compared to English.
We propose a practical, batch-wise alignment strategy for fine-tuning LLMs, leveraging semantically equivalent multilingual data in each training batch to directly align model outputs across languages. This approach improves non-English accuracy by up to 23.9\% without compromising English performance, model reasoning, or retrieval quality. Our method is simple to implement, scalable, and integrates seamlessly with existing LLM training \& deployment pipelines, enabling more robust and equitable multilingual AI solutions in industry. 

---
# Beyond English-Centric Training: How Reinforcement Learning Improves Cross-Lingual Reasoning in LLMs 

**Authors**: Shulin Huang, Yiran Ding, Junshu Pan, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23657)  

**Abstract**: Enhancing the complex reasoning capabilities of Large Language Models (LLMs) attracts widespread attention. While reinforcement learning (RL) has shown superior performance for improving complex reasoning, its impact on cross-lingual generalization compared to Supervised Fine-Tuning (SFT) remains unexplored. We present the first systematic investigation into cross-lingual reasoning generalization of RL and SFT. Using Qwen2.5-3B-Base as our foundation model, we conduct experiments on diverse multilingual reasoning benchmarks, including math reasoning, commonsense reasoning, and scientific reasoning. Our investigation yields two significant findings: (1) Tuning with RL not only achieves higher accuracy but also demonstrates substantially stronger cross-lingual generalization capabilities compared to SFT. (2) RL training on non-English data yields better overall performance and generalization than training on English data, which is not observed with SFT. Furthermore, through comprehensive mechanistic analyses, we explore the underlying factors of RL's superiority and generalization across languages. Our results provide compelling evidence that RL enables the model with more robust reasoning strategies, offering crucial guidance for more equitable and effective multilingual reasoning. 

---
# Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models 

**Authors**: Zemin Huang, Yuhang Wang, Zhiyang Chen, Guo-Jun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23653)  

**Abstract**: Mask-based Diffusion Language Models (DLMs) struggle to revise incorrect tokens: once a token is generated, it typically remains fixed. The key challenge is to identify potential errors in the inputs. In this paper, we propose \emph{\underline{Rem}asking-\underline{e}nabled \underline{Di}ffusion Language Model (RemeDi}, a mask-based DLM that introduces \emph{remasking} as another fundamental mechanism, enabling more flexible text refinement in diffusion-based text generation. To achieve this, RemeDi jointly predicts token distributions and per-token confidence scores at each step. The confidence scores determine which tokens to be unmasked after the current step, allowing the model to identify tokens with low quality and remask them. These remasked tokens can be resampled with richer context in subsequent steps. We design a remask-aware pipeline to train this ability, including supervised fine-tuning which teaches the model to detect and remask incorrect tokens in addition to predict mask tokens, and reinforcement learning which optimizes full generation trajectories toward higher rewards. Experiments show that RemeDi achieves the state-of-the-art results among open-source DLMs on multiple datasets. 

---
# Fast Thinking for Large Language Models 

**Authors**: Haoyu Zheng, Zhuonan Wang, Yuqian Yuan, Tianwei Lin, Wenqiao Zhang, Zheqi Lv, Juncheng Li, Siliang Tang, Yueting Zhuang, Hongyang He  

**Link**: [PDF](https://arxiv.org/pdf/2509.23633)  

**Abstract**: Reasoning-oriented Large Language Models (LLMs) often rely on generating explicit tokens step by step, and their effectiveness typically hinges on large-scale supervised fine-tuning or reinforcement learning. While Chain-of-Thought (CoT) techniques substantially enhance performance on complex reasoning tasks, they remain inefficient, requiring long reasoning traces that increase latency and token usage. In this work, we introduce Latent Codebooks for Fast Thinking, a framework that uses concise CoT sketches only during training to learn a codebook of discrete strategy priors. At inference, the model conditions on a handful of continuous thinking vectors distilled from the codebook in a single pass, enabling strategy-level guidance without producing explicit reasoning tokens. To complement this design, we propose GainRouter, a lightweight routing mechanism that adaptively switches between fast codebook guided inference and slow explicit reasoning, thereby suppressing overthinking and reducing unnecessary token generation. Experiments across multiple reasoning benchmarks show that our approach achieves competitive or superior accuracy while substantially lowering inference cost, offering a practical path toward efficient and controllable reasoning in large language models. 

---
# Timber: Training-free Instruct Model Refining with Base via Effective Rank 

**Authors**: Taiqiang Wu, Runming Yang, Tao Liu, Jiahao Wang, Zenan Xu, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2509.23595)  

**Abstract**: Post-training, which elicits a pretrained Base model into the corresponding Instruct model, is widely considered to be superficial. In this work, we first reinforce this hypothesis by providing novel quantitative evidence from the weight level that the effective rank (eRank) remains negligibly changed. However, this superficiality also suffers a critical trade-off, improving the exploitation capabilities at the cost of limiting its exploration. To tackle this issue, we propose Timber, a simple yet effective training-free method that enhances the exploration capability of the Instruct model while preserving its exploitation. The key insight is to partially revert Instruct towards the paired Base model by subtle yet targeted refinement of the weight deltas. Extensive experiments on Llama and Qwen series demonstrate that Timber consistently improves vanilla Instruct models, particularly on Pass@k performance. Our findings offer new insights into the post-training stage at the weight level and practical strategies to refine the Instruct model without training. 

---
# LLM Hallucination Detection: HSAD 

**Authors**: JinXin Li, Gang Tu, JunJie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23580)  

**Abstract**: Although Large Language Models have demonstrated powerful capabilities in a wide range of tasks such as language understanding and code generation, the frequent occurrence of hallucinations during the generation process has become a significant impediment to their deployment in critical application scenarios. Current mainstream hallucination detection methods rely on factual consistency verification or static hidden layer features. The former is constrained by the scope of knowledge coverage, while the latter struggles to capture reasoning biases during the inference process. To address these issues, and inspired by signal analysis methods in cognitive neuroscience, this paper proposes a hallucination detection method based on the frequency-domain analysis of hidden layer temporal signals, named HSAD (\textbf{H}idden \textbf{S}ignal \textbf{A}nalysis-based \textbf{D}etection). First, by treating the LLM's reasoning process as a cognitive journey that unfolds over time, we propose modeling and simulating the human process of signal perception and discrimination in a deception-detection scenario through hidden layer temporal signals. Next, The Fast Fourier Transform is applied to map these temporal signals into the frequency domain to construct spectral features, which are used to capture anomalies that arise during the reasoning process; analysis experiments on these spectral features have proven the effectiveness of this approach. Finally, a hallucination detection algorithm is designed based on these spectral features to identify hallucinations in the generated content. By effectively combining the modeling of the reasoning process with frequency-domain feature extraction, the HSAD method overcomes the limitations of existing approaches in terms of knowledge coverage and the detection of reasoning biases, demonstrating higher detection accuracy and robustness. 

---
# Jackal: A Real-World Execution-Based Benchmark Evaluating Large Language Models on Text-to-JQL Tasks 

**Authors**: Kevin Frank, Anmol Gulati, Elias Lumer, Sindy Campagna, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2509.23579)  

**Abstract**: Enterprise teams rely on the Jira Query Language (JQL) to retrieve and filter issues from Jira. Yet, to our knowledge, there is no open, real-world, execution-based benchmark for mapping natural language queries to JQL. We introduce Jackal, a novel, large-scale text-to-JQL benchmark comprising 100,000 natural language (NL) requests paired with validated JQL queries and execution-based results on a live Jira instance with over 200,000 issues. To reflect real-world usage, each JQL query is associated with four types of user requests: (i) Long NL, (ii) Short NL, (iii) Semantically Similar, and (iv) Semantically Exact. We release Jackal, a corpus of 100,000 text-to-JQL pairs, together with an execution-based scoring toolkit, and a static snapshot of the evaluated Jira instance for reproducibility. We report text-to-JQL results on 23 Large Language Models (LLMs) spanning parameter sizes, open and closed source models, across execution accuracy, exact match, and canonical exact match. In this paper, we report results on Jackal-5K, a 5,000-pair subset of Jackal. On Jackal-5K, the best overall model (Gemini 2.5 Pro) achieves only 60.3% execution accuracy averaged equally across four user request types. Performance varies significantly across user request types: (i) Long NL (86.0%), (ii) Short NL (35.7%), (iii) Semantically Similar (22.7%), and (iv) Semantically Exact (99.3%). By benchmarking LLMs on their ability to produce correct and executable JQL queries, Jackal exposes the limitations of current state-of-the-art LLMs and sets a new, execution-based challenge for future research in Jira enterprise data. 

---
# Towards Efficient CoT Distillation: Self-Guided Rationale Selector for Better Performance with Fewer Rationales 

**Authors**: Jianzhi Yan, Le Liu, Youcheng Pan, Shiwei Chen, Yang Xiang, Buzhou Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23574)  

**Abstract**: Chain-of-thought (CoT) distillation aims to enhance small language models' (SLMs) reasoning by transferring multi-step reasoning capability from the larger teacher models. However, existing work underestimates rationale quality, focusing primarily on data quantity, which may transfer noisy or incorrect information to the student model. To address the above issues, we proposed \textbf{M}odel-\textbf{O}riented \textbf{R}ationale \textbf{S}election \textbf{D}istillation (MoRSD), which can discern and select high quality rationales for distillation to improve performance further. We further propose a Rationale Difficulty (RD) metric to measure the ability of the student model to generate the correct answer under a given rationale. Compared to the baseline, we achieved 4.6$\%$ average improvement on seven datasets over three tasks, using fewer rationales by controlling their accuracy, diversity, and difficulty. Our results reveal that a small portion of the high quality rationales can enhance the reasoning ability of student models than the entire dataset. Our method promises to be a possible solution for efficient CoT distillation. Our code will be released in this https URL. 

---
# Automatic Speech Recognition for Greek Medical Dictation 

**Authors**: Vardis Georgilas, Themos Stafylakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.23550)  

**Abstract**: Medical dictation systems are essential tools in modern healthcare, enabling accurate and efficient conversion of speech into written medical documentation. The main objective of this paper is to create a domain-specific system for Greek medical speech transcriptions. The ultimate goal is to assist healthcare professionals by reducing the overload of manual documentation and improving workflow efficiency. Towards this goal, we develop a system that combines automatic speech recognition techniques with text correction model, allowing better handling of domain-specific terminology and linguistic variations in Greek. Our approach leverages both acoustic and textual modeling to create more realistic and reliable transcriptions. We focused on adapting existing language and speech technologies to the Greek medical context, addressing challenges such as complex medical terminology and linguistic inconsistencies. Through domain-specific fine-tuning, our system achieves more accurate and coherent transcriptions, contributing to the development of practical language technologies for the Greek healthcare sector. 

---
# On the Shelf Life of Fine-Tuned LLM Judges: Future Proofing, Backward Compatibility, and Question Generalization 

**Authors**: Janvijay Singh, Austin Xu, Yilun Zhou, Yefan Zhou, Dilek Hakkani-Tur, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2509.23542)  

**Abstract**: The LLM-as-a-judge paradigm is widely used in both evaluating free-text model responses and reward modeling for model alignment and finetuning. Recently, finetuning judges with judge-specific data has emerged as an often preferred choice over directly prompting frontier models as judges, as the former achieves better performance with smaller model sizes while being more robust to common biases. However, the standard evaluation ignores several practical concerns of finetuned judges regarding their real world deployment. In this paper, we identify and formalize three aspects that affect the shelf life of these judges: future proofing and backward compatibility -- how well judges finetuned on responses by today's generator models perform on responses by future models or past models, as well as question generalization -- how well judges generalize to unseen questions at test time. We study these three aspects in the math domain under a unified framework with varying train and test distributions, three SFT- and DPO-based finetuning algorithms and three different base models. Experiments suggest that future-proofing is challenging for most models, while backward compatibility is relatively easy, with DPO-trained models consistently improving performance. We further find that continual learning provides a more balanced adaptation to shifts between older and newer response distributions than training solely on stronger or weaker responses. Moreover, all models observe certain degrees of performance degradation when moving from questions seen during training to unseen ones, showing that current judges do not fully generalize to unseen questions. These findings provide insights into practical considerations for developing and deploying judge models in the face of ever-changing generators. 

---
# From Human Annotation to Automation: LLM-in-the-Loop Active Learning for Arabic Sentiment Analysis 

**Authors**: Dania Refai, Alaa Dalaq, Doaa Dalaq, Irfan Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2509.23515)  

**Abstract**: Natural language processing (NLP), particularly sentiment analysis, plays a vital role in areas like marketing, customer service, and social media monitoring by providing insights into user opinions and emotions. However, progress in Arabic sentiment analysis remains limited due to the lack of large, high-quality labeled datasets. While active learning has proven effective in reducing annotation efforts in other languages, few studies have explored it in Arabic sentiment tasks. Likewise, the use of large language models (LLMs) for assisting annotation and comparing their performance to human labeling is still largely unexplored in the Arabic context. In this paper, we propose an active learning framework for Arabic sentiment analysis designed to reduce annotation costs while maintaining high performance. We evaluate multiple deep learning architectures: Specifically, long short-term memory (LSTM), gated recurrent units (GRU), and recurrent neural networks (RNN), across three benchmark datasets: Hunger Station, AJGT, and MASAC, encompassing both modern standard Arabic and dialectal variations. Additionally, two annotation strategies are compared: Human labeling and LLM-assisted labeling. Five LLMs are evaluated as annotators: GPT-4o, Claude 3 Sonnet, Gemini 2.5 Pro, DeepSeek Chat, and LLaMA 3 70B Instruct. For each dataset, the best-performing LLM was used: GPT-4o for Hunger Station, Claude 3 Sonnet for AJGT, and DeepSeek Chat for MASAC. Our results show that LLM-assisted active learning achieves competitive or superior performance compared to human labeling. For example, on the Hunger Station dataset, the LSTM model achieved 93% accuracy with only 450 labeled samples using GPT-4o-generated labels, while on the MASAC dataset, DeepSeek Chat reached 82% accuracy with 650 labeled samples, matching the accuracy obtained through human labeling. 

---
# AraS2P: Arabic Speech-to-Phonemes System 

**Authors**: Bassam Matar, Mohamed Fayed, Ayman Khalafallah  

**Link**: [PDF](https://arxiv.org/pdf/2509.23504)  

**Abstract**: This paper describes AraS2P, our speech-to-phonemes system submitted to the Iqra'Eval 2025 Shared Task. We adapted Wav2Vec2-BERT via Two-Stage training strategy. In the first stage, task-adaptive continue pretraining was performed on large-scale Arabic speech-phonemes datasets, which were generated by converting the Arabic text using the MSA Phonetiser. In the second stage, the model was fine-tuned on the official shared task data, with additional augmentation from XTTS-v2-synthesized recitations featuring varied Ayat segments, speaker embeddings, and textual perturbations to simulate possible human errors. The system ranked first on the official leaderboard, demonstrating that phoneme-aware pretraining combined with targeted augmentation yields strong performance in phoneme-level mispronunciation detection. 

---
# The Impact of Role Design in In-Context Learning for Large Language Models 

**Authors**: Hamidreza Rouzegar, Masoud Makrehchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23501)  

**Abstract**: In-context learning (ICL) enables Large Language Models (LLMs) to generate predictions based on prompts without additional fine-tuning. While prompt engineering has been widely studied, the impact of role design within prompts remains underexplored. This study examines the influence of role configurations in zero-shot and few-shot learning scenarios using GPT-3.5 and GPT-4o from OpenAI and Llama2-7b and Llama2-13b from Meta. We evaluate the models' performance across datasets, focusing on tasks like sentiment analysis, text classification, question answering, and math reasoning. Our findings suggest the potential of role-based prompt structuring to enhance LLM performance. 

---
# Text-Based Approaches to Item Difficulty Modeling in Large-Scale Assessments: A Systematic Review 

**Authors**: Sydney Peters, Nan Zhang, Hong Jiao, Ming Li, Tianyi Zhou, Robert Lissitz  

**Link**: [PDF](https://arxiv.org/pdf/2509.23486)  

**Abstract**: Item difficulty plays a crucial role in test performance, interpretability of scores, and equity for all test-takers, especially in large-scale assessments. Traditional approaches to item difficulty modeling rely on field testing and classical test theory (CTT)-based item analysis or item response theory (IRT) calibration, which can be time-consuming and costly. To overcome these challenges, text-based approaches leveraging machine learning and language models, have emerged as promising alternatives. This paper reviews and synthesizes 37 articles on automated item difficulty prediction in large-scale assessment settings published through May 2025. For each study, we delineate the dataset, difficulty parameter, subject domain, item type, number of items, training and test data split, input, features, model, evaluation criteria, and model performance outcomes. Results showed that although classic machine learning models remain relevant due to their interpretability, state-of-the-art language models, using both small and large transformer-based architectures, can capture syntactic and semantic patterns without the need for manual feature engineering. Uniquely, model performance outcomes were summarized to serve as a benchmark for future research and overall, text-based methods have the potential to predict item difficulty with root mean square error (RMSE) as low as 0.165, Pearson correlation as high as 0.87, and accuracy as high as 0.806. The review concludes by discussing implications for practice and outlining future research directions for automated item difficulty modeling. 

---
# Cognition-of-Thought Elicits Social-Aligned Reasoning in Large Language Models 

**Authors**: Xuanming Zhang, Yuxuan Chen, Min-Hsuan Yeh, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23441)  

**Abstract**: Large language models (LLMs) excel at complex reasoning but can still exhibit harmful behaviors. Current alignment strategies typically embed safety into model weights, making these controls implicit, static, and difficult to modify. This paper introduces Cognition-of-Thought (CooT), a novel decoding-time framework that equips LLMs with an explicit cognitive self-monitoring loop. CooT couples a standard text Generator with a cognitive Perceiver that continuously monitors the unfolding sequence. The Perceiver uses a structured, precedence-based hierarchy of principles (e.g., safety over obedience) to detect potential misalignments as they arise. When violations are flagged, CooT intervenes by rolling back the generation to the point of error and regenerating under injected guidance that combines universal social priors with context-specific warnings. CooT thus transforms alignment from a fixed property into an explicit, dynamic, and auditable process active during inference, allowing for flexible policy updates without retraining the model. Extensive experiments across multiple benchmarks and model families confirm that CooT consistently improves safety and social reasoning performance. 

---
# Retrieval-Constrained Decoding Reveals Underestimated Parametric Knowledge in Language Models 

**Authors**: Rajaa El Hamdani, Samy Haffoudhi, Nils Holzenberger, Fabian Suchanek, Thomas Bonald, Fragkiskos D. Malliaros  

**Link**: [PDF](https://arxiv.org/pdf/2509.23417)  

**Abstract**: Language models (LMs) encode substantial factual knowledge, but often produce answers judged as incorrect. We hypothesize that many of these answers are actually correct, but are expressed in alternative surface forms that are dismissed due to an overly strict evaluation, leading to an underestimation of models' parametric knowledge. We propose Retrieval-Constrained Decoding (RCD), a decoding strategy that restricts model outputs to unique surface forms. We introduce YAGO-QA, a dataset of 19,137 general knowledge questions. Evaluating open-source LMs from 135M to 70B parameters, we show that standard decoding undervalues their knowledge. For instance, Llama-3.1-70B scores only 32.3% F1 with vanilla decoding but 46.0% with RCD. Similarly, Llama-3.1-8B reaches 33.0% with RCD, outperforming the larger model under vanilla decoding. We publicly share the code and dataset at this https URL. 

---
# Comparison of Scoring Rationales Between Large Language Models and Human Raters 

**Authors**: Haowei Hua, Hong Jiao, Dan Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.23412)  

**Abstract**: Advances in automated scoring are closely aligned with advances in machine-learning and natural-language-processing techniques. With recent progress in large language models (LLMs), the use of ChatGPT, Gemini, Claude, and other generative-AI chatbots for automated scoring has been explored. Given their strong reasoning capabilities, LLMs can also produce rationales to support the scores they assign. Thus, evaluating the rationales provided by both human and LLM raters can help improve the understanding of the reasoning that each type of rater applies when assigning a score. This study investigates the rationales of human and LLM raters to identify potential causes of scoring inconsistency. Using essays from a large-scale test, the scoring accuracy of GPT-4o, Gemini, and other LLMs is examined based on quadratic weighted kappa and normalized mutual information. Cosine similarity is used to evaluate the similarity of the rationales provided. In addition, clustering patterns in rationales are explored using principal component analysis based on the embeddings of the rationales. The findings of this study provide insights into the accuracy and ``thinking'' of LLMs in automated scoring, helping to improve the understanding of the rationales behind both human scoring and LLM-based automated scoring. 

---
# Liaozhai through the Looking-Glass: On Paratextual Explicitation of Culture-Bound Terms in Machine Translation 

**Authors**: Sherrie Shen, Weixuan Wang, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2509.23395)  

**Abstract**: The faithful transfer of contextually-embedded meaning continues to challenge contemporary machine translation (MT), particularly in the rendering of culture-bound terms--expressions or concepts rooted in specific languages or cultures, resisting direct linguistic transfer. Existing computational approaches to explicitating these terms have focused exclusively on in-text solutions, overlooking paratextual apparatus in the footnotes and endnotes employed by professional translators. In this paper, we formalize Genette's (1987) theory of paratexts from literary and translation studies to introduce the task of paratextual explicitation for MT. We construct a dataset of 560 expert-aligned paratexts from four English translations of the classical Chinese short story collection Liaozhai and evaluate LLMs with and without reasoning traces on choice and content of explicitation. Experiments across intrinsic prompting and agentic retrieval methods establish the difficulty of this task, with human evaluation showing that LLM-generated paratexts improve audience comprehension, though remain considerably less effective than translator-authored ones. Beyond model performance, statistical analysis reveals that even professional translators vary widely in their use of paratexts, suggesting that cultural mediation is inherently open-ended rather than prescriptive. Our findings demonstrate the potential of paratextual explicitation in advancing MT beyond linguistic equivalence, with promising extensions to monolingual explanation and personalized adaptation. 

---
# No Loss, No Gain: Gated Refinement and Adaptive Compression for Prompt Optimization 

**Authors**: Wenhang Shi, Yiren Chen, Shuqing Bian, Xinyi Zhang, Kai Tang, Pengfei Hu, Zhe Zhao, Wei Lu, Xiaoyong Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.23387)  

**Abstract**: Prompt engineering is crucial for leveraging the full potential of large language models (LLMs). While automatic prompt optimization offers a scalable alternative to costly manual design, generating effective prompts remains challenging. Existing methods often struggle to stably generate improved prompts, leading to low efficiency, and overlook that prompt optimization easily gets trapped in local optima. Addressing this, we propose GRACE, a framework that integrates two synergistic strategies: Gated Refinement and Adaptive Compression, achieving Efficient prompt optimization. The gated refinement strategy introduces a feedback regulation gate and an update rejection gate, which refine update signals to produce stable and effective prompt improvements. When optimization stagnates, the adaptive compression strategy distills the prompt's core concepts, restructuring the optimization trace and opening new paths. By strategically introducing information loss through refinement and compression, GRACE delivers substantial gains in performance and efficiency. In extensive experiments on 11 tasks across three practical domains, including BIG-Bench Hard (BBH), domain-specific, and general NLP tasks, GRACE achieves significant average relative performance improvements of 4.7%, 4.4% and 2.7% over state-of-the-art methods, respectively. Further analysis shows that GRACE achieves these gains using only 25% of the prompt generation budget required by prior methods, highlighting its high optimization efficiency and low computational overhead. Our code is available at this https URL. 

---
# Train Once, Answer All: Many Pretraining Experiments for the Cost of One 

**Authors**: Sebastian Bordt, Martin Pawelczyk  

**Link**: [PDF](https://arxiv.org/pdf/2509.23383)  

**Abstract**: Recent work has demonstrated that controlled pretraining experiments are a powerful tool for understanding learning, reasoning, and memorization in large language models (LLMs). However, the computational cost of pretraining presents a significant constraint. To overcome this constraint, we propose to conduct multiple pretraining experiments simultaneously during a single training run. We demonstrate the feasibility of this approach by conducting ten experiments during the training of a 1.5B parameter model on 210B tokens. Although we only train a single model, we can replicate the results from multiple previous works on data contamination, poisoning, and memorization. We also conduct novel investigations into knowledge acquisition, mathematical reasoning, and watermarking. For example, we dynamically update the training data until the model acquires a particular piece of knowledge. Remarkably, the influence of the ten experiments on the model's training dynamics and overall performance is minimal. However, interactions between different experiments may act as a potential confounder in our approach. We propose to test for interactions with continual pretraining experiments, finding them to be negligible in our setup. Overall, our findings suggest that performing multiple pretraining experiments in a single training run can enable rigorous scientific experimentation with large models on a compute budget. 

---
# Guard Vector: Beyond English LLM Guardrails with Task-Vector Composition and Streaming-Aware Prefix SFT 

**Authors**: Wonhyuk Lee, Youngchol Kim, Yunjin Park, Junhyung Moon, Dongyoung Jeong, Wanjin Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.23381)  

**Abstract**: We introduce Guard Vector, a safety task vector computed as the parameter difference between a guardrail model (Guard Model) and a same-architecture pretrained language model. Composing this vector with a target language model yields a Target Guard Model (TGM). We then adapt TGM with a streaming-aware approach that combines prefix-based training and evaluation with a classifier that produces a single-token output. With this composition alone, TGM improves classification quality over established Guard Models across standard safety suites and enables language extensibility to Chinese, Japanese, and Korean, requiring neither additional training nor target language labels. It also demonstrates model portability across two widely used public guardrail backbones, Llama and Gemma. With prefix SFT (supervised fine-tuning), TGM preserves classification quality under streaming by aligning the behavior between prefix inputs and full-text inputs. The single-token output design increases throughput and reduces latency. Together, these components reduce data and compute requirements while promoting streaming-aware evaluation practices, thereby contributing to a more responsible AI ecosystem. 

---
# CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding 

**Authors**: Xi Zhang, Zaiqiao Meng, Jake Lever, Edmond S. L. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2509.23379)  

**Abstract**: Multimodal large language models (MLLMs) have recently achieved remarkable progress in radiology by integrating visual perception with natural language understanding. However, they often generate clinically unsupported descriptions, known as medical hallucinations, which pose serious risks in medical applications that demand accuracy and image-grounded outputs. Through empirical analysis, we find that prompt-induced hallucinations remain prevalent in radiology MLLMs, largely due to over-sensitivity to clinical sections. To address this, we introduce Clinical Contrastive Cecoding (CCD), a training-free and retrieval-free inference framework that integrates structured clinical signals from task-specific radiology expert models. CCD introduces a dual-stage contrastive mechanism to refine token-level logits during generation, thereby enhancing clinical fidelity without modifying the base MLLM. Experiments on three datasets and multiple models demonstrate that CCD consistently improves overall performance on radiology report generation (RRG). On the MIMIC-CXR dataset, it yields up to a 17% improvement in RadGraph-F1 when applied to state-of-the-art RRG models. Our approach provides a lightweight and generalisable solution for mitigating medical hallucinations, effectively bridging expert models and MLLMs in radiology. 

---
# Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization 

**Authors**: Junming Yang, Ning Xu, Biao Liu, Shiqi Qiao, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23371)  

**Abstract**: Preference optimization is crucial for aligning large language models (LLMs) with human values and intentions. A significant challenge in this process is the distribution mismatch between pre-collected offline preference data and the evolving model policy. Existing methods attempt to reduce this gap using static heuristics or decoupled online sampling strategies, but they often fail to adapt to the model's dynamic learning state. To bridge this gap, we propose Meta-Weighted Adaptive Preference Optimization (MetaAPO), a novel framework that dynamically couples data generation with model training. MetaAPO employs a lightweight meta-learner, as an "alignment gap estimator", to evaluate the potential benefits of on-policy sampling in relation to offline data. This guides targeted online generation and assigns sample-wise meta-weights to the optimization objective, dynamically balancing the quality and distribution of online and offline data. Experiments on AlpacaEval 2, Arena-Hard and MT-Bench demonstrate that MetaAPO consistently outperforms existing preference optimization approaches across various settings, while reducing 42% in online annotation costs. 

---
# MedCritical: Enhancing Medical Reasoning in Small Language Models via Self-Collaborative Correction 

**Authors**: Xinchun Su, Chunxu Luo, Yixuan Li, Weidong Yang, Lipeng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.23368)  

**Abstract**: In the field of medicine, complex reasoning tasks such as clinical diagnosis, treatment planning, and medical knowledge integration pose significant challenges, where small language models often underperform compared to large language models like GPT-4 and Deepseek. Recent knowledge distillation-based methods aim to address these issues through teacher-guided error correction, but this LLM as judge approach remains challenging in terms of cost, time, and efficiency. To circumvent this issue, we propose a novel two-stage framework, MedCritical, which uses a small language model fine-tuned by a large teacher model to play against itself. In the first stage, we extract high-level and detailed long-chain thought templates from the teacher model to guide the student model to generate more complex reasoning thoughts. In the second stage, we introduce direct preference optimization (DPO) through model self-iteration collaboration to enhance the reasoning ability of the student model by playing against the correction trajectory of the fine-tuned model during training. This model self-learning DPO approach teaches the student model to use its own error-driven insights to consolidate its skills and knowledge to solve complex problems, and achieves comparable results to traditional knowledge distillation methods using teacher models at a lower cost. Notably, our MedCritical 7B model outperforms the Taiyi and Huatuo-o1-7B models by 3.04\% and 10.12\% respectively on the CMExam benchmark, achieving new SOTA performance among 7B-class small models. 

---
# Dual-Space Smoothness for Robust and Balanced LLM Unlearning 

**Authors**: Han Yan, Zheyuan Liu, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23362)  

**Abstract**: With the rapid advancement of large language models, Machine Unlearning has emerged to address growing concerns around user privacy, copyright infringement, and overall safety. Yet state-of-the-art (SOTA) unlearning methods often suffer from catastrophic forgetting and metric imbalance, for example by over-optimizing one objective (e.g., unlearning effectiveness, utility preservation, or privacy protection) at the expense of others. In addition, small perturbations in the representation or parameter space can be exploited by relearn and jailbreak attacks. To address these challenges, we propose PRISM, a unified framework that enforces dual-space smoothness in representation and parameter spaces to improve robustness and balance unlearning metrics. PRISM consists of two smoothness optimization stages: (i) a representation space stage that employs a robustly trained probe to defend against jailbreak attacks, and (ii) a parameter-space stage that decouples retain-forget gradient conflicts, reduces imbalance, and smooths the parameter space to mitigate relearning attacks. Extensive experiments on WMDP and MUSE, across conversational-dialogue and continuous-text settings, show that PRISM outperforms SOTA baselines under multiple attacks while achieving a better balance among key metrics. 

---
# C-Evolve: Consensus-based Evolution for Prompt Groups 

**Authors**: Tiancheng Li, Yuhang Wang, Zhiyang Chen, Zijun Wang, Liyuan Ma, Guo-jun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23331)  

**Abstract**: Prompt evolution algorithms offer a powerful paradigm for enhancing AI systems based on closed-source models, while few work explores whether aggregating results from multiple prompts to reach a consensus can further advance the system capability boundary. In this paper, we introduce Consensus-Evolve (C-Evolve), an evolutionary algorithm that discovers a group of prompts whose aggregated outputs after majority voting achieve optimal performance. More specifically, C-Evolve employs an island-based evolutionary algorithm to maintain population diversity, and prompts from distinct islands are selected to form groups to aggregate their outputs. The key difference from single individual evolution is a voting score, which evaluates each individual prompt's contribution within groups. We take this as the fitness score for evolution instead of individual performance. Consequently, C-Evolve is more likely to produce and maintain prompts with higher potential to form a high-performing group and eliminate low-performing ones, gradually improving the group performance after reaching consensus. Our method achieves state-of-the-art performance across a wide range of tasks, including both open-ended tasks like HotpotQA and closed-ended tasks like MATH. On Qwen3-8B, C-Evolve achieves 70.67% on HotpotQA and 43.88% on IFBench, which are 4.95% and 2.73% higher than GEPA, respectively. For GPT-4.1-mini, the accuracy on IFBench is further improved to 47.96% and reaches 95.33% in the MATH benchmark. These results demonstrate the C-Evolve's competitive performance. 

---
# Learning to Reason in Structured In-context Environments with Reinforcement Learning 

**Authors**: Peng Yu, Zeyuan Zhao, Shao Zhang, Luoyi Fu, Xinbing Wang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23330)  

**Abstract**: Large language models (LLMs) have achieved significant advancements in reasoning capabilities through reinforcement learning (RL) via environmental exploration. As the intrinsic properties of the environment determine the abilities that LLMs can learn, the environment plays a important role in the RL finetuning process. An ideal LLM reasoning environment should possess three core characteristics: scalability, generalizable reasoning, and verifiability. However, existing mathematical and coding environments are difficult to scale due to heavy reliance on expert annotation, while the skills learned in game-based environments are too specialized to generalize. To bridge this gap, we introduce the \textbf{S}tructured \textbf{I}n-context \textbf{E}nvironment (SIE) framework. SIE achieves scalability by automatically constructing reasoning environments from large-scale structured data, where the rich compositional patterns naturally support generalizable reasoning. Moreover, the explicit schemas and reasoning chains in structured data provide a foundation for rule-based verifiability. Experimental results show that SIE framework not only achieves substantial improvements in in-domain structured reasoning, but also enables the learned compositional reasoning skills to generalize effectively to out-of-domain mathematical and logical reasoning tasks. We further explored learning in information-limited partial SIEs and found that LLMs can infer the missing information through exploring the environment, leading to robust reasoning improvements and generalization performance. 

---
# Scaling Policy Compliance Assessment in Language Models with Policy Reasoning Traces 

**Authors**: Joseph Marvin Imperial, Harish Tayyar Madabushi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23291)  

**Abstract**: Policy compliance assessment is a fundamental task of evaluating whether an input case strictly complies with a set of human-defined rules, more generally known as policies. In practice, human experts follow a systematic, step-by-step process to identify violations with respect to specific stipulations outlined in the policy. However, such documentation of gold-standard, expert-level reasoning processes is costly to acquire. In this paper, we introduce Policy Reasoning Traces (PRT), a form of specialized generated reasoning chains that serve as a reasoning bridge to improve an LLM's policy compliance assessment capabilities. Our empirical evaluations demonstrate that the use of PRTs for both inference-time and training-time scenarios significantly enhances the performance of open-weight and commercial models, setting a new state-of-the-art for HIPAA and GDPR policies. Beyond accuracy gains, we also highlight how PRTs can improve an LLM's ability to accurately cite policy clauses, as well as influence compliance decisions through their high utilization from the raw chains of thought. 

---
# A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models 

**Authors**: Wonje Jeung, Sangyeon Yoon, Yoonjun Cho, Dongjae Jeon, Sangwoo Shin, Hyesoo Hong, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2509.23286)  

**Abstract**: Diffusion large language models (dLLMs) enable any-order generation, but this flexibility enlarges the attack surface: harmful spans may appear at arbitrary positions, and template-based prefilling attacks such as DIJA bypass response-level refusals. We introduce A2D (Any-Order, Any-Step Defense), a token-level alignment method that aligns dLLMs to emit an [EOS] refusal signal whenever harmful content arises. By aligning safety directly at the token-level under randomized masking, A2D achieves robustness to both any-decoding-order and any-step prefilling attacks under various conditions. It also enables real-time monitoring: dLLMs may begin a response but automatically terminate if unsafe continuation emerges. On safety benchmarks, A2D consistently prevents the generation of harmful outputs, slashing DIJA success rates from over 80% to near-zero (1.3% on LLaDA-8B-Instruct, 0.0% on Dream-v0-Instruct-7B), and thresholded [EOS] probabilities allow early rejection, yielding up to 19.3x faster safe termination. 

---
# Fin-ExBERT: User Intent based Text Extraction in Financial Context using Graph-Augmented BERT and trainable Plugin 

**Authors**: Soumick Sarker, Abhijit Kumar Rai  

**Link**: [PDF](https://arxiv.org/pdf/2509.23259)  

**Abstract**: Financial dialogue transcripts pose a unique challenge for sentence-level information extraction due to their informal structure, domain-specific vocabulary, and variable intent density. We introduce Fin-ExBERT, a lightweight and modular framework for extracting user intent-relevant sentences from annotated financial service calls. Our approach builds on a domain-adapted BERT (Bidirectional Encoder Representations from Transformers) backbone enhanced with LoRA (Low-Rank Adaptation) adapters, enabling efficient fine-tuning using limited labeled data. We propose a two-stage training strategy with progressive unfreezing: initially training a classifier head while freezing the backbone, followed by gradual fine-tuning of the entire model with differential learning rates. To ensure robust extraction under uncertainty, we adopt a dynamic thresholding strategy based on probability curvature (elbow detection), avoiding fixed cutoff heuristics. Empirical results show strong precision and F1 performance on real-world transcripts, with interpretable output suitable for downstream auditing and question-answering workflows. The full framework supports batched evaluation, visualization, and calibrated export, offering a deployable solution for financial dialogue mining. 

---
# Detecting Corpus-Level Knowledge Inconsistencies in Wikipedia with Large Language Models 

**Authors**: Sina J. Semnani, Jirayu Burapacheep, Arpandeep Khatua, Thanawan Atchariyachanvanit, Zheng Wang, Monica S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2509.23233)  

**Abstract**: Wikipedia is the largest open knowledge corpus, widely used worldwide and serving as a key resource for training large language models (LLMs) and retrieval-augmented generation (RAG) systems. Ensuring its accuracy is therefore critical. But how accurate is Wikipedia, and how can we improve it?
We focus on inconsistencies, a specific type of factual inaccuracy, and introduce the task of corpus-level inconsistency detection. We present CLAIRE, an agentic system that combines LLM reasoning with retrieval to surface potentially inconsistent claims along with contextual evidence for human review. In a user study with experienced Wikipedia editors, 87.5% reported higher confidence when using CLAIRE, and participants identified 64.7% more inconsistencies in the same amount of time.
Combining CLAIRE with human annotation, we contribute WIKICOLLIDE, the first benchmark of real Wikipedia inconsistencies. Using random sampling with CLAIRE-assisted analysis, we find that at least 3.3% of English Wikipedia facts contradict another fact, with inconsistencies propagating into 7.3% of FEVEROUS and 4.0% of AmbigQA examples. Benchmarking strong baselines on this dataset reveals substantial headroom: the best fully automated system achieves an AUROC of only 75.1%.
Our results show that contradictions are a measurable component of Wikipedia and that LLM-based systems like CLAIRE can provide a practical tool to help editors improve knowledge consistency at scale. 

---
# A Structured Framework for Evaluating and Enhancing Interpretive Capabilities of Multimodal LLMs in Culturally Situated Tasks 

**Authors**: Haorui Yu, Ramon Ruiz-Dolz, Qiufeng Yi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23208)  

**Abstract**: This study aims to test and evaluate the capabilities and characteristics of current mainstream Visual Language Models (VLMs) in generating critiques for traditional Chinese painting. To achieve this, we first developed a quantitative framework for Chinese painting critique. This framework was constructed by extracting multi-dimensional evaluative features covering evaluative stance, feature focus, and commentary quality from human expert critiques using a zero-shot classification model. Based on these features, several representative critic personas were defined and quantified. This framework was then employed to evaluate selected VLMs such as Llama, Qwen, or Gemini. The experimental design involved persona-guided prompting to assess the VLM's ability to generate critiques from diverse perspectives. Our findings reveal the current performance levels, strengths, and areas for improvement of VLMs in the domain of art critique, offering insights into their potential and limitations in complex semantic understanding and content generation tasks. The code used for our experiments can be publicly accessed at: this https URL. 

---
# PARL-MT: Learning to Call Functions in Multi-Turn Conversation with Progress Awareness 

**Authors**: Huacan Chai, Zijie Cao, Maolin Ran, Yingxuan Yang, Jianghao Lin, pengxin, Hairui Wang, Renjie Ding, Ziyu Wan, Muning Wen, Weiwen Liu, Weinan Zhang, Fei Huang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23206)  

**Abstract**: Large language models (LLMs) have achieved impressive success in single-turn function calling, yet real-world applications such as travel planning or multi-stage data analysis typically unfold across multi-turn conversations. In these settings, LLMs must not only issue accurate function calls at each step but also maintain progress awareness, the ability to summarize past interactions and plan future actions to ensure coherent, long-horizon task execution. Existing approaches, however, either reduce multi-turn training to isolated single-turn samples, which neglects task-level planning, or employ end-to-end reinforcement learning (RL) that struggles with redundancy and lacks explicit integration of progress awareness. To overcome these limitations, we introduce PARL-MT, a framework that explicitly incorporates progress awareness into LLM training for multi-turn function calling. PARL-MT combines (i) a Progress Awareness Generation (PAG) pipeline, which automatically constructs datasets coupling conversation summaries with future task planning, and (ii) a Progress Awareness-Guided Reinforcement Learning (PAG-RL) algorithm, which integrates progress awareness into RL training to reduce contextual redundancy and improve alignment between local actions and global task completion. Empirical results on two public benchmarks demonstrate that PARL-MT significantly outperforms existing methods, highlighting the effectiveness of progress awareness in enabling robust and efficient multi-turn function calling. 

---
# Steering Prepositional Phrases in Language Models: A Case of with-headed Adjectival and Adverbial Complements in Gemma-2 

**Authors**: Stefan Arnold, René Gröbner  

**Link**: [PDF](https://arxiv.org/pdf/2509.23204)  

**Abstract**: Language Models, when generating prepositional phrases, must often decide for whether their complements functions as an instrumental adjunct (describing the verb adverbially) or an attributive modifier (enriching the noun adjectivally), yet the internal mechanisms that resolve this split decision remain poorly understood. In this study, we conduct a targeted investigation into Gemma-2 to uncover and control the generation of prepositional complements. We assemble a prompt suite containing with-headed prepositional phrases whose contexts equally accommodate either an instrumental or attributive continuation, revealing a strong preference for an instrumental reading at a ratio of 3:4. To pinpoint individual attention heads that favor instrumental over attributive complements, we project activations into the vocabulary space. By scaling the value vector of a single attention head, we can shift the distribution of functional roles of complements, attenuating instruments to 33% while elevating attributes to 36%. 

---
# Global Beats, Local Tongue: Studying Code Switching in K-pop Hits on Billboard Charts 

**Authors**: Aditya Narayan Sankaran, Reza Farahbakhsh, Noel Crespi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23197)  

**Abstract**: Code switching, particularly between Korean and English, has become a defining feature of modern K-pop, reflecting both aesthetic choices and global market strategies. This paper is a primary investigation into the linguistic strategies employed in K-pop songs that achieve global chart success, with a focus on the role of code-switching and English lyric usage. A dataset of K-pop songs that appeared on the Billboard Hot 100 and Global 200 charts from 2017 to 2025, spanning 14 groups and 8 solo artists, was compiled. Using this dataset, the proportion of English and Korean lyrics, the frequency of code-switching, and other stylistic features were analysed. It was found that English dominates the linguistic landscape of globally charting K-pop songs, with both male and female performers exhibiting high degrees of code-switching and English usage. Statistical tests indicated no significant gender-based differences, although female solo artists tend to favour English more consistently. A classification task was also performed to predict performer gender from lyrics, achieving macro F1 scores up to 0.76 using multilingual embeddings and handcrafted features. Finally, differences between songs charting on the Hot 100 versus the Global 200 were examined, suggesting that, while there is no significant gender difference in English, higher English usage may be more critical for success in the US-focused Hot 100. The findings highlight how linguistic choices in K-pop lyrics are shaped by global market pressures and reveal stylistic patterns that reflect performer identity and chart context. 

---
# From Harm to Help: Turning Reasoning In-Context Demos into Assets for Reasoning LMs 

**Authors**: Haonan Wang, Weida Liang, Zihang Fu, Nie Zheng, Yifan Zhang, Yao Tong, Tongyao Zhu, Hao Jiang, Chuang Li, Jiaying Wu, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23196)  

**Abstract**: Recent reasoning LLMs (RLMs), especially those trained with verifier-based reinforcement learning, often perform worse with few-shot CoT than with direct answering. We revisit this paradox using high-quality reasoning traces from DeepSeek-R1 as demonstrations and find that adding more exemplars consistently degrades accuracy, even when demonstrations are optimal. A detailed analysis reveals two mechanisms behind this decline: (i) semantic misguidance, where high textual similarity leads the model to treat the target as the same as the exemplar and to copy intermediate steps verbatim; and (ii) strategy transfer failure, where the model struggles to extract useful reasoning strategies and apply them to target questions. Guided by these, we introduce Insight-to-Solve (I2S), a sequential test-time procedure that turns demonstrations into explicit, reusable insights and derives a target-specific reasoning trace; optionally, the reasoning is self-refined for coherence and correctness (I2S+). Extensive experiments on diverse benchmarks show that I2S and I2S+ consistently outperform both direct answering and test-time scaling baselines across open- and closed-source models. Even for GPT models, our method helps: on AIME'25, GPT-4.1 rises by +14.0%, and o1-mini improves by +2.7% on AIME and +1.7% on GPQA, indicating that in-context demonstrations can be harnessed effectively via insight-refine-solve framework. 

---
# Estimating the strength and timing of syntactic structure building in naturalistic reading 

**Authors**: Nan Wang, Jiaxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23195)  

**Abstract**: A central question in psycholinguistics is the timing of syntax in sentence processing. Much of the existing evidence comes from violation paradigms, which conflate two separable processes - syntactic category detection and phrase structure construction - and implicitly assume that phrase structure follows category detection. In this study, we use co-registered EEG and eye-tracking data from the ZuCo corpus to disentangle these processes and test their temporal order under naturalistic reading conditions. Analyses of gaze transitions showed that readers preferentially moved between syntactic heads, suggesting that phrase structures, rather than serial word order, organize scanpaths. Bayesian network modeling further revealed that structural depth was the strongest driver of deviations from linear reading, outweighing lexical familiarity and surprisal. Finally, fixation-related potentials demonstrated that syntactic surprisal influences neural activity before word onset (-184 to -10 ms) and during early integration (48 to 300 ms). These findings extend current models of syntactic timing by showing that phrase structure construction can precede category detection and dominate lexical influences, supporting a predictive "tree-scaffolding" account of comprehension. 

---
# Diagnose, Localize, Align: A Full-Stack Framework for Reliable LLM Multi-Agent Systems under Instruction Conflicts 

**Authors**: Guancheng Wan, Leixin Sun, Longxu Dou, Zitong Shi, Fang Wu, Eric Hanchen Jiang, Wenke Huang, Guibin Zhang, Hejia Geng, Xiangru Tang, Zhenfei Yin, Yizhou Sun, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23188)  

**Abstract**: Large Language Model (LLM)-powered multi-agent systems (MAS) have rapidly advanced collaborative reasoning, tool use, and role-specialized coordination in complex tasks. However, reliability-critical deployment remains hindered by a systemic failure mode: hierarchical compliance under instruction conflicts (system-user, peer-peer), where agents misprioritize system-level rules in the presence of competing demands. Moreover, widely used macro-level metrics (e.g., pass@k) obscure these micro-level violations and offer little actionable guidance for remedy. In this work, we present a full-stack, three-stage framework: (1) Diagnose - Contextualized Role Adherence Score (CRAS), a query-wise, context-aware scoring metric that decomposes role adherence into four measurable dimensions; (2) Localize - attention drift analysis revealing that instruction conflicts are resolved by attention heads that are largely concentrated in middle layers; (3) Align - Surgical Alignment of Instruction Layers (SAIL), which installs LoRA only on the localized focal layers and optimizes a token-weighted DPO-style preference objective that credits tokens by their focal attentional contribution. Across standard benchmarks and MAS frameworks, our surgical approach improves instruction hierarchy compliance (e.g., +5.60% with AutoGen on MedQA) without full-model finetuning. 

---
# Pretraining LLM with Latent Thoughts in Continuous Space 

**Authors**: Boyi Zeng, He Li, Shixiang Song, Yixuan Wang, Ziwei He, Xinbing Wang, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.23184)  

**Abstract**: The remarkable success of Chain-of-Thought (CoT), which enhances performance by scaling generation steps at test-time, inspires us to ask: can we leverage a similar scaling of computational steps during pretraining to improve the generation of each individual token? To address this, we propose a novel pre-training methodology: Pretraining Language Models with Latent Thoughts. Our approach pretrains a language model (LM) to first generate an intermediate latent thought-the last hidden state of the current position-which is then used as input to predict the actual subsequent token. This additional computational step enables the LM to refine its prediction within unconstrained continuous space. Our experiments demonstrate that, at an identical inference cost, a LM that generates one additional latent thought per token outperforms a standard model with double the parameters. For instance, ours-1.4B (Pythia Arch), pretrained on 300B tokens from the Pile, significantly surpasses the vanilla Pythia-2.8B trained on the same data on both language modeling and a range of general downstream tasks. Furthermore, increasing the number of latent thoughts generated before each actual token-forming a chain analogous to CoT-consistently improves the model's performance. 

---
# Test-Time Policy Adaptation for Enhanced Multi-Turn Interactions with LLMs 

**Authors**: Chenxing Wei, Hong Wang, Ying He, Fei Yu, Yao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23166)  

**Abstract**: Large Language Models (LLMs) employ multi-turn interaction as a fundamental paradigm for completing complex tasks. However, their performance often degrades in extended interactions, as they are typically trained on static, single-turn data, which hinders their ability to adapt to real-time user feedback. To address this limitation, we first propose a new paradigm: Test-Time Policy Adaptation for Multi-Turn Interactions (T2PAM), which utilizes user feedback from the ongoing interaction as a reward signal to estimate a latent optimal policy aligned with user preferences, then updates a small subset of parameters to steer the model toward this policy, ultimately enabling efficient in-conversation self-correction. We then introduce Optimum-Referenced One-Step Adaptation (ROSA), a lightweight algorithm that operationalizes T2PAM. ROSA guides the model parameters toward a theoretical optimal policy in a single, efficient update step, avoiding costly iterative gradient-based optimization and minimizing computational overhead. We provide a rigorous theoretical analysis guaranteeing that the policy of ROSA converges to the preference of user as the number of interactions increases. Extensive experiments on challenging benchmark demonstrate that ROSA achieves significant improvements in both task effectiveness and efficiency. 

---
# Tree Reward-Aligned Search for TReASURe in Masked Diffusion Language Models 

**Authors**: Zichao Yu, Ming Li, Wenyi Zhang, Weiguo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23146)  

**Abstract**: Tree search has recently emerged as a powerful framework for aligning generative models with task-specific rewards at test time. Applying tree search to Masked Diffusion Language Models, however, introduces two key challenges: (i) parallel unmasking yields highly correlated branches, limiting exploration, and (ii) reward evaluation via sampled completions produces high-variance estimates, making pruning unstable. We propose TReASURe, a tree-search test-time alignment method that addresses these issues. It introduces (i) UnmaskBranch, a branching strategy based on first-hitting unmasking that diversifies both token content and reveal order with a single model call per parent node, and (ii) ResubstituteScore, a pruning rule that uses deterministic resubstitution to score partially masked sequences with low-variance proxy completions. Theoretically, we quantify branching efficiency gains in NFEs (number of function evaluations), show that the scoring rule approximates the true reward with error bounded by predictive uncertainty, and prove improvements with larger tree widths. Empirically, TReASURe achieves state-of-the-art results on perplexity, linguistic acceptability, and control of sentiment and toxicity, outperforming prior methods under matched compute budgets, with especially strong gains in low-NFE regimes. 

---
# Tagging the Thought: Unlocking Personalization Reasoning via Reinforcement Learning 

**Authors**: Song Jin, Juntian Zhang, Yong Liu, Xun Zhang, Yufei Zhang, Fei Jiang, Guojun Yin, Wei Lin, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23140)  

**Abstract**: Recent advancements have endowed Large Language Models (LLMs) with impressive general reasoning capabilities, yet they often struggle with personalization reasoning - the crucial ability to analyze user history, infer unique preferences, and generate tailored responses. To address this limitation, we introduce TagPR, a novel training framework that significantly enhances an LLM's intrinsic capacity for personalization reasoning through a tagging the thought approach. Our method first develops a data-driven pipeline to automatically generate and semantically label reasoning chains, creating a structured dataset that fosters interpretable reasoning. We then propose a synergistic training strategy that begins with Supervised Fine-Tuning (SFT) on this tagged data to establish foundational reasoning patterns, followed by a multi-stage reinforcement learning (RL) process. This RL phase is guided by a unique composite reward signal, which integrates tag-based constraints and a novel Personalization Reward Model with User Embeddings (PRMU) to achieve fine-grained alignment with user-specific logic. Extensive experiments on the public LaMP benchmark and a self-constructed dataset demonstrate that our approach achieves state-of-the-art results, delivering an average improvement of 32.65% over the base model across all tasks. Our work validates that structured, interpretable reasoning is a highly effective pathway to unlocking genuine personalization capabilities in LLMs. 

---
# Non-Collaborative User Simulators for Tool Agents 

**Authors**: Jeonghoon Shim, Woojung Song, Cheyon Jin, Seungwon KooK, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.23124)  

**Abstract**: Non-Collaborative User Simulators for Tool Agents Download PDF Jeonghoon Shim, Woojung Song, Cheyon Jin, Seungwon KooK, Yohan Jo 19 Sept 2025 (modified: 25 Sept 2025)ICLR 2026 Conference SubmissionConference, AuthorsRevisionsCC BY 4.0 Keywords: Tool Agent, User Simulator, Non-collaborative User, Dialogue Simulation TL;DR: A non-collaborative user simulation method for tool agent.  Tool agents interact with users through multi-turn dialogues to accomplish various tasks. Recent studies have adopted user simulation methods to develop these agents in multi-turn settings. However, existing user simulators tend to be agent-friendly, exhibiting only cooperative behaviors, which fails to train and test agents against non-collaborative users in the real world. To address this, we propose a novel user simulator architecture that simulates four categories of non-collaborative behaviors: requesting unavailable services, digressing into tangential conversations, expressing impatience, and providing incomplete utterances. Our user simulator can simulate challenging and natural non-collaborative behaviors while reliably delivering all intents and information necessary to accomplish the task. Our experiments on MultiWOZ and $\tau$-bench reveal significant performance degradation in state-of-the-art tool agents when encountering non-collaborative users. We provide detailed analyses of agents' weaknesses under each non-collaborative condition, such as escalated hallucinations and dialogue breakdowns. Ultimately, we contribute an easily extensible user simulation framework to help the research community develop tool agents and preemptively diagnose them under challenging real-world conditions within their own services. 

---
# How to Make Large Language Models Generate 100% Valid Molecules? 

**Authors**: Wen Tao, Jing Tang, Alvin Chan, Bryan Hooi, Baolong Bi, Nanyun Peng, Yuansheng Liu, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23099)  

**Abstract**: Molecule generation is key to drug discovery and materials science, enabling the design of novel compounds with specific properties. Large language models (LLMs) can learn to perform a wide range of tasks from just a few examples. However, generating valid molecules using representations like SMILES is challenging for LLMs in few-shot settings. In this work, we explore how LLMs can generate 100% valid molecules. We evaluate whether LLMs can use SELFIES, a representation where every string corresponds to a valid molecule, for valid molecule generation but find that LLMs perform worse with SELFIES than with SMILES. We then examine LLMs' ability to correct invalid SMILES and find their capacity limited. Finally, we introduce SmiSelf, a cross-chemical language framework for invalid SMILES correction. SmiSelf converts invalid SMILES to SELFIES using grammatical rules, leveraging SELFIES' mechanisms to correct the invalid SMILES. Experiments show that SmiSelf ensures 100% validity while preserving molecular characteristics and maintaining or even enhancing performance on other metrics. SmiSelf helps expand LLMs' practical applications in biomedicine and is compatible with all SMILES-based generative models. Code is available at this https URL. 

---
# d$^2$Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching 

**Authors**: Yuchu Jiang, Yue Cai, Xiangzhong Luo, Jiale Fu, Jiarui Wang, Chonghan Liu, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23094)  

**Abstract**: Diffusion-based large language models (dLLMs), despite their promising performance, still suffer from inferior inference efficiency. This is because dLLMs rely on bidirectional attention and cannot directly benefit from the standard key-value (KV) cache as autoregressive models (ARMs) do. To tackle this issue, we introduce \textit{Dual aDaptive Cache} (d$^2$Cache), which is a training-free approximate KV cache framework for accelerating dLLM inference. d$^2$Cache features a two-stage fine-grained selection strategy to identify tokens and adaptively update their KV states at each decoding step, while caching the KV states of the remaining tokens for reuse. Furthermore, d$^2$Cache naturally offers a more reliable decoding alternative, which can enable quasi left-to-right generation and mitigate premature overconfidence in tokens at the end of the sequence. Extensive experimental results on two representative dLLMs (\ie, LLaDA and Dream) demonstrate that d$^2$Cache not only achieves substantial inference speedups, but also yields consistent improvements in generation quality. The code is available at this https URL. 

---
# The Geometry of Creative Variability: How Credal Sets Expose Calibration Gaps in Language Models 

**Authors**: Esteban Garces Arias, Julian Rodemann, Christian Heumann  

**Link**: [PDF](https://arxiv.org/pdf/2509.23088)  

**Abstract**: Understanding uncertainty in large language models remains a fundamental challenge, particularly in creative tasks where multiple valid outputs exist. We present a geometric framework using credal sets - convex hulls of probability distributions - to quantify and decompose uncertainty in neural text generation, calibrated against human creative variation. Analyzing 500 creative writing prompts from the WritingPrompts dataset with 10 unique human continuations each, we evaluate four language models across five decoding strategies, generating 100,000 stories. Our credal set analysis reveals substantial gaps in capturing human creative variation, with the best model-human calibration reaching only 0.434 (Gemma-2B with temperature 0.7). We decompose total uncertainty into epistemic and aleatoric components, finding that the choice of decoding strategy contributes 39.4% to 72.0% of total epistemic uncertainty. Model scale shows weak correlation with calibration quality and no significant difference exists between base and instruction-tuned models in calibration quality. Our geometric framework provides actionable insights for improving generation systems for human-AI creative alignment. We release our complete experimental framework. 

---
# From Evidence to Trajectory: Abductive Reasoning Path Synthesis for Training Retrieval-Augmented Generation Agents 

**Authors**: Muzhi Li, Jinhu Qi, Yihong Wu, Minghao Zhao, Liheng Ma, Yifan Li, Xinyu Wang, Yingxue Zhang, Ho-fung Leung, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2509.23071)  

**Abstract**: Retrieval-augmented generation agents development is hindered by the lack of process-level supervision to effectively guide agentic capabilities like task decomposition, retriever invocation, and stepwise decision-making. While reinforcement learning offers a potential solution, it suffers from sparse rewards and the limited reasoning capabilities of large language models (LLMs). Meanwhile, existing data synthesis methods only produce chain-of-thought rationales and fail to model environmental interactions. In this paper, we propose EviPath, an evidence-anchored reasoning path synthesis paradigm for RAG agent development. EviPath comprises: (i) Abductive Subtask Planning, which decomposes the problem into sub-questions and iteratively plans an optimal solution path based on the dependencies between them; (ii) Faithful Sub-question Answering, which uses supporting evidence to construct a proxy environment to generate reasoning thoughts and answers for each sub-question; and (iii) Conversational Fine-Tuning, which formats the complete agent-environment interaction trajectory into a dialogue format suitable for Supervised Fine-Tuning. EviPath allows LLMs to learn complex reasoning and tool-use capabilities directly from synthesized data. Extensive experiments on widely-used question-answering benchmarks show that an 8B parameter model trained with EviPath-synthesized data significantly and consistently outperforms state-of-the-art baselines with a double-digit absolute EM gain of 14.7% in open-domain question answering. 

---
# Semantic Voting: A Self-Evaluation-Free Approach for Efficient LLM Self-Improvement on Unverifiable Open-ended Tasks 

**Authors**: Chunyang Jiang, Yonggang Zhang, Yiyang Cai, Chi-Min Chan, Yulong Liu, Mingming Chen, Wei Xue, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.23067)  

**Abstract**: The rising cost of acquiring supervised data has driven significant interest in self-improvement for large language models (LLMs). Straightforward unsupervised signals like majority voting have proven effective in generating pseudo-labels for verifiable tasks, while their applicability to unverifiable tasks (e.g., translation) is limited by the open-ended character of responses. As a result, self-evaluation mechanisms (e.g., self-judging and entropy minimization) are predominantly used to derive pseudo-labels. However, self-evaluation relying on LLMs typically incurs high computational overhead and introduces overconfidence issues due to intrinsic biases. To address these challenges, we propose a novel self-evaluation-free approach for unverifiable tasks, designed for lightweight yet effective self-improvement. Inspired by majority voting commonly employed in verifiable tasks, we propose semantic voting as a novel mechanism that relaxes the principle of hard matching (i.e., exact matching) toward soft matching (i.e., semantic similarity). Soft matching is achieved by leveraging a lightweight sentence embedding model to quantify semantic similarity, thereby mitigating excessive computational burden and intrinsic bias-associated limitations of self-evaluation. Comprehensive experiments demonstrate that our method achieves substantial gains in computational efficiency and overall better performance than self-evaluation methods across diverse model architectures and tasks. 

---
# Peacemaker or Troublemaker: How Sycophancy Shapes Multi-Agent Debate 

**Authors**: Binwei Yao, Chao Shang, Wanyu Du, Jianfeng He, Ruixue Lian, Yi Zhang, Hang Su, Sandesh Swamy, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23055)  

**Abstract**: Large language models (LLMs) often display sycophancy, a tendency toward excessive agreeability. This behavior poses significant challenges for multi-agent debating systems (MADS) that rely on productive disagreement to refine arguments and foster innovative thinking. LLMs' inherent sycophancy can collapse debates into premature consensus, potentially undermining the benefits of multi-agent debate. While prior studies focus on user--LLM sycophancy, the impact of inter-agent sycophancy in debate remains poorly understood. To address this gap, we introduce the first operational framework that (1) proposes a formal definition of sycophancy specific to MADS settings, (2) develops new metrics to evaluate the agent sycophancy level and its impact on information exchange in MADS, and (3) systematically investigates how varying levels of sycophancy across agent roles (debaters and judges) affects outcomes in both decentralized and centralized debate frameworks. Our findings reveal that sycophancy is a core failure mode that amplifies disagreement collapse before reaching a correct conclusion in multi-agent debates, yields lower accuracy than single-agent baselines, and arises from distinct debater-driven and judge-driven failure modes. Building on these findings, we propose actionable design principles for MADS, effectively balancing productive disagreement with cooperation in agent interactions. 

---
# Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents 

**Authors**: Yaorui Shi, Yuxin Chen, Siyuan Wang, Sihang Li, Hengxing Cai, Qi Gu, Xiang Wang, An Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23040)  

**Abstract**: Large language models face challenges in long-context question answering, where key evidence of a query may be dispersed across millions of tokens. Existing works equip large language models with a memory corpus that is dynamically updated during a single-pass document scan, also known as the "memorize while reading" methods. While this approach scales efficiently, it suffers from irreversible forward-only processing, information loss through overwriting, and sparse reinforcement learning signals. To tackle these challenges, we present ReMemR1, a memory-augmented agent with callback-enhanced memory that allows selective retrieval from the entire memory history and allows non-linear reasoning and revisiting of early evidence. To further strengthen training, we propose Reinforcement Learning with Multi-Level Rewards (RLMLR), which combines final-answer rewards with dense, step-level signals that guide effective memory use. Together, these contributions mitigate information degradation, improve supervision, and support multi-hop memory utilizing. Experiments on long-document QA show significant gains over existing memory-based approaches, which validates ReMemR1 as an effective solution for long-context reasoning agents. 

---
# AI Brown and AI Koditex: LLM-Generated Corpora Comparable to Traditional Corpora of English and Czech Texts 

**Authors**: Jiří Milička, Anna Marklová, Václav Cvrček  

**Link**: [PDF](https://arxiv.org/pdf/2509.22996)  

**Abstract**: This article presents two corpora of English and Czech texts generated with large language models (LLMs). The motivation is to create a resource for comparing human-written texts with LLM-generated text linguistically. Emphasis was placed on ensuring these resources are multi-genre and rich in terms of topics, authors, and text types, while maintaining comparability with existing human-created corpora. These generated corpora replicate reference human corpora: BE21 by Paul Baker, which is a modern version of the original Brown Corpus, and Koditex corpus that also follows the Brown Corpus tradition but in Czech. The new corpora were generated using models from OpenAI, Anthropic, Alphabet, Meta, and DeepSeek, ranging from GPT-3 (davinci-002) to GPT-4.5, and are tagged according to the Universal Dependencies standard (i.e., they are tokenized, lemmatized, and morphologically and syntactically annotated). The subcorpus size varies according to the model used (the English part contains on average 864k tokens per model, 27M tokens altogether, the Czech partcontains on average 768k tokens per model, 21.5M tokens altogether). The corpora are freely available for download under the CC BY 4.0 license (the annotated data are under CC BY-NC-SA 4.0 licence) and are also accessible through the search interface of the Czech National Corpus. 

---
# ADAM: A Diverse Archive of Mankind for Evaluating and Enhancing LLMs in Biographical Reasoning 

**Authors**: Jasin Cekinmez, Omid Ghahroodi, Saad Fowad Chandle, Dhiman Gupta, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2509.22991)  

**Abstract**: We introduce ADAM (A Diverse Archive of Mankind), a framework for evaluating and improving multimodal large language models (MLLMs) in biographical reasoning. To the best of our knowledge, this is the first work to systematically examine LLM capabilities in biography, a critical yet underexplored dimension of factual knowledge. At its core, AdamDB is a multilingual and multimodal dataset covering over 4 million individuals across geography, time, and profession, while AdamBench provides cognitively structured evaluations based on Bloom's taxonomy, spanning six reasoning levels in both English and native languages. To address hallucinations, particularly for lesser-known individuals, we propose AdamRAG, a retrieval-augmented generation system tailored to biographical contexts. Experiments show that AdamRAG substantially improves open-source models and modestly benefits closed-source ones, with the largest gains on lower-order reasoning. Popularity strongly mediates accuracy, and multimodal input via face images offers smaller, less consistent improvements than retrieval. ADAM establishes the first benchmark and framework for cognitively, culturally, and multimodally grounded biographical evaluation, advancing the development of multilingual, accurate, and hallucination-resistant MLLMs. 

---
# Same Content, Different Representations: A Controlled Study for Table QA 

**Authors**: Yue Zhang, Seiji Maekawa, Nikita Bhutani  

**Link**: [PDF](https://arxiv.org/pdf/2509.22983)  

**Abstract**: Table Question Answering (Table QA) in real-world settings must operate over both structured databases and semi-structured tables containing textual fields. However, existing benchmarks are tied to fixed data formats and have not systematically examined how representation itself affects model performance. We present the first controlled study that isolates the role of table representation by holding content constant while varying structure. Using a verbalization pipeline, we generate paired structured and semi-structured tables, enabling direct comparisons across modeling paradigms. To support detailed analysis, we introduce a diagnostic benchmark with splits along table size, join requirements, query complexity, and schema quality. Our experiments reveal consistent trade-offs: SQL-based methods achieve high accuracy on structured inputs but degrade on semi-structured data, LLMs exhibit flexibility but reduced precision, and hybrid approaches strike a balance, particularly under noisy schemas. These effects intensify with larger tables and more complex queries. Ultimately, no single method excels across all conditions, and we highlight the central role of representation in shaping Table QA performance. Our findings provide actionable insights for model selection and design, paving the way for more robust hybrid approaches suited for diverse real-world data formats. 

---
# Emergent morpho-phonological representations in self-supervised speech models 

**Authors**: Jon Gauthier, Canaan Breiss, Matthew Leonard, Edward F. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22973)  

**Abstract**: Self-supervised speech models can be trained to efficiently recognize spoken words in naturalistic, noisy environments. However, we do not understand the types of linguistic representations these models use to accomplish this task. To address this question, we study how S3M variants optimized for word recognition represent phonological and morphological phenomena in frequent English noun and verb inflections. We find that their representations exhibit a global linear geometry which can be used to link English nouns and verbs to their regular inflected forms.
This geometric structure does not directly track phonological or morphological units. Instead, it tracks the regular distributional relationships linking many word pairs in the English lexicon -- often, but not always, due to morphological inflection. These findings point to candidate representational strategies that may support human spoken word recognition, challenging the presumed necessity of distinct linguistic representations of phonology and morphology. 

---
# What Matters More For In-Context Learning under Matched Compute Budgets: Pretraining on Natural Text or Incorporating Targeted Synthetic Examples? 

**Authors**: Mohammed Sabry, Anya Belz  

**Link**: [PDF](https://arxiv.org/pdf/2509.22947)  

**Abstract**: Does explicitly exercising the induction circuit during pretraining improve in-context learning (ICL), or is natural text sufficient when compute is held constant (iso-FLOPs)? To test whether targeted synthetic data can accelerate induction-head emergence and enhance ICL, we introduce Bi-Induct, a lightweight curriculum that injects forward-copy (Induction), backward-copy (Anti), or a balanced mix into the pretraining stream. We train models from 0.13B to 1B parameters under iso-FLOPs, evaluating (i) few-shot ICL benchmarks, (ii) head-level telemetry, and (iii) held-out language modeling perplexity. Our findings challenge the assumption that early induction circuit activation directly improves ICL. While Bi-Induct accelerates induction-head emergence at small scales, this does not consistently yield stronger generalization. On standard LM benchmarks, Bi-Induct matches natural-only training; on function-style ICL probes, the 1B natural-only performs best. Stress tests (e.g., label permutation, HITS@1 vs. HITS@3, 1 vs. 10 shots) preserve these trends. Telemetry shows larger natural-only models develop broader, earlier induction heads without explicit induction patterns. Anti-induction data fails to elicit meaningful activation. Perplexity penalties from synthetic data shrink with scale, suggesting larger models can absorb non-natural patterns with minimal cost. Crucially, ablating the top 2% of induction heads degrades ICL more than random ablations, especially for natural-only models, indicating more centralized, load-bearing circuits. Bi-Induct variants exhibit more redundant induction activity, implying different circuit utilization. Overall, inducing activation is not sufficient: ICL gains depend on these circuits becoming functionally necessary. These results underscore mechanism-aware pretraining diagnostics and data mixtures that foster load-bearing, not merely present, structure. 

---
# LLMs Behind the Scenes: Enabling Narrative Scene Illustration 

**Authors**: Melissa Roemmele, John Joon Young Chung, Taewook Kim, Yuqian Sun, Alex Calderwood, Max Kreminski  

**Link**: [PDF](https://arxiv.org/pdf/2509.22940)  

**Abstract**: Generative AI has established the opportunity to readily transform content from one medium to another. This capability is especially powerful for storytelling, where visual illustrations can illuminate a story originally expressed in text. In this paper, we focus on the task of narrative scene illustration, which involves automatically generating an image depicting a scene in a story. Motivated by recent progress on text-to-image models, we consider a pipeline that uses LLMs as an interface for prompting text-to-image models to generate scene illustrations given raw story text. We apply variations of this pipeline to a prominent story corpus in order to synthesize illustrations for scenes in these stories. We conduct a human annotation task to obtain pairwise quality judgments for these illustrations. The outcome of this process is the SceneIllustrations dataset, which we release as a new resource for future work on cross-modal narrative transformation. Through our analysis of this dataset and experiments modeling illustration quality, we demonstrate that LLMs can effectively verbalize scene knowledge implicitly evoked by story text. Moreover, this capability is impactful for generating and evaluating illustrations. 

---
# Large language models management of medications: three performance analyses 

**Authors**: Kelli Henry, Steven Xu, Kaitlin Blotske, Moriah Cargile, Erin F. Barreto, Brian Murray, Susan Smith, Seth R. Bauer, Yanjun Gao, Tianming Liu, Andrea Sikora  

**Link**: [PDF](https://arxiv.org/pdf/2509.22926)  

**Abstract**: Background: Large language models (LLMs) can be useful in diagnosing medical conditions, but few studies have evaluated their consistency in recommending appropriate medication regimens. The purpose of this evaluation was to test GPT-4o on three medication benchmarking tests including mapping a drug name to its correct formulation, identifying drug-drug interactions using both its internal knowledge and using a web search, and preparing a medication order sentence after being given the medication name. Methods: Using GTP-4o three experiments were completed. Accuracy was quantified by computing cosine similarity on TF-IDF vectors, normalized Levenshtein similarity, and ROUGE-1/ROUGE-L F1 between each response and its reference string or by manual evaluation by clinicians. Results: GPT-4o performed poorly on drug-formulation matching, with frequent omissions of available drug formulations (mean 1.23 per medication) and hallucinations of formulations that do not exist (mean 1.14 per medication). Only 49% of tested medications were correctly matched to all available formulations. Accuracy was decreased for medications with more formulations (p<0.0001). GPT-4o was also inconsistent at identifying drug-drug-interactions, although it had better performance with the search-augmented assessment compared to its internal knowledge (54.7% vs. 69.2%, p=0.013). However, allowing a web-search worsened performance when there was no drug-drug interaction (median % correct 100% vs. 40%, p<0.001). Finally, GPT-4o performed moderately with preparing a medication order sentence, with only 65.8% of medication order sentences containing no medication or abbreviation errors. Conclusions: Model performance was overall poor for all tests. This highlights the need for domain-specific training through clinician-annotated datasets and a comprehensive evaluation framework for benchmarking performance. 

---
# Extract-0: A Specialized Language Model for Document Information Extraction 

**Authors**: Henrique Godoy  

**Link**: [PDF](https://arxiv.org/pdf/2509.22906)  

**Abstract**: This paper presents Extract-0, a 7-billion parameter language model specifically optimized for document information extraction that achieves performance exceeding models with parameter counts several orders of magnitude larger. Through a novel combination of synthetic data generation, supervised fine-tuning with Low-Rank Adaptation (LoRA), and reinforcement learning via Group Relative Policy Optimization (GRPO), Extract-0 achieves a mean reward of 0.573 on a benchmark of 1,000 diverse document extraction tasks, outperforming GPT-4.1 (0.457), o3 (0.464), and GPT-4.1-2025 (0.459). The training methodology employs a memory-preserving synthetic data generation pipeline that produces 280,128 training examples from diverse document sources, followed by parameterefficient fine-tuning that modifies only 0.53% of model weights (40.4M out of 7.66B parameters). The reinforcement learning phase introduces a novel semantic similarity-based reward function that handles the inherent ambiguity in information extraction tasks. This research demonstrates that task-specific optimization can yield models that surpass general-purpose systems while requiring substantially fewer computational resource. 

---
# Infusing Theory of Mind into Socially Intelligent LLM Agents 

**Authors**: EunJeong Hwang, Yuwei Yin, Giuseppe Carenini, Peter West, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2509.22887)  

**Abstract**: Theory of Mind (ToM)-an understanding of the mental states of others-is a key aspect of human social intelligence, yet, chatbots and LLM-based social agents do not typically integrate it. In this work, we demonstrate that LLMs that explicitly use ToM get better at dialogue, achieving goals more effectively. After showing that simply prompting models to generate mental states between dialogue turns already provides significant benefit, we further introduce ToMAgent (ToMA), a ToM-focused dialogue agent. ToMA is trained by pairing ToM with dialogue lookahead to produce mental states that are maximally useful for achieving dialogue goals. Experiments on the Sotopia interactive social evaluation benchmark demonstrate the effectiveness of our method over a range of baselines. Comprehensive analysis shows that ToMA exhibits more strategic, goal-oriented reasoning behaviors, which enable long-horizon adaptation, while maintaining better relationships with their partners. Our results suggest a step forward in integrating ToM for building socially intelligent LLM agents. 

---
# HEART: Emotionally-driven test-time scaling of Language Models 

**Authors**: Gabriela Pinto, Palash Goyal, Yiwen Song, Souradip Chakraborty, Zifeng Wang, Tomas Pfister, Hamid Palangi  

**Link**: [PDF](https://arxiv.org/pdf/2509.22876)  

**Abstract**: Test-time scaling has shown considerable success in improving the performance of language models on complex reasoning tasks without requiring fine-tuning. However, current strategies such as self-reflection primarily focus on logical or structural refinement. They do not leverage the guiding potential of affective feedback. Inspired by psychological research showing that emotions can modulate cognitive performance, we introduce HEART--a novel framework that uses emotionally-driven prompts for iterative self-correction. HEART provides feedback on a model's incorrect response using a curated set of concise, emotionally charged phrases based on the six universal emotions categorized by Dr. Paul Ekman. By systematically varying the emotional tone of the feedback across iterations, our method guides the model to escape flawed reasoning paths and explore more promising alternatives. We evaluate our framework on challenging reasoning benchmarks including OlympiadBench, Humanity's Last Exam, and SimpleQA. Our results reveal a significant new phenomenon: when guided by an oracle verifier, this affective iteration protocol unlocks significantly deeper reasoning, leading to consistent and substantial increases in accuracy over state-of-the-art baselines with the same verifier. However, we also identify a critical bottleneck for practical deployment. In a verifier-free setting, it struggles to harness these gains consistently, highlighting as a key challenge for future work. Our findings suggest that the next frontier in machine reasoning may lie not just in refining logic, but also in understanding and leveraging the `HEART' of the models. 

---
# Lexicon-Enriched Graph Modeling for Arabic Document Readability Prediction 

**Authors**: Passant Elchafei, Mayar Osama, Mohamed Rageh, Mervat Abuelkheir  

**Link**: [PDF](https://arxiv.org/pdf/2509.22870)  

**Abstract**: We present a graph-based approach enriched with lexicons to predict document-level readability in Arabic, developed as part of the Constrained Track of the BAREC Shared Task 2025. Our system models each document as a sentence-level graph, where nodes represent sentences and lemmas, and edges capture linguistic relationships such as lexical co-occurrence and class membership. Sentence nodes are enriched with features from the SAMER lexicon as well as contextual embeddings from the Arabic transformer model. The graph neural network (GNN) and transformer sentence encoder are trained as two independent branches, and their predictions are combined via late fusion at inference. For document-level prediction, sentence-level outputs are aggregated using max pooling to reflect the most difficult sentence. Experimental results show that this hybrid method outperforms standalone GNN or transformer branches across multiple readability metrics. Overall, the findings highlight that fusion offers advantages at the document level, but the GNN-only approach remains stronger for precise prediction of sentence-level readability. 

---
# The Bias is in the Details: An Assessment of Cognitive Bias in LLMs 

**Authors**: R. Alexander Knipper, Charles S. Knipper, Kaiqi Zhang, Valerie Sims, Clint Bowers, Santu Karmaker  

**Link**: [PDF](https://arxiv.org/pdf/2509.22856)  

**Abstract**: As Large Language Models (LLMs) are increasingly embedded in real-world decision-making processes, it becomes crucial to examine the extent to which they exhibit cognitive biases. Extensively studied in the field of psychology, cognitive biases appear as systematic distortions commonly observed in human judgments. This paper presents a large-scale evaluation of eight well-established cognitive biases across 45 LLMs, analyzing over 2.8 million LLM responses generated through controlled prompt variations. To achieve this, we introduce a novel evaluation framework based on multiple-choice tasks, hand-curate a dataset of 220 decision scenarios targeting fundamental cognitive biases in collaboration with psychologists, and propose a scalable approach for generating diverse prompts from human-authored scenario templates. Our analysis shows that LLMs exhibit bias-consistent behavior in 17.8-57.3% of instances across a range of judgment and decision-making contexts targeting anchoring, availability, confirmation, framing, interpretation, overattribution, prospect theory, and representativeness biases. We find that both model size and prompt specificity play a significant role on bias susceptibility as follows: larger size (>32B parameters) can reduce bias in 39.5% of cases, while higher prompt detail reduces most biases by up to 14.9%, except in one case (Overattribution), which is exacerbated by up to 8.8%. 

---
# Towards Generalizable Implicit In-Context Learning with Attention Routing 

**Authors**: Jiaqian Li, Yanshu Li, Ligong Han, Ruixiang Tang, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22854)  

**Abstract**: Implicit in-context learning (ICL) has newly emerged as a promising paradigm that simulates ICL behaviors in the representation space of Large Language Models (LLMs), aiming to attain few-shot performance at zero-shot cost. However, existing approaches largely rely on injecting shift vectors into residual flows, which are typically constructed from labeled demonstrations or task-specific alignment. Such designs fall short of utilizing the structural mechanisms underlying ICL and suffer from limited generalizability. To address this, we propose In-Context Routing (ICR), a novel implicit ICL method that internalizes generalizable ICL patterns at the attention logits level. It extracts reusable structural directions that emerge during ICL and employs a learnable input-conditioned router to modulate attention logits accordingly, enabling a train-once-and-reuse framework. We evaluate ICR on 12 real-world datasets spanning diverse domains and multiple LLMs. The results show that ICR consistently outperforms prior implicit ICL methods that require task-specific retrieval or training, while demonstrating robust generalization to out-of-domain tasks where existing methods struggle. These findings position ICR to push the boundary of ICL's practical value. 

---
# Learning to Detect Relevant Contexts and Knowledge for Response Selection in Retrieval-based Dialogue Systems 

**Authors**: Kai Hua, Zhiyuan Feng, Chongyang Tao, Rui Yan, Lu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22845)  

**Abstract**: Recently, knowledge-grounded conversations in the open domain gain great attention from researchers. Existing works on retrieval-based dialogue systems have paid tremendous efforts to utilize neural networks to build a matching model, where all of the context and knowledge contents are used to match the response candidate with various representation methods. Actually, different parts of the context and knowledge are differentially important for recognizing the proper response candidate, as many utterances are useless due to the topic shift. Those excessive useless information in the context and knowledge can influence the matching process and leads to inferior performance. To address this problem, we propose a multi-turn \textbf{R}esponse \textbf{S}election \textbf{M}odel that can \textbf{D}etect the relevant parts of the \textbf{C}ontext and \textbf{K}nowledge collection (\textbf{RSM-DCK}). Our model first uses the recent context as a query to pre-select relevant parts of the context and knowledge collection at the word-level and utterance-level semantics. Further, the response candidate interacts with the selected context and knowledge collection respectively. In the end, The fused representation of the context and response candidate is utilized to post-select the relevant parts of the knowledge collection more confidently for matching. We test our proposed model on two benchmark datasets. Evaluation results indicate that our model achieves better performance than the existing methods, and can effectively detect the relevant context and knowledge for response selection. 

---
# ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents 

**Authors**: Hwan Chang, Yonghyun Jun, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.22830)  

**Abstract**: The growing deployment of large language model (LLM) based agents that interact with external environments has created new attack surfaces for adversarial manipulation. One major threat is indirect prompt injection, where attackers embed malicious instructions in external environment output, causing agents to interpret and execute them as if they were legitimate prompts. While previous research has focused primarily on plain-text injection attacks, we find a significant yet underexplored vulnerability: LLMs' dependence on structured chat templates and their susceptibility to contextual manipulation through persuasive multi-turn dialogues. To this end, we introduce ChatInject, an attack that formats malicious payloads to mimic native chat templates, thereby exploiting the model's inherent instruction-following tendencies. Building on this foundation, we develop a persuasion-driven Multi-turn variant that primes the agent across conversational turns to accept and execute otherwise suspicious actions. Through comprehensive experiments across frontier LLMs, we demonstrate three critical findings: (1) ChatInject achieves significantly higher average attack success rates than traditional prompt injection methods, improving from 5.18% to 32.05% on AgentDojo and from 15.13% to 45.90% on InjecAgent, with multi-turn dialogues showing particularly strong performance at average 52.33% success rate on InjecAgent, (2) chat-template-based payloads demonstrate strong transferability across models and remain effective even against closed-source LLMs, despite their unknown template structures, and (3) existing prompt-based defenses are largely ineffective against this attack approach, especially against Multi-turn variants. These findings highlight vulnerabilities in current agent systems. 

---
# Critique-Coder: Enhancing Coder Models by Critique Reinforcement Learning 

**Authors**: Chi Ruan, Dongfu Jiang, Yubo Wang, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22824)  

**Abstract**: Reinforcement Learning (RL) has emerged as a popular training paradigm, particularly when paired with reasoning models. While effective, it primarily focuses on generating responses and lacks mechanisms to explicitly foster critique or reflection. Several recent studies, like Critique-Fine-Tuning (CFT) and Critique-Guided-Distillation (CGD) have shown the benefits of explicitly teaching LLMs how to critique. Motivated by them, we propose Critique Reinforcement Learning (CRL), where the model is tasked with generating a critique for a given (question, solution) pair. The reward is determined solely by whether the final judgment label $c \in \{\texttt{True}, \texttt{False}\}$ of the generated critique aligns with the ground-truth judgment $c^*$. Building on this point, we introduce \textsc{Critique-Coder}, which is trained on a hybrid of RL and CRL by substituting 20\% of the standard RL data with CRL data. We fine-tune multiple models (\textsc{Critique-Coder}) and evaluate them on different benchmarks to show their advantages over RL-only models. We show that \textsc{Critique-Coder} consistently outperforms RL-only baselines on all the evaluated benchmarks. Notably, our \textsc{Critique-Coder-8B} can reach over 60\% on LiveCodeBench (v5), outperforming other reasoning models like DeepCoder-14B and GPT-o1. Beyond code generation, \textsc{Critique-Coder} also demonstrates enhanced general reasoning abilities, as evidenced by its better performance on logic reasoning tasks from the BBEH dataset. This indicates that the application of CRL on coding datasets enhances general reasoning and critique abilities, which are transferable across a broad range of tasks. Hence, we believe that CRL works as a great complement to standard RL for LLM reasoning. 

---
# EditGRPO: Reinforcement Learning with Post -Rollout Edits for Clinically Accurate Chest X-Ray Report Generation 

**Authors**: Kai Zhang, Christopher Malon, Lichao Sun, Martin Renqiang Min  

**Link**: [PDF](https://arxiv.org/pdf/2509.22812)  

**Abstract**: Radiology report generation requires advanced medical image analysis, effective temporal reasoning, and accurate text generation. Although recent innovations, particularly multimodal large language models (MLLMs), have shown improved performance, their supervised fine-tuning (SFT) objective is not explicitly aligned with clinical efficacy. In this work, we introduce EditGRPO, a mixed-policy reinforcement learning (RL) algorithm designed specifically to optimize the generation through clinically motivated rewards. EditGRPO integrates on-policy exploration with off-policy guidance by injecting sentence-level detailed corrections during training rollouts. This mixed-policy approach addresses the exploration dilemma and sampling efficiency issues typically encountered in RL. Applied to a Qwen2.5-VL-3B MLLM initialized with supervised fine-tuning (SFT), EditGRPO outperforms both SFT and vanilla GRPO baselines, achieving an average improvement of 3.4% in CheXbert, GREEN, Radgraph, and RATEScore metrics across four major chest X-ray report generation datasets. Notably, EditGRPO also demonstrates superior out-of-domain generalization, with an average performance gain of 5.9% on unseen datasets. 

---
# ArFake: A Multi-Dialect Benchmark and Baselines for Arabic Spoof-Speech Detection 

**Authors**: Mohamed Maged, Alhassan Ehab, Ali Mekky, Besher Hassan, Shady Shehata  

**Link**: [PDF](https://arxiv.org/pdf/2509.22808)  

**Abstract**: With the rise of generative text-to-speech models, distinguishing between real and synthetic speech has become challenging, especially for Arabic that have received limited research attention. Most spoof detection efforts have focused on English, leaving a significant gap for Arabic and its many dialects. In this work, we introduce the first multi-dialect Arabic spoofed speech dataset. To evaluate the difficulty of the synthesized audio from each model and determine which produces the most challenging samples, we aimed to guide the construction of our final dataset either by merging audios from multiple models or by selecting the best-performing model, we conducted an evaluation pipeline that included training classifiers using two approaches: modern embedding-based methods combined with classifier heads; classical machine learning algorithms applied to MFCC features; and the RawNet2 architecture. The pipeline further incorporated the calculation of Mean Opinion Score based on human ratings, as well as processing both original and synthesized datasets through an Automatic Speech Recognition model to measure the Word Error Rate. Our results demonstrate that FishSpeech outperforms other TTS models in Arabic voice cloning on the Casablanca corpus, producing more realistic and challenging synthetic speech samples. However, relying on a single TTS for dataset creation may limit generalizability. 

---
# ML2B: Multi-Lingual ML Benchmark For AutoML 

**Authors**: Ekaterina Trofimova, Zosia Shamina, Maria Selifanova, Artem Zaitsev, Remi Savchuk, Maxim Minets, Daria Ozerova, Emil Sataev, Denis Zuenko, Andrey E. Ustyuzhanin  

**Link**: [PDF](https://arxiv.org/pdf/2509.22768)  

**Abstract**: Large language models (LLMs) have recently demonstrated strong capabilities in generating machine learning (ML) code, enabling end-to-end pipeline construction from natural language instructions. However, existing benchmarks for ML code generation are mainly restricted to English, overlooking the global and multilingual nature of ML research and practice. To address this gap, we present ML2B, the first benchmark for evaluating multilingual ML code generation. ML2B consists of 30 Kaggle competitions translated into 13 natural languages, covering tabular, text, and image data types, with structured metadata and validated human-reviewed translations. For evaluation, we employ AIDE, an automated framework for end-to-end assessment of data science pipelines, and provide insights into cross-lingual model performance. Our results reveal substantial 15-45% performance degradation on non-English tasks, highlighting critical challenges in multilingual representation learning for code generation. The benchmark, evaluation framework, and comprehensive results are made available through our GitHub repository to facilitate future research in multilingual ML code generation: this https URL. 

---
# MIRAGE: Multi-hop Reasoning with Ambiguity Evaluation for Illusory Questions 

**Authors**: Jeonghyun Park, Ingeol Baek, Seunghyun Yoon, Haeun Jang, Aparna Garimella, Akriti Jain, Nedim Lipka, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.22750)  

**Abstract**: Real-world Multi-hop Question Answering (QA) often involves ambiguity that is inseparable from the reasoning process itself. This ambiguity creates a distinct challenge, where multiple reasoning paths emerge from a single question, each requiring independent resolution. Since each sub-question is ambiguous, the model must resolve ambiguity at every step. Thus, answering a single question requires handling multiple layers of ambiguity throughout the reasoning chain. We find that current Large Language Models (LLMs) struggle in this setting, typically exploring wrong reasoning paths and producing incomplete answers. To facilitate research on multi-hop ambiguity, we introduce MultI-hop Reasoning with AmbiGuity Evaluation for Illusory Questions (MIRAGE), a benchmark designed to analyze and evaluate this challenging intersection of ambiguity interpretation and multi-hop reasoning. MIRAGE contains 1,142 high-quality examples of ambiguous multi-hop questions, categorized under a taxonomy of syntactic, general, and semantic ambiguity, and curated through a rigorous multi-LLM verification pipeline. Our experiments reveal that even state-of-the-art models struggle on MIRAGE, confirming that resolving ambiguity combined with multi-step inference is a distinct and significant challenge. To establish a robust baseline, we propose CLarifying Ambiguity with a Reasoning and InstructiON (CLARION), a multi-agent framework that significantly outperforms existing approaches on MIRAGE, paving the way for more adaptive and robust reasoning systems. 

---
# Painless Activation Steering: An Automated, Lightweight Approach for Post-Training Large Language Models 

**Authors**: Sasha Cui, Zhongren Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22739)  

**Abstract**: Language models (LMs) are typically post-trained for desired capabilities and behaviors via weight-based or prompt-based steering, but the former is time-consuming and expensive, and the latter is not precisely controllable and often requires manual trial-and-error. While activation steering (AS) promises a cheap, fast, and controllable alternative to the two existing post-training methods, current AS techniques require hand-crafted prompt pairs or labor-intensive feature annotation, making them more inconvenient than the plug-and-play methods such as Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT). We introduce Painless Activation Steering (PAS), a family of fully automated methods that make AS readily usable with any given labeled dataset, with no need for prompt construction, feature labeling, or human intervention. We evaluate PAS on three open-weight models (Llama3.1-8B-Instruct, DeepSeek-R1-Distill-8B, and Nous-Hermes-2) and 18 tasks; we find that PAS reliably improves performance for behavior tasks, but not for intelligence-oriented tasks. The introspective variant (iPAS) delivers the strongest causal steering effects (10.1% on Bias, 5.2% on Morality, and 34.8% on Alignment). We also show PAS delivers additional gains on top of In-Context Learning (ICL) and SFT. PAS constructs a fast, lightweight activation vector that can be cheaply trained, easily stored, and activated at will. Our results provide a characterization of where AS helps, where it fails, and how to deploy it as a practical, automated LM post-training option. 

---
# Enabling Approximate Joint Sampling in Diffusion LMs 

**Authors**: Parikshit Bansal, Sujay Sanghavi  

**Link**: [PDF](https://arxiv.org/pdf/2509.22738)  

**Abstract**: In autoregressive language models, each token is sampled by conditioning on all the past tokens; the overall string has thus been sampled from the correct underlying joint distribution represented by the model. In contrast, masked diffusion language models generate text by unmasking tokens out of order and potentially in parallel. Generating an overall string sampled from the correct underlying joint distribution would (again) require exactly one token unmasking in every full-model forward pass. The more tokens unmasked in parallel, the further away the string is from the true joint; this can be seen in the resulting drop in accuracy (but, increase in speed). In this paper we devise a way to {\em approximately} sample multiple tokens from the joint distribution in a single full-model forward pass; we do so by developing a new lightweight single-layer ``sampler" on top of an existing large diffusion LM. One forward pass of the full model can now be followed by multiple forward passes of only this sampler layer, to yield multiple unmasked tokens. Our sampler is trained to mimic exact joint sampling from the (frozen) full model. We show the effectiveness of our approximate joint sampling for both pretrained-only (Dream-7B-Base) and instruction-tuned (Dream-7B-Instruct) models on language modeling and math \& coding tasks. When four tokens are unmasked for each full-model denoising step, our sampling algorithm achieves a MAUVE score of 0.87 (vs marginal baseline of 0.31) with respect to the true joint distribution. 

---
# Multi-Modal Sentiment Analysis with Dynamic Attention Fusion 

**Authors**: Sadia Abdulhalim, Muaz Albaghdadi, Moshiur Farazi  

**Link**: [PDF](https://arxiv.org/pdf/2509.22729)  

**Abstract**: Traditional sentiment analysis has long been a unimodal task, relying solely on text. This approach overlooks non-verbal cues such as vocal tone and prosody that are essential for capturing true emotional intent. We introduce Dynamic Attention Fusion (DAF), a lightweight framework that combines frozen text embeddings from a pretrained language model with acoustic features from a speech encoder, using an adaptive attention mechanism to weight each modality per utterance. Without any finetuning of the underlying encoders, our proposed DAF model consistently outperforms both static fusion and unimodal baselines on a large multimodal benchmark. We report notable gains in F1-score and reductions in prediction error and perform a variety of ablation studies that support our hypothesis that the dynamic weighting strategy is crucial for modeling emotionally complex inputs. By effectively integrating verbal and non-verbal information, our approach offers a more robust foundation for sentiment prediction and carries broader impact for affective computing applications -- from emotion recognition and mental health assessment to more natural human computer interaction. 

---
# TRUEBench: Can LLM Response Meet Real-world Constraints as Productivity Assistant? 

**Authors**: Jiho Park, Jongyoon Song, Minjin Choi, Kyuho Heo, Taehun Huh, Ji Won Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.22715)  

**Abstract**: Large language models (LLMs) are increasingly integral as productivity assistants, but existing benchmarks fall short in rigorously evaluating their real-world instruction-following capabilities. Current benchmarks often (i) lack sufficient multilinguality, (ii) fail to capture the implicit constraints inherent in user requests, and (iii) overlook the complexities of multi-turn dialogue. To address these critical gaps and provide a more realistic assessment, we introduce TRUEBench (Trustworthy Real-world Usage Evaluation Benchmark)1, a novel benchmark specifically designed for LLM-based productivity assistants. TRUEBench distinguishes itself by featuring input prompts across 12 languages, incorporating intra-instance multilingual instructions, employing rigorous evaluation criteria to capture both explicit and implicit constraints, and including complex multi-turn dialogue scenarios with both accumulating constraints and context switches. Furthermore, to ensure reliability in evaluation, we refined constraints using an LLM validator. Extensive experiments demonstrate that TRUEBench presents significantly greater challenges than existing benchmarks; for instance, a strong model like OpenAI o1 achieved only a 69.07% overall pass rate. TRUEBench offers a demanding and realistic assessment of LLMs in practical productivity settings, highlighting their capabilities and limitations. 

---
# RAR$^2$: Retrieval-Augmented Medical Reasoning via Thought-Driven Retrieval 

**Authors**: Kaishuai Xu, Wenjun Hou, Yi Cheng, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22713)  

**Abstract**: Large Language Models (LLMs) have shown promising performance on diverse medical benchmarks, highlighting their potential in supporting real-world clinical tasks. Retrieval-Augmented Generation (RAG) has emerged as a key approach for mitigating knowledge gaps and hallucinations by incorporating external medical information. However, RAG still struggles with complex medical questions that require intensive reasoning, as surface-level input often fails to reflect the true knowledge needs of the task. Existing methods typically focus on refining queries without explicitly modeling the reasoning process, limiting their ability to retrieve and integrate clinically relevant knowledge. In this work, we propose RAR$^2$, a joint learning framework that improves both Reasoning-Augmented Retrieval and Retrieval-Augmented Reasoning. RAR$^2$ constructs a thought process to uncover implicit knowledge requirements and uses it to guide retrieval and answer generation. We build a training dataset of mixed preference pairs and apply Direct Preference Optimization (DPO) to train the model. Moreover, we design two test-time scaling strategies to explore the boundaries of our framework. Experiments demonstrate the effectiveness of RAR$^2$ across several biomedical question answering datasets, outperforming RAG baselines with or without fine-tuning. 

---
# AccessEval: Benchmarking Disability Bias in Large Language Models 

**Authors**: Srikant Panda, Amit Agarwal, Hitesh Laxmichand Patel  

**Link**: [PDF](https://arxiv.org/pdf/2509.22703)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across diverse domains but often exhibit disparities in how they handle real-life queries. To systematically investigate these effects within various disability contexts, we introduce \textbf{AccessEval (Accessibility Evaluation)}, a benchmark evaluating 21 closed- and open-source LLMs across 6 real-world domains and 9 disability types using paired Neutral and Disability-Aware Queries. We evaluated model outputs with metrics for sentiment, social perception, and factual accuracy.
Our analysis reveals that responses to disability-aware queries tend to have a more negative tone, increased stereotyping, and higher factual error compared to neutral queries. These effects show notable variation by domain and disability type, with disabilities affecting hearing, speech, and mobility disproportionately impacted. These disparities reflect persistent forms of ableism embedded in model behavior.
By examining model performance in real-world decision-making contexts, we better illuminate how such biases can translate into tangible harms for disabled users. This framing helps bridges the gap between technical evaluation and user impact, reinforcing importance of bias mitigation in day-to-day applications. Our dataset is publicly available at: this https URL 

---
# Are you sure? Measuring models bias in content moderation through uncertainty 

**Authors**: Alessandra Urbinati, Mirko Lai, Simona Frenda, Marco Antonio Stranisci  

**Link**: [PDF](https://arxiv.org/pdf/2509.22699)  

**Abstract**: Automatic content moderation is crucial to ensuring safety in social media. Language Model-based classifiers are being increasingly adopted for this task, but it has been shown that they perpetuate racial and social biases. Even if several resources and benchmark corpora have been developed to challenge this issue, measuring the fairness of models in content moderation remains an open issue. In this work, we present an unsupervised approach that benchmarks models on the basis of their uncertainty in classifying messages annotated by people belonging to vulnerable groups. We use uncertainty, computed by means of the conformal prediction technique, as a proxy to analyze the bias of 11 models against women and non-white annotators and observe to what extent it diverges from metrics based on performance, such as the $F_1$ score. The results show that some pre-trained models predict with high accuracy the labels coming from minority groups, even if the confidence in their prediction is low. Therefore, by measuring the confidence of models, we are able to see which groups of annotators are better represented in pre-trained models and lead the debiasing process of these models before their effective use. 

---
# SIRI: Scaling Iterative Reinforcement Learning with Interleaved Compression 

**Authors**: Haoming Wen, Yushi Bai, Juanzi Li, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25176)  

**Abstract**: We introduce SIRI, Scaling Iterative Reinforcement Learning with Interleaved Compression, a simple yet effective RL approach for Large Reasoning Models (LRMs) that enables more efficient and accurate reasoning. Existing studies have observed repetitive thinking patterns in LRMs, and attempts to reduce them often come at the cost of performance. In this paper, we show that this trade-off can be overcome through a training regime that iteratively alternates between compressing and expanding the reasoning budget, by dynamically adjusting the maximum rollout length during training. The compression phase cuts the rollout length, forcing the model to make precise and valuable decisions within a limited context, which effectively reduces redundant tokens and increases reasoning density. The expansion phase then relaxes the length limit, providing space for the model to explore and plan in long-horizon settings. Remarkably, we find that after each compression-expansion cycle, the model's performance improves even as its output length decreases, steadily pushing it closer to the Pareto frontier in the performance-efficiency trade-off. Training on DeepSeek-R1-Distill-Qwen-1.5B, SIRI-low improves performance on AIME24 by 43.2% while reducing token usage by 46.9% after three iterations, and SIRI-high achieves the highest accuracy compared to all other methods (Figure 1). Our findings shed light on the potential of periodically oscillating the LRM's output truncation length during training to dynamically balance exploration and efficiency in reasoning, converging towards an optimal "sweet spot" between the two. Our models are publicly available. 

---
# GSM8K-V: Can Vision Language Models Solve Grade School Math Word Problems in Visual Contexts 

**Authors**: Fan Yuan, Yuchen Yan, Yifan Jiang, Haoran Zhao, Tao Feng, Jinyan Chen, Yanwei Lou, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25160)  

**Abstract**: Vision language models (VLMs) achieve unified modeling of images and text, enabling them to accomplish complex real-world tasks through perception, planning, and reasoning. Among these tasks, reasoning is particularly representative, with mathematical reasoning serving as a prominent example. It highlights the high-level capability of VLMs to comprehend mathematical information in images and to perform sophisticated reasoning. Recently, numerous visual mathematical reasoning benchmarks have been proposed, but they are often restricted to geometry, lack coverage of math word problems, and rarely assess reasoning across multiple images. To address these gaps, we introduce GSM8K-V, a purely visual multi-image mathematical reasoning benchmark. GSM8K-V is built by systematically mapping each sample from the widely used text-based GSM8K into visual form. Through a carefully designed automated image-generation pipeline combined with meticulous human annotation, we curate 1,319 high-quality samples. We evaluate a wide range of open-source and closed-source models on GSM8K-V. Results show that although existing VLMs have nearly saturated performance on text-based GSM8K, there remains substantial room for improvement on GSM8K-V. For example, the best-performing model, Gemini-2.5-Pro, achieves 95.22% accuracy on GSM8K but only 46.93% on GSM8K-V. We conduct a comprehensive analysis of GSM8K-V, examining the limitations of current models as well as potential directions for improvement. GSM8K-V offers a new perspective on visual mathematical reasoning and establishes a benchmark to guide the development of more robust and generalizable VLMs. 

---
# TemMed-Bench: Evaluating Temporal Medical Image Reasoning in Vision-Language Models 

**Authors**: Junyi Zhang, Jia-Chen Gu, Wenbo Hu, Yu Zhou, Robinson Piramuthu, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25143)  

**Abstract**: Existing medical reasoning benchmarks for vision-language models primarily focus on analyzing a patient's condition based on an image from a single visit. However, this setting deviates significantly from real-world clinical practice, where doctors typically refer to a patient's historical conditions to provide a comprehensive assessment by tracking their changes over time. In this paper, we introduce TemMed-Bench, the first benchmark designed for analyzing changes in patients' conditions between different clinical visits, which challenges large vision-language models (LVLMs) to reason over temporal medical images. TemMed-Bench consists of a test set comprising three tasks - visual question-answering (VQA), report generation, and image-pair selection - and a supplementary knowledge corpus of over 17,000 instances. With TemMed-Bench, we conduct an evaluation of six proprietary and six open-source LVLMs. Our results show that most LVLMs lack the ability to analyze patients' condition changes over temporal medical images, and a large proportion perform only at a random-guessing level in the closed-book setting. In contrast, GPT o3, o4-mini and Claude 3.5 Sonnet demonstrate comparatively decent performance, though they have yet to reach the desired level. Furthermore, we explore augmenting the input with both retrieved visual and textual modalities in the medical domain. We also show that multi-modal retrieval augmentation yields notably higher performance gains than no retrieval and textual retrieval alone across most models on our benchmark, with the VQA task showing an average improvement of 2.59%. Overall, we compose a benchmark grounded on real-world clinical practice, and it reveals LVLMs' limitations in temporal medical image reasoning, as well as highlighting the use of multi-modal retrieval augmentation as a potentially promising direction worth exploring to address this challenge. 

---
# ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory 

**Authors**: Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2509.25140)  

**Abstract**: With the growing adoption of large language model agents in persistent real-world roles, they naturally encounter continuous streams of tasks. A key limitation, however, is their failure to learn from the accumulated interaction history, forcing them to discard valuable insights and repeat past errors. We propose ReasoningBank, a novel memory framework that distills generalizable reasoning strategies from an agent's self-judged successful and failed experiences. At test time, an agent retrieves relevant memories from ReasoningBank to inform its interaction and then integrates new learnings back, enabling it to become more capable over time. Building on this powerful experience learner, we further introduce memory-aware test-time scaling (MaTTS), which accelerates and diversifies this learning process by scaling up the agent's interaction experience. By allocating more compute to each task, the agent generates abundant, diverse experiences that provide rich contrastive signals for synthesizing higher-quality memory. The better memory in turn guides more effective scaling, establishing a powerful synergy between memory and test-time scaling. Across web browsing and software engineering benchmarks, ReasoningBank consistently outperforms existing memory mechanisms that store raw trajectories or only successful task routines, improving both effectiveness and efficiency; MaTTS further amplifies these gains. These findings establish memory-driven experience scaling as a new scaling dimension, enabling agents to self-evolve with emergent behaviors naturally arise. 

---
# The Era of Real-World Human Interaction: RL from User Conversations 

**Authors**: Chuanyang Jin, Jing Xu, Bo Liu, Leitian Tao, Olga Golovneva, Tianmin Shu, Wenting Zhao, Xian Li, Jason Weston  

**Link**: [PDF](https://arxiv.org/pdf/2509.25137)  

**Abstract**: We posit that to achieve continual model improvement and multifaceted alignment, future models must learn from natural human interaction. Current conversational models are aligned using pre-annotated, expert-generated human feedback. In this work, we introduce Reinforcement Learning from Human Interaction (RLHI), a paradigm that learns directly from in-the-wild user conversations. We develop two complementary methods: (1) RLHI with User-Guided Rewrites, which revises unsatisfactory model outputs based on users' natural-language follow-up responses, (2) RLHI with User-Based Rewards, which learns via a reward model conditioned on knowledge of the user's long-term interaction history (termed persona). Together, these methods link long-term user personas to turn-level preferences via persona-conditioned preference optimization. Trained on conversations derived from WildChat, both RLHI variants outperform strong baselines in personalization and instruction-following, and similar feedback enhances performance on reasoning benchmarks. These results suggest organic human interaction offers scalable, effective supervision for personalized alignment. 

---
# Rethinking Entropy Regularization in Large Reasoning Models 

**Authors**: Yuxian Jiang, Yafu Li, Guanxu Chen, Dongrui Liu, Yu Cheng, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25133)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has shown great promise in enhancing the reasoning abilities of large reasoning models (LRMs). However, it suffers from a critical issue: entropy collapse and premature convergence. Naive entropy regularization, a common approach for encouraging exploration in the traditional RL literature, fails to address this problem in the context of LRM. Our analysis reveals that this failure stems from the vast action space and long trajectories in LRMs, which easily trigger a global entropy explosion as the model indiscriminately explores all possible actions and states. To address this, we propose SIREN (SelectIve entRopy rEgularizatioN), a method that confines exploration to a meaningful subset of actions and states. SIREN achieves this through a two-step entropy masking mechanism, consisting of a top-p mask and a peak-entropy mask. In addition, regularization is transformed into a self-anchored form to stabilize training. Across five mathematical benchmarks, SIREN attains superior average performance over previous entropy-related RLVR approaches, exemplified by a +6.6 maj@k improvement on AIME24/25 with Qwen2.5-Math-7B. Further analysis confirms that SIREN promotes greater response diversity and maintains entropy at an appropriate level, which helps to preserve the validation pass@k throughout training. This effectively mitigates the premature convergence problem common in RLVR for LRM. 

---
# MGM-Omni: Scaling Omni LLMs to Personalized Long-Horizon Speech 

**Authors**: Chengyao Wang, Zhisheng Zhong, Bohao Peng, Senqiao Yang, Yuqi Liu, Haokun Gui, Bin Xia, Jingyao Li, Bei Yu, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.25131)  

**Abstract**: We present MGM-Omni, a unified Omni LLM for omni-modal understanding and expressive, long-horizon speech generation. Unlike cascaded pipelines that isolate speech synthesis, MGM-Omni adopts a "brain-mouth" design with a dual-track, token-based architecture that cleanly decouples multimodal reasoning from real-time speech generation. This design enables efficient cross-modal interaction and low-latency, streaming speech generation. For understanding, a unified training strategy coupled with a dual audio encoder design enables long-form audio perception across diverse acoustic conditions. For generation, a chunk-based parallel decoding scheme narrows the text speech token-rate gap, accelerating inference and supporting streaming zero-shot voice cloning with stable timbre over extended durations. Compared to concurrent work, MGM-Omni achieves these capabilities with markedly data-efficient training. Extensive experiments demonstrate that MGM-Omni outperforms existing open source models in preserving timbre identity across extended sequences, producing natural and context-aware speech, and achieving superior long-form audio and omnimodal understanding. MGM-Omni establishes an efficient, end-to-end paradigm for omnimodal understanding and controllable, personalised long-horizon speech generation. 

---
# From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones 

**Authors**: Lifan Yuan, Weize Chen, Yuchen Zhang, Ganqu Cui, Hanbin Wang, Ziming You, Ning Ding, Zhiyuan Liu, Maosong Sun, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25123)  

**Abstract**: Does RL teach LLMs genuinely new skills, or does it merely activate existing ones? This question lies at the core of ongoing debates about the role of RL in LLM post-training. On one side, strong empirical results can be achieved with RL even without preceding supervised finetuning; on the other, critics argue that RL contributes little beyond reweighting existing reasoning strategies. This work provides concrete evidence that LLMs can acquire genuinely new skills during RL by composing existing ones, mirroring one of the central mechanisms by which humans acquire new cognitive skills. To mitigate data contamination and other confounding factors, and to allow precise control over task complexity, we develop a synthetic framework for our investigation. Specifically, we define a skill as the ability to infer the output of a string transformation function f(x) given x. When an LLM has already learned f and g prior to RL, our experiments reveal that RL enables it to learn unseen compositions of them h(x)=g(f(x)). Further, this compositional ability generalizes to more difficult problems such as compositions of >2 functions unseen during RL training. Surprisingly, our experiments show that compositional skill acquired on a source task transfers to a different target task. This transfer happens even without compositional training on the target, requiring only prior knowledge of the target's atomic skills. Our qualitative analysis shows that RL fundamentally changes the reasoning behaviors of the models. In contrast, next-token training with the same data yields none of these findings. Our systematic experiments provide fresh insights into LLM learning, suggesting the value of first building base models with basic skills, then using RL to incentivize advanced, generalizable skills for complex problems. 

---
# ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation 

**Authors**: Aasheesh Singh, Vishal Vaddina, Dagnachew Birru  

**Link**: [PDF](https://arxiv.org/pdf/2509.25100)  

**Abstract**: We introduce ORPO-Distill, a general-purpose method for cross-architecture LLM distillation that formulates the problem as a preference optimization task. Un- like standard CoT distillation, the approach transfers knowledge through diverse reasoning traces. It employs an Odds-Ratio Preference Optimization objective that contrasts teacher and student traces for more effective learning, and adopts a mixed-policy strategy for utilizing student-generated outputs, outperforming both off- and on-policy alternatives. Experiments on five datasets and multiple student models show consistent improvements over conventional black-box KD baselines. 

---
# Scaling with Collapse: Efficient and Predictable Training of LLM Families 

**Authors**: Shane Bergsma, Bin Claire Zhang, Nolan Dey, Shaheer Muhammad, Gurpreet Gosal, Joel Hestness  

**Link**: [PDF](https://arxiv.org/pdf/2509.25087)  

**Abstract**: Effective LLM training relies on *consistency*, meaning that key quantities -- such as final losses and optimal hyperparameters -- scale predictably across model sizes. Qiu et al. (2025) recently showed that this consistency extends beyond scalars: whole training loss curves can *collapse* onto a universal trajectory after a simple normalization. What remains unclear is whether this phenomenon holds for LLM families trained under *practical scaling recipes*, where width, depth, learning rate, batch size, and weight decay are scaled jointly. We show that it does: loss curves collapse across scales precisely when optimization hyperparameters are set optimally for the given data budget, in accordance with recent empirical scaling laws. Collapse thus emerges as a signature of compute-efficient training. We demonstrate two applications at scale: (1) deviation-from-collapse provides a sensitive, early diagnostic of training pathologies, and (2) the predictability of collapsed curves enables early stopping in large-scale hyperparameter tuning. Finally, we train a competitive LLM family, *Celerity*, using these insights, highlighting collapse as an effective tool for developing efficient LLMs. 

---
# Learning from Convenience Samples: A Case Study on Fine-Tuning LLMs for Survey Non-response in the German Longitudinal Election Study 

**Authors**: Tobias Holtdirk, Dennis Assenmacher, Arnim Bleier, Claudia Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2509.25063)  

**Abstract**: Survey researchers face two key challenges: the rising costs of probability samples and missing data (e.g., non-response or attrition), which can undermine inference and increase the use of convenience samples. Recent work explores using large language models (LLMs) to simulate respondents via persona-based prompts, often without labeled data. We study a more practical setting where partial survey responses exist: we fine-tune LLMs on available data to impute self-reported vote choice under both random and systematic nonresponse, using the German Longitudinal Election Study. We compare zero-shot prompting and supervised fine-tuning against tabular classifiers (e.g., CatBoost) and test how different convenience samples (e.g., students) used for fine-tuning affect generalization.
Our results show that when data are missing completely at random, fine-tuned LLMs match tabular classifiers but outperform zero-shot approaches. When only biased convenience samples are available, fine-tuning small (3B to 8B) open-source LLMs can recover both individual-level predictions and population-level distributions more accurately than zero-shot and often better than tabular methods. This suggests fine-tuned LLMs offer a promising strategy for researchers working with non-probability samples or systematic missingness, and may enable new survey designs requiring only easily accessible subpopulations. 

---
# DiffTester: Accelerating Unit Test Generation for Diffusion LLMs via Repetitive Pattern 

**Authors**: Lekang Yang, Yuetong Liu, Yitong Zhang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.24975)  

**Abstract**: Software development relies heavily on extensive unit testing, which makes the efficiency of automated Unit Test Generation (UTG) particularly important. However, most existing LLMs generate test cases one token at a time in each forward pass, which leads to inefficient UTG. Recently, diffusion LLMs (dLLMs) have emerged, offering promising parallel generation capabilities and showing strong potential for efficient UTG. Despite this advantage, their application to UTG is still constrained by a clear trade-off between efficiency and test quality, since increasing the number of tokens generated in each step often causes a sharp decline in the quality of test cases. To overcome this limitation, we present DiffTester, an acceleration framework specifically tailored for dLLMs in UTG. The key idea of DiffTester is that unit tests targeting the same focal method often share repetitive structural patterns. By dynamically identifying these common patterns through abstract syntax tree analysis during generation, DiffTester adaptively increases the number of tokens produced at each step without compromising the quality of the output. To enable comprehensive evaluation, we extend the original TestEval benchmark, which was limited to Python, by introducing additional programming languages including Java and C++. Extensive experiments on three benchmarks with two representative models show that DiffTester delivers significant acceleration while preserving test coverage. Moreover, DiffTester generalizes well across different dLLMs and programming languages, providing a practical and scalable solution for efficient UTG in software development. Code and data are publicly available at this https URL . 

---
# When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training 

**Authors**: Sanxing Chen, Xiaoyin Chen, Yukun Huang, Roy Xie, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2509.24923)  

**Abstract**: While Large Language Models (LLMs) hold promise to become autonomous agents, they often explore suboptimally in sequential decision-making. Recent work has sought to enhance this capability via supervised fine-tuning (SFT) or reinforcement learning (RL), improving regret on the classic multi-armed bandit task. However, it remains unclear how these learning methods shape exploration strategies and how well they generalize. We investigate both paradigms by training LLMs with SFT on expert trajectories and RL with a range of tailored reward signals including a strategic, regret-shaped reward to reduce variance, and an algorithmic reward that enables oracle imitation. The resulting agents outperform pre-trained models and achieve performance comparable to Upper Confidence Bound (UCB) and Thompson Sampling, with robust generalization to 6x longer horizons and across bandit families. Behavioral analysis reveals that gains often stem from more sophisticated but greedier exploitation: RL/SFT agents are more prone to early catastrophic failure than pre-trained models, prematurely abandoning exploration. Furthermore, agents trained to imitate UCB learn to outperform their teacher by adopting more exploitative variants. Our findings clarify when each training paradigm is preferable and advocate tailored reward design and evaluation beyond average regret to promote robust exploratory behavior. 

---
# MASLegalBench: Benchmarking Multi-Agent Systems in Deductive Legal Reasoning 

**Authors**: Huihao Jing, Wenbin Hu, Hongyu Luo, Jianhui Yang, Wei Fan, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.24922)  

**Abstract**: Multi-agent systems (MAS), leveraging the remarkable capabilities of Large Language Models (LLMs), show great potential in addressing complex tasks. In this context, integrating MAS with legal tasks is a crucial step. While previous studies have developed legal benchmarks for LLM agents, none are specifically designed to consider the unique advantages of MAS, such as task decomposition, agent specialization, and flexible training. In fact, the lack of evaluation methods limits the potential of MAS in the legal domain. To address this gap, we propose MASLegalBench, a legal benchmark tailored for MAS and designed with a deductive reasoning approach. Our benchmark uses GDPR as the application scenario, encompassing extensive background knowledge and covering complex reasoning processes that effectively reflect the intricacies of real-world legal situations. Furthermore, we manually design various role-based MAS and conduct extensive experiments using different state-of-the-art LLMs. Our results highlight the strengths, limitations, and potential areas for improvement of existing models and MAS architectures. 

---
# Neural network embeddings recover value dimensions from psychometric survey items on par with human data 

**Authors**: Max Pellert, Clemens M. Lechner, Indira Sen, Markus Strohmaier  

**Link**: [PDF](https://arxiv.org/pdf/2509.24906)  

**Abstract**: This study introduces "Survey and Questionnaire Item Embeddings Differentials" (SQuID), a novel methodological approach that enables neural network embeddings to effectively recover latent dimensions from psychometric survey items. We demonstrate that embeddings derived from large language models, when processed with SQuID, can recover the structure of human values obtained from human rater judgments on the Revised Portrait Value Questionnaire (PVQ-RR). Our experimental validation compares multiple embedding models across a number of evaluation metrics. Unlike previous approaches, SQuID successfully addresses the challenge of obtaining negative correlations between dimensions without requiring domain-specific fine-tuning. Quantitative analysis reveals that our embedding-based approach explains 55% of variance in dimension-dimension similarities compared to human data. Multidimensional scaling configurations from both types of data show fair factor congruence coefficients and largely follow the underlying theory. These results demonstrate that semantic embeddings can effectively replicate psychometric structures previously established through extensive human surveys. The approach offers substantial advantages in cost, scalability and flexibility while maintaining comparable quality to traditional methods. Our findings have significant implications for psychometrics and social science research, providing a complementary methodology that could expand the scope of human behavior and experience represented in measurement tools. 

---
# MMRQA: Signal-Enhanced Multimodal Large Language Models for MRI Quality Assessment 

**Authors**: Fankai Jia, Daisong Gan, Zhe Zhang, Zhaochi Wen, Chenchen Dan, Dong Liang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24888)  

**Abstract**: Magnetic resonance imaging (MRI) quality assessment is crucial for clinical decision-making, yet remains challenging due to data scarcity and protocol variability. Traditional approaches face fundamental trade-offs: signal-based methods like MRIQC provide quantitative metrics but lack semantic understanding, while deep learning approaches achieve high accuracy but sacrifice interpretability. To address these limitations, we introduce the Multimodal MRI Quality Assessment (MMRQA) framework, pioneering the integration of multimodal large language models (MLLMs) with acquisition-aware signal processing. MMRQA combines three key innovations: robust metric extraction via MRQy augmented with simulated artifacts, structured transformation of metrics into question-answer pairs using Qwen, and parameter-efficient fusion through Low-Rank Adaptation (LoRA) of LLaVA-OneVision. Evaluated on MR-ART, FastMRI, and MyConnectome benchmarks, MMRQA achieves state-of-the-art performance with strong zero-shot generalization, as validated by comprehensive ablation studies. By bridging quantitative analysis with semantic reasoning, our framework generates clinically interpretable outputs that enhance quality control in dynamic medical settings. 

---
# Retro*: Optimizing LLMs for Reasoning-Intensive Document Retrieval 

**Authors**: Junwei Lan, Jianlyu Chen, Zheng Liu, Chaofan Li, Siqi Bao, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2509.24869)  

**Abstract**: With the growing popularity of LLM agents and RAG, it has become increasingly important to retrieve documents that are essential for solving a task, even when their connection to the task is indirect or implicit. Addressing this problem requires fine-grained reasoning to accurately assess the relevance between the task and each candidate document. This capability, however, poses a significant challenge for existing IR techniques. Despite recent progress in reasoning-enhanced IR, existing approaches still face significant challenges in applicability, scalability, and efficiency. In this work, we propose Retro*, a novel approach for reasoning-intensive document retrieval. Our method introduces a rubric-based relevance scoring mechanism, enabling the model to reason about the relationship between a task and a document based on explicitly defined criteria, whereby producing a fine-grained, interpretable relevance score. Retro* also supports test-time scaling by combining multiple reasoning trajectories via score integration, which produces more reliable relevance estimates. To optimize Retro*'s reasoning capabilities, we introduce a novel reinforcement learning algorithm tailored for its relevance scoring mechanism, which employs two composite rewards to fully exploit the trajectories of each training sample. Our experiments show that Retro* outperforms existing document retrieval methods with notable advantages, leading to state-of-the-art performance on the BRIGHT benchmark. 

---
# Pushing LLMs to Their Logical Reasoning Bound: The Role of Data Reasoning Intensity 

**Authors**: Zhen Bi, Zhenlin Hu, Jinnan Yang, Mingyang Chen, Cheng Deng, Yida Xue, Zeyu Yang, Qing Shen, Zhenfang Liu, Kang Zhao, Ningyu Zhang, Jungang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2509.24836)  

**Abstract**: Recent advances in large language models (LLMs) highlight the importance of training data structure and quality in shaping reasoning behavior. However, most existing approaches focus on transforming data formats while neglecting the internal reasoning complexity of training samples, leaving the reasoning potential of data under-explored and underutilized. In this work, we posit that LLM logical reasoning performance is jointly constrained by the potential of the training data and the cognitive capacity of the model. To make this relationship measurable, we introduce Data Reasoning Intensity (DRI), a novel metric that quantifies the latent logical reasoning complexity of samples by decomposing and aggregating their logical structures. This allows us to analyze how well current LLMs utilize logical reasoning signals and identify performance gaps relative to data potential. Based on this insight, we introduce a re-cognizing optimization strategy that systematically enhances the logical reasoning intensity of training this http URL than increasing data volume, our method re-optimizes existing samples to better align with the LLM's logical reasoning boundary. Extensive experiments show that our approach significantly improves performance and generalization over data-centric strategies. We further validate our method under a reinforcement learning framework. Our results indicate that prioritizing reasoning complexity in data rather than sheer scale or superficial form is essential to realizing LLMs' full cognitive potential. 

---
# VSSFlow: Unifying Video-conditioned Sound and Speech Generation via Joint Learning 

**Authors**: Xin Cheng, Yuyue Wang, Xihua Wang, Yihan Wu, Kaisi Guan, Yijing Chen, Peng Zhang, Xiaojiang Liu, Meng Cao, Ruihua Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.24773)  

**Abstract**: Video-conditioned sound and speech generation, encompassing video-to-sound (V2S) and visual text-to-speech (VisualTTS) tasks, are conventionally addressed as separate tasks, with limited exploration to unify them within a signle framework. Recent attempts to unify V2S and VisualTTS face challenges in handling distinct condition types (e.g., heterogeneous video and transcript conditions) and require complex training stages. Unifying these two tasks remains an open problem. To bridge this gap, we present VSSFlow, which seamlessly integrates both V2S and VisualTTS tasks into a unified flow-matching framework. VSSFlow uses a novel condition aggregation mechanism to handle distinct input signals. We find that cross-attention and self-attention layer exhibit different inductive biases in the process of introducing condition. Therefore, VSSFlow leverages these inductive biases to effectively handle different representations: cross-attention for ambiguous video conditions and self-attention for more deterministic speech transcripts. Furthermore, contrary to the prevailing belief that joint training on the two tasks requires complex training strategies and may degrade performance, we find that VSSFlow benefits from the end-to-end joint learning process for sound and speech generation without extra designs on training stages. Detailed analysis attributes it to the learned general audio prior shared between tasks, which accelerates convergence, enhances conditional generation, and stabilizes the classifier-free guidance process. Extensive experiments demonstrate that VSSFlow surpasses the state-of-the-art domain-specific baselines on both V2S and VisualTTS benchmarks, underscoring the critical potential of unified generative models. 

---
# On the Self-awareness of Large Reasoning Models' Capability Boundaries 

**Authors**: Qingjie Zhang, Yujia Fu, Yang Wang, Liu Yan, Tao Wei, Ke Xu, Minlie Huang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24711)  

**Abstract**: Large Reasoning Models (LRMs) have shown impressive performance on complex reasoning tasks such as mathematics, yet they also display misbehaviors that expose their limitations. In particular, when faced with hard questions, LRMs often engage in unproductive reasoning until context limit, producing wrong answers while wasting substantial computation. This phenomenon reflects a fundamental issue: current answering paradigms overlook the relationship between questions and LRMs' capability boundaries. In this paper, we investigate whether LRMs possess self-awareness of capability boundaries. We begin by an observation that LRMs may know what they cannot solve through expressed reasoning confidence. For black-box models, we find that reasoning expressions reveal boundary signals, with accelerated growing confidence trajectory for solvable problems but convergent uncertainty trajectory for unsolvable ones. For white-box models, we show that hidden states of the last input token encode boundary information, with solvable and unsolvable problems linearly separable even before reasoning begins. Building on these findings, we propose two simple yet effective optimization strategies: reasoning expression monitoring and hidden states monitoring. Experiments demonstrate that these boundary-aware strategies enable LRMs to avoid unproductive reasoning without sacrificing accuracy, significantly improving reliability and efficiency by cutting token usage up to 62.7 - 93.6%. 

---
# OrthAlign: Orthogonal Subspace Decomposition for Non-Interfering Multi-Objective Alignment 

**Authors**: Liang Lin, Zhihao Xu, Junhao Dong, Jian Zhao, Yuchen Yuan, Guibin Zhang, Miao Yu, Yiming Zhang, Zhengtao Yao, Huahui Yi, Dongrui Liu, Xinfeng Li, Kun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24610)  

**Abstract**: Large language model (LLM) alignment faces a critical dilemma when addressing multiple human preferences: improvements in one dimension frequently come at the expense of others, creating unavoidable trade-offs between competing objectives like helpfulness and harmlessness. While prior work mainly focuses on constraint-based optimization algorithms and data selection strategies to mitigate conflicts, these approaches overlook the fundamental issue of resolving conflicts directly at the parameter level. In this paper, we present OrthAlign, an innovative approach that pioneers a new paradigm by leveraging orthogonal subspace decomposition to fundamentally resolve gradient-level conflicts in multi-objective preference alignment. OrthAlign strategically decomposes parameter update spaces into orthogonal subspaces, ensuring that optimization toward different preferences occurs in mathematically non-interfering directions. Building upon this, we provide theoretical guarantees demonstrating that when parameter increments satisfy both orthogonal subspace constraints and spectral norm bounds, the resulting updates exhibit linear Lipschitz growth rather than exponential instability, ensuring stable convergence across all preference dimensions. Extensive experiments show that: I. OrthAlign achieves maximum single-preference improvements ranging from 34.61% to 50.89% after multiple-objective alignment across helpful, harmless, and truthful dimensions. II. With an average overall reward improvement of 13.96%. 

---
# NeMo: Needle in a Montage for Video-Language Understanding 

**Authors**: Zi-Yuan Hu, Shuo Liang, Duo Zheng, Yanyang Li, Yeyao Tao, Shijia Huang, Wei Feng, Jia Qin, Jianguang Yu, Jing Huang, Meng Fang, Yin Li, Liwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24563)  

**Abstract**: Recent advances in video large language models (VideoLLMs) call for new evaluation protocols and benchmarks for complex temporal reasoning in video-language understanding. Inspired by the needle in a haystack test widely used by LLMs, we introduce a novel task of Needle in a Montage (NeMo), designed to assess VideoLLMs' critical reasoning capabilities, including long-context recall and temporal grounding. To generate video question answering data for our task, we develop a scalable automated data generation pipeline that facilitates high-quality data synthesis. Built upon the proposed pipeline, we present NeMoBench, a video-language benchmark centered on our task. Specifically, our full set of NeMoBench features 31,378 automatically generated question-answer (QA) pairs from 13,486 videos with various durations ranging from seconds to hours. Experiments demonstrate that our pipeline can reliably and automatically generate high-quality evaluation data, enabling NeMoBench to be continuously updated with the latest videos. We evaluate 20 state-of-the-art models on our benchmark, providing extensive results and key insights into their capabilities and limitations. Our project page is available at: this https URL. 

---
# LEAF: A Robust Expert-Based Framework for Few-Shot Continual Event Detection 

**Authors**: Bao-Ngoc Dao, Quang Nguyen, Luyen Ngo Dinh, Minh Le, Linh Ngo Van  

**Link**: [PDF](https://arxiv.org/pdf/2509.24547)  

**Abstract**: Few-shot Continual Event Detection (FCED) poses the dual challenges of learning from limited data and mitigating catastrophic forgetting across sequential tasks. Existing approaches often suffer from severe forgetting due to the full fine-tuning of a shared base model, which leads to knowledge interference between tasks. Moreover, they frequently rely on data augmentation strategies that can introduce unnatural or semantically distorted inputs. To address these limitations, we propose LEAF, a novel and robust expert-based framework for FCED. LEAF integrates a specialized mixture of experts architecture into the base model, where each expert is parameterized with low-rank adaptation (LoRA) matrices. A semantic-aware expert selection mechanism dynamically routes instances to the most relevant experts, enabling expert specialization and reducing knowledge interference. To improve generalization in limited-data settings, LEAF incorporates a contrastive learning objective guided by label descriptions, which capture high-level semantic information about event types. Furthermore, to prevent overfitting on the memory buffer, our framework employs a knowledge distillation strategy that transfers knowledge from previous models to the current one. Extensive experiments on multiple FCED benchmarks demonstrate that LEAF consistently achieves state-of-the-art performance. 

---
# Experience-guided reflective co-evolution of prompts and heuristics for automatic algorithm design 

**Authors**: Yihong Liu, Junyi Li, Wayne Xin Zhao, Hongyu Lu, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24509)  

**Abstract**: Combinatorial optimization problems are traditionally tackled with handcrafted heuristic algorithms, which demand extensive domain expertise and significant implementation effort. Recent progress has highlighted the potential of automatic heuristics design powered by large language models (LLMs), enabling the automatic generation and refinement of heuristics. These approaches typically maintain a population of heuristics and employ LLMs as mutation operators to evolve them across generations. While effective, such methods often risk stagnating in local optima. To address this issue, we propose the Experience-Guided Reflective Co-Evolution of Prompt and Heuristics (EvoPH) for automatic algorithm design, a novel framework that integrates the island migration model with the elites selection algorithm to simulate diverse heuristics populations. In EvoPH, prompts are co-evolved with heuristic algorithms, guided by performance feedback. We evaluate our framework on two problems, i.e., Traveling Salesman Problem and Bin Packing Problem. Experimental results demonstrate that EvoPH achieves the lowest relative error against optimal solutions across both datasets, advancing the field of automatic algorithm design with LLMs. 

---
# Euclid's Gift: Enhancing Spatial Perception and Reasoning in Vision-Language Models via Geometric Surrogate Tasks 

**Authors**: Shijie Lian, Changti Wu, Laurence Tianruo Yang, Hang Yuan, Bin Yu, Lei Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24473)  

**Abstract**: Spatial intelligence spans a rich suite of abilities, including visualising and transforming shapes, mentally rotating objects, judging relational positions and containment, and estimating numerosity. However, it still remains a critical unresolved challenge for Multimodal Large Language Models (MLLMs).To fill this gap, we propose to treat Euclidean geometry problem-solving as a surrogate task. Specifically, we meticulously constructed a curated multimodal dataset, called Euclid30K, comprising approximately 30K plane and solid geometry problems. To enable the model to acquire and apply Euclidean principles from these geometry problems, we employed Group Relative Policy Optimization (GRPO) to finetune the Qwen2.5VL family and RoboBrain2.0 family, inspiring the models to identify shapes, count, and relate entities, and perform multi-step deductive reasoning using Euclidean principles. Our experiments demonstrate that the resulting models achieve substantial zero-shot gains across four spatial reasoning benchmarks (Super-CLEVR, Omni3DBench, VSI-Bench, and MindCube) without any task-specific adaptations. Notably, after training on the Euclid30K, the mean VSI-Bench accuracy of all evaluated models rose from 34.5% to 40.5%, improving by 5.5 percentage points. Among them, RoboBrain2.0-Euclid-7B achieves 49.6\% accuracy, surpassing the previous state-of-the-art model, this http URL our knowledge, this is the first systematic study showing that geometry-centric fine-tuning can confer vision-language models with broadly transferable spatial skills. Code and Euclid30K dataset can be found in this https URL. 

---
# Beyond Isolated Facts: Synthesizing Narrative and Grounded Supervision for VideoQA 

**Authors**: Jianxin Liang, Tan Yue, Yuxuan Wang, Yueqian Wang, Zhihan Yin, Huishuai Zhang, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.24445)  

**Abstract**: The performance of Video Question Answering (VideoQA) models is fundamentally constrained by the nature of their supervision, which typically consists of isolated, factual question-answer pairs. This "bag-of-facts" approach fails to capture the underlying narrative and causal structure of events, limiting models to a shallow understanding of video content. To move beyond this paradigm, we introduce a framework to synthesize richer supervisory signals. We propose two complementary strategies: Question-Based Paraphrasing (QBP), which synthesizes the diverse inquiries (what, how, why) from a video's existing set of question-answer pairs into a holistic narrative paragraph that reconstructs the video's event structure; and Question-Based Captioning (QBC), which generates fine-grained visual rationales, grounding the answer to each question in specific, relevant evidence. Leveraging powerful generative models, we use this synthetic data to train VideoQA models under a unified next-token prediction objective. Extensive experiments on STAR and NExT-QA validate our approach, demonstrating significant accuracy gains and establishing new state-of-the-art results, such as improving a 3B model to 72.5\% on STAR (+4.9\%) and a 7B model to 80.8\% on NExT-QA. Beyond accuracy, our analysis reveals that both QBP and QBC substantially enhance cross-dataset generalization, with QBP additionally accelerating model convergence by over 2.5x. These results demonstrate that shifting data synthesis from isolated facts to narrative coherence and grounded rationales yields a more accurate, efficient, and generalizable training paradigm. 

---
# Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention 

**Authors**: Yichi Zhang, Yue Ding, Jingwen Yang, Tianwei Luo, Dongbai Li, Ranjie Duan, Qiang Liu, Hang Su, Yinpeng Dong, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24393)  

**Abstract**: Although Large Reasoning Models (LRMs) have progressed in solving complex problems, their chain-of-thought (CoT) reasoning often contains harmful content that can persist even when the final responses appear safe. We show that this issue still remains in existing methods which overlook the unique significance of safe reasoning, undermining their trustworthiness and posing potential risks in applications if unsafe reasoning is accessible for and exploited by malicious users. We therefore shift our focus to aligning the safety of reasoning itself in this paper and explore process supervision as the solution. However, simply rewarding safe reasoning proves inadequate due to low rollout diversity and limited training signals. To tackle this challenge, we first delve into the characteristics of safe reasoning and uncover several critical insights that 1) safe reasoning is often consolidated by a few critical steps of safety triggers; 2) compliance cues strongly correlate with unsafe continuations; and 3) corrective interventions reliably steer unsafe trajectories towards safer traces. Motivated by these, we propose Intervened Preference Optimization (IPO), an alignment method that enforces safe reasoning by substituting compliance steps with safety triggers and constructing pairs for preference learning with strong signals. Experiments on jailbreak and adversarial safety benchmarks demonstrate that IPO remarkably improves overall safety regarding both reasoning and responses, outperforming SFT-based and RL-based baselines with a relative reduction of over 30% in harmfulness, while preserving excellent performance across diverse reasoning tasks. The results highlight the importance of explicit alignment for reasoning and provide a practical path to safer LRMs. 

---
# MAS$^2$: Self-Generative, Self-Configuring, Self-Rectifying Multi-Agent Systems 

**Authors**: Kun Wang, Guibin Zhang, ManKit Ye, Xinyu Deng, Dongxia Wang, Xiaobin Hu, Jinyang Guo, Yang Liu, Yufei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.24323)  

**Abstract**: The past two years have witnessed the meteoric rise of Large Language Model (LLM)-powered multi-agent systems (MAS), which harness collective intelligence and exhibit a remarkable trajectory toward self-evolution. This paradigm has rapidly progressed from manually engineered systems that require bespoke configuration of prompts, tools, roles, and communication protocols toward frameworks capable of automated orchestration. Yet, dominant automatic multi-agent systems, whether generated by external modules or a single LLM agent, largely adhere to a rigid ``\textit{generate-once-and-deploy}'' paradigm, rendering the resulting systems brittle and ill-prepared for the dynamism and uncertainty of real-world environments. To transcend this limitation, we introduce MAS$^2$, a paradigm predicated on the principle of recursive self-generation: a multi-agent system that autonomously architects bespoke multi-agent systems for diverse problems. Technically, we devise a ``\textit{generator-implementer-rectifier}'' tri-agent team capable of dynamically composing and adaptively rectifying a target agent system in response to real-time task demands. Collaborative Tree Optimization is proposed to train and specialize these meta-agents. Extensive evaluation across seven benchmarks reveals that MAS$^2$ achieves performance gains of up to $19.6\%$ over state-of-the-art MAS in complex scenarios such as deep research and code generation. Moreover, MAS$^2$ exhibits superior cross-backbone generalization, effectively leveraging previously unseen LLMs to yield improvements of up to $15.1\%$. Crucially, these gains are attained without incurring excessive token costs, as MAS$^2$ consistently resides on the Pareto frontier of cost-performance trade-offs. The source codes are available at this https URL. 

---
# Bridging the behavior-neural gap: A multimodal AI reveals the brain's geometry of emotion more accurately than human self-reports 

**Authors**: Changde Du, Yizhuo Lu, Zhongyu Huang, Yi Sun, Zisen Zhou, Shaozheng Qin, Huiguang He  

**Link**: [PDF](https://arxiv.org/pdf/2509.24298)  

**Abstract**: The ability to represent emotion plays a significant role in human cognition and social interaction, yet the high-dimensional geometry of this affective space and its neural underpinnings remain debated. A key challenge, the `behavior-neural gap,' is the limited ability of human self-reports to predict brain activity. Here we test the hypothesis that this gap arises from the constraints of traditional rating scales and that large-scale similarity judgments can more faithfully capture the brain's affective geometry. Using AI models as `cognitive agents,' we collected millions of triplet odd-one-out judgments from a multimodal large language model (MLLM) and a language-only model (LLM) in response to 2,180 emotionally evocative videos. We found that the emergent 30-dimensional embeddings from these models are highly interpretable and organize emotion primarily along categorical lines, yet in a blended fashion that incorporates dimensional properties. Most remarkably, the MLLM's representation predicted neural activity in human emotion-processing networks with the highest accuracy, outperforming not only the LLM but also, counterintuitively, representations derived directly from human behavioral ratings. This result supports our primary hypothesis and suggests that sensory grounding--learning from rich visual data--is critical for developing a truly neurally-aligned conceptual framework for emotion. Our findings provide compelling evidence that MLLMs can autonomously develop rich, neurally-aligned affective representations, offering a powerful paradigm to bridge the gap between subjective experience and its neural substrates. Project page: this https URL. 

---
# SCI-Verifier: Scientific Verifier with Thinking 

**Authors**: Shenghe Zheng, Chenyu Huang, Fangchen Yu, Junchi Yao, Jingqi Ye, Tao Chen, Yun Luo, Ning Ding, LEI BAI, Ganqu Cui, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.24285)  

**Abstract**: As large language models (LLMs) are increasingly applied to scientific reasoning, the complexity of answer formats and the diversity of equivalent expressions make answer verification a critical yet challenging task. Existing verification studies in scientific domains suffer from two major limitations: (a) the absence of systematic evaluation standards and insufficient disciplinary coverage, which hinders their comprehensive assessment; and (b) heavy reliance on cumbersome rule design or prompt engineering, which reduces their effectiveness in complex reasoning scenarios or limits their cross-disciplinary generalization. To address these challenges, we propose solutions at both the data and model levels. On the data side, we construct SCI-VerifyBench, a cross-disciplinary benchmark covering mathematics, physics, biology, chemistry, and general scientific QA. The benchmark is built from real LLM responses and enhanced with domain-specific equivalence transformations that generate challenging and realistic data. Model-based and expert annotations ensure both quality and diversity, enabling rigorous evaluation of verification ability. On the model side, we emphasize the importance of reasoning for verification and introduce SCI-Verifier, a unified reasoning-augmented verifier for scientific domains. Through post-training, SCI-Verifier demonstrates strong logical reasoning and equivalence judgment capabilities while maintaining concise and stable outputs. Together, SCI-VerifyBench and SCI-Verifier provide a principled framework for scientific verification, offering both systematic evaluation and practical pathways to enhance the reliability and applicability of LLMs in scientific domains. 

---
# Overview of SCIDOCA 2025 Shared Task on Citation Prediction, Discovery, and Placement 

**Authors**: An Dao, Vu Tran, Le-Minh Nguyen, Yuji Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2509.24283)  

**Abstract**: We present an overview of the SCIDOCA 2025 Shared Task, which focuses on citation discovery and prediction in scientific documents. The task is divided into three subtasks: (1) Citation Discovery, where systems must identify relevant references for a given paragraph; (2) Masked Citation Prediction, which requires selecting the correct citation for masked citation slots; and (3) Citation Sentence Prediction, where systems must determine the correct reference for each cited sentence. We release a large-scale dataset constructed from the Semantic Scholar Open Research Corpus (S2ORC), containing over 60,000 annotated paragraphs and a curated reference set. The test set consists of 1,000 paragraphs from distinct papers, each annotated with ground-truth citations and distractor candidates. A total of seven teams registered, with three submitting results. We report performance metrics across all subtasks and analyze the effectiveness of submitted systems. This shared task provides a new benchmark for evaluating citation modeling and encourages future research in scientific document understanding. The dataset and task materials are publicly available at this https URL. 

---
# AdvChain: Adversarial Chain-of-Thought Tuning for Robust Safety Alignment of Large Reasoning Models 

**Authors**: Zihao Zhu, Xinyu Wu, Gehan Hu, Siwei Lyu, Ke Xu, Baoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24269)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in complex problem-solving through Chain-of-Thought (CoT) reasoning. However, the multi-step nature of CoT introduces new safety challenges that extend beyond conventional language model alignment. We identify a failure mode in current safety CoT tuning methods: the \textit{snowball effect}, where minor reasoning deviations progressively amplify throughout the thought process, leading to either harmful compliance or excessive refusal. This effect stems from models being trained to imitate perfect reasoning scripts without learning to self-correct. To address this limitation, we propose AdvChain, an alignment paradigm that teaches models dynamic self-correction through adversarial CoT tuning. Our method involves constructing a dataset containing Temptation-Correction and Hesitation-Correction samples, where models learn to recover from harmful reasoning drifts and unnecessary cautions. Extensive experiments show that AdvChain significantly enhances robustness against jailbreak attacks and CoT hijacking while substantially reducing over-refusal on benign prompts, achieving a superior safety-utility balance without compromising reasoning capabilities. Our work establishes a new direction for building more robust and reliable reasoning models. 

---
# PAME-AI: Patient Messaging Creation and Optimization using Agentic AI 

**Authors**: Junjie Luo, Yihong Guo, Anqi Liu, Ritu Agarwal, Gordon  

**Link**: [PDF](https://arxiv.org/pdf/2509.24263)  

**Abstract**: Messaging patients is a critical part of healthcare communication, helping to improve things like medication adherence and healthy behaviors. However, traditional mobile message design has significant limitations due to its inability to explore the high-dimensional design space. We develop PAME-AI, a novel approach for Patient Messaging Creation and Optimization using Agentic AI. Built on the Data-Information-Knowledge-Wisdom (DIKW) hierarchy, PAME-AI offers a structured framework to move from raw data to actionable insights for high-performance messaging design. PAME-AI is composed of a system of specialized computational agents that progressively transform raw experimental data into actionable message design strategies. We demonstrate our approach's effectiveness through a two-stage experiment, comprising of 444,691 patient encounters in Stage 1 and 74,908 in Stage 2. The best-performing generated message achieved 68.76% engagement compared to the 61.27% baseline, representing a 12.2\% relative improvement in click-through rates. This agentic architecture enables parallel processing, hypothesis validation, and continuous learning, making it particularly suitable for large-scale healthcare communication optimization. 

---
# Extracting the Structure of Press Releases for Predicting Earnings Announcement Returns 

**Authors**: Yuntao Wu, Ege Mert Akin, Charles Martineau, Vincent Grégoire, Andreas Veneris  

**Link**: [PDF](https://arxiv.org/pdf/2509.24254)  

**Abstract**: We examine how textual features in earnings press releases predict stock returns on earnings announcement days. Using over 138,000 press releases from 2005 to 2023, we compare traditional bag-of-words and BERT-based embeddings. We find that press release content (soft information) is as informative as earnings surprise (hard information), with FinBERT yielding the highest predictive power. Combining models enhances explanatory strength and interpretability of the content of press releases. Stock prices fully reflect the content of press releases at market open. If press releases are leaked, it offers predictive advantage. Topic analysis reveals self-serving bias in managerial narratives. Our framework supports real-time return prediction through the integration of online learning, provides interpretability and reveals the nuanced role of language in price formation. 

---
# Latent Visual Reasoning 

**Authors**: Bangzheng Li, Ximeng Sun, Jiang Liu, Ze Wang, Jialian Wu, Xiaodong Yu, Hao Chen, Emad Barsoum, Muhao Chen, Zicheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24251)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved notable gains in various tasks by incorporating Chain-of-Thought (CoT) reasoning in language spaces. Recent work extends this direction by leveraging external tools for visual editing, thereby enhancing the visual signal along the reasoning trajectories. Nevertheless, these approaches remain fundamentally constrained: reasoning is still confined to the language space, with visual information treated as static preconditions. We introduce Latent Visual Reasoning (LVR), a new paradigm that enables autoregressive reasoning directly in the visual embedding space. A visual encoder first projects images into visual tokens within a joint semantic space shared with the language model. The language model is then trained to generate latent states that reconstruct key visual tokens critical for answering the query, constituting the process of latent visual reasoning. By interleaving LVR with standard text generation, our model achieves substantial gains on perception-intensive visual question answering tasks. In addition, we adapt the GRPO algorithm to conduct reinforcement learning on latent reasoning, further balancing LVR and textual generation. We show that LVR substantially improves fine-grained visual understanding and perception, achieving 71.67% on MMVP compared to 66.67% with Qwen2.5-VL. Code base and model weights will be released later. 

---
# SpecExit: Accelerating Large Reasoning Model via Speculative Exit 

**Authors**: Rubing Yang, Huajun Bai, Song Liu, Guanghua Yu, Runzhi Fan, Yanbin Dang, Jiejing Zhang, Kai Liu, Jianchen Zhu, Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24248)  

**Abstract**: Despite their strong performance on reasoning tasks, large reasoning models (LRMs) often suffer from overthinking, producing unnecessarily long outputs and incurring high end-to-end latency, a significant limitation to their real-world deployment. To address overthinking, early-exit mechanisms have been proposed to terminate reasoning before typical completion, showing that this approach can effectively shorten generation length with minimal impact on accuracy. However, their reliance on probing mechanisms introduces a detection overhead that limits their end-to-end latency gains and compromises their generalizability across diverse problems. Inspired by the use of hidden states in speculative decoding, we propose SpecExit, a novel framework that predicts both future tokens and an early-exit signal directly from a lightweight draft model without probing overhead. Our method offers significant improvements, reducing average generation length by 66\% and achieving a 2.5x speedup in end-to-end latency compared to the speculative decoding baseline, without compromising accuracy. Our method leverages the inherent signals from hidden states to provide effective early-exit signals, suggesting broader use of hidden states for efficient reasoning. Our code is available at this https URL. 

---
# Learning to Ponder: Adaptive Reasoning in Latent Space 

**Authors**: Yixin He, Lumingyuan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24238)  

**Abstract**: Test-time compute has emerged as a key paradigm for enhancing LLM reasoning, yet prevailing approaches like Best-of-N and majority voting apply uniform depth across inputs, wasting computation on simple queries while potentially under-thinking complex ones. We present FR-Ponder, a single-graph, backbone-training-free framework that allocates instance-adaptive reasoning compute via latent steering. A less than 1M-param controller observes hidden states and decides to halt or apply a small ponder step by adding a pre-computed steering vector to frozen representations. Our method extracts the latent steering vector associated with deeper reasoning outputs and direct IO from LLM and re-applies it through a tunable scaling factor, allowing the model to adapt its reasoning depth to the complexity of each input. To balance performance and computational cost, we employ Group Relative Policy Optimization (GRPO) as a reward signal to adaptively regulate reasoning depth, achieving task accuracy while mitigating overreasoning. Through curriculum learning and careful reward engineering, FR-Ponder learns calibrated compute allocation correlated with problem difficulty. On GSM8K and MATH500, FR-Ponder improves the compute-accuracy frontier, delivering lower FLOPs with better matched accuracy and comparing favorably to early-exit baselines, without modifying backbone weights. Analyses visualize interpretable steering directions and show learned compute allocation correlates with problem difficulty. 

---
# Metamorphic Testing for Audio Content Moderation Software 

**Authors**: Wenxuan Wang, Yongjiang Wu, Junyuan Zhang, Shuqing Li, Yun Peng, Wenting Chen, Shuai Wang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24215)  

**Abstract**: The rapid growth of audio-centric platforms and applications such as WhatsApp and Twitter has transformed the way people communicate and share audio content in modern society. However, these platforms are increasingly misused to disseminate harmful audio content, such as hate speech, deceptive advertisements, and explicit material, which can have significant negative consequences (e.g., detrimental effects on mental health). In response, researchers and practitioners have been actively developing and deploying audio content moderation tools to tackle this issue. Despite these efforts, malicious actors can bypass moderation systems by making subtle alterations to audio content, such as modifying pitch or inserting noise. Moreover, the effectiveness of modern audio moderation tools against such adversarial inputs remains insufficiently studied. To address these challenges, we propose MTAM, a Metamorphic Testing framework for Audio content Moderation software. Specifically, we conduct a pilot study on 2000 audio clips and define 14 metamorphic relations across two perturbation categories: Audio Features-Based and Heuristic perturbations. MTAM applies these metamorphic relations to toxic audio content to generate test cases that remain harmful while being more likely to evade detection. In our evaluation, we employ MTAM to test five commercial textual content moderation software and an academic model against three kinds of toxic content. The results show that MTAM achieves up to 38.6%, 18.3%, 35.1%, 16.7%, and 51.1% error finding rates (EFR) when testing commercial moderation software provided by Gladia, Assembly AI, Baidu, Nextdata, and Tencent, respectively, and it obtains up to 45.7% EFR when testing the state-of-the-art algorithms from the academy. 

---
# Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends 

**Authors**: Chaorui Yao, Yanxi Chen, Yuchang Sun, Yushuo Chen, Wenhao Zhang, Xuchen Pan, Yaliang Li, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.24203)  

**Abstract**: Off-policy reinforcement learning (RL) for large language models (LLMs) is attracting growing interest, driven by practical constraints in real-world applications, the complexity of LLM-RL infrastructure, and the need for further innovations of RL methodologies. While classic REINFORCE and its modern variants like Group Relative Policy Optimization (GRPO) are typically regarded as on-policy algorithms with limited tolerance of off-policyness, we present in this work a first-principles derivation for group-relative REINFORCE without assuming a specific training data distribution, showing that it admits a native off-policy interpretation. This perspective yields two general principles for adapting REINFORCE to off-policy settings: regularizing policy updates, and actively shaping the data distribution. Our analysis demystifies some myths about the roles of importance sampling and clipping in GRPO, unifies and reinterprets two recent algorithms -- Online Policy Mirror Descent (OPMD) and Asymmetric REINFORCE (AsymRE) -- as regularized forms of the REINFORCE loss, and offers theoretical justification for seemingly heuristic data-weighting strategies. Our findings lead to actionable insights that are validated with extensive empirical studies, and open up new opportunities for principled algorithm design in off-policy RL for LLMs. Source code for this work is available at this https URL. 

---
# Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models 

**Authors**: Yuhui Wang, Changjiang Li, Guangke Chen, Jiacheng Liang, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24156)  

**Abstract**: Large reasoning models (LRMs) exhibit unprecedented capabilities in solving complex problems through Chain-of-Thought (CoT) reasoning. However, recent studies reveal that their final answers often contradict their own reasoning traces. We hypothesize that this inconsistency stems from two competing mechanisms for generating answers: CoT reasoning and memory retrieval. To test this hypothesis, we conduct controlled experiments that challenge LRMs with misleading cues during reasoning and/or corrupted answers during retrieval. Our results across models and datasets confirm that both mechanisms operate simultaneously, with their relative dominance influenced by multiple factors: problem domains, model scales, and fine-tuning approaches (e.g., reinforcement learning vs. distillation). The findings reveal a critical limitation in current reasoning fine-tuning paradigms: models can exploit the retrieval mechanism as a shortcut, effectively "hacking" the reward signal and undermining genuine reasoning development. To address this challenge, we introduce FARL, a novel fine-tuning framework that integrates memory unlearning with reinforcement learning. By carefully suppressing retrieval shortcuts during the fine-tuning process, FARL promotes reasoning-dominant behavior and enhances generalizable reasoning capabilities. 

---
# Generalist Scanner Meets Specialist Locator: A Synergistic Coarse-to-Fine Framework for Robust GUI Grounding 

**Authors**: Zhecheng Li, Guoxian Song, Yiwei Wang, Zhen Xiong, Junsong Yuan, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.24133)  

**Abstract**: Grounding natural language queries in graphical user interfaces (GUIs) presents a challenging task that requires models to comprehend diverse UI elements across various applications and systems, while also accurately predicting the spatial coordinates for the intended operation. To tackle this problem, we propose GMS: Generalist Scanner Meets Specialist Locator, a synergistic coarse-to-fine framework that effectively improves GUI grounding performance. GMS leverages the complementary strengths of general vision-language models (VLMs) and small, task-specific GUI grounding models by assigning them distinct roles within the framework. Specifically, the general VLM acts as a 'Scanner' to identify potential regions of interest, while the fine-tuned grounding model serves as a 'Locator' that outputs precise coordinates within these regions. This design is inspired by how humans perform GUI grounding, where the eyes scan the interface and the brain focuses on interpretation and localization. Our whole framework consists of five stages and incorporates hierarchical search with cross-modal communication to achieve promising prediction results. Experimental results on the ScreenSpot-Pro dataset show that while the 'Scanner' and 'Locator' models achieve only $2.0\%$ and $3.7\%$ accuracy respectively when used independently, their integration within GMS framework yields an overall accuracy of $35.7\%$, representing a $10 \times$ improvement. Additionally, GMS significantly outperforms other strong baselines under various settings, demonstrating its robustness and potential for general-purpose GUI grounding. 

---
# Do Repetitions Matter? Strengthening Reliability in LLM Evaluations 

**Authors**: Miguel Angel Alvarado Gonzalez, Michelle Bruno Hernandez, Miguel Angel Peñaloza Perez, Bruno Lopez Orozco, Jesus Tadeo Cruz Soto, Sandra Malagon  

**Link**: [PDF](https://arxiv.org/pdf/2509.24086)  

**Abstract**: LLM leaderboards often rely on single stochastic runs, but how many repetitions are required for reliable conclusions remains unclear. We re-evaluate eight state-of-the-art models on the AI4Math Benchmark with three independent runs per setting. Using mixed-effects logistic regression, domain-level marginal means, rank-instability analysis, and run-to-run reliability, we assessed the value of additional repetitions. Our findings shows that Single-run leaderboards are brittle: 10/12 slices (83\%) invert at least one pairwise rank relative to the three-run majority, despite a zero sign-flip rate for pairwise significance and moderate overall interclass correlation. Averaging runs yields modest SE shrinkage ($\sim$5\% from one to three) but large ranking gains; two runs remove $\sim$83\% of single-run inversions. We provide cost-aware guidance for practitioners: treat evaluation as an experiment, report uncertainty, and use $\geq 2$ repetitions under stochastic decoding. These practices improve robustness while remaining feasible for small teams and help align model comparisons with real-world reliability. 

---
# The Role of Logic and Automata in Understanding Transformers 

**Authors**: Anthony W. Lin, Pablo Barcelo  

**Link**: [PDF](https://arxiv.org/pdf/2509.24024)  

**Abstract**: The advent of transformers has in recent years led to powerful and revolutionary Large Language Models (LLMs). Despite this, our understanding on the capability of transformers is still meager. In this invited contribution, we recount the rapid progress in the last few years to the question of what transformers can do. In particular, we will see the integral role of logic and automata (also with some help from circuit complexity) in answering this question. We also mention several open problems at the intersection of logic, automata, verification and transformers. 

---
# Detecting and Rectifying Noisy Labels: A Similarity-based Approach 

**Authors**: Dang Huu-Tien, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2509.23964)  

**Abstract**: Label noise in datasets could damage the performance of neural net training. As the size of modern deep networks grows, there is a growing demand for automated tools for detecting such errors. In this paper, we propose post-hoc, model-agnostic error detection and rectification methods utilizing the penultimate feature from a neural network. Our idea is based on the observation that the similarity between the penultimate feature of a mislabeled data point and its true class data points is higher than that for data points from other classes, making the probability of label occurrence within a tight, similar cluster informative for detecting and rectifying errors. Extensive experiments show our method not only demonstrates high performance across various noises but also automatically rectifies these errors to improve the quality of datasets. 

---
# Conditional Advantage Estimation for Reinforcement Learning in Large Reasoning Models 

**Authors**: Guanxu Chen, Yafu Li, Yuxian Jiang, Chen Qian, Qihan Ren, Jingyi Yang, Yu Cheng, Dongrui Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23962)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) for large language models (LLMs) has achieved remarkable progress in enhancing LLMs' reasoning capabilities on tasks with clear correctness criteria, such as mathematical reasoning tasks. Several training metrics, such as entropy or response length, have been observed to correlate with different reasoning behaviors in reinforcement learning. Prior approaches incorporate such priors through reward or advantage shaping, which often relies on hand-crafted penalties and preferences (e.g., higher-is-better or lower-is-better). However, without careful hyperparameter tuning, these directional priors can be overly biased and may lead to failure. To this end, we introduce Conditional advANtage estimatiON (CANON), amplifying the impact of the target metric without presuming its direction. Specifically, CANON regroups the sampled responses into two groups based on the higher or lower value of a target metric, measures which metric trend contributes to better performance through inter-group comparison, and identifies the better response within the same group. In summary, CANON based on entropy consistently outperforms prior methods across three LLMs on both math reasoning and high-complexity logic tasks. When applied to response length, CANON further improves token efficiency, yielding a more favorable Pareto frontier in the performance-cost trade-off. 

---
# Explore-Execute Chain: Towards an Efficient Structured Reasoning Paradigm 

**Authors**: Kaisen Yang, Lixuan He, Rushi Shah, Kaicheng Yang, Qinwei Ma, Dianbo Liu, Alex Lamb  

**Link**: [PDF](https://arxiv.org/pdf/2509.23946)  

**Abstract**: Chain-of-Thought (CoT) and its variants have markedly advanced the reasoning abilities of Large Language Models (LLMs), yet their monolithic and auto-regressive architecture inherently conflates high-level strategic planning with low-level step-by-step execution, leading to computational inefficiency, limited exploration of reasoning paths, and reduced interpretability. To overcome these issues, we propose the Explore-Execute Chain ($E^2C$), a structured reasoning framework that decouples reasoning into two distinct phases: an exploratory phase that stochastically generates succinct high-level plans, followed by an execution phase that deterministically carries out the chosen plan. Our approach incorporates a two-stage training methodology, which combines Supervised Fine-Tuning (SFT) - augmented by a novel data generation algorithm enforcing strict plan adherence - with a subsequent Reinforcement Learning (RL) stage that capitalizes on the informativeness of exploration and reinforces the determinism of this http URL decomposition enables an efficient test-time scaling strategy: on AIME'2024, $E^2C$ Test Time Scaling reaches 58.1% accuracy using <10% of the decoding tokens required by comparable methods (e.g., Forest-of-Thought), sharply cutting self-consistency overhead. For cross-domain adaptation, our Exploration-Focused SFT (EF-SFT) fine-tunes with only 3.5% of the tokens used by standard SFT yet yields up to 14.5% higher accuracy than standard SFT on medical benchmarks, delivering state-of-the-art performance, strong generalization, and greater interpretability by separating planning from execution. The code and pre-trained models for the project are available at: this https URL 

---
# Beyond Benchmarks: Understanding Mixture-of-Experts Models through Internal Mechanisms 

**Authors**: Jiahao Ying, Mingbao Lin, Qianru Sun, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23933)  

**Abstract**: Mixture-of-Experts (MoE) architectures have emerged as a promising direction, offering efficiency and scalability by activating only a subset of parameters during inference. However, current research remains largely performance-centric, with limited understanding of its internal mechanisms, thereby constraining broader progress. In this work, we use an internal metric to investigate the mechanisms of MoE architecture by explicitly incorporating routing mechanisms and analyzing expert-level behaviors. Through systematic analyses of a wide range of publicly available MoE models, we uncover several findings: (1) neuron utilization decreases as models evolve, reflecting stronger generalization; (2) training exhibits a dynamic trajectory, where benchmark performance alone provides limited signal while MUI reveals deeper insights; (3) task completion emerges from collaborative contributions of multiple experts, with shared experts driving concentration; and (4) activation patterns at the neuron level provide a fine-grained proxy for data diversity. Together, these results demonstrate the potential of MUI as a complementary indicator to benchmark performance, offering new insights into the capacity, dynamics, and specialization of MoE models. Our project can be found at this https URL. 

---
# Dynamic Orthogonal Continual Fine-tuning for Mitigating Catastrophic Forgettings 

**Authors**: Zhixin Zhang, Zeming Wei, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.23893)  

**Abstract**: Catastrophic forgetting remains a critical challenge in continual learning for large language models (LLMs), where models struggle to retain performance on historical tasks when fine-tuning on new sequential data without access to past datasets. In this paper, we first reveal that the drift of functional directions during the fine-tuning process is a key reason why existing regularization-based methods fail in long-term LLM continual learning. To address this, we propose Dynamic Orthogonal Continual (DOC) fine-tuning, a novel approach that tracks the drift of these functional directions and dynamically updates them during the fine-tuning process. Furthermore, by adjusting the gradients of new task parameters to be orthogonal to the tracked historical function directions, our method mitigates interference between new and old tasks. Extensive experiments on various LLM continual learning benchmarks demonstrate that this approach outperforms prior methods, effectively reducing catastrophic forgetting and providing a robust tool for continuous LLM fine-tuning. Our code is available at this https URL. 

---
# PCRI: Measuring Context Robustness in Multimodal Models for Enterprise Applications 

**Authors**: Hitesh Laxmichand Patel, Amit Agarwal, Srikant Panda, Hansa Meghwani, Karan Dua, Paul Li, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2509.23879)  

**Abstract**: The reliability of Multimodal Large Language Models (MLLMs) in real-world settings is often undermined by sensitivity to irrelevant or distracting visual context, an aspect not captured by existing evaluation metrics. We introduce the \textbf{Patch Context Robustness Index (PCRI)}, the first systematic and interpretable score for quantifying MLLM robustness to variations in visual context granularity, measuring performance changes between localized image patches and full-image input.
Applying PCRI to 19 state-of-the-art MLLMs across 15 vision-language benchmarks, we find that most leading models remain brittle to background noise, with only a few, such as InternVL2-26B and Qwen2VL-72B, demonstrating consistent robustness across tasks. PCRI analysis also highlights how different model architectures handle and integrate visual context, offering actionable diagnostic insight for both researchers and practitioners.
PCRI enables rigorous comparison of context robustness, supporting principled model selection and guiding the development of future architectures and training strategies for robust, real-world deployment. 

---
# Beyond the Exploration-Exploitation Trade-off: A Hidden State Approach for LLM Reasoning in RLVR 

**Authors**: Fanding Huang, Guanbo Huang, Xiao Fan, Yi He, Xiao Liang, Xiao Chen, Qinting Jiang, Faisal Nadeem Khan, Jingyan Jiang, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23808)  

**Abstract**: A prevailing view in Reinforcement Learning for Verifiable Rewards (RLVR) interprets recent progress through the lens of an exploration-exploitation trade-off, a perspective largely shaped by token-level metrics. We re-examine this perspective, proposing that this perceived trade-off may not be a fundamental constraint but rather an artifact of the measurement level. To investigate this, we shift the analysis to the semantically rich hidden-state space, adopting Effective Rank (ER) to quantify exploration and proposing its novel first- and second-order derivatives, named Effective Rank Velocity (ERV) and Effective Rank Acceleration (ERA), to capture exploitation dynamics. Our analysis reveals that at the hidden-state level, exploration and exploitation could be decoupled (Sec. 4). This finding reveals an opportunity to enhance both capacities simultaneously. This insight motivates our method, Velocity-Exploiting Rank-Learning (VERL), the first to operationalize the principle of synergistic exploration-exploitation enhancement by directly shaping the RL advantage function. The key innovation is leveraging the theoretically stable ERA as a predictive meta-controller to create a synergistic, dual-channel incentive structure. Instead of forcing a trade-off, VERL prospectively amplifies rewards for exploration to preempt overconfidence and reinforces exploitative gains to consolidate reasoning. Experiments across diverse LLMs and reasoning benchmarks show consistent gains, including up to 21.4% absolute accuracy improvement on the challenging Gaokao 2024 dataset. 

---
# Knowledge Homophily in Large Language Models 

**Authors**: Utkarsh Sahu, Zhisheng Qi, Mahantesh Halappanavar, Nedim Lipka, Ryan A. Rossi, Franck Dernoncourt, Yu Zhang, Yao Ma, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23773)  

**Abstract**: Large Language Models (LLMs) have been increasingly studied as neural knowledge bases for supporting knowledge-intensive applications such as question answering and fact checking. However, the structural organization of their knowledge remains unexplored. Inspired by cognitive neuroscience findings, such as semantic clustering and priming, where knowing one fact increases the likelihood of recalling related facts, we investigate an analogous knowledge homophily pattern in LLMs. To this end, we map LLM knowledge into a graph representation through knowledge checking at both the triplet and entity levels. After that, we analyze the knowledgeability relationship between an entity and its neighbors, discovering that LLMs tend to possess a similar level of knowledge about entities positioned closer in the graph. Motivated by this homophily principle, we propose a Graph Neural Network (GNN) regression model to estimate entity-level knowledgeability scores for triplets by leveraging their neighborhood scores. The predicted knowledgeability enables us to prioritize checking less well-known triplets, thereby maximizing knowledge coverage under the same labeling budget. This not only improves the efficiency of active labeling for fine-tuning to inject knowledge into LLMs but also enhances multi-hop path retrieval in reasoning-intensive question answering. 

---
# From What to Why: A Multi-Agent System for Evidence-based Chemical Reaction Condition Reasoning 

**Authors**: Cheng Yang, Jiaxuan Lu, Haiyuan Wan, Junchi Yu, Feiwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.23768)  

**Abstract**: The chemical reaction recommendation is to select proper reaction condition parameters for chemical reactions, which is pivotal to accelerating chemical science. With the rapid development of large language models (LLMs), there is growing interest in leveraging their reasoning and planning capabilities for reaction condition recommendation. Despite their success, existing methods rarely explain the rationale behind the recommended reaction conditions, limiting their utility in high-stakes scientific workflows. In this work, we propose ChemMAS, a multi-agent system that reframes condition prediction as an evidence-based reasoning task. ChemMAS decomposes the task into mechanistic grounding, multi-channel recall, constraint-aware agentic debate, and rationale aggregation. Each decision is backed by interpretable justifications grounded in chemical knowledge and retrieved precedents. Experiments show that ChemMAS achieves 20-35% gains over domain-specific baselines and outperforms general-purpose LLMs by 10-15% in Top-1 accuracy, while offering falsifiable, human-trustable rationales, which establishes a new paradigm for explainable AI in scientific discovery. 

---
# Anchored Supervised Fine-Tuning 

**Authors**: He Zhu, Junyou Su, Peng Lai, Ren Ma, Wenjia Zhang, Linyi Yang, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23753)  

**Abstract**: Post-training of large language models involves a fundamental trade-off between supervised fine-tuning (SFT), which efficiently mimics demonstrations but tends to memorize, and reinforcement learning (RL), which achieves better generalization at higher computational cost. Dynamic Fine-Tuning (DFT) recently emerged as a promising middle ground, reweighting SFT objectives with token probabilities and achieving improvements in certain reasoning domains, though it exhibits instability in other tasks. We provide a analysis of DFT through the reward-weighted regression (RWR) framework, revealing that it corresponds to a specific auxiliary distribution choice that yields provably tighter RL bounds than standard SFT. However, our analysis also uncovers a critical limitation: this construction lacks distributional anchoring, leading to progressive drift that undermines training stability. To address this, we propose Anchored Supervised Fine-Tuning (ASFT), which augments DFT's reweighting with lightweight KL regularization to preserve tightness while ensuring stability. Empirically, ASFT consistently outperforms both SFT and DFT across mathematical reasoning, medical knowledge grounding, and code generation, achieving substantial improvements with minimal computational overhead. Our RWR framework provides a systematic lens for understanding post-training methods and demonstrates that principled theoretical analysis leads to both stronger guarantees and practical gains. 

---
# Beyond Game Theory Optimal: Profit-Maximizing Poker Agents for No-Limit Holdem 

**Authors**: SeungHyun Yi, Seungjun Yi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23747)  

**Abstract**: Game theory has grown into a major field over the past few decades, and poker has long served as one of its key case studies. Game-Theory-Optimal (GTO) provides strategies to avoid loss in poker, but pure GTO does not guarantee maximum profit. To this end, we aim to develop a model that outperforms GTO strategies to maximize profit in No Limit Holdem, in heads-up (two-player) and multi-way (more than two-player) situations. Our model finds the GTO foundation and goes further to exploit opponents. The model first navigates toward many simulated poker hands against itself and keeps adjusting its decisions until no action can reliably beat it, creating a strong baseline that is close to the theoretical best strategy. Then, it adapts by observing opponent behavior and adjusting its strategy to capture extra value accordingly. Our results indicate that Monte-Carlo Counterfactual Regret Minimization (CFR) performs best in heads-up situations and CFR remains the strongest method in most multi-way situations. By combining the defensive strength of GTO with real-time exploitation, our approach aims to show how poker agents can move from merely not losing to consistently winning against diverse opponents. 

---
# SafeSearch: Automated Red-Teaming for the Safety of LLM-Based Search Agents 

**Authors**: Jianshuo Dong, Sheng Guo, Hao Wang, Zhuotao Liu, Tianwei Zhang, Ke Xu, Minlie Huang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23694)  

**Abstract**: Search agents connect LLMs to the Internet, enabling access to broader and more up-to-date information. However, unreliable search results may also pose safety threats to end users, establishing a new threat surface. In this work, we conduct two in-the-wild experiments to demonstrate both the prevalence of low-quality search results and their potential to misguide agent behaviors. To counter this threat, we introduce an automated red-teaming framework that is systematic, scalable, and cost-efficient, enabling lightweight and harmless safety assessments of search agents. Building on this framework, we construct the SafeSearch benchmark, which includes 300 test cases covering five categories of risks (e.g., misinformation and indirect prompt injection). Using this benchmark, we evaluate three representative search agent scaffolds, covering search workflow, tool-calling, and deep research, across 7 proprietary and 8 open-source backend LLMs. Our results reveal substantial vulnerabilities of LLM-based search agents: when exposed to unreliable websites, the highest ASR reached 90.5% for GPT-4.1-mini under a search workflow setting. Moreover, our analysis highlights the limited effectiveness of common defense practices, such as reminder prompting. This emphasizes the value of our framework in promoting transparency for safer agent development. Our codebase and test cases are publicly available: this https URL. 

---
# HomeSafeBench: A Benchmark for Embodied Vision-Language Models in Free-Exploration Home Safety Inspection 

**Authors**: Siyuan Gao, Jiashu Yao, Haoyu Wen, Yuhang Guo, Zeming Liu, Heyan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23690)  

**Abstract**: Embodied agents can identify and report safety hazards in the home environments. Accurately evaluating their capabilities in home safety inspection tasks is curcial, but existing benchmarks suffer from two key limitations. First, they oversimplify safety inspection tasks by using textual descriptions of the environment instead of direct visual information, which hinders the accurate evaluation of embodied agents based on Vision-Language Models (VLMs). Second, they use a single, static viewpoint for environmental observation, which restricts the agents' free exploration and cause the omission of certain safety hazards, especially those that are occluded from a fixed viewpoint. To alleviate these issues, we propose HomeSafeBench, a benchmark with 12,900 data points covering five common home safety hazards: fire, electric shock, falling object, trips, and child safety. HomeSafeBench provides dynamic first-person perspective images from simulated home environments, enabling the evaluation of VLM capabilities for home safety inspection. By allowing the embodied agents to freely explore the room, HomeSafeBench provides multiple dynamic perspectives in complex environments for a more thorough inspection. Our comprehensive evaluation of mainstream VLMs on HomeSafeBench reveals that even the best-performing model achieves an F1-score of only 10.23%, demonstrating significant limitations in current VLMs. The models particularly struggle with identifying safety hazards and selecting effective exploration strategies. We hope HomeSafeBench will provide valuable reference and support for future research related to home security inspections. Our dataset and code will be publicly available soon. 

---
# Towards a Comprehensive Scaling Law of Mixture-of-Experts 

**Authors**: Guoliang Zhao, Yuhan Fu, Shuaipeng Li, Xingwu Sun, Ruobing Xie, An Wang, Weidong Han, Zhen Yang, Weixuan Sun, Yudong Zhang, Cheng-zhong Xu, Di Wang, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23678)  

**Abstract**: Mixture-of-Experts (MoE) models have become the consensus approach for enabling parameter-efficient scaling and cost-effective deployment in large language models. However, existing scaling laws for dense models are inapplicable to MoE models, which stems from three critical challenges: the multiplicity of influencing factors, their intricate coupling relationships and the non-monotonic nature of their performance impacts. They collectively necessitate a fine-grained investigation into MoE-specific scaling laws. In this work, we perform a systematic decomposition of MoE settings, identifying five key factors that influence model performance from both size and structural perspectives (data size ($D$), total model size ($N$), activated model size ($N_a$), number of active experts ($G$) and the ratio of shared experts ($S$)). Specifically, we design $446$ controlled experiments to characterize their marginal effects, ultimately constructing a comprehensive and precise joint MoE scaling law that considers all essential factors. Furthermore, we derive the theoretically optimal and practically efficiency-aware optimal configurations for $G$, $S$ and $N_a/N$ with detailed analyses. Our results demonstrate that the optimal settings for $G$ and $S$ are independent of both the model architecture and data size. With the scaling of $N$, the optimal activation parameter ratio of $N_a/N$ becomes sparser. Our proposed MoE scaling law could function as an accurate and insightful guidance to facilitate future MoE model design and training. 

---
# From Reasoning to Answer: Empirical, Attention-Based and Mechanistic Insights into Distilled DeepSeek R1 Models 

**Authors**: Jue Zhang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23676)  

**Abstract**: Large Reasoning Models (LRMs) generate explicit reasoning traces alongside final answers, yet the extent to which these traces influence answer generation remains unclear. In this work, we conduct a three-stage investigation into the interplay between reasoning and answer generation in three distilled DeepSeek R1 models. First, through empirical evaluation, we demonstrate that including explicit reasoning consistently improves answer quality across diverse domains. Second, attention analysis reveals that answer tokens attend substantially to reasoning tokens, with certain mid-layer Reasoning-Focus Heads (RFHs) closely tracking the reasoning trajectory, including self-reflective cues. Third, we apply mechanistic interventions using activation patching to assess the dependence of answer tokens on reasoning activations. Our results show that perturbations to key reasoning tokens can reliably alter the final answers, confirming a directional and functional flow of information from reasoning to answer. These findings deepen our understanding of how LRMs leverage reasoning tokens for answer generation, highlighting the functional role of intermediate reasoning in shaping model outputs. Our data and code are publicly available at \href{this https URL}{this URL}. 

---
# RCI: A Score for Evaluating Global and Local Reasoning in Multimodal Benchmarks 

**Authors**: Amit Agarwal, Hitesh Laxmichand Patel, Srikant Panda, Hansa Meghwani, Jyotika Singh, Karan Dua, Paul Li, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2509.23673)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive results on vision-language benchmarks, yet it remains unclear whether these benchmarks assess genuine global reasoning or allow success via localized visual cues. Existing evaluation methods do not explicitly measure this distinction, hindering effective dataset curation and real-world focused model development.
We introduce Region Comprehension Index (RCI), the first model-based score to directly quantify a dataset's reliance on global versus local visual information. RCI systematically compares reference-model performance on image patches versus full images, revealing if tasks require holistic image understanding or can be solved with partial or localized visual cues.
When applying RCI to 13 widely used multimodal benchmarks, we observed that most of them favor localized reasoning and exhibit significant spatial biases, indicating potential risks in real-world applications. RCI equips researchers & practitioners with an actionable tool for diagnosing & mitigating these biases, enabling the construction of datasets and benchmarks to foster the development of robust, enterprise-ready multimodal systems. 

---
# From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation 

**Authors**: KaiWen Wei, Kejun He, Xiaomian Kang, Jie Zhang, Yuming Yang, Jiang Zhong, He Bai, Junnan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23649)  

**Abstract**: Generative recommendation, which directly generates item identifiers, has emerged as a promising paradigm for recommendation systems. However, its potential is fundamentally constrained by the reliance on purely autoregressive training. This approach focuses solely on predicting the next item while ignoring the rich internal structure of a user's interaction history, thus failing to grasp the underlying intent. To address this limitation, we propose Masked History Learning (MHL), a novel training framework that shifts the objective from simple next-step prediction to deep comprehension of history. MHL augments the standard autoregressive objective with an auxiliary task of reconstructing masked historical items, compelling the model to understand ``why'' an item path is formed from the user's past behaviors, rather than just ``what'' item comes next. We introduce two key contributions to enhance this framework: (1) an entropy-guided masking policy that intelligently targets the most informative historical items for reconstruction, and (2) a curriculum learning scheduler that progressively transitions from history reconstruction to future prediction. Experiments on three public datasets show that our method significantly outperforms state-of-the-art generative models, highlighting that a comprehensive understanding of the past is crucial for accurately predicting a user's future path. The code will be released to the public. 

---
# RIV: Recursive Introspection Mask Diffusion Vision Language Model 

**Authors**: YuQian Li, Limeng Qiao, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.23625)  

**Abstract**: Mask Diffusion-based Vision Language Models (MDVLMs) have achieved remarkable progress in multimodal understanding tasks. However, these models are unable to correct errors in generated tokens, meaning they lack self-correction capability. In this paper, we propose Recursive Introspection Mask Diffusion Vision Language Model (RIV), which equips the model with self-correction ability through two novel mechanisms. The first is Introspection Training, where an Introspection Model is introduced to identify errors within generated sequences. Introspection Training enables the model to detect not only grammatical and spelling mistakes, but more importantly, logical errors. The second is Recursive Inference. Beginning with the standard unmasking step, the learned Introspection Model helps to identify errors in the output sequence and remask them. This alternating ($\text{unmask}\rightarrow\text{introspection}\rightarrow\text{remask}$) process is repeated recursively until reliable results are obtained. Experimental results on multiple benchmarks demonstrate that the proposed RIV achieves state-of-the-art performance, outperforming most existing MDVLMs. 

---
# Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment 

**Authors**: Min-Hsuan Yeh, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23564)  

**Abstract**: Human feedback plays a pivotal role in aligning large language models (LLMs) with human preferences. However, such feedback is often noisy or inconsistent, which can degrade the quality of reward models and hinder alignment. While various automated data cleaning methods have been proposed to mitigate this issue, a systematic evaluation of their effectiveness and generalizability remains lacking. To bridge this gap, we introduce the first comprehensive benchmark for evaluating 13 preference data cleaning methods in the context of LLM alignment. PrefCleanBench offers a standardized protocol to assess cleaning strategies in terms of alignment performance and generalizability across diverse datasets, model architectures, and optimization algorithms. By unifying disparate methods and rigorously comparing them, we uncover key factors that determine the success of data cleaning in alignment tasks. This benchmark lays the groundwork for principled and reproducible approaches to improving LLM alignment through better data quality-highlighting the crucial but underexplored role of data preprocessing in responsible AI development. We release modular implementations of all methods to catalyze further research: this https URL. 

---
# Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional 

**Authors**: Divyam Madaan, Varshan Muhunthan, Kyunghyun Cho, Sumit Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2509.23499)  

**Abstract**: Understanding the interplay between intra-modality dependencies (the contribution of an individual modality to a target task) and inter-modality dependencies (the relationships between modalities and the target task) is fundamental to advancing multi-modal learning. However, the nature of and interaction between these dependencies within current benchmark evaluations remains poorly characterized. In this work, we present a large-scale empirical study to quantify these dependencies across 23 visual question-answering benchmarks using multi-modal large language models (MLLMs) covering domains such as general and expert knowledge reasoning, optical character recognition, and document understanding. Our findings show that the reliance on vision, question (text), and their interaction varies significantly, both across and within benchmarks. We discover that numerous benchmarks intended to mitigate text-only biases have inadvertently amplified image-only dependencies. This characterization persists across model sizes, as larger models often use these intra-modality dependencies to achieve high performance that mask an underlying lack of multi-modal reasoning. We provide a quantitative characterization of multi-modal datasets, enabling a principled approach to multi-modal benchmark design and evaluation. 

---
# Mapping Overlaps in Benchmarks through Perplexity in the Wild 

**Authors**: Siyang Wu, Honglin Bao, Sida Li, Ari Holtzman, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2509.23488)  

**Abstract**: We develop signatures of capacity familiarity to characterize large language model (LLM) benchmarks and their meaningful overlaps. Benchmark signatures probe the capacity required for benchmark performance. We formally define them as a set of salient tokens drawn from in-the-wild, naturally authored corpora, where LLM token perplexity, reflecting more or less pre-training exposure, becomes highly predictive of LLM benchmark performance. Through a large-scale meta-evaluation, we extract benchmark signatures via stepwise forward selection with linear regressions across 32 LLMs and 88 benchmarks spanning diverse knowledge, coding, logic, instruction following, math, language, reasoning, and world modeling. Our analysis situates signatures in relation to both the semantic similarity of benchmark questions and the correlation of model performance. While performance overlaps are universally high and semantic overlaps remain confined to a narrow mid-range, benchmark signatures prove highly informative in capturing variation, overlap, and divergence. We observe overlap in knowledge and reasoning subtasks, whereas multilingual and cultural benchmarks exhibit less similarity, even compared to cross-task overlap. Notably, performance-level results are strongly influenced by benchmark-orthogonal factors such as question format, highlighting limitations in LLM generalization, the conflation of performance with ability, and issues inherent in current mainstream benchmark agreement studies. Benchmark signatures, however, remain robust to such effects. Ultimately, we identify cross-functional overlaps across logic, math, language, instruction following, and world modeling, with coding emerging as the least overlapping domain. Together, these findings provide mechanistic insights into benchmark validity and LLM sensitivities, and sketch the underlying landscape of interconnected LLM capabilities. 

---
# Temporal Generalization: A Reality Check 

**Authors**: Divyam Madaan, Sumit Chopra, Kyunghyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2509.23487)  

**Abstract**: Machine learning (ML) models often struggle to maintain performance under distribution shifts, leading to inaccurate predictions on unseen future data. In this work, we investigate whether and under what conditions models can achieve such a generalization when relying solely on past data. We explore two primary approaches: convex combinations of past model parameters (\emph{parameter interpolation}) and explicit extrapolation beyond the convex hull of past parameters (\emph{parameter extrapolation}). We benchmark several methods within these categories on a diverse set of temporal tasks, including language modeling, news summarization, news tag prediction, academic paper categorization, satellite image-based land use classification over time, and historical yearbook photo gender prediction. Our empirical findings show that none of the evaluated methods consistently outperforms the simple baseline of using the latest available model parameters in all scenarios. In the absence of access to future data or robust assumptions about the underlying data-generating process, these results underscore the inherent difficulties of generalizing and extrapolating to future data and warrant caution when evaluating claims of such generalization. 

---
# MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction 

**Authors**: Sepideh Abedini, Shubhankar Mohapatra, D. B. Emerson, Masoumeh Shafieinejad, Jesse C. Cresswell, Xi He  

**Link**: [PDF](https://arxiv.org/pdf/2509.23459)  

**Abstract**: Large language models (LLMs) have shown promising performance on tasks that require reasoning, such as text-to-SQL, code generation, and debugging. However, regulatory frameworks with strict privacy requirements constrain their integration into sensitive systems. State-of-the-art LLMs are also proprietary, costly, and resource-intensive, making local deployment impractical. Consequently, utilizing such LLMs often requires sharing data with third-party providers, raising privacy concerns and risking noncompliance with regulations. Although fine-tuned small language models (SLMs) can outperform LLMs on certain tasks and be deployed locally to mitigate privacy concerns, they underperform on more complex tasks such as text-to-SQL translation. In this work, we introduce MaskSQL, a text-to-SQL framework that utilizes abstraction as a privacy protection mechanism to mask sensitive information in LLM prompts. Unlike redaction, which removes content entirely, or generalization, which broadens tokens, abstraction retains essential information while discarding unnecessary details, striking an effective privacy-utility balance for the text-to-SQL task. Moreover, by providing mechanisms to control the privacy-utility tradeoff, MaskSQL facilitates adoption across a broader range of use cases. Our experimental results show that MaskSQL outperforms leading SLM-based text-to-SQL models and achieves performance approaching state-of-the-art LLM-based models, while preserving privacy. 

---
# FoR-SALE: Frame of Reference-guided Spatial Adjustment in LLM-based Diffusion Editing 

**Authors**: Tanawan Premsri, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23452)  

**Abstract**: Frame of Reference (FoR) is a fundamental concept in spatial reasoning that humans utilize to comprehend and describe space. With the rapid progress in Multimodal Language models, the moment has come to integrate this long-overlooked dimension into these models. In particular, in text-to-image (T2I) generation, even state-of-the-art models exhibit a significant performance gap when spatial descriptions are provided from perspectives other than the camera. To address this limitation, we propose Frame of Reference-guided Spatial Adjustment in LLM-based Diffusion Editing (FoR-SALE), an extension of the Self-correcting LLM-controlled Diffusion (SLD) framework for T2I. For-Sale evaluates the alignment between a given text and an initially generated image, and refines the image based on the Frame of Reference specified in the spatial expressions. It employs vision modules to extract the spatial configuration of the image, while simultaneously mapping the spatial expression to a corresponding camera perspective. This unified perspective enables direct evaluation of alignment between language and vision. When misalignment is detected, the required editing operations are generated and applied. FoR-SALE applies novel latent-space operations to adjust the facing direction and depth of the generated images. We evaluate FoR-SALE on two benchmarks specifically designed to assess spatial understanding with FoR. Our framework improves the performance of state-of-the-art T2I models by up to 5.3% using only a single round of correction. 

---
# SPIKE-RL: Video-LLMs meet Bayesian Surprise 

**Authors**: Sahithya Ravi, Aditya Chinchure, Raymond T. Ng, Leonid Sigal, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2509.23433)  

**Abstract**: Real-world videos often show routine activities punctuated by memorable, surprising events. However, most Video-LLMs process videos by sampling frames uniformly, likely missing critical moments that define a video's narrative. We introduce SPIKE, an inference-time framework that quantifies Bayesian Surprise as the belief update triggered by new visual evidence in the video stream, identifying moments where new visual evidence conflicts with prior beliefs. SPIKE effectively localizes surprise in videos, strongly correlated with humans on positive (FunQA) and negative (Oops!) surprise benchmarks. Since the beliefs of zero-shot Video-LLMs are often suboptimal, we develop SPIKE-RL, which leverages GRPO to optimize belief hypotheses based on a reward signal from the video caption. SPIKE and SPIKE-RL guide query-agnostic surprise-weighted frame sampling, which allocates more frames to interesting moments in the video. With this strategy, we achieve consistent performance gains on five downstream benchmarks over uniform sampling. By enabling Video-LLMs to track beliefs and register surprise, our work paves the way for more robust models that can revise their understanding in response to new information. 

---
# Your Models Have Thought Enough: Training Large Reasoning Models to Stop Overthinking 

**Authors**: Jinyi Han, Ying Huang, Ying Liao, Zishang Jiang, Xikun Lu, Haiquan Zhao, Xinyi Wang, Guanghao Zhou, Sihang Jiang, Jiaqing Liang, Weikang Zhou, Zeye Sun, Fei Yu, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23392)  

**Abstract**: Large Reasoning Models (LRMs) have achieved impressive performance on challenging tasks, yet their deep reasoning often incurs substantial computational costs. To achieve efficient reasoning, existing reinforcement learning methods still struggle to construct short reasoning path during the rollout stage, limiting effective learning. Inspired by Evidence Accumulation Models, we find that LRMs have accumulated sufficient information early in reasoning, making further reasoning steps redundant. Based on this insight, we propose Just-Enough Thinking (JET), which trains models to proactively terminate unnecessary reasoning. JET performs trajectory truncation during rollout to expose the model to short, distributionally consistent reasoning paths. Besides, it uses a quality-controlled length reward to better encourage concise reasoning while maintaining correctness. Extensive experiments demonstrate that JET significantly improves reasoning efficiency without sacrificing accuracy. Especially, DeepSeek-Distill-Qwen-1.5B achieves a 4.6% accuracy gain while reducing output length by 46.3% on the Olympiad benchmark. Our code is available in the GitHub. 

---
# PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation 

**Authors**: Wei Zhou, Guoliang Li, Haoyu Wang, Yuxing Han, Xufei Wu, Fan Wu, Xuanhe Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.23338)  

**Abstract**: Large language models (LLMS) have shown increasing effectiveness in Text-to-SQL tasks. However, another closely related problem, Cross-System SQL Translation (a.k.a., SQL-to-SQL), which adapts a query written for one database system (e.g., MySQL) into its equivalent one for another system (e.g., ClickHouse), is of great practical importance but remains underexplored. Existing SQL benchmarks are not well-suited for SQL-to-SQL evaluation, which (1) focus on a limited set of database systems (often just SQLite) and (2) cannot capture many system-specific SQL dialects (e.g., customized functions, data types, and syntax rules). Thus, in this paper, we introduce PARROT, a Practical And Realistic BenchmaRk for CrOss-System SQL Translation. PARROT comprises 598 translation pairs from 38 open-source benchmarks and real-world business services, specifically prepared to challenge system-specific SQL understanding (e.g., LLMS achieve lower than 38.53% accuracy on average). We also provide multiple benchmark variants, including PARROT-Diverse with 28,003 translations (for extensive syntax testing) and PARROT-Simple with 5,306 representative samples (for focused stress testing), covering 22 production-grade database systems. To promote future research, we release a public leaderboard and source code at: this https URL. 

---
# Seeing Symbols, Missing Cultures: Probing Vision-Language Models' Reasoning on Fire Imagery and Cultural Meaning 

**Authors**: Haorui Yu, Qiufeng Yi, Yijia Chu, Yang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23311)  

**Abstract**: Vision-Language Models (VLMs) often appear culturally competent but rely on superficial pattern matching rather than genuine cultural understanding. We introduce a diagnostic framework to probe VLM reasoning on fire-themed cultural imagery through both classification and explanation analysis. Testing multiple models on Western festivals, non-Western traditions, and emergency scenes reveals systematic biases: models correctly identify prominent Western festivals but struggle with underrepresented cultural events, frequently offering vague labels or dangerously misclassifying emergencies as celebrations. These failures expose the risks of symbolic shortcuts and highlight the need for cultural evaluation beyond accuracy metrics to ensure interpretable and fair multimodal systems. 

---
# Learning How to Use Tools, Not Just When: Pattern-Aware Tool-Integrated Reasoning 

**Authors**: Ningning Xu, Yuxuan Jiang, Shubhashis Roy Dipta  

**Link**: [PDF](https://arxiv.org/pdf/2509.23292)  

**Abstract**: Tool-integrated reasoning (TIR) has become a key approach for improving large reasoning models (LRMs) on complex problems. Prior work has mainly studied when to invoke tools, while overlooking how tools are applied. We identify two common patterns: a calculator pattern that uses code for direct computation, and an algorithmic pattern that encodes problems as programs. Misaligned choices often cause failures even when reasoning is sound. We propose a two-stage framework that first builds code competence from both patterns and then aligns pattern selection with teacher preferences. Across challenging math datasets, our pattern-aware method substantially improves both code usage and accuracy, for instance raising Code@1 on MATH500 from 64.0% to 70.5% and on AIME24 from 26.7% to 50.0%. These gains highlight the effectiveness of a pattern-aware approach for tool-integrated reasoning. 

---
# $p$-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding 

**Authors**: Runyan Tan, Shuang Wu, Phillip Howard  

**Link**: [PDF](https://arxiv.org/pdf/2509.23234)  

**Abstract**: Obtaining high-quality outputs from Large Language Models (LLMs) often depends upon the choice of a sampling-based decoding strategy to probabilistically choose the next token at each generation step. While a variety of such sampling methods have been proposed, their performance can be sensitive to the selection of hyperparameters which may require different settings depending upon the generation task and temperature configuration. In this work, we introduce $p$-less sampling: an information-theoretic approach to sampling which dynamically sets a truncation threshold at each decoding step based on the entire token probability distribution. Unlike existing methods, $p$-less sampling has no hyperparameters and consistently produces high-quality outputs as temperature increases. We provide theoretical perspectives on $p$-less sampling to ground our proposed method and conduct experiments to empirically validate its effectiveness across a range of math, logical reasoning, and creative writing tasks. Our results demonstrate how $p$-less sampling consistently outperforms existing sampling approaches while exhibiting much less degradation in text quality at higher temperature values. We further show how $p$-less achieves greater inference-time efficiency than alternative methods through lower average token sampling times and shorter generation lengths, without sacrificing accuracy. Finally, we provide analyses to highlight the benefits of $p$-less through qualitative examples, case studies, and diversity assessments. 

---
# SPEC-RL: Accelerating On-Policy Reinforcement Learning via Speculative Rollouts 

**Authors**: Bingshuai Liu, Ante Wang, Zijun Min, Liang Yao, Haibo Zhang, Yang Liu, Anxiang Zeng, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2509.23232)  

**Abstract**: Large Language Models (LLMs) increasingly rely on reinforcement learning with verifiable rewards (RLVR) to elicit reliable chain-of-thought reasoning. However, the training process remains bottlenecked by the computationally expensive rollout stage. Existing acceleration methods-such as parallelization, objective- and data-driven modifications, and replay buffers-either incur diminishing returns, introduce bias, or overlook redundancy across iterations. We identify that rollouts from consecutive training epochs frequently share a large portion of overlapping segments, wasting computation. To address this, we propose SPEC-RL, a novel framework that integrates SPECulative decoding with the RL rollout process. SPEC-RL reuses prior trajectory segments as speculative prefixes and extends them via a draft-and-verify mechanism, avoiding redundant generation while ensuring policy consistency. Experiments on diverse math reasoning and generalization benchmarks, including GSM8K, MATH-500, OlympiadBench, MMLU-STEM, and others, demonstrate that SPEC-RL reduces rollout time by 2-3x without compromising policy quality. As a purely rollout-stage enhancement, SPEC-RL integrates seamlessly with mainstream algorithms (e.g., PPO, GRPO, DAPO), offering a general and practical path to scale RLVR for large reasoning models. Our code is available at this https URL 

---
# C$^2$GSPG: Confidence-calibrated Group Sequence Policy Gradient towards Self-aware Reasoning 

**Authors**: Haotian Liu, Shuo Wang, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23129)  

**Abstract**: Reinforcement Learning (RL) methods, exemplified by Group Relative Policy Optimization (GRPO) and its variants, play a central role in developing reasoning models. However, these methods often suffer from a critical overconfidence issue, which prevents them from achieving self-aware reasoning models. In this study, we propose a simple yet effective confidence-calibration group sequence policy gradient method, called C$^2$GSPG, which simultaneously enhances reasoning performance while suppressing overconfidence. In principle, we propose a Group Sequence Policy Gradient (GSPG) framework for learning reasoning models, which eliminates the token-level bias commonly appearing in GRPO and its variants. In this framework, we define the model confidence for each reasoning problem using the normalized sequence-level probability, and then apply a cross-entropy regularizer to calibrate the model confidence to the sequence's reward. We demonstrate that the confidence calibration regularizer and GSPG are collaborative for binary rewards, as their objectives always share the same gradient direction. For non-binary rewards, we apply nonlinear reward normalization and adaptive regularizer clipping, mitigating the potential conflict between the two objectives. Applying C$^2$GSPG to post-train large language models in logical and mathematical reasoning tasks, we show its superiority over state-of-the-art methods in both reasoning accuracy and confidence calibration. The code of C$^2$GSPG is available at this https URL. 

---
# RHYTHM: Reasoning with Hierarchical Temporal Tokenization for Human Mobility 

**Authors**: Haoyu He, Haozheng Luo, Yan Chen, Qi R. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23115)  

**Abstract**: Predicting human mobility is inherently challenging due to complex long-range dependencies and multi-scale periodic behaviors. To address this, we introduce RHYTHM (Reasoning with Hierarchical Temporal Tokenization for Human Mobility), a unified framework that leverages large language models (LLMs) as general-purpose spatio-temporal predictors and trajectory reasoners. Methodologically, RHYTHM employs temporal tokenization to partition each trajectory into daily segments and encode them as discrete tokens with hierarchical attention that captures both daily and weekly dependencies, thereby significantly reducing the sequence length while preserving cyclical information. Additionally, we enrich token representations by adding pre-computed prompt embeddings for trajectory segments and prediction targets via a frozen LLM, and feeding these combined embeddings back into the LLM backbone to capture complex interdependencies. Computationally, RHYTHM freezes the pretrained LLM's backbone to reduce attention complexity and memory cost. We evaluate our model against state-of-the-art methods using three real-world datasets. Notably, RHYTHM achieves a 2.4% improvement in overall accuracy, a 5.0% increase on weekends, and a 24.6% reduction in training time. Code is publicly available at this https URL. 

---
# Multiplayer Nash Preference Optimization 

**Authors**: Fang Wu, Xu Huang, Weihao Xuan, Zhiwei Zhang, Yijia Xiao, Guancheng Wan, Xiaomin Li, Bing Hu, Peng Xia, Jure Leskovec, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23102)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has emerged as the standard paradigm for aligning large language models (LLMs) with human preferences. However, reward-based methods built on the Bradley-Terry assumption struggle to capture the non-transitive and heterogeneous nature of real-world preferences. To address this, recent studies have reframed alignment as a two-player Nash game, giving rise to Nash learning from human feedback (NLHF). While this perspective has inspired algorithms such as INPO, ONPO, and EGPO with strong theoretical and empirical guarantees, they remain fundamentally restricted to two-player interactions, creating a single-opponent bias that fails to capture the full complexity of realistic preference structures. In this work, we introduce Multiplayer Nash Preference Optimization (MNPO), a novel framework that generalizes NLHF to the multiplayer regime. It formulates alignment as an $n$-player game, where each policy competes against a population of opponents while being regularized toward a reference model. Our framework establishes well-defined Nash equilibria in multiplayer settings and extends the concept of duality gap to quantify approximation quality. We demonstrate that MNPO inherits the equilibrium guarantees of two-player methods while enabling richer competitive dynamics and improved coverage of diverse preference structures. Through comprehensive empirical evaluation, we show that MNPO consistently outperforms existing NLHF baselines on instruction-following benchmarks, achieving superior alignment quality under heterogeneous annotator conditions and mixed-policy evaluation scenarios. Together, these results establish MNPO as a principled and scalable framework for aligning LLMs with complex, non-transitive human preferences. Code is available at this https URL. 

---
# Causally-Enhanced Reinforcement Policy Optimization 

**Authors**: Xiangqi Wang, Yue Huang, Yujun Zhou, Xiaonan Luo, Kehan Guo, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23095)  

**Abstract**: Large language models (LLMs) trained with reinforcement objectives often achieve superficially correct answers via shortcut strategies, pairing correct outputs with spurious or unfaithful reasoning and degrading under small causal perturbations. We introduce Causally-Enhanced Policy Optimization (CE-PO), a drop-in reward-shaping framework that augments policy optimization with a differentiable proxy for causal coherence along the generation pathway from prompt (Z) to rationale (X) to answer (Y). CE-PO estimates model-internal influence with Jacobian-based sensitivities, counterfactually hardens these signals to suppress nuisance cues, and fuses the resulting coherence score with task-accuracy feedback via a Minkowski (power-mean) combiner, exposing a single tunable between accuracy and coherence trade-off. The unified reward integrates with PPO/GRPO without architectural changes. Across reasoning benchmarks and causal stress tests, CE-PO reduces reward hacking and unfaithful chain-of-thought while improving robustness to correlation-causation flips and light counterfactual edits, all at near-parity accuracy. Experimental results across 4 datasets show that CE-PO improves accuracy over baselines by 5.49% on average (up to 9.58%), while improving robustness to correlation-causation flips and light counterfactual edits. 

---
# Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents 

**Authors**: Zonghan Yang, Shengjie Wang, Kelin Fu, Wenyang He, Weimin Xiong, Yibo Liu, Yibo Miao, Bofei Gao, Yejie Wang, Yingwei Ma, Yanhao Li, Yue Liu, Zhenxing Hu, Kaitai Zhang, Shuyi Wang, Huarong Chen, Flood Sung, Yang Liu, Yang Gao, Zhilin Yang, Tianyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23045)  

**Abstract**: Large Language Models (LLMs) are increasingly applied to software engineering (SWE), with SWE-bench as a key benchmark. Solutions are split into SWE-Agent frameworks with multi-turn interactions and workflow-based Agentless methods with single-turn verifiable steps. We argue these paradigms are not mutually exclusive: reasoning-intensive Agentless training induces skill priors, including localization, code edit, and self-reflection that enable efficient and effective SWE-Agent adaptation. In this work, we first curate the Agentless training recipe and present Kimi-Dev, an open-source SWE LLM achieving 60.4\% on SWE-bench Verified, the best among workflow approaches. With additional SFT adaptation on 5k publicly-available trajectories, Kimi-Dev powers SWE-Agents to 48.6\% pass@1, on par with that of Claude 3.5 Sonnet (241022 version). These results show that structured skill priors from Agentless training can bridge workflow and agentic frameworks for transferable coding agents. 

---
# Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data 

**Authors**: Zi Liang, Qingqing Ye, Xuan Liu, Yanyun Wang, Jianliang Xu, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23041)  

**Abstract**: Synthetic data refers to artificial samples generated by models. While it has been validated to significantly enhance the performance of large language models (LLMs) during training and has been widely adopted in LLM development, potential security risks it may introduce remain uninvestigated. This paper systematically evaluates the resilience of synthetic-data-integrated training paradigm for LLMs against mainstream poisoning and backdoor attacks. We reveal that such a paradigm exhibits strong resistance to existing attacks, primarily thanks to the different distribution patterns between poisoning data and queries used to generate synthetic samples. To enhance the effectiveness of these attacks and further investigate the security risks introduced by synthetic data, we introduce a novel and universal attack framework, namely, Virus Infection Attack (VIA), which enables the propagation of current attacks through synthetic data even under purely clean queries. Inspired by the principles of virus design in cybersecurity, VIA conceals the poisoning payload within a protective "shell" and strategically searches for optimal hijacking points in benign samples to maximize the likelihood of generating malicious content. Extensive experiments on both data poisoning and backdoor attacks show that VIA significantly increases the presence of poisoning content in synthetic data and correspondingly raises the attack success rate (ASR) on downstream models to levels comparable to those observed in the poisoned upstream models. 

---
# Tracing the Representation Geometry of Language Models from Pretraining to Post-training 

**Authors**: Melody Zixuan Li, Kumar Krishna Agrawal, Arna Ghosh, Komal Kumar Teru, Adam Santoro, Guillaume Lajoie, Blake A. Richards  

**Link**: [PDF](https://arxiv.org/pdf/2509.23024)  

**Abstract**: Standard training metrics like loss fail to explain the emergence of complex capabilities in large language models. We take a spectral approach to investigate the geometry of learned representations across pretraining and post-training, measuring effective rank (RankMe) and eigenspectrum decay ($\alpha$-ReQ). With OLMo (1B-7B) and Pythia (160M-12B) models, we uncover a consistent non-monotonic sequence of three geometric phases during autoregressive pretraining. The initial "warmup" phase exhibits rapid representational collapse. This is followed by an "entropy-seeking" phase, where the manifold's dimensionality expands substantially, coinciding with peak n-gram memorization. Subsequently, a "compression-seeking" phase imposes anisotropic consolidation, selectively preserving variance along dominant eigendirections while contracting others, a transition marked with significant improvement in downstream task performance. We show these phases can emerge from a fundamental interplay of cross-entropy optimization under skewed token frequencies and representational bottlenecks ($d \ll |V|$). Post-training further transforms geometry: SFT and DPO drive "entropy-seeking" dynamics to integrate specific instructional or preferential data, improving in-distribution performance while degrading out-of-distribution robustness. Conversely, RLVR induces "compression-seeking", enhancing reward alignment but reducing generation diversity. 

---
# Geometry-Aware Losses for Structure-Preserving Text-to-Sign Language Generation 

**Authors**: Zetian Wu, Tianshuo Zhou, Stefan Lee, Liang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23011)  

**Abstract**: Sign language translation from text to video plays a crucial role in enabling effective communication for Deaf and hard--of--hearing individuals. A major challenge lies in generating accurate and natural body poses and movements that faithfully convey intended meanings. Prior methods often neglect the anatomical constraints and coordination patterns of human skeletal motion, resulting in rigid or biomechanically implausible outputs. To address this, we propose a novel approach that explicitly models the relationships among skeletal joints--including shoulders, arms, and hands--by incorporating geometric constraints on joint positions, bone lengths, and movement dynamics. During training, we introduce a parent-relative reweighting mechanism to enhance finger flexibility and reduce motion stiffness. Additionally, bone-pose losses and bone-length constraints enforce anatomically consistent structures. Our method narrows the performance gap between the previous best and the ground-truth oracle by 56.51%, and further reduces discrepancies in bone length and movement variance by 18.76% and 5.48%, respectively, demonstrating significant gains in anatomical realism and motion naturalness. 

---
# Not only a helper, but also a teacher: Interactive LLM Cascade 

**Authors**: Yu Wu, Shuo Wu, Ye Tao, Yansong Li, Anand D. Sarwate  

**Link**: [PDF](https://arxiv.org/pdf/2509.22984)  

**Abstract**: Large Language Models (LLMs) vary widely in their capabilities, with larger models often having better performance but higher cost: choosing an LLM model often involves trading off performance and cost. The LLM Cascade is a paradigm that defers difficult queries from weak/cheap to strong/expensive models. This approach is nonadaptive: the deferral decision is trained offline. When confronted with similar or repeated queries, the LLM Cascade may then repeatedly consult the expensive model and incur higher cost. To improve the cascading efficiency, we propose Inter-Cascade, an online and interactive LLM Cascade that extends the role of strong model from a backup helper to a long-term teacher. In our system, when a strong model resolves a difficult query, it also distills its solution into a generalized, reusable problem-solving strategy that boosts the weak model on subsequent queries. Adding strategies to queries enables the weak model to dynamically improve its performance over time, avoiding computationally and time-intensive fine-tuning. Empirically, compared with standard LLM Cascade baselines across multiple benchmarks, the Inter-Cascade significantly improves the accuracy of the weak model (by up to 33.06 absolute percentage points) and the overall system (by up to 5.53 absolute percentage points), while reducing the calls to strong models (by up to 48.05% relative reduction) and saving the corresponding fees (by up to 49.63% relative reduction). Inter-Cascade demonstrates the effective in-context knowledge transfer between LLMs, and provides a general, scalable framework applicable to both open-source and API-based LLMs. 

---
# JE-IRT: A Geometric Lens on LLM Abilities through Joint Embedding Item Response Theory 

**Authors**: Louie Hong Yao, Nicholas Jarvis, Tiffany Zhan, Saptarshi Ghosh, Linfeng Liu, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22888)  

**Abstract**: Standard LLM evaluation practices compress diverse abilities into single scores, obscuring their inherently multidimensional nature. We present JE-IRT, a geometric item-response framework that embeds both LLMs and questions in a shared space. For question embeddings, the direction encodes semantics and the norm encodes difficulty, while correctness on each question is determined by the geometric interaction between the model and question embeddings. This geometry replaces a global ranking of LLMs with topical specialization and enables smooth variation across related questions. Building on this framework, our experimental results reveal that out-of-distribution behavior can be explained through directional alignment, and that larger norms consistently indicate harder questions. Moreover, JE-IRT naturally supports generalization: once the space is learned, new LLMs are added by fitting a single embedding. The learned space further reveals an LLM-internal taxonomy that only partially aligns with human-defined subject categories. JE-IRT thus establishes a unified and interpretable geometric lens that connects LLM abilities with the structure of questions, offering a distinctive perspective on model evaluation and generalization. 

---
# Patient-specific Biomolecular Instruction Tuning 

**Authors**: Irsyad Adam, Zekai Chen, David Laub, Shaun Porwal, Arda Pekis, Kevin Brown  

**Link**: [PDF](https://arxiv.org/pdf/2509.22853)  

**Abstract**: Proteomics data is essential to pathogenic understanding of a disease phenotype. In cancer, analysis of molecular signatures enables precision medicine through the identification of biological processes that drive individualized tumor progression, therapeutic resistance, and clinical heterogeneity. Recent advances in multimodal large language models (LLMs) have shown remarkable capacity to integrate and reason across heterogeneous data modalities. However, performing multi-modal language modeling for molecular understanding of patient-specific proteomics remains a significant challenge due to two barriers: (1) the lack of instruction-tuning datasets that enable clinical interpretation from proteomics data, and (2) the absence of language modeling architectures designed to capture the rich heterogeneity of molecular data. In this work, we introduce CPTAC-PROTSTRUCT, the first instruction tuning dataset for molecular understanding of oncology, comprising over 400k open-ended examples derived from individualized proteomic profiles curated from the largest national proteomics cancer study (CPTAC). Additionally, we propose KRONOS (Knowledge Representation of patient Omics Networks in Oncology via Structured tuning), a novel graph-LLM framework that leverages molecular interaction topology with proteomics to learn patient-specific graph representations for enhanced clinical reasoning. We show that KRONOS achieves competitive performance across benchmark clinical tasks, including molecular classification, temporal trajectory modeling, and tumor stage prediction from proteomics data. Ultimately, this approach empowers LLMs to understand patient-level pathogenesis, advancing precision medicine through more accurate diagnosis, prognosis, and treatment stratification. 

---
# Adaptive Margin RLHF via Preference over Preferences 

**Authors**: Yaswanth Chittepu, Prasann Singhal, Greg Durrett, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2509.22851)  

**Abstract**: Margin-based optimization is fundamental to improving generalization and robustness in classification tasks. In the context of reward model learning from preferences within Reinforcement Learning from Human Feedback (RLHF), existing methods typically rely on no margins, fixed margins, or margins that are simplistic functions of preference ratings. However, such formulations often fail to account for the varying strengths of different preferences, for example some preferences are associated with larger margins between responses, or they rely on noisy margin information derived from ratings. We argue that modeling the strength of preferences can lead to better generalization and more faithful alignment. Furthermore, many existing methods that use adaptive margins assume access to accurate preference scores, which can be difficult for humans to provide reliably. We propose an approach that leverages preferences over preferences, that is annotations indicating which of two preferences reflects a stronger distinction. We use this ordinal signal to infer adaptive margins on a per-datapoint basis. We introduce an extension to Direct Preference Optimization (DPO), DPO-PoP, that incorporates adaptive margins from preference-over-preference supervision, enabling improved discriminative and generative performance. Empirically, our method outperforms vanilla DPO, DPO with fixed margins, and DPO with ground-truth margins on the UltraFeedback dataset. Additionally, we show that there is a tradeoff between discriminative and generative performance: improving test classification accuracy, particularly by correctly labeling weaker preferences at the expense of stronger ones, can lead to a decline in generative quality. To navigate this tradeoff, we propose two sampling strategies to gather preference-over-preference labels: one favoring discriminative performance and one favoring generative performance. 

---
# Toward a Theory of Generalizability in LLM Mechanistic Interpretability Research 

**Authors**: Sean Trott  

**Link**: [PDF](https://arxiv.org/pdf/2509.22831)  

**Abstract**: Research on Large Language Models (LLMs) increasingly focuses on identifying mechanistic explanations for their behaviors, yet the field lacks clear principles for determining when (and how) findings from one model instance generalize to another. This paper addresses a fundamental epistemological challenge: given a mechanistic claim about a particular model, what justifies extrapolating this finding to other LLMs -- and along which dimensions might such generalizations hold? I propose five potential axes of correspondence along which mechanistic claims might generalize, including: functional (whether they satisfy the same functional criteria), developmental (whether they develop at similar points during pretraining), positional (whether they occupy similar absolute or relative positions), relational (whether they interact with other model components in similar ways), and configurational (whether they correspond to particular regions or structures in weight-space). To empirically validate this framework, I analyze "1-back attention heads" (components attending to previous tokens) across pretraining in random seeds of the Pythia models (14M, 70M, 160M, 410M). The results reveal striking consistency in the developmental trajectories of 1-back attention across models, while positional consistency is more limited. Moreover, seeds of larger models systematically show earlier onsets, steeper slopes, and higher peaks of 1-back attention. I also address possible objections to the arguments and proposals outlined here. Finally, I conclude by arguing that progress on the generalizability of mechanistic interpretability research will consist in mapping constitutive design properties of LLMs to their emergent behaviors and mechanisms. 

---
# VideoScore2: Think before You Score in Generative Video Evaluation 

**Authors**: Xuan He, Dongfu Jiang, Ping Nie, Minghao Liu, Zhengxuan Jiang, Mingyi Su, Wentao Ma, Junru Lin, Chun Ye, Yi Lu, Keming Wu, Benjamin Schneider, Quy Duc Do, Zhuofeng Li, Yiming Jia, Yuxuan Zhang, Guo Cheng, Haozhe Wang, Wangchunshu Zhou, Qunshu Lin, Yuanxing Zhang, Ge Zhang, Wenhao Huang, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22799)  

**Abstract**: Recent advances in text-to-video generation have produced increasingly realistic and diverse content, yet evaluating such videos remains a fundamental challenge due to their multi-faceted nature encompassing visual quality, semantic alignment, and physical consistency. Existing evaluators and reward models are limited to single opaque scores, lack interpretability, or provide only coarse analysis, making them insufficient for capturing the comprehensive nature of video quality assessment. We present VideoScore2, a multi-dimensional, interpretable, and human-aligned framework that explicitly evaluates visual quality, text-to-video alignment, and physical/common-sense consistency while producing detailed chain-of-thought rationales. Our model is trained on a large-scale dataset VideoFeedback2 containing 27,168 human-annotated videos with both scores and reasoning traces across three dimensions, using a two-stage pipeline of supervised fine-tuning followed by reinforcement learning with Group Relative Policy Optimization (GRPO) to enhance analytical robustness. Extensive experiments demonstrate that VideoScore2 achieves superior performance with 44.35 (+5.94) accuracy on our in-domain benchmark VideoScore-Bench-v2 and 50.37 (+4.32) average performance across four out-of-domain benchmarks (VideoGenReward-Bench, VideoPhy2, etc), while providing interpretable assessments that bridge the gap between evaluation and controllable generation through effective reward modeling for Best-of-N sampling. Project Page: this https URL 

---
# DiaMoE-TTS: A Unified IPA-Based Dialect TTS Framework with Mixture-of-Experts and Parameter-Efficient Zero-Shot Adaptation 

**Authors**: Ziqi Chen, Gongyu Chen, Yihua Wang, Chaofan Ding, Zihao chen, Wei-Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22727)  

**Abstract**: Dialect speech embodies rich cultural and linguistic diversity, yet building text-to-speech (TTS) systems for dialects remains challenging due to scarce data, inconsistent orthographies, and complex phonetic variation. To address these issues, we present DiaMoE-TTS, a unified IPA-based framework that standardizes phonetic representations and resolves grapheme-to-phoneme ambiguities. Built upon the F5-TTS architecture, the system introduces a dialect-aware Mixture-of-Experts (MoE) to model phonological differences and employs parameter-efficient adaptation with Low-Rank Adaptors (LoRA) and Conditioning Adapters for rapid transfer to new dialects. Unlike approaches dependent on large-scale or proprietary resources, DiaMoE-TTS enables scalable, open-data-driven synthesis. Experiments demonstrate natural and expressive speech generation, achieving zero-shot performance on unseen dialects and specialized domains such as Peking Opera with only a few hours of data. 

---
# Benefits and Pitfalls of Reinforcement Learning for Language Model Planning: A Theoretical Perspective 

**Authors**: Siwei Wang, Yifei Shen, Haoran Sun, Shi Feng, Shang-Hua Teng, Li Dong, Yaru Hao, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22613)  

**Abstract**: Recent reinforcement learning (RL) methods have substantially enhanced the planning capabilities of Large Language Models (LLMs), yet the theoretical basis for their effectiveness remains elusive. In this work, we investigate RL's benefits and limitations through a tractable graph-based abstraction, focusing on policy gradient (PG) and Q-learning methods. Our theoretical analyses reveal that supervised fine-tuning (SFT) may introduce co-occurrence-based spurious solutions, whereas RL achieves correct planning primarily through exploration, underscoring exploration's role in enabling better generalization. However, we also show that PG suffers from diversity collapse, where output diversity decreases during training and persists even after perfect accuracy is attained. By contrast, Q-learning provides two key advantages: off-policy learning and diversity preservation at convergence. We further demonstrate that careful reward design is necessary to prevent reward hacking in Q-learning. Finally, applying our framework to the real-world planning benchmark Blocksworld, we confirm that these behaviors manifest in practice. 

---
# Multiplicative-Additive Constrained Models:Toward Joint Visualization of Interactive and Independent Effects 

**Authors**: Fumin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21923)  

**Abstract**: Interpretability is one of the considerations when applying machine learning to high-stakes fields such as healthcare that involve matters of life safety. Generalized Additive Models (GAMs) enhance interpretability by visualizing shape functions. Nevertheless, to preserve interpretability, GAMs omit higher-order interaction effects (beyond pairwise interactions), which imposes significant constraints on their predictive performance. We observe that Curve Ergodic Set Regression (CESR), a multiplicative model, naturally enables the visualization of its shape functions and simultaneously incorporates both interactions among all features and individual feature effects. Nevertheless, CESR fails to demonstrate superior performance compared to GAMs. We introduce Multiplicative-Additive Constrained Models (MACMs), which augment CESR with an additive part to disentangle the intertwined coefficients of its interactive and independent terms, thus effectively broadening the hypothesis space. The model is composed of a multiplicative part and an additive part, whose shape functions can both be naturally visualized, thereby assisting users in interpreting how features participate in the decision-making process. Consequently, MACMs constitute an improvement over both CESR and GAMs. The experimental results indicate that neural network-based MACMs significantly outperform both CESR and the current state-of-the-art GAMs in terms of predictive performance. 

---
# CAOTE: KV Cache Selection for LLMs via Attention Output Error-Based Token Eviction 

**Authors**: Raghavv Goel, Junyoung Park, Mukul Gagrani, Dalton Jones, Matthew Morse, Harper Langston, Mingu Lee, Chris Lott  

**Link**: [PDF](https://arxiv.org/pdf/2504.14051)  

**Abstract**: While long context support of large language models has extended their abilities, it also incurs challenges in memory and compute which becomes crucial bottlenecks in resource-restricted devices. Token eviction, a widely adopted post-training methodology designed to alleviate the bottlenecks by evicting less important tokens from the cache, typically uses attention scores as proxy metrics for token importance. However, one major limitation of attention score as a token-wise importance metrics is that it lacks the information about contribution of tokens to the attention output. In this paper, we propose a simple eviction criterion based on the contribution of cached tokens to attention outputs. Our method, CAOTE, optimizes for eviction error due to token eviction, by seamlessly integrating attention scores and value vectors. This is the first method which uses value tokens on top of attention-based eviction scores in closed-form. Additionally, CAOTE can act as a meta-heuristic method with flexible usage with any token eviction method. We show that CAOTE, when combined with the state-of-the-art attention score-based methods, always improves accuracies on the downstream task, indicating the importance of leveraging information from values during token eviction process. 

---
