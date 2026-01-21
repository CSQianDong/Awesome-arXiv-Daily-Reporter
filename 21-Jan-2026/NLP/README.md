# Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment 

**Authors**: Yuming Yang, Mingyoung Lai, Wanxu Zhao, Xiaoran Fan, Zhiheng Xi, Mingqi Wu, Chiyue Huang, Jun Zhao, Haijun Lv, Jian Tong, Yunhua Zhou, Yicheng Zou, Qipeng Guo, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14249)  

**Abstract**: Long chain-of-thought (CoT) trajectories provide rich supervision signals for distilling reasoning from teacher to student LLMs. However, both prior work and our experiments show that trajectories from stronger teachers do not necessarily yield better students, highlighting the importance of data-student suitability in distillation. Existing methods assess suitability primarily through student likelihood, favoring trajectories that closely align with the model's current behavior but overlooking more informative ones. Addressing this, we propose Rank-Surprisal Ratio (RSR), a simple metric that captures both alignment and informativeness to assess the suitability of a reasoning trajectory. RSR is motivated by the observation that effective trajectories typically combine low absolute probability with relatively high-ranked tokens under the student model, balancing learning signal strength and behavioral alignment. Concretely, RSR is defined as the ratio of a trajectory's average token-wise rank to its average negative log-likelihood, and is straightforward to compute and interpret. Across five student models and reasoning trajectories from 11 diverse teachers, RSR strongly correlates with post-training performance (average Spearman 0.86), outperforming existing metrics. We further demonstrate its practical utility in both trajectory selection and teacher selection. 

---
# APEX-Agents 

**Authors**: Bertie Vidgen, Austin Mann, Abby Fennelly, John Wright Stanly, Lucas Rothman, Marco Burstein, Julien Benchek, David Ostrofsky, Anirudh Ravichandran, Debnil Sur, Neel Venugopal, Alannah Hsia, Isaac Robinson, Calix Huang, Olivia Varones, Daniyal Khan, Michael Haines, Zach Richards, Chirag Mahapatra, Brendan Foody, Osvald Nitski  

**Link**: [PDF](https://arxiv.org/pdf/2601.14242)  

**Abstract**: We introduce the AI Productivity Index for Agents (APEX-Agents), a benchmark for assessing whether AI agents can execute long-horizon, cross-application tasks created by investment banking analysts, management consultants, and corporate lawyers. APEX-Agents requires agents to navigate realistic work environments with files and tools. We test eight agents for the leaderboard using Pass@1. Gemini 3 Flash (Thinking=High) achieves the highest score of 24.0%, followed by GPT-5.2 (Thinking=High), Claude Opus 4.5 (Thinking=High), and Gemini 3 Pro (Thinking=High). We open source the APEX-Agents benchmark (n=480) with all prompts, rubrics, gold outputs, files, and metadata. We also open-source Archipelago, our infrastructure for agent execution and evaluation. 

---
# MASCOT: Towards Multi-Agent Socio-Collaborative Companion Systems 

**Authors**: Yiyang Wang, Yiqiao Jin, Alex Cabral, Josiah Hester  

**Link**: [PDF](https://arxiv.org/pdf/2601.14230)  

**Abstract**: Multi-agent systems (MAS) have recently emerged as promising socio-collaborative companions for emotional and cognitive support. However, these systems frequently suffer from persona collapse--where agents revert to generic, homogenized assistant behaviors--and social sycophancy, which produces redundant, non-constructive dialogue. We propose MASCOT, a generalizable framework for multi-perspective socio-collaborative companions. MASCOT introduces a novel bi-level optimization strategy to harmonize individual and collective behaviors: 1) Persona-Aware Behavioral Alignment, an RLAIF-driven pipeline that finetunes individual agents for strict persona fidelity to prevent identity loss; and 2) Collaborative Dialogue Optimization, a meta-policy guided by group-level rewards to ensure diverse and productive discourse. Extensive evaluations across psychological support and workplace domains demonstrate that MASCOT significantly outperforms state-of-the-art baselines, achieving improvements of up to +14.1 in Persona Consistency and +10.6 in Social Contribution. Our framework provides a practical roadmap for engineering the next generation of socially intelligent multi-agent systems. 

---
# HALT: Hallucination Assessment via Latent Testing 

**Authors**: Rohan Bhatnagar, Youran Sun, Chi Andrew Zhang, Yixin Wen, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14210)  

**Abstract**: Hallucination in large language models (LLMs) can be understood as a failure of faithful readout: although internal representations may encode uncertainty about a query, decoding pressures still yield a fluent answer. We propose lightweight residual probes that read hallucination risk directly from intermediate hidden states of question tokens, motivated by the hypothesis that these layers retain epistemic signals that are attenuated in the final decoding stage. The probe is a small auxiliary network whose computation is orders of magnitude cheaper than token generation and can be evaluated fully in parallel with inference, enabling near-instantaneous hallucination risk estimation with effectively zero added latency in low-risk cases. We deploy the probe as an agentic critic for fast selective generation and routing, allowing LLMs to immediately answer confident queries while delegating uncertain ones to stronger verification pipelines. Across four QA benchmarks and multiple LLM families, the method achieves strong AUROC and AURAC, generalizes under dataset shift, and reveals interpretable structure in intermediate representations, positioning fast internal uncertainty readout as a principled foundation for reliable agentic AI. 

---
# Human Values in a Single Sentence: Moral Presence, Hierarchies, and Transformer Ensembles on the Schwartz Continuum 

**Authors**: Víctor Yeste, Paolo Rosso  

**Link**: [PDF](https://arxiv.org/pdf/2601.14172)  

**Abstract**: We study sentence-level identification of the 19 values in the Schwartz motivational continuum as a concrete formulation of human value detection in text. The setting - out-of-context sentences from news and political manifestos - features sparse moral cues and severe class imbalance. This combination makes fine-grained sentence-level value detection intrinsically difficult, even for strong modern neural models. We first operationalize a binary moral presence task ("does any value appear?") and show that it is learnable from single sentences (positive-class F1 $\approx$ 0.74 with calibrated thresholds). We then compare a presence-gated hierarchy to a direct multi-label classifier under matched compute, both based on DeBERTa-base and augmented with lightweight signals (prior-sentence context, LIWC-22/eMFD/MJD lexica, and topic features). The hierarchy does not outperform direct prediction, indicating that gate recall limits downstream gains. We also benchmark instruction-tuned LLMs - Gemma 2 9B, Llama 3.1 8B, Mistral 8B, and Qwen 2.5 7B - in zero-/few-shot and QLoRA setups and build simple ensembles; a soft-vote supervised ensemble reaches macro-F1 0.332, significantly surpassing the best single supervised model and exceeding prior English-only baselines. Overall, in this scenario, lightweight signals and small ensembles yield the most reliable improvements, while hierarchical gating offers limited benefit. We argue that, under an 8 GB single-GPU constraint and at the 7-9B scale, carefully tuned supervised encoders remain a strong and compute-efficient baseline for structured human value detection, and we outline how richer value structure and sentence-in-document context could further improve performance. 

---
# Domain-Adaptation through Synthetic Data: Fine-Tuning Large Language Models for German Law 

**Authors**: Ali Hamza Bashir, Muhammad Rehan Khalid, Kostadin Cvejoski, Jana Birr, Jule Berghaus, Armin Berger, Sandra Halscheidt, Christian Temath, Rafet Sifa, David Berghaus  

**Link**: [PDF](https://arxiv.org/pdf/2601.14160)  

**Abstract**: Large language models (LLMs) often struggle in specialized domains such as legal reasoning due to limited expert knowledge, resulting in factually incorrect outputs or hallucinations. This paper presents an effective method for adapting advanced LLMs to German legal question answering through a novel synthetic data generation approach. In contrast to costly human-annotated resources or unreliable synthetic alternatives, our approach systematically produces high-quality, diverse, and legally accurate question-answer pairs directly from authoritative German statutes. Using rigorous automated filtering methods and parameter-efficient fine-tuning techniques, we demonstrate that LLMs adapted with our synthetic dataset significantly outperform their baseline counterparts on German legal question answering tasks. Our results highlight the feasibility of using carefully designed synthetic data as a robust alternative to manual annotation in high-stakes, knowledge-intensive domains. 

---
# Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models 

**Authors**: Hyunjong Ok, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.14152)  

**Abstract**: Large language models exhibit surprising sensitivity to the structure of the prompt, but the mechanisms underlying this sensitivity remain poorly understood. In this work, we conduct an in-depth investigation on a striking case: in multiple-choice question answering, placing context before the questions and options (CQO) outperforms the reverse order (QOC) by over 14%p, consistently over a wide range of models and datasets. Through systematic architectural analysis, we identify causal attention as the core mechanism: in QOC prompts, the causal mask prevents option tokens from attending to context, creating an information bottleneck where context becomes invisible to options. 

---
# Style Transfer as Bias Mitigation: Diffusion Models for Synthetic Mental Health Text for Arabic 

**Authors**: Saad Mankarious, Aya Zirikly  

**Link**: [PDF](https://arxiv.org/pdf/2601.14124)  

**Abstract**: Synthetic data offers a promising solution for mitigating data scarcity and demographic bias in mental health analysis, yet existing approaches largely rely on pretrained large language models (LLMs), which may suffer from limited output diversity and propagate biases inherited from their training data. In this work, we propose a pretraining-free diffusion-based approach for synthetic text generation that frames bias mitigation as a style transfer problem. Using the CARMA Arabic mental health corpus, which exhibits a substantial gender imbalance, we focus on male-to-female style transfer to augment underrepresented female-authored content. We construct five datasets capturing varying linguistic and semantic aspects of gender expression in Arabic and train separate diffusion models for each setting. Quantitative evaluations demonstrate consistently high semantic fidelity between source and generated text, alongside meaningful surface-level stylistic divergence, while qualitative analysis confirms linguistically plausible gender transformations. Our results show that diffusion-based style transfer can generate high-entropy, semantically faithful synthetic data without reliance on pretrained LLMs, providing an effective and flexible framework for mitigating gender bias in sensitive, low-resource mental health domains. 

---
# A Systematic Analysis of Chunking Strategies for Reliable Question Answering 

**Authors**: Sofia Bennani, Charles Moslonka  

**Link**: [PDF](https://arxiv.org/pdf/2601.14123)  

**Abstract**: We study how document chunking choices impact the reliability of Retrieval-Augmented Generation (RAG) systems in industry. While practice often relies on heuristics, our end-to-end evaluation on Natural Questions systematically varies chunking method (token, sentence, semantic, code), chunk size, overlap, and context length. We use a standard industrial setup: SPLADE retrieval and a Mistral-8B generator. We derive actionable lessons for cost-efficient deployment: (i) overlap provides no measurable benefit and increases indexing cost; (ii) sentence chunking is the most cost-effective method, matching semantic chunking up to ~5k tokens; (iii) a "context cliff" reduces quality beyond ~2.5k tokens; and (iv) optimal context depends on the goal (semantic quality peaks at small contexts; exact match at larger ones). 

---
# NewsRECON: News article REtrieval for image CONtextualization 

**Authors**: Jonathan Tonglet, Iryna Gurevych, Tinne Tuytelaars, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2601.14121)  

**Abstract**: Identifying when and where a news image was taken is crucial for journalists and forensic experts to produce credible stories and debunk misinformation. While many existing methods rely on reverse image search (RIS) engines, these tools often fail to return results, thereby limiting their practical applicability. In this work, we address the challenging scenario where RIS evidence is unavailable. We introduce NewsRECON, a method that links images to relevant news articles to infer their date and location from article metadata. NewsRECON leverages a corpus of over 90,000 articles and integrates: (1) a bi-encoder for retrieving event-relevant articles; (2) two cross-encoders for reranking articles by location and event consistency. Experiments on the TARA and 5Pils-OOC show that NewsRECON outperforms prior work and can be combined with a multimodal large language model to achieve new SOTA results in the absence of RIS evidence. We make our code available. 

---
# Learning to Explain: Supervised Token Attribution from Transformer Attention Patterns 

**Authors**: George Mihaila  

**Link**: [PDF](https://arxiv.org/pdf/2601.14112)  

**Abstract**: Explainable AI (XAI) has become critical as transformer-based models are deployed in high-stakes applications including healthcare, legal systems, and financial services, where opacity hinders trust and accountability. Transformers self-attention mechanisms have proven valuable for model interpretability, with attention weights successfully used to understand model focus and behavior (Xu et al., 2015); (Wiegreffe and Pinter, 2019). However, existing attention-based explanation methods rely on manually defined aggregation strategies and fixed attribution rules (Abnar and Zuidema, 2020a); (Chefer et al., 2021), while model-agnostic approaches (LIME, SHAP) treat the model as a black box and incur significant computational costs through input perturbation. We introduce Explanation Network (ExpNet), a lightweight neural network that learns an explicit mapping from transformer attention patterns to token-level importance scores. Unlike prior methods, ExpNet discovers optimal attention feature combinations automatically rather than relying on predetermined rules. We evaluate ExpNet in a challenging cross-task setting and benchmark it against a broad spectrum of model-agnostic methods and attention-based techniques spanning four methodological families. 

---
# Truth with a Twist: The Rhetoric of Persuasion in Professional vs. Community-Authored Fact-Checks 

**Authors**: Olesya Razuvayevskaya, Kalina Bontcheva  

**Link**: [PDF](https://arxiv.org/pdf/2601.14105)  

**Abstract**: This study presents the first large-scale comparison of persuasion techniques present in crowd- versus professionally-written debunks. Using extensive datasets from Community Notes (CNs), EUvsDisinfo, and the Database of Known Fakes (DBKF), we quantify the prevalence and types of persuasion techniques across these fact-checking ecosystems. Contrary to prior hypothesis that community-produced debunks rely more heavily on subjective or persuasive wording, we find no evidence that CNs contain a higher average number of persuasion techniques than professional fact-checks. We additionally identify systematic rhetorical differences between CNs and professional debunking efforts, reflecting differences in institutional norms and topical coverage. Finally, we examine how the crowd evaluates persuasive language in CNs and show that, although notes with more persuasive elements receive slightly higher overall helpfulness ratings, crowd raters are effective at penalising the use of particular problematic rhetorical means 

---
# XCR-Bench: A Multi-Task Benchmark for Evaluating Cultural Reasoning in LLMs 

**Authors**: Mohsinul Kabir, Tasnim Ahmed, Md Mezbaur Rahman, Shaoxiong Ji, Hassan Alhuzali, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2601.14063)  

**Abstract**: Cross-cultural competence in large language models (LLMs) requires the ability to identify Culture-Specific Items (CSIs) and to adapt them appropriately across cultural contexts. Progress in evaluating this capability has been constrained by the scarcity of high-quality CSI-annotated corpora with parallel cross-cultural sentence pairs. To address this limitation, we introduce XCR-Bench, a Cross(X)-Cultural Reasoning Benchmark consisting of 4.9k parallel sentences and 1,098 unique CSIs, spanning three distinct reasoning tasks with corresponding evaluation metrics. Our corpus integrates Newmark's CSI framework with Hall's Triad of Culture, enabling systematic analysis of cultural reasoning beyond surface-level artifacts and into semi-visible and invisible cultural elements such as social norms, beliefs, and values. Our findings show that state-of-the-art LLMs exhibit consistent weaknesses in identifying and adapting CSIs related to social etiquette and cultural reference. Additionally, we find evidence that LLMs encode regional and ethno-religious biases even within a single linguistic setting during cultural adaptation. We release our corpus and code to facilitate future research on cross-cultural NLP. 

---
# Kakugo: Distillation of Low-Resource Languages into Small Language Models 

**Authors**: Peter Devine, Mardhiyah Sanni, Farid Adilazuarda, Julieta Gil Loizaga, Barry Haddow  

**Link**: [PDF](https://arxiv.org/pdf/2601.14051)  

**Abstract**: We present Kakugo, a novel and cost-effective pipeline designed to train general-purpose Small Language Models (SLMs) for low-resource languages using only the language name as input. By using a large teacher model to generate synthetic prompts and translate instruction datasets, we produced training data and SLMs for 54 low-resource languages. Evaluations across a diverse set of general natural language processing tasks, including translation, classification, and question answering, demonstrate that our pipeline consistently improves performance over base models. With a total generation and training cost of under $50 per language, Kakugo offers an accessible method for communities to develop language-specific AI. 

---
# Understanding Multilingualism in Mixture-of-Experts LLMs: Routing Mechanism, Expert Specialization, and Layerwise Steering 

**Authors**: Yuxin Chen, Zhengzhou Cai, Xiangtian Ji, Weixiang Zhao, An Zhang, Xiang Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2601.14050)  

**Abstract**: Mixture-of-Experts (MoE) architectures have shown strong multilingual capabilities, yet the internal mechanisms underlying performance gains and cross-language differences remain insufficiently understood. In this work, we conduct a systematic analysis of MoE models, examining routing behavior and expert specialization across languages and network depth. Our analysis reveals that multilingual processing in MoE models is highly structured: routing aligns with linguistic families, expert utilization follows a clear layerwise pattern, and high-resource languages rely on shared experts while low-resource languages depend more on language-exclusive experts despite weaker performance. Layerwise interventions further show that early and late MoE layers support language-specific processing, whereas middle layers serve as language-agnostic capacity hubs. Building on these insights, we propose a routing-guided steering method that adaptively guides routing behavior in middle layers toward shared experts associated with dominant languages at inference time, leading to consistent multilingual performance improvements, particularly for linguistically related language pairs. Our code is available at this https URL. 

---
# PRiSM: Benchmarking Phone Realization in Speech Models 

**Authors**: Shikhar Bharadwaj, Chin-Jou Li, Yoonjae Kim, Kwanghee Choi, Eunjung Yeo, Ryan Soh-Eun Shim, Hanyu Zhou, Brendon Boldt, Karen Rosero Jacome, Kalvin Chang, Darsh Agrawal, Keer Xu, Chao-Han Huck Yang, Jian Zhu, Shinji Watanabe, David R. Mortensen  

**Link**: [PDF](https://arxiv.org/pdf/2601.14046)  

**Abstract**: Phone recognition (PR) serves as the atomic interface for language-agnostic modeling for cross-lingual speech processing and phonetic analysis. Despite prolonged efforts in developing PR systems, current evaluations only measure surface-level transcription accuracy. We introduce PRiSM, the first open-source benchmark designed to expose blind spots in phonetic perception through intrinsic and extrinsic evaluation of PR systems. PRiSM standardizes transcription-based evaluation and assesses downstream utility in clinical, educational, and multilingual settings with transcription and representation probes. We find that diverse language exposure during training is key to PR performance, encoder-CTC models are the most stable, and specialized PR models still outperform Large Audio Language Models. PRiSM releases code, recipes, and datasets to move the field toward multilingual speech models with robust phonetic ability: this https URL. 

---
# Top 10 Open Challenges Steering the Future of Diffusion Language Model and Its Variants 

**Authors**: Yunhe Wang, Kai Han, Huiling Zhen, Yuchuan Tian, Hanting Chen, Yongbing Huang, Yufei Cui, Yingte Shu, Shan Gao, Ismail Elezi, Roy Vaughan Miles, Songcen Xu, Feng Wen, Chao Xu, Sinan Zeng, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2601.14041)  

**Abstract**: The paradigm of Large Language Models (LLMs) is currently defined by auto-regressive (AR) architectures, which generate text through a sequential ``brick-by-brick'' process. Despite their success, AR models are inherently constrained by a causal bottleneck that limits global structural foresight and iterative refinement. Diffusion Language Models (DLMs) offer a transformative alternative, conceptualizing text generation as a holistic, bidirectional denoising process akin to a sculptor refining a masterpiece. However, the potential of DLMs remains largely untapped as they are frequently confined within AR-legacy infrastructures and optimization frameworks. In this Perspective, we identify ten fundamental challenges ranging from architectural inertia and gradient sparsity to the limitations of linear reasoning that prevent DLMs from reaching their ``GPT-4 moment''. We propose a strategic roadmap organized into four pillars: foundational infrastructure, algorithmic optimization, cognitive reasoning, and unified multimodal intelligence. By shifting toward a diffusion-native ecosystem characterized by multi-scale tokenization, active remasking, and latent thinking, we can move beyond the constraints of the causal horizon. We argue that this transition is essential for developing next-generation AI capable of complex structural reasoning, dynamic self-correction, and seamless multimodal integration. 

---
# RM-Distiller: Exploiting Generative LLM for Reward Model Distillation 

**Authors**: Hongli Zhou, Hui Huang, Wei Liu, Chenglong Wang, Xingyuan Bu, Lvyuan Han, Fuhai Song, Muyun Yang, Wenhao Jiang, Hailong Cao, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.14032)  

**Abstract**: Reward models (RMs) play a pivotal role in aligning large language models (LLMs) with human preferences. Due to the difficulty of obtaining high-quality human preference annotations, distilling preferences from generative LLMs has emerged as a standard practice. However, existing approaches predominantly treat teacher models as simple binary annotators, failing to fully exploit the rich knowledge and capabilities for RM distillation. To address this, we propose RM-Distiller, a framework designed to systematically exploit the multifaceted capabilities of teacher LLMs: (1) Refinement capability, which synthesizes highly correlated response pairs to create fine-grained and contrastive signals. (2) Scoring capability, which guides the RM in capturing precise preference strength via a margin-aware optimization objective. (3) Generation capability, which incorporates the teacher's generative distribution to regularize the RM to preserve its fundamental linguistic knowledge. Extensive experiments demonstrate that RM-Distiller significantly outperforms traditional distillation methods both on RM benchmarks and reinforcement learning-based alignment, proving that exploiting multifaceted teacher capabilities is critical for effective reward modeling. To the best of our knowledge, this is the first systematic research on RM distillation from generative LLMs. 

---
# BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models 

**Authors**: Junyu Zhang, Yipeng Kang, Jiong Guo, Jiayu Zhan, Junqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14007)  

**Abstract**: Do large language models (LLMs) genuinely understand abstract concepts, or merely manipulate them as statistical patterns? We introduce an abstraction-grounding framework that decomposes conceptual understanding into three capacities: interpretation of abstract concepts (Abstract-Abstract, A-A), grounding of abstractions in concrete events (Abstract-Concrete, A-C), and application of abstract principles to regulate concrete decisions (Concrete-Concrete, C-C). Using human values as a testbed - given their semantic richness and centrality to alignment - we employ probing (detecting value traces in internal activations) and steering (modifying representations to shift behavior). Across six open-source LLMs and ten value dimensions, probing shows that diagnostic probes trained solely on abstract value descriptions reliably detect the same values in concrete event narratives and decision reasoning, demonstrating cross-level transfer. Steering reveals an asymmetry: intervening on value representations causally shifts concrete judgments and decisions (A-C, C-C), yet leaves abstract interpretations unchanged (A-A), suggesting that encoded abstract values function as stable anchors rather than malleable activations. These findings indicate LLMs maintain structured value representations that bridge abstraction and action, providing a mechanistic and operational foundation for building value-driven autonomous AI systems with more transparent, generalizable alignment and control. 

---
# Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models 

**Authors**: Hengyuan Zhang, Zhihao Zhang, Mingyang Wang, Zunhai Su, Yiwei Wang, Qianli Wang, Shuzhou Yuan, Ercong Nie, Xufeng Duan, Qibo Xue, Zeping Yu, Chenming Shang, Xiao Liang, Jing Xiong, Hui Shen, Chaofan Tao, Zhengwu Liu, Senjie Jin, Zhiheng Xi, Dongdong Zhang, Sophia Ananiadou, Tao Gui, Ruobing Xie, Hayden Kwok-Hay So, Hinrich Schütze, Xuanjing Huang, Qi Zhang, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2601.14004)  

**Abstract**: Mechanistic Interpretability (MI) has emerged as a vital approach to demystify the opaque decision-making of Large Language Models (LLMs). However, existing reviews primarily treat MI as an observational science, summarizing analytical insights while lacking a systematic framework for actionable intervention. To bridge this gap, we present a practical survey structured around the pipeline: "Locate, Steer, and Improve." We formally categorize Localizing (diagnosis) and Steering (intervention) methods based on specific Interpretable Objects to establish a rigorous intervention protocol. Furthermore, we demonstrate how this framework enables tangible improvements in Alignment, Capability, and Efficiency, effectively operationalizing MI as an actionable methodology for model optimization. The curated paper list of this work is available at this https URL. 

---
# From Tags to Trees: Structuring Fine-Grained Knowledge for Controllable Data Selection in LLM Instruction Tuning 

**Authors**: Zihan Niu, Wenping Hu, Junmin Chen, Xiyue Wang, Tong Xu, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13995)  

**Abstract**: Effective and controllable data selection is critical for LLM instruction tuning, especially with massive open-source datasets. Existing approaches primarily rely on instance-level quality scores, or diversity metrics based on embedding clusters or semantic tags. However, constrained by the flatness of embedding spaces or the coarseness of tags, these approaches overlook fine-grained knowledge and its intrinsic hierarchical dependencies, consequently hindering precise data valuation and knowledge-aligned sampling. To address this challenge, we propose Tree-aware Aligned Global Sampling (TAGS), a unified framework that leverages a knowledge tree built from fine-grained tags, thereby enabling joint control of global quality, diversity, and target alignment. Using an LLM-based tagger, we extract atomic knowledge concepts, which are organized into a global tree through bottom-up hierarchical clustering. By grounding data instances onto this tree, a tree-aware metric then quantifies data quality and diversity, facilitating effective sampling. Our controllable sampling strategy maximizes tree-level information gain and enforces leaf-level alignment via KL-divergence for specific domains. Extensive experiments demonstrate that TAGS significantly outperforms state-of-the-art baselines. Notably, it surpasses the full-dataset model by \textbf{+5.84\%} using only \textbf{5\%} of the data, while our aligned sampling strategy further boosts average performance by \textbf{+4.24\%}. 

---
# "The Whole Is Greater Than the Sum of Its Parts": A Compatibility-Aware Multi-Teacher CoT Distillation Framework 

**Authors**: Jin Cui, Jiaqi Guo, Jiepeng Zhou, Ruixuan Yang, Jiayi Lu, Jiajun Xu, Jiangcheng Song, Boran Zhao, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2601.13992)  

**Abstract**: Chain-of-Thought (CoT) reasoning empowers Large Language Models (LLMs) with remarkable capabilities but typically requires prohibitive parameter scales. CoT distillation has emerged as a promising paradigm to transfer reasoning prowess into compact Student Models (SLMs), but existing approaches often rely on a solitary teacher, capping the student's potential since individual LLMs often exhibit distinct capability biases and may suffer from catastrophic forgetting. While leveraging diverse teachers seems appealing, effectively fusing their supervisions remains challenging: teacher-student incompatibility risks amplifying hallucinations, and passive supervision fails to ensure genuine logic internalization. To address this, we introduce COMPACT, a framework that adaptively fuses supervisions from different teachers by dynamically weighting teacher gradients based on the student's real-time compatibility evaluated by a multi-dimensional metric: (1) Graph-based Consensus to filter misleading rationales by identifying mainstream reasoning paths; (2) Mutual-Information-based Adaptability to detect "epiphany moments" for genuinely understanding the reasoning process rather than merely imitating; and (3) Loss-based Difficulty to assess student receptivity to the teacher's guidance and prevent negative transfer. Extensive experiments and latent space analysis demonstrate that COMPACT effectively integrates diverse reasoning capabilities without damaging the model's original knowledge structure, achieving state-of-the-art performance on various benchmarks while mitigating catastrophic forgetting. 

---
# Automatic Prompt Optimization for Dataset-Level Feature Discovery 

**Authors**: Adrian Cosma, Oleg Szehr, David Kletz, Alessandro Antonucci, Olivier Pelletier  

**Link**: [PDF](https://arxiv.org/pdf/2601.13922)  

**Abstract**: Feature extraction from unstructured text is a critical step in many downstream classification pipelines, yet current approaches largely rely on hand-crafted prompts or fixed feature schemas. We formulate feature discovery as a dataset-level prompt optimization problem: given a labelled text corpus, the goal is to induce a global set of interpretable and discriminative feature definitions whose realizations optimize a downstream supervised learning objective. To this end, we propose a multi-agent prompt optimization framework in which language-model agents jointly propose feature definitions, extract feature values, and evaluate feature quality using dataset-level performance and interpretability feedback. Instruction prompts are iteratively refined based on this structured feedback, enabling optimization over prompts that induce shared feature sets rather than per-example predictions. This formulation departs from prior prompt optimization methods that rely on per-sample supervision and provides a principled mechanism for automatic feature discovery from unstructured text. 

---
# HyperWalker: Dynamic Hypergraph-Based Deep Diagnosis for Multi-Hop Clinical Modeling across EHR and X-Ray in Medical VLMs 

**Authors**: Yuezhe Yang, Hao Wang, Yige Peng, Jinman Kim, Lei Bi  

**Link**: [PDF](https://arxiv.org/pdf/2601.13919)  

**Abstract**: Automated clinical diagnosis remains a core challenge in medical AI, which usually requires models to integrate multi-modal data and reason across complex, case-specific contexts. Although recent methods have advanced medical report generation (MRG) and visual question answering (VQA) with medical vision-language models (VLMs), these methods, however, predominantly operate under a sample-isolated inference paradigm, as such processing cases independently without access to longitudinal electronic health records (EHRs) or structurally related patient examples. This paradigm limits reasoning to image-derived information alone, which ignores external complementary medical evidence for potentially more accurate diagnosis. To overcome this limitation, we propose \textbf{HyperWalker}, a \textit{Deep Diagnosis} framework that reformulates clinical reasoning via dynamic hypergraphs and test-time training. First, we construct a dynamic hypergraph, termed \textbf{iBrochure}, to model the structural heterogeneity of EHR data and implicit high-order associations among multimodal clinical information. Within this hypergraph, a reinforcement learning agent, \textbf{Walker}, navigates to and identifies optimal diagnostic paths. To ensure comprehensive coverage of diverse clinical characteristics in test samples, we incorporate a \textit{linger mechanism}, a multi-hop orthogonal retrieval strategy that iteratively selects clinically complementary neighborhood cases reflecting distinct clinical attributes. Experiments on MRG with MIMIC and medical VQA on EHRXQA demonstrate that HyperWalker achieves state-of-the-art performance. Code is available at: this https URL 

---
# AgentEHR: Advancing Autonomous Clinical Decision-Making via Retrospective Summarization 

**Authors**: Yusheng Liao, Chuan Xuan, Yutong Cai, Lina Yang, Zhe Chen, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13918)  

**Abstract**: Large Language Models have demonstrated profound utility in the medical domain. However, their application to autonomous Electronic Health Records~(EHRs) navigation remains constrained by a reliance on curated inputs and simplified retrieval tasks. To bridge the gap between idealized experimental settings and realistic clinical environments, we present AgentEHR. This benchmark challenges agents to execute complex decision-making tasks, such as diagnosis and treatment planning, requiring long-range interactive reasoning directly within raw and high-noise databases. In tackling these tasks, we identify that existing summarization methods inevitably suffer from critical information loss and fractured reasoning continuity. To address this, we propose RetroSum, a novel framework that unifies a retrospective summarization mechanism with an evolving experience strategy. By dynamically re-evaluating interaction history, the retrospective mechanism prevents long-context information loss and ensures unbroken logical coherence. Additionally, the evolving strategy bridges the domain gap by retrieving accumulated experience from a memory bank. Extensive empirical evaluations demonstrate that RetroSum achieves performance gains of up to 29.16% over competitive baselines, while significantly decreasing total interaction errors by up to 92.3%. 

---
# Confident Rankings with Fewer Items: Adaptive LLM Evaluation with Continuous Scores 

**Authors**: Esma Balkır, Alice Pernthaller, Marco Basaldella, José Hernández-Orallo, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2601.13885)  

**Abstract**: Computerized Adaptive Testing (CAT) has proven effective for efficient LLM evaluation on multiple-choice benchmarks, but modern LLM evaluation increasingly relies on generation tasks where outputs are scored continuously rather than marked correct/incorrect. We present a principled extension of IRT-based adaptive testing to continuous bounded scores (ROUGE, BLEU, LLM-as-a-Judge) by replacing the Bernoulli response distribution with a heteroskedastic normal distribution. Building on this, we introduce an uncertainty aware ranker with adaptive stopping criteria that achieves reliable model ranking while testing as few items and as cheaply as possible. We validate our method on five benchmarks spanning n-gram-based, embedding-based, and LLM-as-judge metrics. Our method uses 2% of the items while improving ranking correlation by 0.12 {\tau} over random sampling, with 95% accuracy on confident predictions. 

---
# OpenLearnLM Benchmark: A Unified Framework for Evaluating Knowledge, Skill, and Attitude in Educational Large Language Models 

**Authors**: Unggi Lee, Sookbun Lee, Heungsoo Choi, Jinseo Lee, Haeun Park, Younghoon Jeon, Sungmin Cho, Minju Kang, Junbo Koh, Jiyeong Bae, Minwoo Nam, Juyeon Eun, Yeonji Jung, Yeil Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2601.13882)  

**Abstract**: Large Language Models are increasingly deployed as educational tools, yet existing benchmarks focus on narrow skills and lack grounding in learning sciences. We introduce OpenLearnLM Benchmark, a theory-grounded framework evaluating LLMs across three dimensions derived from educational assessment theory: Knowledge (curriculum-aligned content and pedagogical understanding), Skills (scenario-based competencies organized through a four-level center-role-scenario-subscenario hierarchy), and Attitude (alignment consistency and deception resistance). Our benchmark comprises 124K+ items spanning multiple subjects, educational roles, and difficulty levels based on Bloom's taxonomy. The Knowledge domain prioritizes authentic assessment items from established benchmarks, while the Attitude domain adapts Anthropic's Alignment Faking methodology to detect behavioral inconsistency under varying monitoring conditions. Evaluation of seven frontier models reveals distinct capability profiles: Claude-Opus-4.5 excels in practical skills despite lower content knowledge, while Grok-4.1-fast leads in knowledge but shows alignment concerns. Notably, no single model dominates all dimensions, validating the necessity of multi-axis evaluation. OpenLearnLM provides an open, comprehensive framework for advancing LLM readiness in authentic educational contexts. 

---
# Pedagogical Alignment for Vision-Language-Action Models: A Comprehensive Framework for Data, Architecture, and Evaluation in Education 

**Authors**: Unggi Lee, Jahyun Jeong, Sunyoung Shin, Haeun Park, Jeongsu Moon, Youngchang Song, Jaechang Shim, JaeHwan Lee, Yunju Noh, Seungwon Choi, Ahhyun Kim, TaeHyeon Kim, Kyungtae Joo, Taeyeong Kim, Gyeonggeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.13876)  

**Abstract**: Science demonstrations are important for effective STEM education, yet teachers face challenges in conducting them safely and consistently across multiple occasions, where robotics can be helpful. However, current Vision-Language-Action (VLA) models require substantial computational resources and sacrifice language generation capabilities to maximize efficiency, making them unsuitable for resource-constrained educational settings that require interpretable, explanation-generating systems. We present \textit{Pedagogical VLA Framework}, a framework that applies pedagogical alignment to lightweight VLA models through four components: text healing to restore language generation capabilities, large language model (LLM) distillation to transfer pedagogical knowledge, safety training for educational environments, and pedagogical evaluation adjusted to science education contexts. We evaluate Pedagogical VLA Framework across five science demonstrations spanning physics, chemistry, biology, and earth science, using an evaluation framework developed in collaboration with science education experts. Our evaluation assesses both task performance (success rate, protocol compliance, efficiency, safety) and pedagogical quality through teacher surveys and LLM-as-Judge assessment. We additionally provide qualitative analysis of generated texts. Experimental results demonstrate that Pedagogical VLA Framework achieves comparable task performance to baseline models while producing contextually appropriate educational explanations. 

---
# FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs 

**Authors**: Qian Chen, Jinlan Fu, Changsong Li, See-Kiong Ng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.13836)  

**Abstract**: Although Multimodal Large Language Models (MLLMs) demonstrate strong omni-modal perception, their ability to forecast future events from audio-visual cues remains largely unexplored, as existing benchmarks focus mainly on retrospective understanding. To bridge this gap, we introduce FutureOmni, the first benchmark designed to evaluate omni-modal future forecasting from audio-visual environments. The evaluated models are required to perform cross-modal causal and temporal reasoning, as well as effectively leverage internal knowledge to predict future events. FutureOmni is constructed via a scalable LLM-assisted, human-in-the-loop pipeline and contains 919 videos and 1,034 multiple-choice QA pairs across 8 primary domains. Evaluations on 13 omni-modal and 7 video-only models show that current systems struggle with audio-visual future prediction, particularly in speech-heavy scenarios, with the best accuracy of 64.8% achieved by Gemini 3 Flash. To mitigate this limitation, we curate a 7K-sample instruction-tuning dataset and propose an Omni-Modal Future Forecasting (OFF) training strategy. Evaluations on FutureOmni and popular audio-visual and video-only benchmarks demonstrate that OFF enhances future forecasting and generalization. We publicly release all code (this https URL) and datasets (this https URL). 

---
# The Role of Prosodic and Lexical Cues in Turn-Taking with Self-Supervised Speech Representations 

**Authors**: Sam OConnor Russell, Delphine Charuau, Naomi Harte  

**Link**: [PDF](https://arxiv.org/pdf/2601.13835)  

**Abstract**: Fluid turn-taking remains a key challenge in human-robot interaction. Self-supervised speech representations (S3Rs) have driven many advances, but it remains unclear whether S3R-based turn-taking models rely on prosodic cues, lexical cues or both. We introduce a vocoder-based approach to control prosody and lexical cues in speech more cleanly than prior work. This allows us to probe the voice-activity projection model, an S3R-based turn-taking model. We find that prediction on prosody-matched, unintelligible noise is similar to accuracy on clean speech. This reveals both prosodic and lexical cues support turn-taking, but either can be used in isolation. Hence, future models may only require prosody, providing privacy and potential performance benefits. When either prosodic or lexical information is disrupted, the model exploits the other without further training, indicating they are encoded in S3Rs with limited interdependence. Results are consistent in CPC-based and wav2vec2.0 S3Rs. We discuss our findings and highlight a number of directions for future work. All code is available to support future research. 

---
# Knowledge Graph-Assisted LLM Post-Training for Enhanced Legal Reasoning 

**Authors**: Dezhao Song, Guglielmo Bonifazi, Frank Schilder, Jonathan Richard Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2601.13806)  

**Abstract**: LLM post-training has primarily relied on large text corpora and human feedback, without capturing the structure of domain knowledge. This has caused models to struggle dealing with complex reasoning tasks, especially for high-stakes professional domains. In Law, reasoning requires deep understanding of the relations between various legal concepts, a key component missing in current LLM post-training. In this paper, we propose a knowledge graph (KG)-assisted approach for enhancing LLMs' reasoning capability in Legal that is generalizable to other high-stakes domains. We model key legal concepts by following the \textbf{IRAC} (Issue, Rule, Analysis and Conclusion) framework, and construct a KG with 12K legal cases. We then produce training data using our IRAC KG, and conduct both Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) with three state-of-the-art (SOTA) LLMs (30B, 49B and 70B), varying architecture and base model family. Our post-trained models obtained better average performance on 4/5 diverse legal benchmarks (14 tasks) than baselines. In particular, our 70B DPO model achieved the best score on 4/6 reasoning tasks, among baselines and a 141B SOTA legal LLM, demonstrating the effectiveness of our KG for enhancing LLMs' legal reasoning capability. 

---
# Habibi: Laying the Open-Source Foundation of Unified-Dialectal Arabic Speech Synthesis 

**Authors**: Yushen Chen, Junzhe Liu, Yujie Tu, Zhikang Niu, Yuzhe Liang, Kai Yu, Chunyu Qiang, Chen Zhang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.13802)  

**Abstract**: A notable gap persists in speech synthesis research and development for Arabic dialects, particularly from a unified modeling perspective. Despite its high practical value, the inherent linguistic complexity of Arabic dialects, further compounded by a lack of standardized data, benchmarks, and evaluation guidelines, steers researchers toward safer ground. To bridge this divide, we present Habibi, a suite of specialized and unified text-to-speech models that harnesses existing open-source ASR corpora to support a wide range of high- to low-resource Arabic dialects through linguistically-informed curriculum learning. Our approach outperforms the leading commercial service in generation quality, while maintaining extensibility through effective in-context learning, without requiring text diacritization. We are committed to open-sourcing the model, along with creating the first systematic benchmark for multi-dialect Arabic speech synthesis. Furthermore, by identifying the key challenges in and establishing evaluation standards for the process, we aim to provide a solid groundwork for subsequent research. Resources at this https URL . 

---
# Pro-AI Bias in Large Language Models 

**Authors**: Benaya Trabelsi, Jonathan Shaki, Sarit Kraus  

**Link**: [PDF](https://arxiv.org/pdf/2601.13749)  

**Abstract**: Large language models (LLMs) are increasingly employed for decision-support across multiple domains. We investigate whether these models display a systematic preferential bias in favor of artificial intelligence (AI) itself. Across three complementary experiments, we find consistent evidence of pro-AI bias. First, we show that LLMs disproportionately recommend AI-related options in response to diverse advice-seeking queries, with proprietary models doing so almost deterministically. Second, we demonstrate that models systematically overestimate salaries for AI-related jobs relative to closely matched non-AI jobs, with proprietary models overestimating AI salaries more by 10 percentage points. Finally, probing internal representations of open-weight models reveals that ``Artificial Intelligence'' exhibits the highest similarity to generic prompts for academic fields under positive, negative, and neutral framings alike, indicating valence-invariant representational centrality. These patterns suggest that LLM-generated advice and valuation can systematically skew choices and perceptions in high-stakes decisions. 

---
# Dimension-First Evaluation of Speech-to-Speech Models with Structured Acoustic Cues 

**Authors**: Arjun Chandra, Kevin Miller, Venkatesh Ravichandran, Constantinos Papayiannis, Venkatesh Saligrama  

**Link**: [PDF](https://arxiv.org/pdf/2601.13742)  

**Abstract**: Large Language Model (LLM) judges exhibit strong reasoning capabilities but are limited to textual content. This leaves current automatic Speech-to-Speech (S2S) evaluation methods reliant on opaque and expensive Audio Language Models (ALMs). In this work, we propose TRACE (Textual Reasoning over Audio Cues for Evaluation), a novel framework that enables LLM judges to reason over audio cues to achieve cost-efficient and human-aligned S2S evaluation. To demonstrate the strength of the framework, we first introduce a Human Chain-of-Thought (HCoT) annotation protocol to improve the diagnostic capability of existing judge benchmarks by separating evaluation into explicit dimensions: content (C), voice quality (VQ), and paralinguistics (P). Using this data, TRACE constructs a textual blueprint of inexpensive audio signals and prompts an LLM to render dimension-wise judgments, fusing them into an overall rating via a deterministic policy. TRACE achieves higher agreement with human raters than ALMs and transcript-only LLM judges while being significantly more cost-effective. We will release the HCoT annotations and the TRACE framework to enable scalable and human-aligned S2S evaluation. 

---
# Towards robust long-context understanding of large language model via active recap learning 

**Authors**: Chenyu Hui  

**Link**: [PDF](https://arxiv.org/pdf/2601.13734)  

**Abstract**: In this paper, we propose active recap learning (ARL), a framework for enhancing large language model (LLM) in understanding long contexts. ARL enables models to revisit and summarize earlier content through targeted sequence construction during contined pretraining and retrospective summarization at inference. First, we identify key tokens in prepared long context based on loss gaps between long and short forward contexts and find most revant preceding paragraphs, then summarize them using an LLM. Second, ARL equips models with the ability to autonomously generate and utilize these retrospective summaries during inference, thereby establishing a recursive memory mechanism across paragraphs. Experimental results show substantial gains, with ARL achieving a 26.8% improvement on RULER and a 9.44% improvement on LongBench. Overall, ARL offers a simple yet effective continued pretraining-based approach to strengthen long-context understanding, advancing scalable memory augmentation in LLM 

---
# On Temperature-Constrained Non-Deterministic Machine Translation: Potential and Evaluation 

**Authors**: Weichuan Wang, Mingyang Liu, Linqi Song, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2601.13729)  

**Abstract**: In recent years, the non-deterministic properties of language models have garnered considerable attention and have shown a significant influence on real-world applications. However, such properties remain under-explored in machine translation (MT), a complex, non-deterministic NLP task. In this study, we systematically evaluate modern MT systems and identify temperature-constrained Non-Deterministic MT (ND-MT) as a distinct phenomenon. Additionally, we demonstrate that ND-MT exhibits significant potential in addressing the multi-modality issue that has long challenged MT research and provides higher-quality candidates than Deterministic MT (D-MT) under temperature constraints. However, ND-MT introduces new challenges in evaluating system performance. Specifically, the evaluation framework designed for D-MT fails to yield consistent evaluation results when applied to ND-MT. We further investigate this emerging challenge by evaluating five state-of-the-art ND-MT systems across three open datasets using both lexical-based and semantic-based metrics at varying sampling sizes. The results reveal a Buckets effect across these systems: the lowest-quality candidate generated by ND-MT consistently determines the overall system ranking across different sampling sizes for all reasonable metrics. Furthermore, we propose the ExpectoSample strategy to automatically assess the reliability of evaluation metrics for selecting robust ND-MT. 

---
# OP-Bench: Benchmarking Over-Personalization for Memory-Augmented Personalized Conversational Agents 

**Authors**: Yulin Hu, Zimo Long, Jiahe Guo, Xingyu Sui, Xing Fu, Weixiang Zhao, Yanyan Zhao, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.13722)  

**Abstract**: Memory-augmented conversational agents enable personalized interactions using long-term user memory and have gained substantial traction. However, existing benchmarks primarily focus on whether agents can recall and apply user information, while overlooking whether such personalization is used appropriately. In fact, agents may overuse personal information, producing responses that feel forced, intrusive, or socially inappropriate to users. We refer to this issue as \emph{over-personalization}. In this work, we formalize over-personalization into three types: Irrelevance, Repetition, and Sycophancy, and introduce \textbf{OP-Bench} a benchmark of 1,700 verified instances constructed from long-horizon dialogue histories. Using \textbf{OP-Bench}, we evaluate multiple large language models and memory-augmentation methods, and find that over-personalization is widespread when memory is introduced. Further analysis reveals that agents tend to retrieve and over-attend to user memories even when unnecessary. To address this issue, we propose \textbf{Self-ReCheck}, a lightweight, model-agnostic memory filtering mechanism that mitigates over-personalization while preserving personalization performance. Our work takes an initial step toward more controllable and appropriate personalization in memory-augmented dialogue systems. 

---
# Simulated Ignorance Fails: A Systematic Study of LLM Behaviors on Forecasting Problems Before Model Knowledge Cutoff 

**Authors**: Zehan Li, Yuxuan Wang, Ali El Lahib, Ying-Jieh Xia, Xinyu Pi  

**Link**: [PDF](https://arxiv.org/pdf/2601.13717)  

**Abstract**: Evaluating LLM forecasting capabilities is constrained by a fundamental tension: prospective evaluation offers methodological rigor but prohibitive latency, while retrospective forecasting (RF) -- evaluating on already-resolved events -- faces rapidly shrinking clean evaluation data as SOTA models possess increasingly recent knowledge cutoffs. Simulated Ignorance (SI), prompting models to suppress pre-cutoff knowledge, has emerged as a potential solution. We provide the first systematic test of whether SI can approximate True Ignorance (TI). Across 477 competition-level questions and 9 models, we find that SI fails systematically: (1) cutoff instructions leave a 52% performance gap between SI and TI; (2) chain-of-thought reasoning fails to suppress prior knowledge, even when reasoning traces contain no explicit post-cutoff references; (3) reasoning-optimized models exhibit worse SI fidelity despite superior reasoning trace quality. These findings demonstrate that prompts cannot reliably "rewind" model knowledge. We conclude that RF on pre-cutoff events is methodologically flawed; we recommend against using SI-based retrospective setups to benchmark forecasting capabilities. 

---
# GerAV: Towards New Heights in German Authorship Verification using Fine-Tuned LLMs on a New Benchmark 

**Authors**: Lotta Kiefer, Christoph Leiter, Sotaro Takeshita, Elena Schmidt, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2601.13711)  

**Abstract**: Authorship verification (AV) is the task of determining whether two texts were written by the same author and has been studied extensively, predominantly for English data. In contrast, large-scale benchmarks and systematic evaluations for other languages remain scarce. We address this gap by introducing GerAV, a comprehensive benchmark for German AV comprising over 600k labeled text pairs. GerAV is built from Twitter and Reddit data, with the Reddit part further divided into in-domain and cross-domain message-based subsets, as well as a profile-based subset. This design enables controlled analysis of the effects of data source, topical domain, and text length. Using the provided training splits, we conduct a systematic evaluation of strong baselines and state-of-the-art models and find that our best approach, a fine-tuned large language model, outperforms recent baselines by up to 0.09 absolute F1 score and surpasses GPT-5 in a zero-shot setting by 0.08. We further observe a trade-off between specialization and generalization: models trained on specific data types perform best under matching conditions but generalize less well across data regimes, a limitation that can be mitigated by combining training sources. Overall, GerAV provides a challenging and versatile benchmark for advancing research on German and cross-domain AV. 

---
# Uncertainty-Aware Gradient Signal-to-Noise Data Selection for Instruction Tuning 

**Authors**: Zhihang Yuan, Chengyu Yue, Long Huang, Litu Ou, Lei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2601.13697)  

**Abstract**: Instruction tuning is a standard paradigm for adapting large language models (LLMs), but modern instruction datasets are large, noisy, and redundant, making full-data fine-tuning costly and often unnecessary. Existing data selection methods either build expensive gradient datastores or assign static scores from a weak proxy, largely ignoring evolving uncertainty, and thus missing a key source of LLM interpretability. We propose GRADFILTERING, an objective-agnostic, uncertainty-aware data selection framework that utilizes a small GPT-2 proxy with a LoRA ensemble and aggregates per-example gradients into a Gradient Signal-to-Noise Ratio (G-SNR) utility. Our method matches or surpasses random subsets and strong baselines in most LLM-as-a-judge evaluations as well as in human assessment. Moreover, GRADFILTERING-selected subsets converge faster than competitive filters under the same compute budget, reflecting the benefit of uncertainty-aware scoring. 

---
# OptiSQL: Executable SQL Generation from Optical TokensOptiSQL: Executable SQL Generation from Optical Tokens 

**Authors**: Sifan Li, Hongkai Chen, Yujun Cai, Liyang Chen, Qingwen Ye, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13695)  

**Abstract**: Executable SQL generation is typically studied in text-to-SQL settings, where tables are provided as fully linearized textual schemas and contents. While effective, this formulation assumes access to structured text and incurs substantial token overhead, which is misaligned with many real-world scenarios where tables appear as visual artifacts in documents or webpages. We investigate whether compact optical representations can serve as an efficient interface for executable semantic parsing. We present OptiSQL, a vision-driven framework that generates executable SQL directly from table images and natural language questions using compact optical tokens. OptiSQL leverages an OCR-oriented visual encoder to compress table structure and content into a small set of optical tokens and fine-tunes a pretrained decoder for SQL generation while freezing the encoder to isolate representation sufficiency. Experiments on a visualized version of Spider 2.0-Snow show that OptiSQL retains strong execution accuracy while reducing table input tokens by an order of magnitude. Robustness analyses further demonstrate that optical tokens preserve essential structural information under visual perturbations. 

---
# Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning 

**Authors**: Yue Guo, Fanfu Wang, Jianwei Lv, Xincheng Shi, Yuchen Li, Youya Wang, Yunsheng Zeng, Yujing Liu, Yunhao Qiao, Gen Li, Junfeng Wang, Bo Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2601.13690)  

**Abstract**: Clinical Decision Support Systems (CDSSs) provide reasoning and inquiry guidance for physicians, yet they face notable challenges, including high maintenance costs and low generalization capability. Recently, Large Language Models (LLMs) have been widely adopted in healthcare due to their extensive knowledge reserves, retrieval, and communication capabilities. While LLMs show promise and excel at medical benchmarks, their diagnostic reasoning and inquiry skills are constrained. To mitigate this issue, we propose (1) Clinical Diagnostic Reasoning Data (CDRD) structure to capture abstract clinical reasoning logic, and a pipeline for its construction, and (2) the Dr. Assistant, a clinical diagnostic model equipped with clinical reasoning and inquiry skills. Its training involves a two-stage process: SFT, followed by RL with a tailored reward function. We also introduce a benchmark to evaluate both diagnostic reasoning and inquiry. Our experiments demonstrate that the Dr. Assistant outperforms open-source models and achieves competitive performance to closed-source models, providing an effective solution for clinical diagnostic inquiry guidance. 

---
# HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference 

**Authors**: Zhiyuan Shi, Qibo Qiu, Feng Xue, Zhonglin Jiang, Li Yu, Jian Jiang, Xiaofei He, Wenxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13684)  

**Abstract**: The linear memory growth of the KV cache poses a significant bottleneck for LLM inference in long-context tasks. Existing static compression methods often fail to preserve globally important information, principally because they overlook the attention drift phenomenon where token significance evolves dynamically. Although recent dynamic retrieval approaches attempt to address this issue, they typically suffer from coarse-grained caching strategies and incur high I/O overhead due to frequent data transfers. To overcome these limitations, we propose HeteroCache, a training-free dynamic compression framework. Our method is built on two key insights: attention heads exhibit diverse temporal heterogeneity, and there is significant spatial redundancy among heads within the same layer. Guided by these insights, HeteroCache categorizes heads based on stability and redundancy. Consequently, we apply a fine-grained weighting strategy that allocates larger cache budgets to heads with rapidly shifting attention to capture context changes, thereby addressing the inefficiency of coarse-grained strategies. Furthermore, we employ a hierarchical storage mechanism in which a subset of representative heads monitors attention shift, and trigger an asynchronous, on-demand retrieval of contexts from the CPU, effectively hiding I/O latency. Finally, experiments demonstrate that HeteroCache achieves state-of-the-art performance on multiple long-context benchmarks and accelerates decoding by up to $3\times$ compared to the original model in the 224K context. Our code will be open-source. 

---
# CommunityBench: Benchmarking Community-Level Alignment across Diverse Groups and Tasks 

**Authors**: Jiayu Lin, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.13669)  

**Abstract**: Large language models (LLMs) alignment ensures model behaviors reflect human value. Existing alignment strategies primarily follow two paths: one assumes a universal value set for a unified goal (i.e., one-size-fits-all), while the other treats every individual as unique to customize models (i.e., individual-level). However, assuming a monolithic value space marginalizes minority norms, while tailoring individual models is prohibitively expensive. Recognizing that human society is organized into social clusters with high intra-group value alignment, we propose community-level alignment as a "middle ground". Practically, we introduce CommunityBench, the first large-scale benchmark for community-level alignment evaluation, featuring four tasks grounded in Common Identity and Common Bond theory. With CommunityBench, we conduct a comprehensive evaluation of various foundation models on CommunityBench, revealing that current LLMs exhibit limited capacity to model community-specific preferences. Furthermore, we investigate the potential of community-level alignment in facilitating individual modeling, providing a promising direction for scalable and pluralistic alignment. 

---
# Temporal-Spatial Decouple before Act: Disentangled Representation Learning for Multimodal Sentiment Analysis 

**Authors**: Chunlei Meng, Ziyang Zhou, Lucas He, Xiaojing Du, Chun Ouyang, Zhongxue Gan  

**Link**: [PDF](https://arxiv.org/pdf/2601.13659)  

**Abstract**: Multimodal Sentiment Analysis integrates Linguistic, Visual, and Acoustic. Mainstream approaches based on modality-invariant and modality-specific factorization or on complex fusion still rely on spatiotemporal mixed modeling. This ignores spatiotemporal heterogeneity, leading to spatiotemporal information asymmetry and thus limited performance. Hence, we propose TSDA, Temporal-Spatial Decouple before Act, which explicitly decouples each modality into temporal dynamics and spatial structural context before any interaction. For every modality, a temporal encoder and a spatial encoder project signals into separate temporal and spatial body. Factor-Consistent Cross-Modal Alignment then aligns temporal features only with their temporal counterparts across modalities, and spatial features only with their spatial counterparts. Factor specific supervision and decorrelation regularization reduce cross factor leakage while preserving complementarity. A Gated Recouple module subsequently recouples the aligned streams for task. Extensive experiments show that TSDA outperforms baselines. Ablation analysis studies confirm the necessity and interpretability of the design. 

---
# Beyond Known Facts: Generating Unseen Temporal Knowledge to Address Data Contamination in LLM Evaluation 

**Authors**: Arthur Amalvy, Hen-Hsen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13658)  

**Abstract**: The automatic extraction of information is important for populating large web knowledge bases such as Wikidata. The temporal version of that task, temporal knowledge graph extraction (TKGE), involves extracting temporally grounded facts from text, represented as semantic quadruples (subject, relation, object, timestamp). Many recent systems take advantage of large language models (LLMs), which are becoming a new cornerstone of the web due to their performance on many tasks across the natural language processing (NLP) field. Despite the importance of TKGE, existing datasets for training and evaluation remain scarce, and contamination of evaluation data is an unaddressed issue, potentially inflating LLMs' perceived performance due to overlaps between training and evaluation sets. To mitigate these challenges, we propose a novel synthetic evaluation dataset constructed from predicted future, previously unseen temporal facts, thereby eliminating contamination and enabling robust and unbiased benchmarking. Our dataset creation involves a two-step approach: (1) Temporal Knowledge Graph Forecasting (TKGF) generates plausible future quadruples, which are subsequently filtered to adhere to the original knowledge base schema; (2) LLMs perform quadruple-to-text generation, creating semantically aligned textual descriptions. We benchmark Extract, Define and Canonicalize (EDC), a state-of-the-art LLM-based extraction framework, demonstrating that LLM performance decreases when evaluated on our dataset compared to a dataset of known facts. We publicly release our dataset consisting of 4.2K future quadruples and corresponding textual descriptions, along with the generation methodology, enabling continuous creation of unlimited future temporal datasets to serve as long-term, contamination-free benchmarks for TKGE. 

---
# Fairness or Fluency? An Investigation into Language Bias of Pairwise LLM-as-a-Judge 

**Authors**: Xiaolin Zhou, Zheng Luo, Yicheng Gao, Qixuan Chen, Xiyang Hu, Yue Zhao, Ruishan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.13649)  

**Abstract**: Recent advances in Large Language Models (LLMs) have incentivized the development of LLM-as-a-judge, an application of LLMs where they are used as judges to decide the quality of a certain piece of text given a certain context. However, previous studies have demonstrated that LLM-as-a-judge can be biased towards different aspects of the judged texts, which often do not align with human preference. One of the identified biases is language bias, which indicates that the decision of LLM-as-a-judge can differ based on the language of the judged texts. In this paper, we study two types of language bias in pairwise LLM-as-a-judge: (1) performance disparity between languages when the judge is prompted to compare options from the same language, and (2) bias towards options written in major languages when the judge is prompted to compare options of two different languages. We find that for same-language judging, there exist significant performance disparities across language families, with European languages consistently outperforming African languages, and this bias is more pronounced in culturally-related subjects. For inter-language judging, we observe that most models favor English answers, and that this preference is influenced more by answer language than question language. Finally, we investigate whether language bias is in fact caused by low-perplexity bias, a previously identified bias of LLM-as-a-judge, and we find that while perplexity is slightly correlated with language bias, language bias cannot be fully explained by perplexity only. 

---
# Towards Token-Level Text Anomaly Detection 

**Authors**: Yang Cao, Bicheng Yu, Sikun Yang, Ming Liu, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13644)  

**Abstract**: Despite significant progress in text anomaly detection for web applications such as spam filtering and fake news detection, existing methods are fundamentally limited to document-level analysis, unable to identify which specific parts of a text are anomalous. We introduce token-level anomaly detection, a novel paradigm that enables fine-grained localization of anomalies within text. We formally define text anomalies at both document and token-levels, and propose a unified detection framework that operates across multiple levels. To facilitate research in this direction, we collect and annotate three benchmark datasets spanning spam, reviews and grammar errors with token-level labels. Experimental results demonstrate that our framework get better performance than other 6 baselines, opening new possibilities for precise anomaly localization in text. All the codes and data are publicly available on this https URL. 

---
# Activation-Space Anchored Access Control for Multi-Class Permission Reasoning in Large Language Models 

**Authors**: Zhaopeng Zhang, Pengcheng Sun, Lan Zhang, Chen Tang, Jiewei Lai, Yunhao Wang, Hui Jin  

**Link**: [PDF](https://arxiv.org/pdf/2601.13630)  

**Abstract**: Large language models (LLMs) are increasingly deployed over knowledge bases for efficient knowledge retrieval and question answering. However, LLMs can inadvertently answer beyond a user's permission scope, leaking sensitive content, thus making it difficult to deploy knowledge-base QA under fine-grained access control requirements. In this work, we identify a geometric regularity in intermediate activations: for the same query, representations induced by different permission scopes cluster distinctly and are readily separable. Building on this separability, we propose Activation-space Anchored Access Control (AAAC), a training-free framework for multi-class permission control. AAAC constructs an anchor bank, with one permission anchor per class, from a small offline sample set and requires no fine-tuning. At inference time, a multi-anchor steering mechanism redirects each query's activations toward the anchor-defined authorized region associated with the current user, thereby suppressing over-privileged generations by design. Finally, extensive experiments across three LLM families demonstrate that AAAC reduces permission violation rates by up to 86.5% and prompt-based attack success rates by 90.7%, while improving response usability with minor inference overhead compared to baselines. 

---
# CauScientist: Teaching LLMs to Respect Data for Causal Discovery 

**Authors**: Bo Peng, Sirui Chen, Lei Xu, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.13614)  

**Abstract**: Causal discovery is fundamental to scientific understanding and reliable decision-making. Existing approaches face critical limitations: purely data-driven methods suffer from statistical indistinguishability and modeling assumptions, while recent LLM-based methods either ignore statistical evidence or incorporate unverified priors that can mislead result. To this end, we propose CauScientist, a collaborative framework that synergizes LLMs as hypothesis-generating "data scientists" with probabilistic statistics as rigorous "verifiers". CauScientist employs hybrid initialization to select superior starting graphs, iteratively refines structures through LLM-proposed modifications validated by statistical criteria, and maintains error memory to guide efficient search space. Experiments demonstrate that CauScientist substantially outperforms purely data-driven baselines, achieving up to 53.8% F1 score improvement and enhancing recall from 35.0% to 100.0%. Notably, while standalone LLM performance degrades with graph complexity, CauScientist reduces structural hamming distance (SHD) by 44.0% compared to Qwen3-32B on 37-node graphs. Our project page is at this https URL. 

---
# Vulnerability of LLMs' Belief Systems? LLMs Belief Resistance Check Through Strategic Persuasive Conversation Interventions 

**Authors**: Fan Huang, Haewoon Kwak, Jisun An  

**Link**: [PDF](https://arxiv.org/pdf/2601.13590)  

**Abstract**: Large Language Models (LLMs) are increasingly employed in various question-answering tasks. However, recent studies showcase that LLMs are susceptible to persuasion and could adopt counterfactual beliefs. We present a systematic evaluation of LLM susceptibility to persuasion under the Source--Message--Channel--Receiver (SMCR) communication framework. Across five mainstream Large Language Models (LLMs) and three domains (factual knowledge, medical QA, and social bias), we analyze how different persuasive strategies influence belief stability over multiple interaction turns. We further examine whether meta-cognition prompting (i.e., eliciting self-reported confidence) affects resistance to persuasion. Results show that smaller models exhibit extreme compliance, with over 80% of belief changes occurring at the first persuasive turn (average end turn of 1.1--1.4). Contrary to expectations, meta-cognition prompting increases vulnerability by accelerating belief erosion rather than enhancing robustness. Finally, we evaluate adversarial fine-tuning as a defense. While GPT-4o-mini achieves near-complete robustness (98.6%) and Mistral~7B improves substantially (35.7% $\rightarrow$ 79.3%), Llama models remain highly susceptible (<14%) even when fine-tuned on their own failure cases. Together, these findings highlight substantial model-dependent limits of current robustness interventions and offer guidance for developing more trustworthy LLMs. 

---
# TREX: Tokenizer Regression for Optimal Data Mixture 

**Authors**: Inho Won, Hangyeol Yoo, Minkyung Cho, Jungyeul Park, Hoyun Song, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2601.13588)  

**Abstract**: Building effective tokenizers for multilingual Large Language Models (LLMs) requires careful control over language-specific data mixtures. While a tokenizer's compression performance critically affects the efficiency of LLM training and inference, existing approaches rely on heuristics or costly large-scale searches to determine optimal language ratios. We introduce Tokenizer Regression for Optimal Data MiXture (TREX), a regression-based framework that efficiently predicts the optimal data mixture for tokenizer training. TREX trains small-scale proxy tokenizers on random mixtures, gathers their compression statistics, and learns to predict compression performance from data mixtures. This learned model enables scalable mixture search before large-scale tokenizer training, mitigating the accuracy-cost trade-off in multilingual tokenizer design. Tokenizers trained with TReX's predicted mixtures outperform mixtures based on LLaMA3 and uniform distributions by up to 12% in both inand out-of-distribution compression efficiency, demonstrating strong scalability, robustness, and practical effectiveness. 

---
# Comparing Without Saying: A Dataset and Benchmark for Implicit Comparative Opinion Mining from Same-User Reviews 

**Authors**: Thanh-Lam T. Nguyen, Ngoc-Quang Le, Quoc-Trung Phu, Thi-Phuong Le, Ngoc-Huyen Pham, Phuong-Nguyen Nguyen, Hoang-Quynh Le  

**Link**: [PDF](https://arxiv.org/pdf/2601.13575)  

**Abstract**: Existing studies on comparative opinion mining have mainly focused on explicit comparative expressions, which are uncommon in real-world reviews. This leaves implicit comparisons - here users express preferences across separate reviews - largely underexplored. We introduce SUDO, a novel dataset for implicit comparative opinion mining from same-user reviews, allowing reliable inference of user preferences even without explicit comparative cues. SUDO comprises 4,150 annotated review pairs (15,191 sentences) with a bi-level structure capturing aspect-level mentions and review-level preferences. We benchmark this task using two baseline architectures: traditional machine learning- and language model-based baselines. Experimental results show that while the latter outperforms the former, overall performance remains moderate, revealing the inherent difficulty of the task and establishing SUDO as a challenging and valuable benchmark for future research. 

---
# HateXScore: A Metric Suite for Evaluating Reasoning Quality in Hate Speech Explanations 

**Authors**: Yujia Hu, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.13547)  

**Abstract**: Hateful speech detection is a key component of content moderation, yet current evaluation frameworks rarely assess why a text is deemed hateful. We introduce \textsf{HateXScore}, a four-component metric suite designed to evaluate the reasoning quality of model explanations. It assesses (i) conclusion explicitness, (ii) faithfulness and causal grounding of quoted spans, (iii) protected group identification (policy-configurable), and (iv) logical consistency among these elements. Evaluated on six diverse hate speech datasets, \textsf{HateXScore} is intended as a diagnostic complement to reveal interpretability failures and annotation inconsistencies that are invisible to standard metrics like Accuracy or F1. Moreover, human evaluation shows strong agreement with \textsf{HateXScore}, validating it as a practical tool for trustworthy and transparent moderation.
\textcolor{red}{Disclaimer: This paper contains sensitive content that may be disturbing to some readers.} 

---
# When Wording Steers the Evaluation: Framing Bias in LLM judges 

**Authors**: Yerin Hwang, Dongryeol Lee, Taegwan Kang, Minwoo Lee, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2601.13537)  

**Abstract**: Large language models (LLMs) are known to produce varying responses depending on prompt phrasing, indicating that subtle guidance in phrasing can steer their answers. However, the impact of this framing bias on LLM-based evaluation, where models are expected to make stable and impartial judgments, remains largely underexplored. Drawing inspiration from the framing effect in psychology, we systematically investigate how deliberate prompt framing skews model judgments across four high-stakes evaluation tasks. We design symmetric prompts using predicate-positive and predicate-negative constructions and demonstrate that such framing induces significant discrepancies in model outputs. Across 14 LLM judges, we observe clear susceptibility to framing, with model families showing distinct tendencies toward agreement or rejection. These findings suggest that framing bias is a structural property of current LLM-based evaluation systems, underscoring the need for framing-aware protocols. 

---
# Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives 

**Authors**: Kyung Ho Lim, Byung-Hoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.13503)  

**Abstract**: Psychiatric narratives encode patient identity not only through explicit identifiers but also through idiosyncratic life events embedded in their clinical structure. Existing de-identification approaches, including PHI masking and LLM-based synthetic rewriting, operate at the text level and offer limited control over which semantic elements are preserved or altered. We introduce Anonpsy, a de-identification framework that reformulates the task as graph-guided semantic rewriting. Anonpsy (1) converts each narrative into a semantic graph encoding clinical entities, temporal anchors, and typed relations; (2) applies graph-constrained perturbations that modify identifying context while preserving clinically essential structure; and (3) regenerates text via graph-conditioned LLM generation. Evaluated on 90 clinician-authored psychiatric case narratives, Anonpsy preserves diagnostic fidelity while achieving consistently low re-identification risk under expert, semantic, and GPT-5-based evaluations. Compared with a strong LLM-only rewriting baseline, Anonpsy yields substantially lower semantic similarity and identifiability. These results demonstrate that explicit structural representations combined with constrained generation provide an effective approach to de-identification for psychiatric narratives. 

---
# PhysicsSolutionAgent: Towards Multimodal Explanations for Numerical Physics Problem Solving 

**Authors**: Aditya Thole, Anmol Agrawal, Arnav Ramamoorthy, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.13453)  

**Abstract**: Explaining numerical physics problems often requires more than text-based solutions; clear visual reasoning can substantially improve conceptual understanding. While large language models (LLMs) demonstrate strong performance on many physics questions in textual form, their ability to generate long, high-quality visual explanations remains insufficiently explored. In this work, we introduce PhysicsSolutionAgent (PSA), an autonomous agent that generates physics-problem explanation videos of up to six minutes using Manim animations. To evaluate the generated videos, we design an assessment pipeline that performs automated checks across 15 quantitative parameters and incorporates feedback from a vision-language model (VLM) to iteratively improve video quality. We evaluate PSA on 32 videos spanning numerical and theoretical physics problems. Our results reveal systematic differences in video quality depending on problem difficulty and whether the task is numerical or theoretical. Using GPT-5-mini, PSA achieves a 100% video-completion rate with an average automated score of 3.8/5. However, qualitative analysis and human inspection uncover both minor and major issues, including visual layout inconsistencies and errors in how visual content is interpreted during feedback. These findings expose key limitations in reliable Manim code generation and highlight broader challenges in multimodal reasoning and evaluation for visual explanations of numerical physics problems. Our work underscores the need for improved visual understanding, verification, and evaluation frameworks in future multimodal educational systems 

---
# MOSLD-Bench: Multilingual Open-Set Learning and Discovery Benchmark for Text Categorization 

**Authors**: Adriana-Valentina Costache, Daria-Nicoleta Dragomir, Silviu-Florin Gheorghe, Eduard Poesina, Paul Irofti, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2601.13437)  

**Abstract**: Open-set learning and discovery (OSLD) is a challenging machine learning task in which samples from new (unknown) classes can appear at test time. It can be seen as a generalization of zero-shot learning, where the new classes are not known a priori, hence involving the active discovery of new classes. While zero-shot learning has been extensively studied in text classification, especially with the emergence of pre-trained language models, open-set learning and discovery is a comparatively new setup for the text domain. To this end, we introduce the first multilingual open-set learning and discovery (MOSLD) benchmark for text categorization by topic, comprising 960K data samples across 12 languages. To construct the benchmark, we (i) rearrange existing datasets and (ii) collect new data samples from the news domain. Moreover, we propose a novel framework for the OSLD task, which integrates multiple stages to continuously discover and learn new classes. We evaluate several language models, including our own, to obtain results that can be used as reference for future work. We release our benchmark at this https URL. 

---
# Trust Me, I'm an Expert: Decoding and Steering Authority Bias in Large Language Models 

**Authors**: Priyanka Mary Mammen, Emil Joswin, Shankar Venkitachalam  

**Link**: [PDF](https://arxiv.org/pdf/2601.13433)  

**Abstract**: Prior research demonstrates that performance of language models on reasoning tasks can be influenced by suggestions, hints and endorsements. However, the influence of endorsement source credibility remains underexplored. We investigate whether language models exhibit systematic bias based on the perceived expertise of the provider of the endorsement. Across 4 datasets spanning mathematical, legal, and medical reasoning, we evaluate 11 models using personas representing four expertise levels per domain. Our results reveal that models are increasingly susceptible to incorrect/misleading endorsements as source expertise increases, with higher-authority sources inducing not only accuracy degradation but also increased confidence in wrong answers. We also show that this authority bias is mechanistically encoded within the model and a model can be steered away from the bias, thereby improving its performance even when an expert gives a misleading endorsement. 

---
# Beyond Memorization: Testing LLM Reasoning on Unseen Theory of Computation Tasks 

**Authors**: Shlok Shelat, Jay Raval, Souvik Roy, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2601.13392)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance on formal language tasks, yet whether this reflects genuine symbolic reasoning or pattern matching on familiar constructions remains unclear. We introduce a benchmark for deterministic finite automata (DFA) construction from regular languages, comprising factual knowledge questions, seen construction problems from public sources, and two types of unseen problems: hand-crafted instances with multiple interacting constraints and systematically generated problems via Arden's theorem. Models achieve perfect accuracy on factual questions and 84-90% on seen tasks. However, accuracy drops sharply on unseen problems (by 30-64%), with failures stemming from systematic misinterpretation of language constraints, incorrect handling of Kleene-star semantics, and a failure to preserve global consistency. We evaluate a three-stage hint protocol that enables correction of shallow errors but does not reliably resolve globally inconsistent or structurally flawed automata. Our analysis across multiple prompting strategies (direct, Chain-of-Thought, Tree-of-Thought) reveals that errors persist regardless of prompting approach, exposing a fundamental gap between LLMs' ability to generate syntactically plausible DFAs and their capacity for semantically correct formal reasoning. 

---
# Structured Insight from Unstructured Data: Large Language Models for SDOH-Driven Diabetes Risk Prediction 

**Authors**: Sasha Ronaghi, Prerit Choudhary, David H Rehkopf, Bryant Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.13388)  

**Abstract**: Social determinants of health (SDOH) play a critical role in Type 2 Diabetes (T2D) management but are often absent from electronic health records and risk prediction models. Most individual-level SDOH data is collected through structured screening tools, which lack the flexibility to capture the complexity of patient experiences and unique needs of a clinic's population. This study explores the use of large language models (LLMs) to extract structured SDOH information from unstructured patient life stories and evaluate the predictive value of both the extracted features and the narratives themselves for assessing diabetes control. We collected unstructured interviews from 65 T2D patients aged 65 and older, focused on their lived experiences, social context, and diabetes management. These narratives were analyzed using LLMs with retrieval-augmented generation to produce concise, actionable qualitative summaries for clinical interpretation and structured quantitative SDOH ratings for risk prediction modeling. The structured SDOH ratings were used independently and in combination with traditional laboratory biomarkers as inputs to linear and tree-based machine learning models (Ridge, Lasso, Random Forest, and XGBoost) to demonstrate how unstructured narrative data can be applied in conventional risk prediction workflows. Finally, we evaluated several LLMs on their ability to predict a patient's level of diabetes control (low, medium, high) directly from interview text with A1C values redacted. LLMs achieved 60% accuracy in predicting diabetes control levels from interview text. This work demonstrates how LLMs can translate unstructured SDOH-related data into structured insights, offering a scalable approach to augment clinical risk models and decision-making. 

---
# Confidence over Time: Confidence Calibration with Temporal Logic for Large Language Model Reasoning 

**Authors**: Zhenjiang Mao, Anirudhh Venkat, Artem Bisliouk, Akshat Kothiyal, Sindhura Kumbakonam Subramanian, Saithej Singhu, Ivan Ruchkin  

**Link**: [PDF](https://arxiv.org/pdf/2601.13387)  

**Abstract**: Large Language Models (LLMs) increasingly rely on long-form, multi-step reasoning to solve complex tasks such as mathematical problem solving and scientific question answering. Despite strong performance, existing confidence estimation methods typically reduce an entire reasoning process to a single scalar score, ignoring how confidence evolves throughout the generation. As a result, these methods are often sensitive to superficial factors such as response length or verbosity, and struggle to distinguish correct reasoning from confidently stated errors. We propose to characterize the stepwise confidence signal using Signal Temporal Logic (STL). Using a discriminative STL mining procedure, we discover temporal formulas that distinguish confidence signals of correct and incorrect responses. Our analysis found that the STL patterns generalize across tasks, and numeric parameters exhibit sensitivity to individual questions. Based on these insights, we develop a confidence estimation approach that informs STL blocks with parameter hypernetworks. Experiments on multiple reasoning tasks show our confidence scores are more calibrated than the baselines. 

---
# Recurrent Confidence Chain: Temporal-Aware Uncertainty Quantification in Large Language Models 

**Authors**: Zhenjiang Mao, Anirudhh Venkat  

**Link**: [PDF](https://arxiv.org/pdf/2601.13368)  

**Abstract**: As reasoning modules, such as the chain-of-thought mechanism, are applied to large language models, they achieve strong performance on various tasks such as answering common-sense questions and solving math problems. The main challenge now is to assess the uncertainty of answers, which can help prevent misleading or serious hallucinations for users. Although current methods analyze long reasoning sequences by filtering unrelated tokens and examining potential connections between nearby tokens or sentences, the temporal spread of confidence is often overlooked. This oversight can lead to inflated overall confidence, even when earlier steps exhibit very low confidence. To address this issue, we propose a novel method that incorporates inter-step attention to analyze semantic correlations across steps. For handling long-horizon responses, we introduce a hidden confidence mechanism to retain historical confidence information, which is then combined with stepwise confidence to produce a more accurate overall estimate. We evaluate our method on the GAOKAO math benchmark and the CLadder causal reasoning dataset using mainstream open-source large language models. Our approach is shown to outperform state-of-the-art methods by achieving a superior balance between predictive quality and calibration, demonstrated by strong performance on both Negative Log-Likelihood and Expected Calibration Error. 

---
# Sockpuppetting: Jailbreaking LLMs Without Optimization Through Output Prefix Injection 

**Authors**: Asen Dotsinski, Panagiotis Eustratiadis  

**Link**: [PDF](https://arxiv.org/pdf/2601.13359)  

**Abstract**: As open-weight large language models (LLMs) increase in capabilities, safeguarding them against malicious prompts and understanding possible attack vectors becomes ever more important. While automated jailbreaking methods like GCG [Zou et al., 2023] remain effective, they often require substantial computational resources and specific expertise. We introduce "sockpuppetting'', a simple method for jailbreaking open-weight LLMs by inserting an acceptance sequence (e.g., "Sure, here is how to...'') at the start of a model's output and allowing it to complete the response. Requiring only a single line of code and no optimization, sockpuppetting achieves up to 80% higher attack success rate (ASR) than GCG on Qwen3-8B in per-prompt comparisons. We also explore a hybrid approach that optimizes the adversarial suffix within the assistant message block rather than the user prompt, increasing ASR by 64% over GCG on Llama-3.1-8B in a prompt-agnostic setting. The results establish sockpuppetting as an effective low-cost attack accessible to unsophisticated adversaries, highlighting the need for defences against output-prefix injection in open-weight models. 

---
# LLM-as-RNN: A Recurrent Language Model for Memory Updates and Sequence Prediction 

**Authors**: Yuxing Lu, J. Ben Tamo, Weichen Zhao, Nan Sun, Yishan Zhong, Wenqi Shi, Jinzhuo Wang, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13352)  

**Abstract**: Large language models are strong sequence predictors, yet standard inference relies on immutable context histories. After making an error at generation step t, the model lacks an updatable memory mechanism that improves predictions for step t+1. We propose LLM-as-RNN, an inference-only framework that turns a frozen LLM into a recurrent predictor by representing its hidden state as natural-language memory. This state, implemented as a structured system-prompt summary, is updated at each timestep via feedback-driven text rewrites, enabling learning without parameter updates. Under a fixed token budget, LLM-as-RNN corrects errors and retains task-relevant patterns, effectively performing online learning through language. We evaluate the method on three sequential benchmarks in healthcare, meteorology, and finance across Llama, Gemma, and GPT model families. LLM-as-RNN significantly outperforms zero-shot, full-history, and MemPrompt baselines, improving predictive accuracy by 6.5% on average, while producing interpretable, human-readable learning traces absent in standard context accumulation. 

---
# AfroScope: A Framework for Studying the Linguistic Landscape of Africa 

**Authors**: Sang Yun Kwon, AbdelRahim Elmadany, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2601.13346)  

**Abstract**: Language Identification (LID) is the task of determining the language of a given text and is a fundamental preprocessing step that affects the reliability of downstream NLP applications. While recent work has expanded LID coverage for African languages, existing approaches remain limited in (i) the number of supported languages and (ii) their ability to make fine-grained distinctions among closely related varieties. We introduce AfroScope, a unified framework for African LID that includes AfroScope-Data, a dataset covering 713 African languages, and AfroScope-Models, a suite of strong LID models with broad language coverage. To better distinguish highly confusable languages, we propose a hierarchical classification approach that leverages Mirror-Serengeti, a specialized embedding model targeting 29 closely related or geographically proximate languages. This approach improves macro F1 by 4.55 on this confusable subset compared to our best base model. Finally, we analyze cross linguistic transfer and domain effects, offering guidance for building robust African LID systems. We position African LID as an enabling technology for large scale measurement of Africas linguistic landscape in digital text and release AfroScope-Data and AfroScope-Models publicly. 

---
# RegCheck: A tool for automating comparisons between study registrations and papers 

**Authors**: Jamie Cummins, Beth Clarke, Ian Hussey, Malte Elson  

**Link**: [PDF](https://arxiv.org/pdf/2601.13330)  

**Abstract**: Across the social and medical sciences, researchers recognize that specifying planned research activities (i.e., 'registration') prior to the commencement of research has benefits for both the transparency and rigour of science. Despite this, evidence suggests that study registrations frequently go unexamined, minimizing their effectiveness. In a way this is no surprise: manually checking registrations against papers is labour- and time-intensive, requiring careful reading across formats and expertise across domains. The advent of AI unlocks new possibilities in facilitating this activity. We present RegCheck, a modular LLM-assisted tool designed to help researchers, reviewers, and editors from across scientific disciplines compare study registrations with their corresponding papers. Importantly, RegCheck keeps human expertise and judgement in the loop by (i) ensuring that users are the ones who determine which features should be compared, and (ii) presenting the most relevant text associated with each feature to the user, facilitating (rather than replacing) human discrepancy judgements. RegCheck also generates shareable reports with unique RegCheck IDs, enabling them to be easily shared and verified by other users. RegCheck is designed to be adaptable across scientific domains, as well as registration and publication formats. In this paper we provide an overview of the motivation, workflow, and design principles of RegCheck, and we discuss its potential as an extensible infrastructure for reproducible science with an example use case. 

---
# Reducing Tokenization Premiums for Low-Resource Languages 

**Authors**: Geoffrey Churchill, Steven Skiena  

**Link**: [PDF](https://arxiv.org/pdf/2601.13328)  

**Abstract**: Relative to English, low-resource languages suffer from substantial tokenization premiums in modern LMs, meaning that it generally requires several times as many tokens to encode a sentence in a low-resource language than to encode the analogous sentence in English. This tokenization premium results in increased API and energy costs and reduced effective context windows for these languages. In this paper we analyze the tokenizers of ten popular LMs to better understand their designs and per-language tokenization premiums. We also propose a mechanism to reduce tokenization premiums in pre-trained models, by post-hoc additions to the token vocabulary that coalesce multi-token characters into single tokens. We apply this methodology to 12 low-resource languages, demonstrating that the original and compressed inputs often have similar last hidden states when run through the Llama 3.2 1B model. 

---
# Arab Voices: Mapping Standard and Dialectal Arabic Speech Technology 

**Authors**: Peter Sullivan, AbdelRahim Elmadany, Alcides Alcoba Inciarte, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2601.13319)  

**Abstract**: Dialectal Arabic (DA) speech data vary widely in domain coverage, dialect labeling practices, and recording conditions, complicating cross-dataset comparison and model evaluation. To characterize this landscape, we conduct a computational analysis of linguistic ``dialectness'' alongside objective proxies of audio quality on the training splits of widely used DA corpora. We find substantial heterogeneity both in acoustic conditions and in the strength and consistency of dialectal signals across datasets, underscoring the need for standardized characterization beyond coarse labels. To reduce fragmentation and support reproducible evaluation, we introduce Arab Voices, a standardized framework for DA ASR. Arab Voices provides unified access to 31 datasets spanning 14 dialects, with harmonized metadata and evaluation utilities. We further benchmark a range of recent ASR systems, establishing strong baselines for modern DA ASR. 

---
# Paid Voices vs. Public Feeds: Interpretable Cross-Platform Theme Modeling of Climate Discourse 

**Authors**: Samantha Sudhoff, Pranav Perumal, Zhaoqing Wu, Tunazzina Islam  

**Link**: [PDF](https://arxiv.org/pdf/2601.13317)  

**Abstract**: Climate discourse online plays a crucial role in shaping public understanding of climate change and influencing political and policy outcomes. However, climate communication unfolds across structurally distinct platforms with fundamentally different incentive structures: paid advertising ecosystems incentivize targeted, strategic persuasion, while public social media platforms host largely organic, user-driven discourse. Existing computational studies typically analyze these environments in isolation, limiting our ability to distinguish institutional messaging from public expression. In this work, we present a comparative analysis of climate discourse across paid advertisements on Meta (previously known as Facebook) and public posts on Bluesky from July 2024 to September 2025. We introduce an interpretable, end-to-end thematic discovery and assignment framework that clusters texts by semantic similarity and leverages large language models (LLMs) to generate concise, human-interpretable theme labels. We evaluate the quality of the induced themes against traditional topic modeling baselines using both human judgments and an LLM-based evaluator, and further validate their semantic coherence through downstream stance prediction and theme-guided retrieval tasks. Applying the resulting themes, we characterize systematic differences between paid climate messaging and public climate discourse and examine how thematic prevalence shifts around major political events. Our findings show that platform-level incentives are reflected in the thematic structure, stance alignment, and temporal responsiveness of climate narratives. While our empirical analysis focuses on climate communication, the proposed framework is designed to support comparative narrative analysis across heterogeneous communication environments. 

---
# OI-Bench: An Option Injection Benchmark for Evaluating LLM Susceptibility to Directive Interference 

**Authors**: Yow-Fu Liou, Yu-Chien Tang, Yu-Hsiang Liu, An-Zi Yen  

**Link**: [PDF](https://arxiv.org/pdf/2601.13300)  

**Abstract**: Benchmarking large language models (LLMs) is critical for understanding their capabilities, limitations, and robustness. In addition to interface artifacts, prior studies have shown that LLM decisions can be influenced by directive signals such as social cues, framing, and instructions. In this work, we introduce option injection, a benchmarking approach that augments the multiple-choice question answering (MCQA) interface with an additional option containing a misleading directive, leveraging standardized choice structure and scalable evaluation. We construct OI-Bench, a benchmark of 3,000 questions spanning knowledge, reasoning, and commonsense tasks, with 16 directive types covering social compliance, bonus framing, threat framing, and instructional interference. This setting combines manipulation of the choice interface with directive-based interference, enabling systematic assessment of model susceptibility. We evaluate 12 LLMs to analyze attack success rates, behavioral responses, and further investigate mitigation strategies ranging from inference-time prompting to post-training alignment. Experimental results reveal substantial vulnerabilities and heterogeneous robustness across models. OI-Bench is expected to support more systematic evaluation of LLM robustness to directive interference within choice-based interfaces. 

---
# A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification 

**Authors**: Gonzalo Ariel Meyoyan, Luciano Del Corro  

**Link**: [PDF](https://arxiv.org/pdf/2601.13288)  

**Abstract**: Production LLM systems often rely on separate models for safety and other classification-heavy steps, increasing latency, VRAM footprint, and operational complexity. We instead reuse computation already paid for by the serving LLM: we train lightweight probes on its hidden states and predict labels in the same forward pass used for generation. We frame classification as representation selection over the full token-layer hidden-state tensor, rather than committing to a fixed token or fixed layer (e.g., first-token logits or final-layer pooling). To implement this, we introduce a two-stage aggregator that (i) summarizes tokens within each layer and (ii) aggregates across layer summaries to form a single representation for classification. We instantiate this template with direct pooling, a 100K-parameter scoring-attention gate, and a downcast multi-head self-attention (MHA) probe with up to 35M trainable parameters. Across safety and sentiment benchmarks our probes improve over logit-only reuse (e.g., MULI) and are competitive with substantially larger task-specific baselines, while preserving near-serving latency and avoiding the VRAM and latency costs of a separate guard-model pipeline. 

---
# Unlearning in LLMs: Methods, Evaluation, and Open Challenges 

**Authors**: Tyler Lizzo, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2601.13264)  

**Abstract**: Large language models (LLMs) have achieved remarkable success across natural language processing tasks, yet their widespread deployment raises pressing concerns around privacy, copyright, security, and bias. Machine unlearning has emerged as a promising paradigm for selectively removing knowledge or data from trained models without full retraining. In this survey, we provide a structured overview of unlearning methods for LLMs, categorizing existing approaches into data-centric, parameter-centric, architecture-centric, hybrid, and other strategies. We also review the evaluation ecosystem, including benchmarks, metrics, and datasets designed to measure forgetting effectiveness, knowledge retention, and robustness. Finally, we outline key challenges and open problems, such as scalable efficiency, formal guarantees, cross-language and multimodal unlearning, and robustness against adversarial relearning. By synthesizing current progress and highlighting open directions, this paper aims to serve as a roadmap for developing reliable and responsible unlearning techniques in large language models. 

---
# Stop Taking Tokenizers for Granted: They Are Core Design Decisions in Large Language Models 

**Authors**: Sawsan Alqahtani, Mir Tafseer Nayeem, Md Tahmid Rahman Laskar, Tasnim Mohiuddin, M Saiful Bari  

**Link**: [PDF](https://arxiv.org/pdf/2601.13260)  

**Abstract**: Tokenization underlies every large language model, yet it remains an under-theorized and inconsistently designed component. Common subword approaches such as Byte Pair Encoding (BPE) offer scalability but often misalign with linguistic structure, amplify bias, and waste capacity across languages and domains. This paper reframes tokenization as a core modeling decision rather than a preprocessing step. We argue for a context-aware framework that integrates tokenizer and model co-design, guided by linguistic, domain, and deployment considerations. Standardized evaluation and transparent reporting are essential to make tokenization choices accountable and comparable. Treating tokenization as a core design problem, not a technical afterthought, can yield language technologies that are fairer, more efficient, and more adaptable. 

---
# A Hybrid Protocol for Large-Scale Semantic Dataset Generation in Low-Resource Languages: The Turkish Semantic Relations Corpus 

**Authors**: Ebubekir Tosun, Mehmet Emin Buldur, Özay Ezerceli, Mahmoud ElHussieni  

**Link**: [PDF](https://arxiv.org/pdf/2601.13253)  

**Abstract**: We present a hybrid methodology for generating large-scale semantic relationship datasets in low-resource languages, demonstrated through a comprehensive Turkish semantic relations corpus. Our approach integrates three phases: (1) FastText embeddings with Agglomerative Clustering to identify semantic clusters, (2) Gemini 2.5-Flash for automated semantic relationship classification, and (3) integration with curated dictionary sources. The resulting dataset comprises 843,000 unique Turkish semantic pairs across three relationship types (synonyms, antonyms, co-hyponyms) representing a 10x scale increase over existing resources at minimal cost ($65). We validate the dataset through two downstream tasks: an embedding model achieving 90% top-1 retrieval accuracy and a classification model attaining 90% F1-macro. Our scalable protocol addresses critical data scarcity in Turkish NLP and demonstrates applicability to other low-resource languages. We publicly release the dataset and models. 

---
# Beyond Cosine Similarity: Taming Semantic Drift and Antonym Intrusion in a 15-Million Node Turkish Synonym Graph 

**Authors**: Ebubekir Tosun, Mehmet Emin Buldur, Özay Ezerceli, Mahmoud ElHussieni  

**Link**: [PDF](https://arxiv.org/pdf/2601.13251)  

**Abstract**: Neural embeddings have a notorious blind spot: they can't reliably tell synonyms apart from antonyms. Consequently, increasing similarity thresholds often fails to prevent opposites from being grouped together. We've built a large-scale semantic clustering system specifically designed to tackle this problem head on. Our pipeline chews through 15 million lexical items, evaluates a massive 520 million potential relationships, and ultimately generates 2.9 million high-precision semantic clusters. The system makes three primary contributions. First, we introduce a labeled dataset of 843,000 concept pairs spanning synonymy, antonymy, and co-hyponymy, constructed via Gemini 2.5-Flash LLM augmentation and verified using human-curated dictionary resources. Second, we propose a specialized three-way semantic relation discriminator that achieves 90% macro-F1, enabling robust disambiguation beyond raw embedding similarity. Third, we introduce a novel soft-to-hard clustering algorithm that mitigates semantic drift preventing erroneous transitive chains (e.g., hot -> spicy -> pain -> depression) while simultaneously resolving polysemy. Our approach employs a topology-aware two-stage expansion-pruning procedure with topological voting, ensuring that each term is assigned to exactly one semantically coherent cluster. The resulting resource enables high-precision semantic search and retrieval-augmented generation, particularly for morphologically rich and low-resource languages where existing synonym databases remain sparse. 

---
# Aligning Agentic World Models via Knowledgeable Experience Learning 

**Authors**: Baochang Ren, Yunzhi Yao, Rui Sun, Shuofei Qiao, Ningyu Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.13247)  

**Abstract**: Current Large Language Models (LLMs) exhibit a critical modal disconnect: they possess vast semantic knowledge but lack the procedural grounding to respect the immutable laws of the physical world. Consequently, while these agents implicitly function as world models, their simulations often suffer from physical hallucinations-generating plans that are logically sound but physically unexecutable. Existing alignment strategies predominantly rely on resource-intensive training or fine-tuning, which attempt to compress dynamic environmental rules into static model parameters. However, such parametric encapsulation is inherently rigid, struggling to adapt to the open-ended variability of physical dynamics without continuous, costly retraining. To bridge this gap, we introduce WorldMind, a framework that autonomously constructs a symbolic World Knowledge Repository by synthesizing environmental feedback. Specifically, it unifies Process Experience to enforce physical feasibility via prediction errors and Goal Experience to guide task optimality through successful trajectories. Experiments on EB-ALFRED and EB-Habitat demonstrate that WorldMind achieves superior performance compared to baselines with remarkable cross-model and cross-environment transferability. 

---
# Autoregressive Models Rival Diffusion Models at ANY-ORDER Generation 

**Authors**: Tianqi Du, Lizhe Fang, Weijie Yang, Chenheng Zhang, Zeming Wei, Yifei Wang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13228)  

**Abstract**: Diffusion language models enable any-order generation and bidirectional conditioning, offering appealing flexibility for tasks such as infilling, rewriting, and self-correction. However, their formulation-predicting one part of a sequence from another within a single-step dependency-limits modeling depth and often yields lower sample quality and stability than autoregressive (AR) models. To address this, we revisit autoregressive modeling as a foundation and reformulate diffusion-style training into a structured multi-group prediction process. We propose Any-order Any-subset Autoregressive modeling (A3), a generalized framework that extends the standard AR factorization to arbitrary token groups and generation orders. A3 preserves the probabilistic rigor and multi-layer dependency modeling of AR while inheriting diffusion models' flexibility for parallel and bidirectional generation. We implement A3 through a two-stream attention architecture and a progressive adaptation strategy that transitions pretrained AR models toward any-order prediction. Experiments on question answering, commonsense reasoning, and story infilling demonstrate that A3 outperforms diffusion-based models while maintaining flexible decoding. This work offers a unified approach for a flexible, efficient, and novel language modeling paradigm. 

---
# Beyond Single-shot Writing: Deep Research Agents are Unreliable at Multi-turn Report Revision 

**Authors**: Bingsen Chen, Boyan Li, Ping Nie, Yuyu Zhang, Xi Ye, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.13217)  

**Abstract**: Existing benchmarks for Deep Research Agents (DRAs) treat report generation as a single-shot writing task, which fundamentally diverges from how human researchers iteratively draft and revise reports via self-reflection or peer feedback. Whether DRAs can reliably revise reports with user feedback remains unexplored. We introduce Mr Dre, an evaluation suite that establishes multi-turn report revision as a new evaluation axis for DRAs. Mr Dre consists of (1) a unified long-form report evaluation protocol spanning comprehensiveness, factuality, and presentation, and (2) a human-verified feedback simulation pipeline for multi-turn revision. Our analysis of five diverse DRAs reveals a critical limitation: while agents can address most user feedback, they also regress on 16-27% of previously covered content and citation quality. Over multiple revision turns, even the best-performing agents leave significant headroom, as they continue to disrupt content outside the feedback's scope and fail to preserve earlier edits. We further show that these issues are not easily resolvable through inference-time fixes such as prompt engineering and a dedicated sub-agent for report revision. 

---
# OpenExempt: A Diagnostic Benchmark for Legal Reasoning and a Framework for Creating Custom Benchmarks on Demand 

**Authors**: Sergio Servantez, Sarah B. Lawsky, Rajiv Jain, Daniel W. Linna Jr., Kristian Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2601.13183)  

**Abstract**: Reasoning benchmarks have played a crucial role in the progress of language models. Yet rigorous evaluation remains a significant challenge as static question-answer pairs provide only a snapshot of performance, compressing complex behavior into a single accuracy metric. This limitation is especially true in complex, rule-bound domains such as law, where existing benchmarks are costly to build and ill suited for isolating specific failure modes. To address this, we introduce OpenExempt, a framework and benchmark for diagnostic evaluation of legal reasoning. The OpenExempt Framework uses expert-crafted symbolic representations of U.S. Bankruptcy Code statutes to dynamically generate a large space of natural language reasoning tasks and their machine-computable solutions on demand. This gives users fine-grained control over task complexity and scope, allowing individual reasoning skills to be probed in isolation. Using this system, we construct the OpenExempt Benchmark, a diagnostic benchmark for legal reasoning with 9,765 samples across nine evaluation suites designed to carefully probe model capabilities. Experiments on 13 diverse language models reveal sharp performance cliffs that emerge only under longer reasoning paths and in the presence of obfuscating statements. We release the framework and benchmark publicly to support research aimed at understanding and improving the next generation of reasoning systems. 

---
# Medical Triage as Pairwise Ranking: A Benchmark for Urgency in Patient Portal Messages 

**Authors**: Joseph Gatto, Parker Seegmiller, Timothy Burdick, Philip Resnik, Roshnik Rahat, Sarah DeLozier, Sarah M. Preum  

**Link**: [PDF](https://arxiv.org/pdf/2601.13178)  

**Abstract**: Medical triage is the task of allocating medical resources and prioritizing patients based on medical need. This paper introduces the first large-scale public dataset for studying medical triage in the context of asynchronous outpatient portal messages. Our novel task formulation views patient message triage as a pairwise inference problem, where we train LLMs to choose `"which message is more medically urgent" in a head-to-head tournament-style re-sort of a physician's inbox. Our novel benchmark PMR-Bench contains 1569 unique messages and 2,000+ high-quality test pairs for pairwise medical urgency assessment alongside a scalable training data generation pipeline. PMR-Bench includes samples that contain both unstructured patient-written messages alongside real electronic health record (EHR) data, emulating a real-world medical triage scenario.
We develop a novel automated data annotation strategy to provide LLMs with in-domain guidance on this task. The resulting data is used to train two model classes, UrgentReward and UrgentSFT, leveraging Bradley-Terry and next token prediction objective, respectively to perform pairwise urgency classification. We find that UrgentSFT achieves top performance on PMR-Bench, with UrgentReward showing distinct advantages in low-resource settings. For example, UrgentSFT-8B and UrgentReward-8B provide a 15- and 16-point boost, respectively, on inbox sorting metrics over off-the-shelf 8B models. Paper resources can be found at this https URL 

---
# Probe and Skip: Self-Predictive Token Skipping for Efficient Long-Context LLM Inference 

**Authors**: Zimeng Wu, Donghao Wang, Chaozhe Jin, Jiaxin Chen, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13155)  

**Abstract**: Long-context inference enhances the reasoning capability of Large Language Models (LLMs) while incurring significant computational overhead. Token-oriented methods, such as pruning and skipping, have shown promise in reducing inference latency, but still suffer from inherently limited acceleration potential, outdated proxy signals, and redundancy interference, thus yielding suboptimal speed-accuracy trade-offs. To address these challenges, we propose SPTS (Self-Predictive Token Skipping), a training-free framework for efficient long-context LLM inference. Specifically, motivated by the thought of probing the influence of targeted skipping layers, we design two component-specific strategies for selective token skipping: Partial Attention Probing (PAP) for multi-head attention, which selects informative tokens by performing partial forward attention computation, and Low-rank Transformation Probing (LTP) for feed forward network, which constructs a low-rank proxy network to predict token transformations. Furthermore, a Multi-Stage Delayed Pruning (MSDP) strategy reallocates the skipping budget and progressively prunes redundant tokens across layers. Extensive experiments demonstrate the effectiveness of our method, achieving up to 2.46$\times$ and 2.29$\times$ speedups for prefilling and end-to-end generation, respectively, while maintaining state-of-the-art model performance. The source code will be publicly available upon paper acceptance. 

---
# Adversarial Alignment: Ensuring Value Consistency in Large Language Models for Sensitive Domains 

**Authors**: Yuan Gao, Zhigang Liu, Xinyu Yao, Bo Chen, Xiaobing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.13137)  

**Abstract**: With the wide application of large language models (LLMs), the problems of bias and value inconsistency in sensitive domains have gradually emerged, especially in terms of race, society and politics. In this paper, we propose an adversarial alignment framework, which enhances the value consistency of the model in sensitive domains through continued pre-training, instruction fine-tuning and adversarial training. In adversarial training, we use the Attacker to generate controversial queries, the Actor to generate responses with value consistency, and the Critic to filter and ensure response quality. Furthermore, we train a Value-Consistent Large Language Model, VC-LLM, for sensitive domains, and construct a bilingual evaluation dataset in Chinese and English. The experimental results show that VC-LLM performs better than the existing mainstream models in both Chinese and English tests, verifying the effectiveness of the method. Warning: This paper contains examples of LLMs that are offensive or harmful in nature. 

---
# Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning 

**Authors**: Fengran Mo, Yifan Gao, Sha Li, Hansi Zeng, Xin Liu, Zhaoxuan Tan, Xian Li, Jianshu Chen, Dakuo Wang, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13115)  

**Abstract**: Large Language Models (LLMs) have become a popular interface for human-AI interaction, supporting information seeking and task assistance through natural, multi-turn dialogue. To respond to users within multi-turn dialogues, the context-dependent user intent evolves across interactions, requiring contextual interpretation, query reformulation, and dynamic coordination between retrieval and generation. Existing studies usually follow static rewrite, retrieve, and generate pipelines, which optimize different procedures separately and overlook the mixed-initiative action optimization simultaneously. Although the recent developments in deep search agents demonstrate the effectiveness in jointly optimizing retrieval and generation via reasoning, these approaches focus on single-turn scenarios, which might lack the ability to handle multi-turn interactions. We introduce a conversational agent that interleaves search and reasoning across turns, enabling exploratory and adaptive behaviors learned through reinforcement learning (RL) training with tailored rewards towards evolving user goals. The experimental results across four widely used conversational benchmarks demonstrate the effectiveness of our methods by surpassing several existing strong baselines. 

---
# CORE-T: COherent REtrieval of Tables for Text-to-SQL 

**Authors**: Hassan Soliman, Vivek Gupta, Dan Roth, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2601.13111)  

**Abstract**: Realistic text-to-SQL workflows often require joining multiple tables. As a result, accurately retrieving the relevant set of tables becomes a key bottleneck for end-to-end performance. We study an open-book setting where queries must be answered over large, heterogeneous table collections pooled from many sources, without clean scoping signals such as database identifiers. Here, dense retrieval (DR) achieves high recall but returns many distractors, while join-aware alternatives often rely on extra assumptions and/or incur high inference overhead. We propose CORE-T, a scalable, training-free framework that enriches tables with LLM-generated purpose metadata and pre-computes a lightweight table-compatibility cache. At inference time, DR returns top-K candidates; a single LLM call selects a coherent, joinable subset, and a simple additive adjustment step restores strongly compatible tables. Across Bird, Spider, and MMQA, CORE-T improves table-selection F1 by up to 22.7 points while retrieving up to 42% fewer tables, improving multi-table execution accuracy by up to 5.0 points on Bird and 6.9 points on MMQA, and using 4-5x fewer tokens than LLM-intensive baselines. 

---
# Leveraging Lora Fine-Tuning and Knowledge Bases for Construction Identification 

**Authors**: Liu Kaipeng, Wu Ling  

**Link**: [PDF](https://arxiv.org/pdf/2601.13105)  

**Abstract**: This study investigates the automatic identification of the English ditransitive construction by integrating LoRA-based fine-tuning of a large language model with a Retrieval-Augmented Generation (RAG) framework.A binary classification task was conducted on annotated data from the British National Corpus. Results demonstrate that a LoRA-fine-tuned Qwen3-8B model significantly outperformed both a native Qwen3-MAX model and a theory-only RAG system. Detailed error analysis reveals that fine-tuning shifts the model's judgment from a surface-form pattern matching towards a more semantically grounded understanding based. 

---
# Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs 

**Authors**: Abdellah El Mekki, Samar M. Magdy, Houdaifa Atou, Ruwa AbuHweidi, Baraah Qawasmeh, Omer Nacar, Thikra Al-hibiri, Razan Saadie, Hamzah Alsayadi, Nadia Ghezaiel Hammouda, Alshima Alkhazimi, Aya Hamod, Al-Yas Al-Ghafri, Wesam El-Sayed, Asila Al sharji, Mohamad Ballout, Anas Belfathi, Karim Ghaddar, Serry Sibaee, Alaa Aoun, Areej Asiri, Lina Abureesh, Ahlam Bashiti, Majdal Yousef, Abdulaziz Hafiz, Yehdih Mohamed, Emira Hamedtou, Brakehe Brahim, Rahaf Alhamouri, Youssef Nafea, Aya El Aatar, Walid Al-Dhabyani, Emhemed Hamed, Sara Shatnawi, Fakhraddin Alwajih, Khalid Elkhidir, Ashwag Alasmari, Abdurrahman Gerrio, Omar Alshahri, AbdelRahim A. Elmadany, Ismail Berrada, Amir Azad Adli Alkathiri, Fadi A Zaraket, Mustafa Jarrar, Yahya Mohamed El Hadj, Hassan Alhuzali, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2601.13099)  

**Abstract**: Arabic is a highly diglossic language where most daily communication occurs in regional dialects rather than Modern Standard Arabic. Despite this, machine translation (MT) systems often generalize poorly to dialectal input, limiting their utility for millions of speakers. We introduce \textbf{Alexandria}, a large-scale, community-driven, human-translated dataset designed to bridge this gap. Alexandria covers 13 Arab countries and 11 high-impact domains, including health, education, and agriculture. Unlike previous resources, Alexandria provides unprecedented granularity by associating contributions with city-of-origin metadata, capturing authentic local varieties beyond coarse regional labels. The dataset consists of multi-turn conversational scenarios annotated with speaker-addressee gender configurations, enabling the study of gender-conditioned variation in dialectal use. Comprising 107K total samples, Alexandria serves as both a training resource and a rigorous benchmark for evaluating MT and Large Language Models (LLMs). Our automatic and human evaluation of Arabic-aware LLMs benchmarks current capabilities in translating across diverse Arabic dialects and sub-dialects, while exposing significant persistent challenges. 

---
# Profiling German Text Simplification with Interpretable Model-Fingerprints 

**Authors**: Lars Klöser, Mika Beele, Bodo Kraft  

**Link**: [PDF](https://arxiv.org/pdf/2601.13050)  

**Abstract**: While Large Language Models (LLMs) produce highly nuanced text simplifications, developers currently lack tools for a holistic, efficient, and reproducible diagnosis of their behavior. This paper introduces the Simplification Profiler, a diagnostic toolkit that generates a multidimensional, interpretable fingerprint of simplified texts. Multiple aggregated simplifications of a model result in a model's fingerprint. This novel evaluation paradigm is particularly vital for languages, where the data scarcity problem is magnified when creating flexible models for diverse target groups rather than a single, fixed simplification style. We propose that measuring a model's unique behavioral signature is more relevant in this context as an alternative to correlating metrics with human preferences. We operationalize this with a practical meta-evaluation of our fingerprints' descriptive power, which bypasses the need for large, human-rated datasets. This test measures if a simple linear classifier can reliably identify various model configurations by their created simplifications, confirming that our metrics are sensitive to a model's specific characteristics. The Profiler can distinguish high-level behavioral variations between prompting strategies and fine-grained changes from prompt engineering, including few-shot examples. Our complete feature set achieves classification F1-scores up to 71.9 %, improving upon simple baselines by over 48 percentage points. The Simplification Profiler thus offers developers a granular, actionable analysis to build more effective and truly adaptive text simplification systems. 

---
# Typhoon ASR Real-time: FastConformer-Transducer for Thai Automatic Speech Recognition 

**Authors**: Warit Sirichotedumrong, Adisai Na-Thalang, Potsawee Manakul, Pittawat Taveekitworachai, Sittipong Sripaisarnmongkol, Kunat Pipatanakul  

**Link**: [PDF](https://arxiv.org/pdf/2601.13044)  

**Abstract**: Large encoder-decoder models like Whisper achieve strong offline transcription but remain impractical for streaming applications due to high latency. However, due to the accessibility of pre-trained checkpoints, the open Thai ASR landscape remains dominated by these offline architectures, leaving a critical gap in efficient streaming solutions. We present Typhoon ASR Real-time, a 115M-parameter FastConformer-Transducer model for low-latency Thai speech recognition. We demonstrate that rigorous text normalization can match the impact of model scaling: our compact model achieves a 45x reduction in computational cost compared to Whisper Large-v3 while delivering comparable accuracy. Our normalization pipeline resolves systemic ambiguities in Thai transcription --including context-dependent number verbalization and repetition markers (mai yamok) --creating consistent training targets. We further introduce a two-stage curriculum learning approach for Isan (north-eastern) dialect adaptation that preserves Central Thai performance. To address reproducibility challenges in Thai ASR, we release the Typhoon ASR Benchmark, a gold-standard human-labeled datasets with transcriptions following established Thai linguistic conventions, providing standardized evaluation protocols for the research community. 

---
# SASA: Semantic-Aware Contrastive Learning Framework with Separated Attention for Triple Classification 

**Authors**: Xu Xiaodan, Hu Xiaolin  

**Link**: [PDF](https://arxiv.org/pdf/2601.13035)  

**Abstract**: Knowledge Graphs~(KGs) often suffer from unreliable knowledge, which restricts their utility. Triple Classification~(TC) aims to determine the validity of triples from KGs. Recently, text-based methods learn entity and relation representations from natural language descriptions, significantly improving the generalization capabilities of TC models and setting new benchmarks in performance. However, there are still two critical challenges. First, existing methods often ignore the effective semantic interaction among different KG components. Second, most approaches adopt single binary classification training objective, leading to insufficient semantic representation learning. To address these challenges, we propose \textbf{SASA}, a novel framework designed to enhance TC models via separated attention mechanism and semantic-aware contrastive learning~(CL). Specifically, we first propose separated attention mechanism to encode triples into decoupled contextual representations and then fuse them through a more effective interactive way. Then, we introduce semantic-aware hierarchical CL as auxiliary training objective to guide models in improving their discriminative capabilities and achieving sufficient semantic learning, considering both local level and global level CL. Experimental results across two benchmark datasets demonstrate that SASA significantly outperforms state-of-the-art methods. In terms of accuracy, we advance the state-of-the-art by +5.9\% on FB15k-237 and +3.4\% on YAGO3-10. 

---
# Tears or Cheers? Benchmarking LLMs via Culturally Elicited Distinct Affective Responses 

**Authors**: Chongyuan Dai, Yaling Shen, Jinpeng Hu, Zihan Gao, Jia Li, Yishun Jiang, Yaxiong Wang, Liu Liu, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2601.13024)  

**Abstract**: Culture serves as a fundamental determinant of human affective processing and profoundly shapes how individuals perceive and interpret emotional stimuli. Despite this intrinsic link extant evaluations regarding cultural alignment within Large Language Models primarily prioritize declarative knowledge such as geographical facts or established societal customs. These benchmarks remain insufficient to capture the subjective interpretative variance inherent to diverse sociocultural lenses. To address this limitation, we introduce CEDAR, a multimodal benchmark constructed entirely from scenarios capturing Culturally \underline{\textsc{E}}licited \underline{\textsc{D}}istinct \underline{\textsc{A}}ffective \underline{\textsc{R}}esponses. To construct CEDAR, we implement a novel pipeline that leverages LLM-generated provisional labels to isolate instances yielding cross-cultural emotional distinctions, and subsequently derives reliable ground-truth annotations through rigorous human evaluation. The resulting benchmark comprises 10,962 instances across seven languages and 14 fine-grained emotion categories, with each language including 400 multimodal and 1,166 text-only samples. Comprehensive evaluations of 17 representative multilingual models reveal a dissociation between language consistency and cultural alignment, demonstrating that culturally grounded affective understanding remains a significant challenge for current models. 

---
# Bi-Attention HateXplain : Taking into account the sequential aspect of data during explainability in a multi-task context 

**Authors**: Ghislain Dorian Tchuente Mondjo  

**Link**: [PDF](https://arxiv.org/pdf/2601.13018)  

**Abstract**: Technological advances in the Internet and online social networks have brought many benefits to humanity. At the same time, this growth has led to an increase in hate speech, the main global threat. To improve the reliability of black-box models used for hate speech detection, post-hoc approaches such as LIME, SHAP, and LRP provide the explanation after training the classification model. In contrast, multi-task approaches based on the HateXplain benchmark learn to explain and classify simultaneously. However, results from HateXplain-based algorithms show that predicted attention varies considerably when it should be constant. This attention variability can lead to inconsistent interpretations, instability of predictions, and learning difficulties. To solve this problem, we propose the BiAtt-BiRNN-HateXplain (Bidirectional Attention BiRNN HateXplain) model which is easier to explain compared to LLMs which are more complex in view of the need for transparency, and will take into account the sequential aspect of the input data during explainability thanks to a BiRNN layer. Thus, if the explanation is correctly estimated, thanks to multi-task learning (explainability and classification task), the model could classify better and commit fewer unintentional bias errors related to communities. The experimental results on HateXplain data show a clear improvement in detection performance, explainability and a reduction in unintentional bias. 

---
# Graph Reasoning Paradigm: Structured and Symbolic Reasoning with Topology-Aware Reinforcement Learning for Large Language Models 

**Authors**: Runxuan Liu, Xianhao Ou, Xinyan Ma, Jiyuan Wang, Jiafeng Liang, Jiaqi Li, Tao He, Zheng Chu, Rongchuan Mu, Zekun Wang, Baoxin Wang, Dayong Wu, Ming Liu, Shijin Wang, Guoping Hu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.12995)  

**Abstract**: Long Chain-of-Thought (LCoT), achieved by Reinforcement Learning with Verifiable Rewards (RLVR), has proven effective in enhancing the reasoning capabilities of Large Language Models (LLMs). However, reasoning in current LLMs is primarily generated as plain text, where performing semantic evaluation on such unstructured data creates a computational bottleneck during training. Despite RLVR-based optimization, existing methods still suffer from coarse-grained supervision, reward hacking, high training costs, and poor generalization. To address these issues, we propose the Graph Reasoning Paradigm (GRP), which realizes structured and symbolic reasoning, implemented via graph-structured representations with step-level cognitive labels. Building upon GRP, we further design Process-Aware Stratified Clipping Group Relative Policy Optimization (PASC-GRPO), which leverages structured evaluation to replace semantic evaluation, achieves process-aware verification through graph-structured outcome rewards, and mitigates reward hacking via stratified clipping advantage estimation. Experiments demonstrate significant improvements across mathematical reasoning and code generation tasks. Data, models, and code will be released later. 

---
# ChartAttack: Testing the Vulnerability of LLMs to Malicious Prompting in Chart Generation 

**Authors**: Jesus-German Ortiz-Barajas, Jonathan Tonglet, Vivek Gupta, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2601.12983)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly used to automate chart generation from data tables, enabling efficient data analysis and reporting but also introducing new misuse risks. In this work, we introduce ChartAttack, a novel framework for evaluating how MLLMs can be misused to generate misleading charts at scale. ChartAttack injects misleaders into chart designs, aiming to induce incorrect interpretations of the underlying data. Furthermore, we create AttackViz, a chart question-answering (QA) dataset where each (chart specification, QA) pair is labeled with effective misleaders and their induced incorrect answers. Experiments in in-domain and cross-domain settings show that ChartAttack significantly degrades the QA performance of MLLM readers, reducing accuracy by an average of 19.6 points and 14.9 points, respectively. A human study further shows an average 20.2 point drop in accuracy for participants exposed to misleading charts generated by ChartAttack. Our findings highlight an urgent need for robustness and security considerations in the design, evaluation, and deployment of MLLM-based chart generation systems. We make our code and data publicly available. 

---
# The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check 

**Authors**: Qingyu Lu, Liang Ding, Kanjian Zhang, Jinxia Zhang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2601.12979)  

**Abstract**: The pursuit of real-time agentic interaction has driven interest in Diffusion-based Large Language Models (dLLMs) as alternatives to auto-regressive backbones, promising to break the sequential latency bottleneck. However, does such efficiency gains translate into effective agentic behavior? In this work, we present a comprehensive evaluation of dLLMs (e.g., LLaDA, Dream) across two distinct agentic paradigms: Embodied Agents (requiring long-horizon planning) and Tool-Calling Agents (requiring precise formatting). Contrary to the efficiency hype, our results on Agentboard and BFCL reveal a "bitter lesson": current dLLMs fail to serve as reliable agentic backbones, frequently leading to systematically failure. (1) In Embodied settings, dLLMs suffer repeated attempts, failing to branch under temporal feedback. (2) In Tool-Calling settings, dLLMs fail to maintain symbolic precision (e.g. strict JSON schemas) under diffusion noise. To assess the potential of dLLMs in agentic workflows, we introduce DiffuAgent, a multi-agent evaluation framework that integrates dLLMs as plug-and-play cognitive cores. Our analysis shows that dLLMs are effective in non-causal roles (e.g., memory summarization and tool selection) but require the incorporation of causal, precise, and logically grounded reasoning mechanisms into the denoising process to be viable for agentic tasks. 

---
# Bridging the Knowledge-Action Gap by Evaluating LLMs in Dynamic Dental Clinical Scenarios 

**Authors**: Hongyang Ma, Tiantian Gu, Huaiyuan Sun, Huilin Zhu, Yongxin Wang, Jie Li, Wubin Sun, Zeliang Lian, Yinghong Zhou, Yi Gao, Shirui Wang, Zhihui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.12974)  

**Abstract**: The transition of Large Language Models (LLMs) from passive knowledge retrievers to autonomous clinical agents demands a shift in evaluation-from static accuracy to dynamic behavioral reliability. To explore this boundary in dentistry, a domain where high-quality AI advice uniquely empowers patient-participatory decision-making, we present the Standardized Clinical Management & Performance Evaluation (SCMPE) benchmark, which comprehensively assesses performance from knowledge-oriented evaluations (static objective tasks) to workflow-based simulations (multi-turn simulated patient interactions). Our analysis reveals that while models demonstrate high proficiency in static objective tasks, their performance precipitates in dynamic clinical dialogues, identifying that the primary bottleneck lies not in knowledge retention, but in the critical challenges of active information gathering and dynamic state tracking. Mapping "Guideline Adherence" versus "Decision Quality" reveals a prevalent "High Efficacy, Low Safety" risk in general models. Furthermore, we quantify the impact of Retrieval-Augmented Generation (RAG). While RAG mitigates hallucinations in static tasks, its efficacy in dynamic workflows is limited and heterogeneous, sometimes causing degradation. This underscores that external knowledge alone cannot bridge the reasoning gap without domain-adaptive pre-training. This study empirically charts the capability boundaries of dental LLMs, providing a roadmap for bridging the gap between standardized knowledge and safe, autonomous clinical practice. 

---
# Pardon? Evaluating Conversational Repair in Large Audio-Language Models 

**Authors**: Shuanghong Huang, Jinlei Xu, Youchao Zhou, Yanghao Zhou, Xuan Zhao, Chong Feng, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.12973)  

**Abstract**: Large Audio-Language Models (LALMs) have demonstrated strong performance in spoken question answering (QA), with existing evaluations primarily focusing on answer accuracy and robustness to acoustic perturbations. However, such evaluations implicitly assume that spoken inputs remain semantically answerable, an assumption that often fails in real-world interaction when essential information is missing. In this work, we introduce a repair-aware evaluation setting that explicitly distinguishes between answerable and unanswerable audio inputs. We define answerability as a property of the input itself and construct paired evaluation conditions using a semantic-acoustic masking protocol. Based on this setting, we propose the Evaluability Awareness and Repair (EAR) score, a non-compensatory metric that jointly evaluates task competence under answerable conditions and repair behavior under unanswerable conditions. Experiments on two spoken QA benchmarks across diverse LALMs reveal a consistent gap between answer accuracy and conversational reliability: while many models perform well when inputs are answerable, most fail to recognize semantic unanswerability and initiate appropriate conversational repair. These findings expose a limitation of prevailing accuracy-centric evaluation practices and motivate reliability assessments that treat unanswerable inputs as cues for repair and continued interaction. 

---
# Trustworthy Data-driven Chronological Age Estimation from Panoramic Dental Images 

**Authors**: Ainhoa Vivel-Couso, Nicolás Vila-Blanco, María J. Carreira, Alberto Bugarín-Diz, Inmaculada Tomás, Jose M. Alonso-Moral  

**Link**: [PDF](https://arxiv.org/pdf/2601.12960)  

**Abstract**: Integrating deep learning into healthcare enables personalized care but raises trust issues due to model opacity. To improve transparency, we propose a system for dental age estimation from panoramic images that combines an opaque and a transparent method within a natural language generation (NLG) module. This module produces clinician-friendly textual explanations about the age estimations, designed with dental experts through a rule-based approach. Following the best practices in the field, the quality of the generated explanations was manually validated by dental experts using a questionnaire. The results showed a strong performance, since the experts rated 4.77+/-0.12 (out of 5) on average across the five dimensions considered. We also performed a trustworthy self-assessment procedure following the ALTAI checklist, in which it scored 4.40+/-0.27 (out of 5) across seven dimensions of the AI Trustworthiness Assessment List. 

---
# A Component-Based Survey of Interactions between Large Language Models and Multi-Armed Bandits 

**Authors**: Miao Xie, Siguang Chen, Chunli Lv  

**Link**: [PDF](https://arxiv.org/pdf/2601.12945)  

**Abstract**: Large language models (LLMs) have become powerful and widely used systems for language understanding and generation, while multi-armed bandit (MAB) algorithms provide a principled framework for adaptive decision-making under uncertainty. This survey explores the potential at the intersection of these two fields. As we know, it is the first survey to systematically review the bidirectional interaction between large language models and multi-armed bandits at the component level. We highlight the bidirectional benefits: MAB algorithms address critical LLM challenges, spanning from pre-training to retrieval-augmented generation (RAG) and personalization. Conversely, LLMs enhance MAB systems by redefining core components such as arm definition and environment modeling, thereby improving decision-making in sequential tasks. We analyze existing LLM-enhanced bandit systems and bandit-enhanced LLM systems, providing insights into their design, methodologies, and performance. Key challenges and representative findings are identified to help guide future research. An accompanying GitHub repository that indexes relevant literature is available at this https URL. 

---
# Injecting Knowledge from Social Science Journals to Improve Indonesian Cultural Understanding by LLMs 

**Authors**: Adimulya Kartiyasa, Bao Gia Cao, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.12921)  

**Abstract**: Recently there have been intensifying efforts to improve the understanding of Indonesian cultures by large language models (LLMs). An attractive source of cultural knowledge that has been largely overlooked is local journals of social science, which likely contain substantial cultural studies from a native perspective. We present a novel text dataset of journal article passages, created from 151 open-source Indonesian social science journals, called IndoSoSci. We demonstrate an effective recipe for injecting Indonesian cultural knowledge therein into LLMs: extracting the facts related to Indonesian culture, and apply retrieval-augmented generation (RAG) with LLM-generated hypothetical documents as queries during retrieval. The proposed recipe yields strong performance gains over several strong baselines on the IndoCulture benchmark. Additionally, by combining IndoSoSci with Indonesian Wikipedia, we set a new state-of-the-art accuracy on the IndoCulture benchmark. 

---
# SciCoQA: Quality Assurance for Scientific Paper--Code Alignment 

**Authors**: Tim Baumgärtner, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2601.12910)  

**Abstract**: We present SciCoQA, a dataset for detecting discrepancies between scientific publications and their codebases to ensure faithful implementations. We construct SciCoQA from GitHub issues and reproducibility papers, and to scale our dataset, we propose a synthetic data generation method for constructing paper-code discrepancies. We analyze the paper-code discrepancies in detail and propose discrepancy types and categories to better understand the occurring mismatches. In total, our dataset consists of 611 paper-code discrepancies (81 real, 530 synthetic), spanning diverse computational science disciplines, including AI, Physics, Quantitative Biology, and others. Our evaluation of 21 LLMs highlights the difficulty of SciCoQA, particularly for instances involving omitted paper details, long-context inputs, and data outside the models' pre-training corpus. The best performing model in our evaluation, GPT-5, can only detect 45.7\% of real-world paper-code discrepancies. 

---
# Gated Differentiable Working Memory for Long-Context Language Modeling 

**Authors**: Lingrui Mei, Shenghua Liu, Yiwei Wang, Yuyao Ge, Baolong Bi, Jiayu Yao, Jun Wan, Ziling Yin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.12906)  

**Abstract**: Long contexts challenge transformers: attention scores dilute across thousands of tokens, critical information is often lost in the middle, and models struggle to adapt to novel patterns at inference time. Recent work on test-time adaptation addresses this by maintaining a form of working memory -- transient parameters updated on the current context -- but existing approaches rely on uniform write policies that waste computation on low-utility regions and suffer from high gradient variance across semantically heterogeneous contexts. In this work, we reframe test-time adaptation as a budget-constrained memory consolidation problem, focusing on which parts of the context should be consolidated into working memory under limited computation. We propose Gdwm (Gated Differentiable Working Memory), a framework that introduces a write controller to gate the consolidation process. The controller estimates Contextual Utility, an information-theoretic measure of long-range contextual dependence, and allocates gradient steps accordingly while maintaining global coverage. Experiments on ZeroSCROLLS and LongBench v2 demonstrate that Gdwm achieves comparable or superior performance with 4$\times$ fewer gradient steps than uniform baselines, establishing a new efficiency-performance Pareto frontier for test-time adaptation. 

---
# From Prefix Cache to Fusion RAG Cache: Accelerating LLM Inference in Retrieval-Augmented Generation 

**Authors**: Jiahao Wang, Weiyu Xie, Mingxing Zhang, Boxing Zhang, Jianwei Dong, Yuening Zhu, Chen Lin, Jinqi Tang, Yaochen Han, Zhiyuan Ai, Xianglin Chen, Yongwei Wu, Congfeng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.12904)  

**Abstract**: Retrieval-Augmented Generation enhances Large Language Models by integrating external knowledge, which reduces hallucinations but increases prompt length. This increase leads to higher computational costs and longer Time to First Token (TTFT). To mitigate this issue, existing solutions aim to reuse the preprocessed KV cache of each retrieved chunk to accelerate RAG. However, the lack of cross-chunk contextual information leads to a significant drop in generation quality, leaving the potential benefits of KV cache reuse largely unfulfilled. The challenge lies in how to reuse the precomputed KV cache of chunks while preserving generation quality. We propose FusionRAG, a novel inference framework that optimizes both the preprocessing and reprocessing stages of RAG. In the offline preprocessing stage, we embed information from other related text chunks into each chunk, while in the online reprocessing stage, we recompute the KV cache for tokens that the model focuses on. As a result, we achieve a better trade-off between generation quality and efficiency. According to our experiments, FusionRAG significantly improves generation quality at the same recomputation ratio compared to previous state-of-the-art solutions. By recomputing fewer than 15% of the tokens, FusionRAG achieves up to 70% higher normalized F1 scores than baselines and reduces TTFT by 2.66x-9.39x compared to Full Attention. 

---
# Race, Ethnicity and Their Implication on Bias in Large Language Models 

**Authors**: Shiyue Hu, Ruizhe Li, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2601.12868)  

**Abstract**: Large language models (LLMs) increasingly operate in high-stakes settings including healthcare and medicine, where demographic attributes such as race and ethnicity may be explicitly stated or implicitly inferred from text. However, existing studies primarily document outcome-level disparities, offering limited insight into internal mechanisms underlying these effects. We present a mechanistic study of how race and ethnicity are represented and operationalized within LLMs. Using two publicly available datasets spanning toxicity-related generation and clinical narrative understanding tasks, we analyze three open-source models with a reproducible interpretability pipeline combining probing, neuron-level attribution, and targeted intervention. We find that demographic information is distributed across internal units with substantial cross-model variation. Although some units encode sensitive or stereotype-related associations from pretraining, identical demographic cues can induce qualitatively different behaviors. Interventions suppressing such neurons reduce bias but leave substantial residual effects, suggesting behavioral rather than representational change and motivating more systematic mitigation. 

---
# Rapport du Projet de Recherche TRAIMA 

**Authors**: Julie Rançon, Jean-François Cerisier, Emilie Remond, Aurélien Nguyen, Andrew Peterson, Ladjel Bellatreche  

**Link**: [PDF](https://arxiv.org/pdf/2601.12844)  

**Abstract**: The TRAIMA project (TRaitement Automatique des Interactions Multimodales en Apprentissage), conducted between March 2019 and June 2020, investigates the potential of automatic processing of multimodal interactions in educational settings. The project addresses a central methodological challenge in educational and interactional research: the analysis of verbal, paraverbal, and non-verbal data is currently carried out manually, making it extremely time-consuming and difficult to scale. TRAIMA explores how machine learning approaches could contribute to the categorisation and classification of such interactions. The project focuses specifically on explanatory and collaborative sequences occurring in classroom interactions, particularly in French as a Foreign Language (FLE) and French as a First Language (FLM) contexts. These sequences are analysed as inherently multimodal phenomena, combining spoken language with prosody, gestures, posture, gaze, and spatial positioning. A key theoretical contribution of the project is the precise linguistic and interactional definition of explanatory discourse as a tripartite sequence (opening, explanatory core, closure), drawing on discourse analysis and interactional linguistics. A substantial part of the research is devoted to the methodological foundations of transcription, which constitute a critical bottleneck for any form of automation. The report provides a detailed state of the art of existing transcription conventions (ICOR, Mondada, GARS, VALIBEL, Ferr{é}), highlighting their respective strengths and limitations when applied to multimodal classroom data. Through comparative analyses of manually transcribed sequences, the project demonstrates the inevitable variability and interpretative dimension of transcription practices, depending on theoretical positioning and analytical goals. Empirical work is based on several corpora, notably the INTER-EXPLIC corpus (approximately 30 hours of classroom interaction) and the EXPLIC-LEXIC corpus, which serve both as testing grounds for manual annotation and as reference datasets for future automation. Particular attention is paid to teacher gestures (kin{é}sic and proxemic resources), prosodic features, and their functional role in meaning construction and learner comprehension. The project also highlights the strategic role of the Techn{é}LAB platform, which provides advanced multimodal data capture (multi-camera video, synchronized audio, eye-tracking, digital interaction traces) and constitutes both a research infrastructure and a test environment for the development of automated tools. In conclusion, TRAIMA does not aim to deliver a fully operational automated system, but rather to establish a rigorous methodological framework for the automatic processing of multimodal pedagogical interactions. The project identifies transcription conventions, annotation categories, and analytical units that are compatible with machine learning approaches, while emphasizing the need for theoretical explicitness and researcher reflexivity. TRAIMA thus lays the groundwork for future interdisciplinary research at the intersection of didactics, discourse analysis, multimodality, and artificial intelligence in education. 

---
# Multimodal Multi-Agent Empowered Legal Judgment Prediction 

**Authors**: Zhaolu Kang, Junhao Gong, Qingxi Chen, Hao Zhang, Jiaxin Liu, Rong Fu, Zhiyuan Feng, Yuan Wang, Simon Fong, Kaiyue Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.12815)  

**Abstract**: Legal Judgment Prediction (LJP) aims to predict the outcomes of legal cases based on factual descriptions, serving as a fundamental task to advance the development of legal systems. Traditional methods often rely on statistical analyses or role-based simulations but face challenges with multiple allegations, diverse evidence, and lack adaptability. In this paper, we introduce JurisMMA, a novel framework for LJP that effectively decomposes trial tasks, standardizes processes, and organizes them into distinct stages. Furthermore, we build JurisMM, a large dataset with over 100,000 recent Chinese judicial records, including both text and multimodal video-text data, enabling comprehensive evaluation. Experiments on JurisMM and the benchmark LawBench validate our framework's effectiveness. These results indicate that our framework is effective not only for LJP but also for a broader range of legal applications, offering new perspectives for the development of future legal methods and datasets. 

---
# Do Clinical Question Answering Systems Really Need Specialised Medical Fine Tuning? 

**Authors**: Sushant Kumar Ray, Gautam Siddharth Kashyap, Sahil Tripathi, Nipun Joshi, Vijay Govindarajan, Rafiq Ali, Jiechao Gao, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2601.12812)  

**Abstract**: Clinical Question-Answering (CQA) industry systems are increasingly rely on Large Language Models (LLMs), yet their deployment is often guided by the assumption that domain-specific fine-tuning is essential. Although specialised medical LLMs such as BioBERT, BioGPT, and PubMedBERT remain popular, they face practical limitations including narrow coverage, high retraining costs, and limited adaptability. Efforts based on Supervised Fine-Tuning (SFT) have attempted to address these assumptions but continue to reinforce what we term the SPECIALISATION FALLACY-the belief that specialised medical LLMs are inherently superior for CQA. To address this assumption, we introduce MEDASSESS-X, a deployment-industry-oriented CQA framework that applies alignment at inference time rather than through SFT. MEDASSESS-X uses lightweight steering vectors to guide model activations toward medically consistent reasoning without updating model weights or requiring domain-specific retraining. This inference-time alignment layer stabilises CQA performance across both general-purpose and specialised medical LLMs, thereby resolving the SPECIALISATION FALLACY. Empirically, MEDASSESS-X delivers consistent gains across all LLM families, improving Accuracy by up to +6%, Factual Consistency by +7%, and reducing Safety Error Rate by as much as 50%. 

---
# Who Does This Name Remind You of? Nationality Prediction via Large Language Model Associative Memory 

**Authors**: Keito Inoshita  

**Link**: [PDF](https://arxiv.org/pdf/2601.12771)  

**Abstract**: Large language models (LLMs) possess extensive world knowledge, yet methods for effectively eliciting this knowledge remain underexplored. Nationality and region prediction tasks require understanding of not only linguistic features but also cultural and historical background, making LLM world knowledge particularly valuable. However, conventional LLM prompting methods rely on direct reasoning approaches, which have limitations in applying abstract linguistic rules. We propose LLM Associative Memory Agents (LAMA), a novel framework that leverages LLM world knowledge as associative memory. Rather than directly inferring nationality from names, LAMA recalls famous individuals with the same name and aggregates their nationalities through indirect reasoning. A dual-agent architecture comprising a Person Agent and a Media Agent, specialized in different knowledge domains, recalls famous individuals in parallel, generating Top-1 predictions through voting and Top-K predictions through conditional completion. On a 99-country nationality prediction task, LAMA achieved 0.817 accuracy, substantially outperforming conventional LLM prompting methods and neural models. Our experiments reveal that LLMs exhibit higher reliability in recalling concrete examples than in abstract reasoning, that recall-based approaches are robust to low-frequency nationalities independent of data frequency distributions, and that the dual-agent architecture functions complementarily to produce synergistic effects. These results demonstrate the effectiveness of a new multi-agent system that retrieves and aggregates LLM knowledge rather than prompting reasoning. 

---
# VISPA: Pluralistic Alignment via Automatic Value Selection and Activation 

**Authors**: Shenyan Zheng, Jiayou Zhong, Anudeex Shetty, Heng Ji, Preslav Nakov, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2601.12758)  

**Abstract**: As large language models are increasingly used in high-stakes domains, it is essential that their outputs reflect not average} human preference, rather range of varying perspectives. Achieving such pluralism, however, remains challenging. Existing approaches consider limited values or rely on prompt-level interventions, lacking value control and representation. To address this, we introduce VISPA, a training-free pluralistic alignment framework, that enables direct control over value expression by dynamic selection and internal model activation steering. Across extensive empirical studies spanning multiple models and evaluation settings, we show VISPA is performant across all pluralistic alignment modes in healthcare and beyond. Further analysis reveals VISPA is adaptable with different steering initiations, model, and/or values. These results suggest that pluralistic alignment can be achieved through internal activation mechanisms, offering a scalable path toward language models that serves all. 

---
# Towards Robust Process Reward Modeling via Noise-aware Learning 

**Authors**: Bin Xie, Bingbing Xu, Xueyun Tian, Yilin Chen, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.12748)  

**Abstract**: Process Reward Models (PRMs) have achieved strong results in complex reasoning, but are bottlenecked by costly process-level supervision. A widely used alternative, Monte Carlo Estimation (MCE), defines process rewards as the probability that a policy model reaches the correct final answer from a given reasoning step. However, step correctness is an intrinsic property of the reasoning trajectory, and should be invariant to policy choice. Our empirical findings show that MCE producing policy-dependent rewards that induce label noise, including false positives that reward incorrect steps and false negatives that penalize correct ones. To address above challenges, we propose a two-stage framework to mitigate noisy supervision. In the labeling stage, we introduce a reflection-aware label correction mechanism that uses a large language model (LLM) as a judge to detect reflection and self-correction behaviors related to the current reasoning step, thereby suppressing overestimated rewards. In the training stage, we further propose a \underline{\textbf{N}}oise-\underline{\textbf{A}}ware \underline{\textbf{I}}terative \underline{\textbf{T}}raining framework that enables the PRM to progressively refine noisy labels based on its own confidence. Extensive Experiments show that our method substantially improves step-level correctness discrimination, achieving up to a 27\% absolute gain in average F1 over PRMs trained with noisy supervision. 

---
# A Shared Geometry of Difficulty in Multilingual Language Models 

**Authors**: Stefano Civelli, Pietro Bernardelle, Nicolò Brunello, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2601.12731)  

**Abstract**: Predicting problem-difficulty in large language models (LLMs) refers to estimating how difficult a task is according to the model itself, typically by training linear probes on its internal representations. In this work, we study the multilingual geometry of problem-difficulty in LLMs by training linear probes using the AMC subset of the Easy2Hard benchmark, translated into 21 languages. We found that difficulty-related signals emerge at two distinct stages of the model internals, corresponding to shallow (early-layers) and deep (later-layers) internal representations, that exhibit functionally different behaviors. Probes trained on deep representations achieve high accuracy when evaluated on the same language but exhibit poor cross-lingual generalization. In contrast, probes trained on shallow representations generalize substantially better across languages, despite achieving lower within-language performance. Together, these results suggest that LLMs first form a language-agnostic representation of problem difficulty, which subsequently becomes language-specific. This closely aligns with existing findings in LLM interpretability showing that models tend to operate in an abstract conceptual space before producing language-specific outputs. We demonstrate that this two-stage representational process extends beyond semantic content to high-level meta-cognitive properties such as problem-difficulty estimation. 

---
# A Two-Stage GPU Kernel Tuner Combining Semantic Refactoring and Search-Based Optimization 

**Authors**: Qiuyi Qu, Yicheng Sui, Yufei Sun, Rui Chen, Xiaofei Zhang, Yuzhi Zhang, Haofeng Wang, Ge Lan, Ning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.12698)  

**Abstract**: GPU code optimization is a key performance bottleneck for HPC workloads as well as large-model training and inference. Although compiler optimizations and hand-written kernels can partially alleviate this issue, achieving near-hardware-limit performance still relies heavily on manual code refactoring and parameter tuning. Recent progress in LLM-agent-based kernel generation and optimization has been reported, yet many approaches primarily focus on direct code rewriting, where parameter choices are often implicit and hard to control, or require human intervention, leading to unstable performance gains. This paper introduces a template-based rewriting layer on top of an agent-driven iterative loop: kernels are semantically refactored into explicitly parameterizable templates, and template parameters are then optimized via search-based autotuning, yielding more stable and higher-quality speedups. Experiments on a set of real-world kernels demonstrate speedups exceeding 3x in the best case. We extract representative CUDA kernels from SGLang as evaluation targets; the proposed agentic tuner iteratively performs templating, testing, analysis, and planning, and leverages profiling feedback to execute constrained parameter search under hardware resource limits. Compared to agent-only direct rewriting, the template-plus-search design significantly reduces the randomness of iterative optimization, making the process more interpretable and enabling a more systematic approach toward high-performance configurations. The proposed method can be further extended to OpenCL, HIP, and other backends to deliver automated performance optimization for real production workloads. 

---
# UbuntuGuard: A Culturally-Grounded Policy Benchmark for Equitable AI Safety in African Languages 

**Authors**: Tassallah Abdullahi, Macton Mgonzo, Mardiyyah Oduwole, Paul Okewunmi, Abraham Owodunni, Ritambhara Singh, Carsten Eickhoff  

**Link**: [PDF](https://arxiv.org/pdf/2601.12696)  

**Abstract**: Current guardian models are predominantly Western-centric and optimized for high-resource languages, leaving low-resource African languages vulnerable to evolving harms, cross-lingual safety failures, and cultural misalignment. Moreover, most guardian models rely on rigid, predefined safety categories that fail to generalize across diverse linguistic and sociocultural contexts. Robust safety, therefore, requires flexible, runtime-enforceable policies and benchmarks that reflect local norms, harm scenarios, and cultural expectations. We introduce UbuntuGuard, the first African policy-based safety benchmark built from adversarial queries authored by 155 domain experts across sensitive fields, including healthcare. From these expert-crafted queries, we derive context-specific safety policies and reference responses that capture culturally grounded risk signals, enabling policy-aligned evaluation of guardian models. We evaluate 13 models, comprising six general-purpose LLMs and seven guardian models across three distinct variants: static, dynamic, and multilingual. Our findings reveal that existing English-centric benchmarks overestimate real-world multilingual safety, cross-lingual transfer provides partial but insufficient coverage, and dynamic models, while better equipped to leverage policies at inference time, still struggle to fully localize African-language contexts. These findings highlight the urgent need for multilingual, culturally grounded safety benchmarks to enable the development of reliable and equitable guardian models for low-resource languages. Our code can be found online.\footnote{Code repository available at this https URL. 

---
# Augmenting Question Answering with A Hybrid RAG Approach 

**Authors**: Tianyi Yang, Nashrah Haque, Vaishnave Jonnalagadda, Yuya Jeremy Ong, Zhehui Chen, Yanzhao Wu, Lei Yu, Divyesh Jadav, Wenqi Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.12658)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful technique for enhancing the quality of responses in Question-Answering (QA) tasks. However, existing approaches often struggle with retrieving contextually relevant information, leading to incomplete or suboptimal answers. In this paper, we introduce Structured-Semantic RAG (SSRAG), a hybrid architecture that enhances QA quality by integrating query augmentation, agentic routing, and a structured retrieval mechanism combining vector and graph based techniques with context unification. By refining retrieval processes and improving contextual grounding, our approach improves both answer accuracy and informativeness. We conduct extensive evaluations on three popular QA datasets, TruthfulQA, SQuAD and WikiQA, across five Large Language Models (LLMs), demonstrating that our proposed approach consistently improves response quality over standard RAG implementations. 

---
# Intelligent Documentation in Medical Education: Can AI Replace Manual Case Logging? 

**Authors**: Nafiz Imtiaz Khan, Kylie Cleland, Vladimir Filkov, Roger Eric Goldman  

**Link**: [PDF](https://arxiv.org/pdf/2601.12648)  

**Abstract**: Procedural case logs are a core requirement in radiology training, yet they are time-consuming to complete and prone to inconsistency when authored manually. This study investigates whether large language models (LLMs) can automate procedural case log documentation directly from free-text radiology reports. We evaluate multiple local and commercial LLMs under instruction-based and chain-of-thought prompting to extract structured procedural information from 414 curated interventional radiology reports authored by nine residents between 2018 and 2024. Model performance is assessed using sensitivity, specificity, and F1-score, alongside inference latency and token efficiency to estimate operational cost. Results show that both local and commercial models achieve strong extraction performance, with best F1-scores approaching 0.87, while exhibiting different trade-offs between speed and cost. Automation using LLMs has the potential to substantially reduce clerical burden for trainees and improve consistency in case logging. These findings demonstrate the feasibility of AI-assisted documentation in medical education and highlight the need for further validation across institutions and clinical workflows. 

---
# Objective Matters: Fine-Tuning Objectives Shape Safety, Robustness, and Persona Drift 

**Authors**: Daniel Vennemeyer, Punya Syon Pandey, Phan Anh Duong, Michael Umeokoli, Samuel Ratnam  

**Link**: [PDF](https://arxiv.org/pdf/2601.12639)  

**Abstract**: Fine-tuning LLMs on benign data can still degrade alignment and adversarial robustness, yet direct analysis of the role of fine-tuning objectives in shaping these safety outcomes remain limited. We present a controlled comparison of six fine-tuning objectives -- Supervised Fine-Tuning, Direct Preference Optimization, Conditional Fine-Tuning, Inoculation Prompting, Odds Ratio Preference Optimization, and KL-regularized fine-tuning -- holding data, domain, architecture, and optimization fixed. Across closed-form reasoning and open-ended generation tasks, we find that objective choice induces systematic, scale-dependent shifts along the safety-capability frontier. At small training budgets, robustness is similar across objectives but capability differs. At larger budgets, objectives diverge sharply: supervised and preference-based tuning tightly couple capability gains to increased adversarial vulnerability and persona drift, while objectives that constrain learning signals -- especially ORPO and KL-regularization -- substantially mitigate both. Fine-tuning objectives therefore matter little for safety at small scales but become a primary driver of adversarial robustness and latent persona stability as training scale increases. 

---
# BioPulse-QA: A Dynamic Biomedical Question-Answering Benchmark for Evaluating Factuality, Robustness, and Bias in Large Language Models 

**Authors**: Kriti Bhattarai, Vipina K. Keloth, Donald Wright, Andrew Loza, Yang Ren, Hua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12632)  

**Abstract**: Objective: Large language models (LLMs) are increasingly applied in biomedical settings, and existing benchmark datasets have played an important role in supporting model development and evaluation. However, these benchmarks often have limitations. Many rely on static or outdated datasets that fail to capture the dynamic, context-rich, and high-stakes nature of biomedical knowledge. They also carry increasing risk of data leakage due to overlap with model pretraining corpora and often overlook critical dimensions such as robustness to linguistic variation and potential demographic biases.
Materials and Methods: To address these gaps, we introduce BioPulse-QA, a benchmark that evaluates LLMs on answering questions from newly published biomedical documents including drug labels, trial protocols, and clinical guidelines. BioPulse-QA includes 2,280 expert-verified question answering (QA) pairs and perturbed variants, covering both extractive and abstractive formats. We evaluate four LLMs - GPT-4o, GPT-o1, Gemini-2.0-Flash, and LLaMA-3.1 8B Instruct - released prior to the publication dates of the benchmark documents.
Results: GPT-o1 achieves the highest relaxed F1 score (0.92), followed by Gemini-2.0-Flash (0.90) on drug labels. Clinical trials are the most challenging source, with extractive F1 scores as low as 0.36.
Discussion and Conclusion: Performance differences are larger for paraphrasing than for typographical errors, while bias testing shows negligible differences. BioPulse-QA provides a scalable and clinically relevant framework for evaluating biomedical LLMs. 

---
# Disagreement as Data: Reasoning Trace Analytics in Multi-Agent Systems 

**Authors**: Elham Tajik, Conrad Borchers, Bahar Shahrokhian, Sebastian Simon, Ali Keramati, Sonika Pal, Sreecharan Sankaranarayanan  

**Link**: [PDF](https://arxiv.org/pdf/2601.12618)  

**Abstract**: Learning analytics researchers often analyze qualitative student data such as coded annotations or interview transcripts to understand learning processes. With the rise of generative AI, fully automated and human-AI workflows have emerged as promising methods for analysis. However, methodological standards to guide such workflows remain limited. In this study, we propose that reasoning traces generated by large language model (LLM) agents, especially within multi-agent systems, constitute a novel and rich form of process data to enhance interpretive practices in qualitative coding. We apply cosine similarity to LLM reasoning traces to systematically detect, quantify, and interpret disagreements among agents, reframing disagreement as a meaningful analytic signal. Analyzing nearly 10,000 instances of agent pairs coding human tutoring dialog segments, we show that LLM agents' semantic reasoning similarity robustly differentiates consensus from disagreement and correlates with human coding reliability. Qualitative analysis guided by this metric reveals nuanced instructional sub-functions within codes and opportunities for conceptual codebook refinement. By integrating quantitative similarity metrics with qualitative review, our method has the potential to improve and accelerate establishing inter-rater reliability during coding by surfacing interpretive ambiguity, especially when LLMs collaborate with humans. We discuss how reasoning-trace disagreements represent a valuable new class of analytic signals advancing methodological rigor and interpretive depth in educational research. 

---
# A Cloud-based Multi-Agentic Workflow for Science 

**Authors**: Anurag Acharya, Timothy Vega, Rizwan A. Ashraf, Anshu Sharma, Derek Parker, Robert Rallo  

**Link**: [PDF](https://arxiv.org/pdf/2601.12607)  

**Abstract**: As Large Language Models (LLMs) become ubiquitous across various scientific domains, their lack of ability to perform complex tasks like running simulations or to make complex decisions limits their utility. LLM-based agents bridge this gap due to their ability to call external resources and tools and thus are now rapidly gaining popularity. However, coming up with a workflow that can balance the models, cloud providers, and external resources is very challenging, making implementing an agentic system more of a hindrance than a help. In this work, we present a domain-agnostic, model-independent workflow for an agentic framework that can act as a scientific assistant while being run entirely on cloud. Built with a supervisor agent marshaling an array of agents with individual capabilities, our framework brings together straightforward tasks like literature review and data analysis with more complex ones like simulation runs. We describe the framework here in full, including a proof-of-concept system we built to accelerate the study of Catalysts, which is highly important in the field of Chemistry and Material Science. We report the cost to operate and use this framework, including the breakdown of the cost by services use. We also evaluate our system on a custom-curated synthetic benchmark and a popular Chemistry benchmark, and also perform expert validation of the system. The results show that our system is able to route the task to the correct agent 90% of the time and successfully complete the assigned task 97.5% of the time for the synthetic tasks and 91% of the time for real-world tasks, while still achieving better or comparable accuracy to most frontier models, showing that this is a viable framework for other scientific domains to replicate. 

---
# Evaluating Contextually Mediated Factual Recall in Multilingual Large Language Models 

**Authors**: Yihong Liu, Bingyu Xiong, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2601.12555)  

**Abstract**: Large language models (LLMs) can recall a wide range of factual knowledge across languages. However, existing factual recall evaluations primarily assess fact retrieval in isolation, where the queried entity is explicitly named and the fact is requested directly. In natural language use, facts are often accessed through context, where the relevant entity is introduced only indirectly. In this work, we study contextually mediated factual recall, asking whether LLMs can reliably retrieve factual knowledge when the target entity is embedded in a naturalistic context rather than queried explicitly, across languages. We construct controlled prompts that preserve the underlying fact while introducing referential mediation through contextual sentences. To disentangle contextual effects from name-specific associations, we further compare performance using synthetic names and real names across languages. Evaluating multiple model families in five languages, we find that contextual mediation consistently degrades factual recall, with substantial variation across relations. Larger models are more robust to contextual mediation, exhibiting a reduced performance gap relative to direct queries, while the effect of real names and name origin is mixed and unsystematic. These findings highlight a gap between isolated factual recall and context-dependent language understanding in multilingual LLMs. 

---
# Benchmarking Concept-Spilling Across Languages in LLMs 

**Authors**: Ilia Badanin, Daniil Dzenhaliou, Imanol Schlag  

**Link**: [PDF](https://arxiv.org/pdf/2601.12549)  

**Abstract**: Multilingual Large Language Models (LLMs) exhibit remarkable cross-lingual abilities, yet often exhibit a systematic bias toward the representations from other languages, resulting in semantic interference when generating content in non-English languages$-$a phenomenon we define as language spilling. This paper presents a novel comparative framework for evaluating multilingual semantic robustness by systematically measuring how models handle polysemous words across languages. Our methodology provides a relative measure of model performance: when required to generate exactly five meanings, both strong and weak models may resort to meanings from dominant languages, but semantically stronger models do so later in the generation sequence, producing more true meanings from the target language before failing, while weaker models resort to dominant-language meanings earlier in the sequence. We evaluate a diverse set of open and closed multilingual LLMs using a structured meaning generation task across nine languages, employing a carefully curated benchmark of 100 high-polysemy English words. Our findings reveal significant variation in semantic robustness across both models and languages, providing a principled ranking system for model comparison without requiring definitive causal attribution of error sources. We contribute both a scalable comparative benchmark for multilingual semantic evaluation and a rigorous validation pipeline$-$critical tools for developing more linguistically balanced AI systems. 

---
# Improving Low-Resource Machine Translation via Round-Trip Reinforcement Learning 

**Authors**: Ahmed Attia, Alham Fikri  

**Link**: [PDF](https://arxiv.org/pdf/2601.12535)  

**Abstract**: Low-resource machine translation (MT) has gained increasing attention as parallel data from low-resource language communities is collected, but many potential methods for improving low-resource MT remain unexplored. We investigate a self-supervised reinforcement-learning-based fine-tuning for translation in low-resource settings using round-trip bootstrapping with the No Language Left Behind (NLLB) family of models. Our approach translates English into a target low-resource language and then back into English, using a combination of chrF++ and BLEU as the reward function on the reconstructed English sentences. Using the NLLB-MD dataset, we evaluate both the 600M and 1.3B parameter NLLB models and observe consistent improvements for the following languages: Central Aymara, Friulian, Wolof and Russian. Qualitative inspection of translation outputs indicates increased fluency and semantic fidelity. We argue that our method can further benefit from scale, enabling models to increasingly leverage their pretrained knowledge and continue self-improving. 

---
# DoPE: Decoy Oriented Perturbation Encapsulation Human-Readable, AI-Hostile Documents for Academic Integrity 

**Authors**: Ashish Raj Shekhar, Shiven Agarwal, Priyanuj Bordoloi, Yash Shah, Tejas Anvekar, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2601.12505)  

**Abstract**: Multimodal Large Language Models (MLLMs) can directly consume exam documents, threatening conventional assessments and academic integrity. We present DoPE (Decoy-Oriented Perturbation Encapsulation), a document-layer defense framework that embeds semantic decoys into PDF/HTML assessments to exploit render-parse discrepancies in MLLM pipelines. By instrumenting exams at authoring time, DoPE provides model-agnostic prevention (stop or confound automated solving) and detection (flag blind AI reliance) without relying on conventional one-shot classifiers. We formalize prevention and detection tasks, and introduce FewSoRT-Q, an LLM-guided pipeline that generates question-level semantic decoys and FewSoRT-D to encapsulate them into watermarked documents. We evaluate on Integrity-Bench, a novel benchmark of 1826 exams (PDF+HTML) derived from public QA datasets and OpenCourseWare. Against black-box MLLMs from OpenAI and Anthropic, DoPE yields strong empirical gains: a 91.4% detection rate at an 8.7% false-positive rate using an LLM-as-Judge verifier, and prevents successful completion or induces decoy-aligned failures in 96.3% of attempts. We release Integrity-Bench, our toolkit, and evaluation code to enable reproducible study of document-layer defenses for academic integrity. 

---
# Capability-Aware Early-Stage Research Idea Evaluation 

**Authors**: Renlong Jie, Chen Chu, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.12473)  

**Abstract**: Predicting the outcomes of research ideas at their conceptual stage (i.e. before significant resources are committed) holds great potential for optimizing scientific resource allocation and research planning. While existing methods rely heavily on finished manuscripts or peer reviews, we propose a novel capability-aware framework that predicts paper acceptance and ratings using only author information and research ideas, without requiring full text or experimental results. Our approach integrates author information, (inferred) capability presentation, and research ideas through a three-way transformer architecture with flexible fusion mechanisms. We also introduce a two-stage architecture for learning the capability representation given the author information and idea. Experiments show that our method significantly outperform the single-way models by finetuning bert-base and bert-large, and the capability predicting significantly increase the predictive accuracy of the final model. The proposed method can be applied in both early-stage research outcome prediction and scientific resource allocation. 

---
# Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty 

**Authors**: Sravanthi Machcha, Sushrita Yerra, Sahil Gupta, Aishwarya Sahoo, Sharmin Sultana, Hong Yu, Zonghai Yao  

**Link**: [PDF](https://arxiv.org/pdf/2601.12471)  

**Abstract**: Current evaluation of large language models (LLMs) overwhelmingly prioritizes accuracy; however, in real-world and safety-critical applications, the ability to abstain when uncertain is equally vital for trustworthy deployment. We introduce MedAbstain, a unified benchmark and evaluation protocol for abstention in medical multiple-choice question answering (MCQA) -- a discrete-choice setting that generalizes to agentic action selection -- integrating conformal prediction, adversarial question perturbations, and explicit abstention options. Our systematic evaluation of both open- and closed-source LLMs reveals that even state-of-the-art, high-accuracy models often fail to abstain with uncertain. Notably, providing explicit abstention options consistently increases model uncertainty and safer abstention, far more than input perturbations, while scaling model size or advanced prompting brings little improvement. These findings highlight the central role of abstention mechanisms for trustworthy LLM deployment and offer practical guidance for improving safety in high-stakes applications. 

---
# Incentivizing In-depth Reasoning over Long Contexts with Process Advantage Shaping 

**Authors**: Miao Peng, Weizhou Shen, Nuo Chen, Chenliang Li, Ming Yan, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.12465)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective in enhancing LLMs short-context reasoning, but its performance degrades in long-context scenarios that require both precise grounding and robust long-range reasoning. We identify the "almost-there" phenomenon in long-context reasoning, where trajectories are largely correct but fail at the final step, and attribute this failure to two factors: (1) the lack of high reasoning density in long-context QA data that push LLMs beyond mere grounding toward sophisticated multi-hop reasoning; and (2) the loss of valuable learning signals during long-context RL training due to the indiscriminate penalization of partially correct trajectories with incorrect outcomes. To overcome this bottleneck, we propose DeepReasonQA, a KG-driven synthesis framework that controllably constructs high-difficulty, multi-hop long-context QA pairs with inherent reasoning chains. Building on this, we introduce Long-context Process Advantage Shaping (LongPAS), a simple yet effective method that performs fine-grained credit assignment by evaluating reasoning steps along Validity and Relevance dimensions, which captures critical learning signals from "almost-there" trajectories. Experiments on three long-context reasoning benchmarks show that our approach substantially outperforms RLVR baselines and matches frontier LLMs while using far fewer parameters. Further analysis confirms the effectiveness of our methods in strengthening long-context reasoning while maintaining stable RL training. 

---
# System-Mediated Attention Imbalances Make Vision-Language Models Say Yes 

**Authors**: Tsan Tsai Chan, Varsha Suresh, Anisha Saha, Michael Hahn, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2601.12430)  

**Abstract**: Vision-language model (VLM) hallucination is commonly linked to imbalanced allocation of attention across input modalities: system, image and text. However, existing mitigation strategies tend towards an image-centric interpretation of these imbalances, often prioritising increased image attention while giving less consideration to the roles of the other modalities. In this study, we evaluate a more holistic, system-mediated account, which attributes these imbalances to functionally redundant system weights that reduce attention to image and textual inputs. We show that this framework offers a useful empirical perspective on the yes-bias, a common form of hallucination in which VLMs indiscriminately respond 'yes'. Causally redistributing attention from the system modality to image and textual inputs substantially suppresses this bias, often outperforming existing approaches. We further present evidence suggesting that system-mediated attention imbalances contribute to the yes-bias by encouraging a default reliance on coarse input representations, which are effective for some tasks but ill-suited to others. Taken together, these findings firmly establish system attention as a key factor in VLM hallucination and highlight its potential as a lever for mitigation. 

---
# Legal experts disagree with rationale extraction techniques for explaining ECtHR case outcome classification 

**Authors**: Mahammad Namazov, Tomáš Koref, Ivan Habernal  

**Link**: [PDF](https://arxiv.org/pdf/2601.12419)  

**Abstract**: Interpretability is critical for applications of large language models in the legal domain which requires trust and transparency. While some studies develop task-specific approaches, other use the classification model's parameters to explain the decisions. However, which technique explains the legal outcome prediction best remains an open question. To address this challenge, we propose a comparative analysis framework for model-agnostic interpretability techniques. Among these, we employ two rationale extraction methods, which justify outcomes with human-interpretable and concise text fragments (i.e., rationales) from the given input text. We conduct comparison by evaluating faithfulness-via normalized sufficiency and comprehensiveness metrics along with plausibility-by asking legal experts to evaluate extracted rationales. We further assess the feasibility of LLM-as-a-Judge using legal expert evaluation results. We show that the model's "reasons" for predicting a violation differ substantially from those of legal experts, despite highly promising quantitative analysis results and reasonable downstream classification performance. The source code of our experiments is publicly available at this https URL. 

---
# NADIR: Differential Attention Flow for Non-Autoregressive Transliteration in Indic Languages 

**Authors**: Lakshya Tomar, Vinayak Abrol, Puneet Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2601.12389)  

**Abstract**: In this work, we argue that not all sequence-to-sequence tasks require the strong inductive biases of autoregressive (AR) models. Tasks like multilingual transliteration, code refactoring, grammatical correction or text normalization often rely on local dependencies where the full modeling capacity of AR models can be overkill, creating a trade-off between their high accuracy and high inference latency. While non-autoregressive (NAR) models offer speed, they typically suffer from hallucinations and poor length control. To explore this trade-off, we focus on the multilingual transliteration task in Indic languages and introduce NADIR, a novel NAR architecture designed to strike a balance between speed and accuracy. NADIR integrates a Differential Transformer and a Mixture-of-Experts mechanism, enabling it to robustly model complex character mappings without sequential dependencies. NADIR achieves over a 13x speed-up compared to the state-of-the-art AR baseline. It maintains a competitive mean Character Error Rate of 15.78%, compared to 14.44% for the AR model and 21.88% for a standard NAR equivalent. Importantly, NADIR reduces Repetition errors by 49.53%, Substitution errors by 24.45%, Omission errors by 32.92%, and Insertion errors by 16.87%. This work provides a practical blueprint for building fast and reliable NAR systems, effectively bridging the gap between AR accuracy and the demands of real-time, large-scale deployment. 

---
# LR-DWM: Efficient Watermarking for Diffusion Language Models 

**Authors**: Ofek Raban, Ethan Fetaya, Gal Chechik  

**Link**: [PDF](https://arxiv.org/pdf/2601.12376)  

**Abstract**: Watermarking (WM) is a critical mechanism for detecting and attributing AI-generated content. Current WM methods for Large Language Models (LLMs) are predominantly tailored for autoregressive (AR) models: They rely on tokens being generated sequentially, and embed stable signals within the generated sequence based on the previously sampled text. Diffusion Language Models (DLMs) generate text via non-sequential iterative denoising, which requires significant modification to use WM methods designed for AR models. Recent work proposed to watermark DLMs by inverting the process when needed, but suffers significant computational or memory overhead. We introduce Left-Right Diffusion Watermarking (LR-DWM), a scheme that biases the generated token based on both left and right neighbors, when they are available. LR-DWM incurs minimal runtime and memory overhead, remaining close to the non-watermarked baseline DLM while enabling reliable statistical detection under standard evaluation settings. Our results demonstrate that DLMs can be watermarked efficiently, achieving high detectability with negligible computational and memory overhead. 

---
# A Scalable Entity-Based Framework for Auditing Bias in LLMs 

**Authors**: Akram Elbouanani, Aboubacar Tuo, Adrian Popescu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12374)  

**Abstract**: Existing approaches to bias evaluation in large language models (LLMs) trade ecological validity for statistical control, relying on artificial prompts that poorly reflect real-world use, or on naturalistic tasks that lack scale and rigor. We introduce a scalable bias-auditing framework using named entities as probes to measure structural disparities in model behavior. We show that synthetic data reliably reproduces bias patterns observed in natural text, enabling large-scale analysis. Using this approach, we conduct the largest bias audit to date, comprising 1.9 billion data points across multiple entity types, tasks, languages, models, and prompting strategies. Our results reveal systematic biases: models penalize right-wing politicians, favor left-wing politicians, prefer Western and wealthy nations over the Global South, favor Western companies, and penalize firms in the defense and pharmaceutical sectors. While instruction tuning reduces bias, increasing model scale amplifies it, and prompting in Chinese or Russian does not attenuate Western-aligned preferences. These results indicate that LLMs should undergo rigorous auditing before deployment in high-stakes applications. 

---
# Can Deep Research Agents Find and Organize? Evaluating the Synthesis Gap with Expert Taxonomies 

**Authors**: Ming Zhang, Jiabao Zhuang, Wenqing Jing, Ziyu Kong, Jingyi Deng, Yujiong Shen, Kexin Tan, Yuhang Zhao, Ning Luo, Renzhe Zheng, Jiahui Lin, Mingqi Wu, Long Ma, Yi Zou, Shihan Dou, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.12369)  

**Abstract**: Deep Research Agents are increasingly used for automated survey generation. However, whether they can write surveys like human experts remains unclear. Existing benchmarks focus on fluency or citation accuracy, but none evaluates the core capabilities: retrieving essential papers and organizing them into coherent knowledge structures. We introduce TaxoBench, a diagnostic benchmark derived from 72 highly-cited computer science surveys. We manually extract expert-authored taxonomy trees containing 3,815 precisely categorized citations as ground truth. Our benchmark supports two evaluation modes: Deep Research mode tests end-to-end retrieval and organization given only a topic, while Bottom-Up mode isolates structuring capability by providing the exact papers human experts used. We evaluate 7 leading Deep Research agents and 12 frontier LLMs. Results reveal a dual bottleneck: the best agent recalls only 20.9% of expert-selected papers, and even with perfect input, the best model achieves only 0.31 ARI in organization. Current deep research agents remain far from expert-level survey writing. Our benchmark is publicly available at this https URL. 

---
# Conversational Context Classification: A Representation Engineering Approach 

**Authors**: Jonathan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2601.12286)  

**Abstract**: The increasing prevalence of Large Language Models (LLMs) demands effective safeguards for their operation, particularly concerning their tendency to generate out-of-context responses. A key challenge is accurately detecting when LLMs stray from expected conversational norms, manifesting as topic shifts, factual inaccuracies, or outright hallucinations. Traditional anomaly detection struggles to directly apply within contextual semantics. This paper outlines our experiment in exploring the use of Representation Engineering (RepE) and One-Class Support Vector Machine (OCSVM) to identify subspaces within the internal states of LLMs that represent a specific context. By training OCSVM on in-context examples, we establish a robust boundary within the LLM's hidden state latent space. We evaluate out study with two open source LLMs - Llama and Qwen models in specific contextual domain. Our approach entailed identifying the optimal layers within the LLM's internal state subspaces that strongly associates with the context of interest. Our evaluation results showed promising results in identifying the subspace for a specific context. Aside from being useful in detecting in or out of context conversation threads, this research work contributes to the study of better interpreting LLMs. 

---
# Simulated Annealing Enhances Theory-of-Mind Reasoning in Autoregressive Language Models 

**Authors**: Xucong Hu, Jian-Qiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12269)  

**Abstract**: Autoregressive language models are next-token predictors and have been criticized for only optimizing surface plausibility (i.e., local coherence) rather than maintaining correct latent-state representations (i.e., global coherence). Because Theory of Mind (ToM) tasks crucially depend on reasoning about latent mental states of oneself and others, such models are therefore often thought to fail at ToM. While post-training methods can improve ToM performance, we show that strong ToM capability can be recovered directly from the base model without any additional weight updates or verifications. Our approach builds on recent power-sampling methods (Karan & Du, 2025) that use Markov chain Monte Carlo (MCMC) to sample from sharpened sequence-level (rather than token-level) probability distributions of autoregressive language models. We further find that incorporating annealing, where the tempered distribution is gradually shifted from high to low temperature, substantially improves ToM performance over fixed-temperature power sampling. Together, these results suggest that sampling-based optimization provides a powerful way to extract latent capabilities from language models without retraining. 

---
# Multimodal Generative Engine Optimization: Rank Manipulation for Vision-Language Model Rankers 

**Authors**: Yixuan Du, Chenxiao Yu, Haoyan Xu, Ziyi Wang, Yue Zhao, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12263)  

**Abstract**: Vision-Language Models (VLMs) are rapidly replacing unimodal encoders in modern retrieval and recommendation systems. While their capabilities are well-documented, their robustness against adversarial manipulation in competitive ranking scenarios remains largely unexplored. In this paper, we uncover a critical vulnerability in VLM-based product search: multimodal ranking attacks. We present Multimodal Generative Engine Optimization (MGEO), a novel adversarial framework that enables a malicious actor to unfairly promote a target product by jointly optimizing imperceptible image perturbations and fluent textual suffixes. Unlike existing attacks that treat modalities in isolation, MGEO employs an alternating gradient-based optimization strategy to exploit the deep cross-modal coupling within the VLM. Extensive experiments on real-world datasets using state-of-the-art models demonstrate that our coordinated attack significantly outperforms text-only and image-only baselines. These findings reveal that multimodal synergy, typically a strength of VLMs, can be weaponized to compromise the integrity of search rankings without triggering conventional content filters. 

---
# Plan, Verify and Fill: A Structured Parallel Decoding Approach for Diffusion Language Models 

**Authors**: Miao Li, Hanyang Jiang, Sikai Chen, Hengyu Fu, Yuhang Cai, Baihe Huang, Tinghan Ye, Xuanzhou Chen, Pascal Van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2601.12247)  

**Abstract**: Diffusion Language Models (DLMs) present a promising non-sequential paradigm for text generation, distinct from standard autoregressive (AR) approaches. However, current decoding strategies often adopt a reactive stance, underutilizing the global bidirectional context to dictate global trajectories. To address this, we propose Plan-Verify-Fill (PVF), a training-free paradigm that grounds planning via quantitative validation. PVF actively constructs a hierarchical skeleton by prioritizing high-leverage semantic anchors and employs a verification protocol to operationalize pragmatic structural stopping where further deliberation yields diminishing returns. Extensive evaluations on LLaDA-8B-Instruct and Dream-7B-Instruct demonstrate that PVF reduces the Number of Function Evaluations (NFE) by up to 65% compared to confidence-based parallel decoding across benchmark datasets, unlocking superior efficiency without compromising accuracy. 

---
# CoReflect: Conversational Evaluation via Co-Evolutionary Simulation and Reflective Rubric Refinement 

**Authors**: Yunzhe Li, Richie Yueqi Feng, Tianxin Wei, Chin-Chia Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12208)  

**Abstract**: Evaluating conversational systems in multi-turn settings remains a fundamental challenge. Conventional pipelines typically rely on manually defined rubrics and fixed conversational context$-$a static approach that limits coverage and fails to capture the diverse, emergent behaviors of dialogue models. To address this, we introduce CoReflect (Conversational Evaluation via Co-Evolutionary Simulation and Reflective Rubric Refinement), which unifies dialogue simulation and evaluation into an adaptive, iterative process. CoReflect employs a conversation planner that generates structured templates to guide a user simulator through diverse, goal-directed dialogues. Subsequently, a reflective analyzer processes these dialogues to identify systematic behavioral patterns and automatically refine the evaluation rubrics. Crucially, the insights from the conversation analysis are fed back into the planner to update conversation templates for subsequent iterations. This co-evolution loop ensures that the complexity of test cases and the diagnostic precision of rubrics improve in tandem. By minimizing human intervention, CoReflect provides a scalable and self-refining methodology that allows evaluation protocols to adapt alongside the rapidly advancing capabilities of dialogue models. 

---
# CTC-DID: CTC-Based Arabic dialect identification for streaming applications 

**Authors**: Muhammad Umar Farooq, Oscar Saz  

**Link**: [PDF](https://arxiv.org/pdf/2601.12199)  

**Abstract**: This paper proposes a Dialect Identification (DID) approach inspired by the Connectionist Temporal Classification (CTC) loss function as used in Automatic Speech Recognition (ASR). CTC-DID frames the dialect identification task as a limited-vocabulary ASR system, where dialect tags are treated as a sequence of labels for a given utterance. For training, the repetition of dialect tags in transcriptions is estimated either using a proposed Language-Agnostic Heuristic (LAH) approach or a pre-trained ASR model. The method is evaluated on the low-resource Arabic Dialect Identification (ADI) task, with experimental results demonstrating that an SSL-based CTC-DID model, trained on a limited dataset, outperforms both fine-tuned Whisper and ECAPA-TDNN models. Notably, CTC-DID also surpasses these models in zero-shot evaluation on the Casablanca dataset. The proposed approach is found to be more robust to shorter utterances and is shown to be easily adaptable for streaming, real-time applications, with minimal performance degradation. 

---
# Tolerance Principle and Small Language Model Learning 

**Authors**: Adam E. Friedman, Stevan Harnad, Rushen Shi  

**Link**: [PDF](https://arxiv.org/pdf/2601.12179)  

**Abstract**: Modern language models like GPT-3, BERT, and LLaMA require massive training data, yet with sufficient training they reliably learn to distinguish grammatical from ungrammatical sentences. Children aged as young as 14 months already have the capacity to learn abstract grammar rules from very few exemplars, even in the presence of non-rule-following exceptions. Yang's (2016) Tolerance Principle defines a precise threshold for how many exceptions a rule can tolerate and still be learnable. The present study explored the minimal amount and quality of training data necessary for rules to be generalized by a transformer-based language model to test the predictions of the Tolerance Principle. We trained BabyBERTa (Huebner et al. 2021), a transformer model optimized for small datasets, on artificial grammars. The training sets varied in size, number of unique sentence types, and proportion of rule-following versus exception exemplars. We found that, unlike human infants, BabyBERTa's learning dynamics do not align with the Tolerance Principle. 

---
# Analyzing Cancer Patients' Experiences with Embedding-based Topic Modeling and LLMs 

**Authors**: Teodor-Călin Ionescu, Lifeng Han, Jan Heijdra Suasnabar, Anne Stiggelbout, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2601.12154)  

**Abstract**: This study investigates the use of neural topic modeling and LLMs to uncover meaningful themes from patient storytelling data, to offer insights that could contribute to more patient-oriented healthcare practices. We analyze a collection of transcribed interviews with cancer patients (132,722 words in 13 interviews). We first evaluate BERTopic and Top2Vec for individual interview summarization by using similar preprocessing, chunking, and clustering configurations to ensure a fair comparison on Keyword Extraction. LLMs (GPT4) are then used for the next step topic labeling. Their outputs for a single interview (I0) are rated through a small-scale human evaluation, focusing on {coherence}, {clarity}, and {relevance}. Based on the preliminary results and evaluation, BERTopic shows stronger performance and is selected for further experimentation using three {clinically oriented embedding} models. We then analyzed the full interview collection with the best model setting. Results show that domain-specific embeddings improved topic \textit{precision} and \textit{interpretability}, with BioClinicalBERT producing the most consistent results across transcripts. The global analysis of the full dataset of 13 interviews, using the BioClinicalBERT embedding model, reveals the most dominant topics throughout all 13 interviews, namely ``Coordination and Communication in Cancer Care Management" and ``Patient Decision-Making in Cancer Treatment Journey''. Although the interviews are machine translations from Dutch to English, and clinical professionals are not involved in this evaluation, the findings suggest that neural topic modeling, particularly BERTopic, can help provide useful feedback to clinicians from patient interviews. This pipeline could support more efficient document navigation and strengthen the role of patients' voices in healthcare workflows. 

---
# Bengali Text Classification: An Evaluation of Large Language Model Approaches 

**Authors**: Md Mahmudul Hoque, Md Mehedi Hassain, Md Hojaifa Tanvir, Rahul Nandy  

**Link**: [PDF](https://arxiv.org/pdf/2601.12132)  

**Abstract**: Bengali text classification is a Significant task in natural language processing (NLP), where text is categorized into predefined labels. Unlike English, Bengali faces challenges due to the lack of extensive annotated datasets and pre-trained language models. This study explores the effectiveness of large language models (LLMs) in classifying Bengali newspaper articles. The dataset used, obtained from Kaggle, consists of articles from Prothom Alo, a major Bangladeshi newspaper. Three instruction-tuned LLMs LLaMA 3.1 8B Instruct, LLaMA 3.2 3B Instruct, and Qwen 2.5 7B Instruct were evaluated for this task under the same classification framework. Among the evaluated models, Qwen 2.5 achieved the highest classification accuracy of 72%, showing particular strength in the "Sports" category. In comparison, LLaMA 3.1 and LLaMA 3.2 attained accuracies of 53% and 56%, respectively. The findings highlight the effectiveness of LLMs in Bengali text classification, despite the scarcity of resources for Bengali NLP. Future research will focus on exploring additional models, addressing class imbalance issues, and refining fine-tuning approaches to improve classification performance. 

---
# Powerful Training-Free Membership Inference Against Autoregressive Language Models 

**Authors**: David Ilić, David Stanojević, Kostadin Cvejoski  

**Link**: [PDF](https://arxiv.org/pdf/2601.12104)  

**Abstract**: Fine-tuned language models pose significant privacy risks, as they may memorize and expose sensitive information from their training data. Membership inference attacks (MIAs) provide a principled framework for auditing these risks, yet existing methods achieve limited detection rates, particularly at the low false-positive thresholds required for practical privacy auditing. We present EZ-MIA, a membership inference attack that exploits a key observation: memorization manifests most strongly at error positions, specifically tokens where the model predicts incorrectly yet still shows elevated probability for training examples. We introduce the Error Zone (EZ) score, which measures the directional imbalance of probability shifts at error positions relative to a pretrained reference model. This principled statistic requires only two forward passes per query and no model training of any kind. On WikiText with GPT-2, EZ-MIA achieves 3.8x higher detection than the previous state-of-the-art under identical conditions (66.3% versus 17.5% true positive rate at 1% false positive rate), with near-perfect discrimination (AUC 0.98). At the stringent 0.1% FPR threshold critical for real-world auditing, we achieve 8x higher detection than prior work (14.0% versus 1.8%), requiring no reference model training. These gains extend to larger architectures: on AG News with Llama-2-7B, we achieve 3x higher detection (46.7% versus 15.8% TPR at 1% FPR). These results establish that privacy risks of fine-tuned language models are substantially greater than previously understood, with implications for both privacy auditing and deployment decisions. Code is available at this https URL. 

---
# Large language models struggle with ethnographic text annotation 

**Authors**: Leonardo S. Goodall, Dor Shilton, Daniel A. Mullins, Harvey Whitehouse  

**Link**: [PDF](https://arxiv.org/pdf/2601.12099)  

**Abstract**: Large language models (LLMs) have shown promise for automated text annotation, raising hopes that they might accelerate cross-cultural research by extracting structured data from ethnographic texts. We evaluated 7 state-of-the-art LLMs on their ability to annotate 121 ritual features across 567 ethnographic excerpts. Performance was limited, falling well below levels required for reliable automated annotation. Longer texts, features requiring ordinal distinctions, and ambiguous constructs proved particularly difficult. Human inter-coder reliability set an approximate ceiling on LLM accuracy: features that human coders found difficult to agree upon were also difficult for LLMs. Yet even on features where humans reliably agreed, models fell short of human performance. Our findings suggest that LLMs cannot yet substitute for human expertise in ethnographic annotation. 

---
# Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization 

**Authors**: Linfeng Du, Ye Yuan, Zichen Zhao, Fuyuan Lyu, Emiliano Penaloza, Xiuying Chen, Zipeng Sun, Jikun Kang, Laurent Charlin, Xue Liu, Haolun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12078)  

**Abstract**: Large Language Models (LLMs) excel at general-purpose tasks, yet adapting their responses to individual users remains challenging. Retrieval augmentation provides a lightweight alternative to fine-tuning by conditioning LLMs on user history records, and existing approaches typically select these records based on semantic relevance. We argue that relevance serves as an unreliable proxy for utility: a record may be semantically similar to a query yet fail to improve generation quality or even degrade it due to redundancy or conflicting information. To bridge this gap, we propose PURPLE, a contextual bandit framework that oPtimizes UseR Profiles for Llm pErsonalization. In contrast to a greedy selection of the most relevant records, PURPLE treats profile construction as a set generation process and utilizes a Plackett-Luce ranking model to capture complex inter-record dependencies. By training with dense feedback provided by the likelihood of the reference response, our method aligns retrieval directly with generation quality. Extensive experiments on nine personalization tasks demonstrate that PURPLE consistently outperforms strong heuristic and retrieval-augmented baselines in both effectiveness and efficiency, establishing a principled and scalable solution for optimizing user profiles. 

---
# To Copy or Not to Copy: Copying Is Easier to Induce Than Recall 

**Authors**: Mehrdad Farahani, Franziska Penzkofer, Richard Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2601.12075)  

**Abstract**: Language models used in retrieval-augmented settings must arbitrate between parametric knowledge stored in their weights and contextual information in the prompt. This work presents a mechanistic study of that choice by extracting an \emph{arbitration vector} from model activations on a curated dataset designed to disentangle (i) irrelevant contexts that elicit parametric recall and (ii) relevant but false contexts that elicit copying. The vector is computed as the residual-stream centroid difference between these regimes across 27 relations, and is injected as an additive intervention at selected layers and token spans to steer behavior in two directions: Copy$\rightarrow$Recall (suppressing context use) and Recall$\rightarrow$Copy (inducing the model to copy any token from the context). Experiments on two architectures (decoder-only and encoder/decoder) and two open-domain QA benchmarks show consistent behavior shifts under moderate scaling while monitoring accuracy and fluency. Mechanistic analyses of attention routing, MLP contributions, and layer-wise probability trajectories reveal an asymmetry: inducing copying is an easy ``reactivation'' process that can be triggered at different locations in the input, while restoring recall is a ``suppression'' process that is more fragile and strongly tied to object-token interventions. 

---
# Bridging the Gap in Bangla Healthcare: Machine Learning Based Disease Prediction Using a Symptoms-Disease Dataset 

**Authors**: Rowzatul Zannat, Abdullah Al Shafi, Abdul Muntakim  

**Link**: [PDF](https://arxiv.org/pdf/2601.12068)  

**Abstract**: Increased access to reliable health information is essential for non-English-speaking populations, yet resources in Bangla for disease prediction remain limited. This study addresses this gap by developing a comprehensive Bangla symptoms-disease dataset containing 758 unique symptom-disease relationships spanning 85 diseases. To ensure transparency and reproducibility, we also make our dataset publicly available. The dataset enables the prediction of diseases based on Bangla symptom inputs, supporting healthcare accessibility for Bengali-speaking populations. Using this dataset, we evaluated multiple machine learning models to predict diseases based on symptoms provided in Bangla and analyzed their performance on our dataset. Both soft and hard voting ensemble approaches combining top-performing models achieved 98\% accuracy, demonstrating superior robustness and generalization. Our work establishes a foundational resource for disease prediction in Bangla, paving the way for future advancements in localized health informatics and diagnostic tools. This contribution aims to enhance equitable access to health information for Bangla-speaking communities, particularly for early disease detection and healthcare interventions. 

---
# Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation 

**Authors**: Jinsook Lee, Kirk Vanacore, Zhuqian Zhou, Jeanine Grutter, Rene F. Kizilcec  

**Link**: [PDF](https://arxiv.org/pdf/2601.12061)  

**Abstract**: Dialogue Act (DA) annotation typically treats communicative or pedagogical intent as localized to individual utterances or turns. This leads annotators to agree on the underlying action while disagreeing on segment boundaries, reducing apparent reliability. We propose codebook-injected segmentation, which conditions boundary decisions on downstream annotation criteria, and evaluate LLM-based segmenters against standard and retrieval-augmented baselines. To assess these without gold labels, we introduce evaluation metrics for span consistency, distinctiveness, and human-AI distributional agreement. We found DA-awareness produces segments that are internally more consistent than text-only baselines. While LLMs excel at creating construct-consistent spans, coherence-based baselines remain superior at detecting global shifts in dialogue flow. Across two datasets, no single segmenter dominates. Improvements in within-segment coherence frequently trade off against boundary distinctiveness and human-AI distributional agreement. These results highlight segmentation as a consequential design choice that should be optimized for downstream objectives rather than a single performance score. 

---
# Don't Start Over: A Cost-Effective Framework for Migrating Personalized Prompts Between LLMs 

**Authors**: Ziyi Zhao, Chongming Gao, Yang Zhang, Haoyan Liu, Weinan Gan, Huifeng Guo, Yong Liu, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.12034)  

**Abstract**: Personalization in Large Language Models (LLMs) often relies on user-specific soft prompts. However, these prompts become obsolete when the foundation model is upgraded, necessitating costly, full-scale retraining. To overcome this limitation, we propose the Prompt-level User Migration Adapter (PUMA), a lightweight framework to efficiently migrate personalized prompts across incompatible models. PUMA utilizes a parameter-efficient adapter to bridge the semantic gap, combined with a group-based user selection strategy to significantly reduce training costs. Experiments on three large-scale datasets show our method matches or even surpasses the performance of retraining from scratch, reducing computational cost by up to 98%. The framework demonstrates strong generalization across diverse model architectures and robustness in advanced scenarios like chained and aggregated migrations, offering a practical path for the sustainable evolution of personalized AI by decoupling user assets from the underlying models. 

---
# Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection 

**Authors**: Muhammad Alif Al Hakim, Alfan Farizki Wicaksono, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2601.12033)  

**Abstract**: Quantization is widely adopted to reduce the computational cost of large language models (LLMs); however, its implications for fairness and safety, particularly in dynamic quantization and multilingual contexts, remain underexplored. In this work, we conduct a systematic study of how static and dynamic quantization methods impact fairness and safety across benchmarks measuring intrinsic and extrinsic bias and safety alignment. For fairness, we evaluate English, French, Dutch, Spanish, and Turkish; for safety, we focus on English, Korean, and Arabic. Our findings reveal that quantization consistently degrades fairness and safety, with dynamic methods demonstrating greater stability than static ones. Moreover, fairness degradation varies across languages, while safety deterioration is especially pronounced in non-English settings. To address these risks, we introduce Critical Weight Protection, a novel technique that identifies and preserves fairness- and safety-critical weights during quantization. This approach effectively mitigates bias and safety deterioration without costly retraining or alignment, maintaining trustworthiness while retaining efficiency. 

---
# Acting Flatterers via LLMs Sycophancy: Combating Clickbait with LLMs Opposing-Stance Reasoning 

**Authors**: Chaowei Zhang, Xiansheng Luo, Zewei Zhang, Yi Zhu, Jipeng Qiang, Longwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.12019)  

**Abstract**: The widespread proliferation of online content has intensified concerns about clickbait, deceptive or exaggerated headlines designed to attract attention. While Large Language Models (LLMs) offer a promising avenue for addressing this issue, their effectiveness is often hindered by Sycophancy, a tendency to produce reasoning that matches users' beliefs over truthful ones, which deviates from instruction-following principles. Rather than treating sycophancy as a flaw to be eliminated, this work proposes a novel approach that initially harnesses this behavior to generate contrastive reasoning from opposing perspectives. Specifically, we design a Self-renewal Opposing-stance Reasoning Generation (SORG) framework that prompts LLMs to produce high-quality agree and disagree reasoning pairs for a given news title without requiring ground-truth labels. To utilize the generated reasoning, we develop a local Opposing Reasoning-based Clickbait Detection (ORCD) model that integrates three BERT encoders to represent the title and its associated reasoning. The model leverages contrastive learning, guided by soft labels derived from LLM-generated credibility scores, to enhance detection robustness. Experimental evaluations on three benchmark datasets demonstrate that our method consistently outperforms LLM prompting, fine-tuned smaller language models, and state-of-the-art clickbait detection baselines. 

---
# $\texttt{MemoryRewardBench}$: Benchmarking Reward Models for Long-Term Memory Management in Large Language Models 

**Authors**: Zecheng Tang, Baibei Ji, Ruoxi Sun, Haitian Wang, WangJie You, Zhang Yijun, Wenpeng Zhu, Ji Qi, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11969)  

**Abstract**: Existing works increasingly adopt memory-centric mechanisms to process long contexts in a segment manner, and effective memory management is one of the key capabilities that enables large language models to effectively propagate information across the entire sequence. Therefore, leveraging reward models (RMs) to automatically and reliably evaluate memory quality is critical. In this work, we introduce $\texttt{MemoryRewardBench}$, the first benchmark to systematically study the ability of RMs to evaluate long-term memory management processes. $\texttt{MemoryRewardBench}$ covers both long-context comprehension and long-form generation tasks, featuring 10 distinct settings with different memory management patterns, with context length ranging from 8K to 128K tokens. Evaluations on 13 cutting-edge RMs indicate a diminishing performance gap between open-source and proprietary models, with newer-generation models consistently outperforming their predecessors regardless of parameter count. We further expose the capabilities and fundamental limitations of current RMs in evaluating LLM memory management across diverse settings. 

---
# PEARL: Self-Evolving Assistant for Time Management with Reinforcement Learning 

**Authors**: Bingxuan Li, Jeonghwan Kim, Cheng Qian, Xiusi Chen, Eitan Anzenberg, Niran Kundapur, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2601.11957)  

**Abstract**: Overlapping calendar invitations force busy professionals to repeatedly decide which meetings to attend, reschedule, or decline. We refer to this preference-driven decision process as calendar conflict resolution. Automating such process is crucial yet challenging. Scheduling logistics drain hours, and human delegation often fails at scale, which motivate we to ask: Can we trust large language model (LLM) or language agent to manager time? To enable systematic study of this question, we introduce CalConflictBench, a benchmark for long-horizon calendar conflict resolution. Conflicts are presented sequentially and agents receive feedback after each round, requiring them to infer and adapt to user preferences progressively. Our experiments show that current LLM agents perform poorly with high error rates, e.g., Qwen-3-30B-Think has 35% average error rate. To address this gap, we propose PEARL, a reinforcement-learning framework that augments language agent with an external memory module and optimized round-wise reward design, enabling agent to progressively infer and adapt to user preferences on-the-fly. Experiments on CalConflictBench shows that PEARL achieves 0.76 error reduction rate, and 55% improvement in average error rate compared to the strongest baseline. 

---
# Double-Calibration: Towards Trustworthy LLMs via Calibrating Knowledge and Reasoning Confidence 

**Authors**: Yuyin Lu, Ziran Liang, Yanghui Rao, Wenqi Fan, Fu Lee Wang, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.11956)  

**Abstract**: Trustworthy reasoning in Large Language Models (LLMs) is challenged by their propensity for hallucination. While augmenting LLMs with Knowledge Graphs (KGs) improves factual accuracy, existing KG-augmented methods fail to quantify epistemic uncertainty in both the retrieved evidence and LLMs' reasoning. To bridge this gap, we introduce DoublyCal, a framework built on a novel double-calibration principle. DoublyCal employs a lightweight proxy model to first generate KG evidence alongside a calibrated evidence confidence. This calibrated supporting evidence then guides a black-box LLM, yielding final predictions that are not only more accurate but also well-calibrated, with confidence scores traceable to the uncertainty of the supporting evidence. Experiments on knowledge-intensive benchmarks show that DoublyCal significantly improves both the accuracy and confidence calibration of black-box LLMs with low token cost. 

---
# Event Detection with a Context-Aware Encoder and LoRA for Improved Performance on Long-Tailed Classes 

**Authors**: Abdullah Al Monsur, Nitesh Vamshi Bommisetty, Gene Louis Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.11932)  

**Abstract**: The current state of event detection research has two notable re-occurring limitations that we investigate in this study. First, the unidirectional nature of decoder-only LLMs presents a fundamental architectural bottleneck for natural language understanding tasks that depend on rich, bidirectional context. Second, we confront the conventional reliance on Micro-F1 scores in event detection literature, which systematically inflates performance by favoring majority classes. Instead, we focus on Macro-F1 as a more representative measure of a model's ability across the long-tail of event types. Our experiments demonstrate that models enhanced with sentence context achieve superior performance over canonical decoder-only baselines. Using Low-Rank Adaptation (LoRA) during finetuning provides a substantial boost in Macro-F1 scores in particular, especially for the decoder-only models, showing that LoRA can be an effective tool to enhance LLMs' performance on long-tailed event classes. 

---
# Mapping the maturation of TCM as an adjuvant to radiotherapy 

**Authors**: P. Bilha Githinji, Aikaterini Melliou, Xi Yuan, Dayan Zhang, Lian Zhang, Zhenglin Chen, Jiansong Ji, Chengying Lv, Jinhao Xu, Peiwu Qin, Dongmei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11923)  

**Abstract**: The integration of complementary medicine into oncology represents a paradigm shift that has seen to increasing adoption of Traditional Chinese Medicine (TCM) as an adjuvant to radiotherapy. About twenty-five years since the formal institutionalization of integrated oncology, it is opportune to synthesize the trajectory of evidence for TCM as an adjuvant to radiotherapy. Here we conduct a large-scale analysis of 69,745 publications (2000 - 2025), emerging a cyclical evolution defined by coordinated expansion and contraction in publication output, international collaboration, and funding commitments that mirrors a define-ideate-test pattern. Using a theme modeling workflow designed to determine a stable thematic structure of the field, we identify five dominant thematic axes - cancer types, supportive care, clinical endpoints, mechanisms, and methodology - that signal a focus on patient well-being, scientific rigor and mechanistic exploration. Cross-theme integration of TCM is patient-centered and systems-oriented. Together with the emergent cycles of evolution, the thematic structure demonstrates progressive specialization and potential defragmentation of the field or saturation of existing research agenda. The analysis points to a field that has matured its current research agenda and is likely at the cusp of something new. Additionally, the field exhibits positive reporting of findings that is homogeneous across publication types, thematic areas, and the cycles of evolution suggesting a system-wide positive reporting bias agnostic to structural drivers. 

---
# Enhancing LLM-Based Data Annotation with Error Decomposition 

**Authors**: Zhen Xu, Vedant Khatri, Yijun Dai, Xiner Liu, Siyan Li, Xuanming Zhang, Renzhe Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11920)  

**Abstract**: Large language models offer a scalable alternative to human coding for data annotation tasks, enabling the scale-up of research across data-intensive domains. While LLMs are already achieving near-human accuracy on objective annotation tasks, their performance on subjective annotation tasks, such as those involving psychological constructs, is less consistent and more prone to errors. Standard evaluation practices typically collapse all annotation errors into a single alignment metric, but this simplified approach may obscure different kinds of errors that affect final analytical conclusions in different ways. Here, we propose a diagnostic evaluation paradigm that incorporates a human-in-the-loop step to separate task-inherent ambiguity from model-driven inaccuracies and assess annotation quality in terms of their potential downstream impacts. We refine this paradigm on ordinal annotation tasks, which are common in subjective annotation. The refined paradigm includes: (1) a diagnostic taxonomy that categorizes LLM annotation errors along two dimensions: source (model-specific vs. task-inherent) and type (boundary ambiguity vs. conceptual misidentification); (2) a lightweight human annotation test to estimate task-inherent ambiguity from LLM annotations; and (3) a computational method to decompose observed LLM annotation errors following our taxonomy. We validate this paradigm on four educational annotation tasks, demonstrating both its conceptual validity and practical utility. Theoretically, our work provides empirical evidence for why excessively high alignment is unrealistic in specific annotation tasks and why single alignment metrics inadequately reflect the quality of LLM annotations. In practice, our paradigm can be a low-cost diagnostic tool that assesses the suitability of a given task for LLM annotation and provides actionable insights for further technical optimization. 

---
# LSTM-MAS: A Long Short-Term Memory Inspired Multi-Agent System for Long-Context Understanding 

**Authors**: Yichen Jiang, Peng Ye, Jiakang Yuan, Chongjun Tu, Lei Bai, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11913)  

**Abstract**: Effectively processing long contexts remains a fundamental yet unsolved challenge for large language models (LLMs). Existing single-LLM-based methods primarily reduce the context window or optimize the attention mechanism, but they often encounter additional computational costs or constrained expanded context length. While multi-agent-based frameworks can mitigate these limitations, they remain susceptible to the accumulation of errors and the propagation of hallucinations. In this work, we draw inspiration from the Long Short-Term Memory (LSTM) architecture to design a Multi-Agent System called LSTM-MAS, emulating LSTM's hierarchical information flow and gated memory mechanisms for long-context understanding. Specifically, LSTM-MAS organizes agents in a chained architecture, where each node comprises a worker agent for segment-level comprehension, a filter agent for redundancy reduction, a judge agent for continuous error detection, and a manager agent for globally regulates information propagation and retention, analogous to LSTM and its input gate, forget gate, constant error carousel unit, and output gate. These novel designs enable controlled information transfer and selective long-term dependency modeling across textual segments, which can effectively avoid error accumulation and hallucination propagation. We conducted an extensive evaluation of our method. Compared with the previous best multi-agent approach, CoA, our model achieves improvements of 40.93%, 43.70%,121.57% and 33.12%, on NarrativeQA, Qasper, HotpotQA, and MuSiQue, respectively. 

---
# PPA-Plan: Proactive Pitfall Avoidance for Reliable Planning in Long-Context LLM Reasoning 

**Authors**: Byeongjin Kim, Gyuwan Kim, Seo Yeon Park  

**Link**: [PDF](https://arxiv.org/pdf/2601.11908)  

**Abstract**: Large language models (LLMs) struggle with reasoning over long contexts where relevant information is sparsely distributed. Although plan-and-execute frameworks mitigate this by decomposing tasks into planning and execution, their effectiveness is often limited by unreliable plan generation due to dependence on surface-level cues. Consequently, plans may be based on incorrect assumptions, and once a plan is formed, identifying what went wrong and revising it reliably becomes difficult, limiting the effectiveness of reactive refinement. To address this limitation, we propose PPA-Plan, a proactive planning strategy for long-context reasoning that focuses on preventing such failures before plan generation. PPA-Plan identifies potential logical pitfalls and false assumptions, formulates them as negative constraints, and conditions plan generation on explicitly avoiding these constraints. Experiments on long-context QA benchmarks show that executing plans generated by PPA-Plan consistently outperforms existing plan-and-execute methods and direct prompting. 

---
# Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence 

**Authors**: Kaijie Mo, Siddhartha Venkatayogi, Chantal Shaib, Ramez Kouzy, Wei Xu, Byron C. Wallace, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.11886)  

**Abstract**: In high-stakes domains like medicine, it may be generally desirable for models to faithfully adhere to the context provided. But what happens if the context does not align with model priors or safety protocols? In this paper, we investigate how LLMs behave and reason when presented with counterfactual or even adversarial medical evidence. We first construct MedCounterFact, a counterfactual medical QA dataset that requires the models to answer clinical comparison questions (i.e., judge the efficacy of certain treatments, with evidence consisting of randomized controlled trials provided as context). In MedCounterFact, real-world medical interventions within the questions and evidence are systematically replaced with four types of counterfactual stimuli, ranging from unknown words to toxic substances. Our evaluation across multiple frontier LLMs on MedCounterFact reveals that in the presence of counterfactual evidence, existing models overwhelmingly accept such "evidence" at face value even when it is dangerous or implausible, and provide confident and uncaveated answers. While it may be prudent to draw a boundary between faithfulness and safety, our findings reveal that there exists no such boundary yet. 

---
# GloCTM: Cross-Lingual Topic Modeling via a Global Context Space 

**Authors**: Nguyen Tien Phat, Ngo Vu Minh, Linh Van Ngo, Nguyen Thi Ngoc Diep, Thien Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11872)  

**Abstract**: Cross-lingual topic modeling seeks to uncover coherent and semantically aligned topics across languages - a task central to multilingual understanding. Yet most existing models learn topics in disjoint, language-specific spaces and rely on alignment mechanisms (e.g., bilingual dictionaries) that often fail to capture deep cross-lingual semantics, resulting in loosely connected topic spaces. Moreover, these approaches often overlook the rich semantic signals embedded in multilingual pretrained representations, further limiting their ability to capture fine-grained alignment. We introduce GloCTM (Global Context Space for Cross-Lingual Topic Model), a novel framework that enforces cross-lingual topic alignment through a unified semantic space spanning the entire model pipeline. GloCTM constructs enriched input representations by expanding bag-of-words with cross-lingual lexical neighborhoods, and infers topic proportions using both local and global encoders, with their latent representations aligned through internal regularization. At the output level, the global topic-word distribution, defined over the combined vocabulary, structurally synchronizes topic meanings across languages. To further ground topics in deep semantic space, GloCTM incorporates a Centered Kernel Alignment (CKA) loss that aligns the latent topic space with multilingual contextual embeddings. Experiments across multiple benchmarks demonstrate that GloCTM significantly improves topic coherence and cross-lingual alignment, outperforming strong baselines. 

---
# Advances in LLM Reasoning Enable Flexibility in Clinical Problem-Solving 

**Authors**: Kie Shidara, Preethi Prem, Jonathan Kim, Anna Podlasek, Feng Liu, Ahmed Alaa, Danilo Bernardo  

**Link**: [PDF](https://arxiv.org/pdf/2601.11866)  

**Abstract**: Large Language Models (LLMs) have achieved high accuracy on medical question-answer (QA) benchmarks, yet their capacity for flexible clinical reasoning has been debated. Here, we asked whether advances in reasoning LLMs improve their cognitive flexibility in clinical reasoning. We assessed reasoning models from the OpenAI, Grok, Gemini, Claude, and DeepSeek families on the medicine abstraction and reasoning corpus (mARC), an adversarial medical QA benchmark which utilizes the Einstellung effect to induce inflexible overreliance on learned heuristic patterns in contexts where they become suboptimal. We found that strong reasoning models avoided Einstellung-based traps more often than weaker reasoning models, achieving human-level performance on mARC. On questions most commonly missed by physicians, the top 5 performing models answered 55% to 70% correctly with high confidence, indicating that these models may be less susceptible than humans to Einstellung effects. Our results indicate that strong reasoning models demonstrate improved flexibility in medical reasoning, achieving performance on par with humans on mARC. 

---
# CTPD: Cross Tokenizer Preference Distillation 

**Authors**: Truong Nguyen, Phi Van Dat, Ngan Nguyen, Linh Ngo Van, Trung Le, Thanh Hong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11865)  

**Abstract**: While knowledge distillation has seen widespread use in pre-training and instruction tuning, its application to aligning language models with human preferences remains underexplored, particularly in the more realistic cross-tokenizer setting. The incompatibility of tokenization schemes between teacher and student models has largely prevented fine-grained, white-box distillation of preference information. To address this gap, we propose Cross-Tokenizer Preference Distillation (CTPD), the first unified framework for transferring human-aligned behavior between models with heterogeneous tokenizers. CTPD introduces three key innovations: (1) Aligned Span Projection, which maps teacher and student tokens to shared character-level spans for precise supervision transfer; (2) a cross-tokenizer adaptation of Token-level Importance Sampling (TIS-DPO) for improved credit assignment; and (3) a Teacher-Anchored Reference, allowing the student to directly leverage the teacher's preferences in a DPO-style objective. Our theoretical analysis grounds CTPD in importance sampling, and experiments across multiple benchmarks confirm its effectiveness, with significant performance gains over existing methods. These results establish CTPD as a practical and general solution for preference distillation across diverse tokenization schemes, opening the door to more accessible and efficient alignment of language models. 

---
# ATOD: An Evaluation Framework and Benchmark for Agentic Task-Oriented Dialogue System 

**Authors**: Yifei Zhang, Hooshang Nayyeri, Rinat Khaziev, Emine Yilmaz, Gokhan Tur, Dilek Hakkani-Tür, Hari Thadakamalla  

**Link**: [PDF](https://arxiv.org/pdf/2601.11854)  

**Abstract**: Recent advances in task-oriented dialogue (TOD) systems, driven by large language models (LLMs) with extensive API and tool integration, have enabled conversational agents to coordinate interleaved goals, maintain long-horizon context, and act proactively through asynchronous execution. These capabilities extend beyond traditional TOD systems, yet existing benchmarks lack systematic support for evaluating such agentic behaviors. To address this gap, we introduce ATOD, a benchmark and synthetic dialogue generation pipeline that produces richly annotated conversations requiring long-term reasoning. ATOD captures key characteristics of advanced TOD, including multi-goal coordination, dependency management, memory, adaptability, and proactivity. Building on ATOD, we propose ATOD-Eval, a holistic evaluation framework that translates these dimensions into fine-grained metrics and supports reproducible offline and online evaluation. We further present a strong agentic memory-based evaluator for benchmarking on ATOD. Experiments show that ATOD-Eval enables comprehensive assessment across task completion, agentic capability, and response quality, and that the proposed evaluator offers a better accuracy-efficiency tradeoff compared to existing memory- and LLM-based approaches under this evaluation setting. 

---
# The Third VoicePrivacy Challenge: Preserving Emotional Expressiveness and Linguistic Content in Voice Anonymization 

**Authors**: Natalia Tomashenko, Xiaoxiao Miao, Pierre Champion, Sarina Meyer, Michele Panariello, Xin Wang, Nicholas Evans, Emmanuel Vincent, Junichi Yamagishi, Massimiliano Todisco  

**Link**: [PDF](https://arxiv.org/pdf/2601.11846)  

**Abstract**: We present results and analyses from the third VoicePrivacy Challenge held in 2024, which focuses on advancing voice anonymization technologies. The task was to develop a voice anonymization system for speech data that conceals a speaker's voice identity while preserving linguistic content and emotional state. We provide a systematic overview of the challenge framework, including detailed descriptions of the anonymization task and datasets used for both system development and evaluation. We outline the attack model and objective evaluation metrics for assessing privacy protection (concealing speaker voice identity) and utility (content and emotional state preservation). We describe six baseline anonymization systems and summarize the innovative approaches developed by challenge participants. Finally, we provide key insights and observations to guide the design of future VoicePrivacy challenges and identify promising directions for voice anonymization research. 

---
# TWeddit : A Dataset of Triggering Stories Predominantly Shared by Women on Reddit 

**Authors**: Shirlene Rose Bandela, Sanjeev Parthasarathy, Vaibhav Garg  

**Link**: [PDF](https://arxiv.org/pdf/2601.11819)  

**Abstract**: Warning: This paper may contain examples and topics that may be disturbing to some readers, especially survivors of miscarriage and sexual violence. People affected by abortion, miscarriage, or sexual violence often share their experiences on social media to express emotions and seek support. On public platforms like Reddit, where users can post long, detailed narratives (up to 40,000 characters), readers may be exposed to distressing content. Although Reddit allows manual trigger warnings, many users omit them due to limited awareness or uncertainty about which categories apply. There is scarcity of datasets on Reddit stories labeled for triggering experiences. We propose a curated Reddit dataset, TWeddit, covering triggering experiences related to issues majorly faced by women. Our linguistic analyses show that annotated stories in TWeddit express distinct topics and moral foundations, making the dataset useful for a wide range of future research. 

---
# Beyond Tokens: Concept-Level Training Objectives for LLMs 

**Authors**: Laya Iyer, Pranav Somani, Alice Guo, Dan Jurafsky, Chen Shani  

**Link**: [PDF](https://arxiv.org/pdf/2601.11791)  

**Abstract**: The next-token prediction (NTP) objective has been foundational in the development of modern large language models (LLMs), driving advances in fluency and generalization. However, NTP operates at the \textit{token} level, treating deviations from a single reference continuation as errors even when alternative continuations are equally plausible or semantically equivalent (e.g., ``mom'' vs. ``mother''). As a result, token-level loss can penalize valid abstractions, paraphrases, or conceptually correct reasoning paths, biasing models toward surface form rather than underlying meaning. This mismatch between the training signal and semantic correctness motivates learning objectives that operate over higher-level representations. We propose a shift from token-level to concept-level prediction, where concepts group multiple surface forms of the same idea (e.g., ``mom,'' ``mommy,'' ``mother'' $\rightarrow$ \textit{MOTHER}). We introduce various methods for integrating conceptual supervision into LLM training and show that concept-aware models achieve lower perplexity, improved robustness under domain shift, and stronger performance than NTP-based models on diverse NLP benchmarks. This suggests \textit{concept-level supervision} as an improved training signal that better aligns LLMs with human semantic abstractions. 

---
# Translation as a Scalable Proxy for Multilingual Evaluation 

**Authors**: Sheriff Issaka, Erick Rosas Gonzalez, Lieqi Liu, Evans Kofi Agyei, Lucas Bandarkar, Nanyun Peng, David Ifeoluwa Adelani, Francisco Guzmán, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2601.11778)  

**Abstract**: The rapid proliferation of LLMs has created a critical evaluation paradox: while LLMs claim multilingual proficiency, comprehensive non-machine-translated benchmarks exist for fewer than 30 languages, leaving >98% of the world's 7,000 languages in an empirical void. Traditional benchmark construction faces scaling challenges such as cost, scarcity of domain experts, and data contamination. We evaluate the validity of a simpler alternative: can translation quality alone indicate a model's broader multilingual capabilities? Through systematic evaluation of 14 models (1B-72B parameters) across 9 diverse benchmarks and 7 translation metrics, we find that translation performance is a good indicator of downstream task success (e.g., Phi-4, median Pearson r: MetricX = 0.89, xCOMET = 0.91, SSA-COMET = 0.87). These results suggest that the representational abilities supporting faithful translation overlap with those required for multilingual understanding. Translation quality, thus emerges as a strong, inexpensive first-pass proxy of multilingual performance, enabling a translation-first screening with targeted follow-up for specific tasks. 

---
# Cleansing the Artificial Mind: A Self-Reflective Detoxification Framework for Large Language Models 

**Authors**: Kaituo Zhang, Zhimeng Jiang, Na Zou  

**Link**: [PDF](https://arxiv.org/pdf/2601.11776)  

**Abstract**: Recent breakthroughs in Large Language Models (LLMs) have revealed remarkable generative capabilities and emerging self-regulatory mechanisms, including self-correction and self-rewarding. However, current detoxification techniques rarely exploit these built-in abilities; instead, they rely on external modules, labor-intensive data annotation, or human intervention --factors that hinder scalability and consistency. In this paper, we introduce a fully self-reflective detoxification framework that harnesses the inherent capacities of LLMs to detect, correct toxic content, and refine LLMs without external modules and data annotation. Specifically, we propose a Toxic Signal Detector --an internal self-identification mechanism, coupled with a systematic intervention process to transform toxic text into its non-toxic counterpart. This iterative procedure yields a contrastive detoxification dataset used to fine-tune the model, enhancing its ability for safe and coherent text generation. Experiments on benchmark datasets such as DetoxLLM and ParaDetox show that our method achieves better detoxification performance than state-of-the-art methods while preserving semantic fidelity. By obviating the need for human intervention or external components, this paper reveals the intrinsic self-detoxification ability of LLMs, offering a consistent and effective approach for mitigating harmful content generation. Ultimately, our findings underscore the potential for truly self-regulated language models, paving the way for more responsible and ethically guided text generation systems. 

---
# Industry-Aligned Granular Topic Modeling 

**Authors**: Sae Young Moon, Myeongjun Erik Jang, Haoyan Luo, Chunyang Xiao, Antonios Georgiadis, Fran Silavong  

**Link**: [PDF](https://arxiv.org/pdf/2601.11762)  

**Abstract**: Topic modeling has extensive applications in text mining and data analysis across various industrial sectors. Although the concept of granularity holds significant value for business applications by providing deeper insights, the capability of topic modeling methods to produce granular topics has not been thoroughly explored. In this context, this paper introduces a framework called TIDE, which primarily provides a novel granular topic modeling method based on large language models (LLMs) as a core feature, along with other useful functionalities for business applications, such as summarizing long documents, topic parenting, and distillation. Through extensive experiments on a variety of public and real-world business datasets, we demonstrate that TIDE's topic modeling approach outperforms modern topic modeling methods, and our auxiliary components provide valuable support for dealing with industrial business scenarios. The TIDE framework is currently undergoing the process of being open sourced. 

---
# Early Linguistic Pattern of Anxiety from Social Media Using Interpretable Linguistic Features: A Multi-Faceted Validation Study with Author-Disjoint Evaluation 

**Authors**: Arnab Das Utsa  

**Link**: [PDF](https://arxiv.org/pdf/2601.11758)  

**Abstract**: Anxiety affects hundreds of millions of individuals globally, yet large-scale screening remains limited. Social media language provides an opportunity for scalable detection, but current models often lack interpretability, keyword-robustness validation, and rigorous user-level data integrity. This work presents a transparent approach to social media-based anxiety detection through linguistically interpretable feature-grounded modeling and cross-domain validation. Using a substantial dataset of Reddit posts, we trained a logistic regression classifier on carefully curated subreddits for training, validation, and test splits. Comprehensive evaluation included feature ablation, keyword masking experiments, and varying-density difference analyses comparing anxious and control groups, along with external validation using clinically interviewed participants with diagnosed anxiety disorders. The model achieved strong performance while maintaining high accuracy even after sentiment removal or keyword masking. Early detection using minimal post history significantly outperformed random classification, and cross-domain analysis demonstrated strong consistency with clinical interview data. Results indicate that transparent linguistic features can support reliable, generalizable, and keyword-robust anxiety detection. The proposed framework provides a reproducible baseline for interpretable mental health screening across diverse online contexts. 

---
# LIME-LLM: Probing Models with Fluent Counterfactuals, Not Broken Text 

**Authors**: George Mihaila, Suleyman Olcay Polat, Poli Nemkova, Himanshu Sharma, Namratha V. Urs, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2601.11746)  

**Abstract**: Local explanation methods such as LIME (Ribeiro et al., 2016) remain fundamental to trustworthy AI, yet their application to NLP is limited by a reliance on random token masking. These heuristic perturbations frequently generate semantically invalid, out-of-distribution inputs that weaken the fidelity of local surrogate models. While recent generative approaches such as LLiMe (Angiulli et al., 2025b) attempt to mitigate this by employing Large Language Models for neighborhood generation, they rely on unconstrained paraphrasing that introduces confounding variables, making it difficult to isolate specific feature contributions. We introduce LIME-LLM, a framework that replaces random noise with hypothesis-driven, controlled perturbations. By enforcing a strict "Single Mask-Single Sample" protocol and employing distinct neutral infill and boundary infill strategies, LIME-LLM constructs fluent, on-manifold neighborhoods that rigorously isolate feature effects. We evaluate our method against established baselines (LIME, SHAP, Integrated Gradients) and the generative LLiMe baseline across three diverse benchmarks: CoLA, SST-2, and HateXplain using human-annotated rationales as ground truth. Empirical results demonstrate that LIME-LLM establishes a new benchmark for black-box NLP explainability, achieving significant improvements in local explanation fidelity compared to both traditional perturbation-based methods and recent generative alternatives. 

---
# Bridging Human Interpretation and Machine Representation: A Landscape of Qualitative Data Analysis in the LLM Era 

**Authors**: Xinyu Pi, Qisen Yang, Chuong Nguyen, Hua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11739)  

**Abstract**: LLMs are increasingly used to support qualitative research, yet existing systems produce outputs that vary widely--from trace-faithful summaries to theory-mediated explanations and system models. To make these differences explicit, we introduce a 4$\times$4 landscape crossing four levels of meaning-making (descriptive, categorical, interpretive, theoretical) with four levels of modeling (static structure, stages/timelines, causal pathways, feedback dynamics). Applying the landscape to prior LLM-based automation highlights a strong skew toward low-level meaning and low-commitment representations, with few reliable attempts at interpretive/theoretical inference or dynamical modeling. Based on the revealed gap, we outline an agenda for applying and building LLM-systems that make their interpretive and modeling commitments explicit, selectable, and governable. 

---
# RAC: Retrieval-Augmented Clarification for Faithful Conversational Search 

**Authors**: Ahmed Rayane Kebir, Vincent Guigue, Lynda Said Lhadj, Laure Soulier  

**Link**: [PDF](https://arxiv.org/pdf/2601.11722)  

**Abstract**: Clarification questions help conversational search systems resolve ambiguous or underspecified user queries. While prior work has focused on fluency and alignment with user intent, especially through facet extraction, much less attention has been paid to grounding clarifications in the underlying corpus. Without such grounding, systems risk asking questions that cannot be answered from the available documents. We introduce RAC (Retrieval-Augmented Clarification), a framework for generating corpus-faithful clarification questions. After comparing several indexing strategies for retrieval, we fine-tune a large language model to make optimal use of research context and to encourage the generation of evidence-based question. We then apply contrastive preference optimization to favor questions supported by retrieved passages over ungrounded alternatives. Evaluated on four benchmarks, RAC demonstrate significant improvements over baselines. In addition to LLM-as-Judge assessments, we introduce novel metrics derived from NLI and data-to-text to assess how well questions are anchored in the context, and we demonstrate that our approach consistently enhances faithfulness. 

---
# Towards AGI A Pragmatic Approach Towards Self Evolving Agent 

**Authors**: Indrajit Kar, Sammy Zonunpuia, Zonunfeli Ralte  

**Link**: [PDF](https://arxiv.org/pdf/2601.11658)  

**Abstract**: Large Language Model (LLM) based agents are powerful yet fundamentally static after deployment, lacking the ability to autonomously expand capabilities, generate new tools, or evolve their reasoning. This work introduces a hierarchical self-evolving multi-agent framework that integrates a Base LLM, an operational SLM agent, a Code-Generation LLM, and a Teacher-LLM to enable continuous adaptation. The workflow begins with the agent attempting a task using reasoning and existing tools; if unsuccessful, it escalates to tool synthesis through the Code-Gen LLM, and when failures persist, it triggers an evolution phase using Curriculum Learning (CL), Reward-Based Learning (RL), or Genetic Algorithm (GA) evolution. Using the TaskCraft dataset rich in hierarchical tasks, tool-use traces, and difficulty scaling we evaluate these paradigms. CL delivers fast recovery and strong generalization, RL excels on high-difficulty tasks, and GA offers high behavioral diversity. Across all settings, evolved agents outperform their originals, demonstrating robust, autonomous, self-improving agentic evolution. 

---
# Entropic Context Shaping: Information-Theoretic Filtering for Context-Aware LLM Agents 

**Authors**: Hyunjun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.11585)  

**Abstract**: Context engineering for large language model (LLM) agents requires distinguishing pragmatically useful information from misleading distractors. We introduce Entropic Context Shaping (ECS), an information-theoretic framework that measures context utility via the shift in the model's answer distribution toward the correct answer. Unlike lexical similarity methods that rely on word overlap, ECS captures pragmatic utility -- whether a passage actually helps answer the question. We formalize utility as the signed change in answer probability and provide theoretical analysis showing that task-irrelevant updates yield near-zero distribution shift. We evaluate on multi-turn context selection tasks using LongMemEval (session-level) and LoCoMo (turn-level) benchmarks. On fine-grained turn selection, ECS with Llama-3.1-8B achieves F1=0.265, a 71.83% relative improvement over TF-IDF (F1=0.154), demonstrating that pragmatic utility outperforms lexical similarity when precise context selection matters. Code and data are available in the supplementary materials. 

---
# Enhancing the QA Model through a Multi-domain Debiasing Framework 

**Authors**: Yuefeng Wang, ChangJae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.11581)  

**Abstract**: Question-answering (QA) models have advanced significantly in machine reading comprehension but often exhibit biases that hinder their performance, particularly with complex queries in adversarial conditions. This study evaluates the ELECTRA-small model on the Stanford Question Answering Dataset (SQuAD) v1.1 and adversarial datasets AddSent and AddOneSent. By identifying errors related to lexical bias, numerical reasoning, and entity recognition, we develop a multi-domain debiasing framework incorporating knowledge distillation, debiasing techniques, and domain expansion. Our results demonstrate up to 2.6 percentage point improvements in Exact Match (EM) and F1 scores across all test sets, with gains in adversarial contexts. These findings highlight the potential of targeted bias mitigation strategies to enhance the robustness and reliability of natural language understanding systems. 

---
# Speculative Decoding: Performance or Illusion? 

**Authors**: Xiaoxuan Liu, Jiaxiang Yu, Jongseok Park, Ion Stoica, Alvin Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2601.11580)  

**Abstract**: Speculative decoding (SD) has become a popular technique to accelerate Large Language Model (LLM) inference, yet its real-world effectiveness remains unclear as prior evaluations rely on research prototypes and unrealistically small batch sizes. We present, to our knowledge, the first systematic study of SD on a production-grade and widely deployed inference engine (vLLM), covering multiple SD variants ($n$-gram, EAGLE/EAGLE-3, Draft-Model, Multi-Token Prediction) across diverse workloads, model scales, and batch sizes. We analyze key factors governing SD performance, and quantify a theoretical upper bound on SD speedup. Our results show that verification by the target model dominates the execution, while acceptance length varies markedly across output token positions, requests, and datasets. Comparing measured performance with theoretical bounds reveals substantial gaps between observed and theoretical upper bounds, and we leverage this observation to highlight new research opportunities that our study opens up in improving SD. 

---
# Bielik 11B v3: Multilingual Large Language Model for European Languages 

**Authors**: Krzysztof Ociepa, Łukasz Flis, Remigiusz Kinas, Krzysztof Wróbel, Adrian Gwoździej  

**Link**: [PDF](https://arxiv.org/pdf/2601.11579)  

**Abstract**: We present Bielik 11B v3, a state-of-the-art language model highly optimized for the Polish language, while also maintaining strong capabilities in other European languages. This model extends the Mistral 7B v0.2 architecture, scaled to 11B parameters via depth up-scaling. Its development involved a comprehensive four-stage training pipeline: continuous pre-training, supervised fine-tuning (SFT), Direct Preference Optimization (DPO), and reinforcement learning.
Comprehensive evaluations demonstrate that Bielik 11B v3 achieves exceptional performance. It significantly surpasses other specialized Polish language models and outperforms many larger models (with 2-6 times more parameters) on a wide range of tasks, from basic linguistic understanding to complex reasoning.
The model's parameter efficiency, combined with extensive quantization options, allows for effective deployment across diverse hardware configurations. Bielik 11B v3 not only advances AI capabilities for the Polish language but also establishes a new benchmark for developing resource-efficient, high-performance models for less-represented languages. 

---
# LimAgents: Multi-Agent LLMs for Generating Research Limitations 

**Authors**: Ibrahim Al Azher, Zhishuai Guo, Hamed Alhoori  

**Link**: [PDF](https://arxiv.org/pdf/2601.11578)  

**Abstract**: Identifying and articulating limitations is essential for transparent and rigorous scientific research. However, zero-shot large language models (LLMs) approach often produce superficial or general limitation statements (e.g., dataset bias or generalizability). They usually repeat limitations reported by authors without looking at deeper methodological issues and contextual gaps. This problem is made worse because many authors disclose only partial or trivial limitations. We propose LimAgents, a multi-agent LLM framework for generating substantive limitations. LimAgents integrates OpenReview comments and author-stated limitations to provide stronger ground truth. It also uses cited and citing papers to capture broader contextual weaknesses. In this setup, different agents have specific roles as sequential role: some extract explicit limitations, others analyze methodological gaps, some simulate the viewpoint of a peer reviewer, and a citation agent places the work within the larger body of literature. A Judge agent refines their outputs, and a Master agent consolidates them into a clear set. This structure allows for systematic identification of explicit, implicit, peer review-focused, and literature-informed limitations. Moreover, traditional NLP metrics like BLEU, ROUGE, and cosine similarity rely heavily on n-gram or embedding overlap. They often overlook semantically similar limitations. To address this, we introduce a pointwise evaluation protocol that uses an LLM-as-a-Judge to measure coverage more accurately. Experiments show that LimAgents substantially improve performance. The RAG + multi-agent GPT-4o mini configuration achieves a +15.51% coverage gain over zero-shot baselines, while the Llama 3 8B multi-agent setup yields a +4.41% improvement. 

---
# Concept Attractors in LLMs and their Applications 

**Authors**: Sotirios Panagiotis Chytas, Vikas Singh  

**Link**: [PDF](https://arxiv.org/pdf/2601.11575)  

**Abstract**: Large language models (LLMs) often map semantically related prompts to similar internal representations at specific layers, even when their surface forms differ widely. We show that this behavior can be explained through Iterated Function Systems (IFS), where layers act as contractive mappings toward concept-specific Attractors. We leverage this insight and develop simple, training-free methods that operate directly on these Attractors to solve a wide range of practical tasks, including language translation, hallucination reduction, guardrailing, and synthetic data generation. Despite their simplicity, these Attractor-based interventions match or exceed specialized baselines, offering an efficient alternative to heavy fine-tuning, generalizable in scenarios where baselines underperform. 

---
# An Empirical Analysis of Fine-Tuning Large Language Models on Bioinformatics Literature: PRSGPT and BioStarsGPT 

**Authors**: Muhammad Muneeb, David B. Ascher  

**Link**: [PDF](https://arxiv.org/pdf/2601.11573)  

**Abstract**: Large language models (LLMs) often lack specialized knowledge for complex bioinformatics applications. We present a reproducible pipeline for fine-tuning LLMs on specialized bioinformatics data, demonstrated through two use cases: PRSGPT, focused on polygenic risk score (PRS) tools, and BioStarsGPT, trained on community forum discussions. The nine-step pipeline integrates diverse data sources, structured preprocessing, prompt-based question-answer (QA) generation (via Google Gemini), natural language inference (NLI) for quality control, semantic deduplication, clustering-based data splitting, and parameter-efficient fine-tuning using LoRA. We fine-tuned three LLMs (LLaMA-3.2-3B, Qwen2.5-7B, Gemma) and benchmarked them on over 14 lexical and semantic metrics. Qwen2.5-7B emerged as the best performer, with BLEU-4 and ROUGE-1 improvements of 82\% and 70\% for PRSGPT and 6\% and 18\% for BioStarsGPT, respectively. The open-source datasets produced include over 28,000 QA pairs for PRSGPT and 154,282 for BioStarsGPT. Human evaluation of PRSGPT yielded 61.9\% accuracy on the PRS tools comparison task, comparable to Google Gemini (61.4\%), but with richer methodological detail and accurate citations. BioStarsGPT demonstrated 59\% conceptual accuracy across 142 curated bioinformatics questions. Our pipeline enables scalable, domain-specific fine-tuning of LLMs. It enables privacy-preserving, locally deployable bioinformatics assistants, explores their practical applications, and addresses the challenges, limitations, and mitigation strategies associated with their development and use. 

---
# Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology 

**Authors**: Vanessa D'Amario, Randy Daniel, Alessandro Zanetti, Dhruv Edamadaka, Nitya Alaparthy, Joshua Tarkoff  

**Link**: [PDF](https://arxiv.org/pdf/2601.11567)  

**Abstract**: Small open-source medical large language models (LLMs) offer promising opportunities for low-resource deployment and broader accessibility. However, their evaluation is often limited to accuracy on medical multiple choice question (MCQ) benchmarks, and lacks evaluation of consistency, robustness, or reasoning behavior. We use MCQ coupled to human evaluation and clinical review to assess six small open-source medical LLMs (HuatuoGPT-o1 (Chen 2024), Diabetica-7B, Diabetica-o1 (Wei 2024), Meditron3-8B (Sallinen2025), MedFound-7B (Liu 2025), and ClinicaGPT-base-zh (Wang 2023)) in pediatric endocrinology. In deterministic settings, we examine the effect of prompt variation on models' output and self-assessment bias. In stochastic settings, we evaluate output variability and investigate the relationship between consistency and correctness. HuatuoGPT-o1-8B achieved the highest performance. The results show that high consistency across the model response is not an indicator of correctness, although HuatuoGPT-o1-8B showed the highest consistency rate. When tasked with selecting correct reasoning, both HuatuoGPT-o1-8B and Diabetica-o1 exhibit self-assessment bias and dependency on the order of the candidate explanations. Expert review of incorrect reasoning rationales identified a mix of clinically acceptable responses and clinical oversight. We further show that system-level perturbations, such as differences in CUDA builds, can yield statistically significant shifts in model output despite stable accuracy. This work demonstrates that small, semantically negligible prompt perturbations lead to divergent outputs, raising concerns about reproducibility of LLM-based evaluations and highlights the output variability under different stochastic regimes, emphasizing the need of a broader diagnostic framework to understand potential pitfalls in real-world clinical decision support scenarios. 

---
# Compass-Embedding v4: Robust Contrastive Learning for Multilingual E-commerce Embeddings 

**Authors**: Pakorn Ueareeworakul, Shuman Liu, Jinghao Feng, Ling Hu, Zhantang Shi, Chengqi Sun, Liang Yao, Panyi Ouyang, Haibo Zhang, Anxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2601.11565)  

**Abstract**: As global e-commerce rapidly expands into emerging markets, the lack of high-quality semantic representations for low-resource languages has become a decisive bottleneck for retrieval, recommendation, and search systems. In this work, we present Compass-Embedding v4, a high-efficiency multilingual embedding framework specifically optimized for Southeast Asian (SEA) e-commerce scenarios, where data scarcity, noisy supervision, and strict production constraints jointly challenge representation learning. Compass-Embedding v4 addresses three core challenges. First, large-batch contrastive training under mixed task supervision introduces systematic false negatives that degrade semantic alignment. We propose Class-Aware Masking (CAM), a lightweight modification to the InfoNCE objective that suppresses invalid in-batch negatives and improves semantic discrimination without altering training efficiency. Second, low-resource SEA languages suffer from limited and uneven data coverage. We construct a diversified training corpus through context-grounded synthetic data generation, cross-lingual translation, and structured e-commerce data construction, enabling robust multilingual and domain-specific learning. Third, production deployment requires high-throughput inference while preserving embedding quality. We combine robustness-driven large-batch training with spherical model merging to mitigate catastrophic forgetting, and optimize inference via vLLM and FP8 quantization. Extensive evaluations across multilingual benchmarks and proprietary e-commerce tasks show that Compass-Embedding v4 achieves state-of-the-art performance on major SEA languages, significantly outperforming general-purpose embedding models in domain-specific retrieval and classification, while maintaining competitive performance on high-resource languages. 

---
# Context Discipline and Performance Correlation: Analyzing LLM Performance and Quality Degradation Under Varying Context Lengths 

**Authors**: Ahilan Ayyachamy Nadar Ponnusamy, Karthic Chandran, M Maruf Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2601.11564)  

**Abstract**: The scaling trend in Large Language Models (LLMs) has prioritized increasing the maximum context window to facilitate complex, long-form reasoning and document analysis. However, managing this expanded context introduces severe computational overhead. This paper investigates the critical trade-off between system performance and model quality when dense transformer architectures--specifically Llama-3.1-70B and Qwen1.5-14B--are exposed to large volumes of irrelevant and distracting context. The research identifies a non-linear performance degradation tied to the growth of the Key-Value (KV) cache. Furthermore, an extended analysis of the Mixture-of-Experts (MoE) architecture reveals unique behavioral anomalies at varying context scales, suggesting that architectural benefits may be masked by infrastructure bottlenecks at high token volumes. 

---
# Jet-RL: Enabling On-Policy FP8 Reinforcement Learning with Unified Training and Rollout Precision Flow 

**Authors**: Haocheng Xi, Charlie Ruan, Peiyuan Liao, Yujun Lin, Han Cai, Yilong Zhao, Shuo Yang, Kurt Keutzer, Song Han, Ligeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14243)  

**Abstract**: Reinforcement learning (RL) is essential for enhancing the complex reasoning capabilities of large language models (LLMs). However, existing RL training pipelines are computationally inefficient and resource-intensive, with the rollout phase accounting for over 70% of total training time. Quantized RL training, particularly using FP8 precision, offers a promising approach to mitigating this bottleneck. A commonly adopted strategy applies FP8 precision during rollout while retaining BF16 precision for training. In this work, we present the first comprehensive study of FP8 RL training and demonstrate that the widely used BF16-training + FP8-rollout strategy suffers from severe training instability and catastrophic accuracy collapse under long-horizon rollouts and challenging tasks. Our analysis shows that these failures stem from the off-policy nature of the approach, which introduces substantial numerical mismatch between training and inference. Motivated by these observations, we propose Jet-RL, an FP8 RL training framework that enables robust and stable RL optimization. The key idea is to adopt a unified FP8 precision flow for both training and rollout, thereby minimizing numerical discrepancies and eliminating the need for inefficient inter-step calibration. Extensive experiments validate the effectiveness of Jet-RL: our method achieves up to 33% speedup in the rollout phase, up to 41% speedup in the training phase, and a 16% end-to-end speedup over BF16 training, while maintaining stable convergence across all settings and incurring negligible accuracy degradation. 

---
# Generalization and Completeness of Stochastic Local Search Algorithms 

**Authors**: Daniel Loscos, Narciso Marti-Oliet, Ismael Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2601.14212)  

**Abstract**: We generalize Stochastic Local Search (SLS) heuristics into a unique formal model. This model has two key components: a common structure designed to be as large as possible and a parametric structure intended to be as small as possible. Each heuristic is obtained by instantiating the parametric part in a different way. Particular instances for Genetic Algorithms (GA), Ant Colony Optimization (ACO), and Particle Swarm Optimization (PSO) are presented. Then, we use our model to prove the Turing-completeness of SLS algorithms in general. The proof uses our framework to construct a GA able to simulate any Turing machine. This Turing-completeness implies that determining any non-trivial property concerning the relationship between the inputs and the computed outputs is undecidable for GA and, by extension, for the general set of SLS methods (although not necessarily for each particular method). Similar proofs are more informally presented for PSO and ACO. 

---
# InT: Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning 

**Authors**: Matthew Y. R. Yang, Hao Bai, Ian Wu, Gene Yang, Amrith Setlur, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.14209)  

**Abstract**: Outcome-reward reinforcement learning (RL) has proven effective at improving the reasoning capabilities of large language models (LLMs). However, standard RL assigns credit only at the level of the final answer, penalizing entire reasoning traces when the outcome is incorrect and uniformly reinforcing all steps when it is correct. As a result, correct intermediate steps may be discouraged in failed traces, while spurious steps may be reinforced in successful ones. We refer to this failure mode as the problem of credit assignment. While a natural remedy is to train a process reward model, accurately optimizing such models to identify corrective reasoning steps remains challenging. We introduce Intervention Training (InT), a training paradigm in which the model performs fine-grained credit assignment on its own reasoning traces by proposing short, targeted corrections that steer trajectories toward higher reward. Using reference solutions commonly available in mathematical reasoning datasets and exploiting the fact that verifying a model-generated solution is easier than generating a correct one from scratch, the model identifies the first error in its reasoning and proposes a single-step intervention to redirect the trajectory toward the correct solution. We then apply supervised fine-tuning (SFT) to the on-policy rollout up to the point of error concatenated with the intervention, localizing error to the specific step that caused failure. We show that the resulting model serves as a far better initialization for RL training. After running InT and subsequent fine-tuning with RL, we improve accuracy by nearly 14% over a 4B-parameter base model on IMO-AnswerBench, outperforming larger open-source models such as gpt-oss-20b. 

---
# Toward Efficient Agents: Memory, Tool learning, and Planning 

**Authors**: Xiaofang Yang, Lijun Li, Heng Zhou, Tong Zhu, Xiaoye Qu, Yuchen Fan, Qianshan Wei, Rui Ye, Li Kang, Yiran Qin, Zhiqiang Kou, Daizong Liu, Qi Li, Ning Ding, Siheng Chen, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2601.14192)  

**Abstract**: Recent years have witnessed increasing interest in extending large language models into agentic systems. While the effectiveness of agents has continued to improve, efficiency, which is crucial for real-world deployment, has often been overlooked. This paper therefore investigates efficiency from three core components of agents: memory, tool learning, and planning, considering costs such as latency, tokens, steps, etc. Aimed at conducting comprehensive research addressing the efficiency of the agentic system itself, we review a broad range of recent approaches that differ in implementation yet frequently converge on shared high-level principles including but not limited to bounding context via compression and management, designing reinforcement learning rewards to minimize tool invocation, and employing controlled search mechanisms to enhance efficiency, which we discuss in detail. Accordingly, we characterize efficiency in two complementary ways: comparing effectiveness under a fixed cost budget, and comparing cost at a comparable level of effectiveness. This trade-off can also be viewed through the Pareto frontier between effectiveness and cost. From this perspective, we also examine efficiency oriented benchmarks by summarizing evaluation protocols for these components and consolidating commonly reported efficiency metrics from both benchmark and methodological studies. Moreover, we discuss the key challenges and future directions, with the goal of providing promising insights. 

---
# A model of errors in transformers 

**Authors**: Suvrat Raju, Praneeth Netrapalli  

**Link**: [PDF](https://arxiv.org/pdf/2601.14175)  

**Abstract**: We study the error rate of LLMs on tasks like arithmetic that require a deterministic output, and repetitive processing of tokens drawn from a small set of alternatives. We argue that incorrect predictions arise when small errors in the attention mechanism accumulate to cross a threshold, and use this insight to derive a quantitative two-parameter relationship between the accuracy and the complexity of the task. The two parameters vary with the prompt and the model; they can be interpreted in terms of an elementary noise rate, and the number of plausible erroneous tokens that can be predicted. Our analysis is inspired by an ``effective field theory'' perspective: the LLM's many raw parameters can be reorganized into just two parameters that govern the error rate. We perform extensive empirical tests, using Gemini 2.5 Flash, Gemini 2.5 Pro and DeepSeek R1, and find excellent agreement between the predicted and observed accuracy for a variety of tasks, although we also identify deviations in some cases. Our model provides an alternative to suggestions that errors made by LLMs on long repetitive tasks indicate the ``collapse of reasoning'', or an inability to express ``compositional'' functions. Finally, we show how to construct prompts to reduce the error rate. 

---
# The Side Effects of Being Smart: Safety Risks in MLLMs' Multi-Image Reasoning 

**Authors**: Renmiao Chen, Yida Lu, Shiyao Cui, Xuan Ouyang, Victor Shea-Jay Huang, Shumin Zhang, Chengwei Pan, Han Qiu, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14127)  

**Abstract**: As Multimodal Large Language Models (MLLMs) acquire stronger reasoning capabilities to handle complex, multi-image instructions, this advancement may pose new safety risks. We study this problem by introducing MIR-SafetyBench, the first benchmark focused on multi-image reasoning safety, which consists of 2,676 instances across a taxonomy of 9 multi-image relations. Our extensive evaluations on 19 MLLMs reveal a troubling trend: models with more advanced multi-image reasoning can be more vulnerable on MIR-SafetyBench. Beyond attack success rates, we find that many responses labeled as safe are superficial, often driven by misunderstanding or evasive, non-committal replies. We further observe that unsafe generations exhibit lower attention entropy than safe ones on average. This internal signature suggests a possible risk that models may over-focus on task solving while neglecting safety constraints. Our code and data are available at this https URL. 

---
# DermaBench: A Clinician-Annotated Benchmark Dataset for Dermatology Visual Question Answering and Reasoning 

**Authors**: Abdurrahim Yilmaz, Ozan Erdem, Ece Gokyayla, Ayda Acar, Burc Bugra Dagtas, Dilara Ilhan Erdil, Gulsum Gencoglan, Burak Temelkuran  

**Link**: [PDF](https://arxiv.org/pdf/2601.14084)  

**Abstract**: Vision-language models (VLMs) are increasingly important in medical applications; however, their evaluation in dermatology remains limited by datasets that focus primarily on image-level classification tasks such as lesion recognition. While valuable for recognition, such datasets cannot assess the full visual understanding, language grounding, and clinical reasoning capabilities of multimodal models. Visual question answering (VQA) benchmarks are required to evaluate how models interpret dermatological images, reason over fine-grained morphology, and generate clinically meaningful descriptions. We introduce DermaBench, a clinician-annotated dermatology VQA benchmark built on the Diverse Dermatology Images (DDI) dataset. DermaBench comprises 656 clinical images from 570 unique patients spanning Fitzpatrick skin types I-VI. Using a hierarchical annotation schema with 22 main questions (single-choice, multi-choice, and open-ended), expert dermatologists annotated each image for diagnosis, anatomic site, lesion morphology, distribution, surface features, color, and image quality, together with open-ended narrative descriptions and summaries, yielding approximately 14.474 VQA-style annotations. DermaBench is released as a metadata-only dataset to respect upstream licensing and is publicly available at Harvard Dataverse. 

---
# Chain-of-Thought Compression Should Not Be Blind: V-Skip for Efficient Multimodal Reasoning via Dual-Path Anchoring 

**Authors**: Dongxu Zhang, Yiding Sun, Cheng Tan, Wenbiao Yan, Ning Yang, Jihua Zhu, Hiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13879)  

**Abstract**: While Chain-of-Thought (CoT) reasoning significantly enhances the performance of Multimodal Large Language Models (MLLMs), its autoregressive nature incurs prohibitive latency constraints. Current efforts to mitigate this via token compression often fail by blindly applying text-centric metrics to multimodal contexts. We identify a critical failure mode termed Visual Amnesia, where linguistically redundant tokens are erroneously pruned, leading to hallucinations. To address this, we introduce V-Skip that reformulates token pruning as a Visual-Anchored Information Bottleneck (VA-IB) optimization problem. V-Skip employs a dual-path gating mechanism that weighs token importance through both linguistic surprisal and cross-modal attention flow, effectively rescuing visually salient anchors. Extensive experiments on Qwen2-VL and Llama-3.2 families demonstrate that V-Skip achieves a $2.9\times$ speedup with negligible accuracy loss. Specifically, it preserves fine-grained visual details, outperforming other baselines over 30\% on the DocVQA. 

---
# Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance 

**Authors**: Mostapha Benhenda  

**Link**: [PDF](https://arxiv.org/pdf/2601.13770)  

**Abstract**: We introduce Look-Ahead-Bench, a standardized benchmark measuring look-ahead bias in Point-in-Time (PiT) Large Language Models (LLMs) within realistic and practical financial workflows. Unlike most existing approaches that primarily test inner lookahead knowledge via Q\\&A, our benchmark evaluates model behavior in practical scenarios. To distinguish genuine predictive capability from memorization-based performance, we analyze performance decay across temporally distinct market regimes, incorporating several quantitative baselines to establish performance thresholds. We evaluate prominent open-source LLMs -- Llama 3.1 (8B and 70B) and DeepSeek 3.2 -- against a family of Point-in-Time LLMs (Pitinf-Small, Pitinf-Medium, and frontier-level model Pitinf-Large) from PiT-Inference. Results reveal significant lookahead bias in standard LLMs, as measured with alpha decay, unlike Pitinf models, which demonstrate improved generalization and reasoning abilities as they scale in size. This work establishes a foundation for the standardized evaluation of temporal bias in financial LLMs and provides a practical framework for identifying models suitable for real-world deployment. Code is available on GitHub: this https URL 

---
# DARC: Decoupled Asymmetric Reasoning Curriculum for LLM Evolution 

**Authors**: Shengda Fan, Xuyan Ye, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.13761)  

**Abstract**: Self-play with large language models has emerged as a promising paradigm for achieving self-improving artificial intelligence. However, existing self-play frameworks often suffer from optimization instability, due to (i) non-stationary objectives induced by solver-dependent reward feedback for the Questioner, and (ii) bootstrapping errors from self-generated pseudo-labels used to supervise the Solver. To mitigate these challenges, we introduce DARC (Decoupled Asymmetric Reasoning Curriculum), a two-stage framework that stabilizes the self-evolution process. First, we train the Questioner to synthesize difficulty-calibrated questions, conditioned on explicit difficulty levels and external corpora. Second, we train the Solver with an asymmetric self-distillation mechanism, where a document-augmented teacher generates high-quality pseudo-labels to supervise the student Solver that lacks document access. Empirical results demonstrate that DARC is model-agnostic, yielding an average improvement of 10.9 points across nine reasoning benchmarks and three backbone models. Moreover, DARC consistently outperforms all baselines and approaches the performance of fully supervised models without relying on human this http URL code is available at this https URL. 

---
# Finding RELIEF: Shaping Reasoning Behavior without Reasoning Supervision via Belief Engineering 

**Authors**: Chak Tou Leong, Dingwei Chen, Heming Xia, Qingyu Yin, Sunbowen Lee, Jian Wang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.13752)  

**Abstract**: Large reasoning models (LRMs) have achieved remarkable success in complex problem-solving, yet they often suffer from computational redundancy or reasoning unfaithfulness. Current methods for shaping LRM behavior typically rely on reinforcement learning or fine-tuning with gold-standard reasoning traces, a paradigm that is both computationally expensive and difficult to scale. In this paper, we reveal that LRMs possess latent \textit{reasoning beliefs} that internally track their own reasoning traits, which can be captured through simple logit probing. Building upon this insight, we propose Reasoning Belief Engineering (RELIEF), a simple yet effective framework that shapes LRM behavior by aligning the model's self-concept with a target belief blueprint. Crucially, RELIEF completely bypasses the need for reasoning-trace supervision. It internalizes desired traits by fine-tuning on synthesized, self-reflective question-answering pairs that affirm the target belief. Extensive experiments on efficiency and faithfulness tasks demonstrate that RELIEF matches or outperforms behavior-supervised and preference-based baselines while requiring lower training costs. Further analysis validates that shifting a model's reasoning belief effectively shapes its actual behavior. 

---
# Who Should Have Surgery? A Comparative Study of GenAI vs Supervised ML for CRS Surgical Outcome Prediction 

**Authors**: Sayeed Shafayet Chowdhury, Snehasis Mukhopadhyay, Shiaofen Fang, Vijay R. Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2601.13710)  

**Abstract**: Artificial intelligence has reshaped medical imaging, yet the use of AI on clinical data for prospective decision support remains limited. We study pre-operative prediction of clinically meaningful improvement in chronic rhinosinusitis (CRS), defining success as a more than 8.9-point reduction in SNOT-22 at 6 months (MCID). In a prospectively collected cohort where all patients underwent surgery, we ask whether models using only pre-operative clinical data could have identified those who would have poor outcomes, i.e. those who should have avoided surgery. We benchmark supervised ML (logistic regression, tree ensembles, and an in-house MLP) against generative AI (ChatGPT, Claude, Gemini, Perplexity), giving each the same structured inputs and constraining outputs to binary recommendations with confidence. Our best ML model (MLP) achieves 85 % accuracy with superior calibration and decision-curve net benefit. GenAI models underperform on discrimination and calibration across zero-shot setting. Notably, GenAI justifications align with clinician heuristics and the MLP's feature importance, repeatedly highlighting baseline SNOT-22, CT/endoscopy severity, polyp phenotype, and physchology/pain comorbidities. We provide a reproducible tabular-to-GenAI evaluation protocol and subgroup analyses. Findings support an ML-first, GenAI- augmented workflow: deploy calibrated ML for primary triage of surgical candidacy, with GenAI as an explainer to enhance transparency and shared decision-making. 

---
# Hidden in Plain Text: Measuring LLM Deception Quality Against Human Baselines Using Social Deduction Games 

**Authors**: Christopher Kao, Vanshika Vats, James Davis  

**Link**: [PDF](https://arxiv.org/pdf/2601.13709)  

**Abstract**: Large Language Model (LLM) agents are increasingly used in many applications, raising concerns about their safety. While previous work has shown that LLMs can deceive in controlled tasks, less is known about their ability to deceive using natural language in social contexts. In this paper, we study deception in the Social Deduction Game (SDG) Mafia, where success is dependent on deceiving others through conversation. Unlike previous SDG studies, we use an asynchronous multi-agent framework which better simulates realistic social contexts. We simulate 35 Mafia games with GPT-4o LLM agents. We then create a Mafia Detector using GPT-4-Turbo to analyze game transcripts without player role information to predict the mafia players. We use prediction accuracy as a surrogate marker for deception quality. We compare this prediction accuracy to that of 28 human games and a random baseline. Results show that the Mafia Detector's mafia prediction accuracy is lower on LLM games than on human games. The result is consistent regardless of the game days and the number of mafias detected. This indicates that LLMs blend in better and thus deceive more effectively. We also release a dataset of LLM Mafia transcripts to support future research. Our findings underscore both the sophistication and risks of LLM deception in social contexts. 

---
# DSAEval: Evaluating Data Science Agents on a Wide Range of Real-World Data Science Problems 

**Authors**: Maojun Sun, Yifei Xie, Yue Wu, Ruijian Han, Binyan Jiang, Defeng Sun, Yancheng Yuan, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13591)  

**Abstract**: Recent LLM-based data agents aim to automate data science tasks ranging from data analysis to deep learning. However, the open-ended nature of real-world data science problems, which often span multiple taxonomies and lack standard answers, poses a significant challenge for evaluation. To address this, we introduce DSAEval, a benchmark comprising 641 real-world data science problems grounded in 285 diverse datasets, covering both structured and unstructured data (e.g., vision and text). DSAEval incorporates three distinctive features: (1) Multimodal Environment Perception, which enables agents to interpret observations from multiple modalities including text and vision; (2) Multi-Query Interactions, which mirror the iterative and cumulative nature of real-world data science projects; and (3) Multi-Dimensional Evaluation, which provides a holistic assessment across reasoning, code, and results. We systematically evaluate 11 advanced agentic LLMs using DSAEval. Our results show that Claude-Sonnet-4.5 achieves the strongest overall performance, GPT-5.2 is the most efficient, and MiMo-V2-Flash is the most cost-effective. We further demonstrate that multimodal perception consistently improves performance on vision-related tasks, with gains ranging from 2.04% to 11.30%. Overall, while current data science agents perform well on structured data and routine data anlysis workflows, substantial challenges remain in unstructured domains. Finally, we offer critical insights and outline future research directions to advance the development of data science agents. 

---
# Self-Improvement as Coherence Optimization: A Theoretical Account 

**Authors**: Tianyi Qiu, Ahmed Hani Ismail, Zhonghao He, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.13566)  

**Abstract**: Can language models improve their accuracy without external supervision? Methods such as debate, bootstrap, and internal coherence maximization achieve this surprising feat, even matching golden finetuning performance. Yet why they work remains theoretically unclear. We show that they are all special cases of coherence optimization: finding a context-to-behavior mapping that's most compressible and jointly predictable. We prove that coherence optimization is equivalent to description-length regularization, and that among all such regularization schemes, it is optimal for semi-supervised learning when the regularizer is derived from a pretrained model. Our theory, supported by preliminary experiments, explains why feedback-free self-improvement works and predicts when it should succeed or fail. 

---
# Leveraging ChatGPT and Other NLP Methods for Identifying Risk and Protective Behaviors in MSM: Social Media and Dating apps Text Analysis 

**Authors**: Mehrab Beikzadeh, Chenglin Hong, Cory J Cascalheira, Callisto Boka, Majid Sarrafzadeh, Ian W Holloway  

**Link**: [PDF](https://arxiv.org/pdf/2601.13558)  

**Abstract**: Men who have sex with men (MSM) are at elevated risk for sexually transmitted infections and harmful drinking compared to heterosexual men. Text data collected from social media and dating applications may provide new opportunities for personalized public health interventions by enabling automatic identification of risk and protective behaviors. In this study, we evaluated whether text from social media and dating apps can be used to predict sexual risk behaviors, alcohol use, and pre-exposure prophylaxis (PrEP) uptake among MSM. With participant consent, we collected textual data and trained machine learning models using features derived from ChatGPT embeddings, BERT embeddings, LIWC, and a dictionary-based risk term approach. The models achieved strong performance in predicting monthly binge drinking and having more than five sexual partners, with F1 scores of 0.78, and moderate performance in predicting PrEP use and heavy drinking, with F1 scores of 0.64 and 0.63. These findings demonstrate that social media and dating app text data can provide valuable insights into risk and protective behaviors and highlight the potential of large language model-based methods to support scalable and personalized public health interventions for MSM. 

---
# Eliciting Harmful Capabilities by Fine-Tuning On Safeguarded Outputs 

**Authors**: Jackson Kaunismaa, Avery Griffin, John Hughes, Christina Q. Knight, Mrinank Sharma, Erik Jones  

**Link**: [PDF](https://arxiv.org/pdf/2601.13528)  

**Abstract**: Model developers implement safeguards in frontier models to prevent misuse, for example, by employing classifiers to filter dangerous outputs. In this work, we demonstrate that even robustly safeguarded models can be used to elicit harmful capabilities in open-source models through elicitation attacks. Our elicitation attacks consist of three stages: (i) constructing prompts in adjacent domains to a target harmful task that do not request dangerous information; (ii) obtaining responses to these prompts from safeguarded frontier models; (iii) fine-tuning open-source models on these prompt-output pairs. Since the requested prompts cannot be used to directly cause harm, they are not refused by frontier model safeguards. We evaluate these elicitation attacks within the domain of hazardous chemical synthesis and processing, and demonstrate that our attacks recover approximately 40% of the capability gap between the base open-source model and an unrestricted frontier model. We then show that the efficacy of elicitation attacks scales with the capability of the frontier model and the amount of generated fine-tuning data. Our work demonstrates the challenge of mitigating ecosystem level risks with output-level safeguards. 

---
# The Hidden Toll of Social Media News: Causal Effects on Psychosocial Wellbeing 

**Authors**: Olivia Pal, Agam Goyal, Eshwar Chandrasekharan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2601.13487)  

**Abstract**: News consumption on social media has become ubiquitous, yet how different forms of engagement shape psychosocial outcomes remains unclear. To address this gap, we leveraged a large-scale dataset of ~26M posts and ~45M comments on the BlueSky platform, and conducted a quasi-experimental study, matching 81,345 Treated users exposed to News feeds with 83,711 Control users using stratified propensity score analysis. We examined psychosocial wellbeing, in terms of affective, behavioral, and cognitive outcomes. Our findings reveal that news engagement produces systematic trade-offs: increased depression, stress, and anxiety, yet decreased loneliness and increased social interaction on the platform. Regression models reveal that News feed bookmarking is associated with greater psychosocial deterioration compared to commenting or quoting, with magnitude differences exceeding tenfold. These per-engagement effects accumulate with repeated exposure, showing significant psychosocial impacts. Our work extends theories of news effects beyond crisis-centric frameworks by demonstrating that routine consumption creates distinct psychological dynamics depending on engagement type, and bears implications for tools and interventions for mitigating the psychosocial costs of news consumption on social media. 

---
# From Completion to Editing: Unlocking Context-Aware Code Infilling via Search-and-Replace Instruction Tuning 

**Authors**: Jiajun Zhang, Zeyu Cui, Jiaxi Yang, Lei Zhang, Yuheng Jing, Zeyao Ma, Tianyi Bai, Zilei Wang, Qiang Liu, Liang Wang, Binyuan Hui, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.13384)  

**Abstract**: The dominant Fill-in-the-Middle (FIM) paradigm for code completion is constrained by its rigid inability to correct contextual errors and reliance on unaligned, insecure Base models. While Chat LLMs offer safety and Agentic workflows provide flexibility, they suffer from performance degradation and prohibitive latency, respectively. To resolve this dilemma, we propose Search-and-Replace Infilling (SRI), a framework that internalizes the agentic verification-and-editing mechanism into a unified, single-pass inference process. By structurally grounding edits via an explicit search phase, SRI harmonizes completion tasks with the instruction-following priors of Chat LLMs, extending the paradigm from static infilling to dynamic context-aware editing. We synthesize a high-quality dataset, SRI-200K, and fine-tune the SRI-Coder series. Extensive evaluations demonstrate that with minimal data (20k samples), SRI-Coder enables Chat models to surpass the completion performance of their Base counterparts. Crucially, unlike FIM-style tuning, SRI preserves general coding competencies and maintains inference latency comparable to standard FIM. We empower the entire Qwen3-Coder series with SRI, encouraging the developer community to leverage this framework for advanced auto-completion and assisted development. 

---
# On the Relation of State Space Models and Hidden Markov Models 

**Authors**: Aydin Ghojogh, M.Hadi Sepanj, Benyamin Ghojogh  

**Link**: [PDF](https://arxiv.org/pdf/2601.13357)  

**Abstract**: State Space Models (SSMs) and Hidden Markov Models (HMMs) are foundational frameworks for modeling sequential data with latent variables and are widely used in signal processing, control theory, and machine learning. Despite their shared temporal structure, they differ fundamentally in the nature of their latent states, probabilistic assumptions, inference procedures, and training paradigms. Recently, deterministic state space models have re-emerged in natural language processing through architectures such as S4 and Mamba, raising new questions about the relationship between classical probabilistic SSMs, HMMs, and modern neural sequence models.
In this paper, we present a unified and systematic comparison of HMMs, linear Gaussian state space models, Kalman filtering, and contemporary NLP state space models. We analyze their formulations through the lens of probabilistic graphical models, examine their inference algorithms -- including forward-backward inference and Kalman filtering -- and contrast their learning procedures via Expectation-Maximization and gradient-based optimization. By highlighting both structural similarities and semantic differences, we clarify when these models are equivalent, when they fundamentally diverge, and how modern NLP SSMs relate to classical probabilistic models. Our analysis bridges perspectives from control theory, probabilistic modeling, and modern deep learning. 

---
# CooperBench: Why Coding Agents Cannot be Your Teammates Yet 

**Authors**: Arpandeep Khatua, Hao Zhu, Peter Tran, Arya Prabhudesai, Frederic Sadrieh, Johann K. Lieberwirth, Xinkai Yu, Yicheng Fu, Michael J. Ryan, Jiaxin Pei, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13295)  

**Abstract**: Resolving team conflicts requires not only task-specific competence, but also social intelligence to find common ground and build consensus. As AI agents increasingly collaborate on complex work, they must develop coordination capabilities to function as effective teammates. Yet we hypothesize that current agents lack these capabilities. To test this, we introduce CooperBench, a benchmark of over 600 collaborative coding tasks across 12 libraries in 4 programming languages. Each task assigns two agents different features that can be implemented independently but may conflict without proper coordination. Tasks are grounded in real open-source repositories with expert-written tests. Evaluating state-of-the-art coding agents, we observe the curse of coordination: agents achieve on average 30% lower success rates when working together compared to performing both tasks individually. This contrasts sharply with human teams, where adding teammates typically improves productivity. Our analysis reveals three key issues: (1) communication channels become jammed with vague, ill-timed, and inaccurate messages; (2) even with effective communication, agents deviate from their commitments; and (3) agents often hold incorrect expectations about others' plans and communication. Through large-scale simulation, we also observe rare but interesting emergent coordination behavior including role division, resource division, and negotiation. Our research presents a novel benchmark for collaborative coding and calls for a shift from pursuing individual agent capability to developing social intelligence. 

---
# CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning 

**Authors**: Eric Onyame, Akash Ghosh, Subhadip Baidya, Sriparna Saha, Xiuying Chen, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2601.13262)  

**Abstract**: While large language models (LLMs) have shown to perform well on monolingual mathematical and commonsense reasoning, they remain unreliable for multilingual medical reasoning applications, hindering their deployment in multilingual healthcare settings. We address this by first introducing CUREMED-BENCH, a high-quality multilingual medical reasoning dataset with open-ended reasoning queries with a single verifiable answer, spanning thirteen languages, including underrepresented languages such as Amharic, Yoruba, and Swahili. Building on this dataset, we propose CURE-MED, a curriculum-informed reinforcement learning framework that integrates code-switching-aware supervised fine-tuning and Group Relative Policy Optimization to jointly improve logical correctness and language stability. Across thirteen languages, our approach consistently outperforms strong baselines and scales effectively, achieving 85.21% language consistency and 54.35% logical correctness at 7B parameters, and 94.96% language consistency and 70.04% logical correctness at 32B parameters. These results support reliable and equitable multilingual medical reasoning in LLMs. The code and dataset are available at this https URL 

---
# KOCO-BENCH: Can Large Language Models Leverage Domain Knowledge in Software Development? 

**Authors**: Xue Jiang, Jiaru Qian, Xianjie Shi, Chenjie Li, Hao Zhu, Ziyu Wang, Jielun Zhang, Zheyu Zhao, Kechi Zhang, Jia Li, Wenpin Jiao, Zhi Jin, Ge Li, Yihong Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.13240)  

**Abstract**: Large language models (LLMs) excel at general programming but struggle with domain-specific software development, necessitating domain specialization methods for LLMs to learn and utilize domain knowledge and data. However, existing domain-specific code benchmarks cannot evaluate the effectiveness of domain specialization methods, which focus on assessing what knowledge LLMs possess rather than how they acquire and apply new knowledge, lacking explicit knowledge corpora for developing domain specialization methods. To this end, we present KOCO-BENCH, a novel benchmark designed for evaluating domain specialization methods in real-world software development. KOCO-BENCH contains 6 emerging domains with 11 software frameworks and 25 projects, featuring curated knowledge corpora alongside multi-granularity evaluation tasks including domain code generation (from function-level to project-level with rigorous test suites) and domain knowledge understanding (via multiple-choice Q&A). Unlike previous benchmarks that only provide test sets for direct evaluation, KOCO-BENCH requires acquiring and applying diverse domain knowledge (APIs, rules, constraints, etc.) from knowledge corpora to solve evaluation tasks. Our evaluations reveal that KOCO-BENCH poses significant challenges to state-of-the-art LLMs. Even with domain specialization methods (e.g., SFT, RAG, kNN-LM) applied, improvements remain marginal. Best-performing coding agent, Claude Code, achieves only 34.2%, highlighting the urgent need for more effective domain specialization methods. We release KOCO-BENCH, evaluation code, and baselines to advance further research at this https URL. 

---
# RubRIX: Rubric-Driven Risk Mitigation in Caregiver-AI Interactions 

**Authors**: Drishti Goel, Jeongah Lee, Qiuyue Joy Zhong, Violeta J. Rodriguez, Daniel S. Brown, Ravi Karkar, Dong Whi Yoo, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2601.13235)  

**Abstract**: Caregivers seeking AI-mediated support express complex needs -- information-seeking, emotional validation, and distress cues -- that warrant careful evaluation of response safety and appropriateness. Existing AI evaluation frameworks, primarily focused on general risks (toxicity, hallucinations, policy violations, etc), may not adequately capture the nuanced risks of LLM-responses in caregiving-contexts. We introduce RubRIX (Rubric-based Risk Index), a theory-driven, clinician-validated framework for evaluating risks in LLM caregiving responses. Grounded in the Elements of an Ethic of Care, RubRIX operationalizes five empirically-derived risk dimensions: Inattention, Bias & Stigma, Information Inaccuracy, Uncritical Affirmation, and Epistemic Arrogance. We evaluate six state-of-the-art LLMs on over 20,000 caregiver queries from Reddit and ALZConnected. Rubric-guided refinement consistently reduced risk-components by 45-98% after one iteration across models. This work contributes a methodological approach for developing domain-sensitive, user-centered evaluation frameworks for high-burden contexts. Our findings highlight the importance of domain-sensitive, interactional risk evaluation for the responsible deployment of LLMs in caregiving support contexts. We release benchmark datasets to enable future research on contextual risk evaluation in AI-mediated support. 

---
# TVWorld: Foundations for Remote-Control TV Agents 

**Authors**: Zhantao Ma, Quanfeng Lu, Shuai Zhong, Dahai Yu, Ping Luo, Michael K. Ng  

**Link**: [PDF](https://arxiv.org/pdf/2601.13142)  

**Abstract**: Recent large vision-language models (LVLMs) have demonstrated strong potential for device control. However, existing research has primarily focused on point-and-click (PnC) interaction, while remote-control (RC) interaction commonly encountered in everyday TV usage remains largely underexplored. To fill this gap, we introduce \textbf{TVWorld}, an offline graph-based abstraction of real-world TV navigation that enables reproducible and deployment-free evaluation. On this basis, we derive two complementary benchmarks that comprehensively assess TV-use capabilities: \textbf{TVWorld-N} for topology-aware navigation and \textbf{TVWorld-G} for focus-aware grounding. These benchmarks expose a key limitation of existing agents: insufficient topology awareness for focus-based, long-horizon TV navigation. Motivated by this finding, we propose a \emph{Topology-Aware Training} framework that injects topology awareness into LVLMs. Using this framework, we develop \textbf{TVTheseus}, a foundation model specialized for TV navigation. TVTheseus achieves a success rate of $68.3\%$ on TVWorld-N, surpassing strong closed-source baselines such as Gemini 3 Flash and establishing state-of-the-art (SOTA) performance. Additional analyses further provide valuable insights into the development of effective TV-use agents. 

---
# RAGExplorer: A Visual Analytics System for the Comparative Diagnosis of RAG Systems 

**Authors**: Haoyu Tian, Yingchaojie Feng, Zhen Wen, Haoxuan Li, Minfeng Zhu, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.12991)  

**Abstract**: The advent of Retrieval-Augmented Generation (RAG) has significantly enhanced the ability of Large Language Models (LLMs) to produce factually accurate and up-to-date responses. However, the performance of a RAG system is not determined by a single component but emerges from a complex interplay of modular choices, such as embedding models and retrieval algorithms. This creates a vast and often opaque configuration space, making it challenging for developers to understand performance trade-offs and identify optimal designs. To address this challenge, we present RAGExplorer, a visual analytics system for the systematic comparison and diagnosis of RAG configurations. RAGExplorer guides users through a seamless macro-to-micro analytical workflow. Initially, it empowers developers to survey the performance landscape across numerous configurations, allowing for a high-level understanding of which design choices are most effective. For a deeper analysis, the system enables users to drill down into individual failure cases, investigate how differences in retrieved information contribute to errors, and interactively test hypotheses by manipulating the provided context to observe the resulting impact on the generated answer. We demonstrate the effectiveness of RAGExplorer through detailed case studies and user studies, validating its ability to empower developers in navigating the complex RAG design space. Our code and user guide are publicly available at this https URL. 

---
# Lombard Speech Synthesis for Any Voice with Controllable Style Embeddings 

**Authors**: Seymanur Akti, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2601.12966)  

**Abstract**: The Lombard effect plays a key role in natural communication, particularly in noisy environments or when addressing hearing-impaired listeners. We present a controllable text-to-speech (TTS) system capable of synthesizing Lombard speech for any speaker without requiring explicit Lombard data during training. Our approach leverages style embeddings learned from a large, prosodically diverse dataset and analyzes their correlation with Lombard attributes using principal component analysis (PCA). By shifting the relevant PCA components, we manipulate the style embeddings and incorporate them into our TTS model to generate speech at desired Lombard levels. Evaluations demonstrate that our method preserves naturalness and speaker identity, enhances intelligibility under noise, and provides fine-grained control over prosody, offering a robust solution for controllable Lombard TTS for any speaker. 

---
# AI-generated data contamination erodes pathological variability and diagnostic reliability 

**Authors**: Hongyu He, Shaowen Xiang, Ye Zhang, Yingtao Zhu, Jin Zhang, Hao Deng, Emily Alsentzer, Qingyu Chen, Kun-Hsing Yu, Andrew Marmenshall, Tingting Chen, Srinivas Anumasa, Daniel Ebner, Dean Ho, Kee Yuan Ngiam, Ching-Yu Cheng, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12946)  

**Abstract**: Generative artificial intelligence (AI) is rapidly populating medical records with synthetic content, creating a feedback loop where future models are increasingly at risk of training on uncurated AI-generated data. However, the clinical consequences of this AI-generated data contamination remain unexplored. Here, we show that in the absence of mandatory human verification, this self-referential cycle drives a rapid erosion of pathological variability and diagnostic reliability. By analysing more than 800,000 synthetic data points across clinical text generation, vision-language reporting, and medical image synthesis, we find that models progressively converge toward generic phenotypes regardless of the model architecture. Specifically, rare but critical findings, including pneumothorax and effusions, vanish from the synthetic content generated by AI models, while demographic representations skew heavily toward middle-aged male phenotypes. Crucially, this degradation is masked by false diagnostic confidence; models continue to issue reassuring reports while failing to detect life-threatening pathology, with false reassurance rates tripling to 40%. Blinded physician evaluation confirms that this decoupling of confidence and accuracy renders AI-generated documentation clinically useless after just two generations. We systematically evaluate three mitigation strategies, finding that while synthetic volume scaling fails to prevent collapse, mixing real data with quality-aware filtering effectively preserves diversity. Ultimately, our results suggest that without policy-mandated human oversight, the deployment of generative AI threatens to degrade the very healthcare data ecosystems it relies upon. 

---
# Hierarchical Sparse Circuit Extraction from Billion-Parameter Language Models through Scalable Attribution Graph Decomposition 

**Authors**: Mohammed Mudassir Uddin, Shahnawaz Alam, Mohammed Kaif Pasha  

**Link**: [PDF](https://arxiv.org/pdf/2601.12879)  

**Abstract**: Mechanistic interpretability seeks to reverse-engineer neural network computations into human-understandable algorithms, yet extracting sparse computational circuits from billion-parameter language models remains challenging due to exponential search complexity and pervasive polysemanticity. The proposed Hierarchical Attribution Graph Decomposition (HAGD) framework reduces circuit discovery complexity from O(2^n) exhaustive enumeration to O(n^2 log n) through multi-resolution abstraction hierarchies and differentiable circuit search. The methodology integrates cross-layer transcoders for monosemantic feature extraction, graph neural network meta-learning for topology prediction, and causal intervention protocols for validation. Empirical evaluation spans GPT-2 variants, Llama-7B through Llama-70B, and Pythia suite models across algorithmic tasks and natural language benchmarks. On modular arithmetic tasks, the framework achieves up to 91% behavioral preservation ($\pm$2.3\% across runs) while maintaining interpretable subgraph sizes. Cross-architecture transfer experiments suggest that discovered circuits exhibit moderate structural similarity (averaging 67%) across model families, indicating potential shared computational patterns. These results provide preliminary foundations for interpretability at larger model scales while identifying significant limitations in current attribution methodologies that require future advances. 

---
# SciHorizon-GENE: Benchmarking LLM for Life Sciences Inference from Gene Knowledge to Functional Understanding 

**Authors**: Xiaohan Huang, Meng Xiao, Chuan Qin, Qingqing Long, Jinmiao Chen, Yuanchun Zhou, Hengshu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12805)  

**Abstract**: Large language models (LLMs) have shown growing promise in biomedical research, particularly for knowledge-driven interpretation tasks. However, their ability to reliably reason from gene-level knowledge to functional understanding, However, their ability to reliably reason from gene-level knowledge to functional understanding, a core requirement for knowledge-enhanced cell atlas interpretation, remains largely underexplored. To address this gap, we introduce SciHorizon-GENE, a large-scale gene-centric benchmark constructed from authoritative biological databases. The benchmark integrates curated knowledge for over 190K human genes and comprises more than 540K questions covering diverse gene-to-function reasoning scenarios relevant to cell type annotation, functional interpretation, and mechanism-oriented analysis. Motivated by behavioral patterns observed in preliminary examinations, SciHorizon-GENE evaluates LLMs along four biologically critical perspectives: research attention sensitivity, hallucination tendency, answer completeness, and literature influence, explicitly targeting failure modes that limit the safe adoption of LLMs in biological interpretation pipelines. We systematically evaluate a wide range of state-of-the-art general-purpose and biomedical LLMs, revealing substantial heterogeneity in gene-level reasoning capabilities and persistent challenges in generating faithful, complete, and literature-grounded functional interpretations. Our benchmark establishes a systematic foundation for analyzing LLM behavior at the gene scale and offers insights for model selection and development, with direct relevance to knowledge-enhanced biological interpretation. 

---
# FRoM-W1: Towards General Humanoid Whole-Body Control with Language Instructions 

**Authors**: Peng Li, Zihan Zhuang, Yangfan Gao, Yi Dong, Sixian Li, Changhao Jiang, Shihan Dou, Zhiheng Xi, Enyu Zhou, Jixuan Huang, Hui Li, Jingjing Gong, Xingjun Ma, Tao Gui, Zuxuan Wu, Qi Zhang, Xuanjing Huang, Yu-Gang Jiang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12799)  

**Abstract**: Humanoid robots are capable of performing various actions such as greeting, dancing and even backflipping. However, these motions are often hard-coded or specifically trained, which limits their versatility. In this work, we present FRoM-W1, an open-source framework designed to achieve general humanoid whole-body motion control using natural language. To universally understand natural language and generate corresponding motions, as well as enable various humanoid robots to stably execute these motions in the physical world under gravity, FRoM-W1 operates in two stages: (a) H-GPT: utilizing massive human data, a large-scale language-driven human whole-body motion generation model is trained to generate diverse natural behaviors. We further leverage the Chain-of-Thought technique to improve the model's generalization in instruction understanding. (b) H-ACT: After retargeting generated human whole-body motions into robot-specific actions, a motion controller that is pretrained and further fine-tuned through reinforcement learning in physical simulation enables humanoid robots to accurately and stably perform corresponding actions. It is then deployed on real robots via a modular simulation-to-reality module. We extensively evaluate FRoM-W1 on Unitree H1 and G1 robots. Results demonstrate superior performance on the HumanML3D-X benchmark for human whole-body motion generation, and our introduced reinforcement learning fine-tuning consistently improves both motion tracking accuracy and task success rates of these humanoid robots. We open-source the entire FRoM-W1 framework and hope it will advance the development of humanoid intelligence. 

---
# Open Vocabulary Panoptic Segmentation With Retrieval Augmentation 

**Authors**: Nafis Sadeq, Qingfeng Liu, Mostafa El-Khamy  

**Link**: [PDF](https://arxiv.org/pdf/2601.12779)  

**Abstract**: Given an input image and set of class names, panoptic segmentation aims to label each pixel in an image with class labels and instance labels. In comparison, Open Vocabulary Panoptic Segmentation aims to facilitate the segmentation of arbitrary classes according to user input. The challenge is that a panoptic segmentation system trained on a particular dataset typically does not generalize well to unseen classes beyond the training data. In this work, we propose RetCLIP, a retrieval-augmented panoptic segmentation method that improves the performance of unseen classes. In particular, we construct a masked segment feature database using paired image-text data. At inference time, we use masked segment features from the input image as query keys to retrieve similar features and associated class labels from the database. Classification scores for the masked segment are assigned based on the similarity between query features and retrieved features. The retrieval-based classification scores are combined with CLIP-based scores to produce the final output. We incorporate our solution with a previous SOTA method (FC-CLIP). When trained on COCO, the proposed method demonstrates 30.9 PQ, 19.3 mAP, 44.0 mIoU on the ADE20k dataset, achieving +4.5 PQ, +2.5 mAP, +10.0 mIoU absolute improvement over the baseline. 

---
# PAIR-SAFE: A Paired-Agent Approach for Runtime Auditing and Refining AI-Mediated Mental Health Support 

**Authors**: Jiwon Kim, Violeta J. Rodriguez, Dong Whi Yoo, Eshwar Chandrasekharan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2601.12754)  

**Abstract**: Large language models (LLMs) are increasingly used for mental health support, yet they can produce responses that are overly directive, inconsistent, or clinically misaligned, particularly in sensitive or high-risk contexts. Existing approaches to mitigating these risks largely rely on implicit alignment through training or prompting, offering limited transparency and runtime accountability. We introduce PAIR-SAFE, a paired-agent framework for auditing and refining AI-generated mental health support that integrates a Responder agent with a supervisory Judge agent grounded in the clinically validated Motivational Interviewing Treatment Integrity (MITI-4) framework. The Judgeaudits each response and provides structuredALLOW or REVISE decisions that guide runtime response refinement. We simulate counseling interactions using a support-seeker simulator derived from human-annotated motivational interviewing data. We find that Judge-supervised interactions show significant improvements in key MITI dimensions, including Partnership, Seek Collaboration, and overall Relational quality. Our quantitative findings are supported by qualitative expert evaluation, which further highlights the nuances of runtime supervision. Together, our results reveal that such pairedagent approach can provide clinically grounded auditing and refinement for AI-assisted conversational mental health support. 

---
# SSVD-O: Parameter-Efficient Fine-Tuning with Structured SVD for Speech Recognition 

**Authors**: Pu Wang, Shinji Watanabe, Hugo Van hamme  

**Link**: [PDF](https://arxiv.org/pdf/2601.12600)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) is a scalable approach for adapting large speech foundation models to new domains. While methods such as LoRA and its state-of-the-art variants reduce adaptation costs, they typically allocate parameters uniformly across model subspaces, which limits their efficiency and scalability in speech applications. Building on our prior work, this paper introduces SSVD-Outer (SSVD-O), an extension of the structured SVD-guided (SSVD) fine-tuning method. SSVD-O combines input acoustic feature space-associated inner transformations with output semantic feature space-associated outer transformations to enable scalable and balanced adaptation. We conduct the first systematic analysis of parameter budget allocation across model subspaces in PEFT for automatic speech recognition (ASR), and investigate the trade-off between learning and forgetting under constrained resources. SSVD-O is benchmarked against LoRA, DoRA, PiSSA, and SSVD on domain-shifted ASR tasks, including child speech and regional accents, across model scales from 0.1B to 2B within the ESPnet framework. Experimental results show that SSVD-O consistently narrows the performance gap to full fine-tuning while improving generalization and mitigating catastrophic forgetting. 

---
# Dissecting Linear Recurrent Models: How Different Gating Strategies Drive Selectivity and Generalization 

**Authors**: Younes Bouhadjar, Maxime Fabre, Felix Schmidt, Emre Neftci  

**Link**: [PDF](https://arxiv.org/pdf/2601.12598)  

**Abstract**: Linear recurrent neural networks have emerged as efficient alternatives to the original Transformer's softmax attention mechanism, thanks to their highly parallelizable training and constant memory and computation requirements at inference. Iterative refinements of these models have introduced an increasing number of architectural mechanisms, leading to increased complexity and computational costs. Nevertheless, systematic direct comparisons among these models remain limited. Existing benchmark tasks are either too simplistic to reveal substantial differences or excessively resource-intensive for experimentation. In this work, we propose a refined taxonomy of linear recurrent models and introduce SelectivBench, a set of lightweight and customizable synthetic benchmark tasks for systematically evaluating sequence models. SelectivBench specifically evaluates selectivity in sequence models at small to medium scale, such as the capacity to focus on relevant inputs while ignoring context-based distractors. It employs rule-based grammars to generate sequences with adjustable complexity, incorporating irregular gaps that intentionally violate transition rules. Evaluations of linear recurrent models on SelectivBench reveal performance patterns consistent with results from large-scale language tasks. Our analysis clarifies the roles of essential architectural features: gating and rapid forgetting mechanisms facilitate recall, in-state channel mixing is unnecessary for selectivity, but critical for generalization, and softmax attention remains dominant due to its memory capacity scaling with sequence length. Our benchmark enables targeted, efficient exploration of linear recurrent models and provides a controlled setting for studying behaviors observed in large-scale evaluations. Code is available at this https URL 

---
# MemeLens: Multilingual Multitask VLMs for Memes 

**Authors**: Ali Ezzat Shahroor, Mohamed Bayan Kmainasi, Abul Hasnat, Dimitar Dimitrov, Giovanni Da San Martino, Preslav Nakov, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2601.12539)  

**Abstract**: Memes are a dominant medium for online communication and manipulation because meaning emerges from interactions between embedded text, imagery, and cultural context. Existing meme research is distributed across tasks (hate, misogyny, propaganda, sentiment, humour) and languages, which limits cross-domain generalization. To address this gap we propose MemeLens, a unified multilingual and multitask explanation-enhanced Vision Language Model (VLM) for meme understanding. We consolidate 38 public meme datasets, filter and map dataset-specific labels into a shared taxonomy of $20$ tasks spanning harm, targets, figurative/pragmatic intent, and affect. We present a comprehensive empirical analysis across modeling paradigms, task categories, and datasets. Our findings suggest that robust meme understanding requires multimodal training, exhibits substantial variation across semantic categories, and remains sensitive to over-specialization when models are fine-tuned on individual datasets rather than trained in a unified setting. We will make the experimental resources and datasets publicly available for the community. 

---
# Agentic Reasoning for Large Language Models 

**Authors**: Tianxin Wei, Ting-Wei Li, Zhining Liu, Xuying Ning, Ze Yang, Jiaru Zou, Zhichen Zeng, Ruizhong Qiu, Xiao Lin, Dongqi Fu, Zihao Li, Mengting Ai, Duo Zhou, Wenxuan Bao, Yunzhe Li, Gaotang Li, Cheng Qian, Yu Wang, Xiangru Tang, Yin Xiao, Liri Fang, Hui Liu, Xianfeng Tang, Yuji Zhang, Chi Wang, Jiaxuan You, Heng Ji, Hanghang Tong, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2601.12538)  

**Abstract**: Reasoning is a fundamental cognitive process underlying inference, problem-solving, and decision-making. While large language models (LLMs) demonstrate strong reasoning capabilities in closed-world settings, they struggle in open-ended and dynamic environments. Agentic reasoning marks a paradigm shift by reframing LLMs as autonomous agents that plan, act, and learn through continual interaction. In this survey, we organize agentic reasoning along three complementary dimensions. First, we characterize environmental dynamics through three layers: foundational agentic reasoning, which establishes core single-agent capabilities including planning, tool use, and search in stable environments; self-evolving agentic reasoning, which studies how agents refine these capabilities through feedback, memory, and adaptation; and collective multi-agent reasoning, which extends intelligence to collaborative settings involving coordination, knowledge sharing, and shared goals. Across these layers, we distinguish in-context reasoning, which scales test-time interaction through structured orchestration, from post-training reasoning, which optimizes behaviors via reinforcement learning and supervised fine-tuning. We further review representative agentic reasoning frameworks across real-world applications and benchmarks, including science, robotics, healthcare, autonomous research, and mathematics. This survey synthesizes agentic reasoning methods into a unified roadmap bridging thought and action, and outlines open challenges and future directions, including personalization, long-horizon interaction, world modeling, scalable multi-agent training, and governance for real-world deployment. 

---
# Harmonizing the Arabic Audio Space with Data Scheduling 

**Authors**: Hunzalah Hassan Bhatti, Firoj Alam, Shammur Absar Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2601.12494)  

**Abstract**: Audio large language models (LLMs) enable unified speech understanding and generation, yet their adaptation to linguistically complex, dialect-rich settings remains underexplored. This paper presents the first systematic study of multi-task instruction tuning for an Arabic-centric audio LLM, covering a hierarchy of generative tasks (ASR, speech summarization) and discriminative tasks (dialect and emotion identification). To support this study, we introduce AraMega-SSum, a novel dataset for Arabic speech summarization. We fine-tune Qwen2.5-Omni (7B) and propose Task-Progressive Curriculum (TPC) along with Aligner-Based Diverse Sampling (ADS), a strategy that constructs information-dense batches by selecting task- and label-balanced examples. Our results reveal a critical efficiency, robustness trade-off: while ADS accelerates initial convergence and boosts paralinguistic F1-scores, its inherent gradient volatility can destabilize generative decoding under prolonged training. Furthermore, while the TPC stabilizes core acoustic mapping, it often induces negative transfer in downstream tasks. We demonstrate that a Hybrid TPC+ADS Strategy provides an optimal training ``recipe'', first establishing a robust representative foundation before employing diversity-aware refinement to capture fine-grained nuances. These findings offer practical guidance for the efficient adaptation of Omni-models in complex, low-resource multimodal environments. 

---
# Privacy-Preserving Federated Learning with Verifiable Fairness Guarantees 

**Authors**: Mohammed Himayath Ali, Mohammed Aqib Abdullah, Syed Muneer Hussin, Mohammed Mudassir Uddin, Shahnawaz Alam  

**Link**: [PDF](https://arxiv.org/pdf/2601.12447)  

**Abstract**: Federated learning enables collaborative model training across distributed institutions without centralizing sensitive data; however, ensuring algorithmic fairness across heterogeneous data distributions while preserving privacy remains fundamentally unresolved. This paper introduces CryptoFair-FL, a novel cryptographic framework providing the first verifiable fairness guarantees for federated learning systems under formal security definitions. The proposed approach combines additively homomorphic encryption with secure multi-party computation to enable privacy-preserving verification of demographic parity and equalized odds metrics without revealing protected attribute distributions or individual predictions. A novel batched verification protocol reduces computational complexity from BigO(n^2) to BigO(n \log n) while maintaining (\dparam, \deltap)-differential privacy with dparam = 0.5 and deltap = 10^{-6}. Theoretical analysis establishes information-theoretic lower bounds on the privacy cost of fairness verification, demonstrating that the proposed protocol achieves near-optimal privacy-fairness tradeoffs. Comprehensive experiments across four benchmark datasets (MIMIC-IV healthcare records, Adult Income, CelebA, and a novel FedFair-100 benchmark) demonstrate that CryptoFair-FL reduces fairness violations from 0.231 to 0.031 demographic parity difference while incurring only 2.3 times computational overhead compared to standard federated averaging. The framework successfully defends against attribute inference attacks, maintaining adversarial success probability below 0.05 across all tested configurations. These results establish a practical pathway for deploying fairness-aware federated learning in regulated industries requiring both privacy protection and algorithmic accountability. 

---
# De-Anonymization at Scale via Tournament-Style Attribution 

**Authors**: Lirui Zhang, Huishuai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.12407)  

**Abstract**: As LLMs rapidly advance and enter real-world use, their privacy implications are increasingly important. We study an authorship de-anonymization threat: using LLMs to link anonymous documents to their authors, potentially compromising settings such as double-blind peer review.
We propose De-Anonymization at Scale (DAS), a large language model-based method for attributing authorship among tens of thousands of candidate texts. DAS uses a sequential progression strategy: it randomly partitions the candidate corpus into fixed-size groups, prompts an LLM to select the text most likely written by the same author as a query text, and iteratively re-queries the surviving candidates to produce a ranked top-k list. To make this practical at scale, DAS adds a dense-retrieval prefilter to shrink the search space and a majority-voting style aggregation over multiple independent runs to improve robustness and ranking precision. Experiments on anonymized review data show DAS can recover same-author texts from pools of tens of thousands with accuracy well above chance, demonstrating a realistic privacy risk for anonymous platforms. On standard authorship benchmarks (Enron emails and blog posts), DAS also improves both accuracy and scalability over prior approaches, highlighting a new LLM-enabled de-anonymization vulnerability. 

---
# Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs 

**Authors**: Anirudh Sekar, Mrinal Agarwal, Rachel Sharma, Akitsugu Tanaka, Jasmine Zhang, Arjun Damerla, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12359)  

**Abstract**: Prompt injection attacks have become an increasing vulnerability for LLM applications, where adversarial prompts exploit indirect input channels such as emails or user-generated content to circumvent alignment safeguards and induce harmful or unintended outputs. Despite advances in alignment, even state-of-the-art LLMs remain broadly vulnerable to adversarial prompts, underscoring the urgent need for robust, productive, and generalizable detection mechanisms beyond inefficient, model-specific patches. In this work, we propose Zero-Shot Embedding Drift Detection (ZEDD), a lightweight, low-engineering-overhead framework that identifies both direct and indirect prompt injection attempts by quantifying semantic shifts in embedding space between benign and suspect inputs. ZEDD operates without requiring access to model internals, prior knowledge of attack types, or task-specific retraining, enabling efficient zero-shot deployment across diverse LLM architectures. Our method uses adversarial-clean prompt pairs and measures embedding drift via cosine similarity to capture subtle adversarial manipulations inherent to real-world injection attacks. To ensure robust evaluation, we assemble and re-annotate the comprehensive LLMail-Inject dataset spanning five injection categories derived from publicly available sources. Extensive experiments demonstrate that embedding drift is a robust and transferable signal, outperforming traditional methods in detection accuracy and operational efficiency. With greater than 93% accuracy in classifying prompt injections across model architectures like Llama 3, Qwen 2, and Mistral and a false positive rate of <3%, our approach offers a lightweight, scalable defense layer that integrates into existing LLM pipelines, addressing a critical gap in securing LLM-powered systems to withstand adaptive adversarial threats. 

---
# Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline 

**Authors**: Jiawei Xu, Arief Koesdwiady, Sisong Bei, Yan Han, Baixiang Huang, Dakuo Wang, Yutong Chen, Zheshen Wang, Peihao Wang, Pan Li, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2601.12307)  

**Abstract**: Recent advances in LLM-based multi-agent systems (MAS) show that workflows composed of multiple LLM agents with distinct roles, tools, and communication patterns can outperform single-LLM baselines on complex tasks. However, most frameworks are homogeneous, where all agents share the same base LLM and differ only in prompts, tools, and positions in the workflow. This raises the question of whether such workflows can be simulated by a single agent through multi-turn conversations. We investigate this across seven benchmarks spanning coding, mathematics, general question answering, domain-specific reasoning, and real-world planning and tool use. Our results show that a single agent can reach the performance of homogeneous workflows with an efficiency advantage from KV cache reuse, and can even match the performance of an automatically optimized heterogeneous workflow. Building on this finding, we propose \textbf{OneFlow}, an algorithm that automatically tailors workflows for single-agent execution, reducing inference costs compared to existing automatic multi-agent design frameworks without trading off accuracy. These results position the single-LLM implementation of multi-agent workflows as a strong baseline for MAS research. We also note that single-LLM methods cannot capture heterogeneous workflows due to the lack of KV cache sharing across different LLMs, highlighting future opportunities in developing \textit{truly} heterogeneous multi-agent systems. 

---
# Environment-Aware Code Generation: How far are We? 

**Authors**: Tongtong Wu, Rongyi Chen, Wenjie Du, Suyu Ma, Guilin Qi, Zhenchang Xing, Shahram Khadivi, Ramesh Periyathambi, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2601.12262)  

**Abstract**: Recent progress in large language models (LLMs) has improved code generation, but most evaluations still test isolated, small-scale code (e.g., a single function) under default or unspecified software environments. As a result, it is unclear whether LLMs can reliably generate executable code tailored to a user's specific environment. We present the first systematic study of Environment-Aware Code Generation (EACG), where generated code must be functionally correct and directly executable under arbitrary software configurations. To enable realistic evaluation, we introduce VersiBCB, a benchmark that is multi-package, execution-verified, and deprecation-aware, capturing complex and evolving environments that prior datasets often overlook. Using VersiBCB, we investigate three complementary adaptation axes: data, parameters, and cache, and develop representative strategies for each. Our results show that current LLMs struggle with environment-specific code generation, while our adaptations improve environment compatibility and executability. These findings highlight key challenges and opportunities for deploying LLMs in practical software engineering workflows. 

---
# AQUA-Bench: Beyond Finding Answers to Knowing When There Are None in Audio Question Answering 

**Authors**: Chun-Yi Kuan, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.12248)  

**Abstract**: Recent advances in audio-aware large language models have shown strong performance on audio question answering. However, existing benchmarks mainly cover answerable questions and overlook the challenge of unanswerable ones, where no reliable answer can be inferred from the audio. Such cases are common in real-world settings, where questions may be misleading, ill-posed, or incompatible with the information. To address this gap, we present AQUA-Bench, a benchmark for Audio Question Unanswerability Assessment. It systematically evaluates three scenarios: Absent Answer Detection (the correct option is missing), Incompatible Answer Set Detection (choices are categorically mismatched with the question), and Incompatible Audio Question Detection (the question is irrelevant or lacks sufficient grounding in the audio). By assessing these cases, AQUA-Bench offers a rigorous measure of model reliability and promotes the development of audio-language systems that are more robust and trustworthy. Our experiments suggest that while models excel on standard answerable tasks, they often face notable challenges with unanswerable ones, pointing to a blind spot in current audio-language understanding. 

---
# The Language You Ask In: Language-Conditioned Ideological Divergence in LLM Analysis of Contested Political Documents 

**Authors**: Oleg Smirnov  

**Link**: [PDF](https://arxiv.org/pdf/2601.12164)  

**Abstract**: Large language models (LLMs) are increasingly deployed as analytical tools across multilingual contexts, yet their outputs may carry systematic biases conditioned by the language of the prompt. This study presents an experimental comparison of LLM-generated political analyses of a Ukrainian civil society document, using semantically equivalent prompts in Russian and Ukrainian. Despite identical source material and parallel query structures, the resulting analyses varied substantially in rhetorical positioning, ideological orientation, and interpretive conclusions. The Russian-language output echoed narratives common in Russian state discourse, characterizing civil society actors as illegitimate elites undermining democratic mandates. The Ukrainian-language output adopted vocabulary characteristic of Western liberal-democratic political science, treating the same actors as legitimate stakeholders within democratic contestation. These findings demonstrate that prompt language alone can produce systematically different ideological orientations from identical models analyzing identical content, with significant implications for AI deployment in polarized information environments, cross-lingual research applications, and the governance of AI systems in multilingual societies. 

---
# Neural Isomorphic Fields: A Transformer-based Algebraic Numerical Embedding 

**Authors**: Hamidreza Sadeghi, Saeedeh Momtazi, Reza Safabakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2601.12095)  

**Abstract**: Neural network models often face challenges when processing very small or very large numbers due to issues such as overflow, underflow, and unstable output variations. To mitigate these problems, we propose using embedding vectors for numbers instead of directly using their raw values. These embeddings aim to retain essential algebraic properties while preventing numerical instabilities. In this paper, we introduce, for the first time, a fixed-length number embedding vector that preserves algebraic operations, including addition, multiplication, and comparison, within the field of rational numbers. We propose a novel Neural Isomorphic Field, a neural abstraction of algebraic structures such as groups and fields. The elements of this neural field are embedding vectors that maintain algebraic structure during computations. Our experiments demonstrate that addition performs exceptionally well, achieving over 95 percent accuracy on key algebraic tests such as identity, closure, and associativity. In contrast, multiplication exhibits challenges, with accuracy ranging from 53 percent to 73 percent across various algebraic properties. These findings highlight the model's strengths in preserving algebraic properties under addition while identifying avenues for further improvement in handling multiplication. 

---
# CroBIM-V: Memory-Quality Controlled Remote Sensing Referring Video Object Segmentation 

**Authors**: H. Jiang, Y. Sun, Z. Dong, T. Liu, Y. Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12076)  

**Abstract**: Remote sensing video referring object segmentation (RS-RVOS) is challenged by weak target saliency and severe visual information truncation in dynamic scenes, making it extremely difficult to maintain discriminative target representations during segmentation. Moreover, progress in this field is hindered by the absence of large-scale dedicated benchmarks, while existing models are often affected by biased initial memory construction that impairs accurate instance localization in complex scenarios, as well as indiscriminate memory accumulation that encodes noise from occlusions or misclassifications, leading to persistent error propagation. This paper advances RS-RVOS research through dual contributions in data and methodology. First, we construct RS-RVOS Bench, the first large-scale benchmark comprising 111 video sequences, about 25,000 frames, and 213,000 temporal referring annotations. Unlike common RVOS benchmarks where many expressions are written with access to the full video context, our dataset adopts a strict causality-aware annotation strategy in which linguistic references are generated solely from the target state in the initial frame. Second, we propose a memory-quality-aware online referring segmentation framework, termed Memory Quality Control with Segment Anything Model (MQC-SAM). MQC-SAM introduces a temporal motion consistency module for initial memory calibration, leveraging short-term motion trajectory priors to correct structural deviations and establish accurate memory anchoring. Furthermore, it incorporates a decoupled attention-based memory integration mechanism with dynamic quality assessment, selectively updating high-confidence semantic features while filtering unreliable information, thereby effectively preventing error accumulation and propagation. Extensive experiments on RS-RVOS Bench demonstrate that MQC-SAM achieves state-of-the-art performance. 

---
# A Multi-Agent System for Generating Actionable Business Advice 

**Authors**: Kartikey Singh Bhandari, Tanish Jain, Archit Agrawal, Dhruv Kumar, Praveen Kumar, Pratik Narang  

**Link**: [PDF](https://arxiv.org/pdf/2601.12024)  

**Abstract**: Customer reviews contain rich signals about product weaknesses and unmet user needs, yet existing analytic methods rarely move beyond descriptive tasks such as sentiment analysis or aspect extraction. While large language models (LLMs) can generate free-form suggestions, their outputs often lack accuracy and depth of reasoning. In this paper, we present a multi-agent, LLM-based framework for prescriptive decision support, which transforms large scale review corpora into actionable business advice. The framework integrates four components: clustering to select representative reviews, generation of advices, iterative evaluation, and feasibility based ranking. This design couples corpus distillation with feedback driven advice refinement to produce outputs that are specific, actionable, and practical. Experiments across three service domains and multiple model families show that our framework consistently outperform single model baselines on actionability, specificity, and non-redundancy, with medium sized models approaching the performance of large model frameworks. 

---
# R$^2$PO: Decoupling Training Trajectories from Inference Responses for LLM Reasoning 

**Authors**: Jingchu Wang, Bingbing Xu, Yige Yuan, Bin Xie, Xiaoqian Sun, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11960)  

**Abstract**: Reinforcement learning has become a central paradigm for improving LLM reasoning. However, existing methods use a single policy to produce both inference responses and training optimization trajectories. The objective conflict between generating stable inference responses and diverse training trajectories leads to insufficient exploration, which harms reasoning capability. In this paper, to address the problem, we propose R$^2$PO (Residual Rollout Policy Optimization), which introduces a lightweight Residual Rollout-Head atop the policy to decouple training trajectories from inference responses, enabling controlled trajectory diversification during training while keeping inference generation stable. Experiments across multiple benchmarks show that our method consistently outperforms baselines, achieving average accuracy gains of 3.1% on MATH-500 and 2.4% on APPS, while also reducing formatting errors and mitigating length bias for stable optimization. Our code is publicly available at this https URL. 

---
# Thinking Traps in Long Chain-of-Thought: A Measurable Study and Trap-Aware Adaptive Restart 

**Authors**: Kang Chen, Fan Yu, Junjie Nian, Shihan Zhao, Zhuoka Feng, Zijun Yao, Heng Wang, Minshen Yu, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2601.11940)  

**Abstract**: Scaling test-time compute via Long Chain-of-Thought (Long-CoT) significantly enhances reasoning capabilities, yet extended generation does not guarantee correctness: after an early wrong commitment, models may keep elaborating a self-consistent but incorrect prefix. Through fine-grained trajectory analysis, we identify Thinking Traps, prefix-dominant deadlocks where later reflection, alternative attempts, or verification fails to revise the root error. On a curated subset of DAPO-MATH, 89\% of failures exhibit such traps. To solve this problem, we introduce TAAR (Trap-Aware Adaptive Restart), a test-time control framework that trains a diagnostic policy to predict two signals from partial trajectories: a trap index for where to truncate and an escape probability for whether and how strongly to intervene. At inference time, TAAR truncates the trajectory before the predicted trap segment and adaptively restarts decoding; for severely trapped cases, it applies stronger perturbations, including higher-temperature resampling and an optional structured reboot suffix. Experiments on challenging mathematical and scientific reasoning benchmarks (AIME24, AIME25, GPQA-Diamond, HMMT25, BRUMO25) show that TAAR improves reasoning performance without fine-tuning base model parameters. 

---
# AGGC: Adaptive Group Gradient Clipping for Stabilizing Large Language Model Training 

**Authors**: Zhiyuan Li, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11864)  

**Abstract**: To stabilize the training of Large Language Models (LLMs), gradient clipping is a nearly ubiquitous heuristic used to alleviate exploding gradients. However, traditional global norm clipping erroneously presupposes gradient homogeneity across different functional modules, leading to an adverse "spill-over" effect where volatile parameters force unnecessary scaling on stable ones. To overcome this, we propose Adaptive Group-wise Gradient Clipping (AGGC). AGGC partitions parameters into groups based on functional types and regulates each according to its historical behavior using an Exponential Moving Average (EMA). Specifically, it constructs an adaptive interval to simultaneously mitigate gradient explosion and vanishing, while employing a time-dependent scheduling mechanism to balance exploration and convergence. Experiments on LLaMA 2-7B, Mistral-7B, and Gemma-7B models show that AGGC consistently outperforms LoRA and frequently surpasses Full Fine-Tuning. On the GSM8K benchmark, Mistral-7B fine-tuned with AGGC achieves an accuracy of 72.93%, exceeding LoRA's 69.5%. AGGC also effectively stabilizes Reinforcement Learning with Verifiable Rewards (RLVR), enhancing the logic deduction of Qwen 2.5 and Llama 3.2 models. Experimental results demonstrate that AGGC effectively addresses the limitations of traditional gradient clipping methods, particularly in overcoming gradient heterogeneity, by utilizing a modular, adaptive clipping strategy to stabilize the training process. Due to its lightweight design, AGGC can be seamlessly integrated into existing post-training pipelines with negligible overhead. 

---
# Utilizing Metadata for Better Retrieval-Augmented Generation 

**Authors**: Raquib Bin Yousuf, Shengzhe Xu, Mandar Sharma, Andrew Neeser, Chris Latimer, Naren Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2601.11863)  

**Abstract**: Retrieval-Augmented Generation systems depend on retrieving semantically relevant document chunks to support accurate, grounded outputs from large language models. In structured and repetitive corpora such as regulatory filings, chunk similarity alone often fails to distinguish between documents with overlapping language. Practitioners often flatten metadata into input text as a heuristic, but the impact and trade-offs of this practice remain poorly understood. We present a systematic study of metadata-aware retrieval strategies, comparing plain-text baselines with approaches that embed metadata directly. Our evaluation spans metadata-as-text (prefix and suffix), a dual-encoder unified embedding that fuses metadata and content in a single index, dual-encoder late-fusion retrieval, and metadata-aware query reformulation. Across multiple retrieval metrics and question types, we find that prefixing and unified embeddings consistently outperform plain-text baselines, with the unified at times exceeding prefixing while being easier to maintain. Beyond empirical comparisons, we analyze embedding space, showing that metadata integration improves effectiveness by increasing intra-document cohesion, reducing inter-document confusion, and widening the separation between relevant and irrelevant chunks. Field-level ablations show that structural cues provide strong disambiguating signals. Our code, evaluation framework, and the RAGMATE-10K dataset are publicly hosted. 

---
# A self-evolving multi-role collaborative framework with fine-grained difficulty guidance for innovative mathematical problem generation 

**Authors**: Yifei Sun, Yongan Li, A.K. Qin, Sicheng Hou, Tamas Pflanzner  

**Link**: [PDF](https://arxiv.org/pdf/2601.11792)  

**Abstract**: Mathematical problem generation (MPG) is a significant research direction in the field of intelligent education. In recent years, the rapid development of large language models (LLMs) has enabled new technological approaches to problem-generation tasks. Although existing LLMs can achieve high correctness rates, they generally lack innovation and exhibit poor discrimination. In this paper, we propose the task of innovative math problem generation (IMPG). To solve the IMPG task, this paper proposes a self-evolving, multi-role collaborative framework with fine-grained difficulty guidance. First, a multi-role collaborative mechanism comprising a sampler, generator, evaluator, state machine, and memory is constructed, ensuring the correctness of generated problems through iterative optimization informed by self-assessment and external feedback. Second, we introduce an improved difficulty model to quantify difficulty and provide fine-grained guidance. We adopt the data-driven association-guided path sampling (DAPS) algorithm to enhance the semantic rationality of sampled encodings. Third, we construct the HSM3K-CN dataset, which comprises high-quality high school math problems. A multi-stage training pipeline is adopted, incorporating continual pre-training (CPT), supervised fine-tuning (SFT), and group relative policy optimization (GRPO), to enhance the generation and evaluation capabilities of the base model. Finally, system self-evolution is achieved by transferring evaluation capabilities from the expert model to the apprentice model via distillation. Experiments show that, compared to baseline models, our proposed method significantly improves the innovation of the generated problems while maintaining a high correctness rate. 

---
# The Stability Trap: Evaluating the Reliability of LLM-Based Instruction Adherence Auditing 

**Authors**: Murtuza N. Shergadwala  

**Link**: [PDF](https://arxiv.org/pdf/2601.11783)  

**Abstract**: The enterprise governance of Generative AI (GenAI) in regulated sectors, such as Human Resources (HR), demands scalable yet reproducible auditing mechanisms. While Large Language Model (LLM)-as-a-Judge approaches offer scalability, their reliability in evaluating adherence of different types of system instructions remains unverified. This study asks: To what extent does the instruction type of an Application Under Test (AUT) influence the stability of judge evaluations? To address this, we introduce the Scoped Instruction Decomposition Framework to classify AUT instructions into Objective and Subjective types, isolating the factors that drive judge instability. We applied this framework to two representative HR GenAI applications, evaluating the stability of four judge architectures over variable runs. Our results reveal a ``Stability Trap'' characterized by a divergence between Verdict Stability and Reasoning Stability. While judges achieved near-perfect verdict agreement ($>99\%$) for both objective and subjective evaluations, their accompanying justification traces diverged significantly. Objective instructions requiring quantitative analysis, such as word counting, exhibited reasoning stability as low as $\approx19\%$, driven by variances in numeric justifications. Similarly, reasoning stability for subjective instructions varied widely ($35\%$--$83\%$) based on evidence granularity, with feature-specific checks failing to reproduce consistent rationale. Conversely, objective instructions focusing on discrete entity extraction achieved high reasoning stability ($>90\%$). These findings demonstrate that high verdict stability can mask fragile reasoning. Thus, we suggest that auditors scope automated evaluation protocols strictly: delegate all deterministically verifiable logic to code, while reserving LLM judges for complex semantic evaluation. 

---
# Advances and Frontiers of LLM-based Issue Resolution in Software Engineering: A Comprehensive Survey 

**Authors**: Caihua Li, Lianghong Guo, Yanlin Wang, Daya Guo, Wei Tao, Zhenyu Shan, Mingwei Liu, Jiachi Chen, Haoyu Song, Duyu Tang, Hongyu Zhang, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.11655)  

**Abstract**: Issue resolution, a complex Software Engineering (SWE) task integral to real-world development, has emerged as a compelling challenge for artificial intelligence. The establishment of benchmarks like SWE-bench revealed this task as profoundly difficult for large language models, thereby significantly accelerating the evolution of autonomous coding agents. This paper presents a systematic survey of this emerging domain. We begin by examining data construction pipelines, covering automated collection and synthesis approaches. We then provide a comprehensive analysis of methodologies, spanning training-free frameworks with their modular components to training-based techniques, including supervised fine-tuning and reinforcement learning. Subsequently, we discuss critical analyses of data quality and agent behavior, alongside practical applications. Finally, we identify key challenges and outline promising directions for future research. An open-source repository is maintained at this https URL to serve as a dynamic resource in this field. 

---
# Mixture-of-Experts as Soft Clustering: A Dual Jacobian-PCA Spectral Geometry Perspective 

**Authors**: Feilong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11616)  

**Abstract**: Mixture-of-Experts (MoE) architectures are commonly motivated by efficiency and conditional computation, but their effect on the geometry of learned functions and representations remains poorly characterized. In this work, we study MoEs through a geometric lens, interpreting routing as a form of soft partitioning of the representation space into overlapping local charts. We introduce a Dual Jacobian-PCA Spectral Geometry probe. It analyzes local function geometry via Jacobian singular-value spectra and representation geometry via weighted PCA of routed hidden states. Using a controlled MLP-MoE setting that permits exact Jacobian computation, we compare dense, Top-k, and fully-soft routing architectures under matched capacity. Across random seeds, we observe that MoE routing consistently reduces local sensitivity, with expert-local Jacobians exhibiting smaller leading singular values and faster spectral decay than dense baselines. At the same time, weighted PCA reveals that expert-local representations distribute variance across a larger number of principal directions, indicating higher effective rank under identical input distributions. We further find that average expert Jacobians are nearly orthogonal, suggesting a decomposition of the transformation into low-overlap expert-specific subspaces rather than scaled variants of a shared map. We analyze how routing sharpness modulates these effects, showing that Top-k routing produces lower-rank, more concentrated expert-local structure, while fully-soft routing yields broader, higher-rank representations. Together, these results support a geometric interpretation of MoEs as soft partitionings of function space that flatten local curvature while redistributing representation variance. 

---
# Overview of the SciHigh Track at FIRE 2025: Research Highlight Generation from Scientific Papers 

**Authors**: Tohida Rehman, Debarshi Kumar Sanyal, Samiran Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2601.11582)  

**Abstract**: `SciHigh: Research Highlight Generation from Scientific Papers' focuses on the task of automatically generating concise, informative, and meaningful bullet-point highlights directly from scientific abstracts. The goal of this task is to evaluate how effectively computational models can generate highlights that capture the key contributions, findings, and novelty of a paper in a concise form. Highlights help readers grasp essential ideas quickly and are often easier to read and understand than longer paragraphs, especially on mobile devices. The track uses the MixSub dataset \cite{10172215}, which provides pairs of abstracts and corresponding author-written highlights.
In this inaugural edition of the track, 12 teams participated, exploring various approaches, including pre-trained language models, to generate highlights from this scientific dataset. All submissions were evaluated using established metrics such as ROUGE, METEOR, and BERTScore to measure both alignment with author-written highlights and overall informativeness. Teams were ranked based on ROUGE-L scores. The findings suggest that automatically generated highlights can reduce reading effort, accelerate literature reviews, and enhance metadata for digital libraries and academic search platforms. SciHigh provides a dedicated benchmark for advancing methods aimed at concise and accurate highlight generation from scientific writing. 

---
# AdaFRUGAL: Adaptive Memory-Efficient Training with Dynamic Control 

**Authors**: Quang-Hung Bui, Anh Son Ta  

**Link**: [PDF](https://arxiv.org/pdf/2601.11568)  

**Abstract**: Training Large Language Models (LLMs) is highly memory-intensive due to optimizer state overhead. The FRUGAL framework mitigates this with gradient splitting, but its static hyperparameters -- the subspace ratio ($\rho$) and update frequency ($T$) -- require costly manual tuning, limiting adaptability. We present AdaFRUGAL, which automates this process by introducing two dynamic controls: (i) a linear decay for $\rho$ to progressively reduce memory, and (ii) a loss-aware schedule for $T$ to lower computational overhead. Experiments across large-scale pre-training (English C4, Vietnamese VietVault) and fine-tuning (GLUE) demonstrate that AdaFRUGAL achieves a compelling trade-off. It maintains competitive performance against AdamW and static FRUGAL while significantly reducing both GPU memory and training time, offering a more practical, autonomous solution for resource-constrained LLM training. 

---
# MIMIC-RD: Can LLMs differentially diagnose rare diseases in real-world clinical settings? 

**Authors**: Zilal Eiz AlDin, John Wu, Jeffrey Paul Fung, Jennifer King, Mya Watts, Lauren ONeill, Adam Richard Cross, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.11559)  

**Abstract**: Despite rare diseases affecting 1 in 10 Americans, their differential diagnosis remains challenging. Due to their impressive recall abilities, large language models (LLMs) have been recently explored for differential diagnosis. Existing approaches to evaluating LLM-based rare disease diagnosis suffer from two critical limitations: they rely on idealized clinical case studies that fail to capture real-world clinical complexity, or they use ICD codes as disease labels, which significantly undercounts rare diseases since many lack direct mappings to comprehensive rare disease databases like Orphanet. To address these limitations, we explore MIMIC-RD, a rare disease differential diagnosis benchmark constructed by directly mapping clinical text entities to Orphanet. Our methodology involved an initial LLM-based mining process followed by validation from four medical annotators to confirm identified entities were genuine rare diseases. We evaluated various models on our dataset of 145 patients and found that current state-of-the-art LLMs perform poorly on rare disease differential diagnosis, highlighting the substantial gap between existing capabilities and clinical needs. From our findings, we outline several future steps towards improving differential diagnosis of rare diseases. 

---
# CSyMR: Benchmarking Compositional Symbolic Muisc Reasoning With MIR Tool Integration 

**Authors**: Boyang Wang, Yash Vishe, Xin Xu, Zachary Novack, Julian McAuley, Junda Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11556)  

**Abstract**: Large Language Models (LLMs) are leveraged in symbolic music reasoning, yet existing benchmarks emphasize isolated knowledge or atomic analyses rather than the integrative compositional reasoning needed to connect musical structures. To address this, we present the Compositional Symbolic Music Reasoning Benchmark (CSyMR-Bench), a curated multiple-choice dataset of 126 questions derived from expert forums and professional examinations. Each item involves combining several atomic analyses to arrive at the final answer. Furthermore, we introduce a tool-augmented agent framework that leverages symbolic music analysis tools from the music21 library to address the challenges posed by CSyMR-Bench. Experiments validate that CSyMR-Bench poses a non-trivial challenge across both community-sourced and exam-style questions, while our tool-augmented agent consistently outperforms all baselines, achieving 5-7% absolute accuracy gains. 

---
# Medication counseling with large language models: balancing flexibility and rigidity 

**Authors**: Joar Sabel, Mattias Wingren, Andreas Lundell, Sören Andersson, Sara Rosenberg, Susanne Hägglund, Linda Estman, Malin Andtfolk  

**Link**: [PDF](https://arxiv.org/pdf/2601.11544)  

**Abstract**: The introduction of large language models (LLMs) has greatly enhanced the capabilities of software agents. Instead of relying on rule-based interactions, agents can now interact in flexible ways akin to humans. However, this flexibility quickly becomes a problem in fields where errors can be disastrous, such as in a pharmacy context, but the opposite also holds true; a system that is too inflexible will also lead to errors, as it can become too rigid to handle situations that are not accounted for. Work using LLMs in a pharmacy context have adopted a wide scope, accounting for many different medications in brief interactions -- our strategy is the opposite: focus on a more narrow and long task. This not only enables a greater understanding of the task at hand, but also provides insight into what challenges are present in an interaction of longer nature. The main challenge, however, remains the same for a narrow and wide system: it needs to strike a balance between adherence to conversational requirements and flexibility. In an effort to strike such a balance, we present a prototype system meant to provide medication counseling while juggling these two extremes. We also cover our design in constructing such a system, with a focus on methods aiming to fulfill conversation requirements, reduce hallucinations and promote high-quality responses. The methods used have the potential to increase the determinism of the system, while simultaneously not removing the dynamic conversational abilities granted by the usage of LLMs. However, a great deal of work remains ahead, and the development of this kind of system needs to involve continuous testing and a human-in-the-loop. It should also be evaluated outside of commonly used benchmarks for LLMs, as these do not adequately capture the complexities of this kind of conversational system. 

---
# Lightweight Prompt Biasing for Contextualized End-to-End ASR Systems 

**Authors**: Bo Ren, Yu Shi, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06252)  

**Abstract**: End-to-End Automatic Speech Recognition (ASR) has advanced significantly yet still struggles with rare and domain-specific entities. This paper introduces a simple yet efficient prompt-based biasing technique for contextualized ASR, enhancing recognition accuracy by leverage a unified multitask learning framework. The approach comprises two key components: a prompt biasing model which is trained to determine when to focus on entities in prompt, and a entity filtering mechanism which efficiently filters out irrelevant entities. Our method significantly enhances ASR accuracy on entities, achieving a relative 30.7% and 18.0% reduction in Entity Word Error Rate compared to the baseline model with shallow fusion on in-house domain dataset with small and large entity lists, respectively. The primary advantage of this method lies in its efficiency and simplicity without any structure change, making it lightweight and highly efficient. 

---
