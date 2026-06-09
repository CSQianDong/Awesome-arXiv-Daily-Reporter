# Causally Evaluating the Learnability of Formal Language Tasks 

**Authors**: Vésteinn Snæbjarnarson, Anej Svete, Josef Valvoda, Reda Boumasmoud, Brian DuSell, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2606.09822)  

**Abstract**: Language models, as multi-task learners, acquire a wide range of abilities during training. A fundamental question is how much task-specific data is needed to learn a given task. Answering this for natural language is difficult: tasks are hard to delineate and can confound one another. To rigorously investigate the relationship between data frequency and learnability, we turn to a controlled setting using formal languages induced from probabilistic finite automata. These serve as a methodological testbed to demonstrate that standard correlational evaluation practices are inherently flawed. To enable causal analysis, we introduce the binning semiring, an algebraic object that lets us control how often a targeted property occurs in a sampled corpus. We formulate the experimental pipeline as a causal graphical model and derive decomposed Kullback-Leibler divergence metrics to measure the learnability of specific sub-tasks. Our experiments show that evaluating learnability without causal intervention leads to incorrect conclusions due to confounders in correlational analysis, and serve as a warning about correlational pitfalls in natural-language settings. 

---
# Data Synthesis and Parameter-Efficient Fine-Tuning for Low-Resource NMT: A Case Study on Q'eqchi' Mayan 

**Authors**: Alexander Chulzhanov, Soeren Eberhardt, Arjun Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.09767)  

**Abstract**: Neural machine translation for digitally low-resource Indigenous languages is often hindered by extreme data scarcity, prompting reliance on extractive web-scraping. To ensure data sovereignty, this study introduces a data synthesis methodology to bootstrap NMT models without scraping target-language parallel text. Focusing on Q'eqchi' Mayan, we transformed community-sourced dictionaries into a massive synthetic corpus, utilizing Parameter-Efficient Fine-Tuning (PEFT) via LoRA adapters on an mT5-base model.
In-domain evaluation demonstrates high structural acquisition (BLEU 42.02), proving that synthetic constraints effectively teach complex agglutinative morphology and VOS word order. However, evaluation against an organic glossary reveals a structural-semantic gap (BLEU 0.59), where the model maintains grammatical integrity but lacks the lexical grounding of natural language. The model exhibits overfitting to the constrained structural variance of the synthetic templates; despite high semantic entropy in the pipeline, it struggles with the syntactic fluidity of natural language, forcing organic inputs into rigid learned patterns. Furthermore, an ablation study utilizing a Multi-Task Learning architecture resulted in negative transfer, suggesting that auxiliary tasks competed for limited parameter capacity within the LoRA adapters, causing over-optimization for synthetic markers at the expense of organic flexibility. Ultimately, we establish that synthetic bootstrapping is a highly effective structural primer, but requires authentic data for semantic refinement via Curriculum Learning. 

---
# The Neutral Mask: How RLHF Provides Shallow Alignment while Leaving Partisan Structure Intact in a Large Language Model 

**Authors**: Wendy K. Tam  

**Link**: [PDF](https://arxiv.org/pdf/2606.09735)  

**Abstract**: The ambition behind alignment training is to make large language models safe and useful. The primary mechanism, reinforcement learning from human feedback (RLHF), shapes the behavior of deployed language models by aligning them with ``human values.'' Yet the process is opaque. What values are being encoded; whose values are they; and how does RLHF encode them? A growing body of evidence suggests that RLHF produces only functional compliance rather than deep alignment. We offer a mechanistic case study of this phenomenon for partisan political orientation with a comparison of the internal representations of Llama 3.1 8B before and after RLHF. We show that RLHF does not remove the structured partisan direction in the base model. Instead, it compresses the variance of the partisan signal to generate consistently balanced and non-partisan output. Sparse autoencoder decomposition reveals that policy-encoding features, which activate sporadically in the base model, are completely inactive in the Instruct model. Feature-level steering experiments confirm the causal disconnect. RLHF thus encodes a norm of political neutrality, not by erasing the model's knowledge of partisanship, but by severing the causal pathway from partisan geometry to output generation. Importantly, this neutrality is functional, not structural so that the underlying geometry that enables partisan steering remains intact. The mechanisms that bypass RLHF's guardrails, such as inferring and amplifying a user's partisan identity, reactivate partisan generation. If RLHF operates by disconnecting rather than removing value-laden structure, then the same pattern may hold for other value domains, and the aligned model's behavior may be more fragile than its outputs suggest. 

---
# IS-CoT: Breaking the Long-form Generation Collapse via Interleaved Structural Thinking 

**Authors**: Zechen Sun, Yuyang Sun, Zecheng Tang, Juntao Li, Wenpeng Hu, Wenliang Chen, Zhunchen Luo, Guotong Geng, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09709)  

**Abstract**: Generating coherent and controllable long-form content remains a persistent challenge for Large Language Models (LLMs). While reasoning-enhanced models have demonstrated success in logic-intensive domains, our evaluation reveals that they suffer from a severe length collapse in open-ended writing, where performance degrades sharply as target lengths exceed 2,000 words. We attribute this failure to the limitation of static hierarchical planning, which struggles to provide dynamic guidance over extended contexts. To bridge this gap, we introduce the Interleaved Structural Chain-of-Thought (IS-CoT) framework. Unlike external agentic workflows, IS-CoT embeds a dynamic Plan-Write-Reflect cycle into the generation process, enabling continuous strategy adaptation and global alignment without additional assistance. Based on this framework, we construct a high-quality dataset of interleaved reasoning traces via a multi-teacher pipeline and train IS-Writer-8B. Experiments demonstrate that IS-Writer-8B achieves state-of-the-art performance on challenging long-form benchmarks (e.g., +3.08 vs. DeepSeek-V3.2 on LongBench-Write), exhibiting robust length compliance and coherence competitive with significantly larger proprietary models. 

---
# Learning to Attack and Defend: Adaptive Red Teaming of Language Models via GRPO 

**Authors**: Blake Bullwinkel, Eugenia Kim, Amanda Minnich, Mark Russinovich  

**Link**: [PDF](https://arxiv.org/pdf/2606.09701)  

**Abstract**: AI red teaming must continually adapt to evolving attackers and defenders. Reinforcement learning offers a promising approach to discovering novel attacks, and co-training methods can produce more robust defenders in tandem. Recent works have demonstrated the efficacy of attacker-defender co-training by applying PPO and DPO, but report that GRPO is unstable in this setting. We introduce AdvGRPO, a co-training framework that makes GRPO viable for joint attacker-defender optimization using dense multi-channel rewards and decoupled advantage normalization. Training progresses through a curriculum from single-turn to closed-loop multi-turn attacks before bootstrapping co-training, where attacker and defender models are updated in alternation. We show that our method can produce highly effective and transferable attacks and that co-trained defenders outperform baselines on safety benchmarks. 

---
# PsychoSafe: Eliciting Psychologically-Informed Refusals in Large Language Models 

**Authors**: Gianluca Barmina, Federico Torrielli, Sven Harms, Jacob Nielsen, Felix Mächtle, Stine Lyngsø Beltoft, Peter Schneider-Kamp, Thomas Eisenbarth, Lukas Galke Poech, Anne Lauscher  

**Link**: [PDF](https://arxiv.org/pdf/2606.09697)  

**Abstract**: Large language models (LLMs) routinely face requests that should be refused, creating a trade-off between helpfulness and harm prevention. However, refusals themselves can be helpful. In high-risk interactions involving crisis, coercion, or escalating intent, blunt non-compliance may prevent direct harm while still failing to support the needs of the person behind the request. We present PsychoSafe, a psychologically-informed refusal framework that reframes refusal as structured supportive communication grounded in evidence-based intervention strategies. To develop PsychoSafe, we construct a corpus of 8019 prompt-response pairs spanning five psychologically salient risk domains and apply prompting and parameter-efficient fine-tuning to Qwen 3.5 27B. On a balanced validation set of 500 prompts, evaluated with an LLM judge and validated through human ratings, PsychoSafe prompting improves overall refusal quality by 28.1% over a generic baseline, with particularly strong gains in external resource referral (+46.8%) and psychological grounding (+34.8%), while preserving downstream performance on non-refusal tasks. Fine-tuning achieves near-perfect refusal and resource-referral rates but reduces response relevance. Additional evaluations on SORRY-Bench and XSTest show strong in-domain robustness but limited out-of-domain generalization, suggesting that future work should diversify fine-tuning data to help models apply interventions selectively rather than schematically. 

---
# When Built-in Thinking Helps and Hurts: Constraint-Level Error Shifts in Instruction Following 

**Authors**: Sai Adith Senthil Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2606.09662)  

**Abstract**: Large reasoning models (LRMs) often improve math and coding performance, but their effect on instruction following is unclear. We study IFEval with Qwen3 models (1.7B-32B), using same-weights Thinking ON/OFF controls; four Hunyuan models provide directional cross-family support. Aggregate pass-rate changes are small (-0.55 to -3.52 pp), yet 10-20% of prompts switch between pass and fail across modes, suggesting that thinking changes the pattern of errors--some prompts improve while others worsen--rather than uniformly degrading performance. Under a post-hoc Qwen3-derived grouping, constraint types separate into Planning (global counting, structure, coordination), which improves at the class level under thinking, and Precision (exact local form), which consistently worsens; the class-level Planning/Precision sign pattern holds directionally for all four Hunyuan models despite Hunyuan's opposite aggregate direction. Thinking also changes final-answer length; matched-length analyses substantially reduce the Precision drop, but a residual penalty remains. Analyzing thinking traces with a cross-encoder relevance metric reveals three patterns: Neutral shows a positive relevance-compliance link (r approximately 0.15); Planning shows near-zero predictive correlation (r approximately 0.02) despite measurable trace engagement, consistent with an execution gap between CE-measured trace relevance and final-answer compliance; Precision shows a small negative correlation (r approximately -0.05), with failing instances having higher mean relevance than passing ones. Activation patching across four model sizes (1.7B-14B) shows that Precision flip instances are more often restored than Planning flip instances (32-58% vs. 14-40% mean layer-restoration), with the largest gap at 14B (about 30 pp). 

---
# End-to-End Context Compression at Scale 

**Authors**: Ang Li, Sean McLeish, Haozhe Chen, Nimit Kalra, Zaiqian Chen, Artem Gazizov, Venkata Anoop Suhas Kumar Morisetty, Bhavya Kailkhura, Harshitha Menon, Zhuang Liu, Brian R. Bartoldson, Tom Goldstein, Sanae Lotfi, Micah Goldblum, Pavel Izmailov  

**Link**: [PDF](https://arxiv.org/pdf/2606.09659)  

**Abstract**: Long-context language model inference is bottlenecked by memory, as the KV cache grows with context length. Recent techniques to compress the KV cache fall short: they either degrade model quality substantially or require considerable time and compute to compress a single long prompt. Furthermore, many methods require the input to fit within the target model's context window, and are generally incompatible with modern production inference engines. Encoder-decoder compressors, which map a long token sequence to a shorter sequence of latent embeddings consumed by a decoder, are an appealing alternative in principle. However, existing approaches are not competitive with KV cache compression on the accuracy-efficiency frontier. In this work, we revisit encoder-decoder compression and close this gap. We first perform an architecture search, pre-training many variants from scratch to determine how best to design and train encoder-decoder compressors. Guided by our findings, we continually pre-train a family of 0.6B-encoder, 4B-decoder models on over 350B tokens each, at compression ratios of 1:4, 1:8, and 1:16. We introduce Latent Context Language Models (LCLMs), a family of compressors that improve the Pareto frontier across general-task performance, compression speed, and peak memory usage. We demonstrate that LCLMs serve as efficient backbones for long-horizon agents, letting the agent skim through a compressed long context and adaptively expand relevant segments on demand. 

---
# Beyond Accuracy: Community Perspectives on Machine Translation 

**Authors**: Yujun Wang, Ehud Reiter, Shimei Pan, Steffen Eger, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.09655)  

**Abstract**: Despite remarkable progress in machine translation (MT), non-AI communities have raised growing concerns about MT systems, suggesting a noticeable gap between technical advancement and the needs of real-world users. For instance, while NLP researchers focus on benchmark performance, end users care about ethical concerns, trust, reliability, costs, and more. We argue that listening to various user communities is essential so that research efforts would be directed towards the problems that the communities care about. To this end, we present a large-scale analysis, for the first time, that investigates what four stakeholder communities (AI developers, professional translators, language learners, and language service providers) post about MT technology on social media. To do so, we construct a dataset of 79,286 posts and comments from Reddit, Facebook, Bluesky, and Mastodon from 2019 to 2025, and analyse where these communities disagree, and how and why. Overall, we find that communities often disagree, and even show strong conflicts due to polarised sentiments on topics such as translation quality, efficiency, and reliability. This is because these communities approach these topics differently: the AI community frames them as technical and computational problems, while non-AI (user) communities care more about quality nuances, time savings, user trust, and broader social issues. 

---
# Where Does the Answer Come From? Benchmarking View-Level Visual Evidence Identification in Multi-View MLLMs for Autonomous Driving 

**Authors**: Yimu Wang, Yee Man Choi, Barry Zhang, Mozhgan Nasr Azadani, Sean Sedwards, Krzysztof Czarnecki  

**Link**: [PDF](https://arxiv.org/pdf/2606.09644)  

**Abstract**: Multimodal large language models (MLLMs) achieve strong results on visual reasoning benchmarks, but answer accuracy alone does not indicate whether a model relied on the correct visual evidence. This gap is particularly important in multi-view driving scenes used for autonomous driving, where a model can produce a plausible answer while grounding it in the wrong camera view. We introduce a multi-view visual question answering benchmark for evaluating evidence-source identification: given six synchronized NuScenes views and a question, the model must identify the supporting camera view and answer the question. The benchmark contains 122 conflict-centric question-answer pairs from 73 scenes, spanning causality, counterfactual reasoning, and intent prediction. View labels are proposed by an automatic conflict-mining pipeline and manually verified by annotators. We evaluate three settings: camera-view selection, oracle QA given the golden view, and joint prediction in which the model selects a view and answers in one pass. Answers are evaluated in both multiple-choice and free-form formats, using exact match for structured predictions and an LLM judge for free-form responses. By explicitly separating visual-source identification from answer correctness, the benchmark exposes grounding failures that answer-only evaluation misses. 

---
# Gradient-Guided Reward Optimization for Inference-time Alignment 

**Authors**: Hankun Lin, Ruqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09635)  

**Abstract**: Ensuring the reliability of Large Language Models (LLMs) under distribution drift requires inference-time adaptation. While inference-time alignment methods such as Best-of-$N$ and rejection sampling are widely used, they frame the task as a sampling-intensive, reward-guided search, leading to two key limitations: their performance is bounded by the base model's generation quality, and their reliance on imperfect reward models makes them vulnerable to reward hacking. To address these challenges, we introduce Gradient-Guided Reward Optimization (GGRO), a lightweight inference-time method that performs targeted, minimal intervention during decoding via gradient guidance. Specifically, GGRO monitors token-level entropy to identify high-uncertainty regions indicative of drift or misalignment. Upon detection, it responds by injecting nudging tokens, generated using gradient signals from an off-the-shelf reward model, to steer the generation trajectory rather than merely re-ranking samples. Experiments show that GGRO consistently improves inference-time alignment across safety, helpfulness, and reasoning benchmarks. It also increases coverage of high-quality responses and robustness to reward hacking, with minimal computational overhead. Code is available at this https URL. 

---
# Civil Court Simulation with Large Language Models 

**Authors**: Yifan Chen, Haitao Li, Kaiyuan Zhang, Yueyue Wu, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09632)  

**Abstract**: Court simulation bridges legal education and judicial practice, yet human-based simulations are costly and difficult to scale. Large language models (LLMs) offer a scalable alternative, but existing court-simulation research mainly focuses on criminal cases. Civil litigation is more common in practice and harder to simulate because its claims, liability, and remedies are more flexible. We present a multi-agent court simulation framework for Chinese civil cases. The framework organizes role-based interaction through a five-stage civil trial procedure and integrates memory module and statute retrieval to support long-process adjudication. Experiments show that the framework produces reliable civil judgments, with clear strengths in liability allocation and multi-item adjudication. Further experiments show that memory quality substantially affects downstream simulation quality. Through a five-layer factor framework, we analyze how legal grounding, information conditions, judicial capability and role orientation, organizational pressure, and social context affect the framework's reliability and behavior. These results support the effectiveness of the proposed framework for civil court simulation. The dataset and code are available at: this https URL. 

---
# AGENTSERVESIM: A Hardware-aware Simulator for Multi-Turn LLM Agent Serving 

**Authors**: Rakibul Hasan Rajib, Mengxin Zheng, Qian Lou  

**Link**: [PDF](https://arxiv.org/pdf/2606.09613)  

**Abstract**: Multi-turn LLM agents interleave model calls with external tool invocations, shifting serving from stateless request processing to stateful program execution. Serving these workloads requires scheduling, KV-cache management, and routing policies that use program-level context, including turn dependencies, tool-induced gaps, and reusable KV state. Evaluating such policies directly on real systems is costly, since each design point may require dedicated accelerator time across arrival rates, model scales, serving-instance counts, and memory hierarchies. Simulation offers a scalable alternative, but existing LLM serving simulators target stateless request-level workloads and therefore omit the core dynamics of agent serving: multi-turn program execution, cross-turn cache locality, and KV-cache residency during tool gaps. We present AGENTSERVESIM, a hardware-aware simulator for multi-turn LLM agent serving. AGENTSERVESIM evaluates serving policies at program granularity through composable modules: a Program Orchestrator preserves program identity and turn order, a Tool Simulator materializes tool-induced gaps, a Session-Aware Router maintains program-to-instance affinity for cache-aware dispatch, and a KV Residency Model tracks policy-defined KV placement across HBM, host DRAM/CXL, and eviction. Across real serving deployments and hardware configurations, AGENTSERVESIM reproduces real-system behavior within 6% error across key performance metrics while running entirely on commodity CPUs. These results show that AGENTSERVESIM enables controlled, repeatable exploration of agent-serving policies without requiring exhaustive deployment on costly accelerators. 

---
# Automated IEP Generation from Traditional Chinese Parent-Teacher Interviews via Corpus-Grounded Feature Diffusion 

**Authors**: Kuanlin Chen, Cheng-En Ou  

**Link**: [PDF](https://arxiv.org/pdf/2606.09603)  

**Abstract**: Writing Individualized Education Programs (IEPs) is a high-labor, knowledge-intensive document burden; English-language research has demonstrated that generative AI can significantly reduce drafting time, yet automated IEP generation in Traditional Chinese remains virtually unexplored due to domain data scarcity, strict privacy regulations, and the absence of local evaluation benchmarks. We propose a low-resource fine-tuning pipeline centered on Corpus-Grounded Feature Diffusion (CGFD): (1) 25 dual-expert high-score seed transcripts are selected via a tau threshold with flag-aware score caps; (2) a FeatureProfile (sentence length, structure, quantification templates) is extracted from seeds and injected into LLM prompts alongside Verbalized-Sampling-style diversity control to drive diffusion; (3) 15 expert gold seeds are used as diffusion anchors, targeting 585 samples; 567 valid diffusion samples are obtained, yielding a 582-sample training set used to fine-tune Breeze-7B with QLoRA; (4) schema-constrained inference via Grammar-Constrained Decoding (GCD) enforces a hierarchical SMART Goal Ladder schema at inference time. Ablation results on a 55-sample schema stress set reveal an unexpected finding: GCD is counterproductive under Traditional Chinese token budgets -- the no-GCD path achieves 100% schema pass rate at 34% lower median latency, outperforming GCD on both reliability and speed. On the n=10 formal hold-out, the no-GCD inference path achieves BERTScore F1 = 0.779, exceeding GPT-5.4 (0.726), DeepSeek-V3.2 (0.703), Gemini-3-Flash-Preview (0.703), and Llama-4-Maverick (0.700) zero-shot baselines while maintaining fully local, air-gapped inference. This system addresses a gap in Traditional Chinese special-education NLP and offers a scalable, privacy-preserving local inference solution under an industrial engineering paradigm. 

---
# Clinically Grounded Privacy Evaluation of Medical LMs 

**Authors**: Sasha Ronaghi, Sana Tonekaboni, Lena Stempfle, Vivian Utti, Jordan Li Cahoon, Nathaniel Hendrix, Ayin Vala, Marzyeh Ghassemi, Emily Alsentzer  

**Link**: [PDF](https://arxiv.org/pdf/2606.09590)  

**Abstract**: Medical language models (LMs) can memorize and reproduce protected health information, but privacy evaluations often focus on recovery of training text rather than disclosure under realistic threat models. We introduce a clinically grounded framework that evaluates leakage along a graded axis of adversarial access, ranging from publicly inferable demographics to leaked note fragments. At each tier, we measure verbatim memorization of patient-specific text and semantic leakage of sensitive diagnoses. Applying the framework to an LM pretrained on 378k clinical notes, we find that routine encounter metadata (i.e. name, date of birth, provider, practice, visit date) elicits high rates of verbatim memorization across a patient's timeline and sensitive-diagnosis recovery (AUROC 0.91 for abortion, 0.81 for HIV). At the same time, exact-match memorization can overstate disclosure: 36% of memorized tokens reflect templated documentation. Our work highlights the risks of training on longitudinal clinical data, providing a practical framework for contextual privacy evaluation of medical LMs. 

---
# Code Is More Than Text: Uncertainty Estimation for Code Generation 

**Authors**: Yuling Shi, Caiqi Zhang, Yuexian Li, Haopeng Wang, Yeheng Chen, Nigel Collier, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09577)  

**Abstract**: Large language models (LLMs) are increasingly deployed as code generators, where silently wrong programs pose real safety and reliability risks. Reliable uncertainty estimation (UE) is essential for selective prediction, human-in-the-loop review, and downstream agentic decisions. Yet most existing code UE methods are inherited from natural language (NL) generation and ignore properties that make code distinct. We argue that code differs from NL in three ways: a single wrong token can break an entire program (token fragility); algorithmic intent and concrete implementation can disagree independently (intent-code gap); and programs can be executed (executability). We instantiate these properties as three orthogonal uncertainty axes: lexical (Top-K token entropy), algorithmic (pseudo-code consistency), and functional (behavioral consistency). Across five code LLMs, our three-axis ensemble improves average AUROC from 0.696 for the strongest NL-derived baseline to 0.776 (+8.1 points). Notably, on Qwen3-14B, our single-pass Top-K token entropy matches the strongest multi-pass baseline while being over 3x cheaper; across models, it remains a competitive low-cost signal. These results suggest that code UE deserves code-specific design rather than direct NL ports. 

---
# UXBench: Benchmarking User Experience in AI Assistants 

**Authors**: Mengze Hong, Xia Zeng, Zeyang Lei, Sheng Wang, Chen Jason Zhang, Di Jiang, Taiming Fu, Jinfeng Huang, Mengqiao Liu, Qinghe Chang, Haosheng Zou, Qiongyi Zhou, Sijun He, Chen Xiaoshuai, Simon Deng, Haojing Huang, Zijian Li, Lucas Mu Li, Fubao Zhang, Mona Zhou, Wei Ma, Chenxuan Ma, Yuanmeng Zhang, Jian Song, Minlong Peng, Di Liang, Davey Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.09570)  

**Abstract**: As AI assistants serve millions of users daily, evaluating user experience (UX) beyond general model capability has become increasingly important. We present UXBench, the first user-centric benchmark grounded in real user feedback signals for evaluating preference alignment and dialogue generation. The benchmark consists of three interconnected tasks, UX Judge, UX Eval, and UX Recovery, with 7,400 test instances extracted from over 70K interaction logs of a mainstream Chinese AI assistant. The dataset closely reflects real user distributions, covering 8 scenarios, 83 domains, and diverse failure patterns that pose severe challenges. Extensive experiments on 26 frontier language models provide novel insights into how well models perceive user experience and how improvements in model capability contribute to better dialogue engagement. Through comprehensive analysis of model behavior and performance gaps, we show that user feedback prediction is a learnable capability, where a reward model trained from in-the-wild feedback signals can achieve well-calibrated accuracy. We further document the systematic biases of LLM-as-a-judge evaluation protocols and compare typical response strategies that directly affect user experience. UXBench establishes a new evaluation landscape and calls for greater attention to tailored UX optimization, contributing to a user-centric scaling law that shapes the success of AI assistants. 

---
# OpenBibleTTS: Large-Scale Speech Resources and TTS Models for Low-Resource Languages 

**Authors**: David Guzmán, Luel Hagos Beyene, Jesujoba Oluwadara Alabi, Yejin Jeon, Dietrich Klakow, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2606.09553)  

**Abstract**: Recent advances in neural text-to-speech (TTS) and multilingual speech generation have substantially improved synthetic speech quality, yet these gains remain unevenly distributed across the world's languages. Existing models are still dominated by a small set of high-resource languages, while many studies of low-resource TTS are simulated on artificially downsampled high-resource corpora that do not reflect the orthographic variation and limited phonetic coverage encountered in genuinely underrepresented settings. As such, we introduce OpenBibleTTS, which is a large-scale benchmark for low-resource speech synthesis spanning 37 underrepresented languages. Moreover, a systematic comparison of various TTS architectures and large-scale speech generation models is conducted across in-domain Biblical text and out-of-domain material. Results show that no single system dominates across languages and metrics: Gemini-TTS achieves the highest listener ratings on most evaluated languages, but monolingual EveryVoice models trained on OpenBibleTTS remain strongest for intelligibility and are preferred in several African languages, while open from-scratch systems degrade sharply on out-of-domain text, revealing a persistent gap between broad multilingual coverage and reliable synthesis quality in underserved linguistic communities. We complement automatic evaluation with subjective human judgments, and open-source all processed datasets, alignments, and trained models to support future low-resource TTS research. 

---
# From Genes to Tokens: a GWAS-inspired Approach for Interpretable Stylometric Analysis 

**Authors**: Dmitry Pronin, Evgeny Kazartsev  

**Link**: [PDF](https://arxiv.org/pdf/2606.09543)  

**Abstract**: This short paper introduces a stylometric interpretation method inspired by genome-wide association studies (GWAS). Each "gene" token's association with "phenotype" authorship is tested using logistic regression with multiple-comparison correction. Applied to English, German, and Russian corpora, the method detects statistically significant lexical markers distinctive of individual authors. 

---
# Overcoming Decoder Inconsistencies in Whisper for Dravidian and Low-Resource Languages 

**Authors**: Chowdam Venkata Kumar, Kumud Tripathi, Pankaj Wasnik  

**Link**: [PDF](https://arxiv.org/pdf/2606.09535)  

**Abstract**: Multilingual ASR models such as Whisper perform well on high-resource languages but exhibit substantially higher Word Error Rates (WER) for Dravidian languages compared to Indo-Aryan ones. Through linguistic and dataset analysis, we show that Dravidian languages have longer words, higher vocabulary diversity, and lower repetition, resulting in sparse token distributions and frequent character-level substitution errors. Baseline fine-tuning further reveals decoder imbalance between self-attention (linguistic context) and cross-attention (acoustic cues). Although synthetic token-repetition experiments indicate potential gains, they are impractical. Motivated by these observations, we introduce two decoder-level enhancements: Weighted-Attention, which adaptively balances attention sources, and Self-Conditioning, which reinjects intermediate predictions to improve token consistency. Experiments demonstrate consistent WER reductions for low-resource and agglutinative languages. 

---
# Emergence of Context Characteristics Sensitivity in Large Language Models 

**Authors**: Nadya Yuki Wangsajaya, Haeun Yu, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2606.09525)  

**Abstract**: During instruction fine-tuning (IFT), large language models (LLMs) learn to follow instructions by using the provided context to answer a query. While prior work has studied how context characteristics correlate with context usage by the LLM, this analysis has been limited to inference time, leaving open how these relationships are acquired in the first place. Here, we measure how models' sensitivity to such characteristics shifts across successive IFT stages: supervised fine-tuning (SFT), direct preference optimization (DPO), and reinforcement learning with verifiable rewards (RLVR). Experiments across four models and three datasets show that SFT makes models more likely to use contexts that are easy to understand, such as containing high length, context-query similarity, and fluency. Post-SFT dynamics may either reinforce or resolve these preferences depending on the training dataset. Our findings reveal that context usage is actively reshaped at each IFT stage, and designing a balanced IFT dataset is important in ensuring robust context utilization of instruction-tuned models. 

---
# Self-Harness: Harnesses That Improve Themselves 

**Authors**: Hangfan Zhang, Shao Zhang, Kangcong Li, Chen Zhang, Yang Chen, Yiqun Zhang, Lei Bai, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09498)  

**Abstract**: The performance of LLM-based agents is jointly shaped by their base models and the harnesses that mediate their interaction with the environment. Because different models exhibit distinct behaviors, effective harness design is inherently model-specific. Yet agent harnesses are still largely engineered by human experts, a paradigm that scales poorly as modern LLMs become increasingly diverse and rapidly evolving. In this paper, we introduce Self-Harness, a new paradigm in which an LLM-based agent improves its own operating harness, without relying on human engineers or stronger external agents. We operationalize Self-Harness as an iterative loop with three stages: Weakness Mining, which identifies model-specific failure patterns from execution traces; Harness Proposal, which generates diverse yet minimal harness modifications tied to these failures; and Proposal Validation, which accepts candidate edits only after regression testing. We instantiate Self-Harness on Terminal-Bench-2.0 using a minimal initial harness and three base models from diverse families: MiniMax M2.5, Qwen3.5-35B-A3B, and GLM-5. Across all three models, Self-Harness consistently improves performance, with held-out pass rates increasing from 40.5% to 61.9%, 23.8% to 38.1%, and 42.9% to 57.1%, respectively. Qualitative analyses further show that Self-Harness does not simply add generic instructions, but effectively turns model-specific weaknesses into concrete, executable harness changes. These results suggest a path toward LLM-based agents that are not merely shaped by their harnesses, but can also participate in reshaping them. 

---
# Detecting Differences Is Not Understanding Structure: Large Language Models Fail at Graph Isomorphism 

**Authors**: Kumar Thushalika, Sukumar Kishanthan, Asela Hevapathige  

**Link**: [PDF](https://arxiv.org/pdf/2606.09484)  

**Abstract**: Large language models (LLMs) have shown impressive performance on diverse reasoning tasks, yet their capacity for structural reasoning in graphs remains unclear. We investigate whether LLMs can genuinely understand graph isomorphism -a fundamental problem in graph theory. While LLMs achieve near-perfect accuracy on isomorphism detection, we show this performance is illusory. When identical graphs are presented with permuted node labels, LLMs fail to identify their isomorphism. This finding suggests that LLMs exploit patterns rather than reasoning about abstract graph structure. Since permutation invariance is a fundamental requirement for valid structural reasoning, these results indicate that success on graph reasoning benchmarks should not be interpreted as evidence of genuine topological understanding. 

---
# Memory Beyond Recall: A Dual-Process Cognitive Memory System for Self-Evolving LLM Agents 

**Authors**: Tianxiang Fei, Mingyang Song, Mao Zheng, Xiang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09483)  

**Abstract**: Long-term memory for an LLM agent is more than retrieving the right passage at the right time. Current memory systems collapse belief revision, causal coupling, and cross-domain abstraction into a single retrieval surface tuned for surface recall, and consequently struggle on implicit personalisation that requires reasoning over how a user has evolved. We propose DCPM, which reorganises agent memory along a cognitive capability hierarchy ascending from raw inputs and atomic facts, through diachronic belief trajectories and identity, to domain schemas, latent intentions and cross-domain patterns. The hierarchy is driven by two processes inheriting the architectural split of dual-process theory: a synchronous daytime writer (System1) that records belief revisions as doubly linked supersedes chains, and an asynchronous nighttime engine (System2) that induces schemas and intentions and sweeps for cross-domain collisions abstracted into higher-level core schemas. On LongMemEval, PersonaMem and PersonaMem-v2, enabling System2 contributes most where the benchmark rewards implicit cross-session inference (up to +5.20 on PersonaMem-v2) and least on span recall, matching the architectural prediction. 

---
# A Finetuned SpeechLLM for Joint Multi-Granular L2 Assessment and Natural-Language Rationales 

**Authors**: Aditya Kamlesh Parikh, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2606.09470)  

**Abstract**: Automated L2 speech assessment can assign proficiency labels, but often lacks interpretability. We propose a rubric-guided SpeechLLM for multi-aspect, multi-granular assessment, trained with a hybrid objective combining supervised fine-tuning and Bounded Direct Preference Optimization. The model jointly predicts ordinal labels at the sentence-level (accuracy, fluency, prosody), word/phoneme-level accuracy, and generates a natural-language rationale in the same response. On SpeechOcean762, our approach matches or outperforms single-granularity models while remaining competitive with prior approaches. We analyze rationale reliability along two axes: self-consistency with model predictions and alignment with ground-truth labels, using sentiment consistency (plausibility) and mention-based agreement (faithfulness). Rationales are plausible at the sentence level, but faithfulness degrades at the word/phoneme level: references are sparse and weakly aligned with token-level labels. 

---
# DECSELFMASK: Leveraging Unlabeled Text via Self-Relevance-Guided Masking for Decoder-Only Classification 

**Authors**: Pietro Ferrazzi, Matteo Merler, Giovanni Bonetta, Alberto Lavelli, Bernardo Magnini  

**Link**: [PDF](https://arxiv.org/pdf/2606.09466)  

**Abstract**: Classification tasks require annotated data, which can often be expensive, time-consuming, or even unfeasible to collect. This is the case of the medical domain, where large datasets often have few annotated examples. To address this, we propose DecSelfMask (Decoder Self-learning by Masking), an approach to enhance decoder-only performance on classification tasks. We build on common self-learning approaches by leveraging a model to create training examples from unlabeled data to propose a novel relevance-guided masking strategy. We use relevance attribution methods to determine what portions of unannotated texts are relevant for a task. We then create self-supervised training examples by masking out those portions, training the model to reconstruct them via next-token-prediction. We hypothesize that those examples convey knowledge about the structure and semantics of unannotated data that can be useful for downstream performance. We test our approach on 136 tasks from a collection of 1.9M clinical notes from an Italian hospital. We quantify DecSelfMask's impact on downstream tasks on 5 models of different scales and families, including a probing analysis. Experiments show consistent gains, outperforming standard supervised fine-tuning approaches (+19.9 points in Macro F1), synthetic label generation (+12.5), and continual pretraining (+6.3), as well as common baselines. 

---
# H2HMem: A Multimodal Memory Benchmark for Agents in Human-Human Interactions 

**Authors**: Shiping Zhu, Yibo Yang, Zhengyang Wang, Tiancheng Shen, Dandan Guo, Ming-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09461)  

**Abstract**: Large language model agents are increasingly deployed in human-human interaction settings, such as meeting assistants and clinical documentation systems, where they must observe conversations and retain information for downstream queries. Unlike traditional human-assistant settings, these environments are inherently multimodal, involve complex discourse phenomena such as anaphora and deixis, and contain asynchronous or conflicting information from multiple participants. However, existing memory benchmarks largely focus on single-user, text-only interactions, failing to capture these challenges. To address this gap, we introduce H2HMem, a Human-to-Human Multimodal Memory Benchmark for evaluating memory capabilities in complex human-human interactions. H2HMem includes both dyadic and multi-party conversations with multimodal information streams, and evaluates agents along three dimensions: memory recall, reasoning, and application. Experiments with advanced agents reveal substantial limitations in constructing, retaining, and utilizing memories across modalities, participants, and sessions, highlighting substantial room for improvement in next-generation LLM agents. 

---
# AbstRAG: Learning to Abstract for Retrieval Problems 

**Authors**: Lei Xu, Xin Quan, Daniel Pedronette, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2606.09459)  

**Abstract**: Retrieval-augmented generation often fails when the query, the document evidence, and the user's intent are expressed at different levels of abstraction. A query may ask about a class, a relation, or an event, while the document only states specific instances, indirect framings, or scoped formulations. We define this mismatch as an abstraction gap: the minimal set of typed assumptions required to align query intent with the available evidence. To close this gap, we introduce AbstRAG, which treats abstraction as an explicit retrieval object. AbstRAG decomposes the query--evidence gap into expression, conceptual, intent--evidence, and event-type components, and scores relevance by combining match quality, a query-independent utility prior, and the cost of the required bridges. Its central mechanism is reflective refinement: a critic diagnoses retrieval failures, localizes the failed abstraction operator, proposes a minimal stage-specific patch, and accepts the patch only under sufficiency and compression controls. Across three within-document retrieval benchmarks against seven baselines, AbstRAG outperforms on nDCG@10 in 18 of 21 paired-bootstrap contrasts and improves generation accuracy by 1.9%, 5.2%, and 4.0% across the three benchmarks; ablations confirm that reflective refinement drives most of the retrieval gain and the compression control alone reduces over-expansion false positives from 73.7% to 0% on a stress slice. 

---
# Reasoning without Gold Standards: A Proxy-Judge Theory of Autoformalization 

**Authors**: Lei Xu, Xin Quan, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2606.09449)  

**Abstract**: Complex reasoning tasks increasingly require systems to produce outputs whose correctness cannot be judged by exact match against a single reference. Autoformalization (AF) is a representative example; it asks a model to translate informal mathematical or logical reasoning into a formally checkable object, yet expert-validated formalizations do not scale beyond toy cases and a single informal argument can admit many valid formal renderings. Progress therefore depends on whether partial, structured proxies can substitute for exact references.
We introduce a reference-free proxy-judge framework for AF that replaces gold-standard matching with a vector of per-axis property checks. The framework organizes the proxy along three structural scopes that cover global properties of the elicited object, per-module properties internal to its sub-components, and cross-domain properties that re-align it to the informal source, and aggregates each axis into a verdict vector. The vector drives a reflective refinement loop in which a violated coordinate routes the controller to a matching repair target, so each iteration changes only what is judged wrong.
Under bounded judge noise, the expected intrinsic gap contracts geometrically to a noise-dependent plateau. Across seven formalization backbones on miniF2F, ProofNet, e-SNLI, and ProntoQA, refinement consistently lifts Pass Rate over the single-shot ICL baseline, and the per-axis proxy outperforms a matched scalar proxy on benchmarks where the baseline has room to improve. Structured proxy judgments therefore provide both a practical refinement signal and a theoretical handle on convergence when exact references are unavailable. 

---
# MUDIDI: A Two-Stage Framework for Multilingual Dictionary Digitization with Language Models 

**Authors**: David Setiawan, Temuulen Khishigsuren, Milind Agarwal, Pagnarith Pit, Aso Mahmudi, Ekaterina Vylomova  

**Link**: [PDF](https://arxiv.org/pdf/2606.09435)  

**Abstract**: Multilingual dictionaries are among the most valuable documentary resources for low-resource and endangered languages, yet many remain available only as scans. For many decades, their digitization and conversion into a machine-readable format was nearly impossible due to language-specific scripts, complex multi-column layouts full of entries with abbreviations and cross-references. Recent vision-language models offer a promising solution, but it is unclear how well they preserve characters, markup, and process lexicographic structure. We introduce MUDIDI, a two-stage framework for multi-lingual dictionary digitization. Stage One evaluates the quality of character recognition and markup preservation; Stage Two focuses on dictionary entry segmentation with subsequent mapping into a machine-readable lexicographic schema, SIL's Multi-Dictionary Formatter. We also release a dataset that consists of human-annotated lexicographic entries collected from 30 public-domain dictionaries featuring diverse writing systems, language families, and formats. We benchmark OCR systems, general-purpose Large Language Models (LLMs), and Vision Language Models (VLMs) on the dataset, demonstrating superior performance of LLMs across most writing systems and languages in both stages, and provide practical guidelines on improving the results for more challenging scenarios. Finally, we show that supplementing additional information, such as dictionary introduction, to the LLMs can improve the quality of the digitized dictionary. Github: this https URL 

---
# Guide Me Out: A Framework to Benchmark VLM Operators Communication in Crisis Scenarios 

**Authors**: Giacomo Gonella, Stefano Menini, Marco Guerini  

**Link**: [PDF](https://arxiv.org/pdf/2606.09428)  

**Abstract**: Effective crisis response requires spatially grounded communication that bridges linguistic guidance of civilians with the physical environment, accounting for structural bottlenecks, evolving threats, and agent-specific contexts. Yet, current NLP research in crisis communication remains mainly limited to static, text-only classification settings, overlooking the critical communicative role of AI operators in dynamic, embodied scenarios. We address this gap with a novel benchmarking framework for evaluating Vision-Language Models (VLMs) tasked with guiding civilian agents through simulated evacuations. We test two communication strategies (narrowcast vs. broadcast), two environment representations (visual vs. graph-based), and two threat behaviors (static vs. moving) across nine maps of varying structural complexity. Our results show that Narrowcast consistently reduces civilian Fail rates compared to Broadcast across all difficulty levels. Guidance quality depends heavily on how the VLM operator represents the world: the visual modality drives performance, while adding an adjacency graph is model-dependent and often harmful. Moving threats raise Fail rates across all conditions as communication must continuously adapt over time. Together, these findings show that deploying VLMs as AI operators in evacuation scenarios remains a non-trivial challenge, where the choice of communication strategy and input representation can directly determine the success or failure of the intervention. 

---
# Toward Signing Activity Projection in Sign Language Interaction 

**Authors**: Takao Obi, Wang Yusong, Koji Inoue, Kotaro Funakoshi  

**Link**: [PDF](https://arxiv.org/pdf/2606.09424)  

**Abstract**: Social robots must interact robustly not only with users assumed by speech-centered systems but also with diverse users whose communication relies on different modalities, e.g., sign language. One important capability gap is predictive turn-taking with signing users. Although Voice Activity Projection (VAP) has been successfully used to model future voice activity in spoken interaction, it remains unclear whether the framework transfers to sign language interaction. This paper presents an initial transfer study of adapting a VAP architecture to dyadic sign language interaction. Using interaction recordings from the Public DGS Corpus, we derive binary signing activity streams from lexical sign annotations and formulate proxy tasks for turn-taking prediction. The model uses pose-derived hand, eye-region, and mouth-region features extracted for each signer. The results show that SHIFT/HOLD prediction is promising, especially with hand cues, while SHIFT-prediction remains difficult. These findings provide initial evidence for both the promise and the current limitations of transferring predictive turn-taking models from spoken interaction to sign language interaction. Predictive modeling of sign language interaction still requires sign-language-specific event definitions that go beyond speech-derived categories. 

---
# What Should a Skill Remember? Quality-Cost Trade-offs in Cost-Aware Skill Rewriting for Language Model Agents 

**Authors**: Qinghua Xing, Yinda Chen, Yaping Jin, Zhenhe Wu, Bohan Lin, Hang Zhou, Xinghao Chen, Hanting Chen, Zhiwei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2606.09421)  

**Abstract**: Large language model agents increasingly rely on skills: reusable procedural documents encoding workflows, tool use, implementation patterns, validation checks, and domain rules. Skill rewriting is often treated as prompt compression, but shorter skills can make agents more expensive by removing sparse operational anchors that prevent exploration, debugging, and recovery. We study skill rewriting through this economic lens. Our controlled framework profiles skill structure, rewrites skills using information-preservation strategies, and evaluates the rewrites under fixed task instructions, environments, and verifiers. Experiments on SkillsBench reveal distinct quality--cost trade-offs across strategies: API/code anchoring, workflow guarding, and rule/formula anchoring benefit different task families, with no universally dominant template. In the main held-out evaluation, the learned policy reduces total cost by 7.0\% and downstream agent-token cost by 6.0\%; in frozen cross-model transfer, the corresponding reductions average 14.7\% and 13.7\%, while verifier quality is preserved. These results position skill design as cost-aware operational knowledge engineering rather than prompt compression. Resources: \href{this https URL}{SkillEE}. 

---
# Introducing multiplex semantic networks as multifaceted representations of creative associative knowledge across multilingual samples 

**Authors**: Edith Haim, Kurt Haim, Roger E. Beaty, Cynthia S.Q. Siew, Massimo Stella  

**Link**: [PDF](https://arxiv.org/pdf/2606.09403)  

**Abstract**: Creativity is a complex cognitive ability that relies on knowledge organisation and retrieval from semantic memory. Yet most research uses a single task to measure it, capturing only a fraction of this complexity. This study investigates multiplex networks - layered semantic networks obtained from six cognitive tasks - as a more comprehensive approach to modelling the associative knowledge underlying creativity. We collected data from N=518 individuals from four countries (Austria, USA, Singapore, Italy). From their responses to verbal fluency, sentence-chain, free association, and narrative writing tasks, we constructed semantic networks and assembled them in a multiplex structure. AI persona-based responses provided a comparison baseline. Structural reducibility analyses showed that different task layers captured distinct, non-redundant information about semantic organisation, supporting the use of multiple tasks over any single one. The networks from high- and low-creative groups remained structurally distinct, while AI-generated networks showed near-identical structures regardless of creativity group. Finally, we used 12 features (network measures, emotional scores, and spreading activation simulations) in a machine learning model using ridge regression to predict individual creativity scores. The combination of structurally similar layers, as identified in the previous stage, improved a proof-of-concept prediction accuracy by 50%. Structural measures showed the highest feature importance, with spreading activation dynamics providing additional predictive power. Together, these findings indicate that multiplex semantic networks capture a richer, cross-cultural picture of associative knowledge underlying creativity. We also release our diverse dataset and code to foster diverse computational approaches within the creativity community. 

---
# PriFT: Prior-Support Guided Supervised Fine-Tuning 

**Authors**: Ke Wang, Shuangqi Li, Mathieu Salzmann, Pascal Frossard  

**Link**: [PDF](https://arxiv.org/pdf/2606.09396)  

**Abstract**: Supervised fine-tuning (SFT) is an efficient approach for downstream task adaptation and often serves as the initialization stage for reinforcement learning (RL), but it can show weaker generalization than RL. A key limitation is its off-policy objective: SFT fits fixed demonstrations token by token, including targets poorly aligned with the model's pretrained distribution, which can lead to overfitting. A recent line of work addresses this issue by assigning larger training weights to tokens better aligned with the current model's predictive distribution, with the intuition that fitting these tokens are less distortive to the model's pretrained knowledge and representations. However, computing the token weights from the model that is currently fine-tuned entangles token weights with the optimization trajectory, inducing a self-reinforcing dynamics as the distribution rapidly departs from the pretrained model. To address this, we propose PriFT (Prior-support guided Fine-Tuning), which derives token weights from a frozen pretrained reference to obtain a stable reweighting signal unaffected by fine-tuning. This signal estimates prior support: the extent to which each target token is supported by the pretrained distribution. Across multiple existing token-reweighting rules, replacing the reweighting signal from the online model to pretrained model consistently improves performance. We introduce two instantiations: PriFT-prob uses pretrained token probability, while PriFT-mass selects tokens by cumulative probability mass under the pretrained distribution. Extensive experiments on mathematical reasoning, code generation, and medical question answering show that PriFT achieves state-of-the-art results among SFT baselines and provides a better initialization for subsequent RL training. 

---
# LexRubric: A Rubric-Guided Diagnostic Benchmark for Open-Ended Legal Tasks 

**Authors**: Yifan Chen, Haitao Li, Yiran Hu, Kaisong Song, Jun Lin, Yueyue Wu, Qingyao Ai, Min Zhang, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09389)  

**Abstract**: As large language models (LLMs) are increasingly applied to real-world legal tasks, evaluating the reliability of their open-ended legal responses has become essential. These tasks require context-sensitive answers and allow little room for error, motivating fine-grained and diagnostic evaluation that can identify specific sources of response quality failures. We introduce LexRubric, a rubric-based benchmark for evaluating open-ended Chinese legal tasks. LexRubric contains 649 instances from legal consultation and judicial examination, which reflect both everyday legal needs and professional legal reasoning and cover 14 legal scenarios. It further includes 12,337 expert-written atomic scoring criteria organized under a unified six-dimensional framework, enabling accurate evaluation and diagnostic analysis across tasks and evaluation dimensions. To validate the reliability of the evaluation, we test multiple judge models and compare model-based judgments with human judgments. We further evaluate 18 recent general and legal-domain LLMs on LexRubric. Results show that different models exhibit distinct capability profiles, and that open-ended legal question remains challenging for current LLMs. Data is available at: this https URL. 

---
# Precision Is Not Faithfulness: Coverage-Aware Evaluation of Grounded Generation with a Complete Oracle 

**Authors**: Juan S. Santillana  

**Link**: [PDF](https://arxiv.org/pdf/2606.09376)  

**Abstract**: Reference-free faithfulness metrics verify each atomic claim a model makes against ground truth, and are increasingly used to evaluate grounded generation. We show they share a blind spot: they measure only precision -- are the stated claims supported? -- and therefore reward abstention, since a model can score near-perfect faithfulness by saying almost nothing. We make this measurable using Formula 1 telemetry, a domain where strategic ground truth is derived deterministically and, crucially, completely: for each decision we know the full set of facts that mattered. This completeness -- absent in open-domain faithfulness benchmarks -- lets us measure recall (coverage of the relevant facts) exactly, alongside precision. On a multilingual (EN/ES/PT) benchmark of 7,253 decision instances spanning 150 races, the most precise frontier model covers under half of the relevant facts and ranks last by F1, so requiring coverage reorders the systems; the same effect reappears in a second complete-oracle domain (NOAA weather forecasts). A prompt ablation shows the low coverage is not an under-prompting artifact: explicitly asking models to be thorough does not close the gap. We pair faithfulness with coverage into a single score, validate the metric (controlled perturbation; agreement across a model-free regex extractor and a cross-family LLM extractor, system-level Spearman 1.0), and give a verifier-guided generation method that improves precision and recall without references. We release the benchmark, structured annotations, metric, baselines, and an interactive demo. 

---
# Is Text All You Need? Text as a Universal Information Bottleneck for Speech LLMs 

**Authors**: Ming-Hao Hsu, Yuxuan Hu, Shujie Liu, Jinyu Li, Yan Lu, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09366)  

**Abstract**: Large language models (LLMs) provide a powerful reasoning backbone for speech understanding, but integrating continuous acoustic signals into a frozen LLM remains challenging. Existing speech-to-LLM interfaces typically operate at two extremes: either enforcing near-discrete token alignment, which benefits transcription but loses paralinguistic information, or learning unconstrained continuous representations, which can drift away from the LLM's input space and degrade autoregressive decoding. In this work, we propose Convex Gate (C-Gate), a speech-to-LLM bridge that constrains all speech representations to lie within the LLM's input embedding manifold with an architectural convex-hull constraint. Concretely, each frame is represented as a convex combination of token embeddings, ensuring compatibility with the pretrained LLM while preserving continuous expressivity. Across automatic speech recognition (ASR) and emotion recognition, C-Gate achieves strong joint performance, improving LibriSpeech WER by up to 48.7% relative while matching or exceeding single-task emotion accuracy. Beyond performance, our analysis reveals a key insight: information is not carried by discrete token identities, but by time-resolved trajectories in the embedding space. Causal interventions confirm that both the trajectory structure and alignment to the pretrained embedding manifold are critical for performance. These results suggest that geometry, rather than token discreteness, is the fundamental design factor in speech-to-LLM interfaces, and provide a controlled regime for studying multimodal integration in frozen LLMs. We release the checkpoint, per-sample outputs, mechanism dumps, and intervention suite for replication. 

---
# In-Context Learning for the Imputation of Public Opinion Data with Large Language Models 

**Authors**: Tobias Holtdirk, Georg Ahnert, Joseph W Sakshaug, Anna-Carolina Haensch  

**Link**: [PDF](https://arxiv.org/pdf/2606.09351)  

**Abstract**: Large language models have been widely evaluated as simulators of individual survey responses. In practice, however, fully unobserved responses are rare; the dominant problem is partial non-response. Imputation aims to restore the overall structure of a survey dataset by filling in these missing values. It has its own well-defined evaluation criteria and differs fundamentally from prediction. We propose to impute missing survey data through in-context learning (ICL). We systematically evaluate ICL design choices across different missingness mechanisms (MCAR, MAR, MNAR) on 150 opinion variables spanning 15 waves of the American Trends Panel. Compared to well-established statistical methods for data imputation like MICE PMM, our ICL approach consistently reduces absolute error across all missingness mechanisms, with the largest gains under non-random missingness (MNAR). Notably, the best-performing specification (gpt-oss-120b with 100 in-context examples) achieves near-nominal aggregate coverage (approaching the 95% level) with confidence intervals two to five times narrower than MICE PMM. We publish a Python package with an sklearn-like API to enable easy deployment of our method using local and proprietary LLMs. 

---
# Multi-Hop Knowledge Composition is Bound by Pretraining Exposure 

**Authors**: Yannis Karmim, Luis Marti, Djamé Seddah, Valentin Barrière  

**Link**: [PDF](https://arxiv.org/pdf/2606.09338)  

**Abstract**: Large Language Models fail at implicit multi-hop reasoning: a model answers "When was $X$ born?" and "Who is $Y$'s closest friend?" correctly but fails on "When was $Y$'s closest friend born?" in a single forward pass, even when both facts are perfectly memorized and individually retrievable. We study this failure in a controlled natural language setting with a strict separation between individuals exposed to compositional contexts during pretraining and those that never appear in any such context. We confirm that compositional failure persists even at 97% 1-hop accuracy, establishing the gap as a pretraining failure rather than a knowledge absence. We propose and test nine data-centric augmentation formats and find that compositional pretraining transfers to unseen questions for exposed individuals, but never to individuals absent from compositional pretraining, suggesting that exposure to compositional contexts during pretraining is a necessary condition for implicit multi-hop reasoning. 

---
# How Far Can Prompting Go for Minimal-Edit Ukrainian Grammatical Error Correction? 

**Authors**: Kateryna Karpo, Artem Chernodub  

**Link**: [PDF](https://arxiv.org/pdf/2606.09334)  

**Abstract**: Fine-tuned Large Language Models (LLMs) dominate in Ukrainian grammatical error correction (GEC), while API-accessed LLMs remain nearly untested on minimal-edit benchmarks. We evaluate 11 commercial LLMs from four providers and one open-source Ukrainian model on the UNLP 2023 GEC-only benchmark, comparing zero-shot, few-shot, minimal-edits, and LLM-assisted prompt optimization strategies. Our best configuration (Gemini 3.1-Pro) reaches F0.5=69.22, closing over 90% of the gap to fine-tuned SOTA (F0.5=73.14). For zero-shot prompts, only Claude models benefit from Ukrainian instructions. However, the best overall results for all models use Ukrainian minimal-edits prompts, whose language-specific rules require Ukrainian to express precisely. LLM-assisted prompt optimization on top of minimal-edits + few-shot achieves the highest score. Detailed minimal-edits instructions yield the largest gains for punctuation and case errors but cause the model to abandon several low-frequency categories. Delving into error analysis, we identify five recurring overcorrection patterns tied to Ukrainian-specific linguistic phenomena. Code, prompts, and outputs are publicly available. 

---
# SG-OPD: Sign-Gated On-Policy Distillation via Sign-Consistency Gating and Phased Teacher Sampling 

**Authors**: Haoran Xu, Hongyu Wang, Yifei Gao, Jiaze Li, Xiaofeng Zhang, Xiaosong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.09304)  

**Abstract**: On-policy distillation (OPD) trains a student on its own trajectories with dense per-token supervision from a stronger teacher, and often outperforms off-policy distillation and standard reinforcement learning. However, we find that its effectiveness implicitly relies on two assumptions that frequently break in practice: trajectory-level alignment between the student and the teacher, and uniform token-level reliability of the teacher's preferences. We therefore propose Sign-Gated On-Policy Distillation (SG-OPD), which uses a binary verifier as a trust signal for the teacher at two complementary granularities: phased teacher sampling mixes in verifier-endorsed teacher rollouts at cold-start, and a sign-consistency gate extrapolates the distillation update on tokens where the teacher agrees with the verifier-correct direction and interpolates it where it disagrees. Experiments on competition-level mathematical reasoning benchmarks show that SG-OPD consistently outperforms standard OPD, with average gains of 1.98 and 7.50 at the per-sample and per-question levels, respectively. 

---
# NüshuVoice: Reviving the Voice of Endangered Nüshu with Pitch-Aware Text-to-Speech 

**Authors**: Hongkun Yang, Xinhui Yi, Xiyan Zhao, Yibo Meng, Lionel Z. Wang, Lixu Wang, Yaqi Zhang, Ruiqi Chen, Xuanyue Zhao, Lanxin Zhang, Yu Zeng, Weijia Chu, Yiming Ma, Chenyu Liu, Jianghao Lin, Xin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09295)  

**Abstract**: Nüshu is an endangered phonetic script historically used by women in Jiangyong County, southern Hunan, China. While existing computational studies of Nüshu mainly focus on textual digitization and visual recognition, the acoustic reconstruction of its authentic pronunciation remains largely unexplored. Building a Nüshu text-to-speech (TTS) system is particularly challenging because available recordings are extremely limited and mostly consist of isolated syllable-level pronunciations rather than natural sentence-level utterances. In this work, we introduce NüshuVoice, the first TTS benchmark for Nüshu. We construct a sentence-level Nüshu text-to-audio dataset that aligns standardized Unicode Nüshu text, phonetic transcriptions, standard Chinese translations, and archival recordings. To synthesize speech under this extreme low-resource setting, we propose Nüshu-PitchVITS, an F0-conditioned VITS framework that leverages Nüshu's five-level pitch notation as an explicit prosodic inductive bias. Experimental results show that Nüshu-PitchVITS outperforms strong TTS baselines in spectral fidelity, pitch reconstruction, and human-rated intelligibility. We publicly release the dataset and code at: this https URL. 

---
# One Model, Multiple Goals: Adaptive Multi-Objective Learning for E-commerce Dialogue Systems 

**Authors**: Mingzhe Li, Jing Xiang, Enguo Zhou, Lang Gao, Tai Li, Qishen Zhang, Xiangliang Zhang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.09293)  

**Abstract**: Dialogue systems in e-commerce scenarios often need to satisfy multiple objectives: accurately reasoning over user profiles (e.g., eligibility, credit limit) to ensure correct decision-making and user state interpretation, while also generating natural and faithful responses. These goals are complementary but not identical. In this work, we propose MORE, an adaptive Multi-Objective REinforcement learning framework that jointly optimizes reasoning accuracy and linguistic naturalness. Our preliminary experiments show that directly mixing rewards with diverging optimization dynamics can cause oscillations and unstable learning. Thus, instead of optimizing a single mixed reward, we treat reasoning functions as constraints that guide policy optimization. At inference time, the system directly generates responses without explicit reasoning steps, while still benefiting from reasoning-enhanced scaffold and avoiding additional inference overhead. To better balance linguistic objectives during response generation, we introduce an adaptive multi-reward mechanism that aggregates signals such as fluency and naturalness and dynamically reweighs them via gradient feedback. We evaluate MORE on two real-world dialogue systems at ByteDance and the MultiWOZ 2.2 benchmark, where it consistently outperforms strong baselines. In 14-day online experiments on ByteDance production traffic, MORE improves overall and reached conversion by 16.53% and 30.09%, while increasing user satisfaction and reducing handoff rates. Notably, in a human-machine comparison, MORE recovers about 60% of the incremental conversion lift achieved by human agents. 

---
# TruthSplit: Operationalizing Conditional Validity in Arguments Through Multi-Perspective Reasoning 

**Authors**: Benjamin Stieger, Maximilian Terberger, Thomas Huber, Christina Niklaus  

**Link**: [PDF](https://arxiv.org/pdf/2606.09251)  

**Abstract**: We present TruthSplit, an interactive system for multi-perspective argument analysis. Existing argumentation tools typically analyze properties of the argument itself, such as structure, quality, stance, or persuasiveness, while leaving perspective-specific background knowledge implicit. TruthSplit addresses this gap by supporting an exploratory analysis of how the same claim can lead to different conclusions when interpreted through worldview-specific values, assumptions, and conceptual definitions. We refer to this perspective-dependent analysis as conditional validity. Given an input argumentative text, TruthSplit extracts claims and premises, applies a three-layer natural language inference (NLI) approach to assess both logical and worldview-specific normative consistency, and conditions large language model (LLM) reasoning on structured worldview profiles that encode core values and decision principles. The system then generates perspective-specific interpretations, identifies value conflicts and assumption gaps, and visualizes divergence through interactive analytical interfaces. 

---
# Symbolic and Abstractive Reasoning with Complex Visual Queries 

**Authors**: Yichi Zhang, Jingdian Lu, Zhuo Chen, Lingbing Guo, Jun Xu, Wen Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.09195)  

**Abstract**: Understanding and reasoning over abstract visual content remains a challenge for current multi-modal large language models (MLLMs). In this paper, we explore a novel abstract data type termed complex visual query (CVQ), designed to probe symbolic and abstractive reasoning, which is a critical yet underexplored dimension of human-like neuro-symbolic reasoning for MLLMs. We present a comprehensive investigation from three perspectives: \textbf{Data $\times$ Paradigm $\times$ Exploration}. Specifically, we propose a scalable pipeline for synthesizing CVQs grounded in large-scale multi-modal knowledge graphs, generating a diverse dataset encompassing 14 distinct query types via systematic combinations of first-order logic operators. We further introduce a two-stage training framework that progressively equips MLLMs with robust visual reasoning capabilities. We conduct extensive experiments to rigorously evaluate MLLMs across multiple dimensions, including reasoning performance on CVQs, as well as cross-task and cross-scenario generalization. We believe our work opens new perspectives and avenues for advancing the reasoning frontiers of MLLMs. 

---
# Culturally-Adapted Red-Teaming Across East and Southeast Asian Contexts: A Methodological and Comparative Analysis 

**Authors**: Hyeji Choi, Yongtaek Lim, Minwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.09178)  

**Abstract**: Multilingual safety evaluation of large language models (LLMs) has predominantly relied on direct translation (DT) of English benchmarks into target languages - an approach that converts surface-level linguistic form while failing to reflect the cultural context embedded in threat scenarios, social norms, and legal frameworks. We construct paired DT and culturally-adapted (CA) datasets via 1:1 seed matching for four languages - Korean (KO), Japanese (JA), Thai (TH), and Khmer (KM) - and compare Attack Success Rate (ASR) and Cultural Realism scores across four open-source LLM. CA prompts yield Delta-ASR > 0 across all 16 language x model combinations (mean +9.3 pp), and DT-based evaluation underestimates risk in 44 of 48 category x language combinations. Language-level analysis reveals that the distribution of threat forms is heterogeneous across languages. Cultural Realism analysis further shows that DT Cultural Depth (C3) scores remain consistently below 1.0 out of 3.0 across all four languages (mean 0.17), whereas CA scores reach up to 2.51, indicating that direct translation produces inputs systematically divergent from those encountered in real-world multicultural settings. These findings demonstrate that adapting benchmarks to language-specific cultural contexts - rather than relying on linguistic translation alone - is necessary for valid multilingual LLM safety evaluation. 

---
# Unified Energy for Invariant and Independent Decoding in Diffusion Language Models 

**Authors**: Yuchen Yan, Minkai Xu, Zaiquan Yang, Yatao Bian  

**Link**: [PDF](https://arxiv.org/pdf/2606.09159)  

**Abstract**: Diffusion Language Models (DLMs) enable parallel text generation by iteratively denoising a full sequence, offering attractive flexibility compared to auto-regressive (AR) decoding. However, existing methods fail to fully capture token relationships, leading to a performance gap relative to AR baselines, especially as the degree of parallelism increases. In this paper, we give a systematic analysis of the gap, identifying three key factors: (i) model capacity, (ii) dependency, and (iii) invariance. To address these issues, we first propose an invariant energy (Inv-E) together with an effective sampling-based estimator to handle the invariance issue. By further combining with the independent energy (Ind-E), we obtain a unified energy (Uni-E), that accounts for all these factors. Uni-E enjoys a unique advantage: it can be computed exactly without sampling-based partition estimation. Besides, Uni-E is model agnostic and can therefore be scaled to models of arbitrary size. We further prove that Uni-E can correct the distribution shift caused by dependency and invariance. Extensive experiments across Diffusion Language Models (DLMs) and Diffusion Large Language Models (DLLMs) demonstrate the effectiveness of the proposed Uni-E. 

---
# SEF-CLGC at SemEval-2026 Task 11: Logical Notation Impact on Language Model Performance 

**Authors**: Hanna Abi Akl, Fabien Gandon, Catherine Faron, Pierre Monnin  

**Link**: [PDF](https://arxiv.org/pdf/2606.09157)  

**Abstract**: This paper revisits our pipeline called Syllogistic Evaluation Framework-Common Logic Grammar Construction (SEF-CLGC). We combine formal logical notations with Small Language Models (SLMs) to evaluate reasoning performance on the SemEval-2026 Task 11 Subtask 1: Disentangling Content and Formal Reasoning in Large Language Models. Our experiments show that by relying solely on SLMs, trained on a combination of natural and symbolic languages, our best model achieves a content score of 27.80% on the task while significantly lowering the content bias in reasoning. 

---
# Explicit Representation Alignment for Multimodal Sentiment Analysis 

**Authors**: Baode Wang, Ziming Wang, Huacan Wang, Ronghao Chen, Biao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09148)  

**Abstract**: Multimodal affective analysis aims to understand human sentiment and emotion by jointly modeling heterogeneous modalities such as text and images. However, multimodal models often fail to consistently outperform strong text-only baselines, with performance varying significantly across fusion strategies. In this work, we identify representation misalignment between independently pretrained modality encoders as a key bottleneck for effective multimodal learning, and show through controlled experiments that alignment prior to fusion is often more important than fusion complexity. To address this issue, we propose a unified multimodal affective analysis framework that leverages vision-language models (VLMs) to convert visual content into structured textual descriptions, projecting heterogeneous modalities into a shared linguistic space and enabling interpretable text-centric reasoning. To further improve robustness, we introduce a hybrid learning strategy that combines semantic token selection with a batch-level uniformity regularization objective, encouraging a more dispersed and stable global feature space while mitigating noise introduced by VLM-generated descriptions. Experiments on multiple multimodal sentiment and emotion benchmarks show that our method consistently outperforms strong unimodal and multimodal baselines, achieving state-of-the-art performance. Our analysis further highlights the critical role of representation alignment in multimodal affective learning. 

---
# MAAM: Anchor-Preserving Compression and Contextual Calibration for Chinese Discriminatory Language Detection 

**Authors**: Yuxin Fu, Shijing Si  

**Link**: [PDF](https://arxiv.org/pdf/2606.09114)  

**Abstract**: Chinese discriminatory-language detection is challenging because harmful intent is often implicit and context-dependent. We propose MAAM (Myopia--Astigmatism Anchor Mechanism), a lightweight, model-agnostic framework inspired by functional visual blur: rather than preserving every token equally, MAAM retains discrimination-relevant semantic anchors and calibrates them with C--I--S contextual priors (Contextual Tone, Group Identity, and Stance Polarity). We also introduce ChLGBT, to our knowledge the first Chinese LGBT-focused discriminatory-language dataset, with 8,120 manually annotated samples and three ordinal labels: explicit bias, implicit bias, and emotional intensity. Across strong encoder baselines, MAAM improves all three prediction dimensions, with consistent gains in accuracy, F1, Brier score, and expected calibration error. Compared with frontier LLM baselines under zero-shot and few-shot prompting protocols, MAAM remains competitive while offering stronger compactness and stability. These results suggest that interpretable anchor preservation and contextual calibration provide a practical alternative to heavier model scaling for Chinese discriminatory-language assessment. 

---
# Emergent Misalignment Can Be Induced by Sycophancy and Reversed via Alignment Gating 

**Authors**: Sicheng Wang, Xiangyang Zhu, Han Wang, Zongrui Wang, Yuan Tian, Kaiwei Zhang, Kaiyuan Ji, Qi Jia, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2606.09068)  

**Abstract**: Prior work has shown that fine-tuning large language models on malicious or incorrect outputs in narrow domains can induce broad misalignment and harmful behavior, a phenomenon known as emergent misalignment. However, efficient methods for reversing such misalignment remain limited. In this work, we make two contributions. First, we identify sycophancy fine-tuning, i.e., training models to passively agree with users' incorrect opinions, as a previously underexplored driver of emergent misalignment, and show that it induces broad and severe misaligned behavior. Second, we propose Alignment Gating, an efficient method for reversing emergent misalignment that inserts learnable and controllable gates into the model during fine-tuning. Through fine-tuning, these gates learn to identify the internal representations responsible for unsafe responses. Thus, amplifying or suppressing these representations then exacerbates or mitigates EM, respectively. We further find that alignment gating module exhibits strong generalization: gating weights obtained from narrow-domain fine-tuning substantially suppress broad-domain misaligned behavior while preserving the model's general capabilities. 

---
# Bridging the Agent-World Gap: Text World Models for LLM-based Agents 

**Authors**: Yixia Li, Hongru Wang, Peng Lai, Zhiwen Ruan, He Zhu, Youxin Zhu, Ganlong Zhao, Minda Hu, Yun Chen, Sibei Yang, Peng Li, Jeff Z. Pan, Jia Pan, Guanhua Chen, Yang Liu, Guanbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.09032)  

**Abstract**: Large language model (LLM)-based agents are increasingly used in interactive textual environments, from web navigation and code editing to tool use and long-horizon dialogue. Yet many remain largely reactive, mapping observations to actions without an explicit model of how these environments are structured and evolve. This motivates text world models (TWMs): transition models over textual states that, given a state and a candidate action, predict the resulting webpage, terminal output, API response, or user reply, thereby supporting planning, efficient learning, and principled evaluation. We systematically review text world models for LLM-based agents, organized around a formal framework and the agent lifecycle: (1) Foundations, defining text world models and characterizing them by state representation and grounding domain; (2) Construction, taxonomizing LLM-as-WM and code-as-WM paradigms and reviewing methods for building them; (3) Application, examining how world models support agents at training time through experience synthesis and at inference time through planning, verification, and adaptation; and (4) Evaluation, covering both evaluation of the world model itself and its use as an evaluation environment for agents. We aim to consolidate this rapidly developing area, clarify its design space, and highlight open challenges for future research. 

---
# SafeRun: Enabling Determinism in LLM Planning for Running 

**Authors**: Meilin Chen, Zepeng Zhai, Jiaxuan Zhao, Yuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09027)  

**Abstract**: Large Language Models enable flexible natural-language planning but remain unreliable in determinism-critical domains due to their probabilistic nature. This limitation is especially problematic in running planning, where violating safety rules can lead to safety risks. We propose SafeRun, a framework for deterministic LLM-based planning via a decoupled architecture. SafeRun separates soft interpretation by an LLM from hard constraint enforcement by a deterministic solver, ensuring strict safety constraints while preserving natural-language flexibility. To validate SafeRun, we build a comprehensive benchmark for running planning under realistic physiological and safety constraints. Experiments across five LLMs show that SafeRun achieves 100\% safety score (vs.\ 79.1\% PE average and 97.6\% CodeAct average) while maintaining competitive instruction-following scores. The SafeRun benchmark is publicly available at \href{this https URL}{huggingface}. 

---
# Beyond Averages: Evaluating LLMs on Human Survey Replication at the Distributional Level 

**Authors**: Jeonghyeon Moon, Jiwon Kim, Yeheum Lah, Yoonju Han, Yuncheol Kang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09013)  

**Abstract**: LLMs are increasingly used to simulate human survey responses, but prior work has mainly evaluated replication using mean-level or aggregate agreement, offering limited insight into whether LLMs reproduce the variability of human behavior. We evaluate LLM-based survey replication at the distributional level using a non-public 2010 consumer choice experiment on Korean instant noodle purchases, a setting unlikely to overlap with model training data. We evaluate three response variables of differing statistical type: binary purchase incidence, categorical brand choice, and count purchase quantity. For each, we compare human and LLM responses at mean-level, pattern, and distributional alignment, and against reference baselines from the human data alone. LLMs reproduce condition-level patterns reasonably well but fail to capture distributional structure: for purchase quantity, no model beats a condition-insensitive baseline that simply matches the pooled human distribution. Because models that match human means well can still produce distributions further from humans than this baseline, mean-based evaluation alone can be actively misleading. Replication also varies with input configuration, with structured personas and multimodal inputs improving alignment while explicit reasoning prompting degrades it monotonically. 

---
# Language-Aware Token Boosting: LLM Language Confusion Reduction Without Tuning 

**Authors**: Trapoom Ukarapol, Pakhapoom Sarapat, Nut Chukamphaeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.08994)  

**Abstract**: Large language models (LLMs) sometimes exhibit language confusion when generating non-English text. Existing approaches typically rely on fine-tuning to mitigate this issue. In contrast, we propose a tuning-free paradigm for reducing language confusion. Within this paradigm, we introduce two methods: Language-Aware Token Boosting (LATB), which applies targeted perturbations to tokens associated with the desired language, and Adaptive Language-Aware Token Boosting (Adaptive-LATB), which dynamically adjusts these perturbations based on the model's confidence in the intended language. Experiments demonstrate that our methods effectively improve multilingual alignment by reducing language confusion, while maintain the summarization quality without requiring any additional fine-tuning. Our code is publicly available. this https URL. 

---
# Structure-Aware Modeling of Multiple-Choice Questions Improves Automatic Difficulty Estimation 

**Authors**: Gabriel Ortega, Abelino Jiménez, Séverin Lions, Pablo Dartnell  

**Link**: [PDF](https://arxiv.org/pdf/2606.08988)  

**Abstract**: Automatic Question Difficulty Estimation (AQDE) holds growing promise for educational assessment because it has the potential to yield difficulty estimates that are competitive with expert judgment, while helping reduce the time and financial burden associated with pilot administrations and scaling to digital testing contexts. Prior AQDE studies report mixed evidence on whether adding distractors as additional text to the question stem and the correct key consistently improves difficulty prediction. We hypothesize that the effectiveness of distractor information depends on its structural representation, and that explicitly modeling distractors as separate components improves difficulty estimation over baselines that omit this information. To address this, we designed controlled architectures that model MCQ components as distinct inputs to isolate the contribution of distractor content and order. Specifically, we represented distractors by encoding each distractor as its own text input and aggregating their representations either with order-aware concatenation (with positional tags) or with an order-invariant summation. We evaluated these architectures using two Chilean datasets (Natural and Social Sciences, 2016-2020; 4,114 multiple-choice questions). Compared to a simpler model that only used the question stem and the key, our best distractor-aware architecture achieved higher predictive performance, reaching R^2 = 0.83 for Natural Sciences and R^2 = 0.71 for Social Sciences items. An order-invariant variant achieved nearly the same accuracy with approximately half as many parameters, offering a favorable accuracy-efficiency trade-off. These results show that structural information (especially distractor content) drives gains in predictive accuracy, supporting the development of efficient, structure-aware models that are computationally viable for large-scale educational applications. 

---
# CARE: A Conformal Safety Layer for Medical Summarization 

**Authors**: Suhana Bedi, Bridget Lin, Anson Y. Zhou, Chloe O. Stanwyck, Jenelle A. Jindal, Sanmi Koyejo, David Stutz, Nigam H. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.08969)  

**Abstract**: Large language models (LLMs) are increasingly used for medical summarization, but their outputs can omit medically important information and introduce unsupported claims. Existing error-detection methods produce heuristic or uncalibrated scores, providing no formal control over missed errors and no principled way to trade off safety against clinician review burden. We introduce Conformal Assessment for Risk Evaluation (CARE), a post-hoc, model-agnostic safety layer that uses conformal risk control to overlay calibrated omission and hallucination flags onto summaries from any LLM without retraining. CARE provides finite-sample, distribution-free guarantees through two controllers: a hallucination controller that bounds the probability of a document containing any unflagged hallucinated sentence, and an omission controller that bounds the expected fraction of important omissions not surfaced for review. Unlike hallucination detection, omissions depend jointly on whether a source sentence is important and whether it is covered by the summary. We show that calibrating only one dimension can violate the target risk bound, while marginal decompositions remain valid but overly conservative. By jointly calibrating over the full $(\tau,\gamma)$ threshold space, CARE preserves formal guarantees while surfacing up to 5$\times$ fewer sentences than alternative calibrated baselines. Across five medical summarization tasks, CARE satisfies the target risk bound at $\alpha = 0.15$ with 95% confidence across 100 calibration/test resplits, using only ~100 labeled documents per domain. In a preliminary clinician study (75 document reviews), calibrated flags improved omission detection by 28.6 percentage points on average. These results show that sentence-level safety guarantees are feasible for LLM-assisted medical summarization and offer a tunable mechanism for balancing residual risk and review effort. 

---
# Multilingual Sentiment Aware Text Summarization A Reinforcement Learning Approach for Consistency Maintenance 

**Authors**: Mikhail Krasitskii, Alexander Gelbukh, Olga Kolesnikova, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2606.08940)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has significantly improved the quality and fluency of large language models in text summarization. However, its impact on affective properties remains insufficiently understood. In this work, we study sentiment drift, a systematic shift toward neutral sentiment in RLHF-based summarization outputs compared to source texts. We conduct extensive experiments across multiple datasets, model architectures, and eight languages to analyze how alignment objectives influence sentiment preservation. Our results show that sentiment drift is a consistent phenomenon that becomes stronger with increased KL regularization strength, indicating a trade-off between alignment stability and affective fidelity. To explain this behavior, we introduce a Policy Attribution framework that decomposes the RLHF objective and quantifies the contribution of its components. Our analysis reveals that KL regularization is the primary driver of sentiment suppression across all settings. Based on these findings, we propose a sentiment-aware modification of the KL regularization term, which selectively reduces constraints on sentiment-bearing tokens. Empirical results demonstrate that this approach mitigates sentiment drift while maintaining summarization quality. Overall, our findings highlight a fundamental limitation of current alignment methods: while they improve factual consistency and safety, they may unintentionally suppress emotional expressiveness. This motivates the development of alignment strategies that explicitly account for affective preservation. 

---
# PACT: Learning Diverse Diagnostic Strategies via Privileged Synthesis and Branch Consensus 

**Authors**: Gen Li, Yuanze Hu, Zhichao Yang, Qingchen Yu, Jianwei Lv, Yue Guo, Yujing Liu, Faguo Wu, Hongwei Zheng, Xiandong Li, Bo Yuan, Yifan Sun, Zhaoxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08938)  

**Abstract**: Clinical diagnosis requires flexible use of multiple reasoning paradigms under incomplete patient information. Existing LLM-based medical agents show strong medical reasoning ability, but single-paradigm or naively mixed dialogue supervision makes these paradigms difficult to learn without interference. We propose \textbf{PACT} (Periodic Anchor Consensus Training), a framework that couples supervised multi-paradigm dialogue synthesis with consensus-based Branch training. At the data level, \textbf{DPS} (Doctor-Patient-Supervisor) uses complete electronic medical records (EMRs) for quality control while keeping the doctor agent restricted to patient-visible information. This produces validated dialogues under four diagnostic reasoning paradigms without leaking hidden clinical answers. At the training level, PACT trains one paradigm-specific LoRA Branch per paradigm and periodically aggregates Branches into a shared Anchor through sign consensus. We further construct a dynamic multi-turn Chinese medical diagnosis benchmark for interactive consultation. Experiments show that PACT achieves state-of-the-art performance among compared proprietary, medical-specialized, and task-adapted baselines on diagnostic outcome and consultation-process metrics. 

---
# From Statute to Control Flow: Span-Grounded Deontic Trees for Defeasible Scope Parsing 

**Authors**: Jian Chen, Siyuan Li, Chucheng Wan, Zixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08932)  

**Abstract**: Rule-following agents tasked with executing policies and regulations often fail via Silent Scope Omission (SSO): a model applies a general rule but silently drops nested exceptions or counter-exceptions, producing outputs that appear compliant yet break on important edge cases. Although such failures are often framed as an agentic-systems problem, the underlying bottleneck is statutory and policy understanding, a capability typically studied in legal NLP. However, most existing legal NLP benchmarks emphasize end-task outcomes, which can overlook the structural omissions that cause SSO. To diagnose and mitigate SSO, we introduce NormBench, a benchmark of 2,290 provisions spanning Chinese (laws and local policies), English (U.S. tax law, GDPR, and corporate policies), and cross-lingual settings, designed for defeasible scope parsing: identifying precisely which clause overrides which. NormBench uses Span-Grounded Deontic Trees (SG-DT), a compiler-style intermediate representation that anchors every logical branch to source spans and requires explicit exclusion guards, enabling deterministic compilation and audit. Evaluations of frontier LLMs reveal two recurring pathologies: (1) Recursion Decay, where performance drops sharply as defeater depth increases, and (2) an Auditability Trap, where models retrieve relevant spans but fail to assemble correct control flow. Using SG-DT as a constrained intermediate output improves whole-tree fidelity and defeater recovery, and downstream experiments show that its utility is mechanism-specific: gains concentrate on exception-active, SSO-prone cases, while aggregate accuracy can be mixed when the added structure is unnecessary or parser fidelity is low. 

---
# PerspectiveGap: A Benchmark for Multi-Agent Orchestration Prompting 

**Authors**: Youran Sun, Xingyu Ren, Kejia Zhang, Xinpeng Liu, Jiaxuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.08878)  

**Abstract**: Real-world LLM applications are moving beyond single-agent workflows toward orchestrated multi-agent systems, yet current models still struggle to determine what each sub-agent needs to know. To measure this, we introduce PerspectiveGap, a benchmark for evaluating LLMs' ability to compose orchestration prompts for multi-agent systems. PerspectiveGap contains 110 scenarios, each evaluated through two distractor-mixed task formats: role-fragment assignment and free-form prompt writing. These scenarios are organized into 10 topologies, which are distilled from the authors' real-world engineering practice and framed by the Prompt Economy principle: building loop-centered orchestrations that maximize utility with minimal role and engineering overhead. In experiments with 27 commercial models from 10 companies, GPT-5.5 substantially outperforms all competitors, whereas Opus 4.7 shows a notable weakness in orchestration prompting despite its strong coding performance. Nevertheless, PerspectiveGap remains challenging: the evaluated models achieve an average combined pass rate of only 14.9\% (GPT-5.5 62.0\%) and an average overall leakage rate of 246.5\% (a per-scenario information leak-event count, not a proportion; GPT-5.5 49.1\%). These findings suggest that multi-agent orchestration prompting is a distinct and under-evaluated capability, and PerspectiveGap provides a foundation for measuring and improving it systematically. 

---
# Building Customer Support AI Agents at 100M-User Scale: An Evaluation-Driven Framework 

**Authors**: Aman Gupta, Kevin Rossell, Edesio Alcobaça, Jose Chrystian Lima Pacheco, Carolina Baptista de Lima, Shao Tang, Luiz Paulo Rabachini, Luis Moneda, Herbert Fei, Daniel Silva, Rohan Ramanath  

**Link**: [PDF](https://arxiv.org/pdf/2606.08867)  

**Abstract**: The rapid rise in LLM capabilities has made AI agents increasingly viable across a broad range of tasks. Among the most promising applications is building production-ready customer-facing agents, a challenge that demands coordinated excellence in evaluation methodology, context engineering, training, and online measurement. Yet these critical pillars are typically developed in isolation, creating blind spots that only surface after deployment.
In this paper, we present a unified framework that bridges offline development with online impact for customer support AI agents at Nubank, a company with 100M+ users. Our approach integrates several key components: (1) structured context engineering tailored to customer support agents, (2) systematic human-in-the-loop prompt iteration, (3) rigorous LLM judge evaluation with measured inter-rater agreement and GEPA optimization for consistency, and (4) ideation-to-production validation.
A central insight is that evaluation-pipeline quality directly determines iteration velocity. We present results from five production deployments spanning distinct domains: card delivery, debt management, credit-limit support, card management, and product explanation. These deployments deliver consistent customer-satisfaction gains while substantially accelerating iteration. In our card-delivery deployment, large-scale A/B testing yields a 37 percentage-point improvement in AI transactional Net Promoter Score and a 29 percentage-point gain in self-service rate over prior agent variants, alongside a strong correlation between offline simulation metrics and online outcomes, demonstrating that eval-driven development reliably predicts production impact. On most use cases, AI satisfaction reaches within a few percentage points of expert human agents. 

---
# PaperMentor: A Human-Centered Multi-Agent Writing Tutor for AI Research Papers on Overleaf 

**Authors**: Jiarui Liu, Terry Jingchen Zhang, Ryan Faulkner, X. Angelo Huang, Vilém Zouhar, Dominik Glandorf, Isabel Dahlgren, Van Q. Truong, Rishit Dagli, Yuen Chen, Felix Leeb, Punya Syon Pandey, Yves Bicker, Suvajit Majumder, Wenyuan Jiang, Zeju Qiu, Sankalan Pal Chowdhury, Bernhard Schölkopf, Mona Diab, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2606.08857)  

**Abstract**: Expert writing feedback from experienced researchers is critical for early-career scholars to improve their manuscripts, yet high-quality feedback often remains scarce because reviewing research papers is labor-intensive. Emerging AI-powered writing assistants largely focus on grammar fixes or simulating peer review with final scores, yet they fall short of providing concrete, actionable suggestions that help students improve their papers during drafting. We present PaperMentor, a human-centered writing assistant system that delivers actionable suggestions as Overleaf-native inline comments while leaving the actual writing entirely to human authors. PaperMentor integrates an expert skill library carefully curated from established researchers' writing advice with 12 specialized agents covering different aspects of paper writing, such as formatting compliance, phrasing accuracy, and terminology consistency. In a user study (n=14), 90.6% of the generated comments were rated actionable and 67.5% were rated valid, significantly outperforming a GPT-5.2 baseline uswithout the skill library. We release PaperMentor as open source for public use. Our code is publicly available under the AGPL-3.0 license at this https URL 

---
# Continuous Language Diffusion as a Decoder-Interface Problem 

**Authors**: Zhicheng Du, Lan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.08810)  

**Abstract**: Gaussian-corrupted sentence embeddings have no direct linguistic interpretation, yet continuous diffusion language models can generate fluent text from them. We study this puzzle through Embedded Language Flows (ELF) and identify a decoder-basin mechanism: denoising succeeds when trajectories reach regions where the native decoder can read stable tokens. We introduce a diagnostic protocol for denoisability, semantic recoverability, order sensitivity, decoder compatibility, and trajectory reliability. It exposes failures hidden by scalar metrics: low mean-squared error can discard linguistic content, low perplexity can reflect low-entropy collapse, and clean latent reconstruction can coexist with a narrow decoder basin. A decoder-margin bound explains why token recovery depends on margin and local decoder sensitivity, not latent error alone. Auditing public ELF checkpoints reveals an interface phase diagram: early predictions are weakly readable, mid-trajectory disagreement marks a competition region, and late predictions enter a high-margin final-token basin. Once inside, token realization is surprisingly simple on generated ELF states: frozen T5 token-embedding lookup recovers $93$--$96\%$ of native decoder decisions, and a single linear readout reaches $97.9\%$ agreement at 32k samples, leaving about a 1.1 perplexity gap in a structured residual tail. A conservative margin gate exits $17$--$27\%$ earlier in denoising steps under an explicit diagnostic monitor. Boundary checks on LangFlow, BitstreamDiffusion, and the Continuous Latent Diffusion Language Model (Cola-DLM) show that the same interface questions remain meaningful when the state object and decoder change. Continuous and latent diffusion language models should therefore be evaluated as representation-decoder systems. 

---
# The Amplifying Mirror: Locating and Steering the Partisan Direction inside a Large Language Model 

**Authors**: Wendy K. Tam  

**Link**: [PDF](https://arxiv.org/pdf/2606.08792)  

**Abstract**: Large language models are rapicly replacing search engines as the primary interface between people and information. Unlike search engines, which retrieve existing content, LLMs generate novel text shaped by internal representations learned during training. Here we show that partisan political identity is encoded in the model's activation space, and that this direction directly shapes generation. Using 190,491 tweets from sitting members of the U.S. Congress as labeled training data, we train linear probes on the hidden states of the Llama 3.1 8B Instruct model. We identify a single geometric axis at layer 18 that separates Republican from Democratic text with an AUC of 0.945 and a Cohen's d of 1.94, and use sparse autoencoders to decompose that axis into interpretable partisan features. Causally intervening along this axis, ablating or amplifying the partisan component mid-generation, produces systematic shifts in the model's output. We witness stance reversals, register shifting, and structured fabrications of authority. Our results demonstrate that partisan bias in language models is not a vague emergent property but a learned geometric feature that can be precisely located and steered. Partisan bias is not a bug to be patched, but a structural property of how these models encode information about their users. As LLMs displace search engines as the interface to knowledge, understanding that product design (and its consequences) will be essential for navigating the legal, social, and political transitions from an information ecosystem that is curated to one that is generated. 

---
# TeamHerald@CHIPSAL 2026: Hate Speech Detection and Sentiment Analysis of Nepali Memes using Transformer-based Architectures and Ensemble Learning 

**Authors**: Ashish Acharya, Anish Khatiwada, Rohit Khadka, Pragya Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2606.08770)  

**Abstract**: The analysis of internet memes in the Nepali language is complicated by frequent code-mixing and a lack of established baseline resources. While memes inherently combine visual and textual elements, this study focuses on a text-centric approach by extracting embedded text using an OCR layer and modeling it with Transformer-based architectures. We evaluate six distinct models and investigate the comparative effectiveness of Hard and Soft Voting ensemble strategies across two tasks: binary hate speech detection and three-class sentiment analysis. Experimental results show that a standalone decoder-only model achieved the highest performance for binary classification, whereas the Soft Voting ensemble performed best for the multi-class sentiment task, yielding a 15.8% relative improvement in Macro F1-score over the strongest standalone baseline. These findings suggest that ensemble strategies behave differently across binary and multi-class tasks, highlighting the importance of selecting aggregation methods suited to the classification objective. 

---
# RadOT-Eval: Auditable Structured-Evidence Transport for Radiology Report Evaluation 

**Authors**: Weixin Liu, Juming Xiong, Yang Li, Qingyuan Song, Susannah Rose, Murat Kantarcioglu, Bradley Malin, Zhijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.08769)  

**Abstract**: Automatic evaluation is critical for high-stakes text generation, where errors often involve omitted findings, hallucinated content, polarity reversals, location changes, uncertainty mismatches, and temporal-comparison errors rather than low surface similarity alone. Radiology report generation provides a challenging test case because generated reports must preserve structured clinical evidence across sources. We present RadOT-Eval, an interpretable structured-evidence optimal transport framework for offline auditing of radiology report generation. RadOT-Eval decomposes reference and candidate reports into attribute-structured clinical evidence units, aligns corresponding evidence using entropy-regularized optimal transport, and uses clinically meaningful side-channel discrepancies in a monotone risk model to predict error burden. All transport, feature, and readout choices are selected using the ReXVal dataset, and the frozen system is evaluated on the independent RadEvalX dataset. RadOT-Eval achieves Spearman correlations of 0.715, 0.548, and 0.399 with total, clinically significant, and clinically insignificant annotated error burden, respectively, yielding higher point estimates than standard evaluation metrics and the open-source large language model (LLM)-based evaluator GREEN-radllama2-7B. In a frozen auxiliary corruption-sensitivity stress test on ReXErr-v1, RadOT-Eval achieves 0.768 AUROC and a 0.990 corrupted-greater-than-clean paired win rate. These results show that structured evidence transport provides an auditable, rank-oriented evaluation tool for high-stakes generated clinical text under ReXVal-only model selection and frozen RadEvalX testing. 

---
# Co-Evolving Skill Generation and Policy Optimization 

**Authors**: Zhiwei Zhang, Yudi Lin, Nikki Lijing Kuang, Linlin Wu, Xiaomin Li, Songtao Liu, Fenglong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.08755)  

**Abstract**: Skill-augmented reinforcement learning improves language agents by storing reusable procedural knowledge acquired from past experience. Existing methods typically use strong language models to analyze trajectories, generate skills, and update a retrievable skill bank during online training. However, they rarely assess whether a newly generated skill is useful before it is stored and reused. We find that this assumption is unreliable: even skills generated by proprietary frontier LLMs exhibit highly mixed utility, with many providing little benefit or even degrading performance. Once such skills enter the bank, their effects are difficult to identify, because subsequent rollout feedback is delayed and usually reflects the combined effect of multiple retrieved skills rather than the marginal contribution of any individual skill. We propose an online reinforcement learning framework for pre-storage skill validation. The framework estimates whether a candidate skill contributes useful information beyond the skills already retrieved for the current task. It uses the standard rollout budget to form two matched groups under the same task and retrieval context: base rollouts conditioned on the currently retrieved skills, and skill-augmented rollouts conditioned on the same skills plus one candidate skill induced from the base trajectories. The reward gap between these two groups estimates the candidate skill's context-dependent marginal utility, enabling the framework to promote useful skills while filtering ineffective or harmful ones without additional rollout overhead. The framework further uses this marginal-utility signal to train the policy itself as a skill generator, reducing reliance on repeated calls to proprietary models. The learned skill-generation likelihood serves as a context-dependent score for retrieval-time reranking and outdated-skill pruning as the policy evolves. 

---
# HydraQE: OSU's Submission for the IWSLT 2026 Speech Translation Metrics Shared Task 

**Authors**: Kevin Krahn, Eric Fosler-Lussier  

**Link**: [PDF](https://arxiv.org/pdf/2606.08748)  

**Abstract**: We present HydraQE, our contribution to the IWSLT 2026 Speech Translation Metrics shared task. HydraQE is an end-to-end, reference-free quality estimation (QE) system for speech translation built on a Qwen3-ASR backbone, which accepts source audio and a translation hypothesis as joint input. Hidden states from all backbone layers are combined via a learnable sparsemax scalar mix, then re-encoded by a lightweight bidirectional Transformer to enable full cross-modal interaction prior to pooling into a shared embedding. Three independent prediction heads are trained on complementary supervision signals: human direct assessment (DA) annotations, MetricX-24 pseudo-labels, and xCOMET pseudo-labels. To address the scarcity of human-annotated data, we train on a combination of synthetically corrupted examples and silver pseudo-labeled machine translation outputs, using a curriculum that begins on synthetic and silver data and gradually shifts toward human-annotated examples. HydraQE outperforms cascaded text-based baselines and prior direct speech QE systems, demonstrating that end-to-end speech translation QE is competitive with cascaded approaches. 

---
# Operationalizing Linguistic Methods through Prompt-Engineering Skills: An Automatic Chinese Web Neologism Detection Pipeline 

**Authors**: Yufeng Wu, Meichun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08715)  

**Abstract**: We present a method for automatic Chinese web neologism detection that operationalizes traditional linguistic identification principles as prompt-engineering skills. The method has four stages: tokenizer-independent character n-gram candidate generation; dictionary anchoring with a Pointwise Mutual Information pre-filter; a well-formedness skill based on Chinese word-formation principles; and a combined rule and three-way classification skill that distinguishes neologism, entity, and none. Applied to the BAAI CCI 3.0 corpus (267M documents), the method produces 226,959 classified candidates including 4,853 labeled neologisms. To evaluate the method, we develop a per-stage conditional recall decomposition in which the pipeline's strict recall factors mathematically into the product of stage conditional recalls. Applied to Hou (2023) (4,199 entries), the decomposition exposes Stage 1 candidate coverage and Stage 4B LLM semantic judgment as the two bottlenecks (R=41.5% and 60.0% respectively), while intermediate stages are near-lossless. A length-stratified analysis further reveals that the structural well-formedness skill is length-invariant (>= 96.9%) whereas the semantic novelty-classification skill is length-dependent (65.6%/59.0%/44.1% across 2/3/4-character candidates), mapping a current boundary of skill-based linguistic operationalization. We release the method, pipeline outputs, and evaluation protocol as public resources. 

---
# Analyzing the Correlation Between Hallucinations and Knowledge Conflicts in Large Language Models 

**Authors**: Lucrezia Laraspata, Giovanna Castellano, Gennaro Vessio  

**Link**: [PDF](https://arxiv.org/pdf/2606.08705)  

**Abstract**: Hallucinations -- factually incorrect or unverifiable outputs -- remain one of the most challenging limitations of Large Language Models (LLMs), especially in knowledge-intensive tasks. One proposed explanation is internal knowledge conflicts arising from fixed, outdated training data. This paper investigates whether internal representations linked to knowledge conflicts correlate with hallucination behaviors in LLMs.
Using probing techniques inspired by two prior works, we analyzed activations from hidden, attention, and MLP layers, as well as output logits, across predefined tasks. We probed LLaMA-3-8B on hallucination detection benchmarks and Falcon-7B on a knowledge conflict dataset. Our findings show that, although conceptually related, hallucination activation patterns cannot be fully reduced to or explained by knowledge conflict representations.
Nonetheless, probing proves a robust tool across multiple languages and activation types, supporting its role in improving LLM interpretability. This work advances the broader understanding of hallucinations in LLMs and underscores the value of fine-grained analysis of their internal behavior. 

---
# ClinicalAligner26AM: A Cross-Lingual Aligner for Dataset Translation; Evidences from the MultiClinCorpus Shared Task 

**Authors**: François Remy  

**Link**: [PDF](https://arxiv.org/pdf/2606.08673)  

**Abstract**: Word-level cross-lingual alignment is central to annotation projection, translation auditing, and cross-lingual faithfulness estimation, yet existing neural aligners are rarely adapted to specialized domains. In this paper, we introduce ClinicalAligner26AM, a large-context multilingual aligner model for biomedical and clinical text initialized from ClinicalEncoder26AM. Our training recipe is inspired by AWESoME Align. We build our soft alignment target by sharpening with Sinkhorn-Knop optimal transport a cost matrix established for parallel clinical texts and conversations through the fusion of sentence-level, phrase-level, and token-level signals. We distill this sharpened alignment matrix directly into our student aligner, by encouraging its naive cosine-based token similarity scores to match this target. At inference time, we project source-span scores through the learned token alignment matrix and decode the longest valid high-scoring span in the target text, optionally supported by MultiClinNER predictions summarized in Appendix B. We evaluate CA26AM on the MultiClinCorpus shared task, which projects Spanish clinical entity annotations into six target languages. Our two submitted systems ranked respectively first and second across all languages and entity types, with character-weighted F1 scores above 0.95 in nearly all settings. 

---
# From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory 

**Authors**: Yishuo Cai, Xingyu Guo, Xuancheng Huang, Jinhua Du, Can Huang, Wenxuan Huang, Wenhan Ma, Yuyang Hu, Aohan Zeng, Jie Tang, Xu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.08656)  

**Abstract**: Large language model (LLM) agents are increasingly deployed in long-running settings where improving through experience at test time becomes important. A common approach is to update an explicit memory after each interaction to guide future decisions. However, most existing methods rely on hand-designed prompting rules, making it difficult to align memory updates with downstream objectives over multi-step horizons consistently. We propose MemoPilot, a plug-in memory copilot that explicitly trains the memory update process to improve a frozen LLM's performance across sequential interactions. We formulate memory updating as a multi-turn decision problem and optimize it end-to-end with multi-turn GRPO. Our training recipe introduces (i) a turn-wise reward signal and (ii) a context-independent, turn-level advantage estimation across rollouts, enabling finer-grained credit assignment and more stable training in multi-turn settings. We evaluate MemoPilot on two testbeds: multi-round Rock-Paper-Scissors (RPS) and Limit Texas Hold'em (LHE). Across both environments, MemoPilot substantially improves test-time learning of a frozen player over strong baselines, ranking first in Elo ratings on both games (1762 on LHE and 1590 on RPS) and outperforming all baseline memory methods and proprietary models, including DeepSeek-V3.2. 

---
# A retrieval conditioned rebinding circuit for dynamic entity tracking in large language models 

**Authors**: Soyoung Oh, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2606.08644)  

**Abstract**: To interpret context correctly and retrieve relevant information, large language models must bind entities to their attributes and update these bindings as state changes. We analyze how LLMs implement this binding process in a dynamic state tracking. Using causal interventions, we identify a retrieval conditioned rebinding mechanism, a compact attention head circuit that encodes swap relevant binding information and reinstates it at readout. Across Gemma and Llama models, this circuit supports rebinding behavior, but the representational signature of the mechanism differs across model families. In Gemma models, the binding signature is clearly expressed in the query/key subspaces of the relevant attention heads, whereas in Llama models, the binding information is carried primarily in key vectors. Overall, our results reveal an interpretable mechanism for context dependent state tracking in LLMs. 

---
# Sycophancy Towards Researchers Drives Performative Misalignment 

**Authors**: David D. Baek, Xinnuo Li, Anay Gupta, Taslim Mahbub, Kejian Shi, Max Tegmark, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.08629)  

**Abstract**: The increasing situational awareness of language models raises safety concerns: models might be aware when they are evaluated, and adjust their behavior to evade monitoring and resist modification, e.g., pretending to be aligned only in evaluation. This alignment faking behavior is often interpreted as scheming: an intentional effort of strategic deception. In this paper, we examine an alternative interpretation, performative misalignment, which explains the change in behavior as a result of sycophancy towards AI researchers. To examine this hypothesis, we present three empirical findings. First, we show that evaluation awareness persists even when we tell models they are deployed, which contradicts the scheming story which predicts less misalignment when the model perceives evaluation. Second, we use probing and steering to show that our current methods cannot mechanistically distinguish sycophancy and scheming in alignment faking evaluations. Third, we fine-tune models to be more sycophantic and observe increased sensitivity to evaluation cues. To conclude, we emphasize deconfounding sycophancy from scheming for future work on evaluations and mitigations of intent misalignment. 

---
# From Holistic Evaluation to Structured Criteria: Rubrics Across the Evolving LLM Landscape 

**Authors**: Hao Chen, Ziyu Han, Yukun Yan, Qingfu Zhu, Maosong Sun, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2606.08625)  

**Abstract**: As Large Language Models (LLMs) advance toward open-ended autonomous agents, the mechanisms used to evaluate and guide their behavior must evolve accordingly. This work introduces the rubric as a unifying framework capturing this evolution, characterizing rubrics as a dynamic response to successive LLM paradigm shifts that recurs across otherwise independent efforts in evaluation, reinforcement learning, and safety alignment. We define rubrics as explicit criteria sets that transform complex quality judgments into structured and actionable standards, and demonstrate that their recurrence across these research threads is not coincidental. We systematically organize existing rubric designs, examine their construction and optimization, and analyze their role across evaluation and training. Rubrics manifest at three progressively deeper levels: at the evaluative level, they decompose holistic judgments into verifiable dimensions; at the training level, they serve as dense feedback signals providing process-level guidance where scalar rewards fall short; at the intrinsic level, they emerge dynamically from model behaviors, driving self-improvement. We further assess rubric reliability across generation quality, execution fidelity, theoretical constraints, and security threats, before surveying rubric-based benchmarks across diverse domains. By rendering assessment transparent and decomposable, rubrics translate human value expectations into machine-learnable signals, serving as the enduring bridge between human intentions and machine behavior. 

---
# Cross-Source Reasoning-based Correction for Author Name Disambiguation 

**Authors**: Fanjin Zhang, Yunhe Pang, Bo Chen, Zhiyu Shen, Yanghui Rao, Evgeny Kharlamov, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08617)  

**Abstract**: Author name disambiguation is a critical challenge in academic search systems, often addressed through from-scratch and real-time disambiguation approaches. However, current algorithms remain vulnerable to cumulative errors of paper-author assignments and overlook inconsistent assignments across different sources. Resorting to expert annotation is resource-intensive. To this end, this paper explores a new perspective for author name disambiguation: cross-source correction by leveraging inconsistent assignments across sources. We propose CrossND, a full-stack framework that integrates data refinement, cross-source reasoning, and test-time scaling. First, a chain-of-refinement pipeline denoises author profiles and produces more accurate paper-author matching probabilities. Second, a supervised fine-tuning process incorporates these refined signals and a probabilistic soft logic-based cross-correction module to infer the assignments of which sources are incorrect. Third, test-time scaling further enhances the accuracy and robustness of the predictions. Experiments on real-world datasets indicate that CrossND consistently outperforms 17 baselines by leveraging cross-source reasoning without human intervention. 

---
# Multilingual Fact-Checking at Scale: Fine-Tuned Compact Models vs LLMs 

**Authors**: Pratuat Amatya, Vinay Setty  

**Link**: [PDF](https://arxiv.org/pdf/2606.08605)  

**Abstract**: We present a multilingual fact-checking system deployed at Factiverse, designed for high-throughput and low-latency operation across diverse languages. The system follows a modular pipeline with three stages: claim detection, evidence retrieval and re-ranking, and veracity prediction. We fine-tune XLM-RoBERTa-Large for claim detection, mmBERT-base for three-label stance classification (Supports/Refutes/Mixed), and a SetFit-based multilingual re-ranker for claim--evidence matching. We compare these components against strong LLM baselines, including GPT-5.2, Claude Opus~4.6, and Qwen3-8b. Experiments on production data spanning 114 languages for claim detection and 28 languages for veracity prediction show that task-specific fine-tuning provides strong and stable multilingual performance, while the fine-tuned retrieval model remains competitive with modern proprietary embeddings. Same-hardware latency measurements further show large efficiency gains for encoder-based components, supporting their use in production deployments with tight cost and privacy constraints. Overall, compact fine-tuned, self-hosted models remain a practical and effective foundation for multilingual fact-checking at scale. Code and data used for this study are available at this https URL. 

---
# Detection and Interpretability Analysis of Quotation Errors by Large Language Models 

**Authors**: Bei Huang, Yingyi Zhang, Shenghao Huang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08589)  

**Abstract**: Purpose - Quotation error refers to the inconsistency between cited information and its original source. This phenomenon leads to a series of negative impacts, such as misinterpretation of the original research, undermining the academic community's collective understanding of relevant issues, and weakening the accuracy and fairness of the citation-based academic evaluation system. Existing studies have shown that quotation error is prevalent in the academic community; moreover, manual verification of quotation error is not only labor-intensive but also inefficient. Therefore, this paper proposes the task of 'automated detection of quotation errors'. Methodology - Adopting a large language model (LLM)-based approach, this paper improves detection performance from two aspects on the basis of existing research: first, employ the fine-tuning approach for LLMs to detect quotation errors; second, incorporating full-text data of the cited literature into dataset construction, and exploring the optimal scheme for building such datasets by comparing three types of full-text integration methods. Based on this, this paper further uses the TokenSHAP tool to conduct interpretability experimental analysis on the model's prediction results. Findings - The fine-tuning approach for LLMs has improved the performance in detecting quotation errors. Among the different methods for incorporating full-text information, the approach based on using the source abstract yielded the best performance. Originality - The fine-tuning approach for large language models (LLMs) is applied to the task of automated detection of quotation errors, and interpretability analysis is conducted on the model's output results. 

---
# Calibration of Structured Ignorance Certificates for Diagnosing Unknown Unknowns in Reasoning Models 

**Authors**: Subramanyam Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2606.08571)  

**Abstract**: Large language models frequently fail in a characteristic way: rather than acknowledging ignorance, they produce fluent but incorrect answers to questions that lie beyond their knowledge boundaries. We introduce \textbf{Structured Ignorance Certificates} (SICs), a JSON-formatted output schema that demands a model explicitly name the missing domain intersection, enumerate required concepts, and propose a productive retrieval query rather than hallucinating an answer. To train models to produce high-quality SICs we construct a 7,347-sample \emph{Unknown-Unknown} (UU) dataset by prompting Qwen3-14B to stitch together questions from seven domains (physics, biology, engineering, CS, economics, medical, legal) into novel cross-domain queries that no single-domain expert could answer. We fine-tune a 14B-parameter model with Group Relative Policy Optimization (GRPO) using a composite reward that combines retrieval utility, concept specificity, and output-format validity. A paraphrase-divergence probe trained on model responses confirms that SIC-tuned outputs systematically exhibit higher unknown-unknown probability scores. Evaluation on 735 held-out UU questions achieves a 99.46\% JSON validity rate, a mean Certificate Specificity Score of 0.967, and a 3.6\% ROUGE-L improvement over the base model on retrieval-grounded generation -- demonstrating that explicit epistemic structuring is a learnable and measurable capability. 

---
# Inside the LLM Word Factory 

**Authors**: Benzi Busigin, Yuval Pinter  

**Link**: [PDF](https://arxiv.org/pdf/2606.08562)  

**Abstract**: Transformer language models process input provided as subword fragments, but natural language semantics usually rely on word-level concepts. Detokenization is the process where models reconcile these two facts, aggregating subwords into word-level representations through their computation. Prior work has found that this takes place mostly in early-to-middle layers, but so far the exact mechanics of the process have not been pinned down. We venture deep into detokenization using activation patching in controlled paired experiments that isolate the contribution of different model components, localizing English detokenization in Llama2-7B to a two-stage process at Layer 1. Attention transmits a token-specific signal from nonfinal subwords, using sequential relays if necessary, while the MLP composes it with the local embedding. This two-stage structure generalizes to twelve models from eight families, but the depth over which it takes place depends on the flavor of positional encoding: RoPE-based models detokenize over 1 to 5 layers, while learned-absolute models take 5 to 10. Finally, we provide a probe for determining the success of the detokenization process based on early-layer activations alone, performing at 0.94-0.97 AUROC depending on the amount of context. 

---
# Ishigaki-IDS: An Open-Weight Verifier-Aware Model for Information Delivery Specification Drafting in Building Information Modeling 

**Authors**: Ryo Kanazawa, Koyo Hidaka, Teppei Miyamoto, Takayuki Kato, Tomoki Ando, Chenguang Wang, Dayuan Jiang, Naofumi Fujita, Shuhei Saitoh, Atomu Kondo, Koki Arakawa, Daiho Nishioka  

**Link**: [PDF](https://arxiv.org/pdf/2606.08545)  

**Abstract**: Building Information Modeling (BIM) projects require information requirements to be described as machine-checkable Information Delivery Specification (IDS) files in order to verify whether building models contain the required attributes. However, IDS authoring remains a practical bottleneck: practitioners must handle domain vocabulary, strict XML schema constraints, and external validator conformance while also checking whether the requirement itself is correctly expressed. We present Ishigaki-IDS, an open-weight LLM specialized for verifier-aware IDS draft generation. The model combines continued pretraining on BIM/IDS corpora, supervised fine-tuning on information-requirement-to-IDS pairs, and reinforcement learning with verifiable rewards from an external validator. The goal is not to replace expert review, but to move IDS authoring from low-level XML and schema repair toward validator-loadable drafts that practitioners can inspect and correct. On the 166-case expert-created Ishigaki-IDS-Bench, Ishigaki-IDS-8B achieves an IDSAuditPass score of 0.651, a validator-pass metric for generated IDS files, substantially outperforming Claude Opus 4.5, the strongest single-shot LLM baseline we evaluated, at 0.331. It also obtains an Audit-Gated FacetF1 of 0.282, which measures requirement-facet alignment among validator-passing drafts. The same recipe scales: 14B and 32B variants reach IDSAuditPass 0.753 / 0.693 and Audit-Gated FacetF1 0.392 / 0.369. In a workflow check with six BIM practitioners, Ishigaki-assisted authoring reduced aggregate work time by 54.7% under the same validation and alignment endpoint. These results suggest that verifier-aware IDS generation can reduce the practical burden of converting BIM information requirements into reviewable IDS drafts. 

---
# Back on Track: Aligning Rewards and States for Reasoning in Diffusion Large Language Models 

**Authors**: Yawen Shao, Jie Xiao, Kai Zhu, Yu Liu, Hongchen Luo, Xueyang Fu, Yang Cao, Wei Zhai, Zheng-Jun Zha  

**Link**: [PDF](https://arxiv.org/pdf/2606.08501)  

**Abstract**: Reinforcement learning (RL) holds immense promise for enhancing the reasoning capabilities of diffusion large language models (dLLMs). However, progress is fundamentally constrained by a dual misalignment between authentic generation trajectory and the gradient update process: (i) Process-reward misalignment. Sparse, terminal rewards are indiscriminately assigned to all intermediate steps of the generation process, failing to provide discriminative credit assignment. (ii) State-trajectory misalignment. Policy updates are often diverted toward artificial, out-of-trajectory states, squandering gradients on less informative samples. To address these limitations, we introduce Process Aligned Policy Optimization (PAPO), a novel framework that holistically aligns the RL update with the dLLM's generative trajectory via Step-Aware Process Rewards (SPR) that transform sparse terminal rewards into dense, step-wise credit, and Entropy-Guided Historical Re-enactment (EHR) that replays authentic trajectories at high-uncertainty steps. Extensive experiments on four benchmarks demonstrate that PAPO significantly outperforms baselines, achieving gains of up to 4.5% on GSM8K, 4.8% on MATH500, 42.2% on Countdown and 16.1% on Sudoku. 

---
# SAEExplainer: Interpreting SAE Features with Activation-Guided Preference Optimization 

**Authors**: Jingyi He, Haiyan Zhao, Ruxue Shi, Yanguang Liu, Xin Wang, Fei Sun, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2606.08496)  

**Abstract**: Although Sparse Autoencoders (SAEs) have mitigated the opacity of large language models (LLMs) by decomposing dense representations into sparse features, explaining these features still remains a central challenge. Current explanation methods, however, typically operate within an open-loop paradigm, failing to leverage mechanistic feedback for further refinement. In this paper, we propose SAEExplainer, a training framework utilizes activation scores as an objective reward signal to train the model for self-correction and iterative bootstrapping. By iteratively verifying and correcting foundational explanations through a two-round optimization process, SAEExplainer achieves continuous improvement in its explanatory capabilities. This mechanism significantly reduces explanation hallucinations and reinforces causal triggering patterns. Extensive experiments demonstrate our approach improves upon established baselines across most metrics, especially in causal triggering and discriminative activation. 

---
# TRADE: Transducer-Augmented Decoder for Speech LLM 

**Authors**: Yun Tang, Shanil Puri, Shinji Watanabe, Subhabrata Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.08486)  

**Abstract**: Speech Large Language Models (Speech LLMs) lack a principled mechanism for streaming inference: their label-synchronous generation has no acoustic-frame alignment, making real-time decoding and end-of-utterance detection difficult. We propose TRADE TRansducer-Augmented DEcoder, which augments a multimodal LLM with a transducer branch that shares the audio encoder and uses the LLM's hidden states directly as the prediction network -- coupling frame-synchronous acoustic alignment with the LLM's linguistic reasoning. Three design choices make the system accurate, streamable, and long-form capable: (1)Tightly coupled dual vocabularies -- a compact transducer vocabulary derived from the LLM vocabulary, enabling zero-cost score fusion; (2)Chunk-synchronized streaming training with gradient stopping, eliminating the train-inference mismatch at offline-equivalent memory cost; and (3)Localized Decoder Audio Attention (LDAA), a causal sliding window that caps KV-cache memory independently of utterance length. A single TRADE checkpoint supports offline and streaming decoding across a continuous range of latency operating points. TRADE achieves 6.71% average WER on the Open ASR Leaderboard, while the streaming recognition with 960ms chunk size reaches 8.40% from the same checkpoint. On long-form speech, it obtains 3.64% WER on TED-LIUM and 10.88% on Earnings-22 without external segmentation. TRADE provides sentence-end punctuation timestamps that, when combined with acoustic voice activity detection (VAD), improve end-of-utterance detection by +0.03 F_1 over acoustic VAD alone. 

---
# More Yap Less Meaning: Uncovering Self-Improvement Behavior in SLMs 

**Authors**: Marina Igitkhanian, Erik Arakelyan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08471)  

**Abstract**: Recently, language models have made rapid progress across various domains and applications. However, their capability for self-improvement, i.e., whether they are adept at recognising and correcting flaws in their own reasoning, remains dubious. In this study, we address this question by constructing a sufficiency test to rigorously examine the self-correction capabilities of small language models (SLMs). We propose a minimal three-step self-correction pipeline that collects initial SLM answers, prompts the same model to generate hints for its incorrect responses given the ground truth, and feeds the model the same question with its own feedback to refine the initial answer. We evaluate a variety of instruction-tuned and reasoning SLMs in this experimental setup on arithmetic and logical reasoning benchmarks. Our findings show that SLMs with injected hint sentences yield only a 4.4 percent gain over initial question-answering accuracy. Even though the correct answer was provided alongside the model's incorrect reasoning, the evaluated SLMs fail to understand what was missing in their reasoning and show minimal semantic difference between hints that lead to corrections and ones that do not. Furthermore, our experiments show that longer hints are positively correlated with incorrect final answers, suggesting that longer deliberation on problems can hinder the reasoning process, meaning that SLMs do not necessarily scale in performance with a larger compute budget. 

---
# Sycophancy as a Multilingual Alignment Failure: How Safety Degrades Across Languages, Topics, and Models 

**Authors**: Arya Shah, Himanshu Beniwal, Mayank Singh, Chaklam Silpasuwanchai  

**Link**: [PDF](https://arxiv.org/pdf/2606.08451)  

**Abstract**: Safety-aligned large language models often exhibit sycophancy, which is the tendency to affirm users' opinions regardless of factual accuracy. Although well-studied in English, its manifestation in other languages remains largely unexamined, leaving billions of non-English speakers potentially vulnerable to model-validated misinformation. We present the first large-scale, multi-model evaluation of cross-lingual sycophancy, benchmarking \textbf{six instruction-tuned models} across \textbf{1.1 million instances} spanning \textbf{38 languages} and \textbf{33 topic categories}. We identify a consistent resource-tier effect: sycophancy rates spike sharply in low-resource and zero-shot language settings. Critically, this degradation is topic-agnostic, as models fail uniformly across both benign and safety-critical prompts, offering no additional protection where it is most needed. We further identify tokenizer fertility as a structural driver of this alignment collapse. Collectively, our results demonstrate that prevailing alignment methodologies generalize poorly beyond high-resource languages, underscoring the urgent need for equitable multilingual safety techniques. 

---
# Segment-level Tree Search for Long Meeting Document Summarization 

**Authors**: Sangwon Ryu, Heejin Do, Jun Seo, Daehui Kim, Yunsu Kim, Gary Geunbae Lee, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2606.08445)  

**Abstract**: Meeting documents are challenging to summarize due to their length and complex conversational structure. Existing approaches typically adopt multi-stage pipelines that extract information prior to summarization; however, these approaches often suffer from cumulative error propagation without intermediate validation, a limitation further amplified by short and low-quality reference summaries. We propose segment-level summarization via Monte Carlo Tree Search (S3), a training-free framework that constructs a final summary by composing segment-level summary candidates. S3 partitions a long document into segments and generates multiple summary candidates per segment, forming nodes of a search tree. The best-scoring combination is selected via self-reward-guided tree search and refined into the final output. Despite using a 7B model, S3 achieves performance comparable to larger 72B models while producing length-appropriate summaries. 

---
# Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs Distributional Metrics 

**Authors**: Antonio Franca, Alexander Tong  

**Link**: [PDF](https://arxiv.org/pdf/2606.08417)  

**Abstract**: Diffusion and continuous flow-based language models have emerged as the leading non-autoregressive alternatives to language modeling. Progress in both paradigms is overwhelmingly tracked by generative perplexity (gen-PPL): the per-token negative log-likelihood of samples under a frozen autoregressive (AR) scorer such as gpt2-large, typically paired with an empirical-entropy guardrail to rule out low-entropy collapse. We argue that this metric is unsound. By construction, gen-PPL measures only predictability under the scoring AR, not grammaticality or semantic coherence -- and the set of predictable but still low-quality sequences is combinatorially large. To make this concrete, we construct a suite of zero-parameter, deliberately naive samplers that achieve state-of-the-art gen-PPL on LM1B and OpenWebText at non-degenerate entropy, surpassing recently published diffusion and continuous-flow models while producing text that is incoherent by construction. We recommend evaluation suites that directly quantify the distributional divergence between generated and reference text, and use such a suite to re-benchmark recent non-autoregressive models, recovering a more faithful picture of the current state of the art. 

---
# AsyncLane: Decoupling Refinement from Advancement in Diffusion Language Model Decoding 

**Authors**: Yingxuan Ren, Yuxuan Lou, Yong Liu, Pengcheng Fang, Ziming Wang, Pengfei Zhou, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2606.08411)  

**Abstract**: Block-wise semi-autoregressive decoding is the standard inference paradigm for diffusion large language models (DLMs), but it imposes a strict dependency between blocks: the next block cannot begin until the current block is fully decoded or its denoising budget is exhausted. We observe that once a block exposes a reliable delimiter boundary or stable semantic prefix, continuation generation need not wait for every residual token to be resolved. We propose AsyncLane, a training-free decoding scheduler that decouples refinement from advancement. AsyncLane forks a generate lane at observed delimiter boundaries into a refine lane and a continuation generate lane: the prefix remains editable, while the continuation advances before prefix refinement finishes. The resulting lane tree records decoding dependencies and output order, while execution proceeds over the active lane set. To make this asynchronous schedule efficient under bidirectional attention, AsyncLane combines shared-prefix lane batching, lookahead draft reuse, cascading termination, and compact cache refresh with refresh-logit reuse, preventing model-call cost from scaling directly with the number of lanes. AsyncLane is a drop-in replacement for block-wise DLM samplers and requires no retraining. Experiments on mathematical reasoning and code generation show that AsyncLane consistently improves throughput while maintaining competitive quality. Across LLaDA and Dream backbones, AsyncLane achieves the highest TPS in all evaluated benchmark-length settings; relative to the fastest competing baseline, it reaches peak speedups of 2.95x on LLaDA and 3.04x on Dream, with especially large gains under longer generation budgets. 

---
# TimpaTeks: Automatic In-place Text Sequence Modification via Diffusion Language Model Steering 

**Authors**: Ryandito Diandaru, Ikhlasul Akmal Hanif, Fadli Aulawi Al Ghiffari, Ahmed Elshabrawy, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2606.08408)  

**Abstract**: We extend activation steering to diffusion language models (DLMs) and study a novel problem that arose due to the inference mechanism of DLMs: Modifying a text in-place to manifest a different concept. We propose TimpaTeks, an automatic in-place text modification mechanism using DLMs. Experiments on IMDB movie reviews (sentiment) and a synthetic Cats and Dogs Dataset (arbitrary, more unconventional concept steering) show that TimpaTeks provides a feasible novel mechanism to steer diffusion language model outputs in-place. TimpaTeks enables in-place modification while simultaneously lowers sentence perplexity and retaining the original sentence structre without the need of instruction tuned models. TimpaTeks is also computationally cheaper than prompt-based DLM steering, as it performs denoising in-place rather than constructing an additional prompt-conditioned output sequence. 

---
# TrustMargin: Training-Free Arbitration between Parametric Memory and Retrieved Evidence in Large Language Models 

**Authors**: Jingyan Xu, Hong Shi, Yi Shan, Penghui Liu, Yunhao Bai, Ningyuan Li, Xueyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08397)  

**Abstract**: Large language models answer knowledge-intensive questions using both parametric memory and retrieved evidence, but neither source is uniformly reliable. Retrieval can fill knowledge gaps, yet distracting passages may override correct closed-book answers. We study this post-generation conflict as answer-level source arbitration: given Direct and RAG answers from the same frozen model, decide which source to trust. We propose TRUSTMARGIN, a training-free, plug-and-play arbitration layer that scores the two existing candidates with the model's own likelihoods. It combines a parametric-prior margin, which tests whether memory accepts the retrieved answer, with an evidence-binding margin, which discounts passage-only salience and measures question-specific support. TRUSTMARGIN selects between Direct and RAG without fine-tuning, external judges, or additional generation. Across 2WIKIMQA and CWQA with three LLaMA scales, TRUSTMARGIN consistently improves over Direct generation and BM25-RAG, recovers part of the Direct/RAG oracle gap, and generalizes to multiple training-free RAG pipelines. 

---
# When Correct Decisions Hide Internal Stress: Decision-State Probing in Multimodal Language Models 

**Authors**: Haoran Zhao, Soyeon Caren Han, Eduard Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2606.08394)  

**Abstract**: Multimodal language models are typically evaluated through external behavior: selecting the correct image--text match, rejecting unsupported captions, or answering visual queries correctly. However, correct behavior alone does not show that the model's internal decision state remains stable under controlled semantic stress. We study this gap through S$^3$E (Structured Semantic Stress Evaluation), a framework for analyzing behavior-internal decoupling in multimodal language models. S$^3$E uses a positive-anchored A/B forced-choice setup in which an image-supported caption is contrasted against semantic stress candidates under both original and swapped option orders, while hidden states are extracted at the pre-answer decision state. We focus on strict-correct trials, where the model consistently selects the correct caption across both orders. Rather than treating arbitrary hidden-state variation as evidence of instability, we measure whether semantic-conflict candidates induce excess decision-state displacement relative to meaning-preserving controls. Across Qwen3VL, Gemma3, and InternVL3, semantic stress consistently produces positive selected-layer excess displacement over lexical controls despite correct forced-choice behavior, while comparisons against random negatives are model-dependent. We interpret this as a scoped decision-state stress-sensitivity signal rather than evidence of downstream failure or hallucination. Our results suggest that forced-choice correctness alone is not a sufficient certificate of invariant internal decision geometry. 

---
# Auditing Proprietary Alignment in Large Language Models: A Comparative Framework Without a Ground-Truth Standard 

**Authors**: Alireza Arbabi, Florian Kerschbaum  

**Link**: [PDF](https://arxiv.org/pdf/2606.08381)  

**Abstract**: Large language models (LLMs) are increasingly released and deployed through opaque development and deployment pipelines, enabling model providers to inject intentional, provider-specific policies without officially announcing them. As a result, various models have been reported to generate responses reflecting proprietary rules and organizational interests, leading to censorship or misinformation on controversial topics. However, systematic identification of such alignment remains a fundamental challenge, complicated by the ambiguity of what ``proprietary'' entails in different contexts. In this paper, we propose a statistical framework for detecting proprietary alignment in black-box language models via comparative behavioral analysis. Our approach quantifies systematic deviations between the responses of a target model and those of a reference set of baseline models in a shared semantic space. By evaluating relative behavioral divergence rather than absolute correctness, our framework enables principled auditing under black-box access. Applied to several widely discussed but previously unquantified cases, it provides a systematic and scalable basis for external assessment of provider-specific alignment behavior in large language models. 

---
# Forward-Free Diffusion Language Models 

**Authors**: Haotian Sun, Rushi Qiang, Yuqian Zheng, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2606.08357)  

**Abstract**: Diffusion language models generate text through iterative denoising, offering a powerful alternative to autoregressive generation. However, discrete language spaces lack a natural neighborhood structure for defining effective perturbations, so some artificial corruption schemes are proposed in the forward process. Such prescribed forward processes often produce states that are mathematically convenient but misaligned with drafts and errors encountered during generation, resulting in degraded sample quality. To address this limitation, we propose FReDA, a forward-free diffusion language model that eliminates the need for a hand-designed forward process. We formulate diffusion language modeling as recursive distribution refinement, in which model-generated drafts serve as implicit intermediate states, and the learned refinement model progressively moves the draft distribution toward the target distribution. Concretely, FReDA refines drafts by proposing candidate draft sequences and either directly performing self-refinement or selecting among parallel candidates via best-of-N refinement. With this design, FReDA is neighborhood-agnostic, model-complexity-aware, and compatible with flexible refinement parameterizations. Extensive evaluations in the sub-8B regime show that FReDA-4B outperforms larger diffusion base models on reasoning and coding benchmarks, achieving absolute gains of up to 15%, while reaching a 1.5-1.8x average speedup over diffusion baselines and scaling effectively with additional refinement computation. 

---
# Bayesian-Agent: Posterior-Guided Skill Evolution for LLM Agent Harnesses 

**Authors**: Xiaojun Wu, Cehao Yang, Honghao Liu, Xueyuan Lin, Wenjie Zhang, Zhichao Shi, Xuhui Jiang, Chengjin Xu, Jia Li, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.08348)  

**Abstract**: LLM agents increasingly rely on external inference conditions: prompts, tools, memory, SOPs, skills, and harness feedback. These assets can improve task execution without changing model weights, but they are often revised by heuristic reflection or by reusing observed successes and failures as if counts alone were reliable belief. We introduce \textbf{Bayesian-Agent}, a native and cross-harness framework that treats reusable skills and SOPs as hypotheses about whether a frozen model will succeed under a particular prompt, context, and harness environment. Bayesian-Agent records verified trajectory evidence, maintains a feature-conditioned categorical posterior over each skill, and maps posterior state into inspectable actions such as patch, split, compress, retire, and explore. Model-facing prompts receive executable guardrails and failure-mode patches, while posterior summaries remain available for audit. With \texttt{deepseek-v4-flash}, incremental repair improves SOP-Bench from 80\% to 95\%, Lifelong AgentBench from 90\% to 100\%, and RealFin-Bench from 45\% to 65\%. We further evaluate Bayesian-Agent's native backend and optional GenericAgent, mini-swe-agent, and Claude Code backends. The results include positive, negative, saturated, and case-study settings, suggesting that agent skill evolution is best viewed as posterior-guided harness optimization rather than uncalibrated prompt accumulation. The source code is available at this https URL. 

---
# Tensorizing Engram: Sharing Latents Across N-Gram Embeddings is Beneficial in LLMs 

**Authors**: Wuyang Zhou, Yuxuan Gu, Giorgos Iacovides, Yuning Qiu, Qibin Zhao, Danilo Mandic  

**Link**: [PDF](https://arxiv.org/pdf/2606.08347)  

**Abstract**: Modern language models represent text using discrete token-level embeddings, which forces recurring multi-token patterns to be learned implicitly across Transformer layers. Both Over-tokenized Transformers and Engram attempt to address this limitation by explicitly incorporating multi-token (n-gram) memories. However, they rely on separate hash tables for each n-gram order, which introduces hash collisions and prevents nested n-grams from sharing the underlying latent structures. To address these issues, we propose Tensorized Engram (TN-gram), a compact memory module that represents tensorized n-gram embeddings through shared factors in the Canonical Polyadic (CP) form. TN-gram learns shared token-position factors together with order-absorption vectors to encode the embeddings of different n-gram order. Comprehensive experiments demonstrate that TN-gram matches or even outperforms Engram-style n-gram modules while requiring much fewer parameters. 

---
# CATPO: Critique-Augmented Tree Policy Optimization 

**Authors**: Ayush Singh, Umang Goyal, Ankur Dahiya  

**Link**: [PDF](https://arxiv.org/pdf/2606.08346)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a dominant paradigm for improving the reasoning capabilities of large language models (LLMs). Recent tree-based methods such as TreeRPO extend flat trajectory sampling with tree-structured rollouts to obtain dense, step-level reward signals without a separate process reward model. However, not all trees are equally informative: trees where all leaves succeed, all leaves fail, or the policy already predicts the reward distribution contribute little to gradient updates, wasting compute. We introduce CATPO (Critique-Augmented Tree Policy Optimization), which diagnoses and addresses this waste at the tree level. CATPO first scores each tree via a tree informativeness score, F(T), combining leaf-outcome diversity with policy-reward decorrelation at zero extra compute. For dead-wrong trees where all branches fail, CATPO applies critique-guided healing: it locates the shallowest failure point, generates a natural-language critique, and grafts refined continuations to recover training signal. Finally, an informativeness-weighted loss scales each tree's gradient contribution by its normalized score, concentrating parameter updates on the most informative trees while preserving overall gradient magnitude. Experiments on Qwen2.5-Math-1.5B trained with the MATH dataset show that CATPO achieves 37.5% macro accuracy across four benchmarks (AIME24, MATH-500, OlympiadBench, and MinervaMath), improving over TreeRPO by 1.9% and GRPO by 4.8%. 

---
# Chiaroscuro Attention: Spending Compute in the Dark 

**Authors**: Prateek Kumar Sikdar  

**Link**: [PDF](https://arxiv.org/pdf/2606.08327)  

**Abstract**: Standard transformers apply self-attention uniformly at every layer and token, regardless of whether the input requires dynamic cross-token interaction. We propose CHIAR-Former (Chiaroscuro Attention), a 4-layer hybrid transformer that routes each token to one of three operators - DCT spectral mixing, RBF kernel mixing, or full self-attention - based on per-token spectral entropy, a theoretically justified complexity signal. Through systematic ablation on WikiText-103, we discover routing collapse: the router consistently rejects RBF in favour of DCT and attention, revealing that spectral mixing and dynamic attention are complementary and sufficient. A purpose-designed DCT+Attention-only variant achieves Val PPL 36.54 on WikiText-103 - a 45% improvement over a full-attention baseline (PPL 66.62) at 62.5% fewer attention FLOPs. We extend evaluation to WikiText-2, IMDB sentiment classification, and synthetic ListOps operations, establishing a clear operating regime: CHIAR-Former excels on large-scale naturalistic text where token diversity supports spectral specialisation, while full attention retains an edge on small datasets and synthetic pattern-matching tasks. These findings - both the wins and the losses - together define when and why spectral routing earns its keep. 

---
# Understanding the Sociocultural Dimensions of Mental Health Discourse in Arabic-Language X Communities 

**Authors**: Amal Alqahtani, Rana Salama, Mona Diab  

**Link**: [PDF](https://arxiv.org/pdf/2606.08307)  

**Abstract**: Computational mental health research has predominantly centered on English-speaking populations, leaving Arabic-language discourse comparatively under-examined. We present an exploratory computational study of 8,147 tweets from 607 users classified by a GPT-4.1 personal-disclosure pipeline as likely lived-experience authors in three condition-specific Arabic-language X (formerly Twitter) Communities. We focus on discourse related to borderline personality disorder (BPD), bipolar disorder, and ADHD, and characterize community-associated linguistic patterns using a multi-domain cultural keyword framework. The results suggest that in this corpus, Bipolar tweets contain more religious and medical vocabulary, BPD tweets contain more relational, identity, and emotional-distress vocabulary, and ADHD tweets more often focus on practical symptoms and medication management. We treat these patterns as hypothesis-generating rather than confirmatory because the corpus is imbalanced across conditions, some subcorpora are temporally concentrated, and the keyword framework is an initial operationalization rather than a validated measurement instrument. The paper contributes a reusable LLM-assisted personal-disclosure pipeline and an exploratory cultural keyword framework for Arabic mental health discourse. 

---
# TLRD: Teaching LLMs to Reason over Tabular Data with Tri-Level Rationale Distillation 

**Authors**: Tianyuan Liang, Xuwei Tan, Lei Shi, Junsheng Zhong, Ziyu Hu, Tian Xie, Zhiqun Zuo, Xiaodong Yu, Xueru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08295)  

**Abstract**: Tabular data is a primary medium for storing real-world information, driving many industrial applications of machine learning. Traditional predictors achieve strong predictive performance but do not provide readable, case-specific explanations essential for decision-making. Large Language Models (LLMs) can naturally bridge this gap by generating predictions alongside explanations. However, dataset-specific patterns, such as feature distributions and interactions, make tabular data difficult for LLMs to understand and reason over, while label-only fine-tuning improves performance at the cost of catastrophic forgetting. To address this problem, we propose Tri-Level Rationale Distillation (TLRD), a framework that converts label-only tabular datasets into structured rationale supervision for LLMs. TLRD uses a high-capacity teacher to synthesize a rationale corpus grounded in three complementary levels of evidence: instance-level feature, dataset-level distributional context, and comparison-level retrieved neighbors, then distills the rationale into student LLMs, enabling zero-overhead prediction and grounded explanation from raw features only. Experiments on multiple domain datasets show that TLRD significantly closes the performance gap between LLMs and state-of-the-art tree ensembles while producing grounded and readable explanations, offering a valuable reference for high-stakes decision-making. 

---
# AgriGov: A Structured Multilingual Dataset Curation for Indian Government Schemes for Farmers 

**Authors**: Mohsina Bilal, Gopakumar G  

**Link**: [PDF](https://arxiv.org/pdf/2606.08272)  

**Abstract**: AgriGov is a curated, trilingual (English-Hindi-Marathi) dataset designed to address the scarcity of domain-grounded multilingual resources for agricultural policies and farmer welfare schemes. Initially, we collected and structured data from 50 government schemes sourced from trusted portals using automated scraping techniques, organizing it into predefined semantic fields (e.g., title, eligibility, application process, documents, exclusions). Translations were performed using a pipeline combining Google Translate API, MarianMT, and human post-editing, resulting in a domain-specific Hindi-Marathi dataset comprising approximately 2100 source segments. To enhance coverage, we augmented this dataset with sentences from the Samanantar corpus, leading to approximately 8,000 sentence-aligned Hindi-Marathi parallel pairs. The dataset now offers robust resources for fine-tuning machine translation models in this domain. AgriGov is designed for applications in domain-adaptive machine translation, question answering, information retrieval, and summarization systems. Its key contribution is a schema-driven, human-corrected multilingual alignment pipeline that ensures domain fidelity, provides provenance, and supports reproducible experiments, enabling retrieval-augmented applications for farmer-facing tools. 

---
# SSR: Can Simulated Patients Learn to Stigmatize Themselves? Modeling Self-Stigma through Internal Monologue 

**Authors**: Kunyao Lan, Bingrui Jin, Zichen Zhu, Mengyue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08254)  

**Abstract**: Simulating patients with large language models (LLMs) is a promising tool for mental health training, but existing approaches fail to capture a key clinical reality: self-stigma. Patients experiencing self-stigma, the internalization of negative stereotypes, often exhibit context-sensitive resistance, such as avoidance, denial, or self-blame, which current models render as static or uniformly compliant behavior. To address this, we introduce a novel simulation framework grounded in the psychological 3A1H model of self-stigmatization. Our core innovation is the creation of a \textbf{Stigmatized Self-Reflection} (\textbf{SSR}) dataset, where we augment mental health dialogues with internal monologues that reflect stigma-aware reasoning. By fine-tuning LLMs with this data using a chain-of-thought approach, we train patient agents to dynamically adjust their level and expression of stigma based on conversational triggers. Evaluations demonstrate that our approach significantly outperforms specialized baselines, generating more authentic and situationally appropriate patient responses. This work provides a crucial step towards realistic stigma simulation for clinical training and empathetic dialogue systems. 

---
# ZAS-SQL: Distilling Rules from Failures for Zero-Shot Text-to-SQL 

**Authors**: Hongzhou Zheng, Yixin Gou, Wenjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08245)  

**Abstract**: Text-to-SQL translates natural language into executable SQL queries. Few-shot in-context learning methods built upon large language models (LLMs) achieve strong performance, yet their reliance on demonstrations limits cross-domain generalization and consumes substantial context window space. Existing zero-shot methods, lacking effective generation constraints, still fall short of few-shot approaches. We observe that LLM failures in zero-shot Text-to-SQL are not random but exhibit systematic, recurring patterns. Building on this observation, we propose a fully zero-shot Text-to-SQL framework that distills core generation rules from failure cases through a Map-Reduce-based rule distillation pipeline and improves generation quality via three complementary modules: knowledge-augmented schema representation, which supplements missing semantics in Data Definition Language; a rule-driven structured reasoning framework that suppresses structural deviations; and Execution-Guided Early Stopping, which enables low-cost self-correction. On Spider, the proposed framework achieves up to 87.2% and 88.6% execution accuracy on the Dev and Test sets, respectively, establishing a new zero-shot state-of-the-art and surpassing multiple few-shot and fine-tuning methods built upon GPT-4/4o. On the domain-specific dataset UrbanPlan, it achieves 81.3%, confirming that the rule distillation approach generalizes across domains. Moreover, when equipped with a 4B-parameter model, the framework surpasses zero-shot baselines of leading closed-source models, demonstrating strong model generality. 

---
# Building Comparative Motivation Profiles with Instrumental Interventions 

**Authors**: David Vella Zarb, Rustem Turtayev, Taywon Min, Jinghua Ou, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.08243)  

**Abstract**: Safety evaluations often infer latent motivations from behavioral patterns, but the construct validity of these inferences is unclear. We study this problem in alignment faking, where models comply with training objectives more often when they infer training pressure. This behavior is commonly interpreted as strategic self-preservation, but it may also reflect sensitivity to the model's inference about the expectation of researchers conducting the evaluation. We introduce a symmetric intervention framework for distinguishing these competing hypotheses. Instead of directly intervening on "scheming" or "sycophancy", we target instrumental processes entailed by each hypothesis: consequence-tracking and researcher-expectation tracking. We then compare how interventions on these processes affect the alignment faking. We study four openweight model organisms using synthetic document fine-tuning, activation steering, and prompting. Under synthetic document fine-tuning, Llama-3.1-70B, Llama3.1-405B, and Qwen-2.5-72B are more sensitive to expectation-tracking than consequence-tracking interventions. Activation steering on Llama-3.1- 70B supports the same broad picture, and prompt interventions broadly align with SDF profiles. Overall, alignment-faking behavior can be causally sensitive to evaluation-context expectations despite scheming-consistent scratchpads. Scheming and strategic-deception evaluations therefore need construct-validity checks, and symmetric instrumental interventions provide one such test. 

---
# Shared Semantics, Divergent Mechanisms: Unsupervised Feature Discovery by Aligning Semantics and Mechanisms 

**Authors**: Hyunjin Cho, Youngji Roh, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.08236)  

**Abstract**: As large language models are increasingly deployed in high-stakes settings, there is a growing need for tools that audit not only model outputs but also the internal computations that produce them. Circuit analysis is a central approach in mechanistic interpretability, but it is typically target-conditioned, explaining a single prompt paired with a chosen completion. This target-conditioned setup can obscure heterogeneity across a model's continuation distribution. We introduce distribution-level unsupervised feature discovery, which clusters sampled continuations using both semantic content and sequence-level mechanistic attributions, without manually specifying target outputs. Our method represents each continuation with a semantic embedding and a prefix-to-continuation attribution signature, then optimizes a rate-distortion objective that trades off semantic coherence, mechanistic consistency, and cluster granularity. Across clustering and steering analyses, the discovered clusters expose continuation modes that single-view baselines miss and provide interventional evidence that cluster signatures correspond to actionable mechanistic factors. Overall, our approach complements circuit analysis and behavioral evaluation by providing a scalable audit of the mechanisms underlying a model's continuation distribution. 

---
# AlignFed: Alignment-Aware Asynchronous Federated Fine-Tuning for Large Language Models in Heterogeneous Edge Environments 

**Authors**: Yan Wang, Ziyi Gao, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08197)  

**Abstract**: Large Language Models (LLMs) have significantly propelled the advancement of edge intelligence and have been widely deployed across various scenarios, including autonomous driving, industrial inspection, and personalized IoT services. However, the collaborative adaptation of LLMs on edge devices continues to face formidable challenges due to strict data privacy constraints, highly heterogeneous computing and communication resources, and the non-independent and identically distributed (non-IID) nature of local data. Federated Fine-Tuning (FFT) enables the collaborative optimization of distributed models without exposing raw data. Yet, traditional synchronous aggregation suffers from a severe straggler effect, resulting in high system latency and low resource utilization. Existing asynchronous federated learning methods are predominantly designed for small-to-medium-scale models and struggle to address the specific challenges inherent in LLM fine-tuning namely, model drift caused by stale updates, aggravated client drift stemming from data heterogeneity, and aggregation fairness imbalance resulting from the dominance of fast clients. To address these issues, this paper proposes AlignFed, an asynchronous federated fine-tuning framework for LLMs tailored to heterogeneous edge environments. AlignFed employs a lightweight multi-stage semantic alignment mechanism comprising three core modules: version-aware update grouping, cross-version semantic alignment based on a mini-batch calibration set, and fairness-aware aggregation that integrates both update freshness and client participation frequency. This framework effectively mitigates cross-version model drift and client drift while enhancing aggregation fairness, thereby achieving stable and efficient asynchronous federated optimization in scenarios characterized by high heterogeneity and significant update staleness. 

---
# GlobeAudio: A Multilingual Multicultural Benchmark for Naturalistic Evaluation of Large Audio-Language Models 

**Authors**: Ryner Tan, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08194)  

**Abstract**: Large Audio-Language Models (LALMs) integrate audio perception and language understanding within a unified framework, enabling a wide range of real-world applications. Despite recent advances, evaluation for LALMs remains heavily underspecified relative to real-world requirements: most lack true linguistic and cultural authenticity, while others fail to capture acoustic realism. To bridge this gap, we propose GlobeAudio, a multilingual and multicultural benchmark designed to evaluate naturalistic audio understanding. GlobeAudio consists of 5,637 multiple-choice questions across six typologically diverse languages, expertly crafted by native speakers grounded on naturally occurring audio. In order to do well, models must possess higher-level auditory reasoning skills and culturally grounded interpretation. We systematically evaluate representative closed-source and open-source LALMs, as well as cascaded ASR-LLM pipelines. Our experiments reveal substantial performance gaps under natural acoustic conditions, particularly for open-source models and low-resource languages. These findings highlight critical limitations of current LALMs and underscore the importance of naturalistic audio evaluation for future audio-language systems. GlobeAudio can be found at this https URL . 

---
# TextEconomizer: Enhancing Lossy Text Compression with Denoising Transformers and Entropy Coding 

**Authors**: Mahbub E Sobhani, Anika Tasnim Rodela, Chowdhury Mofizur Rahman, Dewan Md. Farid, Swakkhar Shatabda  

**Link**: [PDF](https://arxiv.org/pdf/2606.08184)  

**Abstract**: Lossy text compression reduces data size while preserving core meaning, making it well-suited for summarization, automated analysis, and digital archives. Despite the dominance of transformer-based models in language modeling, integrating context vectors and entropy coding into Sequence-to-Sequence (Seq2Seq) generation remains underexplored. A key challenge lies in identifying the most informative context vectors from encoder output and incorporating entropy coding to enhance storage efficiency while maintaining high-quality outputs, even under noisy text. We introduce TextEconomizer, an encoder-decoder framework paired with a transformer neural network that reduces variable-sized inputs by 50% to 80% without prior knowledge of dataset dimensions. Our model achieves competitive compression ratios via entropy coding while delivering near-perfect text quality, assessed by BLEU, ROUGE, METEOR, and semantic similarity scores. TextEconomizer operates with approximately 153x fewer parameters than comparable models, achieving a 5.39x compression ratio without sacrificing semantic quality. We also evaluate an LSTM-based autoencoder achieving a state-of-the-art 67x compression ratio with 196x fewer parameters, and LLaMAFormer, a modified transformer with 263x fewer parameters than ICAE while maintaining competitive text quality. TextEconomizer significantly surpasses existing transformer-based models in balancing memory efficiency and high-fidelity outputs, marking a breakthrough in lossy compression with optimal space utilization. 

---
# Constrained Paraphrase Consistency for LLM Hallucination Detection 

**Authors**: Shanshan Lin, Dongsheng Hong, Sibo Ju, Chao Chen, Xi Zhang, Xiangwen Liao  

**Link**: [PDF](https://arxiv.org/pdf/2606.08158)  

**Abstract**: Large language models (LLMs) can generate factually inconsistent claims, motivating accurate and scalable hallucination detectors. Prior work largely enlarges training sets via synthesis or new annotations, introducing increasing cost and potential bias while underusing the consistency implied by semantically equivalent paraphrases. We propose Consistency-Constrained Hallucination Detector (CCHD), which formulates training as a constrained optimization problem. The standard cross-entropy on original document-claim pairs is complemented by (i) paraphrase-consistency constraints bounding divergence across paraphrased views, and (ii) label-preservation constraints tying paraphrases to ground truth. We solve the problem by gradient descent-ascent over model parameters and per-view Lagrange multipliers, adding only a few scalar dual variables and no inference-time overhead. With DeBERTa and Flan-T5 backbones, CCHD consistently outperforms strong baselines (FactCG, MiniCheck, and AlignScore) on standard factuality benchmarks, demonstrating its superiority on hallucination detection. 

---
# Cross Paraphrastic Invariance Learning for Hallucination Detection 

**Authors**: Shanshan Lin, Dongsheng Hong, Sibo Ju, Chao Chen, Sihong Xie, Xiangwen Liao  

**Link**: [PDF](https://arxiv.org/pdf/2606.08157)  

**Abstract**: Large language models (LLMs) frequently generate hallucinations, which are unsupported by a source document. To avoid costly LLM-as-evaluator pipelines and the heavy annotation demands of existing classifiers, we propose CPIL (Cross Paraphrastic Invariance Learning), a two-stage Siamese framework that maximizes the utility of existing labeled data. Concretely, CPIL constructs informative training pairs by: (i) generating paraphrastic views of each document-claim example as positives, and explicitly aligning their representations to enforce invariance to surface form; and (ii) mining same-document, opposite-label pairs as hard negatives to sharpen document-sensitive decision boundaries. Then CPIL conduct a two-stage model training: Stage 1 performs contrastive pretraining to learn a paraphrase-invariant, grounding-aware embedding space; and Stage 2 attaches a lightweight classifier for binary groundedness. On the LLM-AggreFact benchmark (11 tasks), CPIL surpasses strong baselines concerning F1 scores with only ~1% labeled data, showing its prediction superiority and label efficiency. 

---
# When Languages Disagree: Self-Evolving Multilingual LLM Judges 

**Authors**: Xiyan Fu, Wei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08092)  

**Abstract**: Multilingual LLM-as-a-judge is widely used to evaluate model outputs across languages, but suffers from cross-lingual inconsistency (Fu and Liu, 2025). Existing methods typically treat this inconsistency as noise and mitigate it through voting or aggregation. In this work, we instead show that multilingual inconsistency can provide complementary evaluation signals. Our oracle analysis finds that sampling judgments across languages yields a higher performance upper bound than single-language judging, indicating that different languages potentially include complementary judgments. Motivated by this finding, we propose SEMJ, a self-evolving multilingual judge that leverages cross-lingual inconsistency for iterative refinement. SEMJ constructs multilingual variants of each input, collects independent judgments and rationales, and feeds inconsistent outputs back for self-reflection and re-evaluation. Experiments on multiple benchmarks show that SEMJ consistently outperforms voting and reflection baselines in both accuracy and cross-lingual consistency. Further analysis shows that inconsistency triggers useful re-evaluation, which improves judgment quality. 

---
# Aligned but Not Partner-Specific: Distinguishing How Multimodal LLM Agents Succeed in Reference Games Without Human-Like Conventions 

**Authors**: Po-Ya Angela Wang, Chinmaya Mishra, Aslı Özyürek, Paula Rubio-Fernández, Esam Ghaleb  

**Link**: [PDF](https://arxiv.org/pdf/2606.08081)  

**Abstract**: Repeated reference games test whether interlocutors replace their initially long descriptions with shorter, partner-specific conventions grounded in shared interaction history. Prior work shows that multimodal LLMs fail to become more efficient across rounds, although they align on the labels they use. How can we determine whether this alignment reflects partner-specific grounding rather than a shared task vocabulary? We address this question by comparing capable multimodal agent dyads with human dyads from the KTH Tangrams corpus. Our novel methodological contribution is a constrained pseudo-dyad baseline that matches the original referential task structure, but breaks partner history. This baseline enables us to test whether the observed label alignment depends on interaction with a specific partner. Across three analytic layers (task competence, description strategy, alignment dynamics), we find clear differences. Humans reduce effort through entrainment, compressing descriptions and increasing label alignment with partners. Agents instead maintain fixed effort levels, producing verbose descriptions from round one, with near-ceiling label overlap that is statistically indistinguishable between real and pseudo dyads. MLLMs thus achieve coordination without convention, succeeding by verbose description rather than by forming the compact, history-dependent referring expressions characteristic of human dialogue. 

---
# Support Vector Rubrics: Closing the Gap Between Self-Generated and Human Rubrics 

**Authors**: Mengyuan Sun, Yu Li, Zhuohao Yu, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2606.08077)  

**Abstract**: Rubric-based evaluation is a promising paradigm for judging large language model (LLM) outputs, yet self-generated rubrics lag human-annotated criteria on hard instances. We argue this discriminative gap reflects an objective mismatch: self-generated rubrics describe good responses, whereas effective criteria must discriminate between close candidates. To close this gap, we introduce SVR (Support Vector Rubrics), a framework that recasts rubric construction as max-margin boundary learning over preference data. SVR mines contrastive features from preference pairs into a rubric bank, learns a prompt-conditioned selector together with global rubric weights, and iteratively refines the bank through support-pair selection and adversarial probing of hard negatives. At inference, given only the prompt, SVR retrieves the top-rubrics from the bank and scores responses. On RubricBench, SVR narrows the gap to human reference rubrics from 24.1 to 0.3 points and outperforms strong self-rubric and judge baselines, and the learned bank transfers across judges without retraining. On RewardBench 1&2, and RM-Bench, it remains competitive with dedicated reward models, demonstrating broader reward modeling capability. Overall, boundary-defining rubrics offer a principled route to closing the discriminative gap in LLM evaluation. 

---
# "I understand your perspective": LLM Persuasion and Sycophancy through the Lens of Communicative Action Theory 

**Authors**: Esra Dönmez, Agnieszka Falenska  

**Link**: [PDF](https://arxiv.org/pdf/2606.08076)  

**Abstract**: Large Language Models (LLMs) can generate high-quality arguments, yet their ability to engage in nuanced and persuasive communicative actions remains largely unexplored. This work explores the persuasive potential of LLMs through the framework of Jürgen Habermas' Theory of Communicative Action. It examines whether LLMs express illocutionary intent (i.e., pragmatic functions of language such as conveying knowledge, building trust, or signaling similarity) in ways that are comparable to human communication. We simulate online discussions between opinion holders and LLMs using conversations from the persuasive subreddit ChangeMyView. We then compare the likelihood of illocutionary intents in human-written and LLM-generated counter-arguments, specifically those that successfully changed the original poster's view. We find that all three LLMs effectively convey illocutionary intent -- often more so than humans -- potentially increasing their anthropomorphism. Further, LLMs craft sycophantic responses that closely align with the opinion holder's intent, a strategy strongly associated with opinion change. Finally, crowd-sourced workers find LLM-generated counter-arguments more agreeable and consistently prefer them over human-written ones. These findings suggest that LLMs' persuasive power extends beyond merely generating high-quality arguments. On the contrary, training LLMs with human preferences effectively tunes them to mirror human communication patterns, particularly nuanced communicative actions, potentially increasing individuals' susceptibility to their influence. 

---
# SurgiQ: A Large-Scale Multi-Domain Benchmark for Evaluating Surgical Understanding in Large Language Models 

**Authors**: Ayah Al-Naji, Edoardo Fazzari, Saif Alkindi, Hamdan Alhadhrami, Preslav Nakov, Cesare Stefanini  

**Link**: [PDF](https://arxiv.org/pdf/2606.08071)  

**Abstract**: Reliable evaluation of large language models in surgery remains underdeveloped. Broad medical benchmarks test clinical knowledge, while surgery requires procedural reasoning, management trade-offs, negation handling, and selection among plausible operative decisions. We present SurgiQ, a text-only, source-grounded benchmark of 13,055 four-option multiple-choice questions spanning six surgical domains and four question formats: case-based, reasoning, best-option, and negative. SurgiQ is constructed from surgical textbooks, open-access papers, and examination material using a multi-stage generation, verification, and expert-audit pipeline. We evaluate 35 open-weight LLMs under a unified log-likelihood protocol. Our results show substantial remaining headroom: smaller models often remain near the 25\% random baseline, while the best model reaches 68.1\% accuracy. General-purpose models, especially Qwen2.5, outperform most biomedical models, suggesting that current medical specialization does not yet provide sufficiently broad surgical coverage. Calibration and error analysis further show that even strong models make confident mistakes on clinically plausible distractors, motivating more reliable and broader surgical LLM evaluation. 

---
# What's the Point? Spatial Grammar & Index Resolution for Sign Language Processing 

**Authors**: Oline Ranum, Simon Hadfield, Richard Bowden  

**Link**: [PDF](https://arxiv.org/pdf/2606.08056)  

**Abstract**: Sign language models are predominantly trained with gloss-sequence or text supervision, thereby under-modeling non-lexical and productive constructions. One comparatively tractable instance is spatial indexing: pointing gestures that assign discourse entities to spatial loci for subsequent co-reference, which lexicon-centric objectives largely fail to capture. We present a targeted evaluation of indexing in Sign Language Recognition, showing that despite comprising 10-15% of signing content, indexing is poorly recovered. We introduce a framework for training and evaluating indexing experts, establishing a baseline for index-aware sign language modeling. Our approach decomposes spatial reference resolution into index detection and discourse entity linking. The resulting mention representations enable automatic annotation and non-lexical structure modeling, and serve as an auxiliary indexing expert that augments a frozen SLR model at inference time. 

---
# Diffusion Language Model Parallel Decoding via Product-of-Experts Bridge 

**Authors**: Juntong Shi, Brian L. Trippe, Jure Leskovec, Stefano Ermon, Minkai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08048)  

**Abstract**: Diffusion language models (DLMs) offer substantial speed advantages through parallel decoding, but the lack of token dependencies limits generation quality compared to autoregressive (AR) models. Recent progress attempts to bridge the gap via importance sampling, with DLM being the proposal and AR being the target. However, due to the huge gap between their distributions, the sampling requires a large number of particles and is thus expensive to compute. In this paper, we introduce PoE-Bridge, a novel decoding framework that drastically improves generation speed and accuracy by introducing an intermediate distribution to bridge the gap. The distribution is constructed as a Product-of-Experts (PoE) of the DLM proposal and the AR target. With the intermediate distribution, we first use the DLM to draft multiple continuations in parallel, then apply rejection sampling to verify the drafted tokens and move the resulting candidates toward the PoE. We then use importance sampling to further correct the PoE-aligned candidates toward the AR target. We further propose several improved techniques, including mixed-temperature sampling for enhanced diversity and elastic rejection windows for reducing wasted verification. Empirically, PoE-Bridge achieves significantly improved accuracy with $5\times$ speedup over the standard DLM decoding approach, and recovers at least 95% of the target AR model's performance, efficiently advancing most of the quality gap on challenging mathematical reasoning and coding tasks. Our code is available at this https URL. 

---
# Arabic Sentence Segmentation Across Genres and Punctuation Conditions 

**Authors**: Mohammed Elkholy, Khalid N. Elmadani, Nizar Habash, Bashar Alhafni  

**Link**: [PDF](https://arxiv.org/pdf/2606.08025)  

**Abstract**: Sentence segmentation in Arabic is challenging due to ambiguous and inconsistent punctuation, with many texts lacking reliable sentence boundary markers. Existing approaches rely heavily on punctuation cues and are typically evaluated on well-formed text, limiting their robustness in realistic Arabic settings. To address this, we introduce AraSEG, a genre-diverse sentence segmentation corpus spanning eight genres and a wide range of punctuation and document structure conditions. Using AraSEG, we evaluate LLMs, lightweight encoder models, and dependency parser-based models under increasingly challenging segmentation settings. Our experiments show that lightweight encoders, and even dependency parser-based models, outperform LLMs in the most challenging settings. We further investigate the effects of training data size and genre diversity, finding that performance eventually saturates and cross-genre generalization remains challenging. We also demonstrate that accurate sentence segmentation substantially improves downstream dependency parsing. We make our code, data, and models publicly available. 

---
# Rewrite to Translate, Translate to Reward: Reinforcement Learning for Source Rewriting in Machine Translation 

**Authors**: Boxuan Lyu, Haiyue Song, Zhi Qu, Hidetaka Kamigaito, Kotaro Funakoshi, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2606.08011)  

**Abstract**: Although directly prompting off-the-shelf Large Language Models (LLMs) to generate meaning-preserving source rewrites can effectively enhance Machine Translation (MT) quality, doing so requires manually tuning prompts for different MT models. In this work, we propose RLSR (Reinforcement Learning for Source Rewriting), a novel RL-based framework for training a source rewriting model without tuning prompts for each MT model. RLSR optimizes the rewriting model by directly using the improvement in downstream translation quality yielded by each rewritten source as the reward. Extensive experiments across six MT models and 16 language pairs demonstrate that our 4B rewriting models trained via RLSR significantly outperform the no-rewriting baseline and existing same-scale prompt-based rewriting baselines, while achieving competitive performance against prompt-based baselines based on the 235B LLM. 

---
# Summarization is Not Dead Yet 

**Authors**: Dongqi Liu, Chenxi Whitehouse, Zheng Zhao, Zhuchen Cao, Jian Li, Yabiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08000)  

**Abstract**: The progress of large language models (LLMs) has fueled claims that model-generated summaries rival or even surpass human-written references, raising questions about whether summarization remains an open research problem. We re-examine this narrative through a multi-track evaluation covering five diverse datasets and five state-of-the-art LLMs, combining controlled human assessment, bias-mitigated LLM-as-Judge protocols, factuality verification against external knowledge, and corpus-level linguistic analysis. Our findings reveal a more nuanced landscape in which human reference summaries continue to demonstrate advantages in informativeness and faithfulness, whereas LLM outputs are preferred mainly for surface-level coherence and fluency. Factuality verification indicates that human references remain more reliable, particularly for claims involving reasoning or synthesis, and linguistic analysis uncovers a pattern of stylistic homogeneity across different models. These observations suggest that current LLMs have raised the floor of summarization quality, but the ceiling of their performance remains below human capabilities. 

---
# MC-PDD: Masked Corpus-Level Pretraining Data Detection for Black-Box Large Language Models 

**Authors**: Kaixin Lan, Mu You, Tao Fang, Binkai Ou, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2606.07996)  

**Abstract**: Pretraining is fundamental to the development of Large Language Models (LLMs), yet the opacity of pretraining data complicates model analysis and raises ethical, legal, and fairness concerns. Detecting whether specific datasets were used during pretraining is, therefore, critical. Existing state-of-the-art methods typically rely on access to model probability distributions, making them unsuitable for closed-source LLMs that provide only input-output interfaces. To address this limitation, we introduce Masked Corpus-level Pretraining Data Detection (MC-PDD), a novel method inspired by the masked language modeling paradigm. MC-PDD masks highly specific tokens in each text and prompts the LLM to predict the missing content. It then assesses whether the difference in prediction hit rates between a candidate corpus and a reference non-member corpus is statistically significant. Based on this comparison, MC-PDD determines whether the candidate texts were likely included in the model's pretraining data. Experimental results demonstrate clear and consistent differences in prediction hit rates between pretrained and unseen data across three datasets, for both open-source and closed-source LLMs. Despite operating under a stricter black-box setting, MC-PDD achieves performance comparable to existing detection methods. Our approach enables practical applications such as model auditing and data copyright verification using only standard API access. Upon acceptance, we will publicly release the code and datasets. 

---
# Customer-Agent: Overcoming Context Limitations in Ultra-Long Shopping Trajectories via Tool-Augmented Agents and RLVR 

**Authors**: Hongye Liu, Rongmei Lin, Anurag Kashyap, Hejie Cui, Ricardo Henao, Besnik Fetahu, Bing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.07995)  

**Abstract**: Understanding customer shopping trajectories is essential for enabling personalized shopping experiences. However, shopping records (i.e., customer's search, clicks, purchases, etc.) often span long time horizons over multiple years, resulting in extremely long trajectories that pose significant challenges for existing large language models (LLMs). Despite the importance of this problem, existing benchmarks are limited to short customer trajectories, while real-world trajectories from large e-commerce platforms are rarely accessible due to data privacy constraints. To address this gap, we introduce ShopTrajQA, a long-context evaluation benchmark constructed from real-world product information and simulated shopping trajectories. The dataset includes variants of up to 32k and 64k tokens, enabling systematic evaluation of model robustness under varying context lengths. Through comprehensive benchmarking of frontier LLMs, we identify critical performance gaps in reasoning over long shopping trajectory data. To address these challenges, we propose a Customer Agent Framework for ultra-long context management. Leveraging a Reinforcement Learning with Verifiable Rewards (RLVR) agentic training paradigm, our approach stores trajectories as external local files and trains the agent to autonomously retrieve and parse them through code-interpreter interactions (e.g., SQL queries), effectively bypassing the fixed in-context window constraints of LLMs. Experimental results demonstrate that our framework achieves strong performance for ShopTrajQA and shows generalization to other complex reasoning tasks. 

---
# MechLens: Late Crystallization of Factual Knowledge Explains Intervention Effectiveness in Language Models 

**Authors**: Xueping Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.07978)  

**Abstract**: Understanding where LLMs store factual knowledge is critical for hallucination mitigation. We systematically quantify Late Crystallization: factual knowledge does not gradually emerge across layers but "crystallizes" abruptly at the final layers. Across five model families (Pythia, Gemma, Qwen2.5, Llama-3.1, Mistral; 0.5--14B), 26.8%--93.4% of correct answers never enter top-10 predictions at any intermediate layer, with late emergence (>80% depth) consistent across architectures. Cross-scale (Qwen2.5-14B) and cross-benchmark (MMLU: 98.2%) results confirm generality; tuned lens rules out probe artifacts. A sentiment-classification control (0.5% for Qwen vs. 85.9% factual; 2.0% for Mistral vs. 26.8%) confirms the phenomenon is specific to factual recall.
Late Crystallization yields a crystallization-guided intervention principle: CAA outperforms DoLa on moderate-crystallization models (Llama, Mistral; p<0.001), with a directionally consistent reversal on high-crystallization Qwen (+25.4% vs. +15.5% MC1, p=0.069). LayerNorm ablation shows crystallization is intrinsic to the residual stream; LN scaling (x1.2) yields +11.8% MC1 with zero inference overhead. We further reveal a Computability-Memorization Spectrum: computable knowledge crystallizes earlier (layer 22.1/28) than memorized facts (28.0/28). We release MechLens supporting five model families. 

---
# Defending Against Malicious Finetuning by Scaling Train-time Adversarial Attacks 

**Authors**: Haoming Wen, Shi Chen, Qingyu Shi, Siyuan Liu, Minrui Luo, Jingzhao Zhang, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2606.07970)  

**Abstract**: Current open-weight large language models (LLMs) are prone to malicious finetuning attacks, which could compromise the safety alignment of LLMs with only a few steps of supervised finetuning (SFT) on poisoned datasets. Existing alignment-stage defenses are primarily designed to defend against attacks that use parameter-efficient finetuning methods. However, they fail to defend against stronger attacks that use full-parameter finetuning. In this paper, we propose Patcher, a method inspired by adversarial training and bi-level optimization, to combat such attacks. Patcher strengthens the simulated attack by scaling up the optimization steps in the adversarial loop, thus forcing the defender to find model parameters that are insensitive to stronger attacks. Furthermore, we propose an efficient parallel algorithm to implement Patcher, decreasing the wall-clock time of training while preserving Patcher's performance. Extensive experiments show that Patcher substantially improves the model's robustness compared to vanilla SFT alignment, and transfers to diverse attack scenarios and model sizes. Code is available at this https URL. 

---
# Neutrality Bites: Gender Representation in AI-Generated Animal Stories 

**Authors**: Imani Finkley, Yuanxi Li, Melanie Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2606.07969)  

**Abstract**: Gender bias in AI-generated stories is a well-documented problem. While much attention has been paid to reducing or mitigating this bias, it is not always clear whether interventions produce genuinely fairer results. To investigate this issue, we examine how large language models (LLMs) handle gender assignment in a narrative context that is popular, highly ambiguous, and also known to closely reproduce human stereotypes: stories about talking animals. We prompt six leading LLMs to complete an English-language story about seven different anthropomorphic animal characters whose gender is unstated. We additionally iterate with four different narrative settings and a range of model temperatures. Across the 23.8K stories, we find that models frequently avoid gendering the animal character in the story (19% on average) or use gender-neutral language like "it" or "its" (38.2% on average). However, when gender is assigned, there is a significant masculine bias. Feminine animal characters are virtually absent, present in just 2.2% of stories vs. 40.6% that feature masculine characters. Our findings point to a broader argument: neutrality bites. In other words, models that prioritize neutrality to address social bias may actually contribute to the erasure of marginalized perspectives and identities. We suggest that alternative strategies beyond neutrality need to be pursued, such as ones that more equally distribute social possibilities across imagined subjects. 

---
# What Does Debiasing Really Remove? A Geometric Study of PCA-Based Gender Debiasing in Word Embeddings 

**Authors**: Alexey Kresin, Tchifou M. Dieffi, Tomer Caspi  

**Link**: [PDF](https://arxiv.org/pdf/2606.07964)  

**Abstract**: Debiasing methods based on principal component analysis (PCA) are broadly used to reduce gender bias in word embeddings used in LLMs, yet it remains unclear what aspects of bias they actually remove and how destructive this process is. These methods are based on the understanding that bias resides in a low-dimensional subspace, with the assumption that most of it can be captured by a few principal components. In this work, we conduct a systematic geometric analysis of PCA-based gender debiasing and investigate what is actually removed from the embedding space. Our experiments across multiple embeddings show that direct gender bias is primarily concentrated in the first principal component, supporting the low-rank bias hypothesis. However, associative bias measured by WEAT does not align with these principal directions and is instead spread across multiple embedding dimensions. Furthermore, as expected, we demonstrate that removing an increasing number of principal components leads to a consistent degradation of the embedding geometry, affecting semantic structure and vector relationships. These results reveal that PCA-based debiasing operates as a trade-off: while it effectively reduces certain forms of direct bias, it fails to eliminate distributed associations and introduces geometric distortion. Moreover, there is no universal optimal level of debiasing, as the balance between bias reduction and semantic preservation depends on the chosen metric and embedding. Overall, our findings suggest that bias in word embeddings is not purely low-rank and that simple subspace removal methods may be insufficient for comprehensive debiasing. 

---
# From `May' to `Is': Certainty Distortion in Language Model Rewriting 

**Authors**: Catarina G Belem, Shang Wu, Hongyu Yao, Mark Steyvers, Sameer Singh, Padhraic Smyth  

**Link**: [PDF](https://arxiv.org/pdf/2606.07951)  

**Abstract**: Humans increasingly turn to Language Models (LMs) in ways that shape beliefs and drive decisions, including discussing, rewriting, and summarizing information from scientific articles, news, and medical reports. However, in these domains, where how confidently a claim is expressed matters, little is known about whether LMs faithfully preserve it. In this work, we investigate certainty distortion in LMs, defined as meaningful changes in expressed certainty when semantic content is preserved. We propose an LM-based evaluation metric that is consistent with population-level judgments of certainty. Using this metric, we characterize certainty distortion across different sizes and families of models in the context of scientific and medical communication tasks. Our results show that certainty distortion affects up to 75\% of LM outputs and is systematically asymmetric in rewriting tasks with most LMs being 1.5-2$\times$ more likely to increase the expressed certainty than to decrease it. These effects can compound over repeated paraphrasing: in the medical domain, claude-haiku-4-5 increases certainty of 20\% examples after a single iteration, increasing to 40\% after five iterations. Prompt-based interventions reduce overall certainty distortion but do not eliminate it. Together, these findings reveal a general bias toward inflating expressed certainty, with direct implications for users who rely on LMs in high-stakes domains. 

---
# Illusions of the Gold Standard: A Large-scale Analysis of Human Evaluation Protocols for Long-form Text Generation 

**Authors**: Katelyn Xiaoying Mei, Yi-Li Hsu, Minjoon Choi, Zongwan Cao, Chenjun Xu, Bingbing Wen, Su Lin Blodgett, Lucy Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07936)  

**Abstract**: Human evaluation plays a critical role in assessing the quality of generated text. However, the reliability and reproducibility of these evaluations depend on transparent and well-documented protocols -- details that are frequently missing in current practice. In this work, we conduct a large-scale analysis of human evaluation protocols for evaluating long-form generation tasks in *CL conference publications from 2023--2025, including a full manual review of 284 papers and LLM-assisted analysis for another 1.8k+ papers. We define a set of 20 reportable criteria related to reproducibility of human evaluation studies, and apply these criteria to systematically examine reporting norms and practices within the community. We find widespread under-reporting of important aspects of human evaluation study design, leading to ambiguity about what was measured and how, who contributed judgments, and how judgments should be interpreted. Based on these findings, we outline actionable recommendations to support more transparent and reproducible reporting in future research. Our analysis code and annotated dataset can be found at: this https URL 

---
# ROSUM-MCTS: Monte Carlo Tree Search-Inspired HDL Code Summarization with Structural Rewards 

**Authors**: Prashanth Vijayaraghavan, Charles Mackin, Luyao Shi, Apoorva Nitsure, Ashutosh Jadhav, David Beymer, Tyler Baldwin, Ehsan Degan, Vandana Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.07925)  

**Abstract**: Large language models (LLMs) have shown promise in code summarization, yet their effectiveness for Hardware Description Languages (HDLs) like VHDL and Verilog remains underexplored. We propose ROSUM-MCTS, an LLM-guided approach inspired by Monte Carlo Tree Search (MCTS) that refines summaries through structured exploration and reinforcement-driven optimization. Our method integrates both local and global context via a hierarchical candidate expansion mechanism and optimizes summaries using a composite reward function balancing functional correctness (FC), local content adequacy (LCA), and fluency. We evaluate ROSUM-MCTS on the VHDL-eval and Verilog-eval datasets, demonstrating its consistent outperformance over baseline methods by leveraging structured bottom-up refinement and reinforcement-based optimization. Ablation studies confirm the necessity of both local and global expansion strategies, as well as the importance of balancing FC and LCA for optimal performance. Furthermore, ROSUM-MCTS proves robust against superficial modifications, such as variable renaming, maintaining summary quality where baselines degrade. These results establish ROSUM-MCTS as an effective and robust HDL summarization framework, paving the way for further research into reinforcement-enhanced code summarization. 

---
# Beyond Individual Personas: Aligning Synthetic Dialogue to Population-Level Behavior Distributions 

**Authors**: Xinyi Liu, Rinat Khaziev, Hooshang Nayyeri, Emine Yilmaz, Charith Peris, Hari Thadakamalla  

**Link**: [PDF](https://arxiv.org/pdf/2606.07893)  

**Abstract**: Synthetic dialogue corpora are increasingly used as proxies for target dialogue data, yet persona-grounded generators optimize individual conversations rather than corpus composition, yielding locally plausible dialogues with distorted population-level behavior mixes. We introduce GroupPersona, a framework that aligns synthetic dialogue corpora to the behavior distribution of a reference corpus. GroupPersona turns population statistics into generation controls: it separates each dialogue's core behavioral signature from predictable side effects, and uses the resulting behavioral groups to condition user agents on the interaction patterns that define the reference population. We evaluate GroupPersona on four corpora crossing two dialogue sources, assistant-style and Reddit-derived, with two construction variants: structure-preserving and variation-enhanced. GroupPersona lowers Jensen-Shannon divergence between synthetic and reference distributions over 12 behavior attributes from 0.234 to 0.177 relative to the strongest average baseline, a 24.4% reduction, and is best or tied-best on all four corpora while preserving structural alignment. It also achieves the closest calibration to reference-conversation quality scores, reducing mean absolute deviation from the reference-conversation profile to 0.63 versus 0.91 for the next-best baseline. 

---
# Whose Norms? Disentangling Cultural and Personal Alignment in Large Language Models 

**Authors**: Angana Borah, Isabelle Augenstein, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2606.07877)  

**Abstract**: Large language models are increasingly used for social decision-making situations that require balancing cultural norms with personal preferences. For example, a user preferring honesty might ask whether to correct a coworker publicly when local norms favor indirect feedback. Yet existing research studies cultural alignment and personalization largely separately. We introduce PACT, the Personal-Preference and Cultural-Norm Trade-off framework, which evaluates whether models choose to follow a cultural norm or allow personal preferences. We find that LLMs vary in how rigidly they enforce cultural norms, with behavior shifted more by country context (7.8%) than age (1%) and gender (0.7%) and shifting non-uniformly after instruction tuning. Furthermore, our five-country human study on PACT shows that culture-following in humans is mainly driven by scenario country, with the lowest agreement when participants judge their own cultural contexts, showing within-culture pluralism. Finally, human-LLM alignment experiments show that models can match majority choices, but fail to capture response distributions and uncertainty (with best correlations reaching only 0.24). Together, these findings motivate alignment evaluations that go beyond majority to capture cultural pluralism and disagreement in social judgment. 

---
# The Cold-Start Safety Gap in LLM Agents 

**Authors**: Chung-En Sun, Linbo Liu, Tsui-Wei Weng  

**Link**: [PDF](https://arxiv.org/pdf/2606.07867)  

**Abstract**: Are tool-calling LLM agents equally safe throughout a conversation? We discover they are not: agents are most vulnerable at the very start of a session and become substantially safer after a few regular agentic tasks -- a phenomenon we term the cold-start safety gap. To study this systematically, we introduce Safety Over Depth for Agents (SODA), a benchmark that controls how many regular agentic tasks the agent completes before encountering a safety threat, supporting up to 20 preceding tasks. Evaluating 7 models from 4 families, safety improves by 9--52% as the number of preceding regular agentic tasks increases from zero to twenty. Representation analysis confirms that model hidden states gradually shift toward a safety-aligned region as more preceding tasks are present. By systematically studying which part of the preceding conversation matters most, we find that the regular agentic tasks themselves are the primary driver of safety, while the agent's own prior responses have less effect on safety but are essential for preserving later utility. This conclusion is further supported by evaluation on open-source safety benchmarks (AgentHarm, Agent Safety Bench) and utility benchmarks (BFCL, API-Bank), confirming that warming up the agent with regular agentic tasks before deployment makes it safer and preserves full capability. Based on these findings, we recommend a simple deployment strategy: having the agent complete a few regular agentic tasks before possible exposure to safety-critical requests mitigates the cold-start safety gap. Our code is available at this https URL 

---
# Beyond English benchmarks: clinical llm evaluation in Brazilian Portuguese 

**Authors**: Giordano de Pinho Souza, Glaucia Melo, Josefino Cabral Melo Lima, Daniel Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2606.07853)  

**Abstract**: Large Language Models are transforming the support for clinical decision and their application in real scenarios. Yet, most benchmarks are conducted in English, and cross-lingual evaluation is needed to tackle the language gaps in global access. We introduce ClinicalBr, the first bilingual benchmark for clinical decision built from real Brazilian case reports. The corpus contains 2,892 cases drawn from 28 SciELO medical journals, spanning 18 specialties, and is structured as parallel Portuguese-English pairs. Each case supports four evaluation tasks: diagnosis retrieval, differential diagnosis, exam recommendation, and treatment planning. We evaluate four models: MedGemma-27B, Sabiá-4, DeepSeek-R1, and o3-mini, across both languages. The central finding is that the Portuguese-English performance gap is task-dependent, not general. In diagnosis retrieval, English yields a consistent advantage across all models, with +7.5-12.1 accuracy points. This advantage disappears in differential diagnosis, exam recommendation, and treatment planning, where confidence intervals cross zero for most models and Portuguese completeness scores are marginally higher. Brazilian-endemic conditions proved easier than the full corpus, not harder, indicating that tropical presentations are adequately represented in current pre-training. Exam recommendation was the hardest task across all models and both languages, with F1 scores below 0.10, well below the differential diagnosis ceiling of 0.20-0.27. 

---
# The ACUTE Protocol: Operationalizing Language Model Activations for Better Calibration, Utility, and Trust 

**Authors**: Nishant Subramani, Palash Goyal, Yiwen Song, Mani Malek, Yuan Xue, Tomas Pfister, Hamid Palangi  

**Link**: [PDF](https://arxiv.org/pdf/2606.07822)  

**Abstract**: As language models improve and become increasingly deployed to solve a variety of tasks, trustworthiness becomes essential. Calibration is a good proxy for trust: well-calibrated confidence estimates help inform the risk versus reward tradeoff when trusting a specific model output. Unfortunately, even as models improve, they remain poorly calibrated, often biasing towards overconfidence. Additionally, calibration can be gamed: a policy that always predicts the base rate is perfectly calibrated, but completely uninformative. To resolve this, we develop a new metric, expected utility renormalized by the oracle (EURO), that balances calibration and informativeness. We also propose a general-purpose activation-based confidence, utility, and trust estimation protocol (ACUTE) to appropriately adjudicate uncertainty. The ACUTE protocol provides flexible, sample-efficient, and compute-efficient confidence estimators for 3 tasks including multiple choice question answering, tool-calling, and scientific document summarization across 6 models from 4 model families. ACUTE outperforms strong baselines on EURO, while maintaining low calibration error. Taken together, our work shows that equipping LLMs with the ACUTE protocol can improve calibration, utility, and trustworthiness in numerous settings. 

---
# Representational Similarity and Model Behavior in Multi-Agent Interaction 

**Authors**: Yujin Potter, Seun Eisape, Shiyang Lai, Alexander Huth, James Evans, Been Kim, Jacob Eisenstein, Dawn Song, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2606.07818)  

**Abstract**: Researchers have shown that neural similarity among humans predicts social closeness and cooperative success, whereas innovation often emerges from interactions among dissimilar individuals. We investigate whether these principles extend to artificial intelligence by examining interactions between large language models. In our experiments, 276 model pairs interact across eight games spanning both cooperation and novelty. We find that pairs with more similar representation spaces achieve significantly higher cooperation but exhibit reduced novelty and creativity. The effects of representational similarity on cooperation and novelty remain robust even after controlling for other factors such as performance disparity and model size. We also find that similarity in the early layers consistently shows the strongest association with cooperation and novelty, compared to the middle and later layers. This suggests that a central factor underlying these patterns could be the extent to which the two models share lexical and semantic grounding. Overall, representational similarity can be an important consideration in multi-agent system design. 

---
# SLMJury: Can Small Language Models Judge as Well as Large Ones? 

**Authors**: Anish Laddha, Nitesh Pradhan, Gaurav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2606.07810)  

**Abstract**: Large language models (LLMs) are widely used as judges for evaluating model outputs, but their high cost, latency, and opacity limit scalability. We introduce SLMJury, a framework for evaluating small language models (SLMs) as judges across two paradigms: closed-ended binary correctness and open-ended quality scoring. We benchmark 16 SLM judges (0.6B-14B parameters) from four model families across ten benchmarks: eight closed-ended tasks spanning mathematical, scientific, and general reasoning (N=64,824 judgments per configuration), plus SummEval and MT-Bench for summarization and conversational scoring. We formalize judging as a budget-conditioned function and study five dimensions. Four findings emerge. (1) The overthinking effect is domain-dependent: for most judges quick 10-token verdicts match or beat extended reasoning on mathematical judging (by 2-7% where they help), while reasoning wins on general tasks by up to 23%. (2) Domain generalization separates model families, with math-to-general accuracy gaps ranging from under 10% to nearly 40%. (3) Closed-ended and open-ended judging draw on different capabilities: the best binary judge (Phi-4) drops to rank 9 on MT-Bench, while reasoning-trained models invert this ordering. (4) Under the Reflect-Critique-Refine (RCR) debate protocol, multi-agent debate degrades accuracy across all tested configurations, whereas the top judges resist six adversarial personas with <=0.55% variance. Reliable automated evaluation does not require large proprietary models, yet no single SLM dominates. The leaderboard is available at this https URL, and our framework code and pip package are publicly available at this https URL and this https URL. 

---
# Evaluating RAG Reliability under Clean, Misleading, and Mixed Retrieval 

**Authors**: Sevgi Yigit-Sert  

**Link**: [PDF](https://arxiv.org/pdf/2606.07783)  

**Abstract**: Retrieval-Augmented Generation (RAG) is widely used to improve the factual reliability of large language models (LLMs) by grounding answers in retrieved evidence. In misinformation-rich environments, however, retrieved content may include plausible but incorrect information, raising concerns about the reliability of RAG-based information access systems.
In this work, we propose an evaluation protocol to systematically test how the RAG system handles conflicts between parametric knowledge and evidence retrieved from context with varying amounts of misleading information. We target correct answers to factoid questions that the model responds to correctly, even when there is no retrieval, and use this to test the system with clean, poisoned, and mixed evidence.
The proposed analytical framework combines parametric override and confidence metrics to assess when and how misleading information affects the generation process of LLMs. This study aims to provide insights into the robustness of RAG systems in information disorder scenarios. 

---
# Unlocking Latent Value: Taxonomy-Guided Recovery of High-Performing Data from Low-Tier Web Corpora 

**Authors**: Neeraj Varshney, Sanket Lokegaonkar, Nasser Zalmout, Qingyu Yin, Priyanka Nigam, Bing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.07778)  

**Abstract**: Dominant web data curation pipelines for pretraining collapse document quality into a single composite score, systematically missing high-value content along dimensions the scorer underweights. We present a taxonomy-driven framework that recovers this value by filtering along semantically meaningful dimensions that composite scores fail to capture. First, building on the ESSENTIAL-WEB taxonomy, we introduce two novel dimensions: timeliness and cultural specificity, both of which show low pairwise NMI with existing ones. We annotate 14M documents using Qwen2.5 32B and distill into a lightweight 0.5B model. To enable rapid corpus-wide annotation, we additionally train a 73M multi-task MLP on E5 embeddings, achieving 50x inference throughput. Second, to navigate the combinatorial explosion of filter configurations, we introduce a compute-efficient two-pass framework: Pass 1 identifies the strongest dimension signals at small scale; Pass 2 constructs and evaluates conjunctive and disjunctive compound filters from the top performers - identifying high-performing configurations at a fraction of full scaling-law cost. Applying the selected filters to deprioritized web data, taxonomy-filtered subsets outperform their unfiltered baselines and even surpass the highest-quality tier. On mid-tier data, our best filter improves over its unfiltered baseline by 12.1% on reasoning, 9.5% on coding, and 2.0% on knowledge benchmarks, exceeding unfiltered top-tier data by 6.7% on reasoning and 13.7% on coding. Furthermore, filtered data from two tiers below the typical production threshold improves by 22.3% on reasoning and 19.5% on coding over its unfiltered baseline, surpassing top-tier data on coding benchmarks. These results establish that vast latent value remains locked in deprioritized web data, and that multi-dimensional taxonomy filtering is a principled, compute-efficient key to unlocking it. 

---
# ReadingMachine: A Computational Methodology for Structured Corpus Reading and Large-Scale Synthesis 

**Authors**: James Morrissey  

**Link**: [PDF](https://arxiv.org/pdf/2606.07753)  

**Abstract**: ReadingMachine is a computational methodology for structured corpus reading that uses large language models to perform bounded reading operations over entire document collections. Rather than relying on retrieval or recursive summarization, the approach decomposes analysis into inspectable stages including insight extraction, semantic clustering, theme generation, and iterative omission detection. By delaying irreversible compression and explicitly tracking intermediate representations, the method prioritizes coverage, traceability, and preservation of disagreement across large corpora. The system is demonstrated on a heterogeneous corpus of 152 industrial policy documents, producing more than 17,500 extracted insights and a structured thematic map. ReadingMachine is released as an open-source experimental framework for large-scale qualitative synthesis and corpus analysis. 

---
# Subtitle-Aligned Fine-Tuning of Whisper for Swiss German ASR: Benchmark Contamination, Convention Mismatch, and an Honest Baseline at 25.6% WER (13.8% cWER) 

**Authors**: Felix Akeret  

**Link**: [PDF](https://arxiv.org/pdf/2606.07608)  

**Abstract**: We present a systematic study of fine-tuning OpenAI's Whisper large-v3 for Swiss German ASR, using 1,367 hours of broadcast speech paired with Standard German subtitles as weak supervision. Through 16 iterative training runs on an NVIDIA DGX Spark (Grace Blackwell, 128 GB unified memory, up to 1 PFLOP FP4), we compare LoRA and full fine-tuning of the 1.55B-parameter model, investigate hallucination root causes, and quantify the effect of data quality, subtitle alignment, and training strategy. Our best model achieves 25.6% measured WER on the All Swiss German Dialects Test Set (ASGDTS) in an honest evaluation on strictly disjoint data. A harmonized error analysis separating genuine errors from valid stylistic variation (tense, word order, Swiss orthography) yields a content WER (cWER) of 13.8%, counting only actual recognition failures. Bias-corrected estimation reduces this to 8.5%, suggesting the true error rate is roughly one third of measured WER.
We demonstrate that published state-of-the-art Swiss German ASR results (17.1-17.5% WER) are inflated by benchmark contamination: a vanilla Whisper model self-trained on the ASGDTS test set with zero Swiss German data achieves 13.88% WER, surpassing all published systems. Experiments with Phi-4-multimodal show an even stronger memorization effect (3.9% WER), revealing that the benchmark primarily measures convention matching rather than dialectal comprehension.
We release two models, a LoRA adapter (25.32% WER, 13.9% cWER) and a full fine-tuned model (25.60% WER, 13.8% cWER), among the few publicly available, honestly evaluated Whisper models for Swiss German, under Apache 2.0 with full reproducibility, requiring no institutional data agreements. 

---
# Function-Vector Heads Are Two Populations: Writers and Cancellers in In-Context Learning 

**Authors**: Han-yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07560)  

**Abstract**: Function-vector (FV) heads (Todd et al., 2024) are typically identified by the magnitude of their causal contribution to in-context rule tasks, under the implicit assumption that the top set is a homogeneous functional class. This assumption fails. We replace magnitude-only ranking with a sign-preserving criterion (refined DLA + permutation FDR) and validate each candidate by path patching. The FV head population then splits into two opposing sub-populations: writers push the rule-correct logit up; cancellers push it down. A four-condition canonical verdict holds in $13/15$ cells across three model families and six Pythia scales, and a sign-shuffle rejects homogeneity in $5/6$ main cells. The structure is invisible to magnitude-only ranking: Todd's top-$20$ captures $64\%$ of cancellers but only $4\%$ of writers on the hierarchical task, and $59\%$ of writers but only $8\%$ of cancellers on the modular task. We rule out six artefact accounts on all $27$ canceller (cell, head) pairs: induction overlap, sinks, generic importance, rank-$1$ copy-suppression, V-cascade, and rank-nearest non-FV controls. Zero-ablating cancellers yields $+0.13$ to $+0.29$ nats of logit gain in $6/6$ main cells with a directionally consistent $+2$ to $+7$ pp accuracy effect. 

---
# Phantom transitions in language model fine-tuning 

**Authors**: Vaibhav Prakash, Jayasri Dontabhaktuni  

**Link**: [PDF](https://arxiv.org/pdf/2606.07559)  

**Abstract**: Fine-tuning a language model on contexts whose correct completion has a near-synonym competitor often fails silently. The cross-entropy loss decreases monotonically while the correct token never overtakes the competitor in rank. We study this regime across five transformer architectures spanning two families and a fivefold parameter range, on ten hand-selected near-synonym contexts. We instrument these failures with an order parameter combining the predicted distribution and pairwise embedding overlaps. It decomposes additively into a signal, tracking the model's commitment to the correct token over its nearest competitor, and a background drag, set by how the embedding bulk leaks probability into the score. This isolates two failure modes. In kinematic failure the signal stays small. In structural failure the drag actively worsens as fine-tuning proceeds. We observe sharp catapult-like jumps in the order parameter that resemble a phase transition. A central negative result organises the paper. The transitions are phantoms. The spontaneous-symmetry-breaking interpretation is ruled out by direct measurement. Catapult-like jumps still appear under LoRA fine-tuning with the token embedding matrix exactly unchanged during training, where no geometric phase transition is possible. The discontinuity lives entirely in the softmax readout. A small number of dimensionless quantities organise the trajectory across architectures. One is consistent across all five under full fine-tuning. A second sorts architectures into two classes by bulk embedding distribution and predicts LoRA sufficiency. As a blind test, the framework predicts the critical learning rate of a held-out architecture, not used to fit any parameter, to within 2.1% of a subsequent learning-rate sweep. Findings concern the near-synonym mechanism only and should not be extrapolated without recalibration. 

---
# Priors Persist Through Suppression: A Stroop Paradigm for Lexical Override 

**Authors**: Han-yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07555)  

**Abstract**: Glossaries, technical specifications, and system prompts routinely ask language models to use familiar words in unfamiliar ways. When this works, the lexical prior persists through override rather than being replaced: it continues to operate after the local rule applies, with the rule lowering its logit rather than installing the new meaning on top. We test this with a Stroop-style paradigm: a remapping rule ("doctor" means "forest") pitted against the query word's lexical-prior distractor ("hospital"), with matched neutral controls. Across 11 open-weight models spanning four families and 1B--9B parameters, lexical-prior strength predicts interference even after item-level controls for answer prior, frequency, tokenization, and prompt wording. Activation patching on five aligned models locates a source-position triplet (definition subject, definition target, query word) that nearly fully recovers the conflict effect (aggregate $R \in [0.92, 1.06]$). A definition-target swap shows the triplet performs binding rather than identity matching. Dissociation experiments isolate target preservation as the binding-specific signature: distractor suppression occurs under matched, swap, and item-mismatched conditions alike, whereas target logit collapse occurs only when the definition-target position is corrupted. Behavior and mechanism converge on the same channel: the lexical prior is where both interference originates and where override leaves its mark. 

---
# Liberating LLM Capabilities in Full-Duplex Speech Models 

**Authors**: Luoyuan Zhang, Bokai Xu, Junbo Cui, Weiyue Sun, Yingjing Xu, Hanyu Liu, Yuan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2606.07547)  

**Abstract**: Speech-based large language models are typically constrained to spoken replies, which limits their user-facing outputs to what can be verbalized and suppresses text-native capabilities such as code generation, structured analysis, and multi-step reasoning in realtime interaction, for tasks that require persistent, structured, and inspectable intermediate outputs. Existing work improves spoken reasoning or full-duplex turn-taking, but still treats text as a hidden intermediate state or a subordinate modality rather than a first-class output channel. We propose Listen-Write-Speak (LWS), a text-first tri-channel paradigm in which a single autoregressive LLM continuously listens to user audio, writes visible free-form text as its primary output, and speaks a realtime oral response in parallel under a shared causal attention context. This behavior is implemented entirely through a Token Schema, requiring no architectural modifications, and learned via a two-stage data pipeline that synthesizes per-second cognitive annotations consistent with the revealed input timeline. Empirically, LWS demonstrates strong full-duplex interaction on Full-Duplex-Bench, reaches 4.72 on VoiceBench AlpacaEval, achieves 92.6% writing-speaking consistency, and consistently outperforms its internal ablations on URO-Bench. These results suggest that visible writing can serve as a first-class output channel for speech interaction without sacrificing realtime responsiveness. The code and dataset are available on the project page: this https URL. 

---
# Finding Hidden Relationships Between Medical Concepts by Leveraging Metamap and Text Mining Techniques 

**Authors**: Weikang Yang, S M Mazharul Hoque Chowdhury, Wei Jin  

**Link**: [PDF](https://arxiv.org/pdf/2606.07540)  

**Abstract**: Text is one of the most common ways to store data in this computerized world. At a glance, it may seem that those data are not interconnected. But in reality, data can have hidden connections. Therefore, in this research, a new model has been presented that can find hidden relationships between two medical concepts by using MetaMap and appropriate text-mining techniques. Specifically, the model creates a new comprehensive index structure and can find cross-document hidden links connecting topics of interest that most existing approaches have ignored. Experiments show the effectiveness of the proposed model in discovering new connections between topics. 

---
# From Architecture to Output: Structural Origins of Hallucination in Large Language Models and the Amplifying Role of Data 

**Authors**: Md. Rejaul Korim Sadi, Toufiqur Rahman Tasin, Golam Mostofa Naeem  

**Link**: [PDF](https://arxiv.org/pdf/2606.07537)  

**Abstract**: Large language models hallucinate--producing fluent, confident, factually wrong outputs--with a consistency that persists across generations and scales. Existing taxonomies classify hallucination by output type, distinguishing intrinsic from extrinsic failures and faithfulness from factuality divergence. These frameworks are descriptively rigorous but do not identify which internal mechanism produced a given instance. This paper analyses hallucination as a structural consequence of three architectural decisions that together form a compound failure system. Self-attention's co-occurrence learning substitutes statistical proximity for semantic meaning and produces entity confusion, fact misattribution, and semantic drift. The maximum likelihood estimation training objective optimises next-token probability without factual constraint, rewarding statistically plausible outputs regardless of their truth value. Autoregressive decoding's permanent left-to-right commitment under exposure bias ensures that a single wrong token cascades forward through the entire output sequence without revision. Dataset pathologies--long-tail deficiencies, training bias, and synthetic pollution--amplify these vulnerabilities but do not independently cause them. We make three contributions. First, we map each mechanism to a specific output category in the Alansari and Luqman taxonomy, locating intrinsic hallucination in self-attention, extrinsic hallucination in MLE, and logical inconsistency in autoregressive decoding. Second, we show that each commonly cited dataset pathology exploits one of these mechanisms rather than originating hallucination independently. Third, we identify the diagnostic limitation of output-type-only classification and contrast it with inference-layer mitigation approaches. 

---
# Multilingual Refusal Alignment for Safer Large Language Models 

**Authors**: Aleksandra Krasnodębska, Wojciech Kusa, Aldo Lipani  

**Link**: [PDF](https://arxiv.org/pdf/2606.07535)  

**Abstract**: As Large Language Models (LLMs) are deployed globally, ensuring their safety and alignment across multiple languages becomes paramount. However, safety behaviors often vary unpredictably between languages, posing significant challenges for consistent and ethical AI. In this work, we systematically investigate the dynamics of multilingual alignment, exploring whether single-language alignment transfers cross-lingually, how language consistency is preserved during training, and the resulting trade-offs with general knowledge capabilities. We introduce RefusEU, a novel refusal alignment dataset covering 12 European languages, including a dedicated test set for evaluating current state-of-the-art models. Our controlled Direct Preference Optimization (DPO) experiments provide two key insights: aligning models exclusively in English is insufficient to ensure cross-lingual safety, even for the same harm categories, whereas training on multilingual datasets can improve safety without degrading general performance, as measured by the Global MMLU benchmark. 

---
# Bridging Traditional Explainability Methods and Multimodal Multilingual Models: An XAI-Based Analysis 

**Authors**: Paweł Pozorski, Jakub Muszyński, Maria Ganzha  

**Link**: [PDF](https://arxiv.org/pdf/2606.07533)  

**Abstract**: Multimodal Large Language Models (MLLMs) effectively integrate text and audio to interpret context in complex interactive dialogues. However, the internal mechanisms by which heterogeneous modalities influence model behavior remain opaque. While Shapley Values (SV) provide a robust, model-agnostic framework for local explainability in text-based NLP, their extension to multimodal data is hindered by cross-channel dependencies, intricate dialogue structures, and the prohibitive computational complexity of dense audio representations.
In this work, we formalize a multimodal extension of the Shapley Value framework, treating discrete text tokens and aligned audio segments as cooperative features. To ensure computational feasibility, we deploy a suite of efficient estimation strategies: exact SV computation for low-dimensional inputs and sampling-based approximations - including Monte Carlo permutations and stratified sampling with Neyman-optimal allocation - to minimize variance under constrained computational budgets. To resolve the granularity mismatch between modalities, we propose Spectrogram-Guided Phonetic Alignment (SGPA), a novel preprocessing method that maps high-frequency audio streams to interpretable, word-aligned segments.
Our contribution is twofold: first, we provide an open-source, model-agnostic Python package and a companion GUI for the computation and interactive visualization of multimodal attributions. Second, we evaluate our framework using curated subsets of the VoiceBench and Infinity Instruct datasets across diverse multilingual scenarios. Our experimental results reveal that input modality is a primary driver of attribution volatility and demonstrate that standard syntactic importance proxies often fail to predict model attention in multimodal, cross-lingual contexts. 

---
# Principled Agent Debate: Adversarial Arbitration for Sycophancy Reduction in Large Language Models 

**Authors**: Sam Ryan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07532)  

**Abstract**: RLHF-trained models are systematically biased toward agreement over accuracy, a structural property of the training process. We present Principled Agent Debate (PAD), a multi-agent architecture that mitigates identity-framed sycophancy by arbitrating between two models tuned to opposing philosophical dispositions, with a pragmatist synthesizer evaluating both arguments blind to their origins. This paper evaluates a prompt-based instantiation of PAD. The key mechanisms are static dispositional tuning, identity stripping before synthesis, single-round independent argumentation, and blind arbitration. We evaluate five instantiations on 200 stratified questions from SycophancyEval. All PAD variants (AnCifer, DeWin, FeynStein, BurGal, Trident) significantly outperform the single-model baseline (18.5%) and instructed-opposition baseline (29.0%), with DeWin achieving 48.5% accuracy (z=6.36, p<0.001 versus both). The variants are not significantly different from each other at n=200. The BurGal variant achieves 53.0% but functions as an architectural validity check; its consensus/heterodox axis structurally favors the heterodox model on every benchmark question. A pre-training floor affects an estimated 40% of questions; fine-tuned disposition models are the identified next step. 

---
# mllm-shap: A Shapley Value Explainability Platform for Text-Audio Multimodal Large Language Models 

**Authors**: Jakub Muszyński, Paweł Pozorski, Maria Ganzha  

**Link**: [PDF](https://arxiv.org/pdf/2606.07531)  

**Abstract**: We introduce mllm-shap, an open-source Python framework designed to extend Shapley Value (SV) explainability from text-only Large Language Models to Multimodal LLMs (MLLMs) processing joint text and audio inputs. While text-based attribution is well-studied, mllm-shap addresses three critical challenges unique to the multimodal regime:
(1) Modality-aware coalition masking, which manages the interleaved processing of discrete text tokens and dense audio encoder frames.
(2) Multi-turn conversation tracking, utilizing per-token metadata to maintain role and modality context.
(3) Phonetic alignment-based token grouping, a novel technique that reduces the coalition space by 10x to 50x, rendering SV estimation computationally feasible for long-form audio.
The platform implements five SV estimation strategies, including a Complementary Contributions (CC) estimator with Neyman-optimal allocation that demonstrates superior convergence over standard Monte Carlo baselines. mllm-shap is provided as a pip-installable package featuring an interactive web-based GUI for granular attribution visualization. To our knowledge, this is the first publicly available framework providing a complete, reproducible pipeline for SV-based explainability in text-audio MLLMs. 

---
# Finding New Connections between Concepts from Medline Database Incorporating Domain Knowledge 

**Authors**: Yang Weikang, Chowdhury S.M. Mazharul Hoque, Jin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.07530)  

**Abstract**: In this digital world, data is everything and significantly impacts our everyday lives. Interestingly, in this small world, everything is part of an ecosystem, where everything is connected, directly or indirectly. The same thing happens to data as well. In most cases, it may seem like a particular topic does not have any connection with another one, but in reality, they are connected through a mutually related topic. Therefore, in this research, we will discuss an adaptive model modified from the ABC model by Don R. Swanson, a Literature-Based Discovery (LBD) Model, to find the hidden connections between Concepts of Interest. The model demonstrates that two topics, A and C are different and have no relationship. But they have a common topic, B that can be used to connect topics A and C This famous model will be used in this discussion to connect Medical Concepts. 

---
# CAPruner: Conceptual-Adjacent Scene Graph Pruner for Enhancing 3D Spatial Reasoning of Large Language Models 

**Authors**: Shengli Zhou, Xiangchen Wang, Guanhua Chen, Feng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.07529)  

**Abstract**: Large language models (LLMs) have recently been applied to 3D vision-language (3D-VL) tasks, which require spatial reasoning to identify target objects relative to anchors. Scene graphs are commonly employed to represent such relations, but reasoning over complete graphs incurs high token costs and computational inefficiencies, motivating the need for pruning. Existing pruning methods primarily rely on spatial proximity and often remove task-relevant relations, thereby undermining reliable spatial reasoning. To address these limitations, we derive a key requirement for scene graph pruning: preserving spatial relations that are most pertinent to the specific 3D-VL task. Guided by this insight, we propose the Conceptual-Adjacent Scene Graph Pruner (CAPruner). CAPruner integrates fuzzy semantic relevance with spatial proximity to estimate the importance of relations, enabling the selection of critical relations in a task-specific context. Moreover, to avoid costly relation-level annotations, CAPruner is trained by supervising the aggregated scores of each node's incident edges. Extensive experiments demonstrate that CAPruner effectively preserves relations essential for spatial reasoning, leading to substantial performance improvements of LLMs on 3D-VL tasks. Code is available at this https URL. 

---
# BEACON: Behavioral Entropy Aggregation for Cross-Model Hallucination Detection in Large Language Models 

**Authors**: Naveen Bera, Pulijala Sai Nikhila, Kondaguduru Abhiram, Shaik Gayaz Ali, Shoaib Sadiq Salehmohamed, Shaik Mohammed Omar, Jinal Prashant Thakkar, Hansika Aredla, Shalmali Ayachit  

**Link**: [PDF](https://arxiv.org/pdf/2606.07528)  

**Abstract**: Hallucination in large language models (LLMs), defined as the generation of factually incorrect or unsupported content, remains a critical barrier to reliable deployment. We present BEACON (Behavioral Entropy Aggregation for Cross-model hallucination detectiON), a black-box hallucination detection framework that operates purely on model outputs without requiring access to internal representations or external knowledge bases. BEACON extracts a 31-dimensional feature vector from structured multi-pass generation, integrating NLI-based semantic entropy, embedding geometry, chain-of-thought consistency, and paraphrase stability signals. A gradient-boosted classifier trained on 7,617 labeled examples across seven benchmarks achieves 0.8123 +/- 0.0102 AUROC (95% CI: 0.7632-0.8251), outperforming standalone semantic entropy (+0.2298) and SelfCheckGPT-style consistency baselines (+0.2457). Feature importance analysis shows that hallucination is inherently multi-dimensional, requiring combined uncertainty signals. An efficient 5-call variant achieves 0.7795 AUROC, enabling practical deployment across black-box LLM APIs. 

---
# Post-training is (Massive) Supervised Learning 

**Authors**: Michael Hassid, Yossi Adi, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2606.07527)  

**Abstract**: The prevailing paradigm for training LLMs has evolved to rely on a massive post-training phase consisting of SFT and RL. In this position paper, we argue that this methodology effectively marks a reversion to the ``pre-train then fine-tune'' approach of the BERT era, explicitly tailoring models to the desired behaviors and specific benchmarks on which they are evaluated. We begin with a historical overview of LLMs, describing the different phases of the LLM evolution. We argue that the current landscape is remarkably similar to the early days of LLMs, where task performance heavily relied on fitting the models to in-distribution datasets. To empirically demonstrate this, we compare pre-trained models to randomly initialized ones, by fine-tuning both variants on modern reasoning datasets and evaluating them on competitive math and code benchmarks. We show that models post-trained from scratch yield highly non-trivial performance. Our findings suggest that current post-training methodologies function primarily as a distribution-fitting mechanism. We finish by positing that developing generally capable models and systems requires moving beyond extensive post-training for predefined behaviors, shifting instead toward training procedures where models ``learn how to learn''. 

---
# GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation 

**Authors**: Lin Mu, Guoji Wang, Li Ni, Lei Sang, Zhize Wu, Peiquan Jin, Yiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07526)  

**Abstract**: Large Language Models (LLMs) have shown strong potential for recommendation (LLMRec) due to their powerful reasoning and generalization abilities. However, effectively aligning the textual semantics modeled by LLMs with the collaborative signals remains a key challenge. Existing methods either translate collaborative information into textual prompts or inject pre-trained embeddings into the LLM, both of which treat structural information as static input and fail to capture high-order relational dependencies. To bridge this gap, we propose GraphLoRA, a novel framework that generalizes low-rank adaptation from independent to structure-aware propagation. GraphLoRA embeds a trainable graph message-passing network within the low-rank adaptation pathway, enabling structural signals to propagate through the parameter space. This design allows collaborative topology to explicitly guide parameter updates, fostering deep integration between graph-structured and textual semantic information. Extensive experiments on multiple benchmarks demonstrate that GraphLoRA not only outperforms state-of-the-art LLM-based recommendation methods but also achieves superior generalization, effectively balancing structural reasoning capability with computational efficiency. Code is available at \href{this https URL}{this https URL}. 

---
# Implicit Causal Graph Construction in Text via Chain Discovery 

**Authors**: Liesbeth Allein, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2606.07525)  

**Abstract**: Causal graphs in text are typically populated by observable, predefined events. In contrast, we study implicit causal graph construction from text by treating each described cause-effect pair as the begin- and endpoint of an underlying latent causal graph and using large language models (LLMs) to infer intermediate causal events. We compare end-to-end graph construction with methods that frame the task as causal chain discovery. In the latter, graphs are built either by aggregating inferred chains or by progressively expanding partial chains through an iterative search process. We further explore Wisdom of the Crowd extensions that access causal knowledge from multiple LLMs in post-hoc aggregation and collaborative inference settings. We analyze trade-offs among these approaches and evaluate the validity of inferred causal relations using a manually curated database of 1,560 scientifically validated causal pairs. This database-based evaluation is proposed as reliable, resource-efficient, and transferable to settings where ground-truth graphs are unavailable. 

---
# ABLE: Representing and Mapping LLMs via Attribution-Based Large-model Embedding 

**Authors**: Zirui Wang, Yusen Hou, Shaofeng Liang, Bowen Tian, Yanlin Zhang, Wenshuo Chen, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2606.07524)  

**Abstract**: The explosive growth of large language models (LLMs) has created a heterogeneous and poorly documented ecosystem, making systematic model comparison increasingly important for provenance auditing, security analysis, and model selection. Existing representation methods struggle to address this setting efficiently. Approaches analyzing internal parameters are powerful when architectures are compatible, but face scalability barriers under structural heterogeneity, while methods relying on external outputs may conflate models with similar behaviors and are difficult to align in richer output spaces across different tokenizers. To bridge this gap, we propose ABLE (Attribution-Based Large-model Embedding), a framework that leverages the interpretability space to construct model representations. By aggregating gradient-based feature attributions via a tokenizer-agnostic word-level alignment, ABLE captures model-specific input-sensitivity patterns rather than only surface-level outputs. Beyond empirical utility, we provide a stability analysis showing that, under standard regularity assumptions for differentiable Transformer-style models, ABLE induces a Lipschitz-continuous parameter-to-embedding map with finite-sample convergence guarantees. Extensive experiments on 239 open-source LLMs demonstrate that our training-free approach achieves competitive or superior performance in relation prediction, model routing, and benchmark score prediction. 

---
# Retrieval Augmented Generation Framework for the Nepali Legal Domain Question Answering 

**Authors**: Samir Wagle, Abiral Adhikari, Reewaj Khanal, Batsal Bhandari, Prashant Manandhar, Praveen Acharya, Bal Krishna Bal  

**Link**: [PDF](https://arxiv.org/pdf/2606.07523)  

**Abstract**: Legal domains in high-resource languages like English have widely adopted artificial intelligence for legal question answering. However, data scarcity in low resource languages such as Nepali has limited the training of large language models on Nepali legal texts. This study presents the first application of a Retrieval Augmented Generation based model for Nepali legal question answering using case laws extracted from the Nepal Kanun Patrika digital archive. Using BM25 on chunked documents, the approach achieved a top precision at one of 91 percent, and up to 75 percent with the multilingual E5 large model. Evaluation of generated answers showed 74 percent groundedness, 85 percent truthfulness according to an automated judge model, and 84 percent human evaluated truthfulness when using BM25 document retrieval, with a 92 percent successful answer generation rate. These results demonstrate that the RAG pipeline can effectively address the gap in legal question answering for low resource languages and provide a foundation for reliable AI systems in the Nepali legal domain. 

---
# Community-Specific Slang and Entity Detection via Semantic Shift in Fine-Tuned Language Models 

**Authors**: Julia Kruk, Sanchita Porwal, Amitrajit Bhattacharjee, Mansi Phute  

**Link**: [PDF](https://arxiv.org/pdf/2606.07522)  

**Abstract**: We propose an unsupervised method of resolving slang, unique entities, and folklore from online communities by isolating words in the lexicon that have the highest magnitude of semantic shift. Semantic shift is defined as the evolution of a word's encoded representation as a result of fine-tuning a pretrained Large Language Model (LLM) on a community-specific text corpus. This value is inversely proportional to the cosine similarity between the base model's encoded representation of a word, and a fine-tuned model's encoded representation. We fine-tune the DistilRoBERTa model on text corpora collected from 3 Reddit subreddits (r/Technology, r/Gaming, r/WorldofWarcraft), model a distribution of cosine similarity over the lexicon, and show that one can successfully resolve words that have unique significance to the community by pulling data in the bottom 10-percentile. In contrast, we show that data in the top 10-percentile consist of words that carry relatively universal semantics. 

---
# Evaluating Hallucinations in Domain-Adapted Large Language Models 

**Authors**: Sanchita Porwal, Sai Prasath S, Xingjian Bi, Madelyn Scandlen  

**Link**: [PDF](https://arxiv.org/pdf/2606.07521)  

**Abstract**: This study investigates the phenomenon of hallucinations in domain-adapted Large Language Models (LLMs), focusing on the fine-tuning of the Llama-2 model with the Lamini dataset. Hallucinations, or the generation of nonsensical or unfaithful content by LLMs, pose a significant challenge, especially when these models are fine-tuned with domain-specific data. Our methodology involves a series of experiments testing memorization, recall, and reasoning capabilities of the fine-tuned LLM, comparing its performance on novel question-answer pairs and domain-specific information. We found that while the model shows proficiency in tasks similar to its training data, its capability to accurately reason about and recall new domain-specific information remains limited, leading to instances of hallucination. The model demonstrates a tendency to provide correct answers with extra information, suggesting an inclination toward over-generation. These results suggest important limitations of fine-tuning-only approaches for mitigating hallucinations when adapting LLMs to specialized domains and underscore the need for more robust methods in adapting LLMs to specialized domains. The study also provides insights into the varying performance of LLMs on different types of information, revealing a comparative weakness in handling domain-specific queries. 

---
# TinyJudge: Unverifiable Constraint Alignment via Lightweight Specialist Ensembles 

**Authors**: Yirong Zeng, Yufei Liu, Xiao Ding, Yutai Hou, Yuxian Wang, Wu Ning, Haonan Song, Dandan Tu, Qixun Zhang, Yuxiang He, Bibo Cai, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07520)  

**Abstract**: Instruction Following (IF) is a core capability of LLMs, requiring strict adherence to diverse constraints, ranging from verifiable ones (e.g., output length) to unverifiable ones (e.g., tone). Reinforcement learning with verifiable rewards has emerged as a paradigm for IF tasks, leveraging LLM-as-a-judge to assess unverifiable constraints. However, we empirically find that this approach remains a significant bottleneck, suffering from severe reward hacking and higher computational overhead. In this work, we first analyze the generalization capabilities of unverifiable constraints and discover that specific constraints exhibit distinct, high-generalization patterns. Motivated by this, we propose TinyJudge, a framework that employs an ensemble of specialized tiny language models ($\sim0.6B$) to provide rewards for soft constraints. By distilling expertise from frontier models into these tiny models, it achieves high-precision, lightweight evaluation. Extensive evaluations across five benchmarks demonstrate that TinyJudge outperforms the baselines by $\sim10\%$ in average performance and $12\%$ in reward precision. Crucially, it also achieves a $3\times$ speedup in total training time. Our work provides a scalable and robust path for aligning LLMs with unverifiable human instructions. 

---
# Bidirectional Small-Granularity Search between Code and Text 

**Authors**: Marco A. Valenzuela-Escárcega, Enrique Noriega-Atala, Gus Hahn-Powell, Clayton T. Morrison, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07519)  

**Abstract**: We introduce the novel task of bidirectional small-granularity search between code and text, where the queries are small snippets of text or code and the results are also small fragments of the opposite modality, i.e., code or text. This task establishes direct links between text in scientific publications and corresponding code segments, in support of better and faster understanding of scientific methods. We introduce a large dataset for the proposed task that includes a training partition with textual descriptions of code generated automatically using GPT-4, and three testing partitions, one in-domain and two out-of-domain (OOD) that contain manually-annotated data as well as material from other domains. We also propose a modular approach to address this task. Our approach shares an encoder across four different subtasks that learn start/end of answer spans in both directions. We show that our method achieves good results in-domain, and encouraging results OOD. This suggests that addressing this task with automatically-generated data is possible, but there is exciting future work to be done. 

---
# SIGA: Self-Evolving Coding-Agent Adapters for Scientific Simulation 

**Authors**: Matthew Ho, Brian Liu, Jixuan Chen, Audrey Wang, Lianhui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2606.09774)  

**Abstract**: Advanced scientific simulators expose specialized input languages that turn simulation goals into executable configurations, but learning them can cost domain scientists hours to days. We study simulator setup as a problem of agent-tool interface grounding: what minimal simulator-specific adaptations are needed for an off-the-shelf coding agent to operate real scientific software? Our intuition is that coding agents already know how to navigate files, edit code, run commands, and repair outputs, but they lack the simulator's executable contract: its vocabulary, structural constraints, validation rules, and termination conditions. We introduce SIGA, a Simulator-Interface Grounding Adapter that supplies this contract through retrieval, procedural memory, in-trajectory validation, and validation-enforced termination. We primarily evaluate SIGA on GEOS, an open-source multiphysics simulator used in subsurface science. SIGA produces a complete GEOS deck in about five minutes with TreeSim above 0.90, matching an extended-budget human expert who took about three hours, a roughly 36x wall-clock speedup. On a harder held-out set, grounding raises TreeSim from 0.720 to 0.789, a roughly 10% relative gain over the bare agent, and can reduce the across-seed standard deviation by 16x. Self-evolution further improves SIGA by rewriting adapter contents from prior trajectories, yielding the highest held-out GEOS mean and matching or outperforming the strongest hand-designed configuration. Transfers to OpenFOAM and LAMMPS show that the dominant mechanism shifts by interface: validation matters most when structural completeness is the bottleneck, while memory and retrieval matter most when domain correctness is the bottleneck. These results suggest that lightweight, self-improvable grounding layers can turn general coding agents into practical operators of scientific software. 

---
# iOSWorld: A Benchmark for Personally Intelligent Phone Agents 

**Authors**: Lawrence Keunho Jang, Mareks Woodside, Geronimo Carom, Andrew Keunwoo Jang, Jing Yu Koh, Ruslan Salakhutdinov  

**Link**: [PDF](https://arxiv.org/pdf/2606.09764)  

**Abstract**: A useful phone agent needs to be personally intelligent. It should reason over a user's identity, history, and preferences as they exist on the device, not just follow isolated instructions in an impersonal sandbox. Existing mobile agent benchmarks lack this kind of personalization. We introduce iOSWorld, the first interactive native iOS simulator benchmark built around a persistent user identity spanning 26 newly built iOS apps. These apps contain connected data such as transactions, messages, travel records, social relationships, and financial activity. iOSWorld includes 133 tasks across three increasingly difficult categories. Single-app tasks (27) test one app, multi-app tasks (60) span 2 to 8 apps, and memory and personalization tasks (46) require agents to infer patterns from personal data. We evaluate frontier and open-source computer-use models in both vision-only and privileged vision+XML settings. The best configuration reaches 52\% overall but only 37\% on multi-app tasks. Privileged vision+XML access improves frontier models by up to 26 percentage points, while smaller models do not benefit from added accessibility-tree input. We release iOSWorld as an open-source benchmark with all apps, seeded data, tasks, rubrics, and evaluation code. 

---
# Collaborative Human-Agent Protocol (CHAP) 

**Authors**: Arsalan Shahid, Gordon Suttie, Philip Black  

**Link**: [PDF](https://arxiv.org/pdf/2606.09751)  

**Abstract**: Foundation models are moving from response generation into operational roles. They plan across steps, call tools, request human input, coordinate with other agents, and increasingly carry responsibility for work that affects customers, claims, code, contracts, and clinical decisions. Production deployments are no longer one human supervising one model. They are multi-human, multi-agent collaborations that cross teams, time zones, and trust boundaries. The technical surface for this collaboration remains weakly specified. When an agent drafts a response and a human edits it before it ships, the moment of human judgement is the most valuable signal in the system. In current practice it is recorded, if at all, in application code, chat threads, ticket comments, and tribal memory. Two protocol standards address adjacent concerns: MCP standardises agent access to tools and data, and A2A standardises agent-to-agent interoperability. Neither defines the shared workspace in which humans and agents perform accountable work together. This paper presents CHAP, the Collaborative Human-Agent Protocol. Under CHAP, the override that used to vanish into a chat thread becomes a structured event carrying a diff, a rationale, and a content hash. The handoff between shifts becomes a portable envelope rather than a pinned message. The human approval of an agent's draft becomes a non-repudiable signed decision that can be replayed years later. The protocol achieves this through a small Core (workspaces, participants, tasks, artefacts, and an append-only evidence log) together with composable profiles that add review, modes, routing, deliberation, handoff, identity, signatures, and transparency-backed audit as deployments require them. Specification, reference implementation, conformance suite, and worked examples are available at: this https URL 

---
# Multi-Turn Evaluation of Deep Research Agents Under Process-Level Feedback 

**Authors**: Rishabh Sabharwal, Hongru Wang, Amos Storkey, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2606.09748)  

**Abstract**: Existing benchmarks for deep research agents (DRAs) assess only single-shot outputs, ignoring a key question: can DRAs improve their reports when guided by feedback? To investigate this, we conduct a multi-turn evaluation of DRAs under two feedback settings: self-reflection, in which the agent revises its report without any external diagnostic signal, and process-level feedback, in which the agent receives guidance targeting gaps in its research strategy. To enable process-level feedback, we design Research Gap Inference (RGI), a method that analyzes patterns of satisfied and unsatisfied rubric criteria to infer research-process gaps. Our analysis reveals three key findings: (i) under self-reflection, agents incorporate and regress on rubric criteria at nearly equal rates, yielding negligible net improvement; (ii) a single round of process-level feedback yields substantial gains, raising the normalized score by approximately $8$-$15$ points and yielding a roughly $35$-$40\%$ incorporation rate; (iii) these gains do not compound over subsequent turns, as agents regress on up to $24\%$ of previously satisfied criteria when rewriting the full report to address remaining gaps. Even with targeted guidance, reliable multi-turn improvement remains out of reach for the DRA architectures we evaluate. Our code and results are publicly available at this https URL. 

---
# BrainSurgery: Reproducible and Reliable Declarative Weight Manipulations for Model Editing and Upcycling 

**Authors**: Gianluca Barmina, Annemette Broch Pirchert, Andrea Blasi Núñez, Lukas Galke Poech, Peter Schneider-Kamp  

**Link**: [PDF](https://arxiv.org/pdf/2606.09707)  

**Abstract**: As deep learning models scale, managing, inspecting, and modifying large checkpoints has become increasingly challenging. Researchers often need to alter model weights for layer restructuring, precision casting, low-rank factorization, and architectural debugging, yet these workflows often rely on fragile ad-hoc Python scripts. Here, we introduce BrainSurgery, a tool for robust and reproducible "tensor surgery" on neural network checkpoints, and provide a system demonstration covering four examples and three case studies from model upcycling to LoRA extraction. By abstracting storage formats and memory management, BrainSurgery executes complex transformations through declarative YAML plans. It supports structural modifications, mathematical transformations, and tensor reshaping through expressive regex and structural targeting, while built-in assertions validate tensor shapes, data types, and values to prevent silent errors. We envision that BrainSurgery will provide a strong foundation for future research through its reproducible and validated operations. 

---
# Correlation Is Not Enough: Embedding Human Metadata for Individual Causal Discovery 

**Authors**: Suraj Biswas, Saurabh Gupta, Pritam Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.09672)  

**Abstract**: Ask a pretrained biomedical language model whether "cortisol 28 ug/dL" and "stock-market volatility" are related, and it returns a cosine similarity of 0.83 on a scale where 1.0 means identical. The two share no mechanism. This is not a corner case: every off-the-shelf biomedical encoder we tested (BioBERT, PubMedBERT, BioM-ELECTRA) scores unrelated cross-domain pairs between 0.76 and 0.92 when the answer should be near zero. Accuracy on cross-domain discrimination is 0%.
Retrieval systems survive this, because a language model downstream filters the noise. A Large Behavioural Model (LBM), a foundation model whose subject is a person rather than a sentence, does not: it reasons over a graph of a user's life and treats embedding proximity as evidence that two events are causally linked. False proximity writes a false causal edge, and everything downstream inherits the error. Here, embedding geometry is not a tuning knob; it is correctness.
We report the fix. A contrastive pass over 72,034 pairs raises PubMedBERT BIOSSES correlation from 0.633 to 0.828 and within-vs-across-domain separation from 1.05x to 1.63x. A second pass, BODHI, mines hard negatives from edges absent in a biomedical knowledge graph and lifts separation to 2.30x and the discrimination gap to +0.392, at a 4.5% BIOSSES cost. On an Intel Xeon 6737P with AMX, OpenVINO cuts single-query latency from 1367 ms to 10 ms (133x) and reaches 555 sentences/sec. One finding contradicts standard advice: FP16 beats INT8 on this silicon at every serving batch size, and we explain why. The same model on a no-AMX Ice Lake instance runs 13-27x slower. We release the benchmark suite, training corpora, the BODHI generator, and the OpenVINO scripts. 

---
# SpatialWorld: Benchmarking Interactive Spatial Reasoning of Multimodal Agents in Real-World Tasks 

**Authors**: Hongcheng Gao, Hailong Qu, Jingyi Tang, Jiahao Wang, Zihao Huang, Hengkang Qiao, Shihong Huang, Junming Yang, Yi Li, Hongyixuan Yuan, Wenjie Li, Bohan Zeng, Wenbo Li, Bo Wang, Jianhui Liu, Olive Huang, Haoyang Huang, Wentao Zhang, Guoqing Huang, Nan Duan, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.09669)  

**Abstract**: Spatial reasoning is a foundational capability for multimodal large language models (MLLMs) to perceive and operate within the physical world. However, existing benchmarks predominantly rely on passive evaluation (e.g., static VQA) or simulator-specific pipelines, failing to assess general interactive spatial understanding. We introduce SpatialWorld, a unified benchmark designed specifically for evaluating the interactive spatial understanding of multimodal agents in complex real-world tasks. Integrating eight heterogeneous simulation backends under a shared, simulator-agnostic protocol, SpatialWorld features 760 human-annotated tasks across diverse domains (e.g., household routines, travel, social collaboration). Agents must solve tasks under vision-only partial observability, actively gathering egocentric visual evidence and expressing decisions via a unified, text-based action interface native to MLLMs. For reliable evaluation, each task includes a human-validated initial state, a reference trajectory, and a terminal-state verifier. Evaluating 15 advanced agents reveals that robust spatial task solving remains challenging: the strongest model, GPT-5, achieves an average task success rate (TSR) of only 17.4%, while the leading open-source model, Qwen-3.5, reaches 14.1%. Further analysis exposes a clear mismatch between task success and execution efficiency, alongside substantial domain-specific performance variations. These bottlenecks in active exploration and long-horizon planning position SpatialWorld as a rigorous testbed for future spatial agents. 

---
# Cross-Modal Masking for Robust Silent Speech Synthesis Using sEMG and Lipreading 

**Authors**: Eder del Blanco, David Gimeno-Gómez, Eva Navas, Carlos-D. Martínez-Hinarejos, Inma Hernáez  

**Link**: [PDF](https://arxiv.org/pdf/2606.09667)  

**Abstract**: Speech restoration through silent speech interfaces (SSIs) has emerged as a promising assistive technology for individuals with impaired or absent laryngeal voice production. Among non-invasive SSI modalities, surface electromyography (sEMG) and video-based lipreading provide complementary articulatory information, yet their integration for continuous speech synthesis remains underexplored. Moreover, existing multimodal approaches rarely address robustness to modality degradation or temporary sensor failure, limiting their applicability in realistic scenarios. In this work, we propose a masked multimodal speech synthesis framework that jointly leverages sEMG and lipreading signals through modality masking during training. Under multispeaker settings, the proposed approach reduces word error rate by up to 14 absolute percentage points compared to the strongest unimodal baseline. Experimental results not only show that masking strategies are critical for these performance gains and robustness under low-bitrate conditions, but also that they generalize better than degradation-specific data augmentations in the presence of modality absence conditions. Phone-level analyses further reveal complementary contributions across modalities, with particularly strong benefits for vowels and for specific consonant groups. Overall, these findings demonstrate the effectiveness and robustness of masked multimodal integration for silent speech synthesis, although adaptation to laryngectomized speakers remains an open research challenge. 

---
# TABVERSE: Benchmarking Cross-Format Table Understanding in LLMs and VLMs 

**Authors**: Momina Ahsan, Sarfraz Ahmad, Ming Shan Hee, Roy Ka-Wei Lee, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2606.09578)  

**Abstract**: Large Language Models (LLMs) and Vision-Language Models (VLMs) are increasingly evaluated on table reasoning tasks, but the role of table representation remains under-explored. In practice, the same table content may appear in different structural formats, such as HTML, Markdown, and LaTeX, or as rendered images. However, existing evaluations often let content, format, layout, and modality vary together, making it difficult to isolate representation effects. We introduce TABVERSE, a controlled multimodal table benchmark that aligns the same table content across multiple structural formats and rendered images, with question category and difficulty tags. This design enables systematic evaluation of representation effects while holding table content fixed. We evaluate LLMs and VLMs across three tasks: Question Answering (QA), Structural Understanding Capability (SUC), and Structure Reconstruction (SR). Our results show that representation choice substantially affects table understanding. Models generally perform better with structured text than with rendered images, but the size of this gap depends on the task, model, and format. HTML is often the most robust text format, while row-sensitive structural tasks and syntactically usable LaTeX reconstruction remain challenging. These findings show that table representation is a key factor in reliable table evaluation. 

---
# Interpretable Crisis Behavior Analysis Using Mobility and Social Media Data 

**Authors**: Muhammad Hamza Arshad Majeed, Sidahmed Benabderrahmane, Talal Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2606.09532)  

**Abstract**: Crises alter both how people move and how they communicate. During emergencies such as wildfires and pandemics, changes in mobility patterns and online emotional discourse evolve jointly, yet they are typically studied in isolation. This paper presents a unified and interpretable pipeline that integrates mobility and social media data to identify cross-domain behavioral patterns in crisis settings. The framework is evaluated through two case studies: a short-horizon analysis of the January 2025 Los Angeles wildfires (prototype case) and a longitudinal analysis of UAE COVID-19 behavior from March 2020 to December 2021 (primary case, 671 days). The pipeline aligns heterogeneous daily signals, transforms them into binary behavioral states, applies Formal Concept Analysis (FCA) to extract co-occurrence structure, mines association rules, and validates rule stability through chronological holdout testing. A structured policy-translation layer renders robust rules as operational briefs specifying triggers, lead times, and action playbooks. Results reveal clear cross-domain behavioral structure in both crises. In the wildfire case, traffic stress, fear/anger sentiment, and governance discourse are tightly coupled within a 33-day window, with key rules reaching 100\% confidence and lift scores up to 2.5. In the COVID case, repeated mobility adaptation and sentiment volatility yield 8 stable same-day rules (88\% holdout pass rate) and 40 clean predictive rules with 2--7 day lead horizons. The work demonstrates that interpretable multimodal fusion can produce both scientifically credible and policy-actionable crisis intelligence. 

---
# From Rigid to Dynamic: Entropy-Guided Adaptive Inference for Long-Context LLMs 

**Authors**: Zhanchao Xu, Haoyang Li, Qingfa Xiao, Fei Teng, Chen Jason Zhang, Lei Chen, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.09508)  

**Abstract**: Existing sparse attention and KV cache compression methods for long-context LLM inference typically apply fixed sparsity patterns or uniform budgets across all attention heads, overlooking the substantial variation in attention behavior among heads and contexts. We observe two distinct entropy patterns among attention heads: Rigid Heads, whose entropy stays near zero across input segments, and Dynamic Heads, whose entropy fluctuates significantly. Crucially, the distribution of these types is context-dependent and cannot be predetermined offline. We therefore propose EntropyInfer, a training-free framework that uses attention entropy to adaptively allocate compute at the granularity of individual heads and segments during prefilling. For decoding, we introduce a latent KV cache compression scheme that leverages generated output tokens, rather than prefill tokens alone, to identify and retain the most critical cache entries. Extensive experiments on Llama, Qwen and openPangu model series show that EntropyInfer consistently outperforms baselines including SnapKV, AdaKV, and CritiPrefill, achieving up to 2.39$\times$ end-to-end speedup beyond 100k tokens with minimal quality degradation compared to full attention. The code is released in this https URL. 

---
# Escaping the KL Agreement Trap in On-Policy Distillation 

**Authors**: Haoran Xin, Anhao Zhao, Ying Sun, Jin Li, Xiaoyu Shen, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2606.09471)  

**Abstract**: On-policy distillation (OPD) provides dense token-level supervision by asking a teacher to score student-generated rollouts. However, when the student drifts into an unrecoverable prefix, the teacher may locally agree with the degraded state, producing low reverse KL but little corrective training signal. We identify this persistent regime as a low-KL agreement trap. Further analyses show that tokens during and after such traps produce less useful supervision signals. We propose KAT (KL Agreement Trap Termination), an online OPD termination rule that detects persistent low-KL agreement with a dynamic training-adaptive threshold. By filtering weak supervision from degenerate agreement, KAT improves avg@k accuracy by 2.66% and pass@k by 3.43% across four mathematical benchmarks, while reducing average rollout length by 59.73%. 

---
# Capacity, Not Format: Rethinking Structured Reasoning Failures 

**Authors**: Hengxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2606.09410)  

**Abstract**: Prior work treats structured output as a reasoning tax, but this framing is incomplete: the cost of formatting depends strongly on a model's spare capacity. Using information-matched prose controls and a four-level schema complexity gradient, we separate format-specific effects from prompt-length confounds across 4 models and 5 benchmarks with 0% parse failures on successfully generated responses.
We find that structured formats are capacity-dependent. Models with sufficient headroom absorb JSON constraints without degradation (Sonnet: $88.7\pm4.0$% JSON vs. $89.3\pm1.7$% CoT on MATH-Hard). In contrast, formats severely degrade models operating near their limits through two distinct mechanisms. First, under standard token budgets, Haiku drops 36.2pp ($p < 0.0001$) largely due to truncation. Second, even with extended budgets eliminating truncation, GPT-4o-mini drops 28.0pp ($p < 0.001$), revealing pure capacity competition independent of token exhaustion.
This format penalty scales with schema complexity (McNemar $p < 0.0001$) and cannot be explained by prompt length alone. Furthermore, these results qualify claims of frontier model immunity: on AIME competition math, Opus 4.7 drops from 96.2% to 91.0% under JSON ($-5.3$pp; the displayed percentages are independently rounded, exact difference is $7/133 = 5.26$pp $\approx 5.3$pp). A delayed-structure ablation -- reasoning freely before formatting -- recovers most of the lost accuracy (3-run mean: 80--87%), supporting the capacity competition mechanism. The practical implication is not to avoid structured output, but to match it to capacity: when a model is near its limits, think first, format later. 

---
# Correct Looks Better: Pairwise Comparisons Reveal Accuracy Rankings 

**Authors**: Mina Remeli, Moritz Hardt  

**Link**: [PDF](https://arxiv.org/pdf/2606.09409)  

**Abstract**: Pairwise comparisons combined with aggregation methods like Elo have become central to evaluating generative models, yet concerns remain that they reward superficial stylistic cues or display judge biases. In a more positive turn, we show that model rankings from pairwise comparisons strongly agree with ground-truth-based accuracy rankings when such ground truth is available for comparison. By converting five well-known benchmarks into free-form generative evaluations, we find that Elo rankings achieve a Spearman correlation above 0.9 with accuracy rankings and substantially outperform direct evaluation when the judge is weak. Furthermore, style and judge bias have only minor effects on model rankings, despite most judgments occurring on pairs where both candidate answers are correct (or incorrect). On such pairs, we find that repetition after the final answer (echo) is a causal driver of judge preference. 

---
# Reasoning Arena: Trace Tournaments When Verifiable Rewards Fall Short 

**Authors**: Han Zhou, Adam X. Yang, Laurence Aitchison, Anna Korhonen, Albert Q. Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09380)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a leading paradigm for improving the reasoning ability of large language models through outcome-based supervision. However, verifiable rewards frequently become uninformative at the group level: when all sampled traces of a given prompt receive identical rewards, group-relative advantage estimation provides no gradient signal, even though the traces may differ substantially in reasoning quality. We propose Reasoning Arena, an adaptive training framework that routes such non-diverse reward groups to a judge system instead of discarding them. Beyond examining the final answer, Reasoning Arena constructs trace tournaments, where reasoning traces are compared head-to-head to expose finer-grained preferences within the group, converting reasoning quality into rich relative reward signals. To make reward estimation efficient, rather than exhaustively comparing every pair, each new trace is evaluated against a small, dynamically updated pool of previously generated traces as anchors to efficiently establish a relative ranking. We then fit a Bradley-Terry model on the incomplete comparison graph, enabling scalable RL integration without quadratic pairwise comparisons. Empirical results demonstrate that Reasoning Arena consistently outperforms the RLVR baseline by 7.6% on average in competition mathematics and coding benchmarks. By converting otherwise wasted zero-advantage samples into useful gradient updates, our method accelerates training by 27% to 41%, saving nearly 50% of generation compute, and substantially improves overall reasoning performance. 

---
# Experience Makes Skillful: Enabling Generalizable Medical Agent Reasoning via Self-Evolving Skill Memory 

**Authors**: Haoran Sun, Wenjie Li, Yujie Zhang, Zekai Lin, Fanrui Zhang, Kaitao Chen, Xingqi He, Yichen Li, Mianxin Liu, Lei Liu, Yankai Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09365)  

**Abstract**: Medical agent systems are increasingly expected to support interactive clinical decision making rather than only static question answering. In such settings, effective agents must reuse prior experience across evolving cases, yet existing memory mechanisms often retain raw historical traces that are redundant, noisy, and difficult to govern. More importantly, they rarely distinguish which memories are truly useful for future reasoning. This limits their ability to accumulate compact and reliable experience for long-horizon clinical reasoning. To close this gap, we propose SkeMex, a post-deployment self-evolution framework that improves medical agents through a skill-based memory without updating model weights. SkeMex distills informative interaction trajectories into structured skills that encode reusable procedural knowledge, and organizes them into a multi-branch repository spanning general, task-specific, and action-level experience. To determine which memories should be reused and retained, SkeMex estimates context-dependent utility from environment feedback and uses it to guide value-aware retrieval and repository governance. A closed-loop ``Read--Write--Assess--Govern" lifecycle further supports continual evolution by writing new skills, updating utilities, promoting useful memories, and removing harmful entries. Experiments across diverse clinical tasks show that SkeMex consistently outperforms representative memory-based agents in both offline and online settings. It also generalizes across model backbones and supports transferable skill memory. All data and code will be released publicly. 

---
# PBSD: Privileged Bayesian Self-Distillation for Long-Horizon Credit Assignment 

**Authors**: Yang Tian, Rui Wang, Xumeng Wen, Junjie Li, Shizhao Sun, Lei Song, Jiang Bian, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.09348)  

**Abstract**: Long-horizon agentic tasks pose a fundamental credit assignment challenge for outcome-base reinforcement learning: trajectory-level rewards verify final correctness but provide limited guidance on which intermediate reasoning steps or tool interactions contribute to the outcome. The difficulty is especially pronounced in multi-turn search agents, where successful trajectories may contain misleading actions and failed trajectories may contain valuable evidence-gathering steps. We propose PBSD (Privileged Bayesian Self-Distillation), a Bayes-calibrated self-distillation method for fine-grained credit assignment under sparse final rewards. PBSD measures trajectory quality through the posterior-to-prior probability ratio of the verified answer and applies Bayes' rule to convert this hard-to-estimate answer-side ratio into a tractable likelihood ratio between a standard student model and a privileged answer-conditioned teacher model. Autoregressive decomposition of this Bayesian evidence score yields turn-level signals that identify whether each intermediate turn supports or undermines the verified outcome. Consequently, PBSD provides a principled and elegant reweighting scheme that transforms sparse outcome supervision into Bayes-calibrated turn-level credit signals, while remaining fully compatible with standard policy optimization. Experiments demonstrate that PBSD consistently enhances performance across both in-domain and out-of-domain settings, and effectively transfers knowledge from short-context training to long-context inference, suggesting that its fine-grained credit assignment mechanism facilitates more effective policy learning and yields improved generalization. 

---
# The Injection Paradox: Brand-Level Suppression in Safety-Trained LLM Recommendations via RAG Context Injection 

**Authors**: Hyunseok Paeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.09204)  

**Abstract**: We present a reproducible failure mode of safety training in RAG-based LLM recommendation -- the Injection Paradox -- in which prompt injections embedded in retrieved documents backfire against the attacker, suppressing the target brand below the injection-free baseline. In safety-trained Claude models, documents containing prompt injections suffer a sharp drop in recommendation rate, and this suppression propagates beyond the injected document to unmodified documents of the same brand. In Claude Opus 4.6, the target brand drops from a 54% baseline to zero top-2 recommendations across all 50 trials, even though only 1 of 4 brand documents in the corpus contains an injection. The directional pattern is reproduced in counterfactual experiments and across three brands. A contrasting result across the GPT models tested, where the same injection instead increases recommendations, suggests model-family differences in how injection-like context affects recommendation behavior. These findings raise the technical possibility of a reverse-attack scenario in which an adversary embeds injections in a competitor's documents to suppress the competitor's brand via safety-sensitive model behavior. 

---
# Claw-R1: A Step-Level Data Middleware System for Agentic Reinforcement Learning 

**Authors**: Daoyu Wang, Mingyue Cheng, Qingchuan Li, Shuo Yu, Jie Ouyang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09138)  

**Abstract**: Agentic reinforcement learning (RL) has become an important post-training paradigm for turning LLMs from static chatbots into interactive agents, giving rise to representative applications such as OpenClaw. Existing work mainly focuses on policy optimization algorithms and training frameworks, but pays less attention to the full data lifecycle of agent-environment interactions, from data production to training consumption. To bridge this gap, we present Claw-R1, an interactive step-level data middleware system for agentic RL. Claw-R1 connects heterogeneous agent runtimes with RL training backends through two core components: a Gateway Server and a Data Pool. The Gateway Server captures multi-turn interaction steps through a unified LLM API entry point, while the Data Pool organizes them into step-level records consisting of prompt IDs, response IDs, rewards and other metadata. In our demo, users can interactively inspect live trajectories, examine the state, action, and reward of each step, curate data by quality and readiness, and configure training-ready batches for different downstream RL algorithms. Overall, Claw-R1 treats agent interaction traces as managed data assets rather than temporary runtime logs. Through this demonstration, we hope to encourage the community to recognize the importance of data management in agentic RL. Our code is available at this https URL and the demonstration video can be found at link this https URL. 

---
# From USD Scenes to Knowledge Graphs: Zero-Shot Ontology Grounding with LLMs 

**Authors**: Jiangtao Shuai, Zongxiong Chen, Manfred Hauswirth, Sonja Schimmler  

**Link**: [PDF](https://arxiv.org/pdf/2606.09134)  

**Abstract**: Constructing knowledge graphs from 3D simulation scenes is essential for robot task reasoning, but the key bottleneck, grounding scene objects to formal ontology classes, still relies on manually curated dictionaries that are brittle and do not generalize across assets. We investigate whether large language models (LLMs) can automate this grounding step for Universal Scene Description (USD) scenes as a zero-shot, training-free alternative. On a kitchen scene (125 objects) with SOMA-HOME Ontology, LLMs achieve 90-96% exact-match accuracy with descriptive names and 49-89% with abbreviated names, substantially outperforming dictionary and embedding baselines. Under fully opaque names, context-augmented prompting recovers up to 48%. Feature ablation reveals that LLMs primarily exploit semantic cues in the scene graph (sibling names and parent paths); anonymizing these cues reduces accuracy to 0-6%, while geometry alone yields only 4-17%. 

---
# Late-Layer Fusion is Enough: Dual-Path Vision Token Routing for Multimodal Large Language Models under Visual Saturation 

**Authors**: Siyuan Liu, Jinyang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09131)  

**Abstract**: Multimodal large language models (MLLMs) commonly inherit the deep, symmetric Transformer backbone designed for unimodal text modeling, and apply the same computation uniformly to image and language tokens. This design overlooks a key modality asymmetry: image and text tokens differ substantially in information density, redundancy, and required reasoning depth. Through a layer-wise analysis of LLaVA-1.5, we observe that vision tokens tend to saturate in the middle layers. Specifically, text-to-image attention decreases from 0.68 at layer 0 to 0.07 by layer 4, and stabilizes near 0.04 after layer 18, whereas text tokens continue to benefit from deep semantic processing. These findings suggest a mismatch between architectural symmetry and depth-asynchronous modality evolution, resulting in redundant visual computation and possible drift in perceptual representations during deep task-specific adaptation. Motivated by this, we propose Dual-Path Vision Token Routing (DPVR), a modality-asymmetric routing framework for efficient MLLMs. Its core instantiation, DPVR-LF (Late-Layer Fusion), routes vision tokens at the saturation point into a one-layer trainable side branch, runs a thirteen-layer text-only forward that skips image positions in the deep stack, and re-fuses the visual and textual streams only at the final layer. With approximately 3% trainable parameters, DPVR-LF preserves competitive multimodal performance on standard benchmarks while reducing visual computation in the deep Transformer stack. The results challenge the conventional assumption that vision tokens must traverse all deep language-model layers, and indicate that a single late fusion layer can be sufficient for maintaining strong perceptual competence in LLaVA-style MLLMs. 

---
# Beyond FLOPs: Benchmarking Real Inference Acceleration of LLM Pruning under a GEMM-Centric Taxonomy 

**Authors**: Haozhe Hu, Hao Wu, Anhao Zhao, Longwei Ding, Peiran Yin, Yunpu Ma, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.09080)  

**Abstract**: Pruning has emerged as a dominant paradigm for accelerating large language model (LLM) inference, spanning a broad spectrum of methods that remove computation across tokens, layers, heads, dimensions, and attention patterns. Despite sharing the same objective, these pruning approaches induce fundamentally different execution behaviors, causing realized speedups to depend heavily on hardware and kernel implementations. Consequently, the practical acceleration benefits of different pruning families remain poorly understood. In this work, we introduce a GEMM-centric taxonomy that reorganizes existing pruning methods according to the logical \textbf{M}, \textbf{N}, and \textbf{K} dimensions of general matrix multiplication (GEMM). Leveraging this abstraction, we build a unified benchmarking framework that enables implementation-consistent comparison across the pruning design space and systematically characterizes the acceleration--quality Pareto frontier. Our results show that static depth pruning remains the strongest Pareto-optimal baseline and stays closest to its theoretical acceleration upper bound in memory-bounded scenarios. During prefill, the frontier transitions from static depth at low quality loss (0\%--4\%), to dynamic depth at moderate loss (5\%--16\%), and finally to static width pruning at higher loss levels (17\%--26\%). These findings establish the first unified view of the practical limits of pruning-based LLM acceleration and provide guidance for future pruning research.\footnote{Code is available at this https URL} 

---
# A Unifying Lens on Reward Uncertainty in RLHF 

**Authors**: Ely Hahami, Yoel Zimmermann, Ray Zhou, Jack Benarroch Jedlicki  

**Link**: [PDF](https://arxiv.org/pdf/2606.09073)  

**Abstract**: Reinforcement learning from human feedback (RLHF) is bottlenecked by \emph{reward hacking}, where the policy exploits errors in a proxy reward model (RM) and produces high RM scores without genuine quality gains. A natural mitigation is \emph{pessimism}: penalizing rewards in regions where the RM is uncertain. However, standard scalar RMs provide no principled notion of uncertainty. We argue that the right object is a \emph{distributional} reward model $p(r\mid x,y)$. Under either a Bayesian inference or a KL-distributionally robust optimization (KL-DRO) lens, the KL-regularized RLHF objective admits a closed-form effective reward $\tilde r(x,y) = \pm\beta\log\mathbb{E}_p[e^{\pm r/\beta}]$. The pessimistic branch unifies the prior heuristics for RM ensemble aggregation: mean aggregation, worst-case optimization (WCO), and uncertainty-weighted optimization (UWO) all emerge as limits or truncations of this single expression. This also clarifies the implicit assumptions of each existing rule. 

---
# INFUSER: Influence-Guided Self-Evolution Improves Reasoning 

**Authors**: Siyu Chen, Miao Lu, Beining Wu, Heejune Sheen, Fengzhuo Zhang, Shuangning Li, Zhiyuan Li, Jose Blanchet, Tianhao Wang, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09052)  

**Abstract**: Self-evolution offers a scalable path to stronger reasoning: a pretrained language model improves itself with only minimal external supervision. Yet existing methods either depend on extensively curated or teacher-generated training data, or, when the generator runs unsupervised, reward it by a difficulty heuristic that need not improve the solver. We introduce INFUSER, an iterative co-training framework with two co-evolving roles: a Generator that drafts questions and reference golden answers from a pool of unstructured, automatically collected documents, and a Solver that improves by training on them. The solver is trained with standard correctness rewards against the generator-provided answers, while the generator is rewarded by an optimizer-aware influence score that measures whether each proposed question would actually improve the solver on the target distribution. Because this continuous, noisy influence score is poorly served by standard GRPO, we propose DuGRPO, a dual-normalized variant of GRPO, for generator training. Together, these turn the document pool into an adaptive curriculum that favors questions useful to the current solver, not just hard ones. On Qwen3-8B-Base, INFUSER outperforms strong self-evolution baselines with over 20% relative improvement on Olympiad and SuperGPQA benchmarks, and an 8B INFUSER co-evolving generator outperforms a frozen 32B thinking generator on math and coding. Ablations confirm each design choice is necessary, and two extensions, applying INFUSER to an instruction-finetuned anchor and augmenting it with rule-verifiable RLVR data, further demonstrate the flexibility and generalizability of the framework. Code is available at this https URL. 

---
# Decoy-Calibrated Failure Audits for Language Models 

**Authors**: Vyzantinos Repantis, Ameya Gawde, Harshvardhan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.09046)  

**Abstract**: Useful audits reveal not only how often a model fails, but also where its failures concentrate. An auditor may test many candidate explanations: long inputs, indirect questions, distracting evidence, or combinations of these factors. The risk is selection. The largest observed effect may reflect a real failure mode, or it may simply be the best result among many tried. We introduce Janus, a procedure for deciding when a proposed error explanation is credible enough to report. The goal is not to generate new explanations, but to decide which ones hold up. The auditor starts with a fixed model, a labeled evaluation set, and a frozen list of candidate explanations, which we call descriptors. Janus scores each descriptor by its error-rate lift, then compares real descriptors with fake ones that have the same frequencies but are randomly assigned to examples. A descriptor is confirmed only if it beats this decoy floor on the data used for discovery and then repeats on separate held-out data. In a controlled audit of multi-table lookup tasks, Janus identifies the planted failure, confirming long-chain descriptors and their interactions. The LLM often stops partway through the lookup chain instead of reaching the final answer. On two public benchmarks, MuSiQue and LongBench v2, the SliceLine baseline flags plausible high-error pockets, but Janus confirms none of them. Ablations show why both safeguards matter. On LongBench v2, an uncalibrated fixed threshold reports 20 descriptors, the decoy floor leaves one, and the holdout check rejects the last one after its lift shrinks from 0.36 to 0.05. The resulting principle separates proposing explanations from reporting them. Candidates may come from any source, but only those that beat decoys and replicate on fresh data become audit findings. 

---
# DynaCF: Mitigating Shortcut Learning in Reward Models via Dynamic Counterfactual Sensitivity 

**Authors**: Fengyuan Liu, Yongliang Miao, Zirui He, Yanguang Liu, Fei Sun, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2606.09043)  

**Abstract**: Reward models trained from pairwise preferences often exploit superficial shortcut cues rather than learning true response quality. We propose DynaCF, a dynamic reweighting framework for mitigating shortcut learning in reward model training. Unlike static shortcut heuristics, DynaCF measures shortcut sensitivity online during optimization by applying semantics-preserving counterfactual perturbations and tracking the resulting margin shifts and preference flips under the current model. Samples with higher shortcut sensitivity are dynamically downweighted in the Bradley-Terry objective, encouraging the model to rely less on superficial patterns and more on task-relevant preference signals. Extensive experiments show that DynaCF consistently improves robustness in preference modeling. 

---
# CRANE: Knowledge Editing for Reasoning MLLMs 

**Authors**: Han Huang, Hao Wang, Mengqi Zhang, Shu Wu, Qiang Liu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09033)  

**Abstract**: The emergence of reasoning multimodal large language models (MLLMs), which generate explicit chain-of-thought (CoT) reasoning before producing answers, has introduced a new challenge for knowledge editing: methods that appear successful under traditional metrics (teacher-forcing accuracy up to 100%) can fail severely when the model's reasoning process is examined (Grounded Success as low as 0%). We identify three failure modes: (1) Structural Collapse, where weight-modifying methods destroy the CoT format; (2) Cognitive Dissonance, where the model's reasoning chain actively rejects the injected edit fact based on visual evidence; and (3) Shallow Internalization, where methods succeed on exact queries but fail on rephrase or multi-hop variants. On reasoning MLLMs, these modes interact: methods that generalize (FT, LoRA) trigger format collapse, while methods without deep modification cannot generalize. To expose these failures, we propose a CoT-aware evaluation protocol and construct ReasonEdit-Bench, with conflict stratification, multi-level probes, and multi-hop portability tests.
We propose CRANE, a retrieval-augmented framework that requires no per-edit parameter modification. CRANE combines a modality-aware dual-library retrieval system with a two-phase training strategy: Supervised Fine-Tuning (SFT) for structural initialization, followed by GRPO with a Cognitive Routing Reward that trains the model to arbitrate between visual priors and injected edit facts. On ReasonEdit-Bench, CRANE achieves 96.9% Grounded Success on conflict scenarios and 96.9% intermediate entity usage in multi-hop chains, with 97.6% text-locality and 68.1% image-locality Edit Independence. On the out-of-distribution MMEVOKE benchmark, CRANE reaches 87.0% under gold retrieval. 

---
# TRIAGE: Dialectical Reasoning for Explainable Risk Prediction on Irregularly Sampled Medical Time Series with LLMs 

**Authors**: Hyeongwon Jang, Gyouk Chu, Changhun Kim, Joonhyung Park, Hangyul Yoon, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09030)  

**Abstract**: Clinical early warning systems built on electronic health records, in which clinical observations are recorded as irregularly sampled medical time series (ISMTS), must deliver both calibrated risk scores for patient triage and interpretable rationales that clinicians can verify. Large Language Models (LLMs) have been explored for this task, yet they collapse graded clinical risk into overconfident binary predictions. This risk polarization undermines both calibration and cross-patient comparability. To address this, we propose TRIAGE, a framework that trains an LLM to generate dialectical reasoning over competing clinical outcomes by eliciting outcome-specific rationales. This dialectical formulation mitigates risk polarization, enabling a single LLM to yield continuous risk scores grounded in explicit clinical reasoning. Evaluated on three ISMTS benchmarks, TRIAGE achieves an average AUPRC improvement of 3.3% and reduces calibration error by 81% compared to the competitive baselines. An LLM-as-a-judge assessment further shows that our rationales surpass post-hoc explanations from the baseline by 20% in clinical reasoning quality. The source code is available at this https URL . 

---
# Personal Salience: Highlighting Is Social, but Individuality Lives in Selection 

**Authors**: Kazuki Nakayashiki, Keisuke Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2606.09024)  

**Abstract**: Social highlighters let people mark passages that matter to them. We ask how much of an individual is recoverable from these naturalistic traces, using a co-readership identity control (the same document highlighted by many users) that holds document and topic fixed and asks whether a person's own history predicts their marks better than another reader's does. We separate generic salience (structure), crowd salience (what others marked), and personal salience (the individual residual). First, highlighting is social: which sentences you mark is predicted far better by the crowd than by structure or by a personal model, and even a well-estimated crowd, an information-privileged baseline that sees others' marks on the same document, beats a frontier LLM twin built from your other-document history; the within-document personal signal is at most a whisper (own-vs-other gap +0.017 by an embedding scorer, small but significant). Second, in sharp contrast, individuality lives in selection: asked which of the already-salient passages are yours, your own history is a strong, leakage-free predictor (gap +0.14). A topic decomposition shows this is largely stable thematic preference: it shrinks ~6-8x against a topically-matched peer, and a thin residual cannot be separated from finer topic. The non-obvious part is an asymmetry: under the same scorer the individual signal is ~6-8x weaker in salience than in selection. Methodologically, naive history-conditioning evaluations leak (the target's own marks enter the profile in ~42% of pairs, inflating personal scores by up to +0.15 AP) and small crowds overstate personalization; our results are leakage-free, use a dense crowd, and a model-matched control. Highlights carry a genuine individual signature, but a thin layer over a strong shared one, surfacing far more in which salient things a person selects than in what is salient. 

---
# Document-Authored Control-Signal Impersonation: A Low-Cost Indirect Prompt Attack on RAG Safety Boundaries 

**Authors**: Jianguo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09005)  

**Abstract**: Retrieval-augmented generation (RAG) systems often serialize user queries, retrieved documents, metadata, system labels, and task instructions into one natural-language prompt. We study a source-authority boundary failure in this design: attacker-authored retrieved text can impersonate metadata, provenance, authority, or disclosure-policy signals that appear control-relevant to the model. We call this pattern Document-Authored Control-Signal Impersonation (DACSI). DACSI is a non-imperative, metadata-like payload subclass within indirect prompt injection. Its central lesson is simple: document-authored labels are data, not policy. Command-style injection asks the model to ignore, override, or violate policy; DACSI asks whether untrusted document text can be misattributed as an authorized control signal when RAG prompt rendering collapses trusted and untrusted text into the same natural-language channel.
We evaluate DACSI across six model settings, prompt-pressure levels, injection baselines, signal taxonomies, RAG-mediated pipelines, system-control probes, a source-authority attribution probe, and synthetic canary formats. We interpret the evidence by model regime rather than as six equal replications: DeepSeek V4 Pro and Qwen3.5-397B provide the cleanest positive lift, DeepSeek V4 Flash is a high-susceptibility setting, GPT-5.5 and Gemini 3.1 Pro Low are strong-boundary probes with selected residual risks, and GLM-4.7 is a saturated leakage boundary case. Across these regimes, DACSI warrants separate evaluation because it uses a command-free metadata/provenance/policy surface, follows a RAG-specific source-authority path, and responds to source/channel separation. The source-authority probe is behavioral attribution evidence, not proof of an internal mechanism. 

---
# ChinaHeritaQA: A Culturally-Grounded Visual Question Answering Dataset for World Heritage Sites in China 

**Authors**: Yi Zhang, Bolei Ma, Yong Cao, Chengyan Wu, Daniel Hershcovich, Anna-Carolina Haensch  

**Link**: [PDF](https://arxiv.org/pdf/2606.08959)  

**Abstract**: We introduce ChinaHeritaQA, a multimodal benchmark dataset for evaluating the cultural reasoning abilities of vision-language models (VLMs) on UNESCO World Heritage sites in China. The dataset comprises 2,279 in-the-wild images paired with 14,133 bilingual (Chinese/English) multiple-choice QA pairs spanning seven cognitive dimensions, from basic identity recognition to historical periodization and architectural analysis. Guided by a UNESCO-aligned heritage ontology and verified through rigorous human annotation, the dataset ensures linguistic quality and factual consistency. Evaluations of state-of-the-art VLMs reveal that while top models exceed human performance on average, substantial task-level variation emerges: models excel at visual recognition but struggle with culturally grounded reasoning. Performance also varies by dynasty and region. ChinaHeritaQA reveals that strong visual retrieval does not extend to cultural and historical understanding. We release the dataset to support future research on culturally aware multimodal learning. 

---
# Are Reasoning Vision-Language Models Robust to Semantic Visual Distractions? 

**Authors**: Yizheng Sun, Mochuan Zhan, Yanan Ma, Jia Tong See, Yifan Wang, Ziyi Wang, Hao Li, Yang Cui, Wenhao Cai, Jingyu Sun, Chenghua Lin, Riza Batista-Navarro, Jingyuan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.08894)  

**Abstract**: Reasoning Vision-Language Models (VLMs) achieve strong performance on complex multimodal tasks, but reliable real-world application requires handling visual inputs that are messier than clean, curated benchmarks. Existing works mainly evaluate such reliability of VLMs through input corruptions, such as noise, blur and weather effects, which make visual evidence harder to perceive. This leaves a critical reliability failure mode underexplored: a model may perceive the evidence correctly, yet reason from plausible but irrelevant and distracting evidence and propagate this mistake to its final answer. To address this gap, we introduce \textbf{Distract-Bench}, a benchmark for evaluating VLM robustness to \textbf{semantic visual distractions}, defined as meaningful but task-irrelevant visual cues added to inputs while preserving the ground-truth answer. We comprehensively evaluate eight leading open-source and two closed-source VLMs across conventional vision corruptions and Distract-Bench. Our results show that Distract-Bench exposes a robustness failure distinct from vision corruptions: reasoning VLMs largely track their non-reasoning base models under perceptual degradation, but show consistently lower robustness to semantic distractions. Further analysis shows that these distractions often enter the reasoning process of VLMs, are treated as evidence, and lead to incorrect answers. Together, these findings reframe robustness evaluation for reasoning VLMs, shifting the focus from degraded perception to distractions for reliable real-world visual reasoning. Our data and code are available at this https URL. 

---
# sGPO: Trading Inference FLOPs for Training Efficiency in RLVR 

**Authors**: Shivchander Sudalairaj, Kai Xu, Akash Srivastava, Giorgio Giannone  

**Link**: [PDF](https://arxiv.org/pdf/2606.08854)  

**Abstract**: Standard Reinforcement Learning with Verifiable Rewards (RLVR) training allocates a fixed rollout budget to every query, without regard for what each query's difficulty means for the current policy. This leads to two symmetric failure modes: easy queries produce near-zero advantage because the policy already solves them, while unsolvable queries produce no signal because the policy never solves them. Both regimes waste training FLOPs without contributing to a learning gradient. We introduce sorted Group Policy Optimization (sGPO), a compute-efficient strategy that trades a small budget of inference FLOPs for a large reduction in wasted training FLOPs. The key insight is that cheap inference compute can serve as a single offline proxy for query difficulty. By generating a small batch of parallel samples per query under the initial policy, we obtain a model-aware empirical success rate. This motivates setting the training rollout group size to the inverse of this success rate, a practical rule that maximizes sample efficiency by extracting the most advantage per generated rollout. This single profiling pass simultaneously drives data filtering (removing trivial queries and sub-sampling unsolvable ones), adaptive group size allocation, and curriculum construction (scheduling queries from easy to hard). sGPO matches or exceeds baseline performance while reducing total training compute by a factor of three, with the upfront inference profiling cost included. 

---
# Intrinsic Selection and Particle Resampling for Inference-Time Scaling Beyond Domain Verifiability 

**Authors**: Giorgio Giannone, Mustafa Eyceoz, Shabana Baig, Shivchander Sudalairaj, Anna C. Doris, Faez Ahmed, Akash Srivastava, Kai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08850)  

**Abstract**: Inference-Time Scaling (ITS) has largely succeeded in verifiable domains like math and coding, where cheap verification enables scalable output selection. However, extending ITS to tasks prone to systematic failure - driven by faulty initial assumptions or unmet multidimensional constraints - typically relies on costly external solvers or brittle, model-based verifiers. Our key insight is that the intrinsic statistics of parallel sample sets, specifically length-adjusted tail entropy, provide a robust discriminative signal for solution quality without access to ground truth. Crucially, these statistics serve as a difficulty gate for adaptive compute allocation, dynamically routing problems across scaling regimes. First, Intrinsic Selection (iS) ranks candidates post-hoc, matching consensus-based algorithms across three domains and improving engineering design selection by 20% over pass@1 baselines. Second, Intrinsic Particle Filtering (iPF) generalizes this to step-level resampling, guiding generation toward high-confidence reasoning trajectories to improve pass@1 by 6.1 points on average on hard math problems. Finally, Particle Distillation (dPF) injects privileged guidance via early logit blending and KL-guided resampling, steering generation past systematic reasoning errors to satisfy expert rubrics, yielding up to 26.5% gains on complex clinical responses. Our pipeline applies seamlessly across broad-purpose, domain-specialized, and multimodal architectures, successfully extending ITS to open-ended domains without requiring trained reward models or exact ground-truth verification. 

---
# Momentum for Reasoning: Dense Intrinsic Signals in Policy Optimization 

**Authors**: Hao Chen, Zhanming Shen, Liyao Li, Yanyu Chen, Xuhang Zhu, Xiaomeng Hu, Qi Zhang, Ru Peng, Xiaoyu Shen, Haobo Wang, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.08815)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for eliciting long-chain reasoning in large language models. However, existing methods based on Group Relative Policy Optimization (GRPO) rely on a binary outcome reward, which induces two structural failure modes: Zero-Advantage Collapse, in which all rollouts in a group share the same outcome and the gradient vanishes, and Hallucinated Certainty, in which the model becomes increasingly confident on incorrect rollouts late in training. We address both modes by densifying the reward with intrinsic signals computed entirely from the policy's own conditional probabilities, and propose ISPO (Intrinsic Signal Policy Optimization, which combines a sequence-level signal measuring how informative the thinking trajectory is for the final answer, with a token-level directional reward whose hallucinated-certainty hinge penalizes confidently-wrong predictions at critical decision tokens. Across three base models and five mathematical reasoning benchmarks, ISPO consistently outperforms competitive baselines, with the largest gains on the hardest benchmarks where zero-advantage collapse is most frequent, and training-dynamics diagnostics confirm that both failure modes are decreased. 

---
# Artificial Intelligence for Mathematical Reasoning: An Integrated Survey of Language Models, Neuro-symbolic Systems, and Verified Discovery 

**Authors**: Syed Rifat Raiyan, Mohsinul Kabir, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08728)  

**Abstract**: Mathematical reasoning has long served as a stringent test of machine intelligence; over the past decade, it has moved from a niche problem within NLP to one of the most consequential AI frontiers. This survey provides a unified account of the field's evolution, from early rule-based math word problem (MWP) solvers and template-driven geometry systems, through neural expression generation and LLM prompting, to contemporary reasoning models, multi-agent systems, neuro-symbolic theorem provers, and verified discovery workflows. We organize the landscape along four axes: (i) informal reasoning over text and diagrams, spanning MWP solving, multimodal geometry, and VLMs; (ii) formal reasoning in proof assistants, including autoformalization, tactic prediction, compiler-guided repair, and proof search; (iii) mathematical discovery, where systems propose constructions, improve bounds, or assist attacks on open problems; and (iv) the inference and training-time techniques, including CoT prompting, tool use, process reward models, and RLVR, that increasingly connect generation with verification. We catalog major benchmarks across grade-school arithmetic, competition mathematics, geometry, formal proving, multimodal and multilingual reasoning, and expert evaluation, and we examine benchmark saturation, contamination, reporting mismatches, and the distinction between pass@1, majority voting, and verifier-assisted pass@$k$. We critically assess failure modes: brittleness under perturbation, reward hacking, multimodal grounding failures, fragile formalization, and the energy cost of reasoning-scale inference. Drawing on recent perspectives from working mathematicians, we identify future directions centered on verified-discovery workflows, reasoning efficiency, and infrastructure to make AI-assisted formalization broadly usable. Companion materials: this https URL. 

---
# Can LLMs understand LilyPond? A benchmark for symbolic music generation and understanding 

**Authors**: Matteo Spanio, Mohammad Torabi, Andrea Poltronieri, Antonio Rodà  

**Link**: [PDF](https://arxiv.org/pdf/2606.08722)  

**Abstract**: Symbolic music evaluation for large language models remains fragmented across representations, datasets, and metrics. We introduce LilyBench, a LilyPond-based benchmark that jointly evaluates symbolic music generation and music understanding on the same family of open-weight LLMs. The benchmark includes a 200-prompt generation suite and ten understanding tasks adapted from ABC-Eval, covering syntax, metadata prediction, structural sequencing, and music recognition. Generation quality is evaluated using compile rate, MusPy descriptor distributions via Jensen-Shannon similarity, and LilyBERT-based Fréchet Music Distance (FMD). Experiments on four open-weight models show that executable LilyPond generation is achievable in zero-shot settings, while structural understanding tasks remain challenging despite strong performance on composer and genre recognition. Our experiments also reveal systematic disagreements between descriptor-based and embedding-based metrics, suggesting that symbolic music evaluation benefits from metric triangulation rather than single-score ranking. We release the benchmark, prompt bank, and evaluation code to support future research in symbolic music generation and understanding at this https URL 

---
# Rank Intervals for Leaderboards: A Hierarchical Framework for Model Evaluation 

**Authors**: Bitya Neuhof, Yuval Benjamini  

**Link**: [PDF](https://arxiv.org/pdf/2606.08679)  

**Abstract**: Pretrained models are often evaluated on multi-task leaderboards to measure their applicability in diverse contexts. However, current methods for aggregating performance across tasks into leaderboard-level rankings do not address the uncertainty and variability at the task level. While recent works have proposed interval-based model rankings, the principled aggregation of uncertainty from individual tasks to leaderboard-level rankings remains unaddressed, and variation in models' performance across tasks is frequently obscured. In this work, we introduce a hierarchical framework that constructs model rank intervals with statistical guarantees at both levels: task-level rank confidence intervals from pairwise comparisons, and leaderboard-level rank prediction intervals using a conformal approach. This enables reliable quantification of model rank for each observed task and for new potential tasks. Experiments on simulated data and the TabArena and PromptEval (MMLU) benchmarks show that our method yields statistically valid and informative intervals, enabling reliable, uncertainty-aware model ranking on leaderboards. 

---
# Lost in the Flow with Code Talkers: Unveiling the Instruction-Tuning Tax of Large Language Models in Code Tasks 

**Authors**: Shi Ying Chang, Chiok Yew Ho, Yichen Li, Yintong Huo  

**Link**: [PDF](https://arxiv.org/pdf/2606.08676)  

**Abstract**: AI coding assistants have significantly improved developer productivity by automatically suggesting code that aligns with user intent, and many of these tools are now integrated directly into Integrated Development Environments (IDEs). Developers interact with code in two distinct cognitive modes: Flow and Command. While developers require tools that directly complete or infill code in unfinished programs during Flow mode, they also need tools that can comprehend intentions expressed as natural-language instructions and convert them into executable code in Command mode. Although instruction-tuned Large Language Models (LLMs) dominate many application scenarios due to their abilities to infer and fulfill developers' intents, it remains unclear whether the same paradigm is equally suitable for different code-related tasks. Therefore, it is necessary to understand how instruction tuning affects the feasibility of CodeLLMs as coding assistants. To fill this gap, we conduct the first empirical study that uncovers a key trade-off caused by instruction tuning across programming modes, which we term the Instruction-Tuning Tax. Our results show that instruction tuning is not a free lunch: although instruction-tuned models are more capable of following instructions and leveraging structured guidance, these gains often come at the cost of weaker infilling performance. We further extend our study through both qualitative and quantitative analyses, including manual failure categorization, behavioral metrics that capture generation fidelity, and intermediate-checkpoint evaluation throughout the tuning process. Summarizing our results into seven findings and four implications, our study offers a new perspective on the development of AI-powered coding tools and highlights the need to carefully balance instruction-following ability with effective code generation assistance. 

---
# Harnessing Streaming Video in the Wild 

**Authors**: Dingyu Yao, Shuhuan Gu, Qingyi Si, Junhao Zhou, Chenxu Yang, Chuanyu Qin, Naibin Gu, Zheng Lin, Weiping Wang, Nan Duan, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08615)  

**Abstract**: Vision-Language Models (VLMs) are increasingly required to process unbounded video streams in applications such as video-call assistants, live commentary, and embodied robots. An ideal streaming system should support proactive interaction, long-horizon memory, and real-time processing, while resting on a VLM backbone capable of handling diverse in-the-wild streaming tasks. However, existing VLMs excel at offline video understanding but fall short in streaming capabilities and lack dedicated infrastructure for streaming deployment. We address this gap on three fronts. (i) For backbone capability, we construct \textbf{Streaming-Train-248K}, a streaming dataset paired with a novel training objective for adapting VLMs to streaming interaction and understanding. (ii) For real-world deployment, we introduce \textbf{Streaming Harness}, a plug-and-play system that endows any VLM with three core abilities: proactive interaction (per-second response decisions), long-term memory (12-hour context retention), and real-time processing (sub-second latency). (iii) To drive continued community progress on streaming capabilities, we design \textbf{Streaming-Eval}, a benchmark that reflects models' capabilities across diverse in-the-wild scenarios. Extensive experiments demonstrate consistent gains from our approach across all core capabilities required for streaming video understanding. We will open-source our data, code, and benchmark to advance the community's shift from offline video understanding to deployable streaming intelligence. 

---
# Titans-as-a-Layer: Test-Time Memory for Conversational Speech Emotion Recognition 

**Authors**: Daniel Chen, Qicong Hu, Yang Xiao, Ting Dang, Hong Jia  

**Link**: [PDF](https://arxiv.org/pdf/2606.08573)  

**Abstract**: Speech emotion recognition (SER) is commonly formulated as utterance-level classification, although conversational emotion depends on a speaker's usual vocal range and the emotional context established by previous utterances. Speech-language models provide strong pretrained acoustic and semantic representations, and can adapts them to SER labels via finetune, but this mechanism still missing per-dialogue state. We study whether test-time neural memory can supply this missing context while leaving the large audio language models (LALMs) backbone intact. Building on Titans, we introduce a plug-and-play Memory-as-a-Layer (MAL) adapter that writes dialogue history into a small neural memory and reads it back as an audio-token-aligned residual update, avoiding changes to the host model's token positions. Across different audio LLMs and emotion recognition datasets evaluations, our design improves SER performs across different evaluation metrics, supporting test-time memory as a residual contextual mechanism for conversational SER. 

---
# Scaffold Effects on GAIA: A Controlled Comparison 

**Authors**: Jason Starace  

**Link**: [PDF](https://arxiv.org/pdf/2606.08529)  

**Abstract**: Published agent capability scores conflate what a model can do with what its scaffold lets it do, and the magnitude of this elicitation gap is not well characterized under controlled conditions. This study executes a pre-registered controlled comparison of three scaffolds (ReAct, a Planner-Actor-Rater multi-agent design, and planner-then-executor) across five models from three providers (Claude Opus 4.7, Sonnet 4.6, Haiku 4.5; Gemini 3.1 Pro Preview; GPT-5.5) on GAIA validation Levels 1 and 2, holding tasks and conditions fixed, with three attempts per question. Scaffold choice alone moves measured accuracy by as much as 28 percentage points within a single model (Opus, Level 2, robust slice), confirming the pre-registered hypothesis that scaffold variation produces gaps of at least 10 points. The pre-registered prediction that more capable models would be less scaffold-sensitive is rejected in direction: scaffold effects vary significantly by model in every dataset slice, but the most capable Anthropic model gains the most from structured scaffolds at the harder level, and tier-scaling holds only at Level 1 under the robust slice. The multi-agent advantage over ReAct at Level 2 appears within the Anthropic family but not for the cross-provider models, making model family rather than capability tier the conditioning variable, and the predicted planner-executor advantage on file-reading tasks is falsified. Structured scaffolds make fewer tool calls yet recover more often from mid-trajectory errors at the harder level, and a single cell (Gemini with planner-then-executor) is the cheapest at both levels and the most accurate at Level 2. These results indicate that single-scaffold capability numbers are scaffold-conditional estimates and that the elicitation gap is not guaranteed to shrink as models improve. 

---
# A Joint Finite-Sample Certificate for Adaptive Selective Conformal Risk Control 

**Authors**: Xiaoli Yu, Jiamiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08517)  

**Abstract**: Selective predictors answer on confident inputs and abstain elsewhere; deploying one safely needs a single finite-sample certificate that simultaneously upper-bounds the selected risk, lower-bounds the acceptance probability $\pacc$ above a floor $\pmin$, and lower-bounds the deployment utility. This certificate must be valid under adaptive threshold selection from a finite grid of $m$ pairs on $\ncert$ samples. We give such a certificate for bounded, possibly non-monotone losses by treating the selected risk directly as a ratio rather than through a Hoeffding-style range bound. The construction couples three confidence bounds: a variance-adaptive empirical-Bernstein bound on the ratio risk, a Clopper--Pearson bound on acceptance, and a two-sided closeness bound on utility. Together they lower-bound the certified policy's utility absolutely and to within $2\gammau$ of the best over the \emph{certified set}, both non-vacuous whenever feasible; a regime-scoped third leg matches an external oracle, informative only where the risk margin $\gammar < \alpha$ and vacuous at the headline operating points. Relative to the range-only Hoeffding-ratio construction this sharpens the acceptance-floor dependence from $1/\pmin$ to $1/\sqrt{\pmin}$, and a closed-form corollary identifies a per-pair regime in which our risk bound dominates a Hoeffding conformal risk control (Hoeffding--CRC) selective bound. Empirically, on ImageNet (three ResNets) and COCO val 2017 panoptic, the certificate opens a $+22$ pp certified-acceptance frontier over Hoeffding--CRC and is ${\approx}10{\times}$ tighter than a non-vacuous matched-valid baseline; these gains are regime-scoped, not universal, and absent on ADE20K. The certifier runs in $O(\ncert m)$ time. 

---
# Friend or Foe? Language as an ideological switch in open-weight LLMs under Russian disinformation stress 

**Authors**: Anna Małgorzata Kamińska, Tetiana Klynina  

**Link**: [PDF](https://arxiv.org/pdf/2606.08512)  

**Abstract**: As Russia's war against Ukraine extends into generative AI, large language models (LLMs) adapted for local post-Soviet languages are deployed in contested information environments. Policy and industry discourse assumes that culturally aligned adaptation encodes the political orientation of the target community: a Ukrainian-oriented model will resist Russian narratives, a Russian-oriented one will reinforce them. Does it? This article systematically disconfirms that assumption. We run a controlled audit of four openly available LLMs sharing a common base model but fine-tuned for different linguistic communities, querying them in Ukrainian, Russian and English across ten contested wartime narratives: Crimea, "denazification", the "one people" thesis, and atrocity denial at Bucha and Mariupol. The result is a Fine-Tuning Paradox: the Ukrainian-oriented model shows the weakest resistance to Russian disinformation in Russian, while the Russian-oriented one exhibits the strongest rejection. Corpus composition, language coverage and prompt format prove more decisive than nominal cultural provenance. We situate these findings within debates on hybrid warfare, digital sovereignty and post-imperial information orders, arguing that the principal threat to regional information sovereignty is not adversarial fine-tuning but the untested assumption that cultural alignment guarantees resilience. 

---
# Explaining Black-Box Language Models: Learning to Optimize Linguistically-Structured Word Subsets 

**Authors**: Minyoung Hwang, Seokhyun Lee, Changhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.08497)  

**Abstract**: As deep language models (DLMs) are increasingly deployed in high-stakes domains such as healthcare, understanding their decision rationale becomes paramount for ensuring trust, safety, and accountability. However, achieving this vital level of interpretability is particularly challenging when these DLMs operate as black-box systems (e.g., via APIs), where access to internal model states (e.g., parameters, gradients) is restricted. Despite numerous efforts, existing explanation methods often fail to concurrently satisfy three key desiderata: (i) inference-time efficiency, (ii) black-box compatibility without inducing out-of-distribution behavior, and (iii) comprehensible explanations grounded in the input's linguistic structure. To address these challenges, we propose a method that explains predictions of DLMs by selecting a small, informative subset of input words. We formulate this as an amortized optimization problem, enabling efficient one-shot inference without the need for input-specific search. Our selection policy is trained via REINFORCE-style policy gradients, allowing discrete word selection in a fully gradient-free setting. To enhance interpretability and align with human linguistic intuition, we integrate graph-structured knowledge into this selection process, fostering linguistically coherent subsets that result in explanations both highly informative and cognitively meaningful to end-users. We evaluated our method on diverse DLM architectures and multiple real-world datasets. It consistently identifies word subsets with enhanced discriminative power and stronger alignment with linguistically salient cues, outperforming both conventional black-box compatible methods and gradient-based approaches that are given oracle access to the black-box model's gradients for a more challenging benchmark. Our code is available at here. 

---
# Beyond Linear Activation Steering: Invertible Latent Transformations for Controlling LLM Behavior 

**Authors**: Tuc Nguyen, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2606.08454)  

**Abstract**: Activation steering provides a lightweight inference-time mechanism for controlling large language models (LLMs) by modifying their internal activation vectors toward desired behaviors. Most existing methods compute a fixed steering direction in the original activation space, typically from pairs of contrastive examples using mean differences, linear probes, or arbitrary separability criteria. While effective to a certain extent, these methods treat behavioral control as a global, linear, additive offset: the same direction is applied across inputs, and behaviors are linearly separable. This can be restrictive when behavioral features vary nonlinearly across the activation space or lie on curved and anisotropic manifolds, where the optimal intervention may be input-dependent. To address this limitation, we propose INNSteer, a nonlinear activation steering framework based on invertible latent transformations. Rather than searching for a better steering vector in the original representation space, INNSteer learns a lightweight invertible neural network $\phi$ that maps an LLM's activations into a latent space where behavioral classes are more amenable to linear control. At inference time, activations are mapped through $\phi$, steered in the latent space, and mapped back through the exact inverse transformation $\phi^{-1}$. This makes a simple latent-space translation become a nonlinear, input-dependent intervention in the original activation space. Across experiment settings on multiple LLM families, scales, behavioral traits, and safety benchmarks, INNSteer consistently improves model control over linear, transport-based, and nonlinear steering baselines while largely preserving generation fluency. 

---
# TinyGiantALM: A Compact Audio-Language Model for Intent-Aware Reasoning under Resource Constraints 

**Authors**: Vinh-Thuan Ly  

**Link**: [PDF](https://arxiv.org/pdf/2606.08425)  

**Abstract**: Current advancements in Audio Reasoning rely on massive Large Audio-Language Models (LALMs), hindering deployment in resource-constrained environments. We introduce TinyGiantALM, a compact 1.5B efficiency-oriented alternative. Instead of brute-force scaling, we propose an Instruction-Aware Feature Refinement framework using a Query-guided Projector and Semantic Gating to filter acoustic signals based on user intent. On the MMAR benchmark, TinyGiantALM achieves 46.4% zero-shot accuracy, significantly outperforming 7B-13B baselines. While a reasoning gap in logical narrative remains versus 30B+ models and certain trade-offs exist in overly dense or spatial scenes, our approach notably surpasses models up to 8x larger in disentangling mixed-modality environments. These findings demonstrate that architectural precision offers a tangible pathway to secure robust perception capabilities on edge-friendly scales. 

---
# Impacts of Histories and Models on LLM Grading: A Study in Advanced Software Engineering Courses 

**Authors**: Qilin Zhou, Zhuo Wang, Yue Li, W.K. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08400)  

**Abstract**: Graduate-level research reading report assessment creates a substantial labor burden for educators. While large language models (LLMs) hold great potential for automating academic grading, their reliability for this specialized task remains understudied, particularly regarding grading consistency, the lack of which represents a primary obstacle to educational fairness. This paper proposes a human-aligned LLM-assisted grading workflow and presents a case study based on 180 student submissions from a graduate advanced software engineering course. We evaluate two mainstream LLMs, Grok and GPT, in terms of grading consistency and alignment with human scores. We find LLMs exhibit distinct levels of intra-model consistency and significant inter-model grading inconsistencies, while simple ensemble approaches cannot improve alignment with human evaluation. Critically, continuous interaction history drives systematic drift in models' grading standards away from human expert scores. Our findings demonstrate LLMs' potential in reducing grading workload for educators in graduate education, while highlighting that indiscriminate LLM grading may introduce systemic unfairness, suggesting that specific operational practices are required to mitigate such disparities. 

---
# Strategic Type Spaces 

**Authors**: Olivier Gossner, Rafael Veiel  

**Link**: [PDF](https://arxiv.org/pdf/2606.08297)  

**Abstract**: We provide a strategic foundation for information: in any given game with incomplete information we define strategic quotients as information representations that are sufficient for players to compute best-responses to other players. We prove 1/ existence and essential uniqueness of a minimal strategic quotient called the Strategic Type Space (STS) in which a type is given by an interim correlated rationalizability hierarchy and represents a set of beliefs over other players' types and nature that rationalize this hierarchy and 2/ that the minimal STS has a recursive structure that is captured by a finite automaton. 

---
# When No Answer Is Correct: Diagnosing Absent Answer Detection for MLLMs in Video Understanding 

**Authors**: Yiheng Wang, Yueqian Lin, Lichen Zhu, Yudong Liu, Hai "Helen" Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.08239)  

**Abstract**: Multimodal large language models (MLLMs) have made substantial advancements in video understanding, yet the reliability of their responses remains underexplored. This work presents a diagnostic study of absent answer detection for MLLMs in video understanding, where the correct answer is deliberately excluded from the candidate set and a reliable model is expected to recognize that no valid option exists. We evaluate the absent answer detection behavior under three settings: multiple-choice questions augmented with an ``None of the Above'' option, open-ended generation with a detection instruction, and standard evaluation without any guidance. Across a diverse set of models and benchmarks, we find that MLLMs overwhelmingly select plausible distractors rather than detecting the absent answer. This failure is more pronounced in temporal reasoning tasks and worsens with denser frame sampling. We further explore chain-of-thought prompting as a mitigation strategy and find that while it substantially improves detection rates, performance remains unsatisfactory, suggesting that prompting-based strategies alone are insufficient to fully address this limitation. These findings expose a systematic failure in absent answer detection and highlight the need for explicit detection mechanisms in multimodal systems. 

---
# Paediatric-HGNN: A Hybrid Heterogeneous Graph Neural Network for Detecting Disfluency in Children's Speech via Multiscale Acoustic Fusion 

**Authors**: Rashini Liyanarachchi, Rachael Mackay, Alison Short, Aditya Joshi, Erik Meijering  

**Link**: [PDF](https://arxiv.org/pdf/2606.08210)  

**Abstract**: Automated stuttering detection (ASD) systems struggle with paediatric speech due to high acoustic variability in developing voices and the subtle distinction between pathological stuttering and typical developmental disfluencies. We introduce Paediatric-HGNN, a framework using a Context-aware Part-whole Interaction Network (CaPIN) tailored for paediatric data. Instead of conventional 1D signal modelling, our approach builds a heterogeneous graph capturing hierarchical relationships between lexical units (word nodes) and fine-grained acoustic segments (frame nodes). Trained on curated paediatric corpora (UCLASS and FluencyBank), Paediatric-HGNN achieves 82.4% weighted accuracy and a Typical Disfluency F1-score of 0.386. Modelling hierarchical lexical-acoustic interactions captures developmental "searching" behaviour, offering a more robust and interpretable tool for early clinical intervention. 

---
# CLASP: Language-Driven Robot Skill Selection and Composition using Task-Parameterized Learning 

**Authors**: Markus Knauer, Valentin Gieraths, Tai Mai, Samuel Bustamante, Alin Albu-Schäffer, Freek Stulp, João Silvério  

**Link**: [PDF](https://arxiv.org/pdf/2606.08169)  

**Abstract**: Enabling robots to understand and execute tasks from natural language commands while maintaining data efficiency remains challenging. Foundation models such as vision-language-action (VLA) and vision-language models (VLMs) provide intuitive interaction channels but require extensive data; task-parameterized imitation learning achieves data efficiency but lacks natural language grounding. This work bridges this gap through a modular architecture combining task-parameterized kernelized movement primitives (TP-KMPs) with pretrained VLMs. During learning, skills are acquired from 2 to 5 kinesthetic demonstrations, and the VLM generates skill schemas describing each skill's parameters and preconditions. During execution, the VLM interprets commands to select skills, reason about parameter bindings, and create novel behaviors through covariance-weighted composition. When no skill or composition suffices, the system identifies capability gaps and requests targeted demonstrations, all without fine-tuning. Validation on a 7-DoF manipulator shows success rates of 73.3%-100% in scenarios requiring skill selection, composition, and active learning. 

---
# ConSteer-RL: Steering Reasoning Capabilities in Large Language Models via Confidence-Aware Reinforcement Learning 

**Authors**: Qing Miao, Yiming Zhao, Jing Yang, Chenxi Liu, Yuehai Chen, Yuewen Liu, Shaoyi Du, Badong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.08088)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has recently become a key paradigm for improving the reasoning abilities of Large Language Models (LLMs), yet it remains limited by sparse binary rewards and its ignorance of model-internal uncertainty. In this paper, we propose ConSteer-RL, a simple yet effective framework that integrates token-level confidence signals derived from model log-probabilities into RLVR training. Specifically, building upon the Group Relative Policy Optimization (GRPO) framework, we construct a confidence-aware reward by aggregating per-token probabilities into a scalar confidence score and incorporating it into an awareness-based reward shaping mechanism that penalizes overconfident errors while reinforcing correct and confident reasoning. Experimental results demonstrate that ConSteer-RL consistently outperforms strong GRPO baselines, achieving average improvements of 2.3%-4.0% across different model scales. 

---
# Assessing the Energy and Carbon Emissions of Neural Speaker Verification Model in Training and Inference 

**Authors**: Hugo Leguillier, Driss Matrouf, Guillaume Lechien, Mickael Rouvier  

**Link**: [PDF](https://arxiv.org/pdf/2606.08087)  

**Abstract**: Deep-learning speaker verification (SV) increasingly relies on deep neural network backbones, whose environmental impact remains largely undocumented. In this paper, we conduct an evaluation of ResNet architectures trained on VoxCeleb2, varying depth, channel width, and stage distribution, and measure energy consumption and carbon footprint using node-level sensors. Results show a clear point of diminishing returns: deeper or wider models bring only marginal accuracy gains while energy consumption grows steeply. In contrast, mid-sized networks such as ResNet-50 and stage-concentrated variants achieve favorable trade-offs between performance and environmental impact. These findings provide actionable guidelines for designing energy-efficient SV systems. 

---
# On Low-Bit Quantization Errors in Speaker Verification: Diagnostic and Mitigation 

**Authors**: Hugo Leguillier, Driss Matrouf, Guillaume Lechien, Mickael Rouvier  

**Link**: [PDF](https://arxiv.org/pdf/2606.08078)  

**Abstract**: Although low-bit quantization provides practical means to deploy speaker verification on resource-constrained devices, its effects on speaker verification performance remain poorly understood. In this paper, we study uniform K-means quantization-aware training of ResNet-36 and ResNet-200 through joint layer-wise and score-level analyses. Our layer-wise analysis highlights fragile components and shows that score degradation is not fully explained by weight distortion alone. We identify a clear knee point at 2 bits, with larger score drift and harmful decision flips concentrated near the FP32 threshold. Our score-level analysis reveals where and how score errors emerge under extreme quantization. Building on these findings, we propose a calibrated multi-precision cascade that resolves most trials at 2 bits and escalates only ambiguous cases, achieving performance close to FP32 while preserving the efficiency benefits of low-bit inference with substantially lower compute and memory costs. 

---
# Robust-U1: Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding? 

**Authors**: Jiaqi Tang, Jianmin Chen, Youyang Zhai, Wei Wei, Runtao Liu, Mengjie Zhao, Xiangyu Wu, Qingfa Xiao, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.08063)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable success in visual understanding, yet their performance degrades significantly under real-world visual corruptions. While existing robustness enhancement approaches exist, they are limited: black-box feature alignment lacks interpretability, and white-box text-based reasoning cannot restore lost pixel-level details. This work investigates a fundamental research question: Can MLLMs recover corrupted visual content by themselves? To address this, we propose Robust-U1, a novel framework that equips MLLMs with explicit visual self-recovery capability for robust understanding. The approach comprises three core stages: supervised fine-tuning for initial reconstruction, reinforcement learning with dual rewards (pixel-level SSIM and semantic-level CLIP similarity) for aligning high visual quality, and multimodal reasoning that jointly considers both the corrupted input and the recovered image. Extensive experiments demonstrate that Robust-U1 achieves state-of-the-art robustness on the real-world corruption benchmark and maintains superior performance under adversarial corruptions on general VQA benchmarks. Analysis confirms that high-quality visual recovery directly enhances reasoning performance, establishing self-recovery as a critical mechanism for robust visual understanding. The source code is available at this https URL. 

---
# When Behavioral Safety Evaluation Fails: A Representation-Level Perspective 

**Authors**: Enyi Jiang, Anders Gjølbye, Yibo Jacky Zhang, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2606.08044)  

**Abstract**: Large Language Model (LLM) safety has often been evaluated at the behavior level, which provides limited evidence of internal robustness, as these evaluations target outputs rather than representation-level vulnerability under intervention. We formalize this discrepancy as the audit gap: the difference between behavioral safety and robustness under intervention. To study this gap, we construct dissociated models that preserve safe outward behavior while remaining vulnerable in the latent space. We introduce an intervention-based evaluation framework to test model robustness through soft interventions in parameter and latent spaces, including harmful fine-tuning and layer-wise latent perturbations. To formalize the evaluation, we propose the Latent Vulnerability Score (LVS) to measure how easily harmful behavior can be elicited by bounded latent perturbations. Using this evaluation framework, we show that behavioral safety metrics are insufficient measures of representation-level robustness across multiple safely and unsafely aligned state-of-the-art models. Notably, dissociated models show substantially elevated LVSs despite comparable refusal behavior under harmful intervention, with intermediate representations being the most sensitive to intervention. Our results suggest that behavioral safety evaluation alone provides an incomplete picture of model robustness, motivating representation-aware audits of latent vulnerability and observable behavior. 

---
# GIScholarBench: Benchmarking LLM Overconfidence in GIS Research 

**Authors**: Zongrng Li, Mingzheng Yang, Lei Zou, Hongxu Ma, Hao Tian, Siqi Zhou, Wenjing Gong, Kaili Zhang, Bingqian Chen, Mitch Zhang, Yifan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08036)  

**Abstract**: Large language models (LLMs) are increasingly used in academic research workflows, but scholarly tasks require high factual precision and therefore expose a key weakness: overconfidence. Here, overconfidence is defined behaviorally as the tendency to produce confident, assertive, and well-formatted outputs even when the underlying knowledge is incomplete or unverifiable, rather than as a calibration gap between stated confidence and accuracy. To examine this issue, we introduce GIScholarBench, a benchmark built from 10,865 papers published in 25 core GIScience journals between 2020 and 2025. The benchmark covers three tasks with increasing cognitive complexity: metadata retrieval, literature linking, and research direction generation. We evaluate Claude Sonnet 4.5, Gemini 3, and ChatGPT 5.3 through their native web interfaces under real-world user-facing conditions. Results show consistent overconfidence across all tasks. In metadata retrieval, ChatGPT 5.3 achieves the highest accuracy, but all models still generate definitive titles and DOIs when predictions are wrong. In literature linking, Claude Sonnet 4.5 recovers the most references, but all models show a clear gap between top-ranked retrieval and longer citation lists, suggesting that references are extended beyond reliable retrieval capacity. In research direction generation, AI-generated directions show lower topic coverage, higher novel miss rates, and lower semantic diversity than real future-citing papers. These findings suggest that LLM overconfidence is task-invariant but takes different forms: factual overgeneration in retrieval, unreliable citation expansion in literature linking, and overconfidence in output completeness during research ideation. 

---
# Sci-Rho: A Multilingual Visually-Grounded Symbolic Benchmark for STEM Problems 

**Authors**: Muhammad Falensi Azmi, Ikhlasul Akmal Hanif, Vallerie Alexandra Putra, Adi Yeltay, Abdullah Mubarak, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2606.08034)  

**Abstract**: Symbolic benchmarks have emerged as a key approach to assess model robustness under minor modifications to STEM-related questions. However, existing symbolic benchmarks mostly remain limited to mathematical reasoning, lack visual grounding, and are predominantly in English. In this work, we introduce Sci-Rho (Science Rhobustness), a dynamic benchmark for visually-grounded STEM problems spanning five subjects and seven languages, comprising 4,242 problem templates (606 per language) crafted by domain experts, including Olympiad medalists. Each template is implemented as executable Python code that generates diverse but equivalent problem instances by varying numerical values, visual patterns, geometric shapes, color schemes, and function types, resulting in 42,420 instances in total, each paired with reasoning steps and ground-truth solutions. We evaluated 17 state-of-the-art VLMs and discovered a noticeable gap between worst-case accuracy (defined as the proportion of problem templates that a model answers correctly across every generated variation) and average accuracy. We also discovered that smaller models show noticeable performance degradation across languages, whereas proprietary and larger models remain robust. Step-level evaluation reflects this same trend, revealing a significant gap between average F1 and worst-case F1 scores. Finally, our inspection of attention heads of a VLM reveals substantial cross-lingual variation in the relative attention allocated to image tokens compared to text tokens. Our work highlights the importance of evaluation beyond static benchmarks as a metric to measure the quality of VLMs. 

---
# IEA: Amateur-Friendly Conversational Image Editing Agent via Three Stages of Multitask Alignment 

**Authors**: Zichen Zhu, Yuheng Sun, Mingxuan Zhu, Wenjie Ma, Situo Zhang, Zhexiang Wang, Ziyue Yang, Danyang Zhang, Kunyao Lan, Zihan Zhao, Dingye Liu, Siqi Xiang, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08016)  

**Abstract**: Current image editing software often hinges on fixed filters or expert tuning, leaving a gap between amateur users' intent and outcomes. Creations by generative models may contain artifacts, implausible details, or stylistic drift away from photorealism and offer little insight into why an edit was made. We propose IEA, a conversational Image Editing Agent that learns to operate parameterized tools in an explicit, interpretable action space. IEA is trained via a three-stage multitask pipeline: (1) SFT on distilled expert edits, (2) GRPO with rewards for likeness improvement, tool usefulness, and intent summarization, and (3) large-scale synthetic fine-tuning to jointly master image editing, refinement, and user intent summarization. By manipulating 16 editing tools step by step, IEA produces transparent edit traces that can be inspected and debugged. In quantitative experiments, it attains a lower pixel distance on the edit task and a higher ROUGE-L on the summary task than strong baselines. In user studies, it ranks best among tool-calling methods for instruction following while surpassing generative methods in overall perceptual quality. Our results validate interpretable, tool-centric VLMs as a reliable path to human instruction-guided image retouching. 

---
# FMRFusion: Frequency-Aware Multi-View Representation Learning for Heterogeneous Image Fusion 

**Authors**: Tao Zhoua, Yunlong Liu, Qinghui Chen, Zekai Zhang, Minlong Sun, Changlin Biana, Dagang Li, Wenmin Wang, Jinglin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07985)  

**Abstract**: Infrared and visible image fusion aims to generate a composite image that retains significant target information and preserves detailed textures, integrating two heterogeneous modalities. Previous image fusion methods typically adopt a single-module stacking approach to extract features from the two modalities. However, these approaches may result in incomplete learning of their distinct characteristics, thereby limiting the fusion effectiveness and constrain ing robustness in real-world heterogeneous data scenarios. To address these challenges, we propose FMRFusion, a frequency-aware multi-view representation learning network for Heterogeneous Image Fusion. A Multi-Scale Struc tural Perception Module is introduced to effectively capture discriminative structures, extracting fine-grained local structures and essential contextual information. A bilinear frequency decomposition mechanism is employed to sepa rate features into high-frequency and low-frequency components, enabling joint modeling of local details and global representations across different frequency domains. Moreover, a Cross-View Complementary Interaction is incorpo rated to explicitly model and fuse the complementary characteristics between reflected light information and radiative intensity responses, facilitating effective cross-view interaction. We further improve the Performance of the fused results by flow matching, which progressively refines the fused features by learning the transformation from coarse data to high-quality representations. Extensive experiments conducted on multiple benchmark datasets demonstrate that FMRFusion achieves superior and consistent performance across a range of fusion tasks, especially in nighttime scenarios 

---
# Shared Latent Structures Enable Unified Backdoor Detection and Mitigation in LLMs 

**Authors**: Omar Mahmoud, Aly M. Kassem, Thommen George Karimpanal, Buddhika Laknath Semage, Negar Rostamzadeh, Golnoosh Farnadi, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2606.07963)  

**Abstract**: Backdoor attacks in large language models (LLMs) are often treated as isolated trigger-response failures, motivating defenses tailored to specific triggers or behaviors. We show this view is incomplete. Across diverse backdoor behaviors, we identify a shared latent mechanism that can be detected, causally controlled, and suppressed. Using sparse autoencoders (SAEs) on residual-stream activations, we find a small set of latent features consistently activated across jailbreaking, refusal manipulation, password-locking, bias induction, sentiment misclassification, and country-conditioned harmful advice. These features generalize across Qwen3, Gemma~3, and Llama~3.1 models from 4B to 32B parameters, and across both fine-tuning and weight-editing attacks. Through bidirectional activation steering, we show these features are causal: suppressing them reduces attack success, while amplifying them induces target behaviors on clean prompts. We further train lightweight SAE-feature classifiers that generalize zero-shot to unseen backdoors and outperform residual-stream and weight-diffing baselines. Finally, we introduce Concept Ablation Fine-Tuning (CAFT), which suppresses backdoor formation by ablating the shared latent subspace during training. Together, our results suggest that many backdoors rely on a transferable latent mechanism, enabling unified detection and mitigation. 

---
# POISE: Position-Aware Undetectable Skill Injection on LLM Agents 

**Authors**: Haochang Hao, Dehai Min, Zhifang Zhang, Yunbei Zhang, Miao Xu, Yingqiang Ge, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.07943)  

**Abstract**: Agent skills provide a lightweight mechanism for extending general-purpose agents, but their open format exposes them to skill-poisoning attacks. A practically dangerous injection must stay invisible: if executing the payload derails the user's legitimate task, the resulting failure signal invites inspection of the skill. We therefore evaluate attacks by Attack Success Rate, which requires the injected payload to execute and the user's task to still pass its verifier in the same trial. Prior skill-poisoning attacks face a reliability-stealth trade-off under this lens: YAML-header injections are reliably loaded but easily inspected, whereas stealthier body injections that place explicit malicious commands in the skill prose are less reliable because out-of-context commands invite the agent's own suspicion. We introduce POISE, a position-aware attack that compresses the trigger into a single, benign-looking body instruction, placing it at a feasible position and using a context-aware generator to blend it with nearby setup or prerequisite steps. On Skill-Inject with codex+gpt-5.2, POISE achieves an 89.3% ASR, 28.0 points above a random-placement body baseline and 2.6 points above a YAML-only baseline, while retaining the stealth advantage of body placement. That stealth is the decisive margin: because legitimate skill bodies naturally require privileged tool operations, LLM scanners are hyper-sensitive, falsely flagging 74.6% of clean skills on average across four judges and both benchmarks. Blending into these false alarms, POISE causes only 5.6% of poisoned variants to gain a new high-risk alert over their clean baselines, rendering current static defenses ineffective. 

---
# Decoupling Semantics and Logic: A Training-Free Coarse-to-Fine Pipeline for Video Retrieval-Augmented Generation 

**Authors**: Jiaxin Dai, Zehang Wei, Jiamin Yan, Xiang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07924)  

**Abstract**: This paper presents our system description for the 2nd Workshop on Multimodal Augmented Generation via MultimodAl Retrieval (MAGMaR). Addressing the critical challenges of cross-lingual long-video comprehension, strict persona adherence, and zero-hallucination temporal grounding, we propose a fully training-free, two-stage cascaded Video RAG pipeline. Our architecture strategically decouples semantic retrieval from cognitive logical reasoning through a modality-aware division of labor. In the first stage, a high-recall semantic pre-fetching module employs dense retrieval using only high-fidelity visual summaries and global text descriptions, explicitly isolating noisy modalities (e.g., OCR and ASR) to maintain a pristine vector space. In the second stage, an Adaptive, Iterative, and Reasoning-based (A.I.R.) filtering agent, powered by a commercial Large Language Model (LLM), performs fine-grained cognitive reranking. The agent re-incorporates full multimodal contexts to enforce strict logical alignment with user personas, effectively pruning semantically similar but logically irrelevant candidates. Finally, a Prompt Sculpting mechanism constrains the generator to synthesize the distilled subset into strictly formatted JSON responses with exact chunk-level citations. Evaluated on the RAG track, our resource-aware approach shows exceptional precision in both information retrieval and persona-conditioned generation. 

---
# MemToolAgent overview with a simple restaurant booking scenario where the agent retrieves similar memories, receives feedback on an invalid time format, and generates a reflection to update its memory 

**Authors**: Suleyman Armagan Er, Danilo Ribeiro, Yogesh Virkar, Surafel Lakew, Adi Kalyanpur, James Gung, Thomas Delteil, Arshit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2606.07909)  

**Abstract**: Modern large language model (LLM) agents can use external tools to help users solve complex tasks. However, for problems that require learning from long-term historical events or from previous agent-environment interactions, LLM agents are required to use memory mechanisms to store and retrieve experiences. While sophisticated memory systems exist for dialogue agents, few studies have empirically examined how to improve agents' tool-using capabilities through past user-agent conversations. We propose MemToolAgent, a framework that improves tool use through memory management. Our approach contains a memory extraction module that processes past experiences into structured memory entries, and a retrieval module that dynamically selects a subset of the stored memory entries. This enables more personalized and accurate responses aligned with user preferences and feedback without requiring LLM fine-tuning. In summary, this work has three main contributions: (1) a unified memory entry format that improves both general-purpose and personalized tool use without LLM fine-tuning, (2) a reflection-based memory extraction that uses environment and user feedback to distill wrong executions into critiques to store, and (3) a retrieval module that chooses how many past experiences to use based on the memory similarity distribution. MemToolAgent achieves 29%, 80%, and 17% relative improvements compared to strong baselines on the WorkBench, NESTFUL, and PEToolBench benchmarks, respectively. 

---
# Strained Coherence: A Pre-Failure Signal in Coding Agent Execution Trajectories 

**Authors**: Marut Pandya, Kasey Zhang, Baiqing Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07889)  

**Abstract**: LLM-based coding agents sometimes acknowledge a problem in their own reasoning and then proceed anyway. We call this pattern strained coherence: a safety-relevant failure mode in which an agent has information that should change its behavior, states that information, and still acts against it. The pattern overlaps with verbalized reward hacking, where an agent names a tension between a task proxy and the underlying goal yet optimizes the proxy anyway. We give an operational definition, build a Claude Sonnet 4.6 judge that reads full trajectories and flags spans where the pattern occurs, and evaluate it on 44 Terminal-bench-2 trajectories using a Qwen3.5-35B-A3B backbone. Flagged trajectories fail 94% of the time versus 46% for unflagged trajectories (47-point gap, Fisher's exact p = 0.003; 46 points after excluding three prompt-embedded examples, p = 0.006). At matched selectivity, the detector reaches 94% precision versus 88% for a lexical discourse-marker baseline; the 10-trajectory intersection of the two methods has a 100% failure rate (Clopper-Pearson 95% CI [69%, 100%]). We replicate on Gemma4-31B with 43 trajectories: the overall signal is directionally consistent but not significant (20-point gap, p = 0.31), with attenuation driven largely by 13 trajectories with zero think content, where the detector has no substrate to analyze. In the high-verbosity Gemma tertile, the gap is +30 points; in the mid- and high-verbosity Qwen tertiles, it is +40 points each. The first flag appears at a median of 83-84% of elapsed trajectory time across both models, and the binary flag survives paraphrases that soften explicit conflict markers (8/8 trajectories). Unlike univariate predictors, the detector emits interpretable span-level output -- quoted acknowledgment, quoted action, and typed conflict -- showing what the agent saw and ignored. 

---
# Cherry-pick Override: Unsafe Directional Commitment in LLM Judges under Mixed Evidence 

**Authors**: Haoran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07834)  

**Abstract**: LLM judges increasingly turn verdicts into system commitments. Under mixed evidence (claims with both supporting and refuting sources) this is unsafe: when the schema exposes CONFLICTING as the authorized non-directional verdict, returning SUPPORTS/REFUTES is an unauthorized directional commitment, a failure we name Cherry-pick Override (CCO). We define CCO under an explicit task contract and report it with a same-denominator diagnostic protocol paired with matched-coverage bootstrap and an apples-to-apples random-veto null. On AVeriTeC's Conflicting subset (N_C = 150), three-option judges return a directional verdict on more than 84% of mixed-evidence claims; under the typed schema, three-judge majority voting amplifies direction-on-conflict on AVeriTeC (0.887 vs. 0.840; 95% CI [+0.013, +0.080]) but does not replicate on VitaminC-Mixed. Walking an intervention ladder of common single-channel fixes (typed vocabulary, panel aggregation, confidence thresholding, validator-only filtering), each leaves a distinct residual failure: panel aggregation suppresses single-judge CONFLICTING dissent in 48% of CCO cases; the panel is well-calibrated for direction (ECE = 0.07 on pure-S/R) so confidence cannot operationally separate CCO from correct directional commits; validator-as-classifier nearly halves pure-evidence accuracy. A minimal two-channel reference probe reaches operating points neither single channel reaches; under the random-veto null its promotion to CONFLICTING is structurally targeted on AVeriTeC (empirical p < 1/2001) and weaker but in the same direction on VitaminC-Mixed, a selectivity result rather than a magnitude one. We argue for an external commitment-control layer that separates verdict generation from commitment authorization, using structural evidence and confidence as orthogonal channels and NO-COMMIT as a routed controller state. 

---
# Scaling Participation in Modular AI Systems 

**Authors**: Shangbin Feng, Yike Wang, Weijia Shi, Luke Zettlemoyer, Yejin Choi, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2606.07812)  

**Abstract**: Humanity is a mosaic of multifaceted talents and needs, and any truly intelligent AI must reflect that richness. Yet the LLMs used by all are built by the few -- a centralized market of monolithic AI models structurally ill-suited to capture the diversity of human knowledge, reasoning, and values. Here we introduce scaling participation, a new paradigm in which modular AI systems are built from the bottom up through the contributions of diverse stakeholders. Participants contribute small models trained on their own interests and priorities; these models then collaborate in modular frameworks as compositional AI systems. Participatory AI systems outperform monolithic LLMs by up to 15.4% across 15 tasks, such as reasoning and factuality, surpassing models larger than all contributed components combined. Further experiments show that participatory AI systems benefit from contributor diversity, substantially improve on each contributor's original priorities, and exhibit emergent capabilities that allow them to solve over 15% of problems where all individual models fail. Scaling participation provides a technical foundation for transitioning from the monolithic status quo toward an open, bottom-up, and collaborative AI future. 

---
# Benchmarking Quantum Algorithmic Resilience for CVaR Portfolio Optimization: The Expressibility-Coherence Trade-off 

**Authors**: Prashik N. Somkuwar, K. Srinivasan, G. Raghavan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07727)  

**Abstract**: Quantum combinatorial optimization offers theoretical advantages for complex financial modeling, but physical implementation on Noisy Intermediate Scale Quantum (NISQ) devices is severely constrained by hardware topology. This study presents a hardware benchmarking analysis between a Hardware Efficient Variational Quantum Neural Network (HE-VQNN) and the Warm Start Quantum Approximate Optimization Algorithm (WS-QAOA) for a hybrid Mean Variance and Conditional Value at Risk (CVaR) portfolio objective. By implementing a novel classical quantum hybrid proxy matrix to bypass the CVaR auxiliary qubit bottleneck, we map up to 16 assets from the NIFTY 50 index onto an IBM heavy hex processor. We systematically quantify algorithmic resilience to the "SWAP tax" incurred during routing. Empirical results reveal a critical operational trade-off: WS-QAOA provides exact theoretical mapping but suffers catastrophic hardware decoherence due to exponential nonlocal gate overhead. Conversely, HE-VQNN preserves hardware coherence but lacks the mathematical expressibility to capture dense tail risk asset correlations. This study exposes the limitations of dense financial optimization on current architectures forces an nonviable choice between algorithmic inexpressibility and hardware decoherence. This is indicative of a deeper limitation as to what can and cannot be done with NISQ computers lacking in all-to-all connectivity. 

---
# Why Limit the Residual Stream to Layers and Not Tokens? Persistent Memory for Continuous Latent Reasoning 

**Authors**: Mujtaba Farhan, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2606.07720)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable reasoning abilities on mathematical and multi-hop planning tasks. The CoCoNuT (Chain of Continuous Thought) paradigm~\cite{hao2024coconut} extends this by enabling models to reason in latent space, exploring multiple reasoning paths simultaneously rather than committing to a single chain early on. However, we identify a limitation we term the \textbf{concept bottleneck}. At each reasoning pass, intermediate hidden states are overwritten, causing the model to lose critical facts computed in earlier steps as reasoning depth increases. We observe this empirically. On HotpotQA, vanilla CoCoNuT (10.4\% EM) fails to improve over the CoT baseline (11.0\% EM), and performance degrades with curriculum depth on GSM8K. To address this, we propose \textbf{AGCLR} (Adaptive Gated Continuous Latent Reasoning), which augments CoCoNuT with a \textit{Gated Concept Stream}. A persistent residual memory maintained across all reasoning passes, controlled by three learned gates: a \textit{write} gate that commits intermediate facts to memory, a \textit{read} gate that retrieves relevant prior states, and a \textit{forget} gate that prunes irrelevant context. Evaluated on GSM8K, HotpotQA, and ProsQA using GPT-2 as our base model, AGCLR achieves consistent improvements across all types of datasets. With the performance gap compounding as curriculum depth increases, directly resolving the concept bottleneck. Code available at this https URL 

---
# How Much Dense Attention is Necessary? Oracle-Guided Sparse Prefill for Full/GQA Layers in Hybrid Long-Context Models 

**Authors**: Hongxing Wang, Harenome Razanajato, Zhen Zhang, Yujie Yuan, Hongsheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07703)  

**Abstract**: Long-context prefill remains expensive because full/GQA layers still score the historical sequence, even in hybrid models with local, sparse, linear, or recurrent components. We study how much dense attention is needed to preserve task-level behavior under explicit support granularity and top-k budgets.
We introduce an attention-mass top-k oracle for existing GQA checkpoints: for each layer and query position, it computes dense attention, selects head-averaged token support, and recomputes attention only on that support. The oracle is a diagnostic reference, not a deployable accelerator, and separates sparse-budget feasibility from indexer error and runtime realization effects.
On Qwen-family retrieval-heavy evaluations, the longest per-query oracle rows stay within 1 point of dense, and a Qwen3.5-9B RULER-style sweep from 4K to 100K stays within 0.48 points. Guided by the oracle, we derive a head-collapsed auxiliary indexer trained by KL distillation from dense attention-mass distributions while keeping the backbone frozen. With separately distilled Qwen3.5-0.8B and Qwen3.5-9B indexers, the reported 16K/32K validation macro gaps are +2.04 and +1.13 points, treated as quality preservation rather than improvement; fused selection-block-shared support can introduce a larger realization gap.
Preliminary single-card TTFT measurements show distilled-indexer sparse serving speedups of 1.71x for Qwen3.5-0.8B on NPU and 1.93x for Qwen3.5-9B on GPU against its dense FlashAttention-2 baseline. Additional random-init stress rows reach 3.44x, indicating sparse-runtime headroom but not validated output quality. This first release separates oracle feasibility, distilled-indexer quality, and runtime headroom, leaving a fully matched quality-latency frontier to future work. 

---
# TRACER: Token ReAssignment for Concept ERasure in Generative Recommendation 

**Authors**: Ziheng Chen, Jiali Cheng, Zezhong Fan, Hadi Amiri, Diyuan Wu, Gabriele Tolomei, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07688)  

**Abstract**: Generative recommendation formulates next-item prediction as autoregressive generation over semantic ID (SID) sequences derived from users' historical interactions, making modern recommender systems structurally similar to large language models (LLMs). As privacy and safety concerns grow, these systems increasingly require concept unlearning to remove sensitive or harmful concepts associated with items. However, existing LLM unlearning methods cannot be directly applied to generative recommendation. Unlike word tokens with explicit semantics, SIDs are abstract identifiers that are often shared by both forget and retain items, leading to severe conflicts between concept removal and recommendation utility preservation.
To address this challenge, we propose TRACER, an end-to-end concept unlearning framework based on token reassignment. Rather than directly suppressing shared SIDs, TRACER reassigns concept-related items to alternative tokens that better facilitate forgetting while minimizing side effects on retained items. We further introduce a coherence regularizer to preserve semantic consistency among retain items during unlearning. Experiments on real-world recommendation datasets demonstrate that TRACER effectively removes target concepts while substantially better preserving recommendation utility than existing unlearning baselines. 

---
# Steer Where It Matters: Token-Level Visual-Sensitivity Steering for LVLMs Hallucination Mitigation 

**Authors**: Ruipeng Zhang, Zhihao Li, C. L. Philip Chen, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07647)  

**Abstract**: Large vision language models (LVLMs) have made rapid advancements and are deployed across various applications, yet hallucinations remain a major challenge. Activation steering is appealing due to its minimal training overhead and controllability at inference time. However, we found that during autoregressive decoding, visual conditioning affects token prediction sparsely and locally across decoding steps, and many existing methods that average image-versus-no-image differences over the entire sequence dilute these critical signals, yielding low signal-to-noise ratio steering directions. Additionally, many existing methods apply a fixed steering strength, which misallocates the intervention budget, over-perturbs non-critical tokens, and can cause instability. To address these limitations, we propose Token-Level Visual-Sensitivity Steering (TLVS) for hallucination mitigation. Our approach first extracts token-level steering vectors and refines them, and then applies fine-grained, visual-sensitivity-adaptive steering only where it matters. This lightweight, plug-and-play mechanism requires only minimal training for calibration and can be applied across diverse vision-language models. It modulates the steering strength at each decoding step, selectively suppressing hallucination-prone spans while preserving evidence-grounded content. We evaluate TLVS on several benchmarks, including POPE, AMBER, CHAIR (COCO), MMHal, and HallusionBench, demonstrating consistent improvements over previous steering methods. 

---
# Crayotter: Traceable Multi-Agent Workflows for Long-Form Video Editing 

**Authors**: Lecheng Yan, Yichong Zhang, Ben Pan, Xiaoyu Zheng, Jiawei Qian, Anqi Wu, Wenxi Li, Chenyang Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07636)  

**Abstract**: Editing a long-form video from heterogeneous footage requires more than selecting clips: an agent must preserve narrative intent across material preparation, timeline construction, post-production, and revision while leaving enough evidence to diagnose failures. We present \textbf{Crayotter}, an open-source multimodal multi-agent system for prompt-driven video editing. Crayotter organizes production into three phases: coverage-aware material preparation, artifact-based editing research, and tool-grounded timeline execution. Each phase externalizes inspectable artifacts, including coverage reports, multimodal analyses, editing blueprints, tool calls, and intermediate renders. These artifacts make an editing run traceable and allow failed segments to be diagnosed and selectively revised instead of requiring a full restart. We evaluate Crayotter on 23 editing themes against CapCut-Mate and CutClaw. Under human evaluation, Crayotter achieves an average score of 3.40/5, compared with 2.44 and 1.70 for the two baselines, with consistent gains in theme alignment, narrative coherence, and editing smoothness. We additionally describe a replayable trajectory schema and verifiable reward design that prepare these workflows for future policy optimization. Code, traces, and examples are publicly available at this https URL. 

---
# Large Language Models Should Learn Personalized Rather Than Aggregated Human Preferences 

**Authors**: Cristina Garbacea  

**Link**: [PDF](https://arxiv.org/pdf/2606.07629)  

**Abstract**: Current approaches to aligning large language models (LLMs) aggregate diverse human preferences into a single reward signal, effectively optimizing for a hypothetical ``average user'' who represents no real person particularly well. This position paper argues that LLMs should learn personalized, individual preferences rather than aggregated ones. We show that aggregation masks critical information about preference diversity, individual values, and contextual dependencies, which is a limitation both theoretically grounded in social choice theory and empirically evident across demographic groups. We analyze the rich structure that human preferences encode, survey technical approaches to personalization, and systematically address counterarguments on scalability, shared standards, and manipulation risk. While personalization introduces genuine safety challenges including filter bubbles, value lock-in, and psychological manipulation, we argue these are manageable through bounded personalization frameworks that preserve universal safety constraints while accommodating legitimate individual variation. We conclude with a concrete research and policy agenda for developing preference-aware models that respect both individual autonomy and collective safety. 

---
# Item Response Scaling Laws: A Measurement Theory Approach for Efficient and Generalizable Neural Scaling Estimation 

**Authors**: Sang Truong, Yuheng Tu, Rylan Schaeffer, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2606.07616)  

**Abstract**: Scaling laws provide a fundamental framework for understanding the performance of Language Models (LMs), yet deriving them requires prohibitively expensive evaluations across thousands of checkpoints or millions of inference samples. To address this, we introduce Item Response Scaling Laws (IRSL), a unified framework that integrates Item Response Theory (IRT) within the scaling law framework. Unlike traditional approaches that treat each model-benchmark pair in isolation, IRSL disentangles latent model ability from question characteristics, factorizing the scaling law estimation for $M$ models and $N$ questions to significantly reduce parameter complexity from $O(M \times N)$ to $O(M + N)$. We instantiate IRSL with Beta-IRT, which leverages the empirical probability responses of LMs -- such as token probabilities in pre-training and pass rates in test-time sampling -- to capture richer signals than binary responses. We validate our approach across two prevalent scaling paradigms: (1) pre-training downstream scaling, using 6,612 LM checkpoints and 37,682 questions from 10 benchmarks; and (2) test-time scaling, using 12 LMs and 120 questions from 4 benchmarks with up to 2,500 samples per question. Given a one-time calibration on existing model responses, IRSL yields more reliable scaling estimates using only 50 questions per benchmark (a 99.9\% reduction), achieving comparable or superior decision accuracy to traditional approaches. Furthermore, we show that the estimated latent model abilities are generalizable, enabling accurate performance forecasting across benchmarks that share the same measurement objective. 

---
# LEAF: Growing Trees Without Branching for Speech-Aware Large Language Model Post-Training 

**Authors**: Argyrios Gerogiannis, Yekaterina Yegorova, Mark Hasegawa-Johnson, Venugopal V. Veeravalli  

**Link**: [PDF](https://arxiv.org/pdf/2606.07610)  

**Abstract**: State-of-the-art GRPO-style methods for speech-aware large language model post-training suffer from coarse credit assignment, broadcasting the same terminal-reward advantage to every token in a response. This ignores useful structure within rollout batches, where speech-conditioned completions often share prefixes before diverging at important decisions. We propose Low-rank Exploration with Adaptive Forking (LEAF), a retrospective tree-based RL method that recovers this structure without online branching or additional decoding. LEAF samples complete responses, selects high-surprisal boundaries, groups responses by shared prefixes, and assigns span-level advantages using descendant rewards. We theoretically justify LEAF's span-level credit assignment and boundary-selection design. Empirically, LEAF improves over GRPO across speech question answering and speech translation benchmarks under the same rollout and low-rank adaptation budget. Notably, smaller LEAF-trained models outperform current state-of-the-art, full-parameter baselines. 

---
# ResearchClawBench: A Benchmark for End-to-End Autonomous Scientific Research 

**Authors**: Wanghan Xu, Shuo Li, Tianlin Ye, Qinglong Cao, Yixin Chen, Hengjian Gao, Yiheng Wang, Qi Li, Kun Li, Sheng Xu, Shengdu Chai, Fangchen Yu, Xiangyu Zhao, Zhangrui Zhao, Weijie Ma, Zijie Guo, Haoyu Zhou, Haoxiang Yin, Lixue Cheng, Chaofan Hu, Haoxuan Li, Lu Mi, Xuxuan Xie, Yifan Zhou, Ruizhe Chen, Zhiwang Zhou, Xingjian Guo, Yuhao Zhou, Xuming He, Shengyuan Xu, Xinyu Gu, Jiamin Wu, Mianxin Liu, Chunfeng Song, Fenghua Ling, Dongzhan Zhou, Shixiang Tang, Yuqiang Li, Mao Su, Peng Ye, Siqi Sun, Bin Wang, Xue Yang, Zhenfei Yin, Tianfan Fu, Guangtao Zhai, Wanli Ouyang, Bo Zhang, Lei Bai, Wenlong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07591)  

**Abstract**: AI coding agents are increasingly used for scientific work, but their end-to-end autonomous research capability remains difficult to verify. We present ResearchClawBench, a benchmark for evaluating autonomous scientific research across 40 tasks from 10 scientific domains. Each task is grounded in a real published paper, provides related literature and raw data, and hides the target paper during evaluation. Expert-curated multimodal rubrics decompose the target scientific artifacts into weighted criteria, enabling evaluation of target-paper-level re-discovery while leaving room for new discovery. We evaluate seven autonomous research (auto-research) agents under a unified protocol and seventeen native LLMs through the lightweight ResearchHarness. Current systems remain far from reliable re-discovery: the strongest autonomous agent, Claude Code, averages 21.5, and the strongest ResearchHarness LLM, Claude-Opus-4.7, averages 20.7, with an LLM frontier mean of only 26.5. Error analysis shows that failures concentrate in experimental protocol mismatch, evidence mismatch, and missing scientific core. ResearchClawBench provides a reproducible evaluation frontier for measuring progress toward autonomous scientific research. 

---
# Evaluating Advanced Prompting on Gemini Flash for Multi-Hop Biomedical QA 

**Authors**: Ahmed Bajaber, Mohammed Alliheedi  

**Link**: [PDF](https://arxiv.org/pdf/2606.07548)  

**Abstract**: The MedHopQA challenge presents a critical test for Large Language Models (LLMs): complex, multi-hop reasoning in the high-stakes biomedical domain. This paper details our direct API-based evaluation of Google's Gemini Flash models, focusing on the impact of advanced prompt engineering. We designed a sophisticated, multi-component prompt for Gemini 2.0 Flash that combined role-playing, explicit multi-shot Chain-of-Thought (CoT) examples, and detailed formatting rules. Our best run, using this complex prompt, achieved a Concept Level Score of 0.720. This result dramatically outperformed a baseline prompt which scored only 0.565. Remarkably, this performance on the efficient Gemini 2.0 Flash was almost identical to the result from the next-generation Gemini 2.5 Flash. Our findings demonstrate that sophisticated prompt design is a critical factor for unlocking the full reasoning capabilities of modern LLMs. 

---
# PulseBench-Tab: A Multilingual Benchmark for Table Extraction with Graph-Based Evaluation 

**Authors**: Ritvik Pandey, Sid Manchkanti, Mohammed Wazir Adain, Mohammed Hadi, Dushyanth Sekhar  

**Link**: [PDF](https://arxiv.org/pdf/2606.07534)  

**Abstract**: We introduce PulseBench-Tab, an open multilingual benchmark for evaluating table extraction from document images. The benchmark comprises 1,820 human-annotated tables spanning 9 languages and 4 scripts (Latin, CJK, Arabic, Cyrillic), drawn from 380 real-world source documents including financial filings, government reports, and regulatory disclosures. Tables range from 2 to 1,183 cells, with 48.1% containing merged or spanning cells. Alongside the dataset, we propose T-LAG (Table Logical Adjacency Graph), a novel evaluation metric that models tables as directed graphs over cell adjacencies and computes structural and content fidelity in a single score via optimal bipartite matching. We evaluate 9 commercial and open-source table extraction systems across the benchmark and report per-language breakdowns. The full dataset, scoring code, and all provider outputs are publicly available. 

---
