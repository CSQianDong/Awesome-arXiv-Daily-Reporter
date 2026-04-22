# Pause or Fabricate? Training Language Models for Grounded Reasoning 

**Authors**: Yiwen Qiu, Linjuan Wu, Yizhou Liu, Yuchen Yan, Jin Ma, Xu Tan, Yao Hu, Daoxin Zhang, Wenqi Zhang, Weiming Lu, Jun Xiao, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.19656)  

**Abstract**: Large language models have achieved remarkable progress on complex reasoning tasks. However, they often implicitly fabricate information when inputs are incomplete, producing confident but unreliable conclusions -- a failure mode we term ungrounded reasoning. We argue that this issue arises not from insufficient reasoning capability, but from the lack of inferential boundary awareness -- the ability to recognize when the necessary premises for valid inference are missing. To address this issue, we propose Grounded Reasoning via Interactive Reinforcement Learning (GRIL), a multi-turn reinforcement learning framework for grounded reasoning under incomplete information. GRIL decomposes the reasoning process into two stages: clarify and pause, which identifies whether the available information is sufficient, and grounded reasoning, which performs task solving once the necessary premises are established. We design stage-specific rewards to penalize hallucinations, enabling models to detect gaps, stop proactively, and resume reasoning after clarification. Experiments on GSM8K-Insufficient and MetaMATH-Insufficient show that GRIL significantly improves premise detection (up to 45%), leading to a 30% increase in task success while reducing average response length by over 20%. Additional analyses confirm robustness to noisy user responses and generalization to out-of-distribution tasks. 

---
# Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment 

**Authors**: Bobo Li, Rui Wu, Zibo Ji, Meishan Zhang, Hao Fei, Min Zhang, Mong-Li Lee, Wynne Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19548)  

**Abstract**: Large Language Model agents have rapidly evolved from static text generators into dynamic systems capable of executing complex autonomous workflows. To enhance reliability, multi-agent frameworks assigning specialized roles are increasingly adopted to enable self-reflection and mutual auditing. While such role-playing effectively leverages domain expert knowledge, we find it simultaneously induces a human-like cognitive bias known as Actor-Observer Asymmetry (AOA). Specifically, an agent acting as an actor (during self-reflection) tends to attribute failures to external factors, whereas an observer (during mutual auditing) attributes the same errors to internal faults. We quantify this using our new Ambiguous Failure Benchmark, which reveals that simply swapping perspectives triggers the AOA effect in over 20% of cases for most models. To tame this bias, we introduce ReTAS (Reasoning via Thesis-Antithesis-Synthesis), a model trained through dialectical alignment to enforce perspective-invariant reasoning. By integrating dialectical chain-of-thought with Group Relative Policy Optimization, ReTAS guides agents to synthesize conflicting viewpoints into an objective consensus. Experiments demonstrate that ReTAS effectively mitigates attribution inconsistency and significantly improves fault resolution rates in ambiguous scenarios. 

---
# HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human-LLM Collaborative Writing 

**Authors**: Euntae Kim, Soomin Han, Buru Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19274)  

**Abstract**: Large language models (LLMs) are increasingly used as co-authors in collaborative writing, where users begin with rough drafts and rely on LLMs to complete, revise, and refine their content. However, this capability poses a serious safety risk: malicious users could jailbreak the models-filling incomplete drafts with dangerous content-to force them into generating harmful outputs. In this paper, we identify the vulnerability of current LLMs to such draft-based co-authoring jailbreak attacks and introduce HarDBench, a systematic benchmark designed to evaluate the robustness of LLMs against this emerging threat. HarDBench spans a range of high-risk domains-including Explosives, Drugs, Weapons, and Cyberattacks-and features prompts with realistic structure and domain-specific cues to assess the model susceptibility to harmful completions. To mitigate this risk, we introduce a safety-utility balanced alignment approach based on preference optimization, training models to refuse harmful completions while remaining helpful on benign drafts. Experimental results show that existing LLMs are highly vulnerable in co-authoring contexts and our alignment method significantly reduces harmful outputs without degrading performance on co-authoring capabilities. This presents a new paradigm for evaluating and aligning LLMs in human-LLM collaborative writing settings. Our new benchmark and dataset are available on our project page at this https URL 

---
# Lost in Translation: Do LVLM Judges Generalize Across Languages? 

**Authors**: Md Tahmid Rahman Laskar, Mohammed Saidul Islam, Mir Tafseer Nayeem, Amran Bhuiyan, Mizanur Rahman, Shafiq Joty, Enamul Hoque, Jimmy Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19405)  

**Abstract**: Automatic evaluators such as reward models play a central role in the alignment and evaluation of large vision-language models (LVLMs). Despite their growing importance, these evaluators are almost exclusively assessed on English-centric benchmarks, leaving open the question of how well these evaluators generalize across languages. To answer this question, we introduce MM-JudgeBench, the first large-scale benchmark for multilingual and multimodal judge model evaluation, which includes over 60K pairwise preference instances spanning 25 typologically diverse languages. MM-JudgeBench integrates two complementary subsets: a general vision-language preference evaluation subset extending VL-RewardBench, and a chart-centric visual-text reasoning subset derived from OpenCQA, enabling systematic analysis of reward models (i.e., LVLM judges) across diverse settings. We additionally release a multilingual training set derived from MM-RewardBench, disjoint from our evaluation data, to support domain adaptation. By evaluating 22 LVLMs (15 open-source, 7 proprietary), we uncover substantial cross-lingual performance variance in our proposed benchmark. Our analysis further shows that model size and architecture are poor predictors of multilingual robustness, and that even state-of-the-art LVLM judges exhibit inconsistent behavior across languages. Together, these findings expose fundamental limitations of current reward modeling and underscore the necessity of multilingual, multimodal benchmarks for developing reliable automated evaluators. 

---
# ReflectMT: Internalizing Reflection for Efficient and High-Quality Machine Translation 

**Authors**: Kunquan Li, Yingxue Zhang, Fandong Meng, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.19144)  

**Abstract**: Recent years have witnessed growing interest in applying Large Reasoning Models (LRMs) to Machine Translation (MT). Existing approaches predominantly adopt a "think-first-then-translate" paradigm. Although explicit reasoning trajectories significantly enhance translation quality, they incur prohibitive inference costs and latency. To address these limitations, we propose ReflectMT, a two-stage reflection internalization algorithm for machine translation that employs a "translate-first-think-later" paradigm. Our approach develops the model's "translate-reflect-refine" capability through reinforcement learning. In the first stage, we cultivate the model's capacity for high-quality reflection and refinement, thereby enhancing its semantic comprehension and task-specific knowledge. In the second stage, we train the model to internalize the knowledge acquired during reflection. As a result, during inference, ReflectMT operates in a direct translation mode, producing high-quality translations on the first attempt without any explicit reasoning steps. Experimental results on datasets such as WMT24 demonstrate that our model's first-pass translations during inference outperform multi-step reasoning LRMs such as DeepSeek-R1 in both automatic metrics and GPT-based evaluation, achieving a 2.16-point improvement in GPT-based translation quality evaluation while reducing token consumption by 94.33%. 

---
# TRN-R1-Zero: Text-rich Network Reasoning via LLMs with Reinforcement Learning Only 

**Authors**: Yilun Liu, Ruihong Qiu, Zi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19070)  

**Abstract**: Zero-shot reasoning on text-rich networks (TRNs) remains a challenging frontier, as models must integrate textual semantics with relational structure without task-specific supervision. While graph neural networks rely on fixed label spaces and supervised objectives, recent large language model (LLM)-based approaches often overlook graph context or depend on distillation from larger models, limiting generalisation. We propose TRN-R1-Zero, a post-training framework for TRN reasoning trained solely via reinforcement learning. TRN-R1-Zero directly optimises base LLMs using a Neighbour-aware Group Relative Policy Optimisation objective that dynamically adjusts rewards based on a novel margin gain metric for the informativeness of neighbouring signals, effectively guiding the model toward relational reasoning. Unlike prior methods, TRN-R1-Zero requires no supervised fine-tuning or chain-of-thought data generated from large reasoning models. Extensive experiments across citation, hyperlink, social and co-purchase TRN benchmarks demonstrate the superiority and robustness of TRN-R1-Zero. Moreover, relying strictly on node-level training, TRN-R1-Zero achieves zero-shot inference on edge- and graph-level tasks, extending beyond cross-domain transfer. The codebase is publicly available at this https URL. 

---
# The Rise of Verbal Tics in Large Language Models: A Systematic Analysis Across Frontier Models 

**Authors**: Shuai Wu, Xue Li, Yanna Feng, Yufang Li, Zhijun Wang, Ran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19139)  

**Abstract**: As Large Language Models (LLMs) continue to evolve through alignment techniques such as Reinforcement Learning from Human Feedback (RLHF) and Constitutional AI, a growing and increasingly conspicuous phenomenon has emerged: the proliferation of verbal tics -- repetitive, formulaic linguistic patterns that pervade model outputs. These range from sycophantic openers ("That's a great question!", "Awesome!") to pseudo-empathetic affirmations ("I completely understand your concern", "I'm right here to catch you") and overused vocabulary ("delve", "tapestry", "nuanced"). In this paper, we present a systematic analysis of the verbal tic phenomenon across eight state-of-the-art LLMs: GPT-5.4, Claude Opus 4.7, Gemini 3.1 Pro, Grok 4.2, Doubao-Seed-2.0-pro, Kimi K2.5, DeepSeek V3.2, and MiMo-V2-Pro. Utilizing a custom evaluation framework for standardized API-based evaluation, we assess 10,000 prompts across 10 task categories in both English and Chinese, yielding 160,000 model responses. We introduce the Verbal Tic Index (VTI), a composite metric quantifying tic prevalence, and analyze its correlation with sycophancy, lexical diversity, and human-perceived naturalness. Our findings reveal significant inter-model variation: Gemini 3.1 Pro exhibits the highest VTI (0.590), while DeepSeek V3.2 achieves the lowest (0.295). We further demonstrate that verbal tics accumulate over multi-turn conversations, are amplified in subjective tasks, and show distinct cross-lingual patterns. Human evaluation (N = 120) confirms a strong inverse relationship between sycophancy and perceived naturalness (r = -0.87, p < 0.001). These results underscore the "alignment tax" of current training paradigms and highlight the urgent need for more authentic human-AI interaction frameworks. 

---
# Prioritizing the Best: Incentivizing Reliable Multimodal Reasoning by Rewarding Beyond Answer Correctness 

**Authors**: Mengzhao Jia, Zhihan Zhang, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18892)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) improves multimodal reasoning by rewarding verifiable final answers. Yet answer-correct trajectories may still rely on incomplete derivations, weak evidence, or statements that contradict their conclusions. This gap between answer correctness and reasoning validity, which we call reasoning-answer inconsistency, motivates trajectory supervision in multimodal RL. We compare two main approaches: reward models (RMs), and Generative Rewards (GRs). RMs are efficient and help early in training, but their gains weaken as the policy distribution shifts; GRs improve performance, but may give unstable rewards and computationally expensive. We therefore propose Groupwise Ranking Reward, which ranks verifier-passed trajectories for the same prompt in one pass and redistributes reward accordingly. Groupwise comparison better separates stronger and weaker correct trajectories with lower judge overhead than GRs. Experiments show that RLVR aggravates reasoning-answer inconsistency, while trajectory supervision alleviates it. Groupwise Ranking Reward performs best overall, improving reliability-conditioned accuracy from 47.4% to 54.7% over RLVR. 

---
# EVPO: Explained Variance Policy Optimization for Adaptive Critic Utilization in LLM Post-Training 

**Authors**: Chengjun Pan, Shichun Liu, Jiahang Lin, Dingwei Zhu, Jiazheng Zhang, Shihan Dou, Songyang Gao, Zhenhua Han, Binghai Wang, Rui Zheng, Xuanjing Huang, Tao Gui, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.19485)  

**Abstract**: Reinforcement learning (RL) for LLM post-training faces a fundamental design choice: whether to use a learned critic as a baseline for policy optimization. Classical theory favors critic-based methods such as PPO for variance reduction, yet critic-free alternatives like GRPO have gained widespread adoption due to their simplicity and competitive performance. We show that in sparse-reward settings, a learned critic can inject estimation noise that exceeds the state signal it captures, increasing rather than reducing advantage variance. By casting baseline selection as a Kalman filtering problem, we unify PPO and GRPO as two extremes of the Kalman gain and prove that explained variance (EV), computable from a single training batch, identifies the exact boundary: positive EV indicates the critic reduces variance, while zero or negative EV signals that it inflates variance. Building on this insight, we propose Explained Variance Policy Optimization (EVPO), which monitors batch-level EV at each training step and adaptively switches between critic-based and batch-mean advantage estimation, provably achieving no greater variance than the better of the two at every step. Across four tasks spanning classical control, agentic interaction, and mathematical reasoning, EVPO consistently outperforms both PPO and GRPO regardless of which fixed baseline is stronger on a given task. Further analysis confirms that the adaptive gating tracks critic maturation over training and that the theoretically derived zero threshold is empirically optimal. 

---
# Who Shapes Brazil's Vaccine Debate? Semi-Supervised Modeling of Stance and Polarization in YouTube's Media Ecosystem 

**Authors**: Geovana S. de Oliveira, Ana P. C. Silva, Fabricio Murai, Carlos H. G. Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2604.18586)  

**Abstract**: Vaccination remains a cornerstone of global public health, yet the COVID-19 pandemic exposed how online misinformation, political polarization, and declining institutional trust can undermine immunization efforts. Most of the prior computational studies that analyzed vaccine discourse on social platforms focus on English-language data, specific vaccines, or short time windows, impairing our understanding of long-term dynamics in high-impact, non-English contexts like Brazil, home to one of the world's most comprehensive immunization systems. We here present the largest longitudinal study of Brazil's vaccine discourse on YouTube, leveraging a semi-supervised stance detection framework that combines self-labeling and self-training to classify nearly 1.4 million comments. By integrating stance with temporal patterns, engagement metrics, and channel taxonomy (legacy media, science communicators, digital-native outlets), we map how pro- and anti-vaccine narratives evolve and circulate within a hybrid media ecosystem. Our results show that semi-supervised learning substantially improves stance classification robustness, enabling fine-grained tracking of public attitudes across Brazil's full immunization schedule. Polarization spikes during epidemiological crises, especially COVID-19, but becomes fragmented across vaccines and interaction patterns in the post-pandemic period. Notably, science communication and digital-native channels emerge as the primary loci of both supportive and oppositional engagement, revealing structural vulnerabilities in contemporary health communication. Thus, our work advances computational methods for large-scale stance modeling while offering actionable evidence for public health agencies, platform governance, and online information ecosystems. 

---
# Human-Guided Harm Recovery for Computer Use Agents 

**Authors**: Christy Li, Sky CH-Wang, Andi Peng, Andreea Bobu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18847)  

**Abstract**: As LM agents gain the ability to execute actions on real computer systems, we need ways to not only prevent harmful actions at scale but also effectively remediate harm when prevention fails. We formalize a solution to this neglected challenge in post-execution safeguards as harm recovery: the problem of optimally steering an agent from a harmful state back to a safe one in alignment with human preferences. We ground preference-aligned recovery through a formative user study that identifies valued recovery dimensions and produces a natural language rubric. Our dataset of 1,150 pairwise judgments reveals context-dependent shifts in attribute importance, such as preferences for pragmatic, targeted strategies over comprehensive long-term approaches. We operationalize these learned insights in a reward model, re-ranking multiple candidate recovery plans generated by an agent scaffold at test time. To evaluate recovery capabilities systematically, we introduce BackBench, a benchmark of 50 computer-use tasks that test an agent's ability to recover from harmful states. Human evaluation shows our reward model scaffold yields higher-quality recovery trajectories than base agents and rubric-based scaffolds. Together, these contributions lay the foundation for a new class of agent safety methods -- ones that confront harm not only by preventing it, but by navigating its aftermath with alignment and intent. 

---
# Multi-modal Reasoning with LLMs for Visual Semantic Arithmetic 

**Authors**: Chuou Xu, Liya Ji, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.19567)  

**Abstract**: Reinforcement learning (RL) as post-training is crucial for enhancing the reasoning ability of large language models (LLMs) in coding and math. However, their capacity for visual semantic arithmetic, inferring relationships from images, remains underexplored. The classic text analogy "king"-"man"+"woman" = "queen" illustrates relational reasoning, yet replacing text with images of "king" and "man" significantly reduces performance because it requires commonsense knowledge and the extraction of concise concepts from irrelevant visual details. This capability is important for service and domestic robotics in unstructured environments, where robots must infer semantic relationships among objects, agents, and actions. In a kitchen, recognizing from images that "powder" and "cake" are related by "is made of" grounds symbolic relations in perception, enabling tool substitution, task generalization, and improved semantic reasoning. Prior work approaches semantic arithmetic by decoding image features after vector arithmetic, but suffers from modality gaps and lacks systematic evaluation. In this paper, we formulate two novel tasks, two-term subtraction and three-term operations, and construct the Image-Relation-Pair Dataset (IRPD) for benchmarking. We further propose Semantic Arithmetic Reinforcement Fine-Tuning (SAri-RFT), which post-trains large vision-language models (LVLMs) using a verifiable function and Group Relative Policy Optimization (GRPO). Our method achieves state-of-the-art results on IRPD and the real-world Visual7W-Telling dataset. By equipping LVLMs with robust cross-modal relational reasoning, this work advances domestic robots' ability to ground symbolic reasoning in perception, enhancing decision-making, tool adaptability, and human-robot interaction in complex environments. Datasets and source code are provided in the supplementary material. 

---
# DT2IT-MRM: Debiased Preference Construction and Iterative Training for Multimodal Reward Modeling 

**Authors**: Zhihong Zhang, Jie Zhao, Xiaojian Huang, Jin Xu, Zhuodong Luo, Xin Liu, Jiansheng Wei, Xuejin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.19544)  

**Abstract**: Multimodal reward models (MRMs) play a crucial role in aligning Multimodal Large Language Models (MLLMs) with human preferences. Training a good MRM requires high-quality multimodal preference data. However, existing preference datasets face three key challenges: lack of granularity in preference strength, textual style bias, and unreliable preference signals. Besides, existing open-source multimodal preference datasets suffer from substantial noise, yet there is a lack of effective and scalable curation methods to enhance their quality. To address these limitations, we propose \textbf{DT2IT-MRM}, which integrates a \textbf{D}ebiased preference construction pipeline, a novel reformulation of text-to-image (\textbf{T2I}) preference data, and an \textbf{I}terative \textbf{T}raining framework that curates existing multimodal preference datasets for \textbf{M}ultimodal \textbf{R}eward \textbf{M}odeling. Our experimental results show that DT2IT-MRM achieves new \textbf{state-of-the-art} overall performance on three major benchmarks: VL-RewardBench, Multimodal RewardBench, and MM-RLHF-RewardBench. 

---
# Reinforcement Learning Improves LLM Accuracy and Reasoning in Disease Classification from Radiology Reports 

**Authors**: Yishu Wei, Yi Lin, Adam Flanders, George Shih, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2604.19060)  

**Abstract**: Accurate disease classification from radiology reports is essential for many applications. While supervised fine-tuning (SFT) of lightweight LLMs improves accuracy, it can degrade reasoning. We propose a two-stage approach: SFT on disease labels followed by Group Relative Policy Optimization (GRPO) to refine predictions by optimizing accuracy and format without reasoning supervision. Across three radiologist-annotated datasets, SFT outperformed baselines and GRPO further improved classification and enhanced reasoning recall and comprehensiveness. 

---
# Reasoning-Aware AIGC Detection via Alignment and Reinforcement 

**Authors**: Zhao Wang, Max Xiong, Jianxun Lian, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2604.19172)  

**Abstract**: The rapid advancement and widespread adoption of Large Language Models (LLMs) have elevated the need for reliable AI-generated content (AIGC) detection, which remains challenging as models evolve. We introduce AIGC-text-bank, a comprehensive multi-domain dataset with diverse LLM sources and authorship scenarios, and propose REVEAL, a detection framework that generates interpretable reasoning chains before classification. Our approach uses a two-stage training strategy: supervised fine-tuning to establish reasoning capabilities, followed by reinforcement learning to improve accuracy, improve logical consistency, and reduce hallucinations. Extensive experiments show that REVEAL achieves state-of-the-art performance across multiple benchmarks, offering a robust and transparent solution for AIGC detection. The project is open-source at this https URL 

---
# SAVOIR: Learning Social Savoir-Faire via Shapley-based Reward Attribution 

**Authors**: Xiachong Feng, Yi Jiang, Xiaocheng Feng, Deyi Yin, Libo Qin, Yangfan Ye, Lei Huang, Weitao Ma, Yuxuan Gu, Chonghan Qin, Bing Qin, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2604.18982)  

**Abstract**: Social intelligence, the ability to navigate complex interpersonal interactions, presents a fundamental challenge for language agents. Training such agents via reinforcement learning requires solving the credit assignment problem: determining how individual utterances contribute to multi-turn dialogue outcomes. Existing approaches directly employ language models to distribute episode-level rewards, yielding attributions that are retrospective and lack theoretical grounding. We propose SAVOIR (ShApley Value fOr SocIal RL), a novel principled framework grounded in cooperative game theory. Our approach combines two complementary principles: expected utility shifts evaluation from retrospective attribution to prospective valuation, capturing an utterance's strategic potential for enabling favorable future trajectories; Shapley values ensure fair credit distribution with axiomatic guarantees of efficiency, symmetry, and marginality. Experiments on the SOTOPIA benchmark demonstrate that SAVOIR achieves new state-of-the-art performance across all evaluation settings, with our 7B model matching or exceeding proprietary models including GPT-4o and Claude-3.5-Sonnet. Notably, even large reasoning models consistently underperform, suggesting social intelligence requires qualitatively different capabilities than analytical reasoning. 

---
# OLLM: Options-based Large Language Models 

**Authors**: Shashank Sharma, Janina Hoffmann, Vinay Namboodiri  

**Link**: [PDF](https://arxiv.org/pdf/2604.19087)  

**Abstract**: We introduce Options LLM (OLLM), a simple, general method that replaces the single next-token prediction of standard LLMs with a \textit{set of learned options} for the next token, indexed by a discrete latent variable. Instead of relying on temperature or sampling heuristics to induce diversity, OLLM models variation explicitly: a small latent space parametrizes multiple plausible next-token options which can be selected or searched by a downstream policy. Architecturally, OLLM is a lightweight "plug-in" that inserts two layers: an encoder and a decoder, before the output head, allowing almost any pretrained LLM to be converted with minimal additional parameters. We apply OLLM to a 1.7B-parameter backbone (only $1.56\%$ of parameters trainable) trained on OpenMathReasoning and evaluated on OmniMath. The SOTA LoRA-adapted baselines peak at $51\%$ final answer correctness, while OLLM's option set allows up to $\sim 70\%$ under optimal latent selection. We then train a compact policy in the latent space that emits latents to control generation. Operating in a low-dimensional option space makes reward optimization far more sample-efficient and substantially reduces common misalignments (e.g., language switching or degenerate reasoning), as the policy is constrained to options learned during SFT. Crucially, this alignment arises from model structure rather than additional KL or handcrafted alignment losses. Our results demonstrate that optionized next-token modeling enhances controllability, robustness, and efficiency in math reasoning, and highlight latent-space policy learning as a promising direction for reinforcement learning in LLMs. 

---
# ARES: Adaptive Red-Teaming and End-to-End Repair of Policy-Reward System 

**Authors**: Jiacheng Liang, Yao Ma, Tharindu Kumarage, Satyapriya Krishna, Rahul Gupta, Kai-Wei Chang, Aram Galstyan, Charith Peris  

**Link**: [PDF](https://arxiv.org/pdf/2604.18789)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is central to aligning Large Language Models (LLMs), yet it introduces a critical vulnerability: an imperfect Reward Model (RM) can become a single point of failure when it fails to penalize unsafe behaviors. While existing red-teaming approaches primarily target policy-level weaknesses, they overlook what we term systemic weaknesses cases where both the core LLM and the RM fail in tandem.
We present ARES, a framework that systematically discovers and mitigates such dual vulnerabilities. ARES employs a ``Safety Mentor'' that dynamically composes semantically coherent adversarial prompts by combining structured component types (topics, personas, tactics, goals) and generates corresponding malicious and safe responses. This dual-targeting approach exposes weaknesses in both the core LLM and the RM simultaneously. Using the vulnerabilities gained, ARES implements a two-stage repair process: first fine-tuning the RM to better detect harmful content, then leveraging the improved RM to optimize the core model. Experiments across multiple adversarial safety benchmarks demonstrate that ARES substantially enhances safety robustness while preserving model capabilities, establishing a new paradigm for comprehensive RLHF safety alignment. 

---
# Distillation Traps and Guards: A Calibration Knob for LLM Distillability 

**Authors**: Weixiao Zhan, Yongcheng Jing, Leszek Rutkowski, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.18963)  

**Abstract**: Knowledge distillation (KD) transfers capabilities from large language models (LLMs) to smaller students, yet it can fail unpredictably and also underpins model leakage risks. Our analysis revealed several distillation traps: tail noise, off-policy instability, and, most fundamentally, the teacher-student gap, that distort training signals. These traps manifest as overconfident hallucinations, self-correction collapse, and local decoding degradation, causing distillation to fail. Motivated by these findings, we propose a post-hoc calibration method that, to the best of our knowledge, for the first time enables control over a teacher's distillability via reinforcement fine-tuning (RFT). Our objective combines task utility, KL anchor, and across-tokenizer calibration reward. This makes distillability a practical safety lever for foundation models, connecting robust teacher-student transfer with deployment-aware model protection. Experiments across math, knowledge QA, and instruction-following tasks show that students distilled from distillable calibrated teachers outperform SFT and KD baselines, while undistillable calibrated teachers retain their task performance but cause distilled students to collapse, offering a practical knob for both better KD and model IP protection. 

---
# Fine-Tuning Small Reasoning Models for Quantum Field Theory 

**Authors**: Nathaniel S. Woodward, Zhiqi Gao, Yurii Kvasiuk, Kendrick M. Smith, Frederic Sala, Moritz Münchmeyer  

**Link**: [PDF](https://arxiv.org/pdf/2604.18936)  

**Abstract**: Despite the growing application of Large Language Models (LLMs) to theoretical physics, there is little academic exploration into how domain-specific physics reasoning ability develops while training these models. To investigate this, we perform the first academic fine-tuning study of small (7B-parameter) reasoning models dedicated specifically to theoretical physics. Because open-source verifiable training data required to train such capabilities is scarce, we developed a robust data generation pipeline that can both create synthetic problems and make existing human-authored problems suitable for model training. Selecting Quantum Field Theory (QFT) as our primary domain, we generated over 2,500 synthetic problems alongside a curated collection of human-adapted problems sourced from arXiv and standard pedagogical resources. We conduct both Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT) experiments, benchmarking performance gains as well as generalization to other physics domains. We perform an extensive analysis of model chains-of-though before and after fine-tuning, to understand how reasoning errors evolve during RL and SFT. Finally, we publicly release our data pipeline, verifiable QFT training data, and $\sim$200M tokens of QFT reasoning traces. 

---
# Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning 

**Authors**: Zhiyin Yu, Bo Zhang, Qibin Hou, Zhonghai Wu, Xiao Luo, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2604.18639)  

**Abstract**: Previous LLMs-based RL studies typically follow either supervised learning with high annotation costs, or unsupervised paradigms using voting or entropy-based rewards. However, their performance remains far from satisfactory due to the substantial annotation cost and issues such as model collapse or reward hacking. To address these issues, we introduce a new perspective inspired by cognitive learning theory and propose a novel approach called EasyRL. The core of EasyRL is to simulate the human cognitive acquisition curve by integrating reliable knowledge transfer from easy labeled data with a progressive divide-and-conquer strategy that tackles increasingly difficult unlabeled data. Specifically, we initialize a warm-up model using supervised RL with few-shot labeled data. This is followed by a divide-and-conquer pseudo-labeling strategy on difficult unlabeled data, combining consistency-based selection for low-uncertainty cases and reflection-based resolution for medium-uncertainty cases. Finally, difficulty-progressive self-training with iterative pseudo-labeling and RL further strengthens the model's reasoning capability. EasyRL provides a unified self-evolving framework that facilitates data-efficient post-training of LLMs. Experimental results on mathematical and scientific benchmarks demonstrate that EasyRL, using only 10% of easy labeled data, consistently outperforms state-of-the-art baselines. 

---
# Compile to Compress: Boosting Formal Theorem Provers by Compiler Outputs 

**Authors**: Guchan Li, Rui Tian, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18587)  

**Abstract**: Large language models (LLMs) have demonstrated significant potential in formal theorem proving, yet state-of-the-art performance often necessitates prohibitive test-time compute via massive roll-outs or extended context windows. In this work, we address this scalability bottleneck by exploiting an informative structure in formal verification: the observation that compilers map a vast space of diverse proof attempts to a compact set of structured failure modes. We introduce a learning-to-refine framework that leverages this compression to perform efficient learning and proof exploration. We perform tree search that corrects errors locally conditioned on explicit verifier feedback, thereby circumventing the costs associated with accumulating a long history of proof attempts. Extensive evaluations show that our method consistently amplifies the reasoning capabilities of base provers across varying scales. Notably, our approach achieves state-of-the-art performance on PutnamBench among publicly reported $\sim$8B and $\sim$32B parameter models under comparable test-time budgets, offering a scalable paradigm for next-generation verifier-guided reasoning. 

---
