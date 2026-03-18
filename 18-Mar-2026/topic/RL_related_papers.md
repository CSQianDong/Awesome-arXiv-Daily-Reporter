# EmoLLM: Appraisal-Grounded Cognitive-Emotional Co-Reasoning in Large Language Models 

**Authors**: Yifei Zhang, Mingyang Li, Henry Gao, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16553)  

**Abstract**: Large language models (LLMs) demonstrate strong cognitive intelligence (IQ), yet many real-world interactions also require emotional intelligence (EQ) to produce responses that are both factually reliable and emotionally appropriate. In settings such as emotional support, technical assistance, and consultation, effective dialogue depends on how situations are appraised with respect to the user's needs, goals, and coping capacity. Inspired by appraisal theory, we propose EmoLLM, an appraisal-grounded framework for IQ/EQ co-reasoning in dialogue. EmoLLM uses an explicit Appraisal Reasoning Graph (ARG) to structure intermediate reasoning over contextual facts, inferred user needs, appraisal dimensions, emotional states, and response strategies before generating a reply. We train EmoLLM in a multi-turn role-play environment with reinforcement learning, where reverse-perspective reasoning provides reward signals based on predicted user-side consequences of responses. Across diverse dialogue settings, EmoLLM improves emotional state outcomes and response quality over strong baselines while preserving strong factual reliability. 

---
# PlotTwist: A Creative Plot Generation Framework with Small Language Models 

**Authors**: Abhinav Thorat, Ravi Kolla, Jyotin Goel, Niranjan Pedanekar  

**Link**: [PDF](https://arxiv.org/pdf/2603.16410)  

**Abstract**: Creative plot generation presents a fundamental challenge for language models: transforming a concise premise into a coherent narrative that sustains global structure, character development, and emotional resonance. Although recent Large Language Models (LLMs) demonstrate strong fluency across general-purpose tasks, they typically require preference alignment to perform well on specialized domains such as creative plot generation. However, conducting such alignment at the scale of frontier LLMs is computationally prohibitive, significantly limiting accessibility and practical deployment. To address this, we present PlotTwist, a structured framework that enables Small Language Models (SLMs) with $\leq$ 5B active parameters to generate high-quality, premise-conditioned plots competitive with frontier systems up to $200\times$ larger. Our approach decomposes generation into three specialized components: (1) an Aspect Rating Reward Model trained via a novel Positive-Negative prompting strategy to deliver structured narratives across five Narrative Quality Dimensions (NQDs); (2) a Mixture-of-Experts (MoE) plot generator aligned via Direct Preference Optimization on high-confidence preference pairs; and (3) an Agentic Evaluation module that emulates human critical judgment for unbiased post-hoc assessment. Extensive experiments demonstrate that PlotTwist consistently outperforms frontier models across multiple NQDs despite substantially tighter capacity constraints. Further validation confirms strong sensitivity to narrative quality, as the framework reliably distinguishes plots derived from critically acclaimed versus widely panned screenplays. Together, these results establish structured, preference-based alignment as a resource-efficient approach to high-quality creative plot generation. 

---
# A Family of LLMs Liberated from Static Vocabularies 

**Authors**: Aleph Alpha, Adnen Abdessaied, Artur Baranowski, Lukas Balles, Michael Barlow, Fabien C. Y. Benureau, Felix Berkenkamp, Lukas Bluebaum, Bastian Boll, Thomas F. Burns, Björn Deiseroth, Constantin Eichenberg, David Friede, Pablo Iyu Guerrero, Ahmed Hammam, Bastian Harren, Johann Higl, Yasser Jadidi, Carina Kauf, Johannes Messner, Jan Hendrik Metzen, Max Meuer, Vedant Nanda, Pit Neitemeier, Koen Oostermeijer, Letitia Parcalabescu, Markus Pernpointner, Felix Reinfurt, Dylan Rodriquez, Grégory Schott, Philipp Siedler, Martin Simonovsky, Till Speicher, Volker Stampa, Stephan Wäldchen, Samuel Weinbach, Gregor Ziegltrum  

**Link**: [PDF](https://arxiv.org/pdf/2603.15953)  

**Abstract**: Tokenization is a central component of natural language processing in current large language models (LLMs), enabling models to convert raw text into processable units. Although learned tokenizers are widely adopted, they exhibit notable limitations, including their large, fixed vocabulary sizes and poor adaptability to new domains or languages. We present a family of models with up to 70 billion parameters based on the hierarchical autoregressive transformer (HAT) architecture. In HAT, an encoder transformer aggregates bytes into word embeddings and then feeds them to the backbone, a classical autoregressive transformer. The outputs of the backbone are then cross-attended by the decoder and converted back into bytes. We show that we can reuse available pre-trained models by converting the Llama 3.1 8B and 70B models into the HAT architecture: Llama-3.1-8B-TFree-HAT and Llama-3.1-70B-TFree-HAT are byte-level models whose encoder and decoder are trained from scratch, but where we adapt the pre-trained Llama backbone, i.e., the transformer blocks with the embedding matrix and head removed, to handle word embeddings instead of the original tokens. We also provide a 7B HAT model, Llama-TFree-HAT-Pretrained, trained entirely from scratch on nearly 4 trillion words. The HAT architecture improves text compression by reducing the number of required sequence positions and enhances robustness to intra-word variations, e.g., spelling differences. Through pre-training, as well as subsequent supervised fine-tuning and direct preference optimization in English and German, we show strong proficiency in both languages, improving on the original Llama 3.1 in most benchmarks. We release our models (including 200 pre-training checkpoints) on Hugging Face. 

---
# Efficient Reasoning on the Edge 

**Authors**: Yelysei Bondarenko, Thomas Hehn, Rob Hesselink, Romain Lepert, Fabio Valerio Massoli, Evgeny Mironov, Leyla Mirvakhabova, Tribhuvanesh Orekondy, Spyridon Stasis, Andrey Kuzmin, Anna Kuzina, Markus Nagel, Ankita Nayak, Corrado Rainone, Ork de Rooij, Paul N Whatmough, Arash Behboodi, Babak Ehteshami Bejnordi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16867)  

**Abstract**: Large language models (LLMs) with chain-of-thought reasoning achieve state-of-the-art performance across complex problem-solving tasks, but their verbose reasoning traces and large context requirements make them impractical for edge deployment. These challenges include high token generation costs, large KV-cache footprints, and inefficiencies when distilling reasoning capabilities into smaller models for mobile devices. Existing approaches often rely on distilling reasoning traces from larger models into smaller models, which are verbose and stylistically redundant, undesirable for on-device inference. In this work, we propose a lightweight approach to enable reasoning in small LLMs using LoRA adapters combined with supervised fine-tuning. We further introduce budget forcing via reinforcement learning on these adapters, significantly reducing response length with minimal accuracy loss. To address memory-bound decoding, we exploit parallel test-time scaling, improving accuracy at minor latency increase. Finally, we present a dynamic adapter-switching mechanism that activates reasoning only when needed and a KV-cache sharing strategy during prompt encoding, reducing time-to-first-token for on-device inference. Experiments on Qwen2.5-7B demonstrate that our method achieves efficient, accurate reasoning under strict resource constraints, making LLM reasoning practical for mobile scenarios. Videos demonstrating our solution running on mobile devices are available on our project page. 

---
# MedArena: Comparing LLMs for Medicine-in-the-Wild Clinician Preferences 

**Authors**: Eric Wu, Kevin Wu, Jason Hom, Paul H. Yi, Angela Zhang, Alejandro Lozano, Jeff Nirschl, Jeff Tangney, Kevin Byram, Braydon Dymm, Narender Annapureddy, Eric Topol, David Ouyang, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.15677)  

**Abstract**: Large language models (LLMs) are increasingly central to clinician workflows, spanning clinical decision support, medical education, and patient communication. However, current evaluation methods for medical LLMs rely heavily on static, templated benchmarks that fail to capture the complexity and dynamics of real-world clinical practice, creating a dissonance between benchmark performance and clinical utility. To address these limitations, we present MedArena, an interactive evaluation platform that enables clinicians to directly test and compare leading LLMs using their own medical queries. Given a clinician-provided query, MedArena presents responses from two randomly selected models and asks the user to select the preferred response. Out of 1571 preferences collected across 12 LLMs up to November 1, 2025, Gemini 2.0 Flash Thinking, Gemini 2.5 Pro, and GPT-4o were the top three models by Bradley-Terry rating. Only one-third of clinician-submitted questions resembled factual recall tasks (e.g., MedQA), whereas the majority addressed topics such as treatment selection, clinical documentation, or patient communication, with ~20% involving multi-turn conversations. Additionally, clinicians cited depth and detail and clarity of presentation more often than raw factual accuracy when explaining their preferences, highlighting the importance of readability and clinical nuance. We also confirm that the model rankings remain stable even after controlling for style-related factors like response length and formatting. By grounding evaluation in real-world clinical questions and preferences, MedArena offers a scalable platform for measuring and improving the utility and efficacy of medical LLMs. 

---
# Aligning Paralinguistic Understanding and Generation in Speech LLMs via Multi-Task Reinforcement Learning 

**Authors**: Jingxiang Chen, Minseok Kim, Seong-Gyun Leem, Yin Huang, Rashi Rungta, Zhicheng Ouyang, Haibin Wu, Surya Teja Appini, Ankur Bansal, Yang Bai, Yue Liu, Florian Metze, Ahmed A Aly, Anuj Kumar, Ariya Rastrow, Zhaojiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.15981)  

**Abstract**: Speech large language models (LLMs) observe paralinguistic cues such as prosody, emotion, and non-verbal sounds--crucial for intent understanding. However, leveraging these cues faces challenges: limited training data, annotation difficulty, and models exploiting lexical shortcuts over paralinguistic signals. We propose multi-task reinforcement learning (RL) with chain-of-thought prompting that elicits explicit affective reasoning. To address data scarcity, we introduce a paralinguistics-aware speech LLM (PALLM) that jointly optimizes sentiment classification from audio and paralinguistics-aware response generation via a two-stage pipeline. Experiments demonstrate that our approach improves paralinguistics understanding over both supervised baselines and strong proprietary models (Gemini-2.5-Pro, GPT-4o-audio) by 8-12% on Expresso, IEMOCAP, and RAVDESS. The results show that modeling paralinguistic reasoning with multi-task RL is crucial for building emotionally intelligent speech LLMs. 

---
# Offline Exploration-Aware Fine-Tuning for Long-Chain Mathematical Reasoning 

**Authors**: Yongyu Mu, Jiali Zeng, Fandong Meng, JingBo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16206)  

**Abstract**: Through encouraging self-exploration, reinforcement learning from verifiable rewards (RLVR) has significantly advanced the mathematical reasoning capabilities of large language models. As the starting point for RLVR, the capacity of supervised fine-tuning (SFT) to memorize new chain-of-thought trajectories provides a crucial initialization that shapes the subsequent exploration landscape. However, existing research primarily focuses on facilitating exploration during RLVR training, leaving exploration-aware SFT under-explored. To bridge this gap, we propose Offline eXploration-Aware (OXA) fine-tuning. Specifically, OXA optimizes two objectives: promoting low-confidence verified teacher-distillation data to internalize previously uncaptured reasoning patterns, and suppressing high-confidence incorrect self-distillation data to redistribute probability mass of incorrect patterns toward potentially correct candidates. Experimental results across 6 benchmarks show that OXA consistently improves mathematical reasoning performance, especially achieving an average gain of $+6$ Pass@1 and $+5$ Pass@$k$ points compared to conventional SFT on the Qwen2.5-1.5B-Math. Crucially, OXA elevates initial policy entropy, and performance gains persist throughout extensive RLVR training, demonstrating the long-term value of OXA. 

---
# HIPO: Instruction Hierarchy via Constrained Reinforcement Learning 

**Authors**: Keru Chen, Jun Luo, Sen Lin, Yingbin Liang, Alvaro Velasquez, Nathaniel Bastian, Shaofeng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.16152)  

**Abstract**: Hierarchical Instruction Following (HIF) refers to the problem of prompting large language models with a priority-ordered stack of instructions. Standard methods like RLHF and DPO typically fail in this problem since they mainly optimize for a single objective, failing to explicitly enforce system prompt compliance. Meanwhile, supervised fine-tuning relies on mimicking filtered, compliant data, which fails to establish the priority asymmetry at the algorithmic level. In this paper, we introduce \textsc{HIPO}, a novel alignment framework that formulates HIF as a Constrained Markov Decision Process. \textsc{HIPO} elevates system prompts from mere input context to strict algorithmic boundaries. Using a primal-dual safe reinforcement learning approach, the algorithm dynamically enforces system prompt compliance as an explicit constraint, maximizing user utility strictly within this feasible region. Extensive evaluations across diverse model architectures (e.g., Qwen, Phi, Llama) demonstrate that \textsc{HIPO} significantly improves both system compliance and user utility. Furthermore, mechanistic analysis reveals that this constrained optimization autonomously drives the model to shift its attention toward long-range system tokens, providing a principled foundation for reliable LLM deployment in complex workflows. 

---
# SWE-QA-Pro: A Representative Benchmark and Scalable Training Recipe for Repository-Level Code Understanding 

**Authors**: Songcheng Cai, Zhiheng Lyu, Yuansheng Ni, Xiangchao Chen, Baichuan Zhou, Shenzhe Zhu, Yi Lu, Haozhe Wang, Chi Ruan, Benjamin Schneider, Weixu Zhang, Xiang Li, Andy Zheng, Yuyu Zhang, Ping Nie, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16124)  

**Abstract**: Agentic repository-level code understanding is essential for automating complex software engineering tasks, yet the field lacks reliable benchmarks. Existing evaluations often overlook the long tail topics and rely on popular repositories where Large Language Models (LLMs) can cheat via memorized knowledge. To address this, we introduce SWE-QA-Pro, a benchmark constructed from diverse, long-tail repositories with executable environments. We enforce topical balance via issue-driven clustering to cover under-represented task types and apply a rigorous difficulty calibration process: questions solvable by direct-answer baselines are filtered out. This results in a dataset where agentic workflows significantly outperform direct answering (e.g., a ~13-point gap for Claude Sonnet 4.5), confirming the necessity of agentic codebase exploration. Furthermore, to tackle the scarcity of training data for such complex behaviors, we propose a scalable synthetic data pipeline that powers a two-stage training recipe: Supervised Fine-Tuning (SFT) followed by Reinforcement Learning from AI Feedback (RLAIF). This approach allows small open models to learn efficient tool usage and reasoning. Empirically, a Qwen3-8B model trained with our recipe surpasses GPT-4o by 2.3 points on SWE-QA-Pro and substantially narrows the gap to state-of-the-art proprietary models, demonstrating both the validity of our evaluation and the effectiveness of our agentic training workflow. 

---
# IQuest-Coder-V1 Technical Report 

**Authors**: Jian Yang, Wei Zhang, Shawn Guo, Zhengmao Ye, Lin Jing, Shark Liu, Yizhi Li, Jiajun Wu, Cening Liu, X. Ma, Yuyang Song, Siwei Wu, Yuwen Li, L. Liao, T. Zheng, Ziling Huang, Zelong Huang, Che Liu, Yan Xing, Renyuan Li, Qingsong Cai, Hanxu Yan, Siyue Wang, Shikai Li, Jason Klein Liu, An Huang, Yongsheng Kang, Jinxing Zhang, Chuan Hao, Haowen Wang, Weicheng Gu, Ran Tao, Mingjie Tang, Peihao Wu, Jianzhou Wang, Xianglong Liu, Weifeng Lv, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2603.16733)  

**Abstract**: In this report, we introduce the IQuest-Coder-V1 series-(7B/14B/40B/40B-Loop), a new family of code large language models (LLMs). Moving beyond static code representations, we propose the code-flow multi-stage training paradigm, which captures the dynamic evolution of software logic through different phases of the pipeline. Our models are developed through the evolutionary pipeline, starting with the initial pre-training consisting of code facts, repository, and completion data. Following that, we implement a specialized mid-training stage that integrates reasoning and agentic trajectories in 32k-context and repository-scale in 128k-context to forge deep logical foundations. The models are then finalized with post-training of specialized coding capabilities, which is bifurcated into two specialized paths: the thinking path (utilizing reasoning-driven RL) and the instruct path (optimized for general assistance). IQuest-Coder-V1 achieves state-of-the-art performance among competitive models across critical dimensions of code intelligence: agentic software engineering, competitive programming, and complex tool use. To address deployment constraints, the IQuest-Coder-V1-Loop variant introduces a recurrent mechanism designed to optimize the trade-off between model capacity and deployment footprint, offering an architecturally enhanced path for efficacy-efficiency trade-off. We believe the release of the IQuest-Coder-V1 series, including the complete white-box chain of checkpoints from pre-training bases to the final thinking and instruction models, will advance research in autonomous code intelligence and real-world agentic systems. 

---
# Alternating Reinforcement Learning with Contextual Rubric Rewards 

**Authors**: Guangchen Lan  

**Link**: [PDF](https://arxiv.org/pdf/2603.15646)  

**Abstract**: Reinforcement Learning with Rubric Rewards (RLRR) is a framework that extends conventional reinforcement learning from human feedback (RLHF) and verifiable rewards (RLVR) by replacing scalar preference signals with structured, multi-dimensional, contextual rubric-based evaluations. However, existing approaches in RLRR are limited to linearly compressing vector rewards into a scalar reward with a fixed weightings, which is sensitive to artificial score design and fails to capture correlations among reward dimensions. To overcome the limitations of reward aggregation, this work proposes Alternating Reinforcement Learning with Rubric Rewards (ARL-RR), a framework that eliminates the need for a fixed scalarization by optimizing one semantic rubric meta-class at a time. Theoretically, we show that reward aggregation induces a variance contraction effect, which helps explain the performance gains. We further introduce a lightweight, search-based adaptation procedure that selects the next meta-class dynamically based on task performance, enabling the policy to emphasize critical objectives and thereby improve the model performance. Empirically, our experiments on the HealthBench dataset with experts annotations demonstrate that ARL-RR uniformly outperforms scalarized methods in both model performance and training efficiency across different model scales (1.7B, 4B, 8B, and 14B). 

---
# Learning to Present: Inverse Specification Rewards for Agentic Slide Generation 

**Authors**: Karthik Ragunath Ananda Kumar, Subrahmanyam Arunachalam  

**Link**: [PDF](https://arxiv.org/pdf/2603.16839)  

**Abstract**: Automated presentation generation remains a challenging task requiring coherent content creation, visual design, and audience-aware communication. This work proposes an OpenEnv-compatible reinforcement learning environment where LLM agents learn to research topics, plan content, and generate professional HTML slide presentations through tool use. We introduce a multi-component reward system combining structural validation, render quality assessment, LLM-based aesthetic scoring, content quality metrics, and an inverse specification reward that measures how faithfully generated slides convey their intended purpose. The inverse specification reward, an "inverse task" where an LLM attempts to recover the original specification from generated slides, provides a holistic quality signal. Our approach fine-tunes Qwen2.5-Coder-7B via GRPO, training only 0.5% of parameters on prompts derived from expert demonstrations collected using Claude Opus 4.6. Experiments on 48 diverse business briefs across six models demonstrate that our fine-tuned 7B model achieves 91.2% of Claude Opus 4.6's quality while improving 33.1% over the base model. The six-model comparison reveals that instruction adherence and tool-use compliance, rather than raw parameter count, determine agentic task performance. We contribute SlideRL, an open-source dataset of 288 multi-turn rollout trajectories across all six models: this https URL Code: this https URL 

---
# Follow the Clues, Frame the Truth: Hybrid-evidential Deductive Reasoning in Open-Vocabulary Multimodal Emotion Recognition 

**Authors**: Yu Liu, Lei Zhang, Haoxun Li, Hanlei Shi, Yuxuan Ding, Leyuan Qu, Taihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16463)  

**Abstract**: Open-Vocabulary Multimodal Emotion Recognition (OV-MER) is inherently challenging due to the ambiguity of equivocal multimodal cues, which often stem from distinct unobserved situational dynamics. While Multimodal Large Language Models (MLLMs) offer extensive semantic coverage, their performance is often bottlenecked by premature commitment to dominant data priors, resulting in suboptimal heuristics that overlook crucial, complementary affective cues across modalities. We argue that effective affective reasoning requires more than surface-level association; it necessitates reconstructing nuanced emotional states by synthesizing multiple evidence-grounded rationales that reconcile these observations from diverse latent perspectives. We introduce HyDRA, a Hybrid-evidential Deductive Reasoning Architecture that formalizes inference as a Propose-Verify-Decide protocol. To internalize this abductive process, we employ reinforcement learning with hierarchical reward shaping, aligning the reasoning trajectories with final task performance to ensure they best reconcile the observed multimodal cues. Systematic evaluations validate our design choices, with HyDRA consistently outperforming strong baselines--especially in ambiguous or conflicting scenarios--while providing interpretable, diagnostic evidence traces. 

---
# Via Negativa for AI Alignment: Why Negative Constraints Are Structurally Superior to Positive Preferences 

**Authors**: Quan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.16417)  

**Abstract**: Recent empirical results have demonstrated that training large language models (LLMs) with negative-only feedback can match or exceed standard reinforcement learning from human feedback (RLHF). Negative Sample Reinforcement achieves parity with PPO on mathematical reasoning; Distributional Dispreference Optimization trains effectively using only dispreferred samples; and Constitutional AI outperforms pure RLHF on harmlessness benchmarks. Yet no unified theoretical account explains why negative signals are so effective. This paper proposes such an account: positive preferences and negative constraints are structurally asymmetric. Positive preferences ("which is better") encode continuously coupled, context-dependent human values that cannot be exhaustively specified -- leading models to learn surface correlates such as agreement with the user (sycophancy). Negative constraints ("what is wrong") encode discrete, finite, independently verifiable prohibitions that can converge to a stable boundary. This asymmetry -- rooted in Popper's falsification logic and the epistemology of negative knowledge -- explains both the sycophancy failure of preference-based RLHF and the surprising effectiveness of negative-signal methods. We argue that alignment research should shift its center of gravity from "learning what humans prefer" to "learning what humans reject," and offer testable predictions for this framework. 

---
# ARISE: Agent Reasoning with Intrinsic Skill Evolution in Hierarchical Reinforcement Learning 

**Authors**: Yu Li, Rui Miao, Zhengling Qi, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2603.16060)  

**Abstract**: The dominant paradigm for improving mathematical reasoning in language models relies on Reinforcement Learning with verifiable rewards. Yet existing methods treat each problem instance in isolation without leveraging the reusable strategies that emerge and accumulate during training. To this end, we introduce ARISE (Agent Reasoning via Intrinsic Skill Evolution), a hierarchical reinforcement learning framework, in which a shared policy operates both to manage skills at high-level and to generate responses at low-level (denoted as a Skills Manager and a Worker, respectively). The Manager maintains a tiered skill library through a dedicated skill generation rollout that performs structured summarization of successful solution traces (after execution), while employing a policy-driven selection mechanism to retrieve relevant skills to condition future rollouts (before execution). A hierarchical reward design guides the co-evolution of reasoning ability and library quality. Experiments on two base models and seven benchmarks spanning both competition mathematics and Omni-MATH show that ARISE consistently outperforms GRPO-family algorithms and memory-augmented baselines, with particularly notable gains on out-of-distribution tasks. Ablation studies confirm that each component contributes to the observed improvements and that library quality and reasoning performance improve in tandem throughout training. Code is available at \href{this https URL}{this https URL}. 

---
