# Jointly Reinforcing Diversity and Quality in Language Model Generations 

**Authors**: Tianjian Li, Yiming Zhang, Ping Yu, Swarnadeep Saha, Daniel Khashabi, Jason Weston, Jack Lanchantin, Tianlu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02534)  

**Abstract**: Post-training of Large Language Models (LMs) often prioritizes accuracy and helpfulness at the expense of diversity. This creates a tension: while post-training improves response quality, it also sharpens output distributions and reduces the range of ideas, limiting the usefulness of LMs in creative and exploratory tasks such as brainstorming, storytelling, or problem solving. We address this challenge with Diversity-Aware Reinforcement Learning (DARLING), a framework that jointly optimizes for response quality and semantic diversity. At its core, DARLING introduces a learned partition function to measure diversity beyond surface-level lexical variations. This diversity signal is then combined with a quality reward during online reinforcement learning, encouraging models to generate outputs that are both high-quality and distinct. Experiments across multiple model families and sizes show that DARLING generalizes to two regimes: non-verifiable tasks (instruction following and creative writing) and verifiable tasks (competition math). On five benchmarks in the first setting, DARLING consistently outperforms quality-only RL baselines, producing outputs that are simultaneously of higher quality and novelty. In the second setting, DARLING achieves higher pass@1 (solution quality) and pass@k (solution variety). Most strikingly, explicitly optimizing for diversity catalyzes exploration in online RL, which manifests itself as higher-quality responses. 

---
# GRAM-R$^2$: Self-Training Generative Foundation Reward Models for Reward Reasoning 

**Authors**: Chenglong Wang, Yongyu Mu, Hang Zhou, Yifu Huo, Ziming Zhu, Jiali Zeng, Murun Yang, Bei Li, Tong Xiao, Xiaoyang Hao, Chunliang Zhang, Fandong Meng, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02492)  

**Abstract**: Significant progress in reward modeling over recent years has been driven by a paradigm shift from task-specific designs towards generalist reward models. Despite this trend, developing effective reward models remains a fundamental challenge: the heavy reliance on large-scale labeled preference data. Pre-training on abundant unlabeled data offers a promising direction, but existing approaches fall short of instilling explicit reasoning into reward models. To bridge this gap, we propose a self-training approach that leverages unlabeled data to elicit reward reasoning in reward models. Based on this approach, we develop GRAM-R$^2$, a generative reward model trained to produce not only preference labels but also accompanying reward rationales. GRAM-R$^2$ can serve as a foundation model for reward reasoning and can be applied to a wide range of tasks with minimal or no additional fine-tuning. It can support downstream applications such as response ranking and task-specific reward tuning. Experiments on response ranking, task adaptation, and reinforcement learning from human feedback demonstrate that GRAM-R$^2$ consistently delivers strong performance, outperforming several strong discriminative and generative baselines. 

---
# DCPO: Dynamic Clipping Policy Optimization 

**Authors**: Shihui Yang, Chengfeng Dou, Peidong Guo, Kai Lu, Qiang Ju, Fei Deng, Rihui Xin  

**Link**: [PDF](https://arxiv.org/pdf/2509.02333)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a promising framework for enhancing the reasoning capabilities of large language models. However, existing approaches such as GRPO often suffer from zero gradients. This problem arises primarily due to fixed clipping bounds for token-level probability ratios and the standardization of identical rewards, which can lead to ineffective gradient updates and underutilization of generated responses. In this work, we propose Dynamic Clipping Policy Optimization (DCPO), which introduces a dynamic clipping strategy that adaptively adjusts the clipping bounds based on token-specific prior probabilities to enhance token-level exploration, and a smooth advantage standardization technique that standardizes rewards across cumulative training steps to improve the response-level effective utilization of generated responses. DCPO achieved state-of-the-art performance on four benchmarks based on four different models. In particular, DCPO achieved an Avg@1 of 46.7 under greedy decoding and an Avg@32 of 38.8 under 32 times sampling on the AIME24 benchmark, surpassing both DAPO (36.7/31.6) and GRPO (36.7/32.1) on the Qwen2.5-Math-7B model. On the AIME25 benchmark based on Qwen2.5-14B, DCPO achieves a performance of (23.3/19.0), surpassing GRPO (13.3/10.5) and DAPO (20.0/15.3). Furthermore, DCPO achieved an average 28% improvement in the nonzero advantage over GRPO in four models, doubled the training efficiency over DAPO, and significantly reduced the token clipping ratio by an order of magnitude compared to both GRPO and DAPO, while achieving superior performance. These results highlight DCPO's effectiveness in leveraging generated data more efficiently for reinforcement learning in large language models. 

---
# Reasoning Vectors: Transferring Chain-of-Thought Capabilities via Task Arithmetic 

**Authors**: Mohammad Zbeeb, Hasan Abed Al Kader Hammoud, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2509.01363)  

**Abstract**: Large language models often require costly optimization, such as reinforcement learning, to master complex reasoning tasks. This work demonstrates that reasoning ability, once learned, can be extracted and transferred between models as a compact task vector. We source two publicly available, identically initialized Qwen2.5 models, one fine-tuned with supervised fine-tuning (SFT) and the other with group relative policy optimization (GRPO) on the same dataset. From these, we extract a reasoning vector: $v_{\text{reason}} = \theta_{\text{GRPO}} - \theta_{\text{SFT}}$. We hypothesize that this vector captures the reasoning capability instilled by reinforcement learning while factoring out shared knowledge from the SFT process. When added to compatible instruction-tuned models through simple arithmetic, this vector consistently improves performance across diverse reasoning benchmarks: GSM8K (+4.9%), HumanEval (+4.3%), SciQ (+1.7%), and BigBenchHard (+12.3% for the 1.5B model). The performance improvements persist under adversarial conditions. Conversely, subtracting the vector causes significant performance degradation (-11.8% on GSM8K), demonstrating the vector's strong contribution to the model's reasoning abilities. This work shows how reasoning capabilities, typically developed through expensive training, can be extracted from existing open-source models and reused through simple tensor arithmetic, offering a practical way to enhance models by recycling prior computational investments. 

---
# Modular Techniques for Synthetic Long-Context Data Generation in Language Model Training and Evaluation 

**Authors**: Seganrasan Subramanian, Abhigya Verma  

**Link**: [PDF](https://arxiv.org/pdf/2509.01185)  

**Abstract**: The ability of large language models (LLMs) to process and reason over long textual inputs is critical for a wide range of real-world applications. However, progress in this area is significantly constrained by the absence of high-quality, diverse, and verifiable long-context datasets suitable for both training and evaluation. This work introduces a modular, extensible framework for synthetic long-context data generation via prompt-based interaction with LLMs. The framework supports multiple training and alignment objectives, including Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO). It encompasses four core generation paradigms: multi-turn conversational dialogues, document-grounded input-output pairs, verifiable instruction-response tasks, and long-context reasoning examples. Through templated prompting, a model-agnostic architecture, and metadata-enriched outputs, the proposed approach facilitates scalable, controllable, and purpose-aligned dataset creation for advancing long-context capabilities in LLMs. 

---
# Dream-Coder 7B: An Open Diffusion Language Model for Code 

**Authors**: Zhihui Xie, Jiacheng Ye, Lin Zheng, Jiahui Gao, Jingwei Dong, Zirui Wu, Xueliang Zhao, Shansan Gong, Xin Jiang, Zhenguo Li, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01142)  

**Abstract**: We present Dream-Coder 7B, an open-source discrete diffusion language model for code generation that exhibits emergent any-order generation capabilities. Unlike traditional autoregressive (AR) models that decode strictly left-to-right, Dream-Coder 7B adaptively determines its decoding strategy based on the coding task: sketch-first generation for complex algorithms, left-to-right generation for straightforward completions, and interleaved reasoning generation for code understanding tasks. We adapt a pretrained AR checkpoint to a discrete diffusion frameworks with a continuous-time weighted cross-entropy objective. Our post-training recipe comprises (i) supervised fine-tuning, where we mitigate padding pathologies via random truncation and a padding penalty to improve sample efficiency and stabilize generation; and (ii) reinforcement learning with verifiable rewards over a curated high-quality prompt set drawn from open-source datasets, using a tailored reinforcement learning recipe for diffusion language models. The resulting Dream-Coder 7B Instruct attains 21.4\% pass@1 on LiveCodeBench (2410--2505) and demonstrates competitive performance on HumanEval, MBPP, BigCodeBench, and CRUXEval. We release Dream-Coder-7B and Dream-Coder-7B-Instruct checkpoints, training recipes, preprocessing pipelines, and inference code to facilitate reproducibility and further research. 

---
# We Politely Insist: Your LLM Must Learn the Persian Art of Taarof 

**Authors**: Nikta Gohari Sadr, Sahar Heidariasl, Karine Megerdoomian, Laleh Seyyed-Kalantari, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2509.01035)  

**Abstract**: Large language models (LLMs) struggle to navigate culturally specific communication norms, limiting their effectiveness in global contexts. We focus on Persian taarof, a social norm in Iranian interactions, which is a sophisticated system of ritual politeness that emphasizes deference, modesty, and indirectness, yet remains absent from existing cultural benchmarks. We introduce TaarofBench, the first benchmark for evaluating LLM understanding of taarof, comprising 450 role-play scenarios covering 12 common social interaction topics, validated by native speakers. Our evaluation of five frontier LLMs reveals substantial gaps in cultural competence, with accuracy rates 40-48% below native speakers when taarof is culturally appropriate. Performance varies between interaction topics, improves with Persian-language prompts, and exhibits gender-based asymmetries. We also show that responses rated "polite" by standard metrics often violate taarof norms, indicating the limitations of Western politeness frameworks. Through supervised fine-tuning and Direct Preference Optimization, we achieve 21.8% and 42.3% improvement in model alignment with cultural expectations. Our human study with 33 participants (11 native Persian, 11 heritage, and 11 non-Iranian speakers) forms baselines in varying degrees of familiarity with Persian norms. This work lays the foundation for developing diverse and culturally aware LLMs, enabling applications that better navigate complex social interactions. 

---
# Speaking at the Right Level: Literacy-Controlled Counterspeech Generation with RAG-RL 

**Authors**: Xiaoying Song, Anirban Saha Anik, Dibakar Barua, Pengcheng Luo, Junhua Ding, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01058)  

**Abstract**: Health misinformation spreading online poses a significant threat to public health. Researchers have explored methods for automatically generating counterspeech to health misinformation as a mitigation strategy. Existing approaches often produce uniform responses, ignoring that the health literacy level of the audience could affect the accessibility and effectiveness of counterspeech. We propose a Controlled-Literacy framework using retrieval-augmented generation (RAG) with reinforcement learning (RL) to generate tailored counterspeech adapted to different health literacy levels. In particular, we retrieve knowledge aligned with specific health literacy levels, enabling accessible and factual information to support generation. We design a reward function incorporating subjective user preferences and objective readability-based rewards to optimize counterspeech to the target health literacy level. Experiment results show that Controlled-Literacy outperforms baselines by generating more accessible and user-preferred counterspeech. This research contributes to more equitable and impactful public health communication by improving the accessibility and comprehension of counterspeech to health misinformation. 

---
# RPRO:Ranked Preference Reinforcement Optimization for Enhancing Medical QA and Diagnostic Reasoning 

**Authors**: Chia-Hsuan Hsu, Jun-En Ding, Hsin-Ling Hsu, Feng Liu, Fang-Ming Hung  

**Link**: [PDF](https://arxiv.org/pdf/2509.00974)  

**Abstract**: Medical question answering requires advanced reasoning that integrates domain knowledge with logical inference. However, existing large language models (LLMs) often generate reasoning chains that lack factual accuracy and clinical reliability. We propose Ranked Preference Reinforcement Optimization (RPRO), a novel framework that uniquely combines reinforcement learning with preference-driven reasoning refinement to enhance clinical chain-of-thought (CoT) performance. RPRO differentiates itself from prior approaches by employing task-adaptive reasoning templates and a probabilistic evaluation mechanism that aligns outputs with established clinical workflows, while automatically identifying and correcting low-quality reasoning chains. Unlike traditional pairwise preference methods, RPRO introduces a groupwise ranking optimization based on the Bradley-Terry model and incorporates KL-divergence regularization for stable training. Experiments on PubMedQA and MedQA-USMLE show consistent improvements over strong baselines. Remarkably, our 1.1B parameter model outperforms much larger 7B-13B models, including medical-specialized variants. These findings demonstrate that combining preference optimization with quality-driven refinement offers a scalable and effective approach to building more reliable, clinically grounded medical LLMs. 

---
# Exploring and Mitigating Fawning Hallucinations in Large Language Models 

**Authors**: Zixuan Shangguan, Yanjie Dong, Lanjun Wang, Xiaoyi Fan, Victor C. M. Leung, Xiping Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00869)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional proficiency in language understanding. However, when LLMs align their outputs with deceptive and/or misleading prompts, the generated responses could deviate from the de facto information. Such observations are known as fawning hallucinations, where the model prioritizes alignment with the input's implied perspective over accuracy and truthfulness. In this work, we analyze fawning hallucinations in various natural language processing tasks and tailor the so-termed contrastive decoding method for fawning-hallucination mitigation. Specifically, we design two paradigms to generate corresponding deceptive and/or misleading inputs for the consistent fawning hallucinations induction. Then, we propose the collaborative contrastive decoding (CCD) to handle the fawning hallucinations across different tasks in LLMs. By contrasting the deviation in output distribution between induced and transformed neutral inputs, the proposed CCD can reduce reliance on deceptive and/or misleading information without requiring additional training. Extensive experiments demonstrate that the proposed CCD can effectively mitigate fawning hallucinations and improve the factuality of the generated responses over various tasks. 

---
# Balanced Actor Initialization: Stable RLHF Training of Distillation-Based Reasoning Models 

**Authors**: Chen Zheng, Yiyuan Ma, Yuan Yang, Deyi Liu, Jing Liu, Zuquan Song, Yuxin Song, Cheng Ren, Hang Zhu, Xin Liu, Yiyuan Ma, Siyuan Qiao, Xun Zhou, Liang Xiang, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00309)  

**Abstract**: The development of alignment and reasoning capabilities in large language models has seen remarkable progress through two paradigms: instruction tuning and reinforcement learning from human feedback (RLHF) alignment paradigm, and distillation-based reasoning fine-tuning paradigm. While both approaches prove effective independently, the third paradigm of applying RLHF to distillation-trained models presents significant challenges. Our investigation reveals two critical phenomena that emerge in this paradigm: Sequence Length Collapse, where language generation dramatically reduces during early RLHF training, and the Reward Hockey Stick Curve, featuring severe reward score drops followed by gradual recovery. These instabilities fundamentally compromise the model's alignment and reasoning capabilities. To address these challenges, we propose Balanced Actor Initialization (BAI), a two-stage weighted model merging approach. BAI first merges instruction-following and distillation-based reasoning fine-tuned models, then further combines this intermediate model with the pretrained model to preserve foundational knowledge. Through comprehensive experiments across diverse benchmarks and detailed analysis of training experiments, we demonstrate that BAI resolves Sequence Length Collapse, mitigates the Reward Hockey Stick Curve, and enables continuous sequence length improvement during training. Additionally, our analysis reveals that balanced merging ratios achieve optimal trade-offs between training stability and reasoning capability preservation. Our work provides the effective solution for stable training in this third paradigm, enabling more capable reasoning models that combine distillation efficiency with RLHF alignment. 

---
# Oyster-I: Beyond Refusal -- Constructive Safety Alignment for Responsible Language Models 

**Authors**: Ranjie Duan, Jiexi Liu, Xiaojun Jia, Shiji Zhao, Ruoxi Cheng, Fengxiang Wang, Cheng Wei, Yong Xie, Chang Liu, Defeng Li, Yinpeng Dong, Yichi Zhang, Yuefeng Chen, Chongwen Wang, Xingjun Ma, Xingxing Wei, Yang Liu, Hang Su, Jun Zhu, Xinfeng Li, Yitong Sun, Jie Zhang, Jinzhao Hu, Sha Xu, Yitong Yang, Jialing Tao, Hui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2509.01909)  

**Abstract**: Large language models (LLMs) typically deploy safety mechanisms to prevent harmful content generation. Most current approaches focus narrowly on risks posed by malicious actors, often framing risks as adversarial events and relying on defensive refusals. However, in real-world settings, risks also come from non-malicious users seeking help while under psychological distress (e.g., self-harm intentions). In such cases, the model's response can strongly influence the user's next actions. Simple refusals may lead them to repeat, escalate, or move to unsafe platforms, creating worse outcomes. We introduce Constructive Safety Alignment (CSA), a human-centric paradigm that protects against malicious misuse while actively guiding vulnerable users toward safe and helpful results. Implemented in Oyster-I (Oy1), CSA combines game-theoretic anticipation of user reactions, fine-grained risk boundary discovery, and interpretable reasoning control, turning safety into a trust-building process. Oy1 achieves state-of-the-art safety among open models while retaining high general capabilities. On our Constructive Benchmark, it shows strong constructive engagement, close to GPT-5, and unmatched robustness on the Strata-Sword jailbreak dataset, nearing GPT-o1 levels. By shifting from refusal-first to guidance-first safety, CSA redefines the model-user relationship, aiming for systems that are not just safe, but meaningfully helpful. We release Oy1, code, and the benchmark to support responsible, user-centered AI. 

---
# Reinforced Visual Perception with Tools 

**Authors**: Zetong Zhou, Dongping Chen, Zixian Ma, Zhihan Hu, Mingyang Fu, Sinan Wang, Yao Wan, Zhou Zhao, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2509.01656)  

**Abstract**: Visual reasoning, a cornerstone of human intelligence, encompasses complex perceptual and logical processes essential for solving diverse visual problems. While advances in computer vision have produced powerful models for various perceptual tasks, leveraging these for general visual reasoning remains challenging. Prior work demonstrates that augmenting LLMs with vision models via supervised finetuning improves performance, but faces key limitations such as expensive data generation, reliance on careful data filtering, and poor generalization. To address these issues, we propose ReVPT to enhance multi-modal LLMs' abilities to reason about and use visual tools through reinforcement learning. We introduce a novel RL algorithm based on GRPO, designed to train models to reason with a suite of four visual tools. Through extensive experiments, we show that our method achieves state-of-the-art performance on several perception-heavy benchmarks, including SAT, CV-Bench, BLINK and MMStar, significantly outperforming the supervised and text-based RL finetuning baselines. Notably, Our ReVPT-3B and ReVPT-7B outperform the instruct models by 9.03% and 9.44% on CV-Bench. Finally, we bring to the community new insights on RL-based visual tool-usage through extensive ablations. Our code is available at this https URL. 

---
# Self-Exploring Language Models for Explainable Link Forecasting on Temporal Graphs via Reinforcement Learning 

**Authors**: Zifeng Ding, Shenyang Huang, Zeyu Cao, Emma Kondrup, Zachary Yang, Xingyue Huang, Yuan Sui, Zhangdie Yuan, Yuqicheng Zhu, Xianglong Hu, Yuan He, Farimah Poursafaei, Michael Bronstein, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2509.00975)  

**Abstract**: Forecasting future links is a central task in temporal graph (TG) reasoning, requiring models to leverage historical interactions to predict upcoming ones. Traditional neural approaches, such as temporal graph neural networks, achieve strong performance but lack explainability and cannot be applied to unseen graphs without retraining. Recent studies have begun to explore using large language models (LLMs) for graph reasoning, but most of them are constrained to static graphs or small synthetic TGs and lack the evaluation of the quality of reasoning traces generated by LLMs. In this work, we present Reasoning-Enhanced Learning for Temporal Graphs (ReaL-TG), a reinforcement learning framework that fine-tunes LLMs to perform explainable link forecasting on real-world TGs. ReaL-TG uses outcome-based reward to encourage models to self-explore reasoning strategies from graph structure and to produce explanations that directly justify their predictions. To enable evaluation on LLM-generated reasoning traces, we propose a new evaluation protocol combining ranking metrics with an LLM-as-a-Judge system that assesses both the quality of reasoning and the impact of hallucinations. Experiments with ReaL-TG-4B, obtained by fine-tuning Qwen3-4B under our framework, show that it outperforms much larger frontier LLMs, including GPT-5 mini, on ranking metrics, while producing high-quality explanations confirmed by both the LLM judge and human evaluation. 

---
# Learning to Refine: Self-Refinement of Parallel Reasoning in LLMs 

**Authors**: Qibin Wang, Pu Zhao, Shaohan Huang, Fangkai Yang, Lu Wang, Furu Wei, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00084)  

**Abstract**: To further enhance the ability of Large Language Models (LLMs) to solve complex, multi-step reasoning problems, test-time scaling (TTS) methods have gained widespread attention. Existing approaches such as Best-of-N and majority voting are limited as their performance depends on the quality of candidate responses, making them unable to produce a correct solution when all candidates are incorrect. Introducing an additional model to select the best response also incurs significant deployment costs. To this end, we introduce Generative Self-Refinement (GSR), a novel parallel test-time scaling framework where a unified model first generates a set of candidate responses in parallel and then performs self-refinement to synthesize a new superior solution based on a prompt consisting of the problem and these candidates. However, LLMs struggle to perform refinement effectively when prompted directly. Therefore, we design a hybrid training pipeline by jointly optimizing for two complementary objectives, solving problems directly and refining candidate responses. Experimental results demonstrate that our method achieves state-of-the-art performance across five mathematical benchmarks. We further show that this learned self-refinement skill is a model-agnostic enhancement, robust across different model scales and generalizing to out-of-distribution reasoning tasks. 

---
# Towards Agents That Know When They Don't Know: Uncertainty as a Control Signal for Structured Reasoning 

**Authors**: Josefa Lia Stoisser, Marc Boubnovski Martell, Lawrence Phillips, Gianluca Mazzoni, Lea Mørch Harder, Philip Torr, Jesper Ferkinghoff-Borg, Kaspar Martens, Julien Fauqueur  

**Link**: [PDF](https://arxiv.org/pdf/2509.02401)  

**Abstract**: Large language model (LLM) agents are increasingly deployed in structured biomedical data environments, yet they often produce fluent but overconfident outputs when reasoning over complex multi-table data. We introduce an uncertainty-aware agent for query-conditioned multi-table summarization that leverages two complementary signals: (i) retrieval uncertainty--entropy over multiple table-selection rollouts--and (ii) summary uncertainty--combining self-consistency and perplexity. Summary uncertainty is incorporated into reinforcement learning (RL) with Group Relative Policy Optimization (GRPO), while both retrieval and summary uncertainty guide inference-time filtering and support the construction of higher-quality synthetic datasets.
On multi-omics benchmarks, our approach improves factuality and calibration, nearly tripling correct and useful claims per summary (3.0\(\rightarrow\)8.4 internal; 3.6\(\rightarrow\)9.9 cancer multi-omics) and substantially improving downstream survival prediction (C-index 0.32\(\rightarrow\)0.63). These results demonstrate that uncertainty can serve as a control signal--enabling agents to abstain, communicate confidence, and become more reliable tools for complex structured-data environments. 

---
# SATQuest: A Verifier for Logical Reasoning Evaluation and Reinforcement Fine-Tuning of LLMs 

**Authors**: Yanxiao Zhao, Yaqian Li, Zihao Bo, Rinyoichi Takezoe, Haojia Hui, Mo Guang, Lei Ren, Xiaolin Qin, Kaiwen Long  

**Link**: [PDF](https://arxiv.org/pdf/2509.00930)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated remarkable general reasoning capabilities. However, systematically evaluating and enhancing these reasoning capabilities is challenging due to the lack of controllable and scalable tools for fine-grained analysis. Existing benchmarks and datasets often lack the necessary variable control for multi-dimensional, systematic analysis and training, or have narrow problem types and formats. To address these limitations, we introduce SATQuest, a systematic verifier designed to evaluate and enhance logical reasoning in LLMs by generating diverse, Satisfiability-based logical reasoning problems directly from Conjunctive Normal Form (CNF) instances. SATQuest structures these problems along three orthogonal dimensions: instance scale, problem type, and question format, employing randomized, SAT-based problem generation and objective answer verification via PySAT. This design mitigates memorization issues, allows for nuanced insights into reasoning performance, and enables effective reinforcement fine-tuning. Our extensive evaluation of various LLMs using SATQuest identified significant limitations in their logical reasoning, particularly in generalizing beyond familiar mathematical formats. Furthermore, we show that reinforcement fine-tuning with SATQuest rewards substantially improves targeted task performance and generalizes to more complex instances, while highlighting remaining challenges in cross-format adaptation. Through these demonstrations, we showcase SATQuest's potential as a foundational tool and a valuable starting point for advancing LLM logical reasoning. 

---
# OmniDPO: A Preference Optimization Framework to Address Omni-Modal Hallucination 

**Authors**: Junzhe Chen, Tianshu Zhang, Shiyu Huang, Yuwei Niu, Chao Sun, Rongzhou Zhang, Guanyu Zhou, Lijie Wen, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00723)  

**Abstract**: Recently, Omni-modal large language models (OLLMs) have sparked a new wave of research, achieving impressive results in tasks such as audio-video understanding and real-time environment perception. However, hallucination issues still persist. Similar to the bimodal setting, the priors from the text modality tend to dominate, leading OLLMs to rely more heavily on textual cues while neglecting visual and audio information. In addition, fully multimodal scenarios introduce new challenges. Most existing models align visual or auditory modalities with text independently during training, while ignoring the intrinsic correlations between video and its corresponding audio. This oversight results in hallucinations when reasoning requires interpreting hidden audio cues embedded in video content. To address these challenges, we propose OmniDPO, a preference-alignment framework designed to mitigate hallucinations in OLLMs. Specifically, OmniDPO incorporates two strategies: (1) constructing text-preference sample pairs to enhance the model's understanding of audio-video interactions; and (2) constructing multimodal-preference sample pairs to strengthen the model's attention to visual and auditory information. By tackling both challenges, OmniDPO effectively improves multimodal grounding and reduces hallucination. Experiments conducted on two OLLMs demonstrate that OmniDPO not only effectively mitigates multimodal hallucinations but also significantly enhances the models' reasoning capabilities across modalities. All code and datasets will be released upon paper acceptance. 

---
# Instruction-Level Weight Shaping: A Framework for Self-Improving AI Agents 

**Authors**: Rimom Costa  

**Link**: [PDF](https://arxiv.org/pdf/2509.00251)  

**Abstract**: Large language models (LLMs) are fluent but largely static after pre-training; new or shifting knowledge is typically added with retrieval-augmented generation (RAG) or fine-tuning. RAG raises latency and engineering overhead and often fails to integrate facts; prompt engineering is brittle and can conflict with prior knowledge; fine-tuning is costly and risks catastrophic forgetting. We propose Instruction-Level Weight Shaping (ILWS): curated system instructions act as external, auditable pseudo-parameters updated after each session via reflection and user feedback. A Reflection Engine inspects conversation traces, diagnoses reasoning successes and failures, and proposes typed deltas $\Delta K=(\Delta S,\Delta U,\Delta T)$ over instructions, user preferences, and tools. Deltas are version-controlled, evaluated with a sliding window of 1-5 star ratings, auto-repaired on first failure, and rolled back on repeated failure. When an edit budget crosses a threshold, the agent compiles a rating-weighted synthetic set and distills matured instruction-space gains into parameters, converting prompt-space improvements into weight-space without downtime. ILWS makes explicit the low-rank shaping induced by context in transformer blocks, preserves governance, and removes per-call retrieval. In enterprise support it increased throughput 2.4-5.0x and cut audited hallucinations by about 80% versus a frozen baseline. In an Adobe Commerce Cloud proof of concept "L0 Support", it achieved 4-5x more tickets per hour and about 80% lower time per ticket, with autonomous instruction updates and optional tool synthesis. Because ILWS operates at the instruction layer until controlled distillation, it generalizes to dynamic domains (legal, medical, engineering) requiring adaptive reasoning, tool creation, and low-latency deployment. 

---
# Know When to Explore: Difficulty-Aware Certainty as a Guide for LLM Reinforcement Learning 

**Authors**: Ang Li, Zhihang Yuan, Yang Zhang, Shouda Liu, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00125)  

**Abstract**: Reinforcement Learning with Verifiable Feedback (RLVF) has become a key technique for enhancing the reasoning abilities of Large Language Models (LLMs). However, its reliance on sparse, outcome based rewards, which only indicate if a final answer is correct or not, fails to provide granular guidance on the reasoning process itself. This limitation hinders efficient learning, as the model cannot distinguish between high quality and inefficient solutions, nor can it learn effectively from different types of failures. To address this, we observe that an LLMs self-certainty often correlates with task difficulty and solution quality. We introduce Difficulty Aware Certainty guided Exploration (DACE), a novel RL algorithm that leverages this insight to dynamically balance the exploration exploitation trade-off. DACE assesses task difficulty online based on the policys success rate. It then uses this signal to modulate an intrinsic reward: for difficult tasks where the model is struggling, DACE encourages exploration by penalizing high certainty; for easier tasks, it encourages learning efficiency by rewarding high certainty. Experiments on challenging mathematical reasoning benchmarks (AIME, MATH) show that DACE significantly outperforms strong baselines. The DACE-trained models not only achieve higher accuracy but also demonstrate more robust performance when scaling test-time compute, validating that our adaptive approach fosters effective exploration without sacrificing precision. 

---
# Baichuan-M2: Scaling Medical Capability with Large Verifier System 

**Authors**: Baichuan-M2 Team, Chengfeng Dou, Chong Liu, Fan Yang, Fei Li, Jiyuan Jia, Mingyang Chen, Qiang Ju, Shuai Wang, Shunya Dang, Tianpeng Li, Xiangrong Zeng, Yijie Zhou, Chenzheng Zhu, Da Pan, Fei Deng, Guangwei Ai, Guosheng Dong, Hongda Zhang, Jinyang Tai, Jixiang Hong, Kai Lu, Linzhuang Sun, Peidong Guo, Qian Ma, Rihui Xin, Shihui Yang, Shusen Zhang, Yichuan Mo, Zheng Liang, Zhishou Zhang, Hengfu Cui, Zuyi Zhu, Xiaochuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02208)  

**Abstract**: As large language models (LLMs) advance in conversational and reasoning capabilities, their practical application in healthcare has become a critical research focus. However, there is a notable gap between the performance of medical LLMs on static benchmarks such as USMLE and their utility in real-world clinical decision-making. This discrepancy arises because traditional exams fail to capture the dynamic, interactive nature of medical consultations. To address this challenge, we introduce a novel dynamic verification framework that moves beyond static answer verifier, establishing a large-scale, high-fidelity interactive reinforcement learning system. Our framework comprises two key components: a Patient Simulator that creates realistic clinical environments using de-identified medical records, and a Clinical Rubrics Generator that dynamically produces multi-dimensional evaluation metrics. Building on this foundation, we develop Baichuan-M2, a 32B-parameter medical augmented reasoning model trained through a multi-stage reinforcement learning strategy with an improved Group Relative Policy Optimization (GRPO) algorithm. Evaluated on HealthBench, Baichuan-M2 outperforms all other open-source models and most advanced closed-source counterparts, achieving a score above 32 on the challenging HealthBench Hard benchmark-previously exceeded only by GPT-5. Our work demonstrates that robust dynamic verifier system is essential for aligning LLM capabilities with practical clinical applications, establishing a new Pareto front in the performance-parameter trade-off for medical AI deployment. 

---
