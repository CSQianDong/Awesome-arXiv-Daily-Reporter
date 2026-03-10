# Aligning to Illusions: Choice Blindness in Human and AI Feedback 

**Authors**: Wenbin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.08412)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) assumes annotator preferences reflect stable internal states. We challenge this through three experiments spanning the preference pipeline. In a human choice blindness study, 91% of surreptitiously swapped preferences go undetected, extending choice blindness to third-person evaluative comparison of unfamiliar text. Testing fifteen LLM judges as potential replacements, we find detection relies on shallow text matching rather than genuine self-monitoring: removing prior reasoning from context causes blindness to surge from near-zero to over 50%, while explicit social pressure induces near-universal compliance. In a dose-response experiment across two architectures from 86M to 2B parameters, one-sixth to one-third of labels must be corrupted before the reward signal halves, yet standard pairwise accuracy remains virtually unchanged. A Best-of-N evaluation confirms this translates to downstream policy degradation: at 50% corruption, reward-guided selection produces no improvement over random sampling, while the proxy model reports monotonically increasing scores. Together, these results reveal a preference construction problem: the signal entering RLHF is shaped by elicitation context in ways that neither human metacognition, LLM self-monitoring, nor standard evaluation metrics can detect. 

---
# Revealing Behavioral Plasticity in Large Language Models: A Token-Conditional Perspective 

**Authors**: Liyuan Mao, Le Yu, Jing Zhou, Chujie Zheng, Bowen Yu, Chang Gao, Shixuan Liu, An Yang, Weinan Zhang, JunYang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.08398)  

**Abstract**: In this work, we reveal that Large Language Models (LLMs) possess intrinsic behavioral plasticity-akin to chameleons adapting their coloration to environmental cues-that can be exposed through token-conditional generation and stabilized via reinforcement learning. Specifically, by conditioning generation on carefully selected token prefixes sampled from responses exhibiting desired behaviors, LLMs seamlessly adapt their behavioral modes at inference time (e.g., switching from step-by-step reasoning to direct answering) without retraining. Based on this insight, we propose Token-Conditioned Reinforcement Learning (ToCoRL), a principled framework that leverages RL to internalize this chameleon-like plasticity, transforming transient inference-time adaptations into stable and learnable behavioral patterns. ToCoRL guides exploration with token-conditional generation and keep enhancing exploitation, enabling emergence of appropriate behaviors. Extensive experiments show that ToCoRL enables precise behavioral control without capability degradation. Notably, we show that large reasoning models, while performing strongly on complex mathematics, can be effectively adapted to excel at factual question answering, which was a capability previously hindered by their step-by-step reasoning patterns. 

---
# RexDrug: Reliable Multi-Drug Combination Extraction through Reasoning-Enhanced LLMs 

**Authors**: Zhijun Wang, Ling Luo, Dinghao Pan, Huan Zhuang, Lejing Yu, Yuanyuan Sun, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.08166)  

**Abstract**: Automated Drug Combination Extraction (DCE) from large-scale biomedical literature is crucial for advancing precision medicine and pharmacological research. However, existing relation extraction methods primarily focus on binary interactions and struggle to model variable-length n-ary drug combinations, where complex compatibility logic and distributed evidence need to be considered. To address these limitations, we propose RexDrug, an end-to-end reasoning-enhanced relation extraction framework for n-ary drug combination extraction based on large language models. RexDrug adopts a two-stage training strategy. First, a multi-agent collaborative mechanism is utilized to automatically generate high-quality expert-like reasoning traces for supervised fine-tuning. Second, reinforcement learning with a multi-dimensional reward function specifically tailored for DCE is applied to further refine reasoning quality and extraction accuracy. Extensive experiments on the DrugComb dataset show that RexDrug consistently outperforms state-of-the-art baselines for n-ary extraction. Additional evaluation on the DDI13 corpus confirms its generalizability to binary drugdrug interaction tasks. Human expert assessment and automatic reasoning metrics further indicates that RexDrug produces coherent medical reasoning while accurately identifying complex therapeutic regimens. These results establish RexDrug as a scalable and reliable solution for complex biomedical relation extraction from unstructured text. The source code and data are available at this https URL 

---
# Toward Robust LLM-Based Judges: Taxonomic Bias Evaluation and Debiasing Optimization 

**Authors**: Hongli Zhou, Hui Huang, Rui Zhang, Kehai Chen, Bing Xu, Conghui Zhu, Tiejun Zhao, Muyun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08091)  

**Abstract**: Large language model (LLM)-based judges are widely adopted for automated evaluation and reward modeling, yet their judgments are often affected by judgment biases. Accurately evaluating these biases is essential for ensuring the reliability of LLM-based judges. However, existing studies typically investigate limited biases under a single judge formulation, either generative or discriminative, lacking a comprehensive evaluation. To bridge this gap, we propose JudgeBiasBench, a benchmark for systematically quantifying biases in LLM-based judges. JudgeBiasBench defines a taxonomy of judgment biases across 4 dimensions, and constructs bias-augmented evaluation instances through a controlled bias injection pipeline, covering 12 representative bias types. We conduct extensive experiments across both generative and discriminative judges, revealing that current judges exhibit significant and diverse bias patterns that often compromise the reliability of automated evaluation. To mitigate judgment bias, we propose bias-aware training that explicitly incorporates bias-related attributes into the training process, encouraging judges to disentangle task-relevant quality from bias-correlated cues. By adopting reinforcement learning for generative judges and contrastive learning for discriminative judges, our methods effectively reduce judgment biases while largely preserving general evaluation capability. 

---
# TableMind++: An Uncertainty-Aware Programmatic Agent for Tool-Augmented Table Reasoning 

**Authors**: Mingyue Cheng, Shuo Yu, Chuang Jiang, Xiaoyu Tao, Qingyang Mao, Jie Ouyang, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.07528)  

**Abstract**: Table reasoning requires models to jointly perform semantic understanding and precise numerical operations. Most existing methods rely on a single-turn reasoning paradigm over tables which suffers from context overflow and weak numerical sensitivity. To address these limitations, we previously proposed TableMind as a tuning-based autonomous programmatic agent that simulates human-like interaction within a lightweight large language model (LLM). TableMind internalizes planning, action, and reflection through a two-stage training strategy involving supervised fine-tuning (SFT) on filtered high-quality data and reinforcement learning (RL) via a multi-perspective reward and the Rank-Aware Policy Optimization (RAPO) algorithm. While TableMind establishes a solid foundation for programmatic agents, the inherent stochasticity of LLMs remains a critical challenge that leads to hallucinations. In this paper, we extend this foundation to TableMind++ by introducing a novel uncertainty-aware inference framework to mitigate hallucinations. Specifically, we propose memory-guided plan pruning to retrieve historical trajectories for validating and filtering out logically flawed plans to address epistemic uncertainty. To ensure execution precision, we introduce confidence-based action refinement which monitors token-level probabilities to detect and self-correct syntactic noise for aleatoric uncertainty mitigation. Finally, we employ dual-weighted trajectory aggregation to synthesize a robust consensus from multiple reasoning paths. Extensive experiments on diverse benchmarks demonstrate that TableMind++ consistently outperforms previous baselines and proprietary models to validate the effectiveness of integrating autonomous training with uncertainty quantification. Our code is available. 

---
# Hit-RAG: Learning to Reason with Long Contexts via Preference Alignment 

**Authors**: Junming Liu, Yuqi Li, Shiping Wen, Zhigang Zeng, Tingwen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07023)  

**Abstract**: Despite the promise of Retrieval-Augmented Generation in grounding Multimodal Large Language Models with external knowledge, the transition to extensive contexts often leads to significant attention dilution and reasoning hallucinations. The surge in information density causes critical evidence to be submerged by voluminous noise, which complicates the discernment of relevant fragments within a dense input. In this paper, we propose \textbf{Hit-RAG}, a multi-stage preference alignment framework designed to resolve these cognitive bottlenecks through a progressive optimization pipeline. Our approach systematically refines the utilization of external evidence via three distinct stages. First, Supervised Fine-tuning establishes baseline context awareness to minimize information neglect. Next, Discriminative Preference Alignment enhances robustness against misleading distractors. Finally, Group-Relative Policy Optimization stabilizes logical synthesis to prevent reasoning collapse. Extensive evaluations on eight benchmarks demonstrate that Hit-RAG consistently yields substantial performance gains, enabling models to bridge the gap between context acquisition and accurate reasoning while surpassing much larger counterparts in long-context scenarios. 

---
# Can Safety Emerge from Weak Supervision? A Systematic Analysis of Small Language Models 

**Authors**: Punyajoy Saha, Sudipta Halder, Debjyoti Mondal, Subhadarshi Panda  

**Link**: [PDF](https://arxiv.org/pdf/2603.07017)  

**Abstract**: Safety alignment is critical for deploying large language models (LLMs) in real-world applications, yet most existing approaches rely on large human-annotated datasets and static red-teaming benchmarks that are costly, difficult to scale, and slow to adapt to evolving model behaviors. Moreover, overly conservative safety mechanisms can reduce model usefulness by rejecting sensitive but legitimate queries. We introduce Self-MOA (Self Multi-Objective Alignment), a fully automated framework for aligning small language models using weak supervision from automated evaluator models. Self-MOA operates as a closed loop that dynamically generates model-specific red team prompts, constructs preference data from model-generated responses, and aligns models via multi-objective preference optimization to jointly optimize for safety and helpfulness. Across multiple small language models and safety benchmarks, Self-MOA achieves a 12.41\% improvement in safety while preserving helpfulness, using as little as 11 times less training data than human-supervised alignment baselines. These results demonstrate that adaptive, automated alignment can reduce the dependence on static, human-curated safety pipelines in resource-constrained settings. 

---
# A Dynamic Self-Evolving Extraction System 

**Authors**: Moin Amin-Naseri, Hannah Kim, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2603.06915)  

**Abstract**: The extraction of structured information from raw text is a fundamental component of many NLP applications, including document retrieval, ranking, and relevance estimation. High-quality extractions often require domain-specific accuracy, up-to-date understanding of specialized taxonomies, and the ability to incorporate emerging jargon and rare outliers. In many domains--such as medical, legal, and HR--the extraction model must also adapt to shifting terminology and benefit from explicit reasoning over structured knowledge. We propose DySECT, a Dynamic Self-Evolving Extraction and Curation Toolkit, which continually improves as it is used. The system incrementally populates a versatile, self-expanding knowledge base (KB) with triples extracted by the LLM. The KB further enriches itself through the integration of probabilistic knowledge and graph-based reasoning, gradually accumulating domain concepts and relationships. The enriched KB then feeds back into the LLM extractor via prompt tuning, sampling of relevant few-shot examples, or fine-tuning using KB-derived synthetic data. As a result, the system forms a symbiotic closed-loop cycle in which extraction continuously improves knowledge, and knowledge continuously improves extraction. 

---
# Agentic Critical Training 

**Authors**: Weize Liu, Minghui Liu, Sy-Tuyen Ho, Souradip Chakraborty, Xiyao Wang, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08706)  

**Abstract**: Training large language models (LLMs) as autonomous agents often begins with imitation learning, but it only teaches agents what to do without understanding why: agents never contrast successful actions against suboptimal alternatives and thus lack awareness of action quality. Recent approaches attempt to address this by introducing self-reflection supervision derived from contrasts between expert and alternative actions. However, the training paradigm fundamentally remains imitation learning: the model imitates pre-constructed reflection text rather than learning to reason autonomously. We propose Agentic Critical Training (ACT), a reinforcement learning paradigm that trains agents to identify the better action among alternatives. By rewarding whether the model's judgment is correct, ACT drives the model to autonomously develop reasoning about action quality, producing genuine self-reflection rather than imitating it. Across three challenging agent benchmarks, ACT consistently improves agent performance when combined with different post-training methods. It achieves an average improvement of 5.07 points over imitation learning and 4.62 points over reinforcement learning. Compared to approaches that inject reflection capability through knowledge distillation, ACT also demonstrates clear advantages, yielding an average improvement of 2.42 points. Moreover, ACT enables strong out-of-distribution generalization on agentic benchmarks and improves performance on general reasoning benchmarks without any reasoning-specific training data, highlighting the value of our method. These results suggest that ACT is a promising path toward developing more reflective and capable LLM agents. 

---
# Chart-RL: Generalized Chart Comprehension via Reinforcement Learning with Verifiable Rewards 

**Authors**: Xin Zhang, Xingyu Li, Rongguang Wang, Ruizhong Miao, Zheng Wang, Dan Roth, Chenyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.06958)  

**Abstract**: Accurate chart comprehension represents a critical challenge in advancing multimodal learning systems, as extensive information is compressed into structured visual representations. However, existing vision-language models (VLMs) frequently struggle to generalize on unseen charts because it requires abstract, symbolic, and quantitative reasoning over structured visual representations. In this work, we introduce Chart-RL, an effective reinforcement learning (RL) method that employs mathematically verifiable rewards to enhance chart question answering in VLMs. Our experiments demonstrate that Chart-RL consistently outperforms supervised fine-tuning (SFT) across different chart understanding benchmarks, achieving relative improvements of 16.7% on MutlChartQA, and 11.5% on ChartInsights. We conduct robustness analysis, where Chart-RL achieves enhanced performance in 18 of 25 perturbed chart categories, demonstrating strong consistency and reasoning capability across visual variations. Furthermore, we demonstrate that task difficulty and inherent complexity are more critical than data quantity in RL training. For instance, Chart-RL trained on merely 10 complex chart-query examples significantly outperforms models trained on over 6,000 simple examples. Additionally, training on challenging reasoning tasks not only improves in-domain generalization relative to simpler tasks, but also facilitate strong transfer to out-of-domain visual mathematical problems. 

---
# Know When You're Wrong: Aligning Confidence with Correctness for LLM Error Detection 

**Authors**: Xie Xiaohu, Liu Xiaohu, Yao Benjamin  

**Link**: [PDF](https://arxiv.org/pdf/2603.06604)  

**Abstract**: As large language models (LLMs) are increasingly deployed in critical decision-making systems, the lack of reliable methods to measure their uncertainty presents a fundamental trustworthiness risk. We introduce a normalized confidence score based on output anchor token probabilities: classification labels for structured tasks and self-evaluation responses (Yes/No) for open-ended generation. This enables direct detection of errors and hallucinations with minimal overhead and without external validation. We make three key contributions. First, we propose a normalized confidence score and self-evaluation framework that exposes reliable confidence estimates for error detection across seven diverse benchmark tasks and five LLMs of varying architectures and sizes. Second, our theoretical analysis reveals that supervised fine-tuning (SFT) yields well-calibrated confidence through maximum-likelihood estimation, whereas reinforcement learning methods (PPO, GRPO) and DPO induce overconfidence via reward exploitation. Third, we propose post-RL SFT with self-distillation to restore confidence reliability in RL-trained models. Empirical results demonstrated that SFT improved average confidence-correctness AUROC from 0.806 to 0.879 and reduced calibration error from 0.163 to 0.034 on Qwen3-4B, while GRPO and DPO degraded confidence reliability. We demonstrated practical value through adaptive retrieval-augmented generation (RAG) that selectively retrieves context when the model lacks confidence, using only 58\% of retrieval operations to recover 95\% of the maximum achievable accuracy gain on TriviaQA 

---
# RetroAgent: From Solving to Evolving via Retrospective Dual Intrinsic Feedback 

**Authors**: Xiaoying Zhang, Zichen Liu, Yipeng Zhang, Xia Hu, Wenqi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2603.08561)  

**Abstract**: Large language model (LLM)-based agents trained with reinforcement learning (RL) have shown strong potential on complex interactive tasks. However, standard RL paradigms favor static problem-solving over continuous adaptation: agents often converge to suboptimal strategies due to insufficient exploration, while learned knowledge remains implicit within parameters rather than explicitly retrievable, limiting effective experiential learning. To address these limitations, we introduce RetroAgent, an online RL framework that empowers agents to master complex interactive environments not just by solving, but by evolving. Concretely, RetroAgent features a hindsight self-reflection mechanism that produces dual intrinsic feedback: (1) intrinsic numerical feedback that that tracks incremental subtask completion relative to prior attempts, rewarding promising explorations, and (2) intrinsic language feedback that distills reusable lessons into a memory buffer, retrieved via our proposed Similarity & Utility-Aware Upper Confidence Bound (SimUtil-UCB) strategy balancing relevance, utility, and exploration to effectively leverage past experiences. Extensive experiments on two model families across four challenging agentic tasks demonstrate that RetroAgent significantly outperforms existing methods, achieving state-of-the-art results -- e.g., surpassing Group Relative Policy Optimization (GRPO)-trained agents by +18.3% on ALFWorld, +15.4% on WebShop, +27.1% on Sokoban, and +8.9% on MineSweeper -- while exhibiting strong test-time adaptation and generalization to out-of-distribution scenarios. 

---
# Shorter Thoughts, Same Answers: Difficulty-Scaled Segment-Wise RL for CoT Compression 

**Authors**: Ye Tian, Aijun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07598)  

**Abstract**: Chain-of-thought (CoT) improves reasoning reliability but increases token cost, motivating post-training compression of explicit reasoning traces. However, the shortest sufficient reasoning is not universal: it depends on difficulty, model capacity, and training state, making fixed length targets brittle. In practice, naive RL-based compression can also undesirably shorten the user-facing answer, because a single completion-level learning signal leaks across the think/answer boundary. We propose Difficulty-Scaled Segment-Wise GRPO (DSS-GRPO), which decomposes returns into think and answer components, computes group-relative advantages per segment, and routes them with hard token masks so compression updates act only on think while answer alignment acts only on answer. DSS-GRPO uses prompt-wise within-group shaping and difficulty-aware scaling to encourage concise reasoning without collapsing answer behavior. 

---
# $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving 

**Authors**: Pinzheng Wang, Shuli Xu, Juntao Li, Yu Luo, Dong Li, Jianye Hao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07197)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has shown promise in enhancing the reasoning performance of large language models (LLMs) by increasing test-time compute. However, even after extensive RLVR training, such models still tend to generate unnecessary and low-quality steps in their chain-of-thought (CoT), leading to inefficient overthinking and lower answer quality. We show that when the initial direction or quality of the CoT is suboptimal, the model often fails to reach the correct answer, even after generating several times more tokens than when the initial CoT is well-initialized. To this end, we introduce Reinforcement Learning with Re-solving (Re$^2$), in which LLMs learn to flexibly abandon unproductive reasoning paths and restart the solution process when necessary, rather than always committing to a final answer. Re$^2$ applies pure reinforcement learning without any preliminary supervised fine-tuning, successfully amplifying the rare redo behavior in vanilla models from only 0.5% to over 30%. This leads to substantial performance gains over standard RLVR under the same training compute budget, and also demonstrates notable improvements in test-time performance as the number of samples increases. 

---
# Best-of-Tails: Bridging Optimism and Pessimism in Inference-Time Alignment 

**Authors**: Hsiang Hsu, Eric Lei, Chun-Fu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.06797)  

**Abstract**: Inference-time alignment effectively steers large language models (LLMs) by generating multiple candidates from a reference model and selecting among them with an imperfect reward model. However, current strategies face a fundamental dilemma: ``optimistic'' approaches like Best-of-$N$ suffer from reward hacking, while ``pessimistic'' regularized methods often stifle the exploration needed to discover high-quality responses. In this work, we formalize this trade-off through the lens of regret minimization, demonstrating that the optimal strategy depends critically on the tail behavior of the reward distribution. We show theoretically that light-tailed regimes favor optimism to unearth high-quality outliers, whereas heavy-tailed regimes require pessimism to guard against reward mis-calibration in the extremes. Guided by this insight, we introduce Best-of-Tails (BoT), an adaptive inference-time alignment framework that uses Tsallis divergence as a tunable regularizer to provide a finer granularity of interpolation between these extremes. BoT uses the Hill estimator to characterize reward-tail heaviness on a per-prompt basis and dynamically adjusts its selection rule to balance exploration gains against alignment error. Across math, multiple-choice reasoning, and human-preference evaluations, BoT improves alignment performance across a range of reference and reward model configurations relative to fixed-strategy baselines. 

---
# Disentangling Reasoning in Large Audio-Language Models for Ambiguous Emotion Prediction 

**Authors**: Xiaofeng Yu, Jiaheng Dong, Jean Honorio, Abhirup Ghosh, Hong Jia, Ting Dang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08230)  

**Abstract**: Speech emotion recognition plays an important role in various applications. However, most existing approaches predict a single emotion label, oversimplifying the inherently ambiguous nature of human emotional expression. Recent large audio-language models show promise in generating richer outputs, but their reasoning ability for ambiguous emotional understanding remains limited. In this work, we reformulate ambiguous emotion recognition as a distributional reasoning problem and present the first systematic study of ambiguity-aware reasoning in LALMs. Our framework comprises two complementary components: an ambiguity-aware objective that aligns predictions with human perceptual distributions, and a structured ambiguity-aware chain-of-thought supervision that guides reasoning over emotional cues. Experiments on IEMOCAP and CREMA-D demonstrate consistent improvements across SFT, DPO, and GRPO training strategies. 

---
# DARC: Disagreement-Aware Alignment via Risk-Constrained Decoding 

**Authors**: Mingxi Zou, Jiaxiang Chen, Junfan Li, Langzhang Liang, Qifan Wang, Xu Yinghui, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.08145)  

**Abstract**: Preference-based alignment methods (e.g., RLHF, DPO) typically optimize a single scalar objective, implicitly averaging over heterogeneous human preferences. In practice, systematic annotator and user-group disagreement makes mean-reward maximization brittle and susceptible to proxy over-optimization. We propose **Disagreement-Aware Alignment via Risk-Constrained Decoding (DARC)**, a retraining-free inference-time method that frames response selection as distributionally robust, risk-sensitive decision making. Given multiple preference samples or scalable disagreement proxies, DARC reranks candidates by maximizing a *KL-robust (entropic)* satisfaction objective, and provides simple deployment controls that cap or penalize the corresponding entropic risk premium relative to the mean, enabling explicit risk budgets without retraining. We provide theoretical characterization linking this decoding rule to principled pessimism and KL-based distributionally robust optimization. Experiments on alignment benchmarks show that DARC reduces disagreement and tail risk while maintaining competitive average quality under noisy, heterogeneous feedback. 

---
# UnSCAR: Universal, Scalable, Controllable, and Adaptable Image Restoration 

**Authors**: Debabrata Mandal, Soumitri Chattopadhyay, Yujie Wang, Marc Niethammer, Praneeth Chakravarthula  

**Link**: [PDF](https://arxiv.org/pdf/2603.07406)  

**Abstract**: Universal image restoration aims to recover clean images from arbitrary real-world degradations using a single inference model. Despite significant progress, existing all-in-one restoration networks do not scale to multiple degradations. As the number of degradations increases, training becomes unstable, models grow excessively large, and performance drops across both seen and unseen domains. In this work, we show that scaling universal restoration is fundamentally limited by interference across degradations during joint learning, leading to catastrophic task forgetting. To address this challenge, we introduce a unified inference pipeline with a multi-branch mixture-of-experts architecture that decomposes restoration knowledge across specialized task-adaptable experts. Our approach enables scalable learning (over sixteen degradations), adapts and generalizes robustly to unseen domains, and supports user-controllable restoration across degradations. Beyond achieving superior performance across benchmarks, this work establishes a new design paradigm for scalable and controllable universal image restoration. 

---
# Diffusion Controller: Framework, Algorithms and Parameterization 

**Authors**: Tong Yang, Moonkyung Ryu, Chih-Wei Hsu, Guy Tennenholtz, Yuejie Chi, Craig Boutilier, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2603.06981)  

**Abstract**: Controllable diffusion generation often relies on various heuristics that are seemingly disconnected without a unified understanding. We bridge this gap with Diffusion Controller (DiffCon), a unified control-theoretic view that casts reverse diffusion sampling as state-only stochastic control within (generalized) linearly-solvable Markov Decision Processes (LS-MDPs). Under this framework, control acts by reweighting the pretrained reverse-time transition kernels, balancing terminal objectives against an $f$-divergence cost. From the resulting optimality conditions, we derive practical reinforcement learning methods for diffusion fine-tuning: (i) f-divergence-regularized policy-gradient updates, including a PPO-style rule, and (ii) a regularizer-determined reward-weighted regression objective with a minimizer-preservation guarantee under the Kullback-Leibler (KL) divergence. The LS-MDP framework further implies a principled model form: the optimal score decomposes into a fixed pretrained baseline plus a lightweight control correction, motivating a side-network parameterization conditioned on exposed intermediate denoising outputs, enabling effective gray-box adaptation with a frozen backbone. Experiments on Stable Diffusion v1.4 across supervised and reward-driven finetuning show consistent gains in preference-alignment win rates and improved quality-efficiency trade-offs versus gray-box baselines and even the parameter-efficient white-box adapter LoRA. 

---
# PaLMR: Towards Faithful Visual Reasoning via Multimodal Process Alignment 

**Authors**: Yantao Li, Qiang Hui, Chenyang Yan, Kanzhi Cheng, Fang Zhao, Chao Tan, Huanling Gao, Jianbing Zhang, Kai Wang, Xinyu Dai, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2603.06652)  

**Abstract**: Reinforcement learning has recently improved the reasoning ability of Large Language Models and Multimodal LLMs, yet prevailing reward designs emphasise final-answer correctness and consequently tolerate process hallucinations--cases where models reach the right answer while misperceiving visual evidence. We address this process-level misalignment with PaLMR, a framework that aligns not only outcomes but also the reasoning process itself. PaLMR comprises two complementary components: a perception-aligned data layer that constructs process-aware reasoning data with structured pseudo-ground-truths and verifiable visual facts, and a process-aligned optimisation layer that constructs a hierarchical reward fusion scheme with a process-aware scoring function to encourage visually faithful chains-of-thought and improve training stability. Experiments on Qwen2.5-VL-7B show that our approach substantially reduces reasoning hallucinations and improves visual reasoning fidelity, achieving state-of-the-art results on HallusionBench while maintaining strong performance on MMMU, MathVista, and MathVerse. These findings indicate that PaLMR offers a principled and practical route to process-aligned multimodal reasoning, advancing the reliability and interpretability of MLLMs. 

---
