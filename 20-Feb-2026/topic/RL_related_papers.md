# Learning to Stay Safe: Adaptive Regularization Against Safety Degradation during Fine-Tuning 

**Authors**: Jyotin Goel, Souvik Maji, Pratik Mazumder  

**Link**: [PDF](https://arxiv.org/pdf/2602.17546)  

**Abstract**: Instruction-following language models are trained to be helpful and safe, yet their safety behavior can deteriorate under benign fine-tuning and worsen under adversarial updates. Existing defenses often offer limited protection or force a trade-off between safety and utility. We introduce a training framework that adapts regularization in response to safety risk, enabling models to remain aligned throughout fine-tuning. To estimate safety risk at training time, we explore two distinct approaches: a judge-based Safety Critic that assigns high-level harm scores to training batches, and an activation-based risk predictor built with a lightweight classifier trained on intermediate model activations to estimate harmful intent. Each approach provides a risk signal that is used to constrain updates deemed higher risk to remain close to a safe reference policy, while lower-risk updates proceed with standard training. We empirically verify that harmful intent signals are predictable from pre-generation activations and that judge scores provide effective high-recall safety guidance. Across multiple model families and attack scenarios, adaptive regularization with either risk estimation approach consistently lowers attack success rate compared to standard fine-tuning, preserves downstream performance, and adds no inference-time cost. This work demonstrates a principled mechanism for maintaining safety without sacrificing utility. 

---
# References Improve LLM Alignment in Non-Verifiable Domains 

**Authors**: Kejian Shi, Yixin Liu, Peifeng Wang, Alexander R. Fabbri, Shafiq Joty, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2602.16802)  

**Abstract**: While Reinforcement Learning with Verifiable Rewards (RLVR) has shown strong effectiveness in reasoning tasks, it cannot be directly applied to non-verifiable domains lacking ground-truth verifiers, such as LLM alignment. In this work, we investigate whether reference-guided LLM-evaluators can bridge this gap by serving as soft "verifiers". First, we design evaluation protocols that enhance LLM-based evaluators for LLM alignment using reference outputs. Through comprehensive experiments, we show that a reference-guided approach substantially improves the accuracy of less capable LLM-judges using references from frontier models; stronger LLM-judges can also be enhanced by high-quality (i.e., human-written) references. Building on these improved judges, we demonstrate the utility of high-quality references in alignment tuning, where LLMs guided with references are used as judges to self-improve. We show that reference-guided self-improvement yields clear gains over both direct SFT on reference outputs and self-improvement with reference-free judges, achieving performance comparable to training with ArmoRM, a strong finetuned reward model. Specifically, our method achieves 73.1% and 58.7% on AlpacaEval and Arena-Hard with Llama-3-8B-Instruct, and 70.0% and 74.1% with Qwen2.5-7B, corresponding to average absolute gains of +20.2 / +17.1 points over SFT distillation and +5.3 / +3.6 points over reference-free self-improvement on AlpacaEval / Arena-Hard. These results highlight the potential of using reference-guided LLM-evaluators to enable effective LLM post-training in non-verifiable domains. 

---
# Multi-Objective Alignment of Language Models for Personalized Psychotherapy 

**Authors**: Mehrab Beikzadeh, Yasaman Asadollah Salmanpour, Ashima Suvarna, Sriram Sankararaman, Matteo Malgaroli, Majid Sarrafzadeh, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2602.16053)  

**Abstract**: Mental health disorders affect over 1 billion people worldwide, yet access to care remains limited by workforce shortages and cost constraints. While AI systems show therapeutic promise, current alignment approaches optimize objectives independently, failing to balance patient preferences with clinical safety. We survey 335 individuals with lived mental health experience to collect preference rankings across therapeutic dimensions, then develop a multi-objective alignment framework using direct preference optimization. We train reward models for six criteria -- empathy, safety, active listening, self-motivated change, trust/rapport, and patient autonomy -- and systematically compare multi-objective approaches against single-objective optimization, supervised fine-tuning, and parameter merging. Multi-objective DPO (MODPO) achieves superior balance (77.6% empathy, 62.6% safety) compared to single-objective optimization (93.6% empathy, 47.8% safety), and therapeutic criteria outperform general communication principles by 17.2%. Blinded clinician evaluation confirms MODPO is consistently preferred, with LLM-evaluator agreement comparable to inter-clinician reliability. 

---
# MedClarify: An information-seeking AI agent for medical diagnosis with case-specific follow-up questions 

**Authors**: Hui Min Wong, Philip Heesen, Pascal Janetzky, Martin Bendszus, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2602.17308)  

**Abstract**: Large language models (LLMs) are increasingly used for diagnostic tasks in medicine. In clinical practice, the correct diagnosis can rarely be immediately inferred from the initial patient presentation alone. Rather, reaching a diagnosis often involves systematic history taking, during which clinicians reason over multiple potential conditions through iterative questioning to resolve uncertainty. This process requires considering differential diagnoses and actively excluding emergencies that demand immediate intervention. Yet, the ability of medical LLMs to generate informative follow-up questions and thus reason over differential diagnoses remains underexplored. Here, we introduce MedClarify, an AI agent for information-seeking that can generate follow-up questions for iterative reasoning to support diagnostic decision-making. Specifically, MedClarify computes a list of candidate diagnoses analogous to a differential diagnosis, and then proactively generates follow-up questions aimed at reducing diagnostic uncertainty. By selecting the question with the highest expected information gain, MedClarify enables targeted, uncertainty-aware reasoning to improve diagnostic performance. In our experiments, we first demonstrate the limitations of current LLMs in medical reasoning, which often yield multiple, similarly likely diagnoses, especially when patient cases are incomplete or relevant information for diagnosis is missing. We then show that our information-theoretic reasoning approach can generate effective follow-up questioning and thereby reduces diagnostic errors by ~27 percentage points (p.p.) compared to a standard single-shot LLM baseline. Altogether, MedClarify offers a path to improve medical LLMs through agentic information-seeking and to thus promote effective dialogues with medical LLMs that reflect the iterative and uncertain nature of real-world clinical reasoning. 

---
# MARS: Margin-Aware Reward-Modeling with Self-Refinement 

**Authors**: Payel Bhattacharjee, Osvaldo Simeone, Ravi Tandon  

**Link**: [PDF](https://arxiv.org/pdf/2602.17658)  

**Abstract**: Reward modeling is a core component of modern alignment pipelines including RLHF and RLAIF, underpinning policy optimization methods including PPO and TRPO. However, training reliable reward models relies heavily on human-labeled preference data, which is costly and limited, motivating the use of data augmentation. Existing augmentation approaches typically operate at the representation or semantic level and remain agnostic to the reward model's estimation difficulty. In this paper, we propose MARS, an adaptive, margin-aware augmentation and sampling strategy that explicitly targets ambiguous and failure modes of the reward model. Our proposed framework, MARS, concentrates augmentation on low-margin (ambiguous) preference pairs where the reward model is most uncertain, and iteratively refines the training distribution via hard-sample augmentation. We provide theoretical guarantees showing that this strategy increases the average curvature of the loss function hence enhance information and improves conditioning, along with empirical results demonstrating consistent gains over uniform augmentation for robust reward modeling. 

---
# MASPO: Unifying Gradient Utilization, Probability Mass, and Signal Reliability for Robust and Sample-Efficient LLM Reasoning 

**Authors**: Xiaoliang Fu, Jiaye Lin, Yangyi Fang, Binbin Zheng, Chaowen Hu, Zekai Shao, Cong Qin, Lu Pan, Ke Zeng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2602.17550)  

**Abstract**: Existing Reinforcement Learning with Verifiable Rewards (RLVR) algorithms, such as GRPO, rely on rigid, uniform, and symmetric trust region mechanisms that are fundamentally misaligned with the complex optimization dynamics of Large Language Models (LLMs). In this paper, we identify three critical challenges in these methods: (1) inefficient gradient utilization caused by the binary cutoff of hard clipping, (2) insensitive probability mass arising from uniform ratio constraints that ignore the token distribution, and (3) asymmetric signal reliability stemming from the disparate credit assignment ambiguity between positive and negative samples. To bridge these gaps, we propose Mass-Adaptive Soft Policy Optimization (MASPO), a unified framework designed to harmonize these three dimensions. MASPO integrates a differentiable soft Gaussian gating to maximize gradient utility, a mass-adaptive limiter to balance exploration across the probability spectrum, and an asymmetric risk controller to align update magnitudes with signal confidence. Extensive evaluations demonstrate that MASPO serves as a robust, all-in-one RLVR solution, significantly outperforming strong baselines. Our code is available at: this https URL. 

---
# DeepVision-103K: A Visually Diverse, Broad-Coverage, and Verifiable Mathematical Dataset for Multimodal Reasoning 

**Authors**: Haoxiang Sun, Lizhen Xu, Bing Zhao, Wotao Yin, Wei Wang, Boyu Yang, Rui Wang, Hu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.16742)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has been shown effective in enhancing the visual reflection and reasoning capabilities of Large Multimodal Models (LMMs). However, existing datasets are predominantly derived from either small-scale manual construction or recombination of prior resources, which limits data diversity and coverage, thereby constraining further gains in model performance. To this end, we introduce \textbf{DeepVision-103K}, a comprehensive dataset for RLVR training that covers diverse K12 mathematical topics, extensive knowledge points, and rich visual elements. Models trained on DeepVision achieve strong performance on multimodal mathematical benchmarks, and generalize effectively to general multimodal reasoning tasks. Further analysis reveals enhanced visual perception, reflection and reasoning capabilities in trained models, validating DeepVision's effectiveness for advancing multimodal reasoning. Data: \href{this https URL}{this url}. 

---
