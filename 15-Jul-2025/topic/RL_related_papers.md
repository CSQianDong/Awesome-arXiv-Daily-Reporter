# EduFlow: Advancing MLLMs' Problem-Solving Proficiency through Multi-Stage, Multi-Perspective Critique 

**Authors**: Chenglin Zhu, Tao Zhang, Chong Li, Mingan Lin, Zenan Zhou, Jian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.09374)  

**Abstract**: Multimodal large language models (MLLMs) still perform poorly on scientific tasks, particularly those requiring multi-step and interpretable reasoning. Their limitations include insufficient scientific reasoning patterns, lack of global coherence in multi-step inference, and the absence of reflective self-correction, making them unreliable in structured scientific contexts. We introduce EduFlow, the first end-to-end framework that covers the full pipeline of educational scientific reasoning, including data selection, MCTS-based trajectory construction, model training, and output optimization. At its core is EduPRM, a process-aware reward model that critiques reasoning steps with tags and justifications. EduPRM is trained via curriculum learning on three complementary supervision sources: MCTS-guided trajectories, error-injected critiques, and teacher-student dialogues, enabling dynamic adaptation to multi-stage problem solving and iterative refinement during inference. We further propose EduMCTS, a domain-adapted search framework that introduces bootstrapping actions specifically designed for educational reasoning, such as a self-reflection mechanism that promotes reflective error correction. It further leverages EduPRM's fine-grained feedback to guide the search toward higher-quality reasoning trajectories. By applying self-consistency and rejection sampling, we constructed EduMCTS-160K, a large-scale dataset of educational reasoning trajectories. Extensive experiments demonstrate that EduFlow enhances reasoning consistency and coherence. Code, data, and models will be released. 

---
# Aligning Generative Speech Enhancement with Human Preferences via Direct Preference Optimization 

**Authors**: Haoyang Li, Nana Hou, Yuchen Hu, Jixun Yao, Sabato Marco Siniscalchi, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2507.09929)  

**Abstract**: This work investigates speech enhancement (SE) from the perspective of language models (LMs). We propose a novel method that leverages Direct Preference Optimization (DPO) to improve the perceptual quality of enhanced speech. Using UTMOS, a neural MOS prediction model, as a proxy for human ratings, our approach guides optimization toward perceptually preferred outputs. This differs from existing LM-based SE methods that focus on maximizing the likelihood of clean speech tokens, which may misalign with human perception and degrade quality despite low prediction error. Experiments on the 2020 Deep Noise Suppression Challenge test sets demonstrate that applying DPO to a pretrained LM-based SE model yields consistent improvements across various speech quality metrics, with relative gains of up to 56%. To our knowledge, this is the first application of DPO to SE and the first to incorporate proxy perceptual feedback into LM-based SE training, pointing to a promising direction for perceptually aligned SE. 

---
# Tiny Reward Models 

**Authors**: Sarah Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.09973)  

**Abstract**: Large decoder-based language models have become the dominant architecture for reward modeling in reinforcement learning from human feedback (RLHF). However, as reward models are increasingly deployed in test-time strategies, their inference costs become a growing concern. We present TinyRM, a family of small, bidirectional masked language models (MLMs) with as few as 400 million parameters, that rival the capabilities of models over 175 times larger on reasoning and safety preference modeling tasks. TinyRM combines FLAN-style prompting, Directional Low-Rank Adaptation (DoRA), and layer freezing to achieve strong performance on RewardBench, despite using significantly fewer resources. Our experiments suggest that small models benefit from domain-specific tuning strategies, particularly in reasoning, where lightweight finetuning methods are especially effective. While challenges remain in building generalist models and conversational preference modeling, our preliminary results highlight the promise of lightweight bidirectional architectures as efficient, scalable alternatives for preference modeling. 

---
# Adversarial Activation Patching: A Framework for Detecting and Mitigating Emergent Deception in Safety-Aligned Transformers 

**Authors**: Santhosh Kumar Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2507.09406)  

**Abstract**: Large language models (LLMs) aligned for safety through techniques like reinforcement learning from human feedback (RLHF) often exhibit emergent deceptive behaviors, where outputs appear compliant but subtly mislead or omit critical information. This paper introduces adversarial activation patching, a novel mechanistic interpretability framework that leverages activation patching as an adversarial tool to induce, detect, and mitigate such deception in transformer-based models. By sourcing activations from "deceptive" prompts and patching them into safe forward passes at specific layers, we simulate vulnerabilities and quantify deception rates. Through toy neural network simulations across multiple scenarios (e.g., 1000 trials per setup), we demonstrate that adversarial patching increases deceptive outputs to 23.9% from a 0% baseline, with layer-specific variations supporting our hypotheses. We propose six hypotheses, including transferability across models, exacerbation in multimodal settings, and scaling effects. An expanded literature review synthesizes over 20 key works in interpretability, deception, and adversarial attacks. Mitigation strategies, such as activation anomaly detection and robust fine-tuning, are detailed, alongside ethical considerations and future research directions. This work advances AI safety by highlighting patching's dual-use potential and provides a roadmap for empirical studies on large-scale models. 

---
# ViSP: A PPO-Driven Framework for Sarcasm Generation with Contrastive Learning 

**Authors**: Changli Wang, Rui Wu, Fang Yin  

**Link**: [PDF](https://arxiv.org/pdf/2507.09482)  

**Abstract**: Human emotions are complex, with sarcasm being a subtle and distinctive form. Despite progress in sarcasm research, sarcasm generation remains underexplored, primarily due to the overreliance on textual modalities and the neglect of visual cues, as well as the mismatch between image content and sarcastic intent in existing datasets. In this paper, we introduce M2SaG, a multimodal sarcasm generation dataset with 4,970 samples, each containing an image, a sarcastic text, and a sarcasm target. To benchmark M2SaG, we propose ViSP, a generation framework that integrates Proximal Policy Optimization (PPO) and contrastive learning. PPO utilizes reward scores from DIP to steer the generation of sarcastic texts, while contrastive learning encourages the model to favor outputs with higher reward scores. These strategies improve overall generation quality and produce texts with more pronounced sarcastic intent. We evaluate ViSP across five metric sets and find it surpasses all baselines, including large language models, underscoring their limitations in sarcasm generation. Furthermore, we analyze the distributions of Sarcasm Scores and Factual Incongruity for both M2SaG and the texts generated by ViSP. The generated texts exhibit higher mean Sarcasm Scores (0.898 vs. 0.770) and Factual Incongruity (0.768 vs. 0.739), demonstrating that ViSP produces higher-quality sarcastic content than the original dataset. % The dataset and code will be publicly available. Our dataset and code will be released at \textit{this https URL}. 

---
# Prompt4Trust: A Reinforcement Learning Prompt Augmentation Framework for Clinically-Aligned Confidence Calibration in Multimodal Large Language Models 

**Authors**: Anita Kriz, Elizabeth Laura Janes, Xing Shen, Tal Arbel  

**Link**: [PDF](https://arxiv.org/pdf/2507.09279)  

**Abstract**: Multimodal large language models (MLLMs) hold considerable promise for applications in healthcare. However, their deployment in safety-critical settings is hindered by two key limitations: (i) sensitivity to prompt design, and (ii) a tendency to generate incorrect responses with high confidence. As clinicians may rely on a model's stated confidence to gauge the reliability of its predictions, it is especially important that when a model expresses high confidence, it is also highly accurate. We introduce Prompt4Trust, the first reinforcement learning (RL) framework for prompt augmentation targeting confidence calibration in MLLMs. A lightweight LLM is trained to produce context-aware auxiliary prompts that guide a downstream task MLLM to generate responses in which the expressed confidence more accurately reflects predictive accuracy. Unlike conventional calibration techniques, Prompt4Trust specifically prioritizes aspects of calibration most critical for safe and trustworthy clinical decision-making. Beyond improvements driven by this clinically motivated calibration objective, our proposed method also improves task accuracy, achieving state-of-the-art medical visual question answering (VQA) performance on the PMC-VQA benchmark, which is composed of multiple-choice questions spanning diverse medical imaging modalities. Moreover, our framework trained with a small downstream task MLLM showed promising zero-shot generalization to larger MLLMs in our experiments, suggesting the potential for scalable calibration without the associated computational costs. This work demonstrates the potential of automated yet human-aligned prompt engineering for improving the the trustworthiness of MLLMs in safety critical settings. Our codebase can be found at this https URL. 

---
# ALIGN: Prompt-based Attribute Alignment for Reliable, Responsible, and Personalized LLM-based Decision-Making 

**Authors**: Bharadwaj Ravichandran, David Joy, Paul Elliott, Brian Hu, Jadie Adams, Christopher Funk, Emily Veenhuis, Anthony Hoogs, Arslan Basharat  

**Link**: [PDF](https://arxiv.org/pdf/2507.09037)  

**Abstract**: Large language models (LLMs) are increasingly being used as decision aids. However, users have diverse values and preferences that can affect their decision-making, which requires novel methods for LLM alignment and personalization. Existing LLM comparison tools largely focus on benchmarking tasks, such as knowledge-based question answering. In contrast, our proposed ALIGN system focuses on dynamic personalization of LLM-based decision-makers through prompt-based alignment to a set of fine-grained attributes. Key features of our system include robust configuration management, structured output generation with reasoning, and several algorithm implementations with swappable LLM backbones, enabling different types of analyses. Our user interface enables a qualitative, side-by-side comparison of LLMs and their alignment to various attributes, with a modular backend for easy algorithm integration. Additionally, we perform a quantitative analysis comparing alignment approaches in two different domains: demographic alignment for public opinion surveys and value alignment for medical triage decision-making. The entire ALIGN framework is open source and will enable new research on reliable, responsible, and personalized LLM-based decision-makers. 

---
# wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models 

**Authors**: Xiaohang Tang, Rares Dolga, Sangwoong Yoon, Ilija Bogunovic  

**Link**: [PDF](https://arxiv.org/pdf/2507.08838)  

**Abstract**: Improving the reasoning capabilities of diffusion-based large language models (dLLMs) through reinforcement learning (RL) remains an open problem. The intractability of dLLMs likelihood function necessitates approximating the current, old, and reference policy likelihoods at each policy optimization step. This reliance introduces additional computational overhead and lead to potentially large bias -- particularly when approximation errors occur in the denominator of policy ratios used for importance sampling. To mitigate these issues, we introduce $\mathtt{wd1}$, a novel policy optimization approach that reformulates the objective as a weighted likelihood, requiring only a single approximation for the current parametrized policy likelihood. Experiments on widely used reasoning benchmarks demonstrate that $\mathtt{wd1}$, without supervised fine-tuning (SFT) or any supervised data, outperforms existing RL methods for dLLMs, achieving up to 16% higher accuracy. $\mathtt{wd1}$ delivers additional computational gains, including reduced training time and fewer function evaluations (NFEs) per gradient step. These findings, combined with the simplicity of method's implementation and R1-Zero-like training (no SFT), position $\mathtt{wd1}$ as a more effective and efficient method for applying RL to dLLMs reasoning. 

---
# Can Group Relative Policy Optimization Improve Thai Legal Reasoning and Question Answering? 

**Authors**: Pawitsapak Akarajaradwong, Chompakorn Chaksangchaichot, Pirat Pothavorn, Attapol Thamrongrattanarit-Rutherford, Ekapol Chuangsuwanich, Sarana Nutanong  

**Link**: [PDF](https://arxiv.org/pdf/2507.09638)  

**Abstract**: The Retrieval-Augmented Generation (RAG) systems' performance on Thai legal question answering is still limited, especially for questions requiring extensive, complex legal reasoning. To address these limitations, we introduce an approach aligning LLMs toward improved law citation accuracy and better response quality using Group-Relative Policy Optimization (GRPO). Our approach leverages BGE-M3 embeddings as a cost-efficient semantic-similarity reward, significantly reducing computational expenses up to 2.5x compared to large language model judges. Experiments on the NitiBench benchmark demonstrate substantial improvements: GRPO achieves up to 90% citation-F1 gains from the base model and a 31% increase in joint quality metrics over instruction tuning. Crucially, our method shows enhanced robustness on complex legal reasoning tasks compared to instruction tuning, providing an effective and resource-efficient solution for enhancing Thai legal LLMs. 

---
# Self-Improving Model Steering 

**Authors**: Rongyi Zhu, Yuhui Wang, Tanqiu Jiang, Jiacheng Liang, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08967)  

**Abstract**: Model steering represents a powerful technique that dynamically aligns large language models (LLMs) with human preferences during inference. However, conventional model-steering methods rely heavily on externally annotated data, not only limiting their adaptability to varying contexts but also tethering their effectiveness to annotation quality. In this paper, we present SIMS, the first self-improving model-steering framework that operates without relying on external supervision. At its core, SIMS autonomously generates and refines contrastive samples through iterative self-improvement cycles, enabling adaptive, context-specific steering. Additionally, SIMS employs novel strategies, including prompt ranking and contrast sampling, to further enhance steering efficacy. Extensive evaluation across diverse LLMs and benchmarks demonstrates that SIMS substantially outperforms existing methods in steering effectiveness and adaptability, highlighting self-improving model steering as a promising direction for future research on inference-time LLM alignment. 

---
