# Enhancing User-Oriented Proactivity in Open-Domain Dialogues with Critic Guidance 

**Authors**: Yufeng Wang, Jinwu Hu, Ziteng Huang, Kunyang Lin, Zitian Zhang, Peihao Chen, Yu Hu, Qianyue Wang, Zhuliang Yu, Bin Sun, Xiaofen Xing, Qingfang Zheng, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12334)  

**Abstract**: Open-domain dialogue systems aim to generate natural and engaging conversations, providing significant practical value in real applications such as social robotics and personal assistants. The advent of large language models (LLMs) has greatly advanced this field by improving context understanding and conversational fluency. However, existing LLM-based dialogue systems often fall short in proactively understanding the user's chatting preferences and guiding conversations toward user-centered topics. This lack of user-oriented proactivity can lead users to feel unappreciated, reducing their satisfaction and willingness to continue the conversation in human-computer interactions. To address this issue, we propose a User-oriented Proactive Chatbot (UPC) to enhance the user-oriented proactivity. Specifically, we first construct a critic to evaluate this proactivity inspired by the LLM-as-a-judge strategy. Given the scarcity of high-quality training data, we then employ the critic to guide dialogues between the chatbot and user agents, generating a corpus with enhanced user-oriented proactivity. To ensure the diversity of the user backgrounds, we introduce the ISCO-800, a diverse user background dataset for constructing user agents. Moreover, considering the communication difficulty varies among users, we propose an iterative curriculum learning method that trains the chatbot from easy-to-communicate users to more challenging ones, thereby gradually enhancing its performance. Experiments demonstrate that our proposed training method is applicable to different LLMs, improving user-oriented proactivity and attractiveness in open-domain dialogues. 

---
# Beyond Single-Point Judgment: Distribution Alignment for LLM-as-a-Judge 

**Authors**: Luyu Chen, Zeyu Zhang, Haoran Tan, Quanyu Dai, Hao Yang, Zhenhua Dong, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12301)  

**Abstract**: LLMs have emerged as powerful evaluators in the LLM-as-a-Judge paradigm, offering significant efficiency and flexibility compared to human judgments. However, previous methods primarily rely on single-point evaluations, overlooking the inherent diversity and uncertainty in human evaluations. This approach leads to information loss and decreases the reliability of evaluations. To address this limitation, we propose a novel training framework that explicitly aligns the LLM-generated judgment distribution with empirical human distributions. Specifically, we propose a distributional alignment objective based on KL divergence, combined with an auxiliary cross-entropy regularization to stabilize the training process. Furthermore, considering that empirical distributions may derive from limited human annotations, we incorporate adversarial training to enhance model robustness against distribution perturbations. Extensive experiments across various LLM backbones and evaluation tasks demonstrate that our framework significantly outperforms existing closed-source LLMs and conventional single-point alignment methods, with improved alignment quality, evaluation accuracy, and robustness. 

---
# J4R: Learning to Judge with Equivalent Initial State Group Relative Preference Optimization 

**Authors**: Austin Xu, Yilun Zhou, Xuan-Phi Nguyen, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2505.13346)  

**Abstract**: To keep pace with the increasing pace of large language models (LLM) development, model output evaluation has transitioned away from time-consuming human evaluation to automatic evaluation, where LLMs themselves are tasked with assessing and critiquing other model outputs. LLM-as-judge models are a class of generative evaluators that excel in evaluating relatively simple domains, like chat quality, but struggle in reasoning intensive domains where model responses contain more substantive and challenging content. To remedy existing judge shortcomings, we explore training judges with reinforcement learning (RL). We make three key contributions: (1) We propose the Equivalent Initial State Group Relative Policy Optimization (EIS-GRPO) algorithm, which allows us to train our judge to be robust to positional biases that arise in more complex evaluation settings. (2) We introduce ReasoningJudgeBench, a benchmark that evaluates judges in diverse reasoning settings not covered by prior work. (3) We train Judge for Reasoning (J4R), a 7B judge trained with EIS-GRPO that outperforms GPT-4o and the next best small judge by 6.7% and 9%, matching or exceeding the performance of larger GRPO-trained judges on both JudgeBench and ReasoningJudgeBench. 

---
# LEXam: Benchmarking Legal Reasoning on 340 Law Exams 

**Authors**: Yu Fan, Jingwei Ni, Jakob Merane, Etienne Salimbeni, Yang Tian, Yoan Hermstrüwer, Yinya Huang, Mubashara Akhtar, Florian Geering, Oliver Dreyer, Daniel Brunner, Markus Leippold, Mrinmaya Sachan, Alexander Stremitzer, Christoph Engel, Elliott Ash, Joel Niklaus  

**Link**: [PDF](https://arxiv.org/pdf/2505.12864)  

**Abstract**: Long-form legal reasoning remains a key challenge for large language models (LLMs) in spite of recent advances in test-time scaling. We introduce LEXam, a novel benchmark derived from 340 law exams spanning 116 law school courses across a range of subjects and degree levels. The dataset comprises 4,886 law exam questions in English and German, including 2,841 long-form, open-ended questions and 2,045 multiple-choice questions. Besides reference answers, the open questions are also accompanied by explicit guidance outlining the expected legal reasoning approach such as issue spotting, rule recall, or rule application. Our evaluation on both open-ended and multiple-choice questions present significant challenges for current LLMs; in particular, they notably struggle with open questions that require structured, multi-step legal reasoning. Moreover, our results underscore the effectiveness of the dataset in differentiating between models with varying capabilities. Adopting an LLM-as-a-Judge paradigm with rigorous human expert validation, we demonstrate how model-generated reasoning steps can be evaluated consistently and accurately. Our evaluation setup provides a scalable method to assess legal reasoning quality beyond simple accuracy metrics. Project page: this https URL 

---
# MR. Judge: Multimodal Reasoner as a Judge 

**Authors**: Renjie Pi, Felix Bai, Qibin Chen, Simon Wang, Jiulong Shan, Kieran Liu, Meng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13403)  

**Abstract**: The paradigm of using Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) as evaluative judges has emerged as an effective approach in RLHF and inference-time scaling. In this work, we propose Multimodal Reasoner as a Judge (MR. Judge), a paradigm for empowering general-purpose MLLMs judges with strong reasoning capabilities. Instead of directly assigning scores for each response, we formulate the judgement process as a reasoning-inspired multiple-choice problem. Specifically, the judge model first conducts deliberate reasoning covering different aspects of the responses and eventually selects the best response from them. This reasoning process not only improves the interpretibility of the judgement, but also greatly enhances the performance of MLLM judges. To cope with the lack of questions with scored responses, we propose the following strategy to achieve automatic annotation: 1) Reverse Response Candidates Synthesis: starting from a supervised fine-tuning (SFT) dataset, we treat the original response as the best candidate and prompt the MLLM to generate plausible but flawed negative candidates. 2) Text-based reasoning extraction: we carefully design a data synthesis pipeline for distilling the reasoning capability from a text-based reasoning model, which is adopted to enable the MLLM judges to regain complex reasoning ability via warm up supervised fine-tuning. Experiments demonstrate that our MR. Judge is effective across a wide range of tasks. Specifically, our MR. Judge-7B surpasses GPT-4o by 9.9% on VL-RewardBench, and improves performance on MM-Vet during inference-time scaling by up to 7.7%. 

---
# Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks 

**Authors**: Narek Maloyan, Bislan Ashinov, Dmitry Namiot  

**Link**: [PDF](https://arxiv.org/pdf/2505.13348)  

**Abstract**: Large Language Models (LLMs) are increasingly employed as evaluators (LLM-as-a-Judge) for assessing the quality of machine-generated text. This paradigm offers scalability and cost-effectiveness compared to human annotation. However, the reliability and security of such systems, particularly their robustness against adversarial manipulations, remain critical concerns. This paper investigates the vulnerability of LLM-as-a-Judge architectures to prompt-injection attacks, where malicious inputs are designed to compromise the judge's decision-making process. We formalize two primary attack strategies: Comparative Undermining Attack (CUA), which directly targets the final decision output, and Justification Manipulation Attack (JMA), which aims to alter the model's generated reasoning. Using the Greedy Coordinate Gradient (GCG) optimization method, we craft adversarial suffixes appended to one of the responses being compared. Experiments conducted on the MT-Bench Human Judgments dataset with open-source instruction-tuned LLMs (Qwen2.5-3B-Instruct and Falcon3-3B-Instruct) demonstrate significant susceptibility. The CUA achieves an Attack Success Rate (ASR) exceeding 30\%, while JMA also shows notable effectiveness. These findings highlight substantial vulnerabilities in current LLM-as-a-Judge systems, underscoring the need for robust defense mechanisms and further research into adversarial evaluation and trustworthiness in LLM-based assessment frameworks. 

---
# Decentralized Arena: Towards Democratic and Scalable Automatic Evaluation of Language Models 

**Authors**: Yanbin Yin, Kun Zhou, Zhen Wang, Xiangdong Zhang, Yifei Shao, Shibo Hao, Yi Gu, Jieyuan Liu, Somanshu Singla, Tianyang Liu, Eric P. Xing, Zhengzhong Liu, Haojian Jin, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12808)  

**Abstract**: The recent explosion of large language models (LLMs), each with its own general or specialized strengths, makes scalable, reliable benchmarking more urgent than ever. Standard practices nowadays face fundamental trade-offs: closed-ended question-based benchmarks (eg MMLU) struggle with saturation as newer models emerge, while crowd-sourced leaderboards (eg Chatbot Arena) rely on costly and slow human judges. Recently, automated methods (eg LLM-as-a-judge) shed light on the scalability, but risk bias by relying on one or a few "authority" models. To tackle these issues, we propose Decentralized Arena (dearena), a fully automated framework leveraging collective intelligence from all LLMs to evaluate each other. It mitigates single-model judge bias by democratic, pairwise evaluation, and remains efficient at scale through two key components: (1) a coarse-to-fine ranking algorithm for fast incremental insertion of new models with sub-quadratic complexity, and (2) an automatic question selection strategy for the construction of new evaluation dimensions. Across extensive experiments across 66 LLMs, dearena attains up to 97% correlation with human judgements, while significantly reducing the cost. Our code and data will be publicly released on this https URL. 

---
# ESC-Judge: A Framework for Comparing Emotional Support Conversational Agents 

**Authors**: Navid Madani, Rohini Srihari  

**Link**: [PDF](https://arxiv.org/pdf/2505.12531)  

**Abstract**: Large language models (LLMs) increasingly power mental-health chatbots, yet the field still lacks a scalable, theory-grounded way to decide which model is most effective to deploy. We present ESC-Judge, the first end-to-end evaluation framework that (i) grounds head-to-head comparisons of emotional-support LLMs in Clara Hill's established Exploration-Insight-Action counseling model, providing a structured and interpretable view of performance, and (ii) fully automates the evaluation pipeline at scale. ESC-Judge operates in three stages: first, it synthesizes realistic help-seeker roles by sampling empirically salient attributes such as stressors, personality, and life history; second, it has two candidate support agents conduct separate sessions with the same role, isolating model-specific strategies; and third, it asks a specialized judge LLM to express pairwise preferences across rubric-anchored skills that span the Exploration, Insight, and Action spectrum. In our study, ESC-Judge matched PhD-level annotators on 85 percent of Exploration, 83 percent of Insight, and 86 percent of Action decisions, demonstrating human-level reliability at a fraction of the cost. All code, prompts, synthetic roles, transcripts, and judgment scripts are released to promote transparent progress in emotionally supportive AI. 

---
# LLMSR@XLLM25: An Empirical Study of LLM for Structural Reasoning 

**Authors**: Xinye Li, Mingqi Wan, Dianbo Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.12328)  

**Abstract**: We present Team asdfo123's submission to the LLMSR@XLLM25 shared task, which evaluates large language models on producing fine-grained, controllable, and interpretable reasoning processes. Systems must extract all problem conditions, decompose a chain of thought into statement-evidence pairs, and verify the logical validity of each pair. Leveraging only the off-the-shelf Meta-Llama-3-8B-Instruct, we craft a concise few-shot, multi-turn prompt that first enumerates all conditions and then guides the model to label, cite, and adjudicate every reasoning step. A lightweight post-processor based on regular expressions normalises spans and enforces the official JSON schema. Without fine-tuning, external retrieval, or ensembling, our method ranks 5th overall, achieving macro F1 scores on par with substantially more complex and resource-consuming pipelines. We conclude by analysing the strengths and limitations of our approach and outlining directions for future research in structural reasoning with LLMs. Our code is available at this https URL. 

---
# How Reliable is Multilingual LLM-as-a-Judge? 

**Authors**: Xiyan Fu, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12201)  

**Abstract**: LLM-as-a-Judge has emerged as a popular evaluation strategy, where advanced large language models assess generation results in alignment with human instructions. While these models serve as a promising alternative to human annotators, their reliability in multilingual evaluation remains uncertain. To bridge this gap, we conduct a comprehensive analysis of multilingual LLM-as-a-Judge. Specifically, we evaluate five models from different model families across five diverse tasks involving 25 languages. Our findings reveal that LLMs struggle to achieve consistent judgment results across languages, with an average Fleiss' Kappa of approximately 0.3, and some models performing even worse. To investigate the cause of inconsistency, we analyze various influencing factors. We observe that consistency varies significantly across languages, with particularly poor performance in low-resource languages. Additionally, we find that neither training on multilingual data nor increasing model scale directly improves judgment consistency. These findings suggest that LLMs are not yet reliable for evaluating multilingual predictions. We finally propose an ensemble strategy which improves the consistency of the multilingual judge in real-world applications. 

---
# Teach2Eval: An Indirect Evaluation Method for LLM by Judging How It Teaches 

**Authors**: Yuhang Zhou, Xutian Chen, Yixin Cao, Yuchen Ni, Yu He, Siyu Tian, Xiang Liu, Jian Zhang, Chuanjun Ji, Guangnan Ye, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12259)  

**Abstract**: Recent progress in large language models (LLMs) has outpaced the development of effective evaluation methods. Traditional benchmarks rely on task-specific metrics and static datasets, which often suffer from fairness issues, limited scalability, and contamination risks. In this paper, we introduce Teach2Eval, an indirect evaluation framework inspired by the Feynman Technique. Instead of directly testing LLMs on predefined tasks, our method evaluates a model's multiple abilities to teach weaker student models to perform tasks effectively. By converting open-ended tasks into standardized multiple-choice questions (MCQs) through teacher-generated feedback, Teach2Eval enables scalable, automated, and multi-dimensional assessment. Our approach not only avoids data leakage and memorization but also captures a broad range of cognitive abilities that are orthogonal to current benchmarks. Experimental results across 26 leading LLMs show strong alignment with existing human and model-based dynamic rankings, while offering additional interpretability for training guidance. 

---
# MedGUIDE: Benchmarking Clinical Decision-Making in Large Language Models 

**Authors**: Xiaomin Li, Mingye Gao, Yuexing Hao, Taoran Li, Guangya Wan, Zihan Wang, Yijun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11613)  

**Abstract**: Clinical guidelines, typically structured as decision trees, are central to evidence-based medical practice and critical for ensuring safe and accurate diagnostic decision-making. However, it remains unclear whether Large Language Models (LLMs) can reliably follow such structured protocols. In this work, we introduce MedGUIDE, a new benchmark for evaluating LLMs on their ability to make guideline-consistent clinical decisions. MedGUIDE is constructed from 55 curated NCCN decision trees across 17 cancer types and uses clinical scenarios generated by LLMs to create a large pool of multiple-choice diagnostic questions. We apply a two-stage quality selection process, combining expert-labeled reward models and LLM-as-a-judge ensembles across ten clinical and linguistic criteria, to select 7,747 high-quality samples. We evaluate 25 LLMs spanning general-purpose, open-source, and medically specialized models, and find that even domain-specific LLMs often underperform on tasks requiring structured guideline adherence. We also test whether performance can be improved via in-context guideline inclusion or continued pretraining. Our findings underscore the importance of MedGUIDE in assessing whether LLMs can operate safely within the procedural frameworks expected in real-world clinical settings. 

---
# J1: Exploring Simple Test-Time Scaling for LLM-as-a-Judge 

**Authors**: Chi-Min Chan, Chunpu Xu, Jiaming Ji, Zhen Ye, Pengcheng Wen, Chunyang Jiang, Yaodong Yang, Wei Xue, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11875)  

**Abstract**: The current focus of AI research is shifting from emphasizing model training towards enhancing evaluation quality, a transition that is crucial for driving further advancements in AI systems. Traditional evaluation methods typically rely on reward models assigning scalar preference scores to outputs. Although effective, such approaches lack interpretability, leaving users often uncertain about why a reward model rates a particular response as high or low. The advent of LLM-as-a-Judge provides a more scalable and interpretable method of supervision, offering insights into the decision-making process. Moreover, with the emergence of large reasoning models, which consume more tokens for deeper thinking and answer refinement, scaling test-time computation in the LLM-as-a-Judge paradigm presents an avenue for further boosting performance and providing more interpretability through reasoning traces. In this paper, we introduce $\textbf{J1-7B}$, which is first supervised fine-tuned on reflection-enhanced datasets collected via rejection-sampling and subsequently trained using Reinforcement Learning (RL) with verifiable rewards. At inference time, we apply Simple Test-Time Scaling (STTS) strategies for additional performance improvement. Experimental results demonstrate that $\textbf{J1-7B}$ surpasses the previous state-of-the-art LLM-as-a-Judge by $ \textbf{4.8}$\% and exhibits a $ \textbf{5.1}$\% stronger scaling trend under STTS. Additionally, we present three key findings: (1) Existing LLM-as-a-Judge does not inherently exhibit such scaling trend. (2) Model simply fine-tuned on reflection-enhanced datasets continues to demonstrate similarly weak scaling behavior. (3) Significant scaling trend emerges primarily during the RL phase, suggesting that effective STTS capability is acquired predominantly through RL training. 

---
