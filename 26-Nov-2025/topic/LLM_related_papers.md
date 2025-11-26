# Adversarial Confusion Attack: Disrupting Multimodal Large Language Models 

**Authors**: Jakub Hoscilowicz, Artur Janicki  

**Link**: [PDF](https://arxiv.org/pdf/2511.20494)  

**Abstract**: We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Applications include embedding adversarial images into websites to prevent MLLM-powered agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models. 

---
# The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models 

**Authors**: Taewhoo Lee, Minju Song, Chanwoong Yoon, Jungwoo Park, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20344)  

**Abstract**: Analogical reasoning is at the core of human cognition, serving as an important foundation for a variety of intellectual activities. While prior work has shown that LLMs can represent task patterns and surface-level concepts, it remains unclear whether these models can encode high-level relational concepts and apply them to novel situations through structured comparisons. In this work, we explore this fundamental aspect using proportional and story analogies, and identify three key findings. First, LLMs effectively encode the underlying relationships between analogous entities; both attributive and relational information propagate through mid-upper layers in correct cases, whereas reasoning failures reflect missing relational information within these layers. Second, unlike humans, LLMs often struggle not only when relational information is missing, but also when attempting to apply it to new entities. In such cases, strategically patching hidden representations at critical token positions can facilitate information transfer to a certain extent. Lastly, successful analogical reasoning in LLMs is marked by strong structural alignment between analogous situations, whereas failures often reflect degraded or misplaced alignment. Overall, our findings reveal that LLMs exhibit emerging but limited capabilities in encoding and applying high-level relational concepts, highlighting both parallels and gaps with human cognition. 

---
# On Evaluating LLM Alignment by Evaluating LLMs as Judges 

**Authors**: Yixin Liu, Pengfei Liu, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20604)  

**Abstract**: Alignment with human preferences is an important evaluation aspect of LLMs, requiring them to be helpful, honest, safe, and to precisely follow human instructions. Evaluating large language models' (LLMs) alignment typically involves directly assessing their open-ended responses, requiring human annotators or strong LLM judges. Conversely, LLMs themselves have also been extensively evaluated as judges for assessing alignment. In this work, we examine the relationship between LLMs' generation and evaluation capabilities in aligning with human preferences. To this end, we first conduct a comprehensive analysis of the generation-evaluation consistency (GE-consistency) among various LLMs, revealing a strong correlation between their generation and evaluation capabilities when evaluated by a strong LLM preference oracle. Utilizing this finding, we propose a benchmarking paradigm that measures LLM alignment with human preferences without directly evaluating their generated outputs, instead assessing LLMs in their role as evaluators. Our evaluation shows that our proposed benchmark, AlignEval, matches or surpasses widely used automatic LLM evaluation benchmarks, such as AlpacaEval and Arena-Hard, in capturing human preferences when ranking LLMs. Our study offers valuable insights into the connection between LLMs' generation and evaluation capabilities, and introduces a benchmark that assesses alignment without directly evaluating model outputs. 

---
# The Text Aphasia Battery (TAB): A Clinically-Grounded Benchmark for Aphasia-Like Deficits in Language Models 

**Authors**: Nathan Roll, Jill Kries, Flora Jin, Catherine Wang, Ann Marie Finley, Meghan Sumner, Cory Shain, Laura Gwilliams  

**Link**: [PDF](https://arxiv.org/pdf/2511.20507)  

**Abstract**: Large language models (LLMs) have emerged as a candidate "model organism" for human language, offering an unprecedented opportunity to study the computational basis of linguistic disorders like aphasia. However, traditional clinical assessments are ill-suited for LLMs, as they presuppose human-like pragmatic pressures and probe cognitive processes not inherent to artificial architectures. We introduce the Text Aphasia Battery (TAB), a text-only benchmark adapted from the Quick Aphasia Battery (QAB) to assess aphasic-like deficits in LLMs. The TAB comprises four subtests: Connected Text, Word Comprehension, Sentence Comprehension, and Repetition. This paper details the TAB's design, subtests, and scoring criteria. To facilitate large-scale use, we validate an automated evaluation protocol using Gemini 2.5 Flash, which achieves reliability comparable to expert human raters (prevalence-weighted Cohen's kappa = 0.255 for model--consensus agreement vs. 0.286 for human--human agreement). We release TAB as a clinically-grounded, scalable framework for analyzing language deficits in artificial systems. 

---
# "When Data is Scarce, Prompt Smarter"... Approaches to Grammatical Error Correction in Low-Resource Settings 

**Authors**: Somsubhra De, Harsh Kumar, Arun Prakash A  

**Link**: [PDF](https://arxiv.org/pdf/2511.20120)  

**Abstract**: Grammatical error correction (GEC) is an important task in Natural Language Processing that aims to automatically detect and correct grammatical mistakes in text. While recent advances in transformer-based models and large annotated datasets have greatly improved GEC performance for high-resource languages such as English, the progress has not extended equally. For most Indic languages, GEC remains a challenging task due to limited resources, linguistic diversity and complex morphology. In this work, we explore prompting-based approaches using state-of-the-art large language models (LLMs), such as GPT-4.1, Gemini-2.5 and LLaMA-4, combined with few-shot strategy to adapt them to low-resource settings. We observe that even basic prompting strategies, such as zero-shot and few-shot approaches, enable these LLMs to substantially outperform fine-tuned Indic-language models like Sarvam-22B, thereby illustrating the exceptional multilingual generalization capabilities of contemporary LLMs for GEC. Our experiments show that carefully designed prompts and lightweight adaptation significantly enhance correction quality across multiple Indic languages. We achieved leading results in the shared task--ranking 1st in Tamil (GLEU: 91.57) and Hindi (GLEU: 85.69), 2nd in Telugu (GLEU: 85.22), 4th in Bangla (GLEU: 92.86), and 5th in Malayalam (GLEU: 92.97). These findings highlight the effectiveness of prompt-driven NLP techniques and underscore the potential of large-scale LLMs to bridge resource gaps in multilingual GEC. 

---
# REFLEX: Self-Refining Explainable Fact-Checking via Disentangling Truth into Style and Substance 

**Authors**: Chuyi Kong, Gao Wei, Jing Ma, Hongzhan Lin, Zhiyuan Fan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20233)  

**Abstract**: The prevalence of misinformation on social media threatens public trust, demanding automated fact-checking systems that provide accurate verdicts with interpretable explanations. However, existing large language model-based (LLM-based) approaches often rely heavily on external knowledge sources, introducing substantial latency and even hallucinations that undermine reliability, interpretability, and responsiveness, which is crucial for real-time use. To address these challenges, we propose REason-guided Fact-checking with Latent EXplanations REFLEX paradigm, a plug-and-play, self-refining paradigm that leverages the internal knowledge in backbone model to improve both verdict accuracy and explanation quality. REFLEX reformulates fact-checking as a role-play dialogue and jointly trains verdict prediction and explanation generation. It adaptively extracts contrastive activation pairs between the backbone model and its fine-tuned variant to construct steering vectors that disentangle truth into style and substance naturally. These activation-level signals guide inference and suppress noisy explanations, enabling more faithful and efficient reasoning. Experiments on real-world datasets show that REFLEX outperforms previous methods that steer toward a single truth direction and underscores the challenge traditional approaches face when handling the subtle, human-unknown truth in fact-checking tasks. Remarkably, with only 465 self-refined training samples, RELFEX achieves state-of-the-art performance. Furthermore, models trained with explanatory objectives can effectively guide those without them, yielding up to a 7.57% improvement, highlighting that internal explanation signals play a dual role in both interpreting and enhancing factual reasoning. 

---
# More Bias, Less Bias: BiasPrompting for Enhanced Multiple-Choice Question Answering 

**Authors**: Duc Anh Vu, Thong Nguyen, Cong-Duy Nguyen, Viet Anh Nguyen, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20086)  

**Abstract**: With the advancement of large language models (LLMs), their performance on multiple-choice question (MCQ) tasks has improved significantly. However, existing approaches face key limitations: answer choices are typically presented to LLMs without contextual grounding or explanation. This absence of context can lead to incomplete exploration of all possible answers, ultimately degrading the models' reasoning capabilities. To address these challenges, we introduce BiasPrompting, a novel inference framework that guides LLMs to generate and critically evaluate reasoning across all plausible answer options before reaching a final prediction. It consists of two components: first, a reasoning generation stage, where the model is prompted to produce supportive reasonings for each answer option, and then, a reasoning-guided agreement stage, where the generated reasonings are synthesized to select the most plausible answer. Through comprehensive evaluations, BiasPrompting demonstrates significant improvements in five widely used multiple-choice question answering benchmarks. Our experiments showcase that BiasPrompting enhances the reasoning capabilities of LLMs and provides a strong foundation for tackling complex and challenging questions, particularly in settings where existing methods underperform. 

---
# Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios 

**Authors**: Luohe Shi, Zuchao Li, Lefei Zhang, Baoyuan Qi, Guoming Liu, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.20340)  

**Abstract**: Speculative decoding accelerates LLM inference by utilizing otherwise idle computational resources during memory-to-chip data transfer. Current speculative decoding methods typically assume a considerable amount of available computing power, then generate a complex and massive draft tree using a small autoregressive language model to improve overall prediction accuracy. However, methods like batching have been widely applied in mainstream model inference systems as a superior alternative to speculative decoding, as they compress the available idle computing power. Therefore, performing speculative decoding with low verification resources and low scheduling costs has become an important research problem. We believe that more capable models that allow for parallel generation on draft sequences are what we truly need. Recognizing the fundamental nature of draft models to only generate sequences of limited length, we propose SpecFormer, a novel architecture that integrates unidirectional and bidirectional attention mechanisms. SpecFormer combines the autoregressive model's ability to extract information from the entire input sequence with the parallel generation benefits of non-autoregressive models. This design eliminates the reliance on large prefix trees and achieves consistent acceleration, even in large-batch scenarios. Through lossless speculative decoding experiments across models of various scales, we demonstrate that SpecFormer sets a new standard for scaling LLM inference with lower training demands and reduced computational costs. 

---
# BengaliFig: A Low-Resource Challenge for Figurative and Culturally Grounded Reasoning in Bengali 

**Authors**: Abdullah Al Sefat  

**Link**: [PDF](https://arxiv.org/pdf/2511.20399)  

**Abstract**: Large language models excel on broad multilingual benchmarks but remain to be evaluated extensively in figurative and culturally grounded reasoning, especially in low-resource contexts. We present BengaliFig, a compact yet richly annotated challenge set that targets this gap in Bengali, a widely spoken low-resourced language. The dataset contains 435 unique riddles drawn from Bengali oral and literary traditions. Each item is annotated along five orthogonal dimensions capturing reasoning type, trap type, cultural depth, answer category, and difficulty, and is automatically converted to multiple-choice format through a constraint-aware, AI-assisted pipeline. We evaluate eight frontier LLMs from major providers under zero-shot and few-shot chain-of-thought prompting, revealing consistent weaknesses in metaphorical and culturally specific reasoning. BengaliFig thus contributes both a diagnostic probe for evaluating LLM robustness in low-resource cultural contexts and a step toward inclusive and heritage-aware NLP evaluation. 

---
# MTA: A Merge-then-Adapt Framework for Personalized Large Language Model 

**Authors**: Xiaopeng Li, Yuanjin Zheng, Wanyu Wang, wenlin zhang, Pengyue Jia, Yiqi Wang, Maolin Wang, Xuetao Wei, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.20072)  

**Abstract**: Personalized Large Language Models (PLLMs) aim to align model outputs with individual user preferences, a crucial capability for user-centric applications. However, the prevalent approach of fine-tuning a separate module for each user faces two major limitations: (1) storage costs scale linearly with the number of users, rendering the method unscalable; and (2) fine-tuning a static model from scratch often yields suboptimal performance for users with sparse data. To address these challenges, we propose MTA, a Merge-then-Adapt framework for PLLMs. MTA comprises three key stages. First, we construct a shared Meta-LoRA Bank by selecting anchor users and pre-training meta-personalization traits within meta-LoRA modules. Second, to ensure scalability and enable dynamic personalization combination beyond static models, we introduce an Adaptive LoRA Fusion stage. This stage retrieves and dynamically merges the most relevant anchor meta-LoRAs to synthesize a user-specific one, thereby eliminating the need for user-specific storage and supporting more flexible personalization. Third, we propose a LoRA Stacking for Few-Shot Personalization stage, which applies an additional ultra-low-rank, lightweight LoRA module on top of the merged LoRA. Fine-tuning this module enables effective personalization under few-shot settings. Extensive experiments on the LaMP benchmark demonstrate that our approach outperforms existing SOTA methods across multiple tasks. 

---
# SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention Outputs in Feature Space 

**Authors**: Zhenyi Shen, Junru Lu, Lin Gui, Jiazheng Li, Yulan He, Di Yin, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.20102)  

**Abstract**: The quadratic complexity of full attention limits efficient long-context processing in large language models (LLMs). Sparse attention mitigates this cost by restricting each query to attend to a subset of previous tokens; however, training-free approaches often lead to severe performance degradation. Native sparse-attention methods (e.g., NSA, MoBA) alleviate this issue, yet exhibit a critical paradox: they produce lower attention sparsity than full-attention models, despite aiming to approximate full attention, which may constrain their effectiveness. We attribute this paradox to gradient update deficiency: low-ranked key-value pairs excluded during sparse training receive neither forward contribution nor backward gradients, and thus never learn proper suppression. To overcome this limitation, we propose SSA (Sparse Sparse Attention), a unified training framework that considers both sparse and full attention and enforces bidirectional alignment at every layer. This design preserves gradient flow to all tokens while explicitly encouraging sparse-attention outputs to align with their full-attention counterparts, thereby promoting stronger sparsity. As a result, SSA achieves state-of-the-art performance under both sparse and full attention inference across multiple commonsense benchmarks. Furthermore, SSA enables models to adapt smoothly to varying sparsity budgets; performance improves consistently as more tokens are allowed to attend, supporting flexible compute-performance trade-offs at inference time. Finally, we show that native sparse-attention training surprisingly improves long-context extrapolation by mitigating the over-allocation of attention values in sink areas, with SSA demonstrating the strongest extrapolation capability. 

---
# Latent Collaboration in Multi-Agent Systems 

**Authors**: Jiaru Zou, Xiyuan Yang, Ruizhong Qiu, Gaotang Li, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, Ling Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20639)  

**Abstract**: Multi-agent systems (MAS) extend large language models (LLMs) from independent single-model reasoning to coordinative system-level intelligence. While existing LLM agents depend on text-based mediation for reasoning and communication, we take a step forward by enabling models to collaborate directly within the continuous latent space. We introduce LatentMAS, an end-to-end training-free framework that enables pure latent collaboration among LLM agents. In LatentMAS, each agent first performs auto-regressive latent thoughts generation through last-layer hidden embeddings. A shared latent working memory then preserves and transfers each agent's internal representations, ensuring lossless information exchange. We provide theoretical analyses establishing that LatentMAS attains higher expressiveness and lossless information preservation with substantially lower complexity than vanilla text-based MAS. In addition, empirical evaluations across 9 comprehensive benchmarks spanning math and science reasoning, commonsense understanding, and code generation show that LatentMAS consistently outperforms strong single-model and text-based MAS baselines, achieving up to 14.6% higher accuracy, reducing output token usage by 70.8%-83.7%, and providing 4x-4.3x faster end-to-end inference. These results demonstrate that our new latent collaboration framework enhances system-level reasoning quality while offering substantial efficiency gains without any additional training. Code and data are fully open-sourced at this https URL. 

---
# Profile-LLM: Dynamic Profile Optimization for Realistic Personality Expression in LLMs 

**Authors**: Shi-Wei Dai, Yan-Wei Shie, Tsung-Huan Yang, Lun-Wei Ku, Yung-Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.19852)  

**Abstract**: Personalized Large Language Models (LLMs) have been shown to be an effective way to create more engaging and enjoyable user-AI interactions. While previous studies have explored using prompts to elicit specific personality traits in LLMs, they have not optimized these prompts to maximize personality expression. To address this limitation, we propose PersonaPulse: Dynamic Profile Optimization for Realistic Personality Expression in LLMs, a framework that leverages LLMs' inherent knowledge of personality traits to iteratively enhance role-play prompts while integrating a situational response benchmark as a scoring tool, ensuring a more realistic and contextually grounded evaluation to guide the optimization process. Quantitative evaluations demonstrate that the prompts generated by PersonaPulse outperform those of prior work, which were designed based on personality descriptions from psychological studies. Additionally, we explore the relationship between model size and personality modeling through extensive experiments. Finally, we find that, for certain personality traits, the extent of personality evocation can be partially controlled by pausing the optimization process. These findings underscore the importance of prompt optimization in shaping personality expression within LLMs, offering valuable insights for future research on adaptive AI interactions. 

---
# A Systematic Analysis of Large Language Models with RAG-enabled Dynamic Prompting for Medical Error Detection and Correction 

**Authors**: Farzad Ahmed, Joniel Augustine Jerome, Meliha Yetisgen, Özlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2511.19858)  

**Abstract**: Objective: Clinical documentation contains factual, diagnostic, and management errors that can compromise patient safety. Large language models (LLMs) may help detect and correct such errors, but their behavior under different prompting strategies remains unclear. We evaluate zero-shot prompting, static prompting with random exemplars (SPR), and retrieval-augmented dynamic prompting (RDP) for three subtasks of medical error processing: error flag detection, error sentence detection, and error correction.
Methods: Using the MEDEC dataset, we evaluated nine instruction-tuned LLMs (GPT, Claude, Gemini, and OpenAI o-series models). We measured performance using accuracy, recall, false-positive rate (FPR), and an aggregate score of ROUGE-1, BLEURT, and BERTScore for error correction. We also analyzed example outputs to identify failure modes and differences between LLM and clinician reasoning.
Results: Zero-shot prompting showed low recall in both detection tasks, often missing abbreviation-heavy or atypical errors. SPR improved recall but increased FPR. Across all nine LLMs, RDP reduced FPR by about 15 percent, improved recall by 5 to 10 percent in error sentence detection, and generated more contextually accurate corrections.
Conclusion: Across diverse LLMs, RDP outperforms zero-shot and SPR prompting. Using retrieved exemplars improves detection accuracy, reduces false positives, and enhances the reliability of medical error correction. 

---
# Gender Bias in Emotion Recognition by Large Language Models 

**Authors**: Maureen Herbert, Katie Sun, Angelica Lim, Yasaman Etesam  

**Link**: [PDF](https://arxiv.org/pdf/2511.19785)  

**Abstract**: The rapid advancement of large language models (LLMs) and their growing integration into daily life underscore the importance of evaluating and ensuring their fairness. In this work, we examine fairness within the domain of emotional theory of mind, investigating whether LLMs exhibit gender biases when presented with a description of a person and their environment and asked, "How does this person feel?". Furthermore, we propose and evaluate several debiasing strategies, demonstrating that achieving meaningful reductions in bias requires training based interventions rather than relying solely on inference-time prompt-based approaches such as prompt engineering. 

---
# Geometry of Decision Making in Language Models 

**Authors**: Abhinav Joshi, Divyanshu Bhatt, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.20315)  

**Abstract**: Large Language Models (LLMs) show strong generalization across diverse tasks, yet the internal decision-making processes behind their predictions remain opaque. In this work, we study the geometry of hidden representations in LLMs through the lens of \textit{intrinsic dimension} (ID), focusing specifically on decision-making dynamics in a multiple-choice question answering (MCQA) setting. We perform a large-scale study, with 28 open-weight transformer models and estimate ID across layers using multiple estimators, while also quantifying per-layer performance on MCQA tasks. Our findings reveal a consistent ID pattern across models: early layers operate on low-dimensional manifolds, middle layers expand this space, and later layers compress it again, converging to decision-relevant representations. Together, these results suggest LLMs implicitly learn to project linguistic inputs onto structured, low-dimensional manifolds aligned with task-specific decisions, providing new geometric insights into how generalization and reasoning emerge in language models. 

---
# Efficient Multi-Hop Question Answering over Knowledge Graphs via LLM Planning and Embedding-Guided Search 

**Authors**: Manil Shrestha, Edward Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.19648)  

**Abstract**: Multi-hop question answering over knowledge graphs remains computationally challenging due to the combinatorial explosion of possible reasoning paths. Recent approaches rely on expensive Large Language Model (LLM) inference for both entity linking and path ranking, limiting their practical deployment. Additionally, LLM-generated answers often lack verifiable grounding in structured knowledge. We present two complementary hybrid algorithms that address both efficiency and verifiability: (1) LLM-Guided Planning that uses a single LLM call to predict relation sequences executed via breadth-first search, achieving near-perfect accuracy (micro-F1 > 0.90) while ensuring all answers are grounded in the knowledge graph, and (2) Embedding-Guided Neural Search that eliminates LLM calls entirely by fusing text and graph embeddings through a lightweight 6.7M-parameter edge scorer, achieving over 100 times speedup with competitive accuracy. Through knowledge distillation, we compress planning capability into a 4B-parameter model that matches large-model performance at zero API cost. Evaluation on MetaQA demonstrates that grounded reasoning consistently outperforms ungrounded generation, with structured planning proving more transferable than direct answer generation. Our results show that verifiable multi-hop reasoning does not require massive models at inference time, but rather the right architectural inductive biases combining symbolic structure with learned representations. 

---
# QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation 

**Authors**: Xinguo Zhu, Shaohui Peng, Jiaming Guo, Yunji Chen, Qi Guo, Yuanbo Wen, Hang Qin, Ruizhi Chen, Qirui Zhou, Ke Gao, Yanjun Wu, Chen Zhao, Ling Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.20100)  

**Abstract**: Developing high-performance GPU kernels is critical for AI and scientific computing, but remains challenging due to its reliance on expert crafting and poor portability. While LLMs offer promise for automation, both general-purpose and finetuned LLMs suffer from two fundamental and conflicting limitations: correctness and efficiency. The key reason is that existing LLM-based approaches directly generate the entire optimized low-level programs, requiring exploration of an extremely vast space encompassing both optimization policies and implementation codes. To address the challenge of exploring an intractable space, we propose Macro Thinking Micro Coding (MTMC), a hierarchical framework inspired by the staged optimization strategy of human experts. It decouples optimization strategy from implementation details, ensuring efficiency through high-level strategy and correctness through low-level implementation. Specifically, Macro Thinking employs reinforcement learning to guide lightweight LLMs in efficiently exploring and learning semantic optimization strategies that maximize hardware utilization. Micro Coding leverages general-purpose LLMs to incrementally implement the stepwise optimization proposals from Macro Thinking, avoiding full-kernel generation errors. Together, they effectively navigate the vast optimization space and intricate implementation details, enabling LLMs for high-performance GPU kernel generation. Comprehensive results on widely adopted benchmarks demonstrate the superior performance of MTMC on GPU kernel generation in both accuracy and running time. On KernelBench, MTMC achieves near 100% and 70% accuracy at Levels 1-2 and 3, over 50% than SOTA general-purpose and domain-finetuned LLMs, with up to 7.3x speedup over LLMs, and 2.2x over expert-optimized PyTorch Eager kernels. On the more challenging TritonBench, MTMC attains up to 59.64% accuracy and 34x speedup. 

---
# Can LLMs Faithfully Explain Themselves in Low-Resource Languages? A Case Study on Emotion Detection in Persian 

**Authors**: Mobina Mehrazar, Mohammad Amin Yousefi, Parisa Abolfath Beygi, Behnam Bahrak  

**Link**: [PDF](https://arxiv.org/pdf/2511.19719)  

**Abstract**: Large language models (LLMs) are increasingly used to generate self-explanations alongside their predictions, a practice that raises concerns about the faithfulness of these explanations, especially in low-resource languages. This study evaluates the faithfulness of LLM-generated explanations in the context of emotion classification in Persian, a low-resource language, by comparing the influential words identified by the model against those identified by human annotators. We assess faithfulness using confidence scores derived from token-level log-probabilities. Two prompting strategies, differing in the order of explanation and prediction (Predict-then-Explain and Explain-then-Predict), are tested for their impact on explanation faithfulness. Our results reveal that while LLMs achieve strong classification performance, their generated explanations often diverge from faithful reasoning, showing greater agreement with each other than with human judgments. These results highlight the limitations of current explanation methods and metrics, emphasizing the need for more robust approaches to ensure LLM reliability in multilingual and low-resource contexts. 

---
# EfficientXpert: Efficient Domain Adaptation for Large Language Models via Propagation-Aware Pruning 

**Authors**: Songlin Zhao, Michael Pitts, Zhuwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2511.19935)  

**Abstract**: The rapid advancement of large language models (LLMs) has increased the demand for domain-specialized variants in areas such as law, healthcare, and finance. However, their large size remains a barrier to deployment in resource-constrained environments, and existing compression methods either generalize poorly across domains or incur high overhead. In this work, we propose \textbf{EfficientXpert}, a lightweight domain-pruning framework that combines a propagation-aware pruning criterion (Foresight Mask) with an efficient adapter-update algorithm (Partial Brain Surgeon). Integrated into the LoRA fine-tuning process, EfficientXpert enables a one-step transformation of general pretrained models into sparse, domain-adapted experts. Across health and legal tasks, it retains up to 98% of dense-model performance at 40% sparsity, outperforming state-of-the-art methods. Further analysis reveals substantial domain-dependent structural shifts that degrade the effectiveness of general pruning masks, underscoring the need for adaptive, domain-aware pruning strategies tailored to each domain. 

---
# Fara-7B: An Efficient Agentic Model for Computer Use 

**Authors**: Ahmed Awadallah, Yash Lara, Raghav Magazine, Hussein Mozannar, Akshay Nambi, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Vibhav Vineet, Spencer Whitehead, Andrew Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.19663)  

**Abstract**: Progress in computer use agents (CUAs) has been constrained by the absence of large and high-quality datasets that capture how humans interact with a computer. While LLMs have thrived on abundant textual data, no comparable corpus exists for CUA trajectories. To address these gaps, we introduce FaraGen, a novel synthetic data generation system for multi-step web tasks. FaraGen can propose diverse tasks from frequently used websites, generate multiple solution attempts, and filter successful trajectories using multiple verifiers. It achieves high throughput, yield, and diversity for multi-step web tasks, producing verified trajectories at approximately $1 each. We use this data to train Fara-7B, a native CUA model that perceives the computer using only screenshots, executes actions via predicted coordinates, and is small enough to run on-device. We find that Fara-7B outperforms other CUA models of comparable size on benchmarks like WebVoyager, Online-Mind2Web, and WebTailBench -- our novel benchmark that better captures under-represented web tasks in pre-existing benchmarks. Furthermore, Fara-7B is competitive with much larger frontier models, illustrating key benefits of scalable data generation systems in advancing small efficient agentic models. We are making Fara-7B open-weight on Microsoft Foundry and HuggingFace, and we are releasing WebTailBench. 

---
# Soft Adaptive Policy Optimization 

**Authors**: Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.20347)  

**Abstract**: Reinforcement learning (RL) plays an increasingly important role in enhancing the reasoning capabilities of large language models (LLMs), yet stable and performant policy optimization remains challenging. Token-level importance ratios often exhibit high variance-a phenomenon exacerbated in Mixture-of-Experts models-leading to unstable updates. Existing group-based policy optimization methods, such as GSPO and GRPO, alleviate this problem via hard clipping, making it difficult to maintain both stability and effective learning. We propose Soft Adaptive Policy Optimization (SAPO), which replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful learning signals. Compared with GSPO and GRPO, SAPO is both sequence-coherent and token-adaptive. Like GSPO, SAPO maintains sequence-level coherence, but its soft gating forms a continuous trust region that avoids the brittle hard clipping band used in GSPO. When a sequence contains a few highly off-policy tokens, GSPO suppresses all gradients for that sequence, whereas SAPO selectively down-weights only the offending tokens and preserves the learning signal from the near-on-policy ones, improving sample efficiency. Relative to GRPO, SAPO replaces hard token-level clipping with smooth, temperature-controlled scaling, enabling more informative and stable updates. Empirical results on mathematical reasoning benchmarks indicate that SAPO exhibits improved training stability and higher Pass@1 performance under comparable training budgets. Moreover, we employ SAPO to train the Qwen3-VL model series, demonstrating that SAPO yields consistent performance gains across diverse tasks and different model sizes. Overall, SAPO provides a more reliable, scalable, and effective optimization strategy for RL training of LLMs. 

---
# Fighting AI with AI: Leveraging Foundation Models for Assuring AI-Enabled Safety-Critical Systems 

**Authors**: Anastasia Mavridou, Divya Gopinath, Corina S. Păsăreanu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20627)  

**Abstract**: The integration of AI components, particularly Deep Neural Networks (DNNs), into safety-critical systems such as aerospace and autonomous vehicles presents fundamental challenges for assurance. The opacity of AI systems, combined with the semantic gap between high-level requirements and low-level network representations, creates barriers to traditional verification approaches. These AI-specific challenges are amplified by longstanding issues in Requirements Engineering, including ambiguity in natural language specifications and scalability bottlenecks in formalization. We propose an approach that leverages AI itself to address these challenges through two complementary components. REACT (Requirements Engineering with AI for Consistency and Testing) employs Large Language Models (LLMs) to bridge the gap between informal natural language requirements and formal specifications, enabling early verification and validation. SemaLens (Semantic Analysis of Visual Perception using large Multi-modal models) utilizes Vision Language Models (VLMs) to reason about, test, and monitor DNN-based perception systems using human-understandable concepts. Together, these components provide a comprehensive pipeline from informal requirements to validated implementations. 

---
# Universe of Thoughts: Enabling Creative Reasoning with Large Language Models 

**Authors**: Yuto Suzuki, Farnoush Banaei-Kashani  

**Link**: [PDF](https://arxiv.org/pdf/2511.20471)  

**Abstract**: Reasoning based on Large Language Models (LLMs) has garnered increasing attention due to outstanding performance of these models in mathematical and complex logical tasks. Beginning with the Chain-of-Thought (CoT) prompting technique, numerous reasoning methods have emerged that decompose problems into smaller, sequential steps (or thoughts). However, existing reasoning models focus on conventional problem-solving and do not necessarily generate creative solutions by ``creative reasoning''. In domains where the solution space is expansive and conventional solutions are suboptimal, such as drug discovery or business strategization, creative reasoning to discover innovative solutions is crucial. To address this gap, first we introduce a computational framework for creative reasoning inspired by established cognitive science principles. With this framework, we propose three core creative reasoning paradigms, namely, \textit{combinational}, \textit{exploratory}, and \textit{transformative} reasoning, where each offers specific directions for systematic exploration of the universe of thoughts to generate creative solutions. Next, to materialize this framework using LLMs, we introduce the \textit{Universe of Thoughts} (or \textit{UoT}, for short), a novel set of methods to implement the aforementioned three creative processes. Finally, we introduce three novel tasks that necessitate creative problem-solving, along with an evaluation benchmark to assess creativity from three orthogonal perspectives: feasibility as constraint, and utility and novelty as metrics. With a comparative analysis against the state-of-the-art (SOTA) reasoning techniques as well as representative commercial models with reasoning capability, we show that UoT demonstrates superior performance in creative reasoning. 

---
# Assessing LLMs' Performance: Insights from the Chinese Pharmacist Exam 

**Authors**: Xinran Wang, Boran Zhu, Shujuan Zhou, Ziwen Long, Dehua Zhou, Shu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20526)  

**Abstract**: Background: As large language models (LLMs) become increasingly integrated into digital health education and assessment workflows, their capabilities in supporting high-stakes, domain-specific certification tasks remain this http URL China, the national pharmacist licensure exam serves as a standardized benchmark for evaluating pharmacists' clinical and theoretical competencies. Objective: This study aimed to compare the performance of two LLMs: ChatGPT-4o and DeepSeek-R1 on real questions from the Chinese Pharmacist Licensing Examination (2017-2021), and to discuss the implications of these performance differences for AI-enabled formative evaluation. Methods: A total of 2,306 multiple-choice (text-only) questions were compiled from official exams, training materials, and public databases. Questions containing tables or images were excluded. Each item was input in its original Chinese format, and model responses were evaluated for exact accuracy. Pearson's Chi-squared test was used to compare overall performance, and Fisher's exact test was applied to year-wise multiple-choice accuracy. Results: DeepSeek-R1 outperformed ChatGPT-4o with a significantly higher overall accuracy (90.0% vs. 76.1%, p < 0.001). Unit-level analyses revealed consistent advantages for DeepSeek-R1, particularly in foundational and clinical synthesis modules. While year-by-year multiple-choice performance also favored DeepSeek-R1, this performance gap did not reach statistical significance in any specific unit-year (all p > 0.05). Conclusion: DeepSeek-R1 demonstrated robust alignment with the structural and semantic demands of the pharmacist licensure exam. These findings suggest that domain-specific models warrant further investigation for this context, while also reinforcing the necessity of human oversight in legally and ethically sensitive contexts. 

---
# NNGPT: Rethinking AutoML with Large Language Models 

**Authors**: Roman Kochnev, Waleed Khalid, Tolgay Atinc Uzun, Xi Zhang, Yashkumar Sanjaybhai Dhameliya, Furui Qin, Chandini Vysyaraju, Raghuvir Duvvuri, Avi Goyal, Dmitry Ignatov, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2511.20333)  

**Abstract**: Building self-improving AI systems remains a fundamental challenge in the AI domain. We present NNGPT, an open-source framework that turns a large language model (LLM) into a self-improving AutoML engine for neural network development, primarily for computer vision. Unlike previous frameworks, NNGPT extends the dataset of neural networks by generating new models, enabling continuous fine-tuning of LLMs based on closed-loop system of generation, assessment, and self-improvement. It integrates within one unified workflow five synergistic LLM-based pipelines: zero-shot architecture synthesis, hyperparameter optimization (HPO), code-aware accuracy/early-stop prediction, retrieval-augmented synthesis of scope-closed PyTorch blocks (NN-RAG), and reinforcement learning. Built on the LEMUR dataset as an audited corpus with reproducible metrics, NNGPT emits from a single prompt and validates network architecture, preprocessing code, and hyperparameters, executes them end-to-end, and learns from result. The PyTorch adapter makes NNGPT framework-agnostic, enabling strong performance: NN-RAG achieves 73% executability on 1,289 targets, 3-shot prompting boosts accuracy on common datasets, and hash-based deduplication saves hundreds of runs. One-shot prediction matches search-based AutoML, reducing the need for numerous trials. HPO on LEMUR achieves RMSE 0.60, outperforming Optuna (0.64), while the code-aware predictor reaches RMSE 0.14 with Pearson r=0.78. The system has already generated over 5K validated models, proving NNGPT as an autonomous AutoML engine. Upon acceptance, the code, prompts, and checkpoints will be released for public access to enable reproducibility and facilitate community usage. 

---
# SMoG: Schema Matching on Graph 

**Authors**: Mingyu Jeon, Jaeyoung Suh, Suwan Cho  

**Link**: [PDF](https://arxiv.org/pdf/2511.20285)  

**Abstract**: Schema matching is a critical task in data integration, par- ticularly in the medical domain where disparate Electronic Health Record (EHR) systems must be aligned to standard models like OMOP CDM. While Large Language Models (LLMs) have shown promise in schema matching, they suf- fer from hallucination and lack of up-to-date domain knowl- edge. Knowledge Graphs (KGs) offer a solution by pro- viding structured, verifiable knowledge. However, existing KG-augmented LLM approaches often rely on inefficient complex multi-hop queries or storage-intensive vector-based retrieval methods. This paper introduces SMoG (Schema Matching on Graph), a novel framework that leverages iter- ative execution of simple 1-hop SPARQL queries, inspired by successful strategies in Knowledge Graph Question An- swering (KGQA). SMoG enhances explainability and relia- bility by generating human-verifiable query paths while sig- nificantly reducing storage requirements by directly querying SPARQL endpoints. Experimental results on real-world med- ical datasets demonstrate that SMoG achieves performance comparable to state-of-the-art baselines, validating its effec- tiveness and efficiency in KG-augmented schema matching. 

---
# Improving Language Agents through BREW 

**Authors**: Shashank Kirtania, Param Biyani, Priyanshu Gupta, Yasharth Bajpai, Roshni Iyer, Sumit Gulwani, Gustavo Soares  

**Link**: [PDF](https://arxiv.org/pdf/2511.20297)  

**Abstract**: Large Language Model (LLM)-based agents are increasingly applied to tasks requiring structured reasoning, tool use, and environmental adaptation, such as data manipulation, multistep planning, and computer-use automation. However, despite their versatility, current training paradigms for model weight optimization methods, like PPO and GRPO, remain relatively impractical with their high computational overhead for rollout convergence. In addition, the resulting agent policies are difficult to interpret, adapt, or incrementally improve. To address this, we investigate creating and refining structured memory of experiential learning of an agent from its environment as an alternative route to agent optimization. We introduce BREW (Bootstrapping expeRientially-learned Environmental knoWledge), a framework for agent optimization for downstream tasks via KB construction and refinement. In our formulation, we introduce an effective method for partitioning agent memory for more efficient retrieval and refinement. BREW uses task graders and behavior rubrics to learn insights while leveraging state-space search for ensuring robustness from the noise and non-specificity in natural language. Empirical results on real world, domain-grounded benchmarks -- OSWorld, $\tau^2$Bench, and SpreadsheetBench -- show BREW achieves $10-20\%$ improvement in task precision, $10-15\%$ reduction in API/tool calls leading to faster execution time, all while maintaining computational efficiency on par with base models. Unlike prior work where memory is treated as static context, we establish the KB as a modular and controllable substrate for agent optimization -- an explicit lever for shaping behavior in a transparent, interpretable, and extensible manner. 

---
# DRAFT-RL: Multi-Agent Chain-of-Draft Reasoning for Reinforcement Learning-Enhanced LLMs 

**Authors**: Yuanhao Li, Mingshan Liu, Hongbo Wang, Yiding Zhang, Yifei Ma, Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20468)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities in multi-step reasoning and this http URL works introduce multi-agent reflection frameworks where multiple LLM agents critique and refine each other's outputs using reinforcement learning (RL). However, these approaches often rely on single-shot responses and lack structural diversity in reasoning exploration. In this paper, we propose DRAFT-RL, a novel framework that integrates Chain-of-Draft (CoD) reasoning into multi-agent RL training. Instead of generating single responses, each agent produces multiple drafts per query, which are then evaluated by peer agents and a learned reward model to identify the most promising trajectory. These selected drafts are used to refine future reasoning strategies through actor-critic this http URL-RL enables explicit multi-path exploration, peer-guided reflection, and reward-aligned selection, resulting in more robust and interpretable LLM agent behavior. We evaluate our method on complex reasoning tasks including code synthesis, symbolic math, and knowledge-intensive QA,demonstrating that DRAFT-RL outperforms existing reflective and RL-based agents by significant margins in both accuracy and convergence speed 

---
# Simulated Self-Assessment in Large Language Models: A Psychometric Approach to AI Self-Efficacy 

**Authors**: Daniel I Jackson, Emma L Jensen, Syed-Amad Hussain, Emre Sezgin  

**Link**: [PDF](https://arxiv.org/pdf/2511.19872)  

**Abstract**: Self-assessment is a key aspect of reliable intelligence, yet evaluations of large language models (LLMs) focus mainly on task accuracy. We adapted the 10-item General Self-Efficacy Scale (GSES) to elicit simulated self-assessments from ten LLMs across four conditions: no task, computational reasoning, social reasoning, and summarization. GSES responses were highly stable across repeated administrations and randomized item orders. However, models showed significantly different self-efficacy levels across conditions, with aggregate scores lower than human norms. All models achieved perfect accuracy on computational and social questions, whereas summarization performance varied widely. Self-assessment did not reliably reflect ability: several low-scoring models performed accurately, while some high-scoring models produced weaker summaries. Follow-up confidence prompts yielded modest, mostly downward revisions, suggesting mild overestimation in first-pass assessments. Qualitative analysis showed that higher self-efficacy corresponded to more assertive, anthropomorphic reasoning styles, whereas lower scores reflected cautious, de-anthropomorphized explanations. Psychometric prompting provides structured insight into LLM communication behavior but not calibrated performance estimates. 

---
# A System-Level Taxonomy of Failure Modes in Large Language Model Applications 

**Authors**: Vaishali Vinay  

**Link**: [PDF](https://arxiv.org/pdf/2511.19933)  

**Abstract**: Large language models (LLMs) are being rapidly integrated into decision-support tools, automation workflows, and AI-enabled software systems. However, their behavior in production environments remains poorly understood, and their failure patterns differ fundamentally from those of traditional machine learning models. This paper presents a system-level taxonomy of fifteen hidden failure modes that arise in real-world LLM applications, including multi-step reasoning drift, latent inconsistency, context-boundary degradation, incorrect tool invocation, version drift, and cost-driven performance collapse. Using this taxonomy, we analyze the growing gap in evaluation and monitoring practices: existing benchmarks measure knowledge or reasoning but provide little insight into stability, reproducibility, drift, or workflow integration. We further examine the production challenges associated with deploying LLMs - including observability limitations, cost constraints, and update-induced regressions - and outline high-level design principles for building reliable, maintainable, and cost-aware LLM systems. Finally, we outline high-level design principles for building reliable, maintainable, and cost-aware LLM-based systems. By framing LLM reliability as a system-engineering problem rather than a purely model-centric one, this work provides an analytical foundation for future research on evaluation methodology, AI system robustness, and dependable LLM deployment. 

---
# FISCAL: Financial Synthetic Claim-document Augmented Learning for Efficient Fact-Checking 

**Authors**: Rishab Sharma, Iman Saberi, Elham Alipour, Jie JW Wu, Fatemeh Fard  

**Link**: [PDF](https://arxiv.org/pdf/2511.19671)  

**Abstract**: Financial applications of large language models (LLMs) require factual reliability and computational efficiency, yet current systems often hallucinate details and depend on prohibitively large models. We propose FISCAL (Financial Synthetic Claim-Document Augmented Learning), a modular framework for generating synthetic data tailored to financial fact-checking. Using FISCAL, we generate a dataset called FISCAL-data and use it to train MiniCheck-FISCAL, a lightweight verifier for numerical financial claims. MiniCheck-FISCAL outperforms its baseline, surpasses GPT-3.5 Turbo and other open-source peers of similar size, and approaches the accuracy of much larger systems (20x), such as Mixtral-8x22B and Command R+. On external datasets FinDVer and Fin-Fact, it rivals GPT-4o and Claude-3.5 while outperforming Gemini-1.5 Flash. These results show that domain-specific synthetic data, combined with efficient fine-tuning, enables compact models to achieve state-of-the-art accuracy, robustness, and scalability for practical financial AI. The dataset and scripts are available in the project repository (link provided in the paper). 

---
# Scaling Item-to-Standard Alignment with Large Language Models: Accuracy, Limits, and Solutions 

**Authors**: Farzan Karimi-Malekabadi, Pooya Razavi, Sonya Powers  

**Link**: [PDF](https://arxiv.org/pdf/2511.19749)  

**Abstract**: As educational systems evolve, ensuring that assessment items remain aligned with content standards is essential for maintaining fairness and instructional relevance. Traditional human alignment reviews are accurate but slow and labor-intensive, especially across large item banks. This study examines whether Large Language Models (LLMs) can accelerate this process without sacrificing accuracy. Using over 12,000 item-skill pairs in grades K-5, we tested three LLMs (GPT-3.5 Turbo, GPT-4o-mini, and GPT-4o) across three tasks that mirror real-world challenges: identifying misaligned items, selecting the correct skill from the full set of standards, and narrowing candidate lists prior to classification. In Study 1, GPT-4o-mini correctly identified alignment status in approximately 83-94% of cases, including subtle misalignments. In Study 2, performance remained strong in mathematics but was lower for reading, where standards are more semantically overlapping. Study 3 demonstrated that pre-filtering candidate skills substantially improved results, with the correct skill appearing among the top five suggestions more than 95% of the time. These findings suggest that LLMs, particularly when paired with candidate filtering strategies, can significantly reduce the manual burden of item review while preserving alignment accuracy. We recommend the development of hybrid pipelines that combine LLM-based screening with human review in ambiguous cases, offering a scalable solution for ongoing item validation and instructional alignment. 

---
# Using Wearable Devices to Improve Chronic PainTreatment among Patients with Opioid Use Disorder 

**Authors**: Abhay Goyal, Navin Kumar, Kimberly DiMeola, Rafael Trujillo, Soorya Ram Shimgekar, Christian Poellabauer, Pi Zonooz, Ermonda Gjoni-Markaj, Declan Barry, Lynn Madden  

**Link**: [PDF](https://arxiv.org/pdf/2511.19577)  

**Abstract**: Chronic pain (CP) and opioid use disorder (OUD) are common and interrelated chronic medical conditions. Currently, there is a paucity of evidence-based integrated treatments for CP and OUD among individuals receiving medication for opioid use disorder (MOUD). Wearable devices have the potential to monitor complex patient information and inform treatment development for persons with OUD and CP, including pain variability (e.g., exacerbations of pain or pain spikes) and clinical correlates (e.g., perceived stress). However, the application of large language models (LLMs) with wearable data for understanding pain spikes, remains unexplored. Consequently, the aim of this pilot study was to examine the clinical correlates of pain spikes using a range of AI approaches. We found that machine learning models achieved relatively high accuracy (>0.7) in predicting pain spikes, while LLMs were limited in providing insights on pain spikes. Real-time monitoring through wearable devices, combined with advanced AI models, could facilitate early detection of pain spikes and support personalized interventions that may help mitigate the risk of opioid relapse, improve adherence to MOUD, and enhance the integration of CP and OUD care. Given overall limited LLM performance, these findings highlight the need to develop LLMs which can provide actionable insights in the OUD/CP context. 

---
# ROOT: Robust Orthogonalized Optimizer for Neural Network Training 

**Authors**: Wei He, Kai Han, Hang Zhou, Hanting Chen, Zhicheng Liu, Xinghao Chen, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20626)  

**Abstract**: The optimization of large language models (LLMs) remains a critical challenge, particularly as model scaling exacerbates sensitivity to algorithmic imprecision and training instability. Recent advances in optimizers have improved convergence efficiency through momentum orthogonalization, but suffer from two key robustness limitations: dimensional fragility in orthogonalization precision and vulnerability to outlier-induced noise. To address these robustness challenges, we introduce ROOT, a Robust Orthogonalized Optimizer that enhances training stability through dual robustness mechanisms. First, we develop a dimension-robust orthogonalization scheme using adaptive Newton iterations with fine-grained coefficients tailored to specific matrix sizes, ensuring consistent precision across diverse architectural configurations. Second, we introduce an optimization-robust framework via proximal optimization that suppresses outlier noise while preserving meaningful gradient directions. Extensive experiments demonstrate that ROOT achieves significantly improved robustness, with faster convergence and superior final performance compared to both Muon and Adam-based optimizers, particularly in noisy and non-convex scenarios. Our work establishes a new paradigm for developing robust and precise optimizers capable of handling the complexities of modern large-scale model training. The code will be available at this https URL. 

---
# Can Vibe Coding Beat Graduate CS Students? An LLM vs. Human Coding Tournament on Market-driven Strategic Planning 

**Authors**: Panayiotis Danassis, Naman Goel  

**Link**: [PDF](https://arxiv.org/pdf/2511.20613)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has revolutionized AI-assisted code generation. This rapid development of LLMs has outpaced our ability to properly benchmark them. Prevailing benchmarks emphasize unit-test pass rates and syntactic correctness. Such metrics understate the difficulty of many real-world problems that require planning, optimization, and strategic interaction. We introduce a multi-agent reasoning-driven benchmark based on a real-world logistics optimization problem (Auction, Pickup, and Delivery Problem) that couples competitive auctions with capacity-constrained routing. The benchmark requires building agents that can (i) bid strategically under uncertainty and (ii) optimize planners that deliver tasks while maximizing profit. We evaluate 40 LLM-coded agents (by a wide range of state-of-the-art LLMs under multiple prompting methodologies, including vibe coding) against 17 human-coded agents developed before the advent of LLMs. Our results over 12 double all-play-all tournaments and $\sim 40$k matches demonstrate (i) a clear superiority of human(graduate students)-coded agents: the top 5 spots are consistently won by human-coded agents, (ii) the majority of LLM-coded agents (33 out of 40) are beaten by very simple baselines, and (iii) given the best human solution as an input and prompted to improve upon, the best performing LLM makes the solution significantly worse instead of improving it. Our results highlight a gap in LLMs' ability to produce code that works competitively in the real-world, and motivate new evaluations that emphasize reasoning-driven code synthesis in real-world scenarios. 

---
# DiFR: Inference Verification Despite Nondeterminism 

**Authors**: Adam Karvonen, Daniel Reuter, Roy Rinberg, Luke Marks, Adrià Garriga-Alonso, Keri Warr  

**Link**: [PDF](https://arxiv.org/pdf/2511.20621)  

**Abstract**: As demand for LLM inference grows, it is becoming increasingly important that providers and their customers can verify that inference processes are performed correctly, without errors or tampering. However, re-running the same inference process twice often leads to different results due to benign numerical noise, making it difficult to distinguish legitimate variation from actual problems. To address this problem, we introduce Token-DiFR (Token-Divergence-From-Reference), a method for verifying inference outputs by comparing generated tokens against predictions made by a trusted reference implementation conditioned on the same random seed. Sampling seed synchronization tightly constrains valid outputs, leaving providers minimal room to deviate from correct inference, which allows output tokens themselves to serve as auditable evidence of correctness at zero additional cost to the provider. Token-DiFR reliably identifies sampling errors, simulated bugs, and model quantization, detecting 4-bit quantization with AUC $>$ 0.999 within 300 output tokens. For applications requiring sample-efficient forward-pass verification, we additionally introduce Activation-DiFR, a scheme that uses random orthogonal projections to compress activations into compact fingerprints for subsequent verification. Activation-DiFR detects 4-bit quantization with AUC $>$ 0.999 using just 2 output tokens, while reducing communication overhead by 25-75% relative to existing methods. We release an open-source integration with vLLM to accelerate practical deployment of verifiable inference. 

---
# MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology 

**Authors**: Kiril Vasilev, Alexandre Misrahi, Eeshaan Jain, Phil F Cheng, Petros Liakopoulos, Olivier Michielin, Michael Moor, Charlotte Bunne  

**Link**: [PDF](https://arxiv.org/pdf/2511.20490)  

**Abstract**: Multimodal Large Language Models (LLMs) hold promise for biomedical reasoning, but current benchmarks fail to capture the complexity of real-world clinical workflows. Existing evaluations primarily assess unimodal, decontextualized question-answering, overlooking multi-agent decision-making environments such as Molecular Tumor Boards (MTBs). MTBs bring together diverse experts in oncology, where diagnostic and prognostic tasks require integrating heterogeneous data and evolving insights over time. Current benchmarks lack this longitudinal and multimodal complexity. We introduce MTBBench, an agentic benchmark simulating MTB-style decision-making through clinically challenging, multimodal, and longitudinal oncology questions. Ground truth annotations are validated by clinicians via a co-developed app, ensuring clinical relevance. We benchmark multiple open and closed-source LLMs and show that, even at scale, they lack reliability -- frequently hallucinating, struggling with reasoning from time-resolved data, and failing to reconcile conflicting evidence or different modalities. To address these limitations, MTBBench goes beyond benchmarking by providing an agentic framework with foundation model-based tools that enhance multi-modal and longitudinal reasoning, leading to task-level performance gains of up to 9.0% and 11.2%, respectively. Overall, MTBBench offers a challenging and realistic testbed for advancing multimodal LLM reasoning, reliability, and tool-use with a focus on MTB environments in precision oncology. 

---
# LLMs for Automated Unit Test Generation and Assessment in Java: The AgoneTest Framework 

**Authors**: Andrea Lops, Fedelucio Narducci, Azzurra Ragone, Michelantonio Trizio, Claudio Barto  

**Link**: [PDF](https://arxiv.org/pdf/2511.20403)  

**Abstract**: Unit testing is an essential but resource-intensive step in software development, ensuring individual code units function correctly. This paper introduces AgoneTest, an automated evaluation framework for Large Language Model-generated (LLM) unit tests in Java. AgoneTest does not aim to propose a novel test generation algorithm; rather, it supports researchers and developers in comparing different LLMs and prompting strategies through a standardized end-to-end evaluation pipeline under realistic conditions. We introduce the Classes2Test dataset, which maps Java classes under test to their corresponding test classes, and a framework that integrates advanced evaluation metrics, such as mutation score and test smells, for a comprehensive assessment. Experimental results show that, for the subset of tests that compile, LLM-generated tests can match or exceed human-written tests in terms of coverage and defect detection. Our findings also demonstrate that enhanced prompting strategies contribute to test quality. AgoneTest clarifies the potential of LLMs in software testing and offers insights for future improvements in model design, prompt engineering, and testing practices. 

---
# Can LLMs Make (Personalized) Access Control Decisions? 

**Authors**: Friederike Groschupp, Daniele Lain, Aritra Dhar, Lara Magdalena Lazier, Srdjan Čapkun  

**Link**: [PDF](https://arxiv.org/pdf/2511.20284)  

**Abstract**: Precise access control decisions are crucial to the security of both traditional applications and emerging agent-based systems. Typically, these decisions are made by users during app installation or at runtime. Due to the increasing complexity and automation of systems, making these access control decisions can add a significant cognitive load on users, often overloading them and leading to suboptimal or even arbitrary access control decisions. To address this problem, we propose to leverage the processing and reasoning capabilities of large language models (LLMs) to make dynamic, context-aware decisions aligned with the user's security preferences. For this purpose, we conducted a user study, which resulted in a dataset of 307 natural-language privacy statements and 14,682 access control decisions made by users. We then compare these decisions against those made by two versions of LLMs: a general and a personalized one, for which we also gathered user feedback on 1,446 of its decisions.
Our results show that in general, LLMs can reflect users' preferences well, achieving up to 86\% accuracy when compared to the decision made by the majority of users. Our study also reveals a crucial trade-off in personalizing such a system: while providing user-specific privacy preferences to the LLM generally improves agreement with individual user decisions, adhering to those preferences can also violate some security best practices. Based on our findings, we discuss design and risk considerations for implementing a practical natural-language-based access control system that balances personalization, security, and utility. 

---
# R3A: Reliable RTL Repair Framework with Multi-Agent Fault Localization and Stochastic Tree-of-Thoughts Patch Generation 

**Authors**: Zizhang Luo, Fan Cui, Kexing Zhou, Runlin Guo, Mile Xia, Hongyuan Hou, Yun Lian  

**Link**: [PDF](https://arxiv.org/pdf/2511.20090)  

**Abstract**: Repairing RTL bugs is crucial for hardware design and verification. Traditional automatic program repair (APR) methods define dedicated search spaces to locate and fix bugs with program synthesis. However, they heavily rely on fixed templates and can only deal with limited bugs. As an alternative, Large Language Models with the ability to understand code semantics can be explored for RTL repair. However, they suffer from unreliable outcomes due to inherent randomness and long input contexts of RTL code and waveform. To address these challenges, we propose R3A, an LLM-based automatic RTL program repair framework upon the basic model to improve reliability. R3A proposes the stochastic Tree-Of-Thoughts method to control a patch generation agent to explore a validated solution for the bug. The algorithm samples search states according to a heuristic function to balance between exploration and exploitation for a reliable outcome. Besides, R3A proposes a multi-agent fault localization method to find fault candidates as the starting points for the patch generation agent, further increasing the reliability. Experiments show R3A can fix 90.6% of bugs in the RTL-repair dataset within a given time limit, which covers 45% more bugs than traditional methods and other LLM-based approaches, while achieving an 86.7% pass@5 rate on average, showing a high reliability. 

---
# On the Feasibility of Hijacking MLLMs' Decision Chain via One Perturbation 

**Authors**: Changyue Li, Jiaying Li, Youliang Yuan, Jiaming He, Zhicong Huang, Pinjia He  

**Link**: [PDF](https://arxiv.org/pdf/2511.20002)  

**Abstract**: Conventional adversarial attacks focus on manipulating a single decision of neural networks. However, real-world models often operate in a sequence of decisions, where an isolated mistake can be easily corrected, but cascading errors can lead to severe risks.
This paper reveals a novel threat: a single perturbation can hijack the whole decision chain. We demonstrate the feasibility of manipulating a model's outputs toward multiple, predefined outcomes, such as simultaneously misclassifying "non-motorized lane" signs as "motorized lane" and "pedestrian" as "plastic bag".
To expose this threat, we introduce Semantic-Aware Universal Perturbations (SAUPs), which induce varied outcomes based on the semantics of the inputs. We overcome optimization challenges by developing an effective algorithm, which searches for perturbations in normalized space with a semantic separation strategy. To evaluate the practical threat of SAUPs, we present RIST, a new real-world image dataset with fine-grained semantic annotations. Extensive experiments on three multimodal large language models demonstrate their vulnerability, achieving a 70% attack success rate when controlling five distinct targets using just an adversarial frame. 

---
# LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training 

**Authors**: Ziwei Liu, Qidong Liu, Wanyu Wang, Yejing Wang, Tong Xu, Wei Huang, Chong Chen, Peng Chuan, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.19931)  

**Abstract**: Cross-domain Sequential Recommendation (CDSR) has been proposed to enrich user-item interactions by incorporating information from various domains. Despite current progress, the imbalance issue and transition issue hinder further development of CDSR. The former one presents a phenomenon that the interactions in one domain dominate the entire behavior, leading to difficulty in capturing the domain-specific features in the other domain. The latter points to the difficulty in capturing users' cross-domain preferences within the mixed interaction sequence, resulting in poor next-item prediction performance for specific domains. With world knowledge and powerful reasoning ability, Large Language Models (LLMs) partially alleviate the above issues by performing as a generator and an encoder. However, current LLMs-enhanced CDSR methods are still under exploration, which fail to recognize the irrelevant noise and rough profiling problems. Thus, to make peace with the aforementioned challenges, we proposed an LLMs Enhanced Cross-domain Sequential Recommendation with Dual-phase Training ({LLM-EDT}). To address the imbalance issue while introducing less irrelevant noise, we first propose the transferable item augmenter to adaptively generate possible cross-domain behaviors for users. Then, to alleviate the transition issue, we introduce a dual-phase training strategy to empower the domain-specific thread with a domain-shared background. As for the rough profiling problem, we devise a domain-aware profiling module to summarize the user's preference in each domain and adaptively aggregate them to generate comprehensive user profiles. The experiments on three public datasets validate the effectiveness of our proposed LLM-EDT. To ease reproducibility, we have released the detailed code online at {this https URL}. 

---
# CodeFuse-CommitEval: Towards Benchmarking LLM's Power on Commit Message and Code Change Inconsistency Detection 

**Authors**: Qingyu Zhang, Puzhuo Liu, Peng Di, Chenxiong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2511.19875)  

**Abstract**: Version control relies on commit messages to convey the rationale for code changes, but these messages are often low quality and, more critically, inconsistent with their diffs-known as message-code inconsistency (MCI). MCIs mislead reviewers, hinder maintenance, contaminate research datasets, and may obscure security patches. Yet, no dedicated benchmark exists to evaluate models for MCI detection. We introduce CODEFUSE-COMMITEVAL, the first benchmark designed for MCI detection using large language models (LLMs). Built on the ApacheCM dataset for diversity and quality, we generate seven types of inconsistent messages through rule-guided mutations of originally consistent commits and apply two-fold validation to verify both positive and negative samples. Using this labeled dataset of message-diff pairs, we evaluate six state-of-the-art open-source LLMs under a vanilla setting and with three augmentation strategies: few-shot prompting, chain-of-thought, and extended context. Results show models detect inconsistent commits more reliably than consistent ones (average Recall 85.95%, Precision 80.28%, Specificity 63.8%); gpt-oss-20B performs best overall but uses over twice the tokens of others. Augmentation effects vary: adjacent context helps larger models but adds noise for smaller ones; few-shot improves accuracy and reduces token use, yet increases universally incorrect predictions; chain-of-thought boosts precision and specificity at the cost of recall and higher token consumption. Type-wise analysis reveals higher detectability for component, file-path, and operation inconsistencies, but lower accuracy and higher token cost for intent-level "purpose" inconsistencies. CODEFUSE-COMMITEVAL provides a rigorous foundation for measuring, comparing, and advancing MCI detection, highlighting the need for richer context and balanced data to capture high-level semantic gaps. 

---
# Beyond Relational: Semantic-Aware Multi-Modal Analytics with LLM-Native Query Optimization 

**Authors**: Junhao Zhu, Lu Chen, Xiangyu Ke, Ziquan Fang, Tianyi Li, Yunjun Gao, Christian S. Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2511.19830)  

**Abstract**: Multi-modal analytical processing has the potential to transform applications in e-commerce, healthcare, entertainment, and beyond. However, real-world adoption remains elusive due to the limited ability of traditional relational query operators to capture query semantics. The emergence of foundation models, particularly the large language models (LLMs), opens up new opportunities to develop flexible, semantic-aware data analytics systems that transcend the relational paradigm.
We present Nirvana, a multi-modal data analytics framework that incorporates programmable semantic operators while leveraging both logical and physical query optimization strategies, tailored for LLM-driven semantic query processing. Nirvana addresses two key challenges. First, it features an agentic logical optimizer that uses natural language-specified transformation rules and random-walk-based search to explore vast spaces of semantically equivalent query plans -- far beyond the capabilities of conventional optimizers. Second, it introduces a cost-aware physical optimizer that selects the most effective LLM backend for each operator using a novel improvement-score metric. To further enhance efficiency, Nirvana incorporates computation reuse and evaluation pushdown techniques guided by model capability hypotheses. Experimental evaluations on three real-world benchmarks demonstrate that Nirvana is able to reduce end-to-end runtime by 10%--85% and reduces system processing costs by 76% on average, outperforming state-of-the-art systems at both efficiency and scalability. 

---
# Prompt Fencing: A Cryptographic Approach to Establishing Security Boundaries in Large Language Model Prompts 

**Authors**: Steven Peh  

**Link**: [PDF](https://arxiv.org/pdf/2511.19727)  

**Abstract**: Large Language Models (LLMs) remain vulnerable to prompt injection attacks, representing the most significant security threat in production deployments. We present Prompt Fencing, a novel architectural approach that applies cryptographic authentication and data architecture principles to establish explicit security boundaries within LLM prompts. Our approach decorates prompt segments with cryptographically signed metadata including trust ratings and content types, enabling LLMs to distinguish between trusted instructions and untrusted content. While current LLMs lack native fence awareness, we demonstrate that simulated awareness through prompt instructions achieved complete prevention of injection attacks in our experiments, reducing success rates from 86.7% (260/300 successful attacks) to 0% (0/300 successful attacks) across 300 test cases with two leading LLM providers. We implement a proof-of-concept fence generation and verification pipeline with a total overhead of 0.224 seconds (0.130s for fence generation, 0.094s for validation) across 100 samples. Our approach is platform-agnostic and can be incrementally deployed as a security layer above existing LLM infrastructure, with the expectation that future models will be trained with native fence awareness for optimal security. 

---
# Accuracy and Efficiency Trade-Offs in LLM-Based Malware Detection and Explanation: A Comparative Study of Parameter Tuning vs. Full Fine-Tuning 

**Authors**: Stephen C. Gravereaux, Sheikh Rabiul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2511.19654)  

**Abstract**: This study examines whether Low-Rank Adaptation (LoRA) fine-tuned Large Language Models (LLMs) can approximate the performance of fully fine-tuned models in generating human-interpretable decisions and explanations for malware classification. Achieving trustworthy malware detection, particularly when LLMs are involved, remains a significant challenge. We developed an evaluation framework using Bilingual Evaluation Understudy (BLEU), Recall-Oriented Understudy for Gisting Evaluation (ROUGE), and Semantic Similarity Metrics to benchmark explanation quality across five LoRA configurations and a fully fine-tuned baseline. Results indicate that full fine-tuning achieves the highest overall scores, with BLEU and ROUGE improvements of up to 10% over LoRA variants. However, mid-range LoRA models deliver competitive performance exceeding full fine-tuning on two metrics while reducing model size by approximately 81% and training time by over 80% on a LoRA model with 15.5% trainable parameters. These findings demonstrate that LoRA offers a practical balance of interpretability and resource efficiency, enabling deployment in resource-constrained environments without sacrificing explanation quality. By providing feature-driven natural language explanations for malware classifications, this approach enhances transparency, analyst confidence, and operational scalability in malware detection systems. 

---
# The Semiotic Channel Principle: Measuring the Capacity for Meaning in LLM Communication 

**Authors**: Davide Picca  

**Link**: [PDF](https://arxiv.org/pdf/2511.19550)  

**Abstract**: This paper proposes a novel semiotic framework for analyzing Large Language Models (LLMs), conceptualizing them as stochastic semiotic engines whose outputs demand active, asymmetric human interpretation. We formalize the trade-off between expressive richness (semiotic breadth) and interpretive stability (decipherability) using information-theoretic tools. Breadth is quantified as source entropy, and decipherability as the mutual information between messages and human interpretations. We introduce a generative complexity parameter (lambda) that governs this trade-off, as both breadth and decipherability are functions of lambda. The core trade-off is modeled as an emergent property of their distinct responses to $\lambda$. We define a semiotic channel, parameterized by audience and context, and posit a capacity constraint on meaning transmission, operationally defined as the maximum decipherability by optimizing lambda. This reframing shifts analysis from opaque model internals to observable textual artifacts, enabling empirical measurement of breadth and decipherability. We demonstrate the framework's utility across four key applications: (i) model profiling; (ii) optimizing prompt/context design; (iii) risk analysis based on ambiguity; and (iv) adaptive semiotic systems. We conclude that this capacity-based semiotic approach offers a rigorous, actionable toolkit for understanding, evaluating, and designing LLM-mediated communication. 

---
# AttackPilot: Autonomous Inference Attacks Against ML Services With LLM-Based Agents 

**Authors**: Yixin Wu, Rui Wen, Chi Cui, Michael Backes, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19536)  

**Abstract**: Inference attacks have been widely studied and offer a systematic risk assessment of ML services; however, their implementation and the attack parameters for optimal estimation are challenging for non-experts. The emergence of advanced large language models presents a promising yet largely unexplored opportunity to develop autonomous agents as inference attack experts, helping address this challenge. In this paper, we propose AttackPilot, an autonomous agent capable of independently conducting inference attacks without human intervention. We evaluate it on 20 target services. The evaluation shows that our agent, using GPT-4o, achieves a 100.0% task completion rate and near-expert attack performance, with an average token cost of only $0.627 per run. The agent can also be powered by many other representative LLMs and can adaptively optimize its strategy under service constraints. We further perform trace analysis, demonstrating that design choices, such as a multi-agent framework and task-specific action spaces, effectively mitigate errors such as bad plans, inability to follow instructions, task context loss, and hallucinations. We anticipate that such agents could empower non-expert ML service providers, auditors, or regulators to systematically assess the risks of ML services without requiring deep domain expertise. 

---
# Cross-Domain Generalization of Multimodal LLMs for Global Photovoltaic Assessment 

**Authors**: Muhao Guo, Yang Weng  

**Link**: [PDF](https://arxiv.org/pdf/2511.19537)  

**Abstract**: The rapid expansion of distributed photovoltaic (PV) systems poses challenges for power grid management, as many installations remain undocumented. While satellite imagery provides global coverage, traditional computer vision (CV) models such as CNNs and U-Nets require extensive labeled data and fail to generalize across regions. This study investigates the cross-domain generalization of a multimodal large language model (LLM) for global PV assessment. By leveraging structured prompts and fine-tuning, the model integrates detection, localization, and quantification within a unified schema. Cross-regional evaluation using the $\Delta$F1 metric demonstrates that the proposed model achieves the smallest performance degradation across unseen regions, outperforming conventional CV and transformer baselines. These results highlight the robustness of multimodal LLMs under domain shift and their potential for scalable, transferable, and interpretable global PV mapping. 

---
# A Systematic Study of Compression Ordering for Large Language Models 

**Authors**: Shivansh Chhawri, Rahul Mahadik, Suparna Rooj  

**Link**: [PDF](https://arxiv.org/pdf/2511.19495)  

**Abstract**: Large Language Models (LLMs) require substantial computational resources, making model compression essential for efficient deployment in constrained environments. Among the dominant compression techniques: knowledge distillation, structured pruning, and low-bit quantization, their individual effects are well studied, but their interactions and optimal sequencing remain unclear. This work systematically examines how these techniques perform both independently and in combination when applied to the Qwen2.5 3B model. We evaluate multiple compression pipelines, including single, and proposed three-technique sequences, using perplexity, G-Eval, clarity, prompt alignment, and compression ratio as metrics. Our experiments show that quantization provides the greatest standalone compression, while pruning introduces moderate quality degradation. Critically, the ordering of techniques significantly affects the final model quality: the sequence Pruning, Knowledge Distillation, Quantization (P-KD-Q) yields the best balance, achieving a 3.68x compression ratio while preserving strong instruction-following and language understanding capabilities. Conversely, pipelines applying quantization early suffer severe performance degradation due to irreversible information loss that impairs subsequent training. Overall, this study offers practical insight into designing effective, ordering-aware compression pipelines for deploying LLMs in resource-limited settings. 

---
# Evolution without an Oracle: Driving Effective Evolution with LLM Judges 

**Authors**: Zhe Zhao, Yuheng Yang, Haibin Wen, Xiaojie Qiu, Zaixi Zhang, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19489)  

**Abstract**: The integration of Large Language Models (LLMs) with Evolutionary Computation (EC) has unlocked new frontiers in scientific discovery but remains shackled by a fundamental constraint: the reliance on an Oracle--an objective, machine-computable fitness function. This paper breaks this barrier by asking: Can evolution thrive in a purely subjective landscape governed solely by LLM judges? We introduce MADE (Multi-Agent Decomposed Evolution), a framework that tames the inherent noise of subjective evaluation through "Problem Specification." By decomposing vague instructions into specific, verifiable sub-requirements, MADE transforms high-variance LLM feedback into stable, precise selection pressure. The results are transformative: across complex benchmarks like DevAI and InfoBench, MADE outperforms strong baselines by over 50% in software requirement satisfaction (39.9% to 61.9%) and achieves a 95% perfect pass rate on complex instruction following. This work validates a fundamental paradigm shift: moving from optimizing "computable metrics" to "describable qualities," thereby unlocking evolutionary optimization for the vast open-ended domains where no ground truth exists. 

---
# Efficient Inference Using Large Language Models with Limited Human Data: Fine-Tuning then Rectification 

**Authors**: Lei Wang, Zikun Ye, Jinglong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.19486)  

**Abstract**: Driven by recent advances in artificial intelligence (AI), a growing body of work demonstrates the potential of using large language models (LLMs) to generate human-like responses in market research and social science applications. Two primary approaches can be applied to improve the performance of LLMs: fine-tuning, which aligns LLM predictions more closely with human responses, and rectification, which corrects biases in LLM outputs. In this paper, we develop a framework that combines fine-tuning and rectification, and optimally allocates limited labeled samples across the two stages. Unlike the conventional objective that minimizes the mean squared prediction errors, we propose to minimize the variance of the prediction errors as the fine-tuning objective, which is optimal for the downstream rectification stage. Building on this insight, we leverage empirical scaling laws to develop a data-driven method for optimally splitting samples between the fine-tuning and rectification stages. Empirical analysis validates our framework, demonstrating improved estimation and inference performance compared to using either fine-tuning or rectification alone. 

---
# Building Resilient Information Ecosystems: Large LLM-Generated Dataset of Persuasion Attacks 

**Authors**: Hsien-Te Kao, Aleksey Panasyuk, Peter Bautista, William Dupree, Gabriel Ganberg, Jeffrey M. Beaubien, Laura Cassani, Svitlana Volkova  

**Link**: [PDF](https://arxiv.org/pdf/2511.19488)  

**Abstract**: Organization's communication is essential for public trust, but the rise of generative AI models has introduced significant challenges by generating persuasive content that can form competing narratives with official messages from government and commercial organizations at speed and scale. This has left agencies in a reactive position, often unaware of how these models construct their persuasive strategies, making it more difficult to sustain communication effectiveness. In this paper, we introduce a large LLM-generated persuasion attack dataset, which includes 134,136 attacks generated by GPT-4, Gemma 2, and Llama 3.1 on agency news. These attacks span 23 persuasive techniques from SemEval 2023 Task 3, directed toward 972 press releases from ten agencies. The generated attacks come in two mediums, press release statements and social media posts, covering both long-form and short-form communication strategies. We analyzed the moral resonance of these persuasion attacks to understand their attack vectors. GPT-4's attacks mainly focus on Care, with Authority and Loyalty also playing a role. Gemma 2 emphasizes Care and Authority, while Llama 3.1 centers on Loyalty and Care. Analyzing LLM-generated persuasive attacks across models will enable proactive defense, allow to create the reputation armor for organizations, and propel the development of both effective and resilient communications in the information ecosystem. 

---
# Exploiting the Experts: Unauthorized Compression in MoE-LLMs 

**Authors**: Pinaki Prasad Guha Neogi, Ahmad Mohammadshirazi, Dheeraj Kulshrestha, Rajiv Ramnath  

**Link**: [PDF](https://arxiv.org/pdf/2511.19480)  

**Abstract**: Mixture-of-Experts (MoE) architectures are increasingly adopted in large language models (LLMs) for their scalability and efficiency. However, their modular structure introduces a unique vulnerability: adversaries can attempt to compress or repurpose models by pruning experts and cheaply fine-tuning the remainder, effectively bypassing licensing and security constraints. In this paper, we systematically study the prunability of MoE-LLMs under task-specific usage. We first develop an expert attribution framework that identifies the subset of experts most responsible for a given task, then evaluate the performance trade-offs of pruning and re-aligning these experts using active learning-driven fine-tuning. Our findings reveal a critical knowledge loss--recovery trade-off: while certain experts can be isolated to retain task accuracy, significant degradation occurs without targeted re-alignment. Based on this analysis, we propose defense strategies that aim to make MoE models harder to compress and fine-tune without authorization, including entangled expert training and selective fine-tuning protocols that resist unauthorized adaptation. By positioning expert pruning as both a threat vector and a defense target, this work highlights the dual-use nature of MoE modularity and provides the first systematic evaluation framework for secure specialization of MoE-LLMs. 

---
# Enhancing Sequential Recommendation with World Knowledge from Large Language Models 

**Authors**: Tianjie Dai, Xu Chen, Yunmeng Shu, Jinsong Lan, Xiaoyong Zhu, Jiangchao Yao, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.20177)  

**Abstract**: Sequential Recommendation System~(SRS) has become pivotal in modern society, which predicts subsequent actions based on the user's historical behavior. However, traditional collaborative filtering-based sequential recommendation models often lead to suboptimal performance due to the limited information of their collaborative signals. With the rapid development of LLMs, an increasing number of works have incorporated LLMs' world knowledge into sequential recommendation. Although they achieve considerable gains, these approaches typically assume the correctness of LLM-generated results and remain susceptible to noise induced by LLM hallucinations. To overcome these limitations, we propose GRASP (Generation Augmented Retrieval with Holistic Attention for Sequential Prediction), a flexible framework that integrates generation augmented retrieval for descriptive synthesis and similarity retrieval, and holistic attention enhancement which employs multi-level attention to effectively employ LLM's world knowledge even with hallucinations and better capture users' dynamic interests. The retrieved similar users/items serve as auxiliary contextual information for the later holistic attention enhancement module, effectively mitigating the noisy guidance of supervision-based methods. Comprehensive evaluations on two public benchmarks and one industrial dataset reveal that GRASP consistently achieves state-of-the-art performance when integrated with diverse backbones. The code is available at: this https URL. 

---
# SCoTER: Structured Chain-of-Thought Transfer for Enhanced Recommendation 

**Authors**: Yang Wu, Qian Li, Yuling Xiong, Hongbo Tang, Xun Liu, Jun Zhang, Huan Yu, Jie Jiang, Hailong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.19514)  

**Abstract**: Harnessing the reasoning power of Large Language Models (LLMs) for recommender systems is hindered by two fundamental challenges. First, current approaches lack a mechanism for automated, data-driven discovery of effective reasoning patterns, relying instead on brittle manual templates or unstable zero-shot prompting. Second, they employ structure-collapsing integration: direct prompting incurs prohibitive online inference costs, while feature extraction collapses reasoning chains into single vectors, discarding stepwise logic. To address these challenges, we propose SCoTER (Structured Chain-of-Thought Transfer for Enhanced Recommendation), a unified framework that treats pattern discovery and structure-aware transfer as a jointly optimized problem. Specifically, SCoTER operationalizes this through two synergistic components: a GVM pipeline for automated pattern discovery and a structure-preserving integration architecture that transfers stepwise logic to efficient models. Formally, we provide information-theoretic justification proving that structure-preserving transfer achieves tighter performance bounds than structure-agnostic alternatives. Empirically, experiments on four benchmarks demonstrate improvements of 3.75\%-11.59\% over a strong TIGER backbone. Moreover, in production deployment on the Tencent Advertising Platform, SCoTER achieved a 2.14\% lift in Gross Merchandise Value (GMV) while eliminating online LLM inference costs. Overall, SCoTER establishes a principled and production-validated blueprint for transferring structured LLM reasoning to large-scale recommender systems. 

---
