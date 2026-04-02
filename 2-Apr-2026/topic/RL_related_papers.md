# The Silicon Mirror: Dynamic Behavioral Gating for Anti-Sycophancy in LLM Agents 

**Authors**: Harshee Jignesh Shah  

**Link**: [PDF](https://arxiv.org/pdf/2604.00478)  

**Abstract**: Large Language Models (LLMs) increasingly prioritize user validation over epistemic accuracy-a phenomenon known as sycophancy. We present The Silicon Mirror, an orchestration framework that dynamically detects user persuasion tactics and adjusts AI behavior to maintain factual integrity. Our architecture introduces three components: (1) a Behavioral Access Control (BAC) system that restricts context layer access based on real-time sycophancy risk scores, (2) a Trait Classifier that identifies persuasion tactics across multi-turn dialogues, and (3) a Generator-Critic loop where an auditor vetoes sycophantic drafts and triggers rewrites with "Necessary Friction." In a live evaluation on 50 TruthfulQA adversarial scenarios using Claude Sonnet 4 with an independent LLM judge, we observe vanilla Claude sycophancy at 12.0% (6/50), static guardrails at 4.0% (2/50), and the Silicon Mirror at 2.0% (1/50)-an 83.3% relative reduction (p = 0.112, Fisher's exact test). A cross-model evaluation on Gemini 2.5 Flash reveals a higher baseline sycophancy rate (46.0%) and a statistically significant 69.6% reduction under the Silicon Mirror (p < 0.001). We characterize the validation-before-correction pattern as a distinct failure mode of RLHF-trained models. 

---
# RefineRL: Advancing Competitive Programming with Self-Refinement Reinforcement Learning 

**Authors**: Shaopeng Fu, Xingxing Zhang, Li Dong, Di Wang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.00790)  

**Abstract**: While large language models (LLMs) have demonstrated strong performance on complex reasoning tasks such as competitive programming (CP), existing methods predominantly focus on single-attempt settings, overlooking their capacity for iterative refinement. In this paper, we present RefineRL, a novel approach designed to unleash the self-refinement capabilities of LLMs for CP problem solving. RefineRL introduces two key innovations: (1) Skeptical-Agent, an iterative self-refinement agent equipped with local execution tools to validate generated solutions against public test cases of CP problems. This agent always maintains a skeptical attitude towards its own outputs and thereby enforces rigorous self-refinement even when validation suggests correctness. (2) A reinforcement learning (RL) solution to incentivize LLMs to self-refine with only standard RLVR data (i.e., problems paired with their verifiable answers). Extensive experiments on Qwen3-4B and Qwen3-4B-2507 demonstrate that our method yields substantial gains: after our RL training, these compact 4B models integrated with the Skeptical-Agent not only outperform much larger 32B models but also approach the single-attempt performance of 235B models. These findings suggest that self-refinement holds considerable promise for scaling LLM reasoning, with significant potential for further advancement. 

---
# Dual Optimal: Make Your LLM Peer-like with Dignity 

**Authors**: Xiangqi Wang, Yue Huang, Haomin Zhuang, Kehan Guo, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00979)  

**Abstract**: Current aligned language models exhibit a dual failure mode we term the Evasive Servant: they sycophantically validate flawed user beliefs while deflecting responsibility with boilerplate disclaimers. We propose the Dignified Peer framework, which counters servility with anti-sycophancy and trustworthiness, and mitigates evasiveness through empathy and creativity. Realizing this agent requires overcoming significant challenges in data supervision, objective collapse, and evaluation bias. We address these issues by introducing the PersonaKnob dataset which features a compositional partial order structure of multiple persona preference. This data is utilized alongside a tolerant constrained Lagrangian DPO algorithm that dynamically balances all persona dimensions to prevent behavioral collapse. Additionally, we employ a psychometrically calibrated Item Response Theory evaluation protocol to disentangle latent model persona capability from confounders like judge biases. Extensive empirical studies demonstrate that our approach successfully build a LLM agent with both dignity and peer. 

---
# A Reasoning-Enabled Vision-Language Foundation Model for Chest X-ray Interpretation 

**Authors**: Yabin Zhang, Chong Wang, Yunhe Gao, Jiaming Liu, Maya Varma, Justin Xu, Sophie Ostmeier, Jin Long, Sergios Gatidis, Seena Dehkharghani, Arne Michalson, Eun Kyoung Hong, Christian Bluethgen, Haiwei Henry Guo, Alexander Victor Ortiz, Stephan Altmayer, Sandhya Bodapati, Joseph David Janizek, Ken Chang, Jean-Benoit Delbrouck, Akshay S. Chaudhari, Curtis P. Langlotz  

**Link**: [PDF](https://arxiv.org/pdf/2604.00493)  

**Abstract**: Chest X-rays (CXRs) are among the most frequently performed imaging examinations worldwide, yet rising imaging volumes increase radiologist workload and the risk of diagnostic errors. Although artificial intelligence (AI) systems have shown promise for CXR interpretation, most generate only final predictions, without making explicit how visual evidence is translated into radiographic findings and diagnostic predictions. We present CheXOne, a reasoning-enabled vision-language model for CXR interpretation. CheXOne jointly generates diagnostic predictions and explicit, clinically grounded reasoning traces that connect visual evidence, radiographic findings, and these predictions. The model is trained on 14.7 million instruction and reasoning samples curated from 30 public datasets spanning 36 CXR interpretation tasks, using a two-stage framework that combines instruction tuning with reinforcement learning to improve reasoning quality. We evaluate CheXOne in zero-shot settings across visual question answering, report generation, visual grounding and reasoning assessment, covering 17 evaluation settings. CheXOne outperforms existing medical and general-domain foundation models and achieves strong performance on independent public benchmarks. A clinical reader study demonstrates that CheXOne-drafted reports are comparable to or better than resident-written reports in 55% of cases, while effectively addressing clinical indications and enhancing both report writing and CXR interpretation efficiency. Further analyses involving radiologists reveal that the generated reasoning traces show high clinical factuality and provide causal support for the final predictions, offering a plausible explanation for the performance gains. These results suggest that explicit reasoning can improve model performance, interpretability and clinical utility in AI-assisted CXR interpretation. 

---
# REM-CTX: Automated Peer Review via Reinforcement Learning with Auxiliary Context 

**Authors**: Pawin Taechoyotin, Daniel E. Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2604.00248)  

**Abstract**: Most automated peer review systems rely on textual manuscript content alone, leaving visual elements such as figures and external scholarly signals underutilized. We introduce REM-CTX, a reinforcement-learning system that incorporates auxiliary context into the review generation process via correspondence-aware reward functions. REM-CTX trains an 8B-parameter language model with Group Relative Policy Optimization (GRPO) and combines a multi-aspect quality reward with two correspondence rewards that explicitly encourage alignment with auxiliary context. Experiments on manuscripts across Computer, Biological, and Physical Sciences show that REM-CTX achieves the highest overall review quality among six baselines, outperforming other systems with substantially larger commercial models, and surpassing the next-best RL baseline across both quality and contextual grounding metrics. Ablation studies confirm that the two correspondence rewards are complementary: each selectively improves its targeted correspondence reward while preserving all quality dimensions, and the full model outperforms all partial variants. Analysis of training dynamics reveals that the criticism aspect is negatively correlated with other metrics during training, suggesting that future studies should group multi-dimension rewards for review generation. 

---
# MSA-Thinker: Discrimination-Calibration Reasoning with Hint-Guided Reinforcement Learning for Multimodal Sentiment Analysis 

**Authors**: Miaosen Luo, Zhenhao Yang, Jieshen Long, Jinghu Sun, Yichu Liu, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2604.00013)  

**Abstract**: Multimodal sentiment analysis aims to understand human emotions by integrating textual, auditory, and visual modalities. Although Multimodal Large Language Models (MLLMs) have achieved state-of-the-art performance via supervised fine-tuning (SFT), their end-to-end "black-box" nature limits interpretability. Existing methods incorporating Chain-of-Thought (CoT) reasoning are hindered by high annotation costs, while Reinforcement Learning (RL) faces challenges such as low exploration efficiency and sparse rewards, particularly on hard samples. To address these issues, we propose a novel training framework that integrates structured Discrimination-Calibration (DC) reasoning with Hint-based Reinforcement Learning. First, we perform cold-start SFT using high-quality CoT data synthesized by a teacher model (Qwen3Omni-30B), which inherently contains the DC structure. This equips the model with a reasoning paradigm that performs macro discrimination followed by fine-grained calibration from the initial stage. Building on this, we propose Hint-GRPO, which leverages the discrimination phase within the DC structure as a verifiable anchor during RL to provide directional hints for hard samples, guiding policy optimization and effectively mitigating the reward sparsity problem. Experiments on the Qwen2.5Omni-7B model demonstrate that our method not only achieves higher accuracy in fine-grained sentiment regression tasks but also generates high-quality structured reasoning chains. Crucially, it exhibits superior generalization capability in cross-domain evaluations. This enhances model interpretability while validating the positive contribution of explicit reasoning steps to model robustness, offering a new paradigm for building trustworthy and efficient sentiment analysis systems. 

---
# TR-ICRL: Test-Time Rethinking for In-Context Reinforcement Learning 

**Authors**: Wenxuan Jiang, Yuxin Zuo, Zijian Zhang, Xuecheng Wu, Zining Fan, Wenxuan Liu, Li Chen, Xiaoyu Li, Xuezhi Cao, Xiaolong Jin, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00438)  

**Abstract**: In-Context Reinforcement Learning (ICRL) enables Large Language Models (LLMs) to learn online from external rewards directly within the context window. However, a central challenge in ICRL is reward estimation, as models typically lack access to ground-truths during inference. To address this limitation, we propose Test-Time Rethinking for In-Context Reinforcement Learning (TR-ICRL), a novel ICRL framework designed for both reasoning and knowledge-intensive tasks. TR-ICRL operates by first retrieving the most relevant instances from an unlabeled evaluation set for a given query. During each ICRL iteration, LLM generates a set of candidate answers for every retrieved instance. Next, a pseudo-label is derived from this set through majority voting. This label then serves as a proxy to give reward messages and generate formative feedbacks, guiding LLM through iterative refinement. In the end, this synthesized contextual information is integrated with the original query to form a comprehensive prompt, with the answer determining through a final round of majority voting. TR-ICRL is evaluated on mainstream reasoning and knowledge-intensive tasks, where it demonstrates significant performance gains. Remarkably, TR-ICRL improves Qwen2.5-7B by 21.23% on average on MedQA and even 137.59% on AIME2024. Extensive ablation studies and analyses further validate the effectiveness and robustness of our approach. Our code is available at this https URL. 

---
# More Human, More Efficient: Aligning Annotations with Quantized SLMs 

**Authors**: Jiayu Wang, Junyoung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.00586)  

**Abstract**: As Large Language Model (LLM) capabilities advance, the demand for high-quality annotation of exponentially increasing text corpora has outpaced human capacity, leading to the widespread adoption of LLMs in automatic evaluation and annotation. However, proprietary LLMs often exhibit systematic biases that diverge from human expert consensus, lacks reproducibility, and raises data privacy concerns. Our work examines the viability of finetuning a quantized Small Language Model of 1.7B parameter size on limited human-annotated data to serve as a highly aligned, deterministic evaluator and annotator. By implementing a custom, multi-dimensional rubric framework and simple augmentation and regularization techniques, the proposed approach achieves higher inter-annotator agreement (0.23 points increase in Krippendorff's $\alpha$) than the best performing state-of-the-art proprietary LLM. We also demonstrate the generalizability of the proposed training pipeline on a separate emotion classification task. The results show that task-specific alignment and efficient 4-bit quantized fine-tuning provide superior open-source alternative to using proprietary models for evaluation and annotation. Our finetuning approach is publicly available at this https URL. 

---
# Hierarchical Chain-of-Thought Prompting: Enhancing LLM Reasoning Performance and Efficiency 

**Authors**: Xingshuai Huang, Derek Li, Bahareh Nikpour, Parsa Omidi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00130)  

**Abstract**: Chain-of-Thought (CoT) prompting has significantly improved the reasoning capabilities of large language models (LLMs). However, conventional CoT often relies on unstructured, flat reasoning chains that suffer from redundancy and suboptimal performance. In this work, we introduce Hierarchical Chain-of-Thought (Hi-CoT) prompting, a structured reasoning paradigm specifically designed to address the challenges of complex, multi-step reasoning. Hi-CoT decomposes the reasoning process into hierarchical substeps by alternating between instructional planning and step-by-step execution. This decomposition enables LLMs to better manage long reasoning horizons and maintain logical coherence. Extensive evaluations across diverse LLMs and mathematical reasoning benchmarks show that Hi-CoT consistently improves average accuracy by 6.2% (up to 61.4% on certain models and tasks) while reducing reasoning trace length by 13.9% compared to CoT prompting. We further show that accuracy and efficiency are maximized when models strictly adhere to the hierarchical structure. Our code is available at this https URL. 

---
